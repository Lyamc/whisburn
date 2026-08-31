use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
    config::Config,
    record::{FullPrecisionSettings, HalfPrecisionSettings, NamedMpkFileRecorder, NamedMpkGzFileRecorder, Recorder},
};
use crate::model::*;
use crate::model::conformer::*;
use crate::model::load::helpers::*;
use crate::model::load::whisper::*;
use crate::model::load::parakeet::*;
use crate::model::load::qwen3::load_qwen3_model;
use crate::model::load::vibevoice::load_vibevoice_model;
use crate::token::Gpt2Tokenizer;
use std::error::Error;
use std::path::Path;
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

pub fn load_model_from_npy<B: Backend>(
    path: &str,
    device: &B::Device,
) -> Result<(Model<B>, ModelConfig), Box<dyn Error>> {
    if Path::new(&format!("{}/tone.encoder", path)).exists() || path.contains("t-one") {
        return Err(
            "T-one uses safetensors, not npy. Load via `load_model(\"t-one\", ...)` after download."
                .into(),
        );
    } else if Path::new(&format!("{}/ctc_linear", path)).exists() {
        let parakeet = load_parakeet(path, device)?;
        
        // Detect actual params from the loaded model
        let n_layers = parakeet.encoder.layers.len();
        let d_model = parakeet.encoder.layers[0].final_ln.gamma.dims()[0];
        let n_head = parakeet.encoder.layers[0].attn.n_head;
        let kernel_size = parakeet.encoder.layers[0].conv.depth_conv.weight.dims()[2];
        
        let n_mels = if Path::new(&format!("{}/encoder/n_mels.npy", path)).exists() {
            load_usize::<B>("n_mels", &format!("{}/encoder", path), device)?
        } else {
            128 // Default for Parakeet-TDT v3
        };

        // Burn Linear weight is [d_in, d_out]; joint head output is d_out.
        let n_vocab = parakeet.ctc_linear.weight.dims()[1];

        let config = ModelConfig::Parakeet(ParakeetConfig {
            encoder: ConformerEncoderConfig {
                d_model,
                n_head,
                n_layers,
                kernel_size,
            },
            n_vocab,
            n_mels,
        });
        
        Ok((Model::Parakeet(parakeet), config))
    } else if Path::new(&format!("{}/generic_marker.npy", path)).exists() || path.contains("moonshine") {
        // Placeholder for new architectures like Moonshine, Voxtral, etc.
        if path.contains("moonshine") {
             return Err("Moonshine architecture detected but full Burn implementation is pending.".into());
        }
        return Err("Model loaded successfully but architecture is not yet implemented in Burn.".into());
    } else {
        let (whisper, config) = load_whisper(path, device)?;
        Ok((Model::Whisper(whisper), ModelConfig::Whisper(config)))
    }
}

pub fn load_model<B: Backend>(
    model_name: &str,
    device: &B::Device,
    verbose: bool,
) -> Result<(Gpt2Tokenizer, ModelConfig, Model<B>), Box<dyn Error>> {
    if verbose {
        println!("Initializing model loading for: {}", model_name);
    }

    if model_name == "vibevoice-asr" || model_name == "bitnet-asr" {
        return load_vibevoice_model(model_name, device, verbose);
    }

    if model_name.starts_with("moonshine") {
        return crate::model::load::moonshine::load_moonshine_model(model_name, device, verbose);
    }

    if model_name == "t-one" || model_name.starts_with("t-one") {
        return crate::model::load::tone::load_tone_model(model_name, device, verbose);
    }

    if model_name.starts_with("qwen3-asr-") {
        return load_qwen3_model(model_name, device, verbose);
    }

    let pb = ProgressBar::new_spinner();
    pb.set_style(ProgressStyle::default_spinner()
        .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
        .template("{spinner:.green} {msg}").unwrap());
    pb.set_message(format!("Loading model '{}'...", model_name));
    pb.enable_steady_tick(Duration::from_millis(100));

    let bpe = match Gpt2Tokenizer::new(model_name) {
        Ok(bpe) => bpe,
        Err(e) => {
            pb.finish_and_clear();
            return Err(format!("Failed to load tokenizer for {}: {}", model_name, e).into());
        }
    };

    let config_path1 = format!("models/{}/{}.cfg", model_name, model_name);
    let config_path2 = format!("models/{}/config.cfg", model_name);
    let config_path = if Path::new(&config_path1).exists() {
        config_path1
    } else if Path::new(&config_path2).exists() {
        config_path2
    } else {
        // Search for ANY .cfg in the directory
        let dir = format!("models/{}", model_name);
        let mut found = String::new();
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                if let Some(ext) = entry.path().extension() {
                    if ext == "cfg" {
                        found = entry.path().to_str().unwrap().to_string();
                        break;
                    }
                }
            }
        }
        found
    };

    if config_path.is_empty() {
        pb.finish_and_clear();
        return Err(format!("Config file not found in models/{}", model_name).into());
    }

    if verbose {
        pb.println(format!("Using config file: {}", config_path));
    }

    let model_config = match ModelConfig::load(&config_path) {
        Ok(config) => config,
        Err(_) => {
            match WhisperConfig::load(&config_path) {
                Ok(w) => ModelConfig::Whisper(w),
                Err(e) => {
                    pb.finish_and_clear();
                    return Err(format!("Failed to load model config at {}: {}", config_path, e).into());
                }
            }
        }
    };

    let model_path = format!("models/{}/{}", model_name, model_name);
    let mpk_path = format!("{}.mpk", model_path);
    let gz_path = format!("{}.mpk.gz", model_path);

    let model_mpk_path = format!("models/{}/model.mpk", model_name);
    let model_gz_path = format!("models/{}/model.mpk.gz", model_name);

    let (final_path, final_gz_path) = if Path::new(&mpk_path).exists() || Path::new(&gz_path).exists() {
        (mpk_path, gz_path)
    } else {
        (model_mpk_path, model_gz_path)
    };

    if !Path::new(&final_path).exists() && Path::new(&final_gz_path).exists() {
        pb.set_message(format!("Decompressing {}...", final_gz_path));
        let gz_file = std::fs::File::open(&final_gz_path)?;
        let mut decoder = GzDecoder::new(gz_file);
        let mut mpk_file = std::fs::File::create(&final_path)?;
        std::io::copy(&mut decoder, &mut mpk_file)?;
    }

    if !Path::new(&final_path).exists() {
        pb.finish_and_clear();
        return Err(format!("Model weights not found: {}", final_path).into());
    }

    let load_path = final_path.strip_suffix(".mpk").unwrap_or(&final_path);

    pb.set_message(format!("Loading model weights from {}...", load_path));
    let model: Model<B> = {
        // Prefer full precision — HF-converted bundles are recorded with FullPrecisionSettings.
        let mut result = NamedMpkFileRecorder::<FullPrecisionSettings>::new()
            .load(load_path.to_string().into(), device);

        if result.is_err() {
            result = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new()
                .load(load_path.to_string().into(), device);
        }

        if result.is_err() {
            if verbose {
                pb.println("Failed FullPrecision, trying HalfPrecisionSettings...");
            }
            result = NamedMpkFileRecorder::<HalfPrecisionSettings>::new()
                .load(load_path.to_string().into(), device);

            if result.is_err() {
                result = NamedMpkGzFileRecorder::<HalfPrecisionSettings>::new()
                    .load(load_path.to_string().into(), device);
            }
        }

        match result.map(|record| model_config.init(device).load_record(record)) {
            Ok(m) => m,
            Err(e) => {
                pb.finish_and_clear();
                return Err(format!("Failed to initialize model: {}", e).into());
            }
        }
    };

    pb.set_message("Moving model to device...");
    let model = model.to_device(device);

    pb.finish_with_message(format!("Model '{}' loaded successfully.", model_name));

    Ok((bpe, model_config, model))
}

pub fn get_model_stats<B: Backend>(model_name: &str, device: &B::Device) -> (Option<Tensor<B, 2>>, Option<Tensor<B, 2>>) {
    let path = format!("models/{}", model_name);
    let mean = if Path::new(&format!("{}/mean.npy", path)).exists() {
        load_tensor::<B, 2>("mean", &path, device).ok()
    } else {
        None
    };
    let std = if Path::new(&format!("{}/std.npy", path)).exists() {
        load_tensor::<B, 2>("std", &path, device).ok()
    } else {
        None
    };
    (mean, std)
}
