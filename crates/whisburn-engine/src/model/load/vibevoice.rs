use burn::module::Module;
use burn::tensor::backend::Backend;
use std::error::Error;
use std::path::Path;

use crate::model::*;
use crate::model::vibevoice::quant::LinearQuant;
use crate::model::vibevoice::weights::{
    detect_encoder_layout, load_connectors, load_decoder, load_encoder, VibeVoiceWeightStore,
};
use crate::model::vibevoice::{build_vibevoice_config, load_vibevoice_runtime, VibeVoiceASR};
use crate::token::Gpt2Tokenizer;

pub fn load_vibevoice_model<B: Backend>(
    model_name: &str,
    device: &B::Device,
    verbose: bool,
) -> Result<(Gpt2Tokenizer, ModelConfig, Model<B>), Box<dyn Error>> {
    let model_dir = format!("models/{model_name}");
    let runtime_path = format!("{model_dir}/vibevoice_runtime.json");
    if !std::path::Path::new(&runtime_path).exists() {
        return Err(format!(
            "VibeVoice runtime config missing at {runtime_path}. Run: \
             cargo run -p whisburn-cli -- models download {model_name} --verbose"
        )
        .into());
    }

    let runtime = load_vibevoice_runtime(model_name);
    let model_config = ModelConfig::VibeVoice(
        build_vibevoice_config(model_name, &runtime)
            .map_err(|e| format!("VibeVoice config: {e}"))?,
    );
    let bpe = Gpt2Tokenizer::new(model_name)
        .map_err(|e| format!("Failed to load tokenizer for {model_name}: {e}"))?;
    if verbose {
        if model_name == "bitnet-asr" {
            println!(
                "VibeVoice-BitNet: initializing Qwen2.5-1.5B ternary decoder (Burn I2_S, not ggml/GGUF)"
            );
        } else {
            println!(
                "VibeVoice: initializing Qwen2.5-7B INT8 decoder (GGUF not used — Burn INT8, ~4× smaller than f32)"
            );
        }
    }
    let mut model = model_config.init(device).to_device(device);

    if let Model::VibeVoice(ref mut vv) = model {
        load_vibevoice_weights(vv, &model_dir, model_name, device, verbose)?;
    }

    Ok((bpe, model_config, model))
}

fn load_vibevoice_weights<B: Backend>(
    model: &mut VibeVoiceASR<B>,
    model_dir: &str,
    model_name: &str,
    device: &B::Device,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    let index_path = Path::new(model_dir).join("model.safetensors.index.json");
    if !index_path.exists() {
        if verbose {
            println!("VibeVoice: no safetensors index — using uninitialized weights");
        }
        return Ok(());
    }

    let mut store = VibeVoiceWeightStore::open(Path::new(model_dir))
        .map_err(|e| format!("VibeVoice weight store: {e}"))?;

    if store.has_key("multi_modal_projector.acoustic_linear_1.weight")
        || store.has_key("model.acoustic_connector.fc1.weight")
        || store.has_key("acoustic_connector.fc1.weight")
    {
        let (acoustic, semantic) = load_connectors(&mut store, model, device)?;
        model.acoustic_connector = acoustic;
        model.semantic_connector = semantic;
        if verbose {
            println!("VibeVoice: loaded speech connectors");
        }
    }

    if let Some((prefix, layout)) = detect_encoder_layout(&store, "acoustic") {
        model.acoustic_encoder =
            load_encoder(&mut store, &prefix, layout, &model.acoustic_encoder, device)?;
        if verbose {
            println!("VibeVoice: loaded acoustic encoder ({layout:?}, {prefix})");
        }
    }

    if let Some((prefix, layout)) = detect_encoder_layout(&store, "semantic") {
        model.semantic_encoder =
            load_encoder(&mut store, &prefix, layout, &model.semantic_encoder, device)?;
        if verbose {
            println!("VibeVoice: loaded semantic encoder ({layout:?}, {prefix})");
        }
    }

    if store.has_key("language_model.model.embed_tokens.weight")
        || store.has_key("language_model.lm_head.weight")
        || store.has_key("model.language_model.embed_tokens.weight")
        || store.has_key("lm_head.weight")
    {
        let proj_quant = if model_name == "bitnet-asr" {
            LinearQuant::Ternary
        } else {
            LinearQuant::Int8
        };
        model.decoder = load_decoder(&mut store, &model.decoder, proj_quant, device)?;
        if verbose {
            let label = match proj_quant {
                LinearQuant::Ternary => "ternary I2_S",
                LinearQuant::Int8 => "INT8",
            };
            println!("VibeVoice: loaded Qwen2.5 language model as {label} (greedy STT)");
        }
    } else if verbose {
        println!("VibeVoice: language model weights missing — decoder is uninitialized");
    }

    Ok(())
}