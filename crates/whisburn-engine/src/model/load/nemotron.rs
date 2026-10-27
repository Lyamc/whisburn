use burn::tensor::{Device, DeviceKind};
use std::error::Error;


use crate::model::nemotron::{load_nemotron_weights, read_nemotron_config, NemotronConfig};
use crate::model::{Model, ModelConfig};
use crate::token::Gpt2Tokenizer;

pub fn load_nemotron_model(
    model_name: &str,
    device: &Device,
    verbose: bool,
) -> Result<(Gpt2Tokenizer, ModelConfig, Model), Box<dyn Error>> {
    let model_dir = whisburn_core::resolve_model_dir(model_name);
    let weights = model_dir.join("model.safetensors");
    if !weights.exists() {
        return Err(format!(
            "Nemotron weights missing at {}. Download nvidia/nemotron-3.5-asr-streaming-0.6b into that directory.",
            weights.display()
        )
        .into());
    }

    let config = read_nemotron_config(&model_dir)?;
    let bpe = Gpt2Tokenizer::new(model_name)
        .map_err(|e| format!("Failed to load tokenizer for {model_name}: {e}"))?;
    if verbose {
        println!(
            "Loading Nemotron ASR ({}, {} encoder layers, vocab {})",
            model_name, config.n_layers, config.vocab
        );
    }
    let mut model = NemotronConfig::init(&config, device);
    load_nemotron_weights(&mut model, &model_dir, device)?;
    if verbose {
        println!(
            "Nemotron loaded (blank {}, prompts {}, lookahead {})",
            config.blank_id, config.num_prompts, config.lookahead
        );
    }
    Ok((
        bpe,
        ModelConfig::Nemotron(config),
        Model::Nemotron(model),
    ))
}

pub fn is_nemotron_model(name: &str) -> bool {
    name.contains("nemotron")
}
