use std::collections::HashMap;
use std::fs;

use burn::tensor::{Tensor};
use serde::Deserialize;

use crate::model::nemotron::Nemotron;
use crate::token::Gpt2Tokenizer;

#[derive(Deserialize)]
struct ProcessorFile {
    prompt_dictionary: Option<HashMap<String, usize>>,
}

pub fn decode_nemotron(
    model: &Nemotron,
    bpe: &Gpt2Tokenizer,
    mel: Tensor< 3>,
    model_name: &str,
    language_code: &str,
    verbose: bool,
) -> (String, Vec<usize>) {
    let prompt = nemotron_prompt_id(model_name, language_code);
    if verbose {
        println!("Nemotron prompt id {prompt} for language '{language_code}'");
    }
    let encoded = model.encode(mel, prompt);
    let tokens = model.greedy(encoded);
    if verbose {
        let preview = tokens.len().min(12);
        println!("Nemotron tokens ({}): {:?}", tokens.len(), &tokens[..preview]);
    }
    let text = bpe.decode(&tokens, true).unwrap_or_default();
    (text, tokens)
}

/// Language id from `processor_config.json`. `en` and `en-US` are 0.
/// An unknown code stays on English rather than the checkpoint's auto id.
pub fn nemotron_prompt_id(model_name: &str, language: &str) -> usize {
    let lang = language.trim();
    if let Some(id) = prompt_dictionary(model_name).and_then(|dict| lookup(&dict, lang)) {
        return id;
    }
    match lang.to_ascii_lowercase().as_str() {
        "en" | "en-us" | "english" | "" => 0,
        "en-gb" | "engb" => 1,
        "auto" => 101,
        _ => 0,
    }
}

fn prompt_dictionary(model_name: &str) -> Option<HashMap<String, usize>> {
    let path = whisburn_core::resolve_model_dir(model_name).join("processor_config.json");
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str::<ProcessorFile>(&text)
        .ok()
        .and_then(|file| file.prompt_dictionary)
}

fn lookup(dict: &HashMap<String, usize>, lang: &str) -> Option<usize> {
    if lang.is_empty() {
        return dict.get("en-US").copied();
    }
    if let Some(id) = dict.get(lang) {
        return Some(*id);
    }
    dict.iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(lang))
        .map(|(_, id)| *id)
}
