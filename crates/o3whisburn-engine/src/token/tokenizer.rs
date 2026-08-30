use std::fs;
use std::path::Path;

use serde::Deserialize;
use serde::ser::StdError;
use std::collections::HashMap;
use std::result;
use tokenizers::models::bpe::{BpeBuilder, BPE};
use tokenizers::AddedToken;
use crate::token::special::SpecialToken;

#[derive(Debug, Deserialize)]
struct Qwen3AddedTokenEntry {
    content: String,
    special: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct Qwen3TokenizerConfig {
    added_tokens_decoder: Option<HashMap<String, Qwen3AddedTokenEntry>>,
}

fn read_qwen3_added_tokens(config_path: &str) -> Option<Vec<AddedToken>> {
    let config_raw = fs::read_to_string(config_path).ok()?;
    let config = serde_json::from_str::<Qwen3TokenizerConfig>(&config_raw).ok()?;
    let added = config.added_tokens_decoder?;
    let mut tokens: Vec<AddedToken> = added
        .values()
        .map(|entry| AddedToken::from(entry.content.as_str(), true))
        .collect();
    tokens.sort_by_key(|t| t.content.clone());
    Some(tokens)
}

fn load_qwen3_tokenizer(model_name: &str) -> Result<tokenizers::Tokenizer> {
    let model_dir = format!("models/{model_name}");
    let vocab_path = format!("{model_dir}/vocab.json");
    let merges_path = format!("{model_dir}/merges.txt");
    let config_path = format!("{model_dir}/tokenizer_config.json");

    let (mut vocab, merges) = BPE::read_file(&vocab_path, &merges_path)
        .map_err(|e| format!("failed to read Qwen3 vocab/merges: {e}"))?;

    if let Ok(config_raw) = fs::read_to_string(&config_path) {
        if let Ok(config) = serde_json::from_str::<Qwen3TokenizerConfig>(&config_raw) {
            if let Some(added) = config.added_tokens_decoder {
                for (id_str, entry) in added {
                    if let Ok(id) = id_str.parse::<u32>() {
                        vocab.insert(entry.content, id);
                    }
                }
            }
        }
    }

    let bpe = BpeBuilder::new()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| format!("failed to build Qwen3 BPE: {e}"))?;

    let mut tokenizer = tokenizers::Tokenizer::new(bpe);
    if let Some(specials) = read_qwen3_added_tokens(&config_path) {
        tokenizer.add_special_tokens(&specials);
    }
    Ok(tokenizer)
}

pub type Result<T> = result::Result<T, Box<dyn StdError + Send + Sync + 'static>>;

pub struct Gpt2Tokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl Gpt2Tokenizer {
    pub fn new(model_name: &str) -> Result<Self> {
        let json_path = format!("models/{}/tokenizer.json", model_name);
        
        let vocab_path = format!("models/{}/vocab.json", model_name);
        let merges_path = format!("models/{}/merges.txt", model_name);

        let is_qwen_family = model_name.contains("qwen3-asr")
            || model_name.contains("vibevoice")
            || model_name.contains("bitnet")
            || model_name.contains("moonshine")
            || model_name.contains("t-one");

        let mut tokenizer = if Path::new(&json_path).exists() {
            tokenizers::Tokenizer::from_file(&json_path)?
        } else if is_qwen_family
            && Path::new(&vocab_path).exists()
            && Path::new(&merges_path).exists()
        {
            load_qwen3_tokenizer(model_name)?
        } else if Path::new(&vocab_path).exists() && Path::new(&merges_path).exists() {
            let bpe = tokenizers::models::bpe::BPE::from_file(&vocab_path, &merges_path)
                .build()
                .map_err(|e| format!("failed to load BPE from vocab/merges: {e}"))?;
            tokenizers::Tokenizer::new(bpe)
        } else {
            tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default())
        };

        if !is_qwen_family {
            tokenizer.add_special_tokens(crate::token::special::construct_special_tokens().as_slice());
        }

        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.encode_with_special_tokens(text, true)
    }

    pub fn encode_with_special_tokens(&self, text: &str, add_special_tokens: bool) -> Vec<usize> {
        let tokens = self
            .tokenizer
            .encode(text, add_special_tokens)
            .unwrap();
        tokens.get_ids().iter().map(|t| *t as usize).collect()
    }

    pub fn special_token_by_name(&self, name: &str) -> Option<usize> {
        self.tokenizer.token_to_id(name).map(|id| id as usize)
    }

    pub fn special_token(&self, token: SpecialToken) -> Option<usize> {
        self.tokenizer
            .token_to_id(&token.to_string())
            .map(|id| id as usize)
    }


    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.tokenizer
            .token_to_id(token)
            .map(|id| id as usize)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    pub fn decode(&self, tokens: &[usize], skip_special: bool) -> Result<String> {
        self.tokenizer.decode(
            &tokens.iter().map(|t| *t as u32).collect::<Vec<u32>>(),
            skip_special,
        )
    }

    pub fn is_special(&self, token: usize) -> bool {
        self.tokenizer
            .decode(vec![token as u32].as_slice(), true)
            .ok()
            .map(|s| s.is_empty())
            .unwrap_or(false)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub fn is_timestamp(&self, token: usize) -> bool {
        if let Some(s) = self.tokenizer.id_to_token(token as u32) {
            // Pattern: <|dd.dd|>
            if s.starts_with("<|") && s.ends_with("|>") && s.len() >= 7 {
                let inner = &s[2..s.len()-2];
                return inner.parse::<f64>().is_ok();
            }
        }
        false
    }

    pub fn token_to_timestamp(&self, token: usize) -> Option<f64> {
        if let Some(s) = self.tokenizer.id_to_token(token as u32) {
            if s.starts_with("<|") && s.ends_with("|>") && s.len() >= 7 {
                let inner = &s[2..s.len()-2];
                return inner.parse::<f64>().ok();
            }
        }
        None
    }
}
