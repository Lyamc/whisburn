use std::collections::HashMap;
use std::fs;
use std::path::Path;

use safetensors::SafeTensors;

use super::dtype::bytes_to_f32;
use super::index::VibeVoiceWeightIndex;

pub struct VibeVoiceWeightStore {
    index: VibeVoiceWeightIndex,
    shard_cache: HashMap<String, Vec<u8>>,
}

impl VibeVoiceWeightStore {
    pub fn open(model_dir: &Path) -> Result<Self, String> {
        Ok(Self {
            index: VibeVoiceWeightIndex::open(model_dir)?,
            shard_cache: HashMap::new(),
        })
    }

    pub fn has_key(&self, key: &str) -> bool {
        self.resolve(key).is_some()
    }

    pub fn resolve(&self, key: &str) -> Option<String> {
        for cand in key_aliases(key) {
            if self.index.has_key(&cand) {
                return Some(cand);
            }
        }
        None
    }

    pub fn tensor_f32(&mut self, key: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        let key = self
            .resolve(key)
            .ok_or_else(|| format!("tensor key not in index: {key}"))?;
        let shard_path = self.index.shard_path(&key)?;
        let shard_name = shard_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| "invalid shard path".to_string())?
            .to_string();

        if !self.shard_cache.contains_key(&shard_name) {
            // Keep a single shard in RAM — the 8-shard 7B checkpoint is ~16 GB.
            self.shard_cache.clear();
            let bytes = fs::read(&shard_path)
                .map_err(|e| format!("read shard {}: {e}", shard_path.display()))?;
            self.shard_cache.insert(shard_name.clone(), bytes);
        }

        let bytes = self.shard_cache.get(&shard_name).unwrap();
        let tensors = SafeTensors::deserialize(bytes)
            .map_err(|e| format!("deserialize {}: {e}", shard_path.display()))?;
        let tensor = tensors
            .tensor(&key)
            .map_err(|e| format!("tensor {key} in {}: {e}", shard_path.display()))?;

        let shape: Vec<usize> = tensor.shape().iter().copied().collect();
        let floats = bytes_to_f32(tensor.data(), tensor.dtype())?;
        Ok((floats, shape))
    }
}

/// HF Transformers (`VibeVoice-ASR-HF`) vs original (`VibeVoice-ASR` / BitNet) key spellings.
fn key_aliases(key: &str) -> Vec<String> {
    let mut keys = Vec::new();
    let push = |keys: &mut Vec<String>, k: String| {
        if !keys.iter().any(|e| e == &k) {
            keys.push(k);
        }
    };

    push(&mut keys, key.to_string());
    let stripped = key.strip_prefix("model.").unwrap_or(key);
    push(&mut keys, stripped.to_string());
    if !key.starts_with("model.") {
        push(&mut keys, format!("model.{key}"));
    }

    let lm_rest = stripped
        .strip_prefix("language_model.model.")
        .or_else(|| stripped.strip_prefix("language_model."));
    if let Some(rest) = lm_rest {
        push(&mut keys, format!("language_model.model.{rest}"));
        push(&mut keys, format!("language_model.{rest}"));
        push(&mut keys, format!("model.language_model.{rest}"));
        push(&mut keys, format!("model.language_model.model.{rest}"));
    }

    if stripped == "language_model.lm_head.weight" || stripped.ends_with("lm_head.weight") {
        push(&mut keys, "lm_head.weight".into());
        push(&mut keys, "model.lm_head.weight".into());
        push(&mut keys, "language_model.lm_head.weight".into());
    }

    keys
}

#[cfg(test)]
mod tests {
    use super::key_aliases;

    #[test]
    fn aliases_bitnet_lm_keys() {
        let aliases = key_aliases("language_model.model.embed_tokens.weight");
        assert!(aliases.iter().any(|k| k == "model.language_model.embed_tokens.weight"));
        let head = key_aliases("language_model.lm_head.weight");
        assert!(head.iter().any(|k| k == "lm_head.weight"));
    }
}