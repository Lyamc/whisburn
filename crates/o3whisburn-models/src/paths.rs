use std::path::{Path, PathBuf};

pub fn models_dir() -> PathBuf {
    std::env::var("O3WHISBURN_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("models"))
}

pub fn model_dir(name: &str) -> PathBuf {
    models_dir().join(name)
}

pub fn is_model_ready(name: &str) -> bool {
    let dir = model_dir(name);
    has_burn_bundle(&dir, name)
}

pub fn needs_reconversion(name: &str, dir: &Path) -> bool {
    crate::convert::needs_reconversion_for(name, dir)
}

fn has_vibevoice_bundle(dir: &Path) -> bool {
    let index_path = dir.join("model.safetensors.index.json");
    if !index_path.exists() || !dir.join("tokenizer.json").exists() || !dir.join("config.json").exists() {
        return false;
    }
    if !dir.join("vibevoice_runtime.json").exists() {
        return false;
    }
    let Ok(raw) = std::fs::read_to_string(&index_path) else {
        return false;
    };
    let Ok(index) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return false;
    };
    let Some(map) = index.get("weight_map").and_then(|v| v.as_object()) else {
        return false;
    };
    let shards: std::collections::HashSet<&str> = map
        .values()
        .filter_map(|v| v.as_str())
        .collect();
    !shards.is_empty() && shards.iter().all(|shard| dir.join(shard).exists())
}

fn has_moonshine_bundle(dir: &Path) -> bool {
    dir.join("model.safetensors").exists()
        && dir.join("config.json").exists()
        && dir.join("tokenizer.json").exists()
        && dir.join("moonshine_runtime.json").exists()
}

fn has_tone_bundle(dir: &Path) -> bool {
    dir.join("model.safetensors").exists()
        && dir.join("config.json").exists()
        && dir.join("tone_runtime.json").exists()
}

fn has_qwen3_bundle(dir: &Path) -> bool {
    let has_tokenizer = dir.join("tokenizer.json").exists()
        || (dir.join("vocab.json").exists() && dir.join("merges.txt").exists());
    dir.join("model.safetensors").exists()
        && dir.join("config.json").exists()
        && dir.join("qwen3_runtime.json").exists()
        && has_tokenizer
}

pub fn has_burn_bundle(dir: &Path, name: &str) -> bool {
    if crate::convert::is_qwen3_model(name) {
        return has_qwen3_bundle(dir);
    }
    if crate::convert::is_moonshine_model(name) {
        return has_moonshine_bundle(dir);
    }
    if crate::convert::is_tone_model(name) {
        return has_tone_bundle(dir);
    }
    if crate::convert::is_vibevoice_model(name) {
        return has_vibevoice_bundle(dir);
    }

    let mpk_candidates = [
        dir.join("model.mpk"),
        dir.join(format!("{name}.mpk")),
        dir.join("model.mpk.gz"),
        dir.join(format!("{name}.mpk.gz")),
    ];
    let cfg_candidates = [
        dir.join("config.cfg"),
        dir.join(format!("{name}.cfg")),
    ];
    let tokenizer = dir.join("tokenizer.json");

    mpk_candidates.iter().any(|p| p.exists())
        && cfg_candidates.iter().any(|p| p.exists())
        && tokenizer.exists()
}