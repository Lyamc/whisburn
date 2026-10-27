use std::path::{Path, PathBuf};

/// Directory where new model downloads are written.
///
/// Override with `WHISBURN_MODELS_DIR`. Defaults to `./models` (cwd).
pub fn models_dir() -> PathBuf {
    std::env::var("WHISBURN_MODELS_DIR")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models"))
}

/// Preferred on-disk location for a named model (used when downloading).
pub fn model_dir(name: &str) -> PathBuf {
    models_dir().join(name)
}

/// Directory that actually contains `name`, searching fallbacks used in development
/// and next to the binary. Falls back to [`model_dir`] if nothing is present yet.
pub fn resolve_model_dir(name: &str) -> PathBuf {
    let primary = model_dir(name);
    if model_dir_has_assets(&primary) {
        return primary;
    }
    for candidate in extra_model_dirs(name) {
        if model_dir_has_assets(&candidate) {
            return candidate;
        }
    }
    primary
}

pub fn resolve_model_file(name: &str, file: &str) -> PathBuf {
    resolve_model_dir(name).join(file)
}

/// Archive root with one folder per weight kind: `burn/`, `checkpoint/`, `gguf/`, `diffusers/`, `hf-cache/`.
///
/// Set `WHISBURN_MODEL_STORE`. The hot runtime dir stays `WHISBURN_MODELS_DIR`.
pub fn model_store_root() -> Option<PathBuf> {
    std::env::var("WHISBURN_MODEL_STORE")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
}

fn extra_model_dirs(name: &str) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            dirs.push(parent.join("models").join(name));
        }
    }
    if let Some(store) = model_store_root() {
        for kind in ["burn", "checkpoint", "gguf", "diffusers"] {
            dirs.push(store.join(kind).join(name));
        }
    }
    dirs.extend(hf_snapshot_dirs(name));
    dirs
}

fn hf_hub_dir() -> Option<PathBuf> {
    if let Ok(explicit) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        let explicit = explicit.trim();
        if !explicit.is_empty() {
            return Some(PathBuf::from(explicit));
        }
    }
    if let Ok(home) = std::env::var("HF_HOME") {
        let home = home.trim();
        if !home.is_empty() {
            return Some(PathBuf::from(home).join("hub"));
        }
    }
    model_store_root().map(|store| store.join("hf-cache").join("hub"))
}

fn hf_snapshot_dirs(name: &str) -> Vec<PathBuf> {
    let Some(hub) = hf_hub_dir() else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(&hub) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let folder = entry.file_name().to_string_lossy().to_string();
        if !hub_repo_matches(&folder, name) {
            continue;
        }
        let snapshots = entry.path().join("snapshots");
        let Ok(revs) = std::fs::read_dir(&snapshots) else {
            continue;
        };
        for rev in revs.flatten() {
            if rev.path().is_dir() {
                out.push(rev.path());
            }
        }
    }
    out
}

fn name_needles(name: &str) -> Vec<String> {
    let name = name.to_lowercase();
    let mut needles = vec![name.clone(), name.replace('_', "-"), name.replace('_', ".")];
    if let Some(stem) = name.strip_suffix("_en") {
        needles.push(format!("{stem}.en"));
        needles.push(format!("{}-en", stem.replace('_', "-")));
    }
    if let Some(stem) = name.strip_suffix("-en") {
        needles.push(format!("{stem}.en"));
    }
    needles.sort();
    needles.dedup();
    needles
}

fn hub_repo_matches(folder: &str, name: &str) -> bool {
    let folder = folder.to_lowercase();
    let repo = folder.split("--").last().unwrap_or(&folder);
    name_needles(name).iter().any(|needle| {
        repo == needle
            || repo.ends_with(&format!("-{needle}"))
            || repo.ends_with(&format!(".{needle}"))
    })
}

fn model_dir_has_assets(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    if dir.join("model_index.json").is_file() {
        return true;
    }
    let has_tokenizer = dir.join("tokenizer.json").is_file()
        || (dir.join("vocab.json").is_file() && dir.join("merges.txt").is_file());
    let has_gguf = std::fs::read_dir(dir).ok().is_some_and(|entries| {
        entries.flatten().any(|entry| {
            entry.path().extension().and_then(|ext| ext.to_str()) == Some("gguf")
        })
    });
    let has_weights = has_gguf
        || dir.join("model.bpk").is_file()
        || dir.join("model.safetensors").is_file()
        || dir.join("model.safetensors.index.json").is_file();
    let has_config = dir.join("config.cfg").is_file()
        || dir.join("config.json").is_file()
        || dir.join("qwen3_runtime.json").is_file()
        || dir.join("parakeet_decode.json").is_file()
        || dir.join("vibevoice_runtime.json").is_file()
        || dir.join("moonshine_runtime.json").is_file()
        || dir.join("tone_runtime.json").is_file();
    has_tokenizer && (has_weights || has_config)
}

#[cfg(test)]
mod tests {
    use super::{hub_repo_matches, model_dir_has_assets};
    use std::fs;

    #[test]
    fn asset_detection_counts_burnpack_and_ignores_named_mpk() {
        let path = std::env::temp_dir().join(format!(
            "whisburn-assets-{}-{}",
            std::process::id(),
            "bpk"
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join("tokenizer.json"), "{}").unwrap();
        fs::write(path.join("model.mpk"), b"old").unwrap();
        assert!(
            !model_dir_has_assets(&path),
            "a Named MPK file is not a loadable weight"
        );
        fs::write(path.join("model.bpk"), b"pack").unwrap();
        assert!(model_dir_has_assets(&path));
        let _ = fs::remove_dir_all(&path);
    }

    #[test]
    fn hf_cache_names_do_not_collapse_english_variants() {
        assert!(hub_repo_matches("models--openai--whisper-tiny.en", "tiny_en"));
        assert!(!hub_repo_matches(
            "models--openai--whisper-tiny.en",
            "tiny"
        ));
        assert!(hub_repo_matches("models--openai--whisper-tiny", "tiny"));
        assert!(hub_repo_matches(
            "models--nvidia--parakeet-tdt-0.6b-v3",
            "parakeet-tdt-0.6b-v3"
        ));
        assert!(hub_repo_matches(
            "models--distil-whisper--distil-medium.en",
            "distil-medium-en"
        ));
        assert!(hub_repo_matches(
            "models--openai--whisper-large-v3-turbo",
            "large-v3-turbo"
        ));
    }
}
