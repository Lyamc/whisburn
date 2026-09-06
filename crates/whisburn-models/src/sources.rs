use whisburn_engine::model::registry::{find_model, ModelInfo};

/// Pre-converted Burn Whisper bundles published by Gadersd.
pub const GADERSD_WHISPER_BURN: &str = "Gadersd/whisper-burn";

/// Burn 0.21 runtime bundles published for whisburn (skip local conversion).
pub const WHISBURN_BURN_REPO: &str = "lyamc/whisburn";

const WHISBURN_PUBLISHED_MODELS: &[&str] = &[
    "tiny",
    "tiny_en",
    "base",
    "base_en",
    "small",
    "small_en",
    "medium",
    "medium_en",
    "large-v3-turbo",
    "distil-medium-en",
    "distil-large-v3",
    "parakeet-tdt-0.6b-v3",
    "parakeet-ctc-0.6b",
    "parakeet-ctc-1.1b",
    "t-one",
    "moonshine-tiny",
    "moonshine-base",
    "qwen3-0.6b",
    "bitnet-asr",
];

/// Whisper models with ready-made Burn artifacts on Gadersd/whisper-burn.
const GADERSD_WHISPER_MODELS: &[&str] = &[
    "tiny",
    "tiny_en",
    "base",
    "base_en",
    "small",
    "small_en",
    "medium",
    "medium_en",
    "large-v1",
    "large-v2",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DownloadSource {
    /// Pre-converted `.mpk.gz` + cfg + tokenizer from Gadersd/whisper-burn.
    GadersdBurn {
        repo_id: &'static str,
        model_name: String,
    },
    /// Burn 0.21 `.mpk.gz` bundles on `lyamc/whisburn`.
    WhisburnBurn {
        repo_id: &'static str,
        model_name: String,
    },
    /// Upstream HuggingFace model repo (may require on-device conversion).
    HuggingFace {
        hf_id: String,
        model_name: String,
    },
}

impl DownloadSource {
    pub fn model_name(&self) -> &str {
        match self {
            Self::GadersdBurn { model_name, .. } | Self::WhisburnBurn { model_name, .. } => {
                model_name
            }
            Self::HuggingFace { model_name, .. } => model_name,
        }
    }

    pub fn repo_id(&self) -> &str {
        match self {
            Self::GadersdBurn { repo_id, .. } | Self::WhisburnBurn { repo_id, .. } => repo_id,
            Self::HuggingFace { hf_id, .. } => hf_id,
        }
    }
}

pub fn published_bundle_source(name: &str) -> Option<DownloadSource> {
    find_model(name)?;
    if !WHISBURN_PUBLISHED_MODELS.contains(&name) {
        return None;
    }
    Some(DownloadSource::WhisburnBurn {
        repo_id: WHISBURN_BURN_REPO,
        model_name: name.to_string(),
    })
}

pub fn resolve_download_source(name: &str) -> Option<DownloadSource> {
    let info = find_model(name)?;

    // Always use upstream HF for whisper — Gadersd bundles target burn 0.8 and are incompatible.
    Some(DownloadSource::HuggingFace {
        hf_id: info.hf_id.to_string(),
        model_name: name.to_string(),
    })
}

pub fn gadersd_bundle_files(model_name: &str) -> [String; 3] {
    [
        format!("{model_name}/{model_name}.cfg"),
        format!("{model_name}/{model_name}.mpk.gz"),
        format!("{model_name}/tokenizer.json"),
    ]
}

pub fn is_gadersd_whisper(info: &ModelInfo) -> bool {
    GADERSD_WHISPER_MODELS.contains(&info.name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_en_resolves_to_upstream_hf() {
        let source = resolve_download_source("tiny_en").unwrap();
        assert!(matches!(source, DownloadSource::HuggingFace { .. }));
        assert_eq!(source.repo_id(), "openai/whisper-tiny.en");
    }

    #[test]
    fn tiny_en_has_published_burn_bundle() {
        let source = published_bundle_source("tiny_en").unwrap();
        assert!(matches!(source, DownloadSource::WhisburnBurn { .. }));
        assert_eq!(source.repo_id(), WHISBURN_BURN_REPO);
    }

    #[test]
    fn parakeet_resolves_to_upstream_hf() {
        let source = resolve_download_source("parakeet-tdt-0.6b-v3").unwrap();
        assert!(matches!(source, DownloadSource::HuggingFace { .. }));
        assert_eq!(source.repo_id(), "nvidia/parakeet-tdt-0.6b-v3");
    }

    #[test]
    fn vibevoice_resolves_to_hf_variant() {
        let source = resolve_download_source("vibevoice-asr").unwrap();
        assert_eq!(source.repo_id(), "microsoft/VibeVoice-ASR-HF");
    }

    #[test]
    fn bitnet_asr_resolves_to_vibevoice_bitnet() {
        let source = resolve_download_source("bitnet-asr").unwrap();
        assert_eq!(source.repo_id(), "microsoft/VibeVoice-ASR-BitNet");
    }
}