use whisburn_core::PreloadModels;

fn env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub default_model: String,
    pub device: Option<String>,
    pub hf_token: Option<String>,
    pub verbose: bool,
    pub debug: bool,
    pub open_browser: bool,
    pub max_upload_bytes: usize,
    pub max_audio_seconds: u64,
    pub preload_models: PreloadModels,
    /// Download/use the offline Qwen3-0.6B summarizer.
    pub summarize_enabled: bool,
    /// UI checkbox default (and `summarize` query default when the param is omitted).
    pub summarize_default: bool,
    /// Offline summarizer (`qwen3-0.6b` or `granite-4.2-3b-fp8`).
    pub summarizer_model: String,
    /// Keep at least this many MiB of GPU memory free for the compositor.
    pub vram_reserve_mb: u32,
    /// Keep at least this percentage of total VRAM free (0–100).
    pub vram_reserve_percent: u32,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: std::env::var("WHISBURN_PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(8787),
            default_model: std::env::var("WHISBURN_DEFAULT_MODEL").unwrap_or_else(|_| "tiny_en".to_string()),
            device: std::env::var("WHISBURN_DEVICE").ok(),
            hf_token: std::env::var("HF_TOKEN").ok(),
            verbose: std::env::var("WHISBURN_VERBOSE").is_ok(),
            debug: std::env::var("WHISBURN_DEBUG").is_ok(),
            open_browser: true,
            max_upload_bytes: 512 * 1024 * 1024,
            max_audio_seconds: 24 * 60 * 60,
            preload_models: std::env::var("WHISBURN_PRELOAD_MODELS")
                .map(|v| PreloadModels::from_cli_values(Some(v.split(',').map(|s| s.trim().to_string()).collect())))
                .unwrap_or(PreloadModels::None),
            summarize_enabled: env_flag("WHISBURN_SUMMARIZE").unwrap_or(true),
            summarize_default: env_flag("WHISBURN_SUMMARIZE_DEFAULT")
                .unwrap_or_else(|| env_flag("WHISBURN_SUMMARIZE").unwrap_or(true)),
            summarizer_model: std::env::var("WHISBURN_SUMMARIZER")
                .unwrap_or_else(|_| "qwen3-0.6b".to_string()),
            vram_reserve_mb: std::env::var("WHISBURN_VRAM_RESERVE_MB")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(whisburn_engine::vram::DEFAULT_RESERVE_MB),
            vram_reserve_percent: std::env::var("WHISBURN_VRAM_RESERVE_PERCENT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(whisburn_engine::vram::DEFAULT_RESERVE_PERCENT)
                .min(100),
        }
    }
}

impl ServerConfig {
    pub fn vram_policy(&self) -> whisburn_engine::vram::VramPolicy {
        whisburn_engine::vram::VramPolicy {
            reserve_mb: self.vram_reserve_mb,
            reserve_percent: self.vram_reserve_percent,
        }
    }

    pub fn from_settings_and_cli(
        file: whisburn_core::Settings,
        port: Option<u16>,
        model: Option<String>,
        preload_models: Option<PreloadModels>,
        device: Option<String>,
        hf_token: Option<String>,
        verbose: bool,
        debug: bool,
        open_browser: bool,
    ) -> Self {
        Self {
            port: port.unwrap_or(file.server_port),
            default_model: model.unwrap_or(file.default_model),
            preload_models: preload_models.unwrap_or(file.preload_models),
            device,
            hf_token,
            verbose,
            debug,
            open_browser,
            ..Self::default()
        }
    }
}