use o3whisburn_core::PreloadModels;

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub default_model: String,
    pub device: Option<String>,
    pub hf_token: Option<String>,
    pub verbose: bool,
    pub debug: bool,
    pub max_upload_bytes: usize,
    pub max_audio_seconds: u64,
    pub preload_models: PreloadModels,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: std::env::var("O3WHISBURN_PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(8787),
            default_model: std::env::var("O3WHISBURN_DEFAULT_MODEL").unwrap_or_else(|_| "tiny_en".to_string()),
            device: std::env::var("O3WHISBURN_DEVICE").ok(),
            hf_token: std::env::var("HF_TOKEN").ok(),
            verbose: std::env::var("O3WHISBURN_VERBOSE").is_ok(),
            debug: std::env::var("O3WHISBURN_DEBUG").is_ok(),
            max_upload_bytes: 512 * 1024 * 1024,
            max_audio_seconds: 24 * 60 * 60,
            preload_models: std::env::var("O3WHISBURN_PRELOAD_MODELS")
                .map(|v| PreloadModels::from_cli_values(Some(v.split(',').map(|s| s.trim().to_string()).collect())))
                .unwrap_or(PreloadModels::None),
        }
    }
}

impl ServerConfig {
    pub fn from_settings_and_cli(
        file: o3whisburn_core::Settings,
        port: Option<u16>,
        model: Option<String>,
        preload_models: Option<PreloadModels>,
        device: Option<String>,
        hf_token: Option<String>,
        verbose: bool,
        debug: bool,
    ) -> Self {
        Self {
            port: port.unwrap_or(file.server_port),
            default_model: model.unwrap_or(file.default_model),
            preload_models: preload_models.unwrap_or(file.preload_models),
            device,
            hf_token,
            verbose,
            debug,
            ..Self::default()
        }
    }
}