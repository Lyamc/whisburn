pub mod audio;
pub mod transcribe;
pub mod download;

use burn::backend::wgpu::WgpuDevice;
use clap::Args;
use crate::token::Language;
use strum::IntoEnumIterator;

use crate::model::registry::get_model_names;

#[derive(Args, Debug, Clone)]
pub struct CommonArgs {
    /// Name of the model to use (e.g., tiny_en, base_en)
    #[arg(short, long, value_parser = get_model_names())]
    pub model: String,

    /// Language of the audio (e.g., en, fr). Defaults to English.
    #[arg(short, long, default_value = "en")]
    pub lang: String,

    /// Show detailed processing information
    #[arg(short, long)]
    pub verbose: bool,

    /// Enable low-level debug output
    #[arg(long, default_value_t = false)]
    pub debug: bool,

    /// Suppress real-time transcription output
    #[arg(short, long)]
    pub quiet: bool,

    /// Verify transcription against expected test string
    #[arg(long)]
    pub verify: bool,

    /// Hugging Face authentication token for gated models
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    // ADVANCED PARAMETERS
    /// [Advanced] Beam size for search (higher = better quality, slower)
    #[arg(long, default_value_t = 5, help_heading = "Advanced")]
    pub beam_size: usize,

    /// [Advanced] Maximum tokens to generate per chunk
    #[arg(long, default_value_t = 448, help_heading = "Advanced")]
    pub max_tokens: usize,

    /// [Advanced] Padding added to audio chunks (in frames)
    #[arg(long, default_value_t = 200, help_heading = "Advanced")]
    pub padding: usize,

    /// [Advanced] Backend device to use (0, 1, ... for specific GPU, or 'cpu')
    #[arg(long, help_heading = "Advanced")]
    pub device: Option<String>,
}

pub fn parse_language(lang_str: &str) -> Language {
    if lang_str.eq_ignore_ascii_case("auto") {
        return Language::English;
    }
    match Language::iter().find(|l| l.as_str() == lang_str) {
        Some(l) => l,
        None => {
            let supported: Vec<String> = Language::iter().map(|l| l.as_str().to_string()).collect();
            eprintln!("Invalid language: {}. Supported languages: {:?}", lang_str, supported);
            std::process::exit(1);
        }
    }
}

pub fn get_device(device_arg: &Option<String>) -> WgpuDevice {
    match device_arg {
        Some(s) => {
            if s.to_lowercase() == "cpu" {
                WgpuDevice::Cpu
            } else if let Ok(idx) = s.parse::<u32>() {
                WgpuDevice::DiscreteGpu(idx as usize)
            } else {
                eprintln!("Invalid device: {}. Using default.", s);
                WgpuDevice::default()
            }
        }
        None => WgpuDevice::default(),
    }
}

pub fn get_backend_info(device: &WgpuDevice) -> String {
    match device {
        WgpuDevice::DiscreteGpu(i) => format!("WGPU (Discrete GPU #{})", i),
        WgpuDevice::IntegratedGpu(i) => format!("WGPU (Integrated GPU #{})", i),
        WgpuDevice::VirtualGpu(i) => format!("WGPU (Virtual GPU #{})", i),
        WgpuDevice::Cpu => "WGPU (CPU)".to_string(),
        #[allow(deprecated)]
        WgpuDevice::BestAvailable => "WGPU (Best Available)".to_string(),
        WgpuDevice::DefaultDevice => "WGPU (Default Device)".to_string(),
        WgpuDevice::Existing(i) => format!("WGPU (Existing #{})", i),
    }
}
