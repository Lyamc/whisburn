pub mod load;
pub mod registry;
pub mod attention;
pub mod mlp;
pub mod encoder;
pub mod decoder;
pub mod whisper;
pub mod conformer;
pub mod parakeet;
pub mod parakeet_decoder;
pub mod tone;
pub mod moonshine;
pub mod qwen3;
pub mod vibevoice;

pub use load::*;
pub use attention::*;
pub use mlp::*;
pub use encoder::*;
pub use decoder::*;
pub use whisper::{Whisper, WhisperConfig};
pub use parakeet::*;
pub use tone::*;
pub use qwen3::*;
pub use vibevoice::*;

use burn::{
    config::Config,
    module::Module,
    tensor::backend::Backend,
};

#[derive(Config, Debug)]
pub enum ModelConfig {
    Whisper(whisper::WhisperConfig),
    Parakeet(parakeet::ParakeetConfig),
    TONE(tone::TONEConfig),
    Qwen3(qwen3::Qwen3ASRConfig),
    VibeVoice(vibevoice::VibeVoiceASRConfig),
    Moonshine(moonshine::MoonshineASRConfig),
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        match self {
            ModelConfig::Whisper(config) => Model::Whisper(config.init(device)),
            ModelConfig::Parakeet(config) => Model::Parakeet(config.init(device)),
            ModelConfig::TONE(config) => Model::TONE(config.init(device)),
            ModelConfig::Qwen3(config) => Model::Qwen3(config.init(device)),
            ModelConfig::VibeVoice(config) => Model::VibeVoice(config.init(device)),
            ModelConfig::Moonshine(config) => Model::Moonshine(config.init(device)),
        }
    }
}

#[derive(Module, Debug)]
pub enum Model<B: Backend> {
    Whisper(whisper::Whisper<B>),
    Parakeet(parakeet::Parakeet<B>),
    TONE(tone::TONE<B>),
    Qwen3(qwen3::Qwen3ASR<B>),
    VibeVoice(vibevoice::VibeVoiceASR<B>),
    Moonshine(moonshine::MoonshineASR<B>),
}

impl<B: Backend> Model<B> {
    pub fn encoder_ctx_size(&self) -> usize {
        match self {
            Model::Whisper(m) => m.encoder_ctx_size(),
            Model::Parakeet(m) => m.encoder_ctx_size(),
            Model::TONE(m) => m.encoder_ctx_size(),
            Model::Qwen3(m) => m.encoder_ctx_size(),
            Model::VibeVoice(m) => m.encoder_ctx_size(),
            Model::Moonshine(m) => m.encoder_ctx_size(),
        }
    }

    pub fn encoder_mel_size(&self) -> usize {
        match self {
            Model::Whisper(m) => m.encoder_mel_size(),
            Model::Parakeet(m) => m.encoder_mel_size(),
            Model::TONE(m) => m.encoder_mel_size(),
            Model::Qwen3(m) => m.encoder_mel_size(),
            Model::VibeVoice(m) => m.encoder_mel_size(),
            Model::Moonshine(m) => m.encoder_mel_size(),
        }
    }
}