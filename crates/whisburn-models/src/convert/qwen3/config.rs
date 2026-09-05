use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct Qwen3HfConfig {
    pub thinker_config: Qwen3ThinkerHfConfig,
}

#[derive(Debug, Deserialize)]
pub struct Qwen3ThinkerHfConfig {
    pub audio_config: Qwen3AudioHfConfig,
    pub text_config: Qwen3TextHfConfig,
    pub audio_start_token_id: usize,
    pub audio_end_token_id: usize,
    pub audio_token_id: usize,
}

#[derive(Debug, Deserialize)]
pub struct Qwen3AudioHfConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub num_mel_bins: usize,
    pub output_dim: usize,
    pub downsample_hidden_size: usize,
    pub n_window: usize,
    #[serde(default = "default_n_window_infer")]
    pub n_window_infer: usize,
    #[serde(default = "default_conv_chunksize")]
    pub conv_chunksize: usize,
    #[serde(default = "default_max_source_positions")]
    pub max_source_positions: usize,
}

fn default_n_window_infer() -> usize {
    800
}
fn default_conv_chunksize() -> usize {
    500
}
fn default_max_source_positions() -> usize {
    1500
}

#[derive(Debug, Deserialize)]
pub struct Qwen3TextHfConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: Option<Qwen3RopeScaling>,
}

#[derive(Debug, Deserialize)]
pub struct Qwen3RopeScaling {
    pub mrope_section: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3RuntimeConfig {
    pub burn_bundle_version: String,
    pub sample_rate: usize,
    pub num_mel_bins: usize,
    pub audio_d_model: usize,
    pub audio_layers: usize,
    pub audio_heads: usize,
    pub audio_ffn_dim: usize,
    pub audio_output_dim: usize,
    pub downsample_hidden_size: usize,
    pub n_window: usize,
    pub n_window_infer: usize,
    pub conv_chunksize: usize,
    pub max_source_positions: usize,
    pub text_hidden_size: usize,
    pub text_layers: usize,
    pub text_heads: usize,
    pub text_kv_heads: usize,
    pub text_head_dim: usize,
    pub text_intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub mrope_section: Vec<usize>,
    pub audio_start_token_id: usize,
    pub audio_end_token_id: usize,
    pub audio_pad_token_id: usize,
    pub im_start_token_id: usize,
    pub im_end_token_id: usize,
    pub asr_text_token_id: usize,
    pub eos_token_ids: Vec<usize>,
    pub inference_status: String,
    #[serde(default)]
    pub text_only: bool,
}

impl Qwen3RuntimeConfig {
    pub fn from_hf(hf: &Qwen3HfConfig) -> Self {
        let thinker = &hf.thinker_config;
        let audio = &thinker.audio_config;
        let text = &thinker.text_config;
        let mrope_section = text
            .rope_scaling
            .as_ref()
            .and_then(|r| r.mrope_section.clone())
            .unwrap_or_else(|| vec![24, 20, 20]);
        Self {
            burn_bundle_version: super::BURN_BUNDLE_VERSION.to_string(),
            sample_rate: 16_000,
            num_mel_bins: audio.num_mel_bins,
            audio_d_model: audio.d_model,
            audio_layers: audio.encoder_layers,
            audio_heads: audio.encoder_attention_heads,
            audio_ffn_dim: audio.encoder_ffn_dim,
            audio_output_dim: audio.output_dim,
            downsample_hidden_size: audio.downsample_hidden_size,
            n_window: audio.n_window,
            n_window_infer: audio.n_window_infer,
            conv_chunksize: audio.conv_chunksize,
            max_source_positions: audio.max_source_positions,
            text_hidden_size: text.hidden_size,
            text_layers: text.num_hidden_layers,
            text_heads: text.num_attention_heads,
            text_kv_heads: text.num_key_value_heads,
            text_head_dim: text.head_dim,
            text_intermediate_size: text.intermediate_size,
            vocab_size: text.vocab_size,
            rms_norm_eps: text.rms_norm_eps,
            rope_theta: text.rope_theta,
            mrope_section,
            audio_start_token_id: thinker.audio_start_token_id,
            audio_end_token_id: thinker.audio_end_token_id,
            audio_pad_token_id: thinker.audio_token_id,
            im_start_token_id: 151_644,
            im_end_token_id: 151_645,
            asr_text_token_id: 151_704,
            eos_token_ids: vec![151_643, 151_645],
            inference_status: "ready".to_string(),
            text_only: false,
        }
    }

    pub fn from_text_hf(text: &Qwen3TextOnlyHfConfig) -> Self {
        let mrope_section = text
            .rope_scaling
            .as_ref()
            .and_then(|r| r.mrope_section.clone())
            .unwrap_or_else(|| vec![24, 20, 20]);
        let head_dim = if text.head_dim > 0 {
            text.head_dim
        } else {
            128
        };
        Self {
            burn_bundle_version: super::BURN_BUNDLE_VERSION.to_string(),
            sample_rate: 16_000,
            num_mel_bins: 0,
            audio_d_model: 0,
            audio_layers: 0,
            audio_heads: 0,
            audio_ffn_dim: 0,
            audio_output_dim: 0,
            downsample_hidden_size: 0,
            n_window: 0,
            n_window_infer: 0,
            conv_chunksize: 0,
            max_source_positions: 0,
            text_hidden_size: text.hidden_size,
            text_layers: text.num_hidden_layers,
            text_heads: text.num_attention_heads,
            text_kv_heads: text.num_key_value_heads,
            text_head_dim: head_dim,
            text_intermediate_size: text.intermediate_size,
            vocab_size: text.vocab_size,
            rms_norm_eps: text.rms_norm_eps,
            rope_theta: text.rope_theta,
            mrope_section,
            audio_start_token_id: 0,
            audio_end_token_id: 0,
            audio_pad_token_id: 0,
            im_start_token_id: 151_644,
            im_end_token_id: 151_645,
            asr_text_token_id: 0,
            eos_token_ids: vec![151_643, 151_645],
            inference_status: "text-lm".to_string(),
            text_only: true,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Qwen3TextOnlyHfConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    #[serde(default)]
    pub head_dim: usize,
    #[serde(default = "default_rms")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_theta")]
    pub rope_theta: f64,
    pub rope_scaling: Option<Qwen3RopeScaling>,
}

fn default_rms() -> f64 {
    1e-6
}
fn default_theta() -> f64 {
    1_000_000.0
}