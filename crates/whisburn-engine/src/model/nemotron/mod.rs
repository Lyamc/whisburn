//! Offline Nemotron 3.5 ASR streaming 0.6B (RNN-T).
//!
//! The encoder is a causal FastConformer: three stride-2 convs (8×), then 24
//! macaron blocks whose attention is limited to 14 chunks of left context and
//! one chunk of lookahead. Streaming caches are not implemented. A file is
//! encoded in windows and decoded with greedy RNN-T.

mod weights;

use burn::tensor::{Device, DeviceKind};
use burn::config::Config;
use burn::module::{Initializer, Module, Param};
use burn::nn::conv::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation::{relu, sigmoid, silu};
use burn::tensor::ops::PadMode;
use burn::tensor::{Tensor};

use crate::audio::parakeet_valid_mel_frames;
use crate::model::attention::RelPosMultiHeadAttention;
use crate::model::parakeet_decoder::{ParakeetDecoder, ParakeetDecoderCache, ParakeetDecoderConfig};

pub use weights::{load_nemotron_weights, read_nemotron_config};

/// Mel frames per encoder window. 1500 × 160 samples is 15 s, longer than the
/// 56-frame left context, and well under the 5000-frame position cap.
pub const NEMOTRON_ENCODER_CTX: usize = 1500;

#[derive(Config, Debug)]
pub struct NemotronConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub mel_bins: usize,
    pub conv_kernel: usize,
    pub sub_channels: usize,
    pub sub_kernel: usize,
    pub sub_stride: usize,
    pub sub_stages: usize,
    pub decoder_hidden: usize,
    pub decoder_layers: usize,
    pub vocab: usize,
    pub blank_id: usize,
    pub num_prompts: usize,
    pub prompt_hidden: usize,
    pub max_symbols: usize,
    pub sliding_window: usize,
    pub lookahead: usize,
}

impl NemotronConfig {
    pub fn init(&self, device: &Device) -> Nemotron {
        let conv_in = zero_conv2d([1, self.sub_channels], [self.sub_kernel, self.sub_kernel])
            .with_stride([self.sub_stride, self.sub_stride])
            .init(device);
        let sub_layers = (0..self.sub_stages.saturating_sub(1))
            .map(|_| SubsampleStage {
                depthwise: zero_conv2d(
                    [self.sub_channels, self.sub_channels],
                    [self.sub_kernel, self.sub_kernel],
                )
                .with_stride([self.sub_stride, self.sub_stride])
                .with_groups(self.sub_channels)
                .init(device),
                pointwise: zero_conv2d(
                    [self.sub_channels, self.sub_channels],
                    [1, 1],
                )
                .init(device),
            })
            .collect();
        let freq = nemotron_freq_bins(self.mel_bins, self.sub_stages, self.sub_kernel, self.sub_stride);
        let sub_linear = zero_linear(self.sub_channels * freq, self.hidden, true, device);
        let blocks = (0..self.n_layers)
            .map(|_| NemotronBlock::new(self.hidden, self.intermediate, self.n_heads, self.conv_kernel, device))
            .collect();
        let prompt_1 = zero_linear(self.hidden + self.num_prompts, self.prompt_hidden, true, device);
        let prompt_2 = zero_linear(self.prompt_hidden, self.hidden, true, device);
        let encoder_projector = zero_linear(self.hidden, self.decoder_hidden, true, device);
        let decoder = ParakeetDecoderConfig {
            vocab_size: self.vocab,
            hidden_size: self.decoder_hidden,
            num_layers: self.decoder_layers,
        }
        .init(device);
        let joint = zero_linear(self.decoder_hidden, self.vocab, true, device);
        let chunk_size = self.lookahead.saturating_add(1).max(1);
        let left_chunks = self.sliding_window.saturating_sub(1) / chunk_size;

        Nemotron {
            conv_in,
            sub_layers,
            sub_linear,
            blocks,
            prompt_1,
            prompt_2,
            encoder_projector,
            decoder,
            joint,
            blank_id: self.blank_id,
            max_symbols: self.max_symbols.max(1),
            num_prompts: self.num_prompts.max(1),
            chunk_size,
            left_chunks,
            mel_bins: self.mel_bins,
            sub_stages: self.sub_stages,
        }
    }
}

#[derive(Module, Debug)]
pub struct SubsampleStage {
    pub depthwise: Conv2d,
    pub pointwise: Conv2d,
}

#[derive(Module, Debug)]
pub struct NemotronFf {
    pub lin1: Linear,
    pub lin2: Linear,
}

impl NemotronFf {
    fn new(hidden: usize, intermediate: usize, device: &Device) -> Self {
        Self {
            lin1: zero_linear(hidden, intermediate, false, device),
            lin2: zero_linear(intermediate, hidden, false, device),
        }
    }

    fn forward(&self, x: Tensor< 3>) -> Tensor< 3> {
        let x = silu(self.lin1.forward(x));
        self.lin2.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct NemotronConv {
    pub pointwise1: Conv1d,
    pub depthwise: Conv1d,
    pub norm: LayerNorm,
    pub pointwise2: Conv1d,
}

impl NemotronConv {
    fn new(channels: usize, kernel: usize, device: &Device) -> Self {
        Self {
            pointwise1: Conv1dConfig::new(channels, channels * 2, 1)
                .with_bias(false)
                .with_initializer(Initializer::Zeros)
                .init(device),
            depthwise: Conv1dConfig::new(channels, channels, kernel)
                .with_groups(channels)
                .with_bias(false)
                .with_initializer(Initializer::Zeros)
                .init(device),
            norm: LayerNormConfig::new(channels).init(device),
            pointwise2: Conv1dConfig::new(channels, channels, 1)
                .with_bias(false)
                .with_initializer(Initializer::Zeros)
                .init(device),
        }
    }

    /// Pointwise expand, GLU, causal depthwise, LayerNorm, SiLU, pointwise project.
    fn forward(&self, x: Tensor< 3>) -> Tensor< 3> {
        let x = x.swap_dims(1, 2);
        let x = self.pointwise1.forward(x);
        let [b, c2, t] = x.dims();
        let c = c2 / 2;
        let a = x.clone().slice([0..b, 0..c, 0..t]);
        let g = x.slice([0..b, c..c2, 0..t]);
        let x = a * sigmoid(g);
        let x = causal_conv1d(&self.depthwise, x);
        let x = x.swap_dims(1, 2);
        let x = self.norm.forward(x).swap_dims(1, 2);
        let x = silu(x);
        self.pointwise2.forward(x).swap_dims(1, 2)
    }
}

#[derive(Module, Debug)]
pub struct NemotronBlock {
    pub norm_ff1: LayerNorm,
    pub ff1: NemotronFf,
    pub norm_attn: LayerNorm,
    pub attn: RelPosMultiHeadAttention,
    pub norm_conv: LayerNorm,
    pub conv: NemotronConv,
    pub norm_ff2: LayerNorm,
    pub ff2: NemotronFf,
    pub norm_out: LayerNorm,
}

impl NemotronBlock {
    fn new(hidden: usize, intermediate: usize, heads: usize, kernel: usize, device: &Device) -> Self {
        let head_dim = hidden / heads.max(1);
        let attn = RelPosMultiHeadAttention {
            n_head: heads,
            query: zero_linear(hidden, hidden, false, device),
            key: zero_linear(hidden, hidden, false, device),
            value: zero_linear(hidden, hidden, false, device),
            out: zero_linear(hidden, hidden, false, device),
            pos: zero_linear(hidden, hidden, false, device),
            pos_bias_u: Param::from_tensor(Tensor::zeros([heads, head_dim], device)),
            pos_bias_v: Param::from_tensor(Tensor::zeros([heads, head_dim], device)),
        };
        Self {
            norm_ff1: LayerNormConfig::new(hidden).init(device),
            ff1: NemotronFf::new(hidden, intermediate, device),
            norm_attn: LayerNormConfig::new(hidden).init(device),
            attn,
            norm_conv: LayerNormConfig::new(hidden).init(device),
            conv: NemotronConv::new(hidden, kernel, device),
            norm_ff2: LayerNormConfig::new(hidden).init(device),
            ff2: NemotronFf::new(hidden, intermediate, device),
            norm_out: LayerNormConfig::new(hidden).init(device),
        }
    }

    fn forward(&self, x: Tensor< 3>, mask: Tensor< 2>) -> Tensor< 3> {
        let x = x.clone() + self.ff1.forward(self.norm_ff1.forward(x)).mul_scalar(0.5);
        let x = x.clone() + self.attn.forward(self.norm_attn.forward(x.clone()), Some(mask.clone()));
        let x = x.clone() + self.conv.forward(self.norm_conv.forward(x.clone()));
        let x = x.clone() + self.ff2.forward(self.norm_ff2.forward(x)).mul_scalar(0.5);
        self.norm_out.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct Nemotron {
    pub conv_in: Conv2d,
    pub sub_layers: Vec<SubsampleStage>,
    pub sub_linear: Linear,
    pub blocks: Vec<NemotronBlock>,
    pub prompt_1: Linear,
    pub prompt_2: Linear,
    pub encoder_projector: Linear,
    pub decoder: ParakeetDecoder,
    pub joint: Linear,
    #[module(skip)]
    pub blank_id: usize,
    #[module(skip)]
    pub max_symbols: usize,
    #[module(skip)]
    pub num_prompts: usize,
    #[module(skip)]
    pub chunk_size: usize,
    #[module(skip)]
    pub left_chunks: usize,
    #[module(skip)]
    pub mel_bins: usize,
    #[module(skip)]
    pub sub_stages: usize,
}

impl Nemotron {
    pub fn device(&self) -> Device {
        self.encoder_projector.weight.device()
    }

    pub fn encoder_ctx_size(&self) -> usize {
        NEMOTRON_ENCODER_CTX
    }

    pub fn encoder_mel_size(&self) -> usize {
        self.mel_bins
    }

    /// Mel is `[batch, mel, time]` from the log-mel frontend. Returns the
    /// joint encoder vectors `[batch, frames, decoder_hidden]`.
    pub fn encode(&self, mel: Tensor< 3>, prompt_id: usize) -> Tensor< 3> {
        let [_b, _mels, time] = mel.dims();
        let valid_mel = parakeet_valid_mel_frames(time).min(time);
        // Frontend is [batch, mel, time]. The conv sees [batch, 1, time, mel].
        let x = mel.swap_dims(1, 2).unsqueeze_dim::<4>(1);
        let (x, valid) = self.subsample(x, valid_mel);
        let x = x.permute([0, 2, 1, 3]);
        let [b2, t_sub, c, f] = x.dims();
        let flat = c * f;
        let expected = self.sub_linear.weight.dims()[0];
        let x = if flat == expected {
            x.reshape([b2, t_sub, flat])
        } else if f > 0 && expected % c == 0 {
            let keep = (expected / c).min(f);
            x.slice([0..b2, 0..t_sub, 0..c, 0..keep])
                .reshape([b2, t_sub, c * keep])
        } else {
            x.reshape([b2, t_sub, flat])
        };
        let mut hidden = self.sub_linear.forward(x);
        let t_full = hidden.dims()[1];
        let valid = valid.min(t_full).max(1);
        if valid < t_full {
            let d = hidden.dims()[2];
            hidden = hidden.slice([0..b2, 0..valid, 0..d]);
        }
        let frames = hidden.dims()[1];
        let mask = chunk_mask(frames, self.chunk_size, self.left_chunks, &hidden.device());
        for block in &self.blocks {
            hidden = block.forward(hidden, mask.clone());
        }
        let prompted = self.fuse_prompt(hidden, prompt_id);
        self.encoder_projector.forward(prompted)
    }

    pub fn greedy(&self, encoded: Tensor< 3>) -> Vec<usize> {
        let [_, frames, hidden] = encoded.dims();
        if frames == 0 {
            return Vec::new();
        }
        let device = self.device();
        let mut cache = ParakeetDecoderCache::new(&device, self.decoder.lstm.layers.len(), self.decoder.hidden_size());
        let mut prev = self.blank_id;
        let mut frame = 0usize;
        let mut symbols = 0usize;
        let mut tokens = Vec::new();
        let cap = frames.saturating_mul(self.max_symbols).saturating_add(frames).max(1);
        for _ in 0..cap {
            if frame >= frames {
                break;
            }
            let dec = self.decoder.forward_step(prev, self.blank_id, &mut cache, prev != self.blank_id);
            let enc = encoded
                .clone()
                .slice([0..1, frame..frame + 1, 0..hidden])
                .reshape([1, hidden]);
            let logits = self.joint.forward(relu(enc + dec));
            let id = logits
                .argmax(1)
                .into_data()
                .try_to_vec::<i32>()
                .ok()
                .and_then(|v| v.first().copied())
                .unwrap_or(self.blank_id as i32) as usize;
            let force = id != self.blank_id && symbols + 1 >= self.max_symbols;
            if id != self.blank_id {
                tokens.push(id);
            }
            if id == self.blank_id || force {
                frame += 1;
                symbols = 0;
            } else {
                symbols += 1;
            }
            prev = id;
        }
        tokens
    }

    fn subsample(&self, mut x: Tensor< 4>, mut valid: usize) -> (Tensor< 4>, usize) {
        x = causal_conv2d(&self.conv_in, x);
        valid = subsampled_len(valid);
        x = zero_tail(x, valid);
        x = relu(x);
        for stage in &self.sub_layers {
            x = causal_conv2d(&stage.depthwise, x);
            valid = subsampled_len(valid);
            x = stage.pointwise.forward(x);
            x = zero_tail(x, valid);
            x = relu(x);
        }
        (x, valid)
    }

    fn fuse_prompt(&self, hidden: Tensor< 3>, prompt_id: usize) -> Tensor< 3> {
        let [b, t, _] = hidden.dims();
        let n = self.num_prompts;
        let mut oh = vec![0f32; n];
        oh[prompt_id.min(n - 1)] = 1.0;
        let one = Tensor::< 1>::from_floats(oh.as_slice(), &hidden.device())
            .reshape([1, 1, n])
            .repeat(&[b, t, 1]);
        let fused = Tensor::cat(vec![hidden, one], 2);
        let fused = relu(self.prompt_1.forward(fused));
        self.prompt_2.forward(fused)
    }
}

fn subsampled_len(len: usize) -> usize {
    len / 2 + 1
}

/// Causal conv2d: pad time and frequency by `(kernel - 1, stride - 1)` then stride.
fn causal_conv2d(conv: &Conv2d, x: Tensor< 4>) -> Tensor< 4> {
    let [kh, kw] = conv.kernel_size;
    let [sh, sw] = conv.stride;
    let time = (kh.saturating_sub(1), sh.saturating_sub(1));
    let freq = (kw.saturating_sub(1), sw.saturating_sub(1));
    let x = x.pad((freq.0, freq.1, time.0, time.1), PadMode::Constant(0.0));
    conv.forward(x)
}

fn causal_conv1d(conv: &Conv1d, x: Tensor< 3>) -> Tensor< 3> {
    let left = conv.kernel_size.saturating_sub(1);
    let x = if left > 0 {
        x.pad([(left, 0)], PadMode::Constant(0.0))
    } else {
        x
    };
    conv.forward(x)
}

fn zero_tail(x: Tensor< 4>, valid: usize) -> Tensor< 4> {
    let [b, c, t, f] = x.dims();
    if valid >= t {
        return x;
    }
    let head = x.clone().slice([0..b, 0..c, 0..valid, 0..f]);
    let tail = Tensor::zeros([b, c, t - valid, f], &x.device());
    Tensor::cat(vec![head, tail], 2)
}

fn chunk_mask(frames: usize, chunk: usize, left_chunks: usize, device: &Device) -> Tensor< 2> {
    let mut data = vec![0f32; frames * frames];
    for q in 0..frames {
        for k in 0..frames {
            if !attends(q, k, chunk, left_chunks) {
                data[q * frames + k] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::< 1>::from_floats(data.as_slice(), device).reshape([frames, frames])
}

fn attends(q: usize, kv: usize, chunk: usize, left_chunks: usize) -> bool {
    let chunk = chunk.max(1);
    let diff = (q / chunk) as i64 - (kv / chunk) as i64;
    diff >= 0 && diff <= left_chunks as i64
}

fn nemotron_freq_bins(n_mels: usize, stages: usize, kernel: usize, stride: usize) -> usize {
    let pad = kernel.saturating_sub(1) + stride.saturating_sub(1);
    let mut f = n_mels;
    let stride = stride.max(1);
    for _ in 0..stages {
        f = (f + pad - kernel) / stride + 1;
    }
    f.max(1)
}

#[cfg(test)]
fn nemotron_subsampled_frames(valid_mel: usize, stages: usize) -> usize {
    let mut len = valid_mel;
    for _ in 0..stages {
        len = subsampled_len(len);
    }
    len
}

fn zero_linear(d_in: usize, d_out: usize, bias: bool, device: &Device) -> Linear {
    LinearConfig::new(d_in, d_out)
        .with_bias(bias)
        .with_initializer(Initializer::Zeros)
        .init(device)
}

fn zero_conv2d(channels: [usize; 2], kernel: [usize; 2]) -> Conv2dConfig {
    Conv2dConfig::new(channels, kernel).with_initializer(Initializer::Zeros)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn freq_path_matches_the_4352_projection() {
        assert_eq!(nemotron_freq_bins(128, 3, 3, 2), 17);
        assert_eq!(17 * 256, 4352);
    }

    #[test]
    fn causal_time_length_is_half_plus_one() {
        assert_eq!(nemotron_subsampled_frames(100, 3), 14);
        assert_eq!(nemotron_subsampled_frames(1, 3), 2);
    }

    #[test]
    fn chunk_window_is_fourteen_left_and_one_lookahead_chunk() {
        let chunk = 4;
        let left = 14;
        assert!(attends(0, 0, chunk, left));
        assert!(attends(0, 3, chunk, left));
        assert!(!attends(0, 4, chunk, left));
        assert!(attends(7, 0, chunk, left));
        assert!(!attends(7, 8, chunk, left));
        assert!(attends(56, 0, chunk, left));
        assert!(!attends(60, 0, chunk, left));
    }
}
