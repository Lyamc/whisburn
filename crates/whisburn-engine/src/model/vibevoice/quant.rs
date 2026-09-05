//! Packed linear layers for VibeVoice Qwen2.5 decoders.
//!
//! * 7B (`vibevoice-asr`): per-output-channel INT8.
//! * 1.5B BitNet (`bitnet-asr`): per-tensor absmean ternary `{-1,0,+1}` (I2_S).
//!
//! Weights stay packed as `i8` on the host. Each forward dequantizes one layer
//! into f32, runs the matmul, then drops the f32 copy.

use std::fmt;
use std::sync::Arc;

use burn::module::{Ignored, Module, Param};
use burn::tensor::{backend::Backend, Tensor};

#[derive(Module)]
pub struct QuantLinear<B: Backend> {
    qweight: Ignored<Arc<Vec<i8>>>,
    scale_host: Ignored<Arc<Vec<f32>>>,
    #[module(ignore)]
    d_in: usize,
    #[module(ignore)]
    d_out: usize,
    pub scale: Param<Tensor<B, 1>>,
    pub bias: Option<Param<Tensor<B, 1>>>,
}

impl<B: Backend> fmt::Debug for QuantLinear<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QuantLinear")
            .field("d_in", &self.d_in)
            .field("d_out", &self.d_out)
            .field("packed_bytes", &self.qweight.0.len())
            .finish()
    }
}

impl<B: Backend> QuantLinear<B> {
    pub fn empty(d_in: usize, d_out: usize, with_bias: bool, device: &B::Device) -> Self {
        Self {
            qweight: Ignored(Arc::new(Vec::new())),
            scale_host: Ignored(Arc::new(vec![1.0; d_out])),
            d_in,
            d_out,
            scale: Param::from_tensor(Tensor::<B, 1>::ones([d_out], device)),
            bias: with_bias.then(|| Param::from_tensor(Tensor::<B, 1>::zeros([d_out], device))),
        }
    }

    pub fn from_quantized(
        qweight: Vec<i8>,
        scale: Vec<f32>,
        bias: Option<Vec<f32>>,
        d_in: usize,
        d_out: usize,
        device: &B::Device,
    ) -> Self {
        debug_assert_eq!(qweight.len(), d_in * d_out);
        debug_assert_eq!(scale.len(), d_out);
        let scale_t = Tensor::<B, 1>::from_floats(scale.as_slice(), device);
        let bias = bias.map(|b| Param::from_tensor(Tensor::<B, 1>::from_floats(b.as_slice(), device)));
        Self {
            qweight: Ignored(Arc::new(qweight)),
            scale_host: Ignored(Arc::new(scale)),
            d_in,
            d_out,
            scale: Param::from_tensor(scale_t),
            bias,
        }
    }

    pub fn device(&self) -> B::Device {
        self.scale.val().device()
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, d] = x.dims();
        debug_assert_eq!(d, self.d_in);
        let w = self.dequant(&x.device());
        let mut y = x.reshape([b * s, d]).matmul(w).reshape([b, s, self.d_out]);
        if let Some(bias) = &self.bias {
            y = y + bias.val().unsqueeze::<2>().unsqueeze::<3>();
        }
        y
    }

    fn dequant(&self, device: &B::Device) -> Tensor<B, 2> {
        let q = self.qweight.0.as_slice();
        let sc = self.scale_host.0.as_slice();
        let mut w = vec![0f32; self.d_in * self.d_out];
        for i in 0..self.d_in {
            let row = i * self.d_out;
            for j in 0..self.d_out {
                w[row + j] = q[row + j] as f32 * sc[j];
            }
        }
        Tensor::<B, 1>::from_floats(w.as_slice(), device).reshape([self.d_in, self.d_out])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearQuant {
    /// Symmetric INT8 with a scale per output channel.
    Int8,
    /// BitNet I2_S: per-tensor absmean, values in `{-1,0,+1}`.
    Ternary,
}

/// Symmetric per-output-channel INT8 (Burn layout `[d_in, d_out]`).
pub fn quantize_per_out_channel(weight: &[f32], d_in: usize, d_out: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; d_in * d_out];
    let mut scale = vec![0f32; d_out];
    for j in 0..d_out {
        let mut max = 0f32;
        for i in 0..d_in {
            max = max.max(weight[i * d_out + j].abs());
        }
        let s = (max / 127.0).max(1e-8);
        scale[j] = s;
        for i in 0..d_in {
            let v = (weight[i * d_out + j] / s).round().clamp(-127.0, 127.0);
            q[i * d_out + j] = v as i8;
        }
    }
    (q, scale)
}

/// BitNet absmean ternary used by VibeVoice-ASR-BitNet / VibeASR.cpp.
///
/// `scale = mean(|W|)`, `q = round(W / scale).clamp(-1, 1)`, reconstruct `q * scale`.
/// The scale vector is length `d_out` (identical values) so dequant stays per-column.
pub fn quantize_ternary_absmean(weight: &[f32], d_in: usize, d_out: usize) -> (Vec<i8>, Vec<f32>) {
    debug_assert_eq!(weight.len(), d_in * d_out);
    let n = weight.len().max(1) as f32;
    let mean_abs = weight.iter().map(|w| w.abs()).sum::<f32>() / n;
    let scale = mean_abs.max(1e-5);
    let mut q = vec![0i8; d_in * d_out];
    for (i, &w) in weight.iter().enumerate() {
        q[i] = (w / scale).round().clamp(-1.0, 1.0) as i8;
    }
    (q, vec![scale; d_out])
}

pub fn quantize_linear(
    scheme: LinearQuant,
    weight: &[f32],
    d_in: usize,
    d_out: usize,
) -> (Vec<i8>, Vec<f32>) {
    match scheme {
        LinearQuant::Int8 => quantize_per_out_channel(weight, d_in, d_out),
        LinearQuant::Ternary => quantize_ternary_absmean(weight, d_in, d_out),
    }
}

#[cfg(test)]
mod tests {
    use super::{quantize_per_out_channel, quantize_ternary_absmean};

    #[test]
    fn roundtrip_stays_close() {
        let d_in = 8;
        let d_out = 4;
        let w: Vec<f32> = (0..d_in * d_out)
            .map(|i| (i as f32 - 10.0) * 0.13)
            .collect();
        let (q, scale) = quantize_per_out_channel(&w, d_in, d_out);
        let mut max_err = 0f32;
        for i in 0..d_in {
            for j in 0..d_out {
                let rec = q[i * d_out + j] as f32 * scale[j];
                max_err = max_err.max((rec - w[i * d_out + j]).abs());
            }
        }
        assert!(max_err < 0.02, "max dequant error {max_err}");
    }

    #[test]
    fn ternary_absmean_matches_vibeasr() {
        let d_in = 6;
        let d_out = 4;
        let w: Vec<f32> = (0..d_in * d_out)
            .map(|i| (i as f32 - 11.0) * 0.07)
            .collect();
        let mean = w.iter().map(|x| x.abs()).sum::<f32>() / w.len() as f32;
        let (q, scale) = quantize_ternary_absmean(&w, d_in, d_out);
        assert!(q.iter().all(|&v| v == -1 || v == 0 || v == 1));
        for s in &scale {
            assert!((s - mean.max(1e-5)).abs() < 1e-6);
        }
        let mut max_err = 0f32;
        for i in 0..w.len() {
            let rec = q[i] as f32 * scale[0];
            max_err = max_err.max((rec - w[i]).abs());
        }
        assert!(max_err < mean + 1e-5, "ternary reconstruction error {max_err}");
    }
}
