use burn::tensor::{Device, DeviceKind};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use burn::module::{Module, Param};
use burn::tensor::{activation::silu, activation::softmax, Tensor};
use half::bf16;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

use crate::model::attention::attn_decoder_mask;
use crate::model::conformer::RMSNorm;
use crate::model::qwen3::summarize::{chunk_transcript, strip_think, SummarizeProgress};
use crate::token::Gpt2Tokenizer;
use whisburn_core::{WhisburnError, WhisburnResult};

use super::fp8::f8_e4m3_to_f32;

const CHUNK_WORDS: usize = 700;
const CHUNK_MAX_NEW_TOKENS: usize = 160;
const MERGE_MAX_NEW_TOKENS: usize = 192;
const SYSTEM_PROMPT: &str = "You summarize English speech transcripts. Be concise. Use short bullet points. Cover: what it is about, key points, decisions, and action items. Do not invent facts.";

fn one() -> f64 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct GraniteConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default = "one")]
    pub embedding_multiplier: f64,
    #[serde(default = "one")]
    pub residual_multiplier: f64,
    #[serde(default = "one")]
    pub logits_scaling: f64,
    #[serde(default = "one")]
    pub attention_multiplier: f64,
    #[serde(default)]
    pub eos_token_id: Option<usize>,
}

impl GraniteConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Clone)]
struct Packed {
    dtype: Dtype,
    shape: Vec<usize>,
    start: usize,
    len: usize,
}

struct WeightStore {
    bytes: Vec<u8>,
    tensors: HashMap<String, Packed>,
}

impl WeightStore {
    fn open(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let view = SafeTensors::deserialize(&bytes).map_err(|e| format!("safetensors: {e}"))?;
        let base = bytes.as_ptr() as usize;
        let mut tensors = HashMap::new();
        for name in view.names() {
            let tensor = view.tensor(name).map_err(|e| e.to_string())?;
            let start = tensor.data().as_ptr() as usize - base;
            tensors.insert(
                name.to_string(),
                Packed {
                    dtype: tensor.dtype(),
                    shape: tensor.shape().to_vec(),
                    start,
                    len: tensor.data().len(),
                },
            );
        }
        Ok(Self { bytes, tensors })
    }

    fn get(&self, name: &str) -> Result<&Packed, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}"))
    }

    fn payload(&self, packed: &Packed) -> &[u8] {
        &self.bytes[packed.start..packed.start + packed.len]
    }

    fn f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let packed = self.get(name)?;
        decode_plain(self.payload(packed), packed.dtype)
    }

    /// HF linear `[out, in]` dequantized and transposed to Burn `[in, out]`.
    fn linear_in_out(&self, name: &str) -> Result<(Vec<f32>, usize, usize), String> {
        let weight = self.get(name)?;
        if weight.shape.len() != 2 {
            return Err(format!("{name} rank {:?}", weight.shape));
        }
        let (out, inp) = (weight.shape[0], weight.shape[1]);
        let scale_name = format!("{name}_scale");
        let scales = if self.tensors.contains_key(&scale_name) {
            self.f32(&scale_name)?
        } else {
            vec![1.0; out]
        };
        if scales.len() != out && scales.len() != out * 1 {
            return Err(format!(
                "{scale_name} has {} values, expected {out}",
                scales.len()
            ));
        }
        let raw = self.payload(weight);
        let mut transposed = vec![0.0f32; inp * out];
        match weight.dtype {
            Dtype::F8_E4M3 => {
                if raw.len() < out * inp {
                    return Err(format!("{name} FP8 payload is short"));
                }
                for o in 0..out {
                    let scale = scales[o];
                    for i in 0..inp {
                        let v = f8_e4m3_to_f32(raw[o * inp + i]) * scale;
                        transposed[i * out + o] = v;
                    }
                }
            }
            Dtype::F16 | Dtype::BF16 | Dtype::F32 => {
                let plain = decode_plain(raw, weight.dtype)?;
                if plain.len() < out * inp {
                    return Err(format!("{name} payload is short"));
                }
                for o in 0..out {
                    let scale = scales[o];
                    for i in 0..inp {
                        transposed[i * out + o] = plain[o * inp + i] * scale;
                    }
                }
            }
            other => return Err(format!("{name} dtype {other:?} is not a Granite weight")),
        }
        Ok((transposed, inp, out))
    }
}

fn decode_plain(data: &[u8], dtype: Dtype) -> Result<Vec<f32>, String> {
    match dtype {
        Dtype::F32 => Ok(data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        Dtype::F16 => Ok(data
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        Dtype::BF16 => Ok(data
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        other => Err(format!("unsupported dense dtype {other:?}")),
    }
}

type KvCache = (Tensor< 4>, Tensor< 4>);

pub struct GraniteModel {
    cfg: GraniteConfig,
    store: WeightStore,
    input_norm: Vec<RMSNorm>,
    post_norm: Vec<RMSNorm>,
    final_norm: RMSNorm,
    device: Device,
}

impl GraniteModel {
    pub fn load(model_dir: &Path, device: &Device) -> Result<Self, String> {
        let cfg: GraniteConfig = serde_json::from_str(
            &fs::read_to_string(model_dir.join("config.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("granite config.json: {e}"))?;
        if cfg.hidden_size % cfg.num_attention_heads != 0 {
            return Err("granite hidden_size is not divisible by num_attention_heads".into());
        }
        let store = WeightStore::open(&model_dir.join("model.safetensors"))?;
        let mut input_norm = Vec::with_capacity(cfg.num_hidden_layers);
        let mut post_norm = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            input_norm.push(load_norm(
                &store,
                &format!("model.layers.{i}.input_layernorm.weight"),
                cfg.rms_norm_eps,
                device,
            )?);
            post_norm.push(load_norm(
                &store,
                &format!("model.layers.{i}.post_attention_layernorm.weight"),
                cfg.rms_norm_eps,
                device,
            )?);
        }
        let final_norm = load_norm(&store, "model.norm.weight", cfg.rms_norm_eps, device)?;
        Ok(Self {
            cfg,
            store,
            input_norm,
            post_norm,
            final_norm,
            device: device.clone(),
        })
    }

    pub fn generate(
        &self,
        prefix: &[usize],
        eos_ids: &[usize],
        max_new: usize,
    ) -> Result<Vec<usize>, String> {
        if prefix.is_empty() || max_new == 0 {
            return Ok(Vec::new());
        }
        let mut ids = prefix.to_vec();
        let mut x = self.embed(&ids)?;
        let mut caches: Vec<Option<KvCache>> = vec![None; self.cfg.num_hidden_layers];
        let (mut hidden, filled) = self.forward_layers(x, &mut caches, 0)?;
        caches = filled;
        let mut generated = Vec::new();
        let mut pos = prefix.len();
        for _ in 0..max_new {
            let next = self.argmax_token(&hidden)?;
            if eos_ids.contains(&next) {
                break;
            }
            generated.push(next);
            ids.push(next);
            x = self.embed(&[next])?;
            let (h, filled) = self.forward_layers(x, &mut caches, pos)?;
            hidden = h;
            caches = filled;
            pos += 1;
            let _ = &ids;
        }
        Ok(generated)
    }

    fn embed(&self, ids: &[usize]) -> Result<Tensor< 3>, String> {
        let packed = self.store.get("model.embed_tokens.weight")?;
        if packed.shape.len() != 2 {
            return Err("embed_tokens rank".into());
        }
        let (vocab, hidden) = (packed.shape[0], packed.shape[1]);
        let row_bytes = match packed.dtype {
            Dtype::BF16 | Dtype::F16 => 2,
            Dtype::F32 => 4,
            other => return Err(format!("embed dtype {other:?}")),
        };
        let raw = self.store.payload(packed);
        let mut data = vec![0.0f32; ids.len() * hidden];
        let mult = self.cfg.embedding_multiplier as f32;
        for (t, &id) in ids.iter().enumerate() {
            if id >= vocab {
                return Err(format!("token {id} outside vocab {vocab}"));
            }
            let row = &raw[id * hidden * row_bytes..(id + 1) * hidden * row_bytes];
            let values = decode_plain(row, packed.dtype)?;
            for (h, v) in values.into_iter().enumerate() {
                data[t * hidden + h] = v * mult;
            }
        }
        Ok(Tensor::< 1>::from_floats(data.as_slice(), &self.device)
            .reshape([1, ids.len(), hidden]))
    }

    fn forward_layers(
        &self,
        mut x: Tensor< 3>,
        caches: &mut [Option<KvCache>],
        pos0: usize,
    ) -> Result<(Tensor< 3>, Vec<Option<KvCache>>), String> {
        let seq = x.dims()[1];
        let (cos, sin) = rope_embeddings(
            pos0,
            seq,
            self.cfg.head_dim(),
            self.cfg.rope_theta,
            &self.device,
        );
        let mask = if caches.iter().any(|c| c.is_some()) || pos0 > 0 {
            Tensor::< 2>::zeros([1, pos0 + seq], &self.device)
        } else {
            attn_decoder_mask(seq, &self.device)
        };
        let mut out_caches = Vec::with_capacity(self.cfg.num_hidden_layers);
        for i in 0..self.cfg.num_hidden_layers {
            let residual = x.clone();
            let normed = self.input_norm[i].forward(x);
            let (attn, cache) = self.attention(i, normed, &cos, &sin, &mask, caches[i].take())?;
            x = residual + attn.mul_scalar(self.cfg.residual_multiplier);
            let residual = x.clone();
            let normed = self.post_norm[i].forward(x);
            let mlp = self.mlp(i, normed)?;
            x = residual + mlp.mul_scalar(self.cfg.residual_multiplier);
            out_caches.push(Some(cache));
        }
        Ok((self.final_norm.forward(x), out_caches))
    }

    fn attention(
        &self,
        layer: usize,
        x: Tensor< 3>,
        cos: &Tensor< 3>,
        sin: &Tensor< 3>,
        mask: &Tensor< 2>,
        cache: Option<KvCache>,
    ) -> Result<(Tensor< 3>, KvCache), String> {
        let [batch, seq, _] = x.dims();
        let heads = self.cfg.num_attention_heads;
        let kv_heads = self.cfg.num_key_value_heads;
        let dim = self.cfg.head_dim();
        let n_rep = heads / kv_heads;
        let q = self
            .proj(&format!("model.layers.{layer}.self_attn.q_proj.weight"), x.clone())?
            .reshape([batch, seq, heads, dim]);
        let k = self
            .proj(&format!("model.layers.{layer}.self_attn.k_proj.weight"), x.clone())?
            .reshape([batch, seq, kv_heads, dim]);
        let v = self
            .proj(&format!("model.layers.{layer}.self_attn.v_proj.weight"), x)?
            .reshape([batch, seq, kv_heads, dim]);
        let q = apply_rope(q.swap_dims(1, 2), cos, sin);
        let k_new = apply_rope(k.swap_dims(1, 2), cos, sin);
        let v_new = v.swap_dims(1, 2);
        let (k_full, v_full) = match cache {
            Some((k_cache, v_cache)) => (
                Tensor::cat(vec![k_cache, k_new], 2),
                Tensor::cat(vec![v_cache, v_new], 2),
            ),
            None => (k_new, v_new),
        };
        let k_rep = repeat_kv(k_full.clone(), n_rep);
        let v_rep = repeat_kv(v_full.clone(), n_rep);
        let mut scores = q
            .matmul(k_rep.swap_dims(2, 3))
            .mul_scalar(self.cfg.attention_multiplier);
        scores = scores + mask.clone().unsqueeze::<3>().unsqueeze::<4>();
        scores = softmax(scores, 3);
        let out = scores
            .matmul(v_rep)
            .swap_dims(1, 2)
            .reshape([batch, seq, heads * dim]);
        let out = self.proj(&format!("model.layers.{layer}.self_attn.o_proj.weight"), out)?;
        Ok((out, (k_full, v_full)))
    }

    fn mlp(&self, layer: usize, x: Tensor< 3>) -> Result<Tensor< 3>, String> {
        let gate = silu(self.proj(&format!("model.layers.{layer}.mlp.gate_proj.weight"), x.clone())?);
        let up = self.proj(&format!("model.layers.{layer}.mlp.up_proj.weight"), x)?;
        self.proj(
            &format!("model.layers.{layer}.mlp.down_proj.weight"),
            gate * up,
        )
    }

    fn proj(&self, name: &str, x: Tensor< 3>) -> Result<Tensor< 3>, String> {
        let (weight, inp, out) = self.store.linear_in_out(name)?;
        let [batch, seq, h] = x.dims();
        if h != inp {
            return Err(format!("{name} expected in={inp}, got {h}"));
        }
        let flat = x.reshape([batch * seq, inp]);
        let w = Tensor::< 1>::from_floats(weight.as_slice(), &self.device).reshape([inp, out]);
        Ok(flat.matmul(w).reshape([batch, seq, out]))
    }

    fn argmax_token(&self, hidden: &Tensor< 3>) -> Result<usize, String> {
        let [_, seq, h] = hidden.dims();
        let last = hidden.clone().slice([0..1, seq - 1..seq, 0..h]).reshape([1, h]);
        let packed = self.store.get("lm_head.weight")?;
        let (vocab, hidden_size) = (packed.shape[0], packed.shape[1]);
        if hidden_size != h {
            return Err(format!("lm_head hidden {hidden_size} != {h}"));
        }
        let raw = self.store.payload(packed);
        let row_bytes = match packed.dtype {
            Dtype::BF16 | Dtype::F16 => 2,
            Dtype::F32 => 4,
            other => return Err(format!("lm_head dtype {other:?}")),
        };
        let scale = 1.0 / self.cfg.logits_scaling;
        let mut best_i = 0usize;
        let mut best = f32::NEG_INFINITY;
        let step = 1024usize;
        let mut start = 0usize;
        while start < vocab {
            let n = step.min(vocab - start);
            let mut w = vec![0.0f32; h * n];
            for row in 0..n {
                let src = (start + row) * h * row_bytes;
                let values = decode_plain(&raw[src..src + h * row_bytes], packed.dtype)?;
                for c in 0..h {
                    w[c * n + row] = values[c] * scale as f32;
                }
            }
            let wt = Tensor::< 1>::from_floats(w.as_slice(), &self.device).reshape([h, n]);
            let scores = last.clone().matmul(wt);
            let data = scores.into_data();
            let vals = data
                .try_to_vec::<f32>()
                .map_err(|_| "lm_head scores".to_string())?;
            for (i, v) in vals.iter().enumerate() {
                if *v > best {
                    best = *v;
                    best_i = start + i;
                }
            }
            start += n;
        }
        Ok(best_i)
    }
}

fn load_norm(
    store: &WeightStore,
    name: &str,
    eps: f64,
    device: &Device,
) -> Result<RMSNorm, String> {
    let weight = store.f32(name)?;
    let mut norm = RMSNorm::with_eps(weight.len(), eps, device);
    let mut loaded = norm.clone();
    loaded.gamma = Param::from_tensor(Tensor::< 1>::from_floats(weight.as_slice(), device));
    norm = loaded;
    Ok(norm)
}

fn apply_rope(x: Tensor< 4>, cos: &Tensor< 3>, sin: &Tensor< 3>) -> Tensor< 4> {
    let [b, h, s, d] = x.dims();
    let half = d / 2;
    let x1 = x.clone().slice([0..b, 0..h, 0..s, 0..half]);
    let x2 = x.clone().slice([0..b, 0..h, 0..s, half..d]);
    let rotated = Tensor::cat(vec![x2.mul_scalar(-1.0), x1], 3);
    let cos = cos.clone().unsqueeze_dim::<4>(1);
    let sin = sin.clone().unsqueeze_dim::<4>(1);
    x * cos + rotated * sin
}

fn repeat_kv(x: Tensor< 4>, n_rep: usize) -> Tensor< 4> {
    if n_rep == 1 {
        return x;
    }
    let [b, n_kv, s, d] = x.dims();
    let mut heads = Vec::with_capacity(n_kv * n_rep);
    for i in 0..n_kv {
        let head = x.clone().slice([0..b, i..i + 1, 0..s, 0..d]);
        for _ in 0..n_rep {
            heads.push(head.clone());
        }
    }
    Tensor::cat(heads, 1)
}

fn rope_embeddings(
    pos0: usize,
    seq: usize,
    head_dim: usize,
    theta: f64,
    device: &Device,
) -> (Tensor< 3>, Tensor< 3>) {
    let half = head_dim / 2;
    let mut cos_data = vec![0.0f32; seq * head_dim];
    let mut sin_data = vec![0.0f32; seq * head_dim];
    for t in 0..seq {
        let pos = (pos0 + t) as f64;
        for i in 0..half {
            let inv = 1.0 / theta.powf((2 * i) as f64 / head_dim as f64);
            let freq = pos * inv;
            let c = freq.cos() as f32;
            let s = freq.sin() as f32;
            cos_data[t * head_dim + i] = c;
            cos_data[t * head_dim + half + i] = c;
            sin_data[t * head_dim + i] = s;
            sin_data[t * head_dim + half + i] = s;
        }
    }
    let cos = Tensor::< 1>::from_floats(cos_data.as_slice(), device).reshape([1, seq, head_dim]);
    let sin = Tensor::< 1>::from_floats(sin_data.as_slice(), device).reshape([1, seq, head_dim]);
    (cos, sin)
}

pub fn is_granite_model(name: &str) -> bool {
    name == "granite-4.2-3b-fp8"
}

pub fn load_granite(
    model_name: &str,
    device: &Device,
) -> Result<GraniteModel, String> {
    let dir = whisburn_core::resolve_model_dir(model_name);
    GraniteModel::load(&dir, device)
}

fn build_prompt(user: &str) -> String {
    format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think></think>"
    )
}

pub fn summarize_text(
    model_name: &str,
    device: &Device,
    text: &str,
    verbose: bool,
    progress: Option<SummarizeProgress>,
) -> WhisburnResult<String> {
    if verbose {
        tracing::info!(model = model_name, "loading Granite FP8 summarizer");
    }
    let model = load_granite(model_name, device).map_err(WhisburnError::Model)?;
    let tokenizer = Gpt2Tokenizer::new(model_name)
        .map_err(|e| WhisburnError::Model(format!("granite tokenizer: {e}")))?;
    let mut eos = Vec::new();
    if let Some(id) = model.cfg.eos_token_id {
        eos.push(id);
    }
    for token in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(id) = tokenizer.token_to_id(token) {
            eos.push(id);
        }
    }
    if eos.is_empty() {
        eos.push(100_257);
    }

    let chunks = chunk_transcript(text, CHUNK_WORDS);
    if chunks.is_empty() {
        return Ok(String::new());
    }
    let n = chunks.len();
    let mut partials = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        if let Some(cb) = &progress {
            cb(
                i,
                n + usize::from(n > 1),
                &format!("Summarizing section {} of {n}", i + 1),
            );
        }
        let user = if n == 1 {
            format!("Summarize this English transcript:\n\n{chunk}")
        } else {
            format!(
                "This is section {} of {n} of a long English transcript. Summarize only this section:\n\n{chunk}",
                i + 1
            )
        };
        partials.push(complete(
            &model,
            &tokenizer,
            &build_prompt(&user),
            &eos,
            CHUNK_MAX_NEW_TOKENS,
        )?);
    }
    if partials.len() <= 1 {
        return Ok(partials.pop().unwrap_or_default());
    }
    if let Some(cb) = &progress {
        cb(n, n + 1, "Merging section summaries");
    }
    let joined = partials
        .iter()
        .enumerate()
        .map(|(i, p)| format!("Section {}:\n{p}", i + 1))
        .collect::<Vec<_>>()
        .join("\n\n");
    let merge = format!(
        "Combine these section summaries of a long English conversation into one concise summary with: overview, key points, decisions, action items.\n\n{joined}"
    );
    complete(
        &model,
        &tokenizer,
        &build_prompt(&merge),
        &eos,
        MERGE_MAX_NEW_TOKENS,
    )
}

fn complete(
    model: &GraniteModel,
    tokenizer: &Gpt2Tokenizer,
    prompt: &str,
    eos: &[usize],
    max_new: usize,
) -> WhisburnResult<String> {
    let ids = tokenizer.encode_with_special_tokens(prompt, false);
    let gen = model
        .generate(&ids, eos, max_new)
        .map_err(WhisburnError::Inference)?;
    let raw = tokenizer
        .decode(&gen, true)
        .map_err(|e| WhisburnError::Inference(e.to_string()))?;
    Ok(strip_think(&raw))
}

#[cfg(test)]
mod tests {
    use super::*;
        use half::bf16;
    use safetensors::tensor::{serialize, Dtype as SafeDtype, TensorView};
    use std::collections::HashMap;

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for v in values {
            out.extend_from_slice(&bf16::from_f32(*v).to_le_bytes());
        }
        out
    }

    #[test]
    fn zero_fp8_granite_prefill_returns_a_token() {
        let hidden = 8usize;
        let vocab = 6usize;
        let intermediate = 16usize;
        let heads = 2usize;
        let kv = 2usize;
        let head_dim = hidden / heads;
        let mut owned: Vec<(String, Vec<u8>, Vec<usize>, SafeDtype)> = Vec::new();
        owned.push((
            "model.embed_tokens.weight".into(),
            bf16_bytes(&vec![0.1; vocab * hidden]),
            vec![vocab, hidden],
            SafeDtype::BF16,
        ));
        owned.push((
            "model.norm.weight".into(),
            bf16_bytes(&vec![1.0; hidden]),
            vec![hidden],
            SafeDtype::BF16,
        ));
        owned.push((
            "lm_head.weight".into(),
            bf16_bytes(&vec![0.0; vocab * hidden]),
            vec![vocab, hidden],
            SafeDtype::BF16,
        ));
        let specs = [
            ("self_attn.q_proj", heads * head_dim, hidden),
            ("self_attn.k_proj", kv * head_dim, hidden),
            ("self_attn.v_proj", kv * head_dim, hidden),
            ("self_attn.o_proj", hidden, heads * head_dim),
            ("mlp.gate_proj", intermediate, hidden),
            ("mlp.up_proj", intermediate, hidden),
            ("mlp.down_proj", hidden, intermediate),
        ];
        for (name, out, inp) in specs {
            let weight_name = format!("model.layers.0.{name}.weight");
            owned.push((
                format!("{weight_name}_scale"),
                bf16_bytes(&vec![1.0; out]),
                vec![out, 1],
                SafeDtype::BF16,
            ));
            owned.push((
                weight_name,
                vec![0u8; out * inp],
                vec![out, inp],
                SafeDtype::F8_E4M3,
            ));
        }
        for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            owned.push((
                format!("model.layers.0.{name}"),
                bf16_bytes(&vec![1.0; hidden]),
                vec![hidden],
                SafeDtype::BF16,
            ));
        }

        let views: Vec<(&str, TensorView<'_>)> = owned
            .iter()
            .map(|(name, data, shape, dtype)| {
                (
                    name.as_str(),
                    TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect();
        let bytes = serialize(views, None::<HashMap<String, String>>).unwrap();
        let dir = std::env::temp_dir().join(format!("whisburn-granite-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("model.safetensors"), bytes).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "vocab_size": 6,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000.0,
                "attention_multiplier": 0.5,
                "embedding_multiplier": 1.0,
                "residual_multiplier": 1.0,
                "logits_scaling": 1.0,
                "eos_token_id": 5
            }"#,
        )
        .unwrap();
        let device = Device::flex();
        let model = GraniteModel::load(&dir, &device).unwrap();
        let next = model.generate(&[1, 2], &[5], 1).unwrap();
        assert_eq!(next.len(), 1);
        assert!(next[0] < vocab);
        let _ = fs::remove_dir_all(&dir);
    }
}
