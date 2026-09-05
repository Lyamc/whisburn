use std::collections::HashMap;
use std::fs;
use std::path::Path;

use burn::module::{Module, Param};
use burn::nn::conv::Conv1d;
use burn::nn::{Embedding, LayerNorm, Linear};
use burn::tensor::{backend::Backend, Tensor};
use serde::{Deserialize, Serialize};

use super::{MoonshineASR, MoonshineDecoderLayer, MoonshineEncoderLayer};
use crate::model::vibevoice::weights::dtype::{bytes_to_f32, transpose_linear_weight};

pub const BURN_BUNDLE_VERSION: &str = "0.16.1-moonshine";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoonshineRuntimeConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub pad_head_dim_to_multiple_of: usize,
    pub decoder_start_token_id: usize,
    pub eos_token_id: usize,
    pub max_new_tokens: usize,
    pub sample_rate: usize,
    pub inference_status: String,
}

impl MoonshineRuntimeConfig {
    pub fn from_hf(hf: &MoonshineHfConfig) -> Self {
        Self {
            hidden_size: hf.hidden_size,
            intermediate_size: hf.intermediate_size,
            encoder_layers: hf.encoder_num_hidden_layers,
            decoder_layers: hf.decoder_num_hidden_layers,
            n_heads: hf.encoder_num_attention_heads.max(1),
            n_kv_heads: hf.encoder_num_key_value_heads.unwrap_or(hf.encoder_num_attention_heads).max(1),
            vocab_size: hf.vocab_size,
            rope_theta: hf.rope_theta.unwrap_or(10_000.0),
            partial_rotary_factor: hf.partial_rotary_factor.unwrap_or(0.9),
            pad_head_dim_to_multiple_of: hf.pad_head_dim_to_multiple_of.unwrap_or(8),
            decoder_start_token_id: hf.decoder_start_token_id.unwrap_or(1),
            eos_token_id: hf.eos_token_id.unwrap_or(2),
            max_new_tokens: 194,
            sample_rate: 16_000,
            inference_status: "ready".into(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct MoonshineHfConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub encoder_num_hidden_layers: usize,
    pub decoder_num_hidden_layers: usize,
    pub encoder_num_attention_heads: usize,
    pub encoder_num_key_value_heads: Option<usize>,
    pub vocab_size: usize,
    pub rope_theta: Option<f64>,
    pub partial_rotary_factor: Option<f64>,
    pub pad_head_dim_to_multiple_of: Option<usize>,
    pub decoder_start_token_id: Option<usize>,
    pub eos_token_id: Option<usize>,
}

pub fn load_moonshine_runtime(model_name: &str) -> MoonshineRuntimeConfig {
    let path = whisburn_core::resolve_model_file(model_name, "moonshine_runtime.json");
    fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(MoonshineRuntimeConfig {
            hidden_size: 288,
            intermediate_size: 1152,
            encoder_layers: 6,
            decoder_layers: 6,
            n_heads: 8,
            n_kv_heads: 8,
            vocab_size: 32768,
            rope_theta: 10_000.0,
            partial_rotary_factor: 0.9,
            pad_head_dim_to_multiple_of: 8,
            decoder_start_token_id: 1,
            eos_token_id: 2,
            max_new_tokens: 194,
            sample_rate: 16_000,
            inference_status: "missing-runtime".into(),
        })
}

pub fn weights_present(model_dir: &Path) -> bool {
    model_dir.join("model.safetensors").exists()
}

struct Store {
    tensors: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl Store {
    fn open(model_dir: &Path) -> Result<Self, String> {
        let path = model_dir.join("model.safetensors");
        let bytes = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| format!("parse {}: {e}", path.display()))?;
        let mut tensors = HashMap::new();
        for key in st.names() {
            let tensor = st.tensor(key).map_err(|e| format!("{key}: {e}"))?;
            let shape: Vec<usize> = tensor.shape().iter().copied().collect();
            let floats = bytes_to_f32(tensor.data(), tensor.dtype())?;
            tensors.insert(key.to_string(), (floats, shape));
        }
        if !tensors.contains_key("proj_out.weight") {
            if let Some(embed) = tensors.get("model.decoder.embed_tokens.weight").cloned() {
                tensors.insert("proj_out.weight".into(), embed);
            }
        }
        Ok(Self { tensors })
    }

    fn tensor_f32(&self, key: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        self.tensors
            .get(key)
            .cloned()
            .ok_or_else(|| format!("tensor key not found: {key}"))
    }
}

pub fn load_moonshine_weights<B: Backend>(
    model: &mut MoonshineASR<B>,
    model_dir: &Path,
    device: &B::Device,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if !weights_present(model_dir) {
        if verbose {
            println!("Moonshine: no safetensors weights found");
        }
        return Ok(());
    }
    let store = Store::open(model_dir)?;
    model.conv1 = load_conv1d(&store, "model.encoder.conv1", model.conv1.clone(), device)?;
    model.conv2 = load_conv1d(&store, "model.encoder.conv2", model.conv2.clone(), device)?;
    model.conv3 = load_conv1d(&store, "model.encoder.conv3", model.conv3.clone(), device)?;
    let (gw, _) = store.tensor_f32("model.encoder.groupnorm.weight")?;
    let (gb, _) = store.tensor_f32("model.encoder.groupnorm.bias")?;
    model.groupnorm_weight = Param::from_tensor(Tensor::<B, 1>::from_floats(gw.as_slice(), device));
    model.groupnorm_bias = Param::from_tensor(Tensor::<B, 1>::from_floats(gb.as_slice(), device));
    for (i, layer) in model.encoder_layers.iter_mut().enumerate() {
        *layer = load_encoder_layer(&store, i, layer, device)?;
    }
    model.encoder_norm = load_ln(&store, "model.encoder.layer_norm", model.encoder_norm.clone(), device)?;
    model.embed_tokens = load_embed(&store, "model.decoder.embed_tokens", model.embed_tokens.clone(), device)?;
    for (i, layer) in model.decoder_layers.iter_mut().enumerate() {
        *layer = load_decoder_layer(&store, i, layer, device)?;
    }
    model.decoder_norm = load_ln(&store, "model.decoder.norm", model.decoder_norm.clone(), device)?;
    model.proj_out = load_linear(&store, "proj_out", model.proj_out.clone(), device)?;
    if verbose {
        println!("Moonshine: loaded encoder + decoder weights from safetensors");
    }
    Ok(())
}

fn load_encoder_layer<B: Backend>(
    store: &Store,
    i: usize,
    layer: &MoonshineEncoderLayer<B>,
    device: &B::Device,
) -> Result<MoonshineEncoderLayer<B>, Box<dyn std::error::Error>> {
    let p = format!("model.encoder.layers.{i}");
    let mut record = layer.clone().into_record();
    record.input_layernorm = load_ln(store, &format!("{p}.input_layernorm"), layer.input_layernorm.clone(), device)?.into_record();
    record.q_proj = load_linear(store, &format!("{p}.self_attn.q_proj"), layer.q_proj.clone(), device)?.into_record();
    record.k_proj = load_linear(store, &format!("{p}.self_attn.k_proj"), layer.k_proj.clone(), device)?.into_record();
    record.v_proj = load_linear(store, &format!("{p}.self_attn.v_proj"), layer.v_proj.clone(), device)?.into_record();
    record.o_proj = load_linear(store, &format!("{p}.self_attn.o_proj"), layer.o_proj.clone(), device)?.into_record();
    record.post_attention_layernorm = load_ln(store, &format!("{p}.post_attention_layernorm"), layer.post_attention_layernorm.clone(), device)?.into_record();
    record.fc1 = load_linear(store, &format!("{p}.mlp.fc1"), layer.fc1.clone(), device)?.into_record();
    record.fc2 = load_linear(store, &format!("{p}.mlp.fc2"), layer.fc2.clone(), device)?.into_record();
    Ok(layer.clone().load_record(record))
}

fn load_decoder_layer<B: Backend>(
    store: &Store,
    i: usize,
    layer: &MoonshineDecoderLayer<B>,
    device: &B::Device,
) -> Result<MoonshineDecoderLayer<B>, Box<dyn std::error::Error>> {
    let p = format!("model.decoder.layers.{i}");
    let mut record = layer.clone().into_record();
    record.input_layernorm = load_ln(store, &format!("{p}.input_layernorm"), layer.input_layernorm.clone(), device)?.into_record();
    record.self_q = load_linear(store, &format!("{p}.self_attn.q_proj"), layer.self_q.clone(), device)?.into_record();
    record.self_k = load_linear(store, &format!("{p}.self_attn.k_proj"), layer.self_k.clone(), device)?.into_record();
    record.self_v = load_linear(store, &format!("{p}.self_attn.v_proj"), layer.self_v.clone(), device)?.into_record();
    record.self_o = load_linear(store, &format!("{p}.self_attn.o_proj"), layer.self_o.clone(), device)?.into_record();
    record.post_attention_layernorm = load_ln(store, &format!("{p}.post_attention_layernorm"), layer.post_attention_layernorm.clone(), device)?.into_record();
    record.cross_q = load_linear(store, &format!("{p}.encoder_attn.q_proj"), layer.cross_q.clone(), device)?.into_record();
    record.cross_k = load_linear(store, &format!("{p}.encoder_attn.k_proj"), layer.cross_k.clone(), device)?.into_record();
    record.cross_v = load_linear(store, &format!("{p}.encoder_attn.v_proj"), layer.cross_v.clone(), device)?.into_record();
    record.cross_o = load_linear(store, &format!("{p}.encoder_attn.o_proj"), layer.cross_o.clone(), device)?.into_record();
    record.final_layernorm = load_ln(store, &format!("{p}.final_layernorm"), layer.final_layernorm.clone(), device)?.into_record();
    record.fc1 = load_linear(store, &format!("{p}.mlp.fc1"), layer.fc1.clone(), device)?.into_record();
    record.fc2 = load_linear(store, &format!("{p}.mlp.fc2"), layer.fc2.clone(), device)?.into_record();
    Ok(layer.clone().load_record(record))
}

fn load_linear<B: Backend>(
    store: &Store,
    prefix: &str,
    linear: Linear<B>,
    device: &B::Device,
) -> Result<Linear<B>, Box<dyn std::error::Error>> {
    let (data, shape) = store.tensor_f32(&format!("{prefix}.weight"))?;
    let (weight_data, [d_in, d_out]) = transpose_linear_weight(&data, &shape);
    let weight = Tensor::<B, 1>::from_floats(weight_data.as_slice(), device).reshape([d_in, d_out]);
    let bias = store
        .tensor_f32(&format!("{prefix}.bias"))
        .ok()
        .map(|(b, _)| Tensor::<B, 1>::from_floats(b.as_slice(), device));
    let mut record = linear.clone().into_record();
    record.weight = Param::from_tensor(weight);
    record.bias = bias.map(Param::from_tensor);
    Ok(linear.load_record(record))
}

fn load_conv1d<B: Backend>(
    store: &Store,
    prefix: &str,
    conv: Conv1d<B>,
    device: &B::Device,
) -> Result<Conv1d<B>, Box<dyn std::error::Error>> {
    let (data, shape) = store.tensor_f32(&format!("{prefix}.weight"))?;
    let shape3: [usize; 3] = shape.try_into().map_err(|_| "conv1d weight must be 3D")?;
    let weight = Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape(shape3);
    let bias = store
        .tensor_f32(&format!("{prefix}.bias"))
        .ok()
        .map(|(b, _)| Tensor::<B, 1>::from_floats(b.as_slice(), device));
    let mut record = conv.clone().into_record();
    record.weight = Param::from_tensor(weight);
    record.bias = bias.map(Param::from_tensor);
    Ok(conv.load_record(record))
}

fn load_ln<B: Backend>(
    store: &Store,
    prefix: &str,
    ln: LayerNorm<B>,
    device: &B::Device,
) -> Result<LayerNorm<B>, Box<dyn std::error::Error>> {
    let (w, _) = store.tensor_f32(&format!("{prefix}.weight"))?;
    let weight = Tensor::<B, 1>::from_floats(w.as_slice(), device);
    let bias = match store.tensor_f32(&format!("{prefix}.bias")) {
        Ok((b, _)) => Tensor::<B, 1>::from_floats(b.as_slice(), device),
        Err(_) => Tensor::<B, 1>::zeros([weight.dims()[0]], device),
    };
    let mut record = ln.clone().into_record();
    record.gamma = Param::from_tensor(weight);
    record.beta = Param::from_tensor(bias);
    Ok(ln.load_record(record))
}

fn load_embed<B: Backend>(
    store: &Store,
    prefix: &str,
    embed: Embedding<B>,
    device: &B::Device,
) -> Result<Embedding<B>, Box<dyn std::error::Error>> {
    let (data, shape) = store.tensor_f32(&format!("{prefix}.weight"))?;
    let weight = Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([shape[0], shape[1]]);
    let mut record = embed.clone().into_record();
    record.weight = Param::from_tensor(weight);
    Ok(embed.load_record(record))
}
