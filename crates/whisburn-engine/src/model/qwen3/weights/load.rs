use burn::module::{Module, Param};
use burn::nn::conv::Conv2d;
use burn::nn::{Embedding, LayerNorm, Linear};
use burn::tensor::{backend::Backend, Tensor};
use std::error::Error;
use std::path::Path;

use crate::model::conformer::RMSNorm;
use crate::model::vibevoice::weights::dtype::transpose_linear_weight;
use crate::model::qwen3::audio_tower::Qwen3AudioEncoderLayer;
use crate::model::qwen3::thinker::Qwen3ThinkerLayer;
use crate::model::qwen3::{Qwen3ASR, Qwen3AudioTower, Qwen3Thinker};
use super::Qwen3WeightStore;

pub fn load_qwen3_weights<B: Backend>(
    model: &mut Qwen3ASR<B>,
    model_dir: &Path,
    device: &B::Device,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    if !weights_present(model_dir) {
        if verbose {
            println!("Qwen3: no safetensors weights found");
        }
        return Ok(());
    }

    let store = Qwen3WeightStore::open(model_dir).map_err(|e| format!("{e}"))?;
    let mut store = store;

    model.audio_tower = load_audio_tower(&mut store, &model.audio_tower, device)?;
    model.thinker = load_thinker(&mut store, &model.thinker, device)?;

    if verbose {
        println!("Qwen3: loaded audio_tower + thinker weights from safetensors");
    }
    Ok(())
}

pub fn load_qwen3_lm_weights<B: Backend>(
    thinker: &mut Qwen3Thinker<B>,
    model_dir: &Path,
    device: &B::Device,
    verbose: bool,
) -> Result<(), Box<dyn Error>> {
    let mut store = Qwen3WeightStore::open(model_dir).map_err(|e| format!("{e}"))?;
    if !store.has_key("thinker.model.layers.0.self_attn.q_proj.weight") {
        return Err("Qwen3 text weights not found (expected model.layers.* in safetensors)".into());
    }
    *thinker = load_thinker(&mut store, thinker, device)?;
    if verbose {
        println!("Qwen3 LM: loaded thinker weights from safetensors");
    }
    Ok(())
}

fn weights_present(model_dir: &Path) -> bool {
    super::weights_present(model_dir)
}

fn load_audio_tower<B: Backend>(
    store: &mut Qwen3WeightStore,
    tower: &Qwen3AudioTower<B>,
    device: &B::Device,
) -> Result<Qwen3AudioTower<B>, Box<dyn Error>> {
    let p = "thinker.audio_tower";
    let conv2d1 = load_conv2d(store, &format!("{p}.conv2d1"), tower.conv2d1.clone(), device)?;
    let conv2d2 = load_conv2d(store, &format!("{p}.conv2d2"), tower.conv2d2.clone(), device)?;
    let conv2d3 = load_conv2d(store, &format!("{p}.conv2d3"), tower.conv2d3.clone(), device)?;
    let conv_out = load_linear(store, &format!("{p}.conv_out"), tower.conv_out.clone(), device)?;
    let layers = tower
        .layers
        .iter()
        .enumerate()
        .map(|(i, layer)| load_audio_layer(store, &format!("{p}.layers.{i}"), layer, device))
        .collect::<Result<Vec<_>, _>>()?;
    let ln_post = load_layer_norm(store, &format!("{p}.ln_post"), tower.ln_post.clone(), device)?;
    let proj1 = load_linear(store, &format!("{p}.proj1"), tower.proj1.clone(), device)?;
    let proj2 = load_linear(store, &format!("{p}.proj2"), tower.proj2.clone(), device)?;

    let mut record = tower.clone().into_record();
    record.conv2d1 = conv2d1.into_record();
    record.conv2d2 = conv2d2.into_record();
    record.conv2d3 = conv2d3.into_record();
    record.conv_out = conv_out.into_record();
    record.layers = layers.into_iter().map(|l| l.into_record()).collect();
    record.ln_post = ln_post.into_record();
    record.proj1 = proj1.into_record();
    record.proj2 = proj2.into_record();
    Ok(tower.clone().load_record(record))
}

fn load_audio_layer<B: Backend>(
    store: &mut Qwen3WeightStore,
    prefix: &str,
    layer: &Qwen3AudioEncoderLayer<B>,
    device: &B::Device,
) -> Result<super::super::audio_tower::Qwen3AudioEncoderLayer<B>, Box<dyn Error>> {
    let attn_prefix = format!("{prefix}.self_attn");
    let q_proj = load_linear(store, &format!("{attn_prefix}.q_proj"), layer.self_attn.q_proj.clone(), device)?;
    let k_proj = load_linear(store, &format!("{attn_prefix}.k_proj"), layer.self_attn.k_proj.clone(), device)?;
    let v_proj = load_linear(store, &format!("{attn_prefix}.v_proj"), layer.self_attn.v_proj.clone(), device)?;
    let out_proj = load_linear(store, &format!("{attn_prefix}.out_proj"), layer.self_attn.out_proj.clone(), device)?;
    let self_attn_layer_norm =
        load_layer_norm(store, &format!("{prefix}.self_attn_layer_norm"), layer.self_attn_layer_norm.clone(), device)?;
    let final_layer_norm =
        load_layer_norm(store, &format!("{prefix}.final_layer_norm"), layer.final_layer_norm.clone(), device)?;
    let fc1 = load_linear(store, &format!("{prefix}.fc1"), layer.fc1.clone(), device)?;
    let fc2 = load_linear(store, &format!("{prefix}.fc2"), layer.fc2.clone(), device)?;

    let mut record = layer.clone().into_record();
    record.self_attn.q_proj = q_proj.into_record();
    record.self_attn.k_proj = k_proj.into_record();
    record.self_attn.v_proj = v_proj.into_record();
    record.self_attn.out_proj = out_proj.into_record();
    record.self_attn_layer_norm = self_attn_layer_norm.into_record();
    record.final_layer_norm = final_layer_norm.into_record();
    record.fc1 = fc1.into_record();
    record.fc2 = fc2.into_record();
    Ok(layer.clone().load_record(record))
}

fn load_thinker<B: Backend>(
    store: &mut Qwen3WeightStore,
    thinker: &Qwen3Thinker<B>,
    device: &B::Device,
) -> Result<Qwen3Thinker<B>, Box<dyn Error>> {
    let embed = load_embedding(store, "thinker.model.embed_tokens", thinker.embed_tokens.clone(), device)?;
    let layers = thinker
        .layers
        .iter()
        .enumerate()
        .map(|(i, layer)| load_thinker_layer(store, &format!("thinker.model.layers.{i}"), layer, device))
        .collect::<Result<Vec<_>, _>>()?;
    let norm = load_rms(store, "thinker.model.norm.weight", thinker.norm.clone(), device)?;
    let lm_head = load_linear(store, "thinker.lm_head", thinker.lm_head.clone(), device)?;

    let mut record = thinker.clone().into_record();
    record.embed_tokens = embed.into_record();
    record.layers = layers.into_iter().map(|l| l.into_record()).collect();
    record.norm = norm.into_record();
    record.lm_head = lm_head.into_record();
    Ok(thinker.clone().load_record(record))
}

fn load_thinker_layer<B: Backend>(
    store: &mut Qwen3WeightStore,
    prefix: &str,
    layer: &Qwen3ThinkerLayer<B>,
    device: &B::Device,
) -> Result<super::super::thinker::Qwen3ThinkerLayer<B>, Box<dyn Error>> {
    let attn = format!("{prefix}.self_attn");
    let q_proj = load_linear(store, &format!("{attn}.q_proj"), layer.self_attn.q_proj.clone(), device)?;
    let k_proj = load_linear(store, &format!("{attn}.k_proj"), layer.self_attn.k_proj.clone(), device)?;
    let v_proj = load_linear(store, &format!("{attn}.v_proj"), layer.self_attn.v_proj.clone(), device)?;
    let o_proj = load_linear(store, &format!("{attn}.o_proj"), layer.self_attn.o_proj.clone(), device)?;
    let q_norm = load_rms(store, &format!("{attn}.q_norm.weight"), layer.self_attn.q_norm.clone(), device)?;
    let k_norm = load_rms(store, &format!("{attn}.k_norm.weight"), layer.self_attn.k_norm.clone(), device)?;
    let input_layernorm =
        load_rms(store, &format!("{prefix}.input_layernorm.weight"), layer.input_layernorm.clone(), device)?;
    let post_attention_layernorm = load_rms(
        store,
        &format!("{prefix}.post_attention_layernorm.weight"),
        layer.post_attention_layernorm.clone(),
        device,
    )?;
    let gate_proj = load_linear(store, &format!("{prefix}.mlp.gate_proj"), layer.mlp.gate_proj.clone(), device)?;
    let up_proj = load_linear(store, &format!("{prefix}.mlp.up_proj"), layer.mlp.up_proj.clone(), device)?;
    let down_proj = load_linear(store, &format!("{prefix}.mlp.down_proj"), layer.mlp.down_proj.clone(), device)?;

    let mut record = layer.clone().into_record();
    record.self_attn.q_proj = q_proj.into_record();
    record.self_attn.k_proj = k_proj.into_record();
    record.self_attn.v_proj = v_proj.into_record();
    record.self_attn.o_proj = o_proj.into_record();
    record.self_attn.q_norm = q_norm.into_record();
    record.self_attn.k_norm = k_norm.into_record();
    record.input_layernorm = input_layernorm.into_record();
    record.post_attention_layernorm = post_attention_layernorm.into_record();
    record.mlp.gate_proj = gate_proj.into_record();
    record.mlp.up_proj = up_proj.into_record();
    record.mlp.down_proj = down_proj.into_record();
    Ok(layer.clone().load_record(record))
}

fn load_linear<B: Backend>(
    store: &mut Qwen3WeightStore,
    prefix: &str,
    linear: Linear<B>,
    device: &B::Device,
) -> Result<Linear<B>, Box<dyn Error>> {
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

fn load_conv2d<B: Backend>(
    store: &mut Qwen3WeightStore,
    prefix: &str,
    conv: Conv2d<B>,
    device: &B::Device,
) -> Result<Conv2d<B>, Box<dyn Error>> {
    let (data, shape) = store.tensor_f32(&format!("{prefix}.weight"))?;
    let shape4: [usize; 4] = shape.try_into().map_err(|_| "conv2d weight must be 4D")?;
    let weight = Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape(shape4);
    let bias = store
        .tensor_f32(&format!("{prefix}.bias"))
        .ok()
        .map(|(b, _)| Tensor::<B, 1>::from_floats(b.as_slice(), device));
    let mut record = conv.clone().into_record();
    record.weight = Param::from_tensor(weight);
    record.bias = bias.map(Param::from_tensor);
    Ok(conv.load_record(record))
}

fn load_layer_norm<B: Backend>(
    store: &mut Qwen3WeightStore,
    prefix: &str,
    ln: LayerNorm<B>,
    device: &B::Device,
) -> Result<LayerNorm<B>, Box<dyn Error>> {
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

fn load_rms<B: Backend>(
    store: &mut Qwen3WeightStore,
    key: &str,
    norm: RMSNorm<B>,
    device: &B::Device,
) -> Result<RMSNorm<B>, Box<dyn Error>> {
    let (data, shape) = store.tensor_f32(key)?;
    let dim = shape[0];
    let weight = Tensor::<B, 1>::from_floats(data.as_slice(), device);
    debug_assert_eq!(dim, norm.gamma.dims()[0]);
    let mut record = norm.clone().into_record();
    record.gamma = Param::from_tensor(weight);
    Ok(norm.load_record(record))
}

fn load_embedding<B: Backend>(
    store: &mut Qwen3WeightStore,
    prefix: &str,
    embed: Embedding<B>,
    device: &B::Device,
) -> Result<Embedding<B>, Box<dyn Error>> {
    let (data, shape) = store.tensor_f32(&format!("{prefix}.weight"))?;
    let weight = Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([shape[0], shape[1]]);
    let mut record = embed.clone().into_record();
    record.weight = Param::from_tensor(weight);
    Ok(embed.load_record(record))
}