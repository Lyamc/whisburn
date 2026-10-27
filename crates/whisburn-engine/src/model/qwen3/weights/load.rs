use burn::tensor::{Device, DeviceKind};
use burn::module::{Module, Param};
use burn::nn::conv::Conv2d;
use burn::nn::{Embedding, LayerNorm, Linear};
use burn::tensor::{Tensor};
use std::error::Error;
use std::path::Path;

use crate::model::conformer::RMSNorm;
use crate::model::vibevoice::weights::dtype::transpose_linear_weight;
use crate::model::qwen3::audio_tower::Qwen3AudioEncoderLayer;
use crate::model::qwen3::thinker::Qwen3ThinkerLayer;
use crate::model::qwen3::{Qwen3ASR, Qwen3AudioTower, Qwen3Thinker};
use super::Qwen3WeightStore;

pub fn load_qwen3_weights(
    model: Qwen3ASR,
    model_dir: &Path,
    device: &Device,
    verbose: bool,
) -> Result<Qwen3ASR, Box<dyn Error>> {
    if !weights_present(model_dir) {
        return Err(format!(
            "Qwen3 safetensors missing audio/thinker keys in {}",
            model_dir.display()
        )
        .into());
    }

    let mut store = Qwen3WeightStore::open(model_dir).map_err(|e| format!("{e}"))?;
    let Qwen3ASR {
        audio_tower,
        thinker,
    } = model;
    let thinker = load_thinker(&mut store, thinker, device)?;
    let audio_tower = load_audio_tower(&mut store, audio_tower, device)?;
    trim_allocator();

    if verbose {
        println!("Qwen3: loaded audio_tower + thinker weights from safetensors");
    }
    Ok(Qwen3ASR {
        audio_tower,
        thinker,
    })
}

pub fn load_qwen3_lm_weights(
    thinker: Qwen3Thinker,
    model_dir: &Path,
    device: &Device,
    verbose: bool,
) -> Result<Qwen3Thinker, Box<dyn Error>> {
    let mut store = Qwen3WeightStore::open(model_dir).map_err(|e| format!("{e}"))?;
    if !store.has_key("thinker.model.layers.0.self_attn.q_proj.weight") {
        return Err("Qwen3 text weights not found (expected model.layers.* in safetensors)".into());
    }
    let thinker = load_thinker(&mut store, thinker, device)?;
    trim_allocator();
    if verbose {
        println!("Qwen3 LM: loaded thinker weights from safetensors");
    }
    Ok(thinker)
}

/// Freed Burn zeros sit under the new weights, so glibc will not return them
/// on its own. Trim so the summarizer's high-water mark does not stay resident.
fn trim_allocator() {
    #[cfg(target_os = "linux")]
    unsafe {
        extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        let _ = malloc_trim(0);
    }
}

fn weights_present(model_dir: &Path) -> bool {
    super::weights_present(model_dir)
}

fn load_audio_tower(
    store: &mut Qwen3WeightStore,
    mut tower: Qwen3AudioTower,
    device: &Device,
) -> Result<Qwen3AudioTower, Box<dyn Error>> {
    // Replace one module at a time. Cloning the tower while the new weights
    // already exist kept two full copies and earlyoom killed Qwen ASR loads.
    let p = "thinker.audio_tower";
    let conv2d1 = tower.conv2d1.clone();
    tower.conv2d1 = load_conv2d(store, &format!("{p}.conv2d1"), conv2d1, device)?;
    let conv2d2 = tower.conv2d2.clone();
    tower.conv2d2 = load_conv2d(store, &format!("{p}.conv2d2"), conv2d2, device)?;
    let conv2d3 = tower.conv2d3.clone();
    tower.conv2d3 = load_conv2d(store, &format!("{p}.conv2d3"), conv2d3, device)?;
    let conv_out = tower.conv_out.clone();
    tower.conv_out = load_linear(store, &format!("{p}.conv_out"), conv_out, device)?;

    let old_layers = std::mem::take(&mut tower.layers);
    let mut layers = Vec::with_capacity(old_layers.len());
    for (i, layer) in old_layers.into_iter().enumerate() {
        layers.push(load_audio_layer(
            store,
            &format!("{p}.layers.{i}"),
            layer,
            device,
        )?);
    }
    tower.layers = layers;

    let ln_post = tower.ln_post.clone();
    tower.ln_post = load_layer_norm(store, &format!("{p}.ln_post"), ln_post, device)?;
    let proj1 = tower.proj1.clone();
    tower.proj1 = load_linear(store, &format!("{p}.proj1"), proj1, device)?;
    let proj2 = tower.proj2.clone();
    tower.proj2 = load_linear(store, &format!("{p}.proj2"), proj2, device)?;
    Ok(tower)
}

fn load_audio_layer(
    store: &mut Qwen3WeightStore,
    prefix: &str,
    mut layer: Qwen3AudioEncoderLayer,
    device: &Device,
) -> Result<Qwen3AudioEncoderLayer, Box<dyn Error>> {
    let attn_prefix = format!("{prefix}.self_attn");
    let q_proj = layer.self_attn.q_proj.clone();
    layer.self_attn.q_proj = load_linear(store, &format!("{attn_prefix}.q_proj"), q_proj, device)?;
    let k_proj = layer.self_attn.k_proj.clone();
    layer.self_attn.k_proj = load_linear(store, &format!("{attn_prefix}.k_proj"), k_proj, device)?;
    let v_proj = layer.self_attn.v_proj.clone();
    layer.self_attn.v_proj = load_linear(store, &format!("{attn_prefix}.v_proj"), v_proj, device)?;
    let out_proj = layer.self_attn.out_proj.clone();
    layer.self_attn.out_proj = load_linear(store, &format!("{attn_prefix}.out_proj"), out_proj, device)?;
    let self_attn_layer_norm = layer.self_attn_layer_norm.clone();
    layer.self_attn_layer_norm = load_layer_norm(
        store,
        &format!("{prefix}.self_attn_layer_norm"),
        self_attn_layer_norm,
        device,
    )?;
    let final_layer_norm = layer.final_layer_norm.clone();
    layer.final_layer_norm =
        load_layer_norm(store, &format!("{prefix}.final_layer_norm"), final_layer_norm, device)?;
    let fc1 = layer.fc1.clone();
    layer.fc1 = load_linear(store, &format!("{prefix}.fc1"), fc1, device)?;
    let fc2 = layer.fc2.clone();
    layer.fc2 = load_linear(store, &format!("{prefix}.fc2"), fc2, device)?;
    Ok(layer)
}

fn load_thinker(
    store: &mut Qwen3WeightStore,
    thinker: Qwen3Thinker,
    device: &Device,
) -> Result<Qwen3Thinker, Box<dyn Error>> {
    let mut parts = thinker.into_parts();
    let n_layers = parts.layers.len();
    // init() allocated every layer as zeros. Holding all of them while the
    // 600 MB embedding is copied is what pushed a fresh process over earlyoom.
    let prototype = parts
        .layers
        .pop()
        .ok_or("qwen3 thinker has no layers")?;
    drop(std::mem::take(&mut parts.layers));
    trim_allocator();
    parts.embed_tokens = load_embedding(
        store,
        "thinker.model.embed_tokens",
        parts.embed_tokens,
        device,
    )?;
    trim_allocator();
    parts.norm = load_rms(store, "thinker.model.norm.weight", parts.norm, device)?;
    parts.lm_head = load_linear(store, "thinker.lm_head", parts.lm_head, device)?;
    trim_allocator();
    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        layers.push(load_thinker_layer(
            store,
            &format!("thinker.model.layers.{i}"),
            prototype.clone(),
            device,
        )?);
    }
    drop(prototype);
    parts.layers = layers;
    Ok(Qwen3Thinker::from_parts(parts))
}

fn load_thinker_layer(
    store: &mut Qwen3WeightStore,
    prefix: &str,
    mut layer: Qwen3ThinkerLayer,
    device: &Device,
) -> Result<Qwen3ThinkerLayer, Box<dyn Error>> {
    let attn = format!("{prefix}.self_attn");
    let q_proj = layer.self_attn.q_proj.clone();
    layer.self_attn.q_proj = load_linear(store, &format!("{attn}.q_proj"), q_proj, device)?;
    let k_proj = layer.self_attn.k_proj.clone();
    layer.self_attn.k_proj = load_linear(store, &format!("{attn}.k_proj"), k_proj, device)?;
    let v_proj = layer.self_attn.v_proj.clone();
    layer.self_attn.v_proj = load_linear(store, &format!("{attn}.v_proj"), v_proj, device)?;
    let o_proj = layer.self_attn.o_proj.clone();
    layer.self_attn.o_proj = load_linear(store, &format!("{attn}.o_proj"), o_proj, device)?;
    let q_norm = layer.self_attn.q_norm.clone();
    layer.self_attn.q_norm = load_rms(store, &format!("{attn}.q_norm.weight"), q_norm, device)?;
    let k_norm = layer.self_attn.k_norm.clone();
    layer.self_attn.k_norm = load_rms(store, &format!("{attn}.k_norm.weight"), k_norm, device)?;
    let input_layernorm = layer.input_layernorm.clone();
    layer.input_layernorm = load_rms(
        store,
        &format!("{prefix}.input_layernorm.weight"),
        input_layernorm,
        device,
    )?;
    let post_attention_layernorm = layer.post_attention_layernorm.clone();
    layer.post_attention_layernorm = load_rms(
        store,
        &format!("{prefix}.post_attention_layernorm.weight"),
        post_attention_layernorm,
        device,
    )?;
    let gate_proj = layer.mlp.gate_proj.clone();
    layer.mlp.gate_proj = load_linear(store, &format!("{prefix}.mlp.gate_proj"), gate_proj, device)?;
    let up_proj = layer.mlp.up_proj.clone();
    layer.mlp.up_proj = load_linear(store, &format!("{prefix}.mlp.up_proj"), up_proj, device)?;
    let down_proj = layer.mlp.down_proj.clone();
    layer.mlp.down_proj = load_linear(store, &format!("{prefix}.mlp.down_proj"), down_proj, device)?;
    Ok(layer)
}

fn load_linear(
    store: &mut Qwen3WeightStore,
    prefix: &str,
    linear: Linear,
    device: &Device,
) -> Result<Linear, Box<dyn Error>> {
    let (data, shape) = store.tensor_f32(&format!("{prefix}.weight"))?;
    let (weight_data, [d_in, d_out]) = transpose_linear_weight(&data, &shape);
    // The source and the transpose must not both stay live while Burn copies
    // the matrix. lm_head is ~600 MB, and that overlap tripped earlyoom.
    drop(data);
    let weight = Tensor::< 1>::from_floats(weight_data.as_slice(), device).reshape([d_in, d_out]);
    drop(weight_data);
    let bias = store
        .tensor_f32(&format!("{prefix}.bias"))
        .ok()
        .map(|(b, _)| Tensor::< 1>::from_floats(b.as_slice(), device));
    let mut loaded = linear.clone();
    loaded.weight = Param::from_tensor(weight);
    loaded.bias = bias.map(Param::from_tensor);
    Ok(loaded)
}

fn load_conv2d(
    store: &mut Qwen3WeightStore,
    prefix: &str,
    conv: Conv2d,
    device: &Device,
) -> Result<Conv2d, Box<dyn Error>> {
    let (data, shape) = store.tensor_f32(&format!("{prefix}.weight"))?;
    let shape4: [usize; 4] = shape.try_into().map_err(|_| "conv2d weight must be 4D")?;
    let weight = Tensor::< 1>::from_floats(data.as_slice(), device).reshape(shape4);
    let bias = store
        .tensor_f32(&format!("{prefix}.bias"))
        .ok()
        .map(|(b, _)| Tensor::< 1>::from_floats(b.as_slice(), device));
    let mut loaded = conv.clone();
    loaded.weight = Param::from_tensor(weight);
    loaded.bias = bias.map(Param::from_tensor);
    Ok(loaded)
}

fn load_layer_norm(
    store: &mut Qwen3WeightStore,
    prefix: &str,
    ln: LayerNorm,
    device: &Device,
) -> Result<LayerNorm, Box<dyn Error>> {
    let (w, _) = store.tensor_f32(&format!("{prefix}.weight"))?;
    let weight = Tensor::< 1>::from_floats(w.as_slice(), device);
    let bias = match store.tensor_f32(&format!("{prefix}.bias")) {
        Ok((b, _)) => Tensor::< 1>::from_floats(b.as_slice(), device),
        Err(_) => Tensor::< 1>::zeros([weight.dims()[0]], device),
    };
    let mut loaded = ln.clone();
    loaded.gamma = Param::from_tensor(weight);
    loaded.beta = Some(Param::from_tensor(bias));
    Ok(loaded)
}

fn load_rms(
    store: &mut Qwen3WeightStore,
    key: &str,
    norm: RMSNorm,
    device: &Device,
) -> Result<RMSNorm, Box<dyn Error>> {
    let (data, shape) = store.tensor_f32(key)?;
    let dim = shape[0];
    let weight = Tensor::< 1>::from_floats(data.as_slice(), device);
    debug_assert_eq!(dim, norm.gamma.dims()[0]);
    let mut loaded = norm.clone();
    loaded.gamma = Param::from_tensor(weight);
    Ok(loaded)
}

fn load_embedding(
    store: &mut Qwen3WeightStore,
    prefix: &str,
    embed: Embedding,
    device: &Device,
) -> Result<Embedding, Box<dyn Error>> {
    let (data, shape) = store.tensor_f32(&format!("{prefix}.weight"))?;
    let weight = Tensor::< 1>::from_floats(data.as_slice(), device).reshape([shape[0], shape[1]]);
    drop(data);
    let mut loaded = embed.clone();
    loaded.weight = Param::from_tensor(weight);
    Ok(loaded)
}