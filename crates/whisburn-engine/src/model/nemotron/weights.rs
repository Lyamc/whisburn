//! Load the NVIDIA safetensors checkpoint into the Burn Nemotron graph.
//! Linear weights are stored `[out, in]` and transposed to Burn `[in, out]`.
//! LSTM weights stay `[4H, in]`; the step transposes them at the matmul.

use burn::tensor::{Device, DeviceKind};
use std::fs::{self, File};
use std::path::Path;

use burn::module::{Module, Param};
use memmap2::Mmap;
use burn::nn::conv::{Conv1d, Conv2d};
use burn::nn::{LayerNorm, Linear};
use burn::tensor::{Tensor};
use safetensors::SafeTensors;
use serde::Deserialize;

use super::{Nemotron, NemotronConfig};
use crate::model::vibevoice::weights::dtype::{bytes_to_f32, transpose_linear_weight};

#[derive(Debug, Deserialize)]
struct HfFile {
    blank_token_id: usize,
    decoder_hidden_size: usize,
    vocab_size: usize,
    num_decoder_layers: usize,
    num_prompts: usize,
    prompt_intermediate_size: usize,
    max_symbols_per_step: usize,
    encoder_config: HfEncoder,
}

#[derive(Debug, Deserialize)]
struct HfEncoder {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    conv_kernel_size: usize,
    num_mel_bins: usize,
    subsampling_conv_channels: usize,
    subsampling_conv_kernel_size: usize,
    subsampling_conv_stride: usize,
    subsampling_factor: usize,
    sliding_window: usize,
    default_num_lookahead_tokens: usize,
}

pub fn read_nemotron_config(model_dir: &Path) -> Result<NemotronConfig, String> {
    let path = model_dir.join("config.json");
    let text = fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let hf: HfFile = serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
    let mut factor = hf.encoder_config.subsampling_factor.max(1);
    let mut stages = 0usize;
    while factor > 1 {
        factor /= 2;
        stages += 1;
    }
    Ok(NemotronConfig {
        hidden: hf.encoder_config.hidden_size,
        intermediate: hf.encoder_config.intermediate_size,
        n_layers: hf.encoder_config.num_hidden_layers,
        n_heads: hf.encoder_config.num_attention_heads,
        mel_bins: hf.encoder_config.num_mel_bins,
        conv_kernel: hf.encoder_config.conv_kernel_size,
        sub_channels: hf.encoder_config.subsampling_conv_channels,
        sub_kernel: hf.encoder_config.subsampling_conv_kernel_size,
        sub_stride: hf.encoder_config.subsampling_conv_stride,
        sub_stages: stages.max(1),
        decoder_hidden: hf.decoder_hidden_size,
        decoder_layers: hf.num_decoder_layers,
        vocab: hf.vocab_size,
        blank_id: hf.blank_token_id,
        num_prompts: hf.num_prompts,
        prompt_hidden: hf.prompt_intermediate_size,
        max_symbols: hf.max_symbols_per_step,
        sliding_window: hf.encoder_config.sliding_window,
        lookahead: hf.encoder_config.default_num_lookahead_tokens,
    })
}

pub fn load_nemotron_weights(
    model: &mut Nemotron,
    model_dir: &Path,
    device: &Device,
) -> Result<(), String> {
    let path = model_dir.join("model.safetensors");
    // The file is ~2.5 GB fp32. Reading it into a Vec while the zero-initialized
    // module is still alive is a second anonymous copy, and that is what pushed
    // a model switch over earlyoom.
    let file = File::open(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let bytes = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap {}: {e}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes).map_err(|e| format!("parse {}: {e}", path.display()))?;

    model.conv_in = fill_conv2d(model.conv_in.clone(), &st, "encoder.subsampling.conv_in", device)?;
    for (i, stage) in model.sub_layers.iter_mut().enumerate() {
        let prefix = format!("encoder.subsampling.layers.{i}");
        stage.depthwise = fill_conv2d(stage.depthwise.clone(), &st, &format!("{prefix}.depthwise_conv"), device)?;
        stage.pointwise = fill_conv2d(stage.pointwise.clone(), &st, &format!("{prefix}.pointwise_conv"), device)?;
    }
    model.sub_linear = fill_linear(model.sub_linear.clone(), &st, "encoder.subsampling.linear", true, device)?;

    for (i, block) in model.blocks.iter_mut().enumerate() {
        let p = format!("encoder.layers.{i}");
        block.norm_ff1 = fill_ln(block.norm_ff1.clone(), &st, &format!("{p}.norm_feed_forward1"), device)?;
        block.ff1.lin1 = fill_linear(block.ff1.lin1.clone(), &st, &format!("{p}.feed_forward1.linear1"), false, device)?;
        block.ff1.lin2 = fill_linear(block.ff1.lin2.clone(), &st, &format!("{p}.feed_forward1.linear2"), false, device)?;
        block.norm_attn = fill_ln(block.norm_attn.clone(), &st, &format!("{p}.norm_self_att"), device)?;
        block.attn.query = fill_linear(block.attn.query.clone(), &st, &format!("{p}.self_attn.q_proj"), false, device)?;
        block.attn.key = fill_linear(block.attn.key.clone(), &st, &format!("{p}.self_attn.k_proj"), false, device)?;
        block.attn.value = fill_linear(block.attn.value.clone(), &st, &format!("{p}.self_attn.v_proj"), false, device)?;
        block.attn.out = fill_linear(block.attn.out.clone(), &st, &format!("{p}.self_attn.o_proj"), false, device)?;
        block.attn.pos = fill_linear(block.attn.pos.clone(), &st, &format!("{p}.self_attn.relative_k_proj"), false, device)?;
        block.attn.pos_bias_u = param2(&st, &format!("{p}.self_attn.bias_u"), device)?;
        block.attn.pos_bias_v = param2(&st, &format!("{p}.self_attn.bias_v"), device)?;
        block.norm_conv = fill_ln(block.norm_conv.clone(), &st, &format!("{p}.norm_conv"), device)?;
        block.conv.pointwise1 = fill_conv1d(block.conv.pointwise1.clone(), &st, &format!("{p}.conv.pointwise_conv1"), device)?;
        block.conv.depthwise = fill_conv1d(block.conv.depthwise.clone(), &st, &format!("{p}.conv.depthwise_conv"), device)?;
        block.conv.norm = fill_ln(block.conv.norm.clone(), &st, &format!("{p}.conv.norm"), device)?;
        block.conv.pointwise2 = fill_conv1d(block.conv.pointwise2.clone(), &st, &format!("{p}.conv.pointwise_conv2"), device)?;
        block.norm_ff2 = fill_ln(block.norm_ff2.clone(), &st, &format!("{p}.norm_feed_forward2"), device)?;
        block.ff2.lin1 = fill_linear(block.ff2.lin1.clone(), &st, &format!("{p}.feed_forward2.linear1"), false, device)?;
        block.ff2.lin2 = fill_linear(block.ff2.lin2.clone(), &st, &format!("{p}.feed_forward2.linear2"), false, device)?;
        block.norm_out = fill_ln(block.norm_out.clone(), &st, &format!("{p}.norm_out"), device)?;
    }

    model.prompt_1 = fill_linear(model.prompt_1.clone(), &st, "prompt_projector.linear_1", true, device)?;
    model.prompt_2 = fill_linear(model.prompt_2.clone(), &st, "prompt_projector.linear_2", true, device)?;
    model.encoder_projector = fill_linear(model.encoder_projector.clone(), &st, "encoder_projector", true, device)?;
    model.joint = fill_linear(model.joint.clone(), &st, "joint.head", true, device)?;

    let (embed, shape) = tensor_f32(&st, "decoder.embedding.weight")?;
    if shape.len() != 2 {
        return Err(format!("decoder.embedding.weight rank {}", shape.len()));
    }
    let weight = Tensor::< 1>::from_floats(embed.as_slice(), device).reshape([shape[0], shape[1]]);
    drop(embed);
    let mut loaded = model.decoder.embedding.clone();
    loaded.weight = Param::from_tensor(weight);
    model.decoder.embedding = loaded;

    for (i, layer) in model.decoder.lstm.layers.iter_mut().enumerate() {
        layer.weight_ih = param2(&st, &format!("decoder.lstm.weight_ih_l{i}"), device)?;
        layer.weight_hh = param2(&st, &format!("decoder.lstm.weight_hh_l{i}"), device)?;
        layer.bias_ih = param1(&st, &format!("decoder.lstm.bias_ih_l{i}"), device)?;
        layer.bias_hh = param1(&st, &format!("decoder.lstm.bias_hh_l{i}"), device)?;
    }
    model.decoder.projector = fill_linear(
        model.decoder.projector.clone(),
        &st,
        "decoder.decoder_projector",
        true,
        device,
    )?;
    Ok(())
}

fn tensor_f32(st: &SafeTensors<'_>, key: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
    let tensor = st.tensor(key).map_err(|e| format!("{key}: {e}"))?;
    let data = bytes_to_f32(tensor.data(), tensor.dtype()).map_err(|e| format!("{key}: {e}"))?;
    let shape = tensor.shape().to_vec();
    let expect: usize = shape.iter().product();
    if data.len() != expect {
        return Err(format!("{key}: {} floats for shape {shape:?}", data.len()));
    }
    Ok((data, shape))
}

fn fill_linear(
    linear: Linear,
    st: &SafeTensors<'_>,
    prefix: &str,
    with_bias: bool,
    device: &Device,
) -> Result<Linear, String> {
    let (data, shape) = tensor_f32(st, &format!("{prefix}.weight"))?;
    if shape.len() != 2 {
        return Err(format!("{prefix}.weight rank {}", shape.len()));
    }
    let (weight_data, [d_in, d_out]) = transpose_linear_weight(&data, &shape);
    drop(data);
    let weight = Tensor::< 1>::from_floats(weight_data.as_slice(), device).reshape([d_in, d_out]);
    drop(weight_data);
    let mut loaded = linear.clone();
    loaded.weight = Param::from_tensor(weight);
    if with_bias {
        let (bias, _) = tensor_f32(st, &format!("{prefix}.bias"))?;
        loaded.bias = Some(Param::from_tensor(Tensor::< 1>::from_floats(bias.as_slice(), device)));
    }
    Ok(loaded)
}

fn fill_conv2d(
    conv: Conv2d,
    st: &SafeTensors<'_>,
    prefix: &str,
    device: &Device,
) -> Result<Conv2d, String> {
    let (data, shape) = tensor_f32(st, &format!("{prefix}.weight"))?;
    if shape.len() != 4 {
        return Err(format!("{prefix}.weight rank {}", shape.len()));
    }
    let shape4: [usize; 4] = shape
        .try_into()
        .map_err(|_| format!("{prefix}.weight is not 4D"))?;
    let weight = Tensor::< 1>::from_floats(data.as_slice(), device).reshape(shape4);
    drop(data);
    let mut loaded = conv.clone();
    loaded.weight = Param::from_tensor(weight);
    if loaded.bias.is_some() {
        let (bias, _) = tensor_f32(st, &format!("{prefix}.bias"))?;
        loaded.bias = Some(Param::from_tensor(Tensor::< 1>::from_floats(bias.as_slice(), device)));
    }
    Ok(loaded)
}

fn fill_conv1d(
    conv: Conv1d,
    st: &SafeTensors<'_>,
    prefix: &str,
    device: &Device,
) -> Result<Conv1d, String> {
    let (data, shape) = tensor_f32(st, &format!("{prefix}.weight"))?;
    if shape.len() != 3 {
        return Err(format!("{prefix}.weight rank {}", shape.len()));
    }
    let shape3: [usize; 3] = shape
        .try_into()
        .map_err(|_| format!("{prefix}.weight is not 3D"))?;
    let weight = Tensor::< 1>::from_floats(data.as_slice(), device).reshape(shape3);
    drop(data);
    let mut loaded = conv.clone();
    loaded.weight = Param::from_tensor(weight);
    Ok(loaded)
}

fn fill_ln(
    ln: LayerNorm,
    st: &SafeTensors<'_>,
    prefix: &str,
    device: &Device,
) -> Result<LayerNorm, String> {
    let (weight, _) = tensor_f32(st, &format!("{prefix}.weight"))?;
    let (bias, _) = tensor_f32(st, &format!("{prefix}.bias"))?;
    let mut loaded = ln.clone();
    loaded.gamma = Param::from_tensor(Tensor::< 1>::from_floats(weight.as_slice(), device));
    loaded.beta = Some(Param::from_tensor(Tensor::< 1>::from_floats(bias.as_slice(), device)));
    Ok(loaded)
}

fn param2(st: &SafeTensors<'_>, key: &str, device: &Device) -> Result<Param<Tensor< 2>>, String> {
    let (data, shape) = tensor_f32(st, key)?;
    if shape.len() != 2 {
        return Err(format!("{key} rank {}", shape.len()));
    }
    let tensor = Tensor::< 1>::from_floats(data.as_slice(), device).reshape([shape[0], shape[1]]);
    Ok(Param::from_tensor(tensor))
}

fn param1(st: &SafeTensors<'_>, key: &str, device: &Device) -> Result<Param<Tensor< 1>>, String> {
    let (data, _) = tensor_f32(st, key)?;
    Ok(Param::from_tensor(Tensor::< 1>::from_floats(data.as_slice(), device)))
}
