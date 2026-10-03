//! Reader for `fermion-five-value-parakeet-v1` containers (Phonon-2).
//!
//! The container expands to the same Hugging Face `ParakeetForTDT` names as
//! `nvidia/parakeet-tdt-0.6b-v3`. Pointwise conv kernels stored as `[O, I]` are
//! unsqueezed to `[O, I, 1]` before the existing Parakeet Burn mapping.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::Context;
use half::f16;

use super::parakeet::cleanup::{cleanup_parakeet_npy_artifacts, write_attn_heads};
use super::parakeet::config::ParakeetConfig;
use super::parakeet::layout::layout_tensor;
use super::parakeet::mapping::{map_hf_tensor, should_skip_key};
use super::parakeet::BURN_VERSION;
use crate::convert::npy::NpyDump;
use crate::download::DownloadOptions;

const FORMAT: &str = "fermion-five-value-parakeet-v1";

#[derive(Debug, serde::Deserialize)]
struct Header {
    format: String,
    index: Vec<Record>,
}

#[derive(Debug, serde::Deserialize)]
struct Record {
    k: String,
    b: usize,
    n: String,
    shape: Vec<usize>,
}

pub fn convert_phonon_container(
    container: &Path,
    model_dir: &Path,
    model_name: &str,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    let config_path = model_dir.join("config.json");
    let config: ParakeetConfig = serde_json::from_str(
        &std::fs::read_to_string(&config_path).context("read Parakeet config.json")?,
    )?;
    let arch = config
        .architectures
        .first()
        .map(String::as_str)
        .unwrap_or("ParakeetForTDT");
    let is_tdt = arch.contains("TDT") || arch.contains("RNNT");

    let mut file = File::open(container)
        .with_context(|| format!("open {}", container.display()))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf);
    let mut header_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut header_bytes)?;
    let header: Header = serde_json::from_slice(&header_bytes).context("phonon container header")?;
    if header.format != FORMAT {
        anyhow::bail!(
            "unsupported phonon container format '{}' (expected {FORMAT})",
            header.format
        );
    }

    let enc = &config.encoder_config;
    let mut dump = NpyDump::new(model_dir);
    dump.write_scalar("encoder/n_layers.npy", enc.num_hidden_layers as f32)?;
    dump.write_scalar("encoder/d_model.npy", enc.hidden_size as f32)?;
    dump.write_scalar("encoder/n_head.npy", enc.num_attention_heads as f32)?;
    dump.write_scalar("encoder/n_mels.npy", enc.num_mel_bins as f32)?;

    let num_durations = config.durations.as_ref().map(|d| d.len()).unwrap_or(0);
    let joint_out = config.vocab_size + num_durations;
    dump.write_scalar("actual_n_vocab.npy", joint_out as f32)?;
    let decode_cfg = super::parakeet::ParakeetDecodeConfig {
        blank_token_id: config.blank_id(),
        vocab_size: config.vocab_size,
        duration_start: config.vocab_size,
        num_durations,
        pad_token_id: config.pad_token_id,
    };
    std::fs::write(
        model_dir.join("parakeet_decode.json"),
        serde_json::to_string_pretty(&decode_cfg)?,
    )?;

    let mut written = 0usize;
    let mut unmapped = Vec::new();
    for record in &header.index {
        let mut blob = vec![0u8; record.b];
        file.read_exact(&mut blob)
            .with_context(|| format!("read phonon record {}", record.n))?;
        let (key, data, shape) = expand_record(record, &blob)?;
        if key.ends_with("num_batches_tracked") || should_skip_key(&key) {
            continue;
        }
        let (data, shape) = unsqueeze_pointwise(&key, data, shape);
        if let Some(rel) = map_hf_tensor(&key, is_tdt) {
            let (data, shape) = layout_tensor(&rel, &data, &shape);
            dump.write_f32(&rel, &data, &shape)?;
            written += 1;
        } else {
            unmapped.push(key);
        }
    }
    let consumed = 8 + header_len as u64 + header.index.iter().map(|r| r.b as u64).sum::<u64>();
    let file_len = file.seek(SeekFrom::End(0))?;
    if file_len != consumed {
        anyhow::bail!(
            "phonon container has trailing or truncated bytes (file {file_len}, records {consumed})"
        );
    }

    write_attn_heads(&mut dump, enc.num_hidden_layers, enc.num_attention_heads)?;
    dump.flush_shapes()?;

    if !unmapped.is_empty() {
        anyhow::bail!(
            "phonon container has {} tensors that do not map onto Parakeet (first: {})",
            unmapped.len(),
            unmapped[0]
        );
    }
    if written == 0 {
        anyhow::bail!("phonon container produced no Parakeet tensors");
    }
    if options.verbose {
        tracing::info!("mapped {written} phonon tensors onto Parakeet");
    }

    super::burn_mpk::save_model_from_npy(model_name, options)?;
    cleanup_parakeet_npy_artifacts(model_dir, options.verbose)?;
    std::fs::write(model_dir.join(".burn_version"), BURN_VERSION)?;
    Ok(())
}

fn unsqueeze_pointwise(key: &str, mut data: Vec<f32>, mut shape: Vec<usize>) -> (Vec<f32>, Vec<usize>) {
    let pointwise = key.ends_with(".conv.pointwise_conv1.weight")
        || key.ends_with(".conv.pointwise_conv2.weight");
    if pointwise && shape.len() == 2 {
        shape.push(1);
        let _ = &mut data;
    }
    (data, shape)
}

fn expand_record(record: &Record, blob: &[u8]) -> anyhow::Result<(String, Vec<f32>, Vec<usize>)> {
    if blob.len() != record.b {
        anyhow::bail!("record {} length {} != {}", record.n, blob.len(), record.b);
    }
    match record.k.as_str() {
        "five_value" => {
            let data = expand_five_value(blob, &record.shape)?;
            Ok((format!("{}.weight", record.n), data, record.shape.clone()))
        }
        "fp16" => {
            let data = f16_blob(blob)?;
            let expect: usize = record.shape.iter().product();
            if data.len() != expect {
                anyhow::bail!(
                    "fp16 record {} has {} values, shape {:?} wants {expect}",
                    record.n,
                    data.len(),
                    record.shape
                );
            }
            Ok((record.n.clone(), data, record.shape.clone()))
        }
        other if other.starts_with("int") => {
            let bits: usize = other[3..]
                .parse()
                .with_context(|| format!("phonon quant kind {other}"))?;
            let data = expand_intn(blob, &record.shape, bits)?;
            Ok((record.n.clone(), data, record.shape.clone()))
        }
        other => anyhow::bail!("unknown phonon record kind {other}"),
    }
}

pub(crate) fn expand_five_value(blob: &[u8], shape: &[usize]) -> anyhow::Result<Vec<f32>> {
    if shape.len() != 2 {
        anyhow::bail!("five_value shape must be [out, in], got {shape:?}");
    }
    let (out, inp) = (shape[0], shape[1]);
    let rb = inp.div_ceil(5);
    let code_bytes = out * rb;
    if blob.len() < code_bytes {
        anyhow::bail!("five_value blob shorter than trit plane");
    }
    let codes = unpack_trits(&blob[..code_bytes], out, inp, rb);
    let mut nz_idx = Vec::new();
    for (idx, code) in codes.iter().copied().enumerate() {
        if code != 1 {
            nz_idx.push(idx);
        }
    }
    let nnz = nz_idx.len();
    let rbytes = nnz.div_ceil(8);
    let mut off = code_bytes;
    if blob.len() < off + rbytes + 4 * out {
        anyhow::bail!("five_value blob missing magnitude planes");
    }
    let mut is_hi = vec![false; out * inp];
    for (n, &idx) in nz_idx.iter().enumerate() {
        let bit = (blob[off + n / 8] >> (n % 8)) & 1;
        is_hi[idx] = bit == 1;
    }
    off += rbytes;
    let lo = f16_slice(&blob[off..off + 2 * out])?;
    let hi = f16_slice(&blob[off + 2 * out..off + 4 * out])?;
    if off + 4 * out != blob.len() {
        anyhow::bail!(
            "five_value trailing bytes: consumed {}, blob {}",
            off + 4 * out,
            blob.len()
        );
    }
    let mut out_w = vec![0.0f32; out * inp];
    for o in 0..out {
        for i in 0..inp {
            let idx = o * inp + i;
            let sign = codes[idx] as i8 - 1;
            let mag = if is_hi[idx] { hi[o] } else { lo[o] };
            out_w[idx] = f32::from(sign) * mag;
        }
    }
    Ok(out_w)
}

fn unpack_trits(buf: &[u8], out: usize, inp: usize, rb: usize) -> Vec<u8> {
    let mut codes = vec![0u8; out * inp];
    for o in 0..out {
        for r in 0..rb {
            let mut x = u16::from(buf[o * rb + r]);
            for k in 0..5 {
                let col = r * 5 + k;
                if col >= inp {
                    break;
                }
                codes[o * inp + col] = (x % 3) as u8;
                x /= 3;
            }
        }
    }
    codes
}

pub(crate) fn expand_intn(blob: &[u8], shape: &[usize], bits: usize) -> anyhow::Result<Vec<f32>> {
    if shape.is_empty() {
        anyhow::bail!("int{bits} record missing shape");
    }
    let out = shape[0];
    let total: usize = shape.iter().product();
    if out == 0 || total % out != 0 {
        anyhow::bail!("int{bits} shape {shape:?} is not divisible by the leading dim");
    }
    let scale_bytes = 2 * out;
    if blob.len() < scale_bytes {
        anyhow::bail!("int{bits} blob missing per-row scales");
    }
    let (body, scale_blob) = blob.split_at(blob.len() - scale_bytes);
    let scales = f16_slice(scale_blob)?;
    let q = match bits {
        8 => {
            if body.len() < total {
                anyhow::bail!("int8 body is short");
            }
            body[..total].iter().map(|&b| b as i8 as i32).collect::<Vec<_>>()
        }
        6 => unpack_int6(body, total)?,
        other => anyhow::bail!("unsupported phonon int width {other}"),
    };
    let cols = total / out;
    let mut data = vec![0.0f32; total];
    for o in 0..out {
        let scale = scales[o];
        for c in 0..cols {
            data[o * cols + c] = q[o * cols + c] as f32 * scale;
        }
    }
    Ok(data)
}

fn unpack_int6(body: &[u8], total: usize) -> anyhow::Result<Vec<i32>> {
    if body.len() % 3 != 0 {
        anyhow::bail!("int6 body length {} is not a multiple of 3", body.len());
    }
    let mut q = Vec::with_capacity(total);
    for chunk in body.chunks_exact(3) {
        let packed = u32::from(chunk[0]) | (u32::from(chunk[1]) << 8) | (u32::from(chunk[2]) << 16);
        for shift in [0, 6, 12, 18] {
            if q.len() == total {
                break;
            }
            let u = (packed >> shift) & 0x3F;
            q.push(u as i32 - 32);
        }
    }
    if q.len() < total {
        anyhow::bail!("int6 body produced {} values, need {total}", q.len());
    }
    q.truncate(total);
    Ok(q)
}

fn f16_blob(blob: &[u8]) -> anyhow::Result<Vec<f32>> {
    f16_slice(blob)
}

/// Expand the container and count tensors that land on the Parakeet Burn graph.
pub fn probe_mapping(container: &Path, is_tdt: bool) -> anyhow::Result<(usize, Vec<String>)> {
    let mut file = File::open(container)?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf);
    let mut header_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut header_bytes)?;
    let header: Header = serde_json::from_slice(&header_bytes)?;
    if header.format != FORMAT {
        anyhow::bail!("unsupported phonon container format '{}'", header.format);
    }
    let mut written = 0usize;
    let mut unmapped = Vec::new();
    for record in &header.index {
        let mut blob = vec![0u8; record.b];
        file.read_exact(&mut blob)?;
        let (key, data, shape) = expand_record(record, &blob)?;
        if key.ends_with("num_batches_tracked") || should_skip_key(&key) {
            continue;
        }
        let (data, shape) = unsqueeze_pointwise(&key, data, shape);
        if let Some(rel) = map_hf_tensor(&key, is_tdt) {
            let _ = layout_tensor(&rel, &data, &shape);
            written += 1;
        } else {
            unmapped.push(key);
        }
    }
    Ok((written, unmapped))
}

pub fn is_phonon_model(name: &str) -> bool {
    name == "phonon-2"
}

pub fn needs_phonon_reconversion(model_dir: &Path) -> bool {
    match std::fs::read_to_string(model_dir.join(".burn_version")) {
        Ok(v) => v.trim() != BURN_VERSION,
        Err(_) => true,
    }
}

fn f16_slice(blob: &[u8]) -> anyhow::Result<Vec<f32>> {
    if blob.len() % 2 != 0 {
        anyhow::bail!("fp16 blob length {} is odd", blob.len());
    }
    Ok(blob
        .chunks_exact(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::{expand_five_value, expand_intn};

    #[test]
    fn five_value_uses_lo_plane_when_bit_is_clear() {
        // out=1, in=1, code 2 => sign +1, bit 0 => lo magnitude 1.0
        let blob = vec![2u8, 0, 0x00, 0x3c, 0x00, 0x40];
        let w = expand_five_value(&blob, &[1, 1]).unwrap();
        assert!((w[0] - 1.0).abs() < 1e-6, "{w:?}");
    }

    #[test]
    fn five_value_zero_code_is_exact_zero() {
        // code 1 => sign 0, even if magnitudes are nonzero
        let blob = vec![1u8, 0x00, 0x3c, 0x00, 0x40];
        let w = expand_five_value(&blob, &[1, 1]).unwrap();
        assert_eq!(w, vec![0.0]);
    }

    #[test]
    fn int6_unpacks_four_values_and_row_scale() {
        // unsigned nibbles 32, 33, 31, 48 packed little-endian in 24 bits, scale 1.0
        let packed: u32 = 32 | (33 << 6) | (31 << 12) | (48 << 18);
        let mut blob = vec![
            (packed & 0xff) as u8,
            ((packed >> 8) & 0xff) as u8,
            ((packed >> 16) & 0xff) as u8,
        ];
        blob.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        let w = expand_intn(&blob, &[1, 4], 6).unwrap();
        let expect = [0.0, 1.0, -1.0, 16.0];
        for (got, exp) in w.iter().zip(expect) {
            assert!((got - exp).abs() < 1e-5, "{w:?}");
        }
    }

    use half::f16;

    #[test]
    fn real_container_maps_onto_parakeet_when_present() {
        let Ok(path) = std::env::var("PHONON_CONTAINER") else {
            return;
        };
        let (written, unmapped) = super::probe_mapping(std::path::Path::new(&path), true)
            .unwrap_or_else(|e| panic!("{e}"));
        assert!(written > 600, "mapped {written}");
        assert!(
            unmapped.is_empty(),
            "unmapped {}: {}",
            unmapped.len(),
            unmapped.iter().take(8).cloned().collect::<Vec<_>>().join(", ")
        );
    }
}
