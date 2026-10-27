//! Marks an FP8 Granite checkpoint as ready. Weights stay compressed-tensors
//! F8_E4M3 and are dequantized per projection at inference time.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::Context;

use crate::download::DownloadOptions;

pub const BURN_BUNDLE_VERSION: &str = "0.22.1-granite-fp8";

pub fn prepare_granite_bundle(
    model_dir: &Path,
    _model_name: &str,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    let weights_path = model_dir.join("model.safetensors");
    let config_path = model_dir.join("config.json");
    if !weights_path.exists() || !config_path.exists() {
        anyhow::bail!(
            "granite bundle needs config.json and model.safetensors in {}",
            model_dir.display()
        );
    }
    let (has_fp8, has_scale, n_tensors) = scan_safetensors_header(&weights_path)?;
    if !has_fp8 || !has_scale {
        anyhow::bail!(
            "granite weights are not compressed-tensors FP8 (missing F8_E4M3 or weight_scale)"
        );
    }
    std::fs::write(model_dir.join(".burn_version"), BURN_BUNDLE_VERSION)?;
    if options.verbose {
        tracing::info!(
            "granite FP8 bundle ready ({n_tensors} tensors) at {}",
            model_dir.display()
        );
    }
    Ok(())
}

fn scan_safetensors_header(path: &Path) -> anyhow::Result<(bool, bool, usize)> {
    let mut file = File::open(path)?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)?;
    let n = u64::from_le_bytes(len_buf) as usize;
    if n > 64 * 1024 * 1024 {
        anyhow::bail!("safetensors header is unexpectedly large ({n} bytes)");
    }
    let mut header = vec![0u8; n];
    file.read_exact(&mut header)?;
    let value: serde_json::Value = serde_json::from_slice(&header).context("safetensors header")?;
    let obj = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("safetensors header is not an object"))?;
    let mut has_fp8 = false;
    let mut has_scale = false;
    let mut n_tensors = 0usize;
    for (name, meta) in obj {
        if name == "__metadata__" {
            continue;
        }
        n_tensors += 1;
        if name.ends_with("self_attn.q_proj.weight_scale") {
            has_scale = true;
        }
        if meta.get("dtype").and_then(|d| d.as_str()) == Some("F8_E4M3") {
            has_fp8 = true;
        }
    }
    Ok((has_fp8, has_scale, n_tensors))
}

pub fn is_granite_model(name: &str) -> bool {
    name == "granite-4.2-3b-fp8"
}

pub fn needs_granite_reconversion(model_dir: &Path) -> bool {
    match std::fs::read_to_string(model_dir.join(".burn_version")) {
        Ok(v) => v.trim() != BURN_BUNDLE_VERSION,
        Err(_) => true,
    }
}
