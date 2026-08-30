mod bundle;
mod cleanup;
mod fetch;
mod http;
mod options;
mod moonshine;
mod parakeet;
mod qwen3;
mod tone;
mod vibevoice;
mod whisper;

use std::fs;
use std::path::PathBuf;

use o3whisburn_engine::model::registry::find_model;

use crate::convert::{
    is_moonshine_model, is_parakeet_model, is_qwen3_model, is_tone_model, is_vibevoice_model,
    is_whisper_model, needs_reconversion_for,
};
use crate::paths::{has_burn_bundle, model_dir, models_dir};
use crate::sources::resolve_download_source;

pub use cleanup::save_response_gz;
pub use options::DownloadOptions;

use bundle::download_generic_bundle;
use cleanup::clear_stale_artifacts;
use fetch::build_api;
use moonshine::download_moonshine_hf;
use parakeet::download_parakeet_hf;
use qwen3::download_qwen3_hf;
use tone::download_tone_hf;
use vibevoice::download_vibevoice_hf;
use whisper::download_whisper_hf;

pub fn download_model(name: &str, options: &DownloadOptions) -> anyhow::Result<PathBuf> {
    let info = find_model(name)
        .ok_or_else(|| anyhow::anyhow!("unknown model: {name}"))?;

    if !info.burn_ready {
        anyhow::bail!(
            "model '{name}' is registered but Burn backend is not yet implemented"
        );
    }

    let dir = model_dir(name);
    if !options.force && has_burn_bundle(&dir, name) && !needs_reconversion_for(name, &dir) {
        if options.verbose {
            tracing::info!("model '{name}' already present at {}", dir.display());
        }
        return Ok(dir);
    }

    if options.force {
        clear_stale_artifacts(&dir);
    }

    fs::create_dir_all(&dir)?;
    fs::create_dir_all(models_dir())?;

    let source = resolve_download_source(name)
        .ok_or_else(|| anyhow::anyhow!("could not resolve download source for '{name}'"))?;

    if options.verbose {
        tracing::info!(
            "preparing model '{name}' from {} ({:?})",
            source.repo_id(),
            source
        );
    }

    let api = build_api(&options.hf_token)?;

    if is_whisper_model(name) {
        download_whisper_hf(&api, &source, &dir, name, options)?;
    } else if is_parakeet_model(name) {
        download_parakeet_hf(&api, &source, &dir, name, options)?;
    } else if is_vibevoice_model(name) {
        download_vibevoice_hf(&api, &source, &dir, name, options)?;
    } else if is_qwen3_model(name) {
        download_qwen3_hf(&api, &source, &dir, name, options)?;
    } else if is_moonshine_model(name) {
        download_moonshine_hf(&api, &source, &dir, name, options)?;
    } else if is_tone_model(name) {
        download_tone_hf(&api, &source, &dir, name, options)?;
    } else {
        download_generic_bundle(&api, &source, &dir, name, options)?;
    }

    if !has_burn_bundle(&dir, name) {
        anyhow::bail!(
            "failed to prepare burn bundle for '{name}' at {}. Check disk space and HF access.",
            dir.display()
        );
    }

    Ok(dir)
}