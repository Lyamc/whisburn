use std::fs;
use std::path::Path;

use hf_hub::api::sync::{Api, ApiRepo};

use crate::convert::save_model_from_npy;
use crate::sources::DownloadSource;

use super::cleanup::decompress_mpk_gz_variants;
use super::fetch::fetch_repo_file;
use super::http::fetch_hf_http;
use super::options::DownloadOptions;

pub fn download_generic_bundle(
    api: &Api,
    source: &DownloadSource,
    dir: &Path,
    name: &str,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    let repo = api.model(source.repo_id().to_string());
    let bundle_candidates = [
        format!("{name}/config.cfg"),
        format!("{name}/tokenizer.json"),
        format!("{name}/model.mpk"),
        format!("{name}/model.mpk.gz"),
        "config.cfg".to_string(),
        "tokenizer.json".to_string(),
        "model.mpk".to_string(),
        "model.mpk.gz".to_string(),
    ];

    let fetched = bundle_candidates
        .iter()
        .filter(|remote| {
            fetch_repo_file(&repo, remote, dir, options)
                .or_else(|_| {
                    fetch_hf_http(
                        source.repo_id(),
                        remote,
                        dir,
                        &options.hf_token,
                        options.verbose,
                    )
                })
                .is_ok()
        })
        .count();

    if fetched == 0 {
        download_hf_primitives(&repo, dir, options)?;
        save_model_from_npy(name, options)?;
    } else {
        decompress_mpk_gz_variants(dir, name, options.verbose)?;
    }

    Ok(())
}

fn download_hf_primitives(
    repo: &ApiRepo,
    dir: &Path,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    const CANDIDATES: &[&str] = &[
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "pytorch_model.bin",
    ];

    CANDIDATES
        .iter()
        .filter_map(|file| repo.get(file).ok())
        .try_for_each(|path| {
            let dest = dir.join(
                path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown"),
            );
            fs::copy(&path, &dest)?;
            if options.verbose {
                tracing::info!("cached primitive {}", dest.display());
            }
            Ok(())
        })
}