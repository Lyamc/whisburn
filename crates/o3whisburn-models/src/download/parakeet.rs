use std::path::Path;
use std::process::Command;

use hf_hub::api::sync::Api;

use crate::convert::convert_parakeet_from_hf;
use crate::sources::DownloadSource;

use super::fetch::{fetch_optional_hf_file, fetch_required_hf_file};
use super::http::fetch_hf_http;
use super::options::DownloadOptions;

const PARAKEET_FILES: &[&str] = &["tokenizer.json", "config.json", "model.safetensors"];
const V2_NEMO: &str = "parakeet-tdt-0.6b-v2.nemo";
const V2_ONNX_VOCAB_REPO: &str = "istupakov/parakeet-tdt-0.6b-v2-onnx";

pub fn download_parakeet_hf(
    api: &Api,
    source: &DownloadSource,
    dir: &Path,
    name: &str,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    let repo = api.model(source.repo_id().to_string());

    if dir.join("shapes.json").exists() && dir.join("encoder").is_dir() {
        if options.verbose {
            tracing::info!("using existing npy dump in {}", dir.display());
        }
        return convert_parakeet_from_hf(dir, name, options);
    }

    if name == "parakeet-tdt-0.6b-v2" {
        download_parakeet_v2(&repo, source, dir, options)?;
    } else {
        for file in PARAKEET_FILES {
            fetch_required_hf_file(&repo, source.repo_id(), file, dir, options)?;
        }
        let _ = fetch_optional_hf_file(
            &repo,
            source.repo_id(),
            "preprocessor_config.json",
            dir,
            options,
        )?;
    }

    convert_parakeet_from_hf(dir, name, options)
}

fn download_parakeet_v2(
    repo: &hf_hub::api::sync::ApiRepo,
    source: &DownloadSource,
    dir: &Path,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    // Prefer a transformers-style bundle when NVIDIA publishes one.
    let has_safetensors = fetch_optional_hf_file(
        repo,
        source.repo_id(),
        "model.safetensors",
        dir,
        options,
    )?
    .is_some();
    if has_safetensors {
        for file in ["config.json", "tokenizer.json"] {
            fetch_optional_hf_file(repo, source.repo_id(), file, dir, options)?;
        }
        return Ok(());
    }

    if options.verbose {
        tracing::info!(
            "v2 is a NeMo archive (not transformers safetensors); converting to Burn, not ONNX Runtime"
        );
    }

    fetch_required_hf_file(repo, source.repo_id(), V2_NEMO, dir, options)?;
    let _ = fetch_hf_http(
        V2_ONNX_VOCAB_REPO,
        "vocab.txt",
        dir,
        &options.hf_token,
        options.verbose,
    );

    if dir.join("model.safetensors").exists() && dir.join("config.json").exists() {
        return Ok(());
    }

    extract_nemo_archive(dir, options)
}

fn extract_nemo_archive(dir: &Path, options: &DownloadOptions) -> anyhow::Result<()> {
    let script = Path::new("scripts/extract_nemo_parakeet.py");
    if !script.exists() {
        anyhow::bail!("missing {} — needed to lift .nemo weights into Burn", script.display());
    }
    let nemo = dir.join(V2_NEMO);
    let vocab = dir.join("vocab.txt");
    if options.verbose {
        tracing::info!("extracting {} to HF-style safetensors for Burn conversion", nemo.display());
    }

    let mut args = vec![
        script.to_string_lossy().into_owned(),
        "--nemo".into(),
        nemo.display().to_string(),
        "--out".into(),
        dir.display().to_string(),
    ];
    if vocab.exists() {
        args.push("--vocab".into());
        args.push(vocab.display().to_string());
    }
    let py_cmd = format!(
        "python3 {}",
        args.iter()
            .map(|a| format!("'{}'", a.replace('\'', r"'\''")))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let output = Command::new("python3")
        .args(&args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(Ok)
        .unwrap_or_else(|| {
            Command::new("nix-shell")
                .args(["-p", "python3", "python3Packages.numpy", "--run", &py_cmd])
                .output()
        })?;
    if !output.status.success() {
        anyhow::bail!(
            "nemo extract failed ({}):\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    if options.verbose {
        tracing::info!("{}", String::from_utf8_lossy(&output.stderr));
    }
    if !dir.join("model.safetensors").exists() {
        anyhow::bail!("nemo extract did not produce model.safetensors in {}", dir.display());
    }
    Ok(())
}