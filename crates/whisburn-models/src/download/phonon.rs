use std::fs;
use std::path::Path;
use std::process::Command;

use hf_hub::api::sync::Api;

use crate::convert::phonon::convert_phonon_container;
use crate::sources::DownloadSource;

use super::fetch::fetch_required_hf_file;
use super::http::fetch_hf_http;
use super::options::DownloadOptions;

const ARCHIVE: &str = "phonon-2.bps.tar.zst";
const CONTAINER: &str = "model.fermion";
/// Phonon-2 is a packed checkpoint of this Parakeet TDT checkpoint. Config and
/// tokenizer come from the base repo; the phonon repo's config.json is metadata.
const BASE_REPO: &str = "nvidia/parakeet-tdt-0.6b-v3";
const BASE_FILES: &[&str] = &["config.json", "tokenizer.json"];

pub fn download_phonon_hf(
    api: &Api,
    source: &DownloadSource,
    dir: &Path,
    name: &str,
    options: &DownloadOptions,
) -> anyhow::Result<()> {
    let repo = api.model(source.repo_id().to_string());
    if options.verbose {
        tracing::info!(
            "preparing Phonon-2 from {} (base tokenizer {BASE_REPO})",
            source.repo_id()
        );
    }

    for file in BASE_FILES {
        fetch_hf_http(BASE_REPO, file, dir, options)?;
    }
    let _ = fetch_hf_http(BASE_REPO, "preprocessor_config.json", dir, options);

    let container = dir.join(CONTAINER);
    if !container.exists() {
        let archive = fetch_required_hf_file(&repo, source.repo_id(), ARCHIVE, dir, options)?;
        extract_container(&archive, dir, options)?;
        if archive.exists() {
            let _ = fs::remove_file(&archive);
        }
    }
    if !container.exists() {
        anyhow::bail!(
            "phonon archive did not contain {CONTAINER} in {}",
            dir.display()
        );
    }

    convert_phonon_container(&container, dir, name, options)?;
    if container.exists() {
        let _ = fs::remove_file(&container);
    }
    Ok(())
}

fn extract_container(archive: &Path, dir: &Path, options: &DownloadOptions) -> anyhow::Result<()> {
    options.report(crate::download::PrepProgress::convert(
        "Unpacking Phonon-2 container",
        0.4,
    ));
    // The archive also contains a metadata config.json. Unpack beside the
    // model dir so it cannot replace the base Parakeet config.
    let staging = dir.join(".phonon-unpack");
    if staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    fs::create_dir_all(&staging)?;
    let status = Command::new("tar")
        .args(["-I", "zstd", "-xf"])
        .arg(archive)
        .arg("-C")
        .arg(&staging)
        .status()
        .map_err(|e| anyhow::anyhow!("failed to run tar -I zstd ({e}). zstd is required to unpack Phonon-2"))?;
    if !status.success() {
        anyhow::bail!("tar failed to unpack {}", archive.display());
    }
    let found = find_named(&staging, CONTAINER)?.ok_or_else(|| {
        anyhow::anyhow!("phonon archive has no {CONTAINER}")
    })?;
    fs::rename(&found, dir.join(CONTAINER))?;
    fs::remove_dir_all(&staging)?;
    Ok(())
}

fn find_named(dir: &Path, name: &str) -> anyhow::Result<Option<std::path::PathBuf>> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.file_name().and_then(|n| n.to_str()) == Some(name) {
            return Ok(Some(path));
        }
        if path.is_dir() {
            if let Some(found) = find_named(&path, name)? {
                return Ok(Some(found));
            }
        }
    }
    Ok(None)
}
