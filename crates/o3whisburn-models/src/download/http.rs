use std::fs;
use std::io::copy;
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn fetch_hf_http(
    repo_id: &str,
    remote_path: &str,
    dest_dir: &Path,
    token: &Option<String>,
    verbose: bool,
) -> anyhow::Result<PathBuf> {
    let url = format!("https://huggingface.co/{repo_id}/resolve/main/{remote_path}");
    let file_name = Path::new(remote_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(remote_path);
    let dest = dest_dir.join(file_name);

    if verbose {
        tracing::info!("http GET {url}");
    }

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }

    let ureq_err = match fetch_hf_http_ureq(&url, &dest, token) {
        Ok(()) => return Ok(dest),
        Err(e) => e,
    };

    if cfg!(windows) {
        fetch_hf_http_curl(&url, &dest, token, verbose)
            .map_err(|curl_err| anyhow::anyhow!("ureq: {ureq_err}; curl: {curl_err}"))?;
        return Ok(dest);
    }

    Err(ureq_err)
}

fn fetch_hf_http_ureq(url: &str, dest: &Path, token: &Option<String>) -> anyhow::Result<()> {
    let mut request = ureq::get(url);
    if let Some(token) = token {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }

    let response = request.call().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut reader = response.into_reader();
    let mut file = std::fs::File::create(dest)?;
    copy(&mut reader, &mut file)?;
    Ok(())
}

fn fetch_hf_http_curl(
    url: &str,
    dest: &Path,
    token: &Option<String>,
    verbose: bool,
) -> anyhow::Result<()> {
    let mut cmd = Command::new("curl");
    cmd.arg("-fL")
        .arg("--retry")
        .arg("5")
        .arg("--retry-delay")
        .arg("3")
        .arg("-C")
        .arg("-")
        .arg("-o")
        .arg(dest);

    if let Some(token) = token {
        cmd.arg("-H").arg(format!("Authorization: Bearer {token}"));
    }

    cmd.arg(url);

    if verbose {
        tracing::info!("curl fallback: {url}");
    }

    let status = cmd
        .status()
        .map_err(|e| anyhow::anyhow!("curl spawn failed: {e}"))?;
    if !status.success() {
        anyhow::bail!("curl exited with {status}");
    }

    Ok(())
}