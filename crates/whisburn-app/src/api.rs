use whisburn_core::PreloadModels;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    #[allow(dead_code)]
    pub description: String,
    pub burn_ready: bool,
    pub loaded: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    pub default_model: String,
    #[allow(dead_code)]
    pub preload_models: PreloadModels,
    pub loaded_models: Vec<String>,
}

pub async fn fetch_models(base_url: &str) -> anyhow::Result<Vec<ModelInfo>> {
    let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
    let models = reqwest::Client::new().get(url).send().await?.json().await?;
    Ok(models)
}

pub async fn fetch_config(base_url: &str) -> anyhow::Result<ServerConfig> {
    let url = format!("{}/v1/config", base_url.trim_end_matches('/'));
    let config = reqwest::Client::new().get(url).send().await?.json().await?;
    Ok(config)
}

pub async fn fetch_health(base_url: &str) -> anyhow::Result<()> {
    let url = format!("{}/health", base_url.trim_end_matches('/'));
    reqwest::Client::new()
        .get(url)
        .send()
        .await?
        .error_for_status()?;
    Ok(())
}

pub async fn transcribe_bytes(
    base_url: &str,
    file_name: &str,
    bytes: Vec<u8>,
    model: &str,
    format: &str,
    language: &str,
) -> anyhow::Result<Vec<u8>> {
    let url = format!(
        "{}/v1/transcribe?model={}&format={}&language={}&download=false",
        base_url.trim_end_matches('/'),
        urlencoding_encode(model),
        urlencoding_encode(format),
        urlencoding_encode(language),
    );

    let part = reqwest::multipart::Part::bytes(bytes)
        .file_name(file_name.to_string())
        .mime_str("application/octet-stream")?;

    let form = reqwest::multipart::Form::new().part("audio", part);

    let body = reqwest::Client::new()
        .post(url)
        .multipart(form)
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;

    Ok(body.to_vec())
}

fn urlencoding_encode(value: &str) -> String {
    value
        .chars()
        .map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
            _ => format!("%{:02X}", c as u8),
        })
        .collect()
}