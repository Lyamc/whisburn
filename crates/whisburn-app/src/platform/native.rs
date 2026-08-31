use whisburn_core::{load_settings, save_settings, Settings};

use super::AudioFile;

pub async fn pick_audio() -> Option<AudioFile> {
    let handle = rfd::AsyncFileDialog::new()
        .add_filter("Audio", &["wav", "mp3", "flac", "ogg", "m4a", "opus"])
        .pick_file()
        .await?;

    let path = handle.path().to_path_buf();
    let bytes = tokio::fs::read(&path).await.ok()?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("audio.wav")
        .to_string();

    Some(AudioFile { name, bytes, path })
}

pub fn load_app_settings() -> Settings {
    load_settings()
}

pub fn save_app_settings(settings: &Settings) -> Result<std::path::PathBuf, String> {
    save_settings(settings).map_err(|e| e.to_string())
}

pub fn settings_hint() -> &'static str {
    "Saved to the whisburn config directory (settings.toml). Server reads these on startup unless CLI flags override."
}