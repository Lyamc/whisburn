use axum::body::Bytes;
use axum::extract::Multipart;

use crate::handlers::ApiError;

#[derive(Debug, Clone)]
pub struct UploadedAudio {
    pub bytes: Bytes,
    pub extension: String,
    pub filename_stem: String,
}

pub async fn read_audio_field(multipart: &mut Multipart) -> Result<UploadedAudio, ApiError> {
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::bad_request(e.to_string()))?
    {
        let name = field.name().unwrap_or_default();
        if name == "audio" || name == "file" {
            let mut extension = "wav".to_string();
            let mut filename_stem = "audio".to_string();
            if let Some(filename) = field.file_name() {
                let path = std::path::Path::new(filename);
                if let Some(ext) = path.extension() {
                    extension = ext.to_string_lossy().to_string();
                }
                if let Some(stem) = path.file_stem() {
                    filename_stem = stem.to_string_lossy().to_string();
                }
            }
            let bytes = field
                .bytes()
                .await
                .map_err(|e| ApiError::bad_request(e.to_string()))?;
            return Ok(UploadedAudio {
                bytes,
                extension,
                filename_stem,
            });
        }
    }

    Err(ApiError::bad_request(
        "missing audio file field (use 'audio' or 'file')",
    ))
}

pub fn attachment_filename(stem: &str, extension: &str) -> String {
    let safe_stem: String = stem
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let stem = if safe_stem.is_empty() {
        "whisburn"
    } else {
        safe_stem.as_str()
    };
    format!("{stem}.{extension}")
}