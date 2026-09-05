use std::convert::Infallible;

use axum::body::Body;
use axum::extract::{Multipart, Path, Query, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{sse::Event, sse::KeepAlive, IntoResponse, Response, Sse};
use futures_util::stream::Stream;
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use whisburn_audio::{encode_result_with_options, AudioExportFormat, OpusBitrate};
use whisburn_core::{LanguageCode, OutputFormat, SpeechTask, TaskOptions, WhisburnError};
use whisburn_engine::model::registry::{self, ModelTier};
use crate::gpu::GpuInfo;

use tracing::info;

use crate::jobs::{
    acquire_job, ensure_model_job, transcode_upload, transcribe_upload,
    transcribe_upload_with_progress, TranscribeProgress, PREVIEW_MAX_SECONDS,
};
use crate::state::AppState;
use crate::upload::{attachment_filename, read_audio_field};

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub service: &'static str,
}

pub async fn health() -> impl IntoResponse {
    axum::Json(HealthResponse {
        status: "ok",
        service: "whisburn",
    })
}

#[derive(Serialize)]
pub struct ServerConfigResponse {
    pub default_model: String,
    pub preload_models: whisburn_core::PreloadModels,
    pub loaded_models: Vec<String>,
    pub max_audio_seconds: u64,
    pub gpu: GpuInfo,
}

pub async fn server_config(State(state): State<AppState>) -> impl IntoResponse {
    axum::Json(ServerConfigResponse {
        default_model: state.default_model.clone(),
        preload_models: state.preload_models.clone(),
        loaded_models: state.manager.loaded_models(),
        max_audio_seconds: state.max_audio_seconds,
        gpu: state.gpu.clone(),
    })
}

#[derive(Serialize)]
pub struct ModelTierDto {
    pub size: ModelTier,
    pub quality: ModelTier,
    pub effort: ModelTier,
    pub efficiency: ModelTier,
    pub efficiency_pct: u8,
    pub tagline: String,
}

#[derive(Serialize)]
pub struct ModelListItem {
    pub name: String,
    pub hf_id: String,
    pub description: String,
    pub category: String,
    pub vram_mb: u32,
    pub burn_ready: bool,
    pub loaded: bool,
    /// `ok`, `tight`, `no`, or `unknown` vs server GPU VRAM.
    pub vram_fit: String,
    pub tiers: ModelTierDto,
}

#[derive(Serialize)]
pub struct EnsureModelResponse {
    pub name: String,
    pub ready: bool,
    pub path: String,
}

pub async fn ensure_model(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, ApiError> {
    let _permit = acquire_job(&state.job_semaphore).await?;
    let model_name = name.clone();
    let path = ensure_model_job(state.manager.clone(), name, None).await?;

    Ok(axum::Json(EnsureModelResponse {
        name: model_name,
        ready: true,
        path: path.display().to_string(),
    }))
}

pub async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let loaded: std::collections::HashSet<_> =
        state.manager.loaded_models().into_iter().collect();

    let gpu_vram = state.gpu.vram_mb;
    let models = registry::MODEL_REGISTRY
        .iter()
        .map(|m| ModelListItem {
            name: m.name.to_string(),
            hf_id: m.hf_id.to_string(),
            description: m.description.to_string(),
            category: format!("{:?}", m.category),
            vram_mb: m.vram_mb,
            burn_ready: m.burn_ready,
            loaded: loaded.contains(m.name),
            vram_fit: registry::vram_fit(m.vram_mb, gpu_vram).to_string(),
            tiers: ModelTierDto {
                size: m.size,
                quality: m.quality,
                effort: m.effort,
                efficiency: m.efficiency,
                efficiency_pct: m.efficiency_pct,
                tagline: m.tagline.to_string(),
            },
        })
        .collect::<Vec<_>>();

    axum::Json(models)
}

#[derive(Debug, Deserialize)]
pub struct TranscribeQuery {
    pub model: Option<String>,
    pub task: Option<String>,
    pub language: Option<String>,
    pub format: Option<String>,
    pub timestamps: Option<bool>,
    pub orchestrate: Option<bool>,
    /// Group segments into sentences (affects timed txt/srt/vtt and adds "sentences" to json).
    pub sentences: Option<bool>,
    /// When true, sets Content-Disposition: attachment for browser downloads.
    pub download: Option<bool>,
    /// Override downloaded file extension (e.g. `txt` while format is `srt`).
    pub file_extension: Option<String>,
    /// When true, use the SSE streaming endpoint semantics on the regular route too.
    pub stream: Option<bool>,
    /// Offline English summary with Qwen3-0.6B; also writes `{stem}_summary.txt` in the UI.
    pub summarize: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct TranscodeQuery {
    pub format: Option<String>,
    pub bitrate: Option<String>,
    pub download: Option<bool>,
}

pub async fn transcribe(
    State(state): State<AppState>,
    Query(query): Query<TranscribeQuery>,
    multipart: Multipart,
) -> Result<Response, ApiError> {
    let mut multipart = multipart;
    let upload = read_audio_field(&mut multipart).await?;

    if upload.bytes.len() > state.max_upload_bytes {
        return Err(ApiError::bad_request("upload exceeds size limit"));
    }

    let model = query
        .model
        .unwrap_or_else(|| state.default_model.clone());

    info!(
        file = %upload.filename_stem,
        bytes = upload.bytes.len(),
        model = %model,
        "transcribe upload received"
    );

    let _permit = acquire_job(&state.job_semaphore).await?;

    let task = query
        .task
        .as_deref()
        .and_then(SpeechTask::parse)
        .unwrap_or(SpeechTask::Transcribe);

    let format = query
        .format
        .as_deref()
        .and_then(OutputFormat::parse)
        .unwrap_or(OutputFormat::Json);

    let options = TaskOptions {
        task,
        language: LanguageCode::new(query.language.unwrap_or_else(|| "en".to_string())),
        target_language: None,
        include_timestamps: query.timestamps.unwrap_or(true),
        orchestrate: query.orchestrate.unwrap_or(false),
        group_into_sentences: query.sentences.unwrap_or(false),
        ..TaskOptions::default()
    };

    let include_timestamps = query.timestamps.unwrap_or(true);
    let sentences = query.sentences.unwrap_or(false);
    let result = transcribe_upload(
        state.manager.clone(),
        model,
        upload.bytes,
        upload.extension,
        options,
    )
    .await?;

    let body =
        encode_result_with_options(&result, format, include_timestamps, sentences).map_err(ApiError::from)?;

    let file_ext = query
        .file_extension
        .as_deref()
        .unwrap_or(format.file_extension());

    let mut response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, format.content_type());

    if query.download.unwrap_or(true) {
        let filename = attachment_filename(&upload.filename_stem, file_ext);
        response = response.header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{filename}\""),
        );
    }

    Ok(response.body(Body::from(body)).unwrap())
}

/// Server-Sent Events stream of [`TranscribeProgress`] while transcribing.
pub async fn transcribe_stream(
    State(state): State<AppState>,
    Query(query): Query<TranscribeQuery>,
    multipart: Multipart,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let mut multipart = multipart;
    let upload = read_audio_field(&mut multipart).await?;

    if upload.bytes.len() > state.max_upload_bytes {
        return Err(ApiError::bad_request("upload exceeds size limit"));
    }

    let model = query
        .model
        .unwrap_or_else(|| state.default_model.clone());

    let _permit = acquire_job(&state.job_semaphore).await?;

    let task = query
        .task
        .as_deref()
        .and_then(SpeechTask::parse)
        .unwrap_or(SpeechTask::Transcribe);

    let include_timestamps = query.timestamps.unwrap_or(true);
    let sentences = query.sentences.unwrap_or(false);
    let format = query
        .format
        .as_deref()
        .and_then(OutputFormat::parse)
        .unwrap_or(OutputFormat::Txt);

    let options = TaskOptions {
        task,
        language: LanguageCode::new(query.language.unwrap_or_else(|| "en".to_string())),
        target_language: None,
        include_timestamps,
        orchestrate: query.orchestrate.unwrap_or(false),
        group_into_sentences: sentences,
        ..TaskOptions::default()
    };

    let manager = state.manager.clone();
    let bytes = upload.bytes;
    let extension = upload.extension;
    let summarize = query.summarize.unwrap_or(false);
    let (tx, rx) = tokio::sync::mpsc::channel::<TranscribeProgress>(256);
    let stream_start = std::time::Instant::now();

    tokio::spawn(async move {
        match transcribe_upload_with_progress(
            manager.clone(),
            model,
            bytes,
            extension,
            options,
            Some(tx.clone()),
        )
        .await
        {
            Ok(result) => {
                let encoded = encode_result_with_options(&result, format, include_timestamps, sentences)
                    .map(|b| String::from_utf8(b).unwrap_or_else(|_| result.text.clone()))
                    .unwrap_or_else(|_| result.text.clone());
                let summary = if summarize && !result.text.trim().is_empty() {
                    let _ = tx
                        .send(TranscribeProgress {
                            phase: "summarize".into(),
                            task_label: "Summarizing with offline Qwen3-0.6B…".into(),
                            task_pct: 0.0,
                            overall_pct: 98.0,
                            chunk: None,
                            chunks_total: None,
                            partial_text: Some(encoded.clone()),
                            latest_segment: result.segments.last().cloned(),
                            elapsed_secs: Some(stream_start.elapsed().as_secs_f64()),
                            eta_secs: None,
                            summary: None,
                        })
                        .await;
                    let text = result.text.clone();
                    let tx_prog = tx.clone();
                    let progress = std::sync::Arc::new(move |i: usize, n: usize, label: &str| {
                        let pct = if n == 0 { 100.0 } else { (i as f64 / n as f64) * 100.0 };
                        let overall = 98.0 + (pct / 100.0) * 1.5;
                        let _ = tx_prog.try_send(TranscribeProgress {
                            phase: "summarize".into(),
                            task_label: label.to_string(),
                            task_pct: pct,
                            overall_pct: overall,
                            chunk: Some(i + 1),
                            chunks_total: Some(n),
                            partial_text: None,
                            latest_segment: None,
                            elapsed_secs: None,
                            eta_secs: None,
                            summary: None,
                        });
                    });
                    match tokio::task::spawn_blocking(move || manager.summarize_text(&text, Some(progress)))
                        .await
                    {
                        Ok(Ok(s)) => Some(s),
                        Ok(Err(e)) => {
                            tracing::warn!("offline summarize failed: {e}");
                            Some(format!("(summary failed: {e})"))
                        }
                        Err(e) => {
                            tracing::warn!("offline summarize panicked: {e}");
                            Some(format!("(summary failed: {e})"))
                        }
                    }
                } else {
                    None
                };
                let elapsed = stream_start.elapsed().as_secs_f64();
                let _ = tx
                    .send(TranscribeProgress {
                        phase: "done".into(),
                        task_label: if summary.as_ref().is_some_and(|s| !s.starts_with("(summary failed")) {
                            "Complete (transcript + summary)".into()
                        } else {
                            "Complete".into()
                        },
                        task_pct: 100.0,
                        overall_pct: 100.0,
                        chunk: None,
                        chunks_total: None,
                        partial_text: Some(encoded),
                        latest_segment: result.segments.last().cloned(),
                        elapsed_secs: Some(elapsed),
                        eta_secs: Some(0.0),
                        summary,
                    })
                    .await;
            }
            Err(e) => {
                let _ = tx
                    .send(TranscribeProgress {
                        phase: "error".into(),
                        task_label: e.to_string(),
                        task_pct: 0.0,
                        overall_pct: 0.0,
                        chunk: None,
                        chunks_total: None,
                        partial_text: None,
                        latest_segment: None,
                        elapsed_secs: None,
                        eta_secs: None,
                        summary: None,
                    })
                    .await;
            }
        }
    });

    let stream = ReceiverStream::new(rx).map(|progress| {
        Ok(Event::default()
            .event("progress")
            .data(serde_json::to_string(&progress).unwrap_or_else(|_| "{}".into())))
    });

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

pub async fn transcode(
    State(state): State<AppState>,
    Query(query): Query<TranscodeQuery>,
    multipart: Multipart,
) -> Result<Response, ApiError> {
    let mut multipart = multipart;
    let upload = read_audio_field(&mut multipart).await?;

    if upload.bytes.len() > state.max_upload_bytes {
        return Err(ApiError::bad_request("upload exceeds size limit"));
    }

    let _permit = acquire_job(&state.job_semaphore).await?;

    let format = query
        .format
        .as_deref()
        .and_then(AudioExportFormat::parse)
        .unwrap_or(AudioExportFormat::OpusOgg);

    let bitrate = query
        .bitrate
        .as_deref()
        .and_then(OpusBitrate::parse)
        .unwrap_or(OpusBitrate::Medium);

    let body = transcode_upload(upload.bytes, upload.extension, format, bitrate, None).await?;

    let mut response = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, format.content_type());

    if query.download.unwrap_or(true) {
        let filename = attachment_filename(&upload.filename_stem, format.file_extension());
        response = response.header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{filename}\""),
        );
    }

    Ok(response.body(Body::from(body)).unwrap())
}

/// Chunked Opus/Ogg stream for progressive browser playback.
pub async fn transcode_stream(
    State(state): State<AppState>,
    Query(query): Query<TranscodeQuery>,
    multipart: Multipart,
) -> Result<Response, ApiError> {
    let mut multipart = multipart;
    let upload = read_audio_field(&mut multipart).await?;

    if upload.bytes.len() > state.max_upload_bytes {
        return Err(ApiError::bad_request("upload exceeds size limit"));
    }

    let _permit = acquire_job(&state.job_semaphore).await?;

    if !whisburn_audio::ffmpeg_available() {
        return Err(ApiError::from(WhisburnError::UnsupportedCapability {
            model: "transcode".into(),
            capability: "opus streaming requires ffmpeg in PATH".into(),
        }));
    }

    let bitrate = query
        .bitrate
        .as_deref()
        .and_then(OpusBitrate::parse)
        .unwrap_or(OpusBitrate::Medium);

    let body_bytes = transcode_upload(
        upload.bytes,
        upload.extension,
        AudioExportFormat::OpusOgg,
        bitrate,
        Some(PREVIEW_MAX_SECONDS),
    )
    .await?;
    let body = Body::from(body_bytes);

    let filename = attachment_filename(&upload.filename_stem, "ogg");
    let disposition = format!("inline; filename=\"{filename}\"");

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "audio/ogg")
        .header(header::CACHE_CONTROL, "no-store")
        .header(
            header::CONTENT_DISPOSITION,
            HeaderValue::from_str(&disposition).unwrap_or_else(|_| HeaderValue::from_static("inline")),
        )
        .body(body)
        .unwrap())
}

pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.message.fmt(f)
    }
}

impl ApiError {
    pub(crate) fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }
}

impl From<WhisburnError> for ApiError {
    fn from(value: WhisburnError) -> Self {
        let status = match &value {
            WhisburnError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
            WhisburnError::UnsupportedCapability { .. } => StatusCode::NOT_IMPLEMENTED,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self {
            status,
            message: value.to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({ "error": self.message });
        (self.status, axum::Json(body)).into_response()
    }
}