use std::sync::Arc;
use std::time::Instant;

use axum::body::Bytes;
use whisburn_audio::{
    decode_bytes_to_mono_pcm_with_max_duration, encode_wav, for_each_decoded_segment,
    format_txt_timestamped, probe_bytes_duration_secs, should_transcribe_in_segments,
    transcode_bytes, transcode_waveform, AudioExportFormat, OpusBitrate, Waveform,
    TRANSCRIBE_SEGMENT_OVERLAP_SECS, TRANSCRIBE_SEGMENT_SECS,
};
use whisburn_core::{SpeechTask, WhisburnError, TaskOptions, TranscriptResult, TranscriptSegment};
use whisburn_models::diarize::assign_alternating_speakers;
use whisburn_models::SharedModelManager;
use serde::Serialize;
use tokio::sync::{mpsc, Semaphore, SemaphorePermit};
use tracing::{info, warn};

use crate::handlers::ApiError;

pub const PREVIEW_MAX_SECONDS: u64 = 180;

#[derive(Debug, Clone, Serialize)]
pub struct TranscribeProgress {
    pub phase: String,
    pub task_label: String,
    pub task_pct: f64,
    pub overall_pct: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunks_total: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latest_segment: Option<TranscriptSegment>,
    /// Seconds since this transcription job started (for ETA calculations).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elapsed_secs: Option<f64>,
    /// Estimated seconds remaining (best-effort, based on current overall progress).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eta_secs: Option<f64>,
}

impl TranscribeProgress {
    fn emit(phase: &str, task_label: &str, task_pct: f64, overall_pct: f64) -> Self {
        Self {
            phase: phase.to_string(),
            task_label: task_label.to_string(),
            task_pct,
            overall_pct,
            chunk: None,
            chunks_total: None,
            partial_text: None,
            latest_segment: None,
            elapsed_secs: None,
            eta_secs: None,
        }
    }
}

pub type ProgressSender = mpsc::Sender<TranscribeProgress>;

pub async fn acquire_job(sem: &Arc<Semaphore>) -> Result<SemaphorePermit<'_>, ApiError> {
    sem.acquire()
        .await
        .map_err(|_| ApiError::bad_request("server is shutting down"))
}

fn send_progress(tx: &Option<ProgressSender>, event: TranscribeProgress) {
    if let Some(tx) = tx {
        if tx.try_send(event).is_err() {
            warn!("transcribe progress channel full; dropping event");
        }
    }
}

/// Attach elapsed time + ETA estimate to a progress event.
/// `overall_pct_override` lets the caller force a better current % for ETA math.
fn with_timing(mut p: TranscribeProgress, start: Instant, overall_pct_override: Option<f64>) -> TranscribeProgress {
    let elapsed = start.elapsed().as_secs_f64();
    p.elapsed_secs = Some(elapsed);
    let pct = overall_pct_override.unwrap_or(p.overall_pct).clamp(0.0, 99.9);
    if pct > 0.5 {
        let est_total = elapsed / (pct / 100.0);
        let remaining = (est_total - elapsed).max(0.0);
        p.eta_secs = Some(remaining);
    }
    p
}

fn estimate_chunks(duration_secs: Option<f64>, segmented: bool) -> Option<usize> {
    if !segmented {
        return Some(1);
    }
    duration_secs.map(|secs| {
        let step = TRANSCRIBE_SEGMENT_SECS - TRANSCRIBE_SEGMENT_OVERLAP_SECS;
        ((secs / step).ceil() as usize).max(1)
    })
}

pub async fn decode_upload(
    bytes: Bytes,
    extension: String,
    max_seconds: Option<u64>,
) -> Result<Waveform, ApiError> {
    tokio::task::spawn_blocking(move || {
        decode_bytes_to_mono_pcm_with_max_duration(bytes.to_vec(), &extension, max_seconds)
    })
    .await
    .map_err(|e| ApiError::bad_request(format!("decode task panicked: {e}")))?
    .map_err(|e| ApiError::bad_request(e.to_string()))
}

pub async fn transcribe_upload(
    manager: SharedModelManager,
    model: String,
    bytes: Bytes,
    extension: String,
    options: TaskOptions,
) -> Result<TranscriptResult, ApiError> {
    transcribe_upload_with_progress(manager, model, bytes, extension, options, None).await
}

pub async fn transcribe_upload_with_progress(
    manager: SharedModelManager,
    model: String,
    bytes: Bytes,
    extension: String,
    options: TaskOptions,
    progress: Option<ProgressSender>,
) -> Result<TranscriptResult, ApiError> {
    let job_start = Instant::now();
    let bytes_vec = bytes.to_vec();
    let byte_len = bytes_vec.len();
    let ext = extension.clone();

    send_progress(
        &progress,
        with_timing(TranscribeProgress::emit("probe", "Analyzing audio duration", 0.0, 1.0), job_start, Some(1.0)),
    );

    let duration = tokio::task::spawn_blocking({
        let probe_bytes = bytes_vec.clone();
        let ext = extension.clone();
        move || probe_bytes_duration_secs(&probe_bytes, &ext)
    })
    .await
    .map_err(|e| ApiError::bad_request(format!("probe task panicked: {e}")))?
    .map_err(|e| ApiError::bad_request(e.to_string()))?;

    let segmented = should_transcribe_in_segments(byte_len, duration);
    let chunks_total = estimate_chunks(duration, segmented);
    if let Some(secs) = duration {
        info!(bytes = byte_len, duration_secs = secs, segmented, "transcribe audio probed");
    }

    // Emit a rough upfront ETA hint as soon as we know size + chunk count
    if let (Some(dur), Some(ch)) = (duration, chunks_total) {
        let rough = (dur * 0.22 + ch as f64 * 1.8).max(2.5);
        send_progress(
            &progress,
            with_timing(TranscribeProgress {
                phase: "estimate".into(),
                task_label: format!("~{:.0}s est. for {:.1}s audio ({} segments)", rough, dur, ch),
                task_pct: 4.0,
                overall_pct: 4.0,
                chunk: None,
                chunks_total: Some(ch),
                partial_text: None,
                latest_segment: None,
                elapsed_secs: Some(job_start.elapsed().as_secs_f64()),
                eta_secs: Some(rough),
            }, job_start, Some(4.0)),
        );
    }

    send_progress(
        &progress,
        with_timing(TranscribeProgress::emit("probe", "Analyzing audio duration", 100.0, 5.0), job_start, Some(5.0)),
    );

    send_progress(
        &progress,
        with_timing(TranscribeProgress::emit("model", "Loading model (download if needed)", 0.0, 5.0), job_start, Some(5.0)),
    );

    ensure_model_job(manager.clone(), model.clone()).await?;

    send_progress(
        &progress,
        with_timing(TranscribeProgress::emit("model", "Model ready", 100.0, 15.0), job_start, Some(15.0)),
    );

    let progress_for_blocking = progress.clone();
    let model_name = model.clone();
    let opts = options.clone();
    let diarize = options.task == SpeechTask::Diarize;
    let show_timestamps = options.include_timestamps;

    if !segmented {
        send_progress(
            &progress,
            with_timing(TranscribeProgress::emit("decode", "Decoding audio", 0.0, 15.0), job_start, Some(15.0)),
        );
        let waveform = decode_upload(Bytes::from(bytes_vec), ext, None).await?;
        send_progress(
            &progress,
            with_timing(TranscribeProgress::emit("decode", "Decoding audio", 100.0, 25.0), job_start, Some(25.0)),
        );
        send_progress(
            &progress,
            with_timing(TranscribeProgress {
                phase: "transcribe".into(),
                task_label: "Transcribing".into(),
                task_pct: 0.0,
                overall_pct: 25.0,
                chunk: Some(1),
                chunks_total: Some(1),
                partial_text: None,
                latest_segment: None,
                elapsed_secs: None,
                eta_secs: None,
            }, job_start, Some(25.0)),
        );

        let mut chunk_options = opts.clone();
        if diarize {
            chunk_options.task = SpeechTask::Transcribe;
        }
        let result = transcribe_waveform(
            manager,
            model,
            waveform.samples,
            waveform.sample_rate,
            chunk_options,
        )
        .await?;

        let mut final_result = result;
        if diarize {
            final_result.task = "diarize".into();
            final_result = assign_alternating_speakers(final_result);
        }

        send_progress(
            &progress,
            with_timing(TranscribeProgress {
                phase: "transcribe".into(),
                task_label: "Transcribing".into(),
                task_pct: 100.0,
                overall_pct: 95.0,
                chunk: Some(1),
                chunks_total: Some(1),
                partial_text: Some(preview_text(&final_result, show_timestamps, diarize)),
                latest_segment: final_result.segments.last().cloned(),
                elapsed_secs: None,
                eta_secs: None,
            }, job_start, Some(95.0)),
        );
        return Ok(final_result);
    }

    let task_label = options.task.as_str().to_string();
    let language = options.language.as_str().to_string();
    let total = chunks_total.unwrap_or(1);

    tokio::task::spawn_blocking(move || {
        let mut merged = TranscriptResult::empty(&model_name, &task_label);
        merged.language = Some(language);
        let mut chunk_idx = 0usize;
        let mut chunk_options = opts.clone();
        if diarize {
            chunk_options.task = SpeechTask::Transcribe;
        }

        send_progress(
            &progress_for_blocking,
            with_timing(TranscribeProgress {
                phase: "decode".into(),
                task_label: "Decoding & transcribing segments".into(),
                task_pct: 0.0,
                overall_pct: 15.0,
                chunk: Some(0),
                chunks_total: Some(total),
                partial_text: None,
                latest_segment: None,
                elapsed_secs: None,
                eta_secs: None,
            }, job_start, Some(15.0)),
        );

        for_each_decoded_segment(
            bytes_vec,
            &ext,
            TRANSCRIBE_SEGMENT_SECS,
            TRANSCRIBE_SEGMENT_OVERLAP_SECS,
            None,
            |waveform| {
                let part = manager
                    .process_waveform_sync(
                        &model_name,
                        waveform.samples,
                        waveform.sample_rate,
                        &chunk_options,
                    )
                    .map_err(|e| {
                        warn!(model = %model_name, chunk = chunk_idx, error = %e, "segment transcribe failed");
                        WhisburnError::Inference(e.to_string())
                    })?;

                let offset = chunk_idx as f64
                    * (TRANSCRIBE_SEGMENT_SECS - TRANSCRIBE_SEGMENT_OVERLAP_SECS);
                let skip_before = if chunk_idx == 0 {
                    0.0
                } else {
                    TRANSCRIBE_SEGMENT_OVERLAP_SECS
                };
                merge_transcript(&mut merged, part, offset, skip_before);
                chunk_idx += 1;

                let task_pct = (chunk_idx as f64 / total as f64) * 100.0;
                let overall_pct = 15.0 + (chunk_idx as f64 / total as f64) * 80.0;
                send_progress(
                    &progress_for_blocking,
                    with_timing(TranscribeProgress {
                        phase: "transcribe".into(),
                        task_label: format!("Transcribing segment {chunk_idx} of {total}"),
                        task_pct,
                        overall_pct,
                        chunk: Some(chunk_idx),
                        chunks_total: Some(total),
                        partial_text: Some(preview_text(&merged, show_timestamps, diarize)),
                        latest_segment: merged.segments.last().cloned(),
                        elapsed_secs: None,
                        eta_secs: None,
                    }, job_start, Some(overall_pct)),
                );
                Ok(())
            },
        )
        .map_err(|e| ApiError::bad_request(e.to_string()))?;

        if diarize {
            merged.task = "diarize".into();
            merged = assign_alternating_speakers(merged);
        }

        info!(chunks = chunk_idx, "segmented transcribe complete");
        // Final progress with timing (we don't have a perfect 100% here, the caller may send "done")
        Ok(merged)
    })
    .await
    .map_err(|e| ApiError::bad_request(format!("transcribe task panicked: {e}")))?
}

fn preview_text(result: &TranscriptResult, timestamps: bool, diarize: bool) -> String {
    let preview = if diarize {
        assign_alternating_speakers(result.clone())
    } else {
        result.clone()
    };
    if timestamps && !preview.segments.is_empty() {
        let speakers = preview.segments.iter().any(|s| s.speaker.is_some());
        format_txt_timestamped(&preview.segments, speakers)
    } else {
        preview.text.clone()
    }
}

fn merge_transcript(
    acc: &mut TranscriptResult,
    part: TranscriptResult,
    offset_secs: f64,
    skip_before_secs: f64,
) {
    for seg in part.segments {
        if skip_before_secs > 0.0 && seg.end <= skip_before_secs {
            continue;
        }
        let start = seg.start + offset_secs;
        let end = seg.end + offset_secs;
        if end <= start {
            continue;
        }
        acc.segments.push(TranscriptSegment {
            start,
            end,
            text: seg.text,
            speaker: seg.speaker,
        });
    }

    let text = part.text.trim();
    if !text.is_empty() {
        if !acc.text.is_empty() {
            acc.text.push(' ');
        }
        acc.text.push_str(text);
    }
}

pub async fn transcribe_waveform(
    manager: SharedModelManager,
    model: String,
    samples: Vec<f32>,
    sample_rate: usize,
    options: TaskOptions,
) -> Result<TranscriptResult, ApiError> {
    manager
        .process_waveform_sync(&model, samples, sample_rate, &options)
        .map_err(|e| {
            warn!(model = %model, error = %e, "transcribe failed");
            ApiError::from(e)
        })
}

pub async fn transcode_upload(
    bytes: Bytes,
    extension: String,
    format: AudioExportFormat,
    bitrate: OpusBitrate,
    max_seconds: Option<u64>,
) -> Result<Vec<u8>, ApiError> {
    let use_ffmpeg = max_seconds.is_some()
        || should_transcribe_in_segments(
            bytes.len(),
            probe_bytes_duration_secs(&bytes, &extension).ok().flatten(),
        );

    if use_ffmpeg {
        let data = bytes.to_vec();
        let ext = extension;
        return tokio::task::spawn_blocking(move || {
            transcode_bytes(&data, &ext, format, bitrate, max_seconds)
        })
        .await
        .map_err(|e| ApiError::bad_request(format!("transcode task panicked: {e}")))?
        .map_err(ApiError::from);
    }

    let waveform = decode_upload(bytes, extension, None).await?;
    transcode_waveform_job(waveform, format, bitrate).await
}

pub async fn transcode_waveform_job(
    waveform: Waveform,
    format: AudioExportFormat,
    bitrate: OpusBitrate,
) -> Result<Vec<u8>, ApiError> {
    tokio::task::spawn_blocking(move || transcode_waveform(&waveform, format, bitrate))
        .await
        .map_err(|e| ApiError::bad_request(format!("transcode task panicked: {e}")))?
        .map_err(ApiError::from)
}

pub async fn encode_wav_job(waveform: Waveform) -> Result<Vec<u8>, ApiError> {
    tokio::task::spawn_blocking(move || encode_wav(&waveform))
        .await
        .map_err(|e| ApiError::bad_request(format!("wav encode task panicked: {e}")))?
        .map_err(ApiError::from)
}

pub async fn ensure_model_job(
    manager: SharedModelManager,
    name: String,
) -> Result<std::path::PathBuf, ApiError> {
    tokio::task::spawn_blocking(move || manager.ensure_model_sync(&name))
        .await
        .map_err(|e| ApiError::bad_request(format!("ensure task panicked: {e}")))?
        .map_err(ApiError::from)
}