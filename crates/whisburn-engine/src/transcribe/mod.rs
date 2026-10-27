pub mod mel;
pub mod mels_to_text;
pub mod utils;
pub mod decode;
pub mod task;
pub mod whisper_config;

pub use task::DecodeTask;

pub use utils::{get_output_path, format_srt, format_timestamp, parse_time, TranscriptSegment, clean_text, find_chunk_overlap, trim_trailing_hallucination};

use std::cell::RefCell;
use std::sync::Arc;

use crate::audio::max_waveform_samples;
use crate::model::{get_model_stats, Model};
use crate::token::{self, *};
use self::mel::{mel_window_count, waveform_to_mel_tensor};
use self::decode::*;

thread_local! {
    static DECODE_PROGRESS: RefCell<Option<Arc<dyn Fn(usize, usize) + Send + Sync>>> =
        RefCell::new(None);
    /// Set while one server segment is split into shorter GPU slices.
    /// `parent_samples == 0` means each `waveform_to_text` reports its own windows.
    static DECODE_WINDOW_SPAN: RefCell<DecodeWindowSpan> = RefCell::new(DecodeWindowSpan::EMPTY);
}

#[derive(Clone, Copy)]
struct DecodeWindowSpan {
    parent_samples: usize,
    sample_rate: usize,
    origin: usize,
    span: usize,
}

impl DecodeWindowSpan {
    const EMPTY: Self = Self {
        parent_samples: 0,
        sample_rate: 0,
        origin: 0,
        span: 0,
    };
}

/// Fires on this thread before each encoder window (`index` is 0-based).
/// Dropping the guard clears the callback and any slice span so later jobs stay quiet.
pub struct DecodeProgressGuard;

impl Drop for DecodeProgressGuard {
    fn drop(&mut self) {
        DECODE_PROGRESS.with(|slot| *slot.borrow_mut() = None);
        clear_decode_window_span();
    }
}

pub fn set_decode_progress(
    cb: Arc<dyn Fn(usize, usize) + Send + Sync>,
) -> DecodeProgressGuard {
    DECODE_PROGRESS.with(|slot| *slot.borrow_mut() = Some(cb));
    DecodeProgressGuard
}

fn report_decode_progress(index: usize, total: usize) {
    DECODE_PROGRESS.with(|slot| {
        if let Some(cb) = slot.borrow().clone() {
            cb(index, total);
        }
    });
}

fn clear_decode_window_span() {
    DECODE_WINDOW_SPAN.with(|slot| *slot.borrow_mut() = DecodeWindowSpan::EMPTY);
}

/// Later `waveform_to_text` calls on this thread are slices of one buffer.
/// Window indexes then keep climbing instead of restarting at 0 on each slice.
/// Drop the guard when that buffer is finished.
pub struct DecodeWindowSpanGuard;

impl Drop for DecodeWindowSpanGuard {
    fn drop(&mut self) {
        clear_decode_window_span();
    }
}

pub fn begin_decode_window_span(
    parent_samples: usize,
    sample_rate: usize,
) -> DecodeWindowSpanGuard {
    DECODE_WINDOW_SPAN.with(|slot| {
        *slot.borrow_mut() = DecodeWindowSpan {
            parent_samples,
            sample_rate,
            origin: 0,
            span: 0,
        };
    });
    DecodeWindowSpanGuard
}

/// `(origin, total)` for this slice. `total` is the parent window count when a span is active.
fn decode_window_report_bounds(
    local_len: usize,
    window_samples: usize,
    sample_rate: usize,
) -> (usize, usize) {
    DECODE_WINDOW_SPAN.with(|slot| {
        let mut span = slot.borrow_mut();
        if span.parent_samples == 0 || span.sample_rate != sample_rate {
            return (0, mel_window_count(local_len, window_samples, sample_rate));
        }
        if span.span == 0 {
            span.span = mel_window_count(span.parent_samples, window_samples, sample_rate).max(1);
        }
        let total = span.span.max(1);
        let origin = span.origin.min(total.saturating_sub(1));
        (origin, total)
    })
}

fn progress_index(origin: usize, offset: usize, total: usize) -> usize {
    let total = total.max(1);
    origin.saturating_add(offset).min(total - 1)
}

/// One clamped index per local encoder window. Does not advance the span;
/// call `finish_decode_window_slice` after the slice succeeds.
fn slice_progress_indexes(
    local_len: usize,
    window_samples: usize,
    sample_rate: usize,
) -> (Vec<usize>, usize) {
    let (origin, total) = decode_window_report_bounds(local_len, window_samples, sample_rate);
    let total = total.max(1);
    let n = mel_window_count(local_len, window_samples, sample_rate);
    let indexes = (0..n).map(|i| progress_index(origin, i, total)).collect();
    (indexes, total)
}

fn finish_decode_window_slice(local_windows: usize) {
    DECODE_WINDOW_SPAN.with(|slot| {
        let mut span = slot.borrow_mut();
        if span.parent_samples == 0 {
            return;
        }
        span.origin = span.origin.saturating_add(local_windows);
    });
}

use indicatif::{ProgressBar, ProgressStyle};

/// Decode a long buffer as short slices. `search_expand_sec` is 0.5, so a chunk
/// can run that far past `max_chunk_sec`. Callers must pick a cap under the
/// encoder context.
fn decode_in_time_chunks(
    waveform: &[f32],
    sample_rate: usize,
    max_chunk_sec: f64,
    mut decode_chunk: impl FnMut(&[f32]) -> String,
) -> (String, Vec<TranscriptSegment>) {
    let chunks = crate::audio::qwen3_chunk::split_audio_into_chunks(
        waveform,
        sample_rate,
        max_chunk_sec,
        0.5,
        100.0,
        0.0,
    );
    let mut parts = Vec::new();
    let mut segments = Vec::new();
    for (chunk, offset) in chunks {
        let secs = chunk.len() as f64 / sample_rate.max(1) as f64;
        let text = trim_trailing_hallucination(&clean_text(&decode_chunk(&chunk)));
        if text.is_empty() {
            continue;
        }
        parts.push(text.clone());
        segments.push(TranscriptSegment {
            start: offset,
            end: offset + secs,
            text,
            diarization: None,
            location: None,
        });
    }
    let final_text = parts.join(" ");
    if segments.is_empty() {
        let duration = waveform.len() as f64 / sample_rate.max(1) as f64;
        segments.push(TranscriptSegment {
            start: 0.0,
            end: duration,
            text: final_text.clone(),
            diarization: None,
            location: None,
        });
    }
    (final_text, segments)
}

pub fn waveform_to_text(
    model: &Model,
    model_name: &str,
    bpe: &Gpt2Tokenizer,
    lang: Language,
    language_code: &str,
    waveform: Vec<f32>,
    stereo_audio: Option<Vec<[f32; 2]>>,
    sample_rate: usize,
    streaming_mode: bool,
    verbose: bool,
    debug: bool,
    quiet: bool,
    include_timestamps: bool,
    backend_info: String,
    beam_size: usize,
    max_tokens: usize,
    padding: usize,
    chunk_size_samples: Option<usize>,
    decode_task: Option<DecodeTask>,
) -> token::Result<(String, Vec<TranscriptSegment>)> {
    let decode_task = decode_task.unwrap_or(DecodeTask::Transcribe);

    if let Model::Moonshine(m) = model {
        // Encoder context is 30s (480_000 samples at 16 kHz). A 120s GPU slice
        // or a 600s server segment encoded in one shot comes back empty or looped.
        let (final_text, segments) =
            decode_in_time_chunks(&waveform, sample_rate, 29.0, |chunk| {
                decode_moonshine(m, bpe, chunk, sample_rate, model_name, verbose).0
            });
        return Ok((final_text, segments));
    }

    if let Model::TONE(m) = model {
        let (text, _tokens) = decode_tone(m, bpe, &waveform, sample_rate, model_name, verbose);
        let final_text = trim_trailing_hallucination(&clean_text(&text));
        let duration = waveform.len() as f64 / sample_rate as f64;
        let segments = vec![TranscriptSegment {
            start: 0.0,
            end: duration,
            text: final_text.clone(),
            diarization: None,
            location: None,
        }];
        return Ok((final_text, segments));
    }

    if let Model::VibeVoice(m) = model {
        // The acoustic encoder consumes the whole buffer. A 600s segment of the
        // 7B model is one activation pass and has been enough to be killed.
        let (final_text, segments) =
            decode_in_time_chunks(&waveform, sample_rate, 30.0, |chunk| {
                decode_vibevoice(m, bpe, chunk, sample_rate, model_name, verbose).0
            });
        return Ok((final_text, segments));
    }

    if let Model::Qwen3(m) = model {
        use burn::tensor::Tensor;
        use crate::audio::prep_audio;
        use crate::audio::qwen3_chunk::{
            split_audio_into_chunks, QWEN3_MAX_ASR_CHUNK_SEC, QWEN3_MIN_ASR_CHUNK_SEC,
        };
        use crate::model::qwen3::{qwen3_forced_language_from_code, QWEN3_DEFAULT_MAX_NEW_TOKENS};

        let device = m.device();
        let n_mels = m.encoder_mel_size();
        let forced_language = qwen3_forced_language_from_code(language_code);
        let _ = lang;

        let chunks = split_audio_into_chunks(
            &waveform,
            sample_rate,
            QWEN3_MAX_ASR_CHUNK_SEC,
            5.0,
            100.0,
            QWEN3_MIN_ASR_CHUNK_SEC,
        );
        let mut accumulated = String::new();
        let mut all_tokens: Vec<usize> = Vec::new();

        for (chunk_idx, (chunk_wav, _offset_sec)) in chunks.into_iter().enumerate() {
            if verbose && !quiet {
                println!(
                    "Qwen3: processing chunk {} ({} samples)",
                    chunk_idx + 1,
                    chunk_wav.len()
                );
            }
            let w = Tensor::< 1>::from_floats(chunk_wav.as_slice(), &device).unsqueeze();
            let mel = prep_audio(w, sample_rate as f64, n_mels, false, model_name);
            let qwen3_max_tokens = max_tokens.max(QWEN3_DEFAULT_MAX_NEW_TOKENS);
            let (text, tokens) = decode_qwen3(
                m,
                bpe,
                mel,
                model_name,
                forced_language,
                qwen3_max_tokens,
                verbose,
            );
            let cleaned = clean_text(&text);
            if !cleaned.is_empty() {
                accumulated.push_str(&cleaned);
            }
            all_tokens.extend(tokens);
        }

        let final_text = trim_trailing_hallucination(&accumulated);
        let duration = waveform.len() as f64 / sample_rate as f64;
        let segments = vec![TranscriptSegment {
            start: 0.0,
            end: duration,
            text: final_text.clone(),
            diarization: None,
            location: None,
        }];
        let _ = all_tokens;
        return Ok((final_text, segments));
    }

    let device = match model {
        Model::Whisper(m) => m.encoder.conv1.weight.device(),
        Model::Parakeet(m) => m.encoder.layers[0].final_ln.gamma.device(),
        Model::TONE(_) => unreachable!("t-one handled before mel loop"),
        Model::Qwen3(_) => unreachable!("qwen3 handled before mel loop"),
        Model::VibeVoice(m) => m.acoustic_connector.fc1.weight.device(),
        Model::Moonshine(_) => unreachable!("moonshine handled before mel loop"),
        Model::Nemotron(m) => m.device(),
    };

    let n_waveform_samples_per_window = chunk_size_samples.unwrap_or(match model {
        Model::Whisper(_) => max_waveform_samples(model.encoder_ctx_size() - padding),
        _ => model.encoder_ctx_size() * 160,
    });

    let is_nemo = matches!(model, Model::Parakeet(_));
    let n_mels = model.encoder_mel_size();

    if debug {
        println!("DEBUG: Waveform samples: {}, model: {}", waveform.len(), model_name);
    }

    // One spectrogram at a time. Collecting every window up front pinned a
    // multi-hour file (or a 10-minute server segment) in VRAM before decode
    // started and wgpu aborted with Out of Memory.
    let (progress_indexes, progress_total) = slice_progress_indexes(
        waveform.len(),
        n_waveform_samples_per_window,
        sample_rate,
    );
    let total_chunks = progress_indexes.len();
    let mel_iter = waveform_to_mel_tensor(waveform, sample_rate, n_waveform_samples_per_window, &device, n_mels, is_nemo, model_name);
    let (mean, std) = get_model_stats(model_name, &device);

    if debug { println!("DEBUG WAVEFORM_TO_TEXT: total_chunks: {}", total_chunks); }
    
    let pb = if !streaming_mode && !quiet {
        let pb = ProgressBar::new(total_chunks as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) | {per_sec} | ETA: {eta} | {msg}")
            .unwrap()
            .progress_chars("#>-"));
        pb.set_message(backend_info);
        pb.enable_steady_tick(std::time::Duration::from_secs(1));
        Some(pb)
    } else {
        None
    };

    let chunk_duration = n_waveform_samples_per_window as f64 / sample_rate as f64;
    let chunk_overlap = 3.0; 
    let shift_duration = chunk_duration - chunk_overlap;

    let mut global_last_end = 0.0;
    let mut recent_texts: std::collections::VecDeque<String> = std::collections::VecDeque::with_capacity(10);
    let mut tokens: Vec<usize> = Vec::new();
    let mut segments: Vec<TranscriptSegment> = Vec::new();
    let mut accumulated_text = String::new();

    for (i, mel) in mel_iter.enumerate() {
        if let Some(&index) = progress_indexes.get(i) {
            let previous = i
                .checked_sub(1)
                .and_then(|prev| progress_indexes.get(prev))
                .copied();
            if previous != Some(index) {
                report_decode_progress(index, progress_total);
            }
        }
        let mel = if let (Some(m), Some(s)) = (&mean, &std) {
            (mel - m.clone().unsqueeze()) / s.clone().unsqueeze()
        } else {
            mel
        };
        if verbose && !quiet {
            let msg = format!("Processing chunk {}/{}...", i + 1, total_chunks);
            if let Some(ref p) = pb { p.println(msg); } else { println!("{}", msg); }
        }

        let (new_text, new_tokens) = match model {
            Model::Whisper(m) => decode_whisper(
                m,
                bpe,
                lang,
                mel,
                padding,
                streaming_mode,
                include_timestamps,
                beam_size,
                max_tokens,
                decode_task,
                model_name,
            )?,
            Model::Parakeet(m) => decode_parakeet(m, bpe, mel, model_name, verbose),
            Model::TONE(_) => unreachable!("t-one handled before mel loop"),
            Model::Qwen3(_) => unreachable!("qwen3 handled before mel loop"),
            Model::VibeVoice(_) => unreachable!("vibevoice handled before mel loop"),
            Model::Moonshine(_) => unreachable!("moonshine handled before mel loop"),
            Model::Nemotron(m) => decode_nemotron(m, bpe, mel, model_name, language_code, verbose),
        };

        let cleaned = clean_text(&new_text);
        if !cleaned.is_empty() {
            if !accumulated_text.is_empty() && !cleaned.starts_with(' ') {
                accumulated_text.push(' ');
            }
            accumulated_text.push_str(&cleaned);
        }

        let chunk_offset = i as f64 * shift_duration;
        if include_timestamps {
            let mut current_start = chunk_offset.max(global_last_end);
            let mut current_text_tokens = Vec::new();
            for &token in &new_tokens {
                if let Some(ts) = bpe.token_to_timestamp(token) {
                    let absolute_ts = chunk_offset + ts;
                    if !current_text_tokens.is_empty() {
                        let text = bpe.decode(&current_text_tokens, true)?;
                        let trimmed = clean_text(&text);
                        if !trimmed.is_empty() {
                            let mut segment = TranscriptSegment {
                                start: current_start,
                                end: absolute_ts.max(current_start + 0.05),
                                text: trimmed,
                                diarization: None,
                                location: None,
                            };
                            if let Some(ref stereo) = stereo_audio {
                                let s_idx = (segment.start * sample_rate as f64) as usize;
                                let e_idx = (segment.end * sample_rate as f64).min(stereo.len() as f64) as usize;
                                if e_idx > s_idx {
                                    let mut sum_l = 0.0; let mut sum_r = 0.0;
                                    for j in s_idx..e_idx { sum_l += stereo[j][0].abs(); sum_r += stereo[j][1].abs(); }
                                    let pan = ((sum_r - sum_l) / (sum_l + sum_r + 1e-6)) as f64;
                                    segment.location = Some(pan);
                                    segment.diarization = Some(if pan < 0.0 { "Speaker 1".to_string() } else { "Speaker 2".to_string() });
                                }
                            }
                            if segment.end > segment.start {
                                if !recent_texts.contains(&segment.text) {
                                    if !quiet {
                                        let msg = format!("[{:.2}s - {:.2}s]: {}", segment.start, segment.end, segment.text);
                                        if let Some(ref p) = pb { p.println(msg); } else { println!("{}", msg); }
                                    }
                                    global_last_end = segment.end;
                                    if recent_texts.len() == 10 { recent_texts.pop_front(); }
                                    recent_texts.push_back(segment.text.clone());
                                    segments.push(segment);
                                }
                            }
                        }
                        current_text_tokens.clear();
                    }
                    current_start = absolute_ts.max(current_start);
                } else if !bpe.is_special(token) {
                    current_text_tokens.push(token);
                }
            }
        }

        if let Some((prev_index, curr_index)) = find_chunk_overlap(&tokens[..], &new_tokens[..], 40, 3) {
            tokens.truncate(prev_index);
            tokens.extend(&new_tokens[curr_index..]);
        } else {
            tokens.extend(new_tokens);
        }

        if !include_timestamps && !quiet && !new_text.trim().is_empty() {
            let msg = format!("[{:.2}s - {:.2}s]: {}", chunk_offset, chunk_offset + chunk_duration, new_text.trim());
            if let Some(ref p) = pb { p.println(msg); } else { println!("{}", msg); }
        }

        if let Some(ref p) = pb { p.inc(1); }
    }
    finish_decode_window_slice(total_chunks);

    if let Some(p) = pb { p.finish_with_message("Transcription complete."); }

    let final_text = if accumulated_text.is_empty() {
        trim_trailing_hallucination(&clean_text(&bpe.decode(&tokens[..], true)?))
    } else {
        trim_trailing_hallucination(&accumulated_text)
    };
    if segments.is_empty() {
        segments.push(TranscriptSegment {
            start: 0.0,
            end: (total_chunks as f64 * shift_duration) + chunk_overlap,
            text: final_text.clone(),
            diarization: None,
            location: None,
        });
    }

    Ok((final_text, segments))
}

#[cfg(test)]
mod progress_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn moonshine_chunks_stay_under_encoder_context() {
        let sr = 16_000usize;
        let wav = vec![0.2f32; sr * 120];
        let (text, segments) = decode_in_time_chunks(&wav, sr, 29.0, |chunk| {
            assert!(
                chunk.len() <= 480_000,
                "chunk of {} samples exceeds the 30s moonshine context",
                chunk.len()
            );
            "hello".into()
        });
        assert!(segments.len() > 1, "a 120s buffer must be more than one 29s slice");
        assert!(text.contains("hello"));
        let mut prev = -0.01;
        for seg in &segments {
            assert!(seg.start + 1e-6 >= prev);
            assert!(seg.end - seg.start <= 29.51);
            prev = seg.start;
        }
    }

    #[test]
    fn decode_progress_is_per_thread_and_cleared_on_drop() {
        let seen = Arc::new(AtomicUsize::new(0));
        let tick = seen.clone();
        let guard = set_decode_progress(Arc::new(move |index, total| {
            assert_eq!(total, 4);
            tick.fetch_add(index + 1, Ordering::SeqCst);
        }));
        report_decode_progress(0, 4);
        report_decode_progress(3, 4);
        assert_eq!(seen.load(Ordering::SeqCst), 5);
        drop(guard);
        report_decode_progress(1, 4);
        assert_eq!(seen.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn split_slices_keep_one_climbing_window_index() {
        let sr = 16_000;
        // Phonon / Parakeet encoder context is 1500 frames, 160 samples each (15s).
        let window = 240_000;
        let parent = sr * 600;
        let slice = sr * 120;
        let span = mel_window_count(parent, window, sr);
        let local = mel_window_count(slice, window, sr);
        assert!(local > 1);
        assert!(local < span);

        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let tick = seen.clone();
        let _progress = set_decode_progress(Arc::new(move |index, total| {
            assert_eq!(total, span);
            tick.lock().unwrap().push(index);
        }));
        let _span_guard = begin_decode_window_span(parent, sr);

        for _ in 0..2 {
            let (indexes, total) = slice_progress_indexes(slice, window, sr);
            assert_eq!(total, span);
            let mut last = None;
            for index in indexes {
                if last != Some(index) {
                    report_decode_progress(index, total);
                    last = Some(index);
                }
            }
            finish_decode_window_slice(local);
        }

        let got = seen.lock().unwrap().clone();
        assert_eq!(got.first().copied(), Some(0));
        assert!(
            got.windows(2).all(|pair| pair[1] > pair[0]),
            "window index restarted inside one segment: {got:?}"
        );
        assert_eq!(*got.last().unwrap(), local * 2 - 1);
        assert!(got.iter().all(|index| *index < span));
    }

    #[test]
    fn dropping_progress_guard_clears_a_slice_span() {
        let sr = 16_000;
        let window = 240_000;
        let parent = sr * 600;
        let slice = sr * 120;
        let progress = set_decode_progress(Arc::new(|_, _| {}));
        let leaked = begin_decode_window_span(parent, sr);
        std::mem::forget(leaked);

        let (origin, total) = decode_window_report_bounds(slice, window, sr);
        assert_eq!(origin, 0);
        assert_eq!(total, mel_window_count(parent, window, sr));

        drop(progress);
        let (origin, total) = decode_window_report_bounds(slice, window, sr);
        assert_eq!(origin, 0);
        assert_eq!(total, mel_window_count(slice, window, sr));
    }
}
