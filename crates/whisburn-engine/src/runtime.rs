use std::collections::{HashMap, HashSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Once, Weak};
use std::thread;
use std::time::Duration;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{
    wgpu::{Wgpu, WgpuDevice, WgpuRuntime},
    NdArray,
};
use burn::tensor::backend::Backend;
use cubecl::prelude::Runtime;
use whisburn_core::{
    SpeechTask, TaskOptions, TranscriptResult, TranscriptSegment, WhisburnError, WhisburnResult,
};

use crate::cli::{get_backend_info, get_device, parse_language, CommonArgs};
use crate::model::diarize::{is_diarize_model, run_diarize};
use crate::model::load_model;
use crate::model::registry::find_model;
use crate::model::vad::{is_vad_model, run_vad};
use crate::model::Model;
use crate::orchestrator::Orchestrator;
use crate::token::{Gpt2Tokenizer, Language};
use crate::audio::qwen3_chunk::split_audio_into_chunks;
use crate::transcribe::{waveform_to_text, DecodeTask};
use crate::vram::{
    looks_like_gpu_failure, plan_gpu_audio, probe, process_vram_mb, GpuAudioPlan, VramPolicy,
};

struct LoadedModel<B: Backend> {
    bpe: Gpt2Tokenizer,
    model: Model<B>,
}

struct GpuBackend {
    device: WgpuDevice,
    cache: HashMap<String, Arc<LoadedModel<Wgpu>>>,
}

struct CpuBackend {
    device: NdArrayDevice,
    cache: HashMap<String, Arc<LoadedModel<NdArray>>>,
}

struct RuntimeState {
    gpu: Option<GpuBackend>,
    cpu: CpuBackend,
    prefer_gpu: bool,
    device_arg: Option<String>,
    warmed: HashSet<String>,
}

struct RuntimeInner {
    state: Mutex<RuntimeState>,
    policy: VramPolicy,
    verbose: bool,
    debug: bool,
    in_flight: AtomicBool,
    poisoned: AtomicBool,
    stop: AtomicBool,
}

pub struct InferenceRuntime {
    inner: Arc<RuntimeInner>,
}

fn summarize_on<B: Backend>(
    model_name: &str,
    device: &B::Device,
    text: &str,
    verbose: bool,
    progress: Option<crate::model::qwen3::summarize::SummarizeProgress>,
) -> WhisburnResult<String> {
    let _ = verbose;
    tracing::info!(model = model_name, chars = text.len(), "loading offline summarizer");
    if crate::model::granite::is_granite_model(model_name) {
        return crate::model::granite::summarize_text::<B>(
            model_name, device, text, verbose, progress,
        );
    }
    let (tokenizer, thinker) =
        crate::model::qwen3::summarize::load_summarizer::<B>(model_name, device, verbose)?;
    tracing::info!(model = model_name, "summarizer ready");
    crate::model::qwen3::summarize::summarize_with_thinker(
        &thinker,
        &tokenizer,
        text,
        device,
        progress,
    )
}

fn init_cpu_pool() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        match rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
        {
            Ok(()) => tracing::info!(threads, "NdArray CPU pool (rayon + SIMD kernels)"),
            Err(_) => tracing::debug!(
                threads = rayon::current_num_threads(),
                "rayon pool already started"
            ),
        }
    });
}

fn wants_cpu(device: &Option<String>) -> bool {
    device
        .as_deref()
        .is_some_and(|s| s.eq_ignore_ascii_case("cpu"))
}

fn lock_state(inner: &RuntimeInner) -> MutexGuard<'_, RuntimeState> {
    inner.state.lock().unwrap_or_else(|e| e.into_inner())
}

fn model_vram_mb(model_name: &str) -> u32 {
    find_model(model_name).map(|m| m.vram_mb).unwrap_or(0)
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic during inference".into()
    }
}

fn cleanup_wgpu(device: &WgpuDevice) {
    let client = WgpuRuntime::client(device);
    client.memory_cleanup();
}

fn drop_gpu(state: &mut RuntimeState) {
    if let Some(gpu) = state.gpu.take() {
        let names: Vec<String> = gpu.cache.keys().cloned().collect();
        drop(gpu.cache);
        cleanup_wgpu(&gpu.device);
        drop(gpu.device);
        if !names.is_empty() {
            tracing::warn!(
                models = ?names,
                "released GPU model cache to free VRAM"
            );
        }
    }
    state.warmed.clear();
}

/// cubecl keeps the wgpu device in a process-global client. After a worker-thread
/// OOM, dropping our handles often leaves the allocation with the driver.
/// Exit so systemd restarts the unit and the driver reclaims the card.
fn exit_if_process_still_holds_vram() {
    let still_held = |mb: u32| mb > 256;
    let Some(first) = process_vram_mb().filter(|mb| still_held(*mb)) else {
        return;
    };
    // memory_cleanup is asynchronous. A second sample separates a freed pool
    // from a wgpu device that survived the worker-thread OOM.
    thread::sleep(Duration::from_millis(750));
    let Some(mb) = process_vram_mb().filter(|mb| still_held(*mb)) else {
        return;
    };
    tracing::error!(
        mb,
        was = first,
        "this process still holds GPU memory after release; exiting so the driver can reclaim it"
    );
    #[cfg(not(test))]
    std::process::exit(1);
}

struct InFlight<'a>(&'a AtomicBool);

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

impl InferenceRuntime {
    pub fn new(device: Option<String>, verbose: bool, debug: bool) -> Self {
        Self::new_with_policy(device, verbose, debug, VramPolicy::from_env())
    }

    pub fn new_with_policy(
        device: Option<String>,
        verbose: bool,
        debug: bool,
        policy: VramPolicy,
    ) -> Self {
        let prefer_gpu = !wants_cpu(&device);
        init_cpu_pool();
        let inner = Arc::new(RuntimeInner {
            state: Mutex::new(RuntimeState {
                gpu: None,
                cpu: CpuBackend {
                    device: NdArrayDevice::Cpu,
                    cache: HashMap::new(),
                },
                prefer_gpu,
                device_arg: device,
                warmed: HashSet::new(),
            }),
            policy,
            verbose,
            debug,
            in_flight: AtomicBool::new(false),
            poisoned: AtomicBool::new(false),
            stop: AtomicBool::new(false),
        });
        #[cfg(not(test))]
        spawn_vram_guard(Arc::downgrade(&inner));
        Self { inner }
    }

    pub fn policy(&self) -> VramPolicy {
        self.inner.policy
    }

    pub fn backend_name(&self) -> String {
        let state = lock_state(&self.inner);
        if let Some(gpu) = &state.gpu {
            get_backend_info(&gpu.device)
        } else if state.prefer_gpu {
            "NdArray (CPU offload)".to_string()
        } else {
            "NdArray (CPU)".to_string()
        }
    }

    /// Drop GPU tensors and cubecl pools so the compositor can allocate again.
    pub fn release_gpu(&self) {
        let mut state = lock_state(&self.inner);
        drop_gpu(&mut state);
        drop(state);
        self.inner.poisoned.store(false, Ordering::SeqCst);
        exit_if_process_still_holds_vram();
    }

    fn begin_job(&self) -> InFlight<'_> {
        self.inner.in_flight.store(true, Ordering::SeqCst);
        InFlight(&self.inner.in_flight)
    }

    fn should_use_cpu(&self, extra_needed_mb: u32) -> bool {
        if self.inner.poisoned.load(Ordering::SeqCst) {
            return true;
        }
        let state = lock_state(&self.inner);
        if !state.prefer_gpu {
            return true;
        }
        drop(state);
        self.inner
            .policy
            .should_offload(&probe(), extra_needed_mb)
    }

    fn maybe_offload_idle(&self) {
        if self.inner.in_flight.load(Ordering::SeqCst) {
            return;
        }
        if self.should_use_cpu(0) {
            tracing::warn!(
                policy = ?self.inner.policy,
                snap = ?probe(),
                "VRAM below reserve; offloading to system RAM"
            );
            self.release_gpu();
        }
    }

    /// Offline English summary. The speech model is unloaded first, and the text
    /// model always runs on NdArray. An 8 GB card cannot hold Whisper medium and
    /// a summarizer at the same time; a wgpu OOM there is fatal to the device thread.
    pub fn summarize_text(
        &self,
        text: &str,
        model_name: &str,
        progress: Option<crate::model::qwen3::summarize::SummarizeProgress>,
    ) -> WhisburnResult<String> {
        let _job = self.begin_job();
        self.release_gpu();
        {
            let mut state = lock_state(&self.inner);
            state.cpu.cache.clear();
            state.warmed.clear();
        }
        tracing::info!(
            model = model_name,
            chars = text.len(),
            "summarizer loading on CPU after unloading the speech model"
        );
        let device = {
            let state = lock_state(&self.inner);
            state.cpu.device
        };
        let verbose = self.inner.verbose;
        recover_gpu_failure(
            catch_unwind(AssertUnwindSafe(|| {
                summarize_on::<NdArray>(model_name, &device, text, verbose, progress)
            })),
            &self.inner,
        )
    }

    pub fn loaded_models(&self) -> Vec<String> {
        let state = lock_state(&self.inner);
        let mut names: Vec<String> = state.cpu.cache.keys().cloned().collect();
        if let Some(gpu) = &state.gpu {
            names.extend(gpu.cache.keys().cloned());
        }
        names.sort();
        names.dedup();
        names
    }

    pub fn warm_model(&self, model_name: &str) -> WhisburnResult<()> {
        {
            let state = lock_state(&self.inner);
            if state.warmed.contains(model_name) {
                return Ok(());
            }
        }
        if is_vad_model(model_name) || is_diarize_model(model_name) {
            lock_state(&self.inner)
                .warmed
                .insert(model_name.to_string());
            return Ok(());
        }
        let _job = self.begin_job();
        let extra = model_vram_mb(model_name);
        if self.should_use_cpu(extra) {
            self.release_gpu();
            self.ensure_cpu_cached(model_name)?;
        } else {
            match catch_unwind(AssertUnwindSafe(|| self.ensure_gpu_cached(model_name))) {
                Ok(Ok(())) => {}
                Ok(Err(e)) if looks_like_gpu_failure(&e.to_string()) => {
                    tracing::error!(error = %e, model = model_name, "GPU load failed; warming on CPU");
                    self.release_gpu();
                    self.ensure_cpu_cached(model_name)?;
                }
                Ok(Err(e)) => {
                    self.cleanup_gpu_pools();
                    return Err(e);
                }
                Err(payload) => {
                    self.inner.poisoned.store(true, Ordering::SeqCst);
                    tracing::error!(
                        panic = %panic_message(payload),
                        model = model_name,
                        "panic while loading GPU model; warming on CPU"
                    );
                    self.release_gpu();
                    self.ensure_cpu_cached(model_name)?;
                }
            }
        }
        lock_state(&self.inner)
            .warmed
            .insert(model_name.to_string());
        self.maybe_offload_idle();
        Ok(())
    }

    fn ensure_gpu_device(&self) -> WgpuDevice {
        let mut state = lock_state(&self.inner);
        if state.gpu.is_none() {
            state.gpu = Some(GpuBackend {
                device: get_device(&state.device_arg),
                cache: HashMap::new(),
            });
        }
        state.gpu.as_ref().expect("gpu inserted").device.clone()
    }

    fn ensure_gpu_cached(&self, model_name: &str) -> WhisburnResult<()> {
        let device = self.ensure_gpu_device();
        let mut state = lock_state(&self.inner);
        let gpu = state.gpu.as_mut().expect("gpu inserted");
        if gpu.cache.contains_key(model_name) {
            return Ok(());
        }
        gpu.cache.clear();
        drop(state);
        let (bpe, _config, model) = load_model::<Wgpu>(model_name, &device, self.inner.verbose)
            .map_err(|e| WhisburnError::Model(e.to_string()))?;
        let mut state = lock_state(&self.inner);
        let gpu = state.gpu.get_or_insert_with(|| GpuBackend {
            device: device.clone(),
            cache: HashMap::new(),
        });
        gpu.cache.insert(
            model_name.to_string(),
            Arc::new(LoadedModel { bpe, model }),
        );
        Ok(())
    }

    fn ensure_cpu_cached(&self, model_name: &str) -> WhisburnResult<()> {
        let device = {
            let state = lock_state(&self.inner);
            if state.cpu.cache.contains_key(model_name) {
                return Ok(());
            }
            state.cpu.device
        };
        let (bpe, _config, model) = load_model::<NdArray>(model_name, &device, self.inner.verbose)
            .map_err(|e| WhisburnError::Model(e.to_string()))?;
        let mut state = lock_state(&self.inner);
        state.cpu.cache.clear();
        state.cpu.cache.insert(
            model_name.to_string(),
            Arc::new(LoadedModel { bpe, model }),
        );
        Ok(())
    }

    fn cleanup_gpu_pools(&self) {
        let state = lock_state(&self.inner);
        if let Some(gpu) = &state.gpu {
            cleanup_wgpu(&gpu.device);
        }
    }

    pub fn transcribe_waveform(
        &self,
        model_name: &str,
        samples: Vec<f32>,
        sample_rate: usize,
        options: &TaskOptions,
    ) -> WhisburnResult<TranscriptResult> {
        if is_vad_model(model_name) {
            return run_vad(model_name, &samples, sample_rate)
                .map_err(|e| WhisburnError::Inference(e));
        }
        if is_diarize_model(model_name) {
            return run_diarize(model_name, &samples, sample_rate)
                .map_err(|e| WhisburnError::Inference(e));
        }

        let decode_task = decode_task_for(options.task);
        let _job = self.begin_job();

        if options.orchestrate && !self.should_use_cpu(model_vram_mb(model_name)) {
            let result = catch_unwind(AssertUnwindSafe(|| {
                self.transcribe_orchestrated(
                    model_name,
                    samples.clone(),
                    sample_rate,
                    options,
                    decode_task,
                )
            }));
            match recover_gpu_failure(result, &self.inner) {
                Ok(r) => {
                    self.maybe_offload_idle();
                    return Ok(r);
                }
                Err(e)
                    if looks_like_gpu_failure(&e.to_string())
                        || self.inner.poisoned.load(Ordering::SeqCst) =>
                {
                    tracing::error!(
                        error = %e,
                        "orchestrate GPU failure; falling back to single-pass CPU"
                    );
                    self.release_gpu();
                }
                Err(e) => {
                    self.cleanup_gpu_pools();
                    return Err(e);
                }
            }
        } else if options.orchestrate {
            tracing::warn!("orchestrate requested but VRAM reserve forces CPU; using single-pass");
        }

        let extra = model_vram_mb(model_name);
        let audio_sec = samples.len() as f64 / sample_rate.max(1) as f64;
        let (prefer_gpu, loaded) = {
            let state = lock_state(&self.inner);
            let loaded = state
                .gpu
                .as_ref()
                .is_some_and(|g| g.cache.contains_key(model_name));
            (state.prefer_gpu, loaded)
        };
        // An explicit CPU device stays on NdArray (rayon + SIMD). The VRAM plan
        // only chooses among GPU, split-on-GPU, and offload.
        let plan = if !prefer_gpu || self.inner.poisoned.load(Ordering::SeqCst) {
            GpuAudioPlan::Cpu
        } else {
            plan_gpu_audio(&self.inner.policy, &probe(), extra, audio_sec, loaded)
        };
        let result = match plan {
            GpuAudioPlan::Cpu => {
                self.release_gpu();
                self.transcribe_on_cpu(model_name, samples, sample_rate, options, decode_task)
            }
            GpuAudioPlan::Split { chunk_sec } => self.transcribe_split_gpu(
                model_name,
                samples,
                sample_rate,
                options,
                decode_task,
                chunk_sec,
            ),
            GpuAudioPlan::SinglePass => self.transcribe_gpu_or_cpu(
                model_name,
                samples,
                sample_rate,
                options,
                decode_task,
            ),
        };
        if result.is_err() {
            self.release_gpu();
        } else {
            self.maybe_offload_idle();
        }
        result
    }

    fn transcribe_on_gpu(
        &self,
        model_name: &str,
        samples: Vec<f32>,
        sample_rate: usize,
        options: &TaskOptions,
        decode_task: DecodeTask,
    ) -> WhisburnResult<TranscriptResult> {
        self.ensure_gpu_cached(model_name)?;
        let loaded = {
            let state = lock_state(&self.inner);
            state
                .gpu
                .as_ref()
                .and_then(|g| g.cache.get(model_name).cloned())
                .ok_or_else(|| WhisburnError::Model(format!("GPU model '{model_name}' missing")))?
        };
        run_waveform(
            &loaded.model,
            &loaded.bpe,
            model_name,
            samples,
            sample_rate,
            options,
            decode_task,
            self.backend_name(),
            self.inner.verbose,
            self.inner.debug,
        )
    }

    fn transcribe_on_cpu(
        &self,
        model_name: &str,
        samples: Vec<f32>,
        sample_rate: usize,
        options: &TaskOptions,
        decode_task: DecodeTask,
    ) -> WhisburnResult<TranscriptResult> {
        self.ensure_cpu_cached(model_name)?;
        let loaded = {
            let state = lock_state(&self.inner);
            state
                .cpu
                .cache
                .get(model_name)
                .cloned()
                .ok_or_else(|| WhisburnError::Model(format!("CPU model '{model_name}' missing")))?
        };
        run_waveform(
            &loaded.model,
            &loaded.bpe,
            model_name,
            samples,
            sample_rate,
            options,
            decode_task,
            "NdArray (CPU)".to_string(),
            self.inner.verbose,
            self.inner.debug,
        )
    }

    fn transcribe_gpu_or_cpu(
        &self,
        model_name: &str,
        samples: Vec<f32>,
        sample_rate: usize,
        options: &TaskOptions,
        decode_task: DecodeTask,
    ) -> WhisburnResult<TranscriptResult> {
        let gpu_result = catch_unwind(AssertUnwindSafe(|| {
            self.transcribe_on_gpu(
                model_name,
                samples.clone(),
                sample_rate,
                options,
                decode_task,
            )
        }));
        match recover_gpu_failure(gpu_result, &self.inner) {
            Ok(r) => Ok(r),
            Err(e) => {
                tracing::error!(error = %e, "GPU inference failed; retrying on CPU");
                self.release_gpu();
                recover_gpu_failure(
                    catch_unwind(AssertUnwindSafe(|| {
                        self.transcribe_on_cpu(
                            model_name,
                            samples,
                            sample_rate,
                            options,
                            decode_task,
                        )
                    })),
                    &self.inner,
                )
            }
        }
    }

    fn transcribe_split_gpu(
        &self,
        model_name: &str,
        samples: Vec<f32>,
        sample_rate: usize,
        options: &TaskOptions,
        decode_task: DecodeTask,
        chunk_sec: f64,
    ) -> WhisburnResult<TranscriptResult> {
        let chunks = split_audio_into_chunks(&samples, sample_rate, chunk_sec, 2.0, 100.0, 0.0);
        tracing::info!(
            chunks = chunks.len(),
            chunk_sec,
            seconds = samples.len() as f64 / sample_rate.max(1) as f64,
            "splitting long audio so GPU inference stays above the VRAM reserve"
        );
        drop(samples);
        let n = chunks.len();
        let mut parts = Vec::with_capacity(n);
        for (i, (chunk, offset)) in chunks.into_iter().enumerate() {
            if chunk.is_empty() {
                continue;
            }
            let r = self.transcribe_one_split_chunk(
                model_name,
                chunk,
                sample_rate,
                options,
                decode_task,
                i,
                n,
            )?;
            parts.push((offset, r));
            self.cleanup_gpu_pools();
        }
        Ok(merge_chunk_results(
            model_name,
            options.task.as_str(),
            Some(options.language.as_str().to_string()),
            parts,
        ))
    }

    fn transcribe_one_split_chunk(
        &self,
        model_name: &str,
        chunk: Vec<f32>,
        sample_rate: usize,
        options: &TaskOptions,
        decode_task: DecodeTask,
        index: usize,
        total: usize,
    ) -> WhisburnResult<TranscriptResult> {
        let gpu_result = catch_unwind(AssertUnwindSafe(|| {
            self.transcribe_on_gpu(model_name, chunk.clone(), sample_rate, options, decode_task)
        }));
        match recover_gpu_failure(gpu_result, &self.inner) {
            Ok(r) => Ok(r),
            Err(e) => {
                tracing::error!(
                    chunk = index + 1,
                    of = total,
                    error = %e,
                    "GPU chunk failed; transcribing this slice on CPU"
                );
                self.cleanup_gpu_pools();
                if self.inner.poisoned.load(Ordering::SeqCst) {
                    self.release_gpu();
                }
                recover_gpu_failure(
                    catch_unwind(AssertUnwindSafe(|| {
                        self.transcribe_on_cpu(model_name, chunk, sample_rate, options, decode_task)
                    })),
                    &self.inner,
                )
            }
        }
    }

    fn transcribe_orchestrated(
        &self,
        model_name: &str,
        samples: Vec<f32>,
        sample_rate: usize,
        options: &TaskOptions,
        decode_task: DecodeTask,
    ) -> WhisburnResult<TranscriptResult> {
        let scout = options.scout_model.as_deref().unwrap_or("tiny_en");

        let common = CommonArgs {
            model: model_name.to_string(),
            lang: options.language.as_str().to_string(),
            verbose: self.inner.verbose,
            debug: self.inner.debug,
            quiet: true,
            verify: false,
            hf_token: None,
            beam_size: options.beam_size,
            max_tokens: options.max_tokens,
            padding: options.padding,
            device: None,
        };

        let device = self.ensure_gpu_device();
        let orchestrator = Orchestrator::new(device, self.inner.verbose, self.inner.debug);
        let (text, segments) = orchestrator
            .run_waveform(samples, sample_rate, common, scout, decode_task)
            .map_err(|e| WhisburnError::Inference(e.to_string()))?;

        Ok(TranscriptResult {
            text,
            segments: segments
                .into_iter()
                .map(|s| TranscriptSegment {
                    start: s.start,
                    end: s.end,
                    text: s.text,
                    speaker: s.diarization,
                })
                .collect(),
            language: Some(options.language.as_str().to_string()),
            model: model_name.to_string(),
            task: options.task.as_str().to_string(),
        })
    }
}

impl Drop for InferenceRuntime {
    fn drop(&mut self) {
        self.inner.stop.store(true, Ordering::SeqCst);
        if Arc::strong_count(&self.inner) == 1 {
            self.release_gpu();
        }
    }
}

fn merge_chunk_results(
    model: &str,
    task: &str,
    language: Option<String>,
    parts: Vec<(f64, TranscriptResult)>,
) -> TranscriptResult {
    let mut text = String::new();
    let mut segments = Vec::new();
    for (offset, part) in parts {
        if !part.text.is_empty() {
            if !text.is_empty() && !text.ends_with(char::is_whitespace) {
                text.push(' ');
            }
            text.push_str(part.text.trim());
        }
        for mut seg in part.segments {
            seg.start += offset;
            seg.end += offset;
            segments.push(seg);
        }
    }
    TranscriptResult {
        text,
        segments,
        language,
        model: model.to_string(),
        task: task.to_string(),
    }
}

fn recover_gpu_failure<T>(
    result: Result<WhisburnResult<T>, Box<dyn std::any::Any + Send>>,
    inner: &RuntimeInner,
) -> WhisburnResult<T> {
    match result {
        Ok(inner_result) => inner_result,
        Err(payload) => {
            inner.poisoned.store(true, Ordering::SeqCst);
            let msg = panic_message(payload);
            tracing::error!(panic = %msg, "GPU worker panicked; releasing VRAM");
            drop_gpu(&mut lock_state(inner));
            Err(WhisburnError::Inference(msg))
        }
    }
}

fn run_waveform<B: Backend>(
    model: &Model<B>,
    bpe: &Gpt2Tokenizer,
    model_name: &str,
    samples: Vec<f32>,
    sample_rate: usize,
    options: &TaskOptions,
    decode_task: DecodeTask,
    backend_info: String,
    verbose: bool,
    debug: bool,
) -> WhisburnResult<TranscriptResult> {
    let lang = map_language(options.language.as_str());
    let (text, segments) = waveform_to_text(
        model,
        model_name,
        bpe,
        lang,
        options.language.as_str(),
        samples,
        None,
        sample_rate,
        false,
        verbose,
        debug,
        true,
        options.include_timestamps,
        backend_info,
        options.beam_size,
        options.max_tokens,
        options.padding,
        None,
        Some(decode_task),
    )
    .map_err(|e| WhisburnError::Inference(e.to_string()))?;

    Ok(TranscriptResult {
        text,
        segments: segments
            .into_iter()
            .map(|s| TranscriptSegment {
                start: s.start,
                end: s.end,
                text: s.text,
                speaker: s.diarization,
            })
            .collect(),
        language: Some(options.language.as_str().to_string()),
        model: model_name.to_string(),
        task: options.task.as_str().to_string(),
    })
}

#[cfg_attr(test, allow(dead_code))]
fn spawn_vram_guard(inner: Weak<RuntimeInner>) {
    let _ = thread::Builder::new()
        .name("whisburn-vram-guard".into())
        .spawn(move || loop {
            thread::sleep(Duration::from_secs(2));
            let Some(inner) = inner.upgrade() else {
                break;
            };
            if inner.stop.load(Ordering::SeqCst) {
                break;
            }
            if inner.in_flight.load(Ordering::SeqCst) {
                continue;
            }
            let poisoned = inner.poisoned.load(Ordering::SeqCst);
            let snap = probe();
            if poisoned || inner.policy.should_offload(&snap, 0) {
                if poisoned {
                    tracing::error!("GPU panic flag set; releasing VRAM");
                } else {
                    tracing::warn!(?snap, "idle VRAM below reserve; releasing GPU tensors");
                }
                drop_gpu(&mut lock_state(&inner));
                inner.poisoned.store(false, Ordering::SeqCst);
                exit_if_process_still_holds_vram();
            }
        });
}

fn decode_task_for(task: SpeechTask) -> DecodeTask {
    match task {
        SpeechTask::Translate => DecodeTask::Translate,
        _ => DecodeTask::Transcribe,
    }
}

fn map_language(code: &str) -> Language {
    parse_language(code)
}

#[cfg(test)]
mod runtime_tests {
    use super::*;

    #[test]
    fn cpu_runtime_release_is_idempotent() {
        let rt = InferenceRuntime::new_with_policy(
            Some("cpu".into()),
            false,
            false,
            VramPolicy::disabled(),
        );
        assert_eq!(rt.backend_name(), "NdArray (CPU)");
        rt.release_gpu();
        rt.release_gpu();
        assert!(rt.loaded_models().is_empty());
    }

    #[test]
    fn cpu_pool_matches_available_cores() {
        init_cpu_pool();
        let want = std::thread::available_parallelism().unwrap().get();
        assert_eq!(rayon::current_num_threads(), want);
    }

    #[test]
    fn merge_chunk_results_offsets_segments() {
        let a = TranscriptResult {
            text: "hello".into(),
            segments: vec![TranscriptSegment {
                start: 0.0,
                end: 1.5,
                text: "hello".into(),
                speaker: None,
            }],
            language: Some("en".into()),
            model: "x".into(),
            task: "transcribe".into(),
        };
        let b = TranscriptResult {
            text: "world".into(),
            segments: vec![TranscriptSegment {
                start: 0.2,
                end: 1.0,
                text: "world".into(),
                speaker: None,
            }],
            language: Some("en".into()),
            model: "x".into(),
            task: "transcribe".into(),
        };
        let merged = merge_chunk_results("x", "transcribe", Some("en".into()), vec![(0.0, a), (15.0, b)]);
        assert_eq!(merged.text, "hello world");
        assert_eq!(merged.segments.len(), 2);
        assert!((merged.segments[1].start - 15.2).abs() < 1e-9);
        assert!((merged.segments[1].end - 16.0).abs() < 1e-9);
    }
}