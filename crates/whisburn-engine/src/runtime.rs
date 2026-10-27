use burn::tensor::{Device, DeviceKind};
use std::collections::{HashMap, HashSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Once, Weak};
use std::thread;
use std::time::{Duration, Instant};

use whisburn_core::{
    SpeechTask, TaskOptions, TranscriptResult, TranscriptSegment, WhisburnError, WhisburnResult,
};

use crate::audio::qwen3_chunk::split_audio_into_chunks;
use crate::cli::{get_backend_info, get_device, parse_language, CommonArgs};
use crate::model::diarize::{is_diarize_model, run_diarize};
use crate::model::load_model;
use crate::model::registry::find_model;
use crate::model::vad::{is_vad_model, run_vad};
use crate::model::Model;
use crate::orchestrator::Orchestrator;
use crate::token::{Gpt2Tokenizer, Language};
use crate::transcribe::{waveform_to_text, DecodeTask};
use crate::idle::IdleUnloadPolicy;
use crate::placement::{
    admit_inference, plan_summarize_handoff, ram_available_mb, tier_of, HostBudget, ResidentModel,
    Tier, TierMove,
};
use crate::vram::{
    activation_mb, compositor_exit_required, looks_like_gpu_failure, plan_gpu_audio, probe,
    process_vram_mb, window_workspace_mb, GpuAudioPlan, VramPolicy, MAX_GPU_CHUNK_SEC,
    MIN_GPU_CHUNK_SEC,
};

struct LoadedModel {
    bpe: Gpt2Tokenizer,
    model: Model,
}

struct GpuBackend {
    device: Device,
    cache: HashMap<String, Arc<LoadedModel>>,
}

struct CpuBackend {
    device: Device,
    cache: HashMap<String, Arc<LoadedModel>>,
}

struct CachedSummarizer {
    tokenizer: Gpt2Tokenizer,
    thinker: crate::model::qwen3::Qwen3Thinker,
}

struct GpuSummarizer {
    tokenizer: Gpt2Tokenizer,
    thinker: crate::model::qwen3::Qwen3Thinker,
}

struct RuntimeState {
    gpu: Option<GpuBackend>,
    cpu: CpuBackend,
    prefer_gpu: bool,
    device_arg: Option<String>,
    warmed: HashSet<String>,
    /// Kept after the first summary. Reloading qwen3-0.6b peaked near 6 GB
    /// and earlyoom's swap is already full, so a second load can kill the server.
    summarizer: Option<(String, CachedSummarizer)>,
    /// Summarizer promoted to VRAM. Dropped before the next speech model loads.
    summarizer_vram: Option<(String, GpuSummarizer)>,
}

struct RuntimeInner {
    state: Mutex<RuntimeState>,
    policy: VramPolicy,
    idle: IdleUnloadPolicy,
    /// Wall time of the last weight load into device memory. `None` until measured.
    last_load: Mutex<Option<Duration>>,
    /// Set when `jobs` drops to zero. Cleared when a new job starts.
    idle_since: Mutex<Option<Instant>>,
    /// Lets unit tests pretend a model is resident without loading weights.
    #[cfg(test)]
    idle_probe: AtomicBool,
    verbose: bool,
    debug: bool,
    /// Nested transcribe, summarize, and HTTP-request holds. The vram guard
    /// must not exit while this is above zero.
    jobs: AtomicUsize,
    /// Forward passes currently on the GPU. Admission will not demote a VRAM
    /// resident while this is above zero.
    gpu_inflight: AtomicUsize,
    poisoned: AtomicBool,
    /// Permanent: wgpu could not open an adapter. Cleared poison must not retry it.
    gpu_unavailable: AtomicBool,
    stop: AtomicBool,
}

pub struct InferenceRuntime {
    inner: Arc<RuntimeInner>,
}

fn summarize_on(
    model_name: &str,
    device: &Device,
    text: &str,
    verbose: bool,
    progress: Option<crate::model::qwen3::summarize::SummarizeProgress>,
) -> WhisburnResult<String> {
    let _ = verbose;
    tracing::info!(
        model = model_name,
        chars = text.len(),
        "loading offline summarizer"
    );
    if crate::model::granite::is_granite_model(model_name) {
        return crate::model::granite::summarize_text(
            model_name, device, text, verbose, progress,
        );
    }
    let (tokenizer, thinker) =
        crate::model::qwen3::summarize::load_summarizer(model_name, device, verbose)?;
    tracing::info!(model = model_name, "summarizer ready");
    crate::model::qwen3::summarize::summarize_with_thinker(
        &thinker, &tokenizer, text, device, progress,
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
            Ok(()) => tracing::info!(threads, "Flex CPU pool (rayon + SIMD kernels)"),
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

fn cleanup_wgpu(device: &Device) {
    // A failed request_adapter leaves no client. Touching it panics the cubecl
    // worker a second time and aborts the CPU fallback.
    if crate::gpu_env::adapter_missing() {
        tracing::warn!("skipping wgpu cleanup; Vulkan exposed no adapter");
        return;
    }
    let device = device.clone();
    if let Err(payload) = catch_unwind(AssertUnwindSafe(|| {
        device.memory_cleanup();
    })) {
        tracing::error!(
            panic = %panic_message(payload),
            "wgpu cleanup panicked; leaving the device for process exit"
        );
    }
}

/// glibc keeps a freed model's high-water mark. Return it before the next
/// checkpoint is read, or the two peaks stack and earlyoom kills the process.
fn trim_allocator() {
    #[cfg(target_os = "linux")]
    unsafe {
        extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        let _ = malloc_trim(0);
    }
}

fn drop_gpu(state: &mut RuntimeState) {
    state.summarizer_vram.take();
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

/// cubecl keeps the wgpu device in a process-global client. Dropping our handles
/// can leave the allocation with the driver. Exit only after the response is out,
/// and only when this process is what starved the compositor.
fn exit_if_compositor_starved(inner: &RuntimeInner) {
    if inner.jobs.load(Ordering::SeqCst) > 0 {
        return;
    }
    let first_held = process_vram_mb();
    let first_snap = probe();
    if !compositor_exit_required(0, first_held, &first_snap, &inner.policy) {
        return;
    }
    // memory_cleanup is asynchronous. A second sample separates a freed pool
    // from a wgpu device that survived the worker-thread OOM.
    thread::sleep(Duration::from_millis(750));
    let jobs = inner.jobs.load(Ordering::SeqCst);
    if jobs > 0 {
        return;
    }
    let held = process_vram_mb();
    let snap = probe();
    if !compositor_exit_required(jobs, held, &snap, &inner.policy) {
        return;
    }
    tracing::error!(
        mb = held,
        was = first_held,
        ?snap,
        "this process still holds GPU memory and the compositor reserve is gone; exiting so the driver can reclaim it"
    );
    #[cfg(not(test))]
    std::process::exit(1);
}

/// Keeps the vram guard from exiting for as long as the guard is alive.
pub struct JobGuard {
    inner: Arc<RuntimeInner>,
}

impl Drop for JobGuard {
    fn drop(&mut self) {
        let prev = self.inner.jobs.fetch_sub(1, Ordering::SeqCst);
        if prev == 1 {
            if let Ok(mut idle) = self.inner.idle_since.lock() {
                *idle = Some(Instant::now());
            }
        }
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
        Self::new_with_policies(
            device,
            verbose,
            debug,
            policy,
            IdleUnloadPolicy::from_env(),
        )
    }

    pub fn new_with_policies(
        device: Option<String>,
        verbose: bool,
        debug: bool,
        policy: VramPolicy,
        idle: IdleUnloadPolicy,
    ) -> Self {
        let prep = crate::gpu_env::prepare_gpu_runtime();
        crate::gpu_env::log_gpu_runtime();
        let loader_ready = matches!(prep, crate::gpu_env::GpuRuntimePrep::Ready { .. });
        let prefer_gpu = !wants_cpu(&device) && loader_ready;
        let gpu_unavailable = !loader_ready && !wants_cpu(&device);
        init_cpu_pool();
        let inner = Arc::new(RuntimeInner {
            state: Mutex::new(RuntimeState {
                gpu: None,
                cpu: CpuBackend {
                    device: Device::flex(),
                    cache: HashMap::new(),
                },
                prefer_gpu,
                device_arg: device,
                warmed: HashSet::new(),
                summarizer: None,
                summarizer_vram: None,
            }),
            policy,
            idle,
            last_load: Mutex::new(None),
            idle_since: Mutex::new(None),
            #[cfg(test)]
            idle_probe: AtomicBool::new(false),
            verbose,
            debug,
            jobs: AtomicUsize::new(0),
            gpu_inflight: AtomicUsize::new(0),
            poisoned: AtomicBool::new(false),
            gpu_unavailable: AtomicBool::new(gpu_unavailable),
            stop: AtomicBool::new(false),
        });
        #[cfg(not(test))]
        spawn_vram_guard(Arc::downgrade(&inner));
        Self { inner }
    }

    pub fn policy(&self) -> VramPolicy {
        self.inner.policy
    }

    pub fn idle_policy(&self) -> IdleUnloadPolicy {
        self.inner.idle
    }

    /// Seconds of the last measured weight load, if any.
    pub fn last_model_load_ms(&self) -> Option<u64> {
        self.inner
            .last_load
            .lock()
            .ok()
            .and_then(|g| g.map(|d| d.as_millis() as u64))
    }

    /// Effective idle window for the last load (or the unknown-load fallback).
    pub fn idle_timeout(&self) -> Option<Duration> {
        let last = self.inner.last_load.lock().ok().and_then(|g| *g);
        self.inner.idle.timeout_for(last)
    }

    pub fn backend_name(&self) -> String {
        let state = lock_state(&self.inner);
        if state.gpu.is_some() {
            get_backend_info(&state.device_arg)
        } else if state.prefer_gpu {
            "Flex (CPU offload)".to_string()
        } else {
            "Flex (CPU)".to_string()
        }
    }

    /// Drop GPU tensors. This does not exit: a web request still has to summarize
    /// and send its response while the guard in `begin_request` is held.
    pub fn release_gpu(&self) {
        if self.inner.gpu_inflight.load(Ordering::SeqCst) > 0 {
            tracing::warn!("leaving GPU resident; another job is in inference");
            return;
        }
        let mut state = lock_state(&self.inner);
        drop_gpu(&mut state);
        drop(state);
        self.inner.poisoned.store(false, Ordering::SeqCst);
        trim_allocator();
    }

    /// Drop every resident speech model and the cached summarizer.
    pub fn release_cached_models(&self) {
        release_cached_models(&self.inner);
    }

    fn note_weight_load(&self, elapsed: Duration) {
        if let Ok(mut slot) = self.inner.last_load.lock() {
            *slot = Some(elapsed);
        }
        // Preload and other loads outside a request should start the idle clock.
        // A load inside a job waits until the last JobGuard drops.
        if self.inner.jobs.load(Ordering::SeqCst) == 0 {
            if let Ok(mut idle) = self.inner.idle_since.lock() {
                *idle = Some(Instant::now());
            }
        }
        if let Some(timeout) = self.inner.idle.timeout_for(Some(elapsed)) {
            tracing::info!(
                load_ms = elapsed.as_millis() as u64,
                idle_secs = timeout.as_secs(),
                mode = self.inner.idle.mode_label(),
                "idle unload window set from weight load"
            );
        }
    }

    /// Unload cached models when the idle window has elapsed and no job is running.
    #[cfg(test)]
    fn poll_idle_unload(&self) -> bool {
        poll_idle_unload(&self.inner)
    }

    fn begin_job(&self) -> JobGuard {
        let inner = Arc::clone(&self.inner);
        let prev = inner.jobs.fetch_add(1, Ordering::SeqCst);
        if prev == 0 {
            if let Ok(mut idle) = inner.idle_since.lock() {
                *idle = None;
            }
        }
        JobGuard { inner }
    }

    /// Hold across an HTTP transcribe, including the gap before summarization
    /// and the send of the final event.
    pub fn begin_request(&self) -> JobGuard {
        self.begin_job()
    }

    fn should_use_cpu(&self, extra_needed_mb: u32) -> bool {
        if self.inner.gpu_unavailable.load(Ordering::SeqCst)
            || self.inner.poisoned.load(Ordering::SeqCst)
        {
            return true;
        }
        let state = lock_state(&self.inner);
        if !state.prefer_gpu {
            return true;
        }
        drop(state);
        self.inner.policy.should_offload(&probe(), extra_needed_mb)
    }

    fn maybe_offload_idle(&self) {
        if self.inner.jobs.load(Ordering::SeqCst) > 0 {
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

    fn host_budget(&self) -> HostBudget {
        let snap = probe();
        let total = snap.total_mb.unwrap_or(0);
        let known = snap.free_mb.is_some() || snap.used_mb.is_some();
        let free = snap.free_mb.or_else(|| {
            snap.used_mb
                .map(|used| total.saturating_sub(used))
        });
        HostBudget {
            vram_known: known && total > 0,
            vram_free_mb: free.unwrap_or(0),
            vram_reserve_mb: self.inner.policy.reserve_mb_for(total),
            ram_free_mb: ram_available_mb().unwrap_or(0),
        }
    }

    fn placement_residents(&self) -> Vec<ResidentModel> {
        let state = lock_state(&self.inner);
        let live = self.inner.gpu_inflight.load(Ordering::SeqCst) > 0;
        let mut out = Vec::new();
        if let Some(gpu) = &state.gpu {
            for name in gpu.cache.keys() {
                out.push(ResidentModel {
                    name: name.clone(),
                    tier: Tier::Vram,
                    weight_mb: model_vram_mb(name),
                    active: live,
                });
            }
        }
        if let Some((name, _)) = &state.summarizer_vram {
            if !out.iter().any(|m| m.name == *name) {
                out.push(ResidentModel {
                    name: name.clone(),
                    tier: Tier::Vram,
                    weight_mb: model_vram_mb(name),
                    active: live,
                });
            }
        }
        for name in state.cpu.cache.keys() {
            if out.iter().any(|m| m.name == *name && m.tier == Tier::Vram) {
                continue;
            }
            out.push(ResidentModel {
                name: name.clone(),
                tier: Tier::Ram,
                weight_mb: model_vram_mb(name),
                active: false,
            });
        }
        if let Some((name, _)) = &state.summarizer {
            if !out.iter().any(|m| m.name == *name) {
                out.push(ResidentModel {
                    name: name.clone(),
                    tier: Tier::Ram,
                    weight_mb: model_vram_mb(name),
                    active: false,
                });
            }
        }
        out
    }

    fn speech_resident(&self, summarizer: &str) -> Option<ResidentModel> {
        let residents = self.placement_residents();
        residents
            .iter()
            .find(|m| m.name != summarizer && m.tier == Tier::Vram)
            .cloned()
            .or_else(|| {
                residents
                    .into_iter()
                    .find(|m| m.name != summarizer)
            })
    }

    fn drop_named_to_disk(&self, name: &str) {
        let mut state = lock_state(&self.inner);
        if state
            .summarizer
            .as_ref()
            .is_some_and(|(n, _)| n == name)
        {
            state.summarizer.take();
        }
        if state
            .summarizer_vram
            .as_ref()
            .is_some_and(|(n, _)| n == name)
        {
            state.summarizer_vram.take();
        }
        state.cpu.cache.retain(|n, _| n != name);
        let gpu_hit = state
            .gpu
            .as_ref()
            .is_some_and(|g| g.cache.contains_key(name));
        if gpu_hit {
            self.drop_gpu_if_idle(&mut state);
        }
        drop(state);
        trim_allocator();
    }

    fn drop_gpu_if_idle(&self, state: &mut RuntimeState) {
        if self.inner.gpu_inflight.load(Ordering::SeqCst) > 0 {
            tracing::warn!("skipping GPU teardown; another job is in inference");
            return;
        }
        drop_gpu(state);
    }

    fn park_in_ram(&self, name: &str) -> WhisburnResult<()> {
        if crate::model::registry::find_model(name)
            .is_some_and(|m| m.category == crate::model::registry::ModelCategory::Llm)
        {
            // Granite has no Flex cache; qwen is staged here so the VRAM
            // copy can be dropped without losing the warm weights.
            self.ensure_summarizer_ram(name)?;
            let mut state = lock_state(&self.inner);
            if state
                .summarizer_vram
                .as_ref()
                .is_some_and(|(n, _)| n == name)
            {
                state.summarizer_vram.take();
            }
            let on_gpu = state
                .gpu
                .as_ref()
                .is_some_and(|g| g.cache.contains_key(name));
            if on_gpu {
                self.drop_gpu_if_idle(&mut state);
            }
            drop(state);
            trim_allocator();
            return Ok(());
        }
        self.ensure_cpu_cached(name)?;
        let mut state = lock_state(&self.inner);
        let on_gpu = state
            .gpu
            .as_ref()
            .is_some_and(|g| g.cache.contains_key(name));
        if on_gpu {
            self.drop_gpu_if_idle(&mut state);
        }
        drop(state);
        trim_allocator();
        Ok(())
    }

    fn ensure_summarizer_ram(&self, model_name: &str) -> WhisburnResult<()> {
        if crate::model::granite::is_granite_model(model_name) {
            return Ok(());
        }
        let already = {
            let state = lock_state(&self.inner);
            state
                .summarizer
                .as_ref()
                .is_some_and(|(name, _)| name == model_name)
        };
        if already {
            return Ok(());
        }
        let device = {
            let state = lock_state(&self.inner);
            state.cpu.device.clone()
        };
        let started = Instant::now();
        let (tokenizer, thinker) = crate::model::qwen3::summarize::load_summarizer(
            model_name,
            &device,
            self.inner.verbose,
        )?;
        self.note_weight_load(started.elapsed());
        let mut state = lock_state(&self.inner);
        state.summarizer = Some((
            model_name.to_string(),
            CachedSummarizer { tokenizer, thinker },
        ));
        Ok(())
    }

    fn apply_tier_move(&self, mv: &TierMove) -> WhisburnResult<()> {
        match mv.to {
            Tier::Disk => {
                self.drop_named_to_disk(&mv.name);
                Ok(())
            }
            Tier::Ram => self.park_in_ram(&mv.name),
            Tier::Vram => Ok(()),
        }
    }

    fn summarize_from_vram(
        &self,
        model_name: &str,
        text: &str,
        progress: Option<crate::model::qwen3::summarize::SummarizeProgress>,
    ) -> WhisburnResult<String> {
        if crate::model::granite::is_granite_model(model_name) {
            let device = self.ensure_gpu_device();
            let _live = self.enter_gpu();
            return summarize_on(model_name, &device, text, self.inner.verbose, progress);
        }
        let device = self.ensure_gpu_device();
        let reused = {
            let mut state = lock_state(&self.inner);
            match state.summarizer_vram.take() {
                Some((name, gpu)) if name == model_name => Some(gpu),
                other => {
                    state.summarizer_vram = other;
                    None
                }
            }
        };
        let gpu = if let Some(gpu) = reused {
            tracing::info!(model = model_name, "reusing VRAM summarizer");
            gpu
        } else {
            let started = Instant::now();
            let (tokenizer, thinker) = crate::model::qwen3::summarize::load_summarizer(
                model_name,
                &device,
                self.inner.verbose,
            )?;
            self.note_weight_load(started.elapsed());
            GpuSummarizer { tokenizer, thinker }
        };
        let _live = self.enter_gpu();
        let result = recover_gpu_failure(
            catch_unwind(AssertUnwindSafe(|| {
                crate::model::qwen3::summarize::summarize_with_thinker(
                    &gpu.thinker,
                    &gpu.tokenizer,
                    text,
                    &device,
                    progress,
                )
            })),
            &self.inner,
        );
        {
            let mut state = lock_state(&self.inner);
            state.summarizer.take();
            state.summarizer_vram = Some((model_name.to_string(), gpu));
        }
        result
    }

    fn gpu_execution_blocked(&self) -> bool {
        let prefer_gpu = lock_state(&self.inner).prefer_gpu;
        !prefer_gpu
            || self.inner.poisoned.load(Ordering::SeqCst)
            || self.inner.gpu_unavailable.load(Ordering::SeqCst)
            || crate::gpu_env::adapter_missing()
    }

    /// Demote other VRAM residents, then place this speech model. A footprint
    /// that still breaches the reserve after demotion runs from RAM. Callers
    /// hold the job slot before this, so a queued job has not allocated yet.
    fn prepare_speech_admission(&self, model_name: &str, audio_sec: f64) -> WhisburnResult<Tier> {
        if self.gpu_execution_blocked() {
            self.release_gpu();
            return Ok(Tier::Ram);
        }
        let weight = model_vram_mb(model_name);
        let window = audio_sec.min(MAX_GPU_CHUNK_SEC).max(0.0);
        let activation = activation_mb(window, weight);
        let plan = admit_inference(
            &self.host_budget(),
            &self.placement_residents(),
            model_name,
            weight,
            activation,
        );
        tracing::info!(
            model = model_name,
            run_on = ?plan.run_on,
            ?plan.moves,
            "admitted speech job"
        );
        for mv in &plan.moves {
            if plan.run_on == Tier::Vram && mv.name == model_name {
                continue;
            }
            self.apply_tier_move(mv)?;
        }
        Ok(plan.run_on)
    }

    /// Offline English summary. Placement follows [`plan_summarize_handoff`]:
    /// stage the summarizer in RAM, then put it on VRAM and park or spill the
    /// speech model. A GPU load failure falls back to the RAM copy.
    pub fn summarize_text(
        &self,
        text: &str,
        model_name: &str,
        progress: Option<crate::model::qwen3::summarize::SummarizeProgress>,
    ) -> WhisburnResult<String> {
        let _job = self.begin_job();
        let speech = self.speech_resident(model_name);
        let budget = self.host_budget();
        let weight = model_vram_mb(model_name);
        let activation = window_workspace_mb(weight);
        let moves = plan_summarize_handoff(
            &budget,
            speech.as_ref(),
            model_name,
            weight,
            activation,
        );
        tracing::info!(model = model_name, ?moves, "summarize tier handoff");
        let promote_vram = moves
            .iter()
            .any(|m| m.name == model_name && m.to == Tier::Vram);
        for mv in &moves {
            if mv.name == model_name && mv.to == Tier::Vram {
                continue;
            }
            self.apply_tier_move(mv)?;
        }
        if promote_vram && !self.gpu_execution_blocked() {
            match self.summarize_from_vram(model_name, text, progress.clone()) {
                Ok(summary) => return Ok(summary),
                Err(e)
                    if looks_like_gpu_failure(&e.to_string())
                        || self.inner.poisoned.load(Ordering::SeqCst) =>
                {
                    tracing::error!(
                        error = %e,
                        "GPU summarizer failed; using the RAM copy"
                    );
                    self.release_gpu();
                }
                Err(e) => return Err(e),
            }
        }
        self.summarize_from_ram(model_name, text, progress)
    }

    fn summarize_from_ram(
        &self,
        model_name: &str,
        text: &str,
        progress: Option<crate::model::qwen3::summarize::SummarizeProgress>,
    ) -> WhisburnResult<String> {
        trim_allocator();
        tracing::info!(
            model = model_name,
            chars = text.len(),
            "summarizer running from RAM"
        );
        let device = {
            let state = lock_state(&self.inner);
            state.cpu.device.clone()
        };
        let verbose = self.inner.verbose;
        if crate::model::granite::is_granite_model(model_name) {
            return recover_gpu_failure(
                catch_unwind(AssertUnwindSafe(|| {
                    summarize_on(model_name, &device, text, verbose, progress)
                })),
                &self.inner,
            );
        }
        let mut cached = {
            let mut state = lock_state(&self.inner);
            match state.summarizer.take() {
                Some((name, summ)) if name == model_name => Some(summ),
                other => {
                    state.summarizer = other;
                    None
                }
            }
        };
        if cached.is_none() {
            let started = Instant::now();
            let (tokenizer, thinker) =
                crate::model::qwen3::summarize::load_summarizer(
                    model_name, &device, verbose,
                )?;
            self.note_weight_load(started.elapsed());
            cached = Some(CachedSummarizer { tokenizer, thinker });
        } else {
            tracing::info!(model = model_name, "reusing CPU summarizer");
        }
        let summ = cached.expect("summarizer loaded");
        let result = recover_gpu_failure(
            catch_unwind(AssertUnwindSafe(|| {
                crate::model::qwen3::summarize::summarize_with_thinker(
                    &summ.thinker,
                    &summ.tokenizer,
                    text,
                    &device,
                    progress,
                )
            })),
            &self.inner,
        );
        {
            let mut state = lock_state(&self.inner);
            state.summarizer = Some((model_name.to_string(), summ));
        }
        result
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
        self.warm_for_audio(model_name, MIN_GPU_CHUNK_SEC)
    }

    /// True when a cached "warm" flag would leave a model on the wrong tier.
    /// A previous warm must not skip admission while another resident still
    /// has to move, or while this model is not actually in the chosen cache.
    fn resident_tier_must_change(&self, model_name: &str, audio_sec: f64) -> bool {
        if self.gpu_execution_blocked() {
            let state = lock_state(&self.inner);
            let on_cpu = state.cpu.cache.contains_key(model_name);
            let gpu_held = state.gpu.as_ref().is_some_and(|g| !g.cache.is_empty())
                || state.summarizer_vram.is_some();
            return !on_cpu || gpu_held;
        }
        let residents = self.placement_residents();
        let weight = model_vram_mb(model_name);
        let window = audio_sec.min(MAX_GPU_CHUNK_SEC).max(0.0);
        let plan = admit_inference(
            &self.host_budget(),
            &residents,
            model_name,
            weight,
            activation_mb(window, weight),
        );
        plan.moves
            .iter()
            .any(|mv| tier_of(&residents, &mv.name) != Some(mv.to))
    }

    /// Place weights before they are read. `audio_sec` is the clip the caller
    /// is about to run, so the same [`admit_inference`] plan used at transcribe
    /// time decides VRAM vs RAM before this load.
    pub fn warm_for_audio(&self, model_name: &str, audio_sec: f64) -> WhisburnResult<()> {
        if is_vad_model(model_name) || is_diarize_model(model_name) {
            lock_state(&self.inner)
                .warmed
                .insert(model_name.to_string());
            return Ok(());
        }
        let already_warm = lock_state(&self.inner).warmed.contains(model_name);
        if already_warm && !self.resident_tier_must_change(model_name, audio_sec) {
            return Ok(());
        }
        let _job = self.begin_job();
        let run_on = self.prepare_speech_admission(model_name, audio_sec)?;
        if run_on == Tier::Vram {
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
                    let msg = panic_message(payload);
                    if crate::gpu_env::missing_adapter_message(&msg)
                        || crate::gpu_env::adapter_missing()
                    {
                        self.inner.gpu_unavailable.store(true, Ordering::SeqCst);
                    }
                    tracing::error!(
                        panic = %msg,
                        model = model_name,
                        "panic while loading GPU model; warming on CPU"
                    );
                    self.release_gpu();
                    self.ensure_cpu_cached(model_name)?;
                }
            }
        } else {
            self.ensure_cpu_cached(model_name)?;
        }
        lock_state(&self.inner)
            .warmed
            .insert(model_name.to_string());
        self.maybe_offload_idle();
        Ok(())
    }

    fn ensure_gpu_device(&self) -> Device {
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
        let busy = self.inner.gpu_inflight.load(Ordering::SeqCst) > 0;
        {
            let mut state = lock_state(&self.inner);
            let gpu = state.gpu.as_mut().expect("gpu inserted");
            if gpu.cache.contains_key(model_name) {
                return Ok(());
            }
            // Another forward pass is using this device. Insert beside it
            // instead of clearing the cache out from under that job.
            if !busy {
                gpu.cache.clear();
                state.summarizer_vram.take();
            }
        }
        trim_allocator();
        let started = Instant::now();
        let (bpe, _config, model) = load_model(model_name, &device, self.inner.verbose)
            .map_err(|e| WhisburnError::Model(e.to_string()))?;
        self.note_weight_load(started.elapsed());
        let mut state = lock_state(&self.inner);
        let gpu = state.gpu.get_or_insert_with(|| GpuBackend {
            device: device.clone(),
            cache: HashMap::new(),
        });
        gpu.cache
            .insert(model_name.to_string(), Arc::new(LoadedModel { bpe, model }));
        Ok(())
    }

    fn ensure_cpu_cached(&self, model_name: &str) -> WhisburnResult<()> {
        let device = {
            let mut state = lock_state(&self.inner);
            if state.cpu.cache.contains_key(model_name) {
                return Ok(());
            }
            state.cpu.cache.clear();
            state.cpu.device.clone()
        };
        trim_allocator();
        let started = Instant::now();
        let (bpe, _config, model) = load_model(model_name, &device, self.inner.verbose)
            .map_err(|e| WhisburnError::Model(e.to_string()))?;
        self.note_weight_load(started.elapsed());
        let mut state = lock_state(&self.inner);
        state.cpu.cache.clear();
        state
            .cpu
            .cache
            .insert(model_name.to_string(), Arc::new(LoadedModel { bpe, model }));
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
        let audio_sec = samples.len() as f64 / sample_rate.max(1) as f64;
        let run_on = self.prepare_speech_admission(model_name, audio_sec)?;

        if options.orchestrate && run_on == Tier::Vram {
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

        if run_on == Tier::Ram {
            let result =
                self.transcribe_on_cpu(model_name, samples, sample_rate, options, decode_task);
            if result.is_err() {
                self.release_gpu();
            } else {
                self.maybe_offload_idle();
            }
            return result;
        }

        let extra = model_vram_mb(model_name);
        let (prefer_gpu, loaded) = {
            let state = lock_state(&self.inner);
            let loaded = state
                .gpu
                .as_ref()
                .is_some_and(|g| g.cache.contains_key(model_name));
            (state.prefer_gpu, loaded)
        };
        // An explicit CPU device stays on Flex (rayon + SIMD). The VRAM plan
        // only chooses among GPU, split-on-GPU, and offload.
        let plan = if !prefer_gpu
            || self.inner.poisoned.load(Ordering::SeqCst)
            || self.inner.gpu_unavailable.load(Ordering::SeqCst)
            || crate::gpu_env::adapter_missing()
        {
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
            GpuAudioPlan::SinglePass => {
                self.transcribe_gpu_or_cpu(model_name, samples, sample_rate, options, decode_task)
            }
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
        let _live = self.enter_gpu();
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

    fn enter_gpu(&self) -> GpuInflight<'_> {
        self.inner.gpu_inflight.fetch_add(1, Ordering::SeqCst);
        GpuInflight { inner: &self.inner }
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
        let loaded =
            {
                let state = lock_state(&self.inner);
                state.cpu.cache.get(model_name).cloned().ok_or_else(|| {
                    WhisburnError::Model(format!("CPU model '{model_name}' missing"))
                })?
            };
        run_waveform(
            &loaded.model,
            &loaded.bpe,
            model_name,
            samples,
            sample_rate,
            options,
            decode_task,
            "Flex (CPU)".to_string(),
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
        // The server's "segment N of M" is this whole buffer. GPU slices are shorter,
        // and each slice would otherwise restart the window counter at 0.
        let _window_span = crate::transcribe::begin_decode_window_span(samples.len(), sample_rate);
        // Do not pad a silence sliver up to a fake second. Those slices are skipped below.
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
            // A quiet-boundary search can leave a few samples. Parakeet's stride-2
            // stack then reshapes a 0-length axis and aborts the whole file.
            if chunk.len() < sample_rate.max(1) {
                tracing::warn!(
                    chunk = i + 1,
                    of = n,
                    samples = chunk.len(),
                    "skipping sub-second audio slice"
                );
                continue;
            }
            match self.transcribe_one_split_chunk(
                model_name,
                chunk,
                sample_rate,
                options,
                decode_task,
                i,
                n,
            ) {
                Ok(r) => parts.push((offset, r)),
                Err(e) => {
                    tracing::error!(
                        chunk = i + 1,
                        of = n,
                        error = %e,
                        "skipping audio slice that failed on GPU and CPU"
                    );
                }
            }
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
        let _live = self.enter_gpu();
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
            if crate::gpu_env::missing_adapter_message(&msg) || crate::gpu_env::adapter_missing() {
                inner.gpu_unavailable.store(true, Ordering::SeqCst);
            }
            if inner.gpu_inflight.load(Ordering::SeqCst) > 0 {
                tracing::error!(
                    panic = %msg,
                    "GPU worker panicked; leaving the device for the in-flight job"
                );
                return Err(WhisburnError::Inference(msg));
            }
            tracing::error!(panic = %msg, "GPU worker panicked; releasing VRAM");
            drop_gpu(&mut lock_state(inner));
            Err(WhisburnError::Inference(msg))
        }
    }
}

fn run_waveform(
    model: &Model,
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

fn resident_models(inner: &RuntimeInner) -> bool {
    #[cfg(test)]
    if inner.idle_probe.load(Ordering::SeqCst) {
        return true;
    }
    let state = lock_state(inner);
    !state.cpu.cache.is_empty()
        || state.summarizer.is_some()
        || state.summarizer_vram.is_some()
        || state.gpu.as_ref().is_some_and(|g| !g.cache.is_empty())
}

/// Unload cached models when the idle window has elapsed and no job is running.
fn poll_idle_unload(inner: &RuntimeInner) -> bool {
    if inner.idle.is_disabled() {
        return false;
    }
    if inner.jobs.load(Ordering::SeqCst) > 0 {
        return false;
    }
    if !resident_models(inner) {
        return false;
    }
    let last = inner.last_load.lock().ok().and_then(|g| *g);
    let Some(timeout) = inner.idle.timeout_for(last) else {
        return false;
    };
    let since = match inner.idle_since.lock() {
        Ok(guard) => *guard,
        Err(_) => return false,
    };
    let Some(since) = since else {
        return false;
    };
    if since.elapsed() < timeout {
        return false;
    }
    if inner.jobs.load(Ordering::SeqCst) > 0 {
        return false;
    }
    tracing::info!(
        idle_secs = since.elapsed().as_secs(),
        timeout_secs = timeout.as_secs(),
        "idle timeout elapsed; unloading cached models"
    );
    release_cached_models(inner);
    if let Ok(mut idle) = inner.idle_since.lock() {
        *idle = None;
    }
    true
}

fn release_cached_models(inner: &RuntimeInner) {
    let mut state = lock_state(inner);
    let cpu_names: Vec<String> = state.cpu.cache.keys().cloned().collect();
    let summ = state.summarizer.take().map(|(name, _)| name);
    let summ_vram = state.summarizer_vram.take().map(|(name, _)| name);
    state.cpu.cache.clear();
    state.warmed.clear();
    drop_gpu(&mut state);
    drop(state);
    inner.poisoned.store(false, Ordering::SeqCst);
    #[cfg(test)]
    inner.idle_probe.store(false, Ordering::SeqCst);
    if !cpu_names.is_empty() || summ.is_some() || summ_vram.is_some() {
        tracing::info!(
            cpu_models = ?cpu_names,
            summarizer = ?summ,
            summarizer_vram = ?summ_vram,
            "released CPU model cache after idle"
        );
    }
    trim_allocator();
}

#[cfg_attr(test, allow(dead_code))]
fn spawn_vram_guard(inner: Weak<RuntimeInner>) {
    let _ = thread::Builder::new()
        .name("whisburn-vram-guard".into())
        .spawn(move || {
            let mut idle_polls = 0u32;
            loop {
                thread::sleep(Duration::from_secs(2));
                let Some(inner) = inner.upgrade() else {
                    break;
                };
                if inner.stop.load(Ordering::SeqCst) {
                    break;
                }
                if inner.jobs.load(Ordering::SeqCst) > 0 {
                    idle_polls = 0;
                    continue;
                }
                // Time-based unload is independent of VRAM pressure.
                poll_idle_unload(&inner);
                if inner.jobs.load(Ordering::SeqCst) > 0 {
                    idle_polls = 0;
                    continue;
                }
                let poisoned = inner.poisoned.load(Ordering::SeqCst);
                let snap = probe();
                let starved = inner.policy.should_offload(&snap, 0);
                if !poisoned && !starved {
                    idle_polls = 0;
                    continue;
                }
                // ~6s after the last job so the SSE done event can leave the process.
                idle_polls = idle_polls.saturating_add(1);
                if idle_polls < 3 {
                    continue;
                }
                idle_polls = 0;
                if inner.jobs.load(Ordering::SeqCst) > 0 {
                    continue;
                }
                if poisoned {
                    tracing::error!("GPU panic flag set; releasing VRAM");
                } else {
                    tracing::warn!(?snap, "idle VRAM below reserve; releasing GPU tensors");
                }
                drop_gpu(&mut lock_state(&inner));
                inner.poisoned.store(false, Ordering::SeqCst);
                exit_if_compositor_starved(&inner);
            }
        });
}

struct GpuInflight<'a> {
    inner: &'a RuntimeInner,
}

impl Drop for GpuInflight<'_> {
    fn drop(&mut self) {
        self.inner.gpu_inflight.fetch_sub(1, Ordering::SeqCst);
    }
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
    fn panic_recovery_leaves_the_device_while_another_forward_is_in_flight() {
        let rt = InferenceRuntime::new_with_policy(
            Some("cpu".into()),
            false,
            false,
            VramPolicy::disabled(),
        );
        lock_state(&rt.inner)
            .warmed
            .insert("live-model".to_string());
        rt.inner.gpu_inflight.store(1, Ordering::SeqCst);
        let err = recover_gpu_failure::<()>(Err(Box::new("sibling still running")), &rt.inner)
            .expect_err("the panicked job still fails");
        assert!(err.to_string().contains("sibling still running"));
        assert!(
            lock_state(&rt.inner).warmed.contains("live-model"),
            "in-flight sibling must keep the GPU device"
        );
        rt.inner.gpu_inflight.store(0, Ordering::SeqCst);
        let _ = recover_gpu_failure::<()>(Err(Box::new("only job")), &rt.inner);
        assert!(
            !lock_state(&rt.inner).warmed.contains("live-model"),
            "the last job's panic still releases VRAM"
        );
    }

    #[test]
    fn cpu_warm_still_loads_the_ram_cache_when_marked_warm() {
        let rt = InferenceRuntime::new_with_policy(
            Some("cpu".into()),
            false,
            false,
            VramPolicy::disabled(),
        );
        let model = "not-a-real-model";
        lock_state(&rt.inner).warmed.insert(model.to_string());
        let err = rt
            .warm_for_audio(model, 4.0)
            .expect_err("a RAM admission must try the CPU cache");
        let msg = err.to_string();
        assert!(
            msg.contains("Config file not found") || msg.contains("model error"),
            "RAM warm should reach the weight loader, got {msg}"
        );
        assert!(rt.loaded_models().is_empty());
        assert!(
            !lock_state(&rt.inner).cpu.cache.contains_key(model),
            "a failed RAM load must not pretend the cache is warm"
        );
    }

    #[test]
    fn cpu_runtime_release_is_idempotent() {
        let rt = InferenceRuntime::new_with_policy(
            Some("cpu".into()),
            false,
            false,
            VramPolicy::disabled(),
        );
        assert_eq!(rt.backend_name(), "Flex (CPU)");
        rt.release_gpu();
        rt.release_gpu();
        assert!(rt.loaded_models().is_empty());
    }

    #[test]
    fn request_hold_keeps_jobs_above_zero_until_the_last_guard_drops() {
        let rt = InferenceRuntime::new_with_policy(
            Some("cpu".into()),
            false,
            false,
            VramPolicy::disabled(),
        );
        let outer = rt.begin_request();
        let inner = rt.begin_job();
        assert_eq!(rt.inner.jobs.load(Ordering::SeqCst), 2);
        drop(inner);
        assert_eq!(rt.inner.jobs.load(Ordering::SeqCst), 1);
        drop(outer);
        assert_eq!(rt.inner.jobs.load(Ordering::SeqCst), 0);
        assert!(rt.inner.idle_since.lock().unwrap().is_some());
        rt.release_gpu();
    }

    #[test]
    fn idle_unload_waits_out_the_request_then_releases() {
        let rt = InferenceRuntime::new_with_policies(
            Some("cpu".into()),
            false,
            false,
            VramPolicy::disabled(),
            IdleUnloadPolicy::Fixed(Duration::from_millis(15)),
        );
        rt.inner.idle_probe.store(true, Ordering::SeqCst);
        let hold = rt.begin_request();
        *rt.inner.idle_since.lock().unwrap() = Some(Instant::now() - Duration::from_secs(30));
        assert!(
            !rt.poll_idle_unload(),
            "a live request must keep the model resident"
        );
        assert!(rt.inner.idle_probe.load(Ordering::SeqCst));
        drop(hold);
        thread::sleep(Duration::from_millis(40));
        assert!(
            rt.poll_idle_unload(),
            "idle window after the request should unload"
        );
        assert!(!rt.inner.idle_probe.load(Ordering::SeqCst));
        assert!(!rt.poll_idle_unload());
    }

    #[test]
    fn disabled_idle_policy_keeps_the_probe() {
        let rt = InferenceRuntime::new_with_policies(
            Some("cpu".into()),
            false,
            false,
            VramPolicy::disabled(),
            IdleUnloadPolicy::Disabled,
        );
        rt.inner.idle_probe.store(true, Ordering::SeqCst);
        *rt.inner.idle_since.lock().unwrap() = Some(Instant::now() - Duration::from_secs(3600));
        assert!(!rt.poll_idle_unload());
        assert!(rt.inner.idle_probe.load(Ordering::SeqCst));
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
        let merged = merge_chunk_results(
            "x",
            "transcribe",
            Some("en".into()),
            vec![(0.0, a), (15.0, b)],
        );
        assert_eq!(merged.text, "hello world");
        assert_eq!(merged.segments.len(), 2);
        assert!((merged.segments[1].start - 15.2).abs() < 1e-9);
        assert!((merged.segments[1].end - 16.0).abs() < 1e-9);
    }
}
