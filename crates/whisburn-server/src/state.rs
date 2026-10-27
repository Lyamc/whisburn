use std::sync::Arc;

use whisburn_core::PreloadModels;
use whisburn_models::SharedModelManager;
use tokio::sync::Semaphore;

use crate::gpu::GpuInfo;

#[derive(Clone)]
pub struct AppState {
    pub manager: SharedModelManager,
    pub max_upload_bytes: usize,
    pub max_audio_seconds: u64,
    pub default_model: String,
    pub preload_models: PreloadModels,
    pub summarize_enabled: bool,
    pub summarize_default: bool,
    pub summarizer_model: String,
    pub gpu: GpuInfo,
    /// Inference jobs allowed to allocate at once. `1` is serial.
    pub max_parallel_jobs: u32,
    /// Caps concurrent speech/summarize/transcode work. Permits match
    /// [`whisburn_engine::placement::parallelism_limit`].
    pub job_semaphore: Arc<Semaphore>,
}