use std::sync::Arc;

use o3whisburn_core::PreloadModels;
use o3whisburn_models::SharedModelManager;
use tokio::sync::Semaphore;

use crate::gpu::GpuInfo;

#[derive(Clone)]
pub struct AppState {
    pub manager: SharedModelManager,
    pub max_upload_bytes: usize,
    pub max_audio_seconds: u64,
    pub default_model: String,
    pub preload_models: PreloadModels,
    pub gpu: GpuInfo,
    /// Serialize GPU/ffmpeg jobs so concurrent uploads cannot abort connections.
    pub job_semaphore: Arc<Semaphore>,
}