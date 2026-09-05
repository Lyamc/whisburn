use std::path::PathBuf;
use std::sync::{Arc, Mutex as StdMutex};

use tokio::sync::Mutex;
use whisburn_core::{SpeechTask, TaskOptions, TranscriptResult, WhisburnError, WhisburnResult};
use whisburn_engine::model::registry::{find_model, ModelCategory};
use whisburn_engine::runtime::InferenceRuntime;

use crate::diarize::assign_alternating_speakers;
use crate::download::{download_model, DownloadOptions, PrepProgressFn};
use crate::paths::{is_model_ready, resolve_model_dir};

pub struct ModelManager {
    runtime: InferenceRuntime,
    download_options: DownloadOptions,
    gate: Mutex<()>,
    sync_gate: StdMutex<()>,
}

impl ModelManager {
    pub fn new(device: Option<String>, verbose: bool, debug: bool) -> Self {
        Self {
            runtime: InferenceRuntime::new(device, verbose, debug),
            download_options: DownloadOptions {
                verbose,
                ..DownloadOptions::default()
            },
            gate: Mutex::new(()),
            sync_gate: StdMutex::new(()),
        }
    }

    pub fn with_hf_token(mut self, token: Option<String>) -> Self {
        self.download_options.hf_token = token;
        self
    }

    pub async fn ensure_model(&self, name: &str) -> WhisburnResult<PathBuf> {
        if is_model_ready(name) {
            return Ok(resolve_model_dir(name));
        }

        let _guard = self.gate.lock().await;
        if is_model_ready(name) {
            return Ok(resolve_model_dir(name));
        }

        download_model(name, &self.download_options)
            .map_err(|e| WhisburnError::Model(e.to_string()))
    }

    pub fn loaded_models(&self) -> Vec<String> {
        self.runtime.loaded_models()
    }

    pub fn is_loaded(&self, name: &str) -> bool {
        self.runtime.loaded_models().iter().any(|n| n == name)
    }

    pub async fn warm_model(&self, name: &str) -> WhisburnResult<()> {
        self.ensure_model(name).await?;
        self.runtime.warm_model(name)
    }

    pub fn warm_model_sync(&self, name: &str) -> WhisburnResult<()> {
        self.ensure_model_sync(name)?;
        tracing::info!(
            model = name,
            backend = %self.runtime.backend_name(),
            "loading model weights"
        );
        self.runtime.warm_model(name)?;
        tracing::info!(model = name, "model weights loaded");
        Ok(())
    }

    pub async fn process_waveform(
        &self,
        model_name: &str,
        samples: Vec<f32>,
        sample_rate: usize,
        options: &TaskOptions,
    ) -> WhisburnResult<TranscriptResult> {
        self.validate_task(model_name, options)?;
        self.ensure_model(model_name).await?;
        self.process_waveform_sync(model_name, samples, sample_rate, options)
    }

    pub fn ensure_model_sync(&self, name: &str) -> WhisburnResult<PathBuf> {
        if is_model_ready(name) {
            return Ok(resolve_model_dir(name));
        }

        let _guard = self
            .sync_gate
            .lock()
            .map_err(|e| WhisburnError::Model(e.to_string()))?;
        if is_model_ready(name) {
            return Ok(resolve_model_dir(name));
        }

        self.download_with_progress(name, None)
    }

    pub fn ensure_model_sync_with_progress(
        &self,
        name: &str,
        progress: Option<PrepProgressFn>,
    ) -> WhisburnResult<PathBuf> {
        if is_model_ready(name) {
            return Ok(resolve_model_dir(name));
        }

        let _guard = self
            .sync_gate
            .lock()
            .map_err(|e| WhisburnError::Model(e.to_string()))?;
        if is_model_ready(name) {
            return Ok(resolve_model_dir(name));
        }

        self.download_with_progress(name, progress)
    }

    fn download_with_progress(
        &self,
        name: &str,
        progress: Option<PrepProgressFn>,
    ) -> WhisburnResult<PathBuf> {
        let mut opts = self.download_options.clone();
        if progress.is_some() {
            opts.progress = progress;
        }
        download_model(name, &opts).map_err(|e| WhisburnError::Model(e.to_string()))
    }

    pub fn process_waveform_sync(
        &self,
        model_name: &str,
        samples: Vec<f32>,
        sample_rate: usize,
        options: &TaskOptions,
    ) -> WhisburnResult<TranscriptResult> {
        self.validate_task(model_name, options)?;
        if !is_model_ready(model_name) {
            return Err(WhisburnError::Model(format!(
                "model '{model_name}' is not ready — POST /v1/models/{model_name}/ensure first"
            )));
        }

        let mut result = self
            .runtime
            .transcribe_waveform(model_name, samples, sample_rate, options)?;

        if matches!(options.task, SpeechTask::Diarize) {
            result.task = "diarize".to_string();
            result = assign_alternating_speakers(result);
        }

        Ok(result)
    }

    pub fn summarize_text(
        &self,
        text: &str,
        progress: Option<whisburn_engine::model::qwen3::summarize::SummarizeProgress>,
    ) -> WhisburnResult<String> {
        let name = whisburn_engine::model::qwen3::summarize::DEFAULT_SUMMARIZER_MODEL;
        let prep = progress.clone().map(|cb| {
            std::sync::Arc::new(move |p: crate::download::PrepProgress| {
                cb(0, 1, &p.label);
            }) as crate::download::PrepProgressFn
        });
        self.ensure_model_sync_with_progress(name, prep)?;
        self.runtime.summarize_text(text, name, progress)
    }

    fn validate_task(&self, model_name: &str, options: &TaskOptions) -> WhisburnResult<()> {
        let info = find_model(model_name)
            .ok_or_else(|| WhisburnError::Model(format!("unknown model: {model_name}")))?;

        match options.task {
            SpeechTask::Transcribe | SpeechTask::Stt => {
                if info.category != ModelCategory::Asr {
                    return Err(WhisburnError::UnsupportedCapability {
                        model: model_name.to_string(),
                        capability: "transcribe".to_string(),
                    });
                }
            }
            SpeechTask::Translate => {
                if info.category != ModelCategory::Asr {
                    return Err(WhisburnError::UnsupportedCapability {
                        model: model_name.to_string(),
                        capability: "translate".to_string(),
                    });
                }
            }
            SpeechTask::Diarize => {
                if info.category != ModelCategory::Asr {
                    return Err(WhisburnError::UnsupportedCapability {
                        model: model_name.to_string(),
                        capability: "diarize".to_string(),
                    });
                }
            }
            SpeechTask::Tts | SpeechTask::Sts => {
                return Err(WhisburnError::UnsupportedCapability {
                    model: model_name.to_string(),
                    capability: options.task.as_str().to_string(),
                });
            }
        }

        Ok(())
    }
}

pub type SharedModelManager = Arc<ModelManager>;