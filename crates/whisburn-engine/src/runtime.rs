use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{NdArray, wgpu::{Wgpu, WgpuDevice}};
use burn::tensor::backend::Backend;
use whisburn_core::{
    SpeechTask, TaskOptions, TranscriptResult, TranscriptSegment, WhisburnError, WhisburnResult,
};

use crate::cli::{get_backend_info, get_device, parse_language, CommonArgs};
use crate::model::load_model;
use crate::model::Model;
use crate::orchestrator::Orchestrator;
use crate::token::{Gpt2Tokenizer, Language};
use crate::transcribe::{waveform_to_text, DecodeTask};

struct LoadedModel<B: Backend> {
    bpe: Gpt2Tokenizer,
    model: Model<B>,
}

enum BackendRuntime {
    Wgpu {
        device: WgpuDevice,
        cache: Mutex<HashMap<String, LoadedModel<Wgpu>>>,
    },
    NdArray {
        device: NdArrayDevice,
        cache: Mutex<HashMap<String, LoadedModel<NdArray>>>,
    },
}

pub struct InferenceRuntime {
    backend: BackendRuntime,
    verbose: bool,
    debug: bool,
    warmed: Mutex<HashSet<String>>,
}

fn wants_cpu(device: &Option<String>) -> bool {
    device
        .as_deref()
        .is_some_and(|s| s.eq_ignore_ascii_case("cpu"))
}

impl InferenceRuntime {
    pub fn new(device: Option<String>, verbose: bool, debug: bool) -> Self {
        let backend = if wants_cpu(&device) {
            BackendRuntime::NdArray {
                device: NdArrayDevice::Cpu,
                cache: Mutex::new(HashMap::new()),
            }
        } else {
            BackendRuntime::Wgpu {
                device: get_device(&device),
                cache: Mutex::new(HashMap::new()),
            }
        };
        Self {
            backend,
            verbose,
            debug,
            warmed: Mutex::new(HashSet::new()),
        }
    }

    pub fn backend_name(&self) -> String {
        match &self.backend {
            BackendRuntime::Wgpu { device, .. } => get_backend_info(device),
            BackendRuntime::NdArray { .. } => "NdArray (CPU)".to_string(),
        }
    }

    pub fn loaded_models(&self) -> Vec<String> {
        let mut names: Vec<String> = match &self.backend {
            BackendRuntime::Wgpu { cache, .. } => {
                cache.lock().expect("cache lock").keys().cloned().collect()
            }
            BackendRuntime::NdArray { cache, .. } => {
                cache.lock().expect("cache lock").keys().cloned().collect()
            }
        };
        names.sort();
        names
    }

    pub fn warm_model(&self, model_name: &str) -> WhisburnResult<()> {
        {
            let warmed = self.warmed.lock().expect("warmed lock");
            if warmed.contains(model_name) {
                return Ok(());
            }
        }
        match &self.backend {
            BackendRuntime::Wgpu { device, cache } => {
                Self::ensure_cached::<Wgpu>(model_name, device, cache, self.verbose)?;
            }
            BackendRuntime::NdArray { device, cache } => {
                Self::ensure_cached::<NdArray>(model_name, device, cache, self.verbose)?;
            }
        }
        self.warmed
            .lock()
            .expect("warmed lock")
            .insert(model_name.to_string());
        Ok(())
    }

    fn ensure_cached<B: Backend>(
        model_name: &str,
        device: &B::Device,
        cache: &Mutex<HashMap<String, LoadedModel<B>>>,
        verbose: bool,
    ) -> WhisburnResult<()> {
        let mut cache = cache.lock().expect("cache lock");
        if cache.contains_key(model_name) {
            return Ok(());
        }
        // A 12 GB GPU cannot hold several Whisper/Qwen checkpoints at once.
        // The HTTP server processes one job at a time, so keep only the active model.
        cache.clear();
        let (bpe, _config, model) = load_model::<B>(model_name, device, verbose)
            .map_err(|e| WhisburnError::Model(e.to_string()))?;
        cache.insert(model_name.to_string(), LoadedModel { bpe, model });
        Ok(())
    }

    pub fn transcribe_waveform(
        &self,
        model_name: &str,
        samples: Vec<f32>,
        sample_rate: usize,
        options: &TaskOptions,
    ) -> WhisburnResult<TranscriptResult> {
        let decode_task = decode_task_for(options.task);

        if options.orchestrate {
            return self.transcribe_orchestrated(model_name, samples, sample_rate, options, decode_task);
        }

        let lang = map_language(options.language.as_str());
        let backend_info = self.backend_name();

        match &self.backend {
            BackendRuntime::Wgpu { device, cache } => Self::transcribe_with::<Wgpu>(
                model_name,
                device,
                cache,
                samples,
                sample_rate,
                options,
                lang,
                decode_task,
                backend_info,
                self.verbose,
                self.debug,
            ),
            BackendRuntime::NdArray { device, cache } => Self::transcribe_with::<NdArray>(
                model_name,
                device,
                cache,
                samples,
                sample_rate,
                options,
                lang,
                decode_task,
                backend_info,
                self.verbose,
                self.debug,
            ),
        }
    }

    fn transcribe_with<B: Backend>(
        model_name: &str,
        device: &B::Device,
        cache: &Mutex<HashMap<String, LoadedModel<B>>>,
        samples: Vec<f32>,
        sample_rate: usize,
        options: &TaskOptions,
        lang: Language,
        decode_task: DecodeTask,
        backend_info: String,
        verbose: bool,
        debug: bool,
    ) -> WhisburnResult<TranscriptResult> {
        Self::ensure_cached::<B>(model_name, device, cache, verbose)?;
        let cache = cache.lock().expect("cache lock");
        let loaded = cache.get(model_name).expect("model inserted above");
        let (text, segments) = waveform_to_text(
            &loaded.model,
            model_name,
            &loaded.bpe,
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

    fn transcribe_orchestrated(
        &self,
        model_name: &str,
        samples: Vec<f32>,
        sample_rate: usize,
        options: &TaskOptions,
        decode_task: DecodeTask,
    ) -> WhisburnResult<TranscriptResult> {
        let scout = options
            .scout_model
            .as_deref()
            .unwrap_or("tiny_en");

        let common = CommonArgs {
            model: model_name.to_string(),
            lang: options.language.as_str().to_string(),
            verbose: self.verbose,
            debug: self.debug,
            quiet: true,
            verify: false,
            hf_token: None,
            beam_size: options.beam_size,
            max_tokens: options.max_tokens,
            padding: options.padding,
            device: None,
        };

        let device = match &self.backend {
            BackendRuntime::Wgpu { device, .. } => device.clone(),
            BackendRuntime::NdArray { .. } => {
                return Err(WhisburnError::Inference(
                    "orchestrate is not supported on the NdArray CPU backend".into(),
                ));
            }
        };
        let orchestrator = Orchestrator::new(device, self.verbose, self.debug);
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

fn decode_task_for(task: SpeechTask) -> DecodeTask {
    match task {
        SpeechTask::Translate => DecodeTask::Translate,
        _ => DecodeTask::Transcribe,
    }
}

fn map_language(code: &str) -> Language {
    parse_language(code)
}