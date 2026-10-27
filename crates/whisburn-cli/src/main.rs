use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use whisburn_core::{load_settings, OutputFormat, PreloadModels, SpeechTask, TaskOptions};
use whisburn_engine::vram::VramPolicy;
use whisburn_models::{download_model, DownloadOptions, ModelManager};
use whisburn_server::ServerConfig;

/// Very rough real-time factor based estimate for how long transcription will take.
/// These are ballpark numbers for a decent GPU/CPU; actual time varies a lot.
fn estimate_transcribe_secs(audio_secs: f64, model: &str) -> f64 {
    let factor: f64 = if model.contains("tiny") || model.contains("parakeet") {
        0.09
    } else if model.contains("base") {
        0.16
    } else if model.contains("small") || model.contains("distil-medium") {
        0.28
    } else if model.contains("qwen3") {
        0.38
    } else if model.contains("medium") {
        0.55
    } else {
        0.35
    };
    (audio_secs * factor).max(1.5)
}

#[derive(Parser, Debug)]
#[command(
    name = "whisburn",
    about = "whisburn: Burn-based speech processing server and CLI"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start the server and open the web UI (http://127.0.0.1:8787/)
    Serve {
        #[arg(long, env = "WHISBURN_PORT")]
        port: Option<u16>,
        #[arg(long, env = "WHISBURN_DEFAULT_MODEL")]
        model: Option<String>,
        #[arg(long, env = "WHISBURN_DEVICE")]
        device: Option<String>,
        #[arg(long, env = "HF_TOKEN")]
        hf_token: Option<String>,
        #[arg(short, long, env = "WHISBURN_VERBOSE")]
        verbose: bool,
        #[arg(long, env = "WHISBURN_DEBUG")]
        debug: bool,
        /// Do not open a browser tab for the web UI.
        #[arg(long = "no-open")]
        no_open: bool,
        /// Load models into GPU memory at startup. Omit for lazy load on first audio.
        /// Pass alone or `all` for every burn_ready model; pass names for specific models.
        #[arg(long = "preload-models", value_delimiter = ',', num_args = 0.., env = "WHISBURN_PRELOAD_MODELS")]
        preload_models: Option<Vec<String>>,
        /// Keep at least this many MiB of GPU memory free (compositor / other apps).
        #[arg(long, env = "WHISBURN_VRAM_RESERVE_MB")]
        vram_reserve_mb: Option<u32>,
        /// Keep at least this percentage of total VRAM free.
        #[arg(long, env = "WHISBURN_VRAM_RESERVE_PERCENT")]
        vram_reserve_percent: Option<u32>,
        /// Seconds to keep the last model in memory after a job. Omit for auto
        /// (5× weight-load time, 30s–15m). `0` keeps models loaded.
        /// Non-numeric overrides (`off`, `auto`) belong in `WHISBURN_IDLE_UNLOAD_SECS`.
        #[arg(long)]
        idle_unload_secs: Option<u32>,
        /// Inference jobs allowed to allocate at once. `0` or `1` is serial.
        /// `off`/`disabled` are accepted via `WHISBURN_MAX_PARALLEL_JOBS`.
        #[arg(long)]
        max_parallel_jobs: Option<u32>,
    },
    /// Open the Iced desktop transcription client
    #[cfg(feature = "app")]
    App,
    /// Download and prepare a model locally
    Models {
        #[command(subcommand)]
        action: ModelsCommand,
    },
    /// Transcribe, translate, or diarize a local audio file
    Transcribe {
        #[arg(short, long)]
        input: String,
        #[arg(short, long, default_value = "tiny_en", env = "WHISBURN_DEFAULT_MODEL")]
        model: String,
        #[arg(short, long, default_value = "en")]
        language: String,
        #[arg(long, default_value = "transcribe")]
        task: String,
        #[arg(long, default_value = "json")]
        format: String,
        #[arg(long, env = "WHISBURN_DEVICE")]
        device: Option<String>,
        #[arg(long, env = "WHISBURN_VRAM_RESERVE_MB")]
        vram_reserve_mb: Option<u32>,
        #[arg(long, env = "WHISBURN_VRAM_RESERVE_PERCENT")]
        vram_reserve_percent: Option<u32>,
        #[arg(long, env = "HF_TOKEN")]
        hf_token: Option<String>,
        #[arg(long)]
        timestamps: Option<bool>,
        #[arg(long)]
        orchestrate: bool,
        #[arg(long, default_value = "tiny_en")]
        scout_model: String,
        #[arg(long, default_value_t = 5)]
        beam_size: usize,
        #[arg(long, default_value_t = 448)]
        max_tokens: usize,
        #[arg(long)]
        sentences: bool,
        /// Also write `{input_stem}_summary.txt` with an offline English summarizer.
        #[arg(long)]
        summarize: bool,
        /// Summarizer used with `--summarize` (`qwen3-0.6b` or `granite-4.2-3b-fp8`).
        #[arg(long, env = "WHISBURN_SUMMARIZER", default_value = "qwen3-0.6b")]
        summarizer: String,
        #[arg(short, long)]
        verbose: bool,
        #[arg(long)]
        debug: bool,
    },
    /// Verify model outputs across formats/settings (json/txt/srt, diarize, sentences)
    Verify {
        /// Input audio file to use for verification (defaults to bundled sample)
        #[arg(short, long)]
        input: Option<String>,
        /// Comma separated list of models (or "burn-ready" for all ready ones)
        #[arg(
            long,
            default_value = "tiny_en,base_en,parakeet-tdt-0.6b-v3,qwen3-asr-0.6b"
        )]
        models: String,
        #[arg(long, env = "WHISBURN_DEVICE")]
        device: Option<String>,
        #[arg(long, env = "WHISBURN_VRAM_RESERVE_MB")]
        vram_reserve_mb: Option<u32>,
        #[arg(long, env = "WHISBURN_VRAM_RESERVE_PERCENT")]
        vram_reserve_percent: Option<u32>,
        #[arg(long, env = "HF_TOKEN")]
        hf_token: Option<String>,
        /// Also exercise diarization mode
        #[arg(long)]
        diarize: bool,
        /// Also exercise sentence grouping
        #[arg(long)]
        sentences: bool,
        #[arg(short, long)]
        verbose: bool,
        #[arg(long)]
        debug: bool,
        /// Directory to write verification artifacts (default: ./verify-outputs)
        #[arg(long)]
        out_dir: Option<String>,
    },
}

#[derive(Subcommand, Debug)]
enum ModelsCommand {
    /// Download a model by name
    Download {
        name: String,
        #[arg(long)]
        force: bool,
        #[arg(long, env = "HF_TOKEN")]
        hf_token: Option<String>,
        #[arg(short, long)]
        verbose: bool,
    },
    /// List registered models
    List,
}

fn init_tracing(verbose: bool, debug: bool) {
    let default = if debug { "debug" } else { "info" };
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        let crates = [
            "whisburn",
            "whisburn_cli",
            "whisburn_server",
            "whisburn_engine",
            "whisburn_models",
            "whisburn_audio",
        ];
        let spec = crates
            .iter()
            .map(|c| format!("{c}={default}"))
            .collect::<Vec<_>>()
            .join(",");
        let _ = verbose;
        EnvFilter::new(spec)
    });
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

fn vram_policy_from_cli(reserve_mb: Option<u32>, reserve_percent: Option<u32>) -> VramPolicy {
    let env = VramPolicy::from_env();
    VramPolicy {
        reserve_mb: reserve_mb.unwrap_or(env.reserve_mb),
        reserve_percent: reserve_percent.unwrap_or(env.reserve_percent).min(100),
    }
}

fn main() -> Result<()> {
    // Before the tokio runtime starts threads. wgpu dlopens libvulkan later
    // and reads whatever LD_LIBRARY_PATH is set to here.
    whisburn_engine::gpu_env::prepare_gpu_runtime();
    async_main()
}

#[tokio::main]
async fn async_main() -> Result<()> {
    let cli = Cli::parse();
    let (verbose, debug) = match &cli.command {
        Commands::Serve { verbose, debug, .. }
        | Commands::Transcribe { verbose, debug, .. }
        | Commands::Verify { verbose, debug, .. } => (*verbose, *debug),
        Commands::Models {
            action: ModelsCommand::Download { verbose, .. },
        } => (*verbose, false),
        _ => (false, false),
    };
    init_tracing(verbose, debug);

    match cli.command {
        Commands::Serve {
            port,
            model,
            device,
            hf_token,
            verbose,
            debug,
            no_open,
            preload_models,
            vram_reserve_mb,
            vram_reserve_percent,
            idle_unload_secs,
            max_parallel_jobs,
        } => {
            let file_settings = load_settings();
            let preload = preload_models.map(|names| PreloadModels::from_cli_values(Some(names)));
            let mut config = ServerConfig::from_settings_and_cli(
                file_settings,
                port,
                model,
                preload,
                device,
                hf_token,
                verbose,
                debug,
                !no_open,
            );
            if let Some(mb) = vram_reserve_mb {
                config.vram_reserve_mb = mb;
            }
            if let Some(pct) = vram_reserve_percent {
                config.vram_reserve_percent = pct.min(100);
            }
            if let Some(secs) = idle_unload_secs {
                config.idle_unload =
                    whisburn_engine::idle::IdleUnloadPolicy::from_secs_override(Some(secs));
            }
            if let Some(n) = max_parallel_jobs {
                config.max_parallel_jobs =
                    whisburn_engine::placement::parallelism_limit(n) as u32;
            }
            whisburn_server::run_server(config).await?;
        }
        #[cfg(feature = "app")]
        Commands::App => {
            whisburn_app::run()?;
        }
        Commands::Models { action } => match action {
            ModelsCommand::Download {
                name,
                force,
                hf_token,
                verbose,
            } => {
                let path = download_model(
                    &name,
                    &DownloadOptions {
                        hf_token,
                        verbose,
                        force,
                        progress: None,
                    },
                )?;
                println!("model ready at {}", path.display());
            }
            ModelsCommand::List => {
                for m in whisburn_engine::model::registry::MODEL_REGISTRY {
                    println!(
                        "{:<24} {:<8} vram={:>4}MB ready={}",
                        m.name,
                        format!("{:?}", m.category),
                        m.vram_mb,
                        m.burn_ready
                    );
                }
            }
        },
        Commands::Transcribe {
            input,
            model,
            language,
            task,
            format,
            device,
            vram_reserve_mb,
            vram_reserve_percent,
            hf_token,
            timestamps,
            orchestrate,
            scout_model,
            beam_size,
            max_tokens,
            sentences,
            summarize,
            summarizer,
            verbose,
            debug,
        } => {
            let speech_task = SpeechTask::parse(&task).unwrap_or(SpeechTask::Transcribe);
            let manager = ModelManager::new_with_policy(
                device,
                verbose,
                debug,
                vram_policy_from_cli(vram_reserve_mb, vram_reserve_percent),
            )
            .with_hf_token(hf_token);

            // Fast probe for estimation before heavy work (scoped to avoid holding full audio bytes)
            let ext = std::path::Path::new(&input)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("wav");
            let audio_dur = {
                let b = std::fs::read(&input).map_err(|e| anyhow!("failed to read input: {e}"))?;
                whisburn_audio::probe_bytes_duration_secs(&b, ext)
                    .ok()
                    .flatten()
            };

            if let Some(d) = audio_dur {
                let est = estimate_transcribe_secs(d, &model);
                eprintln!("Audio duration: ~{:.1}s | Estimated time for {}: ~{:.0}s (rough, hardware dependent)", d, model, est);
            } else {
                eprintln!("Starting transcription with {}...", model);
            }

            manager.ensure_model(&model).await?;

            let start = std::time::Instant::now();
            let waveform = whisburn_audio::decode_to_mono_pcm(std::path::Path::new(&input))?;
            let output_format = OutputFormat::parse(&format).ok_or_else(|| {
                anyhow::anyhow!(
                    "unsupported format '{}'. Use json, txt, srt or vtt.",
                    format
                )
            })?;

            let include_timestamps =
                timestamps.unwrap_or_else(|| !model.ends_with("_en") && !model.contains(".en"));

            let options = TaskOptions {
                task: speech_task,
                language: whisburn_core::LanguageCode::new(language),
                include_timestamps,
                orchestrate,
                scout_model: Some(scout_model),
                beam_size,
                max_tokens,
                group_into_sentences: sentences,
                ..TaskOptions::default()
            };

            // Simple live elapsed reporter (non-blocking ticker)
            let ticker_model = model.clone();
            let ticker_dur = audio_dur;
            let ticker = tokio::spawn(async move {
                let t0 = std::time::Instant::now();
                loop {
                    tokio::time::sleep(std::time::Duration::from_millis(750)).await;
                    let el = t0.elapsed().as_secs_f64();
                    if let Some(d) = ticker_dur {
                        let est = estimate_transcribe_secs(d, &ticker_model);
                        let rem = (est - el).max(0.0);
                        eprint!(
                            "\r[elapsed {:.0}s | est. remaining ~{:.0}s] processing...   ",
                            el, rem
                        );
                    } else {
                        eprint!("\r[elapsed {:.0}s] processing...   ", el);
                    }
                }
            });

            let result = manager
                .process_waveform(&model, waveform.samples, waveform.sample_rate, &options)
                .await?;

            // Stop the ticker
            ticker.abort();
            let _ = ticker.await;
            eprint!("\r"); // clear the line

            let elapsed = start.elapsed().as_secs_f64();
            eprintln!("Transcription finished in {:.1}s", elapsed);

            let encoded = whisburn_audio::encode_result_with_options(
                &result,
                output_format,
                include_timestamps,
                sentences,
            )?;
            std::io::Write::write_all(&mut std::io::stdout(), &encoded)?;

            if summarize {
                eprintln!("Summarizing transcript with {summarizer}…");
                let summary = manager.summarize_with_model(&summarizer, &result.text, None)?;
                let stem = std::path::Path::new(&input)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("transcript");
                let parent = std::path::Path::new(&input)
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or_else(|| std::path::Path::new("."));
                let sum_path = parent.join(format!("{stem}_summary.txt"));
                std::fs::write(&sum_path, summary)?;
                eprintln!("Wrote summary to {}", sum_path.display());
            }
        }
        Commands::Verify {
            input,
            models,
            device,
            vram_reserve_mb,
            vram_reserve_percent,
            hf_token,
            diarize,
            sentences,
            verbose,
            debug,
            out_dir,
        } => {
            let sample = input.unwrap_or_else(|| "samples/jfk.wav".to_string());
            let out_base = out_dir.unwrap_or_else(|| "verify-outputs".to_string());
            std::fs::create_dir_all(&out_base)?;
            let mgr = ModelManager::new_with_policy(
                device.clone(),
                verbose,
                debug,
                vram_policy_from_cli(vram_reserve_mb, vram_reserve_percent),
            )
            .with_hf_token(hf_token.clone());

            let model_list: Vec<String> = if models.eq_ignore_ascii_case("burn-ready") {
                whisburn_engine::model::registry::MODEL_REGISTRY
                    .iter()
                    .filter(|m| m.burn_ready)
                    .map(|m| m.name.to_string())
                    .collect()
            } else {
                models
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            };

            let formats = ["json", "txt", "srt"];
            let mut summary = Vec::new();

            let waveform = whisburn_audio::decode_to_mono_pcm(std::path::Path::new(&sample))?;

            for m in &model_list {
                if let Err(e) = mgr.ensure_model(m).await {
                    eprintln!("skip {} (not ready): {}", m, e);
                    continue;
                }
                for &fmt in &formats {
                    for &do_diar in &[false, diarize] {
                        for &do_sent in &[false, sentences] {
                            let task = if do_diar {
                                SpeechTask::Diarize
                            } else {
                                SpeechTask::Transcribe
                            };
                            let opts = TaskOptions {
                                task,
                                language: whisburn_core::LanguageCode::new("en".to_string()),
                                include_timestamps: true,
                                group_into_sentences: do_sent,
                                ..TaskOptions::default()
                            };
                            let res = match mgr
                                .process_waveform(
                                    m,
                                    waveform.samples.clone(),
                                    waveform.sample_rate,
                                    &opts,
                                )
                                .await
                            {
                                Ok(r) => r,
                                Err(e) => {
                                    eprintln!("  {} failed: {}", m, e);
                                    continue;
                                }
                            };

                            let of = OutputFormat::parse(fmt).unwrap();
                            let bytes = whisburn_audio::encode_result_with_options(
                                &res, of, true, do_sent,
                            )?;
                            let tag = format!(
                                "{m}__{}__{}{}{}",
                                fmt,
                                if do_diar { "diarize_" } else { "" },
                                if do_sent { "sentences_" } else { "" },
                                if do_diar && do_sent { "" } else { "" }
                            )
                            .trim_end_matches('_')
                            .to_string();
                            let fname = format!(
                                "{}/{}_{}.{}",
                                out_base,
                                tag.replace(['/', ':'], "_"),
                                "out",
                                fmt
                            );
                            std::fs::write(&fname, &bytes)?;
                            let preview = String::from_utf8_lossy(&bytes)
                                .chars()
                                .take(160)
                                .collect::<String>();
                            summary.push((
                                m.clone(),
                                fmt.to_string(),
                                do_diar,
                                do_sent,
                                fname,
                                preview,
                            ));
                        }
                    }
                }
            }

            println!("\n=== Verification Summary ===");
            for (m, f, d, s, path, prev) in &summary {
                println!(
                    "model={} fmt={} diarize={} sentences={} -> {} | {}",
                    m,
                    f,
                    d,
                    s,
                    path,
                    prev.replace('\n', " ")
                );
            }
            println!("Wrote {} artifacts to {}", summary.len(), out_base);
        }
    }

    Ok(())
}
