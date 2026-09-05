pub mod config;
pub mod gpu;
pub mod handlers;
pub mod jobs;
pub mod routes;
pub mod state;
pub mod upload;
pub mod web;

pub use config::ServerConfig;
pub use state::AppState;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::DefaultBodyLimit;
use axum::Router;
use whisburn_engine::model::registry;
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::cors::CorsLayer;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;
use tracing::info;

use whisburn_models::ModelManager;

pub async fn run_server(config: ServerConfig) -> anyhow::Result<()> {
    let manager = Arc::new(
        ModelManager::new(config.device.clone(), config.verbose, config.debug)
            .with_hf_token(config.hf_token.clone()),
    );

    let preload_names = config.preload_models.resolve_names(
        registry::MODEL_REGISTRY
            .iter()
            .filter(|m| m.burn_ready)
            .map(|m| m.name),
    );

    if !preload_names.is_empty() {
        let preload_manager = manager.clone();
        let names = preload_names.clone();
        tokio::spawn(async move {
            for name in names {
                match preload_manager.warm_model(&name).await {
                    Ok(()) => tracing::info!("preloaded model '{name}'"),
                    Err(e) => tracing::warn!("failed to preload model '{name}': {e}"),
                }
            }
        });
    }

    let gpu = gpu::probe_gpu();
    info!(
        gpu = %gpu.name,
        vram_mb = ?gpu.vram_mb,
        source = %gpu.source,
        "GPU detected for model compatibility hints"
    );

    let state = AppState {
        manager,
        max_upload_bytes: config.max_upload_bytes,
        max_audio_seconds: config.max_audio_seconds,
        default_model: config.default_model.clone(),
        preload_models: config.preload_models.clone(),
        gpu,
        job_semaphore: Arc::new(Semaphore::new(1)),
    };

    let app = Router::new()
        .merge(routes::router())
        .layer(CorsLayer::permissive())
        .layer(CatchPanicLayer::new())
        .layer(DefaultBodyLimit::max(config.max_upload_bytes))
        .layer(RequestBodyLimitLayer::new(config.max_upload_bytes))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    let local = format!("http://127.0.0.1:{}", config.port);
    println!("whisburn web UI: {local}/");
    println!("API:            {local}/v1/  (health at {local}/health)");
    println!("desktop client: whisburn app");
    info!("whisburn listening on {local} (bound {addr})");

    let listener = TcpListener::bind(addr).await?;
    if config.open_browser {
        open_browser(&format!("{local}/"));
    }
    axum::serve(listener, app).await?;
    Ok(())
}

fn open_browser(url: &str) {
    let result = {
        #[cfg(windows)]
        {
            std::process::Command::new("cmd")
                .args(["/C", "start", "", url])
                .spawn()
                .map(|_| ())
        }
        #[cfg(target_os = "macos")]
        {
            std::process::Command::new("open").arg(url).spawn().map(|_| ())
        }
        #[cfg(all(unix, not(target_os = "macos")))]
        {
            std::process::Command::new("xdg-open").arg(url).spawn().map(|_| ())
        }
        #[cfg(not(any(windows, unix)))]
        {
            Ok(())
        }
    };
    if let Err(e) = result {
        tracing::warn!("could not open browser for {url}: {e}");
        println!("Open {url} in a browser for the upload UI");
    }
}