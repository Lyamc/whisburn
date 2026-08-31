use axum::routing::{get, post};
use axum::Router;

use crate::handlers;
use crate::state::AppState;
use crate::web;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(web::app_page))
        .route("/app", get(web::app_page))
        .route("/samples/jfk.wav", get(web::sample_jfk))
        .route("/health", get(handlers::health))
        .route("/v1/config", get(handlers::server_config))
        .route("/v1/models", get(handlers::list_models))
        .route("/v1/models/{name}/ensure", post(handlers::ensure_model))
        .route("/v1/transcribe", post(handlers::transcribe))
        .route("/v1/transcribe/stream", post(handlers::transcribe_stream))
        .route("/v1/transcode", post(handlers::transcode))
        .route("/v1/transcode/stream", post(handlers::transcode_stream))
}