use axum::http::{header, HeaderValue};
use axum::response::{IntoResponse, Response};

const INDEX_HTML: &str = include_str!("../static/index.html");
const JFK_WAV: &[u8] = include_bytes!("../../../samples/jfk.wav");

pub async fn app_page() -> Response {
    (
        [
            (header::CACHE_CONTROL, HeaderValue::from_static("no-store")),
            (header::CONTENT_TYPE, HeaderValue::from_static("text/html; charset=utf-8")),
        ],
        INDEX_HTML,
    )
        .into_response()
}

pub async fn sample_jfk() -> Response {
    (
        [
            (header::CONTENT_TYPE, HeaderValue::from_static("audio/wav")),
            (
                header::CONTENT_DISPOSITION,
                HeaderValue::from_static("inline; filename=\"jfk.wav\""),
            ),
            (
                header::CACHE_CONTROL,
                HeaderValue::from_static("public, max-age=86400"),
            ),
        ],
        JFK_WAV,
    )
        .into_response()
}