use axum::body::Body;

use axum::http::{header, Request, StatusCode};
use http_body_util::BodyExt;
use tower::ServiceExt;

use whisburn_core::PreloadModels;
use whisburn_server::{routes, upload::attachment_filename, AppState};
use whisburn_models::ModelManager;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Semaphore;

fn sample_jfk_wav() -> Option<Vec<u8>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../samples/jfk.wav");
    std::fs::read(path).ok()
}

fn test_state() -> AppState {
    AppState {
        manager: Arc::new(ModelManager::new(None, false, false)),
        max_upload_bytes: 1024 * 1024,
        max_audio_seconds: 24 * 60 * 60,
        default_model: "tiny_en".to_string(),
        preload_models: PreloadModels::None,
        gpu: whisburn_server::gpu::GpuInfo {
            name: "test-gpu".into(),
            vram_mb: Some(8192),
            source: "test".into(),
        },
        job_semaphore: Arc::new(Semaphore::new(1)),
    }
}

fn multipart_audio_request(uri: &str, field: &str, filename: &str, bytes: &[u8]) -> Request<Body> {
    let boundary = "----whisburn-test-boundary";
    let mut body = Vec::new();
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\n\
             Content-Type: application/octet-stream\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());

    Request::builder()
        .method("POST")
        .uri(uri)
        .header(
            header::CONTENT_TYPE,
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body))
        .unwrap()
}

async fn json_body(response: axum::response::Response) -> serde_json::Value {
    let body = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&body).unwrap_or(serde_json::Value::Null)
}

#[tokio::test]
async fn health_endpoint_returns_ok() {
    let app = routes::router().with_state(test_state());

    let response = app
        .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let json = json_body(response).await;
    assert_eq!(json["status"], "ok");
    assert_eq!(json["service"], "whisburn");
}

#[tokio::test]
async fn root_endpoint_serves_web_ui() {
    let app = routes::router().with_state(test_state());

    let response = app
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "text/html; charset=utf-8"
    );
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let html = String::from_utf8_lossy(&body);
    assert!(html.contains("</style>"), "unclosed <style> makes the browser render a blank page");
    assert!(html.contains("whisburn"));
    assert!(html.contains("Process &amp; download"));
    assert!(html.contains("Batch queue"));
    assert!(html.contains("btn-output-folder"));
    assert!(html.contains("multiple"));
    assert!(html.contains("btn-sample"));
    assert!(html.contains("btn-preview"));
    assert!(html.contains("for=\"file-input\""));
    assert!(html.contains("progress-wrap"));
    assert!(html.contains("Preload model"));
    assert!(html.contains("progress-task"));
    assert!(html.contains("progress-overall"));
    assert!(html.contains("id=\"diarize\""));
    assert!(html.contains("id=\"file-extension\""));
    assert!(html.contains("btn-cancel"));
    assert!(html.contains("queue-card"));
    assert!(html.contains("model-meta"));
    assert!(html.contains("id=\"summarize\""), "summarize toggle");
    assert!(html.contains("id=\"steps\""), "wizard steps container");
    assert!(html.contains("data-step=\"1\""));
    assert!(html.contains("data-step=\"2\""));
    assert!(html.contains("data-step=\"3\""));
    assert!(html.contains("step-header"));
    assert!(html.contains("btn-step-continue"));
    assert!(html.contains("flex-direction: row"), "desktop steps sit in a row");
    assert!(html.contains("flex-direction: column"), "phone steps stack vertically");
}

#[tokio::test]
async fn sample_jfk_endpoint_returns_wav() {
    let app = routes::router().with_state(test_state());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/samples/jfk.wav")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "audio/wav"
    );
    let body = response.into_body().collect().await.unwrap().to_bytes();
    assert!(body.len() > 44);
    assert_eq!(&body[0..4], b"RIFF");
}

#[tokio::test]
async fn app_endpoint_serves_web_ui() {
    let app = routes::router().with_state(test_state());

    let response = app
        .oneshot(Request::builder().uri("/app").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "text/html; charset=utf-8"
    );
}

#[tokio::test]
async fn models_endpoint_lists_registry() {
    let app = routes::router().with_state(test_state());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
    assert!(!json.is_empty());
    assert!(json[0].get("name").is_some());
    assert!(json[0].get("loaded").is_some());
    assert_eq!(json[0]["loaded"], false);
}

#[tokio::test]
async fn models_list_includes_whisper_and_parakeet_entries() {
    let app = routes::router().with_state(test_state());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let body = response.into_body().collect().await.unwrap().to_bytes();
    let json: Vec<serde_json::Value> = serde_json::from_slice(&body).unwrap();
    let names: Vec<String> = json
        .iter()
        .filter_map(|m| m.get("name").and_then(|n| n.as_str()).map(String::from))
        .collect();

    assert!(names.contains(&"tiny_en".to_string()));
    assert!(names.contains(&"parakeet-tdt-0.6b-v3".to_string()));
}

#[tokio::test]
async fn config_endpoint_returns_default_model() {
    let app = routes::router().with_state(test_state());

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/config")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let json = json_body(response).await;
    assert_eq!(json["default_model"], "tiny_en");
    assert_eq!(json["preload_models"], "none");
    assert_eq!(json["max_audio_seconds"], 86400);
    assert!(json["loaded_models"].is_array());
    assert!(json["loaded_models"].as_array().unwrap().is_empty());
    assert!(json["gpu"].is_object());
    assert_eq!(json["gpu"]["vram_mb"], 8192);
}

#[tokio::test]
async fn transcribe_without_audio_returns_bad_request() {
    let app = routes::router().with_state(test_state());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/transcribe?model=tiny_en&format=txt")
                .header(header::CONTENT_TYPE, "multipart/form-data; boundary=x")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn transcode_without_audio_returns_bad_request() {
    let app = routes::router().with_state(test_state());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/transcode?format=wav")
                .header(header::CONTENT_TYPE, "multipart/form-data; boundary=x")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn transcode_stream_without_audio_returns_bad_request() {
    let app = routes::router().with_state(test_state());

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/transcode/stream?format=opus")
                .header(header::CONTENT_TYPE, "multipart/form-data; boundary=x")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn transcode_accepts_wav_upload() {
    let Some(wav) = sample_jfk_wav() else {
        return;
    };
    let app = routes::router().with_state(test_state());

    let response = app
        .oneshot(multipart_audio_request(
            "/v1/transcode?format=wav&download=false",
            "audio",
            "jfk.wav",
            &wav,
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "audio/wav"
    );
}

#[tokio::test]
async fn transcribe_accepts_wav_upload_and_returns_txt() {
    let Some(wav) = sample_jfk_wav() else {
        return;
    };
    let app = routes::router().with_state(test_state());

    let response = app
        .oneshot(multipart_audio_request(
            "/v1/transcribe?model=tiny_en&format=txt&language=en&download=false",
            "audio",
            "jfk.wav",
            &wav,
        ))
        .await
        .unwrap();

    if response.status() == StatusCode::INTERNAL_SERVER_ERROR {
        let json = json_body(response).await;
        let err = json["error"].as_str().unwrap_or("");
        assert!(
            err.contains("model") || err.contains("Model") || err.contains("not found"),
            "unexpected error: {err}"
        );
        return;
    }

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&body);
    assert!(!text.trim().is_empty());
}

#[tokio::test]
async fn ensure_model_endpoint_returns_path_when_ready() {
    let app = routes::router().with_state(test_state());
    if !whisburn_models::paths::is_model_ready("tiny_en") {
        return;
    }

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/models/tiny_en/ensure")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let json = json_body(response).await;
    assert_eq!(json["name"], "tiny_en");
    assert_eq!(json["ready"], true);
    assert!(json["path"].as_str().unwrap_or("").contains("tiny_en"));
}

#[test]
fn attachment_filename_sanitizes_stem() {
    assert_eq!(attachment_filename("my talk!", "json"), "my_talk_.json");
    assert_eq!(attachment_filename("", "txt"), "whisburn.txt");
}