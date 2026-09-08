use std::{net::SocketAddr, sync::Arc, time::Instant};

use axum::{
    Json, Router,
    body::Body,
    extract::{ConnectInfo, Request, State},
    http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures_util::StreamExt;
use http_body_util::{BodyExt, LengthLimitError, Limited};
use multer::{Constraints, Multipart, SizeLimit};
use serde_json::json;
use tower_http::cors::{Any, CorsLayer};

use crate::{
    auth::Authenticator,
    config::{Config, ImageLimits},
    error::{AppError, Result},
    scheduler::{CompletedJob, Scheduler, Submission},
    types::{Model, WorkClass},
};

#[derive(Clone)]
struct HttpState {
    config: Arc<Config>,
    scheduler: Scheduler,
}

#[derive(Clone, Copy)]
struct RequestStarted(Instant);

pub fn router(config: Arc<Config>, scheduler: Scheduler) -> Router {
    let auth = Arc::new(Authenticator::new(config.clone()));
    let state = HttpState {
        config: config.clone(),
        scheduler,
    };
    let app = Router::new()
        .route(
            "/remove-bg",
            post(remove_bg).route_layer(middleware::from_fn_with_state(auth, authenticate)),
        )
        .route("/health", get(health))
        .with_state(state);
    if config.cors_origins.is_empty() {
        return app;
    }
    let cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST])
        .allow_headers([HeaderName::from_static("x-api-key"), header::CONTENT_TYPE])
        .expose_headers([
            HeaderName::from_static("x-model-used"),
            HeaderName::from_static("x-admission-time-ms"),
            HeaderName::from_static("x-decode-time-ms"),
            HeaderName::from_static("x-queue-time-ms"),
            HeaderName::from_static("x-inference-time-ms"),
            HeaderName::from_static("x-encode-time-ms"),
        ]);
    let cors = if config.cors_origins.as_slice() == ["*"] {
        cors.allow_origin(Any)
    } else {
        cors.allow_origin(
            config
                .cors_origins
                .iter()
                .map(|origin| {
                    HeaderValue::from_str(origin).expect("CORS-Ursprung wurde beim Start validiert")
                })
                .collect::<Vec<_>>(),
        )
    };
    // Work-Class remains private to authenticated server-to-server requests.
    app.layer(cors)
}

async fn authenticate(
    State(auth): State<Arc<Authenticator>>,
    mut request: Request,
    next: Next,
) -> Response {
    request
        .extensions_mut()
        .insert(RequestStarted(Instant::now()));
    let Some(ConnectInfo(peer)) = request.extensions().get::<ConnectInfo<SocketAddr>>() else {
        tracing::error!("Direkte Clientadresse fehlt im HTTP-Dienst");
        return AppError::Internal.into_response();
    };
    if let Err(error) = auth.authenticate(request.headers(), peer.ip()) {
        return error.into_response();
    }
    next.run(request).await
}

async fn health(State(state): State<HttpState>) -> Response {
    let ready = state.scheduler.ready();
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(json!({
            "status": if ready { "ok" } else { "unavailable" },
            "ready": ready,
            "inference_provider": state.scheduler.provider(),
            "models": state.scheduler.model_status(),
            "gpu_queue": state.scheduler.status(),
        })),
    )
        .into_response()
}

async fn remove_bg(State(state): State<HttpState>, request: Request) -> Response {
    match process_upload(state, request).await {
        Ok(response) => response,
        Err(error) => error.into_response(),
    }
}

async fn process_upload(state: HttpState, request: Request) -> Result<Response> {
    let started = request
        .extensions()
        .get::<RequestStarted>()
        .ok_or(AppError::Internal)?
        .0;
    let (model, class) = request_options(&request)?;
    let deadline = started
        .checked_add(state.config.timeout(model))
        .ok_or(AppError::Internal)?;
    let boundary = upload_boundary(request.headers(), state.config.image.max_multipart_bytes)?;
    if Instant::now() >= deadline {
        return Err(AppError::UploadTimeout);
    }
    // Ownership spans upload, submission and every error/cancellation boundary.
    // No body extractor or body poll is allowed before this reservation.
    let admission = state.scheduler.try_admit(model)?;
    let (input, content_type) =
        receive_upload(request.into_body(), boundary, &state.config.image, deadline).await?;
    let completed = state
        .scheduler
        .submit(
            admission,
            Submission {
                input,
                content_type,
                model,
                class,
                started,
                deadline,
            },
        )
        .wait()
        .await?;
    png_response(completed)
}

fn request_options(request: &Request) -> Result<(Model, WorkClass)> {
    let pairs: Vec<(String, String)> =
        serde_urlencoded::from_str(request.uri().query().unwrap_or(""))
            .map_err(|_| AppError::InvalidRequest)?;
    let mut selected = None;
    for (name, value) in pairs {
        if name == "model" {
            if selected.is_some() {
                return Err(AppError::InvalidRequest);
            }
            selected = Some(match value.as_str() {
                "fast" => Model::Fast,
                "quality" => Model::Quality,
                _ => return Err(AppError::InvalidRequest),
            });
        }
    }
    let mut classes = request.headers().get_all("x-backremove-work-class").iter();
    let class = match classes.next().map(HeaderValue::as_bytes) {
        None | Some(b"foreground") => WorkClass::Foreground,
        Some(b"background") => WorkClass::Background,
        _ => return Err(AppError::InvalidRequest),
    };
    if classes.next().is_some() {
        return Err(AppError::InvalidRequest);
    }
    Ok((selected.unwrap_or(Model::Fast), class))
}

fn upload_boundary(headers: &HeaderMap, whole_limit: usize) -> Result<String> {
    let mut lengths = headers.get_all(header::CONTENT_LENGTH).iter();
    if let Some(length) = lengths.next() {
        let length: u64 = length
            .to_str()
            .ok()
            .and_then(|value| value.parse().ok())
            .ok_or(AppError::InvalidRequest)?;
        if lengths.next().is_some() {
            return Err(AppError::InvalidRequest);
        }
        if length > whole_limit as u64 {
            return Err(AppError::TooLarge);
        }
    }
    let mut types = headers.get_all(header::CONTENT_TYPE).iter();
    let content_type = types
        .next()
        .and_then(|value| value.to_str().ok())
        .ok_or(AppError::InvalidRequest)?;
    if types.next().is_some() {
        return Err(AppError::InvalidRequest);
    }
    let boundary = multer::parse_boundary(content_type).map_err(|_| AppError::InvalidRequest)?;
    // RFC 2046: bounded boundary length, no control characters or trailing space.
    if boundary.is_empty()
        || boundary.len() > 70
        || boundary.ends_with(' ')
        || !boundary
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"'()+_,-./:=? ".contains(&byte))
    {
        return Err(AppError::InvalidRequest);
    }
    Ok(boundary)
}

async fn receive_upload(
    body: Body,
    boundary: String,
    limits: &ImageLimits,
    deadline: Instant,
) -> Result<(Vec<u8>, String)> {
    if Instant::now() >= deadline {
        return Err(AppError::UploadTimeout);
    }
    let result = tokio::time::timeout_at(
        tokio::time::Instant::from_std(deadline),
        parse_upload(body, boundary, limits),
    )
    .await
    .map_err(|_| AppError::UploadTimeout)?;
    // A wholly ready stream can finish without yielding to Tokio's timer.
    if Instant::now() >= deadline {
        return Err(AppError::UploadTimeout);
    }
    result
}

async fn parse_upload(
    body: Body,
    boundary: String,
    limits: &ImageLimits,
) -> Result<(Vec<u8>, String)> {
    let mut stream = Limited::new(body, limits.max_multipart_bytes).into_data_stream();
    let constraints = Constraints::new().size_limit(
        SizeLimit::new()
            .whole_stream(limits.max_multipart_bytes as u64)
            .per_field(limits.max_file_bytes as u64),
    );
    let mut multipart = Multipart::with_constraints(&mut stream, boundary, constraints);
    let mut field = multipart
        .next_field()
        .await
        .map_err(multipart_error)?
        .ok_or(AppError::InvalidRequest)?;
    if field.name() != Some("file") || field.file_name().is_none() {
        return Err(AppError::InvalidRequest);
    }
    let content_type = field
        .content_type()
        .map(|mime| mime.as_ref())
        .ok_or(AppError::UnsupportedMedia)?;
    if !matches!(
        content_type,
        "image/jpeg" | "image/png" | "image/webp" | "image/gif" | "image/avif" | "image/svg+xml"
    ) {
        return Err(AppError::UnsupportedMedia);
    }
    let content_type = content_type.to_owned();
    let mut input = Vec::new();
    while let Some(chunk) = field.chunk().await.map_err(multipart_error)? {
        let needed = input
            .len()
            .checked_add(chunk.len())
            .ok_or(AppError::TooLarge)?;
        if needed > limits.max_file_bytes {
            return Err(AppError::TooLarge);
        }
        if needed > input.capacity() {
            let capacity = needed
                .max(input.capacity().saturating_mul(2))
                .min(limits.max_file_bytes);
            input
                .try_reserve_exact(capacity - input.len())
                .map_err(|_| {
                    tracing::error!("Speicherreservierung für Upload fehlgeschlagen");
                    AppError::Internal
                })?;
        }
        input.extend_from_slice(&chunk);
    }
    drop(field);
    if multipart
        .next_field()
        .await
        .map_err(multipart_error)?
        .is_some()
    {
        return Err(AppError::InvalidRequest);
    }
    drop(multipart);
    // Multer stops at the terminating boundary. The original stream must still
    // reach EOF: epilogue bytes count, and a stalled epilogue is an upload timeout.
    while let Some(chunk) = stream.next().await {
        chunk.map_err(body_error)?;
    }
    if input.is_empty() {
        return Err(AppError::InvalidImage);
    }
    Ok((input, content_type))
}

fn body_error(error: Box<dyn std::error::Error + Send + Sync>) -> AppError {
    if error.is::<LengthLimitError>() {
        AppError::TooLarge
    } else {
        tracing::debug!("Upload-Datenstrom wurde unterbrochen");
        AppError::InvalidRequest
    }
}

fn multipart_error(error: multer::Error) -> AppError {
    match error {
        multer::Error::FieldSizeExceeded { .. } | multer::Error::StreamSizeExceeded { .. } => {
            AppError::TooLarge
        }
        multer::Error::StreamReadFailed(error) => body_error(error),
        multer::Error::LockFailure => {
            tracing::error!("Multipart-Zustand konnte nicht exklusiv gelesen werden");
            AppError::Internal
        }
        _ => AppError::InvalidRequest,
    }
}

fn png_response(job: CompletedJob) -> Result<Response> {
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("image/png"));
    headers.insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_static("attachment; filename=no-bg.png"),
    );
    headers.insert("x-model-used", HeaderValue::from_static(job.model.as_str()));
    for (name, value) in [
        ("x-admission-time-ms", job.timings.admission_ms),
        ("x-decode-time-ms", job.timings.decode_ms),
        ("x-queue-time-ms", job.timings.queue_ms),
        ("x-inference-time-ms", job.timings.inference_ms),
        ("x-encode-time-ms", job.timings.encode_ms),
    ] {
        if !value.is_finite() || value < 0.0 {
            tracing::error!(stage = name, "Ungültiger interner Zeitmesswert");
            return Err(AppError::Internal);
        }
        headers.insert(
            name,
            HeaderValue::from_str(&format!("{value:.1}")).map_err(|_| AppError::Internal)?,
        );
    }
    // Moving Bytes (not rebuilding/copying its buffer) retains the scheduler's
    // admission and memory ownership until the final transport consumer drops it.
    let mut response = Response::new(Body::from(job.body));
    *response.headers_mut() = headers;
    Ok(response)
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status =
            StatusCode::from_u16(self.status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let mut response = (
            status,
            Json(json!({ "detail": self.message(), "code": self.code() })),
        )
            .into_response();
        if let Some(seconds) = self.retry_after() {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from(seconds));
        }
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{config::Device, types::Timings};
    use bytes::Bytes;
    use futures_util::stream;
    use std::{
        convert::Infallible,
        sync::atomic::{AtomicBool, Ordering},
        time::Duration,
    };
    use tower::ServiceExt;

    fn config() -> Arc<Config> {
        Arc::new(Config {
            bind: "127.0.0.1:8585".parse().unwrap(),
            api_key: b"test-key".to_vec(),
            cors_origins: Vec::new(),
            trusted_proxies: Vec::new(),
            manifest: "unused-in-http-tests.json".into(),
            device: Device::Cpu,
            quality_enabled: false,
            queue_capacity: 8,
            max_jobs: 9,
            input_budget_bytes: 256 * 1024 * 1024,
            memory_budget_bytes: 1024 * 1024 * 1024,
            fast_timeout: Duration::from_secs(9),
            quality_timeout: Duration::from_secs(29),
            shutdown_grace: Duration::from_secs(30),
            image: ImageLimits::default(),
            auth_max_entries: 10,
        })
    }

    fn multipart(data: &str) -> String {
        format!(
            "--test\r\nContent-Disposition: form-data; name=\"file\"; filename=\"image.png\"\r\nContent-Type: image/png\r\n\r\n{data}\r\n--test--\r\n"
        )
    }

    fn limits(file: usize, whole: usize) -> ImageLimits {
        ImageLimits {
            max_file_bytes: file,
            max_multipart_bytes: whole,
            ..ImageLimits::default()
        }
    }

    fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(2)
    }

    #[tokio::test]
    async fn authentication_precedes_query_validation_and_never_polls_rejected_bodies() {
        async fn downstream(request: Request) -> Response {
            if let Err(error) = request_options(&request) {
                return error.into_response();
            }
            let _ = axum::body::to_bytes(request.into_body(), 1).await;
            StatusCode::NO_CONTENT.into_response()
        }
        let app = Router::new().route("/remove-bg", post(downstream)).layer(
            middleware::from_fn_with_state(Arc::new(Authenticator::new(config())), authenticate),
        );
        for uri in ["/remove-bg?model=fast", "/remove-bg?model=QUALITY"] {
            let body = Body::from_stream(stream::poll_fn(
                |_| -> std::task::Poll<Option<std::result::Result<Bytes, Infallible>>> {
                    panic!("An unauthenticated body was polled");
                },
            ));
            let request = Request::builder()
                .method(Method::POST)
                .uri(uri)
                .extension(ConnectInfo("192.0.2.1:1234".parse::<SocketAddr>().unwrap()))
                .header("x-api-key", "incorrect")
                .header(header::CONTENT_TYPE, "not multipart")
                .body(body)
                .unwrap();
            let response = app.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
            let body = axum::body::to_bytes(response.into_body(), 1024)
                .await
                .unwrap();
            assert_eq!(
                serde_json::from_slice::<serde_json::Value>(&body).unwrap()["code"],
                "unauthorized"
            );
        }
        let request = Request::builder()
            .method(Method::POST)
            .uri("/remove-bg")
            .extension(ConnectInfo("192.0.2.2:1234".parse::<SocketAddr>().unwrap()))
            .header("x-api-key", "test-key")
            .header("x-api-key", "test-key")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            app.oneshot(request).await.unwrap().status(),
            StatusCode::UNAUTHORIZED
        );
    }

    #[test]
    fn model_and_work_class_reject_ambiguous_or_normalized_values() {
        for uri in [
            "/remove-bg?model=fast&model=quality",
            "/remove-bg?model=Fast",
            "/remove-bg?model=",
            "/remove-bg?model=quality%20",
        ] {
            let request = Request::builder().uri(uri).body(Body::empty()).unwrap();
            assert!(matches!(
                request_options(&request),
                Err(AppError::InvalidRequest)
            ));
        }
        let request = Request::builder()
            .uri("/remove-bg?model=quality")
            .header("x-backremove-work-class", "background")
            .body(Body::empty())
            .unwrap();
        assert!(matches!(
            request_options(&request),
            Ok((Model::Quality, WorkClass::Background))
        ));
        let duplicate = Request::builder()
            .header("x-backremove-work-class", "foreground")
            .header("x-backremove-work-class", "background")
            .body(Body::empty())
            .unwrap();
        assert!(matches!(
            request_options(&duplicate),
            Err(AppError::InvalidRequest)
        ));
        let normalized = Request::builder()
            .header("x-backremove-work-class", "Background")
            .body(Body::empty())
            .unwrap();
        assert!(matches!(
            request_options(&normalized),
            Err(AppError::InvalidRequest)
        ));
    }

    #[tokio::test]
    async fn file_limit_accepts_exact_bytes_across_split_boundaries_and_rejects_one_more() {
        let data = multipart("DATA");
        let chunks: Vec<_> = data
            .as_bytes()
            .chunks(1)
            .map(|chunk| Ok::<_, Infallible>(Bytes::copy_from_slice(chunk)))
            .collect();
        let (file, mime) = receive_upload(
            Body::from_stream(stream::iter(chunks)),
            "test".into(),
            &limits(4, data.len()),
            deadline(),
        )
        .await
        .unwrap();
        assert_eq!(file, b"DATA");
        assert_eq!(mime, "image/png");
        let oversized = receive_upload(
            Body::from(multipart("DATA!")),
            "test".into(),
            &limits(4, 1024),
            deadline(),
        )
        .await;
        assert!(matches!(oversized, Err(AppError::TooLarge)));
    }

    #[tokio::test]
    async fn total_stream_limit_counts_delayed_epilogue_without_content_length() {
        let data = multipart("DATA");
        let whole = data.len() + 3;
        let stream = stream::once(async move { Ok::<_, Infallible>(Bytes::from(data)) }).chain(
            stream::once(async {
                tokio::task::yield_now().await;
                Ok::<_, Infallible>(Bytes::from_static(b"four"))
            }),
        );
        let result = receive_upload(
            Body::from_stream(stream),
            "test".into(),
            &limits(4, whole),
            deadline(),
        )
        .await;
        assert!(matches!(result, Err(AppError::TooLarge)));
    }

    #[tokio::test(start_paused = true)]
    async fn upload_deadline_is_total_despite_regular_progress() {
        let data = multipart("DATA");
        let chunks: Vec<_> = data
            .as_bytes()
            .chunks(30)
            .map(Bytes::copy_from_slice)
            .collect();
        let stream = stream::iter(chunks).then(|chunk| async move {
            tokio::time::sleep(Duration::from_millis(40)).await;
            Ok::<_, Infallible>(chunk)
        });
        let result = receive_upload(
            Body::from_stream(stream),
            "test".into(),
            &limits(4, 1024),
            Instant::now() + Duration::from_millis(100),
        )
        .await;
        assert!(matches!(result, Err(AppError::UploadTimeout)));
    }

    #[tokio::test(start_paused = true)]
    async fn final_boundary_does_not_hide_a_stalled_http_body() {
        let stream = stream::iter([Ok::<_, Infallible>(Bytes::from(multipart("DATA")))])
            .chain(stream::pending());
        let result = receive_upload(
            Body::from_stream(stream),
            "test".into(),
            &limits(4, 1024),
            Instant::now() + Duration::from_millis(100),
        )
        .await;
        assert!(matches!(result, Err(AppError::UploadTimeout)));
    }

    #[tokio::test]
    async fn exactly_one_file_is_required_and_truncation_is_not_success() {
        let one = multipart("DATA");
        let second = one.replace("--test--\r\n", "") + &multipart("MORE");
        let result = receive_upload(
            Body::from(second),
            "test".into(),
            &limits(4, 1024),
            deadline(),
        )
        .await;
        assert!(matches!(result, Err(AppError::InvalidRequest)));
        let truncated = one.replace("\r\n--test--\r\n", "");
        let result = receive_upload(
            Body::from(truncated),
            "test".into(),
            &limits(4, 1024),
            deadline(),
        )
        .await;
        assert!(matches!(result, Err(AppError::InvalidRequest)));
        let result = receive_upload(
            Body::from("--test--\r\n"),
            "test".into(),
            &limits(4, 1024),
            deadline(),
        )
        .await;
        assert!(matches!(result, Err(AppError::InvalidRequest)));
    }

    #[tokio::test]
    async fn error_codes_distinguish_retryable_capacity_from_unavailable_and_deadlines() {
        for (error, status, code, retry) in [
            (AppError::Busy, 503, "busy", Some("2")),
            (AppError::Unavailable, 503, "model_unavailable", None),
            (AppError::Deadline, 504, "deadline_exceeded", None),
            (AppError::UploadTimeout, 408, "upload_timeout", None),
            (AppError::RateLimited, 429, "rate_limited", Some("900")),
        ] {
            let response = error.into_response();
            assert_eq!(response.status().as_u16(), status);
            assert_eq!(
                response
                    .headers()
                    .get(header::RETRY_AFTER)
                    .map(|value| value.to_str().unwrap()),
                retry
            );
            let bytes = axum::body::to_bytes(response.into_body(), 1024)
                .await
                .unwrap();
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["code"], code);
            assert!(body["detail"].is_string());
        }
    }

    #[tokio::test]
    async fn png_transport_retains_byte_owner_until_final_consumer_releases_it() {
        struct OwnedBytes {
            dropped: Arc<AtomicBool>,
        }
        impl AsRef<[u8]> for OwnedBytes {
            fn as_ref(&self) -> &[u8] {
                b"\x89PNG\r\n\x1a\n"
            }
        }
        impl Drop for OwnedBytes {
            fn drop(&mut self) {
                self.dropped.store(true, Ordering::SeqCst);
            }
        }
        let dropped = Arc::new(AtomicBool::new(false));
        let response = png_response(CompletedJob {
            body: Bytes::from_owner(OwnedBytes {
                dropped: dropped.clone(),
            }),
            model: Model::Quality,
            timings: Timings::default(),
        })
        .unwrap();
        assert_eq!(response.headers()[header::CONTENT_TYPE], "image/png");
        assert_eq!(
            response.headers()[header::CONTENT_DISPOSITION],
            "attachment; filename=no-bg.png"
        );
        assert_eq!(response.headers()["x-model-used"], "quality");
        assert!(!dropped.load(Ordering::SeqCst));
        let consumed = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        assert_eq!(&consumed[..], b"\x89PNG\r\n\x1a\n");
        assert!(!dropped.load(Ordering::SeqCst));
        drop(consumed);
        assert!(dropped.load(Ordering::SeqCst));
    }
}
