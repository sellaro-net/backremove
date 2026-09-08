use axum::serve::IncomingStream;
use backremove::{
    artifacts::ArtifactSet, config::Config, http, image_pipeline::ImagePipeline,
    inference::InferenceEngine, scheduler::Scheduler, transport::GracefulTcpListener,
};
use std::{
    future::IntoFuture,
    io::{Read, Write},
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpStream},
    process::ExitCode,
    sync::Arc,
    time::Duration,
};
use tower::Service;
use tracing_subscriber::EnvFilter;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> ExitCode {
    match dotenvy::dotenv() {
        Ok(_) => {}
        Err(dotenvy::Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => {
            eprintln!("Die lokale Umgebungskonfiguration konnte nicht gelesen werden.");
            return ExitCode::FAILURE;
        }
    }
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match arguments.as_slice() {
        [flag] if flag == "--version" => {
            println!("backremove {}", env!("CARGO_PKG_VERSION"));
            return ExitCode::SUCCESS;
        }
        [flag] if flag == "--help" || flag == "-h" => {
            println!(
                "BackRemove {} – nativer HTTP-Dienst\n\nAufruf: backremove [--healthcheck | --version | --help]\n\nKonfiguration über Umgebungsvariablen oder lokale .env:\nAPI_KEY (erforderlich), ARTIFACT_MANIFEST, HOST, PORT, INFERENCE_DEVICE,\nQUALITY_MODEL_ENABLED, QUEUE_CAPACITY, INPUT_BUDGET_MB, MEMORY_BUDGET_MB,\nFAST_TIMEOUT, QUALITY_TIMEOUT, SHUTDOWN_GRACE, TRUSTED_PROXIES, CORS_ORIGINS.\n\nPOST /remove-bg?model=fast|quality, GET /health\nX-API-Key erforderlich für die Verarbeitung. Built with DINOv3.",
                env!("CARGO_PKG_VERSION")
            );
            return ExitCode::SUCCESS;
        }
        [flag] if flag == "--healthcheck" => {
            return if healthcheck().is_ok() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            };
        }
        [] => {}
        _ => {
            eprintln!("Unbekannte Argumente. Hilfe: backremove --help");
            return ExitCode::FAILURE;
        }
    }
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .init();
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            tracing::error!(%error, "BackRemove wurde beendet.");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = Arc::new(Config::load().map_err(std::io::Error::other)?);
    // Reserve the address before allocating model resources. Nothing is served
    // until every artifact, font and enabled model has passed startup and warm-up.
    let listener = tokio::net::TcpListener::bind(config.bind).await?;
    tracing::info!(manifest = %config.manifest.display(), "Prüfe native Laufzeit und Modellpaket.");
    let artifacts = ArtifactSet::load(&config.manifest, config.device, config.quality_enabled)?;
    let pipeline = Arc::new(ImagePipeline::new(
        config.image.clone(),
        &artifacts.font_paths,
    )?);
    let engine = InferenceEngine::load(artifacts, config.device, config.quality_enabled)?;
    let scheduler = Scheduler::start(config.clone(), pipeline, engine)?;
    let router = http::router(config.clone(), scheduler.clone());
    let listener = GracefulTcpListener::new(listener, config.image.max_multipart_bytes);
    let mut make_service = router.into_make_service_with_connect_info::<SocketAddr>();
    let make_service =
        tower::service_fn(move |incoming: IncomingStream<'_, GracefulTcpListener>| {
            make_service.call(*incoming.remote_addr())
        });
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::watch::channel(false);
    let server = axum::serve(listener, make_service)
        .with_graceful_shutdown(async move {
            let _ = shutdown_rx.changed().await;
        })
        .into_future();
    tokio::pin!(server);
    tracing::info!(address = %config.bind, provider = scheduler.provider(), version = env!("CARGO_PKG_VERSION"), "BackRemove ist bereit.");
    let result = tokio::select! {
        result = &mut server => {
            scheduler.shutdown();
            result
        },
        _ = shutdown_signal() => {
            tracing::info!("Beende Annahme und warte auf laufende native Verarbeitung.");
            scheduler.shutdown();
            let _ = shutdown_tx.send(true);
            match tokio::time::timeout(config.shutdown_grace, async {
                let result = server.await;
                scheduler.join().await;
                result
            }).await {
                Ok(result) => result,
                Err(_) => {
                    tracing::error!(grace_seconds = config.shutdown_grace.as_secs_f64(), "Shutdown-Frist überschritten; native Arbeit wird nicht als freigegeben ausgegeben. Prozess wird beendet.");
                    // Native CUDA calls are not safely preemptible. Ending the owning
                    // process is the only hard boundary after the explicit grace.
                    std::process::exit(1);
                }
            }
        }
    };
    if tokio::time::timeout(config.shutdown_grace, scheduler.join())
        .await
        .is_err()
    {
        tracing::error!("Native Worker wurden nicht innerhalb der Shutdown-Frist beendet.");
        std::process::exit(1);
    }
    result?;
    Ok(())
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut terminate) => {
                tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = terminate.recv() => {} }
            }
            Err(error) => {
                tracing::error!(%error, "SIGTERM-Behandlung konnte nicht eingerichtet werden.");
                let _ = tokio::signal::ctrl_c().await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

fn healthcheck() -> Result<(), Box<dyn std::error::Error>> {
    let host: IpAddr = std::env::var("HOST")
        .unwrap_or_else(|_| "127.0.0.1".into())
        .parse()?;
    let host = match host {
        IpAddr::V4(ip) if ip.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V6(ip) if ip.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
        ip => ip,
    };
    let port: u16 = std::env::var("PORT")
        .unwrap_or_else(|_| "8585".into())
        .parse()?;
    let mut connection =
        TcpStream::connect_timeout(&SocketAddr::new(host, port), Duration::from_secs(2))?;
    connection.set_read_timeout(Some(Duration::from_secs(2)))?;
    connection.set_write_timeout(Some(Duration::from_secs(2)))?;
    connection
        .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")?;
    let mut response = Vec::with_capacity(64);
    connection.take(64).read_to_end(&mut response)?;
    if response.starts_with(b"HTTP/1.1 200 ") || response.starts_with(b"HTTP/1.0 200 ") {
        Ok(())
    } else {
        Err(std::io::Error::other("Der Dienst ist nicht bereit.").into())
    }
}
