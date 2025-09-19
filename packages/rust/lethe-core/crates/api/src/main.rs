use lethe_api::{create_app, AppState};
use lethe_shared::LetheConfig;
use std::{net::SocketAddr, path::PathBuf, sync::Arc};
use tokio::net::TcpListener;
use tower::ServiceBuilder;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(clap::Parser)]
#[command(name = "lethe-api")]
#[command(about = "Lethe RAG System API Server")]
struct Args {
    /// Server host
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Server port
    #[arg(long, default_value = "3000")]
    port: u16,

    /// Log level
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Configuration file path
    #[arg(long)]
    config: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args = <Args as clap::Parser>::parse();

    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                format!(
                    "lethe_api={},tower_http=debug,axum::rejection=trace",
                    args.log_level
                )
                .into()
            }),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting Lethe API server...");

    // Load configuration
    let raw_config = load_configuration(args.config.as_deref()).await?;
    let resolved_config = Arc::new(
        raw_config
            .resolve()
            .map_err(|err| -> Box<dyn std::error::Error> { Box::new(err) })?,
    );

    let storage_root = PathBuf::from(&resolved_config.storage.index_root);
    tracing::info!(root = %storage_root.display(), "Using parquet storage root");
    let app_state = AppState::initialise(resolved_config.clone(), &storage_root)
        .await
        .map_err(|err| -> Box<dyn std::error::Error> { Box::new(err) })?;

    // Perform health check
    match app_state.health_check().await {
        Ok(health) => {
            tracing::info!(?health, "Health check passed");
        }
        Err(e) => {
            tracing::error!(error = %e, "Health check failed");
            return Err(e.into());
        }
    }

    // Create application
    let app = create_app(app_state).layer(ServiceBuilder::new().layer(TraceLayer::new_for_http()));

    // Start server
    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    tracing::info!(addr = %addr, "Server starting");

    let listener = TcpListener::bind(addr).await?;
    tracing::info!("Server ready to accept connections");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("Server shutdown complete");
    Ok(())
}

/// Load configuration from file or use defaults
async fn load_configuration(
    config_path: Option<&str>,
) -> Result<LetheConfig, Box<dyn std::error::Error>> {
    if let Some(path) = config_path {
        tracing::info!(path = %path, "Loading configuration from file");
        let path_buf = std::path::PathBuf::from(path);
        let config = LetheConfig::from_file(&path_buf)?;
        Ok(config)
    } else {
        tracing::info!("Using default configuration");
        Ok(LetheConfig::default())
    }
}

/// Graceful shutdown signal handler
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("Received Ctrl+C, starting graceful shutdown");
        },
        _ = terminate => {
            tracing::info!("Received terminate signal, starting graceful shutdown");
        },
    }
}
