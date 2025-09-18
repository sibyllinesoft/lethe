use lethe_api::{create_app, AppState};
use lethe_domain::{
    EmbeddingConfig as DomainEmbeddingConfig, EmbeddingProvider as DomainEmbeddingProvider,
    EmbeddingServiceFactory, PipelineConfig, PipelineFactory,
};
use lethe_infrastructure::{
    DatabaseManager, PgChunkRepository, PgEmbeddingRepository, PgMessageRepository,
    PgSessionRepository,
};
use lethe_shared::{EmbeddingProvider as SharedEmbeddingProvider, LetheConfig};
use std::{net::SocketAddr, sync::Arc};
use tokio::net::TcpListener;
use tower::ServiceBuilder;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(clap::Parser)]
#[command(name = "lethe-api")]
#[command(about = "Lethe RAG System API Server")]
struct Args {
    /// Database URL
    #[arg(long, env = "DATABASE_URL")]
    database_url: Option<String>,

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

    /// Preload multiple repositories (comma-separated paths)
    #[arg(long, value_delimiter = ',')]
    preload_repos: Vec<String>,

    /// Preload a single repository (can be used multiple times)
    #[arg(long)]
    preload_repo: Vec<String>,

    /// Skip repository preloading even if configured
    #[arg(long)]
    skip_preload: bool,
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
    let mut config = load_configuration(args.config.as_deref()).await?;

    // Merge command line repository arguments
    merge_cli_repos(&mut config, &args)?;

    let domain_embedding_config = DomainEmbeddingConfig {
        provider: match &config.embedding.provider {
            SharedEmbeddingProvider::Ollama { base_url, model } => {
                DomainEmbeddingProvider::Ollama {
                    base_url: base_url.clone(),
                    model: model.clone(),
                }
            }
            SharedEmbeddingProvider::Fallback => DomainEmbeddingProvider::Fallback,
        },
        model_name: match &config.embedding.provider {
            SharedEmbeddingProvider::Ollama { model, .. } => model.clone(),
            SharedEmbeddingProvider::Fallback => "fallback".to_string(),
        },
        dimension: config.embedding.dimension,
        batch_size: 32,
        timeout_ms: config.embedding.timeout_ms,
    };

    let config = Arc::new(config);

    // Initialize database
    let database_url = args
        .database_url
        .or_else(|| std::env::var("DATABASE_URL").ok())
        .unwrap_or_else(|| config.database.connection_url());

    tracing::info!(url = %database_url, "Connecting to database");
    let db_manager = Arc::new(DatabaseManager::new(&database_url).await?);

    // Create repositories
    let message_repository = Arc::new(PgMessageRepository::new(db_manager.pool().clone()));
    let chunk_repository = Arc::new(PgChunkRepository::new(db_manager.pool().clone()));
    let embedding_repository = Arc::new(PgEmbeddingRepository::new(db_manager.pool().clone()));
    let session_repository = Arc::new(PgSessionRepository::new(db_manager.pool().clone()));

    // Create embedding service
    let embedding_service = EmbeddingServiceFactory::create(&domain_embedding_config).await?;

    // Create query pipeline
    let mut pipeline_config = PipelineConfig::default();
    if let Some(features) = &config.features {
        pipeline_config.enable_hyde = features.enable_hyde;
        pipeline_config.enable_query_understanding = features.enable_query_understanding;
        pipeline_config.enable_ml_prediction = features.enable_ml_prediction;
        pipeline_config.rerank_enabled = features.enable_state_tracking;
    }

    let query_pipeline = Arc::new(PipelineFactory::create_pipeline(
        pipeline_config,
        chunk_repository.clone(),
        embedding_service.clone(),
        None, // No LLM service for now
        None, // No reranking service for now
    ));

    // Create application state
    let app_state = AppState::new(
        config.clone(),
        db_manager.clone(),
        message_repository,
        chunk_repository,
        embedding_repository,
        session_repository,
        embedding_service,
        None, // No LLM service
        None, // No reranking service
        query_pipeline,
    );

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

    if !args.skip_preload {
        tracing::info!(
            "Repository preloading is not wired into the current build;--skip-preload is ignored"
        );
    }

    // Create application
    let app = create_app(app_state).layer(ServiceBuilder::new().layer(TraceLayer::new_for_http()));

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
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

/// Merge command line repository arguments into configuration
fn merge_cli_repos(
    _config: &mut LetheConfig,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    if !args.preload_repos.is_empty() || !args.preload_repo.is_empty() {
        tracing::warn!("Repository preloading flags are currently ignored in this build");
    }
    Ok(())
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
