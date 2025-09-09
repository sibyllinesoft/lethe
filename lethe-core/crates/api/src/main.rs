use axum::http::Method;
use lethe_api::{create_app, AppState};
use lethe_domain::{
    EmbeddingServiceFactory, OllamaEmbeddingService, FallbackEmbeddingService,
    PipelineFactory, PipelineConfig,
};
use lethe_infrastructure::{
    DatabaseManager, PgMessageRepository, PgChunkRepository, 
    PgEmbeddingRepository, PgSessionRepository,
};
use lethe_shared::{LetheConfig, EmbeddingConfig, EmbeddingProvider};
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
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args = <Args as clap::Parser>::parse();
    
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                format!("lethe_api={},tower_http=debug,axum::rejection=trace", args.log_level).into()
            }),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting Lethe API server...");

    // Load configuration
    let config = load_configuration(args.config.as_deref()).await?;
    let config = Arc::new(config);

    // Initialize database
    let database_url = args.database_url
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
    let embedding_service = Arc::new(create_embedding_service(&config.embedding).await?);

    // Create query pipeline
    let pipeline_config = PipelineConfig {
        enable_hyde: config.features.hyde_enabled,
        enable_query_understanding: true,
        enable_ml_prediction: true,
        max_candidates: config.retrieval.max_candidates,
        rerank_enabled: config.features.rerank_enabled,
        rerank_top_k: 20,
        timeout_seconds: config.timeouts.query_timeout as u64,
    };

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

    // Create application
    let app = create_app(app_state)
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
        );

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
async fn load_configuration(config_path: Option<&str>) -> Result<LetheConfig, Box<dyn std::error::Error>> {
    if let Some(path) = config_path {
        tracing::info!(path = %path, "Loading configuration from file");
        let content = tokio::fs::read_to_string(path).await?;
        let config: LetheConfig = serde_json::from_str(&content)?;
        Ok(config)
    } else {
        tracing::info!("Using default configuration");
        Ok(LetheConfig::default())
    }
}

/// Create embedding service from configuration
async fn create_embedding_service(
    config: &EmbeddingConfig,
) -> Result<Box<dyn lethe_domain::EmbeddingService>, Box<dyn std::error::Error>> {
    match &config.provider {
        EmbeddingProvider::Ollama { base_url, model } => {
            tracing::info!(provider = "ollama", model = %model, "Creating Ollama embedding service");
            let service = OllamaEmbeddingService::new(base_url.clone(), model.clone());
            Ok(Box::new(service))
        }
        EmbeddingProvider::Fallback => {
            tracing::info!(provider = "fallback", "Creating fallback embedding service");
            let service = FallbackEmbeddingService::new(384); // Default dimension
            Ok(Box::new(service))
        }
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