use lethe_api::{create_app, AppState};
use lethe_domain::{
    corpus::ParquetCorpus, EmbeddingConfig as DomainEmbeddingConfig,
    EmbeddingProvider as DomainEmbeddingProvider, EmbeddingRerankingService,
    EmbeddingServiceFactory, LlmServiceConfig, LlmServiceFactory, PipelineConfig, PipelineFactory,
    RerankingService,
};
use lethe_shared::{EmbeddingProvider as SharedEmbeddingProvider, LetheConfig};
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
    let config = load_configuration(args.config.as_deref()).await?;

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

    let storage_root = PathBuf::from(&config.storage.index_root);
    tracing::info!(root = %storage_root.display(), "Using parquet storage root");
    let corpus = Arc::new(ParquetCorpus::new(&storage_root));
    corpus.health_check().await?;

    // Create embedding service
    let embedding_service = EmbeddingServiceFactory::create(&domain_embedding_config).await?;

    // Construct optional LLM service (used by HyDE and reranking layers)
    let llm_service = match config.llm.as_ref() {
        Some(llm_cfg) => {
            let domain_llm_config = LlmServiceConfig::from_shared(llm_cfg);
            let model_name = match &llm_cfg.provider {
                lethe_shared::LlmProvider::Ollama { model, .. } => model.as_str(),
            };
            match LlmServiceFactory::create(&domain_llm_config).await {
                Ok(service) => {
                    tracing::info!(model = %model_name, "LLM service initialised for HyDE");
                    Some(service)
                }
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        "LLM service initialisation failed; HyDE will run in fallback mode"
                    );
                    None
                }
            }
        }
        None => {
            tracing::info!("LLM service disabled in configuration");
            None
        }
    };

    // Create query pipeline
    let mut pipeline_config = PipelineConfig::default();
    if let Some(features) = &config.features {
        pipeline_config.enable_hyde = features.enable_hyde;
        pipeline_config.enable_query_understanding = features.enable_query_understanding;
        pipeline_config.enable_ml_prediction = features.enable_ml_prediction;
        pipeline_config.rerank_enabled = features.enable_state_tracking;
    }
    pipeline_config.timeout_seconds = (config.timeouts.hyde_ms.value() / 1000).max(1);

    let reranking_service: Option<Arc<dyn RerankingService>> = if pipeline_config.rerank_enabled {
        let service: Arc<dyn RerankingService> =
            Arc::new(EmbeddingRerankingService::new(embedding_service.clone()));
        Some(service)
    } else {
        None
    };

    let doc_repo: Arc<dyn lethe_domain::retrieval::DocumentRepository> = corpus.clone();
    let query_pipeline = Arc::new(PipelineFactory::create_pipeline(
        pipeline_config,
        doc_repo,
        embedding_service.clone(),
        llm_service.clone(),
        reranking_service.clone(),
    ));

    let app_state = AppState::new(
        config.clone(),
        corpus,
        embedding_service,
        llm_service,
        reranking_service,
        query_pipeline,
    )
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
