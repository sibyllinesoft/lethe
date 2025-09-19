use super::Command;
use crate::utils::AppContext;
use async_trait::async_trait;
use clap::Args;
use lethe_shared::Result;

#[derive(Debug, Args)]
pub struct ServeCommand {
    /// Server host
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Server port
    #[arg(long, short, default_value = "3000")]
    port: u16,

    /// Number of worker threads
    #[arg(long)]
    workers: Option<usize>,

    /// Enable development mode (auto-reload)
    #[arg(long)]
    dev: bool,

    /// Log level for the server
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[async_trait]
impl Command for ServeCommand {
    async fn execute(&self, context: &AppContext) -> Result<()> {
        use lethe_api::{create_app, AppState};
        use lethe_domain::{
            corpus::ParquetCorpus, EmbeddingRerankingService, EmbeddingServiceFactory,
            LlmServiceConfig, LlmServiceFactory, PipelineConfig, PipelineFactory, RerankingService,
        };
        use std::{net::SocketAddr, sync::Arc};
        use tokio::net::TcpListener;
        use tower::ServiceBuilder;
        use tower_http::trace::TraceLayer;

        if !context.quiet {
            println!("🚀 Starting Lethe API server...");
            println!("   🌐 Host: {}", self.host);
            println!("   🔌 Port: {}", self.port);
        }

        // Prepare storage-backed corpus
        let storage_root = context.storage_root.clone();
        let corpus = Arc::new(ParquetCorpus::new(&storage_root));
        corpus.health_check().await?;

        // Create embedding service
        if !context.quiet {
            println!("   🧠 Initializing embedding service...");
        }
        let embedding_config = super::to_domain_embedding_config(&context.config.embedding);
        let embedding_service = EmbeddingServiceFactory::create(&embedding_config).await?;

        // Optional LLM service (for HyDE / reranking)
        let llm_service = match context.config.llm.as_ref() {
            Some(cfg) => {
                if !context.quiet {
                    println!("   🧾 Initializing LLM service...");
                }
                let domain_cfg = LlmServiceConfig::from_shared(cfg);
                match LlmServiceFactory::create(&domain_cfg).await {
                    Ok(service) => Some(service),
                    Err(err) => {
                        eprintln!("   ⚠️  LLM unavailable: {}", err);
                        None
                    }
                }
            }
            None => None,
        };

        // Configure pipeline based on feature toggles
        let features = context
            .config
            .features
            .as_ref()
            .cloned()
            .unwrap_or_default();

        let mut pipeline_config = PipelineConfig::default();
        pipeline_config.enable_hyde = features.enable_hyde;
        pipeline_config.enable_query_understanding = features.enable_query_understanding;
        pipeline_config.enable_ml_prediction = features.enable_ml_prediction;
        pipeline_config.rerank_enabled = features.enable_state_tracking;
        pipeline_config.timeout_seconds = (context.config.timeouts.hyde_ms.value() / 1000).max(1);

        let reranking_service: Option<Arc<dyn RerankingService>> = if pipeline_config.rerank_enabled
        {
            Some(Arc::new(EmbeddingRerankingService::new(embedding_service.clone())) as Arc<_>)
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

        // Create application state
        let app_state = AppState::new(
            Arc::new(context.config.clone()),
            corpus.clone(),
            embedding_service.clone(),
            llm_service.clone(),
            reranking_service.clone(),
            query_pipeline.clone(),
        )
        .map_err(|err| {
            lethe_shared::LetheError::internal(format!("Failed to initialise API state: {}", err))
        })?;

        // Perform health check
        if !context.quiet {
            println!("   🏥 Performing health check...");
        }
        match app_state.health_check().await {
            Ok(health) => {
                if !context.quiet {
                    println!("   ✅ Health check passed");
                    for component in &health.components {
                        println!("      • {} → {:?}", component.name, component.status);
                    }
                }
            }
            Err(e) => {
                return Err(lethe_shared::LetheError::internal(format!(
                    "Health check failed: {}",
                    e
                )));
            }
        }

        // Create application with middleware
        let app =
            create_app(app_state).layer(ServiceBuilder::new().layer(TraceLayer::new_for_http()));

        // Start server
        let addr: SocketAddr = format!("{}:{}", self.host, self.port)
            .parse()
            .map_err(|e| {
                lethe_shared::LetheError::config(format!(
                    "Invalid listen address {}:{} ({})",
                    self.host, self.port, e
                ))
            })?;

        let listener = TcpListener::bind(addr).await.map_err(|e| {
            lethe_shared::LetheError::internal(format!(
                "Failed to bind to {}:{} - {}",
                self.host, self.port, e
            ))
        })?;

        if !context.quiet {
            println!("🎯 Server ready!");
            println!("   📡 API URL: http://{}:{}", self.host, self.port);
            println!(
                "   🏥 Health endpoint: http://{}:{}/api/v1/health",
                self.host, self.port
            );
            println!("   📖 Press Ctrl+C to stop");
        }

        // Setup graceful shutdown
        let quiet = context.quiet;
        let shutdown_signal = async move {
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
                    if !quiet {
                        println!("\n🛑 Received Ctrl+C, shutting down gracefully...");
                    }
                },
                _ = terminate => {
                    if !quiet {
                        println!("\n🛑 Received terminate signal, shutting down gracefully...");
                    }
                },
            }
        };

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal)
            .await
            .map_err(|e| lethe_shared::LetheError::internal(format!("Server error: {}", e)))?;

        if !context.quiet {
            println!("✅ Server shutdown complete");
        }

        Ok(())
    }
}
