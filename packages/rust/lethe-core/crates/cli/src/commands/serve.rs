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
        use lethe_domain::{EmbeddingServiceFactory, PipelineConfig, PipelineFactory};
        use lethe_infrastructure::{
            DatabaseManager, PgChunkRepository, PgEmbeddingRepository, PgMessageRepository,
            PgSessionRepository,
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

        // Initialize database
        let database_url = context.database_url.as_ref().ok_or_else(|| {
            lethe_shared::LetheError::config("Database URL is required for server")
        })?;

        if !context.quiet {
            println!("   🗄️  Connecting to database...");
        }
        let db_manager = Arc::new(DatabaseManager::new(database_url).await?);

        // Create repositories
        let message_repository = Arc::new(PgMessageRepository::new(db_manager.pool().clone()));
        let chunk_repository = Arc::new(PgChunkRepository::new(db_manager.pool().clone()));
        let embedding_repository = Arc::new(PgEmbeddingRepository::new(db_manager.pool().clone()));
        let session_repository = Arc::new(PgSessionRepository::new(db_manager.pool().clone()));

        // Create embedding service
        if !context.quiet {
            println!("   🧠 Initializing embedding service...");
        }
        let embedding_config = super::to_domain_embedding_config(&context.config.embedding);
        let embedding_service = EmbeddingServiceFactory::create(&embedding_config).await?;

        // Create query pipeline
        let features = context
            .config
            .features
            .as_ref()
            .cloned()
            .unwrap_or_default();

        let timeout_seconds = (context.config.timeouts.hyde_ms.value() / 1000).max(1);

        let pipeline_config = PipelineConfig {
            enable_hyde: features.enable_hyde,
            enable_query_understanding: features.enable_query_understanding,
            enable_ml_prediction: features.enable_ml_prediction,
            max_candidates: 50,
            rerank_enabled: features.enable_plan_selection,
            rerank_top_k: 20,
            timeout_seconds,
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
            Arc::new(context.config.clone()),
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
        let addr = SocketAddr::from(([0, 0, 0, 0], self.port));

        if !context.quiet {
            println!("🎯 Server ready!");
            println!("   📡 API URL: http://{}:{}", self.host, self.port);
            println!(
                "   🏥 Health endpoint: http://{}:{}/api/v1/health",
                self.host, self.port
            );
            println!("   📖 Press Ctrl+C to stop");
        }

        let listener = TcpListener::bind(addr).await.map_err(|e| {
            lethe_shared::LetheError::internal(format!(
                "Failed to bind to {}:{} - {}",
                self.host, self.port, e
            ))
        })?;

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
