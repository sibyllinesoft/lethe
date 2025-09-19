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
        use std::{net::SocketAddr, sync::Arc};
        use tokio::net::TcpListener;
        use tower::ServiceBuilder;
        use tower_http::trace::TraceLayer;

        if !context.quiet {
            println!("🚀 Starting Lethe API server...");
            println!("   🌐 Host: {}", self.host);
            println!("   🔌 Port: {}", self.port);
        }

        let config_arc = Arc::new(context.resolved_config.clone());
        let storage_root = context.storage_root.clone();
        if !context.quiet {
            println!("   📦 Using storage root: {}", storage_root.display());
        }

        let app_state = AppState::initialise(config_arc, &storage_root)
            .await
            .map_err(|err| {
                lethe_shared::LetheError::internal(format!(
                    "Failed to initialise API state: {}",
                    err
                ))
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
