use async_trait::async_trait;
use clap::Args;
use lethe_shared::Result;
use crate::utils::AppContext;
use super::Command;

#[derive(Debug, Args)]
pub struct DiagnoseCommand {
    /// Include detailed system information
    #[arg(long)]
    detailed: bool,

    /// Test database connectivity
    #[arg(long)]
    test_db: bool,

    /// Test embedding service
    #[arg(long)]
    test_embeddings: bool,

    /// Test all components
    #[arg(long)]
    test_all: bool,
}

#[async_trait]
impl Command for DiagnoseCommand {
    async fn execute(&self, context: &AppContext) -> Result<()> {
        use lethe_infrastructure::DatabaseManager;
        use lethe_domain::EmbeddingServiceFactory;
        use std::sync::Arc;

        if !context.quiet {
            println!("🔍 Running Lethe system diagnostics...\n");
        }

        let mut all_good = true;

        // System information
        println!("📋 System Information:");
        println!("   OS: {}", std::env::consts::OS);
        println!("   Arch: {}", std::env::consts::ARCH);
        println!("   Rust version: {}", env!("RUSTC_VERSION"));
        println!("   Lethe version: {}", env!("CARGO_PKG_VERSION"));
        println!();

        // Configuration check
        println!("⚙️  Configuration:");
        match &context.output_format {
            crate::utils::OutputFormat::Json => {
                println!("{}", serde_json::to_string_pretty(&context.config)?);
            }
            _ => {
                println!("   Database URL: {}", 
                    context.database_url.as_deref().unwrap_or("Not configured"));
                println!("   Embedding provider: {:?}", context.config.embedding.provider);
            }
        }
        println!();

        // Database connectivity test
        if self.test_db || self.test_all {
            print!("🗄️  Database connectivity: ");
            match context.database_url.as_ref() {
                Some(db_url) => {
                    match DatabaseManager::new(db_url).await {
                        Ok(_) => println!("✅ Connected"),
                        Err(e) => {
                            println!("❌ Failed - {}", e);
                            all_good = false;
                        }
                    }
                }
                None => {
                    println!("❌ No database URL configured");
                    all_good = false;
                }
            }
        }

        // Embedding service test
        if self.test_embeddings || self.test_all {
            print!("🧠 Embedding service: ");
            match EmbeddingServiceFactory::create_service(&context.config.embedding).await {
                Ok(service) => {
                    match service.embed("test").await {
                        Ok(vector) => {
                            println!("✅ Working ({}D vector)", vector.len());
                        }
                        Err(e) => {
                            println!("❌ Test failed - {}", e);
                            all_good = false;
                        }
                    }
                }
                Err(e) => {
                    println!("❌ Creation failed - {}", e);
                    all_good = false;
                }
            }
        }

        println!();
        if all_good {
            println!("✅ All systems operational");
        } else {
            println!("❌ Some issues detected");
            std::process::exit(1);
        }

        Ok(())
    }
}