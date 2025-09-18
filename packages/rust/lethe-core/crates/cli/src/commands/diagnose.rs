use super::Command;
use crate::utils::AppContext;
use async_trait::async_trait;
use clap::Args;
use lethe_shared::Result;

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
        use lethe_domain::EmbeddingServiceFactory;
        use lethe_infrastructure::DatabaseManager;
        use std::process::Command;

        if !context.quiet {
            println!("🔍 Running Lethe system diagnostics...\n");
        }

        let mut all_good = true;

        // System information
        println!("📋 System Information:");
        println!("   OS: {}", std::env::consts::OS);
        println!("   Arch: {}", std::env::consts::ARCH);
        let rust_version = Command::new("rustc")
            .arg("--version")
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        println!("   Rust version: {}", rust_version);
        println!("   Lethe version: {}", env!("CARGO_PKG_VERSION"));
        println!();

        // Configuration check
        println!("⚙️  Configuration:");
        match &context.output_format {
            crate::utils::OutputFormat::Json => {
                println!("{}", serde_json::to_string_pretty(&context.config)?);
            }
            _ => {
                println!(
                    "   Database URL: {}",
                    context.database_url.as_deref().unwrap_or("Not configured")
                );
                println!(
                    "   Embedding provider: {:?}",
                    context.config.embedding.provider
                );
            }
        }
        println!();

        // Database connectivity test
        if self.test_db || self.test_all {
            print!("🗄️  Database connectivity: ");
            match context.database_url.as_ref() {
                Some(db_url) => match DatabaseManager::new(db_url).await {
                    Ok(_) => println!("✅ Connected"),
                    Err(e) => {
                        println!("❌ Failed - {}", e);
                        all_good = false;
                    }
                },
                None => {
                    println!("❌ No database URL configured");
                    all_good = false;
                }
            }
        }

        // Embedding service test
        if self.test_embeddings || self.test_all {
            print!("🧠 Embedding service: ");
            let embedding_config = super::to_domain_embedding_config(&context.config.embedding);
            match EmbeddingServiceFactory::create(&embedding_config).await {
                Ok(service) => match service.embed_single("test").await {
                    Ok(vector) => {
                        println!("✅ Working ({}D vector)", vector.dimension);
                    }
                    Err(e) => {
                        println!("❌ Test failed - {}", e);
                        all_good = false;
                    }
                },
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
