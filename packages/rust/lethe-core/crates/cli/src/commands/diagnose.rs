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

    /// Verify storage availability and basic health
    #[arg(long)]
    test_storage: bool,

    /// Test embedding service
    #[arg(long)]
    test_embeddings: bool,

    /// Test LLM connectivity (if configured)
    #[arg(long)]
    test_llm: bool,

    /// Test all components
    #[arg(long)]
    test_all: bool,
}

#[async_trait]
impl Command for DiagnoseCommand {
    async fn execute(&self, context: &AppContext) -> Result<()> {
        use lethe_domain::{EmbeddingServiceFactory, LlmServiceConfig, LlmServiceFactory};
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
                    "   Embedding provider: {:?}",
                    context.config.embedding.provider
                );
                println!("   Storage root: {}", context.storage_root.display());
            }
        }
        println!();

        // Storage health check
        if self.test_storage || self.test_all {
            print!("💾 Storage root: ");
            let storage_path = &context.storage_root;
            if !storage_path.exists() {
                if let Err(e) = std::fs::create_dir_all(storage_path) {
                    println!("❌ Failed to create {} ({})", storage_path.display(), e);
                    all_good = false;
                } else {
                    println!("⚠️  Created missing directory {}", storage_path.display());
                }
            } else if !storage_path.is_dir() {
                println!("❌ {} is not a directory", storage_path.display());
                all_good = false;
            } else if std::fs::read_dir(storage_path).is_err() {
                println!("❌ Unable to read directory {}", storage_path.display());
                all_good = false;
            } else {
                println!("✅ Accessible");
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

        // LLM service test
        if self.test_llm || self.test_all {
            print!("🗣️  LLM service: ");
            match context.config.llm.as_ref() {
                Some(llm_cfg) => {
                    let domain_config = LlmServiceConfig::from_shared(llm_cfg);
                    match LlmServiceFactory::create(&domain_config).await {
                        Ok(service) => {
                            let prompt = "Summarise the health of the Lethe system";
                            match service
                                .generate_text(prompt, &lethe_domain::HydeConfig::default())
                                .await
                            {
                                Ok(outputs) => {
                                    let preview = outputs
                                        .get(0)
                                        .map(|s| {
                                            let mut snippet =
                                                s.chars().take(48).collect::<String>();
                                            if s.len() > 48 {
                                                snippet.push_str("…");
                                            }
                                            snippet
                                        })
                                        .unwrap_or_else(|| "(empty response)".to_string());
                                    println!("✅ Responded ({})", preview);
                                }
                                Err(e) => {
                                    println!("❌ Generation failed - {}", e);
                                    all_good = false;
                                }
                            }
                        }
                        Err(e) => {
                            println!("❌ Initialisation failed - {}", e);
                            all_good = false;
                        }
                    }
                }
                None => {
                    println!("⚠️  Disabled in configuration");
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
