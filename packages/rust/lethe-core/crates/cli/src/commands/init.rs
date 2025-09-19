use super::Command;
use crate::utils::AppContext;
use async_trait::async_trait;
use clap::Args;
use lethe_shared::{LetheConfig, Result};
use std::path::PathBuf;

#[derive(Debug, Args)]
pub struct InitCommand {
    /// Configuration file path to create
    #[arg(long, short = 'o', default_value = "lethe.json")]
    output: PathBuf,

    /// Force overwrite existing configuration
    #[arg(long)]
    force: bool,

    /// Embedding service provider
    #[arg(long, value_enum, default_value = "fallback")]
    embedding_provider: EmbeddingProviderArg,

    /// Ollama base URL (if using Ollama provider)
    #[arg(long)]
    ollama_url: Option<String>,

    /// Ollama model name (if using Ollama provider)
    #[arg(long)]
    ollama_model: Option<String>,

    /// Storage root for parquet/tantivy assets
    #[arg(long)]
    storage_root: Option<PathBuf>,
}

#[derive(Debug, Clone, clap::ValueEnum)]
enum EmbeddingProviderArg {
    Ollama,
    Fallback,
}

#[async_trait]
impl Command for InitCommand {
    async fn execute(&self, context: &AppContext) -> Result<()> {
        use lethe_shared::{EmbeddingConfig, EmbeddingProvider};
        use std::io::Write;

        // Check if file exists and not forcing
        if self.output.exists() && !self.force {
            return Err(lethe_shared::LetheError::config(
                "Configuration file already exists. Use --force to overwrite.",
            ));
        }

        // Create configuration
        let embedding_provider = match self.embedding_provider {
            EmbeddingProviderArg::Ollama => {
                let base_url = self
                    .ollama_url
                    .clone()
                    .unwrap_or_else(|| "http://localhost:11434".to_string());
                let model = self
                    .ollama_model
                    .clone()
                    .unwrap_or_else(|| "all-minilm".to_string());

                EmbeddingProvider::Ollama { base_url, model }
            }
            EmbeddingProviderArg::Fallback => EmbeddingProvider::Fallback,
        };

        let mut embedding = EmbeddingConfig::default();
        embedding.provider = embedding_provider;

        let mut config = LetheConfig::default();
        config.embedding = embedding;
        if let Some(root) = self.storage_root.as_ref() {
            config.storage.index_root = root.display().to_string();
        } else {
            config.storage.index_root = context.storage_root.display().to_string();
        }

        // Serialize and write configuration
        let config_json = serde_json::to_string_pretty(&config)?;

        let mut file = std::fs::File::create(&self.output)?;

        file.write_all(config_json.as_bytes())?;

        if !context.quiet {
            println!(
                "✅ Configuration file created at: {}",
                self.output.display()
            );
            println!("📝 Edit the configuration to customize settings for your environment.");

            if matches!(config.embedding.provider, EmbeddingProvider::Ollama { .. }) {
                println!("🔧 Make sure Ollama is running at the specified URL.");
            }
        }

        Ok(())
    }
}
