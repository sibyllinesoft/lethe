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

    /// Database URL to use in configuration
    #[arg(long)]
    database_url: Option<String>,

    /// Embedding service provider
    #[arg(long, value_enum, default_value = "fallback")]
    embedding_provider: EmbeddingProviderArg,

    /// Ollama base URL (if using Ollama provider)
    #[arg(long)]
    ollama_url: Option<String>,

    /// Ollama model name (if using Ollama provider)
    #[arg(long)]
    ollama_model: Option<String>,
}

#[derive(Debug, Clone, clap::ValueEnum)]
enum EmbeddingProviderArg {
    Ollama,
    Fallback,
}

#[async_trait]
impl Command for InitCommand {
    async fn execute(&self, context: &AppContext) -> Result<()> {
        use lethe_shared::{DatabaseConfig, EmbeddingConfig, EmbeddingProvider};
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

        let mut database = DatabaseConfig::default();
        database.url = self
            .database_url
            .clone()
            .or_else(|| context.database_url.clone())
            .unwrap_or_else(|| "postgresql://localhost/lethe".to_string());

        let mut embedding = EmbeddingConfig::default();
        embedding.provider = embedding_provider;

        let config = LetheConfig {
            database,
            embedding,
            ..Default::default()
        };

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
