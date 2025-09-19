use crate::utils::AppContext;
use async_trait::async_trait;
use lethe_domain::{
    EmbeddingConfig as DomainEmbeddingConfig, EmbeddingProvider as DomainEmbeddingProvider,
};
use lethe_shared::{
    config::EmbeddingProvider as SharedEmbeddingProvider, EmbeddingConfig as SharedEmbeddingConfig,
    Result,
};

// Re-export all command types
pub use benchmark::BenchmarkCommand;
pub use config::ConfigCommand;
pub use diagnose::DiagnoseCommand;
pub use index::IndexCommand;
pub use init::InitCommand;
pub use query::QueryCommand;
pub use serve::ServeCommand;

// Command modules
pub mod benchmark;
pub mod config;
pub mod diagnose;
pub mod index;
pub mod init;
pub mod query;
pub mod serve;

/// Common trait for all CLI commands
#[async_trait]
pub trait Command {
    async fn execute(&self, context: &AppContext) -> Result<()>;
}

/// Helper to bridge shared embedding configuration into the domain representation.
pub fn to_domain_embedding_config(shared: &SharedEmbeddingConfig) -> DomainEmbeddingConfig {
    let mut config = DomainEmbeddingConfig::default();

    config.provider = match &shared.provider {
        SharedEmbeddingProvider::Ollama { base_url, model } => DomainEmbeddingProvider::Ollama {
            base_url: base_url.clone(),
            model: model.clone(),
        },
        SharedEmbeddingProvider::Fallback => DomainEmbeddingProvider::Fallback,
    };

    let default_model = config.model_name.clone();
    config.model_name = match &config.provider {
        DomainEmbeddingProvider::Ollama { model, .. } => model.clone(),
        DomainEmbeddingProvider::Fallback => default_model,
        DomainEmbeddingProvider::Custom { .. } => default_model,
    };

    config.dimension = shared.dimension;
    config.timeout_ms = shared.timeout_ms;
    config.cache = shared.cache.clone();
    config
}
