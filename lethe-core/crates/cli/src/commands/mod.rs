use async_trait::async_trait;
use clap::{Args, Subcommand};
use lethe_shared::Result;
use crate::utils::AppContext;

// Re-export all command types
pub use init::InitCommand;
pub use ingest::IngestCommand;
pub use index::IndexCommand;
pub use query::QueryCommand;
pub use session::SessionCommand;
pub use message::MessageCommand;
pub use chunk::ChunkCommand;
pub use embedding::EmbeddingCommand;
pub use serve::ServeCommand;
pub use diagnose::DiagnoseCommand;
pub use database::DatabaseCommand;
pub use config::ConfigCommand;
pub use benchmark::BenchmarkCommand;

// Command modules
pub mod init;
pub mod ingest;
pub mod index;
pub mod query;
pub mod session;
pub mod message;
pub mod chunk;
pub mod embedding;
pub mod serve;
pub mod diagnose;
pub mod database;
pub mod config;
pub mod benchmark;

/// Common trait for all CLI commands
#[async_trait]
pub trait Command {
    async fn execute(&self, context: &AppContext) -> Result<()>;
}