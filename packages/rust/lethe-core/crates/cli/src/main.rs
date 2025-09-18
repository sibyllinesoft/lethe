use clap::{Parser, Subcommand};
use lethe_shared::Result;
use std::path::PathBuf;
use tokio;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod commands;
mod config;
mod utils;

use commands::*;

#[derive(Parser)]
#[command(name = "lethe")]
#[command(about = "Lethe RAG System CLI")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(author = "Lethe Contributors")]
struct Cli {
    /// Configuration file path
    #[arg(long, short, global = true)]
    config: Option<PathBuf>,

    /// Database URL
    #[arg(long, global = true, env = "DATABASE_URL")]
    database_url: Option<String>,

    /// Verbose logging
    #[arg(long, short, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Quiet mode (suppress output)
    #[arg(long, short, global = true)]
    quiet: bool,

    /// Output format
    #[arg(long, global = true, default_value = "table")]
    format: OutputFormat,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new Lethe configuration
    Init(InitCommand),

    /// Ingest documents into the system
    Ingest(IngestCommand),

    /// Build search indices
    Index(IndexCommand),

    /// Query the RAG system
    Query(QueryCommand),

    /// Manage sessions
    Session(SessionCommand),

    /// Manage messages
    Message(MessageCommand),

    /// Manage chunks
    Chunk(ChunkCommand),

    /// Manage embeddings
    Embedding(EmbeddingCommand),

    /// Server management
    Serve(ServeCommand),

    /// System diagnostics
    Diagnose(DiagnoseCommand),

    /// Database operations
    Database(DatabaseCommand),

    /// Configuration management
    Config(ConfigCommand),

    /// Performance benchmarks
    Benchmark(BenchmarkCommand),
}

#[derive(Debug, Clone, clap::ValueEnum)]
enum OutputFormat {
    Table,
    Json,
    Yaml,
    Pretty,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = match (cli.quiet, cli.verbose) {
        (true, _) => "error",
        (_, 0) => "info",
        (_, 1) => "debug",
        (_, _) => "trace",
    };

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| format!("lethe_cli={}", log_level).into()),
        )
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .init();

    // Load configuration
    let config = config::load_config(cli.config.as_deref()).await?;

    // Create application context
    let app_context = utils::AppContext {
        config,
        database_url: cli.database_url,
        output_format: cli.format.into(),
        quiet: cli.quiet,
    };

    // Execute command
    match cli.command {
        Commands::Init(cmd) => cmd.execute(&app_context).await,
        Commands::Ingest(cmd) => cmd.execute(&app_context).await,
        Commands::Index(cmd) => cmd.execute(&app_context).await,
        Commands::Query(cmd) => cmd.execute(&app_context).await,
        Commands::Session(cmd) => cmd.execute(&app_context).await,
        Commands::Message(cmd) => cmd.execute(&app_context).await,
        Commands::Chunk(cmd) => cmd.execute(&app_context).await,
        Commands::Embedding(cmd) => cmd.execute(&app_context).await,
        Commands::Serve(cmd) => cmd.execute(&app_context).await,
        Commands::Diagnose(cmd) => cmd.execute(&app_context).await,
        Commands::Database(cmd) => cmd.execute(&app_context).await,
        Commands::Config(cmd) => cmd.execute(&app_context).await,
        Commands::Benchmark(cmd) => cmd.execute(&app_context).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_cli() {
        use clap::CommandFactory;
        Cli::command().debug_assert()
    }
}
