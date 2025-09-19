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
    #[arg(long, short = 'C', global = true)]
    config: Option<PathBuf>,

    /// Storage root override
    #[arg(long, global = true)]
    storage_root: Option<PathBuf>,

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

    /// Build search indices
    Index(IndexCommand),

    /// Query the RAG system
    Query(QueryCommand),

    /// Server management
    Serve(ServeCommand),

    /// System diagnostics
    Diagnose(DiagnoseCommand),

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
    let config::LoadedConfig { config, path } = config::load_config(cli.config.as_deref()).await?;

    // Create application context
    let storage_root = cli
        .storage_root
        .unwrap_or_else(|| PathBuf::from(config.storage.index_root.clone()));

    let app_context = utils::AppContext {
        config,
        config_path: path,
        storage_root,
        output_format: cli.format.into(),
        quiet: cli.quiet,
    };

    // Execute command
    match cli.command {
        Commands::Init(cmd) => cmd.execute(&app_context).await,
        Commands::Index(cmd) => cmd.execute(&app_context).await,
        Commands::Query(cmd) => cmd.execute(&app_context).await,
        Commands::Serve(cmd) => cmd.execute(&app_context).await,
        Commands::Diagnose(cmd) => cmd.execute(&app_context).await,
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
