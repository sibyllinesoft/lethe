use super::Command;
use crate::utils::AppContext;
use async_trait::async_trait;
use clap::Args;
use lethe_shared::Result;

#[derive(Debug, Args)]
pub struct IndexCommand {
    /// Rebuild all indices
    #[arg(long)]
    rebuild: bool,

    /// Index specific session
    #[arg(long)]
    session_id: Option<String>,

    /// Batch size for indexing
    #[arg(long, default_value = "100")]
    batch_size: usize,
}

#[async_trait]
impl Command for IndexCommand {
    async fn execute(&self, _context: &AppContext) -> Result<()> {
        println!("⚙️  Offline indexing pipeline is not yet available in the Rust workspace.");
        println!(
            "   • Use 'lethe ingest' to load fresh content; embeddings are generated on the fly."
        );
        println!("   • For bulk re-indexing, run the ingestion or determinism services directly.");

        Ok(())
    }
}
