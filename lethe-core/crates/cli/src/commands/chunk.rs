use async_trait::async_trait;
use clap::{Args, Subcommand};
use lethe_shared::Result;
use crate::utils::AppContext;
use super::Command;

#[derive(Debug, Args)]
pub struct ChunkCommand {
    #[command(subcommand)]
    action: ChunkAction,
}

#[derive(Debug, Subcommand)]
enum ChunkAction {
    /// List chunks
    List {
        /// Session ID to filter by
        #[arg(long)]
        session_id: Option<String>,
        /// Message ID to filter by
        #[arg(long)]
        message_id: Option<String>,
        /// Limit number of results
        #[arg(long, short, default_value = "10")]
        limit: usize,
    },
    /// Show chunk details
    Show {
        /// Chunk ID to show
        chunk_id: String,
    },
    /// Delete a chunk
    Delete {
        /// Chunk ID to delete
        chunk_id: String,
        /// Force deletion without confirmation
        #[arg(long)]
        force: bool,
    },
}

#[async_trait]
impl Command for ChunkCommand {
    async fn execute(&self, context: &AppContext) -> Result<()> {
        use lethe_infrastructure::{DatabaseManager, PgChunkRepository};
        use std::sync::Arc;

        let db_url = context.database_url.as_ref()
            .ok_or("Database URL is required for chunk management")?;
        let db_manager = Arc::new(DatabaseManager::new(db_url).await?);
        let chunk_repo = Arc::new(PgChunkRepository::new(db_manager.pool().clone()));

        match &self.action {
            ChunkAction::List { session_id, message_id, limit } => {
                let chunks = if let Some(session_id) = session_id {
                    chunk_repo.find_by_session(session_id).await?
                } else if let Some(message_id) = message_id {
                    let message_uuid = uuid::Uuid::parse_str(message_id)?;
                    chunk_repo.find_by_message(&message_uuid).await?
                } else {
                    chunk_repo.find_recent(*limit).await?
                };

                match &context.output_format {
                    crate::utils::OutputFormat::Json => {
                        println!("{}", serde_json::to_string_pretty(&chunks)?);
                    }
                    _ => {
                        if chunks.is_empty() {
                            println!("No chunks found");
                        } else {
                            println!("📄 Chunks ({})", chunks.len());
                            for chunk in chunks {
                                println!("  🆔 {} [{}]: {}", 
                                    chunk.id, 
                                    chunk.idx,
                                    if chunk.text.len() > 60 { 
                                        format!("{}...", &chunk.text[..57]) 
                                    } else { 
                                        chunk.text.clone() 
                                    }
                                );
                            }
                        }
                    }
                }
            }
            ChunkAction::Show { chunk_id } => {
                let chunk_uuid = uuid::Uuid::parse_str(chunk_id)?;
                let chunk = chunk_repo.find_by_id(&chunk_uuid).await?
                    .ok_or_else(|| format!("Chunk not found: {}", chunk_id))?;

                match &context.output_format {
                    crate::utils::OutputFormat::Json => {
                        println!("{}", serde_json::to_string_pretty(&chunk)?);
                    }
                    _ => {
                        println!("📄 Chunk: {}", chunk.id);
                        println!("   Message: {}", chunk.message_id);
                        println!("   Session: {}", chunk.session_id);
                        println!("   Index: {}", chunk.idx);
                        println!("   Time: {}", chunk.ts);
                        println!("   Text:\n{}", chunk.text);
                        if let Some(meta) = &chunk.meta {
                            println!("   Meta: {}", serde_json::to_string_pretty(meta)?);
                        }
                    }
                }
            }
            ChunkAction::Delete { chunk_id, force } => {
                if !force && !context.quiet {
                    use dialoguer::Confirm;
                    if !Confirm::new()
                        .with_prompt(format!("Delete chunk '{}'?", chunk_id))
                        .interact()? 
                    {
                        println!("Cancelled");
                        return Ok(());
                    }
                }

                let chunk_uuid = uuid::Uuid::parse_str(chunk_id)?;
                chunk_repo.delete(&chunk_uuid).await?;
                
                if !context.quiet {
                    println!("✅ Deleted chunk: {}", chunk_id);
                }
            }
        }

        Ok(())
    }
}