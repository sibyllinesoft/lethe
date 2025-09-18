use super::Command;
use crate::utils::AppContext;
use async_trait::async_trait;
use clap::{Args, Subcommand};
use lethe_shared::Result;

#[derive(Debug, Args)]
pub struct ChunkCommand {
    #[command(subcommand)]
    action: ChunkAction,
}

#[derive(Debug, Subcommand)]
enum ChunkAction {
    /// List chunks for a session or message
    List {
        /// Session ID to filter by
        #[arg(long)]
        session_id: Option<String>,
        /// Message ID to filter by
        #[arg(long)]
        message_id: Option<String>,
        /// Maximum number of chunks to display
        #[arg(long, default_value = "20")]
        limit: usize,
    },
    /// Show chunk details
    Show {
        /// Chunk ID to inspect
        chunk_id: String,
    },
    /// Delete a chunk (and associated embedding)
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
        use lethe_infrastructure::{
            ChunkRepository, DatabaseManager, EmbeddingRepository, PgChunkRepository,
            PgEmbeddingRepository,
        };
        use std::sync::Arc;
        use uuid::Uuid;

        let db_url = context.database_url.as_ref().ok_or_else(|| {
            lethe_shared::LetheError::config("Database URL is required for chunk management")
        })?;

        let db_manager = Arc::new(DatabaseManager::new(db_url).await?);
        let chunk_repo = Arc::new(PgChunkRepository::new(db_manager.pool().clone()));
        let embedding_repo = Arc::new(PgEmbeddingRepository::new(db_manager.pool().clone()));

        match &self.action {
            ChunkAction::List {
                session_id,
                message_id,
                limit,
            } => {
                let chunks = if let Some(message_id) = message_id {
                    let message_uuid = Uuid::parse_str(message_id).map_err(|e| {
                        lethe_shared::LetheError::validation(
                            "message_id",
                            format!("Invalid UUID: {}", e),
                        )
                    })?;
                    chunk_repo.get_chunks_by_message(&message_uuid).await?
                } else if let Some(session_id) = session_id {
                    chunk_repo.get_chunks_by_session(session_id).await?
                } else {
                    return Err(lethe_shared::LetheError::validation(
                        "arguments",
                        "Provide either --session-id or --message-id",
                    ));
                };

                let mut chunks = chunks;
                chunks.truncate(*limit);

                if chunks.is_empty() {
                    println!("No chunks found matching the provided filters.");
                    return Ok(());
                }

                match &context.output_format {
                    crate::utils::OutputFormat::Json => {
                        let json = serde_json::to_string_pretty(&chunks)?;
                        println!("{}", json);
                    }
                    _ => {
                        println!("📄 Chunks (showing up to {}):", limit);
                        for chunk in chunks {
                            println!(
                                "  • {} [{}–{}] ({} tokens) — {}",
                                chunk.id,
                                chunk.offset_start,
                                chunk.offset_end,
                                chunk.tokens,
                                truncate(&chunk.text, 120)
                            );
                        }
                    }
                }
            }
            ChunkAction::Show { chunk_id } => {
                if let Some(chunk) = chunk_repo.get_chunk(chunk_id).await? {
                    match &context.output_format {
                        crate::utils::OutputFormat::Json => {
                            let json = serde_json::to_string_pretty(&chunk)?;
                            println!("{}", json);
                        }
                        _ => {
                            println!("📄 Chunk {}", chunk.id);
                            println!("   • Session: {}", chunk.session_id);
                            println!("   • Message: {}", chunk.message_id);
                            println!(
                                "   • Offsets: {} – {}",
                                chunk.offset_start, chunk.offset_end
                            );
                            println!("   • Kind: {}", chunk.kind);
                            println!("   • Tokens: {}", chunk.tokens);
                            println!("   • Text:\n{}", chunk.text);
                        }
                    }
                } else {
                    println!("Chunk '{}' not found.", chunk_id);
                }
            }
            ChunkAction::Delete { chunk_id, force } => {
                if !force && !context.quiet {
                    use dialoguer::Confirm;
                    let proceed = Confirm::new()
                        .with_prompt(format!("Delete chunk '{}' and its embedding?", chunk_id))
                        .interact()
                        .map_err(|e| {
                            lethe_shared::LetheError::internal(format!("Prompt failed: {}", e))
                        })?;
                    if !proceed {
                        println!("Cancelled");
                        return Ok(());
                    }
                }

                let embedding_removed = embedding_repo.delete_embedding(chunk_id).await?;
                let chunk_removed = chunk_repo.delete_chunk(chunk_id).await?;

                if chunk_removed {
                    println!("✅ Deleted chunk '{}'.", chunk_id);
                    if embedding_removed {
                        println!("   • Associated embedding removed");
                    }
                } else {
                    println!("Chunk '{}' did not exist.", chunk_id);
                }
            }
        }

        Ok(())
    }
}

fn truncate(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        text.to_string()
    } else {
        format!("{}...", &text[..limit.saturating_sub(3)])
    }
}
