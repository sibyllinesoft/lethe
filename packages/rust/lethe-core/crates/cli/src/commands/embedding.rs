use super::Command;
use crate::utils::AppContext;
use async_trait::async_trait;
use clap::{Args, Subcommand};
use lethe_shared::Result;

#[derive(Debug, Args)]
pub struct EmbeddingCommand {
    #[command(subcommand)]
    action: EmbeddingAction,
}

#[derive(Debug, Subcommand)]
enum EmbeddingAction {
    /// List embeddings for a session
    List {
        /// Session ID to filter by (required)
        #[arg(long)]
        session_id: Option<String>,
        /// Maximum number of embeddings to show
        #[arg(long, default_value = "10")]
        limit: usize,
    },
    /// Show embedding details for a chunk
    Show {
        /// Chunk ID to show embedding for
        chunk_id: String,
    },
    /// Delete an embedding by chunk ID
    Delete {
        /// Chunk ID to delete embedding for
        chunk_id: String,
        /// Force deletion without confirmation
        #[arg(long)]
        force: bool,
    },
    /// Search embeddings by similarity
    Search {
        /// Text to embed and search for
        query: String,
        /// Number of results to return
        #[arg(long, default_value = "5")]
        limit: usize,
    },
}

#[async_trait]
impl Command for EmbeddingCommand {
    async fn execute(&self, context: &AppContext) -> Result<()> {
        use lethe_domain::EmbeddingServiceFactory;
        use lethe_infrastructure::{DatabaseManager, EmbeddingRepository, PgEmbeddingRepository};
        use std::sync::Arc;

        let db_url = context.database_url.as_ref().ok_or_else(|| {
            lethe_shared::LetheError::config("Database URL is required for embedding management")
        })?;

        let db_manager = Arc::new(DatabaseManager::new(db_url).await?);
        let embedding_repo = Arc::new(PgEmbeddingRepository::new(db_manager.pool().clone()));

        match &self.action {
            EmbeddingAction::List { session_id, limit } => {
                let session_id = session_id.as_deref().ok_or_else(|| {
                    lethe_shared::LetheError::validation(
                        "session_id",
                        "Parameter is required when listing embeddings",
                    )
                })?;

                let mut embeddings = embedding_repo.get_embeddings_by_session(session_id).await?;
                embeddings.truncate(*limit);

                if embeddings.is_empty() {
                    println!("No embeddings found for session '{}'.", session_id);
                    return Ok(());
                }

                match &context.output_format {
                    crate::utils::OutputFormat::Json => {
                        let json = serde_json::to_string_pretty(&embeddings)?;
                        println!("{}", json);
                    }
                    _ => {
                        println!("🧠 Embeddings for session '{}':", session_id);
                        for (chunk_id, vector) in embeddings {
                            println!("  • Chunk {} → dimension {}", chunk_id, vector.dimension);
                        }
                    }
                }
            }
            EmbeddingAction::Show { chunk_id } => {
                match embedding_repo.get_embedding(chunk_id).await? {
                    Some(vector) => match &context.output_format {
                        crate::utils::OutputFormat::Json => {
                            let json = serde_json::to_string_pretty(&vector)?;
                            println!("{}", json);
                        }
                        _ => {
                            println!("🧠 Embedding for chunk '{}':", chunk_id);
                            println!("   • Dimension: {}", vector.dimension);
                            println!(
                                "   • Preview: {:?}",
                                &vector.data[..vector.data.len().min(8)]
                            );
                        }
                    },
                    None => println!("No embedding found for chunk '{}'.", chunk_id),
                }
            }
            EmbeddingAction::Delete { chunk_id, force } => {
                if !force && !context.quiet {
                    use dialoguer::Confirm;
                    let proceed = Confirm::new()
                        .with_prompt(format!("Delete embedding for chunk '{}' ?", chunk_id))
                        .interact()
                        .map_err(|e| {
                            lethe_shared::LetheError::internal(format!("Prompt failed: {}", e))
                        })?;
                    if !proceed {
                        println!("Cancelled");
                        return Ok(());
                    }
                }

                let removed = embedding_repo.delete_embedding(chunk_id).await?;
                if removed {
                    println!("✅ Deleted embedding for chunk '{}'.", chunk_id);
                } else {
                    println!("Embedding for chunk '{}' did not exist.", chunk_id);
                }
            }
            EmbeddingAction::Search { query, limit } => {
                let embedding_config = super::to_domain_embedding_config(&context.config.embedding);
                let embedding_service = EmbeddingServiceFactory::create(&embedding_config).await?;
                let query_vector = embedding_service.embed_single(query).await?;

                let results = embedding_repo
                    .search_similar_embeddings(&query_vector, *limit as i32)
                    .await?;

                if results.is_empty() {
                    println!("No similar embeddings found for '{}'.", query);
                    return Ok(());
                }

                match &context.output_format {
                    crate::utils::OutputFormat::Json => {
                        let json = serde_json::to_string_pretty(&results)?;
                        println!("{}", json);
                    }
                    _ => {
                        println!("🔍 Similar embeddings for '{}':", query);
                        for (idx, (chunk_id, score)) in results.iter().enumerate() {
                            println!(
                                "  {}. Chunk {} → similarity {:.4}",
                                idx + 1,
                                chunk_id,
                                score
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
