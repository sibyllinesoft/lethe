use async_trait::async_trait;
use clap::{Args, Subcommand};
use lethe_shared::Result;
use crate::utils::AppContext;
use super::Command;

#[derive(Debug, Args)]
pub struct EmbeddingCommand {
    #[command(subcommand)]
    action: EmbeddingAction,
}

#[derive(Debug, Subcommand)]
enum EmbeddingAction {
    /// List embeddings
    List {
        /// Session ID to filter by
        #[arg(long)]
        session_id: Option<String>,
        /// Limit number of results
        #[arg(long, short, default_value = "10")]
        limit: usize,
    },
    /// Show embedding details
    Show {
        /// Chunk ID to show embedding for
        chunk_id: String,
    },
    /// Delete an embedding
    Delete {
        /// Chunk ID to delete embedding for
        chunk_id: String,
        /// Force deletion without confirmation
        #[arg(long)]
        force: bool,
    },
    /// Search embeddings by similarity
    Search {
        /// Text to search for
        query: String,
        /// Number of results to return
        #[arg(long, short, default_value = "5")]
        limit: usize,
        /// Minimum similarity threshold
        #[arg(long)]
        threshold: Option<f32>,
    },
}

#[async_trait]
impl Command for EmbeddingCommand {
    async fn execute(&self, context: &AppContext) -> Result<()> {
        use lethe_infrastructure::{DatabaseManager, PgEmbeddingRepository};
        use lethe_domain::EmbeddingServiceFactory;
        use std::sync::Arc;

        let db_url = context.database_url.as_ref()
            .ok_or("Database URL is required for embedding management")?;
        let db_manager = Arc::new(DatabaseManager::new(db_url).await?);
        let embedding_repo = Arc::new(PgEmbeddingRepository::new(db_manager.pool().clone()));

        match &self.action {
            EmbeddingAction::List { session_id, limit } => {
                let embeddings = if let Some(session_id) = session_id {
                    embedding_repo.find_by_session(session_id).await?
                } else {
                    embedding_repo.find_recent(*limit).await?
                };

                match &context.output_format {
                    crate::utils::OutputFormat::Json => {
                        println!("{}", serde_json::to_string_pretty(&embeddings)?);
                    }
                    _ => {
                        if embeddings.is_empty() {
                            println!("No embeddings found");
                        } else {
                            println!("🧠 Embeddings ({})", embeddings.len());
                            for embedding in embeddings {
                                println!("  🆔 {}: {} ({}D vector, model: {})", 
                                    embedding.id, 
                                    embedding.chunk_id,
                                    embedding.vector.len(),
                                    embedding.model
                                );
                            }
                        }
                    }
                }
            }
            EmbeddingAction::Show { chunk_id } => {
                let chunk_uuid = uuid::Uuid::parse_str(chunk_id)?;
                let embedding = embedding_repo.find_by_chunk_id(&chunk_uuid).await?
                    .ok_or_else(|| format!("Embedding not found for chunk: {}", chunk_id))?;

                match &context.output_format {
                    crate::utils::OutputFormat::Json => {
                        println!("{}", serde_json::to_string_pretty(&embedding)?);
                    }
                    _ => {
                        println!("🧠 Embedding: {}", embedding.id);
                        println!("   Chunk: {}", embedding.chunk_id);
                        println!("   Model: {}", embedding.model);
                        println!("   Dimensions: {}", embedding.vector.len());
                        println!("   Created: {}", embedding.ts);
                        println!("   Vector preview: {:?}...", &embedding.vector[..embedding.vector.len().min(5)]);
                    }
                }
            }
            EmbeddingAction::Delete { chunk_id, force } => {
                if !force && !context.quiet {
                    use dialoguer::Confirm;
                    if !Confirm::new()
                        .with_prompt(format!("Delete embedding for chunk '{}'?", chunk_id))
                        .interact()? 
                    {
                        println!("Cancelled");
                        return Ok(());
                    }
                }

                let chunk_uuid = uuid::Uuid::parse_str(chunk_id)?;
                embedding_repo.delete(&chunk_uuid).await?;
                
                if !context.quiet {
                    println!("✅ Deleted embedding for chunk: {}", chunk_id);
                }
            }
            EmbeddingAction::Search { query, limit, threshold } => {
                let embedding_service = EmbeddingServiceFactory::create_service(&context.config.embedding).await?;
                let query_vector = embedding_service.embed(query).await?;

                let results = embedding_repo.find_similar(&query_vector, *limit, threshold.unwrap_or(0.0)).await?;

                match &context.output_format {
                    crate::utils::OutputFormat::Json => {
                        println!("{}", serde_json::to_string_pretty(&results)?);
                    }
                    _ => {
                        if results.is_empty() {
                            println!("No similar embeddings found for query: '{}'", query);
                        } else {
                            println!("🔍 Similar embeddings for '{}' ({} results):", query, results.len());
                            for (i, (embedding, similarity)) in results.iter().enumerate() {
                                println!("  {}. 🆔 {} (similarity: {:.4})", 
                                    i + 1, 
                                    embedding.chunk_id,
                                    similarity
                                );
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}