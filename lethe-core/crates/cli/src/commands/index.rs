use async_trait::async_trait;
use clap::Args;
use lethe_shared::Result;
use crate::utils::AppContext;
use super::Command;

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
    async fn execute(&self, context: &AppContext) -> Result<()> {
        use lethe_infrastructure::{DatabaseManager, PgChunkRepository, PgEmbeddingRepository};
        use lethe_domain::EmbeddingServiceFactory;
        use std::sync::Arc;

        if !context.quiet {
            println!("🔄 Building search indices...");
        }

        let db_url = context.database_url.as_ref()
            .ok_or("Database URL is required for indexing")?;
        let db_manager = Arc::new(DatabaseManager::new(db_url).await?);

        let chunk_repo = Arc::new(PgChunkRepository::new(db_manager.pool().clone()));
        let embedding_repo = Arc::new(PgEmbeddingRepository::new(db_manager.pool().clone()));
        
        let embedding_service = EmbeddingServiceFactory::create_service(&context.config.embedding).await?;

        // Get chunks that need indexing
        let chunks = if let Some(session_id) = &self.session_id {
            chunk_repo.find_by_session(session_id).await?
        } else {
            chunk_repo.find_all().await?
        };

        if !context.quiet {
            println!("📊 Found {} chunks to index", chunks.len());
        }

        let mut indexed_count = 0;
        for chunk_batch in chunks.chunks(self.batch_size) {
            for chunk in chunk_batch {
                // Check if embedding exists
                let existing = embedding_repo.find_by_chunk_id(&chunk.id).await?;
                
                if existing.is_none() || self.rebuild {
                    let embedding_vector = embedding_service.embed(&chunk.text).await?;
                    
                    let embedding = lethe_shared::Embedding {
                        id: uuid::Uuid::new_v4(),
                        chunk_id: chunk.id,
                        vector: embedding_vector,
                        model: embedding_service.model_name().to_string(),
                        ts: chrono::Utc::now(),
                    };

                    if existing.is_some() && self.rebuild {
                        embedding_repo.delete(&chunk.id).await?;
                    }
                    
                    embedding_repo.create(&embedding).await?;
                    indexed_count += 1;

                    if !context.quiet && indexed_count % 10 == 0 {
                        println!("   📝 Indexed {} chunks...", indexed_count);
                    }
                }
            }
        }

        if !context.quiet {
            println!("✅ Indexing complete: {} chunks indexed", indexed_count);
        }

        Ok(())
    }
}