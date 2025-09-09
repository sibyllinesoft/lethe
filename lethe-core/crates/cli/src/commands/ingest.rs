use async_trait::async_trait;
use clap::Args;
use lethe_shared::Result;
use std::path::PathBuf;
use crate::utils::AppContext;
use super::Command;

#[derive(Debug, Args)]
pub struct IngestCommand {
    /// Directory or file to ingest
    #[arg(required = true)]
    input: Vec<PathBuf>,

    /// Session ID to associate with ingested documents
    #[arg(long, short)]
    session_id: Option<String>,

    /// Recursive directory traversal
    #[arg(long, short)]
    recursive: bool,

    /// File patterns to include (glob patterns)
    #[arg(long)]
    include: Vec<String>,

    /// File patterns to exclude (glob patterns)
    #[arg(long)]
    exclude: Vec<String>,

    /// Chunk size for text processing
    #[arg(long, default_value = "1000")]
    chunk_size: usize,

    /// Chunk overlap for text processing
    #[arg(long, default_value = "200")]
    chunk_overlap: usize,

    /// Skip files that are already ingested
    #[arg(long)]
    skip_existing: bool,

    /// Batch size for processing
    #[arg(long, default_value = "10")]
    batch_size: usize,
}

#[async_trait]
impl Command for IngestCommand {
    async fn execute(&self, context: &AppContext) -> Result<()> {
        use lethe_domain::{ChunkerService, EmbeddingServiceFactory};
        use lethe_infrastructure::{DatabaseManager, PgMessageRepository, PgChunkRepository, PgEmbeddingRepository};
        use lethe_shared::{Message, MessageRole};
        use std::fs;
        use walkdir::WalkDir;
        use uuid::Uuid;
        use chrono::Utc;
        use std::sync::Arc;

        if !context.quiet {
            println!("🔄 Starting document ingestion...");
        }

        // Initialize database connection
        let db_url = context.database_url.as_ref()
            .ok_or("Database URL is required for ingestion")?;
        let db_manager = Arc::new(DatabaseManager::new(db_url).await?);

        // Initialize repositories
        let message_repo = Arc::new(PgMessageRepository::new(db_manager.pool().clone()));
        let chunk_repo = Arc::new(PgChunkRepository::new(db_manager.pool().clone()));
        let embedding_repo = Arc::new(PgEmbeddingRepository::new(db_manager.pool().clone()));

        // Initialize services
        let chunker = ChunkerService::new(self.chunk_size, self.chunk_overlap);
        let embedding_service = EmbeddingServiceFactory::create_service(&context.config.embedding).await?;

        // Generate session ID if not provided
        let session_id = self.session_id.clone()
            .unwrap_or_else(|| format!("ingest-{}", Uuid::new_v4()));

        // Collect files to process
        let mut files_to_process = Vec::new();

        for input_path in &self.input {
            if input_path.is_file() {
                if self.should_process_file(input_path) {
                    files_to_process.push(input_path.clone());
                }
            } else if input_path.is_dir() {
                let walker = if self.recursive {
                    WalkDir::new(input_path)
                } else {
                    WalkDir::new(input_path).max_depth(1)
                };

                for entry in walker {
                    let entry = entry.map_err(|e| format!("Directory traversal error: {}", e))?;
                    let path = entry.path();

                    if path.is_file() && self.should_process_file(path) {
                        files_to_process.push(path.to_path_buf());
                    }
                }
            } else {
                return Err(format!("Path does not exist: {}", input_path.display()).into());
            }
        }

        if files_to_process.is_empty() {
            if !context.quiet {
                println!("⚠️  No files found to process");
            }
            return Ok(());
        }

        if !context.quiet {
            println!("📁 Found {} files to process", files_to_process.len());
        }

        let mut processed_count = 0;
        let mut error_count = 0;

        // Process files in batches
        for batch in files_to_process.chunks(self.batch_size) {
            for file_path in batch {
                match self.process_file(
                    file_path,
                    &session_id,
                    &chunker,
                    &embedding_service,
                    &message_repo,
                    &chunk_repo,
                    &embedding_repo,
                    context,
                ).await {
                    Ok(_) => {
                        processed_count += 1;
                        if !context.quiet {
                            println!("✅ Processed: {}", file_path.display());
                        }
                    }
                    Err(e) => {
                        error_count += 1;
                        eprintln!("❌ Error processing {}: {}", file_path.display(), e);
                    }
                }
            }
        }

        if !context.quiet {
            println!("\n📊 Ingestion Summary:");
            println!("   ✅ Successfully processed: {}", processed_count);
            if error_count > 0 {
                println!("   ❌ Failed to process: {}", error_count);
            }
            println!("   📝 Session ID: {}", session_id);
        }

        Ok(())
    }
}

impl IngestCommand {
    fn should_process_file(&self, path: &PathBuf) -> bool {
        // Skip directories
        if path.is_dir() {
            return false;
        }

        let path_str = path.to_string_lossy();

        // Check exclude patterns first
        for pattern in &self.exclude {
            if glob::Pattern::new(pattern)
                .map(|p| p.matches(&path_str))
                .unwrap_or(false)
            {
                return false;
            }
        }

        // If include patterns specified, file must match at least one
        if !self.include.is_empty() {
            return self.include.iter().any(|pattern| {
                glob::Pattern::new(pattern)
                    .map(|p| p.matches(&path_str))
                    .unwrap_or(false)
            });
        }

        // Default: process common text files
        matches!(
            path.extension().and_then(|s| s.to_str()),
            Some("txt" | "md" | "rst" | "json" | "yaml" | "yml" | "toml" | "csv" | "tsv")
        )
    }

    async fn process_file(
        &self,
        file_path: &PathBuf,
        session_id: &str,
        chunker: &ChunkerService,
        embedding_service: &Arc<dyn lethe_domain::EmbeddingService>,
        message_repo: &Arc<PgMessageRepository>,
        chunk_repo: &Arc<PgChunkRepository>,
        embedding_repo: &Arc<PgEmbeddingRepository>,
        _context: &AppContext,
    ) -> Result<()> {
        use lethe_shared::{Chunk, Embedding};

        // Read file content
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| format!("Failed to read file {}: {}", file_path.display(), e))?;

        // Create a message for this document
        let message_id = Uuid::new_v4();
        let message = Message {
            id: message_id,
            session_id: session_id.to_string(),
            turn: 0,
            role: MessageRole::User,
            text: content.clone(),
            ts: Utc::now(),
            meta: Some(serde_json::json!({
                "source_file": file_path.to_string_lossy(),
                "ingestion_type": "document"
            })),
        };

        // Save message
        message_repo.create(&message).await?;

        // Chunk the document
        let chunks = chunker.chunk_text(&content, Some(file_path.to_string_lossy().to_string()))?;

        // Process chunks
        for (i, chunk_text) in chunks.into_iter().enumerate() {
            // Create chunk
            let chunk_id = Uuid::new_v4();
            let chunk = Chunk {
                id: chunk_id,
                message_id,
                session_id: session_id.to_string(),
                idx: i as i32,
                text: chunk_text.clone(),
                ts: Utc::now(),
                meta: Some(serde_json::json!({
                    "source_file": file_path.to_string_lossy(),
                    "chunk_index": i
                })),
            };

            // Save chunk
            chunk_repo.create(&chunk).await?;

            // Generate and save embedding
            let embedding_vector = embedding_service.embed(&chunk_text).await?;
            let embedding = Embedding {
                id: Uuid::new_v4(),
                chunk_id,
                vector: embedding_vector,
                model: embedding_service.model_name().to_string(),
                ts: Utc::now(),
            };

            embedding_repo.create(&embedding).await?;
        }

        Ok(())
    }
}