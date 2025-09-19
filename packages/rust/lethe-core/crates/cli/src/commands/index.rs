use super::Command;
use crate::utils::AppContext;
use async_trait::async_trait;
use clap::Args;
use lethe_domain::{
    repository_indexer::{
        ChunkRepository as IndexerChunkRepository, MessageRepository as IndexerMessageRepository,
    },
    EmbeddingService, EmbeddingServiceFactory, RepositoryChunkingConfig, RepositoryIndexerFactory,
};
use lethe_shared::{
    utils::HashUtils, Chunk, LetheError, Message, RepositoryConfig, RepositoryPreloadingConfig,
    Result,
};
use lethe_storage::write_session_artifacts;
use serde::Serialize;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use tokio::{fs, sync::Mutex};
use uuid::Uuid;

#[derive(Debug, Args)]
pub struct IndexCommand {
    /// Rebuild all indices even if data already exists
    #[arg(long)]
    rebuild: bool,

    /// Index only the repository matching this session ID or name
    #[arg(long)]
    session_id: Option<String>,

    /// Batch size used when generating embeddings
    #[arg(long, default_value = "100")]
    batch_size: usize,

    /// Directory where taintivy indexes will be stored
    #[arg(long, default_value = "./taintivy-index")]
    index_dir: PathBuf,
}

#[derive(Default)]
struct InMemoryChunkRepository {
    chunks: Mutex<HashMap<String, Vec<Chunk>>>,
}

#[async_trait]
impl IndexerChunkRepository for InMemoryChunkRepository {
    async fn create_chunk(&self, chunk: &Chunk) -> Result<Chunk> {
        let mut guard = self.chunks.lock().await;
        guard
            .entry(chunk.session_id.clone())
            .or_default()
            .push(chunk.clone());
        Ok(chunk.clone())
    }

    async fn batch_create_chunks(&self, chunks: &[Chunk]) -> Result<Vec<Chunk>> {
        let mut guard = self.chunks.lock().await;
        let mut created = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            guard
                .entry(chunk.session_id.clone())
                .or_default()
                .push(chunk.clone());
            created.push(chunk.clone());
        }
        Ok(created)
    }

    async fn get_chunks_by_session(&self, session_id: &str) -> Result<Vec<Chunk>> {
        let guard = self.chunks.lock().await;
        Ok(guard.get(session_id).cloned().unwrap_or_default())
    }

    async fn delete_chunks_by_session(&self, session_id: &str) -> Result<bool> {
        let mut guard = self.chunks.lock().await;
        Ok(guard.remove(session_id).is_some())
    }
}

#[derive(Default)]
struct InMemoryMessageRepository {
    messages: Mutex<HashMap<Uuid, Message>>,
}

impl InMemoryMessageRepository {
    async fn messages_for_session(&self, session_id: &str) -> Vec<Message> {
        let guard = self.messages.lock().await;
        guard
            .values()
            .filter(|message| message.session_id == session_id)
            .cloned()
            .collect()
    }

    async fn clear(&self) {
        let mut guard = self.messages.lock().await;
        guard.clear();
    }
}

#[async_trait]
impl IndexerMessageRepository for InMemoryMessageRepository {
    async fn create_message(&self, message: &Message) -> Result<Message> {
        let mut guard = self.messages.lock().await;
        guard.insert(message.id, message.clone());
        Ok(message.clone())
    }

    async fn get_message(&self, id: &Uuid) -> Result<Option<Message>> {
        let guard = self.messages.lock().await;
        Ok(guard.get(id).cloned())
    }

    async fn delete_message(&self, id: &Uuid) -> Result<bool> {
        let mut guard = self.messages.lock().await;
        Ok(guard.remove(id).is_some())
    }
}

#[derive(Serialize)]
struct StoredEmbedding {
    chunk_id: String,
    vector: Vec<f32>,
}

#[async_trait]
impl Command for IndexCommand {
    async fn execute(&self, context: &AppContext) -> Result<()> {
        if !context.quiet {
            println!(
                "⚙️  Preparing taintivy index directory at {}",
                self.index_dir.display()
            );
        }
        fs::create_dir_all(&self.index_dir).await?;

        let preloading = &context.resolved_config.repository_preloading;

        if preloading.repositories.is_empty() && self.session_id.is_none() {
            if !context.quiet {
                println!(
                    "⚠️  No repositories configured for preloading; add entries or target a session with --session-id."
                );
            }
            return Ok(());
        }

        if !preloading.enabled && self.session_id.is_none() {
            if !context.quiet {
                println!(
                    "⚠️  Repository preloading is disabled. Enable it in configuration or target a specific repository with --session-id."
                );
            }
            return Ok(());
        }

        let repositories = self.select_repositories(preloading)?;
        if repositories.is_empty() {
            if !context.quiet {
                println!("⚠️  No repositories matched the requested filters.");
            }
            return Ok(());
        }

        let batch_size = self.batch_size.max(1);

        if !context.quiet {
            println!(
                "📚 Indexing {} repository{} with taintivy...",
                repositories.len(),
                if repositories.len() == 1 { "" } else { "ies" }
            );
        }

        let embedding_config =
            super::to_domain_embedding_config(&context.resolved_config.embedding);
        let embedding_service = Arc::new(EmbeddingServiceFactory::create(&embedding_config).await?);

        let chunking_config = RepositoryChunkingConfig {
            target_tokens: context.resolved_config.chunking.target_tokens.value(),
            overlap: context.resolved_config.chunking.overlap,
        };
        let indexer = RepositoryIndexerFactory::create_indexer(chunking_config, preloading)?;

        let mut processed_repos = 0usize;
        let mut total_files = 0usize;
        let mut total_chunks = 0usize;
        let mut total_embeddings = 0usize;
        let mut total_errors = 0usize;

        for repo in repositories {
            let session_id = Self::compute_session_id(repo);
            let session_dir = self.index_dir.join(&session_id);

            if session_dir.exists() {
                if self.rebuild {
                    fs::remove_dir_all(&session_dir).await?;
                } else {
                    if !context.quiet {
                        println!(
                            "⏭️  Skipping {} (session: {}) because an index already exists. Use --rebuild to regenerate.",
                            repo.path, session_id
                        );
                    }
                    continue;
                }
            }

            fs::create_dir_all(&session_dir).await?;

            if !context.quiet {
                println!("\n➡️  {} (session: {})", repo.path, session_id);
            }

            let chunk_repo = InMemoryChunkRepository::default();
            let message_repo = InMemoryMessageRepository::default();

            let timer = Instant::now();
            let result = indexer
                .index_repository(repo, &chunk_repo, &message_repo)
                .await?;

            processed_repos += 1;
            total_files += result.indexed_files;
            total_chunks += result.total_chunks;
            total_errors += result.errors.len();

            if !context.quiet {
                println!(
                    "   ✅ Indexed {} files → {} chunks ({}ms)",
                    result.indexed_files, result.total_chunks, result.duration_ms
                );
                for error in &result.errors {
                    println!("     ❌ {}: {}", error.file_path, error.error);
                }
            }

            let chunks = chunk_repo.get_chunks_by_session(&session_id).await?;
            let messages = message_repo.messages_for_session(&session_id).await;
            if chunks.is_empty() {
                if !context.quiet {
                    println!("   ⚠️  No chunks produced; skipping taintivy indexing.");
                }
                message_repo.clear().await;
                continue;
            }

            write_session_artifacts(&session_dir, &session_id, repo, &chunks, &messages)?;

            if !context.quiet {
                println!(
                    "   🧠 Generating embeddings for {} chunks (batch size {})...",
                    chunks.len(),
                    batch_size
                );
            }
            let created =
                Self::embed_chunks(&embedding_service, &chunks, batch_size, &session_dir).await?;
            total_embeddings += created;

            message_repo.clear().await;

            if !context.quiet {
                let elapsed_ms = timer.elapsed().as_millis();
                println!(
                    "   🎯 Indexing and vector export completed in {}ms",
                    elapsed_ms
                );
            }
        }

        if !context.quiet {
            println!("\n📊 Indexing Summary:");
            println!("   📁 Repositories processed: {}", processed_repos);
            println!("   📄 Files indexed: {}", total_files);
            println!("   🧩 Chunks stored: {}", total_chunks);
            println!("   🧠 Embeddings exported: {}", total_embeddings);
            println!("   📦 Index root: {}", self.index_dir.display());
            if total_errors > 0 {
                println!("   ⚠️  Files with errors: {}", total_errors);
            }
        }

        Ok(())
    }
}

impl IndexCommand {
    fn compute_session_id(repo: &RepositoryConfig) -> String {
        format!(
            "repo_{}_{}",
            repo.name.as_deref().unwrap_or("unnamed"),
            HashUtils::short_hash(&repo.path)
        )
    }

    fn select_repositories<'a>(
        &'a self,
        config: &'a RepositoryPreloadingConfig,
    ) -> Result<Vec<&'a RepositoryConfig>> {
        if let Some(ref filter) = self.session_id {
            let matches: Vec<&RepositoryConfig> = config
                .repositories
                .iter()
                .filter(|repo| {
                    let session_id = Self::compute_session_id(repo);
                    session_id == *filter
                        || repo
                            .name
                            .as_ref()
                            .map(|name| name.eq_ignore_ascii_case(filter))
                            .unwrap_or(false)
                })
                .collect();

            if matches.is_empty() {
                Err(LetheError::config(format!(
                    "No repository found for session or name '{}'.",
                    filter
                )))
            } else {
                Ok(matches)
            }
        } else {
            Ok(config
                .repositories
                .iter()
                .filter(|repo| repo.enabled)
                .collect())
        }
    }

    async fn embed_chunks(
        embedding_service: &Arc<dyn EmbeddingService>,
        chunks: &[Chunk],
        batch_size: usize,
        session_dir: &Path,
    ) -> Result<usize> {
        if chunks.is_empty() {
            return Ok(0);
        }

        let embeddings_dir = session_dir.join("embeddings");
        fs::create_dir_all(&embeddings_dir).await?;
        let mut stored = Vec::new();

        for batch in chunks.chunks(batch_size) {
            let texts: Vec<String> = batch.iter().map(|chunk| chunk.text.clone()).collect();
            let embeddings = embedding_service.embed(&texts).await?;

            if embeddings.len() != batch.len() {
                return Err(LetheError::embedding(format!(
                    "Embedding service returned {} vectors for {} chunks",
                    embeddings.len(),
                    batch.len()
                )));
            }

            stored.extend(batch.iter().cloned().zip(embeddings.into_iter()).map(
                |(chunk, vector)| StoredEmbedding {
                    chunk_id: chunk.id,
                    vector: vector.data,
                },
            ));
        }

        let file_path = embeddings_dir.join("vectors.json");
        let payload = serde_json::to_vec_pretty(&stored)?;
        fs::write(file_path, payload).await?;

        Ok(stored.len())
    }
}
