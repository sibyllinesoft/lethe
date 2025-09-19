use super::Command;
use crate::utils::AppContext;
use async_trait::async_trait;
use clap::Args;
use lethe_domain::{
    bloom::SimpleBloom,
    corpus::ChunkBloomExport,
    repository_indexer::{
        ChunkRepository as IndexerChunkRepository, MessageRepository as IndexerMessageRepository,
    },
    EmbeddingService, EmbeddingServiceFactory, RepositoryChunkingConfig, RepositoryIndexerFactory,
};
use lethe_shared::{
    utils::HashUtils, Chunk, LetheError, Message, RepositoryConfig, RepositoryPreloadingConfig,
    Result,
};
use parquet::{
    column::writer::ColumnWriter,
    data_type::ByteArray,
    file::properties::WriterProperties,
    file::writer::SerializedFileWriter,
    schema::{parser::parse_message_type, types::ColumnPath},
};
use serde::Serialize;
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use tantivy::{
    schema::{Schema, TantivyDocument, STORED, TEXT},
    Index, Term,
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

        let preloading = match context.config.repository_preloading.as_ref() {
            Some(cfg) => cfg,
            None => {
                if !context.quiet {
                    println!("⚠️  No repository preloading configuration found; nothing to index.");
                }
                return Ok(());
            }
        };

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

        let embedding_config = super::to_domain_embedding_config(&context.config.embedding);
        let embedding_service = Arc::new(EmbeddingServiceFactory::create(&embedding_config).await?);

        let chunking_config = RepositoryChunkingConfig {
            target_tokens: context.config.chunking.target_tokens.value(),
            overlap: context.config.chunking.overlap,
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

            Self::write_taintivy_index(&session_dir, &session_id, repo, &chunks)?;
            Self::write_parquet_messages(&session_dir, &messages)?;

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

    fn write_parquet_chunks(session_dir: &Path, _session_id: &str, chunks: &[Chunk]) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        const CHUNK_SCHEMA: &str = "
        message chunk_record {
          REQUIRED BINARY id (UTF8);
          REQUIRED BINARY message_id (UTF8);
          REQUIRED BINARY session_id (UTF8);
          REQUIRED INT32 offset_start;
          REQUIRED INT32 offset_end;
          REQUIRED BINARY kind (UTF8);
          REQUIRED BINARY text (UTF8);
          REQUIRED INT32 tokens;
        }
        ";

        let path = session_dir.join("chunks.parquet");
        let file = File::create(&path).map_err(|e| {
            LetheError::internal(format!(
                "Failed to create parquet file {}: {}",
                path.display(),
                e
            ))
        })?;

        let schema =
            Arc::new(parse_message_type(CHUNK_SCHEMA).map_err(|e| {
                LetheError::internal(format!("Failed to parse chunk schema: {}", e))
            })?);

        let props = Arc::new(
            WriterProperties::builder()
                .set_column_bloom_filter_enabled(ColumnPath::from(vec!["text".to_string()]), true)
                .build(),
        );

        let mut writer = SerializedFileWriter::new(file, schema, props).map_err(|e| {
            LetheError::internal(format!("Failed to initialise parquet writer: {}", e))
        })?;

        let ids: Vec<ByteArray> = chunks
            .iter()
            .map(|chunk| ByteArray::from(chunk.id.as_bytes()))
            .collect();
        let message_ids: Vec<ByteArray> = chunks
            .iter()
            .map(|chunk| ByteArray::from(chunk.message_id.to_string().into_bytes()))
            .collect();
        let session_ids: Vec<ByteArray> = chunks
            .iter()
            .map(|chunk| ByteArray::from(chunk.session_id.as_bytes()))
            .collect();
        let offset_start: Vec<i32> = chunks
            .iter()
            .map(|chunk| chunk.offset_start as i32)
            .collect();
        let offset_end: Vec<i32> = chunks.iter().map(|chunk| chunk.offset_end as i32).collect();
        let kinds: Vec<ByteArray> = chunks
            .iter()
            .map(|chunk| ByteArray::from(chunk.kind.as_bytes()))
            .collect();
        let texts: Vec<ByteArray> = chunks
            .iter()
            .map(|chunk| ByteArray::from(chunk.text.as_bytes()))
            .collect();
        let tokens: Vec<i32> = chunks.iter().map(|chunk| chunk.tokens).collect();

        let mut row_group = writer
            .next_row_group()
            .map_err(|e| LetheError::internal(format!("Failed to create row group: {}", e)))?;
        let mut column_index = 0;

        while let Some(mut column_writer) = row_group
            .next_column()
            .map_err(|e| LetheError::internal(format!("Failed to obtain column writer: {}", e)))?
        {
            let untyped = column_writer.untyped();
            match column_index {
                0 => {
                    if let ColumnWriter::ByteArrayColumnWriter(ref mut writer) = untyped {
                        writer.write_batch(&ids, None, None).map_err(|e| {
                            LetheError::internal(format!("Failed to write id column: {}", e))
                        })?;
                    } else {
                        return Err(LetheError::internal(
                            "Unexpected column writer type for chunk id",
                        ));
                    }
                }
                1 => {
                    if let ColumnWriter::ByteArrayColumnWriter(ref mut writer) = untyped {
                        writer.write_batch(&message_ids, None, None).map_err(|e| {
                            LetheError::internal(format!(
                                "Failed to write message_id column: {}",
                                e
                            ))
                        })?;
                    } else {
                        return Err(LetheError::internal(
                            "Unexpected column writer type for message_id",
                        ));
                    }
                }
                2 => {
                    if let ColumnWriter::ByteArrayColumnWriter(ref mut writer) = untyped {
                        writer.write_batch(&session_ids, None, None).map_err(|e| {
                            LetheError::internal(format!(
                                "Failed to write session_id column: {}",
                                e
                            ))
                        })?;
                    } else {
                        return Err(LetheError::internal(
                            "Unexpected column writer type for session_id",
                        ));
                    }
                }
                3 => {
                    if let ColumnWriter::Int32ColumnWriter(ref mut writer) = untyped {
                        writer.write_batch(&offset_start, None, None).map_err(|e| {
                            LetheError::internal(format!(
                                "Failed to write offset_start column: {}",
                                e
                            ))
                        })?;
                    } else {
                        return Err(LetheError::internal(
                            "Unexpected column writer type for offset_start",
                        ));
                    }
                }
                4 => {
                    if let ColumnWriter::Int32ColumnWriter(ref mut writer) = untyped {
                        writer.write_batch(&offset_end, None, None).map_err(|e| {
                            LetheError::internal(format!(
                                "Failed to write offset_end column: {}",
                                e
                            ))
                        })?;
                    } else {
                        return Err(LetheError::internal(
                            "Unexpected column writer type for offset_end",
                        ));
                    }
                }
                5 => {
                    if let ColumnWriter::ByteArrayColumnWriter(ref mut writer) = untyped {
                        writer.write_batch(&kinds, None, None).map_err(|e| {
                            LetheError::internal(format!("Failed to write kind column: {}", e))
                        })?;
                    } else {
                        return Err(LetheError::internal(
                            "Unexpected column writer type for kind",
                        ));
                    }
                }
                6 => {
                    if let ColumnWriter::ByteArrayColumnWriter(ref mut writer) = untyped {
                        writer.write_batch(&texts, None, None).map_err(|e| {
                            LetheError::internal(format!("Failed to write text column: {}", e))
                        })?;
                    } else {
                        return Err(LetheError::internal(
                            "Unexpected column writer type for text",
                        ));
                    }
                }
                7 => {
                    if let ColumnWriter::Int32ColumnWriter(ref mut writer) = untyped {
                        writer.write_batch(&tokens, None, None).map_err(|e| {
                            LetheError::internal(format!("Failed to write tokens column: {}", e))
                        })?;
                    } else {
                        return Err(LetheError::internal(
                            "Unexpected column writer type for tokens",
                        ));
                    }
                }
                _ => {}
            }

            column_writer.close().map_err(|e| {
                LetheError::internal(format!("Failed to close column writer: {}", e))
            })?;
            column_index += 1;
        }

        row_group
            .close()
            .map_err(|e| LetheError::internal(format!("Failed to close row group: {}", e)))?;
        writer
            .close()
            .map_err(|e| LetheError::internal(format!("Failed to close parquet writer: {}", e)))?;

        Ok(())
    }

    fn write_parquet_messages(session_dir: &Path, messages: &[Message]) -> Result<()> {
        if messages.is_empty() {
            return Ok(());
        }

        const MESSAGE_SCHEMA: &str = "
        message message_record {
          REQUIRED BINARY id (UTF8);
          REQUIRED BINARY session_id (UTF8);
          REQUIRED INT32 turn;
          REQUIRED BINARY role (UTF8);
          REQUIRED BINARY text (UTF8);
          OPTIONAL BINARY meta (UTF8);
          REQUIRED BINARY ts (UTF8);
        }
        ";

        let path = session_dir.join("messages.parquet");
        let file = File::create(&path).map_err(|e| {
            LetheError::internal(format!(
                "Failed to create message parquet {}: {}",
                path.display(),
                e
            ))
        })?;

        let schema =
            Arc::new(parse_message_type(MESSAGE_SCHEMA).map_err(|e| {
                LetheError::internal(format!("Failed to parse message schema: {}", e))
            })?);

        let props = Arc::new(WriterProperties::builder().build());
        let mut writer = SerializedFileWriter::new(file, schema, props).map_err(|e| {
            LetheError::internal(format!("Failed to initialise message writer: {}", e))
        })?;

        let ids: Vec<ByteArray> = messages
            .iter()
            .map(|message| ByteArray::from(message.id.to_string().into_bytes()))
            .collect();
        let session_ids: Vec<ByteArray> = messages
            .iter()
            .map(|message| ByteArray::from(message.session_id.as_bytes()))
            .collect();
        let turns: Vec<i32> = messages.iter().map(|message| message.turn).collect();
        let roles: Vec<ByteArray> = messages
            .iter()
            .map(|message| ByteArray::from(message.role.as_bytes()))
            .collect();
        let texts: Vec<ByteArray> = messages
            .iter()
            .map(|message| ByteArray::from(message.text.as_bytes()))
            .collect();
        let meta_values: Vec<Option<ByteArray>> = messages
            .iter()
            .map(|message| {
                message
                    .meta
                    .as_ref()
                    .and_then(|meta| serde_json::to_vec(meta).ok())
                    .map(ByteArray::from)
            })
            .collect();
        let meta_packed: Vec<ByteArray> = meta_values
            .iter()
            .filter_map(|value| value.clone())
            .collect();
        let meta_def_levels: Vec<i16> = meta_values
            .iter()
            .map(|value| if value.is_some() { 1 } else { 0 })
            .collect();
        let timestamps: Vec<ByteArray> = messages
            .iter()
            .map(|message| ByteArray::from(message.ts.to_rfc3339().into_bytes()))
            .collect();

        let mut row_group = writer.next_row_group().map_err(|e| {
            LetheError::internal(format!("Failed to create message row group: {}", e))
        })?;
        let mut column_index = 0;

        while let Some(mut column_writer) = row_group.next_column().map_err(|e| {
            LetheError::internal(format!("Failed to obtain message column writer: {}", e))
        })? {
            let untyped = column_writer.untyped();
            match column_index {
                0 => {
                    if let ColumnWriter::ByteArrayColumnWriter(ref mut writer) = untyped {
                        writer.write_batch(&ids, None, None).map_err(|e| {
                            LetheError::internal(format!(
                                "Failed to write message id column: {}",
                                e
                            ))
                        })?;
                    }
                }
                1 => {
                    if let ColumnWriter::ByteArrayColumnWriter(ref mut writer) = untyped {
                        writer.write_batch(&session_ids, None, None).map_err(|e| {
                            LetheError::internal(format!(
                                "Failed to write message session column: {}",
                                e
                            ))
                        })?;
                    }
                }
                2 => {
                    if let ColumnWriter::Int32ColumnWriter(ref mut writer) = untyped {
                        writer.write_batch(&turns, None, None).map_err(|e| {
                            LetheError::internal(format!(
                                "Failed to write message turn column: {}",
                                e
                            ))
                        })?;
                    }
                }
                3 => {
                    if let ColumnWriter::ByteArrayColumnWriter(ref mut writer) = untyped {
                        writer.write_batch(&roles, None, None).map_err(|e| {
                            LetheError::internal(format!(
                                "Failed to write message role column: {}",
                                e
                            ))
                        })?;
                    }
                }
                4 => {
                    if let ColumnWriter::ByteArrayColumnWriter(ref mut writer) = untyped {
                        writer.write_batch(&texts, None, None).map_err(|e| {
                            LetheError::internal(format!(
                                "Failed to write message text column: {}",
                                e
                            ))
                        })?;
                    }
                }
                5 => {
                    if let ColumnWriter::ByteArrayColumnWriter(ref mut writer) = untyped {
                        writer
                            .write_batch(&meta_packed, Some(&meta_def_levels), None)
                            .map_err(|e| {
                                LetheError::internal(format!(
                                    "Failed to write message metadata column: {}",
                                    e
                                ))
                            })?;
                    }
                }
                6 => {
                    if let ColumnWriter::ByteArrayColumnWriter(ref mut writer) = untyped {
                        writer.write_batch(&timestamps, None, None).map_err(|e| {
                            LetheError::internal(format!(
                                "Failed to write message timestamp column: {}",
                                e
                            ))
                        })?;
                    }
                }
                _ => {}
            }

            column_writer.close().map_err(|e| {
                LetheError::internal(format!("Failed to close message column: {}", e))
            })?;
            column_index += 1;
        }

        row_group.close().map_err(|e| {
            LetheError::internal(format!("Failed to close message row group: {}", e))
        })?;
        writer
            .close()
            .map_err(|e| LetheError::internal(format!("Failed to close message writer: {}", e)))?;

        Ok(())
    }

    fn tokenize_for_bloom(text: &str) -> HashSet<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase())
            .collect()
    }

    fn write_chunk_bloom(session_dir: &Path, chunks: &[Chunk]) -> Result<()> {
        if chunks.is_empty() {
            let path = session_dir.join("chunks.bloom");
            if path.exists() {
                std::fs::remove_file(path).ok();
            }
            return Ok(());
        }

        let mut exports = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let tokens = Self::tokenize_for_bloom(&chunk.text);
            let mut bloom = SimpleBloom::new(tokens.len().max(1), 0.01);
            for token in tokens {
                bloom.insert(&token);
            }
            exports.push(ChunkBloomExport {
                chunk_id: chunk.id.clone(),
                filter: bloom,
            });
        }

        let data = bincode::serialize(&exports).map_err(|e| {
            LetheError::internal(format!("Failed to serialise bloom filters: {}", e))
        })?;
        std::fs::write(session_dir.join("chunks.bloom"), data)
            .map_err(|e| LetheError::internal(format!("Failed to write bloom filters: {}", e)))?;
        Ok(())
    }

    fn build_schema() -> Schema {
        let mut builder = Schema::builder();
        builder.add_text_field("session_id", TEXT | STORED);
        builder.add_text_field("repository_path", TEXT | STORED);
        builder.add_text_field("doc_id", TEXT | STORED);
        builder.add_text_field("kind", TEXT | STORED);
        builder.add_text_field("text", TEXT | STORED);
        builder.add_text_field("metadata", STORED);
        builder.build()
    }

    fn write_taintivy_index(
        session_dir: &Path,
        session_id: &str,
        repo: &RepositoryConfig,
        chunks: &[Chunk],
    ) -> Result<()> {
        Self::write_parquet_chunks(session_dir, session_id, chunks)?;
        Self::write_chunk_bloom(session_dir, chunks)?;

        let schema = Self::build_schema();
        let index = if session_dir.join("meta.json").exists() {
            Index::open_in_dir(session_dir).map_err(|e| {
                LetheError::internal(format!("Failed to open taintivy index: {}", e))
            })?
        } else {
            Index::create_in_dir(session_dir, schema.clone()).map_err(|e| {
                LetheError::internal(format!("Failed to create taintivy index: {}", e))
            })?
        };

        let schema = index.schema();
        let session_field = schema
            .get_field("session_id")
            .map_err(|_| LetheError::internal("taintivy schema missing session_id field"))?;
        let repo_field = schema
            .get_field("repository_path")
            .map_err(|_| LetheError::internal("taintivy schema missing repository_path field"))?;
        let doc_id_field = schema
            .get_field("doc_id")
            .map_err(|_| LetheError::internal("taintivy schema missing doc_id field"))?;
        let kind_field = schema
            .get_field("kind")
            .map_err(|_| LetheError::internal("taintivy schema missing kind field"))?;
        let text_field = schema
            .get_field("text")
            .map_err(|_| LetheError::internal("taintivy schema missing text field"))?;
        let metadata_field = schema
            .get_field("metadata")
            .map_err(|_| LetheError::internal("taintivy schema missing metadata field"))?;

        let mut writer = index.writer(50_000_000).map_err(|e| {
            LetheError::internal(format!("Failed to create taintivy writer: {}", e))
        })?;

        writer.delete_term(Term::from_field_text(session_field, session_id));

        for chunk in chunks {
            let mut document = TantivyDocument::new();
            document.add_text(session_field, session_id);
            document.add_text(repo_field, &repo.path);
            document.add_text(doc_id_field, &chunk.id);
            document.add_text(kind_field, &chunk.kind);
            document.add_text(text_field, &chunk.text);
            document.add_text(
                metadata_field,
                serde_json::to_string(chunk).map_err(|e| {
                    LetheError::internal(format!("Failed to serialize chunk: {}", e))
                })?,
            );
            let _ = writer.add_document(document);
        }

        writer
            .commit()
            .map_err(|e| LetheError::internal(format!("Failed to commit taintivy index: {}", e)))?;

        Ok(())
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
