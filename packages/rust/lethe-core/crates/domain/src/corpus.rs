use crate::{bloom::SimpleBloom, retrieval::DocumentRepository};
use async_trait::async_trait;
use lethe_shared::{Candidate, Chunk, DfIdf, EmbeddingVector, LetheError, Result};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::{Path, PathBuf},
};
use tokio::sync::RwLock;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredEmbedding {
    chunk_id: String,
    vector: Vec<f32>,
}

#[derive(Debug)]
struct SessionCache {
    chunks: Vec<Chunk>,
    dfidf: Vec<DfIdf>,
}

#[derive(Default)]
struct CorpusCaches {
    sessions: HashMap<String, SessionCache>,
    chunk_map: HashMap<String, Chunk>,
    bloom: HashMap<String, SimpleBloom>,
    embeddings: Option<HashMap<String, EmbeddingVector>>,
}

/// Parquet-backed corpus for retrieval and vector search
pub struct ParquetCorpus {
    root: PathBuf,
    caches: RwLock<CorpusCaches>,
}

impl ParquetCorpus {
    pub fn new<P: Into<PathBuf>>(root: P) -> Self {
        Self {
            root: root.into(),
            caches: RwLock::new(CorpusCaches::default()),
        }
    }

    fn session_dir(&self, session_id: &str) -> PathBuf {
        self.root.join(session_id)
    }

    async fn load_session(&self, session_id: &str) -> Result<(Vec<Chunk>, Vec<DfIdf>)> {
        {
            let caches = self.caches.read().await;
            if let Some(session) = caches.sessions.get(session_id) {
                return Ok((session.chunks.clone(), session.dfidf.clone()));
            }
        }

        let chunks = self.read_chunks(session_id)?;
        let dfidf = self.compute_dfidf(session_id, &chunks);

        let mut cache = self.caches.write().await;
        let entry = cache
            .sessions
            .entry(session_id.to_string())
            .or_insert(SessionCache {
                chunks: chunks.clone(),
                dfidf: dfidf.clone(),
            });
        entry.chunks = chunks.clone();
        entry.dfidf = dfidf.clone();

        for chunk in &chunks {
            cache.chunk_map.insert(chunk.id.clone(), chunk.clone());
        }

        Ok((chunks, dfidf))
    }

    fn read_chunks(&self, session_id: &str) -> Result<Vec<Chunk>> {
        let path = self.session_dir(session_id).join("chunks.parquet");
        if !path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&path).map_err(|e| {
            LetheError::internal(format!(
                "Failed to open parquet file {}: {}",
                path.display(),
                e
            ))
        })?;
        let reader = SerializedFileReader::new(file).map_err(|e| {
            LetheError::internal(format!(
                "Failed to read parquet file {}: {}",
                path.display(),
                e
            ))
        })?;

        let mut iter = reader
            .get_row_iter(None)
            .map_err(|e| LetheError::internal(format!("Failed to iterate parquet rows: {}", e)))?;
        let mut chunks = Vec::new();

        while let Some(record) = iter.next() {
            let record = record
                .map_err(|e| LetheError::internal(format!("Failed to read parquet row: {}", e)))?;

            let id = record
                .get_string(0)
                .map(|s| s.to_string())
                .map_err(|e| LetheError::internal(format!("Invalid chunk id: {}", e)))?;
            let message_id_str = record
                .get_string(1)
                .map_err(|e| LetheError::internal(format!("Invalid message id: {}", e)))?;
            let message_id = Uuid::parse_str(message_id_str).map_err(|_| {
                LetheError::internal(format!(
                    "Invalid message UUID in parquet: {}",
                    message_id_str
                ))
            })?;
            let session = record
                .get_string(2)
                .map(|s| s.to_string())
                .unwrap_or_else(|_| session_id.to_string());
            let offset_start = record.get_int(3).unwrap_or(0) as usize;
            let offset_end = record.get_int(4).unwrap_or(0) as usize;
            let kind = record
                .get_string(5)
                .map(|s| s.to_string())
                .unwrap_or_else(|_| "text".to_string());
            let text = record
                .get_string(6)
                .map(|s| s.to_string())
                .unwrap_or_default();
            let tokens = record.get_int(7).unwrap_or_default();

            chunks.push(Chunk {
                id,
                message_id,
                session_id: session,
                offset_start,
                offset_end,
                kind,
                text,
                tokens,
            });
        }

        Ok(chunks)
    }

    fn compute_dfidf(&self, session_id: &str, chunks: &[Chunk]) -> Vec<DfIdf> {
        if chunks.is_empty() {
            return Vec::new();
        }

        let mut doc_freq: HashMap<String, usize> = HashMap::new();
        for chunk in chunks {
            let mut seen = HashSet::new();
            for term in Self::tokenize(&chunk.text) {
                if seen.insert(term.clone()) {
                    *doc_freq.entry(term).or_insert(0) += 1;
                }
            }
        }

        let total_docs = chunks.len() as f64;
        doc_freq
            .into_iter()
            .map(|(term, df)| DfIdf {
                term,
                session_id: session_id.to_string(),
                df: df as i32,
                idf: ((total_docs + 1.0) / (df as f64 + 1.0)).ln(),
            })
            .collect()
    }

    fn tokenize(text: &str) -> impl Iterator<Item = String> + '_ {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase())
    }

    async fn load_bloom(&self, session_id: &str) -> Result<()> {
        let mut cache = self.caches.write().await;
        let has_entries = cache.bloom.keys().any(|key| key.starts_with(session_id));
        if has_entries {
            return Ok(());
        }

        let path = self.session_dir(session_id).join("chunks.bloom");
        if !path.exists() {
            return Ok(());
        }

        let data = std::fs::read(&path).map_err(|e| {
            LetheError::internal(format!(
                "Failed to read bloom filter {}: {}",
                path.display(),
                e
            ))
        })?;
        let entries: Vec<ChunkBloomExport> = bincode::deserialize(&data).map_err(|e| {
            LetheError::internal(format!(
                "Failed to deserialize bloom filter {}: {}",
                path.display(),
                e
            ))
        })?;

        for entry in entries {
            cache
                .bloom
                .insert(format!("{}/{}", session_id, entry.chunk_id), entry.filter);
        }

        Ok(())
    }

    async fn load_embeddings(&self) -> Result<HashMap<String, EmbeddingVector>> {
        {
            let cache = self.caches.read().await;
            if let Some(ref embeddings) = cache.embeddings {
                return Ok(embeddings.clone());
            }
        }

        let mut combined = HashMap::new();
        if !self.root.exists() {
            return Ok(combined);
        }

        for entry in std::fs::read_dir(&self.root).map_err(|e| {
            LetheError::internal(format!(
                "Failed to read storage root {}: {}",
                self.root.display(),
                e
            ))
        })? {
            let entry = entry.map_err(|e| {
                LetheError::internal(format!("Failed to read storage entry: {}", e))
            })?;
            if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                let embedding_path = entry.path().join("embeddings").join("vectors.json");
                if embedding_path.exists() {
                    let data = std::fs::read_to_string(&embedding_path).map_err(|e| {
                        LetheError::internal(format!(
                            "Failed to read embeddings {}: {}",
                            embedding_path.display(),
                            e
                        ))
                    })?;
                    let entries: Vec<StoredEmbedding> =
                        serde_json::from_str(&data).map_err(|e| {
                            LetheError::internal(format!(
                                "Failed to deserialize embeddings {}: {}",
                                embedding_path.display(),
                                e
                            ))
                        })?;
                    for entry in entries {
                        combined.insert(
                            entry.chunk_id.clone(),
                            EmbeddingVector {
                                data: entry.vector.clone(),
                                dimension: entry.vector.len(),
                            },
                        );
                    }
                }
            }
        }

        let mut cache = self.caches.write().await;
        cache.embeddings = Some(combined.clone());
        Ok(combined)
    }

    pub async fn stats(&self) -> Result<CorpusStats> {
        let root = self.root.clone();
        tokio::task::spawn_blocking(move || collect_stats(&root))
            .await
            .map_err(|e| LetheError::internal(format!("Failed to join stats task: {}", e)))?
    }

    pub async fn health_check(&self) -> Result<()> {
        if !self.root.exists() {
            std::fs::create_dir_all(&self.root).map_err(|e| {
                LetheError::internal(format!(
                    "Failed to create storage directory {}: {}",
                    self.root.display(),
                    e
                ))
            })?;
        }
        Ok(())
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (*x as f64) * (*y as f64))
            .sum();
        let norm_a: f64 = a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkBloomExport {
    pub chunk_id: String,
    pub filter: SimpleBloom,
}

#[derive(Debug, Clone)]
pub struct CorpusStats {
    pub session_count: usize,
    pub chunk_count: usize,
    pub message_count: usize,
    pub embedding_count: usize,
}

fn collect_stats(root: &Path) -> Result<CorpusStats> {
    let mut stats = CorpusStats {
        session_count: 0,
        chunk_count: 0,
        message_count: 0,
        embedding_count: 0,
    };

    if !root.exists() {
        return Ok(stats);
    }

    for entry in std::fs::read_dir(root).map_err(|e| {
        LetheError::internal(format!(
            "Failed to read storage root {}: {}",
            root.display(),
            e
        ))
    })? {
        let entry = entry
            .map_err(|e| LetheError::internal(format!("Failed to read storage entry: {}", e)))?;
        if !entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
            continue;
        }
        stats.session_count += 1;
        let session_dir = entry.path();

        let chunk_path = session_dir.join("chunks.parquet");
        if chunk_path.exists() {
            let file = File::open(&chunk_path).map_err(|e| {
                LetheError::internal(format!(
                    "Failed to open chunk parquet {}: {}",
                    chunk_path.display(),
                    e
                ))
            })?;
            let reader = SerializedFileReader::new(file).map_err(|e| {
                LetheError::internal(format!("Failed to read chunk parquet: {}", e))
            })?;
            stats.chunk_count += reader.metadata().file_metadata().num_rows() as usize;
        }

        let messages_path = session_dir.join("messages.parquet");
        if messages_path.exists() {
            let file = File::open(&messages_path).map_err(|e| {
                LetheError::internal(format!(
                    "Failed to open message parquet {}: {}",
                    messages_path.display(),
                    e
                ))
            })?;
            let reader = SerializedFileReader::new(file).map_err(|e| {
                LetheError::internal(format!("Failed to read message parquet: {}", e))
            })?;
            stats.message_count += reader.metadata().file_metadata().num_rows() as usize;
        }

        let embedding_path = session_dir.join("embeddings").join("vectors.json");
        if embedding_path.exists() {
            let data = std::fs::read_to_string(&embedding_path).map_err(|e| {
                LetheError::internal(format!(
                    "Failed to read embeddings {}: {}",
                    embedding_path.display(),
                    e
                ))
            })?;
            let entries: Vec<StoredEmbedding> = serde_json::from_str(&data).map_err(|e| {
                LetheError::internal(format!(
                    "Failed to deserialize embeddings {}: {}",
                    embedding_path.display(),
                    e
                ))
            })?;
            stats.embedding_count += entries.len();
        }
    }

    Ok(stats)
}

#[async_trait]
impl DocumentRepository for ParquetCorpus {
    async fn get_chunks_by_session(&self, session_id: &str) -> Result<Vec<Chunk>> {
        let (chunks, _) = self.load_session(session_id).await?;
        self.load_bloom(session_id).await?;
        Ok(chunks)
    }

    async fn get_dfidf_by_session(&self, session_id: &str) -> Result<Vec<DfIdf>> {
        let (_, dfidf) = self.load_session(session_id).await?;
        Ok(dfidf)
    }

    async fn get_chunk_by_id(&self, chunk_id: &str) -> Result<Option<Chunk>> {
        let cache = self.caches.read().await;
        Ok(cache.chunk_map.get(chunk_id).cloned())
    }

    async fn vector_search(
        &self,
        query_vector: &EmbeddingVector,
        k: i32,
    ) -> Result<Vec<Candidate>> {
        let embeddings = self.load_embeddings().await?;
        if embeddings.is_empty() {
            return Ok(Vec::new());
        }

        let chunk_snapshot = {
            let cache = self.caches.read().await;
            cache.chunk_map.clone()
        };

        let mut scored = Vec::new();
        for (chunk_id, vector) in embeddings {
            if vector.data.len() != query_vector.data.len() {
                continue;
            }
            let score = Self::cosine_similarity(&query_vector.data, &vector.data);
            if let Some(chunk) = chunk_snapshot.get(&chunk_id) {
                scored.push(Candidate {
                    doc_id: chunk.id.clone(),
                    score,
                    text: Some(chunk.text.clone()),
                    kind: Some(chunk.kind.clone()),
                });
            }
        }

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k.max(0) as usize);
        Ok(scored)
    }

    async fn prefilter_chunks(
        &self,
        session_id: &str,
        terms: &[String],
    ) -> Result<Option<Vec<Chunk>>> {
        if terms.is_empty() {
            return Ok(None);
        }

        self.load_bloom(session_id).await?;
        let (chunks, _) = self.load_session(session_id).await?;
        if chunks.is_empty() {
            return Ok(Some(Vec::new()));
        }

        let term_refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();

        let cache = self.caches.read().await;
        let filtered = chunks
            .into_iter()
            .filter(|chunk| {
                let key = format!("{}/{}", session_id, chunk.id);
                cache
                    .bloom
                    .get(&key)
                    .map(|filter| filter.contains_any(term_refs.iter().copied()))
                    .unwrap_or(true)
            })
            .collect::<Vec<_>>();
        drop(cache);

        Ok(Some(filtered))
    }
}
