use async_trait::async_trait;
use futures::TryStreamExt;
use lethe_shared::{EmbeddingVector, LetheError, Result};
use sqlx::{PgPool, Row};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{OnceCell, RwLock};
use tracing::{debug, warn};
use uuid::Uuid;

/// Repository trait for embedding operations
#[async_trait]
pub trait EmbeddingRepository: Send + Sync {
    async fn create_embedding(&self, chunk_id: &str, embedding: &EmbeddingVector) -> Result<()>;
    async fn get_embedding(&self, chunk_id: &str) -> Result<Option<EmbeddingVector>>;
    async fn get_embeddings_by_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<(String, EmbeddingVector)>>;
    async fn delete_embedding(&self, chunk_id: &str) -> Result<bool>;
    async fn batch_create_embeddings(&self, embeddings: &[(String, EmbeddingVector)])
        -> Result<()>;
    async fn search_similar_embeddings(
        &self,
        query_embedding: &EmbeddingVector,
        limit: i32,
    ) -> Result<Vec<(String, f32)>>;
}

/// PostgreSQL implementation of EmbeddingRepository
pub struct PgEmbeddingRepository {
    pool: PgPool,
    vector_index: Arc<InMemoryEmbeddingIndex>,
}

impl PgEmbeddingRepository {
    pub fn new(pool: PgPool, vector_dimension: usize) -> Self {
        let vector_index = Arc::new(InMemoryEmbeddingIndex::new(vector_dimension));

        Self { pool, vector_index }
    }
}

#[async_trait]
impl EmbeddingRepository for PgEmbeddingRepository {
    async fn create_embedding(&self, chunk_id: &str, embedding: &EmbeddingVector) -> Result<()> {
        // Convert embedding vector to bytes for storage
        let embedding_bytes = bincode::serialize(embedding)
            .map_err(|e| LetheError::internal(format!("Failed to serialize embedding: {}", e)))?;

        sqlx::query(
            r#"
            INSERT INTO embeddings (chunk_id, embedding, dimension)
            VALUES ($1, $2, $3)
            ON CONFLICT (chunk_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                dimension = EXCLUDED.dimension,
                updated_at = NOW()
            "#,
        )
        .bind(chunk_id)
        .bind(&embedding_bytes)
        .bind(embedding.dimension as i32)
        .execute(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to create embedding: {}", e)))?;

        self.vector_index
            .ensure_bootstrapped(Some(&self.pool))
            .await?;
        self.vector_index
            .upsert_embedding(chunk_id, embedding)
            .await?;

        Ok(())
    }

    async fn get_embedding(&self, chunk_id: &str) -> Result<Option<EmbeddingVector>> {
        let row = sqlx::query("SELECT embedding FROM embeddings WHERE chunk_id = $1")
            .bind(chunk_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| LetheError::database(format!("Failed to get embedding: {}", e)))?;

        match row {
            Some(row) => {
                let bytes: Vec<u8> = row.get("embedding");
                let embedding: EmbeddingVector = bincode::deserialize(&bytes).map_err(|e| {
                    LetheError::internal(format!("Failed to deserialize embedding: {}", e))
                })?;
                Ok(Some(embedding))
            }
            None => Ok(None),
        }
    }

    async fn get_embeddings_by_session(
        &self,
        session_id: &str,
    ) -> Result<Vec<(String, EmbeddingVector)>> {
        let rows = sqlx::query(
            r#"
            SELECT e.chunk_id, e.embedding
            FROM embeddings e
            INNER JOIN chunks c ON e.chunk_id = c.id
            WHERE c.session_id = $1
            "#,
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to get embeddings by session: {}", e)))?;

        let mut embeddings = Vec::new();
        for row in rows {
            let chunk_id: String = row.get("chunk_id");
            let bytes: Vec<u8> = row.get("embedding");
            let embedding: EmbeddingVector = bincode::deserialize(&bytes).map_err(|e| {
                LetheError::internal(format!("Failed to deserialize embedding: {}", e))
            })?;
            embeddings.push((chunk_id, embedding));
        }

        Ok(embeddings)
    }

    async fn delete_embedding(&self, chunk_id: &str) -> Result<bool> {
        let result = sqlx::query("DELETE FROM embeddings WHERE chunk_id = $1")
            .bind(chunk_id)
            .execute(&self.pool)
            .await
            .map_err(|e| LetheError::database(format!("Failed to delete embedding: {}", e)))?;
        let removed = result.rows_affected() > 0;

        if removed {
            self.vector_index
                .ensure_bootstrapped(Some(&self.pool))
                .await?;
            self.vector_index.delete_embedding(chunk_id).await?;
        }

        Ok(removed)
    }

    async fn batch_create_embeddings(
        &self,
        embeddings: &[(String, EmbeddingVector)],
    ) -> Result<()> {
        if embeddings.is_empty() {
            return Ok(());
        }

        // Use a transaction for batch insertion
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| LetheError::database(format!("Failed to begin transaction: {}", e)))?;

        for (chunk_id, embedding) in embeddings {
            let embedding_bytes = bincode::serialize(embedding).map_err(|e| {
                LetheError::internal(format!("Failed to serialize embedding: {}", e))
            })?;

            sqlx::query(
                r#"
                INSERT INTO embeddings (chunk_id, embedding, dimension)
                VALUES ($1, $2, $3)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    dimension = EXCLUDED.dimension,
                    updated_at = NOW()
                "#,
            )
            .bind(chunk_id)
            .bind(&embedding_bytes)
            .bind(embedding.dimension as i32)
            .execute(&mut *tx)
            .await
            .map_err(|e| {
                LetheError::database(format!("Failed to create embedding in batch: {}", e))
            })?;
        }

        tx.commit()
            .await
            .map_err(|e| LetheError::database(format!("Failed to commit transaction: {}", e)))?;

        self.vector_index
            .ensure_bootstrapped(Some(&self.pool))
            .await?;
        self.vector_index.upsert_embeddings(embeddings).await?;

        Ok(())
    }

    async fn search_similar_embeddings(
        &self,
        query_embedding: &EmbeddingVector,
        limit: i32,
    ) -> Result<Vec<(String, f32)>> {
        if limit <= 0 {
            return Ok(Vec::new());
        }

        self.vector_index
            .ensure_bootstrapped(Some(&self.pool))
            .await?;

        self.vector_index
            .search(query_embedding, limit as usize)
            .await
    }
}

struct InMemoryEmbeddingIndex {
    dimension: usize,
    store: RwLock<EmbeddingStore>,
    bootstrap: OnceCell<()>,
}

impl InMemoryEmbeddingIndex {
    fn new(dimension: usize) -> Self {
        Self {
            dimension,
            store: RwLock::new(EmbeddingStore::new()),
            bootstrap: OnceCell::new(),
        }
    }

    async fn ensure_bootstrapped(&self, pool: Option<&PgPool>) -> Result<()> {
        if self.bootstrap.get().is_some() {
            return Ok(());
        }

        self.bootstrap
            .get_or_try_init(|| async {
                if let Some(pool) = pool {
                    self.load_from_database(pool).await?;
                }
                Ok(())
            })
            .await
            .map(|_| ())
    }

    async fn load_from_database(&self, pool: &PgPool) -> Result<()> {
        debug!("Bootstrapping in-memory embedding index from database");

        let mut rows = sqlx::query("SELECT chunk_id, embedding FROM embeddings").fetch(pool);

        let mut store = self.store.write().await;
        store.clear();

        while let Some(row) = rows
            .try_next()
            .await
            .map_err(|e| LetheError::database(format!("Failed to stream embeddings: {}", e)))?
        {
            let chunk_id: Uuid = row
                .try_get("chunk_id")
                .map_err(|e| LetheError::database(format!("Failed to decode chunk_id: {}", e)))?;
            let bytes: Vec<u8> = row.try_get("embedding").map_err(|e| {
                LetheError::database(format!("Failed to read embedding bytes: {}", e))
            })?;
            let embedding: EmbeddingVector = bincode::deserialize(&bytes).map_err(|e| {
                LetheError::internal(format!("Failed to deserialize embedding: {}", e))
            })?;

            if embedding.data.len() != self.dimension {
                warn!(
                    chunk_id = %chunk_id,
                    expected = self.dimension,
                    actual = embedding.data.len(),
                    "Skipping embedding due to dimension mismatch"
                );
                continue;
            }

            let normalized = normalize_vector(&embedding.data).0;
            store.insert_normalized(chunk_id.to_string(), normalized, self.dimension);
        }

        Ok(())
    }

    async fn upsert_embedding(&self, chunk_id: &str, embedding: &EmbeddingVector) -> Result<()> {
        if embedding.data.len() != self.dimension {
            return Err(LetheError::validation(
                "embedding.dimension",
                format!(
                    "Embedding dimension {} does not match index dimension {}",
                    embedding.data.len(),
                    self.dimension
                ),
            ));
        }

        let normalized = normalize_vector(&embedding.data).0;

        let mut store = self.store.write().await;
        store.upsert(chunk_id, normalized, self.dimension);
        Ok(())
    }

    async fn upsert_embeddings(&self, embeddings: &[(String, EmbeddingVector)]) -> Result<()> {
        let mut store = self.store.write().await;
        for (chunk_id, embedding) in embeddings {
            if embedding.data.len() != self.dimension {
                warn!(
                    chunk_id = %chunk_id,
                    expected = self.dimension,
                    actual = embedding.data.len(),
                    "Skipping embedding due to dimension mismatch"
                );
                continue;
            }

            let normalized = normalize_vector(&embedding.data).0;
            store.upsert(chunk_id, normalized, self.dimension);
        }

        Ok(())
    }

    async fn delete_embedding(&self, chunk_id: &str) -> Result<()> {
        let mut store = self.store.write().await;
        store.remove(chunk_id, self.dimension);
        Ok(())
    }

    async fn search(&self, query: &EmbeddingVector, limit: usize) -> Result<Vec<(String, f32)>> {
        if query.data.len() != self.dimension {
            return Err(LetheError::validation(
                "embedding.dimension",
                format!(
                    "Query vector dimension {} does not match index dimension {}",
                    query.data.len(),
                    self.dimension
                ),
            ));
        }

        if limit == 0 {
            return Ok(Vec::new());
        }

        let (normalized_query, query_norm) = normalize_vector(&query.data);
        if query_norm <= f32::EPSILON {
            return Ok(Vec::new());
        }

        let store = self.store.read().await;
        if store.is_empty() {
            return Ok(Vec::new());
        }

        let mut scored = Vec::with_capacity(store.len());
        for idx in 0..store.len() {
            let vector = store.vector_slice(idx, self.dimension);
            let similarity = dot_product(&normalized_query, vector);
            scored.push((store.chunk_ids[idx].clone(), similarity));
        }

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        if scored.len() > limit {
            scored.truncate(limit);
        }

        Ok(scored)
    }
}

struct EmbeddingStore {
    chunk_ids: Vec<String>,
    vectors: Vec<f32>,
    id_to_index: HashMap<String, usize>,
}

impl EmbeddingStore {
    fn new() -> Self {
        Self {
            chunk_ids: Vec::new(),
            vectors: Vec::new(),
            id_to_index: HashMap::new(),
        }
    }

    fn clear(&mut self) {
        self.chunk_ids.clear();
        self.vectors.clear();
        self.id_to_index.clear();
    }

    fn len(&self) -> usize {
        self.chunk_ids.len()
    }

    fn is_empty(&self) -> bool {
        self.chunk_ids.is_empty()
    }

    fn vector_slice(&self, index: usize, dimension: usize) -> &[f32] {
        let start = index * dimension;
        &self.vectors[start..start + dimension]
    }

    fn insert_normalized(&mut self, chunk_id: String, normalized: Vec<f32>, dimension: usize) {
        debug_assert_eq!(normalized.len(), dimension);
        let index = self.chunk_ids.len();
        self.vectors.extend_from_slice(&normalized);
        self.id_to_index.insert(chunk_id.clone(), index);
        self.chunk_ids.push(chunk_id);
        debug_assert_eq!(self.vectors.len(), self.chunk_ids.len() * dimension);
    }

    fn upsert(&mut self, chunk_id: &str, normalized: Vec<f32>, dimension: usize) {
        if let Some(&index) = self.id_to_index.get(chunk_id) {
            let start = index * dimension;
            self.vectors[start..start + dimension].copy_from_slice(&normalized);
        } else {
            self.insert_normalized(chunk_id.to_string(), normalized, dimension);
        }
    }

    fn remove(&mut self, chunk_id: &str, dimension: usize) -> bool {
        if let Some(index) = self.id_to_index.remove(chunk_id) {
            let last_index = self.chunk_ids.len() - 1;
            let last_vector_start = last_index * dimension;

            if index != last_index {
                let moved_id = self.chunk_ids[last_index].clone();
                let dest_start = index * dimension;
                self.vectors
                    .copy_within(last_vector_start..last_vector_start + dimension, dest_start);
                self.chunk_ids[index] = moved_id.clone();
                self.id_to_index.insert(moved_id, index);
            }

            self.chunk_ids.pop();
            self.vectors.truncate(last_index * dimension);
            true
        } else {
            false
        }
    }
}

fn dot_product(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).sum()
}

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|v| v * v).sum::<f32>().sqrt()
}

fn normalize_vector(data: &[f32]) -> (Vec<f32>, f32) {
    let norm = l2_norm(data);
    if norm <= f32::EPSILON {
        (vec![0.0; data.len()], norm)
    } else {
        (
            data.iter().map(|value| value / norm).collect::<Vec<f32>>(),
            norm,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn in_memory_index_returns_top_matches() {
        let index = InMemoryEmbeddingIndex::new(3);
        index.bootstrap.set(()).unwrap();

        index
            .upsert_embedding(
                "a",
                &EmbeddingVector {
                    data: vec![1.0, 0.0, 0.0],
                    dimension: 3,
                },
            )
            .await
            .unwrap();
        index
            .upsert_embedding(
                "b",
                &EmbeddingVector {
                    data: vec![0.0, 1.0, 0.0],
                    dimension: 3,
                },
            )
            .await
            .unwrap();

        let results = index
            .search(
                &EmbeddingVector {
                    data: vec![1.0, 0.0, 0.0],
                    dimension: 3,
                },
                2,
            )
            .await
            .unwrap();

        assert_eq!(results.first().unwrap().0, "a");
        assert!(results.first().unwrap().1 > results[1].1);
    }

    #[test]
    fn serialization_roundtrip() {
        let embedding = EmbeddingVector {
            data: vec![0.1, 0.2, 0.3, 0.4],
            dimension: 4,
        };
        let serialized = bincode::serialize(&embedding).unwrap();
        let deserialized: EmbeddingVector = bincode::deserialize(&serialized).unwrap();

        assert_eq!(embedding.data, deserialized.data);
        assert_eq!(embedding.dimension, deserialized.dimension);
    }
}
