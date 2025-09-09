use async_trait::async_trait;
use lethe_shared::{EmbeddingVector, Result, LetheError};
use sqlx::PgPool;

/// Repository trait for embedding operations
#[async_trait]
pub trait EmbeddingRepository: Send + Sync {
    async fn create_embedding(&self, chunk_id: &str, embedding: &EmbeddingVector) -> Result<()>;
    async fn get_embedding(&self, chunk_id: &str) -> Result<Option<EmbeddingVector>>;
    async fn get_embeddings_by_session(&self, session_id: &str) -> Result<Vec<(String, EmbeddingVector)>>;
    async fn delete_embedding(&self, chunk_id: &str) -> Result<bool>;
    async fn batch_create_embeddings(&self, embeddings: &[(String, EmbeddingVector)]) -> Result<()>;
    async fn search_similar_embeddings(&self, query_embedding: &EmbeddingVector, limit: i32) -> Result<Vec<(String, f32)>>;
}

/// PostgreSQL implementation of EmbeddingRepository
pub struct PgEmbeddingRepository {
    pool: PgPool,
}

impl PgEmbeddingRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl EmbeddingRepository for PgEmbeddingRepository {
    async fn create_embedding(&self, chunk_id: &str, embedding: &EmbeddingVector) -> Result<()> {
        // Convert embedding vector to bytes for storage
        let embedding_bytes = bincode::serialize(embedding)
            .map_err(|e| LetheError::internal(format!("Failed to serialize embedding: {}", e)))?;

        sqlx::query!(
            r#"
            INSERT INTO embeddings (chunk_id, embedding, dimension)
            VALUES ($1, $2, $3)
            ON CONFLICT (chunk_id) DO UPDATE SET 
                embedding = EXCLUDED.embedding,
                dimension = EXCLUDED.dimension,
                updated_at = NOW()
            "#,
            chunk_id,
            embedding_bytes,
            embedding.dimension as i32
        )
        .execute(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to create embedding: {}", e)))?;

        Ok(())
    }

    async fn get_embedding(&self, chunk_id: &str) -> Result<Option<EmbeddingVector>> {
        let row = sqlx::query!(
            "SELECT embedding FROM embeddings WHERE chunk_id = $1",
            chunk_id
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to get embedding: {}", e)))?;

        match row {
            Some(row) => {
                let embedding: EmbeddingVector = bincode::deserialize(&row.embedding)
                    .map_err(|e| LetheError::internal(format!("Failed to deserialize embedding: {}", e)))?;
                Ok(Some(embedding))
            }
            None => Ok(None),
        }
    }

    async fn get_embeddings_by_session(&self, session_id: &str) -> Result<Vec<(String, EmbeddingVector)>> {
        let rows = sqlx::query!(
            r#"
            SELECT e.chunk_id, e.embedding
            FROM embeddings e
            INNER JOIN chunks c ON e.chunk_id = c.id
            WHERE c.session_id = $1
            "#,
            session_id
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to get embeddings by session: {}", e)))?;

        let mut embeddings = Vec::new();
        for row in rows {
            let embedding: EmbeddingVector = bincode::deserialize(&row.embedding)
                .map_err(|e| LetheError::internal(format!("Failed to deserialize embedding: {}", e)))?;
            embeddings.push((row.chunk_id, embedding));
        }

        Ok(embeddings)
    }

    async fn delete_embedding(&self, chunk_id: &str) -> Result<bool> {
        let result = sqlx::query!("DELETE FROM embeddings WHERE chunk_id = $1", chunk_id)
            .execute(&self.pool)
            .await
            .map_err(|e| LetheError::database(format!("Failed to delete embedding: {}", e)))?;

        Ok(result.rows_affected() > 0)
    }

    async fn batch_create_embeddings(&self, embeddings: &[(String, EmbeddingVector)]) -> Result<()> {
        if embeddings.is_empty() {
            return Ok(());
        }

        // Use a transaction for batch insertion
        let mut tx = self.pool
            .begin()
            .await
            .map_err(|e| LetheError::database(format!("Failed to begin transaction: {}", e)))?;

        for (chunk_id, embedding) in embeddings {
            let embedding_bytes = bincode::serialize(embedding)
                .map_err(|e| LetheError::internal(format!("Failed to serialize embedding: {}", e)))?;

            sqlx::query!(
                r#"
                INSERT INTO embeddings (chunk_id, embedding, dimension)
                VALUES ($1, $2, $3)
                ON CONFLICT (chunk_id) DO UPDATE SET 
                    embedding = EXCLUDED.embedding,
                    dimension = EXCLUDED.dimension,
                    updated_at = NOW()
                "#,
                chunk_id,
                embedding_bytes,
                embedding.dimension as i32
            )
            .execute(&mut *tx)
            .await
            .map_err(|e| LetheError::database(format!("Failed to create embedding in batch: {}", e)))?;
        }

        tx.commit()
            .await
            .map_err(|e| LetheError::database(format!("Failed to commit transaction: {}", e)))?;

        Ok(())
    }

    async fn search_similar_embeddings(&self, query_embedding: &EmbeddingVector, limit: i32) -> Result<Vec<(String, f32)>> {
        // This is a simplified implementation using cosine similarity
        // In a production system, you would use pgvector or similar for efficient vector search
        let query_bytes = bincode::serialize(query_embedding)
            .map_err(|e| LetheError::internal(format!("Failed to serialize query embedding: {}", e)))?;

        let rows = sqlx::query!(
            r#"
            SELECT 
                chunk_id,
                embedding,
                -- Placeholder for similarity calculation
                -- In practice, use pgvector's cosine similarity
                0.5 as similarity
            FROM embeddings
            ORDER BY similarity DESC
            LIMIT $1
            "#,
            limit as i64
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to search similar embeddings: {}", e)))?;

        let mut results = Vec::new();
        for row in rows {
            // In a real implementation, this would be calculated by the database
            // using pgvector or similar vector similarity functions
            let stored_embedding: EmbeddingVector = bincode::deserialize(&row.embedding)
                .map_err(|e| LetheError::internal(format!("Failed to deserialize stored embedding: {}", e)))?;
            
            // Calculate cosine similarity
            let similarity = cosine_similarity(&query_embedding.data, &stored_embedding.data);
            results.push((row.chunk_id, similarity));
        }

        // Sort by similarity (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }
}

/// Calculate cosine similarity between two embedding vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-6);

        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    #[ignore] // Requires database setup
    async fn test_create_and_get_embedding() {
        // Test implementation would require database setup
        // let pool = setup_test_database().await;
        // let repo = PgEmbeddingRepository::new(pool);
        // let embedding = vec![0.1, 0.2, 0.3, 0.4];
        // 
        // repo.create_embedding("test-chunk-1", &embedding).await.unwrap();
        // let retrieved = repo.get_embedding("test-chunk-1").await.unwrap();
        // 
        // assert!(retrieved.is_some());
        // assert_eq!(retrieved.unwrap(), embedding);
    }

    #[test]
    fn test_embedding_serialization() {
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