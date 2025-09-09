use async_trait::async_trait;
use lethe_domain::DocumentRepository;
use lethe_shared::{Chunk, DfIdf, Candidate, Result, LetheError, EmbeddingVector};
use sqlx::PgPool;
use uuid::Uuid;

/// Repository trait for chunk operations
#[async_trait]
pub trait ChunkRepository: Send + Sync {
    async fn create_chunk(&self, chunk: &Chunk) -> Result<Chunk>;
    async fn get_chunk(&self, id: &str) -> Result<Option<Chunk>>;
    async fn get_chunks_by_session(&self, session_id: &str) -> Result<Vec<Chunk>>;
    async fn get_chunks_by_message(&self, message_id: &Uuid) -> Result<Vec<Chunk>>;
    async fn delete_chunk(&self, id: &str) -> Result<bool>;
    async fn batch_create_chunks(&self, chunks: &[Chunk]) -> Result<Vec<Chunk>>;
}

/// PostgreSQL implementation of ChunkRepository
pub struct PgChunkRepository {
    pool: PgPool,
}

impl PgChunkRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl ChunkRepository for PgChunkRepository {
    async fn create_chunk(&self, chunk: &Chunk) -> Result<Chunk> {
        let row = sqlx::query!(
            r#"
            INSERT INTO chunks (id, message_id, session_id, offset_start, offset_end, kind, text, tokens)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id, message_id, session_id, offset_start, offset_end, kind, text, tokens
            "#,
            chunk.id,
            chunk.message_id,
            chunk.session_id,
            chunk.offset_start as i32,
            chunk.offset_end as i32,
            chunk.kind,
            chunk.text,
            chunk.tokens
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to create chunk: {}", e)))?;

        Ok(Chunk {
            id: row.id,
            message_id: row.message_id,
            session_id: row.session_id,
            offset_start: row.offset_start as usize,
            offset_end: row.offset_end as usize,
            kind: row.kind,
            text: row.text,
            tokens: row.tokens,
        })
    }

    async fn get_chunk(&self, id: &str) -> Result<Option<Chunk>> {
        let row = sqlx::query!(
            r#"
            SELECT id, message_id, session_id, offset_start, offset_end, kind, text, tokens 
            FROM chunks 
            WHERE id = $1
            "#,
            id
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to get chunk: {}", e)))?;

        Ok(row.map(|r| Chunk {
            id: r.id,
            message_id: r.message_id,
            session_id: r.session_id,
            offset_start: r.offset_start as usize,
            offset_end: r.offset_end as usize,
            kind: r.kind,
            text: r.text,
            tokens: r.tokens,
        }))
    }

    async fn get_chunks_by_session(&self, session_id: &str) -> Result<Vec<Chunk>> {
        let rows = sqlx::query!(
            r#"
            SELECT id, message_id, session_id, offset_start, offset_end, kind, text, tokens 
            FROM chunks 
            WHERE session_id = $1
            ORDER BY message_id, offset_start
            "#,
            session_id
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to get chunks by session: {}", e)))?;

        Ok(rows
            .into_iter()
            .map(|r| Chunk {
                id: r.id,
                message_id: r.message_id,
                session_id: r.session_id,
                offset_start: r.offset_start as usize,
                offset_end: r.offset_end as usize,
                kind: r.kind,
                text: r.text,
                tokens: r.tokens,
            })
            .collect())
    }

    async fn get_chunks_by_message(&self, message_id: &Uuid) -> Result<Vec<Chunk>> {
        let rows = sqlx::query!(
            r#"
            SELECT id, message_id, session_id, offset_start, offset_end, kind, text, tokens 
            FROM chunks 
            WHERE message_id = $1
            ORDER BY offset_start
            "#,
            message_id
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to get chunks by message: {}", e)))?;

        Ok(rows
            .into_iter()
            .map(|r| Chunk {
                id: r.id,
                message_id: r.message_id,
                session_id: r.session_id,
                offset_start: r.offset_start as usize,
                offset_end: r.offset_end as usize,
                kind: r.kind,
                text: r.text,
                tokens: r.tokens,
            })
            .collect())
    }

    async fn delete_chunk(&self, id: &str) -> Result<bool> {
        let result = sqlx::query!("DELETE FROM chunks WHERE id = $1", id)
            .execute(&self.pool)
            .await
            .map_err(|e| LetheError::database(format!("Failed to delete chunk: {}", e)))?;

        Ok(result.rows_affected() > 0)
    }

    async fn batch_create_chunks(&self, chunks: &[Chunk]) -> Result<Vec<Chunk>> {
        let mut created_chunks = Vec::new();
        
        // Use a transaction for batch insertion
        let mut tx = self.pool
            .begin()
            .await
            .map_err(|e| LetheError::database(format!("Failed to begin transaction: {}", e)))?;

        for chunk in chunks {
            let row = sqlx::query!(
                r#"
                INSERT INTO chunks (id, message_id, session_id, offset_start, offset_end, kind, text, tokens)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id, message_id, session_id, offset_start, offset_end, kind, text, tokens
                "#,
                chunk.id,
                chunk.message_id,
                chunk.session_id,
                chunk.offset_start as i32,
                chunk.offset_end as i32,
                chunk.kind,
                chunk.text,
                chunk.tokens
            )
            .fetch_one(&mut *tx)
            .await
            .map_err(|e| LetheError::database(format!("Failed to create chunk in batch: {}", e)))?;

            created_chunks.push(Chunk {
                id: row.id,
                message_id: row.message_id,
                session_id: row.session_id,
                offset_start: row.offset_start as usize,
                offset_end: row.offset_end as usize,
                kind: row.kind,
                text: row.text,
                tokens: row.tokens,
            });
        }

        tx.commit()
            .await
            .map_err(|e| LetheError::database(format!("Failed to commit transaction: {}", e)))?;

        Ok(created_chunks)
    }
}

/// Implementation of DocumentRepository trait for PgChunkRepository
#[async_trait]
impl DocumentRepository for PgChunkRepository {
    async fn get_chunks_by_session(&self, session_id: &str) -> Result<Vec<Chunk>> {
        ChunkRepository::get_chunks_by_session(self, session_id).await
    }

    async fn get_dfidf_by_session(&self, session_id: &str) -> Result<Vec<DfIdf>> {
        let rows = sqlx::query!(
            r#"
            SELECT term, session_id, df, idf 
            FROM dfidf 
            WHERE session_id = $1
            "#,
            session_id
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to get dfidf by session: {}", e)))?;

        Ok(rows
            .into_iter()
            .map(|r| DfIdf {
                term: r.term,
                session_id: r.session_id,
                df: r.df,
                idf: r.idf,
            })
            .collect())
    }

    async fn get_chunk_by_id(&self, chunk_id: &str) -> Result<Option<Chunk>> {
        self.get_chunk(chunk_id).await
    }

    async fn vector_search(&self, query_vector: &EmbeddingVector, k: i32) -> Result<Vec<Candidate>> {
        // This is a simplified implementation
        // In practice, you would use pgvector or similar for efficient vector search
        let rows = sqlx::query!(
            r#"
            SELECT c.id, c.kind, c.text, 0.5 as score
            FROM chunks c
            INNER JOIN embeddings e ON c.id = e.chunk_id
            ORDER BY RANDOM()
            LIMIT $1
            "#,
            k as i64
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to perform vector search: {}", e)))?;

        Ok(rows
            .into_iter()
            .map(|r| Candidate {
                doc_id: r.id,
                score: r.score.unwrap_or(0.0),
                text: Some(r.text),
                kind: Some(r.kind),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn create_test_chunk() -> Chunk {
        Chunk {
            id: "test-chunk-1".to_string(),
            message_id: Uuid::new_v4(),
            session_id: "test-session".to_string(),
            offset_start: 0,
            offset_end: 100,
            kind: "text".to_string(),
            text: "This is a test chunk".to_string(),
            tokens: 5,
        }
    }

    #[tokio::test]
    #[ignore] // Requires database setup
    async fn test_create_and_get_chunk() {
        // Test implementation would require database setup
        // let pool = setup_test_database().await;
        // let repo = PgChunkRepository::new(pool);
        // let chunk = create_test_chunk();
        // 
        // let created = repo.create_chunk(&chunk).await.unwrap();
        // assert_eq!(created.text, chunk.text);
        // 
        // let retrieved = repo.get_chunk(&created.id).await.unwrap();
        // assert!(retrieved.is_some());
        // assert_eq!(retrieved.unwrap().text, chunk.text);
    }

    #[test]
    fn test_chunk_serialization() {
        let chunk = create_test_chunk();
        let json = serde_json::to_string(&chunk).unwrap();
        let deserialized: Chunk = serde_json::from_str(&json).unwrap();
        
        assert_eq!(chunk.id, deserialized.id);
        assert_eq!(chunk.text, deserialized.text);
    }
}