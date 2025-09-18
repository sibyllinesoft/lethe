use async_trait::async_trait;
use lethe_domain::DocumentRepository;
use lethe_shared::{Candidate, Chunk, DfIdf, EmbeddingVector, LetheError, Result};
use sqlx::{postgres::PgRow, PgPool, Row};
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

fn map_chunk(row: PgRow) -> Chunk {
    Chunk {
        id: row.get("id"),
        message_id: row.get("message_id"),
        session_id: row.get("session_id"),
        offset_start: row.get::<i32, _>("offset_start") as usize,
        offset_end: row.get::<i32, _>("offset_end") as usize,
        kind: row.get("kind"),
        text: row.get("text"),
        tokens: row.get("tokens"),
    }
}

#[async_trait]
impl ChunkRepository for PgChunkRepository {
    async fn create_chunk(&self, chunk: &Chunk) -> Result<Chunk> {
        let row = sqlx::query(
            r#"
            INSERT INTO chunks (id, message_id, session_id, offset_start, offset_end, kind, text, tokens)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id, message_id, session_id, offset_start, offset_end, kind, text, tokens
            "#
        )
        .bind(&chunk.id)
        .bind(&chunk.message_id)
        .bind(&chunk.session_id)
        .bind(chunk.offset_start as i32)
        .bind(chunk.offset_end as i32)
        .bind(&chunk.kind)
        .bind(&chunk.text)
        .bind(&chunk.tokens)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to create chunk: {}", e)))?;

        Ok(map_chunk(row))
    }

    async fn get_chunk(&self, id: &str) -> Result<Option<Chunk>> {
        let row = sqlx::query(
            r#"
            SELECT id, message_id, session_id, offset_start, offset_end, kind, text, tokens
            FROM chunks
            WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to get chunk: {}", e)))?;

        Ok(row.map(map_chunk))
    }

    async fn get_chunks_by_session(&self, session_id: &str) -> Result<Vec<Chunk>> {
        let rows = sqlx::query(
            r#"
            SELECT id, message_id, session_id, offset_start, offset_end, kind, text, tokens
            FROM chunks
            WHERE session_id = $1
            ORDER BY message_id, offset_start
            "#,
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to get chunks by session: {}", e)))?;

        Ok(rows.into_iter().map(map_chunk).collect())
    }

    async fn get_chunks_by_message(&self, message_id: &Uuid) -> Result<Vec<Chunk>> {
        let rows = sqlx::query(
            r#"
            SELECT id, message_id, session_id, offset_start, offset_end, kind, text, tokens
            FROM chunks
            WHERE message_id = $1
            ORDER BY offset_start
            "#,
        )
        .bind(message_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to get chunks by message: {}", e)))?;

        Ok(rows.into_iter().map(map_chunk).collect())
    }

    async fn delete_chunk(&self, id: &str) -> Result<bool> {
        let result = sqlx::query("DELETE FROM chunks WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(|e| LetheError::database(format!("Failed to delete chunk: {}", e)))?;

        Ok(result.rows_affected() > 0)
    }

    async fn batch_create_chunks(&self, chunks: &[Chunk]) -> Result<Vec<Chunk>> {
        let mut created_chunks = Vec::new();

        // Use a transaction for batch insertion
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| LetheError::database(format!("Failed to begin transaction: {}", e)))?;

        for chunk in chunks {
            let row = sqlx::query(
                r#"
                INSERT INTO chunks (id, message_id, session_id, offset_start, offset_end, kind, text, tokens)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id, message_id, session_id, offset_start, offset_end, kind, text, tokens
                "#
            )
            .bind(&chunk.id)
            .bind(&chunk.message_id)
            .bind(&chunk.session_id)
            .bind(chunk.offset_start as i32)
            .bind(chunk.offset_end as i32)
            .bind(&chunk.kind)
            .bind(&chunk.text)
            .bind(&chunk.tokens)
            .fetch_one(&mut *tx)
            .await
            .map_err(|e| LetheError::database(format!("Failed to create chunk in batch: {}", e)))?;

            created_chunks.push(map_chunk(row));
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
        let rows = sqlx::query(
            r#"
            SELECT term, session_id, df, idf
            FROM dfidf
            WHERE session_id = $1
            "#,
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to get dfidf by session: {}", e)))?;

        Ok(rows
            .into_iter()
            .map(|row| DfIdf {
                term: row.get("term"),
                session_id: row.get("session_id"),
                df: row.get("df"),
                idf: row.get("idf"),
            })
            .collect())
    }

    async fn get_chunk_by_id(&self, chunk_id: &str) -> Result<Option<Chunk>> {
        self.get_chunk(chunk_id).await
    }

    async fn vector_search(
        &self,
        _query_vector: &EmbeddingVector,
        k: i32,
    ) -> Result<Vec<Candidate>> {
        // This is a simplified implementation
        // In practice, you would use pgvector or similar for efficient vector search
        let rows = sqlx::query(
            r#"
            SELECT c.id, c.kind, c.text, 0.5 as score
            FROM chunks c
            INNER JOIN embeddings e ON c.id = e.chunk_id
            ORDER BY RANDOM()
            LIMIT $1
            "#,
        )
        .bind(k as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to perform vector search: {}", e)))?;

        Ok(rows
            .into_iter()
            .map(|row| Candidate {
                doc_id: row.get("id"),
                score: row.get::<Option<f64>, _>("score").unwrap_or(0.0),
                text: Some(row.get("text")),
                kind: Some(row.get("kind")),
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
