use async_trait::async_trait;
use lethe_shared::{LetheError, Message, Result};
use sqlx::{postgres::PgRow, PgPool, Row};
use uuid::Uuid;

/// Repository trait for message operations
#[async_trait]
pub trait MessageRepository: Send + Sync {
    async fn create_message(&self, message: &Message) -> Result<Message>;
    async fn get_message(&self, id: &Uuid) -> Result<Option<Message>>;
    async fn get_messages_by_session(
        &self,
        session_id: &str,
        limit: Option<i32>,
    ) -> Result<Vec<Message>>;
    async fn update_message(&self, message: &Message) -> Result<Message>;
    async fn delete_message(&self, id: &Uuid) -> Result<bool>;
    async fn get_recent_messages(&self, session_id: &str, count: i32) -> Result<Vec<Message>>;
}

/// PostgreSQL implementation of MessageRepository
pub struct PgMessageRepository {
    pool: PgPool,
}

impl PgMessageRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

fn map_message(row: PgRow) -> Message {
    Message {
        id: row.get("id"),
        session_id: row.get("session_id"),
        turn: row.get("turn"),
        role: row.get("role"),
        text: row.get("text"),
        ts: row.get("ts"),
        meta: row.get("meta"),
    }
}

#[async_trait]
impl MessageRepository for PgMessageRepository {
    async fn create_message(&self, message: &Message) -> Result<Message> {
        let row = sqlx::query(
            r#"
            INSERT INTO messages (id, session_id, turn, role, text, ts, meta)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id, session_id, turn, role, text, ts, meta
            "#,
        )
        .bind(message.id)
        .bind(&message.session_id)
        .bind(message.turn)
        .bind(&message.role)
        .bind(&message.text)
        .bind(message.ts)
        .bind(&message.meta)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to create message: {}", e)))?;

        Ok(map_message(row))
    }

    async fn get_message(&self, id: &Uuid) -> Result<Option<Message>> {
        let row = sqlx::query(
            "SELECT id, session_id, turn, role, text, ts, meta FROM messages WHERE id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to get message: {}", e)))?;

        Ok(row.map(map_message))
    }

    async fn get_messages_by_session(
        &self,
        session_id: &str,
        limit: Option<i32>,
    ) -> Result<Vec<Message>> {
        let limit = limit.unwrap_or(1000) as i64;

        let rows = sqlx::query(
            r#"
            SELECT id, session_id, turn, role, text, ts, meta
            FROM messages
            WHERE session_id = $1
            ORDER BY turn ASC, ts ASC
            LIMIT $2
            "#,
        )
        .bind(session_id)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to get messages by session: {}", e)))?;

        Ok(rows.into_iter().map(map_message).collect())
    }

    async fn update_message(&self, message: &Message) -> Result<Message> {
        let row = sqlx::query(
            r#"
            UPDATE messages
            SET session_id = $2, turn = $3, role = $4, text = $5, ts = $6, meta = $7
            WHERE id = $1
            RETURNING id, session_id, turn, role, text, ts, meta
            "#,
        )
        .bind(message.id)
        .bind(&message.session_id)
        .bind(message.turn)
        .bind(&message.role)
        .bind(&message.text)
        .bind(message.ts)
        .bind(&message.meta)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to update message: {}", e)))?;

        Ok(map_message(row))
    }

    async fn delete_message(&self, id: &Uuid) -> Result<bool> {
        let result = sqlx::query("DELETE FROM messages WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(|e| LetheError::database(format!("Failed to delete message: {}", e)))?;

        Ok(result.rows_affected() > 0)
    }

    async fn get_recent_messages(&self, session_id: &str, count: i32) -> Result<Vec<Message>> {
        let rows = sqlx::query(
            r#"
            SELECT id, session_id, turn, role, text, ts, meta
            FROM messages
            WHERE session_id = $1
            ORDER BY ts DESC
            LIMIT $2
            "#,
        )
        .bind(session_id)
        .bind(count as i64)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to get recent messages: {}", e)))?;

        Ok(rows.into_iter().rev().map(map_message).collect())
    }
}

/// Create a batch of messages in a single transaction
pub async fn batch_create_messages(
    repository: &dyn MessageRepository,
    messages: &[Message],
) -> Result<Vec<Message>> {
    // Note: This is a simplified version. In a real implementation,
    // you might want to use a transaction and batch insert
    let mut created_messages = Vec::new();

    for message in messages {
        let created = repository.create_message(message).await?;
        created_messages.push(created);
    }

    Ok(created_messages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    fn create_test_message() -> Message {
        Message {
            id: Uuid::new_v4(),
            session_id: "test-session".to_string(),
            turn: 1,
            role: "user".to_string(),
            text: "Hello, world!".to_string(),
            ts: Utc::now(),
            meta: Some(serde_json::json!({"test": true})),
        }
    }

    // Note: These tests would require a test database setup
    // They are included to show the intended test structure

    #[tokio::test]
    #[ignore] // Ignore by default as it requires database setup
    async fn test_create_and_get_message() {
        // This test would require setting up a test database
        // let pool = setup_test_database().await;
        // let repo = PgMessageRepository::new(pool);
        // let message = create_test_message();
        //
        // let created = repo.create_message(&message).await.unwrap();
        // assert_eq!(created.text, message.text);
        //
        // let retrieved = repo.get_message(&created.id).await.unwrap();
        // assert!(retrieved.is_some());
        // assert_eq!(retrieved.unwrap().text, message.text);
    }

    #[tokio::test]
    #[ignore]
    async fn test_get_messages_by_session() {
        // let pool = setup_test_database().await;
        // let repo = PgMessageRepository::new(pool);
        //
        // // Create multiple messages for the same session
        // let mut messages = Vec::new();
        // for i in 1..=3 {
        //     let mut message = create_test_message();
        //     message.turn = i;
        //     message.text = format!("Message {}", i);
        //     messages.push(repo.create_message(&message).await.unwrap());
        // }
        //
        // let retrieved = repo.get_messages_by_session("test-session", None).await.unwrap();
        // assert_eq!(retrieved.len(), 3);
        // assert_eq!(retrieved[0].turn, 1);
        // assert_eq!(retrieved[2].turn, 3);
    }
}
