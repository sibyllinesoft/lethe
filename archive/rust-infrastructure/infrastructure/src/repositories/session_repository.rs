use async_trait::async_trait;
use chrono::{DateTime, Utc};
use lethe_shared::{LetheError, Result};
use serde::{Deserialize, Serialize};
use sqlx::{postgres::PgRow, PgPool, Row};

/// Session information for tracking conversation state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: Option<serde_json::Value>,
}

/// Session state information for planning and adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    pub session_id: String,
    pub state_key: String,
    pub state_value: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Repository trait for session operations
#[async_trait]
pub trait SessionRepository: Send + Sync {
    async fn create_session(&self, session: &Session) -> Result<Session>;
    async fn get_session(&self, id: &str) -> Result<Option<Session>>;
    async fn update_session(&self, session: &Session) -> Result<Session>;
    async fn delete_session(&self, id: &str) -> Result<bool>;
    async fn list_sessions(&self, limit: Option<i32>, offset: Option<i32>) -> Result<Vec<Session>>;

    // Session state operations
    async fn set_session_state(
        &self,
        session_id: &str,
        key: &str,
        value: &serde_json::Value,
    ) -> Result<()>;
    async fn get_session_state(
        &self,
        session_id: &str,
        key: &str,
    ) -> Result<Option<serde_json::Value>>;
    async fn get_all_session_state(&self, session_id: &str) -> Result<Vec<SessionState>>;
    async fn delete_session_state(&self, session_id: &str, key: &str) -> Result<bool>;
    async fn clear_session_state(&self, session_id: &str) -> Result<()>;
}

/// PostgreSQL implementation of SessionRepository
pub struct PgSessionRepository {
    pool: PgPool,
}

impl PgSessionRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

fn map_session(row: PgRow) -> Session {
    Session {
        id: row.get("id"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
        metadata: row.get("metadata"),
    }
}

fn map_session_state(row: PgRow) -> SessionState {
    SessionState {
        session_id: row.get("session_id"),
        state_key: row.get("state_key"),
        state_value: row.get("state_value"),
        created_at: row.get("created_at"),
        updated_at: row.get("updated_at"),
    }
}

#[async_trait]
impl SessionRepository for PgSessionRepository {
    async fn create_session(&self, session: &Session) -> Result<Session> {
        let row = sqlx::query(
            r#"
            INSERT INTO sessions (id, metadata, created_at, updated_at)
            VALUES ($1, $2, $3, $4)
            RETURNING id, metadata, created_at, updated_at
            "#,
        )
        .bind(&session.id)
        .bind(&session.metadata)
        .bind(session.created_at)
        .bind(session.updated_at)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to create session: {}", e)))?;

        Ok(map_session(row))
    }

    async fn get_session(&self, id: &str) -> Result<Option<Session>> {
        let row =
            sqlx::query("SELECT id, metadata, created_at, updated_at FROM sessions WHERE id = $1")
                .bind(id)
                .fetch_optional(&self.pool)
                .await
                .map_err(|e| LetheError::database(format!("Failed to get session: {}", e)))?;

        Ok(row.map(map_session))
    }

    async fn update_session(&self, session: &Session) -> Result<Session> {
        let row = sqlx::query(
            r#"
            UPDATE sessions
            SET metadata = $2, updated_at = $3
            WHERE id = $1
            RETURNING id, metadata, created_at, updated_at
            "#,
        )
        .bind(&session.id)
        .bind(&session.metadata)
        .bind(session.updated_at)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to update session: {}", e)))?;

        Ok(map_session(row))
    }

    async fn delete_session(&self, id: &str) -> Result<bool> {
        let result = sqlx::query("DELETE FROM sessions WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await
            .map_err(|e| LetheError::database(format!("Failed to delete session: {}", e)))?;

        Ok(result.rows_affected() > 0)
    }

    async fn list_sessions(&self, limit: Option<i32>, offset: Option<i32>) -> Result<Vec<Session>> {
        let limit = limit.unwrap_or(100) as i64;
        let offset = offset.unwrap_or(0) as i64;

        let rows = sqlx::query(
            r#"
            SELECT id, metadata, created_at, updated_at
            FROM sessions
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            "#,
        )
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to list sessions: {}", e)))?;

        Ok(rows.into_iter().map(map_session).collect())
    }

    async fn set_session_state(
        &self,
        session_id: &str,
        key: &str,
        value: &serde_json::Value,
    ) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO session_state (session_id, state_key, state_value, created_at, updated_at)
            VALUES ($1, $2, $3, NOW(), NOW())
            ON CONFLICT (session_id, state_key) DO UPDATE SET
                state_value = EXCLUDED.state_value,
                updated_at = NOW()
            "#,
        )
        .bind(session_id)
        .bind(key)
        .bind(value)
        .execute(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to set session state: {}", e)))?;

        Ok(())
    }

    async fn get_session_state(
        &self,
        session_id: &str,
        key: &str,
    ) -> Result<Option<serde_json::Value>> {
        let row = sqlx::query(
            "SELECT state_value FROM session_state WHERE session_id = $1 AND state_key = $2",
        )
        .bind(session_id)
        .bind(key)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to get session state: {}", e)))?;

        Ok(row.map(|r| r.get("state_value")))
    }

    async fn get_all_session_state(&self, session_id: &str) -> Result<Vec<SessionState>> {
        let rows = sqlx::query(
            r#"
            SELECT session_id, state_key, state_value, created_at, updated_at
            FROM session_state
            WHERE session_id = $1
            ORDER BY created_at ASC
            "#,
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to get all session state: {}", e)))?;

        Ok(rows.into_iter().map(map_session_state).collect())
    }

    async fn delete_session_state(&self, session_id: &str, key: &str) -> Result<bool> {
        let result =
            sqlx::query("DELETE FROM session_state WHERE session_id = $1 AND state_key = $2")
                .bind(session_id)
                .bind(key)
                .execute(&self.pool)
                .await
                .map_err(|e| {
                    LetheError::database(format!("Failed to delete session state: {}", e))
                })?;

        Ok(result.rows_affected() > 0)
    }

    async fn clear_session_state(&self, session_id: &str) -> Result<()> {
        sqlx::query("DELETE FROM session_state WHERE session_id = $1")
            .bind(session_id)
            .execute(&self.pool)
            .await
            .map_err(|e| LetheError::database(format!("Failed to clear session state: {}", e)))?;

        Ok(())
    }
}

/// Create a new session with default values
pub fn create_new_session(id: String) -> Session {
    let now = Utc::now();
    Session {
        id,
        created_at: now,
        updated_at: now,
        metadata: None,
    }
}

/// Create a new session with metadata
pub fn create_session_with_metadata(id: String, metadata: serde_json::Value) -> Session {
    let now = Utc::now();
    Session {
        id,
        created_at: now,
        updated_at: now,
        metadata: Some(metadata),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_create_new_session() {
        let session = create_new_session("test-session-1".to_string());
        assert_eq!(session.id, "test-session-1");
        assert!(session.metadata.is_none());
    }

    #[test]
    fn test_create_session_with_metadata() {
        let metadata = json!({
            "user_id": "user123",
            "preferences": {
                "theme": "dark",
                "language": "en"
            }
        });

        let session = create_session_with_metadata("test-session-2".to_string(), metadata.clone());
        assert_eq!(session.id, "test-session-2");
        assert_eq!(session.metadata, Some(metadata));
    }

    #[tokio::test]
    #[ignore] // Requires database setup
    async fn test_create_and_get_session() {
        // Test implementation would require database setup
        // let pool = setup_test_database().await;
        // let repo = PgSessionRepository::new(pool);
        // let session = create_new_session("test-session-1".to_string());
        //
        // let created = repo.create_session(&session).await.unwrap();
        // assert_eq!(created.id, session.id);
        //
        // let retrieved = repo.get_session(&created.id).await.unwrap();
        // assert!(retrieved.is_some());
        // assert_eq!(retrieved.unwrap().id, session.id);
    }

    #[tokio::test]
    #[ignore] // Requires database setup
    async fn test_session_state_operations() {
        // Test implementation would require database setup
        // let pool = setup_test_database().await;
        // let repo = PgSessionRepository::new(pool);
        // let session_id = "test-session-1";
        // let key = "user_preferences";
        // let value = json!({"theme": "dark"});
        //
        // // Set state
        // repo.set_session_state(session_id, key, &value).await.unwrap();
        //
        // // Get state
        // let retrieved = repo.get_session_state(session_id, key).await.unwrap();
        // assert_eq!(retrieved, Some(value.clone()));
        //
        // // Delete state
        // let deleted = repo.delete_session_state(session_id, key).await.unwrap();
        // assert!(deleted);
        //
        // // Verify deleted
        // let retrieved = repo.get_session_state(session_id, key).await.unwrap();
        // assert!(retrieved.is_none());
    }

    #[test]
    fn test_session_serialization() {
        let session = create_new_session("test".to_string());
        let json = serde_json::to_string(&session).unwrap();
        let deserialized: Session = serde_json::from_str(&json).unwrap();

        assert_eq!(session.id, deserialized.id);
    }
}
