use lethe_shared::{LetheError, Result};
use sqlx::{PgPool, Postgres, Row};
use std::time::Duration;

/// Database connection manager
pub struct DatabaseManager {
    pool: PgPool,
}

impl DatabaseManager {
    /// Create a new database manager with connection pool
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = sqlx::postgres::PgPoolOptions::new()
            .max_connections(20)
            .min_connections(5)
            .max_lifetime(Duration::from_secs(30 * 60)) // 30 minutes
            .idle_timeout(Duration::from_secs(10 * 60)) // 10 minutes
            .acquire_timeout(Duration::from_secs(30)) // 30 seconds
            .connect(database_url)
            .await
            .map_err(|e| LetheError::database(format!("Failed to connect to database: {}", e)))?;

        // Run migrations
        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .map_err(|e| LetheError::database(format!("Failed to run migrations: {}", e)))?;

        tracing::info!("Database connection pool established");

        Ok(Self { pool })
    }

    /// Get a reference to the connection pool
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Test database connectivity
    pub async fn health_check(&self) -> Result<()> {
        sqlx::query("SELECT 1")
            .fetch_one(&self.pool)
            .await
            .map_err(|e| LetheError::database(format!("Health check failed: {}", e)))?;

        Ok(())
    }

    /// Get database statistics
    pub async fn get_stats(&self) -> Result<DatabaseStats> {
        let row = sqlx::query(
            r#"
            SELECT
                (SELECT COUNT(*) FROM messages) as message_count,
                (SELECT COUNT(*) FROM chunks) as chunk_count,
                (SELECT COUNT(*) FROM embeddings) as embedding_count,
                (SELECT COUNT(DISTINCT session_id) FROM messages) as session_count
            "#,
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|e| LetheError::database(format!("Failed to get stats: {}", e)))?;

        Ok(DatabaseStats {
            message_count: row.get::<Option<i64>, _>("message_count").unwrap_or(0),
            chunk_count: row.get::<Option<i64>, _>("chunk_count").unwrap_or(0),
            embedding_count: row.get::<Option<i64>, _>("embedding_count").unwrap_or(0),
            session_count: row.get::<Option<i64>, _>("session_count").unwrap_or(0),
        })
    }

    /// Begin a database transaction
    pub async fn begin_transaction(&self) -> Result<sqlx::Transaction<'_, Postgres>> {
        self.pool
            .begin()
            .await
            .map_err(|e| LetheError::database(format!("Failed to begin transaction: {}", e)))
    }

    /// Close the connection pool
    pub async fn close(&self) {
        self.pool.close().await;
        tracing::info!("Database connection pool closed");
    }
}

/// Database statistics
#[derive(Debug, Clone)]
pub struct DatabaseStats {
    pub message_count: i64,
    pub chunk_count: i64,
    pub embedding_count: i64,
    pub session_count: i64,
}

/// Database configuration
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: String,
    pub database: String,
    pub max_connections: u32,
    pub min_connections: u32,
    pub connection_timeout_secs: u64,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 5432,
            username: "lethe".to_string(),
            password: "lethe".to_string(),
            database: "lethe".to_string(),
            max_connections: 20,
            min_connections: 5,
            connection_timeout_secs: 30,
        }
    }
}

impl DatabaseConfig {
    /// Build connection URL from configuration
    pub fn connection_url(&self) -> String {
        format!(
            "postgresql://{}:{}@{}:{}/{}",
            self.username, self.password, self.host, self.port, self.database
        )
    }

    /// Create database manager from configuration
    pub async fn create_manager(&self) -> Result<DatabaseManager> {
        DatabaseManager::new(&self.connection_url()).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_config_url() {
        let config = DatabaseConfig::default();
        let url = config.connection_url();
        assert!(url.starts_with("postgresql://"));
        assert!(url.contains("localhost:5432"));
    }

    #[test]
    fn test_database_config_custom() {
        let config = DatabaseConfig {
            host: "db.example.com".to_string(),
            port: 5433,
            username: "user".to_string(),
            password: "pass".to_string(),
            database: "mydb".to_string(),
            ..Default::default()
        };

        let url = config.connection_url();
        assert_eq!(url, "postgresql://user:pass@db.example.com:5433/mydb");
    }
}
