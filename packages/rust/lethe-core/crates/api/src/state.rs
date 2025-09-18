use lethe_domain::{EmbeddingService, EnhancedQueryPipeline, LlmService, RerankingService};
use lethe_infrastructure::{
    ChunkRepository, DatabaseManager, EmbeddingRepository, MessageRepository, SessionRepository,
};
use lethe_shared::LetheConfig;
use std::sync::Arc;

/// Application state containing all services and repositories
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<LetheConfig>,
    pub db_manager: Arc<DatabaseManager>,
    pub message_repository: Arc<dyn MessageRepository>,
    pub chunk_repository: Arc<dyn ChunkRepository>,
    pub embedding_repository: Arc<dyn EmbeddingRepository>,
    pub session_repository: Arc<dyn SessionRepository>,
    pub embedding_service: Arc<dyn EmbeddingService>,
    pub llm_service: Option<Arc<dyn LlmService>>,
    pub reranking_service: Option<Arc<dyn RerankingService>>,
    pub query_pipeline: Arc<EnhancedQueryPipeline>,
}

impl AppState {
    /// Create a new AppState instance
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: Arc<LetheConfig>,
        db_manager: Arc<DatabaseManager>,
        message_repository: Arc<dyn MessageRepository>,
        chunk_repository: Arc<dyn ChunkRepository>,
        embedding_repository: Arc<dyn EmbeddingRepository>,
        session_repository: Arc<dyn SessionRepository>,
        embedding_service: Arc<dyn EmbeddingService>,
        llm_service: Option<Arc<dyn LlmService>>,
        reranking_service: Option<Arc<dyn RerankingService>>,
        query_pipeline: Arc<EnhancedQueryPipeline>,
    ) -> Self {
        Self {
            config,
            db_manager,
            message_repository,
            chunk_repository,
            embedding_repository,
            session_repository,
            embedding_service,
            llm_service,
            reranking_service,
            query_pipeline,
        }
    }

    /// Health check for the application state
    pub async fn health_check(&self) -> crate::error::ApiResult<HealthStatus> {
        let db_healthy = self.db_manager.health_check().await.is_ok();

        // Lightweight embedding check to ensure the service responds. We keep
        // the string short to minimise overhead.
        let embedding_healthy = self
            .embedding_service
            .embed(&["ping".to_string()])
            .await
            .map(|vectors| !vectors.is_empty())
            .unwrap_or(false);

        let overall_healthy = db_healthy && embedding_healthy;
        let status = if overall_healthy {
            ServiceStatus::Healthy
        } else {
            ServiceStatus::Unhealthy
        };

        let mut components = vec![
            ComponentHealth {
                name: "embedding_service".to_string(),
                status: if embedding_healthy {
                    ServiceStatus::Healthy
                } else {
                    ServiceStatus::Unhealthy
                },
                details: None,
            },
            ComponentHealth {
                name: "database".to_string(),
                status: if db_healthy {
                    ServiceStatus::Healthy
                } else {
                    ServiceStatus::Unhealthy
                },
                details: None,
            },
        ];

        components.push(ComponentHealth {
            name: "llm_service".to_string(),
            status: self
                .llm_service
                .as_ref()
                .map(|_| ServiceStatus::Healthy)
                .unwrap_or(ServiceStatus::Disabled),
            details: None,
        });

        components.push(ComponentHealth {
            name: "reranking_service".to_string(),
            status: self
                .reranking_service
                .as_ref()
                .map(|_| ServiceStatus::Healthy)
                .unwrap_or(ServiceStatus::Disabled),
            details: None,
        });

        Ok(HealthStatus {
            status,
            components,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Get application statistics
    pub async fn get_stats(&self) -> crate::error::ApiResult<AppStats> {
        let db_stats = self.db_manager.get_stats().await.map_err(|e| {
            crate::error::ApiError::internal(format!("Failed to get database stats: {}", e))
        })?;

        Ok(AppStats {
            messages_count: db_stats.message_count as usize,
            chunks_count: db_stats.chunk_count as usize,
            embeddings_count: db_stats.embedding_count as usize,
            sessions_count: db_stats.session_count as usize,
            uptime_seconds: 0, // TODO: track real uptime
            version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono::Utc::now(),
        })
    }
}

/// Health status response
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HealthStatus {
    pub status: ServiceStatus,
    pub components: Vec<ComponentHealth>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Individual component health
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: ServiceStatus,
    pub details: Option<serde_json::Value>,
}

/// Service status enumeration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ServiceStatus {
    Healthy,
    Unhealthy,
    Disabled,
}

/// Application statistics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AppStats {
    pub messages_count: usize,
    pub chunks_count: usize,
    pub embeddings_count: usize,
    pub sessions_count: usize,
    pub uptime_seconds: usize,
    pub version: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}
