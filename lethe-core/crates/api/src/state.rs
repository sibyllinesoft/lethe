use lethe_domain::{
    EmbeddingService, EnhancedQueryPipeline, LlmService, RerankingService,
};

#[cfg(feature = "database")]
use lethe_infrastructure::{
    DatabaseManager, MessageRepository, ChunkRepository, EmbeddingRepository, SessionRepository,
};

use lethe_shared::LetheConfig;
use std::sync::Arc;

/// Application state containing all services and repositories
#[derive(Clone)]
pub struct AppState {
    // Configuration
    pub config: Arc<LetheConfig>,
    
    #[cfg(feature = "database")]
    // Database
    pub db_manager: Arc<DatabaseManager>,
    
    #[cfg(feature = "database")]
    // Repositories
    pub message_repository: Arc<dyn MessageRepository>,
    #[cfg(feature = "database")]
    pub chunk_repository: Arc<dyn ChunkRepository>,
    #[cfg(feature = "database")]
    pub embedding_repository: Arc<dyn EmbeddingRepository>,
    #[cfg(feature = "database")]
    pub session_repository: Arc<dyn SessionRepository>,
    
    // Domain services
    pub embedding_service: Arc<dyn EmbeddingService>,
    pub llm_service: Option<Arc<dyn LlmService>>,
    pub reranking_service: Option<Arc<dyn RerankingService>>,
    pub query_pipeline: Arc<EnhancedQueryPipeline>,
}

impl AppState {
    #[cfg(feature = "database")]
    /// Create a new AppState instance with database
    pub fn new_with_database(
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

    #[cfg(not(feature = "database"))]
    /// Create a new AppState instance without database
    pub fn new(
        config: Arc<LetheConfig>,
        embedding_service: Arc<dyn EmbeddingService>,
        llm_service: Option<Arc<dyn LlmService>>,
        reranking_service: Option<Arc<dyn RerankingService>>,
        query_pipeline: Arc<EnhancedQueryPipeline>,
    ) -> Self {
        Self {
            config,
            embedding_service,
            llm_service,
            reranking_service,
            query_pipeline,
        }
    }

    /// Health check for the application state
    pub async fn health_check(&self) -> crate::error::ApiResult<HealthStatus> {
        #[cfg(feature = "database")]
        let db_healthy = self.db_manager.health_check().await.is_ok();
        #[cfg(not(feature = "database"))]
        let db_healthy = false;
        
        // Check embedding service (simple test)
        let embedding_healthy = self.embedding_service
            .embed("health check")
            .await
            .is_ok();
        
        let overall_healthy = embedding_healthy && (cfg!(not(feature = "database")) || db_healthy);
        let status = if overall_healthy {
            ServiceStatus::Healthy
        } else {
            ServiceStatus::Unhealthy
        };

        let mut components = vec![
            ComponentHealth {
                name: "embedding_service".to_string(),
                status: if embedding_healthy { ServiceStatus::Healthy } else { ServiceStatus::Unhealthy },
                details: None,
            },
            ComponentHealth {
                name: "llm_service".to_string(),
                status: if self.llm_service.is_some() { ServiceStatus::Healthy } else { ServiceStatus::Disabled },
                details: None,
            },
            ComponentHealth {
                name: "reranking_service".to_string(),
                status: if self.reranking_service.is_some() { ServiceStatus::Healthy } else { ServiceStatus::Disabled },
                details: None,
            },
        ];

        #[cfg(feature = "database")]
        components.push(ComponentHealth {
            name: "database".to_string(),
            status: if db_healthy { ServiceStatus::Healthy } else { ServiceStatus::Unhealthy },
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
        #[cfg(feature = "database")]
        {
            let db_stats = self.db_manager.get_stats().await
                .map_err(|e| crate::error::ApiError::internal(format!("Failed to get database stats: {}", e)))?;

            Ok(AppStats {
                messages_count: db_stats.message_count as usize,
                chunks_count: db_stats.chunk_count as usize,
                embeddings_count: db_stats.embedding_count as usize,
                sessions_count: db_stats.session_count as usize,
                uptime_seconds: 0, // TODO: Track application start time
                version: env!("CARGO_PKG_VERSION").to_string(),
                timestamp: chrono::Utc::now(),
            })
        }
        
        #[cfg(not(feature = "database"))]
        {
            Ok(AppStats {
                messages_count: 0,
                chunks_count: 0,
                embeddings_count: 0,
                sessions_count: 0,
                uptime_seconds: 0,
                version: env!("CARGO_PKG_VERSION").to_string(),
                timestamp: chrono::Utc::now(),
            })
        }
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
    pub uptime_seconds: u64,
    pub version: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_status_serialization() {
        let status = ServiceStatus::Healthy;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"healthy\"");

        let status: ServiceStatus = serde_json::from_str("\"unhealthy\"").unwrap();
        assert!(matches!(status, ServiceStatus::Unhealthy));
    }

    #[test]
    fn test_health_status_creation() {
        let health = HealthStatus {
            status: ServiceStatus::Healthy,
            components: vec![
                ComponentHealth {
                    name: "database".to_string(),
                    status: ServiceStatus::Healthy,
                    details: None,
                },
            ],
            timestamp: chrono::Utc::now(),
        };

        assert_eq!(health.components.len(), 1);
        assert_eq!(health.components[0].name, "database");
    }
}