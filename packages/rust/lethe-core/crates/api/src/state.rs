use crate::security::SecurityContext;
use lethe_domain::{
    corpus::ParquetCorpus, EmbeddingService, EnhancedQueryPipeline, LlmService, RerankingService,
};
use lethe_shared::LetheConfig;
use std::{sync::Arc, time::Instant};

/// Application state containing all services and repositories
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<LetheConfig>,
    pub corpus: Arc<ParquetCorpus>,
    pub embedding_service: Arc<dyn EmbeddingService>,
    pub llm_service: Option<Arc<dyn LlmService>>,
    pub reranking_service: Option<Arc<dyn RerankingService>>,
    pub query_pipeline: Arc<EnhancedQueryPipeline>,
    pub security: Arc<SecurityContext>,
    pub server_started_at: Instant,
}

impl AppState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: Arc<LetheConfig>,
        corpus: Arc<ParquetCorpus>,
        embedding_service: Arc<dyn EmbeddingService>,
        llm_service: Option<Arc<dyn LlmService>>,
        reranking_service: Option<Arc<dyn RerankingService>>,
        query_pipeline: Arc<EnhancedQueryPipeline>,
    ) -> crate::error::ApiResult<Self> {
        let security = Arc::new(SecurityContext::from_config(&config.security)?);

        Ok(Self {
            config,
            corpus,
            embedding_service,
            llm_service,
            reranking_service,
            query_pipeline,
            security,
            server_started_at: Instant::now(),
        })
    }

    pub async fn health_check(&self) -> crate::error::ApiResult<HealthStatus> {
        let storage_healthy = self.corpus.health_check().await.is_ok();
        let embedding_healthy = self
            .embedding_service
            .embed(&["ping".to_string()])
            .await
            .map(|vectors| !vectors.is_empty())
            .unwrap_or(false);

        let overall_healthy = storage_healthy && embedding_healthy;
        let status = if overall_healthy {
            ServiceStatus::Healthy
        } else {
            ServiceStatus::Unhealthy
        };

        let mut components = vec![
            ComponentHealth {
                name: "storage".to_string(),
                status: if storage_healthy {
                    ServiceStatus::Healthy
                } else {
                    ServiceStatus::Unhealthy
                },
                details: None,
            },
            ComponentHealth {
                name: "embedding_service".to_string(),
                status: if embedding_healthy {
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

    pub async fn get_stats(&self) -> crate::error::ApiResult<AppStats> {
        let stats = self.corpus.stats().await.map_err(|e| {
            crate::error::ApiError::internal(format!("Failed to read storage stats: {}", e))
        })?;

        Ok(AppStats {
            messages_count: stats.message_count,
            chunks_count: stats.chunk_count,
            embeddings_count: stats.embedding_count,
            sessions_count: stats.session_count,
            uptime_seconds: self.server_started_at.elapsed().as_secs() as usize,
            version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono::Utc::now(),
        })
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HealthStatus {
    pub status: ServiceStatus,
    pub components: Vec<ComponentHealth>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: ServiceStatus,
    pub details: Option<serde_json::Value>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ServiceStatus {
    Healthy,
    Unhealthy,
    Disabled,
}

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
