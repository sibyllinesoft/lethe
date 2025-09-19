use crate::security::SecurityContext;
use lethe_domain::{
    CachedEmbeddingService, EmbeddingCache, EmbeddingConfig as DomainEmbeddingConfig,
    EmbeddingRerankingService, EmbeddingService, EmbeddingServiceFactory, EnhancedQueryPipeline,
    LlmService, LlmServiceConfig, LlmServiceFactory, MLPredictionConfig, MLPredictionService,
    PipelineConfig, PipelineFactory, RerankingService,
};
use lethe_shared::ResolvedLetheConfig;
use lethe_storage::ParquetCorpus;
use moka::future::Cache;
use std::{
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

/// Application state containing all services and repositories
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<ResolvedLetheConfig>,
    pub corpus: Arc<ParquetCorpus>,
    pub embedding_cache: EmbeddingCache,
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
        config: Arc<ResolvedLetheConfig>,
        corpus: Arc<ParquetCorpus>,
        embedding_cache: EmbeddingCache,
        embedding_service: Arc<dyn EmbeddingService>,
        llm_service: Option<Arc<dyn LlmService>>,
        reranking_service: Option<Arc<dyn RerankingService>>,
        query_pipeline: Arc<EnhancedQueryPipeline>,
    ) -> crate::error::ApiResult<Self> {
        let security = Arc::new(SecurityContext::from_config(&config.security)?);

        Ok(Self {
            config,
            corpus,
            embedding_cache,
            embedding_service,
            llm_service,
            reranking_service,
            query_pipeline,
            security,
            server_started_at: Instant::now(),
        })
    }

    pub async fn initialise<P: AsRef<Path>>(
        config: Arc<ResolvedLetheConfig>,
        storage_root: P,
    ) -> crate::error::ApiResult<Self> {
        let domain_embedding_config = DomainEmbeddingConfig::from_shared(&config.embedding);

        let corpus = Arc::new(ParquetCorpus::new(storage_root.as_ref()));
        corpus.health_check().await.map_err(|err| {
            crate::error::ApiError::internal(format!(
                "Failed to initialise corpus storage: {}",
                err
            ))
        })?;

        let base_embedding_service =
            EmbeddingServiceFactory::create(&domain_embedding_config).await?;

        let cache_settings = &config.embedding.cache;
        let mut cache_builder = Cache::builder().max_capacity(cache_settings.max_entries.max(1));
        if cache_settings.ttl_secs > 0 {
            cache_builder =
                cache_builder.time_to_live(Duration::from_secs(cache_settings.ttl_secs));
        }
        let embedding_cache: EmbeddingCache = cache_builder.build();

        let embedding_service: Arc<dyn EmbeddingService> = Arc::new(CachedEmbeddingService::new(
            base_embedding_service,
            embedding_cache.clone(),
        ));

        let llm_service = if config.llm.enabled {
            let domain_llm_config = LlmServiceConfig::from_shared(&config.llm.settings);
            match LlmServiceFactory::create(&domain_llm_config).await {
                Ok(service) => Some(service),
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        "LLM service initialisation failed; HyDE will run in fallback mode"
                    );
                    None
                }
            }
        } else {
            None
        };

        let pipeline_config = PipelineConfig::from_resolved_config(&config);

        let reranking_service: Option<Arc<dyn RerankingService>> = if pipeline_config.rerank_enabled
        {
            Some(Arc::new(EmbeddingRerankingService::new(
                embedding_service.clone(),
            )))
        } else {
            None
        };

        let ml_prediction_service = match config.ml.static_rules.path.as_ref() {
            Some(path) => {
                let path_ref = Path::new(path);
                match MLPredictionService::from_rules_path(
                    MLPredictionConfig::default(),
                    Some(path_ref),
                ) {
                    Ok(service) => {
                        tracing::info!(rules = %path_ref.display(), "Loaded ML strategy rules from configuration");
                        service
                    }
                    Err(err) => {
                        tracing::warn!(
                            error = %err,
                            path = %path_ref.display(),
                            "Failed to load ML strategy rules; using bundled defaults"
                        );
                        MLPredictionService::default()
                    }
                }
            }
            None => MLPredictionService::default(),
        };

        let document_repository: Arc<dyn lethe_domain::retrieval::DocumentRepository> =
            corpus.clone();
        let query_pipeline = Arc::new(PipelineFactory::create_pipeline(
            pipeline_config,
            document_repository,
            embedding_service.clone(),
            llm_service.clone(),
            reranking_service.clone(),
            Some(ml_prediction_service),
        ));

        Self::new(
            config,
            corpus,
            embedding_cache,
            embedding_service,
            llm_service,
            reranking_service,
            query_pipeline,
        )
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
