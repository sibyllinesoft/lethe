use async_trait::async_trait;
use lethe_shared::{
    config::EmbeddingCacheConfig as SharedEmbeddingCacheConfig, EmbeddingVector, LetheError, Result,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

/// Configuration for embedding providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub provider: EmbeddingProvider,
    pub model_name: String,
    pub dimension: usize,
    pub batch_size: usize,
    pub timeout_ms: u64,
    pub cache: SharedEmbeddingCacheConfig,
}

impl EmbeddingConfig {
    pub fn from_shared(shared: &lethe_shared::config::EmbeddingConfig) -> Self {
        let provider = match &shared.provider {
            lethe_shared::config::EmbeddingProvider::Ollama { base_url, model } => {
                EmbeddingProvider::Ollama {
                    base_url: base_url.clone(),
                    model: model.clone(),
                }
            }
            lethe_shared::config::EmbeddingProvider::Fallback => EmbeddingProvider::Fallback,
        };

        let model_name = match &shared.provider {
            lethe_shared::config::EmbeddingProvider::Ollama { model, .. } => model.clone(),
            lethe_shared::config::EmbeddingProvider::Fallback => "fallback".to_string(),
        };

        Self {
            provider,
            model_name,
            dimension: shared.dimension,
            batch_size: 32,
            timeout_ms: shared.timeout_ms,
            cache: shared.cache.clone(),
        }
    }
}

/// Available embedding providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmbeddingProvider {
    Ollama {
        base_url: String,
        model: String,
    },
    Fallback,
    Custom {
        name: String,
        settings: Option<Value>,
    },
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: EmbeddingProvider::Fallback,
            model_name: "fallback".to_string(),
            dimension: 384,
            batch_size: 32,
            timeout_ms: 30000,
            cache: SharedEmbeddingCacheConfig::default(),
        }
    }
}

/// Trait for embedding providers
#[async_trait]
pub trait EmbeddingService: Send + Sync {
    /// Get the provider name
    fn name(&self) -> &str;

    /// Get embedding dimension
    fn dimension(&self) -> usize;

    /// Generate embeddings for a batch of texts
    async fn embed(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>>;

    /// Generate a single embedding
    async fn embed_single(&self, text: &str) -> Result<EmbeddingVector> {
        let results = self.embed(&[text.to_string()]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| LetheError::embedding("No embedding returned for single text"))
    }
}

#[async_trait]
pub trait EmbeddingProviderBuilder: Send + Sync {
    fn id(&self) -> &'static str;
    fn matches(&self, provider: &EmbeddingProvider) -> bool;
    async fn build(
        &self,
        config: &EmbeddingConfig,
        provider: &EmbeddingProvider,
    ) -> Result<Arc<dyn EmbeddingService>>;
}

fn embedding_provider_registry(
) -> &'static RwLock<HashMap<&'static str, Arc<dyn EmbeddingProviderBuilder>>> {
    static REGISTRY: OnceLock<RwLock<HashMap<&'static str, Arc<dyn EmbeddingProviderBuilder>>>> =
        OnceLock::new();

    REGISTRY.get_or_init(|| {
        let mut providers: HashMap<&'static str, Arc<dyn EmbeddingProviderBuilder>> =
            HashMap::new();
        providers.insert("ollama", Arc::new(OllamaEmbeddingProvider));
        providers.insert("fallback", Arc::new(FallbackEmbeddingProvider));
        RwLock::new(providers)
    })
}

pub fn register_embedding_provider(builder: Arc<dyn EmbeddingProviderBuilder>) {
    let registry = embedding_provider_registry();
    if let Ok(mut guard) = registry.write() {
        guard.insert(builder.id(), builder);
    }
}

/// Ollama embedding service
pub struct OllamaEmbeddingService {
    base_url: String,
    model: String,
    dimension: usize,
    client: reqwest::Client,
}

impl OllamaEmbeddingService {
    /// Create a new Ollama embedding service
    pub fn new(base_url: String, model: String, dimension: usize) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            base_url,
            model,
            dimension,
            client,
        }
    }

    /// Test connectivity to Ollama service
    pub async fn test_connectivity(&self) -> Result<bool> {
        let url = format!("{}/api/version", self.base_url);

        match tokio::time::timeout(
            std::time::Duration::from_millis(500),
            self.client.get(&url).send(),
        )
        .await
        {
            Ok(Ok(response)) => Ok(response.status().is_success()),
            _ => Ok(false),
        }
    }
}

#[async_trait]
impl EmbeddingService for OllamaEmbeddingService {
    fn name(&self) -> &str {
        "ollama"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>> {
        let mut embeddings = Vec::new();

        for text in texts {
            let request_body = serde_json::json!({
                "model": self.model,
                "prompt": text,
            });

            let url = format!("{}/api/embeddings", self.base_url);
            let response = self
                .client
                .post(&url)
                .json(&request_body)
                .send()
                .await
                .map_err(|e| LetheError::embedding(format!("Ollama request failed: {}", e)))?;

            if !response.status().is_success() {
                return Err(LetheError::embedding(format!(
                    "Ollama API error: {}",
                    response.status()
                )));
            }

            let response_json: serde_json::Value = response.json().await.map_err(|e| {
                LetheError::embedding(format!("Failed to parse Ollama response: {}", e))
            })?;

            let embedding_data = response_json
                .get("embedding")
                .and_then(|e| e.as_array())
                .ok_or_else(|| LetheError::embedding("No embedding data in Ollama response"))?;

            let data: Vec<f32> = embedding_data
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();

            embeddings.push(EmbeddingVector {
                data,
                dimension: self.dimension,
            });
        }

        Ok(embeddings)
    }
}

struct OllamaEmbeddingProvider;

#[async_trait]
impl EmbeddingProviderBuilder for OllamaEmbeddingProvider {
    fn id(&self) -> &'static str {
        "ollama"
    }

    fn matches(&self, provider: &EmbeddingProvider) -> bool {
        matches!(provider, EmbeddingProvider::Ollama { .. })
    }

    async fn build(
        &self,
        config: &EmbeddingConfig,
        provider: &EmbeddingProvider,
    ) -> Result<Arc<dyn EmbeddingService>> {
        if let EmbeddingProvider::Ollama { base_url, model } = provider {
            let service =
                OllamaEmbeddingService::new(base_url.clone(), model.clone(), config.dimension);

            if service.test_connectivity().await? {
                tracing::info!("Using Ollama embeddings with model: {}", model);
                Ok(Arc::new(service))
            } else {
                tracing::warn!("Ollama not available, falling back to zero vectors");
                Ok(Arc::new(FallbackEmbeddingService::new(config.dimension)))
            }
        } else {
            Err(LetheError::config(
                "Ollama embedding provider received incompatible configuration",
            ))
        }
    }
}

/// Fallback embedding service that returns zero vectors
pub struct FallbackEmbeddingService {
    dimension: usize,
}

impl FallbackEmbeddingService {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

#[async_trait]
impl EmbeddingService for FallbackEmbeddingService {
    fn name(&self) -> &str {
        "fallback"
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>> {
        tracing::warn!(
            "Using fallback zero-vector embeddings for {} texts - vector search will be disabled",
            texts.len()
        );

        let embeddings = texts
            .iter()
            .map(|_| EmbeddingVector {
                data: vec![0.0; self.dimension],
                dimension: self.dimension,
            })
            .collect();

        Ok(embeddings)
    }
}

struct FallbackEmbeddingProvider;

#[async_trait]
impl EmbeddingProviderBuilder for FallbackEmbeddingProvider {
    fn id(&self) -> &'static str {
        "fallback"
    }

    fn matches(&self, provider: &EmbeddingProvider) -> bool {
        matches!(provider, EmbeddingProvider::Fallback)
    }

    async fn build(
        &self,
        config: &EmbeddingConfig,
        _provider: &EmbeddingProvider,
    ) -> Result<Arc<dyn EmbeddingService>> {
        Ok(Arc::new(FallbackEmbeddingService::new(config.dimension)))
    }
}

/// Factory for creating embedding services
pub struct EmbeddingServiceFactory;

impl EmbeddingServiceFactory {
    /// Create an embedding service based on configuration
    pub async fn create(config: &EmbeddingConfig) -> Result<Arc<dyn EmbeddingService>> {
        if let Some(result) = Self::build_from_registry(config, &config.provider).await {
            return result;
        }

        match &config.provider {
            EmbeddingProvider::Custom { name, .. } => Err(LetheError::config(format!(
                "No embedding provider registered for '{}'",
                name
            ))),
            _ => Err(LetheError::config(
                "No embedding provider available for requested configuration",
            )),
        }
    }

    /// Create embedding service with preference detection
    pub async fn create_with_preference(
        preference: Option<&str>,
    ) -> Result<Arc<dyn EmbeddingService>> {
        let config = match preference {
            Some("ollama") => EmbeddingConfig {
                provider: EmbeddingProvider::Ollama {
                    base_url: "http://localhost:11434".to_string(),
                    model: "nomic-embed-text".to_string(),
                },
                model_name: "nomic-embed-text".to_string(),
                dimension: 768,
                ..Default::default()
            },
            Some("transformersjs") => {
                tracing::warn!(
                    "Transformers.js embedding preference is no longer supported; using fallback embeddings"
                );
                EmbeddingConfig::default()
            }
            _ => EmbeddingConfig::default(),
        };

        Self::create(&config).await
    }

    async fn build_from_registry(
        config: &EmbeddingConfig,
        provider: &EmbeddingProvider,
    ) -> Option<Result<Arc<dyn EmbeddingService>>> {
        let builder = {
            let registry = embedding_provider_registry();
            let guard = registry.read().ok()?;

            match provider {
                EmbeddingProvider::Custom { name, .. } => guard.get(name.as_str()).cloned(),
                _ => guard
                    .values()
                    .find(|builder| builder.matches(provider))
                    .cloned(),
            }
        }?;

        Some(builder.build(config, provider).await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fallback_embedding_service() {
        let service = FallbackEmbeddingService::new(384);
        let texts = vec!["hello".to_string(), "world".to_string()];

        let embeddings = service.embed(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].dimension, 384);
        assert_eq!(embeddings[0].data.len(), 384);
        assert!(embeddings[0].data.iter().all(|&x| x == 0.0));
    }

    #[tokio::test]
    async fn test_embedding_service_factory() {
        let config = EmbeddingConfig {
            provider: EmbeddingProvider::Fallback,
            dimension: 512,
            ..Default::default()
        };

        let service = EmbeddingServiceFactory::create(&config).await.unwrap();

        assert_eq!(service.name(), "fallback");
        assert_eq!(service.dimension(), 512);
    }

    #[test]
    fn test_embedding_config_serialization() {
        let config = EmbeddingConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: EmbeddingConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.dimension, deserialized.dimension);
        assert_eq!(config.batch_size, deserialized.batch_size);
    }

    #[tokio::test]
    async fn test_single_embedding() {
        let service = FallbackEmbeddingService::new(128);
        let embedding = service.embed_single("test text").await.unwrap();

        assert_eq!(embedding.dimension, 128);
        assert_eq!(embedding.data.len(), 128);
    }

    #[tokio::test]
    async fn test_empty_text_embedding() {
        let service = FallbackEmbeddingService::new(256);

        // Test empty string
        let embedding = service.embed_single("").await.unwrap();
        assert_eq!(embedding.dimension, 256);
        assert_eq!(embedding.data.len(), 256);

        // Test whitespace only
        let embedding = service.embed_single("   ").await.unwrap();
        assert_eq!(embedding.dimension, 256);
        assert!(embedding.data.iter().all(|&x| x == 0.0));
    }

    #[tokio::test]
    async fn test_large_batch_embedding() {
        let service = FallbackEmbeddingService::new(128);

        // Create a large batch of texts
        let texts: Vec<String> = (0..100).map(|i| format!("text {}", i)).collect();

        let embeddings = service.embed(&texts).await.unwrap();

        assert_eq!(embeddings.len(), 100);
        for (i, embedding) in embeddings.iter().enumerate() {
            assert_eq!(embedding.dimension, 128);
            assert_eq!(embedding.data.len(), 128);
            // Each embedding should be zero vectors for fallback
            assert!(
                embedding.data.iter().all(|&x| x == 0.0),
                "Embedding {} should be zero vector",
                i
            );
        }
    }

    #[tokio::test]
    async fn test_embedding_vector_properties() {
        let service = FallbackEmbeddingService::new(512);
        let embedding = service.embed_single("sample text").await.unwrap();

        // Test that embedding has correct properties
        assert_eq!(embedding.dimension, 512);
        assert_eq!(embedding.data.len(), 512);

        // For fallback service, all values should be 0.0
        assert!(embedding.data.iter().all(|&x| x.is_finite()));
        assert!(embedding.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_embedding_config_default_values() {
        let config = EmbeddingConfig::default();

        assert_eq!(config.dimension, 384);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.timeout_ms, 30000);
        assert_eq!(config.model_name, "fallback");

        match config.provider {
            EmbeddingProvider::Fallback => {}
            _ => panic!("Expected fallback provider"),
        }
    }

    #[test]
    fn test_embedding_provider_variants() {
        let ollama_provider = EmbeddingProvider::Ollama {
            base_url: "http://localhost:11434".to_string(),
            model: "embeddings".to_string(),
        };

        let fallback_provider = EmbeddingProvider::Fallback;

        let custom_provider = EmbeddingProvider::Custom {
            name: "custom".to_string(),
            settings: None,
        };

        // Test that all variants can be created
        match ollama_provider {
            EmbeddingProvider::Ollama { base_url, model } => {
                assert_eq!(base_url, "http://localhost:11434");
                assert_eq!(model, "embeddings");
            }
            _ => panic!("Expected Ollama variant"),
        }

        match fallback_provider {
            EmbeddingProvider::Fallback => {}
            _ => panic!("Expected Fallback variant"),
        }

        match custom_provider {
            EmbeddingProvider::Custom { name, settings } => {
                assert_eq!(name, "custom");
                assert!(settings.is_none());
            }
            _ => panic!("Expected Custom variant"),
        }
    }

    #[tokio::test]
    async fn test_embedding_service_interface() {
        let service = FallbackEmbeddingService::new(256);

        // Test name
        assert_eq!(service.name(), "fallback");

        // Test dimension
        assert_eq!(service.dimension(), 256);

        // Test embed method
        let texts = vec!["text1".to_string(), "text2".to_string()];
        let embeddings = service.embed(&texts).await.unwrap();
        assert_eq!(embeddings.len(), 2);

        // Test embed_single method
        let single_embedding = service.embed_single("single").await.unwrap();
        assert_eq!(single_embedding.dimension, 256);
    }

    #[test]
    fn test_embedding_config_clone_and_debug() {
        let config = EmbeddingConfig::default();

        // Test Clone trait
        let cloned_config = config.clone();
        assert_eq!(config.dimension, cloned_config.dimension);
        assert_eq!(config.batch_size, cloned_config.batch_size);

        // Test Debug trait
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("EmbeddingConfig"));
        assert!(debug_str.contains("dimension"));
        assert!(debug_str.contains("batch_size"));
    }

    #[tokio::test]
    async fn test_embedding_error_scenarios() {
        let service = FallbackEmbeddingService::new(64);

        // Test with very long text (should still work with fallback)
        let long_text = "a".repeat(10000);
        let embedding = service.embed_single(&long_text).await.unwrap();
        assert_eq!(embedding.dimension, 64);

        // Test with special characters
        let special_text = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~";
        let embedding = service.embed_single(special_text).await.unwrap();
        assert_eq!(embedding.dimension, 64);

        // Test with unicode
        let unicode_text = "Hello 世界 🌍 тест";
        let embedding = service.embed_single(unicode_text).await.unwrap();
        assert_eq!(embedding.dimension, 64);
    }
}
