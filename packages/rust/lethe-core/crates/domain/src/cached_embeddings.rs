use crate::embeddings::EmbeddingService;
use async_trait::async_trait;
use lethe_shared::{utils::HashUtils, EmbeddingVector, LetheError, Result};
use moka::future::Cache;
use std::sync::Arc;

pub type EmbeddingCache = Cache<String, Arc<EmbeddingVector>>;

/// Wraps an embedding service with a cache keyed by the SHA-256 hash of the input text.
pub struct CachedEmbeddingService {
    inner_service: Arc<dyn EmbeddingService>,
    cache: EmbeddingCache,
}

impl CachedEmbeddingService {
    pub fn new(inner_service: Arc<dyn EmbeddingService>, cache: EmbeddingCache) -> Self {
        Self {
            inner_service,
            cache,
        }
    }

    /// Obtain a cloned handle to the cache for external inspection or metrics.
    pub fn cache(&self) -> EmbeddingCache {
        self.cache.clone()
    }
}

#[async_trait]
impl EmbeddingService for CachedEmbeddingService {
    fn name(&self) -> &str {
        self.inner_service.name()
    }

    fn dimension(&self) -> usize {
        self.inner_service.dimension()
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut cached_results: Vec<Option<Arc<EmbeddingVector>>> = vec![None; texts.len()];
        let mut misses: Vec<(usize, String, String)> = Vec::new();

        for (idx, text) in texts.iter().enumerate() {
            let key = HashUtils::sha256_hash(text);
            if let Some(embedding) = self.cache.get(&key).await {
                cached_results[idx] = Some(embedding);
            } else {
                misses.push((idx, key, text.clone()));
            }
        }

        if !misses.is_empty() {
            let miss_inputs: Vec<String> = misses.iter().map(|(_, _, text)| text.clone()).collect();
            let new_embeddings = self.inner_service.embed(&miss_inputs).await?;

            for (offset, embedding) in new_embeddings.into_iter().enumerate() {
                let (original_idx, key, _) = &misses[offset];
                let embedding_arc = Arc::new(embedding);
                self.cache.insert(key.clone(), embedding_arc.clone()).await;
                cached_results[*original_idx] = Some(embedding_arc);
            }
        }

        let mut resolved = Vec::with_capacity(cached_results.len());

        for entry in cached_results {
            let embedding = entry.ok_or_else(|| {
                LetheError::embedding("embedding cache resolution failed: missing entry")
            })?;
            resolved.push((*embedding).clone());
        }

        Ok(resolved)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingEmbeddingService {
        calls: AtomicUsize,
    }

    impl CountingEmbeddingService {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }

        fn call_count(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl EmbeddingService for CountingEmbeddingService {
        fn name(&self) -> &str {
            "counting"
        }

        fn dimension(&self) -> usize {
            1
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(texts
                .iter()
                .map(|text| EmbeddingVector {
                    data: vec![text.len() as f32],
                    dimension: 1,
                })
                .collect())
        }
    }

    fn new_cache() -> EmbeddingCache {
        Cache::builder().max_capacity(100).build()
    }

    #[tokio::test]
    async fn caches_repeated_embeddings() {
        let service = Arc::new(CountingEmbeddingService::new());
        let cached = CachedEmbeddingService::new(service.clone(), new_cache());

        let inputs = vec!["alpha".to_string()];
        let first = cached.embed(&inputs).await.unwrap();
        let second = cached.embed(&inputs).await.unwrap();

        assert_eq!(first, second);
        assert_eq!(service.call_count(), 1);
    }

    #[tokio::test]
    async fn only_misses_trigger_inner_calls() {
        let service = Arc::new(CountingEmbeddingService::new());
        let cached = CachedEmbeddingService::new(service.clone(), new_cache());

        let batch_one = vec!["alpha".to_string(), "beta".to_string()];
        let batch_two = vec!["beta".to_string(), "gamma".to_string()];

        let first = cached.embed(&batch_one).await.unwrap();
        let second = cached.embed(&batch_two).await.unwrap();

        assert_eq!(first.len(), 2);
        assert_eq!(second.len(), 2);

        // Initial call + second call for the single miss ("gamma")
        assert_eq!(service.call_count(), 2);
        // Ensure cache served the shared element without altering output
        assert_eq!(second[0].data, vec![4.0]);
        assert_eq!(second[1].data, vec![5.0]);
    }
}
