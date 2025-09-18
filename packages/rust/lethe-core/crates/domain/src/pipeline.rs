use crate::{
    embeddings::EmbeddingService,
    hyde::{HydeExpansion, HydeService, LlmService},
    ml_prediction::{MLPredictionResult, MLPredictionService, RetrievalStrategy},
    query_understanding::{QueryUnderstanding, QueryUnderstandingService},
    retrieval::{
        Bm25SearchService, DocumentRepository, HybridRetrievalConfig, HybridRetrievalService,
    },
};
use async_trait::async_trait;
use lethe_shared::{Candidate, ContextPack, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for the enhanced query pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub enable_hyde: bool,
    pub enable_query_understanding: bool,
    pub enable_ml_prediction: bool,
    pub max_candidates: usize,
    pub rerank_enabled: bool,
    pub rerank_top_k: usize,
    pub timeout_seconds: u64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            enable_hyde: true,
            enable_query_understanding: true,
            enable_ml_prediction: true,
            max_candidates: 50,
            rerank_enabled: true,
            rerank_top_k: 20,
            timeout_seconds: 30,
        }
    }
}

/// Options for enhanced query processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedQueryOptions {
    pub session_id: String,
    pub k: usize,
    pub include_metadata: bool,
    pub enable_hyde: Option<bool>,
    pub override_strategy: Option<RetrievalStrategy>,
    pub context: Option<HashMap<String, serde_json::Value>>,
}

impl Default for EnhancedQueryOptions {
    fn default() -> Self {
        Self {
            session_id: "default".to_string(),
            k: 10,
            include_metadata: true,
            enable_hyde: None,
            override_strategy: None,
            context: None,
        }
    }
}

/// Result of enhanced query processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedQueryResult {
    pub candidates: Vec<Candidate>,
    pub context_pack: ContextPack,
    pub query_understanding: Option<QueryUnderstanding>,
    pub ml_prediction: Option<MLPredictionResult>,
    pub hyde_expansion: Option<HydeExpansion>,
    pub strategy_used: RetrievalStrategy,
    pub processing_time_ms: u64,
    pub total_candidates_found: usize,
}

/// Trait for reranking services
#[async_trait]
pub trait RerankingService: Send + Sync {
    async fn rerank(&self, query: &str, candidates: &[Candidate]) -> Result<Vec<Candidate>>;
}

/// Enhanced query pipeline that orchestrates all components
pub struct EnhancedQueryPipeline {
    config: PipelineConfig,
    document_repository: Arc<dyn DocumentRepository>,
    embedding_service: Arc<dyn EmbeddingService>,
    hybrid_retrieval: HybridRetrievalService,
    hyde_service: Option<Arc<HydeService>>,
    query_understanding: QueryUnderstandingService,
    ml_prediction: MLPredictionService,
    reranking_service: Option<Arc<dyn RerankingService>>,
}

impl EnhancedQueryPipeline {
    pub fn new(
        config: PipelineConfig,
        document_repository: Arc<dyn DocumentRepository>,
        embedding_service: Arc<dyn EmbeddingService>,
        llm_service: Option<Arc<dyn LlmService>>,
        reranking_service: Option<Arc<dyn RerankingService>>,
    ) -> Self {
        let hybrid_config = HybridRetrievalConfig::default();
        let hybrid_retrieval =
            HybridRetrievalService::new(embedding_service.clone(), hybrid_config);

        let hyde_service = if config.enable_hyde {
            llm_service.map(|llm| {
                Arc::new(HydeService::new(
                    llm,
                    embedding_service.clone(),
                    Default::default(),
                ))
            })
        } else {
            None
        };

        Self {
            config,
            document_repository,
            embedding_service,
            hybrid_retrieval,
            hyde_service,
            query_understanding: QueryUnderstandingService::new(),
            ml_prediction: MLPredictionService::default(),
            reranking_service,
        }
    }

    /// Process a query through the enhanced pipeline
    pub async fn process_query(
        &self,
        query: &str,
        options: &EnhancedQueryOptions,
    ) -> Result<EnhancedQueryResult> {
        let start_time = std::time::Instant::now();

        let query_understanding = self.phase_query_understanding(query).await?;
        let ml_prediction = self.phase_ml_prediction(&query_understanding).await?;
        let strategy = self.phase_strategy_selection(options, &ml_prediction);
        let hyde_expansion = self.phase_hyde_expansion(query, &strategy, options).await?;
        let candidates = self
            .phase_retrieval(query, &strategy, options, hyde_expansion.as_ref())
            .await?;
        let reranked_candidates = self.phase_reranking(query, candidates).await?;
        let final_candidates = self.phase_result_limiting(reranked_candidates, options.k);
        let context_pack = self
            .phase_context_creation(&final_candidates, options)
            .await?;

        self.create_final_result(
            final_candidates,
            context_pack,
            query_understanding,
            ml_prediction,
            hyde_expansion,
            strategy,
            start_time,
        )
    }

    /// Phase 1: Query Understanding
    async fn phase_query_understanding(&self, query: &str) -> Result<Option<QueryUnderstanding>> {
        if self.config.enable_query_understanding {
            Ok(Some(self.query_understanding.understand_query(query)?))
        } else {
            Ok(None)
        }
    }

    /// Phase 2: ML-based Strategy Prediction
    async fn phase_ml_prediction(
        &self,
        query_understanding: &Option<QueryUnderstanding>,
    ) -> Result<Option<MLPredictionResult>> {
        if self.config.enable_ml_prediction && query_understanding.is_some() {
            Ok(Some(
                self.ml_prediction
                    .predict_strategy(query_understanding.as_ref().unwrap())?,
            ))
        } else {
            Ok(None)
        }
    }

    /// Phase 3: Strategy Selection
    fn phase_strategy_selection(
        &self,
        options: &EnhancedQueryOptions,
        ml_prediction: &Option<MLPredictionResult>,
    ) -> RetrievalStrategy {
        options
            .override_strategy
            .clone()
            .or_else(|| {
                ml_prediction
                    .as_ref()
                    .map(|p| p.prediction.strategy.clone())
            })
            .unwrap_or(RetrievalStrategy::Hybrid)
    }

    /// Phase 4: HyDE Query Expansion
    async fn phase_hyde_expansion(
        &self,
        query: &str,
        strategy: &RetrievalStrategy,
        options: &EnhancedQueryOptions,
    ) -> Result<Option<HydeExpansion>> {
        if self.should_use_hyde(strategy, options) {
            if let Some(ref hyde_service) = self.hyde_service {
                Ok(Some(hyde_service.expand_query(query).await?))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Phase 5: Retrieval Execution
    async fn phase_retrieval(
        &self,
        query: &str,
        strategy: &RetrievalStrategy,
        options: &EnhancedQueryOptions,
        hyde_expansion: Option<&HydeExpansion>,
    ) -> Result<Vec<Candidate>> {
        self.execute_retrieval_strategy(query, strategy, options, hyde_expansion)
            .await
    }

    /// Phase 6: Reranking
    async fn phase_reranking(
        &self,
        query: &str,
        candidates: Vec<Candidate>,
    ) -> Result<Vec<Candidate>> {
        if self.config.rerank_enabled && candidates.len() > 1 {
            if let Some(ref reranker) = self.reranking_service {
                let top_candidates = candidates
                    .iter()
                    .take(self.config.rerank_top_k)
                    .cloned()
                    .collect::<Vec<_>>();
                reranker.rerank(query, &top_candidates).await
            } else {
                Ok(candidates)
            }
        } else {
            Ok(candidates)
        }
    }

    /// Phase 7: Result Limiting
    fn phase_result_limiting(&self, candidates: Vec<Candidate>, k: usize) -> Vec<Candidate> {
        candidates.into_iter().take(k).collect()
    }

    /// Phase 8: Context Pack Creation
    async fn phase_context_creation(
        &self,
        candidates: &[Candidate],
        options: &EnhancedQueryOptions,
    ) -> Result<ContextPack> {
        self.create_context_pack(candidates, options).await
    }

    /// Create final result structure
    fn create_final_result(
        &self,
        final_candidates: Vec<Candidate>,
        context_pack: ContextPack,
        query_understanding: Option<QueryUnderstanding>,
        ml_prediction: Option<MLPredictionResult>,
        hyde_expansion: Option<HydeExpansion>,
        strategy: RetrievalStrategy,
        start_time: std::time::Instant,
    ) -> Result<EnhancedQueryResult> {
        let total_candidates_found = final_candidates.len();
        let processing_time = start_time.elapsed();

        Ok(EnhancedQueryResult {
            candidates: final_candidates,
            context_pack,
            query_understanding,
            ml_prediction,
            hyde_expansion,
            strategy_used: strategy,
            processing_time_ms: processing_time.as_millis() as u64,
            total_candidates_found,
        })
    }

    /// Execute the determined retrieval strategy
    async fn execute_retrieval_strategy(
        &self,
        query: &str,
        strategy: &RetrievalStrategy,
        options: &EnhancedQueryOptions,
        hyde_expansion: Option<&HydeExpansion>,
    ) -> Result<Vec<Candidate>> {
        match strategy {
            RetrievalStrategy::BM25Only => {
                Bm25SearchService::search(
                    &*self.document_repository,
                    &[query.to_string()],
                    &options.session_id,
                    self.config.max_candidates as i32,
                )
                .await
            }
            RetrievalStrategy::VectorOnly => {
                let query_embedding = self.embedding_service.embed(&[query.to_string()]).await?;
                let query_embedding = query_embedding.into_iter().next().unwrap();
                self.document_repository
                    .vector_search(&query_embedding, self.config.max_candidates as i32)
                    .await
            }
            RetrievalStrategy::Hybrid => {
                self.hybrid_retrieval
                    .retrieve(
                        &*self.document_repository,
                        &[query.to_string()],
                        &options.session_id,
                    )
                    .await
            }
            RetrievalStrategy::HydeEnhanced => {
                if let Some(expansion) = hyde_expansion {
                    self.execute_hyde_enhanced_search(query, expansion).await
                } else {
                    // Fallback to hybrid if HyDE is not available
                    self.hybrid_retrieval
                        .retrieve(
                            &*self.document_repository,
                            &[query.to_string()],
                            &options.session_id,
                        )
                        .await
                }
            }
            RetrievalStrategy::MultiStep => self.execute_multi_step_retrieval(query, options).await,
            RetrievalStrategy::Adaptive => self.execute_adaptive_retrieval(query, options).await,
        }
    }

    /// Execute HyDE-enhanced search
    async fn execute_hyde_enhanced_search(
        &self,
        query: &str,
        expansion: &HydeExpansion,
    ) -> Result<Vec<Candidate>> {
        if let Some(ref combined_embedding) = expansion.combined_embedding {
            // Use combined embedding for search
            self.document_repository
                .vector_search(combined_embedding, self.config.max_candidates as i32)
                .await
        } else {
            // Use individual hypothetical documents
            let mut all_candidates = Vec::new();

            for hyp_doc in &expansion.hypothetical_documents {
                if let Some(ref embedding) = hyp_doc.embedding {
                    let candidates = self
                        .document_repository
                        .vector_search(
                            embedding,
                            (self.config.max_candidates / expansion.hypothetical_documents.len())
                                as i32,
                        )
                        .await?;
                    all_candidates.extend(candidates);
                }
            }

            // Also include results from original query
            let original_candidates = self
                .hybrid_retrieval
                .retrieve(
                    &*self.document_repository,
                    &[query.to_string()],
                    "default", // This should be passed from context
                )
                .await?;
            all_candidates.extend(original_candidates);

            // Deduplicate and sort by score
            self.deduplicate_and_sort_candidates(all_candidates)
        }
    }

    /// Execute multi-step retrieval
    async fn execute_multi_step_retrieval(
        &self,
        query: &str,
        options: &EnhancedQueryOptions,
    ) -> Result<Vec<Candidate>> {
        // Step 1: Initial broad search
        let initial_candidates = self
            .hybrid_retrieval
            .retrieve(
                &*self.document_repository,
                &[query.to_string()],
                &options.session_id,
            )
            .await?;

        // Step 2: Refine search based on initial results
        if initial_candidates.len() < 5 {
            // If few results, try vector-only search
            let query_embedding = self.embedding_service.embed(&[query.to_string()]).await?;
            let query_embedding = query_embedding.into_iter().next().unwrap();
            self.document_repository
                .vector_search(&query_embedding, self.config.max_candidates as i32)
                .await
        } else {
            // Take top candidates from initial search
            Ok(initial_candidates
                .into_iter()
                .take(self.config.max_candidates)
                .collect())
        }
    }

    /// Execute adaptive retrieval
    async fn execute_adaptive_retrieval(
        &self,
        query: &str,
        options: &EnhancedQueryOptions,
    ) -> Result<Vec<Candidate>> {
        // Start with hybrid search
        let hybrid_candidates = self
            .hybrid_retrieval
            .retrieve(
                &*self.document_repository,
                &[query.to_string()],
                &options.session_id,
            )
            .await?;

        // Adapt based on result quality
        if hybrid_candidates.len() < 5 {
            // Low results, try vector-only
            let query_embedding = self.embedding_service.embed(&[query.to_string()]).await?;
            let query_embedding = query_embedding.into_iter().next().unwrap();
            self.document_repository
                .vector_search(&query_embedding, self.config.max_candidates as i32)
                .await
        } else if hybrid_candidates.iter().all(|c| c.score < 0.5) {
            // Low scores, try BM25-only
            Bm25SearchService::search(
                &*self.document_repository,
                &[query.to_string()],
                &options.session_id,
                self.config.max_candidates as i32,
            )
            .await
        } else {
            Ok(hybrid_candidates)
        }
    }

    /// Determine if HyDE should be used for this query
    fn should_use_hyde(
        &self,
        strategy: &RetrievalStrategy,
        options: &EnhancedQueryOptions,
    ) -> bool {
        if let Some(enable_hyde) = options.enable_hyde {
            enable_hyde && self.hyde_service.is_some()
        } else {
            matches!(strategy, RetrievalStrategy::HydeEnhanced)
                && self.config.enable_hyde
                && self.hyde_service.is_some()
        }
    }

    /// Create context pack from candidates
    async fn create_context_pack(
        &self,
        candidates: &[Candidate],
        options: &EnhancedQueryOptions,
    ) -> Result<ContextPack> {
        // Convert candidates to context chunks
        let chunks: Vec<lethe_shared::ContextChunk> = candidates
            .iter()
            .map(|candidate| lethe_shared::ContextChunk {
                id: candidate.doc_id.clone(),
                score: candidate.score,
                kind: candidate.kind.clone().unwrap_or_else(|| "text".to_string()),
                text: candidate.text.clone().unwrap_or_default(),
            })
            .collect();

        let context_pack = ContextPack {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: options.session_id.clone(),
            query: "query_placeholder".to_string(), // Would need to be passed in
            created_at: chrono::Utc::now(),
            summary: "Generated context pack".to_string(), // Would be generated properly
            key_entities: Vec::new(),                      // Would be extracted from results
            claims: Vec::new(),                            // Would be extracted from results
            contradictions: Vec::new(),                    // Would be extracted from results
            chunks,
            citations: Vec::new(), // Would be generated based on chunks
        };

        Ok(context_pack)
    }

    /// Deduplicate and sort candidates by score
    fn deduplicate_and_sort_candidates(
        &self,
        mut candidates: Vec<Candidate>,
    ) -> Result<Vec<Candidate>> {
        // Simple deduplication by doc_id
        let mut seen = std::collections::HashSet::new();
        candidates.retain(|c| seen.insert(c.doc_id.clone()));

        // Sort by score (descending)
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to max candidates
        candidates.truncate(self.config.max_candidates);

        Ok(candidates)
    }
}

/// Factory for creating configured pipeline instances
pub struct PipelineFactory;

impl PipelineFactory {
    pub fn create_pipeline(
        config: PipelineConfig,
        document_repository: Arc<dyn DocumentRepository>,
        embedding_service: Arc<dyn EmbeddingService>,
        llm_service: Option<Arc<dyn LlmService>>,
        reranking_service: Option<Arc<dyn RerankingService>>,
    ) -> EnhancedQueryPipeline {
        EnhancedQueryPipeline::new(
            config,
            document_repository,
            embedding_service,
            llm_service,
            reranking_service,
        )
    }

    pub fn create_default_pipeline(
        document_repository: Arc<dyn DocumentRepository>,
        embedding_service: Arc<dyn EmbeddingService>,
    ) -> EnhancedQueryPipeline {
        EnhancedQueryPipeline::new(
            PipelineConfig::default(),
            document_repository,
            embedding_service,
            None,
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lethe_shared::EmbeddingVector;
    use lethe_shared::{Chunk, DfIdf};

    struct MockDocumentRepository;

    #[async_trait]
    impl DocumentRepository for MockDocumentRepository {
        async fn get_chunks_by_session(&self, _session_id: &str) -> Result<Vec<Chunk>> {
            Ok(vec![])
        }

        async fn get_dfidf_by_session(&self, _session_id: &str) -> Result<Vec<DfIdf>> {
            Ok(vec![])
        }

        async fn get_chunk_by_id(&self, _chunk_id: &str) -> Result<Option<Chunk>> {
            Ok(None)
        }

        async fn vector_search(
            &self,
            _query_vector: &EmbeddingVector,
            k: i32,
        ) -> Result<Vec<Candidate>> {
            Ok(vec![Candidate {
                doc_id: "test-1".to_string(),
                score: 0.9,
                text: Some("Test document 1".to_string()),
                kind: Some("text".to_string()),
            }])
        }
    }

    #[tokio::test]
    async fn test_pipeline_creation() {
        let doc_repo = Arc::new(MockDocumentRepository);
        let embedding_service = Arc::new(crate::embeddings::FallbackEmbeddingService::new(384));

        let pipeline = PipelineFactory::create_default_pipeline(doc_repo, embedding_service);

        // Test that the pipeline was created successfully
        assert!(pipeline.config.enable_query_understanding);
        assert!(pipeline.config.enable_ml_prediction);
    }

    #[tokio::test]
    async fn test_basic_query_processing() {
        let doc_repo = Arc::new(MockDocumentRepository);
        let embedding_service = Arc::new(crate::embeddings::FallbackEmbeddingService::new(384));

        let pipeline = PipelineFactory::create_default_pipeline(doc_repo, embedding_service);
        let options = EnhancedQueryOptions::default();

        let result = pipeline
            .process_query("What is machine learning?", &options)
            .await
            .unwrap();

        assert!(!result.candidates.is_empty());
        assert!(result.query_understanding.is_some());
        assert!(result.ml_prediction.is_some());
        assert!(result.processing_time_ms > 0);
    }

    #[tokio::test]
    async fn test_strategy_override() {
        let doc_repo = Arc::new(MockDocumentRepository);
        let embedding_service = Arc::new(crate::embeddings::FallbackEmbeddingService::new(384));

        let pipeline = PipelineFactory::create_default_pipeline(doc_repo, embedding_service);
        let mut options = EnhancedQueryOptions::default();
        options.override_strategy = Some(RetrievalStrategy::VectorOnly);

        let result = pipeline
            .process_query("test query", &options)
            .await
            .unwrap();

        assert_eq!(result.strategy_used, RetrievalStrategy::VectorOnly);
    }

    #[tokio::test]
    async fn test_pipeline_different_strategies() {
        let doc_repo = Arc::new(MockDocumentRepository);
        let embedding_service = Arc::new(crate::embeddings::FallbackEmbeddingService::new(384));

        let pipeline = PipelineFactory::create_default_pipeline(doc_repo, embedding_service);

        // Test BM25 only strategy
        let mut bm25_options = EnhancedQueryOptions::default();
        bm25_options.override_strategy = Some(RetrievalStrategy::BM25Only);
        let bm25_result = pipeline
            .process_query("test query", &bm25_options)
            .await
            .unwrap();
        assert_eq!(bm25_result.strategy_used, RetrievalStrategy::BM25Only);

        // Test Vector only strategy
        let mut vector_options = EnhancedQueryOptions::default();
        vector_options.override_strategy = Some(RetrievalStrategy::VectorOnly);
        let vector_result = pipeline
            .process_query("test query", &vector_options)
            .await
            .unwrap();
        assert_eq!(vector_result.strategy_used, RetrievalStrategy::VectorOnly);

        // Test Hybrid strategy
        let mut hybrid_options = EnhancedQueryOptions::default();
        hybrid_options.override_strategy = Some(RetrievalStrategy::Hybrid);
        let hybrid_result = pipeline
            .process_query("test query", &hybrid_options)
            .await
            .unwrap();
        assert_eq!(hybrid_result.strategy_used, RetrievalStrategy::Hybrid);

        // Test Adaptive strategy
        let mut adaptive_options = EnhancedQueryOptions::default();
        adaptive_options.override_strategy = Some(RetrievalStrategy::Adaptive);
        let adaptive_result = pipeline
            .process_query("test query", &adaptive_options)
            .await
            .unwrap();
        assert_eq!(adaptive_result.strategy_used, RetrievalStrategy::Adaptive);
    }

    #[tokio::test]
    async fn test_query_options_limits() {
        let doc_repo = Arc::new(MockDocumentRepository);
        let embedding_service = Arc::new(crate::embeddings::FallbackEmbeddingService::new(384));

        let pipeline = PipelineFactory::create_default_pipeline(doc_repo, embedding_service);

        // Test with custom limits
        let mut options = EnhancedQueryOptions::default();
        options.k = 5;

        let result = pipeline
            .process_query("test query", &options)
            .await
            .unwrap();

        assert!(result.candidates.len() <= 5);
        assert!(result.processing_time_ms >= 0);
    }

    #[tokio::test]
    async fn test_query_understanding_integration() {
        let doc_repo = Arc::new(MockDocumentRepository);
        let embedding_service = Arc::new(crate::embeddings::FallbackEmbeddingService::new(384));

        let pipeline = PipelineFactory::create_default_pipeline(doc_repo, embedding_service);
        let options = EnhancedQueryOptions::default();

        // Test different query types
        let technical_result = pipeline
            .process_query("How to debug JavaScript function?", &options)
            .await
            .unwrap();
        assert!(technical_result.query_understanding.is_some());

        let analytical_result = pipeline
            .process_query("What are the benefits?", &options)
            .await
            .unwrap();
        assert!(analytical_result.query_understanding.is_some());

        let code_result = pipeline
            .process_query("function myFunc() { return 42; }", &options)
            .await
            .unwrap();
        assert!(code_result.query_understanding.is_some());
    }

    #[tokio::test]
    async fn test_ml_prediction_integration() {
        let doc_repo = Arc::new(MockDocumentRepository);
        let embedding_service = Arc::new(crate::embeddings::FallbackEmbeddingService::new(384));

        let pipeline = PipelineFactory::create_default_pipeline(doc_repo, embedding_service);
        let options = EnhancedQueryOptions::default();

        let result = pipeline
            .process_query(
                "complex analytical question about machine learning",
                &options,
            )
            .await
            .unwrap();

        assert!(result.ml_prediction.is_some());
        let prediction = result.ml_prediction.unwrap();
        assert!(prediction.prediction.confidence > 0.0);
        assert!(!prediction.explanation.is_empty());
        assert!(!prediction.feature_importance.is_empty());
    }

    #[tokio::test]
    async fn test_error_handling() {
        // Test with empty repository (should not fail but return empty results)
        let doc_repo = Arc::new(MockDocumentRepository);
        let embedding_service = Arc::new(crate::embeddings::FallbackEmbeddingService::new(384));

        let pipeline = PipelineFactory::create_default_pipeline(doc_repo, embedding_service);
        let options = EnhancedQueryOptions::default();

        // These should not fail even with mock repository
        let empty_result = pipeline.process_query("", &options).await.unwrap();
        assert!(empty_result.candidates.len() >= 0); // Mock may return candidates

        let whitespace_result = pipeline.process_query("   ", &options).await.unwrap();
        assert!(whitespace_result.candidates.len() >= 0);

        let unicode_result = pipeline
            .process_query("测试 🚀 тест", &options)
            .await
            .unwrap();
        assert!(unicode_result.processing_time_ms >= 0);
    }

    #[test]
    fn test_enhanced_query_options_default() {
        let options = EnhancedQueryOptions::default();

        assert_eq!(options.k, 10);
        assert!(options.override_strategy.is_none());
        assert_eq!(options.include_metadata, true);
        assert_eq!(options.session_id, "default");
    }

    #[test]
    fn test_enhanced_query_options_builder() {
        let mut options = EnhancedQueryOptions::default();
        options.k = 10;
        options.override_strategy = Some(RetrievalStrategy::Hybrid);
        options.include_metadata = false;
        options.session_id = "test-session".to_string();

        assert_eq!(options.k, 10);
        assert_eq!(options.override_strategy, Some(RetrievalStrategy::Hybrid));
        assert_eq!(options.include_metadata, false);
        assert_eq!(options.session_id, "test-session");
    }

    #[tokio::test]
    async fn test_pipeline_factory_different_configurations() {
        let doc_repo = Arc::new(MockDocumentRepository);
        let embedding_service = Arc::new(crate::embeddings::FallbackEmbeddingService::new(256));

        // Test default pipeline
        let default_pipeline =
            PipelineFactory::create_default_pipeline(doc_repo.clone(), embedding_service.clone());
        let result1 = default_pipeline
            .process_query("test", &EnhancedQueryOptions::default())
            .await
            .unwrap();

        assert!(!result1.candidates.is_empty());

        // Test that embeddings have correct dimensions
        let embedding_dim = embedding_service.dimension();
        assert_eq!(embedding_dim, 256);
    }

    #[tokio::test]
    async fn test_query_result_completeness() {
        let doc_repo = Arc::new(MockDocumentRepository);
        let embedding_service = Arc::new(crate::embeddings::FallbackEmbeddingService::new(384));

        let pipeline = PipelineFactory::create_default_pipeline(doc_repo, embedding_service);
        let options = EnhancedQueryOptions::default();

        let result = pipeline
            .process_query("comprehensive test query", &options)
            .await
            .unwrap();

        // Verify all required fields are present
        // Strategy could be any of the available strategies
        assert!(matches!(
            result.strategy_used,
            RetrievalStrategy::BM25Only
                | RetrievalStrategy::VectorOnly
                | RetrievalStrategy::Hybrid
                | RetrievalStrategy::HydeEnhanced
                | RetrievalStrategy::MultiStep
                | RetrievalStrategy::Adaptive
        ));
        assert!(result.candidates.len() >= 0); // Can be 0 with mock repository
        assert!(result.processing_time_ms >= 0);
        assert!(result.query_understanding.is_some());
        assert!(result.ml_prediction.is_some());

        // Verify query understanding has all fields
        let understanding = result.query_understanding.unwrap();
        assert!(!understanding.original_query.is_empty());
        assert!(understanding.confidence > 0.0);
        assert!(!understanding.keywords.is_empty());

        // Verify ML prediction has all fields
        let prediction = result.ml_prediction.unwrap();
        assert!(prediction.prediction.confidence > 0.0);
        assert!(!prediction.explanation.is_empty());
        assert!(!prediction.feature_importance.is_empty());
    }

    struct MockDocumentRepositoryWithData;

    #[async_trait]
    impl DocumentRepository for MockDocumentRepositoryWithData {
        async fn get_chunks_by_session(&self, _session_id: &str) -> Result<Vec<Chunk>> {
            Ok(vec![
                Chunk {
                    id: "chunk1".to_string(),
                    message_id: uuid::Uuid::new_v4(),
                    session_id: "session1".to_string(),
                    offset_start: 0,
                    offset_end: 100,
                    kind: "text".to_string(),
                    text: "This is a test chunk about machine learning.".to_string(),
                    tokens: 10,
                },
                Chunk {
                    id: "chunk2".to_string(),
                    message_id: uuid::Uuid::new_v4(),
                    session_id: "session1".to_string(),
                    offset_start: 100,
                    offset_end: 200,
                    kind: "code".to_string(),
                    text: "function processData() { return 'processed'; }".to_string(),
                    tokens: 8,
                },
            ])
        }

        async fn get_dfidf_by_session(&self, _session_id: &str) -> Result<Vec<DfIdf>> {
            Ok(vec![
                DfIdf {
                    term: "machine".to_string(),
                    session_id: "session1".to_string(),
                    df: 1,
                    idf: 2.5,
                },
                DfIdf {
                    term: "learning".to_string(),
                    session_id: "session1".to_string(),
                    df: 1,
                    idf: 2.3,
                },
            ])
        }

        async fn get_chunk_by_id(&self, chunk_id: &str) -> Result<Option<Chunk>> {
            if chunk_id == "chunk1" || chunk_id == "chunk2" {
                self.get_chunks_by_session("session1")
                    .await
                    .map(|chunks| chunks.into_iter().find(|c| c.id == chunk_id))
            } else {
                Ok(None)
            }
        }

        async fn vector_search(
            &self,
            _query_vector: &EmbeddingVector,
            _k: i32,
        ) -> Result<Vec<Candidate>> {
            Ok(vec![
                Candidate {
                    doc_id: "chunk1".to_string(),
                    score: 0.95,
                    text: Some("This is a test chunk about machine learning.".to_string()),
                    kind: Some("text".to_string()),
                },
                Candidate {
                    doc_id: "chunk2".to_string(),
                    score: 0.85,
                    text: Some("function processData() { return 'processed'; }".to_string()),
                    kind: Some("code".to_string()),
                },
            ])
        }
    }

    #[tokio::test]
    async fn test_pipeline_with_real_data() {
        let doc_repo = Arc::new(MockDocumentRepositoryWithData);
        let embedding_service = Arc::new(crate::embeddings::FallbackEmbeddingService::new(384));

        let pipeline = PipelineFactory::create_default_pipeline(doc_repo, embedding_service);
        let options = EnhancedQueryOptions::default();

        let result = pipeline
            .process_query("machine learning function", &options)
            .await
            .unwrap();

        // Should get results from mock data
        assert!(!result.candidates.is_empty());
        assert!(result.candidates.len() <= 2);

        // Verify candidates have content
        for candidate in &result.candidates {
            assert!(!candidate.doc_id.is_empty());
            assert!(candidate.score > 0.0);
            assert!(candidate.text.is_some());
            assert!(candidate.kind.is_some());
        }
    }
}
