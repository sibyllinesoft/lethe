use crate::{
    embeddings::EmbeddingService,
    hyde::{HydeConfig, HydeExpansion, HydeService, LlmService},
    ml_prediction::{MLPredictionResult, MLPredictionService, RetrievalStrategy},
    query_understanding::{QueryUnderstanding, QueryUnderstandingService},
    retrieval::{
        Bm25SearchService, DocumentRepository, HybridRetrievalConfig, HybridRetrievalService,
    },
};
use async_trait::async_trait;
use lethe_shared::{
    utils::TextProcessor, Candidate, Citation, ContextChunk, ContextPack, LetheError,
    ResolvedLetheConfig, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
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

impl PipelineConfig {
    /// Build a pipeline configuration using the resolved application settings.
    pub fn from_resolved_config(config: &ResolvedLetheConfig) -> Self {
        let mut resolved = Self::default();
        resolved.enable_hyde = config.features.enable_hyde;
        resolved.enable_query_understanding = config.query_understanding.enabled;
        resolved.enable_ml_prediction = config.ml.enabled;
        resolved.rerank_enabled = config.features.enable_state_tracking;
        resolved.timeout_seconds = (config.timeouts.hyde_ms.value() / 1000).max(1);
        resolved
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

#[async_trait]
pub trait PairwiseScoringModel: Send + Sync {
    async fn score_batch(&self, query: &str, candidates: &[Candidate]) -> Result<Vec<f64>>;
}

struct EmbeddingSimilarityModel {
    embedding_service: Arc<dyn EmbeddingService>,
}

impl EmbeddingSimilarityModel {
    fn new(embedding_service: Arc<dyn EmbeddingService>) -> Self {
        Self { embedding_service }
    }
}

#[async_trait]
impl PairwiseScoringModel for EmbeddingSimilarityModel {
    async fn score_batch(&self, query: &str, candidates: &[Candidate]) -> Result<Vec<f64>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let query_embedding = self
            .embedding_service
            .embed(&[query.to_string()])
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| {
                LetheError::embedding("Embedding service returned no vector for reranker query")
            })?;

        let mut scores = vec![0.0; candidates.len()];
        let mut texts = Vec::new();
        let mut indices = Vec::new();

        for (idx, candidate) in candidates.iter().enumerate() {
            if let Some(text) = candidate.text.as_ref() {
                if !text.trim().is_empty() {
                    texts.push(text.clone());
                    indices.push(idx);
                }
            }
        }

        if texts.is_empty() {
            return Ok(scores);
        }

        let embeddings = self.embedding_service.embed(&texts).await?;
        for (embedding_idx, &candidate_idx) in indices.iter().enumerate() {
            if let Some(candidate_embedding) = embeddings.get(embedding_idx) {
                let similarity =
                    cosine_similarity_vec(&query_embedding.data, &candidate_embedding.data);
                scores[candidate_idx] = (similarity + 1.0) / 2.0;
            }
        }

        Ok(scores)
    }
}

fn cosine_similarity_vec(a: &[f32], b: &[f32]) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();

    let norm_a = a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

/// Trait for reranking services
#[async_trait]
pub trait RerankingService: Send + Sync {
    async fn rerank(&self, query: &str, candidates: &[Candidate]) -> Result<Vec<Candidate>>;
}

/// Reranking implementation that refines scores using embedding similarity
pub struct EmbeddingRerankingService {
    model: Arc<dyn PairwiseScoringModel>,
    weight_original: f64,
    weight_similarity: f64,
}

impl EmbeddingRerankingService {
    /// Create a reranker with default weighting (70% similarity, 30% original score)
    pub fn new(embedding_service: Arc<dyn EmbeddingService>) -> Self {
        Self::with_weights(embedding_service, 0.3, 0.7)
    }

    /// Create a reranker with custom weighting
    pub fn with_weights(
        embedding_service: Arc<dyn EmbeddingService>,
        weight_original: f64,
        weight_similarity: f64,
    ) -> Self {
        let model = Arc::new(EmbeddingSimilarityModel::new(embedding_service));
        Self::with_model(model, weight_original, weight_similarity)
    }

    pub fn with_model(
        model: Arc<dyn PairwiseScoringModel>,
        weight_original: f64,
        weight_similarity: f64,
    ) -> Self {
        let total = weight_original + weight_similarity;
        let (weight_original, weight_similarity) = if total > 0.0 {
            (weight_original / total, weight_similarity / total)
        } else {
            (0.5, 0.5)
        };

        Self {
            model,
            weight_original,
            weight_similarity,
        }
    }
}

#[async_trait]
impl RerankingService for EmbeddingRerankingService {
    async fn rerank(&self, query: &str, candidates: &[Candidate]) -> Result<Vec<Candidate>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let max_original_score = candidates
            .iter()
            .map(|c| c.score.abs())
            .fold(0.0_f64, f64::max)
            .max(1e-6);

        let mut reranked = candidates.to_vec();
        let model_scores = self.model.score_batch(query, &reranked).await?;

        for (candidate, similarity) in reranked.iter_mut().zip(model_scores.into_iter()) {
            let similarity = similarity.clamp(0.0, 1.0);
            let original_normalised = (candidate.score / max_original_score).clamp(0.0, 1.0);
            let combined =
                self.weight_similarity * similarity + self.weight_original * original_normalised;
            candidate.score = combined * max_original_score;
        }

        reranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(reranked)
    }
}

/// Enhanced query pipeline that orchestrates all components
pub struct EnhancedQueryPipeline {
    config: PipelineConfig,
    document_repository: Arc<dyn DocumentRepository>,
    embedding_service: Arc<dyn EmbeddingService>,
    llm_service: Option<Arc<dyn LlmService>>,
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
        ml_prediction: Option<MLPredictionService>,
    ) -> Self {
        let hybrid_config = HybridRetrievalConfig::default();
        let hybrid_retrieval =
            HybridRetrievalService::new(embedding_service.clone(), hybrid_config);

        let hyde_service = if config.enable_hyde {
            llm_service.as_ref().map(|llm| {
                Arc::new(HydeService::new(
                    Arc::clone(llm),
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
            llm_service,
            hybrid_retrieval,
            hyde_service,
            query_understanding: QueryUnderstandingService::new(),
            ml_prediction: ml_prediction.unwrap_or_else(MLPredictionService::default),
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
            .phase_context_creation(query, &final_candidates, options)
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
                let mut iter = candidates.into_iter();
                let top_candidates: Vec<Candidate> =
                    iter.by_ref().take(self.config.rerank_top_k).collect();
                let remainder: Vec<Candidate> = iter.collect();

                if top_candidates.is_empty() {
                    Ok(remainder)
                } else {
                    let mut reranked = reranker.rerank(query, &top_candidates).await?;
                    reranked.extend(remainder);
                    Ok(reranked)
                }
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
        query: &str,
        candidates: &[Candidate],
        options: &EnhancedQueryOptions,
    ) -> Result<ContextPack> {
        self.create_context_pack(query, candidates, options).await
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
                self.hybrid_retrieval
                    .vector_search_with_embedding(
                        &*self.document_repository,
                        &options.session_id,
                        &query_embedding,
                        self.config.max_candidates as i32,
                    )
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
                    self.execute_hyde_enhanced_search(query, &options.session_id, expansion)
                        .await
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
        session_id: &str,
        expansion: &HydeExpansion,
    ) -> Result<Vec<Candidate>> {
        if let Some(ref combined_embedding) = expansion.combined_embedding {
            return self
                .hybrid_retrieval
                .vector_search_with_embedding(
                    &*self.document_repository,
                    session_id,
                    combined_embedding,
                    self.config.max_candidates as i32,
                )
                .await;
        }

        let mut all_candidates = Vec::new();
        let docs_len = expansion.hypothetical_documents.len().max(1);
        let per_doc_k = (self.config.max_candidates / docs_len).max(1) as i32;

        for hyp_doc in &expansion.hypothetical_documents {
            if let Some(ref embedding) = hyp_doc.embedding {
                let candidates = self
                    .hybrid_retrieval
                    .vector_search_with_embedding(
                        &*self.document_repository,
                        session_id,
                        embedding,
                        per_doc_k,
                    )
                    .await?;
                all_candidates.extend(candidates);
            }
        }

        let original_candidates = self
            .hybrid_retrieval
            .retrieve(&*self.document_repository, &[query.to_string()], session_id)
            .await?;
        all_candidates.extend(original_candidates);

        self.deduplicate_and_sort_candidates(all_candidates)
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
            self.hybrid_retrieval
                .vector_search_with_embedding(
                    &*self.document_repository,
                    &options.session_id,
                    &query_embedding,
                    self.config.max_candidates as i32,
                )
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
            self.hybrid_retrieval
                .vector_search_with_embedding(
                    &*self.document_repository,
                    &options.session_id,
                    &query_embedding,
                    self.config.max_candidates as i32,
                )
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
        query: &str,
        candidates: &[Candidate],
        options: &EnhancedQueryOptions,
    ) -> Result<ContextPack> {
        let chunks: Vec<ContextChunk> = candidates
            .iter()
            .map(|candidate| ContextChunk {
                id: candidate.doc_id.clone(),
                score: candidate.score,
                kind: candidate.kind.clone().unwrap_or_else(|| "text".to_string()),
                text: candidate.text.clone().unwrap_or_default(),
            })
            .collect();

        let llm_analysis = self.analyse_context_with_llm(query, &chunks).await;

        let summary = llm_analysis
            .as_ref()
            .map(|analysis| analysis.summary.clone())
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| self.build_summary(query, &chunks));

        let key_entities = llm_analysis
            .as_ref()
            .map(|analysis| analysis.key_entities.clone())
            .filter(|items| !items.is_empty())
            .unwrap_or_else(|| self.extract_key_entities(&chunks));
        let key_entities = Self::clean_string_list_with_limit(key_entities, 8);

        let claims = llm_analysis
            .as_ref()
            .map(|analysis| analysis.claims.clone())
            .filter(|items| !items.is_empty())
            .unwrap_or_else(|| self.extract_claims(&chunks));
        let claims = Self::clean_string_list_with_limit(claims, 5);

        let contradictions = llm_analysis
            .as_ref()
            .map(|analysis| analysis.contradictions.clone())
            .filter(|items| !items.is_empty())
            .unwrap_or_else(|| self.extract_contradictions(&chunks));
        let contradictions = Self::clean_string_list_with_limit(contradictions, 3);

        let citations = self.build_citations(&chunks);

        Ok(ContextPack {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: options.session_id.clone(),
            query: query.to_string(),
            created_at: chrono::Utc::now(),
            summary: summary.trim().to_string(),
            key_entities,
            claims,
            contradictions,
            chunks,
            citations,
        })
    }

    fn build_summary(&self, query: &str, chunks: &[ContextChunk]) -> String {
        if chunks.is_empty() {
            return String::new();
        }

        let mut sentences: Vec<(String, f64)> = Vec::new();
        for chunk in chunks.iter().take(5) {
            if chunk.text.trim().is_empty() {
                continue;
            }
            for sentence in TextProcessor::split_sentences(&chunk.text) {
                let trimmed = sentence.trim();
                if trimmed.is_empty() {
                    continue;
                }
                sentences.push((trimmed.to_string(), chunk.score));
            }
        }

        sentences.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sentences.truncate(3);

        if sentences.is_empty() {
            let fallback = chunks
                .iter()
                .max_by(|a, b| {
                    a.score
                        .partial_cmp(&b.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|chunk| chunk.text.trim().chars().take(280).collect::<String>())
                .filter(|text| !text.is_empty())
                .unwrap_or_default();

            return if fallback.is_empty() {
                format!("Context for query: {}", query)
            } else {
                fallback
            };
        }

        sentences
            .into_iter()
            .map(|(sentence, _)| sentence)
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn extract_key_entities(&self, chunks: &[ContextChunk]) -> Vec<String> {
        let mut entities = Vec::new();
        let mut seen = HashSet::new();

        for chunk in chunks.iter().take(5) {
            for token in chunk.text.split_whitespace() {
                let cleaned = token
                    .trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_')
                    .trim_matches('"');
                if cleaned.len() < 3 {
                    continue;
                }

                let mut chars = cleaned.chars();
                let first = match chars.next() {
                    Some(ch) => ch,
                    None => continue,
                };

                if !first.is_uppercase() {
                    continue;
                }

                if cleaned.chars().any(char::is_lowercase)
                    || cleaned.chars().all(char::is_uppercase)
                {
                    let candidate = cleaned.to_string();
                    if seen.insert(candidate.clone()) {
                        entities.push(candidate);
                    }
                }

                if entities.len() >= 8 {
                    return entities;
                }
            }
        }

        entities
    }

    fn extract_claims(&self, chunks: &[ContextChunk]) -> Vec<String> {
        self.collect_sentences(chunks, &|sentence: &str| {
            let lower = sentence.to_lowercase();
            lower.contains("should")
                || lower.contains("must")
                || lower.contains("need to")
                || lower.contains("recommended")
                || lower.contains("best practice")
        })
    }

    fn extract_contradictions(&self, chunks: &[ContextChunk]) -> Vec<String> {
        self.collect_sentences(chunks, &|sentence: &str| {
            let lower = sentence.to_lowercase();
            (lower.contains("however") || lower.contains("but ") || lower.contains("in contrast"))
                && lower.contains("not")
        })
    }

    fn collect_sentences(
        &self,
        chunks: &[ContextChunk],
        predicate: &dyn Fn(&str) -> bool,
    ) -> Vec<String> {
        let mut matches = Vec::new();
        let mut seen = HashSet::new();

        for chunk in chunks.iter().take(6) {
            for sentence in TextProcessor::split_sentences(&chunk.text) {
                let normalized = sentence.trim();
                if normalized.is_empty() {
                    continue;
                }

                if predicate(normalized) {
                    let key = normalized.to_lowercase();
                    if seen.insert(key) {
                        matches.push(normalized.to_string());
                    }
                }

                if matches.len() >= 5 {
                    return matches;
                }
            }
        }

        matches
    }

    fn build_citations(&self, chunks: &[ContextChunk]) -> Vec<Citation> {
        chunks
            .iter()
            .enumerate()
            .map(|(idx, chunk)| Citation {
                id: idx as i32,
                chunk_id: chunk.id.clone(),
                relevance: chunk.score,
            })
            .collect()
    }

    async fn analyse_context_with_llm(
        &self,
        query: &str,
        chunks: &[ContextChunk],
    ) -> Option<LlmContextAnalysisResult> {
        let llm = Arc::clone(self.llm_service.as_ref()?);

        let prompt = self.build_context_summary_prompt(query, chunks)?;
        let config = HydeConfig {
            num_documents: 1,
            temperature: 0.2,
            max_tokens: 512,
            combine_with_query: false,
        };

        let responses = match llm.generate_text(&prompt, &config).await {
            Ok(value) => value,
            Err(err) => {
                tracing::warn!(error = %err, "LLM context analysis request failed");
                return None;
            }
        };

        let raw_response = responses.into_iter().find(|r| !r.trim().is_empty())?;
        let json_candidate =
            Self::extract_json_payload(&raw_response).unwrap_or_else(|| raw_response.clone());

        let payload: LlmContextAnalysisPayload = match serde_json::from_str(&json_candidate) {
            Ok(payload) => payload,
            Err(err) => {
                tracing::warn!(error = %err, "Failed to parse LLM context analysis JSON");
                return None;
            }
        };

        let result = Self::transform_llm_payload(payload);
        if result.is_meaningful() {
            Some(result)
        } else {
            None
        }
    }

    fn build_context_summary_prompt(&self, query: &str, chunks: &[ContextChunk]) -> Option<String> {
        let mut sections = Vec::new();

        for (idx, chunk) in chunks.iter().take(6).enumerate() {
            let text = chunk.text.trim();
            if text.is_empty() {
                continue;
            }

            let snippet: String = text.chars().take(800).collect();
            let kind = chunk.kind.as_str();
            sections.push(format!(
                "Chunk {idx} (score {score:.3}, kind {kind}):\n{snippet}",
                idx = idx + 1,
                score = chunk.score,
                kind = kind,
                snippet = snippet
            ));
        }

        if sections.is_empty() {
            return None;
        }

        let evidence = sections.join("\n\n");
        let prompt = format!(
            "You are assisting a retrieval-augmented generation system.\n\nSummary task:\n- Provide a concise paragraph summary (max 4 sentences).\n- Extract up to 6 key entities or proper nouns.\n- List up to 5 actionable claims or recommendations.\n- Note up to 3 contradictions, caveats, or disagreements.\n\nRespond *only* with JSON using this schema:\n{{\"summary\": string, \"key_entities\": [string], \"claims\": [string], \"contradictions\": [string]}}.\nUse double quotes for every string and keep list items short.\n\nUser query: {query}\n\nEvidence:\n{evidence}",
        );

        Some(prompt)
    }

    fn extract_json_payload(response: &str) -> Option<String> {
        let trimmed = response.trim();
        let trimmed = if let Some(stripped) = trimmed.strip_prefix("```json") {
            stripped.trim_start()
        } else if let Some(stripped) = trimmed.strip_prefix("```") {
            stripped.trim_start()
        } else {
            trimmed
        };

        let trimmed = if let Some(stripped) = trimmed.strip_suffix("```") {
            stripped.trim_end()
        } else {
            trimmed
        };

        let start = trimmed.find('{')?;
        let end = trimmed.rfind('}')?;
        if end <= start {
            return None;
        }

        Some(trimmed[start..=end].to_string())
    }

    fn clean_string_list_with_limit(raw: Vec<String>, limit: usize) -> Vec<String> {
        let mut cleaned = Vec::new();
        let mut seen = HashSet::new();

        for item in raw {
            let trimmed = item
                .trim()
                .trim_matches(|c: char| c == '"' || c == '\'' || c == '`')
                .trim();
            if trimmed.is_empty() {
                continue;
            }

            let canonical = trimmed.to_ascii_lowercase();
            if seen.insert(canonical) {
                cleaned.push(trimmed.to_string());
            }
            if cleaned.len() >= limit {
                break;
            }
        }

        cleaned
    }

    fn transform_llm_payload(payload: LlmContextAnalysisPayload) -> LlmContextAnalysisResult {
        let summary = payload
            .summary
            .unwrap_or_default()
            .trim()
            .chars()
            .take(600)
            .collect::<String>();

        let key_entities =
            Self::clean_string_list_with_limit(payload.key_entities.unwrap_or_default(), 8);
        let claims = Self::clean_string_list_with_limit(payload.claims.unwrap_or_default(), 5);
        let contradictions =
            Self::clean_string_list_with_limit(payload.contradictions.unwrap_or_default(), 3);

        LlmContextAnalysisResult {
            summary,
            key_entities,
            claims,
            contradictions,
        }
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

#[derive(Debug, Deserialize)]
struct LlmContextAnalysisPayload {
    summary: Option<String>,
    key_entities: Option<Vec<String>>,
    claims: Option<Vec<String>>,
    contradictions: Option<Vec<String>>,
}

#[derive(Debug)]
struct LlmContextAnalysisResult {
    summary: String,
    key_entities: Vec<String>,
    claims: Vec<String>,
    contradictions: Vec<String>,
}

impl LlmContextAnalysisResult {
    fn is_meaningful(&self) -> bool {
        !self.summary.trim().is_empty()
            || !self.key_entities.is_empty()
            || !self.claims.is_empty()
            || !self.contradictions.is_empty()
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
        ml_prediction: Option<MLPredictionService>,
    ) -> EnhancedQueryPipeline {
        EnhancedQueryPipeline::new(
            config,
            document_repository,
            embedding_service,
            llm_service,
            reranking_service,
            ml_prediction,
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
            None,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lethe_shared::{Candidate, Chunk, DfIdf, EmbeddingVector};
    use std::sync::Arc;
    use uuid::Uuid;

    fn mock_chunks(session_id: &str) -> Vec<Chunk> {
        vec![
            Chunk {
                id: format!("{}::alpha", session_id),
                message_id: Uuid::nil(),
                session_id: session_id.to_string(),
                offset_start: 0,
                offset_end: 50,
                kind: "text".to_string(),
                text: "Alpha topic detailed explanation".to_string(),
                tokens: 5,
            },
            Chunk {
                id: format!("{}::beta", session_id),
                message_id: Uuid::nil(),
                session_id: session_id.to_string(),
                offset_start: 51,
                offset_end: 120,
                kind: "text".to_string(),
                text: "Beta topic overview with examples".to_string(),
                tokens: 6,
            },
        ]
    }

    fn chunk_by_id(chunk_id: &str) -> Option<Chunk> {
        if let Some((session_id, _)) = chunk_id.split_once("::") {
            mock_chunks(session_id)
                .into_iter()
                .find(|chunk| chunk.id == chunk_id)
        } else {
            None
        }
    }

    struct MockDocumentRepository;

    #[async_trait]
    impl DocumentRepository for MockDocumentRepository {
        async fn get_chunks_by_session(&self, session_id: &str) -> Result<Vec<Chunk>> {
            Ok(mock_chunks(session_id))
        }

        async fn get_dfidf_by_session(&self, session_id: &str) -> Result<Vec<DfIdf>> {
            Ok(vec![
                DfIdf {
                    term: "alpha".to_string(),
                    session_id: session_id.to_string(),
                    df: 1,
                    idf: 1.2,
                },
                DfIdf {
                    term: "beta".to_string(),
                    session_id: session_id.to_string(),
                    df: 1,
                    idf: 1.0,
                },
            ])
        }

        async fn get_chunk_by_id(&self, chunk_id: &str) -> Result<Option<Chunk>> {
            Ok(chunk_by_id(chunk_id))
        }

        async fn vector_search(
            &self,
            _query_vector: &EmbeddingVector,
            _k: i32,
        ) -> Result<Vec<Candidate>> {
            Ok(mock_chunks("default")
                .into_iter()
                .map(|chunk| Candidate {
                    doc_id: chunk.id,
                    score: 0.5,
                    text: Some(chunk.text),
                    kind: Some(chunk.kind),
                })
                .collect())
        }
    }

    #[tokio::test]
    async fn test_pipeline_creation() {
        let doc_repo = Arc::new(MockDocumentRepository);
        let embedding_service = Arc::new(StubEmbeddingService);

        let pipeline = PipelineFactory::create_default_pipeline(doc_repo, embedding_service);

        // Test that the pipeline was created successfully
        assert!(pipeline.config.enable_query_understanding);
        assert!(pipeline.config.enable_ml_prediction);
    }

    #[tokio::test]
    async fn test_basic_query_processing() {
        let doc_repo = Arc::new(MockDocumentRepository);
        let embedding_service = Arc::new(StubEmbeddingService);

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

    struct StubEmbeddingService;

    #[async_trait]
    impl EmbeddingService for StubEmbeddingService {
        fn name(&self) -> &str {
            "stub"
        }

        fn dimension(&self) -> usize {
            2
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>> {
            let vectors = texts
                .iter()
                .map(|text| {
                    if text.to_lowercase().contains("alpha") {
                        EmbeddingVector {
                            data: vec![1.0, 0.0],
                            dimension: 2,
                        }
                    } else if text.to_lowercase().contains("beta") {
                        EmbeddingVector {
                            data: vec![0.0, 1.0],
                            dimension: 2,
                        }
                    } else {
                        EmbeddingVector {
                            data: vec![0.5, 0.5],
                            dimension: 2,
                        }
                    }
                })
                .collect();

            Ok(vectors)
        }
    }

    #[tokio::test]
    async fn test_embedding_reranker_prioritises_semantically_relevant_results() {
        let embedding_service = Arc::new(StubEmbeddingService);
        let reranker = EmbeddingRerankingService::new(embedding_service);

        let query = "alpha topic";
        let candidates = vec![
            Candidate {
                doc_id: "doc-alpha".to_string(),
                score: 0.2,
                text: Some("Detailed alpha description".to_string()),
                kind: None,
            },
            Candidate {
                doc_id: "doc-beta".to_string(),
                score: 0.9,
                text: Some("Unrelated beta content".to_string()),
                kind: None,
            },
        ];

        let reranked = reranker.rerank(query, &candidates).await.unwrap();

        assert_eq!(reranked.len(), 2);
        assert_eq!(reranked[0].doc_id, "doc-alpha");
    }

    #[tokio::test]
    async fn test_strategy_override() {
        let doc_repo = Arc::new(MockDocumentRepository);
        let embedding_service = Arc::new(StubEmbeddingService);

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
        let embedding_service = Arc::new(StubEmbeddingService);

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
        let embedding_service = Arc::new(StubEmbeddingService);

        let pipeline = PipelineFactory::create_default_pipeline(doc_repo, embedding_service);

        // Test with custom limits
        let mut options = EnhancedQueryOptions::default();
        options.k = 5;

        let result = pipeline
            .process_query("test query", &options)
            .await
            .unwrap();

        assert!(result.candidates.len() <= 5);
        assert!(result.processing_time_ms <= pipeline.config.timeout_seconds * 1_000);
    }

    #[tokio::test]
    async fn test_query_understanding_integration() {
        let doc_repo = Arc::new(MockDocumentRepository);
        let embedding_service = Arc::new(StubEmbeddingService);

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
        let embedding_service = Arc::new(StubEmbeddingService);

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
        let embedding_service = Arc::new(StubEmbeddingService);

        let pipeline = PipelineFactory::create_default_pipeline(doc_repo, embedding_service);
        let options = EnhancedQueryOptions::default();

        // These should not fail even with mock repository
        let empty_result = pipeline.process_query("", &options).await.unwrap();
        assert!(empty_result.candidates.len() <= pipeline.config.max_candidates);

        let whitespace_result = pipeline.process_query("   ", &options).await.unwrap();
        assert!(whitespace_result.candidates.len() <= pipeline.config.max_candidates);

        let unicode_result = pipeline
            .process_query("测试 🚀 тест", &options)
            .await
            .unwrap();
        assert!(unicode_result.candidates.len() <= pipeline.config.max_candidates);
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
        let embedding_service = Arc::new(StubEmbeddingService);

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
        assert!(result.candidates.len() <= pipeline.config.max_candidates);
        assert!(result.processing_time_ms <= pipeline.config.timeout_seconds * 1_000);
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
                    message_id: Uuid::new_v4(),
                    session_id: "session1".to_string(),
                    offset_start: 0,
                    offset_end: 100,
                    kind: "text".to_string(),
                    text: "This is a test chunk about machine learning.".to_string(),
                    tokens: 10,
                },
                Chunk {
                    id: "chunk2".to_string(),
                    message_id: Uuid::new_v4(),
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
        let embedding_service = Arc::new(StubEmbeddingService);

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
