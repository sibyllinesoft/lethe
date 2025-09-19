use crate::embeddings::EmbeddingService;
use async_trait::async_trait;
use hnsw_rs::prelude::*;
use lethe_shared::utils::{QueryFeatures, TextProcessor};
use lethe_shared::{Candidate, Chunk, DfIdf, EmbeddingVector, LetheError, Result};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Configuration for hybrid retrieval
#[derive(Debug, Clone)]
pub struct HybridRetrievalConfig {
    pub alpha: f64,                             // Weight for lexical (BM25) score
    pub beta: f64,                              // Weight for vector score
    pub gamma_kind_boost: HashMap<String, f64>, // Boost for specific content types
    pub rerank: bool,                           // Enable reranking
    pub diversify: bool,                        // Enable diversification
    pub diversify_method: String,               // Diversification method
    pub diversify_lambda: f64,                  // Trade-off parameter for diversification
    pub k_initial: i32,                         // Initial retrieval size
    pub k_final: i32,                           // Final result size
    pub fusion_dynamic: bool,                   // Enable dynamic fusion
}

impl Default for HybridRetrievalConfig {
    fn default() -> Self {
        let mut gamma_kind_boost = HashMap::new();
        gamma_kind_boost.insert("code".to_string(), 1.2);
        gamma_kind_boost.insert("import".to_string(), 1.1);
        gamma_kind_boost.insert("function".to_string(), 1.15);
        gamma_kind_boost.insert("error".to_string(), 1.3);

        Self {
            alpha: 0.7,
            beta: 0.3,
            gamma_kind_boost,
            rerank: true,
            diversify: true,
            diversify_method: "entity".to_string(),
            diversify_lambda: 0.7,
            k_initial: 50,
            k_final: 20,
            fusion_dynamic: false,
        }
    }
}

const EPHEMERAL_HNSW_MAX_CONNECTIONS: usize = 16;
const EPHEMERAL_HNSW_MAX_LAYER: usize = 8;
const EPHEMERAL_HNSW_EF_CONSTRUCTION: usize = 64;
const EPHEMERAL_HNSW_BASE_EF_SEARCH: usize = 64;

/// Trait for document repositories
#[async_trait]
pub trait DocumentRepository: Send + Sync {
    /// Get all chunks for a session
    async fn get_chunks_by_session(&self, session_id: &str) -> Result<Vec<Chunk>>;

    /// Get DF-IDF data for a session
    async fn get_dfidf_by_session(&self, session_id: &str) -> Result<Vec<DfIdf>>;

    /// Get chunk by ID
    async fn get_chunk_by_id(&self, chunk_id: &str) -> Result<Option<Chunk>>;

    /// Search vectors by similarity
    async fn vector_search(&self, query_vector: &EmbeddingVector, k: i32)
        -> Result<Vec<Candidate>>;

    /// Optional prefiltering hook (e.g., bloom filters)
    async fn prefilter_chunks(
        &self,
        _session_id: &str,
        _terms: &[String],
    ) -> Result<Option<Vec<Chunk>>> {
        Ok(None)
    }
}

/// BM25 search service
pub struct Bm25SearchService;

impl Bm25SearchService {
    /// Search documents using BM25 algorithm
    pub async fn search<R: DocumentRepository + ?Sized>(
        repository: &R,
        queries: &[String],
        session_id: &str,
        k: i32,
    ) -> Result<Vec<Candidate>> {
        let mut chunks = repository.get_chunks_by_session(session_id).await?;
        if chunks.is_empty() {
            return Ok(vec![]);
        }

        let dfidf_data = repository.get_dfidf_by_session(session_id).await?;
        let term_idf_map: HashMap<String, f64> = dfidf_data
            .into_iter()
            .map(|entry| (entry.term, entry.idf))
            .collect();

        // Calculate average document length
        let total_length: i32 = chunks
            .iter()
            .map(|chunk| Self::tokenize(&chunk.text).len() as i32)
            .sum();
        let avg_doc_length = if chunks.is_empty() {
            0.0
        } else {
            total_length as f64 / chunks.len() as f64
        };

        // Combine all query terms
        let all_query_terms: HashSet<String> = queries
            .iter()
            .flat_map(|query| Self::tokenize(query))
            .collect();

        let term_vec: Vec<String> = all_query_terms.iter().cloned().collect();
        if let Some(prefiltered) = repository.prefilter_chunks(session_id, &term_vec).await? {
            chunks = prefiltered;
            if chunks.is_empty() {
                return Ok(vec![]);
            }
        }

        // Score each chunk
        let mut candidates = Vec::new();

        for chunk in chunks {
            let doc_terms = Self::tokenize(&chunk.text);
            let doc_length = doc_terms.len() as f64;

            // Calculate term frequencies for query terms only
            let mut term_freqs = HashMap::new();
            for term in &doc_terms {
                if all_query_terms.contains(term) {
                    *term_freqs.entry(term.clone()).or_insert(0) += 1;
                }
            }

            // Skip documents with no query terms
            if term_freqs.is_empty() {
                continue;
            }

            let score = Self::calculate_bm25(
                &term_freqs,
                doc_length,
                avg_doc_length,
                &term_idf_map,
                1.2,
                0.75,
            );
            if score > 0.0 {
                candidates.push(Candidate {
                    doc_id: chunk.id,
                    score,
                    text: Some(chunk.text),
                    kind: Some(chunk.kind),
                });
            }
        }

        // Sort by score descending and take top k
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        candidates.truncate(k as usize);

        Ok(candidates)
    }

    /// Tokenize text for BM25 processing
    fn tokenize(text: &str) -> Vec<String> {
        TextProcessor::tokenize(text)
    }

    /// Calculate BM25 score
    fn calculate_bm25(
        term_freqs: &HashMap<String, i32>,
        doc_length: f64,
        avg_doc_length: f64,
        term_idf_map: &HashMap<String, f64>,
        k1: f64,
        b: f64,
    ) -> f64 {
        let mut score = 0.0;

        for (term, &tf) in term_freqs {
            let idf = term_idf_map.get(term).copied().unwrap_or(0.0);
            if idf <= 0.0 {
                continue;
            }

            let numerator = (tf as f64) * (k1 + 1.0);
            let denominator = (tf as f64) + k1 * (1.0 - b + b * (doc_length / avg_doc_length));

            score += idf * (numerator / denominator);
        }

        score
    }

    /// Calculate BM25 score with default parameters
    #[allow(dead_code)]
    fn calculate_bm25_default(
        term_freqs: &HashMap<String, i32>,
        doc_length: f64,
        avg_doc_length: f64,
        term_idf_map: &HashMap<String, f64>,
    ) -> f64 {
        Self::calculate_bm25(
            term_freqs,
            doc_length,
            avg_doc_length,
            term_idf_map,
            1.2,
            0.75,
        )
    }
}

/// Vector search service
pub struct VectorSearchService {
    embedding_service: Arc<dyn EmbeddingService>,
}

impl VectorSearchService {
    pub fn new(embedding_service: Arc<dyn EmbeddingService>) -> Self {
        Self { embedding_service }
    }

    pub fn embedding_service(&self) -> Arc<dyn EmbeddingService> {
        Arc::clone(&self.embedding_service)
    }

    /// Search documents using vector similarity
    pub async fn search<R: DocumentRepository + ?Sized>(
        &self,
        repository: &R,
        session_id: &str,
        query: &str,
        k: i32,
    ) -> Result<Vec<Candidate>> {
        let query_embedding = self.embedding_service.embed_single(query).await?;
        self.search_with_embedding(repository, session_id, &query_embedding, k)
            .await
    }

    pub async fn search_with_embedding<R: DocumentRepository + ?Sized>(
        &self,
        repository: &R,
        session_id: &str,
        query_embedding: &EmbeddingVector,
        k: i32,
    ) -> Result<Vec<Candidate>> {
        if k <= 0 {
            return Ok(Vec::new());
        }

        let chunks = repository.get_chunks_by_session(session_id).await?;
        if chunks.is_empty() {
            return Ok(Vec::new());
        }

        let texts: Vec<String> = chunks.iter().map(|chunk| chunk.text.clone()).collect();
        let chunk_embeddings = self.embedding_service.embed(&texts).await?;
        if chunk_embeddings.is_empty() {
            return Ok(Vec::new());
        }

        let k = k.max(0) as usize;
        let query_vector = query_embedding.data.clone();
        let embeddings_for_index = chunk_embeddings.clone();

        let search_results = tokio::task::spawn_blocking(move || {
            hnsw_search(&embeddings_for_index, &query_vector, k)
        })
        .await
        .map_err(|err| LetheError::internal(format!("Vector search task failed: {}", err)))??;

        let mut dedup = HashSet::new();
        let mut candidates = Vec::with_capacity(search_results.len());

        for (idx, score) in search_results {
            if idx >= chunks.len() {
                continue;
            }

            let chunk = &chunks[idx];
            if !dedup.insert(chunk.id.clone()) {
                continue;
            }

            candidates.push(Candidate {
                doc_id: chunk.id.clone(),
                score,
                text: Some(chunk.text.clone()),
                kind: Some(chunk.kind.clone()),
            });
        }

        Ok(candidates)
    }
}

/// Hybrid retrieval service combining BM25 and vector search
pub struct HybridRetrievalService {
    vector_service: VectorSearchService,
    config: HybridRetrievalConfig,
}

impl HybridRetrievalService {
    pub fn new(
        embedding_service: Arc<dyn EmbeddingService>,
        config: HybridRetrievalConfig,
    ) -> Self {
        Self {
            vector_service: VectorSearchService::new(embedding_service),
            config,
        }
    }

    /// Perform hybrid retrieval combining lexical and semantic search
    pub async fn retrieve<R: DocumentRepository + ?Sized>(
        &self,
        repository: &R,
        queries: &[String],
        session_id: &str,
    ) -> Result<Vec<Candidate>> {
        let combined_query = queries.join(" ");

        tracing::info!("Starting hybrid retrieval for {} queries", queries.len());

        // Run BM25 and vector search in parallel
        let (lexical_results, vector_results) = tokio::try_join!(
            Bm25SearchService::search(repository, queries, session_id, self.config.k_initial),
            self.vector_service.search(
                repository,
                session_id,
                &combined_query,
                self.config.k_initial
            )
        )?;

        tracing::debug!(
            "BM25 found {} candidates, Vector search found {} candidates",
            lexical_results.len(),
            vector_results.len()
        );

        // Combine results using hybrid scoring
        let candidates = self.hybrid_score(lexical_results, vector_results, &combined_query)?;

        tracing::info!("Hybrid scoring produced {} candidates", candidates.len());

        // Apply post-processing (reranking, diversification)
        let final_candidates = self.post_process(repository, candidates).await?;

        tracing::info!("Final result: {} candidates", final_candidates.len());
        Ok(final_candidates)
    }

    pub async fn vector_search<R: DocumentRepository + ?Sized>(
        &self,
        repository: &R,
        session_id: &str,
        query: &str,
        k: i32,
    ) -> Result<Vec<Candidate>> {
        self.vector_service
            .search(repository, session_id, query, k)
            .await
    }

    pub async fn vector_search_with_embedding<R: DocumentRepository + ?Sized>(
        &self,
        repository: &R,
        session_id: &str,
        query_embedding: &EmbeddingVector,
        k: i32,
    ) -> Result<Vec<Candidate>> {
        self.vector_service
            .search_with_embedding(repository, session_id, query_embedding, k)
            .await
    }

    /// Combine lexical and vector results using hybrid scoring
    fn hybrid_score(
        &self,
        lexical_results: Vec<Candidate>,
        vector_results: Vec<Candidate>,
        query: &str,
    ) -> Result<Vec<Candidate>> {
        // Normalize scores
        let lexical_normalized = self.normalize_bm25_scores(lexical_results);
        let vector_normalized = self.normalize_cosine_scores(vector_results);

        // Create lookup maps
        let lexical_map: HashMap<String, f64> = lexical_normalized
            .into_iter()
            .map(|c| (c.doc_id, c.score))
            .collect();

        let vector_map: HashMap<String, f64> = vector_normalized
            .into_iter()
            .map(|c| (c.doc_id, c.score))
            .collect();

        // Get all unique document IDs
        let all_doc_ids: HashSet<String> = lexical_map
            .keys()
            .chain(vector_map.keys())
            .cloned()
            .collect();

        // Extract query features for dynamic gamma boosting
        let query_features = QueryFeatures::extract_features(query);

        let mut candidates = Vec::new();

        for doc_id in all_doc_ids {
            let lex_score = lexical_map.get(&doc_id).copied().unwrap_or(0.0);
            let vec_score = vector_map.get(&doc_id).copied().unwrap_or(0.0);

            // Calculate base hybrid score
            let mut hybrid_score = self.config.alpha * lex_score + self.config.beta * vec_score;

            // Apply gamma boost based on content kind (if available)
            // This would require getting the kind from the document, simplified here
            let kind = "text"; // Placeholder - would get from document
            let dynamic_boost = QueryFeatures::gamma_boost(kind, &query_features);
            let static_boost = self
                .config
                .gamma_kind_boost
                .get(kind)
                .copied()
                .unwrap_or(0.0);
            let total_boost = 1.0 + dynamic_boost + static_boost;
            hybrid_score *= total_boost;

            candidates.push(Candidate {
                doc_id,
                score: hybrid_score,
                text: None, // Will be enriched later if needed
                kind: Some(kind.to_string()),
            });
        }

        // Sort by hybrid score descending
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        Ok(candidates)
    }

    /// Normalize BM25 scores to [0,1] range
    fn normalize_bm25_scores(&self, candidates: Vec<Candidate>) -> Vec<Candidate> {
        if candidates.is_empty() {
            return candidates;
        }

        let max_score = candidates.iter().map(|c| c.score).fold(0.0, f64::max);

        if max_score == 0.0 {
            return candidates;
        }

        candidates
            .into_iter()
            .map(|mut c| {
                c.score /= max_score;
                c
            })
            .collect()
    }

    /// Normalize cosine scores from [-1,1] to [0,1] range
    fn normalize_cosine_scores(&self, candidates: Vec<Candidate>) -> Vec<Candidate> {
        candidates
            .into_iter()
            .map(|mut c| {
                c.score = (c.score + 1.0) / 2.0;
                c
            })
            .collect()
    }

    /// Apply post-processing (reranking, diversification)
    async fn post_process<R: DocumentRepository + ?Sized>(
        &self,
        repository: &R,
        candidates: Vec<Candidate>,
    ) -> Result<Vec<Candidate>> {
        if candidates.is_empty() {
            return Ok(candidates);
        }

        let mut processed = candidates;

        if self.config.diversify && processed.len() > self.config.k_final as usize {
            let fallback = processed.clone();
            let diversification_result = self.diversify_candidates(repository, processed).await;
            processed = match diversification_result {
                Ok(diversified) => diversified,
                Err(err) => {
                    tracing::warn!("Diversification failed: {}", err);
                    fallback
                }
            };
        }

        processed.truncate(self.config.k_final as usize);
        Ok(processed)
    }

    async fn enrich_candidates<R: DocumentRepository + ?Sized>(
        &self,
        repository: &R,
        mut candidates: Vec<Candidate>,
    ) -> Result<Vec<Candidate>> {
        for candidate in candidates.iter_mut() {
            if candidate.text.is_none() {
                if let Some(chunk) = repository.get_chunk_by_id(&candidate.doc_id).await? {
                    candidate.text = Some(chunk.text);
                    if candidate.kind.is_none() {
                        candidate.kind = Some(chunk.kind);
                    }
                }
            }
        }
        Ok(candidates)
    }

    async fn diversify_candidates<R: DocumentRepository + ?Sized>(
        &self,
        repository: &R,
        candidates: Vec<Candidate>,
    ) -> Result<Vec<Candidate>> {
        let target_k = self.config.k_final.max(1) as usize;
        if candidates.len() <= target_k {
            return Ok(candidates);
        }

        let mut pool = self
            .enrich_candidates(repository, candidates)
            .await?
            .into_iter()
            .take(self.config.k_initial.max(self.config.k_final) as usize)
            .collect::<Vec<_>>();

        if pool.len() <= target_k {
            pool.truncate(target_k);
            return Ok(pool);
        }

        let embedding_service = self.vector_service.embedding_service();
        if embedding_service.dimension() == 0 {
            pool.truncate(target_k);
            return Ok(pool);
        }

        let texts: Vec<String> = pool
            .iter()
            .map(|candidate| candidate.text.clone().unwrap_or_default())
            .collect();

        if texts.iter().all(|text| text.trim().is_empty()) {
            pool.truncate(target_k);
            return Ok(pool);
        }

        let embeddings = embedding_service.embed(&texts).await?;
        if embeddings.len() != pool.len() {
            tracing::warn!(
                "Embedding service returned mismatched vector count (expected {}, got {})",
                pool.len(),
                embeddings.len()
            );
            pool.truncate(target_k);
            return Ok(pool);
        }

        let normalised_embeddings: Vec<Vec<f32>> = embeddings
            .into_iter()
            .map(|embedding| normalise_vector(&embedding.data))
            .collect();

        let normalised_scores = normalise_scores(&pool);
        let lambda = self.config.diversify_lambda.clamp(0.0, 1.0);

        let mut selected_indices = Vec::with_capacity(target_k);
        let mut remaining: Vec<usize> = (0..pool.len()).collect();

        if let Some((best_idx, _)) = remaining
            .iter()
            .map(|&idx| (idx, normalised_scores[idx]))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            selected_indices.push(best_idx);
            remaining.retain(|&idx| idx != best_idx);
        }

        while selected_indices.len() < target_k && !remaining.is_empty() {
            let mut best_candidate = remaining[0];
            let mut best_score = f64::MIN;

            for &candidate_idx in &remaining {
                let relevance = normalised_scores[candidate_idx];
                let max_similarity = selected_indices
                    .iter()
                    .map(|&selected_idx| {
                        cosine_similarity(
                            normalised_embeddings[candidate_idx].as_slice(),
                            normalised_embeddings[selected_idx].as_slice(),
                        )
                    })
                    .fold(0.0_f64, f64::max);

                let mmr_score = lambda * relevance - (1.0 - lambda) * max_similarity;
                if mmr_score > best_score {
                    best_score = mmr_score;
                    best_candidate = candidate_idx;
                }
            }

            selected_indices.push(best_candidate);
            remaining.retain(|&idx| idx != best_candidate);
        }

        let mut diversified = Vec::with_capacity(selected_indices.len());
        for idx in selected_indices {
            if let Some(candidate) = pool.get(idx) {
                diversified.push(candidate.clone());
            }
        }

        Ok(diversified)
    }
}

fn effective_search_ef(k: usize, dimension: usize) -> usize {
    let base = EPHEMERAL_HNSW_BASE_EF_SEARCH.max(dimension.saturating_mul(2));
    base.max(k.saturating_mul(4)).max(32)
}

fn hnsw_search(
    embeddings: &[EmbeddingVector],
    query: &[f32],
    k: usize,
) -> Result<Vec<(usize, f64)>> {
    if embeddings.is_empty() || k == 0 {
        return Ok(Vec::new());
    }

    let dimension = query.len();
    if dimension == 0 {
        return Err(LetheError::embedding(
            "Cannot perform vector search with zero-dimensional query",
        ));
    }

    let mut hnsw = Hnsw::new(
        EPHEMERAL_HNSW_MAX_CONNECTIONS,
        embeddings.len().max(1),
        EPHEMERAL_HNSW_MAX_LAYER,
        EPHEMERAL_HNSW_EF_CONSTRUCTION,
        DistCosine::default(),
    );
    hnsw.set_keeping_pruned(true);

    for (idx, embedding) in embeddings.iter().enumerate() {
        if embedding.data.len() != dimension {
            return Err(LetheError::embedding(format!(
                "Embedding dimension mismatch for chunk index {} (expected {}, found {})",
                idx,
                dimension,
                embedding.data.len()
            )));
        }

        hnsw.insert((&embedding.data, idx));
    }

    hnsw.set_searching_mode(true);

    let neighbours = hnsw.search(query, k, effective_search_ef(k, dimension));
    let mut results = Vec::with_capacity(neighbours.len());
    for neighbour in neighbours {
        let idx = neighbour.get_origin_id();
        let similarity = (1.0f32 - neighbour.distance).clamp(0.0, 1.0) as f64;
        results.push((idx, similarity));
    }

    Ok(results)
}

fn normalise_vector(vector: &[f32]) -> Vec<f32> {
    let norm = vector
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>()
        .sqrt();

    if norm == 0.0 {
        vec![0.0; vector.len()]
    } else {
        let inv = 1.0f64 / norm;
        vector
            .iter()
            .map(|value| (*value as f64 * inv) as f32)
            .collect()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum::<f64>()
        .clamp(-1.0, 1.0)
}

fn normalise_scores(candidates: &[Candidate]) -> Vec<f64> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let (min_score, max_score) = candidates.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(min_s, max_s), candidate| (min_s.min(candidate.score), max_s.max(candidate.score)),
    );

    let range = (max_score - min_score).abs().max(1e-6);

    candidates
        .iter()
        .map(|candidate| ((candidate.score - min_score) / range).clamp(0.0, 1.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::FallbackEmbeddingService;
    use lethe_shared::Chunk;
    use std::sync::Arc;
    use uuid::Uuid;

    // Mock repository for testing
    struct MockRepository {
        chunks: Vec<Chunk>,
        dfidf: Vec<DfIdf>,
    }

    #[async_trait]
    impl DocumentRepository for MockRepository {
        async fn get_chunks_by_session(&self, _session_id: &str) -> Result<Vec<Chunk>> {
            Ok(self.chunks.clone())
        }

        async fn get_dfidf_by_session(&self, _session_id: &str) -> Result<Vec<DfIdf>> {
            Ok(self.dfidf.clone())
        }

        async fn get_chunk_by_id(&self, chunk_id: &str) -> Result<Option<Chunk>> {
            Ok(self.chunks.iter().find(|c| c.id == chunk_id).cloned())
        }

        async fn vector_search(
            &self,
            _query_vector: &EmbeddingVector,
            k: i32,
        ) -> Result<Vec<Candidate>> {
            // Return mock vector search results
            let candidates: Vec<Candidate> = self
                .chunks
                .iter()
                .take(k as usize)
                .map(|chunk| Candidate {
                    doc_id: chunk.id.clone(),
                    score: 0.8, // Mock similarity score
                    text: Some(chunk.text.clone()),
                    kind: Some(chunk.kind.clone()),
                })
                .collect();
            Ok(candidates)
        }
    }

    fn create_test_chunk(id: &str, text: &str, kind: &str) -> Chunk {
        Chunk {
            id: id.to_string(),
            message_id: Uuid::new_v4(),
            session_id: "test-session".to_string(),
            offset_start: 0,
            offset_end: text.len(),
            kind: kind.to_string(),
            text: text.to_string(),
            tokens: text.split_whitespace().count() as i32,
        }
    }

    #[tokio::test]
    async fn test_bm25_search() {
        let chunks = vec![
            create_test_chunk("1", "hello world", "text"),
            create_test_chunk("2", "world peace", "text"),
            create_test_chunk("3", "goodbye world", "text"),
        ];

        let dfidf = vec![
            DfIdf {
                term: "hello".to_string(),
                session_id: "test-session".to_string(),
                df: 1,
                idf: 1.0,
            },
            DfIdf {
                term: "world".to_string(),
                session_id: "test-session".to_string(),
                df: 3,
                idf: 0.5,
            },
        ];

        let repository = MockRepository { chunks, dfidf };
        let queries = vec!["hello world".to_string()];

        let results = Bm25SearchService::search(&repository, &queries, "test-session", 10)
            .await
            .unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].doc_id, "1"); // Should rank "hello world" highest
    }

    #[tokio::test]
    async fn test_hybrid_retrieval() {
        let chunks = vec![
            create_test_chunk("1", "async programming in rust", "text"),
            create_test_chunk("2", "rust error handling", "text"),
            create_test_chunk("3", "javascript async await", "text"),
        ];

        let dfidf = vec![
            DfIdf {
                term: "async".to_string(),
                session_id: "test-session".to_string(),
                df: 2,
                idf: 0.4,
            },
            DfIdf {
                term: "rust".to_string(),
                session_id: "test-session".to_string(),
                df: 2,
                idf: 0.4,
            },
        ];

        let repository = MockRepository { chunks, dfidf };
        let embedding_service = Arc::new(FallbackEmbeddingService::new(384));
        let config = HybridRetrievalConfig::default();
        let service = HybridRetrievalService::new(embedding_service, config);

        let queries = vec!["rust async programming".to_string()];
        let results = service
            .retrieve(&repository, &queries, "test-session")
            .await
            .unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_score_normalization() {
        let embedding_service = Arc::new(FallbackEmbeddingService::new(384));
        let config = HybridRetrievalConfig::default();
        let service = HybridRetrievalService::new(embedding_service, config);

        let candidates = vec![
            Candidate {
                doc_id: "1".to_string(),
                score: 10.0,
                text: None,
                kind: None,
            },
            Candidate {
                doc_id: "2".to_string(),
                score: 5.0,
                text: None,
                kind: None,
            },
        ];

        let normalized = service.normalize_bm25_scores(candidates);
        assert_eq!(normalized[0].score, 1.0);
        assert_eq!(normalized[1].score, 0.5);
    }

    #[test]
    fn test_query_features() {
        let features = QueryFeatures::extract_features("function_name() error in /path/file.rs");
        assert!(features.has_code_symbol);
        assert!(features.has_error_token);
        assert!(features.has_path_or_file);

        let boost = QueryFeatures::gamma_boost("code", &features);
        assert!(boost > 0.0);
    }

    #[test]
    fn test_query_features_comprehensive() {
        // Test code symbols
        let features1 = QueryFeatures::extract_features("call myFunction() here");
        assert!(features1.has_code_symbol);
        assert!(!features1.has_error_token);

        // Test namespace symbols
        let features2 = QueryFeatures::extract_features("use MyClass::StaticMethod");
        assert!(features2.has_code_symbol);

        // Test error tokens
        let features3 = QueryFeatures::extract_features("NullPointerException occurred");
        assert!(features3.has_error_token);
        assert!(!features3.has_code_symbol);

        // Test file paths
        let features4 = QueryFeatures::extract_features("check /home/user/file.txt");
        assert!(features4.has_path_or_file);
        assert!(!features4.has_error_token);

        // Test Windows paths
        let features5 = QueryFeatures::extract_features("see C:\\Users\\Name\\doc.docx");
        assert!(features5.has_path_or_file);

        // Test numeric IDs
        let features6 = QueryFeatures::extract_features("issue 1234 needs fixing");
        assert!(features6.has_numeric_id);
        assert!(!features6.has_code_symbol);

        // Test empty query
        let features7 = QueryFeatures::extract_features("");
        assert!(!features7.has_code_symbol);
        assert!(!features7.has_error_token);
        assert!(!features7.has_path_or_file);
        assert!(!features7.has_numeric_id);
    }

    #[test]
    fn test_gamma_boost_combinations() {
        // Test code symbol boost with different content kinds
        let features = QueryFeatures::extract_features("myFunction() returns value");

        let code_boost = QueryFeatures::gamma_boost("code", &features);
        assert!(code_boost > 0.0);

        let user_code_boost = QueryFeatures::gamma_boost("user_code", &features);
        assert!(user_code_boost > 0.0);

        let text_boost = QueryFeatures::gamma_boost("text", &features);
        assert_eq!(text_boost, 0.0); // Should not boost for text content

        // Test error token boost
        let error_features = QueryFeatures::extract_features("RuntimeError in execution");
        let tool_boost = QueryFeatures::gamma_boost("tool_result", &error_features);
        assert!(tool_boost > 0.0);

        // Test path boost
        let path_features = QueryFeatures::extract_features("file located at /src/main.rs");
        let code_path_boost = QueryFeatures::gamma_boost("code", &path_features);
        assert!(code_path_boost > 0.0);

        // Test combined features
        let combined_features =
            QueryFeatures::extract_features("function() error in /path/file.rs with ID 1234");
        assert!(combined_features.has_code_symbol);
        assert!(combined_features.has_error_token);
        assert!(combined_features.has_path_or_file);
        assert!(combined_features.has_numeric_id);

        let combined_boost = QueryFeatures::gamma_boost("code", &combined_features);
        assert!(combined_boost > 0.1); // Should have multiple boosts
    }

    #[tokio::test]
    async fn test_hybrid_retrieval_creation() {
        use crate::embeddings::FallbackEmbeddingService;

        let embedding_service = Arc::new(FallbackEmbeddingService::new(384));
        let service = HybridRetrievalService::new(
            embedding_service.clone(),
            HybridRetrievalConfig::default(),
        );

        // Test service creation
        assert_eq!(service.config.alpha, 0.7); // Default alpha value
        assert_eq!(service.config.beta, 0.3); // Default beta value
        assert!(service.config.gamma_kind_boost.contains_key("code"));
    }

    #[tokio::test]
    async fn test_retrieval_service_configurations() {
        use crate::embeddings::FallbackEmbeddingService;

        let embedding_service = Arc::new(FallbackEmbeddingService::new(384));

        // Test custom configuration
        let custom_config = HybridRetrievalConfig {
            alpha: 0.3,
            beta: 0.7,
            gamma_kind_boost: std::collections::HashMap::from([
                ("code".to_string(), 0.15),
                ("user_code".to_string(), 0.12),
            ]),
            rerank: true,
            diversify: false,
            diversify_method: "simple".to_string(),
            diversify_lambda: 0.65,
            k_initial: 50,
            k_final: 10,
            fusion_dynamic: false,
        };

        let service = HybridRetrievalService::new(embedding_service.clone(), custom_config.clone());

        // Verify configuration is applied
        assert_eq!(service.config.alpha, 0.3);
        assert_eq!(service.config.beta, 0.7);
        assert_eq!(service.config.gamma_kind_boost.get("code"), Some(&0.15));
    }

    #[test]
    fn test_bm25_service_properties() {
        let service = Bm25SearchService;

        // Test that service has expected behavior
        // Since Bm25SearchService doesn't have these methods, test what's available
        // The actual BM25 implementation seems to be elsewhere
        // This test validates the service can be instantiated
        let _ = service;
    }

    #[test]
    fn test_vector_search_service_properties() {
        use crate::embeddings::FallbackEmbeddingService;

        let embedding_service = Arc::new(FallbackEmbeddingService::new(384));
        let service = VectorSearchService::new(embedding_service.clone());

        // Test that service can be created
        assert_eq!(service.embedding_service.name(), "fallback");

        // Test dimension access
        assert_eq!(service.embedding_service.dimension(), 384);
    }

    #[test]
    fn test_retrieval_config_defaults() {
        // Test that default config has expected values
        let config = HybridRetrievalConfig::default();

        assert_eq!(config.alpha, 0.7);
        assert_eq!(config.beta, 0.3);
        assert_eq!(config.k_initial, 50);
        assert_eq!(config.k_final, 20);
        assert!(config.diversify);
        assert!(config.gamma_kind_boost.contains_key("code"));

        // Test gamma boost value for code
        assert_eq!(config.gamma_kind_boost.get("code"), Some(&1.2));
    }
}
