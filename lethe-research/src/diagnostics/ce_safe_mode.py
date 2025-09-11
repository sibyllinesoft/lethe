"""
Cross-Encoder Safe Mode Implementation
====================================

Fallback scoring system that bypasses the broken cross-encoder while
fixing is in progress. Uses bi-encoder + BM25F hybrid scoring to
maintain reasonable retrieval quality during debugging.

Safe Mode Configuration:
- 60% bi-encoder dot product similarity  
- 40% BM25F lexical matching
- γ=0.8 (facility location emphasis)
- δ=0 (disable DPP diversity temporarily)
- Increased K1=4000-6000, K2=1000-1500 for larger candidate pools

This allows system operation while cross-encoder issues are resolved.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
import math
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

@dataclass
class SafeModeConfig:
    """Configuration for safe mode operation."""
    bi_encoder_weight: float = 0.6
    bm25_weight: float = 0.4
    facility_gamma: float = 0.8
    diversity_delta: float = 0.0  # Disabled during safe mode
    k1_candidate_pool: int = 5000
    k2_rerank_budget: int = 1200
    dims_full: int = 768  # Use full dimensionality
    min_score: float = 0.0
    max_score: float = 1.0

@dataclass
class SafeModeResult:
    """Results from safe mode scoring."""
    scores: Dict[str, float]
    method_used: str
    bi_encoder_scores: Dict[str, float]
    bm25_scores: Dict[str, float]
    execution_time_ms: float
    fallback_active: bool

class CrossEncoderSafeMode:
    """
    Safe mode implementation for cross-encoder fallback.
    
    Provides hybrid bi-encoder + BM25F scoring when cross-encoder
    is producing flat scores or encountering errors.
    """
    
    def __init__(self, 
                 config: Optional[SafeModeConfig] = None,
                 bi_encoder: Any = None,
                 bm25_scorer: Any = None):
        """
        Initialize safe mode system.
        
        Args:
            config: Safe mode configuration
            bi_encoder: Bi-encoder for semantic similarity
            bm25_scorer: BM25F scorer for lexical matching
        """
        self.config = config or SafeModeConfig()
        self.bi_encoder = bi_encoder
        self.bm25_scorer = bm25_scorer
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Track safe mode usage
        self._safe_mode_active = False
        self._activation_time = None
        self._total_safe_mode_calls = 0
    
    def activate_safe_mode(self, reason: str = "Cross-encoder malfunction"):
        """
        Activate safe mode operation.
        
        Args:
            reason: Reason for activation
        """
        if not self._safe_mode_active:
            self._safe_mode_active = True
            self._activation_time = time.time()
            self.logger.warning("🛡️ CROSS-ENCODER SAFE MODE ACTIVATED")
            self.logger.warning(f"   Reason: {reason}")
            self.logger.warning(f"   Fallback: {self.config.bi_encoder_weight:.1%} bi-encoder + {self.config.bm25_weight:.1%} BM25F")
            self.logger.warning("   Cross-encoder scoring bypassed until resolved")
    
    def deactivate_safe_mode(self):
        """Deactivate safe mode and return to normal operation."""
        if self._safe_mode_active:
            duration = time.time() - self._activation_time if self._activation_time else 0
            self.logger.info("🛡️ SAFE MODE DEACTIVATED")
            self.logger.info(f"   Duration: {duration:.1f} seconds")
            self.logger.info(f"   Total calls in safe mode: {self._total_safe_mode_calls}")
            
            self._safe_mode_active = False
            self._activation_time = None
    
    def is_safe_mode_active(self) -> bool:
        """Check if safe mode is currently active."""
        return self._safe_mode_active
    
    def safe_mode_score_pairs(self, 
                            query: str,
                            doc_ids: List[str],
                            documents: Optional[Dict[str, str]] = None,
                            query_embedding: Optional[np.ndarray] = None) -> SafeModeResult:
        """
        Score query-document pairs using safe mode hybrid approach.
        
        Args:
            query: Query text
            doc_ids: List of document IDs to score
            documents: Optional document content mapping
            query_embedding: Pre-computed query embedding
            
        Returns:
            SafeModeResult with hybrid scores
        """
        start_time = time.time()
        self._total_safe_mode_calls += 1
        
        if not self._safe_mode_active:
            self.logger.warning("Safe mode scoring called but not activated - activating now")
            self.activate_safe_mode("Safe mode scoring requested")
        
        self.logger.debug(f"🛡️ Safe mode scoring {len(doc_ids)} pairs")
        
        # Get bi-encoder scores
        bi_encoder_scores = self._get_bi_encoder_scores(query, doc_ids, documents, query_embedding)
        
        # Get BM25F scores
        bm25_scores = self._get_bm25f_scores(query, doc_ids, documents)
        
        # Combine scores using hybrid weighting
        combined_scores = self._combine_scores(bi_encoder_scores, bm25_scores)
        
        # Apply safe mode configuration adjustments
        final_scores = self._apply_safe_mode_adjustments(combined_scores)
        
        execution_time = (time.time() - start_time) * 1000
        
        result = SafeModeResult(
            scores=final_scores,
            method_used="hybrid_bi_encoder_bm25f",
            bi_encoder_scores=bi_encoder_scores,
            bm25_scores=bm25_scores,
            execution_time_ms=execution_time,
            fallback_active=True
        )
        
        self.logger.debug(f"🛡️ Safe mode scoring completed in {execution_time:.1f}ms")
        
        return result
    
    def _get_bi_encoder_scores(self, 
                             query: str,
                             doc_ids: List[str],
                             documents: Optional[Dict[str, str]],
                             query_embedding: Optional[np.ndarray]) -> Dict[str, float]:
        """Get bi-encoder semantic similarity scores."""
        scores = {}
        
        try:
            if self.bi_encoder is None:
                self.logger.warning("No bi-encoder available - using fallback similarity")
                return self._fallback_similarity_scores(query, doc_ids, documents)
            
            # Get query embedding if not provided
            if query_embedding is None:
                if hasattr(self.bi_encoder, 'encode_query'):
                    query_embedding = self.bi_encoder.encode_query(query)
                elif hasattr(self.bi_encoder, 'encode'):
                    query_embedding = self.bi_encoder.encode([query])[0]
                else:
                    self.logger.warning("Cannot get query embedding from bi-encoder")
                    return self._fallback_similarity_scores(query, doc_ids, documents)
            
            # Get document embeddings and compute similarities
            for doc_id in doc_ids:
                try:
                    if documents and doc_id in documents:
                        doc_text = documents[doc_id]
                    else:
                        doc_text = f"Document {doc_id}"  # Fallback
                    
                    # Get document embedding
                    if hasattr(self.bi_encoder, 'encode_document'):
                        doc_embedding = self.bi_encoder.encode_document(doc_text)
                    elif hasattr(self.bi_encoder, 'encode'):
                        doc_embedding = self.bi_encoder.encode([doc_text])[0]
                    else:
                        self.logger.warning(f"Cannot encode document {doc_id}")
                        scores[doc_id] = 0.5  # Neutral score
                        continue
                    
                    # Compute dot product similarity
                    similarity = float(np.dot(query_embedding, doc_embedding))
                    
                    # Normalize to [0, 1] range (assuming embeddings are normalized)
                    normalized_score = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
                    scores[doc_id] = normalized_score
                    
                except Exception as e:
                    self.logger.warning(f"Bi-encoder scoring failed for {doc_id}: {e}")
                    scores[doc_id] = 0.5  # Neutral score
            
        except Exception as e:
            self.logger.warning(f"Bi-encoder scoring failed: {e}")
            return self._fallback_similarity_scores(query, doc_ids, documents)
        
        return scores
    
    def _get_bm25f_scores(self, 
                        query: str,
                        doc_ids: List[str],
                        documents: Optional[Dict[str, str]]) -> Dict[str, float]:
        """Get BM25F lexical matching scores."""
        scores = {}
        
        try:
            if self.bm25_scorer is not None:
                # Use existing BM25 scorer
                for doc_id in doc_ids:
                    try:
                        if documents and doc_id in documents:
                            doc_text = documents[doc_id]
                        else:
                            doc_text = f"Document {doc_id}"
                        
                        if hasattr(self.bm25_scorer, 'score'):
                            score = self.bm25_scorer.score(query, doc_text)
                        elif hasattr(self.bm25_scorer, 'get_scores'):
                            score = self.bm25_scorer.get_scores(query, [doc_text])[0]
                        else:
                            score = self._simple_bm25_score(query, doc_text)
                        
                        # Normalize to [0, 1]
                        normalized_score = max(0.0, min(1.0, score / 10.0))  # Assume max BM25 score ~10
                        scores[doc_id] = float(normalized_score)
                        
                    except Exception as e:
                        self.logger.warning(f"BM25 scoring failed for {doc_id}: {e}")
                        scores[doc_id] = self._simple_bm25_score(query, documents.get(doc_id, "") if documents else "")
            else:
                # Use simple BM25 implementation
                for doc_id in doc_ids:
                    doc_text = documents.get(doc_id, f"Document {doc_id}") if documents else f"Document {doc_id}"
                    scores[doc_id] = self._simple_bm25_score(query, doc_text)
            
        except Exception as e:
            self.logger.warning(f"BM25F scoring failed: {e}")
            # Fallback to simple lexical overlap
            for doc_id in doc_ids:
                doc_text = documents.get(doc_id, "") if documents else ""
                scores[doc_id] = self._lexical_overlap_score(query, doc_text)
        
        return scores
    
    def _simple_bm25_score(self, query: str, document: str) -> float:
        """Simple BM25 scoring implementation."""
        try:
            # BM25 parameters
            k1 = 1.2
            b = 0.75
            
            # Tokenize
            query_terms = query.lower().split()
            doc_terms = document.lower().split()
            doc_length = len(doc_terms)
            
            if doc_length == 0:
                return 0.0
            
            # Assume average document length (could be computed from corpus)
            avg_doc_length = 100.0
            
            # Term frequencies
            doc_term_freq = Counter(doc_terms)
            
            score = 0.0
            for term in set(query_terms):
                if term in doc_term_freq:
                    tf = doc_term_freq[term]
                    # Simplified IDF (assume term appears in half of documents)
                    idf = math.log(2.0 / 1.0)  
                    
                    # BM25 formula
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                    score += idf * (numerator / denominator)
            
            # Normalize to [0, 1]
            return max(0.0, min(1.0, score / 10.0))
            
        except Exception as e:
            self.logger.warning(f"Simple BM25 calculation failed: {e}")
            return self._lexical_overlap_score(query, document)
    
    def _lexical_overlap_score(self, query: str, document: str) -> float:
        """Simple lexical overlap scoring."""
        try:
            query_words = set(query.lower().split())
            doc_words = set(document.lower().split())
            
            if not query_words:
                return 0.0
            
            overlap = len(query_words.intersection(doc_words))
            return overlap / len(query_words)
            
        except Exception as e:
            self.logger.warning(f"Lexical overlap calculation failed: {e}")
            return 0.5
    
    def _fallback_similarity_scores(self, 
                                  query: str,
                                  doc_ids: List[str],
                                  documents: Optional[Dict[str, str]]) -> Dict[str, float]:
        """Fallback similarity when bi-encoder is unavailable."""
        scores = {}
        
        for doc_id in doc_ids:
            doc_text = documents.get(doc_id, f"Document {doc_id}") if documents else f"Document {doc_id}"
            
            # Use lexical overlap as fallback
            overlap_score = self._lexical_overlap_score(query, doc_text)
            
            # Add some random variation to avoid completely flat scores
            random_factor = (hash(doc_id) % 100) / 1000.0  # Small variation ±0.05
            final_score = max(0.0, min(1.0, overlap_score + random_factor))
            
            scores[doc_id] = final_score
        
        return scores
    
    def _combine_scores(self, 
                       bi_encoder_scores: Dict[str, float],
                       bm25_scores: Dict[str, float]) -> Dict[str, float]:
        """Combine bi-encoder and BM25F scores using configured weights."""
        combined = {}
        
        # Get all document IDs
        all_doc_ids = set(bi_encoder_scores.keys()) | set(bm25_scores.keys())
        
        for doc_id in all_doc_ids:
            bi_score = bi_encoder_scores.get(doc_id, 0.5)  # Neutral default
            bm25_score = bm25_scores.get(doc_id, 0.5)     # Neutral default
            
            # Weighted combination
            combined_score = (
                self.config.bi_encoder_weight * bi_score +
                self.config.bm25_weight * bm25_score
            )
            
            combined[doc_id] = float(combined_score)
        
        return combined
    
    def _apply_safe_mode_adjustments(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Apply safe mode configuration adjustments."""
        adjusted_scores = {}
        
        # Apply facility location emphasis (γ=0.8)
        # This increases scores for documents that cover unique content
        if self.config.facility_gamma > 0:
            # Simple facility location: boost diverse scores
            score_values = list(scores.values())
            if score_values:
                mean_score = np.mean(score_values)
                
                for doc_id, score in scores.items():
                    # Boost scores that deviate from mean (encourage diversity)
                    diversity_bonus = abs(score - mean_score) * self.config.facility_gamma * 0.1
                    adjusted_score = score + diversity_bonus
                    
                    # Clamp to valid range
                    adjusted_scores[doc_id] = max(self.config.min_score, 
                                                min(self.config.max_score, adjusted_score))
        else:
            adjusted_scores = scores
        
        # Note: DPP diversity (δ) is disabled in safe mode (set to 0)
        
        return adjusted_scores
    
    def get_safe_mode_stats(self) -> Dict[str, Any]:
        """Get safe mode usage statistics."""
        stats = {
            'safe_mode_active': self._safe_mode_active,
            'total_calls': self._total_safe_mode_calls,
            'config': {
                'bi_encoder_weight': self.config.bi_encoder_weight,
                'bm25_weight': self.config.bm25_weight,
                'facility_gamma': self.config.facility_gamma,
                'diversity_delta': self.config.diversity_delta,
                'k1_candidate_pool': self.config.k1_candidate_pool,
                'k2_rerank_budget': self.config.k2_rerank_budget
            }
        }
        
        if self._activation_time:
            stats['active_duration_seconds'] = time.time() - self._activation_time
        
        return stats
    
    def update_parameters(self, **kwargs):
        """Update safe mode parameters on the fly."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Safe mode parameter updated: {key} = {value}")
            else:
                self.logger.warning(f"Unknown safe mode parameter: {key}")
        
        # Log current configuration
        self.logger.info("Updated safe mode configuration:")
        self.logger.info(f"  Bi-encoder weight: {self.config.bi_encoder_weight}")
        self.logger.info(f"  BM25F weight: {self.config.bm25_weight}")
        self.logger.info(f"  Facility gamma: {self.config.facility_gamma}")
        self.logger.info(f"  Diversity delta: {self.config.diversity_delta}")
        self.logger.info(f"  K1 candidate pool: {self.config.k1_candidate_pool}")
        self.logger.info(f"  K2 rerank budget: {self.config.k2_rerank_budget}")

class SafeModeRetrievalPipeline:
    """
    Wrapper for retrieval pipeline with integrated safe mode.
    
    Automatically falls back to safe mode when cross-encoder issues are detected.
    """
    
    def __init__(self, 
                 base_pipeline: Any,
                 safe_mode: CrossEncoderSafeMode,
                 auto_fallback: bool = True):
        """
        Initialize safe mode pipeline wrapper.
        
        Args:
            base_pipeline: Base retrieval pipeline
            safe_mode: Safe mode implementation  
            auto_fallback: Automatically activate safe mode on CE errors
        """
        self.base_pipeline = base_pipeline
        self.safe_mode = safe_mode
        self.auto_fallback = auto_fallback
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Track cross-encoder failures
        self._ce_failure_count = 0
        self._ce_failure_threshold = 3
    
    async def retrieve(self, query: str, k: int = 10) -> Any:
        """Retrieve with automatic safe mode fallback."""
        try:
            # Try normal retrieval
            if not self.safe_mode.is_safe_mode_active():
                results = await self.base_pipeline.retrieve(query, k)
                
                # Check if results indicate CE issues (e.g., flat scores)
                if self._detect_ce_issues(results):
                    self._ce_failure_count += 1
                    
                    if self._ce_failure_count >= self._ce_failure_threshold and self.auto_fallback:
                        self.safe_mode.activate_safe_mode("Cross-encoder producing flat scores")
                        return await self._safe_mode_retrieve(query, k)
                
                return results
            else:
                # Already in safe mode
                return await self._safe_mode_retrieve(query, k)
                
        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            
            if self.auto_fallback:
                self.safe_mode.activate_safe_mode(f"Retrieval error: {str(e)}")
                return await self._safe_mode_retrieve(query, k)
            else:
                raise
    
    async def _safe_mode_retrieve(self, query: str, k: int) -> Any:
        """Perform retrieval using safe mode."""
        self.logger.debug(f"🛡️ Safe mode retrieval: query='{query[:50]}...', k={k}")
        
        try:
            # Get candidates from S1 (bi-encoder) layer with increased pool size
            candidates = await self._get_s1_candidates(query, self.safe_mode.config.k1_candidate_pool)
            
            # Score using safe mode hybrid approach
            doc_ids = [c.doc_id if hasattr(c, 'doc_id') else str(i) for i, c in enumerate(candidates)]
            documents = {doc_id: getattr(c, 'text', str(c)) for doc_id, c in zip(doc_ids, candidates)}
            
            safe_result = self.safe_mode.safe_mode_score_pairs(query, doc_ids, documents)
            
            # Apply selection with increased rerank budget
            selected = self._select_top_k(candidates, safe_result.scores, 
                                        min(k, self.safe_mode.config.k2_rerank_budget))
            
            return selected
            
        except Exception as e:
            self.logger.error(f"Safe mode retrieval failed: {e}")
            # Return empty results rather than failing completely
            return []
    
    async def _get_s1_candidates(self, query: str, k: int) -> List[Any]:
        """Get S1 (bi-encoder) candidates."""
        if hasattr(self.base_pipeline, 'get_s1_candidates'):
            return await self.base_pipeline.get_s1_candidates(query, k)
        elif hasattr(self.base_pipeline, 'bi_encoder_retrieve'):
            return await self.base_pipeline.bi_encoder_retrieve(query, k)
        else:
            self.logger.warning("Cannot access S1 candidates - using base retrieve")
            return await self.base_pipeline.retrieve(query, k)
    
    def _select_top_k(self, candidates: List[Any], scores: Dict[str, float], k: int) -> List[Any]:
        """Select top-k candidates based on safe mode scores."""
        if not candidates or not scores:
            return candidates[:k]
        
        # Create scored candidates
        scored_candidates = []
        for i, candidate in enumerate(candidates):
            doc_id = getattr(candidate, 'doc_id', str(i))
            score = scores.get(doc_id, 0.0)
            scored_candidates.append((candidate, score))
        
        # Sort by score (descending) and take top-k
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [candidate for candidate, _ in scored_candidates[:k]]
    
    def _detect_ce_issues(self, results: Any) -> bool:
        """Detect if cross-encoder is producing problematic scores."""
        try:
            # Extract scores from results
            scores = []
            if hasattr(results, 'scores'):
                scores = results.scores
            elif isinstance(results, list):
                scores = [getattr(r, 'score', 0.0) for r in results if hasattr(r, 'score')]
            
            if not scores or len(scores) < 2:
                return False
            
            # Check for flat scoring
            score_std = np.std(scores)
            score_range = max(scores) - min(scores)
            
            # Detect issues
            flat_scoring = score_std < 0.01 or score_range < 0.05
            all_identical = len(set(np.round(scores, 3))) <= 2
            
            return flat_scoring or all_identical
            
        except Exception as e:
            self.logger.warning(f"CE issue detection failed: {e}")
            return False