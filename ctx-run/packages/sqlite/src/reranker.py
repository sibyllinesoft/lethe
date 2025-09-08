#!/usr/bin/env python3
"""
Lightweight Cross-Encoder Reranking System

Optional CPU-compatible cross-encoder reranking for top-K candidates.
Designed to be fast, lightweight (≤50M params), and configurable.

Key Features:
- CPU-only execution (no GPU dependencies)
- Configurable and OFF by default for performance  
- Lightweight models (sentence-transformers cross-encoders)
- Batch processing for efficiency
- Fallback to no reranking if model unavailable
- Comprehensive telemetry and timing
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import time
import os
from pathlib import Path

# Optional dependencies - gracefully handle missing imports
try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    CrossEncoder = None

logger = logging.getLogger(__name__)

@dataclass
class RerankingConfig:
    """Configuration for lightweight cross-encoder reranking."""
    
    # Model configuration
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ~23M parameters
    model_cache_dir: Optional[str] = None  # Use default cache
    
    # Performance settings
    batch_size: int = 32              # Batch size for inference
    max_length: int = 512             # Max sequence length
    device: str = "cpu"               # Force CPU execution
    
    # Reranking behavior
    enabled: bool = False             # OFF by default
    top_k_rerank: int = 100          # Only rerank top-K candidates
    
    # Fallback behavior
    silent_fallback: bool = True      # Don't error if model unavailable
    
    # Caching
    cache_embeddings: bool = False    # Don't cache for memory efficiency
    
    def __post_init__(self):
        """Validate configuration."""
        if self.top_k_rerank <= 0:
            raise ValueError(f"top_k_rerank must be positive, got {self.top_k_rerank}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

@dataclass
class RerankingResult:
    """Result of cross-encoder reranking."""
    
    doc_ids: List[str]
    original_scores: List[float]      # Original retrieval scores  
    rerank_scores: List[float]        # Cross-encoder scores
    final_scores: List[float]         # Final combined scores
    
    # Metadata
    model_name: str
    candidates_reranked: int
    reranking_time_ms: float
    batch_count: int
    fallback_used: bool               # True if reranking was skipped

class LightweightCrossEncoderReranker:
    """
    CPU-compatible lightweight cross-encoder for reranking.
    
    Uses compact cross-encoder models (≤50M params) optimized for CPU inference.
    Gracefully falls back to no reranking if models unavailable.
    """
    
    def __init__(self, config: Optional[RerankingConfig] = None):
        """Initialize reranker."""
        self.config = config or RerankingConfig()
        self.model: Optional[Any] = None
        self.model_loaded = False
        
        if self.config.enabled:
            self._load_model()
        
        logger.info(
            f"LightweightCrossEncoderReranker initialized: "
            f"enabled={self.config.enabled}, model={self.config.model_name}"
        )
    
    def _load_model(self):
        """Load cross-encoder model with error handling."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            if not self.config.silent_fallback:
                raise ImportError(
                    "sentence-transformers not available. Install with: "
                    "pip install sentence-transformers"
                )
            logger.warning(
                "sentence-transformers not available, reranking disabled"
            )
            return
        
        try:
            logger.info(f"Loading cross-encoder model: {self.config.model_name}")
            
            # Configure model loading
            model_kwargs = {
                'device': self.config.device,
                'max_length': self.config.max_length
            }
            
            if self.config.model_cache_dir:
                model_kwargs['cache_folder'] = self.config.model_cache_dir
            
            # Load model
            self.model = CrossEncoder(
                self.config.model_name,
                **model_kwargs
            )
            
            # Verify model size (approximate)
            try:
                param_count = sum(p.numel() for p in self.model.model.parameters())
                param_count_millions = param_count / 1e6
                
                if param_count_millions > 50:
                    logger.warning(
                        f"Model has {param_count_millions:.1f}M parameters, "
                        f"exceeding 50M limit. Consider using a smaller model."
                    )
                else:
                    logger.info(f"Model loaded: {param_count_millions:.1f}M parameters")
                    
            except Exception as e:
                logger.debug(f"Could not count parameters: {e}")
            
            self.model_loaded = True
            
        except Exception as e:
            error_msg = f"Failed to load cross-encoder model: {e}"
            
            if not self.config.silent_fallback:
                raise RuntimeError(error_msg)
            
            logger.warning(f"{error_msg} - reranking disabled")
            self.model = None
            self.model_loaded = False
    
    def rerank(
        self,
        query: str,
        doc_ids: List[str],
        doc_texts: List[str],
        original_scores: List[float]
    ) -> RerankingResult:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: Query text
            doc_ids: Document identifiers
            doc_texts: Document texts for reranking
            original_scores: Original retrieval scores
            
        Returns:
            RerankingResult with reranked documents
        """
        start_time = time.time()
        
        # Validate inputs
        if len(doc_ids) != len(doc_texts) or len(doc_ids) != len(original_scores):
            raise ValueError("doc_ids, doc_texts, and original_scores must have same length")
        
        # Early return if reranking disabled or no model
        if not self.config.enabled or not self.model_loaded or not self.model:
            return self._create_fallback_result(
                doc_ids, original_scores, start_time, "reranking disabled"
            )
        
        # Limit to top-K candidates for efficiency
        if len(doc_ids) > self.config.top_k_rerank:
            # Take top-K by original scores
            sorted_indices = sorted(
                range(len(original_scores)),
                key=lambda i: original_scores[i],
                reverse=True
            )[:self.config.top_k_rerank]
            
            rerank_doc_ids = [doc_ids[i] for i in sorted_indices]
            rerank_doc_texts = [doc_texts[i] for i in sorted_indices]
            rerank_orig_scores = [original_scores[i] for i in sorted_indices]
            
            logger.debug(f"Limiting reranking to top-{self.config.top_k_rerank} candidates")
        else:
            rerank_doc_ids = doc_ids
            rerank_doc_texts = doc_texts
            rerank_orig_scores = original_scores
            sorted_indices = list(range(len(doc_ids)))
        
        try:
            # Prepare query-document pairs
            query_doc_pairs = [(query, text) for text in rerank_doc_texts]
            
            # Batch inference
            rerank_scores = self._batch_predict(query_doc_pairs)
            
            # Combine reranked and non-reranked results
            final_doc_ids, final_original_scores, final_rerank_scores = self._combine_results(
                doc_ids, original_scores, sorted_indices, 
                rerank_doc_ids, rerank_orig_scores, rerank_scores
            )
            
            # Use rerank scores as final scores (could also blend)
            final_scores = final_rerank_scores
            
            batch_count = (len(query_doc_pairs) + self.config.batch_size - 1) // self.config.batch_size
            reranking_time = (time.time() - start_time) * 1000
            
            result = RerankingResult(
                doc_ids=final_doc_ids,
                original_scores=final_original_scores,
                rerank_scores=final_rerank_scores,
                final_scores=final_scores,
                model_name=self.config.model_name,
                candidates_reranked=len(rerank_doc_ids),
                reranking_time_ms=reranking_time,
                batch_count=batch_count,
                fallback_used=False
            )
            
            logger.info(
                f"Reranking complete: {len(rerank_doc_ids)} candidates, "
                f"{batch_count} batches, {reranking_time:.1f}ms"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Reranking failed: {e}"
            
            if not self.config.silent_fallback:
                raise RuntimeError(error_msg)
            
            logger.warning(f"{error_msg} - falling back to original scores")
            return self._create_fallback_result(
                doc_ids, original_scores, start_time, str(e)
            )
    
    def _batch_predict(self, query_doc_pairs: List[Tuple[str, str]]) -> List[float]:
        """Perform batch prediction with cross-encoder."""
        all_scores = []
        
        # Process in batches for memory efficiency
        for i in range(0, len(query_doc_pairs), self.config.batch_size):
            batch = query_doc_pairs[i:i + self.config.batch_size]
            
            try:
                # Get relevance scores from cross-encoder
                batch_scores = self.model.predict(batch)
                
                # Convert to list if numpy array
                if hasattr(batch_scores, 'tolist'):
                    batch_scores = batch_scores.tolist()
                elif not isinstance(batch_scores, list):
                    batch_scores = [float(batch_scores)]
                
                all_scores.extend(batch_scores)
                
            except Exception as e:
                logger.error(f"Batch prediction failed for batch {i//self.config.batch_size}: {e}")
                # Fallback to neutral scores for this batch
                all_scores.extend([0.5] * len(batch))
        
        return all_scores
    
    def _combine_results(
        self,
        all_doc_ids: List[str],
        all_original_scores: List[float], 
        reranked_indices: List[int],
        reranked_doc_ids: List[str],
        reranked_orig_scores: List[float],
        rerank_scores: List[float]
    ) -> Tuple[List[str], List[float], List[float]]:
        """
        Combine reranked and non-reranked results.
        
        Strategy: Sort reranked candidates by rerank score, then append 
        non-reranked candidates sorted by original score.
        """
        # Create rerank score mapping
        rerank_score_map = dict(zip(reranked_doc_ids, rerank_scores))
        
        # Sort reranked candidates by rerank score (descending)
        reranked_sorted = sorted(
            zip(reranked_doc_ids, reranked_orig_scores),
            key=lambda x: rerank_score_map[x[0]],
            reverse=True
        )
        
        # Get non-reranked candidates
        reranked_set = set(reranked_doc_ids)
        non_reranked = [
            (doc_id, orig_score) for doc_id, orig_score in zip(all_doc_ids, all_original_scores)
            if doc_id not in reranked_set
        ]
        
        # Sort non-reranked by original score (descending)
        non_reranked_sorted = sorted(non_reranked, key=lambda x: x[1], reverse=True)
        
        # Combine results
        final_doc_ids = []
        final_original_scores = []
        final_rerank_scores = []
        
        # Add reranked candidates first
        for doc_id, orig_score in reranked_sorted:
            final_doc_ids.append(doc_id)
            final_original_scores.append(orig_score)
            final_rerank_scores.append(rerank_score_map[doc_id])
        
        # Add non-reranked candidates
        for doc_id, orig_score in non_reranked_sorted:
            final_doc_ids.append(doc_id)
            final_original_scores.append(orig_score)
            final_rerank_scores.append(orig_score)  # Use original score as fallback
        
        return final_doc_ids, final_original_scores, final_rerank_scores
    
    def _create_fallback_result(
        self,
        doc_ids: List[str],
        original_scores: List[float],
        start_time: float,
        reason: str
    ) -> RerankingResult:
        """Create fallback result when reranking is unavailable."""
        reranking_time = (time.time() - start_time) * 1000
        
        return RerankingResult(
            doc_ids=doc_ids,
            original_scores=original_scores,
            rerank_scores=original_scores,  # Use original scores
            final_scores=original_scores,
            model_name=f"fallback ({reason})",
            candidates_reranked=0,
            reranking_time_ms=reranking_time,
            batch_count=0,
            fallback_used=True
        )
    
    def is_available(self) -> bool:
        """Check if reranking is available."""
        return self.config.enabled and self.model_loaded and self.model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self.model_loaded or not self.model:
            return {
                'available': False,
                'reason': 'model not loaded'
            }
        
        info = {
            'available': True,
            'model_name': self.config.model_name,
            'device': self.config.device,
            'max_length': self.config.max_length,
            'batch_size': self.config.batch_size
        }
        
        try:
            # Add parameter count if available
            if hasattr(self.model, 'model'):
                param_count = sum(p.numel() for p in self.model.model.parameters())
                info['parameter_count'] = param_count
                info['parameter_count_millions'] = param_count / 1e6
        except:
            pass
        
        return info


def create_reranker(config: Optional[RerankingConfig] = None) -> LightweightCrossEncoderReranker:
    """Create lightweight cross-encoder reranker."""
    return LightweightCrossEncoderReranker(config)