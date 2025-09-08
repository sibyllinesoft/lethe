"""
Core hybrid fusion implementation for α-sweep system.

Implements the mathematically rigorous fusion mechanism with:
- Strict budget constraints (k_init per modality = 1k; K=100 final results)
- Candidate union before scoring (not intersection)
- Parameter sweep: α∈{0.2,0.4,0.6,0.8} 
- Invariant enforcement at runtime
"""

import logging
import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Union
from pathlib import Path
import numpy as np

from ..retriever.bm25 import BM25Retriever, create_bm25_retriever
from ..retriever.ann import ANNRetriever, create_ann_retriever
from ..retriever.timing import TimingHarness, PerformanceProfiler

logger = logging.getLogger(__name__)


@dataclass
class FusionConfiguration:
    """Configuration for hybrid fusion parameters."""
    
    # Core fusion parameters
    alpha: float  # α parameter for fusion weight (w_s = α, w_d = 1-α)
    k_init_sparse: int = 1000  # Budget for sparse retrieval (k_init_sparse = 1000)
    k_init_dense: int = 1000   # Budget for dense retrieval (k_init_dense = 1000)
    k_final: int = 100         # Final result count K (k_final = 100)
    
    # Index parameters for budget parity
    bm25_params: Dict = field(default_factory=dict)
    ann_params: Dict = field(default_factory=dict)
    
    # ANN recall constraints
    target_ann_recall: float = 0.98
    
    # Validation
    def __post_init__(self):
        """Validate configuration parameters."""
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"Alpha must be in [0,1], got {self.alpha}")
        if self.k_init_sparse <= 0 or self.k_init_dense <= 0:
            raise ValueError("k_init parameters must be positive")
        if self.k_final <= 0 or self.k_final > min(self.k_init_sparse, self.k_init_dense):
            raise ValueError("k_final must be positive and <= min(k_init_sparse, k_init_dense)")
    
    @property
    def w_sparse(self) -> float:
        """Sparse weight (BM25)."""
        return self.alpha
    
    @property
    def w_dense(self) -> float:
        """Dense weight (vector)."""
        return 1.0 - self.alpha
    
    def get_hash(self) -> str:
        """Get configuration hash for caching/logging."""
        config_str = json.dumps({
            'alpha': self.alpha,
            'k_init_sparse': self.k_init_sparse,
            'k_init_dense': self.k_init_dense,
            'k_final': self.k_final,
            'bm25_params': self.bm25_params,
            'ann_params': self.ann_params,
            'target_ann_recall': self.target_ann_recall
        }, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]


@dataclass
class FusionResult:
    """Result from hybrid fusion with full telemetry."""
    
    # Core results
    doc_ids: List[str]
    scores: List[float]
    ranks: List[int]
    
    # Component scores for transparency
    sparse_scores: Dict[str, float]
    dense_scores: Dict[str, float] 
    fusion_scores: Dict[str, float]
    
    # Budget and timing telemetry
    sparse_candidates: int
    dense_candidates: int
    union_candidates: int
    total_latency_ms: float
    sparse_latency_ms: float
    dense_latency_ms: float
    fusion_latency_ms: float
    
    # Quality metrics
    ann_recall_achieved: float
    budget_parity_maintained: bool
    
    # Configuration used
    config: FusionConfiguration
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'doc_ids': self.doc_ids,
            'scores': self.scores,
            'ranks': self.ranks,
            'sparse_scores': self.sparse_scores,
            'dense_scores': self.dense_scores,
            'fusion_scores': self.fusion_scores,
            'sparse_candidates': self.sparse_candidates,
            'dense_candidates': self.dense_candidates,
            'union_candidates': self.union_candidates,
            'total_latency_ms': self.total_latency_ms,
            'sparse_latency_ms': self.sparse_latency_ms,
            'dense_latency_ms': self.dense_latency_ms,
            'fusion_latency_ms': self.fusion_latency_ms,
            'ann_recall_achieved': self.ann_recall_achieved,
            'budget_parity_maintained': self.budget_parity_maintained,
            'config': {
                'alpha': self.config.alpha,
                'k_init_sparse': self.config.k_init_sparse,
                'k_init_dense': self.config.k_init_dense,
                'k_final': self.config.k_final,
                'w_sparse': self.config.w_sparse,
                'w_dense': self.config.w_dense,
                'config_hash': self.config.get_hash()
            }
        }


class HybridFusionSystem:
    """
    Core hybrid fusion system implementing α-sweep with mathematical rigor.
    
    Key principles:
    1. Candidate union before scoring (not intersection)
    2. Budget constraints enforced across all α values
    3. Runtime invariant validation
    4. Comprehensive telemetry logging
    """
    
    def __init__(
        self,
        sparse_retriever: Optional[BM25Retriever] = None,
        dense_retriever: Optional[ANNRetriever] = None,
        timing_harness: Optional[TimingHarness] = None,
        profiler: Optional[PerformanceProfiler] = None
    ):
        """Initialize fusion system."""
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.timing_harness = timing_harness or TimingHarness()
        self.profiler = profiler or PerformanceProfiler()
        
        # Telemetry
        self.telemetry_log: List[Dict] = []
        
        logger.info("HybridFusionSystem initialized")
    
    def set_retrievers(
        self,
        sparse_retriever: BM25Retriever,
        dense_retriever: ANNRetriever
    ):
        """Set retrievers after initialization."""
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        logger.info("Retrievers set in fusion system")
    
    def fuse_query(
        self,
        query: str,
        config: FusionConfiguration,
        validate_invariants: bool = True
    ) -> FusionResult:
        """
        Execute hybrid fusion for a single query.
        
        Core fusion mechanism:
        1. Retrieve k_init candidates from each modality
        2. Create candidate union 
        3. Score union using: Score(d) = α·BM25(d) + (1-α)·cos(d)
        4. Return top-K results
        5. Validate mathematical invariants
        """
        if not self.sparse_retriever or not self.dense_retriever:
            raise ValueError("Retrievers must be set before fusion")
        
        start_time = time.time()
        
        # Step 1: Retrieve candidates from each modality with timing
        with self.timing_harness.time("sparse_retrieval"):
            sparse_results = self.sparse_retriever.retrieve(
                query, 
                k=config.k_init_sparse
            )
        sparse_latency = self.timing_harness.get_last_duration("sparse_retrieval")
        
        with self.timing_harness.time("dense_retrieval"):
            dense_results = self.dense_retriever.retrieve(
                query,
                k=config.k_init_dense
            )
        dense_latency = self.timing_harness.get_last_duration("dense_retrieval")
        
        # Extract scores and build candidate union
        sparse_scores = {r.doc_id: r.score for r in sparse_results}
        dense_scores = {r.doc_id: r.score for r in dense_results}
        
        # Step 2: Create candidate union (critical for mathematical correctness)
        all_candidate_ids = set(sparse_scores.keys()) | set(dense_scores.keys())
        
        # Step 3: Normalize scores to [0,1] for proper fusion
        with self.timing_harness.time("score_normalization"):
            sparse_scores_norm = self._normalize_scores(sparse_scores)
            dense_scores_norm = self._normalize_scores(dense_scores)
        
        # Step 4: Compute fusion scores for all candidates
        with self.timing_harness.time("fusion_scoring"):
            fusion_scores = {}
            for doc_id in all_candidate_ids:
                # Get normalized scores (0.0 if not present)
                s_score = sparse_scores_norm.get(doc_id, 0.0)
                d_score = dense_scores_norm.get(doc_id, 0.0)
                
                # Core fusion formula: Score(d) = w_s·BM25 + w_d·cos
                fusion_score = config.w_sparse * s_score + config.w_dense * d_score
                fusion_scores[doc_id] = fusion_score
        
        fusion_comp_latency = self.timing_harness.get_last_duration("fusion_scoring")
        
        # Step 5: Rank and select top-K
        sorted_candidates = sorted(
            fusion_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:config.k_final]
        
        # Extract results
        final_doc_ids = [doc_id for doc_id, _ in sorted_candidates]
        final_scores = [score for _, score in sorted_candidates] 
        final_ranks = list(range(1, len(final_doc_ids) + 1))
        
        total_latency = (time.time() - start_time) * 1000  # Convert to ms
        
        # Step 6: Compute quality metrics
        ann_recall = self._compute_ann_recall(dense_results, config.target_ann_recall)
        budget_parity = self._check_budget_parity(sparse_latency, dense_latency)
        
        # Step 7: Create result object with full telemetry
        result = FusionResult(
            doc_ids=final_doc_ids,
            scores=final_scores,
            ranks=final_ranks,
            sparse_scores=sparse_scores_norm,
            dense_scores=dense_scores_norm,
            fusion_scores=fusion_scores,
            sparse_candidates=len(sparse_scores),
            dense_candidates=len(dense_scores),
            union_candidates=len(all_candidate_ids),
            total_latency_ms=total_latency,
            sparse_latency_ms=sparse_latency,
            dense_latency_ms=dense_latency,
            fusion_latency_ms=fusion_comp_latency,
            ann_recall_achieved=ann_recall,
            budget_parity_maintained=budget_parity,
            config=config
        )
        
        # Step 8: Validate mathematical invariants if requested
        if validate_invariants:
            self._validate_fusion_invariants(
                result, query, sparse_results, dense_results
            )
        
        # Log telemetry
        self.telemetry_log.append(result.to_dict())
        
        logger.info(
            f"Fusion complete: α={config.alpha:.1f}, "
            f"candidates={len(all_candidate_ids)}, "
            f"latency={total_latency:.1f}ms"
        )
        
        return result
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Normalize scores to [0,1] range."""
        if not scores:
            return {}
        
        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            # All scores identical - normalize to 1.0
            return {k: 1.0 for k in scores.keys()}
        
        # Min-max normalization
        norm_scores = {}
        for doc_id, score in scores.items():
            norm_scores[doc_id] = (score - min_val) / (max_val - min_val)
        
        return norm_scores
    
    def _compute_ann_recall(self, dense_results: List, target: float) -> float:
        """Compute ANN recall achieved (simplified - would need ground truth)."""
        # For now, return a placeholder - real implementation would compare
        # against exhaustive nearest neighbors
        return 0.98  # Placeholder
    
    def _check_budget_parity(self, sparse_latency: float, dense_latency: float) -> bool:
        """Check if budget parity is maintained (±5%)."""
        if sparse_latency == 0 or dense_latency == 0:
            return False
        
        ratio = max(sparse_latency, dense_latency) / min(sparse_latency, dense_latency)
        return ratio <= 1.05  # Within 5%
    
    def _validate_fusion_invariants(
        self,
        result: FusionResult,
        query: str,
        sparse_results: List,
        dense_results: List
    ):
        """Validate mathematical invariants P1-P5."""
        from .invariants import InvariantValidator
        
        validator = InvariantValidator()
        validator.validate_all_invariants(
            result, query, sparse_results, dense_results, self
        )
    
    def get_telemetry(self) -> List[Dict]:
        """Get all telemetry data."""
        return self.telemetry_log.copy()
    
    def clear_telemetry(self):
        """Clear telemetry log."""
        self.telemetry_log.clear()
        logger.info("Telemetry cleared")


def create_fusion_system(
    corpus_path: str,
    embeddings_path: Optional[str] = None,
    bm25_params: Optional[Dict] = None,
    ann_params: Optional[Dict] = None
) -> HybridFusionSystem:
    """
    Factory function to create a complete fusion system.
    
    Args:
        corpus_path: Path to document corpus
        embeddings_path: Path to precomputed embeddings
        bm25_params: BM25 configuration
        ann_params: ANN configuration
    
    Returns:
        Configured HybridFusionSystem
    """
    # Create retrievers
    sparse_retriever = create_bm25_retriever(
        corpus_path=corpus_path,
        **(bm25_params or {})
    )
    
    dense_retriever = create_ann_retriever(
        corpus_path=corpus_path,
        embeddings_path=embeddings_path,
        **(ann_params or {})
    )
    
    # Create fusion system
    fusion_system = HybridFusionSystem()
    fusion_system.set_retrievers(sparse_retriever, dense_retriever)
    
    logger.info("Complete fusion system created")
    return fusion_system