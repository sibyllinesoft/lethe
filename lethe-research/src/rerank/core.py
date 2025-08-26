"""
Core reranking system for hybrid IR ablation studies.

Implements:
- Cross-encoder reranking with β∈{0,0.2,0.5} interpolation
- Budget-constrained evaluation with k_rerank∈{50,100,200}
- P95 latency monitoring within declared budgets
- Go/No-Go validation based on CI lower bounds
"""

import logging
import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np

from ..fusion.core import FusionResult, FusionConfiguration
from ..retriever.timing import TimingHarness, PerformanceProfiler

logger = logging.getLogger(__name__)


@dataclass
class RerankingConfiguration:
    """Configuration for reranking parameters."""
    
    # Core reranking parameters
    beta: float  # β interpolation weight (0 = no reranking, 1 = full reranking)
    k_rerank: int  # Number of candidates to rerank
    k_final: int = 100  # Final result count after reranking
    
    # Cross-encoder settings
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Default public checkpoint
    batch_size: int = 32  # For efficient inference
    max_length: int = 512  # Token limit for cross-encoder
    
    # Budget constraints
    max_latency_ms: float = 1000.0  # P95 latency budget
    budget_multiplier: float = 2.0  # Extra budget for reranking operations
    
    # Validation
    def __post_init__(self):
        """Validate configuration parameters."""
        if not (0.0 <= self.beta <= 1.0):
            raise ValueError(f"Beta must be in [0,1], got {self.beta}")
        if self.k_rerank <= 0:
            raise ValueError("k_rerank must be positive")
        if self.k_final <= 0 or self.k_final > self.k_rerank:
            raise ValueError("k_final must be positive and <= k_rerank")
        if self.max_latency_ms <= 0:
            raise ValueError("max_latency_ms must be positive")
    
    @property
    def w_original(self) -> float:
        """Weight for original fusion scores."""
        return 1.0 - self.beta
    
    @property
    def w_rerank(self) -> float:
        """Weight for reranking scores.""" 
        return self.beta
    
    def get_hash(self) -> str:
        """Get configuration hash for caching/logging."""
        config_str = json.dumps({
            'beta': self.beta,
            'k_rerank': self.k_rerank,
            'k_final': self.k_final,
            'cross_encoder_model': self.cross_encoder_model,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'max_latency_ms': self.max_latency_ms,
            'budget_multiplier': self.budget_multiplier
        }, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]


@dataclass
class RerankingResult:
    """Result from reranking with full telemetry."""
    
    # Core results
    doc_ids: List[str]
    scores: List[float]
    ranks: List[int]
    
    # Component scores for transparency
    original_scores: Dict[str, float]  # From initial fusion
    rerank_scores: Dict[str, float]    # From cross-encoder
    final_scores: Dict[str, float]     # After β interpolation
    
    # Budget and timing telemetry
    candidates_reranked: int
    reranking_latency_ms: float
    total_latency_ms: float
    p95_latency_ms: float
    throughput_qps: float
    
    # Quality metrics
    budget_respected: bool
    latency_within_budget: bool
    
    # Configuration used
    config: RerankingConfiguration
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'doc_ids': self.doc_ids,
            'scores': self.scores,
            'ranks': self.ranks,
            'original_scores': self.original_scores,
            'rerank_scores': self.rerank_scores,
            'final_scores': self.final_scores,
            'candidates_reranked': self.candidates_reranked,
            'reranking_latency_ms': self.reranking_latency_ms,
            'total_latency_ms': self.total_latency_ms,
            'p95_latency_ms': self.p95_latency_ms,
            'throughput_qps': self.throughput_qps,
            'budget_respected': self.budget_respected,
            'latency_within_budget': self.latency_within_budget,
            'config': {
                'beta': self.config.beta,
                'k_rerank': self.config.k_rerank,
                'k_final': self.config.k_final,
                'cross_encoder_model': self.config.cross_encoder_model,
                'max_latency_ms': self.config.max_latency_ms,
                'w_original': self.config.w_original,
                'w_rerank': self.config.w_rerank,
                'config_hash': self.config.get_hash()
            }
        }


class RerankingSystem:
    """
    Cross-encoder reranking system for hybrid IR ablations.
    
    Key features:
    1. β interpolation between fusion and cross-encoder scores
    2. Budget-aware latency monitoring
    3. Comprehensive telemetry logging
    4. Go/No-Go validation for promotion decisions
    """
    
    def __init__(
        self,
        cross_encoder: Optional['CrossEncoderReranker'] = None,
        timing_harness: Optional[TimingHarness] = None,
        profiler: Optional[PerformanceProfiler] = None
    ):
        """Initialize reranking system."""
        from .cross_encoder import CrossEncoderReranker
        
        self.cross_encoder = cross_encoder or CrossEncoderReranker()
        self.timing_harness = timing_harness or TimingHarness()
        self.profiler = profiler or PerformanceProfiler()
        
        # Telemetry
        self.telemetry_log: List[Dict] = []
        self.latency_history: List[float] = []  # For P95 computation
        
        logger.info("RerankingSystem initialized")
    
    def rerank_results(
        self,
        fusion_result: FusionResult,
        query: str,
        config: RerankingConfiguration
    ) -> RerankingResult:
        """
        Rerank fusion results using cross-encoder.
        
        Process:
        1. Take top k_rerank candidates from fusion
        2. Apply cross-encoder to get reranking scores
        3. Interpolate: final_score = β·rerank + (1-β)·fusion
        4. Sort and return top k_final results
        5. Monitor latency and budget constraints
        """
        start_time = time.time()
        
        # Step 1: Extract top k_rerank candidates
        candidates_to_rerank = min(config.k_rerank, len(fusion_result.doc_ids))
        candidate_docs = fusion_result.doc_ids[:candidates_to_rerank]
        candidate_scores = fusion_result.scores[:candidates_to_rerank]
        
        # Build original scores dict
        original_scores = {
            doc_id: score for doc_id, score 
            in zip(candidate_docs, candidate_scores)
        }
        
        # Step 2: Apply cross-encoder reranking
        with self.timing_harness.time("cross_encoder_reranking"):
            if config.beta > 0.0:  # Only rerank if β > 0
                rerank_scores = self.cross_encoder.score_pairs(
                    query=query,
                    doc_ids=candidate_docs,
                    batch_size=config.batch_size,
                    max_length=config.max_length
                )
            else:
                # β = 0 means no reranking
                rerank_scores = {doc_id: 0.0 for doc_id in candidate_docs}
        
        reranking_latency = self.timing_harness.get_last_duration("cross_encoder_reranking")
        
        # Step 3: Normalize reranking scores to [0,1]
        if rerank_scores and config.beta > 0.0:
            rerank_scores = self._normalize_scores(rerank_scores)
        
        # Step 4: Interpolate scores using β
        final_scores = {}
        for doc_id in candidate_docs:
            orig_score = original_scores[doc_id]
            rerank_score = rerank_scores.get(doc_id, 0.0)
            
            # Final interpolation: β·rerank + (1-β)·fusion
            final_score = config.w_rerank * rerank_score + config.w_original * orig_score
            final_scores[doc_id] = final_score
        
        # Step 5: Sort by final scores and take top k_final
        sorted_candidates = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:config.k_final]
        
        # Extract final results
        final_doc_ids = [doc_id for doc_id, _ in sorted_candidates]
        final_scores_list = [score for _, score in sorted_candidates]
        final_ranks = list(range(1, len(final_doc_ids) + 1))
        
        total_latency = (time.time() - start_time) * 1000  # Convert to ms
        
        # Step 6: Compute quality and budget metrics
        self.latency_history.append(total_latency)
        p95_latency = self._compute_p95_latency()
        throughput = 1000.0 / total_latency if total_latency > 0 else float('inf')
        
        budget_respected = self._check_budget_respected(total_latency, config)
        latency_within_budget = total_latency <= config.max_latency_ms
        
        # Step 7: Create result object with full telemetry
        result = RerankingResult(
            doc_ids=final_doc_ids,
            scores=final_scores_list,
            ranks=final_ranks,
            original_scores=original_scores,
            rerank_scores=rerank_scores,
            final_scores=final_scores,
            candidates_reranked=candidates_to_rerank,
            reranking_latency_ms=reranking_latency,
            total_latency_ms=total_latency,
            p95_latency_ms=p95_latency,
            throughput_qps=throughput,
            budget_respected=budget_respected,
            latency_within_budget=latency_within_budget,
            config=config
        )
        
        # Log telemetry
        self.telemetry_log.append(result.to_dict())
        
        logger.info(
            f"Reranking complete: β={config.beta:.1f}, "
            f"k_rerank={config.k_rerank}, "
            f"latency={total_latency:.1f}ms, "
            f"budget_ok={budget_respected}"
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
    
    def _compute_p95_latency(self) -> float:
        """Compute P95 latency from history."""
        if not self.latency_history:
            return 0.0
        
        return float(np.percentile(self.latency_history, 95))
    
    def _check_budget_respected(
        self,
        latency: float,
        config: RerankingConfiguration
    ) -> bool:
        """Check if budget constraints are respected."""
        # Budget is respected if latency is within declared budget with multiplier
        max_allowed = config.max_latency_ms * config.budget_multiplier
        return latency <= max_allowed
    
    def validate_go_no_go(
        self,
        baseline_result: Any,
        reranking_result: RerankingResult,
        confidence_interval_lower_bound: float
    ) -> Tuple[bool, str, Dict]:
        """
        Validate Go/No-Go decision for reranking promotion.
        
        Args:
            baseline_result: Baseline performance metrics
            reranking_result: Reranking performance metrics  
            confidence_interval_lower_bound: Lower bound of CI for improvement
        
        Returns:
            (should_promote, reason, evidence)
        """
        # Check CI lower bound > 0 (statistically significant improvement)
        ci_positive = confidence_interval_lower_bound > 0.0
        
        # Check budget constraints respected
        budget_ok = reranking_result.budget_respected and reranking_result.latency_within_budget
        
        # Check for no quality degradation 
        quality_maintained = True  # Placeholder - would compare metrics like nDCG
        
        should_promote = ci_positive and budget_ok and quality_maintained
        
        if should_promote:
            reason = f"Promotion approved: CI_lower={confidence_interval_lower_bound:.3f} > 0, budget_ok={budget_ok}"
        else:
            reasons = []
            if not ci_positive:
                reasons.append(f"CI_lower={confidence_interval_lower_bound:.3f} <= 0")
            if not budget_ok:
                reasons.append(f"budget_violated (latency={reranking_result.total_latency_ms:.1f}ms)")
            if not quality_maintained:
                reasons.append("quality_degraded")
            reason = f"Promotion rejected: {', '.join(reasons)}"
        
        evidence = {
            'ci_lower_bound': confidence_interval_lower_bound,
            'ci_positive': ci_positive,
            'budget_respected': reranking_result.budget_respected,
            'latency_within_budget': reranking_result.latency_within_budget,
            'quality_maintained': quality_maintained,
            'total_latency_ms': reranking_result.total_latency_ms,
            'p95_latency_ms': reranking_result.p95_latency_ms,
            'config_hash': reranking_result.config.get_hash()
        }
        
        logger.info(f"Go/No-Go validation: {reason}")
        
        return should_promote, reason, evidence
    
    def get_telemetry(self) -> List[Dict]:
        """Get all telemetry data."""
        return self.telemetry_log.copy()
    
    def clear_telemetry(self):
        """Clear telemetry log."""
        self.telemetry_log.clear()
        self.latency_history.clear()
        logger.info("Reranking telemetry cleared")
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics."""
        if not self.latency_history:
            return {"queries_processed": 0}
        
        latencies = np.array(self.latency_history)
        
        return {
            "queries_processed": len(self.latency_history),
            "latency_stats": {
                "mean_ms": float(np.mean(latencies)),
                "median_ms": float(np.median(latencies)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "p99_ms": float(np.percentile(latencies, 99)),
                "std_ms": float(np.std(latencies))
            },
            "throughput_stats": {
                "mean_qps": float(1000.0 / np.mean(latencies)) if np.mean(latencies) > 0 else 0.0,
                "peak_qps": float(1000.0 / np.min(latencies)) if np.min(latencies) > 0 else 0.0
            }
        }