#!/usr/bin/env python3
"""
Enhanced Hybrid Retrieval System - Milestone 3 Integration

Integrates adaptive planning, hybrid fusion, entity diversification, and 
optional reranking for sophisticated agent-context retrieval.

Key Features:
- End-to-end sub-200ms latency optimization
- Adaptive planning policy (VERIFY/EXPLORE/EXPLOIT)
- Proper BM25/cosine score normalization with α-weighting
- Entity-based diversification with session-IDF
- Optional lightweight cross-encoder reranking (≤50M params)
- Exact identifier matching guarantees
- Comprehensive configuration and telemetry
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any, NamedTuple
from pathlib import Path

# Import our Milestone 3 components
from .planning import AdaptivePlanningEngine, PlanningConfiguration, PlanningResult
from .diversification import EntityDiversificationEngine, DiversificationConfig, DiversificationResult
from .reranker import LightweightCrossEncoderReranker, RerankingConfig, RerankingResult

# Import existing components (assuming they exist)
try:
    from .fusion.core import HybridFusionSystem, FusionConfiguration, FusionResult
    from .retriever.bm25 import BM25Retriever
    from .retriever.ann import ANNRetriever
    from .retriever.timing import TimingHarness, PerformanceProfiler
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to import existing components: {e}")
    # Create placeholder classes for development
    class HybridFusionSystem: pass
    class BM25Retriever: pass
    class ANNRetriever: pass
    class TimingHarness: pass
    class PerformanceProfiler: pass
    class FusionConfiguration: pass
    class FusionResult: pass

logger = logging.getLogger(__name__)

class RetrievalResult(NamedTuple):
    """End-to-end retrieval result with full pipeline metadata."""
    
    # Final results
    doc_ids: List[str]
    scores: List[float]
    
    # Pipeline stage results
    planning_result: PlanningResult
    fusion_result: Optional[FusionResult]
    diversification_result: DiversificationResult
    reranking_result: Optional[RerankingResult]
    
    # Performance metrics
    total_latency_ms: float
    stage_latencies_ms: Dict[str, float]
    
    # Quality metrics
    exact_matches_included: int
    entity_diversity_score: float
    final_token_count: int

@dataclass
class HybridRetrievalConfig:
    """Master configuration for the complete hybrid retrieval system."""
    
    # Component configurations
    planning_config: Optional[PlanningConfiguration] = None
    fusion_config: Optional[FusionConfiguration] = None 
    diversification_config: Optional[DiversificationConfig] = None
    reranking_config: Optional[RerankingConfig] = None
    
    # Performance targets
    target_latency_ms: float = 200.0      # Sub-200ms end-to-end target
    enable_performance_monitoring: bool = True
    
    # Pipeline control
    enable_reranking: bool = False        # OFF by default per requirements
    enable_diversification: bool = True   # Entity diversification enabled
    enable_exact_matching: bool = True    # Exact identifier guarantees
    
    # Fallback behavior
    strict_latency_enforcement: bool = False  # Abort if over target
    graceful_degradation: bool = True         # Fallback to simpler pipeline
    
    # Caching and optimization
    cache_planning_decisions: bool = True
    batch_optimize_retrievers: bool = True
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.planning_config is None:
            self.planning_config = PlanningConfiguration()
        if self.diversification_config is None:
            self.diversification_config = DiversificationConfig()
        if self.reranking_config is None:
            self.reranking_config = RerankingConfig(enabled=self.enable_reranking)
        # fusion_config will be created dynamically based on planning

class EnhancedHybridRetrievalSystem:
    """
    Complete hybrid retrieval system implementing Milestone 3 requirements.
    
    Pipeline:
    1. Adaptive Planning → strategy (VERIFY/EXPLORE/EXPLOIT) + α value
    2. Hybrid Fusion → BM25 + Vector with α-weighting
    3. Optional Reranking → Cross-encoder scores (if enabled)
    4. Entity Diversification → Session-IDF weighted entity coverage
    5. Final Selection → Budget-constrained results with exact match guarantees
    """
    
    def __init__(
        self,
        config: Optional[HybridRetrievalConfig] = None,
        sparse_retriever: Optional[BM25Retriever] = None,
        dense_retriever: Optional[ANNRetriever] = None
    ):
        """Initialize enhanced hybrid retrieval system."""
        self.config = config or HybridRetrievalConfig()
        
        # Core components
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        
        # Milestone 3 components
        self.planning_engine = AdaptivePlanningEngine(self.config.planning_config)
        self.diversification_engine = EntityDiversificationEngine(self.config.diversification_config)
        self.reranker = LightweightCrossEncoderReranker(self.config.reranking_config) if self.config.enable_reranking else None
        
        # Performance monitoring
        self.timing_harness = TimingHarness() if self.config.enable_performance_monitoring else None
        self.profiler = PerformanceProfiler() if self.config.enable_performance_monitoring else None
        
        # Legacy fusion system (will be enhanced)
        self.fusion_system = HybridFusionSystem(
            sparse_retriever=sparse_retriever,
            dense_retriever=dense_retriever,
            timing_harness=self.timing_harness,
            profiler=self.profiler
        )
        
        # Query cache for performance
        self._planning_cache: Dict[str, PlanningResult] = {}
        
        # Performance tracking
        self.retrieval_stats = {
            'total_queries': 0,
            'avg_latency_ms': 0.0,
            'latency_p95_ms': 0.0,
            'exact_match_rate': 0.0,
            'diversification_rate': 0.0
        }
        
        logger.info("EnhancedHybridRetrievalSystem initialized")
    
    def retrieve(
        self,
        query: str,
        session_id: str,
        turn_idx: int,
        session_entities: Optional[List[Tuple[str, str, float]]] = None,
        doc_texts: Optional[Dict[str, str]] = None,  # For reranking and exact matching
        term_idfs: Optional[Dict[str, float]] = None
    ) -> RetrievalResult:
        """
        Execute complete hybrid retrieval pipeline.
        
        Args:
            query: Input query text
            session_id: Session identifier for context
            turn_idx: Turn index in conversation
            session_entities: Session entities for diversification
            doc_texts: Document texts for reranking/exact matching
            term_idfs: Pre-computed term IDF scores
            
        Returns:
            Complete retrieval result with all pipeline metadata
        """
        overall_start = time.time()
        stage_timings = {}
        
        try:
            # Stage 1: Adaptive Planning
            with self._time_stage("planning") as timer:
                planning_result = self._plan_retrieval(query, session_id, turn_idx, term_idfs)
            stage_timings["planning"] = timer.duration_ms
            
            # Stage 2: Hybrid Fusion with α-weighting
            with self._time_stage("fusion") as timer:
                fusion_result = self._execute_fusion(query, planning_result)
            stage_timings["fusion"] = timer.duration_ms
            
            # Extract candidates for downstream processing
            candidate_doc_ids = fusion_result.doc_ids if fusion_result else []
            candidate_scores = fusion_result.scores if fusion_result else []
            
            # Stage 3: Optional Reranking
            reranking_result = None
            if self.config.enable_reranking and self.reranker and doc_texts:
                with self._time_stage("reranking") as timer:
                    reranking_result = self._execute_reranking(
                        query, candidate_doc_ids, doc_texts, candidate_scores
                    )
                    # Update candidates with reranked results
                    candidate_doc_ids = reranking_result.doc_ids
                    candidate_scores = reranking_result.final_scores
                stage_timings["reranking"] = timer.duration_ms
            
            # Stage 4: Update Entity Database for Diversification
            if session_entities:
                self._update_entity_database(candidate_doc_ids, doc_texts or {}, session_entities)
            
            # Stage 5: Entity-Based Diversification
            with self._time_stage("diversification") as timer:
                diversification_result = self._execute_diversification(
                    query, candidate_doc_ids, candidate_scores
                )
            stage_timings["diversification"] = timer.duration_ms
            
            # Compute final metrics
            total_latency = (time.time() - overall_start) * 1000
            exact_matches = diversification_result.exact_matches_count
            entity_diversity = diversification_result.objective_value
            token_count = diversification_result.tokens_used
            
            # Create final result
            result = RetrievalResult(
                doc_ids=diversification_result.doc_ids,
                scores=diversification_result.scores,
                planning_result=planning_result,
                fusion_result=fusion_result,
                diversification_result=diversification_result,
                reranking_result=reranking_result,
                total_latency_ms=total_latency,
                stage_latencies_ms=stage_timings,
                exact_matches_included=exact_matches,
                entity_diversity_score=entity_diversity,
                final_token_count=token_count
            )
            
            # Performance monitoring
            self._update_performance_stats(result)
            
            # Check latency target
            if total_latency > self.config.target_latency_ms:
                logger.warning(
                    f"Query exceeded latency target: {total_latency:.1f}ms > {self.config.target_latency_ms}ms"
                )
                
                if self.config.strict_latency_enforcement:
                    raise RuntimeError(f"Latency target exceeded: {total_latency:.1f}ms")
            
            logger.info(
                f"Retrieval complete: {len(result.doc_ids)} docs, "
                f"strategy={planning_result.strategy.value}, "
                f"exact_matches={exact_matches}, "
                f"latency={total_latency:.1f}ms"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Retrieval pipeline failed: {e}")
            
            # Graceful degradation if enabled
            if self.config.graceful_degradation:
                return self._fallback_retrieval(query, session_id, overall_start)
            
            raise
    
    def _plan_retrieval(
        self,
        query: str,
        session_id: str,
        turn_idx: int,
        term_idfs: Optional[Dict[str, float]]
    ) -> PlanningResult:
        """Execute adaptive planning with optional caching."""
        cache_key = f"{query}:{session_id}:{turn_idx}" if self.config.cache_planning_decisions else None
        
        if cache_key and cache_key in self._planning_cache:
            return self._planning_cache[cache_key]
        
        planning_result = self.planning_engine.plan_retrieval(
            query, session_id, turn_idx, term_idfs
        )
        
        if cache_key:
            self._planning_cache[cache_key] = planning_result
        
        return planning_result
    
    def _execute_fusion(self, query: str, planning_result: PlanningResult) -> Optional[FusionResult]:
        """Execute hybrid fusion with adaptive α value."""
        if not self.sparse_retriever or not self.dense_retriever:
            logger.warning("Retrievers not available for fusion")
            return None
        
        # Create dynamic fusion config based on planning
        fusion_config = FusionConfiguration(
            alpha=planning_result.alpha,
            k_init_sparse=1000,  # Budget parameters
            k_init_dense=1000,
            k_final=100,
            ann_params={'ef_search': planning_result.ef_search}
        )
        
        return self.fusion_system.fuse_query(query, fusion_config)
    
    def _execute_reranking(
        self,
        query: str,
        doc_ids: List[str],
        doc_texts: Dict[str, str],
        scores: List[float]
    ) -> Optional[RerankingResult]:
        """Execute optional cross-encoder reranking."""
        if not self.reranker or not self.reranker.is_available():
            return None
        
        # Extract texts for available documents
        available_texts = []
        available_ids = []
        available_scores = []
        
        for doc_id, score in zip(doc_ids, scores):
            if doc_id in doc_texts:
                available_ids.append(doc_id)
                available_texts.append(doc_texts[doc_id])
                available_scores.append(score)
        
        if not available_texts:
            return None
        
        return self.reranker.rerank(query, available_ids, available_texts, available_scores)
    
    def _execute_diversification(
        self,
        query: str,
        doc_ids: List[str],
        scores: List[float]
    ) -> DiversificationResult:
        """Execute entity-based diversification."""
        return self.diversification_engine.diversify_selection(
            query, doc_ids, scores, enforce_budget=True
        )
    
    def _update_entity_database(
        self,
        doc_ids: List[str],
        doc_texts: Dict[str, str],
        session_entities: List[Tuple[str, str, float]]
    ):
        """Update entity database with session information."""
        for doc_id in doc_ids:
            if doc_id in doc_texts:
                # In practice, would extract entities per document
                # For now, assume all entities apply to all docs (simplified)
                token_count = len(doc_texts[doc_id].split())  # Simple tokenization
                
                self.diversification_engine.update_entity_database(
                    doc_id, doc_texts[doc_id], session_entities, token_count
                )
    
    def _time_stage(self, stage_name: str):
        """Context manager for timing pipeline stages."""
        return StageTimer(stage_name, self.timing_harness)
    
    def _update_performance_stats(self, result: RetrievalResult):
        """Update running performance statistics."""
        self.retrieval_stats['total_queries'] += 1
        
        # Update average latency (simple moving average)
        current_avg = self.retrieval_stats['avg_latency_ms']
        new_avg = (current_avg + result.total_latency_ms) / 2
        self.retrieval_stats['avg_latency_ms'] = new_avg
        
        # Update exact match rate
        has_exact_match = result.exact_matches_included > 0
        current_rate = self.retrieval_stats['exact_match_rate']
        new_rate = (current_rate + (1.0 if has_exact_match else 0.0)) / 2
        self.retrieval_stats['exact_match_rate'] = new_rate
        
        # Update diversification effectiveness
        diversity_effective = result.entity_diversity_score > 0.0
        current_div_rate = self.retrieval_stats['diversification_rate']
        new_div_rate = (current_div_rate + (1.0 if diversity_effective else 0.0)) / 2
        self.retrieval_stats['diversification_rate'] = new_div_rate
    
    def _fallback_retrieval(
        self,
        query: str,
        session_id: str,
        start_time: float
    ) -> RetrievalResult:
        """Fallback to simple retrieval if main pipeline fails."""
        logger.warning("Executing fallback retrieval")
        
        try:
            # Simple BM25-only retrieval
            if self.sparse_retriever:
                sparse_results = self.sparse_retriever.retrieve(query, k=20)
                doc_ids = [r.doc_id for r in sparse_results]
                scores = [r.score for r in sparse_results]
            else:
                doc_ids, scores = [], []
            
            total_latency = (time.time() - start_time) * 1000
            
            # Create minimal result
            return RetrievalResult(
                doc_ids=doc_ids,
                scores=scores,
                planning_result=PlanningResult(
                    strategy="fallback", alpha=0.5, ef_search=50,
                    features=None, confidence=0.0, reasoning="fallback mode"
                ),
                fusion_result=None,
                diversification_result=DiversificationResult(
                    doc_ids=doc_ids, scores=scores, entity_coverage={},
                    objective_value=0.0, exact_matches_count=0, tokens_used=0,
                    selection_time_ms=0.0
                ),
                reranking_result=None,
                total_latency_ms=total_latency,
                stage_latencies_ms={},
                exact_matches_included=0,
                entity_diversity_score=0.0,
                final_token_count=0
            )
            
        except Exception as fallback_error:
            logger.error(f"Fallback retrieval also failed: {fallback_error}")
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        stats = self.retrieval_stats.copy()
        
        # Add component-specific stats
        if self.planning_engine:
            stats['planning_decisions'] = len(self.planning_engine.get_decision_log())
        
        if self.diversification_engine:
            stats['entity_stats'] = self.diversification_engine.get_entity_stats()
        
        if self.reranker:
            stats['reranker_info'] = self.reranker.get_model_info()
        
        return stats
    
    def clear_caches(self):
        """Clear all caches for memory management."""
        self._planning_cache.clear()
        
        if self.planning_engine:
            self.planning_engine.clear_decision_log()
        
        logger.info("All caches cleared")


class StageTimer:
    """Context manager for timing pipeline stages."""
    
    def __init__(self, stage_name: str, timing_harness: Optional[TimingHarness]):
        self.stage_name = stage_name
        self.timing_harness = timing_harness
        self.duration_ms = 0.0
        self._start_time = 0.0
    
    def __enter__(self):
        self._start_time = time.time()
        if self.timing_harness:
            self.timing_harness.start(self.stage_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration_ms = (time.time() - self._start_time) * 1000
        if self.timing_harness:
            self.timing_harness.stop(self.stage_name)


def create_hybrid_retrieval_system(
    config: Optional[HybridRetrievalConfig] = None,
    sparse_retriever: Optional[BM25Retriever] = None,
    dense_retriever: Optional[ANNRetriever] = None
) -> EnhancedHybridRetrievalSystem:
    """Create enhanced hybrid retrieval system with all Milestone 3 components."""
    return EnhancedHybridRetrievalSystem(config, sparse_retriever, dense_retriever)