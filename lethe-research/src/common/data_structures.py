"""
Common Data Structures

Provides shared data structures to eliminate redundant definitions across the codebase.
These structures consolidate common patterns for evaluation results, performance metrics,
and query/document representations.

Usage:
    from common.data_structures import (
        EvaluationResult, PerformanceMetrics, QueryInfo, DocumentInfo
    )

    # Create result with embedded performance metrics
    result = EvaluationResult(
        query_id="q1",
        system_name="bm25",
        performance=PerformanceMetrics(latency_ms=150.5, memory_mb=45.2),
        ranking_metrics={"ndcg_10": 0.75}
    )
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import numpy as np


@dataclass
class PerformanceMetrics:
    """Standardized performance metrics for any operation."""
    
    # Core timing metrics
    latency_ms: float = 0.0
    
    # Memory metrics (in MB) 
    memory_mb: float = 0.0
    memory_peak_mb: Optional[float] = None
    
    # CPU utilization (%)
    cpu_percent: float = 0.0
    
    # Computational cost estimates
    flops_estimate: Optional[int] = None
    
    # Execution metadata
    timestamp: float = field(default_factory=time.time)
    execution_context: Optional[str] = None
    
    # Statistical metadata (for aggregated metrics)
    measurements_count: int = 1
    
    def __post_init__(self):
        """Initialize derived fields."""
        if self.memory_peak_mb is None:
            self.memory_peak_mb = self.memory_mb
    
    def add_measurement(self, other: 'PerformanceMetrics'):
        """Add another measurement to create aggregated metrics."""
        total_count = self.measurements_count + other.measurements_count
        
        # Weighted averages
        self.latency_ms = (
            self.latency_ms * self.measurements_count + 
            other.latency_ms * other.measurements_count
        ) / total_count
        
        self.memory_mb = (
            self.memory_mb * self.measurements_count +
            other.memory_mb * other.measurements_count  
        ) / total_count
        
        self.cpu_percent = (
            self.cpu_percent * self.measurements_count +
            other.cpu_percent * other.measurements_count
        ) / total_count
        
        # Take max for peak values
        if self.memory_peak_mb and other.memory_peak_mb:
            self.memory_peak_mb = max(self.memory_peak_mb, other.memory_peak_mb)
        
        # Update count
        self.measurements_count = total_count


@dataclass
class QueryInfo:
    """Standardized query representation."""
    
    query_id: str
    text: str
    
    # Query metadata
    session_id: Optional[str] = None
    domain: str = "general"  
    complexity: str = "medium"
    query_length: Optional[int] = None
    
    # Ground truth for evaluation
    ground_truth_docs: List[str] = field(default_factory=list)
    relevance_judgments: Dict[str, int] = field(default_factory=dict)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize derived fields."""
        if self.query_length is None:
            self.query_length = len(self.text.split())


@dataclass  
class DocumentInfo:
    """Standardized document representation."""
    
    doc_id: str
    content: str
    
    # Document type and metadata
    kind: str = "text"  # 'text', 'code', 'tool_output'
    
    # Optional embeddings (for dense retrieval)
    embedding: Optional[np.ndarray] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Single retrieval result with score and metadata."""
    
    doc_id: str
    score: float
    rank: int
    
    # Document content (optional for efficiency)
    content: Optional[str] = None
    kind: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """
    Consolidated evaluation result structure.
    
    Replaces both BaselineResult and MetricsResult with a unified structure
    that can handle both individual query results and aggregated statistics.
    """
    
    # Identifiers
    query_id: str
    system_name: str  # replaces baseline_name
    
    # Query information
    query_text: Optional[str] = None
    query_info: Optional[QueryInfo] = None
    
    # Retrieval results
    retrieved_docs: List[str] = field(default_factory=list)
    retrieval_results: List[RetrievalResult] = field(default_factory=list)
    
    # Performance metrics (consolidated)
    performance: Optional[PerformanceMetrics] = None
    
    # Ranking metrics (flexible structure)
    ranking_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Validation and quality metrics
    validation_passed: bool = True
    candidate_count: Optional[int] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_baseline_result(cls, baseline_result) -> 'EvaluationResult':
        """Convert from legacy BaselineResult structure."""
        # Create performance metrics
        performance = PerformanceMetrics(
            latency_ms=getattr(baseline_result, 'latency_ms', 0.0),
            memory_mb=getattr(baseline_result, 'memory_mb', 0.0),
            cpu_percent=getattr(baseline_result, 'cpu_percent', 0.0),
            flops_estimate=getattr(baseline_result, 'flops_estimate', None)
        )
        
        # Create retrieval results
        retrieval_results = []
        if hasattr(baseline_result, 'retrieved_docs') and hasattr(baseline_result, 'relevance_scores'):
            for i, doc_id in enumerate(baseline_result.retrieved_docs):
                score = baseline_result.relevance_scores[i] if i < len(baseline_result.relevance_scores) else 0.0
                rank = baseline_result.ranks[i] if hasattr(baseline_result, 'ranks') and i < len(baseline_result.ranks) else i + 1
                
                retrieval_results.append(RetrievalResult(
                    doc_id=doc_id,
                    score=score,
                    rank=rank
                ))
        
        return cls(
            query_id=baseline_result.query_id,
            system_name=baseline_result.baseline_name,
            query_text=getattr(baseline_result, 'query_text', None),
            retrieved_docs=getattr(baseline_result, 'retrieved_docs', []),
            retrieval_results=retrieval_results,
            performance=performance,
            validation_passed=getattr(baseline_result, 'non_empty_validated', True) and 
                            getattr(baseline_result, 'smoke_test_passed', True),
            candidate_count=getattr(baseline_result, 'candidate_count', None)
        )
    
    @classmethod 
    def from_metrics_result(cls, metrics_result) -> 'EvaluationResult':
        """Convert from legacy MetricsResult structure."""
        # Create performance metrics
        performance = PerformanceMetrics(
            latency_ms=getattr(metrics_result, 'latency_ms', 0.0),
            memory_mb=getattr(metrics_result, 'memory_mb', 0.0),
            flops_estimate=getattr(metrics_result, 'flops_estimate', None)
        )
        
        # Extract ranking metrics
        ranking_metrics = {}
        for attr in ['ndcg_10', 'ndcg_5', 'recall_10', 'recall_20', 'mrr_10', 
                     'precision_10', 'map_score']:
            if hasattr(metrics_result, attr):
                ranking_metrics[attr] = getattr(metrics_result, attr)
        
        # Create query info if available
        query_info = None
        if hasattr(metrics_result, 'num_relevant') or hasattr(metrics_result, 'query_length'):
            query_info = QueryInfo(
                query_id=metrics_result.query_id,
                text="",  # Not available in MetricsResult
                query_length=getattr(metrics_result, 'query_length', None)
            )
        
        return cls(
            query_id=metrics_result.query_id,
            system_name=metrics_result.baseline_name,
            query_info=query_info,
            performance=performance,
            ranking_metrics=ranking_metrics,
            metadata={
                'num_relevant': getattr(metrics_result, 'num_relevant', None),
                'num_retrieved': getattr(metrics_result, 'num_retrieved', None)
            }
        )
    
    def get_ranking_metric(self, metric_name: str, default: float = 0.0) -> float:
        """Get ranking metric with default value."""
        return self.ranking_metrics.get(metric_name, default)
    
    def set_ranking_metric(self, metric_name: str, value: float):
        """Set ranking metric."""
        self.ranking_metrics[metric_name] = value
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary dictionary."""
        if not self.performance:
            return {}
        
        return {
            'latency_ms': self.performance.latency_ms,
            'memory_mb': self.performance.memory_mb, 
            'cpu_percent': self.performance.cpu_percent,
            'flops_estimate': self.performance.flops_estimate
        }


@dataclass
class AggregatedResults:
    """Aggregated results across multiple queries/systems."""
    
    system_name: str
    
    # Individual results
    individual_results: List[EvaluationResult] = field(default_factory=list)
    
    # Aggregated metrics
    mean_metrics: Dict[str, float] = field(default_factory=dict)
    std_metrics: Dict[str, float] = field(default_factory=dict)
    median_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Performance summary
    mean_performance: Optional[PerformanceMetrics] = None
    
    # Statistical metadata
    query_count: int = 0
    
    def add_result(self, result: EvaluationResult):
        """Add individual result to aggregation."""
        self.individual_results.append(result)
        self.query_count = len(self.individual_results)
        self._recalculate_aggregates()
    
    def _recalculate_aggregates(self):
        """Recalculate aggregated statistics."""
        if not self.individual_results:
            return
        
        # Aggregate ranking metrics
        all_metrics = {}
        for result in self.individual_results:
            for metric, value in result.ranking_metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        # Calculate statistics
        self.mean_metrics = {}
        self.std_metrics = {}
        self.median_metrics = {}
        
        for metric, values in all_metrics.items():
            if values:
                import statistics
                self.mean_metrics[metric] = statistics.mean(values)
                self.std_metrics[metric] = statistics.stdev(values) if len(values) > 1 else 0.0
                self.median_metrics[metric] = statistics.median(values)
        
        # Aggregate performance metrics
        performance_metrics = [r.performance for r in self.individual_results if r.performance]
        if performance_metrics:
            self.mean_performance = PerformanceMetrics(
                latency_ms=statistics.mean([p.latency_ms for p in performance_metrics]),
                memory_mb=statistics.mean([p.memory_mb for p in performance_metrics]),
                cpu_percent=statistics.mean([p.cpu_percent for p in performance_metrics]),
                measurements_count=len(performance_metrics)
            )


# Export commonly used structures
__all__ = [
    'PerformanceMetrics',
    'QueryInfo', 
    'DocumentInfo',
    'RetrievalResult',
    'EvaluationResult',
    'AggregatedResults'
]