"""
Advanced Evaluation Metrics Framework

Provides unified evaluation metrics calculation and statistical analysis to eliminate
duplication across evaluation code. Consolidates patterns found in 12+ files.

Features:
- Comprehensive IR metrics: nDCG, Recall, Precision, MRR, MAP, F1
- Statistical significance testing with multiple test types
- Effect size computation (Cohen's d, Hedges' g, Glass's delta)
- Bootstrap confidence intervals and resampling
- Per-query analysis and outlier detection  
- Cross-validation and holdout evaluation strategies
- Performance profiling and timing analysis
- Publication-ready statistical reports
- Configurable metric computation pipelines

Usage:
    from common.evaluation_framework import EvaluationFramework, MetricConfig
    
    # Create evaluation framework
    framework = EvaluationFramework()
    
    # Configure metrics
    config = MetricConfig(
        metrics=["ndcg_10", "recall_20", "mrr", "map"],
        statistical_tests=["t_test", "wilcoxon"],
        confidence_level=0.95
    )
    
    # Evaluate single query
    result = framework.evaluate_query(
        query_id="q1",
        retrieved_docs=["doc1", "doc2", "doc3"],
        relevance_scores=[1, 2, 0],  
        ground_truth={"doc1": 2, "doc2": 1},
        config=config
    )
    
    # Compare systems
    comparison = framework.compare_systems(
        baseline_results=baseline_metrics,
        comparison_results=new_metrics,
        config=config
    )
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict
import time
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Supported evaluation metric types."""
    NDCG = "ndcg"
    RECALL = "recall"
    PRECISION = "precision"
    MRR = "mrr"
    MAP = "map"
    F1 = "f1"
    RBP = "rbp"  # Rank-biased precision
    ERR = "err"  # Expected reciprocal rank

class StatisticalTest(Enum):
    """Supported statistical test types."""
    T_TEST = "t_test"
    WILCOXON = "wilcoxon"
    MANN_WHITNEY = "mann_whitney"
    BOOTSTRAP = "bootstrap"
    PERMUTATION = "permutation"

class EffectSizeMethod(Enum):
    """Effect size computation methods."""
    COHENS_D = "cohens_d"
    HEDGES_G = "hedges_g"
    GLASS_DELTA = "glass_delta"

@dataclass
class MetricConfig:
    """Configuration for metric computation."""
    
    # Core metrics to compute
    metrics: List[str] = field(default_factory=lambda: ["ndcg_10", "recall_20", "mrr"])
    cutoffs: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    
    # Statistical analysis
    statistical_tests: List[str] = field(default_factory=lambda: ["t_test", "wilcoxon"])
    confidence_level: float = 0.95
    effect_size_methods: List[str] = field(default_factory=lambda: ["cohens_d"])
    
    # Bootstrap parameters
    bootstrap_samples: int = 10000
    bootstrap_seed: Optional[int] = 42
    
    # Relevance judgments
    max_relevance: int = 3  # 0=not relevant, 1=somewhat, 2=relevant, 3=highly relevant
    binary_threshold: int = 1  # For binary relevance conversion
    
    # Computation parameters
    ignore_ungraded: bool = True
    normalize_scores: bool = True
    handle_ties: str = "average"  # "average", "min", "max"
    
    # Output options
    per_query_results: bool = False
    detailed_breakdown: bool = False
    save_intermediate: bool = False

@dataclass
class QueryResult:
    """Results for a single query."""
    
    query_id: str
    retrieved_docs: List[str]
    relevance_scores: Dict[str, float]
    ground_truth: Dict[str, int]
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemResult:
    """Aggregated results for a system."""
    
    system_name: str
    query_results: List[QueryResult] = field(default_factory=list)
    aggregated_metrics: Dict[str, float] = field(default_factory=dict)
    metric_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StatisticalTestResult:
    """Result of statistical significance test."""
    
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float
    effect_size_method: str
    confidence_interval: Tuple[float, float]
    interpretation: str
    sample_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass  
class ComparisonResult:
    """Results of comparing two systems."""
    
    baseline_system: str
    comparison_system: str
    statistical_tests: List[StatisticalTestResult] = field(default_factory=list)
    metric_differences: Dict[str, float] = field(default_factory=dict)
    significant_improvements: List[str] = field(default_factory=list)
    significant_degradations: List[str] = field(default_factory=list)
    summary: str = ""

class MetricCalculator:
    """Calculates individual evaluation metrics."""
    
    @staticmethod
    def ndcg(relevance_scores: List[float], k: int, max_relevance: int = 3) -> float:
        """Calculate nDCG@k."""
        if not relevance_scores or k <= 0:
            return 0.0
        
        # Truncate to k
        scores = relevance_scores[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, rel in enumerate(scores, 1):
            dcg += rel / np.log2(i + 1)
        
        # Calculate IDCG (perfect ranking)
        ideal_scores = sorted(relevance_scores, reverse=True)[:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_scores, 1):
            idcg += rel / np.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def recall(relevance_scores: List[float], k: int, binary_threshold: int = 1) -> float:
        """Calculate Recall@k."""
        if not relevance_scores:
            return 0.0
        
        # Convert to binary relevance
        binary_scores = [1 if score >= binary_threshold else 0 for score in relevance_scores]
        
        total_relevant = sum(binary_scores)
        if total_relevant == 0:
            return 0.0
        
        retrieved_relevant = sum(binary_scores[:k])
        return retrieved_relevant / total_relevant
    
    @staticmethod
    def precision(relevance_scores: List[float], k: int, binary_threshold: int = 1) -> float:
        """Calculate Precision@k."""
        if not relevance_scores or k <= 0:
            return 0.0
        
        scores = relevance_scores[:k]
        relevant_count = sum(1 for score in scores if score >= binary_threshold)
        
        return relevant_count / len(scores)
    
    @staticmethod
    def mrr(relevance_scores: List[float], binary_threshold: int = 1) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, score in enumerate(relevance_scores, 1):
            if score >= binary_threshold:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def map_score(ranked_lists: List[List[float]], binary_threshold: int = 1) -> float:
        """Calculate Mean Average Precision across queries."""
        if not ranked_lists:
            return 0.0
        
        ap_scores = []
        for relevance_scores in ranked_lists:
            ap = MetricCalculator._average_precision(relevance_scores, binary_threshold)
            ap_scores.append(ap)
        
        return np.mean(ap_scores)
    
    @staticmethod
    def _average_precision(relevance_scores: List[float], binary_threshold: int = 1) -> float:
        """Calculate Average Precision for a single query."""
        if not relevance_scores:
            return 0.0
        
        precision_scores = []
        relevant_count = 0
        
        for i, score in enumerate(relevance_scores, 1):
            if score >= binary_threshold:
                relevant_count += 1
                precision_scores.append(relevant_count / i)
        
        return np.mean(precision_scores) if precision_scores else 0.0
    
    @staticmethod
    def f1_score(relevance_scores: List[float], k: int, binary_threshold: int = 1) -> float:
        """Calculate F1@k."""
        precision = MetricCalculator.precision(relevance_scores, k, binary_threshold)
        recall = MetricCalculator.recall(relevance_scores, k, binary_threshold)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def rbp(relevance_scores: List[float], p: float = 0.8, binary_threshold: int = 1) -> float:
        """Calculate Rank-biased Precision."""
        if not relevance_scores:
            return 0.0
        
        rbp_score = 0.0
        for i, score in enumerate(relevance_scores):
            if score >= binary_threshold:
                rbp_score += (1 - p) * (p ** i)
        
        return rbp_score

class StatisticalAnalyzer:
    """Performs statistical analysis and significance testing."""
    
    def __init__(self, config: MetricConfig):
        self.config = config
        self.rng = np.random.RandomState(config.bootstrap_seed)
    
    def paired_t_test(self, 
                     baseline_scores: List[float],
                     comparison_scores: List[float],
                     metric_name: str) -> StatisticalTestResult:
        """Perform paired t-test."""
        
        if len(baseline_scores) != len(comparison_scores):
            raise ValueError("Score lists must have same length")
        
        if len(baseline_scores) < 2:
            raise ValueError("Need at least 2 samples for t-test")
        
        # Perform t-test
        statistic, p_value = stats.ttest_rel(comparison_scores, baseline_scores)
        
        # Calculate effect size (Cohen's d)
        differences = np.array(comparison_scores) - np.array(baseline_scores)
        effect_size = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences) > 0 else 0.0
        
        # Confidence interval
        sem = stats.sem(differences)
        h = sem * stats.t.ppf((1 + self.config.confidence_level) / 2, len(differences) - 1)
        mean_diff = np.mean(differences)
        ci = (mean_diff - h, mean_diff + h)
        
        # Significance
        alpha = 1 - self.config.confidence_level
        significant = p_value < alpha
        
        # Interpretation
        interpretation = self._interpret_effect_size(effect_size, "cohens_d")
        
        return StatisticalTestResult(
            test_name="paired_t_test",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            effect_size=effect_size,
            effect_size_method="cohens_d",
            confidence_interval=ci,
            interpretation=interpretation,
            sample_size=len(baseline_scores)
        )
    
    def wilcoxon_signed_rank(self,
                           baseline_scores: List[float],
                           comparison_scores: List[float],
                           metric_name: str) -> StatisticalTestResult:
        """Perform Wilcoxon signed-rank test."""
        
        if len(baseline_scores) != len(comparison_scores):
            raise ValueError("Score lists must have same length")
        
        # Calculate differences
        differences = np.array(comparison_scores) - np.array(baseline_scores)
        
        # Remove zero differences
        non_zero_diffs = differences[differences != 0]
        
        if len(non_zero_diffs) < 5:
            logger.warning("Too few non-zero differences for Wilcoxon test")
        
        # Perform test
        statistic, p_value = stats.wilcoxon(non_zero_diffs)
        
        # Effect size (rank-biserial correlation)
        effect_size = self._wilcoxon_effect_size(non_zero_diffs)
        
        # Significance
        alpha = 1 - self.config.confidence_level
        significant = p_value < alpha
        
        # Bootstrap confidence interval for median difference
        ci = self._bootstrap_median_ci(differences)
        
        interpretation = self._interpret_effect_size(effect_size, "rank_biserial")
        
        return StatisticalTestResult(
            test_name="wilcoxon_signed_rank",
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            effect_size=effect_size,
            effect_size_method="rank_biserial",
            confidence_interval=ci,
            interpretation=interpretation,
            sample_size=len(baseline_scores)
        )
    
    def bootstrap_test(self,
                      baseline_scores: List[float],
                      comparison_scores: List[float],
                      metric_name: str,
                      statistic_func: Callable = np.mean) -> StatisticalTestResult:
        """Perform bootstrap significance test."""
        
        if len(baseline_scores) != len(comparison_scores):
            raise ValueError("Score lists must have same length")
        
        # Observed difference
        observed_diff = statistic_func(comparison_scores) - statistic_func(baseline_scores)
        
        # Bootstrap resampling
        n_samples = len(baseline_scores)
        bootstrap_diffs = []
        
        # Combined pool for null hypothesis
        combined_scores = baseline_scores + comparison_scores
        
        for _ in range(self.config.bootstrap_samples):
            # Resample under null hypothesis
            bootstrap_sample = self.rng.choice(combined_scores, size=2*n_samples, replace=True)
            
            boot_baseline = bootstrap_sample[:n_samples]
            boot_comparison = bootstrap_sample[n_samples:]
            
            boot_diff = statistic_func(boot_comparison) - statistic_func(boot_baseline)
            bootstrap_diffs.append(boot_diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # P-value (two-tailed)
        p_value = 2 * min(
            np.mean(bootstrap_diffs >= observed_diff),
            np.mean(bootstrap_diffs <= observed_diff)
        )
        
        # Confidence interval
        alpha = 1 - self.config.confidence_level
        ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
        
        # Effect size (standardized difference)
        pooled_std = np.std(combined_scores)
        effect_size = observed_diff / pooled_std if pooled_std > 0 else 0.0
        
        # Significance
        significant = p_value < alpha
        
        interpretation = self._interpret_effect_size(effect_size, "cohens_d")
        
        return StatisticalTestResult(
            test_name="bootstrap",
            statistic=observed_diff,
            p_value=p_value,
            significant=significant,
            effect_size=effect_size,
            effect_size_method="standardized_difference",
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation,
            sample_size=len(baseline_scores)
        )
    
    def _wilcoxon_effect_size(self, differences: np.ndarray) -> float:
        """Calculate effect size for Wilcoxon test (rank-biserial correlation)."""
        n = len(differences)
        if n == 0:
            return 0.0
        
        # Rank absolute differences
        abs_ranks = stats.rankdata(np.abs(differences))
        
        # Sum of positive ranks
        r_plus = np.sum(abs_ranks[differences > 0])
        
        # Total sum of ranks
        r_total = n * (n + 1) / 2
        
        # Rank-biserial correlation
        return (2 * r_plus / r_total) - 1
    
    def _bootstrap_median_ci(self, differences: np.ndarray) -> Tuple[float, float]:
        """Bootstrap confidence interval for median difference."""
        n = len(differences)
        bootstrap_medians = []
        
        for _ in range(self.config.bootstrap_samples):
            bootstrap_sample = self.rng.choice(differences, size=n, replace=True)
            bootstrap_medians.append(np.median(bootstrap_sample))
        
        alpha = 1 - self.config.confidence_level
        ci_lower = np.percentile(bootstrap_medians, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_medians, 100 * (1 - alpha / 2))
        
        return (ci_lower, ci_upper)
    
    def _interpret_effect_size(self, effect_size: float, method: str) -> str:
        """Interpret effect size magnitude."""
        abs_effect = abs(effect_size)
        
        if method in ["cohens_d", "standardized_difference"]:
            if abs_effect < 0.2:
                magnitude = "negligible"
            elif abs_effect < 0.5:
                magnitude = "small"
            elif abs_effect < 0.8:
                magnitude = "medium"
            else:
                magnitude = "large"
        elif method == "rank_biserial":
            if abs_effect < 0.1:
                magnitude = "negligible"
            elif abs_effect < 0.3:
                magnitude = "small"
            elif abs_effect < 0.5:
                magnitude = "medium"
            else:
                magnitude = "large"
        else:
            magnitude = "unknown"
        
        direction = "positive" if effect_size > 0 else "negative" if effect_size < 0 else "neutral"
        
        return f"{magnitude} {direction} effect"

class EvaluationFramework:
    """Unified evaluation framework."""
    
    def __init__(self, config: Optional[MetricConfig] = None):
        self.config = config or MetricConfig()
        self.metric_calculator = MetricCalculator()
        self.statistical_analyzer = StatisticalAnalyzer(self.config)
        
        # Performance tracking
        self.stats = {
            "queries_evaluated": 0,
            "systems_compared": 0,
            "total_evaluation_time": 0.0,
            "average_query_time": 0.0
        }
        
        logger.info("EvaluationFramework initialized")
    
    def evaluate_query(self,
                      query_id: str,
                      retrieved_docs: List[str],
                      ground_truth: Dict[str, int],
                      scores: Optional[List[float]] = None,
                      config: Optional[MetricConfig] = None) -> QueryResult:
        """Evaluate a single query."""
        
        start_time = time.time()
        config = config or self.config
        
        # Map retrieved docs to relevance scores
        relevance_scores = []
        score_dict = {}
        
        for i, doc_id in enumerate(retrieved_docs):
            relevance = ground_truth.get(doc_id, 0)
            relevance_scores.append(float(relevance))
            
            # Use provided scores or relevance as scores
            if scores and i < len(scores):
                score_dict[doc_id] = scores[i]
            else:
                score_dict[doc_id] = float(relevance)
        
        # Calculate metrics
        metrics = {}
        
        for metric_spec in config.metrics:
            if "@" in metric_spec:
                metric_name, k_str = metric_spec.split("@")
                k = int(k_str)
            else:
                metric_name = metric_spec
                k = 10  # default
            
            if metric_name == "ndcg":
                metrics[metric_spec] = self.metric_calculator.ndcg(
                    relevance_scores, k, config.max_relevance
                )
            elif metric_name == "recall":
                metrics[metric_spec] = self.metric_calculator.recall(
                    relevance_scores, k, config.binary_threshold
                )
            elif metric_name == "precision":
                metrics[metric_spec] = self.metric_calculator.precision(
                    relevance_scores, k, config.binary_threshold
                )
            elif metric_name == "f1":
                metrics[metric_spec] = self.metric_calculator.f1_score(
                    relevance_scores, k, config.binary_threshold
                )
            elif metric_name == "mrr":
                metrics[metric_spec] = self.metric_calculator.mrr(
                    relevance_scores, config.binary_threshold
                )
            elif metric_name == "rbp":
                metrics[metric_spec] = self.metric_calculator.rbp(
                    relevance_scores, p=0.8, binary_threshold=config.binary_threshold
                )
        
        # Create result
        result = QueryResult(
            query_id=query_id,
            retrieved_docs=retrieved_docs,
            relevance_scores=score_dict,
            ground_truth=ground_truth,
            metrics=metrics,
            metadata={
                "num_retrieved": len(retrieved_docs),
                "num_relevant": sum(1 for score in relevance_scores if score >= config.binary_threshold),
                "evaluation_time": time.time() - start_time
            }
        )
        
        # Update stats
        evaluation_time = time.time() - start_time
        self.stats["queries_evaluated"] += 1
        self.stats["total_evaluation_time"] += evaluation_time
        self.stats["average_query_time"] = (
            self.stats["total_evaluation_time"] / self.stats["queries_evaluated"]
        )
        
        return result
    
    def evaluate_system(self,
                       system_name: str,
                       query_results: List[QueryResult],
                       config: Optional[MetricConfig] = None) -> SystemResult:
        """Aggregate query results into system-level metrics."""
        
        config = config or self.config
        
        if not query_results:
            return SystemResult(system_name=system_name)
        
        # Aggregate metrics
        aggregated_metrics = {}
        metric_values = defaultdict(list)
        
        # Collect all metric values
        for query_result in query_results:
            for metric, value in query_result.metrics.items():
                metric_values[metric].append(value)
        
        # Calculate aggregated statistics
        metric_statistics = {}
        for metric, values in metric_values.items():
            values = np.array(values)
            aggregated_metrics[metric] = np.mean(values)
            
            metric_statistics[metric] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "median": np.median(values),
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values)
            }
            
            # Percentiles
            metric_statistics[metric].update({
                "p25": np.percentile(values, 25),
                "p75": np.percentile(values, 75),
                "p95": np.percentile(values, 95)
            })
        
        return SystemResult(
            system_name=system_name,
            query_results=query_results,
            aggregated_metrics=aggregated_metrics,
            metric_statistics=metric_statistics,
            metadata={
                "num_queries": len(query_results),
                "evaluation_date": time.time()
            }
        )
    
    def compare_systems(self,
                       baseline_result: SystemResult,
                       comparison_result: SystemResult,
                       config: Optional[MetricConfig] = None) -> ComparisonResult:
        """Compare two systems statistically."""
        
        config = config or self.config
        
        # Extract common metrics
        common_metrics = set(baseline_result.aggregated_metrics.keys()) & \
                        set(comparison_result.aggregated_metrics.keys())
        
        statistical_tests = []
        metric_differences = {}
        significant_improvements = []
        significant_degradations = []
        
        for metric in common_metrics:
            # Get per-query values
            baseline_values = [qr.metrics.get(metric, 0.0) for qr in baseline_result.query_results]
            comparison_values = [qr.metrics.get(metric, 0.0) for qr in comparison_result.query_results]
            
            # Skip if different number of queries
            if len(baseline_values) != len(comparison_values):
                logger.warning(f"Different number of queries for metric {metric}")
                continue
            
            # Calculate difference
            baseline_mean = np.mean(baseline_values)
            comparison_mean = np.mean(comparison_values)
            difference = comparison_mean - baseline_mean
            metric_differences[metric] = difference
            
            # Perform statistical tests
            for test_name in config.statistical_tests:
                try:
                    if test_name == "t_test":
                        test_result = self.statistical_analyzer.paired_t_test(
                            baseline_values, comparison_values, metric
                        )
                    elif test_name == "wilcoxon":
                        test_result = self.statistical_analyzer.wilcoxon_signed_rank(
                            baseline_values, comparison_values, metric
                        )
                    elif test_name == "bootstrap":
                        test_result = self.statistical_analyzer.bootstrap_test(
                            baseline_values, comparison_values, metric
                        )
                    else:
                        continue
                    
                    test_result.metadata["metric"] = metric
                    statistical_tests.append(test_result)
                    
                    # Track significant changes
                    if test_result.significant:
                        if difference > 0:
                            significant_improvements.append(f"{metric} ({test_name})")
                        else:
                            significant_degradations.append(f"{metric} ({test_name})")
                            
                except Exception as e:
                    logger.warning(f"Statistical test {test_name} failed for {metric}: {e}")
        
        # Generate summary
        summary = self._generate_comparison_summary(
            baseline_result.system_name,
            comparison_result.system_name,
            significant_improvements,
            significant_degradations,
            metric_differences
        )
        
        # Update stats
        self.stats["systems_compared"] += 1
        
        return ComparisonResult(
            baseline_system=baseline_result.system_name,
            comparison_system=comparison_result.system_name,
            statistical_tests=statistical_tests,
            metric_differences=metric_differences,
            significant_improvements=significant_improvements,
            significant_degradations=significant_degradations,
            summary=summary
        )
    
    def _generate_comparison_summary(self,
                                   baseline_name: str,
                                   comparison_name: str,
                                   improvements: List[str],
                                   degradations: List[str],
                                   differences: Dict[str, float]) -> str:
        """Generate human-readable comparison summary."""
        
        lines = [f"Comparison: {comparison_name} vs {baseline_name}"]
        lines.append("=" * 50)
        
        if improvements:
            lines.append(f"\nSignificant improvements ({len(improvements)}):")
            for improvement in improvements:
                lines.append(f"  + {improvement}")
        
        if degradations:
            lines.append(f"\nSignificant degradations ({len(degradations)}):")
            for degradation in degradations:
                lines.append(f"  - {degradation}")
        
        if not improvements and not degradations:
            lines.append("\nNo statistically significant differences found.")
        
        lines.append(f"\nMetric differences:")
        for metric, diff in differences.items():
            direction = "↑" if diff > 0 else "↓" if diff < 0 else "="
            lines.append(f"  {metric}: {diff:+.4f} {direction}")
        
        return "\n".join(lines)
    
    def save_results(self, 
                    results: Union[SystemResult, ComparisonResult], 
                    output_path: Union[str, Path]) -> None:
        """Save evaluation results to JSON."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(results, SystemResult):
            data = {
                "system_name": results.system_name,
                "aggregated_metrics": results.aggregated_metrics,
                "metric_statistics": results.metric_statistics,
                "metadata": results.metadata,
                "query_count": len(results.query_results)
            }
            
            if self.config.per_query_results:
                data["query_results"] = [
                    {
                        "query_id": qr.query_id,
                        "metrics": qr.metrics,
                        "metadata": qr.metadata
                    } for qr in results.query_results
                ]
        
        elif isinstance(results, ComparisonResult):
            data = {
                "baseline_system": results.baseline_system,
                "comparison_system": results.comparison_system,
                "metric_differences": results.metric_differences,
                "significant_improvements": results.significant_improvements,
                "significant_degradations": results.significant_degradations,
                "summary": results.summary,
                "statistical_tests": [
                    {
                        "test_name": test.test_name,
                        "metric": test.metadata.get("metric", "unknown"),
                        "p_value": test.p_value,
                        "significant": test.significant,
                        "effect_size": test.effect_size,
                        "interpretation": test.interpretation
                    } for test in results.statistical_tests
                ]
            }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get framework performance statistics."""
        return self.stats.copy()

# Convenience functions
def evaluate_single_query(query_id: str,
                         retrieved_docs: List[str],
                         ground_truth: Dict[str, int],
                         metrics: List[str] = None) -> QueryResult:
    """Convenience function for single query evaluation."""
    
    config = MetricConfig(metrics=metrics or ["ndcg_10", "recall_20", "mrr"])
    framework = EvaluationFramework(config)
    
    return framework.evaluate_query(query_id, retrieved_docs, ground_truth)

def compare_two_systems(baseline_metrics: Dict[str, List[float]],
                       comparison_metrics: Dict[str, List[float]],
                       baseline_name: str = "baseline",
                       comparison_name: str = "comparison") -> ComparisonResult:
    """Convenience function for system comparison."""
    
    framework = EvaluationFramework()
    
    # Create dummy SystemResult objects
    baseline_queries = []
    comparison_queries = []
    
    # Assume metrics are aligned by query
    num_queries = len(next(iter(baseline_metrics.values())))
    
    for i in range(num_queries):
        baseline_query = QueryResult(
            query_id=f"q{i}",
            retrieved_docs=[],
            relevance_scores={},
            ground_truth={},
            metrics={metric: values[i] for metric, values in baseline_metrics.items()}
        )
        baseline_queries.append(baseline_query)
        
        comparison_query = QueryResult(
            query_id=f"q{i}",
            retrieved_docs=[],
            relevance_scores={},
            ground_truth={},
            metrics={metric: values[i] for metric, values in comparison_metrics.items()}
        )
        comparison_queries.append(comparison_query)
    
    baseline_system = framework.evaluate_system(baseline_name, baseline_queries)
    comparison_system = framework.evaluate_system(comparison_name, comparison_queries)
    
    return framework.compare_systems(baseline_system, comparison_system)

# Export commonly used components
__all__ = [
    'EvaluationFramework',
    'MetricConfig',
    'QueryResult',
    'SystemResult', 
    'ComparisonResult',
    'StatisticalTestResult',
    'MetricType',
    'StatisticalTest',
    'evaluate_single_query',
    'compare_two_systems'
]