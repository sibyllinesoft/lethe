#!/usr/bin/env python3
"""
Evaluation Engine with Statistical Rigor
=========================================

Implements fair evaluation protocols with:
- Matched budget constraints (8%, 15%, 30% keep_ratio)
- Statistical significance testing (bootstrap + permutation)
- Multiple comparison correction (Holm method)
- Performance profiling and quality metrics
"""

import logging
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
import json
from pathlib import Path
from scipy import stats
from scipy.stats import bootstrap
import pandas as pd

from .datasets.base import DatasetSample
from .competitors.base import BaseCompetitor, CompetitorResult
from .config import EvaluationConfig

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results for a single competitor on a single dataset."""
    
    competitor_name: str
    dataset_name: str
    keep_ratio: float
    k_value: int
    
    # Individual query results
    query_results: List[CompetitorResult]
    
    # Aggregated metrics
    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    precision_at_k: float
    recall_at_k: float
    
    mean_delta_cbu_per_1k: float  # CBU = Context Budget Units
    mean_tokens_kept: float
    
    # Quality metrics
    exact_match_rate: float
    
    # Success metrics
    success_rate: float
    error_count: int
    
    # Optional quality metrics with defaults
    mean_entity_diversity: float = 0.0
    
    # Raw data for statistical testing
    latencies: List[float] = field(default_factory=list)
    precisions: List[float] = field(default_factory=list)
    recalls: List[float] = field(default_factory=list)
    delta_cbus: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Compute derived fields from query results."""
        if not self.query_results:
            return
        
        successful_results = [r for r in self.query_results if r.success]
        
        # Extract raw metrics for statistical testing
        self.latencies = [r.latency_ms for r in successful_results]
        self.precisions = []  # Will be computed with ground truth
        self.recalls = []     # Will be computed with ground truth  
        self.delta_cbus = []  # Will be computed with budget analysis
        
        # Update success metrics
        self.success_rate = len(successful_results) / len(self.query_results)
        self.error_count = len(self.query_results) - len(successful_results)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'competitor_name': self.competitor_name,
            'dataset_name': self.dataset_name,
            'keep_ratio': self.keep_ratio,
            'k_value': self.k_value,
            'mean_latency_ms': self.mean_latency_ms,
            'p95_latency_ms': self.p95_latency_ms,
            'p99_latency_ms': self.p99_latency_ms,
            'precision_at_k': self.precision_at_k,
            'recall_at_k': self.recall_at_k,
            'mean_delta_cbu_per_1k': self.mean_delta_cbu_per_1k,
            'mean_tokens_kept': self.mean_tokens_kept,
            'exact_match_rate': self.exact_match_rate,
            'mean_entity_diversity': self.mean_entity_diversity,
            'success_rate': self.success_rate,
            'error_count': self.error_count,
            'query_count': len(self.query_results)
        }


@dataclass
class StatisticalComparison:
    """Statistical comparison between two competitors."""
    
    competitor_a: str
    competitor_b: str
    metric_name: str
    
    # Effect size and confidence interval
    effect_size: float  # Cohen's d or similar
    confidence_interval: Tuple[float, float]
    
    # Statistical tests
    bootstrap_p_value: float
    permutation_p_value: float
    corrected_p_value: float  # After multiple comparison correction
    
    # Practical significance
    is_significant: bool
    practical_improvement: bool
    
    # Raw statistics
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'competitor_a': self.competitor_a,
            'competitor_b': self.competitor_b,
            'metric_name': self.metric_name,
            'effect_size': self.effect_size,
            'confidence_interval': list(self.confidence_interval),
            'bootstrap_p_value': self.bootstrap_p_value,
            'permutation_p_value': self.permutation_p_value,
            'corrected_p_value': self.corrected_p_value,
            'is_significant': self.is_significant,
            'practical_improvement': self.practical_improvement,
            'mean_a': self.mean_a,
            'mean_b': self.mean_b,
            'std_a': self.std_a,
            'std_b': self.std_b
        }


class EvaluationEngine:
    """Engine for fair evaluation with statistical rigor."""
    
    def __init__(self, config: EvaluationConfig):
        """Initialize evaluation engine."""
        self.config = config
        self.results_cache: Dict[str, EvaluationResult] = {}
        
        # Ground truth cache for relevance judgments
        self.ground_truth_cache: Dict[str, Set[str]] = {}
        
        logger.info("EvaluationEngine initialized")
    
    def evaluate_competitor(
        self,
        competitor: BaseCompetitor,
        dataset: List[DatasetSample],
        dataset_name: str,
        keep_ratio: float,
        k: int = 100
    ) -> EvaluationResult:
        """Evaluate a single competitor on a dataset."""
        
        logger.info(
            f"Evaluating {competitor.name} on {dataset_name} "
            f"with keep_ratio={keep_ratio:.1%}, k={k}"
        )
        
        query_results = []
        start_time = time.time()
        
        for i, sample in enumerate(dataset):
            if i % 50 == 0:
                logger.info(f"Processing sample {i}/{len(dataset)}")
            
            try:
                # Execute retrieval with timing
                query_start = time.time()
                result = competitor.retrieve(
                    query=sample.query,
                    context=sample.context,
                    keep_ratio=keep_ratio,
                    k=k
                )
                query_time = (time.time() - query_start) * 1000
                
                # Update result with computed metrics
                result.latency_ms = query_time
                result.competitor_name = competitor.name
                result.original_context_tokens = sample.context_length
                
                # Cache ground truth for this sample
                cache_key = f"{dataset_name}_{sample.id}"
                if cache_key not in self.ground_truth_cache:
                    self.ground_truth_cache[cache_key] = self._extract_ground_truth(sample)
                
                query_results.append(result)
                
            except Exception as e:
                logger.error(f"Query failed for sample {sample.id}: {e}")
                # Create failed result
                failed_result = CompetitorResult(
                    doc_ids=[],
                    scores=[],
                    latency_ms=0.0,
                    competitor_name=competitor.name,
                    success=False,
                    error_message=str(e)
                )
                query_results.append(failed_result)
        
        total_time = time.time() - start_time
        logger.info(f"Completed evaluation in {total_time:.1f}s")
        
        # Compute aggregated metrics
        evaluation_result = self._compute_aggregated_metrics(
            competitor.name, dataset_name, keep_ratio, k, query_results, dataset
        )
        
        # Cache result
        cache_key = f"{competitor.name}_{dataset_name}_{keep_ratio}_{k}"
        self.results_cache[cache_key] = evaluation_result
        
        return evaluation_result
    
    def evaluate_all_competitors(
        self,
        competitors: List[BaseCompetitor],
        datasets: Dict[str, List[DatasetSample]],
        max_workers: int = 4
    ) -> Dict[str, List[EvaluationResult]]:
        """Evaluate all competitors on all datasets with all budget ratios."""
        
        # Generate all evaluation tasks
        tasks = []
        for competitor in competitors:
            for dataset_name, dataset_samples in datasets.items():
                for keep_ratio in self.config.keep_ratios:
                    tasks.append((competitor, dataset_samples, dataset_name, keep_ratio))
        
        logger.info(f"Starting evaluation of {len(tasks)} tasks with {max_workers} workers")
        
        results = {}
        completed_tasks = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for competitor, dataset_samples, dataset_name, keep_ratio in tasks:
                future = executor.submit(
                    self.evaluate_competitor,
                    competitor, dataset_samples, dataset_name, keep_ratio
                )
                future_to_task[future] = (competitor.name, dataset_name, keep_ratio)
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                competitor_name, dataset_name, keep_ratio = future_to_task[future]
                
                try:
                    result = future.result()
                    
                    # Store result
                    key = f"{competitor_name}_{dataset_name}"
                    if key not in results:
                        results[key] = []
                    results[key].append(result)
                    
                    completed_tasks += 1
                    progress = completed_tasks / len(tasks) * 100
                    
                    logger.info(
                        f"Completed {competitor_name} on {dataset_name} "
                        f"with keep_ratio={keep_ratio:.1%} ({progress:.1f}% done)"
                    )
                    
                except Exception as e:
                    logger.error(
                        f"Failed evaluation for {competitor_name} on {dataset_name}: {e}"
                    )
        
        logger.info(f"All evaluations completed: {len(results)} competitor-dataset pairs")
        return results
    
    def compare_competitors(
        self,
        results: Dict[str, List[EvaluationResult]],
        baseline_competitor: str = "lethe_hybrid"
    ) -> List[StatisticalComparison]:
        """Perform statistical comparisons between competitors."""
        
        logger.info(f"Computing statistical comparisons with baseline: {baseline_competitor}")
        
        comparisons = []
        metrics_to_compare = [
            "mean_latency_ms", "precision_at_k", "recall_at_k", 
            "mean_delta_cbu_per_1k", "exact_match_rate"
        ]
        
        # Find baseline results
        baseline_results = {}
        for key, result_list in results.items():
            if baseline_competitor in key:
                baseline_results[key] = result_list
        
        if not baseline_results:
            logger.warning(f"No baseline results found for {baseline_competitor}")
            return comparisons
        
        # Compare each competitor against baseline
        for competitor_key, competitor_results in results.items():
            if baseline_competitor in competitor_key:
                continue  # Skip self-comparison
                
            competitor_name = competitor_key.split('_')[0]
            
            # Find matching baseline results (same dataset)
            dataset_name = '_'.join(competitor_key.split('_')[1:])
            baseline_key = f"{baseline_competitor}_{dataset_name}"
            
            if baseline_key not in baseline_results:
                continue
            
            baseline_data = baseline_results[baseline_key]
            
            # Compare each metric
            for metric_name in metrics_to_compare:
                try:
                    comparison = self._statistical_comparison(
                        competitor_results, baseline_data,
                        competitor_name, baseline_competitor, metric_name
                    )
                    comparisons.append(comparison)
                except Exception as e:
                    logger.error(f"Failed comparison {competitor_name} vs {baseline_competitor} on {metric_name}: {e}")
        
        # Apply multiple comparison correction
        self._apply_multiple_comparison_correction(comparisons)
        
        logger.info(f"Computed {len(comparisons)} statistical comparisons")
        return comparisons
    
    def _compute_aggregated_metrics(
        self,
        competitor_name: str,
        dataset_name: str, 
        keep_ratio: float,
        k: int,
        query_results: List[CompetitorResult],
        dataset_samples: List[DatasetSample]
    ) -> EvaluationResult:
        """Compute aggregated metrics from individual query results."""
        
        successful_results = [r for r in query_results if r.success]
        
        if not successful_results:
            # All queries failed
            return EvaluationResult(
                competitor_name=competitor_name,
                dataset_name=dataset_name,
                keep_ratio=keep_ratio,
                k_value=k,
                query_results=query_results,
                mean_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                precision_at_k=0.0,
                recall_at_k=0.0,
                mean_delta_cbu_per_1k=0.0,
                mean_tokens_kept=0.0,
                exact_match_rate=0.0,
                success_rate=0.0,
                error_count=len(query_results)
            )
        
        # Compute latency statistics
        latencies = [r.latency_ms for r in successful_results]
        latencies.sort()
        n = len(latencies)
        
        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # Compute precision and recall metrics
        precisions = []
        recalls = []
        exact_matches = []
        delta_cbus = []
        tokens_kept_list = []
        
        for i, result in enumerate(successful_results):
            sample = dataset_samples[i] if i < len(dataset_samples) else None
            if not sample:
                continue
            
            # Get ground truth for this sample
            cache_key = f"{dataset_name}_{sample.id}"
            ground_truth = self.ground_truth_cache.get(cache_key, set())
            
            # Compute precision and recall
            if result.doc_ids:
                retrieved_set = set(result.doc_ids[:k])
                if ground_truth:
                    precision = len(retrieved_set & ground_truth) / len(retrieved_set)
                    recall = len(retrieved_set & ground_truth) / len(ground_truth)
                else:
                    # No ground truth available, assume perfect relevance for top result
                    precision = 1.0 if result.doc_ids else 0.0
                    recall = precision
                
                precisions.append(precision)
                recalls.append(recall)
            
            # Exact match detection (simplified - check if answer appears in retrieved text)
            exact_match = 1.0 if sample.answer.lower() in " ".join(result.doc_ids).lower() else 0.0
            exact_matches.append(exact_match)
            
            # Compute delta CBU (Context Budget Units)
            if result.original_context_tokens > 0:
                cbu_ratio = result.tokens_kept / result.original_context_tokens
                delta_cbu = (cbu_ratio - keep_ratio) * 1000  # Per 1k normalization
                delta_cbus.append(delta_cbu)
            
            tokens_kept_list.append(result.tokens_kept)
        
        # Aggregate metrics
        result = EvaluationResult(
            competitor_name=competitor_name,
            dataset_name=dataset_name,
            keep_ratio=keep_ratio,
            k_value=k,
            query_results=query_results,
            mean_latency_ms=mean_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            precision_at_k=np.mean(precisions) if precisions else 0.0,
            recall_at_k=np.mean(recalls) if recalls else 0.0,
            mean_delta_cbu_per_1k=np.mean(delta_cbus) if delta_cbus else 0.0,
            mean_tokens_kept=np.mean(tokens_kept_list) if tokens_kept_list else 0.0,
            exact_match_rate=np.mean(exact_matches) if exact_matches else 0.0,
            success_rate=len(successful_results) / len(query_results),
            error_count=len(query_results) - len(successful_results)
        )
        
        # Store raw data for statistical testing
        result.latencies = latencies
        result.precisions = precisions
        result.recalls = recalls 
        result.delta_cbus = delta_cbus
        
        return result
    
    def _extract_ground_truth(self, sample: DatasetSample) -> Set[str]:
        """Extract ground truth relevant documents for a sample."""
        # For most datasets, we don't have explicit relevance judgments
        # Use simple heuristics based on the task type
        
        ground_truth = set()
        
        task_type = sample.metadata.get("task_type", "")
        
        if task_type in ["needle_in_haystack", "key_value_retrieval", "number_retrieval"]:
            # For retrieval tasks, the answer itself is the ground truth
            ground_truth.add(sample.answer)
        
        elif task_type in ["code_debugging", "code_qa"]:
            # For code tasks, look for function/class names mentioned in the answer
            import re
            code_entities = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', sample.answer)
            ground_truth.update(code_entities[:5])  # Top 5 entities
        
        elif sample.answer:
            # General case - use answer text as ground truth
            ground_truth.add(sample.answer)
        
        return ground_truth
    
    def _statistical_comparison(
        self,
        results_a: List[EvaluationResult],
        results_b: List[EvaluationResult], 
        competitor_a: str,
        competitor_b: str,
        metric_name: str
    ) -> StatisticalComparison:
        """Perform statistical comparison between two result sets."""
        
        # Extract metric values
        values_a = []
        values_b = []
        
        for result in results_a:
            if hasattr(result, metric_name):
                values_a.append(getattr(result, metric_name))
        
        for result in results_b:
            if hasattr(result, metric_name):
                values_b.append(getattr(result, metric_name))
        
        if not values_a or not values_b:
            raise ValueError(f"No values found for metric {metric_name}")
        
        # Convert to numpy arrays
        a_array = np.array(values_a)
        b_array = np.array(values_b)
        
        # Basic statistics
        mean_a = np.mean(a_array)
        mean_b = np.mean(b_array)
        std_a = np.std(a_array, ddof=1) if len(a_array) > 1 else 0.0
        std_b = np.std(b_array, ddof=1) if len(b_array) > 1 else 0.0
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(a_array) - 1) * std_a**2 + (len(b_array) - 1) * std_b**2) / 
                            (len(a_array) + len(b_array) - 2))
        effect_size = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0
        
        # Bootstrap confidence interval for difference
        def diff_statistic(x, y):
            return np.mean(x) - np.mean(y)
        
        try:
            # Create bootstrap sample
            rng = np.random.default_rng(42)
            bootstrap_result = bootstrap(
                (a_array, b_array),
                diff_statistic,
                paired=False,
                n_resamples=self.config.statistical_testing["bootstrap_iterations"],
                confidence_level=self.config.statistical_testing["confidence_level"],
                random_state=rng
            )
            confidence_interval = (bootstrap_result.confidence_interval.low, 
                                 bootstrap_result.confidence_interval.high)
            
        except Exception as e:
            logger.warning(f"Bootstrap CI failed: {e}")
            confidence_interval = (mean_a - mean_b, mean_a - mean_b)
        
        # Permutation test
        try:
            combined = np.concatenate([a_array, b_array])
            observed_diff = mean_a - mean_b
            
            n_a = len(a_array)
            n_permutations = self.config.statistical_testing["permutation_iterations"]
            
            rng = np.random.default_rng(42)
            permuted_diffs = []
            
            for _ in range(n_permutations):
                shuffled = rng.permutation(combined)
                perm_a = shuffled[:n_a]
                perm_b = shuffled[n_a:]
                perm_diff = np.mean(perm_a) - np.mean(perm_b)
                permuted_diffs.append(perm_diff)
            
            permuted_diffs = np.array(permuted_diffs)
            permutation_p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
            
        except Exception as e:
            logger.warning(f"Permutation test failed: {e}")
            # Fallback to t-test
            t_stat, permutation_p_value = stats.ttest_ind(a_array, b_array)
            permutation_p_value = abs(permutation_p_value)
        
        # Bootstrap p-value (proportion of bootstrap samples with opposite sign)
        try:
            bootstrap_diffs = []
            rng = np.random.default_rng(42)
            for _ in range(self.config.statistical_testing["bootstrap_iterations"]):
                boot_a = rng.choice(a_array, size=len(a_array), replace=True)
                boot_b = rng.choice(b_array, size=len(b_array), replace=True)
                bootstrap_diffs.append(np.mean(boot_a) - np.mean(boot_b))
            
            bootstrap_diffs = np.array(bootstrap_diffs)
            bootstrap_p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        except Exception:
            bootstrap_p_value = permutation_p_value
        
        # Practical significance
        effect_threshold = self.config.statistical_testing.get("effect_size_threshold", 0.1)
        practical_improvement = abs(effect_size) > effect_threshold
        
        return StatisticalComparison(
            competitor_a=competitor_a,
            competitor_b=competitor_b,
            metric_name=metric_name,
            effect_size=effect_size,
            confidence_interval=confidence_interval,
            bootstrap_p_value=bootstrap_p_value,
            permutation_p_value=permutation_p_value,
            corrected_p_value=permutation_p_value,  # Will be corrected later
            is_significant=False,  # Will be updated after correction
            practical_improvement=practical_improvement,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b
        )
    
    def _apply_multiple_comparison_correction(self, comparisons: List[StatisticalComparison]):
        """Apply Holm correction for multiple comparisons."""
        
        # Extract p-values
        p_values = [comp.permutation_p_value for comp in comparisons]
        
        # Apply Holm correction
        corrected_results = self._holm_correction(p_values)
        
        # Update comparison objects
        alpha = 1 - self.config.statistical_testing["confidence_level"]
        
        for i, comp in enumerate(comparisons):
            comp.corrected_p_value = corrected_results[i]
            comp.is_significant = corrected_results[i] < alpha
    
    def _holm_correction(self, p_values: List[float]) -> List[float]:
        """Apply Holm step-down correction."""
        n = len(p_values)
        if n <= 1:
            return p_values
        
        # Sort p-values with their original indices
        sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
        
        # Apply Holm correction
        corrected = [0.0] * n
        for i, (original_idx, p_val) in enumerate(sorted_pairs):
            corrected_p = min(p_val * (n - i), 1.0)
            corrected[original_idx] = corrected_p
        
        return corrected
    
    def save_results(
        self, 
        results: Dict[str, List[EvaluationResult]],
        comparisons: List[StatisticalComparison],
        output_path: Path
    ):
        """Save evaluation results to disk."""
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual results
        results_data = {}
        for key, result_list in results.items():
            results_data[key] = [r.to_dict() for r in result_list]
        
        with open(output_path / "evaluation_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save statistical comparisons
        comparisons_data = [comp.to_dict() for comp in comparisons]
        with open(output_path / "statistical_comparisons.json", 'w') as f:
            json.dump(comparisons_data, f, indent=2)
        
        # Save summary statistics
        summary = self._generate_summary(results, comparisons)
        with open(output_path / "evaluation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")
    
    def _generate_summary(
        self,
        results: Dict[str, List[EvaluationResult]],
        comparisons: List[StatisticalComparison]
    ) -> Dict[str, Any]:
        """Generate evaluation summary."""
        
        summary = {
            "total_evaluations": sum(len(result_list) for result_list in results.values()),
            "total_comparisons": len(comparisons),
            "significant_improvements": len([c for c in comparisons if c.is_significant and c.effect_size > 0]),
            "competitors_evaluated": len(set(key.split('_')[0] for key in results.keys())),
            "datasets_evaluated": len(set('_'.join(key.split('_')[1:]) for key in results.keys())),
        }
        
        # Add per-competitor summary
        competitor_summaries = {}
        for key, result_list in results.items():
            competitor_name = key.split('_')[0]
            if competitor_name not in competitor_summaries:
                competitor_summaries[competitor_name] = {
                    "total_evaluations": 0,
                    "mean_success_rate": 0.0,
                    "mean_latency_ms": 0.0,
                    "mean_precision": 0.0
                }
            
            comp_summary = competitor_summaries[competitor_name]
            comp_summary["total_evaluations"] += len(result_list)
            
            # Aggregate metrics
            success_rates = [r.success_rate for r in result_list]
            latencies = [r.mean_latency_ms for r in result_list]
            precisions = [r.precision_at_k for r in result_list]
            
            comp_summary["mean_success_rate"] = np.mean(success_rates)
            comp_summary["mean_latency_ms"] = np.mean(latencies)
            comp_summary["mean_precision"] = np.mean(precisions)
        
        summary["competitor_summaries"] = competitor_summaries
        
        return summary