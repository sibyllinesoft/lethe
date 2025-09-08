#!/usr/bin/env python3
"""
Task 3 Metrics and Statistical Analysis
======================================

Comprehensive metrics computation and statistical analysis for baseline evaluation.

Features:
- Standard IR metrics: nDCG@{5,10}, Recall@{10,20}, MRR@10, MAP
- Statistical significance testing (paired t-tests, Wilcoxon)
- Effect size computation (Cohen's d)
- Per-query analysis and outlier detection
- Confidence intervals and bootstrap sampling
"""

import numpy as np
import scipy.stats as stats
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class StatisticalTest:
    """Result of statistical significance test"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str

class StatisticalAnalyzer:
    """Statistical analysis for baseline comparison"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        
    def paired_t_test(self, 
                     baseline_scores: List[float],
                     comparison_scores: List[float],
                     metric_name: str) -> StatisticalTest:
        """Perform paired t-test between two sets of scores"""
        
        if len(baseline_scores) != len(comparison_scores):
            raise ValueError("Score lists must have same length")
            
        if len(baseline_scores) < 3:
            raise ValueError("Need at least 3 paired observations")
            
        # Compute differences
        differences = np.array(comparison_scores) - np.array(baseline_scores)
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(comparison_scores, baseline_scores)
        
        # Effect size (Cohen's d for paired samples)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0
        
        # Confidence interval for mean difference
        n = len(differences)
        se_diff = std_diff / np.sqrt(n)
        t_critical = stats.t.ppf(1 - self.alpha/2, n - 1)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff
        
        # Interpretation
        is_significant = p_value < self.alpha
        if is_significant:
            direction = "better" if mean_diff > 0 else "worse"
            interpretation = f"Significant {direction} performance (p={p_value:.3f}, d={cohens_d:.3f})"
        else:
            interpretation = f"No significant difference (p={p_value:.3f})"
            
        return StatisticalTest(
            test_name="Paired t-test",
            statistic=t_stat,
            p_value=p_value,
            significant=is_significant,
            effect_size=cohens_d,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )
        
    def wilcoxon_test(self,
                     baseline_scores: List[float],
                     comparison_scores: List[float],
                     metric_name: str) -> StatisticalTest:
        """Perform Wilcoxon signed-rank test (non-parametric)"""
        
        if len(baseline_scores) != len(comparison_scores):
            raise ValueError("Score lists must have same length")
            
        try:
            # Wilcoxon signed-rank test
            w_stat, p_value = stats.wilcoxon(comparison_scores, baseline_scores)
            
            # Effect size (r = Z / sqrt(N))
            n = len(baseline_scores)
            z_score = stats.norm.ppf(1 - p_value/2) if p_value > 0 else 0
            effect_r = abs(z_score) / np.sqrt(n)
            
            # Median difference (robust estimate)
            differences = np.array(comparison_scores) - np.array(baseline_scores)
            median_diff = np.median(differences)
            
            # Bootstrap CI for median difference
            ci_lower, ci_upper = self._bootstrap_ci(differences, np.median)
            
            # Interpretation
            is_significant = p_value < self.alpha
            if is_significant:
                direction = "better" if median_diff > 0 else "worse"
                interpretation = f"Significant {direction} performance (p={p_value:.3f}, r={effect_r:.3f})"
            else:
                interpretation = f"No significant difference (p={p_value:.3f})"
                
            return StatisticalTest(
                test_name="Wilcoxon signed-rank",
                statistic=w_stat,
                p_value=p_value,
                significant=is_significant,
                effect_size=effect_r,
                confidence_interval=(ci_lower, ci_upper),
                interpretation=interpretation
            )
            
        except ValueError as e:
            logger.warning(f"Wilcoxon test failed: {e}")
            # Return non-significant result
            return StatisticalTest(
                test_name="Wilcoxon signed-rank", 
                statistic=0.0,
                p_value=1.0,
                significant=False,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                interpretation="Test failed - insufficient data"
            )
    
    def _bootstrap_ci(self, 
                     data: np.ndarray, 
                     statistic_func: callable,
                     n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Compute bootstrap confidence interval"""
        bootstrap_stats = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic_func(bootstrap_sample))
            
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return ci_lower, ci_upper
        
    def compare_all_baselines(self,
                            metrics_by_baseline: Dict[str, List[float]],
                            metric_name: str,
                            baseline_name: str = None) -> Dict[str, Dict[str, StatisticalTest]]:
        """Compare all baselines against a reference baseline"""
        
        if baseline_name is None:
            baseline_name = list(metrics_by_baseline.keys())[0]
            
        if baseline_name not in metrics_by_baseline:
            raise ValueError(f"Baseline '{baseline_name}' not found")
            
        baseline_scores = metrics_by_baseline[baseline_name]
        results = {}
        
        for method_name, method_scores in metrics_by_baseline.items():
            if method_name == baseline_name:
                continue
                
            if len(method_scores) != len(baseline_scores):
                logger.warning(f"Score length mismatch for {method_name}, skipping")
                continue
                
            # Run both parametric and non-parametric tests
            tests = {}
            
            try:
                tests['t_test'] = self.paired_t_test(
                    baseline_scores, method_scores, metric_name)
            except Exception as e:
                logger.warning(f"t-test failed for {method_name}: {e}")
                
            try:
                tests['wilcoxon'] = self.wilcoxon_test(
                    baseline_scores, method_scores, metric_name)
            except Exception as e:
                logger.warning(f"Wilcoxon test failed for {method_name}: {e}")
                
            results[method_name] = tests
            
        return results

class MetricsAggregator:
    """Aggregate and analyze metrics across queries and baselines"""
    
    @staticmethod
    def aggregate_by_baseline(metrics_results: List[Any]) -> Dict[str, Dict[str, Any]]:
        """Aggregate metrics by baseline method"""
        
        from collections import defaultdict
        
        by_baseline = defaultdict(list)
        
        # Group by baseline
        for result in metrics_results:
            by_baseline[result.baseline_name].append(result)
            
        # Compute aggregates
        aggregated = {}
        
        for baseline_name, baseline_results in by_baseline.items():
            
            # Extract metric values
            ndcg_10 = [r.ndcg_10 for r in baseline_results]
            ndcg_5 = [r.ndcg_5 for r in baseline_results]
            recall_10 = [r.recall_10 for r in baseline_results]
            recall_20 = [r.recall_20 for r in baseline_results]
            mrr_10 = [r.mrr_10 for r in baseline_results]
            latency_ms = [r.latency_ms for r in baseline_results]
            
            aggregated[baseline_name] = {
                'count': len(baseline_results),
                
                # Effectiveness metrics (mean ± std)
                'ndcg_10': {
                    'mean': np.mean(ndcg_10),
                    'std': np.std(ndcg_10),
                    'median': np.median(ndcg_10),
                    'values': ndcg_10
                },
                'ndcg_5': {
                    'mean': np.mean(ndcg_5),
                    'std': np.std(ndcg_5),
                    'median': np.median(ndcg_5),
                    'values': ndcg_5
                },
                'recall_10': {
                    'mean': np.mean(recall_10),
                    'std': np.std(recall_10),
                    'median': np.median(recall_10),
                    'values': recall_10
                },
                'recall_20': {
                    'mean': np.mean(recall_20),
                    'std': np.std(recall_20),
                    'median': np.median(recall_20),
                    'values': recall_20
                },
                'mrr_10': {
                    'mean': np.mean(mrr_10),
                    'std': np.std(mrr_10),
                    'median': np.median(mrr_10),
                    'values': mrr_10
                },
                
                # Efficiency metrics
                'latency_ms': {
                    'mean': np.mean(latency_ms),
                    'std': np.std(latency_ms),
                    'p50': np.percentile(latency_ms, 50),
                    'p95': np.percentile(latency_ms, 95),
                    'p99': np.percentile(latency_ms, 99),
                    'values': latency_ms
                }
            }
            
        return aggregated
        
    @staticmethod
    def identify_outliers(values: List[float], method: str = 'iqr') -> List[int]:
        """Identify outlier indices using specified method"""
        
        values = np.array(values)
        
        if method == 'iqr':
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = np.where((values < lower_bound) | (values > upper_bound))[0]
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(values))
            outliers = np.where(z_scores > 3)[0]  # 3-sigma rule
            
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
            
        return outliers.tolist()
        
    @staticmethod
    def compute_confidence_intervals(values: List[float], 
                                   confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval for mean"""
        
        n = len(values)
        if n < 2:
            return 0.0, 0.0
            
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        se = std / np.sqrt(n)
        
        alpha = 1 - confidence
        t_critical = stats.t.ppf(1 - alpha/2, n - 1)
        
        margin = t_critical * se
        
        return mean - margin, mean + margin

class PerformanceAnalyzer:
    """Analyze performance characteristics and trade-offs"""
    
    @staticmethod
    def compute_efficiency_frontier(aggregated_metrics: Dict[str, Dict[str, Any]],
                                  effectiveness_metric: str = 'ndcg_10',
                                  efficiency_metric: str = 'latency_ms') -> List[Dict[str, Any]]:
        """Compute Pareto frontier for effectiveness vs efficiency"""
        
        points = []
        
        for baseline_name, metrics in aggregated_metrics.items():
            effectiveness = metrics[effectiveness_metric]['mean']
            efficiency = metrics[efficiency_metric]['mean']  # Lower is better for latency
            
            points.append({
                'baseline': baseline_name,
                'effectiveness': effectiveness,
                'efficiency': efficiency,
                'efficiency_inverse': 1.0 / efficiency if efficiency > 0 else 0.0  # Higher is better
            })
            
        # Sort by effectiveness (descending) then efficiency (ascending for latency)
        points.sort(key=lambda x: (-x['effectiveness'], x['efficiency']))
        
        # Find Pareto frontier
        frontier = []
        current_best_efficiency = float('inf')
        
        for point in points:
            if point['efficiency'] < current_best_efficiency:
                frontier.append(point)
                current_best_efficiency = point['efficiency']
                
        return frontier
        
    @staticmethod
    def analyze_scalability(metrics_by_query_length: Dict[int, Dict[str, List[float]]]) -> Dict[str, Any]:
        """Analyze how performance scales with query complexity"""
        
        scalability_analysis = {}
        
        for baseline_name in next(iter(metrics_by_query_length.values())).keys():
            query_lengths = sorted(metrics_by_query_length.keys())
            latencies = []
            
            for length in query_lengths:
                if baseline_name in metrics_by_query_length[length]:
                    mean_latency = np.mean(metrics_by_query_length[length][baseline_name])
                    latencies.append(mean_latency)
                else:
                    latencies.append(np.nan)
                    
            # Compute correlation between query length and latency
            valid_pairs = [(l, lat) for l, lat in zip(query_lengths, latencies) if not np.isnan(lat)]
            
            if len(valid_pairs) >= 3:
                lengths, lats = zip(*valid_pairs)
                correlation, p_value = stats.pearsonr(lengths, lats)
                
                scalability_analysis[baseline_name] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'query_length_range': [min(lengths), max(lengths)],
                    'latency_range': [min(lats), max(lats)],
                    'scaling_interpretation': (
                        'Poor scaling' if correlation > 0.7 else
                        'Moderate scaling' if correlation > 0.3 else
                        'Good scaling'
                    )
                }
            else:
                scalability_analysis[baseline_name] = {
                    'correlation': 0.0,
                    'p_value': 1.0,
                    'scaling_interpretation': 'Insufficient data'
                }
                
        return scalability_analysis

def generate_statistical_report(metrics_results: List[Any],
                              baseline_name: str = "BM25") -> Dict[str, Any]:
    """Generate comprehensive statistical analysis report"""
    
    # Aggregate metrics
    aggregated = MetricsAggregator.aggregate_by_baseline(metrics_results)
    
    # Statistical comparisons
    analyzer = StatisticalAnalyzer()
    
    # Extract metric values by baseline for comparison
    metrics_by_baseline = {}
    for name, data in aggregated.items():
        metrics_by_baseline[name] = {
            'ndcg_10': data['ndcg_10']['values'],
            'ndcg_5': data['ndcg_5']['values'], 
            'recall_10': data['recall_10']['values'],
            'mrr_10': data['mrr_10']['values']
        }
    
    # Run statistical tests
    statistical_tests = {}
    for metric in ['ndcg_10', 'ndcg_5', 'recall_10', 'mrr_10']:
        metric_values = {name: data[metric] for name, data in metrics_by_baseline.items()}
        statistical_tests[metric] = analyzer.compare_all_baselines(
            metric_values, metric, baseline_name)
    
    # Performance analysis
    efficiency_frontier = PerformanceAnalyzer.compute_efficiency_frontier(aggregated)
    
    # Outlier analysis
    outlier_analysis = {}
    for baseline_name, data in aggregated.items():
        outlier_analysis[baseline_name] = {
            'ndcg_10_outliers': MetricsAggregator.identify_outliers(data['ndcg_10']['values']),
            'latency_outliers': MetricsAggregator.identify_outliers(data['latency_ms']['values'])
        }
        
    return {
        'aggregated_metrics': aggregated,
        'statistical_tests': statistical_tests,
        'efficiency_frontier': efficiency_frontier,
        'outlier_analysis': outlier_analysis,
        'summary': {
            'baselines_compared': len(aggregated),
            'total_queries': sum(data['count'] for data in aggregated.values()),
            'significant_improvements': sum(
                1 for metric_tests in statistical_tests.values()
                for baseline_tests in metric_tests.values()
                for test in baseline_tests.values()
                if test.significant and test.effect_size > 0
            )
        }
    }