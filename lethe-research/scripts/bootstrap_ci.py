#!/usr/bin/env python3
"""
BCa Bootstrap Confidence Intervals for Lethe Statistical Analysis
================================================================

Implements bias-corrected and accelerated (BCa) bootstrap confidence intervals
with 10k bootstrap samples and FDR correction for multiple comparisons.

Key Features:
- BCa correction for skewed distributions
- 10,000 bootstrap samples for high precision
- FDR control within metric families
- Primary metrics: nDCG@10, Recall@20, MRR@10
- Effect size confidence intervals
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import norm
import multiprocessing as mp
from functools import partial


class BCaBootstrapAnalyzer:
    """BCa Bootstrap confidence interval analyzer with FDR correction"""
    
    def __init__(self, n_bootstrap: int = 10000, confidence_level: float = 0.95, 
                 fdr_q: float = 0.05, n_jobs: int = -1):
        """
        Initialize BCa Bootstrap analyzer
        
        Args:
            n_bootstrap: Number of bootstrap samples (default 10,000)
            confidence_level: Confidence level for intervals (default 0.95)  
            fdr_q: FDR control level (default 0.05)
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.fdr_q = fdr_q
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        
        # Primary metrics for Task 5
        self.primary_metrics = ['ndcg_at_10', 'recall_at_20', 'mrr_at_10']
        
        # Metric families for FDR correction
        self.metric_families = {
            'retrieval_quality': ['ndcg_at_10', 'recall_at_20', 'recall_at_50', 'mrr_at_10'],
            'coverage': ['coverage_at_n'],
            'latency': ['latency_ms_total', 'latency_p95', 'latency_p50'],
            'quality_issues': ['contradiction_rate', 'hallucination_rate']
        }
        
        print(f"Initialized BCa Bootstrap Analyzer")
        print(f"Bootstrap samples: {self.n_bootstrap:,}")
        print(f"Confidence level: {self.confidence_level}")
        print(f"FDR q-value: {self.fdr_q}")
        print(f"Parallel jobs: {self.n_jobs}")
    
    def _bootstrap_statistic(self, data: np.ndarray, indices: np.ndarray, 
                           statistic_func=np.mean) -> float:
        """Compute statistic for bootstrap sample"""
        bootstrap_sample = data[indices]
        return statistic_func(bootstrap_sample)
    
    def _jackknife_estimates(self, data: np.ndarray, statistic_func=np.mean) -> np.ndarray:
        """Compute jackknife estimates for acceleration parameter"""
        n = len(data)
        jackknife_estimates = np.zeros(n)
        
        for i in range(n):
            # Leave-one-out sample
            jackknife_sample = np.concatenate([data[:i], data[i+1:]])
            jackknife_estimates[i] = statistic_func(jackknife_sample)
        
        return jackknife_estimates
    
    def bca_confidence_interval(self, data: np.ndarray, statistic_func=np.mean,
                               seed: Optional[int] = None) -> Dict[str, float]:
        """
        Compute BCa bootstrap confidence interval
        
        Args:
            data: Input data array
            statistic_func: Statistic function to compute (default: mean)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with CI bounds, bias correction, and acceleration
        """
        if seed is not None:
            np.random.seed(seed)
        
        n = len(data)
        
        # Original statistic
        original_stat = statistic_func(data)
        
        # Bootstrap samples
        bootstrap_indices = np.random.randint(0, n, (self.n_bootstrap, n))
        
        # Compute bootstrap statistics in parallel
        bootstrap_func = partial(self._bootstrap_statistic, data, 
                               statistic_func=statistic_func)
        
        if self.n_jobs == 1:
            bootstrap_stats = [bootstrap_func(indices) for indices in bootstrap_indices]
        else:
            with mp.Pool(processes=self.n_jobs) as pool:
                bootstrap_stats = pool.map(bootstrap_func, bootstrap_indices)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Bias correction parameter
        # Proportion of bootstrap statistics less than original
        bias_correction = norm.ppf((bootstrap_stats < original_stat).mean())
        
        # Acceleration parameter using jackknife
        jackknife_stats = self._jackknife_estimates(data, statistic_func)
        jackknife_mean = jackknife_stats.mean()
        
        # Acceleration formula
        numerator = ((jackknife_mean - jackknife_stats) ** 3).sum()
        denominator = 6 * (((jackknife_mean - jackknife_stats) ** 2).sum() ** 1.5)
        
        acceleration = numerator / denominator if denominator != 0 else 0
        
        # Adjusted alpha levels for BCa
        z_alpha_2 = norm.ppf(self.alpha / 2)
        z_1_alpha_2 = norm.ppf(1 - self.alpha / 2)
        
        alpha_1 = norm.cdf(bias_correction + 
                          (bias_correction + z_alpha_2) / 
                          (1 - acceleration * (bias_correction + z_alpha_2)))
        alpha_2 = norm.cdf(bias_correction + 
                          (bias_correction + z_1_alpha_2) / 
                          (1 - acceleration * (bias_correction + z_1_alpha_2)))
        
        # Ensure alpha values are in valid range
        alpha_1 = np.clip(alpha_1, 0.001, 0.999)
        alpha_2 = np.clip(alpha_2, 0.001, 0.999)
        
        # Get percentiles from bootstrap distribution
        lower_percentile = 100 * alpha_1
        upper_percentile = 100 * alpha_2
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return {
            'statistic': float(original_stat),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'bootstrap_mean': float(bootstrap_stats.mean()),
            'bootstrap_std': float(bootstrap_stats.std()),
            'bias_correction': float(bias_correction),
            'acceleration': float(acceleration),
            'alpha_1': float(alpha_1),
            'alpha_2': float(alpha_2),
            'n_bootstrap': self.n_bootstrap,
            'confidence_level': self.confidence_level
        }
    
    def benjamini_hochberg_fdr(self, p_values: np.ndarray, 
                              alpha: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Benjamini-Hochberg FDR correction
        
        Args:
            p_values: Array of p-values
            alpha: FDR level (defaults to self.fdr_q)
            
        Returns:
            Tuple of (rejected nulls, adjusted p-values)
        """
        if alpha is None:
            alpha = self.fdr_q
        
        p_values = np.asarray(p_values)
        n = len(p_values)
        
        # Sort p-values and track original indices
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]
        
        # BH procedure
        rejected = np.zeros(n, dtype=bool)
        
        # Find largest k such that p_k <= k/n * alpha
        for i in range(n - 1, -1, -1):
            if sorted_p_values[i] <= ((i + 1) / n) * alpha:
                # Reject all hypotheses up to and including k
                rejected[sorted_indices[:i + 1]] = True
                break
        
        # Adjusted p-values (step-up method)
        adjusted_p_values = np.zeros(n)
        adjusted_p_values[sorted_indices] = np.minimum.accumulate(
            sorted_p_values[::-1] * n / np.arange(n, 0, -1)
        )[::-1]
        
        # Ensure adjusted p-values don't exceed 1
        adjusted_p_values = np.minimum(adjusted_p_values, 1.0)
        
        # Restore original order
        final_rejected = rejected.copy()
        final_adjusted = adjusted_p_values.copy()
        
        return final_rejected, final_adjusted
    
    def compute_effect_size_ci(self, group1: np.ndarray, group2: np.ndarray,
                              seed: Optional[int] = None) -> Dict[str, float]:
        """
        Compute Cohen's d effect size with BCa confidence interval
        
        Args:
            group1: First group data
            group2: Second group data
            seed: Random seed
            
        Returns:
            Effect size statistics with CI
        """
        
        def cohens_d_func(combined_data):
            """Cohen's d calculation from combined bootstrap sample"""
            n1 = len(group1)
            bootstrap_group1 = combined_data[:n1]
            bootstrap_group2 = combined_data[n1:]
            
            mean1, mean2 = bootstrap_group1.mean(), bootstrap_group2.mean()
            
            # Pooled standard deviation
            var1, var2 = bootstrap_group1.var(ddof=1), bootstrap_group2.var(ddof=1)
            pooled_std = np.sqrt(((len(bootstrap_group1) - 1) * var1 + 
                                 (len(bootstrap_group2) - 1) * var2) / 
                                (len(bootstrap_group1) + len(bootstrap_group2) - 2))
            
            return (mean2 - mean1) / pooled_std if pooled_std > 0 else 0
        
        # Combine data for bootstrap sampling
        combined_data = np.concatenate([group1, group2])
        
        # Get BCa CI for Cohen's d
        ci_result = self.bca_confidence_interval(combined_data, cohens_d_func, seed=seed)
        
        # Interpret effect size magnitude
        d_magnitude = abs(ci_result['statistic'])
        if d_magnitude < 0.2:
            magnitude = 'negligible'
        elif d_magnitude < 0.5:
            magnitude = 'small'
        elif d_magnitude < 0.8:
            magnitude = 'medium'
        else:
            magnitude = 'large'
        
        return {
            **ci_result,
            'cohens_d': ci_result['statistic'],
            'magnitude': magnitude,
            'd_magnitude': d_magnitude
        }
    
    def paired_bootstrap_test(self, group1: np.ndarray, group2: np.ndarray,
                             seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Paired bootstrap significance test with effect size
        
        Args:
            group1: First group (baseline)
            group2: Second group (treatment)
            seed: Random seed
            
        Returns:
            Test results with p-value and effect size CI
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Ensure equal lengths for paired test
        min_len = min(len(group1), len(group2))
        group1_paired = group1[:min_len]
        group2_paired = group2[:min_len]
        
        differences = group2_paired - group1_paired
        observed_mean_diff = differences.mean()
        
        # Bootstrap test for difference
        diff_ci = self.bca_confidence_interval(differences, np.mean, seed=seed)
        
        # Effect size with CI
        effect_size_ci = self.compute_effect_size_ci(group1_paired, group2_paired, seed=seed)
        
        # Two-sided p-value: proportion of bootstrap samples with opposite sign
        bootstrap_diffs = np.random.choice(differences, 
                                         (self.n_bootstrap, len(differences)), 
                                         replace=True).mean(axis=1)
        
        # Two-sided test
        if observed_mean_diff >= 0:
            p_value = 2 * (bootstrap_diffs <= -abs(observed_mean_diff)).mean()
        else:
            p_value = 2 * (bootstrap_diffs >= abs(observed_mean_diff)).mean()
        
        p_value = max(p_value, 1.0 / self.n_bootstrap)  # Avoid p=0
        
        # Statistical significance
        significant = diff_ci['ci_lower'] > 0 or diff_ci['ci_upper'] < 0
        
        return {
            'observed_difference': float(observed_mean_diff),
            'difference_ci': diff_ci,
            'effect_size_ci': effect_size_ci,
            'p_value': float(p_value),
            'significant': significant,
            'mean1': float(group1_paired.mean()),
            'mean2': float(group2_paired.mean()),
            'n1': len(group1_paired),
            'n2': len(group2_paired),
            'test_type': 'paired_bootstrap'
        }
    
    def analyze_methods_with_fdr(self, data: pd.DataFrame, 
                                baseline_method: str = 'baseline_bm25_vector_simple',
                                seed: int = 42) -> Dict[str, Any]:
        """
        Comprehensive method comparison with FDR correction
        
        Args:
            data: Experimental data
            baseline_method: Reference baseline method
            seed: Random seed for reproducibility
            
        Returns:
            Complete analysis results with FDR correction
        """
        print("Computing BCa confidence intervals with FDR correction...")
        
        methods = [m for m in data['method'].unique() 
                  if m != baseline_method and not m.startswith('baseline_')]
        
        results = {
            'baseline_method': baseline_method,
            'comparison_methods': methods,
            'fdr_q': self.fdr_q,
            'n_bootstrap': self.n_bootstrap,
            'confidence_level': self.confidence_level,
            'metric_families': self.metric_families,
            'method_comparisons': {},
            'fdr_corrections': {},
            'summary': {}
        }
        
        # Get baseline data
        baseline_data = data[data['method'] == baseline_method]
        
        for family_name, family_metrics in self.metric_families.items():
            print(f"Processing {family_name} metric family...")
            
            # Collect all p-values for this family
            family_p_values = []
            family_comparisons = []
            
            results['method_comparisons'][family_name] = {}
            
            for method in methods:
                method_data = data[data['method'] == method]
                
                if len(method_data) == 0:
                    continue
                
                results['method_comparisons'][family_name][method] = {}
                
                for metric in family_metrics:
                    if metric not in baseline_data.columns or metric not in method_data.columns:
                        continue
                    
                    baseline_values = baseline_data[metric].dropna().values
                    method_values = method_data[metric].dropna().values
                    
                    if len(baseline_values) < 5 or len(method_values) < 5:
                        continue
                    
                    # Paired bootstrap test
                    test_result = self.paired_bootstrap_test(
                        baseline_values, method_values, seed=seed + hash(method + metric) % 1000
                    )
                    
                    results['method_comparisons'][family_name][method][metric] = test_result
                    
                    # Store p-value for FDR correction
                    family_p_values.append(test_result['p_value'])
                    family_comparisons.append((method, metric))
            
            # Apply FDR correction within this family
            if family_p_values:
                family_p_array = np.array(family_p_values)
                rejected, adjusted_p = self.benjamini_hochberg_fdr(family_p_array, self.fdr_q)
                
                results['fdr_corrections'][family_name] = {
                    'original_p_values': family_p_values,
                    'adjusted_p_values': adjusted_p.tolist(),
                    'rejected_nulls': rejected.tolist(),
                    'significant_discoveries': int(rejected.sum()),
                    'total_tests': len(family_p_values),
                    'fdr_q': self.fdr_q
                }
                
                # Update significance based on FDR correction
                for i, (method, metric) in enumerate(family_comparisons):
                    if method in results['method_comparisons'][family_name]:
                        if metric in results['method_comparisons'][family_name][method]:
                            comparison = results['method_comparisons'][family_name][method][metric]
                            comparison['fdr_significant'] = rejected[i]
                            comparison['adjusted_p_value'] = float(adjusted_p[i])
        
        # Generate summary statistics
        results['summary'] = self._generate_summary_stats(results)
        
        return results
    
    def _generate_summary_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for the analysis"""
        total_comparisons = 0
        significant_comparisons = 0
        fdr_significant_comparisons = 0
        
        for family_name, family_data in results['method_comparisons'].items():
            for method, method_data in family_data.items():
                for metric, comparison in method_data.items():
                    total_comparisons += 1
                    if comparison.get('significant', False):
                        significant_comparisons += 1
                    if comparison.get('fdr_significant', False):
                        fdr_significant_comparisons += 1
        
        # Effect size distribution
        effect_sizes = []
        for family_name, family_data in results['method_comparisons'].items():
            for method, method_data in family_data.items():
                for metric, comparison in method_data.items():
                    if 'effect_size_ci' in comparison:
                        effect_sizes.append(comparison['effect_size_ci']['cohens_d'])
        
        return {
            'total_comparisons': total_comparisons,
            'significant_uncorrected': significant_comparisons,
            'significant_fdr_corrected': fdr_significant_comparisons,
            'fdr_reduction_ratio': 1 - (fdr_significant_comparisons / significant_comparisons) 
                                 if significant_comparisons > 0 else 0,
            'effect_size_distribution': {
                'mean': float(np.mean(effect_sizes)) if effect_sizes else 0,
                'median': float(np.median(effect_sizes)) if effect_sizes else 0,
                'std': float(np.std(effect_sizes)) if effect_sizes else 0,
                'count': len(effect_sizes)
            }
        }
    
    def save_results(self, results: Dict[str, Any], output_file: Path) -> None:
        """Save analysis results to JSON file"""
        
        # Add metadata
        results['analysis_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'analyzer': 'BCaBootstrapAnalyzer',
            'version': '1.0.0',
            'n_bootstrap': self.n_bootstrap,
            'confidence_level': self.confidence_level,
            'fdr_q': self.fdr_q,
            'n_jobs': self.n_jobs
        }
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"BCa Bootstrap results saved to: {output_file}")


def load_experimental_data(artifacts_dir: Path) -> pd.DataFrame:
    """Load experimental data from artifacts"""
    
    # Try multiple data sources
    data_sources = [
        artifacts_dir / "synthetic_dataset.json",
        artifacts_dir / "statistical_analysis_results.json",
        artifacts_dir / "publication_statistical_results.json"
    ]
    
    for source in data_sources:
        if source.exists():
            print(f"Loading data from {source}")
            with open(source, 'r') as f:
                data = json.load(f)
            
            if 'datapoints' in data:
                return pd.DataFrame(data['datapoints'])
            elif 'summary_statistics' in data:
                # Reconstruct from summary statistics (as in publication analyzer)
                return reconstruct_from_summary_stats(data)
    
    raise FileNotFoundError("No experimental data found in artifacts directory")


def reconstruct_from_summary_stats(data: Dict[str, Any]) -> pd.DataFrame:
    """Reconstruct dataframe from summary statistics for bootstrap analysis"""
    
    summary_stats = data.get('summary_statistics', {})
    all_data = []
    
    for method, method_stats in summary_stats.items():
        for metric, metric_stats in method_stats.items():
            if 'mean' in metric_stats and 'std' in metric_stats and 'count' in metric_stats:
                mean = metric_stats['mean']
                std = metric_stats['std']
                count = metric_stats['count']
                
                # Generate synthetic samples matching the statistics
                np.random.seed(42)  # Reproducible
                synthetic_values = np.random.normal(mean, std, count)
                
                # Apply realistic bounds
                if metric in ['ndcg_at_10', 'recall_at_20', 'recall_at_50', 'mrr_at_10', 'coverage_at_n']:
                    synthetic_values = np.clip(synthetic_values, 0, 1)
                elif metric == 'latency_ms_total':
                    synthetic_values = np.maximum(synthetic_values, 0)
                elif metric in ['contradiction_rate', 'hallucination_rate']:
                    synthetic_values = np.clip(synthetic_values, 0, 1)
                
                # Create dataframe rows
                for i, value in enumerate(synthetic_values):
                    row = {
                        'method': method,
                        'query_id': f'{method}_query_{i}',
                        'iteration': 0 if method.startswith('baseline') else int(method[-1]),
                        'domain': 'mixed',
                        metric: value
                    }
                    all_data.append(row)
    
    return pd.DataFrame(all_data)


def main():
    """Main execution function"""
    artifacts_dir = Path("artifacts")
    output_dir = Path("analysis")
    
    # Initialize analyzer
    analyzer = BCaBootstrapAnalyzer(
        n_bootstrap=10000,
        confidence_level=0.95,
        fdr_q=0.05,
        n_jobs=-1
    )
    
    # Load data
    print("Loading experimental data...")
    df = load_experimental_data(artifacts_dir)
    
    print(f"Loaded {len(df)} data points")
    print(f"Methods: {sorted(df['method'].unique())}")
    print(f"Metrics: {[col for col in df.columns if col not in ['method', 'query_id', 'iteration', 'domain']]}")
    
    # Run BCa bootstrap analysis with FDR correction
    results = analyzer.analyze_methods_with_fdr(df)
    
    # Save results
    output_file = output_dir / "bca_bootstrap_results.json"
    analyzer.save_results(results, output_file)
    
    # Print summary
    print("\n" + "="*60)
    print("BCa BOOTSTRAP ANALYSIS SUMMARY")
    print("="*60)
    print(f"Bootstrap samples: {analyzer.n_bootstrap:,}")
    print(f"Confidence level: {analyzer.confidence_level}")
    print(f"FDR q-value: {analyzer.fdr_q}")
    print(f"Total comparisons: {results['summary']['total_comparisons']}")
    print(f"Significant (uncorrected): {results['summary']['significant_uncorrected']}")
    print(f"Significant (FDR corrected): {results['summary']['significant_fdr_corrected']}")
    print(f"FDR reduction: {results['summary']['fdr_reduction_ratio']:.1%}")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()