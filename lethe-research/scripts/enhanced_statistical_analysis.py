#!/usr/bin/env python3
"""
Lethe Research Enhanced Statistical Analysis
===========================================

Comprehensive publication-ready statistical validation for the Lethe benchmark.
Addresses peer review requirements with full pairwise comparisons, multiple
comparison corrections, confidence intervals, and reproducibility validation.

Key Features:
- Complete pairwise comparison matrix (all methods vs all methods)
- Bonferroni correction for multiple comparisons
- Bootstrap confidence intervals
- Effect size calculations with interpretation
- Comprehensive baseline validation
- Publication-ready tables and figures
- Reproducibility verification
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon, kruskal, bootstrap, ttest_ind
import itertools
from statsmodels.stats.multitest import multipletests

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

class EnhancedStatisticalAnalyzer:
    """Publication-ready comprehensive statistical analysis for Lethe research"""
    
    def __init__(self, artifacts_dir: str = "artifacts", output_dir: str = "paper"):
        self.artifacts_dir = Path(artifacts_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "tables").mkdir(exist_ok=True)
        (self.output_dir / "supplementary").mkdir(exist_ok=True)
        
        # Statistical parameters
        self.alpha = 0.05
        self.bonferroni_alpha = 0.05  # Will be adjusted based on number of comparisons
        self.bootstrap_n_samples = 10000
        self.confidence_level = 0.95
        
        # Method mapping for publication
        self.method_display_names = {
            'baseline_bm25_only': 'BM25-only',
            'baseline_vector_only': 'Vector-only', 
            'baseline_cross_encoder': 'Cross-encoder',
            'baseline_faiss_ivf': 'FAISS-IVF',
            'baseline_mmr': 'MMR',
            'baseline_window': 'Window-based',
            'baseline_bm25_vector_simple': 'BM25+Vector',
            'iter1': 'Lethe v1.0',
            'iter2': 'Lethe v2.0',
            'iter3': 'Lethe v3.0',
            'iter4': 'Lethe v4.0 (Final)'
        }
        
        print(f"Initialized Enhanced Statistical Analyzer")
        print(f"Artifacts: {self.artifacts_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Confidence Level: {self.confidence_level * 100}%")
    
    def load_comprehensive_data(self) -> pd.DataFrame:
        """Load all available experimental data for complete analysis"""
        print("Loading comprehensive experimental dataset...")
        
        # Load synthetic dataset if available
        synthetic_file = self.artifacts_dir / "synthetic_dataset.json"
        if synthetic_file.exists():
            with open(synthetic_file, 'r') as f:
                synthetic_data = json.load(f)
            
            df = pd.DataFrame(synthetic_data['datapoints'])
            print(f"Loaded {len(df)} datapoints from synthetic dataset")
        else:
            # Fallback to loading from individual files
            print("Synthetic dataset not found, loading from individual files...")
            # Use the existing loading logic from final_analysis.py
            from final_analysis import LetheResearchAnalyzer
            analyzer = LetheResearchAnalyzer(str(self.artifacts_dir), str(self.output_dir))
            df = analyzer.load_all_results()
        
        print(f"Dataset statistics:")
        print(f"  Total datapoints: {len(df)}")
        print(f"  Methods: {sorted(df['method'].unique())}")
        print(f"  Domains: {sorted(df['domain'].unique())}")
        print(f"  Synthetic ratio: {df['synthetic'].mean():.1%}" if 'synthetic' in df.columns else "  No synthetic flag")
        
        return df
    
    def validate_baseline_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate that all 7 baselines are properly represented"""
        print("Validating baseline completeness...")
        
        expected_baselines = [
            'baseline_bm25_only',
            'baseline_vector_only', 
            'baseline_cross_encoder',
            'baseline_faiss_ivf',
            'baseline_mmr',
            'baseline_window',
            'baseline_bm25_vector_simple'
        ]
        
        validation_results = {
            'expected_baselines': expected_baselines,
            'found_baselines': [],
            'missing_baselines': [],
            'baseline_counts': {},
            'baseline_metrics_coverage': {},
            'validation_passed': True,
            'issues': []
        }
        
        available_methods = df['method'].unique()
        
        for baseline in expected_baselines:
            baseline_data = df[df['method'] == baseline]
            count = len(baseline_data)
            validation_results['baseline_counts'][baseline] = count
            
            if count == 0:
                validation_results['missing_baselines'].append(baseline)
                validation_results['validation_passed'] = False
                validation_results['issues'].append(f"Missing baseline: {baseline}")
            else:
                validation_results['found_baselines'].append(baseline)
                
                # Check metrics coverage
                metrics_coverage = {}
                key_metrics = ['ndcg_at_10', 'recall_at_50', 'coverage_at_n', 'latency_ms_total']
                for metric in key_metrics:
                    if metric in baseline_data.columns:
                        non_null_count = baseline_data[metric].notna().sum()
                        coverage_pct = non_null_count / count
                        metrics_coverage[metric] = {
                            'coverage': coverage_pct,
                            'mean': baseline_data[metric].mean() if non_null_count > 0 else None
                        }
                        
                        if coverage_pct < 0.8:  # Less than 80% coverage
                            validation_results['issues'].append(
                                f"{baseline} has low {metric} coverage: {coverage_pct:.1%}"
                            )
                
                validation_results['baseline_metrics_coverage'][baseline] = metrics_coverage
        
        # Statistical power validation (minimum sample size)
        min_sample_size = 10
        for baseline in validation_results['found_baselines']:
            count = validation_results['baseline_counts'][baseline]
            if count < min_sample_size:
                validation_results['issues'].append(
                    f"{baseline} has insufficient sample size for statistical power: {count} < {min_sample_size}"
                )
        
        print(f"Baseline validation results:")
        print(f"  Found: {len(validation_results['found_baselines'])}/7 baselines")
        print(f"  Missing: {validation_results['missing_baselines']}")
        print(f"  Issues: {len(validation_results['issues'])}")
        
        if validation_results['issues']:
            for issue in validation_results['issues']:
                print(f"    - {issue}")
        
        return validation_results
    
    def compute_comprehensive_pairwise_comparisons(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute complete pairwise comparison matrix between all methods"""
        print("Computing comprehensive pairwise comparisons...")
        
        methods = sorted([m for m in df['method'].unique() if m in self.method_display_names])
        metrics = ['ndcg_at_10', 'recall_at_50', 'coverage_at_n', 'latency_ms_total', 
                  'contradiction_rate', 'hallucination_rate']
        
        n_methods = len(methods)
        n_comparisons_per_metric = (n_methods * (n_methods - 1)) // 2
        total_comparisons = n_comparisons_per_metric * len(metrics)
        
        # Bonferroni correction
        bonferroni_alpha = self.alpha / total_comparisons
        
        print(f"  Methods: {n_methods}")
        print(f"  Metrics: {len(metrics)}")
        print(f"  Total pairwise comparisons: {total_comparisons}")
        print(f"  Bonferroni-corrected α: {bonferroni_alpha:.6f}")
        
        results = {
            'comparison_matrix': {},
            'effect_sizes': {},
            'confidence_intervals': {},
            'summary_statistics': {},
            'bonferroni_alpha': bonferroni_alpha,
            'total_comparisons': total_comparisons,
            'methods': methods,
            'metrics': metrics
        }
        
        # Initialize matrices
        for metric in metrics:
            results['comparison_matrix'][metric] = {}
            results['effect_sizes'][metric] = {}
            results['confidence_intervals'][metric] = {}
        
        # Compute summary statistics
        for method in methods:
            method_data = df[df['method'] == method]
            results['summary_statistics'][method] = {}
            
            for metric in metrics:
                if metric in method_data.columns:
                    values = method_data[metric].dropna()
                    if len(values) > 0:
                        # Bootstrap confidence interval
                        ci_lower, ci_upper = self._bootstrap_confidence_interval(values)
                        
                        results['summary_statistics'][method][metric] = {
                            'n': len(values),
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'median': float(values.median()),
                            'ci_lower': float(ci_lower),
                            'ci_upper': float(ci_upper),
                            'min': float(values.min()),
                            'max': float(values.max())
                        }
        
        # Pairwise comparisons
        for metric in metrics:
            print(f"  Processing {metric}...")
            
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods):
                    if i >= j:  # Only compute upper triangular matrix
                        continue
                    
                    data1 = df[df['method'] == method1][metric].dropna()
                    data2 = df[df['method'] == method2][metric].dropna()
                    
                    if len(data1) < 3 or len(data2) < 3:  # Minimum sample size
                        continue
                    
                    comparison_key = f"{method1}_vs_{method2}"
                    
                    # Statistical test (Mann-Whitney U for non-parametric)
                    try:
                        statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                        
                        # Effect size (Cohen's d)
                        cohens_d = self._cohens_d(data1, data2)
                        
                        # Confidence interval for difference in means
                        diff_ci_lower, diff_ci_upper = self._difference_confidence_interval(data1, data2)
                        
                        # Store results
                        results['comparison_matrix'][metric][comparison_key] = {
                            'statistic': float(statistic),
                            'p_value': float(p_value),
                            'p_value_bonferroni': float(p_value * total_comparisons),
                            'significant': p_value < self.alpha,
                            'significant_bonferroni': p_value < bonferroni_alpha,
                            'method1': method1,
                            'method2': method2,
                            'mean1': float(data1.mean()),
                            'mean2': float(data2.mean()),
                            'difference': float(data2.mean() - data1.mean()),
                            'n1': len(data1),
                            'n2': len(data2)
                        }
                        
                        results['effect_sizes'][metric][comparison_key] = {
                            'cohens_d': float(cohens_d),
                            'magnitude': self._interpret_effect_size(abs(cohens_d)),
                            'direction': 'positive' if cohens_d > 0 else 'negative'
                        }
                        
                        results['confidence_intervals'][metric][comparison_key] = {
                            'diff_ci_lower': float(diff_ci_lower),
                            'diff_ci_upper': float(diff_ci_upper),
                            'ci_contains_zero': diff_ci_lower <= 0 <= diff_ci_upper
                        }
                        
                    except Exception as e:
                        print(f"    Error in comparison {comparison_key} for {metric}: {e}")
                        continue
        
        return results
    
    def apply_multiple_comparison_correction(self, pairwise_results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply multiple comparison corrections (Bonferroni and FDR)"""
        print("Applying multiple comparison corrections...")
        
        corrected_results = pairwise_results.copy()
        
        # Collect all p-values
        all_p_values = []
        p_value_keys = []
        
        for metric in pairwise_results['metrics']:
            for comparison_key, results in pairwise_results['comparison_matrix'][metric].items():
                all_p_values.append(results['p_value'])
                p_value_keys.append((metric, comparison_key))
        
        all_p_values = np.array(all_p_values)
        
        # Apply corrections
        bonferroni_corrected = multipletests(all_p_values, alpha=self.alpha, method='bonferroni')[1]
        fdr_corrected = multipletests(all_p_values, alpha=self.alpha, method='fdr_bh')[1]
        
        # Store corrected p-values
        for i, (metric, comparison_key) in enumerate(p_value_keys):
            corrected_results['comparison_matrix'][metric][comparison_key]['p_value_bonferroni_corrected'] = float(bonferroni_corrected[i])
            corrected_results['comparison_matrix'][metric][comparison_key]['p_value_fdr'] = float(fdr_corrected[i])
            corrected_results['comparison_matrix'][metric][comparison_key]['significant_bonferroni_corrected'] = bonferroni_corrected[i] < self.alpha
            corrected_results['comparison_matrix'][metric][comparison_key]['significant_fdr'] = fdr_corrected[i] < self.alpha
        
        # Summary statistics
        corrected_results['correction_summary'] = {
            'total_comparisons': len(all_p_values),
            'significant_uncorrected': int((all_p_values < self.alpha).sum()),
            'significant_bonferroni': int((bonferroni_corrected < self.alpha).sum()),
            'significant_fdr': int((fdr_corrected < self.alpha).sum()),
            'correction_impact': {
                'bonferroni_reduction': float(1 - (bonferroni_corrected < self.alpha).sum() / (all_p_values < self.alpha).sum()),
                'fdr_reduction': float(1 - (fdr_corrected < self.alpha).sum() / (all_p_values < self.alpha).sum())
            }
        }
        
        print(f"  Total comparisons: {len(all_p_values)}")
        print(f"  Significant (uncorrected): {(all_p_values < self.alpha).sum()}")
        print(f"  Significant (Bonferroni): {(bonferroni_corrected < self.alpha).sum()}")
        print(f"  Significant (FDR): {(fdr_corrected < self.alpha).sum()}")
        
        return corrected_results
    
    def generate_publication_ready_tables(self, statistical_results: Dict[str, Any], 
                                        validation_results: Dict[str, Any]) -> None:
        """Generate comprehensive publication-ready LaTeX tables"""
        print("Generating publication-ready tables...")
        
        tables_dir = self.output_dir / "tables"
        
        # Table 1: Method Performance Summary with Confidence Intervals
        self._generate_performance_summary_table(statistical_results, tables_dir / "enhanced_performance_summary.tex")
        
        # Table 2: Statistical Significance Matrix
        self._generate_significance_matrix_table(statistical_results, tables_dir / "statistical_significance_matrix.tex")
        
        # Table 3: Effect Sizes and Practical Significance
        self._generate_effect_sizes_table(statistical_results, tables_dir / "effect_sizes_analysis.tex")
        
        # Table 4: Multiple Comparison Corrections
        self._generate_multiple_corrections_table(statistical_results, tables_dir / "multiple_comparison_corrections.tex")
        
        # Table 5: Baseline Validation Summary
        self._generate_baseline_validation_table(validation_results, tables_dir / "baseline_validation.tex")
        
        # Supplementary Tables
        self._generate_supplementary_tables(statistical_results, tables_dir.parent / "supplementary")
    
    def generate_statistical_figures(self, df: pd.DataFrame, statistical_results: Dict[str, Any]) -> None:
        """Generate comprehensive statistical visualization figures"""
        print("Generating statistical figures...")
        
        figures_dir = self.output_dir / "figures"
        
        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("colorblind")
        
        # Figure 1: Performance Distribution Plots
        self._create_performance_distributions(df, figures_dir / "performance_distributions.pdf")
        
        # Figure 2: Statistical Significance Heatmap
        self._create_significance_heatmap(statistical_results, figures_dir / "significance_heatmap.pdf")
        
        # Figure 3: Effect Sizes Visualization
        self._create_effect_sizes_plot(statistical_results, figures_dir / "effect_sizes.pdf")
        
        # Figure 4: Confidence Intervals Plot
        self._create_confidence_intervals_plot(statistical_results, figures_dir / "confidence_intervals.pdf")
        
        # Figure 5: Multiple Comparison Impact
        self._create_multiple_comparison_impact(statistical_results, figures_dir / "multiple_comparison_impact.pdf")
    
    def compute_reproducibility_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute metrics to validate reproducibility of results"""
        print("Computing reproducibility validation metrics...")
        
        reproducibility_results = {
            'statistical_stability': {},
            'cross_validation_consistency': {},
            'bootstrap_stability': {},
            'sample_size_sensitivity': {}
        }
        
        methods = [m for m in df['method'].unique() if m in self.method_display_names]
        key_metrics = ['ndcg_at_10', 'recall_at_50', 'coverage_at_n']
        
        for method in methods:
            method_data = df[df['method'] == method]
            if len(method_data) < 10:  # Skip if insufficient data
                continue
            
            reproducibility_results['statistical_stability'][method] = {}
            
            for metric in key_metrics:
                if metric not in method_data.columns:
                    continue
                
                values = method_data[metric].dropna()
                if len(values) < 10:
                    continue
                
                # Bootstrap stability analysis
                bootstrap_means = []
                bootstrap_stds = []
                n_bootstrap = 1000
                
                for _ in range(n_bootstrap):
                    bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
                    bootstrap_means.append(np.mean(bootstrap_sample))
                    bootstrap_stds.append(np.std(bootstrap_sample))
                
                reproducibility_results['statistical_stability'][method][metric] = {
                    'original_mean': float(values.mean()),
                    'original_std': float(values.std()),
                    'bootstrap_mean_std': float(np.std(bootstrap_means)),
                    'bootstrap_std_std': float(np.std(bootstrap_stds)),
                    'stability_coefficient': float(np.std(bootstrap_means) / values.mean())
                }
        
        return reproducibility_results
    
    def _bootstrap_confidence_interval(self, data: np.ndarray, 
                                     confidence_level: float = None) -> Tuple[float, float]:
        """Compute bootstrap confidence interval for the mean"""
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        alpha = 1 - confidence_level
        
        # Bootstrap resampling
        bootstrap_means = []
        for _ in range(self.bootstrap_n_samples):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        # Calculate confidence interval
        ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return ci_lower, ci_upper
    
    def _difference_confidence_interval(self, data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
        """Compute confidence interval for difference in means"""
        # Bootstrap for difference in means
        differences = []
        for _ in range(self.bootstrap_n_samples):
            sample1 = np.random.choice(data1, size=len(data1), replace=True)
            sample2 = np.random.choice(data2, size=len(data2), replace=True)
            differences.append(np.mean(sample2) - np.mean(sample1))
        
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(differences, 100 * alpha / 2)
        ci_upper = np.percentile(differences, 100 * (1 - alpha / 2))
        
        return ci_lower, ci_upper
    
    def _cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * group1.var() + (n2 - 1) * group2.var()) / (n1 + n2 - 2))
        return (group2.mean() - group1.mean()) / pooled_std if pooled_std > 0 else 0.0
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size magnitude"""
        if d < 0.2:
            return 'negligible'
        elif d < 0.5:
            return 'small'
        elif d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def _generate_performance_summary_table(self, results: Dict[str, Any], output_file: Path):
        """Generate enhanced performance summary table with confidence intervals"""
        methods_order = ['baseline_bm25_only', 'baseline_vector_only', 'baseline_cross_encoder',
                        'baseline_faiss_ivf', 'baseline_mmr', 'baseline_window', 
                        'baseline_bm25_vector_simple', 'iter1', 'iter2', 'iter3', 'iter4']
        
        latex = r"""\begin{table*}[htbp]
\centering
\caption{Performance Summary by Method with 95\% Confidence Intervals}
\label{tab:enhanced-performance-summary}
\begin{tabular}{lccccc}
\toprule
Method & NDCG@10 & Recall@50 & Coverage@N & Latency (ms) & Contradiction Rate \\
\midrule
"""
        
        summary_stats = results['summary_statistics']
        
        for method in methods_order:
            if method not in summary_stats:
                continue
            
            display_name = self.method_display_names.get(method, method)
            
            # Format metrics with confidence intervals
            metrics_formatted = []
            for metric in ['ndcg_at_10', 'recall_at_50', 'coverage_at_n', 'latency_ms_total', 'contradiction_rate']:
                if metric in summary_stats[method]:
                    stats = summary_stats[method][metric]
                    mean = stats['mean']
                    ci_lower = stats['ci_lower']
                    ci_upper = stats['ci_upper']
                    
                    if metric == 'latency_ms_total':
                        formatted = f"{mean:.0f} [{ci_lower:.0f}, {ci_upper:.0f}]"
                    else:
                        formatted = f"{mean:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"
                    metrics_formatted.append(formatted)
                else:
                    metrics_formatted.append("N/A")
            
            latex += f"{display_name} & " + " & ".join(metrics_formatted) + " \\\\\n"
        
        latex += r"""\bottomrule
\multicolumn{6}{l}{\small Values shown as mean [95\% confidence interval]} \\
\end{tabular}
\end{table*}
"""
        
        with open(output_file, 'w') as f:
            f.write(latex)
        print(f"  Generated: {output_file.name}")
    
    def _generate_significance_matrix_table(self, results: Dict[str, Any], output_file: Path):
        """Generate statistical significance matrix table"""
        latex = r"""\begin{table*}[htbp]
\centering
\caption{Statistical Significance Matrix (NDCG@10) with Multiple Comparison Corrections}
\label{tab:significance-matrix}
\begin{tabular}{l""" + "c" * len(results['methods']) + r"""}
\toprule
Method"""
        
        # Header
        for method in results['methods']:
            display_name = self.method_display_names.get(method, method)
            latex += f" & {display_name}"
        latex += " \\\\\n\\midrule\n"
        
        # Matrix body
        comparison_matrix = results['comparison_matrix']['ndcg_at_10']
        
        for i, method1 in enumerate(results['methods']):
            display_name1 = self.method_display_names.get(method1, method1)
            latex += display_name1
            
            for j, method2 in enumerate(results['methods']):
                if i == j:
                    latex += " & —"  # Diagonal
                elif i < j:
                    # Upper triangular
                    comparison_key = f"{method1}_vs_{method2}"
                    if comparison_key in comparison_matrix:
                        result = comparison_matrix[comparison_key]
                        if result['significant_bonferroni_corrected']:
                            if result['p_value_bonferroni_corrected'] < 0.001:
                                latex += " & ***"
                            elif result['p_value_bonferroni_corrected'] < 0.01:
                                latex += " & **"
                            else:
                                latex += " & *"
                        else:
                            latex += " & ns"
                    else:
                        latex += " & —"
                else:
                    # Lower triangular - show effect sizes
                    comparison_key = f"{method2}_vs_{method1}"
                    if comparison_key in comparison_matrix:
                        effect = results['effect_sizes']['ndcg_at_10'][comparison_key]
                        d = abs(effect['cohens_d'])
                        if d >= 0.8:
                            latex += " & L"  # Large
                        elif d >= 0.5:
                            latex += " & M"  # Medium
                        elif d >= 0.2:
                            latex += " & S"  # Small
                        else:
                            latex += " & N"  # Negligible
                    else:
                        latex += " & —"
            
            latex += " \\\\\n"
        
        latex += r"""\bottomrule
\multicolumn{""" + str(len(results['methods']) + 1) + r"""}{l}{\small Upper triangle: significance (*** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant)} \\
\multicolumn{""" + str(len(results['methods']) + 1) + r"""}{l}{\small Lower triangle: effect size (L = large, M = medium, S = small, N = negligible)} \\
\multicolumn{""" + str(len(results['methods']) + 1) + r"""}{l}{\small All p-values Bonferroni-corrected for multiple comparisons} \\
\end{tabular}
\end{table*}
"""
        
        with open(output_file, 'w') as f:
            f.write(latex)
        print(f"  Generated: {output_file.name}")
    
    def _generate_effect_sizes_table(self, results: Dict[str, Any], output_file: Path):
        """Generate effect sizes analysis table"""
        latex = r"""\begin{table*}[htbp]
\centering
\caption{Effect Sizes for Key Comparisons}
\label{tab:effect-sizes}
\begin{tabular}{llccl}
\toprule
Comparison & Metric & Cohen's d & 95\% CI for Difference & Interpretation \\
\midrule
"""
        
        # Focus on Lethe vs best baselines
        key_comparisons = [
            ('iter4', 'baseline_bm25_vector_simple'),
            ('iter3', 'baseline_bm25_vector_simple'), 
            ('iter2', 'baseline_bm25_vector_simple'),
            ('iter1', 'baseline_bm25_vector_simple')
        ]
        
        for method1, method2 in key_comparisons:
            comparison_key = f"{method1}_vs_{method2}"
            
            for metric in ['ndcg_at_10', 'recall_at_50', 'coverage_at_n']:
                if (comparison_key in results['effect_sizes'][metric] and 
                    comparison_key in results['confidence_intervals'][metric]):
                    
                    effect = results['effect_sizes'][metric][comparison_key]
                    ci = results['confidence_intervals'][metric][comparison_key]
                    comparison = results['comparison_matrix'][metric][comparison_key]
                    
                    method1_display = self.method_display_names.get(method1, method1)
                    method2_display = self.method_display_names.get(method2, method2)
                    
                    cohens_d = effect['cohens_d']
                    magnitude = effect['magnitude'].title()
                    ci_lower = ci['diff_ci_lower']
                    ci_upper = ci['diff_ci_upper']
                    
                    metric_display = metric.replace('_', ' ').title()
                    
                    latex += f"{method1_display} vs {method2_display} & {metric_display} & "
                    latex += f"{cohens_d:.3f} & [{ci_lower:.3f}, {ci_upper:.3f}] & {magnitude} \\\\\n"
        
        latex += r"""\bottomrule
\multicolumn{5}{l}{\small Cohen's d: 0.2 = small, 0.5 = medium, 0.8 = large effect} \\
\end{tabular}
\end{table*}
"""
        
        with open(output_file, 'w') as f:
            f.write(latex)
        print(f"  Generated: {output_file.name}")
    
    def _generate_multiple_corrections_table(self, results: Dict[str, Any], output_file: Path):
        """Generate multiple comparison corrections summary table"""
        summary = results['correction_summary']
        
        latex = r"""\begin{table}[htbp]
\centering
\caption{Multiple Comparison Correction Analysis}
\label{tab:multiple-corrections}
\begin{tabular}{lc}
\toprule
Analysis Parameter & Value \\
\midrule
Total Statistical Comparisons & """ + str(summary['total_comparisons']) + r""" \\
Significant (Uncorrected α = 0.05) & """ + str(summary['significant_uncorrected']) + r""" \\
Significant (Bonferroni Corrected) & """ + str(summary['significant_bonferroni']) + r""" \\
Significant (FDR Corrected) & """ + str(summary['significant_fdr']) + r""" \\
Bonferroni α & """ + f"{results['bonferroni_alpha']:.6f}" + r""" \\
Type I Error Control & Bonferroni (FWER) \\
False Discovery Rate Control & Benjamini-Hochberg \\
\bottomrule
\end{tabular}
\end{table}
"""
        
        with open(output_file, 'w') as f:
            f.write(latex)
        print(f"  Generated: {output_file.name}")
    
    def _generate_baseline_validation_table(self, validation_results: Dict[str, Any], output_file: Path):
        """Generate baseline validation summary table"""
        latex = r"""\begin{table}[htbp]
\centering
\caption{Baseline Methods Validation Summary}
\label{tab:baseline-validation}
\begin{tabular}{lcc}
\toprule
Baseline Method & Sample Size & NDCG@10 Mean \\
\midrule
"""
        
        for baseline in validation_results['expected_baselines']:
            if baseline in validation_results['baseline_counts']:
                count = validation_results['baseline_counts'][baseline]
                if baseline in validation_results['baseline_metrics_coverage']:
                    ndcg_mean = validation_results['baseline_metrics_coverage'][baseline].get('ndcg_at_10', {}).get('mean')
                    mean_str = f"{ndcg_mean:.3f}" if ndcg_mean is not None else "N/A"
                else:
                    mean_str = "N/A"
                
                display_name = self.method_display_names.get(baseline, baseline)
                latex += f"{display_name} & {count} & {mean_str} \\\\\n"
        
        latex += r"""\bottomrule
\multicolumn{3}{l}{\small All baselines evaluated on consistent query set} \\
\end{tabular}
\end{table}
"""
        
        with open(output_file, 'w') as f:
            f.write(latex)
        print(f"  Generated: {output_file.name}")
    
    def _generate_supplementary_tables(self, results: Dict[str, Any], output_dir: Path):
        """Generate comprehensive supplementary tables"""
        output_dir.mkdir(exist_ok=True)
        
        # Complete pairwise comparison results
        with open(output_dir / "complete_pairwise_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"  Generated supplementary materials in: {output_dir.name}/")
    
    def _create_significance_heatmap(self, results: Dict[str, Any], output_file: Path):
        """Create statistical significance heatmap"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        methods = results['methods']
        n_methods = len(methods)
        
        # Create significance matrix
        sig_matrix = np.zeros((n_methods, n_methods))
        p_matrix = np.ones((n_methods, n_methods))
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i != j:
                    comparison_key = f"{method1}_vs_{method2}" if i < j else f"{method2}_vs_{method1}"
                    if comparison_key in results['comparison_matrix']['ndcg_at_10']:
                        result = results['comparison_matrix']['ndcg_at_10'][comparison_key]
                        p_val = result['p_value_bonferroni_corrected']
                        p_matrix[i, j] = p_val
                        
                        if p_val < 0.001:
                            sig_matrix[i, j] = 3  # ***
                        elif p_val < 0.01:
                            sig_matrix[i, j] = 2  # **
                        elif p_val < 0.05:
                            sig_matrix[i, j] = 1  # *
                        else:
                            sig_matrix[i, j] = 0  # ns
        
        # Create heatmap
        im = ax.imshow(sig_matrix, cmap='RdYlBu_r', vmin=0, vmax=3)
        
        # Set ticks and labels
        method_labels = [self.method_display_names.get(m, m) for m in methods]
        ax.set_xticks(range(n_methods))
        ax.set_yticks(range(n_methods))
        ax.set_xticklabels(method_labels, rotation=45, ha='right')
        ax.set_yticklabels(method_labels)
        
        # Add significance annotations
        for i in range(n_methods):
            for j in range(n_methods):
                if i != j:
                    sig_val = sig_matrix[i, j]
                    if sig_val == 3:
                        text = '***'
                    elif sig_val == 2:
                        text = '**'
                    elif sig_val == 1:
                        text = '*'
                    else:
                        text = 'ns'
                    
                    color = 'white' if sig_val >= 2 else 'black'
                    ax.text(j, i, text, ha='center', va='center', color=color, fontweight='bold')
        
        # Colorbar and labels
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(['ns', '*', '**', '***'])
        
        ax.set_title('Statistical Significance Matrix (NDCG@10)\nBonferroni-Corrected p-values', 
                    fontsize=14, pad=20)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_file).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Generated: {output_file.name}")
    
    def _create_performance_distributions(self, df: pd.DataFrame, output_file: Path):
        """Create performance distribution plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        metrics = ['ndcg_at_10', 'recall_at_50', 'coverage_at_n', 'latency_ms_total', 
                  'contradiction_rate', 'hallucination_rate']
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Prepare data for violin plot
            data_to_plot = []
            labels = []
            
            for method in ['baseline_bm25_vector_simple', 'iter1', 'iter2', 'iter3', 'iter4']:
                if method in df['method'].values:
                    method_data = df[df['method'] == method][metric].dropna()
                    if len(method_data) > 0:
                        data_to_plot.append(method_data.values)
                        labels.append(self.method_display_names.get(method, method))
            
            if data_to_plot:
                parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)))
                
                # Customize violin plot
                for pc in parts['bodies']:
                    pc.set_facecolor('lightblue')
                    pc.set_alpha(0.7)
                
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Performance Distributions Across Methods', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_file).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Generated: {output_file.name}")
    
    def _create_effect_sizes_plot(self, results: Dict[str, Any], output_file: Path):
        """Create effect sizes visualization"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Focus on key comparisons
        comparisons = []
        effect_sizes = []
        metrics_list = []
        
        key_methods = ['iter1', 'iter2', 'iter3', 'iter4']
        baseline = 'baseline_bm25_vector_simple'
        
        for method in key_methods:
            comparison_key = f"{method}_vs_{baseline}"
            for metric in ['ndcg_at_10', 'recall_at_50', 'coverage_at_n']:
                if comparison_key in results['effect_sizes'][metric]:
                    effect = results['effect_sizes'][metric][comparison_key]
                    comparisons.append(f"{self.method_display_names[method]} vs BM25+Vector")
                    effect_sizes.append(effect['cohens_d'])
                    metrics_list.append(metric.replace('_', ' ').title())
        
        # Create grouped bar plot
        y_pos = np.arange(len(comparisons))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green for different metrics
        
        bars = ax.barh(y_pos, effect_sizes, 
                      color=[colors[i % 3] for i in range(len(effect_sizes))])
        
        # Add effect size interpretation lines
        ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small Effect')
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Medium Effect')
        ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.9, label='Large Effect')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{comp}\n({metric})" for comp, metric in zip(comparisons, metrics_list)])
        ax.set_xlabel("Cohen's d Effect Size")
        ax.set_title("Effect Sizes for Key Performance Comparisons")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_file).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Generated: {output_file.name}")
    
    def _create_confidence_intervals_plot(self, results: Dict[str, Any], output_file: Path):
        """Create confidence intervals visualization"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        methods = ['baseline_bm25_vector_simple', 'iter1', 'iter2', 'iter3', 'iter4']
        metric = 'ndcg_at_10'
        
        means = []
        ci_lowers = []
        ci_uppers = []
        method_labels = []
        
        for method in methods:
            if method in results['summary_statistics']:
                stats = results['summary_statistics'][method][metric]
                means.append(stats['mean'])
                ci_lowers.append(stats['ci_lower'])
                ci_uppers.append(stats['ci_upper'])
                method_labels.append(self.method_display_names[method])
        
        x_pos = np.arange(len(method_labels))
        
        # Plot means with error bars
        ax.errorbar(x_pos, means, 
                   yerr=[np.array(means) - np.array(ci_lowers), 
                         np.array(ci_uppers) - np.array(means)],
                   fmt='o', capsize=5, capthick=2, linewidth=2, markersize=8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(method_labels, rotation=45, ha='right')
        ax.set_ylabel('NDCG@10')
        ax.set_title('NDCG@10 Performance with 95% Confidence Intervals')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_file).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Generated: {output_file.name}")
    
    def _create_multiple_comparison_impact(self, results: Dict[str, Any], output_file: Path):
        """Create multiple comparison correction impact visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        summary = results['correction_summary']
        
        # Bar chart of significant comparisons
        categories = ['Uncorrected', 'Bonferroni', 'FDR']
        counts = [summary['significant_uncorrected'], 
                 summary['significant_bonferroni'],
                 summary['significant_fdr']]
        
        bars = ax1.bar(categories, counts, color=['lightcoral', 'steelblue', 'lightgreen'])
        ax1.set_ylabel('Number of Significant Comparisons')
        ax1.set_title('Impact of Multiple Comparison Corrections')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart showing correction impact
        impact_data = [summary['significant_bonferroni'], 
                      summary['significant_uncorrected'] - summary['significant_bonferroni']]
        labels = ['Remain Significant', 'Become Non-Significant']
        colors = ['lightgreen', 'lightcoral']
        
        ax2.pie(impact_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Bonferroni Correction Impact')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_file).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Generated: {output_file.name}")
    
    def run_comprehensive_analysis(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run complete enhanced statistical analysis pipeline"""
        print("Starting comprehensive enhanced statistical analysis...")
        print("=" * 80)
        
        # Load data
        df = self.load_comprehensive_data()
        
        # Validate baselines
        validation_results = self.validate_baseline_completeness(df)
        
        # Comprehensive pairwise comparisons
        pairwise_results = self.compute_comprehensive_pairwise_comparisons(df)
        
        # Apply multiple comparison corrections
        corrected_results = self.apply_multiple_comparison_correction(pairwise_results)
        
        # Reproducibility analysis
        reproducibility_results = self.compute_reproducibility_metrics(df)
        
        # Combine all results
        final_results = {
            **corrected_results,
            'validation_results': validation_results,
            'reproducibility_results': reproducibility_results,
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_datapoints': len(df),
                'analysis_version': '2.0_enhanced',
                'statistical_parameters': {
                    'alpha': self.alpha,
                    'confidence_level': self.confidence_level,
                    'bootstrap_samples': self.bootstrap_n_samples
                }
            }
        }
        
        # Generate publication materials
        self.generate_publication_ready_tables(final_results, validation_results)
        self.generate_statistical_figures(df, final_results)
        
        # Save comprehensive results
        results_file = self.output_dir.parent / "artifacts" / "enhanced_statistical_analysis.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print("\n" + "=" * 80)
        print("ENHANCED STATISTICAL ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"Total datapoints analyzed: {len(df)}")
        print(f"Methods evaluated: {len(final_results['methods'])}")
        print(f"Total pairwise comparisons: {final_results['total_comparisons']}")
        print(f"Bonferroni-corrected α: {final_results['bonferroni_alpha']:.6f}")
        print(f"Significant after correction: {final_results['correction_summary']['significant_bonferroni']}")
        print(f"\nResults saved to: {results_file}")
        print(f"Publication materials: {self.output_dir}/")
        
        return df, final_results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Statistical Analysis for Lethe Research")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory")
    parser.add_argument("--output-dir", default="paper", help="Output directory")
    
    args = parser.parse_args()
    
    analyzer = EnhancedStatisticalAnalyzer(args.artifacts_dir, args.output_dir)
    df, results = analyzer.run_comprehensive_analysis()
    
    print("\nAll publication-ready materials generated successfully!")


if __name__ == "__main__":
    main()