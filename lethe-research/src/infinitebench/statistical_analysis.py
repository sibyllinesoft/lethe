"""
InfiniteBench Statistical Analysis Integration
===========================================

Statistical analysis utilities for InfiniteBench evaluation results,
integrating with Lethe's existing BCa bootstrap framework for academic
publication standards.

Author: Lethe Research Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import logging
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Import from existing Lethe statistical framework
from evaluation.bootstrap_ci import BootstrapResult, ComparisonResult, LetHeEvaluationValidator

logger = logging.getLogger(__name__)

@dataclass
class InfiniteBenchAnalysisResult:
    """Results from InfiniteBench statistical analysis."""
    
    experiment_name: str
    method_comparisons: Dict[str, Dict[str, ComparisonResult]]
    task_analysis: Dict[str, Dict[str, Any]]
    overall_ranking: List[Tuple[str, float, float]]  # (method, score, ci_lower)
    significant_improvements: List[Tuple[str, str, float]]  # (method1, method2, p_value)
    publication_tables: Dict[str, pd.DataFrame]
    
class InfiniteBenchStatistics:
    """
    Statistical analysis for InfiniteBench evaluation results.
    
    Integrates with Lethe's existing BCa bootstrap framework to provide
    publication-quality statistical analysis of long-context retrieval performance.
    """
    
    def __init__(self, confidence_level: float = 0.95, n_bootstrap: int = 10000):
        """
        Initialize statistical analysis.
        
        Args:
            confidence_level: Confidence level for bootstrap CIs
            n_bootstrap: Number of bootstrap iterations
        """
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.validator = LetHeEvaluationValidator(
            confidence_level=confidence_level,
            n_bootstrap_iterations=n_bootstrap
        )
    
    def analyze_experiment_results(self, 
                                 results_file: Path,
                                 baseline_method: str = "BM25") -> InfiniteBenchAnalysisResult:
        """
        Perform comprehensive statistical analysis of InfiniteBench results.
        
        Args:
            results_file: Path to experiment results JSON file
            baseline_method: Baseline method for comparisons
            
        Returns:
            InfiniteBenchAnalysisResult with comprehensive analysis
        """
        logger.info(f"Analyzing InfiniteBench results from {results_file}")
        
        # Load results
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        # Extract method results
        method_results = self._extract_method_results(results_data)
        
        # Perform pairwise method comparisons
        method_comparisons = self._perform_method_comparisons(method_results, baseline_method)
        
        # Analyze task-specific performance
        task_analysis = self._analyze_task_performance(method_results)
        
        # Create overall ranking with confidence intervals
        overall_ranking = self._create_overall_ranking(method_results)
        
        # Identify significant improvements
        significant_improvements = self._identify_significant_improvements(method_comparisons)
        
        # Generate publication tables
        publication_tables = self._generate_publication_tables(method_results, method_comparisons)
        
        return InfiniteBenchAnalysisResult(
            experiment_name=results_data.get('config', {}).get('experiment_name', 'Unknown'),
            method_comparisons=method_comparisons,
            task_analysis=task_analysis,
            overall_ranking=overall_ranking,
            significant_improvements=significant_improvements,
            publication_tables=publication_tables
        )
    
    def _extract_method_results(self, results_data: Dict[str, Any]) -> Dict[str, Dict[str, List[float]]]:
        """Extract method performance scores by task."""
        
        method_results = {}
        
        for result in results_data.get('method_results', []):
            method_name = result['method_name']
            task_name = result['task_name']
            overall_score = result['overall_score']
            
            if method_name not in method_results:
                method_results[method_name] = {}
            
            if task_name not in method_results[method_name]:
                method_results[method_name][task_name] = []
            
            method_results[method_name][task_name].append(overall_score)
        
        return method_results
    
    def _perform_method_comparisons(self, 
                                  method_results: Dict[str, Dict[str, List[float]]],
                                  baseline_method: str) -> Dict[str, Dict[str, ComparisonResult]]:
        """Perform statistical comparisons between methods."""
        
        comparisons = {}
        methods = list(method_results.keys())
        
        for method1 in methods:
            comparisons[method1] = {}
            
            for method2 in methods:
                if method1 == method2:
                    continue
                
                # Collect scores across all tasks for both methods
                scores1 = []
                scores2 = []
                
                for task in method_results[method1]:
                    if task in method_results[method2]:
                        scores1.extend(method_results[method1][task])
                        scores2.extend(method_results[method2][task])
                
                if scores1 and scores2:
                    # Use existing Lethe statistical framework
                    comparison = self.validator.compare_methods(
                        method_a_scores=scores1,
                        method_b_scores=scores2,
                        method_a_name=method1,
                        method_b_name=method2
                    )
                    
                    comparisons[method1][method2] = comparison
        
        return comparisons
    
    def _analyze_task_performance(self, 
                                method_results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, Any]]:
        """Analyze performance patterns across tasks."""
        
        task_analysis = {}
        
        # Get all unique tasks
        all_tasks = set()
        for method_data in method_results.values():
            all_tasks.update(method_data.keys())
        
        for task in all_tasks:
            task_scores = []
            method_scores = {}
            
            for method, method_data in method_results.items():
                if task in method_data:
                    scores = method_data[task]
                    task_scores.extend(scores)
                    method_scores[method] = np.mean(scores)
            
            if task_scores:
                # Task difficulty metrics
                task_analysis[task] = {
                    'mean_score': np.mean(task_scores),
                    'std_score': np.std(task_scores),
                    'min_score': np.min(task_scores),
                    'max_score': np.max(task_scores),
                    'score_range': np.max(task_scores) - np.min(task_scores),
                    'num_methods': len(method_scores),
                    'method_scores': method_scores,
                    'difficulty_ranking': len(all_tasks) - np.argsort([np.mean(task_scores)])[0] + 1
                }
        
        return task_analysis
    
    def _create_overall_ranking(self, 
                              method_results: Dict[str, Dict[str, List[float]]]) -> List[Tuple[str, float, float]]:
        """Create overall method ranking with confidence intervals."""
        
        method_rankings = []
        
        for method, method_data in method_results.items():
            # Collect all scores for this method
            all_scores = []
            for task_scores in method_data.values():
                all_scores.extend(task_scores)
            
            if all_scores:
                # Calculate bootstrap confidence interval
                bootstrap_result = self.validator.bootstrap_confidence_interval(
                    data=all_scores,
                    statistic=np.mean,
                    metric_name=f"{method}_overall"
                )
                
                method_rankings.append((
                    method,
                    bootstrap_result.original_value,
                    bootstrap_result.ci_lower
                ))
        
        # Sort by mean score (descending)
        method_rankings.sort(key=lambda x: x[1], reverse=True)
        
        return method_rankings
    
    def _identify_significant_improvements(self, 
                                        method_comparisons: Dict[str, Dict[str, ComparisonResult]]) -> List[Tuple[str, str, float]]:
        """Identify statistically significant improvements."""
        
        significant_improvements = []
        
        for method1, comparisons in method_comparisons.items():
            for method2, comparison in comparisons.items():
                if (comparison.is_significant and 
                    comparison.effect_size > 0 and
                    comparison.improvement_percentage > 5.0):  # At least 5% improvement
                    
                    significant_improvements.append((
                        method1, 
                        method2, 
                        comparison.p_value
                    ))
        
        # Sort by effect size (largest improvements first)
        significant_improvements.sort(key=lambda x: method_comparisons[x[0]][x[1]].effect_size, reverse=True)
        
        return significant_improvements
    
    def _generate_publication_tables(self, 
                                   method_results: Dict[str, Dict[str, List[float]]],
                                   method_comparisons: Dict[str, Dict[str, ComparisonResult]]) -> Dict[str, pd.DataFrame]:
        """Generate publication-ready tables."""
        
        tables = {}
        
        # Table 1: Method performance by task
        performance_data = []
        
        all_tasks = set()
        for method_data in method_results.values():
            all_tasks.update(method_data.keys())
        all_tasks = sorted(all_tasks)
        
        for method, method_data in method_results.items():
            row = {'Method': method}
            
            for task in all_tasks:
                if task in method_data:
                    scores = method_data[task]
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    row[task] = f"{mean_score:.3f} ± {std_score:.3f}"
                else:
                    row[task] = "—"
            
            # Overall average
            all_scores = []
            for task_scores in method_data.values():
                all_scores.extend(task_scores)
            
            if all_scores:
                overall_mean = np.mean(all_scores)
                overall_std = np.std(all_scores)
                row['Overall'] = f"{overall_mean:.3f} ± {overall_std:.3f}"
            else:
                row['Overall'] = "—"
            
            performance_data.append(row)
        
        tables['performance_by_task'] = pd.DataFrame(performance_data)
        
        # Table 2: Statistical significance matrix
        methods = sorted(method_results.keys())
        significance_matrix = []
        
        for method1 in methods:
            row = {'Method': method1}
            
            for method2 in methods:
                if method1 == method2:
                    row[method2] = "—"
                elif method2 in method_comparisons.get(method1, {}):
                    comparison = method_comparisons[method1][method2]
                    if comparison.is_significant:
                        if comparison.effect_size > 0:
                            row[method2] = f"↑ {comparison.improvement_percentage:.1f}%"
                        else:
                            row[method2] = f"↓ {abs(comparison.improvement_percentage):.1f}%"
                    else:
                        row[method2] = "n.s."
                else:
                    row[method2] = "—"
            
            significance_matrix.append(row)
        
        tables['significance_matrix'] = pd.DataFrame(significance_matrix)
        
        # Table 3: Effect sizes and confidence intervals
        effect_size_data = []
        
        for method1, comparisons in method_comparisons.items():
            for method2, comparison in comparisons.items():
                if comparison.is_significant:
                    effect_size_data.append({
                        'Method A': method1,
                        'Method B': method2,
                        'Effect Size (Cohen\'s d)': f"{comparison.effect_size:.3f}",
                        'Improvement %': f"{comparison.improvement_percentage:.1f}%",
                        'p-value': f"{comparison.p_value:.4f}",
                        '95% CI': f"[{comparison.ci_lower:.3f}, {comparison.ci_upper:.3f}]"
                    })
        
        if effect_size_data:
            # Sort by effect size
            effect_size_data.sort(key=lambda x: float(x['Effect Size (Cohen\'s d)'].replace('—', '0')), reverse=True)
            tables['effect_sizes'] = pd.DataFrame(effect_size_data)
        
        return tables
    
    def generate_publication_report(self, 
                                  analysis_result: InfiniteBenchAnalysisResult,
                                  output_dir: Path) -> Path:
        """Generate a comprehensive publication report."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / f"{analysis_result.experiment_name}_publication_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# InfiniteBench Evaluation Report: {analysis_result.experiment_name}\n\n")
            
            # Executive summary
            f.write("## Executive Summary\n\n")
            f.write(f"This report presents a comprehensive evaluation of long-context retrieval methods ")
            f.write(f"on the InfiniteBench dataset, following academic publication standards.\n\n")
            
            # Method ranking
            f.write("## Overall Method Ranking\n\n")
            f.write("| Rank | Method | Mean Score | 95% CI Lower Bound |\n")
            f.write("|------|--------|------------|--------------------|\n")
            
            for i, (method, score, ci_lower) in enumerate(analysis_result.overall_ranking, 1):
                f.write(f"| {i} | {method} | {score:.3f} | {ci_lower:.3f} |\n")
            
            f.write("\n")
            
            # Significant improvements
            if analysis_result.significant_improvements:
                f.write("## Statistically Significant Improvements\n\n")
                f.write("The following method comparisons show statistically significant improvements (p < 0.05):\n\n")
                
                for method1, method2, p_value in analysis_result.significant_improvements:
                    comparison = analysis_result.method_comparisons[method1][method2]
                    f.write(f"- **{method1} vs {method2}**: ")
                    f.write(f"{comparison.improvement_percentage:.1f}% improvement ")
                    f.write(f"(p = {p_value:.4f}, d = {comparison.effect_size:.3f})\n")
                
                f.write("\n")
            
            # Task analysis
            f.write("## Task Difficulty Analysis\n\n")
            f.write("| Task | Mean Score | Std Dev | Score Range | Difficulty Rank |\n")
            f.write("|------|------------|---------|-------------|----------------|\n")
            
            sorted_tasks = sorted(
                analysis_result.task_analysis.items(),
                key=lambda x: x[1]['mean_score']
            )
            
            for task, stats in sorted_tasks:
                f.write(f"| {task} | {stats['mean_score']:.3f} | {stats['std_score']:.3f} | ")
                f.write(f"{stats['score_range']:.3f} | {stats['difficulty_ranking']} |\n")
            
            f.write("\n")
            
            # Performance tables
            for table_name, table_df in analysis_result.publication_tables.items():
                f.write(f"## {table_name.replace('_', ' ').title()}\n\n")
                f.write(table_df.to_markdown(index=False))
                f.write("\n\n")
            
            # Methodology
            f.write("## Statistical Methodology\n\n")
            f.write("Statistical analysis was performed using bias-corrected and accelerated (BCa) ")
            f.write(f"bootstrap confidence intervals with {self.n_bootstrap:,} iterations. ")
            f.write("Statistical significance was assessed using paired t-tests with Bonferroni ")
            f.write("correction for multiple comparisons. Effect sizes are reported using Cohen's d.\n\n")
            
            f.write("### Significance Criteria\n\n")
            f.write(f"- Confidence level: {self.confidence_level * 100}%\n")
            f.write("- Minimum improvement threshold: 5%\n")
            f.write("- Effect size interpretation: small (0.2), medium (0.5), large (0.8)\n\n")
        
        logger.info(f"Publication report saved to {report_file}")
        return report_file
    
    def create_visualization_plots(self,
                                 analysis_result: InfiniteBenchAnalysisResult,
                                 output_dir: Path) -> List[Path]:
        """Create publication-quality visualization plots."""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_files = []
        
        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        
        # Plot 1: Method ranking with confidence intervals
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = [item[0] for item in analysis_result.overall_ranking]
        scores = [item[1] for item in analysis_result.overall_ranking]
        ci_lowers = [item[2] for item in analysis_result.overall_ranking]
        
        # Calculate error bars (assuming symmetric for display)
        errors = [score - ci_lower for score, ci_lower in zip(scores, ci_lowers)]
        
        bars = ax.bar(range(len(methods)), scores, yerr=errors, 
                     capsize=5, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Mean Performance Score')
        ax.set_title('InfiniteBench Method Performance with 95% Confidence Intervals')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, score, error in zip(bars, scores, errors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + error + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plot_file = output_dir / 'method_ranking.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plot_files.append(plot_file)
        plt.close()
        
        # Plot 2: Task difficulty heatmap
        if analysis_result.publication_tables.get('performance_by_task') is not None:
            df = analysis_result.publication_tables['performance_by_task'].copy()
            
            # Extract numeric values from "mean ± std" format
            numeric_df = df.set_index('Method')
            for col in numeric_df.columns:
                if col != 'Overall':
                    numeric_df[col] = numeric_df[col].apply(
                        lambda x: float(x.split(' ± ')[0]) if '±' in str(x) and str(x) != '—' else np.nan
                    )
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            mask = numeric_df.isna()
            sns.heatmap(numeric_df, annot=True, cmap='RdYlBu_r', center=0.5,
                       mask=mask, fmt='.3f', cbar_kws={'label': 'Performance Score'},
                       ax=ax, linewidths=0.5)
            
            ax.set_title('Method Performance Across InfiniteBench Tasks')
            ax.set_xlabel('Task')
            ax.set_ylabel('Method')
            
            plt.tight_layout()
            plot_file = output_dir / 'task_performance_heatmap.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plot_files.append(plot_file)
            plt.close()
        
        logger.info(f"Generated {len(plot_files)} visualization plots in {output_dir}")
        return plot_files

def main():
    """Example usage of InfiniteBench statistical analysis."""
    
    # Example analysis of results
    stats = InfiniteBenchStatistics()
    
    # Mock results file path (would be actual experiment results)
    results_file = Path("results/infinitebench_test/infinitebench_test_results.json")
    
    if results_file.exists():
        try:
            # Perform analysis
            analysis = stats.analyze_experiment_results(results_file)
            
            # Generate report
            output_dir = Path("results/infinitebench_analysis")
            report_file = stats.generate_publication_report(analysis, output_dir)
            
            # Create visualizations
            plot_files = stats.create_visualization_plots(analysis, output_dir)
            
            print(f"Analysis complete!")
            print(f"Report: {report_file}")
            print(f"Plots: {plot_files}")
            
            # Print key findings
            print("\nKey Findings:")
            print("=" * 50)
            
            print("\nTop 3 Methods:")
            for i, (method, score, ci_lower) in enumerate(analysis.overall_ranking[:3], 1):
                print(f"{i}. {method}: {score:.3f} (CI: {ci_lower:.3f}+)")
            
            if analysis.significant_improvements:
                print(f"\nSignificant Improvements Found: {len(analysis.significant_improvements)}")
                for method1, method2, p_value in analysis.significant_improvements[:3]:
                    comparison = analysis.method_comparisons[method1][method2]
                    print(f"  {method1} > {method2}: {comparison.improvement_percentage:.1f}% (p={p_value:.4f})")
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
    else:
        print(f"Results file not found: {results_file}")
        print("Run an InfiniteBench experiment first to generate results.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()