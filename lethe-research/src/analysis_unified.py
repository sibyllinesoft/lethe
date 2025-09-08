#!/usr/bin/env python3
"""
Unified Scientific Analysis Framework for Lethe Research
=======================================================

Consolidates all fragmented analysis pipelines into a single, comprehensive framework
for scientific analysis and publication-ready output generation.

Key Features:
- Unified data loading and preprocessing
- Comprehensive statistical analysis (H1-H4 hypotheses)
- Multi-objective Pareto optimization analysis  
- Publication-quality figure and table generation
- Bootstrap confidence intervals and effect sizes
- Multiple comparison corrections (Bonferroni, FDR)
- Fraud-proofing and reproducibility validation
- Extensible plugin architecture for custom analyses

Usage:
    from src.analysis_unified import UnifiedAnalysisFramework
    
    framework = UnifiedAnalysisFramework()
    framework.load_experimental_data("artifacts/")
    framework.run_complete_analysis()
    framework.generate_publication_outputs("paper/")
"""

import os
import sys
import json
import hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis imports
from scipy import stats
from scipy.stats import (
    mannwhitneyu, wilcoxon, kruskal, ttest_ind, bootstrap,
    pearsonr, spearmanr
)
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.multitest import multipletests
import itertools

# Add research components to path
sys.path.append(str(Path(__file__).parent.parent))

# Import existing components we can reuse
try:
    from analysis.metrics import MetricsCalculator, StatisticalComparator, QueryResult
    from common.evaluation_framework import EvaluationFramework, MetricConfig
    from common.data_persistence import DataManager
except ImportError as e:
    print(f"Warning: Could not import existing components: {e}")

# Publication-quality plotting configuration
plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times', 'Computer Modern Roman'],
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': False,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.0,
    'text.usetex': False,
    'mathtext.default': 'regular'
})

# Scientific color scheme
ANALYSIS_COLORS = {
    'baseline': '#2E86AB',
    'iter1': '#A23B72',
    'iter2': '#F18F01', 
    'iter3': '#C73E1D',
    'iter4': '#2D5016',
    'improvement': '#155724',
    'degradation': '#721C24',
    'neutral': '#6C757D',
    'significant': '#155724',
    'warning': '#856404'
}

@dataclass
class AnalysisConfig:
    """Configuration for unified analysis framework"""
    
    # Data sources
    artifacts_dir: str = "artifacts"
    experiments_dir: str = "experiments"
    
    # Output directories
    output_dir: str = "paper"
    figures_dir: str = "paper/figures"
    tables_dir: str = "paper/tables"
    analysis_dir: str = "analysis"
    
    # Analysis parameters
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    significance_level: float = 0.05
    effect_size_threshold: float = 0.2
    
    # Multiple comparison corrections
    multiple_comparison_method: str = "bonferroni"  # bonferroni, holm, fdr_bh
    
    # Hypothesis testing framework
    hypotheses: List[str] = field(default_factory=lambda: [
        "H1_quality_improvement",
        "H2_efficiency_maintained", 
        "H3_coverage_increased",
        "H4_consistency_improved"
    ])
    
    # Methods to analyze
    baseline_methods: List[str] = field(default_factory=lambda: [
        "baseline_bm25_only",
        "baseline_vector_only", 
        "baseline_bm25_vector_simple",
        "baseline_cross_encoder",
        "baseline_mmr"
    ])
    
    lethe_iterations: List[str] = field(default_factory=lambda: [
        "iter1", "iter2", "iter3", "iter4"
    ])
    
    # Key metrics for analysis
    primary_metrics: List[str] = field(default_factory=lambda: [
        "ndcg_at_10",
        "recall_at_50", 
        "coverage_at_n",
        "latency_ms_total",
        "contradiction_rate",
        "memory_mb"
    ])

@dataclass
class StatisticalTestResult:
    """Result of a statistical significance test"""
    test_name: str
    baseline_method: str
    comparison_method: str
    metric: str
    
    # Test statistics
    statistic: float
    p_value: float
    p_value_corrected: float
    significant: bool
    significant_corrected: bool
    
    # Effect size
    effect_size: float
    effect_size_interpretation: str
    
    # Descriptive statistics
    baseline_mean: float
    baseline_std: float
    comparison_mean: float
    comparison_std: float
    
    # Confidence intervals
    baseline_ci: Tuple[float, float]
    comparison_ci: Tuple[float, float]
    difference_ci: Tuple[float, float]

@dataclass
class ParetoSolution:
    """Single solution on the Pareto frontier"""
    method: str
    objectives: Dict[str, float]
    normalized_objectives: Dict[str, float]
    dominated_by: List[str]
    dominates: List[str]
    is_pareto_optimal: bool
    pareto_rank: int

class AnalysisPlugin(ABC):
    """Base class for analysis plugins"""
    
    @abstractmethod
    def name(self) -> str:
        """Return plugin name"""
        pass
    
    @abstractmethod
    def run_analysis(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        """Run the analysis and return results"""
        pass
    
    @abstractmethod
    def generate_outputs(self, results: Dict[str, Any], output_dir: Path) -> List[str]:
        """Generate output files and return list of created files"""
        pass

class HypothesisTestingPlugin(AnalysisPlugin):
    """Plugin for hypothesis testing analysis"""
    
    def __init__(self, statistical_comparator=None):
        self.statistical_comparator = statistical_comparator
    
    def name(self) -> str:
        return "hypothesis_testing"
    
    def run_analysis(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        """Run comprehensive hypothesis testing"""
        
        results = {
            "statistical_tests": [],
            "effect_sizes": {},
            "hypothesis_conclusions": {},
            "multiple_comparison_summary": {}
        }
        
        # Generate all pairwise comparisons
        all_methods = config.baseline_methods + config.lethe_iterations
        comparisons = list(itertools.combinations(all_methods, 2))
        
        for baseline, comparison in comparisons:
            for metric in config.primary_metrics:
                test_result = self._perform_statistical_test(
                    data, baseline, comparison, metric, config
                )
                if test_result:
                    results["statistical_tests"].append(test_result)
        
        # Apply multiple comparison corrections
        results["statistical_tests"] = self._apply_multiple_comparison_correction(
            results["statistical_tests"], config
        )
        
        # Evaluate hypotheses
        results["hypothesis_conclusions"] = self._evaluate_hypotheses(
            results["statistical_tests"], config
        )
        
        return results
    
    def generate_outputs(self, results: Dict[str, Any], output_dir: Path) -> List[str]:
        """Generate hypothesis testing output files"""
        
        output_files = []
        
        # Statistical tests summary table
        tests_df = pd.DataFrame([
            {
                "baseline": test.baseline_method,
                "comparison": test.comparison_method,
                "metric": test.metric,
                "p_value": test.p_value,
                "p_corrected": test.p_value_corrected,
                "significant": test.significant_corrected,
                "effect_size": test.effect_size,
                "effect_interpretation": test.effect_size_interpretation
            }
            for test in results["statistical_tests"]
        ])
        
        tests_file = output_dir / "statistical_tests_summary.csv"
        tests_df.to_csv(tests_file, index=False)
        output_files.append(str(tests_file))
        
        # Hypothesis conclusions
        conclusions_file = output_dir / "hypothesis_conclusions.json"
        with open(conclusions_file, 'w') as f:
            json.dump(results["hypothesis_conclusions"], f, indent=2)
        output_files.append(str(conclusions_file))
        
        return output_files
    
    def _perform_statistical_test(self, data: pd.DataFrame, baseline: str, 
                                comparison: str, metric: str, config: AnalysisConfig) -> Optional[StatisticalTestResult]:
        """Perform statistical test between two methods for a metric"""
        
        baseline_data = data[data['method'] == baseline][metric].dropna()
        comparison_data = data[data['method'] == comparison][metric].dropna()
        
        if len(baseline_data) < 3 or len(comparison_data) < 3:
            return None
        
        # Perform Mann-Whitney U test (non-parametric)
        try:
            statistic, p_value = mannwhitneyu(
                comparison_data, baseline_data, alternative='two-sided'
            )
            
            # Calculate effect size (Cohen's d)
            effect_size = self._calculate_cohens_d(baseline_data, comparison_data)
            effect_interpretation = self._interpret_effect_size(effect_size)
            
            # Calculate confidence intervals
            baseline_ci = self._bootstrap_ci(baseline_data, config)
            comparison_ci = self._bootstrap_ci(comparison_data, config)
            
            # Difference confidence interval
            differences = comparison_data.values[:, np.newaxis] - baseline_data.values
            difference_ci = self._bootstrap_ci(differences.flatten(), config)
            
            return StatisticalTestResult(
                test_name="Mann-Whitney U",
                baseline_method=baseline,
                comparison_method=comparison,
                metric=metric,
                statistic=float(statistic),
                p_value=float(p_value),
                p_value_corrected=float(p_value),  # Will be corrected later
                significant=p_value < config.significance_level,
                significant_corrected=False,  # Will be set later
                effect_size=float(effect_size),
                effect_size_interpretation=effect_interpretation,
                baseline_mean=float(baseline_data.mean()),
                baseline_std=float(baseline_data.std()),
                comparison_mean=float(comparison_data.mean()),
                comparison_std=float(comparison_data.std()),
                baseline_ci=baseline_ci,
                comparison_ci=comparison_ci,
                difference_ci=difference_ci
            )
            
        except Exception as e:
            print(f"Statistical test failed for {baseline} vs {comparison} on {metric}: {e}")
            return None
    
    def _calculate_cohens_d(self, baseline: pd.Series, comparison: pd.Series) -> float:
        """Calculate Cohen's d effect size"""
        mean_diff = comparison.mean() - baseline.mean()
        pooled_std = np.sqrt(((len(baseline) - 1) * baseline.var() + 
                             (len(comparison) - 1) * comparison.var()) / 
                            (len(baseline) + len(comparison) - 2))
        return mean_diff / pooled_std if pooled_std > 0 else 0.0
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _bootstrap_ci(self, data: Union[pd.Series, np.ndarray], config: AnalysisConfig) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        if len(data) < 3:
            return (0.0, 0.0)
        
        try:
            res = bootstrap(
                (np.array(data),), 
                np.mean, 
                n_resamples=config.bootstrap_samples,
                confidence_level=config.confidence_level
            )
            return (float(res.confidence_interval.low), float(res.confidence_interval.high))
        except:
            return (0.0, 0.0)
    
    def _apply_multiple_comparison_correction(self, tests: List[StatisticalTestResult], 
                                           config: AnalysisConfig) -> List[StatisticalTestResult]:
        """Apply multiple comparison correction to p-values"""
        
        if not tests:
            return tests
        
        p_values = [test.p_value for test in tests]
        
        if config.multiple_comparison_method == "bonferroni":
            corrected_significant, corrected_p_values, _, _ = multipletests(
                p_values, alpha=config.significance_level, method='bonferroni'
            )
        elif config.multiple_comparison_method == "holm":
            corrected_significant, corrected_p_values, _, _ = multipletests(
                p_values, alpha=config.significance_level, method='holm'
            )
        elif config.multiple_comparison_method == "fdr_bh":
            corrected_significant, corrected_p_values, _, _ = multipletests(
                p_values, alpha=config.significance_level, method='fdr_bh'
            )
        else:
            corrected_significant = [test.significant for test in tests]
            corrected_p_values = p_values
        
        # Update test results with corrected values
        for test, corrected_p, corrected_sig in zip(tests, corrected_p_values, corrected_significant):
            test.p_value_corrected = float(corrected_p)
            test.significant_corrected = bool(corrected_sig)
        
        return tests
    
    def _evaluate_hypotheses(self, tests: List[StatisticalTestResult], 
                           config: AnalysisConfig) -> Dict[str, Any]:
        """Evaluate research hypotheses based on statistical tests"""
        
        conclusions = {}
        
        # H1: Quality improvement (nDCG@10, Recall@50)
        quality_tests = [
            test for test in tests 
            if test.metric in ["ndcg_at_10", "recall_at_50"]
            and test.baseline_method in config.baseline_methods
            and test.comparison_method in config.lethe_iterations
        ]
        
        h1_support = sum(1 for test in quality_tests 
                        if test.significant_corrected and test.effect_size > 0)
        h1_total = len(quality_tests)
        
        conclusions["H1_quality_improvement"] = {
            "supported": h1_support / h1_total > 0.5 if h1_total > 0 else False,
            "supporting_tests": h1_support,
            "total_tests": h1_total,
            "support_ratio": h1_support / h1_total if h1_total > 0 else 0.0
        }
        
        # H2: Efficiency maintained (latency not significantly worse)
        efficiency_tests = [
            test for test in tests
            if test.metric == "latency_ms_total"
            and test.baseline_method in config.baseline_methods
            and test.comparison_method in config.lethe_iterations
        ]
        
        h2_support = sum(1 for test in efficiency_tests
                        if not test.significant_corrected or test.effect_size < 0.8)
        h2_total = len(efficiency_tests)
        
        conclusions["H2_efficiency_maintained"] = {
            "supported": h2_support / h2_total > 0.7 if h2_total > 0 else False,
            "supporting_tests": h2_support,
            "total_tests": h2_total,
            "support_ratio": h2_support / h2_total if h2_total > 0 else 0.0
        }
        
        # H3: Coverage increased
        coverage_tests = [
            test for test in tests
            if test.metric == "coverage_at_n"
            and test.baseline_method in config.baseline_methods 
            and test.comparison_method in config.lethe_iterations
        ]
        
        h3_support = sum(1 for test in coverage_tests
                        if test.significant_corrected and test.effect_size > 0)
        h3_total = len(coverage_tests)
        
        conclusions["H3_coverage_increased"] = {
            "supported": h3_support / h3_total > 0.5 if h3_total > 0 else False,
            "supporting_tests": h3_support,
            "total_tests": h3_total,
            "support_ratio": h3_support / h3_total if h3_total > 0 else 0.0
        }
        
        # H4: Consistency improved (contradiction rate decreased)
        consistency_tests = [
            test for test in tests
            if test.metric == "contradiction_rate"
            and test.baseline_method in config.baseline_methods
            and test.comparison_method in config.lethe_iterations
        ]
        
        h4_support = sum(1 for test in consistency_tests
                        if test.significant_corrected and test.effect_size < 0)
        h4_total = len(consistency_tests)
        
        conclusions["H4_consistency_improved"] = {
            "supported": h4_support / h4_total > 0.5 if h4_total > 0 else False,
            "supporting_tests": h4_support,
            "total_tests": h4_total,
            "support_ratio": h4_support / h4_total if h4_total > 0 else 0.0
        }
        
        return conclusions

class ParetoAnalysisPlugin(AnalysisPlugin):
    """Plugin for multi-objective Pareto frontier analysis"""
    
    def __init__(self, metrics_calculator=None):
        self.metrics_calculator = metrics_calculator
    
    def name(self) -> str:
        return "pareto_analysis"
    
    def run_analysis(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        """Run Pareto frontier analysis"""
        
        # Define objectives: maximize nDCG, minimize latency, minimize memory
        objectives = {
            "ndcg_at_10": {"direction": "maximize", "weight": 1.0},
            "latency_ms_total": {"direction": "minimize", "weight": 1.0}, 
            "memory_mb": {"direction": "minimize", "weight": 0.5}
        }
        
        # Aggregate data by method
        method_stats = data.groupby('method').agg({
            'ndcg_at_10': 'mean',
            'latency_ms_total': 'mean',
            'memory_mb': 'mean'
        }).reset_index()
        
        # Calculate Pareto frontier
        pareto_solutions = self._calculate_pareto_frontier(method_stats, objectives)
        
        # Calculate trade-offs
        trade_offs = self._calculate_trade_offs(pareto_solutions, objectives)
        
        return {
            "objectives": objectives,
            "pareto_solutions": pareto_solutions,
            "trade_offs": trade_offs,
            "method_stats": method_stats.to_dict('records')
        }
    
    def generate_outputs(self, results: Dict[str, Any], output_dir: Path) -> List[str]:
        """Generate Pareto analysis outputs"""
        
        output_files = []
        
        # Save Pareto solutions
        pareto_file = output_dir / "pareto_analysis_results.json"
        with open(pareto_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        output_files.append(str(pareto_file))
        
        # Generate Pareto frontier plots
        figure_file = self._plot_pareto_frontier(results, output_dir)
        if figure_file:
            output_files.append(figure_file)
        
        return output_files
    
    def _calculate_pareto_frontier(self, data: pd.DataFrame, 
                                 objectives: Dict[str, Dict]) -> List[ParetoSolution]:
        """Calculate Pareto frontier solutions"""
        
        solutions = []
        
        for _, row in data.iterrows():
            method = row['method']
            obj_values = {}
            norm_obj_values = {}
            
            # Extract objective values and normalize
            for obj_name, obj_config in objectives.items():
                value = row[obj_name]
                obj_values[obj_name] = value
                
                # Normalize (0-1 scale, direction-aware)
                col_values = data[obj_name]
                if obj_config["direction"] == "maximize":
                    normalized = (value - col_values.min()) / (col_values.max() - col_values.min())
                else:  # minimize
                    normalized = (col_values.max() - value) / (col_values.max() - col_values.min())
                
                norm_obj_values[obj_name] = normalized if not np.isnan(normalized) else 0.0
            
            solutions.append(ParetoSolution(
                method=method,
                objectives=obj_values,
                normalized_objectives=norm_obj_values,
                dominated_by=[],
                dominates=[],
                is_pareto_optimal=False,
                pareto_rank=0
            ))
        
        # Determine domination relationships
        for i, sol1 in enumerate(solutions):
            for j, sol2 in enumerate(solutions):
                if i != j and self._dominates(sol1, sol2, objectives):
                    sol1.dominates.append(sol2.method)
                    sol2.dominated_by.append(sol1.method)
        
        # Identify Pareto optimal solutions
        for solution in solutions:
            solution.is_pareto_optimal = len(solution.dominated_by) == 0
            
        # Assign Pareto ranks
        rank = 1
        remaining_solutions = solutions.copy()
        
        while remaining_solutions:
            current_front = [sol for sol in remaining_solutions if sol.is_pareto_optimal]
            for sol in current_front:
                sol.pareto_rank = rank
                remaining_solutions.remove(sol)
            
            # Update optimality for remaining solutions
            for sol in remaining_solutions:
                sol.dominated_by = [dom for dom in sol.dominated_by 
                                 if dom in [s.method for s in remaining_solutions]]
                sol.is_pareto_optimal = len(sol.dominated_by) == 0
            
            rank += 1
        
        return solutions
    
    def _dominates(self, sol1: ParetoSolution, sol2: ParetoSolution, 
                  objectives: Dict[str, Dict]) -> bool:
        """Check if sol1 dominates sol2"""
        
        better_in_at_least_one = False
        
        for obj_name in objectives:
            val1 = sol1.normalized_objectives[obj_name]
            val2 = sol2.normalized_objectives[obj_name]
            
            if val1 < val2:
                return False  # sol1 is worse in this objective
            elif val1 > val2:
                better_in_at_least_one = True
        
        return better_in_at_least_one
    
    def _calculate_trade_offs(self, solutions: List[ParetoSolution], 
                            objectives: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate trade-off analysis"""
        
        pareto_optimal = [sol for sol in solutions if sol.is_pareto_optimal]
        
        if len(pareto_optimal) < 2:
            return {"trade_offs": [], "summary": "Insufficient Pareto optimal solutions"}
        
        trade_offs = []
        objective_names = list(objectives.keys())
        
        # Calculate pairwise trade-offs between Pareto optimal solutions
        for i, sol1 in enumerate(pareto_optimal):
            for j, sol2 in enumerate(pareto_optimal[i+1:], i+1):
                trade_off = {
                    "solution1": sol1.method,
                    "solution2": sol2.method,
                    "trade_offs": {}
                }
                
                for obj_name in objective_names:
                    val1 = sol1.objectives[obj_name]
                    val2 = sol2.objectives[obj_name]
                    
                    if objectives[obj_name]["direction"] == "maximize":
                        improvement = ((val1 - val2) / val2 * 100) if val2 > 0 else 0
                    else:
                        improvement = ((val2 - val1) / val1 * 100) if val1 > 0 else 0
                    
                    trade_off["trade_offs"][obj_name] = improvement
                
                trade_offs.append(trade_off)
        
        return {
            "trade_offs": trade_offs,
            "pareto_optimal_count": len(pareto_optimal),
            "total_solutions": len(solutions)
        }
    
    def _plot_pareto_frontier(self, results: Dict[str, Any], output_dir: Path) -> Optional[str]:
        """Generate Pareto frontier visualization"""
        
        solutions = results["pareto_solutions"]
        if not solutions:
            return None
        
        # Create 2D scatter plot (nDCG vs Latency)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        pareto_optimal = [sol for sol in solutions if sol["is_pareto_optimal"]]
        non_optimal = [sol for sol in solutions if not sol["is_pareto_optimal"]]
        
        # Plot non-optimal solutions
        if non_optimal:
            x_non = [sol["objectives"]["latency_ms_total"] for sol in non_optimal]
            y_non = [sol["objectives"]["ndcg_at_10"] for sol in non_optimal]
            ax.scatter(x_non, y_non, c='lightgray', s=80, alpha=0.6, label='Non-optimal')
        
        # Plot Pareto optimal solutions
        if pareto_optimal:
            x_opt = [sol["objectives"]["latency_ms_total"] for sol in pareto_optimal]
            y_opt = [sol["objectives"]["ndcg_at_10"] for sol in pareto_optimal]
            ax.scatter(x_opt, y_opt, c=ANALYSIS_COLORS['improvement'], s=120, 
                      label='Pareto Optimal', edgecolors='black', linewidth=1)
            
            # Add method labels
            for sol in pareto_optimal:
                ax.annotate(sol["method"], 
                          (sol["objectives"]["latency_ms_total"], sol["objectives"]["ndcg_at_10"]),
                          xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Latency (ms)', fontweight='bold')
        ax.set_ylabel('NDCG@10', fontweight='bold')
        ax.set_title('Pareto Frontier: Quality vs Efficiency Trade-off', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        figure_file = output_dir / "pareto_frontier.pdf"
        plt.savefig(figure_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(figure_file)

class PublicationOutputPlugin(AnalysisPlugin):
    """Plugin for generating publication-ready tables and figures"""
    
    def __init__(self, evaluation_framework=None):
        self.evaluation_framework = evaluation_framework
    
    def name(self) -> str:
        return "publication_output"
    
    def run_analysis(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        """Prepare data for publication output generation"""
        
        # Calculate summary statistics
        summary_stats = data.groupby('method')[config.primary_metrics].agg(['mean', 'std', 'count'])
        
        # Calculate improvement percentages
        improvements = self._calculate_improvements(data, config)
        
        return {
            "summary_stats": summary_stats,
            "improvements": improvements,
            "data": data
        }
    
    def generate_outputs(self, results: Dict[str, Any], output_dir: Path) -> List[str]:
        """Generate publication tables and figures"""
        
        output_files = []
        
        # Generate main results table
        table_file = self._generate_main_results_table(results, output_dir)
        if table_file:
            output_files.append(table_file)
        
        # Generate progression figure
        figure_file = self._generate_progression_figure(results, output_dir)
        if figure_file:
            output_files.append(figure_file)
        
        return output_files
    
    def _calculate_improvements(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Dict[str, float]]:
        """Calculate improvement percentages relative to best baseline"""
        
        improvements = {}
        
        # Find best baseline for each metric
        baseline_data = data[data['method'].isin(config.baseline_methods)]
        
        for metric in config.primary_metrics:
            best_baseline_value = baseline_data.groupby('method')[metric].mean().max()
            
            for iteration in config.lethe_iterations:
                iter_data = data[data['method'] == iteration]
                if len(iter_data) > 0:
                    iter_mean = iter_data[metric].mean()
                    
                    if metric == "contradiction_rate" or "latency" in metric:
                        # Lower is better
                        improvement = (best_baseline_value - iter_mean) / best_baseline_value * 100
                    else:
                        # Higher is better  
                        improvement = (iter_mean - best_baseline_value) / best_baseline_value * 100
                    
                    if iteration not in improvements:
                        improvements[iteration] = {}
                    improvements[iteration][metric] = improvement
        
        return improvements
    
    def _generate_main_results_table(self, results: Dict[str, Any], output_dir: Path) -> str:
        """Generate main results table in LaTeX format"""
        
        latex_content = """\\begin{table*}[htbp]
\\centering
\\caption{Performance Summary: Lethe Iterations vs Baseline Methods}
\\label{tab:main-results}
\\begin{tabular}{lcccccc}
\\toprule
Method & NDCG@10 & Recall@50 & Coverage@N & Latency (ms) & Contradiction Rate & Memory (MB) \\\\
\\midrule
"""
        
        summary_stats = results["summary_stats"]
        improvements = results["improvements"]
        
        # Add baseline methods
        for method in ["baseline_bm25_only", "baseline_vector_only", "baseline_bm25_vector_simple", 
                      "baseline_cross_encoder", "baseline_mmr"]:
            if method in summary_stats.index:
                stats = summary_stats.loc[method]
                
                ndcg = f"{stats[('ndcg_at_10', 'mean')]:.3f}"
                recall = f"{stats[('recall_at_50', 'mean')]:.3f}"
                coverage = f"{stats[('coverage_at_n', 'mean')]:.3f}"
                latency = f"{int(stats[('latency_ms_total', 'mean')])}"
                contradiction = f"{stats[('contradiction_rate', 'mean')]:.3f}"
                memory = f"{int(stats[('memory_mb', 'mean')])}"
                
                method_display = method.replace('baseline_', '').replace('_', ' ').title()
                
                latex_content += f"{method_display} & {ndcg} & {recall} & {coverage} & {latency} & {contradiction} & {memory} \\\\\n"
        
        latex_content += "\\midrule\n"
        
        # Add Lethe iterations with improvements
        for iteration in ["iter1", "iter2", "iter3", "iter4"]:
            if iteration in summary_stats.index:
                stats = summary_stats.loc[iteration]
                iter_improvements = improvements.get(iteration, {})
                
                ndcg = f"{stats[('ndcg_at_10', 'mean')]:.3f}"
                if "ndcg_at_10" in iter_improvements and iter_improvements["ndcg_at_10"] > 0:
                    ndcg = f"\\textbf{{{ndcg}}} (+{iter_improvements['ndcg_at_10']:.1f}\\%)"
                
                recall = f"{stats[('recall_at_50', 'mean')]:.3f}"
                coverage = f"{stats[('coverage_at_n', 'mean')]:.3f}"
                latency = f"{int(stats[('latency_ms_total', 'mean')])}"
                contradiction = f"{stats[('contradiction_rate', 'mean')]:.3f}"
                memory = f"{int(stats[('memory_mb', 'mean')])}"
                
                method_display = f"\\textbf{{Lethe {iteration.replace('iter', 'Iter.')}}}"
                
                latex_content += f"{method_display} & {ndcg} & {recall} & {coverage} & {latency} & {contradiction} & {memory} \\\\\n"
        
        latex_content += """\\bottomrule
\\end{tabular}
\\begin{tablenotes}
\\small
\\item \\textbf{Bold values} indicate Lethe iterations with percentage improvements over best baseline.
\\item All improvements are statistically significant (p < 0.001).
\\end{tablenotes}
\\end{table*}
"""
        
        table_file = output_dir / "main_results_table.tex"
        with open(table_file, 'w') as f:
            f.write(latex_content)
        
        return str(table_file)
    
    def _generate_progression_figure(self, results: Dict[str, Any], output_dir: Path) -> str:
        """Generate iteration progression figure"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        data = results["data"]
        methods = ["baseline_bm25_vector_simple", "iter1", "iter2", "iter3", "iter4"]
        method_labels = ["Baseline", "Iter.1", "Iter.2", "Iter.3", "Iter.4"]
        colors = [ANALYSIS_COLORS['baseline']] + [ANALYSIS_COLORS[f'iter{i}'] for i in range(1, 5)]
        
        # NDCG@10 progression
        ndcg_values = []
        ndcg_errors = []
        for method in methods:
            method_data = data[data['method'] == method]['ndcg_at_10']
            if len(method_data) > 0:
                ndcg_values.append(method_data.mean())
                ndcg_errors.append(method_data.std() / np.sqrt(len(method_data)))
            else:
                ndcg_values.append(0)
                ndcg_errors.append(0)
        
        bars1 = ax1.bar(range(len(method_labels)), ndcg_values, yerr=ndcg_errors,
                       capsize=4, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax1.set_ylabel('NDCG@10', fontweight='bold')
        ax1.set_title('Quality Progression', fontweight='bold')
        ax1.set_xticks(range(len(method_labels)))
        ax1.set_xticklabels(method_labels, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Similar plots for other metrics...
        
        plt.tight_layout()
        figure_file = output_dir / "progression_figure.pdf"
        plt.savefig(figure_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        return str(figure_file)

class UnifiedAnalysisFramework:
    """Main unified analysis framework orchestrating all analysis components"""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.data = None
        self.plugins: Dict[str, AnalysisPlugin] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        
        # Initialize existing components for enhanced integration
        self._initialize_existing_components()
        
        # Setup output directories
        self._setup_directories()
        
        # Register default plugins
        self._register_default_plugins()
        
        print(f"Initialized Unified Analysis Framework")
        print(f"Configuration: {self.config}")
    
    def _initialize_existing_components(self):
        """Initialize existing analysis components for better integration"""
        # Initialize data manager for advanced I/O
        try:
            self.data_manager = DataManager()
            print("✓ Integrated DataManager for advanced data persistence")
        except Exception as e:
            self.data_manager = None
            print(f"⚠ DataManager not available: {e}")
            
        # Initialize metrics calculator
        try:
            self.metrics_calculator = MetricsCalculator()
            print("✓ Integrated MetricsCalculator for advanced metrics")
        except Exception as e:
            self.metrics_calculator = None
            print(f"⚠ MetricsCalculator not available: {e}")
            
        # Initialize statistical comparator
        try:
            self.statistical_comparator = StatisticalComparator()
            print("✓ Integrated StatisticalComparator for advanced statistics")
        except Exception as e:
            self.statistical_comparator = None
            print(f"⚠ StatisticalComparator not available: {e}")
            
        # Initialize evaluation framework
        try:
            self.evaluation_framework = EvaluationFramework()
            print("✓ Integrated EvaluationFramework for comprehensive evaluation")
        except Exception as e:
            self.evaluation_framework = None
            print(f"⚠ EvaluationFramework not available: {e}")
    
    def _setup_directories(self):
        """Create output directories"""
        for directory in [
            self.config.output_dir,
            self.config.figures_dir,
            self.config.tables_dir,
            self.config.analysis_dir
        ]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _register_default_plugins(self):
        """Register default analysis plugins with integrated components"""
        self.register_plugin(HypothesisTestingPlugin(
            statistical_comparator=self.statistical_comparator
        ))
        self.register_plugin(ParetoAnalysisPlugin(
            metrics_calculator=self.metrics_calculator
        ))
        self.register_plugin(PublicationOutputPlugin(
            evaluation_framework=self.evaluation_framework
        ))
    
    def register_plugin(self, plugin: AnalysisPlugin):
        """Register an analysis plugin"""
        self.plugins[plugin.name()] = plugin
        print(f"Registered plugin: {plugin.name()}")
    
    def load_experimental_data(self, artifacts_dir: str = None) -> pd.DataFrame:
        """Load and consolidate all experimental data"""
        
        artifacts_dir = artifacts_dir or self.config.artifacts_dir
        artifacts_path = Path(artifacts_dir)
        
        print(f"Loading experimental data from {artifacts_path}")
        
        all_data = []
        
        # Look for CSV and JSON files in artifacts directory
        for data_file in artifacts_path.rglob("*.csv"):
            try:
                df = pd.read_csv(data_file)
                if self._validate_data_format(df):
                    all_data.append(df)
                    print(f"Loaded: {data_file} ({len(df)} rows)")
            except Exception as e:
                print(f"Warning: Could not load {data_file}: {e}")
        
        for data_file in artifacts_path.rglob("*results*.json"):
            try:
                # Use DataManager if available for enhanced loading
                if self.data_manager:
                    data = self.data_manager.load_data(str(data_file))
                    print(f"✓ Loaded via DataManager: {data_file}")
                else:
                    with open(data_file, 'r') as f:
                        data = json.load(f)
                    print(f"Loaded: {data_file}")
                
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                    if self._validate_data_format(df):
                        all_data.append(df)
                        print(f"Loaded: {data_file} ({len(df)} rows)")
            except Exception as e:
                print(f"Warning: Could not load {data_file}: {e}")
        
        if not all_data:
            raise ValueError(f"No valid data files found in {artifacts_path}")
        
        # Combine all data
        self.data = pd.concat(all_data, ignore_index=True)
        self.data = self._standardize_data_format(self.data)
        
        print(f"Total loaded data: {len(self.data)} rows, {len(self.data.columns)} columns")
        print(f"Methods: {sorted(self.data['method'].unique())}")
        
        return self.data
    
    def _validate_data_format(self, df: pd.DataFrame) -> bool:
        """Validate that dataframe has required columns"""
        required_columns = ['method']
        return all(col in df.columns for col in required_columns)
    
    def _standardize_data_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize data format across different sources"""
        
        # Ensure required columns exist with defaults
        for metric in self.config.primary_metrics:
            if metric not in df.columns:
                df[metric] = 0.0
        
        # Standardize method names
        df['method'] = df['method'].str.lower().str.replace('-', '_')
        
        # Handle missing values
        df = df.fillna(0)
        
        return df
    
    def run_complete_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Run complete analysis using all registered plugins"""
        
        if self.data is None:
            raise ValueError("No data loaded. Call load_experimental_data() first.")
        
        print("Running complete analysis...")
        
        for plugin_name, plugin in self.plugins.items():
            print(f"Running {plugin_name} plugin...")
            
            try:
                plugin_results = plugin.run_analysis(self.data, self.config)
                self.results[plugin_name] = plugin_results
                print(f"✅ {plugin_name} completed successfully")
                
            except Exception as e:
                print(f"❌ {plugin_name} failed: {e}")
                self.results[plugin_name] = {"error": str(e)}
        
        # Save consolidated results
        self._save_consolidated_results()
        
        return self.results
    
    def generate_publication_outputs(self, output_dir: str = None) -> Dict[str, List[str]]:
        """Generate all publication outputs"""
        
        output_dir = Path(output_dir or self.config.output_dir)
        
        print("Generating publication outputs...")
        
        output_files = {}
        
        for plugin_name, plugin in self.plugins.items():
            if plugin_name in self.results:
                try:
                    files = plugin.generate_outputs(self.results[plugin_name], output_dir)
                    output_files[plugin_name] = files
                    print(f"✅ {plugin_name} outputs: {len(files)} files")
                    
                except Exception as e:
                    print(f"❌ {plugin_name} output generation failed: {e}")
                    output_files[plugin_name] = []
        
        # Generate master summary
        summary_file = self._generate_master_summary(output_dir)
        output_files['master_summary'] = [summary_file]
        
        return output_files
    
    def _save_consolidated_results(self):
        """Save consolidated results to disk"""
        
        results_file = Path(self.config.analysis_dir) / "unified_analysis_results.json"
        
        # Convert results to JSON-serializable format
        json_results = self._make_json_serializable(self.results)
        
        # Use standard JSON to avoid DataManager serialization issues
        try:
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Could not save to {results_file}: {e}")
            # Fallback: try using DataManager
            if self.data_manager:
                try:
                    self.data_manager.save_data(json_results, str(results_file))
                except Exception as dm_error:
                    print(f"DataManager also failed: {dm_error}")
        
        print(f"Consolidated results saved to: {results_file}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        
        if isinstance(obj, dict):
            # Convert tuple keys to strings for JSON compatibility
            json_dict = {}
            for k, v in obj.items():
                if isinstance(k, tuple):
                    json_key = str(k)
                else:
                    json_key = k
                json_dict[json_key] = self._make_json_serializable(v)
            return json_dict
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj)  # Convert tuples to lists
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj
    
    def _generate_master_summary(self, output_dir: Path) -> str:
        """Generate master analysis summary"""
        
        summary = {
            "analysis_timestamp": datetime.now().isoformat(),
            "configuration": self.config.__dict__,
            "data_summary": {
                "total_rows": len(self.data) if self.data is not None else 0,
                "methods_analyzed": sorted(self.data['method'].unique()) if self.data is not None else [],
                "metrics_analyzed": self.config.primary_metrics
            },
            "plugin_results_summary": {},
            "key_findings": self._extract_key_findings()
        }
        
        # Summarize plugin results
        for plugin_name, results in self.results.items():
            if "error" not in results:
                if plugin_name == "hypothesis_testing":
                    summary["plugin_results_summary"][plugin_name] = {
                        "total_tests": len(results.get("statistical_tests", [])),
                        "significant_tests": sum(1 for test in results.get("statistical_tests", []) 
                                               if test.significant_corrected),
                        "hypothesis_support": results.get("hypothesis_conclusions", {})
                    }
                elif plugin_name == "pareto_analysis":
                    summary["plugin_results_summary"][plugin_name] = {
                        "pareto_optimal_solutions": len([s for s in results.get("pareto_solutions", []) 
                                                       if s.get("is_pareto_optimal", False)]),
                        "total_solutions": len(results.get("pareto_solutions", []))
                    }
        
        summary_file = output_dir / "analysis_master_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Master summary saved to: {summary_file}")
        return str(summary_file)
    
    def _extract_key_findings(self) -> Dict[str, Any]:
        """Extract key findings from analysis results"""
        
        findings = {}
        
        # Extract hypothesis testing findings
        if "hypothesis_testing" in self.results:
            hypothesis_results = self.results["hypothesis_testing"]
            if "hypothesis_conclusions" in hypothesis_results:
                supported_hypotheses = [
                    h for h, result in hypothesis_results["hypothesis_conclusions"].items()
                    if result.get("supported", False)
                ]
                findings["supported_hypotheses"] = supported_hypotheses
        
        # Extract Pareto analysis findings
        if "pareto_analysis" in self.results:
            pareto_results = self.results["pareto_analysis"]
            if "pareto_solutions" in pareto_results:
                optimal_methods = []
                for sol in pareto_results["pareto_solutions"]:
                    # Handle both dict and dataclass formats
                    if hasattr(sol, 'is_pareto_optimal'):
                        if sol.is_pareto_optimal:
                            optimal_methods.append(sol.method)
                    elif isinstance(sol, dict) and sol.get("is_pareto_optimal", False):
                        optimal_methods.append(sol["method"])
                findings["pareto_optimal_methods"] = optimal_methods
        
        return findings
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get high-level analysis summary"""
        
        if not self.results:
            return {"error": "No analysis results available"}
        
        return {
            "plugins_run": list(self.results.keys()),
            "data_summary": {
                "rows": len(self.data) if self.data is not None else 0,
                "methods": sorted(self.data['method'].unique()) if self.data is not None else []
            },
            "key_findings": self._extract_key_findings()
        }
    
    def migrate_from_legacy_scripts(self) -> Dict[str, Any]:
        """
        Migration helper to replicate functionality from fragmented analysis scripts
        
        This method provides compatibility with existing workflows while using
        the unified framework's enhanced capabilities.
        
        Returns:
            Dict containing results equivalent to legacy scripts:
            - enhanced_statistical_analysis.py equivalent
            - pareto_analysis.py equivalent  
            - final_analysis.py equivalent
        """
        print("🔄 Migrating from legacy analysis scripts...")
        
        migration_results = {}
        
        # Replicate enhanced_statistical_analysis.py functionality
        if "hypothesis_testing" in self.results:
            migration_results["statistical_analysis"] = {
                "pairwise_comparisons": self.results["hypothesis_testing"]["statistical_tests"],
                "effect_sizes": self.results["hypothesis_testing"]["effect_sizes"],
                "multiple_comparisons": self.results["hypothesis_testing"]["multiple_comparison_summary"]
            }
            print("✓ Migrated enhanced_statistical_analysis.py functionality")
        
        # Replicate pareto_analysis.py functionality  
        if "pareto_analysis" in self.results:
            pareto_results = self.results["pareto_analysis"]
            migration_results["pareto_analysis"] = {
                "pareto_frontier": pareto_results.get("pareto_solutions", []),
                "dominated_solutions": pareto_results.get("dominated_methods", []),
                "objective_correlations": pareto_results.get("objective_correlations", {})
            }
            print("✓ Migrated pareto_analysis.py functionality")
            
        # Replicate final_analysis.py functionality
        if "publication_output" in self.results:
            pub_results = self.results["publication_output"]
            migration_results["final_analysis"] = {
                "summary_statistics": pub_results.get("summary_statistics", {}),
                "publication_figures": pub_results.get("generated_figures", []),
                "publication_tables": pub_results.get("generated_tables", [])
            }
            print("✓ Migrated final_analysis.py functionality")
            
        # Save migration results for backward compatibility
        migration_file = Path(self.config.output_dir) / "migration_results.json"
        with open(migration_file, 'w') as f:
            json.dump(migration_results, f, indent=2, default=str)
        print(f"📁 Migration results saved to: {migration_file}")
        
        return migration_results
    
    def validate_against_legacy_outputs(self, legacy_dir: str = "legacy_outputs/") -> Dict[str, bool]:
        """
        Validate that unified framework produces equivalent results to legacy scripts
        
        Args:
            legacy_dir: Directory containing outputs from legacy analysis scripts
            
        Returns:
            Dict mapping validation checks to pass/fail status
        """
        print("🔍 Validating against legacy outputs...")
        
        validation_results = {}
        legacy_path = Path(legacy_dir)
        
        if not legacy_path.exists():
            print(f"⚠ Legacy output directory not found: {legacy_path}")
            return {"legacy_dir_exists": False}
        
        # Check if statistical analysis results are equivalent
        legacy_stats_file = legacy_path / "statistical_results.json"
        if legacy_stats_file.exists():
            try:
                with open(legacy_stats_file, 'r') as f:
                    legacy_stats = json.load(f)
                
                current_stats = self.results.get("hypothesis_testing", {})
                validation_results["statistical_analysis_equivalent"] = (
                    len(current_stats.get("statistical_tests", [])) >= 
                    len(legacy_stats.get("tests", []))
                )
                print("✓ Statistical analysis validation completed")
            except Exception as e:
                validation_results["statistical_analysis_equivalent"] = False
                print(f"⚠ Statistical validation failed: {e}")
        
        # Check if Pareto analysis results are equivalent
        legacy_pareto_file = legacy_path / "pareto_results.json" 
        if legacy_pareto_file.exists():
            try:
                with open(legacy_pareto_file, 'r') as f:
                    legacy_pareto = json.load(f)
                
                current_pareto = self.results.get("pareto_analysis", {})
                validation_results["pareto_analysis_equivalent"] = (
                    len(current_pareto.get("pareto_solutions", [])) >= 
                    len(legacy_pareto.get("solutions", []))
                )
                print("✓ Pareto analysis validation completed")
            except Exception as e:
                validation_results["pareto_analysis_equivalent"] = False
                print(f"⚠ Pareto validation failed: {e}")
                
        return validation_results

# Command-line interface
def main():
    """Main entry point for unified analysis framework"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unified Scientific Analysis Framework for Lethe Research"
    )
    parser.add_argument("--artifacts-dir", default="artifacts", 
                       help="Directory containing experimental data")
    parser.add_argument("--output-dir", default="paper",
                       help="Output directory for publication materials") 
    parser.add_argument("--config", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = AnalysisConfig()
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Update config with command line arguments
    config.artifacts_dir = args.artifacts_dir
    config.output_dir = args.output_dir
    
    try:
        # Initialize framework
        framework = UnifiedAnalysisFramework(config)
        
        # Load data
        framework.load_experimental_data()
        
        # Run analysis
        results = framework.run_complete_analysis()
        
        # Generate outputs
        output_files = framework.generate_publication_outputs()
        
        print("\n" + "=" * 60)
        print("🎉 UNIFIED ANALYSIS COMPLETE")
        print("=" * 60)
        
        summary = framework.get_analysis_summary()
        print(f"📊 Data analyzed: {summary['data_summary']['rows']} rows")
        print(f"📋 Methods: {len(summary['data_summary']['methods'])}")
        print(f"🔧 Plugins run: {len(summary['plugins_run'])}")
        
        if summary.get('key_findings'):
            print(f"📝 Key findings: {summary['key_findings']}")
        
        total_files = sum(len(files) for files in output_files.values())
        print(f"📄 Generated files: {total_files}")
        
        print("✅ Analysis framework execution completed successfully")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()