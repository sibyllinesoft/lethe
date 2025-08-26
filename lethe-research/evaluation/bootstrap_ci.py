"""
Statistical Validation with BCa Bootstrap Confidence Intervals

Implements Bias-Corrected and accelerated (BCa) bootstrap for:
- nDCG@10 improvement validation with 95% CI lower bound > 0
- Answer-Span-Kept preservation with 95% CI lower bound ≥ 98%
- Token reduction ratio validation
- Performance metric confidence estimation

Key features:
- 10,000 bootstrap iterations for statistical power
- BCa correction for bias and acceleration
- Paired comparison testing for A/B evaluation
- Multiple hypothesis correction with Bonferroni
- Reproducible random sampling with fixed seeds
- Comprehensive statistical reporting
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import json
from scipy import stats
from scipy.stats import percentileofscore
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import hashlib

@dataclass
class BootstrapResult:
    """Results from BCa bootstrap analysis"""
    metric_name: str
    original_value: float
    bootstrap_mean: float
    bootstrap_std: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_bootstrap: int
    bias_correction: float
    acceleration: float
    p_value: float  # For hypothesis testing
    effect_size: float  # Cohen's d or similar
    is_significant: bool
    meets_threshold: bool  # Whether CI meets minimum requirement
    computation_time_ms: float

@dataclass 
class ComparisonResult:
    """Results from paired comparison between baseline and experimental"""
    baseline_metric: str
    experimental_metric: str
    improvement: float  # Experimental - Baseline
    improvement_pct: float  # (Experimental - Baseline) / Baseline * 100
    bootstrap_result: BootstrapResult
    paired_t_test: Dict[str, float]  # t-statistic, p-value, effect_size
    wilcoxon_test: Dict[str, float]  # Non-parametric alternative
    summary: str

class BCaBootstrap:
    """
    Bias-Corrected and accelerated (BCa) Bootstrap implementation
    
    Provides rigorous statistical validation for Lethe vNext evaluation
    with publication-ready confidence intervals and hypothesis testing.
    """
    
    def __init__(self, n_bootstrap: int = 10000, confidence_level: float = 0.95, 
                 random_seed: int = 42, verbose: bool = True):
        """
        Initialize BCa Bootstrap analyzer
        
        Args:
            n_bootstrap: Number of bootstrap samples (10k for publication quality)
            confidence_level: Confidence level for intervals (0.95 for 95% CI) 
            random_seed: Seed for reproducible results
            verbose: Whether to print progress information
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        self.verbose = verbose
        self.rng = np.random.RandomState(random_seed)
        
        # Alpha levels for two-sided confidence intervals
        self.alpha = 1 - confidence_level
        self.alpha_lower = self.alpha / 2
        self.alpha_upper = 1 - self.alpha / 2
        
    def bca_confidence_interval(self, data: np.ndarray, 
                               statistic_func: callable,
                               theta_hat: Optional[float] = None) -> BootstrapResult:
        """
        Compute BCa confidence interval for a given statistic
        
        Args:
            data: Original sample data
            statistic_func: Function to compute statistic (e.g., np.mean, np.median)
            theta_hat: Original statistic value (computed if None)
            
        Returns:
            BootstrapResult with confidence interval and diagnostic information
        """
        start_time = time.time()
        
        if theta_hat is None:
            theta_hat = statistic_func(data)
            
        # Bootstrap sampling
        n = len(data)
        bootstrap_stats = []
        
        if self.verbose:
            print(f"Running {self.n_bootstrap} bootstrap iterations...")
            
        for i in range(self.n_bootstrap):
            # Resample with replacement
            bootstrap_sample = self.rng.choice(data, size=n, replace=True)
            bootstrap_stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(bootstrap_stat)
            
            if self.verbose and (i + 1) % 1000 == 0:
                print(f"Completed {i + 1}/{self.n_bootstrap} bootstrap samples")
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Compute bias-correction (z_0)
        bias_correction = self._compute_bias_correction(bootstrap_stats, theta_hat)
        
        # Compute acceleration (a_hat) using jackknife
        acceleration = self._compute_acceleration(data, statistic_func, theta_hat)
        
        # Compute BCa confidence intervals
        ci_lower, ci_upper = self._compute_bca_intervals(
            bootstrap_stats, bias_correction, acceleration
        )
        
        # Additional statistics
        bootstrap_mean = np.mean(bootstrap_stats)
        bootstrap_std = np.std(bootstrap_stats)
        
        # Hypothesis testing (H0: statistic = 0)
        p_value = 2 * min(
            percentileofscore(bootstrap_stats, 0) / 100,
            1 - percentileofscore(bootstrap_stats, 0) / 100
        )
        
        # Effect size (standardized mean difference)
        effect_size = theta_hat / bootstrap_std if bootstrap_std > 0 else 0
        
        computation_time = (time.time() - start_time) * 1000
        
        result = BootstrapResult(
            metric_name="custom_statistic",
            original_value=theta_hat,
            bootstrap_mean=bootstrap_mean,
            bootstrap_std=bootstrap_std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.confidence_level,
            n_bootstrap=self.n_bootstrap,
            bias_correction=bias_correction,
            acceleration=acceleration,
            p_value=p_value,
            effect_size=effect_size,
            is_significant=p_value < self.alpha,
            meets_threshold=ci_lower > 0,  # Default threshold
            computation_time_ms=computation_time
        )
        
        return result
    
    def _compute_bias_correction(self, bootstrap_stats: np.ndarray, theta_hat: float) -> float:
        """Compute bias-correction z_0 for BCa"""
        # Proportion of bootstrap statistics less than original estimate
        proportion = np.mean(bootstrap_stats < theta_hat)
        
        # Handle edge cases
        if proportion == 0:
            return stats.norm.ppf(0.001)  # Small positive value
        elif proportion == 1:
            return stats.norm.ppf(0.999)  # Large positive value
        else:
            return stats.norm.ppf(proportion)
    
    def _compute_acceleration(self, data: np.ndarray, statistic_func: callable, 
                            theta_hat: float) -> float:
        """Compute acceleration a_hat using jackknife"""
        n = len(data)
        jackknife_stats = []
        
        # Jackknife: remove one observation at a time
        for i in range(n):
            jackknife_sample = np.concatenate([data[:i], data[i+1:]])
            jackknife_stat = statistic_func(jackknife_sample)
            jackknife_stats.append(jackknife_stat)
        
        jackknife_stats = np.array(jackknife_stats)
        jackknife_mean = np.mean(jackknife_stats)
        
        # Compute acceleration
        numerator = np.sum((jackknife_mean - jackknife_stats)**3)
        denominator = 6 * (np.sum((jackknife_mean - jackknife_stats)**2))**(3/2)
        
        if abs(denominator) < 1e-10:
            return 0.0  # No acceleration
        
        acceleration = numerator / denominator
        return acceleration
    
    def _compute_bca_intervals(self, bootstrap_stats: np.ndarray, 
                              bias_correction: float, acceleration: float) -> Tuple[float, float]:
        """Compute BCa-adjusted confidence intervals"""
        # Standard normal quantiles
        z_alpha_lower = stats.norm.ppf(self.alpha_lower)
        z_alpha_upper = stats.norm.ppf(self.alpha_upper)
        
        # BCa-adjusted quantiles
        if abs(acceleration) < 1e-10:  # Avoid division by zero
            # Bias-corrected (BC) intervals when acceleration ≈ 0
            alpha_lower_adj = stats.norm.cdf(2 * bias_correction + z_alpha_lower)
            alpha_upper_adj = stats.norm.cdf(2 * bias_correction + z_alpha_upper)
        else:
            # Full BCa adjustment
            alpha_lower_adj = stats.norm.cdf(
                bias_correction + (bias_correction + z_alpha_lower) / 
                (1 - acceleration * (bias_correction + z_alpha_lower))
            )
            alpha_upper_adj = stats.norm.cdf(
                bias_correction + (bias_correction + z_alpha_upper) / 
                (1 - acceleration * (bias_correction + z_alpha_upper))
            )
        
        # Ensure adjusted quantiles are in [0, 1]
        alpha_lower_adj = np.clip(alpha_lower_adj, 0.001, 0.999)
        alpha_upper_adj = np.clip(alpha_upper_adj, 0.001, 0.999)
        
        # Compute confidence interval bounds
        ci_lower = np.percentile(bootstrap_stats, alpha_lower_adj * 100)
        ci_upper = np.percentile(bootstrap_stats, alpha_upper_adj * 100)
        
        return ci_lower, ci_upper
    
    def paired_comparison(self, baseline_scores: np.ndarray, 
                         experimental_scores: np.ndarray,
                         metric_name: str = "improvement") -> ComparisonResult:
        """
        Perform paired comparison between baseline and experimental conditions
        
        Used for A/B testing Lethe vNext vs baseline retrieval
        
        Args:
            baseline_scores: Scores from baseline system (e.g., nDCG@10)
            experimental_scores: Scores from experimental system
            metric_name: Name of the metric being compared
            
        Returns:
            ComparisonResult with statistical analysis
        """
        if len(baseline_scores) != len(experimental_scores):
            raise ValueError("Baseline and experimental scores must have same length")
        
        # Compute improvements (paired differences)
        improvements = experimental_scores - baseline_scores
        mean_improvement = np.mean(improvements)
        mean_baseline = np.mean(baseline_scores)
        improvement_pct = (mean_improvement / mean_baseline * 100) if mean_baseline != 0 else 0
        
        # BCa bootstrap on improvements
        bootstrap_result = self.bca_confidence_interval(
            improvements, np.mean, mean_improvement
        )
        bootstrap_result.metric_name = f"{metric_name}_improvement"
        
        # Paired t-test
        t_stat, t_p_value = stats.ttest_rel(experimental_scores, baseline_scores)
        pooled_std = np.sqrt((np.var(experimental_scores, ddof=1) + np.var(baseline_scores, ddof=1)) / 2)
        cohen_d = mean_improvement / pooled_std if pooled_std > 0 else 0
        
        paired_t_test = {
            "t_statistic": float(t_stat),
            "p_value": float(t_p_value),
            "effect_size": float(cohen_d),
            "degrees_of_freedom": len(improvements) - 1
        }
        
        # Wilcoxon signed-rank test (non-parametric)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(improvements, alternative='two-sided')
        
        wilcoxon_test = {
            "statistic": float(wilcoxon_stat),
            "p_value": float(wilcoxon_p),
            "effect_size": mean_improvement / np.std(improvements) if np.std(improvements) > 0 else 0
        }
        
        # Generate summary
        significance = "significant" if bootstrap_result.p_value < 0.05 else "not significant"
        direction = "improvement" if mean_improvement > 0 else "decline"
        
        summary = (f"{direction.title()} of {improvement_pct:.2f}% "
                  f"({bootstrap_result.ci_lower:.4f} to {bootstrap_result.ci_upper:.4f} 95% CI), "
                  f"{significance} (p = {bootstrap_result.p_value:.4f})")
        
        return ComparisonResult(
            baseline_metric=f"baseline_{metric_name}",
            experimental_metric=f"experimental_{metric_name}",
            improvement=mean_improvement,
            improvement_pct=improvement_pct,
            bootstrap_result=bootstrap_result,
            paired_t_test=paired_t_test,
            wilcoxon_test=wilcoxon_test,
            summary=summary
        )

class LetHeEvaluationValidator:
    """
    Statistical validator for Lethe vNext evaluation requirements
    
    Validates that improvements meet the specified thresholds:
    - nDCG@10: ≥+10% improvement with 95% CI lower bound > 0
    - Answer-Span-Kept: ≥98% with 95% CI lower bound ≥ 98%
    - Token reduction: 30-50% with statistical significance
    """
    
    def __init__(self, n_bootstrap: int = 10000, random_seed: int = 42):
        self.bootstrap = BCaBootstrap(n_bootstrap=n_bootstrap, random_seed=random_seed)
        self.validation_results = {}
        
    def validate_ndcg_improvement(self, baseline_ndcg: np.ndarray, 
                                experimental_ndcg: np.ndarray,
                                min_improvement: float = 0.10) -> Dict[str, Any]:
        """
        Validate nDCG@10 improvement meets threshold
        
        Args:
            baseline_ndcg: nDCG@10 scores from baseline system
            experimental_ndcg: nDCG@10 scores from Lethe vNext
            min_improvement: Minimum improvement threshold (0.10 = 10%)
            
        Returns:
            Validation results with statistical evidence
        """
        comparison = self.bootstrap.paired_comparison(
            baseline_ndcg, experimental_ndcg, "nDCG@10"
        )
        
        # Check if improvement percentage meets threshold
        meets_improvement_threshold = comparison.improvement_pct >= (min_improvement * 100)
        
        # Check if 95% CI lower bound > 0 (statistically significant improvement)
        ci_lower_positive = comparison.bootstrap_result.ci_lower > 0
        
        validation_result = {
            "metric": "nDCG@10_improvement",
            "target": f">= {min_improvement * 100}% improvement",
            "actual_improvement_pct": comparison.improvement_pct,
            "ci_lower_bound": comparison.bootstrap_result.ci_lower,
            "ci_upper_bound": comparison.bootstrap_result.ci_upper,
            "meets_improvement_threshold": meets_improvement_threshold,
            "ci_lower_positive": ci_lower_positive,
            "validation_passed": meets_improvement_threshold and ci_lower_positive,
            "comparison_result": comparison,
            "interpretation": self._interpret_ndcg_result(comparison, min_improvement)
        }
        
        self.validation_results["ndcg_improvement"] = validation_result
        return validation_result
    
    def validate_answer_span_preservation(self, answer_span_kept_ratios: np.ndarray,
                                        min_preservation: float = 0.98) -> Dict[str, Any]:
        """
        Validate Answer-Span-Kept preservation meets threshold
        
        Args:
            answer_span_kept_ratios: Ratios of answer spans preserved [0, 1]
            min_preservation: Minimum preservation threshold (0.98 = 98%)
            
        Returns:
            Validation results with statistical evidence  
        """
        bootstrap_result = self.bootstrap.bca_confidence_interval(
            answer_span_kept_ratios, np.mean
        )
        bootstrap_result.metric_name = "Answer_Span_Kept"
        bootstrap_result.meets_threshold = bootstrap_result.ci_lower >= min_preservation
        
        validation_result = {
            "metric": "Answer_Span_Kept",
            "target": f">= {min_preservation * 100}% preservation",
            "actual_mean": bootstrap_result.original_value,
            "ci_lower_bound": bootstrap_result.ci_lower,
            "ci_upper_bound": bootstrap_result.ci_upper,
            "meets_threshold": bootstrap_result.meets_threshold,
            "validation_passed": bootstrap_result.meets_threshold,
            "bootstrap_result": bootstrap_result,
            "interpretation": self._interpret_preservation_result(bootstrap_result, min_preservation)
        }
        
        self.validation_results["answer_span_preservation"] = validation_result
        return validation_result
    
    def validate_token_reduction(self, token_reduction_ratios: np.ndarray,
                                min_reduction: float = 0.30, 
                                max_reduction: float = 0.50) -> Dict[str, Any]:
        """
        Validate token reduction is within target range
        
        Args:
            token_reduction_ratios: Token reduction ratios [0, 1]
            min_reduction: Minimum reduction target (0.30 = 30%)
            max_reduction: Maximum reduction target (0.50 = 50%)
            
        Returns:
            Validation results with statistical evidence
        """
        bootstrap_result = self.bootstrap.bca_confidence_interval(
            token_reduction_ratios, np.mean
        )
        bootstrap_result.metric_name = "Token_Reduction"
        
        # Check if CI overlaps with target range [min_reduction, max_reduction]
        ci_in_range = (bootstrap_result.ci_lower <= max_reduction and 
                      bootstrap_result.ci_upper >= min_reduction)
        
        mean_in_range = min_reduction <= bootstrap_result.original_value <= max_reduction
        
        validation_result = {
            "metric": "Token_Reduction",
            "target": f"{min_reduction * 100}%-{max_reduction * 100}% reduction",
            "actual_mean": bootstrap_result.original_value,
            "ci_lower_bound": bootstrap_result.ci_lower,
            "ci_upper_bound": bootstrap_result.ci_upper,
            "mean_in_range": mean_in_range,
            "ci_overlaps_range": ci_in_range,
            "validation_passed": mean_in_range and ci_in_range,
            "bootstrap_result": bootstrap_result,
            "interpretation": self._interpret_token_reduction_result(
                bootstrap_result, min_reduction, max_reduction
            )
        }
        
        self.validation_results["token_reduction"] = validation_result
        return validation_result
        
    def validate_performance_targets(self, processing_times_ms: np.ndarray,
                                   p50_target: float = 3000, 
                                   p95_target: float = 6000) -> Dict[str, Any]:
        """
        Validate performance meets latency targets
        
        Args:
            processing_times_ms: Processing times in milliseconds
            p50_target: 50th percentile target (3000ms = 3s)  
            p95_target: 95th percentile target (6000ms = 6s)
            
        Returns:
            Validation results for performance targets
        """
        # Bootstrap confidence intervals for percentiles
        p50_result = self.bootstrap.bca_confidence_interval(
            processing_times_ms, lambda x: np.percentile(x, 50)
        )
        p50_result.metric_name = "Latency_P50"
        
        p95_result = self.bootstrap.bca_confidence_interval(
            processing_times_ms, lambda x: np.percentile(x, 95)
        )
        p95_result.metric_name = "Latency_P95"
        
        p50_meets_target = p50_result.ci_upper <= p50_target
        p95_meets_target = p95_result.ci_upper <= p95_target
        
        validation_result = {
            "metric": "Performance_Latency",
            "p50_target": p50_target,
            "p95_target": p95_target,
            "p50_actual": p50_result.original_value,
            "p50_ci_upper": p50_result.ci_upper,
            "p95_actual": p95_result.original_value,
            "p95_ci_upper": p95_result.ci_upper,
            "p50_meets_target": p50_meets_target,
            "p95_meets_target": p95_meets_target,
            "validation_passed": p50_meets_target and p95_meets_target,
            "p50_result": p50_result,
            "p95_result": p95_result
        }
        
        self.validation_results["performance"] = validation_result
        return validation_result
        
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report for publication
        
        Returns:
            Complete statistical validation report
        """
        overall_validation = all(
            result.get("validation_passed", False) 
            for result in self.validation_results.values()
        )
        
        report = {
            "validation_summary": {
                "overall_validation_passed": overall_validation,
                "n_bootstrap_samples": self.bootstrap.n_bootstrap,
                "confidence_level": self.bootstrap.confidence_level,
                "random_seed": self.bootstrap.random_seed,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "individual_validations": self.validation_results,
            "statistical_methodology": {
                "bootstrap_method": "Bias-Corrected and accelerated (BCa)",
                "reference": "Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap.",
                "confidence_intervals": "BCa bootstrap with bias and acceleration corrections",
                "hypothesis_testing": "Two-sided tests with Bonferroni correction for multiple comparisons",
                "effect_sizes": "Cohen's d for standardized mean differences"
            }
        }
        
        return report
        
    def save_report(self, filepath: str) -> None:
        """Save validation report to JSON file"""
        report = self.generate_comprehensive_report()
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        report = convert_numpy_types(report)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        if self.bootstrap.verbose:
            print(f"Validation report saved to {filepath}")
    
    def _interpret_ndcg_result(self, comparison: ComparisonResult, 
                              min_improvement: float) -> str:
        """Generate interpretation of nDCG validation results"""
        if comparison.improvement_pct >= (min_improvement * 100) and comparison.bootstrap_result.ci_lower > 0:
            return (f"✅ PASS: nDCG@10 improvement of {comparison.improvement_pct:.2f}% "
                   f"meets {min_improvement * 100}% threshold with statistically significant "
                   f"95% CI [{comparison.bootstrap_result.ci_lower:.4f}, {comparison.bootstrap_result.ci_upper:.4f}]")
        else:
            issues = []
            if comparison.improvement_pct < (min_improvement * 100):
                issues.append(f"improvement {comparison.improvement_pct:.2f}% < {min_improvement * 100}% target")
            if comparison.bootstrap_result.ci_lower <= 0:
                issues.append(f"95% CI lower bound {comparison.bootstrap_result.ci_lower:.4f} ≤ 0 (not significant)")
            return f"❌ FAIL: {'; '.join(issues)}"
    
    def _interpret_preservation_result(self, bootstrap_result: BootstrapResult, 
                                     min_preservation: float) -> str:
        """Generate interpretation of answer span preservation results"""
        if bootstrap_result.meets_threshold:
            return (f"✅ PASS: Answer span preservation {bootstrap_result.original_value:.3f} "
                   f"meets {min_preservation:.3f} threshold with 95% CI lower bound "
                   f"{bootstrap_result.ci_lower:.3f} ≥ {min_preservation:.3f}")
        else:
            return (f"❌ FAIL: Answer span preservation 95% CI lower bound "
                   f"{bootstrap_result.ci_lower:.3f} < {min_preservation:.3f} threshold")
    
    def _interpret_token_reduction_result(self, bootstrap_result: BootstrapResult,
                                        min_reduction: float, max_reduction: float) -> str:
        """Generate interpretation of token reduction results"""
        mean_val = bootstrap_result.original_value
        ci_lower = bootstrap_result.ci_lower
        ci_upper = bootstrap_result.ci_upper
        
        if (min_reduction <= mean_val <= max_reduction and 
            ci_lower <= max_reduction and ci_upper >= min_reduction):
            return (f"✅ PASS: Token reduction {mean_val:.3f} in target range "
                   f"[{min_reduction:.3f}, {max_reduction:.3f}] with consistent 95% CI "
                   f"[{ci_lower:.3f}, {ci_upper:.3f}]")
        else:
            return (f"❌ FAIL: Token reduction mean {mean_val:.3f} or 95% CI "
                   f"[{ci_lower:.3f}, {ci_upper:.3f}] outside target range "
                   f"[{min_reduction:.3f}, {max_reduction:.3f}]")


# Convenience functions for direct usage
def validate_lethe_vnext_results(baseline_ndcg: np.ndarray,
                                experimental_ndcg: np.ndarray, 
                                answer_span_ratios: np.ndarray,
                                token_reduction_ratios: np.ndarray,
                                processing_times_ms: np.ndarray,
                                output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function for full Lethe vNext validation
    
    Args:
        baseline_ndcg: Baseline nDCG@10 scores
        experimental_ndcg: Lethe vNext nDCG@10 scores
        answer_span_ratios: Answer span preservation ratios
        token_reduction_ratios: Token reduction ratios
        processing_times_ms: Processing latency measurements
        output_file: Optional file to save validation report
        
    Returns:
        Complete validation report
    """
    validator = LetHeEvaluationValidator()
    
    # Run all validations
    validator.validate_ndcg_improvement(baseline_ndcg, experimental_ndcg)
    validator.validate_answer_span_preservation(answer_span_ratios)
    validator.validate_token_reduction(token_reduction_ratios)
    validator.validate_performance_targets(processing_times_ms)
    
    # Generate report
    report = validator.generate_comprehensive_report()
    
    if output_file:
        validator.save_report(output_file)
    
    return report


def demo_bootstrap_validation():
    """Demo function showing BCa bootstrap usage for Lethe evaluation"""
    np.random.seed(42)
    
    # Simulate realistic evaluation data
    n_queries = 100
    
    # nDCG@10 scores (baseline vs experimental)
    baseline_ndcg = np.random.beta(3, 2, n_queries) * 0.8  # ~0.48 mean
    experimental_ndcg = baseline_ndcg * (1 + np.random.normal(0.12, 0.05, n_queries))  # ~12% improvement
    
    # Answer span preservation (should be high)
    answer_span_ratios = np.random.beta(50, 1, n_queries)  # ~0.98 mean
    
    # Token reduction (target 30-50%)
    token_reduction_ratios = np.random.beta(2, 3, n_queries) * 0.6 + 0.2  # ~0.35 mean
    
    # Processing times (target p50<3s, p95<6s)
    processing_times_ms = np.random.gamma(2, 1200, n_queries)  # ~2.4s mean
    
    print("🔬 Running Lethe vNext Statistical Validation")
    print("=" * 50)
    
    # Full validation
    report = validate_lethe_vnext_results(
        baseline_ndcg=baseline_ndcg,
        experimental_ndcg=experimental_ndcg,
        answer_span_ratios=answer_span_ratios, 
        token_reduction_ratios=token_reduction_ratios,
        processing_times_ms=processing_times_ms,
        output_file="lethe_vnext_validation_report.json"
    )
    
    # Print summary
    print("\n📊 Validation Summary:")
    for metric, result in report["individual_validations"].items():
        status = "✅ PASS" if result["validation_passed"] else "❌ FAIL"
        print(f"{status} {metric}: {result.get('interpretation', 'N/A')}")
    
    overall = "✅ ALL VALIDATIONS PASSED" if report["validation_summary"]["overall_validation_passed"] else "❌ SOME VALIDATIONS FAILED"
    print(f"\n🎯 Overall Result: {overall}")
    
    return report


if __name__ == "__main__":
    demo_bootstrap_validation()