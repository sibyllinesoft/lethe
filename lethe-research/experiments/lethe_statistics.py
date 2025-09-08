#!/usr/bin/env python3
"""
Advanced Statistical Analysis Framework for Lethe Research
=========================================================

This module implements publication-grade statistical analysis with bootstrap
confidence intervals, effect size calculations, and multiple comparison corrections
for the Lethe hybrid retrieval system evaluation.

Features:
- 10k paired bootstrap with BCa 95% CI
- Effect size calculations (Cohen's d, Glass's delta)
- Multiple comparison corrections (Bonferroni, Holm, FDR)
- Statistical significance testing with power analysis
- Comprehensive reporting with visualization-ready outputs
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optimize
from scipy.stats import bootstrap, ttest_rel, mannwhitneyu, wilcoxon, permutation_test
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import ttest_power
import statsmodels.stats.api as sms
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import warnings
from dataclasses import dataclass, asdict, field
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import hashlib
import time
from datetime import datetime
import sys

@dataclass
class BootstrapResult:
    """Bootstrap confidence interval result with enhanced diagnostics"""
    statistic: float
    confidence_interval: Tuple[float, float]
    standard_error: float
    bias: float
    bias_corrected_statistic: float
    acceleration: float
    n_bootstrap: int
    confidence_level: float
    method: str  # 'BCa', 'percentile', 'basic', 'studentized'
    jackknife_values: Optional[List[float]] = None
    bootstrap_distribution: Optional[List[float]] = None
    convergence_achieved: bool = True
    effective_samples: int = 0

@dataclass
class EffectSizeResult:
    """Comprehensive effect size calculation result with confidence intervals"""
    cohens_d: float
    glass_delta: float
    hedges_g: float
    cliff_delta: float
    common_language_effect: float
    effect_size_interpretation: str
    practical_significance: str
    sample_sizes: Tuple[int, int]
    
    # Confidence intervals (optional)
    cohens_d_ci: Optional[Tuple[float, float]] = None
    glass_delta_ci: Optional[Tuple[float, float]] = None
    hedges_g_ci: Optional[Tuple[float, float]] = None
    cliff_delta_ci: Optional[Tuple[float, float]] = None
    common_language_effect_ci: Optional[Tuple[float, float]] = None

@dataclass
class StatisticalTest:
    """Enhanced statistical test result with comprehensive diagnostics"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: EffectSizeResult
    confidence_interval: BootstrapResult
    degrees_of_freedom: Optional[int]
    sample_size: Tuple[int, int]
    power: float
    significant: bool
    alpha: float

@dataclass
class MultipleComparisonResult:
    """Multiple comparison correction result"""
    method: str
    original_p_values: List[float]
    corrected_p_values: List[float]
    rejected: List[bool]
    alpha: float
    n_comparisons: int
    family_wise_error_rate: float

@dataclass 
class PairedComparisonMatrix:
    """Complete pairwise comparison matrix"""
    methods: List[str]
    metric: str
    p_values: np.ndarray
    effect_sizes: np.ndarray
    significance_matrix: np.ndarray
    correction_method: str
    alpha: float

class AdvancedStatisticalAnalyzer:
    """
    Publication-quality statistical analysis framework for NeurIPS 2025.
    
    Implements rigorous bootstrap analysis with BCa confidence intervals,
    comprehensive effect size calculations, FDR-controlled multiple comparisons,
    and extensive statistical validation for machine learning research.
    
    Features:
    - 10k+ paired bootstrap with bias-corrected & accelerated (BCa) intervals
    - Multiple effect size measures with confidence intervals
    - FDR control using Benjamini-Hochberg procedure
    - Statistical power analysis and sample size calculations
    - Assumption testing and non-parametric alternatives
    - Reproducible analysis with comprehensive provenance tracking
    """
    
    def __init__(
        self, 
        bootstrap_samples: int = 10000,
        confidence_level: float = 0.95,
        alpha: float = 0.05,
        min_effect_size: float = 0.1,
        power_threshold: float = 0.8,
        n_jobs: int = -1,
        random_state: int = 42,
        cache_results: bool = True,
        verbose: bool = True
    ):
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.alpha = alpha
        self.min_effect_size = min_effect_size
        self.power_threshold = power_threshold
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        self.random_state = random_state
        self.cache_results = cache_results
        self.verbose = verbose
        
        # Initialize random state
        np.random.seed(random_state)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
        self.logger = logging.getLogger(__name__)
        
        # Suppress warnings in production
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        
        # Analysis provenance tracking
        self.analysis_metadata = {
            'analyzer_version': '3.0.0',
            'bootstrap_samples': bootstrap_samples,
            'confidence_level': confidence_level,
            'alpha': alpha,
            'min_effect_size': min_effect_size,
            'power_threshold': power_threshold,
            'random_state': random_state,
            'created_at': datetime.now().isoformat(),
            'python_version': sys.version,
            'numpy_version': np.__version__,
            'scipy_version': stats.__version__
        }
        
        # Result caching
        self._cache = {} if cache_results else None
        
        if self.verbose:
            self.logger.info(f"🧮 Advanced Statistical Analyzer initialized")
            self.logger.info(f"📊 Bootstrap: {bootstrap_samples:,} samples, CI: {confidence_level:.1%}")
            self.logger.info(f"🎯 Alpha: {alpha}, Min effect: {min_effect_size}, Power: {power_threshold}")
            self.logger.info(f"⚡ Parallel jobs: {self.n_jobs}, Caching: {cache_results}")

    def _cache_key(self, *args, **kwargs) -> str:
        """Generate cache key for memoization"""
        if not self.cache_results:
            return None
        key_data = f"{args}_{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def bootstrap_confidence_interval(
        self, 
        data: Union[np.ndarray, List[float]],
        statistic_function: Callable = np.mean,
        method: str = 'BCa',
        paired_data: Optional[Union[np.ndarray, List[float]]] = None,
        return_distribution: bool = False
    ) -> BootstrapResult:
        """
        Compute publication-quality bootstrap confidence interval.
        
        Uses bias-corrected and accelerated (BCa) bootstrap by default,
        which provides improved coverage for skewed distributions and
        small sample sizes.
        
        Args:
            data: Sample data (array-like)
            statistic_function: Function to compute statistic (default: np.mean)
            method: 'BCa' (recommended), 'percentile', 'basic', or 'studentized'
            paired_data: If provided, compute statistic on paired differences
            return_distribution: Whether to return bootstrap distribution
            
        Returns:
            BootstrapResult with comprehensive bootstrap diagnostics
            
        References:
            Efron, B., & Tibshirani, R. J. (1993). An introduction to the bootstrap.
        """
        # Input validation and preprocessing
        data = np.asarray(data, dtype=float)
        if len(data) < 3:
            raise ValueError("Need at least 3 data points for bootstrap analysis")
            
        if paired_data is not None:
            paired_data = np.asarray(paired_data, dtype=float)
            if len(data) != len(paired_data):
                raise ValueError("Data arrays must have same length for paired analysis")
            # Compute differences for paired bootstrap
            data = data - paired_data
            
        # Remove any NaN values
        data = data[~np.isnan(data)]
        if len(data) < 3:
            raise ValueError("Insufficient valid data points after removing NaN values")
            
        # Check for cache hit
        cache_key = self._cache_key(data.tobytes(), statistic_function.__name__, method, self.bootstrap_samples)
        if cache_key and cache_key in self._cache:
            if self.verbose:
                self.logger.debug("Cache hit for bootstrap analysis")
            return self._cache[cache_key]
        
        # Compute observed statistic
        try:
            observed_stat = float(statistic_function(data))
        except Exception as e:
            raise ValueError(f"Statistic function failed on data: {e}")
            
        if not np.isfinite(observed_stat):
            raise ValueError(f"Statistic function returned non-finite value: {observed_stat}")
        
        # Generate bootstrap samples with progress tracking
        rng = np.random.RandomState(self.random_state)
        bootstrap_stats = np.zeros(self.bootstrap_samples)
        
        # Vectorized bootstrap for efficiency
        n = len(data)
        
        start_time = time.time()
        for i in range(self.bootstrap_samples):
            bootstrap_indices = rng.randint(0, n, size=n)
            bootstrap_sample = data[bootstrap_indices]
            try:
                bootstrap_stats[i] = statistic_function(bootstrap_sample)
            except Exception:
                # Handle edge cases where statistic computation fails
                bootstrap_stats[i] = np.nan
                
        # Remove any failed bootstrap samples
        valid_mask = np.isfinite(bootstrap_stats)
        bootstrap_stats = bootstrap_stats[valid_mask]
        effective_samples = len(bootstrap_stats)
        
        if effective_samples < self.bootstrap_samples * 0.95:
            self.logger.warning(f"Only {effective_samples}/{self.bootstrap_samples} bootstrap samples succeeded")
            
        if effective_samples < 100:
            raise ValueError(f"Too few valid bootstrap samples: {effective_samples}")
        
        # Compute confidence interval based on method
        if method == 'percentile':
            lower_percentile = (1 - self.confidence_level) / 2 * 100
            upper_percentile = (1 + self.confidence_level) / 2 * 100
            ci_lower = np.percentile(bootstrap_stats, lower_percentile)
            ci_upper = np.percentile(bootstrap_stats, upper_percentile)
            acceleration = 0.0
            jackknife_values = None
            
        elif method == 'basic':
            lower_percentile = (1 - self.confidence_level) / 2 * 100
            upper_percentile = (1 + self.confidence_level) / 2 * 100
            ci_lower = 2 * observed_stat - np.percentile(bootstrap_stats, upper_percentile)
            ci_upper = 2 * observed_stat - np.percentile(bootstrap_stats, lower_percentile)
            acceleration = 0.0
            jackknife_values = None
            
        elif method == 'BCa':
            # Bias-corrected and accelerated bootstrap (recommended)
            ci_lower, ci_upper, acceleration, jackknife_values = self._bca_confidence_interval(
                data, bootstrap_stats, observed_stat, statistic_function
            )
            
        elif method == 'studentized':
            # Studentized bootstrap (for improved accuracy)
            ci_lower, ci_upper, acceleration = self._studentized_bootstrap(
                data, bootstrap_stats, observed_stat, statistic_function
            )
            jackknife_values = None
            
        else:
            raise ValueError(f"Unknown bootstrap method: {method}. Use 'BCa', 'percentile', 'basic', or 'studentized'")
            
        # Ensure finite confidence intervals
        if not (np.isfinite(ci_lower) and np.isfinite(ci_upper)):
            self.logger.warning(f"Non-finite confidence interval computed, falling back to percentile method")
            lower_percentile = (1 - self.confidence_level) / 2 * 100
            upper_percentile = (1 + self.confidence_level) / 2 * 100
            ci_lower = np.percentile(bootstrap_stats, lower_percentile)
            ci_upper = np.percentile(bootstrap_stats, upper_percentile)
            acceleration = 0.0
            
        # Compute comprehensive bootstrap diagnostics
        standard_error = np.std(bootstrap_stats, ddof=1)
        bias = np.mean(bootstrap_stats) - observed_stat
        bias_corrected_statistic = observed_stat - bias
        
        # Check for convergence (standard error stability)
        if effective_samples >= 1000:
            # Check if doubling bootstrap samples would change SE by < 5%
            mid_point = effective_samples // 2
            se_first_half = np.std(bootstrap_stats[:mid_point], ddof=1)
            se_second_half = np.std(bootstrap_stats[mid_point:], ddof=1)
            convergence_achieved = abs(se_first_half - se_second_half) / standard_error < 0.05
        else:
            convergence_achieved = True  # Assume convergence for small samples
            
        computation_time = time.time() - start_time
        
        if self.verbose:
            self.logger.info(f"🔄 Bootstrap completed: {effective_samples:,} samples in {computation_time:.2f}s")
            self.logger.info(f"📈 Statistic: {observed_stat:.4f}, SE: {standard_error:.4f}, Bias: {bias:.4f}")
            if not convergence_achieved:
                self.logger.warning("⚠️  Bootstrap may not have converged - consider more samples")
        
        result = BootstrapResult(
            statistic=observed_stat,
            confidence_interval=(ci_lower, ci_upper),
            standard_error=standard_error,
            bias=bias,
            bias_corrected_statistic=bias_corrected_statistic,
            acceleration=acceleration,
            n_bootstrap=self.bootstrap_samples,
            confidence_level=self.confidence_level,
            method=method,
            jackknife_values=jackknife_values,
            bootstrap_distribution=bootstrap_stats.tolist() if return_distribution else None,
            convergence_achieved=convergence_achieved,
            effective_samples=effective_samples
        )
        
        # Cache result
        if cache_key:
            self._cache[cache_key] = result
            
        return result
    
    def _bca_confidence_interval(
        self, 
        data: np.ndarray, 
        bootstrap_stats: np.ndarray, 
        observed_stat: float,
        statistic_function: Callable
    ) -> Tuple[float, float, float, List[float]]:
        """
        Compute bias-corrected and accelerated (BCa) confidence interval.
        
        The BCa method provides improved coverage probability, especially
        for skewed distributions and transformation-respecting intervals.
        
        Returns:
            Tuple of (ci_lower, ci_upper, acceleration, jackknife_values)
            
        References:
            Efron, B. (1987). Better bootstrap confidence intervals.
            Journal of the American statistical Association, 82(397), 171-185.
        """
        n = len(data)
        n_boot = len(bootstrap_stats)
        
        # Bias correction (z0)
        n_below = np.sum(bootstrap_stats <= observed_stat)  # Use <= for ties
        if n_below == 0:
            n_below = 0.5  # Avoid infinite bias correction
        elif n_below == n_boot:
            n_below = n_boot - 0.5
            
        bias_correction = stats.norm.ppf(n_below / n_boot)
        
        # Acceleration correction using jackknife
        jackknife_stats = np.zeros(n)
        
        # Efficient jackknife computation
        for i in range(n):
            # Leave-one-out sample
            jackknife_sample = np.concatenate([data[:i], data[i+1:]])
            try:
                jackknife_stats[i] = statistic_function(jackknife_sample)
            except Exception:
                # Handle edge cases where jackknife fails
                jackknife_stats[i] = observed_stat
                
        jackknife_mean = np.mean(jackknife_stats)
        centered_jackknife = jackknife_mean - jackknife_stats
        
        # Acceleration parameter (a)
        numerator = np.sum(centered_jackknife ** 3)
        denominator = 6 * (np.sum(centered_jackknife ** 2)) ** (3/2)
        
        if abs(denominator) < 1e-10:
            acceleration = 0.0
        else:
            acceleration = numerator / denominator
            
        # Clip acceleration to reasonable bounds
        acceleration = np.clip(acceleration, -0.99, 0.99)
        
        # Adjusted percentiles
        z_alpha = stats.norm.ppf((1 - self.confidence_level) / 2)
        z_1_alpha = -z_alpha
        
        # Lower confidence limit
        denominator_lower = 1 - acceleration * (bias_correction + z_alpha)
        if abs(denominator_lower) < 1e-10:
            alpha_1 = (1 - self.confidence_level) / 2
        else:
            alpha_1 = stats.norm.cdf(bias_correction + (bias_correction + z_alpha) / denominator_lower)
            
        # Upper confidence limit  
        denominator_upper = 1 - acceleration * (bias_correction + z_1_alpha)
        if abs(denominator_upper) < 1e-10:
            alpha_2 = (1 + self.confidence_level) / 2
        else:
            alpha_2 = stats.norm.cdf(bias_correction + (bias_correction + z_1_alpha) / denominator_upper)
            
        # Ensure percentiles are within valid range
        alpha_1 = np.clip(alpha_1, 0.0001, 0.9999)
        alpha_2 = np.clip(alpha_2, 0.0001, 0.9999)
        
        # Ensure proper ordering
        if alpha_1 > alpha_2:
            alpha_1, alpha_2 = alpha_2, alpha_1
            
        # Compute confidence interval
        ci_lower = np.percentile(bootstrap_stats, alpha_1 * 100, method='interpolated_inverted_cdf')
        ci_upper = np.percentile(bootstrap_stats, alpha_2 * 100, method='interpolated_inverted_cdf')
        
        return ci_lower, ci_upper, acceleration, jackknife_stats.tolist()
        
    def _studentized_bootstrap(
        self,
        data: np.ndarray,
        bootstrap_stats: np.ndarray, 
        observed_stat: float,
        statistic_function: Callable
    ) -> Tuple[float, float, float]:
        """
        Compute studentized bootstrap confidence interval.
        
        More accurate than basic bootstrap when the statistic's
        variance depends on the parameter being estimated.
        """
        n = len(data)
        
        # Estimate the variance of the statistic using nested bootstrap
        def variance_estimator(sample):
            stat = statistic_function(sample)
            # Nested bootstrap for variance estimation (smaller sample)
            nested_boots = []
            rng = np.random.RandomState(self.random_state)
            
            for _ in range(min(200, self.bootstrap_samples // 10)):
                nested_sample = rng.choice(sample, size=len(sample), replace=True)
                nested_boots.append(statistic_function(nested_sample))
                
            return np.var(nested_boots, ddof=1)
        
        # Compute studentized statistics
        studentized_stats = []
        rng = np.random.RandomState(self.random_state)
        
        for _ in range(min(1000, len(bootstrap_stats))):
            boot_sample = rng.choice(data, size=n, replace=True) 
            boot_stat = statistic_function(boot_sample)
            boot_var = variance_estimator(boot_sample)
            
            if boot_var > 0:
                studentized_stats.append((boot_stat - observed_stat) / np.sqrt(boot_var))
                
        if len(studentized_stats) < 50:
            # Fall back to basic bootstrap if studentization fails
            lower_percentile = (1 - self.confidence_level) / 2 * 100
            upper_percentile = (1 + self.confidence_level) / 2 * 100
            ci_lower = 2 * observed_stat - np.percentile(bootstrap_stats, upper_percentile)
            ci_upper = 2 * observed_stat - np.percentile(bootstrap_stats, lower_percentile)
            return ci_lower, ci_upper, 0.0
            
        studentized_stats = np.array(studentized_stats)
        
        # Estimate variance of original statistic
        original_var = variance_estimator(data)
        
        if original_var <= 0:
            # Fall back to basic method
            lower_percentile = (1 - self.confidence_level) / 2 * 100
            upper_percentile = (1 + self.confidence_level) / 2 * 100
            ci_lower = 2 * observed_stat - np.percentile(bootstrap_stats, upper_percentile) 
            ci_upper = 2 * observed_stat - np.percentile(bootstrap_stats, lower_percentile)
            return ci_lower, ci_upper, 0.0
            
        # Compute confidence interval
        alpha_level = (1 - self.confidence_level) / 2
        t_lower = np.percentile(studentized_stats, alpha_level * 100)
        t_upper = np.percentile(studentized_stats, (1 - alpha_level) * 100)
        
        se = np.sqrt(original_var)
        ci_lower = observed_stat - t_upper * se
        ci_upper = observed_stat - t_lower * se
        
        return ci_lower, ci_upper, 0.0

    def compute_effect_size(
        self, 
        group1: Union[np.ndarray, List[float]], 
        group2: Union[np.ndarray, List[float]],
        paired: bool = False,
        compute_ci: bool = True
    ) -> EffectSizeResult:
        """
        Compute comprehensive effect size metrics.
        
        Args:
            group1: First group data
            group2: Second group data  
            paired: Whether the groups are paired (e.g., before/after)
            
        Returns:
            EffectSizeResult with multiple effect size metrics
        """
        group1 = np.asarray(group1, dtype=float)
        group2 = np.asarray(group2, dtype=float)
        
        # Remove any NaN values
        group1 = group1[~np.isnan(group1)]
        group2 = group2[~np.isnan(group2)]
        
        if len(group1) < 2 or len(group2) < 2:
            raise ValueError("Need at least 2 valid observations per group")
        
        # Basic statistics
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        # Cohen's d (standardized mean difference)
        if paired:
            # For paired data, use standard deviation of differences
            differences = group1 - group2
            cohens_d = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences, ddof=1) > 0 else 0.0
        else:
            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
        
        # Glass's delta (uses control group SD)
        glass_delta = (mean1 - mean2) / std2 if std2 > 0 else 0.0
        
        # Hedges' g (bias-corrected Cohen's d)
        if paired:
            hedges_g = cohens_d  # No correction needed for paired data
        else:
            correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
            hedges_g = cohens_d * correction_factor
        
        # Cliff's delta (non-parametric effect size)
        cliff_delta = self._compute_cliff_delta(group1, group2)
        
        # Common language effect size (probability of superiority)
        if paired:
            # For paired data: P(X1 > X2)
            common_language_effect = np.mean(group1 > group2)
        else:
            # For independent groups: P(random X1 > random X2)
            comparison_matrix = group1[:, np.newaxis] > group2[np.newaxis, :]
            common_language_effect = np.mean(comparison_matrix)
        
        # Compute confidence intervals via bootstrap if requested
        cohens_d_ci = glass_delta_ci = hedges_g_ci = cliff_delta_ci = cles_ci = None
        
        if compute_ci:
            # Bootstrap confidence intervals for effect sizes
            def cohens_d_func(g1, g2):
                if paired:
                    diffs = g1 - g2
                    return np.mean(diffs) / np.std(diffs, ddof=1) if np.std(diffs, ddof=1) > 0 else 0.0
                else:
                    m1, m2 = np.mean(g1), np.mean(g2)
                    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
                    pooled = np.sqrt(((len(g1)-1)*s1**2 + (len(g2)-1)*s2**2) / (len(g1)+len(g2)-2))
                    return (m1 - m2) / pooled if pooled > 0 else 0.0
                    
            def glass_delta_func(g1, g2):
                return (np.mean(g1) - np.mean(g2)) / np.std(g2, ddof=1) if np.std(g2, ddof=1) > 0 else 0.0
                
            def cliff_delta_func(g1, g2):
                return self._compute_cliff_delta(g1, g2)
                
            def cles_func(g1, g2):
                if paired:
                    return np.mean(g1 > g2)
                else:
                    return np.mean(g1[:, np.newaxis] > g2[np.newaxis, :])
            
            try:
                # Use smaller bootstrap samples for effect sizes to balance accuracy and speed
                original_samples = self.bootstrap_samples
                self.bootstrap_samples = min(5000, self.bootstrap_samples)
                
                cohens_d_bootstrap = self._bootstrap_difference(group1, group2, cohens_d_func)
                cohens_d_ci = cohens_d_bootstrap.confidence_interval
                
                glass_delta_bootstrap = self._bootstrap_difference(group1, group2, glass_delta_func)
                glass_delta_ci = glass_delta_bootstrap.confidence_interval
                
                cliff_delta_bootstrap = self._bootstrap_difference(group1, group2, cliff_delta_func)
                cliff_delta_ci = cliff_delta_bootstrap.confidence_interval
                
                cles_bootstrap = self._bootstrap_difference(group1, group2, cles_func)
                cles_ci = cles_bootstrap.confidence_interval
                
                # Restore original bootstrap samples
                self.bootstrap_samples = original_samples
                
            except Exception as e:
                if self.verbose:
                    self.logger.warning(f"Bootstrap confidence intervals failed: {e}")
        
        # Effect size interpretation with confidence bounds
        effect_interpretation = self._interpret_effect_size(abs(cohens_d))
        
        # Practical significance assessment
        practical_significance = self._assess_practical_significance(
            cohens_d, cohens_d_ci, threshold=0.2
        )
        
        return EffectSizeResult(
            cohens_d=cohens_d,
            cohens_d_ci=cohens_d_ci,
            glass_delta=glass_delta,
            glass_delta_ci=glass_delta_ci,
            hedges_g=hedges_g,
            hedges_g_ci=(cohens_d_ci[0] * (1 - (3 / (4 * (n1 + n2) - 9))), 
                        cohens_d_ci[1] * (1 - (3 / (4 * (n1 + n2) - 9)))) if cohens_d_ci and n1 + n2 > 3 else cohens_d_ci,
            cliff_delta=cliff_delta,
            cliff_delta_ci=cliff_delta_ci,
            common_language_effect=common_language_effect,
            common_language_effect_ci=cles_ci,
            effect_size_interpretation=effect_interpretation,
            practical_significance=practical_significance,
            sample_sizes=(n1, n2)
        )
    
    def _compute_cliff_delta(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Compute Cliff's delta (non-parametric effect size)"""
        n1, n2 = len(group1), len(group2)
        
        # Count pairs where group1[i] > group2[j] and group1[i] < group2[j]
        greater_count = 0
        less_count = 0
        
        for x1 in group1:
            for x2 in group2:
                if x1 > x2:
                    greater_count += 1
                elif x1 < x2:
                    less_count += 1
        
        total_pairs = n1 * n2
        cliff_delta = (greater_count - less_count) / total_pairs
        
        return cliff_delta
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """
        Interpret Cohen's d effect size magnitude using established conventions.
        
        References:
            Cohen, J. (1988). Statistical power analysis for the behavioral sciences (2nd ed.).
            Ferguson, C. J. (2009). An effect size primer: A guide for clinicians and researchers.
        """
        abs_effect = abs(effect_size)
        if abs_effect < 0.01:
            return "trivial"
        elif abs_effect < 0.2:
            return "negligible"  
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        elif abs_effect < 1.2:
            return "large"
        else:
            return "very large"
    
    def _assess_practical_significance(
        self, 
        effect_size: float, 
        effect_size_ci: Optional[Tuple[float, float]], 
        threshold: float = 0.2
    ) -> str:
        """
        Assess practical significance based on effect size and confidence interval.
        
        Args:
            effect_size: Point estimate of effect size
            effect_size_ci: Confidence interval for effect size
            threshold: Minimum effect size for practical significance
            
        Returns:
            Assessment of practical significance
        """
        abs_effect = abs(effect_size)
        
        if effect_size_ci is None:
            # Without CI, use point estimate only
            return "practically significant" if abs_effect >= threshold else "not practically significant"
        
        ci_lower, ci_upper = effect_size_ci
        abs_ci_lower, abs_ci_upper = abs(ci_lower), abs(ci_upper)
        
        # Conservative assessment: entire CI must exceed threshold
        if min(abs_ci_lower, abs_ci_upper) >= threshold:
            return "practically significant"
        elif max(abs_ci_lower, abs_ci_upper) < threshold:
            return "not practically significant"
        else:
            return "inconclusive practical significance"

    def statistical_test(
        self, 
        group1: Union[np.ndarray, List[float]], 
        group2: Union[np.ndarray, List[float]],
        test_type: str = 'auto',
        paired: bool = False,
        alternative: str = 'two-sided'
    ) -> StatisticalTest:
        """
        Perform statistical test with bootstrap CI and effect size.
        
        Args:
            group1: First group data
            group2: Second group data
            test_type: 'auto', 't-test', 'mann-whitney', 'wilcoxon'
            paired: Whether groups are paired
            alternative: 'two-sided', 'less', 'greater'
            
        Returns:
            StatisticalTest with comprehensive results
        """
        group1 = np.asarray(group1)
        group2 = np.asarray(group2)
        
        # Determine test type automatically if needed
        if test_type == 'auto':
            if paired:
                # Check normality of differences for paired data
                differences = group1 - group2
                _, p_norm = stats.shapiro(differences) if len(differences) <= 5000 else stats.normaltest(differences)
                test_type = 'wilcoxon' if p_norm < 0.05 else 't-test'
            else:
                # Check normality for independent groups
                _, p_norm1 = stats.shapiro(group1) if len(group1) <= 5000 else stats.normaltest(group1)
                _, p_norm2 = stats.shapiro(group2) if len(group2) <= 5000 else stats.normaltest(group2)
                
                # Check homogeneity of variance
                _, p_levene = stats.levene(group1, group2)
                
                if p_norm1 < 0.05 or p_norm2 < 0.05 or p_levene < 0.05:
                    test_type = 'mann-whitney'
                else:
                    test_type = 't-test'
        
        # Perform statistical test
        if test_type == 't-test':
            if paired:
                statistic, p_value = ttest_rel(group1, group2, alternative=alternative)
                df = len(group1) - 1
                test_name = 'Paired t-test'
            else:
                statistic, p_value = stats.ttest_ind(group1, group2, alternative=alternative)
                df = len(group1) + len(group2) - 2
                test_name = 'Independent t-test'
                
        elif test_type == 'mann-whitney':
            statistic, p_value = mannwhitneyu(group1, group2, alternative=alternative)
            df = None
            test_name = 'Mann-Whitney U test'
            
        elif test_type == 'wilcoxon':
            statistic, p_value = wilcoxon(group1, group2, alternative=alternative)
            df = None
            test_name = 'Wilcoxon signed-rank test'
            
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Compute effect size
        effect_size = self.compute_effect_size(group1, group2, paired=paired)
        
        # Compute bootstrap confidence interval for difference in means
        if paired:
            differences = group1 - group2
            bootstrap_ci = self.bootstrap_confidence_interval(differences, np.mean)
        else:
            # Bootstrap difference of means
            def mean_difference(data1, data2):
                return np.mean(data1) - np.mean(data2)
            
            bootstrap_ci = self._bootstrap_difference(group1, group2, mean_difference)
        
        # Compute statistical power
        power = self._compute_power(group1, group2, effect_size.cohens_d, test_type, alpha=self.alpha)
        
        # Determine significance
        significant = p_value < self.alpha
        
        return StatisticalTest(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=bootstrap_ci,
            degrees_of_freedom=df,
            sample_size=(len(group1), len(group2)),
            power=power,
            significant=significant,
            alpha=self.alpha
        )
    
    def _bootstrap_difference(
        self, 
        group1: np.ndarray, 
        group2: np.ndarray, 
        statistic_function: callable
    ) -> BootstrapResult:
        """Bootstrap confidence interval for difference between two groups"""
        observed_stat = statistic_function(group1, group2)
        
        rng = np.random.RandomState(self.random_state)
        bootstrap_stats = []
        
        for _ in range(self.bootstrap_samples):
            bootstrap_sample1 = rng.choice(group1, size=len(group1), replace=True)
            bootstrap_sample2 = rng.choice(group2, size=len(group2), replace=True)
            bootstrap_stat = statistic_function(bootstrap_sample1, bootstrap_sample2)
            bootstrap_stats.append(bootstrap_stat)
        
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Compute percentile CI
        lower_percentile = (1 - self.confidence_level) / 2 * 100
        upper_percentile = (1 + self.confidence_level) / 2 * 100
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        standard_error = np.std(bootstrap_stats)
        bias = np.mean(bootstrap_stats) - observed_stat
        
        return BootstrapResult(
            statistic=observed_stat,
            confidence_interval=(ci_lower, ci_upper),
            standard_error=standard_error,
            bias=bias,
            n_bootstrap=self.bootstrap_samples,
            confidence_level=self.confidence_level,
            method='percentile'
        )
    
    def _compute_power(
        self, 
        group1: np.ndarray, 
        group2: np.ndarray, 
        effect_size: float,
        test_type: str, 
        alpha: float
    ) -> float:
        """Compute statistical power for the test"""
        n1, n2 = len(group1), len(group2)
        
        if test_type in ['t-test', 'Paired t-test', 'Independent t-test']:
            # Use scipy's ttest_power for t-tests
            if 'Paired' in test_type:
                # For paired t-test, use n1 as sample size
                power = stats.ttest_power(effect_size, n1, alpha)
            else:
                # For independent t-test, use harmonic mean of sample sizes
                n_harmonic = 2 * n1 * n2 / (n1 + n2)
                power = stats.ttest_power(effect_size, n_harmonic, alpha)
        else:
            # For non-parametric tests, use approximation based on asymptotic relative efficiency
            # Mann-Whitney has ARE of ~0.955 compared to t-test
            # Wilcoxon has ARE of ~0.955 compared to paired t-test
            are = 0.955
            effective_n = n1 * are if 'Paired' in test_type else (2 * n1 * n2 / (n1 + n2)) * are
            power = stats.ttest_power(effect_size, effective_n, alpha)
        
        return min(1.0, max(0.0, power))  # Clamp to [0, 1]
    
    def permutation_test(
        self, 
        group1: Union[np.ndarray, List[float]], 
        group2: Union[np.ndarray, List[float]],
        statistic_function: Callable = lambda x, y: np.mean(x) - np.mean(y),
        n_permutations: int = 10000,
        alternative: str = 'two-sided'
    ) -> float:
        """
        Perform permutation test for hypothesis testing.
        
        Provides exact p-values without distributional assumptions.
        Particularly useful for small samples or non-normal data.
        
        Args:
            group1: First group data
            group2: Second group data
            statistic_function: Function to compute test statistic
            n_permutations: Number of permutations
            alternative: 'two-sided', 'less', 'greater'
            
        Returns:
            Permutation test p-value
            
        References:
            Good, P. I. (2005). Permutation, Parametric and Bootstrap Tests of Hypotheses.
        """
        group1 = np.asarray(group1, dtype=float)
        group2 = np.asarray(group2, dtype=float)
        
        # Remove NaN values
        group1 = group1[~np.isnan(group1)]
        group2 = group2[~np.isnan(group2)]
        
        if len(group1) == 0 or len(group2) == 0:
            raise ValueError("Groups cannot be empty after removing NaN values")
        
        # Observed test statistic
        observed_stat = statistic_function(group1, group2)
        
        # Combine groups for permutation
        combined = np.concatenate([group1, group2])
        n1, n2 = len(group1), len(group2)
        
        # Generate permutation distribution
        rng = np.random.RandomState(self.random_state)
        perm_stats = np.zeros(n_permutations)
        
        for i in range(n_permutations):
            # Randomly permute the combined data
            permuted = rng.permutation(combined)
            perm_group1 = permuted[:n1]
            perm_group2 = permuted[n1:]
            
            try:
                perm_stats[i] = statistic_function(perm_group1, perm_group2)
            except Exception:
                perm_stats[i] = np.nan
        
        # Remove any NaN results from failed computations
        perm_stats = perm_stats[~np.isnan(perm_stats)]
        
        if len(perm_stats) < n_permutations * 0.95:
            self.logger.warning(f"Only {len(perm_stats)}/{n_permutations} permutations succeeded")
        
        # Compute p-value based on alternative hypothesis
        if alternative == 'two-sided':
            p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))
        elif alternative == 'greater':
            p_value = np.mean(perm_stats >= observed_stat)
        elif alternative == 'less':
            p_value = np.mean(perm_stats <= observed_stat)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")
        
        return max(1.0 / n_permutations, p_value)  # Avoid exactly zero p-values
    
    def sample_size_calculation(
        self,
        effect_size: float,
        power: float = 0.8,
        alpha: float = None,
        test_type: str = 't-test',
        alternative: str = 'two-sided'
    ) -> int:
        """
        Calculate required sample size for desired statistical power.
        
        Essential for proper experimental design and ensuring
        adequate power to detect meaningful effects.
        
        Args:
            effect_size: Expected effect size (Cohen's d)
            power: Desired statistical power (default 0.8)
            alpha: Significance level (uses self.alpha if None)
            test_type: Type of statistical test
            alternative: 'two-sided', 'greater', 'less'
            
        Returns:
            Required sample size per group
            
        References:
            Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
        """
        if alpha is None:
            alpha = self.alpha
            
        if test_type in ['t-test', 'Independent t-test']:
            # For independent t-test
            try:
                from statsmodels.stats.power import ttest_power, tt_solve_power
                n_required = tt_solve_power(
                    effect_size=effect_size,
                    power=power,
                    alpha=alpha,
                    alternative=alternative
                )
                return int(np.ceil(n_required))
            except ImportError:
                # Fallback approximation for independent t-test
                if alternative == 'two-sided':
                    z_alpha = stats.norm.ppf(1 - alpha/2)
                    z_beta = stats.norm.ppf(power)
                else:
                    z_alpha = stats.norm.ppf(1 - alpha)
                    z_beta = stats.norm.ppf(power)
                    
                n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
                return int(np.ceil(n))
                
        elif test_type in ['Paired t-test']:
            # For paired t-test (smaller sample size needed)
            try:
                from statsmodels.stats.power import ttest_power, tt_solve_power
                n_required = tt_solve_power(
                    effect_size=effect_size,
                    power=power,
                    alpha=alpha,
                    alternative=alternative
                )
                return int(np.ceil(n_required))
            except ImportError:
                # Fallback approximation
                if alternative == 'two-sided':
                    z_alpha = stats.norm.ppf(1 - alpha/2)
                    z_beta = stats.norm.ppf(power)
                else:
                    z_alpha = stats.norm.ppf(1 - alpha)
                    z_beta = stats.norm.ppf(power)
                    
                n = ((z_alpha + z_beta) / effect_size) ** 2
                return int(np.ceil(n))
                
        else:
            # For non-parametric tests, use t-test approximation with ARE adjustment
            # Mann-Whitney and Wilcoxon have ARE ≈ 0.955 vs t-test
            are = 0.955
            adjusted_n = self.sample_size_calculation(
                effect_size, power, alpha, 't-test', alternative
            )
            return int(np.ceil(adjusted_n / are))
    
    def power_analysis(
        self,
        group1: Union[np.ndarray, List[float]],
        group2: Union[np.ndarray, List[float]],
        effect_sizes: List[float] = None,
        sample_sizes: List[int] = None,
        alpha: float = None
    ) -> Dict[str, List[float]]:
        """
        Comprehensive power analysis for study planning and interpretation.
        
        Args:
            group1: First group data (for effect size estimation)
            group2: Second group data (for effect size estimation)
            effect_sizes: Effect sizes to analyze (if None, uses observed)
            sample_sizes: Sample sizes to analyze (if None, uses range)
            alpha: Significance level (uses self.alpha if None)
            
        Returns:
            Dictionary with power curves for different scenarios
        """
        if alpha is None:
            alpha = self.alpha
            
        group1 = np.asarray(group1, dtype=float)
        group2 = np.asarray(group2, dtype=float)
        
        # Estimate observed effect size if not provided
        if effect_sizes is None:
            observed_effect = self.compute_effect_size(group1, group2, compute_ci=False)
            effect_sizes = [0.2, 0.5, 0.8, observed_effect.cohens_d]
            
        if sample_sizes is None:
            current_n = min(len(group1), len(group2))
            sample_sizes = list(range(10, min(200, current_n * 3), 10))
            
        results = {
            'effect_sizes': effect_sizes,
            'sample_sizes': sample_sizes,
            'power_matrix': [],
            'required_n_80_power': []
        }
        
        # Compute power for each effect size
        for effect_size in effect_sizes:
            power_curve = []
            for n in sample_sizes:
                power = stats.ttest_power(effect_size, n, alpha)
                power_curve.append(min(1.0, max(0.0, power)))
            results['power_matrix'].append(power_curve)
            
            # Find required sample size for 80% power
            required_n = self.sample_size_calculation(effect_size, 0.8, alpha)
            results['required_n_80_power'].append(required_n)
            
        return results

    def multiple_comparisons_correction(
        self, 
        p_values: List[float], 
        method: str = 'fdr_bh',
        alpha: float = None
    ) -> MultipleComparisonResult:
        """
        Apply multiple comparisons correction.
        
        Args:
            p_values: List of p-values to correct
            method: 'bonferroni', 'holm', 'fdr_bh' (Benjamini-Hochberg), 
                   'fdr_by' (Benjamini-Yekutieli)
            alpha: Significance level (uses self.alpha if None)
            
        Returns:
            MultipleComparisonResult with corrected p-values
        """
        if alpha is None:
            alpha = self.alpha
            
        p_values = np.asarray(p_values)
        
        # Apply correction
        rejected, corrected_p_values, _, _ = multipletests(
            p_values, alpha=alpha, method=method
        )
        
        # Calculate family-wise error rate
        if method == 'bonferroni':
            fwer = min(1.0, len(p_values) * alpha)
        elif method == 'holm':
            fwer = alpha  # Controls FWER at alpha level
        elif method.startswith('fdr'):
            fwer = np.sum(rejected) / len(p_values) * alpha if len(p_values) > 0 else 0
        else:
            fwer = alpha  # Conservative estimate
        
        return MultipleComparisonResult(
            method=method,
            original_p_values=p_values.tolist(),
            corrected_p_values=corrected_p_values.tolist(),
            rejected=rejected.tolist(),
            alpha=alpha,
            n_comparisons=len(p_values),
            family_wise_error_rate=fwer
        )

    def pairwise_comparison_matrix(
        self, 
        data_dict: Dict[str, List[float]], 
        metric_name: str,
        test_type: str = 'auto',
        correction_method: str = 'fdr_bh'
    ) -> PairedComparisonMatrix:
        """
        Perform all pairwise comparisons between methods with multiple comparison correction.
        
        Args:
            data_dict: Dictionary mapping method names to performance data
            metric_name: Name of the metric being compared
            test_type: Statistical test type
            correction_method: Multiple comparison correction method
            
        Returns:
            PairedComparisonMatrix with complete comparison results
        """
        methods = list(data_dict.keys())
        n_methods = len(methods)
        
        # Initialize matrices
        p_values = np.full((n_methods, n_methods), 1.0)
        effect_sizes = np.zeros((n_methods, n_methods))
        
        # Collect all p-values for correction
        all_p_values = []
        comparison_pairs = []
        
        # Perform pairwise comparisons
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                method1, method2 = methods[i], methods[j]
                data1, data2 = data_dict[method1], data_dict[method2]
                
                # Perform statistical test
                test_result = self.statistical_test(
                    data1, data2, 
                    test_type=test_type, 
                    paired=False,  # Assuming independent methods
                    alternative='two-sided'
                )
                
                p_values[i, j] = test_result.p_value
                p_values[j, i] = test_result.p_value  # Symmetric matrix
                
                effect_sizes[i, j] = test_result.effect_size.cohens_d
                effect_sizes[j, i] = -test_result.effect_size.cohens_d  # Reverse effect
                
                all_p_values.append(test_result.p_value)
                comparison_pairs.append((i, j))
        
        # Apply multiple comparison correction
        correction_result = self.multiple_comparisons_correction(
            all_p_values, method=correction_method
        )
        
        # Update p-values matrix with corrected values
        corrected_p_values = np.full((n_methods, n_methods), 1.0)
        for idx, (i, j) in enumerate(comparison_pairs):
            corrected_p = correction_result.corrected_p_values[idx]
            corrected_p_values[i, j] = corrected_p
            corrected_p_values[j, i] = corrected_p
        
        # Create significance matrix
        significance_matrix = corrected_p_values < self.alpha
        
        return PairedComparisonMatrix(
            methods=methods,
            metric=metric_name,
            p_values=corrected_p_values,
            effect_sizes=effect_sizes,
            significance_matrix=significance_matrix,
            correction_method=correction_method,
            alpha=self.alpha
        )

    def comprehensive_analysis_report(
        self, 
        results_data: Dict[str, Dict[str, List[float]]],
        output_file: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive statistical analysis report.
        
        Args:
            results_data: Nested dict {method: {metric: [values]}}
            output_file: Optional file path to save JSON report
            
        Returns:
            Complete analysis report dictionary
        """
        self.logger.info("Generating comprehensive statistical analysis report...")
        
        report = {
            "analysis_metadata": {
                "bootstrap_samples": self.bootstrap_samples,
                "confidence_level": self.confidence_level,
                "alpha": self.alpha,
                "random_state": self.random_state,
                "timestamp": pd.Timestamp.now().isoformat()
            },
            "methods": list(results_data.keys()),
            "metrics": list(next(iter(results_data.values())).keys()),
            "bootstrap_results": {},
            "pairwise_comparisons": {},
            "effect_size_summaries": {},
            "multiple_comparison_corrections": {}
        }
        
        # Bootstrap confidence intervals for each method-metric combination
        for method, method_data in results_data.items():
            report["bootstrap_results"][method] = {}
            
            for metric, values in method_data.items():
                bootstrap_result = self.bootstrap_confidence_interval(
                    values, statistic_function=np.mean, method='BCa'
                )
                report["bootstrap_results"][method][metric] = asdict(bootstrap_result)
        
        # Pairwise comparisons for each metric
        for metric in report["metrics"]:
            self.logger.info(f"Computing pairwise comparisons for {metric}")
            
            # Extract data for this metric across all methods
            metric_data = {
                method: method_data[metric] 
                for method, method_data in results_data.items()
            }
            
            # Compute pairwise comparison matrix
            comparison_matrix = self.pairwise_comparison_matrix(
                metric_data, metric, correction_method='fdr_bh'
            )
            
            # Convert numpy arrays to lists for JSON serialization
            report["pairwise_comparisons"][metric] = {
                "methods": comparison_matrix.methods,
                "p_values": comparison_matrix.p_values.tolist(),
                "effect_sizes": comparison_matrix.effect_sizes.tolist(),
                "significance_matrix": comparison_matrix.significance_matrix.tolist(),
                "correction_method": comparison_matrix.correction_method,
                "alpha": comparison_matrix.alpha
            }
            
            # Effect size summary statistics
            effect_sizes_flat = comparison_matrix.effect_sizes[
                np.triu_indices_from(comparison_matrix.effect_sizes, k=1)
            ]
            
            report["effect_size_summaries"][metric] = {
                "mean_effect_size": float(np.mean(np.abs(effect_sizes_flat))),
                "median_effect_size": float(np.median(np.abs(effect_sizes_flat))),
                "max_effect_size": float(np.max(np.abs(effect_sizes_flat))),
                "n_large_effects": int(np.sum(np.abs(effect_sizes_flat) >= 0.8)),
                "n_medium_effects": int(np.sum((np.abs(effect_sizes_flat) >= 0.5) & 
                                             (np.abs(effect_sizes_flat) < 0.8))),
                "n_small_effects": int(np.sum((np.abs(effect_sizes_flat) >= 0.2) & 
                                            (np.abs(effect_sizes_flat) < 0.5)))
            }
        
        # Multiple comparison correction analysis
        for metric in report["metrics"]:
            p_values = np.array(report["pairwise_comparisons"][metric]["p_values"])
            # Extract upper triangle (unique comparisons)
            unique_p_values = p_values[np.triu_indices_from(p_values, k=1)]
            
            # Compare different correction methods
            correction_methods = ['bonferroni', 'holm', 'fdr_bh', 'fdr_by']
            corrections_comparison = {}
            
            for method in correction_methods:
                correction_result = self.multiple_comparisons_correction(
                    unique_p_values, method=method
                )
                corrections_comparison[method] = {
                    "n_significant": int(np.sum(correction_result.rejected)),
                    "proportion_significant": float(np.mean(correction_result.rejected)),
                    "family_wise_error_rate": correction_result.family_wise_error_rate
                }
            
            report["multiple_comparison_corrections"][metric] = corrections_comparison
        
        # Save report if output file specified
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Comprehensive analysis report saved to {output_file}")
        
        self.logger.info("Comprehensive statistical analysis completed")
        return report

def main():
    """Example usage and testing of the statistical analysis framework"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Statistical Analysis Framework")
    parser.add_argument("--demo", action="store_true", help="Run demonstration analysis")
    parser.add_argument("--data-file", help="JSON file with results data")
    parser.add_argument("--output", help="Output file for analysis report")
    parser.add_argument("--bootstrap-samples", type=int, default=10000, 
                       help="Number of bootstrap samples")
    parser.add_argument("--alpha", type=float, default=0.05, 
                       help="Significance level")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = AdvancedStatisticalAnalyzer(
        bootstrap_samples=args.bootstrap_samples,
        alpha=args.alpha
    )
    
    if args.demo:
        # Generate demonstration data
        np.random.seed(42)
        
        demo_data = {
            "baseline_bm25": {
                "ndcg_at_10": np.random.normal(0.45, 0.1, 50).tolist(),
                "recall_at_50": np.random.normal(0.65, 0.15, 50).tolist(),
                "latency_p95": np.random.normal(150, 30, 50).tolist()
            },
            "lethe_hybrid": {
                "ndcg_at_10": np.random.normal(0.72, 0.12, 50).tolist(),
                "recall_at_50": np.random.normal(0.85, 0.12, 50).tolist(),
                "latency_p95": np.random.normal(280, 50, 50).tolist()
            },
            "lethe_full": {
                "ndcg_at_10": np.random.normal(0.78, 0.10, 50).tolist(),
                "recall_at_50": np.random.normal(0.88, 0.10, 50).tolist(),
                "latency_p95": np.random.normal(450, 80, 50).tolist()
            }
        }
        
        # Generate comprehensive analysis
        report = analyzer.comprehensive_analysis_report(
            demo_data, 
            output_file=Path(args.output) if args.output else None
        )
        
        print("🧮 Advanced Statistical Analysis Complete!")
        print(f"📊 Methods analyzed: {len(report['methods'])}")
        print(f"📈 Metrics analyzed: {len(report['metrics'])}")
        print(f"🔬 Bootstrap samples: {report['analysis_metadata']['bootstrap_samples']:,}")
        
        # Show key results
        for metric in report["metrics"]:
            print(f"\n📋 {metric.upper()} Results:")
            effect_summary = report["effect_size_summaries"][metric]
            print(f"  Mean effect size: {effect_summary['mean_effect_size']:.3f}")
            print(f"  Large effects: {effect_summary['n_large_effects']}")
            
            corrections = report["multiple_comparison_corrections"][metric]
            print(f"  Significant (FDR): {corrections['fdr_bh']['n_significant']}")
            print(f"  Significant (Bonferroni): {corrections['bonferroni']['n_significant']}")
    
    elif args.data_file:
        # Load and analyze real data
        with open(args.data_file, 'r') as f:
            data = json.load(f)
            
        report = analyzer.comprehensive_analysis_report(
            data,
            output_file=Path(args.output) if args.output else None
        )
        
        print(f"Analysis complete! Report {'saved to ' + args.output if args.output else 'generated'}")
    
    else:
        print("Please specify --demo or --data-file")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())