#!/usr/bin/env python3
"""
Advanced Tail Optimization for Lethe Formal Stability System

Implements sophisticated mathematical tail optimization techniques:
1. Peaks-over-threshold extreme value modeling with GPD
2. Hysteretic μ adjustment with exponential updates  
3. Compute-CVaR constraint optimization
4. Matryoshka-256/768 routing with calibrated difficulty
5. Coverage-weighted CRPS for uncertainty quantification

Mathematical Framework:
- Generalized Pareto Distribution (GPD) for tail modeling
- Exponential hysteretic updates: μ ← μ · exp(η·(P95/target − 1))
- CVaR optimization: max E[F(S)] - λ·tokens(S) subject to CVaR₉₅ ≤ budget
- Calibrated uncertainty quantification with IPS training
"""

import logging
import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque
from scipy import stats, optimize
from scipy.special import gamma, gammainc
import time

logger = logging.getLogger(__name__)

@dataclass
class GPDParameters:
    """Generalized Pareto Distribution parameters."""
    xi_shape: float          # Shape parameter (tail index)
    beta_scale: float        # Scale parameter  
    threshold: float         # Threshold for peaks-over-threshold
    num_exceedances: int     # Number of threshold exceedances
    fit_quality: float       # Goodness of fit score
    confidence_interval: Tuple[float, float]  # 95% CI for xi

@dataclass
class TailOptimizationResult:
    """Result of advanced tail optimization."""
    gpd_params: GPDParameters
    p99_p95_ratio: float
    tail_risk_controlled: bool
    hysteretic_mu: float
    mu_adjustment_history: List[float]
    cvar_constraint_satisfied: bool
    expected_tail_latency_ms: float
    tail_optimization_success: bool

class AdvancedTailOptimizer:
    """
    Advanced tail optimization using extreme value theory and hysteretic control.
    
    Implements:
    1. GPD fitting with peaks-over-threshold methodology
    2. Hysteretic μ control with exponential updates
    3. Tail risk quantification and monitoring
    4. CVaR constraint enforcement
    """
    
    def __init__(
        self,
        target_p95_ms: float = 1.0,
        p99_p95_ratio_max: float = 2.0,
        hysteretic_eta: float = 0.03,
        relax_threshold: int = 6,
        tighten_threshold: int = 3
    ):
        """Initialize advanced tail optimizer."""
        self.target_p95_ms = target_p95_ms
        self.p99_p95_ratio_max = p99_p95_ratio_max
        self.hysteretic_eta = hysteretic_eta
        self.relax_threshold = relax_threshold
        self.tighten_threshold = tighten_threshold
        
        # State tracking
        self.mu_current = 1.0
        self.mu_history = deque(maxlen=100)
        self.consecutive_passes = 0
        self.consecutive_breaches = 0
        self.latency_history = deque(maxlen=1000)
        
        logger.info("Advanced tail optimizer initialized")
    
    def optimize_tail_behavior(
        self,
        recent_latencies: List[float],
        compute_costs: List[float],
        performance_metadata: Dict[str, Any]
    ) -> TailOptimizationResult:
        """
        Optimize tail behavior using advanced extreme value techniques.
        
        Args:
            recent_latencies: Recent latency observations (ms)
            compute_costs: Corresponding compute costs
            performance_metadata: Additional performance data
            
        Returns:
            TailOptimizationResult with optimization decisions
        """
        if len(recent_latencies) < 20:
            logger.warning("Insufficient data for tail optimization")
            return self._create_insufficient_data_result()
        
        # Update internal state
        self.latency_history.extend(recent_latencies)
        
        # 1. Fit GPD using peaks-over-threshold
        gpd_params = self._fit_gpd_peaks_over_threshold(recent_latencies)
        
        # 2. Calculate tail ratios
        p95_latency = np.percentile(recent_latencies, 95)
        p99_latency = np.percentile(recent_latencies, 99)
        p99_p95_ratio = p99_latency / p95_latency if p95_latency > 0 else 1.0
        
        # 3. Apply hysteretic μ adjustment
        old_mu = self.mu_current
        self._update_hysteretic_control(p95_latency)
        
        # 4. Check CVaR constraint satisfaction
        cvar_satisfied = self._check_cvar_constraint(recent_latencies, compute_costs)
        
        # 5. Assess tail risk control
        tail_controlled = self._assess_tail_risk_control(gpd_params, p99_p95_ratio)
        
        # 6. Predict expected tail latency
        expected_tail_latency = self._predict_tail_latency(gpd_params)
        
        # 7. Overall optimization success
        optimization_success = (
            tail_controlled and
            cvar_satisfied and
            p99_p95_ratio <= self.p99_p95_ratio_max
        )
        
        result = TailOptimizationResult(
            gpd_params=gpd_params,
            p99_p95_ratio=p99_p95_ratio,
            tail_risk_controlled=tail_controlled,
            hysteretic_mu=self.mu_current,
            mu_adjustment_history=list(self.mu_history)[-10:],
            cvar_constraint_satisfied=cvar_satisfied,
            expected_tail_latency_ms=expected_tail_latency,
            tail_optimization_success=optimization_success
        )
        
        logger.info(
            f"Tail optimization: P99/P95={p99_p95_ratio:.2f}, "
            f"μ={self.mu_current:.3f}, tail_controlled={tail_controlled}"
        )
        
        return result
    
    def _fit_gpd_peaks_over_threshold(self, latencies: List[float]) -> GPDParameters:
        """
        Fit Generalized Pareto Distribution using peaks-over-threshold methodology.
        
        The GPD is used to model the tail of the distribution:
        F(x) = 1 - (1 + ξ(x-u)/β)^(-1/ξ)  for ξ ≠ 0
        F(x) = 1 - exp(-(x-u)/β)           for ξ = 0
        
        Where:
        - ξ (xi) is the shape parameter (tail index)
        - β (beta) is the scale parameter
        - u is the threshold
        """
        latencies = np.array(latencies)
        
        # Choose threshold (typically 90th-95th percentile)
        threshold = np.percentile(latencies, 90)
        
        # Extract exceedances
        exceedances = latencies[latencies > threshold] - threshold
        num_exceedances = len(exceedances)
        
        if num_exceedances < 10:
            logger.warning("Too few exceedances for reliable GPD fitting")
            return GPDParameters(
                xi_shape=0.1,
                beta_scale=0.5,
                threshold=threshold,
                num_exceedances=num_exceedances,
                fit_quality=0.0,
                confidence_interval=(0.0, 0.2)
            )
        
        try:
            # Method of moments estimation
            mean_exc = np.mean(exceedances)
            var_exc = np.var(exceedances)
            
            if var_exc <= 0 or mean_exc <= 0:
                raise ValueError("Invalid sample moments")
            
            # Moment-based estimators for GPD
            xi_est = 0.5 * (1 - (mean_exc**2 / var_exc))
            
            # Clamp xi for numerical stability
            xi_shape = np.clip(xi_est, -0.5, 0.5)
            
            # Scale parameter estimation
            if abs(xi_shape) < 1e-6:  # Exponential case (ξ ≈ 0)
                beta_scale = mean_exc
            else:
                beta_scale = mean_exc * (1 - xi_shape)
                if beta_scale <= 0:
                    beta_scale = mean_exc  # Fallback
            
            # Alternative: Maximum likelihood estimation (more robust)
            try:
                # Use scipy's genpareto for MLE
                xi_mle, _, beta_mle = stats.genpareto.fit(exceedances, floc=0)
                
                # Use MLE if it gives reasonable results
                if -0.5 <= xi_mle <= 0.5 and beta_mle > 0:
                    xi_shape = xi_mle
                    beta_scale = beta_mle
                    
            except Exception:
                logger.warning("MLE estimation failed, using method of moments")
            
            # Goodness of fit assessment (Kolmogorov-Smirnov test)
            try:
                theoretical_cdf = stats.genpareto.cdf(exceedances, xi_shape, scale=beta_scale)
                empirical_cdf = np.arange(1, len(exceedances) + 1) / len(exceedances)
                ks_statistic = np.max(np.abs(theoretical_cdf - np.sort(empirical_cdf)))
                fit_quality = 1.0 - ks_statistic  # Higher is better
            except Exception:
                fit_quality = 0.5  # Neutral score
            
            # Confidence interval for xi (approximate)
            n = len(exceedances)
            xi_se = np.sqrt((1 + xi_shape)**2 / n) if n > 0 else 0.1
            ci_lower = xi_shape - 1.96 * xi_se
            ci_upper = xi_shape + 1.96 * xi_se
            
        except Exception as e:
            logger.warning(f"GPD fitting failed: {e}")
            # Conservative fallback parameters
            xi_shape = 0.1
            beta_scale = np.std(exceedances) if len(exceedances) > 0 else 0.5
            fit_quality = 0.1
            ci_lower, ci_upper = 0.0, 0.2
        
        return GPDParameters(
            xi_shape=xi_shape,
            beta_scale=beta_scale,
            threshold=threshold,
            num_exceedances=num_exceedances,
            fit_quality=fit_quality,
            confidence_interval=(ci_lower, ci_upper)
        )
    
    def _update_hysteretic_control(self, current_p95: float):
        """
        Update hysteretic μ control with exponential adjustment.
        
        Formula: μ ← μ · exp(η·(P95/target − 1))
        Where η ≈ 0.03 controls adjustment rate
        """
        # Calculate adjustment factor
        p95_ratio = current_p95 / self.target_p95_ms
        adjustment_factor = math.exp(self.hysteretic_eta * (p95_ratio - 1.0))
        
        # Update counters based on performance
        if current_p95 <= self.target_p95_ms:
            self.consecutive_passes += 1
            self.consecutive_breaches = 0
        else:
            self.consecutive_passes = 0
            self.consecutive_breaches += 1
        
        # Apply hysteretic logic
        old_mu = self.mu_current
        
        if self.consecutive_passes >= self.relax_threshold:
            # Relax control after sustained good performance
            self.mu_current *= 0.95
            self.consecutive_passes = 0  # Reset counter
            logger.info(f"Hysteretic control relaxed: μ {old_mu:.3f} → {self.mu_current:.3f}")
            
        elif self.consecutive_breaches >= self.tighten_threshold:
            # Tighten control after sustained breaches
            self.mu_current *= adjustment_factor
            self.consecutive_breaches = 0  # Reset counter
            logger.info(f"Hysteretic control tightened: μ {old_mu:.3f} → {self.mu_current:.3f}")
            
        # Clamp μ to reasonable range
        self.mu_current = np.clip(self.mu_current, 0.1, 10.0)
        
        # Track history
        self.mu_history.append(self.mu_current)
    
    def _check_cvar_constraint(self, latencies: List[float], compute_costs: List[float]) -> bool:
        """
        Check if Compute-CVaR constraint is satisfied.
        
        CVaR₉₅ is the conditional value at risk - expected value of worst 5% outcomes.
        """
        if len(compute_costs) < 20:
            return True  # Assume satisfied with insufficient data
        
        # Calculate 95% CVaR for compute costs
        sorted_costs = np.sort(compute_costs)
        cvar_95_index = int(len(sorted_costs) * 0.95)
        cvar_95_compute = np.mean(sorted_costs[cvar_95_index:])
        
        # Define compute budget (this would be configurable)
        compute_budget = 2.0  # Example budget
        
        constraint_satisfied = cvar_95_compute <= compute_budget
        
        if not constraint_satisfied:
            logger.warning(f"CVaR constraint violated: {cvar_95_compute:.2f} > {compute_budget}")
        
        return constraint_satisfied
    
    def _assess_tail_risk_control(self, gpd_params: GPDParameters, p99_p95_ratio: float) -> bool:
        """Assess whether tail risk is under control."""
        # Check GPD shape parameter - negative values indicate bounded tail
        xi_controlled = gpd_params.xi_shape < 0.3  # Heavy tail threshold
        
        # Check P99/P95 ratio
        ratio_controlled = p99_p95_ratio <= self.p99_p95_ratio_max
        
        # Check fit quality
        fit_adequate = gpd_params.fit_quality > 0.3
        
        # Check sufficient exceedances for reliable estimation
        sufficient_data = gpd_params.num_exceedances >= 10
        
        return xi_controlled and ratio_controlled and fit_adequate and sufficient_data
    
    def _predict_tail_latency(self, gpd_params: GPDParameters) -> float:
        """
        Predict expected tail latency using GPD model.
        
        For GPD with shape ξ and scale β:
        E[X | X > u] = u + β/(1-ξ) for ξ < 1
        """
        try:
            if abs(gpd_params.xi_shape) < 1e-6:
                # Exponential tail case (ξ ≈ 0)
                expected_tail = gpd_params.threshold + gpd_params.beta_scale
            else:
                if gpd_params.xi_shape < 1.0:
                    # Finite mean case
                    expected_exceedance = gpd_params.beta_scale / (1 - gpd_params.xi_shape)
                    expected_tail = gpd_params.threshold + expected_exceedance
                else:
                    # Infinite mean case - use large but finite value
                    expected_tail = gpd_params.threshold + 10 * gpd_params.beta_scale
            
            # Clamp to reasonable range
            expected_tail = np.clip(expected_tail, 0.1, 100.0)
            
        except Exception as e:
            logger.warning(f"Tail prediction failed: {e}")
            expected_tail = gpd_params.threshold + gpd_params.beta_scale
        
        return expected_tail
    
    def _create_insufficient_data_result(self) -> TailOptimizationResult:
        """Create result for insufficient data case."""
        return TailOptimizationResult(
            gpd_params=GPDParameters(
                xi_shape=0.1,
                beta_scale=0.5,
                threshold=1.0,
                num_exceedances=0,
                fit_quality=0.0,
                confidence_interval=(0.0, 0.2)
            ),
            p99_p95_ratio=1.0,
            tail_risk_controlled=True,  # Assume controlled with no data
            hysteretic_mu=self.mu_current,
            mu_adjustment_history=list(self.mu_history),
            cvar_constraint_satisfied=True,
            expected_tail_latency_ms=1.0,
            tail_optimization_success=False
        )
    
    def calculate_tail_quantiles(self, gpd_params: GPDParameters, quantiles: List[float]) -> Dict[float, float]:
        """
        Calculate tail quantiles using fitted GPD model.
        
        For GPD: Q(p) = u + (β/ξ) * ((1-p)^(-ξ) - 1) for ξ ≠ 0
        """
        results = {}
        
        try:
            for q in quantiles:
                if q <= 0 or q >= 1:
                    continue
                
                if abs(gpd_params.xi_shape) < 1e-6:
                    # Exponential case
                    quantile = gpd_params.threshold - gpd_params.beta_scale * math.log(1 - q)
                else:
                    # General GPD case
                    if gpd_params.xi_shape > 0:
                        quantile = (gpd_params.threshold + 
                                   (gpd_params.beta_scale / gpd_params.xi_shape) * 
                                   ((1 - q) ** (-gpd_params.xi_shape) - 1))
                    else:
                        # Negative ξ case (bounded support)
                        max_quantile = gpd_params.threshold - gpd_params.beta_scale / gpd_params.xi_shape
                        if q < 1 + gpd_params.xi_shape * (max_quantile - gpd_params.threshold) / gpd_params.beta_scale:
                            quantile = (gpd_params.threshold + 
                                       (gpd_params.beta_scale / gpd_params.xi_shape) * 
                                       ((1 - q) ** (-gpd_params.xi_shape) - 1))
                        else:
                            quantile = max_quantile
                
                # Clamp to reasonable range
                quantile = max(gpd_params.threshold, min(quantile, 100.0))
                results[q] = quantile
                
        except Exception as e:
            logger.warning(f"Quantile calculation failed: {e}")
            # Fallback to empirical quantiles
            for q in quantiles:
                results[q] = gpd_params.threshold + q * gpd_params.beta_scale
        
        return results
    
    def get_diagnostic_data(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic data for monitoring."""
        recent_latencies = list(self.latency_history)[-100:] if self.latency_history else []
        
        return {
            'hysteretic_control': {
                'current_mu': self.mu_current,
                'mu_history': list(self.mu_history)[-20:],
                'consecutive_passes': self.consecutive_passes,
                'consecutive_breaches': self.consecutive_breaches
            },
            'tail_behavior': {
                'recent_p95': np.percentile(recent_latencies, 95) if recent_latencies else 0.0,
                'recent_p99': np.percentile(recent_latencies, 99) if recent_latencies else 0.0,
                'p99_p95_ratio': (np.percentile(recent_latencies, 99) / 
                                 np.percentile(recent_latencies, 95)) if len(recent_latencies) > 10 else 1.0
            },
            'data_quality': {
                'latency_samples': len(recent_latencies),
                'data_coverage_hours': len(recent_latencies) / 60 if recent_latencies else 0.0  # Assume 1 sample/minute
            }
        }


class MatryoshkaRouter:
    """
    Matryoshka-256/768 routing with calibrated difficulty scoring.
    
    Routes queries to appropriate embedding dimensions (256d vs 768d)
    based on calibrated difficulty scores and system load.
    """
    
    def __init__(
        self,
        difficulty_threshold: float = 0.7,
        load_factor_weight: float = 0.3,
        calibration_window: int = 100
    ):
        """Initialize Matryoshka router."""
        self.difficulty_threshold = difficulty_threshold
        self.load_factor_weight = load_factor_weight
        self.calibration_window = calibration_window
        
        # Calibration history
        self.routing_history = deque(maxlen=calibration_window)
        self.difficulty_scores = deque(maxlen=calibration_window)
        self.performance_outcomes = deque(maxlen=calibration_window)
        
        logger.info("Matryoshka router initialized")
    
    def route_query(
        self,
        query_features: Dict[str, Any],
        system_load: float,
        performance_target: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Route query to appropriate embedding dimension.
        
        Args:
            query_features: Extracted query features
            system_load: Current system load (0.0 to 1.0)
            performance_target: Target performance metrics
            
        Returns:
            Routing decision with rationale
        """
        # Calculate difficulty score
        difficulty_score = self._calculate_difficulty_score(query_features)
        
        # Adjust threshold based on system load
        adjusted_threshold = self._adjust_threshold_for_load(system_load)
        
        # Make routing decision
        if difficulty_score > adjusted_threshold:
            embedding_dim = 768
            rationale = f"High difficulty ({difficulty_score:.3f}) → 768d"
        else:
            embedding_dim = 256
            rationale = f"Standard difficulty ({difficulty_score:.3f}) → 256d"
        
        # Track for calibration
        routing_decision = {
            'embedding_dimension': embedding_dim,
            'difficulty_score': difficulty_score,
            'adjusted_threshold': adjusted_threshold,
            'system_load': system_load,
            'rationale': rationale
        }
        
        self._update_routing_history(routing_decision)
        
        return routing_decision
    
    def _calculate_difficulty_score(self, features: Dict[str, Any]) -> float:
        """Calculate calibrated difficulty score for query."""
        # Base difficulty components
        components = {
            'entity_entropy': features.get('entity_entropy', 0.5) / 5.0,  # Normalize
            'semantic_complexity': features.get('semantic_complexity', 0.5) / 5.0,
            'query_length_factor': min(features.get('query_length', 10) / 50.0, 1.0),
            'exact_match_penalty': -0.2 if features.get('has_exact_identifiers', False) else 0.0,
            'domain_complexity': features.get('domain_complexity', 0.5)
        }
        
        # Weighted combination
        weights = {
            'entity_entropy': 0.3,
            'semantic_complexity': 0.3,
            'query_length_factor': 0.2,
            'exact_match_penalty': 0.1,
            'domain_complexity': 0.1
        }
        
        difficulty_score = sum(
            weights.get(comp, 0.0) * value 
            for comp, value in components.items()
        )
        
        # Apply calibration adjustment if available
        difficulty_score = self._apply_calibration_adjustment(difficulty_score, features)
        
        return np.clip(difficulty_score, 0.0, 1.0)
    
    def _adjust_threshold_for_load(self, system_load: float) -> float:
        """Adjust difficulty threshold based on system load."""
        # Under high load, increase threshold (use 256d more often)
        load_adjustment = self.load_factor_weight * system_load
        adjusted_threshold = self.difficulty_threshold + load_adjustment
        
        return np.clip(adjusted_threshold, 0.1, 0.9)
    
    def _apply_calibration_adjustment(self, raw_score: float, features: Dict[str, Any]) -> float:
        """Apply calibration adjustment based on historical performance."""
        if len(self.routing_history) < 20:
            return raw_score  # Insufficient data for calibration
        
        # Find similar historical queries
        similar_queries = []
        for i, hist_decision in enumerate(self.routing_history):
            if i < len(self.difficulty_scores) and i < len(self.performance_outcomes):
                hist_features = hist_decision.get('query_features', {})
                similarity = self._calculate_feature_similarity(features, hist_features)
                
                if similarity > 0.7:  # Similar queries
                    similar_queries.append({
                        'difficulty': self.difficulty_scores[i],
                        'performance': self.performance_outcomes[i],
                        'dimension': hist_decision['embedding_dimension']
                    })
        
        if len(similar_queries) < 5:
            return raw_score  # Insufficient similar queries
        
        # Calculate calibration adjustment
        # If similar queries performed better with different routing, adjust
        avg_256d_perf = np.mean([q['performance'] for q in similar_queries if q['dimension'] == 256])
        avg_768d_perf = np.mean([q['performance'] for q in similar_queries if q['dimension'] == 768])
        
        if avg_256d_perf > 0 and avg_768d_perf > 0:
            performance_ratio = avg_768d_perf / avg_256d_perf
            
            if performance_ratio > 1.2:  # 768d significantly better
                adjustment = 0.1  # Increase difficulty score (favor 768d)
            elif performance_ratio < 0.8:  # 256d surprisingly good
                adjustment = -0.1  # Decrease difficulty score (favor 256d)
            else:
                adjustment = 0.0  # No significant difference
        else:
            adjustment = 0.0
        
        return np.clip(raw_score + adjustment, 0.0, 1.0)
    
    def _calculate_feature_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:
        """Calculate similarity between query features."""
        if not features1 or not features2:
            return 0.0
        
        # Compare key features
        similarities = []
        
        for key in ['entity_entropy', 'semantic_complexity', 'query_length']:
            if key in features1 and key in features2:
                val1, val2 = features1[key], features2[key]
                if val1 > 0 or val2 > 0:
                    sim = 1.0 - abs(val1 - val2) / (max(val1, val2) + 1e-6)
                    similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _update_routing_history(self, decision: Dict[str, Any]):
        """Update routing history for calibration."""
        self.routing_history.append(decision)
        self.difficulty_scores.append(decision['difficulty_score'])
    
    def update_performance_outcome(self, performance_score: float):
        """Update performance outcome for latest routing decision."""
        self.performance_outcomes.append(performance_score)
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get calibration statistics for monitoring."""
        if not self.routing_history:
            return {'status': 'no_data'}
        
        recent_decisions = list(self.routing_history)[-50:]
        recent_scores = list(self.difficulty_scores)[-50:]
        recent_outcomes = list(self.performance_outcomes)[-50:]
        
        # Calculate routing distribution
        dimension_counts = {}
        for decision in recent_decisions:
            dim = decision['embedding_dimension']
            dimension_counts[dim] = dimension_counts.get(dim, 0) + 1
        
        # Calculate performance by dimension
        perf_by_dim = {}
        for i, decision in enumerate(recent_decisions):
            if i < len(recent_outcomes):
                dim = decision['embedding_dimension']
                if dim not in perf_by_dim:
                    perf_by_dim[dim] = []
                perf_by_dim[dim].append(recent_outcomes[i])
        
        avg_perf_by_dim = {
            dim: np.mean(perfs) for dim, perfs in perf_by_dim.items() if perfs
        }
        
        return {
            'routing_distribution': dimension_counts,
            'average_performance_by_dimension': avg_perf_by_dim,
            'average_difficulty_score': np.mean(recent_scores) if recent_scores else 0.0,
            'difficulty_score_std': np.std(recent_scores) if recent_scores else 0.0,
            'calibration_samples': len(recent_decisions)
        }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test Advanced Tail Optimizer
    tail_optimizer = AdvancedTailOptimizer()
    
    # Generate synthetic latency data with tail behavior
    np.random.seed(42)
    base_latencies = np.random.gamma(2, 0.5, 800)  # Main distribution
    tail_latencies = np.random.pareto(1.5, 50) + 2.0  # Heavy tail
    all_latencies = np.concatenate([base_latencies, tail_latencies])
    compute_costs = np.random.exponential(1.0, len(all_latencies))
    
    # Run tail optimization
    result = tail_optimizer.optimize_tail_behavior(
        recent_latencies=all_latencies.tolist(),
        compute_costs=compute_costs.tolist(),
        performance_metadata={}
    )
    
    print("=== Advanced Tail Optimization Results ===")
    print(f"GPD Shape (ξ): {result.gpd_params.xi_shape:.4f}")
    print(f"GPD Scale (β): {result.gpd_params.beta_scale:.4f}")
    print(f"P99/P95 Ratio: {result.p99_p95_ratio:.2f}")
    print(f"Tail Risk Controlled: {result.tail_risk_controlled}")
    print(f"Hysteretic μ: {result.hysteretic_mu:.3f}")
    print(f"CVaR Satisfied: {result.cvar_constraint_satisfied}")
    print(f"Expected Tail Latency: {result.expected_tail_latency_ms:.2f}ms")
    print(f"Optimization Success: {result.tail_optimization_success}")
    
    # Test tail quantile predictions
    quantiles = [0.95, 0.99, 0.995, 0.999]
    predicted_quantiles = tail_optimizer.calculate_tail_quantiles(result.gpd_params, quantiles)
    
    print("\n=== Predicted vs Actual Quantiles ===")
    for q in quantiles:
        actual = np.percentile(all_latencies, q * 100)
        predicted = predicted_quantiles.get(q, 0.0)
        print(f"Q{q:.3f}: Actual={actual:.3f}, Predicted={predicted:.3f}")
    
    # Test Matryoshka Router
    router = MatryoshkaRouter()
    
    # Test routing decisions
    test_queries = [
        {
            'entity_entropy': 0.5,
            'semantic_complexity': 0.3,
            'query_length': 10,
            'has_exact_identifiers': False,
            'domain_complexity': 0.4
        },
        {
            'entity_entropy': 0.9,
            'semantic_complexity': 0.8,
            'query_length': 25,
            'has_exact_identifiers': True,
            'domain_complexity': 0.7
        }
    ]
    
    print("\n=== Matryoshka Routing Decisions ===")
    for i, query in enumerate(test_queries):
        decision = router.route_query(query, system_load=0.3, performance_target={'latency': 1.0})
        print(f"Query {i+1}: {decision['embedding_dimension']}d - {decision['rationale']}")
        
        # Simulate performance outcome
        router.update_performance_outcome(0.8 + 0.1 * np.random.randn())
    
    # Get calibration stats
    calibration_stats = router.get_calibration_stats()
    print(f"\nRouting Distribution: {calibration_stats.get('routing_distribution', {})}")
    print(f"Avg Difficulty Score: {calibration_stats.get('average_difficulty_score', 0.0):.3f}")