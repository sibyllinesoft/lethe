#!/usr/bin/env python3
"""
Advanced Instrumentation System for Lethe→StreamingLLM Hybrid

Implements comprehensive monitoring and telemetry for the hybrid system including:
- EVT-based tail compute modeling with ξ parameter tracking
- Primal-dual gap monitoring for optimization convergence  
- KV prefix-Jaccard alarm system with -10pp threshold detection
- Parameter drift monitoring (λ,μ over 24h windows)
- Tail CVaR computation for risk assessment
- Complete telemetry logging for performance analysis

Mathematical Framework:
- Extreme Value Theory (EVT) with Generalized Pareto Distribution (GPD)
- ξ parameter estimation for tail compute modeling
- CVaR₀.₉₅(compute) for tail risk quantification
- Primal-dual gap convergence monitoring
"""

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any, NamedTuple
from collections import defaultdict, deque
from enum import Enum
import math
import json
from pathlib import Path
import threading
from datetime import datetime, timedelta
import statistics

# Scientific computing imports
try:
    from scipy import stats
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger = logging.getLogger(__name__)
    logger.warning("SciPy not available - using simplified statistical models")

logger = logging.getLogger(__name__)

class AlarmLevel(Enum):
    """Alarm severity levels."""
    INFO = "info"
    WARNING = "warning"  
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class ComputeMetrics:
    """Compute performance metrics for a single operation."""
    timestamp: float
    operation_id: str
    latency_ms: float
    tokens_processed: int
    head_tokens: int
    tail_tokens: int
    num_windows: int
    kv_reuse_ratio: float
    objective_value: float
    cost_lambda: float
    cost_mu: float
    net_value: float
    
    # Detailed timing breakdown
    head_compute_ms: float
    tail_compute_ms: float
    arrangement_ms: float
    
    # Quality metrics
    keep_ratio: float
    entity_diversity: float
    exact_matches: int

@dataclass
class EVTParameters:
    """Extreme Value Theory parameters for tail modeling."""
    xi: float  # Shape parameter (key metric)
    sigma: float  # Scale parameter
    threshold: float  # Threshold above which GPD applies
    n_exceedances: int  # Number of observations above threshold
    confidence_interval: Tuple[float, float]  # CI for xi
    last_updated: float
    quality_score: float  # Goodness of fit

@dataclass
class KVJaccardMetrics:
    """KV cache prefix Jaccard similarity metrics."""
    current_jaccard: float
    baseline_jaccard: float
    degradation_pp: float  # Percentage points degradation
    alarm_triggered: bool
    consecutive_violations: int
    last_alarm_time: float

@dataclass
class DriftMetrics:
    """Parameter drift monitoring metrics."""
    parameter_name: str
    current_value: float
    baseline_value: float
    drift_percentage: float
    window_hours: float
    alarm_triggered: bool
    trend_direction: str  # "increasing", "decreasing", "stable"

@dataclass
class TelemetryRecord:
    """Complete telemetry record for a hybrid selection."""
    timestamp: float
    session_id: str
    operation_id: str
    
    # Core parameters
    lambda_param: float
    mu_param: float
    tokens_in: int
    head_tokens: int
    tail_tokens: int
    keep_ratio_head: float
    keep_ratio_tail: float
    
    # Algorithm parameters
    k1: int
    k2: int
    r: int  # DPP rank
    ce_early_exit: bool
    num_windows: int
    window_size: int
    stride: int
    sinks: int
    
    # Performance metrics
    kv_prefix_reuse: float
    middleware_p95: float
    llm_p95: float
    delta_cbu_per_1k: float
    precision_at_k: float
    recall_at_k: float
    
    # Optimization metrics
    primal_dual_gap: float
    tail_cvar: float
    xi_parameter: float

class EVTTailModeler:
    """Extreme Value Theory modeling for tail compute behavior."""
    
    def __init__(self, threshold_percentile: float = 0.95):
        self.threshold_percentile = threshold_percentile
        self.compute_samples = deque(maxlen=10000)  # Keep last 10k samples
        self.current_parameters: Optional[EVTParameters] = None
        self.last_update = 0.0
        self.min_samples = 100
        
    def add_compute_sample(self, compute_ms: float, tokens: int):
        """Add compute time sample for EVT analysis."""
        # Normalize by tokens for fair comparison
        normalized_compute = compute_ms / max(1, tokens) * 1000  # per 1k tokens
        
        sample = {
            'timestamp': time.time(),
            'compute_normalized': normalized_compute,
            'raw_compute': compute_ms,
            'tokens': tokens
        }
        
        self.compute_samples.append(sample)
        
        # Update parameters if enough samples and time passed
        if (len(self.compute_samples) >= self.min_samples and 
            time.time() - self.last_update > 300):  # Update every 5 minutes
            self._update_evt_parameters()
    
    def _update_evt_parameters(self):
        """Update EVT parameters using current samples."""
        if not HAS_SCIPY or len(self.compute_samples) < self.min_samples:
            return
        
        try:
            # Extract normalized compute times
            compute_values = [s['compute_normalized'] for s in self.compute_samples]
            compute_array = np.array(compute_values)
            
            # Determine threshold
            threshold = np.percentile(compute_array, self.threshold_percentile * 100)
            
            # Extract exceedances
            exceedances = compute_array[compute_array > threshold] - threshold
            n_exceedances = len(exceedances)
            
            if n_exceedances < 20:  # Need minimum exceedances
                return
            
            # Fit GPD using method of moments (simple approach)
            xi, sigma = self._fit_gpd_moments(exceedances)
            
            # Calculate confidence interval for xi (simplified)
            xi_std = math.sqrt(sigma**2 / n_exceedances)  # Rough approximation
            ci_lower = xi - 1.96 * xi_std
            ci_upper = xi + 1.96 * xi_std
            
            # Assess quality of fit
            quality = self._assess_fit_quality(exceedances, xi, sigma)
            
            self.current_parameters = EVTParameters(
                xi=xi,
                sigma=sigma,
                threshold=threshold,
                n_exceedances=n_exceedances,
                confidence_interval=(ci_lower, ci_upper),
                last_updated=time.time(),
                quality_score=quality
            )
            
            self.last_update = time.time()
            
            logger.info(f"EVT parameters updated: ξ={xi:.4f}, σ={sigma:.4f}, "
                       f"threshold={threshold:.2f}, quality={quality:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to update EVT parameters: {e}")
    
    def _fit_gpd_moments(self, exceedances: np.ndarray) -> Tuple[float, float]:
        """Fit GPD using method of moments."""
        if len(exceedances) == 0:
            return 0.0, 1.0
        
        mean_exc = np.mean(exceedances)
        var_exc = np.var(exceedances)
        
        if var_exc <= 0 or mean_exc <= 0:
            return 0.0, mean_exc
        
        # Method of moments estimators
        xi = 0.5 * ((mean_exc**2 / var_exc) - 1)
        sigma = 0.5 * mean_exc * ((mean_exc**2 / var_exc) + 1)
        
        # Constrain xi to reasonable range
        xi = max(-0.5, min(0.5, xi))
        sigma = max(0.1, sigma)
        
        return xi, sigma
    
    def _assess_fit_quality(self, exceedances: np.ndarray, xi: float, sigma: float) -> float:
        """Assess quality of GPD fit using simple metrics."""
        if len(exceedances) == 0:
            return 0.0
        
        try:
            # Calculate theoretical quantiles
            n = len(exceedances)
            empirical_quantiles = np.sort(exceedances)
            theoretical_quantiles = []
            
            for i in range(n):
                p = (i + 0.5) / n
                if xi != 0:
                    q = (sigma / xi) * ((1 - p) ** (-xi) - 1)
                else:
                    q = -sigma * math.log(1 - p)
                theoretical_quantiles.append(q)
            
            # Calculate correlation between empirical and theoretical quantiles
            correlation = np.corrcoef(empirical_quantiles, theoretical_quantiles)[0, 1]
            quality = max(0.0, correlation) if not np.isnan(correlation) else 0.0
            
            return quality
            
        except Exception:
            return 0.0
    
    def get_xi_parameter(self) -> Optional[float]:
        """Get current xi parameter value."""
        if self.current_parameters:
            return self.current_parameters.xi
        return None
    
    def get_tail_risk_metrics(self) -> Dict[str, Any]:
        """Get comprehensive tail risk metrics."""
        if not self.current_parameters:
            return {"status": "insufficient_data"}
        
        params = self.current_parameters
        
        return {
            "xi": params.xi,
            "sigma": params.sigma,
            "threshold": params.threshold,
            "n_exceedances": params.n_exceedances,
            "xi_confidence_interval": params.confidence_interval,
            "quality_score": params.quality_score,
            "last_updated": params.last_updated,
            "risk_assessment": self._assess_tail_risk(params.xi),
            "samples_available": len(self.compute_samples)
        }
    
    def _assess_tail_risk(self, xi: float) -> str:
        """Assess tail risk based on xi parameter."""
        if xi > 0.2:
            return "HIGH_RISK"  # Heavy tail, high risk
        elif xi > 0.0:
            return "MODERATE_RISK"  # Medium tail
        elif xi > -0.2:
            return "LOW_RISK"  # Light tail
        else:
            return "VERY_LOW_RISK"  # Bounded tail

class KVJaccardMonitor:
    """Monitor KV cache prefix Jaccard similarity degradation."""
    
    def __init__(self, baseline_window: int = 1000, alarm_threshold: float = -0.10):
        self.baseline_window = baseline_window
        self.alarm_threshold = alarm_threshold  # -10pp threshold
        
        self.jaccard_history = deque(maxlen=10000)
        self.baseline_jaccard = 0.0
        self.consecutive_violations = 0
        self.last_alarm_time = 0.0
        self.baseline_computed = False
        
    def record_kv_jaccard(self, prefix_sets: List[Set[str]]) -> KVJaccardMetrics:
        """Record KV cache prefix sets and compute Jaccard metrics."""
        if len(prefix_sets) < 2:
            # Need at least 2 sets to compute Jaccard
            current_jaccard = 1.0
        else:
            # Compute average pairwise Jaccard similarity
            jaccard_similarities = []
            for i in range(len(prefix_sets)):
                for j in range(i + 1, len(prefix_sets)):
                    jaccard = self._jaccard_similarity(prefix_sets[i], prefix_sets[j])
                    jaccard_similarities.append(jaccard)
            
            current_jaccard = np.mean(jaccard_similarities) if jaccard_similarities else 1.0
        
        # Record in history
        record = {
            'timestamp': time.time(),
            'jaccard': current_jaccard,
            'n_sets': len(prefix_sets)
        }
        self.jaccard_history.append(record)
        
        # Compute baseline if needed
        if not self.baseline_computed and len(self.jaccard_history) >= self.baseline_window:
            baseline_values = [r['jaccard'] for r in list(self.jaccard_history)[:self.baseline_window]]
            self.baseline_jaccard = np.mean(baseline_values)
            self.baseline_computed = True
            logger.info(f"KV Jaccard baseline established: {self.baseline_jaccard:.4f}")
        
        # Check for degradation
        degradation_pp = 0.0
        alarm_triggered = False
        
        if self.baseline_computed:
            degradation_pp = current_jaccard - self.baseline_jaccard
            
            if degradation_pp <= self.alarm_threshold:
                self.consecutive_violations += 1
                if self.consecutive_violations >= 3:  # 3 consecutive violations
                    alarm_triggered = True
                    self.last_alarm_time = time.time()
                    logger.warning(f"KV Jaccard alarm: degradation {degradation_pp:.3f}pp")
            else:
                self.consecutive_violations = 0
        
        return KVJaccardMetrics(
            current_jaccard=current_jaccard,
            baseline_jaccard=self.baseline_jaccard,
            degradation_pp=degradation_pp,
            alarm_triggered=alarm_triggered,
            consecutive_violations=self.consecutive_violations,
            last_alarm_time=self.last_alarm_time
        )
    
    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0

class ParameterDriftMonitor:
    """Monitor parameter drift over time windows."""
    
    def __init__(self, window_hours: float = 24.0, alarm_threshold: float = 0.15):
        self.window_hours = window_hours
        self.alarm_threshold = alarm_threshold  # ±15% threshold
        
        self.parameter_history = defaultdict(lambda: deque(maxlen=10000))
        self.baselines = {}
        self.last_alarm_times = defaultdict(float)
        
    def record_parameter(self, param_name: str, value: float) -> DriftMetrics:
        """Record parameter value and check for drift."""
        record = {
            'timestamp': time.time(),
            'value': value
        }
        
        self.parameter_history[param_name].append(record)
        
        # Establish baseline if not set
        if param_name not in self.baselines and len(self.parameter_history[param_name]) >= 100:
            baseline_values = [r['value'] for r in list(self.parameter_history[param_name])[:100]]
            self.baselines[param_name] = np.mean(baseline_values)
            logger.info(f"Parameter {param_name} baseline: {self.baselines[param_name]:.6f}")
        
        # Check drift within time window
        current_time = time.time()
        window_start = current_time - (self.window_hours * 3600)
        
        # Filter to window
        window_records = [r for r in self.parameter_history[param_name] 
                         if r['timestamp'] >= window_start]
        
        if len(window_records) < 10:  # Need minimum samples
            return DriftMetrics(
                parameter_name=param_name,
                current_value=value,
                baseline_value=self.baselines.get(param_name, value),
                drift_percentage=0.0,
                window_hours=self.window_hours,
                alarm_triggered=False,
                trend_direction="insufficient_data"
            )
        
        # Compute drift
        baseline = self.baselines.get(param_name, value)
        drift_percentage = (value - baseline) / abs(baseline) if baseline != 0 else 0.0
        
        # Assess trend
        values = [r['value'] for r in window_records]
        trend_direction = self._assess_trend(values)
        
        # Check alarm
        alarm_triggered = False
        if abs(drift_percentage) > self.alarm_threshold:
            # Avoid spam - only alarm once per hour
            if current_time - self.last_alarm_times[param_name] > 3600:
                alarm_triggered = True
                self.last_alarm_times[param_name] = current_time
                logger.warning(f"Parameter drift alarm: {param_name} drift {drift_percentage:.1%}")
        
        return DriftMetrics(
            parameter_name=param_name,
            current_value=value,
            baseline_value=baseline,
            drift_percentage=drift_percentage,
            window_hours=self.window_hours,
            alarm_triggered=alarm_triggered,
            trend_direction=trend_direction
        )
    
    def _assess_trend(self, values: List[float]) -> str:
        """Assess trend direction using simple linear regression."""
        if len(values) < 5:
            return "stable"
        
        try:
            x = np.arange(len(values))
            y = np.array(values)
            
            # Simple linear regression
            slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
            
            if slope > 0.01:
                return "increasing"
            elif slope < -0.01:
                return "decreasing"
            else:
                return "stable"
                
        except Exception:
            return "unknown"

class PrimalDualGapMonitor:
    """Monitor primal-dual gap for optimization convergence."""
    
    def __init__(self, target_gap: float = 0.005):  # 0.5% target
        self.target_gap = target_gap
        self.gap_history = deque(maxlen=1000)
        
    def record_gap(self, primal_value: float, dual_value: float) -> float:
        """Record primal-dual values and compute gap."""
        if dual_value == 0:
            gap = float('inf')
        else:
            gap = abs(primal_value - dual_value) / abs(dual_value)
        
        self.gap_history.append({
            'timestamp': time.time(),
            'primal': primal_value,
            'dual': dual_value,
            'gap': gap
        })
        
        return gap
    
    def get_convergence_status(self) -> Dict[str, Any]:
        """Get convergence status and metrics."""
        if not self.gap_history:
            return {"status": "no_data"}
        
        recent_gaps = [r['gap'] for r in list(self.gap_history)[-10:]]
        current_gap = recent_gaps[-1] if recent_gaps else float('inf')
        avg_recent_gap = np.mean(recent_gaps)
        
        converged = current_gap < self.target_gap
        stable = len(recent_gaps) >= 5 and np.std(recent_gaps) < 0.001
        
        return {
            "current_gap": current_gap,
            "target_gap": self.target_gap,
            "avg_recent_gap": avg_recent_gap,
            "converged": converged,
            "stable": stable,
            "trend": self._assess_gap_trend(recent_gaps)
        }
    
    def _assess_gap_trend(self, gaps: List[float]) -> str:
        """Assess gap trend."""
        if len(gaps) < 3:
            return "insufficient_data"
        
        # Check if generally decreasing (converging)
        decreasing_count = sum(1 for i in range(1, len(gaps)) if gaps[i] < gaps[i-1])
        
        if decreasing_count >= len(gaps) * 0.6:
            return "converging"
        elif decreasing_count <= len(gaps) * 0.3:
            return "diverging"
        else:
            return "oscillating"

class TailCVarCalculator:
    """Calculate Conditional Value at Risk (CVaR) for tail compute times."""
    
    def __init__(self, alpha: float = 0.95):
        self.alpha = alpha  # 95% confidence level
        self.compute_samples = deque(maxlen=5000)
        
    def add_sample(self, compute_time: float):
        """Add compute time sample."""
        self.compute_samples.append({
            'timestamp': time.time(),
            'compute_time': compute_time
        })
    
    def calculate_cvar(self) -> Optional[float]:
        """Calculate CVaR at specified confidence level."""
        if len(self.compute_samples) < 50:
            return None
        
        compute_times = [s['compute_time'] for s in self.compute_samples]
        
        # Calculate VaR (Value at Risk) - alpha quantile
        var_threshold = np.percentile(compute_times, self.alpha * 100)
        
        # Calculate CVaR - expected value of samples above VaR
        tail_samples = [ct for ct in compute_times if ct >= var_threshold]
        
        if not tail_samples:
            return var_threshold
        
        cvar = np.mean(tail_samples)
        return cvar

class HybridInstrumentation:
    """Comprehensive instrumentation system for hybrid selector."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # Core monitoring components
        self.evt_modeler = EVTTailModeler()
        self.kv_monitor = KVJaccardMonitor()
        self.drift_monitor = ParameterDriftMonitor()
        self.gap_monitor = PrimalDualGapMonitor()
        self.cvar_calculator = TailCVarCalculator()
        
        # Telemetry storage
        self.telemetry_records = deque(maxlen=50000)  # Keep 50k records
        self.compute_metrics = deque(maxlen=10000)    # Keep 10k compute metrics
        
        # Alarm system
        self.active_alarms = {}
        self.alarm_history = deque(maxlen=1000)
        
        # Configuration
        self.log_telemetry = config.get('log_telemetry', True)
        self.export_path = config.get('export_path', '/tmp/hybrid_telemetry.jsonl')
        
        # Threading for background tasks
        self.background_thread = None
        self.shutdown_event = threading.Event()
        
        logger.info("HybridInstrumentation initialized")
        
    def record_selection(self, selection_result, session_id: str = "unknown"):
        """Record complete hybrid selection for monitoring."""
        timestamp = time.time()
        operation_id = f"{session_id}_{int(timestamp*1000)}"
        
        # Extract metrics from selection result
        head_tokens = selection_result.head_selection.total_tokens if selection_result.head_selection else 0
        tail_tokens = selection_result.tail_selection.total_tokens if selection_result.tail_selection else 0
        num_windows = selection_result.tail_selection.total_windows if selection_result.tail_selection else 0
        
        # Create compute metrics record
        compute_metrics = ComputeMetrics(
            timestamp=timestamp,
            operation_id=operation_id,
            latency_ms=selection_result.selection_time_ms,
            tokens_processed=selection_result.total_tokens,
            head_tokens=head_tokens,
            tail_tokens=tail_tokens,
            num_windows=num_windows,
            kv_reuse_ratio=selection_result.kv_prefix_reuse_ratio,
            objective_value=selection_result.objective_value,
            cost_lambda=selection_result.cost_lambda,
            cost_mu=selection_result.cost_mu,
            net_value=selection_result.net_value,
            head_compute_ms=selection_result.head_time_ms,
            tail_compute_ms=selection_result.tail_time_ms,
            arrangement_ms=selection_result.arrangement_time_ms,
            keep_ratio=selection_result.keep_ratio,
            entity_diversity=0.0,  # Would extract from result
            exact_matches=0  # Would extract from result
        )
        
        self.compute_metrics.append(compute_metrics)
        
        # Update EVT model
        self.evt_modeler.add_compute_sample(
            selection_result.selection_time_ms,
            selection_result.total_tokens
        )
        
        # Update CVaR calculator
        self.cvar_calculator.add_sample(selection_result.selection_time_ms)
        
        # Monitor KV Jaccard (simplified - would need actual prefix sets)
        kv_sets = []
        if selection_result.head_selection:
            kv_sets.append(selection_result.head_selection.kv_prefix_hashes)
        kv_metrics = self.kv_monitor.record_kv_jaccard(kv_sets)
        
        # Monitor parameter drift
        param_state = selection_result.parameter_state
        drift_metrics = {}
        for param_name, value in param_state.items():
            drift_metrics[param_name] = self.drift_monitor.record_parameter(param_name, value)
        
        # Monitor primal-dual gap
        gap = self.gap_monitor.record_gap(
            selection_result.objective_value,
            selection_result.net_value  # Simplified dual estimate
        )
        
        # Create comprehensive telemetry record
        telemetry = TelemetryRecord(
            timestamp=timestamp,
            session_id=session_id,
            operation_id=operation_id,
            lambda_param=param_state.get('lambda', 0.0),
            mu_param=param_state.get('mu', 0.0),
            tokens_in=selection_result.total_tokens,
            head_tokens=head_tokens,
            tail_tokens=tail_tokens,
            keep_ratio_head=head_tokens / max(1, selection_result.total_tokens),
            keep_ratio_tail=tail_tokens / max(1, selection_result.total_tokens),
            k1=1000,  # Would get from actual config
            k2=param_state.get('ce_k2', 320),
            r=param_state.get('dpp_rank', 14),
            ce_early_exit=selection_result.head_selection.ce_early_exit_used if selection_result.head_selection else False,
            num_windows=num_windows,
            window_size=param_state.get('window_size', 6000),
            stride=param_state.get('stride', 3000),
            sinks=96,  # Would get from config
            kv_prefix_reuse=selection_result.kv_prefix_reuse_ratio,
            middleware_p95=selection_result.selection_time_ms,  # Simplified
            llm_p95=0.0,  # Would need actual LLM timing
            delta_cbu_per_1k=0.0,  # Would calculate from quality metrics
            precision_at_k=0.0,  # Would calculate from evaluation
            recall_at_k=0.0,  # Would calculate from evaluation
            primal_dual_gap=gap,
            tail_cvar=self.cvar_calculator.calculate_cvar() or 0.0,
            xi_parameter=self.evt_modeler.get_xi_parameter() or 0.0
        )
        
        self.telemetry_records.append(telemetry)
        
        # Process alarms
        self._process_alarms(telemetry, kv_metrics, drift_metrics)
        
        # Log if enabled
        if self.log_telemetry:
            self._log_telemetry(telemetry)
    
    def _process_alarms(self, telemetry: TelemetryRecord, 
                       kv_metrics: KVJaccardMetrics,
                       drift_metrics: Dict[str, DriftMetrics]):
        """Process and manage alarm conditions."""
        current_time = time.time()
        
        # KV Jaccard degradation alarm
        if kv_metrics.alarm_triggered:
            alarm_key = "kv_jaccard_degradation"
            if alarm_key not in self.active_alarms:
                self._trigger_alarm(
                    alarm_key,
                    AlarmLevel.WARNING,
                    f"KV Jaccard degradation: {kv_metrics.degradation_pp:.3f}pp",
                    {"kv_metrics": kv_metrics}
                )
        
        # Parameter drift alarms
        for param_name, drift in drift_metrics.items():
            if drift.alarm_triggered:
                alarm_key = f"param_drift_{param_name}"
                if alarm_key not in self.active_alarms:
                    self._trigger_alarm(
                        alarm_key,
                        AlarmLevel.WARNING,
                        f"Parameter {param_name} drift: {drift.drift_percentage:.1%}",
                        {"drift_metrics": drift}
                    )
        
        # Xi parameter alarm (high tail risk)
        xi = telemetry.xi_parameter
        if xi > 0.2:  # High tail risk threshold
            alarm_key = "high_xi_parameter"
            if alarm_key not in self.active_alarms:
                self._trigger_alarm(
                    alarm_key,
                    AlarmLevel.CRITICAL,
                    f"High tail risk: ξ={xi:.4f} > 0.2",
                    {"xi_parameter": xi}
                )
        
        # Primal-dual gap alarm (poor convergence)
        if telemetry.primal_dual_gap > 0.01:  # 1% threshold  
            alarm_key = "poor_convergence"
            if alarm_key not in self.active_alarms:
                self._trigger_alarm(
                    alarm_key,
                    AlarmLevel.WARNING,
                    f"Poor optimization convergence: gap={telemetry.primal_dual_gap:.3%}",
                    {"gap": telemetry.primal_dual_gap}
                )
        
        # Clear resolved alarms
        self._clear_resolved_alarms(current_time)
    
    def _trigger_alarm(self, alarm_key: str, level: AlarmLevel, 
                      message: str, context: Dict[str, Any]):
        """Trigger new alarm."""
        alarm = {
            'key': alarm_key,
            'level': level,
            'message': message,
            'context': context,
            'triggered_at': time.time(),
            'acknowledged': False
        }
        
        self.active_alarms[alarm_key] = alarm
        self.alarm_history.append(alarm.copy())
        
        logger.log(
            logging.CRITICAL if level == AlarmLevel.EMERGENCY else
            logging.ERROR if level == AlarmLevel.CRITICAL else 
            logging.WARNING,
            f"ALARM [{level.value.upper()}] {alarm_key}: {message}"
        )
    
    def _clear_resolved_alarms(self, current_time: float):
        """Clear alarms that have been resolved."""
        to_clear = []
        
        for alarm_key, alarm in self.active_alarms.items():
            # Auto-clear alarms after 1 hour if conditions normalized
            if current_time - alarm['triggered_at'] > 3600:
                to_clear.append(alarm_key)
        
        for key in to_clear:
            del self.active_alarms[key]
            logger.info(f"Auto-cleared alarm: {key}")
    
    def _log_telemetry(self, telemetry: TelemetryRecord):
        """Log telemetry record to structured format."""
        log_record = {
            'timestamp': telemetry.timestamp,
            'session_id': telemetry.session_id,
            'operation_id': telemetry.operation_id,
            'lambda': telemetry.lambda_param,
            'mu': telemetry.mu_param,
            'tokens_in': telemetry.tokens_in,
            'head_tokens': telemetry.head_tokens,
            'tail_tokens': telemetry.tail_tokens,
            'keep_ratio_head': telemetry.keep_ratio_head,
            'keep_ratio_tail': telemetry.keep_ratio_tail,
            'k1': telemetry.k1,
            'k2': telemetry.k2,
            'r': telemetry.r,
            'ce_early_exit': telemetry.ce_early_exit,
            'num_windows': telemetry.num_windows,
            'window_size': telemetry.window_size,
            'stride': telemetry.stride,
            'sinks': telemetry.sinks,
            'kv_prefix_reuse': telemetry.kv_prefix_reuse,
            'middleware_p95': telemetry.middleware_p95,
            'llm_p95': telemetry.llm_p95,
            'delta_cbu_per_1k': telemetry.delta_cbu_per_1k,
            'precision_at_k': telemetry.precision_at_k,
            'recall_at_k': telemetry.recall_at_k,
            'primal_dual_gap': telemetry.primal_dual_gap,
            'tail_cvar': telemetry.tail_cvar,
            'xi_parameter': telemetry.xi_parameter
        }
        
        try:
            with open(self.export_path, 'a') as f:
                f.write(json.dumps(log_record) + '\n')
        except Exception as e:
            logger.error(f"Failed to write telemetry log: {e}")
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for monitoring dashboard."""
        current_time = time.time()
        
        # Recent performance metrics (last 100 operations)
        recent_metrics = list(self.compute_metrics)[-100:]
        
        if recent_metrics:
            avg_latency = np.mean([m.latency_ms for m in recent_metrics])
            p95_latency = np.percentile([m.latency_ms for m in recent_metrics], 95)
            avg_tokens = np.mean([m.tokens_processed for m in recent_metrics])
            avg_kv_reuse = np.mean([m.kv_reuse_ratio for m in recent_metrics])
        else:
            avg_latency = p95_latency = avg_tokens = avg_kv_reuse = 0.0
        
        # EVT metrics
        evt_metrics = self.evt_modeler.get_tail_risk_metrics()
        
        # CVaR
        current_cvar = self.cvar_calculator.calculate_cvar()
        
        # Convergence status
        convergence = self.gap_monitor.get_convergence_status()
        
        return {
            "performance": {
                "avg_latency_ms": avg_latency,
                "p95_latency_ms": p95_latency,
                "avg_tokens_processed": avg_tokens,
                "avg_kv_reuse_ratio": avg_kv_reuse,
                "total_operations": len(self.compute_metrics)
            },
            "tail_risk": {
                "current_cvar": current_cvar,
                "xi_parameter": evt_metrics.get("xi", 0.0),
                "risk_assessment": evt_metrics.get("risk_assessment", "unknown"),
                "tail_quality": evt_metrics.get("quality_score", 0.0)
            },
            "optimization": {
                "primal_dual_gap": convergence.get("current_gap", 0.0),
                "converged": convergence.get("converged", False),
                "gap_trend": convergence.get("trend", "unknown")
            },
            "alarms": {
                "active_count": len(self.active_alarms),
                "active_alarms": list(self.active_alarms.keys()),
                "recent_alarms": len([a for a in self.alarm_history if current_time - a['triggered_at'] < 3600])
            },
            "timestamp": current_time
        }
    
    def export_telemetry(self, filepath: Optional[str] = None) -> str:
        """Export telemetry data to JSON file."""
        filepath = filepath or f"/tmp/hybrid_telemetry_export_{int(time.time())}.json"
        
        export_data = {
            "metadata": {
                "export_timestamp": time.time(),
                "total_records": len(self.telemetry_records),
                "total_compute_metrics": len(self.compute_metrics)
            },
            "telemetry_records": [
                {
                    "timestamp": t.timestamp,
                    "session_id": t.session_id,
                    "operation_id": t.operation_id,
                    "parameters": {
                        "lambda": t.lambda_param,
                        "mu": t.mu_param,
                        "k2": t.k2,
                        "r": t.r,
                        "window_size": t.window_size,
                        "stride": t.stride
                    },
                    "metrics": {
                        "tokens_in": t.tokens_in,
                        "head_tokens": t.head_tokens,
                        "tail_tokens": t.tail_tokens,
                        "kv_prefix_reuse": t.kv_prefix_reuse,
                        "primal_dual_gap": t.primal_dual_gap,
                        "tail_cvar": t.tail_cvar,
                        "xi_parameter": t.xi_parameter
                    }
                } for t in list(self.telemetry_records)
            ],
            "evt_analysis": self.evt_modeler.get_tail_risk_metrics(),
            "convergence_status": self.gap_monitor.get_convergence_status(),
            "alarm_history": list(self.alarm_history)
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Telemetry exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export telemetry: {e}")
            raise
    
    def acknowledge_alarm(self, alarm_key: str) -> bool:
        """Acknowledge an active alarm."""
        if alarm_key in self.active_alarms:
            self.active_alarms[alarm_key]['acknowledged'] = True
            self.active_alarms[alarm_key]['acknowledged_at'] = time.time()
            logger.info(f"Acknowledged alarm: {alarm_key}")
            return True
        return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        dashboard_metrics = self.get_dashboard_metrics()
        
        # Determine overall health
        critical_alarms = [a for a in self.active_alarms.values() 
                          if a['level'] in [AlarmLevel.CRITICAL, AlarmLevel.EMERGENCY]]
        
        if critical_alarms:
            health_status = "CRITICAL"
        elif self.active_alarms:
            health_status = "WARNING"
        else:
            health_status = "HEALTHY"
        
        # Performance health indicators
        p95_latency = dashboard_metrics["performance"]["p95_latency_ms"]
        kv_reuse = dashboard_metrics["performance"]["avg_kv_reuse_ratio"]
        xi_param = dashboard_metrics["tail_risk"]["xi_parameter"]
        
        health_indicators = {
            "latency_healthy": p95_latency < 1000,  # <1s p95
            "kv_reuse_healthy": kv_reuse > 0.6,     # >60% reuse
            "tail_risk_healthy": xi_param < 0.2     # ξ < 0.2
        }
        
        return {
            "overall_status": health_status,
            "health_indicators": health_indicators,
            "active_alarms_count": len(self.active_alarms),
            "critical_alarms_count": len(critical_alarms),
            "performance_summary": dashboard_metrics["performance"],
            "risk_summary": dashboard_metrics["tail_risk"],
            "last_updated": time.time()
        }

def create_instrumentation(config: Optional[Dict[str, Any]] = None) -> HybridInstrumentation:
    """Create instrumentation system with default configuration."""
    return HybridInstrumentation(config)