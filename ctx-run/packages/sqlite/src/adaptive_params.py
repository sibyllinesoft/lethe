#!/usr/bin/env python3
"""
Adaptive Parameter System for Lethe→StreamingLLM Hybrid

Implements automatic parameter adaptation based on system performance feedback
and optimization constraints. Provides continuous optimization of head size,
stride parameters, and compute budgets based on KV degradation, tail risk,
and performance objectives.

Key Features:
- Automatic head size reduction (2-3%) on KV degradation detection
- Stride reduction (20%) on heavy tail conditions (ξ > 0.2)
- Parameter stabilization on drift alarms
- Continuous feedback loop integration with instrumentation
- Multi-objective optimization balancing quality, latency, and cost
- Exploration-exploitation strategies for parameter tuning
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
import threading
from datetime import datetime, timedelta

# Import system components - handle imports gracefully
try:
    from hybrid_selector import HybridConfig
    from instrumentation import HybridInstrumentation, AlarmLevel
except ImportError:
    # Fallback for testing
    HybridConfig = type('HybridConfig', (), {})
    HybridInstrumentation = type('HybridInstrumentation', (), {})
    AlarmLevel = type('AlarmLevel', (), {'CRITICAL': 'critical', 'WARNING': 'warning'})

logger = logging.getLogger(__name__)

class AdaptationStrategy(Enum):
    """Parameter adaptation strategies."""
    CONSERVATIVE = "conservative"    # Small, safe adjustments
    AGGRESSIVE = "aggressive"       # Larger adjustments for faster convergence
    EXPLORATORY = "exploratory"     # Exploration-based tuning
    STABILIZATION = "stabilization" # Focus on stability over optimization

class ParameterBounds:
    """Parameter bounds and constraints."""
    
    def __init__(self):
        # Head configuration bounds
        self.head_keep_ratio_min = 0.05
        self.head_keep_ratio_max = 0.30
        self.dpp_rank_min = 8
        self.dpp_rank_max = 20
        
        # Tail configuration bounds  
        self.window_size_min = 2000
        self.window_size_max = 12000
        self.stride_ratio_min = 0.3  # stride/window_size
        self.stride_ratio_max = 0.8
        self.sink_tokens_min = 32
        self.sink_tokens_max = 128
        
        # Optimization bounds
        self.lambda_min = 0.001
        self.lambda_max = 0.1
        self.mu_min = 0.001
        self.mu_max = 0.1
    
    def constrain(self, param_name: str, value: float) -> float:
        """Apply constraints to parameter value."""
        bounds_map = {
            'head_keep_ratio': (self.head_keep_ratio_min, self.head_keep_ratio_max),
            'dpp_rank': (self.dpp_rank_min, self.dpp_rank_max),
            'window_size': (self.window_size_min, self.window_size_max),
            'stride_ratio': (self.stride_ratio_min, self.stride_ratio_max),
            'sink_tokens': (self.sink_tokens_min, self.sink_tokens_max),
            'lambda_param': (self.lambda_min, self.lambda_max),
            'mu_param': (self.mu_min, self.mu_max)
        }
        
        if param_name in bounds_map:
            min_val, max_val = bounds_map[param_name]
            return max(min_val, min(max_val, value))
        
        return value

@dataclass
class AdaptationRule:
    """Single parameter adaptation rule."""
    parameter_name: str
    trigger_condition: str  # e.g., "kv_degradation > -0.10"
    adjustment_type: str    # "multiply", "add", "set"
    adjustment_value: float
    cooldown_seconds: float = 300.0  # 5 minute cooldown
    max_applications: int = 5  # Max times this rule can be applied per hour
    
    # Tracking state
    last_applied: float = 0.0
    applications_count: int = 0
    applications_reset_time: float = 0.0

@dataclass
class OptimizationObjective:
    """Multi-objective optimization configuration."""
    # Objective weights (should sum to 1.0)
    quality_weight: float = 0.4      # CBU/P@k/R@k importance
    latency_weight: float = 0.3      # Speed importance
    cost_weight: float = 0.2         # Token/compute cost importance
    stability_weight: float = 0.1    # Parameter stability importance
    
    # Target values
    target_latency_ms: float = 200.0
    target_cost_per_1k_tokens: float = 0.05
    target_quality_score: float = 0.8
    target_kv_reuse_ratio: float = 0.7

@dataclass
class ParameterExploration:
    """Parameter exploration configuration."""
    exploration_rate: float = 0.1    # Probability of exploration vs exploitation
    exploration_magnitude: float = 0.05  # Size of exploration steps (5%)
    exploration_decay: float = 0.99   # Decay rate for exploration over time
    min_exploration_rate: float = 0.01
    
    # Exploration history
    explorations_attempted: int = 0
    explorations_successful: int = 0
    last_exploration_time: float = 0.0

@dataclass
class AdaptationResult:
    """Result of parameter adaptation attempt."""
    parameter_name: str
    old_value: float
    new_value: float
    adjustment_magnitude: float
    rule_applied: Optional[str]
    success: bool
    reason: str
    timestamp: float

class PerformanceTracker:
    """Track performance metrics for parameter adaptation."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.baseline_metrics = None
        self.baseline_established = False
        
    def record_metrics(self, metrics: Dict[str, float]):
        """Record performance metrics."""
        timestamped_metrics = {
            'timestamp': time.time(),
            **metrics
        }
        self.metrics_history.append(timestamped_metrics)
        
        # Establish baseline after sufficient samples
        if not self.baseline_established and len(self.metrics_history) >= 20:
            self._establish_baseline()
            
    def _establish_baseline(self):
        """Establish baseline performance metrics."""
        if len(self.metrics_history) < 20:
            return
            
        baseline_window = list(self.metrics_history)[:20]
        self.baseline_metrics = {}
        
        # Calculate baseline averages
        for key in baseline_window[0].keys():
            if key != 'timestamp':
                values = [m[key] for m in baseline_window if key in m]
                if values:
                    self.baseline_metrics[key] = np.mean(values)
        
        self.baseline_established = True
        logger.info(f"Performance baseline established: {self.baseline_metrics}")
    
    def get_performance_trend(self, metric_name: str, window_size: int = 10) -> str:
        """Get performance trend for a specific metric."""
        if len(self.metrics_history) < window_size:
            return "insufficient_data"
        
        recent_metrics = list(self.metrics_history)[-window_size:]
        values = [m.get(metric_name, 0) for m in recent_metrics]
        
        if len(values) < 3:
            return "insufficient_data"
        
        # Simple trend analysis using linear regression slope
        x = np.arange(len(values))
        correlation = np.corrcoef(x, values)[0, 1]
        
        if correlation > 0.3:
            return "improving"
        elif correlation < -0.3:
            return "degrading"
        else:
            return "stable"
    
    def get_current_performance(self) -> Optional[Dict[str, float]]:
        """Get most recent performance metrics."""
        if not self.metrics_history:
            return None
        
        return self.metrics_history[-1]
    
    def get_performance_delta(self, metric_name: str) -> Optional[float]:
        """Get performance delta from baseline."""
        if not self.baseline_established or not self.metrics_history:
            return None
        
        current = self.metrics_history[-1].get(metric_name)
        baseline = self.baseline_metrics.get(metric_name)
        
        if current is None or baseline is None or baseline == 0:
            return None
        
        return (current - baseline) / baseline

class AdaptiveParameterController:
    """Main controller for adaptive parameter optimization."""
    
    def __init__(self, 
                 initial_config: HybridConfig,
                 instrumentation: HybridInstrumentation,
                 objectives: Optional[OptimizationObjective] = None):
        
        self.current_config = initial_config
        self.instrumentation = instrumentation
        self.objectives = objectives or OptimizationObjective()
        self.bounds = ParameterBounds()
        self.exploration = ParameterExploration()
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # Adaptation rules
        self.adaptation_rules = self._create_default_rules()
        
        # Adaptation history
        self.adaptation_history = deque(maxlen=1000)
        self.parameter_history = defaultdict(lambda: deque(maxlen=500))
        
        # Control state
        self.adaptation_enabled = True
        self.stabilization_mode = False
        self.last_adaptation_time = 0.0
        self.adaptation_cooldown = 60.0  # 1 minute between adaptations
        
        # Threading for background adaptation
        self.background_thread = None
        self.shutdown_event = threading.Event()
        
        logger.info("AdaptiveParameterController initialized")
        
    def _create_default_rules(self) -> List[AdaptationRule]:
        """Create default adaptation rules based on TODO.md specifications."""
        rules = [
            # Head size reduction on KV degradation (2-3%)
            AdaptationRule(
                parameter_name="head_keep_ratio",
                trigger_condition="kv_degradation_pp <= -0.10",
                adjustment_type="multiply",
                adjustment_value=0.97,  # 3% reduction
                cooldown_seconds=300.0
            ),
            
            # Stride reduction on heavy tail (20% reduction)
            AdaptationRule(
                parameter_name="stride_ratio",
                trigger_condition="xi_parameter > 0.2",
                adjustment_type="multiply", 
                adjustment_value=0.80,  # 20% reduction
                cooldown_seconds=600.0
            ),
            
            # Lambda increase on high latency
            AdaptationRule(
                parameter_name="lambda_param",
                trigger_condition="p95_latency_ms > 1000",
                adjustment_type="multiply",
                adjustment_value=1.1,  # 10% increase
                cooldown_seconds=300.0
            ),
            
            # Mu increase on high compute cost
            AdaptationRule(
                parameter_name="mu_param", 
                trigger_condition="avg_compute_cost > 0.1",
                adjustment_type="multiply",
                adjustment_value=1.1,  # 10% increase
                cooldown_seconds=300.0
            ),
            
            # DPP rank reduction on poor convergence
            AdaptationRule(
                parameter_name="dpp_rank",
                trigger_condition="primal_dual_gap > 0.01",
                adjustment_type="add",
                adjustment_value=-1,  # Reduce by 1
                cooldown_seconds=600.0
            ),
            
            # Window size adjustment based on tail risk
            AdaptationRule(
                parameter_name="window_size",
                trigger_condition="tail_cvar > 500",
                adjustment_type="multiply",
                adjustment_value=0.9,  # 10% reduction
                cooldown_seconds=300.0
            )
        ]
        
        return rules
    
    def update_performance_metrics(self, metrics: Dict[str, float]):
        """Update performance metrics and trigger adaptation if needed."""
        self.performance_tracker.record_metrics(metrics)
        
        # Check if adaptation should be triggered
        current_time = time.time()
        if (self.adaptation_enabled and 
            current_time - self.last_adaptation_time > self.adaptation_cooldown):
            
            self._evaluate_adaptation_triggers(metrics)
    
    def _evaluate_adaptation_triggers(self, current_metrics: Dict[str, float]):
        """Evaluate whether adaptation rules should be triggered."""
        current_time = time.time()
        
        # Check stabilization mode
        if self._should_enter_stabilization_mode():
            if not self.stabilization_mode:
                logger.info("Entering stabilization mode")
                self.stabilization_mode = True
            return
        else:
            if self.stabilization_mode:
                logger.info("Exiting stabilization mode") 
                self.stabilization_mode = False
        
        # Skip adaptation in stabilization mode
        if self.stabilization_mode:
            return
        
        # Evaluate each adaptation rule
        triggered_adaptations = []
        
        for rule in self.adaptation_rules:
            if self._evaluate_rule_condition(rule, current_metrics):
                if self._can_apply_rule(rule, current_time):
                    adaptation = self._apply_adaptation_rule(rule, current_metrics)
                    if adaptation and adaptation.success:
                        triggered_adaptations.append(adaptation)
                        self._mark_rule_applied(rule, current_time)
        
        # Apply exploration if no rules triggered
        if not triggered_adaptations and self._should_explore():
            exploration_adaptation = self._apply_exploration()
            if exploration_adaptation:
                triggered_adaptations.append(exploration_adaptation)
        
        # Update adaptation state
        if triggered_adaptations:
            self.last_adaptation_time = current_time
            
            for adaptation in triggered_adaptations:
                self.adaptation_history.append(adaptation)
                self.parameter_history[adaptation.parameter_name].append(adaptation)
            
            logger.info(f"Applied {len(triggered_adaptations)} parameter adaptations")
    
    def _evaluate_rule_condition(self, rule: AdaptationRule, metrics: Dict[str, float]) -> bool:
        """Evaluate if rule condition is met."""
        condition = rule.trigger_condition
        
        try:
            # Simple condition evaluation (in practice, would use proper parser)
            if "kv_degradation_pp <= -0.10" in condition:
                return metrics.get('kv_degradation_pp', 0) <= -0.10
            
            elif "xi_parameter > 0.2" in condition:
                return metrics.get('xi_parameter', 0) > 0.2
            
            elif "p95_latency_ms > 1000" in condition:
                return metrics.get('p95_latency_ms', 0) > 1000
            
            elif "avg_compute_cost > 0.1" in condition:
                return metrics.get('avg_compute_cost', 0) > 0.1
            
            elif "primal_dual_gap > 0.01" in condition:
                return metrics.get('primal_dual_gap', 0) > 0.01
            
            elif "tail_cvar > 500" in condition:
                return metrics.get('tail_cvar', 0) > 500
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating rule condition '{condition}': {e}")
            return False
    
    def _can_apply_rule(self, rule: AdaptationRule, current_time: float) -> bool:
        """Check if rule can be applied based on cooldown and limits."""
        # Check cooldown
        if current_time - rule.last_applied < rule.cooldown_seconds:
            return False
        
        # Check hourly application limit
        if current_time - rule.applications_reset_time > 3600:  # Reset hourly
            rule.applications_count = 0
            rule.applications_reset_time = current_time
        
        if rule.applications_count >= rule.max_applications:
            return False
        
        return True
    
    def _apply_adaptation_rule(self, rule: AdaptationRule, metrics: Dict[str, float]) -> Optional[AdaptationResult]:
        """Apply adaptation rule to update parameter."""
        param_name = rule.parameter_name
        current_value = getattr(self.current_config, param_name, None)
        
        if current_value is None:
            logger.error(f"Parameter {param_name} not found in config")
            return None
        
        # Calculate new value
        if rule.adjustment_type == "multiply":
            new_value = current_value * rule.adjustment_value
        elif rule.adjustment_type == "add":
            new_value = current_value + rule.adjustment_value
        elif rule.adjustment_type == "set":
            new_value = rule.adjustment_value
        else:
            logger.error(f"Unknown adjustment type: {rule.adjustment_type}")
            return None
        
        # Apply bounds constraint
        constrained_value = self.bounds.constrain(param_name, new_value)
        
        # Check if change is significant enough
        min_change_threshold = 0.001  # 0.1% minimum change
        relative_change = abs(constrained_value - current_value) / max(abs(current_value), 1e-6)
        
        if relative_change < min_change_threshold:
            return AdaptationResult(
                parameter_name=param_name,
                old_value=current_value,
                new_value=constrained_value,
                adjustment_magnitude=relative_change,
                rule_applied=rule.trigger_condition,
                success=False,
                reason="change_too_small",
                timestamp=time.time()
            )
        
        # Apply parameter change
        try:
            setattr(self.current_config, param_name, constrained_value)
            
            # Special handling for dependent parameters
            if param_name == "stride_ratio":
                # Update actual stride based on current window size
                self.current_config.stride = int(self.current_config.window_size * constrained_value)
            
            adjustment_magnitude = abs(constrained_value - current_value) / max(abs(current_value), 1e-6)
            
            logger.info(f"Adapted {param_name}: {current_value:.6f} → {constrained_value:.6f} "
                       f"({adjustment_magnitude:.1%} change)")
            
            return AdaptationResult(
                parameter_name=param_name,
                old_value=current_value,
                new_value=constrained_value,
                adjustment_magnitude=adjustment_magnitude,
                rule_applied=rule.trigger_condition,
                success=True,
                reason="rule_triggered",
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Failed to apply parameter adaptation: {e}")
            return AdaptationResult(
                parameter_name=param_name,
                old_value=current_value,
                new_value=constrained_value,
                adjustment_magnitude=0.0,
                rule_applied=rule.trigger_condition,
                success=False,
                reason=f"application_error: {e}",
                timestamp=time.time()
            )
    
    def _mark_rule_applied(self, rule: AdaptationRule, current_time: float):
        """Mark rule as applied and update tracking."""
        rule.last_applied = current_time
        rule.applications_count += 1
    
    def _should_enter_stabilization_mode(self) -> bool:
        """Determine if system should enter stabilization mode."""
        # Check for critical alarms
        dashboard_metrics = self.instrumentation.get_dashboard_metrics()
        active_alarms = dashboard_metrics.get("alarms", {}).get("active_count", 0)
        
        if active_alarms > 0:
            return True
        
        # Check for high parameter volatility
        recent_adaptations = [a for a in self.adaptation_history 
                            if time.time() - a.timestamp < 3600]  # Last hour
        
        if len(recent_adaptations) > 10:  # Too many recent changes
            return True
        
        # Check for performance degradation
        performance_trends = {}
        for metric in ['avg_latency_ms', 'kv_reuse_ratio', 'objective_value']:
            trend = self.performance_tracker.get_performance_trend(metric)
            performance_trends[metric] = trend
        
        degrading_count = sum(1 for trend in performance_trends.values() if trend == "degrading")
        if degrading_count >= 2:  # Multiple metrics degrading
            return True
        
        return False
    
    def _should_explore(self) -> bool:
        """Determine if exploration should be attempted."""
        current_time = time.time()
        
        # Check exploration rate with decay
        exploration_rate = max(
            self.exploration.min_exploration_rate,
            self.exploration.exploration_rate * (self.exploration.exploration_decay ** 
                                                 self.exploration.explorations_attempted)
        )
        
        # Random exploration decision
        if np.random.random() > exploration_rate:
            return False
        
        # Check cooldown
        if current_time - self.exploration.last_exploration_time < 900:  # 15 min cooldown
            return False
        
        return True
    
    def _apply_exploration(self) -> Optional[AdaptationResult]:
        """Apply exploration-based parameter adjustment."""
        # Select random parameter to explore
        explorable_params = [
            'head_keep_ratio', 'window_size', 'lambda_param', 'mu_param', 'dpp_rank'
        ]
        
        param_name = np.random.choice(explorable_params)
        current_value = getattr(self.current_config, param_name, None)
        
        if current_value is None:
            return None
        
        # Generate exploration perturbation
        perturbation = np.random.uniform(-1, 1) * self.exploration.exploration_magnitude
        new_value = current_value * (1 + perturbation)
        
        # Apply bounds
        constrained_value = self.bounds.constrain(param_name, new_value)
        
        try:
            setattr(self.current_config, param_name, constrained_value)
            
            self.exploration.explorations_attempted += 1
            self.exploration.last_exploration_time = time.time()
            
            adjustment_magnitude = abs(constrained_value - current_value) / max(abs(current_value), 1e-6)
            
            logger.info(f"Exploration adaptation {param_name}: {current_value:.6f} → {constrained_value:.6f}")
            
            return AdaptationResult(
                parameter_name=param_name,
                old_value=current_value,
                new_value=constrained_value,
                adjustment_magnitude=adjustment_magnitude,
                rule_applied=None,
                success=True,
                reason="exploration",
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Failed to apply exploration: {e}")
            return None
    
    def evaluate_adaptation_success(self, adaptation: AdaptationResult, 
                                  post_metrics: Dict[str, float]) -> bool:
        """Evaluate if adaptation was successful based on performance changes."""
        if not adaptation.success:
            return False
        
        # Get pre-adaptation metrics (simplified)
        baseline_metrics = self.performance_tracker.baseline_metrics
        if not baseline_metrics:
            return True  # Assume success if no baseline
        
        # Calculate multi-objective score improvement
        pre_score = self._calculate_objective_score(baseline_metrics)
        post_score = self._calculate_objective_score(post_metrics)
        
        improvement = post_score - pre_score
        
        # Mark exploration as successful if improvement
        if adaptation.rule_applied is None and improvement > 0:
            self.exploration.explorations_successful += 1
        
        success = improvement > 0.01  # 1% improvement threshold
        
        logger.info(f"Adaptation success evaluation: {adaptation.parameter_name} "
                   f"improvement={improvement:.3f}, success={success}")
        
        return success
    
    def _calculate_objective_score(self, metrics: Dict[str, float]) -> float:
        """Calculate multi-objective performance score."""
        # Normalize metrics to 0-1 scale and apply weights
        quality_score = min(1.0, metrics.get('avg_quality_score', 0) / self.objectives.target_quality_score)
        
        # Invert latency (lower is better)
        latency_score = max(0.0, 1.0 - metrics.get('p95_latency_ms', 1000) / (self.objectives.target_latency_ms * 2))
        
        # Invert cost (lower is better)
        cost_score = max(0.0, 1.0 - metrics.get('avg_cost_per_1k', 0.1) / (self.objectives.target_cost_per_1k_tokens * 2))
        
        # KV reuse as stability proxy
        stability_score = min(1.0, metrics.get('avg_kv_reuse', 0) / self.objectives.target_kv_reuse_ratio)
        
        # Weighted combination
        total_score = (
            quality_score * self.objectives.quality_weight +
            latency_score * self.objectives.latency_weight +
            cost_score * self.objectives.cost_weight +
            stability_score * self.objectives.stability_weight
        )
        
        return total_score
    
    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get current adaptation system status."""
        current_time = time.time()
        
        # Recent adaptations
        recent_adaptations = [a for a in self.adaptation_history 
                            if current_time - a.timestamp < 3600]
        
        successful_adaptations = sum(1 for a in recent_adaptations if a.success)
        
        # Parameter stability
        parameter_stability = {}
        for param_name, history in self.parameter_history.items():
            if len(history) >= 2:
                recent_changes = [a for a in history if current_time - a.timestamp < 3600]
                parameter_stability[param_name] = {
                    'recent_changes': len(recent_changes),
                    'avg_change_magnitude': np.mean([a.adjustment_magnitude for a in recent_changes]) if recent_changes else 0.0
                }
        
        return {
            "adaptation_enabled": self.adaptation_enabled,
            "stabilization_mode": self.stabilization_mode,
            "recent_adaptations": len(recent_adaptations),
            "successful_adaptations": successful_adaptations,
            "success_rate": successful_adaptations / max(1, len(recent_adaptations)),
            "exploration_stats": {
                "attempted": self.exploration.explorations_attempted,
                "successful": self.exploration.explorations_successful,
                "success_rate": self.exploration.explorations_successful / max(1, self.exploration.explorations_attempted)
            },
            "parameter_stability": parameter_stability,
            "current_config": {
                "head_keep_ratio": self.current_config.head_keep_ratio,
                "window_size": self.current_config.window_size,
                "stride": self.current_config.stride,
                "lambda_param": self.current_config.lambda_param,
                "mu_param": self.current_config.mu_param,
                "dpp_rank": self.current_config.dpp_rank
            },
            "last_adaptation_time": self.last_adaptation_time,
            "next_adaptation_available": current_time - self.last_adaptation_time > self.adaptation_cooldown
        }
    
    def force_stabilization(self, duration_seconds: float = 1800):
        """Force stabilization mode for specified duration."""
        self.stabilization_mode = True
        self._stabilization_end_time = time.time() + duration_seconds
        logger.info(f"Forced stabilization mode for {duration_seconds} seconds")
        
        # Schedule automatic exit from stabilization
        def exit_stabilization():
            time.sleep(duration_seconds)
            if hasattr(self, '_stabilization_end_time'):
                if time.time() >= self._stabilization_end_time:
                    self.stabilization_mode = False
                    logger.info("Exited forced stabilization mode")
        
        threading.Thread(target=exit_stabilization, daemon=True).start()
    
    def reset_exploration_state(self):
        """Reset exploration state and statistics."""
        self.exploration.explorations_attempted = 0
        self.exploration.explorations_successful = 0
        self.exploration.last_exploration_time = 0.0
        logger.info("Reset exploration state")
    
    def export_adaptation_history(self, filepath: Optional[str] = None) -> str:
        """Export adaptation history to JSON file."""
        filepath = filepath or f"/tmp/adaptation_history_{int(time.time())}.json"
        
        export_data = {
            "metadata": {
                "export_timestamp": time.time(),
                "total_adaptations": len(self.adaptation_history),
                "adaptation_enabled": self.adaptation_enabled,
                "stabilization_mode": self.stabilization_mode
            },
            "current_config": {
                "head_keep_ratio": self.current_config.head_keep_ratio,
                "window_size": self.current_config.window_size,
                "stride": self.current_config.stride,
                "lambda_param": self.current_config.lambda_param,
                "mu_param": self.current_config.mu_param,
                "dpp_rank": self.current_config.dpp_rank
            },
            "adaptation_history": [
                {
                    "timestamp": a.timestamp,
                    "parameter_name": a.parameter_name,
                    "old_value": a.old_value,
                    "new_value": a.new_value,
                    "adjustment_magnitude": a.adjustment_magnitude,
                    "rule_applied": a.rule_applied,
                    "success": a.success,
                    "reason": a.reason
                } for a in list(self.adaptation_history)
            ],
            "exploration_stats": {
                "attempted": self.exploration.explorations_attempted,
                "successful": self.exploration.explorations_successful,
                "success_rate": self.exploration.explorations_successful / max(1, self.exploration.explorations_attempted)
            }
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Adaptation history exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export adaptation history: {e}")
            raise

def create_adaptive_controller(config: HybridConfig, 
                             instrumentation: HybridInstrumentation,
                             objectives: Optional[OptimizationObjective] = None) -> AdaptiveParameterController:
    """Create adaptive parameter controller with specified configuration."""
    return AdaptiveParameterController(config, instrumentation, objectives)