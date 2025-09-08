#!/usr/bin/env python3
"""
Comprehensive Performance Monitoring and Diagnostics System

Real-time monitoring system for Lethe's Lagrangian optimization with
comprehensive metrics, alerting, and dashboard-ready data export.

Key Features:
- Dual diagnostics (monotone size(λ), <0.5% dual gap)
- λ-drift ≤ ±15% monitoring with alerts
- CBU-elasticity smoothness tracking
- ECE × type × budget slicing for long-tail safety
- KV prefix-reuse monitoring (≥10pp Jaccard drop detection)
- Performance regression detection
- Real-time dashboard data export

Operational Requirements:
- Maintain promotion criteria: ΔCBU/GB ≥ +10% or P95 improvement ≥5ms
- Alert on performance degradation patterns
- Track quality preservation across optimization strategies
"""

import logging
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Set, Any, NamedTuple, Union
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from pathlib import Path
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class MetricType(Enum):
    """Types of metrics being tracked."""
    LATENCY = "latency"
    QUALITY = "quality"
    COMPUTATIONAL = "computational"
    OPERATIONAL = "operational"

@dataclass
class PerformanceAlert:
    """Performance alert with detailed context."""
    timestamp: datetime
    alert_level: AlertLevel
    metric_type: MetricType
    title: str
    description: str
    current_value: float
    threshold_value: float
    context: Dict[str, Any] = field(default_factory=dict)
    resolution_suggestion: str = ""

@dataclass
class MetricSnapshot:
    """Point-in-time metric snapshot."""
    timestamp: datetime
    lambda_multiplier: float
    p95_latency_ms: float
    cbu_improvement: float
    dual_gap: float
    lambda_drift: float
    computational_savings: float
    quality_preservation: float
    prefix_jaccard_similarity: float
    
    # Query distribution
    standard_query_rate: float
    hard_query_rate: float
    early_exit_rate: float
    
    # Operational metrics
    cpu_utilization: float
    memory_usage_mb: float
    throughput_qps: float

@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring."""
    
    # Monitoring windows
    real_time_window_size: int = 100      # Real-time metrics window
    trend_analysis_window_size: int = 1000  # Trend analysis window
    alert_evaluation_interval_s: int = 30    # Alert check frequency
    
    # Performance targets and thresholds
    target_p95_latency_ms: float = 1.0      # Target P95 latency
    target_cbu_improvement: float = 10.0    # Minimum CBU for promotion
    max_lambda_drift: float = 0.15          # ±15% λ drift tolerance
    max_dual_gap: float = 0.005             # <0.5% dual gap threshold
    min_prefix_jaccard: float = 0.10        # ≥10pp Jaccard drop alert
    
    # Quality thresholds
    min_quality_preservation: float = 0.85  # Minimum quality preservation
    min_cbu_elasticity_smoothness: float = 0.8  # CBU elasticity smoothness
    
    # Computational thresholds
    max_cpu_utilization: float = 0.8       # 80% CPU threshold
    max_memory_usage_mb: float = 4096      # 4GB memory threshold
    
    # Promotion criteria
    promotion_cbu_delta_threshold: float = 10.0     # ΔCBU/GB ≥ +10%
    promotion_p95_improvement_threshold: float = 5.0  # P95 improvement ≥5ms
    
    # Data retention
    metrics_retention_hours: int = 168     # 7 days
    alerts_retention_hours: int = 720      # 30 days
    
    # Export settings
    enable_dashboard_export: bool = True
    dashboard_export_interval_s: int = 60
    dashboard_export_path: Optional[str] = None

class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for Lagrangian optimization.
    
    Provides real-time monitoring, alerting, trend analysis, and dashboard
    data export for all aspects of the latency optimization system.
    
    Key Capabilities:
    1. Real-time performance tracking with configurable windows
    2. Automated alerting based on thresholds and trends
    3. Quality preservation monitoring across optimization strategies
    4. Computational efficiency tracking and budget analysis
    5. λ-drift detection and dual gap monitoring
    6. Dashboard-ready data export for visualization
    7. Performance regression detection and root cause analysis
    """
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        """Initialize performance monitoring system."""
        self.config = config or PerformanceConfig()
        
        # Metric storage
        self.metric_history: deque[MetricSnapshot] = deque(
            maxlen=self.config.trend_analysis_window_size
        )
        self.real_time_metrics: deque[MetricSnapshot] = deque(
            maxlen=self.config.real_time_window_size
        )
        
        # Alert system
        self.active_alerts: List[PerformanceAlert] = []
        self.alert_history: deque[PerformanceAlert] = deque(maxlen=10000)
        
        # Specialized tracking
        self.lambda_drift_history: deque[Tuple[datetime, float]] = deque(maxlen=1000)
        self.dual_gap_history: deque[Tuple[datetime, float]] = deque(maxlen=1000)
        self.cbu_elasticity_data: deque[Tuple[float, float]] = deque(maxlen=500)  # (λ, CBU)
        self.prefix_jaccard_history: deque[Tuple[datetime, float]] = deque(maxlen=1000)
        
        # ECE slicing data (Error Calibration × Type × Budget)
        self.ece_slices: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)  # type -> (error, confidence, budget)
        
        # Performance tracking
        self.query_type_distribution: Dict[str, int] = defaultdict(int)
        self.optimization_strategy_effectiveness: Dict[str, List[float]] = defaultdict(list)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background monitoring
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        
        logger.info(f"PerformanceMonitor initialized with {self.config.real_time_window_size} real-time window")
    
    def record_optimization_result(
        self,
        lambda_multiplier: float,
        p95_latency_ms: float,
        cbu_improvement: float,
        dual_gap: float,
        lambda_drift: float,
        computational_savings: float,
        quality_preservation: float,
        prefix_jaccard_similarity: float,
        query_type: str = "standard",
        early_exit_used: bool = False,
        cpu_utilization: float = 0.0,
        memory_usage_mb: float = 0.0,
        throughput_qps: float = 0.0,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        """Record optimization result and update all tracking metrics."""
        
        with self._lock:
            timestamp = datetime.now()
            
            # Create metric snapshot
            snapshot = MetricSnapshot(
                timestamp=timestamp,
                lambda_multiplier=lambda_multiplier,
                p95_latency_ms=p95_latency_ms,
                cbu_improvement=cbu_improvement,
                dual_gap=dual_gap,
                lambda_drift=lambda_drift,
                computational_savings=computational_savings,
                quality_preservation=quality_preservation,
                prefix_jaccard_similarity=prefix_jaccard_similarity,
                standard_query_rate=self._compute_query_rate("standard"),
                hard_query_rate=self._compute_query_rate("hard"),
                early_exit_rate=self._compute_early_exit_rate(),
                cpu_utilization=cpu_utilization,
                memory_usage_mb=memory_usage_mb,
                throughput_qps=throughput_qps
            )
            
            # Store snapshots
            self.metric_history.append(snapshot)
            self.real_time_metrics.append(snapshot)
            
            # Update specialized tracking
            self._update_specialized_tracking(snapshot, query_type, early_exit_used, additional_context)
            
            # Check for alerts
            self._evaluate_alerts(snapshot)
            
            # Update query distribution
            self.query_type_distribution[query_type] += 1
            
        logger.debug(f"Recorded optimization: λ={lambda_multiplier:.3f}, P95={p95_latency_ms:.2f}ms, CBU={cbu_improvement:.1f}%")
    
    def _update_specialized_tracking(
        self,
        snapshot: MetricSnapshot,
        query_type: str,
        early_exit_used: bool,
        additional_context: Optional[Dict[str, Any]]
    ):
        """Update specialized tracking data structures."""
        
        # λ-drift tracking
        self.lambda_drift_history.append((snapshot.timestamp, snapshot.lambda_drift))
        
        # Dual gap tracking
        self.dual_gap_history.append((snapshot.timestamp, snapshot.dual_gap))
        
        # CBU elasticity tracking (for smoothness analysis)
        self.cbu_elasticity_data.append((snapshot.lambda_multiplier, snapshot.cbu_improvement))
        
        # Prefix Jaccard similarity tracking
        self.prefix_jaccard_history.append((snapshot.timestamp, snapshot.prefix_jaccard_similarity))
        
        # ECE slicing (Error Calibration × Type × Budget)
        if additional_context:
            error_rate = additional_context.get('error_rate', 0.0)
            confidence = additional_context.get('confidence', 0.9)
            budget_used = additional_context.get('computational_budget_used', 1.0)
            
            ece_key = f"{query_type}_{early_exit_used}"
            self.ece_slices[ece_key].append((error_rate, confidence, budget_used))
            
            # Limit slice history
            if len(self.ece_slices[ece_key]) > 500:
                self.ece_slices[ece_key] = self.ece_slices[ece_key][-500:]
    
    def _evaluate_alerts(self, snapshot: MetricSnapshot):
        """Evaluate current snapshot against alert thresholds."""
        
        alerts = []
        
        # P95 latency alert
        if snapshot.p95_latency_ms > self.config.target_p95_latency_ms * 1.5:
            alerts.append(PerformanceAlert(
                timestamp=snapshot.timestamp,
                alert_level=AlertLevel.CRITICAL if snapshot.p95_latency_ms > self.config.target_p95_latency_ms * 2 else AlertLevel.WARNING,
                metric_type=MetricType.LATENCY,
                title="P95 Latency Exceeded",
                description=f"P95 latency {snapshot.p95_latency_ms:.2f}ms exceeds target {self.config.target_p95_latency_ms}ms",
                current_value=snapshot.p95_latency_ms,
                threshold_value=self.config.target_p95_latency_ms,
                resolution_suggestion="Increase λ multiplier to prioritize speed over quality"
            ))
        
        # CBU improvement alert
        if snapshot.cbu_improvement < self.config.target_cbu_improvement:
            alerts.append(PerformanceAlert(
                timestamp=snapshot.timestamp,
                alert_level=AlertLevel.WARNING,
                metric_type=MetricType.QUALITY,
                title="CBU Below Promotion Threshold",
                description=f"CBU improvement {snapshot.cbu_improvement:.1f}% below promotion threshold {self.config.target_cbu_improvement}%",
                current_value=snapshot.cbu_improvement,
                threshold_value=self.config.target_cbu_improvement,
                resolution_suggestion="Decrease λ multiplier to prioritize quality over speed"
            ))
        
        # λ-drift alert
        if abs(snapshot.lambda_drift) > self.config.max_lambda_drift:
            alerts.append(PerformanceAlert(
                timestamp=snapshot.timestamp,
                alert_level=AlertLevel.WARNING,
                metric_type=MetricType.OPERATIONAL,
                title="Lambda Drift Exceeded",
                description=f"λ drift {snapshot.lambda_drift:.3f} exceeds tolerance {self.config.max_lambda_drift}",
                current_value=abs(snapshot.lambda_drift),
                threshold_value=self.config.max_lambda_drift,
                resolution_suggestion="Check for system instability or requirement changes"
            ))
        
        # Dual gap alert
        if snapshot.dual_gap > self.config.max_dual_gap:
            alerts.append(PerformanceAlert(
                timestamp=snapshot.timestamp,
                alert_level=AlertLevel.CRITICAL,
                metric_type=MetricType.OPERATIONAL,
                title="Dual Gap Threshold Exceeded",
                description=f"Dual gap {snapshot.dual_gap:.4f} exceeds threshold {self.config.max_dual_gap}",
                current_value=snapshot.dual_gap,
                threshold_value=self.config.max_dual_gap,
                resolution_suggestion="System optimization may be unstable, review algorithm convergence"
            ))
        
        # Prefix Jaccard similarity alert (KV reuse efficiency)
        if len(self.prefix_jaccard_history) >= 2:
            recent_jaccard = [x[1] for x in list(self.prefix_jaccard_history)[-10:]]
            if len(recent_jaccard) >= 2:
                jaccard_drop = max(recent_jaccard) - min(recent_jaccard)
                if jaccard_drop >= self.config.min_prefix_jaccard:
                    alerts.append(PerformanceAlert(
                        timestamp=snapshot.timestamp,
                        alert_level=AlertLevel.WARNING,
                        metric_type=MetricType.COMPUTATIONAL,
                        title="KV Prefix Reuse Degradation",
                        description=f"Prefix Jaccard similarity dropped {jaccard_drop:.3f} (≥{self.config.min_prefix_jaccard:.3f} threshold)",
                        current_value=jaccard_drop,
                        threshold_value=self.config.min_prefix_jaccard,
                        resolution_suggestion="Check query distribution changes or cache invalidation patterns"
                    ))
        
        # Quality preservation alert
        if snapshot.quality_preservation < self.config.min_quality_preservation:
            alerts.append(PerformanceAlert(
                timestamp=snapshot.timestamp,
                alert_level=AlertLevel.WARNING,
                metric_type=MetricType.QUALITY,
                title="Quality Preservation Below Threshold",
                description=f"Quality preservation {snapshot.quality_preservation:.3f} below threshold {self.config.min_quality_preservation}",
                current_value=snapshot.quality_preservation,
                threshold_value=self.config.min_quality_preservation,
                resolution_suggestion="Reduce optimization aggressiveness or increase quality weights"
            ))
        
        # CPU utilization alert
        if snapshot.cpu_utilization > self.config.max_cpu_utilization:
            alerts.append(PerformanceAlert(
                timestamp=snapshot.timestamp,
                alert_level=AlertLevel.CRITICAL if snapshot.cpu_utilization > 0.95 else AlertLevel.WARNING,
                metric_type=MetricType.OPERATIONAL,
                title="High CPU Utilization",
                description=f"CPU utilization {snapshot.cpu_utilization:.1%} exceeds threshold {self.config.max_cpu_utilization:.1%}",
                current_value=snapshot.cpu_utilization,
                threshold_value=self.config.max_cpu_utilization,
                resolution_suggestion="Scale up resources or reduce computational load"
            ))
        
        # Add new alerts
        for alert in alerts:
            self._add_alert(alert)
    
    def _add_alert(self, alert: PerformanceAlert):
        """Add new alert to active alerts and history."""
        
        # Check if similar alert already active (deduplication)
        for active_alert in self.active_alerts:
            if (active_alert.title == alert.title and 
                (alert.timestamp - active_alert.timestamp) < timedelta(minutes=5)):
                # Update existing alert instead of creating duplicate
                active_alert.current_value = alert.current_value
                active_alert.timestamp = alert.timestamp
                return
        
        # Add to active alerts
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # Log alert
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.CRITICAL: logging.ERROR,
            AlertLevel.EMERGENCY: logging.CRITICAL
        }[alert.alert_level]
        
        logger.log(log_level, f"ALERT [{alert.alert_level.value.upper()}] {alert.title}: {alert.description}")
    
    def _compute_query_rate(self, query_type: str) -> float:
        """Compute rate of specific query type in recent window."""
        total_queries = sum(self.query_type_distribution.values())
        if total_queries == 0:
            return 0.0
        
        return self.query_type_distribution[query_type] / total_queries
    
    def _compute_early_exit_rate(self) -> float:
        """Compute early exit usage rate in recent window."""
        if len(self.real_time_metrics) < 10:
            return 0.0
        
        # This would be tracked separately in practice
        # For now, estimate based on computational savings
        recent_savings = [m.computational_savings for m in list(self.real_time_metrics)[-10:]]
        return sum(1 for s in recent_savings if s > 0.1) / len(recent_savings)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data for visualization."""
        
        with self._lock:
            if not self.real_time_metrics:
                return {'status': 'no_data'}
            
            recent_metrics = list(self.real_time_metrics)
            latest = recent_metrics[-1]
            
            # Compute trends
            trends = self._compute_trends(recent_metrics)
            
            # Get promotion criteria status
            promotion_status = self._evaluate_promotion_criteria()
            
            # CBU elasticity analysis
            cbu_elasticity = self._analyze_cbu_elasticity()
            
            # Alert summary
            alert_summary = self._summarize_alerts()
            
            return {
                'timestamp': latest.timestamp.isoformat(),
                'current_metrics': {
                    'lambda_multiplier': latest.lambda_multiplier,
                    'p95_latency_ms': latest.p95_latency_ms,
                    'cbu_improvement': latest.cbu_improvement,
                    'dual_gap': latest.dual_gap,
                    'lambda_drift': latest.lambda_drift,
                    'computational_savings': latest.computational_savings,
                    'quality_preservation': latest.quality_preservation,
                    'prefix_jaccard_similarity': latest.prefix_jaccard_similarity,
                    'cpu_utilization': latest.cpu_utilization,
                    'throughput_qps': latest.throughput_qps
                },
                'targets': {
                    'p95_latency_target_ms': self.config.target_p95_latency_ms,
                    'cbu_improvement_target': self.config.target_cbu_improvement,
                    'max_lambda_drift': self.config.max_lambda_drift,
                    'max_dual_gap': self.config.max_dual_gap
                },
                'trends': trends,
                'promotion_status': promotion_status,
                'cbu_elasticity': cbu_elasticity,
                'query_distribution': dict(self.query_type_distribution),
                'alert_summary': alert_summary,
                'ece_analysis': self._analyze_ece_slices(),
                'system_health': {
                    'monitoring_window_size': len(self.real_time_metrics),
                    'total_queries_tracked': sum(self.query_type_distribution.values()),
                    'active_alerts_count': len(self.active_alerts),
                    'uptime_hours': (datetime.now() - recent_metrics[0].timestamp).total_seconds() / 3600 if recent_metrics else 0
                }
            }
    
    def _compute_trends(self, metrics: List[MetricSnapshot]) -> Dict[str, Any]:
        """Compute trend analysis for key metrics."""
        if len(metrics) < 10:
            return {}
        
        # Extract time series data
        latencies = [m.p95_latency_ms for m in metrics[-50:]]
        cbu_scores = [m.cbu_improvement for m in metrics[-50:]]
        lambda_values = [m.lambda_multiplier for m in metrics[-50:]]
        
        return {
            'latency_trend': {
                'direction': 'increasing' if latencies[-1] > latencies[0] else 'decreasing',
                'slope': np.polyfit(range(len(latencies)), latencies, 1)[0] if len(latencies) > 1 else 0,
                'volatility': np.std(latencies)
            },
            'cbu_trend': {
                'direction': 'increasing' if cbu_scores[-1] > cbu_scores[0] else 'decreasing',
                'slope': np.polyfit(range(len(cbu_scores)), cbu_scores, 1)[0] if len(cbu_scores) > 1 else 0,
                'volatility': np.std(cbu_scores)
            },
            'lambda_trend': {
                'direction': 'increasing' if lambda_values[-1] > lambda_values[0] else 'decreasing',
                'slope': np.polyfit(range(len(lambda_values)), lambda_values, 1)[0] if len(lambda_values) > 1 else 0,
                'stability': 1.0 - min(1.0, np.std(lambda_values))
            }
        }
    
    def _evaluate_promotion_criteria(self) -> Dict[str, Any]:
        """Evaluate promotion criteria: ΔCBU/GB ≥ +10% or P95 improvement ≥5ms."""
        
        if len(self.real_time_metrics) < 10:
            return {'status': 'insufficient_data'}
        
        recent = list(self.real_time_metrics)[-10:]
        
        # CBU criteria
        avg_cbu = np.mean([m.cbu_improvement for m in recent])
        cbu_meets_threshold = avg_cbu >= self.config.promotion_cbu_delta_threshold
        
        # P95 improvement criteria (vs target)
        avg_p95 = np.mean([m.p95_latency_ms for m in recent])
        p95_improvement = max(0, self.config.target_p95_latency_ms * 5.0 - avg_p95)  # Baseline is 5x target
        p95_meets_threshold = p95_improvement >= self.config.promotion_p95_improvement_threshold
        
        meets_criteria = cbu_meets_threshold or p95_meets_threshold
        
        return {
            'meets_promotion_criteria': meets_criteria,
            'cbu_status': {
                'current': avg_cbu,
                'threshold': self.config.promotion_cbu_delta_threshold,
                'meets_threshold': cbu_meets_threshold
            },
            'p95_status': {
                'current_improvement': p95_improvement,
                'threshold': self.config.promotion_p95_improvement_threshold,
                'meets_threshold': p95_meets_threshold
            }
        }
    
    def _analyze_cbu_elasticity(self) -> Dict[str, Any]:
        """Analyze CBU elasticity smoothness (ΔCBU/Δλ monotone around knee)."""
        
        if len(self.cbu_elasticity_data) < 20:
            return {'status': 'insufficient_data'}
        
        # Extract λ and CBU pairs
        lambda_cbu_pairs = list(self.cbu_elasticity_data)
        lambda_values = [x[0] for x in lambda_cbu_pairs]
        cbu_values = [x[1] for x in lambda_cbu_pairs]
        
        # Sort by λ values
        sorted_pairs = sorted(zip(lambda_values, cbu_values))
        sorted_lambdas = [x[0] for x in sorted_pairs]
        sorted_cbus = [x[1] for x in sorted_pairs]
        
        # Compute elasticity (ΔCBU/Δλ)
        elasticities = []
        for i in range(1, len(sorted_pairs)):
            delta_lambda = sorted_lambdas[i] - sorted_lambdas[i-1]
            delta_cbu = sorted_cbus[i] - sorted_cbus[i-1]
            if abs(delta_lambda) > 1e-6:
                elasticity = delta_cbu / delta_lambda
                elasticities.append(elasticity)
        
        if not elasticities:
            return {'status': 'no_elasticity_data'}
        
        # Check for monotonicity (smoothness)
        monotone_score = self._compute_monotonicity_score(elasticities)
        
        # Find knee point (maximum curvature)
        knee_index = self._find_knee_point(sorted_cbus)
        
        return {
            'elasticity_mean': np.mean(elasticities),
            'elasticity_std': np.std(elasticities),
            'monotonicity_score': monotone_score,
            'smoothness_assessment': 'smooth' if monotone_score > self.config.min_cbu_elasticity_smoothness else 'rough',
            'knee_point': {
                'lambda': sorted_lambdas[knee_index] if knee_index < len(sorted_lambdas) else None,
                'cbu': sorted_cbus[knee_index] if knee_index < len(sorted_cbus) else None
            },
            'total_samples': len(elasticities)
        }
    
    def _compute_monotonicity_score(self, values: List[float]) -> float:
        """Compute monotonicity score (0=not monotonic, 1=perfectly monotonic)."""
        if len(values) < 2:
            return 1.0
        
        # Count monotonic transitions
        monotonic_transitions = 0
        for i in range(1, len(values)):
            if values[i] >= values[i-1]:  # Allow flat segments
                monotonic_transitions += 1
        
        return monotonic_transitions / (len(values) - 1)
    
    def _find_knee_point(self, values: List[float]) -> int:
        """Find knee point using maximum curvature method."""
        if len(values) < 3:
            return 0
        
        # Compute second derivatives (curvature approximation)
        curvatures = []
        for i in range(1, len(values) - 1):
            curvature = abs(values[i+1] - 2*values[i] + values[i-1])
            curvatures.append(curvature)
        
        # Return index of maximum curvature
        max_curvature_idx = np.argmax(curvatures) if curvatures else 0
        return max_curvature_idx + 1  # Adjust for offset
    
    def _analyze_ece_slices(self) -> Dict[str, Any]:
        """Analyze Error Calibration × Type × Budget slices for long-tail safety."""
        
        analysis = {}
        
        for slice_key, slice_data in self.ece_slices.items():
            if len(slice_data) < 10:
                continue
            
            errors = [x[0] for x in slice_data]
            confidences = [x[1] for x in slice_data]
            budgets = [x[2] for x in slice_data]
            
            # ECE analysis
            ece_score = self._compute_expected_calibration_error(errors, confidences)
            
            # Budget efficiency
            budget_efficiency = np.mean([c/b for c, b in zip(confidences, budgets) if b > 0])
            
            # Long-tail analysis (95th percentile)
            error_p95 = np.percentile(errors, 95)
            
            analysis[slice_key] = {
                'expected_calibration_error': ece_score,
                'budget_efficiency': budget_efficiency,
                'error_p95': error_p95,
                'sample_count': len(slice_data),
                'mean_confidence': np.mean(confidences),
                'mean_budget_utilization': np.mean(budgets)
            }
        
        return analysis
    
    def _compute_expected_calibration_error(self, errors: List[float], confidences: List[float]) -> float:
        """Compute Expected Calibration Error (ECE) for confidence calibration."""
        if len(errors) != len(confidences) or len(errors) == 0:
            return 1.0
        
        # Simple ECE: |confidence - accuracy| averaged over samples
        ece = 0.0
        for error, confidence in zip(errors, confidences):
            accuracy = 1.0 - error  # Convert error to accuracy
            ece += abs(confidence - accuracy)
        
        return ece / len(errors)
    
    def _summarize_alerts(self) -> Dict[str, Any]:
        """Summarize current alert status."""
        
        if not self.active_alerts:
            return {'status': 'no_active_alerts'}
        
        alert_counts = defaultdict(int)
        for alert in self.active_alerts:
            alert_counts[alert.alert_level.value] += 1
        
        # Get most recent critical alerts
        critical_alerts = [a for a in self.active_alerts if a.alert_level == AlertLevel.CRITICAL]
        critical_alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        return {
            'total_active_alerts': len(self.active_alerts),
            'alert_counts_by_level': dict(alert_counts),
            'most_recent_critical': [
                {
                    'title': alert.title,
                    'description': alert.description,
                    'timestamp': alert.timestamp.isoformat()
                }
                for alert in critical_alerts[:3]
            ]
        }
    
    def clear_resolved_alerts(self, resolution_threshold_minutes: int = 30):
        """Clear alerts that have been resolved (no recent occurrences)."""
        
        current_time = datetime.now()
        threshold_time = current_time - timedelta(minutes=resolution_threshold_minutes)
        
        with self._lock:
            # Keep only recent alerts
            self.active_alerts = [
                alert for alert in self.active_alerts
                if alert.timestamp > threshold_time
            ]
        
        logger.info(f"Cleared resolved alerts older than {resolution_threshold_minutes} minutes")
    
    def export_dashboard_data(self, file_path: Optional[str] = None) -> str:
        """Export dashboard data to JSON file."""
        
        export_path = file_path or self.config.dashboard_export_path
        if not export_path:
            export_path = f"lethe_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        dashboard_data = self.get_dashboard_data()
        
        try:
            with open(export_path, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            
            logger.info(f"Dashboard data exported to {export_path}")
            return export_path
            
        except Exception as e:
            logger.error(f"Failed to export dashboard data: {e}")
            raise
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get high-level performance summary."""
        
        with self._lock:
            if not self.real_time_metrics:
                return {'status': 'no_data'}
            
            recent = list(self.real_time_metrics)[-10:]
            latest = recent[-1]
            
            # Compute averages
            avg_latency = np.mean([m.p95_latency_ms for m in recent])
            avg_cbu = np.mean([m.cbu_improvement for m in recent])
            avg_savings = np.mean([m.computational_savings for m in recent])
            avg_quality = np.mean([m.quality_preservation for m in recent])
            
            return {
                'summary': {
                    'current_p95_latency_ms': latest.p95_latency_ms,
                    'target_p95_latency_ms': self.config.target_p95_latency_ms,
                    'latency_target_met': latest.p95_latency_ms <= self.config.target_p95_latency_ms,
                    'current_cbu_improvement': latest.cbu_improvement,
                    'cbu_promotion_threshold': self.config.target_cbu_improvement,
                    'cbu_promotion_ready': latest.cbu_improvement >= self.config.target_cbu_improvement,
                    'computational_savings': avg_savings,
                    'quality_preservation': avg_quality,
                    'system_stability': abs(latest.lambda_drift) <= self.config.max_lambda_drift and latest.dual_gap <= self.config.max_dual_gap
                },
                'optimization_effectiveness': {
                    'avg_p95_latency_ms': avg_latency,
                    'avg_cbu_improvement': avg_cbu,
                    'avg_computational_savings': avg_savings,
                    'avg_quality_preservation': avg_quality,
                    'lambda_stability': 1.0 - min(1.0, abs(latest.lambda_drift) / self.config.max_lambda_drift)
                },
                'alert_status': {
                    'active_alerts': len(self.active_alerts),
                    'critical_alerts': len([a for a in self.active_alerts if a.alert_level == AlertLevel.CRITICAL]),
                    'system_health': 'healthy' if len([a for a in self.active_alerts if a.alert_level == AlertLevel.CRITICAL]) == 0 else 'degraded'
                }
            }


def create_performance_monitor(config: Optional[PerformanceConfig] = None) -> PerformanceMonitor:
    """Create comprehensive performance monitoring system."""
    return PerformanceMonitor(config)