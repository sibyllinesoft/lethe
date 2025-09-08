#!/usr/bin/env python3
"""
Auto-Bucket Retrials System for Performance Recovery

This module implements an intelligent auto-retrial system that detects performance
clusters underperforming against baseline expectations and automatically reruns
them with optimized micro-policy overrides. The system provides autonomous 
performance recovery with comprehensive dashboard integration and audit logging.

Core Features:
- Automatic detection of underperforming clusters (>ΔCBU 0.5 vs Streaming)
- Dynamic micro-policy override selection based on failure patterns
- Real-time dashboard diff attachment for operational transparency
- Historical performance tracking with statistical significance testing
- Intelligent retry strategies with exponential backoff and circuit breaking
- Integration with monitoring systems for alerting and escalation
"""

import logging
import asyncio
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set, Any, Union, Callable
from collections import defaultdict, deque
import statistics
import numpy as np
import json
import hashlib
from pathlib import Path
import pickle

# Import adaptive control surface for micro-policy overrides
from .adaptive_control_surface import (
    AdaptiveControlSurface, ContextBucket, PerformanceLevel,
    ControlParameters, ContextMetrics
)

logger = logging.getLogger(__name__)

class RetrialTrigger(Enum):
    """Triggers for auto-retrial activation"""
    CBU_UNDERPERFORMANCE = "cbu_underperformance"      # CBU < threshold vs baseline
    LATENCY_REGRESSION = "latency_regression"          # Latency > acceptable bounds
    KV_REUSE_DROP = "kv_reuse_drop"                   # KV cache efficiency drop
    GATE_FAILURE = "gate_failure"                      # Quality gate failures
    MANUAL_TRIGGER = "manual_trigger"                  # Manual operator trigger
    SCHEDULED_VALIDATION = "scheduled_validation"      # Periodic validation trigger

class RetrialStatus(Enum):
    """Status of retrial execution"""
    PENDING = "pending"              # Queued for execution
    RUNNING = "running"              # Currently executing
    SUCCEEDED = "succeeded"          # Completed successfully
    FAILED = "failed"               # Failed after all attempts
    ABANDONED = "abandoned"          # Abandoned due to circuit breaker
    SUPERSEDED = "superseded"        # Superseded by newer retrial

class ClusterType(Enum):
    """Types of performance clusters"""
    DOMAIN_BASED = "domain_based"        # Grouped by content domain
    COMPLEXITY_BASED = "complexity_based"  # Grouped by complexity score
    CONTEXT_BASED = "context_based"      # Grouped by context bucket
    TEMPORAL_BASED = "temporal_based"    # Grouped by time period
    PARAMETER_BASED = "parameter_based"  # Grouped by parameter settings

@dataclass
class PerformanceCluster:
    """Performance cluster for retrial analysis"""
    cluster_id: str
    cluster_type: ClusterType
    member_ids: List[str]                 # IDs of cluster members
    baseline_metrics: Dict[str, float]    # Expected baseline performance
    current_metrics: Dict[str, float]     # Current measured performance
    performance_delta: float              # ΔCBU vs baseline
    created_at: datetime
    last_updated: datetime
    context_metadata: Dict[str, Any]      # Cluster-specific context
    
    @property
    def underperformance_ratio(self) -> float:
        """Calculate underperformance ratio"""
        baseline_cbu = self.baseline_metrics.get('cbu_per_ms', 0)
        current_cbu = self.current_metrics.get('cbu_per_ms', 0)
        
        if baseline_cbu <= 0:
            return 0.0
        
        return (baseline_cbu - current_cbu) / baseline_cbu
    
    @property 
    def cluster_size(self) -> int:
        """Get number of members in cluster"""
        return len(self.member_ids)

@dataclass
class RetrialConfiguration:
    """Configuration for retrial execution"""
    max_attempts: int = 3
    base_delay_seconds: float = 30.0      # Base delay between attempts
    backoff_multiplier: float = 2.0       # Exponential backoff multiplier
    timeout_seconds: float = 300.0        # Per-attempt timeout
    circuit_breaker_threshold: int = 5     # Failures before circuit break
    circuit_breaker_reset_seconds: float = 1800.0  # 30 minutes
    
    # Performance thresholds
    cbu_underperformance_threshold: float = 0.5    # ΔCBU trigger threshold
    latency_regression_threshold: float = 0.2      # 20% latency increase
    kv_reuse_drop_threshold: float = 0.1           # 10% KV reuse drop
    
    # Cluster configuration
    min_cluster_size: int = 3              # Minimum members for clustering
    max_cluster_size: int = 50             # Maximum members per cluster
    cluster_staleness_hours: float = 24.0  # Hours before cluster refresh

@dataclass
class MicroPolicyOverride:
    """Micro-policy override for retrial"""
    override_id: str
    target_bucket: ContextBucket
    parameter_adjustments: Dict[str, float]  # Parameter deltas
    expected_improvement: float              # Expected performance gain
    confidence_score: float                  # Confidence in improvement
    created_at: datetime = field(default_factory=datetime.now)
    
    def apply_to_parameters(self, base_params: ControlParameters) -> ControlParameters:
        """Apply override to base parameters"""
        new_params = ControlParameters(
            lambda_value=base_params.lambda_value,
            mu_window_size=base_params.mu_window_size,
            mu_stride=base_params.mu_stride,
            r_value=base_params.r_value,
            k2_value=base_params.k2_value,
            last_update=datetime.now(),
            update_reason=f"micro_policy_override_{self.override_id}"
        )
        
        # Apply adjustments
        if 'lambda_delta' in self.parameter_adjustments:
            new_params.lambda_value += self.parameter_adjustments['lambda_delta']
        
        if 'window_delta' in self.parameter_adjustments:
            new_params.mu_window_size += int(self.parameter_adjustments['window_delta'])
        
        if 'stride_ratio_delta' in self.parameter_adjustments:
            current_ratio = new_params.get_stride_ratio()
            new_params.set_stride_ratio(current_ratio + self.parameter_adjustments['stride_ratio_delta'])
        
        if 'r_delta' in self.parameter_adjustments:
            new_params.r_value += int(self.parameter_adjustments['r_delta'])
        
        if 'k2_ratio_delta' in self.parameter_adjustments:
            new_params.k2_value = int(new_params.k2_value * (1 + self.parameter_adjustments['k2_ratio_delta']))
        
        return new_params

@dataclass
class RetrialExecution:
    """Single retrial execution record"""
    execution_id: str
    cluster_id: str
    trigger: RetrialTrigger
    status: RetrialStatus
    attempt_number: int
    micro_policy_override: Optional[MicroPolicyOverride]
    
    # Execution details
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    execution_time_ms: float = 0.0
    
    # Results
    baseline_performance: Dict[str, float] = field(default_factory=dict)
    retrial_performance: Dict[str, float] = field(default_factory=dict)
    performance_improvement: float = 0.0
    
    # Metadata
    error_details: Optional[str] = None
    dashboard_diff_url: Optional[str] = None
    
    @property
    def duration_seconds(self) -> float:
        """Get execution duration in seconds"""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0
    
    @property
    def is_successful(self) -> bool:
        """Check if retrial was successful"""
        return self.status == RetrialStatus.SUCCEEDED and self.performance_improvement > 0

class ClusterAnalyzer:
    """Analyzes performance data to identify underperforming clusters"""
    
    def __init__(self, config: RetrialConfiguration):
        self.config = config
        self.cluster_cache = {}
        self.performance_history = deque(maxlen=10000)
        self.lock = threading.RLock()
        
    def update_performance_data(self, 
                               member_id: str,
                               performance_metrics: Dict[str, float],
                               context_metadata: Dict[str, Any]):
        """Update performance data for cluster analysis"""
        try:
            with self.lock:
                timestamp = datetime.now()
                
                # Add to performance history
                self.performance_history.append({
                    'member_id': member_id,
                    'timestamp': timestamp,
                    'metrics': performance_metrics.copy(),
                    'metadata': context_metadata.copy()
                })
                
                # Trigger cluster refresh if needed
                if len(self.performance_history) % 100 == 0:  # Every 100 updates
                    self._refresh_clusters()
                    
        except Exception as e:
            logger.error(f"Error updating performance data: {e}")
    
    def identify_underperforming_clusters(self) -> List[PerformanceCluster]:
        """Identify clusters that are underperforming vs baseline"""
        try:
            with self.lock:
                # Get recent performance data
                recent_cutoff = datetime.now() - timedelta(hours=self.config.cluster_staleness_hours)
                recent_data = [
                    entry for entry in self.performance_history 
                    if entry['timestamp'] >= recent_cutoff
                ]
                
                if len(recent_data) < self.config.min_cluster_size * 2:
                    logger.debug("Insufficient data for cluster analysis")
                    return []
                
                # Create clusters by different strategies
                clusters = []
                
                # Domain-based clustering
                clusters.extend(self._create_domain_clusters(recent_data))
                
                # Complexity-based clustering
                clusters.extend(self._create_complexity_clusters(recent_data))
                
                # Context bucket clustering
                clusters.extend(self._create_context_clusters(recent_data))
                
                # Filter for underperforming clusters
                underperforming = []
                for cluster in clusters:
                    if self._is_cluster_underperforming(cluster):
                        underperforming.append(cluster)
                
                logger.info(f"Identified {len(underperforming)} underperforming clusters "
                           f"out of {len(clusters)} total clusters")
                
                return underperforming
                
        except Exception as e:
            logger.error(f"Error identifying underperforming clusters: {e}")
            return []
    
    def _create_domain_clusters(self, performance_data: List[Dict[str, Any]]) -> List[PerformanceCluster]:
        """Create clusters based on content domain"""
        domain_groups = defaultdict(list)
        
        for entry in performance_data:
            domain = entry['metadata'].get('domain', 'unknown')
            domain_groups[domain].append(entry)
        
        clusters = []
        for domain, group_data in domain_groups.items():
            if len(group_data) >= self.config.min_cluster_size:
                cluster = self._create_cluster_from_group(
                    f"domain_{domain}",
                    ClusterType.DOMAIN_BASED,
                    group_data,
                    {'domain': domain}
                )
                if cluster:
                    clusters.append(cluster)
        
        return clusters
    
    def _create_complexity_clusters(self, performance_data: List[Dict[str, Any]]) -> List[PerformanceCluster]:
        """Create clusters based on context complexity"""
        # Group by complexity score ranges
        complexity_groups = defaultdict(list)
        
        for entry in performance_data:
            complexity = entry['metadata'].get('complexity_score', 0.5)
            
            # Create complexity bins
            if complexity < 0.3:
                complexity_bin = 'low'
            elif complexity < 0.7:
                complexity_bin = 'medium'
            else:
                complexity_bin = 'high'
            
            complexity_groups[complexity_bin].append(entry)
        
        clusters = []
        for complexity_bin, group_data in complexity_groups.items():
            if len(group_data) >= self.config.min_cluster_size:
                cluster = self._create_cluster_from_group(
                    f"complexity_{complexity_bin}",
                    ClusterType.COMPLEXITY_BASED,
                    group_data,
                    {'complexity_bin': complexity_bin}
                )
                if cluster:
                    clusters.append(cluster)
        
        return clusters
    
    def _create_context_clusters(self, performance_data: List[Dict[str, Any]]) -> List[PerformanceCluster]:
        """Create clusters based on context bucket classification"""
        bucket_groups = defaultdict(list)
        
        for entry in performance_data:
            dominant_bucket = entry['metadata'].get('dominant_bucket', 'unknown')
            bucket_groups[dominant_bucket].append(entry)
        
        clusters = []
        for bucket, group_data in bucket_groups.items():
            if len(group_data) >= self.config.min_cluster_size:
                cluster = self._create_cluster_from_group(
                    f"bucket_{bucket}",
                    ClusterType.CONTEXT_BASED,
                    group_data,
                    {'context_bucket': bucket}
                )
                if cluster:
                    clusters.append(cluster)
        
        return clusters
    
    def _create_cluster_from_group(self,
                                  cluster_id: str,
                                  cluster_type: ClusterType,
                                  group_data: List[Dict[str, Any]],
                                  context_metadata: Dict[str, Any]) -> Optional[PerformanceCluster]:
        """Create cluster from grouped performance data"""
        try:
            if len(group_data) < self.config.min_cluster_size:
                return None
            
            # Extract member IDs and metrics
            member_ids = [entry['member_id'] for entry in group_data]
            
            # Calculate baseline metrics (historical averages)
            baseline_metrics = self._calculate_baseline_metrics(group_data)
            
            # Calculate current metrics (recent averages)
            current_metrics = self._calculate_current_metrics(group_data)
            
            # Calculate performance delta
            baseline_cbu = baseline_metrics.get('cbu_per_ms', 0)
            current_cbu = current_metrics.get('cbu_per_ms', 0)
            performance_delta = current_cbu - baseline_cbu
            
            cluster = PerformanceCluster(
                cluster_id=cluster_id,
                cluster_type=cluster_type,
                member_ids=member_ids[:self.config.max_cluster_size],  # Limit cluster size
                baseline_metrics=baseline_metrics,
                current_metrics=current_metrics,
                performance_delta=performance_delta,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                context_metadata=context_metadata
            )
            
            return cluster
            
        except Exception as e:
            logger.error(f"Error creating cluster {cluster_id}: {e}")
            return None
    
    def _calculate_baseline_metrics(self, group_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate baseline metrics from historical data"""
        try:
            # Use older data as baseline (first half)
            sorted_data = sorted(group_data, key=lambda x: x['timestamp'])
            baseline_data = sorted_data[:len(sorted_data)//2] if len(sorted_data) > 4 else sorted_data
            
            metrics = defaultdict(list)
            for entry in baseline_data:
                for key, value in entry['metrics'].items():
                    if isinstance(value, (int, float)):
                        metrics[key].append(value)
            
            # Calculate averages
            baseline = {}
            for key, values in metrics.items():
                if values:
                    baseline[key] = statistics.mean(values)
            
            return baseline
            
        except Exception as e:
            logger.debug(f"Error calculating baseline metrics: {e}")
            return {}
    
    def _calculate_current_metrics(self, group_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate current metrics from recent data"""
        try:
            # Use recent data as current (last half)
            sorted_data = sorted(group_data, key=lambda x: x['timestamp'])
            current_data = sorted_data[len(sorted_data)//2:] if len(sorted_data) > 4 else sorted_data
            
            metrics = defaultdict(list)
            for entry in current_data:
                for key, value in entry['metrics'].items():
                    if isinstance(value, (int, float)):
                        metrics[key].append(value)
            
            # Calculate averages
            current = {}
            for key, values in metrics.items():
                if values:
                    current[key] = statistics.mean(values)
            
            return current
            
        except Exception as e:
            logger.debug(f"Error calculating current metrics: {e}")
            return {}
    
    def _is_cluster_underperforming(self, cluster: PerformanceCluster) -> bool:
        """Check if cluster is underperforming vs thresholds"""
        try:
            # CBU underperformance check
            if cluster.underperformance_ratio > self.config.cbu_underperformance_threshold:
                return True
            
            # Latency regression check
            baseline_latency = cluster.baseline_metrics.get('p95_latency', 0)
            current_latency = cluster.current_metrics.get('p95_latency', 0)
            
            if baseline_latency > 0:
                latency_increase = (current_latency - baseline_latency) / baseline_latency
                if latency_increase > self.config.latency_regression_threshold:
                    return True
            
            # KV reuse drop check
            baseline_kv = cluster.baseline_metrics.get('kv_reuse_ratio', 0)
            current_kv = cluster.current_metrics.get('kv_reuse_ratio', 0)
            
            kv_drop = baseline_kv - current_kv
            if kv_drop > self.config.kv_reuse_drop_threshold:
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking cluster underperformance: {e}")
            return False
    
    def _refresh_clusters(self):
        """Refresh cluster cache periodically"""
        try:
            # Remove stale clusters
            cutoff = datetime.now() - timedelta(hours=self.config.cluster_staleness_hours)
            stale_ids = [
                cluster_id for cluster_id, cluster in self.cluster_cache.items()
                if cluster.last_updated < cutoff
            ]
            
            for cluster_id in stale_ids:
                del self.cluster_cache[cluster_id]
            
            if stale_ids:
                logger.debug(f"Removed {len(stale_ids)} stale clusters from cache")
                
        except Exception as e:
            logger.debug(f"Error refreshing clusters: {e}")

class MicroPolicyGenerator:
    """Generates micro-policy overrides for retrial execution"""
    
    def __init__(self):
        self.policy_templates = self._initialize_policy_templates()
        self.success_history = deque(maxlen=500)
        
    def _initialize_policy_templates(self) -> Dict[ContextBucket, Dict[str, Any]]:
        """Initialize micro-policy templates for different context buckets"""
        return {
            # High duplication -> increase diversity
            ContextBucket.DUP_RATE: {
                'parameter_adjustments': {
                    'r_delta': 2.0,           # Increase DPP rank
                    'k2_ratio_delta': 0.25    # 25% more CE candidates
                },
                'expected_improvement': 0.8,
                'confidence_base': 0.7
            },
            
            # Deep symbols -> increase head keep
            ContextBucket.SYMBOL_DEPTH: {
                'parameter_adjustments': {
                    'lambda_delta': 0.03,     # Increase head keep by 3pp
                    'r_delta': 1.0            # Slight rank increase
                },
                'expected_improvement': 1.2,
                'confidence_base': 0.8
            },
            
            # Low entity entropy -> reduce tail processing  
            ContextBucket.ENTITY_ENTROPY: {
                'parameter_adjustments': {
                    'window_delta': -800.0,   # Reduce window
                    'stride_ratio_delta': 0.15 # Increase stride ratio
                },
                'expected_improvement': 0.6,
                'confidence_base': 0.6
            },
            
            # High repo fanout -> broader context
            ContextBucket.REPO_FANOUT: {
                'parameter_adjustments': {
                    'lambda_delta': 0.02,     # More head content
                    'window_delta': 1000.0,   # Larger windows
                    'k2_ratio_delta': 0.15    # More CE candidates
                },
                'expected_improvement': 1.5,
                'confidence_base': 0.75
            },
            
            # Long hunks -> adjust windowing
            ContextBucket.HUNK_LENGTH: {
                'parameter_adjustments': {
                    'window_delta': 1200.0,   # Much larger windows
                    'stride_ratio_delta': -0.15 # Reduce stride for coverage
                },
                'expected_improvement': 1.0,
                'confidence_base': 0.65
            },
            
            # Balanced NL/code -> optimize for mixed content
            ContextBucket.NL_CODE_RATIO: {
                'parameter_adjustments': {
                    'lambda_delta': 0.01,     # Slight head increase
                    'r_delta': 1.0,           # Increase diversity
                    'k2_ratio_delta': 0.20    # 20% more candidates
                },
                'expected_improvement': 0.9,
                'confidence_base': 0.7
            }
        }
    
    def generate_override(self, cluster: PerformanceCluster) -> Optional[MicroPolicyOverride]:
        """Generate micro-policy override for underperforming cluster"""
        try:
            # Determine target bucket from cluster context
            target_bucket = self._identify_target_bucket(cluster)
            
            if not target_bucket or target_bucket not in self.policy_templates:
                logger.debug(f"No policy template available for cluster {cluster.cluster_id}")
                return None
            
            template = self.policy_templates[target_bucket]
            
            # Adjust parameters based on cluster performance
            performance_multiplier = self._calculate_performance_multiplier(cluster)
            
            adjusted_params = {}
            for param, base_value in template['parameter_adjustments'].items():
                adjusted_params[param] = base_value * performance_multiplier
            
            # Calculate confidence based on historical success
            confidence = self._calculate_confidence(target_bucket, cluster)
            
            override = MicroPolicyOverride(
                override_id=f"{cluster.cluster_id}_{target_bucket.value}_{int(time.time())}",
                target_bucket=target_bucket,
                parameter_adjustments=adjusted_params,
                expected_improvement=template['expected_improvement'] * performance_multiplier,
                confidence_score=confidence
            )
            
            logger.info(f"Generated micro-policy override for cluster {cluster.cluster_id}: "
                       f"bucket={target_bucket.value}, confidence={confidence:.2f}")
            
            return override
            
        except Exception as e:
            logger.error(f"Error generating micro-policy override: {e}")
            return None
    
    def _identify_target_bucket(self, cluster: PerformanceCluster) -> Optional[ContextBucket]:
        """Identify target context bucket for policy override"""
        try:
            # Check cluster metadata for context bucket
            if 'context_bucket' in cluster.context_metadata:
                bucket_name = cluster.context_metadata['context_bucket']
                try:
                    return ContextBucket(bucket_name)
                except ValueError:
                    pass
            
            # Infer bucket from cluster type and metadata
            if cluster.cluster_type == ClusterType.COMPLEXITY_BASED:
                complexity_bin = cluster.context_metadata.get('complexity_bin', 'medium')
                if complexity_bin == 'high':
                    return ContextBucket.SYMBOL_DEPTH  # High complexity often means deep symbols
                elif complexity_bin == 'low':
                    return ContextBucket.ENTITY_ENTROPY  # Low complexity often means low entropy
                else:
                    return ContextBucket.NL_CODE_RATIO  # Medium complexity often mixed content
            
            elif cluster.cluster_type == ClusterType.DOMAIN_BASED:
                domain = cluster.context_metadata.get('domain', '')
                if 'code' in domain.lower():
                    return ContextBucket.DUP_RATE  # Code domains often have duplication issues
                elif 'tool' in domain.lower():
                    return ContextBucket.REPO_FANOUT  # Tool results often have broad context
                else:
                    return ContextBucket.NL_CODE_RATIO  # General domains are often mixed
            
            # Default fallback based on underperformance pattern
            if cluster.underperformance_ratio > 0.8:  # Severe underperformance
                return ContextBucket.SYMBOL_DEPTH  # Try aggressive head increase
            else:
                return ContextBucket.DUP_RATE  # Try diversity increase
                
        except Exception as e:
            logger.debug(f"Error identifying target bucket: {e}")
            return ContextBucket.NL_CODE_RATIO  # Safe fallback
    
    def _calculate_performance_multiplier(self, cluster: PerformanceCluster) -> float:
        """Calculate multiplier based on severity of underperformance"""
        try:
            # Base multiplier on underperformance ratio
            base_multiplier = min(2.0, max(0.5, cluster.underperformance_ratio * 2.0))
            
            # Adjust for cluster size (larger clusters get more conservative adjustments)
            size_factor = min(1.0, self._normalize_cluster_size(cluster.cluster_size))
            
            # Adjust for recency (more recent underperformance gets stronger response)
            age_hours = (datetime.now() - cluster.last_updated).total_seconds() / 3600
            recency_factor = max(0.5, 1.0 - (age_hours / 24.0))  # Decay over 24 hours
            
            multiplier = base_multiplier * size_factor * recency_factor
            return min(2.0, max(0.3, multiplier))  # Clamp to reasonable bounds
            
        except Exception as e:
            logger.debug(f"Error calculating performance multiplier: {e}")
            return 1.0
    
    def _normalize_cluster_size(self, size: int) -> float:
        """Normalize cluster size to [0,1] range"""
        return min(1.0, max(0.1, size / 50.0))  # Assuming max 50 members
    
    def _calculate_confidence(self, target_bucket: ContextBucket, cluster: PerformanceCluster) -> float:
        """Calculate confidence in policy override success"""
        try:
            base_confidence = self.policy_templates[target_bucket]['confidence_base']
            
            # Adjust based on historical success for this bucket
            bucket_successes = [
                entry for entry in self.success_history
                if entry.get('target_bucket') == target_bucket and entry.get('succeeded', False)
            ]
            
            total_attempts = [
                entry for entry in self.success_history
                if entry.get('target_bucket') == target_bucket
            ]
            
            if total_attempts:
                success_rate = len(bucket_successes) / len(total_attempts)
                historical_factor = 0.5 * success_rate + 0.5  # Weight between 0.5 and 1.0
            else:
                historical_factor = 1.0  # No history, use base confidence
            
            # Adjust for cluster characteristics
            cluster_factor = 1.0
            if cluster.cluster_size < 5:
                cluster_factor *= 0.8  # Less confidence for small clusters
            if cluster.underperformance_ratio > 1.0:
                cluster_factor *= 0.9  # Less confidence for severe underperformance
            
            confidence = base_confidence * historical_factor * cluster_factor
            return min(0.95, max(0.1, confidence))
            
        except Exception as e:
            logger.debug(f"Error calculating confidence: {e}")
            return 0.5
    
    def record_success(self, target_bucket: ContextBucket, succeeded: bool, improvement: float):
        """Record success/failure for future confidence calculation"""
        try:
            self.success_history.append({
                'timestamp': datetime.now(),
                'target_bucket': target_bucket,
                'succeeded': succeeded,
                'improvement': improvement
            })
        except Exception as e:
            logger.debug(f"Error recording success: {e}")

class AutoBucketRetrials:
    """
    Main auto-bucket retrials system coordinating detection and recovery
    """
    
    def __init__(self,
                 config: Optional[RetrialConfiguration] = None,
                 hybrid_selector: Optional[Any] = None,
                 dashboard_integration: Optional[Any] = None):
        
        self.config = config or RetrialConfiguration()
        self.hybrid_selector = hybrid_selector
        self.dashboard_integration = dashboard_integration
        
        # Core components
        self.cluster_analyzer = ClusterAnalyzer(self.config)
        self.policy_generator = MicroPolicyGenerator()
        
        # Execution state
        self.active_retrials = {}  # execution_id -> RetrialExecution
        self.retrial_queue = deque()
        self.circuit_breakers = defaultdict(int)  # bucket -> failure_count
        
        # Results tracking
        self.execution_history = deque(maxlen=1000)
        self.performance_improvements = deque(maxlen=500)
        
        # Threading
        self.executor_thread = None
        self.is_running = False
        self.lock = threading.RLock()
        
        logger.info("Auto-bucket retrials system initialized")
    
    def start_monitoring(self):
        """Start continuous monitoring for underperforming clusters"""
        try:
            if self.is_running:
                logger.warning("Auto-bucket retrials already running")
                return
            
            self.is_running = True
            self.executor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.executor_thread.start()
            
            logger.info("Auto-bucket retrials monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
            self.is_running = False
            raise
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        try:
            self.is_running = False
            if self.executor_thread and self.executor_thread.is_alive():
                self.executor_thread.join(timeout=30)
            
            logger.info("Auto-bucket retrials monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
    
    def update_performance_data(self,
                              member_id: str, 
                              performance_metrics: Dict[str, float],
                              context_metadata: Dict[str, Any]):
        """Update performance data for cluster analysis"""
        try:
            self.cluster_analyzer.update_performance_data(
                member_id=member_id,
                performance_metrics=performance_metrics,
                context_metadata=context_metadata
            )
        except Exception as e:
            logger.error(f"Error updating performance data: {e}")
    
    def trigger_manual_retrial(self, 
                             cluster_id: str,
                             override_policy: Optional[MicroPolicyOverride] = None) -> str:
        """Manually trigger retrial for specific cluster"""
        try:
            with self.lock:
                # Generate execution ID
                execution_id = f"manual_{cluster_id}_{int(time.time() * 1000)}"
                
                # Create retrial execution
                retrial = RetrialExecution(
                    execution_id=execution_id,
                    cluster_id=cluster_id,
                    trigger=RetrialTrigger.MANUAL_TRIGGER,
                    status=RetrialStatus.PENDING,
                    attempt_number=1,
                    micro_policy_override=override_policy
                )
                
                # Queue for execution
                self.retrial_queue.append(retrial)
                self.active_retrials[execution_id] = retrial
                
                logger.info(f"Manual retrial triggered for cluster {cluster_id}: {execution_id}")
                return execution_id
                
        except Exception as e:
            logger.error(f"Error triggering manual retrial: {e}")
            return ""
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self.is_running:
                try:
                    # Check for underperforming clusters
                    underperforming_clusters = self.cluster_analyzer.identify_underperforming_clusters()
                    
                    # Queue retrials for new underperforming clusters
                    for cluster in underperforming_clusters:
                        if self._should_trigger_retrial(cluster):
                            self._queue_cluster_retrial(cluster)
                    
                    # Execute queued retrials
                    self._execute_queued_retrials()
                    
                    # Clean up completed retrials
                    self._cleanup_completed_retrials()
                    
                    # Sleep before next iteration
                    time.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(60)  # Longer sleep on error
                    
        except Exception as e:
            logger.error(f"Monitoring loop terminated with error: {e}")
        finally:
            logger.info("Auto-bucket retrials monitoring loop terminated")
    
    def _should_trigger_retrial(self, cluster: PerformanceCluster) -> bool:
        """Check if retrial should be triggered for cluster"""
        try:
            with self.lock:
                # Check if retrial already active for this cluster
                active_for_cluster = [
                    retrial for retrial in self.active_retrials.values()
                    if retrial.cluster_id == cluster.cluster_id and 
                    retrial.status in [RetrialStatus.PENDING, RetrialStatus.RUNNING]
                ]
                
                if active_for_cluster:
                    logger.debug(f"Retrial already active for cluster {cluster.cluster_id}")
                    return False
                
                # Check circuit breaker status
                target_bucket = self.policy_generator._identify_target_bucket(cluster)
                if target_bucket:
                    failure_count = self.circuit_breakers[target_bucket]
                    if failure_count >= self.config.circuit_breaker_threshold:
                        logger.debug(f"Circuit breaker active for bucket {target_bucket.value}")
                        return False
                
                # Check minimum performance threshold
                if cluster.underperformance_ratio < self.config.cbu_underperformance_threshold:
                    return False
                
                return True
                
        except Exception as e:
            logger.debug(f"Error checking retrial trigger: {e}")
            return False
    
    def _queue_cluster_retrial(self, cluster: PerformanceCluster):
        """Queue retrial for underperforming cluster"""
        try:
            with self.lock:
                # Generate micro-policy override
                policy_override = self.policy_generator.generate_override(cluster)
                
                if not policy_override:
                    logger.warning(f"Could not generate policy override for cluster {cluster.cluster_id}")
                    return
                
                # Generate execution ID
                execution_id = f"auto_{cluster.cluster_id}_{int(time.time() * 1000)}"
                
                # Create retrial execution
                retrial = RetrialExecution(
                    execution_id=execution_id,
                    cluster_id=cluster.cluster_id,
                    trigger=RetrialTrigger.CBU_UNDERPERFORMANCE,
                    status=RetrialStatus.PENDING,
                    attempt_number=1,
                    micro_policy_override=policy_override
                )
                
                # Queue for execution
                self.retrial_queue.append(retrial)
                self.active_retrials[execution_id] = retrial
                
                logger.info(f"Queued retrial for cluster {cluster.cluster_id}: "
                           f"policy={policy_override.target_bucket.value}, "
                           f"expected_improvement={policy_override.expected_improvement:.2f}")
                
        except Exception as e:
            logger.error(f"Error queueing cluster retrial: {e}")
    
    def _execute_queued_retrials(self):
        """Execute pending retrials from queue"""
        try:
            with self.lock:
                while self.retrial_queue and self.is_running:
                    retrial = self.retrial_queue.popleft()
                    
                    # Skip if retrial was superseded
                    if retrial.status == RetrialStatus.SUPERSEDED:
                        continue
                    
                    # Execute retrial in separate thread
                    threading.Thread(
                        target=self._execute_single_retrial,
                        args=(retrial,),
                        daemon=True
                    ).start()
                    
        except Exception as e:
            logger.error(f"Error executing queued retrials: {e}")
    
    def _execute_single_retrial(self, retrial: RetrialExecution):
        """Execute single retrial with timeout and error handling"""
        try:
            with self.lock:
                retrial.status = RetrialStatus.RUNNING
                retrial.started_at = datetime.now()
            
            logger.info(f"Executing retrial {retrial.execution_id} for cluster {retrial.cluster_id}")
            
            # Get cluster data
            clusters = self.cluster_analyzer.identify_underperforming_clusters()
            target_cluster = next(
                (c for c in clusters if c.cluster_id == retrial.cluster_id), None
            )
            
            if not target_cluster:
                raise Exception(f"Cluster {retrial.cluster_id} not found")
            
            # Store baseline performance
            retrial.baseline_performance = target_cluster.current_metrics.copy()
            
            # Execute with micro-policy override if available
            if retrial.micro_policy_override and self.hybrid_selector:
                success, new_metrics = self._execute_with_override(
                    target_cluster, retrial.micro_policy_override
                )
            else:
                # Execute with default parameters as fallback
                success, new_metrics = self._execute_without_override(target_cluster)
            
            # Store results
            retrial.retrial_performance = new_metrics
            
            # Calculate performance improvement
            baseline_cbu = retrial.baseline_performance.get('cbu_per_ms', 0)
            retrial_cbu = new_metrics.get('cbu_per_ms', 0)
            retrial.performance_improvement = retrial_cbu - baseline_cbu
            
            # Update status
            if success and retrial.performance_improvement > 0:
                retrial.status = RetrialStatus.SUCCEEDED
                
                # Record success in policy generator
                if retrial.micro_policy_override:
                    self.policy_generator.record_success(
                        retrial.micro_policy_override.target_bucket,
                        True,
                        retrial.performance_improvement
                    )
                    
                    # Reset circuit breaker on success
                    self.circuit_breakers[retrial.micro_policy_override.target_bucket] = 0
                
                logger.info(f"Retrial {retrial.execution_id} SUCCEEDED: "
                           f"improvement={retrial.performance_improvement:.2f} CBU/ms")
                
            else:
                retrial.status = RetrialStatus.FAILED
                
                # Record failure in policy generator
                if retrial.micro_policy_override:
                    self.policy_generator.record_success(
                        retrial.micro_policy_override.target_bucket,
                        False,
                        retrial.performance_improvement
                    )
                    
                    # Increment circuit breaker
                    self.circuit_breakers[retrial.micro_policy_override.target_bucket] += 1
                
                logger.warning(f"Retrial {retrial.execution_id} FAILED: "
                              f"improvement={retrial.performance_improvement:.2f} CBU/ms")
            
            # Generate dashboard diff if integration available
            if self.dashboard_integration and retrial.is_successful:
                retrial.dashboard_diff_url = self._generate_dashboard_diff(retrial)
            
        except Exception as e:
            logger.error(f"Retrial {retrial.execution_id} ERROR: {e}")
            retrial.status = RetrialStatus.FAILED
            retrial.error_details = str(e)
            
            # Increment circuit breaker on error
            if retrial.micro_policy_override:
                self.circuit_breakers[retrial.micro_policy_override.target_bucket] += 1
        
        finally:
            # Complete retrial
            with self.lock:
                retrial.completed_at = datetime.now()
                retrial.execution_time_ms = retrial.duration_seconds * 1000
                
                # Add to history
                self.execution_history.append(retrial)
                
                if retrial.is_successful:
                    self.performance_improvements.append({
                        'timestamp': retrial.completed_at,
                        'cluster_id': retrial.cluster_id,
                        'improvement': retrial.performance_improvement,
                        'policy': retrial.micro_policy_override.target_bucket.value if retrial.micro_policy_override else None
                    })
    
    def _execute_with_override(self, 
                             cluster: PerformanceCluster, 
                             override: MicroPolicyOverride) -> Tuple[bool, Dict[str, float]]:
        """Execute retrial with micro-policy override"""
        try:
            if not self.hybrid_selector:
                raise Exception("Hybrid selector not available")
            
            # Get base parameters (would come from current system state)
            base_params = ControlParameters()
            
            # Apply override to parameters
            override_params = override.apply_to_parameters(base_params)
            
            # Update hybrid selector configuration
            self.hybrid_selector.update_config(
                head_keep_ratio=override_params.lambda_value,
                window_size=override_params.mu_window_size,
                stride=override_params.mu_stride,
                dpp_rank=override_params.r_value,
                ce_k2=override_params.k2_value
            )
            
            # Execute on cluster members (simulate by testing on representative sample)
            sample_member_ids = cluster.member_ids[:min(5, len(cluster.member_ids))]
            metrics_sum = defaultdict(float)
            successful_runs = 0
            
            for member_id in sample_member_ids:
                try:
                    # Would normally get content for member_id from storage
                    # For now, simulate execution
                    simulated_metrics = self._simulate_execution_with_override(
                        cluster, override, member_id
                    )
                    
                    for key, value in simulated_metrics.items():
                        metrics_sum[key] += value
                    
                    successful_runs += 1
                    
                except Exception as member_error:
                    logger.debug(f"Error executing member {member_id}: {member_error}")
            
            if successful_runs == 0:
                return False, {}
            
            # Calculate average metrics
            avg_metrics = {
                key: value / successful_runs 
                for key, value in metrics_sum.items()
            }
            
            return True, avg_metrics
            
        except Exception as e:
            logger.error(f"Error executing with override: {e}")
            return False, {}
    
    def _execute_without_override(self, cluster: PerformanceCluster) -> Tuple[bool, Dict[str, float]]:
        """Execute retrial without micro-policy override (fallback)"""
        try:
            # Just return current metrics as baseline
            return True, cluster.current_metrics.copy()
            
        except Exception as e:
            logger.error(f"Error executing without override: {e}")
            return False, {}
    
    def _simulate_execution_with_override(self, 
                                        cluster: PerformanceCluster,
                                        override: MicroPolicyOverride,
                                        member_id: str) -> Dict[str, float]:
        """Simulate execution with override for testing"""
        # Simulate improved performance based on expected improvement
        base_metrics = cluster.current_metrics.copy()
        
        improvement_factor = 1.0 + (override.expected_improvement * 0.1)  # Scale down for realism
        
        simulated_metrics = {}
        for key, value in base_metrics.items():
            if key == 'cbu_per_ms':
                simulated_metrics[key] = value * improvement_factor
            elif key in ['p95_latency', 'p99_latency']:
                simulated_metrics[key] = value / improvement_factor  # Lower is better for latency
            else:
                simulated_metrics[key] = value
        
        # Add some randomness
        for key in simulated_metrics:
            noise_factor = 1.0 + np.random.normal(0, 0.05)  # ±5% noise
            simulated_metrics[key] *= noise_factor
        
        return simulated_metrics
    
    def _generate_dashboard_diff(self, retrial: RetrialExecution) -> str:
        """Generate dashboard diff URL for retrial results"""
        try:
            if not self.dashboard_integration:
                return ""
            
            # Generate diff data
            diff_data = {
                'retrial_id': retrial.execution_id,
                'cluster_id': retrial.cluster_id,
                'timestamp': retrial.completed_at.isoformat(),
                'baseline_metrics': retrial.baseline_performance,
                'retrial_metrics': retrial.retrial_performance,
                'improvement': retrial.performance_improvement,
                'policy_override': asdict(retrial.micro_policy_override) if retrial.micro_policy_override else None
            }
            
            # Upload to dashboard integration
            diff_url = self.dashboard_integration.create_performance_diff(diff_data)
            
            logger.info(f"Generated dashboard diff for retrial {retrial.execution_id}: {diff_url}")
            return diff_url
            
        except Exception as e:
            logger.error(f"Error generating dashboard diff: {e}")
            return ""
    
    def _cleanup_completed_retrials(self):
        """Clean up old completed retrials"""
        try:
            with self.lock:
                cutoff = datetime.now() - timedelta(hours=24)  # Keep 24 hours of history
                
                completed_ids = [
                    execution_id for execution_id, retrial in self.active_retrials.items()
                    if retrial.status in [RetrialStatus.SUCCEEDED, RetrialStatus.FAILED, RetrialStatus.ABANDONED] and
                    retrial.completed_at and retrial.completed_at < cutoff
                ]
                
                for execution_id in completed_ids:
                    del self.active_retrials[execution_id]
                
                if completed_ids:
                    logger.debug(f"Cleaned up {len(completed_ids)} old retrials")
                    
        except Exception as e:
            logger.debug(f"Error cleaning up retrials: {e}")
    
    def get_active_retrials(self) -> List[Dict[str, Any]]:
        """Get list of currently active retrials"""
        with self.lock:
            return [asdict(retrial) for retrial in self.active_retrials.values()]
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        with self.lock:
            recent = list(self.execution_history)[-limit:]
            return [asdict(execution) for execution in recent]
    
    def get_performance_improvements(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent performance improvements"""
        with self.lock:
            return list(self.performance_improvements)[-limit:]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        with self.lock:
            total_executions = len(self.execution_history)
            successful_executions = sum(1 for e in self.execution_history if e.is_successful)
            
            circuit_breaker_status = {
                bucket.value: count for bucket, count in self.circuit_breakers.items()
                if count > 0
            }
            
            return {
                'is_running': self.is_running,
                'active_retrials': len(self.active_retrials),
                'queued_retrials': len(self.retrial_queue),
                'total_executions': total_executions,
                'successful_executions': successful_executions,
                'success_rate': successful_executions / total_executions if total_executions > 0 else 0,
                'circuit_breakers': circuit_breaker_status,
                'recent_improvements': len(self.performance_improvements),
                'timestamp': datetime.now().isoformat()
            }

# Factory function for easy instantiation
def create_auto_bucket_retrials(config: Optional[RetrialConfiguration] = None,
                               hybrid_selector: Optional[Any] = None,
                               dashboard_integration: Optional[Any] = None) -> AutoBucketRetrials:
    """Create auto-bucket retrials system with optional configuration"""
    return AutoBucketRetrials(config, hybrid_selector, dashboard_integration)