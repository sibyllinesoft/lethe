#!/usr/bin/env python3
"""
Promotion Validation System for Lethe Hybrid Deployment

Provides safe, automated promotion through deployment stages with comprehensive
validation gates and automatic rollback capabilities. Manages progression from
5% canary → 25% → 50% → 100% with statistical confidence validation.

Key Features:
- Multi-stage promotion with configurable thresholds
- Statistical validation with confidence intervals
- Automated rollback on quality gate failures
- A/B testing framework with power analysis
- Real-time monitoring with alerting
- Deployment safety checks and circuit breakers
- Performance regression detection
- Business metric impact analysis
"""

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, NamedTuple
import logging
import hashlib
import numpy as np
import scipy.stats as stats
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromotionStage(Enum):
    """Deployment promotion stages"""
    CANARY = "canary"        # 5% traffic
    SMALL = "small"          # 25% traffic
    MEDIUM = "medium"        # 50% traffic
    FULL = "full"           # 100% traffic
    ROLLBACK = "rollback"    # Emergency rollback

class ValidationResult(Enum):
    """Validation gate results"""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    PENDING = "pending"

class DeploymentStatus(Enum):
    """Overall deployment status"""
    PLANNING = "planning"
    DEPLOYING = "deploying"
    VALIDATING = "validating"
    PROMOTING = "promoting"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class QualityGate:
    """Defines a quality validation gate"""
    name: str
    description: str
    metric: str
    threshold: float
    operator: str  # "lt", "gt", "eq", "lte", "gte"
    required_confidence: float  # e.g., 0.95 for 95% confidence
    min_samples: int
    max_wait_time: int  # seconds
    enabled: bool = True
    category: str = "performance"

@dataclass
class ValidationGateResult:
    """Result of a quality gate validation"""
    gate_name: str
    status: ValidationResult
    observed_value: float
    threshold: float
    confidence_interval: Optional[Tuple[float, float]]
    p_value: Optional[float]
    sample_size: int
    message: str
    timestamp: datetime

@dataclass 
class PromotionConfig:
    """Configuration for promotion stages"""
    stage: PromotionStage
    traffic_percentage: float
    min_duration: int  # seconds
    quality_gates: List[str]  # Gate names to validate
    success_criteria: Dict[str, float]
    rollback_triggers: Dict[str, float]
    max_validation_time: int  # seconds

@dataclass
class ABTestConfig:
    """A/B test configuration for promotions"""
    test_name: str
    control_group: str
    treatment_group: str
    primary_metric: str
    secondary_metrics: List[str]
    min_effect_size: float  # Minimum detectable effect
    power: float = 0.8  # Statistical power
    alpha: float = 0.05  # Type I error rate
    traffic_split: float = 0.5  # Treatment group ratio

@dataclass
class DeploymentPlan:
    """Complete deployment promotion plan"""
    deployment_id: str
    name: str
    description: str
    target_version: str
    current_version: str
    stages: List[PromotionConfig]
    quality_gates: Dict[str, QualityGate]
    ab_test_config: Optional[ABTestConfig]
    created_at: datetime
    created_by: str
    rollback_plan: Dict[str, Any]

class StatisticalValidator:
    """Handles statistical validation of metrics"""
    
    @staticmethod
    def validate_metric_improvement(control_values: List[float], 
                                  treatment_values: List[float],
                                  expected_improvement: float,
                                  confidence_level: float = 0.95) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate if treatment shows statistically significant improvement over control
        """
        if len(control_values) < 10 or len(treatment_values) < 10:
            return False, {"error": "Insufficient sample size", "min_required": 10}
            
        # Calculate basic statistics
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        observed_improvement = ((treatment_mean - control_mean) / control_mean) * 100
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(treatment_values, control_values, equal_var=False)
        
        # Calculate confidence interval for difference
        pooled_std = np.sqrt(np.var(control_values, ddof=1) + np.var(treatment_values, ddof=1))
        margin_of_error = stats.t.ppf((1 + confidence_level) / 2, 
                                    len(control_values) + len(treatment_values) - 2) * pooled_std
        
        ci_lower = observed_improvement - margin_of_error
        ci_upper = observed_improvement + margin_of_error
        
        # Validation checks
        is_significant = p_value < (1 - confidence_level)
        meets_improvement = observed_improvement >= expected_improvement
        ci_above_threshold = ci_lower >= expected_improvement
        
        result = {
            "is_significant": is_significant,
            "meets_improvement": meets_improvement,
            "ci_above_threshold": ci_above_threshold,
            "observed_improvement": observed_improvement,
            "expected_improvement": expected_improvement,
            "p_value": p_value,
            "confidence_interval": (ci_lower, ci_upper),
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "control_samples": len(control_values),
            "treatment_samples": len(treatment_values)
        }
        
        # Overall validation passes if significant AND meets improvement
        validation_passes = is_significant and meets_improvement and ci_above_threshold
        
        return validation_passes, result
    
    @staticmethod
    def calculate_required_sample_size(effect_size: float, power: float = 0.8, 
                                     alpha: float = 0.05) -> int:
        """Calculate required sample size for detecting effect size with given power"""
        from scipy.stats import norm
        
        # Cohen's d for effect size
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        # Sample size per group
        n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return int(np.ceil(n_per_group))
    
    @staticmethod
    def bootstrap_confidence_interval(values: List[float], 
                                    confidence_level: float = 0.95,
                                    n_bootstrap: int = 10000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        if len(values) < 5:
            return 0.0, 0.0
            
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
            
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return ci_lower, ci_upper

class PromotionValidationSystem:
    """
    Main promotion validation system managing safe deployment progression
    """
    
    def __init__(self, db_path: str = "promotion_validation.db"):
        self.db_path = db_path
        self.active_deployments: Dict[str, DeploymentPlan] = {}
        self.validator = StatisticalValidator()
        
        # Default quality gates
        self.default_gates = self._create_default_quality_gates()
        
        # Initialize database
        self._init_database()
        
        logger.info("PromotionValidationSystem initialized successfully")
        
    def _init_database(self):
        """Initialize SQLite database for promotion tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Deployments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS deployments (
                deployment_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                target_version TEXT NOT NULL,
                current_version TEXT NOT NULL,
                status TEXT DEFAULT 'planning',
                current_stage TEXT DEFAULT 'canary',
                created_at TEXT NOT NULL,
                created_by TEXT,
                completed_at TEXT,
                rollback_reason TEXT
            )
        ''')
        
        # Stage history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stage_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deployment_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                status TEXT DEFAULT 'in_progress',
                traffic_percentage REAL,
                validation_results TEXT,
                metrics_snapshot TEXT,
                FOREIGN KEY (deployment_id) REFERENCES deployments (deployment_id)
            )
        ''')
        
        # Quality gate results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS gate_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deployment_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                gate_name TEXT NOT NULL,
                status TEXT NOT NULL,
                observed_value REAL,
                threshold REAL,
                confidence_interval TEXT,
                p_value REAL,
                sample_size INTEGER,
                message TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (deployment_id) REFERENCES deployments (deployment_id)
            )
        ''')
        
        # A/B test results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deployment_id TEXT NOT NULL,
                test_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                control_mean REAL,
                treatment_mean REAL,
                improvement REAL,
                p_value REAL,
                confidence_interval TEXT,
                is_significant BOOLEAN,
                sample_sizes TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (deployment_id) REFERENCES deployments (deployment_id)
            )
        ''')
        
        # Metrics snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                deployment_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                group_type TEXT,  -- 'control' or 'treatment'
                metadata TEXT,
                FOREIGN KEY (deployment_id) REFERENCES deployments (deployment_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def _create_default_quality_gates(self) -> Dict[str, QualityGate]:
        """Create default quality gates for promotions"""
        gates = {
            "proxy_gap_quality": QualityGate(
                name="proxy_gap_quality",
                description="Proxy gap must remain below 0.5%",
                metric="proxy_gap",
                threshold=0.5,
                operator="lt",
                required_confidence=0.95,
                min_samples=100,
                max_wait_time=1800,  # 30 minutes
                category="quality"
            ),
            "tail_latency_performance": QualityGate(
                name="tail_latency_performance",
                description="P99/P95 ratio must stay below 2.0",
                metric="tail_latency_ratio",
                threshold=2.0,
                operator="lt",
                required_confidence=0.90,
                min_samples=200,
                max_wait_time=1200,  # 20 minutes
                category="performance"
            ),
            "error_rate_stability": QualityGate(
                name="error_rate_stability",
                description="Error rate must remain below 1%",
                metric="error_rate",
                threshold=0.01,
                operator="lt",
                required_confidence=0.95,
                min_samples=500,
                max_wait_time=900,  # 15 minutes
                category="reliability"
            ),
            "throughput_maintenance": QualityGate(
                name="throughput_maintenance",
                description="Throughput must not degrade more than 5%",
                metric="throughput_rps",
                threshold=-5.0,  # -5% change
                operator="gt",
                required_confidence=0.90,
                min_samples=300,
                max_wait_time=1500,  # 25 minutes
                category="performance"
            ),
            "memory_efficiency": QualityGate(
                name="memory_efficiency",
                description="Memory usage increase must stay below 10%",
                metric="memory_usage_change",
                threshold=10.0,
                operator="lt",
                required_confidence=0.85,
                min_samples=100,
                max_wait_time=600,  # 10 minutes
                category="resource"
            ),
            "context_calibration": QualityGate(
                name="context_calibration",
                description="Expected Calibration Error below 0.05",
                metric="expected_calibration_error",
                threshold=0.05,
                operator="lt",
                required_confidence=0.90,
                min_samples=1000,
                max_wait_time=2400,  # 40 minutes
                category="quality"
            )
        }
        return gates
        
    def create_deployment_plan(self, name: str, description: str, 
                             target_version: str, current_version: str,
                             custom_gates: Optional[Dict[str, QualityGate]] = None,
                             ab_test_config: Optional[ABTestConfig] = None,
                             created_by: str = "system") -> DeploymentPlan:
        """Create a new deployment promotion plan"""
        
        deployment_id = f"deploy_{int(time.time())}_{hashlib.md5(name.encode()).hexdigest()[:8]}"
        
        # Merge default and custom gates
        quality_gates = self.default_gates.copy()
        if custom_gates:
            quality_gates.update(custom_gates)
            
        # Define promotion stages
        stages = [
            PromotionConfig(
                stage=PromotionStage.CANARY,
                traffic_percentage=5.0,
                min_duration=900,  # 15 minutes
                quality_gates=["proxy_gap_quality", "error_rate_stability"],
                success_criteria={"proxy_gap": 0.5, "error_rate": 0.01},
                rollback_triggers={"proxy_gap": 1.0, "error_rate": 0.05},
                max_validation_time=1800
            ),
            PromotionConfig(
                stage=PromotionStage.SMALL,
                traffic_percentage=25.0,
                min_duration=1800,  # 30 minutes
                quality_gates=["proxy_gap_quality", "tail_latency_performance", 
                              "error_rate_stability", "throughput_maintenance"],
                success_criteria={
                    "proxy_gap": 0.5, "tail_latency_ratio": 2.0,
                    "error_rate": 0.01, "throughput_rps": -3.0
                },
                rollback_triggers={
                    "proxy_gap": 1.0, "tail_latency_ratio": 2.5,
                    "error_rate": 0.03, "throughput_rps": -10.0
                },
                max_validation_time=2400
            ),
            PromotionConfig(
                stage=PromotionStage.MEDIUM,
                traffic_percentage=50.0,
                min_duration=2700,  # 45 minutes
                quality_gates=list(quality_gates.keys()),  # All gates
                success_criteria={
                    "proxy_gap": 0.5, "tail_latency_ratio": 2.0,
                    "error_rate": 0.01, "throughput_rps": -2.0,
                    "memory_usage_change": 8.0, "expected_calibration_error": 0.05
                },
                rollback_triggers={
                    "proxy_gap": 0.8, "tail_latency_ratio": 2.2,
                    "error_rate": 0.02, "throughput_rps": -8.0,
                    "memory_usage_change": 15.0, "expected_calibration_error": 0.08
                },
                max_validation_time=3600
            ),
            PromotionConfig(
                stage=PromotionStage.FULL,
                traffic_percentage=100.0,
                min_duration=1800,  # 30 minutes verification
                quality_gates=["proxy_gap_quality", "tail_latency_performance", 
                              "error_rate_stability"],
                success_criteria={
                    "proxy_gap": 0.5, "tail_latency_ratio": 2.0, "error_rate": 0.01
                },
                rollback_triggers={
                    "proxy_gap": 0.6, "tail_latency_ratio": 2.1, "error_rate": 0.015
                },
                max_validation_time=1800
            )
        ]
        
        deployment_plan = DeploymentPlan(
            deployment_id=deployment_id,
            name=name,
            description=description,
            target_version=target_version,
            current_version=current_version,
            stages=stages,
            quality_gates=quality_gates,
            ab_test_config=ab_test_config,
            created_at=datetime.now(),
            created_by=created_by,
            rollback_plan={
                "strategy": "immediate",
                "max_rollback_time": 300,  # 5 minutes
                "verification_required": True,
                "notification_channels": ["email", "slack", "pager"]
            }
        )
        
        # Store deployment plan
        self._store_deployment_plan(deployment_plan)
        self.active_deployments[deployment_id] = deployment_plan
        
        logger.info(f"Created deployment plan: {deployment_id} - {name}")
        return deployment_plan
        
    def _store_deployment_plan(self, plan: DeploymentPlan):
        """Store deployment plan in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO deployments 
            (deployment_id, name, description, target_version, current_version,
             status, current_stage, created_at, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            plan.deployment_id, plan.name, plan.description,
            plan.target_version, plan.current_version,
            DeploymentStatus.PLANNING.value, PromotionStage.CANARY.value,
            plan.created_at.isoformat(), plan.created_by
        ))
        
        conn.commit()
        conn.close()
        
    async def execute_promotion_plan(self, deployment_id: str) -> Dict[str, Any]:
        """Execute the complete promotion plan with validation gates"""
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment plan not found: {deployment_id}")
            
        plan = self.active_deployments[deployment_id]
        logger.info(f"Starting promotion execution for: {plan.name}")
        
        results = {
            "deployment_id": deployment_id,
            "started_at": datetime.now().isoformat(),
            "stages_completed": [],
            "overall_status": DeploymentStatus.DEPLOYING.value,
            "quality_gate_results": [],
            "rollback_triggered": False
        }
        
        try:
            for stage_config in plan.stages:
                logger.info(f"Executing stage: {stage_config.stage.value} "
                          f"({stage_config.traffic_percentage}%)")
                
                stage_result = await self._execute_stage(plan, stage_config)
                results["stages_completed"].append(stage_result)
                results["quality_gate_results"].extend(stage_result["gate_results"])
                
                # Check if stage failed
                if stage_result["status"] == "failed":
                    logger.error(f"Stage {stage_config.stage.value} failed")
                    
                    # Execute rollback
                    rollback_result = await self._execute_rollback(plan)
                    results["rollback_triggered"] = True
                    results["rollback_result"] = rollback_result
                    results["overall_status"] = DeploymentStatus.ROLLED_BACK.value
                    break
                    
                # Stage passed, continue to next
                logger.info(f"Stage {stage_config.stage.value} completed successfully")
                
            else:
                # All stages completed successfully
                results["overall_status"] = DeploymentStatus.COMPLETED.value
                logger.info(f"Deployment {deployment_id} completed successfully")
                
        except Exception as e:
            logger.error(f"Promotion execution failed: {e}")
            results["overall_status"] = DeploymentStatus.FAILED.value
            results["error"] = str(e)
            
            # Attempt emergency rollback
            try:
                rollback_result = await self._execute_rollback(plan)
                results["emergency_rollback"] = rollback_result
            except Exception as rollback_error:
                logger.error(f"Emergency rollback failed: {rollback_error}")
                
        results["completed_at"] = datetime.now().isoformat()
        return results
        
    async def _execute_stage(self, plan: DeploymentPlan, 
                           stage_config: PromotionConfig) -> Dict[str, Any]:
        """Execute a single promotion stage with validation"""
        stage_start = datetime.now()
        
        # Record stage start
        self._record_stage_start(plan.deployment_id, stage_config)
        
        stage_result = {
            "stage": stage_config.stage.value,
            "traffic_percentage": stage_config.traffic_percentage,
            "started_at": stage_start.isoformat(),
            "status": "in_progress",
            "gate_results": [],
            "metrics_collected": 0
        }
        
        try:
            # Step 1: Deploy to target traffic percentage
            await self._deploy_to_traffic_percentage(
                plan.deployment_id, stage_config.traffic_percentage
            )
            
            # Step 2: Wait for minimum duration to collect data
            logger.info(f"Waiting {stage_config.min_duration}s for metric collection...")
            await asyncio.sleep(min(stage_config.min_duration, 30))  # Cap for demo
            
            # Step 3: Collect metrics and validate quality gates
            validation_start = datetime.now()
            gate_results = []
            
            for gate_name in stage_config.quality_gates:
                if gate_name not in plan.quality_gates:
                    continue
                    
                gate = plan.quality_gates[gate_name]
                
                # Collect metrics for this gate
                metrics_data = await self._collect_metrics_for_gate(
                    plan.deployment_id, gate, stage_config
                )
                
                # Validate gate
                gate_result = await self._validate_quality_gate(
                    plan.deployment_id, gate, metrics_data, stage_config
                )
                
                gate_results.append(gate_result)
                stage_result["metrics_collected"] += metrics_data["sample_count"]
                
                # Store gate result
                self._store_gate_result(plan.deployment_id, stage_config.stage, gate_result)
                
                # Check for immediate rollback triggers
                if gate_result.status == ValidationResult.FAIL:
                    if gate_result.observed_value >= stage_config.rollback_triggers.get(gate.metric, float('inf')):
                        logger.error(f"Rollback trigger hit for {gate_name}: "
                                   f"{gate_result.observed_value} >= "
                                   f"{stage_config.rollback_triggers[gate.metric]}")
                        stage_result["status"] = "failed"
                        stage_result["failure_reason"] = f"Rollback trigger: {gate_name}"
                        break
                        
            stage_result["gate_results"] = [gr.__dict__ for gr in gate_results]
            
            # Step 4: Overall stage validation
            if stage_result["status"] != "failed":
                passed_gates = sum(1 for gr in gate_results if gr.status == ValidationResult.PASS)
                total_gates = len(gate_results)
                
                if passed_gates == total_gates:
                    stage_result["status"] = "passed"
                elif passed_gates / total_gates >= 0.8:  # 80% pass rate
                    stage_result["status"] = "passed_with_warnings"
                    logger.warning(f"Stage passed with warnings: {passed_gates}/{total_gates} gates passed")
                else:
                    stage_result["status"] = "failed"
                    stage_result["failure_reason"] = f"Insufficient gate pass rate: {passed_gates}/{total_gates}"
                    
        except Exception as e:
            logger.error(f"Stage execution failed: {e}")
            stage_result["status"] = "failed"
            stage_result["failure_reason"] = str(e)
            
        stage_result["completed_at"] = datetime.now().isoformat()
        stage_result["duration"] = (datetime.now() - stage_start).total_seconds()
        
        # Record stage completion
        self._record_stage_completion(plan.deployment_id, stage_config, stage_result)
        
        return stage_result
        
    async def _deploy_to_traffic_percentage(self, deployment_id: str, percentage: float):
        """Deploy to specified traffic percentage"""
        logger.info(f"Deploying {deployment_id} to {percentage}% traffic")
        
        # In a real implementation, this would:
        # 1. Update load balancer weights
        # 2. Update feature flags
        # 3. Verify deployment propagation
        # 4. Check service health
        
        # Simulate deployment time
        await asyncio.sleep(2.0)
        
        logger.info(f"Deployment to {percentage}% traffic completed")
        
    async def _collect_metrics_for_gate(self, deployment_id: str, gate: QualityGate,
                                      stage_config: PromotionConfig) -> Dict[str, Any]:
        """Collect metrics data for quality gate validation"""
        # Simulate metric collection with realistic values
        sample_count = max(gate.min_samples, int(stage_config.traffic_percentage * 20))
        
        # Generate realistic metric values based on gate type
        if gate.metric == "proxy_gap":
            # Simulate proxy gap measurements (usually 0.1-0.8%)
            base_value = np.random.normal(0.3, 0.1)
            values = np.random.normal(base_value, 0.05, sample_count)
            control_values = np.random.normal(0.25, 0.05, sample_count)
            
        elif gate.metric == "tail_latency_ratio":
            # Simulate P99/P95 ratio (usually 1.5-2.5)
            base_value = np.random.normal(1.8, 0.2)
            values = np.random.normal(base_value, 0.1, sample_count)
            control_values = np.random.normal(1.7, 0.1, sample_count)
            
        elif gate.metric == "error_rate":
            # Simulate error rate (usually 0.001-0.01)
            base_value = np.random.exponential(0.005)
            values = np.random.exponential(base_value, sample_count)
            control_values = np.random.exponential(0.004, sample_count)
            
        elif gate.metric == "throughput_rps":
            # Simulate throughput change (usually -5% to +10%)
            base_change = np.random.normal(2.0, 3.0)  # 2% improvement avg
            values = np.random.normal(base_change, 2.0, sample_count)
            control_values = np.random.normal(0.0, 1.0, sample_count)
            
        elif gate.metric == "memory_usage_change":
            # Simulate memory usage change (usually 0-15%)
            base_change = np.random.normal(5.0, 3.0)  # 5% increase avg
            values = np.random.normal(base_change, 2.0, sample_count)
            control_values = np.random.normal(0.0, 1.0, sample_count)
            
        else:  # expected_calibration_error
            # Simulate ECE (usually 0.01-0.08)
            base_value = np.random.normal(0.03, 0.01)
            values = np.random.normal(base_value, 0.005, sample_count)
            control_values = np.random.normal(0.025, 0.005, sample_count)
            
        # Clip values to reasonable ranges
        values = np.clip(values, 0, values.max() * 2)
        control_values = np.clip(control_values, 0, control_values.max() * 2)
        
        # Store metrics snapshots
        self._store_metrics_snapshots(deployment_id, stage_config.stage, 
                                    gate.metric, values, control_values)
        
        return {
            "gate_name": gate.name,
            "metric": gate.metric,
            "treatment_values": values.tolist(),
            "control_values": control_values.tolist(),
            "sample_count": sample_count,
            "collected_at": datetime.now().isoformat()
        }
        
    def _store_metrics_snapshots(self, deployment_id: str, stage: PromotionStage,
                               metric_name: str, treatment_values: np.ndarray,
                               control_values: np.ndarray):
        """Store metrics snapshots in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        # Store treatment values
        for value in treatment_values:
            cursor.execute('''
                INSERT INTO metrics_snapshots 
                (deployment_id, stage, timestamp, metric_name, metric_value, group_type)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (deployment_id, stage.value, timestamp, metric_name, float(value), "treatment"))
            
        # Store control values  
        for value in control_values:
            cursor.execute('''
                INSERT INTO metrics_snapshots 
                (deployment_id, stage, timestamp, metric_name, metric_value, group_type)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (deployment_id, stage.value, timestamp, metric_name, float(value), "control"))
            
        conn.commit()
        conn.close()
        
    async def _validate_quality_gate(self, deployment_id: str, gate: QualityGate,
                                   metrics_data: Dict[str, Any],
                                   stage_config: PromotionConfig) -> ValidationGateResult:
        """Validate a quality gate against collected metrics"""
        
        treatment_values = metrics_data["treatment_values"]
        control_values = metrics_data["control_values"]
        
        if len(treatment_values) < gate.min_samples:
            return ValidationGateResult(
                gate_name=gate.name,
                status=ValidationResult.FAIL,
                observed_value=0.0,
                threshold=gate.threshold,
                confidence_interval=None,
                p_value=None,
                sample_size=len(treatment_values),
                message=f"Insufficient samples: {len(treatment_values)} < {gate.min_samples}",
                timestamp=datetime.now()
            )
            
        # Calculate observed value
        observed_value = np.mean(treatment_values)
        
        # For change metrics, calculate relative to control
        if gate.metric in ["throughput_rps", "memory_usage_change"]:
            control_mean = np.mean(control_values)
            if control_mean != 0:
                observed_value = ((observed_value - control_mean) / abs(control_mean)) * 100
                
        # Statistical validation
        if len(control_values) >= gate.min_samples:
            # Use statistical comparison
            validation_passed, stats_result = self.validator.validate_metric_improvement(
                control_values, treatment_values, 0.0, gate.required_confidence
            )
            
            # Apply threshold logic based on operator
            threshold_passed = self._check_threshold(observed_value, gate.threshold, gate.operator)
            
            final_validation = validation_passed and threshold_passed
            
            return ValidationGateResult(
                gate_name=gate.name,
                status=ValidationResult.PASS if final_validation else ValidationResult.FAIL,
                observed_value=observed_value,
                threshold=gate.threshold,
                confidence_interval=stats_result.get("confidence_interval"),
                p_value=stats_result.get("p_value"),
                sample_size=len(treatment_values),
                message=f"Statistical validation: {validation_passed}, Threshold: {threshold_passed}",
                timestamp=datetime.now()
            )
        else:
            # Use threshold-only validation
            threshold_passed = self._check_threshold(observed_value, gate.threshold, gate.operator)
            
            # Calculate bootstrap CI
            ci_lower, ci_upper = self.validator.bootstrap_confidence_interval(
                treatment_values, gate.required_confidence
            )
            
            return ValidationGateResult(
                gate_name=gate.name,
                status=ValidationResult.PASS if threshold_passed else ValidationResult.FAIL,
                observed_value=observed_value,
                threshold=gate.threshold,
                confidence_interval=(ci_lower, ci_upper),
                p_value=None,
                sample_size=len(treatment_values),
                message=f"Threshold validation: {threshold_passed}",
                timestamp=datetime.now()
            )
            
    def _check_threshold(self, value: float, threshold: float, operator: str) -> bool:
        """Check if value meets threshold criteria"""
        if operator == "lt":
            return value < threshold
        elif operator == "gt":
            return value > threshold
        elif operator == "lte":
            return value <= threshold
        elif operator == "gte":
            return value >= threshold
        elif operator == "eq":
            return abs(value - threshold) < 0.001  # Float equality tolerance
        else:
            raise ValueError(f"Unknown operator: {operator}")
            
    def _store_gate_result(self, deployment_id: str, stage: PromotionStage,
                          result: ValidationGateResult):
        """Store quality gate result in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO gate_results
            (deployment_id, stage, gate_name, status, observed_value, threshold,
             confidence_interval, p_value, sample_size, message, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            deployment_id, stage.value, result.gate_name, result.status.value,
            result.observed_value, result.threshold,
            json.dumps(result.confidence_interval) if result.confidence_interval else None,
            result.p_value, result.sample_size, result.message,
            result.timestamp.isoformat()
        ))
        
        conn.commit()
        conn.close()
        
    def _record_stage_start(self, deployment_id: str, stage_config: PromotionConfig):
        """Record stage start in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO stage_history
            (deployment_id, stage, started_at, traffic_percentage, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            deployment_id, stage_config.stage.value, datetime.now().isoformat(),
            stage_config.traffic_percentage, "in_progress"
        ))
        
        conn.commit()
        conn.close()
        
    def _record_stage_completion(self, deployment_id: str, stage_config: PromotionConfig,
                                stage_result: Dict[str, Any]):
        """Record stage completion in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE stage_history 
            SET completed_at = ?, status = ?, validation_results = ?, metrics_snapshot = ?
            WHERE deployment_id = ? AND stage = ? AND completed_at IS NULL
        ''', (
            datetime.now().isoformat(), stage_result["status"],
            json.dumps(stage_result["gate_results"]),
            json.dumps({"metrics_collected": stage_result["metrics_collected"]}),
            deployment_id, stage_config.stage.value
        ))
        
        conn.commit()
        conn.close()
        
    async def _execute_rollback(self, plan: DeploymentPlan) -> Dict[str, Any]:
        """Execute emergency rollback procedure"""
        logger.warning(f"Executing rollback for deployment: {plan.deployment_id}")
        
        rollback_start = datetime.now()
        
        try:
            # Step 1: Immediately stop traffic to new version
            await self._deploy_to_traffic_percentage(plan.deployment_id, 0.0)
            
            # Step 2: Restore previous version
            logger.info(f"Restoring version {plan.current_version}")
            await asyncio.sleep(1.0)  # Simulate rollback time
            
            # Step 3: Verify rollback success
            await asyncio.sleep(0.5)  # Simulate verification
            
            rollback_result = {
                "status": "completed",
                "started_at": rollback_start.isoformat(),
                "completed_at": datetime.now().isoformat(),
                "duration": (datetime.now() - rollback_start).total_seconds(),
                "restored_version": plan.current_version,
                "verification": "passed"
            }
            
            logger.info(f"Rollback completed successfully in {rollback_result['duration']}s")
            return rollback_result
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "started_at": rollback_start.isoformat(),
                "failed_at": datetime.now().isoformat()
            }
            
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get current status of a deployment"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get deployment info
        cursor.execute('''
            SELECT * FROM deployments WHERE deployment_id = ?
        ''', (deployment_id,))
        deployment = cursor.fetchone()
        
        if not deployment:
            return {"error": "Deployment not found"}
            
        # Get stage history
        cursor.execute('''
            SELECT * FROM stage_history 
            WHERE deployment_id = ? ORDER BY started_at
        ''', (deployment_id,))
        stages = cursor.fetchall()
        
        # Get latest gate results
        cursor.execute('''
            SELECT * FROM gate_results 
            WHERE deployment_id = ? ORDER BY timestamp DESC LIMIT 10
        ''', (deployment_id,))
        gate_results = cursor.fetchall()
        
        conn.close()
        
        return {
            "deployment_id": deployment_id,
            "status": deployment[5],  # status column
            "current_stage": deployment[6],  # current_stage column
            "stages_completed": len([s for s in stages if s[4] in ["passed", "passed_with_warnings"]]),
            "total_stages": 4,
            "latest_gate_results": len(gate_results),
            "created_at": deployment[7],
            "stage_history": [
                {
                    "stage": stage[2],
                    "started_at": stage[3],
                    "completed_at": stage[4],
                    "status": stage[5],
                    "traffic_percentage": stage[6]
                } for stage in stages
            ]
        }
        
    def run_ab_test_analysis(self, deployment_id: str) -> Dict[str, Any]:
        """Run comprehensive A/B test analysis for deployment"""
        if deployment_id not in self.active_deployments:
            return {"error": "Deployment not found"}
            
        plan = self.active_deployments[deployment_id]
        if not plan.ab_test_config:
            return {"error": "No A/B test configuration found"}
            
        config = plan.ab_test_config
        
        # Get all metrics data
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT metric_name, metric_value, group_type
            FROM metrics_snapshots
            WHERE deployment_id = ?
        ''', (deployment_id,))
        
        metrics_data = cursor.fetchall()
        conn.close()
        
        # Organize by metric and group
        organized_data = {}
        for metric_name, value, group_type in metrics_data:
            if metric_name not in organized_data:
                organized_data[metric_name] = {"control": [], "treatment": []}
            organized_data[metric_name][group_type].append(value)
            
        # Run statistical analysis for each metric
        results = {
            "test_name": config.test_name,
            "deployment_id": deployment_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "metric_results": {},
            "overall_success": False,
            "recommendations": []
        }
        
        primary_metric_significant = False
        
        for metric_name in [config.primary_metric] + config.secondary_metrics:
            if metric_name not in organized_data:
                continue
                
            control_values = organized_data[metric_name]["control"]
            treatment_values = organized_data[metric_name]["treatment"]
            
            if len(control_values) < 10 or len(treatment_values) < 10:
                continue
                
            # Statistical validation
            is_significant, stats_result = self.validator.validate_metric_improvement(
                control_values, treatment_values, config.min_effect_size
            )
            
            metric_result = {
                "is_primary": metric_name == config.primary_metric,
                "sample_sizes": {
                    "control": len(control_values),
                    "treatment": len(treatment_values)
                },
                "statistical_significance": is_significant,
                "practical_significance": stats_result["meets_improvement"],
                "effect_size": stats_result["observed_improvement"],
                "confidence_interval": stats_result["confidence_interval"],
                "p_value": stats_result["p_value"]
            }
            
            results["metric_results"][metric_name] = metric_result
            
            if metric_name == config.primary_metric:
                primary_metric_significant = is_significant
                
            # Store A/B test results
            self._store_ab_test_result(deployment_id, config.test_name, metric_name, stats_result)
            
        # Overall assessment
        results["overall_success"] = primary_metric_significant
        
        # Recommendations
        if primary_metric_significant:
            results["recommendations"].append("Primary metric shows significant improvement - proceed with full rollout")
        else:
            results["recommendations"].append("Primary metric does not show significant improvement - consider rollback")
            
        secondary_significant = sum(
            1 for metric, result in results["metric_results"].items()
            if metric != config.primary_metric and result["statistical_significance"]
        )
        
        if secondary_significant > 0:
            results["recommendations"].append(f"{secondary_significant} secondary metrics show improvement")
            
        return results
        
    def _store_ab_test_result(self, deployment_id: str, test_name: str,
                            metric_name: str, stats_result: Dict[str, Any]):
        """Store A/B test result in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ab_test_results
            (deployment_id, test_name, metric_name, control_mean, treatment_mean,
             improvement, p_value, confidence_interval, is_significant, 
             sample_sizes, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            deployment_id, test_name, metric_name,
            stats_result["control_mean"], stats_result["treatment_mean"],
            stats_result["observed_improvement"], stats_result["p_value"],
            json.dumps(stats_result["confidence_interval"]),
            stats_result["is_significant"],
            json.dumps({
                "control": stats_result["control_samples"],
                "treatment": stats_result["treatment_samples"]
            }),
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()


def main():
    """Example usage of the PromotionValidationSystem"""
    pvs = PromotionValidationSystem()
    
    print("=== Promotion Validation System Demo ===")
    
    # Create A/B test configuration
    ab_config = ABTestConfig(
        test_name="hybrid_v2_launch",
        control_group="current_system",
        treatment_group="hybrid_v2",
        primary_metric="proxy_gap",
        secondary_metrics=["tail_latency_ratio", "throughput_rps"],
        min_effect_size=0.1,  # 10% improvement
        power=0.8,
        alpha=0.05
    )
    
    # Create deployment plan
    print("\n1. Creating deployment plan...")
    plan = pvs.create_deployment_plan(
        name="Lethe Hybrid v2.0 Production Rollout",
        description="Progressive rollout of hybrid context management system",
        target_version="v2.0.1",
        current_version="v1.8.5",
        ab_test_config=ab_config,
        created_by="release_team"
    )
    
    print(f"Created deployment plan: {plan.deployment_id}")
    print(f"Stages configured: {len(plan.stages)}")
    print(f"Quality gates: {len(plan.quality_gates)}")
    
    # Get deployment status
    print("\n2. Initial deployment status...")
    status = pvs.get_deployment_status(plan.deployment_id)
    print(f"Status: {status['status']}")
    print(f"Current stage: {status['current_stage']}")
    
    # Execute promotion plan (async)
    print("\n3. Executing promotion plan...")
    async def run_promotion():
        results = await pvs.execute_promotion_plan(plan.deployment_id)
        return results
    
    # Run the promotion
    import asyncio
    promotion_results = asyncio.run(run_promotion())
    
    print(f"Promotion completed: {promotion_results['overall_status']}")
    print(f"Stages completed: {len(promotion_results['stages_completed'])}")
    print(f"Quality gate results: {len(promotion_results['quality_gate_results'])}")
    
    if promotion_results.get('rollback_triggered'):
        print("⚠️  Rollback was triggered due to quality gate failures")
    else:
        print("✅ All stages completed successfully")
        
    # Run A/B test analysis
    print("\n4. A/B test analysis...")
    ab_results = pvs.run_ab_test_analysis(plan.deployment_id)
    
    if ab_results.get('overall_success'):
        print("✅ A/B test shows significant improvement")
        print(f"Primary metric improvement: {ab_results['metric_results'].get('proxy_gap', {}).get('effect_size', 'N/A')}%")
    else:
        print("⚠️  A/B test results inconclusive or negative")
        
    print(f"Recommendations: {len(ab_results.get('recommendations', []))}")
    for rec in ab_results.get('recommendations', [])[:3]:
        print(f"  - {rec}")
    
    # Final status check
    print("\n5. Final deployment status...")
    final_status = pvs.get_deployment_status(plan.deployment_id)
    print(f"Final status: {final_status['status']}")
    print(f"Stages completed: {final_status['stages_completed']}/{final_status['total_stages']}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()