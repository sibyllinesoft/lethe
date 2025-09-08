#!/usr/bin/env python3
"""
Campaign Monitoring and Reporting System
========================================

Comprehensive monitoring and reporting for campaign execution including:
1. Real-time campaign progress monitoring
2. Resource usage and performance tracking  
3. Alert system for failures and anomalies
4. Detailed reporting and analytics
5. Historical trend analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque
import sqlite3

# Integration with existing framework
from campaign_manager import Campaign, CampaignManager, CampaignStatus, Trial, TrialStatus
from validation import PromotionDecision, ValidationResult, GuardrailViolation
from priority_scoring import CampaignPriority

logger = logging.getLogger(__name__)

@dataclass
class MetricSnapshot:
    """Single metric measurement at a point in time"""
    timestamp: datetime
    campaign_id: str
    metric_name: str
    metric_value: float
    trial_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """System alert for campaign issues"""
    alert_id: str
    timestamp: datetime
    severity: str  # "low", "medium", "high", "critical"
    category: str  # "performance", "failure", "resource", "validation"
    
    campaign_id: Optional[str]
    message: str
    details: Dict[str, Any]
    
    # Alert state
    acknowledged: bool = False
    resolved: bool = False
    resolution_notes: Optional[str] = None

@dataclass
class CampaignHealthStatus:
    """Overall health status for a campaign"""
    campaign_id: str
    overall_health: str  # "healthy", "warning", "critical", "failed"
    
    # Component health
    execution_health: str
    resource_health: str 
    quality_health: str
    validation_health: str
    
    # Key metrics
    progress_percentage: float
    success_rate: float
    average_trial_duration: float
    resource_utilization: float
    
    # Issues summary
    active_alerts: int
    critical_alerts: int
    recent_failures: int
    
    last_updated: datetime

class CampaignMonitor:
    """Real-time campaign monitoring system"""
    
    def __init__(self, 
                 db_path: str = "./campaign_monitoring.db",
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 polling_interval: float = 30.0):
        """
        Initialize campaign monitor.
        
        Args:
            db_path: SQLite database for storing metrics
            alert_thresholds: Custom alert thresholds
            polling_interval: How often to check campaigns (seconds)
        """
        self.db_path = db_path
        self.polling_interval = polling_interval
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Alert system
        self.alerts: Dict[str, Alert] = {}
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Metric storage
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.current_snapshots: Dict[str, Dict[str, MetricSnapshot]] = defaultdict(dict)
        
        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "max_trial_duration_minutes": 60.0,
            "min_success_rate": 0.3,
            "max_memory_usage_gb": 8.0,
            "max_cpu_usage_percent": 80.0,
            "max_consecutive_failures": 3,
            "min_progress_rate": 0.05,  # trials per hour
            "max_latency_p95_ms": 5000.0
        }
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Initialized CampaignMonitor with polling interval {polling_interval}s")
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    campaign_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    trial_id TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    category TEXT NOT NULL,
                    campaign_id TEXT,
                    message TEXT NOT NULL,
                    details TEXT NOT NULL,
                    acknowledged INTEGER DEFAULT 0,
                    resolved INTEGER DEFAULT 0,
                    resolution_notes TEXT
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_campaign ON metrics(campaign_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_campaign ON alerts(campaign_id)")
            
        logger.info(f"Initialized monitoring database at {self.db_path}")
    
    def start_monitoring(self, campaign_manager: CampaignManager):
        """Start monitoring campaigns"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.campaign_manager = campaign_manager
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="CampaignMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("Started campaign monitoring")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Stopped campaign monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                start_time = time.time()
                
                # Monitor all active campaigns
                for campaign_id, campaign in self.campaign_manager.campaigns.items():
                    if campaign.status in [CampaignStatus.RUNNING, CampaignStatus.PENDING]:
                        self._monitor_campaign(campaign)
                
                # Check for system-wide issues
                self._check_system_health()
                
                # Sleep until next poll
                elapsed = time.time() - start_time
                sleep_time = max(0, self.polling_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.polling_interval)
    
    def _monitor_campaign(self, campaign: Campaign):
        """Monitor individual campaign"""
        campaign_id = campaign.campaign_id
        now = datetime.now()
        
        # Collect basic metrics
        total_trials = len(campaign.trials)
        completed_trials = len([t for t in campaign.trials if t.status == TrialStatus.COMPLETED])
        failed_trials = len([t for t in campaign.trials if t.status == TrialStatus.FAILED])
        running_trials = len([t for t in campaign.trials if t.status == TrialStatus.RUNNING])
        
        # Progress metrics
        progress_percentage = (completed_trials + failed_trials) / campaign.spec.n_trials * 100
        success_rate = completed_trials / max(1, completed_trials + failed_trials)
        
        # Record metrics
        self._record_metric(campaign_id, "progress_percentage", progress_percentage)
        self._record_metric(campaign_id, "success_rate", success_rate)
        self._record_metric(campaign_id, "total_trials", total_trials)
        self._record_metric(campaign_id, "completed_trials", completed_trials)
        self._record_metric(campaign_id, "failed_trials", failed_trials)
        self._record_metric(campaign_id, "running_trials", running_trials)
        
        # Performance metrics
        if completed_trials > 0:
            completed_trial_objects = [t for t in campaign.trials if t.status == TrialStatus.COMPLETED]
            
            # Trial duration statistics
            durations = [t.duration_seconds for t in completed_trial_objects if t.duration_seconds]
            if durations:
                avg_duration = np.mean(durations)
                max_duration = max(durations)
                self._record_metric(campaign_id, "avg_trial_duration_seconds", avg_duration)
                self._record_metric(campaign_id, "max_trial_duration_seconds", max_duration)
                
                # Check for slow trials
                if max_duration > self.alert_thresholds["max_trial_duration_minutes"] * 60:
                    self._create_alert(
                        severity="medium",
                        category="performance", 
                        campaign_id=campaign_id,
                        message=f"Trial duration exceeded threshold: {max_duration/60:.1f} minutes",
                        details={"max_duration_seconds": max_duration, "threshold_minutes": self.alert_thresholds["max_trial_duration_minutes"]}
                    )
            
            # Objective value statistics
            objectives = [t.objective_value for t in completed_trial_objects if t.objective_value is not None]
            if objectives:
                best_objective = max(objectives)
                mean_objective = np.mean(objectives)
                self._record_metric(campaign_id, "best_objective", best_objective)
                self._record_metric(campaign_id, "mean_objective", mean_objective)
        
        # Resource utilization (mock implementation)
        resource_util = self._estimate_resource_utilization(campaign)
        self._record_metric(campaign_id, "estimated_resource_utilization", resource_util)
        
        # Check for issues
        self._check_campaign_health(campaign)
    
    def _estimate_resource_utilization(self, campaign: Campaign) -> float:
        """Estimate resource utilization for campaign"""
        # Mock implementation - in practice would integrate with system monitoring
        running_trials = len([t for t in campaign.trials if t.status == TrialStatus.RUNNING])
        base_utilization = min(running_trials * 0.25, 1.0)  # 25% per running trial
        
        # Add some variation based on campaign complexity
        complexity_factor = len(campaign.spec.knob_spaces) / 10.0
        return min(base_utilization + complexity_factor, 1.0)
    
    def _check_campaign_health(self, campaign: Campaign):
        """Check campaign health and create alerts"""
        campaign_id = campaign.campaign_id
        
        # Check success rate
        completed_trials = len([t for t in campaign.trials if t.status == TrialStatus.COMPLETED])
        failed_trials = len([t for t in campaign.trials if t.status == TrialStatus.FAILED])
        
        if completed_trials + failed_trials >= 3:  # Need minimum trials to assess
            success_rate = completed_trials / (completed_trials + failed_trials)
            
            if success_rate < self.alert_thresholds["min_success_rate"]:
                self._create_alert(
                    severity="high",
                    category="failure",
                    campaign_id=campaign_id,
                    message=f"Campaign success rate below threshold: {success_rate:.1%}",
                    details={"success_rate": success_rate, "threshold": self.alert_thresholds["min_success_rate"]}
                )
        
        # Check for consecutive failures
        recent_trials = campaign.trials[-5:]  # Last 5 trials
        consecutive_failures = 0
        for trial in reversed(recent_trials):
            if trial.status == TrialStatus.FAILED:
                consecutive_failures += 1
            else:
                break
        
        if consecutive_failures >= self.alert_thresholds["max_consecutive_failures"]:
            self._create_alert(
                severity="critical",
                category="failure",
                campaign_id=campaign_id,
                message=f"Campaign has {consecutive_failures} consecutive failures",
                details={"consecutive_failures": consecutive_failures}
            )
        
        # Check progress rate (if campaign has been running long enough)
        if campaign.start_time:
            runtime_hours = (datetime.now() - campaign.start_time).total_seconds() / 3600
            if runtime_hours > 1.0:  # At least 1 hour of runtime
                progress_rate = completed_trials / runtime_hours
                if progress_rate < self.alert_thresholds["min_progress_rate"]:
                    self._create_alert(
                        severity="medium",
                        category="performance",
                        campaign_id=campaign_id,
                        message=f"Campaign progress rate below threshold: {progress_rate:.2f} trials/hour",
                        details={"progress_rate": progress_rate, "threshold": self.alert_thresholds["min_progress_rate"]}
                    )
    
    def _check_system_health(self):
        """Check overall system health"""
        # This would integrate with system monitoring in practice
        pass
    
    def _record_metric(self, 
                      campaign_id: str, 
                      metric_name: str, 
                      metric_value: float,
                      trial_id: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None):
        """Record metric measurement"""
        
        snapshot = MetricSnapshot(
            timestamp=datetime.now(),
            campaign_id=campaign_id,
            metric_name=metric_name,
            metric_value=metric_value,
            trial_id=trial_id,
            metadata=metadata or {}
        )
        
        # Store in memory
        key = f"{campaign_id}:{metric_name}"
        self.metric_history[key].append(snapshot)
        self.current_snapshots[campaign_id][metric_name] = snapshot
        
        # Store in database
        self._save_metric_to_db(snapshot)
    
    def _save_metric_to_db(self, snapshot: MetricSnapshot):
        """Save metric to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO metrics (timestamp, campaign_id, metric_name, metric_value, trial_id, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        snapshot.timestamp.isoformat(),
                        snapshot.campaign_id,
                        snapshot.metric_name,
                        snapshot.metric_value,
                        snapshot.trial_id,
                        json.dumps(snapshot.metadata)
                    )
                )
        except Exception as e:
            logger.error(f"Failed to save metric to database: {str(e)}")
    
    def _create_alert(self, 
                     severity: str,
                     category: str,
                     message: str,
                     details: Dict[str, Any],
                     campaign_id: Optional[str] = None):
        """Create new alert"""
        
        alert_id = f"{category}_{campaign_id or 'system'}_{int(time.time())}"
        
        # Check if similar alert already exists (avoid spam)
        existing_alerts = [
            a for a in self.alerts.values()
            if a.category == category and a.campaign_id == campaign_id 
            and not a.resolved and (datetime.now() - a.timestamp).seconds < 3600  # Within last hour
        ]
        
        if existing_alerts:
            return  # Don't create duplicate alert
        
        alert = Alert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            severity=severity,
            category=category,
            campaign_id=campaign_id,
            message=message,
            details=details
        )
        
        self.alerts[alert_id] = alert
        
        # Save to database
        self._save_alert_to_db(alert)
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {str(e)}")
        
        logger.warning(f"ALERT [{severity.upper()}] {category}: {message}")
    
    def _save_alert_to_db(self, alert: Alert):
        """Save alert to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO alerts (alert_id, timestamp, severity, category, campaign_id, message, details, acknowledged, resolved, resolution_notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        alert.alert_id,
                        alert.timestamp.isoformat(),
                        alert.severity,
                        alert.category,
                        alert.campaign_id,
                        alert.message,
                        json.dumps(alert.details),
                        alert.acknowledged,
                        alert.resolved,
                        alert.resolution_notes
                    )
                )
        except Exception as e:
            logger.error(f"Failed to save alert to database: {str(e)}")
    
    def get_campaign_health(self, campaign_id: str) -> CampaignHealthStatus:
        """Get current health status for campaign"""
        campaign = self.campaign_manager.campaigns.get(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")
        
        # Get current metrics
        current_metrics = self.current_snapshots.get(campaign_id, {})
        
        progress_percentage = current_metrics.get("progress_percentage", MetricSnapshot(datetime.now(), campaign_id, "progress_percentage", 0.0)).metric_value
        success_rate = current_metrics.get("success_rate", MetricSnapshot(datetime.now(), campaign_id, "success_rate", 1.0)).metric_value
        avg_trial_duration = current_metrics.get("avg_trial_duration_seconds", MetricSnapshot(datetime.now(), campaign_id, "avg_trial_duration_seconds", 0.0)).metric_value
        resource_utilization = current_metrics.get("estimated_resource_utilization", MetricSnapshot(datetime.now(), campaign_id, "estimated_resource_utilization", 0.0)).metric_value
        
        # Count alerts
        campaign_alerts = [a for a in self.alerts.values() if a.campaign_id == campaign_id and not a.resolved]
        active_alerts = len(campaign_alerts)
        critical_alerts = len([a for a in campaign_alerts if a.severity == "critical"])
        
        # Determine component health
        execution_health = "healthy"
        if campaign.status == CampaignStatus.FAILED:
            execution_health = "failed"
        elif critical_alerts > 0:
            execution_health = "critical"
        elif success_rate < 0.5:
            execution_health = "warning"
        
        resource_health = "healthy"
        if resource_utilization > 0.9:
            resource_health = "critical"
        elif resource_utilization > 0.7:
            resource_health = "warning"
        
        quality_health = "healthy"
        if campaign.best_objective is not None:
            if campaign.best_objective < 0.001:  # Very low improvement
                quality_health = "warning"
        
        validation_health = "healthy"  # Would be updated by validation pipeline
        
        # Overall health
        component_healths = [execution_health, resource_health, quality_health, validation_health]
        if "failed" in component_healths:
            overall_health = "failed"
        elif "critical" in component_healths:
            overall_health = "critical"
        elif "warning" in component_healths:
            overall_health = "warning"
        else:
            overall_health = "healthy"
        
        return CampaignHealthStatus(
            campaign_id=campaign_id,
            overall_health=overall_health,
            execution_health=execution_health,
            resource_health=resource_health,
            quality_health=quality_health,
            validation_health=validation_health,
            progress_percentage=progress_percentage,
            success_rate=success_rate,
            average_trial_duration=avg_trial_duration,
            resource_utilization=resource_utilization,
            active_alerts=active_alerts,
            critical_alerts=critical_alerts,
            recent_failures=len([t for t in campaign.trials[-5:] if t.status == TrialStatus.FAILED]),
            last_updated=datetime.now()
        )
    
    def get_metrics_history(self, 
                           campaign_id: str, 
                           metric_name: Optional[str] = None,
                           hours_back: int = 24) -> pd.DataFrame:
        """Get historical metrics for campaign"""
        query = """
            SELECT timestamp, campaign_id, metric_name, metric_value, trial_id, metadata
            FROM metrics 
            WHERE campaign_id = ? AND timestamp > ?
        """
        params = [campaign_id, (datetime.now() - timedelta(hours=hours_back)).isoformat()]
        
        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)
        
        query += " ORDER BY timestamp"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def acknowledge_alert(self, alert_id: str, notes: Optional[str] = None):
        """Acknowledge alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            if notes:
                self.alerts[alert_id].resolution_notes = notes
            
            self._save_alert_to_db(self.alerts[alert_id])
            logger.info(f"Acknowledged alert {alert_id}")
    
    def resolve_alert(self, alert_id: str, resolution_notes: str):
        """Resolve alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolution_notes = resolution_notes
            
            self._save_alert_to_db(self.alerts[alert_id])
            logger.info(f"Resolved alert {alert_id}: {resolution_notes}")

class CampaignReporter:
    """Generates comprehensive campaign reports and analytics"""
    
    def __init__(self, monitor: CampaignMonitor):
        self.monitor = monitor
    
    def generate_campaign_summary_report(self, campaign_id: str) -> Dict[str, Any]:
        """Generate comprehensive campaign summary report"""
        campaign = self.monitor.campaign_manager.campaigns.get(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")
        
        health_status = self.monitor.get_campaign_health(campaign_id)
        
        # Get metrics history
        metrics_df = self.monitor.get_metrics_history(campaign_id, hours_back=168)  # 1 week
        
        # Basic campaign info
        report = {
            "campaign_info": {
                "campaign_id": campaign_id,
                "name": campaign.spec.name,
                "slice_name": campaign.spec.slice_candidate.slice_name,
                "budget_tier": campaign.spec.budget_tier,
                "status": campaign.status.value,
                "created": campaign.created_time.isoformat(),
                "started": campaign.start_time.isoformat() if campaign.start_time else None,
                "ended": campaign.end_time.isoformat() if campaign.end_time else None
            },
            "health_status": {
                "overall_health": health_status.overall_health,
                "progress_percentage": health_status.progress_percentage,
                "success_rate": health_status.success_rate,
                "active_alerts": health_status.active_alerts,
                "critical_alerts": health_status.critical_alerts
            },
            "execution_summary": self._create_execution_summary(campaign),
            "performance_analysis": self._create_performance_analysis(campaign, metrics_df),
            "optimization_analysis": self._create_optimization_analysis(campaign),
            "alerts_summary": self._create_alerts_summary(campaign_id),
            "recommendations": self._generate_recommendations(campaign, health_status)
        }
        
        return report
    
    def generate_multi_campaign_dashboard(self, campaign_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate dashboard view of multiple campaigns"""
        if campaign_ids is None:
            campaign_ids = list(self.monitor.campaign_manager.campaigns.keys())
        
        dashboard = {
            "generated_at": datetime.now().isoformat(),
            "total_campaigns": len(campaign_ids),
            "campaigns": [],
            "system_summary": {
                "healthy_campaigns": 0,
                "warning_campaigns": 0,
                "critical_campaigns": 0,
                "failed_campaigns": 0
            },
            "active_alerts": 0,
            "resource_utilization": 0.0
        }
        
        total_resource_util = 0.0
        
        for campaign_id in campaign_ids:
            try:
                health = self.monitor.get_campaign_health(campaign_id)
                campaign = self.monitor.campaign_manager.campaigns[campaign_id]
                
                campaign_card = {
                    "campaign_id": campaign_id,
                    "name": campaign.spec.name,
                    "slice_name": campaign.spec.slice_candidate.slice_name,
                    "status": campaign.status.value,
                    "health": health.overall_health,
                    "progress": health.progress_percentage,
                    "success_rate": health.success_rate,
                    "active_alerts": health.active_alerts,
                    "best_improvement": campaign.best_objective
                }
                
                dashboard["campaigns"].append(campaign_card)
                
                # Update system summary
                if health.overall_health == "healthy":
                    dashboard["system_summary"]["healthy_campaigns"] += 1
                elif health.overall_health == "warning":
                    dashboard["system_summary"]["warning_campaigns"] += 1
                elif health.overall_health == "critical":
                    dashboard["system_summary"]["critical_campaigns"] += 1
                else:
                    dashboard["system_summary"]["failed_campaigns"] += 1
                
                dashboard["active_alerts"] += health.active_alerts
                total_resource_util += health.resource_utilization
                
            except Exception as e:
                logger.error(f"Error processing campaign {campaign_id} for dashboard: {str(e)}")
        
        dashboard["resource_utilization"] = total_resource_util / len(campaign_ids) if campaign_ids else 0.0
        
        return dashboard
    
    def _create_execution_summary(self, campaign: Campaign) -> Dict[str, Any]:
        """Create execution summary section"""
        trials_by_status = defaultdict(int)
        for trial in campaign.trials:
            trials_by_status[trial.status.value] += 1
        
        runtime_info = {}
        if campaign.start_time:
            if campaign.end_time:
                runtime_info["total_runtime_hours"] = (campaign.end_time - campaign.start_time).total_seconds() / 3600
            else:
                runtime_info["current_runtime_hours"] = (datetime.now() - campaign.start_time).total_seconds() / 3600
        
        return {
            "total_trials": len(campaign.trials),
            "target_trials": campaign.spec.n_trials,
            "trials_by_status": dict(trials_by_status),
            "runtime_info": runtime_info,
            "best_objective": campaign.best_objective,
            "optimization_method": "Bayesian Optimization" if len(campaign.trials) > 5 else "Grid Search"
        }
    
    def _create_performance_analysis(self, campaign: Campaign, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Create performance analysis section"""
        completed_trials = [t for t in campaign.trials if t.status == TrialStatus.COMPLETED]
        
        analysis = {
            "trial_performance": {},
            "resource_usage": {},
            "optimization_efficiency": {}
        }
        
        if completed_trials:
            durations = [t.duration_seconds for t in completed_trials if t.duration_seconds]
            if durations:
                analysis["trial_performance"] = {
                    "avg_duration_minutes": np.mean(durations) / 60,
                    "max_duration_minutes": max(durations) / 60,
                    "min_duration_minutes": min(durations) / 60,
                    "duration_std_minutes": np.std(durations) / 60
                }
        
        # Analyze metrics trends
        if not metrics_df.empty:
            for metric_name in ["success_rate", "estimated_resource_utilization"]:
                metric_data = metrics_df[metrics_df['metric_name'] == metric_name]
                if not metric_data.empty:
                    values = metric_data['metric_value'].values
                    if len(values) > 1:
                        trend = "increasing" if values[-1] > values[0] else "decreasing"
                        analysis["resource_usage"][f"{metric_name}_trend"] = trend
        
        return analysis
    
    def _create_optimization_analysis(self, campaign: Campaign) -> Dict[str, Any]:
        """Create optimization-specific analysis"""
        completed_trials = [t for t in campaign.trials if t.status == TrialStatus.COMPLETED and t.objective_value is not None]
        
        if not completed_trials:
            return {"message": "No completed trials with objective values"}
        
        objectives = [t.objective_value for t in completed_trials]
        
        # Convergence analysis
        cumulative_best = []
        current_best = objectives[0]
        for obj in objectives:
            current_best = max(current_best, obj)
            cumulative_best.append(current_best)
        
        # Parameter exploration analysis
        param_analysis = {}
        if len(completed_trials) > 1:
            # Analyze parameter ranges explored
            all_params = defaultdict(list)
            for trial in completed_trials:
                for param_name, param_value in trial.parameters.items():
                    if isinstance(param_value, (int, float)):
                        all_params[param_name].append(param_value)
            
            for param_name, values in all_params.items():
                if len(values) > 1:
                    param_analysis[param_name] = {
                        "range_explored": [min(values), max(values)],
                        "exploration_ratio": (max(values) - min(values)) / (max(values) + min(values)) if max(values) > 0 else 0
                    }
        
        return {
            "convergence": {
                "improvement_curve": cumulative_best,
                "final_improvement": cumulative_best[-1] if cumulative_best else 0.0,
                "convergence_rate": (cumulative_best[-1] - cumulative_best[0]) / len(cumulative_best) if len(cumulative_best) > 1 else 0.0
            },
            "exploration": param_analysis,
            "best_configuration": campaign.best_trial.parameters if campaign.best_trial else None
        }
    
    def _create_alerts_summary(self, campaign_id: str) -> Dict[str, Any]:
        """Create alerts summary section"""
        campaign_alerts = [a for a in self.monitor.alerts.values() if a.campaign_id == campaign_id]
        
        alerts_by_severity = defaultdict(int)
        alerts_by_category = defaultdict(int)
        
        for alert in campaign_alerts:
            alerts_by_severity[alert.severity] += 1
            alerts_by_category[alert.category] += 1
        
        recent_alerts = sorted(
            [a for a in campaign_alerts if not a.resolved],
            key=lambda x: x.timestamp,
            reverse=True
        )[:5]
        
        return {
            "total_alerts": len(campaign_alerts),
            "active_alerts": len([a for a in campaign_alerts if not a.resolved]),
            "alerts_by_severity": dict(alerts_by_severity),
            "alerts_by_category": dict(alerts_by_category),
            "recent_alerts": [
                {
                    "alert_id": a.alert_id,
                    "severity": a.severity,
                    "category": a.category,
                    "message": a.message,
                    "timestamp": a.timestamp.isoformat()
                }
                for a in recent_alerts
            ]
        }
    
    def _generate_recommendations(self, campaign: Campaign, health_status: CampaignHealthStatus) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Performance recommendations
        if health_status.success_rate < 0.5:
            recommendations.append("Consider adjusting campaign parameters - low success rate may indicate poor knob ranges")
        
        if health_status.progress_percentage < 50 and health_status.average_trial_duration > 300:  # 5 minutes
            recommendations.append("Trials are running slowly - consider reducing evaluation complexity or adding more resources")
        
        if health_status.resource_utilization > 0.8:
            recommendations.append("High resource utilization detected - consider reducing concurrent trials or upgrading resources")
        
        # Quality recommendations
        if campaign.best_objective is not None and campaign.best_objective < 0.01:
            recommendations.append("Low objective improvements observed - review knob sensitivity and parameter ranges")
        
        # Alert-based recommendations
        if health_status.critical_alerts > 0:
            recommendations.append("Critical alerts require immediate attention - review alert details and take corrective action")
        
        if health_status.recent_failures > 2:
            recommendations.append("Multiple recent failures detected - investigate trial execution issues")
        
        # Campaign-specific recommendations
        if len(campaign.trials) > campaign.spec.n_trials * 0.8 and not campaign.best_trial:
            recommendations.append("Campaign nearing completion without successful trials - consider extending or adjusting gates")
        
        return recommendations

if __name__ == "__main__":
    # Test monitoring system
    import logging
    import tempfile
    import os
    
    logging.basicConfig(level=logging.INFO)
    
    from campaign_manager import CampaignManager
    from specific_campaigns import create_demo_slice_candidates, CampaignFactory
    
    # Create temporary database
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    
    try:
        # Initialize components
        monitor = CampaignMonitor(db_path=temp_db.name, polling_interval=1.0)
        campaign_manager = CampaignManager(output_dir="./test_campaigns")
        reporter = CampaignReporter(monitor)
        
        # Create test campaign
        candidates = create_demo_slice_candidates()
        spec = CampaignFactory.create_campaign_spec("zh_qa_8", candidates["zh_qa_8"])
        campaign = campaign_manager.create_campaign(spec)
        
        # Start monitoring
        monitor.start_monitoring(campaign_manager)
        
        print("=== Starting test campaign ===")
        campaign_manager.start_campaign(campaign.campaign_id)
        
        # Monitor for a few seconds
        time.sleep(5.0)
        
        # Get health status
        health = monitor.get_campaign_health(campaign.campaign_id)
        print(f"\n=== Campaign Health Status ===")
        print(f"Overall health: {health.overall_health}")
        print(f"Progress: {health.progress_percentage:.1f}%")
        print(f"Success rate: {health.success_rate:.1%}")
        print(f"Active alerts: {health.active_alerts}")
        
        # Generate report
        report = reporter.generate_campaign_summary_report(campaign.campaign_id)
        print(f"\n=== Campaign Report Summary ===")
        print(f"Campaign: {report['campaign_info']['name']}")
        print(f"Status: {report['campaign_info']['status']}")
        print(f"Trials: {report['execution_summary']['total_trials']}/{report['execution_summary']['target_trials']}")
        print(f"Best objective: {report['execution_summary']['best_objective']}")
        
        # Generate dashboard
        dashboard = reporter.generate_multi_campaign_dashboard()
        print(f"\n=== Dashboard Summary ===")
        print(f"Total campaigns: {dashboard['total_campaigns']}")
        print(f"System health: {dashboard['system_summary']}")
        print(f"Active alerts: {dashboard['active_alerts']}")
        
        # Stop monitoring
        monitor.stop_monitoring()
        campaign_manager.shutdown()
        
        print("\n=== Test completed ===")
        
    finally:
        # Clean up
        os.unlink(temp_db.name)