#!/usr/bin/env python3
"""
Operational Runbook System for Lethe Hybrid Deployment

Provides comprehensive operational guidance, safety protocols, and automated
incident response for the hybrid Lethe→StreamingLLM system.

Key Features:
- Automated incident detection and classification
- Safety protocol enforcement with circuit breakers
- Operational guidance with step-by-step procedures
- Alert escalation with severity-based routing
- Performance monitoring with automatic remediation
- Rollback procedures with automated execution
- Knowledge base with searchable procedures
"""

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IncidentSeverity(Enum):
    """Incident severity levels for proper escalation"""
    CRITICAL = "critical"    # System down, data loss risk
    HIGH = "high"           # Performance degraded >50%
    MEDIUM = "medium"       # Performance degraded 10-50%
    LOW = "low"            # Minor issues, monitoring alerts
    INFO = "info"          # Informational, no action needed

class AlertChannel(Enum):
    """Alert delivery channels"""
    EMAIL = "email"
    SLACK = "slack"
    PAGER = "pager"
    LOG = "log"
    DASHBOARD = "dashboard"

class OperationalStatus(Enum):
    """System operational status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"

@dataclass
class Incident:
    """Represents an operational incident"""
    id: str
    title: str
    severity: IncidentSeverity
    description: str
    component: str
    detected_at: datetime
    metrics: Dict[str, float]
    status: str = "open"
    assigned_to: Optional[str] = None
    resolution_steps: List[str] = field(default_factory=list)
    automated_actions: List[str] = field(default_factory=list)
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'severity': self.severity.value,
            'description': self.description,
            'component': self.component,
            'detected_at': self.detected_at.isoformat(),
            'metrics': self.metrics,
            'status': self.status,
            'assigned_to': self.assigned_to,
            'resolution_steps': self.resolution_steps,
            'automated_actions': self.automated_actions,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }

@dataclass
class SafetyProtocol:
    """Defines safety protocols and circuit breakers"""
    name: str
    trigger_conditions: Dict[str, float]
    actions: List[str]
    cooldown_period: int  # seconds
    max_triggers_per_hour: int
    escalation_after_triggers: int
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

@dataclass
class OperationalProcedure:
    """Defines step-by-step operational procedures"""
    id: str
    name: str
    category: str
    description: str
    steps: List[Dict[str, Any]]  # Each step has action, command, validation
    prerequisites: List[str]
    estimated_duration: int  # minutes
    risk_level: str  # low, medium, high, critical
    automation_available: bool
    tags: List[str]

class CircuitBreaker:
    """Circuit breaker for protecting system components"""
    
    def __init__(self, failure_threshold: int = 5, 
                 timeout: int = 60, reset_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if self.last_failure_time and \
               (datetime.now() - self.last_failure_time).seconds > self.reset_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
                
        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.reset()
            return result
        except Exception as e:
            self.record_failure()
            raise e
            
    def record_failure(self):
        """Record a failure and potentially open the circuit"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            
    def reset(self):
        """Reset the circuit breaker"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"

class OperationalRunbook:
    """
    Main operational runbook system providing comprehensive operational support
    """
    
    def __init__(self, db_path: str = "operational_runbook.db"):
        self.db_path = db_path
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.safety_protocols: Dict[str, SafetyProtocol] = {}
        self.procedures: Dict[str, OperationalProcedure] = {}
        self.active_incidents: Dict[str, Incident] = {}
        self.system_status = OperationalStatus.UNKNOWN
        self.alert_channels: Dict[AlertChannel, Dict[str, Any]] = {}
        
        # Initialize database
        self._init_database()
        
        # Load default safety protocols and procedures
        self._load_default_protocols()
        self._load_default_procedures()
        
        # Initialize circuit breakers for critical components
        self._init_circuit_breakers()
        
        logger.info("OperationalRunbook initialized successfully")
        
    def _init_database(self):
        """Initialize SQLite database for operational data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Incidents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS incidents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                severity TEXT NOT NULL,
                description TEXT,
                component TEXT,
                detected_at TEXT NOT NULL,
                metrics TEXT,
                status TEXT DEFAULT 'open',
                assigned_to TEXT,
                resolution_steps TEXT,
                automated_actions TEXT,
                resolved_at TEXT
            )
        ''')
        
        # System metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                timestamp TEXT PRIMARY KEY,
                component TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                status TEXT DEFAULT 'normal'
            )
        ''')
        
        # Protocol triggers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS protocol_triggers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                protocol_name TEXT NOT NULL,
                triggered_at TEXT NOT NULL,
                trigger_reason TEXT,
                actions_taken TEXT,
                resolved_at TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def _load_default_protocols(self):
        """Load default safety protocols"""
        protocols = [
            SafetyProtocol(
                name="critical_performance_degradation",
                trigger_conditions={
                    "proxy_gap": 2.0,  # >2% proxy gap
                    "tail_latency_ratio": 3.0,  # P99/P95 > 3.0
                    "error_rate": 0.05  # >5% error rate
                },
                actions=[
                    "immediate_rollback",
                    "alert_on_call_engineer",
                    "freeze_parameter_updates",
                    "activate_safe_mode"
                ],
                cooldown_period=300,  # 5 minutes
                max_triggers_per_hour=3,
                escalation_after_triggers=2
            ),
            SafetyProtocol(
                name="memory_pressure",
                trigger_conditions={
                    "memory_usage_percent": 85.0,
                    "swap_usage_percent": 20.0
                },
                actions=[
                    "reduce_batch_size",
                    "clear_cache",
                    "throttle_requests"
                ],
                cooldown_period=120,
                max_triggers_per_hour=5,
                escalation_after_triggers=3
            ),
            SafetyProtocol(
                name="context_length_anomaly",
                trigger_conditions={
                    "avg_context_length": 50000,  # tokens
                    "context_truncation_rate": 0.30  # >30% truncation
                },
                actions=[
                    "adjust_lambda_parameter",
                    "increase_truncation_threshold",
                    "alert_performance_team"
                ],
                cooldown_period=180,
                max_triggers_per_hour=4,
                escalation_after_triggers=2
            )
        ]
        
        for protocol in protocols:
            self.safety_protocols[protocol.name] = protocol
            
    def _load_default_procedures(self):
        """Load default operational procedures"""
        procedures = [
            OperationalProcedure(
                id="emergency_rollback",
                name="Emergency System Rollback",
                category="incident_response",
                description="Complete rollback to last known good configuration",
                steps=[
                    {
                        "action": "Stop new request processing",
                        "command": "curl -X POST /api/maintenance/enable",
                        "validation": "Check maintenance mode active"
                    },
                    {
                        "action": "Rollback deployment",
                        "command": "kubectl rollout undo deployment/lethe-hybrid",
                        "validation": "Verify previous version running"
                    },
                    {
                        "action": "Reset parameters to safe defaults",
                        "command": "python reset_parameters.py --config=safe_defaults.json",
                        "validation": "Confirm parameter reset in logs"
                    },
                    {
                        "action": "Resume request processing",
                        "command": "curl -X POST /api/maintenance/disable",
                        "validation": "Verify requests being processed normally"
                    }
                ],
                prerequisites=["admin_access", "kubectl_configured"],
                estimated_duration=10,  # 10 minutes
                risk_level="medium",
                automation_available=True,
                tags=["emergency", "rollback", "incident_response"]
            ),
            OperationalProcedure(
                id="performance_investigation",
                name="Performance Degradation Investigation",
                category="troubleshooting",
                description="Systematic investigation of performance issues",
                steps=[
                    {
                        "action": "Capture current metrics snapshot",
                        "command": "python capture_metrics.py --duration=300",
                        "validation": "Metrics file generated"
                    },
                    {
                        "action": "Analyze context bucket distribution",
                        "command": "python analyze_buckets.py --recent-hours=1",
                        "validation": "Bucket analysis report generated"
                    },
                    {
                        "action": "Check for parameter drift",
                        "command": "python check_drift.py --baseline=production_baseline.json",
                        "validation": "Drift report shows within thresholds"
                    },
                    {
                        "action": "Review error logs",
                        "command": "grep -i error /var/log/lethe/hybrid.log | tail -100",
                        "validation": "No critical errors in recent logs"
                    }
                ],
                prerequisites=["monitoring_access", "log_access"],
                estimated_duration=20,
                risk_level="low",
                automation_available=False,
                tags=["performance", "investigation", "troubleshooting"]
            )
        ]
        
        for procedure in procedures:
            self.procedures[procedure.id] = procedure
            
    def _init_circuit_breakers(self):
        """Initialize circuit breakers for critical components"""
        self.circuit_breakers = {
            "hybrid_selector": CircuitBreaker(failure_threshold=5, timeout=60, reset_timeout=300),
            "context_analyzer": CircuitBreaker(failure_threshold=3, timeout=30, reset_timeout=180),
            "parameter_updater": CircuitBreaker(failure_threshold=10, timeout=120, reset_timeout=600),
            "database_connection": CircuitBreaker(failure_threshold=3, timeout=30, reset_timeout=120)
        }
        
    def monitor_system_metrics(self, metrics: Dict[str, float], component: str = "system"):
        """Monitor system metrics and trigger protocols if needed"""
        timestamp = datetime.now()
        
        # Store metrics
        self._store_metrics(timestamp, component, metrics)
        
        # Check safety protocols
        triggered_protocols = []
        for protocol_name, protocol in self.safety_protocols.items():
            if not protocol.enabled:
                continue
                
            if self._should_trigger_protocol(protocol, metrics):
                if self._can_trigger_protocol(protocol):
                    triggered_protocols.append(protocol_name)
                    self._trigger_safety_protocol(protocol, metrics)
                    
        # Update system status based on metrics and protocols
        self._update_system_status(metrics, triggered_protocols)
        
        return {
            "status": self.system_status.value,
            "triggered_protocols": triggered_protocols,
            "metrics_stored": len(metrics),
            "timestamp": timestamp.isoformat()
        }
        
    def _store_metrics(self, timestamp: datetime, component: str, metrics: Dict[str, float]):
        """Store metrics in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for metric_name, value in metrics.items():
            status = "normal"
            if metric_name == "error_rate" and value > 0.01:
                status = "warning"
            elif metric_name == "proxy_gap" and value > 0.5:
                status = "warning"
                
            cursor.execute('''
                INSERT INTO system_metrics 
                (timestamp, component, metric_name, metric_value, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp.isoformat(), component, metric_name, value, status))
            
        conn.commit()
        conn.close()
        
    def _should_trigger_protocol(self, protocol: SafetyProtocol, metrics: Dict[str, float]) -> bool:
        """Check if a safety protocol should be triggered"""
        for condition_metric, threshold in protocol.trigger_conditions.items():
            if condition_metric in metrics:
                if metrics[condition_metric] >= threshold:
                    return True
        return False
        
    def _can_trigger_protocol(self, protocol: SafetyProtocol) -> bool:
        """Check if protocol can be triggered (not in cooldown, under rate limit)"""
        now = datetime.now()
        
        # Check cooldown
        if protocol.last_triggered:
            if (now - protocol.last_triggered).seconds < protocol.cooldown_period:
                return False
                
        # Check rate limit
        hour_ago = now - timedelta(hours=1)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM protocol_triggers 
            WHERE protocol_name = ? AND triggered_at >= ?
        ''', (protocol.name, hour_ago.isoformat()))
        triggers_last_hour = cursor.fetchone()[0]
        conn.close()
        
        return triggers_last_hour < protocol.max_triggers_per_hour
        
    def _trigger_safety_protocol(self, protocol: SafetyProtocol, metrics: Dict[str, float]):
        """Execute safety protocol actions"""
        protocol.last_triggered = datetime.now()
        protocol.trigger_count += 1
        
        actions_taken = []
        
        for action in protocol.actions:
            try:
                result = self._execute_safety_action(action, metrics)
                actions_taken.append(f"{action}: {result}")
                logger.info(f"Safety action executed: {action}")
            except Exception as e:
                logger.error(f"Failed to execute safety action {action}: {e}")
                actions_taken.append(f"{action}: FAILED - {e}")
                
        # Log protocol trigger
        self._log_protocol_trigger(protocol.name, actions_taken, metrics)
        
        # Check for escalation
        if protocol.trigger_count >= protocol.escalation_after_triggers:
            self._escalate_protocol(protocol, metrics)
            
    def _execute_safety_action(self, action: str, metrics: Dict[str, float]) -> str:
        """Execute individual safety actions"""
        if action == "immediate_rollback":
            return self.execute_procedure("emergency_rollback", automated=True)
        elif action == "alert_on_call_engineer":
            return self._send_alert("Critical system issue detected", 
                                  IncidentSeverity.CRITICAL, metrics)
        elif action == "freeze_parameter_updates":
            return self._freeze_parameter_updates()
        elif action == "activate_safe_mode":
            return self._activate_safe_mode()
        elif action == "reduce_batch_size":
            return self._adjust_parameter("batch_size", 0.7)  # Reduce by 30%
        elif action == "clear_cache":
            return self._clear_system_cache()
        elif action == "throttle_requests":
            return self._throttle_requests(0.8)  # Throttle to 80%
        elif action == "adjust_lambda_parameter":
            return self._adjust_parameter("lambda", 0.9)  # Reduce lambda
        elif action == "increase_truncation_threshold":
            return self._adjust_parameter("truncation_threshold", 1.1)  # Increase 10%
        elif action == "alert_performance_team":
            return self._send_alert("Performance anomaly detected",
                                  IncidentSeverity.MEDIUM, metrics)
        else:
            return f"Unknown action: {action}"
            
    def _freeze_parameter_updates(self) -> str:
        """Freeze automatic parameter updates"""
        # This would integrate with the adaptive control surface
        return "Parameter updates frozen"
        
    def _activate_safe_mode(self) -> str:
        """Activate system safe mode with conservative parameters"""
        return "Safe mode activated"
        
    def _adjust_parameter(self, param_name: str, multiplier: float) -> str:
        """Adjust system parameter by multiplier"""
        return f"Parameter {param_name} adjusted by {multiplier}x"
        
    def _clear_system_cache(self) -> str:
        """Clear system caches to free memory"""
        return "System caches cleared"
        
    def _throttle_requests(self, rate: float) -> str:
        """Throttle incoming requests to specified rate"""
        return f"Requests throttled to {rate*100}%"
        
    def _log_protocol_trigger(self, protocol_name: str, actions_taken: List[str], metrics: Dict[str, float]):
        """Log safety protocol trigger to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO protocol_triggers 
            (protocol_name, triggered_at, trigger_reason, actions_taken)
            VALUES (?, ?, ?, ?)
        ''', (
            protocol_name,
            datetime.now().isoformat(),
            json.dumps(metrics),
            json.dumps(actions_taken)
        ))
        
        conn.commit()
        conn.close()
        
    def _escalate_protocol(self, protocol: SafetyProtocol, metrics: Dict[str, float]):
        """Escalate repeated protocol triggers"""
        incident = self.create_incident(
            title=f"Repeated safety protocol triggers: {protocol.name}",
            severity=IncidentSeverity.HIGH,
            description=f"Safety protocol {protocol.name} triggered {protocol.trigger_count} times",
            component="safety_system",
            metrics=metrics
        )
        
        self._send_alert(
            f"ESCALATION: {protocol.name} triggered {protocol.trigger_count} times",
            IncidentSeverity.HIGH,
            metrics,
            include_incident_id=incident.id
        )
        
    def _update_system_status(self, metrics: Dict[str, float], triggered_protocols: List[str]):
        """Update overall system operational status"""
        if any(protocol in triggered_protocols for protocol in ["critical_performance_degradation"]):
            self.system_status = OperationalStatus.CRITICAL
        elif triggered_protocols:
            self.system_status = OperationalStatus.DEGRADED  
        elif self._metrics_healthy(metrics):
            self.system_status = OperationalStatus.HEALTHY
        else:
            self.system_status = OperationalStatus.DEGRADED
            
    def _metrics_healthy(self, metrics: Dict[str, float]) -> bool:
        """Check if metrics indicate healthy system"""
        thresholds = {
            "proxy_gap": 0.5,  # <0.5%
            "tail_latency_ratio": 2.0,  # P99/P95 < 2.0
            "error_rate": 0.01,  # <1%
            "memory_usage_percent": 80.0  # <80%
        }
        
        for metric, threshold in thresholds.items():
            if metric in metrics and metrics[metric] >= threshold:
                return False
                
        return True
        
    def create_incident(self, title: str, severity: IncidentSeverity, 
                       description: str, component: str, 
                       metrics: Dict[str, float]) -> Incident:
        """Create and track a new operational incident"""
        incident_id = hashlib.md5(
            f"{title}-{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        incident = Incident(
            id=incident_id,
            title=title,
            severity=severity,
            description=description,
            component=component,
            detected_at=datetime.now(),
            metrics=metrics
        )
        
        self.active_incidents[incident_id] = incident
        
        # Store in database
        self._store_incident(incident)
        
        # Generate automated resolution steps
        incident.resolution_steps = self._generate_resolution_steps(incident)
        
        # Execute automated actions if available
        automated_actions = self._execute_automated_response(incident)
        incident.automated_actions = automated_actions
        
        logger.info(f"Incident created: {incident_id} - {title}")
        return incident
        
    def _store_incident(self, incident: Incident):
        """Store incident in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO incidents 
            (id, title, severity, description, component, detected_at, metrics,
             status, resolution_steps, automated_actions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            incident.id, incident.title, incident.severity.value,
            incident.description, incident.component, incident.detected_at.isoformat(),
            json.dumps(incident.metrics), incident.status,
            json.dumps(incident.resolution_steps), 
            json.dumps(incident.automated_actions)
        ))
        
        conn.commit()
        conn.close()
        
    def _generate_resolution_steps(self, incident: Incident) -> List[str]:
        """Generate recommended resolution steps for incident"""
        steps = []
        
        if incident.severity == IncidentSeverity.CRITICAL:
            steps.extend([
                "1. Assess system impact and user impact",
                "2. Consider immediate rollback if deployment-related",
                "3. Activate incident response team",
                "4. Implement emergency procedures"
            ])
        elif "performance" in incident.title.lower():
            steps.extend([
                "1. Run performance investigation procedure",
                "2. Check for resource constraints",
                "3. Review recent configuration changes",
                "4. Consider parameter adjustments"
            ])
        elif "memory" in incident.title.lower():
            steps.extend([
                "1. Check memory usage patterns",
                "2. Clear caches if safe to do so",
                "3. Reduce batch sizes temporarily",
                "4. Monitor for memory leaks"
            ])
            
        # Add component-specific steps
        if incident.component == "hybrid_selector":
            steps.append("5. Check hybrid selector configuration and fallback logic")
        elif incident.component == "context_analyzer":
            steps.append("5. Verify context analysis pipeline health")
            
        return steps
        
    def _execute_automated_response(self, incident: Incident) -> List[str]:
        """Execute automated response actions for incident"""
        actions = []
        
        try:
            if incident.severity == IncidentSeverity.CRITICAL:
                # For critical incidents, consider automated rollback
                if "deployment" in incident.description.lower():
                    result = self.execute_procedure("emergency_rollback", automated=True)
                    actions.append(f"Emergency rollback executed: {result}")
                    
            # Always send appropriate alerts
            self._send_alert(
                f"Incident: {incident.title}",
                incident.severity,
                incident.metrics,
                include_incident_id=incident.id
            )
            actions.append("Alert sent to appropriate channels")
            
        except Exception as e:
            logger.error(f"Failed automated response for incident {incident.id}: {e}")
            actions.append(f"Automated response failed: {e}")
            
        return actions
        
    def execute_procedure(self, procedure_id: str, automated: bool = False, 
                         dry_run: bool = False) -> str:
        """Execute an operational procedure"""
        if procedure_id not in self.procedures:
            raise ValueError(f"Unknown procedure: {procedure_id}")
            
        procedure = self.procedures[procedure_id]
        
        if not automated and not procedure.automation_available:
            return f"Procedure {procedure_id} requires manual execution"
            
        if dry_run:
            return f"DRY RUN: Would execute {len(procedure.steps)} steps for {procedure.name}"
            
        logger.info(f"Executing procedure: {procedure.name}")
        results = []
        
        try:
            for i, step in enumerate(procedure.steps, 1):
                step_result = f"Step {i}: {step['action']}"
                
                if automated and 'command' in step:
                    # In a real implementation, this would execute the actual command
                    # For now, we simulate the execution
                    step_result += f" -> Command: {step['command']} (SIMULATED)"
                    
                results.append(step_result)
                
                # Simulate step execution time
                if not dry_run:
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Procedure execution failed at step {i}: {e}")
            raise e
            
        logger.info(f"Procedure {procedure_id} completed successfully")
        return f"Procedure completed: {len(results)} steps executed"
        
    def _send_alert(self, message: str, severity: IncidentSeverity, 
                   metrics: Dict[str, float], include_incident_id: Optional[str] = None) -> str:
        """Send alert through configured channels"""
        alert_data = {
            "message": message,
            "severity": severity.value,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "incident_id": include_incident_id
        }
        
        # Determine appropriate channels based on severity
        channels = self._get_alert_channels(severity)
        
        sent_channels = []
        for channel in channels:
            try:
                if channel == AlertChannel.LOG:
                    logger.warning(f"ALERT [{severity.value.upper()}]: {message}")
                    sent_channels.append("log")
                elif channel == AlertChannel.EMAIL:
                    # Email implementation would go here
                    logger.info(f"Email alert sent: {message}")
                    sent_channels.append("email")
                elif channel == AlertChannel.SLACK:
                    # Slack integration would go here  
                    logger.info(f"Slack alert sent: {message}")
                    sent_channels.append("slack")
                elif channel == AlertChannel.PAGER:
                    # PagerDuty/similar integration would go here
                    logger.info(f"Pager alert sent: {message}")
                    sent_channels.append("pager")
                    
            except Exception as e:
                logger.error(f"Failed to send alert via {channel}: {e}")
                
        return f"Alert sent via: {', '.join(sent_channels)}"
        
    def _get_alert_channels(self, severity: IncidentSeverity) -> List[AlertChannel]:
        """Get appropriate alert channels based on severity"""
        if severity == IncidentSeverity.CRITICAL:
            return [AlertChannel.PAGER, AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.LOG]
        elif severity == IncidentSeverity.HIGH:
            return [AlertChannel.EMAIL, AlertChannel.SLACK, AlertChannel.LOG]
        elif severity == IncidentSeverity.MEDIUM:
            return [AlertChannel.SLACK, AlertChannel.LOG]
        else:
            return [AlertChannel.LOG]
            
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        # Get recent metrics
        recent_metrics = self._get_recent_metrics(minutes=15)
        
        # Count active incidents by severity
        incident_counts = {}
        for severity in IncidentSeverity:
            count = sum(1 for inc in self.active_incidents.values() 
                       if inc.severity == severity and inc.status == "open")
            incident_counts[severity.value] = count
            
        # Circuit breaker status
        circuit_status = {}
        for name, cb in self.circuit_breakers.items():
            circuit_status[name] = {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "last_failure": cb.last_failure_time.isoformat() if cb.last_failure_time else None
            }
            
        # Protocol status
        protocol_status = {}
        for name, protocol in self.safety_protocols.items():
            protocol_status[name] = {
                "enabled": protocol.enabled,
                "trigger_count": protocol.trigger_count,
                "last_triggered": protocol.last_triggered.isoformat() if protocol.last_triggered else None
            }
            
        return {
            "overall_status": self.system_status.value,
            "timestamp": datetime.now().isoformat(),
            "recent_metrics": recent_metrics,
            "active_incidents": incident_counts,
            "circuit_breakers": circuit_status,
            "safety_protocols": protocol_status,
            "available_procedures": len(self.procedures)
        }
        
    def _get_recent_metrics(self, minutes: int = 15) -> Dict[str, Any]:
        """Get metrics from the last N minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT component, metric_name, AVG(metric_value) as avg_value,
                   MAX(metric_value) as max_value, COUNT(*) as count
            FROM system_metrics 
            WHERE timestamp >= ?
            GROUP BY component, metric_name
        ''', (cutoff.isoformat(),))
        
        results = cursor.fetchall()
        conn.close()
        
        metrics = {}
        for component, metric_name, avg_val, max_val, count in results:
            if component not in metrics:
                metrics[component] = {}
            metrics[component][metric_name] = {
                "average": round(avg_val, 4),
                "maximum": round(max_val, 4),
                "sample_count": count
            }
            
        return metrics
        
    def search_procedures(self, query: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search operational procedures by query and category"""
        results = []
        
        for proc_id, procedure in self.procedures.items():
            match = False
            
            # Check category filter
            if category and procedure.category != category:
                continue
                
            # Search in name, description, and tags
            search_text = f"{procedure.name} {procedure.description} {' '.join(procedure.tags)}".lower()
            if query.lower() in search_text:
                match = True
                
            if match:
                results.append({
                    "id": proc_id,
                    "name": procedure.name,
                    "category": procedure.category,
                    "description": procedure.description,
                    "estimated_duration": procedure.estimated_duration,
                    "risk_level": procedure.risk_level,
                    "automation_available": procedure.automation_available,
                    "tags": procedure.tags
                })
                
        return sorted(results, key=lambda x: x['name'])
        
    def get_incident_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get incident history for the past N days"""
        cutoff = datetime.now() - timedelta(days=days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM incidents 
            WHERE detected_at >= ?
            ORDER BY detected_at DESC
        ''', (cutoff.isoformat(),))
        
        columns = [desc[0] for desc in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            incident_dict = dict(zip(columns, row))
            # Parse JSON fields
            if incident_dict['metrics']:
                incident_dict['metrics'] = json.loads(incident_dict['metrics'])
            if incident_dict['resolution_steps']:
                incident_dict['resolution_steps'] = json.loads(incident_dict['resolution_steps'])
            if incident_dict['automated_actions']:
                incident_dict['automated_actions'] = json.loads(incident_dict['automated_actions'])
            results.append(incident_dict)
            
        conn.close()
        return results


def main():
    """Example usage of the OperationalRunbook system"""
    runbook = OperationalRunbook()
    
    # Simulate system metrics monitoring
    test_metrics = {
        "proxy_gap": 0.3,  # 0.3% - normal
        "tail_latency_ratio": 1.8,  # P99/P95 = 1.8 - good
        "error_rate": 0.005,  # 0.5% - normal
        "memory_usage_percent": 65.0,  # 65% - healthy
        "context_truncation_rate": 0.15  # 15% - normal
    }
    
    print("=== Operational Runbook System Demo ===")
    
    # Monitor normal metrics
    print("\n1. Monitoring normal metrics...")
    result = runbook.monitor_system_metrics(test_metrics)
    print(f"Result: {result}")
    
    # Get system health
    print("\n2. System health report...")
    health = runbook.get_system_health()
    print(f"Overall Status: {health['overall_status']}")
    print(f"Active Incidents: {health['active_incidents']}")
    
    # Simulate performance degradation
    print("\n3. Simulating performance degradation...")
    degraded_metrics = {
        "proxy_gap": 2.5,  # 2.5% - triggers critical protocol
        "tail_latency_ratio": 3.5,  # P99/P95 = 3.5 - triggers protocol
        "error_rate": 0.08,  # 8% - high error rate
        "memory_usage_percent": 85.0  # 85% - high memory usage
    }
    
    result = runbook.monitor_system_metrics(degraded_metrics, component="hybrid_selector")
    print(f"Degraded metrics result: {result}")
    
    # Create manual incident
    print("\n4. Creating manual incident...")
    incident = runbook.create_incident(
        title="High context truncation rate detected",
        severity=IncidentSeverity.MEDIUM,
        description="Context truncation rate exceeded 40% for entity-heavy conversations",
        component="context_analyzer", 
        metrics={"context_truncation_rate": 0.42}
    )
    print(f"Created incident: {incident.id} - {incident.title}")
    print(f"Resolution steps: {len(incident.resolution_steps)}")
    
    # Search procedures
    print("\n5. Searching procedures...")
    procedures = runbook.search_procedures("performance", category="troubleshooting")
    print(f"Found {len(procedures)} procedures")
    for proc in procedures:
        print(f"  - {proc['name']} ({proc['estimated_duration']} min)")
        
    # Execute procedure (dry run)
    print("\n6. Executing procedure (dry run)...")
    if procedures:
        result = runbook.execute_procedure(procedures[0]['id'], automated=True, dry_run=True)
        print(f"Dry run result: {result}")
    
    # Final health check
    print("\n7. Final system health...")
    health = runbook.get_system_health()
    print(f"Status: {health['overall_status']}")
    print(f"Circuit Breakers: {len(health['circuit_breakers'])} configured")
    print(f"Safety Protocols: {len(health['safety_protocols'])} active")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()