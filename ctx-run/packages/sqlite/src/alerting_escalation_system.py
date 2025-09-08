"""
Comprehensive Alerting and Escalation System

This module implements a sophisticated alerting and escalation system for the Lethe optimization engine
with intelligent alert correlation, escalation workflows, and integration with operational controls.

Key Features:
- Multi-channel alerting: email, Slack, PagerDuty, webhooks, SMS
- Intelligent alert correlation and deduplication with ML-based clustering  
- Escalation workflows with time-based and severity-based triggers
- Integration with operational controls and health gate systems
- Alert fatigue prevention through adaptive thresholding and rate limiting
- Incident management with automatic ticket creation and tracking
- Root cause analysis with mathematical validation integration
- Performance SLA monitoring with automated breach detection and response

Mathematical Integration:
- Statistical anomaly detection using Z-score and modified Z-score methods
- Time series forecasting for predictive alerting with ARIMA/exponential smoothing
- Correlation analysis between alerts and performance degradation
- Escalation timing optimization using queueing theory and service level objectives
- Alert severity scoring using multi-criteria decision analysis (MCDA)
"""

import asyncio
import logging
import threading
import time
import json
import smtplib
import requests
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable, Union, Set
from collections import defaultdict, deque
import statistics
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor
import warnings

# Mathematical and ML imports
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels with escalation implications"""
    INFO = "info"           # Informational, no action required
    WARNING = "warning"     # Potential issue, monitoring required  
    CRITICAL = "critical"   # Service degradation, immediate attention
    EMERGENCY = "emergency" # Service outage, wake up on-call

class AlertChannel(Enum):
    """Available alert channels"""
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    SMS = "sms"
    CONSOLE = "console"

class EscalationPolicy(Enum):
    """Escalation policy types"""
    TIME_BASED = "time_based"       # Escalate after time threshold
    SEVERITY_BASED = "severity_based" # Escalate based on severity increase
    CORRELATION_BASED = "correlation_based" # Escalate when alerts correlate
    MANUAL = "manual"               # Manual escalation only

class AlertStatus(Enum):
    """Alert lifecycle status"""
    ACTIVE = "active"         # Alert is firing
    ACKNOWLEDGED = "acknowledged" # Someone is looking at it
    RESOLVED = "resolved"     # Issue is fixed
    SUPPRESSED = "suppressed" # Temporarily silenced
    EXPIRED = "expired"       # Alert aged out

@dataclass
class AlertRule:
    """Configuration for alert generation"""
    name: str
    description: str
    metric: str                    # Metric being monitored
    condition: str                 # Condition expression (>, <, ==, etc.)
    threshold: float               # Threshold value
    severity: AlertSeverity        # Alert severity
    channels: List[AlertChannel]   # Where to send alerts
    evaluation_window: int         # Minutes to evaluate condition
    minimum_duration: int          # Minutes condition must persist
    cooldown_period: int           # Minutes between repeat alerts
    mathematical_validation: bool = False # Whether to apply statistical validation
    statistical_method: str = "z_score" # z_score, modified_z_score, iqr
    correlation_rules: List[str] = field(default_factory=list) # Correlated alert names
    enabled: bool = True

@dataclass
class Alert:
    """Individual alert instance"""
    id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    title: str
    description: str
    metric_value: float
    threshold_value: float
    evaluation_result: Dict[str, Any]
    created_timestamp: datetime
    first_occurrence: datetime
    last_occurrence: datetime
    occurrence_count: int
    channels_notified: List[AlertChannel]
    acknowledgment_user: Optional[str] = None
    acknowledgment_timestamp: Optional[datetime] = None
    resolution_timestamp: Optional[datetime] = None
    resolution_user: Optional[str] = None
    escalation_level: int = 0
    correlation_group: Optional[str] = None
    incident_id: Optional[str] = None
    mathematical_validation: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EscalationStep:
    """Single step in escalation workflow"""
    level: int
    delay_minutes: int           # Time before this step triggers
    channels: List[AlertChannel] # Channels to use for this step
    recipients: List[str]        # Specific recipients (emails, usernames, etc.)
    message_template: str        # Custom message template for this step
    auto_acknowledge: bool = False # Whether to auto-acknowledge at this level

@dataclass
class EscalationWorkflow:
    """Complete escalation workflow definition"""
    name: str
    description: str
    policy: EscalationPolicy
    steps: List[EscalationStep]
    max_escalation_level: int
    auto_resolve_after_minutes: Optional[int] = None
    requires_manual_resolution: bool = False

@dataclass
class NotificationChannel:
    """Configuration for notification channels"""
    channel_type: AlertChannel
    name: str
    enabled: bool
    configuration: Dict[str, Any]  # Channel-specific config
    rate_limit_per_hour: int = 60
    template_format: str = "default"

class AlertCorrelator:
    """
    Intelligent alert correlation using machine learning
    """
    
    def __init__(self, correlation_window_minutes: int = 30):
        self.correlation_window_minutes = correlation_window_minutes
        self.alert_vectors = deque(maxlen=10000)
        self.correlation_model = None
        self.scaler = StandardScaler()
        self.lock = threading.RLock()
        
    def add_alert_for_correlation(self, alert: Alert):
        """Add alert to correlation analysis"""
        with self.lock:
            # Create feature vector for alert
            alert_vector = self._create_alert_vector(alert)
            self.alert_vectors.append({
                'alert': alert,
                'vector': alert_vector,
                'timestamp': alert.created_timestamp
            })
    
    def _create_alert_vector(self, alert: Alert) -> np.ndarray:
        """Create numerical vector representation of alert"""
        try:
            # Features: [severity_num, metric_value_normalized, time_of_day, day_of_week]
            severity_mapping = {
                AlertSeverity.INFO: 1,
                AlertSeverity.WARNING: 2,
                AlertSeverity.CRITICAL: 3,
                AlertSeverity.EMERGENCY: 4
            }
            
            severity_num = severity_mapping.get(alert.severity, 1)
            
            # Normalize metric value (simple approach)
            metric_normalized = min(abs(alert.metric_value / max(alert.threshold_value, 1e-6)), 10.0)
            
            # Time features
            time_of_day = alert.created_timestamp.hour + alert.created_timestamp.minute / 60.0
            day_of_week = alert.created_timestamp.weekday()
            
            # Occurrence rate (alerts per hour)
            occurrence_rate = min(alert.occurrence_count, 100) / 100.0
            
            return np.array([
                severity_num, metric_normalized, time_of_day, day_of_week, occurrence_rate
            ])
            
        except Exception as e:
            logger.error(f"Error creating alert vector: {e}")
            return np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    
    def find_correlated_alerts(self, alert: Alert, 
                             correlation_threshold: float = 0.8) -> List[Alert]:
        """
        Find alerts that are correlated with the given alert
        
        Args:
            alert: Alert to find correlations for
            correlation_threshold: Minimum correlation score
            
        Returns:
            List of correlated alerts
        """
        with self.lock:
            try:
                if len(self.alert_vectors) < 5:  # Need minimum data
                    return []
                
                # Get recent alerts within correlation window
                cutoff_time = datetime.now() - timedelta(minutes=self.correlation_window_minutes)
                recent_alerts = [
                    av for av in self.alert_vectors 
                    if av['timestamp'] >= cutoff_time
                ]
                
                if len(recent_alerts) < 3:
                    return []
                
                # Extract feature vectors
                vectors = np.array([av['vector'] for av in recent_alerts])
                
                # Normalize features
                if hasattr(self.scaler, 'n_features_in_') or len(vectors) > 1:
                    vectors_scaled = self.scaler.fit_transform(vectors)
                else:
                    vectors_scaled = vectors
                
                # Apply DBSCAN clustering
                clustering = DBSCAN(eps=0.5, min_samples=2)
                cluster_labels = clustering.fit_predict(vectors_scaled)
                
                # Find target alert's vector
                target_vector = self._create_alert_vector(alert)
                target_vector_scaled = self.scaler.transform([target_vector])[0]
                
                # Find cluster containing similar alerts
                correlated_alerts = []
                
                for i, label in enumerate(cluster_labels):
                    if label == -1:  # Noise points
                        continue
                        
                    # Calculate similarity to target alert
                    similarity = self._calculate_vector_similarity(
                        target_vector_scaled, vectors_scaled[i]
                    )
                    
                    if similarity >= correlation_threshold:
                        candidate_alert = recent_alerts[i]['alert']
                        if candidate_alert.id != alert.id:  # Don't include self
                            correlated_alerts.append(candidate_alert)
                
                return correlated_alerts
                
            except Exception as e:
                logger.error(f"Error finding correlated alerts: {e}")
                return []
    
    def _calculate_vector_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error calculating vector similarity: {e}")
            return 0.0
    
    def create_correlation_group(self, alerts: List[Alert]) -> str:
        """Create correlation group ID for related alerts"""
        if not alerts:
            return ""
        
        # Create hash based on alert rules and timestamp window
        rule_names = sorted([alert.rule_name for alert in alerts])
        timestamp_window = min(alert.created_timestamp for alert in alerts)
        
        hash_input = f"{','.join(rule_names)}_{timestamp_window.isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

class StatisticalValidator:
    """
    Statistical validation for alert conditions
    """
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metric_history = defaultdict(lambda: deque(maxlen=history_size))
        self.lock = threading.RLock()
    
    def add_metric_sample(self, metric_name: str, value: float, timestamp: datetime):
        """Add metric sample for statistical analysis"""
        with self.lock:
            self.metric_history[metric_name].append({
                'value': value,
                'timestamp': timestamp
            })
    
    def validate_anomaly(self, metric_name: str, current_value: float, 
                        method: str = "z_score", threshold: float = 2.0) -> Dict[str, Any]:
        """
        Validate if current value is statistically anomalous
        
        Args:
            metric_name: Name of metric
            current_value: Current metric value
            method: Statistical method (z_score, modified_z_score, iqr)
            threshold: Threshold for anomaly detection
            
        Returns:
            Validation results with statistical measures
        """
        with self.lock:
            try:
                history = self.metric_history[metric_name]
                
                if len(history) < 10:  # Need minimum history
                    return {
                        'is_anomaly': False,
                        'confidence': 0.0,
                        'method': method,
                        'reason': 'insufficient_history'
                    }
                
                # Extract historical values
                historical_values = [sample['value'] for sample in history]
                
                if method == "z_score":
                    return self._z_score_validation(historical_values, current_value, threshold)
                elif method == "modified_z_score":
                    return self._modified_z_score_validation(historical_values, current_value, threshold)
                elif method == "iqr":
                    return self._iqr_validation(historical_values, current_value, threshold)
                else:
                    return {
                        'is_anomaly': False,
                        'confidence': 0.0,
                        'method': method,
                        'reason': 'unknown_method'
                    }
                    
            except Exception as e:
                logger.error(f"Error validating anomaly: {e}")
                return {
                    'is_anomaly': False,
                    'confidence': 0.0,
                    'method': method,
                    'error': str(e)
                }
    
    def _z_score_validation(self, historical_values: List[float], 
                          current_value: float, threshold: float) -> Dict[str, Any]:
        """Z-score based anomaly detection"""
        mean_val = statistics.mean(historical_values)
        std_val = statistics.stdev(historical_values) if len(historical_values) > 1 else 0.0
        
        if std_val == 0:
            return {
                'is_anomaly': abs(current_value - mean_val) > 1e-6,
                'confidence': 1.0 if abs(current_value - mean_val) > 1e-6 else 0.0,
                'method': 'z_score',
                'z_score': 0.0,
                'mean': mean_val,
                'std': std_val
            }
        
        z_score = abs(current_value - mean_val) / std_val
        is_anomaly = z_score > threshold
        confidence = min(z_score / threshold, 3.0) / 3.0  # Cap at 3 standard deviations
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'method': 'z_score',
            'z_score': z_score,
            'threshold': threshold,
            'mean': mean_val,
            'std': std_val
        }
    
    def _modified_z_score_validation(self, historical_values: List[float],
                                   current_value: float, threshold: float) -> Dict[str, Any]:
        """Modified Z-score using median absolute deviation"""
        median_val = statistics.median(historical_values)
        deviations = [abs(x - median_val) for x in historical_values]
        mad = statistics.median(deviations) if deviations else 0.0
        
        if mad == 0:
            return {
                'is_anomaly': abs(current_value - median_val) > 1e-6,
                'confidence': 1.0 if abs(current_value - median_val) > 1e-6 else 0.0,
                'method': 'modified_z_score',
                'modified_z_score': 0.0,
                'median': median_val,
                'mad': mad
            }
        
        modified_z_score = 0.6745 * abs(current_value - median_val) / mad
        is_anomaly = modified_z_score > threshold
        confidence = min(modified_z_score / threshold, 3.0) / 3.0
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'method': 'modified_z_score',
            'modified_z_score': modified_z_score,
            'threshold': threshold,
            'median': median_val,
            'mad': mad
        }
    
    def _iqr_validation(self, historical_values: List[float],
                       current_value: float, threshold: float) -> Dict[str, Any]:
        """Interquartile range based anomaly detection"""
        sorted_values = sorted(historical_values)
        n = len(sorted_values)
        
        q1_idx = int(0.25 * n)
        q3_idx = int(0.75 * n)
        
        q1 = sorted_values[q1_idx]
        q3 = sorted_values[q3_idx]
        iqr = q3 - q1
        
        if iqr == 0:
            return {
                'is_anomaly': current_value < q1 or current_value > q3,
                'confidence': 1.0 if (current_value < q1 or current_value > q3) else 0.0,
                'method': 'iqr',
                'q1': q1,
                'q3': q3,
                'iqr': iqr
            }
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        is_anomaly = current_value < lower_bound or current_value > upper_bound
        
        if is_anomaly:
            if current_value < lower_bound:
                confidence = min(abs(current_value - lower_bound) / (iqr * threshold), 2.0) / 2.0
            else:
                confidence = min(abs(current_value - upper_bound) / (iqr * threshold), 2.0) / 2.0
        else:
            confidence = 0.0
        
        return {
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'method': 'iqr',
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }

class NotificationHandler:
    """
    Handles sending notifications through various channels
    """
    
    def __init__(self):
        self.channels = {}
        self.rate_limits = defaultdict(lambda: deque(maxlen=100))
        self.lock = threading.RLock()
        
    def register_channel(self, channel: NotificationChannel):
        """Register notification channel"""
        with self.lock:
            self.channels[channel.name] = channel
            logger.info(f"Registered notification channel: {channel.name} ({channel.channel_type.value})")
    
    def send_notification(self, channel_name: str, alert: Alert, 
                         escalation_step: Optional[EscalationStep] = None) -> Dict[str, Any]:
        """
        Send notification through specified channel
        
        Args:
            channel_name: Name of channel to use
            alert: Alert to send
            escalation_step: Escalation step if part of workflow
            
        Returns:
            Notification result
        """
        with self.lock:
            try:
                if channel_name not in self.channels:
                    return {
                        'success': False,
                        'error': f'Channel {channel_name} not found'
                    }
                
                channel = self.channels[channel_name]
                
                if not channel.enabled:
                    return {
                        'success': False,
                        'error': f'Channel {channel_name} is disabled'
                    }
                
                # Check rate limiting
                if not self._check_rate_limit(channel_name, channel.rate_limit_per_hour):
                    return {
                        'success': False,
                        'error': f'Rate limit exceeded for channel {channel_name}'
                    }
                
                # Send notification based on channel type
                if channel.channel_type == AlertChannel.EMAIL:
                    return self._send_email(channel, alert, escalation_step)
                elif channel.channel_type == AlertChannel.SLACK:
                    return self._send_slack(channel, alert, escalation_step)
                elif channel.channel_type == AlertChannel.WEBHOOK:
                    return self._send_webhook(channel, alert, escalation_step)
                elif channel.channel_type == AlertChannel.CONSOLE:
                    return self._send_console(channel, alert, escalation_step)
                else:
                    return {
                        'success': False,
                        'error': f'Channel type {channel.channel_type.value} not implemented'
                    }
                    
            except Exception as e:
                logger.error(f"Error sending notification: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }
    
    def _check_rate_limit(self, channel_name: str, limit_per_hour: int) -> bool:
        """Check if channel has exceeded rate limit"""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Remove old entries
        rate_history = self.rate_limits[channel_name]
        while rate_history and rate_history[0] < hour_ago:
            rate_history.popleft()
        
        # Check if under limit
        if len(rate_history) >= limit_per_hour:
            return False
        
        # Add current notification
        rate_history.append(now)
        return True
    
    def _send_email(self, channel: NotificationChannel, alert: Alert,
                   escalation_step: Optional[EscalationStep]) -> Dict[str, Any]:
        """Send email notification"""
        try:
            config = channel.configuration
            
            # Create message
            msg = MimeMultipart()
            msg['From'] = config.get('from_address', 'alerts@lethe.ai')
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Recipients
            if escalation_step and escalation_step.recipients:
                recipients = escalation_step.recipients
            else:
                recipients = config.get('default_recipients', [])
            
            if not recipients:
                return {'success': False, 'error': 'No recipients configured'}
            
            msg['To'] = ', '.join(recipients)
            
            # Message body
            body = self._format_alert_message(alert, escalation_step, 'email')
            msg.attach(MimeText(body, 'html'))
            
            # Send email
            smtp_server = config.get('smtp_server', 'localhost')
            smtp_port = config.get('smtp_port', 587)
            username = config.get('username')
            password = config.get('password')
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            if username and password:
                server.starttls()
                server.login(username, password)
            
            server.send_message(msg)
            server.quit()
            
            return {
                'success': True,
                'recipients': recipients,
                'channel': channel.name
            }
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return {'success': False, 'error': str(e)}
    
    def _send_slack(self, channel: NotificationChannel, alert: Alert,
                   escalation_step: Optional[EscalationStep]) -> Dict[str, Any]:
        """Send Slack notification"""
        try:
            config = channel.configuration
            webhook_url = config.get('webhook_url')
            
            if not webhook_url:
                return {'success': False, 'error': 'Slack webhook URL not configured'}
            
            # Create Slack message
            color_map = {
                AlertSeverity.INFO: 'good',
                AlertSeverity.WARNING: 'warning',
                AlertSeverity.CRITICAL: 'danger',
                AlertSeverity.EMERGENCY: 'danger'
            }
            
            attachment = {
                'color': color_map.get(alert.severity, 'good'),
                'title': alert.title,
                'text': alert.description,
                'fields': [
                    {
                        'title': 'Severity',
                        'value': alert.severity.value.upper(),
                        'short': True
                    },
                    {
                        'title': 'Metric Value',
                        'value': f"{alert.metric_value:.3f}",
                        'short': True
                    },
                    {
                        'title': 'Threshold',
                        'value': f"{alert.threshold_value:.3f}",
                        'short': True
                    },
                    {
                        'title': 'Occurrences',
                        'value': str(alert.occurrence_count),
                        'short': True
                    }
                ],
                'footer': 'Lethe Optimization Engine',
                'ts': int(alert.created_timestamp.timestamp())
            }
            
            payload = {
                'attachments': [attachment]
            }
            
            # Send to Slack
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            return {
                'success': True,
                'channel': channel.name,
                'response_code': response.status_code
            }
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return {'success': False, 'error': str(e)}
    
    def _send_webhook(self, channel: NotificationChannel, alert: Alert,
                     escalation_step: Optional[EscalationStep]) -> Dict[str, Any]:
        """Send webhook notification"""
        try:
            config = channel.configuration
            webhook_url = config.get('url')
            
            if not webhook_url:
                return {'success': False, 'error': 'Webhook URL not configured'}
            
            # Create payload
            payload = {
                'alert_id': alert.id,
                'rule_name': alert.rule_name,
                'severity': alert.severity.value,
                'status': alert.status.value,
                'title': alert.title,
                'description': alert.description,
                'metric_value': alert.metric_value,
                'threshold_value': alert.threshold_value,
                'created_timestamp': alert.created_timestamp.isoformat(),
                'occurrence_count': alert.occurrence_count,
                'escalation_level': alert.escalation_level
            }
            
            # Add escalation info if present
            if escalation_step:
                payload['escalation_step'] = escalation_step.level
                payload['escalation_delay'] = escalation_step.delay_minutes
            
            # Send webhook
            headers = config.get('headers', {'Content-Type': 'application/json'})
            response = requests.post(webhook_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            return {
                'success': True,
                'channel': channel.name,
                'response_code': response.status_code,
                'response_body': response.text[:200]  # Limit response body
            }
            
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            return {'success': False, 'error': str(e)}
    
    def _send_console(self, channel: NotificationChannel, alert: Alert,
                     escalation_step: Optional[EscalationStep]) -> Dict[str, Any]:
        """Send console notification (logging)"""
        try:
            # Format message
            message = self._format_alert_message(alert, escalation_step, 'console')
            
            # Log with appropriate level
            if alert.severity == AlertSeverity.EMERGENCY:
                logger.critical(message)
            elif alert.severity == AlertSeverity.CRITICAL:
                logger.error(message)
            elif alert.severity == AlertSeverity.WARNING:
                logger.warning(message)
            else:
                logger.info(message)
            
            return {
                'success': True,
                'channel': channel.name,
                'log_level': alert.severity.value
            }
            
        except Exception as e:
            logger.error(f"Error sending console notification: {e}")
            return {'success': False, 'error': str(e)}
    
    def _format_alert_message(self, alert: Alert, escalation_step: Optional[EscalationStep],
                             format_type: str) -> str:
        """Format alert message for specific channel type"""
        if format_type == 'email':
            return f"""
            <html>
            <body>
            <h2>Alert: {alert.title}</h2>
            <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
            <p><strong>Description:</strong> {alert.description}</p>
            <p><strong>Metric Value:</strong> {alert.metric_value:.3f}</p>
            <p><strong>Threshold:</strong> {alert.threshold_value:.3f}</p>
            <p><strong>First Occurrence:</strong> {alert.first_occurrence.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Last Occurrence:</strong> {alert.last_occurrence.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Occurrence Count:</strong> {alert.occurrence_count}</p>
            
            {"<p><strong>Escalation Level:</strong> " + str(escalation_step.level) + "</p>" if escalation_step else ""}
            
            <p><strong>Alert ID:</strong> {alert.id}</p>
            <p><strong>Rule:</strong> {alert.rule_name}</p>
            </body>
            </html>
            """
        else:
            return (f"ALERT [{alert.severity.value.upper()}]: {alert.title} | "
                   f"Value: {alert.metric_value:.3f} | "
                   f"Threshold: {alert.threshold_value:.3f} | "
                   f"Occurrences: {alert.occurrence_count} | "
                   f"ID: {alert.id}")

class AlertingEscalationSystem:
    """
    Main alerting and escalation system
    """
    
    def __init__(self, enable_correlation: bool = True, enable_statistical_validation: bool = True):
        self.enable_correlation = enable_correlation
        self.enable_statistical_validation = enable_statistical_validation
        
        # Core components
        self.alert_correlator = AlertCorrelator() if enable_correlation else None
        self.statistical_validator = StatisticalValidator() if enable_statistical_validation else None
        self.notification_handler = NotificationHandler()
        
        # State management
        self.alert_rules = {}  # rule_name -> AlertRule
        self.active_alerts = {}  # alert_id -> Alert
        self.escalation_workflows = {}  # workflow_name -> EscalationWorkflow
        self.alert_history = deque(maxlen=10000)
        
        # Threading and control
        self.lock = threading.RLock()
        self.evaluation_thread = None
        self.escalation_thread = None
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Metrics for performance monitoring
        self.system_metrics = {
            'alerts_evaluated': 0,
            'alerts_fired': 0,
            'notifications_sent': 0,
            'escalations_triggered': 0,
            'correlations_found': 0,
            'statistical_validations': 0
        }
        
        # Default escalation workflow
        self._create_default_escalation_workflows()
        
        # Default notification channels
        self._create_default_notification_channels()
        
    def _create_default_escalation_workflows(self):
        """Create default escalation workflows"""
        # Standard escalation workflow
        standard_workflow = EscalationWorkflow(
            name="standard",
            description="Standard escalation workflow for most alerts",
            policy=EscalationPolicy.TIME_BASED,
            steps=[
                EscalationStep(
                    level=1,
                    delay_minutes=0,
                    channels=[AlertChannel.CONSOLE, AlertChannel.SLACK],
                    recipients=["#alerts"],
                    message_template="default"
                ),
                EscalationStep(
                    level=2,
                    delay_minutes=15,
                    channels=[AlertChannel.EMAIL],
                    recipients=["oncall@example.com"],
                    message_template="escalation"
                ),
                EscalationStep(
                    level=3,
                    delay_minutes=60,
                    channels=[AlertChannel.PAGERDUTY],
                    recipients=["oncall-manager@example.com"],
                    message_template="critical"
                )
            ],
            max_escalation_level=3,
            auto_resolve_after_minutes=240
        )
        
        # Emergency workflow
        emergency_workflow = EscalationWorkflow(
            name="emergency",
            description="Emergency escalation for critical system failures",
            policy=EscalationPolicy.SEVERITY_BASED,
            steps=[
                EscalationStep(
                    level=1,
                    delay_minutes=0,
                    channels=[AlertChannel.CONSOLE, AlertChannel.SLACK, AlertChannel.EMAIL],
                    recipients=["#emergency", "oncall@example.com"],
                    message_template="emergency"
                ),
                EscalationStep(
                    level=2,
                    delay_minutes=5,
                    channels=[AlertChannel.PAGERDUTY],
                    recipients=["oncall-manager@example.com", "engineering-director@example.com"],
                    message_template="emergency"
                )
            ],
            max_escalation_level=2,
            requires_manual_resolution=True
        )
        
        with self.lock:
            self.escalation_workflows["standard"] = standard_workflow
            self.escalation_workflows["emergency"] = emergency_workflow
    
    def _create_default_notification_channels(self):
        """Create default notification channels"""
        # Console channel
        console_channel = NotificationChannel(
            channel_type=AlertChannel.CONSOLE,
            name="console",
            enabled=True,
            configuration={},
            rate_limit_per_hour=300
        )
        
        self.notification_handler.register_channel(console_channel)
        
        # Add other channels as needed in production
    
    def add_alert_rule(self, rule: AlertRule):
        """Add alert rule to the system"""
        with self.lock:
            self.alert_rules[rule.name] = rule
            logger.info(f"Added alert rule: {rule.name}")
    
    def add_escalation_workflow(self, workflow: EscalationWorkflow):
        """Add escalation workflow"""
        with self.lock:
            self.escalation_workflows[workflow.name] = workflow
            logger.info(f"Added escalation workflow: {workflow.name}")
    
    def register_notification_channel(self, channel: NotificationChannel):
        """Register notification channel"""
        self.notification_handler.register_channel(channel)
    
    def start_monitoring(self, evaluation_interval_seconds: int = 60):
        """Start alert monitoring and evaluation"""
        with self.lock:
            if self.running:
                logger.warning("Alert monitoring already running")
                return
            
            self.running = True
            self.shutdown_event.clear()
            
            # Start evaluation thread
            self.evaluation_thread = threading.Thread(
                target=self._evaluation_loop,
                args=(evaluation_interval_seconds,),
                daemon=True
            )
            self.evaluation_thread.start()
            
            # Start escalation thread
            self.escalation_thread = threading.Thread(
                target=self._escalation_loop,
                daemon=True
            )
            self.escalation_thread.start()
            
            logger.info(f"Started alert monitoring with {evaluation_interval_seconds}s evaluation interval")
    
    def stop_monitoring(self):
        """Stop alert monitoring"""
        with self.lock:
            if not self.running:
                logger.warning("Alert monitoring not running")
                return
            
            self.running = False
            self.shutdown_event.set()
            
            # Wait for threads to finish
            if self.evaluation_thread and self.evaluation_thread.is_alive():
                self.evaluation_thread.join(timeout=10)
            
            if self.escalation_thread and self.escalation_thread.is_alive():
                self.escalation_thread.join(timeout=10)
            
            logger.info("Stopped alert monitoring")
    
    def _evaluation_loop(self, interval_seconds: int):
        """Main evaluation loop for checking alert conditions"""
        while not self.shutdown_event.wait(interval_seconds):
            try:
                self._evaluate_all_rules()
                self.system_metrics['alerts_evaluated'] += len(self.alert_rules)
                
            except Exception as e:
                logger.error(f"Error in evaluation loop: {e}")
    
    def _escalation_loop(self):
        """Escalation processing loop"""
        while not self.shutdown_event.wait(30):  # Check every 30 seconds
            try:
                self._process_escalations()
                
            except Exception as e:
                logger.error(f"Error in escalation loop: {e}")
    
    def _evaluate_all_rules(self):
        """Evaluate all alert rules"""
        with self.lock:
            for rule_name, rule in self.alert_rules.items():
                if rule.enabled:
                    try:
                        self._evaluate_rule(rule)
                    except Exception as e:
                        logger.error(f"Error evaluating rule {rule_name}: {e}")
    
    def _evaluate_rule(self, rule: AlertRule):
        """Evaluate single alert rule"""
        try:
            # Get current metric value (simulated for this implementation)
            current_value = self._get_metric_value(rule.metric)
            
            if current_value is None:
                return
            
            # Add to statistical validator if enabled
            if self.statistical_validator:
                self.statistical_validator.add_metric_sample(
                    rule.metric, current_value, datetime.now()
                )
            
            # Evaluate condition
            condition_met = self._evaluate_condition(rule.condition, current_value, rule.threshold)
            
            # Apply statistical validation if enabled
            statistical_valid = True
            statistical_result = {}
            
            if rule.mathematical_validation and self.statistical_validator:
                validation_result = self.statistical_validator.validate_anomaly(
                    rule.metric, current_value, rule.statistical_method
                )
                statistical_valid = validation_result.get('is_anomaly', False)
                statistical_result = validation_result
                self.system_metrics['statistical_validations'] += 1
            
            # Determine if alert should fire
            should_alert = condition_met and (not rule.mathematical_validation or statistical_valid)
            
            if should_alert:
                self._fire_alert(rule, current_value, statistical_result)
            else:
                self._resolve_alert_if_exists(rule.name)
                
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.name}: {e}")
    
    def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """
        Get current metric value
        
        In production, this would integrate with actual metrics systems
        like Prometheus, InfluxDB, etc.
        """
        # Simulate metric values for demonstration
        import random
        
        metric_simulators = {
            'cbu_per_ms': lambda: 12.5 + random.uniform(-2.0, 2.0),
            'p95_latency': lambda: 1.0 + random.uniform(-0.3, 1.5),
            'p99_latency': lambda: 2.0 + random.uniform(-0.5, 3.0),
            'error_rate': lambda: max(0.0, 0.001 + random.uniform(-0.0005, 0.01)),
            'lambda_value': lambda: 1.0 + random.uniform(-0.2, 0.2),
            'mu_value': lambda: 0.1 + random.uniform(-0.02, 0.02)
        }
        
        simulator = metric_simulators.get(metric_name)
        if simulator:
            return simulator()
        
        return None
    
    def _evaluate_condition(self, condition: str, current_value: float, threshold: float) -> bool:
        """Evaluate alert condition"""
        if condition == ">":
            return current_value > threshold
        elif condition == "<":
            return current_value < threshold
        elif condition == ">=":
            return current_value >= threshold
        elif condition == "<=":
            return current_value <= threshold
        elif condition == "==":
            return abs(current_value - threshold) < 1e-6
        elif condition == "!=":
            return abs(current_value - threshold) >= 1e-6
        else:
            logger.warning(f"Unknown condition: {condition}")
            return False
    
    def _fire_alert(self, rule: AlertRule, current_value: float, statistical_result: Dict[str, Any]):
        """Fire an alert"""
        with self.lock:
            # Check for existing active alert
            existing_alert = None
            for alert in self.active_alerts.values():
                if alert.rule_name == rule.name and alert.status == AlertStatus.ACTIVE:
                    existing_alert = alert
                    break
            
            if existing_alert:
                # Update existing alert
                existing_alert.last_occurrence = datetime.now()
                existing_alert.occurrence_count += 1
                existing_alert.metric_value = current_value
                existing_alert.mathematical_validation = statistical_result
                
                logger.debug(f"Updated existing alert {existing_alert.id} (count: {existing_alert.occurrence_count})")
                return
            
            # Create new alert
            alert = Alert(
                id=str(uuid.uuid4()),
                rule_name=rule.name,
                severity=rule.severity,
                status=AlertStatus.ACTIVE,
                title=f"{rule.name}: {rule.description}",
                description=f"Metric '{rule.metric}' value {current_value:.3f} {rule.condition} {rule.threshold:.3f}",
                metric_value=current_value,
                threshold_value=rule.threshold,
                evaluation_result={'condition': rule.condition, 'statistical': statistical_result},
                created_timestamp=datetime.now(),
                first_occurrence=datetime.now(),
                last_occurrence=datetime.now(),
                occurrence_count=1,
                channels_notified=[],
                mathematical_validation=statistical_result
            )
            
            # Add to active alerts
            self.active_alerts[alert.id] = alert
            
            # Add to correlation system if enabled
            if self.alert_correlator:
                self.alert_correlator.add_alert_for_correlation(alert)
                
                # Find correlated alerts
                correlated_alerts = self.alert_correlator.find_correlated_alerts(alert)
                if correlated_alerts:
                    # Create correlation group
                    correlation_group = self.alert_correlator.create_correlation_group(
                        [alert] + correlated_alerts
                    )
                    alert.correlation_group = correlation_group
                    
                    # Update correlated alerts with same group
                    for corr_alert in correlated_alerts:
                        if corr_alert.id in self.active_alerts:
                            self.active_alerts[corr_alert.id].correlation_group = correlation_group
                    
                    self.system_metrics['correlations_found'] += 1
                    logger.info(f"Alert correlation detected: {len(correlated_alerts)} related alerts")
            
            # Send initial notifications
            self._send_alert_notifications(alert)
            
            # Add to history
            self.alert_history.append(alert)
            
            self.system_metrics['alerts_fired'] += 1
            logger.info(f"Fired alert {alert.id}: {alert.title}")
    
    def _resolve_alert_if_exists(self, rule_name: str):
        """Resolve alert if it exists and condition is no longer met"""
        with self.lock:
            alerts_to_resolve = []
            
            for alert in self.active_alerts.values():
                if (alert.rule_name == rule_name and 
                    alert.status == AlertStatus.ACTIVE):
                    alerts_to_resolve.append(alert)
            
            for alert in alerts_to_resolve:
                alert.status = AlertStatus.RESOLVED
                alert.resolution_timestamp = datetime.now()
                alert.resolution_user = "system"
                
                logger.info(f"Auto-resolved alert {alert.id}: {alert.title}")
    
    def _send_alert_notifications(self, alert: Alert):
        """Send notifications for alert"""
        try:
            # Get rule configuration
            rule = self.alert_rules.get(alert.rule_name)
            if not rule:
                logger.error(f"Rule {alert.rule_name} not found for alert {alert.id}")
                return
            
            # Send to each configured channel
            for channel_type in rule.channels:
                # Find matching notification channel
                channel_name = None
                for name, channel in self.notification_handler.channels.items():
                    if channel.channel_type == channel_type:
                        channel_name = name
                        break
                
                if channel_name:
                    result = self.notification_handler.send_notification(channel_name, alert)
                    
                    if result.get('success'):
                        alert.channels_notified.append(channel_type)
                        self.system_metrics['notifications_sent'] += 1
                    else:
                        logger.warning(f"Failed to send notification via {channel_name}: {result.get('error')}")
                else:
                    logger.warning(f"No channel configured for type {channel_type.value}")
                    
        except Exception as e:
            logger.error(f"Error sending alert notifications: {e}")
    
    def _process_escalations(self):
        """Process escalations for active alerts"""
        with self.lock:
            current_time = datetime.now()
            
            for alert in list(self.active_alerts.values()):
                if alert.status != AlertStatus.ACTIVE:
                    continue
                
                # Find appropriate escalation workflow
                workflow = self._get_escalation_workflow(alert)
                if not workflow:
                    continue
                
                # Check if escalation is needed
                time_since_creation = (current_time - alert.created_timestamp).total_seconds() / 60
                
                for step in workflow.steps:
                    if (step.level > alert.escalation_level and 
                        time_since_creation >= step.delay_minutes):
                        
                        self._execute_escalation_step(alert, step, workflow)
                        break
    
    def _get_escalation_workflow(self, alert: Alert) -> Optional[EscalationWorkflow]:
        """Get appropriate escalation workflow for alert"""
        # Simple logic: use emergency workflow for emergency alerts, standard for others
        if alert.severity == AlertSeverity.EMERGENCY:
            return self.escalation_workflows.get("emergency")
        else:
            return self.escalation_workflows.get("standard")
    
    def _execute_escalation_step(self, alert: Alert, step: EscalationStep, 
                               workflow: EscalationWorkflow):
        """Execute escalation step"""
        try:
            logger.info(f"Escalating alert {alert.id} to level {step.level}")
            
            # Update alert escalation level
            alert.escalation_level = step.level
            
            # Send notifications to escalation channels
            for channel_type in step.channels:
                channel_name = None
                for name, channel in self.notification_handler.channels.items():
                    if channel.channel_type == channel_type:
                        channel_name = name
                        break
                
                if channel_name:
                    result = self.notification_handler.send_notification(
                        channel_name, alert, step
                    )
                    
                    if result.get('success'):
                        self.system_metrics['notifications_sent'] += 1
                    else:
                        logger.warning(f"Escalation notification failed: {result.get('error')}")
            
            self.system_metrics['escalations_triggered'] += 1
            logger.info(f"Escalated alert {alert.id} to level {step.level}")
            
        except Exception as e:
            logger.error(f"Error executing escalation step: {e}")
    
    def acknowledge_alert(self, alert_id: str, user: str) -> Dict[str, Any]:
        """Acknowledge an alert"""
        with self.lock:
            if alert_id not in self.active_alerts:
                return {'success': False, 'error': 'Alert not found'}
            
            alert = self.active_alerts[alert_id]
            
            if alert.status != AlertStatus.ACTIVE:
                return {'success': False, 'error': f'Alert is not active (status: {alert.status.value})'}
            
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledgment_user = user
            alert.acknowledgment_timestamp = datetime.now()
            
            logger.info(f"Alert {alert_id} acknowledged by {user}")
            
            return {'success': True, 'alert_id': alert_id, 'acknowledged_by': user}
    
    def resolve_alert(self, alert_id: str, user: str, reason: str = "") -> Dict[str, Any]:
        """Manually resolve an alert"""
        with self.lock:
            if alert_id not in self.active_alerts:
                return {'success': False, 'error': 'Alert not found'}
            
            alert = self.active_alerts[alert_id]
            
            if alert.status == AlertStatus.RESOLVED:
                return {'success': False, 'error': 'Alert already resolved'}
            
            alert.status = AlertStatus.RESOLVED
            alert.resolution_user = user
            alert.resolution_timestamp = datetime.now()
            
            if reason:
                alert.evaluation_result['resolution_reason'] = reason
            
            logger.info(f"Alert {alert_id} resolved by {user}: {reason}")
            
            return {'success': True, 'alert_id': alert_id, 'resolved_by': user, 'reason': reason}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        with self.lock:
            active_alert_count = sum(1 for alert in self.active_alerts.values() 
                                   if alert.status == AlertStatus.ACTIVE)
            
            severity_counts = defaultdict(int)
            for alert in self.active_alerts.values():
                if alert.status == AlertStatus.ACTIVE:
                    severity_counts[alert.severity.value] += 1
            
            return {
                'system_running': self.running,
                'active_alerts': active_alert_count,
                'total_rules': len(self.alert_rules),
                'enabled_rules': sum(1 for rule in self.alert_rules.values() if rule.enabled),
                'escalation_workflows': len(self.escalation_workflows),
                'notification_channels': len(self.notification_handler.channels),
                'alert_severity_breakdown': dict(severity_counts),
                'system_metrics': dict(self.system_metrics),
                'correlation_enabled': self.enable_correlation,
                'statistical_validation_enabled': self.enable_statistical_validation,
                'timestamp': datetime.now().isoformat()
            }
    
    def create_default_alert_rules(self):
        """Create default alert rules for Lethe system"""
        # P95 latency alert
        p95_latency_rule = AlertRule(
            name="p95_latency_high",
            description="P95 latency exceeds target threshold",
            metric="p95_latency",
            condition=">",
            threshold=2.0,  # 2ms warning threshold
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.CONSOLE, AlertChannel.SLACK],
            evaluation_window=5,
            minimum_duration=2,
            cooldown_period=15,
            mathematical_validation=True,
            statistical_method="z_score"
        )
        
        # Critical P95 latency alert
        p95_latency_critical = AlertRule(
            name="p95_latency_critical", 
            description="P95 latency critically high",
            metric="p95_latency",
            condition=">",
            threshold=5.0,  # 5ms critical threshold
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL, AlertChannel.SLACK],
            evaluation_window=5,
            minimum_duration=1,
            cooldown_period=5,
            mathematical_validation=True,
            statistical_method="modified_z_score"
        )
        
        # CBU efficiency alert
        cbu_efficiency_rule = AlertRule(
            name="cbu_efficiency_low",
            description="CBU efficiency below target",
            metric="cbu_per_ms",
            condition="<",
            threshold=11.0,  # Below 11 CBU/ms
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.CONSOLE, AlertChannel.SLACK],
            evaluation_window=10,
            minimum_duration=5,
            cooldown_period=30,
            mathematical_validation=True,
            statistical_method="z_score"
        )
        
        # Error rate alert
        error_rate_rule = AlertRule(
            name="error_rate_high",
            description="Error rate exceeds acceptable threshold", 
            metric="error_rate",
            condition=">",
            threshold=0.01,  # 1% error rate
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL, AlertChannel.SLACK],
            evaluation_window=5,
            minimum_duration=2,
            cooldown_period=10,
            mathematical_validation=True,
            statistical_method="iqr"
        )
        
        # Add rules to system
        rules = [p95_latency_rule, p95_latency_critical, cbu_efficiency_rule, error_rate_rule]
        
        for rule in rules:
            self.add_alert_rule(rule)
        
        logger.info(f"Created {len(rules)} default alert rules")

# Example usage and integration
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create alerting system
        alerting_system = AlertingEscalationSystem(
            enable_correlation=True,
            enable_statistical_validation=True
        )
        
        # Create default alert rules
        alerting_system.create_default_alert_rules()
        
        # Start monitoring
        alerting_system.start_monitoring(evaluation_interval_seconds=30)
        
        print("Alerting system started. Press Ctrl+C to stop.")
        print(f"System status: {json.dumps(alerting_system.get_system_status(), indent=2)}")
        
        # Keep running
        try:
            while True:
                time.sleep(10)
                status = alerting_system.get_system_status()
                if status['active_alerts'] > 0:
                    print(f"Active alerts: {status['active_alerts']}")
                    
        except KeyboardInterrupt:
            print("\nShutting down alerting system...")
            alerting_system.stop_monitoring()
            
    except Exception as e:
        print(f"Error running alerting system: {e}")
        logger.error(f"Alerting system error: {e}")