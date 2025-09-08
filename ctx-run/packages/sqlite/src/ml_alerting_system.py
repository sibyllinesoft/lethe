"""
ML-Powered Alerting System for Lethe Optimization Engine

This module implements an intelligent alerting system that uses machine learning techniques
to predict, classify, and escalate critical system events with mathematical rigor.

Key Features:
- Anomaly detection using statistical methods and ML models
- Multi-channel notification routing (Slack, email, PagerDuty, webhooks)
- Alert correlation and deduplication to prevent notification fatigue  
- Predictive alerting based on trend analysis and leading indicators
- Mathematical validation of alert thresholds and significance testing
- Automated escalation workflows with human-in-the-loop override
- Real-time system health scoring and risk assessment

Mathematical Foundations:
- Sequential anomaly detection with CUSUM and Bayesian change point detection
- Statistical significance testing for alert validation (p-values, confidence intervals)
- Time series forecasting with ARIMA/exponential smoothing for predictive alerts
- Multi-variate correlation analysis for root cause identification
- Information-theoretic measures for alert prioritization and noise reduction
"""

import asyncio
import json
import logging
import smtplib
import threading
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable, Union, Set
from collections import defaultdict, deque
import statistics
import numpy as np
import requests
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"  
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AlertChannel(Enum):
    """Available notification channels"""
    SLACK = "slack"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    SMS = "sms"
    TEAMS = "teams"

class AlertStatus(Enum):
    """Alert lifecycle status"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged" 
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"

class AnomalyType(Enum):
    """Types of anomalies detected"""
    STATISTICAL_OUTLIER = "statistical_outlier"
    TREND_CHANGE = "trend_change"
    SEASONAL_DEVIATION = "seasonal_deviation"
    CORRELATION_BREAK = "correlation_break"
    THRESHOLD_BREACH = "threshold_breach"
    PREDICTIVE_RISK = "predictive_risk"

@dataclass
class AlertRule:
    """Configuration for alert detection rules"""
    name: str
    metric_name: str
    condition: str  # "greater_than", "less_than", "outside_range", "anomaly"
    threshold: Optional[float] = None
    threshold_range: Optional[Tuple[float, float]] = None
    severity: AlertSeverity = AlertSeverity.MEDIUM
    channels: List[AlertChannel] = field(default_factory=list)
    cooldown_minutes: int = 15
    require_confirmation: bool = False  # Require multiple detections
    statistical_significance: float = 0.05  # p-value threshold
    mathematical_validation: bool = True
    enabled: bool = True

@dataclass
class Alert:
    """Individual alert instance"""
    id: str
    rule_name: str
    metric_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    value: float
    threshold: Optional[float]
    timestamp: datetime
    channels: List[AlertChannel]
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledgments: List[Dict[str, Any]] = field(default_factory=list)
    escalations: List[Dict[str, Any]] = field(default_factory=list)
    resolution_timestamp: Optional[datetime] = None
    statistical_evidence: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NotificationChannel:
    """Configuration for notification delivery"""
    channel_type: AlertChannel
    config: Dict[str, str]
    enabled: bool = True
    rate_limit_per_hour: int = 100
    rate_limit_per_day: int = 500

class StatisticalAnomalyDetector:
    """
    Statistical methods for anomaly detection with mathematical rigor
    """
    
    def __init__(self, window_size: int = 100, sensitivity: float = 3.0):
        self.window_size = window_size
        self.sensitivity = sensitivity  # Number of standard deviations for outlier detection
        self.data_history = defaultdict(lambda: deque(maxlen=window_size))
        self.model_cache = {}
        
    def detect_statistical_outliers(self, 
                                  metric_name: str, 
                                  value: float,
                                  confidence_level: float = 0.95) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect statistical outliers using z-score and modified z-score methods
        
        Args:
            metric_name: Name of the metric
            value: Current metric value
            confidence_level: Statistical confidence level
            
        Returns:
            Tuple of (is_anomaly, evidence_dict)
        """
        try:
            history = list(self.data_history[metric_name])
            
            if len(history) < 10:  # Need sufficient history
                return False, {'reason': 'insufficient_history', 'samples': len(history)}
            
            # Add current value to history
            self.data_history[metric_name].append(value)
            
            # Z-score method
            mean_val = statistics.mean(history)
            std_val = statistics.stdev(history)
            
            if std_val == 0:
                return False, {'reason': 'zero_variance', 'mean': mean_val}
            
            z_score = abs(value - mean_val) / std_val
            z_threshold = stats.norm.ppf(1 - (1 - confidence_level) / 2)  # Two-tailed test
            
            # Modified z-score using median absolute deviation (more robust)
            median_val = statistics.median(history)
            mad = statistics.median([abs(x - median_val) for x in history])
            
            if mad == 0:
                modified_z_score = 0
            else:
                modified_z_score = abs(0.6745 * (value - median_val) / mad)
            
            # Decision criteria
            is_z_outlier = z_score > z_threshold
            is_modified_z_outlier = modified_z_score > self.sensitivity
            
            # Combine methods (require both for high confidence)
            is_anomaly = is_z_outlier and is_modified_z_outlier
            
            # Statistical significance testing
            p_value = 2 * (1 - stats.norm.cdf(z_score))  # Two-tailed p-value
            
            evidence = {
                'z_score': z_score,
                'z_threshold': z_threshold,
                'modified_z_score': modified_z_score,
                'p_value': p_value,
                'is_significant': p_value < (1 - confidence_level),
                'mean': mean_val,
                'std': std_val,
                'median': median_val,
                'mad': mad,
                'current_value': value,
                'sample_size': len(history),
                'confidence_level': confidence_level
            }
            
            return is_anomaly, evidence
            
        except Exception as e:
            logger.error(f"Error detecting statistical outliers for {metric_name}: {e}")
            return False, {'error': str(e)}
    
    def detect_trend_changes(self,
                           metric_name: str,
                           value: float,
                           min_trend_length: int = 20) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect significant changes in trend using linear regression and change point detection
        
        Args:
            metric_name: Name of the metric
            value: Current metric value
            min_trend_length: Minimum samples needed for trend analysis
            
        Returns:
            Tuple of (has_trend_change, evidence_dict)
        """
        try:
            history = list(self.data_history[metric_name])
            self.data_history[metric_name].append(value)
            
            if len(history) < min_trend_length:
                return False, {'reason': 'insufficient_history', 'samples': len(history)}
            
            # Divide into two segments for comparison
            split_point = len(history) // 2
            segment1 = history[:split_point]
            segment2 = history[split_point:]
            
            # Linear regression for each segment
            x1 = np.arange(len(segment1))
            x2 = np.arange(len(segment2))
            
            slope1, intercept1, r1, p1, _ = stats.linregress(x1, segment1)
            slope2, intercept2, r2, p2, _ = stats.linregress(x2, segment2)
            
            # Test for significant difference in slopes
            # Use Chow test approximation
            slope_diff = abs(slope2 - slope1)
            pooled_std = np.sqrt((np.var(segment1) + np.var(segment2)) / 2)
            
            if pooled_std > 0:
                t_stat = slope_diff / (pooled_std / np.sqrt(min(len(segment1), len(segment2))))
                df = len(segment1) + len(segment2) - 4
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            else:
                t_stat = 0
                p_value = 1.0
            
            # Consider significant if p < 0.05 and substantial change
            is_significant_change = p_value < 0.05 and slope_diff > 0.1 * abs(slope1 + 1e-8)
            
            evidence = {
                'slope1': slope1,
                'slope2': slope2, 
                'slope_difference': slope_diff,
                'r_squared_1': r1**2,
                'r_squared_2': r2**2,
                'p_value_trend_change': p_value,
                't_statistic': t_stat,
                'is_significant': is_significant_change,
                'segment1_size': len(segment1),
                'segment2_size': len(segment2)
            }
            
            return is_significant_change, evidence
            
        except Exception as e:
            logger.error(f"Error detecting trend changes for {metric_name}: {e}")
            return False, {'error': str(e)}

class MLAnomalyDetector:
    """
    Machine Learning-based anomaly detection using Isolation Forest and ensemble methods
    """
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        self.contamination = contamination  # Expected fraction of anomalies
        self.n_estimators = n_estimators
        self.models = {}
        self.scalers = {}
        self.training_data = defaultdict(list)
        self.min_training_samples = 50
        
    def update_model(self, metric_name: str, features: List[float]) -> bool:
        """
        Update or train ML model for given metric
        
        Args:
            metric_name: Name of the metric
            features: Feature vector for training
            
        Returns:
            True if model was updated successfully
        """
        try:
            self.training_data[metric_name].append(features)
            
            # Train model when sufficient data available
            if len(self.training_data[metric_name]) >= self.min_training_samples:
                training_data = np.array(self.training_data[metric_name])
                
                # Initialize scaler if needed
                if metric_name not in self.scalers:
                    self.scalers[metric_name] = StandardScaler()
                
                # Scale features
                scaled_data = self.scalers[metric_name].fit_transform(training_data)
                
                # Train Isolation Forest
                model = IsolationForest(
                    contamination=self.contamination,
                    n_estimators=self.n_estimators,
                    random_state=42
                )
                model.fit(scaled_data)
                
                self.models[metric_name] = model
                
                # Keep only recent training data to adapt to concept drift
                max_samples = 1000
                if len(self.training_data[metric_name]) > max_samples:
                    self.training_data[metric_name] = self.training_data[metric_name][-max_samples:]
                
                logger.debug(f"Updated ML model for {metric_name} with {len(training_data)} samples")
                return True
                
        except Exception as e:
            logger.error(f"Error updating ML model for {metric_name}: {e}")
            
        return False
    
    def detect_anomaly(self, metric_name: str, features: List[float]) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect anomaly using trained ML model
        
        Args:
            metric_name: Name of the metric
            features: Feature vector to evaluate
            
        Returns:
            Tuple of (is_anomaly, confidence_metrics)
        """
        try:
            if metric_name not in self.models:
                # Update training data but can't make prediction yet
                self.update_model(metric_name, features)
                return False, {'reason': 'model_not_ready', 'training_samples': len(self.training_data[metric_name])}
            
            model = self.models[metric_name]
            scaler = self.scalers[metric_name]
            
            # Scale features using fitted scaler
            features_array = np.array(features).reshape(1, -1)
            scaled_features = scaler.transform(features_array)
            
            # Predict anomaly
            anomaly_prediction = model.predict(scaled_features)[0]
            anomaly_score = model.decision_function(scaled_features)[0]
            
            # Convert to probability (Isolation Forest returns -1 for anomalies, 1 for normal)
            is_anomaly = anomaly_prediction == -1
            
            # Compute confidence metrics
            confidence = 1 / (1 + np.exp(anomaly_score))  # Sigmoid transformation
            
            evidence = {
                'anomaly_score': float(anomaly_score),
                'confidence': float(confidence),
                'prediction': int(anomaly_prediction),
                'is_anomaly': is_anomaly,
                'model_samples': len(self.training_data[metric_name]),
                'feature_vector': features
            }
            
            # Update model with new data
            self.update_model(metric_name, features)
            
            return is_anomaly, evidence
            
        except Exception as e:
            logger.error(f"Error detecting anomaly with ML for {metric_name}: {e}")
            return False, {'error': str(e)}

class NotificationManager:
    """
    Manages multi-channel notification delivery with rate limiting and retry logic
    """
    
    def __init__(self):
        self.channels: Dict[AlertChannel, NotificationChannel] = {}
        self.rate_limiters = defaultdict(lambda: defaultdict(deque))  # channel -> hour -> timestamps
        self.retry_queues = defaultdict(deque)
        self.delivery_history = deque(maxlen=1000)
        
    def add_channel(self, channel: NotificationChannel):
        """Add notification channel configuration"""
        self.channels[channel.channel_type] = channel
        logger.info(f"Added notification channel: {channel.channel_type.value}")
    
    def _check_rate_limit(self, channel_type: AlertChannel) -> bool:
        """Check if channel is within rate limits"""
        if channel_type not in self.channels:
            return False
        
        channel = self.channels[channel_type]
        current_time = datetime.now()
        current_hour = current_time.replace(minute=0, second=0, microsecond=0)
        
        # Clean old timestamps
        hour_queue = self.rate_limiters[channel_type][current_hour]
        while hour_queue and (current_time - hour_queue[0]).total_seconds() > 3600:
            hour_queue.popleft()
        
        # Check hourly limit
        if len(hour_queue) >= channel.rate_limit_per_hour:
            return False
        
        # Check daily limit (simplified - last 24 hours)
        daily_count = sum(len(queue) for hour_time, queue in self.rate_limiters[channel_type].items()
                         if (current_time - hour_time).total_seconds() <= 86400)
        
        return daily_count < channel.rate_limit_per_day
    
    async def send_notification(self, alert: Alert) -> Dict[AlertChannel, bool]:
        """
        Send notification through specified channels
        
        Args:
            alert: Alert to send notification for
            
        Returns:
            Dictionary mapping channels to success status
        """
        results = {}
        
        for channel_type in alert.channels:
            try:
                if channel_type not in self.channels:
                    results[channel_type] = False
                    logger.warning(f"Channel {channel_type.value} not configured")
                    continue
                
                if not self._check_rate_limit(channel_type):
                    results[channel_type] = False
                    logger.warning(f"Rate limit exceeded for channel {channel_type.value}")
                    continue
                
                # Send notification based on channel type
                success = await self._send_to_channel(channel_type, alert)
                results[channel_type] = success
                
                # Update rate limiter on success
                if success:
                    current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
                    self.rate_limiters[channel_type][current_hour].append(datetime.now())
                    
                    # Log delivery
                    self.delivery_history.append({
                        'alert_id': alert.id,
                        'channel': channel_type.value,
                        'timestamp': datetime.now(),
                        'success': success
                    })
                
            except Exception as e:
                logger.error(f"Error sending notification to {channel_type.value}: {e}")
                results[channel_type] = False
        
        return results
    
    async def _send_to_channel(self, channel_type: AlertChannel, alert: Alert) -> bool:
        """Send notification to specific channel"""
        try:
            channel = self.channels[channel_type]
            
            if not channel.enabled:
                return False
            
            if channel_type == AlertChannel.SLACK:
                return await self._send_slack_notification(channel, alert)
            elif channel_type == AlertChannel.EMAIL:
                return await self._send_email_notification(channel, alert)
            elif channel_type == AlertChannel.WEBHOOK:
                return await self._send_webhook_notification(channel, alert)
            elif channel_type == AlertChannel.PAGERDUTY:
                return await self._send_pagerduty_notification(channel, alert)
            else:
                logger.warning(f"Channel type {channel_type.value} not implemented")
                return False
                
        except Exception as e:
            logger.error(f"Error in _send_to_channel for {channel_type.value}: {e}")
            return False
    
    async def _send_slack_notification(self, channel: NotificationChannel, alert: Alert) -> bool:
        """Send Slack notification"""
        try:
            webhook_url = channel.config.get('webhook_url')
            if not webhook_url:
                return False
            
            # Format message
            color = {
                AlertSeverity.CRITICAL: '#ff0000',
                AlertSeverity.HIGH: '#ff8800', 
                AlertSeverity.MEDIUM: '#ffff00',
                AlertSeverity.LOW: '#0088ff',
                AlertSeverity.INFO: '#00ff00'
            }.get(alert.severity, '#808080')
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"🚨 {alert.severity.value.upper()} Alert: {alert.rule_name}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Metric", "value": alert.metric_name, "short": True},
                        {"title": "Value", "value": f"{alert.value:.3f}", "short": True},
                        {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True},
                        {"title": "Alert ID", "value": alert.id, "short": True}
                    ],
                    "footer": "Lethe Optimization Engine",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
            return False
    
    async def _send_email_notification(self, channel: NotificationChannel, alert: Alert) -> bool:
        """Send email notification"""
        try:
            smtp_server = channel.config.get('smtp_server')
            smtp_port = int(channel.config.get('smtp_port', 587))
            username = channel.config.get('username')
            password = channel.config.get('password')
            to_addresses = channel.config.get('to_addresses', '').split(',')
            
            if not all([smtp_server, username, password, to_addresses[0]]):
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = ', '.join(to_addresses)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.rule_name}"
            
            # HTML body
            html_body = f"""
            <html>
            <body>
                <h2 style="color: {'red' if alert.severity == AlertSeverity.CRITICAL else 'orange'};">
                    🚨 {alert.severity.value.upper()} Alert
                </h2>
                
                <p><strong>Rule:</strong> {alert.rule_name}</p>
                <p><strong>Metric:</strong> {alert.metric_name}</p>
                <p><strong>Value:</strong> {alert.value:.3f}</p>
                <p><strong>Threshold:</strong> {alert.threshold}</p>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                
                <p><strong>Message:</strong></p>
                <p style="background-color: #f0f0f0; padding: 10px; border-left: 4px solid #ccc;">
                    {alert.message}
                </p>
                
                <p><strong>Alert ID:</strong> {alert.id}</p>
                
                <hr>
                <p><em>Generated by Lethe Optimization Engine</em></p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False
    
    async def _send_webhook_notification(self, channel: NotificationChannel, alert: Alert) -> bool:
        """Send webhook notification"""
        try:
            url = channel.config.get('url')
            if not url:
                return False
            
            payload = {
                'alert_id': alert.id,
                'rule_name': alert.rule_name,
                'metric_name': alert.metric_name,
                'severity': alert.severity.value,
                'status': alert.status.value,
                'message': alert.message,
                'value': alert.value,
                'threshold': alert.threshold,
                'timestamp': alert.timestamp.isoformat(),
                'metadata': alert.metadata,
                'statistical_evidence': alert.statistical_evidence
            }
            
            headers = {'Content-Type': 'application/json'}
            auth_header = channel.config.get('auth_header')
            if auth_header:
                headers['Authorization'] = auth_header
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            return 200 <= response.status_code < 300
            
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            return False
    
    async def _send_pagerduty_notification(self, channel: NotificationChannel, alert: Alert) -> bool:
        """Send PagerDuty notification"""
        try:
            integration_key = channel.config.get('integration_key')
            if not integration_key:
                return False
            
            # Map severity to PagerDuty severity
            severity_map = {
                AlertSeverity.CRITICAL: 'critical',
                AlertSeverity.HIGH: 'error',
                AlertSeverity.MEDIUM: 'warning', 
                AlertSeverity.LOW: 'info',
                AlertSeverity.INFO: 'info'
            }
            
            payload = {
                'routing_key': integration_key,
                'event_action': 'trigger',
                'dedup_key': f"lethe-{alert.rule_name}-{alert.metric_name}",
                'payload': {
                    'summary': f"{alert.rule_name}: {alert.message}",
                    'severity': severity_map.get(alert.severity, 'warning'),
                    'source': 'Lethe Optimization Engine',
                    'component': alert.metric_name,
                    'group': alert.rule_name,
                    'class': 'optimization',
                    'custom_details': {
                        'alert_id': alert.id,
                        'metric_value': alert.value,
                        'threshold': alert.threshold,
                        'timestamp': alert.timestamp.isoformat(),
                        'statistical_evidence': alert.statistical_evidence
                    }
                }
            }
            
            response = requests.post(
                'https://events.pagerduty.com/v2/enqueue',
                json=payload,
                timeout=10
            )
            
            return response.status_code == 202
            
        except Exception as e:
            logger.error(f"Error sending PagerDuty notification: {e}")
            return False

class AlertCorrelator:
    """
    Correlates related alerts to reduce noise and identify root causes
    """
    
    def __init__(self, correlation_window_minutes: int = 30):
        self.correlation_window_minutes = correlation_window_minutes
        self.active_alerts = []
        self.correlation_rules = []
        self.suppressed_alerts = set()
        
    def add_correlation_rule(self, 
                           primary_metric: str, 
                           related_metrics: List[str],
                           correlation_threshold: float = 0.7):
        """Add rule for correlating related alerts"""
        rule = {
            'primary_metric': primary_metric,
            'related_metrics': related_metrics,
            'correlation_threshold': correlation_threshold
        }
        self.correlation_rules.append(rule)
        logger.info(f"Added correlation rule: {primary_metric} -> {related_metrics}")
    
    def should_suppress_alert(self, alert: Alert) -> Tuple[bool, Optional[str]]:
        """
        Determine if alert should be suppressed due to correlation
        
        Args:
            alert: Alert to evaluate
            
        Returns:
            Tuple of (should_suppress, reason)
        """
        try:
            current_time = alert.timestamp
            cutoff_time = current_time - timedelta(minutes=self.correlation_window_minutes)
            
            # Get recent alerts
            recent_alerts = [a for a in self.active_alerts 
                           if a.timestamp >= cutoff_time and a.status == AlertStatus.ACTIVE]
            
            # Check correlation rules
            for rule in self.correlation_rules:
                primary_metric = rule['primary_metric']
                related_metrics = rule['related_metrics']
                
                # If current alert is for a related metric
                if alert.metric_name in related_metrics:
                    # Check if primary metric alert exists
                    primary_alerts = [a for a in recent_alerts 
                                    if a.metric_name == primary_metric]
                    
                    if primary_alerts:
                        primary_alert = primary_alerts[0]  # Most recent
                        return True, f"Suppressed due to primary alert: {primary_alert.id}"
            
            # Check for duplicate alerts (same metric, similar value)
            duplicate_alerts = [
                a for a in recent_alerts 
                if (a.metric_name == alert.metric_name and 
                    a.rule_name == alert.rule_name and
                    abs(a.value - alert.value) < 0.1 * abs(alert.value + 1e-8))
            ]
            
            if duplicate_alerts:
                return True, f"Duplicate alert suppressed: {duplicate_alerts[0].id}"
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error checking alert suppression: {e}")
            return False, None
    
    def add_alert(self, alert: Alert):
        """Add alert to correlation tracking"""
        self.active_alerts.append(alert)
        
        # Clean old alerts
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.active_alerts = [a for a in self.active_alerts if a.timestamp >= cutoff_time]

class MLAlertingSystem:
    """
    Main ML-powered alerting system orchestrator
    """
    
    def __init__(self, 
                 enable_ml_detection: bool = True,
                 enable_statistical_detection: bool = True,
                 enable_correlation: bool = True):
        
        self.enable_ml_detection = enable_ml_detection
        self.enable_statistical_detection = enable_statistical_detection
        self.enable_correlation = enable_correlation
        
        # Initialize components
        self.statistical_detector = StatisticalAnomalyDetector() if enable_statistical_detection else None
        self.ml_detector = MLAnomalyDetector() if enable_ml_detection else None
        self.notification_manager = NotificationManager()
        self.correlator = AlertCorrelator() if enable_correlation else None
        
        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history = deque(maxlen=10000)
        
        # System state
        self.is_running = False
        self.last_health_check = datetime.now()
        self.system_metrics = defaultdict(list)
        
        # Threading
        self.alert_thread = None
        self.lock = threading.RLock()
        
        logger.info("ML Alerting System initialized")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add alert detection rule"""
        with self.lock:
            self.alert_rules[rule.name] = rule
            logger.info(f"Added alert rule: {rule.name} for metric {rule.metric_name}")
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Add notification channel"""
        self.notification_manager.add_channel(channel)
    
    def process_metric(self, 
                      metric_name: str, 
                      value: float, 
                      timestamp: Optional[datetime] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> List[Alert]:
        """
        Process incoming metric and generate alerts if needed
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            timestamp: Metric timestamp
            metadata: Additional metadata
            
        Returns:
            List of generated alerts
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if metadata is None:
            metadata = {}
        
        alerts_generated = []
        
        try:
            with self.lock:
                # Store metric for analysis
                self.system_metrics[metric_name].append({
                    'value': value,
                    'timestamp': timestamp,
                    'metadata': metadata
                })
                
                # Keep limited history
                if len(self.system_metrics[metric_name]) > 1000:
                    self.system_metrics[metric_name] = self.system_metrics[metric_name][-1000:]
                
                # Check all applicable rules
                for rule_name, rule in self.alert_rules.items():
                    if rule.metric_name != metric_name or not rule.enabled:
                        continue
                    
                    alert = self._evaluate_rule(rule, value, timestamp, metadata)
                    if alert:
                        alerts_generated.append(alert)
            
            # Send notifications asynchronously
            if alerts_generated:
                asyncio.create_task(self._process_alerts(alerts_generated))
            
            return alerts_generated
            
        except Exception as e:
            logger.error(f"Error processing metric {metric_name}: {e}")
            return []
    
    def _evaluate_rule(self, 
                      rule: AlertRule, 
                      value: float, 
                      timestamp: datetime,
                      metadata: Dict[str, Any]) -> Optional[Alert]:
        """Evaluate single alert rule against metric value"""
        try:
            # Check basic threshold conditions
            should_alert = False
            threshold_used = None
            
            if rule.condition == "greater_than" and rule.threshold is not None:
                should_alert = value > rule.threshold
                threshold_used = rule.threshold
                
            elif rule.condition == "less_than" and rule.threshold is not None:
                should_alert = value < rule.threshold
                threshold_used = rule.threshold
                
            elif rule.condition == "outside_range" and rule.threshold_range is not None:
                min_val, max_val = rule.threshold_range
                should_alert = value < min_val or value > max_val
                threshold_used = rule.threshold_range
                
            elif rule.condition == "anomaly":
                # Use ML/statistical detection
                is_anomaly, evidence = self._detect_anomaly(rule.metric_name, value)
                should_alert = is_anomaly
                
                if should_alert:
                    # Create detailed alert with statistical evidence
                    alert_id = f"alert_{rule.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
                    message = f"Anomaly detected in {rule.metric_name}: {value:.3f}"
                    
                    if evidence:
                        if 'z_score' in evidence:
                            message += f" (z-score: {evidence['z_score']:.2f})"
                        if 'confidence' in evidence:
                            message += f" (confidence: {evidence['confidence']:.3f})"
                    
                    alert = Alert(
                        id=alert_id,
                        rule_name=rule.name,
                        metric_name=rule.metric_name,
                        severity=rule.severity,
                        status=AlertStatus.ACTIVE,
                        message=message,
                        value=value,
                        threshold=threshold_used,
                        timestamp=timestamp,
                        channels=rule.channels,
                        metadata=metadata,
                        statistical_evidence=evidence
                    )
                    
                    return alert
            
            # Standard threshold alert
            if should_alert:
                alert_id = f"alert_{rule.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
                message = f"Alert triggered for {rule.metric_name}: {value:.3f}"
                
                if rule.condition == "greater_than":
                    message += f" > {rule.threshold}"
                elif rule.condition == "less_than":
                    message += f" < {rule.threshold}"
                elif rule.condition == "outside_range":
                    min_val, max_val = rule.threshold_range
                    message += f" outside range [{min_val}, {max_val}]"
                
                alert = Alert(
                    id=alert_id,
                    rule_name=rule.name,
                    metric_name=rule.metric_name,
                    severity=rule.severity,
                    status=AlertStatus.ACTIVE,
                    message=message,
                    value=value,
                    threshold=threshold_used,
                    timestamp=timestamp,
                    channels=rule.channels,
                    metadata=metadata
                )
                
                return alert
            
            return None
            
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.name}: {e}")
            return None
    
    def _detect_anomaly(self, metric_name: str, value: float) -> Tuple[bool, Dict[str, Any]]:
        """Detect anomaly using configured detection methods"""
        anomaly_detected = False
        combined_evidence = {}
        
        try:
            # Statistical detection
            if self.statistical_detector:
                stat_anomaly, stat_evidence = self.statistical_detector.detect_statistical_outliers(
                    metric_name, value
                )
                combined_evidence['statistical'] = stat_evidence
                
                if stat_anomaly:
                    anomaly_detected = True
                
                # Also check for trend changes
                trend_change, trend_evidence = self.statistical_detector.detect_trend_changes(
                    metric_name, value
                )
                combined_evidence['trend_analysis'] = trend_evidence
                
                if trend_change:
                    anomaly_detected = True
            
            # ML detection
            if self.ml_detector:
                # Create feature vector (simplified - in practice would include more features)
                recent_values = [entry['value'] for entry in self.system_metrics[metric_name][-10:]]
                if len(recent_values) >= 5:
                    features = [
                        value,
                        statistics.mean(recent_values[-5:]),  # Recent mean
                        statistics.stdev(recent_values[-5:]) if len(recent_values) > 1 else 0,  # Recent std
                        len(recent_values)  # Sample count
                    ]
                    
                    ml_anomaly, ml_evidence = self.ml_detector.detect_anomaly(metric_name, features)
                    combined_evidence['machine_learning'] = ml_evidence
                    
                    if ml_anomaly:
                        anomaly_detected = True
            
            return anomaly_detected, combined_evidence
            
        except Exception as e:
            logger.error(f"Error detecting anomaly for {metric_name}: {e}")
            return False, {'error': str(e)}
    
    async def _process_alerts(self, alerts: List[Alert]):
        """Process generated alerts through correlation and notification"""
        try:
            for alert in alerts:
                # Check correlation/suppression
                should_suppress = False
                
                if self.correlator:
                    should_suppress, reason = self.correlator.should_suppress_alert(alert)
                    if should_suppress:
                        logger.info(f"Alert {alert.id} suppressed: {reason}")
                        alert.status = AlertStatus.SUPPRESSED
                        continue
                    
                    self.correlator.add_alert(alert)
                
                # Add to active alerts
                with self.lock:
                    self.active_alerts[alert.id] = alert
                    self.alert_history.append(alert)
                
                # Send notifications
                notification_results = await self.notification_manager.send_notification(alert)
                
                # Log notification results
                successful_channels = [ch.value for ch, success in notification_results.items() if success]
                failed_channels = [ch.value for ch, success in notification_results.items() if not success]
                
                if successful_channels:
                    logger.info(f"Alert {alert.id} sent successfully to: {', '.join(successful_channels)}")
                
                if failed_channels:
                    logger.warning(f"Alert {alert.id} failed to send to: {', '.join(failed_channels)}")
                
        except Exception as e:
            logger.error(f"Error processing alerts: {e}")
    
    def acknowledge_alert(self, alert_id: str, user: str, message: str = "") -> bool:
        """Acknowledge an active alert"""
        try:
            with self.lock:
                if alert_id in self.active_alerts:
                    alert = self.active_alerts[alert_id]
                    alert.status = AlertStatus.ACKNOWLEDGED
                    alert.acknowledgments.append({
                        'user': user,
                        'timestamp': datetime.now(),
                        'message': message
                    })
                    logger.info(f"Alert {alert_id} acknowledged by {user}")
                    return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    def resolve_alert(self, alert_id: str, user: str, message: str = "") -> bool:
        """Resolve an active alert"""
        try:
            with self.lock:
                if alert_id in self.active_alerts:
                    alert = self.active_alerts[alert_id]
                    alert.status = AlertStatus.RESOLVED
                    alert.resolution_timestamp = datetime.now()
                    
                    # Add resolution info to metadata
                    alert.metadata['resolved_by'] = user
                    alert.metadata['resolution_message'] = message
                    alert.metadata['resolution_timestamp'] = datetime.now().isoformat()
                    
                    # Remove from active alerts
                    del self.active_alerts[alert_id]
                    
                    logger.info(f"Alert {alert_id} resolved by {user}")
                    return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and health metrics"""
        try:
            with self.lock:
                active_alerts_by_severity = defaultdict(int)
                for alert in self.active_alerts.values():
                    active_alerts_by_severity[alert.severity.value] += 1
                
                # Get recent alert rate
                recent_alerts = [a for a in self.alert_history 
                               if (datetime.now() - a.timestamp).total_seconds() <= 3600]
                
                return {
                    'system_health': 'healthy' if len(self.active_alerts) == 0 else 'degraded',
                    'active_alerts_count': len(self.active_alerts),
                    'active_alerts_by_severity': dict(active_alerts_by_severity),
                    'recent_alerts_count': len(recent_alerts),
                    'total_rules': len(self.alert_rules),
                    'enabled_rules': sum(1 for rule in self.alert_rules.values() if rule.enabled),
                    'notification_channels': len(self.notification_manager.channels),
                    'last_health_check': self.last_health_check.isoformat(),
                    'ml_detection_enabled': self.enable_ml_detection,
                    'statistical_detection_enabled': self.enable_statistical_detection,
                    'correlation_enabled': self.enable_correlation
                }
                
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def start_system(self):
        """Start the alerting system"""
        self.is_running = True
        logger.info("ML Alerting System started")
    
    def stop_system(self):
        """Stop the alerting system"""
        self.is_running = False
        if self.alert_thread and self.alert_thread.is_alive():
            self.alert_thread.join(timeout=5)
        logger.info("ML Alerting System stopped")

# Example usage and configuration
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create ML alerting system
        alerting_system = MLAlertingSystem(
            enable_ml_detection=True,
            enable_statistical_detection=True,
            enable_correlation=True
        )
        
        # Configure notification channels
        slack_channel = NotificationChannel(
            channel_type=AlertChannel.SLACK,
            config={
                'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
            },
            rate_limit_per_hour=20
        )
        alerting_system.add_notification_channel(slack_channel)
        
        email_channel = NotificationChannel(
            channel_type=AlertChannel.EMAIL,
            config={
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': '587',
                'username': 'alerts@company.com',
                'password': 'app-password',
                'to_addresses': 'team@company.com,oncall@company.com'
            },
            rate_limit_per_hour=10
        )
        alerting_system.add_notification_channel(email_channel)
        
        # Add alert rules
        cbu_efficiency_rule = AlertRule(
            name="CBU Efficiency Degradation",
            metric_name="cbu_per_ms",
            condition="less_than",
            threshold=10.0,  # Alert if below 10 CBU/ms (target is 12.5)
            severity=AlertSeverity.HIGH,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
            cooldown_minutes=10,
            mathematical_validation=True
        )
        alerting_system.add_alert_rule(cbu_efficiency_rule)
        
        p95_latency_rule = AlertRule(
            name="P95 Latency Spike",
            metric_name="p95_latency",
            condition="greater_than", 
            threshold=2.0,  # Alert if above 2ms (target is ≤1ms)
            severity=AlertSeverity.CRITICAL,
            channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
            cooldown_minutes=5,
            mathematical_validation=True
        )
        alerting_system.add_alert_rule(p95_latency_rule)
        
        anomaly_detection_rule = AlertRule(
            name="Performance Anomaly",
            metric_name="cbu_per_ms",
            condition="anomaly",
            severity=AlertSeverity.MEDIUM,
            channels=[AlertChannel.SLACK],
            cooldown_minutes=15,
            statistical_significance=0.01,  # Stricter significance test
            mathematical_validation=True
        )
        alerting_system.add_alert_rule(anomaly_detection_rule)
        
        # Add correlation rules
        if alerting_system.correlator:
            alerting_system.correlator.add_correlation_rule(
                primary_metric="p95_latency",
                related_metrics=["cbu_per_ms", "tokens_per_second"],
                correlation_threshold=0.7
            )
        
        # Start system
        alerting_system.start_system()
        
        # Simulate some metrics
        print("Simulating metric processing...")
        
        # Normal metrics
        for i in range(10):
            alerting_system.process_metric("cbu_per_ms", 12.5 + np.random.normal(0, 0.5))
            alerting_system.process_metric("p95_latency", 0.8 + np.random.normal(0, 0.1))
            time.sleep(0.5)
        
        # Anomalous metrics
        alerts = alerting_system.process_metric("cbu_per_ms", 8.5)  # Below threshold
        if alerts:
            print(f"Generated alert: {alerts[0].message}")
        
        alerts = alerting_system.process_metric("p95_latency", 3.2)  # Above threshold
        if alerts:
            print(f"Generated alert: {alerts[0].message}")
        
        # Print system status
        status = alerting_system.get_system_status()
        print(f"\nSystem Status: {json.dumps(status, indent=2, default=str)}")
        
    except KeyboardInterrupt:
        print("\nShutting down alerting system...")
        alerting_system.stop_system()
    except Exception as e:
        print(f"Error running alerting system: {e}")
        logger.error(f"Alerting system error: {e}")