use crate::types::*;
use chrono::Utc;
use dashmap::DashMap;
use std::{
    collections::HashMap,
    sync::{
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::{sync::RwLock, time::interval};
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Burn-in monitoring dashboard state and data aggregation
pub struct DashboardState {
    metrics_collector: Arc<MetricsCollector>,
    alert_manager: Arc<AlertManager>,
    health_checker: Arc<HealthChecker>,
    data_aggregator: Arc<DataAggregator>,
}

/// Collects and aggregates metrics from the determinism service
struct MetricsCollector {
    success_rate_history: RwLock<Vec<DataPoint>>,
    performance_history: RwLock<Vec<DataPoint>>,
    violation_history: RwLock<Vec<ViolationDataPoint>>,
    clock_skew_history: RwLock<Vec<ClockSkewDataPoint>>,
    collection_start: Instant,
}

/// Manages alerts based on system health and performance
struct AlertManager {
    active_alerts: DashMap<Uuid, Alert>,
    alert_rules: Vec<AlertRule>,
    suppression_rules: Vec<SuppressionRule>,
    notification_channels: Vec<NotificationChannel>,
}

/// Performs comprehensive health checks
struct HealthChecker {
    component_health: DashMap<String, ComponentHealth>,
    health_check_interval: Duration,
    degraded_threshold: f64,
    unhealthy_threshold: f64,
}

/// Aggregates data for dashboard visualization
struct DataAggregator {
    time_series_cache: RwLock<HashMap<String, Vec<DataPoint>>>,
    aggregation_window: Duration,
    max_data_points: usize,
}

#[derive(Debug, Clone)]
struct AlertRule {
    id: String,
    name: String,
    condition: AlertCondition,
    severity: ViolationSeverity,
    cooldown: Duration,
    last_fired: Option<Instant>,
}

#[derive(Debug, Clone)]
enum AlertCondition {
    DeterminismSuccessRateBelow(f64),
    PerformanceBudgetViolationAbove(f64),
    InvariantViolationRateAbove(f64),
    ClockSkewToleranceExceeded(u64),
    ComponentUnhealthy(String),
}

#[derive(Debug, Clone)]
struct SuppressionRule {
    alert_pattern: String,
    duration: Duration,
    reason: String,
}

#[derive(Debug, Clone)]
enum NotificationChannel {
    Webhook { url: String },
    Email { addresses: Vec<String> },
    Slack { webhook_url: String },
    Console,
}

impl DashboardState {
    pub fn new() -> Self {
        let alert_rules = vec![
            AlertRule {
                id: "determinism_success_rate_low".to_string(),
                name: "Determinism Success Rate Below Threshold".to_string(),
                condition: AlertCondition::DeterminismSuccessRateBelow(0.99),
                severity: ViolationSeverity::High,
                cooldown: Duration::from_secs(15 * 60),
                last_fired: None,
            },
            AlertRule {
                id: "performance_budget_violations".to_string(),
                name: "Performance Budget Violations High".to_string(),
                condition: AlertCondition::PerformanceBudgetViolationAbove(0.05),
                severity: ViolationSeverity::Medium,
                cooldown: Duration::from_secs(10 * 60),
                last_fired: None,
            },
            AlertRule {
                id: "invariant_violations_high".to_string(),
                name: "Invariant Violation Rate High".to_string(),
                condition: AlertCondition::InvariantViolationRateAbove(0.01),
                severity: ViolationSeverity::Critical,
                cooldown: Duration::from_secs(5 * 60),
                last_fired: None,
            },
            AlertRule {
                id: "clock_skew_exceeded".to_string(),
                name: "Clock Skew Tolerance Exceeded".to_string(),
                condition: AlertCondition::ClockSkewToleranceExceeded(1000), // 1 second
                severity: ViolationSeverity::High,
                cooldown: Duration::from_secs(20 * 60),
                last_fired: None,
            },
        ];

        Self {
            metrics_collector: Arc::new(MetricsCollector::new()),
            alert_manager: Arc::new(AlertManager::new(alert_rules)),
            health_checker: Arc::new(HealthChecker::new(
                Duration::from_secs(30), // Health check every 30s
                0.95,                    // Degraded below 95%
                0.90,                    // Unhealthy below 90%
            )),
            data_aggregator: Arc::new(DataAggregator::new(
                Duration::from_secs(5 * 60), // 5-minute aggregation windows
                1440,                      // Keep 1440 data points (5 days at 5-min intervals)
            )),
        }
    }

    /// Get comprehensive dashboard data
    pub async fn get_data(&self) -> DashboardData {
        let success_rates = self.metrics_collector.get_success_rate_data().await;
        let performance_metrics = self.metrics_collector.get_performance_data().await;
        let invariant_violations = self.metrics_collector.get_violation_data().await;
        let clock_skew_tests = self.metrics_collector.get_clock_skew_data().await;
        let system_health = self.health_checker.get_system_health().await;

        DashboardData {
            success_rates,
            performance_metrics,
            invariant_violations,
            clock_skew_tests,
            system_health,
        }
    }

    /// Record a determinism test result
    pub async fn record_determinism_result(&self, report: &DeterminismReport) {
        // Update success rate
        let success_rate = if report.determinism_check.is_deterministic { 1.0 } else { 0.0 };
        self.metrics_collector.record_success_rate(success_rate).await;

        // Update performance metrics
        let avg_p95 = (report.run1.performance_metrics.p95_latency_ms 
                      + report.run2.performance_metrics.p95_latency_ms) / 2.0;
        self.metrics_collector.record_performance_metric("p95_latency", avg_p95).await;

        // Record invariant violations
        if !report.invariant_report.all_passed {
            for violation in &report.invariant_report.violations {
                self.metrics_collector.record_violation(violation).await;
            }
        }

        // Check for alerts
        self.alert_manager.check_alerts(report).await;
        
        // Update component health
        self.health_checker.update_component_health("determinism_service", success_rate).await;
    }

    /// Record clock skew test results
    pub async fn record_clock_skew_results(&self, results: &[ClockSkewDataPoint]) {
        for result in results {
            self.metrics_collector.record_clock_skew_data(result.clone()).await;
            
            if !result.tolerance_met {
                self.alert_manager.check_clock_skew_alert(result).await;
            }
        }
    }

    /// Start background monitoring tasks
    pub async fn start_background_monitoring(&self) {
        let metrics_collector = self.metrics_collector.clone();
        let alert_manager = self.alert_manager.clone();
        let health_checker = self.health_checker.clone();
        let data_aggregator = self.data_aggregator.clone();

        // Start metrics collection task
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60)); // Every minute
            loop {
                interval.tick().await;
                
                if let Err(e) = metrics_collector.collect_system_metrics().await {
                    warn!("Failed to collect system metrics: {}", e);
                }
            }
        });

        // Start health check task
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30)); // Every 30 seconds
            loop {
                interval.tick().await;
                
                if let Err(e) = health_checker.perform_health_checks().await {
                    warn!("Failed to perform health checks: {}", e);
                }
            }
        });

        // Start data aggregation task
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(5 * 60)); // Every 5 minutes
            loop {
                interval.tick().await;
                
                if let Err(e) = data_aggregator.aggregate_data().await {
                    warn!("Failed to aggregate data: {}", e);
                }
            }
        });

        // Start alert processing task
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10)); // Every 10 seconds
            loop {
                interval.tick().await;
                
                if let Err(e) = alert_manager.process_alerts().await {
                    warn!("Failed to process alerts: {}", e);
                }
            }
        });

        info!("Background monitoring tasks started");
    }
}

impl MetricsCollector {
    fn new() -> Self {
        Self {
            success_rate_history: RwLock::new(Vec::new()),
            performance_history: RwLock::new(Vec::new()),
            violation_history: RwLock::new(Vec::new()),
            clock_skew_history: RwLock::new(Vec::new()),
            collection_start: Instant::now(),
        }
    }

    async fn record_success_rate(&self, rate: f64) {
        let mut history = self.success_rate_history.write().await;
        history.push(DataPoint {
            timestamp: Utc::now(),
            value: rate,
            label: "determinism_success_rate".to_string(),
        });
        
        // Keep only last 1000 points
        if history.len() > 1000 {
            let len = history.len();
            history.drain(0..len - 1000);
        }
    }

    async fn record_performance_metric(&self, metric_name: &str, value: f64) {
        let mut history = self.performance_history.write().await;
        history.push(DataPoint {
            timestamp: Utc::now(),
            value,
            label: metric_name.to_string(),
        });
        
        // Keep only last 1000 points
        if history.len() > 1000 {
            let len = history.len();
            history.drain(0..len - 1000);
        }
    }

    async fn record_violation(&self, violation: &InvariantViolation) {
        let mut history = self.violation_history.write().await;
        history.push(ViolationDataPoint {
            timestamp: violation.timestamp,
            violation_type: violation.invariant_type.clone(),
            severity: violation.severity.clone(),
            count: 1,
        });
        
        // Keep only last 1000 points
        if history.len() > 1000 {
            let len = history.len();
            history.drain(0..len - 1000);
        }
    }

    async fn record_clock_skew_data(&self, data: ClockSkewDataPoint) {
        let mut history = self.clock_skew_history.write().await;
        history.push(data);
        
        // Keep only last 1000 points
        if history.len() > 1000 {
            let len = history.len();
            history.drain(0..len - 1000);
        }
    }

    async fn get_success_rate_data(&self) -> Vec<DataPoint> {
        self.success_rate_history.read().await.clone()
    }

    async fn get_performance_data(&self) -> Vec<DataPoint> {
        self.performance_history.read().await.clone()
    }

    async fn get_violation_data(&self) -> Vec<ViolationDataPoint> {
        self.violation_history.read().await.clone()
    }

    async fn get_clock_skew_data(&self) -> Vec<ClockSkewDataPoint> {
        self.clock_skew_history.read().await.clone()
    }

    async fn collect_system_metrics(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // In a real implementation, this would collect actual system metrics
        // For now, we'll simulate some metrics
        
        let cpu_usage = self.get_cpu_usage();
        let memory_usage = self.get_memory_usage();
        let uptime = self.collection_start.elapsed().as_secs() as f64;
        
        self.record_performance_metric("cpu_usage", cpu_usage).await;
        self.record_performance_metric("memory_usage", memory_usage).await;
        self.record_performance_metric("uptime_seconds", uptime).await;
        
        debug!("System metrics collected: CPU: {:.1}%, Memory: {:.1}%, Uptime: {:.0}s", 
               cpu_usage, memory_usage, uptime);
        
        Ok(())
    }

    fn get_cpu_usage(&self) -> f64 {
        // Simulate CPU usage - in real implementation, read from /proc/stat or similar
        use rand::Rng;
        rand::thread_rng().gen_range(10.0..30.0)
    }

    fn get_memory_usage(&self) -> f64 {
        // Simulate memory usage - in real implementation, read from /proc/meminfo or similar
        use rand::Rng;
        rand::thread_rng().gen_range(40.0..70.0)
    }
}

impl AlertManager {
    fn new(rules: Vec<AlertRule>) -> Self {
        Self {
            active_alerts: DashMap::new(),
            alert_rules: rules,
            suppression_rules: Vec::new(),
            notification_channels: vec![NotificationChannel::Console],
        }
    }

    async fn check_alerts(&self, report: &DeterminismReport) {
        let success_rate = if report.determinism_check.is_deterministic { 1.0 } else { 0.0 };
        let performance_ratio = report.performance_budget_check.performance_ratio;
        let has_violations = !report.invariant_report.all_passed;

        for rule in &self.alert_rules {
            let should_fire = match &rule.condition {
                AlertCondition::DeterminismSuccessRateBelow(threshold) => success_rate < *threshold,
                AlertCondition::PerformanceBudgetViolationAbove(threshold) => performance_ratio > (1.0 + threshold),
                AlertCondition::InvariantViolationRateAbove(_threshold) => has_violations,
                _ => false,
            };

            if should_fire && self.should_fire_alert(rule).await {
                self.fire_alert(rule, &format!("Alert triggered by report: {}", report.slice_id)).await;
            }
        }
    }

    async fn check_clock_skew_alert(&self, data: &ClockSkewDataPoint) {
        for rule in &self.alert_rules {
            if let AlertCondition::ClockSkewToleranceExceeded(threshold) = &rule.condition {
                if data.skew_ms.abs() as u64 > *threshold && self.should_fire_alert(rule).await {
                    self.fire_alert(rule, &format!("Clock skew exceeded: {}ms", data.skew_ms)).await;
                }
            }
        }
    }

    async fn should_fire_alert(&self, rule: &AlertRule) -> bool {
        // Check cooldown period
        if let Some(last_fired) = rule.last_fired {
            if last_fired.elapsed() < rule.cooldown {
                return false;
            }
        }

        // Check suppression rules
        for suppression in &self.suppression_rules {
            if rule.id.contains(&suppression.alert_pattern) {
                debug!("Alert {} suppressed: {}", rule.id, suppression.reason);
                return false;
            }
        }

        true
    }

    async fn fire_alert(&self, rule: &AlertRule, message: &str) {
        let alert_id = Uuid::new_v4();
        let alert = Alert {
            id: alert_id,
            severity: rule.severity.clone(),
            message: format!("{}: {}", rule.name, message),
            timestamp: Utc::now(),
            acknowledged: false,
        };

        self.active_alerts.insert(alert_id, alert.clone());
        
        // Send notifications
        for channel in &self.notification_channels {
            if let Err(e) = self.send_notification(channel, &alert).await {
                warn!("Failed to send alert notification: {}", e);
            }
        }

        warn!("🚨 ALERT FIRED: {} - {}", rule.name, message);
    }

    async fn send_notification(&self, channel: &NotificationChannel, alert: &Alert) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match channel {
            NotificationChannel::Console => {
                println!("🚨 ALERT: [{:?}] {} at {}", alert.severity, alert.message, alert.timestamp);
            }
            NotificationChannel::Webhook { url } => {
                // In real implementation, send HTTP POST to webhook
                debug!("Would send webhook notification to: {}", url);
            }
            NotificationChannel::Email { addresses } => {
                // In real implementation, send email
                debug!("Would send email notification to: {:?}", addresses);
            }
            NotificationChannel::Slack { webhook_url } => {
                // In real implementation, send Slack message
                debug!("Would send Slack notification to: {}", webhook_url);
            }
        }
        Ok(())
    }

    async fn process_alerts(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Clean up old alerts
        let cutoff_time = Utc::now() - chrono::Duration::hours(24);
        
        self.active_alerts.retain(|_id, alert| alert.timestamp > cutoff_time);
        
        Ok(())
    }
}

impl HealthChecker {
    fn new(interval: Duration, degraded_threshold: f64, unhealthy_threshold: f64) -> Self {
        Self {
            component_health: DashMap::new(),
            health_check_interval: interval,
            degraded_threshold,
            unhealthy_threshold,
        }
    }

    async fn update_component_health(&self, component: &str, success_rate: f64) {
        let status = if success_rate >= self.degraded_threshold {
            ServiceHealth::Healthy
        } else if success_rate >= self.unhealthy_threshold {
            ServiceHealth::Degraded
        } else {
            ServiceHealth::Unhealthy
        };

        let health = ComponentHealth {
            status,
            last_check: Utc::now(),
            error_rate: 1.0 - success_rate,
        };

        self.component_health.insert(component.to_string(), health);
    }

    async fn perform_health_checks(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Check determinism service health
        self.check_determinism_service_health().await?;
        
        // Check database connectivity
        self.check_database_health().await?;
        
        // Check external dependencies
        self.check_external_dependencies().await?;
        
        Ok(())
    }

    async fn check_determinism_service_health(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // In real implementation, perform actual health checks
        let health = ComponentHealth {
            status: ServiceHealth::Healthy,
            last_check: Utc::now(),
            error_rate: 0.01,
        };
        
        self.component_health.insert("determinism_service".to_string(), health);
        Ok(())
    }

    async fn check_database_health(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // In real implementation, test database connectivity
        let health = ComponentHealth {
            status: ServiceHealth::Healthy,
            last_check: Utc::now(),
            error_rate: 0.0,
        };
        
        self.component_health.insert("database".to_string(), health);
        Ok(())
    }

    async fn check_external_dependencies(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // In real implementation, check external services
        Ok(())
    }

    async fn get_system_health(&self) -> SystemHealthData {
        let components: HashMap<String, ComponentHealth> = self.component_health.iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();

        let overall_status = if components.values().all(|h| matches!(h.status, ServiceHealth::Healthy)) {
            ServiceHealth::Healthy
        } else if components.values().any(|h| matches!(h.status, ServiceHealth::Unhealthy)) {
            ServiceHealth::Unhealthy
        } else {
            ServiceHealth::Degraded
        };

        SystemHealthData {
            overall_status,
            components,
            alerts: vec![], // Would be populated with active alerts
        }
    }
}

impl DataAggregator {
    fn new(window: Duration, max_points: usize) -> Self {
        Self {
            time_series_cache: RwLock::new(HashMap::new()),
            aggregation_window: window,
            max_data_points: max_points,
        }
    }

    async fn aggregate_data(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // In real implementation, aggregate raw metrics into time series data
        info!("Aggregating data for dashboard visualization");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_dashboard_state() {
        let dashboard = DashboardState::new();
        
        // Test getting initial data
        let data = dashboard.get_data().await;
        assert!(data.success_rates.is_empty());
        assert!(data.performance_metrics.is_empty());
    }

    #[tokio::test]
    async fn test_alert_manager() {
        let rules = vec![
            AlertRule {
                id: "test_rule".to_string(),
                name: "Test Rule".to_string(),
                condition: AlertCondition::DeterminismSuccessRateBelow(0.99),
                severity: ViolationSeverity::High,
                cooldown: Duration::from_secs(60),
                last_fired: None,
            }
        ];
        
        let alert_manager = AlertManager::new(rules);
        
        // Create a test report that should trigger the alert
        let report = DeterminismReport {
            slice_id: "test".to_string(),
            run_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            run1: ProcessingResult {
                slice_id: "test".to_string(),
                timestamp: Utc::now(),
                result_hash: "hash1".to_string(),
                performance_metrics: PerformanceMetrics {
                    duration_ms: 100,
                    memory_usage_mb: 64.0,
                    cpu_usage_percent: 15.0,
                    p95_latency_ms: 1.0,
                    throughput_ops_per_sec: 1000.0,
                },
                invariants: InvariantChecks {
                    monotone_timestamps: true,
                    causal_ordering: true,
                    data_consistency: true,
                    structural_integrity: true,
                },
                metadata: HashMap::new(),
            },
            run2: ProcessingResult {
                slice_id: "test".to_string(),
                timestamp: Utc::now(),
                result_hash: "hash2".to_string(), // Different hash = non-deterministic
                performance_metrics: PerformanceMetrics {
                    duration_ms: 100,
                    memory_usage_mb: 64.0,
                    cpu_usage_percent: 15.0,
                    p95_latency_ms: 1.0,
                    throughput_ops_per_sec: 1000.0,
                },
                invariants: InvariantChecks {
                    monotone_timestamps: true,
                    causal_ordering: true,
                    data_consistency: true,
                    structural_integrity: true,
                },
                metadata: HashMap::new(),
            },
            determinism_check: DeterminismCheck {
                is_deterministic: false,
                hash_match: false,
                timestamp_jitter_ms: 0,
                differences: vec!["Hash mismatch".to_string()],
                tolerance_met: true,
            },
            performance_budget_check: PerformanceBudgetCheck {
                budget_met: true,
                p95_latency_ms: 1.0,
                budget_threshold_ms: 2.0,
                performance_ratio: 0.5,
                sampling_rate: 1.0,
            },
            invariant_report: InvariantReport {
                all_passed: true,
                violations: vec![],
                score: 1.0,
            },
        };
        
        // This should trigger an alert due to low success rate
        alert_manager.check_alerts(&report).await;
        
        // Check that an alert was created
        assert!(!alert_manager.active_alerts.is_empty());
    }
}