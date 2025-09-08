/**
 * Monitoring Integration and Alerting System
 * Unified monitoring with intelligent alerting and escalation
 */

export interface AlertingConfig {
  alert_channels: {
    email: { enabled: boolean; recipients: string[]; rate_limit_minutes: number };
    slack: { enabled: boolean; webhook_url: string; channels: string[] };
    pagerduty: { enabled: boolean; integration_key: string; severity_mapping: Record<string, string> };
    webhook: { enabled: boolean; endpoints: string[] };
  };
  escalation_rules: Array<{
    condition: string;
    delay_minutes: number;
    action: 'EMAIL' | 'SLACK' | 'PAGERDUTY' | 'WEBHOOK';
    targets: string[];
  }>;
  alert_suppression: {
    duplicate_threshold_minutes: number;
    burst_protection_max_alerts: number;
    burst_protection_window_minutes: number;
  };
  quality_thresholds: {
    ece_critical: number;
    ece_warning: number;
    ilp_critical: number;
    ilp_warning: number;
    lambda_drift_warning: number;
    lambda_drift_critical: number;
    performance_degradation_warning: number;
    performance_degradation_critical: number;
  };
}

export interface Alert {
  id: string;
  timestamp: number;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' | 'EMERGENCY';
  source: string;
  title: string;
  description: string;
  metrics: Record<string, number>;
  tags: string[];
  status: 'ACTIVE' | 'ACKNOWLEDGED' | 'RESOLVED' | 'SUPPRESSED';
  escalation_level: number;
  acknowledged_by?: string;
  resolved_by?: string;
  resolution_time?: number;
  related_alerts: string[];
}

export interface AlertingRule {
  id: string;
  name: string;
  condition: string;
  severity: Alert['severity'];
  cooldown_minutes: number;
  enabled: boolean;
  last_triggered?: number;
  trigger_count: number;
}

export interface MonitoringDashboard {
  dashboard_id: string;
  name: string;
  panels: Array<{
    panel_id: string;
    title: string;
    type: 'METRIC' | 'GRAPH' | 'TABLE' | 'ALERT_LIST';
    data_source: string;
    config: Record<string, any>;
  }>;
  refresh_interval_seconds: number;
  time_range: {
    from: string;
    to: string;
  };
}

/**
 * Unified Metrics Collector
 * Collects metrics from all production validation systems
 */
export class UnifiedMetricsCollector {
  private metrics: Map<string, Array<{ timestamp: number; value: number; tags: Record<string, string> }>> = new Map();
  private metricsBuffer: Array<{ metric: string; timestamp: number; value: number; tags: Record<string, string> }> = [];

  /**
   * Collect metric from any production validation system
   */
  collectMetric(
    metricName: string,
    value: number,
    tags: Record<string, string> = {}
  ): void {
    const timestamp = Date.now();
    
    // Add to buffer for immediate processing
    this.metricsBuffer.push({
      metric: metricName,
      timestamp,
      value,
      tags
    });

    // Add to historical storage
    if (!this.metrics.has(metricName)) {
      this.metrics.set(metricName, []);
    }

    const metricHistory = this.metrics.get(metricName)!;
    metricHistory.push({ timestamp, value, tags });

    // Keep only last 1000 points per metric
    if (metricHistory.length > 1000) {
      metricHistory.splice(0, metricHistory.length - 500);
    }
  }

  /**
   * Batch collect metrics from validation systems
   */
  batchCollectFromValidationSystems(data: {
    coreProofs: { ece: number; ilp: number; statistical_power: number };
    monitoring: { lambda: number; cbu: number; size: number };
    canary: { performance_improvement: number; error_rate_delta: number };
    qualityGates: { overall_score: number; violations: number };
    healthChecks: { availability: number; response_time: number };
    riskBudget: { consumed_percentage: number; shadow_price: number };
    chaosTests: { success_rate: number; recovery_time: number };
  }): void {
    const timestamp = Date.now();
    const commonTags = { source: 'production_validation', timestamp: timestamp.toString() };

    // Core Proofs metrics
    this.collectMetric('validation.ece', data.coreProofs.ece, { ...commonTags, component: 'core_proofs' });
    this.collectMetric('validation.ilp', data.coreProofs.ilp, { ...commonTags, component: 'core_proofs' });
    this.collectMetric('validation.statistical_power', data.coreProofs.statistical_power, { ...commonTags, component: 'core_proofs' });

    // Monitoring metrics
    this.collectMetric('monitoring.lambda', data.monitoring.lambda, { ...commonTags, component: 'monitoring' });
    this.collectMetric('monitoring.cbu', data.monitoring.cbu, { ...commonTags, component: 'monitoring' });
    this.collectMetric('monitoring.size', data.monitoring.size, { ...commonTags, component: 'monitoring' });

    // Canary metrics
    this.collectMetric('canary.performance_improvement', data.canary.performance_improvement, { ...commonTags, component: 'canary' });
    this.collectMetric('canary.error_rate_delta', data.canary.error_rate_delta, { ...commonTags, component: 'canary' });

    // Quality Gates metrics
    this.collectMetric('quality_gates.overall_score', data.qualityGates.overall_score, { ...commonTags, component: 'quality_gates' });
    this.collectMetric('quality_gates.violations', data.qualityGates.violations, { ...commonTags, component: 'quality_gates' });

    // Health Check metrics
    this.collectMetric('health.availability_percentage', data.healthChecks.availability, { ...commonTags, component: 'health_checks' });
    this.collectMetric('health.avg_response_time_ms', data.healthChecks.response_time, { ...commonTags, component: 'health_checks' });

    // Risk Budget metrics
    this.collectMetric('risk_budget.consumed_percentage', data.riskBudget.consumed_percentage, { ...commonTags, component: 'risk_budget' });
    this.collectMetric('risk_budget.shadow_price', data.riskBudget.shadow_price, { ...commonTags, component: 'risk_budget' });

    // Chaos Testing metrics
    this.collectMetric('chaos.success_rate', data.chaosTests.success_rate, { ...commonTags, component: 'chaos_testing' });
    this.collectMetric('chaos.recovery_time_seconds', data.chaosTests.recovery_time, { ...commonTags, component: 'chaos_testing' });
  }

  /**
   * Get recent metrics for alerting evaluation
   */
  getRecentMetrics(metricName: string, lastMinutes: number = 5): Array<{ timestamp: number; value: number; tags: Record<string, string> }> {
    const cutoffTime = Date.now() - (lastMinutes * 60 * 1000);
    const metricHistory = this.metrics.get(metricName) || [];
    
    return metricHistory.filter(m => m.timestamp >= cutoffTime);
  }

  /**
   * Get all metric names
   */
  getMetricNames(): string[] {
    return Array.from(this.metrics.keys());
  }

  /**
   * Get aggregated statistics for a metric
   */
  getMetricStatistics(metricName: string, windowMinutes: number = 60): {
    count: number;
    min: number;
    max: number;
    avg: number;
    p95: number;
    latest: number;
  } | null {
    const recentMetrics = this.getRecentMetrics(metricName, windowMinutes);
    
    if (recentMetrics.length === 0) return null;

    const values = recentMetrics.map(m => m.value).sort((a, b) => a - b);
    const count = values.length;
    const min = values[0];
    const max = values[count - 1];
    const avg = values.reduce((sum, v) => sum + v, 0) / count;
    const p95Index = Math.floor(count * 0.95);
    const p95 = values[p95Index] || max;
    const latest = recentMetrics[recentMetrics.length - 1].value;

    return { count, min, max, avg, p95, latest };
  }

  /**
   * Flush metrics buffer (for external systems)
   */
  flushBuffer(): Array<{ metric: string; timestamp: number; value: number; tags: Record<string, string> }> {
    const buffer = [...this.metricsBuffer];
    this.metricsBuffer = [];
    return buffer;
  }
}

/**
 * Intelligent Alerting Engine
 * Evaluates metrics and generates contextual alerts
 */
export class IntelligentAlertingEngine {
  private config: AlertingConfig;
  private metricsCollector: UnifiedMetricsCollector;
  private alertingRules: AlertingRule[] = [];
  private activeAlerts: Alert[] = [];
  private alertHistory: Alert[] = [];
  private suppressedAlerts: Set<string> = new Set();

  constructor(config: AlertingConfig, metricsCollector: UnifiedMetricsCollector) {
    this.config = config;
    this.metricsCollector = metricsCollector;
    
    // Initialize default alerting rules
    this.initializeDefaultAlertingRules();
  }

  /**
   * Evaluate all alerting rules and generate alerts
   */
  async evaluateAlerts(): Promise<Alert[]> {
    const newAlerts: Alert[] = [];
    const now = Date.now();

    for (const rule of this.alertingRules) {
      if (!rule.enabled) continue;
      
      // Check cooldown
      if (rule.last_triggered && (now - rule.last_triggered) < (rule.cooldown_minutes * 60 * 1000)) {
        continue;
      }

      try {
        const alertTriggered = await this.evaluateRule(rule);
        
        if (alertTriggered) {
          const alert = await this.createAlertFromRule(rule);
          
          // Check for suppression
          if (!this.shouldSuppressAlert(alert)) {
            newAlerts.push(alert);
            rule.last_triggered = now;
            rule.trigger_count++;
          }
        }
      } catch (error) {
        console.error(`Error evaluating alerting rule ${rule.name}: ${error}`);
      }
    }

    // Add new alerts to active alerts
    for (const alert of newAlerts) {
      this.activeAlerts.push(alert);
      this.alertHistory.push({ ...alert });
    }

    // Cleanup old alerts
    this.cleanupOldAlerts();

    return newAlerts;
  }

  /**
   * Create alert for specific validation system failure
   */
  createValidationSystemAlert(
    source: string,
    title: string,
    description: string,
    severity: Alert['severity'],
    metrics: Record<string, number>
  ): Alert {
    const alert: Alert = {
      id: `${source}-${Date.now()}`,
      timestamp: Date.now(),
      severity,
      source,
      title,
      description,
      metrics,
      tags: [source, 'validation_system'],
      status: 'ACTIVE',
      escalation_level: 0,
      related_alerts: []
    };

    this.activeAlerts.push(alert);
    this.alertHistory.push({ ...alert });

    return alert;
  }

  /**
   * Acknowledge alert
   */
  acknowledgeAlert(alertId: string, acknowledgedBy: string): boolean {
    const alert = this.activeAlerts.find(a => a.id === alertId);
    
    if (alert && alert.status === 'ACTIVE') {
      alert.status = 'ACKNOWLEDGED';
      alert.acknowledged_by = acknowledgedBy;
      return true;
    }
    
    return false;
  }

  /**
   * Resolve alert
   */
  resolveAlert(alertId: string, resolvedBy: string): boolean {
    const alert = this.activeAlerts.find(a => a.id === alertId);
    
    if (alert && (alert.status === 'ACTIVE' || alert.status === 'ACKNOWLEDGED')) {
      alert.status = 'RESOLVED';
      alert.resolved_by = resolvedBy;
      alert.resolution_time = Date.now();
      return true;
    }
    
    return false;
  }

  /**
   * Get active alerts
   */
  getActiveAlerts(severity?: Alert['severity']): Alert[] {
    let alerts = this.activeAlerts.filter(a => a.status === 'ACTIVE');
    
    if (severity) {
      alerts = alerts.filter(a => a.severity === severity);
    }
    
    return alerts.sort((a, b) => {
      // Sort by severity first, then by timestamp
      const severityOrder = { EMERGENCY: 5, CRITICAL: 4, HIGH: 3, MEDIUM: 2, LOW: 1 };
      const severityDiff = severityOrder[b.severity] - severityOrder[a.severity];
      
      if (severityDiff !== 0) return severityDiff;
      return b.timestamp - a.timestamp;
    });
  }

  /**
   * Get alert statistics
   */
  getAlertStatistics(hoursBack: number = 24): {
    total_alerts: number;
    by_severity: Record<Alert['severity'], number>;
    by_source: Record<string, number>;
    resolution_stats: {
      avg_resolution_time_minutes: number;
      unresolved_count: number;
    };
  } {
    const cutoffTime = Date.now() - (hoursBack * 60 * 60 * 1000);
    const recentAlerts = this.alertHistory.filter(a => a.timestamp >= cutoffTime);

    const bySeverity = recentAlerts.reduce((acc, alert) => {
      acc[alert.severity] = (acc[alert.severity] || 0) + 1;
      return acc;
    }, {} as Record<Alert['severity'], number>);

    const bySource = recentAlerts.reduce((acc, alert) => {
      acc[alert.source] = (acc[alert.source] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const resolvedAlerts = recentAlerts.filter(a => a.status === 'RESOLVED' && a.resolution_time);
    const avgResolutionTime = resolvedAlerts.length > 0
      ? resolvedAlerts.reduce((sum, a) => {
          return sum + (a.resolution_time! - a.timestamp);
        }, 0) / resolvedAlerts.length / (60 * 1000) // Convert to minutes
      : 0;

    const unresolvedCount = recentAlerts.filter(a => a.status === 'ACTIVE').length;

    return {
      total_alerts: recentAlerts.length,
      by_severity: bySeverity,
      by_source: bySource,
      resolution_stats: {
        avg_resolution_time_minutes: avgResolutionTime,
        unresolved_count: unresolvedCount
      }
    };
  }

  private initializeDefaultAlertingRules(): void {
    const rules: Omit<AlertingRule, 'last_triggered' | 'trigger_count'>[] = [
      {
        id: 'ece_critical',
        name: 'Expected Calibration Error Critical',
        condition: 'validation.ece > 0.08',
        severity: 'CRITICAL',
        cooldown_minutes: 5,
        enabled: true
      },
      {
        id: 'ilp_critical',
        name: 'Information Leakage Percentage Critical',
        condition: 'validation.ilp > 0.05',
        severity: 'CRITICAL',
        cooldown_minutes: 5,
        enabled: true
      },
      {
        id: 'lambda_drift_critical',
        name: 'Lambda Drift Out of Bounds',
        condition: 'abs(monitoring.lambda - baseline) > drift_bounds',
        severity: 'HIGH',
        cooldown_minutes: 10,
        enabled: true
      },
      {
        id: 'performance_degradation',
        name: 'Performance Degradation Detected',
        condition: 'canary.performance_improvement < -10',
        severity: 'MEDIUM',
        cooldown_minutes: 15,
        enabled: true
      },
      {
        id: 'availability_critical',
        name: 'System Availability Critical',
        condition: 'health.availability_percentage < 95',
        severity: 'CRITICAL',
        cooldown_minutes: 2,
        enabled: true
      },
      {
        id: 'risk_budget_exceeded',
        name: 'Risk Budget Exceeded',
        condition: 'risk_budget.consumed_percentage > 90',
        severity: 'HIGH',
        cooldown_minutes: 30,
        enabled: true
      },
      {
        id: 'chaos_test_failure',
        name: 'Chaos Test Failure Rate High',
        condition: 'chaos.success_rate < 0.8',
        severity: 'MEDIUM',
        cooldown_minutes: 60,
        enabled: true
      }
    ];

    this.alertingRules = rules.map(rule => ({
      ...rule,
      trigger_count: 0
    }));
  }

  private async evaluateRule(rule: AlertingRule): Promise<boolean> {
    // Simple rule evaluation - in production, this would use a proper expression engine
    try {
      switch (rule.id) {
        case 'ece_critical':
          const eceStats = this.metricsCollector.getMetricStatistics('validation.ece', 5);
          return eceStats ? eceStats.latest > this.config.quality_thresholds.ece_critical : false;

        case 'ilp_critical':
          const ilpStats = this.metricsCollector.getMetricStatistics('validation.ilp', 5);
          return ilpStats ? ilpStats.latest > this.config.quality_thresholds.ilp_critical : false;

        case 'lambda_drift_critical':
          const lambdaStats = this.metricsCollector.getMetricStatistics('monitoring.lambda', 10);
          if (!lambdaStats) return false;
          
          // Simplified drift detection - would use actual bounds in production
          const baseline = 1.0; // Would be calculated from historical data
          const drift = Math.abs(lambdaStats.latest - baseline);
          return drift > this.config.quality_thresholds.lambda_drift_critical;

        case 'performance_degradation':
          const perfStats = this.metricsCollector.getMetricStatistics('canary.performance_improvement', 15);
          return perfStats ? perfStats.latest < this.config.quality_thresholds.performance_degradation_critical : false;

        case 'availability_critical':
          const availStats = this.metricsCollector.getMetricStatistics('health.availability_percentage', 5);
          return availStats ? availStats.latest < 95 : false;

        case 'risk_budget_exceeded':
          const budgetStats = this.metricsCollector.getMetricStatistics('risk_budget.consumed_percentage', 30);
          return budgetStats ? budgetStats.latest > 90 : false;

        case 'chaos_test_failure':
          const chaosStats = this.metricsCollector.getMetricStatistics('chaos.success_rate', 60);
          return chaosStats ? chaosStats.latest < 0.8 : false;

        default:
          return false;
      }
    } catch (error) {
      console.error(`Error evaluating rule ${rule.name}: ${error}`);
      return false;
    }
  }

  private async createAlertFromRule(rule: AlertingRule): Promise<Alert> {
    const metrics: Record<string, number> = {};
    
    // Gather relevant metrics for context
    const metricNames = this.metricsCollector.getMetricNames();
    for (const metricName of metricNames) {
      const stats = this.metricsCollector.getMetricStatistics(metricName, 5);
      if (stats) {
        metrics[metricName] = stats.latest;
      }
    }

    return {
      id: `${rule.id}-${Date.now()}`,
      timestamp: Date.now(),
      severity: rule.severity,
      source: 'alerting_engine',
      title: rule.name,
      description: `Alert triggered by rule: ${rule.condition}`,
      metrics,
      tags: ['automated', 'production_validation'],
      status: 'ACTIVE',
      escalation_level: 0,
      related_alerts: []
    };
  }

  private shouldSuppressAlert(alert: Alert): boolean {
    // Check for duplicate suppression
    const duplicateKey = `${alert.source}:${alert.title}`;
    const now = Date.now();
    
    // Check if we've seen this alert recently
    const recentSimilarAlerts = this.alertHistory.filter(a => {
      const alertKey = `${a.source}:${a.title}`;
      const timeDiff = now - a.timestamp;
      
      return alertKey === duplicateKey && 
             timeDiff < (this.config.alert_suppression.duplicate_threshold_minutes * 60 * 1000);
    });

    if (recentSimilarAlerts.length > 0) {
      return true;
    }

    // Check burst protection
    const recentAlerts = this.alertHistory.filter(a => {
      const timeDiff = now - a.timestamp;
      return timeDiff < (this.config.alert_suppression.burst_protection_window_minutes * 60 * 1000);
    });

    if (recentAlerts.length > this.config.alert_suppression.burst_protection_max_alerts) {
      return true;
    }

    return false;
  }

  private cleanupOldAlerts(): void {
    const cutoffTime = Date.now() - (7 * 24 * 60 * 60 * 1000); // 7 days
    
    // Remove old alerts from history
    this.alertHistory = this.alertHistory.filter(a => a.timestamp >= cutoffTime);
    
    // Remove resolved alerts from active list
    this.activeAlerts = this.activeAlerts.filter(a => 
      a.status === 'ACTIVE' || 
      a.status === 'ACKNOWLEDGED' ||
      (a.resolution_time && (Date.now() - a.resolution_time) < (24 * 60 * 60 * 1000))
    );
  }
}

/**
 * Alert Delivery Manager
 * Handles delivery of alerts through various channels
 */
export class AlertDeliveryManager {
  private config: AlertingConfig;

  constructor(config: AlertingConfig) {
    this.config = config;
  }

  /**
   * Deliver alert through configured channels
   */
  async deliverAlert(alert: Alert): Promise<{
    delivered: boolean;
    channels_used: string[];
    delivery_errors: string[];
  }> {
    const channelsUsed: string[] = [];
    const deliveryErrors: string[] = [];

    // Email delivery
    if (this.config.alert_channels.email.enabled) {
      try {
        await this.deliverEmailAlert(alert);
        channelsUsed.push('email');
      } catch (error) {
        deliveryErrors.push(`Email delivery failed: ${error}`);
      }
    }

    // Slack delivery
    if (this.config.alert_channels.slack.enabled) {
      try {
        await this.deliverSlackAlert(alert);
        channelsUsed.push('slack');
      } catch (error) {
        deliveryErrors.push(`Slack delivery failed: ${error}`);
      }
    }

    // PagerDuty delivery (for critical/emergency alerts)
    if (this.config.alert_channels.pagerduty.enabled && 
        (alert.severity === 'CRITICAL' || alert.severity === 'EMERGENCY')) {
      try {
        await this.deliverPagerDutyAlert(alert);
        channelsUsed.push('pagerduty');
      } catch (error) {
        deliveryErrors.push(`PagerDuty delivery failed: ${error}`);
      }
    }

    // Webhook delivery
    if (this.config.alert_channels.webhook.enabled) {
      try {
        await this.deliverWebhookAlert(alert);
        channelsUsed.push('webhook');
      } catch (error) {
        deliveryErrors.push(`Webhook delivery failed: ${error}`);
      }
    }

    return {
      delivered: channelsUsed.length > 0,
      channels_used: channelsUsed,
      delivery_errors: deliveryErrors
    };
  }

  private async deliverEmailAlert(alert: Alert): Promise<void> {
    // Simulate email delivery
    console.log(`📧 Email alert delivered: ${alert.title} (${alert.severity})`);
    await this.sleep(100);
  }

  private async deliverSlackAlert(alert: Alert): Promise<void> {
    // Simulate Slack delivery
    const emoji = this.getSeverityEmoji(alert.severity);
    console.log(`${emoji} Slack alert delivered: ${alert.title} (${alert.severity})`);
    await this.sleep(150);
  }

  private async deliverPagerDutyAlert(alert: Alert): Promise<void> {
    // Simulate PagerDuty delivery
    console.log(`📟 PagerDuty alert delivered: ${alert.title} (${alert.severity})`);
    await this.sleep(200);
  }

  private async deliverWebhookAlert(alert: Alert): Promise<void> {
    // Simulate webhook delivery
    console.log(`🔗 Webhook alert delivered: ${alert.title} (${alert.severity})`);
    await this.sleep(80);
  }

  private getSeverityEmoji(severity: Alert['severity']): string {
    switch (severity) {
      case 'EMERGENCY': return '🚨';
      case 'CRITICAL': return '🔴';
      case 'HIGH': return '🟠';
      case 'MEDIUM': return '🟡';
      case 'LOW': return '🟢';
      default: return '❓';
    }
  }

  private async sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Production Monitoring Dashboard Manager
 * Manages real-time dashboards for production validation
 */
export class ProductionDashboardManager {
  private dashboards: Map<string, MonitoringDashboard> = new Map();
  private metricsCollector: UnifiedMetricsCollector;

  constructor(metricsCollector: UnifiedMetricsCollector) {
    this.metricsCollector = metricsCollector;
    this.initializeDefaultDashboards();
  }

  /**
   * Get dashboard data for rendering
   */
  getDashboardData(dashboardId: string): {
    dashboard: MonitoringDashboard;
    panel_data: Record<string, any>;
  } | null {
    const dashboard = this.dashboards.get(dashboardId);
    if (!dashboard) return null;

    const panelData: Record<string, any> = {};

    for (const panel of dashboard.panels) {
      switch (panel.type) {
        case 'METRIC':
          panelData[panel.panel_id] = this.getMetricPanelData(panel);
          break;
        case 'GRAPH':
          panelData[panel.panel_id] = this.getGraphPanelData(panel);
          break;
        case 'TABLE':
          panelData[panel.panel_id] = this.getTablePanelData(panel);
          break;
        case 'ALERT_LIST':
          panelData[panel.panel_id] = this.getAlertListPanelData(panel);
          break;
      }
    }

    return { dashboard, panel_data: panelData };
  }

  /**
   * Get all available dashboards
   */
  getAvailableDashboards(): Array<{ id: string; name: string; description: string }> {
    return Array.from(this.dashboards.entries()).map(([id, dashboard]) => ({
      id,
      name: dashboard.name,
      description: `${dashboard.panels.length} panels`
    }));
  }

  private initializeDefaultDashboards(): void {
    // Production Validation Overview Dashboard
    const overviewDashboard: MonitoringDashboard = {
      dashboard_id: 'production_overview',
      name: 'Production Validation Overview',
      panels: [
        {
          panel_id: 'system_health',
          title: 'System Health Status',
          type: 'TABLE',
          data_source: 'health_metrics',
          config: {
            columns: ['Component', 'Status', 'Response Time', 'Last Check'],
            refresh_seconds: 30
          }
        },
        {
          panel_id: 'quality_metrics',
          title: 'Quality Gate Metrics',
          type: 'METRIC',
          data_source: 'quality_gates',
          config: {
            metrics: ['validation.ece', 'validation.ilp', 'quality_gates.overall_score'],
            thresholds: { ece: 0.08, ilp: 0.05, score: 0.8 }
          }
        },
        {
          panel_id: 'performance_trend',
          title: 'Performance Trends',
          type: 'GRAPH',
          data_source: 'performance_metrics',
          config: {
            metrics: ['canary.performance_improvement', 'health.avg_response_time_ms'],
            time_range: '1h',
            chart_type: 'line'
          }
        },
        {
          panel_id: 'active_alerts',
          title: 'Active Alerts',
          type: 'ALERT_LIST',
          data_source: 'alerts',
          config: {
            max_alerts: 10,
            filter_severity: ['CRITICAL', 'HIGH']
          }
        }
      ],
      refresh_interval_seconds: 30,
      time_range: { from: '-1h', to: 'now' }
    };

    // Lambda/CBU/Size Monitoring Dashboard
    const monitoringDashboard: MonitoringDashboard = {
      dashboard_id: 'lambda_cbu_monitoring',
      name: 'λ/CBU/Size Monitoring',
      panels: [
        {
          panel_id: 'lambda_trend',
          title: 'Lambda Parameter Trend',
          type: 'GRAPH',
          data_source: 'monitoring_metrics',
          config: {
            metrics: ['monitoring.lambda'],
            bounds: 'show_drift_bounds',
            time_range: '6h'
          }
        },
        {
          panel_id: 'cbu_metrics',
          title: 'CBU Utilization',
          type: 'METRIC',
          data_source: 'monitoring_metrics',
          config: {
            metrics: ['monitoring.cbu'],
            display_type: 'gauge',
            thresholds: { warning: 80, critical: 95 }
          }
        },
        {
          panel_id: 'size_distribution',
          title: 'Context Size Distribution',
          type: 'GRAPH',
          data_source: 'monitoring_metrics',
          config: {
            metrics: ['monitoring.size'],
            chart_type: 'histogram',
            time_range: '1h'
          }
        }
      ],
      refresh_interval_seconds: 15,
      time_range: { from: '-6h', to: 'now' }
    };

    // Canary Deployment Dashboard
    const canaryDashboard: MonitoringDashboard = {
      dashboard_id: 'canary_deployment',
      name: 'Canary Deployment Status',
      panels: [
        {
          panel_id: 'canary_status',
          title: 'Canary Status',
          type: 'METRIC',
          data_source: 'canary_metrics',
          config: {
            display_type: 'status_card',
            metrics: ['canary.performance_improvement', 'canary.error_rate_delta']
          }
        },
        {
          panel_id: 'promotion_criteria',
          title: 'Promotion Criteria Progress',
          type: 'TABLE',
          data_source: 'canary_metrics',
          config: {
            columns: ['Criteria', 'Current', 'Threshold', 'Status'],
            criteria: ['ΔCBU/GB', 'P95 Improvement', 'Statistical Power']
          }
        },
        {
          panel_id: 'traffic_split',
          title: 'Traffic Split Over Time',
          type: 'GRAPH',
          data_source: 'canary_metrics',
          config: {
            chart_type: 'area',
            time_range: '7d'
          }
        }
      ],
      refresh_interval_seconds: 60,
      time_range: { from: '-7d', to: 'now' }
    };

    this.dashboards.set('production_overview', overviewDashboard);
    this.dashboards.set('lambda_cbu_monitoring', monitoringDashboard);
    this.dashboards.set('canary_deployment', canaryDashboard);
  }

  private getMetricPanelData(panel: any): any {
    const data: any = { values: {}, thresholds: panel.config.thresholds || {} };
    
    for (const metric of panel.config.metrics || []) {
      const stats = this.metricsCollector.getMetricStatistics(metric);
      data.values[metric] = stats ? stats.latest : null;
    }
    
    return data;
  }

  private getGraphPanelData(panel: any): any {
    const timeRange = panel.config.time_range || '1h';
    const minutes = this.parseTimeRange(timeRange);
    
    const data: any = { series: [] };
    
    for (const metric of panel.config.metrics || []) {
      const points = this.metricsCollector.getRecentMetrics(metric, minutes);
      data.series.push({
        name: metric,
        data: points.map(p => ({ x: p.timestamp, y: p.value }))
      });
    }
    
    return data;
  }

  private getTablePanelData(panel: any): any {
    // Return mock table data - would be populated from actual sources in production
    return {
      columns: panel.config.columns || [],
      rows: [
        ['Core Proofs', 'HEALTHY', '45ms', '30s ago'],
        ['Quality Gates', 'WARNING', '67ms', '15s ago'],
        ['Monitoring', 'HEALTHY', '23ms', '10s ago']
      ]
    };
  }

  private getAlertListPanelData(panel: any): any {
    // Return mock alert data - would integrate with actual alerting system
    return {
      alerts: [
        { id: '1', severity: 'HIGH', title: 'Lambda drift detected', time: '2m ago' },
        { id: '2', severity: 'MEDIUM', title: 'Performance degradation', time: '5m ago' }
      ]
    };
  }

  private parseTimeRange(timeRange: string): number {
    const match = timeRange.match(/^(\d+)([hmd])$/);
    if (!match) return 60; // Default 1 hour
    
    const value = parseInt(match[1]);
    const unit = match[2];
    
    switch (unit) {
      case 'm': return value;
      case 'h': return value * 60;
      case 'd': return value * 24 * 60;
      default: return 60;
    }
  }
}

/**
 * Monitoring Integration Orchestrator
 * Coordinates all monitoring, alerting, and dashboard systems
 */
export class MonitoringIntegrationOrchestrator {
  private metricsCollector: UnifiedMetricsCollector;
  private alertingEngine: IntelligentAlertingEngine;
  private deliveryManager: AlertDeliveryManager;
  private dashboardManager: ProductionDashboardManager;
  private config: AlertingConfig;
  private isRunning: boolean = false;

  constructor(config: AlertingConfig) {
    this.config = config;
    this.metricsCollector = new UnifiedMetricsCollector();
    this.alertingEngine = new IntelligentAlertingEngine(config, this.metricsCollector);
    this.deliveryManager = new AlertDeliveryManager(config);
    this.dashboardManager = new ProductionDashboardManager(this.metricsCollector);
  }

  /**
   * Start integrated monitoring system
   */
  async startMonitoring(): Promise<void> {
    if (this.isRunning) return;

    this.isRunning = true;
    console.log('🚀 Starting integrated monitoring system');

    // Start alert evaluation loop
    setInterval(async () => {
      if (this.isRunning) {
        await this.processAlertingCycle();
      }
    }, 30000); // Every 30 seconds

    console.log('✅ Monitoring integration active');
  }

  /**
   * Stop monitoring system
   */
  stopMonitoring(): void {
    this.isRunning = false;
    console.log('🛑 Stopped monitoring integration');
  }

  /**
   * Ingest metrics from production validation systems
   */
  ingestMetrics(validationData: Parameters<UnifiedMetricsCollector['batchCollectFromValidationSystems']>[0]): void {
    this.metricsCollector.batchCollectFromValidationSystems(validationData);
  }

  /**
   * Get comprehensive monitoring status
   */
  getMonitoringStatus(): {
    system_health: {
      is_running: boolean;
      metrics_collected: number;
      active_alerts: number;
      dashboards_available: number;
    };
    recent_alerts: Alert[];
    alert_statistics: ReturnType<IntelligentAlertingEngine['getAlertStatistics']>;
    available_dashboards: ReturnType<ProductionDashboardManager['getAvailableDashboards']>;
  } {
    const activeAlerts = this.alertingEngine.getActiveAlerts();
    const alertStats = this.alertingEngine.getAlertStatistics();
    const dashboards = this.dashboardManager.getAvailableDashboards();
    const metricNames = this.metricsCollector.getMetricNames();

    return {
      system_health: {
        is_running: this.isRunning,
        metrics_collected: metricNames.length,
        active_alerts: activeAlerts.length,
        dashboards_available: dashboards.length
      },
      recent_alerts: activeAlerts.slice(0, 10),
      alert_statistics: alertStats,
      available_dashboards: dashboards
    };
  }

  /**
   * Get dashboard data
   */
  getDashboard(dashboardId: string): ReturnType<ProductionDashboardManager['getDashboardData']> {
    return this.dashboardManager.getDashboardData(dashboardId);
  }

  /**
   * Create manual alert
   */
  createManualAlert(
    source: string,
    title: string,
    description: string,
    severity: Alert['severity'],
    metrics: Record<string, number> = {}
  ): Alert {
    return this.alertingEngine.createValidationSystemAlert(source, title, description, severity, metrics);
  }

  /**
   * Acknowledge alert
   */
  acknowledgeAlert(alertId: string, acknowledgedBy: string): boolean {
    return this.alertingEngine.acknowledgeAlert(alertId, acknowledgedBy);
  }

  /**
   * Resolve alert
   */
  resolveAlert(alertId: string, resolvedBy: string): boolean {
    return this.alertingEngine.resolveAlert(alertId, resolvedBy);
  }

  private async processAlertingCycle(): Promise<void> {
    try {
      // Evaluate all alerting rules
      const newAlerts = await this.alertingEngine.evaluateAlerts();
      
      // Deliver new alerts
      for (const alert of newAlerts) {
        const deliveryResult = await this.deliveryManager.deliverAlert(alert);
        
        if (!deliveryResult.delivered) {
          console.error(`Failed to deliver alert ${alert.id}: ${deliveryResult.delivery_errors.join(', ')}`);
        } else {
          console.log(`📢 Alert delivered via ${deliveryResult.channels_used.join(', ')}: ${alert.title}`);
        }
      }
    } catch (error) {
      console.error(`Error in alerting cycle: ${error}`);
    }
  }

  /**
   * Health check for monitoring integration
   */
  healthCheck(): {
    healthy: boolean;
    issues: string[];
    metrics: {
      monitoring_running: boolean;
      metrics_flowing: boolean;
      alerting_functional: boolean;
      dashboards_accessible: boolean;
    };
  } {
    const issues: string[] = [];
    
    if (!this.isRunning) {
      issues.push('Monitoring system not running');
    }
    
    const metricNames = this.metricsCollector.getMetricNames();
    if (metricNames.length === 0) {
      issues.push('No metrics being collected');
    }
    
    const activeAlerts = this.alertingEngine.getActiveAlerts();
    const criticalAlerts = activeAlerts.filter(a => a.severity === 'CRITICAL' || a.severity === 'EMERGENCY');
    if (criticalAlerts.length > 5) {
      issues.push(`Too many critical alerts: ${criticalAlerts.length}`);
    }
    
    return {
      healthy: issues.length === 0,
      issues,
      metrics: {
        monitoring_running: this.isRunning,
        metrics_flowing: metricNames.length > 0,
        alerting_functional: true, // Simplified check
        dashboards_accessible: this.dashboardManager.getAvailableDashboards().length > 0
      }
    };
  }
}