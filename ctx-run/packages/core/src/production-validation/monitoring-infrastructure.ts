/**
 * Real-time Monitoring Infrastructure
 * λ/size/CBU dashboards with CUSUM alerts for production validation
 */

export interface MonitoringMetrics {
  lambda: number; // Model parameter λ
  size: number; // Context/embedding size
  cbu: number; // Context Budget Units
  timestamp: number;
  metadata?: Record<string, any>;
}

export interface Alert {
  id: string;
  type: 'CUSUM' | 'THRESHOLD' | 'TREND';
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  message: string;
  metrics: MonitoringMetrics;
  timestamp: number;
  acknowledged: boolean;
}

export interface DashboardConfig {
  lambda_bounds: [number, number];
  size_bounds: [number, number];
  cbu_bounds: [number, number];
  cusum_threshold: number;
  alert_cooldown_ms: number;
  retention_hours: number;
}

/**
 * CUSUM (Cumulative Sum) Alert System
 * Detects statistical shifts in monitoring metrics
 */
export class CUSUMAlertSystem {
  private config: DashboardConfig;
  private cusumState: {
    lambda: { sum: number, baseline: number, count: number },
    size: { sum: number, baseline: number, count: number },
    cbu: { sum: number, baseline: number, count: number }
  };
  private lastAlertTime: Record<string, number> = {};

  constructor(config: DashboardConfig) {
    this.config = config;
    this.cusumState = {
      lambda: { sum: 0, baseline: 0, count: 0 },
      size: { sum: 0, baseline: 0, count: 0 },
      cbu: { sum: 0, baseline: 0, count: 0 }
    };
  }

  /**
   * Update CUSUM state and check for alerts
   */
  checkMetrics(metrics: MonitoringMetrics): Alert[] {
    const alerts: Alert[] = [];
    const now = Date.now();

    // Update baselines if this is initial data
    if (this.cusumState.lambda.count === 0) {
      this.initializeBaselines(metrics);
      return alerts;
    }

    // Lambda CUSUM check
    const lambdaShift = this.updateCUSUM('lambda', metrics.lambda, metrics);
    if (lambdaShift && this.shouldAlert('lambda', now)) {
      alerts.push(this.createAlert('lambda', 'CUSUM', lambdaShift, metrics));
    }

    // Size CUSUM check
    const sizeShift = this.updateCUSUM('size', metrics.size, metrics);
    if (sizeShift && this.shouldAlert('size', now)) {
      alerts.push(this.createAlert('size', 'CUSUM', sizeShift, metrics));
    }

    // CBU CUSUM check
    const cbuShift = this.updateCUSUM('cbu', metrics.cbu, metrics);
    if (cbuShift && this.shouldAlert('cbu', now)) {
      alerts.push(this.createAlert('cbu', 'CUSUM', cbuShift, metrics));
    }

    // Threshold alerts
    alerts.push(...this.checkThresholdAlerts(metrics, now));

    return alerts;
  }

  private initializeBaselines(metrics: MonitoringMetrics): void {
    this.cusumState.lambda.baseline = metrics.lambda;
    this.cusumState.size.baseline = metrics.size;
    this.cusumState.cbu.baseline = metrics.cbu;
    this.cusumState.lambda.count = 1;
    this.cusumState.size.count = 1;
    this.cusumState.cbu.count = 1;
  }

  private updateCUSUM(
    metric: 'lambda' | 'size' | 'cbu',
    value: number,
    metrics: MonitoringMetrics
  ): { severity: string, shift: number } | null {
    const state = this.cusumState[metric];
    
    // Calculate deviation from baseline
    const deviation = value - state.baseline;
    
    // Update CUSUM (reset to 0 if negative)
    state.sum = Math.max(0, state.sum + deviation - this.config.cusum_threshold);
    state.count++;

    // Update baseline with exponential moving average
    const alpha = 0.1; // Smoothing factor
    state.baseline = alpha * value + (1 - alpha) * state.baseline;

    // Check if CUSUM exceeds threshold
    if (state.sum > this.config.cusum_threshold * 3) {
      const severity = state.sum > this.config.cusum_threshold * 5 ? 'CRITICAL' : 'HIGH';
      return {
        severity,
        shift: state.sum
      };
    }

    return null;
  }

  private checkThresholdAlerts(metrics: MonitoringMetrics, now: number): Alert[] {
    const alerts: Alert[] = [];

    // Lambda bounds check
    if (metrics.lambda < this.config.lambda_bounds[0] || metrics.lambda > this.config.lambda_bounds[1]) {
      if (this.shouldAlert('lambda_threshold', now)) {
        alerts.push({
          id: `lambda_threshold_${now}`,
          type: 'THRESHOLD',
          severity: 'HIGH',
          message: `Lambda ${metrics.lambda} outside bounds ${this.config.lambda_bounds}`,
          metrics,
          timestamp: now,
          acknowledged: false
        });
      }
    }

    // Size bounds check
    if (metrics.size < this.config.size_bounds[0] || metrics.size > this.config.size_bounds[1]) {
      if (this.shouldAlert('size_threshold', now)) {
        alerts.push({
          id: `size_threshold_${now}`,
          type: 'THRESHOLD',
          severity: 'MEDIUM',
          message: `Size ${metrics.size} outside bounds ${this.config.size_bounds}`,
          metrics,
          timestamp: now,
          acknowledged: false
        });
      }
    }

    // CBU bounds check
    if (metrics.cbu < this.config.cbu_bounds[0] || metrics.cbu > this.config.cbu_bounds[1]) {
      if (this.shouldAlert('cbu_threshold', now)) {
        alerts.push({
          id: `cbu_threshold_${now}`,
          type: 'THRESHOLD',
          severity: 'HIGH',
          message: `CBU ${metrics.cbu} outside bounds ${this.config.cbu_bounds}`,
          metrics,
          timestamp: now,
          acknowledged: false
        });
      }
    }

    return alerts;
  }

  private shouldAlert(alertType: string, now: number): boolean {
    const lastAlert = this.lastAlertTime[alertType] || 0;
    if (now - lastAlert >= this.config.alert_cooldown_ms) {
      this.lastAlertTime[alertType] = now;
      return true;
    }
    return false;
  }

  private createAlert(
    metric: string,
    type: 'CUSUM',
    shift: { severity: string, shift: number },
    metrics: MonitoringMetrics
  ): Alert {
    return {
      id: `${metric}_cusum_${Date.now()}`,
      type,
      severity: shift.severity as Alert['severity'],
      message: `CUSUM shift detected in ${metric}: ${shift.shift.toFixed(3)} (threshold: ${this.config.cusum_threshold})`,
      metrics,
      timestamp: Date.now(),
      acknowledged: false
    };
  }

  /**
   * Get current CUSUM state for debugging
   */
  getState(): typeof this.cusumState {
    return { ...this.cusumState };
  }

  /**
   * Reset CUSUM state (useful for system restarts)
   */
  reset(): void {
    this.cusumState = {
      lambda: { sum: 0, baseline: 0, count: 0 },
      size: { sum: 0, baseline: 0, count: 0 },
      cbu: { sum: 0, baseline: 0, count: 0 }
    };
    this.lastAlertTime = {};
  }
}

/**
 * Real-time Dashboard Data Manager
 * Maintains sliding window of metrics and provides aggregations
 */
export class DashboardDataManager {
  private metrics: MonitoringMetrics[] = [];
  private config: DashboardConfig;
  private maxDataPoints: number;

  constructor(config: DashboardConfig) {
    this.config = config;
    // Calculate max data points based on retention period
    this.maxDataPoints = Math.max(1000, config.retention_hours * 3600); // Assume 1 point per second
  }

  /**
   * Add new metrics data point
   */
  addMetrics(metrics: MonitoringMetrics): void {
    this.metrics.push(metrics);
    
    // Remove old data points beyond retention period
    const cutoffTime = Date.now() - (this.config.retention_hours * 60 * 60 * 1000);
    this.metrics = this.metrics.filter(m => m.timestamp >= cutoffTime);
    
    // Limit total data points to prevent memory issues
    if (this.metrics.length > this.maxDataPoints) {
      this.metrics = this.metrics.slice(-this.maxDataPoints);
    }
  }

  /**
   * Get recent metrics for dashboard display
   */
  getRecentMetrics(lastNMinutes: number = 60): MonitoringMetrics[] {
    const cutoffTime = Date.now() - (lastNMinutes * 60 * 1000);
    return this.metrics.filter(m => m.timestamp >= cutoffTime);
  }

  /**
   * Get aggregated statistics for dashboard
   */
  getAggregatedStats(lastNMinutes: number = 60): {
    lambda: { min: number, max: number, avg: number, current: number },
    size: { min: number, max: number, avg: number, current: number },
    cbu: { min: number, max: number, avg: number, current: number },
    dataPoints: number,
    timeRange: [number, number]
  } {
    const recentMetrics = this.getRecentMetrics(lastNMinutes);
    
    if (recentMetrics.length === 0) {
      return {
        lambda: { min: 0, max: 0, avg: 0, current: 0 },
        size: { min: 0, max: 0, avg: 0, current: 0 },
        cbu: { min: 0, max: 0, avg: 0, current: 0 },
        dataPoints: 0,
        timeRange: [0, 0]
      };
    }

    const lambdaValues = recentMetrics.map(m => m.lambda);
    const sizeValues = recentMetrics.map(m => m.size);
    const cbuValues = recentMetrics.map(m => m.cbu);
    
    const latest = recentMetrics[recentMetrics.length - 1];
    const earliest = recentMetrics[0];

    return {
      lambda: {
        min: Math.min(...lambdaValues),
        max: Math.max(...lambdaValues),
        avg: lambdaValues.reduce((a, b) => a + b, 0) / lambdaValues.length,
        current: latest.lambda
      },
      size: {
        min: Math.min(...sizeValues),
        max: Math.max(...sizeValues),
        avg: sizeValues.reduce((a, b) => a + b, 0) / sizeValues.length,
        current: latest.size
      },
      cbu: {
        min: Math.min(...cbuValues),
        max: Math.max(...cbuValues),
        avg: cbuValues.reduce((a, b) => a + b, 0) / cbuValues.length,
        current: latest.cbu
      },
      dataPoints: recentMetrics.length,
      timeRange: [earliest.timestamp, latest.timestamp]
    };
  }

  /**
   * Get time-series data for charting
   */
  getTimeSeriesData(lastNMinutes: number = 60): {
    timestamps: number[],
    lambda: number[],
    size: number[],
    cbu: number[]
  } {
    const recentMetrics = this.getRecentMetrics(lastNMinutes);
    
    return {
      timestamps: recentMetrics.map(m => m.timestamp),
      lambda: recentMetrics.map(m => m.lambda),
      size: recentMetrics.map(m => m.size),
      cbu: recentMetrics.map(m => m.cbu)
    };
  }

  /**
   * Calculate performance trends
   */
  calculateTrends(lastNMinutes: number = 60): {
    lambda: { trend: number, confidence: number },
    size: { trend: number, confidence: number },
    cbu: { trend: number, confidence: number }
  } {
    const data = this.getTimeSeriesData(lastNMinutes);
    
    if (data.timestamps.length < 2) {
      return {
        lambda: { trend: 0, confidence: 0 },
        size: { trend: 0, confidence: 0 },
        cbu: { trend: 0, confidence: 0 }
      };
    }

    return {
      lambda: this.calculateLinearTrend(data.timestamps, data.lambda),
      size: this.calculateLinearTrend(data.timestamps, data.size),
      cbu: this.calculateLinearTrend(data.timestamps, data.cbu)
    };
  }

  private calculateLinearTrend(x: number[], y: number[]): { trend: number, confidence: number } {
    const n = x.length;
    if (n < 2) return { trend: 0, confidence: 0 };

    // Normalize x values to reduce numerical errors
    const minX = Math.min(...x);
    const normalizedX = x.map(val => val - minX);

    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    
    for (let i = 0; i < n; i++) {
      sumX += normalizedX[i];
      sumY += y[i];
      sumXY += normalizedX[i] * y[i];
      sumXX += normalizedX[i] * normalizedX[i];
    }

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    
    // Calculate R-squared for confidence
    const meanY = sumY / n;
    let ssRes = 0, ssTot = 0;
    
    for (let i = 0; i < n; i++) {
      const predicted = slope * normalizedX[i] + (sumY - slope * sumX) / n;
      ssRes += Math.pow(y[i] - predicted, 2);
      ssTot += Math.pow(y[i] - meanY, 2);
    }
    
    const rSquared = ssTot === 0 ? 1 : 1 - (ssRes / ssTot);
    
    return {
      trend: slope,
      confidence: Math.max(0, Math.min(1, rSquared))
    };
  }
}

/**
 * Production Monitoring Orchestrator
 * Coordinates real-time monitoring, alerting, and dashboard data
 */
export class ProductionMonitoringOrchestrator {
  private cusumAlerts: CUSUMAlertSystem;
  private dashboardData: DashboardDataManager;
  private alerts: Alert[] = [];
  private config: DashboardConfig;
  private isRunning: boolean = false;
  private intervalId?: NodeJS.Timeout;

  constructor(config: DashboardConfig) {
    this.config = config;
    this.cusumAlerts = new CUSUMAlertSystem(config);
    this.dashboardData = new DashboardDataManager(config);
  }

  /**
   * Start monitoring system
   */
  start(): void {
    if (this.isRunning) return;
    
    this.isRunning = true;
    console.log('🚀 Production monitoring system started');
    
    // Start periodic alert cleanup
    this.intervalId = setInterval(() => {
      this.cleanupOldAlerts();
    }, 60000); // Cleanup every minute
  }

  /**
   * Stop monitoring system
   */
  stop(): void {
    if (!this.isRunning) return;
    
    this.isRunning = false;
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = undefined;
    }
    console.log('🛑 Production monitoring system stopped');
  }

  /**
   * Process new metrics and generate alerts
   */
  processMetrics(metrics: MonitoringMetrics): Alert[] {
    if (!this.isRunning) {
      throw new Error('Monitoring system not running');
    }

    // Add to dashboard data
    this.dashboardData.addMetrics(metrics);
    
    // Check for alerts
    const newAlerts = this.cusumAlerts.checkMetrics(metrics);
    
    // Store alerts
    this.alerts.push(...newAlerts);
    
    // Log critical alerts immediately
    newAlerts.filter(a => a.severity === 'CRITICAL').forEach(alert => {
      console.error(`🚨 CRITICAL ALERT: ${alert.message}`);
    });
    
    return newAlerts;
  }

  /**
   * Get current dashboard state
   */
  getDashboardState(timeWindow: number = 60): {
    stats: ReturnType<DashboardDataManager['getAggregatedStats']>,
    trends: ReturnType<DashboardDataManager['calculateTrends']>,
    timeSeries: ReturnType<DashboardDataManager['getTimeSeriesData']>,
    alerts: Alert[],
    cusumState: ReturnType<CUSUMAlertSystem['getState']>
  } {
    return {
      stats: this.dashboardData.getAggregatedStats(timeWindow),
      trends: this.dashboardData.calculateTrends(timeWindow),
      timeSeries: this.dashboardData.getTimeSeriesData(timeWindow),
      alerts: this.getActiveAlerts(),
      cusumState: this.cusumAlerts.getState()
    };
  }

  /**
   * Acknowledge alert
   */
  acknowledgeAlert(alertId: string): boolean {
    const alert = this.alerts.find(a => a.id === alertId);
    if (alert) {
      alert.acknowledged = true;
      return true;
    }
    return false;
  }

  /**
   * Get active (unacknowledged) alerts
   */
  getActiveAlerts(): Alert[] {
    return this.alerts.filter(a => !a.acknowledged);
  }

  /**
   * Get all alerts within time window
   */
  getAllAlerts(lastNMinutes: number = 60): Alert[] {
    const cutoffTime = Date.now() - (lastNMinutes * 60 * 1000);
    return this.alerts.filter(a => a.timestamp >= cutoffTime);
  }

  /**
   * Health check for monitoring system
   */
  healthCheck(): {
    healthy: boolean,
    issues: string[],
    metrics: {
      isRunning: boolean,
      alertCount: number,
      dataPoints: number,
      criticalAlerts: number
    }
  } {
    const issues: string[] = [];
    
    if (!this.isRunning) {
      issues.push('Monitoring system not running');
    }
    
    const criticalAlerts = this.alerts.filter(a => 
      a.severity === 'CRITICAL' && !a.acknowledged
    ).length;
    
    if (criticalAlerts > 0) {
      issues.push(`${criticalAlerts} unacknowledged critical alerts`);
    }
    
    const recentData = this.dashboardData.getRecentMetrics(5);
    if (recentData.length === 0 && this.isRunning) {
      issues.push('No recent metrics data');
    }
    
    return {
      healthy: issues.length === 0,
      issues,
      metrics: {
        isRunning: this.isRunning,
        alertCount: this.alerts.length,
        dataPoints: this.dashboardData.getRecentMetrics(60).length,
        criticalAlerts
      }
    };
  }

  private cleanupOldAlerts(): void {
    const cutoffTime = Date.now() - (this.config.retention_hours * 60 * 60 * 1000);
    const initialCount = this.alerts.length;
    
    this.alerts = this.alerts.filter(alert => 
      alert.timestamp >= cutoffTime || !alert.acknowledged
    );
    
    const removed = initialCount - this.alerts.length;
    if (removed > 0) {
      console.log(`🧹 Cleaned up ${removed} old alerts`);
    }
  }

  /**
   * Export monitoring data for external systems
   */
  exportData(format: 'json' | 'csv' = 'json', lastNMinutes: number = 60): string {
    const data = {
      stats: this.dashboardData.getAggregatedStats(lastNMinutes),
      timeSeries: this.dashboardData.getTimeSeriesData(lastNMinutes),
      alerts: this.getAllAlerts(lastNMinutes),
      exportTime: Date.now()
    };
    
    if (format === 'json') {
      return JSON.stringify(data, null, 2);
    }
    
    // Simple CSV export for time series data
    const { timestamps, lambda, size, cbu } = data.timeSeries;
    let csv = 'timestamp,lambda,size,cbu\n';
    
    for (let i = 0; i < timestamps.length; i++) {
      csv += `${timestamps[i]},${lambda[i]},${size[i]},${cbu[i]}\n`;
    }
    
    return csv;
  }
}