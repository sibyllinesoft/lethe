/**
 * Production Monitoring & Alerting System
 * 
 * Implements comprehensive real-time monitoring for Lethe production deployment:
 * - λ/size/CBU dashboards with drift detection
 * - Proxy gap alerts (median gap > 0.5% threshold)
 * - λ-drift bounds and acceptance rate monitoring  
 * - CBU elasticity tracking (ΔCBU/Δλ smoothness)
 * - Shadow-price consistency invariants
 * - Risk budget ledger with automated responses
 * 
 * Provides operational insight and automated responses per TODO.md requirements.
 */

import type { DB } from '@lethe/sqlite';
import type { Candidate } from './index.js';

// Core monitoring data structures
export interface SystemMetrics {
  timestamp: string;
  session_id: string;
  
  // Lambda and size metrics
  lambda_value: number;
  context_size_tokens: number;
  size_lambda_ratio: number;
  
  // CBU (Compute Budget Unit) metrics
  cbu_consumption: number;
  cbu_efficiency: number; // CBU per useful token
  delta_cbu_delta_lambda: number; // CBU elasticity
  
  // Performance metrics
  p95_latency_ms: number;
  throughput_qps: number;
  accept_rate: number;
  
  // Quality metrics
  primal_dual_gap: number;
  shadow_price: number;
  ece_score: number; // Calibration error
  ilp_incidence: number; // Integer Linear Programming usage rate
  
  // System health
  memory_usage_mb: number;
  cpu_utilization_percent: number;
  error_rate: number;
}

export interface AlertThresholds {
  // Gap and consistency thresholds
  max_primal_dual_gap: number; // 0.5% of λ·tokens
  shadow_price_stability_window: number; // ±15%
  lambda_drift_bound: number; // 10%
  
  // Performance thresholds
  max_p95_latency_ms: number; // 160ms target
  min_accept_rate: number; // 85%
  max_ece_score: number; // 8%
  max_ilp_incidence: number; // 5%
  
  // CBU elasticity bounds
  cbu_elasticity_smoothness_threshold: number;
  cbu_efficiency_degradation_limit: number;
  
  // System resource limits
  max_memory_usage_mb: number;
  max_cpu_utilization: number;
  max_error_rate: number;
}

export interface Alert {
  alert_id: string;
  timestamp: string;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  category: 'PERFORMANCE' | 'QUALITY' | 'CONSISTENCY' | 'RESOURCE' | 'BUSINESS';
  metric_name: string;
  current_value: number;
  threshold_value: number;
  session_id?: string;
  message: string;
  recommended_action: string;
  auto_resolution_attempted: boolean;
}

export interface DashboardData {
  timestamp: string;
  overview: {
    system_health_score: number; // 0-100
    active_sessions: number;
    total_requests_last_hour: number;
    current_lambda_distribution: number[];
    cbu_burn_rate_per_hour: number;
  };
  
  lambda_size_analysis: {
    lambda_values: number[];
    size_values: number[];
    monotonicity_score: number;
    trend_slope: number;
    outlier_count: number;
  };
  
  cbu_tracking: {
    hourly_consumption: number[];
    efficiency_trend: number[];
    elasticity_curve: Array<{ lambda: number; cbu_rate: number }>;
    budget_utilization: number; // 0-1
  };
  
  quality_metrics: {
    ece_distribution: number[];
    ilp_usage_rate: number;
    calibration_trend: number[];
    gap_distribution: number[];
  };
  
  alerts_summary: {
    active_alerts: Alert[];
    alerts_last_24h: number;
    critical_alerts_active: number;
    auto_resolutions_successful: number;
  };
}

export interface RiskBudgetStatus {
  current_risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  budget_utilization_percent: number;
  remaining_budget_hours: number;
  
  risk_contributors: Array<{
    factor: string;
    contribution_percent: number;
    severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
    trend: 'IMPROVING' | 'STABLE' | 'DEGRADING';
  }>;
  
  recommended_actions: string[];
  automatic_mitigations_active: string[];
}

export const DEFAULT_ALERT_THRESHOLDS: AlertThresholds = {
  // Gap and consistency (per TODO.md specifications)
  max_primal_dual_gap: 0.005, // 0.5%
  shadow_price_stability_window: 0.15, // ±15%
  lambda_drift_bound: 0.10, // 10%
  
  // Performance targets
  max_p95_latency_ms: 160, // 150-160ms P95 target
  min_accept_rate: 0.85, // 85%
  max_ece_score: 0.08, // 8% ECE threshold
  max_ilp_incidence: 0.05, // 5% ILP usage
  
  // CBU elasticity
  cbu_elasticity_smoothness_threshold: 0.05,
  cbu_efficiency_degradation_limit: 0.20, // 20% degradation
  
  // System resources
  max_memory_usage_mb: 8192, // 8GB
  max_cpu_utilization: 0.80, // 80%
  max_error_rate: 0.01, // 1%
};

/**
 * Production Monitoring System
 * 
 * Real-time monitoring with automated alerting and response capabilities
 */
export class ProductionMonitoringSystem {
  private db: DB;
  private thresholds: AlertThresholds;
  
  // State tracking
  private metricsHistory: SystemMetrics[] = [];
  private activeAlerts: Map<string, Alert> = new Map();
  private riskBudget: RiskBudgetStatus;
  
  // Monitoring intervals
  private metricsCollectionInterval: NodeJS.Timeout | null = null;
  private alertingInterval: NodeJS.Timeout | null = null;
  
  // Lambda drift tracking
  private lambdaBaseline: number | null = null;
  private shadowPriceWindow: number[] = [];
  private cbuElasticityHistory: Array<{ lambda: number; cbu_rate: number }> = [];

  constructor(db: DB, thresholds: Partial<AlertThresholds> = {}) {
    this.db = db;
    this.thresholds = { ...DEFAULT_ALERT_THRESHOLDS, ...thresholds };
    
    this.riskBudget = {
      current_risk_level: 'LOW',
      budget_utilization_percent: 0,
      remaining_budget_hours: 24,
      risk_contributors: [],
      recommended_actions: [],
      automatic_mitigations_active: [],
    };
  }

  /**
   * Start continuous monitoring
   */
  startMonitoring(metricsIntervalMs: number = 30000, alertingIntervalMs: number = 10000): void {
    console.log('🔍 Starting production monitoring system...');
    
    // Start metrics collection
    this.metricsCollectionInterval = setInterval(async () => {
      await this.collectSystemMetrics();
    }, metricsIntervalMs);
    
    // Start alerting
    this.alertingInterval = setInterval(async () => {
      await this.evaluateAlertsAndRespond();
    }, alertingIntervalMs);
    
    console.log(`✅ Monitoring active: metrics every ${metricsIntervalMs}ms, alerting every ${alertingIntervalMs}ms`);
  }

  /**
   * Stop monitoring
   */
  stopMonitoring(): void {
    if (this.metricsCollectionInterval) {
      clearInterval(this.metricsCollectionInterval);
      this.metricsCollectionInterval = null;
    }
    
    if (this.alertingInterval) {
      clearInterval(this.alertingInterval);
      this.alertingInterval = null;
    }
    
    console.log('🔍 Production monitoring stopped');
  }

  /**
   * Collect comprehensive system metrics
   */
  private async collectSystemMetrics(): Promise<void> {
    try {
      const timestamp = new Date().toISOString();
      
      // Get active session metrics
      const activeSessions = await this.getActiveSessionIds();
      
      for (const sessionId of activeSessions) {
        const metrics = await this.collectSessionMetrics(sessionId, timestamp);
        this.metricsHistory.push(metrics);
        
        // Update lambda baseline and drift tracking
        this.updateLambdaTracking(metrics.lambda_value);
        
        // Update shadow price consistency tracking
        this.updateShadowPriceTracking(metrics.shadow_price);
        
        // Update CBU elasticity tracking
        this.updateCBUElasticityTracking(metrics.lambda_value, metrics.delta_cbu_delta_lambda);
      }
      
      // Trim history to keep last 24 hours
      const cutoffTime = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
      this.metricsHistory = this.metricsHistory.filter(m => m.timestamp >= cutoffTime);
      
    } catch (error) {
      console.error(`❌ Metrics collection failed: ${error}`);
    }
  }

  /**
   * Collect metrics for a specific session
   */
  private async collectSessionMetrics(sessionId: string, timestamp: string): Promise<SystemMetrics> {
    // In production, these would be collected from actual system measurements
    // This is a framework showing the structure and calculations
    
    const lambdaValue = await this.getCurrentLambda(sessionId);
    const contextSize = await this.getContextSizeTokens(sessionId);
    const cbuConsumption = await this.getCBUConsumption(sessionId);
    const performanceMetrics = await this.getPerformanceMetrics(sessionId);
    const qualityMetrics = await this.getQualityMetrics(sessionId);
    const systemHealth = await this.getSystemHealthMetrics();

    return {
      timestamp,
      session_id: sessionId,
      
      // Lambda and size
      lambda_value: lambdaValue,
      context_size_tokens: contextSize,
      size_lambda_ratio: contextSize / lambdaValue,
      
      // CBU metrics
      cbu_consumption: cbuConsumption.total,
      cbu_efficiency: cbuConsumption.efficiency,
      delta_cbu_delta_lambda: cbuConsumption.elasticity,
      
      // Performance
      p95_latency_ms: performanceMetrics.p95_latency,
      throughput_qps: performanceMetrics.throughput,
      accept_rate: performanceMetrics.accept_rate,
      
      // Quality
      primal_dual_gap: qualityMetrics.gap,
      shadow_price: qualityMetrics.shadow_price,
      ece_score: qualityMetrics.ece,
      ilp_incidence: qualityMetrics.ilp_rate,
      
      // System health
      memory_usage_mb: systemHealth.memory_mb,
      cpu_utilization_percent: systemHealth.cpu_percent,
      error_rate: systemHealth.error_rate,
    };
  }

  /**
   * Evaluate all alert conditions and trigger responses
   */
  private async evaluateAlertsAndRespond(): Promise<void> {
    if (this.metricsHistory.length === 0) return;

    const recentMetrics = this.metricsHistory.slice(-10); // Last 10 data points
    
    // Evaluate each alert condition
    await Promise.all([
      this.evaluatePrimalDualGapAlert(recentMetrics),
      this.evaluateLambdaDriftAlert(recentMetrics),
      this.evaluateShadowPriceConsistencyAlert(),
      this.evaluatePerformanceAlerts(recentMetrics),
      this.evaluateQualityAlerts(recentMetrics),
      this.evaluateCBUElasticityAlert(),
      this.evaluateSystemHealthAlerts(recentMetrics),
    ]);

    // Update risk budget status
    this.updateRiskBudgetStatus();

    // Attempt automatic resolutions
    await this.attemptAutomaticResolutions();
  }

  /**
   * Core Alert Evaluators
   */
  
  private async evaluatePrimalDualGapAlert(metrics: SystemMetrics[]): Promise<void> {
    const gaps = metrics.map(m => m.primal_dual_gap);
    const medianGap = this.calculateMedian(gaps);
    
    // Calculate dynamic threshold based on λ·tokens
    const avgLambda = metrics.reduce((sum, m) => sum + m.lambda_value, 0) / metrics.length;
    const avgSize = metrics.reduce((sum, m) => sum + m.context_size_tokens, 0) / metrics.length;
    const dynamicThreshold = this.thresholds.max_primal_dual_gap * avgLambda * avgSize;

    if (medianGap > dynamicThreshold) {
      await this.createAlert({
        severity: 'HIGH',
        category: 'QUALITY',
        metric_name: 'primal_dual_gap',
        current_value: medianGap,
        threshold_value: dynamicThreshold,
        message: `Median primal-dual gap ${medianGap.toFixed(6)} exceeds threshold ${dynamicThreshold.toFixed(6)}`,
        recommended_action: 'Check knapsack optimizer convergence and cache coherence',
      });
    }
  }

  private async evaluateLambdaDriftAlert(metrics: SystemMetrics[]): Promise<void> {
    if (!this.lambdaBaseline) return;

    const currentLambda = metrics[metrics.length - 1].lambda_value;
    const driftPercent = Math.abs(currentLambda - this.lambdaBaseline) / this.lambdaBaseline;

    if (driftPercent > this.thresholds.lambda_drift_bound) {
      await this.createAlert({
        severity: 'MEDIUM',
        category: 'CONSISTENCY',
        metric_name: 'lambda_drift',
        current_value: driftPercent,
        threshold_value: this.thresholds.lambda_drift_bound,
        message: `Lambda drift ${(driftPercent * 100).toFixed(1)}% exceeds bounds`,
        recommended_action: 'Investigate lambda control stability and recalibrate if needed',
      });
    }
  }

  private async evaluateShadowPriceConsistencyAlert(): Promise<void> {
    if (this.shadowPriceWindow.length < 10) return;

    const median = this.calculateMedian(this.shadowPriceWindow);
    const stabilityWindow = this.thresholds.shadow_price_stability_window;
    
    const violations = this.shadowPriceWindow.filter(price => 
      Math.abs(price - median) / median > stabilityWindow
    ).length;

    const violationRate = violations / this.shadowPriceWindow.length;

    if (violationRate > 0.1) { // >10% violations
      await this.createAlert({
        severity: 'MEDIUM',
        category: 'CONSISTENCY',
        metric_name: 'shadow_price_consistency',
        current_value: violationRate,
        threshold_value: 0.1,
        message: `Shadow price violations ${(violationRate * 100).toFixed(1)}% of samples`,
        recommended_action: 'Check constraint stability and optimization convergence',
      });
    }
  }

  private async evaluatePerformanceAlerts(metrics: SystemMetrics[]): Promise<void> {
    const latestMetrics = metrics[metrics.length - 1];

    // P95 latency alert
    if (latestMetrics.p95_latency_ms > this.thresholds.max_p95_latency_ms) {
      await this.createAlert({
        severity: 'HIGH',
        category: 'PERFORMANCE',
        metric_name: 'p95_latency',
        current_value: latestMetrics.p95_latency_ms,
        threshold_value: this.thresholds.max_p95_latency_ms,
        message: `P95 latency ${latestMetrics.p95_latency_ms.toFixed(1)}ms exceeds target`,
        recommended_action: 'Check system load and optimize hot paths',
      });
    }

    // Accept rate alert
    if (latestMetrics.accept_rate < this.thresholds.min_accept_rate) {
      await this.createAlert({
        severity: 'HIGH',
        category: 'PERFORMANCE',
        metric_name: 'accept_rate',
        current_value: latestMetrics.accept_rate,
        threshold_value: this.thresholds.min_accept_rate,
        message: `Accept rate ${(latestMetrics.accept_rate * 100).toFixed(1)}% below threshold`,
        recommended_action: 'Investigate request processing pipeline and capacity',
      });
    }
  }

  private async evaluateQualityAlerts(metrics: SystemMetrics[]): Promise<void> {
    const latestMetrics = metrics[metrics.length - 1];

    // ECE score alert
    if (latestMetrics.ece_score > this.thresholds.max_ece_score) {
      await this.createAlert({
        severity: 'MEDIUM',
        category: 'QUALITY',
        metric_name: 'ece_score',
        current_value: latestMetrics.ece_score,
        threshold_value: this.thresholds.max_ece_score,
        message: `ECE score ${(latestMetrics.ece_score * 100).toFixed(1)}% exceeds target`,
        recommended_action: 'Recalibrate confidence estimation and isotonic regression',
      });
    }

    // ILP incidence alert
    if (latestMetrics.ilp_incidence > this.thresholds.max_ilp_incidence) {
      await this.createAlert({
        severity: 'MEDIUM',
        category: 'QUALITY',
        metric_name: 'ilp_incidence',
        current_value: latestMetrics.ilp_incidence,
        threshold_value: this.thresholds.max_ilp_incidence,
        message: `ILP usage ${(latestMetrics.ilp_incidence * 100).toFixed(1)}% exceeds target`,
        recommended_action: 'Tune knapsack approximation and DPP rank selection',
      });
    }
  }

  private async evaluateCBUElasticityAlert(): Promise<void> {
    if (this.cbuElasticityHistory.length < 5) return;

    // Check for smoothness in ΔCBU/Δλ curve
    let jaggednessScore = 0;
    for (let i = 1; i < this.cbuElasticityHistory.length - 1; i++) {
      const prev = this.cbuElasticityHistory[i - 1];
      const curr = this.cbuElasticityHistory[i];
      const next = this.cbuElasticityHistory[i + 1];
      
      // Calculate second derivative approximation
      const secondDerivative = Math.abs((next.cbu_rate - curr.cbu_rate) - (curr.cbu_rate - prev.cbu_rate));
      jaggednessScore += secondDerivative;
    }

    const avgJaggedness = jaggednessScore / (this.cbuElasticityHistory.length - 2);

    if (avgJaggedness > this.thresholds.cbu_elasticity_smoothness_threshold) {
      await this.createAlert({
        severity: 'MEDIUM',
        category: 'CONSISTENCY',
        metric_name: 'cbu_elasticity_smoothness',
        current_value: avgJaggedness,
        threshold_value: this.thresholds.cbu_elasticity_smoothness_threshold,
        message: `CBU elasticity jaggedness ${avgJaggedness.toFixed(4)} indicates constraint bugs`,
        recommended_action: 'Check latent constraint bugs and lambda control smoothness',
      });
    }
  }

  private async evaluateSystemHealthAlerts(metrics: SystemMetrics[]): Promise<void> {
    const latestMetrics = metrics[metrics.length - 1];

    // Memory usage alert
    if (latestMetrics.memory_usage_mb > this.thresholds.max_memory_usage_mb) {
      await this.createAlert({
        severity: 'HIGH',
        category: 'RESOURCE',
        metric_name: 'memory_usage',
        current_value: latestMetrics.memory_usage_mb,
        threshold_value: this.thresholds.max_memory_usage_mb,
        message: `Memory usage ${latestMetrics.memory_usage_mb}MB exceeds limit`,
        recommended_action: 'Check for memory leaks and optimize caching',
      });
    }

    // CPU utilization alert
    if (latestMetrics.cpu_utilization_percent > this.thresholds.max_cpu_utilization) {
      await this.createAlert({
        severity: 'MEDIUM',
        category: 'RESOURCE',
        metric_name: 'cpu_utilization',
        current_value: latestMetrics.cpu_utilization_percent,
        threshold_value: this.thresholds.max_cpu_utilization,
        message: `CPU utilization ${(latestMetrics.cpu_utilization_percent * 100).toFixed(1)}% high`,
        recommended_action: 'Scale up capacity or optimize computational efficiency',
      });
    }

    // Error rate alert
    if (latestMetrics.error_rate > this.thresholds.max_error_rate) {
      await this.createAlert({
        severity: 'CRITICAL',
        category: 'BUSINESS',
        metric_name: 'error_rate',
        current_value: latestMetrics.error_rate,
        threshold_value: this.thresholds.max_error_rate,
        message: `Error rate ${(latestMetrics.error_rate * 100).toFixed(2)}% exceeds threshold`,
        recommended_action: 'Investigate error patterns and implement fixes immediately',
      });
    }
  }

  /**
   * Alert creation and management
   */
  private async createAlert(alertData: Partial<Alert>): Promise<void> {
    const alertId = `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const alert: Alert = {
      alert_id: alertId,
      timestamp: new Date().toISOString(),
      severity: alertData.severity || 'MEDIUM',
      category: alertData.category || 'PERFORMANCE',
      metric_name: alertData.metric_name || 'unknown',
      current_value: alertData.current_value || 0,
      threshold_value: alertData.threshold_value || 0,
      session_id: alertData.session_id,
      message: alertData.message || 'Threshold exceeded',
      recommended_action: alertData.recommended_action || 'Investigate and resolve',
      auto_resolution_attempted: false,
    };

    // Check if similar alert already exists (deduplication)
    const existingAlert = Array.from(this.activeAlerts.values()).find(a => 
      a.metric_name === alert.metric_name && 
      a.session_id === alert.session_id &&
      Math.abs(Date.now() - new Date(a.timestamp).getTime()) < 5 * 60 * 1000 // 5 minutes
    );

    if (!existingAlert) {
      this.activeAlerts.set(alertId, alert);
      console.log(`🚨 ALERT [${alert.severity}] ${alert.category}: ${alert.message}`);
      console.log(`   Recommended: ${alert.recommended_action}`);
      
      // Persist alert to database
      await this.persistAlert(alert);
    }
  }

  private async persistAlert(alert: Alert): Promise<void> {
    // In production, persist to database for historical analysis
    // This is a placeholder for the actual database persistence
    console.log(`💾 Persisting alert ${alert.alert_id} to database`);
  }

  /**
   * Automatic resolution attempts
   */
  private async attemptAutomaticResolutions(): Promise<void> {
    const criticalAlerts = Array.from(this.activeAlerts.values())
      .filter(alert => alert.severity === 'CRITICAL' && !alert.auto_resolution_attempted);

    for (const alert of criticalAlerts) {
      try {
        const resolved = await this.attemptAlertResolution(alert);
        if (resolved) {
          alert.auto_resolution_attempted = true;
          this.activeAlerts.delete(alert.alert_id);
          console.log(`✅ Auto-resolved alert ${alert.alert_id}`);
        } else {
          alert.auto_resolution_attempted = true;
          console.log(`⚠️ Auto-resolution failed for alert ${alert.alert_id}`);
        }
      } catch (error) {
        console.error(`❌ Auto-resolution error for ${alert.alert_id}: ${error}`);
      }
    }
  }

  private async attemptAlertResolution(alert: Alert): Promise<boolean> {
    switch (alert.metric_name) {
      case 'error_rate':
        return await this.attemptErrorRateResolution(alert);
      case 'memory_usage':
        return await this.attemptMemoryResolution(alert);
      case 'primal_dual_gap':
        return await this.attemptGapResolution(alert);
      default:
        return false;
    }
  }

  private async attemptErrorRateResolution(alert: Alert): Promise<boolean> {
    console.log('🔧 Attempting error rate resolution...');
    
    // Example automatic resolutions:
    // 1. Clear problematic cache entries
    // 2. Restart failed services
    // 3. Switch to backup algorithms
    
    // Simulate resolution attempt
    await new Promise(resolve => setTimeout(resolve, 1000));
    return Math.random() > 0.3; // 70% success rate simulation
  }

  private async attemptMemoryResolution(alert: Alert): Promise<boolean> {
    console.log('🔧 Attempting memory usage resolution...');
    
    // Example automatic resolutions:
    // 1. Force garbage collection
    // 2. Clear non-essential caches
    // 3. Reduce concurrent operations
    
    await new Promise(resolve => setTimeout(resolve, 500));
    return Math.random() > 0.5; // 50% success rate simulation
  }

  private async attemptGapResolution(alert: Alert): Promise<boolean> {
    console.log('🔧 Attempting primal-dual gap resolution...');
    
    // Example automatic resolutions:
    // 1. Clear optimizer cache
    // 2. Reset convergence parameters
    // 3. Switch to more conservative algorithm
    
    await new Promise(resolve => setTimeout(resolve, 800));
    return Math.random() > 0.4; // 60% success rate simulation
  }

  /**
   * Dashboard data generation
   */
  async generateDashboardData(): Promise<DashboardData> {
    const timestamp = new Date().toISOString();
    const recentMetrics = this.metricsHistory.slice(-100); // Last 100 data points

    return {
      timestamp,
      overview: this.generateOverviewData(recentMetrics),
      lambda_size_analysis: this.generateLambdaSizeAnalysis(recentMetrics),
      cbu_tracking: this.generateCBUTracking(recentMetrics),
      quality_metrics: this.generateQualityMetrics(recentMetrics),
      alerts_summary: this.generateAlertsSummary(),
    };
  }

  private generateOverviewData(metrics: SystemMetrics[]) {
    const activeSessions = new Set(metrics.map(m => m.session_id)).size;
    const totalRequests = metrics.length;
    
    const systemHealthScores = metrics.map(m => this.calculateSystemHealthScore(m));
    const avgHealthScore = systemHealthScores.reduce((sum, score) => sum + score, 0) / systemHealthScores.length || 0;
    
    const lambdaDistribution = metrics.map(m => m.lambda_value);
    const cbuPerHour = metrics.reduce((sum, m) => sum + m.cbu_consumption, 0);

    return {
      system_health_score: Math.round(avgHealthScore),
      active_sessions: activeSessions,
      total_requests_last_hour: totalRequests,
      current_lambda_distribution: this.calculateDistribution(lambdaDistribution),
      cbu_burn_rate_per_hour: cbuPerHour,
    };
  }

  private generateLambdaSizeAnalysis(metrics: SystemMetrics[]) {
    const lambdaValues = metrics.map(m => m.lambda_value);
    const sizeValues = metrics.map(m => m.context_size_tokens);
    
    const monotonicityScore = this.calculateMonotonicity(lambdaValues, sizeValues);
    const trendSlope = this.calculateTrendSlope(lambdaValues, sizeValues);
    const outlierCount = this.detectOutliers(lambdaValues, sizeValues).length;

    return {
      lambda_values: lambdaValues.slice(-20), // Last 20 values
      size_values: sizeValues.slice(-20),
      monotonicity_score: monotonicityScore,
      trend_slope: trendSlope,
      outlier_count: outlierCount,
    };
  }

  private generateCBUTracking(metrics: SystemMetrics[]) {
    const hourlyConsumption = this.aggregateHourly(metrics.map(m => ({ timestamp: m.timestamp, value: m.cbu_consumption })));
    const efficiencyTrend = metrics.map(m => m.cbu_efficiency);
    const elasticityCurve = this.cbuElasticityHistory.slice(-20);
    
    const totalBudget = 10000; // Example budget
    const totalConsumption = metrics.reduce((sum, m) => sum + m.cbu_consumption, 0);

    return {
      hourly_consumption: hourlyConsumption,
      efficiency_trend: efficiencyTrend,
      elasticity_curve: elasticityCurve,
      budget_utilization: Math.min(1, totalConsumption / totalBudget),
    };
  }

  private generateQualityMetrics(metrics: SystemMetrics[]) {
    return {
      ece_distribution: this.calculateDistribution(metrics.map(m => m.ece_score)),
      ilp_usage_rate: metrics.reduce((sum, m) => sum + m.ilp_incidence, 0) / metrics.length || 0,
      calibration_trend: metrics.map(m => m.ece_score).slice(-20),
      gap_distribution: this.calculateDistribution(metrics.map(m => m.primal_dual_gap)),
    };
  }

  private generateAlertsSummary() {
    const activeAlerts = Array.from(this.activeAlerts.values());
    const criticalActive = activeAlerts.filter(a => a.severity === 'CRITICAL').length;
    const autoResolutionsSuccessful = activeAlerts.filter(a => a.auto_resolution_attempted).length;

    return {
      active_alerts: activeAlerts,
      alerts_last_24h: activeAlerts.length,
      critical_alerts_active: criticalActive,
      auto_resolutions_successful: autoResolutionsSuccessful,
    };
  }

  /**
   * Risk budget management
   */
  private updateRiskBudgetStatus(): void {
    const recentMetrics = this.metricsHistory.slice(-20);
    if (recentMetrics.length === 0) return;

    const riskFactors = this.calculateRiskFactors(recentMetrics);
    const totalRisk = riskFactors.reduce((sum, factor) => sum + factor.contribution_percent, 0);
    
    this.riskBudget.budget_utilization_percent = Math.min(100, totalRisk);
    this.riskBudget.risk_contributors = riskFactors;
    
    // Determine risk level
    if (totalRisk < 25) {
      this.riskBudget.current_risk_level = 'LOW';
    } else if (totalRisk < 50) {
      this.riskBudget.current_risk_level = 'MEDIUM';
    } else if (totalRisk < 75) {
      this.riskBudget.current_risk_level = 'HIGH';
    } else {
      this.riskBudget.current_risk_level = 'CRITICAL';
    }

    // Calculate remaining budget hours
    const burnRate = totalRisk / 24; // Risk per hour
    this.riskBudget.remaining_budget_hours = Math.max(0, (100 - totalRisk) / burnRate);

    // Generate recommendations
    this.riskBudget.recommended_actions = this.generateRiskRecommendations(riskFactors);
  }

  private calculateRiskFactors(metrics: SystemMetrics[]) {
    const factors = [];

    // Performance risk
    const avgLatency = metrics.reduce((sum, m) => sum + m.p95_latency_ms, 0) / metrics.length;
    const latencyRisk = Math.max(0, (avgLatency - this.thresholds.max_p95_latency_ms) / this.thresholds.max_p95_latency_ms * 30);
    
    factors.push({
      factor: 'Performance Degradation',
      contribution_percent: latencyRisk,
      severity: latencyRisk > 20 ? 'HIGH' : latencyRisk > 10 ? 'MEDIUM' : 'LOW' as 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL',
      trend: this.calculateTrend(metrics.slice(-5).map(m => m.p95_latency_ms)),
    });

    // Quality risk
    const avgECE = metrics.reduce((sum, m) => sum + m.ece_score, 0) / metrics.length;
    const qualityRisk = Math.max(0, (avgECE - this.thresholds.max_ece_score) / this.thresholds.max_ece_score * 25);
    
    factors.push({
      factor: 'Quality Degradation',
      contribution_percent: qualityRisk,
      severity: qualityRisk > 15 ? 'HIGH' : qualityRisk > 8 ? 'MEDIUM' : 'LOW' as 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL',
      trend: this.calculateTrend(metrics.slice(-5).map(m => m.ece_score)),
    });

    // System resource risk
    const avgMemory = metrics.reduce((sum, m) => sum + m.memory_usage_mb, 0) / metrics.length;
    const resourceRisk = Math.max(0, (avgMemory - this.thresholds.max_memory_usage_mb) / this.thresholds.max_memory_usage_mb * 20);
    
    factors.push({
      factor: 'Resource Exhaustion',
      contribution_percent: resourceRisk,
      severity: resourceRisk > 15 ? 'CRITICAL' : resourceRisk > 10 ? 'HIGH' : 'LOW' as 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL',
      trend: this.calculateTrend(metrics.slice(-5).map(m => m.memory_usage_mb)),
    });

    return factors;
  }

  private generateRiskRecommendations(riskFactors: any[]): string[] {
    const recommendations: string[] = [];

    for (const factor of riskFactors) {
      if (factor.severity === 'CRITICAL' || factor.severity === 'HIGH') {
        switch (factor.factor) {
          case 'Performance Degradation':
            recommendations.push('Scale up compute capacity immediately');
            recommendations.push('Optimize hot path algorithms');
            break;
          case 'Quality Degradation':
            recommendations.push('Recalibrate quality models');
            recommendations.push('Review recent configuration changes');
            break;
          case 'Resource Exhaustion':
            recommendations.push('Increase memory allocation');
            recommendations.push('Implement aggressive garbage collection');
            break;
        }
      }
    }

    if (recommendations.length === 0) {
      recommendations.push('Continue monitoring - system operating within risk budget');
    }

    return recommendations;
  }

  // Utility methods for calculations and analysis
  
  private calculateSystemHealthScore(metrics: SystemMetrics): number {
    let score = 100;

    // Deduct points for threshold violations
    if (metrics.p95_latency_ms > this.thresholds.max_p95_latency_ms) {
      score -= 20;
    }
    if (metrics.ece_score > this.thresholds.max_ece_score) {
      score -= 15;
    }
    if (metrics.error_rate > this.thresholds.max_error_rate) {
      score -= 30;
    }
    if (metrics.accept_rate < this.thresholds.min_accept_rate) {
      score -= 25;
    }
    if (metrics.primal_dual_gap > this.thresholds.max_primal_dual_gap) {
      score -= 10;
    }

    return Math.max(0, score);
  }

  private calculateMedian(values: number[]): number {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
      ? (sorted[mid - 1] + sorted[mid]) / 2
      : sorted[mid];
  }

  private calculateDistribution(values: number[]): number[] {
    if (values.length === 0) return [0, 0, 0, 0, 0];
    const sorted = [...values].sort((a, b) => a - b);
    const percentiles = [0.1, 0.25, 0.5, 0.75, 0.9];
    return percentiles.map(p => {
      const index = Math.floor(p * (sorted.length - 1));
      return sorted[index];
    });
  }

  private calculateMonotonicity(lambdaValues: number[], sizeValues: number[]): number {
    if (lambdaValues.length <= 1) return 1;
    
    let monotonicPairs = 0;
    let totalPairs = 0;

    for (let i = 0; i < lambdaValues.length - 1; i++) {
      for (let j = i + 1; j < lambdaValues.length; j++) {
        if (lambdaValues[j] > lambdaValues[i]) {
          totalPairs++;
          if (sizeValues[j] >= sizeValues[i]) {
            monotonicPairs++;
          }
        }
      }
    }

    return totalPairs > 0 ? monotonicPairs / totalPairs : 1;
  }

  private calculateTrendSlope(xValues: number[], yValues: number[]): number {
    const n = xValues.length;
    if (n <= 1) return 0;

    const sumX = xValues.reduce((a, b) => a + b, 0);
    const sumY = yValues.reduce((a, b) => a + b, 0);
    const sumXY = xValues.reduce((sum, x, i) => sum + x * yValues[i], 0);
    const sumXX = xValues.reduce((sum, x) => sum + x * x, 0);

    return (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  }

  private detectOutliers(lambdaValues: number[], sizeValues: number[]): number[] {
    // Simple IQR-based outlier detection
    const q1 = this.calculateDistribution(sizeValues)[1];
    const q3 = this.calculateDistribution(sizeValues)[3];
    const iqr = q3 - q1;
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;

    return sizeValues.map((size, i) => i).filter(i => 
      sizeValues[i] < lowerBound || sizeValues[i] > upperBound
    );
  }

  private aggregateHourly(data: Array<{ timestamp: string; value: number }>): number[] {
    // Group by hour and sum values
    const hourlyData = new Map<string, number>();
    
    for (const point of data) {
      const hour = new Date(point.timestamp).toISOString().slice(0, 13); // YYYY-MM-DDTHH
      hourlyData.set(hour, (hourlyData.get(hour) || 0) + point.value);
    }

    return Array.from(hourlyData.values()).slice(-24); // Last 24 hours
  }

  private calculateTrend(values: number[]): 'IMPROVING' | 'STABLE' | 'DEGRADING' {
    if (values.length < 2) return 'STABLE';
    
    const first = values[0];
    const last = values[values.length - 1];
    const change = (last - first) / first;

    if (change > 0.05) return 'DEGRADING'; // 5% worse
    if (change < -0.05) return 'IMPROVING'; // 5% better
    return 'STABLE';
  }

  private updateLambdaTracking(lambdaValue: number): void {
    if (this.lambdaBaseline === null) {
      this.lambdaBaseline = lambdaValue;
    }

    // Update baseline with exponential moving average
    const alpha = 0.1; // Smoothing factor
    this.lambdaBaseline = alpha * lambdaValue + (1 - alpha) * this.lambdaBaseline;
  }

  private updateShadowPriceTracking(shadowPrice: number): void {
    this.shadowPriceWindow.push(shadowPrice);
    
    // Keep window size manageable (last 50 values)
    if (this.shadowPriceWindow.length > 50) {
      this.shadowPriceWindow.shift();
    }
  }

  private updateCBUElasticityTracking(lambda: number, cbuRate: number): void {
    this.cbuElasticityHistory.push({ lambda, cbu_rate: cbuRate });
    
    // Keep history manageable (last 100 points)
    if (this.cbuElasticityHistory.length > 100) {
      this.cbuElasticityHistory.shift();
    }
  }

  // Placeholder methods for actual system integration
  private async getActiveSessionIds(): Promise<string[]> {
    // In production, query active sessions from database
    return ['session_1', 'session_2', 'session_3'];
  }

  private async getCurrentLambda(sessionId: string): Promise<number> {
    // In production, get current lambda from session state
    return 0.8 + Math.random() * 0.4; // Simulated
  }

  private async getContextSizeTokens(sessionId: string): Promise<number> {
    // In production, calculate actual context size
    return Math.floor(2000 + Math.random() * 6000); // Simulated
  }

  private async getCBUConsumption(sessionId: string) {
    // In production, track actual CBU consumption
    return {
      total: Math.floor(100 + Math.random() * 400),
      efficiency: 0.7 + Math.random() * 0.3,
      elasticity: -0.1 + Math.random() * 0.2, // Should be negative
    };
  }

  private async getPerformanceMetrics(sessionId: string) {
    // In production, collect from APM systems
    return {
      p95_latency: 120 + Math.random() * 80,
      throughput: 50 + Math.random() * 100,
      accept_rate: 0.8 + Math.random() * 0.15,
    };
  }

  private async getQualityMetrics(sessionId: string) {
    // In production, calculate from model outputs
    return {
      gap: Math.random() * 0.01, // Should be small
      shadow_price: 0.5 + Math.random() * 0.3,
      ece: Math.random() * 0.12,
      ilp_rate: Math.random() * 0.08,
    };
  }

  private async getSystemHealthMetrics() {
    // In production, get from system monitoring
    return {
      memory_mb: 4000 + Math.random() * 3000,
      cpu_percent: 0.4 + Math.random() * 0.4,
      error_rate: Math.random() * 0.02,
    };
  }

  /**
   * Public API methods
   */

  getCurrentMetrics(): SystemMetrics[] {
    return this.metricsHistory.slice(-10); // Last 10 metrics
  }

  getActiveAlerts(): Alert[] {
    return Array.from(this.activeAlerts.values());
  }

  getRiskBudgetStatus(): RiskBudgetStatus {
    return this.riskBudget;
  }

  async exportMetrics(startTime: string, endTime: string): Promise<SystemMetrics[]> {
    return this.metricsHistory.filter(m => 
      m.timestamp >= startTime && m.timestamp <= endTime
    );
  }
}

/**
 * Utility function to create and start monitoring
 */
export function createProductionMonitoring(
  db: DB,
  thresholds?: Partial<AlertThresholds>
): ProductionMonitoringSystem {
  return new ProductionMonitoringSystem(db, thresholds);
}