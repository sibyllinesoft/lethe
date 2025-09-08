/**
 * 7-day Canary Deployment System with EmbeddingGemma-300M
 * Automated promotion gates based on statistical validation
 */

export interface CanaryConfig {
  duration_days: number; // 7-day default
  traffic_split: number; // % traffic to canary (start with 5%)
  model_name: string; // 'EmbeddingGemma-300M'
  promotion_thresholds: {
    delta_cbu_gb_threshold: number; // ΔCBU/GB ≥ +10% for promotion
    p95_improvement_threshold: number; // P95 improves ≥5ms for promotion
    error_rate_threshold: number; // Max error rate increase
    statistical_power_threshold: number; // O(10⁴-10⁵) turns
  };
  rollback_thresholds: {
    max_error_rate_increase: number; // Immediate rollback trigger
    max_latency_increase: number; // P95 latency increase limit
    max_failure_streak: number; // Consecutive validation failures
  };
}

export interface CanaryMetrics {
  timestamp: number;
  model: string;
  version: string;
  traffic_percentage: number;
  metrics: {
    cbu_per_gb: number;
    p95_latency_ms: number;
    error_rate: number;
    throughput_qps: number;
    memory_usage_mb: number;
    cpu_utilization: number;
  };
  validation_results: {
    dual_sanity_passed: boolean;
    ood_resilience_passed: boolean;
    long_horizon_passed: boolean;
  };
  baseline_comparison: {
    delta_cbu_gb: number; // % improvement in CBU/GB
    delta_p95_ms: number; // ms improvement in P95 latency
    delta_error_rate: number; // % change in error rate
  };
}

export interface CanaryState {
  status: 'INITIALIZING' | 'RUNNING' | 'PROMOTING' | 'PROMOTED' | 'ROLLING_BACK' | 'ROLLED_BACK' | 'FAILED';
  start_time: number;
  current_phase: number; // 0-6 for 7 phases
  traffic_percentage: number;
  metrics_collected: CanaryMetrics[];
  alerts: string[];
  last_validation: number;
  promotion_criteria_met: boolean;
  rollback_triggered: boolean;
}

/**
 * Statistical Validation Engine for Canary Deployment
 * Ensures O(10⁴-10⁵) statistical power for promotion decisions
 */
export class CanaryStatisticalValidator {
  private config: CanaryConfig;
  private baselineMetrics: CanaryMetrics[] = [];

  constructor(config: CanaryConfig) {
    this.config = config;
  }

  /**
   * Initialize baseline metrics from production system
   */
  setBaseline(baselineData: CanaryMetrics[]): void {
    this.baselineMetrics = baselineData;
    console.log(`📊 Baseline established with ${baselineData.length} data points`);
  }

  /**
   * Validate canary performance against baseline with statistical significance
   */
  validateCanaryPerformance(canaryMetrics: CanaryMetrics[]): {
    passed: boolean;
    statistical_power: number;
    confidence: number;
    promotion_ready: boolean;
    detailed_results: {
      cbu_improvement: { significant: boolean; effect_size: number; p_value: number };
      latency_improvement: { significant: boolean; effect_size: number; p_value: number };
      error_rate_check: { passed: boolean; increase: number };
      sample_size_adequate: boolean;
    };
  } {
    if (this.baselineMetrics.length === 0) {
      throw new Error('Baseline metrics not initialized');
    }

    if (canaryMetrics.length < 1000) {
      return {
        passed: false,
        statistical_power: canaryMetrics.length / this.config.promotion_thresholds.statistical_power_threshold,
        confidence: 0,
        promotion_ready: false,
        detailed_results: {
          cbu_improvement: { significant: false, effect_size: 0, p_value: 1 },
          latency_improvement: { significant: false, effect_size: 0, p_value: 1 },
          error_rate_check: { passed: false, increase: 0 },
          sample_size_adequate: false
        }
      };
    }

    // Extract metrics for statistical tests
    const baselineCBU = this.baselineMetrics.map(m => m.metrics.cbu_per_gb);
    const canaryCBU = canaryMetrics.map(m => m.metrics.cbu_per_gb);
    
    const baselineP95 = this.baselineMetrics.map(m => m.metrics.p95_latency_ms);
    const canaryP95 = canaryMetrics.map(m => m.metrics.p95_latency_ms);
    
    const baselineError = this.baselineMetrics.map(m => m.metrics.error_rate);
    const canaryError = canaryMetrics.map(m => m.metrics.error_rate);

    // Perform statistical tests
    const cbuTest = this.performWelchTTest(canaryCBU, baselineCBU);
    const latencyTest = this.performWelchTTest(baselineP95, canaryP95); // Baseline - Canary for improvement
    const errorTest = this.performWelchTTest(canaryError, baselineError);

    // Calculate effect sizes (Cohen's d)
    const cbuEffectSize = this.calculateCohenD(canaryCBU, baselineCBU);
    const latencyEffectSize = this.calculateCohenD(baselineP95, canaryP95);
    
    // Check promotion thresholds
    const avgCBUImprovment = (this.mean(canaryCBU) - this.mean(baselineCBU)) / this.mean(baselineCBU) * 100;
    const avgLatencyImprovement = this.mean(baselineP95) - this.mean(canaryP95);
    const avgErrorIncrease = (this.mean(canaryError) - this.mean(baselineError)) / this.mean(baselineError) * 100;

    const cbuPromotionMet = avgCBUImprovment >= this.config.promotion_thresholds.delta_cbu_gb_threshold;
    const latencyPromotionMet = avgLatencyImprovement >= this.config.promotion_thresholds.p95_improvement_threshold;
    const errorRateOk = avgErrorIncrease <= this.config.promotion_thresholds.error_rate_threshold;
    
    const sampleSizeAdequate = canaryMetrics.length >= this.config.promotion_thresholds.statistical_power_threshold;
    
    const detailed_results = {
      cbu_improvement: {
        significant: cbuTest.p_value < 0.05 && cbuEffectSize > 0.2, // Small effect size threshold
        effect_size: cbuEffectSize,
        p_value: cbuTest.p_value
      },
      latency_improvement: {
        significant: latencyTest.p_value < 0.05 && latencyEffectSize > 0.2,
        effect_size: latencyEffectSize,
        p_value: latencyTest.p_value
      },
      error_rate_check: {
        passed: errorRateOk,
        increase: avgErrorIncrease
      },
      sample_size_adequate: sampleSizeAdequate
    };

    const passed = detailed_results.cbu_improvement.significant || 
                   detailed_results.latency_improvement.significant;
    
    const promotion_ready = (cbuPromotionMet || latencyPromotionMet) && 
                           errorRateOk && 
                           sampleSizeAdequate && 
                           passed;

    const statistical_power = Math.min(1.0, canaryMetrics.length / this.config.promotion_thresholds.statistical_power_threshold);
    const confidence = Math.min(
      1 - Math.max(cbuTest.p_value, latencyTest.p_value),
      statistical_power
    );

    return {
      passed,
      statistical_power,
      confidence,
      promotion_ready,
      detailed_results
    };
  }

  /**
   * Welch's t-test for unequal variances
   */
  private performWelchTTest(sample1: number[], sample2: number[]): { t_statistic: number; p_value: number } {
    const n1 = sample1.length;
    const n2 = sample2.length;
    
    const mean1 = this.mean(sample1);
    const mean2 = this.mean(sample2);
    
    const var1 = this.variance(sample1);
    const var2 = this.variance(sample2);
    
    const se = Math.sqrt(var1 / n1 + var2 / n2);
    const t_statistic = (mean1 - mean2) / se;
    
    // Degrees of freedom for Welch's t-test
    const df = Math.pow(var1 / n1 + var2 / n2, 2) / 
               (Math.pow(var1 / n1, 2) / (n1 - 1) + Math.pow(var2 / n2, 2) / (n2 - 1));
    
    // Approximate p-value using simplified t-distribution
    const p_value = this.approximateTDistributionPValue(Math.abs(t_statistic), df);
    
    return { t_statistic, p_value };
  }

  private mean(arr: number[]): number {
    return arr.reduce((sum, val) => sum + val, 0) / arr.length;
  }

  private variance(arr: number[]): number {
    const m = this.mean(arr);
    return arr.reduce((sum, val) => sum + Math.pow(val - m, 2), 0) / (arr.length - 1);
  }

  private calculateCohenD(sample1: number[], sample2: number[]): number {
    const mean1 = this.mean(sample1);
    const mean2 = this.mean(sample2);
    const pooledStd = Math.sqrt((this.variance(sample1) + this.variance(sample2)) / 2);
    return Math.abs(mean1 - mean2) / pooledStd;
  }

  private approximateTDistributionPValue(t: number, df: number): number {
    // Simplified approximation for p-value calculation
    // For production use, consider using a proper statistical library
    if (df > 100) {
      // Approximate with normal distribution for large df
      return 2 * (1 - this.normalCDF(t));
    }
    
    // Rough approximation for t-distribution
    const x = t / Math.sqrt(df);
    return 2 * (1 - this.normalCDF(x * Math.sqrt(df)));
  }

  private normalCDF(x: number): number {
    // Approximate normal CDF using error function approximation
    return 0.5 * (1 + this.erf(x / Math.sqrt(2)));
  }

  private erf(x: number): number {
    // Approximation of error function
    const a1 =  0.254829592;
    const a2 = -0.284496736;
    const a3 =  1.421413741;
    const a4 = -1.453152027;
    const a5 =  1.061405429;
    const p  =  0.3275911;
    
    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x);
    
    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    
    return sign * y;
  }
}

/**
 * Canary Deployment Controller
 * Manages the full 7-day canary lifecycle with automated promotion/rollback
 */
export class CanaryDeploymentController {
  private config: CanaryConfig;
  private validator: CanaryStatisticalValidator;
  private state: CanaryState;
  private intervalId?: NodeJS.Timeout;

  constructor(config: CanaryConfig) {
    this.config = config;
    this.validator = new CanaryStatisticalValidator(config);
    this.state = {
      status: 'INITIALIZING',
      start_time: 0,
      current_phase: 0,
      traffic_percentage: 5, // Start with 5% traffic
      metrics_collected: [],
      alerts: [],
      last_validation: 0,
      promotion_criteria_met: false,
      rollback_triggered: false
    };
  }

  /**
   * Start canary deployment
   */
  async startCanary(baselineMetrics: CanaryMetrics[]): Promise<void> {
    console.log('🚀 Starting 7-day canary deployment with EmbeddingGemma-300M');
    
    // Initialize validator with baseline
    this.validator.setBaseline(baselineMetrics);
    
    // Update state
    this.state = {
      ...this.state,
      status: 'RUNNING',
      start_time: Date.now(),
      current_phase: 0,
      traffic_percentage: 5
    };

    // Start monitoring loop
    this.startMonitoringLoop();
    
    console.log(`📈 Canary deployment started - Phase ${this.state.current_phase + 1}/7`);
    console.log(`🎯 Traffic split: ${this.state.traffic_percentage}%`);
  }

  /**
   * Process new canary metrics
   */
  addMetrics(metrics: CanaryMetrics): void {
    if (this.state.status !== 'RUNNING' && this.state.status !== 'PROMOTING') {
      return;
    }

    this.state.metrics_collected.push(metrics);
    
    // Check for immediate rollback conditions
    this.checkRollbackConditions(metrics);
    
    // Limit metrics history to prevent memory issues
    if (this.state.metrics_collected.length > 100000) {
      this.state.metrics_collected = this.state.metrics_collected.slice(-50000);
    }
  }

  /**
   * Get current canary status
   */
  getStatus(): {
    state: CanaryState;
    time_remaining_hours: number;
    next_validation_in_ms: number;
    current_traffic_percentage: number;
    recommendation: string;
  } {
    const elapsed_ms = Date.now() - this.state.start_time;
    const total_duration_ms = this.config.duration_days * 24 * 60 * 60 * 1000;
    const time_remaining_hours = Math.max(0, (total_duration_ms - elapsed_ms) / (60 * 60 * 1000));
    
    const next_validation = this.getNextValidationTime();
    const next_validation_in_ms = Math.max(0, next_validation - Date.now());
    
    let recommendation = '';
    if (this.state.promotion_criteria_met && time_remaining_hours < 1) {
      recommendation = 'Ready for promotion - all criteria met';
    } else if (this.state.rollback_triggered) {
      recommendation = 'Rollback recommended due to performance degradation';
    } else if (this.state.metrics_collected.length < 1000) {
      recommendation = `Collecting metrics - need ${1000 - this.state.metrics_collected.length} more samples`;
    } else {
      recommendation = 'Canary running normally - monitoring continues';
    }

    return {
      state: this.state,
      time_remaining_hours,
      next_validation_in_ms,
      current_traffic_percentage: this.state.traffic_percentage,
      recommendation
    };
  }

  /**
   * Force promotion (manual override)
   */
  async forcePromotion(): Promise<{ success: boolean; message: string }> {
    if (this.state.status !== 'RUNNING') {
      return { success: false, message: `Cannot promote from status: ${this.state.status}` };
    }

    if (this.state.metrics_collected.length < 1000) {
      return { success: false, message: `Insufficient metrics: ${this.state.metrics_collected.length} < 1000` };
    }

    console.log('🚀 Force promoting canary deployment');
    await this.promoteCanary();
    return { success: true, message: 'Canary promoted successfully' };
  }

  /**
   * Force rollback (manual override)
   */
  async forceRollback(reason: string): Promise<{ success: boolean; message: string }> {
    if (this.state.status === 'ROLLED_BACK' || this.state.status === 'FAILED') {
      return { success: false, message: `Already in status: ${this.state.status}` };
    }

    console.log(`🔄 Force rolling back canary deployment: ${reason}`);
    this.state.alerts.push(`Manual rollback: ${reason}`);
    await this.rollbackCanary();
    return { success: true, message: 'Canary rolled back successfully' };
  }

  private startMonitoringLoop(): void {
    // Check every 5 minutes
    this.intervalId = setInterval(() => {
      this.performPeriodicCheck();
    }, 5 * 60 * 1000);
  }

  private async performPeriodicCheck(): Promise<void> {
    if (this.state.status !== 'RUNNING') return;

    const now = Date.now();
    const elapsed_hours = (now - this.state.start_time) / (60 * 60 * 1000);
    
    // Update phase based on elapsed time (7 phases over 7 days)
    const new_phase = Math.min(6, Math.floor(elapsed_hours / 24));
    if (new_phase > this.state.current_phase) {
      this.state.current_phase = new_phase;
      this.updateTrafficSplit();
      console.log(`📈 Canary advanced to Phase ${new_phase + 1}/7`);
      console.log(`🎯 Traffic split: ${this.state.traffic_percentage}%`);
    }

    // Perform validation if enough time has passed and we have sufficient data
    const should_validate = (now - this.state.last_validation) >= 60 * 60 * 1000 && // 1 hour between validations
                           this.state.metrics_collected.length >= 1000;

    if (should_validate) {
      await this.performValidation();
    }

    // Check if canary period is complete
    if (elapsed_hours >= this.config.duration_days * 24) {
      if (this.state.promotion_criteria_met) {
        console.log('✅ Canary period complete - promoting');
        await this.promoteCanary();
      } else {
        console.log('❌ Canary period complete but promotion criteria not met - rolling back');
        await this.rollbackCanary();
      }
    }
  }

  private updateTrafficSplit(): void {
    // Gradually increase traffic split over the 7 phases
    const trafficSchedule = [5, 10, 20, 35, 50, 75, 100];
    this.state.traffic_percentage = trafficSchedule[this.state.current_phase];
  }

  private async performValidation(): Promise<void> {
    console.log('🔍 Performing canary validation...');
    
    try {
      const validation = this.validator.validateCanaryPerformance(this.state.metrics_collected);
      this.state.last_validation = Date.now();
      
      console.log(`📊 Validation results:
        - Statistical Power: ${(validation.statistical_power * 100).toFixed(1)}%
        - Confidence: ${(validation.confidence * 100).toFixed(1)}%
        - CBU Improvement Significant: ${validation.detailed_results.cbu_improvement.significant}
        - Latency Improvement Significant: ${validation.detailed_results.latency_improvement.significant}
        - Error Rate Check: ${validation.detailed_results.error_rate_check.passed}
      `);

      // Update promotion readiness
      this.state.promotion_criteria_met = validation.promotion_ready;
      
      if (validation.promotion_ready) {
        this.state.alerts.push('✅ Promotion criteria met - ready for deployment');
      } else if (!validation.detailed_results.error_rate_check.passed) {
        this.state.alerts.push(`⚠️ Error rate increase: ${validation.detailed_results.error_rate_check.increase.toFixed(2)}%`);
      }

    } catch (error) {
      console.error('❌ Validation failed:', error);
      this.state.alerts.push(`Validation error: ${error}`);
    }
  }

  private checkRollbackConditions(metrics: CanaryMetrics): void {
    // Check immediate rollback conditions
    const baseline_avg_error = 0.01; // Assume 1% baseline error rate
    const error_increase = (metrics.metrics.error_rate - baseline_avg_error) / baseline_avg_error;
    
    if (error_increase > this.config.rollback_thresholds.max_error_rate_increase) {
      console.log(`🚨 Immediate rollback triggered - error rate increase: ${(error_increase * 100).toFixed(2)}%`);
      this.state.rollback_triggered = true;
      this.rollbackCanary();
      return;
    }

    // Check latency increase
    const baseline_avg_latency = 50; // Assume 50ms baseline P95
    const latency_increase = (metrics.metrics.p95_latency_ms - baseline_avg_latency) / baseline_avg_latency;
    
    if (latency_increase > this.config.rollback_thresholds.max_latency_increase) {
      console.log(`🚨 Immediate rollback triggered - latency increase: ${(latency_increase * 100).toFixed(2)}%`);
      this.state.rollback_triggered = true;
      this.rollbackCanary();
      return;
    }

    // Check validation failures
    const failed_validations = this.state.metrics_collected
      .slice(-10) // Last 10 metrics
      .filter(m => !m.validation_results.dual_sanity_passed || 
                   !m.validation_results.ood_resilience_passed ||
                   !m.validation_results.long_horizon_passed)
      .length;
      
    if (failed_validations >= this.config.rollback_thresholds.max_failure_streak) {
      console.log(`🚨 Rollback triggered - ${failed_validations} consecutive validation failures`);
      this.state.rollback_triggered = true;
      this.rollbackCanary();
      return;
    }
  }

  private async promoteCanary(): Promise<void> {
    this.state.status = 'PROMOTING';
    
    try {
      console.log('🚀 Promoting canary to production...');
      
      // In a real implementation, this would trigger infrastructure changes
      // to route 100% traffic to the canary version
      
      // Simulate promotion process
      await this.simulatePromotion();
      
      this.state.status = 'PROMOTED';
      this.state.traffic_percentage = 100;
      
      this.stopMonitoring();
      
      console.log('✅ Canary promoted successfully to production');
      
    } catch (error) {
      console.error('❌ Promotion failed:', error);
      this.state.status = 'FAILED';
      this.state.alerts.push(`Promotion failed: ${error}`);
    }
  }

  private async rollbackCanary(): Promise<void> {
    this.state.status = 'ROLLING_BACK';
    
    try {
      console.log('🔄 Rolling back canary deployment...');
      
      // In a real implementation, this would trigger infrastructure changes
      // to route 100% traffic back to the baseline version
      
      // Simulate rollback process
      await this.simulateRollback();
      
      this.state.status = 'ROLLED_BACK';
      this.state.traffic_percentage = 0;
      
      this.stopMonitoring();
      
      console.log('✅ Canary rolled back successfully');
      
    } catch (error) {
      console.error('❌ Rollback failed:', error);
      this.state.status = 'FAILED';
      this.state.alerts.push(`Rollback failed: ${error}`);
    }
  }

  private async simulatePromotion(): Promise<void> {
    // Simulate the time it takes for infrastructure changes
    await new Promise(resolve => setTimeout(resolve, 5000));
  }

  private async simulateRollback(): Promise<void> {
    // Simulate the time it takes for infrastructure changes
    await new Promise(resolve => setTimeout(resolve, 3000));
  }

  private getNextValidationTime(): number {
    return this.state.last_validation + (60 * 60 * 1000); // Every hour
  }

  private stopMonitoring(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = undefined;
    }
  }

  /**
   * Health check for canary system
   */
  healthCheck(): {
    healthy: boolean;
    issues: string[];
    status: CanaryState['status'];
    metrics_count: number;
  } {
    const issues: string[] = [];
    
    if (this.state.status === 'FAILED') {
      issues.push('Canary deployment failed');
    }
    
    if (this.state.rollback_triggered && this.state.status === 'RUNNING') {
      issues.push('Rollback triggered but not executed');
    }
    
    const recent_metrics = this.state.metrics_collected.filter(
      m => Date.now() - m.timestamp < 10 * 60 * 1000 // Last 10 minutes
    );
    
    if (this.state.status === 'RUNNING' && recent_metrics.length === 0) {
      issues.push('No recent metrics received');
    }
    
    return {
      healthy: issues.length === 0,
      issues,
      status: this.state.status,
      metrics_count: this.state.metrics_collected.length
    };
  }

  /**
   * Export canary deployment report
   */
  exportReport(): {
    deployment_summary: {
      model: string;
      start_time: number;
      duration_hours: number;
      final_status: CanaryState['status'];
      traffic_phases: number[];
      total_metrics: number;
    };
    performance_summary: {
      avg_cbu_improvement: number;
      avg_latency_improvement: number;
      error_rate_change: number;
      promotion_criteria_met: boolean;
    };
    timeline: Array<{
      phase: number;
      start_time: number;
      traffic_percentage: number;
      metrics_count: number;
      alerts: string[];
    }>;
  } {
    const elapsed_hours = (Date.now() - this.state.start_time) / (60 * 60 * 1000);
    
    // Calculate performance summary
    let avg_cbu_improvement = 0;
    let avg_latency_improvement = 0;
    let error_rate_change = 0;
    
    if (this.state.metrics_collected.length > 0) {
      const recent_metrics = this.state.metrics_collected.slice(-1000);
      avg_cbu_improvement = recent_metrics.reduce((sum, m) => sum + m.baseline_comparison.delta_cbu_gb, 0) / recent_metrics.length;
      avg_latency_improvement = recent_metrics.reduce((sum, m) => sum + m.baseline_comparison.delta_p95_ms, 0) / recent_metrics.length;
      error_rate_change = recent_metrics.reduce((sum, m) => sum + m.baseline_comparison.delta_error_rate, 0) / recent_metrics.length;
    }
    
    return {
      deployment_summary: {
        model: this.config.model_name,
        start_time: this.state.start_time,
        duration_hours: elapsed_hours,
        final_status: this.state.status,
        traffic_phases: [5, 10, 20, 35, 50, 75, 100].slice(0, this.state.current_phase + 1),
        total_metrics: this.state.metrics_collected.length
      },
      performance_summary: {
        avg_cbu_improvement,
        avg_latency_improvement,
        error_rate_change,
        promotion_criteria_met: this.state.promotion_criteria_met
      },
      timeline: Array.from({ length: this.state.current_phase + 1 }, (_, phase) => ({
        phase,
        start_time: this.state.start_time + (phase * 24 * 60 * 60 * 1000),
        traffic_percentage: [5, 10, 20, 35, 50, 75, 100][phase],
        metrics_count: this.state.metrics_collected.filter(
          m => m.timestamp >= this.state.start_time + (phase * 24 * 60 * 60 * 1000) &&
               m.timestamp < this.state.start_time + ((phase + 1) * 24 * 60 * 60 * 1000)
        ).length,
        alerts: this.state.alerts.filter((_, i) => i >= phase * 2 && i < (phase + 1) * 2) // Rough alert distribution
      }))
    };
  }
}