/**
 * Quality Gate Enforcement System
 * ECE ≤ 0.08, ILP ≤ 5%, λ-drift bounds maintained, automated rollback triggers
 */

export interface QualityGateConfig {
  ece_threshold: number; // Expected Calibration Error ≤ 0.08
  ilp_threshold: number; // Information Leakage Percentage ≤ 5% (0.05)
  lambda_drift_bounds: [number, number]; // λ-drift acceptable range
  performance_thresholds: {
    delta_cbu_gb_min: number; // ΔCBU/GB ≥ +10% for promotion
    p95_improvement_min: number; // P95 improves ≥5ms for promotion
    error_rate_max_increase: number; // Max allowable error rate increase
  };
  statistical_requirements: {
    min_sample_size: number; // O(10⁴-10⁵) turns
    confidence_level: number; // 80% confidence minimum
  };
  enforcement_actions: {
    warning_threshold_percentage: number; // When to issue warnings
    blocking_threshold_percentage: number; // When to block deployments
    emergency_rollback_threshold_percentage: number; // Immediate rollback
  };
}

export interface QualityMetrics {
  timestamp: number;
  ece: number; // Expected Calibration Error
  ilp: number; // Information Leakage Percentage
  lambda: number; // Current λ value
  delta_cbu_gb: number; // ΔCBU/GB performance improvement
  p95_latency_delta: number; // P95 latency improvement (ms)
  error_rate_delta: number; // Error rate change (%)
  sample_size: number; // Number of data points
  confidence: number; // Statistical confidence level
  coverage_weighted_crps: number; // Uncertainty quantification metric
}

export interface QualityGateResult {
  gate_id: string;
  timestamp: number;
  status: 'PASSED' | 'WARNING' | 'FAILED' | 'BLOCKED' | 'EMERGENCY_ROLLBACK';
  overall_score: number; // 0-1, composite quality score
  individual_scores: {
    ece_score: number; // 0-1, where 1 = perfect calibration
    ilp_score: number; // 0-1, where 1 = no information leakage
    lambda_drift_score: number; // 0-1, where 1 = within bounds
    performance_score: number; // 0-1, based on improvements
    statistical_power_score: number; // 0-1, based on sample size
  };
  violations: Array<{
    metric: string;
    current_value: number;
    threshold: number;
    severity: 'WARNING' | 'CRITICAL' | 'EMERGENCY';
    recommendation: string;
  }>;
  actions_taken: string[];
  next_evaluation_time: number;
}

export interface QualityGateEnforcementAction {
  action_type: 'LOG_WARNING' | 'BLOCK_DEPLOYMENT' | 'TRIGGER_ROLLBACK' | 'ALERT_TEAM' | 'AUTO_SCALE';
  triggered_by: string; // Which violation triggered this action
  timestamp: number;
  details: Record<string, any>;
  success: boolean;
  execution_time_ms: number;
}

/**
 * Expected Calibration Error (ECE) Calculator
 * Measures calibration quality of confidence estimates
 */
export class ECECalculator {
  private readonly NUM_BINS = 10;

  /**
   * Calculate Expected Calibration Error
   */
  calculateECE(
    predictions: Array<{ confidence: number; actual: boolean }>
  ): { ece: number; bin_stats: Array<{ bin_range: [number, number]; accuracy: number; confidence: number; count: number }> } {
    if (predictions.length === 0) {
      return { ece: 1.0, bin_stats: [] };
    }

    const binSize = 1.0 / this.NUM_BINS;
    const bins: Array<{ confidences: number[]; accuracies: number[] }> = Array(this.NUM_BINS).fill(null).map(() => ({ confidences: [], accuracies: [] }));
    
    // Assign predictions to bins
    for (const pred of predictions) {
      const binIndex = Math.min(this.NUM_BINS - 1, Math.floor(pred.confidence / binSize));
      bins[binIndex].confidences.push(pred.confidence);
      bins[binIndex].accuracies.push(pred.actual ? 1 : 0);
    }

    let ece = 0;
    const binStats = [];
    const totalSamples = predictions.length;

    // Calculate ECE across all bins
    for (let i = 0; i < this.NUM_BINS; i++) {
      const bin = bins[i];
      
      if (bin.confidences.length === 0) continue;

      const avgConfidence = bin.confidences.reduce((sum, c) => sum + c, 0) / bin.confidences.length;
      const accuracy = bin.accuracies.reduce((sum, a) => sum + a, 0) / bin.accuracies.length;
      const binWeight = bin.confidences.length / totalSamples;
      
      ece += binWeight * Math.abs(avgConfidence - accuracy);
      
      binStats.push({
        bin_range: [i * binSize, (i + 1) * binSize] as [number, number],
        accuracy,
        confidence: avgConfidence,
        count: bin.confidences.length
      });
    }

    return { ece, bin_stats: binStats };
  }

  /**
   * Calculate calibration reliability diagram data
   */
  getCalibrationDiagram(
    predictions: Array<{ confidence: number; actual: boolean }>
  ): { perfect_calibration_line: number[]; actual_calibration: number[]; bin_counts: number[] } {
    const { bin_stats } = this.calculateECE(predictions);
    
    const perfectLine = Array.from({ length: this.NUM_BINS }, (_, i) => (i + 0.5) / this.NUM_BINS);
    const actualCalibration = bin_stats.map(stat => stat.accuracy);
    const binCounts = bin_stats.map(stat => stat.count);
    
    return {
      perfect_calibration_line: perfectLine,
      actual_calibration: actualCalibration,
      bin_counts: binCounts
    };
  }
}

/**
 * Information Leakage Percentage (ILP) Calculator
 * Measures how much model relies on spurious correlations
 */
export class ILPCalculator {
  /**
   * Calculate Information Leakage Percentage
   */
  calculateILP(
    inDistributionResults: Array<{ input: any; output: any; confidence: number }>,
    outOfDistributionResults: Array<{ input: any; output: any; confidence: number; shift_type: string }>
  ): {
    ilp: number;
    breakdown: {
      id_performance: number;
      ood_performance: number;
      performance_drop: number;
      leakage_sources: Array<{ shift_type: string; leakage: number }>;
    };
  } {
    if (inDistributionResults.length === 0 || outOfDistributionResults.length === 0) {
      return {
        ilp: 0,
        breakdown: {
          id_performance: 0,
          ood_performance: 0,
          performance_drop: 0,
          leakage_sources: []
        }
      };
    }

    // Calculate in-distribution performance
    const idPerformance = inDistributionResults.reduce((sum, r) => sum + r.confidence, 0) / inDistributionResults.length;
    
    // Calculate out-of-distribution performance
    const oodPerformance = outOfDistributionResults.reduce((sum, r) => sum + r.confidence, 0) / outOfDistributionResults.length;
    
    // Information leakage is the performance drop from ID to OOD
    const performanceDrop = (idPerformance - oodPerformance) / idPerformance;
    const ilp = Math.max(0, performanceDrop);

    // Break down leakage by shift type
    const shiftTypes = [...new Set(outOfDistributionResults.map(r => r.shift_type))];
    const leakageSources = shiftTypes.map(shiftType => {
      const shiftResults = outOfDistributionResults.filter(r => r.shift_type === shiftType);
      const shiftPerformance = shiftResults.reduce((sum, r) => sum + r.confidence, 0) / shiftResults.length;
      const shiftLeakage = Math.max(0, (idPerformance - shiftPerformance) / idPerformance);
      
      return { shift_type: shiftType, leakage: shiftLeakage };
    });

    return {
      ilp,
      breakdown: {
        id_performance: idPerformance,
        ood_performance: oodPerformance,
        performance_drop: performanceDrop,
        leakage_sources: leakageSources
      }
    };
  }

  /**
   * Identify most problematic shift types for information leakage
   */
  identifyProblematicShifts(
    leakageSources: Array<{ shift_type: string; leakage: number }>
  ): Array<{ shift_type: string; leakage: number; severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' }> {
    return leakageSources.map(source => {
      let severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
      
      if (source.leakage < 0.1) severity = 'LOW';
      else if (source.leakage < 0.25) severity = 'MEDIUM';
      else if (source.leakage < 0.5) severity = 'HIGH';
      else severity = 'CRITICAL';
      
      return { ...source, severity };
    }).sort((a, b) => b.leakage - a.leakage);
  }
}

/**
 * Lambda Drift Monitor
 * Monitors λ parameter drift and bounds compliance
 */
export class LambdaDriftMonitor {
  private lambdaHistory: Array<{ timestamp: number; lambda: number }> = [];
  private bounds: [number, number];

  constructor(bounds: [number, number]) {
    this.bounds = bounds;
  }

  /**
   * Add lambda measurement
   */
  addLambdaMeasurement(lambda: number): void {
    this.lambdaHistory.push({
      timestamp: Date.now(),
      lambda
    });

    // Keep only last 1000 measurements to prevent memory issues
    if (this.lambdaHistory.length > 1000) {
      this.lambdaHistory = this.lambdaHistory.slice(-500);
    }
  }

  /**
   * Check if lambda is within acceptable bounds
   */
  checkBoundsCompliance(): {
    within_bounds: boolean;
    current_lambda: number;
    bounds: [number, number];
    deviation: number;
    trend: 'STABLE' | 'INCREASING' | 'DECREASING' | 'VOLATILE';
  } {
    if (this.lambdaHistory.length === 0) {
      return {
        within_bounds: true,
        current_lambda: 0,
        bounds: this.bounds,
        deviation: 0,
        trend: 'STABLE'
      };
    }

    const currentLambda = this.lambdaHistory[this.lambdaHistory.length - 1].lambda;
    const withinBounds = currentLambda >= this.bounds[0] && currentLambda <= this.bounds[1];
    
    // Calculate deviation from bounds (0 if within bounds)
    let deviation = 0;
    if (currentLambda < this.bounds[0]) {
      deviation = (this.bounds[0] - currentLambda) / this.bounds[0];
    } else if (currentLambda > this.bounds[1]) {
      deviation = (currentLambda - this.bounds[1]) / this.bounds[1];
    }

    // Calculate trend
    const trend = this.calculateLambdaTrend();

    return {
      within_bounds: withinBounds,
      current_lambda: currentLambda,
      bounds: this.bounds,
      deviation,
      trend
    };
  }

  /**
   * Get lambda drift statistics
   */
  getLambdaDriftStats(windowSize: number = 100): {
    mean: number;
    std_dev: number;
    min: number;
    max: number;
    drift_rate: number; // Change per time unit
    volatility: number; // Coefficient of variation
  } {
    if (this.lambdaHistory.length === 0) {
      return {
        mean: 0,
        std_dev: 0,
        min: 0,
        max: 0,
        drift_rate: 0,
        volatility: 0
      };
    }

    const recentHistory = this.lambdaHistory.slice(-windowSize);
    const lambdaValues = recentHistory.map(h => h.lambda);
    
    const mean = lambdaValues.reduce((sum, l) => sum + l, 0) / lambdaValues.length;
    const variance = lambdaValues.reduce((sum, l) => sum + Math.pow(l - mean, 2), 0) / lambdaValues.length;
    const stdDev = Math.sqrt(variance);
    const min = Math.min(...lambdaValues);
    const max = Math.max(...lambdaValues);
    
    // Calculate drift rate (slope of linear regression)
    let driftRate = 0;
    if (recentHistory.length > 1) {
      const timeDeltas = [];
      const lambdaDeltas = [];
      
      for (let i = 1; i < recentHistory.length; i++) {
        timeDeltas.push(recentHistory[i].timestamp - recentHistory[i-1].timestamp);
        lambdaDeltas.push(recentHistory[i].lambda - recentHistory[i-1].lambda);
      }
      
      if (timeDeltas.length > 0) {
        const avgTimeDelta = timeDeltas.reduce((sum, t) => sum + t, 0) / timeDeltas.length;
        const avgLambdaDelta = lambdaDeltas.reduce((sum, l) => sum + l, 0) / lambdaDeltas.length;
        driftRate = avgLambdaDelta / avgTimeDelta; // Lambda change per ms
      }
    }
    
    const volatility = mean > 0 ? stdDev / mean : 0;

    return {
      mean,
      std_dev: stdDev,
      min,
      max,
      drift_rate: driftRate,
      volatility
    };
  }

  private calculateLambdaTrend(): 'STABLE' | 'INCREASING' | 'DECREASING' | 'VOLATILE' {
    if (this.lambdaHistory.length < 10) return 'STABLE';
    
    const recent = this.lambdaHistory.slice(-20); // Last 20 measurements
    const values = recent.map(h => h.lambda);
    
    // Calculate linear trend
    const n = values.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    
    for (let i = 0; i < n; i++) {
      sumX += i;
      sumY += values[i];
      sumXY += i * values[i];
      sumXX += i * i;
    }
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    
    // Calculate volatility (coefficient of variation)
    const mean = sumY / n;
    const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / n;
    const stdDev = Math.sqrt(variance);
    const volatility = mean > 0 ? stdDev / mean : 0;
    
    // Determine trend
    if (volatility > 0.2) return 'VOLATILE';
    if (Math.abs(slope) < 0.001) return 'STABLE';
    return slope > 0 ? 'INCREASING' : 'DECREASING';
  }
}

/**
 * Quality Gate Enforcement Engine
 * Coordinates all quality checks and enforcement actions
 */
export class QualityGateEnforcementEngine {
  private config: QualityGateConfig;
  private eceCalculator: ECECalculator;
  private ilpCalculator: ILPCalculator;
  private lambdaDriftMonitor: LambdaDriftMonitor;
  private enforcementHistory: QualityGateResult[] = [];
  private actionHistory: QualityGateEnforcementAction[] = [];

  constructor(config: QualityGateConfig) {
    this.config = config;
    this.eceCalculator = new ECECalculator();
    this.ilpCalculator = new ILPCalculator();
    this.lambdaDriftMonitor = new LambdaDriftMonitor(config.lambda_drift_bounds);
  }

  /**
   * Evaluate all quality gates for given metrics
   */
  async evaluateQualityGates(metrics: QualityMetrics): Promise<QualityGateResult> {
    const gateId = `quality-gate-${Date.now()}`;
    const timestamp = Date.now();
    
    console.log(`🚦 Evaluating quality gates for metrics at ${new Date(timestamp).toISOString()}`);

    // Update lambda drift monitoring
    this.lambdaDriftMonitor.addLambdaMeasurement(metrics.lambda);

    // Calculate individual scores
    const eceScore = this.calculateECEScore(metrics.ece);
    const ilpScore = this.calculateILPScore(metrics.ilp);
    const lambdaDriftScore = this.calculateLambdaDriftScore();
    const performanceScore = this.calculatePerformanceScore(metrics);
    const statisticalPowerScore = this.calculateStatisticalPowerScore(metrics);

    // Calculate overall composite score
    const weights = { ece: 0.25, ilp: 0.25, lambda: 0.2, performance: 0.2, statistical: 0.1 };
    const overallScore = (
      eceScore * weights.ece +
      ilpScore * weights.ilp +
      lambdaDriftScore * weights.lambda +
      performanceScore * weights.performance +
      statisticalPowerScore * weights.statistical
    );

    // Check for violations
    const violations = await this.checkViolations(metrics);
    
    // Determine overall status
    const status = this.determineGateStatus(overallScore, violations);
    
    // Create result
    const result: QualityGateResult = {
      gate_id: gateId,
      timestamp,
      status,
      overall_score: overallScore,
      individual_scores: {
        ece_score: eceScore,
        ilp_score: ilpScore,
        lambda_drift_score: lambdaDriftScore,
        performance_score: performanceScore,
        statistical_power_score: statisticalPowerScore
      },
      violations,
      actions_taken: [],
      next_evaluation_time: timestamp + (5 * 60 * 1000) // Next evaluation in 5 minutes
    };

    // Execute enforcement actions
    if (violations.length > 0) {
      const actions = await this.executeEnforcementActions(violations);
      result.actions_taken = actions.map(a => `${a.action_type}: ${a.triggered_by}`);
    }

    // Store result
    this.enforcementHistory.push(result);
    
    // Cleanup old history
    if (this.enforcementHistory.length > 1000) {
      this.enforcementHistory = this.enforcementHistory.slice(-500);
    }

    console.log(`🚦 Quality gate evaluation complete: ${status} (Score: ${overallScore.toFixed(3)})`);
    if (violations.length > 0) {
      console.log(`⚠️ Found ${violations.length} violations, executed ${result.actions_taken.length} actions`);
    }

    return result;
  }

  private calculateECEScore(ece: number): number {
    // ECE score: 1.0 when ECE = 0, 0.0 when ECE >= threshold
    return Math.max(0, 1 - (ece / this.config.ece_threshold));
  }

  private calculateILPScore(ilp: number): number {
    // ILP score: 1.0 when ILP = 0, 0.0 when ILP >= threshold  
    return Math.max(0, 1 - (ilp / this.config.ilp_threshold));
  }

  private calculateLambdaDriftScore(): number {
    const boundsCheck = this.lambdaDriftMonitor.checkBoundsCompliance();
    
    if (boundsCheck.within_bounds) {
      return 1.0;
    } else {
      // Score decreases with deviation from bounds
      return Math.max(0, 1 - boundsCheck.deviation);
    }
  }

  private calculatePerformanceScore(metrics: QualityMetrics): number {
    let score = 0;
    let factors = 0;
    
    // ΔCBU/GB improvement score
    if (metrics.delta_cbu_gb >= this.config.performance_thresholds.delta_cbu_gb_min) {
      score += 1;
    } else if (metrics.delta_cbu_gb > 0) {
      score += metrics.delta_cbu_gb / this.config.performance_thresholds.delta_cbu_gb_min;
    }
    factors++;
    
    // P95 latency improvement score
    if (metrics.p95_latency_delta >= this.config.performance_thresholds.p95_improvement_min) {
      score += 1;
    } else if (metrics.p95_latency_delta > 0) {
      score += metrics.p95_latency_delta / this.config.performance_thresholds.p95_improvement_min;
    }
    factors++;
    
    // Error rate constraint score
    if (metrics.error_rate_delta <= this.config.performance_thresholds.error_rate_max_increase) {
      score += 1;
    } else {
      score += Math.max(0, 1 - (metrics.error_rate_delta / this.config.performance_thresholds.error_rate_max_increase));
    }
    factors++;
    
    return factors > 0 ? score / factors : 0;
  }

  private calculateStatisticalPowerScore(metrics: QualityMetrics): number {
    let score = 0;
    let factors = 0;
    
    // Sample size score
    const sampleScore = Math.min(1, metrics.sample_size / this.config.statistical_requirements.min_sample_size);
    score += sampleScore;
    factors++;
    
    // Confidence level score
    const confidenceScore = Math.min(1, metrics.confidence / this.config.statistical_requirements.confidence_level);
    score += confidenceScore;
    factors++;
    
    return factors > 0 ? score / factors : 0;
  }

  private async checkViolations(metrics: QualityMetrics): Promise<QualityGateResult['violations']> {
    const violations: QualityGateResult['violations'] = [];
    
    // ECE violation
    if (metrics.ece > this.config.ece_threshold) {
      const severity = metrics.ece > this.config.ece_threshold * 2 ? 'EMERGENCY' : 'CRITICAL';
      violations.push({
        metric: 'ECE',
        current_value: metrics.ece,
        threshold: this.config.ece_threshold,
        severity,
        recommendation: 'Improve model calibration through temperature scaling or recalibration'
      });
    }
    
    // ILP violation
    if (metrics.ilp > this.config.ilp_threshold) {
      const severity = metrics.ilp > this.config.ilp_threshold * 2 ? 'EMERGENCY' : 'CRITICAL';
      violations.push({
        metric: 'ILP',
        current_value: metrics.ilp,
        threshold: this.config.ilp_threshold,
        severity,
        recommendation: 'Address spurious correlations in training data or use domain-adversarial training'
      });
    }
    
    // Lambda drift violation
    const lambdaCheck = this.lambdaDriftMonitor.checkBoundsCompliance();
    if (!lambdaCheck.within_bounds) {
      const severity = lambdaCheck.deviation > 0.5 ? 'EMERGENCY' : 'CRITICAL';
      violations.push({
        metric: 'Lambda Drift',
        current_value: lambdaCheck.current_lambda,
        threshold: lambdaCheck.bounds[1], // Using upper bound as reference
        severity,
        recommendation: 'Investigate model parameter drift and consider retraining or parameter reset'
      });
    }
    
    // Performance violations
    if (metrics.delta_cbu_gb < this.config.performance_thresholds.delta_cbu_gb_min) {
      violations.push({
        metric: 'CBU Performance',
        current_value: metrics.delta_cbu_gb,
        threshold: this.config.performance_thresholds.delta_cbu_gb_min,
        severity: 'WARNING',
        recommendation: 'Performance improvement below threshold - consider optimization'
      });
    }
    
    if (metrics.error_rate_delta > this.config.performance_thresholds.error_rate_max_increase) {
      const severity = metrics.error_rate_delta > this.config.performance_thresholds.error_rate_max_increase * 2 ? 'EMERGENCY' : 'CRITICAL';
      violations.push({
        metric: 'Error Rate',
        current_value: metrics.error_rate_delta,
        threshold: this.config.performance_thresholds.error_rate_max_increase,
        severity,
        recommendation: 'Error rate increase exceeds threshold - investigate and rollback if necessary'
      });
    }
    
    // Statistical power violations
    if (metrics.sample_size < this.config.statistical_requirements.min_sample_size) {
      violations.push({
        metric: 'Sample Size',
        current_value: metrics.sample_size,
        threshold: this.config.statistical_requirements.min_sample_size,
        severity: 'WARNING',
        recommendation: 'Insufficient sample size for statistical significance - collect more data'
      });
    }
    
    return violations;
  }

  private determineGateStatus(
    overallScore: number,
    violations: QualityGateResult['violations']
  ): QualityGateResult['status'] {
    // Emergency rollback takes precedence
    const emergencyViolations = violations.filter(v => v.severity === 'EMERGENCY');
    if (emergencyViolations.length > 0) {
      return 'EMERGENCY_ROLLBACK';
    }
    
    // Critical violations block deployment
    const criticalViolations = violations.filter(v => v.severity === 'CRITICAL');
    if (criticalViolations.length > 0) {
      return 'BLOCKED';
    }
    
    // Check overall score thresholds
    if (overallScore < this.config.enforcement_actions.emergency_rollback_threshold_percentage / 100) {
      return 'EMERGENCY_ROLLBACK';
    } else if (overallScore < this.config.enforcement_actions.blocking_threshold_percentage / 100) {
      return 'BLOCKED';
    } else if (overallScore < this.config.enforcement_actions.warning_threshold_percentage / 100) {
      return 'WARNING';
    }
    
    // Check if there are any warnings
    if (violations.some(v => v.severity === 'WARNING')) {
      return 'WARNING';
    }
    
    return 'PASSED';
  }

  private async executeEnforcementActions(
    violations: QualityGateResult['violations']
  ): Promise<QualityGateEnforcementAction[]> {
    const actions: QualityGateEnforcementAction[] = [];
    
    for (const violation of violations) {
      const actionStartTime = Date.now();
      
      try {
        let actionType: QualityGateEnforcementAction['action_type'];
        
        switch (violation.severity) {
          case 'EMERGENCY':
            actionType = 'TRIGGER_ROLLBACK';
            await this.triggerEmergencyRollback(violation);
            break;
          case 'CRITICAL':
            actionType = 'BLOCK_DEPLOYMENT';
            await this.blockDeployment(violation);
            break;
          case 'WARNING':
            actionType = 'LOG_WARNING';
            await this.logWarning(violation);
            break;
        }
        
        const executionTime = Date.now() - actionStartTime;
        
        const action: QualityGateEnforcementAction = {
          action_type: actionType,
          triggered_by: `${violation.metric} violation`,
          timestamp: actionStartTime,
          details: {
            metric: violation.metric,
            current_value: violation.current_value,
            threshold: violation.threshold,
            recommendation: violation.recommendation
          },
          success: true,
          execution_time_ms: executionTime
        };
        
        actions.push(action);
        this.actionHistory.push(action);
        
      } catch (error) {
        const executionTime = Date.now() - actionStartTime;
        
        const failedAction: QualityGateEnforcementAction = {
          action_type: 'LOG_WARNING',
          triggered_by: `${violation.metric} violation`,
          timestamp: actionStartTime,
          details: {
            error: error.toString(),
            metric: violation.metric
          },
          success: false,
          execution_time_ms: executionTime
        };
        
        actions.push(failedAction);
        this.actionHistory.push(failedAction);
      }
    }
    
    return actions;
  }

  private async triggerEmergencyRollback(violation: any): Promise<void> {
    console.log(`🚨 EMERGENCY ROLLBACK triggered by ${violation.metric} violation`);
    console.log(`   Value: ${violation.current_value}, Threshold: ${violation.threshold}`);
    
    // In a real implementation, this would trigger actual rollback procedures
    // For now, we simulate the action
    await this.sleep(1000);
  }

  private async blockDeployment(violation: any): Promise<void> {
    console.log(`🚫 DEPLOYMENT BLOCKED due to ${violation.metric} violation`);
    console.log(`   Value: ${violation.current_value}, Threshold: ${violation.threshold}`);
    
    // In a real implementation, this would block CI/CD pipeline
    await this.sleep(500);
  }

  private async logWarning(violation: any): Promise<void> {
    console.log(`⚠️ WARNING: ${violation.metric} approaching threshold`);
    console.log(`   Value: ${violation.current_value}, Threshold: ${violation.threshold}`);
    console.log(`   Recommendation: ${violation.recommendation}`);
  }

  /**
   * Get quality gate history and trends
   */
  getQualityGateHistory(lastNHours: number = 24): {
    results: QualityGateResult[];
    trends: {
      overall_score_trend: number;
      ece_trend: number;
      ilp_trend: number;
      lambda_stability: string;
    };
    summary: {
      total_evaluations: number;
      passed: number;
      warnings: number;
      blocked: number;
      emergency_rollbacks: number;
    };
  } {
    const cutoffTime = Date.now() - (lastNHours * 60 * 60 * 1000);
    const recentResults = this.enforcementHistory.filter(r => r.timestamp >= cutoffTime);
    
    // Calculate trends
    const overallScores = recentResults.map(r => r.overall_score);
    const eceScores = recentResults.map(r => r.individual_scores.ece_score);
    const ilpScores = recentResults.map(r => r.individual_scores.ilp_score);
    
    const overallScoreTrend = this.calculateTrend(overallScores);
    const eceTrend = this.calculateTrend(eceScores);
    const ilpTrend = this.calculateTrend(ilpScores);
    
    const lambdaDriftStats = this.lambdaDriftMonitor.getLambdaDriftStats();
    const lambdaStability = lambdaDriftStats.volatility < 0.1 ? 'STABLE' : 
                           lambdaDriftStats.volatility < 0.3 ? 'MODERATE' : 'UNSTABLE';
    
    // Calculate summary
    const passed = recentResults.filter(r => r.status === 'PASSED').length;
    const warnings = recentResults.filter(r => r.status === 'WARNING').length;
    const blocked = recentResults.filter(r => r.status === 'BLOCKED' || r.status === 'FAILED').length;
    const emergencyRollbacks = recentResults.filter(r => r.status === 'EMERGENCY_ROLLBACK').length;
    
    return {
      results: recentResults,
      trends: {
        overall_score_trend: overallScoreTrend,
        ece_trend: eceTrend,
        ilp_trend: ilpTrend,
        lambda_stability: lambdaStability
      },
      summary: {
        total_evaluations: recentResults.length,
        passed,
        warnings,
        blocked,
        emergency_rollbacks: emergencyRollbacks
      }
    };
  }

  /**
   * Health check for quality gate system
   */
  healthCheck(): {
    healthy: boolean;
    issues: string[];
    metrics: {
      recent_evaluations: number;
      recent_failures: number;
      lambda_within_bounds: boolean;
      action_success_rate: number;
    };
  } {
    const issues: string[] = [];
    const now = Date.now();
    const oneHourAgo = now - (60 * 60 * 1000);
    
    // Check recent evaluations
    const recentEvaluations = this.enforcementHistory.filter(r => r.timestamp >= oneHourAgo);
    const recentFailures = recentEvaluations.filter(r => 
      r.status === 'BLOCKED' || r.status === 'FAILED' || r.status === 'EMERGENCY_ROLLBACK'
    ).length;
    
    if (recentEvaluations.length === 0) {
      issues.push('No quality gate evaluations in the last hour');
    }
    
    // Check lambda bounds
    const lambdaCheck = this.lambdaDriftMonitor.checkBoundsCompliance();
    if (!lambdaCheck.within_bounds) {
      issues.push(`Lambda drift out of bounds: ${lambdaCheck.current_lambda} (bounds: ${lambdaCheck.bounds})`);
    }
    
    // Check action success rate
    const recentActions = this.actionHistory.filter(a => a.timestamp >= oneHourAgo);
    const successfulActions = recentActions.filter(a => a.success).length;
    const actionSuccessRate = recentActions.length > 0 ? successfulActions / recentActions.length : 1;
    
    if (actionSuccessRate < 0.9) {
      issues.push(`Low enforcement action success rate: ${(actionSuccessRate * 100).toFixed(1)}%`);
    }
    
    // Check failure rate
    if (recentEvaluations.length > 0 && recentFailures / recentEvaluations.length > 0.2) {
      issues.push(`High quality gate failure rate: ${(recentFailures / recentEvaluations.length * 100).toFixed(1)}%`);
    }
    
    return {
      healthy: issues.length === 0,
      issues,
      metrics: {
        recent_evaluations: recentEvaluations.length,
        recent_failures: recentFailures,
        lambda_within_bounds: lambdaCheck.within_bounds,
        action_success_rate: actionSuccessRate
      }
    };
  }

  private calculateTrend(values: number[]): number {
    if (values.length < 2) return 0;
    
    // Simple linear regression slope
    const n = values.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    
    for (let i = 0; i < n; i++) {
      sumX += i;
      sumY += values[i];
      sumXY += i * values[i];
      sumXX += i * i;
    }
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    return slope;
  }

  private async sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}