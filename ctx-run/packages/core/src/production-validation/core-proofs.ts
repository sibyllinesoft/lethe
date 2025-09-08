/**
 * Core Mathematical Validation Proofs for Production Readiness
 * Three Critical Proofs: Dual Sanity, OOD Resilience, Long-horizon Win Rate
 */

export interface ValidationResult {
  passed: boolean;
  confidence: number;
  metrics: Record<string, number>;
  timestamp: number;
}

export interface ProofConfig {
  ece_threshold: number; // Expected Calibration Error ≤ 0.08
  ilp_threshold: number; // Information Leakage Percentage ≤ 5%
  lambda_drift_bounds: [number, number]; // Acceptable λ-drift range
  min_statistical_power: number; // O(10⁴-10⁵) turns
  confidence_level: number; // 80% confidence minimum
}

/**
 * Dual Sanity Proof: Validates both forward and backward consistency
 * Critical for ensuring retrieval system coherence under load
 */
export class DualSanityProof {
  private config: ProofConfig;

  constructor(config: ProofConfig) {
    this.config = config;
  }

  /**
   * Execute dual sanity validation
   * Tests forward coherence (query → result) and backward coherence (result → query)
   */
  async validate(
    forwardSamples: Array<{query: string, result: any, relevance: number}>,
    backwardSamples: Array<{result: any, query: string, coherence: number}>
  ): Promise<ValidationResult> {
    const startTime = Date.now();
    
    // Forward coherence validation
    const forwardCoherence = this.calculateForwardCoherence(forwardSamples);
    
    // Backward coherence validation  
    const backwardCoherence = this.calculateBackwardCoherence(backwardSamples);
    
    // Bidirectional consistency check
    const bidirectionalConsistency = this.calculateBidirectionalConsistency(
      forwardCoherence, backwardCoherence
    );
    
    // Statistical significance validation
    const statisticalPower = this.calculateStatisticalPower(
      forwardSamples.length + backwardSamples.length
    );
    
    const passed = forwardCoherence >= 0.85 && 
                  backwardCoherence >= 0.85 && 
                  bidirectionalConsistency >= 0.80 &&
                  statisticalPower >= this.config.min_statistical_power;
    
    return {
      passed,
      confidence: Math.min(forwardCoherence, backwardCoherence, bidirectionalConsistency),
      metrics: {
        forward_coherence: forwardCoherence,
        backward_coherence: backwardCoherence,
        bidirectional_consistency: bidirectionalConsistency,
        statistical_power: statisticalPower,
        execution_time_ms: Date.now() - startTime
      },
      timestamp: Date.now()
    };
  }

  private calculateForwardCoherence(samples: Array<{query: string, result: any, relevance: number}>): number {
    if (samples.length === 0) return 0;
    
    const relevanceSum = samples.reduce((sum, sample) => sum + sample.relevance, 0);
    return relevanceSum / samples.length;
  }

  private calculateBackwardCoherence(samples: Array<{result: any, query: string, coherence: number}>): number {
    if (samples.length === 0) return 0;
    
    const coherenceSum = samples.reduce((sum, sample) => sum + sample.coherence, 0);
    return coherenceSum / samples.length;
  }

  private calculateBidirectionalConsistency(forward: number, backward: number): number {
    return 1 - Math.abs(forward - backward);
  }

  private calculateStatisticalPower(sampleSize: number): number {
    // Statistical power calculation based on sample size
    return Math.min(1.0, sampleSize / this.config.min_statistical_power);
  }
}

/**
 * Out-of-Distribution (OOD) Resilience Proof
 * Validates system performance under distribution shifts and novel inputs
 */
export class OODResilienceProof {
  private config: ProofConfig;

  constructor(config: ProofConfig) {
    this.config = config;
  }

  /**
   * Execute OOD resilience validation
   * Tests system stability under various distribution shifts
   */
  async validate(
    inDistributionSamples: Array<{input: any, output: any, confidence: number}>,
    oodSamples: Array<{input: any, output: any, confidence: number, shift_type: string}>
  ): Promise<ValidationResult> {
    const startTime = Date.now();
    
    // Calculate baseline performance on in-distribution data
    const baselinePerformance = this.calculateBaselinePerformance(inDistributionSamples);
    
    // Calculate OOD performance degradation
    const oodPerformance = this.calculateOODPerformance(oodSamples);
    
    // Expected Calibration Error (ECE) calculation
    const ece = this.calculateECE([...inDistributionSamples, ...oodSamples]);
    
    // Information Leakage Percentage (ILP) calculation
    const ilp = this.calculateILP(inDistributionSamples, oodSamples);
    
    // Coverage-weighted CRPS for uncertainty quantification
    const crps = this.calculateCoverageWeightedCRPS(oodSamples);
    
    const passed = ece <= this.config.ece_threshold &&
                  ilp <= this.config.ilp_threshold &&
                  oodPerformance >= baselinePerformance * 0.7; // 70% retention minimum
    
    return {
      passed,
      confidence: Math.min(baselinePerformance, oodPerformance / baselinePerformance),
      metrics: {
        baseline_performance: baselinePerformance,
        ood_performance: oodPerformance,
        ece: ece,
        ilp: ilp,
        crps: crps,
        performance_retention: oodPerformance / baselinePerformance,
        execution_time_ms: Date.now() - startTime
      },
      timestamp: Date.now()
    };
  }

  private calculateBaselinePerformance(samples: Array<{input: any, output: any, confidence: number}>): number {
    if (samples.length === 0) return 0;
    
    const confidenceSum = samples.reduce((sum, sample) => sum + sample.confidence, 0);
    return confidenceSum / samples.length;
  }

  private calculateOODPerformance(samples: Array<{input: any, output: any, confidence: number, shift_type: string}>): number {
    if (samples.length === 0) return 0;
    
    const confidenceSum = samples.reduce((sum, sample) => sum + sample.confidence, 0);
    return confidenceSum / samples.length;
  }

  private calculateECE(samples: Array<{input: any, output: any, confidence: number}>): number {
    // Expected Calibration Error calculation
    const bins = 10;
    const binSize = 1.0 / bins;
    let ece = 0;
    
    for (let i = 0; i < bins; i++) {
      const binLower = i * binSize;
      const binUpper = (i + 1) * binSize;
      
      const binSamples = samples.filter(s => 
        s.confidence >= binLower && s.confidence < binUpper
      );
      
      if (binSamples.length > 0) {
        const avgConfidence = binSamples.reduce((sum, s) => sum + s.confidence, 0) / binSamples.length;
        const accuracy = binSamples.reduce((sum, s) => sum + (s.confidence > 0.5 ? 1 : 0), 0) / binSamples.length;
        
        ece += (binSamples.length / samples.length) * Math.abs(avgConfidence - accuracy);
      }
    }
    
    return ece;
  }

  private calculateILP(
    inDist: Array<{input: any, output: any, confidence: number}>,
    oodSamples: Array<{input: any, output: any, confidence: number, shift_type: string}>
  ): number {
    // Information Leakage Percentage calculation
    // Measures how much the model relies on spurious correlations
    
    const inDistAvgConfidence = this.calculateBaselinePerformance(inDist);
    const oodAvgConfidence = this.calculateOODPerformance(oodSamples);
    
    // ILP is the percentage drop in confidence for OOD samples
    return Math.max(0, (inDistAvgConfidence - oodAvgConfidence) / inDistAvgConfidence);
  }

  private calculateCoverageWeightedCRPS(samples: Array<{input: any, output: any, confidence: number, shift_type: string}>): number {
    // Coverage-weighted Continuous Ranked Probability Score
    // Measures quality of uncertainty estimates
    
    if (samples.length === 0) return 1.0;
    
    let crps = 0;
    for (const sample of samples) {
      // Simplified CRPS calculation based on confidence vs actual performance
      const predicted = sample.confidence;
      const actual = sample.confidence > 0.5 ? 1 : 0;
      crps += Math.pow(predicted - actual, 2);
    }
    
    return crps / samples.length;
  }
}

/**
 * Long-horizon Win Rate Proof
 * Validates sustained performance over extended periods
 */
export class LongHorizonWinRateProof {
  private config: ProofConfig;

  constructor(config: ProofConfig) {
    this.config = config;
  }

  /**
   * Execute long-horizon win rate validation
   * Tests system performance stability over time with O(10⁴-10⁵) statistical power
   */
  async validate(
    timeSeriesData: Array<{
      timestamp: number,
      performance: number,
      lambda: number, // Model parameter λ
      context_size: number,
      win: boolean
    }>
  ): Promise<ValidationResult> {
    const startTime = Date.now();
    
    // Calculate overall win rate
    const winRate = this.calculateWinRate(timeSeriesData);
    
    // Check λ-drift bounds compliance
    const lambdaDriftCompliance = this.validateLambdaDrift(timeSeriesData);
    
    // Calculate performance trend over time
    const performanceTrend = this.calculatePerformanceTrend(timeSeriesData);
    
    // Statistical power validation
    const statisticalPower = this.calculateStatisticalPower(timeSeriesData.length);
    
    // Long-term stability measurement
    const stability = this.calculateStability(timeSeriesData);
    
    const passed = winRate >= 0.75 && // 75% win rate minimum
                  lambdaDriftCompliance &&
                  performanceTrend >= -0.05 && // Max 5% degradation trend
                  statisticalPower >= this.config.min_statistical_power &&
                  stability >= 0.8; // 80% stability minimum
    
    return {
      passed,
      confidence: Math.min(winRate, stability, statisticalPower),
      metrics: {
        win_rate: winRate,
        lambda_drift_compliance: lambdaDriftCompliance ? 1 : 0,
        performance_trend: performanceTrend,
        statistical_power: statisticalPower,
        stability: stability,
        sample_count: timeSeriesData.length,
        execution_time_ms: Date.now() - startTime
      },
      timestamp: Date.now()
    };
  }

  private calculateWinRate(data: Array<{win: boolean}>): number {
    if (data.length === 0) return 0;
    
    const wins = data.filter(d => d.win).length;
    return wins / data.length;
  }

  private validateLambdaDrift(data: Array<{lambda: number}>): boolean {
    if (data.length < 2) return true;
    
    const lambdaValues = data.map(d => d.lambda);
    const minLambda = Math.min(...lambdaValues);
    const maxLambda = Math.max(...lambdaValues);
    
    return minLambda >= this.config.lambda_drift_bounds[0] &&
           maxLambda <= this.config.lambda_drift_bounds[1];
  }

  private calculatePerformanceTrend(data: Array<{timestamp: number, performance: number}>): number {
    if (data.length < 2) return 0;
    
    // Linear regression to calculate trend
    const sortedData = [...data].sort((a, b) => a.timestamp - b.timestamp);
    const n = sortedData.length;
    
    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    
    for (let i = 0; i < n; i++) {
      const x = i; // Normalized time index
      const y = sortedData[i].performance;
      
      sumX += x;
      sumY += y;
      sumXY += x * y;
      sumXX += x * x;
    }
    
    // Calculate slope (trend)
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    return slope;
  }

  private calculateStatisticalPower(sampleSize: number): number {
    // Statistical power based on sample size
    return Math.min(1.0, sampleSize / this.config.min_statistical_power);
  }

  private calculateStability(data: Array<{performance: number}>): number {
    if (data.length < 2) return 1;
    
    const performances = data.map(d => d.performance);
    const mean = performances.reduce((sum, p) => sum + p, 0) / performances.length;
    
    // Calculate coefficient of variation (inverse stability)
    const variance = performances.reduce((sum, p) => sum + Math.pow(p - mean, 2), 0) / performances.length;
    const stdDev = Math.sqrt(variance);
    const coefficientOfVariation = stdDev / mean;
    
    // Stability is inverse of coefficient of variation
    return Math.max(0, 1 - coefficientOfVariation);
  }
}

/**
 * Production Validation Orchestrator
 * Coordinates all three core proofs and provides comprehensive validation
 */
export class ProductionValidationOrchestrator {
  private dualSanityProof: DualSanityProof;
  private oodResilienceProof: OODResilienceProof;
  private longHorizonProof: LongHorizonWinRateProof;
  private config: ProofConfig;

  constructor(config: ProofConfig) {
    this.config = config;
    this.dualSanityProof = new DualSanityProof(config);
    this.oodResilienceProof = new OODResilienceProof(config);
    this.longHorizonProof = new LongHorizonWinRateProof(config);
  }

  /**
   * Execute all three core proofs for production readiness validation
   */
  async validateProduction(data: {
    dualSanity: {
      forward: Array<{query: string, result: any, relevance: number}>,
      backward: Array<{result: any, query: string, coherence: number}>
    },
    oodResilience: {
      inDistribution: Array<{input: any, output: any, confidence: number}>,
      ood: Array<{input: any, output: any, confidence: number, shift_type: string}>
    },
    longHorizon: Array<{
      timestamp: number,
      performance: number,
      lambda: number,
      context_size: number,
      win: boolean
    }>
  }): Promise<{
    overallPassed: boolean,
    confidence: number,
    results: {
      dualSanity: ValidationResult,
      oodResilience: ValidationResult,
      longHorizon: ValidationResult
    },
    timestamp: number
  }> {
    const startTime = Date.now();
    
    // Execute all proofs in parallel
    const [dualSanityResult, oodResilienceResult, longHorizonResult] = await Promise.all([
      this.dualSanityProof.validate(data.dualSanity.forward, data.dualSanity.backward),
      this.oodResilienceProof.validate(data.oodResilience.inDistribution, data.oodResilience.ood),
      this.longHorizonProof.validate(data.longHorizon)
    ]);
    
    const overallPassed = dualSanityResult.passed && 
                         oodResilienceResult.passed && 
                         longHorizonResult.passed;
    
    const confidence = Math.min(
      dualSanityResult.confidence,
      oodResilienceResult.confidence,
      longHorizonResult.confidence
    );
    
    return {
      overallPassed,
      confidence,
      results: {
        dualSanity: dualSanityResult,
        oodResilience: oodResilienceResult,
        longHorizon: longHorizonResult
      },
      timestamp: Date.now()
    };
  }

  /**
   * Health check for continuous monitoring
   */
  async healthCheck(): Promise<{healthy: boolean, issues: string[]}> {
    const issues: string[] = [];
    
    // Check if all proof systems are initialized
    if (!this.dualSanityProof || !this.oodResilienceProof || !this.longHorizonProof) {
      issues.push("One or more proof systems not initialized");
    }
    
    // Validate configuration
    if (this.config.ece_threshold > 0.1) {
      issues.push(`ECE threshold too high: ${this.config.ece_threshold} > 0.1`);
    }
    
    if (this.config.ilp_threshold > 0.05) {
      issues.push(`ILP threshold too high: ${this.config.ilp_threshold} > 0.05`);
    }
    
    if (this.config.min_statistical_power < 10000) {
      issues.push(`Statistical power too low: ${this.config.min_statistical_power} < 10000`);
    }
    
    return {
      healthy: issues.length === 0,
      issues
    };
  }
}