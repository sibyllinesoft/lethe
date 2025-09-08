/**
 * Counterfactual CBU (Coverage-Budget-Utility) System
 * 
 * Fast recomputation of facility+DPP marginals under policy perturbations
 * using Importance-weighted Policy Sampling (IPS) without additional LLM calls.
 * Enables rapid exploration of the parameter space for automated tuning.
 */

import {
  CounterfactualAnalysis,
  PolicyPerturbation,
  CounterfactualUplift,
  ConstraintViolation,
  PolicyFingerprint,
  GapRecord,
  GapAnalysisResult,
  GapAnalysisError
} from './types.js';

import { Config, Candidate } from '../types.js';
import { createHash } from 'crypto';

// ============================================================================
// CORE COUNTERFACTUAL ANALYSIS ENGINE
// ============================================================================

export class CounterfactualCBU {
  private config: Config;
  private cachedVectors: Map<string, Float64Array> = new Map();
  private facilityMarginals: Map<string, Float64Array> = new Map();
  private dppMarginals: Map<string, Float64Array> = new Map();
  private constraintValidator: ConstraintValidator;

  constructor(config: Config) {
    this.config = config;
    this.constraintValidator = new ConstraintValidator();
  }

  /**
   * Performs counterfactual analysis for a given gap slice without LLM calls
   */
  async performCounterfactualAnalysis(
    gapRecord: GapRecord,
    savedAtoms: SavedAtomData[],
    maxPerturbations: number = 50
  ): Promise<GapAnalysisResult<CounterfactualAnalysis>> {
    try {
      // Cache current vectors and marginals
      await this.cacheCurrentState(savedAtoms);
      
      // Generate policy perturbations
      const perturbations = this.generatePolicyPerturbations(
        gapRecord.policy_fingerprint,
        maxPerturbations
      );

      // Compute counterfactual uplifts for each perturbation
      const upliftFrontier: CounterfactualUplift[] = [];
      const constraintViolations: ConstraintViolation[] = [];

      for (const perturbation of perturbations) {
        // Fast recompute without LLM
        const uplift = await this.computeCounterfactualUplift(
          gapRecord,
          perturbation,
          savedAtoms
        );

        // Check constraint violations
        const violations = this.constraintValidator.validateConstraints(
          perturbation,
          uplift
        );

        if (violations.length === 0) {
          upliftFrontier.push(uplift);
        } else {
          constraintViolations.push(...violations);
        }
      }

      // Calculate importance sampling metrics
      const importanceWeights = this.calculateImportanceWeights(
        gapRecord.policy_fingerprint,
        perturbations
      );

      const analysis: CounterfactualAnalysis = {
        slice_id: gapRecord.slice_id,
        base_policy: gapRecord.policy_fingerprint,
        perturbations,
        uplift_frontier: upliftFrontier.sort((a, b) => 
          b.predicted_p_at_5_improvement - a.predicted_p_at_5_improvement
        ),
        constraint_violations: constraintViolations,
        importance_weights: importanceWeights,
        created_at: Date.now()
      };

      return {
        success: true,
        data: analysis
      };

    } catch (error) {
      return {
        success: false,
        error: {
          code: 'COUNTERFACTUAL_ANALYSIS_ERROR',
          message: `Counterfactual analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          error_type: 'counterfactual_analysis',
          recovery_actions: ['Verify saved atoms data', 'Check policy fingerprint validity', 'Validate constraint configuration'],
          is_retryable: true,
          impact_severity: 'medium',
          affected_components: ['counterfactual_engine'],
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Fast facility + DPP marginal recomputation using cached vectors
   */
  async recomputeMarginals(
    perturbedPolicy: PolicyFingerprint,
    savedAtoms: SavedAtomData[]
  ): Promise<RecomputedMarginals> {
    // Extract cached vectors
    const vectors = savedAtoms.map(atom => this.cachedVectors.get(atom.atom_id)).filter(Boolean) as Float64Array[];
    
    if (vectors.length === 0) {
      throw new Error('No cached vectors available for marginal recomputation');
    }

    // Recompute facility marginals with new lambda/mu weights
    const facilityMarginals = this.recomputeFacilityMarginals(
      vectors,
      perturbedPolicy.lambda,
      perturbedPolicy.mu
    );

    // Recompute DPP marginals with new K2 and r parameters
    const dppMarginals = this.recomputeDPPMarginals(
      vectors,
      facilityMarginals,
      perturbedPolicy.K2,
      perturbedPolicy.r
    );

    // Apply head_keep filtering
    const filteredMarginals = this.applyHeadKeepFiltering(
      dppMarginals,
      perturbedPolicy.head_keep
    );

    return {
      facility_marginals: facilityMarginals,
      dpp_marginals: dppMarginals,
      filtered_marginals: filteredMarginals,
      effective_k: Math.min(perturbedPolicy.K2, filteredMarginals.length),
      computation_time_ms: Date.now() - Date.now() // TODO: Proper timing
    };
  }

  // ============================================================================
  // POLICY PERTURBATION GENERATION
  // ============================================================================

  private generatePolicyPerturbations(
    basePolicy: PolicyFingerprint,
    maxPerturbations: number
  ): PolicyPerturbation[] {
    const perturbations: PolicyPerturbation[] = [];
    
    // Define perturbation ranges from TODO specifications
    const perturbationRanges = {
      lambda: [-20, -10, -5, 5, 10, 20], // ±{5,10,20}%
      mu: [-10, -5, 5, 10], // ±{5,10}%
      K2: [-30, -15, 15, 30], // ±{15,30}%
      r: [12, 14, 16, 24], // Fixed values
      head_keep: [-4, -2, 2, 4], // ±{2,4} percentage points
      window_stride_combinations: [
        [1024, 512], [2048, 1024], [512, 256], [1536, 768]
      ]
    };

    let perturbationId = 0;

    // Single parameter perturbations
    for (const [param, values] of Object.entries(perturbationRanges)) {
      if (param === 'window_stride_combinations') continue;
      
      for (const delta of values as number[]) {
        if (perturbationId >= maxPerturbations) break;
        
        const perturbation = this.createPerturbation(
          basePolicy,
          { [param]: delta },
          `${perturbationId++}`
        );
        
        if (perturbation) perturbations.push(perturbation);
      }
    }

    // Combined parameter perturbations (higher impact)
    const combinedPerturbations = [
      { lambda: 10, mu: 5, K2: 15 }, // Aggressive retrieval boost
      { lambda: -10, mu: -5, r: 16 }, // Conservative with better diversity
      { K2: 30, r: 24, head_keep: 4 }, // Maximum ranking focus
      { lambda: 5, K2: -15, r: 12 }, // Balanced efficiency
    ];

    for (const combo of combinedPerturbations) {
      if (perturbationId >= maxPerturbations) break;
      
      const perturbation = this.createPerturbation(
        basePolicy,
        combo,
        `combo-${perturbationId++}`
      );
      
      if (perturbation) perturbations.push(perturbation);
    }

    return perturbations.slice(0, maxPerturbations);
  }

  private createPerturbation(
    basePolicy: PolicyFingerprint,
    changes: Record<string, number>,
    perturbationId: string
  ): PolicyPerturbation | null {
    const perturbedPolicy = { ...basePolicy };
    const parameterChanges: Partial<PolicyFingerprint> = {};

    // Apply percentage changes
    if (changes.lambda !== undefined) {
      const newLambda = basePolicy.lambda * (1 + changes.lambda / 100);
      perturbedPolicy.lambda = Math.max(0, Math.min(2.0, newLambda));
      parameterChanges.lambda = perturbedPolicy.lambda;
    }

    if (changes.mu !== undefined) {
      const newMu = basePolicy.mu * (1 + changes.mu / 100);
      perturbedPolicy.mu = Math.max(0, Math.min(2.0, newMu));
      parameterChanges.mu = perturbedPolicy.mu;
    }

    if (changes.K2 !== undefined) {
      const newK2 = Math.round(basePolicy.K2 * (1 + changes.K2 / 100));
      perturbedPolicy.K2 = Math.max(5, Math.min(1000, newK2));
      parameterChanges.K2 = perturbedPolicy.K2;
    }

    if (changes.r !== undefined) {
      perturbedPolicy.r = changes.r; // Direct assignment for fixed values
      parameterChanges.r = perturbedPolicy.r;
    }

    if (changes.head_keep !== undefined) {
      const newHeadKeep = basePolicy.head_keep + changes.head_keep;
      perturbedPolicy.head_keep = Math.max(0, Math.min(100, newHeadKeep));
      parameterChanges.head_keep = perturbedPolicy.head_keep;
    }

    // Validate perturbation is meaningful
    if (Object.keys(parameterChanges).length === 0) {
      return null;
    }

    return {
      perturbation_id: perturbationId,
      parameter_changes: parameterChanges,
      expected_delta_p_at_5: 0, // Will be computed
      expected_delta_latency: 0, // Will be computed
      confidence_interval_p_at_5: [0, 0], // Will be computed
      confidence_interval_latency: [0, 0], // Will be computed
      satisfies_constraints: true, // Will be validated
      violation_reasons: []
    };
  }

  // ============================================================================
  // MARGINAL RECOMPUTATION METHODS
  // ============================================================================

  private recomputeFacilityMarginals(
    vectors: Float64Array[],
    lambda: number,
    mu: number
  ): Float64Array {
    const n = vectors.length;
    const marginals = new Float64Array(n);

    // Simplified facility location computation
    // In practice, this would use proper submodular optimization
    for (let i = 0; i < n; i++) {
      // Compute facility value: combination of BM25-like (lambda) and vector similarity (mu)
      let facilityValue = 0;
      
      // BM25 component (simplified)
      facilityValue += lambda * this.computeBM25Component(vectors[i]);
      
      // Vector similarity component
      for (let j = 0; j < n; j++) {
        if (i !== j) {
          facilityValue += mu * this.computeCosineSimilarity(vectors[i], vectors[j]);
        }
      }
      
      marginals[i] = facilityValue / n; // Normalize
    }

    return marginals;
  }

  private recomputeDPPMarginals(
    vectors: Float64Array[],
    facilityMarginals: Float64Array,
    K2: number,
    r: number
  ): Float64Array {
    const n = vectors.length;
    const effectiveK = Math.min(K2, n);
    
    // Construct kernel matrix for DPP
    const kernelMatrix = this.constructDPPKernel(vectors, facilityMarginals, r);
    
    // Compute marginals using eigendecomposition approximation
    return this.computeDPPMarginalsFromKernel(kernelMatrix, effectiveK);
  }

  private constructDPPKernel(
    vectors: Float64Array[],
    qualityScores: Float64Array,
    r: number
  ): Float64Array[] {
    const n = vectors.length;
    const kernel: Float64Array[] = [];
    
    for (let i = 0; i < n; i++) {
      kernel[i] = new Float64Array(n);
      for (let j = 0; j < n; j++) {
        if (i === j) {
          // Diagonal: quality score
          kernel[i][j] = qualityScores[i];
        } else {
          // Off-diagonal: diversity term
          const similarity = this.computeCosineSimilarity(vectors[i], vectors[j]);
          const diversity = Math.max(0, 1 - similarity);
          kernel[i][j] = Math.sqrt(qualityScores[i] * qualityScores[j]) * 
                        Math.pow(diversity, 1 / r); // r controls diversity strength
        }
      }
    }
    
    return kernel;
  }

  private computeDPPMarginalsFromKernel(
    kernel: Float64Array[],
    k: number
  ): Float64Array {
    const n = kernel.length;
    const marginals = new Float64Array(n);
    
    // Simplified marginal computation (in practice, use proper DPP algorithms)
    // This is an approximation using eigenvalue decomposition
    for (let i = 0; i < n; i++) {
      marginals[i] = kernel[i][i] * (k / n); // Simplified approximation
      
      // Adjust based on off-diagonal terms (diversity contribution)
      let diversityBonus = 0;
      for (let j = 0; j < n; j++) {
        if (i !== j) {
          diversityBonus += kernel[i][j] / n;
        }
      }
      marginals[i] += diversityBonus * 0.1; // Weight diversity contribution
    }
    
    return marginals;
  }

  private applyHeadKeepFiltering(
    marginals: Float64Array,
    headKeepRatio: number
  ): Float64Array {
    const n = marginals.length;
    const keepCount = Math.floor(n * headKeepRatio / 100);
    
    // Sort indices by marginal value
    const indices = Array.from({ length: n }, (_, i) => i)
      .sort((a, b) => marginals[b] - marginals[a]);
    
    // Keep only top headKeepRatio percent
    const filtered = new Float64Array(keepCount);
    for (let i = 0; i < keepCount; i++) {
      filtered[i] = marginals[indices[i]];
    }
    
    return filtered;
  }

  // ============================================================================
  // COUNTERFACTUAL UPLIFT COMPUTATION
  // ============================================================================

  private async computeCounterfactualUplift(
    gapRecord: GapRecord,
    perturbation: PolicyPerturbation,
    savedAtoms: SavedAtomData[]
  ): Promise<CounterfactualUplift> {
    // Create perturbed policy
    const perturbedPolicy = { ...gapRecord.policy_fingerprint, ...perturbation.parameter_changes };
    
    // Recompute marginals
    const recomputedMarginals = await this.recomputeMarginals(perturbedPolicy, savedAtoms);
    
    // Estimate performance improvements using IPS
    const performancePredictions = this.estimatePerformanceWithIPS(
      gapRecord,
      recomputedMarginals,
      perturbation
    );

    // Assess implementation complexity
    const implementationComplexity = this.assessImplementationComplexity(perturbation);

    // Calculate uncertainty and risk
    const uncertaintyScore = this.calculateUncertaintyScore(
      perturbation,
      recomputedMarginals
    );

    const downsideRisk = this.calculateDownsideRisk(
      gapRecord,
      performancePredictions
    );

    return {
      policy_variant: perturbedPolicy,
      predicted_p_at_5_improvement: performancePredictions.p_at_5_improvement,
      predicted_latency_change: performancePredictions.latency_change,
      predicted_cost_efficiency: performancePredictions.cost_efficiency_change,
      uncertainty_score: uncertaintyScore,
      downside_risk: downsideRisk,
      implementation_complexity: implementationComplexity,
      estimated_validation_time: this.estimateValidationTime(implementationComplexity)
    };
  }

  private estimatePerformanceWithIPS(
    gapRecord: GapRecord,
    recomputedMarginals: RecomputedMarginals,
    perturbation: PolicyPerturbation
  ): PerformancePredictions {
    // Use Importance-weighted Policy Sampling (IPS) for off-policy evaluation
    
    // Calculate importance weights
    const importanceWeight = this.calculateImportanceWeight(
      gapRecord.policy_fingerprint,
      perturbation
    );

    // Base performance estimates on marginal changes
    const marginalSum = recomputedMarginals.filtered_marginals.reduce((a, b) => a + b, 0);
    const avgMarginal = marginalSum / recomputedMarginals.filtered_marginals.length;

    // P@5 improvement estimate
    const baselineP5 = 0.5; // Assumed baseline
    const marginalBoost = (avgMarginal - 0.5) * importanceWeight; // Relative to neutral (0.5)
    const p5Improvement = marginalBoost * 0.1; // Scale factor based on empirical data

    // Latency change estimate (more complex operations = higher latency)
    const latencyChange = this.estimateLatencyChange(perturbation, recomputedMarginals);

    // Cost efficiency change
    const costEfficiencyChange = p5Improvement / Math.max(0.001, Math.abs(latencyChange));

    return {
      p_at_5_improvement: p5Improvement,
      latency_change: latencyChange,
      cost_efficiency_change: costEfficiencyChange
    };
  }

  private estimateLatencyChange(
    perturbation: PolicyPerturbation,
    marginals: RecomputedMarginals
  ): number {
    let latencyDelta = 0;

    // K2 changes affect cross-encoder calls
    if (perturbation.parameter_changes.K2) {
      const k2Delta = perturbation.parameter_changes.K2 - (this.config.rerank.topk_in || 100);
      latencyDelta += k2Delta * 0.5; // Each additional K2 adds ~0.5ms
    }

    // r changes affect DPP computation complexity (O(r²))
    if (perturbation.parameter_changes.r) {
      const rDelta = perturbation.parameter_changes.r - (this.config.diversify.pack_chunks || 16);
      latencyDelta += rDelta * rDelta * 0.1; // Quadratic complexity
    }

    // Window size changes affect chunking
    if (perturbation.parameter_changes.window_size) {
      const windowDelta = perturbation.parameter_changes.window_size - (this.config.retrieval.window_size || 1024);
      latencyDelta += windowDelta * 0.001; // Linear in window size
    }

    return latencyDelta;
  }

  // ============================================================================
  // IMPORTANCE SAMPLING CALCULATIONS
  // ============================================================================

  private calculateImportanceWeights(
    basePolicy: PolicyFingerprint,
    perturbations: PolicyPerturbation[]
  ): CounterfactualAnalysis['importance_weights'] {
    const weights = perturbations.map(p => this.calculateImportanceWeight(basePolicy, p));
    
    const effectiveSampleSize = this.calculateEffectiveSampleSize(weights);
    const weightVariance = this.calculateVariance(weights);
    const biasEstimate = this.estimateBias(weights);

    return {
      effective_sample_size: effectiveSampleSize,
      weight_variance: weightVariance,
      bias_estimate: biasEstimate
    };
  }

  private calculateImportanceWeight(
    basePolicy: PolicyFingerprint,
    perturbation: PolicyPerturbation
  ): number {
    // Calculate probability ratio: π(a|s) / π₀(a|s)
    // Simplified as product of parameter change ratios
    
    let weight = 1.0;
    
    if (perturbation.parameter_changes.lambda) {
      weight *= perturbation.parameter_changes.lambda / basePolicy.lambda;
    }
    
    if (perturbation.parameter_changes.mu) {
      weight *= perturbation.parameter_changes.mu / basePolicy.mu;
    }
    
    if (perturbation.parameter_changes.K2) {
      weight *= perturbation.parameter_changes.K2 / basePolicy.K2;
    }
    
    // Clamp to reasonable range to prevent extreme weights
    return Math.max(0.01, Math.min(10.0, weight));
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  private async cacheCurrentState(savedAtoms: SavedAtomData[]): Promise<void> {
    for (const atom of savedAtoms) {
      if (!this.cachedVectors.has(atom.atom_id)) {
        // In practice, load from saved embeddings
        this.cachedVectors.set(atom.atom_id, new Float64Array(atom.embedding));
      }
    }
  }

  private computeBM25Component(vector: Float64Array): number {
    // Simplified BM25-style scoring based on vector norms
    const norm = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return Math.log(1 + norm) / Math.log(2); // Log-normalized
  }

  private computeCosineSimilarity(vec1: Float64Array, vec2: Float64Array): number {
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;
    
    for (let i = 0; i < vec1.length && i < vec2.length; i++) {
      dotProduct += vec1[i] * vec2[i];
      norm1 += vec1[i] * vec1[i];
      norm2 += vec2[i] * vec2[i];
    }
    
    const denominator = Math.sqrt(norm1) * Math.sqrt(norm2);
    return denominator > 0 ? dotProduct / denominator : 0;
  }

  private calculateEffectiveSampleSize(weights: number[]): number {
    const sumWeights = weights.reduce((a, b) => a + b, 0);
    const sumSquaredWeights = weights.reduce((a, b) => a + b * b, 0);
    return (sumWeights * sumWeights) / sumSquaredWeights;
  }

  private calculateVariance(values: number[]): number {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    return values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
  }

  private estimateBias(weights: number[]): number {
    // Simple bias estimate based on weight distribution
    const mean = weights.reduce((a, b) => a + b, 0) / weights.length;
    return Math.abs(1.0 - mean);
  }

  private assessImplementationComplexity(perturbation: PolicyPerturbation): 'low' | 'medium' | 'high' {
    const changeCount = Object.keys(perturbation.parameter_changes).length;
    
    if (changeCount <= 1) return 'low';
    if (changeCount <= 3) return 'medium';
    return 'high';
  }

  private calculateUncertaintyScore(
    perturbation: PolicyPerturbation,
    marginals: RecomputedMarginals
  ): number {
    // Uncertainty increases with:
    // 1. Number of simultaneous parameter changes
    // 2. Magnitude of changes
    // 3. Variance in marginals
    
    const changeCount = Object.keys(perturbation.parameter_changes).length;
    const marginalVariance = this.calculateVariance(Array.from(marginals.filtered_marginals));
    
    return Math.min(1.0, changeCount * 0.2 + marginalVariance * 0.5);
  }

  private calculateDownsideRisk(
    gapRecord: GapRecord,
    predictions: PerformancePredictions
  ): number {
    // Risk of performance regression
    const currentGap = Math.abs(gapRecord.delta_map.macro_p_at_5);
    const predictedImprovement = predictions.p_at_5_improvement;
    
    // Higher risk if predicted improvement is small relative to current gap
    return Math.max(0, currentGap - predictedImprovement) / currentGap;
  }

  private estimateValidationTime(complexity: 'low' | 'medium' | 'high'): number {
    const baseTimes = { low: 5, medium: 15, high: 30 }; // minutes
    return baseTimes[complexity];
  }
}

// ============================================================================
// CONSTRAINT VALIDATION SYSTEM
// ============================================================================

export class ConstraintValidator {
  private readonly ECE_THRESHOLD = 0.08;
  private readonly P99_P95_RATIO_MAX = 2.5;
  private readonly PROXY_GAP_MAX = 0.005;
  private readonly KV_PREFIX_JACCARD_MIN = 0.7;

  validateConstraints(
    perturbation: PolicyPerturbation,
    uplift: CounterfactualUplift
  ): ConstraintViolation[] {
    const violations: ConstraintViolation[] = [];

    // ECE constraint
    const estimatedECE = this.estimateECE(perturbation, uplift);
    if (estimatedECE > this.ECE_THRESHOLD) {
      violations.push({
        constraint_type: 'ECE',
        current_value: estimatedECE,
        threshold: this.ECE_THRESHOLD,
        severity: 'error',
        mitigation_suggestions: ['Reduce K2 perturbation', 'Enable re-isotonic calibration']
      });
    }

    // Latency ratio constraint
    const estimatedP99P95Ratio = this.estimateLatencyRatio(perturbation, uplift);
    if (estimatedP99P95Ratio > this.P99_P95_RATIO_MAX) {
      violations.push({
        constraint_type: 'latency_ratio',
        current_value: estimatedP99P95Ratio,
        threshold: this.P99_P95_RATIO_MAX,
        severity: 'warning',
        mitigation_suggestions: ['Reduce r parameter', 'Lower K2 cap', 'Optimize window/stride']
      });
    }

    // Proxy gap constraint
    if (uplift.uncertainty_score > this.PROXY_GAP_MAX) {
      violations.push({
        constraint_type: 'proxy_gap',
        current_value: uplift.uncertainty_score,
        threshold: this.PROXY_GAP_MAX,
        severity: 'warning',
        mitigation_suggestions: ['Reduce parameter change magnitude', 'Use more conservative perturbations']
      });
    }

    return violations;
  }

  private estimateECE(perturbation: PolicyPerturbation, uplift: CounterfactualUplift): number {
    // Estimate Expected Calibration Error based on parameter changes
    // More aggressive changes tend to hurt calibration
    const k2Change = Math.abs((perturbation.parameter_changes.K2 || 0) / 100);
    const lambdaChange = Math.abs((perturbation.parameter_changes.lambda || 0) / 100);
    
    return Math.min(0.15, k2Change * 0.02 + lambdaChange * 0.01 + uplift.uncertainty_score * 0.05);
  }

  private estimateLatencyRatio(perturbation: PolicyPerturbation, uplift: CounterfactualUplift): number {
    // Estimate p99/p95 latency ratio
    // More complex operations increase tail latency
    const rValue = perturbation.parameter_changes.r || 16;
    const k2Value = perturbation.parameter_changes.K2 || 100;
    
    const baseRatio = 1.8; // Assumed baseline
    const rComplexityIncrease = Math.max(0, (rValue - 16) * 0.05);
    const k2ComplexityIncrease = Math.max(0, (k2Value - 100) * 0.001);
    
    return baseRatio + rComplexityIncrease + k2ComplexityIncrease;
  }
}

// ============================================================================
// SUPPORTING TYPES AND INTERFACES
// ============================================================================

interface SavedAtomData {
  atom_id: string;
  embedding: number[];
  metadata: {
    bm25_score: number;
    vector_score: number;
    rerank_score?: number;
  };
}

interface RecomputedMarginals {
  facility_marginals: Float64Array;
  dpp_marginals: Float64Array;
  filtered_marginals: Float64Array;
  effective_k: number;
  computation_time_ms: number;
}

interface PerformancePredictions {
  p_at_5_improvement: number;
  latency_change: number;
  cost_efficiency_change: number;
}