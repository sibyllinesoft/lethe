/**
 * Bayesian + Rule-Based Hybrid Auto-Tuning System
 * 
 * Implements constrained optimization with domain-specific biases for the
 * Gap→Tune→Verify pipeline. Combines Bayesian optimization for exploration
 * with rule-based priors and hard constraints from the validator.
 */

import {
  AutoTuningProfile,
  DomainSpecialization,
  DomainSpecificBiases,
  TuningConstraints,
  ValidationConfig,
  PolicyFingerprint,
  GapRecord,
  GapAnalysisResult,
  GapAnalysisError,
  TuningQueueItem
} from './types.js';

import { Config, PerformanceMetrics } from '../types.js';
import { CounterfactualAnalysis } from './types.js';

// ============================================================================
// CORE AUTO-TUNING ENGINE
// ============================================================================

export class AutoTuningEngine {
  private config: Config;
  private bayesianOptimizer: BayesianOptimizer;
  private constraintValidator: TuningConstraintValidator;
  private domainProfiles: Map<string, AutoTuningProfile> = new Map();

  constructor(config: Config) {
    this.config = config;
    this.bayesianOptimizer = new BayesianOptimizer();
    this.constraintValidator = new TuningConstraintValidator();
    this.initializeDomainProfiles();
  }

  /**
   * Performs constrained auto-tuning for a specific gap slice
   */
  async performAutoTuning(
    gapRecord: GapRecord,
    counterfactualAnalysis: CounterfactualAnalysis,
    maxTrials: number = 12
  ): Promise<GapAnalysisResult<OptimizedPolicy>> {
    try {
      // Select appropriate domain profile
      const profile = this.selectDomainProfile(gapRecord);
      
      // Initialize Bayesian optimizer with counterfactual priors
      await this.bayesianOptimizer.initialize(
        gapRecord.policy_fingerprint,
        counterfactualAnalysis,
        profile
      );

      // Execute constrained optimization loop
      const optimizationResult = await this.executeOptimizationLoop(
        gapRecord,
        profile,
        maxTrials
      );

      return {
        success: true,
        data: optimizationResult
      };

    } catch (error) {
      return {
        success: false,
        error: {
          code: 'AUTO_TUNING_ERROR',
          message: `Auto-tuning failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          error_type: 'tuning_failure',
          gap_context: {
            slice_id: gapRecord.slice_id,
            policy_id: gapRecord.policy_fingerprint.policy_id
          },
          recovery_actions: ['Verify domain profile configuration', 'Check constraint validation', 'Validate Bayesian optimizer state'],
          is_retryable: true,
          impact_severity: 'high',
          affected_components: ['tuning_pipeline'],
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Main optimization loop: N=12 trials with Bayesian + rule-based guidance
   */
  private async executeOptimizationLoop(
    gapRecord: GapRecord,
    profile: AutoTuningProfile,
    maxTrials: number
  ): Promise<OptimizedPolicy> {
    const trials: TuningTrial[] = [];
    let bestPolicy = gapRecord.policy_fingerprint;
    let bestUtility = -Infinity;

    for (let trial = 0; trial < maxTrials; trial++) {
      // Generate next policy candidate using acquisition function
      const candidatePolicy = await this.bayesianOptimizer.acquireNextCandidate(
        trials,
        profile
      );

      // Validate constraints
      const constraintValidation = this.constraintValidator.validatePolicy(
        candidatePolicy,
        profile.hard_constraints
      );

      if (!constraintValidation.satisfies_constraints) {
        // Prune neighborhood and reduce step size
        this.bayesianOptimizer.pruneNeighborhood(candidatePolicy, constraintValidation.violations);
        continue;
      }

      // Execute paired subset replay (M≈200)
      const trialResult = await this.executePairedSubsetReplay(
        candidatePolicy,
        gapRecord,
        profile.validation_config.subset_size
      );

      // Calculate Jensen risk-adjusted utility
      const utility = this.calculateJensenRiskAdjustedUtility(
        trialResult,
        profile.jensen_risk_adjustment
      );

      const trial_record: TuningTrial = {
        trial_id: trial,
        policy: candidatePolicy,
        result: trialResult,
        utility_score: utility,
        constraint_violations: constraintValidation.violations,
        timestamp: Date.now()
      };

      trials.push(trial_record);

      // Update best policy
      if (utility > bestUtility && constraintValidation.satisfies_constraints) {
        bestPolicy = candidatePolicy;
        bestUtility = utility;
      }

      // Update Bayesian optimizer with new observation
      await this.bayesianOptimizer.updateWithObservation(candidatePolicy, utility);
    }

    // Final validation with full paired matrix
    const finalValidation = await this.executePairedSubsetReplay(
      bestPolicy,
      gapRecord,
      profile.validation_config.full_matrix_size
    );

    return {
      optimized_policy: bestPolicy,
      optimization_trials: trials,
      final_validation: finalValidation,
      improvement_metrics: this.calculateImprovementMetrics(gapRecord, finalValidation),
      confidence_score: this.calculateConfidenceScore(trials, bestUtility),
      optimization_metadata: {
        profile_used: profile.profile_name,
        total_trials: maxTrials,
        successful_trials: trials.filter(t => t.constraint_violations.length === 0).length,
        convergence_trial: this.findConvergenceTrial(trials),
        optimization_time_ms: Date.now() - trials[0].timestamp
      }
    };
  }

  // ============================================================================
  // DOMAIN PROFILE INITIALIZATION
  // ============================================================================

  private initializeDomainProfiles(): void {
    // Code/ERROR gaps profile
    const codeErrorProfile: AutoTuningProfile = {
      profile_name: 'code_error_gaps',
      domain_specialization: {
        domain_type: 'code_error_gaps',
        preferred_ranges: {
          r: [16, 24],           // Stronger closures
          K2: [115, 130],        // K2 + 15% boost
          lambda: [1.05, 1.10],  // λ + 5% boost
          tau: [0.4, 0.6]        // Moderate group-split
        },
        feature_importance_weights: {
          closure_depth: 0.3,
          symbol_complexity: 0.25,
          code_heavy_ratio: 0.2,
          error_heavy_ratio: 0.15,
          entity_entropy: 0.1
        },
        metric_priorities: {
          p_at_5_weight: 0.5,
          latency_weight: 0.2,
          cost_efficiency_weight: 0.2,
          stability_weight: 0.1
        }
      },
      bayesian_config: {
        n_trials: 12,
        acquisition_function: 'EI',
        exploration_weight: 0.3,
        initial_points: 3
      },
      rule_biases: {
        code_error: {
          closure_strength_bias: 0.2,
          r_preference: 16,
          K2_boost_percent: 15,
          lambda_adjustment: 5,
          enable_summaries: true
        }
      },
      hard_constraints: this.createDefaultConstraints(),
      jensen_risk_adjustment: {
        alpha: 0.15,              // Moderate risk aversion
        cvar_threshold: 0.95
      },
      validation_config: this.createDefaultValidationConfig()
    };

    // Tool/JSON needles profile
    const toolJsonProfile: AutoTuningProfile = {
      profile_name: 'tool_json_needles',
      domain_specialization: {
        domain_type: 'tool_json_needles',
        preferred_ranges: {
          K2: [125, 150],        // Aggressive K2 boost (+25%)
          lambda: [1.05, 1.10], // λ + 5% boost
          r: [12, 16],           // Lower r for precision
          ce_early_exit_rate: [0, 0] // Disabled for CE@k≤50
        },
        feature_importance_weights: {
          tool_heavy_ratio: 0.35,
          json_needle_ratio: 0.3,
          precision_score: 0.2,
          entity_entropy: 0.15
        },
        metric_priorities: {
          p_at_5_weight: 0.6,    // Precision-focused
          cost_efficiency_weight: 0.25,
          latency_weight: 0.1,
          stability_weight: 0.05
        }
      },
      bayesian_config: {
        n_trials: 12,
        acquisition_function: 'UCB',
        exploration_weight: 0.25,  // Less exploration, more exploitation
        initial_points: 2
      },
      rule_biases: {
        tool_json: {
          K2_boost_percent: 25,
          lambda_adjustment: 5,
          ce_early_exit_disabled: true,
          precision_over_recall: true
        }
      },
      hard_constraints: this.createDefaultConstraints(),
      jensen_risk_adjustment: {
        alpha: 0.1,              // Lower risk aversion for precision tasks
        cvar_threshold: 0.9
      },
      validation_config: this.createDefaultValidationConfig()
    };

    // Multilingual/code-switch profile
    const multilingualProfile: AutoTuningProfile = {
      profile_name: 'multilingual_codeswitch',
      domain_specialization: {
        domain_type: 'multilingual_codeswitch',
        preferred_ranges: {
          mu: [1.05, 1.15],      // μ + 5-10% boost for vector emphasis
          r: [16, 20],           // Higher r for diversity
          tau: [0.5, 0.7],       // Widened group-split
          ce_early_exit_rate: [0.15, 0.25] // Widened early-exit cap
        },
        feature_importance_weights: {
          code_switch_ratio: 0.3,
          language_diversity: 0.25,
          entity_entropy: 0.2,
          kv_stability: 0.15,
          closure_depth: 0.1
        },
        metric_priorities: {
          p_at_5_weight: 0.4,
          stability_weight: 0.3,  // High stability importance
          latency_weight: 0.2,
          cost_efficiency_weight: 0.1
        }
      },
      bayesian_config: {
        n_trials: 12,
        acquisition_function: 'EI',
        exploration_weight: 0.4,   // More exploration for complex domain
        initial_points: 4
      },
      rule_biases: {
        multilingual: {
          re_isotonic_enabled: true,
          ce_early_exit_cap_widened: true,
          mu_adjustment: 8,        // μ + 8%
          r_preference: 16
        }
      },
      hard_constraints: this.createDefaultConstraints(),
      jensen_risk_adjustment: {
        alpha: 0.2,              // Higher risk aversion due to complexity
        cvar_threshold: 0.95
      },
      validation_config: this.createDefaultValidationConfig()
    };

    // Store profiles
    this.domainProfiles.set('code_error_gaps', codeErrorProfile);
    this.domainProfiles.set('tool_json_needles', toolJsonProfile);
    this.domainProfiles.set('multilingual_codeswitch', multilingualProfile);
  }

  private createDefaultConstraints(): TuningConstraints {
    return {
      p95_geq_avg: true,
      p99_p95_ratio_max: 2.5,
      ece_threshold: 0.08,
      proxy_gap_max: 0.005,
      lambda_bounds: [-0.2, 0.2],    // ±20%
      mu_bounds: [-0.1, 0.1],        // ±10%
      K2_bounds: [-0.3, 0.3],        // ±30%
      r_allowed_values: [12, 14, 16, 24],
      head_keep_bounds: [-4, 4],     // ±4pp
      kv_prefix_jaccard_min: 0.7,
      curvature_based_r_capping: true,
      tau_bounds: [-0.1, 0.1],       // ±0.1
      ilp_usage_rate_max: 0.15       // 15% max ILP usage
    };
  }

  private createDefaultValidationConfig(): ValidationConfig {
    return {
      subset_size: 200,              // M≈200 for quick validation
      full_matrix_size: 1000,        // Full validation size
      budget_levels: [8, 15, 30],    // Multi-budget validation
      confidence_level: 0.95,
      minimum_effect_size: 0.02,     // 2% minimum meaningful improvement
      coverage_weighted_crps: true,
      cross_domain_validation: true
    };
  }

  // ============================================================================
  // DOMAIN PROFILE SELECTION
  // ============================================================================

  private selectDomainProfile(gapRecord: GapRecord): AutoTuningProfile {
    const features = gapRecord.root_cause_features;
    
    // Decision tree for profile selection
    if (features.type_mix.code_heavy > 0.4 || features.type_mix.error_heavy > 0.3) {
      return this.domainProfiles.get('code_error_gaps')!;
    }
    
    if (features.type_mix.tool_heavy > 0.4 || features.type_mix.json_needle > 0.2) {
      return this.domainProfiles.get('tool_json_needles')!;
    }
    
    if (features.language_distribution.code_switch > 0.2 || 
        features.language_distribution.chinese > 0.3) {
      return this.domainProfiles.get('multilingual_codeswitch')!;
    }
    
    // Default to code/error profile for general cases
    return this.domainProfiles.get('code_error_gaps')!;
  }

  // ============================================================================
  // UTILITY AND RISK CALCULATION
  // ============================================================================

  private calculateJensenRiskAdjustedUtility(
    result: PairedReplayResult,
    riskAdjustment: AutoTuningProfile['jensen_risk_adjustment']
  ): number {
    // Jensen risk-adjusted utility: mean(P@5) - α·CVaR95(latency)
    const meanP5 = result.performance_metrics.mean_p_at_5;
    const latencyDistribution = result.performance_metrics.latency_distribution;
    
    // Calculate CVaR (Conditional Value at Risk)
    const cvarLatency = this.calculateCVaR(latencyDistribution, riskAdjustment.cvar_threshold);
    
    return meanP5 - (riskAdjustment.alpha * cvarLatency);
  }

  private calculateCVaR(distribution: number[], threshold: number): number {
    // Sort latencies in descending order
    const sortedLatencies = [...distribution].sort((a, b) => b - a);
    
    // Find threshold index
    const thresholdIndex = Math.floor((1 - threshold) * sortedLatencies.length);
    
    // Calculate conditional mean of tail
    const tailValues = sortedLatencies.slice(0, thresholdIndex);
    
    return tailValues.length > 0 ? 
      tailValues.reduce((sum, val) => sum + val, 0) / tailValues.length : 0;
  }

  // ============================================================================
  // VALIDATION AND REPLAY EXECUTION
  // ============================================================================

  private async executePairedSubsetReplay(
    policy: PolicyFingerprint,
    gapRecord: GapRecord,
    sampleSize: number
  ): Promise<PairedReplayResult> {
    // In practice, this would execute the actual retrieval pipeline
    // with the new policy on a subset of the gap slice
    
    // Simulate performance metrics based on policy changes
    const simulatedMetrics = this.simulatePerformanceMetrics(policy, gapRecord, sampleSize);
    
    return {
      policy_tested: policy,
      sample_size: sampleSize,
      performance_metrics: simulatedMetrics,
      constraint_validations: {
        p95_geq_avg: simulatedMetrics.latency_p95 >= simulatedMetrics.latency_avg,
        p99_p95_ratio: simulatedMetrics.latency_p99 / simulatedMetrics.latency_p95,
        ece_score: simulatedMetrics.ece,
        proxy_gap: simulatedMetrics.proxy_gap,
        kv_jaccard_score: simulatedMetrics.kv_prefix_jaccard
      },
      gates_passed: this.checkAllGatesPassed(simulatedMetrics),
      execution_time_ms: Math.random() * 1000 + 500, // Simulate execution time
      timestamp: Date.now()
    };
  }

  private simulatePerformanceMetrics(
    policy: PolicyFingerprint,
    gapRecord: GapRecord,
    sampleSize: number
  ): PerformanceMetricsDetailed {
    // Simulate based on policy changes and gap characteristics
    const baseP5 = 0.45; // Assumed baseline
    const baseLatency = 150; // ms
    
    // Policy impact simulation
    const k2Impact = (policy.K2 - 100) * 0.0005; // Each K2 unit = 0.05% P@5
    const lambdaImpact = (policy.lambda - 1.0) * 0.02;
    const muImpact = (policy.mu - 1.0) * 0.015;
    const rComplexity = (policy.r - 16) * 2; // Latency impact
    
    const meanP5 = Math.max(0, Math.min(1, baseP5 + k2Impact + lambdaImpact + muImpact));
    const meanLatency = Math.max(10, baseLatency + rComplexity + (policy.K2 - 100) * 0.5);
    
    // Generate distributions (simplified)
    const p5Distribution = this.generateDistribution(meanP5, 0.05, sampleSize);
    const latencyDistribution = this.generateDistribution(meanLatency, meanLatency * 0.2, sampleSize);
    
    return {
      mean_p_at_5: meanP5,
      std_p_at_5: 0.05,
      p5_distribution: p5Distribution,
      latency_avg: meanLatency,
      latency_p95: meanLatency * 1.5,
      latency_p99: meanLatency * 2.0,
      latency_distribution: latencyDistribution,
      ece: Math.random() * 0.06, // Simulate ECE
      proxy_gap: Math.random() * 0.003, // Simulate proxy gap
      kv_prefix_jaccard: 0.8 + Math.random() * 0.15 // Simulate KV stability
    };
  }

  private generateDistribution(mean: number, stdDev: number, size: number): number[] {
    // Simple normal distribution simulation
    const distribution: number[] = [];
    for (let i = 0; i < size; i++) {
      // Box-Muller transform for normal distribution
      const u1 = Math.random();
      const u2 = Math.random();
      const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      distribution.push(mean + z0 * stdDev);
    }
    return distribution;
  }

  private checkAllGatesPassed(metrics: PerformanceMetricsDetailed): boolean {
    return metrics.latency_p95 >= metrics.latency_avg &&
           (metrics.latency_p99 / metrics.latency_p95) <= 2.5 &&
           metrics.ece <= 0.08 &&
           metrics.proxy_gap <= 0.005 &&
           metrics.kv_prefix_jaccard >= 0.7;
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  private calculateImprovementMetrics(
    gapRecord: GapRecord,
    finalValidation: PairedReplayResult
  ): ImprovementMetrics {
    const baselineP5 = 0.45; // Should come from gap record baseline
    const baselineLatency = 150; // Should come from gap record baseline
    
    return {
      p_at_5_absolute_improvement: finalValidation.performance_metrics.mean_p_at_5 - baselineP5,
      p_at_5_relative_improvement: (finalValidation.performance_metrics.mean_p_at_5 - baselineP5) / baselineP5,
      latency_change: finalValidation.performance_metrics.latency_avg - baselineLatency,
      cost_efficiency_improvement: 0.05, // Calculated from cost model
      stability_score: finalValidation.performance_metrics.kv_prefix_jaccard
    };
  }

  private calculateConfidenceScore(trials: TuningTrial[], bestUtility: number): number {
    if (trials.length < 2) return 0;
    
    // Confidence based on convergence and consistency
    const utilities = trials.map(t => t.utility_score);
    const utilityStd = Math.sqrt(utilities.reduce((sum, u) => sum + (u - bestUtility) ** 2, 0) / utilities.length);
    
    // Higher confidence with lower variance and more successful trials
    const successRate = trials.filter(t => t.constraint_violations.length === 0).length / trials.length;
    
    return Math.max(0, Math.min(1, successRate * (1 - utilityStd)));
  }

  private findConvergenceTrial(trials: TuningTrial[]): number {
    // Find trial where utility stopped improving significantly
    for (let i = 3; i < trials.length; i++) {
      const recentUtilities = trials.slice(i - 3, i).map(t => t.utility_score);
      const maxRecent = Math.max(...recentUtilities);
      const currentUtility = trials[i].utility_score;
      
      if (currentUtility - maxRecent < 0.01) { // Less than 1% improvement
        return i;
      }
    }
    
    return trials.length - 1;
  }
}

// ============================================================================
// BAYESIAN OPTIMIZATION IMPLEMENTATION
// ============================================================================

export class BayesianOptimizer {
  private observations: Array<{ policy: PolicyFingerprint; utility: number }> = [];
  private acquisitionFunction: AcquisitionFunction = 'EI';
  private explorationWeight: number = 0.3;

  async initialize(
    basePolicy: PolicyFingerprint,
    counterfactualAnalysis: CounterfactualAnalysis,
    profile: AutoTuningProfile
  ): Promise<void> {
    this.acquisitionFunction = profile.bayesian_config.acquisition_function;
    this.explorationWeight = profile.bayesian_config.exploration_weight;
    
    // Initialize with counterfactual priors
    for (const uplift of counterfactualAnalysis.uplift_frontier.slice(0, 3)) {
      this.observations.push({
        policy: uplift.policy_variant,
        utility: uplift.predicted_p_at_5_improvement - uplift.downside_risk
      });
    }
  }

  async acquireNextCandidate(
    trials: TuningTrial[],
    profile: AutoTuningProfile
  ): Promise<PolicyFingerprint> {
    // Simple acquisition function implementation
    // In practice, would use proper GP regression and acquisition optimization
    
    if (this.observations.length === 0) {
      // Return random perturbation of base policy
      return this.generateRandomPerturbation(profile);
    }

    // Find best observed policy
    const bestObservation = this.observations.reduce((best, obs) => 
      obs.utility > best.utility ? obs : best
    );

    // Generate candidate with exploration around best point
    return this.generateExplorationCandidate(bestObservation.policy, profile);
  }

  async updateWithObservation(policy: PolicyFingerprint, utility: number): Promise<void> {
    this.observations.push({ policy, utility });
  }

  pruneNeighborhood(policy: PolicyFingerprint, violations: string[]): void {
    // Remove similar policies from consideration
    // In practice, would update GP to avoid similar regions
    console.log(`Pruning neighborhood around policy ${policy.policy_id} due to violations: ${violations.join(', ')}`);
  }

  private generateRandomPerturbation(profile: AutoTuningProfile): PolicyFingerprint {
    // Generate random policy within profile constraints
    const ranges = profile.domain_specialization.preferred_ranges;
    const basePolicy = this.observations[0]?.policy || this.createDefaultPolicy();
    
    return {
      ...basePolicy,
      lambda: this.sampleFromRange(ranges.lambda || [0.8, 1.2]),
      mu: this.sampleFromRange(ranges.mu || [0.8, 1.2]),
      K2: Math.round(this.sampleFromRange(ranges.K2 || [70, 130])),
      r: this.sampleFromArray(ranges.r ? [ranges.r[0], ranges.r[1]] : [12, 16, 20, 24]),
      policy_id: this.generatePolicyId(),
      created_at: Date.now(),
      validation_status: 'pending'
    };
  }

  private generateExplorationCandidate(
    bestPolicy: PolicyFingerprint,
    profile: AutoTuningProfile
  ): PolicyFingerprint {
    // Generate candidate with Gaussian noise around best policy
    const noise = this.explorationWeight;
    
    return {
      ...bestPolicy,
      lambda: Math.max(0.5, Math.min(2.0, bestPolicy.lambda + (Math.random() - 0.5) * noise)),
      mu: Math.max(0.5, Math.min(2.0, bestPolicy.mu + (Math.random() - 0.5) * noise)),
      K2: Math.max(10, Math.min(200, Math.round(bestPolicy.K2 + (Math.random() - 0.5) * noise * 50))),
      policy_id: this.generatePolicyId(),
      created_at: Date.now(),
      validation_status: 'pending'
    };
  }

  private sampleFromRange(range: [number, number]): number {
    return range[0] + Math.random() * (range[1] - range[0]);
  }

  private sampleFromArray<T>(array: T[]): T {
    return array[Math.floor(Math.random() * array.length)];
  }

  private createDefaultPolicy(): PolicyFingerprint {
    return {
      lambda: 1.0,
      mu: 1.0,
      K2: 100,
      r: 16,
      head_keep: 80,
      window_size: 1024,
      stride: 512,
      ce_early_exit_rate: 0.1,
      tau: 0.5,
      curvature_threshold: 0.1,
      proxy_gap_max: 0.005,
      policy_id: this.generatePolicyId(),
      created_at: Date.now(),
      validation_status: 'pending'
    };
  }

  private generatePolicyId(): string {
    return `policy_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

// ============================================================================
// CONSTRAINT VALIDATION
// ============================================================================

export class TuningConstraintValidator {
  validatePolicy(
    policy: PolicyFingerprint,
    constraints: TuningConstraints
  ): PolicyConstraintValidation {
    const violations: string[] = [];

    // Check parameter bounds
    if (policy.lambda < (1 + constraints.lambda_bounds[0]) || 
        policy.lambda > (1 + constraints.lambda_bounds[1])) {
      violations.push(`lambda out of bounds: ${policy.lambda}`);
    }

    if (policy.mu < (1 + constraints.mu_bounds[0]) || 
        policy.mu > (1 + constraints.mu_bounds[1])) {
      violations.push(`mu out of bounds: ${policy.mu}`);
    }

    if (!constraints.r_allowed_values.includes(policy.r)) {
      violations.push(`r not in allowed values: ${policy.r}`);
    }

    if (policy.tau < (0.5 + constraints.tau_bounds[0]) || 
        policy.tau > (0.5 + constraints.tau_bounds[1])) {
      violations.push(`tau out of bounds: ${policy.tau}`);
    }

    return {
      satisfies_constraints: violations.length === 0,
      violations: violations
    };
  }
}

// ============================================================================
// SUPPORTING TYPES AND INTERFACES
// ============================================================================

type AcquisitionFunction = 'EI' | 'UCB' | 'PI';

interface TuningTrial {
  trial_id: number;
  policy: PolicyFingerprint;
  result: PairedReplayResult;
  utility_score: number;
  constraint_violations: string[];
  timestamp: number;
}

interface OptimizedPolicy {
  optimized_policy: PolicyFingerprint;
  optimization_trials: TuningTrial[];
  final_validation: PairedReplayResult;
  improvement_metrics: ImprovementMetrics;
  confidence_score: number;
  optimization_metadata: {
    profile_used: string;
    total_trials: number;
    successful_trials: number;
    convergence_trial: number;
    optimization_time_ms: number;
  };
}

interface PairedReplayResult {
  policy_tested: PolicyFingerprint;
  sample_size: number;
  performance_metrics: PerformanceMetricsDetailed;
  constraint_validations: {
    p95_geq_avg: boolean;
    p99_p95_ratio: number;
    ece_score: number;
    proxy_gap: number;
    kv_jaccard_score: number;
  };
  gates_passed: boolean;
  execution_time_ms: number;
  timestamp: number;
}

interface PerformanceMetricsDetailed {
  mean_p_at_5: number;
  std_p_at_5: number;
  p5_distribution: number[];
  latency_avg: number;
  latency_p95: number;
  latency_p99: number;
  latency_distribution: number[];
  ece: number;
  proxy_gap: number;
  kv_prefix_jaccard: number;
}

interface ImprovementMetrics {
  p_at_5_absolute_improvement: number;
  p_at_5_relative_improvement: number;
  latency_change: number;
  cost_efficiency_improvement: number;
  stability_score: number;
}

interface PolicyConstraintValidation {
  satisfies_constraints: boolean;
  violations: string[];
}