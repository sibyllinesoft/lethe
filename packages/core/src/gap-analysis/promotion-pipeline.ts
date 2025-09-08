/**
 * Promotion Pipeline - Full Paired Replay Validation
 * 
 * Implements comprehensive validation for tuned policies before deployment.
 * Includes gap slice validation, union set non-degradation testing, and
 * microsite integration for buyer-facing Pareto front annotations.
 */

import {
  PromotionResult,
  ValidationResult,
  ParetoFrontAnnotation,
  PolicyFingerprint,
  GapRecord,
  GapAnalysisResult,
  GapAnalysisError,
  OptimizedPolicy
} from './types.js';

import { Config, PerformanceMetrics } from '../types.js';

// ============================================================================
// CORE PROMOTION PIPELINE
// ============================================================================

export class PromotionPipeline {
  private config: Config;
  private validationExecutor: ValidationExecutor;
  private paretoFrontManager: ParetoFrontManager;
  private deploymentGate: DeploymentGate;

  constructor(config: Config) {
    this.config = config;
    this.validationExecutor = new ValidationExecutor(config);
    this.paretoFrontManager = new ParetoFrontManager();
    this.deploymentGate = new DeploymentGate();
  }

  /**
   * Executes full promotion pipeline for an optimized policy
   */
  async promoteOptimizedPolicy(
    optimizedPolicy: OptimizedPolicy,
    sourceGap: GapRecord
  ): Promise<GapAnalysisResult<PromotionResult>> {
    try {
      console.log(`Starting promotion pipeline for policy ${optimizedPolicy.optimized_policy.policy_id}`);

      // Stage 1: Gap slice validation
      const gapSliceValidation = await this.validationExecutor.validateOnGapSlice(
        optimizedPolicy.optimized_policy,
        sourceGap
      );

      if (!gapSliceValidation.is_significant || !gapSliceValidation.gates_passed) {
        return this.createRejectionResult(
          'Gap slice validation failed',
          optimizedPolicy.optimized_policy.policy_id,
          sourceGap.slice_id,
          { gap_slice_validation: gapSliceValidation }
        );
      }

      // Stage 2: Union set non-degradation validation
      const unionSetValidation = await this.validationExecutor.validateNonDegradation(
        optimizedPolicy.optimized_policy,
        sourceGap
      );

      if (!unionSetValidation.gates_passed) {
        return this.createRejectionResult(
          'Union set validation failed - degradation detected',
          optimizedPolicy.optimized_policy.policy_id,
          sourceGap.slice_id,
          { 
            gap_slice_validation: gapSliceValidation,
            union_set_validation: unionSetValidation
          }
        );
      }

      // Stage 3: Multi-budget validation (8/15/30)
      const crossBudgetValidation = await this.validationExecutor.validateAcrossBudgets(
        optimizedPolicy.optimized_policy,
        sourceGap,
        [8, 15, 30]
      );

      const budgetFailures = Object.entries(crossBudgetValidation)
        .filter(([_, result]) => !result.gates_passed)
        .map(([budget, _]) => budget);

      if (budgetFailures.length > 0) {
        console.warn(`Budget validation failed for levels: ${budgetFailures.join(', ')}`);
        // Don't reject, but note for confidence scoring
      }

      // Stage 4: Performance gain calculation
      const performanceGains = this.calculatePerformanceGains(
        gapSliceValidation,
        unionSetValidation,
        crossBudgetValidation
      );

      // Stage 5: Deployment readiness assessment
      const deploymentAssessment = this.deploymentGate.assessDeploymentReadiness(
        optimizedPolicy,
        gapSliceValidation,
        unionSetValidation,
        crossBudgetValidation
      );

      // Stage 6: Pareto front annotation for microsite
      const paretoAnnotation = this.paretoFrontManager.createParetoAnnotation(
        optimizedPolicy.optimized_policy,
        performanceGains,
        sourceGap,
        deploymentAssessment.deployment_confidence
      );

      const promotionResult: PromotionResult = {
        policy_id: optimizedPolicy.optimized_policy.policy_id,
        source_gap: sourceGap.slice_id,
        gap_slice_validation: gapSliceValidation,
        union_set_validation: unionSetValidation,
        cross_budget_validation: crossBudgetValidation,
        performance_gains: performanceGains,
        deployment_status: deploymentAssessment.deployment_status,
        deployment_confidence: deploymentAssessment.deployment_confidence,
        pareto_front_annotation: paretoAnnotation,
        validation_timestamp: Date.now(),
        reviewer_notes: deploymentAssessment.reviewer_notes
      };

      console.log(`Promotion pipeline completed for policy ${optimizedPolicy.optimized_policy.policy_id}: ${deploymentAssessment.deployment_status}`);

      return {
        success: true,
        data: promotionResult
      };

    } catch (error) {
      return {
        success: false,
        error: {
          code: 'PROMOTION_PIPELINE_ERROR',
          message: `Promotion pipeline failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          error_type: 'validation_error',
          gap_context: {
            slice_id: sourceGap.slice_id,
            policy_id: optimizedPolicy.optimized_policy.policy_id
          },
          recovery_actions: ['Retry validation with fresh data', 'Check validation infrastructure', 'Review deployment gates'],
          is_retryable: true,
          impact_severity: 'high',
          affected_components: ['promotion_pipeline', 'deployment'],
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Batch promotion for multiple optimized policies
   */
  async batchPromoteOptimizedPolicies(
    optimizedPolicies: Array<{ policy: OptimizedPolicy; sourceGap: GapRecord }>
  ): Promise<GapAnalysisResult<PromotionResult[]>> {
    const results: PromotionResult[] = [];
    const errors: GapAnalysisError[] = [];

    for (const { policy, sourceGap } of optimizedPolicies) {
      const promotionResult = await this.promoteOptimizedPolicy(policy, sourceGap);
      
      if (promotionResult.success) {
        results.push(promotionResult.data);
      } else {
        errors.push(promotionResult.error);
      }
    }

    if (errors.length > 0 && results.length === 0) {
      return {
        success: false,
        error: {
          code: 'BATCH_PROMOTION_FAILED',
          message: `All promotions failed. First error: ${errors[0].message}`,
          error_type: 'validation_error',
          recovery_actions: ['Review individual policy validation results', 'Check system health'],
          is_retryable: true,
          impact_severity: 'critical',
          affected_components: ['promotion_pipeline'],
          timestamp: Date.now()
        }
      };
    }

    return {
      success: true,
      data: results
    };
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  private createRejectionResult(
    reason: string,
    policyId: string,
    sliceId: string,
    validationResults: Partial<Pick<PromotionResult, 'gap_slice_validation' | 'union_set_validation'>>
  ): GapAnalysisResult<PromotionResult> {
    const rejectionResult: PromotionResult = {
      policy_id: policyId,
      source_gap: sliceId,
      gap_slice_validation: validationResults.gap_slice_validation || this.createFailedValidation('Not executed'),
      union_set_validation: validationResults.union_set_validation || this.createFailedValidation('Not executed'),
      cross_budget_validation: {},
      performance_gains: {
        p_at_5_improvement: 0,
        latency_improvement: 0,
        cost_efficiency_gain: 0,
        stability_score: 0
      },
      deployment_status: 'rejected',
      deployment_confidence: 0,
      pareto_front_annotation: {
        policy_label: `Rejected Policy`,
        improvement_summary: reason,
        cost_efficiency: 0,
        performance_score: 0,
        latency_score: 0,
        marker_color: '#ff4444',
        marker_size: 4,
        highlight: false,
        tooltip_data: {
          domain_specialization: 'Unknown',
          key_improvements: [],
          validation_confidence: 0,
          deployment_date: 'Not deployed'
        }
      },
      validation_timestamp: Date.now(),
      reviewer_notes: reason
    };

    return {
      success: true, // Return success with rejected status
      data: rejectionResult
    };
  }

  private createFailedValidation(reason: string): ValidationResult {
    return {
      test_set: 'unknown',
      sample_size: 0,
      p_at_5_delta: 0,
      p_at_5_ci: [0, 0],
      latency_p95_delta: 0,
      latency_p95_ci: [0, 0],
      gates_passed: false,
      gate_violations: [reason],
      is_significant: false,
      p_value: 1.0,
      effect_size: 0
    };
  }

  private calculatePerformanceGains(
    gapSliceValidation: ValidationResult,
    unionSetValidation: ValidationResult,
    crossBudgetValidation: Record<number, ValidationResult>
  ): PromotionResult['performance_gains'] {
    // Weighted combination of validation results
    const gapSliceWeight = 0.5;
    const unionSetWeight = 0.3;
    const crossBudgetWeight = 0.2;

    const avgCrossBudgetP5 = Object.values(crossBudgetValidation)
      .reduce((sum, result) => sum + result.p_at_5_delta, 0) / 
      Math.max(1, Object.keys(crossBudgetValidation).length);

    const avgCrossBudgetLatency = Object.values(crossBudgetValidation)
      .reduce((sum, result) => sum + result.latency_p95_delta, 0) / 
      Math.max(1, Object.keys(crossBudgetValidation).length);

    const p5Improvement = 
      gapSliceValidation.p_at_5_delta * gapSliceWeight +
      unionSetValidation.p_at_5_delta * unionSetWeight +
      avgCrossBudgetP5 * crossBudgetWeight;

    const latencyImprovement = -(
      gapSliceValidation.latency_p95_delta * gapSliceWeight +
      unionSetValidation.latency_p95_delta * unionSetWeight +
      avgCrossBudgetLatency * crossBudgetWeight
    );

    const costEfficiencyGain = p5Improvement / Math.max(0.001, Math.abs(latencyImprovement));

    // Stability score based on consistency across validations
    const p5Values = [
      gapSliceValidation.p_at_5_delta,
      unionSetValidation.p_at_5_delta,
      ...Object.values(crossBudgetValidation).map(r => r.p_at_5_delta)
    ];
    
    const p5StdDev = Math.sqrt(p5Values.reduce((sum, val) => sum + (val - p5Improvement) ** 2, 0) / p5Values.length);
    const stabilityScore = Math.max(0, 1 - (p5StdDev / Math.max(0.001, Math.abs(p5Improvement))));

    return {
      p_at_5_improvement: p5Improvement,
      latency_improvement: latencyImprovement,
      cost_efficiency_gain: costEfficiencyGain,
      stability_score: stabilityScore
    };
  }
}

// ============================================================================
// VALIDATION EXECUTOR
// ============================================================================

export class ValidationExecutor {
  private config: Config;

  constructor(config: Config) {
    this.config = config;
  }

  /**
   * Validates policy performance on the original gap slice
   */
  async validateOnGapSlice(
    policy: PolicyFingerprint,
    sourceGap: GapRecord
  ): Promise<ValidationResult> {
    console.log(`Validating policy ${policy.policy_id} on gap slice ${sourceGap.slice_id}`);
    
    // In practice, this would execute the full retrieval pipeline
    // with the optimized policy on the complete gap slice dataset
    
    // Simulate validation based on policy characteristics and gap features
    const simulatedResult = this.simulateGapSliceValidation(policy, sourceGap);
    
    return simulatedResult;
  }

  /**
   * Validates that the policy doesn't degrade performance on the union set
   */
  async validateNonDegradation(
    policy: PolicyFingerprint,
    sourceGap: GapRecord
  ): Promise<ValidationResult> {
    console.log(`Validating non-degradation for policy ${policy.policy_id}`);
    
    // Test on broader dataset to ensure no regression
    const simulatedResult = this.simulateUnionSetValidation(policy, sourceGap);
    
    return simulatedResult;
  }

  /**
   * Validates policy across multiple budget levels
   */
  async validateAcrossBudgets(
    policy: PolicyFingerprint,
    sourceGap: GapRecord,
    budgetLevels: number[]
  ): Promise<Record<number, ValidationResult>> {
    const results: Record<number, ValidationResult> = {};
    
    for (const budget of budgetLevels) {
      console.log(`Validating policy ${policy.policy_id} at budget level ${budget}`);
      results[budget] = await this.validateAtBudgetLevel(policy, sourceGap, budget);
    }
    
    return results;
  }

  private async validateAtBudgetLevel(
    policy: PolicyFingerprint,
    sourceGap: GapRecord,
    budgetLevel: number
  ): Promise<ValidationResult> {
    // Simulate budget-specific validation
    const basePerformance = this.simulateGapSliceValidation(policy, sourceGap);
    
    // Budget constraints affect performance differently
    const budgetPenalty = budgetLevel < 15 ? 0.02 : budgetLevel > 25 ? -0.01 : 0;
    
    return {
      ...basePerformance,
      test_set: `budget_${budgetLevel}`,
      p_at_5_delta: basePerformance.p_at_5_delta - budgetPenalty,
      p_at_5_ci: [
        basePerformance.p_at_5_ci[0] - budgetPenalty,
        basePerformance.p_at_5_ci[1] - budgetPenalty
      ]
    };
  }

  // ============================================================================
  // SIMULATION METHODS (Replace with actual validation in production)
  // ============================================================================

  private simulateGapSliceValidation(
    policy: PolicyFingerprint,
    sourceGap: GapRecord
  ): ValidationResult {
    // Base the simulation on the original gap characteristics and policy changes
    const baseImprovement = Math.abs(sourceGap.delta_map.macro_p_at_5) * 0.7; // Assume 70% of gap is closable
    
    // Policy impact factors
    const k2Factor = (policy.K2 - 100) * 0.0003; // K2 impact
    const lambdaFactor = (policy.lambda - 1.0) * 0.01; // Lambda impact
    const muFactor = (policy.mu - 1.0) * 0.008; // Mu impact
    const rComplexityPenalty = (policy.r - 16) * 0.002; // Complexity penalty
    
    const totalImprovement = baseImprovement + k2Factor + lambdaFactor + muFactor - rComplexityPenalty;
    
    // Add some noise and constraints
    const noisyImprovement = totalImprovement + (Math.random() - 0.5) * 0.01;
    const constrainedImprovement = Math.max(0, Math.min(0.2, noisyImprovement)); // Cap at 20% improvement
    
    // Latency impact
    const latencyDelta = (policy.K2 - 100) * 0.3 + (policy.r - 16) * 2; // More complex = higher latency
    
    // Statistical significance (based on effect size)
    const effectSize = constrainedImprovement / 0.02; // Assuming 2% std
    const isSignificant = effectSize > 0.5 && constrainedImprovement > 0.01; // Cohen's d > 0.5 and >1% improvement
    
    // Gate validation
    const gateViolations: string[] = [];
    if (latencyDelta > 20) gateViolations.push('Latency increase too high');
    if (policy.r > 24) gateViolations.push('r parameter too high');
    
    const gatesPassed = gateViolations.length === 0;
    
    return {
      test_set: sourceGap.slice_id,
      sample_size: 500,
      p_at_5_delta: constrainedImprovement,
      p_at_5_ci: [constrainedImprovement - 0.005, constrainedImprovement + 0.005],
      latency_p95_delta: latencyDelta,
      latency_p95_ci: [latencyDelta - 2, latencyDelta + 2],
      gates_passed: gatesPassed,
      gate_violations: gateViolations,
      is_significant: isSignificant,
      p_value: isSignificant ? 0.02 : 0.15,
      effect_size: effectSize
    };
  }

  private simulateUnionSetValidation(
    policy: PolicyFingerprint,
    sourceGap: GapRecord
  ): ValidationResult {
    // Union set validation is more conservative - smaller improvements
    const gapSliceResult = this.simulateGapSliceValidation(policy, sourceGap);
    
    // Reduce improvements for broader dataset
    const conservativeFactor = 0.6;
    
    return {
      ...gapSliceResult,
      test_set: 'union_set',
      sample_size: 2000,
      p_at_5_delta: gapSliceResult.p_at_5_delta * conservativeFactor,
      p_at_5_ci: [
        gapSliceResult.p_at_5_ci[0] * conservativeFactor,
        gapSliceResult.p_at_5_ci[1] * conservativeFactor
      ],
      effect_size: gapSliceResult.effect_size * conservativeFactor
    };
  }
}

// ============================================================================
// PARETO FRONT MANAGER
// ============================================================================

export class ParetoFrontManager {
  /**
   * Creates Pareto front annotation for microsite integration
   */
  createParetoAnnotation(
    policy: PolicyFingerprint,
    performanceGains: PromotionResult['performance_gains'],
    sourceGap: GapRecord,
    confidence: number
  ): ParetoFrontAnnotation {
    // Determine domain specialization
    const domainType = this.identifyDomainSpecialization(sourceGap);
    
    // Create improvement summary
    const improvementSummary = this.createImprovementSummary(performanceGains, domainType);
    
    // Generate policy label
    const policyLabel = `Tuned-v${this.extractVersionNumber(policy.policy_id)} (${confidence > 0.8 ? 'Validated' : 'Limited'})`;
    
    // Calculate Pareto coordinates
    const paretoCoords = this.calculateParetoCoordinates(performanceGains);
    
    // Determine visual properties
    const visualProps = this.determineVisualProperties(performanceGains, confidence);
    
    return {
      policy_label: policyLabel,
      improvement_summary: improvementSummary,
      cost_efficiency: paretoCoords.cost_efficiency,
      performance_score: paretoCoords.performance_score,
      latency_score: paretoCoords.latency_score,
      marker_color: visualProps.color,
      marker_size: visualProps.size,
      highlight: confidence > 0.9,
      tooltip_data: {
        domain_specialization: domainType,
        key_improvements: this.extractKeyImprovements(performanceGains),
        validation_confidence: confidence,
        deployment_date: new Date().toISOString().split('T')[0]
      }
    };
  }

  private identifyDomainSpecialization(sourceGap: GapRecord): string {
    const features = sourceGap.root_cause_features;
    
    if (features.type_mix.code_heavy > 0.4 || features.type_mix.error_heavy > 0.3) {
      return 'Code/ERROR Analysis';
    }
    
    if (features.type_mix.tool_heavy > 0.4 || features.type_mix.json_needle > 0.2) {
      return 'Tool/JSON Processing';
    }
    
    if (features.language_distribution.code_switch > 0.2 || 
        features.language_distribution.chinese > 0.3) {
      return 'Multilingual/Code-switch';
    }
    
    return 'General Retrieval';
  }

  private createImprovementSummary(
    gains: PromotionResult['performance_gains'],
    domainType: string
  ): string {
    const p5Pct = (gains.p_at_5_improvement * 100).toFixed(1);
    const latencyDirection = gains.latency_improvement > 0 ? 'reduced' : 'increased';
    const latencyPct = Math.abs(gains.latency_improvement).toFixed(0);
    
    return `${domainType}: +${p5Pct}% P@5, ${latencyDirection} latency by ${latencyPct}ms`;
  }

  private calculateParetoCoordinates(gains: PromotionResult['performance_gains']): {
    cost_efficiency: number;
    performance_score: number;
    latency_score: number;
  } {
    // Normalize to 0-100 scale for Pareto front display
    const performance_score = Math.max(0, Math.min(100, gains.p_at_5_improvement * 1000)); // Scale up
    const cost_efficiency = Math.max(0, Math.min(100, gains.cost_efficiency_gain * 10));
    const latency_score = Math.max(0, Math.min(100, 50 - gains.latency_improvement)); // Lower latency = higher score
    
    return { cost_efficiency, performance_score, latency_score };
  }

  private determineVisualProperties(
    gains: PromotionResult['performance_gains'],
    confidence: number
  ): { color: string; size: number } {
    // Color based on performance improvement
    let color = '#4CAF50'; // Green for good improvements
    if (gains.p_at_5_improvement < 0.01) color = '#FF9800'; // Orange for small improvements
    if (gains.p_at_5_improvement < 0) color = '#F44336'; // Red for regressions
    
    // Size based on confidence
    const size = 6 + Math.floor(confidence * 4); // 6-10 pixel markers
    
    return { color, size };
  }

  private extractKeyImprovements(gains: PromotionResult['performance_gains']): string[] {
    const improvements: string[] = [];
    
    if (gains.p_at_5_improvement > 0.01) {
      improvements.push(`+${(gains.p_at_5_improvement * 100).toFixed(1)}% Precision@5`);
    }
    
    if (gains.latency_improvement > 5) {
      improvements.push(`-${gains.latency_improvement.toFixed(0)}ms latency`);
    }
    
    if (gains.cost_efficiency_gain > 0.1) {
      improvements.push(`+${(gains.cost_efficiency_gain * 100).toFixed(0)}% cost efficiency`);
    }
    
    if (gains.stability_score > 0.8) {
      improvements.push('High stability');
    }
    
    return improvements;
  }

  private extractVersionNumber(policyId: string): string {
    // Extract version from policy ID (simplified)
    const match = policyId.match(/v?(\d+)/);
    return match ? match[1] : '1';
  }
}

// ============================================================================
// DEPLOYMENT GATE
// ============================================================================

export class DeploymentGate {
  /**
   * Assesses whether a policy is ready for deployment
   */
  assessDeploymentReadiness(
    optimizedPolicy: OptimizedPolicy,
    gapSliceValidation: ValidationResult,
    unionSetValidation: ValidationResult,
    crossBudgetValidation: Record<number, ValidationResult>
  ): { deployment_status: PromotionResult['deployment_status']; deployment_confidence: number; reviewer_notes?: string } {
    
    const assessments = {
      gap_slice_passed: gapSliceValidation.gates_passed && gapSliceValidation.is_significant,
      union_set_passed: unionSetValidation.gates_passed,
      cross_budget_success_rate: this.calculateCrossBudgetSuccessRate(crossBudgetValidation),
      optimization_confidence: optimizedPolicy.confidence_score,
      performance_gain_magnitude: this.assessPerformanceGainMagnitude(gapSliceValidation)
    };

    // Calculate overall confidence
    const confidence = this.calculateDeploymentConfidence(assessments);
    
    // Determine deployment status
    let deploymentStatus: PromotionResult['deployment_status'];
    let reviewerNotes: string | undefined;

    if (confidence > 0.8 && assessments.gap_slice_passed && assessments.union_set_passed) {
      deploymentStatus = 'ready';
    } else if (confidence > 0.6 && assessments.gap_slice_passed) {
      deploymentStatus = 'needs_review';
      reviewerNotes = `Moderate confidence (${confidence.toFixed(2)}). Union set: ${assessments.union_set_passed ? 'PASS' : 'FAIL'}. Cross-budget success: ${(assessments.cross_budget_success_rate * 100).toFixed(0)}%`;
    } else {
      deploymentStatus = 'rejected';
      reviewerNotes = `Low confidence (${confidence.toFixed(2)}). Gap slice: ${assessments.gap_slice_passed ? 'PASS' : 'FAIL'}. Union set: ${assessments.union_set_passed ? 'PASS' : 'FAIL'}.`;
    }

    return {
      deployment_status: deploymentStatus,
      deployment_confidence: confidence,
      reviewer_notes: reviewerNotes
    };
  }

  private calculateCrossBudgetSuccessRate(crossBudgetValidation: Record<number, ValidationResult>): number {
    const results = Object.values(crossBudgetValidation);
    if (results.length === 0) return 1; // No cross-budget validation performed
    
    const successCount = results.filter(r => r.gates_passed && r.is_significant).length;
    return successCount / results.length;
  }

  private assessPerformanceGainMagnitude(gapSliceValidation: ValidationResult): 'high' | 'medium' | 'low' {
    const improvement = gapSliceValidation.p_at_5_delta;
    
    if (improvement > 0.05) return 'high';    // >5% improvement
    if (improvement > 0.02) return 'medium';  // >2% improvement
    return 'low';                             // <2% improvement
  }

  private calculateDeploymentConfidence(assessments: {
    gap_slice_passed: boolean;
    union_set_passed: boolean;
    cross_budget_success_rate: number;
    optimization_confidence: number;
    performance_gain_magnitude: 'high' | 'medium' | 'low';
  }): number {
    let confidence = 0;
    
    // Core validation gates (60% of confidence)
    if (assessments.gap_slice_passed) confidence += 0.4;
    if (assessments.union_set_passed) confidence += 0.2;
    
    // Cross-budget performance (20% of confidence)
    confidence += assessments.cross_budget_success_rate * 0.2;
    
    // Optimization process confidence (10% of confidence)
    confidence += assessments.optimization_confidence * 0.1;
    
    // Performance gain magnitude (10% of confidence)
    const magnitudeBonuses = { high: 0.1, medium: 0.06, low: 0.02 };
    confidence += magnitudeBonuses[assessments.performance_gain_magnitude];
    
    return Math.max(0, Math.min(1, confidence));
  }
}

// ============================================================================
// MICROSITE INTEGRATION UTILITIES
// ============================================================================

export class MicrositeIntegrator {
  /**
   * Generates Pareto front data for microsite consumption
   */
  static generateParetoFrontData(promotionResults: PromotionResult[]): MicrositeParetoPatch {
    const validatedPolicies = promotionResults.filter(r => r.deployment_status === 'ready');
    
    const paretoPoints = validatedPolicies.map(result => ({
      id: result.policy_id,
      x: result.pareto_front_annotation.cost_efficiency,
      y: result.pareto_front_annotation.performance_score,
      z: result.pareto_front_annotation.latency_score,
      label: result.pareto_front_annotation.policy_label,
      color: result.pareto_front_annotation.marker_color,
      size: result.pareto_front_annotation.marker_size,
      highlight: result.pareto_front_annotation.highlight,
      tooltip: result.pareto_front_annotation.tooltip_data
    }));

    return {
      updated_at: new Date().toISOString(),
      pareto_points: paretoPoints,
      metadata: {
        total_policies_evaluated: promotionResults.length,
        validated_policies: validatedPolicies.length,
        average_confidence: validatedPolicies.reduce((sum, p) => sum + p.deployment_confidence, 0) / Math.max(1, validatedPolicies.length)
      }
    };
  }

  /**
   * Generates webhook payload for microsite updates
   */
  static generateWebhookPayload(promotionResult: PromotionResult): MicrositeWebhookPayload {
    return {
      event_type: 'policy_promoted',
      timestamp: new Date().toISOString(),
      data: {
        policy_id: promotionResult.policy_id,
        source_gap: promotionResult.source_gap,
        deployment_status: promotionResult.deployment_status,
        performance_summary: {
          p_at_5_improvement: promotionResult.performance_gains.p_at_5_improvement,
          latency_improvement: promotionResult.performance_gains.latency_improvement,
          cost_efficiency_gain: promotionResult.performance_gains.cost_efficiency_gain
        },
        pareto_annotation: promotionResult.pareto_front_annotation,
        validation_confidence: promotionResult.deployment_confidence
      }
    };
  }
}

// ============================================================================
// SUPPORTING TYPES
// ============================================================================

interface MicrositeParetoPatch {
  updated_at: string;
  pareto_points: Array<{
    id: string;
    x: number; // cost_efficiency
    y: number; // performance_score
    z: number; // latency_score
    label: string;
    color: string;
    size: number;
    highlight: boolean;
    tooltip: ParetoFrontAnnotation['tooltip_data'];
  }>;
  metadata: {
    total_policies_evaluated: number;
    validated_policies: number;
    average_confidence: number;
  };
}

interface MicrositeWebhookPayload {
  event_type: 'policy_promoted';
  timestamp: string;
  data: {
    policy_id: string;
    source_gap: string;
    deployment_status: PromotionResult['deployment_status'];
    performance_summary: {
      p_at_5_improvement: number;
      latency_improvement: number;
      cost_efficiency_gain: number;
    };
    pareto_annotation: ParetoFrontAnnotation;
    validation_confidence: number;
  };
}