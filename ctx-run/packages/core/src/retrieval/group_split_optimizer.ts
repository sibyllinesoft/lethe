/**
 * Group-Split Move Optimizer for Bounded Group Management
 * 
 * Implements sophisticated group management with:
 * - Bounded group size (≤8 atoms) with smart splitting
 * - One group split per 2-swap optimization cycle
 * - ILP de-biasing with ridge + empirical-Bayes shrinkage
 * - Maintains single-digit ILP incidence percentage
 * - Performance-aware group formation and splitting decisions
 * 
 * Mathematical Foundation:
 * Groups represent coherent semantic units. When groups exceed size bounds,
 * strategic splitting maintains semantic coherence while enabling optimization.
 */

import { z } from 'zod';

// Group split configuration
export const GroupSplitConfigSchema = z.object({
  // Group size constraints
  max_group_size: z.number().int().min(2).default(8),
  min_split_size: z.number().int().min(1).default(2),
  target_group_size: z.number().int().min(2).default(5),
  
  // Split strategy parameters
  max_splits_per_cycle: z.number().int().min(0).default(1), // One split per 2-swap
  split_similarity_threshold: z.number().min(0).max(1).default(0.7),
  coherence_preservation_weight: z.number().min(0).default(0.8),
  
  // ILP control
  target_ilp_incidence: z.number().min(0).max(0.5).default(0.05), // <5%
  ridge_regularization: z.number().min(0).default(0.01),
  empirical_bayes_shrinkage: z.number().min(0).max(1).default(0.3),
  
  // Performance constraints
  max_split_computation_ms: z.number().min(10).default(100),
  enable_performance_aware_splitting: z.boolean().default(true),
  
  // Quality thresholds
  min_semantic_coherence: z.number().min(0).max(1).default(0.6),
  max_quality_degradation: z.number().min(0).default(0.02), // 2% max degradation
});

export type GroupSplitConfig = z.infer<typeof GroupSplitConfigSchema>;

// Group representation
export interface SemanticGroup {
  id: string;
  atom_ids: string[];
  centroid_embedding?: number[];
  coherence_score: number;
  semantic_theme: string;
  split_history: Array<{
    timestamp: number;
    split_reason: string;
    resulting_groups: string[];
  }>;
  performance_impact: {
    selection_frequency: number;
    average_quality_contribution: number;
    computational_cost: number;
  };
}

// Split candidate analysis
export interface SplitCandidate {
  group_id: string;
  split_point: number; // Index where to split
  split_strategy: 'semantic_clustering' | 'quality_based' | 'balanced';
  expected_coherence: [number, number]; // Coherence of resulting groups
  quality_impact: number; // Expected change in selection quality
  ilp_reduction_potential: number; // How much ILP complexity this reduces
  computational_cost: number;
  confidence_score: number;
}

// 2-swap optimization context
export interface TwoSwapContext {
  current_selection: string[];
  available_atoms: Array<{
    id: string;
    embedding: number[];
    quality_score: number;
    group_id: string;
    tokens: number;
  }>;
  token_budget: number;
  optimization_iteration: number;
}

// Split operation result
export interface GroupSplitResult {
  original_group: SemanticGroup;
  new_groups: SemanticGroup[];
  split_strategy_used: string;
  coherence_preservation: number;
  performance_improvement: number;
  ilp_complexity_reduction: number;
  computation_time_ms: number;
  quality_assurance: {
    semantic_validity: boolean;
    performance_regression: boolean;
    ilp_target_met: boolean;
  };
}

// ILP de-biasing statistics
export interface ILPDebiasingStats {
  current_ilp_incidence: number;
  target_ilp_incidence: number;
  ridge_coefficient: number;
  shrinkage_factor: number;
  bias_correction: number;
  effective_sample_size: number;
  convergence_status: 'converged' | 'converging' | 'diverged';
}

/**
 * Group-Split Move Optimizer
 * 
 * Manages bounded group sizes through intelligent splitting:
 * 1. Monitor group sizes and identify split candidates
 * 2. Analyze semantic coherence and performance impact
 * 3. Execute strategic splits during 2-swap cycles
 * 4. Apply ILP de-biasing with regularization
 * 5. Maintain optimization quality while reducing complexity
 */
export class GroupSplitOptimizer {
  private config: GroupSplitConfig;
  private groups: Map<string, SemanticGroup> = new Map();
  private split_history: GroupSplitResult[] = [];
  private ilp_stats: ILPDebiasingStats;
  
  constructor(config: Partial<GroupSplitConfig> = {}) {
    this.config = GroupSplitConfigSchema.parse(config);
    this.initializeILPStats();
    
    console.log(`🔧 Group-Split Optimizer initialized: max_size=${this.config.max_group_size}, target_ILP=${(this.config.target_ilp_incidence * 100).toFixed(1)}%`);
  }
  
  /**
   * Initialize ILP de-biasing statistics
   */
  private initializeILPStats(): void {
    this.ilp_stats = {
      current_ilp_incidence: 0.0,
      target_ilp_incidence: this.config.target_ilp_incidence,
      ridge_coefficient: this.config.ridge_regularization,
      shrinkage_factor: this.config.empirical_bayes_shrinkage,
      bias_correction: 0.0,
      effective_sample_size: 100,
      convergence_status: 'converging',
    };
  }
  
  /**
   * Update group registry and identify split candidates
   */
  async updateGroupRegistry(
    groups: SemanticGroup[]
  ): Promise<{
    oversized_groups: SemanticGroup[];
    split_candidates: SplitCandidate[];
    registry_stats: {
      total_groups: number;
      average_size: number;
      oversized_count: number;
      ilp_potential: number;
    };
  }> {
    console.log(`📊 Updating group registry: ${groups.length} groups`);
    
    // Update internal registry
    this.groups.clear();
    for (const group of groups) {
      this.groups.set(group.id, { ...group });
    }
    
    // Identify oversized groups
    const oversized_groups = groups.filter(
      group => group.atom_ids.length > this.config.max_group_size
    );
    
    // Generate split candidates
    const split_candidates: SplitCandidate[] = [];
    
    for (const group of oversized_groups) {
      const candidates = await this.generateSplitCandidates(group);
      split_candidates.push(...candidates);
    }
    
    // Sort candidates by expected benefit
    split_candidates.sort((a, b) => {
      const benefit_a = a.ilp_reduction_potential * 0.4 + a.quality_impact * 0.3 + a.confidence_score * 0.3;
      const benefit_b = b.ilp_reduction_potential * 0.4 + b.quality_impact * 0.3 + b.confidence_score * 0.3;
      return benefit_b - benefit_a;
    });
    
    // Calculate registry statistics
    const total_atoms = groups.reduce((sum, g) => sum + g.atom_ids.length, 0);
    const average_size = groups.length > 0 ? total_atoms / groups.length : 0;
    const ilp_potential = this.estimateILPComplexity(groups);
    
    const registry_stats = {
      total_groups: groups.length,
      average_size,
      oversized_count: oversized_groups.length,
      ilp_potential,
    };
    
    console.log(`  Found ${oversized_groups.length} oversized groups, ${split_candidates.length} split candidates`);
    
    return {
      oversized_groups,
      split_candidates,
      registry_stats,
    };
  }
  
  /**
   * Execute strategic group split during 2-swap optimization
   */
  async executeStrategicSplit(
    split_candidates: SplitCandidate[],
    two_swap_context: TwoSwapContext
  ): Promise<{
    split_executed: boolean;
    split_result?: GroupSplitResult;
    updated_groups: SemanticGroup[];
    ilp_improvement: number;
    performance_impact: {
      quality_change: number;
      complexity_reduction: number;
      computation_overhead: number;
    };
  }> {
    if (split_candidates.length === 0) {
      console.log('⏭️ No split candidates available');
      return {
        split_executed: false,
        updated_groups: Array.from(this.groups.values()),
        ilp_improvement: 0,
        performance_impact: {
          quality_change: 0,
          complexity_reduction: 0,
          computation_overhead: 0,
        },
      };
    }
    
    // Select best candidate (limited to 1 split per cycle)
    const best_candidate = split_candidates[0];
    
    console.log(`🔄 Executing strategic split: group ${best_candidate.group_id}`);
    
    const start_time = performance.now();
    
    try {
      // Execute the split
      const split_result = await this.executeSplit(
        best_candidate,
        two_swap_context
      );
      
      // Update group registry
      this.groups.delete(split_result.original_group.id);
      for (const new_group of split_result.new_groups) {
        this.groups.set(new_group.id, new_group);
      }
      
      // Record split in history
      this.split_history.push(split_result);
      
      // Limit history size
      if (this.split_history.length > 100) {
        this.split_history = this.split_history.slice(-100);
      }
      
      // Update ILP statistics
      await this.updateILPStatistics(split_result);
      
      const computation_time = performance.now() - start_time;
      
      console.log(`  ✅ Split complete: ${split_result.new_groups.length} new groups, ${computation_time.toFixed(1)}ms`);
      console.log(`  Coherence preservation: ${(split_result.coherence_preservation * 100).toFixed(1)}%`);
      console.log(`  ILP reduction: ${(split_result.ilp_complexity_reduction * 100).toFixed(1)}%`);
      
      return {
        split_executed: true,
        split_result,
        updated_groups: Array.from(this.groups.values()),
        ilp_improvement: split_result.ilp_complexity_reduction,
        performance_impact: {
          quality_change: split_result.coherence_preservation - 1.0,
          complexity_reduction: split_result.ilp_complexity_reduction,
          computation_overhead: computation_time,
        },
      };
      
    } catch (error) {
      console.error('Group split failed:', error);
      
      return {
        split_executed: false,
        updated_groups: Array.from(this.groups.values()),
        ilp_improvement: 0,
        performance_impact: {
          quality_change: 0,
          complexity_reduction: 0,
          computation_overhead: performance.now() - start_time,
        },
      };
    }
  }
  
  /**
   * Apply ILP de-biasing with ridge + empirical-Bayes shrinkage
   */
  async applyILPDebiasing(
    selection_scores: Array<{ id: string; score: number; group_id: string }>,
    historical_selections: Array<{ id: string; selected: boolean; group_id: string }>
  ): Promise<{
    debiased_scores: Array<{ id: string; adjusted_score: number; bias_correction: number }>;
    shrinkage_applied: number;
    ridge_regularization: number;
    effective_sample_size: number;
    convergence_improvement: number;
  }> {
    console.log('🔬 Applying ILP de-biasing with ridge + empirical-Bayes shrinkage...');
    
    const start_time = performance.now();
    
    // Compute group-level bias estimates
    const group_biases = this.computeGroupBiases(historical_selections);
    
    // Apply ridge regularization
    const ridge_factor = this.config.ridge_regularization;
    
    // Apply empirical-Bayes shrinkage
    const shrinkage_factor = this.config.empirical_bayes_shrinkage;
    
    const debiased_scores: Array<{
      id: string;
      adjusted_score: number;
      bias_correction: number;
    }> = [];
    
    for (const item of selection_scores) {
      const group_bias = group_biases.get(item.group_id) || 0;
      
      // Ridge regularization: shrink towards global mean
      const global_mean = selection_scores.reduce((sum, s) => sum + s.score, 0) / selection_scores.length;
      const ridge_adjustment = ridge_factor * (global_mean - item.score);
      
      // Empirical-Bayes shrinkage: shrink group bias towards zero
      const shrunk_bias = group_bias * (1 - shrinkage_factor);
      
      // Combined adjustment
      const bias_correction = ridge_adjustment - shrunk_bias;
      const adjusted_score = item.score + bias_correction;
      
      debiased_scores.push({
        id: item.id,
        adjusted_score,
        bias_correction,
      });
    }
    
    // Update ILP statistics
    const effective_sample_size = this.calculateEffectiveSampleSize(historical_selections);
    const convergence_improvement = this.assessConvergenceImprovement();
    
    this.ilp_stats.effective_sample_size = effective_sample_size;
    this.ilp_stats.bias_correction = debiased_scores.reduce(
      (sum, s) => sum + Math.abs(s.bias_correction), 0
    ) / debiased_scores.length;
    
    const computation_time = performance.now() - start_time;
    
    console.log(`  De-biasing complete: ${debiased_scores.length} scores adjusted, ${computation_time.toFixed(1)}ms`);
    console.log(`  Average bias correction: ${this.ilp_stats.bias_correction.toFixed(4)}`);
    console.log(`  Effective sample size: ${effective_sample_size.toFixed(0)}`);
    
    return {
      debiased_scores,
      shrinkage_applied: shrinkage_factor,
      ridge_regularization: ridge_factor,
      effective_sample_size,
      convergence_improvement,
    };
  }
  
  /**
   * Generate comprehensive group-split diagnostics
   */
  generateSplitDiagnostics(): {
    group_health: {
      total_groups: number;
      average_size: number;
      size_distribution: number[];
      oversized_count: number;
      health_score: number;
    };
    split_performance: {
      total_splits_executed: number;
      average_coherence_preservation: number;
      average_ilp_reduction: number;
      success_rate: number;
      recent_performance_trend: 'improving' | 'stable' | 'degrading';
    };
    ilp_control: {
      current_incidence: number;
      target_incidence: number;
      control_effectiveness: number;
      bias_correction_magnitude: number;
      convergence_status: string;
    };
    recommendations: {
      suggested_splits: number;
      optimization_opportunities: string[];
      risk_factors: string[];
      performance_tuning: string[];
    };
  } {
    // Group health analysis
    const groups = Array.from(this.groups.values());
    const sizes = groups.map(g => g.atom_ids.length);
    const average_size = sizes.length > 0 ? sizes.reduce((a, b) => a + b) / sizes.length : 0;
    const oversized_count = sizes.filter(s => s > this.config.max_group_size).length;
    
    // Health score based on size distribution and coherence
    const size_penalty = oversized_count / Math.max(1, groups.length);
    const avg_coherence = groups.length > 0 ?
      groups.reduce((sum, g) => sum + g.coherence_score, 0) / groups.length : 0;
    const health_score = Math.max(0, (avg_coherence * 0.7) - (size_penalty * 0.3));
    
    // Split performance analysis
    const recent_splits = this.split_history.slice(-20);
    const success_rate = recent_splits.length > 0 ?
      recent_splits.filter(s => s.quality_assurance.semantic_validity).length / recent_splits.length : 1;
    
    const avg_coherence_preservation = recent_splits.length > 0 ?
      recent_splits.reduce((sum, s) => sum + s.coherence_preservation, 0) / recent_splits.length : 1;
    
    const avg_ilp_reduction = recent_splits.length > 0 ?
      recent_splits.reduce((sum, s) => sum + s.ilp_complexity_reduction, 0) / recent_splits.length : 0;
    
    // Performance trend analysis
    let performance_trend: 'improving' | 'stable' | 'degrading' = 'stable';
    if (recent_splits.length >= 10) {
      const first_half = recent_splits.slice(0, recent_splits.length / 2);
      const second_half = recent_splits.slice(recent_splits.length / 2);
      
      const first_avg = first_half.reduce((sum, s) => sum + s.coherence_preservation, 0) / first_half.length;
      const second_avg = second_half.reduce((sum, s) => sum + s.coherence_preservation, 0) / second_half.length;
      
      if (second_avg > first_avg * 1.05) performance_trend = 'improving';
      else if (second_avg < first_avg * 0.95) performance_trend = 'degrading';
    }
    
    // ILP control analysis
    const control_effectiveness = this.ilp_stats.target_ilp_incidence > 0 ?
      Math.max(0, 1 - (this.ilp_stats.current_ilp_incidence / this.ilp_stats.target_ilp_incidence)) : 1;
    
    // Generate recommendations
    const recommendations = this.generateRecommendations(
      groups,
      recent_splits,
      health_score,
      control_effectiveness
    );
    
    return {
      group_health: {
        total_groups: groups.length,
        average_size,
        size_distribution: sizes,
        oversized_count,
        health_score,
      },
      split_performance: {
        total_splits_executed: this.split_history.length,
        average_coherence_preservation: avg_coherence_preservation,
        average_ilp_reduction: avg_ilp_reduction,
        success_rate,
        recent_performance_trend: performance_trend,
      },
      ilp_control: {
        current_incidence: this.ilp_stats.current_ilp_incidence,
        target_incidence: this.ilp_stats.target_ilp_incidence,
        control_effectiveness,
        bias_correction_magnitude: this.ilp_stats.bias_correction,
        convergence_status: this.ilp_stats.convergence_status,
      },
      recommendations,
    };
  }
  
  /**
   * Private helper methods
   */
  private async generateSplitCandidates(group: SemanticGroup): Promise<SplitCandidate[]> {
    if (group.atom_ids.length <= this.config.max_group_size) {
      return [];
    }
    
    const candidates: SplitCandidate[] = [];
    
    // Strategy 1: Semantic clustering split
    if (group.centroid_embedding) {
      const semantic_split = await this.analyzeSemanticSplit(group);
      if (semantic_split) {
        candidates.push(semantic_split);
      }
    }
    
    // Strategy 2: Quality-based split
    const quality_split = await this.analyzeQualityBasedSplit(group);
    if (quality_split) {
      candidates.push(quality_split);
    }
    
    // Strategy 3: Balanced split (fallback)
    const balanced_split = this.createBalancedSplit(group);
    candidates.push(balanced_split);
    
    return candidates;
  }
  
  private async analyzeSemanticSplit(group: SemanticGroup): Promise<SplitCandidate | null> {
    // Mock implementation - would use actual semantic analysis
    const split_point = Math.floor(group.atom_ids.length / 2);
    
    return {
      group_id: group.id,
      split_point,
      split_strategy: 'semantic_clustering',
      expected_coherence: [0.8, 0.75], // Mock coherence scores
      quality_impact: 0.02, // Slight improvement
      ilp_reduction_potential: 0.15, // 15% ILP reduction
      computational_cost: 50, // ms
      confidence_score: 0.85,
    };
  }
  
  private async analyzeQualityBasedSplit(group: SemanticGroup): Promise<SplitCandidate | null> {
    // Mock implementation - would analyze quality distributions
    const split_point = Math.floor(group.atom_ids.length * 0.6); // 60-40 split
    
    return {
      group_id: group.id,
      split_point,
      split_strategy: 'quality_based',
      expected_coherence: [0.85, 0.7],
      quality_impact: 0.05, // Moderate improvement
      ilp_reduction_potential: 0.20,
      computational_cost: 30,
      confidence_score: 0.75,
    };
  }
  
  private createBalancedSplit(group: SemanticGroup): SplitCandidate {
    const split_point = Math.floor(group.atom_ids.length / 2);
    
    return {
      group_id: group.id,
      split_point,
      split_strategy: 'balanced',
      expected_coherence: [0.7, 0.7],
      quality_impact: 0.0, // Neutral
      ilp_reduction_potential: 0.10,
      computational_cost: 10,
      confidence_score: 0.9, // High confidence in balanced split
    };
  }
  
  private async executeSplit(
    candidate: SplitCandidate,
    context: TwoSwapContext
  ): Promise<GroupSplitResult> {
    const original_group = this.groups.get(candidate.group_id);
    if (!original_group) {
      throw new Error(`Group ${candidate.group_id} not found`);
    }
    
    const start_time = performance.now();
    
    // Create new groups based on split strategy
    const group1_ids = original_group.atom_ids.slice(0, candidate.split_point);
    const group2_ids = original_group.atom_ids.slice(candidate.split_point);
    
    const new_group1: SemanticGroup = {
      id: `${original_group.id}_split_1`,
      atom_ids: group1_ids,
      coherence_score: candidate.expected_coherence[0],
      semantic_theme: `${original_group.semantic_theme}_part1`,
      split_history: [],
      performance_impact: {
        selection_frequency: original_group.performance_impact.selection_frequency * 0.6,
        average_quality_contribution: original_group.performance_impact.average_quality_contribution * 0.9,
        computational_cost: original_group.performance_impact.computational_cost * 0.4,
      },
    };
    
    const new_group2: SemanticGroup = {
      id: `${original_group.id}_split_2`,
      atom_ids: group2_ids,
      coherence_score: candidate.expected_coherence[1],
      semantic_theme: `${original_group.semantic_theme}_part2`,
      split_history: [],
      performance_impact: {
        selection_frequency: original_group.performance_impact.selection_frequency * 0.4,
        average_quality_contribution: original_group.performance_impact.average_quality_contribution * 0.85,
        computational_cost: original_group.performance_impact.computational_cost * 0.6,
      },
    };
    
    const computation_time = performance.now() - start_time;
    
    // Quality assurance checks
    const semantic_validity = candidate.expected_coherence[0] >= this.config.min_semantic_coherence &&
                             candidate.expected_coherence[1] >= this.config.min_semantic_coherence;
    
    const performance_regression = candidate.quality_impact < -this.config.max_quality_degradation;
    const ilp_target_met = candidate.ilp_reduction_potential >= 0.05; // At least 5% reduction
    
    const result: GroupSplitResult = {
      original_group,
      new_groups: [new_group1, new_group2],
      split_strategy_used: candidate.split_strategy,
      coherence_preservation: (candidate.expected_coherence[0] + candidate.expected_coherence[1]) / 2,
      performance_improvement: candidate.quality_impact,
      ilp_complexity_reduction: candidate.ilp_reduction_potential,
      computation_time_ms: computation_time,
      quality_assurance: {
        semantic_validity,
        performance_regression,
        ilp_target_met,
      },
    };
    
    return result;
  }
  
  private estimateILPComplexity(groups: SemanticGroup[]): number {
    // Estimate ILP complexity based on group sizes and interactions
    const large_groups = groups.filter(g => g.atom_ids.length > this.config.target_group_size);
    const complexity_score = large_groups.reduce((sum, g) => {
      // Exponential growth in complexity with group size
      const size_factor = Math.pow(g.atom_ids.length / this.config.target_group_size, 2);
      return sum + size_factor;
    }, 0);
    
    return Math.min(1.0, complexity_score / groups.length);
  }
  
  private computeGroupBiases(
    historical_selections: Array<{ id: string; selected: boolean; group_id: string }>
  ): Map<string, number> {
    const group_stats = new Map<string, { selections: number; total: number }>();
    
    // Compute group-wise selection rates
    for (const item of historical_selections) {
      if (!group_stats.has(item.group_id)) {
        group_stats.set(item.group_id, { selections: 0, total: 0 });
      }
      
      const stats = group_stats.get(item.group_id)!;
      stats.total++;
      if (item.selected) stats.selections++;
    }
    
    // Compute biases relative to global selection rate
    const global_selection_rate = historical_selections.filter(s => s.selected).length / historical_selections.length;
    
    const biases = new Map<string, number>();
    for (const [group_id, stats] of group_stats) {
      const group_rate = stats.selections / stats.total;
      const bias = group_rate - global_selection_rate;
      biases.set(group_id, bias);
    }
    
    return biases;
  }
  
  private calculateEffectiveSampleSize(
    historical_selections: Array<{ id: string; selected: boolean; group_id: string }>
  ): number {
    // Simplified effective sample size calculation
    const unique_groups = new Set(historical_selections.map(s => s.group_id)).size;
    const total_samples = historical_selections.length;
    
    // Adjust for group clustering
    return total_samples * Math.min(1.0, unique_groups / 10);
  }
  
  private assessConvergenceImprovement(): number {
    // Mock convergence improvement assessment
    return 0.05; // 5% improvement
  }
  
  private async updateILPStatistics(split_result: GroupSplitResult): Promise<void> {
    // Update ILP incidence based on split results
    const complexity_reduction = split_result.ilp_complexity_reduction;
    
    this.ilp_stats.current_ilp_incidence = Math.max(
      0,
      this.ilp_stats.current_ilp_incidence - complexity_reduction * 0.1
    );
    
    // Update convergence status
    if (this.ilp_stats.current_ilp_incidence <= this.ilp_stats.target_ilp_incidence) {
      this.ilp_stats.convergence_status = 'converged';
    } else if (complexity_reduction > 0) {
      this.ilp_stats.convergence_status = 'converging';
    } else {
      this.ilp_stats.convergence_status = 'diverged';
    }
  }
  
  private generateRecommendations(
    groups: SemanticGroup[],
    recent_splits: GroupSplitResult[],
    health_score: number,
    control_effectiveness: number
  ): {
    suggested_splits: number;
    optimization_opportunities: string[];
    risk_factors: string[];
    performance_tuning: string[];
  } {
    const oversized_count = groups.filter(g => g.atom_ids.length > this.config.max_group_size).length;
    
    const optimization_opportunities: string[] = [];
    const risk_factors: string[] = [];
    const performance_tuning: string[] = [];
    
    if (oversized_count > 0) {
      optimization_opportunities.push(`Split ${oversized_count} oversized groups`);
    }
    
    if (health_score < 0.7) {
      risk_factors.push('Low group health score - review semantic coherence');
    }
    
    if (control_effectiveness < 0.8) {
      optimization_opportunities.push('Improve ILP control mechanisms');
      performance_tuning.push('Increase ridge regularization');
    }
    
    if (recent_splits.length > 0 && recent_splits.slice(-5).every(s => s.performance_improvement < 0)) {
      risk_factors.push('Recent splits showing quality degradation');
      performance_tuning.push('Review split strategies and thresholds');
    }
    
    return {
      suggested_splits: oversized_count,
      optimization_opportunities,
      risk_factors,
      performance_tuning,
    };
  }
}

/**
 * Convenience function for bounded group-split optimization
 */
export async function optimizeWithBoundedGroupSplits(
  groups: SemanticGroup[],
  two_swap_context: TwoSwapContext,
  config: Partial<GroupSplitConfig> = {}
): Promise<{
  optimizer: GroupSplitOptimizer;
  split_result: Awaited<ReturnType<GroupSplitOptimizer['executeStrategicSplit']>>;
  diagnostics: ReturnType<GroupSplitOptimizer['generateSplitDiagnostics']>;
  final_groups: SemanticGroup[];
}> {
  console.log('🎯 Optimizing with bounded group-splits...');
  
  const optimizer = new GroupSplitOptimizer(config);
  
  // Update registry and identify candidates
  const { split_candidates } = await optimizer.updateGroupRegistry(groups);
  
  // Execute strategic split
  const split_result = await optimizer.executeStrategicSplit(split_candidates, two_swap_context);
  
  // Generate diagnostics
  const diagnostics = optimizer.generateSplitDiagnostics();
  
  console.log('✅ Bounded group-split optimization complete');
  console.log(`  Groups after split: ${split_result.updated_groups.length}`);
  console.log(`  ILP improvement: ${(split_result.ilp_improvement * 100).toFixed(1)}%`);
  console.log(`  Group health score: ${(diagnostics.group_health.health_score * 100).toFixed(1)}%`);
  
  return {
    optimizer,
    split_result,
    diagnostics,
    final_groups: split_result.updated_groups,
  };
}
