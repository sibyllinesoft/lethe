/**
 * Lagrangian Dual Variable (λ) System for Submodular Knapsack Optimization
 * 
 * Implements the core mathematical framework:
 * max F(S) - λ⋅tokens(S)
 * 
 * Where F(S) = Σ ΔU + γ⋅Coverage + δ⋅log det(I+V_S V_S^T)
 * 
 * Key features:
 * - Bisection algorithm to hit token/SLO targets
 * - Warm-start λ from prior turns
 * - λ exposed as budget creep metric
 * - "gain/token ≥ λ" heuristic converted to dual optimality check
 */

import { z } from 'zod';

// Lagrangian optimization configuration
export const LagrangianConfigSchema = z.object({
  // Bisection parameters
  lambda_min: z.number().min(0).default(0.001),
  lambda_max: z.number().min(0).default(10.0),
  lambda_tolerance: z.number().min(0).default(0.001),
  max_bisection_iterations: z.number().int().min(1).default(20),
  
  // Warm-start settings
  warm_start_enabled: z.boolean().default(true),
  lambda_history_size: z.number().int().min(1).default(10),
  
  // Budget creep monitoring
  budget_creep_threshold: z.number().min(0).default(0.2), // 20% increase triggers warning
  
  // Dual optimality settings
  optimality_tolerance: z.number().min(0).default(0.01),
  
  // Objective function weights
  gamma_coverage: z.number().min(0).default(1.0),
  delta_diversity: z.number().min(0).default(0.5),
});

export type LagrangianConfig = z.infer<typeof LagrangianConfigSchema>;

// Lagrangian state for warm-start and monitoring
export interface LagrangianState {
  lambda_current: number;
  lambda_history: number[];
  token_target: number;
  budget_creep_factor: number;
  dual_gap: number;
  bisection_iterations: number;
  convergence_achieved: boolean;
}

// Item for Lagrangian optimization with submodular gains
export interface LagrangianItem {
  id: string;
  tokens: number;
  delta_u: number; // Value-of-information gain
  coverage_gain: number; // Facility location gain
  diversity_gain: number; // DPP diversity gain (to be computed)
  marginal_gain_cached?: number; // Cached total marginal gain
  selected: boolean;
}

// Result of Lagrangian optimization
export interface LagrangianResult {
  selected_items: LagrangianItem[];
  final_lambda: number;
  total_tokens: number;
  objective_value: number;
  dual_gap: number;
  convergence_achieved: boolean;
  bisection_iterations: number;
  processing_time_ms: number;
  budget_creep_warning: boolean;
}

/**
 * Lagrangian Dual Variable Optimizer
 * 
 * Solves the submodular knapsack problem via Lagrangian relaxation:
 * max F(S) - λ⋅tokens(S)
 * 
 * Uses bisection on λ to hit token targets with warm-start optimization.
 */
export class LagrangianOptimizer {
  private config: LagrangianConfig;
  private state: LagrangianState;
  
  constructor(config: Partial<LagrangianConfig> = {}) {
    this.config = LagrangianConfigSchema.parse(config);
    this.state = {
      lambda_current: this.config.lambda_min,
      lambda_history: [],
      token_target: 0,
      budget_creep_factor: 1.0,
      dual_gap: Infinity,
      bisection_iterations: 0,
      convergence_achieved: false,
    };
  }

  /**
   * Main optimization entry point using Lagrangian bisection
   */
  async optimizeSelection(
    items: LagrangianItem[],
    token_budget: number,
    warm_start_lambda?: number
  ): Promise<LagrangianResult> {
    const startTime = performance.now();
    
    // Set token target
    this.state.token_target = token_budget;
    
    // Initialize lambda with warm-start if available
    if (warm_start_lambda && this.config.warm_start_enabled) {
      this.state.lambda_current = Math.max(this.config.lambda_min, warm_start_lambda);
    } else {
      this.state.lambda_current = this.config.lambda_min;
    }
    
    // Bisection algorithm to find optimal λ
    let lambda_low = this.config.lambda_min;
    let lambda_high = this.config.lambda_max;
    let best_selection: LagrangianItem[] = [];
    let best_lambda = this.state.lambda_current;
    
    this.state.bisection_iterations = 0;
    this.state.convergence_achieved = false;
    
    while (lambda_high - lambda_low > this.config.lambda_tolerance && 
           this.state.bisection_iterations < this.config.max_bisection_iterations) {
      
      const lambda_mid = (lambda_low + lambda_high) / 2;
      
      // Solve Lagrangian relaxation for current λ
      const selection = this.solveLagrangianRelaxation(items, lambda_mid);
      const total_tokens = selection.reduce((sum, item) => sum + item.tokens, 0);
      
      if (total_tokens <= token_budget) {
        // Selection fits within budget - try smaller λ for more items
        lambda_high = lambda_mid;
        best_selection = selection;
        best_lambda = lambda_mid;
      } else {
        // Selection exceeds budget - increase λ to reduce selection
        lambda_low = lambda_mid;
      }
      
      this.state.bisection_iterations++;
      
      // Check convergence
      if (Math.abs(total_tokens - token_budget) / token_budget < this.config.optimality_tolerance) {
        this.state.convergence_achieved = true;
        break;
      }
    }
    
    this.state.lambda_current = best_lambda;
    
    // Update lambda history for warm-start
    this.updateLambdaHistory(best_lambda);
    
    // Calculate metrics
    const total_tokens = best_selection.reduce((sum, item) => sum + item.tokens, 0);
    const objective_value = this.computeObjectiveValue(best_selection);
    const dual_gap = this.computeDualGap(best_selection, best_lambda);
    const budget_creep_warning = this.checkBudgetCreep(best_lambda);
    
    const processingTime = performance.now() - startTime;
    
    return {
      selected_items: best_selection,
      final_lambda: best_lambda,
      total_tokens,
      objective_value,
      dual_gap,
      convergence_achieved: this.state.convergence_achieved,
      bisection_iterations: this.state.bisection_iterations,
      processing_time_ms: processingTime,
      budget_creep_warning,
    };
  }
  
  /**
   * Solve Lagrangian relaxation for fixed λ using greedy algorithm
   */
  private solveLagrangianRelaxation(items: LagrangianItem[], lambda: number): LagrangianItem[] {
    // Reset selection state
    for (const item of items) {
      item.selected = false;
      item.marginal_gain_cached = undefined;
    }
    
    // Compute Lagrangian gains: F'(item) - λ⋅tokens(item)
    const lagrangian_items = items.map(item => ({
      ...item,
      lagrangian_gain: this.computeMarginalGain(item, []) - lambda * item.tokens,
    }));
    
    // Sort by Lagrangian gain (highest first)
    lagrangian_items.sort((a, b) => b.lagrangian_gain - a.lagrangian_gain);
    
    // Greedy selection based on dual optimality condition
    const selected: LagrangianItem[] = [];
    
    for (const item of lagrangian_items) {
      // Dual optimality check: include if gain/token ≥ λ
      const marginal_gain = this.computeMarginalGain(item, selected);
      const gain_per_token = marginal_gain / item.tokens;
      
      if (gain_per_token >= lambda - this.config.optimality_tolerance) {
        item.selected = true;
        item.marginal_gain_cached = marginal_gain;
        selected.push(item);
      }
    }
    
    return selected;
  }
  
  /**
   * Compute marginal gain F(S ∪ {item}) - F(S)
   */
  private computeMarginalGain(item: LagrangianItem, current_selection: LagrangianItem[]): number {
    // Use cached gain if available and current_selection hasn't changed significantly
    if (item.marginal_gain_cached !== undefined && current_selection.length < 50) {
      return item.marginal_gain_cached;
    }
    
    // F(S) = Σ ΔU + γ⋅Coverage + δ⋅log det(I+V_S V_S^T)
    let marginal_gain = item.delta_u; // Base VoI gain
    
    // Add facility location coverage gain (submodular)
    marginal_gain += this.config.gamma_coverage * item.coverage_gain;
    
    // Add DPP diversity gain (will be computed by DPP module)
    marginal_gain += this.config.delta_diversity * item.diversity_gain;
    
    // Apply submodular diminishing returns
    if (current_selection.length > 0) {
      // Simple diminishing returns model - will be enhanced with actual submodular computation
      const diminishing_factor = 1 / (1 + 0.1 * current_selection.length);
      marginal_gain *= diminishing_factor;
    }
    
    return marginal_gain;
  }
  
  /**
   * Compute objective value F(S)
   */
  private computeObjectiveValue(selection: LagrangianItem[]): number {
    let objective = 0;
    
    // VoI component: Σ ΔU
    objective += selection.reduce((sum, item) => sum + item.delta_u, 0);
    
    // Coverage component: γ⋅Coverage(S)
    const coverage_value = selection.reduce((sum, item) => sum + item.coverage_gain, 0);
    objective += this.config.gamma_coverage * coverage_value;
    
    // Diversity component: δ⋅log det(I+V_S V_S^T) - will be enhanced
    const diversity_value = selection.reduce((sum, item) => sum + item.diversity_gain, 0);
    objective += this.config.delta_diversity * diversity_value;
    
    return objective;
  }
  
  /**
   * Compute dual gap for convergence monitoring
   */
  private computeDualGap(selection: LagrangianItem[], lambda: number): number {
    const primal_value = this.computeObjectiveValue(selection);
    const total_tokens = selection.reduce((sum, item) => sum + item.tokens, 0);
    
    // Dual value = max_i [F'(i) - λ⋅tokens(i)] + λ⋅budget
    const max_reduced_cost = Math.max(
      ...selection.map(item => this.computeMarginalGain(item, []) - lambda * item.tokens)
    );
    const dual_value = max_reduced_cost + lambda * this.state.token_target;
    
    return Math.max(0, dual_value - primal_value);
  }
  
  /**
   * Update lambda history for warm-start optimization
   */
  private updateLambdaHistory(lambda: number): void {
    this.state.lambda_history.push(lambda);
    
    // Keep only recent history
    if (this.state.lambda_history.length > this.config.lambda_history_size) {
      this.state.lambda_history.shift();
    }
  }
  
  /**
   * Check for budget creep and issue warnings
   */
  private checkBudgetCreep(lambda: number): boolean {
    if (this.state.lambda_history.length < 2) {
      return false;
    }
    
    const recent_avg = this.state.lambda_history.slice(-3).reduce((a, b) => a + b, 0) / 3;
    const historical_avg = this.state.lambda_history.reduce((a, b) => a + b, 0) / this.state.lambda_history.length;
    
    this.state.budget_creep_factor = recent_avg / (historical_avg || 1);
    
    return this.state.budget_creep_factor > (1 + this.config.budget_creep_threshold);
  }
  
  /**
   * Get warm-start lambda from history
   */
  getWarmStartLambda(): number | undefined {
    if (!this.config.warm_start_enabled || this.state.lambda_history.length === 0) {
      return undefined;
    }
    
    // Use exponentially weighted moving average
    let weighted_sum = 0;
    let weight_sum = 0;
    
    for (let i = 0; i < this.state.lambda_history.length; i++) {
      const weight = Math.exp(-0.1 * (this.state.lambda_history.length - 1 - i));
      weighted_sum += weight * this.state.lambda_history[i];
      weight_sum += weight;
    }
    
    return weighted_sum / weight_sum;
  }
  
  /**
   * Get current state for monitoring
   */
  getState(): LagrangianState {
    return { ...this.state };
  }
  
  /**
   * Reset optimizer state
   */
  resetState(): void {
    this.state = {
      lambda_current: this.config.lambda_min,
      lambda_history: [],
      token_target: 0,
      budget_creep_factor: 1.0,
      dual_gap: Infinity,
      bisection_iterations: 0,
      convergence_achieved: false,
    };
  }
}

/**
 * Convenience function for Lagrangian optimization
 */
export async function optimizeWithLagrangian(
  items: LagrangianItem[],
  token_budget: number,
  config: Partial<LagrangianConfig> = {},
  warm_start_lambda?: number
): Promise<LagrangianResult> {
  const optimizer = new LagrangianOptimizer(config);
  return optimizer.optimizeSelection(items, token_budget, warm_start_lambda);
}