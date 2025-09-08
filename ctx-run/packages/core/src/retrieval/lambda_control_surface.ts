/**
 * Lambda Control Surface - First-Class λ Management System
 * 
 * Implements sophisticated λ as control surface with:
 * - Monotonic size(λ) and CBU(λ) tracking per domain
 * - Primal-dual gap computation: g = λ⋅tokens(S*) - Σ Δgain_λ(a) 
 * - λ-drift detection and accept-rate monitoring
 * - Real-time dual behavior verification
 * 
 * Mathematical Foundation:
 * The Lagrangian dual λ serves as the fundamental control parameter governing
 * the trade-off between token budget constraints and selection quality.
 */

import { z } from 'zod';

// λ control configuration
export const LambdaControlConfigSchema = z.object({
  // Dual variable bounds
  lambda_min: z.number().min(0.001).default(0.01),
  lambda_max: z.number().max(100).default(50.0),
  lambda_epsilon: z.number().min(1e-6).default(1e-4), // Convergence tolerance
  
  // Bisection parameters
  max_bisection_iterations: z.number().int().min(5).default(25),
  lambda_drift_threshold: z.number().min(0.001).default(0.05), // 5% drift alert
  
  // Monitoring configuration
  track_domain_curves: z.boolean().default(true),
  track_accept_rates: z.boolean().default(true),
  track_primal_dual_gap: z.boolean().default(true),
  
  // Performance bounds
  max_lambda_computation_ms: z.number().min(10).default(50),
  enable_warm_start: z.boolean().default(true),
});

export type LambdaControlConfig = z.infer<typeof LambdaControlConfigSchema>;

// Domain-specific λ tracking
export interface DomainLambdaMetrics {
  domain: string;
  lambda_values: number[];
  size_curve: number[]; // tokens(λ) - should be monotonically decreasing
  cbu_curve: number[];  // CBU(λ) - contextual business utility 
  accept_rates: number[];
  monotonicity_violations: number;
  last_updated: number;
}

// Primal-dual gap computation
export interface PrimalDualGap {
  primal_value: number;      // λ⋅tokens(S*)
  dual_bound: number;        // Σ Δgain_λ(a) 
  gap: number;               // |primal - dual|
  gap_ratio: number;         // gap / primal
  convergence_achieved: boolean;
  computation_time_ms: number;
}

// λ-drift detection result
export interface LambdaDriftAnalysis {
  current_lambda: number;
  previous_lambda?: number;
  drift_magnitude: number;
  drift_direction: 'increase' | 'decrease' | 'stable';
  drift_rate: number; // drift per iteration
  alert_triggered: boolean;
  moving_average_lambda: number;
}

// Complete λ control state
export interface LambdaControlState {
  current_lambda: number;
  domain_metrics: Map<string, DomainLambdaMetrics>;
  primal_dual_gap: PrimalDualGap;
  drift_analysis: LambdaDriftAnalysis;
  bisection_history: number[];
  convergence_trajectory: Array<{
    iteration: number;
    lambda: number;
    gap: number;
    tokens_selected: number;
    objective_value: number;
  }>;
  performance_stats: {
    total_bisection_iterations: number;
    average_convergence_time_ms: number;
    lambda_stability_score: number;
  };
}

/**
 * Lambda Control Surface - Advanced λ Management Engine
 * 
 * Core responsibilities:
 * 1. Maintain λ as first-class control surface
 * 2. Track monotonic size(λ) and CBU(λ) curves per domain
 * 3. Compute primal-dual gaps with convergence monitoring
 * 4. Detect λ-drift and acceptance rate anomalies
 * 5. Provide real-time λ diagnostics and alerts
 */
export class LambdaControlSurface {
  private config: LambdaControlConfig;
  private state: LambdaControlState;
  private warm_start_lambda?: number;
  
  constructor(config: Partial<LambdaControlConfig> = {}) {
    this.config = LambdaControlConfigSchema.parse(config);
    this.initializeState();
    
    console.log('🎛️ Lambda Control Surface initialized with advanced dual variable management');
  }
  
  /**
   * Initialize control surface state
   */
  private initializeState(): void {
    this.state = {
      current_lambda: this.config.lambda_min,
      domain_metrics: new Map(),
      primal_dual_gap: {
        primal_value: 0,
        dual_bound: 0,
        gap: Infinity,
        gap_ratio: Infinity,
        convergence_achieved: false,
        computation_time_ms: 0,
      },
      drift_analysis: {
        current_lambda: this.config.lambda_min,
        drift_magnitude: 0,
        drift_direction: 'stable',
        drift_rate: 0,
        alert_triggered: false,
        moving_average_lambda: this.config.lambda_min,
      },
      bisection_history: [],
      convergence_trajectory: [],
      performance_stats: {
        total_bisection_iterations: 0,
        average_convergence_time_ms: 0,
        lambda_stability_score: 1.0,
      },
    };
  }
  
  /**
   * Execute sophisticated λ bisection with domain tracking
   */
  async executeLambdaBisection(
    objective_function: (lambda: number) => Promise<{
      selected_items: Array<{ id: string; tokens: number; delta_gain: number; domain: string }>;
      total_tokens: number;
      objective_value: number;
    }>,
    target_tokens: number,
    domain_context?: string
  ): Promise<{
    optimal_lambda: number;
    convergence_result: {
      converged: boolean;
      iterations: number;
      final_gap: number;
      computation_time_ms: number;
    };
    domain_analysis: DomainLambdaMetrics[];
  }> {
    const start_time = performance.now();
    
    console.log(`🎯 Executing λ bisection: target=${target_tokens} tokens, domain=${domain_context || 'all'}`);
    
    let lambda_low = this.config.lambda_min;
    let lambda_high = this.config.lambda_max;
    let current_lambda = this.warm_start_lambda || (lambda_low + lambda_high) / 2;
    
    let iteration = 0;
    let converged = false;
    let best_result: any = null;
    
    // Track convergence trajectory
    const trajectory: Array<{
      iteration: number;
      lambda: number;
      gap: number;
      tokens_selected: number;
      objective_value: number;
    }> = [];
    
    while (iteration < this.config.max_bisection_iterations && !converged) {
      iteration++;
      
      try {
        // Execute objective function at current λ
        const result = await objective_function(current_lambda);
        
        // Update domain metrics
        this.updateDomainMetrics(current_lambda, result, domain_context);
        
        // Compute primal-dual gap
        const gap_analysis = this.computePrimalDualGap(
          current_lambda,
          result.selected_items,
          result.total_tokens
        );
        
        // Track convergence trajectory
        trajectory.push({
          iteration,
          lambda: current_lambda,
          gap: gap_analysis.gap,
          tokens_selected: result.total_tokens,
          objective_value: result.objective_value,
        });
        
        console.log(`  Iter ${iteration}: λ=${current_lambda.toFixed(4)}, tokens=${result.total_tokens}, gap=${gap_analysis.gap.toFixed(3)}`);
        
        // Check convergence criteria
        const token_error = Math.abs(result.total_tokens - target_tokens);
        const relative_token_error = token_error / target_tokens;
        
        if (relative_token_error < this.config.lambda_epsilon) {
          converged = true;
          best_result = result;
          break;
        }
        
        // Update bisection bounds
        if (result.total_tokens > target_tokens) {
          // Too many tokens selected, increase λ (penalize more)
          lambda_low = current_lambda;
        } else {
          // Too few tokens selected, decrease λ (penalize less)
          lambda_high = current_lambda;
        }
        
        // Update λ for next iteration
        const previous_lambda = current_lambda;
        current_lambda = (lambda_low + lambda_high) / 2;
        
        // Check for λ-drift
        this.updateDriftAnalysis(current_lambda, previous_lambda);
        
        best_result = result;
        
      } catch (error) {
        console.warn(`λ bisection failed at iteration ${iteration}:`, error);
        break;
      }
    }
    
    const computation_time = performance.now() - start_time;
    
    // Update state
    this.state.current_lambda = current_lambda;
    this.state.convergence_trajectory = trajectory;
    this.state.performance_stats.total_bisection_iterations += iteration;
    
    // Update warm start for next optimization
    if (this.config.enable_warm_start && converged) {
      this.warm_start_lambda = current_lambda;
    }
    
    console.log(`🎯 λ bisection complete: λ=${current_lambda.toFixed(4)}, converged=${converged}, ${iteration} iterations, ${computation_time.toFixed(1)}ms`);
    
    return {
      optimal_lambda: current_lambda,
      convergence_result: {
        converged,
        iterations: iteration,
        final_gap: this.state.primal_dual_gap.gap,
        computation_time_ms: computation_time,
      },
      domain_analysis: Array.from(this.state.domain_metrics.values()),
    };
  }
  
  /**
   * Update domain-specific λ metrics with monotonicity checking
   */
  private updateDomainMetrics(
    lambda: number,
    result: {
      selected_items: Array<{ id: string; tokens: number; delta_gain: number; domain: string }>;
      total_tokens: number;
      objective_value: number;
    },
    domain_context?: string
  ): void {
    const domains = domain_context ? [domain_context] : 
      [...new Set(result.selected_items.map(item => item.domain))];
    
    for (const domain of domains) {
      if (!this.state.domain_metrics.has(domain)) {
        this.state.domain_metrics.set(domain, {
          domain,
          lambda_values: [],
          size_curve: [],
          cbu_curve: [],
          accept_rates: [],
          monotonicity_violations: 0,
          last_updated: Date.now(),
        });
      }
      
      const metrics = this.state.domain_metrics.get(domain)!;
      
      // Calculate domain-specific metrics
      const domain_items = result.selected_items.filter(item => item.domain === domain);
      const domain_tokens = domain_items.reduce((sum, item) => sum + item.tokens, 0);
      const domain_cbu = domain_items.reduce((sum, item) => sum + item.delta_gain, 0);
      const accept_rate = domain_items.length / result.selected_items.length;
      
      // Update curves
      metrics.lambda_values.push(lambda);
      metrics.size_curve.push(domain_tokens);
      metrics.cbu_curve.push(domain_cbu);
      metrics.accept_rates.push(accept_rate);
      
      // Check monotonicity (size should decrease as λ increases)
      if (metrics.size_curve.length > 1) {
        const previous_size = metrics.size_curve[metrics.size_curve.length - 2];
        const previous_lambda = metrics.lambda_values[metrics.lambda_values.length - 2];
        
        if (lambda > previous_lambda && domain_tokens > previous_size) {
          metrics.monotonicity_violations++;
          console.warn(`⚠️ Monotonicity violation in domain '${domain}': size increased with λ`);
        }
      }
      
      metrics.last_updated = Date.now();
      
      // Limit history to prevent memory growth
      const max_history = 100;
      if (metrics.lambda_values.length > max_history) {
        metrics.lambda_values = metrics.lambda_values.slice(-max_history);
        metrics.size_curve = metrics.size_curve.slice(-max_history);
        metrics.cbu_curve = metrics.cbu_curve.slice(-max_history);
        metrics.accept_rates = metrics.accept_rates.slice(-max_history);
      }
    }
  }
  
  /**
   * Compute primal-dual gap: g = λ⋅tokens(S*) - Σ Δgain_λ(a)
   */
  private computePrimalDualGap(
    lambda: number,
    selected_items: Array<{ id: string; tokens: number; delta_gain: number; domain: string }>,
    total_tokens: number
  ): PrimalDualGap {
    const start_time = performance.now();
    
    // Primal value: λ⋅tokens(S*)
    const primal_value = lambda * total_tokens;
    
    // Dual bound: Σ Δgain_λ(a) for selected items
    const dual_bound = selected_items.reduce((sum, item) => {
      // Lagrangian gain: utility - λ * cost
      const lagrangian_gain = item.delta_gain - lambda * item.tokens;
      return sum + Math.max(0, lagrangian_gain); // Only positive contributions
    }, 0);
    
    // Gap computation
    const gap = Math.abs(primal_value - dual_bound);
    const gap_ratio = primal_value > 0 ? gap / primal_value : Infinity;
    const convergence_achieved = gap_ratio < this.config.lambda_epsilon;
    
    const computation_time = performance.now() - start_time;
    
    const result: PrimalDualGap = {
      primal_value,
      dual_bound,
      gap,
      gap_ratio,
      convergence_achieved,
      computation_time_ms: computation_time,
    };
    
    // Update state
    this.state.primal_dual_gap = result;
    
    return result;
  }
  
  /**
   * Update λ-drift analysis with moving averages and alerts
   */
  private updateDriftAnalysis(current_lambda: number, previous_lambda?: number): void {
    if (!previous_lambda) {
      this.state.drift_analysis.current_lambda = current_lambda;
      return;
    }
    
    const drift_magnitude = Math.abs(current_lambda - previous_lambda);
    const drift_direction = current_lambda > previous_lambda ? 'increase' : 
                          current_lambda < previous_lambda ? 'decrease' : 'stable';
    
    // Update moving average (exponential smoothing)
    const alpha = 0.3; // Smoothing parameter
    const moving_average = alpha * current_lambda + (1 - alpha) * this.state.drift_analysis.moving_average_lambda;
    
    // Calculate drift rate
    const drift_rate = this.state.bisection_history.length > 0 ?
      drift_magnitude / Math.max(1, this.state.bisection_history.length) : 0;
    
    // Alert trigger
    const relative_drift = previous_lambda > 0 ? drift_magnitude / previous_lambda : 0;
    const alert_triggered = relative_drift > this.config.lambda_drift_threshold;
    
    this.state.drift_analysis = {
      current_lambda,
      previous_lambda,
      drift_magnitude,
      drift_direction,
      drift_rate,
      alert_triggered,
      moving_average_lambda: moving_average,
    };
    
    // Update bisection history
    this.state.bisection_history.push(current_lambda);
    if (this.state.bisection_history.length > 50) {
      this.state.bisection_history.shift();
    }
    
    if (alert_triggered) {
      console.warn(`🚨 λ-drift alert: ${(relative_drift * 100).toFixed(1)}% drift detected`);
    }
  }
  
  /**
   * Generate comprehensive λ diagnostics dashboard
   */
  generateDiagnosticsDashboard(): {
    lambda_control_summary: {
      current_lambda: number;
      convergence_status: 'converged' | 'converging' | 'diverged';
      stability_score: number;
      drift_alert_active: boolean;
    };
    domain_analysis: Array<{
      domain: string;
      monotonicity_score: number;
      latest_size: number;
      latest_cbu: number;
      accept_rate: number;
      violations_count: number;
    }>;
    primal_dual_summary: {
      gap: number;
      gap_ratio: number;
      convergence_achieved: boolean;
      dual_behavior_verified: boolean;
    };
    performance_metrics: {
      average_bisection_time_ms: number;
      convergence_rate: number;
      lambda_stability_trend: 'improving' | 'stable' | 'degrading';
    };
  } {
    // Analyze convergence status
    const recent_gaps = this.state.convergence_trajectory.slice(-5).map(t => t.gap);
    const convergence_status = this.analyzeConvergenceStatus(recent_gaps);
    
    // Calculate stability score
    const stability_score = this.calculateLambdaStabilityScore();
    
    // Domain analysis
    const domain_analysis = Array.from(this.state.domain_metrics.entries()).map(([domain, metrics]) => {
      const monotonicity_score = Math.max(0, 1 - (metrics.monotonicity_violations / Math.max(1, metrics.lambda_values.length)));
      
      return {
        domain,
        monotonicity_score,
        latest_size: metrics.size_curve[metrics.size_curve.length - 1] || 0,
        latest_cbu: metrics.cbu_curve[metrics.cbu_curve.length - 1] || 0,
        accept_rate: metrics.accept_rates[metrics.accept_rates.length - 1] || 0,
        violations_count: metrics.monotonicity_violations,
      };
    });
    
    // Verify dual behavior (monotonic size curves)
    const dual_behavior_verified = domain_analysis.every(d => d.monotonicity_score > 0.8);
    
    return {
      lambda_control_summary: {
        current_lambda: this.state.current_lambda,
        convergence_status,
        stability_score,
        drift_alert_active: this.state.drift_analysis.alert_triggered,
      },
      domain_analysis,
      primal_dual_summary: {
        gap: this.state.primal_dual_gap.gap,
        gap_ratio: this.state.primal_dual_gap.gap_ratio,
        convergence_achieved: this.state.primal_dual_gap.convergence_achieved,
        dual_behavior_verified,
      },
      performance_metrics: {
        average_bisection_time_ms: this.state.performance_stats.average_convergence_time_ms,
        convergence_rate: this.calculateConvergenceRate(),
        lambda_stability_trend: this.analyzeLambdaStabilityTrend(),
      },
    };
  }
  
  /**
   * Utility methods for analysis
   */
  private analyzeConvergenceStatus(recent_gaps: number[]): 'converged' | 'converging' | 'diverged' {
    if (recent_gaps.length < 3) return 'converging';
    
    const latest_gap = recent_gaps[recent_gaps.length - 1];
    const trend = this.calculateTrend(recent_gaps);
    
    if (latest_gap < this.config.lambda_epsilon) return 'converged';
    if (trend < 0) return 'converging'; // Gap is decreasing
    return 'diverged';
  }
  
  private calculateLambdaStabilityScore(): number {
    if (this.state.bisection_history.length < 3) return 1.0;
    
    const recent_lambdas = this.state.bisection_history.slice(-10);
    const variance = this.calculateVariance(recent_lambdas);
    const mean = recent_lambdas.reduce((a, b) => a + b) / recent_lambdas.length;
    
    const coefficient_of_variation = mean > 0 ? Math.sqrt(variance) / mean : 0;
    return Math.max(0, 1 - coefficient_of_variation);
  }
  
  private calculateConvergenceRate(): number {
    const total_optimizations = Math.max(1, this.state.performance_stats.total_bisection_iterations / 20);
    const successful_convergences = this.state.convergence_trajectory.filter(
      t => t.gap < this.config.lambda_epsilon
    ).length;
    
    return successful_convergences / total_optimizations;
  }
  
  private analyzeLambdaStabilityTrend(): 'improving' | 'stable' | 'degrading' {
    if (this.state.bisection_history.length < 10) return 'stable';
    
    const recent_stability = this.calculateLambdaStabilityScore();
    
    // Compare with historical stability (simplified)
    const historical_lambdas = this.state.bisection_history.slice(-20, -10);
    if (historical_lambdas.length < 3) return 'stable';
    
    const historical_variance = this.calculateVariance(historical_lambdas);
    const recent_variance = this.calculateVariance(this.state.bisection_history.slice(-10));
    
    if (recent_variance < historical_variance * 0.8) return 'improving';
    if (recent_variance > historical_variance * 1.2) return 'degrading';
    return 'stable';
  }
  
  private calculateTrend(values: number[]): number {
    if (values.length < 2) return 0;
    
    const n = values.length;
    const sum_x = (n * (n - 1)) / 2; // 0 + 1 + ... + (n-1)
    const sum_y = values.reduce((a, b) => a + b);
    const sum_xy = values.reduce((sum, y, x) => sum + x * y, 0);
    const sum_x_squared = values.reduce((sum, _, x) => sum + x * x, 0);
    
    const slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x * sum_x);
    return slope;
  }
  
  private calculateVariance(values: number[]): number {
    if (values.length < 2) return 0;
    
    const mean = values.reduce((a, b) => a + b) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    return variance;
  }
  
  /**
   * Export λ control state for monitoring and analysis
   */
  exportControlState(): LambdaControlState {
    return JSON.parse(JSON.stringify(this.state));
  }
  
  /**
   * Reset control surface state
   */
  reset(): void {
    this.initializeState();
    this.warm_start_lambda = undefined;
    console.log('🎛️ Lambda Control Surface reset');
  }
  
  /**
   * Get current λ for external use
   */
  getCurrentLambda(): number {
    return this.state.current_lambda;
  }
  
  /**
   * Manual λ override (for testing/debugging)
   */
  overrideLambda(lambda: number): void {
    if (lambda < this.config.lambda_min || lambda > this.config.lambda_max) {
      throw new Error(`Lambda ${lambda} outside bounds [${this.config.lambda_min}, ${this.config.lambda_max}]`);
    }
    
    this.state.current_lambda = lambda;
    this.warm_start_lambda = lambda;
    console.log(`🎛️ Lambda manually set to ${lambda}`);
  }
}

/**
 * Convenience function for λ bisection with control surface
 */
export async function executeLambdaBisectionWithControl(
  objective_function: (lambda: number) => Promise<{
    selected_items: Array<{ id: string; tokens: number; delta_gain: number; domain: string }>;
    total_tokens: number;
    objective_value: number;
  }>,
  target_tokens: number,
  config: Partial<LambdaControlConfig> = {},
  domain_context?: string
): Promise<{
  optimal_lambda: number;
  control_surface: LambdaControlSurface;
  diagnostics: ReturnType<LambdaControlSurface['generateDiagnosticsDashboard']>;
}> {
  const control_surface = new LambdaControlSurface(config);
  
  const result = await control_surface.executeLambdaBisection(
    objective_function,
    target_tokens,
    domain_context
  );
  
  const diagnostics = control_surface.generateDiagnosticsDashboard();
  
  return {
    optimal_lambda: result.optimal_lambda,
    control_surface,
    diagnostics,
  };
}
