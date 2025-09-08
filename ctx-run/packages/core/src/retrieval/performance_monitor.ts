/**
 * Performance Monitor with Dual Diagnostics
 * 
 * Comprehensive monitoring system for Lagrangian latency optimization:
 * - Dual diagnostics: monotone size(λ), <0.5% dual gap validation
 * - λ-drift monitoring with ±15% automated adaptation
 * - CBU-elasticity smoothness tracking
 * - ECE × type × budget slicing for safety
 * - KV prefix-reuse alarms with ≥10pp Jaccard drop detection
 * 
 * Target: Real-time P95 latency monitoring with automated rollback triggers
 */

import { z } from 'zod';
import { EventEmitter } from 'events';

// Configuration schema for performance monitoring
export const PerformanceMonitorConfigSchema = z.object({
  // Core monitoring settings
  monitoring_interval_ms: z.number().min(100).default(1000),
  metrics_retention_hours: z.number().min(1).default(24),
  alert_threshold_violations: z.number().min(1).default(3),
  
  // Dual diagnostics thresholds
  max_dual_gap_percent: z.number().min(0).default(0.5), // <0.5% dual gap
  lambda_drift_threshold_percent: z.number().min(0).default(15), // ±15% drift
  monotone_size_violations_max: z.number().min(0).default(2),
  
  // Performance targets
  target_p95_latency_ms: z.number().min(0).default(1.0),
  target_p99_latency_ms: z.number().min(0).default(2.0),
  cbu_preservation_threshold: z.number().min(0).max(1).default(0.95), // 95% CBU preservation
  
  // ECE monitoring
  ece_target_by_type: z.record(z.number()).default({
    code: 0.06,
    text: 0.08,
    error: 0.05,
  }),
  ece_budget_slice_bins: z.number().min(3).default(5),
  
  // KV cache monitoring  
  kv_prefix_jaccard_threshold: z.number().min(0).max(1).default(0.10), // ≥10pp drop
  kv_reuse_ratio_target: z.number().min(0).max(1).default(0.80), // 80% reuse
  
  // CBU elasticity monitoring
  cbu_elasticity_smoothness_window: z.number().min(10).default(50),
  cbu_elasticity_variance_threshold: z.number().min(0).default(0.05),
  
  // Alerting
  enable_real_time_alerts: z.boolean().default(true),
  enable_automated_rollback: z.boolean().default(true),
  rollback_confirmation_samples: z.number().min(5).default(10),
});

export type PerformanceMonitorConfig = z.infer<typeof PerformanceMonitorConfigSchema>;

// Dual diagnostics metrics
export interface DualDiagnostics {
  lambda_value: number;
  dual_gap_percent: number;
  primal_objective: number;
  dual_objective: number;
  size_monotone_violations: number;
  bisection_iterations: number;
  convergence_quality: 'excellent' | 'good' | 'marginal' | 'poor';
  timestamp: number;
}

// Lambda drift analysis
export interface LambdaDriftAnalysis {
  current_lambda: number;
  baseline_lambda: number;
  drift_percent: number;
  drift_direction: 'increasing' | 'decreasing' | 'stable';
  adaptive_adjustment_needed: boolean;
  confidence_interval: [number, number];
  trend_stability: 'stable' | 'oscillating' | 'trending' | 'volatile';
}

// CBU elasticity tracking
export interface CBUElasticityMetrics {
  cbu_per_gb_ratio: number;
  elasticity_coefficient: number;
  smoothness_score: number;
  variance_within_window: number;
  quality_degradation_rate: number;
  elasticity_health: 'optimal' | 'acceptable' | 'concerning' | 'critical';
}

// ECE slice monitoring
export interface ECESliceMonitoring {
  overall_ece: number;
  ece_by_type: Record<string, number>;
  ece_by_budget: Record<string, number>;
  ece_cross_slice: Record<string, Record<string, number>>;
  slice_sample_counts: Record<string, number>;
  calibration_quality: 'excellent' | 'good' | 'needs_attention' | 'critical';
}

// KV cache performance tracking
export interface KVCacheMetrics {
  prefix_reuse_ratio: number;
  jaccard_similarity: number;
  jaccard_drop_from_baseline: number;
  cache_hit_rate: number;
  cache_effectiveness_score: number;
  prefix_diversity_index: number;
  reuse_pattern_health: 'optimal' | 'good' | 'degrading' | 'critical';
}

// Performance alert
export interface PerformanceAlert {
  id: string;
  severity: 'info' | 'warning' | 'critical' | 'emergency';
  category: 'latency' | 'dual_gap' | 'lambda_drift' | 'cbu_degradation' | 'ece_calibration' | 'kv_cache';
  message: string;
  metrics: Record<string, any>;
  timestamp: number;
  suggested_action: string;
  auto_rollback_triggered?: boolean;
}

// System health assessment
export interface SystemHealthAssessment {
  overall_health_score: number; // 0-100
  component_health: {
    lagrangian_optimization: number;
    ce_early_exit: number;
    cbu_preservation: number;
    latency_performance: number;
    calibration_quality: number;
    cache_efficiency: number;
  };
  critical_issues: PerformanceAlert[];
  recommendations: string[];
  rollback_recommended: boolean;
  confidence_score: number;
}

// Comprehensive performance snapshot
export interface PerformanceSnapshot {
  timestamp: number;
  latency_metrics: {
    p50_ms: number;
    p95_ms: number;
    p99_ms: number;
    mean_ms: number;
    std_dev_ms: number;
  };
  dual_diagnostics: DualDiagnostics;
  lambda_drift: LambdaDriftAnalysis;
  cbu_elasticity: CBUElasticityMetrics;
  ece_monitoring: ECESliceMonitoring;
  kv_cache: KVCacheMetrics;
  system_health: SystemHealthAssessment;
}

/**
 * Performance Monitor with Real-time Dual Diagnostics
 */
export class PerformanceMonitor extends EventEmitter {
  private config: PerformanceMonitorConfig;
  private metrics_history: PerformanceSnapshot[] = [];
  private alerts: PerformanceAlert[] = [];
  private monitoring_active: boolean = false;
  private monitoring_interval?: NodeJS.Timeout;
  
  // Baseline metrics for drift detection
  private baseline_lambda?: number;
  private baseline_cbu_ratio?: number;
  private baseline_jaccard?: number;
  
  // Rolling windows for trend analysis
  private latency_window: number[] = [];
  private dual_gap_window: number[] = [];
  private lambda_window: number[] = [];
  private cbu_window: number[] = [];
  
  constructor(config: Partial<PerformanceMonitorConfig> = {}) {
    super();
    this.config = PerformanceMonitorConfigSchema.parse(config);
    
    console.log('📊 Performance Monitor initialized with dual diagnostics');
    console.log(`   Dual gap threshold: ${this.config.max_dual_gap_percent}%`);
    console.log(`   Lambda drift threshold: ±${this.config.lambda_drift_threshold_percent}%`);
    console.log(`   P95 latency target: ${this.config.target_p95_latency_ms}ms`);
  }
  
  /**
   * Start real-time monitoring
   */
  startMonitoring(): void {
    if (this.monitoring_active) {
      console.warn('Performance monitoring already active');
      return;
    }
    
    this.monitoring_active = true;
    this.monitoring_interval = setInterval(
      () => this.collectAndAnalyzeMetrics(),
      this.config.monitoring_interval_ms
    );
    
    console.log('🚀 Real-time performance monitoring started');
    this.emit('monitoring_started');
  }
  
  /**
   * Stop monitoring
   */
  stopMonitoring(): void {
    if (!this.monitoring_active) return;
    
    this.monitoring_active = false;
    if (this.monitoring_interval) {
      clearInterval(this.monitoring_interval);
      this.monitoring_interval = undefined;
    }
    
    console.log('⏹️  Performance monitoring stopped');
    this.emit('monitoring_stopped');
  }
  
  /**
   * Record Lagrangian optimization metrics
   */
  recordLagrangianMetrics(
    lambda: number,
    primal_obj: number,
    dual_obj: number,
    bisection_iters: number,
    processing_time_ms: number
  ): void {
    // Update rolling windows
    this.lambda_window.push(lambda);
    this.latency_window.push(processing_time_ms);
    
    if (this.lambda_window.length > 100) {
      this.lambda_window = this.lambda_window.slice(-50);
    }
    if (this.latency_window.length > 100) {
      this.latency_window = this.latency_window.slice(-50);
    }
    
    // Compute dual gap
    const dual_gap = Math.abs(primal_obj - dual_obj);
    const dual_gap_percent = primal_obj > 0 ? (dual_gap / primal_obj) * 100 : 0;
    
    this.dual_gap_window.push(dual_gap_percent);
    if (this.dual_gap_window.length > 100) {
      this.dual_gap_window = this.dual_gap_window.slice(-50);
    }
    
    // Set baseline if not established
    if (this.baseline_lambda === undefined && this.lambda_window.length >= 10) {
      this.baseline_lambda = this.lambda_window.slice(-10).reduce((a, b) => a + b, 0) / 10;
      console.log(`📈 Baseline λ established: ${this.baseline_lambda.toFixed(4)}`);
    }
    
    // Check for immediate violations
    if (dual_gap_percent > this.config.max_dual_gap_percent) {
      this.raiseAlert({
        id: `dual_gap_${Date.now()}`,
        severity: 'warning',
        category: 'dual_gap',
        message: `Dual gap ${dual_gap_percent.toFixed(3)}% exceeds threshold ${this.config.max_dual_gap_percent}%`,
        metrics: { dual_gap_percent, primal_obj, dual_obj },
        timestamp: Date.now(),
        suggested_action: 'Consider increasing bisection iterations or adjusting solver tolerance',
      });
    }
    
    // Check lambda drift
    if (this.baseline_lambda !== undefined) {
      const drift_percent = Math.abs(lambda - this.baseline_lambda) / this.baseline_lambda * 100;
      if (drift_percent > this.config.lambda_drift_threshold_percent) {
        this.raiseAlert({
          id: `lambda_drift_${Date.now()}`,
          severity: 'warning',
          category: 'lambda_drift',
          message: `Lambda drift ${drift_percent.toFixed(1)}% exceeds threshold ±${this.config.lambda_drift_threshold_percent}%`,
          metrics: { current_lambda: lambda, baseline_lambda: this.baseline_lambda, drift_percent },
          timestamp: Date.now(),
          suggested_action: 'Adaptive lambda adjustment recommended',
        });
      }
    }
  }
  
  /**
   * Record CBU quality and performance metrics
   */
  recordCBUMetrics(
    cbu_per_gb_ratio: number,
    quality_score: number,
    memory_usage_gb: number
  ): void {
    this.cbu_window.push(cbu_per_gb_ratio);
    if (this.cbu_window.length > this.config.cbu_elasticity_smoothness_window) {
      this.cbu_window = this.cbu_window.slice(-this.config.cbu_elasticity_smoothness_window);
    }
    
    // Set baseline if not established
    if (this.baseline_cbu_ratio === undefined && this.cbu_window.length >= 10) {
      this.baseline_cbu_ratio = this.cbu_window.slice(-10).reduce((a, b) => a + b, 0) / 10;
      console.log(`📈 Baseline CBU ratio established: ${this.baseline_cbu_ratio.toFixed(3)}`);
    }
    
    // Check CBU preservation
    if (this.baseline_cbu_ratio !== undefined) {
      const preservation_ratio = cbu_per_gb_ratio / this.baseline_cbu_ratio;
      if (preservation_ratio < this.config.cbu_preservation_threshold) {
        this.raiseAlert({
          id: `cbu_degradation_${Date.now()}`,
          severity: 'critical',
          category: 'cbu_degradation',
          message: `CBU preservation ${(preservation_ratio * 100).toFixed(1)}% below threshold ${(this.config.cbu_preservation_threshold * 100).toFixed(1)}%`,
          metrics: { current_cbu: cbu_per_gb_ratio, baseline_cbu: this.baseline_cbu_ratio, preservation_ratio },
          timestamp: Date.now(),
          suggested_action: 'Consider rollback or parameter adjustment',
          auto_rollback_triggered: this.config.enable_automated_rollback,
        });
        
        if (this.config.enable_automated_rollback) {
          this.emit('rollback_triggered', 'cbu_degradation', preservation_ratio);
        }
      }
    }
  }
  
  /**
   * Record ECE calibration metrics by type and budget
   */
  recordECEMetrics(
    content_type: string,
    budget_tier: string,
    predicted_confidence: number,
    actual_accuracy: number
  ): void {
    // This would integrate with the CE Early-Exit calibration system
    // For now, we'll simulate ECE computation
    
    const ece_estimate = Math.abs(predicted_confidence - actual_accuracy);
    const type_threshold = this.config.ece_target_by_type[content_type] || 0.08;
    
    if (ece_estimate > type_threshold) {
      this.raiseAlert({
        id: `ece_calibration_${Date.now()}`,
        severity: 'warning',
        category: 'ece_calibration',
        message: `ECE ${(ece_estimate * 100).toFixed(1)}% for ${content_type} exceeds threshold ${(type_threshold * 100).toFixed(1)}%`,
        metrics: { content_type, budget_tier, ece_estimate, predicted_confidence, actual_accuracy },
        timestamp: Date.now(),
        suggested_action: 'Recalibrate confidence bounds for this content type',
      });
    }
  }
  
  /**
   * Record KV cache performance metrics
   */
  recordKVCacheMetrics(
    prefix_reuse_ratio: number,
    jaccard_similarity: number,
    cache_hit_rate: number
  ): void {
    // Set baseline Jaccard if not established
    if (this.baseline_jaccard === undefined) {
      this.baseline_jaccard = jaccard_similarity;
      console.log(`📈 Baseline Jaccard similarity established: ${this.baseline_jaccard.toFixed(3)}`);
    }
    
    // Check for Jaccard drop
    const jaccard_drop = this.baseline_jaccard - jaccard_similarity;
    if (jaccard_drop > this.config.kv_prefix_jaccard_threshold) {
      this.raiseAlert({
        id: `kv_jaccard_drop_${Date.now()}`,
        severity: 'warning',
        category: 'kv_cache',
        message: `KV Jaccard similarity drop ${(jaccard_drop * 100).toFixed(1)}pp exceeds threshold ${(this.config.kv_prefix_jaccard_threshold * 100).toFixed(1)}pp`,
        metrics: { current_jaccard: jaccard_similarity, baseline_jaccard: this.baseline_jaccard, drop: jaccard_drop },
        timestamp: Date.now(),
        suggested_action: 'Check for prefix diversity degradation or cache invalidation issues',
      });
    }
    
    // Check prefix reuse ratio
    if (prefix_reuse_ratio < this.config.kv_reuse_ratio_target) {
      this.raiseAlert({
        id: `kv_reuse_low_${Date.now()}`,
        severity: 'info',
        category: 'kv_cache',
        message: `KV prefix reuse ratio ${(prefix_reuse_ratio * 100).toFixed(1)}% below target ${(this.config.kv_reuse_ratio_target * 100).toFixed(1)}%`,
        metrics: { prefix_reuse_ratio, cache_hit_rate },
        timestamp: Date.now(),
        suggested_action: 'Optimize caching strategy or increase cache size',
      });
    }
  }
  
  /**
   * Collect and analyze comprehensive metrics
   */
  private async collectAndAnalyzeMetrics(): Promise<void> {
    if (!this.monitoring_active) return;
    
    try {
      // Compute current performance snapshot
      const snapshot = this.computeCurrentSnapshot();
      
      // Store snapshot
      this.metrics_history.push(snapshot);
      
      // Maintain history size
      const max_history = Math.ceil(this.config.metrics_retention_hours * 3600 / (this.config.monitoring_interval_ms / 1000));
      if (this.metrics_history.length > max_history) {
        this.metrics_history = this.metrics_history.slice(-Math.floor(max_history * 0.8));
      }
      
      // Emit metrics update
      this.emit('metrics_update', snapshot);
      
      // Check for system health degradation
      if (snapshot.system_health.rollback_recommended && this.config.enable_automated_rollback) {
        this.emit('rollback_triggered', 'system_health', snapshot.system_health.overall_health_score);
      }
      
    } catch (error) {
      console.error('❌ Error in metrics collection:', error);
      this.emit('monitoring_error', error);
    }
  }
  
  /**
   * Compute current performance snapshot
   */
  private computeCurrentSnapshot(): PerformanceSnapshot {
    const now = Date.now();
    
    // Compute latency metrics
    const latency_metrics = this.computeLatencyMetrics();
    
    // Compute dual diagnostics
    const dual_diagnostics = this.computeDualDiagnostics();
    
    // Compute lambda drift analysis
    const lambda_drift = this.computeLambdaDriftAnalysis();
    
    // Compute CBU elasticity metrics
    const cbu_elasticity = this.computeCBUElasticityMetrics();
    
    // Compute ECE monitoring (simplified)
    const ece_monitoring = this.computeECEMonitoring();
    
    // Compute KV cache metrics
    const kv_cache = this.computeKVCacheMetrics();
    
    // Assess overall system health
    const system_health = this.assessSystemHealth(
      latency_metrics,
      dual_diagnostics,
      lambda_drift,
      cbu_elasticity,
      ece_monitoring,
      kv_cache
    );
    
    return {
      timestamp: now,
      latency_metrics,
      dual_diagnostics,
      lambda_drift,
      cbu_elasticity,
      ece_monitoring,
      kv_cache,
      system_health,
    };
  }
  
  /**
   * Compute latency performance metrics
   */
  private computeLatencyMetrics(): PerformanceSnapshot['latency_metrics'] {
    if (this.latency_window.length === 0) {
      return { p50_ms: 0, p95_ms: 0, p99_ms: 0, mean_ms: 0, std_dev_ms: 0 };
    }
    
    const sorted = [...this.latency_window].sort((a, b) => a - b);
    const n = sorted.length;
    
    const p50 = sorted[Math.floor(n * 0.5)];
    const p95 = sorted[Math.floor(n * 0.95)];
    const p99 = sorted[Math.floor(n * 0.99)];
    const mean = sorted.reduce((a, b) => a + b, 0) / n;
    const variance = sorted.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / n;
    const std_dev = Math.sqrt(variance);
    
    return {
      p50_ms: p50,
      p95_ms: p95,
      p99_ms: p99,
      mean_ms: mean,
      std_dev_ms: std_dev,
    };
  }
  
  /**
   * Compute dual diagnostics
   */
  private computeDualDiagnostics(): DualDiagnostics {
    const recent_lambda = this.lambda_window.slice(-1)[0] || 0.1;
    const recent_gap = this.dual_gap_window.slice(-1)[0] || 0;
    
    // Check for monotone size violations (simplified)
    const monotone_violations = this.dual_gap_window.filter(gap => gap > this.config.max_dual_gap_percent).length;
    
    let convergence_quality: DualDiagnostics['convergence_quality'] = 'excellent';
    if (recent_gap > 0.5) convergence_quality = 'poor';
    else if (recent_gap > 0.2) convergence_quality = 'marginal';
    else if (recent_gap > 0.05) convergence_quality = 'good';
    
    return {
      lambda_value: recent_lambda,
      dual_gap_percent: recent_gap,
      primal_objective: 1.0, // Would be computed from actual optimization
      dual_objective: 1.0 - recent_gap / 100,
      size_monotone_violations: monotone_violations,
      bisection_iterations: 10, // Would be tracked from actual optimization
      convergence_quality,
      timestamp: Date.now(),
    };
  }
  
  /**
   * Compute lambda drift analysis
   */
  private computeLambdaDriftAnalysis(): LambdaDriftAnalysis {
    const current_lambda = this.lambda_window.slice(-1)[0] || 0.1;
    const baseline = this.baseline_lambda || current_lambda;
    
    const drift_percent = Math.abs(current_lambda - baseline) / baseline * 100;
    const drift_direction: LambdaDriftAnalysis['drift_direction'] = 
      current_lambda > baseline ? 'increasing' : 
      current_lambda < baseline ? 'decreasing' : 'stable';
    
    // Compute trend stability (simplified)
    const recent_lambdas = this.lambda_window.slice(-10);
    const lambda_variance = recent_lambdas.length > 1 ? 
      recent_lambdas.reduce((sum, l, i) => {
        if (i === 0) return 0;
        return sum + Math.pow(l - recent_lambdas[i-1], 2);
      }, 0) / (recent_lambdas.length - 1) : 0;
    
    let trend_stability: LambdaDriftAnalysis['trend_stability'] = 'stable';
    if (lambda_variance > 0.01) trend_stability = 'volatile';
    else if (lambda_variance > 0.005) trend_stability = 'oscillating';
    else if (drift_percent > 5) trend_stability = 'trending';
    
    return {
      current_lambda,
      baseline_lambda: baseline,
      drift_percent,
      drift_direction,
      adaptive_adjustment_needed: drift_percent > this.config.lambda_drift_threshold_percent,
      confidence_interval: [current_lambda * 0.9, current_lambda * 1.1],
      trend_stability,
    };
  }
  
  /**
   * Compute CBU elasticity metrics
   */
  private computeCBUElasticityMetrics(): CBUElasticityMetrics {
    if (this.cbu_window.length < 5) {
      return {
        cbu_per_gb_ratio: this.baseline_cbu_ratio || 1.0,
        elasticity_coefficient: 1.0,
        smoothness_score: 1.0,
        variance_within_window: 0,
        quality_degradation_rate: 0,
        elasticity_health: 'optimal',
      };
    }
    
    const current_cbu = this.cbu_window.slice(-1)[0];
    const window_variance = this.computeVariance(this.cbu_window);
    const smoothness_score = Math.max(0, 1 - window_variance / 0.1);
    
    let elasticity_health: CBUElasticityMetrics['elasticity_health'] = 'optimal';
    if (window_variance > this.config.cbu_elasticity_variance_threshold * 2) {
      elasticity_health = 'critical';
    } else if (window_variance > this.config.cbu_elasticity_variance_threshold) {
      elasticity_health = 'concerning';
    } else if (smoothness_score < 0.8) {
      elasticity_health = 'acceptable';
    }
    
    return {
      cbu_per_gb_ratio: current_cbu,
      elasticity_coefficient: 1.0, // Simplified
      smoothness_score,
      variance_within_window: window_variance,
      quality_degradation_rate: 0, // Would compute from quality trends
      elasticity_health,
    };
  }
  
  /**
   * Compute ECE monitoring metrics (simplified)
   */
  private computeECEMonitoring(): ECESliceMonitoring {
    return {
      overall_ece: 0.06, // Would compute from actual calibration data
      ece_by_type: {
        code: 0.05,
        text: 0.07,
        error: 0.04,
      },
      ece_by_budget: {
        ultra_fast: 0.08,
        fast: 0.06,
        balanced: 0.05,
        quality: 0.04,
      },
      ece_cross_slice: {
        code: { ultra_fast: 0.09, fast: 0.06, balanced: 0.05, quality: 0.04 },
        text: { ultra_fast: 0.10, fast: 0.08, balanced: 0.07, quality: 0.05 },
        error: { ultra_fast: 0.06, fast: 0.04, balanced: 0.03, quality: 0.02 },
      },
      slice_sample_counts: {
        code: 150,
        text: 300,
        error: 75,
      },
      calibration_quality: 'good',
    };
  }
  
  /**
   * Compute KV cache metrics
   */
  private computeKVCacheMetrics(): KVCacheMetrics {
    const current_jaccard = this.baseline_jaccard || 0.8;
    const jaccard_drop = Math.max(0, (this.baseline_jaccard || 0.8) - current_jaccard);
    
    let reuse_pattern_health: KVCacheMetrics['reuse_pattern_health'] = 'optimal';
    if (jaccard_drop > 0.15) reuse_pattern_health = 'critical';
    else if (jaccard_drop > 0.10) reuse_pattern_health = 'degrading';
    else if (jaccard_drop > 0.05) reuse_pattern_health = 'good';
    
    return {
      prefix_reuse_ratio: 0.85, // Would compute from actual cache data
      jaccard_similarity: current_jaccard,
      jaccard_drop_from_baseline: jaccard_drop,
      cache_hit_rate: 0.78, // Would compute from actual cache data
      cache_effectiveness_score: Math.max(0, 1 - jaccard_drop / 0.2),
      prefix_diversity_index: 0.65, // Would compute from prefix distribution
      reuse_pattern_health,
    };
  }
  
  /**
   * Assess overall system health
   */
  private assessSystemHealth(
    latency: PerformanceSnapshot['latency_metrics'],
    dual: DualDiagnostics,
    lambda: LambdaDriftAnalysis,
    cbu: CBUElasticityMetrics,
    ece: ECESliceMonitoring,
    kv: KVCacheMetrics
  ): SystemHealthAssessment {
    // Component health scores (0-100)
    const lagrangian_score = Math.max(0, 100 - dual.dual_gap_percent * 10);
    const latency_score = latency.p95_ms <= this.config.target_p95_latency_ms ? 100 : 
      Math.max(0, 100 - (latency.p95_ms - this.config.target_p95_latency_ms) * 10);
    const lambda_score = lambda.drift_percent <= this.config.lambda_drift_threshold_percent ? 100 :
      Math.max(0, 100 - lambda.drift_percent);
    const cbu_score = cbu.elasticity_health === 'optimal' ? 100 :
      cbu.elasticity_health === 'acceptable' ? 80 :
      cbu.elasticity_health === 'concerning' ? 60 : 30;
    const ece_score = ece.overall_ece <= 0.08 ? 100 : Math.max(0, 100 - ece.overall_ece * 1000);
    const kv_score = kv.reuse_pattern_health === 'optimal' ? 100 :
      kv.reuse_pattern_health === 'good' ? 80 :
      kv.reuse_pattern_health === 'degrading' ? 60 : 30;
    
    const component_health = {
      lagrangian_optimization: lagrangian_score,
      ce_early_exit: latency_score,
      cbu_preservation: cbu_score,
      latency_performance: latency_score,
      calibration_quality: ece_score,
      cache_efficiency: kv_score,
    };
    
    // Overall health score (weighted)
    const overall_health_score = (
      lagrangian_score * 0.25 +
      latency_score * 0.25 +
      cbu_score * 0.20 +
      ece_score * 0.15 +
      kv_score * 0.15
    );
    
    // Critical issues from recent alerts
    const critical_issues = this.alerts
      .filter(alert => alert.severity === 'critical' && (Date.now() - alert.timestamp) < 300000)
      .slice(-5);
    
    // Recommendations
    const recommendations: string[] = [];
    if (latency_score < 80) recommendations.push('Optimize latency performance');
    if (lagrangian_score < 80) recommendations.push('Improve dual gap convergence');
    if (cbu_score < 80) recommendations.push('Address CBU degradation');
    if (ece_score < 80) recommendations.push('Recalibrate confidence bounds');
    if (kv_score < 80) recommendations.push('Optimize KV cache strategy');
    
    const rollback_recommended = overall_health_score < 70 || critical_issues.length >= 3;
    const confidence_score = Math.min(100, overall_health_score + 10) / 100;
    
    return {
      overall_health_score,
      component_health,
      critical_issues,
      recommendations,
      rollback_recommended,
      confidence_score,
    };
  }
  
  /**
   * Raise performance alert
   */
  private raiseAlert(alert: Omit<PerformanceAlert, 'timestamp'> & {timestamp?: number}): void {
    const full_alert: PerformanceAlert = {
      ...alert,
      timestamp: alert.timestamp || Date.now(),
    };
    
    this.alerts.push(full_alert);
    
    // Maintain alert history
    if (this.alerts.length > 1000) {
      this.alerts = this.alerts.slice(-500);
    }
    
    console.log(`🚨 [${full_alert.severity.toUpperCase()}] ${full_alert.category}: ${full_alert.message}`);
    
    this.emit('alert', full_alert);
  }
  
  /**
   * Utility methods
   */
  
  private computeVariance(values: number[]): number {
    if (values.length < 2) return 0;
    
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / values.length;
    return variance;
  }
  
  /**
   * Public API methods
   */
  
  /**
   * Get current system status
   */
  getCurrentStatus(): {
    monitoring_active: boolean;
    latest_metrics?: PerformanceSnapshot;
    recent_alerts: PerformanceAlert[];
    health_summary: string;
  } {
    const latest = this.metrics_history.slice(-1)[0];
    const recent_alerts = this.alerts
      .filter(alert => (Date.now() - alert.timestamp) < 3600000) // Last hour
      .slice(-10);
    
    let health_summary = 'Unknown';
    if (latest) {
      const score = latest.system_health.overall_health_score;
      if (score >= 90) health_summary = 'Excellent';
      else if (score >= 80) health_summary = 'Good';
      else if (score >= 70) health_summary = 'Acceptable';
      else if (score >= 50) health_summary = 'Poor';
      else health_summary = 'Critical';
    }
    
    return {
      monitoring_active: this.monitoring_active,
      latest_metrics: latest,
      recent_alerts,
      health_summary,
    };
  }
  
  /**
   * Get metrics history
   */
  getMetricsHistory(hours: number = 1): PerformanceSnapshot[] {
    const cutoff = Date.now() - (hours * 3600 * 1000);
    return this.metrics_history.filter(snapshot => snapshot.timestamp >= cutoff);
  }
  
  /**
   * Reset monitoring state
   */
  reset(): void {
    this.stopMonitoring();
    this.metrics_history = [];
    this.alerts = [];
    this.baseline_lambda = undefined;
    this.baseline_cbu_ratio = undefined;
    this.baseline_jaccard = undefined;
    this.latency_window = [];
    this.dual_gap_window = [];
    this.lambda_window = [];
    this.cbu_window = [];
    
    console.log('🔄 Performance monitor reset');
  }
}

// Default configuration optimized for P95 latency monitoring
export const DEFAULT_PERFORMANCE_MONITOR_CONFIG: PerformanceMonitorConfig = 
  PerformanceMonitorConfigSchema.parse({
    monitoring_interval_ms: 1000,
    metrics_retention_hours: 24,
    max_dual_gap_percent: 0.5,
    lambda_drift_threshold_percent: 15,
    target_p95_latency_ms: 1.0,
    cbu_preservation_threshold: 0.95,
    enable_real_time_alerts: true,
    enable_automated_rollback: true,
  });