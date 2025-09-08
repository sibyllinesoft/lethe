/**
 * Lagrangian Deployment Orchestrator
 * 
 * Master coordination system for comprehensive Lagrangian latency optimization deployment.
 * Integrates all mathematical components with safety mechanisms and monitoring.
 * 
 * Target: Deploy 85% latency reduction (6.8ms → ≤1ms P95) while maintaining +12.5% CBU
 * 
 * Core Integration:
 * 1. Mathematical Orchestrator (Lagrangian optimization)
 * 2. CE Early-Exit System (calibrated stopping)
 * 3. Performance Monitor (dual diagnostics)
 * 4. Safety & Rollback Systems
 * 5. Statistical Validation Framework
 */

import { z } from 'zod';
import { EventEmitter } from 'events';
import type { Candidate } from './index.js';

// Import core systems
import { 
  MathematicalOrchestrator, 
  type MathematicalCandidate,
  type MathematicalOptimizationResult,
  type MathematicalOrchestratorConfig 
} from './mathematical_orchestrator.js';

import { 
  CEEarlyExitSystem, 
  EarlyExitCrossEncoderReranker,
  type CEEarlyExitConfig,
  type EarlyExitDecision,
  type MatryoshkaRouting 
} from './ce_early_exit.js';

import { 
  PerformanceMonitor,
  type PerformanceMonitorConfig,
  type PerformanceSnapshot,
  type PerformanceAlert,
  type SystemHealthAssessment 
} from './performance_monitor.js';

// Deployment orchestrator configuration
export const LagrangianDeploymentConfigSchema = z.object({
  // Deployment settings
  enable_canary_deployment: z.boolean().default(true),
  canary_traffic_percent: z.number().min(0).max(100).default(5),
  rollout_stages: z.array(z.number()).default([5, 15, 35, 65, 100]),
  stage_duration_minutes: z.number().min(1).default(10),
  
  // Performance targets
  target_p95_latency_ms: z.number().min(0).default(1.0), // ≤1ms
  baseline_p95_latency_ms: z.number().min(0).default(6.8), // Current: +6.8ms
  target_latency_improvement: z.number().min(0).default(0.85), // 85% reduction
  
  // Quality preservation
  target_cbu_preservation: z.number().min(0).max(1).default(1.125), // Maintain +12.5% CBU
  min_cbu_threshold: z.number().min(0).max(1).default(1.05), // Minimum +5% CBU
  quality_degradation_tolerance: z.number().min(0).max(1).default(0.05), // 5% tolerance
  
  // Safety mechanisms
  enable_automated_rollback: z.boolean().default(true),
  rollback_trigger_threshold: z.number().min(0).max(1).default(0.7), // Health score
  rollback_confirmation_samples: z.number().min(5).default(20),
  max_rollback_attempts: z.number().min(1).default(3),
  
  // Statistical validation
  enable_statistical_validation: z.boolean().default(true),
  significance_level: z.number().min(0).max(1).default(0.05), // p < 0.05
  minimum_sample_size: z.number().min(10).default(100),
  validation_window_minutes: z.number().min(5).default(30),
  
  // Component configurations
  mathematical_orchestrator: z.object({}).default({}),
  ce_early_exit: z.object({}).default({}),
  performance_monitor: z.object({}).default({}),
  
  // Monitoring and logging
  enable_comprehensive_logging: z.boolean().default(true),
  log_level: z.enum(['debug', 'info', 'warning', 'error']).default('info'),
  metrics_collection_interval_ms: z.number().min(100).default(1000),
});

export type LagrangianDeploymentConfig = z.infer<typeof LagrangianDeploymentConfigSchema>;

// Deployment state
export interface DeploymentState {
  stage: 'initializing' | 'canary' | 'rollout' | 'complete' | 'rolling_back' | 'failed';
  current_stage_percent: number;
  start_time: number;
  stage_start_time: number;
  total_samples_processed: number;
  successful_samples: number;
  rollback_count: number;
  health_score: number;
}

// Deployment metrics
export interface DeploymentMetrics {
  // Latency performance
  baseline_p95_ms: number;
  current_p95_ms: number;
  latency_improvement_percent: number;
  latency_target_achieved: boolean;
  
  // Quality preservation
  baseline_cbu_ratio: number;
  current_cbu_ratio: number;
  cbu_preservation_ratio: number;
  cbu_target_achieved: boolean;
  
  // System health
  overall_health_score: number;
  component_health_scores: Record<string, number>;
  critical_alerts_count: number;
  
  // Statistical validation
  statistical_significance_achieved: boolean;
  p_value: number;
  confidence_interval: [number, number];
  effect_size: number;
  
  // Deployment progress
  deployment_state: DeploymentState;
  rollout_progress_percent: number;
  estimated_completion_time: number;
}

// Rollback decision analysis
export interface RollbackAnalysis {
  should_rollback: boolean;
  trigger_reasons: string[];
  confidence_score: number;
  impact_assessment: {
    latency_degradation: number;
    quality_degradation: number;
    system_stability_risk: number;
  };
  recommended_action: 'continue' | 'pause' | 'rollback' | 'investigate';
  rollback_strategy: 'immediate' | 'gradual' | 'staged';
}

// Canary analysis result
export interface CanaryAnalysisResult {
  canary_successful: boolean;
  metrics_comparison: {
    latency_improvement: number;
    quality_preservation: number;
    stability_score: number;
  };
  statistical_significance: {
    latency_p_value: number;
    quality_p_value: number;
    effect_size: number;
  };
  recommendation: 'proceed' | 'extend_canary' | 'rollback';
  next_stage_percent: number;
}

/**
 * Lagrangian Deployment Orchestrator
 * 
 * Master system that coordinates the deployment of all Lagrangian optimization
 * components with comprehensive safety mechanisms and monitoring.
 */
export class LagrangianDeploymentOrchestrator extends EventEmitter {
  private config: LagrangianDeploymentConfig;
  private deployment_state: DeploymentState;
  
  // Core systems
  private math_orchestrator?: MathematicalOrchestrator;
  private ce_early_exit?: CEEarlyExitSystem;
  private early_exit_reranker?: EarlyExitCrossEncoderReranker;
  private performance_monitor: PerformanceMonitor;
  
  // Deployment tracking
  private baseline_metrics?: DeploymentMetrics;
  private current_metrics?: DeploymentMetrics;
  private rollback_history: Array<{timestamp: number, reason: string, stage: number}> = [];
  
  // Statistical tracking
  private latency_samples: number[] = [];
  private quality_samples: number[] = [];
  private baseline_latency_samples: number[] = [];
  private baseline_quality_samples: number[] = [];
  
  constructor(config: Partial<LagrangianDeploymentConfig> = {}) {
    super();
    this.config = LagrangianDeploymentConfigSchema.parse(config);
    
    this.deployment_state = {
      stage: 'initializing',
      current_stage_percent: 0,
      start_time: Date.now(),
      stage_start_time: Date.now(),
      total_samples_processed: 0,
      successful_samples: 0,
      rollback_count: 0,
      health_score: 100,
    };
    
    // Initialize performance monitor
    this.performance_monitor = new PerformanceMonitor({
      ...this.config.performance_monitor,
      enable_automated_rollback: this.config.enable_automated_rollback,
      target_p95_latency_ms: this.config.target_p95_latency_ms,
    } as Partial<PerformanceMonitorConfig>);
    
    // Set up event handlers
    this.setupEventHandlers();
    
    console.log('🚀 Lagrangian Deployment Orchestrator initialized');
    console.log(`   Target: ${this.config.target_p95_latency_ms}ms P95 (${(this.config.target_latency_improvement * 100).toFixed(1)}% improvement)`);
    console.log(`   CBU preservation: ${(this.config.target_cbu_preservation * 100).toFixed(1)}%`);
    console.log(`   Canary deployment: ${this.config.enable_canary_deployment ? '✅' : '❌'}`);
  }
  
  /**
   * Initialize all component systems
   */
  async initializeSystems(embedding_dimension: number = 384): Promise<void> {
    console.log('🔧 Initializing Lagrangian optimization systems...');
    
    try {
      // Initialize mathematical orchestrator
      this.math_orchestrator = new MathematicalOrchestrator(
        embedding_dimension,
        {
          ...this.config.mathematical_orchestrator,
          target_p95_latency_ms: this.config.target_p95_latency_ms,
          track_performance_metrics: true,
        } as Partial<MathematicalOrchestratorConfig>
      );
      
      // Initialize CE Early-Exit system
      this.ce_early_exit = new CEEarlyExitSystem({
        ...this.config.ce_early_exit,
        target_p95_latency_ms: this.config.target_p95_latency_ms,
        enable_performance_monitoring: true,
      } as Partial<CEEarlyExitConfig>);
      
      // Initialize Early-Exit Cross-Encoder Reranker
      this.early_exit_reranker = new EarlyExitCrossEncoderReranker(
        "Xenova/ms-marco-MiniLM-L-6-v2",
        {
          ...this.config.ce_early_exit,
          target_p95_latency_ms: this.config.target_p95_latency_ms,
        } as Partial<CEEarlyExitConfig>
      );
      
      await this.early_exit_reranker.init();
      
      // Start performance monitoring
      this.performance_monitor.startMonitoring();
      
      console.log('✅ All systems initialized successfully');
      this.emit('systems_initialized');
      
    } catch (error) {
      console.error('❌ System initialization failed:', error);
      this.emit('initialization_error', error);
      throw error;
    }
  }
  
  /**
   * Execute comprehensive Lagrangian optimization deployment
   */
  async deployLagrangianOptimization(
    candidates: Candidate[],
    token_budget: number,
    query_context?: string
  ): Promise<{
    optimized_candidates: Candidate[];
    deployment_metrics: DeploymentMetrics;
    early_exit_decision: EarlyExitDecision;
    mathematical_result: MathematicalOptimizationResult;
    rollback_analysis: RollbackAnalysis;
  }> {
    const deployment_start = Date.now();
    console.log(`🎯 Starting Lagrangian optimization deployment: ${candidates.length} candidates, ${token_budget} token budget`);
    
    try {
      // Ensure systems are initialized
      if (!this.math_orchestrator || !this.ce_early_exit || !this.early_exit_reranker) {
        await this.initializeSystems();
      }
      
      // Stage 1: Baseline measurement (if needed)
      if (!this.baseline_metrics) {
        await this.establishBaseline(candidates, token_budget, query_context);
      }
      
      // Stage 2: Execute canary deployment if enabled
      if (this.config.enable_canary_deployment && this.deployment_state.stage === 'initializing') {
        const canary_result = await this.executeCanaryDeployment(candidates, token_budget, query_context);
        
        if (!canary_result.canary_successful) {
          const rollback_analysis = await this.analyzeRollbackDecision('canary_failed');
          return {
            optimized_candidates: candidates, // Return original
            deployment_metrics: this.current_metrics!,
            early_exit_decision: {
              should_exit: true,
              confidence_score: 0.3,
              calibrated_confidence: 0.3,
              exit_reason: 'timeout',
              processing_time_ms: Date.now() - deployment_start,
              candidates_processed: 0,
              quality_estimate: 0.5,
              statistical_validity: {
                ece_slice: 0.2,
                confidence_interval: [0.2, 0.4],
                sample_size: 0,
                p_value: 1.0,
              },
            },
            mathematical_result: await this.createFallbackMathematicalResult(candidates, token_budget),
            rollback_analysis,
          };
        }
      }
      
      // Stage 3: Execute full mathematical optimization
      const mathematical_result = await this.executeOptimizedRetrieval(candidates, token_budget, query_context);
      
      // Stage 4: Execute early-exit reranking
      const reranking_result = await this.executeEarlyExitReranking(
        mathematical_result.selected_candidates,
        query_context || ''
      );
      
      // Stage 5: Update deployment metrics
      const deployment_metrics = await this.updateDeploymentMetrics(
        reranking_result.reranked,
        mathematical_result,
        reranking_result.early_exit_decision
      );
      
      // Stage 6: Analyze rollback necessity
      const rollback_analysis = await this.analyzeRollbackDecision('performance_check');
      
      // Stage 7: Update deployment state
      if (rollback_analysis.should_rollback) {
        await this.executeRollback(rollback_analysis);
        return {
          optimized_candidates: candidates, // Return original after rollback
          deployment_metrics,
          early_exit_decision: reranking_result.early_exit_decision,
          mathematical_result,
          rollback_analysis,
        };
      } else {
        this.advanceDeploymentStage();
      }
      
      const total_time = Date.now() - deployment_start;
      console.log(`🎯 Lagrangian deployment complete: ${total_time}ms total, P95 target ${deployment_metrics.latency_target_achieved ? '✅' : '❌'}`);
      
      return {
        optimized_candidates: reranking_result.reranked,
        deployment_metrics,
        early_exit_decision: reranking_result.early_exit_decision,
        mathematical_result,
        rollback_analysis,
      };
      
    } catch (error) {
      console.error('❌ Lagrangian deployment failed:', error);
      
      // Emergency fallback
      const fallback_analysis: RollbackAnalysis = {
        should_rollback: true,
        trigger_reasons: [`Deployment error: ${error}`],
        confidence_score: 0.95,
        impact_assessment: {
          latency_degradation: 1.0,
          quality_degradation: 0.8,
          system_stability_risk: 0.9,
        },
        recommended_action: 'rollback',
        rollback_strategy: 'immediate',
      };
      
      await this.executeRollback(fallback_analysis);
      
      return {
        optimized_candidates: candidates,
        deployment_metrics: this.current_metrics || await this.createFallbackMetrics(),
        early_exit_decision: {
          should_exit: true,
          confidence_score: 0.1,
          calibrated_confidence: 0.1,
          exit_reason: 'timeout',
          processing_time_ms: Date.now() - deployment_start,
          candidates_processed: 0,
          quality_estimate: 0.3,
          statistical_validity: {
            ece_slice: 0.3,
            confidence_interval: [0.0, 0.2],
            sample_size: 0,
            p_value: 1.0,
          },
        },
        mathematical_result: await this.createFallbackMathematicalResult(candidates, token_budget),
        rollback_analysis: fallback_analysis,
      };
    }
  }
  
  /**
   * Establish baseline metrics for comparison
   */
  private async establishBaseline(
    candidates: Candidate[],
    token_budget: number,
    query_context?: string
  ): Promise<void> {
    console.log('📊 Establishing baseline metrics...');
    
    // Simulate baseline processing (in production, this would use existing system)
    const baseline_start = Date.now();
    
    // Simple baseline: top candidates by score
    const baseline_candidates = candidates
      .slice()
      .sort((a, b) => b.score - a.score)
      .slice(0, Math.min(20, candidates.length));
    
    const baseline_processing_time = Date.now() - baseline_start;
    
    // Record baseline metrics
    this.baseline_latency_samples.push(baseline_processing_time);
    this.baseline_quality_samples.push(this.estimateQuality(baseline_candidates));
    
    this.baseline_metrics = {
      baseline_p95_ms: this.config.baseline_p95_latency_ms,
      current_p95_ms: baseline_processing_time,
      latency_improvement_percent: 0,
      latency_target_achieved: false,
      
      baseline_cbu_ratio: 1.0, // Normalized baseline
      current_cbu_ratio: 1.0,
      cbu_preservation_ratio: 1.0,
      cbu_target_achieved: true,
      
      overall_health_score: 100,
      component_health_scores: {
        lagrangian_optimization: 100,
        ce_early_exit: 100,
        cbu_preservation: 100,
        latency_performance: 100,
        calibration_quality: 100,
        cache_efficiency: 100,
      },
      critical_alerts_count: 0,
      
      statistical_significance_achieved: false,
      p_value: 1.0,
      confidence_interval: [0.9, 1.1],
      effect_size: 0.0,
      
      deployment_state: this.deployment_state,
      rollout_progress_percent: 0,
      estimated_completion_time: Date.now() + (this.config.rollout_stages.length * this.config.stage_duration_minutes * 60000),
    };
    
    console.log('✅ Baseline established');
    this.emit('baseline_established', this.baseline_metrics);
  }
  
  /**
   * Execute canary deployment
   */
  private async executeCanaryDeployment(
    candidates: Candidate[],
    token_budget: number,
    query_context?: string
  ): Promise<CanaryAnalysisResult> {
    console.log(`🐤 Executing canary deployment: ${this.config.canary_traffic_percent}% traffic`);
    
    this.deployment_state.stage = 'canary';
    this.deployment_state.current_stage_percent = this.config.canary_traffic_percent;
    this.deployment_state.stage_start_time = Date.now();
    
    // Execute optimized processing on canary traffic
    const canary_samples = Math.min(10, candidates.length); // Small canary sample
    const canary_candidates = candidates.slice(0, canary_samples);
    
    const canary_result = await this.executeOptimizedRetrieval(canary_candidates, token_budget, query_context);
    const reranking_result = await this.executeEarlyExitReranking(
      canary_result.selected_candidates,
      query_context || ''
    );
    
    // Analyze canary results
    const latency_improvement = this.baseline_metrics ? 
      (this.baseline_metrics.current_p95_ms - reranking_result.early_exit_decision.processing_time_ms) / this.baseline_metrics.current_p95_ms : 0;
    
    const quality_estimate = this.estimateQuality(reranking_result.reranked);
    const quality_preservation = this.baseline_metrics ? 
      quality_estimate / (this.baseline_quality_samples[0] || 1) : 1;
    
    const stability_score = canary_result.performance_target_met ? 0.9 : 0.4;
    
    const canary_successful = 
      latency_improvement >= this.config.target_latency_improvement * 0.8 && // 80% of target
      quality_preservation >= 0.95 && // Maintain 95% quality
      stability_score >= 0.8; // High stability
    
    const result: CanaryAnalysisResult = {
      canary_successful,
      metrics_comparison: {
        latency_improvement,
        quality_preservation,
        stability_score,
      },
      statistical_significance: {
        latency_p_value: 0.05, // Simulated
        quality_p_value: 0.1,
        effect_size: latency_improvement,
      },
      recommendation: canary_successful ? 'proceed' : 'rollback',
      next_stage_percent: canary_successful ? this.config.rollout_stages[1] : 0,
    };
    
    console.log(`🐤 Canary analysis: ${canary_successful ? '✅ SUCCESS' : '❌ FAILED'}`);
    console.log(`   Latency improvement: ${(latency_improvement * 100).toFixed(1)}%`);
    console.log(`   Quality preservation: ${(quality_preservation * 100).toFixed(1)}%`);
    
    this.emit('canary_completed', result);
    return result;
  }
  
  /**
   * Execute optimized mathematical retrieval
   */
  private async executeOptimizedRetrieval(
    candidates: Candidate[],
    token_budget: number,
    query_context?: string
  ): Promise<MathematicalOptimizationResult> {
    if (!this.math_orchestrator) {
      throw new Error('Mathematical orchestrator not initialized');
    }
    
    // Convert candidates to mathematical format
    const math_candidates: MathematicalCandidate[] = candidates.map(candidate => ({
      ...candidate,
      delta_u: candidate.score * 0.8,
      coverage_gain: candidate.score * 0.3,
      embedding: undefined, // Will be generated if needed
      chunk_type_detailed: candidate.kind || 'text',
      timestamp: Date.now(),
    }));
    
    // Execute mathematical optimization
    const result = await this.math_orchestrator.optimizeSelection(
      math_candidates,
      token_budget,
      query_context
    );
    
    // Record performance metrics
    this.performance_monitor.recordLagrangianMetrics(
      result.final_lambda,
      result.lagrangian_objective,
      result.lagrangian_objective - result.dual_gap,
      result.bisection_iterations,
      result.total_processing_time_ms
    );
    
    return result;
  }
  
  /**
   * Execute early-exit reranking
   */
  private async executeEarlyExitReranking(
    candidates: Candidate[],
    query: string
  ): Promise<{
    reranked: Candidate[];
    early_exit_decision: EarlyExitDecision;
    matryoshka_routing: MatryoshkaRouting;
  }> {
    if (!this.early_exit_reranker) {
      throw new Error('Early-exit reranker not initialized');
    }
    
    const reranking_result = await this.early_exit_reranker.rerank(query, candidates);
    
    // Record performance metrics
    this.latency_samples.push(reranking_result.early_exit_decision.processing_time_ms);
    this.quality_samples.push(reranking_result.early_exit_decision.quality_estimate);
    
    return {
      reranked: reranking_result.reranked,
      early_exit_decision: reranking_result.early_exit_decision,
      matryoshka_routing: reranking_result.matryoshka_routing,
    };
  }
  
  /**
   * Update deployment metrics
   */
  private async updateDeploymentMetrics(
    optimized_candidates: Candidate[],
    mathematical_result: MathematicalOptimizationResult,
    early_exit_decision: EarlyExitDecision
  ): Promise<DeploymentMetrics> {
    // Get performance snapshot
    const performance_snapshot = this.performance_monitor.getCurrentStatus();
    const latest_metrics = performance_snapshot.latest_metrics;
    
    // Compute latency metrics
    const current_p95 = this.computeP95(this.latency_samples);
    const latency_improvement = this.baseline_metrics ? 
      (this.baseline_metrics.baseline_p95_ms - current_p95) / this.baseline_metrics.baseline_p95_ms : 0;
    
    // Compute quality metrics  
    const current_quality = this.estimateQuality(optimized_candidates);
    const baseline_quality = this.baseline_quality_samples.length > 0 ? 
      this.baseline_quality_samples[0] : 1.0;
    const cbu_preservation = current_quality / baseline_quality;
    
    // Statistical validation
    const statistical_result = this.performStatisticalValidation();
    
    this.current_metrics = {
      baseline_p95_ms: this.baseline_metrics?.baseline_p95_ms || this.config.baseline_p95_latency_ms,
      current_p95_ms: current_p95,
      latency_improvement_percent: latency_improvement,
      latency_target_achieved: latency_improvement >= this.config.target_latency_improvement,
      
      baseline_cbu_ratio: 1.0,
      current_cbu_ratio: cbu_preservation,
      cbu_preservation_ratio: cbu_preservation,
      cbu_target_achieved: cbu_preservation >= this.config.min_cbu_threshold,
      
      overall_health_score: latest_metrics?.system_health.overall_health_score || 85,
      component_health_scores: latest_metrics?.system_health.component_health || {},
      critical_alerts_count: performance_snapshot.recent_alerts.filter(a => a.severity === 'critical').length,
      
      statistical_significance_achieved: statistical_result.significant,
      p_value: statistical_result.p_value,
      confidence_interval: statistical_result.confidence_interval,
      effect_size: latency_improvement,
      
      deployment_state: this.deployment_state,
      rollout_progress_percent: this.deployment_state.current_stage_percent,
      estimated_completion_time: Date.now() + (60000 * this.config.stage_duration_minutes),
    };
    
    this.deployment_state.total_samples_processed += optimized_candidates.length;
    this.deployment_state.successful_samples += mathematical_result.performance_target_met ? optimized_candidates.length : 0;
    this.deployment_state.health_score = this.current_metrics.overall_health_score;
    
    return this.current_metrics;
  }
  
  /**
   * Analyze rollback decision
   */
  private async analyzeRollbackDecision(trigger: string): Promise<RollbackAnalysis> {
    const reasons: string[] = [];
    let should_rollback = false;
    let confidence_score = 0.8;
    
    // Check performance degradation
    if (this.current_metrics) {
      if (!this.current_metrics.latency_target_achieved) {
        reasons.push('Latency target not achieved');
        should_rollback = true;
      }
      
      if (!this.current_metrics.cbu_target_achieved) {
        reasons.push('CBU quality target not achieved');
        should_rollback = true;
      }
      
      if (this.current_metrics.overall_health_score < this.config.rollback_trigger_threshold * 100) {
        reasons.push('System health score below threshold');
        should_rollback = true;
      }
      
      if (this.current_metrics.critical_alerts_count >= 3) {
        reasons.push('Too many critical alerts');
        should_rollback = true;
      }
    }
    
    // Check rollback history
    if (this.rollback_history.length >= this.config.max_rollback_attempts) {
      reasons.push('Maximum rollback attempts reached');
      should_rollback = true;
      confidence_score = 0.95;
    }
    
    const latency_degradation = this.current_metrics ? 
      Math.max(0, this.current_metrics.current_p95_ms - this.config.target_p95_latency_ms) / this.config.target_p95_latency_ms : 0;
    
    const quality_degradation = this.current_metrics ? 
      Math.max(0, 1 - this.current_metrics.cbu_preservation_ratio) : 0;
    
    const stability_risk = this.current_metrics ? 
      Math.max(0, 1 - this.current_metrics.overall_health_score / 100) : 0;
    
    return {
      should_rollback,
      trigger_reasons: reasons,
      confidence_score,
      impact_assessment: {
        latency_degradation,
        quality_degradation,
        system_stability_risk: stability_risk,
      },
      recommended_action: should_rollback ? 'rollback' : 
        (reasons.length > 0 ? 'investigate' : 'continue'),
      rollback_strategy: should_rollback ? 'immediate' : 'gradual',
    };
  }
  
  /**
   * Execute rollback
   */
  private async executeRollback(analysis: RollbackAnalysis): Promise<void> {
    console.log(`🔄 Executing rollback: ${analysis.trigger_reasons.join(', ')}`);
    
    this.deployment_state.stage = 'rolling_back';
    this.rollback_history.push({
      timestamp: Date.now(),
      reason: analysis.trigger_reasons.join(', '),
      stage: this.deployment_state.current_stage_percent,
    });
    this.deployment_state.rollback_count++;
    
    // Reset systems to baseline
    this.math_orchestrator?.reset();
    this.ce_early_exit?.reset();
    this.early_exit_reranker?.reset();
    
    // Clear performance data
    this.latency_samples = [...this.baseline_latency_samples];
    this.quality_samples = [...this.baseline_quality_samples];
    
    console.log('✅ Rollback completed');
    this.emit('rollback_completed', analysis);
    
    this.deployment_state.stage = 'failed';
  }
  
  /**
   * Advance deployment to next stage
   */
  private advanceDeploymentStage(): void {
    const current_index = this.config.rollout_stages.indexOf(this.deployment_state.current_stage_percent);
    
    if (current_index >= 0 && current_index < this.config.rollout_stages.length - 1) {
      this.deployment_state.current_stage_percent = this.config.rollout_stages[current_index + 1];
      this.deployment_state.stage_start_time = Date.now();
      console.log(`📈 Advanced to deployment stage: ${this.deployment_state.current_stage_percent}%`);
      this.emit('stage_advanced', this.deployment_state.current_stage_percent);
    } else {
      this.deployment_state.stage = 'complete';
      console.log('🎉 Deployment completed successfully');
      this.emit('deployment_completed');
    }
  }
  
  /**
   * Setup event handlers
   */
  private setupEventHandlers(): void {
    // Performance monitor alerts
    this.performance_monitor.on('alert', (alert: PerformanceAlert) => {
      console.log(`🚨 Performance alert: ${alert.category} - ${alert.message}`);
      
      if (alert.severity === 'critical' || alert.auto_rollback_triggered) {
        this.emit('critical_alert', alert);
      }
    });
    
    // Rollback triggers
    this.performance_monitor.on('rollback_triggered', async (reason: string, value: number) => {
      console.log(`⚠️ Automated rollback triggered: ${reason} (${value})`);
      const analysis = await this.analyzeRollbackDecision(`automated_${reason}`);
      if (analysis.should_rollback) {
        await this.executeRollback(analysis);
      }
    });
  }
  
  /**
   * Utility methods
   */
  
  private computeP95(values: number[]): number {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.floor(sorted.length * 0.95);
    return sorted[index];
  }
  
  private estimateQuality(candidates: Candidate[]): number {
    if (candidates.length === 0) return 0;
    const avg_score = candidates.reduce((sum, c) => sum + c.score, 0) / candidates.length;
    const top_score = Math.max(...candidates.map(c => c.score));
    return (avg_score * 0.6 + top_score * 0.4);
  }
  
  private performStatisticalValidation(): {
    significant: boolean;
    p_value: number;
    confidence_interval: [number, number];
  } {
    // Simplified statistical validation
    if (this.latency_samples.length < this.config.minimum_sample_size) {
      return {
        significant: false,
        p_value: 1.0,
        confidence_interval: [0, 1],
      };
    }
    
    // Compute improvement effect
    const current_mean = this.latency_samples.reduce((a, b) => a + b, 0) / this.latency_samples.length;
    const baseline_mean = this.baseline_latency_samples.length > 0 ? 
      this.baseline_latency_samples.reduce((a, b) => a + b, 0) / this.baseline_latency_samples.length : 
      this.config.baseline_p95_latency_ms;
    
    const improvement = (baseline_mean - current_mean) / baseline_mean;
    const effect_size = Math.abs(improvement);
    
    // Simple significance test (would use proper t-test in production)
    const p_value = effect_size > 0.1 ? 0.01 : 0.5;
    const significant = p_value < this.config.significance_level && effect_size >= this.config.target_latency_improvement * 0.8;
    
    return {
      significant,
      p_value,
      confidence_interval: [improvement - 0.1, improvement + 0.1],
    };
  }
  
  private async createFallbackMathematicalResult(candidates: Candidate[], token_budget: number): Promise<MathematicalOptimizationResult> {
    return {
      selected_candidates: candidates.slice(0, Math.min(10, candidates.length)),
      final_lambda: 0.1,
      lagrangian_objective: 1.0,
      dual_gap: 0.05,
      bisection_iterations: 0,
      lambda_warm_started: false,
      diversity_score: 0.5,
      orthogonal_mass: 0,
      dpp_rank_utilized: 0,
      causal_groups_count: candidates.length,
      average_group_size: 1,
      constraint_violations: 0,
      ilp_escalation_required: false,
      voi_ece: 0.15,
      voi_calibrated: false,
      ips_effective_sample_size: 0,
      total_processing_time_ms: 50,
      component_timings: {
        lagrangian_ms: 20,
        dpp_ms: 10,
        causal_ms: 5,
        voi_ms: 5,
        rust_ms: 10,
        lambda_control_ms: 0,
        dpp_diagnostics_ms: 0,
        group_split_ms: 0,
        tradeoff_analysis_ms: 0,
      },
      performance_target_met: false,
      mathematical_validation_passed: false,
      total_tokens: Math.min(token_budget, candidates.reduce((sum, c) => 
        sum + Math.ceil((c.text?.length || 0) / 4), 0)),
      budget_utilization: 0.8,
      optimization_health: {
        overall_score: 40,
        lambda_stability: 'needs_attention',
        diversity_quality: 'needs_attention', 
        performance_efficiency: 'needs_attention',
        recommendation_confidence: 0.3,
      },
    };
  }
  
  private async createFallbackMetrics(): Promise<DeploymentMetrics> {
    return {
      baseline_p95_ms: this.config.baseline_p95_latency_ms,
      current_p95_ms: this.config.baseline_p95_latency_ms,
      latency_improvement_percent: 0,
      latency_target_achieved: false,
      baseline_cbu_ratio: 1.0,
      current_cbu_ratio: 0.8,
      cbu_preservation_ratio: 0.8,
      cbu_target_achieved: false,
      overall_health_score: 40,
      component_health_scores: {},
      critical_alerts_count: 5,
      statistical_significance_achieved: false,
      p_value: 1.0,
      confidence_interval: [0, 0],
      effect_size: 0,
      deployment_state: this.deployment_state,
      rollout_progress_percent: 0,
      estimated_completion_time: Date.now(),
    };
  }
  
  /**
   * Public API methods
   */
  
  /**
   * Get current deployment status
   */
  getCurrentDeploymentStatus(): {
    deployment_state: DeploymentState;
    current_metrics?: DeploymentMetrics;
    rollback_history: typeof this.rollback_history;
    performance_status: ReturnType<PerformanceMonitor['getCurrentStatus']>;
  } {
    return {
      deployment_state: this.deployment_state,
      current_metrics: this.current_metrics,
      rollback_history: this.rollback_history,
      performance_status: this.performance_monitor.getCurrentStatus(),
    };
  }
  
  /**
   * Force rollback (emergency use)
   */
  async forceRollback(reason: string): Promise<void> {
    const analysis: RollbackAnalysis = {
      should_rollback: true,
      trigger_reasons: [`Manual rollback: ${reason}`],
      confidence_score: 1.0,
      impact_assessment: {
        latency_degradation: 0,
        quality_degradation: 0,
        system_stability_risk: 0,
      },
      recommended_action: 'rollback',
      rollback_strategy: 'immediate',
    };
    
    await this.executeRollback(analysis);
  }
  
  /**
   * Shutdown orchestrator
   */
  shutdown(): void {
    this.performance_monitor.stopMonitoring();
    this.math_orchestrator?.reset();
    this.ce_early_exit?.reset();
    this.early_exit_reranker?.reset();
    
    console.log('🛑 Lagrangian Deployment Orchestrator shutdown');
    this.emit('shutdown');
  }
}

// Default configuration optimized for P95 latency deployment
export const DEFAULT_LAGRANGIAN_DEPLOYMENT_CONFIG: LagrangianDeploymentConfig = 
  LagrangianDeploymentConfigSchema.parse({
    target_p95_latency_ms: 1.0,
    baseline_p95_latency_ms: 6.8,
    target_latency_improvement: 0.85, // 85% reduction
    target_cbu_preservation: 1.125, // +12.5% CBU
    enable_canary_deployment: true,
    enable_automated_rollback: true,
    enable_statistical_validation: true,
  });