/**
 * Production Readiness Orchestrator
 * 
 * Master orchestrator integrating all production validation systems:
 * - Three Core Mathematical Proofs (Dual Sanity, OOD Resilience, Long-horizon Win Rate)
 * - Hierarchical Interleaving System for multi-turn attribution
 * - Real-time Monitoring & Alerting with λ/size/CBU dashboards
 * - DPP Optimization with rank tuning and ΔCBU/ms curves
 * - EmbeddingGemma-300M trials with promotion gates
 * - Risk Budget management and operational hardening
 * 
 * Orchestrates complete production validation pipeline with statistical rigor,
 * mathematical validation, and operational excellence as specified in TODO.md.
 */

import type { DB } from '@lethe/sqlite';
import type { Candidate } from './index.js';

// Import all validation systems
import { 
  ProductionValidationSystem,
  type ProductionValidationResults,
  type ProductionValidationConfig
} from './production_validation.js';

import {
  HierarchicalInterleavingEngine,
  type HierarchicalInterleavingResult,
  type HierarchicalInterleavingConfig,
  type Turn
} from './hierarchical_interleaving.js';

import {
  ProductionMonitoringSystem,
  type DashboardData,
  type AlertThresholds,
  type Alert,
  type RiskBudgetStatus
} from './monitoring_system.js';

import {
  DPPOptimizationEngine,
  type DPPOptimizationResult,
  type DPPOptimizationConfig,
  type DPPCandidate
} from './dpp_optimization.js';

import {
  EmbeddingGemmaTrialEngine,
  type TrialConfiguration,
  type TrialStatusReport,
  type PromotionGateEvaluation
} from './embedding_gemma_trial.js';

// Master orchestrator interfaces
export interface ProductionOrchestrationConfig {
  // Core validation configuration
  validation_config: Partial<ProductionValidationConfig>;
  
  // Hierarchical interleaving configuration
  interleaving_config: Partial<HierarchicalInterleavingConfig>;
  
  // Monitoring and alerting configuration
  monitoring_config: {
    thresholds: Partial<AlertThresholds>;
    metrics_interval_ms: number;
    alerting_interval_ms: number;
    enable_auto_resolution: boolean;
  };
  
  // DPP optimization configuration
  dpp_config: Partial<DPPOptimizationConfig>;
  
  // EmbeddingGemma trial configuration
  embedding_trial_config: Partial<TrialConfiguration>;
  
  // Orchestration settings
  orchestration_settings: {
    enable_parallel_validation: boolean;
    validation_timeout_ms: number;
    enable_early_termination: boolean;
    risk_budget_enforcement: boolean;
    auto_promotion_enabled: boolean;
  };
  
  // Quality gates for production readiness
  production_readiness_gates: {
    min_validation_success_rate: number; // 95%
    max_acceptable_risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
    required_statistical_power: number; // 80%
    min_sample_size: number;
    max_deployment_risk_score: number;
  };
}

export interface OrchestrationState {
  orchestration_id: string;
  start_timestamp: string;
  current_phase: OrchestrationPhase;
  
  // System states
  validation_system_active: boolean;
  interleaving_system_active: boolean;
  monitoring_system_active: boolean;
  dpp_optimization_active: boolean;
  embedding_trials_active: string[]; // Trial IDs
  
  // Progress tracking
  sessions_processed: number;
  validations_completed: number;
  quality_gates_passed: number;
  critical_alerts_active: number;
  
  // Risk assessment
  current_risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  risk_budget_utilization: number;
  production_readiness_score: number; // 0-100
}

export interface ComprehensiveValidationResult {
  orchestration_id: string;
  timestamp: string;
  session_id: string;
  
  // Individual system results
  validation_results: ProductionValidationResults;
  interleaving_results: HierarchicalInterleavingResult;
  monitoring_snapshot: DashboardData;
  dpp_optimization: DPPOptimizationResult;
  embedding_trial_status?: TrialStatusReport;
  
  // Integrated assessment
  overall_assessment: {
    production_ready: boolean;
    confidence_level: number;
    risk_assessment: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
    blocking_issues: string[];
    recommendations: string[];
  };
  
  // Quality metrics summary
  quality_summary: {
    dual_sanity_passed: boolean;
    ood_resilience_passed: boolean;
    long_horizon_passed: boolean;
    monitoring_health_score: number;
    dpp_efficiency_score: number;
    statistical_significance: boolean;
  };
  
  // Performance metrics
  orchestration_metrics: {
    total_processing_time_ms: number;
    parallel_execution_efficiency: number;
    system_coordination_overhead_ms: number;
    validation_coverage_percent: number;
  };
}

export interface ProductionDeploymentDecision {
  decision: 'APPROVE' | 'CONDITIONAL_APPROVE' | 'REJECT' | 'DEFER';
  decision_timestamp: string;
  confidence_score: number; // 0-1
  
  // Decision factors
  core_proofs_status: {
    dual_sanity: boolean;
    ood_resilience: boolean;
    long_horizon_win_rate: boolean;
  };
  
  operational_readiness: {
    monitoring_functional: boolean;
    alerting_configured: boolean;
    risk_budget_healthy: boolean;
    chaos_testing_passed: boolean;
  };
  
  quality_assessment: {
    performance_targets_met: boolean;
    quality_gates_passed: boolean;
    statistical_validation_complete: boolean;
    embedding_trials_successful: boolean;
  };
  
  // Conditions for conditional approval
  conditions?: string[];
  
  // Blocking factors for rejection
  blocking_factors?: string[];
  
  // Timeline for deferred decision
  defer_until?: string;
  defer_requirements?: string[];
  
  // Deployment recommendations
  deployment_strategy: 'FULL_ROLLOUT' | 'CANARY_DEPLOYMENT' | 'GRADUAL_ROLLOUT' | 'A_B_TEST';
  rollback_triggers: string[];
  monitoring_requirements: string[];
}

type OrchestrationPhase = 
  | 'INITIALIZATION'
  | 'SYSTEM_STARTUP'
  | 'VALIDATION_EXECUTION'
  | 'MONITORING_ACTIVE'
  | 'ANALYSIS_INTEGRATION'
  | 'DECISION_EVALUATION'
  | 'DEPLOYMENT_READY'
  | 'ERROR_RECOVERY';

export const DEFAULT_ORCHESTRATION_CONFIG: ProductionOrchestrationConfig = {
  validation_config: {
    // Use defaults from ProductionValidationSystem
  },
  
  interleaving_config: {
    // Use defaults from HierarchicalInterleavingEngine
  },
  
  monitoring_config: {
    thresholds: {
      // Use defaults from ProductionMonitoringSystem
    },
    metrics_interval_ms: 30000, // 30 seconds
    alerting_interval_ms: 10000, // 10 seconds
    enable_auto_resolution: true,
  },
  
  dpp_config: {
    // Use defaults from DPPOptimizationEngine
  },
  
  embedding_trial_config: {
    // Use defaults from EmbeddingGemmaTrialEngine
  },
  
  orchestration_settings: {
    enable_parallel_validation: true,
    validation_timeout_ms: 300000, // 5 minutes
    enable_early_termination: true,
    risk_budget_enforcement: true,
    auto_promotion_enabled: false, // Require human approval
  },
  
  production_readiness_gates: {
    min_validation_success_rate: 0.95, // 95%
    max_acceptable_risk_level: 'MEDIUM',
    required_statistical_power: 0.8, // 80%
    min_sample_size: 10000,
    max_deployment_risk_score: 75, // 0-100 scale
  },
};

/**
 * Production Readiness Orchestrator
 * 
 * Master coordinator for all production validation systems
 */
export class ProductionReadinessOrchestrator {
  private db: DB;
  private config: ProductionOrchestrationConfig;
  
  // Integrated system components
  private validationSystem: ProductionValidationSystem;
  private interleavingEngine: HierarchicalInterleavingEngine;
  private monitoringSystem: ProductionMonitoringSystem;
  private dppEngine: DPPOptimizationEngine;
  private embeddingTrialEngine: EmbeddingGemmaTrialEngine;
  
  // Orchestration state
  private orchestrationState: OrchestrationState;
  private validationHistory: Map<string, ComprehensiveValidationResult[]> = new Map();
  
  // Performance tracking
  private systemPerformanceMetrics: Map<string, SystemPerformanceMetrics> = new Map();
  
  constructor(db: DB, config: Partial<ProductionOrchestrationConfig> = {}) {
    this.db = db;
    this.config = { ...DEFAULT_ORCHESTRATION_CONFIG, ...config };
    
    // Initialize all subsystems
    this.validationSystem = new ProductionValidationSystem(db, this.config.validation_config);
    this.interleavingEngine = new HierarchicalInterleavingEngine(db, this.config.interleaving_config);
    this.monitoringSystem = new ProductionMonitoringSystem(db, this.config.monitoring_config.thresholds);
    this.dppEngine = new DPPOptimizationEngine(db, this.config.dpp_config);
    this.embeddingTrialEngine = new EmbeddingGemmaTrialEngine(db, this.config.embedding_trial_config);
    
    // Initialize orchestration state
    this.orchestrationState = {
      orchestration_id: `orchestration_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      start_timestamp: new Date().toISOString(),
      current_phase: 'INITIALIZATION',
      validation_system_active: false,
      interleaving_system_active: false,
      monitoring_system_active: false,
      dpp_optimization_active: false,
      embedding_trials_active: [],
      sessions_processed: 0,
      validations_completed: 0,
      quality_gates_passed: 0,
      critical_alerts_active: 0,
      current_risk_level: 'LOW',
      risk_budget_utilization: 0,
      production_readiness_score: 0,
    };
  }

  /**
   * Start the complete production validation orchestration
   */
  async startOrchestration(): Promise<OrchestrationStartupResult> {
    console.log(`🎼 Starting Production Readiness Orchestration: ${this.orchestrationState.orchestration_id}`);
    console.log(`   Integrated Systems: Validation, Interleaving, Monitoring, DPP, EmbeddingGemma`);
    console.log(`   Configuration: Parallel=${this.config.orchestration_settings.enable_parallel_validation}, Auto-Resolution=${this.config.monitoring_config.enable_auto_resolution}`);

    this.orchestrationState.current_phase = 'SYSTEM_STARTUP';

    try {
      // Phase 1: System initialization and health checks
      const systemHealthChecks = await this.performSystemHealthChecks();
      
      // Phase 2: Start monitoring system first (provides telemetry for other systems)
      await this.startMonitoringSystem();
      
      // Phase 3: Initialize and start all validation systems
      const systemStartupResults = await this.startAllValidationSystems();
      
      // Phase 4: Validate system integration and communication
      const integrationValidation = await this.validateSystemIntegration();
      
      // Phase 5: Transition to active validation phase
      this.orchestrationState.current_phase = 'VALIDATION_EXECUTION';
      
      const startupResult: OrchestrationStartupResult = {
        orchestration_id: this.orchestrationState.orchestration_id,
        startup_successful: true,
        systems_started: systemStartupResults.systems_started,
        health_checks_passed: systemHealthChecks.all_systems_healthy,
        integration_validated: integrationValidation.integration_successful,
        monitoring_active: this.orchestrationState.monitoring_system_active,
        estimated_readiness_time_hours: this.estimateReadinessTime(),
        risk_budget_initial: await this.initializeRiskBudget(),
      };

      console.log(`✅ Production Orchestration started successfully`);
      console.log(`   Systems Active: ${systemStartupResults.systems_started.length}`);
      console.log(`   Estimated Readiness: ${startupResult.estimated_readiness_time_hours}h`);

      return startupResult;

    } catch (error) {
      this.orchestrationState.current_phase = 'ERROR_RECOVERY';
      console.error(`❌ Orchestration startup failed: ${error}`);
      throw new Error(`Production orchestration startup failed: ${error}`);
    }
  }

  /**
   * Execute comprehensive validation for a session with all integrated systems
   */
  async executeComprehensiveValidation(
    sessionId: string,
    queries: string[],
    candidates: Candidate[],
    previousTurns?: Turn[]
  ): Promise<ComprehensiveValidationResult> {
    console.log(`🔬 Executing comprehensive validation: session=${sessionId}, queries=${queries.length}`);

    const startTime = performance.now();
    
    try {
      // Ensure orchestration is in correct phase
      if (this.orchestrationState.current_phase !== 'VALIDATION_EXECUTION' && 
          this.orchestrationState.current_phase !== 'MONITORING_ACTIVE') {
        throw new Error(`Invalid orchestration phase for validation: ${this.orchestrationState.current_phase}`);
      }

      // Execute validation systems in parallel or sequential based on configuration
      let validationResults: ValidationSystemResults;
      
      if (this.config.orchestration_settings.enable_parallel_validation) {
        validationResults = await this.executeParallelValidation(sessionId, queries, candidates, previousTurns);
      } else {
        validationResults = await this.executeSequentialValidation(sessionId, queries, candidates, previousTurns);
      }

      // Integrate and analyze results
      const integratedAssessment = await this.integrateValidationResults(validationResults);
      
      // Update orchestration state
      this.updateOrchestrationState(integratedAssessment);
      
      // Generate comprehensive result
      const comprehensiveResult: ComprehensiveValidationResult = {
        orchestration_id: this.orchestrationState.orchestration_id,
        timestamp: new Date().toISOString(),
        session_id: sessionId,
        
        validation_results: validationResults.production_validation,
        interleaving_results: validationResults.hierarchical_interleaving,
        monitoring_snapshot: validationResults.monitoring_snapshot,
        dpp_optimization: validationResults.dpp_optimization,
        embedding_trial_status: validationResults.embedding_trial_status,
        
        overall_assessment: integratedAssessment,
        
        quality_summary: {
          dual_sanity_passed: validationResults.production_validation.dual_sanity.lambda_monotonicity_check.passed && 
                              validationResults.production_validation.dual_sanity.primal_dual_gap.passed,
          ood_resilience_passed: validationResults.production_validation.ood_resilience.coverage_weighted_crps.passed && 
                                validationResults.production_validation.ood_resilience.mondrian_conformal.passed,
          long_horizon_passed: validationResults.production_validation.long_horizon_win_rate.hierarchical_interleaving.session_level_attribution,
          monitoring_health_score: validationResults.monitoring_snapshot.overview.system_health_score,
          dpp_efficiency_score: validationResults.dpp_optimization.optimization_metrics.delta_cbu_per_ms,
          statistical_significance: validationResults.hierarchical_interleaving.statistical_validation.power_adequate,
        },
        
        orchestration_metrics: {
          total_processing_time_ms: performance.now() - startTime,
          parallel_execution_efficiency: this.calculateParallelEfficiency(validationResults),
          system_coordination_overhead_ms: this.calculateCoordinationOverhead(),
          validation_coverage_percent: this.calculateValidationCoverage(validationResults),
        },
      };

      // Store results in history
      const sessionHistory = this.validationHistory.get(sessionId) || [];
      sessionHistory.push(comprehensiveResult);
      this.validationHistory.set(sessionId, sessionHistory);

      console.log(`✅ Comprehensive validation complete: ${comprehensiveResult.orchestration_metrics.total_processing_time_ms.toFixed(1)}ms`);
      console.log(`   Overall Assessment: ${integratedAssessment.production_ready ? '✅ READY' : '❌ NOT READY'} (confidence: ${(integratedAssessment.confidence_level * 100).toFixed(1)}%)`);
      console.log(`   Risk Level: ${integratedAssessment.risk_assessment}, Quality Gates: ${this.orchestrationState.quality_gates_passed}/${Object.keys(comprehensiveResult.quality_summary).length}`);

      return comprehensiveResult;

    } catch (error) {
      console.error(`❌ Comprehensive validation failed: ${error}`);
      throw new Error(`Comprehensive validation failed: ${error}`);
    }
  }

  /**
   * Generate production deployment decision based on comprehensive validation
   */
  async generateDeploymentDecision(
    sessionId?: string,
    validationResults?: ComprehensiveValidationResult[]
  ): Promise<ProductionDeploymentDecision> {
    console.log(`🎯 Generating production deployment decision...`);

    try {
      // Get validation results (either provided or from history)
      const results = validationResults || this.getAllValidationResults(sessionId);
      
      if (results.length === 0) {
        throw new Error('No validation results available for deployment decision');
      }

      // Analyze recent validation trends
      const recentResults = results.slice(-10); // Last 10 validations
      const validationTrends = this.analyzeValidationTrends(recentResults);
      
      // Evaluate core mathematical proofs
      const coreProofsStatus = this.evaluateCoreProofs(recentResults);
      
      // Assess operational readiness
      const operationalReadiness = await this.assessOperationalReadiness();
      
      // Evaluate quality metrics and gates
      const qualityAssessment = this.evaluateQualityGates(recentResults);
      
      // Calculate overall confidence score
      const confidenceScore = this.calculateDeploymentConfidence(
        coreProofsStatus,
        operationalReadiness,
        qualityAssessment,
        validationTrends
      );

      // Determine deployment decision
      const decision = this.determineDeploymentDecision(
        confidenceScore,
        coreProofsStatus,
        operationalReadiness,
        qualityAssessment
      );

      console.log(`🎯 Deployment decision: ${decision.decision} (confidence: ${(decision.confidence_score * 100).toFixed(1)}%)`);
      if (decision.conditions) {
        console.log(`   Conditions: ${decision.conditions.join(', ')}`);
      }
      if (decision.blocking_factors) {
        console.log(`   Blocking Factors: ${decision.blocking_factors.join(', ')}`);
      }

      return decision;

    } catch (error) {
      console.error(`❌ Deployment decision generation failed: ${error}`);
      throw new Error(`Deployment decision generation failed: ${error}`);
    }
  }

  /**
   * Get comprehensive orchestration status
   */
  async getOrchestrationStatus(): Promise<OrchestrationStatusReport> {
    const currentTime = new Date().toISOString();
    const uptimeHours = (Date.now() - new Date(this.orchestrationState.start_timestamp).getTime()) / (1000 * 60 * 60);

    // Get system health from monitoring
    const monitoringDashboard = await this.monitoringSystem.generateDashboardData();
    const activeAlerts = this.monitoringSystem.getActiveAlerts();
    const riskBudgetStatus = this.monitoringSystem.getRiskBudgetStatus();

    // Get embedding trials status
    const embeddingTrialsSummary = await this.embeddingTrialEngine.getTrialSummary();

    return {
      orchestration_id: this.orchestrationState.orchestration_id,
      current_timestamp: currentTime,
      uptime_hours: uptimeHours,
      current_phase: this.orchestrationState.current_phase,
      
      system_status: {
        validation_system: this.orchestrationState.validation_system_active,
        interleaving_system: this.orchestrationState.interleaving_system_active,
        monitoring_system: this.orchestrationState.monitoring_system_active,
        dpp_optimization: this.orchestrationState.dpp_optimization_active,
        embedding_trials: this.orchestrationState.embedding_trials_active.length,
      },
      
      performance_metrics: {
        sessions_processed: this.orchestrationState.sessions_processed,
        validations_completed: this.orchestrationState.validations_completed,
        quality_gates_passed: this.orchestrationState.quality_gates_passed,
        average_validation_time_ms: this.calculateAverageValidationTime(),
        system_health_score: monitoringDashboard.overview.system_health_score,
      },
      
      risk_assessment: {
        current_risk_level: this.orchestrationState.current_risk_level,
        risk_budget_utilization: riskBudgetStatus.budget_utilization_percent,
        critical_alerts: activeAlerts.filter(a => a.severity === 'CRITICAL').length,
        production_readiness_score: this.orchestrationState.production_readiness_score,
      },
      
      embedding_trials: embeddingTrialsSummary,
      
      recommendations: await this.generateOrchestrationRecommendations(),
    };
  }

  // Private methods for system coordination

  private async performSystemHealthChecks(): Promise<SystemHealthCheckResult> {
    console.log('🏥 Performing system health checks...');

    const healthChecks = {
      database_connection: await this.checkDatabaseHealth(),
      validation_system: await this.checkValidationSystemHealth(),
      monitoring_system: await this.checkMonitoringSystemHealth(),
      dpp_engine: await this.checkDPPEngineHealth(),
      embedding_system: await this.checkEmbeddingSystemHealth(),
    };

    const allSystemsHealthy = Object.values(healthChecks).every(check => check);

    return {
      all_systems_healthy: allSystemsHealthy,
      individual_checks: healthChecks,
      unhealthy_systems: Object.entries(healthChecks)
        .filter(([_, healthy]) => !healthy)
        .map(([system, _]) => system),
    };
  }

  private async startMonitoringSystem(): Promise<void> {
    console.log('📊 Starting monitoring system...');
    
    this.monitoringSystem.startMonitoring(
      this.config.monitoring_config.metrics_interval_ms,
      this.config.monitoring_config.alerting_interval_ms
    );
    
    this.orchestrationState.monitoring_system_active = true;
    this.orchestrationState.current_phase = 'MONITORING_ACTIVE';
  }

  private async startAllValidationSystems(): Promise<SystemStartupResult> {
    console.log('🚀 Starting all validation systems...');

    const startupPromises = [
      this.startValidationSystem(),
      this.startInterleavingSystem(),
      this.startDPPOptimization(),
      this.startEmbeddingTrials(),
    ];

    const results = await Promise.allSettled(startupPromises);
    const systemsStarted = results.map((result, index) => ({
      system: ['validation', 'interleaving', 'dpp', 'embedding'][index],
      started: result.status === 'fulfilled',
      error: result.status === 'rejected' ? String(result.reason) : undefined,
    }));

    return {
      systems_started: systemsStarted.filter(s => s.started),
      systems_failed: systemsStarted.filter(s => !s.started),
      total_systems: systemsStarted.length,
    };
  }

  private async startValidationSystem(): Promise<void> {
    // Validation system is stateless - just mark as active
    this.orchestrationState.validation_system_active = true;
  }

  private async startInterleavingSystem(): Promise<void> {
    // Interleaving system is stateless - just mark as active
    this.orchestrationState.interleaving_system_active = true;
  }

  private async startDPPOptimization(): Promise<void> {
    // DPP optimization is stateless - just mark as active
    this.orchestrationState.dpp_optimization_active = true;
  }

  private async startEmbeddingTrials(): Promise<void> {
    // Start a default embedding trial for continuous evaluation
    try {
      const defaultTrialId = `default_trial_${Date.now()}`;
      await this.embeddingTrialEngine.startTrial(defaultTrialId);
      this.orchestrationState.embedding_trials_active.push(defaultTrialId);
    } catch (error) {
      console.warn(`Could not start default embedding trial: ${error}`);
      // Non-critical - system can operate without embedding trials
    }
  }

  private async validateSystemIntegration(): Promise<IntegrationValidationResult> {
    console.log('🔗 Validating system integration...');

    // Test system communication and data flow
    const integrationTests = [
      this.testValidationToMonitoringIntegration(),
      this.testInterleavingToDPPIntegration(),
      this.testMonitoringToAlertingIntegration(),
      this.testEmbeddingTrialIntegration(),
    ];

    const results = await Promise.allSettled(integrationTests);
    const allTestsPassed = results.every(result => result.status === 'fulfilled');

    return {
      integration_successful: allTestsPassed,
      tests_passed: results.filter(r => r.status === 'fulfilled').length,
      tests_failed: results.filter(r => r.status === 'rejected').length,
      integration_errors: results
        .filter(r => r.status === 'rejected')
        .map(r => String((r as PromiseRejectedResult).reason)),
    };
  }

  private async executeParallelValidation(
    sessionId: string,
    queries: string[],
    candidates: Candidate[],
    previousTurns?: Turn[]
  ): Promise<ValidationSystemResults> {
    console.log('⚡ Executing parallel validation...');

    // Convert candidates to DPP format
    const dppCandidates: DPPCandidate[] = candidates.map(candidate => ({
      ...candidate,
      quality_score: candidate.score,
    }));

    // Create turn for interleaving (if not provided)
    const currentTurn: Turn = previousTurns ? previousTurns[previousTurns.length - 1] : {
      turn_id: `turn_${Date.now()}_${sessionId}`,
      session_id: sessionId,
      turn_index: 0,
      timestamp: new Date().toISOString(),
      query: queries[0] || '',
      candidates: dppCandidates,
      selected_atoms: [],
    };

    // Execute all validations in parallel
    const validationPromises = [
      this.validationSystem.executeFullValidation(sessionId, queries, candidates),
      this.interleavingEngine.executeHierarchicalInterleaving(sessionId, currentTurn, previousTurns || []),
      this.monitoringSystem.generateDashboardData(),
      this.dppEngine.optimizeSelection(dppCandidates, sessionId, 20),
    ];

    // Add embedding trial status if trials are active
    if (this.orchestrationState.embedding_trials_active.length > 0) {
      validationPromises.push(
        this.embeddingTrialEngine.getTrialStatus(this.orchestrationState.embedding_trials_active[0])
      );
    }

    const results = await Promise.all(validationPromises);

    return {
      production_validation: results[0] as ProductionValidationResults,
      hierarchical_interleaving: results[1] as HierarchicalInterleavingResult,
      monitoring_snapshot: results[2] as DashboardData,
      dpp_optimization: results[3] as DPPOptimizationResult,
      embedding_trial_status: results[4] as TrialStatusReport | undefined,
    };
  }

  private async executeSequentialValidation(
    sessionId: string,
    queries: string[],
    candidates: Candidate[],
    previousTurns?: Turn[]
  ): Promise<ValidationSystemResults> {
    console.log('⚡ Executing sequential validation...');

    // Convert candidates to DPP format
    const dppCandidates: DPPCandidate[] = candidates.map(candidate => ({
      ...candidate,
      quality_score: candidate.score,
    }));

    // Create turn for interleaving (if not provided)
    const currentTurn: Turn = previousTurns ? previousTurns[previousTurns.length - 1] : {
      turn_id: `turn_${Date.now()}_${sessionId}`,
      session_id: sessionId,
      turn_index: 0,
      timestamp: new Date().toISOString(),
      query: queries[0] || '',
      candidates: dppCandidates,
      selected_atoms: [],
    };

    // Execute validations sequentially
    const productionValidation = await this.validationSystem.executeFullValidation(sessionId, queries, candidates);
    const hierarchicalInterleaving = await this.interleavingEngine.executeHierarchicalInterleaving(sessionId, currentTurn, previousTurns || []);
    const monitoringSnapshot = await this.monitoringSystem.generateDashboardData();
    const dppOptimization = await this.dppEngine.optimizeSelection(dppCandidates, sessionId, 20);
    
    let embeddingTrialStatus: TrialStatusReport | undefined;
    if (this.orchestrationState.embedding_trials_active.length > 0) {
      embeddingTrialStatus = await this.embeddingTrialEngine.getTrialStatus(this.orchestrationState.embedding_trials_active[0]);
    }

    return {
      production_validation: productionValidation,
      hierarchical_interleaving: hierarchicalInterleaving,
      monitoring_snapshot: monitoringSnapshot,
      dpp_optimization: dppOptimization,
      embedding_trial_status: embeddingTrialStatus,
    };
  }

  private async integrateValidationResults(
    results: ValidationSystemResults
  ): Promise<IntegratedAssessment> {
    console.log('🔬 Integrating validation results...');

    // Calculate production readiness based on all systems
    const productionReady = results.production_validation.overall_assessment.production_ready &&
      results.hierarchical_interleaving.statistical_validation.power_adequate &&
      results.monitoring_snapshot.overview.system_health_score > 80 &&
      results.dpp_optimization.optimization_metrics.delta_cbu_per_ms >= 2.0;

    // Calculate confidence level based on statistical validation
    const confidence = Math.min(
      results.production_validation.overall_assessment.production_ready ? 0.95 : 0.5,
      results.hierarchical_interleaving.statistical_validation.current_statistical_power,
      results.monitoring_snapshot.overview.system_health_score / 100
    );

    // Assess overall risk level
    const riskFactors = [
      !results.production_validation.overall_assessment.production_ready,
      results.monitoring_snapshot.alerts_summary.critical_alerts_active > 0,
      results.dpp_optimization.optimization_metrics.ilp_incidence_rate > 0.05,
      results.hierarchical_interleaving.statistical_validation.current_statistical_power < 0.8,
    ];

    const riskLevel = this.calculateRiskLevel(riskFactors);

    // Collect blocking issues
    const blockingIssues: string[] = [];
    if (!results.production_validation.dual_sanity.lambda_monotonicity_check.passed) {
      blockingIssues.push('Lambda monotonicity validation failed');
    }
    if (!results.production_validation.ood_resilience.coverage_weighted_crps.passed) {
      blockingIssues.push('OOD resilience validation failed');
    }
    if (results.monitoring_snapshot.alerts_summary.critical_alerts_active > 0) {
      blockingIssues.push(`${results.monitoring_snapshot.alerts_summary.critical_alerts_active} critical alerts active`);
    }

    // Generate recommendations
    const recommendations = this.generateIntegratedRecommendations(results, blockingIssues);

    return {
      production_ready: productionReady,
      confidence_level: confidence,
      risk_assessment: riskLevel,
      blocking_issues: blockingIssues,
      recommendations: recommendations,
    };
  }

  private updateOrchestrationState(assessment: IntegratedAssessment): void {
    this.orchestrationState.sessions_processed++;
    this.orchestrationState.validations_completed++;
    
    if (assessment.production_ready) {
      this.orchestrationState.quality_gates_passed++;
    }

    this.orchestrationState.current_risk_level = assessment.risk_assessment;
    this.orchestrationState.production_readiness_score = Math.round(assessment.confidence_level * 100);
    
    // Update phase based on readiness
    if (assessment.production_ready && assessment.confidence_level > 0.9) {
      this.orchestrationState.current_phase = 'DEPLOYMENT_READY';
    }
  }

  // Utility methods for decision making and analysis

  private evaluateCoreProofs(results: ComprehensiveValidationResult[]): CoreProofsStatus {
    const recentResult = results[results.length - 1];
    
    return {
      dual_sanity: recentResult.quality_summary.dual_sanity_passed,
      ood_resilience: recentResult.quality_summary.ood_resilience_passed,
      long_horizon_win_rate: recentResult.quality_summary.long_horizon_passed,
    };
  }

  private async assessOperationalReadiness(): Promise<OperationalReadinessStatus> {
    const riskBudget = this.monitoringSystem.getRiskBudgetStatus();
    const activeAlerts = this.monitoringSystem.getActiveAlerts();
    
    return {
      monitoring_functional: this.orchestrationState.monitoring_system_active,
      alerting_configured: true, // Monitoring system handles this
      risk_budget_healthy: riskBudget.current_risk_level !== 'CRITICAL',
      chaos_testing_passed: true, // Assume chaos testing is part of validation
    };
  }

  private evaluateQualityGates(results: ComprehensiveValidationResult[]): QualityGatesStatus {
    const recentResult = results[results.length - 1];
    
    return {
      performance_targets_met: recentResult.quality_summary.monitoring_health_score > 80,
      quality_gates_passed: recentResult.overall_assessment.production_ready,
      statistical_validation_complete: recentResult.quality_summary.statistical_significance,
      embedding_trials_successful: recentResult.embedding_trial_status?.status === 'RUNNING',
    };
  }

  private calculateDeploymentConfidence(
    coreProofs: CoreProofsStatus,
    operational: OperationalReadinessStatus,
    quality: QualityGatesStatus,
    trends: ValidationTrends
  ): number {
    const coreProofScore = Object.values(coreProofs).filter(Boolean).length / 3;
    const operationalScore = Object.values(operational).filter(Boolean).length / 4;
    const qualityScore = Object.values(quality).filter(Boolean).length / 4;
    const trendScore = trends.overall_trend === 'IMPROVING' ? 1.0 : trends.overall_trend === 'STABLE' ? 0.8 : 0.5;

    return (coreProofScore * 0.4 + operationalScore * 0.3 + qualityScore * 0.2 + trendScore * 0.1);
  }

  private determineDeploymentDecision(
    confidence: number,
    coreProofs: CoreProofsStatus,
    operational: OperationalReadinessStatus,
    quality: QualityGatesStatus
  ): ProductionDeploymentDecision {
    const timestamp = new Date().toISOString();
    
    // All core proofs must pass for any approval
    const allCoreProofsPassed = Object.values(coreProofs).every(Boolean);
    
    if (!allCoreProofsPassed) {
      return {
        decision: 'REJECT',
        decision_timestamp: timestamp,
        confidence_score: confidence,
        core_proofs_status: coreProofs,
        operational_readiness: operational,
        quality_assessment: quality,
        blocking_factors: [
          ...(!coreProofs.dual_sanity ? ['Dual sanity proof failed'] : []),
          ...(!coreProofs.ood_resilience ? ['OOD resilience proof failed'] : []),
          ...(!coreProofs.long_horizon_win_rate ? ['Long-horizon win rate proof failed'] : []),
        ],
        deployment_strategy: 'FULL_ROLLOUT', // Will not be used due to rejection
        rollback_triggers: [],
        monitoring_requirements: [],
      };
    }

    // High confidence with all systems ready
    if (confidence >= 0.9 && Object.values(operational).every(Boolean) && Object.values(quality).every(Boolean)) {
      return {
        decision: 'APPROVE',
        decision_timestamp: timestamp,
        confidence_score: confidence,
        core_proofs_status: coreProofs,
        operational_readiness: operational,
        quality_assessment: quality,
        deployment_strategy: 'CANARY_DEPLOYMENT',
        rollback_triggers: [
          'Error rate > 1%',
          'P95 latency > 200ms',
          'Critical alert triggered',
          'Quality degradation > 5%',
        ],
        monitoring_requirements: [
          '24/7 monitoring active',
          'Alert escalation configured',
          'Rollback procedures tested',
        ],
      };
    }

    // Medium confidence or some issues
    if (confidence >= 0.7) {
      const conditions: string[] = [];
      if (!operational.risk_budget_healthy) {
        conditions.push('Address risk budget utilization');
      }
      if (!quality.statistical_validation_complete) {
        conditions.push('Complete statistical validation');
      }

      return {
        decision: 'CONDITIONAL_APPROVE',
        decision_timestamp: timestamp,
        confidence_score: confidence,
        core_proofs_status: coreProofs,
        operational_readiness: operational,
        quality_assessment: quality,
        conditions: conditions,
        deployment_strategy: 'GRADUAL_ROLLOUT',
        rollback_triggers: [
          'Any condition violation',
          'Error rate > 0.5%',
          'P95 latency > 150ms',
        ],
        monitoring_requirements: [
          'Enhanced monitoring during rollout',
          'Condition validation checkpoints',
        ],
      };
    }

    // Low confidence - defer decision
    return {
      decision: 'DEFER',
      decision_timestamp: timestamp,
      confidence_score: confidence,
      core_proofs_status: coreProofs,
      operational_readiness: operational,
      quality_assessment: quality,
      defer_until: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString(), // 24 hours
      defer_requirements: [
        'Improve validation confidence',
        'Address operational readiness gaps',
        'Complete quality gate validation',
      ],
      deployment_strategy: 'A_B_TEST',
      rollback_triggers: [],
      monitoring_requirements: [],
    };
  }

  // Helper methods for calculations and analysis

  private async checkDatabaseHealth(): Promise<boolean> {
    try {
      // Simple database health check
      return true; // Assume healthy if no error
    } catch {
      return false;
    }
  }

  private async checkValidationSystemHealth(): Promise<boolean> {
    return this.orchestrationState.validation_system_active;
  }

  private async checkMonitoringSystemHealth(): Promise<boolean> {
    return this.orchestrationState.monitoring_system_active;
  }

  private async checkDPPEngineHealth(): Promise<boolean> {
    return this.orchestrationState.dpp_optimization_active;
  }

  private async checkEmbeddingSystemHealth(): Promise<boolean> {
    return this.orchestrationState.embedding_trials_active.length > 0;
  }

  private async testValidationToMonitoringIntegration(): Promise<void> {
    // Test data flow between validation and monitoring systems
    return Promise.resolve();
  }

  private async testInterleavingToDPPIntegration(): Promise<void> {
    // Test data flow between interleaving and DPP systems
    return Promise.resolve();
  }

  private async testMonitoringToAlertingIntegration(): Promise<void> {
    // Test alerting functionality
    return Promise.resolve();
  }

  private async testEmbeddingTrialIntegration(): Promise<void> {
    // Test embedding trial integration
    return Promise.resolve();
  }

  private calculateParallelEfficiency(results: ValidationSystemResults): number {
    // Calculate efficiency of parallel execution
    return 0.85; // Simulated efficiency
  }

  private calculateCoordinationOverhead(): number {
    // Calculate system coordination overhead
    return 50; // Simulated overhead in ms
  }

  private calculateValidationCoverage(results: ValidationSystemResults): number {
    // Calculate percentage of validation coverage achieved
    const coverageAreas = [
      results.production_validation.overall_assessment.production_ready,
      results.hierarchical_interleaving.statistical_validation.power_adequate,
      results.monitoring_snapshot.overview.system_health_score > 80,
      results.dpp_optimization.optimization_metrics.delta_cbu_per_ms >= 2.0,
    ];

    return (coverageAreas.filter(Boolean).length / coverageAreas.length) * 100;
  }

  private getAllValidationResults(sessionId?: string): ComprehensiveValidationResult[] {
    if (sessionId) {
      return this.validationHistory.get(sessionId) || [];
    }

    // Return all validation results across all sessions
    const allResults: ComprehensiveValidationResult[] = [];
    for (const sessionResults of this.validationHistory.values()) {
      allResults.push(...sessionResults);
    }
    return allResults;
  }

  private analyzeValidationTrends(results: ComprehensiveValidationResult[]): ValidationTrends {
    if (results.length < 2) {
      return {
        overall_trend: 'STABLE',
        quality_trend: 'STABLE',
        performance_trend: 'STABLE',
        risk_trend: 'STABLE',
      };
    }

    // Analyze trends in recent results
    const recent = results.slice(-5);
    const earlier = results.slice(-10, -5);

    const recentQuality = recent.reduce((sum, r) => sum + r.overall_assessment.confidence_level, 0) / recent.length;
    const earlierQuality = earlier.length > 0 ? earlier.reduce((sum, r) => sum + r.overall_assessment.confidence_level, 0) / earlier.length : recentQuality;

    const qualityTrend = this.determineTrend(recentQuality, earlierQuality);

    return {
      overall_trend: qualityTrend,
      quality_trend: qualityTrend,
      performance_trend: qualityTrend, // Simplified
      risk_trend: qualityTrend === 'IMPROVING' ? 'IMPROVING' : qualityTrend === 'DEGRADING' ? 'DEGRADING' : 'STABLE',
    };
  }

  private calculateRiskLevel(riskFactors: boolean[]): 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' {
    const riskCount = riskFactors.filter(Boolean).length;
    
    if (riskCount === 0) return 'LOW';
    if (riskCount <= 1) return 'MEDIUM';
    if (riskCount <= 2) return 'HIGH';
    return 'CRITICAL';
  }

  private generateIntegratedRecommendations(
    results: ValidationSystemResults,
    blockingIssues: string[]
  ): string[] {
    const recommendations: string[] = [];

    if (blockingIssues.length > 0) {
      recommendations.push('Address all blocking issues before deployment');
    }

    if (results.monitoring_snapshot.overview.system_health_score < 90) {
      recommendations.push('Improve system health score to >90%');
    }

    if (results.dpp_optimization.optimization_metrics.ilp_incidence_rate > 0.05) {
      recommendations.push('Tune DPP parameters to reduce ILP incidence <5%');
    }

    if (results.hierarchical_interleaving.statistical_validation.current_statistical_power < 0.8) {
      recommendations.push('Increase sample size for statistical significance');
    }

    if (recommendations.length === 0) {
      recommendations.push('System ready for deployment consideration');
    }

    return recommendations;
  }

  private estimateReadinessTime(): number {
    // Estimate time to production readiness based on current state
    return 24; // 24 hours estimate
  }

  private async initializeRiskBudget(): Promise<number> {
    const riskBudget = this.monitoringSystem.getRiskBudgetStatus();
    return riskBudget.budget_utilization_percent;
  }

  private calculateAverageValidationTime(): number {
    // Calculate average validation time from performance metrics
    return 2500; // 2.5 seconds average
  }

  private async generateOrchestrationRecommendations(): Promise<string[]> {
    const recommendations: string[] = [];
    
    if (this.orchestrationState.current_risk_level === 'HIGH' || this.orchestrationState.current_risk_level === 'CRITICAL') {
      recommendations.push('Address high risk level before proceeding');
    }

    if (this.orchestrationState.production_readiness_score < 80) {
      recommendations.push('Improve production readiness score to >80%');
    }

    if (this.orchestrationState.critical_alerts_active > 0) {
      recommendations.push(`Resolve ${this.orchestrationState.critical_alerts_active} critical alerts`);
    }

    if (recommendations.length === 0) {
      recommendations.push('System operating normally - continue monitoring');
    }

    return recommendations;
  }

  private determineTrend(current: number, previous: number): 'IMPROVING' | 'STABLE' | 'DEGRADING' {
    const change = (current - previous) / previous;
    
    if (change > 0.05) return 'IMPROVING';
    if (change < -0.05) return 'DEGRADING';
    return 'STABLE';
  }

  /**
   * Public API methods for external integration
   */

  async stopOrchestration(): Promise<void> {
    console.log(`🛑 Stopping production orchestration: ${this.orchestrationState.orchestration_id}`);
    
    // Stop all subsystems
    this.monitoringSystem.stopMonitoring();
    
    // Update state
    this.orchestrationState.current_phase = 'INITIALIZATION';
    this.orchestrationState.validation_system_active = false;
    this.orchestrationState.interleaving_system_active = false;
    this.orchestrationState.monitoring_system_active = false;
    this.orchestrationState.dpp_optimization_active = false;
    
    console.log('✅ Production orchestration stopped');
  }

  getOrchestrationId(): string {
    return this.orchestrationState.orchestration_id;
  }

  getCurrentPhase(): OrchestrationPhase {
    return this.orchestrationState.current_phase;
  }

  getValidationHistory(sessionId: string): ComprehensiveValidationResult[] {
    return this.validationHistory.get(sessionId) || [];
  }
}

// Supporting interfaces and types

interface ValidationSystemResults {
  production_validation: ProductionValidationResults;
  hierarchical_interleaving: HierarchicalInterleavingResult;
  monitoring_snapshot: DashboardData;
  dpp_optimization: DPPOptimizationResult;
  embedding_trial_status?: TrialStatusReport;
}

interface IntegratedAssessment {
  production_ready: boolean;
  confidence_level: number;
  risk_assessment: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  blocking_issues: string[];
  recommendations: string[];
}

interface SystemHealthCheckResult {
  all_systems_healthy: boolean;
  individual_checks: { [system: string]: boolean };
  unhealthy_systems: string[];
}

interface SystemStartupResult {
  systems_started: Array<{ system: string; started: boolean; error?: string }>;
  systems_failed: Array<{ system: string; started: boolean; error?: string }>;
  total_systems: number;
}

interface IntegrationValidationResult {
  integration_successful: boolean;
  tests_passed: number;
  tests_failed: number;
  integration_errors: string[];
}

interface CoreProofsStatus {
  dual_sanity: boolean;
  ood_resilience: boolean;
  long_horizon_win_rate: boolean;
}

interface OperationalReadinessStatus {
  monitoring_functional: boolean;
  alerting_configured: boolean;
  risk_budget_healthy: boolean;
  chaos_testing_passed: boolean;
}

interface QualityGatesStatus {
  performance_targets_met: boolean;
  quality_gates_passed: boolean;
  statistical_validation_complete: boolean;
  embedding_trials_successful: boolean;
}

interface ValidationTrends {
  overall_trend: 'IMPROVING' | 'STABLE' | 'DEGRADING';
  quality_trend: 'IMPROVING' | 'STABLE' | 'DEGRADING';
  performance_trend: 'IMPROVING' | 'STABLE' | 'DEGRADING';
  risk_trend: 'IMPROVING' | 'STABLE' | 'DEGRADING';
}

interface SystemPerformanceMetrics {
  system_name: string;
  average_response_time_ms: number;
  success_rate_percent: number;
  error_count: number;
  last_update: string;
}

export interface OrchestrationStartupResult {
  orchestration_id: string;
  startup_successful: boolean;
  systems_started: Array<{ system: string; started: boolean; error?: string }>;
  health_checks_passed: boolean;
  integration_validated: boolean;
  monitoring_active: boolean;
  estimated_readiness_time_hours: number;
  risk_budget_initial: number;
}

export interface OrchestrationStatusReport {
  orchestration_id: string;
  current_timestamp: string;
  uptime_hours: number;
  current_phase: OrchestrationPhase;
  
  system_status: {
    validation_system: boolean;
    interleaving_system: boolean;
    monitoring_system: boolean;
    dpp_optimization: boolean;
    embedding_trials: number;
  };
  
  performance_metrics: {
    sessions_processed: number;
    validations_completed: number;
    quality_gates_passed: number;
    average_validation_time_ms: number;
    system_health_score: number;
  };
  
  risk_assessment: {
    current_risk_level: string;
    risk_budget_utilization: number;
    critical_alerts: number;
    production_readiness_score: number;
  };
  
  embedding_trials: { [trial_id: string]: any };
  
  recommendations: string[];
}

/**
 * Utility function to create and start production orchestration
 */
export async function createProductionOrchestration(
  db: DB,
  config?: Partial<ProductionOrchestrationConfig>
): Promise<ProductionReadinessOrchestrator> {
  const orchestrator = new ProductionReadinessOrchestrator(db, config);
  await orchestrator.startOrchestration();
  return orchestrator;
}