/**
 * Gap→Tune→Verify Framework - Main Entry Point
 * 
 * This is the complete implementation of the Gap→Tune→Verify framework as specified
 * in the TODO.md plan. It provides a comprehensive system for:
 * 
 * - Automatically identifying performance gaps in retrieval systems
 * - Generating counterfactual analyses for policy optimization
 * - Running constrained auto-tuning with domain-specific biases
 * - Validating improvements through paired replay testing
 * - Integrating results with marketing microsites for buyer-facing demonstrations
 * - Comprehensive monitoring and validation infrastructure
 * 
 * @author Gap Analysis System
 * @version 1.0.0
 * @since 2024-01-01
 */

// ============================================================================
// Core Types and Interfaces
// ============================================================================

export * from './types';

// ============================================================================
// Main Component Exports
// ============================================================================

// Gap detection and analysis
export { GapBoard } from './gap-board';

// Counterfactual analysis and IPS
export { CounterfactualCBU } from './counterfactual-cbu';

// Bayesian + rule-based auto-tuning
export { AutoTuningEngine } from './auto-tuning';

// Paired replay validation and promotion
export { PromotionPipeline } from './promotion-pipeline';

// Lightweight GBM difficulty gating
export { DifficultyGateSystem } from './difficulty-gate';

// Automatic stratification and slice mining
export { SliceMiningEngine } from './slice-mining';

// Buyer-facing Pareto front generation
export { MicrositeIntegrationSystem } from './microsite-integration';

// Comprehensive monitoring system
export { 
  GapAnalysisMonitor, 
  MonitoringFactory,
  DEFAULT_MONITORING_CONFIG 
} from './monitoring';

// Validation and quality assurance
export { 
  GapAnalysisValidator, 
  ValidationFactory,
  DEFAULT_VALIDATION_CONFIG 
} from './validation';

// Main orchestration system
export { 
  GapAnalysisOrchestrator, 
  OrchestratorFactory,
  DEFAULT_ORCHESTRATOR_CONFIG 
} from './orchestrator';

// ============================================================================
// Convenience API
// ============================================================================

import { 
  GapAnalysisOrchestrator,
  OrchestratorFactory,
  WorkflowRequest,
  WorkflowInput,
  WorkflowResult,
  OrchestratorStatus
} from './orchestrator';

import {
  GapAnalysisResult,
  PolicyFingerprint,
  GapRecord,
  CounterfactualAnalysis,
  OptimizedPolicy,
  PromotionResult,
  SliceMiningResult,
  MicrositePackage,
  EvaluationResult,
  CompetitorBaseline,
  SavedAtomData
} from './types';

/**
 * High-level API for the Gap→Tune→Verify framework
 * 
 * This class provides a simplified interface for the most common workflows
 * while still allowing access to the full orchestrator capabilities.
 */
export class GapAnalysisSystem {
  private orchestrator: GapAnalysisOrchestrator;
  private initialized = false;

  constructor(orchestrator?: GapAnalysisOrchestrator) {
    this.orchestrator = orchestrator || OrchestratorFactory.createOrchestrator();
  }

  // ========================================================================
  // System Lifecycle
  // ========================================================================

  /**
   * Initialize the Gap Analysis System
   * 
   * This starts all monitoring, validation, and component systems.
   * Must be called before running any workflows.
   */
  public async initialize(): Promise<GapAnalysisResult<void>> {
    if (this.initialized) {
      return {
        success: true,
        data: undefined,
        message: 'Gap Analysis System already initialized'
      };
    }

    const initResult = await this.orchestrator.initialize();
    if (!initResult.success) return initResult;

    const startResult = await this.orchestrator.start();
    if (!startResult.success) return startResult;

    this.initialized = true;
    return {
      success: true,
      data: undefined,
      message: 'Gap Analysis System initialized and started successfully'
    };
  }

  /**
   * Shutdown the Gap Analysis System
   * 
   * Stops all workflows, monitoring, and component systems gracefully.
   */
  public async shutdown(): Promise<GapAnalysisResult<void>> {
    if (!this.initialized) {
      return {
        success: true,
        data: undefined,
        message: 'Gap Analysis System not initialized'
      };
    }

    const result = await this.orchestrator.stop();
    this.initialized = false;
    return result;
  }

  // ========================================================================
  // High-Level Workflow APIs
  // ========================================================================

  /**
   * Run complete Gap→Tune→Verify workflow
   * 
   * This is the main entry point for running the complete framework.
   * It will automatically detect gaps, perform counterfactual analysis,
   * run auto-tuning, validate through paired replay, and generate
   * buyer-facing materials.
   * 
   * @param input - Complete input including validator output, competitor results, etc.
   * @param options - Workflow options like timeouts and feature flags
   */
  public async runCompleteWorkflow(
    input: CompleteWorkflowInput,
    options: WorkflowOptions = {}
  ): Promise<GapAnalysisResult<CompleteWorkflowOutput>> {
    try {
      if (!this.initialized) {
        const initResult = await this.initialize();
        if (!initResult.success) return initResult as GapAnalysisResult<CompleteWorkflowOutput>;
      }

      const workflowId = options.workflowId || `workflow_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      const request: WorkflowRequest = {
        id: workflowId,
        type: 'complete_workflow',
        priority: options.priority || 'medium',
        input: {
          validatorJsonl: input.validatorJsonl,
          competitorResults: input.competitorResults,
          evaluationResults: input.evaluationResults,
          savedAtoms: input.savedAtoms,
          maxPerturbations: input.maxPerturbations || 50,
          maxTrials: input.maxTrials || 12,
          enableSliceMining: input.enableSliceMining ?? true,
          generateMicrosite: input.generateMicrosite ?? true
        },
        metadata: options.metadata
      };

      const submitResult = await this.orchestrator.submitWorkflow(request);
      if (!submitResult.success) {
        return submitResult as GapAnalysisResult<CompleteWorkflowOutput>;
      }

      // Wait for completion if requested
      if (options.waitForCompletion !== false) {
        const result = await this.waitForWorkflowCompletion(workflowId, options.timeoutMs);
        if (!result.success) return result as GapAnalysisResult<CompleteWorkflowOutput>;

        const workflowResult = result.data!;
        if (workflowResult.status !== 'completed') {
          return {
            success: false,
            error: 'WORKFLOW_FAILED',
            message: `Workflow failed: ${workflowResult.error || 'Unknown error'}`
          };
        }

        return {
          success: true,
          data: {
            workflowId,
            gaps: workflowResult.gapDetectionResult?.data || [],
            counterfactualAnalysis: workflowResult.counterfactualResult?.data,
            optimizedPolicy: workflowResult.autoTuningResult?.data,
            promotionResult: workflowResult.promotionResult?.data,
            sliceMiningResult: workflowResult.sliceMiningResult?.data,
            micrositePackage: workflowResult.micrositeResult?.data,
            validationReport: workflowResult.validationReport,
            executionTimeMs: workflowResult.executionTimeMs,
            warnings: workflowResult.warnings,
            recommendations: workflowResult.recommendations
          },
          message: `Complete workflow executed successfully in ${(workflowResult.executionTimeMs / 1000).toFixed(1)}s`
        };
      } else {
        return {
          success: true,
          data: {
            workflowId,
            gaps: [],
            executionTimeMs: 0
          },
          message: `Workflow ${workflowId} submitted successfully`
        };
      }

    } catch (error) {
      return {
        success: false,
        error: 'WORKFLOW_EXECUTION_ERROR',
        message: `Failed to execute complete workflow: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
  }

  /**
   * Detect performance gaps only
   * 
   * Runs just the gap detection phase without subsequent optimization.
   * Useful for analysis and reporting purposes.
   */
  public async detectGaps(
    validatorJsonl: string[],
    competitorResults: CompetitorBaseline[]
  ): Promise<GapAnalysisResult<GapRecord[]>> {
    try {
      if (!this.initialized) {
        const initResult = await this.initialize();
        if (!initResult.success) return initResult as GapAnalysisResult<GapRecord[]>;
      }

      const workflowId = `gap_detection_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      const request: WorkflowRequest = {
        id: workflowId,
        type: 'gap_detection',
        priority: 'medium',
        input: { validatorJsonl, competitorResults }
      };

      const submitResult = await this.orchestrator.submitWorkflow(request);
      if (!submitResult.success) {
        return submitResult as GapAnalysisResult<GapRecord[]>;
      }

      const result = await this.waitForWorkflowCompletion(workflowId);
      if (!result.success) return result as GapAnalysisResult<GapRecord[]>;

      const workflowResult = result.data!;
      if (workflowResult.status !== 'completed' || !workflowResult.gapDetectionResult?.success) {
        return {
          success: false,
          error: 'GAP_DETECTION_FAILED',
          message: workflowResult.error || 'Gap detection failed'
        };
      }

      return {
        success: true,
        data: workflowResult.gapDetectionResult.data || [],
        message: `Detected ${workflowResult.gapDetectionResult.data?.length || 0} performance gaps`
      };

    } catch (error) {
      return {
        success: false,
        error: 'GAP_DETECTION_ERROR',
        message: `Gap detection failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
  }

  /**
   * Run auto-tuning for a specific gap
   * 
   * Given a gap and counterfactual analysis, runs the Bayesian + rule-based
   * optimization to find an improved policy configuration.
   */
  public async optimizePolicy(
    gapRecord: GapRecord,
    savedAtoms: SavedAtomData[],
    options: OptimizationOptions = {}
  ): Promise<GapAnalysisResult<OptimizedPolicy>> {
    try {
      if (!this.initialized) {
        const initResult = await this.initialize();
        if (!initResult.success) return initResult as GapAnalysisResult<OptimizedPolicy>;
      }

      // First run counterfactual analysis
      const workflowId = `counterfactual_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      let request: WorkflowRequest = {
        id: workflowId,
        type: 'counterfactual_analysis',
        priority: options.priority || 'medium',
        input: {
          gapRecord,
          savedAtoms,
          maxPerturbations: options.maxPerturbations || 50
        }
      };

      let submitResult = await this.orchestrator.submitWorkflow(request);
      if (!submitResult.success) {
        return submitResult as GapAnalysisResult<OptimizedPolicy>;
      }

      let result = await this.waitForWorkflowCompletion(workflowId);
      if (!result.success) return result as GapAnalysisResult<OptimizedPolicy>;

      const counterfactualResult = result.data!.counterfactualResult;
      if (!counterfactualResult?.success) {
        return {
          success: false,
          error: 'COUNTERFACTUAL_ANALYSIS_FAILED',
          message: counterfactualResult?.message || 'Counterfactual analysis failed'
        };
      }

      // Then run auto-tuning
      const tuningWorkflowId = `auto_tuning_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      request = {
        id: tuningWorkflowId,
        type: 'auto_tuning',
        priority: options.priority || 'medium',
        input: {
          gapRecord,
          counterfactualAnalysis: counterfactualResult.data!,
          maxTrials: options.maxTrials || 12
        }
      };

      submitResult = await this.orchestrator.submitWorkflow(request);
      if (!submitResult.success) {
        return submitResult as GapAnalysisResult<OptimizedPolicy>;
      }

      result = await this.waitForWorkflowCompletion(tuningWorkflowId);
      if (!result.success) return result as GapAnalysisResult<OptimizedPolicy>;

      const workflowResult = result.data!;
      if (workflowResult.status !== 'completed' || !workflowResult.autoTuningResult?.success) {
        return {
          success: false,
          error: 'AUTO_TUNING_FAILED',
          message: workflowResult.error || 'Auto-tuning failed'
        };
      }

      return {
        success: true,
        data: workflowResult.autoTuningResult.data!,
        message: `Policy optimization completed with ${workflowResult.autoTuningResult.data?.optimization_iterations || 0} iterations`
      };

    } catch (error) {
      return {
        success: false,
        error: 'OPTIMIZATION_ERROR',
        message: `Policy optimization failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
  }

  /**
   * Generate buyer-facing microsite package
   * 
   * Creates marketing materials, Pareto fronts, and interactive tools
   * for demonstrating the system's capabilities to potential buyers.
   */
  public async generateMicrositePackage(
    promotionResults: PromotionResult[],
    competitorData: CompetitorBaseline[]
  ): Promise<GapAnalysisResult<MicrositePackage>> {
    try {
      if (!this.initialized) {
        const initResult = await this.initialize();
        if (!initResult.success) return initResult as GapAnalysisResult<MicrositePackage>;
      }

      // Use microsite integration directly since this is a standalone operation
      const micrositeIntegration = new (await import('./microsite-integration')).MicrositeIntegrationSystem();
      
      const result = await micrositeIntegration.generateMicrositePackage(
        promotionResults,
        competitorData
      );

      return result;

    } catch (error) {
      return {
        success: false,
        error: 'MICROSITE_GENERATION_ERROR',
        message: `Microsite generation failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
  }

  // ========================================================================
  // System Status and Monitoring
  // ========================================================================

  /**
   * Get current system status
   */
  public getSystemStatus(): OrchestratorStatus {
    return this.orchestrator.getOrchestratorStatus();
  }

  /**
   * Generate comprehensive status report
   */
  public generateStatusReport(): string {
    return this.orchestrator.generateStatusReport();
  }

  /**
   * Get workflow status by ID
   */
  public async getWorkflowStatus(workflowId: string): Promise<GapAnalysisResult<WorkflowResult | null>> {
    return await this.orchestrator.getWorkflowStatus(workflowId);
  }

  /**
   * Cancel a running or queued workflow
   */
  public async cancelWorkflow(workflowId: string): Promise<GapAnalysisResult<void>> {
    return await this.orchestrator.cancelWorkflow(workflowId);
  }

  // ========================================================================
  // Helper Methods
  // ========================================================================

  private async waitForWorkflowCompletion(
    workflowId: string,
    timeoutMs = 1800000 // 30 minutes default
  ): Promise<GapAnalysisResult<WorkflowResult>> {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeoutMs) {
      const statusResult = await this.orchestrator.getWorkflowStatus(workflowId);
      if (!statusResult.success) return statusResult as GapAnalysisResult<WorkflowResult>;

      const workflow = statusResult.data;
      if (!workflow) {
        return {
          success: false,
          error: 'WORKFLOW_NOT_FOUND',
          message: `Workflow ${workflowId} not found`
        };
      }

      if (workflow.status === 'completed' || workflow.status === 'failed' || workflow.status === 'cancelled') {
        return {
          success: true,
          data: workflow,
          message: `Workflow ${workflowId} ${workflow.status}`
        };
      }

      // Wait before checking again
      await new Promise(resolve => setTimeout(resolve, 5000)); // Check every 5 seconds
    }

    return {
      success: false,
      error: 'WORKFLOW_TIMEOUT',
      message: `Workflow ${workflowId} timed out after ${timeoutMs / 1000}s`
    };
  }
}

// ============================================================================
// Convenience Types for High-Level API
// ============================================================================

export interface CompleteWorkflowInput {
  validatorJsonl: string[];
  competitorResults: CompetitorBaseline[];
  evaluationResults?: EvaluationResult[];
  savedAtoms: SavedAtomData[];
  maxPerturbations?: number;
  maxTrials?: number;
  enableSliceMining?: boolean;
  generateMicrosite?: boolean;
}

export interface CompleteWorkflowOutput {
  workflowId: string;
  gaps: GapRecord[];
  counterfactualAnalysis?: CounterfactualAnalysis;
  optimizedPolicy?: OptimizedPolicy;
  promotionResult?: PromotionResult;
  sliceMiningResult?: SliceMiningResult;
  micrositePackage?: MicrositePackage;
  validationReport?: any;
  executionTimeMs: number;
  warnings?: string[];
  recommendations?: string[];
}

export interface WorkflowOptions {
  workflowId?: string;
  priority?: 'low' | 'medium' | 'high' | 'critical';
  waitForCompletion?: boolean;
  timeoutMs?: number;
  metadata?: Record<string, any>;
}

export interface OptimizationOptions {
  maxPerturbations?: number;
  maxTrials?: number;
  priority?: 'low' | 'medium' | 'high' | 'critical';
}

// ============================================================================
// Factory Functions for Common Use Cases
// ============================================================================

/**
 * Create a development-optimized Gap Analysis System
 * 
 * - Disabled monitoring and validation for faster iteration
 * - Lower resource requirements
 * - Simplified error handling
 */
export function createDevelopmentSystem(): GapAnalysisSystem {
  const orchestrator = OrchestratorFactory.createDevelopmentOrchestrator();
  return new GapAnalysisSystem(orchestrator);
}

/**
 * Create a production-ready Gap Analysis System
 * 
 * - Full monitoring and validation enabled
 * - High availability and fault tolerance
 * - Comprehensive audit logging
 * - Performance optimization
 */
export function createProductionSystem(): GapAnalysisSystem {
  const orchestrator = OrchestratorFactory.createProductionOrchestrator();
  return new GapAnalysisSystem(orchestrator);
}

/**
 * Create a custom Gap Analysis System with specific configuration
 */
export function createCustomSystem(config: Partial<import('./orchestrator').OrchestratorConfig>): GapAnalysisSystem {
  const orchestrator = OrchestratorFactory.createOrchestrator(config);
  return new GapAnalysisSystem(orchestrator);
}

// ============================================================================
// Version and Metadata
// ============================================================================

export const VERSION = '1.0.0';
export const SYSTEM_NAME = 'Gap→Tune→Verify Framework';
export const DESCRIPTION = 'Comprehensive system for automatic gap detection, counterfactual analysis, policy optimization, and buyer-facing demonstration generation';

/**
 * Get system information
 */
export function getSystemInfo() {
  return {
    name: SYSTEM_NAME,
    version: VERSION,
    description: DESCRIPTION,
    components: [
      'GapBoard v1 (delta maps + root-cause)',
      'Counterfactual CBU (fast recomputation)',
      'Auto-tuning profiles (Bayesian + rules)',
      'Promotion pipeline (paired replay)',
      'Difficulty gate (lightweight GBM)',
      'Slice mining (auto-stratification)',
      'Microsite integration (buyer-facing Pareto)',
      'Comprehensive monitoring and validation'
    ],
    capabilities: [
      'Automatic performance gap detection',
      'Counterfactual policy analysis with IPS',
      'Constrained Bayesian optimization',
      'Paired replay validation',
      'Statistical significance testing',
      'Domain-specific tuning profiles',
      'Buyer-facing marketing generation',
      'Real-time monitoring and alerting',
      'Comprehensive quality validation'
    ],
    compatibility: {
      runtime: 'Node.js 18+',
      language: 'TypeScript 5.0+',
      frameworks: ['Lethe Core', 'Modern JS/TS ecosystem'],
      deployment: ['Development', 'Production', 'Cloud-native']
    }
  };
}

// ============================================================================
// Default Export for Convenience
// ============================================================================

export default GapAnalysisSystem;