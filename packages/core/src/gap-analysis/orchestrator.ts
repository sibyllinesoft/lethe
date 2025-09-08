/**
 * Main orchestrator for the complete Gap→Tune→Verify framework
 * 
 * This module provides:
 * - End-to-end workflow orchestration
 * - Component integration and coordination
 * - Monitoring and validation integration
 * - Error handling and recovery
 * - Production deployment support
 */

import { EventEmitter } from 'events';
import { 
  GapAnalysisResult, PolicyFingerprint, GapRecord, CounterfactualAnalysis,
  OptimizedPolicy, PromotionResult, SliceMiningResult, MicrositePackage,
  EvaluationResult, CompetitorBaseline, SavedAtomData
} from './types';

import { GapBoard } from './gap-board';
import { CounterfactualCBU } from './counterfactual-cbu';
import { AutoTuningEngine } from './auto-tuning';
import { PromotionPipeline } from './promotion-pipeline';
import { DifficultyGateSystem } from './difficulty-gate';
import { SliceMiningEngine } from './slice-mining';
import { MicrositeIntegrationSystem } from './microsite-integration';
import { GapAnalysisMonitor, MonitoringConfig } from './monitoring';
import { GapAnalysisValidator, ValidationConfig, ValidationReport } from './validation';

// ============================================================================
// Orchestrator Types and Configuration
// ============================================================================

export interface OrchestratorConfig {
  // Component configurations
  monitoring: MonitoringConfig;
  validation: ValidationConfig;
  
  // Workflow settings
  enableMonitoring: boolean;
  enableValidation: boolean;
  enableRecovery: boolean;
  
  // Performance settings
  maxConcurrentWorkflows: number;
  workflowTimeoutMs: number;
  retryAttempts: number;
  backoffMultiplier: number;
  
  // Quality settings
  requireValidationBeforePromotion: boolean;
  requireMonitoringBeforeDeployment: boolean;
  autoRejectOnValidationFailure: boolean;
  
  // Persistence settings
  persistIntermediateResults: boolean;
  resultStoragePath?: string;
  enableAuditLogging: boolean;
}

export interface WorkflowRequest {
  id: string;
  type: 'gap_detection' | 'counterfactual_analysis' | 'auto_tuning' | 'promotion' | 'complete_workflow';
  priority: 'low' | 'medium' | 'high' | 'critical';
  input: WorkflowInput;
  metadata?: Record<string, any>;
}

export interface WorkflowInput {
  // For gap detection
  validatorJsonl?: string[];
  competitorResults?: CompetitorBaseline[];
  
  // For counterfactual analysis
  gapRecord?: GapRecord;
  savedAtoms?: SavedAtomData[];
  
  // For auto-tuning
  counterfactualAnalysis?: CounterfactualAnalysis;
  
  // For promotion
  optimizedPolicy?: OptimizedPolicy;
  
  // For complete workflow
  evaluationResults?: EvaluationResult[];
  
  // Common parameters
  maxPerturbations?: number;
  maxTrials?: number;
  enableSliceMining?: boolean;
  generateMicrosite?: boolean;
}

export interface WorkflowResult {
  id: string;
  status: 'completed' | 'failed' | 'cancelled' | 'timeout';
  startTime: Date;
  endTime: Date;
  executionTimeMs: number;
  
  // Results from each stage
  gapDetectionResult?: GapAnalysisResult<GapRecord[]>;
  counterfactualResult?: GapAnalysisResult<CounterfactualAnalysis>;
  autoTuningResult?: GapAnalysisResult<OptimizedPolicy>;
  promotionResult?: GapAnalysisResult<PromotionResult>;
  sliceMiningResult?: GapAnalysisResult<SliceMiningResult>;
  micrositeResult?: GapAnalysisResult<MicrositePackage>;
  
  // Quality assurance results
  validationReport?: ValidationReport;
  monitoringMetrics?: any;
  
  // Error information
  error?: string;
  warnings?: string[];
  recommendations?: string[];
}

export interface OrchestratorStatus {
  isRunning: boolean;
  activeWorkflows: number;
  completedWorkflows: number;
  failedWorkflows: number;
  queuedWorkflows: number;
  
  systemHealth: 'healthy' | 'degraded' | 'critical';
  lastHealthCheck: Date;
  
  performance: {
    avgWorkflowTime: number;
    successRate: number;
    throughput: number;
  };
}

// ============================================================================
// Main Orchestrator Class
// ============================================================================

export class GapAnalysisOrchestrator extends EventEmitter {
  private config: OrchestratorConfig;
  
  // Component instances
  private gapBoard: GapBoard;
  private counterfactualCBU: CounterfactualCBU;
  private autoTuning: AutoTuningEngine;
  private promotionPipeline: PromotionPipeline;
  private difficultyGate: DifficultyGateSystem;
  private sliceMining: SliceMiningEngine;
  private micrositeIntegration: MicrositeIntegrationSystem;
  
  // Monitoring and validation
  private monitor?: GapAnalysisMonitor;
  private validator?: GapAnalysisValidator;
  
  // State management
  private isInitialized = false;
  private isRunning = false;
  private workflowQueue: WorkflowRequest[] = [];
  private activeWorkflows = new Map<string, WorkflowContext>();
  private completedWorkflows: WorkflowResult[] = [];
  
  constructor(config: OrchestratorConfig) {
    super();
    this.config = config;
    
    // Initialize components
    this.gapBoard = new GapBoard();
    this.counterfactualCBU = new CounterfactualCBU();
    this.autoTuning = new AutoTuningEngine();
    this.promotionPipeline = new PromotionPipeline();
    this.difficultyGate = new DifficultyGateSystem();
    this.sliceMining = new SliceMiningEngine();
    this.micrositeIntegration = new MicrositeIntegrationSystem();
  }

  // ========================================================================
  // Initialization and Lifecycle
  // ========================================================================

  public async initialize(): Promise<GapAnalysisResult<void>> {
    try {
      if (this.isInitialized) {
        return {
          success: true,
          data: undefined,
          message: 'Orchestrator already initialized'
        };
      }

      // Initialize monitoring if enabled
      if (this.config.enableMonitoring) {
        this.monitor = new GapAnalysisMonitor(this.config.monitoring);
        await this.monitor.start();
        this.setupMonitoringEventHandlers();
      }

      // Initialize validation if enabled
      if (this.config.enableValidation) {
        this.validator = new GapAnalysisValidator(this.config.validation);
      }

      // Setup component event handlers
      this.setupComponentEventHandlers();

      this.isInitialized = true;
      this.emit('orchestrator:initialized');

      return {
        success: true,
        data: undefined,
        message: 'Gap Analysis Orchestrator initialized successfully'
      };

    } catch (error) {
      return {
        success: false,
        error: 'ORCHESTRATOR_INIT_FAILED',
        message: `Failed to initialize orchestrator: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
  }

  public async start(): Promise<GapAnalysisResult<void>> {
    try {
      if (!this.isInitialized) {
        const initResult = await this.initialize();
        if (!initResult.success) return initResult;
      }

      if (this.isRunning) {
        return {
          success: false,
          error: 'ORCHESTRATOR_ALREADY_RUNNING',
          message: 'Orchestrator is already running'
        };
      }

      this.isRunning = true;
      this.startWorkflowProcessor();

      this.emit('orchestrator:started');

      return {
        success: true,
        data: undefined,
        message: 'Gap Analysis Orchestrator started successfully'
      };

    } catch (error) {
      return {
        success: false,
        error: 'ORCHESTRATOR_START_FAILED',
        message: `Failed to start orchestrator: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
  }

  public async stop(): Promise<GapAnalysisResult<void>> {
    try {
      this.isRunning = false;

      // Wait for active workflows to complete or timeout
      await this.drainActiveWorkflows();

      // Stop monitoring
      if (this.monitor) {
        await this.monitor.stop();
      }

      this.emit('orchestrator:stopped');

      return {
        success: true,
        data: undefined,
        message: 'Gap Analysis Orchestrator stopped successfully'
      };

    } catch (error) {
      return {
        success: false,
        error: 'ORCHESTRATOR_STOP_FAILED',
        message: `Failed to stop orchestrator: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
  }

  // ========================================================================
  // Workflow Management
  // ========================================================================

  public async submitWorkflow(request: WorkflowRequest): Promise<GapAnalysisResult<string>> {
    try {
      if (!this.isRunning) {
        return {
          success: false,
          error: 'ORCHESTRATOR_NOT_RUNNING',
          message: 'Orchestrator is not running'
        };
      }

      // Validate request
      const validationResult = this.validateWorkflowRequest(request);
      if (!validationResult.success) {
        return validationResult as GapAnalysisResult<string>;
      }

      // Check capacity
      if (this.activeWorkflows.size >= this.config.maxConcurrentWorkflows) {
        // Add to queue
        this.workflowQueue.push(request);
        this.emit('workflow:queued', request);
        
        return {
          success: true,
          data: request.id,
          message: `Workflow ${request.id} queued (${this.workflowQueue.length} in queue)`
        };
      }

      // Start workflow immediately
      this.startWorkflow(request);

      return {
        success: true,
        data: request.id,
        message: `Workflow ${request.id} started`
      };

    } catch (error) {
      return {
        success: false,
        error: 'WORKFLOW_SUBMISSION_FAILED',
        message: `Failed to submit workflow: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
  }

  public async getWorkflowStatus(workflowId: string): Promise<GapAnalysisResult<WorkflowResult | null>> {
    try {
      // Check active workflows
      const activeWorkflow = this.activeWorkflows.get(workflowId);
      if (activeWorkflow) {
        return {
          success: true,
          data: this.createWorkflowResult(activeWorkflow, 'running'),
          message: 'Workflow is currently running'
        };
      }

      // Check completed workflows
      const completedWorkflow = this.completedWorkflows.find(w => w.id === workflowId);
      if (completedWorkflow) {
        return {
          success: true,
          data: completedWorkflow,
          message: 'Workflow completed'
        };
      }

      // Check queued workflows
      const queuedWorkflow = this.workflowQueue.find(w => w.id === workflowId);
      if (queuedWorkflow) {
        return {
          success: true,
          data: {
            id: workflowId,
            status: 'queued' as const,
            startTime: new Date(),
            endTime: new Date(),
            executionTimeMs: 0
          },
          message: 'Workflow is queued'
        };
      }

      return {
        success: true,
        data: null,
        message: 'Workflow not found'
      };

    } catch (error) {
      return {
        success: false,
        error: 'WORKFLOW_STATUS_ERROR',
        message: `Failed to get workflow status: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
  }

  public async cancelWorkflow(workflowId: string): Promise<GapAnalysisResult<void>> {
    try {
      // Remove from queue if present
      const queueIndex = this.workflowQueue.findIndex(w => w.id === workflowId);
      if (queueIndex >= 0) {
        this.workflowQueue.splice(queueIndex, 1);
        this.emit('workflow:cancelled', { id: workflowId, stage: 'queued' });
        return {
          success: true,
          data: undefined,
          message: 'Workflow removed from queue'
        };
      }

      // Cancel active workflow
      const activeWorkflow = this.activeWorkflows.get(workflowId);
      if (activeWorkflow) {
        activeWorkflow.cancelled = true;
        this.emit('workflow:cancelled', { id: workflowId, stage: 'running' });
        return {
          success: true,
          data: undefined,
          message: 'Workflow cancellation requested'
        };
      }

      return {
        success: false,
        error: 'WORKFLOW_NOT_FOUND',
        message: 'Workflow not found or already completed'
      };

    } catch (error) {
      return {
        success: false,
        error: 'WORKFLOW_CANCELLATION_FAILED',
        message: `Failed to cancel workflow: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
  }

  // ========================================================================
  // Status and Reporting
  // ========================================================================

  public getOrchestratorStatus(): OrchestratorStatus {
    const completedCount = this.completedWorkflows.length;
    const failedCount = this.completedWorkflows.filter(w => w.status === 'failed').length;
    const avgWorkflowTime = completedCount > 0 ? 
      this.completedWorkflows.reduce((sum, w) => sum + w.executionTimeMs, 0) / completedCount : 0;

    return {
      isRunning: this.isRunning,
      activeWorkflows: this.activeWorkflows.size,
      completedWorkflows: completedCount,
      failedWorkflows: failedCount,
      queuedWorkflows: this.workflowQueue.length,
      
      systemHealth: this.monitor?.getSystemHealth().overall || 'unknown' as any,
      lastHealthCheck: this.monitor?.getSystemHealth().lastUpdated || new Date(),
      
      performance: {
        avgWorkflowTime,
        successRate: completedCount > 0 ? (completedCount - failedCount) / completedCount : 1,
        throughput: completedCount > 0 ? 1000 / avgWorkflowTime : 0 // workflows per second
      }
    };
  }

  public generateStatusReport(): string {
    const status = this.getOrchestratorStatus();
    const monitoringStatus = this.monitor?.generateHealthReport() || 'Monitoring disabled';

    return `
# Gap Analysis Orchestrator Status Report
Generated: ${new Date().toISOString()}

## Orchestrator Status
- Running: ${status.isRunning}
- System Health: ${status.systemHealth.toUpperCase()}
- Active Workflows: ${status.activeWorkflows}
- Queued Workflows: ${status.queuedWorkflows}
- Completed Workflows: ${status.completedWorkflows}
- Failed Workflows: ${status.failedWorkflows}

## Performance Metrics
- Average Workflow Time: ${(status.performance.avgWorkflowTime / 1000).toFixed(1)}s
- Success Rate: ${(status.performance.successRate * 100).toFixed(1)}%
- Throughput: ${status.performance.throughput.toFixed(3)} workflows/sec

## System Monitoring
${monitoringStatus}

## Recent Workflows
${this.completedWorkflows.slice(-5).map(w => 
  `- ${w.id}: ${w.status} (${(w.executionTimeMs / 1000).toFixed(1)}s)`
).join('\n')}
`;
  }

  // ========================================================================
  // Workflow Execution Engine
  // ========================================================================

  private startWorkflowProcessor(): void {
    // Process queued workflows periodically
    const processQueue = async () => {
      if (!this.isRunning) return;

      while (this.workflowQueue.length > 0 && this.activeWorkflows.size < this.config.maxConcurrentWorkflows) {
        const request = this.workflowQueue.shift();
        if (request) {
          this.startWorkflow(request);
        }
      }

      setTimeout(processQueue, 1000); // Check every second
    };

    processQueue();
  }

  private startWorkflow(request: WorkflowRequest): void {
    const context: WorkflowContext = {
      request,
      startTime: Date.now(),
      currentStage: 'initializing',
      cancelled: false,
      results: {}
    };

    this.activeWorkflows.set(request.id, context);
    this.emit('workflow:started', request);

    // Execute workflow asynchronously
    this.executeWorkflow(context)
      .then(result => this.handleWorkflowCompletion(context, result))
      .catch(error => this.handleWorkflowError(context, error));
  }

  private async executeWorkflow(context: WorkflowContext): Promise<WorkflowResult> {
    const { request } = context;
    
    try {
      // Check if cancelled
      if (context.cancelled) {
        return this.createWorkflowResult(context, 'cancelled');
      }

      // Execute based on workflow type
      switch (request.type) {
        case 'gap_detection':
          return await this.executeGapDetectionWorkflow(context);
        case 'counterfactual_analysis':
          return await this.executeCounterfactualAnalysisWorkflow(context);
        case 'auto_tuning':
          return await this.executeAutoTuningWorkflow(context);
        case 'promotion':
          return await this.executePromotionWorkflow(context);
        case 'complete_workflow':
          return await this.executeCompleteWorkflow(context);
        default:
          throw new Error(`Unknown workflow type: ${request.type}`);
      }

    } catch (error) {
      context.error = error instanceof Error ? error.message : 'Unknown error';
      return this.createWorkflowResult(context, 'failed');
    }
  }

  private async executeCompleteWorkflow(context: WorkflowContext): Promise<WorkflowResult> {
    const { request } = context;
    const { input } = request;

    // Stage 1: Gap Detection (if needed)
    if (input.validatorJsonl && input.competitorResults) {
      context.currentStage = 'gap_detection';
      this.recordMetric(context, 'stage_start', 'gap_detection');

      const gapResult = await this.gapBoard.processValidatorOutput(
        input.validatorJsonl,
        input.competitorResults
      );

      context.results.gapDetectionResult = gapResult;
      this.recordMetric(context, 'stage_complete', 'gap_detection');

      if (!gapResult.success) {
        context.error = `Gap detection failed: ${gapResult.message}`;
        return this.createWorkflowResult(context, 'failed');
      }
    }

    // Stage 2: Slice Mining (if enabled and evaluation results available)
    if (input.enableSliceMining && input.evaluationResults) {
      context.currentStage = 'slice_mining';
      this.recordMetric(context, 'stage_start', 'slice_mining');

      const sliceResult = await this.sliceMining.performSliceMining(
        input.evaluationResults,
        input.competitorResults || []
      );

      context.results.sliceMiningResult = sliceResult;
      this.recordMetric(context, 'stage_complete', 'slice_mining');
    }

    // Stage 3: Counterfactual Analysis (for each gap)
    const gaps = context.results.gapDetectionResult?.data || [];
    if (gaps.length > 0 && input.savedAtoms) {
      context.currentStage = 'counterfactual_analysis';
      this.recordMetric(context, 'stage_start', 'counterfactual_analysis');

      // Process first gap for demo - in production, would process all
      const counterfactualResult = await this.counterfactualCBU.performCounterfactualAnalysis(
        gaps[0],
        input.savedAtoms,
        input.maxPerturbations || 50
      );

      context.results.counterfactualResult = counterfactualResult;
      this.recordMetric(context, 'stage_complete', 'counterfactual_analysis');

      if (!counterfactualResult.success) {
        context.error = `Counterfactual analysis failed: ${counterfactualResult.message}`;
        return this.createWorkflowResult(context, 'failed');
      }
    }

    // Stage 4: Auto-Tuning
    if (context.results.counterfactualResult?.data && gaps.length > 0) {
      context.currentStage = 'auto_tuning';
      this.recordMetric(context, 'stage_start', 'auto_tuning');

      const autoTuningResult = await this.autoTuning.performAutoTuning(
        gaps[0],
        context.results.counterfactualResult.data,
        input.maxTrials || 12
      );

      context.results.autoTuningResult = autoTuningResult;
      this.recordMetric(context, 'stage_complete', 'auto_tuning');

      if (!autoTuningResult.success) {
        context.error = `Auto-tuning failed: ${autoTuningResult.message}`;
        return this.createWorkflowResult(context, 'failed');
      }
    }

    // Stage 5: Promotion (if validation passes)
    if (context.results.autoTuningResult?.data && gaps.length > 0) {
      // Run validation if required
      if (this.config.requireValidationBeforePromotion && this.validator) {
        context.currentStage = 'validation';
        this.recordMetric(context, 'stage_start', 'validation');

        const validationResult = await this.validator.runFullValidation();
        context.results.validationReport = validationResult;
        this.recordMetric(context, 'stage_complete', 'validation');

        if (validationResult.overallStatus === 'failed' && this.config.autoRejectOnValidationFailure) {
          context.error = `Validation failed: ${validationResult.blockers.map(b => b.message).join(', ')}`;
          return this.createWorkflowResult(context, 'failed');
        }
      }

      context.currentStage = 'promotion';
      this.recordMetric(context, 'stage_start', 'promotion');

      const promotionResult = await this.promotionPipeline.promoteOptimizedPolicy(
        context.results.autoTuningResult.data,
        gaps[0]
      );

      context.results.promotionResult = promotionResult;
      this.recordMetric(context, 'stage_complete', 'promotion');

      if (!promotionResult.success) {
        context.error = `Promotion failed: ${promotionResult.message}`;
        return this.createWorkflowResult(context, 'failed');
      }
    }

    // Stage 6: Microsite Generation (if enabled)
    if (input.generateMicrosite && context.results.promotionResult?.data) {
      context.currentStage = 'microsite_generation';
      this.recordMetric(context, 'stage_start', 'microsite_generation');

      const micrositeResult = await this.micrositeIntegration.generateMicrositePackage(
        [context.results.promotionResult.data],
        input.competitorResults || []
      );

      context.results.micrositeResult = micrositeResult;
      this.recordMetric(context, 'stage_complete', 'microsite_generation');
    }

    return this.createWorkflowResult(context, 'completed');
  }

  private async executeGapDetectionWorkflow(context: WorkflowContext): Promise<WorkflowResult> {
    const { input } = context.request;

    if (!input.validatorJsonl || !input.competitorResults) {
      throw new Error('Gap detection requires validatorJsonl and competitorResults');
    }

    context.currentStage = 'gap_detection';
    const result = await this.gapBoard.processValidatorOutput(
      input.validatorJsonl,
      input.competitorResults
    );

    context.results.gapDetectionResult = result;
    return this.createWorkflowResult(context, result.success ? 'completed' : 'failed');
  }

  private async executeCounterfactualAnalysisWorkflow(context: WorkflowContext): Promise<WorkflowResult> {
    const { input } = context.request;

    if (!input.gapRecord || !input.savedAtoms) {
      throw new Error('Counterfactual analysis requires gapRecord and savedAtoms');
    }

    context.currentStage = 'counterfactual_analysis';
    const result = await this.counterfactualCBU.performCounterfactualAnalysis(
      input.gapRecord,
      input.savedAtoms,
      input.maxPerturbations || 50
    );

    context.results.counterfactualResult = result;
    return this.createWorkflowResult(context, result.success ? 'completed' : 'failed');
  }

  private async executeAutoTuningWorkflow(context: WorkflowContext): Promise<WorkflowResult> {
    const { input } = context.request;

    if (!input.gapRecord || !input.counterfactualAnalysis) {
      throw new Error('Auto-tuning requires gapRecord and counterfactualAnalysis');
    }

    context.currentStage = 'auto_tuning';
    const result = await this.autoTuning.performAutoTuning(
      input.gapRecord,
      input.counterfactualAnalysis,
      input.maxTrials || 12
    );

    context.results.autoTuningResult = result;
    return this.createWorkflowResult(context, result.success ? 'completed' : 'failed');
  }

  private async executePromotionWorkflow(context: WorkflowContext): Promise<WorkflowResult> {
    const { input } = context.request;

    if (!input.optimizedPolicy || !input.gapRecord) {
      throw new Error('Promotion requires optimizedPolicy and gapRecord');
    }

    context.currentStage = 'promotion';
    const result = await this.promotionPipeline.promoteOptimizedPolicy(
      input.optimizedPolicy,
      input.gapRecord
    );

    context.results.promotionResult = result;
    return this.createWorkflowResult(context, result.success ? 'completed' : 'failed');
  }

  // ========================================================================
  // Event Handling and Monitoring
  // ========================================================================

  private setupMonitoringEventHandlers(): void {
    if (!this.monitor) return;

    this.monitor.on('alert:created', (alert) => {
      this.emit('system:alert', alert);
      
      // Handle critical alerts
      if (alert.severity === 'critical') {
        this.handleCriticalAlert(alert);
      }
    });

    this.monitor.on('health:updated', (health) => {
      this.emit('system:health_updated', health);
      
      // Handle degraded health
      if (health.overall === 'critical' && this.config.enableRecovery) {
        this.handleSystemCritical(health);
      }
    });
  }

  private setupComponentEventHandlers(): void {
    // In real implementation, components would emit events that we can listen to
    // For now, we'll set up the structure for future integration
  }

  private recordMetric(context: WorkflowContext, event: string, stage: string): void {
    if (this.monitor) {
      this.monitor.recordMetric({
        timestamp: new Date(),
        component: 'orchestrator',
        operation: `${context.request.type}_${event}_${stage}`,
        duration: Date.now() - context.startTime,
        success: true,
        customMetrics: {
          workflow_id: context.request.id,
          workflow_type: context.request.type,
          stage,
          priority: context.request.priority === 'critical' ? 4 : 
                    context.request.priority === 'high' ? 3 : 
                    context.request.priority === 'medium' ? 2 : 1
        }
      });
    }
  }

  private handleCriticalAlert(alert: any): void {
    // In production, this would trigger emergency procedures
    console.error(`CRITICAL ALERT: ${alert.message}`, alert);
    this.emit('orchestrator:critical_alert', alert);
  }

  private handleSystemCritical(health: any): void {
    // In production, this would trigger recovery procedures
    console.error('SYSTEM CRITICAL: Attempting recovery', health);
    this.emit('orchestrator:system_critical', health);
  }

  // ========================================================================
  // Workflow Completion Handling
  // ========================================================================

  private handleWorkflowCompletion(context: WorkflowContext, result: WorkflowResult): void {
    this.activeWorkflows.delete(context.request.id);
    this.completedWorkflows.push(result);

    // Limit completed workflows in memory
    if (this.completedWorkflows.length > 1000) {
      this.completedWorkflows.splice(0, 100); // Remove oldest 100
    }

    this.emit('workflow:completed', result);

    // Record completion metrics
    if (this.monitor) {
      this.monitor.recordMetric({
        timestamp: new Date(),
        component: 'orchestrator',
        operation: 'workflow_completed',
        duration: result.executionTimeMs,
        success: result.status === 'completed',
        errorType: result.status === 'failed' ? 'workflow_failure' : undefined,
        customMetrics: {
          workflow_type: context.request.type,
          final_stage: context.currentStage
        }
      });
    }
  }

  private handleWorkflowError(context: WorkflowContext, error: Error): void {
    context.error = error.message;
    const result = this.createWorkflowResult(context, 'failed');
    this.handleWorkflowCompletion(context, result);
  }

  // ========================================================================
  // Helper Methods
  // ========================================================================

  private validateWorkflowRequest(request: WorkflowRequest): GapAnalysisResult<void> {
    if (!request.id) {
      return {
        success: false,
        error: 'INVALID_WORKFLOW_REQUEST',
        message: 'Workflow ID is required'
      };
    }

    if (!request.type) {
      return {
        success: false,
        error: 'INVALID_WORKFLOW_REQUEST',
        message: 'Workflow type is required'
      };
    }

    // Check for duplicate IDs
    if (this.activeWorkflows.has(request.id) || 
        this.workflowQueue.some(w => w.id === request.id) ||
        this.completedWorkflows.some(w => w.id === request.id)) {
      return {
        success: false,
        error: 'DUPLICATE_WORKFLOW_ID',
        message: 'Workflow ID already exists'
      };
    }

    return {
      success: true,
      data: undefined,
      message: 'Workflow request is valid'
    };
  }

  private createWorkflowResult(context: WorkflowContext, status: WorkflowResult['status']): WorkflowResult {
    const endTime = Date.now();
    
    return {
      id: context.request.id,
      status,
      startTime: new Date(context.startTime),
      endTime: new Date(endTime),
      executionTimeMs: endTime - context.startTime,
      ...context.results,
      error: context.error,
      warnings: context.warnings,
      recommendations: context.recommendations
    };
  }

  private async drainActiveWorkflows(): Promise<void> {
    const timeout = this.config.workflowTimeoutMs || 300000; // 5 minutes default
    const startTime = Date.now();

    while (this.activeWorkflows.size > 0 && (Date.now() - startTime) < timeout) {
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    // Force cancel any remaining workflows
    for (const [id, context] of this.activeWorkflows.entries()) {
      context.cancelled = true;
      const result = this.createWorkflowResult(context, 'cancelled');
      this.completedWorkflows.push(result);
      this.emit('workflow:force_cancelled', result);
    }

    this.activeWorkflows.clear();
  }
}

// ============================================================================
// Workflow Context (Internal)
// ============================================================================

interface WorkflowContext {
  request: WorkflowRequest;
  startTime: number;
  currentStage: string;
  cancelled: boolean;
  error?: string;
  warnings?: string[];
  recommendations?: string[];
  results: {
    gapDetectionResult?: GapAnalysisResult<GapRecord[]>;
    counterfactualResult?: GapAnalysisResult<CounterfactualAnalysis>;
    autoTuningResult?: GapAnalysisResult<OptimizedPolicy>;
    promotionResult?: GapAnalysisResult<PromotionResult>;
    sliceMiningResult?: GapAnalysisResult<SliceMiningResult>;
    micrositeResult?: GapAnalysisResult<MicrositePackage>;
    validationReport?: ValidationReport;
  };
}

// ============================================================================
// Default Configuration
// ============================================================================

export const DEFAULT_ORCHESTRATOR_CONFIG: OrchestratorConfig = {
  // Use defaults from monitoring and validation modules
  monitoring: {
    healthCheckInterval: 30000,
    metricsCollectionInterval: 10000,
    alertCheckInterval: 60000,
    maxGapDetectionTime: 30000,
    maxCounterfactualTime: 45000,
    maxAutoTuningTime: 120000,
    maxPromotionTime: 300000,
    minSuccessRate: 0.95,
    maxErrorRate: 0.05,
    maxValidationFailureRate: 0.02,
    alertChannels: [],
    metricsRetentionDays: 90,
    detailedLogsRetentionDays: 30
  },
  validation: {
    timeoutMs: 300000,
    maxRetries: 3,
    parallelExecution: true,
    minTestCoverage: 0.90,
    maxErrorRate: 0.01,
    minPerformanceScore: 0.85,
    maxRegressionThreshold: 0.05,
    goldenDatasetPath: './golden/dataset.json',
    goldenResultsPath: './golden/results/',
    regressionTolerancePct: 2.0,
    propertyTestIterations: 100,
    requiredValidations: ['unit_tests', 'integration_tests', 'performance_tests'],
    blockerSeverities: ['critical', 'error']
  },
  
  enableMonitoring: true,
  enableValidation: true,
  enableRecovery: true,
  
  maxConcurrentWorkflows: 10,
  workflowTimeoutMs: 1800000,      // 30 minutes
  retryAttempts: 3,
  backoffMultiplier: 2,
  
  requireValidationBeforePromotion: true,
  requireMonitoringBeforeDeployment: false,
  autoRejectOnValidationFailure: true,
  
  persistIntermediateResults: true,
  resultStoragePath: './workflow_results',
  enableAuditLogging: true
};

// ============================================================================
// Orchestrator Factory
// ============================================================================

export class OrchestratorFactory {
  public static createOrchestrator(config?: Partial<OrchestratorConfig>): GapAnalysisOrchestrator {
    const finalConfig = { ...DEFAULT_ORCHESTRATOR_CONFIG, ...config };
    return new GapAnalysisOrchestrator(finalConfig);
  }

  public static createDevelopmentOrchestrator(): GapAnalysisOrchestrator {
    return this.createOrchestrator({
      enableMonitoring: false,
      enableValidation: false,
      maxConcurrentWorkflows: 3,
      workflowTimeoutMs: 300000,      // 5 minutes for dev
      requireValidationBeforePromotion: false,
      autoRejectOnValidationFailure: false
    });
  }

  public static createProductionOrchestrator(): GapAnalysisOrchestrator {
    return this.createOrchestrator({
      enableMonitoring: true,
      enableValidation: true,
      enableRecovery: true,
      maxConcurrentWorkflows: 50,
      workflowTimeoutMs: 3600000,     // 1 hour for production
      requireValidationBeforePromotion: true,
      requireMonitoringBeforeDeployment: true,
      autoRejectOnValidationFailure: true,
      persistIntermediateResults: true,
      enableAuditLogging: true
    });
  }
}