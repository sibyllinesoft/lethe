/**
 * Lethe Context Orchestrator - Dependency Injection Version
 * Workstream A, Phase 2.3: Refactor with testable dependency injection patterns
 * 
 * This refactored orchestrator enables:
 * - Comprehensive unit testing through dependency mocking
 * - Modular, replaceable service implementations
 * - Clear separation of concerns between components
 * - Graceful error handling and fallback mechanisms
 */

import type { DB } from '@lethe/sqlite';
import type { Embeddings } from '@lethe/embeddings';
import type { PlanSelection } from '../state/index.js';
import type { ConversationTurn, QueryUnderstandingConfig } from '../query-understanding/index.js';
import type { MLPrediction, MLConfig } from '../ml-prediction/index.js';

// Service interfaces for dependency injection
export interface HydeService {
  generateHyde(db: DB, query: string): Promise<string[]>;
}

export interface SummarizationService {
  buildContextPack(db: DB, embeddings: Embeddings, sessionId: string, queries: string[], plan: PlanSelection, enableLLMReranking?: boolean, llmRerankConfig?: any): Promise<any>;
}

export interface RetrievalService {
  hybridRetrieval(db: DB, embeddings: Embeddings, sessionId: string, queries: string[], plan: PlanSelection): Promise<any>;
}

export interface QueryUnderstandingService {
  processQuery(db: DB, query: string, recentTurns: ConversationTurn[], config?: Partial<QueryUnderstandingConfig>): Promise<any>;
}

export interface StateService {
  getRecentContext(sessionId: string): any;
}

export interface MLPredictionService {
  predictParameters(query: string, context: any): Promise<MLPrediction>;
  initialize(): Promise<boolean>;
}

export interface PlanSelectionService {
  selectPlan(query: string, context?: any): PlanSelection;
  getParametersForPlan(plan: string): any;
}

// Configuration for the orchestrator
export interface OrchestratorConfig {
  enableHyde: boolean;
  enableSummarization: boolean;
  enablePlanSelection: boolean;
  enableQueryUnderstanding: boolean;
  enableMLPrediction: boolean;
  queryUnderstandingConfig?: Partial<QueryUnderstandingConfig>;
  mlConfig?: Partial<MLConfig>;
  llmRerankConfig?: {
    use_llm?: boolean;
    llm_budget_ms?: number;
    llm_model?: string;
    contradiction_enabled?: boolean;
    contradiction_penalty?: number;
  };
}

// Request and response types
export interface OrchestratorRequest {
  query: string;
  sessionId: string;
  db: DB;
  embeddings: Embeddings;
  recentTurns?: ConversationTurn[];
}

export interface OrchestratorResult {
  pack: any;
  plan: PlanSelection;
  hydeQueries?: string[];
  queryUnderstanding?: {
    canonical_query?: string;
    subqueries?: string[];
    rewrite_success: boolean;
    decompose_success: boolean;
    llm_calls_made: number;
    errors: string[];
  };
  mlPrediction?: {
    alpha?: number;
    beta?: number;
    predicted_plan?: string;
    prediction_time_ms?: number;
    model_loaded?: boolean;
  };
  duration: {
    total: number;
    queryUnderstanding?: number;
    hyde?: number;
    retrieval: number;
    summarization?: number;
    mlPrediction?: number;
  };
  debug: {
    originalQuery: string;
    finalQueries: string[];
    retrievalCandidates: number;
    plan: PlanSelection;
    queryProcessingEnabled?: boolean;
    mlPredictionEnabled?: boolean;
    staticAlpha?: number;
    staticBeta?: number;
    predictedAlpha?: number;
    predictedBeta?: number;
  };
}

/**
 * Dependency-injected Context Orchestrator
 * 
 * This class orchestrates the entire context retrieval pipeline with proper
 * dependency injection, making it fully testable and modular.
 */
export class ContextOrchestrator {
  private hydeService: HydeService;
  private summarizationService: SummarizationService;
  private retrievalService: RetrievalService;
  private queryUnderstandingService: QueryUnderstandingService;
  private stateService: StateService;
  private mlPredictionService: MLPredictionService;
  private planSelectionService: PlanSelectionService;
  private config: OrchestratorConfig;

  constructor(services: {
    hyde: HydeService;
    summarization: SummarizationService;
    retrieval: RetrievalService;
    queryUnderstanding: QueryUnderstandingService;
    state: StateService;
    mlPrediction: MLPredictionService;
    planSelection: PlanSelectionService;
  }, config: OrchestratorConfig) {
    this.hydeService = services.hyde;
    this.summarizationService = services.summarization;
    this.retrievalService = services.retrieval;
    this.queryUnderstandingService = services.queryUnderstanding;
    this.stateService = services.state;
    this.mlPredictionService = services.mlPrediction;
    this.planSelectionService = services.planSelection;
    this.config = config;
  }

  /**
   * Execute the complete context orchestration pipeline
   */
  async orchestrate(request: OrchestratorRequest): Promise<OrchestratorResult> {
    const startTime = Date.now();
    const { query, sessionId, db, embeddings, recentTurns = [] } = request;
    
    console.log(`🚀 Enhanced query pipeline for: "${query}"`);

    // Initialize tracking variables
    let processedQuery = query;
    let hydeQueries: string[] = [];
    let queryUnderstandingResult: any = undefined;
    let mlPrediction: MLPrediction | undefined;
    let plan: PlanSelection;

    // Timing tracking
    const timing = {
      queryUnderstanding: 0,
      hyde: 0,
      retrieval: 0,
      summarization: 0,
      mlPrediction: 0
    };

    try {
      // Step 1: Query Understanding
      if (this.config.enableQueryUnderstanding) {
        const stepStart = Date.now();
        
        try {
          queryUnderstandingResult = await this.queryUnderstandingService.processQuery(
            db, 
            query, 
            recentTurns, 
            this.config.queryUnderstandingConfig
          );
          
          // Use rewritten query if successful
          processedQuery = queryUnderstandingResult.canonical_query || query;
          
          console.log(`🔄 Query understanding: ${queryUnderstandingResult.rewrite_success ? 'rewritten' : 'unchanged'}, ${queryUnderstandingResult.subqueries?.length || 0} subqueries`);
          
        } catch (error) {
          console.warn(`Query understanding failed: ${error}`);
          queryUnderstandingResult = {
            canonical_query: query,
            subqueries: [],
            rewrite_success: false,
            decompose_success: false,
            llm_calls_made: 0,
            errors: [String(error)]
          };
        }
        
        timing.queryUnderstanding = Date.now() - stepStart;
      }

      // Step 2: ML-Enhanced Plan Selection
      if (this.config.enableMLPrediction) {
        const stepStart = Date.now();
        
        try {
          await this.mlPredictionService.initialize();
          
          const sessionContext = this.stateService.getRecentContext(sessionId);
          const mlContext = {
            contradictions: sessionContext.lastPackContradictions?.length || 0,
            entity_overlap: sessionContext.entityCount > 0 ? 0.5 : 0.1
          };
          
          mlPrediction = await this.mlPredictionService.predictParameters(processedQuery, mlContext);
          
          if (mlPrediction.model_loaded && mlPrediction.plan) {
            plan = {
              plan: mlPrediction.plan,
              reasoning: `ML-predicted plan (${mlPrediction.prediction_time_ms.toFixed(1)}ms)`,
              parameters: this.planSelectionService.getParametersForPlan(mlPrediction.plan)
            };
            console.log(`🤖 ML predicted plan: ${mlPrediction.plan}`);
          } else {
            plan = this.planSelectionService.selectPlan(processedQuery);
            console.log(`📊 Heuristic plan fallback: ${plan.plan}`);
          }
          
        } catch (error) {
          console.warn(`ML prediction failed: ${error}`);
          plan = this.planSelectionService.selectPlan(processedQuery);
        }
        
        timing.mlPrediction = Date.now() - stepStart;
        
      } else if (this.config.enablePlanSelection) {
        plan = this.planSelectionService.selectPlan(processedQuery);
        console.log(`📊 Plan selected: ${plan.plan} - ${plan.reasoning}`);
      } else {
        // Default plan
        plan = {
          plan: 'exploit',
          reasoning: 'Plan selection disabled, using default exploit strategy',
          parameters: this.planSelectionService.getParametersForPlan('exploit')
        };
      }

      // Step 3: HyDE Query Enhancement
      let finalQueries = [processedQuery];
      
      if (this.config.enableHyde) {
        const stepStart = Date.now();
        
        try {
          hydeQueries = await this.hydeService.generateHyde(db, processedQuery);
          
          if (hydeQueries.length > 0) {
            finalQueries = [...finalQueries, ...hydeQueries];
            console.log(`💡 Generated ${hydeQueries.length} HyDE queries`);
          }
          
        } catch (error) {
          console.warn(`HyDE generation failed: ${error}`);
        }
        
        timing.hyde = Date.now() - stepStart;
      }

      // Step 4: Retrieval
      const retrievalStart = Date.now();
      let retrievalResults: any;
      
      try {
        retrievalResults = await this.retrievalService.hybridRetrieval(
          db, 
          embeddings, 
          sessionId, 
          finalQueries, 
          plan
        );
        
        console.log(`🔍 Retrieved ${retrievalResults?.candidates?.length || 0} candidates`);
        
      } catch (error) {
        console.error(`Retrieval failed: ${error}`);
        throw new Error(`Retrieval pipeline failed: ${error}`);
      }
      
      timing.retrieval = Date.now() - retrievalStart;

      // Step 5: Summarization/Context Packing
      let pack: any;
      
      if (this.config.enableSummarization) {
        const stepStart = Date.now();
        
        try {
          pack = await this.summarizationService.buildContextPack(
            db,
            embeddings,
            sessionId,
            finalQueries,
            plan,
            this.config.llmRerankConfig?.use_llm,
            this.config.llmRerankConfig
          );
          
          console.log(`📦 Context pack built with ${pack?.chunks?.length || 0} chunks`);
          
        } catch (error) {
          console.warn(`Summarization failed: ${error}`);
          // Create minimal pack from retrieval results
          pack = {
            chunks: retrievalResults?.candidates || [],
            metadata: { error: String(error) }
          };
        }
        
        timing.summarization = Date.now() - stepStart;
        
      } else {
        // Create minimal pack from retrieval results
        pack = {
          chunks: retrievalResults?.candidates || [],
          metadata: { summarization_disabled: true }
        };
      }

      // Build final result
      const totalTime = Date.now() - startTime;
      
      const result: OrchestratorResult = {
        pack,
        plan,
        hydeQueries: hydeQueries.length > 0 ? hydeQueries : undefined,
        queryUnderstanding: queryUnderstandingResult,
        mlPrediction: mlPrediction ? {
          alpha: mlPrediction.alpha,
          beta: mlPrediction.beta,
          predicted_plan: mlPrediction.plan,
          prediction_time_ms: mlPrediction.prediction_time_ms,
          model_loaded: mlPrediction.model_loaded
        } : undefined,
        duration: {
          total: totalTime,
          queryUnderstanding: this.config.enableQueryUnderstanding ? timing.queryUnderstanding : undefined,
          hyde: this.config.enableHyde ? timing.hyde : undefined,
          retrieval: timing.retrieval,
          summarization: this.config.enableSummarization ? timing.summarization : undefined,
          mlPrediction: this.config.enableMLPrediction ? timing.mlPrediction : undefined
        },
        debug: {
          originalQuery: query,
          finalQueries,
          retrievalCandidates: retrievalResults?.candidates?.length || 0,
          plan,
          queryProcessingEnabled: this.config.enableQueryUnderstanding,
          mlPredictionEnabled: this.config.enableMLPrediction,
          staticAlpha: plan.parameters?.alpha,
          staticBeta: plan.parameters?.beta,
          predictedAlpha: mlPrediction?.alpha,
          predictedBeta: mlPrediction?.beta
        }
      };

      console.log(`✅ Pipeline completed in ${totalTime}ms`);
      return result;

    } catch (error) {
      console.error(`❌ Pipeline failed: ${error}`);
      throw error;
    }
  }

  /**
   * Update configuration at runtime
   */
  updateConfig(newConfig: Partial<OrchestratorConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * Get current configuration
   */
  getConfig(): OrchestratorConfig {
    return { ...this.config };
  }

  /**
   * Health check - verify all services are operational
   */
  async healthCheck(): Promise<{
    healthy: boolean;
    services: Record<string, boolean>;
    details: Record<string, string>;
  }> {
    const services: Record<string, boolean> = {};
    const details: Record<string, string> = {};

    try {
      // Test ML service if enabled
      if (this.config.enableMLPrediction) {
        const mlHealthy = await this.mlPredictionService.initialize();
        services.mlPrediction = mlHealthy;
        details.mlPrediction = mlHealthy ? 'Service healthy' : 'Service unavailable, using fallbacks';
      } else {
        services.mlPrediction = true;
        details.mlPrediction = 'Disabled';
      }

      // Other services are assumed healthy (they're typically local/in-process)
      services.hyde = true;
      services.summarization = true;
      services.retrieval = true;
      services.queryUnderstanding = true;
      services.state = true;
      services.planSelection = true;

      const allHealthy = Object.values(services).every(s => s);

      return {
        healthy: allHealthy,
        services,
        details
      };

    } catch (error) {
      return {
        healthy: false,
        services: { error: false },
        details: { error: String(error) }
      };
    }
  }
}