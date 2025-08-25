/**
 * Service Implementations for Dependency Injection
 * Workstream A, Phase 2.3: Concrete implementations of orchestrator services
 * 
 * These implementations wrap existing functions to provide the dependency
 * injection interfaces required by the ContextOrchestrator.
 */

import type { DB } from '@lethe/sqlite';
import type { Embeddings } from '@lethe/embeddings';
import type { PlanSelection } from '../state/index.js';
import type { ConversationTurn, QueryUnderstandingConfig } from '../query-understanding/index.js';
import type { MLPrediction } from '../ml-prediction/index.js';

// Import existing implementations
import { generateHyde } from '../hyde/index.js';
import { buildContextPack } from '../summarize/index.js';
import { hybridRetrieval } from '../retrieval/index.js';
import { processQuery } from '../query-understanding/index.js';
import { getStateManager } from '../state/index.js';
import { getMLPredictor } from '../ml-prediction/index.js';

// Import service interfaces
import type {
  HydeService,
  SummarizationService,
  RetrievalService,
  QueryUnderstandingService,
  StateService,
  MLPredictionService,
  PlanSelectionService
} from './orchestrator.js';

/**
 * HyDE service implementation
 */
export class DefaultHydeService implements HydeService {
  async generateHyde(db: DB, query: string): Promise<string[]> {
    return await generateHyde(db, query);
  }
}

/**
 * Summarization service implementation
 */
export class DefaultSummarizationService implements SummarizationService {
  async buildContextPack(
    db: DB, 
    embeddings: Embeddings, 
    sessionId: string, 
    queries: string[], 
    plan: PlanSelection,
    enableLLMReranking = false,
    llmRerankConfig: any = {}
  ): Promise<any> {
    return await buildContextPack(
      db, 
      embeddings, 
      sessionId, 
      queries, 
      plan, 
      enableLLMReranking, 
      llmRerankConfig
    );
  }
}

/**
 * Retrieval service implementation
 */
export class DefaultRetrievalService implements RetrievalService {
  async hybridRetrieval(
    db: DB, 
    embeddings: Embeddings, 
    sessionId: string, 
    queries: string[], 
    plan: PlanSelection
  ): Promise<any> {
    return await hybridRetrieval(db, embeddings, sessionId, queries, plan);
  }
}

/**
 * Query understanding service implementation
 */
export class DefaultQueryUnderstandingService implements QueryUnderstandingService {
  async processQuery(
    db: DB, 
    query: string, 
    recentTurns: ConversationTurn[], 
    config?: Partial<QueryUnderstandingConfig>
  ): Promise<any> {
    return await processQuery(db, query, recentTurns, config);
  }
}

/**
 * State service implementation
 */
export class DefaultStateService implements StateService {
  private db: DB;

  constructor(db: DB) {
    this.db = db;
  }

  getRecentContext(sessionId: string): any {
    const stateManager = getStateManager(this.db);
    return stateManager.getRecentContext(sessionId);
  }
}

/**
 * ML prediction service implementation
 */
export class DefaultMLPredictionService implements MLPredictionService {
  private predictor: any;
  private initialized = false;

  constructor(config?: any) {
    this.predictor = getMLPredictor(config);
  }

  async initialize(): Promise<boolean> {
    if (!this.initialized) {
      this.initialized = await this.predictor.initialize();
    }
    return this.initialized;
  }

  async predictParameters(query: string, context: any): Promise<MLPrediction> {
    return await this.predictor.predictParameters(query, context);
  }
}

/**
 * Plan selection service implementation
 * 
 * This wraps the plan selection logic from the original pipeline
 */
export class DefaultPlanSelectionService implements PlanSelectionService {
  selectPlan(query: string, context?: any): PlanSelection {
    // This replicates the heuristic plan selection logic from the original
    const queryLower = query.toLowerCase();
    
    // Exploration patterns
    if (this.isExplorationQuery(queryLower)) {
      return {
        plan: 'explore',
        reasoning: 'Query appears to be exploratory (how/why/what/guide/tutorial patterns)',
        parameters: this.getParametersForPlan('explore')
      };
    }
    
    // Verification patterns (debugging, errors, specific issues)
    if (this.isVerificationQuery(queryLower)) {
      return {
        plan: 'verify',
        reasoning: 'Query appears to be verification-focused (error/bug/fix/debug patterns)',
        parameters: this.getParametersForPlan('verify')
      };
    }
    
    // Default to exploitation
    return {
      plan: 'exploit',
      reasoning: 'Default exploitation strategy for focused queries',
      parameters: this.getParametersForPlan('exploit')
    };
  }

  getParametersForPlan(plan: string): any {
    switch (plan) {
      case 'explore':
        return {
          alpha: 0.3,
          beta: 0.7,
          diversityWeight: 0.8,
          maxCandidates: 150
        };
      
      case 'verify':
        return {
          alpha: 0.6,
          beta: 0.4,
          diversityWeight: 0.5,
          maxCandidates: 100
        };
      
      case 'exploit':
      default:
        return {
          alpha: 0.7,
          beta: 0.3,
          diversityWeight: 0.3,
          maxCandidates: 80
        };
    }
  }

  private isExplorationQuery(query: string): boolean {
    const explorationPatterns = [
      /\b(how to|how do|how can|how should)\b/,
      /\b(what is|what are|what does|what should)\b/,
      /\b(why|when|where|which)\b/,
      /\b(guide|tutorial|learn|example|introduction)\b/,
      /\b(best practices?|recommendations?|approach|strategy)\b/,
      /\b(overview|comparison|difference|vs)\b/
    ];
    
    return explorationPatterns.some(pattern => pattern.test(query));
  }

  private isVerificationQuery(query: string): boolean {
    const verificationPatterns = [
      /\b(error|exception|fail|crash|bug|issue|problem)\b/,
      /\b(debug|troubleshoot|fix|solve|resolve)\b/,
      /\b(not working|doesn't work|broken|wrong)\b/,
      /\b(null|undefined|reference|syntax)\b/,
      /\b(stack trace|traceback|errno)\b/
    ];
    
    return verificationPatterns.some(pattern => pattern.test(query));
  }
}

/**
 * Service factory for creating default service implementations
 */
export class ServiceFactory {
  static createDefaultServices(db: DB, mlConfig?: any) {
    return {
      hyde: new DefaultHydeService(),
      summarization: new DefaultSummarizationService(),
      retrieval: new DefaultRetrievalService(),
      queryUnderstanding: new DefaultQueryUnderstandingService(),
      state: new DefaultStateService(db),
      mlPrediction: new DefaultMLPredictionService(mlConfig),
      planSelection: new DefaultPlanSelectionService()
    };
  }
}

/**
 * Convenience function to create a fully configured orchestrator
 */
export async function createDefaultOrchestrator(db: DB, config: any) {
  const services = ServiceFactory.createDefaultServices(db, config.mlConfig);
  
  // Import the orchestrator class
  const { ContextOrchestrator } = await import('./orchestrator.js');
  
  return new ContextOrchestrator(services, {
    enableHyde: config.enableHyde ?? true,
    enableSummarization: config.enableSummarization ?? true,
    enablePlanSelection: config.enablePlanSelection ?? true,
    enableQueryUnderstanding: config.enableQueryUnderstanding ?? true,
    enableMLPrediction: config.enableMLPrediction ?? false,
    queryUnderstandingConfig: config.queryUnderstandingConfig,
    mlConfig: config.mlConfig,
    llmRerankConfig: config.llmRerankConfig
  });
}

/**
 * Mock service factory for testing
 * Note: This uses a generic mock function interface to be framework-agnostic
 */
export class MockServiceFactory {
  static createMockServices(): any {
    // Use a mock function creator that works with the current test framework
    const mockFn = (returnValue: any) => {
      const fn = (() => returnValue) as any;
      fn.mockResolvedValue = (value: any) => {
        fn._mockResolvedValue = value;
        return fn;
      };
      fn.mockReturnValue = (value: any) => {
        fn._mockReturnValue = value;
        return fn;
      };
      return fn;
    };

    return {
      hyde: {
        generateHyde: mockFn(['mock hyde query'])
      },
      summarization: {
        buildContextPack: mockFn({ chunks: [], metadata: {} })
      },
      retrieval: {
        hybridRetrieval: mockFn({ candidates: [] })
      },
      queryUnderstanding: {
        processQuery: mockFn({
          canonical_query: 'processed query',
          subqueries: [],
          rewrite_success: true,
          decompose_success: false,
          llm_calls_made: 1,
          errors: []
        })
      },
      state: {
        getRecentContext: mockFn({
          lastPackContradictions: [],
          entityCount: 0
        })
      },
      mlPrediction: {
        initialize: mockFn(true),
        predictParameters: mockFn({
          alpha: 0.7,
          beta: 0.5,
          plan: 'exploit',
          prediction_time_ms: 10,
          model_loaded: true
        })
      },
      planSelection: {
        selectPlan: mockFn({
          plan: 'exploit',
          reasoning: 'Mock plan selection',
          parameters: { alpha: 0.7, beta: 0.3 }
        }),
        getParametersForPlan: mockFn({ alpha: 0.7, beta: 0.3 })
      }
    };
  }
}