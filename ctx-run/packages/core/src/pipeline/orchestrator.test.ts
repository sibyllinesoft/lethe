import { describe, it, expect, beforeEach, vi } from 'vitest';
import { ContextOrchestrator, type OrchestratorConfig, type OrchestratorRequest } from './orchestrator.js';

// Mock database and embeddings
const mockDB = {} as any;
const mockEmbeddings = {} as any;

describe('ContextOrchestrator', () => {
  let orchestrator: ContextOrchestrator;
  let mockServices: any;
  let defaultConfig: OrchestratorConfig;

  beforeEach(() => {
    // Create mock services
    mockServices = {
      hyde: {
        generateHyde: vi.fn().mockResolvedValue(['hyde query 1', 'hyde query 2'])
      },
      summarization: {
        buildContextPack: vi.fn().mockResolvedValue({ 
          chunks: [{ id: 'chunk1', text: 'test chunk' }], 
          metadata: {} 
        })
      },
      retrieval: {
        hybridRetrieval: vi.fn().mockResolvedValue({ 
          candidates: [{ id: 'result1', score: 0.9 }] 
        })
      },
      queryUnderstanding: {
        processQuery: vi.fn().mockResolvedValue({
          canonical_query: 'processed test query',
          subqueries: ['subquery 1'],
          rewrite_success: true,
          decompose_success: true,
          llm_calls_made: 2,
          errors: []
        })
      },
      state: {
        getRecentContext: vi.fn().mockReturnValue({
          lastPackContradictions: [],
          entityCount: 0
        })
      },
      mlPrediction: {
        initialize: vi.fn().mockResolvedValue(true),
        predictParameters: vi.fn().mockResolvedValue({
          alpha: 0.8,
          beta: 0.4,
          plan: 'explore',
          prediction_time_ms: 15,
          model_loaded: true
        })
      },
      planSelection: {
        selectPlan: vi.fn().mockReturnValue({
          plan: 'exploit',
          reasoning: 'Focused query pattern',
          parameters: { alpha: 0.7, beta: 0.3 }
        }),
        getParametersForPlan: vi.fn().mockReturnValue({ alpha: 0.7, beta: 0.3 })
      }
    };

    defaultConfig = {
      enableHyde: true,
      enableSummarization: true,
      enablePlanSelection: true,
      enableQueryUnderstanding: true,
      enableMLPrediction: true,
      queryUnderstandingConfig: { temperature: 0.1 },
      mlConfig: { fusion_dynamic: true, plan_learned: true },
      llmRerankConfig: { use_llm: false }
    };

    orchestrator = new ContextOrchestrator(mockServices, defaultConfig);
  });

  describe('Basic Orchestration', () => {
    it('should execute the complete pipeline successfully', async () => {
      const request: OrchestratorRequest = {
        query: 'test query',
        sessionId: 'test-session',
        db: mockDB,
        embeddings: mockEmbeddings,
        recentTurns: []
      };

      const result = await orchestrator.orchestrate(request);

      // Verify result structure
      expect(result).toBeDefined();
      expect(result.pack).toBeDefined();
      expect(result.plan).toBeDefined();
      expect(result.duration.total).toBeGreaterThan(0);
      expect(result.debug.originalQuery).toBe('test query');

      // Verify services were called
      expect(mockServices.queryUnderstanding.processQuery).toHaveBeenCalledWith(
        mockDB, 
        'test query', 
        [], 
        { temperature: 0.1 }
      );
      expect(mockServices.mlPrediction.initialize).toHaveBeenCalled();
      expect(mockServices.mlPrediction.predictParameters).toHaveBeenCalled();
      expect(mockServices.hyde.generateHyde).toHaveBeenCalledWith(mockDB, 'processed test query');
      expect(mockServices.retrieval.hybridRetrieval).toHaveBeenCalled();
      expect(mockServices.summarization.buildContextPack).toHaveBeenCalled();
    });

    it('should handle disabled features correctly', async () => {
      const minimalConfig: OrchestratorConfig = {
        enableHyde: false,
        enableSummarization: false,
        enablePlanSelection: true,
        enableQueryUnderstanding: false,
        enableMLPrediction: false
      };

      const minimalOrchestrator = new ContextOrchestrator(mockServices, minimalConfig);
      
      const request: OrchestratorRequest = {
        query: 'minimal test',
        sessionId: 'test-session',
        db: mockDB,
        embeddings: mockEmbeddings
      };

      const result = await minimalOrchestrator.orchestrate(request);

      // Verify only enabled services were called
      expect(mockServices.queryUnderstanding.processQuery).not.toHaveBeenCalled();
      expect(mockServices.mlPrediction.initialize).not.toHaveBeenCalled();
      expect(mockServices.hyde.generateHyde).not.toHaveBeenCalled();
      
      // Plan selection and retrieval should still work
      expect(mockServices.planSelection.selectPlan).toHaveBeenCalled();
      expect(mockServices.retrieval.hybridRetrieval).toHaveBeenCalled();

      // Pack should be minimal without summarization
      expect(result.pack.metadata).toHaveProperty('summarization_disabled', true);
      expect(result.hydeQueries).toBeUndefined();
      expect(result.queryUnderstanding).toBeUndefined();
      expect(result.mlPrediction).toBeUndefined();
    });
  });

  describe('Error Handling', () => {
    it('should handle query understanding failures gracefully', async () => {
      mockServices.queryUnderstanding.processQuery.mockRejectedValueOnce(
        new Error('Query understanding failed')
      );

      const request: OrchestratorRequest = {
        query: 'test query',
        sessionId: 'test-session',
        db: mockDB,
        embeddings: mockEmbeddings
      };

      const result = await orchestrator.orchestrate(request);

      expect(result.queryUnderstanding?.rewrite_success).toBe(false);
      expect(result.queryUnderstanding?.errors).toContain('Error: Query understanding failed');
      expect(result.debug.originalQuery).toBe('test query');
      
      // Pipeline should continue with original query
      expect(mockServices.hyde.generateHyde).toHaveBeenCalledWith(mockDB, 'test query');
    });

    it('should handle ML prediction failures gracefully', async () => {
      mockServices.mlPrediction.predictParameters.mockRejectedValueOnce(
        new Error('ML service unavailable')
      );

      const request: OrchestratorRequest = {
        query: 'test query',
        sessionId: 'test-session',
        db: mockDB,
        embeddings: mockEmbeddings
      };

      const result = await orchestrator.orchestrate(request);

      // Should fall back to heuristic plan selection
      expect(mockServices.planSelection.selectPlan).toHaveBeenCalled();
      expect(result.plan.plan).toBe('exploit'); // From mock
      
      // ML prediction should be undefined due to failure
      expect(result.mlPrediction).toBeUndefined();
    });

    it('should handle HyDE generation failures gracefully', async () => {
      mockServices.hyde.generateHyde.mockRejectedValueOnce(
        new Error('HyDE generation failed')
      );

      const request: OrchestratorRequest = {
        query: 'test query',
        sessionId: 'test-session',
        db: mockDB,
        embeddings: mockEmbeddings
      };

      const result = await orchestrator.orchestrate(request);

      // Should continue with original query only
      expect(result.hydeQueries).toBeUndefined();
      expect(result.debug.finalQueries).toEqual(['processed test query']);
      
      // Retrieval should still be called with just the processed query
      expect(mockServices.retrieval.hybridRetrieval).toHaveBeenCalledWith(
        mockDB,
        mockEmbeddings,
        'test-session',
        ['processed test query'],
        expect.any(Object)
      );
    });

    it('should handle summarization failures gracefully', async () => {
      mockServices.summarization.buildContextPack.mockRejectedValueOnce(
        new Error('Summarization failed')
      );

      const request: OrchestratorRequest = {
        query: 'test query',
        sessionId: 'test-session',
        db: mockDB,
        embeddings: mockEmbeddings
      };

      const result = await orchestrator.orchestrate(request);

      // Should create minimal pack from retrieval results
      expect(result.pack.chunks).toEqual([{ id: 'result1', score: 0.9 }]);
      expect(result.pack.metadata.error).toContain('Summarization failed');
    });

    it('should fail fast on retrieval failures', async () => {
      mockServices.retrieval.hybridRetrieval.mockRejectedValueOnce(
        new Error('Retrieval failed')
      );

      const request: OrchestratorRequest = {
        query: 'test query',
        sessionId: 'test-session',
        db: mockDB,
        embeddings: mockEmbeddings
      };

      await expect(orchestrator.orchestrate(request)).rejects.toThrow('Retrieval pipeline failed');
    });
  });

  describe('ML Integration', () => {
    it('should use ML-predicted plan when available', async () => {
      const request: OrchestratorRequest = {
        query: 'how to debug this error?',
        sessionId: 'test-session',
        db: mockDB,
        embeddings: mockEmbeddings
      };

      const result = await orchestrator.orchestrate(request);

      // Should use ML prediction
      expect(result.plan.plan).toBe('explore'); // From ML mock
      expect(result.plan.reasoning).toContain('ML-predicted plan');
      expect(result.mlPrediction?.predicted_plan).toBe('explore');
      expect(result.mlPrediction?.model_loaded).toBe(true);
      expect(result.debug.mlPredictionEnabled).toBe(true);
      expect(result.debug.predictedAlpha).toBe(0.8);
      expect(result.debug.predictedBeta).toBe(0.4);
    });

    it('should fall back to heuristic when ML fails', async () => {
      mockServices.mlPrediction.predictParameters.mockResolvedValueOnce({
        alpha: 0.7,
        beta: 0.5,
        plan: 'exploit',
        prediction_time_ms: 10,
        model_loaded: false // Model not loaded
      });

      const request: OrchestratorRequest = {
        query: 'test query',
        sessionId: 'test-session',
        db: mockDB,
        embeddings: mockEmbeddings
      };

      const result = await orchestrator.orchestrate(request);

      // Should fall back to heuristic
      expect(mockServices.planSelection.selectPlan).toHaveBeenCalled();
      expect(result.plan.plan).toBe('exploit'); // From heuristic mock
      expect(result.plan.reasoning).not.toContain('ML-predicted');
    });
  });

  describe('Configuration Management', () => {
    it('should allow runtime configuration updates', () => {
      const newConfig = {
        enableHyde: false,
        enableMLPrediction: false
      };

      orchestrator.updateConfig(newConfig);
      const currentConfig = orchestrator.getConfig();

      expect(currentConfig.enableHyde).toBe(false);
      expect(currentConfig.enableMLPrediction).toBe(false);
      expect(currentConfig.enableSummarization).toBe(true); // Should remain unchanged
    });

    it('should return current configuration', () => {
      const config = orchestrator.getConfig();
      
      expect(config.enableHyde).toBe(true);
      expect(config.enableSummarization).toBe(true);
      expect(config.enableQueryUnderstanding).toBe(true);
      expect(config.mlConfig?.fusion_dynamic).toBe(true);
    });
  });

  describe('Health Check', () => {
    it('should report healthy status when all services are operational', async () => {
      const health = await orchestrator.healthCheck();

      expect(health.healthy).toBe(true);
      expect(health.services.mlPrediction).toBe(true);
      expect(health.services.hyde).toBe(true);
      expect(health.details.mlPrediction).toBe('Service healthy');
    });

    it('should report unhealthy ML service but overall healthy', async () => {
      mockServices.mlPrediction.initialize.mockResolvedValueOnce(false);

      const health = await orchestrator.healthCheck();

      expect(health.healthy).toBe(false); // ML service failed
      expect(health.services.mlPrediction).toBe(false);
      expect(health.details.mlPrediction).toBe('Service unavailable, using fallbacks');
    });

    it('should handle health check failures gracefully', async () => {
      mockServices.mlPrediction.initialize.mockRejectedValueOnce(new Error('Health check failed'));

      const health = await orchestrator.healthCheck();

      expect(health.healthy).toBe(false);
      expect(health.services.error).toBe(false);
      expect(health.details.error).toContain('Health check failed');
    });
  });

  describe('Performance and Timing', () => {
    it('should track timing for all pipeline stages', async () => {
      const request: OrchestratorRequest = {
        query: 'performance test query',
        sessionId: 'test-session',
        db: mockDB,
        embeddings: mockEmbeddings
      };

      const result = await orchestrator.orchestrate(request);

      expect(result.duration.total).toBeGreaterThanOrEqual(0);
      expect(result.duration.queryUnderstanding).toBeGreaterThanOrEqual(0);
      expect(result.duration.hyde).toBeGreaterThanOrEqual(0);
      expect(result.duration.retrieval).toBeGreaterThanOrEqual(0);
      expect(result.duration.summarization).toBeGreaterThanOrEqual(0);
      expect(result.duration.mlPrediction).toBeGreaterThanOrEqual(0);
    });

    it('should not include timing for disabled stages', async () => {
      const minimalConfig: OrchestratorConfig = {
        enableHyde: false,
        enableSummarization: false,
        enablePlanSelection: true,
        enableQueryUnderstanding: false,
        enableMLPrediction: false
      };

      const minimalOrchestrator = new ContextOrchestrator(mockServices, minimalConfig);

      const request: OrchestratorRequest = {
        query: 'minimal test',
        sessionId: 'test-session',
        db: mockDB,
        embeddings: mockEmbeddings
      };

      const result = await minimalOrchestrator.orchestrate(request);

      expect(result.duration.queryUnderstanding).toBeUndefined();
      expect(result.duration.hyde).toBeUndefined();
      expect(result.duration.summarization).toBeUndefined();
      expect(result.duration.mlPrediction).toBeUndefined();
      expect(result.duration.retrieval).toBeGreaterThanOrEqual(0);
    });
  });
});