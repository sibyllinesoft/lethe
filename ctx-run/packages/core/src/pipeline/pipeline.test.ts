import { describe, it, expect, vi, beforeEach } from 'vitest';
import { enhancedQuery, diagnoseEnhancedPipeline } from './index.js';
import type { EnhancedQueryOptions } from './index.js';

// Mock all dependencies
vi.mock('@lethe/sqlite', () => ({
  getConfig: vi.fn(() => ({ version: '1.0.0' }))
}));

vi.mock('../state/index.js', () => ({
  getStateManager: vi.fn(() => ({
    selectPlan: vi.fn(() => ({
      plan: 'exploit',
      reasoning: 'Default plan for testing',
      parameters: { hyde_k: 3, beta: 0.5, granularity: 'medium', k_final: 10 }
    })),
    updateSessionState: vi.fn(),
    getAllSessions: vi.fn(() => ['test-session']),
    getRecentContext: vi.fn(() => ({
      lastPackContradictions: [],
      entityCount: 0
    }))
  }))
}));

vi.mock('../hyde/index.js', () => ({
  generateHyde: vi.fn().mockResolvedValue({
    queries: ['test query', 'hyde variant 1', 'hyde variant 2']
  })
}));

vi.mock('../summarize/index.js', () => ({
  buildContextPack: vi.fn().mockResolvedValue({
    id: 'test-pack',
    session_id: 'test-session',
    query: 'test query',
    summary: 'Test summary',
    key_entities: ['entity1'],
    claims: ['claim1'],
    contradictions: [],
    chunks: [{ id: 'chunk1', score: 0.8, kind: 'text', text: 'test content' }],
    citations: [{ id: 1, chunk_id: 'chunk1', relevance: 0.8 }]
  })
}));

vi.mock('../retrieval/index.js', () => ({
  hybridRetrieval: vi.fn().mockResolvedValue([
    { docId: 'doc1', score: 0.9, text: 'test content 1', kind: 'text' },
    { docId: 'doc2', score: 0.8, text: 'test content 2', kind: 'text' }
  ])
}));

vi.mock('../query-understanding/index.js', () => ({
  processQuery: vi.fn().mockResolvedValue({
    canonical_query: 'processed test query',
    subqueries: ['subquery1', 'subquery2'],
    rewrite_success: true,
    decompose_success: true,
    llm_calls_made: 2,
    errors: []
  })
}));

vi.mock('../ml-prediction/index.js', () => ({
  getMLPredictor: vi.fn(() => ({
    predictParameters: vi.fn().mockResolvedValue({
      alpha: 0.6,
      beta: 0.4,
      plan: 'explore',
      prediction_time_ms: 150,
      model_loaded: true
    })
  }))
}));

vi.mock('../ollama/index.js', () => ({
  testOllamaConnection: vi.fn().mockResolvedValue({
    available: true,
    models: ['llama3.2:1b', 'llama3.2:3b']
  })
}));

describe('Pipeline Module', () => {
  const mockDB = {
    prepare: vi.fn(() => ({
      all: vi.fn(() => []),
      get: vi.fn(),
      run: vi.fn()
    })),
    exec: vi.fn()
  } as any;

  const mockEmbeddings = {
    embed: vi.fn().mockResolvedValue([0.1, 0.2, 0.3]),
    embedBatch: vi.fn().mockResolvedValue([[0.1, 0.2, 0.3]])
  } as any;

  const baseOptions: EnhancedQueryOptions = {
    db: mockDB,
    embeddings: mockEmbeddings,
    sessionId: 'test-session'
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  describe('enhancedQuery', () => {
    it('should execute basic query pipeline with default options', async () => {
      const result = await enhancedQuery('test query', baseOptions);

      expect(result).toHaveProperty('pack');
      expect(result).toHaveProperty('plan');
      expect(result).toHaveProperty('duration');
      expect(result).toHaveProperty('debug');
      expect(result.debug.originalQuery).toBe('test query');
      expect(result.plan.plan).toBe('exploit');
    });

    it('should handle HyDE generation when enabled', async () => {
      const options = { ...baseOptions, enableHyde: true };
      const result = await enhancedQuery('test query', options);

      expect(result.hydeQueries).toBeDefined();
      expect(result.hydeQueries).toHaveLength(3);
      expect(result.debug.finalQueries).toEqual(['test query', 'hyde variant 1', 'hyde variant 2']);
    });

    it('should skip HyDE when disabled', async () => {
      const options = { ...baseOptions, enableHyde: false, enableQueryUnderstanding: false };
      const result = await enhancedQuery('test query', options);

      expect(result.hydeQueries).toBeUndefined();
      expect(result.debug.finalQueries).toEqual(['test query']);
    });

    it('should handle query understanding when enabled', async () => {
      const options = { 
        ...baseOptions, 
        enableQueryUnderstanding: true,
        queryUnderstandingConfig: { enabled: true },
        recentTurns: []
      };
      
      const result = await enhancedQuery('test query', options);

      expect(result.queryUnderstanding).toBeDefined();
      expect(result.queryUnderstanding?.canonical_query).toBe('processed test query');
      expect(result.queryUnderstanding?.rewrite_success).toBe(true);
      expect(result.duration).toHaveProperty('total');
    });

    it('should handle ML prediction when enabled', async () => {
      const options = { 
        ...baseOptions, 
        enableMLPrediction: true,
        mlConfig: { fusion_dynamic: true, plan_learned: true }
      };
      
      const result = await enhancedQuery('test query', options);

      expect(result.mlPrediction).toBeDefined();
      expect(result.mlPrediction?.predicted_plan).toBe('explore');
      expect(result.mlPrediction?.model_loaded).toBe(true);
      expect(result.duration).toHaveProperty('total');
    });

    it('should handle LLM reranking configuration', async () => {
      const options = { 
        ...baseOptions,
        llmRerankConfig: {
          use_llm: true,
          llm_budget_ms: 2000,
          llm_model: 'llama3.2:3b',
          contradiction_enabled: true,
          contradiction_penalty: 0.2
        }
      };
      
      const result = await enhancedQuery('test query', options);
      
      expect(result.pack).toBeDefined();
      expect(result.debug.finalQueries).toEqual(['test query', 'hyde variant 1', 'hyde variant 2']);
    });

    it('should handle summarization failure gracefully', async () => {
      const { buildContextPack } = await import('../summarize/index.js');
      vi.mocked(buildContextPack).mockRejectedValueOnce(new Error('Summarization failed'));

      const result = await enhancedQuery('test query', baseOptions);

      expect(result.pack).toBeDefined();
      expect(result.pack.summary).toContain('Retrieved');
    });

    it('should handle plan selection disabled', async () => {
      const options = { ...baseOptions, enablePlanSelection: false };
      const result = await enhancedQuery('test query', options);

      expect(result.plan.plan).toBe('exploit');
      expect(result.plan.reasoning).toBe('Plan selection disabled');
    });

    it('should handle query understanding failure', async () => {
      const { processQuery } = await import('../query-understanding/index.js');
      vi.mocked(processQuery).mockRejectedValueOnce(new Error('Query understanding failed'));

      const options = { ...baseOptions, enableQueryUnderstanding: true };
      const result = await enhancedQuery('test query', options);

      expect(result.queryUnderstanding).toBeDefined();
      expect(result.queryUnderstanding?.rewrite_success).toBe(false);
      expect(result.queryUnderstanding?.errors).toHaveLength(1);
    });

    it('should handle ML prediction failure gracefully', async () => {
      const { getMLPredictor } = await import('../ml-prediction/index.js');
      const mockPredictor = {
        predictParameters: vi.fn().mockRejectedValue(new Error('ML prediction failed'))
      };
      vi.mocked(getMLPredictor).mockReturnValue(mockPredictor);

      const options = { 
        ...baseOptions, 
        enableMLPrediction: true,
        mlConfig: { plan_learned: true }
      };
      
      const result = await enhancedQuery('test query', options);

      expect(result.plan.plan).toBe('exploit'); // Falls back to heuristic
    });

    it('should measure execution duration correctly', async () => {
      const result = await enhancedQuery('test query', baseOptions);

      expect(result.duration.total).toBeGreaterThanOrEqual(0);
      expect(result.duration.retrieval).toBeGreaterThanOrEqual(0);
      expect(typeof result.duration.total).toBe('number');
    });

    it('should handle different plan types', async () => {
      const { getStateManager } = await import('../state/index.js');
      const mockStateManager = {
        selectPlan: vi.fn(() => ({
          plan: 'verify',
          reasoning: 'Verification plan selected',
          parameters: { hyde_k: 5, beta: 0.4, granularity: 'tight', k_final: 8 }
        })),
        updateSessionState: vi.fn(),
        getAllSessions: vi.fn(() => ['test-session']),
        getRecentContext: vi.fn(() => ({ lastPackContradictions: [], entityCount: 0 }))
      };
      vi.mocked(getStateManager).mockReturnValue(mockStateManager);

      const result = await enhancedQuery('test query', baseOptions);

      expect(result.plan.plan).toBe('verify');
      expect(result.plan.parameters.granularity).toBe('tight');
    });
  });

  describe('diagnoseEnhancedPipeline', () => {
    it('should return diagnostic information', async () => {
      const result = await diagnoseEnhancedPipeline(mockDB);

      expect(result).toHaveProperty('ollama');
      expect(result).toHaveProperty('state');
      expect(result).toHaveProperty('config');
      expect(result.ollama.available).toBe(true);
      expect(result.ollama.models).toEqual(['llama3.2:1b', 'llama3.2:3b']);
      expect(result.state.sessions).toEqual(['test-session']);
    });

    it('should handle Ollama connection failure', async () => {
      const { testOllamaConnection } = await import('../ollama/index.js');
      vi.mocked(testOllamaConnection).mockResolvedValueOnce({
        available: false,
        models: []
      });

      const result = await diagnoseEnhancedPipeline(mockDB);

      expect(result.ollama.available).toBe(false);
      expect(result.ollama.models).toEqual([]);
    });
  });

  describe('Helper Functions', () => {
    it('should create fallback pack when summarization fails', async () => {
      const { buildContextPack } = await import('../summarize/index.js');
      vi.mocked(buildContextPack).mockRejectedValueOnce(new Error('Summarization error'));

      const result = await enhancedQuery('test query', baseOptions);

      expect(result.pack).toBeDefined();
      expect(result.pack.query).toBe('processed test query');
      expect(result.pack.session_id).toBe('test-session');
      expect(result.pack.chunks).toHaveLength(2);
      expect(result.pack.summary).toContain('Retrieved 2 relevant chunks');
    });

    it('should handle empty retrieval results', async () => {
      // Mock both hybridRetrieval and buildContextPack for empty results
      const { hybridRetrieval } = await import('../retrieval/index.js');
      const { buildContextPack } = await import('../summarize/index.js');
      
      vi.mocked(hybridRetrieval).mockResolvedValueOnce([]);
      vi.mocked(buildContextPack).mockResolvedValueOnce({
        id: 'empty-pack',
        session_id: 'test-session',
        query: 'processed test query',
        summary: 'Retrieved 0 relevant chunks using hybrid search',
        key_entities: [],
        claims: [],
        contradictions: [],
        chunks: [],
        citations: []
      });

      const result = await enhancedQuery('test query', baseOptions);

      expect(result.pack.chunks).toHaveLength(0);
      expect(result.debug.retrievalCandidates).toBe(0);
    });
  });
});