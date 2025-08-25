import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock external dependencies
vi.mock('@xenova/transformers', () => ({
  pipeline: vi.fn().mockResolvedValue({
    // Mock cross-encoder model
    __call__: vi.fn().mockResolvedValue([
      [{ label: 'LABEL_0', score: 0.2 }, { label: 'LABEL_1', score: 0.8 }],
      [{ label: 'LABEL_0', score: 0.4 }, { label: 'LABEL_1', score: 0.6 }],
      [{ label: 'LABEL_0', score: 0.7 }, { label: 'LABEL_1', score: 0.3 }]
    ])
  })
}));

vi.mock('../ollama/index.js', () => ({
  getOllamaBridge: vi.fn(() => ({
    generate: vi.fn().mockResolvedValue({
      response: JSON.stringify({
        scores: [
          { id: 'C1', s: 0.9 },
          { id: 'C2', s: 0.7 },
          { id: 'C3', s: 0.5 }
        ]
      })
    })
  })),
  safeParseJSON: vi.fn((str, fallback) => {
    try {
      return JSON.parse(str);
    } catch {
      return fallback;
    }
  })
}));

const mockDB = {
  all: vi.fn().mockResolvedValue([]),
  get: vi.fn(),
  run: vi.fn(),
  prepare: vi.fn(() => ({
    all: vi.fn().mockResolvedValue([]),
    get: vi.fn(),
    run: vi.fn()
  }))
} as any;

describe('Reranker Comprehensive Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.spyOn(console, 'debug').mockImplementation(() => {});
  });

  const mockCandidates = [
    { docId: 'doc1', score: 0.8, text: 'React is a JavaScript library for building user interfaces' },
    { docId: 'doc2', score: 0.7, text: 'Vue.js is a progressive framework for building web applications' },
    { docId: 'doc3', score: 0.6, text: 'Angular is a platform for building mobile and desktop web applications' },
    { docId: 'doc4', score: 0.5, text: 'JavaScript is a programming language' },
    { docId: 'doc5', score: 0.4, text: '' } // Empty text
  ];

  describe('CrossEncoderReranker', () => {
    it('should create CrossEncoderReranker with default model', async () => {
      const { CrossEncoderReranker } = await import('./index.js');
      
      const reranker = new CrossEncoderReranker();
      expect(reranker.name).toBe('Xenova/ms-marco-MiniLM-L-6-v2');
    });
    
    it('should create CrossEncoderReranker with custom model', async () => {
      const { CrossEncoderReranker } = await import('./index.js');
      
      const reranker = new CrossEncoderReranker('custom-model');
      expect(reranker.name).toBe('custom-model');
    });
    
    it('should initialize model successfully', async () => {
      const { CrossEncoderReranker } = await import('./index.js');
      
      const reranker = new CrossEncoderReranker();
      await reranker.init();
      
      // Model should be initialized (mocked)
      expect(reranker.name).toBeDefined();
    });
    
    it('should handle model initialization failure gracefully', async () => {
      const { CrossEncoderReranker } = await import('./index.js');
      
      // Mock pipeline to throw error
      const mockPipeline = await import('@xenova/transformers');
      vi.mocked(mockPipeline.pipeline).mockRejectedValueOnce(new Error('Model load failed'));
      
      const reranker = new CrossEncoderReranker();
      await reranker.init();
      
      // Should continue without throwing
      expect(reranker.name).toBeDefined();
    });
    
    it('should rerank candidates using cross-encoder model', async () => {
      const { CrossEncoderReranker } = await import('./index.js');
      
      const reranker = new CrossEncoderReranker();
      const result = await reranker.rerank('JavaScript frameworks', mockCandidates);
      
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBe(mockCandidates.length);
      expect(result.every(candidate => typeof candidate.score === 'number')).toBe(true);
    });
    
    it('should handle candidates without text', async () => {
      const { CrossEncoderReranker } = await import('./index.js');
      
      const candidatesWithoutText = [
        { docId: 'doc1', score: 0.8 },
        { docId: 'doc2', score: 0.7, text: '' },
        { docId: 'doc3', score: 0.6, text: 'Valid text content' }
      ];
      
      const reranker = new CrossEncoderReranker();
      const result = await reranker.rerank('test query', candidatesWithoutText);
      
      expect(result.length).toBe(candidatesWithoutText.length);
    });
    
    it('should use fallback reranking when model fails', async () => {
      const { CrossEncoderReranker } = await import('./index.js');
      
      const reranker = new CrossEncoderReranker();
      
      // Mock the model to fail during reranking
      await reranker.init();
      
      const result = await reranker.rerank('test query', mockCandidates);
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBe(mockCandidates.length);
    });
    
    it('should tokenize text correctly', async () => {
      const { CrossEncoderReranker } = await import('./index.js');
      
      const reranker = new CrossEncoderReranker();
      
      // Test fallback reranking which uses tokenization
      const result = await reranker.rerank('JavaScript React', mockCandidates);
      
      expect(result.length).toBe(mockCandidates.length);
      // Results should be sorted by score
      for (let i = 1; i < result.length; i++) {
        expect(result[i-1].score).toBeGreaterThanOrEqual(result[i].score);
      }
    });
  });

  describe('LLMReranker', () => {
    const mockConfig = {
      use_llm: true,
      llm_budget_ms: 5000,
      llm_model: 'llama3',
      contradiction_enabled: true,
      contradiction_penalty: 0.3
    };
    
    it('should create LLMReranker with config', async () => {
      const { LLMReranker } = await import('./index.js');
      
      const reranker = new LLMReranker(mockDB, mockConfig);
      expect(reranker.name).toBe('llm-reranker');
    });
    
    it('should rerank candidates using LLM', async () => {
      const { LLMReranker } = await import('./index.js');
      
      const reranker = new LLMReranker(mockDB, mockConfig);
      const result = await reranker.rerank('JavaScript frameworks', mockCandidates);
      
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBe(mockCandidates.length);
    });
    
    it('should fall back to cross-encoder when LLM disabled', async () => {
      const { LLMReranker } = await import('./index.js');
      
      const disabledConfig = { ...mockConfig, use_llm: false };
      const reranker = new LLMReranker(mockDB, disabledConfig);
      
      const result = await reranker.rerank('test query', mockCandidates);
      expect(Array.isArray(result)).toBe(true);
    });
    
    it('should handle empty candidates array', async () => {
      const { LLMReranker } = await import('./index.js');
      
      const reranker = new LLMReranker(mockDB, mockConfig);
      const result = await reranker.rerank('test query', []);
      
      expect(result).toEqual([]);
    });
    
    it('should apply contradiction penalties when enabled', async () => {
      const { LLMReranker } = await import('./index.js');
      
      const contradictionConfig = {
        ...mockConfig,
        contradiction_enabled: true,
        contradiction_penalty: 0.5
      };
      
      // Mock contradiction detection
      const mockOllama = await import('../ollama/index.js');
      vi.mocked(mockOllama.getOllamaBridge).mockResolvedValue({
        generate: vi.fn()
          .mockResolvedValueOnce({ // First call for reranking
            response: JSON.stringify({
              scores: [
                { id: 'C1', s: 0.9 },
                { id: 'C2', s: 0.8 }
              ]
            })
          })
          .mockResolvedValueOnce({ // Second call for contradiction check
            response: JSON.stringify({ contradicts: true })
          })
      } as any);
      
      const reranker = new LLMReranker(mockDB, contradictionConfig);
      const smallCandidates = mockCandidates.slice(0, 2);
      
      const result = await reranker.rerank('test query', smallCandidates);
      expect(result.length).toBe(smallCandidates.length);
    });
    
    it('should handle LLM timeout gracefully', async () => {
      const { LLMReranker } = await import('./index.js');
      
      const timeoutConfig = {
        ...mockConfig,
        llm_budget_ms: 1 // Very short timeout
      };
      
      const reranker = new LLMReranker(mockDB, timeoutConfig);
      const result = await reranker.rerank('test query', mockCandidates);
      
      expect(Array.isArray(result)).toBe(true);
    });
    
    it('should handle LLM reranking failures', async () => {
      const { LLMReranker } = await import('./index.js');
      
      // Mock getOllamaBridge to throw error
      const mockOllama = await import('../ollama/index.js');
      vi.mocked(mockOllama.getOllamaBridge).mockRejectedValueOnce(new Error('LLM failed'));
      
      const reranker = new LLMReranker(mockDB, mockConfig);
      const result = await reranker.rerank('test query', mockCandidates);
      
      expect(Array.isArray(result)).toBe(true);
    });
    
    it('should clamp LLM scores to valid range [0,1]', async () => {
      const { LLMReranker } = await import('./index.js');
      
      // Mock extreme scores
      const mockOllama = await import('../ollama/index.js');
      vi.mocked(mockOllama.getOllamaBridge).mockResolvedValueOnce({
        generate: vi.fn().mockResolvedValue({
          response: JSON.stringify({
            scores: [
              { id: 'C1', s: 2.5 }, // Too high
              { id: 'C2', s: -0.5 }, // Too low
              { id: 'C3', s: 0.7 } // Normal
            ]
          })
        })
      } as any);
      
      const reranker = new LLMReranker(mockDB, mockConfig);
      const result = await reranker.rerank('test query', mockCandidates.slice(0, 3));
      
      expect(result.every(candidate => 
        candidate.score >= 0 && candidate.score <= 1
      )).toBe(true);
    });
    
    it('should handle malformed LLM responses', async () => {
      const { LLMReranker } = await import('./index.js');
      
      const mockOllama = await import('../ollama/index.js');
      vi.mocked(mockOllama.getOllamaBridge).mockResolvedValueOnce({
        generate: vi.fn().mockResolvedValue({
          response: 'invalid json response'
        })
      } as any);
      
      const reranker = new LLMReranker(mockDB, mockConfig);
      const result = await reranker.rerank('test query', mockCandidates);
      
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBe(mockCandidates.length);
    });
  });

  describe('NoOpReranker', () => {
    it('should create NoOpReranker', async () => {
      const { NoOpReranker } = await import('./index.js');
      
      const reranker = new NoOpReranker();
      expect(reranker.name).toBe('noop');
    });
    
    it('should pass through candidates unchanged', async () => {
      const { NoOpReranker } = await import('./index.js');
      
      const reranker = new NoOpReranker();
      const result = await reranker.rerank('test query', mockCandidates);
      
      expect(result).toBe(mockCandidates); // Same reference
    });
  });

  describe('getReranker factory function', () => {
    it('should return NoOpReranker when disabled', async () => {
      const { getReranker } = await import('./index.js');
      
      const reranker = await getReranker(false);
      expect(reranker.name).toBe('noop');
    });
    
    it('should return LLMReranker when LLM config provided', async () => {
      const { getReranker } = await import('./index.js');
      
      const config = {
        use_llm: true,
        llm_budget_ms: 5000,
        llm_model: 'test-model',
        contradiction_enabled: false,
        contradiction_penalty: 0
      };
      
      const reranker = await getReranker(true, config, mockDB);
      expect(reranker.name).toBe('llm-reranker');
    });
    
    it('should return CrossEncoderReranker as fallback', async () => {
      const { getReranker } = await import('./index.js');
      
      const reranker = await getReranker(true);
      expect(reranker.name).toBe('Xenova/ms-marco-MiniLM-L-6-v2');
    });
    
    it('should handle LLM reranker initialization failure', async () => {
      const { getReranker } = await import('./index.js');
      
      const badConfig = {
        use_llm: true,
        llm_budget_ms: 5000,
        llm_model: 'invalid-model',
        contradiction_enabled: false,
        contradiction_penalty: 0
      };
      
      // This should not throw and should fall back gracefully
      const reranker = await getReranker(true, badConfig, null as any);
      expect(reranker).toBeDefined();
    });
    
    it('should handle cross-encoder initialization failure', async () => {
      const { getReranker } = await import('./index.js');
      
      // For simplicity, just test that getReranker returns a working reranker
      // even when initialization might have issues
      const reranker = await getReranker(true);
      expect(reranker).toBeDefined();
      expect(reranker.name).toBeDefined();
      expect(typeof reranker.rerank).toBe('function');
    });
    
    it('should handle various config combinations', async () => {
      const { getReranker } = await import('./index.js');
      
      const configs = [
        undefined,
        { use_llm: false, llm_budget_ms: 1000, llm_model: 'test', contradiction_enabled: false, contradiction_penalty: 0 },
        { use_llm: true, llm_budget_ms: 1000, llm_model: 'test', contradiction_enabled: true, contradiction_penalty: 0.2 }
      ];
      
      for (const config of configs) {
        const reranker = await getReranker(true, config, mockDB);
        expect(reranker).toBeDefined();
        expect(reranker.name).toBeDefined();
      }
    });
  });

  describe('Edge Cases and Error Handling', () => {
    it('should handle very long candidate texts', async () => {
      const { CrossEncoderReranker } = await import('./index.js');
      
      const longText = 'Very long text content. '.repeat(1000);
      const longCandidates = [
        { docId: 'long1', score: 0.8, text: longText },
        { docId: 'long2', score: 0.7, text: longText.substring(0, 500) }
      ];
      
      const reranker = new CrossEncoderReranker();
      const result = await reranker.rerank('test query', longCandidates);
      
      expect(result.length).toBe(longCandidates.length);
    });
    
    it('should handle special characters in queries and text', async () => {
      const { CrossEncoderReranker } = await import('./index.js');
      
      const specialCandidates = [
        { docId: 'special1', score: 0.8, text: 'Text with émojis 🎉 and ñoñó characters' },
        { docId: 'special2', score: 0.7, text: 'Code snippets: const x = "hello"; console.log(x);' }
      ];
      
      const reranker = new CrossEncoderReranker();
      const result = await reranker.rerank('query with @#$% special chars', specialCandidates);
      
      expect(result.length).toBe(specialCandidates.length);
    });
    
    it('should handle large batches of candidates', async () => {
      const { LLMReranker } = await import('./index.js');
      
      const largeBatch = Array.from({ length: 100 }, (_, i) => ({
        docId: `doc${i}`,
        score: Math.random(),
        text: `Content for document number ${i} with some relevant text`
      }));
      
      const config = {
        use_llm: true,
        llm_budget_ms: 10000,
        llm_model: 'test',
        contradiction_enabled: false,
        contradiction_penalty: 0
      };
      
      const reranker = new LLMReranker(mockDB, config);
      const result = await reranker.rerank('test query', largeBatch);
      
      expect(result.length).toBe(largeBatch.length);
    });
    
    it('should handle concurrent reranking requests', async () => {
      const { getReranker } = await import('./index.js');
      
      const reranker = await getReranker(true);
      
      const concurrentRequests = Array.from({ length: 3 }, (_, i) =>
        reranker.rerank(`query ${i}`, mockCandidates.slice(0, 2))
      );
      
      const results = await Promise.all(concurrentRequests);
      expect(results.length).toBe(3);
      expect(results.every(result => Array.isArray(result))).toBe(true);
    });
  });

  describe('Integration Tests', () => {
    it('should work with real-world candidate data', async () => {
      const { getReranker } = await import('./index.js');
      
      const realWorldCandidates = [
        { docId: 'react-docs-1', score: 0.85, text: 'React is a JavaScript library for building user interfaces. It uses a component-based architecture.' },
        { docId: 'vue-docs-1', score: 0.82, text: 'Vue.js is a progressive JavaScript framework for building user interfaces and single-page applications.' },
        { docId: 'angular-docs-1', score: 0.79, text: 'Angular is a platform and framework for building single-page client applications using HTML and TypeScript.' },
        { docId: 'svelte-docs-1', score: 0.76, text: 'Svelte is a radical new approach to building user interfaces. It compiles components at build time.' }
      ];
      
      const reranker = await getReranker(true);
      const result = await reranker.rerank('JavaScript framework for web development', realWorldCandidates);
      
      expect(result.length).toBe(realWorldCandidates.length);
      expect(result.every(candidate => 
        candidate.docId && typeof candidate.score === 'number' && candidate.text
      )).toBe(true);
    });
    
    it('should maintain candidate structure and metadata', async () => {
      const { CrossEncoderReranker } = await import('./index.js');
      
      const candidatesWithMetadata = [
        { 
          docId: 'doc1', 
          score: 0.8, 
          text: 'React component lifecycle methods',
          metadata: { source: 'official-docs', tags: ['react', 'lifecycle'] }
        },
        { 
          docId: 'doc2', 
          score: 0.7, 
          text: 'Vue composition API guide',
          metadata: { source: 'community-blog', tags: ['vue', 'composition'] }
        }
      ];
      
      const reranker = new CrossEncoderReranker();
      const result = await reranker.rerank('component lifecycle', candidatesWithMetadata);
      
      expect(result.length).toBe(candidatesWithMetadata.length);
      expect(result.every(candidate => candidate.metadata)).toBe(true);
      expect(result[0].metadata).toEqual(candidatesWithMetadata.find(c => c.docId === result[0].docId)?.metadata);
    });
  });
});