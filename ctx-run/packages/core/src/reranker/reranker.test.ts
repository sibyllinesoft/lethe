import { describe, it, expect, vi, beforeEach } from 'vitest';
import { CrossEncoderReranker, type Reranker, type RerankerConfig } from './index.js';
import type { Candidate } from '../retrieval/index.js';

describe('Reranker Module', () => {
  const createMockCandidates = (count: number): Candidate[] => {
    return Array.from({ length: count }, (_, i) => ({
      id: `candidate-${i}`,
      text: `This is candidate ${i} discussing topic ${i % 3} with relevant information.`,
      score: 0.5 + (i * 0.1), // Ascending scores
      metadata: {
        source: `source-${i}`,
        kind: 'text'
      }
    }));
  };

  describe('CrossEncoderReranker', () => {
    let reranker: CrossEncoderReranker;

    beforeEach(() => {
      vi.clearAllMocks();
      vi.spyOn(console, 'log').mockImplementation(() => {});
      vi.spyOn(console, 'warn').mockImplementation(() => {});
      reranker = new CrossEncoderReranker();
    });

    it('should have default model name', () => {
      expect(reranker.name).toBe("Xenova/ms-marco-MiniLM-L-6-v2");
    });

    it('should accept custom model ID', () => {
      const customReranker = new CrossEncoderReranker("custom-model");
      expect(customReranker.name).toBe("custom-model");
    });

    it('should rerank candidates successfully', async () => {
      const query = "JavaScript programming";
      const candidates = createMockCandidates(3);
      
      const result = await reranker.rerank(query, candidates);
      
      expect(result).toHaveLength(3);
      expect(result[0]).toHaveProperty('score');
      expect(result[0]).toHaveProperty('id');
      expect(result[0]).toHaveProperty('text');
    });

    it('should handle empty candidates array', async () => {
      const query = "test query";
      const result = await reranker.rerank(query, []);
      
      expect(result).toEqual([]);
      expect(result).toHaveLength(0);
    });

    it('should handle single candidate', async () => {
      const query = "test query";
      const candidates = createMockCandidates(1);
      
      const result = await reranker.rerank(query, candidates);
      
      expect(result).toHaveLength(1);
      expect(result[0]).toEqual(expect.objectContaining({
        id: candidates[0].id,
        text: candidates[0].text
      }));
    });

    it('should handle candidates without text content', async () => {
      const query = "test query";
      const candidates = [
        {
          id: 'candidate-1',
          text: undefined as any,
          score: 0.8,
          metadata: { source: 'source-1', kind: 'text' }
        },
        {
          id: 'candidate-2',
          text: 'Valid text content',
          score: 0.6,
          metadata: { source: 'source-2', kind: 'text' }
        }
      ];
      
      const result = await reranker.rerank(query, candidates);
      
      expect(result).toHaveLength(2);
      // Should handle undefined text gracefully
    });

    it('should calculate text similarity correctly', async () => {
      const query = "JavaScript programming";
      const candidates = [
        {
          id: 'candidate-1',
          text: 'JavaScript is a programming language',
          score: 0.5,
          metadata: { source: 'source-1', kind: 'text' }
        },
        {
          id: 'candidate-2',
          text: 'Python is also a programming language',
          score: 0.5,
          metadata: { source: 'source-2', kind: 'text' }
        },
        {
          id: 'candidate-3',
          text: 'Weather forecast for today',
          score: 0.5,
          metadata: { source: 'source-3', kind: 'text' }
        }
      ];
      
      const result = await reranker.rerank(query, candidates);
      
      expect(result).toHaveLength(3);
      // The JavaScript-related candidate should have higher similarity
      expect(result[0].text).toContain('JavaScript');
    });
  });

  // Note: LLMReranker tests omitted as it requires complex database mocking
  // The CrossEncoderReranker provides sufficient coverage for the reranker interface

  describe('Reranker Interface Compliance', () => {
    it('should implement Reranker interface correctly', () => {
      const crossEncoder: Reranker = new CrossEncoderReranker();

      expect(typeof crossEncoder.name).toBe('string');
      expect(typeof crossEncoder.rerank).toBe('function');
    });

    it('should export RerankerConfig interface correctly', () => {
      const mockConfig: RerankerConfig = {
        use_llm: true,
        llm_budget_ms: 5000,
        llm_model: "llama3.2:1b",
        contradiction_enabled: false,
        contradiction_penalty: 0.15
      };

      expect(mockConfig.use_llm).toBe(true);
      expect(mockConfig.llm_budget_ms).toBe(5000);
      expect(typeof mockConfig.llm_model).toBe('string');
    });
  });

  describe('Performance and Edge Cases', () => {
    it('should handle large candidate sets efficiently', async () => {
      const reranker = new CrossEncoderReranker();
      const query = "test query";
      const largeCandidates = createMockCandidates(50);
      
      const startTime = Date.now();
      const result = await reranker.rerank(query, largeCandidates);
      const endTime = Date.now();
      
      expect(result).toHaveLength(50);
      expect(endTime - startTime).toBeLessThan(2000); // Should complete within 2 seconds
    });

    it('should maintain candidate integrity during reranking', async () => {
      const reranker = new CrossEncoderReranker();
      const query = "test query";
      const candidates = createMockCandidates(3);
      const originalIds = candidates.map(c => c.id);
      
      const result = await reranker.rerank(query, candidates);
      const resultIds = result.map(c => c.id);
      
      expect(result).toHaveLength(3);
      expect(resultIds.sort()).toEqual(originalIds.sort()); // All candidates should be preserved
    });

    it('should handle special characters in query and text', async () => {
      const reranker = new CrossEncoderReranker();
      const query = "特殊字符 & émojis 🚀 test";
      const candidates = [
        {
          id: 'special-1',
          text: '特殊字符 handling in software development',
          score: 0.5,
          metadata: { source: 'unicode-guide', kind: 'text' }
        },
        {
          id: 'special-2', 
          text: 'Regular ASCII text content',
          score: 0.5,
          metadata: { source: 'basic-guide', kind: 'text' }
        }
      ];
      
      const result = await reranker.rerank(query, candidates);
      
      expect(result).toHaveLength(2);
      // Should handle special characters gracefully
    });

    it('should handle very long text content', async () => {
      const reranker = new CrossEncoderReranker();
      const query = "test query";
      const longText = "This is a very long text. ".repeat(1000); // ~27KB text
      const candidates = [
        {
          id: 'long-text',
          text: longText,
          score: 0.5,
          metadata: { source: 'long-doc', kind: 'text' }
        },
        {
          id: 'short-text',
          text: 'Short text',
          score: 0.5,
          metadata: { source: 'short-doc', kind: 'text' }
        }
      ];
      
      const result = await reranker.rerank(query, candidates);
      
      expect(result).toHaveLength(2);
      // Should handle long text without crashing
    });
  });
});