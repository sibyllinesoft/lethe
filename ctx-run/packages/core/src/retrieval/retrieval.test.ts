import { describe, it, expect, vi, beforeEach } from 'vitest';
import { bm25SearchWithDb } from './index.js';

// Mock database dependencies
vi.mock('@lethe/sqlite', () => ({
  getDFIdf: vi.fn(() => [
    { term: 'test', idf: 1.5 },
    { term: 'query', idf: 2.0 },
    { term: 'search', idf: 1.8 }
  ]),
  getChunksBySession: vi.fn(() => [
    { id: 'chunk1', text: 'This is a test document with query terms', session_id: 'test-session' },
    { id: 'chunk2', text: 'Another test search document', session_id: 'test-session' },
    { id: 'chunk3', text: 'Some unrelated content here', session_id: 'test-session' }
  ])
}));

// Mock reranker dependency
vi.mock('../reranker/index.js', () => ({
  getReranker: vi.fn(() => ({
    rerank: vi.fn().mockResolvedValue([
      { docId: 'chunk1', score: 0.9, text: 'reranked content 1' },
      { docId: 'chunk2', score: 0.7, text: 'reranked content 2' }
    ])
  }))
}));

// Mock diversifier dependency  
vi.mock('../diversifier/index.js', () => ({
  getDiversifier: vi.fn(() => ({
    diversify: vi.fn((candidates) => candidates.slice(0, 2)) // Simple mock diversification
  }))
}));

// Mock ML prediction dependency
vi.mock('../ml-prediction/index.js', () => ({
  getMLPredictor: vi.fn(() => ({
    predictParameters: vi.fn().mockResolvedValue({
      alpha: 0.6,
      beta: 0.4,
      prediction_time_ms: 100,
      model_loaded: true
    })
  }))
}));

describe('Retrieval Module', () => {
  const mockDB = {
    prepare: vi.fn(() => ({
      all: vi.fn(() => []),
      get: vi.fn(),
      run: vi.fn()
    })),
    exec: vi.fn()
  } as any;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('bm25SearchWithDb', () => {
    it('should return empty array for session with no chunks', async () => {
      const { getChunksBySession } = await import('@lethe/sqlite');
      vi.mocked(getChunksBySession).mockReturnValueOnce([]);

      const result = await bm25SearchWithDb(mockDB, ['test query'], 'empty-session', 5);

      expect(result).toEqual([]);
    });

    it('should perform basic BM25 search', async () => {
      const result = await bm25SearchWithDb(mockDB, ['test query'], 'test-session', 5);

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
      expect(result.length).toBeGreaterThan(0);
      
      // Check that results have correct structure
      result.forEach(candidate => {
        expect(candidate).toHaveProperty('docId');
        expect(candidate).toHaveProperty('score');
        expect(typeof candidate.score).toBe('number');
      });
    });

    it('should handle multiple queries', async () => {
      const result = await bm25SearchWithDb(mockDB, ['test', 'query', 'search'], 'test-session', 5);

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
      // Should find documents matching any of the query terms
      expect(result.length).toBeGreaterThan(0);
    });

    it('should respect k parameter for result limiting', async () => {
      const result = await bm25SearchWithDb(mockDB, ['test'], 'test-session', 1);

      expect(result.length).toBeLessThanOrEqual(1);
    });

    it('should handle empty queries gracefully', async () => {
      const result = await bm25SearchWithDb(mockDB, [], 'test-session', 5);

      expect(result).toEqual([]);
    });

    it('should handle queries with no matching terms', async () => {
      const result = await bm25SearchWithDb(mockDB, ['nonexistent terms'], 'test-session', 5);

      // Should return empty array when no documents match
      expect(Array.isArray(result)).toBe(true);
    });

    it('should tokenize queries correctly', async () => {
      // Test with punctuation and mixed case
      const result = await bm25SearchWithDb(mockDB, ['Test, Query!'], 'test-session', 5);

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
    });

    it('should calculate BM25 scores correctly', async () => {
      const result = await bm25SearchWithDb(mockDB, ['test'], 'test-session', 5);

      // Scores should be positive numbers
      result.forEach(candidate => {
        expect(candidate.score).toBeGreaterThan(0);
        expect(typeof candidate.score).toBe('number');
      });

      // Results should be sorted by score (highest first)
      for (let i = 0; i < result.length - 1; i++) {
        expect(result[i].score).toBeGreaterThanOrEqual(result[i + 1].score);
      }
    });

    it('should handle DF/IDF data correctly', async () => {
      const { getDFIdf } = await import('@lethe/sqlite');
      vi.mocked(getDFIdf).mockReturnValueOnce([
        { term: 'rare', idf: 5.0 },
        { term: 'common', idf: 0.5 }
      ]);

      const result = await bm25SearchWithDb(mockDB, ['rare common'], 'test-session', 5);

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
    });

    it('should skip documents with zero query term matches', async () => {
      const { getChunksBySession } = await import('@lethe/sqlite');
      vi.mocked(getChunksBySession).mockReturnValueOnce([
        { id: 'chunk1', text: 'completely unrelated content xyz', session_id: 'test-session' }
      ]);

      const result = await bm25SearchWithDb(mockDB, ['test query'], 'test-session', 5);

      expect(result).toEqual([]);
    });
  });

  describe('Tokenization', () => {
    it('should handle text tokenization correctly', async () => {
      // Test with mixed content that should match our tokenization
      const { getChunksBySession } = await import('@lethe/sqlite');
      vi.mocked(getChunksBySession).mockReturnValueOnce([
        { id: 'chunk1', text: 'Test-Query with punctuation! And UPPERCASE.', session_id: 'test-session' }
      ]);

      const result = await bm25SearchWithDb(mockDB, ['test query uppercase'], 'test-session', 5);

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
      if (result.length > 0) {
        expect(result[0].docId).toBe('chunk1');
      }
    });

    it('should filter single characters', async () => {
      const { getChunksBySession } = await import('@lethe/sqlite');
      vi.mocked(getChunksBySession).mockReturnValueOnce([
        { id: 'chunk1', text: 'a b test query content', session_id: 'test-session' }
      ]);

      const result = await bm25SearchWithDb(mockDB, ['test'], 'test-session', 5);

      expect(result).toBeDefined();
      // Should find the document despite single characters
      expect(result.length).toBeGreaterThan(0);
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty text chunks', async () => {
      const { getChunksBySession } = await import('@lethe/sqlite');
      vi.mocked(getChunksBySession).mockReturnValueOnce([
        { id: 'chunk1', text: '', session_id: 'test-session' },
        { id: 'chunk2', text: 'test content', session_id: 'test-session' }
      ]);

      const result = await bm25SearchWithDb(mockDB, ['test'], 'test-session', 5);

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
      // Should only find chunk2
      if (result.length > 0) {
        expect(result[0].docId).toBe('chunk2');
      }
    });

    it('should handle missing DF/IDF terms', async () => {
      const { getDFIdf } = await import('@lethe/sqlite');
      vi.mocked(getDFIdf).mockReturnValueOnce([]); // No DF/IDF data

      const result = await bm25SearchWithDb(mockDB, ['test'], 'test-session', 5);

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
      // Should return empty results when no IDF data available
      expect(result.length).toBe(0);
    });

    it('should handle very long documents', async () => {
      const longText = 'test '.repeat(1000) + 'query';
      const { getChunksBySession } = await import('@lethe/sqlite');
      vi.mocked(getChunksBySession).mockReturnValueOnce([
        { id: 'chunk1', text: longText, session_id: 'test-session' }
      ]);

      const result = await bm25SearchWithDb(mockDB, ['test query'], 'test-session', 5);

      expect(result).toBeDefined();
      expect(Array.isArray(result)).toBe(true);
      if (result.length > 0) {
        expect(result[0].score).toBeGreaterThan(0);
      }
    });
  });
});