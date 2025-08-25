import { describe, it, expect, vi } from 'vitest';

// Mock Ollama bridge
vi.mock('../ollama/index.js', () => ({
  getOllamaBridge: vi.fn(() => ({
    checkOllamaHealth: vi.fn().mockResolvedValue(false),
    generateText: vi.fn().mockResolvedValue('rewritten query'),
    generateWithTimeout: vi.fn().mockResolvedValue('generated text')
  })),
  safeParseJSON: vi.fn(str => {
    try { return JSON.parse(str); } catch { return null; }
  })
}));

describe('Query Understanding Basic Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  it('should import query understanding module without errors', async () => {
    const queryModule = await import('./index.js');
    expect(queryModule).toBeDefined();
  });

  it('should have rewriteQuery function', async () => {
    const { rewriteQuery } = await import('./index.js');
    expect(rewriteQuery).toBeDefined();
    expect(typeof rewriteQuery).toBe('function');
  });

  it('should have decomposeQuery function', async () => {
    const { decomposeQuery } = await import('./index.js');
    expect(decomposeQuery).toBeDefined();
    expect(typeof decomposeQuery).toBe('function');
  });

  it('should have processQuery function', async () => {
    const { processQuery } = await import('./index.js');
    expect(processQuery).toBeDefined();
    expect(typeof processQuery).toBe('function');
  });

  it('should have DEFAULT_QUERY_UNDERSTANDING_CONFIG', async () => {
    const { DEFAULT_QUERY_UNDERSTANDING_CONFIG } = await import('./index.js');
    expect(DEFAULT_QUERY_UNDERSTANDING_CONFIG).toBeDefined();
    expect(DEFAULT_QUERY_UNDERSTANDING_CONFIG).toHaveProperty('enabled');
    expect(DEFAULT_QUERY_UNDERSTANDING_CONFIG).toHaveProperty('query_rewrite');
  });

  it('should have testQueryUnderstanding function', async () => {
    const { testQueryUnderstanding } = await import('./index.js');
    expect(testQueryUnderstanding).toBeDefined();
    expect(typeof testQueryUnderstanding).toBe('function');
  });

  it('should have getQueryUnderstandingConfig function', async () => {
    const { getQueryUnderstandingConfig } = await import('./index.js');
    expect(getQueryUnderstandingConfig).toBeDefined();
    expect(typeof getQueryUnderstandingConfig).toBe('function');
  });
});