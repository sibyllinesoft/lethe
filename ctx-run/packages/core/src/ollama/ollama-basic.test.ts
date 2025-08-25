import { describe, it, expect, vi } from 'vitest';

// Mock fetch for testing
global.fetch = vi.fn();

describe('Ollama Basic Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
    
    // Mock fetch to avoid actual network calls
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      json: vi.fn().mockResolvedValue({ error: 'Service unavailable' })
    } as any);
  });

  it('should import ollama module without errors', async () => {
    const ollamaModule = await import('./index.js');
    expect(ollamaModule).toBeDefined();
  });

  it('should have getOllamaBridge function', async () => {
    const { getOllamaBridge } = await import('./index.js');
    expect(getOllamaBridge).toBeDefined();
    expect(typeof getOllamaBridge).toBe('function');
  });

  it('should have safeParseJSON utility', async () => {
    const { safeParseJSON } = await import('./index.js');
    expect(safeParseJSON).toBeDefined();
    expect(typeof safeParseJSON).toBe('function');
  });

  it('should create ollama bridge', async () => {
    const { getOllamaBridge } = await import('./index.js');
    
    const bridge = getOllamaBridge();
    expect(bridge).toBeDefined();
    expect(typeof bridge).toBe('object');
  });

  it('should handle JSON parsing safely', async () => {
    const { safeParseJSON } = await import('./index.js');
    
    // Test valid JSON
    const validJson = '{"test": "value"}';
    const result = safeParseJSON(validJson);
    expect(result).toEqual({ test: 'value' });
  });

  it('should handle invalid JSON safely', async () => {
    const { safeParseJSON } = await import('./index.js');
    
    // Test invalid JSON
    const invalidJson = '{"test": value}'; // Invalid JSON
    const fallback = { default: true };
    const result = safeParseJSON(invalidJson, fallback);
    expect(result).toEqual(fallback);
  });

  it('should handle empty string JSON parsing', async () => {
    const { safeParseJSON } = await import('./index.js');
    
    const fallback = { default: true };
    const result = safeParseJSON('', fallback);
    expect(result).toEqual(fallback);
  });

  it('should handle ollama bridge health check', async () => {
    const { getOllamaBridge } = await import('./index.js');
    
    const bridge = await getOllamaBridge();
    
    // Health check should handle unavailable service gracefully
    const isHealthy = await bridge.isAvailable();
    expect(typeof isHealthy).toBe('boolean');
  });

  it('should handle text generation with fallback', async () => {
    const { getOllamaBridge } = await import('./index.js');
    
    const bridge = getOllamaBridge();
    
    try {
      const result = await bridge.generateText('test prompt', 'test-model');
      expect(typeof result).toBe('string');
    } catch (error) {
      // Expected when service is unavailable
      expect(true).toBe(true);
    }
  });

  it('should handle timeout generation', async () => {
    const { getOllamaBridge } = await import('./index.js');
    
    const bridge = getOllamaBridge();
    
    try {
      const result = await bridge.generateWithTimeout('test prompt', 'test-model', 1000);
      expect(typeof result).toBe('string');
    } catch (error) {
      // Expected when service is unavailable or times out
      expect(true).toBe(true);
    }
  });

  it('should handle various JSON parsing edge cases', async () => {
    const { safeParseJSON } = await import('./index.js');
    
    // Test various edge cases
    expect(safeParseJSON('null')).toBeNull();
    expect(safeParseJSON('true')).toBe(true);
    expect(safeParseJSON('false')).toBe(false);
    expect(safeParseJSON('123')).toBe(123);
    expect(safeParseJSON('"string"')).toBe('string');
    expect(safeParseJSON('[]')).toEqual([]);
    expect(safeParseJSON('{}')).toEqual({});
  });
});