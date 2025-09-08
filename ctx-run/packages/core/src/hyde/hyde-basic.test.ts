import { describe, it, expect, vi } from 'vitest';

// Mock database and ollama dependencies
vi.mock('@lethe/sqlite', () => ({
  getConfig: vi.fn(() => ({ hyde: { enabled: true } }))
}));

vi.mock('../ollama/index.js', () => ({
  getOllamaBridge: vi.fn(() => ({
    checkOllamaHealth: vi.fn().mockResolvedValue(false),
    generateText: vi.fn().mockResolvedValue('generated text'),
    generateWithTimeout: vi.fn().mockResolvedValue('generated text')
  })),
  safeParseJSON: vi.fn(str => {
    try { return JSON.parse(str); } catch { return null; }
  })
}));

describe('HyDE Basic Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  it('should import HyDE module without errors', async () => {
    const hydeModule = await import('./index.js');
    expect(hydeModule).toBeDefined();
  });

  it('should have DEFAULT_HYDE_CONFIG', async () => {
    const { DEFAULT_HYDE_CONFIG } = await import('./index.js');
    expect(DEFAULT_HYDE_CONFIG).toBeDefined();
    expect(DEFAULT_HYDE_CONFIG).toHaveProperty('enabled');
    expect(DEFAULT_HYDE_CONFIG).toHaveProperty('model');
    expect(DEFAULT_HYDE_CONFIG).toHaveProperty('temperature');
  });

  it('should have HydeResult and HydeConfig interfaces', async () => {
    const hydeModule = await import('./index.js');
    expect(hydeModule).toHaveProperty('DEFAULT_HYDE_CONFIG');
  });

  it('should have generateHyde function', async () => {
    const { generateHyde } = await import('./index.js');
    expect(generateHyde).toBeDefined();
    expect(typeof generateHyde).toBe('function');
  });

  it('should have testHyde function', async () => {
    const { testHyde } = await import('./index.js');
    expect(testHyde).toBeDefined();
    expect(typeof testHyde).toBe('function');
  });

  it('should handle function exports', async () => {
    const hydeModule = await import('./index.js');
    
    // Check for main exports
    expect(hydeModule.generateHyde).toBeDefined();
    expect(hydeModule.testHyde).toBeDefined();
    expect(hydeModule.DEFAULT_HYDE_CONFIG).toBeDefined();
  });
});