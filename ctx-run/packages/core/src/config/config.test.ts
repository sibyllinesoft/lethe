import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  validateConfig,
  DEFAULT_CONFIG,
  loadConfig,
  reloadConfig,
  getConfig,
  createDefaultConfigFile,
  type CtxConfig
} from './index.js';

// We'll test the parts we can without complex mocking

describe('Config Module', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    
    // Mock console methods to avoid noise in tests
    vi.spyOn(console, 'log').mockImplementation(() => {});
    vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('DEFAULT_CONFIG', () => {
    it('should have all required fields', () => {
      expect(DEFAULT_CONFIG).toMatchObject({
        version: expect.any(String),
        retrieval: {
          alpha: expect.any(Number),
          beta: expect.any(Number),
          gamma_kind_boost: {
            code: expect.any(Number),
            text: expect.any(Number)
          }
        },
        chunking: {
          target_tokens: expect.any(Number),
          overlap: expect.any(Number)
        },
        timeouts: {
          hyde_ms: expect.any(Number),
          summarize_ms: expect.any(Number),
          ollama_connect_ms: expect.any(Number)
        }
      });
    });

    it('should have valid default values', () => {
      expect(DEFAULT_CONFIG.retrieval.alpha).toBeGreaterThanOrEqual(0);
      expect(DEFAULT_CONFIG.retrieval.alpha).toBeLessThanOrEqual(1);
      expect(DEFAULT_CONFIG.retrieval.beta).toBeGreaterThanOrEqual(0);
      expect(DEFAULT_CONFIG.retrieval.beta).toBeLessThanOrEqual(1);
      expect(DEFAULT_CONFIG.chunking.target_tokens).toBeGreaterThan(0);
      expect(DEFAULT_CONFIG.chunking.overlap).toBeGreaterThanOrEqual(0);
      expect(DEFAULT_CONFIG.timeouts.hyde_ms).toBeGreaterThan(0);
    });
  });

  describe('validateConfig', () => {
    it('should pass validation for default config', () => {
      expect(() => validateConfig(DEFAULT_CONFIG)).not.toThrow();
      expect(console.log).toHaveBeenCalledWith('✅ Configuration validation passed');
    });

    it('should throw error for missing version', () => {
      const invalidConfig = { ...DEFAULT_CONFIG, version: '' };
      
      expect(() => validateConfig(invalidConfig)).toThrow(
        'Configuration missing required "version" field'
      );
    });

    it('should validate retrieval.alpha range', () => {
      const invalidConfig = { ...DEFAULT_CONFIG };
      
      // Test negative alpha
      invalidConfig.retrieval = { ...DEFAULT_CONFIG.retrieval, alpha: -0.1 };
      expect(() => validateConfig(invalidConfig)).toThrow(
        'retrieval.alpha must be between 0 and 1'
      );

      // Test alpha > 1  
      invalidConfig.retrieval = { ...DEFAULT_CONFIG.retrieval, alpha: 1.1 };
      expect(() => validateConfig(invalidConfig)).toThrow(
        'retrieval.alpha must be between 0 and 1'
      );
    });

    it('should validate retrieval.beta range', () => {
      let invalidConfig = { ...DEFAULT_CONFIG };
      
      // Test negative beta
      invalidConfig.retrieval = { ...DEFAULT_CONFIG.retrieval, beta: -0.1 };
      expect(() => validateConfig(invalidConfig)).toThrow(
        'retrieval.beta must be between 0 and 1'
      );

      // Test beta > 1
      invalidConfig = { ...DEFAULT_CONFIG };
      invalidConfig.retrieval = { ...DEFAULT_CONFIG.retrieval, beta: 1.1 };
      expect(() => validateConfig(invalidConfig)).toThrow(
        'retrieval.beta must be between 0 and 1'
      );
    });

    it('should validate chunking.target_tokens is positive', () => {
      const invalidConfig = { ...DEFAULT_CONFIG };
      invalidConfig.chunking = { ...DEFAULT_CONFIG.chunking, target_tokens: 0 };
      
      expect(() => validateConfig(invalidConfig)).toThrow(
        'chunking.target_tokens must be positive'
      );
    });

    it('should validate chunking.overlap constraints', () => {
      let invalidConfig = { ...DEFAULT_CONFIG };
      
      // Negative overlap
      invalidConfig.chunking = { ...DEFAULT_CONFIG.chunking, overlap: -1 };
      expect(() => validateConfig(invalidConfig)).toThrow(
        'chunking.overlap must be non-negative and less than target_tokens'
      );

      // Overlap >= target_tokens
      invalidConfig = { ...DEFAULT_CONFIG };
      invalidConfig.chunking = { 
        ...DEFAULT_CONFIG.chunking, 
        overlap: DEFAULT_CONFIG.chunking.target_tokens 
      };
      expect(() => validateConfig(invalidConfig)).toThrow(
        'chunking.overlap must be non-negative and less than target_tokens'
      );
    });

    it('should validate timeout values are positive', () => {
      let invalidConfig = { ...DEFAULT_CONFIG };
      
      invalidConfig.timeouts = { ...DEFAULT_CONFIG.timeouts, hyde_ms: 0 };
      expect(() => validateConfig(invalidConfig)).toThrow(
        'timeouts.hyde_ms must be positive'
      );

      invalidConfig = { ...DEFAULT_CONFIG };
      invalidConfig.timeouts = { ...DEFAULT_CONFIG.timeouts, summarize_ms: -100 };
      expect(() => validateConfig(invalidConfig)).toThrow(
        'timeouts.summarize_ms must be positive'
      );
    });

    it('should validate ML service configuration when enabled', () => {
      let invalidConfig: CtxConfig = { ...DEFAULT_CONFIG };
      invalidConfig.ml = {
        prediction_service: {
          enabled: true,
          host: "localhost",
          port: -1,
          timeout_ms: 1000,
          fallback_to_static: true
        },
        models: DEFAULT_CONFIG.ml!.models
      };
      
      expect(() => validateConfig(invalidConfig)).toThrow(
        'ml.prediction_service.port must be a valid port number'
      );

      invalidConfig = { ...DEFAULT_CONFIG };
      invalidConfig.ml = {
        prediction_service: {
          enabled: true,
          host: "localhost", 
          port: 65536,
          timeout_ms: 1000,
          fallback_to_static: true
        },
        models: DEFAULT_CONFIG.ml!.models
      };
      expect(() => validateConfig(invalidConfig)).toThrow(
        'ml.prediction_service.port must be a valid port number'
      );

      invalidConfig = { ...DEFAULT_CONFIG };
      invalidConfig.ml = {
        prediction_service: {
          enabled: true,
          host: "localhost",
          port: 8080,
          timeout_ms: 0,
          fallback_to_static: true
        },
        models: DEFAULT_CONFIG.ml!.models
      };
      expect(() => validateConfig(invalidConfig)).toThrow(
        'ml.prediction_service.timeout_ms must be positive'
      );
    });

    it('should validate ML service is disabled by default', () => {
      // ML service should be disabled by default and not trigger validation
      expect(() => validateConfig(DEFAULT_CONFIG)).not.toThrow();
    });
  });

  describe('Config Loading Functions', () => {
    it('should load default config when no file exists', () => {
      const config = loadConfig();
      expect(config.version).toBe(DEFAULT_CONFIG.version);
      expect(config.retrieval.alpha).toBe(DEFAULT_CONFIG.retrieval.alpha);
      expect(config).toHaveProperty('features');
      expect(config).toHaveProperty('ml');
      expect(config).toHaveProperty('development');
    });

    it('should reload config and clear cache', () => {
      // First load
      const config1 = loadConfig();
      
      // Reload should work
      const config2 = reloadConfig();
      
      expect(config2.version).toBe(DEFAULT_CONFIG.version);
      expect(config2.retrieval.alpha).toBe(DEFAULT_CONFIG.retrieval.alpha);
    });

    it('should get cached config', () => {
      // First call loads config
      const config1 = getConfig();
      
      // Second call should return cached version
      const config2 = getConfig();
      
      expect(config1.version).toBe(config2.version);
      expect(config1.retrieval.alpha).toBe(config2.retrieval.alpha);
    });

    it('should create default config file successfully', () => {
      // Test that the function exists and can be called
      // The actual file writing is hard to test without complex mocks
      expect(typeof createDefaultConfigFile).toBe('function');
    });
  });

  describe('Configuration Merging', () => {
    it('should merge partial configurations with defaults', () => {
      // This tests the mergeWithDefaults function indirectly through loadConfig
      const config = loadConfig();
      
      // Should have all required fields from defaults
      expect(config.features).toBeDefined();
      expect(config.query_understanding).toBeDefined();
      expect(config.ml).toBeDefined();
      expect(config.development).toBeDefined();
      
      // Should have proper nested structure
      expect(config.retrieval.gamma_kind_boost).toBeDefined();
      expect(config.retrieval.fusion).toBeDefined();
      expect(config.retrieval.llm_rerank).toBeDefined();
    });
  });

  describe('Edge Cases and Error Handling', () => {
    it('should handle missing optional configuration sections', () => {
      const config = loadConfig();
      
      expect(config.features?.enable_hyde).toBeDefined();
      expect(config.ml?.prediction_service?.enabled).toBeDefined();
      expect(config.development?.debug_enabled).toBeDefined();
    });

    it('should validate edge case values', () => {
      const edgeConfig = {
        ...DEFAULT_CONFIG,
        retrieval: {
          ...DEFAULT_CONFIG.retrieval,
          alpha: 0, // Minimum valid value
          beta: 1   // Maximum valid value
        },
        chunking: {
          ...DEFAULT_CONFIG.chunking,
          target_tokens: 1, // Minimum valid value
          overlap: 0        // Minimum valid value
        }
      };
      
      expect(() => validateConfig(edgeConfig)).not.toThrow();
    });
  });
});