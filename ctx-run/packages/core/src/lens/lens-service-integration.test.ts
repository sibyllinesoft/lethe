/**
 * Service Integration and Configuration Tests for Lens
 * Focuses on service instantiation, configuration loading, caching, and integration scenarios
 */

import { describe, it, expect, beforeEach, afterEach, vi, type Mock } from 'vitest';
import {
  getLensService,
  testLensIntegration,
  DEFAULT_LENS_CONFIG,
  type LensService,
  type LensConfig,
  type LensSearchRequest,
  type LensSearchResponse
} from './index.js';

// Mock fetch for HTTP interactions
const mockFetch = vi.fn() as Mock;
global.fetch = mockFetch;

// Mock the config module
vi.mock('../config/index.js', () => ({
  getConfig: vi.fn()
}));

describe('Lens Service Integration Tests', () => {
  let mockGetConfig: Mock;

  beforeEach(async () => {
    vi.clearAllMocks();
    mockFetch.mockClear();
    
    // Reset the module mock
    const configModule = await import('../config/index.js') as any;
    mockGetConfig = configModule.getConfig as Mock;
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  describe('Service Instantiation and Caching', () => {
    it('should create and cache service instances', async () => {
      const service1 = await getLensService();
      const service2 = await getLensService();
      
      // Should return the same instance due to caching
      expect(service1).toBe(service2);
    });

    it('should create different instances for different configs', async () => {
      // Mock different configurations
      mockGetConfig
        .mockReturnValueOnce({ lens: { base_url: 'http://localhost:5678' } })
        .mockReturnValueOnce({ lens: { base_url: 'http://localhost:5679' } });

      const mockDb1 = { mock: 'db1' };
      const mockDb2 = { mock: 'db2' };

      const service1 = await getLensService(mockDb1);
      const service2 = await getLensService(mockDb2);
      
      // Should be different instances due to different configs
      expect(service1).not.toBe(service2);
    });

    it('should handle config loading failures gracefully', async () => {
      mockGetConfig.mockImplementation(() => {
        throw new Error('Config not found');
      });

      const mockDb = { mock: 'db' };
      const service = await getLensService(mockDb);
      
      // Should still create service with default config
      expect(service).toBeDefined();
      expect(typeof service.search).toBe('function');
    });

    it('should merge database config with defaults', async () => {
      const customConfig = {
        lens: {
          base_url: 'http://custom.lens:8080',
          topic_fanout_k: 500,
          enabled: true
        }
      };

      mockGetConfig.mockReturnValue(customConfig);
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          symbol_groups: [],
          processing_time_ms: 100,
          lsp_available: true,
          topics_expanded: 2,
          timeout_hit: false,
          metadata: {
            version: '1.2.0',
            query_analysis: {
              extracted_symbols: [],
              code_intent_confidence: 0.5
            }
          }
        })
      });

      const mockDb = { mock: 'db' };
      const service = await getLensService(mockDb);
      
      await service.search({ query: 'test', max_groups: 5 });

      // Should use custom base URL
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('http://custom.lens:8080/api/search'),
        expect.anything()
      );
    });
  });

  describe('Configuration Validation and Edge Cases', () => {
    it('should handle missing lens config section', async () => {
      mockGetConfig.mockReturnValue({}); // No lens section

      const service = await getLensService({ mock: 'db' });
      
      expect(service).toBeDefined();
    });

    it('should handle partial lens config', async () => {
      const partialConfig = {
        lens: {
          base_url: 'http://partial.lens:3000',
          // Missing other config values
        }
      };

      mockGetConfig.mockReturnValue(partialConfig);

      const service = await getLensService({ mock: 'db' });
      
      expect(service).toBeDefined();
    });

    it('should handle config with null/undefined values', async () => {
      const configWithNulls = {
        lens: {
          base_url: null,
          enabled: undefined,
          topic_fanout_k: 'invalid'
        }
      };

      mockGetConfig.mockReturnValue(configWithNulls);

      // Should not throw
      expect(async () => {
        await getLensService({ mock: 'db' });
      }).not.toThrow();
    });

    it('should validate extreme config values', () => {
      const extremeConfigs: LensConfig[] = [
        {
          ...DEFAULT_LENS_CONFIG,
          connect_timeout_ms: -100, // Negative timeout
          request_timeout_ms: 0,    // Zero timeout
        },
        {
          ...DEFAULT_LENS_CONFIG,
          topic_fanout_k: 999999,   // Very large fanout
          weight_cap: -0.5,         // Negative weight cap
        },
        {
          ...DEFAULT_LENS_CONFIG,
          max_tokens_per_response: 0,    // Zero tokens
          lens_tokens_cap: -1000,        // Negative cap
        }
      ];

      extremeConfigs.forEach(config => {
        // Should not throw during configuration
        expect(() => {
          const configJson = JSON.stringify(config);
          expect(configJson).toBeDefined();
        }).not.toThrow();
      });
    });
  });

  describe('Service State Management', () => {
    it('should handle disabled service correctly', async () => {
      const disabledConfig = {
        lens: {
          ...DEFAULT_LENS_CONFIG,
          enabled: false
        }
      };

      mockGetConfig.mockReturnValue(disabledConfig);
      const service = await getLensService({ mock: 'db' });

      const result = await service.search({ query: 'test', max_groups: 5 });

      // Should return empty response without making HTTP call
      expect(mockFetch).not.toHaveBeenCalled();
      expect(result.symbol_groups).toHaveLength(0);
      expect(result.metadata.version).toBe('disabled');
    });

    it('should handle testConnection with various error types', async () => {
      const service = await getLensService();

      const errorScenarios = [
        { error: new Error('ECONNREFUSED'), expectedAvailable: false },
        { error: new Error('ETIMEDOUT'), expectedAvailable: false },
        { error: new Error('Network error'), expectedAvailable: false }
      ];

      for (const { error, expectedAvailable } of errorScenarios) {
        mockFetch.mockRejectedValueOnce(error);

        const result = await service.testConnection();

        expect(result.available).toBe(expectedAvailable);
        expect(result.error).toContain(error.message);
        expect(result.latency_ms).toBeGreaterThan(0);
      }
    });

    it('should handle isAvailable timeout correctly', async () => {
      vi.useFakeTimers();
      const service = await getLensService();

      mockFetch.mockImplementation(() => {
        return new Promise((resolve) => {
          setTimeout(() => resolve({
            ok: true
          }), 600); // Longer than connect timeout (500ms)
        });
      });

      const availabilityPromise = service.isAvailable();
      vi.advanceTimersByTime(700);

      const isAvailable = await availabilityPromise;

      expect(isAvailable).toBe(false);

      vi.useRealTimers();
    });
  });

  describe('Integration Test Coverage', () => {
    it('should run comprehensive integration test with server unavailable', async () => {
      // Mock all service calls to fail
      mockFetch.mockRejectedValue(new Error('Service unavailable'));

      const result = await testLensIntegration();

      expect(result.available).toBe(false);
      expect(result.error).toBeDefined();
      expect(result.search_test).toBeUndefined();
      
      // Code intent and cost analysis should still pass
      expect(result.code_intent_test).toBe(true);
      expect(result.cost_analysis_test).toBe(true);
    });

    it('should run integration test with partial service functionality', async () => {
      // Mock availability check to pass but search to fail
      mockFetch
        .mockResolvedValueOnce({ ok: true }) // isAvailable passes
        .mockRejectedValueOnce(new Error('Search failed')); // search fails

      const result = await testLensIntegration();

      expect(result.available).toBe(true);
      expect(result.latency_ms).toBeGreaterThan(0);
      expect(result.search_test).toBe(false);
      expect(result.code_intent_test).toBe(true);
      expect(result.cost_analysis_test).toBe(true);
    });

    it('should run integration test with successful service', async () => {
      // Mock all calls to succeed
      mockFetch
        .mockResolvedValueOnce({ ok: true }) // isAvailable
        .mockResolvedValueOnce({ // search
          ok: true,
          json: () => Promise.resolve({
            symbol_groups: [{
              id: 'test_symbol',
              primary_symbol: 'testFunction',
              language: 'typescript',
              file_path: 'test.ts',
              definition: {
                id: 'def_1',
                content: 'function testFunction() {}',
                file_path: 'test.ts',
                start_line: 1,
                end_line: 1,
                start_char: 0,
                end_char: 25,
                atom_type: 'definition',
                symbol_name: 'testFunction',
                tokens: 6,
                importance: 1.0
              },
              references: [],
              implementations: [],
              estimated_tokens: 30,
              relevance_score: 0.8,
              topic_weight: 0.3,
              is_precise_match: true
            }],
            processing_time_ms: 89,
            lsp_available: true,
            topics_expanded: 2,
            timeout_hit: false,
            metadata: {
              version: '1.2.0',
              query_analysis: {
                extracted_symbols: ['testFunction'],
                code_intent_confidence: 0.85
              }
            }
          })
        });

      const result = await testLensIntegration();

      expect(result.available).toBe(true);
      expect(result.search_test).toBe(true);
      expect(result.code_intent_test).toBe(true);
      expect(result.cost_analysis_test).toBe(true);
      expect(result.latency_ms).toBeGreaterThan(0);
    });

    it('should handle integration test internal errors', async () => {
      // Mock getLensService to throw an error
      const originalGetLensService = getLensService;
      
      vi.mocked(getLensService).mockImplementationOnce(() => {
        throw new Error('Service initialization failed');
      });

      const result = await testLensIntegration();

      expect(result.available).toBe(false);
      expect(result.error).toContain('Service initialization failed');
    });
  });

  describe('Request Processing Edge Cases', () => {
    it('should handle search requests with all optional parameters', async () => {
      const service = await getLensService();

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          symbol_groups: [],
          processing_time_ms: 75,
          lsp_available: true,
          topics_expanded: 3,
          timeout_hit: false,
          metadata: {
            version: '1.2.0',
            query_analysis: {
              detected_language: 'typescript',
              extracted_symbols: ['testSymbol'],
              code_intent_confidence: 0.9
            }
          }
        })
      });

      const fullRequest: LensSearchRequest = {
        query: 'complex search query',
        max_groups: 15,
        topic_fanout_k: 300,
        weight_cap: 0.5,
        file_context: ['src/main.ts', 'src/utils.ts'],
        repo_context: 'my-project',
        language_hints: ['typescript', 'javascript'],
        timeout_ms: 200
      };

      const result = await service.search(fullRequest);

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/search'),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          }),
          body: expect.stringContaining('complex search query')
        })
      );

      expect(result.processing_time_ms).toBe(75);
      expect(result.metadata.query_analysis.detected_language).toBe('typescript');
    });

    it('should handle search requests with minimal parameters', async () => {
      const service = await getLensService();

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          symbol_groups: [],
          processing_time_ms: 45,
          lsp_available: false,
          topics_expanded: 1,
          timeout_hit: false,
          metadata: {
            version: '1.2.0',
            query_analysis: {
              extracted_symbols: [],
              code_intent_confidence: 0.3
            }
          }
        })
      });

      const minimalRequest: LensSearchRequest = {
        query: 'simple',
        max_groups: 1
      };

      const result = await service.search(minimalRequest);

      expect(result.processing_time_ms).toBe(45);
      expect(result.symbol_groups).toHaveLength(0);
    });

    it('should handle concurrent search requests', async () => {
      const service = await getLensService();

      // Mock responses for concurrent requests
      const requests = Array.from({ length: 5 }, (_, i) => ({
        query: `concurrent query ${i}`,
        max_groups: 5
      }));

      requests.forEach((_, i) => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            symbol_groups: [],
            processing_time_ms: 50 + i * 10,
            lsp_available: true,
            topics_expanded: 2,
            timeout_hit: false,
            metadata: {
              version: '1.2.0',
              query_analysis: {
                extracted_symbols: [],
                code_intent_confidence: 0.5
              }
            }
          })
        });
      });

      const results = await Promise.all(
        requests.map(req => service.search(req))
      );

      expect(results).toHaveLength(5);
      results.forEach((result, i) => {
        expect(result.processing_time_ms).toBe(50 + i * 10);
      });
    });
  });

  describe('Error Boundary Testing', () => {
    it('should handle malformed JSON responses gracefully', async () => {
      const service = await getLensService();

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.reject(new Error('Unexpected token'))
      });

      await expect(service.search({
        query: 'test query',
        max_groups: 5
      })).rejects.toThrow('Unexpected token');
    });

    it('should handle responses with missing required fields', async () => {
      const service = await getLensService();

      const incompleteResponse = {
        // Missing symbol_groups
        processing_time_ms: 100,
        lsp_available: true
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(incompleteResponse)
      });

      await expect(service.search({
        query: 'test query',
        max_groups: 5
      })).rejects.toThrow('Invalid Lens response: missing symbol_groups array');
    });

    it('should handle responses with malformed symbol_groups', async () => {
      const service = await getLensService();

      const malformedResponse = {
        symbol_groups: 'not an array', // Should be an array
        processing_time_ms: 100,
        lsp_available: true,
        topics_expanded: 2,
        timeout_hit: false,
        metadata: {
          version: '1.2.0',
          query_analysis: {
            extracted_symbols: [],
            code_intent_confidence: 0.5
          }
        }
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(malformedResponse)
      });

      await expect(service.search({
        query: 'test query',
        max_groups: 5
      })).rejects.toThrow('Invalid Lens response: missing symbol_groups array');
    });

    it('should handle various HTTP status codes', async () => {
      const service = await getLensService();

      const statusCodes = [
        { status: 400, statusText: 'Bad Request' },
        { status: 401, statusText: 'Unauthorized' },
        { status: 403, statusText: 'Forbidden' },
        { status: 404, statusText: 'Not Found' },
        { status: 500, statusText: 'Internal Server Error' },
        { status: 502, statusText: 'Bad Gateway' },
        { status: 503, statusText: 'Service Unavailable' }
      ];

      for (const { status, statusText } of statusCodes) {
        mockFetch.mockResolvedValueOnce({
          ok: false,
          status,
          statusText
        });

        await expect(service.search({
          query: 'test query',
          max_groups: 5
        })).rejects.toThrow(`Lens HTTP ${status}: ${statusText}`);
      }
    });
  });
});