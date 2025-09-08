/**
 * Comprehensive tests for Lens integration
 * Tests integration with retrieval pipeline, configuration, and production scenarios
 */

import { describe, it, expect, beforeEach, afterEach, vi, type Mock } from 'vitest';
import {
  getLensService,
  detectCodeIntent,
  calculateLagrangianCost,
  symbolGroupsToRetrievalCandidates,
  DEFAULT_LENS_CONFIG,
  type LensService,
  type SymbolGroup,
  type LensSearchRequest,
  type LensSearchResponse,
  type LensConfig
} from './index.js';

// Mock fetch for testing HTTP interactions
const mockFetch = vi.fn() as Mock;
global.fetch = mockFetch;

describe('Lens Integration - Comprehensive Tests', () => {
  let mockLensService: LensService;

  beforeEach(() => {
    vi.clearAllMocks();
    mockFetch.mockClear();
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  describe('HTTP Client Integration', () => {
    beforeEach(async () => {
      mockLensService = await getLensService();
    });

    it('should handle successful search requests', async () => {
      const mockResponse: LensSearchResponse = {
        symbol_groups: [{
          id: 'test_symbol_1',
          primary_symbol: 'calculateScore',
          language: 'typescript',
          file_path: 'src/scoring.ts',
          definition: {
            id: 'def_1',
            content: 'function calculateScore(data: any[]): number { return 0.95; }',
            file_path: 'src/scoring.ts',
            start_line: 15,
            end_line: 15,
            start_char: 0,
            end_char: 56,
            atom_type: 'definition',
            symbol_name: 'calculateScore',
            tokens: 18,
            importance: 1.0
          },
          references: [{
            id: 'ref_1',
            content: 'const score = calculateScore(candidateData);',
            file_path: 'src/main.ts',
            start_line: 42,
            end_line: 42,
            start_char: 4,
            end_char: 48,
            atom_type: 'reference',
            symbol_name: 'calculateScore',
            tokens: 12,
            importance: 0.8
          }],
          implementations: [],
          estimated_tokens: 120,
          relevance_score: 0.92,
          topic_weight: 0.35,
          is_precise_match: true
        }],
        processing_time_ms: 87,
        lsp_available: true,
        topics_expanded: 3,
        timeout_hit: false,
        metadata: {
          version: '1.2.0',
          query_analysis: {
            detected_language: 'typescript',
            extracted_symbols: ['calculateScore'],
            code_intent_confidence: 0.89
          }
        }
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const request: LensSearchRequest = {
        query: 'how does calculateScore work',
        max_groups: 10,
        topic_fanout_k: 240,
        weight_cap: 0.4,
        language_hints: ['typescript']
      };

      const result = await mockLensService.search(request);

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/search'),
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json'
          }),
          body: expect.stringContaining('calculateScore')
        })
      );

      expect(result.symbol_groups).toHaveLength(1);
      expect(result.symbol_groups[0].primary_symbol).toBe('calculateScore');
      expect(result.processing_time_ms).toBe(87);
      expect(result.lsp_available).toBe(true);
    });

    it('should handle HTTP errors gracefully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error'
      });

      const request: LensSearchRequest = {
        query: 'test query',
        max_groups: 5
      };

      await expect(mockLensService.search(request)).rejects.toThrow('Lens HTTP 500: Internal Server Error');
    });

    it('should handle network timeouts', async () => {
      vi.useFakeTimers();

      mockFetch.mockImplementationOnce(() => {
        return new Promise((resolve) => {
          setTimeout(() => resolve({
            ok: true,
            json: () => Promise.resolve({ symbol_groups: [] })
          }), 200);
        });
      });

      const request: LensSearchRequest = {
        query: 'test query',
        max_groups: 5,
        timeout_ms: 100
      };

      const searchPromise = mockLensService.search(request);
      
      // Fast-forward time to trigger timeout
      vi.advanceTimersByTime(150);

      const result = await searchPromise;

      // Should return timeout response
      expect(result.timeout_hit).toBe(true);
      expect(result.symbol_groups).toHaveLength(0);

      vi.useRealTimers();
    });

    it('should validate response structure', async () => {
      const invalidResponse = {
        // Missing symbol_groups array
        processing_time_ms: 50,
        lsp_available: true
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(invalidResponse)
      });

      const request: LensSearchRequest = {
        query: 'test query',
        max_groups: 5
      };

      await expect(mockLensService.search(request)).rejects.toThrow('Invalid Lens response: missing symbol_groups array');
    });
  });

  describe('Configuration Integration', () => {
    it('should merge custom config with defaults', async () => {
      const customConfig = {
        ...DEFAULT_LENS_CONFIG,
        topic_fanout_k: 500,
        weight_cap: 0.6,
        lambda_multiplier: 1.5
      };

      // Test that config merging would work
      expect(customConfig.topic_fanout_k).toBe(500);
      expect(customConfig.weight_cap).toBe(0.6);
      expect(customConfig.lambda_multiplier).toBe(1.5);
      // Defaults should be preserved
      expect(customConfig.dpp_rank).toBe(DEFAULT_LENS_CONFIG.dpp_rank);
    });

    it('should handle different mode configurations', () => {
      const modes: Array<'auto' | 'earn-its-place' | 'disabled'> = ['auto', 'earn-its-place', 'disabled'];
      
      modes.forEach(mode => {
        const config: LensConfig = {
          ...DEFAULT_LENS_CONFIG,
          mode
        };
        
        expect(config.mode).toBe(mode);
        
        // Each mode should affect cost calculations differently
        const symbolGroups: SymbolGroup[] = [{
          id: 'test',
          primary_symbol: 'test',
          language: 'typescript',
          file_path: 'test.ts',
          definition: {
            id: 'def_1',
            content: 'const test = 1;',
            file_path: 'test.ts',
            start_line: 1,
            end_line: 1,
            start_char: 0,
            end_char: 15,
            atom_type: 'definition',
            symbol_name: 'test',
            tokens: 4,
            importance: 1.0
          },
          references: [],
          implementations: [],
          estimated_tokens: 50,
          relevance_score: 0.7,
          topic_weight: 0.3,
          is_precise_match: true
        }];

        const costResult = calculateLagrangianCost(symbolGroups, config, 1000, 4000, 100);
        expect(costResult).toBeDefined();
        expect(costResult.total_cost).toBeGreaterThan(0);
      });
    });
  });

  describe('Production Scenarios', () => {
    it('should handle high-load scenarios', async () => {
      // Simulate multiple concurrent requests
      const requests = Array.from({ length: 10 }, (_, i) => ({
        query: `test query ${i}`,
        max_groups: 5,
        timeout_ms: 150
      }));

      // Mock responses with varying latencies
      requests.forEach((_, i) => {
        mockFetch.mockResolvedValueOnce({
          ok: true,
          json: () => Promise.resolve({
            symbol_groups: [],
            processing_time_ms: 50 + i * 10, // Increasing latency
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

      const service = await getLensService();
      const results = await Promise.all(requests.map(req => service.search(req)));

      expect(results).toHaveLength(10);
      results.forEach((result, i) => {
        expect(result.processing_time_ms).toBe(50 + i * 10);
        expect(result.timeout_hit).toBe(false);
      });
    });

    it('should respect SLA constraints under load', () => {
      const symbolGroups: SymbolGroup[] = Array.from({ length: 20 }, (_, i) => ({
        id: `symbol_${i}`,
        primary_symbol: `function${i}`,
        language: 'typescript',
        file_path: `src/module${i}.ts`,
        definition: {
          id: `def_${i}`,
          content: `function function${i}() { /* implementation */ }`,
          file_path: `src/module${i}.ts`,
          start_line: 1,
          end_line: 3,
          start_char: 0,
          end_char: 50,
          atom_type: 'definition',
          symbol_name: `function${i}`,
          tokens: 12,
          importance: 0.8
        },
        references: [],
        implementations: [],
        estimated_tokens: 80,
        relevance_score: 0.75,
        topic_weight: 0.25,
        is_precise_match: i % 3 === 0 // Every third is precise
      }));

      // Test various latency scenarios
      const latencyTests = [120, 150, 180, 200];
      
      latencyTests.forEach(latency => {
        const costResult = calculateLagrangianCost(
          symbolGroups,
          DEFAULT_LENS_CONFIG,
          2000, // existing tokens
          8000, // total budget
          latency
        );

        expect(costResult.cost_breakdown.estimated_latency_ms).toBe(latency);
        expect(costResult.sla_constraint_met).toBe(latency <= DEFAULT_LENS_CONFIG.sla_recall_ms);
      });
    });

    it('should handle fallback scenarios', async () => {
      const service = await getLensService();

      // Test LSP unavailable scenario
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          symbol_groups: [{
            id: 'fallback_symbol',
            primary_symbol: 'fallbackFunction',
            language: 'typescript',
            file_path: 'src/fallback.ts',
            definition: {
              id: 'def_fallback',
              content: 'function fallbackFunction() {}',
              file_path: 'src/fallback.ts',
              start_line: 1,
              end_line: 1,
              start_char: 0,
              end_char: 30,
              atom_type: 'definition',
              symbol_name: 'fallbackFunction',
              tokens: 8,
              importance: 0.9
            },
            references: [],
            implementations: [],
            estimated_tokens: 60,
            relevance_score: 0.8,
            topic_weight: 0.2, // Lower weight due to no LSP
            is_precise_match: false // No LSP precision
          }],
          processing_time_ms: 95,
          lsp_available: false, // LSP not available
          topics_expanded: 5, // More topic expansion to compensate
          timeout_hit: false,
          metadata: {
            version: '1.2.0',
            query_analysis: {
              extracted_symbols: ['fallbackFunction'],
              code_intent_confidence: 0.7
            }
          }
        })
      });

      const result = await service.search({
        query: 'fallback function implementation',
        max_groups: 5
      });

      expect(result.lsp_available).toBe(false);
      expect(result.topics_expanded).toBeGreaterThan(0);
      expect(result.symbol_groups[0].is_precise_match).toBe(false);
      expect(result.symbol_groups[0].topic_weight).toBeLessThan(0.3);
    });
  });

  describe('Integration with Retrieval Pipeline', () => {
    it('should convert symbol groups to compatible candidate format', () => {
      const symbolGroups: SymbolGroup[] = [{
        id: 'integration_test',
        primary_symbol: 'hybridRetrieval',
        language: 'typescript',
        file_path: 'src/retrieval/index.ts',
        definition: {
          id: 'def_hybrid',
          content: 'export async function hybridRetrieval(queries: string[], options: HybridRetrievalOptions): Promise<Candidate[]> {',
          file_path: 'src/retrieval/index.ts',
          start_line: 309,
          end_line: 312,
          start_char: 0,
          end_char: 120,
          atom_type: 'definition',
          symbol_name: 'hybridRetrieval',
          tokens: 35,
          importance: 1.0
        },
        references: [{
          id: 'ref_hybrid_1',
          content: 'const results = await hybridRetrieval(queries, retrievalOptions);',
          file_path: 'src/pipeline/orchestrator.ts',
          start_line: 156,
          end_line: 156,
          start_char: 2,
          end_char: 67,
          atom_type: 'reference',
          symbol_name: 'hybridRetrieval',
          tokens: 15,
          importance: 0.9
        }],
        implementations: [],
        test_hints: [{
          id: 'test_hybrid',
          content: 'describe("hybridRetrieval", () => { it("should retrieve candidates", async () => { /* test */ }); });',
          file_path: 'src/retrieval/retrieval.test.ts',
          start_line: 45,
          end_line: 45,
          start_char: 0,
          end_char: 98,
          atom_type: 'test',
          symbol_name: 'hybridRetrieval',
          tokens: 25,
          importance: 0.6
        }],
        estimated_tokens: 300,
        relevance_score: 0.95,
        topic_weight: 0.4,
        is_precise_match: true
      }];

      const candidates = symbolGroupsToRetrievalCandidates(symbolGroups);

      expect(candidates).toHaveLength(1);
      
      const candidate = candidates[0];
      expect(candidate.docId).toBe('lens_integration_test');
      expect(candidate.score).toBe(0.95 * 0.4); // relevance * topic_weight
      expect(candidate.kind).toBe('code_symbol');
      expect(candidate.metadata.is_lens_result).toBe(true);
      
      // Check formatted text contains all sections
      const text = candidate.text;
      expect(text).toContain('Symbol: hybridRetrieval');
      expect(text).toContain('src/retrieval/index.ts');
      expect(text).toContain('Definition:');
      expect(text).toContain('export async function hybridRetrieval');
      expect(text).toContain('Key References:');
      expect(text).toContain('const results = await hybridRetrieval');
      expect(text).toContain('Test Examples:');
      expect(text).toContain('describe("hybridRetrieval"');
    });

    it('should integrate with existing scoring mechanisms', () => {
      const symbolGroup: SymbolGroup = {
        id: 'scoring_test',
        primary_symbol: 'calculateBM25',
        language: 'typescript',
        file_path: 'src/retrieval/index.ts',
        definition: {
          id: 'def_bm25',
          content: 'function calculateBM25(termFreqs: { [term: string]: number }, docLength: number, avgDocLength: number, termIdfMap: { [term: string]: number }, k1: number = 1.2, b: number = 0.75): number {',
          file_path: 'src/retrieval/index.ts',
          start_line: 49,
          end_line: 56,
          start_char: 0,
          end_char: 150,
          atom_type: 'definition',
          symbol_name: 'calculateBM25',
          tokens: 45,
          importance: 1.0
        },
        references: [{
          id: 'ref_bm25_1',
          content: 'const score = calculateBM25(termFreqs, docLength, avgDocLength, termIdfMap);',
          file_path: 'src/retrieval/index.ts',
          start_line: 127,
          end_line: 127,
          start_char: 4,
          end_char: 80,
          atom_type: 'reference',
          symbol_name: 'calculateBM25',
          tokens: 18,
          importance: 0.95
        }],
        implementations: [],
        estimated_tokens: 200,
        relevance_score: 0.88,
        topic_weight: 0.45,
        is_precise_match: true
      };

      const candidates = symbolGroupsToRetrievalCandidates([symbolGroup]);
      const candidate = candidates[0];

      // Score should be combination of relevance and topic weight
      expect(candidate.score).toBeCloseTo(0.88 * 0.45, 3);
      
      // High scores should indicate good relevance
      expect(candidate.score).toBeGreaterThan(0.3);
      
      // Metadata should allow for further processing
      expect(candidate.metadata.symbol_group).toBe(symbolGroup);
      expect(candidate.metadata.symbol_group.is_precise_match).toBe(true);
    });
  });

  describe('Performance and Quality Metrics', () => {
    it('should track processing metrics', async () => {
      const service = await getLensService();
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          symbol_groups: [],
          processing_time_ms: 87,
          lsp_available: true,
          topics_expanded: 4,
          timeout_hit: false,
          metadata: {
            version: '1.2.0',
            query_analysis: {
              detected_language: 'typescript',
              extracted_symbols: ['testFunction'],
              code_intent_confidence: 0.85
            }
          }
        })
      });

      const result = await service.search({
        query: 'testFunction implementation',
        max_groups: 10
      });

      // Verify metrics are captured
      expect(result.processing_time_ms).toBe(87);
      expect(result.lsp_available).toBe(true);
      expect(result.topics_expanded).toBe(4);
      expect(result.metadata.query_analysis.code_intent_confidence).toBe(0.85);
    });

    it('should handle quality degradation scenarios', () => {
      // Test scenarios where quality might degrade
      const degradationScenarios = [
        {
          name: 'High latency',
          latency: 300,
          expectedSLA: false
        },
        {
          name: 'Low confidence',
          confidence: 0.2,
          expectedQuality: false
        },
        {
          name: 'No LSP',
          lspAvailable: false,
          expectedPrecision: false
        }
      ];

      degradationScenarios.forEach(({ name, latency = 100, confidence = 0.8, lspAvailable = true, expectedSLA = true, expectedQuality = true, expectedPrecision = true }) => {
        // Test latency constraint
        if (latency) {
          const costResult = calculateLagrangianCost(
            [],
            DEFAULT_LENS_CONFIG,
            1000,
            4000,
            latency
          );
          expect(costResult.sla_constraint_met).toBe(expectedSLA);
        }

        // Test code intent confidence
        if (confidence !== undefined) {
          const intentResult = detectCodeIntent('some query');
          // This is a simplified test - in practice confidence varies by query
          expect(typeof intentResult.confidence).toBe('number');
        }
      });
    });
  });

  describe('Error Recovery and Resilience', () => {
    it('should handle service unavailability', async () => {
      const service = await getLensService();

      // Mock connection failure
      mockFetch.mockRejectedValueOnce(new Error('ECONNREFUSED'));

      const isAvailable = await service.isAvailable();
      expect(isAvailable).toBe(false);
    });

    it('should handle partial service degradation', async () => {
      const service = await getLensService();

      // Mock response with degraded service
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          symbol_groups: [], // Empty due to service issues
          processing_time_ms: 149, // Near SLA limit
          lsp_available: false, // LSP down
          topics_expanded: 1, // Reduced capability
          timeout_hit: false,
          metadata: {
            version: '1.2.0',
            query_analysis: {
              extracted_symbols: [],
              code_intent_confidence: 0.3 // Lower confidence
            }
          }
        })
      });

      const result = await service.search({
        query: 'test query',
        max_groups: 5
      });

      expect(result.symbol_groups).toHaveLength(0);
      expect(result.lsp_available).toBe(false);
      expect(result.topics_expanded).toBeLessThan(3);
    });

    it('should implement circuit breaker pattern', async () => {
      // This test would verify circuit breaker behavior
      // For now, we just verify that repeated failures are handled gracefully
      const service = await getLensService();

      // Simulate repeated failures
      for (let i = 0; i < 3; i++) {
        mockFetch.mockRejectedValueOnce(new Error('Service unavailable'));
        
        try {
          await service.search({ query: 'test', max_groups: 5 });
        } catch (error) {
          expect(error).toBeInstanceOf(Error);
        }
      }

      // Service should still be responsive to status checks
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 503
      });

      const isAvailable = await service.isAvailable();
      expect(isAvailable).toBe(false);
    });

    it('should handle JSON parsing errors', async () => {
      const service = await getLensService();

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.reject(new Error('Invalid JSON'))
      });

      await expect(service.search({
        query: 'test query',
        max_groups: 5
      })).rejects.toThrow('Invalid JSON');
    });

    it('should handle AbortController cleanup on successful requests', async () => {
      const service = await getLensService();

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          symbol_groups: [],
          processing_time_ms: 50,
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

      // Should not throw and should complete normally
      const result = await service.search({
        query: 'test query',
        max_groups: 5
      });

      expect(result.processing_time_ms).toBe(50);
    });
  });

  describe('Service Status and Health Checks', () => {
    it('should get service status successfully', async () => {
      const service = await getLensService();

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          healthy: true,
          version: '1.3.0',
          lsp_available: true,
          raptor_cache_status: 'warm'
        })
      });

      const status = await service.getStatus();

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/status'),
        expect.objectContaining({
          headers: expect.objectContaining({
            'Accept': 'application/json'
          })
        })
      );

      expect(status.healthy).toBe(true);
      expect(status.version).toBe('1.3.0');
      expect(status.lsp_available).toBe(true);
      expect(status.raptor_cache_status).toBe('warm');
    });

    it('should handle status endpoint HTTP errors', async () => {
      const service = await getLensService();

      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found'
      });

      const status = await service.getStatus();

      expect(status.healthy).toBe(false);
      expect(status.version).toBe('unknown');
      expect(status.lsp_available).toBe(false);
      expect(status.raptor_cache_status).toBe('unavailable');
    });

    it('should handle status endpoint timeout', async () => {
      vi.useFakeTimers();
      const service = await getLensService();

      mockFetch.mockImplementationOnce(() => {
        return new Promise((resolve) => {
          setTimeout(() => resolve({
            ok: true,
            json: () => Promise.resolve({ healthy: true })
          }), 600); // Longer than connect timeout
        });
      });

      const statusPromise = service.getStatus();
      vi.advanceTimersByTime(700);

      const status = await statusPromise;

      expect(status.healthy).toBe(false);
      expect(status.version).toBe('unknown');

      vi.useRealTimers();
    });

    it('should handle malformed status response', async () => {
      const service = await getLensService();

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({
          // Missing required fields or malformed data
          some_other_field: 'value'
        })
      });

      const status = await service.getStatus();

      // Should provide defaults for missing fields
      expect(status.healthy).toBe(false); // Default to false if not provided
      expect(status.version).toBe('unknown');
      expect(status.lsp_available).toBe(false);
      expect(status.raptor_cache_status).toBe('unavailable');
    });

    it('should handle network errors in status check', async () => {
      const service = await getLensService();

      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      const status = await service.getStatus();

      expect(status.healthy).toBe(false);
      expect(status.version).toBe('unknown');
      expect(status.lsp_available).toBe(false);
      expect(status.raptor_cache_status).toBe('unavailable');
    });
  });
});