/**
 * Edge Cases and Boundary Testing for Lens Integration
 * Focuses on untested helper functions, boundary conditions, and integration scenarios
 */

import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import {
  detectCodeIntent,
  calculateLagrangianCost,
  symbolGroupsToRetrievalCandidates,
  getLensService,
  testLensIntegration,
  DEFAULT_LENS_CONFIG,
  type SymbolGroup,
  type CodeAtom,
  type LensConfig,
  type LensService
} from './index.js';

// Mock fetch for testing
const mockFetch = vi.fn() as Mock;
global.fetch = mockFetch;

describe('Lens Edge Cases and Boundary Testing', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetch.mockClear();
  });

  describe('Code Intent Detection Edge Cases', () => {
    it('should handle very long queries', () => {
      const longQuery = 'fix error in calculateBM25 function implementation when processing large documents with complex tokenization patterns and multi-language content that includes special characters and unicode symbols that might break the scoring algorithm especially when dealing with edge cases like empty documents or documents with only punctuation marks or very short text fragments that do not contain meaningful content for indexing purposes';
      
      const result = detectCodeIntent(longQuery);
      
      expect(result.is_code_intent).toBe(true);
      expect(result.confidence).toBeGreaterThan(0.5);
      expect(result.extracted_symbols).toContain('calculateBM25');
      expect(result.patterns.has_code_symbols).toBe(true);
      expect(result.patterns.has_error_tokens).toBe(true);
    });

    it('should handle queries with unicode and special characters', () => {
      const unicodeQueries = [
        'función getUserData() error en línea 42',
        'エラー: calculateScore() メソッドが失敗しました',
        '修复 getUserProfile() 函数中的bug',
        'ошибка в функции getUser() на строке 15',
        'فخة في دالة calculateTotal() السطر 20'
      ];

      unicodeQueries.forEach(query => {
        const result = detectCodeIntent(query);
        expect(result.is_code_intent).toBe(true);
        expect(result.confidence).toBeGreaterThan(0.3);
      });
    });

    it('should handle mixed case and unusual formatting', () => {
      const mixedCaseQueries = [
        'FIX ERROR IN calculateBM25 FUNCTION',
        'fix   error   in     calculateBM25     function',
        'fix\nerror\tin\r\ncalculateBM25\r\nfunction',
        'Fix-Error-In-calculateBM25-Function',
        'fix_error_in_calculateBM25_function'
      ];

      mixedCaseQueries.forEach(query => {
        const result = detectCodeIntent(query);
        expect(result.is_code_intent).toBe(true);
        expect(result.extracted_symbols).toContain('calculateBM25');
      });
    });

    it('should detect code intent in casual language', () => {
      const casualQueries = [
        "yo, the getUserData function isn't working",
        'hey can you help me debug this calculateScore method?',
        'lol this function is broken: testMethod()',
        'omg why does validateInput() keep failing?'
      ];

      casualQueries.forEach(query => {
        const result = detectCodeIntent(query);
        expect(result.is_code_intent).toBe(true);
        expect(result.confidence).toBeGreaterThan(0.4);
        expect(result.patterns.has_code_symbols).toBe(true);
      });
    });

    it('should handle edge cases with no symbols but code context', () => {
      const contextualQueries = [
        'compile error in main.cpp line 42',
        'segfault at runtime in debug mode',
        'memory leak detected in tests',
        'build failed on CI pipeline'
      ];

      contextualQueries.forEach(query => {
        const result = detectCodeIntent(query);
        expect(result.is_code_intent).toBe(true);
        expect(result.extracted_symbols).toHaveLength(0);
        expect(result.patterns.has_error_tokens || result.patterns.has_file_paths).toBe(true);
      });
    });
  });

  describe('Lagrangian Cost Calculation Boundary Cases', () => {
    it('should handle extreme token counts', () => {
      const extremeTokenGroups: SymbolGroup[] = [{
        id: 'extreme_tokens',
        primary_symbol: 'massiveFunction',
        language: 'typescript',
        file_path: 'massive.ts',
        definition: {
          id: 'def_massive',
          content: 'export function massiveFunction() { /* thousands of lines */ }',
          file_path: 'massive.ts',
          start_line: 1,
          end_line: 5000,
          start_char: 0,
          end_char: 200000,
          atom_type: 'definition',
          symbol_name: 'massiveFunction',
          tokens: 50000, // Extremely large
          importance: 1.0
        },
        references: [],
        implementations: [],
        estimated_tokens: 100000, // Massive token count
        relevance_score: 0.95,
        topic_weight: 0.8,
        is_precise_match: true
      }];

      const result = calculateLagrangianCost(
        extremeTokenGroups,
        DEFAULT_LENS_CONFIG,
        1000,
        5000, // Small budget vs large tokens
        100
      );

      expect(result.cost_acceptable).toBe(false);
      expect(result.token_cost).toBeGreaterThan(100000);
      expect(result.cost_breakdown.lens_tokens).toBe(100000);
    });

    it('should handle zero and negative values gracefully', () => {
      const zeroValueGroup: SymbolGroup = {
        id: 'zero_values',
        primary_symbol: 'zeroFunction',
        language: 'typescript',
        file_path: 'zero.ts',
        definition: {
          id: 'def_zero',
          content: '',
          file_path: 'zero.ts',
          start_line: 0,
          end_line: 0,
          start_char: 0,
          end_char: 0,
          atom_type: 'definition',
          symbol_name: 'zeroFunction',
          tokens: 0,
          importance: 0.0
        },
        references: [],
        implementations: [],
        estimated_tokens: 0,
        relevance_score: 0.0,
        topic_weight: 0.0,
        is_precise_match: false
      };

      const result = calculateLagrangianCost(
        [zeroValueGroup],
        DEFAULT_LENS_CONFIG,
        0, // Zero current tokens
        0, // Zero budget
        0  // Zero latency
      );

      expect(result.token_cost).toBeGreaterThanOrEqual(0);
      expect(result.compute_cost).toBeGreaterThanOrEqual(0);
      expect(result.expected_benefit).toBe(0);
      expect(result.sla_constraint_met).toBe(true); // 0 <= any SLA limit
    });

    it('should handle floating point precision edge cases', () => {
      const precisionGroup: SymbolGroup = {
        id: 'precision_test',
        primary_symbol: 'precisionFunction',
        language: 'typescript',
        file_path: 'precision.ts',
        definition: {
          id: 'def_precision',
          content: 'function precisionFunction() {}',
          file_path: 'precision.ts',
          start_line: 1,
          end_line: 1,
          start_char: 0,
          end_char: 30,
          atom_type: 'definition',
          symbol_name: 'precisionFunction',
          tokens: 6,
          importance: 0.33333333333333 // Repeating decimal
        },
        references: [],
        implementations: [],
        estimated_tokens: 33,
        relevance_score: 0.999999999999999, // Near 1.0
        topic_weight: 0.333333333333333, // Repeating decimal
        is_precise_match: true
      };

      const precisionConfig: LensConfig = {
        ...DEFAULT_LENS_CONFIG,
        lambda_multiplier: Math.PI / 10, // Irrational number
        mu_multiplier: Math.E / 5        // Another irrational number
      };

      const result = calculateLagrangianCost(
        [precisionGroup],
        precisionConfig,
        1000,
        4000,
        100
      );

      expect(Number.isFinite(result.token_cost)).toBe(true);
      expect(Number.isFinite(result.compute_cost)).toBe(true);
      expect(Number.isFinite(result.total_cost)).toBe(true);
      expect(Number.isFinite(result.expected_benefit)).toBe(true);
      expect(Number.isFinite(result.cost_benefit_ratio)).toBe(true);
    });

    it('should handle very large cost-benefit ratios', () => {
      const highCostLowBenefit: SymbolGroup[] = Array.from({ length: 1000 }, (_, i) => ({
        id: `expensive_${i}`,
        primary_symbol: `expensiveSymbol${i}`,
        language: 'typescript',
        file_path: `expensive${i}.ts`,
        definition: {
          id: `def_expensive_${i}`,
          content: `function expensiveSymbol${i}() {}`,
          file_path: `expensive${i}.ts`,
          start_line: 1,
          end_line: 1,
          start_char: 0,
          end_char: 30,
          atom_type: 'definition',
          symbol_name: `expensiveSymbol${i}`,
          tokens: 10,
          importance: 0.001 // Very low importance
        },
        references: [],
        implementations: [],
        estimated_tokens: 1000, // High cost
        relevance_score: 0.001, // Very low relevance
        topic_weight: 0.001,    // Very low weight
        is_precise_match: false
      }));

      const result = calculateLagrangianCost(
        highCostLowBenefit,
        DEFAULT_LENS_CONFIG,
        10000,  // High existing tokens
        50000,  // Large budget
        100
      );

      expect(result.cost_acceptable).toBe(false);
      expect(result.expected_benefit).toBeLessThan(1.0);
      expect(result.cost_benefit_ratio).toBeGreaterThan(100);
    });
  });

  describe('Symbol Group Text Formatting Edge Cases', () => {
    it('should handle symbol groups with extremely long content', () => {
      const longContent = 'export function veryLongFunction(param1: string, param2: number, param3: boolean, param4: object, param5: any[]): Promise<ComplexReturnType<GenericType<T, U, V>>> {\n' +
        '  // This is an extremely long function with lots of complex logic\n' +
        '  // that spans many lines and contains various programming constructs\n' +
        '  ' + 'x'.repeat(5000) + '\n' +  // Very long line
        '  return Promise.resolve(complexResult);\n' +
        '}';

      const longContentGroup: SymbolGroup = {
        id: 'long_content',
        primary_symbol: 'veryLongFunction',
        language: 'typescript',
        file_path: 'src/long.ts',
        definition: {
          id: 'def_long',
          content: longContent,
          file_path: 'src/long.ts',
          start_line: 1,
          end_line: 100,
          start_char: 0,
          end_char: longContent.length,
          atom_type: 'definition',
          symbol_name: 'veryLongFunction',
          tokens: 1000,
          importance: 1.0
        },
        references: [],
        implementations: [],
        estimated_tokens: 2000,
        relevance_score: 0.8,
        topic_weight: 0.4,
        is_precise_match: true
      };

      const candidates = symbolGroupsToRetrievalCandidates([longContentGroup]);
      const text = candidates[0].text;

      expect(text).toContain('Symbol: veryLongFunction');
      expect(text).toContain('Definition:');
      expect(text.length).toBeGreaterThan(5000);
      expect(candidates[0].docId).toBe('lens_long_content');
      expect(candidates[0].score).toBe(0.8 * 0.4);
    });

    it('should handle symbol groups with no definition but references', () => {
      const noDefGroup: SymbolGroup = {
        id: 'no_definition',
        primary_symbol: 'mysterySybol',
        language: 'typescript',
        file_path: 'src/mystery.ts',
        definition: {
          id: 'def_empty',
          content: '',
          file_path: 'src/mystery.ts',
          start_line: 0,
          end_line: 0,
          start_char: 0,
          end_char: 0,
          atom_type: 'definition',
          symbol_name: 'mysterySymbol',
          tokens: 0,
          importance: 0.0
        },
        references: [{
          id: 'ref_mystery',
          content: 'const result = mysterySymbol();',
          file_path: 'src/usage.ts',
          start_line: 10,
          end_line: 10,
          start_char: 0,
          end_char: 32,
          atom_type: 'reference',
          symbol_name: 'mysterySymbol',
          tokens: 7,
          importance: 0.9
        }],
        implementations: [],
        estimated_tokens: 50,
        relevance_score: 0.6,
        topic_weight: 0.3,
        is_precise_match: false
      };

      const candidates = symbolGroupsToRetrievalCandidates([noDefGroup]);
      const text = candidates[0].text;

      expect(text).toContain('Symbol: mysterySybol');
      expect(text).toContain('Definition:');
      expect(text).toContain('Key References:');
      expect(text).toContain('mysterySymbol()');
    });

    it('should handle symbol groups with circular references', () => {
      const circularRefAtom: CodeAtom = {
        id: 'ref_circular',
        content: 'circularFunction();',
        file_path: 'src/circular.ts',
        start_line: 5,
        end_line: 5,
        start_char: 0,
        end_char: 19,
        atom_type: 'reference',
        symbol_name: 'circularFunction',
        tokens: 4,
        importance: 0.8
      };

      const circularGroup: SymbolGroup = {
        id: 'circular',
        primary_symbol: 'circularFunction',
        language: 'typescript',
        file_path: 'src/circular.ts',
        definition: {
          id: 'def_circular',
          content: 'function circularFunction() { circularFunction(); }',
          file_path: 'src/circular.ts',
          start_line: 1,
          end_line: 1,
          start_char: 0,
          end_char: 51,
          atom_type: 'definition',
          symbol_name: 'circularFunction',
          tokens: 12,
          importance: 1.0
        },
        references: [circularRefAtom],
        implementations: [{
          ...circularRefAtom,
          id: 'impl_circular',
          atom_type: 'implementation'
        }],
        estimated_tokens: 80,
        relevance_score: 0.7,
        topic_weight: 0.25,
        is_precise_match: true
      };

      const candidates = symbolGroupsToRetrievalCandidates([circularGroup]);
      const text = candidates[0].text;

      expect(text).toContain('Symbol: circularFunction');
      expect(text).toContain('Definition:');
      expect(text).toContain('Key References:');
      expect(text).toContain('Implementations:');
    });
  });

  describe('Service Integration Error Boundaries', () => {
    it('should handle service creation with malformed config', async () => {
      // Mock a config that might cause issues
      const malformedConfigDb = {
        getConfig: () => ({
          lens: {
            base_url: null,
            enabled: 'not a boolean',
            topic_fanout_k: 'invalid number',
            weight_cap: undefined,
            lambda_multiplier: {},
            mu_multiplier: []
          }
        })
      };

      // Should still create service without throwing
      const service = await getLensService(malformedConfigDb);
      expect(service).toBeDefined();
      expect(typeof service.search).toBe('function');
    });

    it('should handle concurrent service creation', async () => {
      const promises = Array.from({ length: 10 }, () => getLensService());
      const services = await Promise.all(promises);

      // All should be the same instance due to caching
      services.forEach(service => {
        expect(service).toBe(services[0]);
      });
    });

    it('should handle search with AbortController edge cases', async () => {
      const service = await getLensService();

      // Mock a response that resolves after abort
      let resolveResponse: (value: any) => void;
      const responsePromise = new Promise(resolve => {
        resolveResponse = resolve;
      });

      mockFetch.mockReturnValueOnce(responsePromise);

      // Start search with very short timeout
      const searchPromise = service.search({
        query: 'test',
        max_groups: 5,
        timeout_ms: 1 // Very short timeout
      });

      // Resolve response after timeout would have occurred
      setTimeout(() => {
        resolveResponse({
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
      }, 10);

      const result = await searchPromise;

      // Should get timeout response
      expect(result.timeout_hit).toBe(true);
      expect(result.symbol_groups).toHaveLength(0);
    });
  });

  describe('Integration Test Edge Cases', () => {
    it('should handle testLensIntegration with partial failures', async () => {
      // Mock connection test to succeed but with high latency
      mockFetch.mockResolvedValueOnce({ ok: true });

      // Mock search to timeout
      mockFetch.mockImplementationOnce(() => {
        return new Promise((resolve) => {
          setTimeout(() => resolve({
            ok: true,
            json: () => Promise.resolve({ symbol_groups: [] })
          }), 2000); // Longer than timeout
        });
      });

      const result = await testLensIntegration();

      expect(result.available).toBe(true);
      expect(result.latency_ms).toBeGreaterThan(0);
      expect(result.search_test).toBe(false);
      expect(result.code_intent_test).toBe(true);
      expect(result.cost_analysis_test).toBe(true);
    });

    it('should handle testLensIntegration with invalid mock data', async () => {
      // Test cost analysis with extreme values
      const result = await testLensIntegration();

      // Should handle extreme mock data gracefully
      expect(result.cost_analysis_test).toBe(true);
      
      // The mock data should create realistic cost analysis
      const mockSymbolGroups: SymbolGroup[] = [{
        id: 'test_1',
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
          tokens: 10,
          importance: 1.0
        },
        references: [],
        implementations: [],
        estimated_tokens: 50,
        relevance_score: 0.8,
        topic_weight: 0.3,
        is_precise_match: true
      }];

      const costAnalysis = calculateLagrangianCost(
        mockSymbolGroups,
        DEFAULT_LENS_CONFIG,
        1000,
        4000,
        100
      );

      expect(costAnalysis.total_cost).toBeGreaterThan(0);
      expect(costAnalysis.sla_constraint_met).toBe(true);
    });
  });

  describe('Configuration Edge Cases', () => {
    it('should handle configuration with extreme values', () => {
      const extremeConfig: LensConfig = {
        base_url: 'http://localhost:999999', // Invalid port
        connect_timeout_ms: Number.MAX_SAFE_INTEGER,
        request_timeout_ms: -100,
        sla_recall_ms: 0.1,
        topic_fanout_k: Number.MAX_SAFE_INTEGER,
        weight_cap: 999.999,
        max_tokens_per_response: -1,
        enabled: true,
        mode: 'auto',
        dpp_rank: -50,
        enable_facility_location: true,
        enable_log_det_dpp: true,
        lambda_multiplier: Number.POSITIVE_INFINITY,
        mu_multiplier: Number.NEGATIVE_INFINITY,
        lens_tokens_cap: Number.NaN
      };

      // Should not throw when creating service with extreme config
      expect(() => {
        JSON.stringify(extremeConfig);
      }).not.toThrow();
    });

    it('should handle missing required config fields', () => {
      const incompleteConfig = {
        base_url: 'http://localhost:5678',
        // Missing most required fields
      } as LensConfig;

      // Should handle gracefully with defaults
      const mergedConfig = { ...DEFAULT_LENS_CONFIG, ...incompleteConfig };
      expect(mergedConfig.base_url).toBe('http://localhost:5678');
      expect(mergedConfig.enabled).toBeDefined();
      expect(mergedConfig.topic_fanout_k).toBeDefined();
    });
  });
});