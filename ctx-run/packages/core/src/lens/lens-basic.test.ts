/**
 * Basic tests for Lens integration
 * Tests core functionality, code intent detection, and cost analysis
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  detectCodeIntent,
  calculateLagrangianCost,
  getLensService,
  symbolGroupsToRetrievalCandidates,
  testLensIntegration,
  DEFAULT_LENS_CONFIG,
  type SymbolGroup,
  type CodeIntentResult,
  type LagrangianCostResult
} from './index.js';

describe('Lens Integration - Basic Tests', () => {
  describe('Code Intent Detection', () => {
    it('should detect code intent for function queries', () => {
      const result = detectCodeIntent('fix error in calculateBM25 function');
      
      expect(result.is_code_intent).toBe(true);
      expect(result.confidence).toBeGreaterThan(0.5);
      expect(result.patterns.has_code_symbols).toBe(true);
      expect(result.patterns.has_error_tokens).toBe(true);
      expect(result.extracted_symbols).toContain('calculateBM25');
    });

    it('should detect code intent for file path queries', () => {
      const result = detectCodeIntent('show me the implementation in src/retrieval/index.ts');
      
      expect(result.is_code_intent).toBe(true);
      expect(result.patterns.has_file_paths).toBe(true);
      expect(result.detected_language).toBe('javascript');
    });

    it('should detect code intent for error messages', () => {
      const result = detectCodeIntent('TypeError: cannot read property length of undefined');
      
      expect(result.is_code_intent).toBe(true);
      expect(result.patterns.has_error_tokens).toBe(true);
      expect(result.confidence).toBeGreaterThan(0.3);
    });

    it('should NOT detect code intent for general queries', () => {
      const result = detectCodeIntent('what is the weather today');
      
      expect(result.is_code_intent).toBe(false);
      expect(result.confidence).toBeLessThan(0.3);
    });

    it('should boost confidence with recent code activity', () => {
      const withoutContext = detectCodeIntent('fix bug');
      const withContext = detectCodeIntent('fix bug', ['src/index.ts', 'src/utils.js'], 'code');
      
      expect(withContext.confidence).toBeGreaterThan(withoutContext.confidence);
    });

    it('should extract multiple symbols from complex queries', () => {
      const result = detectCodeIntent('refactor getUserData() and updateUserProfile() methods');
      
      expect(result.extracted_symbols).toContain('getUserData');
      expect(result.extracted_symbols).toContain('updateUserProfile');
      expect(result.extracted_symbols).toHaveLength(2);
    });

    it('should handle language-specific patterns', () => {
      const tests = [
        { query: 'python flask application error', expected: 'python' },
        { query: 'rust cargo build failed', expected: 'rust' },
        { query: 'golang http handler', expected: 'go' },
        { query: 'java spring boot configuration', expected: 'java' },
        { query: 'c++ header file missing', expected: 'cpp' }
      ];

      tests.forEach(({ query, expected }) => {
        const result = detectCodeIntent(query);
        expect(result.detected_language).toBe(expected);
        expect(result.is_code_intent).toBe(true);
      });
    });
  });

  describe('Lagrangian Cost Analysis', () => {
    let mockSymbolGroups: SymbolGroup[];

    beforeEach(() => {
      mockSymbolGroups = [
        {
          id: 'group_1',
          primary_symbol: 'testFunction',
          language: 'typescript',
          file_path: 'src/test.ts',
          definition: {
            id: 'def_1',
            content: 'function testFunction(param: string): boolean { return true; }',
            file_path: 'src/test.ts',
            start_line: 10,
            end_line: 10,
            start_char: 0,
            end_char: 58,
            atom_type: 'definition',
            symbol_name: 'testFunction',
            tokens: 15,
            importance: 1.0
          },
          references: [],
          implementations: [],
          estimated_tokens: 100,
          relevance_score: 0.8,
          topic_weight: 0.3,
          is_precise_match: true
        },
        {
          id: 'group_2',
          primary_symbol: 'helperMethod',
          language: 'typescript',
          file_path: 'src/helper.ts',
          definition: {
            id: 'def_2',
            content: 'private helperMethod(): void {}',
            file_path: 'src/helper.ts',
            start_line: 5,
            end_line: 5,
            start_char: 0,
            end_char: 31,
            atom_type: 'definition',
            symbol_name: 'helperMethod',
            tokens: 8,
            importance: 0.6
          },
          references: [],
          implementations: [],
          estimated_tokens: 50,
          relevance_score: 0.6,
          topic_weight: 0.2,
          is_precise_match: false
        }
      ];
    });

    it('should calculate basic token and compute costs', () => {
      const result = calculateLagrangianCost(
        mockSymbolGroups,
        DEFAULT_LENS_CONFIG,
        1000, // current tokens
        4000, // total budget
        120   // estimated latency
      );

      expect(result.token_cost).toBeGreaterThan(0);
      expect(result.compute_cost).toBeGreaterThan(0);
      expect(result.total_cost).toBe(result.token_cost + result.compute_cost);
      expect(result.cost_breakdown.lens_tokens).toBe(150); // 100 + 50
      expect(result.cost_breakdown.base_tokens).toBe(1000);
    });

    it('should respect lambda multiplier for token costs', () => {
      const standardConfig = { ...DEFAULT_LENS_CONFIG, lambda_multiplier: 1.0 };
      const highLambdaConfig = { ...DEFAULT_LENS_CONFIG, lambda_multiplier: 2.0 };

      const standardCost = calculateLagrangianCost(mockSymbolGroups, standardConfig, 1000, 4000, 120);
      const highLambdaCost = calculateLagrangianCost(mockSymbolGroups, highLambdaConfig, 1000, 4000, 120);

      expect(highLambdaCost.token_cost).toBe(standardCost.token_cost * 2);
    });

    it('should respect mu multiplier for compute costs', () => {
      const standardConfig = { ...DEFAULT_LENS_CONFIG, mu_multiplier: 1.0 };
      const highMuConfig = { ...DEFAULT_LENS_CONFIG, mu_multiplier: 1.5 };

      const standardCost = calculateLagrangianCost(mockSymbolGroups, standardConfig, 1000, 4000, 120);
      const highMuCost = calculateLagrangianCost(mockSymbolGroups, highMuConfig, 1000, 4000, 120);

      expect(highMuCost.compute_cost).toBe(standardCost.compute_cost * 1.5);
    });

    it('should enforce SLA constraint', () => {
      const withinSLA = calculateLagrangianCost(mockSymbolGroups, DEFAULT_LENS_CONFIG, 1000, 4000, 120);
      const outsideSLA = calculateLagrangianCost(mockSymbolGroups, DEFAULT_LENS_CONFIG, 1000, 4000, 200);

      expect(withinSLA.sla_constraint_met).toBe(true);
      expect(outsideSLA.sla_constraint_met).toBe(false);
    });

    it('should reject costs when budget exceeded', () => {
      const result = calculateLagrangianCost(
        mockSymbolGroups,
        DEFAULT_LENS_CONFIG,
        3900, // already near budget
        4000, // total budget
        120
      );

      expect(result.cost_acceptable).toBe(false);
    });

    it('should calculate expected benefit based on precision and diversity', () => {
      const result = calculateLagrangianCost(mockSymbolGroups, DEFAULT_LENS_CONFIG, 1000, 4000, 120);

      expect(result.expected_benefit).toBeGreaterThan(0);
      expect(result.cost_benefit_ratio).toBeGreaterThan(0);
      
      // Should have precision spine bonus for group_1
      const precisionGroups = mockSymbolGroups.filter(g => g.is_precise_match);
      expect(precisionGroups.length).toBe(1);
    });
  });

  describe('Symbol Group to Retrieval Candidate Conversion', () => {
    it('should convert symbol groups to retrieval candidates', () => {
      const symbolGroups: SymbolGroup[] = [{
        id: 'test_group',
        primary_symbol: 'testSymbol',
        language: 'typescript',
        file_path: 'test.ts',
        definition: {
          id: 'def_1',
          content: 'function testSymbol() {}',
          file_path: 'test.ts',
          start_line: 1,
          end_line: 1,
          start_char: 0,
          end_char: 22,
          atom_type: 'definition',
          symbol_name: 'testSymbol',
          tokens: 6,
          importance: 1.0
        },
        references: [],
        implementations: [],
        estimated_tokens: 50,
        relevance_score: 0.9,
        topic_weight: 0.4,
        is_precise_match: true
      }];

      const candidates = symbolGroupsToRetrievalCandidates(symbolGroups);

      expect(candidates).toHaveLength(1);
      expect(candidates[0].docId).toBe('lens_test_group');
      expect(candidates[0].score).toBe(0.9 * 0.4); // relevance * topic_weight
      expect(candidates[0].kind).toBe('code_symbol');
      expect(candidates[0].metadata.is_lens_result).toBe(true);
      expect(candidates[0].text).toContain('testSymbol');
      expect(candidates[0].text).toContain('function testSymbol() {}');
    });

    it('should format text with definition, references, and implementations', () => {
      const symbolGroup: SymbolGroup = {
        id: 'complex_group',
        primary_symbol: 'ComplexClass',
        language: 'typescript',
        file_path: 'src/complex.ts',
        definition: {
          id: 'def_1',
          content: 'class ComplexClass {\n  constructor() {}\n}',
          file_path: 'src/complex.ts',
          start_line: 1,
          end_line: 3,
          start_char: 0,
          end_char: 40,
          atom_type: 'definition',
          symbol_name: 'ComplexClass',
          tokens: 12,
          importance: 1.0
        },
        references: [{
          id: 'ref_1',
          content: 'const instance = new ComplexClass();',
          file_path: 'src/usage.ts',
          start_line: 10,
          end_line: 10,
          start_char: 0,
          end_char: 37,
          atom_type: 'reference',
          symbol_name: 'ComplexClass',
          tokens: 8,
          importance: 0.8
        }],
        implementations: [{
          id: 'impl_1',
          content: 'class ExtendedClass extends ComplexClass {}',
          file_path: 'src/extended.ts',
          start_line: 5,
          end_line: 5,
          start_char: 0,
          end_char: 43,
          atom_type: 'implementation',
          symbol_name: 'ComplexClass',
          tokens: 9,
          importance: 0.7
        }],
        test_hints: [{
          id: 'test_1',
          content: 'describe("ComplexClass", () => { /* tests */ });',
          file_path: 'src/complex.test.ts',
          start_line: 1,
          end_line: 1,
          start_char: 0,
          end_char: 47,
          atom_type: 'test',
          symbol_name: 'ComplexClass',
          tokens: 10,
          importance: 0.5
        }],
        estimated_tokens: 200,
        relevance_score: 0.95,
        topic_weight: 0.6,
        is_precise_match: true
      };

      const candidates = symbolGroupsToRetrievalCandidates([symbolGroup]);
      const text = candidates[0].text;

      expect(text).toContain('Symbol: ComplexClass');
      expect(text).toContain('Definition:');
      expect(text).toContain('class ComplexClass');
      expect(text).toContain('Key References:');
      expect(text).toContain('const instance = new ComplexClass()');
      expect(text).toContain('Implementations:');
      expect(text).toContain('class ExtendedClass extends ComplexClass');
      expect(text).toContain('Test Examples:');
      expect(text).toContain('describe("ComplexClass"');
    });
  });

  describe('Service Integration', () => {
    it('should create lens service with default config', async () => {
      const service = await getLensService();
      expect(service).toBeDefined();
      expect(typeof service.search).toBe('function');
      expect(typeof service.isAvailable).toBe('function');
    });

    it('should handle disabled lens service', async () => {
      const service = await getLensService();
      
      // Mock disabled config
      const mockService = {
        ...service,
        search: vi.fn().mockResolvedValue({
          symbol_groups: [],
          processing_time_ms: 0,
          lsp_available: false,
          topics_expanded: 0,
          timeout_hit: false,
          metadata: {
            version: 'disabled',
            query_analysis: {
              extracted_symbols: [],
              code_intent_confidence: 0
            }
          }
        })
      };

      const result = await mockService.search({
        query: 'test query',
        max_groups: 5
      });

      expect(result.symbol_groups).toHaveLength(0);
      expect(result.metadata.version).toBe('disabled');
    });
  });

  describe('Integration Test Suite', () => {
    it('should run comprehensive test suite', async () => {
      // This will test against mocked responses since we don't have a real server
      const result = await testLensIntegration();

      // Test should indicate service is not available (no real server)
      // but other components should work
      expect(result).toHaveProperty('available');
      expect(result).toHaveProperty('code_intent_test');
      expect(result).toHaveProperty('cost_analysis_test');

      // Code intent and cost analysis should pass even without server
      if (result.code_intent_test !== undefined) {
        expect(result.code_intent_test).toBe(true);
      }
      if (result.cost_analysis_test !== undefined) {
        expect(result.cost_analysis_test).toBe(true);
      }
    });
  });
});

describe('Edge Cases and Error Handling', () => {
  it('should handle empty symbol groups in cost calculation', () => {
    const result = calculateLagrangianCost(
      [], // empty groups
      DEFAULT_LENS_CONFIG,
      1000,
      4000,
      100
    );

    expect(result.token_cost).toBeGreaterThanOrEqual(0);
    expect(result.compute_cost).toBe(0);
    expect(result.expected_benefit).toBe(0);
  });

  it('should handle malformed queries in code intent detection', () => {
    const testCases = ['', '   ', '\n\t\n', '!@#$%^&*()'];
    
    testCases.forEach(query => {
      const result = detectCodeIntent(query);
      expect(result).toHaveProperty('is_code_intent');
      expect(result).toHaveProperty('confidence');
      expect(result.confidence).toBeGreaterThanOrEqual(0);
      expect(result.confidence).toBeLessThanOrEqual(1);
    });
  });

  it('should handle extreme cost scenarios', () => {
    const largeSymbolGroups: SymbolGroup[] = Array.from({ length: 100 }, (_, i) => ({
      id: `group_${i}`,
      primary_symbol: `symbol${i}`,
      language: 'typescript',
      file_path: `src/file${i}.ts`,
      definition: {
        id: `def_${i}`,
        content: `function symbol${i}() {}`,
        file_path: `src/file${i}.ts`,
        start_line: 1,
        end_line: 1,
        start_char: 0,
        end_char: 20,
        atom_type: 'definition',
        symbol_name: `symbol${i}`,
        tokens: 5,
        importance: 0.5
      },
      references: [],
      implementations: [],
      estimated_tokens: 1000, // Large token count
      relevance_score: 0.5,
      topic_weight: 0.1,
      is_precise_match: false
    }));

    const result = calculateLagrangianCost(
      largeSymbolGroups,
      DEFAULT_LENS_CONFIG,
      5000, // already high token count
      10000, // budget
      500    // high latency
    );

    expect(result.cost_acceptable).toBe(false); // Should reject due to budget
    expect(result.sla_constraint_met).toBe(false); // Should fail SLA
  });

  it('should validate config bounds', () => {
    const invalidConfigs = [
      { ...DEFAULT_LENS_CONFIG, weight_cap: 1.5 }, // > 1.0
      { ...DEFAULT_LENS_CONFIG, topic_fanout_k: -1 }, // negative
      { ...DEFAULT_LENS_CONFIG, sla_recall_ms: 0 }, // zero
      { ...DEFAULT_LENS_CONFIG, lambda_multiplier: -0.5 } // negative
    ];

    invalidConfigs.forEach(config => {
      // The cost calculation should still work but may produce unrealistic results
      expect(() => calculateLagrangianCost([], config, 100, 1000, 100)).not.toThrow();
    });
  });

  it('should handle zero multipliers in cost calculation', () => {
    const zeroMultiplierConfig = {
      ...DEFAULT_LENS_CONFIG,
      lambda_multiplier: 0,
      mu_multiplier: 0
    };

    const mockSymbolGroups: SymbolGroup[] = [{
      id: 'test_group',
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
      estimated_tokens: 50,
      relevance_score: 0.8,
      topic_weight: 0.3,
      is_precise_match: true
    }];

    const result = calculateLagrangianCost(
      mockSymbolGroups,
      zeroMultiplierConfig,
      1000,
      4000,
      100
    );

    expect(result.token_cost).toBe(0);
    expect(result.compute_cost).toBe(0);
    expect(result.total_cost).toBe(0);
    expect(result.cost_acceptable).toBe(true); // No cost should be acceptable
  });

  it('should handle extreme precision spine scenarios', () => {
    const allPreciseGroups: SymbolGroup[] = Array.from({ length: 10 }, (_, i) => ({
      id: `precise_${i}`,
      primary_symbol: `preciseSymbol${i}`,
      language: 'typescript',
      file_path: `src/precise${i}.ts`,
      definition: {
        id: `def_precise_${i}`,
        content: `function preciseSymbol${i}() {}`,
        file_path: `src/precise${i}.ts`,
        start_line: 1,
        end_line: 1,
        start_char: 0,
        end_char: 30,
        atom_type: 'definition',
        symbol_name: `preciseSymbol${i}`,
        tokens: 8,
        importance: 1.0
      },
      references: [],
      implementations: [],
      estimated_tokens: 80,
      relevance_score: 0.9,
      topic_weight: 0.4,
      is_precise_match: true // All are precise matches
    }));

    const result = calculateLagrangianCost(
      allPreciseGroups,
      DEFAULT_LENS_CONFIG,
      1000,
      8000,
      120
    );

    // Should have high expected benefit due to precision spine bonus
    expect(result.expected_benefit).toBeGreaterThan(3.0); // 10 * 0.3 precision bonus
    expect(result.cost_benefit_ratio).toBeLessThan(2.0);
  });

  it('should handle single symbol group DPP compute cost', () => {
    const mockSymbolGroup: SymbolGroup = {
      id: 'single_group',
      primary_symbol: 'singleFunction',
      language: 'typescript',
      file_path: 'single.ts',
      definition: {
        id: 'def_single',
        content: 'function singleFunction() {}',
        file_path: 'single.ts',
        start_line: 1,
        end_line: 1,
        start_char: 0,
        end_char: 28,
        atom_type: 'definition',
        symbol_name: 'singleFunction',
        tokens: 6,
        importance: 1.0
      },
      references: [],
      implementations: [],
      estimated_tokens: 50,
      relevance_score: 0.8,
      topic_weight: 0.3,
      is_precise_match: true
    };
    
    const singleGroup = [mockSymbolGroup];
    
    const result = calculateLagrangianCost(
      singleGroup,
      DEFAULT_LENS_CONFIG,
      1000,
      4000,
      100
    );

    // DPP cost should be 0 for single group
    expect(result.cost_breakdown.dpp_compute).toBe(0);
    expect(result.cost_breakdown.ce_compute).toBe(0.5); // One group * 0.5
  });

  it('should handle infinity cost-benefit ratio', () => {
    const zeroBenefitGroups: SymbolGroup[] = [{
      id: 'zero_benefit',
      primary_symbol: 'zeroSymbol',
      language: 'typescript',
      file_path: 'src/zero.ts',
      definition: {
        id: 'def_zero',
        content: 'const zeroSymbol = null;',
        file_path: 'src/zero.ts',
        start_line: 1,
        end_line: 1,
        start_char: 0,
        end_char: 24,
        atom_type: 'definition',
        symbol_name: 'zeroSymbol',
        tokens: 6,
        importance: 0.0
      },
      references: [],
      implementations: [],
      estimated_tokens: 50,
      relevance_score: 0.0,
      topic_weight: 0.0, // Zero topic weight
      is_precise_match: false
    }];

    const result = calculateLagrangianCost(
      zeroBenefitGroups,
      DEFAULT_LENS_CONFIG,
      1000,
      4000,
      100
    );

    expect(result.expected_benefit).toBe(0);
    expect(result.cost_benefit_ratio).toBe(Infinity);
    expect(result.cost_acceptable).toBe(false);
  });
});

describe('Advanced Code Intent Detection', () => {
  it('should detect complex programming patterns', () => {
    const complexPatterns = [
      {
        query: 'async/await pattern in getUserData() method',
        expectedSymbols: ['getUserData'],
        expectedLanguage: 'javascript',
        expectedConfidence: 0.6
      },
      {
        query: 'std::vector<int> implementation error',
        expectedSymbols: ['std::vector'],
        expectedLanguage: 'cpp',
        expectedConfidence: 0.7
      },
      {
        query: 'np.array reshape function not working',
        expectedSymbols: ['np.array'],
        expectedLanguage: 'python',
        expectedConfidence: 0.8
      },
      {
        query: 'torch.nn.Module forward pass issue',
        expectedSymbols: ['torch.nn.Module'],
        expectedLanguage: 'python',
        expectedConfidence: 0.8
      }
    ];

    complexPatterns.forEach(({ query, expectedSymbols, expectedLanguage, expectedConfidence }) => {
      const result = detectCodeIntent(query);
      
      expect(result.is_code_intent).toBe(true);
      expect(result.confidence).toBeGreaterThanOrEqual(expectedConfidence);
      expect(result.detected_language).toBe(expectedLanguage);
      expectedSymbols.forEach(symbol => {
        expect(result.extracted_symbols.join(' ')).toContain(symbol);
      });
    });
  });

  it('should handle multi-language mixed queries', () => {
    const multiLangQuery = 'convert Python pandas DataFrame to Rust Vec<struct>';
    const result = detectCodeIntent(multiLangQuery);

    expect(result.is_code_intent).toBe(true);
    expect(result.confidence).toBeGreaterThan(0.5);
    expect(result.patterns.has_language_keywords).toBe(true);
    // Should detect one of the languages (implementation chooses the first match)
    expect(['python', 'rust'].includes(result.detected_language || '')).toBe(true);
  });

  it('should handle false positive edge cases', () => {
    const falsePositiveCases = [
      'function of government in society',
      'class schedule for next semester',
      'import duties on foreign goods',
      'module 3 homework assignment'
    ];

    falsePositiveCases.forEach(query => {
      const result = detectCodeIntent(query);
      // These should either be low confidence or false
      expect(result.confidence < 0.5 || !result.is_code_intent).toBe(true);
    });
  });

  it('should boost confidence with strong recent context', () => {
    const recentFiles = [
      'src/utils/helpers.ts',
      'src/components/Button.jsx',
      'tests/integration/api.test.js'
    ];
    
    const query = 'button component not rendering';
    const resultWithContext = detectCodeIntent(query, recentFiles, 'code');
    const resultWithoutContext = detectCodeIntent(query);

    expect(resultWithContext.confidence).toBeGreaterThan(resultWithoutContext.confidence);
    expect(resultWithContext.confidence).toBeGreaterThan(0.5);
  });

  it('should handle numeric IDs and issue references', () => {
    const issueQueries = [
      'fix bug #1234 in authentication',
      'implement feature from issue 567',
      'PR #890 breaking change analysis',
      'ticket 12345 performance regression'
    ];

    issueQueries.forEach(query => {
      const result = detectCodeIntent(query);
      expect(result.is_code_intent).toBe(true);
      expect(result.patterns.has_numeric_ids).toBe(true);
      expect(result.confidence).toBeGreaterThan(0.4);
    });
  });

  it('should extract multiple symbols from complex expressions', () => {
    const complexQuery = 'refactor UserService.authenticate() and TokenManager.refresh() to use AuthProvider.validate()';
    const result = detectCodeIntent(complexQuery);

    expect(result.extracted_symbols).toContain('UserService');
    expect(result.extracted_symbols).toContain('TokenManager');
    expect(result.extracted_symbols).toContain('AuthProvider');
    expect(result.extracted_symbols.length).toBeGreaterThanOrEqual(3);
    expect(result.confidence).toBeGreaterThan(0.7);
  });

  it('should handle path patterns on different operating systems', () => {
    const pathTests = [
      {
        query: 'error in /home/user/project/src/main.py line 42',
        expectedPath: true,
        os: 'unix'
      },
      {
        query: 'C:\\Users\\dev\\project\\src\\Main.java compilation failed',
        expectedPath: true,
        os: 'windows'
      },
      {
        query: 'src/components/Button.tsx type error',
        expectedPath: true,
        os: 'relative'
      },
      {
        query: 'build/dist/bundle.js.map missing',
        expectedPath: true,
        os: 'build'
      }
    ];

    pathTests.forEach(({ query, expectedPath }) => {
      const result = detectCodeIntent(query);
      expect(result.patterns.has_file_paths).toBe(expectedPath);
      expect(result.is_code_intent).toBe(true);
      expect(result.confidence).toBeGreaterThan(0.5);
    });
  });
});

describe('Symbol Group Text Formatting Edge Cases', () => {
  it('should format minimal symbol groups', () => {
    const minimalGroup: SymbolGroup = {
      id: 'minimal',
      primary_symbol: 'x',
      language: 'js',
      file_path: 'x.js',
      definition: {
        id: 'def_min',
        content: 'let x;',
        file_path: 'x.js',
        start_line: 1,
        end_line: 1,
        start_char: 0,
        end_char: 6,
        atom_type: 'definition',
        symbol_name: 'x',
        tokens: 2,
        importance: 0.1
      },
      references: [],
      implementations: [],
      estimated_tokens: 5,
      relevance_score: 0.1,
      topic_weight: 0.1,
      is_precise_match: false
    };

    const candidates = symbolGroupsToRetrievalCandidates([minimalGroup]);
    const text = candidates[0].text;

    expect(text).toContain('Symbol: x');
    expect(text).toContain('x.js');
    expect(text).toContain('Definition:');
    expect(text).toContain('let x;');
    // Should not contain reference/implementation sections for empty arrays
    expect(text).not.toContain('Key References:');
    expect(text).not.toContain('Implementations:');
  });

  it('should handle symbol groups with many references', () => {
    const manyReferencesGroup: SymbolGroup = {
      id: 'many_refs',
      primary_symbol: 'popularFunction',
      language: 'typescript',
      file_path: 'src/popular.ts',
      definition: {
        id: 'def_popular',
        content: 'export function popularFunction(arg: string): string { return arg.toUpperCase(); }',
        file_path: 'src/popular.ts',
        start_line: 1,
        end_line: 1,
        start_char: 0,
        end_char: 82,
        atom_type: 'definition',
        symbol_name: 'popularFunction',
        tokens: 20,
        importance: 1.0
      },
      references: Array.from({ length: 10 }, (_, i) => ({
        id: `ref_${i}`,
        content: `const result${i} = popularFunction('test${i}');`,
        file_path: `src/file${i}.ts`,
        start_line: 10 + i,
        end_line: 10 + i,
        start_char: 0,
        end_char: 40,
        atom_type: 'reference' as const,
        symbol_name: 'popularFunction',
        tokens: 8,
        importance: 1.0 - (i * 0.1) // Decreasing importance
      })),
      implementations: [],
      estimated_tokens: 300,
      relevance_score: 0.9,
      topic_weight: 0.5,
      is_precise_match: true
    };

    const candidates = symbolGroupsToRetrievalCandidates([manyReferencesGroup]);
    const text = candidates[0].text;

    expect(text).toContain('Key References:');
    // Should only show top 3 references
    expect(text).toContain('result0 = popularFunction');
    expect(text).toContain('result1 = popularFunction');
    expect(text).toContain('result2 = popularFunction');
    // Should not show all 10 references
    expect(text).not.toContain('result9 = popularFunction');
  });

  it('should handle special characters in symbol names', () => {
    const specialCharsGroup: SymbolGroup = {
      id: 'special',
      primary_symbol: 'user_data_$handler',
      language: 'javascript',
      file_path: 'src/special-chars.js',
      definition: {
        id: 'def_special',
        content: 'const user_data_$handler = (data) => { /* handle user data */ };',
        file_path: 'src/special-chars.js',
        start_line: 1,
        end_line: 1,
        start_char: 0,
        end_char: 65,
        atom_type: 'definition',
        symbol_name: 'user_data_$handler',
        tokens: 15,
        importance: 0.8
      },
      references: [],
      implementations: [],
      estimated_tokens: 80,
      relevance_score: 0.7,
      topic_weight: 0.3,
      is_precise_match: true
    };

    const candidates = symbolGroupsToRetrievalCandidates([specialCharsGroup]);
    const text = candidates[0].text;

    expect(text).toContain('Symbol: user_data_$handler');
    expect(text).toContain('const user_data_$handler');
  });

  it('should sort references by importance', () => {
    const sortedRefsGroup: SymbolGroup = {
      id: 'sorted_refs',
      primary_symbol: 'sortedFunction',
      language: 'typescript',
      file_path: 'src/sorted.ts',
      definition: {
        id: 'def_sorted',
        content: 'function sortedFunction() {}',
        file_path: 'src/sorted.ts',
        start_line: 1,
        end_line: 1,
        start_char: 0,
        end_char: 26,
        atom_type: 'definition',
        symbol_name: 'sortedFunction',
        tokens: 6,
        importance: 1.0
      },
      references: [
        {
          id: 'ref_low',
          content: 'sortedFunction(); // low importance',
          file_path: 'src/low.ts',
          start_line: 1,
          end_line: 1,
          start_char: 0,
          end_char: 35,
          atom_type: 'reference',
          symbol_name: 'sortedFunction',
          tokens: 8,
          importance: 0.2 // Lowest importance
        },
        {
          id: 'ref_high',
          content: 'sortedFunction(); // high importance',
          file_path: 'src/high.ts',
          start_line: 1,
          end_line: 1,
          start_char: 0,
          end_char: 36,
          atom_type: 'reference',
          symbol_name: 'sortedFunction',
          tokens: 8,
          importance: 0.9 // Highest importance
        },
        {
          id: 'ref_mid',
          content: 'sortedFunction(); // mid importance',
          file_path: 'src/mid.ts',
          start_line: 1,
          end_line: 1,
          start_char: 0,
          end_char: 35,
          atom_type: 'reference',
          symbol_name: 'sortedFunction',
          tokens: 8,
          importance: 0.5 // Middle importance
        }
      ],
      implementations: [],
      estimated_tokens: 100,
      relevance_score: 0.8,
      topic_weight: 0.4,
      is_precise_match: true
    };

    const candidates = symbolGroupsToRetrievalCandidates([sortedRefsGroup]);
    const text = candidates[0].text;

    // High importance reference should appear first
    const highImportanceIndex = text.indexOf('high importance');
    const midImportanceIndex = text.indexOf('mid importance');
    const lowImportanceIndex = text.indexOf('low importance');

    expect(highImportanceIndex).toBeLessThan(midImportanceIndex);
    expect(midImportanceIndex).toBeLessThan(lowImportanceIndex);
  });
});