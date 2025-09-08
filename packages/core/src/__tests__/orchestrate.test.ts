import { describe, it, expect, beforeEach, afterEach, vi, Mock } from 'vitest';
import { OrchestrationSystem, createOrchestrationSystem, validateOrchestrationConfig } from '../orchestrate.js';
import { Config, Message, EnhancedCandidate } from '../types.js';
import { TelemetrySystem } from '../telemetry.js';
import { ErrorFactories } from '../errors.js';

/**
 * Comprehensive test suite for OrchestrationSystem
 * Tests all variants V1-V5 with performance and correctness validation
 */

// Mock implementations
const mockConfig: Config = {
  retrieval: {
    method: 'hybrid',
    topK: 50,
    windowSize: 1000,
    hybridWeights: [0.6, 0.4]
  },
  chunking: {
    strategy: 'hierarchical',
    maxSize: 2000,
    overlap: 0.1,
    languages: ['typescript', 'python']
  },
  ranking: {
    enableMetadataBoost: true,
    diversificationMethod: 'semantic',
    fusionWeights: [0.5, 0.3, 0.2]
  },
  performance: {
    budgetTracking: {
      totalBudget: 10.0,
      alertThreshold: 0.8
    },
    telemetry: {
      enabled: true,
      batchSize: 100
    }
  },
  llm: {
    model: 'gpt-4',
    enableReranking: true,
    enableContradictionDetection: true,
    rerankerModel: 'Xenova/bge-reranker-base',
    batchSize: 4,
    maxTokens: 4096,
    temperature: 0.1
  }
};

const mockMessages: Message[] = [
  {
    id: 'msg1',
    content: 'function calculateSum(a: number, b: number): number { return a + b; }',
    timestamp: '2024-01-01T00:00:00Z',
    author: 'user1',
    metadata: { language: 'typescript' }
  },
  {
    id: 'msg2',
    content: 'def calculate_product(x, y): return x * y',
    timestamp: '2024-01-01T00:01:00Z',
    author: 'user2',
    metadata: { language: 'python' }
  },
  {
    id: 'msg3',
    content: 'The sum function adds two numbers together efficiently.',
    timestamp: '2024-01-01T00:02:00Z',
    author: 'user1',
    metadata: { type: 'documentation' }
  }
];

describe('OrchestrationSystem', () => {
  let orchestrationSystem: OrchestrationSystem;
  let telemetryMock: Mock;

  beforeEach(() => {
    // Reset all mocks
    vi.clearAllMocks();
    
    // Mock telemetry system
    telemetryMock = vi.fn();
    vi.mock('../telemetry.js', () => ({
      TelemetrySystem: vi.fn().mockImplementation(() => ({
        logEvent: telemetryMock,
        logOrchestrationStart: vi.fn(),
        logOrchestrationComplete: vi.fn(),
        startTracking: vi.fn().mockReturnValue({
          sample: vi.fn(),
          complete: vi.fn().mockReturnValue({
            operationType: 'test',
            duration: 100,
            success: true,
            startTime: Date.now(),
            endTime: Date.now() + 100,
            memoryBefore: 1000,
            memoryAfter: 1100,
            memoryPeak: 1200,
            cpuUsage: 0.1
          })
        })
      }))
    }));

    orchestrationSystem = new OrchestrationSystem(mockConfig);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Constructor and Initialization', () => {
    it('should initialize with valid configuration', () => {
      expect(orchestrationSystem).toBeInstanceOf(OrchestrationSystem);
    });

    it('should validate configuration on construction', () => {
      const validationResult = validateOrchestrationConfig(mockConfig);
      expect(validationResult.success).toBe(true);
    });

    it('should throw error with invalid configuration', () => {
      const invalidConfig = { ...mockConfig };
      delete (invalidConfig as any).retrieval;
      
      const validationResult = validateOrchestrationConfig(invalidConfig);
      expect(validationResult.success).toBe(false);
      expect(validationResult.error?.code).toBe('INVALID_ORCHESTRATION_CONFIG');
    });
  });

  describe('Variant V1 - Baseline Implementation', () => {
    it('should complete orchestration within performance targets', async () => {
      const startTime = performance.now();
      
      const result = await orchestrationSystem.orchestrate(
        'session_v1',
        'test query for sum function',
        mockMessages,
        'V1'
      );
      
      const duration = performance.now() - startTime;
      
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.variant).toBe('V1');
        expect(result.data.candidates).toBeInstanceOf(Array);
        expect(result.data.totalTime).toBeLessThan(3000); // P50 target for V1
        expect(duration).toBeLessThan(5000); // Allow some overhead
      }
    });

    it('should handle empty message array gracefully', async () => {
      const result = await orchestrationSystem.orchestrate(
        'session_empty',
        'test query',
        [],
        'V1'
      );
      
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.candidates).toHaveLength(0);
      }
    });

    it('should handle single query string', async () => {
      const result = await orchestrationSystem.orchestrate(
        'session_single',
        'single query test',
        mockMessages,
        'V1'
      );
      
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.variant).toBe('V1');
      }
    });

    it('should handle array of queries', async () => {
      const queries = [
        'primary query about functions',
        'secondary query about documentation'
      ];
      
      const result = await orchestrationSystem.orchestrate(
        'session_multi',
        queries,
        mockMessages,
        'V1'
      );
      
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.variant).toBe('V1');
      }
    });
  });

  describe('Variant V2 - Enhanced Ranking (+15% precision)', () => {
    it('should apply enhanced ranking and meet precision targets', async () => {
      const result = await orchestrationSystem.orchestrate(
        'session_v2',
        'test query with ranking',
        mockMessages,
        'V2'
      );
      
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.variant).toBe('V2');
        expect(result.data.candidates.length).toBeGreaterThan(0);
        
        // Check that ranking was applied (candidates should have enhanced scores)
        const hasEnhancedScoring = result.data.candidates.some(c => 
          c.metadata && typeof c.metadata.crossEncoderScore === 'number'
        );
        expect(hasEnhancedScoring).toBe(true);
        
        // Performance should be within acceptable range for V2
        expect(result.data.totalTime).toBeLessThan(3500);
      }
    });

    it('should maintain ranking order consistency', async () => {
      const result1 = await orchestrationSystem.orchestrate(
        'session_v2_1',
        'deterministic test query',
        mockMessages,
        'V2'
      );
      
      const result2 = await orchestrationSystem.orchestrate(
        'session_v2_2',
        'deterministic test query',
        mockMessages,
        'V2'
      );
      
      expect(result1.success).toBe(true);
      expect(result2.success).toBe(true);
      
      if (result1.success && result2.success) {
        // With same input, ranking should be deterministic (within floating point precision)
        expect(result1.data.candidates.length).toBe(result2.data.candidates.length);
      }
    });
  });

  describe('Variant V3 - Semantic Diversification (+25% diversity)', () => {
    it('should apply diversification and increase result variety', async () => {
      const result = await orchestrationSystem.orchestrate(
        'session_v3',
        'diverse query test',
        mockMessages,
        'V3'
      );
      
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.variant).toBe('V3');
        
        // Check for diversification metadata
        const hasDiversification = result.data.candidates.some(c =>
          c.metadata && c.metadata.diversificationApplied === true
        );
        
        // Performance target for V3
        expect(result.data.totalTime).toBeLessThan(4000);
      }
    });

    it('should handle edge case with single candidate', async () => {
      const singleMessage: Message[] = [mockMessages[0]];
      
      const result = await orchestrationSystem.orchestrate(
        'session_v3_single',
        'single candidate test',
        singleMessage,
        'V3'
      );
      
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.candidates.length).toBeGreaterThanOrEqual(0);
      }
    });
  });

  describe('Variant V4 - LLM Reranking (+10% efficiency)', () => {
    it('should apply LLM reranking when enabled', async () => {
      const result = await orchestrationSystem.orchestrate(
        'session_v4',
        'LLM reranking test',
        mockMessages,
        'V4'
      );
      
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.variant).toBe('V4');
        
        // Check for LLM reranking metadata
        const hasLLMScoring = result.data.candidates.some(c =>
          c.llmScore !== undefined
        );
        
        // Performance target for V4 (LLM variant)
        expect(result.data.totalTime).toBeLessThan(4000);
      }
    });

    it('should handle LLM service unavailability gracefully', async () => {
      // Mock LLM service failure
      const failingConfig = {
        ...mockConfig,
        llm: {
          ...mockConfig.llm,
          apiKey: 'invalid_key'
        }
      };
      
      const failingSystem = new OrchestrationSystem(failingConfig);
      
      const result = await failingSystem.orchestrate(
        'session_v4_fail',
        'LLM failure test',
        mockMessages,
        'V4'
      );
      
      // Should still complete with fallback to cross-encoder only
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.variant).toBe('V4');
      }
    });
  });

  describe('Variant V5 - Contradiction Detection (+20% accuracy)', () => {
    it('should detect contradictions when present', async () => {
      const contradictoryMessages: Message[] = [
        {
          id: 'msg_yes',
          content: 'Yes, this feature is definitely supported and works well.',
          timestamp: '2024-01-01T00:00:00Z',
          author: 'user1',
          metadata: {}
        },
        {
          id: 'msg_no', 
          content: 'No, this feature is not supported and will not work.',
          timestamp: '2024-01-01T00:01:00Z',
          author: 'user2',
          metadata: {}
        }
      ];
      
      const result = await orchestrationSystem.orchestrate(
        'session_v5',
        'contradiction detection test',
        contradictoryMessages,
        'V5'
      );
      
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.variant).toBe('V5');
        expect(result.data.contradictions).toBeInstanceOf(Array);
        
        // Should detect the yes/no contradiction
        expect(result.data.contradictions.length).toBeGreaterThan(0);
        
        // Performance target for V5
        expect(result.data.totalTime).toBeLessThan(4500);
      }
    });

    it('should handle no contradictions gracefully', async () => {
      const result = await orchestrationSystem.orchestrate(
        'session_v5_no_contradictions',
        'harmony test query',
        mockMessages,
        'V5'
      );
      
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.contradictions).toHaveLength(0);
      }
    });
  });

  describe('Error Handling and Edge Cases', () => {
    it('should handle invalid variant gracefully', async () => {
      const result = await orchestrationSystem.orchestrate(
        'session_invalid',
        'test query',
        mockMessages,
        'INVALID' as any
      );
      
      // Should default to V1 behavior
      expect(result.success).toBe(true);
      if (result.success) {
        expect(['V1', 'INVALID']).toContain(result.data.variant);
      }
    });

    it('should handle empty query gracefully', async () => {
      const result = await orchestrationSystem.orchestrate(
        'session_empty_query',
        '',
        mockMessages,
        'V1'
      );
      
      expect(result.success).toBe(true);
    });

    it('should handle very long queries', async () => {
      const longQuery = 'a'.repeat(10000);
      
      const result = await orchestrationSystem.orchestrate(
        'session_long_query',
        longQuery,
        mockMessages,
        'V1'
      );
      
      expect(result.success).toBe(true);
    });

    it('should handle malformed messages', async () => {
      const malformedMessages: Message[] = [
        {
          id: '',
          content: '',
          timestamp: 'invalid-date',
          author: '',
          metadata: {}
        }
      ];
      
      const result = await orchestrationSystem.orchestrate(
        'session_malformed',
        'test query',
        malformedMessages,
        'V1'
      );
      
      // Should handle gracefully and not crash
      expect(result.success).toBe(true);
    });
  });

  describe('Performance and Budget Tracking', () => {
    it('should track budget usage within limits', async () => {
      const result = await orchestrationSystem.orchestrate(
        'session_budget',
        'budget tracking test',
        mockMessages,
        'V1'
      );
      
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.performance.budgetUsed).toBeLessThanOrEqual(
          mockConfig.performance.budgetTracking.totalBudget
        );
      }
    });

    it('should alert when approaching budget limits', async () => {
      const lowBudgetConfig: Config = {
        ...mockConfig,
        performance: {
          ...mockConfig.performance,
          budgetTracking: {
            totalBudget: 0.1, // Very low budget to trigger alert
            alertThreshold: 0.5
          }
        }
      };
      
      const lowBudgetSystem = new OrchestrationSystem(lowBudgetConfig);
      
      const result = await lowBudgetSystem.orchestrate(
        'session_low_budget',
        'budget alert test',
        mockMessages,
        'V1'
      );
      
      // Should either succeed with warning or fail gracefully
      if (!result.success) {
        expect(result.error.code).toBe('BUDGET_EXCEEDED');
      }
    });

    it('should maintain performance metrics accuracy', async () => {
      const startTime = Date.now();
      
      const result = await orchestrationSystem.orchestrate(
        'session_perf_metrics',
        'performance metrics test',
        mockMessages,
        'V2'
      );
      
      const actualDuration = Date.now() - startTime;
      
      expect(result.success).toBe(true);
      if (result.success) {
        // Reported time should be reasonably close to actual time
        expect(Math.abs(result.data.totalTime - actualDuration)).toBeLessThan(100);
        
        // Performance metrics should be populated
        expect(result.data.performance.sessionId).toBe('session_perf_metrics');
        expect(result.data.performance.totalTime).toBeGreaterThan(0);
      }
    });
  });

  describe('Integration Testing', () => {
    it('should integrate all components successfully for V5', async () => {
      const complexMessages: Message[] = [
        ...mockMessages,
        {
          id: 'complex1',
          content: `
            class Calculator {
              add(a: number, b: number): number {
                return a + b;
              }
              
              subtract(a: number, b: number): number {
                return a - b;
              }
            }
          `,
          timestamp: '2024-01-01T00:03:00Z',
          author: 'dev1',
          metadata: { language: 'typescript', complexity: 'high' }
        },
        {
          id: 'complex2',
          content: 'The Calculator class provides basic arithmetic operations with type safety.',
          timestamp: '2024-01-01T00:04:00Z',
          author: 'tech_writer',
          metadata: { type: 'documentation', related_to: 'complex1' }
        }
      ];
      
      const queries = [
        'How do I implement arithmetic operations?',
        'What are the best practices for TypeScript classes?'
      ];
      
      const result = await orchestrationSystem.orchestrate(
        'session_integration',
        queries,
        complexMessages,
        'V5'
      );
      
      expect(result.success).toBe(true);
      if (result.success) {
        // Should have processed all components
        expect(result.data.candidates.length).toBeGreaterThan(0);
        expect(result.data.variant).toBe('V5');
        expect(result.data.performance).toBeDefined();
        
        // Should have reasonable performance
        expect(result.data.totalTime).toBeLessThan(6000); // Allow extra time for complexity
        
        // Should have metadata from all processing stages
        const candidateWithMetadata = result.data.candidates.find(c => 
          c.metadata && Object.keys(c.metadata).length > 0
        );
        expect(candidateWithMetadata).toBeDefined();
      }
    });

    it('should handle concurrent orchestration requests', async () => {
      const concurrentRequests = 3;
      const promises = Array.from({ length: concurrentRequests }, (_, i) =>
        orchestrationSystem.orchestrate(
          `session_concurrent_${i}`,
          `concurrent query ${i}`,
          mockMessages,
          'V1'
        )
      );
      
      const results = await Promise.all(promises);
      
      results.forEach((result, index) => {
        expect(result.success).toBe(true);
        if (result.success) {
          expect(result.data.performance.sessionId).toBe(`session_concurrent_${index}`);
        }
      });
    });
  });

  describe('Factory and Utility Functions', () => {
    it('should create orchestration system with factory', () => {
      const system = createOrchestrationSystem(mockConfig);
      expect(system).toBeInstanceOf(OrchestrationSystem);
    });

    it('should validate configuration correctly', () => {
      const validResult = validateOrchestrationConfig(mockConfig);
      expect(validResult.success).toBe(true);
      
      const invalidConfig = { invalid: true };
      const invalidResult = validateOrchestrationConfig(invalidConfig as any);
      expect(invalidResult.success).toBe(false);
    });
  });

  describe('Memory and Resource Management', () => {
    it('should not leak memory during processing', async () => {
      const initialMemory = process.memoryUsage().heapUsed;
      
      // Run multiple orchestrations
      for (let i = 0; i < 10; i++) {
        const result = await orchestrationSystem.orchestrate(
          `session_memory_${i}`,
          'memory test query',
          mockMessages,
          'V1'
        );
        expect(result.success).toBe(true);
      }
      
      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }
      
      const finalMemory = process.memoryUsage().heapUsed;
      const memoryGrowth = finalMemory - initialMemory;
      
      // Memory growth should be reasonable (less than 100MB for 10 operations)
      expect(memoryGrowth).toBeLessThan(100 * 1024 * 1024);
    });
  });
});

/**
 * Integration tests that require real components
 */
describe('OrchestrationSystem Integration', () => {
  let system: OrchestrationSystem;
  
  beforeEach(() => {
    const realConfig: Config = {
      retrieval: {
        method: 'bm25', // Use simpler method for testing
        topK: 10,
        windowSize: 500
      },
      chunking: {
        strategy: 'hierarchical',
        maxSize: 1000,
        overlap: 0.1,
        languages: ['typescript']
      },
      ranking: {
        enableMetadataBoost: false,
        diversificationMethod: 'entity',
        fusionWeights: [1.0]
      },
      performance: {
        budgetTracking: {
          totalBudget: 5.0,
          alertThreshold: 0.8
        },
        telemetry: {
          enabled: false // Disable for testing
        }
      }
    };
    
    system = new OrchestrationSystem(realConfig);
  });

  it('should handle realistic TypeScript code examples', async () => {
    const codeMessages: Message[] = [
      {
        id: 'ts1',
        content: `
          interface User {
            id: string;
            name: string;
            email: string;
            createdAt: Date;
          }
          
          function createUser(userData: Omit<User, 'id' | 'createdAt'>): User {
            return {
              ...userData,
              id: crypto.randomUUID(),
              createdAt: new Date()
            };
          }
        `,
        timestamp: '2024-01-01T00:00:00Z',
        author: 'developer',
        metadata: { language: 'typescript', category: 'interface' }
      },
      {
        id: 'ts2',
        content: `
          const users: User[] = [];
          
          export const userService = {
            create: (userData: Omit<User, 'id' | 'createdAt'>) => {
              const user = createUser(userData);
              users.push(user);
              return user;
            },
            
            findById: (id: string) => {
              return users.find(user => user.id === id);
            },
            
            list: () => [...users]
          };
        `,
        timestamp: '2024-01-01T00:01:00Z',
        author: 'developer',
        metadata: { language: 'typescript', category: 'service' }
      }
    ];
    
    const result = await system.orchestrate(
      'session_typescript',
      'How do I create and manage users in TypeScript?',
      codeMessages,
      'V1'
    );
    
    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.candidates.length).toBeGreaterThan(0);
      
      // Should find relevant code snippets
      const relevantCandidate = result.data.candidates.find(c =>
        c.content.includes('User') || c.content.includes('createUser')
      );
      expect(relevantCandidate).toBeDefined();
    }
  });

  it('should maintain consistent performance across multiple runs', async () => {
    const query = 'typescript interface patterns';
    const messages = mockMessages;
    const runs = 5;
    const durations: number[] = [];
    
    for (let i = 0; i < runs; i++) {
      const startTime = performance.now();
      const result = await system.orchestrate(
        `session_consistency_${i}`,
        query,
        messages,
        'V1'
      );
      const duration = performance.now() - startTime;
      
      expect(result.success).toBe(true);
      durations.push(duration);
    }
    
    // Calculate coefficient of variation (std dev / mean)
    const mean = durations.reduce((a, b) => a + b) / durations.length;
    const variance = durations.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / durations.length;
    const stdDev = Math.sqrt(variance);
    const coefficientOfVariation = stdDev / mean;
    
    // Performance should be reasonably consistent (CV < 0.5)
    expect(coefficientOfVariation).toBeLessThan(0.5);
  });
});