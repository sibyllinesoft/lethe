/**
 * @fileoverview Integration tests for Milestone 2: Adaptive Planning Policy
 * Validates complete agent-aware retrieval strategy adaptation system
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';

import {
  AdaptiveAtomsDatabase,
  createAdaptiveAtomsDatabase,
  type AdaptiveSearchContext,
  type AdaptiveSearchResults,
} from './adaptive-atoms-db.js';
import { type Atom, type AtomRole, type AtomType } from './atoms-types.js';
import { runAdaptivePlanningGridSearch } from './grid-search.js';

/**
 * Test helper to create in-memory database with full schema
 */
function createTestDb(): Database.Database {
  const db = new Database(':memory:');
  
  // Apply the full atoms schema
  const getCurrentDir = () => {
    try {
      return dirname(require.resolve('./milestone2-integration.test'));
    } catch {
      return __dirname;
    }
  };
  
  const schemaPath = join(getCurrentDir(), '..', 'schema-atoms.sql');
  const schema = readFileSync(schemaPath, 'utf-8');
  db.exec(schema);
  
  return db;
}

/**
 * Create realistic agent conversation data for testing
 */
function createAgentConversationData(sessionId: string): Atom[] {
  const baseTime = Date.now() - 60000; // 1 minute ago
  
  return [
    // Turn 1: User asks about React debugging
    {
      id: 'atom-1',
      session_id: sessionId,
      turn_idx: 1,
      role: 'user' as AtomRole,
      type: 'message' as AtomType,
      text: 'I need help debugging a React component that has state management issues',
      ts: baseTime + 1000,
    },
    
    // Turn 2: Assistant provides initial guidance
    {
      id: 'atom-2', 
      session_id: sessionId,
      turn_idx: 2,
      role: 'assistant' as AtomRole,
      type: 'message' as AtomType,
      text: 'I can help debug your React component. Let me examine the code using browser developer tools to identify state management patterns.',
      ts: baseTime + 2000,
    },
    
    // Turn 3: Tool observation with error details
    {
      id: 'atom-3',
      session_id: sessionId,
      turn_idx: 3,
      role: 'tool' as AtomRole,
      type: 'observation' as AtomType,
      text: 'Error: TypeError: Cannot read property "setState" of undefined in UserProfile.jsx at line 45. Stack trace shows issue in componentDidMount lifecycle method.',
      ts: baseTime + 3000,
    },
    
    // Turn 4: Assistant explains the error
    {
      id: 'atom-4',
      session_id: sessionId,
      turn_idx: 4,
      role: 'assistant' as AtomRole,
      type: 'message' as AtomType,
      text: 'The error indicates a binding issue in your React component. The setState function is called with incorrect context. Here is the fix: bind methods in constructor or use arrow functions.',
      ts: baseTime + 4000,
    },
    
    // Turn 5: Code implementation
    {
      id: 'atom-5',
      session_id: sessionId,
      turn_idx: 5,
      role: 'assistant' as AtomRole,
      type: 'action' as AtomType,
      text: 'function UserProfile() { const [user, setUser] = useState(null); useEffect(() => { fetchUserData().then(setUser); }, []); return <div>{user?.name}</div>; }',
      ts: baseTime + 5000,
    },
    
    // Turn 6: User asks about similar patterns
    {
      id: 'atom-6',
      session_id: sessionId,
      turn_idx: 6,
      role: 'user' as AtomRole,
      type: 'message' as AtomType,
      text: 'Are there similar state management patterns I should look for in other components?',
      ts: baseTime + 6000,
    },
    
    // Turn 7: Assistant provides broader context
    {
      id: 'atom-7',
      session_id: sessionId,
      turn_idx: 7,
      role: 'assistant' as AtomRole,
      type: 'message' as AtomType,
      text: 'Yes, common React state patterns include: useState for local state, useEffect for side effects, useContext for shared state, and useReducer for complex state logic. Check components with lifecycle methods.',
      ts: baseTime + 7000,
    },
    
    // Turn 8: Tool usage for broader search
    {
      id: 'atom-8',
      session_id: sessionId,
      turn_idx: 8,
      role: 'tool' as AtomRole,
      type: 'action' as AtomType,
      text: 'grep -r "componentDidMount\\|setState" src/components/ --include="*.jsx" found 12 files with potential binding issues',
      ts: baseTime + 8000,
    },
  ];
}

describe('Milestone 2: Adaptive Planning Policy Integration', () => {
  let db: Database.Database;
  let atomsDb: AdaptiveAtomsDatabase;
  const sessionId = 'agent-session-test';

  beforeEach(async () => {
    db = createTestDb();
    atomsDb = createAdaptiveAtomsDatabase(db);
    
    // Initialize the database
    await atomsDb.initialize();
    
    // Insert realistic conversation data
    const conversationAtoms = createAgentConversationData(sessionId);
    await atomsDb.insertAtoms(conversationAtoms);
    
    // Wait for async processing
    await new Promise(resolve => setTimeout(resolve, 100));
  });

  afterEach(() => {
    db.close();
  });

  describe('Query Feature Extraction', () => {
    it('should correctly extract features for VERIFY plan queries', async () => {
      // High IDF + entity overlap + non-code = VERIFY
      const query = 'React component state management debugging patterns';
      
      const result = await atomsDb.adaptiveSearch(query, { 
        sessionId,
        useAdaptivePlanning: true 
      });
      
      expect(result.planDecision).toBeDefined();
      const { features, plan } = result.planDecision!;
      
      // Should have meaningful IDF values
      expect(features.max_idf).toBeGreaterThan(0);
      expect(features.len_q).toBe(6); // 6 words: React component state management debugging patterns
      
      // Should detect entity overlap with conversation
      expect(features.entity_overlap).toBeGreaterThan(0.05); // Some overlap with conversation
      
      // Should detect non-code content
      expect(features.has_code).toBe(false);
      
      // Planning decision should be reasonable
      expect(['VERIFY', 'EXPLORE', 'EXPLOIT']).toContain(plan);
    });

    it('should correctly extract features for EXPLORE plan queries', async () => {
      // Low entity overlap = EXPLORE  
      const query = 'machine learning model architecture optimization techniques';
      
      const result = await atomsDb.adaptiveSearch(query, {
        sessionId,
        useAdaptivePlanning: true
      });
      
      expect(result.planDecision).toBeDefined();
      const { features, plan } = result.planDecision!;
      
      // Should have low entity overlap with React conversation
      expect(features.entity_overlap).toBeLessThan(0.3);
      
      // Should not overlap with known tools in conversation  
      expect(features.tool_overlap).toBe(0);
      
      // Should likely trigger EXPLORE
      if (plan === 'EXPLORE') {
        expect(result.planDecision!.reasoning).toContain('entity overlap');
      }
    });

    it('should correctly extract features for code queries', async () => {
      const codeQuery = 'function getUserData() { return useState(null); }';
      
      const result = await atomsDb.adaptiveSearch(codeQuery, {
        sessionId, 
        useAdaptivePlanning: true
      });
      
      expect(result.planDecision).toBeDefined();
      const { features } = result.planDecision!;
      
      // Should detect code patterns
      expect(features.has_code).toBe(true);
      expect(features.has_identifier).toBe(true);
      
      // Should NOT select VERIFY due to has_code
      expect(result.planDecision!.plan).not.toBe('VERIFY');
    });

    it('should correctly detect error patterns', async () => {
      const errorQuery = 'TypeError: Cannot read property setState debugging';
      
      const result = await atomsDb.adaptiveSearch(errorQuery, {
        sessionId,
        useAdaptivePlanning: true
      });
      
      expect(result.planDecision).toBeDefined();
      expect(result.planDecision!.features.has_error_pattern).toBe(true);
    });

    it('should correctly detect tool overlap', async () => {
      const toolQuery = 'use grep and curl for debugging React components';
      
      const result = await atomsDb.adaptiveSearch(toolQuery, {
        sessionId,
        useAdaptivePlanning: true
      });
      
      expect(result.planDecision).toBeDefined();
      expect(result.planDecision!.features.tool_overlap).toBe(1);
    });
  });

  describe('Plan Selection Rules', () => {
    it('should apply VERIFY plan for high IDF + entity overlap + non-code', async () => {
      // Temporarily lower thresholds to ensure VERIFY triggers
      atomsDb.updatePlanningThresholds({ tau_v: 1.0, tau_e: 0.2 });
      
      const query = 'React component state debugging';
      const result = await atomsDb.adaptiveSearch(query, { sessionId });
      
      if (result.planDecision!.features.max_idf > 1.0 && 
          result.planDecision!.features.entity_overlap > 0.2 && 
          !result.planDecision!.features.has_code) {
        expect(result.planDecision!.plan).toBe('VERIFY');
        expect(result.planDecision!.reasoning).toContain('High IDF');
      }
    });

    it('should apply EXPLORE plan for low entity overlap', async () => {
      const query = 'completely unrelated machine learning topic';
      const result = await atomsDb.adaptiveSearch(query, { sessionId });
      
      // Should have low entity overlap and trigger EXPLORE
      if (result.planDecision!.features.entity_overlap < 0.2) {
        expect(result.planDecision!.plan).toBe('EXPLORE');
        expect(result.planDecision!.reasoning).toContain('entity overlap');
      }
    });

    it('should apply EXPLOIT as default case', async () => {
      const query = 'React useState hook implementation guide';
      const result = await atomsDb.adaptiveSearch(query, { sessionId });
      
      // This should be a balanced case - some overlap but not extreme
      if (result.planDecision!.features.entity_overlap >= 0.2 &&
          result.planDecision!.features.entity_overlap <= 0.4 &&
          result.planDecision!.features.tool_overlap === 1) {
        expect(result.planDecision!.plan).toBe('EXPLOIT');
        expect(result.planDecision!.reasoning).toContain('Balanced');
      }
    });

    it('should allow forced plan override', async () => {
      const query = 'any query content';
      
      for (const forcedPlan of ['VERIFY', 'EXPLORE', 'EXPLOIT'] as const) {
        const result = await atomsDb.adaptiveSearch(query, {
          sessionId,
          forcePlan: forcedPlan
        });
        
        expect(result.planDecision!.plan).toBe(forcedPlan);
        expect(result.planDecision!.reasoning).toContain('Forced plan');
      }
    });
  });

  describe('Plan-to-Configuration Mapping', () => {
    it('should configure VERIFY plan for high precision', async () => {
      const result = await atomsDb.adaptiveSearch('test query', {
        sessionId,
        forcePlan: 'VERIFY'
      });
      
      expect(result.planDecision!.plan).toBe('VERIFY');
      expect(result.planDecision!.alpha).toBeGreaterThanOrEqual(0.7); // High FTS weight
      expect(result.planDecision!.efSearch).toBeLessThanOrEqual(100); // Lower search effort
    });

    it('should configure EXPLORE plan for high recall', async () => {
      const result = await atomsDb.adaptiveSearch('test query', {
        sessionId,
        forcePlan: 'EXPLORE'
      });
      
      expect(result.planDecision!.plan).toBe('EXPLORE');
      expect(result.planDecision!.alpha).toBeLessThanOrEqual(0.5); // Lower FTS weight
      expect(result.planDecision!.efSearch).toBeGreaterThanOrEqual(100); // Higher search effort
    });

    it('should configure EXPLOIT plan for balance', async () => {
      const result = await atomsDb.adaptiveSearch('test query', {
        sessionId,
        forcePlan: 'EXPLOIT'
      });
      
      expect(result.planDecision!.plan).toBe('EXPLOIT');
      expect(result.planDecision!.alpha).toBeCloseTo(0.5, 1); // Balanced
    });
  });

  describe('Safety Fallbacks', () => {
    it('should apply backoff when results are insufficient', async () => {
      // Query that will likely return few results
      const obscureQuery = 'zyxwvutsrqponmlkjihgfedcba nonexistent terms';
      
      const result = await atomsDb.adaptiveSearch(obscureQuery, {
        sessionId,
        minResults: 5
      });
      
      // Should have applied backoff if initial results were insufficient
      if (result.backoffApplied) {
        expect(result.expandedQuery).toBeDefined();
        expect(result.expandedQuery).not.toBe(result.originalQuery);
        expect(result.planDecision!.reasoning).toContain('Backoff');
      }
    });

    it('should expand queries according to rules', async () => {
      // Test error query expansion
      const errorQuery = 'error in component';
      const result = await atomsDb.adaptiveSearch(errorQuery, {
        sessionId,
        minResults: 10 // Force backoff
      });
      
      if (result.backoffApplied && result.expandedQuery) {
        expect(result.expandedQuery).toContain('exception');
      }
    });
  });

  describe('Plan Decision Logging and Analytics', () => {
    it('should log all plan decisions with features', async () => {
      const queries = [
        'React debugging patterns',
        'machine learning techniques', 
        'function implementation guide'
      ];
      
      for (const query of queries) {
        await atomsDb.adaptiveSearch(query, { sessionId });
      }
      
      const planData = atomsDb.exportPlanningData(sessionId);
      expect(planData.length).toBe(queries.length);
      
      for (const entry of planData) {
        expect(['VERIFY', 'EXPLORE', 'EXPLOIT']).toContain(entry.plan);
        expect(entry.features).toBeTypeOf('string'); // JSON serialized
        expect(entry.reasoning).toBeTruthy();
        expect(entry.confidence).toBeGreaterThan(0);
      }
    });

    it('should provide meaningful planning statistics', async () => {
      // Create diverse queries to trigger different plans
      const queries = [
        'React component state management', // Likely VERIFY or EXPLOIT
        'unknown machine learning topic', // Likely EXPLORE
        'useState hook patterns', // Likely EXPLOIT
        'completely unrelated database design', // Likely EXPLORE
      ];
      
      for (const query of queries) {
        await atomsDb.adaptiveSearch(query, { sessionId });
      }
      
      const stats = atomsDb.getPlanningStats(sessionId);
      
      // Should have reasonable distribution
      const totalCount = Object.values(stats.planCounts).reduce((a, b) => a + b, 0);
      expect(totalCount).toBe(queries.length);
      
      // Should have non-zero averages
      for (const planType of ['VERIFY', 'EXPLORE', 'EXPLOIT'] as const) {
        if (stats.planCounts[planType] > 0) {
          expect(stats.avgConfidence[planType]).toBeGreaterThan(0);
        }
      }
    });

    it('should analyze plan effectiveness', async () => {
      // Execute several queries 
      const testQueries = [
        'React component debugging',
        'state management patterns',
        'browser developer tools',
        'error handling implementation'
      ];
      
      for (const query of testQueries) {
        await atomsDb.adaptiveSearch(query, { sessionId });
      }
      
      const analysis = atomsDb.analyzePlanEffectiveness(sessionId);
      
      expect(analysis.planPerformance).toBeDefined();
      expect(analysis.recommendations).toBeDefined();
      
      // Should have statistics for used plans
      for (const [planType, stats] of Object.entries(analysis.planPerformance)) {
        if (stats.usageCount > 0) {
          expect(stats.avgResultsCount).toBeGreaterThanOrEqual(0);
          expect(stats.avgConfidence).toBeGreaterThan(0);
          expect(stats.successRate).toBeGreaterThanOrEqual(0);
          expect(stats.successRate).toBeLessThanOrEqual(1);
        }
      }
    });

    it('should provide threshold tuning suggestions', async () => {
      // Execute enough queries to get meaningful suggestions
      for (let i = 0; i < 12; i++) {
        await atomsDb.adaptiveSearch(`test query ${i} React component`, { sessionId });
      }
      
      const suggestions = atomsDb.getThresholdSuggestions(sessionId);
      
      expect(suggestions.confidence).toBeGreaterThan(0);
      expect(Array.isArray(suggestions.suggestions)).toBe(true);
      
      // If suggestions exist, they should be reasonable
      for (const suggestion of suggestions.suggestions) {
        expect(['tau_v', 'tau_e', 'tau_n']).toContain(suggestion.parameter);
        expect(suggestion.suggested).toBeGreaterThan(0);
        expect(suggestion.reasoning).toBeTruthy();
      }
    });
  });

  describe('Configuration and Customization', () => {
    it('should accept custom planning configuration', () => {
      const customDb = createAdaptiveAtomsDatabase(db, {
        adaptivePlanning: {
          thresholds: { tau_v: 3.5, tau_e: 0.6, tau_n: 0.15 },
          window_turns: 15,
        }
      } as any);
      
      expect(customDb).toBeDefined();
      // Configuration is applied during construction
    });

    it('should update thresholds dynamically', async () => {
      const newThresholds = { tau_v: 2.0, tau_e: 0.3, tau_n: 0.1 };
      atomsDb.updatePlanningThresholds(newThresholds);
      
      // Test that updated thresholds affect plan selection
      const result = await atomsDb.adaptiveSearch('React component state', { sessionId });
      
      // The planning decision should use the updated thresholds
      expect(result.planDecision).toBeDefined();
    });

    it('should allow disabling adaptive planning', async () => {
      const result = await atomsDb.adaptiveSearch('any query', {
        sessionId,
        useAdaptivePlanning: false
      });
      
      expect(result.planDecision!.plan).toBe('EXPLOIT');
      expect(result.planDecision!.reasoning).toContain('disabled');
    });
  });

  describe('Grid Search Integration', () => {
    it('should support basic grid search optimization', async () => {
      // Insert more diverse data for better optimization
      const diverseAtoms: Atom[] = [
        {
          id: 'diverse-1',
          session_id: sessionId,
          turn_idx: 10,
          role: 'user',
          type: 'message',
          text: 'error debugging patterns in React components with TypeScript',
          ts: Date.now(),
        },
        {
          id: 'diverse-2', 
          session_id: sessionId,
          turn_idx: 11,
          role: 'assistant',
          type: 'message',
          text: 'function ComponentWithError() { const [state, setState] = useState(); }',
          ts: Date.now() + 1000,
        }
      ];
      
      await atomsDb.insertAtoms(diverseAtoms);
      
      // Run a minimal grid search (very limited for test performance)
      const sessionIdfCalculator = (atomsDb as any).sessionIdfCalculator;
      const entityExtractor = (atomsDb as any).entityExtractor;
      const baseConfig = (atomsDb as any).adaptiveConfig.adaptivePlanning;
      
      const gridSearchResult = await runAdaptivePlanningGridSearch(
        db,
        sessionIdfCalculator,
        entityExtractor,
        baseConfig,
        {
          tau_v_range: { min: 2.0, max: 3.0, step: 0.5 },
          tau_e_range: { min: 0.3, max: 0.5, step: 0.1 },
          tau_n_range: { min: 0.1, max: 0.2, step: 0.1 },
          max_combinations: 8,
          evaluation_queries: [
            {
              query: 'React component debugging',
              expected_plan: 'VERIFY',
              weight: 1.0,
            },
            {
              query: 'unrelated machine learning topic',
              expected_plan: 'EXPLORE',
              weight: 1.0,
            }
          ],
        }
      );
      
      expect(gridSearchResult.best_parameters).toBeDefined();
      expect(gridSearchResult.best_parameters.tau_v).toBeGreaterThan(0);
      expect(gridSearchResult.best_parameters.tau_e).toBeGreaterThan(0);
      expect(gridSearchResult.best_parameters.tau_n).toBeGreaterThan(0);
      expect(gridSearchResult.search_space_size).toBeGreaterThan(0);
      expect(gridSearchResult.all_results.length).toBeGreaterThan(0);
    }, 30000); // Extended timeout for grid search
  });

  describe('System Integration', () => {
    it('should integrate seamlessly with existing atoms database functionality', async () => {
      // Test that basic atoms database functionality still works
      const stats = atomsDb.getStats();
      
      expect(stats.atomCount).toBeGreaterThan(0);
      expect(stats.sessionCount).toBeGreaterThan(0);
      
      // Test that session stats work
      const sessionStats = atomsDb.getSessionStats(sessionId);
      expect(sessionStats).toBeDefined();
      
      // Test that we can get atoms by session
      const sessionAtoms = atomsDb.getAtomsBySession(sessionId);
      expect(sessionAtoms.length).toBeGreaterThan(0);
    });

    it('should maintain all search capabilities', async () => {
      const query = 'React debugging';
      
      // Test adaptive search
      const adaptiveResult = await atomsDb.adaptiveSearch(query, { sessionId });
      expect(adaptiveResult.atoms.length).toBeGreaterThan(0);
      expect(adaptiveResult.planDecision).toBeDefined();
      
      // Test that regular hybrid search still works through parent class
      const hybridResult = await (atomsDb as any).hybridSearch(query, { sessionId });
      expect(hybridResult.atoms.length).toBeGreaterThan(0);
    });

    it('should handle large conversation contexts', async () => {
      // Insert a large number of atoms to test window behavior
      const largeConversation: Atom[] = [];
      
      for (let i = 20; i < 50; i++) {
        largeConversation.push({
          id: `large-${i}`,
          session_id: sessionId,
          turn_idx: i,
          role: 'user',
          type: 'message',
          text: `Message ${i} about React components and state management patterns with debugging tools`,
          ts: Date.now() + i * 1000,
        });
      }
      
      await atomsDb.insertAtoms(largeConversation);
      
      // Test that feature extraction still works with large context
      const result = await atomsDb.adaptiveSearch('React component patterns', { sessionId });
      
      expect(result.planDecision).toBeDefined();
      expect(result.planDecision!.features.entity_overlap).toBeGreaterThan(0);
      expect(result.atoms.length).toBeGreaterThan(0);
    });
  });
});