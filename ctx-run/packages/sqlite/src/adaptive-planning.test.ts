/**
 * @fileoverview Tests for Adaptive Planning Policy
 * Milestone 2: Test suite for agent-aware retrieval strategy adaptation
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';

// TypeScript declaration for custom matcher
declare module 'vitest' {
  interface Assertion<T = any> {
    toBeOneOf(expected: T[]): T;
  }
}

// Custom vitest matcher for toBeOneOf
expect.extend({
  toBeOneOf(received, expected) {
    const pass = expected.includes(received);
    if (pass) {
      return {
        message: () => `expected ${received} not to be one of ${expected}`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected ${received} to be one of ${expected}`,
        pass: false,
      };
    }
  },
});
import Database from 'better-sqlite3';
import { 
  AdaptivePlanner, 
  createAdaptivePlanner,
  DEFAULT_ADAPTIVE_CONFIG,
  type QueryFeatures,
  type PlanDecision,
  type AdaptivePlanningConfig 
} from './adaptive-planning.js';
import { SessionIdfCalculator, createSessionIdfCalculator } from './session-idf.js';
import { EntityExtractor, createEntityExtractor } from './entity-extraction.js';
import { Atom, Entity, AtomRole, AtomType, EntityKind } from './atoms-types.js';

/**
 * Test helper to create in-memory database with schema
 */
function createTestDb(): Database.Database {
  const db = new Database(':memory:');
  
  // Create atoms schema
  db.exec(`
    CREATE TABLE atoms (
      id TEXT PRIMARY KEY,
      session_id TEXT NOT NULL,
      turn_idx INTEGER NOT NULL,
      role TEXT NOT NULL,
      type TEXT NOT NULL,
      text TEXT NOT NULL,
      json_meta TEXT,
      ts INTEGER NOT NULL
    );
    
    CREATE TABLE entities (
      atom_id TEXT NOT NULL,
      entity TEXT NOT NULL,
      kind TEXT NOT NULL,
      weight REAL NOT NULL,
      FOREIGN KEY (atom_id) REFERENCES atoms(id)
    );
    
    CREATE TABLE session_idf (
      session_id TEXT NOT NULL,
      term TEXT NOT NULL,
      df INTEGER NOT NULL,
      idf REAL NOT NULL,
      updated_at INTEGER NOT NULL,
      PRIMARY KEY (session_id, term)
    );
    
    CREATE INDEX idx_atoms_session ON atoms(session_id, turn_idx);
    CREATE INDEX idx_entities_atom ON entities(atom_id);
  `);
  
  return db;
}

/**
 * Test helper to insert test atoms and entities
 */
function insertTestData(db: Database.Database, sessionId: string): void {
  const insertAtom = db.prepare(`
    INSERT INTO atoms (id, session_id, turn_idx, role, type, text, ts)
    VALUES (?, ?, ?, ?, ?, ?, ?)
  `);
  
  const insertEntity = db.prepare(`
    INSERT INTO entities (atom_id, entity, kind, weight)
    VALUES (?, ?, ?, ?)
  `);
  
  // Create a realistic agent conversation with tools, code, and entities
  const atoms: Array<Omit<Atom, 'json_meta'>> = [
    {
      id: 'atom1',
      session_id: sessionId,
      turn_idx: 1,
      role: 'user' as AtomRole,
      type: 'message' as AtomType,
      text: 'I need to debug an issue with my React component',
      ts: Date.now() - 10000,
    },
    {
      id: 'atom2', 
      session_id: sessionId,
      turn_idx: 2,
      role: 'assistant' as AtomRole,
      type: 'message' as AtomType,
      text: 'I can help you debug the React component. Let me examine the code using the browser developer tools',
      ts: Date.now() - 9000,
    },
    {
      id: 'atom3',
      session_id: sessionId,
      turn_idx: 3,
      role: 'tool' as AtomRole,
      type: 'observation' as AtomType,
      text: 'Error: TypeError: Cannot read property "state" of undefined in UserProfile.jsx line 42',
      ts: Date.now() - 8000,
    },
    {
      id: 'atom4',
      session_id: sessionId,
      turn_idx: 4,
      role: 'assistant' as AtomRole,
      type: 'message' as AtomType,
      text: 'Found the error - the component is trying to access state before initialization. Let me fix this with proper state handling',
      ts: Date.now() - 7000,
    },
    {
      id: 'atom5',
      session_id: sessionId,
      turn_idx: 5,
      role: 'user' as AtomRole,
      type: 'message' as AtomType,
      text: 'Can you also check for any similar patterns in other components?',
      ts: Date.now() - 6000,
    },
  ];
  
  const entities: Entity[] = [
    { atom_id: 'atom1', entity: 'React', kind: 'tool' as EntityKind, weight: 0.8 },
    { atom_id: 'atom1', entity: 'component', kind: 'id' as EntityKind, weight: 0.7 },
    { atom_id: 'atom1', entity: 'debug', kind: 'tool' as EntityKind, weight: 0.6 },
    
    { atom_id: 'atom2', entity: 'browser', kind: 'tool' as EntityKind, weight: 0.5 },
    { atom_id: 'atom2', entity: 'developer tools', kind: 'tool' as EntityKind, weight: 0.8 },
    
    { atom_id: 'atom3', entity: 'TypeError', kind: 'error' as EntityKind, weight: 0.9 },
    { atom_id: 'atom3', entity: 'UserProfile.jsx', kind: 'file' as EntityKind, weight: 0.8 },
    { atom_id: 'atom3', entity: 'state', kind: 'id' as EntityKind, weight: 0.7 },
    
    { atom_id: 'atom4', entity: 'state', kind: 'id' as EntityKind, weight: 0.8 },
    { atom_id: 'atom4', entity: 'initialization', kind: 'id' as EntityKind, weight: 0.6 },
    
    { atom_id: 'atom5', entity: 'patterns', kind: 'misc' as EntityKind, weight: 0.5 },
    { atom_id: 'atom5', entity: 'components', kind: 'id' as EntityKind, weight: 0.7 },
  ];
  
  // Insert atoms
  for (const atom of atoms) {
    insertAtom.run(atom.id, atom.session_id, atom.turn_idx, atom.role, atom.type, atom.text, atom.ts);
  }
  
  // Insert entities  
  for (const entity of entities) {
    insertEntity.run(entity.atom_id, entity.entity, entity.kind, entity.weight);
  }
}

describe('AdaptivePlanner', () => {
  let db: Database.Database;
  let sessionIdf: SessionIdfCalculator;
  let entityExtractor: EntityExtractor;
  let planner: AdaptivePlanner;
  const sessionId = 'test-session-123';

  beforeEach(() => {
    db = createTestDb();
    insertTestData(db, sessionId);
    
    sessionIdf = createSessionIdfCalculator(db);
    entityExtractor = createEntityExtractor();
    planner = createAdaptivePlanner(db, sessionIdf, entityExtractor);
    
    // Compute session IDF for test data
    sessionIdf.recomputeSessionIdf(sessionId);
  });

  afterEach(() => {
    db.close();
  });

  describe('Query Feature Extraction', () => {
    it('should extract basic IDF features correctly', async () => {
      const query = 'React component state initialization error';
      const features = await planner.extractQueryFeatures(query, sessionId);
      
      // Should have meaningful IDF values
      expect(features.max_idf).toBeGreaterThan(0);
      expect(features.avg_idf).toBeGreaterThan(0);
      expect(features.len_q).toBe(5); // 5 tokens
    });

    it('should calculate entity overlap with recent history', async () => {
      const query = 'React component state error patterns'; // Overlaps with conversation entities
      const features = await planner.extractQueryFeatures(query, sessionId);
      
      // Should detect overlap with entities from conversation
      expect(features.entity_overlap).toBeGreaterThan(0.3); // Significant overlap
    });

    it('should detect code patterns', async () => {
      const queries = [
        'function getUserData() { return state.user; }',
        'const MyComponent = () => { const [count, setCount] = useState(0); }',
        'class UserProfile extends React.Component',
      ];
      
      for (const query of queries) {
        const features = await planner.extractQueryFeatures(query, sessionId);
        expect(features.has_code).toBe(true);
      }
    });

    it('should detect error patterns', async () => {
      const queries = [
        'TypeError: Cannot read property',
        'Error: Failed to fetch data',
        'Exception occurred in UserService',
        '404 error when calling API',
      ];
      
      for (const query of queries) {
        const features = await planner.extractQueryFeatures(query, sessionId);
        expect(features.has_error_pattern).toBe(true);
      }
    });

    it('should detect identifier patterns', async () => {
      const queries = [
        'getUserData() function implementation',
        'UserProfile.render() method',
        '${userData} variable access',
      ];
      
      for (const query of queries) {
        const features = await planner.extractQueryFeatures(query, sessionId);
        expect(features.has_identifier).toBe(true);
      }
    });

    it('should calculate tool overlap correctly', async () => {
      const queryWithTools = 'use git and npm to fix React component';
      const queryWithoutTools = 'general implementation question about design patterns';
      
      const featuresWithTools = await planner.extractQueryFeatures(queryWithTools, sessionId);
      const featuresWithoutTools = await planner.extractQueryFeatures(queryWithoutTools, sessionId);
      
      expect(featuresWithTools.tool_overlap).toBe(1);
      expect(featuresWithoutTools.tool_overlap).toBe(0);
    });
  });

  describe('Plan Selection Logic', () => {
    it('should select VERIFY for high IDF + entity overlap + non-code query', () => {
      const features: QueryFeatures = {
        max_idf: 3.0,        // > tau_v (2.5)
        avg_idf: 2.0,
        len_q: 4,
        entity_overlap: 0.6,  // > tau_e (0.4)
        tool_overlap: 1,
        has_code: false,      // Non-code query
        has_error_pattern: false,
        has_identifier: false,
        recent_tool_count: 2,
        turn_position: 0.5,
      };
      
      const result = planner.selectPlan(features);
      expect(result.plan).toBe('VERIFY');
      expect(result.reasoning).toContain('High IDF');
      expect(result.reasoning).toContain('entity overlap');
      expect(result.reasoning).toContain('non-code');
      expect(result.confidence).toBeGreaterThan(0.8);
    });

    it('should select EXPLORE for low entity overlap', () => {
      const features: QueryFeatures = {
        max_idf: 1.5,
        avg_idf: 1.0,
        len_q: 3,
        entity_overlap: 0.1,  // < tau_n (0.2)  
        tool_overlap: 1,
        has_code: false,
        has_error_pattern: false,
        has_identifier: false,
        recent_tool_count: 1,
        turn_position: 0.3,
      };
      
      const result = planner.selectPlan(features);
      expect(result.plan).toBe('EXPLORE');
      expect(result.reasoning).toContain('Low entity overlap');
    });

    it('should select EXPLORE for no tool overlap', () => {
      const features: QueryFeatures = {
        max_idf: 2.0,
        avg_idf: 1.5,
        len_q: 4,
        entity_overlap: 0.3,  // Between tau_n and tau_e
        tool_overlap: 0,      // No tool overlap
        has_code: false,
        has_error_pattern: false,
        has_identifier: false,
        recent_tool_count: 0,
        turn_position: 0.7,
      };
      
      const result = planner.selectPlan(features);
      expect(result.plan).toBe('EXPLORE');
      expect(result.reasoning).toContain('no tool overlap');
    });

    it('should select EXPLOIT as default case', () => {
      const features: QueryFeatures = {
        max_idf: 2.0,        // < tau_v
        avg_idf: 1.5,
        len_q: 4,
        entity_overlap: 0.3, // Between tau_n and tau_e
        tool_overlap: 1,     // Has tool overlap
        has_code: false,
        has_error_pattern: false,
        has_identifier: false,
        recent_tool_count: 2,
        turn_position: 0.5,
      };
      
      const result = planner.selectPlan(features);
      expect(result.plan).toBe('EXPLOIT');
      expect(result.reasoning).toContain('Balanced case');
    });

    it('should NOT select VERIFY for code-heavy queries even with high IDF', () => {
      const features: QueryFeatures = {
        max_idf: 3.0,        // > tau_v
        avg_idf: 2.0,
        len_q: 6,
        entity_overlap: 0.6, // > tau_e
        tool_overlap: 1,
        has_code: true,      // Code-heavy query
        has_error_pattern: false,
        has_identifier: true,
        recent_tool_count: 2,
        turn_position: 0.4,
      };
      
      const result = planner.selectPlan(features);
      expect(result.plan).not.toBe('VERIFY'); // Should be EXPLOIT due to has_code
    });
  });

  describe('Plan Decision Integration', () => {
    it('should make complete plan decisions with correct weights', async () => {
      const query = 'React component debugging with browser tools';
      const decision = await planner.makePlanDecision(query, sessionId);
      
      expect(decision.plan).toBeOneOf(['VERIFY', 'EXPLORE', 'EXPLOIT']);
      expect(decision.features).toBeDefined();
      expect(decision.reasoning).toBeTypeOf('string');
      expect(decision.confidence).toBeGreaterThan(0);
      expect(decision.alpha).toBeGreaterThan(0);
      expect(decision.efSearch).toBeGreaterThan(0);
      expect(decision.timestamp).toBeCloseTo(Date.now(), -2); // Within ~100ms
    });

    it('should apply correct weights for each plan type', async () => {
      // Force different plan types by adjusting config temporarily
      const originalConfig = planner.getConfig();
      
      // Test VERIFY weights
      planner.updateThresholds({ tau_v: 0.1, tau_e: 0.1 }); // Lower thresholds
      const verifyQuery = 'simple non-code query';
      const verifyDecision = await planner.makePlanDecision(verifyQuery, sessionId);
      
      if (verifyDecision.plan === 'VERIFY') {
        expect(verifyDecision.alpha).toBe(DEFAULT_ADAPTIVE_CONFIG.weights.VERIFY.alpha);
        expect(verifyDecision.efSearch).toBe(DEFAULT_ADAPTIVE_CONFIG.weights.VERIFY.efSearch);
      }
      
      // Test EXPLORE weights  
      planner.updateThresholds({ tau_n: 0.9 }); // High threshold to force EXPLORE
      const exploreQuery = 'completely new topic unrelated to conversation';
      const exploreDecision = await planner.makePlanDecision(exploreQuery, sessionId);
      
      if (exploreDecision.plan === 'EXPLORE') {
        expect(exploreDecision.alpha).toBe(DEFAULT_ADAPTIVE_CONFIG.weights.EXPLORE.alpha);
        expect(exploreDecision.efSearch).toBe(DEFAULT_ADAPTIVE_CONFIG.weights.EXPLORE.efSearch);
      }
      
      // Restore original config
      planner.updateThresholds(originalConfig.thresholds);
    });
  });

  describe('Plan Logging and Analytics', () => {
    it('should log plan decisions to database', async () => {
      const query = 'test query for logging';
      await planner.makePlanDecision(query, sessionId);
      
      const logStmt = db.prepare('SELECT * FROM plan_decisions WHERE session_id = ? AND query = ?');
      const logEntry = logStmt.get(sessionId, query) as any;
      
      expect(logEntry).toBeDefined();
      expect(logEntry.plan).toBeOneOf(['VERIFY', 'EXPLORE', 'EXPLOIT']);
      expect(logEntry.features).toBeTypeOf('string'); // JSON serialized
      expect(logEntry.reasoning).toBeTypeOf('string');
      expect(logEntry.confidence).toBeGreaterThan(0);
    });

    it('should update results count for logged decisions', async () => {
      const query = 'test query for results update';
      await planner.makePlanDecision(query, sessionId);
      
      planner.updatePlanResults(sessionId, query, 5);
      
      const logStmt = db.prepare('SELECT results_count FROM plan_decisions WHERE session_id = ? AND query = ?');
      const result = logStmt.get(sessionId, query) as { results_count: number };
      
      expect(result.results_count).toBe(5);
    });

    it('should generate meaningful plan statistics', async () => {
      // Create multiple decisions for statistics
      const queries = [
        'high IDF entity overlap query',
        'completely unrelated new topic',
        'balanced query with some context',
        'another balanced query',
      ];
      
      for (const query of queries) {
        await planner.makePlanDecision(query, sessionId);
      }
      
      const stats = planner.getPlanStats(sessionId);
      
      expect(Object.values(stats.planCounts).reduce((a, b) => a + b, 0)).toBe(queries.length);
      expect(stats.avgConfidence).toBeDefined();
      expect(stats.featureDistributions).toBeDefined();
    });
  });

  describe('Threshold Tuning', () => {
    it('should suggest lowering tau_v when VERIFY never triggers', async () => {
      // Set very high tau_v to prevent VERIFY
      planner.updateThresholds({ tau_v: 10.0 });
      
      // Make several decisions that would normally trigger VERIFY
      const queries = [
        'React component state initialization',
        'browser developer tools debugging',
        'UserProfile component error handling',
        'component lifecycle methods testing',
        'state management debugging patterns',
        'event handler optimization strategies',
        'render performance analysis',
        'memory leak detection techniques',
        'component prop validation',
        'async data loading patterns',
        'error boundary implementation',
        'testing framework configuration',
      ];
      
      for (const query of queries) {
        await planner.makePlanDecision(query, sessionId);
      }
      
      const suggestions = planner.suggestThresholdTuning(sessionId);
      
      expect(suggestions.confidence).toBeGreaterThan(0);
      const tauVSuggestion = suggestions.suggestions.find(s => s.parameter === 'tau_v');
      if (tauVSuggestion) {
        expect(tauVSuggestion.suggested).toBeLessThan(tauVSuggestion.current);
        expect(tauVSuggestion.reasoning).toContain('VERIFY');
      }
    });

    it('should suggest adjusting tau_n when EXPLORE dominates', async () => {
      // Set low tau_n to force EXPLORE dominance
      planner.updateThresholds({ tau_n: 0.8 });
      
      // Make many decisions
      for (let i = 0; i < 15; i++) {
        await planner.makePlanDecision(`query number ${i}`, sessionId);
      }
      
      const suggestions = planner.suggestThresholdTuning(sessionId);
      
      if (suggestions.suggestions.length > 0) {
        const tauNSuggestion = suggestions.suggestions.find(s => s.parameter === 'tau_n');
        if (tauNSuggestion) {
          expect(tauNSuggestion.reasoning).toContain('EXPLORE');
        }
      }
    });

    it('should update thresholds correctly', () => {
      const newThresholds = { tau_v: 3.0, tau_e: 0.5 };
      planner.updateThresholds(newThresholds);
      
      const config = planner.getConfig();
      expect(config.thresholds.tau_v).toBe(3.0);
      expect(config.thresholds.tau_e).toBe(0.5);
      expect(config.thresholds.tau_n).toBe(DEFAULT_ADAPTIVE_CONFIG.thresholds.tau_n); // Unchanged
    });
  });

  describe('Configuration and Customization', () => {
    it('should accept custom configuration', () => {
      const customConfig: Partial<AdaptivePlanningConfig> = {
        thresholds: { tau_v: 3.5, tau_e: 0.6, tau_n: 0.15 },
        window_turns: 15,
        min_results: 5,
      };
      
      const customPlanner = createAdaptivePlanner(db, sessionIdf, entityExtractor, customConfig);
      const config = customPlanner.getConfig();
      
      expect(config.thresholds.tau_v).toBe(3.5);
      expect(config.thresholds.tau_e).toBe(0.6);
      expect(config.thresholds.tau_n).toBe(0.15);
      expect(config.window_turns).toBe(15);
      expect(config.min_results).toBe(5);
    });

    it('should merge custom patterns with defaults', () => {
      const customConfig: Partial<AdaptivePlanningConfig> = {
        code_patterns: [/custom_pattern/g],
        known_tools: new Set(['custom-tool']),
      };
      
      const customPlanner = createAdaptivePlanner(db, sessionIdf, entityExtractor, customConfig);
      const config = customPlanner.getConfig();
      
      expect(config.code_patterns).toContain(customConfig.code_patterns![0]);
      expect(config.known_tools.has('custom-tool')).toBe(true);
      expect(config.known_tools.has('git')).toBe(true); // Should still have defaults
    });
  });

  describe('Edge Cases and Error Handling', () => {
    it('should handle empty queries gracefully', async () => {
      const decision = await planner.makePlanDecision('', sessionId);
      
      expect(decision.plan).toBeOneOf(['VERIFY', 'EXPLORE', 'EXPLOIT']);
      expect(decision.features.len_q).toBe(0);
      expect(decision.confidence).toBeGreaterThan(0);
    });

    it('should handle sessions with no history', async () => {
      const newSessionId = 'empty-session';
      const decision = await planner.makePlanDecision('test query', newSessionId);
      
      expect(decision.features.entity_overlap).toBe(0.0); // No recent entities means no overlap
      expect(decision.features.recent_tool_count).toBe(0);
      expect(decision.features.turn_position).toBe(0);
    });

    it('should handle very long queries', async () => {
      const longQuery = 'word '.repeat(100) + 'React component debugging';
      const decision = await planner.makePlanDecision(longQuery, sessionId);
      
      expect(decision.features.len_q).toBeGreaterThan(100);
      expect(decision.plan).toBeDefined();
    });
  });
});