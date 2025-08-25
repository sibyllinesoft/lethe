import { describe, it, expect, vi, beforeEach } from 'vitest';
import { StateManager, getStateManager, type SessionState, type PlanSelection } from './index.js';

describe('State Management Module', () => {
  let mockDB: any;
  let mockPrepare: any;
  let mockRun: any;
  let mockGet: any;
  let mockAll: any;

  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.spyOn(console, 'log').mockImplementation(() => {});

    mockRun = vi.fn();
    mockGet = vi.fn();
    mockAll = vi.fn();
    
    mockPrepare = vi.fn(() => ({
      run: mockRun,
      get: mockGet,
      all: mockAll
    }));

    mockDB = {
      exec: vi.fn(),
      prepare: mockPrepare
    };
  });

  describe('StateManager', () => {
    it('should initialize with database and create table', () => {
      const stateManager = new StateManager(mockDB);
      
      expect(stateManager).toBeInstanceOf(StateManager);
      expect(mockDB.exec).toHaveBeenCalledWith(expect.stringContaining('CREATE TABLE IF NOT EXISTS session_states'));
    });

    it('should handle table creation errors gracefully', () => {
      mockDB.exec.mockImplementationOnce(() => {
        throw new Error('Table creation failed');
      });

      const stateManager = new StateManager(mockDB);
      
      expect(stateManager).toBeInstanceOf(StateManager);
      expect(console.warn).toHaveBeenCalledWith(expect.stringContaining('Could not create state table'));
    });

    it('should get session state for existing session', () => {
      const mockSessionState = {
        session_id: 'test-session',
        recent_entities: '["entity1", "entity2"]',
        last_pack_claims: '["claim1"]',
        last_pack_contradictions: '["contradiction1"]',
        last_pack_id: 'pack-123',
        updated_at: '2024-01-01T00:00:00.000Z'
      };

      mockGet.mockReturnValueOnce(mockSessionState);

      const stateManager = new StateManager(mockDB);
      const result = stateManager.getSessionState('test-session');

      expect(result).toEqual({
        sessionId: 'test-session',
        recentEntities: ['entity1', 'entity2'],
        lastPackClaims: ['claim1'],
        lastPackContradictions: ['contradiction1'],
        lastPackId: 'pack-123',
        updatedAt: '2024-01-01T00:00:00.000Z'
      });
    });

    it('should return empty state for non-existing session', () => {
      mockGet.mockReturnValueOnce(null);

      const stateManager = new StateManager(mockDB);
      const result = stateManager.getSessionState('non-existent-session');

      expect(result).toEqual({
        sessionId: 'non-existent-session',
        recentEntities: [],
        lastPackClaims: [],
        lastPackContradictions: [],
        lastPackId: undefined,
        updatedAt: expect.any(String)
      });
    });

    it('should handle malformed JSON in session state gracefully', () => {
      const mockSessionState = {
        session_id: 'test-session',
        recent_entities: 'invalid-json',
        last_pack_claims: 'invalid-json-too',
        last_pack_contradictions: 'also-invalid-json',
        last_pack_id: 'pack-123',
        updated_at: '2024-01-01T00:00:00.000Z'
      };

      mockGet.mockReturnValueOnce(mockSessionState);

      const stateManager = new StateManager(mockDB);
      const result = stateManager.getSessionState('test-session');

      // All malformed JSON should result in empty arrays
      expect(result.recentEntities).toEqual([]);
      expect(result.lastPackClaims).toEqual([]);
      expect(result.lastPackContradictions).toEqual([]);
    });

    it('should update session state with new context pack', () => {
      const mockContextPack = {
        id: 'new-pack-456',
        session_id: 'test-session',
        key_entities: ['entity3', 'entity4'],
        claims: ['new claim'],
        contradictions: ['new contradiction'],
        created_at: '2024-01-01T00:00:00.000Z',
        query: 'test query',
        summary: 'test summary',
        chunks: [],
        citations: []
      };

      const stateManager = new StateManager(mockDB);
      stateManager.updateSessionState('test-session', mockContextPack);

      expect(mockPrepare).toHaveBeenCalledWith(expect.stringContaining('INSERT OR REPLACE'));
      expect(mockRun).toHaveBeenCalledWith(
        'test-session',
        expect.stringContaining('entity3'),
        expect.stringContaining('new claim'),
        expect.stringContaining('new contradiction'),
        'new-pack-456',
        expect.any(String)
      );
    });

    it('should handle entity limit when updating session state', () => {
      // Create mock context pack with many entities
      const manyEntities = Array.from({ length: 250 }, (_, i) => `entity${i}`);
      const mockContextPack = {
        id: 'pack-789',
        session_id: 'test-session',
        key_entities: manyEntities,
        claims: ['claim'],
        contradictions: [],
        created_at: '2024-01-01T00:00:00.000Z',
        query: 'test query',
        summary: 'test summary',
        chunks: [],
        citations: []
      };

      const stateManager = new StateManager(mockDB);
      stateManager.updateSessionState('test-session', mockContextPack);

      // Should limit to 200 entities
      expect(mockRun).toHaveBeenCalled();
      const entitiesArg = mockRun.mock.calls[0][1];
      const entities = JSON.parse(entitiesArg);
      expect(entities).toHaveLength(200);
    });

    it('should select explore plan for sessions with few entities', () => {
      mockGet.mockReturnValueOnce({
        session_id: 'test-session',
        recent_entities: '["entity1"]',
        last_pack_claims: '[]',
        last_pack_contradictions: '[]',
        last_pack_id: null,
        updated_at: '2024-01-01T00:00:00.000Z'
      });

      const stateManager = new StateManager(mockDB);
      const result = stateManager.selectPlan('test-session', 'what is machine learning?');

      expect(result.plan).toBe('explore');
      expect(result.reasoning).toContain('new topic exploration');
      expect(result.parameters).toHaveProperty('hyde_k');
      expect(result.parameters).toHaveProperty('beta');
    });

    it('should select verify plan when contradictions exist', () => {
      mockGet.mockReturnValueOnce({
        session_id: 'test-session',
        recent_entities: '["entity1", "entity2", "entity3", "entity4", "entity5"]',
        last_pack_claims: '["claim1"]',
        last_pack_contradictions: '["contradiction1", "contradiction2"]',
        last_pack_id: 'pack-123',
        updated_at: '2024-01-01T00:00:00.000Z'
      });

      const stateManager = new StateManager(mockDB);
      const result = stateManager.selectPlan('test-session', 'verify this information');

      expect(result.plan).toBe('verify');
      expect(result.reasoning).toContain('contradictions');
      expect(result.parameters.granularity).toBe('tight');
    });

    it('should select exploit plan for established sessions', () => {
      mockGet.mockReturnValueOnce({
        session_id: 'test-session',
        recent_entities: '["machine", "learning", "details", "more"]', // Entities that overlap with query
        last_pack_claims: '["claim1", "claim2"]',
        last_pack_contradictions: '[]',
        last_pack_id: 'pack-123',
        updated_at: '2024-01-01T00:00:00.000Z'
      });

      const stateManager = new StateManager(mockDB);
      const result = stateManager.selectPlan('test-session', 'give me more machine learning details');

      expect(result.plan).toBe('exploit');
      expect(result.reasoning).toContain('High entity overlap');
      expect(result.parameters.granularity).toBe('medium');
    });

    it('should handle plan selection with specific query patterns', () => {
      mockGet.mockReturnValueOnce({
        session_id: 'test-session',
        recent_entities: '["entity1"]',
        last_pack_claims: '[]',
        last_pack_contradictions: '[]',
        last_pack_id: null,
        updated_at: '2024-01-01T00:00:00.000Z'
      });

      const stateManager = new StateManager(mockDB);
      
      // Test different query patterns
      const howQuery = stateManager.selectPlan('test-session', 'how does this work?');
      expect(howQuery.plan).toBe('explore');

      const whatQuery = stateManager.selectPlan('test-session', 'what is the definition?');
      expect(whatQuery.plan).toBe('explore');
    });

    it('should get all sessions', () => {
      mockAll.mockReturnValueOnce([
        { session_id: 'session1' },
        { session_id: 'session2' },
        { session_id: 'session3' }
      ]);

      const stateManager = new StateManager(mockDB);
      const sessions = stateManager.getAllSessions();

      expect(sessions).toEqual(['session1', 'session2', 'session3']);
    });

    it('should get recent context for session', () => {
      mockGet.mockReturnValueOnce({
        session_id: 'test-session',
        recent_entities: '["entity1", "entity2"]',
        last_pack_claims: '["claim1"]',
        last_pack_contradictions: '["contradiction1"]',
        last_pack_id: 'pack-123',
        updated_at: '2024-01-01T00:00:00.000Z'
      });

      const stateManager = new StateManager(mockDB);
      const context = stateManager.getRecentContext('test-session');

      expect(context).toEqual({
        entityCount: 2,
        recentEntities: ['entity1', 'entity2'],
        lastPackContradictions: ['contradiction1']
      });
    });

    it('should handle database errors during state operations', () => {
      mockGet.mockImplementationOnce(() => {
        throw new Error('Database error');
      });

      const stateManager = new StateManager(mockDB);
      const result = stateManager.getSessionState('test-session');

      // Should return default state on error
      expect(result.sessionId).toBe('test-session');
      expect(result.recentEntities).toEqual([]);
    });

    it('should handle database errors during state updates', () => {
      mockRun.mockImplementationOnce(() => {
        throw new Error('Update failed');
      });

      const mockContextPack = {
        id: 'pack-error',
        session_id: 'test-session',
        key_entities: ['entity1'],
        claims: ['claim1'],
        contradictions: [],
        created_at: '2024-01-01T00:00:00.000Z',
        query: 'test query',
        summary: 'test summary',
        chunks: [],
        citations: []
      };

      const stateManager = new StateManager(mockDB);
      
      // Should not throw error
      expect(() => {
        stateManager.updateSessionState('test-session', mockContextPack);
      }).not.toThrow();

      expect(console.warn).toHaveBeenCalledWith(expect.stringContaining('update session state'));
    });

    it('should handle empty session list gracefully', () => {
      mockAll.mockReturnValueOnce([]);

      const stateManager = new StateManager(mockDB);
      const sessions = stateManager.getAllSessions();

      expect(sessions).toEqual([]);
    });
  });

  describe('getStateManager factory function', () => {
    it('should create StateManager instance', () => {
      const stateManager = getStateManager(mockDB);
      
      expect(stateManager).toBeInstanceOf(StateManager);
    });

    it('should create new instance each time', () => {
      const stateManager1 = getStateManager(mockDB);
      const stateManager2 = getStateManager(mockDB);
      
      // Each call creates a new instance
      expect(stateManager1).not.toBe(stateManager2);
      expect(stateManager1).toBeInstanceOf(StateManager);
      expect(stateManager2).toBeInstanceOf(StateManager);
    });
  });

  describe('Edge Cases and Validation', () => {
    it('should handle null context pack gracefully', () => {
      const stateManager = new StateManager(mockDB);
      
      expect(() => {
        stateManager.updateSessionState('test-session', null as any);
      }).not.toThrow();
    });

    it('should handle empty session ID', () => {
      const stateManager = new StateManager(mockDB);
      const result = stateManager.getSessionState('');
      
      expect(result.sessionId).toBe('');
    });

    it('should handle very long entity names', () => {
      const longEntity = 'entity_'.repeat(100);
      const mockContextPack = {
        id: 'pack-long',
        session_id: 'test-session',
        key_entities: [longEntity],
        claims: ['claim1'],
        contradictions: [],
        created_at: '2024-01-01T00:00:00.000Z',
        query: 'test query',
        summary: 'test summary',
        chunks: [],
        citations: []
      };

      const stateManager = new StateManager(mockDB);
      stateManager.updateSessionState('test-session', mockContextPack);

      expect(mockRun).toHaveBeenCalled();
    });

    it('should handle special characters in session data', () => {
      const mockContextPack = {
        id: 'pack-special',
        session_id: 'test-session',
        key_entities: ['entity with "quotes" and \'apostrophes\''],
        claims: ['claim with\nnewlines and\ttabs'],
        contradictions: ['contradiction with 🚀 emojis'],
        created_at: '2024-01-01T00:00:00.000Z',
        query: 'test query',
        summary: 'test summary',
        chunks: [],
        citations: []
      };

      const stateManager = new StateManager(mockDB);
      
      expect(() => {
        stateManager.updateSessionState('test-session', mockContextPack);
      }).not.toThrow();
    });
  });
});