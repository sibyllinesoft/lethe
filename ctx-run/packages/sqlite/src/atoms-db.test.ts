/**
 * @fileoverview Tests for Lethe agent context atoms system
 * Milestone 1: Validate data model implementation
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import { randomUUID } from 'crypto';

import { AtomsDatabase, createAtomsDatabase } from './atoms-db.js';
import { Atom, AtomRole, AtomType, EntityKind } from './atoms-types.js';

describe('AtomsDatabase', () => {
  let db: Database.Database;
  let atomsDb: AtomsDatabase;
  let testSessionId: string;

  beforeEach(async () => {
    db = new Database(':memory:');
    atomsDb = createAtomsDatabase(db, {
      enableFts: true,
      enableVectors: true,
      enableEntities: true,
      embedding: {
        modelName: 'test-model',
        dimension: 384,
        maxTokens: 512,
        batchSize: 32,
        cpuOnly: true,
      },
    });
    
    await atomsDb.initialize();
    testSessionId = randomUUID();
  });

  afterEach(() => {
    db.close();
  });

  describe('Schema and Initialization', () => {
    it('should create all required tables', () => {
      const tables = db
        .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        .all() as { name: string }[];
      
      const tableNames = tables.map(t => t.name);
      
      expect(tableNames).toContain('atoms');
      expect(tableNames).toContain('entities');
      expect(tableNames).toContain('vectors');
      expect(tableNames).toContain('session_idf');
      expect(tableNames).toContain('fts_atoms');
    });

    it('should create all required indexes', () => {
      const indexes = db
        .prepare("SELECT name FROM sqlite_master WHERE type='index' ORDER BY name")
        .all() as { name: string }[];
      
      const indexNames = indexes.map(i => i.name).filter(n => !n.startsWith('sqlite_'));
      
      expect(indexNames).toContain('idx_atoms_session_turn');
      expect(indexNames).toContain('idx_entities_kind');
      expect(indexNames).toContain('idx_vectors_dim');
      expect(indexNames).toContain('idx_session_idf_session');
    });

    it('should create views', () => {
      const views = db
        .prepare("SELECT name FROM sqlite_master WHERE type='view' ORDER BY name")
        .all() as { name: string }[];
      
      const viewNames = views.map(v => v.name);
      
      expect(viewNames).toContain('atoms_with_entities');
      expect(viewNames).toContain('session_stats');
    });
  });

  describe('Atom Insertion', () => {
    it('should insert a single atom', async () => {
      const atom: Atom = {
        id: randomUUID(),
        session_id: testSessionId,
        turn_idx: 1,
        role: 'user' as AtomRole,
        type: 'message' as AtomType,
        text: 'Hello, I need help with debugging my TypeScript code.',
        json_meta: { context: 'coding_help' },
        ts: Date.now(),
      };

      await atomsDb.insertAtom(atom);

      // Verify atom was inserted
      const inserted = db
        .prepare('SELECT * FROM atoms WHERE id = ?')
        .get(atom.id) as any;
      
      expect(inserted).toBeDefined();
      expect(inserted.session_id).toBe(testSessionId);
      expect(inserted.text).toBe(atom.text);
    });

    it('should insert multiple atoms in batch', async () => {
      const atoms: Atom[] = [
        {
          id: randomUUID(),
          session_id: testSessionId,
          turn_idx: 1,
          role: 'user',
          type: 'message',
          text: 'I need to fix a bug in my React component.',
          ts: Date.now(),
        },
        {
          id: randomUUID(),
          session_id: testSessionId,
          turn_idx: 2,
          role: 'assistant',
          type: 'message',
          text: 'I can help you debug your React component. Please share the code.',
          ts: Date.now(),
        },
        {
          id: randomUUID(),
          session_id: testSessionId,
          turn_idx: 3,
          role: 'tool',
          type: 'action',
          text: 'Reading file: src/components/UserProfile.tsx',
          json_meta: { tool_name: 'file_reader', file_path: 'src/components/UserProfile.tsx' },
          ts: Date.now(),
        },
      ];

      await atomsDb.insertAtoms(atoms);

      // Verify all atoms were inserted
      const count = db
        .prepare('SELECT COUNT(*) as count FROM atoms WHERE session_id = ?')
        .get(testSessionId) as { count: number };
      
      expect(count.count).toBe(3);
    });
  });

  describe('Entity Extraction', () => {
    it('should extract entities from atom text', async () => {
      const atom: Atom = {
        id: randomUUID(),
        session_id: testSessionId,
        turn_idx: 1,
        role: 'user',
        type: 'message',
        text: 'I have an ERROR_CODE_123 in my UserService.js file when calling the /api/users endpoint',
        ts: Date.now(),
      };

      await atomsDb.insertAtom(atom);

      // Check that entities were extracted
      const entities = db
        .prepare('SELECT * FROM entities WHERE atom_id = ?')
        .all(atom.id) as any[];

      expect(entities.length).toBeGreaterThan(0);
      
      // Should extract various entity types
      const entityKinds = new Set(entities.map(e => e.kind));
      expect(entityKinds.size).toBeGreaterThan(1); // Multiple kinds
      
      // Check for specific patterns
      const entityTexts = entities.map(e => e.entity.toLowerCase());
      expect(entityTexts.some(text => text.includes('error'))).toBeTruthy();
      expect(entityTexts.some(text => text.includes('userservice'))).toBeTruthy();
      expect(entityTexts.some(text => text.includes('api'))).toBeTruthy();
    });

    it('should assign weights to entities', async () => {
      const atom: Atom = {
        id: randomUUID(),
        session_id: testSessionId,
        turn_idx: 1,
        role: 'user',
        type: 'message',
        text: 'Fix the calculateTotal function in PaymentProcessor.ts',
        ts: Date.now(),
      };

      await atomsDb.insertAtom(atom);

      const entities = db
        .prepare('SELECT * FROM entities WHERE atom_id = ? ORDER BY weight DESC')
        .all(atom.id) as any[];

      expect(entities.length).toBeGreaterThan(0);
      
      // All entities should have positive weights
      for (const entity of entities) {
        expect(entity.weight).toBeGreaterThan(0);
      }
      
      // Weights should be in descending order
      for (let i = 1; i < entities.length; i++) {
        expect(entities[i-1].weight).toBeGreaterThanOrEqual(entities[i].weight);
      }
    });
  });

  describe('Session IDF', () => {
    it('should compute session IDF for terms', async () => {
      const atoms: Atom[] = [
        {
          id: randomUUID(),
          session_id: testSessionId,
          turn_idx: 1,
          role: 'user',
          type: 'message',
          text: 'I need help with JavaScript debugging',
          ts: Date.now(),
        },
        {
          id: randomUUID(),
          session_id: testSessionId,
          turn_idx: 2,
          role: 'assistant',
          type: 'message',
          text: 'JavaScript debugging can be tricky. Let me help you.',
          ts: Date.now(),
        },
      ];

      await atomsDb.insertAtoms(atoms);

      // Check session IDF was computed
      const idfEntries = db
        .prepare('SELECT * FROM session_idf WHERE session_id = ?')
        .all(testSessionId) as any[];

      expect(idfEntries.length).toBeGreaterThan(0);
      
      // Should have IDF values for common terms
      const terms = idfEntries.map(entry => entry.term);
      expect(terms).toContain('javascript');
      expect(terms).toContain('debugging');
      
      // IDF values should be computed
      for (const entry of idfEntries) {
        expect(entry.idf).toBeTypeOf('number');
        expect(entry.df).toBeGreaterThan(0);
      }
    });

    it('should handle incremental IDF updates', async () => {
      // Insert first atom
      const atom1: Atom = {
        id: randomUUID(),
        session_id: testSessionId,
        turn_idx: 1,
        role: 'user',
        type: 'message',
        text: 'TypeScript compilation error',
        ts: Date.now(),
      };

      await atomsDb.insertAtom(atom1);

      const initialCount = db
        .prepare('SELECT COUNT(*) as count FROM session_idf WHERE session_id = ?')
        .get(testSessionId) as { count: number };

      // Insert second atom with overlapping terms
      const atom2: Atom = {
        id: randomUUID(),
        session_id: testSessionId,
        turn_idx: 2,
        role: 'assistant',
        type: 'message',
        text: 'TypeScript error handling is important',
        ts: Date.now(),
      };

      await atomsDb.insertAtom(atom2);

      const finalCount = db
        .prepare('SELECT COUNT(*) as count FROM session_idf WHERE session_id = ?')
        .get(testSessionId) as { count: number };

      // Should have updated IDF for existing terms and added new ones
      expect(finalCount.count).toBeGreaterThanOrEqual(initialCount.count);
    });
  });

  describe('FTS5 Integration', () => {
    it('should maintain FTS index automatically', async () => {
      const atom: Atom = {
        id: randomUUID(),
        session_id: testSessionId,
        turn_idx: 1,
        role: 'user',
        type: 'message',
        text: 'How to implement binary search algorithm in Python?',
        ts: Date.now(),
      };

      await atomsDb.insertAtom(atom);

      // Check FTS index was populated
      const ftsEntries = db
        .prepare('SELECT * FROM fts_atoms WHERE atom_id = ?')
        .all(atom.id) as any[];

      expect(ftsEntries.length).toBe(1);
      expect(ftsEntries[0].text).toBe(atom.text);
    });

    it('should perform FTS search', async () => {
      const atoms: Atom[] = [
        {
          id: randomUUID(),
          session_id: testSessionId,
          turn_idx: 1,
          role: 'user',
          type: 'message',
          text: 'Binary search implementation in Python',
          ts: Date.now(),
        },
        {
          id: randomUUID(),
          session_id: testSessionId,
          turn_idx: 2,
          role: 'assistant',
          type: 'message',
          text: 'Here is a Python binary search function',
          ts: Date.now(),
        },
      ];

      await atomsDb.insertAtoms(atoms);

      const results = atomsDb.searchFts('binary search');
      expect(results.length).toBeGreaterThan(0);
      
      const atomIds = results.map(r => r.atom_id);
      expect(atomIds).toContain(atoms[0].id);
    });
  });

  describe('Vector Search', () => {
    it('should store vectors for atoms', async () => {
      const atom: Atom = {
        id: randomUUID(),
        session_id: testSessionId,
        turn_idx: 1,
        role: 'user',
        type: 'message',
        text: 'Machine learning model training with PyTorch',
        ts: Date.now(),
      };

      await atomsDb.insertAtom(atom);

      // Check vector was stored
      const vector = db
        .prepare('SELECT * FROM vectors WHERE atom_id = ?')
        .get(atom.id) as any;

      expect(vector).toBeDefined();
      expect(vector.dim).toBeGreaterThan(0);
      expect(vector.blob).toBeInstanceOf(Buffer);
    });

    it('should perform vector search', async () => {
      const atoms: Atom[] = [
        {
          id: randomUUID(),
          session_id: testSessionId,
          turn_idx: 1,
          role: 'user',
          type: 'message',
          text: 'Deep learning neural networks',
          ts: Date.now(),
        },
        {
          id: randomUUID(),
          session_id: testSessionId,
          turn_idx: 2,
          role: 'assistant',
          type: 'message',
          text: 'Neural network architecture design',
          ts: Date.now(),
        },
      ];

      await atomsDb.insertAtoms(atoms);

      const results = await atomsDb.searchVectors('machine learning neural networks', 5);
      expect(results.length).toBeGreaterThan(0);
      
      // Results should have similarity scores
      for (const result of results) {
        expect(result.similarity).toBeTypeOf('number');
        expect(result.similarity).toBeGreaterThanOrEqual(0);
        expect(result.similarity).toBeLessThanOrEqual(1);
      }
    });
  });

  describe('Hybrid Search', () => {
    it('should perform hybrid search combining FTS and vectors', async () => {
      const atoms: Atom[] = [
        {
          id: randomUUID(),
          session_id: testSessionId,
          turn_idx: 1,
          role: 'user',
          type: 'message',
          text: 'I need help debugging a React component state issue',
          ts: Date.now(),
        },
        {
          id: randomUUID(),
          session_id: testSessionId,
          turn_idx: 2,
          role: 'assistant',
          type: 'message',
          text: 'React state debugging requires examining component lifecycle',
          ts: Date.now(),
        },
        {
          id: randomUUID(),
          session_id: testSessionId,
          turn_idx: 3,
          role: 'tool',
          type: 'observation',
          text: 'Component StateManager.tsx shows useState hook issues',
          ts: Date.now(),
        },
      ];

      await atomsDb.insertAtoms(atoms);

      const results = await atomsDb.hybridSearch(
        'React component debugging',
        { sessionId: testSessionId },
        undefined,
        10
      );

      expect(results.atoms.length).toBeGreaterThan(0);
      expect(results.metadata.totalCandidates).toBeGreaterThan(0);
      expect(results.metadata.searchTime).toBeGreaterThan(0);
      expect(results.metadata.method).toBe('hybrid');

      // Results should have score components
      for (const result of results.atoms) {
        expect(result.scoreComponents.final).toBeTypeOf('number');
        expect(result.entities).toBeInstanceOf(Array);
      }
    });
  });

  describe('Database Statistics and Utilities', () => {
    it('should provide database statistics', async () => {
      const atoms: Atom[] = [
        {
          id: randomUUID(),
          session_id: testSessionId,
          turn_idx: 1,
          role: 'user',
          type: 'message',
          text: 'Test message 1',
          ts: Date.now(),
        },
        {
          id: randomUUID(),
          session_id: testSessionId,
          turn_idx: 2,
          role: 'assistant',
          type: 'message',
          text: 'Test message 2',
          ts: Date.now(),
        },
      ];

      await atomsDb.insertAtoms(atoms);

      const stats = atomsDb.getStats();
      
      expect(stats.atomCount).toBe(2);
      expect(stats.sessionCount).toBe(1);
      expect(stats.entityCount).toBeGreaterThan(0);
      expect(stats.vectorCount).toBe(2);
      expect(stats.indexStats).toBeDefined();
    });

    it('should get session statistics', async () => {
      const atoms: Atom[] = [
        {
          id: randomUUID(),
          session_id: testSessionId,
          turn_idx: 1,
          role: 'user',
          type: 'message',
          text: 'First message',
          ts: Date.now() - 1000,
        },
        {
          id: randomUUID(),
          session_id: testSessionId,
          turn_idx: 2,
          role: 'assistant',
          type: 'message',
          text: 'Second message',
          ts: Date.now(),
        },
      ];

      await atomsDb.insertAtoms(atoms);

      const sessionStats = atomsDb.getSessionStats(testSessionId);
      
      expect(sessionStats).toBeDefined();
      expect(sessionStats!.total_atoms).toBe(2);
      expect(sessionStats!.session_id).toBe(testSessionId);
      expect(sessionStats!.max_turns).toBe(2);
    });

    it('should delete session data', async () => {
      const atom: Atom = {
        id: randomUUID(),
        session_id: testSessionId,
        turn_idx: 1,
        role: 'user',
        type: 'message',
        text: 'Test message',
        ts: Date.now(),
      };

      await atomsDb.insertAtom(atom);

      // Verify insertion
      const beforeCount = db
        .prepare('SELECT COUNT(*) as count FROM atoms WHERE session_id = ?')
        .get(testSessionId) as { count: number };
      expect(beforeCount.count).toBe(1);

      // Delete session
      const deletedCount = atomsDb.deleteSession(testSessionId);
      expect(deletedCount).toBe(1);

      // Verify deletion
      const afterCount = db
        .prepare('SELECT COUNT(*) as count FROM atoms WHERE session_id = ?')
        .get(testSessionId) as { count: number };
      expect(afterCount.count).toBe(0);
    });
  });

  describe('Acceptance Criteria Validation', () => {
    it('should satisfy: Inserting atoms populates FTS5, vectors, and entities consistently', async () => {
      const atom: Atom = {
        id: randomUUID(),
        session_id: testSessionId,
        turn_idx: 1,
        role: 'user',
        type: 'message',
        text: 'Debug the calculateTax function in PaymentProcessor.js with API_KEY_ERROR',
        ts: Date.now(),
      };

      await atomsDb.insertAtom(atom);

      // Check atoms table
      const atomExists = db.prepare('SELECT 1 FROM atoms WHERE id = ?').get(atom.id);
      expect(atomExists).toBeDefined();

      // Check FTS5 populated
      const ftsExists = db.prepare('SELECT 1 FROM fts_atoms WHERE atom_id = ?').get(atom.id);
      expect(ftsExists).toBeDefined();

      // Check vectors populated
      const vectorExists = db.prepare('SELECT 1 FROM vectors WHERE atom_id = ?').get(atom.id);
      expect(vectorExists).toBeDefined();

      // Check entities populated
      const entitiesExist = db.prepare('SELECT COUNT(*) as count FROM entities WHERE atom_id = ?').get(atom.id) as { count: number };
      expect(entitiesExist.count).toBeGreaterThan(0);
    });

    it('should satisfy: Session-idf view returns nonzero values for session terms', async () => {
      const atoms: Atom[] = [
        {
          id: randomUUID(),
          session_id: testSessionId,
          turn_idx: 1,
          role: 'user',
          type: 'message',
          text: 'JavaScript function debugging',
          ts: Date.now(),
        },
        {
          id: randomUUID(),
          session_id: testSessionId,
          turn_idx: 2,
          role: 'assistant',
          type: 'message',
          text: 'Function debugging requires careful analysis',
          ts: Date.now(),
        },
      ];

      await atomsDb.insertAtoms(atoms);

      // Check that session IDF values are computed and nonzero
      const idfValues = db
        .prepare('SELECT term, idf FROM session_idf WHERE session_id = ? AND idf > 0')
        .all(testSessionId) as Array<{ term: string; idf: number }>;

      expect(idfValues.length).toBeGreaterThan(0);
      
      // All IDF values should be positive
      for (const entry of idfValues) {
        expect(entry.idf).toBeGreaterThan(0);
      }
    });

    it('should satisfy: ANN search returns neighbors; FTS5 returns lexical matches', async () => {
      const atoms: Atom[] = [
        {
          id: randomUUID(),
          session_id: testSessionId,
          turn_idx: 1,
          role: 'user',
          type: 'message',
          text: 'Machine learning model training with neural networks',
          ts: Date.now(),
        },
        {
          id: randomUUID(),
          session_id: testSessionId,
          turn_idx: 2,
          role: 'assistant',
          type: 'message',
          text: 'Deep learning requires extensive dataset preparation',
          ts: Date.now(),
        },
      ];

      await atomsDb.insertAtoms(atoms);

      // Test FTS5 lexical matching
      const ftsResults = atomsDb.searchFts('machine learning');
      expect(ftsResults.length).toBeGreaterThan(0);
      expect(ftsResults[0].atom_id).toBe(atoms[0].id); // Should match exact terms

      // Test ANN semantic similarity
      const vectorResults = await atomsDb.searchVectors('artificial intelligence neural networks');
      expect(vectorResults.length).toBeGreaterThan(0);
      
      // Vector search should find semantic similarity even without exact term matches
      const vectorResultIds = vectorResults.map(r => r.atom_id);
      expect(vectorResultIds).toContain(atoms[0].id); // Should find semantically similar content
    });
  });
});

describe('Integration Tests', () => {
  it('should work end-to-end with a realistic agent conversation', async () => {
    const db = new Database(':memory:');
    const atomsDb = createAtomsDatabase(db);
    await atomsDb.initialize();

    const sessionId = randomUUID();
    
    // Simulate a realistic agent conversation
    const conversation: Atom[] = [
      {
        id: randomUUID(),
        session_id: sessionId,
        turn_idx: 1,
        role: 'user',
        type: 'message',
        text: 'I have a bug in my React component. The useState hook is not updating properly.',
        ts: Date.now() - 5000,
      },
      {
        id: randomUUID(),
        session_id: sessionId,
        turn_idx: 2,
        role: 'assistant',
        type: 'message',
        text: 'Let me help you debug the React useState issue. Can you show me the component code?',
        ts: Date.now() - 4000,
      },
      {
        id: randomUUID(),
        session_id: sessionId,
        turn_idx: 3,
        role: 'tool',
        type: 'action',
        text: 'file_reader',
        json_meta: { tool: 'file_reader', path: 'src/components/UserProfile.tsx' },
        ts: Date.now() - 3000,
      },
      {
        id: randomUUID(),
        session_id: sessionId,
        turn_idx: 3,
        role: 'tool',
        type: 'observation',
        text: 'Found React component with useState hook:\nconst [user, setUser] = useState(null);',
        json_meta: { file_path: 'src/components/UserProfile.tsx' },
        ts: Date.now() - 2000,
      },
      {
        id: randomUUID(),
        session_id: sessionId,
        turn_idx: 4,
        role: 'assistant',
        type: 'plan',
        text: 'I need to analyze the useState implementation and check for common pitfalls',
        ts: Date.now() - 1000,
      },
    ];

    await atomsDb.insertAtoms(conversation);

    // Test various search scenarios
    const reactSearchResults = await atomsDb.hybridSearch(
      'React useState component',
      { sessionId },
      undefined,
      5
    );

    expect(reactSearchResults.atoms.length).toBeGreaterThan(0);
    expect(reactSearchResults.metadata.method).toBe('hybrid');

    // Test entity-based search
    const fileSearchResults = await atomsDb.hybridSearch(
      'UserProfile.tsx',
      { sessionId, includeEntities: ['file'] },
      undefined,
      3
    );

    expect(fileSearchResults.atoms.length).toBeGreaterThan(0);

    // Verify session statistics
    const stats = atomsDb.getSessionStats(sessionId);
    expect(stats).toBeDefined();
    expect(stats!.total_atoms).toBe(5);
    expect(stats!.distinct_roles).toBeGreaterThan(1);

    db.close();
  });
});