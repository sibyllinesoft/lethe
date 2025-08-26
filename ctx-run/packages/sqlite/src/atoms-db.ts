/**
 * @fileoverview Main database interface for Lethe agent context atoms
 * Coordinates atoms storage, entity extraction, session-IDF, and embedding indexing
 */

import type Database from 'better-sqlite3';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';

type DB = Database.Database;

import {
  Atom,
  Entity,
  Vector,
  AtomBatch,
  SearchResults,
  SearchContext,
  HybridSearchConfig,
  AtomDbConfig,
  FtsResult,
  VectorSearchResult,
  SessionStats,
} from './atoms-types.js';

import { EntityExtractor, createEntityExtractor } from './entity-extraction.js';
import { SessionIdfCalculator, createSessionIdfCalculator } from './session-idf.js';
import { EmbeddingManager, createEmbeddingManager } from './embeddings.js';

// Use require.resolve to get current file directory
const getCurrentDir = () => {
  try {
    return dirname(require.resolve('./atoms-db'));
  } catch {
    return __dirname;
  }
};

/**
 * Default hybrid search configuration
 */
export const DEFAULT_HYBRID_SEARCH_CONFIG: HybridSearchConfig = {
  ftsWeight: 0.7,
  vectorWeight: 0.3,
  entityWeight: 0.1,
  maxCandidates: 100,
  efSearch: 50,
  diversify: true,
  diversityLambda: 0.7,
};

/**
 * Main atoms database class
 */
export class AtomsDatabase {
  private db: DB;
  private config: AtomDbConfig;
  private entityExtractor: EntityExtractor;
  private sessionIdfCalculator: SessionIdfCalculator;
  private embeddingManager: EmbeddingManager;
  private initialized = false;

  constructor(db: DB, config: AtomDbConfig) {
    this.db = db;
    this.config = config;
    
    // Initialize components
    this.entityExtractor = createEntityExtractor(config.entityExtraction);
    this.sessionIdfCalculator = createSessionIdfCalculator(db, config.sessionIdf);
    this.embeddingManager = createEmbeddingManager(db, config.embedding, config.hnsw);
  }

  /**
   * Initialize database schema and indexes
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    // Apply schema
    const schemaPath = join(getCurrentDir(), '..', 'schema-atoms.sql');
    const schema = readFileSync(schemaPath, 'utf-8');
    this.db.exec(schema);

    // Apply pragmas if configured
    if (this.config.pragmas) {
      for (const [key, value] of Object.entries(this.config.pragmas)) {
        this.db.exec(`PRAGMA ${key} = ${value}`);
      }
    }

    // Build embedding index from existing data
    if (this.config.enableVectors) {
      await this.embeddingManager.buildIndex();
    }

    this.initialized = true;
    console.log('AtomsDatabase initialized');
  }

  /**
   * Insert a single atom with all associated data
   */
  async insertAtom(atom: Atom): Promise<void> {
    await this.insertAtoms([atom]);
  }

  /**
   * Insert multiple atoms as a batch
   */
  async insertAtoms(atoms: Atom[]): Promise<void> {
    if (atoms.length === 0) return;

    // 1. Insert atoms in transaction
    const atomStmt = this.db.prepare(`
      INSERT OR REPLACE INTO atoms 
      (id, session_id, turn_idx, role, type, text, json_meta, ts)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);

    const transaction = this.db.transaction((atomsToInsert: Atom[]) => {
      for (const atom of atomsToInsert) {
        atomStmt.run(
          atom.id,
          atom.session_id,
          atom.turn_idx,
          atom.role,
          atom.type,
          atom.text,
          atom.json_meta ? JSON.stringify(atom.json_meta) : null,
          atom.ts
        );
      }
    });

    transaction(atoms);

    // 2. Extract and insert entities (outside transaction)
    if (this.config.enableEntities) {
      await this.processEntities(atoms);
    }

    // 3. Update session-IDF (outside transaction)
    const sessionGroups = this.groupAtomsBySession(atoms);
    for (const [sessionId, sessionAtoms] of sessionGroups) {
      const texts = sessionAtoms.map(a => a.text);
      if (this.config.sessionIdf.incremental) {
        await this.sessionIdfCalculator.batchUpdateSessionIdf(sessionId, texts);
      } else {
        await this.sessionIdfCalculator.recomputeSessionIdf(sessionId);
      }
    }

    // 4. Generate and store embeddings (outside transaction)
    if (this.config.enableVectors) {
      await this.processEmbeddings(atoms);
    }
  }

  /**
   * Group atoms by session for batch processing
   */
  private groupAtomsBySession(atoms: Atom[]): Map<string, Atom[]> {
    const groups = new Map<string, Atom[]>();
    
    for (const atom of atoms) {
      const existing = groups.get(atom.session_id) || [];
      existing.push(atom);
      groups.set(atom.session_id, existing);
    }
    
    return groups;
  }

  /**
   * Process entities for atoms
   */
  private async processEntities(atoms: Atom[]): Promise<void> {
    const allEntities: Entity[] = [];

    for (const atom of atoms) {
      // Get session IDF weights for better entity weighting
      const idfWeights = this.sessionIdfCalculator.getSessionIdfWeights(atom.session_id);
      
      // Extract entities
      const entities = await this.entityExtractor.createEntitiesForAtom(
        atom.id,
        atom.text,
        idfWeights
      );
      
      allEntities.push(...entities);
    }

    // Insert entities
    if (allEntities.length > 0) {
      this.insertEntities(allEntities);
    }
  }

  /**
   * Process embeddings for atoms
   */
  private async processEmbeddings(atoms: Atom[]): Promise<void> {
    const atomIds = atoms.map(a => a.id);
    const texts = atoms.map(a => a.text);
    
    await this.embeddingManager.addAtomEmbeddings(atomIds, texts);
  }

  /**
   * Insert entities
   */
  private insertEntities(entities: Entity[]): void {
    const stmt = this.db.prepare(
      'INSERT OR REPLACE INTO entities (atom_id, entity, kind, weight) VALUES (?, ?, ?, ?)'
    );

    const transaction = this.db.transaction(() => {
      for (const entity of entities) {
        stmt.run(entity.atom_id, entity.entity, entity.kind, entity.weight);
      }
    });

    transaction();
  }

  /**
   * Full-text search using FTS5
   */
  searchFts(query: string, limit: number = 20): FtsResult[] {
    if (!this.config.enableFts) {
      return [];
    }

    const stmt = this.db.prepare(`
      SELECT 
        f.atom_id,
        f.text,
        f.rank
      FROM fts_atoms f
      WHERE fts_atoms MATCH ?
      ORDER BY rank
      LIMIT ?
    `);

    try {
      return stmt.all(query, limit) as FtsResult[];
    } catch (error) {
      console.warn('FTS search failed:', error);
      return [];
    }
  }

  /**
   * Vector search using HNSW
   */
  async searchVectors(
    query: string,
    limit: number = 20,
    efSearch?: number
  ): Promise<VectorSearchResult[]> {
    if (!this.config.enableVectors) {
      return [];
    }

    try {
      return await this.embeddingManager.search(query, limit, efSearch);
    } catch (error) {
      console.warn('Vector search failed:', error);
      return [];
    }
  }

  /**
   * Hybrid search combining FTS and vector search
   */
  async hybridSearch(
    query: string,
    context: SearchContext,
    config: HybridSearchConfig = DEFAULT_HYBRID_SEARCH_CONFIG,
    limit: number = 20
  ): Promise<SearchResults> {
    const startTime = Date.now();
    
    // Parallel search execution
    const [ftsResults, vectorResults] = await Promise.all([
      this.config.enableFts ? 
        this.searchFts(query, config.maxCandidates) : 
        Promise.resolve([]),
      this.config.enableVectors ? 
        this.searchVectors(query, config.maxCandidates, config.efSearch) :
        Promise.resolve([]),
    ]);

    // Collect unique atom IDs
    const candidateIds = new Set<string>();
    
    // Create score maps
    const ftsScores = new Map<string, number>();
    const vectorScores = new Map<string, number>();

    // Process FTS results
    for (const result of ftsResults) {
      candidateIds.add(result.atom_id);
      ftsScores.set(result.atom_id, 1 / (1 + result.rank)); // Convert rank to score
    }

    // Process vector results
    for (const result of vectorResults) {
      candidateIds.add(result.atom_id);
      vectorScores.set(result.atom_id, result.similarity);
    }

    // Get atom details and entities
    const atoms = this.getAtomsByIds(Array.from(candidateIds));
    const atomEntities = this.getEntitiesByAtomIds(Array.from(candidateIds));

    // Normalize scores
    const maxFtsScore = Math.max(...Array.from(ftsScores.values()), 1);
    const maxVectorScore = Math.max(...Array.from(vectorScores.values()), 1);

    // Calculate hybrid scores
    const scoredResults = atoms.map(atom => {
      const ftsScore = (ftsScores.get(atom.id) || 0) / maxFtsScore;
      const vectorScore = (vectorScores.get(atom.id) || 0) / maxVectorScore;
      
      // Calculate entity overlap bonus if requested
      let entityScore = 0;
      if (config.entityWeight > 0) {
        const atomEntityList = atomEntities.get(atom.id) || [];
        const atomEntitySet = new Set(
          atomEntityList.map(e => e.entity.toLowerCase())
        );
        const queryTokens = new Set(query.toLowerCase().split(/\s+/));
        
        const overlap = [...queryTokens].filter(token => 
          [...atomEntitySet].some(entity => entity.includes(token))
        ).length;
        
        entityScore = overlap / Math.max(queryTokens.size, 1);
      }

      const finalScore = 
        config.ftsWeight * ftsScore +
        config.vectorWeight * vectorScore +
        config.entityWeight * entityScore;

      return {
        atom,
        score: finalScore,
        scoreComponents: {
          fts: ftsScore,
          vector: vectorScore,
          entity: entityScore,
          final: finalScore,
        },
        entities: atomEntities.get(atom.id) || [],
      };
    });

    // Sort by score and apply session filtering
    let filteredResults = scoredResults
      .filter(result => {
        // Apply session context filtering
        if (context.includeEntities && context.includeEntities.length > 0) {
          return result.entities.some(e => context.includeEntities!.includes(e.kind));
        }
        if (context.excludeEntities && context.excludeEntities.length > 0) {
          return !result.entities.some(e => context.excludeEntities!.includes(e.kind));
        }
        return true;
      })
      .sort((a, b) => b.score - a.score);

    // Apply diversity if requested
    if (config.diversify) {
      filteredResults = this.diversifyResults(filteredResults, config.diversityLambda);
    }

    const searchTime = Date.now() - startTime;

    return {
      atoms: filteredResults.slice(0, limit),
      metadata: {
        totalCandidates: candidateIds.size,
        searchTime,
        method: 'hybrid',
      },
    };
  }

  /**
   * Diversify results using entity-based MMR
   */
  private diversifyResults(
    results: SearchResults['atoms'],
    lambda: number
  ): SearchResults['atoms'] {
    if (results.length <= 1) return results;

    const selected: SearchResults['atoms'] = [];
    const remaining = [...results];
    
    // Always select the highest scoring result first
    selected.push(remaining.shift()!);

    while (remaining.length > 0 && selected.length < 20) {
      let bestIdx = 0;
      let bestScore = -Infinity;

      for (let i = 0; i < remaining.length; i++) {
        const candidate = remaining[i];
        const relevanceScore = candidate.score;
        
        // Calculate max similarity to already selected items
        let maxSimilarity = 0;
        for (const selectedItem of selected) {
          const similarity = this.calculateEntitySimilarity(
            candidate.entities || [],
            selectedItem.entities || []
          );
          maxSimilarity = Math.max(maxSimilarity, similarity);
        }

        // MMR score: balance relevance and diversity
        const mmrScore = lambda * relevanceScore - (1 - lambda) * maxSimilarity;
        
        if (mmrScore > bestScore) {
          bestScore = mmrScore;
          bestIdx = i;
        }
      }

      selected.push(remaining.splice(bestIdx, 1)[0]);
    }

    return selected;
  }

  /**
   * Calculate entity-based similarity between two entity sets
   */
  private calculateEntitySimilarity(entities1: Entity[], entities2: Entity[]): number {
    if (entities1.length === 0 || entities2.length === 0) {
      return 0;
    }

    const set1 = new Set(entities1.map(e => `${e.entity}:${e.kind}`));
    const set2 = new Set(entities2.map(e => `${e.entity}:${e.kind}`));

    const intersection = [...set1].filter(x => set2.has(x)).length;
    const union = set1.size + set2.size - intersection;

    return union > 0 ? intersection / union : 0; // Jaccard similarity
  }

  /**
   * Get atoms by IDs
   */
  private getAtomsByIds(ids: string[]): Atom[] {
    if (ids.length === 0) return [];

    const placeholders = ids.map(() => '?').join(',');
    const stmt = this.db.prepare(`
      SELECT id, session_id, turn_idx, role, type, text, json_meta, ts
      FROM atoms 
      WHERE id IN (${placeholders})
    `);

    const results = stmt.all(...ids) as Array<{
      id: string;
      session_id: string;
      turn_idx: number;
      role: string;
      type: string;
      text: string;
      json_meta: string | null;
      ts: number;
    }>;

    return results.map(row => ({
      ...row,
      json_meta: row.json_meta ? JSON.parse(row.json_meta) : undefined,
    })) as Atom[];
  }

  /**
   * Get entities by atom IDs
   */
  private getEntitiesByAtomIds(atomIds: string[]): Map<string, Entity[]> {
    if (atomIds.length === 0) return new Map();

    const placeholders = atomIds.map(() => '?').join(',');
    const stmt = this.db.prepare(`
      SELECT atom_id, entity, kind, weight
      FROM entities 
      WHERE atom_id IN (${placeholders})
      ORDER BY weight DESC
    `);

    const results = stmt.all(...atomIds) as Entity[];
    
    const entityMap = new Map<string, Entity[]>();
    for (const entity of results) {
      const existing = entityMap.get(entity.atom_id) || [];
      existing.push(entity);
      entityMap.set(entity.atom_id, existing);
    }

    return entityMap;
  }

  /**
   * Get session statistics
   */
  getSessionStats(sessionId: string): SessionStats | null {
    const stmt = this.db.prepare(
      'SELECT * FROM session_stats WHERE session_id = ?'
    );
    return stmt.get(sessionId) as SessionStats | null;
  }

  /**
   * Get atoms by session with pagination
   */
  getAtomsBySession(
    sessionId: string,
    limit: number = 100,
    offset: number = 0
  ): Atom[] {
    const stmt = this.db.prepare(`
      SELECT id, session_id, turn_idx, role, type, text, json_meta, ts
      FROM atoms 
      WHERE session_id = ?
      ORDER BY turn_idx, ts
      LIMIT ? OFFSET ?
    `);

    const results = stmt.all(sessionId, limit, offset) as Array<{
      id: string;
      session_id: string;
      turn_idx: number;
      role: string;
      type: string;
      text: string;
      json_meta: string | null;
      ts: number;
    }>;

    return results.map(row => ({
      ...row,
      json_meta: row.json_meta ? JSON.parse(row.json_meta) : undefined,
    })) as Atom[];
  }

  /**
   * Delete atoms by session
   */
  deleteSession(sessionId: string): number {
    const stmt = this.db.prepare('DELETE FROM atoms WHERE session_id = ?');
    const result = stmt.run(sessionId);
    
    // Clear from vector index (atoms will be removed by trigger)
    // But we need to rebuild the index
    if (this.config.enableVectors) {
      // Could be optimized to only remove specific vectors
      console.log(`Session ${sessionId} deleted, consider rebuilding vector index`);
    }
    
    return result.changes;
  }

  /**
   * Get database statistics
   */
  getStats(): {
    atomCount: number;
    sessionCount: number;
    entityCount: number;
    vectorCount: number;
    indexStats: any;
  } {
    const atomCount = (this.db.prepare('SELECT COUNT(*) as count FROM atoms').get() as { count: number }).count;
    const sessionCount = (this.db.prepare('SELECT COUNT(DISTINCT session_id) as count FROM atoms').get() as { count: number }).count;
    const entityCount = (this.db.prepare('SELECT COUNT(*) as count FROM entities').get() as { count: number }).count;
    const vectorCount = (this.db.prepare('SELECT COUNT(*) as count FROM vectors').get() as { count: number }).count;

    return {
      atomCount,
      sessionCount,
      entityCount,
      vectorCount,
      indexStats: this.embeddingManager.getIndexStats(),
    };
  }
}

/**
 * Create atoms database with default configuration
 */
export function createAtomsDatabase(db: DB, config?: Partial<AtomDbConfig>): AtomsDatabase {
  const defaultConfig: AtomDbConfig = {
    path: ':memory:',
    enableFts: true,
    enableVectors: true,
    enableEntities: true,
    entityExtraction: {
      patterns: {},
      ner: { enabled: false, fallbackToPatterns: true },
      useSessionIdf: true,
      minWeight: 0.1,
    } as any,
    sessionIdf: {
      smoothing: 0.5,
      minDf: 1,
      incremental: true,
    },
    hnsw: {
      M: 16,
      efConstruction: 200,
      efSearch: 50,
      maxElements: 100000,
      randomSeed: 12345,
    },
    embedding: {
      modelName: 'sentence-transformers/all-MiniLM-L6-v2',
      dimension: 384,
      maxTokens: 512,
      batchSize: 32,
      cpuOnly: true,
    },
    hybridSearch: DEFAULT_HYBRID_SEARCH_CONFIG,
    batchSize: 100,
    pragmas: {
      journal_mode: 'WAL',
      synchronous: 'NORMAL',
      cache_size: 10000,
      foreign_keys: 'ON',
    },
  };

  const fullConfig = {
    ...defaultConfig,
    ...config,
    entityExtraction: {
      ...defaultConfig.entityExtraction,
      ...config?.entityExtraction,
    },
    sessionIdf: {
      ...defaultConfig.sessionIdf,
      ...config?.sessionIdf,
    },
    hnsw: {
      ...defaultConfig.hnsw,
      ...config?.hnsw,
    },
    embedding: {
      ...defaultConfig.embedding,
      ...config?.embedding,
    },
    hybridSearch: {
      ...defaultConfig.hybridSearch,
      ...config?.hybridSearch,
    },
    pragmas: {
      ...defaultConfig.pragmas,
      ...config?.pragmas,
    },
  };

  return new AtomsDatabase(db, fullConfig);
}