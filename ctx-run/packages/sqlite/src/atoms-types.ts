/**
 * @fileoverview TypeScript types for Lethe Agent-Context Manager atoms data model
 * Milestone 1: Conversation atoms, entities, and indexes
 */

// Core atom types
export type AtomRole = 'user' | 'assistant' | 'tool' | 'system';
export type AtomType = 'message' | 'action' | 'args' | 'observation' | 'plan' | 'error' | 'result';
export type EntityKind = 'id' | 'file' | 'error' | 'api' | 'tool' | 'person' | 'org' | 'misc';

/**
 * Core conversation atom - represents any piece of agent conversation
 */
export interface Atom {
  id: string;
  session_id: string;
  turn_idx: number;
  role: AtomRole;
  type: AtomType;
  text: string;
  json_meta?: Record<string, any>;
  ts: number;
}

/**
 * Extracted entity from atom content
 */
export interface Entity {
  atom_id: string;
  entity: string;
  kind: EntityKind;
  weight: number;
}

/**
 * Dense vector embedding for atom
 */
export interface Vector {
  atom_id: string;
  dim: number;
  blob: Buffer;
}

/**
 * Session-level IDF statistics for terms
 */
export interface SessionIdf {
  session_id: string;
  term: string;
  df: number; // document frequency (atoms containing term)
  idf: number; // inverse document frequency
  updated_at: number;
}

/**
 * Atom with associated entities (from view)
 */
export interface AtomWithEntities extends Atom {
  entities?: string; // Concatenated entities string
}

/**
 * Session statistics (from view)
 */
export interface SessionStats {
  session_id: string;
  total_atoms: number;
  distinct_types: number;
  distinct_roles: number;
  start_ts: number;
  end_ts: number;
  max_turns: number;
}

/**
 * Full-text search result
 */
export interface FtsResult {
  atom_id: string;
  text: string;
  rank: number;
}

/**
 * Vector search result
 */
export interface VectorSearchResult {
  atom_id: string;
  similarity: number;
}

/**
 * Entity extraction patterns and configuration
 */
export interface EntityExtractionConfig {
  // Regex patterns for different entity types
  patterns: {
    id: RegExp[];        // Code identifiers
    file: RegExp[];      // File paths
    error: RegExp[];     // Error codes
    api: RegExp[];       // API endpoints
    tool: RegExp[];      // Tool names
    misc: RegExp[];      // Miscellaneous patterns
  };
  
  // Named Entity Recognition model config
  ner?: {
    modelPath?: string;
    enabled: boolean;
    fallbackToPatterns: boolean;
  };
  
  // Weight calculation
  useSessionIdf: boolean;
  minWeight: number;
}

/**
 * Session IDF computation parameters
 */
export interface SessionIdfConfig {
  // Smoothing parameters for IDF calculation
  smoothing: number; // Added to denominator (default 0.5)
  
  // Minimum DF threshold
  minDf: number;
  
  // Whether to compute incrementally
  incremental: boolean;
  
  // Terms to exclude from IDF
  stopWords?: Set<string>;
}

/**
 * HNSW index configuration
 */
export interface HnswConfig {
  M: number;                    // Number of bi-directional links for each node (default 16)
  efConstruction: number;       // Size of dynamic candidate list (default 200)
  efSearch: number;            // Size of dynamic candidate list for search (adjustable)
  maxElements: number;         // Maximum number of elements
  randomSeed: number;          // Random seed for reproducibility
}

/**
 * Embedding model configuration
 */
export interface EmbeddingConfig {
  modelName: string;           // Name/path of embedding model
  dimension: number;           // Embedding dimension
  maxTokens: number;          // Maximum input tokens
  batchSize: number;          // Batch size for processing
  cpuOnly: boolean;           // Force CPU-only inference
}

/**
 * Entity extraction result
 */
export interface ExtractedEntities {
  entities: Array<{
    text: string;
    kind: EntityKind;
    confidence: number;
    source: 'regex' | 'ner';
  }>;
  processingTime: number;
}

/**
 * Atom insertion batch
 */
export interface AtomBatch {
  atoms: Atom[];
  entities: Entity[];
  vectors?: Vector[];
}

/**
 * Search query context
 */
export interface SearchContext {
  sessionId: string;
  windowTurns?: number;        // Look back N turns for context
  entityOverlapThreshold?: number;
  includeEntities?: EntityKind[];
  excludeEntities?: EntityKind[];
}

/**
 * Hybrid search configuration
 */
export interface HybridSearchConfig {
  // Weights for combining scores
  ftsWeight: number;           // FTS (BM25) weight
  vectorWeight: number;        // Vector similarity weight
  entityWeight: number;        // Entity overlap bonus
  
  // Search parameters
  maxCandidates: number;       // Max candidates per method
  efSearch: number;           // HNSW search parameter
  
  // Diversification
  diversify: boolean;
  diversityLambda: number;     // MMR-style diversity parameter
}

/**
 * Search results with metadata
 */
export interface SearchResults {
  atoms: Array<{
    atom: Atom;
    score: number;
    scoreComponents: {
      fts: number;
      vector: number;
      entity: number;
      final: number;
    };
    entities: Entity[];
  }>;
  metadata: {
    totalCandidates: number;
    searchTime: number;
    method: 'hybrid' | 'fts' | 'vector';
  };
}

/**
 * Database connection and configuration
 */
export interface AtomDbConfig {
  path: string;
  
  // Feature flags
  enableFts: boolean;
  enableVectors: boolean;
  enableEntities: boolean;
  
  // Configuration objects
  entityExtraction: EntityExtractionConfig;
  sessionIdf: SessionIdfConfig;
  hnsw: HnswConfig;
  embedding: EmbeddingConfig;
  hybridSearch: HybridSearchConfig;
  
  // Performance settings
  batchSize: number;
  pragmas?: Record<string, string | number>;
}