/**
 * @fileoverview Main exports for Lethe agent context atoms system
 * Milestone 1: Complete data model implementation
 */

// Core database functionality
export { AtomsDatabase, createAtomsDatabase, DEFAULT_HYBRID_SEARCH_CONFIG } from './atoms-db.js';

// Type definitions
export * from './atoms-types.js';

// Entity extraction
export { 
  EntityExtractor, 
  createEntityExtractor, 
  DEFAULT_ENTITY_PATTERNS, 
  DEFAULT_ENTITY_CONFIG,
  type NerModel 
} from './entity-extraction.js';

// Session IDF calculation
export { 
  SessionIdfCalculator, 
  createSessionIdfCalculator, 
  DEFAULT_SESSION_IDF_CONFIG 
} from './session-idf.js';

// Embeddings and HNSW
export { 
  EmbeddingManager, 
  createEmbeddingManager, 
  CpuEmbeddingModel, 
  MemoryHnswIndex, 
  DEFAULT_EMBEDDING_CONFIG, 
  DEFAULT_HNSW_CONFIG,
  type EmbeddingModel,
  type HnswIndex 
} from './embeddings.js';

// Legacy exports for compatibility
export * from './index.js';