/**
 * @fileoverview Embedding and HNSW dense index for Lethe agent context atoms
 * CPU-compatible embedding models with HNSW indexing
 */

import type Database from 'better-sqlite3';

type DB = Database.Database;
import { Vector, EmbeddingConfig, HnswConfig, VectorSearchResult } from './atoms-types.js';

/**
 * Default embedding configuration
 * Uses a small, CPU-compatible model (≤100M params)
 */
export const DEFAULT_EMBEDDING_CONFIG: EmbeddingConfig = {
  modelName: 'sentence-transformers/all-MiniLM-L6-v2', // 22M params, 384 dims
  dimension: 384,
  maxTokens: 512,
  batchSize: 32,
  cpuOnly: true,
};

/**
 * Default HNSW configuration
 */
export const DEFAULT_HNSW_CONFIG: HnswConfig = {
  M: 16,                    // Number of bi-directional links
  efConstruction: 200,      // Construction parameter
  efSearch: 50,            // Default search parameter (adjustable)
  maxElements: 100000,      // Max elements in index
  randomSeed: 12345,       // For reproducibility
};

/**
 * Abstract embedding model interface
 * Allows different implementations (transformers.js, ONNX, etc.)
 */
export interface EmbeddingModel {
  embed(texts: string[]): Promise<Float32Array[]>;
  embedSingle(text: string): Promise<Float32Array>;
  getDimension(): number;
  getModelName(): string;
}

/**
 * HNSW index interface
 * Can be implemented with different libraries (hnswlib-node, custom WASM, etc.)
 */
export interface HnswIndex {
  addVector(id: string, vector: Float32Array): void;
  search(query: Float32Array, k: number, efSearch?: number): VectorSearchResult[];
  getSize(): number;
  save(path?: string): void;
  load(path?: string): void;
  clear(): void;
}

/**
 * Simple CPU-based embedding model using transformers.js
 * Falls back to random vectors if model loading fails
 */
export class CpuEmbeddingModel implements EmbeddingModel {
  private config: EmbeddingConfig;
  private model: any = null;
  private initialized = false;

  constructor(config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG) {
    this.config = config;
  }

  /**
   * Initialize the model (lazy loading)
   */
  private async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      // Try to use transformers.js for CPU inference
      const { pipeline } = await import('@xenova/transformers');
      
      this.model = await pipeline(
        'feature-extraction',
        this.config.modelName
      );
      
      console.log(`Loaded embedding model: ${this.config.modelName}`);
    } catch (error) {
      console.warn('Failed to load transformers.js model, using fallback:', error);
      // Model will remain null, fallback to random embeddings
    }
    
    this.initialized = true;
  }

  /**
   * Generate random embedding (fallback)
   */
  private generateRandomEmbedding(): Float32Array {
    const embedding = new Float32Array(this.config.dimension);
    for (let i = 0; i < this.config.dimension; i++) {
      embedding[i] = (Math.random() - 0.5) * 2; // Range [-1, 1]
    }
    
    // L2 normalize
    let norm = 0;
    for (let i = 0; i < this.config.dimension; i++) {
      norm += embedding[i] * embedding[i];
    }
    norm = Math.sqrt(norm);
    
    if (norm > 0) {
      for (let i = 0; i < this.config.dimension; i++) {
        embedding[i] /= norm;
      }
    }
    
    return embedding;
  }

  /**
   * Embed multiple texts
   */
  async embed(texts: string[]): Promise<Float32Array[]> {
    await this.initialize();
    
    if (!this.model) {
      // Fallback to random embeddings
      return texts.map(() => this.generateRandomEmbedding());
    }

    try {
      const results = await this.model(texts, {
        pooling: 'mean',
        normalize: true,
      });
      
      // Convert to Float32Array
      return results.tolist().map((embedding: number[]) => 
        new Float32Array(embedding)
      );
    } catch (error) {
      console.warn('Model inference failed, using random embeddings:', error);
      return texts.map(() => this.generateRandomEmbedding());
    }
  }

  /**
   * Embed single text
   */
  async embedSingle(text: string): Promise<Float32Array> {
    const results = await this.embed([text]);
    return results[0];
  }

  /**
   * Get embedding dimension
   */
  getDimension(): number {
    return this.config.dimension;
  }

  /**
   * Get model name
   */
  getModelName(): string {
    return this.config.modelName;
  }
}

/**
 * In-memory HNSW index implementation
 * Uses a simple brute-force search as fallback if HNSW library not available
 */
export class MemoryHnswIndex implements HnswIndex {
  private config: HnswConfig;
  private vectors = new Map<string, Float32Array>();
  private hnswLib: any = null;
  private index: any = null;
  private initialized = false;

  constructor(config: HnswConfig = DEFAULT_HNSW_CONFIG) {
    this.config = config;
  }

  /**
   * Initialize HNSW library (lazy loading)
   */
  private async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      // Try to use hnswlib-node (with type assertion)
      // @ts-ignore - Optional dependency, may not be available
      this.hnswLib = await import('hnswlib-node') as any;
      
      this.index = new this.hnswLib.HierarchicalNSW('cosine', 384); // Default dimension
      this.index.initIndex(this.config.maxElements, this.config.M, this.config.efConstruction, this.config.randomSeed);
      this.index.setEfSearch(this.config.efSearch);
      
      console.log('Initialized HNSW index with native library');
    } catch (error) {
      console.warn('Failed to load hnswlib-node, using brute-force search:', error);
      // index remains null, will use brute-force
    }
    
    this.initialized = true;
  }

  /**
   * Cosine similarity calculation
   */
  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    return denominator > 0 ? dotProduct / denominator : 0;
  }

  /**
   * Brute-force search (fallback)
   */
  private bruteForceSearch(query: Float32Array, k: number): VectorSearchResult[] {
    const results: Array<{ id: string; similarity: number }> = [];
    
    for (const [id, vector] of this.vectors) {
      const similarity = this.cosineSimilarity(query, vector);
      results.push({ id, similarity });
    }
    
    // Sort by similarity (descending) and take top k
    results.sort((a, b) => b.similarity - a.similarity);
    
    return results.slice(0, k).map(r => ({
      atom_id: r.id,
      similarity: r.similarity,
    }));
  }

  /**
   * Add vector to index
   */
  addVector(id: string, vector: Float32Array): void {
    this.vectors.set(id, vector);
    
    if (this.index) {
      try {
        this.index.addPoint(vector, id);
      } catch (error) {
        console.warn('Failed to add vector to HNSW index:', error);
      }
    }
  }

  /**
   * Search for similar vectors
   */
  search(query: Float32Array, k: number, efSearch?: number): VectorSearchResult[] {
    if (!this.initialized) {
      this.initialize();
    }

    if (this.index) {
      try {
        if (efSearch) {
          this.index.setEfSearch(efSearch);
        }
        
        const results = this.index.searchKnn(query, k);
        return results.neighbors.map((id: string, idx: number) => ({
          atom_id: id,
          similarity: 1 - results.distances[idx], // Convert distance to similarity
        }));
      } catch (error) {
        console.warn('HNSW search failed, falling back to brute-force:', error);
      }
    }
    
    // Fallback to brute-force search
    return this.bruteForceSearch(query, k);
  }

  /**
   * Get index size
   */
  getSize(): number {
    return this.vectors.size;
  }

  /**
   * Save index (not implemented for memory index)
   */
  save(path?: string): void {
    console.warn('Save not implemented for memory HNSW index');
  }

  /**
   * Load index (not implemented for memory index)
   */
  load(path?: string): void {
    console.warn('Load not implemented for memory HNSW index');
  }

  /**
   * Clear index
   */
  clear(): void {
    this.vectors.clear();
    if (this.index) {
      this.index = null;
      this.initialized = false;
    }
  }
}

/**
 * Embedding manager for atoms
 */
export class EmbeddingManager {
  private db: DB;
  private model: EmbeddingModel;
  private index: HnswIndex;
  private embeddingConfig: EmbeddingConfig;
  private hnswConfig: HnswConfig;

  constructor(
    db: DB,
    embeddingConfig: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG,
    hnswConfig: HnswConfig = DEFAULT_HNSW_CONFIG
  ) {
    this.db = db;
    this.embeddingConfig = embeddingConfig;
    this.hnswConfig = hnswConfig;
    this.model = new CpuEmbeddingModel(embeddingConfig);
    this.index = new MemoryHnswIndex(hnswConfig);
  }

  /**
   * Set custom embedding model
   */
  setModel(model: EmbeddingModel): void {
    this.model = model;
  }

  /**
   * Set custom HNSW index
   */
  setIndex(index: HnswIndex): void {
    this.index = index;
  }

  /**
   * Generate embeddings for atoms
   */
  async generateEmbeddings(texts: string[]): Promise<Float32Array[]> {
    return await this.model.embed(texts);
  }

  /**
   * Store vectors in database
   */
  storeVectors(vectors: Vector[]): void {
    const stmt = this.db.prepare(
      'INSERT OR REPLACE INTO vectors (atom_id, dim, blob) VALUES (?, ?, ?)'
    );

    const transaction = this.db.transaction(() => {
      for (const vector of vectors) {
        stmt.run(vector.atom_id, vector.dim, vector.blob);
      }
    });

    transaction();
  }

  /**
   * Load vectors from database and build index
   */
  async buildIndex(): Promise<void> {
    console.log('Building vector index...');
    
    const stmt = this.db.prepare('SELECT atom_id, dim, blob FROM vectors');
    const vectors = stmt.all() as Vector[];
    
    this.index.clear();
    
    for (const vectorRow of vectors) {
      const vector = new Float32Array(
        vectorRow.blob.buffer,
        vectorRow.blob.byteOffset,
        vectorRow.blob.byteLength / 4
      );
      
      this.index.addVector(vectorRow.atom_id, vector);
    }
    
    console.log(`Built index with ${vectors.length} vectors`);
  }

  /**
   * Search for similar atoms
   */
  async search(queryText: string, k: number, efSearch?: number): Promise<VectorSearchResult[]> {
    const queryVector = await this.model.embedSingle(queryText);
    return this.index.search(queryVector, k, efSearch);
  }

  /**
   * Add new atom embeddings
   */
  async addAtomEmbeddings(atomIds: string[], texts: string[]): Promise<void> {
    if (atomIds.length !== texts.length) {
      throw new Error('atomIds and texts arrays must have the same length');
    }

    if (atomIds.length === 0) return;

    // Generate embeddings
    const embeddings = await this.generateEmbeddings(texts);
    
    // Convert to database format
    const vectors: Vector[] = embeddings.map((embedding, idx) => ({
      atom_id: atomIds[idx],
      dim: embedding.length,
      blob: Buffer.from(embedding.buffer),
    }));

    // Store in database
    this.storeVectors(vectors);

    // Add to index
    for (let i = 0; i < atomIds.length; i++) {
      this.index.addVector(atomIds[i], embeddings[i]);
    }
  }

  /**
   * Get index statistics
   */
  getIndexStats(): {
    vectorCount: number;
    dimension: number;
    modelName: string;
    hnswConfig: HnswConfig;
  } {
    return {
      vectorCount: this.index.getSize(),
      dimension: this.model.getDimension(),
      modelName: this.model.getModelName(),
      hnswConfig: this.hnswConfig,
    };
  }
}

/**
 * Utility function to create embedding manager
 */
export function createEmbeddingManager(
  db: DB,
  embeddingConfig?: Partial<EmbeddingConfig>,
  hnswConfig?: Partial<HnswConfig>
): EmbeddingManager {
  const fullEmbeddingConfig = {
    ...DEFAULT_EMBEDDING_CONFIG,
    ...embeddingConfig,
  };
  
  const fullHnswConfig = {
    ...DEFAULT_HNSW_CONFIG,
    ...hnswConfig,
  };
  
  return new EmbeddingManager(db, fullEmbeddingConfig, fullHnswConfig);
}