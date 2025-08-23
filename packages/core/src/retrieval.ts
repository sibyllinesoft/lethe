import type { CtxDatabase, Embedding } from '@ctx-run/sqlite';
import type { EmbeddingProvider } from '@ctx-run/embeddings';
import type { CrossEncoderReranker } from '@ctx-run/reranker';
import type { Config, Candidate } from './types.js';
import { DfIdfBuilder } from './dfidf.js';

export interface SearchResult {
  chunkId: string;
  score: number;
  metadata?: any;
}

export class RetrievalSystem {
  private db: CtxDatabase;
  private embeddings: EmbeddingProvider;
  private reranker: CrossEncoderReranker;
  private dfIdfBuilder: DfIdfBuilder;
  private config: Config;

  constructor(
    db: CtxDatabase,
    embeddings: EmbeddingProvider,
    reranker: CrossEncoderReranker,
    config: Config
  ) {
    this.db = db;
    this.embeddings = embeddings;
    this.reranker = reranker;
    this.dfIdfBuilder = new DfIdfBuilder(db);
    this.config = config;
  }

  // Main retrieval pipeline
  async search(
    sessionId: string,
    queries: string[],
    topK: number = 50
  ): Promise<Candidate[]> {
    // Get BM25 results for all queries
    const bm25Results = await this.bm25Search(queries, sessionId, topK * 2);
    
    // Get vector results for all queries
    const vectorResults = await this.vectorSearch(queries, topK * 2);
    
    // Combine and hybrid score
    const candidates = await this.hybridScore(bm25Results, vectorResults, sessionId);
    
    return candidates.slice(0, topK);
  }

  // BM25 lexical search
  async bm25Search(
    queries: string[],
    sessionId: string,
    k: number
  ): Promise<Map<string, number>> {
    const allScores = new Map<string, number>();
    
    for (const query of queries) {
      // Use FTS for initial candidate retrieval
      const ftsResults = this.db.searchChunks(query, sessionId);
      const candidateIds = ftsResults.map(r => r.chunkId);
      
      if (candidateIds.length === 0) continue;
      
      // Calculate precise BM25 scores for candidates
      const scores = await this.dfIdfBuilder.calculateBM25Scores(
        sessionId,
        query,
        candidateIds
      );
      
      // Merge scores (take max for each chunk)
      for (const [chunkId, score] of scores) {
        const currentScore = allScores.get(chunkId) || 0;
        allScores.set(chunkId, Math.max(currentScore, score));
      }
    }
    
    return allScores;
  }

  // Vector semantic search
  async vectorSearch(queries: string[], k: number): Promise<Map<string, number>> {
    const allScores = new Map<string, number>();
    
    // Get embeddings for all queries
    const queryEmbeddings = await this.embeddings.embed(queries);
    
    // Get all chunk embeddings
    const chunkEmbeddings = this.db.getAllEmbeddings();
    
    if (chunkEmbeddings.length === 0) {
      return allScores;
    }
    
    for (let i = 0; i < queries.length; i++) {
      const queryVec = queryEmbeddings[i];
      const similarities: Array<{ chunkId: string; score: number }> = [];
      
      // Calculate cosine similarity with all chunks
      for (const embedding of chunkEmbeddings) {
        const similarity = this.cosineSimilarity(queryVec, embedding.vec);
        similarities.push({ chunkId: embedding.chunkId, score: similarity });
      }
      
      // Sort and take top-k
      similarities.sort((a, b) => b.score - a.score);
      const topSimilarities = similarities.slice(0, k);
      
      // Merge scores (take max for each chunk)
      for (const { chunkId, score } of topSimilarities) {
        const currentScore = allScores.get(chunkId) || 0;
        allScores.set(chunkId, Math.max(currentScore, score));
      }
    }
    
    return allScores;
  }

  // Hybrid scoring that combines BM25 and vector scores
  private async hybridScore(
    bm25Scores: Map<string, number>,
    vectorScores: Map<string, number>,
    sessionId: string
  ): Promise<Candidate[]> {
    const candidates: Candidate[] = [];
    const allChunkIds = new Set([
      ...bm25Scores.keys(),
      ...vectorScores.keys()
    ]);
    
    const chunks = this.db.getChunksByIds([...allChunkIds]);
    const chunkMap = new Map(chunks.map(c => [c.id, c]));
    
    // Normalize scores to 0-1 range
    const bm25Max = Math.max(...bm25Scores.values()) || 1;
    const vectorMax = Math.max(...vectorScores.values()) || 1;
    
    for (const chunkId of allChunkIds) {
      const chunk = chunkMap.get(chunkId);
      if (!chunk) continue;
      
      const bm25Score = (bm25Scores.get(chunkId) || 0) / bm25Max;
      const vectorScore = (vectorScores.get(chunkId) || 0) / vectorMax;
      
      // Apply kind-specific boosts
      const kindBoost = this.config.retrieval.gamma_kind_boost[chunk.kind] || 0;
      
      // Hybrid formula: α*BM25 + β*Vector + γ*KindBoost
      const hybridScore = 
        this.config.retrieval.alpha * bm25Score +
        this.config.retrieval.beta * vectorScore +
        kindBoost;
      
      candidates.push({
        id: chunkId,
        text: chunk.text,
        messageId: chunk.messageId,
        kind: chunk.kind,
        bm25Score,
        vectorScore,
        hybridScore
      });
    }
    
    // Sort by hybrid score
    candidates.sort((a, b) => b.hybridScore - a.hybridScore);
    return candidates;
  }

  // Rerank candidates using cross-encoder
  async rerank(
    query: string,
    candidates: Candidate[],
    topkOut: number
  ): Promise<Candidate[]> {
    if (candidates.length === 0) return [];
    
    const rankingPairs = candidates.map(candidate => ({
      query,
      text: candidate.text,
      id: candidate.id
    }));
    
    const rerankResults = await this.reranker.rankPairs(
      rankingPairs,
      Math.min(topkOut, candidates.length)
    );
    
    const rerankMap = new Map(rerankResults.map(r => [r.id, r.score]));
    const rerankedCandidates = candidates
      .filter(c => rerankMap.has(c.id))
      .map(c => ({ ...c, rerankScore: rerankMap.get(c.id)! }))
      .sort((a, b) => b.rerankScore! - a.rerankScore!);
    
    return rerankedCandidates;
  }

  // Submodular selection for diversity
  submodularSelect(candidates: Candidate[], maxChunks: number): Candidate[] {
    if (candidates.length <= maxChunks) return candidates;
    
    const selected: Candidate[] = [];
    const remaining = [...candidates];
    const selectedEntities = new Set<string>();
    
    for (let i = 0; i < maxChunks && remaining.length > 0; i++) {
      let bestIdx = 0;
      let bestScore = -1;
      
      for (let j = 0; j < remaining.length; j++) {
        const candidate = remaining[j];
        const entities = this.extractEntities(candidate.text);
        
        // Count new entities this candidate would add
        const newEntities = entities.filter(e => !selectedEntities.has(e));
        const diversityScore = newEntities.length;
        
        // Combined score: relevance + diversity
        const combinedScore = 
          (candidate.rerankScore || candidate.hybridScore) + 
          0.1 * diversityScore;
        
        if (combinedScore > bestScore) {
          bestScore = combinedScore;
          bestIdx = j;
        }
      }
      
      // Add best candidate
      const selected_candidate = remaining.splice(bestIdx, 1)[0];
      selected.push(selected_candidate);
      
      // Update selected entities
      const entities = this.extractEntities(selected_candidate.text);
      entities.forEach(e => selectedEntities.add(e));
    }
    
    return selected;
  }

  private extractEntities(text: string): string[] {
    // Simple entity extraction - look for capitalized words and code identifiers
    const entities: string[] = [];
    
    // Capitalized words (potential proper nouns)
    const capitalizedWords = text.match(/\b[A-Z][a-z]+\b/g) || [];
    entities.push(...capitalizedWords);
    
    // Code identifiers (camelCase, snake_case)
    const codeIdentifiers = text.match(/\b[a-z]+[A-Z][a-zA-Z]*\b|\b[a-z]+_[a-z_]+\b/g) || [];
    entities.push(...codeIdentifiers);
    
    // File extensions and paths
    const filePaths = text.match(/[\w-]+\.[a-z]{2,4}\b/g) || [];
    entities.push(...filePaths);
    
    return [...new Set(entities)]; // Deduplicate
  }

  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) {
      throw new Error('Vectors must have same dimension');
    }
    
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    if (normA === 0 || normB === 0) return 0;
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  // Indexing operations
  async ensureEmbeddings(sessionId: string): Promise<void> {
    const missingChunkIds = this.db.getChunksWithoutEmbeddings(sessionId);
    
    if (missingChunkIds.length === 0) {
      return;
    }
    
    console.log(`Generating embeddings for ${missingChunkIds.length} chunks...`);
    
    const chunks = this.db.getChunksByIds(missingChunkIds);
    const texts = chunks.map(c => c.text);
    
    // Generate embeddings in batches
    const embeddings = await this.embeddings.embed(texts);
    
    // Store embeddings
    const embeddingData: Embedding[] = chunks.map((chunk, i) => ({
      chunkId: chunk.id,
      dim: this.embeddings.dim,
      vec: embeddings[i]
    }));
    
    this.db.insertEmbeddings(embeddingData);
    console.log(`Embeddings generated and stored for ${embeddingData.length} chunks`);
  }

  async rebuildDfIdf(sessionId: string): Promise<void> {
    await this.dfIdfBuilder.rebuild(sessionId);
  }
}