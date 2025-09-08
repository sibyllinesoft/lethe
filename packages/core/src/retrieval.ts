import type { CtxDatabase, Embedding, Chunk } from '@ctx-run/sqlite';
import type { EmbeddingProvider } from '@ctx-run/embeddings';
import type { CrossEncoderReranker } from '@ctx-run/reranker';
import type { 
  Config, 
  Candidate, 
  EnhancedCandidate,
  Result,
  LetheError,
  PerformanceMetrics,
  BudgetTracker
} from './types.js';
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
  private budgetTracker: BudgetTracker;
  private performanceMetrics: PerformanceMetrics[];

  constructor(
    db: CtxDatabase,
    embeddings: EmbeddingProvider,
    reranker: CrossEncoderReranker,
    config: Config,
    budgetTracker?: BudgetTracker
  ) {
    this.db = db;
    this.embeddings = embeddings;
    this.reranker = reranker;
    this.dfIdfBuilder = new DfIdfBuilder(db);
    this.config = config;
    this.budgetTracker = budgetTracker || this.initDefaultBudgetTracker();
    this.performanceMetrics = [];
  }

  // Enhanced main retrieval pipeline with configurable variants
  async search(
    sessionId: string,
    queries: string[],
    topK: number = 50
  ): Promise<Result<EnhancedCandidate[], LetheError>> {
    const startTime = performance.now();
    const memoryBefore = this.getMemoryUsage();

    try {
      let candidates: EnhancedCandidate[] = [];

      switch (this.config.retrieval.variant) {
        case 'window':
          candidates = await this.windowBasedRetrieval(sessionId, queries, topK);
          break;
        case 'bm25':
          candidates = await this.bm25OnlyRetrieval(sessionId, queries, topK);
          break;
        case 'vector':
          candidates = await this.vectorOnlyRetrieval(sessionId, queries, topK);
          break;
        case 'hybrid':
        default:
          candidates = await this.hybridRetrieval(sessionId, queries, topK);
          break;
      }

      const endTime = performance.now();
      const memoryAfter = this.getMemoryUsage();

      // Track performance metrics
      this.recordPerformanceMetrics(
        startTime,
        endTime,
        memoryBefore,
        memoryAfter,
        candidates.length
      );

      return { success: true, data: candidates };

    } catch (error) {
      return {
        success: false,
        error: {
          code: 'RETRIEVAL_ERROR',
          message: `Retrieval failed: ${error.message}`,
          timestamp: Date.now(),
          details: { 
            variant: this.config.retrieval.variant,
            sessionId, 
            queries: queries.length,
            topK 
          }
        }
      };
    }
  }

  // Window-based retrieval for local context
  private async windowBasedRetrieval(
    sessionId: string,
    queries: string[],
    topK: number
  ): Promise<EnhancedCandidate[]> {
    const windowSize = this.config.retrieval.window_size || 5;
    const candidates = new Map<string, EnhancedCandidate>();
    
    for (const query of queries) {
      const ftsResults = this.db.searchChunks(query, sessionId);
      
      // For each FTS result, collect window of surrounding chunks
      for (const result of ftsResults.slice(0, topK)) {
        const windowChunks = await this.getChunkWindow(
          result.chunkId, 
          sessionId, 
          windowSize
        );
        
        for (const chunk of windowChunks) {
          const score = this.calculateWindowScore(chunk, result, query);
          
          if (!candidates.has(chunk.id) || (candidates.get(chunk.id)?.hybridScore || 0) < score) {
            candidates.set(chunk.id, {
              id: chunk.id,
              text: chunk.text,
              messageId: chunk.messageId,
              kind: chunk.kind,
              bm25Score: score * 0.6,
              vectorScore: score * 0.4,
              hybridScore: score,
              metadata: {
                entities: this.extractEntities(chunk.text),
                citation_spans: [{ start: 0, end: chunk.text.length }]
              }
            });
          }
        }
      }
    }
    
    return Array.from(candidates.values())
      .sort((a, b) => (b.hybridScore || 0) - (a.hybridScore || 0))
      .slice(0, topK);
  }

  // BM25-only retrieval
  private async bm25OnlyRetrieval(
    sessionId: string,
    queries: string[],
    topK: number
  ): Promise<EnhancedCandidate[]> {
    const bm25Results = await this.bm25Search(queries, sessionId, topK * 2);
    return this.convertToCandidates(bm25Results, sessionId, 'bm25');
  }

  // Vector-only retrieval
  private async vectorOnlyRetrieval(
    sessionId: string,
    queries: string[],
    topK: number
  ): Promise<EnhancedCandidate[]> {
    const vectorResults = await this.vectorSearch(queries, topK * 2);
    return this.convertToCandidates(vectorResults, sessionId, 'vector');
  }

  // Enhanced hybrid retrieval with dynamic weighting
  private async hybridRetrieval(
    sessionId: string,
    queries: string[],
    topK: number
  ): Promise<EnhancedCandidate[]> {
    // Get BM25 results for all queries
    const bm25Results = await this.bm25Search(queries, sessionId, topK * 2);
    
    // Get vector results for all queries
    const vectorResults = await this.vectorSearch(queries, topK * 2);
    
    // Enhanced hybrid scoring with adaptive weights
    const candidates = await this.adaptiveHybridScore(
      bm25Results, 
      vectorResults, 
      sessionId,
      queries
    );
    
    return candidates.slice(0, topK);
  }

  // Enhanced BM25 search with per-session DF/IDF
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
      
      // Calculate precise BM25 scores for candidates with per-session DF/IDF
      const scores = await this.dfIdfBuilder.calculateBM25Scores(
        sessionId,
        query,
        candidateIds
      );
      
      // Merge scores using dynamic fusion strategy
      for (const [chunkId, score] of scores) {
        const currentScore = allScores.get(chunkId) || 0;
        // Use max fusion for now, could be configurable
        allScores.set(chunkId, Math.max(currentScore, score));
      }
    }
    
    this.updateBudgetTracker(allScores.size, 'bm25');
    return allScores;
  }

  // Enhanced vector semantic search with optimized similarity
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
      
      // Calculate optimized cosine similarity with all chunks
      for (const embedding of chunkEmbeddings) {
        const similarity = this.optimizedCosineSimilarity(queryVec, embedding.vec);
        similarities.push({ chunkId: embedding.chunkId, score: similarity });
      }
      
      // Sort and take top-k
      similarities.sort((a, b) => b.score - a.score);
      const topSimilarities = similarities.slice(0, k);
      
      // Merge scores using configured fusion strategy
      for (const { chunkId, score } of topSimilarities) {
        const currentScore = allScores.get(chunkId) || 0;
        allScores.set(chunkId, Math.max(currentScore, score));
      }
    }
    
    this.updateBudgetTracker(chunkEmbeddings.length * queries.length, 'vector');
    return allScores;
  }

  // Adaptive hybrid scoring with query-aware weighting
  private async adaptiveHybridScore(
    bm25Scores: Map<string, number>,
    vectorScores: Map<string, number>,
    sessionId: string,
    queries: string[]
  ): Promise<EnhancedCandidate[]> {
    const candidates: EnhancedCandidate[] = [];
    const allChunkIds = new Set([
      ...bm25Scores.keys(),
      ...vectorScores.keys()
    ]);
    
    const chunks = this.db.getChunksByIds([...allChunkIds]);
    const chunkMap = new Map(chunks.map(c => [c.id, c]));
    
    // Adaptive weight calculation based on query characteristics
    const { adaptiveAlpha, adaptiveBeta } = this.calculateAdaptiveWeights(queries);
    
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
      
      // Enhanced hybrid formula with adaptive weights
      const hybridScore = 
        adaptiveAlpha * bm25Score +
        adaptiveBeta * vectorScore +
        kindBoost;
      
      candidates.push({
        id: chunkId,
        text: chunk.text,
        messageId: chunk.messageId,
        kind: chunk.kind,
        bm25Score,
        vectorScore,
        hybridScore,
        metadata: {
          entities: this.extractEntities(chunk.text),
          citation_spans: [{ start: 0, end: chunk.text.length }]
        }
      });
    }
    
    // Sort by hybrid score
    candidates.sort((a, b) => (b.hybridScore || 0) - (a.hybridScore || 0));
    return candidates;
  }

  // Enhanced reranking with LLM integration support
  async rerank(
    query: string,
    candidates: EnhancedCandidate[],
    topkOut: number
  ): Promise<Result<EnhancedCandidate[], LetheError>> {
    try {
      if (candidates.length === 0) {
        return { success: true, data: [] };
      }
      
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
        .map(c => ({ 
          ...c, 
          rerankScore: rerankMap.get(c.id)!,
          hybridScore: (c.hybridScore || 0) * 0.7 + (rerankMap.get(c.id)! * 0.3)
        }))
        .sort((a, b) => (b.hybridScore || 0) - (a.hybridScore || 0));
      
      return { success: true, data: rerankedCandidates };

    } catch (error) {
      return {
        success: false,
        error: {
          code: 'RERANK_ERROR',
          message: `Reranking failed: ${error.message}`,
          timestamp: Date.now(),
          details: { query, candidateCount: candidates.length, topkOut }
        }
      };
    }
  }

  // Enhanced submodular selection with improved diversity metrics
  submodularSelect(
    candidates: EnhancedCandidate[], 
    maxChunks: number
  ): Result<EnhancedCandidate[], LetheError> {
    try {
      if (candidates.length <= maxChunks) {
        return { success: true, data: candidates };
      }
      
      const selected: EnhancedCandidate[] = [];
      const remaining = [...candidates];
      const selectedEntities = new Set<string>();
      const selectedClusters = new Set<number>();
      
      for (let i = 0; i < maxChunks && remaining.length > 0; i++) {
        let bestIdx = 0;
        let bestScore = -Infinity;
        
        for (let j = 0; j < remaining.length; j++) {
          const candidate = remaining[j];
          const entities = candidate.metadata?.entities || this.extractEntities(candidate.text);
          
          // Count new entities and clusters this candidate would add
          const newEntities = entities.filter(e => !selectedEntities.has(e));
          const clusterDiversity = candidate.metadata?.semantic_cluster !== undefined && 
            !selectedClusters.has(candidate.metadata.semantic_cluster) ? 1 : 0;
          
          // Enhanced submodular score with multiple diversity factors
          const diversityScore = 
            newEntities.length * 0.1 + 
            clusterDiversity * 0.15;
          
          const relevanceScore = candidate.rerankScore || candidate.hybridScore || 0;
          const submodularScore = relevanceScore + diversityScore;
          
          if (submodularScore > bestScore) {
            bestScore = submodularScore;
            bestIdx = j;
          }
        }
        
        // Add best candidate
        const selectedCandidate = remaining.splice(bestIdx, 1)[0];
        selected.push(selectedCandidate);
        
        // Update diversity tracking
        const entities = selectedCandidate.metadata?.entities || [];
        entities.forEach(e => selectedEntities.add(e));
        
        if (selectedCandidate.metadata?.semantic_cluster !== undefined) {
          selectedClusters.add(selectedCandidate.metadata.semantic_cluster);
        }
      }
      
      return { success: true, data: selected };

    } catch (error) {
      return {
        success: false,
        error: {
          code: 'SELECTION_ERROR',
          message: `Submodular selection failed: ${error.message}`,
          timestamp: Date.now(),
          details: { candidateCount: candidates.length, maxChunks }
        }
      };
    }
  }

  // Private helper methods

  private calculateAdaptiveWeights(queries: string[]): { adaptiveAlpha: number; adaptiveBeta: number } {
    // Analyze query characteristics to determine optimal BM25 vs Vector weights
    let lexicalSignal = 0;
    let semanticSignal = 0;
    
    for (const query of queries) {
      // Heuristics for query type detection
      const hasSpecificTerms = /\b(function|class|error|code|fix|bug)\b/i.test(query);
      const hasQuestionWords = /\b(what|how|why|when|where|which)\b/i.test(query);
      const hasCodeSyntax = /[{}()\[\];]/.test(query);
      
      if (hasSpecificTerms || hasCodeSyntax) {
        lexicalSignal += 1;
      }
      
      if (hasQuestionWords || query.length > 50) {
        semanticSignal += 1;
      }
    }
    
    const totalSignal = lexicalSignal + semanticSignal || 1;
    const lexicalRatio = lexicalSignal / totalSignal;
    const semanticRatio = semanticSignal / totalSignal;
    
    // Adaptive weights based on query characteristics
    const baseAlpha = this.config.retrieval.alpha;
    const baseBeta = this.config.retrieval.beta;
    
    const adaptiveAlpha = baseAlpha * (0.5 + lexicalRatio * 0.5);
    const adaptiveBeta = baseBeta * (0.5 + semanticRatio * 0.5);
    
    return { adaptiveAlpha, adaptiveBeta };
  }

  private async getChunkWindow(
    centralChunkId: string,
    sessionId: string,
    windowSize: number
  ): Promise<Chunk[]> {
    // Get surrounding chunks in the same conversation
    const centralChunk = this.db.getChunksByIds([centralChunkId])[0];
    if (!centralChunk) return [];
    
    const allChunks = this.db.getChunks(sessionId);
    const centralIndex = allChunks.findIndex(c => c.id === centralChunkId);
    
    if (centralIndex === -1) return [centralChunk];
    
    const start = Math.max(0, centralIndex - windowSize);
    const end = Math.min(allChunks.length, centralIndex + windowSize + 1);
    
    return allChunks.slice(start, end);
  }

  private calculateWindowScore(chunk: Chunk, ftsResult: any, query: string): number {
    // Score based on distance from central chunk and FTS relevance
    const baseScore = ftsResult.score || 0.5;
    const proximityBonus = 0.1; // Bonus for being in window
    
    return Math.min(1.0, baseScore + proximityBonus);
  }

  private convertToCandidates(
    scores: Map<string, number>,
    sessionId: string,
    scoreType: 'bm25' | 'vector'
  ): EnhancedCandidate[] {
    const chunks = this.db.getChunksByIds([...scores.keys()]);
    const chunkMap = new Map(chunks.map(c => [c.id, c]));
    
    const candidates: EnhancedCandidate[] = [];
    
    for (const [chunkId, score] of scores.entries()) {
      const chunk = chunkMap.get(chunkId);
      if (!chunk) continue;
      
      candidates.push({
        id: chunkId,
        text: chunk.text,
        messageId: chunk.messageId,
        kind: chunk.kind,
        bm25Score: scoreType === 'bm25' ? score : 0,
        vectorScore: scoreType === 'vector' ? score : 0,
        hybridScore: score,
        metadata: {
          entities: this.extractEntities(chunk.text),
          citation_spans: [{ start: 0, end: chunk.text.length }]
        }
      });
    }
    
    return candidates.sort((a, b) => (b.hybridScore || 0) - (a.hybridScore || 0));
  }

  private extractEntities(text: string): string[] {
    const entities: string[] = [];
    
    // Enhanced entity extraction
    const capitalizedWords = text.match(/\b[A-Z][a-z]+\b/g) || [];
    entities.push(...capitalizedWords);
    
    const codeIdentifiers = text.match(/\b[a-z]+[A-Z][a-zA-Z]*\b|[A-Z][a-zA-Z]*|[a-z_]+[a-z_]*[a-z]\b/g) || [];
    entities.push(...codeIdentifiers);
    
    const filePaths = text.match(/[\w-]+\.[a-z]{2,4}\b|\b[\w-]+\/[\w.-]+/g) || [];
    entities.push(...filePaths);
    
    const urls = text.match(/https?:\/\/[\w.-]+|[\w.-]+\.[a-z]{2,}/g) || [];
    entities.push(...urls);
    
    const functionCalls = text.match(/\b\w+\([^)]*\)|\b\w+\(\)/g) || [];
    entities.push(...functionCalls);

    return [...new Set(entities)];
  }

  private optimizedCosineSimilarity(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) {
      throw new Error('Vectors must have same dimension');
    }
    
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    // Vectorized computation
    for (let i = 0; i < a.length; i++) {
      const aVal = a[i];
      const bVal = b[i];
      dotProduct += aVal * bVal;
      normA += aVal * aVal;
      normB += bVal * bVal;
    }
    
    if (normA === 0 || normB === 0) return 0;
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  private initDefaultBudgetTracker(): BudgetTracker {
    return {
      params_baseline: 1000000,
      flops_baseline: 10000000,
      current_params: 0,
      current_flops: 0,
      within_budget: true,
      variance_percent: 0
    };
  }

  private updateBudgetTracker(operationCount: number, operationType: string): void {
    const estimatedFlops = operationType === 'vector' ? 
      operationCount * 1000 : operationCount * 100;
    
    this.budgetTracker.current_flops += estimatedFlops;
    
    const variance = Math.abs(this.budgetTracker.current_flops - this.budgetTracker.flops_baseline) / 
      this.budgetTracker.flops_baseline;
    
    this.budgetTracker.within_budget = variance <= 0.05; // ±5% budget parity
    this.budgetTracker.variance_percent = variance * 100;
  }

  private getMemoryUsage(): number {
    // Simplified memory usage calculation
    if (typeof process !== 'undefined' && process.memoryUsage) {
      return process.memoryUsage().rss / 1024 / 1024; // MB
    }
    return 0;
  }

  private recordPerformanceMetrics(
    startTime: number,
    endTime: number,
    memoryBefore: number,
    memoryAfter: number,
    resultCount: number
  ): void {
    const latency = endTime - startTime;
    const memoryUsed = memoryAfter - memoryBefore;
    
    this.performanceMetrics.push({
      latency_p50: latency,
      latency_p95: latency, // Would be calculated from historical data
      memory_rss_mb: memoryAfter,
      cpu_usage_percent: 0, // Would be measured if available
      timestamp: Date.now()
    });
    
    // Keep only recent metrics (last 100 measurements)
    if (this.performanceMetrics.length > 100) {
      this.performanceMetrics = this.performanceMetrics.slice(-100);
    }
  }

  // Public performance monitoring methods
  getPerformanceMetrics(): PerformanceMetrics[] {
    return [...this.performanceMetrics];
  }

  getBudgetStatus(): BudgetTracker {
    return { ...this.budgetTracker };
  }

  // Indexing operations (enhanced)
  async ensureEmbeddings(sessionId: string): Promise<Result<void, LetheError>> {
    try {
      const missingChunkIds = this.db.getChunksWithoutEmbeddings(sessionId);
      
      if (missingChunkIds.length === 0) {
        return { success: true, data: undefined };
      }
      
      console.log(`Generating embeddings for ${missingChunkIds.length} chunks...`);
      
      const chunks = this.db.getChunksByIds(missingChunkIds);
      const texts = chunks.map(c => c.text);
      
      // Generate embeddings in batches to manage memory
      const batchSize = 32;
      const embeddingData: Embedding[] = [];
      
      for (let i = 0; i < texts.length; i += batchSize) {
        const batchTexts = texts.slice(i, i + batchSize);
        const batchEmbeddings = await this.embeddings.embed(batchTexts);
        
        const batchData = chunks.slice(i, i + batchSize).map((chunk, idx) => ({
          chunkId: chunk.id,
          dim: this.embeddings.dim,
          vec: batchEmbeddings[idx]
        }));
        
        embeddingData.push(...batchData);
      }
      
      this.db.insertEmbeddings(embeddingData);
      console.log(`Embeddings generated and stored for ${embeddingData.length} chunks`);
      
      return { success: true, data: undefined };

    } catch (error) {
      return {
        success: false,
        error: {
          code: 'EMBEDDING_ERROR',
          message: `Failed to ensure embeddings: ${error.message}`,
          timestamp: Date.now(),
          details: { sessionId }
        }
      };
    }
  }

  async rebuildDfIdf(sessionId: string): Promise<Result<void, LetheError>> {
    try {
      await this.dfIdfBuilder.rebuild(sessionId);
      return { success: true, data: undefined };
    } catch (error) {
      return {
        success: false,
        error: {
          code: 'DFIDF_REBUILD_ERROR',
          message: `Failed to rebuild DF/IDF: ${error.message}`,
          timestamp: Date.now(),
          details: { sessionId }
        }
      };
    }
  }
}