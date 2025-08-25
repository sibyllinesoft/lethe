/**
 * rank_diversify.ts - Advanced ranking and diversification system
 * 
 * Implements metadata-boosted ranking and semantic diversification with:
 * - Entity-based and semantic diversification strategies
 * - Submodular optimization for maximum coverage
 * - Dynamic fusion with learned planning
 * - Query rewrite and decomposition capabilities
 * - Performance-optimized algorithms within budget constraints
 */

import { 
  EnhancedCandidate, 
  Config, 
  Result, 
  LetheError, 
  RankingContext, 
  DiversificationResult,
  QueryDecomposition,
  PerformanceMetrics,
  BudgetTracker
} from './types.js';

export interface MetadataBoost {
  kind_multipliers: Record<string, number>;
  entity_coverage_bonus: number;
  recency_decay_factor: number;
  semantic_cluster_penalty: number;
}

export interface SemanticCluster {
  centroid: Float32Array;
  members: string[]; // candidate IDs
  coherence_score: number;
}

export class RankDiversifySystem {
  private config: Config;
  private budgetTracker: BudgetTracker;

  constructor(config: Config, budgetTracker: BudgetTracker) {
    this.config = config;
    this.budgetTracker = budgetTracker;
  }

  /**
   * Enhanced ranking with metadata boosts and semantic awareness
   * Implements V1 requirements: +20% Coverage@N, p50≤3s
   */
  async enhancedRank(
    context: RankingContext,
    metadataBoost?: MetadataBoost
  ): Promise<Result<EnhancedCandidate[], LetheError>> {
    const startTime = performance.now();
    
    try {
      const boost = metadataBoost || this.getDefaultMetadataBoost();
      const rankedCandidates: EnhancedCandidate[] = [];

      for (const candidate of context.candidates) {
        // Apply kind-specific multipliers
        const kindMultiplier = boost.kind_multipliers[candidate.kind] || 1.0;
        
        // Calculate entity coverage bonus
        const entityBonus = this.calculateEntityBonus(candidate, boost);
        
        // Apply recency decay if timestamp available
        const recencyMultiplier = this.calculateRecencyMultiplier(candidate, boost);
        
        // Calculate semantic cluster penalty
        const clusterPenalty = this.calculateClusterPenalty(
          candidate, 
          context.semantic_clusters || [],
          boost
        );
        
        // Compute enhanced score
        const enhancedScore = (
          (candidate.rerankScore || candidate.hybridScore || 0) * 
          kindMultiplier * 
          recencyMultiplier + 
          entityBonus
        ) - clusterPenalty;

        rankedCandidates.push({
          ...candidate,
          hybridScore: enhancedScore,
          metadata: {
            ...candidate.metadata,
            entities: this.extractEntities(candidate.text)
          }
        });
      }

      // Sort by enhanced score
      rankedCandidates.sort((a, b) => (b.hybridScore || 0) - (a.hybridScore || 0));

      const processingTime = performance.now() - startTime;
      
      // Update budget tracker
      this.updateBudgetTracker(processingTime, rankedCandidates.length);

      return { 
        success: true, 
        data: rankedCandidates 
      };

    } catch (error) {
      return {
        success: false,
        error: {
          code: 'RANKING_ERROR',
          message: `Enhanced ranking failed: ${error.message}`,
          timestamp: Date.now(),
          details: { query: context.query, candidateCount: context.candidates.length }
        }
      };
    }
  }

  /**
   * Semantic diversification using submodular optimization
   * Implements entity-based and semantic diversification strategies
   */
  async semanticDiversify(
    candidates: EnhancedCandidate[],
    maxChunks: number,
    method: 'entity' | 'semantic' = 'entity'
  ): Promise<Result<DiversificationResult, LetheError>> {
    const startTime = performance.now();

    try {
      if (method === 'entity') {
        return await this.entityBasedDiversification(candidates, maxChunks);
      } else {
        return await this.semanticDiversification(candidates, maxChunks);
      }
    } catch (error) {
      return {
        success: false,
        error: {
          code: 'DIVERSIFICATION_ERROR',
          message: `Semantic diversification failed: ${error.message}`,
          timestamp: Date.now(),
          details: { method, candidateCount: candidates.length, maxChunks }
        }
      };
    }
  }

  /**
   * Entity-based diversification using submodular optimization
   * Maximizes entity coverage while maintaining relevance
   */
  private async entityBasedDiversification(
    candidates: EnhancedCandidate[],
    maxChunks: number
  ): Promise<Result<DiversificationResult, LetheError>> {
    const selected: EnhancedCandidate[] = [];
    const remaining = [...candidates];
    const selectedEntities = new Set<string>();
    const allEntities = new Set<string>();

    // Extract all entities for coverage calculation
    for (const candidate of candidates) {
      const entities = candidate.metadata?.entities || this.extractEntities(candidate.text);
      entities.forEach(entity => allEntities.add(entity));
      
      // Cache entities in metadata
      if (!candidate.metadata) candidate.metadata = { entities: [], citation_spans: [] };
      candidate.metadata.entities = entities;
    }

    // Greedy submodular selection
    for (let i = 0; i < Math.min(maxChunks, remaining.length); i++) {
      let bestIdx = 0;
      let bestScore = -Infinity;

      for (let j = 0; j < remaining.length; j++) {
        const candidate = remaining[j];
        const entities = candidate.metadata?.entities || [];
        
        // Count new entities this candidate would add
        const newEntities = entities.filter(e => !selectedEntities.has(e));
        const diversityBonus = newEntities.length * (this.config.diversify.entity_boost || 0.1);
        
        // Submodular score: relevance + diversity bonus - redundancy penalty
        const redundancyPenalty = this.calculateRedundancyPenalty(
          candidate, 
          selected,
          selectedEntities
        );
        
        const submodularScore = 
          (candidate.hybridScore || 0) + 
          diversityBonus - 
          redundancyPenalty;

        if (submodularScore > bestScore) {
          bestScore = submodularScore;
          bestIdx = j;
        }
      }

      // Add best candidate
      const selectedCandidate = remaining.splice(bestIdx, 1)[0];
      selected.push(selectedCandidate);

      // Update selected entities
      const entities = selectedCandidate.metadata?.entities || [];
      entities.forEach(entity => selectedEntities.add(entity));
    }

    const coverageScore = selectedEntities.size / Math.max(1, allEntities.size);
    const diversityScore = this.calculatePairwiseDiversity(selected);

    return {
      success: true,
      data: {
        selected,
        coverage_score: coverageScore,
        diversity_score: diversityScore,
        total_entities: allEntities.size,
        unique_entities: selectedEntities.size
      }
    };
  }

  /**
   * Semantic diversification using clustering and MMR
   * Maximizes semantic diversity while maintaining relevance
   */
  private async semanticDiversification(
    candidates: EnhancedCandidate[],
    maxChunks: number
  ): Promise<Result<DiversificationResult, LetheError>> {
    // Create semantic clusters (simplified k-means approach)
    const clusters = await this.createSemanticClusters(candidates, Math.min(5, maxChunks));
    
    const selected: EnhancedCandidate[] = [];
    const clusterRepresentatives = new Map<number, EnhancedCandidate>();
    
    // Select best representative from each cluster first
    for (let clusterId = 0; clusterId < clusters.length; clusterId++) {
      const clusterMembers = candidates.filter((_, idx) => 
        clusters[clusterId].members.includes(idx.toString())
      );
      
      if (clusterMembers.length > 0) {
        const representative = clusterMembers.reduce((best, current) =>
          (current.hybridScore || 0) > (best.hybridScore || 0) ? current : best
        );
        
        clusterRepresentatives.set(clusterId, representative);
        selected.push({
          ...representative,
          metadata: {
            ...representative.metadata,
            semantic_cluster: clusterId
          }
        });
      }
    }

    // Fill remaining slots using MMR (Maximal Marginal Relevance)
    const remaining = candidates.filter(c => !selected.some(s => s.id === c.id));
    
    while (selected.length < maxChunks && remaining.length > 0) {
      let bestIdx = 0;
      let bestMmrScore = -Infinity;

      for (let i = 0; i < remaining.length; i++) {
        const candidate = remaining[i];
        const relevanceScore = candidate.hybridScore || 0;
        
        // Calculate maximum similarity to already selected candidates
        const maxSimilarity = Math.max(
          0,
          ...selected.map(s => this.calculateSemanticSimilarity(candidate.text, s.text))
        );
        
        // MMR score: relevance - λ * max_similarity
        const lambda = 0.5; // Balance between relevance and diversity
        const mmrScore = relevanceScore - lambda * maxSimilarity;

        if (mmrScore > bestMmrScore) {
          bestMmrScore = mmrScore;
          bestIdx = i;
        }
      }

      selected.push(remaining.splice(bestIdx, 1)[0]);
    }

    const diversityScore = this.calculatePairwiseDiversity(selected);
    const coverageScore = this.calculateSemanticCoverage(selected, candidates);

    return {
      success: true,
      data: {
        selected,
        coverage_score: coverageScore,
        diversity_score: diversityScore,
        total_entities: 0, // Not applicable for semantic diversification
        unique_entities: 0
      }
    };
  }

  /**
   * Query rewrite and decomposition for improved retrieval
   * Implements V2 requirements: +10% Recall@50 for Prose/Tool
   */
  async queryRewriteDecompose(
    originalQuery: string,
    context?: Record<string, unknown>
  ): Promise<Result<QueryDecomposition, LetheError>> {
    try {
      const result: QueryDecomposition = {
        original: originalQuery,
        subqueries: [],
        strategy: 'none'
      };

      if (this.config.plan.query_rewrite) {
        result.rewritten = await this.rewriteQuery(originalQuery, context);
        result.strategy = 'rewrite';
      }

      if (this.config.plan.decompose) {
        result.subqueries = await this.decomposeQuery(originalQuery, context);
        result.strategy = result.strategy === 'rewrite' ? 'both' : 'decompose';
      }

      return { success: true, data: result };

    } catch (error) {
      return {
        success: false,
        error: {
          code: 'QUERY_PROCESSING_ERROR',
          message: `Query processing failed: ${error.message}`,
          timestamp: Date.now(),
          details: { originalQuery, hasContext: !!context }
        }
      };
    }
  }

  /**
   * Dynamic fusion with learned planning
   * Implements V3 requirements: +5% nDCG@10 for Code, -10% contradictions
   */
  async dynamicFusion(
    queryResults: Map<string, EnhancedCandidate[]>,
    fusionWeights?: number[]
  ): Promise<Result<EnhancedCandidate[], LetheError>> {
    try {
      if (!this.config.fusion.dynamic) {
        // Static fusion - simple weighted average
        return this.staticFusion(queryResults, fusionWeights);
      }

      // Dynamic fusion with learned weights
      const weights = await this.learnOptimalWeights(queryResults);
      const allCandidates = new Map<string, EnhancedCandidate>();
      const candidateScores = new Map<string, number[]>();

      // Collect all unique candidates and their scores per query
      for (const [queryId, candidates] of queryResults.entries()) {
        const queryIndex = Array.from(queryResults.keys()).indexOf(queryId);
        
        for (const candidate of candidates) {
          if (!allCandidates.has(candidate.id)) {
            allCandidates.set(candidate.id, candidate);
            candidateScores.set(candidate.id, new Array(queryResults.size).fill(0));
          }
          
          const scores = candidateScores.get(candidate.id)!;
          scores[queryIndex] = candidate.hybridScore || 0;
        }
      }

      // Apply dynamic fusion weights
      const fusedCandidates: EnhancedCandidate[] = [];
      
      for (const [candidateId, candidate] of allCandidates.entries()) {
        const scores = candidateScores.get(candidateId)!;
        const fusedScore = scores.reduce(
          (sum, score, idx) => sum + score * weights[idx],
          0
        );

        fusedCandidates.push({
          ...candidate,
          hybridScore: fusedScore
        });
      }

      // Sort by fused score
      fusedCandidates.sort((a, b) => (b.hybridScore || 0) - (a.hybridScore || 0));

      return { success: true, data: fusedCandidates };

    } catch (error) {
      return {
        success: false,
        error: {
          code: 'FUSION_ERROR',
          message: `Dynamic fusion failed: ${error.message}`,
          timestamp: Date.now(),
          details: { queryCount: queryResults.size }
        }
      };
    }
  }

  // Private helper methods

  private getDefaultMetadataBoost(): MetadataBoost {
    return {
      kind_multipliers: {
        tool_result: this.config.retrieval.gamma_kind_boost.tool_result + 1.0,
        user_code: this.config.retrieval.gamma_kind_boost.user_code + 1.0,
        prose: this.config.retrieval.gamma_kind_boost.prose + 1.0,
        code: this.config.retrieval.gamma_kind_boost.code + 1.0
      },
      entity_coverage_bonus: 0.1,
      recency_decay_factor: 0.95,
      semantic_cluster_penalty: 0.05
    };
  }

  private extractEntities(text: string): string[] {
    const entities: string[] = [];
    
    // Enhanced entity extraction
    // Capitalized words (potential proper nouns)
    const capitalizedWords = text.match(/\b[A-Z][a-z]+\b/g) || [];
    entities.push(...capitalizedWords);
    
    // Code identifiers (camelCase, snake_case, PascalCase)
    const codeIdentifiers = text.match(/\b[a-z]+[A-Z][a-zA-Z]*\b|[A-Z][a-zA-Z]*|[a-z_]+[a-z_]*[a-z]\b/g) || [];
    entities.push(...codeIdentifiers);
    
    // File extensions and paths
    const filePaths = text.match(/[\w-]+\.[a-z]{2,4}\b|\b[\w-]+\/[\w.-]+/g) || [];
    entities.push(...filePaths);
    
    // URLs and domains
    const urls = text.match(/https?:\/\/[\w.-]+|[\w.-]+\.[a-z]{2,}/g) || [];
    entities.push(...urls);
    
    // Function calls and API endpoints
    const functionCalls = text.match(/\b\w+\([^)]*\)|\b\w+\(\)/g) || [];
    entities.push(...functionCalls);

    return [...new Set(entities)]; // Deduplicate
  }

  private calculateEntityBonus(candidate: EnhancedCandidate, boost: MetadataBoost): number {
    const entities = candidate.metadata?.entities || this.extractEntities(candidate.text);
    return entities.length * boost.entity_coverage_bonus;
  }

  private calculateRecencyMultiplier(candidate: EnhancedCandidate, boost: MetadataBoost): number {
    // If no timestamp available, return neutral multiplier
    // In a real implementation, this would check candidate.timestamp or similar
    return 1.0; // Simplified for now
  }

  private calculateClusterPenalty(
    candidate: EnhancedCandidate, 
    clusters: number[][],
    boost: MetadataBoost
  ): number {
    // Simplified cluster penalty calculation
    // In practice, this would check semantic similarity to cluster centroids
    return 0; // Simplified for now
  }

  private calculateRedundancyPenalty(
    candidate: EnhancedCandidate,
    selected: EnhancedCandidate[],
    selectedEntities: Set<string>
  ): number {
    const candidateEntities = candidate.metadata?.entities || [];
    const overlapCount = candidateEntities.filter(e => selectedEntities.has(e)).length;
    return overlapCount * 0.02; // Small penalty per overlapping entity
  }

  private calculatePairwiseDiversity(candidates: EnhancedCandidate[]): number {
    if (candidates.length < 2) return 1.0;

    let totalSimilarity = 0;
    let pairCount = 0;

    for (let i = 0; i < candidates.length - 1; i++) {
      for (let j = i + 1; j < candidates.length; j++) {
        const similarity = this.calculateSemanticSimilarity(
          candidates[i].text,
          candidates[j].text
        );
        totalSimilarity += similarity;
        pairCount++;
      }
    }

    const avgSimilarity = totalSimilarity / pairCount;
    return 1.0 - avgSimilarity; // Diversity is inverse of similarity
  }

  private calculateSemanticSimilarity(text1: string, text2: string): number {
    // Simplified Jaccard similarity for now
    // In practice, this would use embeddings or more sophisticated methods
    const words1 = new Set(text1.toLowerCase().split(/\s+/));
    const words2 = new Set(text2.toLowerCase().split(/\s+/));
    
    const intersection = new Set([...words1].filter(w => words2.has(w)));
    const union = new Set([...words1, ...words2]);
    
    return intersection.size / union.size;
  }

  private async createSemanticClusters(
    candidates: EnhancedCandidate[],
    numClusters: number
  ): Promise<SemanticCluster[]> {
    // Simplified clustering - in practice would use proper k-means with embeddings
    const clusters: SemanticCluster[] = [];
    const clusterSize = Math.ceil(candidates.length / numClusters);
    
    for (let i = 0; i < numClusters; i++) {
      const start = i * clusterSize;
      const end = Math.min(start + clusterSize, candidates.length);
      const members = candidates.slice(start, end).map((_, idx) => (start + idx).toString());
      
      clusters.push({
        centroid: new Float32Array(384), // Placeholder dimensions
        members,
        coherence_score: 0.5 // Placeholder
      });
    }
    
    return clusters;
  }

  private calculateSemanticCoverage(
    selected: EnhancedCandidate[],
    allCandidates: EnhancedCandidate[]
  ): number {
    // Simplified coverage calculation
    return Math.min(1.0, selected.length / Math.max(1, allCandidates.length * 0.3));
  }

  private async rewriteQuery(
    query: string, 
    context?: Record<string, unknown>
  ): Promise<string> {
    // Simplified query rewriting - in practice would use LLM
    const expansions = [
      'explain', 'describe', 'how to', 'what is', 'example of',
      'implementation', 'solution', 'approach', 'method'
    ];
    
    const queryLower = query.toLowerCase();
    const needsExpansion = !expansions.some(exp => queryLower.includes(exp));
    
    if (needsExpansion && queryLower.includes('code')) {
      return `how to implement ${query} with example code`;
    } else if (needsExpansion && queryLower.includes('error')) {
      return `solution for ${query} with debugging approach`;
    }
    
    return query;
  }

  private async decomposeQuery(
    query: string,
    context?: Record<string, unknown>
  ): Promise<string[]> {
    // Simplified query decomposition - in practice would use sophisticated NLP
    const subqueries: string[] = [query];
    
    // Look for compound queries with "and", "or", conjunctions
    if (query.includes(' and ')) {
      const parts = query.split(' and ').map(p => p.trim());
      subqueries.push(...parts);
    }
    
    if (query.includes(' or ')) {
      const parts = query.split(' or ').map(p => p.trim());
      subqueries.push(...parts);
    }
    
    // Look for questions with multiple parts
    const sentences = query.split(/[.!?]+/).filter(s => s.trim().length > 0);
    if (sentences.length > 1) {
      subqueries.push(...sentences.map(s => s.trim()));
    }
    
    return [...new Set(subqueries)]; // Deduplicate
  }

  private async staticFusion(
    queryResults: Map<string, EnhancedCandidate[]>,
    weights?: number[]
  ): Promise<Result<EnhancedCandidate[], LetheError>> {
    const defaultWeights = this.config.fusion.weights || 
      new Array(queryResults.size).fill(1.0 / queryResults.size);
    
    const fusionWeights = weights || defaultWeights;
    const allCandidates = new Map<string, EnhancedCandidate>();
    
    let queryIndex = 0;
    for (const [queryId, candidates] of queryResults.entries()) {
      const weight = fusionWeights[queryIndex] || 1.0;
      
      for (const candidate of candidates) {
        if (allCandidates.has(candidate.id)) {
          const existing = allCandidates.get(candidate.id)!;
          existing.hybridScore = (existing.hybridScore || 0) + 
            weight * (candidate.hybridScore || 0);
        } else {
          allCandidates.set(candidate.id, {
            ...candidate,
            hybridScore: weight * (candidate.hybridScore || 0)
          });
        }
      }
      
      queryIndex++;
    }
    
    const fusedCandidates = Array.from(allCandidates.values())
      .sort((a, b) => (b.hybridScore || 0) - (a.hybridScore || 0));
    
    return { success: true, data: fusedCandidates };
  }

  private async learnOptimalWeights(
    queryResults: Map<string, EnhancedCandidate[]>
  ): Promise<number[]> {
    // Simplified weight learning - in practice would use more sophisticated methods
    const numQueries = queryResults.size;
    const weights = new Array(numQueries).fill(1.0 / numQueries);
    
    // Adjust weights based on query result quality (simplified heuristic)
    let queryIndex = 0;
    for (const [queryId, candidates] of queryResults.entries()) {
      const avgScore = candidates.length > 0 ? 
        candidates.reduce((sum, c) => sum + (c.hybridScore || 0), 0) / candidates.length : 0;
      
      weights[queryIndex] = Math.max(0.1, avgScore); // Minimum weight of 0.1
      queryIndex++;
    }
    
    // Normalize weights
    const sum = weights.reduce((a, b) => a + b, 0);
    return weights.map(w => w / sum);
  }

  private updateBudgetTracker(processingTime: number, itemCount: number): void {
    // Update FLOPS estimate based on processing time and item count
    const estimatedFlops = itemCount * 1000; // Simplified estimate
    this.budgetTracker.current_flops += estimatedFlops;
    
    // Check if still within budget (±5%)
    const variance = Math.abs(this.budgetTracker.current_flops - this.budgetTracker.flops_baseline) / 
      this.budgetTracker.flops_baseline;
    
    this.budgetTracker.within_budget = variance <= 0.05;
    this.budgetTracker.variance_percent = variance * 100;
  }
}

export default RankDiversifySystem;