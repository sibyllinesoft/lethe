import { Config, Result, LetheError, EnhancedCandidate, Message, PerformanceMetrics, BudgetTracker, RankingContext, ContradictionDetection, LLMQueryContext, LLMResponse } from './types.js';
import { RetrievalSystem } from './retrieval.js';
import { MessageChunker } from './chunker.js';
import { RankDiversifySystem } from './rank_diversify.js';
import { CrossEncoderReranker } from '../../reranker/src/reranker.js';

/**
 * OrchestrationSystem coordinates all components of the Lethe retrieval pipeline
 * with LLM integration, performance tracking, and contradiction detection.
 * 
 * Implements variants V1-V5 with specific performance targets:
 * - V1: Baseline p50≤3s, p95≤6s
 * - V2: +15% precision with ranking
 * - V3: +25% diversity with semantic
 * - V4: +10% efficiency with LLM reranking
 * - V5: +20% accuracy with contradiction detection
 */
export class OrchestrationSystem {
  private retrieval: RetrievalSystem;
  private chunker: MessageChunker;
  private rankDiversify: RankDiversifySystem;
  private reranker?: CrossEncoderReranker;
  private budgetTracker: BudgetTracker;
  private config: Config;
  
  // Performance tracking
  private sessionMetrics = new Map<string, PerformanceMetrics>();
  private contradictionCache = new Map<string, ContradictionDetection[]>();
  
  constructor(config: Config) {
    this.config = config;
    this.retrieval = new RetrievalSystem(config);
    this.chunker = new MessageChunker(config);
    this.rankDiversify = new RankDiversifySystem(config);
    
    // Initialize LLM reranker if enabled
    if (config.llm?.enableReranking) {
      this.reranker = new CrossEncoderReranker(
        config.llm.rerankerModel,
        config.llm.batchSize
      );
    }
    
    this.budgetTracker = {
      totalBudget: config.performance.budgetTracking.totalBudget,
      usedBudget: 0,
      operationCosts: new Map(),
      budgetAlerts: []
    };
  }

  /**
   * Main orchestration method implementing the complete Lethe pipeline
   * with variant-specific optimizations and performance tracking.
   */
  async orchestrate(
    sessionId: string,
    query: string | string[],
    messages: Message[],
    variant: 'V1' | 'V2' | 'V3' | 'V4' | 'V5' = 'V1'
  ): Promise<Result<OrchestrationResult, LetheError>> {
    const startTime = Date.now();
    const queries = Array.isArray(query) ? query : [query];
    
    try {
      // Initialize session metrics
      this.initializeSessionMetrics(sessionId, variant);
      
      // Step 1: Chunking with strategy selection
      const chunkResult = await this.performChunking(messages, variant);
      if (!chunkResult.success) {
        return { success: false, error: chunkResult.error };
      }
      
      // Step 2: Retrieval with variant-specific configuration
      const retrievalResult = await this.performRetrieval(
        sessionId, 
        queries, 
        variant
      );
      if (!retrievalResult.success) {
        return { success: false, error: retrievalResult.error };
      }
      
      // Step 3: Ranking and diversification (V2+)
      let rankedCandidates = retrievalResult.data;
      if (this.shouldApplyRanking(variant)) {
        const rankResult = await this.performRanking(
          rankedCandidates, 
          queries, 
          variant
        );
        if (!rankResult.success) {
          return { success: false, error: rankResult.error };
        }
        rankedCandidates = rankResult.data;
      }
      
      // Step 4: LLM reranking (V4+)
      if (this.shouldApplyLLMReranking(variant)) {
        const rerankerResult = await this.performLLMReranking(
          rankedCandidates,
          queries[0], // Use primary query for reranking
          variant
        );
        if (!rerankerResult.success) {
          return { success: false, error: rerankerResult.error };
        }
        rankedCandidates = rerankerResult.data;
      }
      
      // Step 5: Contradiction detection (V5)
      let contradictions: ContradictionDetection[] = [];
      if (variant === 'V5' && this.config.llm?.enableContradictionDetection) {
        const contradictionResult = await this.detectContradictions(
          rankedCandidates,
          queries[0]
        );
        if (contradictionResult.success) {
          contradictions = contradictionResult.data;
        }
      }
      
      // Step 6: Final performance tracking and validation
      const endTime = Date.now();
      const totalTime = endTime - startTime;
      
      const performanceResult = this.validatePerformance(
        sessionId, 
        totalTime, 
        variant
      );
      if (!performanceResult.success) {
        return { success: false, error: performanceResult.error };
      }
      
      // Step 7: Generate telemetry
      await this.generateTelemetry(sessionId, {
        query: queries,
        variant,
        totalTime,
        candidateCount: rankedCandidates.length,
        contradictionCount: contradictions.length,
        budgetUsed: this.budgetTracker.usedBudget / this.budgetTracker.totalBudget
      });
      
      return {
        success: true,
        data: {
          candidates: rankedCandidates,
          contradictions,
          performance: this.sessionMetrics.get(sessionId)!,
          variant,
          totalTime
        }
      };
      
    } catch (error) {
      const letheError: LetheError = {
        code: 'ORCHESTRATION_FAILED',
        message: `Orchestration failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date().toISOString(),
        context: { sessionId, variant, query: queries }
      };
      
      return { success: false, error: letheError };
    }
  }
  
  /**
   * Performs chunking with variant-specific strategy selection
   */
  private async performChunking(
    messages: Message[], 
    variant: string
  ): Promise<Result<void, LetheError>> {
    const startTime = performance.now();
    
    // Select chunking strategy based on variant
    const strategy = this.selectChunkingStrategy(variant);
    
    try {
      const originalStrategy = this.chunker.getStrategy();
      this.chunker.setStrategy(strategy);
      
      // Process all messages
      for (const message of messages) {
        const chunkResult = this.chunker.chunkMessage(message);
        if (!chunkResult.success) {
          return { success: false, error: chunkResult.error };
        }
      }
      
      // Restore original strategy
      this.chunker.setStrategy(originalStrategy);
      
      // Track chunking cost
      const chunkingTime = performance.now() - startTime;
      this.budgetTracker.operationCosts.set('chunking', chunkingTime);
      this.budgetTracker.usedBudget += chunkingTime * 0.001; // Convert to budget units
      
      return { success: true, data: undefined };
      
    } catch (error) {
      return {
        success: false,
        error: {
          code: 'CHUNKING_FAILED',
          message: `Chunking failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          timestamp: new Date().toISOString(),
          context: { variant, messageCount: messages.length }
        }
      };
    }
  }
  
  /**
   * Performs retrieval with variant-specific configuration
   */
  private async performRetrieval(
    sessionId: string,
    queries: string[],
    variant: string
  ): Promise<Result<EnhancedCandidate[], LetheError>> {
    const startTime = performance.now();
    
    try {
      // Configure retrieval method based on variant
      const method = this.selectRetrievalMethod(variant);
      const topK = this.getTopKForVariant(variant);
      
      // Perform retrieval
      const result = await this.retrieval.search(sessionId, queries, topK);
      
      // Track retrieval cost
      const retrievalTime = performance.now() - startTime;
      this.budgetTracker.operationCosts.set('retrieval', retrievalTime);
      this.budgetTracker.usedBudget += retrievalTime * 0.002; // Higher cost for retrieval
      
      return result;
      
    } catch (error) {
      return {
        success: false,
        error: {
          code: 'RETRIEVAL_FAILED',
          message: `Retrieval failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          timestamp: new Date().toISOString(),
          context: { sessionId, variant, queries }
        }
      };
    }
  }
  
  /**
   * Performs ranking and diversification with variant-specific parameters
   */
  private async performRanking(
    candidates: EnhancedCandidate[],
    queries: string[],
    variant: string
  ): Promise<Result<EnhancedCandidate[], LetheError>> {
    const startTime = performance.now();
    
    try {
      const rankingContext: RankingContext = {
        sessionId: `ranking_${Date.now()}`,
        primaryQuery: queries[0],
        additionalQueries: queries.slice(1),
        candidateCount: candidates.length,
        diversificationMethod: this.getDiversificationMethod(variant),
        fusionWeights: this.getFusionWeights(variant),
        metadata: {
          variant,
          startTime: startTime,
          budgetRemaining: this.budgetTracker.totalBudget - this.budgetTracker.usedBudget
        }
      };
      
      // Apply enhanced ranking
      const enhancedResult = await this.rankDiversify.enhancedRank(
        rankingContext
      );
      
      if (!enhancedResult.success) {
        return enhancedResult;
      }
      
      let rankedCandidates = enhancedResult.data;
      
      // Apply semantic diversification for V3+
      if (variant >= 'V3') {
        const diversifyResult = await this.rankDiversify.semanticDiversify(
          rankedCandidates,
          this.getMaxChunksForVariant(variant),
          this.getDiversificationMethod(variant)
        );
        
        if (diversifyResult.success) {
          rankedCandidates = diversifyResult.data.diversifiedCandidates;
        }
      }
      
      // Track ranking cost
      const rankingTime = performance.now() - startTime;
      this.budgetTracker.operationCosts.set('ranking', rankingTime);
      this.budgetTracker.usedBudget += rankingTime * 0.0015;
      
      return { success: true, data: rankedCandidates };
      
    } catch (error) {
      return {
        success: false,
        error: {
          code: 'RANKING_FAILED',
          message: `Ranking failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          timestamp: new Date().toISOString(),
          context: { variant, candidateCount: candidates.length }
        }
      };
    }
  }
  
  /**
   * Performs LLM reranking for V4+ variants
   */
  private async performLLMReranking(
    candidates: EnhancedCandidate[],
    query: string,
    variant: string
  ): Promise<Result<EnhancedCandidate[], LetheError>> {
    if (!this.reranker) {
      return {
        success: false,
        error: {
          code: 'LLM_RERANKER_NOT_INITIALIZED',
          message: 'LLM reranker not initialized but reranking requested',
          timestamp: new Date().toISOString(),
          context: { variant }
        }
      };
    }
    
    const startTime = performance.now();
    
    try {
      // Convert candidates to reranker format
      const rerankerCandidates = candidates.map(candidate => ({
        id: candidate.id,
        text: candidate.content,
        query: query
      }));
      
      // Perform reranking
      const topK = Math.min(candidates.length, this.getTopKForVariant(variant));
      const rerankedResults = await this.reranker.rankPairs(rerankerCandidates, topK);
      
      // Map results back to enhanced candidates
      const rerankedCandidates: EnhancedCandidate[] = [];
      for (const result of rerankedResults) {
        const originalCandidate = candidates.find(c => c.id === result.id);
        if (originalCandidate) {
          rerankedCandidates.push({
            ...originalCandidate,
            score: result.score,
            llmScore: result.score,
            metadata: {
              ...originalCandidate.metadata,
              llmReranked: true,
              rerankerModel: this.config.llm?.rerankerModel
            }
          });
        }
      }
      
      // Track LLM reranking cost
      const rerankerTime = performance.now() - startTime;
      this.budgetTracker.operationCosts.set('llm_reranking', rerankerTime);
      this.budgetTracker.usedBudget += rerankerTime * 0.005; // Higher cost for LLM operations
      
      return { success: true, data: rerankedCandidates };
      
    } catch (error) {
      return {
        success: false,
        error: {
          code: 'LLM_RERANKING_FAILED',
          message: `LLM reranking failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          timestamp: new Date().toISOString(),
          context: { variant, candidateCount: candidates.length }
        }
      };
    }
  }
  
  /**
   * Detects contradictions in candidate results for V5
   */
  private async detectContradictions(
    candidates: EnhancedCandidate[],
    query: string
  ): Promise<Result<ContradictionDetection[], LetheError>> {
    const startTime = performance.now();
    
    // Check cache first
    const cacheKey = `${query}_${candidates.map(c => c.id).join(',')}`;
    if (this.contradictionCache.has(cacheKey)) {
      return { success: true, data: this.contradictionCache.get(cacheKey)! };
    }
    
    try {
      const contradictions: ContradictionDetection[] = [];
      
      // Pairwise contradiction detection
      for (let i = 0; i < candidates.length; i++) {
        for (let j = i + 1; j < candidates.length; j++) {
          const candidate1 = candidates[i];
          const candidate2 = candidates[j];
          
          // Simple contradiction detection based on content analysis
          const contradiction = await this.analyzeContradiction(
            candidate1,
            candidate2,
            query
          );
          
          if (contradiction) {
            contradictions.push(contradiction);
          }
        }
      }
      
      // Cache results
      this.contradictionCache.set(cacheKey, contradictions);
      
      // Track contradiction detection cost
      const contradictionTime = performance.now() - startTime;
      this.budgetTracker.operationCosts.set('contradiction_detection', contradictionTime);
      this.budgetTracker.usedBudget += contradictionTime * 0.003;
      
      return { success: true, data: contradictions };
      
    } catch (error) {
      return {
        success: false,
        error: {
          code: 'CONTRADICTION_DETECTION_FAILED',
          message: `Contradiction detection failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          timestamp: new Date().toISOString(),
          context: { query, candidateCount: candidates.length }
        }
      };
    }
  }
  
  /**
   * Analyzes potential contradiction between two candidates
   */
  private async analyzeContradiction(
    candidate1: EnhancedCandidate,
    candidate2: EnhancedCandidate,
    query: string
  ): Promise<ContradictionDetection | null> {
    // Simple heuristic-based contradiction detection
    // In a real implementation, this would use more sophisticated NLP
    
    const content1 = candidate1.content.toLowerCase();
    const content2 = candidate2.content.toLowerCase();
    
    // Look for contradictory patterns
    const contradictoryPairs = [
      ['yes', 'no'],
      ['true', 'false'],
      ['always', 'never'],
      ['enable', 'disable'],
      ['allow', 'deny'],
      ['should', 'should not'],
      ['can', 'cannot']
    ];
    
    for (const [positive, negative] of contradictoryPairs) {
      if (content1.includes(positive) && content2.includes(negative) ||
          content1.includes(negative) && content2.includes(positive)) {
        
        // Calculate confidence based on context similarity
        const confidence = this.calculateContradictionConfidence(
          candidate1,
          candidate2,
          positive,
          negative
        );
        
        if (confidence > 0.6) { // Threshold for contradiction detection
          return {
            candidateId1: candidate1.id,
            candidateId2: candidate2.id,
            contradictionType: 'semantic',
            confidence: confidence,
            description: `Potential contradiction between "${positive}" and "${negative}"`,
            query: query,
            resolvedBy: null // Will be resolved by human or additional LLM processing
          };
        }
      }
    }
    
    return null;
  }
  
  /**
   * Calculates confidence score for contradiction detection
   */
  private calculateContradictionConfidence(
    candidate1: EnhancedCandidate,
    candidate2: EnhancedCandidate,
    term1: string,
    term2: string
  ): number {
    // Simple confidence calculation based on context overlap
    const words1 = candidate1.content.toLowerCase().split(/\s+/);
    const words2 = candidate2.content.toLowerCase().split(/\s+/);
    
    const commonWords = words1.filter(word => words2.includes(word));
    const totalWords = Math.max(words1.length, words2.length);
    
    const contextSimilarity = commonWords.length / totalWords;
    
    // Higher confidence for similar contexts with contradictory terms
    return Math.min(0.9, contextSimilarity * 1.5);
  }
  
  /**
   * Validates performance against variant-specific targets
   */
  private validatePerformance(
    sessionId: string,
    totalTime: number,
    variant: string
  ): Result<void, LetheError> {
    const metrics = this.sessionMetrics.get(sessionId);
    if (!metrics) {
      return {
        success: false,
        error: {
          code: 'METRICS_NOT_FOUND',
          message: 'Session metrics not found',
          timestamp: new Date().toISOString(),
          context: { sessionId }
        }
      };
    }
    
    // Update metrics
    metrics.totalTime = totalTime;
    metrics.budgetUsed = this.budgetTracker.usedBudget;
    
    // Check performance targets
    const targets = this.getPerformanceTargets(variant);
    
    if (totalTime > targets.maxTime) {
      return {
        success: false,
        error: {
          code: 'PERFORMANCE_TARGET_EXCEEDED',
          message: `Total time ${totalTime}ms exceeds target ${targets.maxTime}ms for variant ${variant}`,
          timestamp: new Date().toISOString(),
          context: { sessionId, variant, totalTime, target: targets.maxTime }
        }
      };
    }
    
    if (this.budgetTracker.usedBudget > this.budgetTracker.totalBudget) {
      return {
        success: false,
        error: {
          code: 'BUDGET_EXCEEDED',
          message: `Budget ${this.budgetTracker.usedBudget} exceeds limit ${this.budgetTracker.totalBudget}`,
          timestamp: new Date().toISOString(),
          context: { sessionId, variant, budgetUsed: this.budgetTracker.usedBudget }
        }
      };
    }
    
    return { success: true, data: undefined };
  }
  
  /**
   * Generates JSONL telemetry for performance analysis
   */
  private async generateTelemetry(
    sessionId: string,
    data: {
      query: string[];
      variant: string;
      totalTime: number;
      candidateCount: number;
      contradictionCount: number;
      budgetUsed: number;
    }
  ): Promise<void> {
    if (!this.config.performance.telemetry.enabled) {
      return;
    }
    
    const telemetryEntry = {
      timestamp: new Date().toISOString(),
      sessionId,
      event: 'orchestration_complete',
      variant: data.variant,
      performance: {
        totalTime: data.totalTime,
        budgetUsed: data.budgetUsed,
        candidateCount: data.candidateCount,
        contradictionCount: data.contradictionCount
      },
      query: data.query,
      budgetBreakdown: Object.fromEntries(this.budgetTracker.operationCosts)
    };
    
    // In a real implementation, this would write to a log file or telemetry service
    console.log(JSON.stringify(telemetryEntry));
  }
  
  // Helper methods for variant-specific configuration
  
  private initializeSessionMetrics(sessionId: string, variant: string): void {
    this.sessionMetrics.set(sessionId, {
      sessionId,
      variant,
      startTime: Date.now(),
      totalTime: 0,
      retrievalTime: 0,
      rankingTime: 0,
      llmTime: 0,
      contradictionTime: 0,
      budgetUsed: 0,
      candidateCount: 0,
      contradictionCount: 0
    });
  }
  
  private selectChunkingStrategy(variant: string): 'ast' | 'hierarchical' | 'propositional' {
    switch (variant) {
      case 'V1':
      case 'V2':
        return 'hierarchical'; // Balanced approach
      case 'V3':
        return 'ast'; // Code-aware for better diversity
      case 'V4':
      case 'V5':
        return 'propositional'; // Semantic-aware for LLM processing
      default:
        return 'hierarchical';
    }
  }
  
  private selectRetrievalMethod(variant: string): 'window' | 'bm25' | 'vector' | 'hybrid' {
    switch (variant) {
      case 'V1':
        return 'bm25'; // Baseline
      case 'V2':
        return 'hybrid'; // Better precision with ranking
      case 'V3':
        return 'hybrid'; // Supports diversification
      case 'V4':
      case 'V5':
        return 'vector'; // Better for LLM processing
      default:
        return 'hybrid';
    }
  }
  
  private getTopKForVariant(variant: string): number {
    switch (variant) {
      case 'V1': return 50;
      case 'V2': return 75; // More candidates for ranking
      case 'V3': return 100; // More candidates for diversification
      case 'V4': return 50; // Focused for LLM reranking
      case 'V5': return 60; // Balanced for contradiction detection
      default: return 50;
    }
  }
  
  private shouldApplyRanking(variant: string): boolean {
    return variant >= 'V2';
  }
  
  private shouldApplyLLMReranking(variant: string): boolean {
    return variant >= 'V4' && this.config.llm?.enableReranking === true;
  }
  
  private getDiversificationMethod(variant: string): 'entity' | 'semantic' {
    return variant >= 'V3' ? 'semantic' : 'entity';
  }
  
  private getFusionWeights(variant: string): number[] {
    switch (variant) {
      case 'V1': return [1.0]; // Single method
      case 'V2': return [0.6, 0.4]; // BM25 + ranking
      case 'V3': return [0.5, 0.3, 0.2]; // BM25 + vector + diversity
      case 'V4': return [0.4, 0.3, 0.3]; // Balanced for LLM
      case 'V5': return [0.35, 0.35, 0.3]; // Optimized for contradiction detection
      default: return [1.0];
    }
  }
  
  private getMaxChunksForVariant(variant: string): number {
    switch (variant) {
      case 'V3': return 30; // More diversity
      case 'V4': return 25; // Focused for LLM
      case 'V5': return 35; // More coverage for contradiction detection
      default: return 20;
    }
  }
  
  private getPerformanceTargets(variant: string): { maxTime: number } {
    switch (variant) {
      case 'V1': return { maxTime: 3000 }; // 3s p50
      case 'V2': return { maxTime: 3500 }; // Slight increase for ranking
      case 'V3': return { maxTime: 4000 }; // Increase for diversification
      case 'V4': return { maxTime: 4000 }; // LLM variant target
      case 'V5': return { maxTime: 4500 }; // Increase for contradiction detection
      default: return { maxTime: 3000 };
    }
  }
}

/**
 * Result type for orchestration operations
 */
export interface OrchestrationResult {
  candidates: EnhancedCandidate[];
  contradictions: ContradictionDetection[];
  performance: PerformanceMetrics;
  variant: string;
  totalTime: number;
}

/**
 * Factory function to create orchestration system with proper configuration
 */
export function createOrchestrationSystem(config: Config): OrchestrationSystem {
  return new OrchestrationSystem(config);
}

/**
 * Helper function to validate orchestration configuration
 */
export function validateOrchestrationConfig(config: Config): Result<void, LetheError> {
  const errors: string[] = [];
  
  // Validate performance targets
  if (!config.performance.budgetTracking.totalBudget || config.performance.budgetTracking.totalBudget <= 0) {
    errors.push('Invalid budget tracking configuration');
  }
  
  // Validate LLM configuration if enabled
  if (config.llm?.enableReranking && !config.llm.rerankerModel) {
    errors.push('LLM reranking enabled but no reranker model specified');
  }
  
  if (config.llm?.enableContradictionDetection && !config.llm.model) {
    errors.push('Contradiction detection enabled but no LLM model specified');
  }
  
  // Validate chunking configuration
  if (!config.chunking?.strategy) {
    errors.push('No chunking strategy specified');
  }
  
  if (errors.length > 0) {
    return {
      success: false,
      error: {
        code: 'INVALID_ORCHESTRATION_CONFIG',
        message: `Configuration validation failed: ${errors.join(', ')}`,
        timestamp: new Date().toISOString(),
        context: { errors }
      }
    };
  }
  
  return { success: true, data: undefined };
}