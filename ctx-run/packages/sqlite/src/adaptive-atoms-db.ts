/**
 * @fileoverview Enhanced AtomsDatabase with Adaptive Planning Policy
 * Milestone 2: Intelligent retrieval strategy adaptation integrated with atoms database
 */

import type Database from 'better-sqlite3';
import { 
  AtomsDatabase,
  createAtomsDatabase,
  DEFAULT_HYBRID_SEARCH_CONFIG 
} from './atoms-db.js';
import {
  AdaptivePlanner,
  createAdaptivePlanner,
  DEFAULT_ADAPTIVE_CONFIG,
  type AdaptivePlanningConfig,
  type PlanDecision,
  type PlanType
} from './adaptive-planning.js';
import {
  type SearchResults,
  type SearchContext,
  type HybridSearchConfig,
  type AtomDbConfig,
} from './atoms-types.js';

type DB = Database.Database;

/**
 * Enhanced search context with adaptive planning
 */
export interface AdaptiveSearchContext extends Omit<SearchContext, 'sessionId'> {
  sessionId?: string;
  // Override plan selection
  forcePlan?: PlanType;
  
  // Enable/disable plan-aware search
  useAdaptivePlanning?: boolean;
  
  // Minimum results threshold for backoff
  minResults?: number;
  
  // Query expansion options for backoff
  queryExpansion?: {
    enabled: boolean;
    synonyms?: string[];
    stemming?: boolean;
  };
}

/**
 * Enhanced search results with planning metadata
 */
export interface AdaptiveSearchResults extends SearchResults {
  planDecision?: PlanDecision;
  backoffApplied?: boolean;
  originalQuery?: string;
  expandedQuery?: string;
}

/**
 * Configuration for adaptive atoms database
 */
export interface AdaptiveAtomDbConfig extends AtomDbConfig {
  adaptivePlanning: AdaptivePlanningConfig;
  
  // Safety and fallback settings
  backoffStrategy: {
    enabled: boolean;
    minResultsThreshold: number;
    queryExpansionRules: Array<{
      pattern: RegExp;
      replacement: string;
      description: string;
    }>;
  };
}

/**
 * Enhanced AtomsDatabase with adaptive planning policy
 */
export class AdaptiveAtomsDatabase extends AtomsDatabase {
  private adaptivePlanner: AdaptivePlanner;
  private adaptiveConfig: AdaptiveAtomDbConfig;

  constructor(db: DB, config: AdaptiveAtomDbConfig) {
    super(db, config);
    this.adaptiveConfig = config;
    
    // Initialize adaptive planner
    const sessionIdfCalculator = (this as any).sessionIdfCalculator;
    const entityExtractor = (this as any).entityExtractor;
    
    this.adaptivePlanner = createAdaptivePlanner(
      db,
      sessionIdfCalculator,
      entityExtractor,
      config.adaptivePlanning
    );
  }

  /**
   * Adaptive hybrid search with intelligent strategy selection
   */
  async adaptiveSearch(
    query: string,
    context: AdaptiveSearchContext,
    limit: number = 20
  ): Promise<AdaptiveSearchResults> {
    const startTime = Date.now();
    
    if (!context.sessionId) {
      throw new Error('sessionId is required for adaptive search');
    }
    
    const sessionId = context.sessionId;
    
    // 1. Make planning decision (unless forced)
    let planDecision: PlanDecision;
    
    if (context.forcePlan) {
      // Use forced plan with default weights
      const planWeights = this.adaptiveConfig.adaptivePlanning.weights[context.forcePlan];
      planDecision = {
        plan: context.forcePlan,
        features: {} as any, // Skip feature extraction for forced plans
        reasoning: `Forced plan: ${context.forcePlan}`,
        confidence: 1.0,
        timestamp: Date.now(),
        alpha: planWeights.alpha,
        efSearch: planWeights.efSearch,
      };
    } else if (context.useAdaptivePlanning !== false) {
      // Make intelligent planning decision
      planDecision = await this.adaptivePlanner.makePlanDecision(query, sessionId);
    } else {
      // Use default EXPLOIT plan
      const exploitWeights = this.adaptiveConfig.adaptivePlanning.weights.EXPLOIT;
      planDecision = {
        plan: 'EXPLOIT',
        features: {
          max_idf: 0, avg_idf: 0, len_q: 0, entity_overlap: 0, tool_overlap: 0,
          has_code: false, has_error_pattern: false, has_identifier: false,
          recent_tool_count: 0, turn_position: 0
        },
        reasoning: 'Adaptive planning disabled, using EXPLOIT',
        confidence: 0.5,
        timestamp: Date.now(),
        alpha: exploitWeights.alpha,
        efSearch: exploitWeights.efSearch,
      };
    }

    // 2. Configure hybrid search based on plan
    const hybridConfig = this.planToHybridConfig(planDecision, context);
    
    // 3. Execute search with plan-specific configuration
    const searchContext: SearchContext = { ...context, sessionId };
    let searchResults = await super.hybridSearch(query, searchContext, hybridConfig, limit);
    
    // 4. Apply safety fallback if needed
    let backoffApplied = false;
    let originalQuery = query;
    let expandedQuery = query;
    
    const minResults = context.minResults || this.adaptiveConfig.backoffStrategy.minResultsThreshold;
    
    if (searchResults.atoms.length < minResults && this.adaptiveConfig.backoffStrategy.enabled) {
      console.log(`🔄 Applying backoff: ${searchResults.atoms.length} < ${minResults} results`);
      
      // Try EXPLORE strategy with query expansion
      expandedQuery = this.expandQuery(query);
      const exploreConfig = this.planToHybridConfig({
        ...planDecision,
        plan: 'EXPLORE',
        alpha: this.adaptiveConfig.adaptivePlanning.weights.EXPLORE.alpha,
        efSearch: this.adaptiveConfig.adaptivePlanning.weights.EXPLORE.efSearch,
        reasoning: 'Backoff to EXPLORE due to insufficient results',
      }, context);
      
      searchResults = await super.hybridSearch(expandedQuery, searchContext, exploreConfig, limit);
      backoffApplied = true;
    }

    // 5. Update plan decision with results count
    this.adaptivePlanner.updatePlanResults(
      sessionId,
      query,
      searchResults.atoms.length
    );

    const totalTime = Date.now() - startTime;

    return {
      ...searchResults,
      planDecision,
      backoffApplied,
      originalQuery,
      expandedQuery: backoffApplied ? expandedQuery : undefined,
      metadata: {
        ...searchResults.metadata,
        searchTime: totalTime,
        method: 'hybrid' as const,
      },
    };
  }

  /**
   * Convert plan decision to hybrid search configuration
   */
  private planToHybridConfig(
    planDecision: PlanDecision,
    context: AdaptiveSearchContext
  ): HybridSearchConfig {
    const baseConfig = this.adaptiveConfig.hybridSearch;
    const alpha = planDecision.alpha;
    const efSearch = planDecision.efSearch;

    switch (planDecision.plan) {
      case 'VERIFY':
        // High precision: favor FTS (exact matches), lower vector search effort
        return {
          ...baseConfig,
          ftsWeight: Math.max(alpha, 0.7),      // High FTS weight
          vectorWeight: Math.min(1 - alpha, 0.3), // Lower vector weight
          entityWeight: 0.15,                   // Moderate entity boost
          efSearch,                            // Lower efSearch for speed
          diversify: false,                    // Less diversity, more precision
        };

      case 'EXPLORE':
        // High recall: favor vectors (semantic similarity), higher search effort
        return {
          ...baseConfig,
          ftsWeight: Math.min(alpha, 0.3),      // Lower FTS weight  
          vectorWeight: Math.max(1 - alpha, 0.7), // High vector weight
          entityWeight: 0.05,                  // Lower entity weight (exploring new areas)
          efSearch,                            // Higher efSearch for recall
          diversify: true,                     // High diversity
          diversityLambda: 0.5,               // More diversity
        };

      case 'EXPLOIT':
      default:
        // Balanced: use configured alpha for FTS/vector balance
        return {
          ...baseConfig,
          ftsWeight: alpha,
          vectorWeight: 1 - alpha,
          entityWeight: 0.1,                   // Standard entity weight
          efSearch,                            // Medium efSearch
          diversify: true,                     // Moderate diversity
          diversityLambda: 0.7,               // Balanced diversity
        };
    }
  }

  /**
   * Query expansion for backoff strategy
   */
  private expandQuery(query: string): string {
    let expandedQuery = query;
    
    // Apply query expansion rules
    for (const rule of this.adaptiveConfig.backoffStrategy.queryExpansionRules) {
      if (rule.pattern.test(query)) {
        expandedQuery = expandedQuery.replace(rule.pattern, rule.replacement);
        console.log(`📝 Applied expansion rule: ${rule.description}`);
        break; // Apply only first matching rule
      }
    }
    
    // If no rules applied, add general broadening terms
    if (expandedQuery === query) {
      const tokens = query.toLowerCase().split(/\s+/);
      const broadeningSuffixes = ['related', 'similar', 'pattern', 'example', 'approach'];
      
      // Add one broadening term
      if (tokens.length > 0) {
        const randomSuffix = broadeningSuffixes[Math.floor(Math.random() * broadeningSuffixes.length)];
        expandedQuery = `${query} ${randomSuffix}`;
      }
    }
    
    return expandedQuery;
  }

  /**
   * Get adaptive planning statistics
   */
  getPlanningStats(sessionId?: string, timeWindow?: number) {
    return this.adaptivePlanner.getPlanStats(sessionId, timeWindow);
  }

  /**
   * Get threshold tuning suggestions
   */
  getThresholdSuggestions(sessionId?: string) {
    return this.adaptivePlanner.suggestThresholdTuning(sessionId);
  }

  /**
   * Update planning thresholds
   */
  updatePlanningThresholds(newThresholds: Partial<AdaptivePlanningConfig['thresholds']>) {
    this.adaptivePlanner.updateThresholds(newThresholds);
    console.log('🔧 Updated planning thresholds:', newThresholds);
  }

  /**
   * Export planning decisions for analysis
   */
  exportPlanningData(sessionId?: string): Array<{
    session_id: string;
    query: string;
    plan: PlanType;
    features: string;
    reasoning: string;
    confidence: number;
    results_count: number;
    timestamp: number;
  }> {
    const db = (this as any).db as DB;
    
    let whereClause = '1=1';
    const params: any[] = [];
    
    if (sessionId) {
      whereClause = 'session_id = ?';
      params.push(sessionId);
    }
    
    const stmt = db.prepare(`
      SELECT session_id, query, plan, features, reasoning, confidence, results_count, timestamp
      FROM plan_decisions 
      WHERE ${whereClause}
      ORDER BY timestamp DESC
    `);
    
    return stmt.all(...params) as any[];
  }

  /**
   * Analyze plan effectiveness over time
   */
  analyzePlanEffectiveness(sessionId?: string): {
    planPerformance: Record<PlanType, {
      avgResultsCount: number;
      avgConfidence: number;
      usageCount: number;
      successRate: number; // % of queries returning >= minResults
    }>;
    recommendations: string[];
  } {
    const planData = this.exportPlanningData(sessionId);
    const minResults = this.adaptiveConfig.backoffStrategy.minResultsThreshold;
    
    const planPerformance: Record<PlanType, {
      avgResultsCount: number;
      avgConfidence: number;
      usageCount: number;
      successRate: number;
    }> = {
      VERIFY: { avgResultsCount: 0, avgConfidence: 0, usageCount: 0, successRate: 0 },
      EXPLORE: { avgResultsCount: 0, avgConfidence: 0, usageCount: 0, successRate: 0 },
      EXPLOIT: { avgResultsCount: 0, avgConfidence: 0, usageCount: 0, successRate: 0 },
    };
    
    // Group by plan type
    const planGroups: Record<PlanType, typeof planData> = {
      VERIFY: [], EXPLORE: [], EXPLOIT: []
    };
    
    for (const entry of planData) {
      planGroups[entry.plan as PlanType].push(entry);
    }
    
    // Calculate statistics for each plan
    for (const [plan, entries] of Object.entries(planGroups) as Array<[PlanType, typeof planData]>) {
      if (entries.length === 0) continue;
      
      const totalResults = entries.reduce((sum, e) => sum + e.results_count, 0);
      const totalConfidence = entries.reduce((sum, e) => sum + e.confidence, 0);
      const successfulQueries = entries.filter(e => e.results_count >= minResults).length;
      
      planPerformance[plan] = {
        avgResultsCount: totalResults / entries.length,
        avgConfidence: totalConfidence / entries.length,
        usageCount: entries.length,
        successRate: successfulQueries / entries.length,
      };
    }
    
    // Generate recommendations
    const recommendations: string[] = [];
    
    // Check for low success rates
    for (const [plan, stats] of Object.entries(planPerformance) as Array<[PlanType, typeof planPerformance[PlanType]]>) {
      if (stats.usageCount >= 5 && stats.successRate < 0.6) {
        recommendations.push(`${plan} plan has low success rate (${(stats.successRate * 100).toFixed(1)}%) - consider adjusting thresholds`);
      }
    }
    
    // Check for unused plans
    for (const [plan, stats] of Object.entries(planPerformance) as Array<[PlanType, typeof planPerformance[PlanType]]>) {
      if (stats.usageCount === 0) {
        recommendations.push(`${plan} plan never used - thresholds may be too restrictive`);
      }
    }
    
    // Check for imbalanced usage
    const totalUsage = Object.values(planPerformance).reduce((sum, stats) => sum + stats.usageCount, 0);
    if (totalUsage > 0) {
      for (const [plan, stats] of Object.entries(planPerformance) as Array<[PlanType, typeof planPerformance[PlanType]]>) {
        const usagePercent = stats.usageCount / totalUsage;
        if (usagePercent > 0.8) {
          recommendations.push(`${plan} plan dominates (${(usagePercent * 100).toFixed(1)}%) - consider rebalancing thresholds`);
        }
      }
    }
    
    return { planPerformance, recommendations };
  }
}

/**
 * Default adaptive configuration with sensible backoff strategy
 */
export const DEFAULT_ADAPTIVE_ATOMS_CONFIG: Omit<AdaptiveAtomDbConfig, keyof AtomDbConfig> = {
  adaptivePlanning: DEFAULT_ADAPTIVE_CONFIG,
  
  backoffStrategy: {
    enabled: true,
    minResultsThreshold: 3,
    queryExpansionRules: [
      {
        pattern: /\berror\b/gi,
        replacement: 'error exception problem issue',
        description: 'Expand error queries with related terms',
      },
      {
        pattern: /\bfunction\s+(\w+)/gi,
        replacement: 'function $1 method implementation',
        description: 'Expand function queries with related terms',
      },
      {
        pattern: /\bapi\b/gi,
        replacement: 'api endpoint service interface',
        description: 'Expand API queries with related terms',
      },
      {
        pattern: /\bcomponent\b/gi,
        replacement: 'component element widget module',
        description: 'Expand component queries with related terms',
      },
      {
        pattern: /\bdebug\b/gi,
        replacement: 'debug troubleshoot fix solve',
        description: 'Expand debugging queries with related terms',
      },
    ],
  },
};

/**
 * Create adaptive atoms database with intelligent planning
 */
export function createAdaptiveAtomsDatabase(
  db: DB, 
  config?: Partial<AdaptiveAtomDbConfig>
): AdaptiveAtomsDatabase {
  // Merge with default atoms database config
  const baseAtomsConfig = {
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

  const fullConfig: AdaptiveAtomDbConfig = {
    ...baseAtomsConfig,
    ...config,
    ...DEFAULT_ADAPTIVE_ATOMS_CONFIG,
    adaptivePlanning: {
      ...DEFAULT_ADAPTIVE_ATOMS_CONFIG.adaptivePlanning,
      ...config?.adaptivePlanning,
      thresholds: {
        ...DEFAULT_ADAPTIVE_ATOMS_CONFIG.adaptivePlanning.thresholds,
        ...config?.adaptivePlanning?.thresholds,
      },
      weights: {
        ...DEFAULT_ADAPTIVE_ATOMS_CONFIG.adaptivePlanning.weights,
        ...config?.adaptivePlanning?.weights,
      },
    },
    backoffStrategy: {
      ...DEFAULT_ADAPTIVE_ATOMS_CONFIG.backoffStrategy,
      ...config?.backoffStrategy,
    },
  };

  return new AdaptiveAtomsDatabase(db, fullConfig);
}