/**
 * @fileoverview Adaptive Planning Policy for Lethe agent-context manager
 * Milestone 2: Intelligent retrieval strategy adaptation based on agent conversation patterns
 */

import type Database from 'better-sqlite3';
import { SessionIdfCalculator } from './session-idf.js';
import { EntityExtractor } from './entity-extraction.js';
import { AtomWithEntities, EntityKind, SearchContext } from './atoms-types.js';

type DB = Database.Database;

/**
 * Planning strategies for retrieval
 */
export type PlanType = 'VERIFY' | 'EXPLORE' | 'EXPLOIT';

/**
 * Per-query feature vector for plan selection
 */
export interface QueryFeatures {
  // Core IDF features
  max_idf: number;          // Maximum IDF score for query terms
  avg_idf: number;          // Average IDF score for query terms  
  len_q: number;            // Query length in tokens

  // Context overlap features
  entity_overlap: number;   // Jaccard similarity with recent entities (window W)
  tool_overlap: number;     // Binary indicator for tool mention overlap

  // Content type indicators
  has_code: boolean;        // Contains code patterns
  has_error_pattern: boolean; // Contains error/exception patterns
  has_identifier: boolean;  // Contains variable/function identifiers

  // Agent conversation context
  recent_tool_count: number; // Count of tools mentioned in window W
  turn_position: number;     // Position in conversation (0-1 normalized)
}

/**
 * Adaptive planning configuration
 */
export interface AdaptivePlanningConfig {
  // Rule thresholds for plan selection
  thresholds: {
    tau_v: number;    // VERIFY threshold for max_idf
    tau_e: number;    // Entity overlap threshold for VERIFY
    tau_n: number;    // Entity overlap threshold for EXPLORE (< tau_n)
  };

  // Plan -> weights mapping
  weights: {
    [K in PlanType]: {
      alpha: number;      // Hybrid fusion weight
      efSearch: number;   // HNSW efSearch parameter
    }
  };

  // Context window settings  
  window_turns: number;   // W: turns to look back for context
  
  // Safety and fallback
  min_results: number;    // k: minimum results required
  backoff_query_expansion: boolean; // Enable query broadening on low results
  
  // Feature extraction
  code_patterns: RegExp[];      // Patterns for detecting code content
  error_patterns: RegExp[];     // Patterns for detecting errors
  identifier_patterns: RegExp[]; // Patterns for detecting identifiers
  known_tools: Set<string>;     // Known agent tools to detect
  
  // Logging
  enable_logging: boolean;
  log_features: boolean;
}

/**
 * Plan decision with metadata
 */
export interface PlanDecision {
  plan: PlanType;
  features: QueryFeatures;
  reasoning: string;
  confidence: number;
  timestamp: number;
  
  // Applied configuration
  alpha: number;
  efSearch: number;
}

/**
 * Plan decision log entry
 */
export interface PlanLog {
  session_id: string;
  query: string;
  plan: PlanType;
  features: QueryFeatures;
  reasoning: string;
  confidence: number;
  results_count: number;
  timestamp: number;
}

/**
 * Default adaptive planning configuration
 */
export const DEFAULT_ADAPTIVE_CONFIG: AdaptivePlanningConfig = {
  thresholds: {
    tau_v: 2.5,    // High IDF threshold for VERIFY
    tau_e: 0.4,    // Entity overlap threshold for VERIFY  
    tau_n: 0.2,    // Low entity overlap threshold for EXPLORE
  },
  
  weights: {
    VERIFY: {
      alpha: 0.7,     // Higher precision focus
      efSearch: 50,   // Lower search effort
    },
    EXPLORE: {
      alpha: 0.3,     // Higher recall focus  
      efSearch: 200,  // Higher search effort
    },
    EXPLOIT: {
      alpha: 0.5,     // Balanced approach
      efSearch: 100,  // Medium search effort
    },
  },
  
  window_turns: 10,
  min_results: 3,
  backoff_query_expansion: true,
  
  // Code detection patterns
  code_patterns: [
    /\b(?:function|class|const|let|var|def|import|from|require)\s+/gi,
    /\b(?:if|else|for|while|try|catch|finally)\s*\(/gi,
    /(?:=>|->|\+=|-=|\*=|\/=)/g,
    /\{[^}]*\}/g,                    // Code blocks
    /\([^)]*\)\s*(?:=>|{)/g,        // Function definitions
  ],
  
  // Error detection patterns  
  error_patterns: [
    /\b(?:error|exception|traceback|stack\s*trace)\b/gi,
    /\b(?:failed|failure|crash|abort|timeout)\b/gi,
    /\b\d{3}\s*error\b/gi,          // HTTP errors
    /\b[A-Z][a-zA-Z]*Error\b/g,     // Error classes
    /\bERR_[A-Z0-9_]+\b/g,          // Error codes
  ],
  
  // Identifier detection patterns
  identifier_patterns: [
    /\b[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\b/g, // module.function
    /\b[a-zA-Z_][a-zA-Z0-9_]*\(\)/g,                         // function()
    /\$\{[^}]+\}/g,                                          // ${variable}
    /\b[A-Z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]*)*\b/g,         // PascalCase
  ],
  
  // Common agent tools
  known_tools: new Set([
    'git', 'npm', 'node', 'python', 'pip', 'curl', 'docker', 'kubectl',
    'grep', 'awk', 'sed', 'bash', 'ssh', 'scp', 'rsync', 'tar', 'zip',
    'jest', 'vitest', 'cypress', 'playwright', 'webpack', 'vite', 'rollup',
    'typescript', 'eslint', 'prettier', 'babel', 'parcel',
  ]),
  
  enable_logging: true,
  log_features: true,
};

/**
 * Adaptive planning engine
 */
export class AdaptivePlanner {
  private db: DB;
  private config: AdaptivePlanningConfig;
  private sessionIdf: SessionIdfCalculator;
  private entityExtractor: EntityExtractor;

  constructor(
    db: DB,
    sessionIdf: SessionIdfCalculator,
    entityExtractor: EntityExtractor,
    config: AdaptivePlanningConfig = DEFAULT_ADAPTIVE_CONFIG
  ) {
    this.db = db;
    this.config = config;
    this.sessionIdf = sessionIdf;
    this.entityExtractor = entityExtractor;
    
    this.initializePlanLogging();
  }

  /**
   * Initialize planning decision log table
   */
  private initializePlanLogging(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS plan_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        query TEXT NOT NULL,
        plan TEXT NOT NULL CHECK (plan IN ('VERIFY', 'EXPLORE', 'EXPLOIT')),
        features TEXT NOT NULL, -- JSON serialized QueryFeatures
        reasoning TEXT NOT NULL,
        confidence REAL NOT NULL,
        results_count INTEGER DEFAULT 0,
        timestamp INTEGER NOT NULL
      );
      
      CREATE INDEX IF NOT EXISTS idx_plan_decisions_session 
      ON plan_decisions(session_id);
      
      CREATE INDEX IF NOT EXISTS idx_plan_decisions_timestamp 
      ON plan_decisions(timestamp);
    `);
  }

  /**
   * Tokenize query for analysis
   */
  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(token => token.length > 1);
  }

  /**
   * Extract per-query features
   */
  async extractQueryFeatures(
    query: string,
    sessionId: string,
    context: Partial<SearchContext> = {}
  ): Promise<QueryFeatures> {
    const tokens = this.tokenize(query);
    const windowTurns = context.windowTurns || this.config.window_turns;
    
    // 1. IDF features
    const idfWeights = this.sessionIdf.getSessionIdfWeights(sessionId, tokens);
    const idfValues = tokens.map(t => idfWeights.get(t) || 0);
    const max_idf = idfValues.length > 0 ? Math.max(...idfValues) : 0;
    const avg_idf = idfValues.length > 0 ? idfValues.reduce((a, b) => a + b) / idfValues.length : 0;
    
    // 2. Query length
    const len_q = tokens.length;
    
    // 3. Entity overlap with recent history
    const queryEntities = await this.entityExtractor.extractEntities(query);
    const queryEntitySet = new Set(
      queryEntities.entities.map(e => e.text.toLowerCase())
    );
    
    const recentEntities = this.getRecentEntities(sessionId, windowTurns);
    const recentEntitySet = new Set(recentEntities.map(e => e.toLowerCase()));
    
    const entity_overlap = this.calculateJaccardSimilarity(queryEntitySet, recentEntitySet);
    
    // 4. Tool overlap
    const recentTools = this.getRecentTools(sessionId, windowTurns);
    const queryMentionsKnownTools = tokens.some(t => 
      this.config.known_tools.has(t) || recentTools.has(t)
    );
    const tool_overlap = queryMentionsKnownTools ? 1 : 0;
    
    // 5. Content type indicators
    // Reset regex lastIndex to avoid global flag state issues
    const has_code = this.config.code_patterns.some(pattern => {
      pattern.lastIndex = 0;
      return pattern.test(query);
    });
    const has_error_pattern = this.config.error_patterns.some(pattern => {
      pattern.lastIndex = 0;
      return pattern.test(query);
    });  
    const has_identifier = this.config.identifier_patterns.some(pattern => {
      pattern.lastIndex = 0;
      return pattern.test(query);
    });
    
    
    // 6. Agent conversation context
    const recent_tool_count = recentTools.size;
    const turn_position = this.getTurnPosition(sessionId);
    
    return {
      max_idf,
      avg_idf,
      len_q,
      entity_overlap,
      tool_overlap,
      has_code,
      has_error_pattern,
      has_identifier,
      recent_tool_count,
      turn_position,
    };
  }

  /**
   * Get recent entities from conversation history
   */
  private getRecentEntities(sessionId: string, windowTurns: number): string[] {
    const stmt = this.db.prepare(`
      SELECT DISTINCT e.entity 
      FROM entities e 
      JOIN atoms a ON e.atom_id = a.id
      WHERE a.session_id = ? 
        AND a.turn_idx >= (
          SELECT MAX(turn_idx) - ? FROM atoms WHERE session_id = ?
        )
      ORDER BY a.turn_idx DESC
    `);
    
    const results = stmt.all(sessionId, windowTurns, sessionId) as Array<{ entity: string }>;
    return results.map(r => r.entity);
  }

  /**
   * Get recently mentioned tools
   */
  private getRecentTools(sessionId: string, windowTurns: number): Set<string> {
    const stmt = this.db.prepare(`
      SELECT a.text 
      FROM atoms a
      WHERE a.session_id = ? 
        AND a.turn_idx >= (
          SELECT MAX(turn_idx) - ? FROM atoms WHERE session_id = ?
        )
      ORDER BY a.turn_idx DESC
    `);
    
    const results = stmt.all(sessionId, windowTurns, sessionId) as Array<{ text: string }>;
    const tools = new Set<string>();
    
    for (const result of results) {
      const tokens = this.tokenize(result.text);
      for (const token of tokens) {
        if (this.config.known_tools.has(token)) {
          tools.add(token);
        }
      }
    }
    
    return tools;
  }

  /**
   * Get normalized turn position in conversation (0-1)
   */
  private getTurnPosition(sessionId: string): number {
    const stmt = this.db.prepare(`
      SELECT 
        MAX(turn_idx) as max_turn,
        MIN(turn_idx) as min_turn
      FROM atoms 
      WHERE session_id = ?
    `);
    
    const result = stmt.get(sessionId) as { max_turn: number; min_turn: number } | undefined;
    if (!result || result.max_turn === result.min_turn) {
      return 0;
    }
    
    return (result.max_turn - result.min_turn) / Math.max(result.max_turn, 1);
  }

  /**
   * Calculate Jaccard similarity between two sets
   */
  private calculateJaccardSimilarity<T>(setA: Set<T>, setB: Set<T>): number {
    if (setA.size === 0 && setB.size === 0) {
      return 1.0; // Both empty = perfect similarity
    }
    
    const intersection = new Set([...setA].filter(x => setB.has(x)));
    const union = new Set([...setA, ...setB]);
    
    return intersection.size / union.size;
  }

  /**
   * Select plan based on calibrated rule policy
   */
  selectPlan(features: QueryFeatures): { plan: PlanType; reasoning: string; confidence: number } {
    const { tau_v, tau_e, tau_n } = this.config.thresholds;
    
    // VERIFY: High IDF + entity overlap + NOT code-heavy
    if (features.max_idf > tau_v && features.entity_overlap > tau_e && !features.has_code) {
      return {
        plan: 'VERIFY',
        reasoning: `High IDF (${features.max_idf.toFixed(2)} > ${tau_v}) + entity overlap (${features.entity_overlap.toFixed(2)} > ${tau_e}) + non-code query`,
        confidence: Math.min(features.max_idf / tau_v * features.entity_overlap / tau_e, 1.0),
      };
    }
    
    // EXPLORE: Low entity overlap OR no tool overlap  
    if (features.entity_overlap < tau_n || features.tool_overlap === 0) {
      return {
        plan: 'EXPLORE',
        reasoning: `Low entity overlap (${features.entity_overlap.toFixed(2)} < ${tau_n}) or no tool overlap (${features.tool_overlap})`,
        confidence: Math.max(0.1, 1.0 - Math.max(features.entity_overlap / tau_n, features.tool_overlap)),
      };
    }
    
    // EXPLOIT: Default case - balanced exploration
    return {
      plan: 'EXPLOIT',
      reasoning: `Balanced case: entity_overlap=${features.entity_overlap.toFixed(2)}, tool_overlap=${features.tool_overlap}, has_code=${features.has_code}`,
      confidence: 0.7, // Moderate confidence for default case
    };
  }

  /**
   * Make planning decision for query
   */
  async makePlanDecision(
    query: string,
    sessionId: string,
    context: Partial<SearchContext> = {}
  ): Promise<PlanDecision> {
    const features = await this.extractQueryFeatures(query, sessionId, context);
    const { plan, reasoning, confidence } = this.selectPlan(features);
    
    const weights = this.config.weights[plan];
    const decision: PlanDecision = {
      plan,
      features,
      reasoning,
      confidence,
      timestamp: Date.now(),
      alpha: weights.alpha,
      efSearch: weights.efSearch,
    };
    
    // Log decision
    if (this.config.enable_logging) {
      this.logPlanDecision(sessionId, query, decision);
    }
    
    return decision;
  }

  /**
   * Log plan decision for analysis
   */
  private logPlanDecision(sessionId: string, query: string, decision: PlanDecision): void {
    const stmt = this.db.prepare(`
      INSERT INTO plan_decisions (
        session_id, query, plan, features, reasoning, confidence, timestamp
      ) VALUES (?, ?, ?, ?, ?, ?, ?)
    `);
    
    stmt.run(
      sessionId,
      query,
      decision.plan,
      JSON.stringify(decision.features),
      decision.reasoning,
      decision.confidence,
      decision.timestamp
    );
  }

  /**
   * Update plan decision with results count
   */
  updatePlanResults(sessionId: string, query: string, resultsCount: number): void {
    const stmt = this.db.prepare(`
      UPDATE plan_decisions 
      SET results_count = ?
      WHERE session_id = ? AND query = ? 
      ORDER BY timestamp DESC 
      LIMIT 1
    `);
    
    stmt.run(resultsCount, sessionId, query);
  }

  /**
   * Get plan decision statistics for analysis
   */
  getPlanStats(sessionId?: string, timeWindow?: number): {
    planCounts: Record<PlanType, number>;
    avgConfidence: Record<PlanType, number>;
    avgResultsCount: Record<PlanType, number>;
    featureDistributions: Record<keyof QueryFeatures, { mean: number; std: number }>;
  } {
    let whereClause = '1=1';
    const params: any[] = [];
    
    if (sessionId) {
      whereClause += ' AND session_id = ?';
      params.push(sessionId);
    }
    
    if (timeWindow) {
      const cutoff = Date.now() - timeWindow;
      whereClause += ' AND timestamp > ?';
      params.push(cutoff);
    }
    
    // Get plan counts and averages
    const statsStmt = this.db.prepare(`
      SELECT 
        plan,
        COUNT(*) as count,
        AVG(confidence) as avg_confidence,
        AVG(results_count) as avg_results
      FROM plan_decisions 
      WHERE ${whereClause}
      GROUP BY plan
    `);
    
    const statsResults = statsStmt.all(...params) as Array<{
      plan: PlanType;
      count: number;
      avg_confidence: number;
      avg_results: number;
    }>;
    
    const planCounts: Record<PlanType, number> = { VERIFY: 0, EXPLORE: 0, EXPLOIT: 0 };
    const avgConfidence: Record<PlanType, number> = { VERIFY: 0, EXPLORE: 0, EXPLOIT: 0 };
    const avgResultsCount: Record<PlanType, number> = { VERIFY: 0, EXPLORE: 0, EXPLOIT: 0 };
    
    for (const result of statsResults) {
      planCounts[result.plan] = result.count;
      avgConfidence[result.plan] = result.avg_confidence;
      avgResultsCount[result.plan] = result.avg_results;
    }
    
    // Get feature distributions (simplified for now)
    const featuresStmt = this.db.prepare(`
      SELECT features FROM plan_decisions WHERE ${whereClause}
    `);
    
    const featureResults = featuresStmt.all(...params) as Array<{ features: string }>;
    const featureDistributions: Record<keyof QueryFeatures, { mean: number; std: number }> = 
      {} as any;
    
    if (featureResults.length > 0) {
      const parsedFeatures = featureResults.map(r => JSON.parse(r.features) as QueryFeatures);
      
      // Calculate means and std devs for numeric features
      const numericFeatures: (keyof QueryFeatures)[] = [
        'max_idf', 'avg_idf', 'len_q', 'entity_overlap', 'tool_overlap',
        'recent_tool_count', 'turn_position'
      ];
      
      for (const feature of numericFeatures) {
        const values = parsedFeatures.map(f => f[feature] as number);
        const mean = values.reduce((a, b) => a + b) / values.length;
        const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
        const std = Math.sqrt(variance);
        
        featureDistributions[feature] = { mean, std };
      }
    }
    
    return {
      planCounts,
      avgConfidence,
      avgResultsCount,
      featureDistributions,
    };
  }

  /**
   * Suggest threshold adjustments based on observed performance
   */
  suggestThresholdTuning(sessionId?: string): {
    suggestions: Array<{
      parameter: keyof AdaptivePlanningConfig['thresholds'];
      current: number;
      suggested: number;
      reasoning: string;
    }>;
    confidence: number;
  } {
    const stats = this.getPlanStats(sessionId);
    const suggestions: Array<{
      parameter: keyof AdaptivePlanningConfig['thresholds'];
      current: number;
      suggested: number;
      reasoning: string;
    }> = [];
    
    // Simple heuristic-based suggestions
    const totalDecisions = Object.values(stats.planCounts).reduce((a, b) => a + b, 0);
    
    if (totalDecisions < 10) {
      return { suggestions: [], confidence: 0 }; // Not enough data
    }
    
    // If VERIFY never triggers, lower tau_v
    if (stats.planCounts.VERIFY === 0) {
      suggestions.push({
        parameter: 'tau_v',
        current: this.config.thresholds.tau_v,
        suggested: this.config.thresholds.tau_v * 0.8,
        reasoning: 'VERIFY plan never selected - consider lowering tau_v threshold',
      });
    }
    
    // If EXPLORE dominates (>70%), adjust tau_n
    if (stats.planCounts.EXPLORE / totalDecisions > 0.7) {
      suggestions.push({
        parameter: 'tau_n',
        current: this.config.thresholds.tau_n,
        suggested: this.config.thresholds.tau_n * 0.7,
        reasoning: 'EXPLORE plan dominates - consider lowering tau_n to increase EXPLOIT usage',
      });
    }
    
    const confidence = Math.min(totalDecisions / 50, 1.0); // More confident with more data
    
    return { suggestions, confidence };
  }

  /**
   * Update configuration thresholds
   */
  updateThresholds(newThresholds: Partial<AdaptivePlanningConfig['thresholds']>): void {
    this.config.thresholds = {
      ...this.config.thresholds,
      ...newThresholds,
    };
  }

  /**
   * Get current configuration
   */
  getConfig(): AdaptivePlanningConfig {
    return { ...this.config };
  }
}

/**
 * Utility function to create adaptive planner with defaults
 */
export function createAdaptivePlanner(
  db: DB,
  sessionIdf: SessionIdfCalculator,
  entityExtractor: EntityExtractor,
  config?: Partial<AdaptivePlanningConfig>
): AdaptivePlanner {
  const fullConfig = {
    ...DEFAULT_ADAPTIVE_CONFIG,
    ...config,
    thresholds: {
      ...DEFAULT_ADAPTIVE_CONFIG.thresholds,
      ...config?.thresholds,
    },
    weights: {
      ...DEFAULT_ADAPTIVE_CONFIG.weights,
      ...config?.weights,
    },
    // Merge arrays and sets specially
    code_patterns: [
      ...DEFAULT_ADAPTIVE_CONFIG.code_patterns,
      ...(config?.code_patterns || []),
    ],
    error_patterns: [
      ...DEFAULT_ADAPTIVE_CONFIG.error_patterns,
      ...(config?.error_patterns || []),
    ],
    identifier_patterns: [
      ...DEFAULT_ADAPTIVE_CONFIG.identifier_patterns,
      ...(config?.identifier_patterns || []),
    ],
    known_tools: new Set([
      ...DEFAULT_ADAPTIVE_CONFIG.known_tools,
      ...(config?.known_tools || []),
    ]),
  };
  
  return new AdaptivePlanner(db, sessionIdf, entityExtractor, fullConfig);
}