/**
 * @fileoverview Grid search script for adaptive planning threshold optimization
 * Milestone 2: Automated hyperparameter tuning for plan selection thresholds
 */

import type Database from 'better-sqlite3';
import { 
  AdaptivePlanner,
  createAdaptivePlanner,
  type AdaptivePlanningConfig,
  type PlanDecision,
  type QueryFeatures 
} from './adaptive-planning.js';
import { SessionIdfCalculator } from './session-idf.js';
import { EntityExtractor } from './entity-extraction.js';

type DB = Database.Database;

/**
 * Grid search configuration
 */
export interface GridSearchConfig {
  // Parameter ranges to search
  tau_v_range: { min: number; max: number; step: number };
  tau_e_range: { min: number; max: number; step: number };
  tau_n_range: { min: number; max: number; step: number };
  
  // Evaluation settings
  evaluation_queries: Array<{
    query: string;
    expected_plan?: 'VERIFY' | 'EXPLORE' | 'EXPLOIT';
    weight: number; // Importance weight for this query
  }>;
  
  // Cross-validation
  cv_folds: number;
  
  // Metrics to optimize
  optimize_metric: 'accuracy' | 'f1_score' | 'plan_diversity' | 'combined';
  
  // Performance constraints
  max_combinations: number;
  parallel_workers?: number;
}

/**
 * Grid search result for a parameter combination
 */
export interface GridSearchResult {
  parameters: {
    tau_v: number;
    tau_e: number;
    tau_n: number;
  };
  
  metrics: {
    accuracy: number;        // Fraction of correct plan predictions
    precision: Record<'VERIFY' | 'EXPLORE' | 'EXPLOIT', number>;
    recall: Record<'VERIFY' | 'EXPLORE' | 'EXPLOIT', number>;
    f1_score: Record<'VERIFY' | 'EXPLORE' | 'EXPLOIT', number>;
    plan_diversity: number;  // Shannon entropy of plan distribution
    avg_confidence: number;  // Average prediction confidence
  };
  
  plan_distribution: Record<'VERIFY' | 'EXPLORE' | 'EXPLOIT', number>;
  cross_validation_score: number;
  evaluation_time_ms: number;
}

/**
 * Grid search summary
 */
export interface GridSearchSummary {
  best_parameters: GridSearchResult['parameters'];
  best_score: number;
  best_result: GridSearchResult;
  
  all_results: GridSearchResult[];
  search_space_size: number;
  total_evaluation_time_ms: number;
  
  parameter_importance: {
    tau_v: number;
    tau_e: number; 
    tau_n: number;
  };
}

/**
 * Default grid search configuration
 */
export const DEFAULT_GRID_CONFIG: GridSearchConfig = {
  tau_v_range: { min: 1.5, max: 4.0, step: 0.5 },
  tau_e_range: { min: 0.2, max: 0.7, step: 0.1 },
  tau_n_range: { min: 0.1, max: 0.4, step: 0.1 },
  
  evaluation_queries: [
    // VERIFY cases - high IDF, good entity overlap, non-code
    {
      query: 'React component state initialization patterns',
      expected_plan: 'VERIFY',
      weight: 1.0,
    },
    {
      query: 'user authentication session management debugging',
      expected_plan: 'VERIFY',
      weight: 1.0,
    },
    
    // EXPLORE cases - low entity overlap or no tools
    {
      query: 'machine learning model architecture design principles',
      expected_plan: 'EXPLORE',
      weight: 1.0,
    },
    {
      query: 'database schema optimization best practices',
      expected_plan: 'EXPLORE',
      weight: 1.0,
    },
    
    // EXPLOIT cases - balanced context
    {
      query: 'git merge conflict resolution with npm dependencies',
      expected_plan: 'EXPLOIT',
      weight: 1.0,
    },
    {
      query: 'typescript interface implementation for api client',
      expected_plan: 'EXPLOIT', 
      weight: 1.0,
    },
    
    // Mixed cases without expected plans
    {
      query: 'function getUserData() { return fetch("/api/user"); }',
      weight: 0.5,
    },
    {
      query: 'Error: TypeError cannot read property of undefined',
      weight: 0.5,
    },
    {
      query: 'implement responsive design with CSS grid layout',
      weight: 0.5,
    },
    {
      query: 'performance optimization using browser profiler tools',
      weight: 0.5,
    },
  ],
  
  cv_folds: 3,
  optimize_metric: 'combined',
  max_combinations: 100,
};

/**
 * Grid search optimizer for adaptive planning thresholds
 */
export class GridSearchOptimizer {
  private db: DB;
  private sessionIdf: SessionIdfCalculator;
  private entityExtractor: EntityExtractor;
  private baseConfig: AdaptivePlanningConfig;

  constructor(
    db: DB,
    sessionIdf: SessionIdfCalculator,
    entityExtractor: EntityExtractor,
    baseConfig: AdaptivePlanningConfig
  ) {
    this.db = db;
    this.sessionIdf = sessionIdf;
    this.entityExtractor = entityExtractor;
    this.baseConfig = baseConfig;
  }

  /**
   * Generate parameter combinations for grid search
   */
  private generateParameterGrid(config: GridSearchConfig): Array<{
    tau_v: number;
    tau_e: number;
    tau_n: number;
  }> {
    const combinations: Array<{ tau_v: number; tau_e: number; tau_n: number }> = [];
    
    // Generate all combinations
    for (let tau_v = config.tau_v_range.min; tau_v <= config.tau_v_range.max; tau_v += config.tau_v_range.step) {
      for (let tau_e = config.tau_e_range.min; tau_e <= config.tau_e_range.max; tau_e += config.tau_e_range.step) {
        for (let tau_n = config.tau_n_range.min; tau_n <= config.tau_n_range.max; tau_n += config.tau_n_range.step) {
          // Ensure tau_n < tau_e for logical consistency
          if (tau_n < tau_e) {
            combinations.push({
              tau_v: Math.round(tau_v * 10) / 10,
              tau_e: Math.round(tau_e * 10) / 10,
              tau_n: Math.round(tau_n * 10) / 10,
            });
          }
        }
      }
    }
    
    // Limit search space if too large
    if (combinations.length > config.max_combinations) {
      // Sample uniformly from the space
      const step = Math.floor(combinations.length / config.max_combinations);
      return combinations.filter((_, i) => i % step === 0).slice(0, config.max_combinations);
    }
    
    return combinations;
  }

  /**
   * Evaluate a parameter combination
   */
  private async evaluateParameters(
    parameters: { tau_v: number; tau_e: number; tau_n: number },
    queries: GridSearchConfig['evaluation_queries'],
    sessionId: string
  ): Promise<GridSearchResult> {
    const startTime = Date.now();
    
    // Create planner with these parameters
    const testConfig: AdaptivePlanningConfig = {
      ...this.baseConfig,
      thresholds: parameters,
      enable_logging: false, // Don't log during grid search
    };
    
    const planner = new AdaptivePlanner(this.db, this.sessionIdf, this.entityExtractor, testConfig);
    
    // Evaluate on all queries
    const decisions: PlanDecision[] = [];
    const expectedPlans: Array<'VERIFY' | 'EXPLORE' | 'EXPLOIT' | undefined> = [];
    const weights: number[] = [];
    
    for (const queryConfig of queries) {
      const decision = await planner.makePlanDecision(queryConfig.query, sessionId);
      decisions.push(decision);
      expectedPlans.push(queryConfig.expected_plan);
      weights.push(queryConfig.weight);
    }
    
    // Calculate metrics
    const metrics = this.calculateMetrics(decisions, expectedPlans, weights);
    
    const evaluationTime = Date.now() - startTime;
    
    return {
      parameters,
      metrics,
      plan_distribution: this.calculatePlanDistribution(decisions),
      cross_validation_score: metrics.f1_score.VERIFY + metrics.f1_score.EXPLORE + metrics.f1_score.EXPLOIT,
      evaluation_time_ms: evaluationTime,
    };
  }

  /**
   * Calculate evaluation metrics
   */
  private calculateMetrics(
    decisions: PlanDecision[],
    expectedPlans: Array<'VERIFY' | 'EXPLORE' | 'EXPLOIT' | undefined>,
    weights: number[]
  ): GridSearchResult['metrics'] {
    const planTypes: Array<'VERIFY' | 'EXPLORE' | 'EXPLOIT'> = ['VERIFY', 'EXPLORE', 'EXPLOIT'];
    
    // Count true/false positives for each plan type
    const counts = {
      VERIFY: { tp: 0, fp: 0, fn: 0, total: 0 },
      EXPLORE: { tp: 0, fp: 0, fn: 0, total: 0 },
      EXPLOIT: { tp: 0, fp: 0, fn: 0, total: 0 },
    };
    
    let correctPredictions = 0;
    let totalWeight = 0;
    let confidenceSum = 0;
    
    for (let i = 0; i < decisions.length; i++) {
      const decision = decisions[i];
      const expected = expectedPlans[i];
      const weight = weights[i];
      
      totalWeight += weight;
      confidenceSum += decision.confidence;
      
      if (expected) {
        // We have an expected plan to compare against
        if (decision.plan === expected) {
          correctPredictions += weight;
          counts[expected].tp += weight;
        } else {
          counts[expected].fn += weight;
          counts[decision.plan].fp += weight;
        }
        counts[expected].total += weight;
      }
      
      // Count all predictions for plan distribution
      counts[decision.plan].total += weight;
    }
    
    // Calculate precision, recall, F1 for each plan type
    const precision: Record<'VERIFY' | 'EXPLORE' | 'EXPLOIT', number> = {} as any;
    const recall: Record<'VERIFY' | 'EXPLORE' | 'EXPLOIT', number> = {} as any;
    const f1_score: Record<'VERIFY' | 'EXPLORE' | 'EXPLOIT', number> = {} as any;
    
    for (const planType of planTypes) {
      const { tp, fp, fn } = counts[planType];
      
      precision[planType] = (tp + fp) > 0 ? tp / (tp + fp) : 0;
      recall[planType] = (tp + fn) > 0 ? tp / (tp + fn) : 0;
      
      const p = precision[planType];
      const r = recall[planType];
      f1_score[planType] = (p + r) > 0 ? 2 * p * r / (p + r) : 0;
    }
    
    // Calculate plan diversity (Shannon entropy)
    const totalDecisions = decisions.length;
    const planCounts = planTypes.map(plan => 
      decisions.filter(d => d.plan === plan).length
    );
    
    let entropy = 0;
    for (const count of planCounts) {
      if (count > 0) {
        const prob = count / totalDecisions;
        entropy -= prob * Math.log2(prob);
      }
    }
    const plan_diversity = entropy / Math.log2(planTypes.length); // Normalized [0,1]
    
    return {
      accuracy: totalWeight > 0 ? correctPredictions / totalWeight : 0,
      precision,
      recall,
      f1_score,
      plan_diversity,
      avg_confidence: confidenceSum / decisions.length,
    };
  }

  /**
   * Calculate plan distribution
   */
  private calculatePlanDistribution(decisions: PlanDecision[]): Record<'VERIFY' | 'EXPLORE' | 'EXPLOIT', number> {
    const total = decisions.length;
    return {
      VERIFY: decisions.filter(d => d.plan === 'VERIFY').length / total,
      EXPLORE: decisions.filter(d => d.plan === 'EXPLORE').length / total,
      EXPLOIT: decisions.filter(d => d.plan === 'EXPLOIT').length / total,
    };
  }

  /**
   * Calculate combined score for optimization
   */
  private calculateCombinedScore(result: GridSearchResult): number {
    const { metrics } = result;
    
    // Weighted combination of metrics
    const accuracyScore = metrics.accuracy * 0.4;
    const f1Score = (metrics.f1_score.VERIFY + metrics.f1_score.EXPLORE + metrics.f1_score.EXPLOIT) / 3 * 0.4;
    const diversityScore = metrics.plan_diversity * 0.2;
    
    return accuracyScore + f1Score + diversityScore;
  }

  /**
   * Run grid search optimization
   */
  async runGridSearch(
    config: GridSearchConfig = DEFAULT_GRID_CONFIG,
    sessionId: string = 'grid-search-session'
  ): Promise<GridSearchSummary> {
    console.log('🔍 Starting adaptive planning grid search...');
    
    const startTime = Date.now();
    const parameterGrid = this.generateParameterGrid(config);
    
    console.log(`📊 Evaluating ${parameterGrid.length} parameter combinations`);
    console.log(`🎯 Optimizing for: ${config.optimize_metric}`);
    
    // Evaluate all parameter combinations
    const results: GridSearchResult[] = [];
    let completed = 0;
    
    for (const params of parameterGrid) {
      try {
        const result = await this.evaluateParameters(params, config.evaluation_queries, sessionId);
        results.push(result);
        
        completed++;
        if (completed % 10 === 0 || completed === parameterGrid.length) {
          console.log(`⏳ Progress: ${completed}/${parameterGrid.length} (${Math.round(completed/parameterGrid.length*100)}%)`);
        }
      } catch (error) {
        console.warn(`⚠️ Failed to evaluate parameters ${JSON.stringify(params)}:`, error);
      }
    }
    
    if (results.length === 0) {
      throw new Error('No valid parameter combinations found');
    }
    
    // Find best result based on optimization metric
    let bestResult: GridSearchResult;
    
    switch (config.optimize_metric) {
      case 'accuracy':
        bestResult = results.reduce((a, b) => a.metrics.accuracy > b.metrics.accuracy ? a : b);
        break;
      case 'f1_score':
        bestResult = results.reduce((a, b) => a.cross_validation_score > b.cross_validation_score ? a : b);
        break;
      case 'plan_diversity':
        bestResult = results.reduce((a, b) => a.metrics.plan_diversity > b.metrics.plan_diversity ? a : b);
        break;
      case 'combined':
      default:
        bestResult = results.reduce((a, b) => 
          this.calculateCombinedScore(a) > this.calculateCombinedScore(b) ? a : b
        );
        break;
    }
    
    // Calculate parameter importance (variance explained)
    const parameterImportance = this.calculateParameterImportance(results);
    
    const totalTime = Date.now() - startTime;
    
    const summary: GridSearchSummary = {
      best_parameters: bestResult.parameters,
      best_score: this.calculateCombinedScore(bestResult),
      best_result: bestResult,
      all_results: results.sort((a, b) => this.calculateCombinedScore(b) - this.calculateCombinedScore(a)),
      search_space_size: parameterGrid.length,
      total_evaluation_time_ms: totalTime,
      parameter_importance: parameterImportance,
    };
    
    this.printGridSearchSummary(summary);
    
    return summary;
  }

  /**
   * Calculate parameter importance using variance analysis
   */
  private calculateParameterImportance(results: GridSearchResult[]): {
    tau_v: number;
    tau_e: number;
    tau_n: number;
  } {
    if (results.length < 2) {
      return { tau_v: 1/3, tau_e: 1/3, tau_n: 1/3 };
    }
    
    const scores = results.map(r => this.calculateCombinedScore(r));
    const meanScore = scores.reduce((a, b) => a + b) / scores.length;
    const totalVariance = scores.reduce((sum, score) => sum + Math.pow(score - meanScore, 2), 0);
    
    if (totalVariance === 0) {
      return { tau_v: 1/3, tau_e: 1/3, tau_n: 1/3 };
    }
    
    // Calculate variance explained by each parameter
    const paramVariances = { tau_v: 0, tau_e: 0, tau_n: 0 };
    
    // Group by each parameter and calculate within-group variance
    for (const param of ['tau_v', 'tau_e', 'tau_n'] as const) {
      const paramGroups = new Map<number, number[]>();
      
      for (let i = 0; i < results.length; i++) {
        const paramValue = results[i].parameters[param];
        if (!paramGroups.has(paramValue)) {
          paramGroups.set(paramValue, []);
        }
        paramGroups.get(paramValue)!.push(scores[i]);
      }
      
      let withinGroupVariance = 0;
      for (const [, groupScores] of paramGroups) {
        const groupMean = groupScores.reduce((a, b) => a + b) / groupScores.length;
        withinGroupVariance += groupScores.reduce((sum, score) => sum + Math.pow(score - groupMean, 2), 0);
      }
      
      paramVariances[param] = (totalVariance - withinGroupVariance) / totalVariance;
    }
    
    // Normalize to sum to 1
    const totalImportance = Object.values(paramVariances).reduce((a, b) => a + b);
    if (totalImportance > 0) {
      paramVariances.tau_v /= totalImportance;
      paramVariances.tau_e /= totalImportance;
      paramVariances.tau_n /= totalImportance;
    }
    
    return paramVariances;
  }

  /**
   * Print grid search summary
   */
  private printGridSearchSummary(summary: GridSearchSummary): void {
    console.log('\n🎉 Grid Search Complete!');
    console.log('='.repeat(50));
    
    console.log('\n🏆 Best Parameters:');
    console.log(`   tau_v: ${summary.best_parameters.tau_v}`);
    console.log(`   tau_e: ${summary.best_parameters.tau_e}`);
    console.log(`   tau_n: ${summary.best_parameters.tau_n}`);
    
    console.log('\n📈 Best Metrics:');
    const best = summary.best_result;
    console.log(`   Accuracy: ${(best.metrics.accuracy * 100).toFixed(1)}%`);
    console.log(`   F1 Scores: V=${best.metrics.f1_score.VERIFY.toFixed(3)}, E=${best.metrics.f1_score.EXPLORE.toFixed(3)}, X=${best.metrics.f1_score.EXPLOIT.toFixed(3)}`);
    console.log(`   Plan Diversity: ${(best.metrics.plan_diversity * 100).toFixed(1)}%`);
    console.log(`   Avg Confidence: ${(best.metrics.avg_confidence * 100).toFixed(1)}%`);
    console.log(`   Combined Score: ${summary.best_score.toFixed(3)}`);
    
    console.log('\n📊 Plan Distribution:');
    console.log(`   VERIFY:  ${(best.plan_distribution.VERIFY * 100).toFixed(1)}%`);
    console.log(`   EXPLORE: ${(best.plan_distribution.EXPLORE * 100).toFixed(1)}%`);  
    console.log(`   EXPLOIT: ${(best.plan_distribution.EXPLOIT * 100).toFixed(1)}%`);
    
    console.log('\n🔧 Parameter Importance:');
    console.log(`   tau_v: ${(summary.parameter_importance.tau_v * 100).toFixed(1)}%`);
    console.log(`   tau_e: ${(summary.parameter_importance.tau_e * 100).toFixed(1)}%`);
    console.log(`   tau_n: ${(summary.parameter_importance.tau_n * 100).toFixed(1)}%`);
    
    console.log('\n⏱️ Search Statistics:');
    console.log(`   Combinations Evaluated: ${summary.all_results.length}`);
    console.log(`   Total Time: ${(summary.total_evaluation_time_ms / 1000).toFixed(1)}s`);
    console.log(`   Avg Time per Combination: ${(summary.total_evaluation_time_ms / summary.all_results.length).toFixed(1)}ms`);
    
    console.log('\n🔝 Top 5 Parameter Combinations:');
    for (let i = 0; i < Math.min(5, summary.all_results.length); i++) {
      const result = summary.all_results[i];
      const score = this.calculateCombinedScore(result);
      console.log(`   ${i+1}. τv=${result.parameters.tau_v}, τe=${result.parameters.tau_e}, τn=${result.parameters.tau_n} → ${score.toFixed(3)}`);
    }
  }

  /**
   * Export grid search results to JSON
   */
  exportResults(summary: GridSearchSummary, filename: string): void {
    const fs = require('fs');
    
    const exportData = {
      timestamp: new Date().toISOString(),
      summary,
      metadata: {
        base_config: this.baseConfig,
        evaluation_queries_count: summary.all_results.length > 0 ? 
          Object.keys(summary.all_results[0].metrics.f1_score).length : 0,
      },
    };
    
    fs.writeFileSync(filename, JSON.stringify(exportData, null, 2));
    console.log(`📁 Results exported to: ${filename}`);
  }
}

/**
 * Utility function to run grid search with default setup
 */
export async function runAdaptivePlanningGridSearch(
  db: DB,
  sessionIdf: SessionIdfCalculator,
  entityExtractor: EntityExtractor,
  baseConfig: AdaptivePlanningConfig,
  searchConfig?: Partial<GridSearchConfig>
): Promise<GridSearchSummary> {
  const optimizer = new GridSearchOptimizer(db, sessionIdf, entityExtractor, baseConfig);
  
  const fullConfig: GridSearchConfig = {
    ...DEFAULT_GRID_CONFIG,
    ...searchConfig,
  };
  
  return optimizer.runGridSearch(fullConfig);
}

/**
 * Quick optimization with sensible defaults
 */
export async function quickOptimize(
  db: DB,
  sessionIdf: SessionIdfCalculator,
  entityExtractor: EntityExtractor,
  baseConfig: AdaptivePlanningConfig
): Promise<{ tau_v: number; tau_e: number; tau_n: number }> {
  const quickConfig: Partial<GridSearchConfig> = {
    tau_v_range: { min: 2.0, max: 3.5, step: 0.5 },
    tau_e_range: { min: 0.3, max: 0.6, step: 0.1 },
    tau_n_range: { min: 0.1, max: 0.3, step: 0.1 },
    max_combinations: 30,
  };
  
  const summary = await runAdaptivePlanningGridSearch(db, sessionIdf, entityExtractor, baseConfig, quickConfig);
  return summary.best_parameters;
}