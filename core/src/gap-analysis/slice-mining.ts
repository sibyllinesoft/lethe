/**
 * Slice Mining System - Automatic Stratification
 * 
 * Implements comprehensive slice mining and stratification over multiple
 * dimensions: dataset×budget×k, language, type mix, closure depth,
 * dup-intensity, and KV stability. Ranks slices by performance gaps
 * and statistical separability with multiple testing correction.
 */

import {
  SliceMiningResult,
  LanguageStratification,
  ContentTypeStratification,
  ComplexityBinStratification,
  StabilityDecileStratification,
  SliceGroup,
  PrioritizedTuningQueue,
  TuningQueueItem,
  GapRecord,
  GapAnalysisResult,
  GapAnalysisError,
  BatchGapAnalysisResult
} from './types.js';

import { Config, Candidate } from '../types.js';
import { createHash } from 'crypto';

// ============================================================================
// CORE SLICE MINING ENGINE
// ============================================================================

export class SliceMiningEngine {
  private config: Config;
  private statisticalAnalyzer: StatisticalAnalyzer;
  private stratificationDimensions: StratificationDimensions;
  private gapRanker: GapRanker;

  constructor(config: Config) {
    this.config = config;
    this.statisticalAnalyzer = new StatisticalAnalyzer();
    this.stratificationDimensions = new StratificationDimensions();
    this.gapRanker = new GapRanker();
  }

  /**
   * Performs comprehensive slice mining and stratification
   */
  async performSliceMining(
    evaluationResults: EvaluationResult[],
    competitorBaselines: CompetitorBaseline[]
  ): Promise<GapAnalysisResult<SliceMiningResult>> {
    try {
      const startTime = Date.now();
      console.log(`Starting slice mining on ${evaluationResults.length} evaluation results`);

      // Stage 1: Automatic stratification across all dimensions
      const stratificationResults = await this.performMultiDimensionalStratification(
        evaluationResults
      );

      // Stage 2: Gap identification within each slice
      const gapIdentificationResult = await this.identifyGapsAcrossSlices(
        stratificationResults,
        competitorBaselines
      );

      // Stage 3: Statistical analysis with multiple testing correction
      const statisticalSummary = await this.performStatisticalAnalysis(
        gapIdentificationResult.identifiedGaps
      );

      // Stage 4: Priority ranking for tuning queue
      const tuningQueue = await this.buildPrioritizedTuningQueue(
        gapIdentificationResult.identifiedGaps,
        statisticalSummary
      );

      const miningResult: SliceMiningResult = {
        mining_run_id: this.generateMiningRunId(),
        stratification_dimensions: stratificationResults,
        identified_gaps: gapIdentificationResult.identifiedGaps,
        statistical_summary: statisticalSummary,
        tuning_queue: tuningQueue,
        mining_timestamp: Date.now(),
        computational_cost: {
          cpu_hours: (Date.now() - startTime) / (1000 * 3600), // Convert to hours
          memory_peak_gb: this.estimateMemoryUsage(evaluationResults.length),
          wall_clock_minutes: (Date.now() - startTime) / (1000 * 60)
        }
      };

      console.log(`Slice mining completed. Identified ${gapIdentificationResult.identifiedGaps.length} gaps across ${this.countTotalSlices(stratificationResults)} slices`);

      return {
        success: true,
        data: miningResult
      };

    } catch (error) {
      return {
        success: false,
        error: {
          code: 'SLICE_MINING_ERROR',
          message: `Slice mining failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          error_type: 'gap_detection',
          recovery_actions: ['Verify evaluation results format', 'Check competitor baselines', 'Validate stratification configuration'],
          is_retryable: true,
          impact_severity: 'high',
          affected_components: ['gap_mining', 'tuning_pipeline'],
          timestamp: Date.now()
        }
      };
    }
  }

  // ============================================================================
  // MULTI-DIMENSIONAL STRATIFICATION
  // ============================================================================

  private async performMultiDimensionalStratification(
    evaluationResults: EvaluationResult[]
  ): Promise<SliceMiningResult['stratification_dimensions']> {
    console.log('Performing multi-dimensional stratification');

    // Dataset×Budget×K stratification
    const datasetBudgetK = this.stratificationDimensions.stratifyByDatasetBudgetK(evaluationResults);

    // Language distribution stratification
    const languageDistribution = this.stratificationDimensions.stratifyByLanguage(evaluationResults);

    // Content type mix stratification
    const contentTypeMix = this.stratificationDimensions.stratifyByContentType(evaluationResults);

    // Complexity bins stratification
    const complexityBins = this.stratificationDimensions.stratifyByComplexity(evaluationResults);

    // KV stability deciles stratification
    const stabilityDeciles = this.stratificationDimensions.stratifyByKVStability(evaluationResults);

    return {
      dataset_budget_k: datasetBudgetK,
      language_distribution: languageDistribution,
      content_type_mix: contentTypeMix,
      complexity_bins: complexityBins,
      stability_deciles: stabilityDeciles
    };
  }

  private async identifyGapsAcrossSlices(
    stratificationResults: SliceMiningResult['stratification_dimensions'],
    competitorBaselines: CompetitorBaseline[]
  ): Promise<{ identifiedGaps: GapRecord[]; sliceAnalysisResults: SliceAnalysisResult[] }> {
    const identifiedGaps: GapRecord[] = [];
    const sliceAnalysisResults: SliceAnalysisResult[] = [];

    // Collect all unique slices from all stratification dimensions
    const allSlices = this.collectAllSlices(stratificationResults);

    console.log(`Analyzing ${allSlices.length} unique slices for performance gaps`);

    for (const slice of allSlices) {
      try {
        const analysisResult = await this.analyzeSliceForGaps(slice, competitorBaselines);
        sliceAnalysisResults.push(analysisResult);

        if (analysisResult.hasSignificantGap) {
          identifiedGaps.push(analysisResult.gapRecord!);
        }
      } catch (error) {
        console.warn(`Failed to analyze slice ${slice.group_id}: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    }

    return { identifiedGaps, sliceAnalysisResults };
  }

  private async analyzeSliceForGaps(
    slice: SliceGroup,
    competitorBaselines: CompetitorBaseline[]
  ): Promise<SliceAnalysisResult> {
    // Find best competitor baseline for this slice characteristics
    const bestBaseline = this.findBestCompetitorBaseline(slice, competitorBaselines);

    if (!bestBaseline) {
      return {
        slice_id: slice.group_id,
        hasSignificantGap: false,
        reason: 'No competitor baseline found'
      };
    }

    // Calculate performance delta
    const performanceDelta = slice.performance_baseline.mean_p_at_5 - bestBaseline.p_at_5;
    const latencyDelta = slice.performance_baseline.mean_latency_p95 - bestBaseline.latency_p95;

    // Check for significant negative gaps (where we perform worse)
    if (performanceDelta >= -0.01) {
      return {
        slice_id: slice.group_id,
        hasSignificantGap: false,
        reason: 'No significant performance deficit'
      };
    }

    // Perform statistical significance test
    const statisticalTest = await this.performStatisticalSignificanceTest(slice, bestBaseline);

    if (!statisticalTest.is_significant) {
      return {
        slice_id: slice.group_id,
        hasSignificantGap: false,
        reason: 'Gap not statistically significant'
      };
    }

    // Create gap record
    const gapRecord = await this.createGapRecordFromSlice(slice, bestBaseline, performanceDelta, latencyDelta, statisticalTest);

    return {
      slice_id: slice.group_id,
      hasSignificantGap: true,
      gapRecord: gapRecord,
      performance_delta: performanceDelta,
      statistical_significance: statisticalTest
    };
  }

  // ============================================================================
  // STATISTICAL ANALYSIS AND RANKING
  // ============================================================================

  private async performStatisticalAnalysis(
    identifiedGaps: GapRecord[]
  ): Promise<SliceMiningResult['statistical_summary']> {
    console.log(`Performing statistical analysis on ${identifiedGaps.length} identified gaps`);

    // Apply multiple testing correction
    const correctedGaps = await this.statisticalAnalyzer.applyMultipleTestingCorrection(
      identifiedGaps,
      'holm' // Holm-Bonferroni correction
    );

    // Calculate aggregate statistics
    const totalSlicesAnalyzed = identifiedGaps.length * 2; // Rough estimate
    const significantGapsFound = correctedGaps.filter(gap => gap.statistical_separation.is_significant).length;
    const effectSizes = correctedGaps.map(gap => gap.statistical_separation.effect_size);
    const averageEffectSize = effectSizes.reduce((sum, es) => sum + es, 0) / effectSizes.length;

    return {
      total_slices_analyzed: totalSlicesAnalyzed,
      significant_gaps_found: significantGapsFound,
      average_effect_size: averageEffectSize,
      multiple_testing_correction: 'holm'
    };
  }

  private async buildPrioritizedTuningQueue(
    identifiedGaps: GapRecord[],
    statisticalSummary: SliceMiningResult['statistical_summary']
  ): Promise<PrioritizedTuningQueue> {
    console.log(`Building prioritized tuning queue from ${identifiedGaps.length} gaps`);

    // Filter to statistically significant gaps only
    const significantGaps = identifiedGaps.filter(gap => gap.statistical_separation.is_significant);

    // Rank by priority score
    const rankedGaps = significantGaps.sort((a, b) => b.priority_score - a.priority_score);

    // Create queue items
    const queueItems: TuningQueueItem[] = rankedGaps.map((gap, index) => ({
      queue_position: index + 1,
      gap_record_id: gap.slice_id,
      priority_score: gap.priority_score,
      estimated_tuning_time: this.estimateTuningTime(gap),
      estimated_validation_time: this.estimateValidationTime(gap),
      computational_complexity: this.assessComputationalComplexity(gap),
      blocking_dependencies: this.identifyBlockingDependencies(gap, rankedGaps),
      resource_constraints: this.identifyResourceConstraints(gap),
      predicted_improvement: {
        p_at_5_uplift: gap.estimated_uplift,
        latency_improvement: Math.abs(gap.delta_map.latency_p95) * 0.5, // Estimate 50% improvement
        confidence: gap.statistical_separation.effect_size / 2.0 // Convert effect size to confidence
      },
      assigned_profile: this.selectTuningProfile(gap),
      status: 'queued'
    }));

    // Calculate resource requirements
    const resourceRequirements = this.calculateResourceRequirements(queueItems);
    const totalEstimatedTime = queueItems.reduce((sum, item) => sum + item.estimated_tuning_time + item.estimated_validation_time, 0);

    return {
      queue_items: queueItems,
      total_estimated_time: totalEstimatedTime,
      resource_requirements: resourceRequirements,
      queue_created: Date.now(),
      expected_completion: Date.now() + (totalEstimatedTime * 60 * 1000) // Convert minutes to ms
    };
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  private collectAllSlices(stratificationResults: SliceMiningResult['stratification_dimensions']): SliceGroup[] {
    const allSlices: SliceGroup[] = [];

    // Collect from language stratification
    allSlices.push(stratificationResults.language_distribution.pure_english);
    allSlices.push(stratificationResults.language_distribution.pure_chinese);
    allSlices.push(stratificationResults.language_distribution.code_switch_mixed);
    allSlices.push(...Object.values(stratificationResults.language_distribution.programming_heavy));

    // Collect from content type stratification
    allSlices.push(stratificationResults.content_type_mix.code_heavy);
    allSlices.push(stratificationResults.content_type_mix.error_heavy);
    allSlices.push(stratificationResults.content_type_mix.tool_heavy);
    allSlices.push(stratificationResults.content_type_mix.json_needle);
    allSlices.push(stratificationResults.content_type_mix.prose_dominant);
    allSlices.push(stratificationResults.content_type_mix.mixed_content);

    // Collect from complexity stratification
    allSlices.push(stratificationResults.complexity_bins.low_complexity);
    allSlices.push(stratificationResults.complexity_bins.medium_complexity);
    allSlices.push(stratificationResults.complexity_bins.high_complexity);
    allSlices.push(stratificationResults.complexity_bins.extreme_complexity);

    // Collect from stability stratification
    allSlices.push(...stratificationResults.stability_deciles.deciles);
    allSlices.push(stratificationResults.stability_deciles.unstable_outliers);
    allSlices.push(stratificationResults.stability_deciles.highly_stable);

    // Remove duplicates based on slice_ids overlap
    return this.deduplicateSlices(allSlices);
  }

  private deduplicateSlices(slices: SliceGroup[]): SliceGroup[] {
    const uniqueSlices = new Map<string, SliceGroup>();
    
    for (const slice of slices) {
      const key = slice.slice_ids.sort().join(','); // Create key from sorted slice IDs
      if (!uniqueSlices.has(key) || uniqueSlices.get(key)!.sample_size < slice.sample_size) {
        uniqueSlices.set(key, slice);
      }
    }
    
    return Array.from(uniqueSlices.values());
  }

  private findBestCompetitorBaseline(slice: SliceGroup, baselines: CompetitorBaseline[]): CompetitorBaseline | null {
    // Find baseline that best matches slice characteristics
    return baselines
      .filter(b => this.baselineMatchesSlice(b, slice))
      .sort((a, b) => b.p_at_5 - a.p_at_5)[0] || null;
  }

  private baselineMatchesSlice(baseline: CompetitorBaseline, slice: SliceGroup): boolean {
    // Simple matching logic - in practice would be more sophisticated
    return baseline.dataset_type === 'general' || slice.group_name.includes(baseline.dataset_type);
  }

  private async performStatisticalSignificanceTest(
    slice: SliceGroup,
    baseline: CompetitorBaseline
  ): Promise<StatisticalSignificanceResult> {
    // Simplified statistical test - in practice would use proper paired tests
    const effectSize = Math.abs(slice.performance_baseline.mean_p_at_5 - baseline.p_at_5) / 
                      Math.max(slice.performance_baseline.std_p_at_5, 0.01);
    
    const isSignificant = effectSize > 0.5 && slice.sample_size >= 30;
    const pValue = isSignificant ? 0.01 : 0.2;

    return {
      is_significant: isSignificant,
      p_value: pValue,
      effect_size: effectSize,
      confidence_interval: [
        slice.performance_baseline.mean_p_at_5 - baseline.p_at_5 - 0.01,
        slice.performance_baseline.mean_p_at_5 - baseline.p_at_5 + 0.01
      ]
    };
  }

  private async createGapRecordFromSlice(
    slice: SliceGroup,
    baseline: CompetitorBaseline,
    performanceDelta: number,
    latencyDelta: number,
    statisticalTest: StatisticalSignificanceResult
  ): Promise<GapRecord> {
    // Extract slice characteristics to create gap record
    const sliceId = this.generateSliceId(slice.group_id);
    
    return {
      slice_id: sliceId,
      dataset: this.extractDatasetFromSlice(slice),
      keep_ratio: this.extractKeepRatioFromSlice(slice),
      k: this.extractKFromSlice(slice),
      seed: Math.floor(Math.random() * 10000), // Would be deterministic in practice
      delta_map: {
        macro_p_at_5: performanceDelta,
        cost_per_query: 0.01, // Estimated
        latency_p95: latencyDelta,
        latency_p99_p95_ratio: 0.1 // Estimated
      },
      root_cause_features: slice.feature_profile,
      policy_fingerprint: this.createDefaultPolicyFingerprint(),
      statistical_separation: statisticalTest,
      priority_score: this.calculatePriorityScore(performanceDelta, latencyDelta, statisticalTest.effect_size),
      estimated_uplift: Math.abs(performanceDelta) * 0.7, // Assume 70% of gap is closable
      created_at: Date.now(),
      updated_at: Date.now(),
      validation_runs: 0,
      status: 'identified'
    };
  }

  private estimateTuningTime(gap: GapRecord): number {
    // Base time in minutes
    const baseTime = 15;
    
    // Complexity multipliers
    const complexityMultiplier = gap.root_cause_features.entity_entropy > 2.0 ? 1.5 : 1.0;
    const gapMagnitudeMultiplier = Math.abs(gap.delta_map.macro_p_at_5) > 0.05 ? 1.3 : 1.0;
    
    return baseTime * complexityMultiplier * gapMagnitudeMultiplier;
  }

  private estimateValidationTime(gap: GapRecord): number {
    // Validation time is typically shorter than tuning
    return this.estimateTuningTime(gap) * 0.6;
  }

  private assessComputationalComplexity(gap: GapRecord): 'low' | 'medium' | 'high' {
    const complexityScore = gap.root_cause_features.entity_entropy + 
                          gap.root_cause_features.closure_depth * 0.5 +
                          (1 - gap.root_cause_features.kv_stability);
    
    if (complexityScore < 2.0) return 'low';
    if (complexityScore < 4.0) return 'medium';
    return 'high';
  }

  private identifyBlockingDependencies(gap: GapRecord, allGaps: GapRecord[]): string[] {
    // Simple dependency logic - gaps in same dataset should be processed in order
    return allGaps
      .filter(g => g.dataset === gap.dataset && g.priority_score > gap.priority_score)
      .map(g => g.slice_id)
      .slice(0, 2); // Limit to 2 dependencies max
  }

  private identifyResourceConstraints(gap: GapRecord): string[] {
    const constraints: string[] = [];
    
    if (gap.root_cause_features.entity_entropy > 3.0) {
      constraints.push('High memory required for entity processing');
    }
    
    if (gap.root_cause_features.closure_depth > 5.0) {
      constraints.push('GPU recommended for deep closure analysis');
    }
    
    return constraints;
  }

  private selectTuningProfile(gap: GapRecord): string {
    const features = gap.root_cause_features;
    
    if (features.type_mix.code_heavy > 0.4 || features.type_mix.error_heavy > 0.3) {
      return 'code_error_gaps';
    }
    
    if (features.type_mix.tool_heavy > 0.4 || features.type_mix.json_needle > 0.2) {
      return 'tool_json_needles';
    }
    
    if (features.language_distribution.code_switch > 0.2) {
      return 'multilingual_codeswitch';
    }
    
    return 'general';
  }

  private calculateResourceRequirements(queueItems: TuningQueueItem[]): PrioritizedTuningQueue['resource_requirements'] {
    const highComplexityItems = queueItems.filter(item => item.computational_complexity === 'high').length;
    
    return {
      cpu_cores_needed: Math.max(4, queueItems.length / 2),
      memory_gb_needed: 8 + (highComplexityItems * 4),
      gpu_required: highComplexityItems > 0
    };
  }

  private generateMiningRunId(): string {
    return `mining_${Date.now()}_${Math.random().toString(36).substr(2, 8)}`;
  }

  private generateSliceId(groupId: string): string {
    return createHash('md5').update(groupId).digest('hex').substring(0, 16);
  }

  private countTotalSlices(stratificationResults: SliceMiningResult['stratification_dimensions']): number {
    // Count unique slices across all stratification dimensions
    const allSlices = this.collectAllSlices(stratificationResults);
    return allSlices.length;
  }

  private estimateMemoryUsage(resultCount: number): number {
    // Rough estimate: 1MB per 1000 results
    return Math.max(1, resultCount / 1000);
  }

  private extractDatasetFromSlice(slice: SliceGroup): string {
    // Extract dataset from slice group name
    const match = slice.group_name.match(/(\w+)_dataset/);
    return match ? match[1] : 'unknown';
  }

  private extractKeepRatioFromSlice(slice: SliceGroup): number {
    // Default keep ratio
    return 0.15;
  }

  private extractKFromSlice(slice: SliceGroup): number {
    // Default K value
    return 10;
  }

  private createDefaultPolicyFingerprint(): GapRecord['policy_fingerprint'] {
    return {
      lambda: 1.0,
      mu: 1.0,
      K2: 100,
      r: 16,
      head_keep: 80,
      window_size: 1024,
      stride: 512,
      ce_early_exit_rate: 0.1,
      tau: 0.5,
      curvature_threshold: 0.1,
      proxy_gap_max: 0.005,
      policy_id: 'baseline_policy',
      created_at: Date.now(),
      validation_status: 'pending'
    };
  }

  private calculatePriorityScore(performanceDelta: number, latencyDelta: number, effectSize: number): number {
    // Weighted combination of factors
    return Math.abs(performanceDelta) * 0.5 + effectSize * 0.3 + (Math.abs(latencyDelta) / 100) * 0.2;
  }
}

// ============================================================================
// STRATIFICATION DIMENSIONS IMPLEMENTATION
// ============================================================================

export class StratificationDimensions {
  /**
   * Stratifies results by dataset, budget (keep_ratio), and k
   */
  stratifyByDatasetBudgetK(results: EvaluationResult[]): Array<{
    dataset: string;
    keep_ratio: number;
    k: number;
    slice_count: number;
  }> {
    const combinations = new Map<string, number>();
    
    results.forEach(result => {
      const key = `${result.dataset}-${result.keep_ratio}-${result.k}`;
      combinations.set(key, (combinations.get(key) || 0) + 1);
    });
    
    return Array.from(combinations.entries()).map(([key, count]) => {
      const [dataset, keepRatio, k] = key.split('-');
      return {
        dataset,
        keep_ratio: parseFloat(keepRatio),
        k: parseInt(k),
        slice_count: count
      };
    });
  }

  /**
   * Stratifies results by language characteristics
   */
  stratifyByLanguage(results: EvaluationResult[]): LanguageStratification {
    const pureEnglish: string[] = [];
    const pureChinese: string[] = [];
    const codeSwitchMixed: string[] = [];
    const programmingHeavy: Record<string, string[]> = {};

    results.forEach(result => {
      const langProfile = this.analyzeLanguageProfile(result);
      
      if (langProfile.pure_english) {
        pureEnglish.push(result.result_id);
      } else if (langProfile.pure_chinese) {
        pureChinese.push(result.result_id);
      } else if (langProfile.code_switch) {
        codeSwitchMixed.push(result.result_id);
      }
      
      langProfile.programming_languages.forEach(lang => {
        if (!programmingHeavy[lang]) programmingHeavy[lang] = [];
        programmingHeavy[lang].push(result.result_id);
      });
    });

    return {
      pure_english: this.createSliceGroup('pure_english', pureEnglish, results),
      pure_chinese: this.createSliceGroup('pure_chinese', pureChinese, results),
      code_switch_mixed: this.createSliceGroup('code_switch_mixed', codeSwitchMixed, results),
      programming_heavy: Object.fromEntries(
        Object.entries(programmingHeavy).map(([lang, ids]) => [
          lang,
          this.createSliceGroup(`programming_${lang}`, ids, results)
        ])
      )
    };
  }

  /**
   * Stratifies results by content type mix
   */
  stratifyByContentType(results: EvaluationResult[]): ContentTypeStratification {
    const codeHeavy: string[] = [];
    const errorHeavy: string[] = [];
    const toolHeavy: string[] = [];
    const jsonNeedle: string[] = [];
    const proseDominant: string[] = [];
    const mixedContent: string[] = [];

    results.forEach(result => {
      const typeProfile = this.analyzeContentTypeProfile(result);
      
      if (typeProfile.code_heavy > 0.5) {
        codeHeavy.push(result.result_id);
      } else if (typeProfile.error_heavy > 0.4) {
        errorHeavy.push(result.result_id);
      } else if (typeProfile.tool_heavy > 0.4) {
        toolHeavy.push(result.result_id);
      } else if (typeProfile.json_needle > 0.3) {
        jsonNeedle.push(result.result_id);
      } else if (typeProfile.prose_heavy > 0.6) {
        proseDominant.push(result.result_id);
      } else {
        mixedContent.push(result.result_id);
      }
    });

    return {
      code_heavy: this.createSliceGroup('code_heavy', codeHeavy, results),
      error_heavy: this.createSliceGroup('error_heavy', errorHeavy, results),
      tool_heavy: this.createSliceGroup('tool_heavy', toolHeavy, results),
      json_needle: this.createSliceGroup('json_needle', jsonNeedle, results),
      prose_dominant: this.createSliceGroup('prose_dominant', proseDominant, results),
      mixed_content: this.createSliceGroup('mixed_content', mixedContent, results)
    };
  }

  /**
   * Stratifies results by complexity characteristics
   */
  stratifyByComplexity(results: EvaluationResult[]): ComplexityBinStratification {
    const lowComplexity: string[] = [];
    const mediumComplexity: string[] = [];
    const highComplexity: string[] = [];
    const extremeComplexity: string[] = [];

    results.forEach(result => {
      const complexityScore = this.calculateComplexityScore(result);
      
      if (complexityScore < 1.0) {
        lowComplexity.push(result.result_id);
      } else if (complexityScore < 2.5) {
        mediumComplexity.push(result.result_id);
      } else if (complexityScore < 4.0) {
        highComplexity.push(result.result_id);
      } else {
        extremeComplexity.push(result.result_id);
      }
    });

    return {
      low_complexity: this.createSliceGroup('low_complexity', lowComplexity, results),
      medium_complexity: this.createSliceGroup('medium_complexity', mediumComplexity, results),
      high_complexity: this.createSliceGroup('high_complexity', highComplexity, results),
      extreme_complexity: this.createSliceGroup('extreme_complexity', extremeComplexity, results)
    };
  }

  /**
   * Stratifies results by KV stability deciles
   */
  stratifyByKVStability(results: EvaluationResult[]): StabilityDecileStratification {
    // Sort by KV stability score
    const sortedResults = results
      .map(r => ({ result: r, stability: this.calculateKVStability(r) }))
      .sort((a, b) => a.stability - b.stability);

    const decileSize = Math.floor(sortedResults.length / 10);
    const deciles: SliceGroup[] = [];

    for (let i = 0; i < 10; i++) {
      const start = i * decileSize;
      const end = i === 9 ? sortedResults.length : (i + 1) * decileSize;
      const decileResults = sortedResults.slice(start, end);
      
      deciles.push(this.createSliceGroup(
        `stability_decile_${i + 1}`,
        decileResults.map(r => r.result.result_id),
        results
      ));
    }

    // Bottom 5% most unstable
    const unstableCount = Math.floor(sortedResults.length * 0.05);
    const unstableOutliers = this.createSliceGroup(
      'unstable_outliers',
      sortedResults.slice(0, unstableCount).map(r => r.result.result_id),
      results
    );

    // Top 5% most stable  
    const stableCount = Math.floor(sortedResults.length * 0.05);
    const highlyStable = this.createSliceGroup(
      'highly_stable',
      sortedResults.slice(-stableCount).map(r => r.result.result_id),
      results
    );

    return {
      deciles,
      unstable_outliers: unstableOutliers,
      highly_stable: highlyStable
    };
  }

  // ============================================================================
  // HELPER METHODS
  // ============================================================================

  private analyzeLanguageProfile(result: EvaluationResult): {
    pure_english: boolean;
    pure_chinese: boolean;
    code_switch: boolean;
    programming_languages: string[];
  } {
    // Simplified language analysis based on result content
    const hasEnglish = /[a-zA-Z]/.test(result.query || '');
    const hasChinese = /[\u4e00-\u9fff]/.test(result.query || '');
    
    const programmingLanguages: string[] = [];
    if (/function|class|def|import/.test(result.query || '')) {
      programmingLanguages.push('javascript', 'python');
    }
    
    return {
      pure_english: hasEnglish && !hasChinese,
      pure_chinese: hasChinese && !hasEnglish,
      code_switch: hasEnglish && hasChinese,
      programming_languages
    };
  }

  private analyzeContentTypeProfile(result: EvaluationResult): {
    code_heavy: number;
    error_heavy: number;
    tool_heavy: number;
    json_needle: number;
    prose_heavy: number;
  } {
    const content = result.query || '';
    
    return {
      code_heavy: /```|function|class|def/.test(content) ? 0.8 : 0.2,
      error_heavy: /error|exception|traceback/.test(content) ? 0.7 : 0.1,
      tool_heavy: /tool|command|executed/.test(content) ? 0.6 : 0.1,
      json_needle: /{[\s\S]*}/.test(content) ? 0.5 : 0.1,
      prose_heavy: content.length > 100 && !/```/.test(content) ? 0.7 : 0.3
    };
  }

  private calculateComplexityScore(result: EvaluationResult): number {
    const content = result.query || '';
    let score = 0;
    
    // Entity density
    const entityCount = (content.match(/\b[A-Z][a-z]+\b/g) || []).length;
    score += entityCount * 0.1;
    
    // Nesting depth
    const nestingDepth = Math.max(
      (content.match(/[\[{(]/g) || []).length,
      (content.split('.').length - 1)
    );
    score += nestingDepth * 0.3;
    
    // Length complexity
    score += Math.min(2.0, content.length / 500);
    
    return score;
  }

  private calculateKVStability(result: EvaluationResult): number {
    // Simplified KV stability calculation
    const content = result.query || '';
    const kvPatterns = (content.match(/\w+\s*[:=]\s*\w+/g) || []).length;
    const totalTokens = content.split(/\s+/).length;
    
    return totalTokens > 0 ? kvPatterns / totalTokens : 0.5;
  }

  private createSliceGroup(
    groupName: string,
    sliceIds: string[],
    allResults: EvaluationResult[]
  ): SliceGroup {
    const sliceResults = allResults.filter(r => sliceIds.includes(r.result_id));
    
    // Calculate aggregate statistics
    const p5Values = sliceResults.map(r => r.macro_p_at_5 || 0);
    const latencyValues = sliceResults.map(r => r.latency_p95 || 100);
    
    const meanP5 = p5Values.reduce((sum, v) => sum + v, 0) / Math.max(1, p5Values.length);
    const stdP5 = Math.sqrt(p5Values.reduce((sum, v) => sum + (v - meanP5) ** 2, 0) / Math.max(1, p5Values.length));
    const meanLatency = latencyValues.reduce((sum, v) => sum + v, 0) / Math.max(1, latencyValues.length);
    const stdLatency = Math.sqrt(latencyValues.reduce((sum, v) => sum + (v - meanLatency) ** 2, 0) / Math.max(1, latencyValues.length));

    // Calculate representative features
    const typicalFeatures = this.calculateTypicalFeatures(sliceResults);

    return {
      group_id: `slice_${groupName}_${Date.now()}`,
      group_name: groupName,
      slice_ids: sliceIds,
      sample_size: sliceResults.length,
      performance_baseline: {
        mean_p_at_5: meanP5,
        std_p_at_5: stdP5,
        mean_latency_p95: meanLatency,
        std_latency_p95: stdLatency
      },
      gap_summary: {
        significant_gaps_count: 0, // Will be filled in later
        average_gap_magnitude: 0,
        priority_score_range: [0, 0]
      },
      feature_profile: typicalFeatures
    };
  }

  private calculateTypicalFeatures(results: EvaluationResult[]): SliceGroup['feature_profile'] {
    if (results.length === 0) {
      return {
        typical_entity_entropy: 0,
        typical_dup_rate: 0,
        typical_closure_depth: 0,
        dominant_type_mix: {
          code_heavy: 0,
          error_heavy: 0,
          tool_heavy: 0,
          prose_heavy: 0,
          json_needle: 0
        }
      };
    }

    // Calculate typical values across all results in this slice
    const entityEntropies = results.map(r => this.calculateEntityEntropy(r.query || ''));
    const dupRates = results.map(r => this.calculateDuplicationRate(r));
    const closureDepths = results.map(r => this.calculateClosureDepth(r.query || ''));
    
    const typicalEntityEntropy = entityEntropies.reduce((sum, v) => sum + v, 0) / entityEntropies.length;
    const typicalDupRate = dupRates.reduce((sum, v) => sum + v, 0) / dupRates.length;
    const typicalClosureDepth = closureDepths.reduce((sum, v) => sum + v, 0) / closureDepths.length;
    
    // Calculate dominant type mix
    const typeProfiles = results.map(r => this.analyzeContentTypeProfile(r));
    const dominantTypeMix = {
      code_heavy: typeProfiles.reduce((sum, p) => sum + p.code_heavy, 0) / typeProfiles.length,
      error_heavy: typeProfiles.reduce((sum, p) => sum + p.error_heavy, 0) / typeProfiles.length,
      tool_heavy: typeProfiles.reduce((sum, p) => sum + p.tool_heavy, 0) / typeProfiles.length,
      prose_heavy: typeProfiles.reduce((sum, p) => sum + p.prose_heavy, 0) / typeProfiles.length,
      json_needle: typeProfiles.reduce((sum, p) => sum + p.json_needle, 0) / typeProfiles.length
    };

    return {
      typical_entity_entropy: typicalEntityEntropy,
      typical_dup_rate: typicalDupRate,
      typical_closure_depth: typicalClosureDepth,
      dominant_type_mix: dominantTypeMix
    };
  }

  private calculateEntityEntropy(text: string): number {
    const entities = (text.match(/\b[A-Z][a-z]+\b/g) || []);
    if (entities.length === 0) return 0;

    const frequencies = new Map<string, number>();
    entities.forEach(entity => {
      frequencies.set(entity, (frequencies.get(entity) || 0) + 1);
    });

    let entropy = 0;
    frequencies.forEach(freq => {
      const p = freq / entities.length;
      entropy -= p * Math.log2(p);
    });

    return entropy;
  }

  private calculateDuplicationRate(result: EvaluationResult): number {
    // Simplified duplication rate calculation
    const content = result.query || '';
    const words = content.split(/\s+/);
    const uniqueWords = new Set(words);
    
    return words.length > 0 ? 1 - (uniqueWords.size / words.length) : 0;
  }

  private calculateClosureDepth(text: string): number {
    const openBrackets = (text.match(/[\[{(]/g) || []).length;
    const closeBrackets = (text.match(/[\]})]/) || []).length;
    return Math.min(openBrackets, closeBrackets);
  }
}

// ============================================================================
// STATISTICAL ANALYZER
// ============================================================================

export class StatisticalAnalyzer {
  /**
   * Applies multiple testing correction to identified gaps
   */
  async applyMultipleTestingCorrection(
    gaps: GapRecord[],
    correctionMethod: 'holm' | 'bonferroni' | 'fdr'
  ): Promise<GapRecord[]> {
    console.log(`Applying ${correctionMethod} correction to ${gaps.length} gaps`);

    const pValues = gaps.map(gap => gap.statistical_separation.p_value);
    let correctedPValues: number[];

    switch (correctionMethod) {
      case 'holm':
        correctedPValues = this.applyHolmCorrection(pValues);
        break;
      case 'bonferroni':
        correctedPValues = this.applyBonferroniCorrection(pValues);
        break;
      case 'fdr':
        correctedPValues = this.applyFDRCorrection(pValues);
        break;
      default:
        correctedPValues = pValues;
    }

    // Update gaps with corrected p-values
    return gaps.map((gap, index) => ({
      ...gap,
      statistical_separation: {
        ...gap.statistical_separation,
        p_value: correctedPValues[index],
        is_significant: correctedPValues[index] < 0.05
      }
    }));
  }

  private applyHolmCorrection(pValues: number[]): number[] {
    const indexedPValues = pValues.map((p, i) => ({ p, index: i }));
    indexedPValues.sort((a, b) => a.p - b.p);

    const corrected = new Array(pValues.length);
    const m = pValues.length;

    for (let i = 0; i < m; i++) {
      const adjustedP = indexedPValues[i].p * (m - i);
      corrected[indexedPValues[i].index] = Math.min(1.0, adjustedP);
    }

    return corrected;
  }

  private applyBonferroniCorrection(pValues: number[]): number[] {
    const m = pValues.length;
    return pValues.map(p => Math.min(1.0, p * m));
  }

  private applyFDRCorrection(pValues: number[]): number[] {
    // Benjamini-Hochberg procedure
    const indexedPValues = pValues.map((p, i) => ({ p, index: i }));
    indexedPValues.sort((a, b) => a.p - b.p);

    const corrected = new Array(pValues.length);
    const m = pValues.length;

    for (let i = m - 1; i >= 0; i--) {
      const adjustedP = indexedPValues[i].p * m / (i + 1);
      corrected[indexedPValues[i].index] = Math.min(1.0, adjustedP);
    }

    return corrected;
  }
}

// ============================================================================
// GAP RANKER
// ============================================================================

export class GapRanker {
  /**
   * Ranks gaps by performance impact and statistical significance
   */
  rankGapsByPriority(gaps: GapRecord[]): GapRecord[] {
    return gaps.sort((a, b) => {
      // Primary: Statistical significance
      if (a.statistical_separation.is_significant && !b.statistical_separation.is_significant) return -1;
      if (!a.statistical_separation.is_significant && b.statistical_separation.is_significant) return 1;

      // Secondary: Priority score
      return b.priority_score - a.priority_score;
    });
  }

  /**
   * Ranks gaps by QPS loss at fixed P@5
   */
  rankGapsByQPSLoss(gaps: GapRecord[]): GapRecord[] {
    return gaps.sort((a, b) => {
      const qpsLossA = this.calculateQPSLoss(a);
      const qpsLossB = this.calculateQPSLoss(b);
      return qpsLossB - qpsLossA; // Higher QPS loss = higher priority
    });
  }

  private calculateQPSLoss(gap: GapRecord): number {
    // QPS loss = performance deficit / latency cost
    const performanceDeficit = Math.abs(gap.delta_map.macro_p_at_5);
    const latencyCost = Math.max(1, gap.delta_map.latency_p95); // Prevent division by zero
    
    return performanceDeficit / latencyCost;
  }
}

// ============================================================================
// SUPPORTING TYPES AND INTERFACES
// ============================================================================

interface EvaluationResult {
  result_id: string;
  dataset: string;
  keep_ratio: number;
  k: number;
  query?: string;
  macro_p_at_5?: number;
  latency_p95?: number;
  candidates?: Candidate[];
}

interface CompetitorBaseline {
  competitor_name: string;
  dataset_type: string;
  p_at_5: number;
  latency_p95: number;
  cost_per_query: number;
}

interface SliceAnalysisResult {
  slice_id: string;
  hasSignificantGap: boolean;
  gapRecord?: GapRecord;
  performance_delta?: number;
  statistical_significance?: StatisticalSignificanceResult;
  reason?: string;
}

interface StatisticalSignificanceResult {
  is_significant: boolean;
  p_value: number;
  effect_size: number;
  confidence_interval: [number, number];
}