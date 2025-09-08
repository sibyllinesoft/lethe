/**
 * GapBoard v1: Delta Maps and Root-Cause Feature Analysis
 * 
 * This module implements the core GapBoard system that attaches to the validator's
 * JSONL output to compute performance deltas, root-cause features, and policy
 * fingerprints for the Gap→Tune→Verify pipeline.
 */

import {
  GapRecord,
  PolicyFingerprint,
  TypeMixProfile,
  LanguageProfile,
  GapAnalysisResult,
  GapAnalysisError,
  SliceMiningResult,
  SliceGroup
} from './types.js';

import { Config, PerformanceMetrics, Candidate, Result } from '../types.js';
import { createHash } from 'crypto';

// ============================================================================
// CORE GAPBOARD IMPLEMENTATION
// ============================================================================

export class GapBoard {
  private config: Config;
  private gapRecords: Map<string, GapRecord> = new Map();
  private competitorBaselines: Map<string, PerformanceBaseline> = new Map();
  private statisticalSignificanceThreshold = 0.05;
  
  constructor(config: Config) {
    this.config = config;
  }

  /**
   * Processes validator JSONL output to identify and analyze performance gaps
   */
  async processValidatorOutput(
    validatorJsonl: string[],
    competitorResults: CompetitorResults[]
  ): Promise<GapAnalysisResult<GapRecord[]>> {
    try {
      const validationResults = this.parseValidatorJsonl(validatorJsonl);
      const gaps: GapRecord[] = [];

      for (const result of validationResults) {
        // Generate slice ID from pairing keys
        const sliceId = this.generateSliceId(
          result.dataset,
          result.keep_ratio,
          result.k,
          result.seed
        );

        // Compute delta map against best competitor
        const deltaMap = await this.computeDeltaMap(result, competitorResults);
        
        // Extract root-cause features
        const rootCauseFeatures = await this.extractRootCauseFeatures(result);
        
        // Generate policy fingerprint
        const policyFingerprint = this.generatePolicyFingerprint(result.config);
        
        // Perform statistical validation
        const statisticalSeparation = await this.validateStatisticalSeparation(
          result,
          competitorResults
        );

        // Only create gap record if statistically significant
        if (statisticalSeparation.is_significant) {
          const gapRecord: GapRecord = {
            slice_id: sliceId,
            dataset: result.dataset,
            keep_ratio: result.keep_ratio,
            k: result.k,
            seed: result.seed,
            delta_map: deltaMap,
            root_cause_features: rootCauseFeatures,
            policy_fingerprint: policyFingerprint,
            statistical_separation: statisticalSeparation,
            priority_score: this.calculatePriorityScore(deltaMap, rootCauseFeatures),
            estimated_uplift: this.estimateUpliftPotential(deltaMap, rootCauseFeatures),
            created_at: Date.now(),
            updated_at: Date.now(),
            validation_runs: 0,
            status: 'identified'
          };

          gaps.push(gapRecord);
          this.gapRecords.set(sliceId, gapRecord);
        }
      }

      return {
        success: true,
        data: gaps
      };

    } catch (error) {
      return {
        success: false,
        error: {
          code: 'GAP_BOARD_PROCESSING_ERROR',
          message: `Failed to process validator output: ${error instanceof Error ? error.message : 'Unknown error'}`,
          error_type: 'gap_detection',
          recovery_actions: ['Verify JSONL format', 'Check competitor results format', 'Validate configuration'],
          is_retryable: true,
          impact_severity: 'high',
          affected_components: ['gap_mining'],
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Computes performance delta map comparing current results to best competitors
   */
  private async computeDeltaMap(
    result: ValidationResult,
    competitorResults: CompetitorResults[]
  ): Promise<GapRecord['delta_map']> {
    // Find best competitor for this slice
    const sliceKey = `${result.dataset}-${result.keep_ratio}-${result.k}`;
    const bestCompetitor = this.findBestCompetitor(sliceKey, competitorResults);
    
    if (!bestCompetitor) {
      throw new Error(`No competitor baseline found for slice: ${sliceKey}`);
    }

    return {
      macro_p_at_5: result.macro_p_at_5 - bestCompetitor.macro_p_at_5,
      cost_per_query: result.cost_per_query - bestCompetitor.cost_per_query,
      latency_p95: result.latency_p95 - bestCompetitor.latency_p95,
      latency_p99_p95_ratio: result.latency_p99_p95_ratio - bestCompetitor.latency_p99_p95_ratio
    };
  }

  /**
   * Extracts root-cause features for gap analysis
   */
  private async extractRootCauseFeatures(
    result: ValidationResult
  ): Promise<GapRecord['root_cause_features']> {
    // Entity entropy calculation
    const entityEntropy = this.calculateEntityEntropy(result.candidates);
    
    // Duplication rate analysis
    const dupRate = this.calculateDuplicationRate(result.candidates);
    
    // Type mix profile
    const typeMix = this.analyzeTypeMix(result.candidates);
    
    // Closure depth analysis
    const closureDepth = this.calculateAverageClosureDepth(result.candidates);
    
    // Symbol length analysis
    const symbolLengthAvg = this.calculateAverageSymbolLength(result.candidates);
    
    // Language distribution
    const languageDistribution = this.analyzeLanguageDistribution(result.candidates);
    
    // KV stability measure
    const kvStability = this.calculateKVStability(result.candidates);

    return {
      entity_entropy: entityEntropy,
      dup_rate: dupRate,
      type_mix: typeMix,
      closure_depth: closureDepth,
      symbol_length_avg: symbolLengthAvg,
      language_distribution: languageDistribution,
      kv_stability: kvStability
    };
  }

  /**
   * Generates policy fingerprint from current configuration
   */
  private generatePolicyFingerprint(config: Config): PolicyFingerprint {
    const fingerprintData = {
      lambda: config.retrieval.alpha,
      mu: config.retrieval.beta,
      K2: config.rerank.topk_in,
      r: config.diversify.pack_chunks,
      head_keep: config.retrieval.window_size || 512,
      window_size: config.retrieval.window_size || 1024,
      stride: Math.floor((config.retrieval.window_size || 1024) / 2),
      ce_early_exit_rate: 0.1, // TODO: Extract from actual config
      tau: 0.5, // TODO: Extract from actual config
      curvature_threshold: 0.1,
      proxy_gap_max: 0.005,
    };

    return {
      ...fingerprintData,
      policy_id: this.generatePolicyId(fingerprintData),
      created_at: Date.now(),
      validation_status: 'pending'
    };
  }

  /**
   * Validates statistical separation using paired bootstrap with Holm correction
   */
  private async validateStatisticalSeparation(
    result: ValidationResult,
    competitorResults: CompetitorResults[]
  ): Promise<GapRecord['statistical_separation']> {
    const sliceKey = `${result.dataset}-${result.keep_ratio}-${result.k}`;
    const bestCompetitor = this.findBestCompetitor(sliceKey, competitorResults);
    
    if (!bestCompetitor) {
      return {
        is_significant: false,
        p_value: 1.0,
        confidence_interval: [0, 0],
        effect_size: 0
      };
    }

    // Perform paired bootstrap test
    const bootstrapResults = await this.performPairedBootstrap(
      result.sample_results,
      bestCompetitor.sample_results,
      1000 // bootstrap iterations
    );

    // Apply Holm correction for multiple testing
    const correctedPValue = this.applyHolmCorrection(
      bootstrapResults.p_value,
      this.gapRecords.size + 1 // total number of tests
    );

    // Calculate effect size (Cohen's d)
    const effectSize = this.calculateCohensD(
      result.sample_results,
      bestCompetitor.sample_results
    );

    return {
      is_significant: correctedPValue < this.statisticalSignificanceThreshold,
      p_value: correctedPValue,
      confidence_interval: bootstrapResults.confidence_interval,
      effect_size: effectSize
    };
  }

  // ============================================================================
  // FEATURE EXTRACTION METHODS
  // ============================================================================

  /**
   * Calculates entity entropy as information density measure
   */
  private calculateEntityEntropy(candidates: Candidate[]): number {
    const entityFrequencies = new Map<string, number>();
    let totalEntities = 0;

    // Extract entities using simple regex patterns
    candidates.forEach(candidate => {
      const entities = this.extractEntities(candidate.text);
      entities.forEach(entity => {
        entityFrequencies.set(entity, (entityFrequencies.get(entity) || 0) + 1);
        totalEntities++;
      });
    });

    if (totalEntities === 0) return 0;

    // Calculate Shannon entropy
    let entropy = 0;
    entityFrequencies.forEach(frequency => {
      const probability = frequency / totalEntities;
      entropy -= probability * Math.log2(probability);
    });

    return entropy;
  }

  /**
   * Calculates content duplication rate
   */
  private calculateDuplicationRate(candidates: Candidate[]): number {
    const textHashes = new Set<string>();
    let duplicateCount = 0;

    candidates.forEach(candidate => {
      const textHash = createHash('md5').update(candidate.text).digest('hex');
      if (textHashes.has(textHash)) {
        duplicateCount++;
      } else {
        textHashes.add(textHash);
      }
    });

    return candidates.length > 0 ? duplicateCount / candidates.length : 0;
  }

  /**
   * Analyzes type mix profile
   */
  private analyzeTypeMix(candidates: Candidate[]): TypeMixProfile {
    const typeCounts = {
      code_heavy: 0,
      error_heavy: 0,
      tool_heavy: 0,
      prose_heavy: 0,
      json_needle: 0
    };

    candidates.forEach(candidate => {
      if (this.isCodeHeavy(candidate)) typeCounts.code_heavy++;
      if (this.isErrorHeavy(candidate)) typeCounts.error_heavy++;
      if (this.isToolHeavy(candidate)) typeCounts.tool_heavy++;
      if (this.isProseHeavy(candidate)) typeCounts.prose_heavy++;
      if (this.containsJsonNeedle(candidate)) typeCounts.json_needle++;
    });

    const total = candidates.length || 1;
    return {
      code_heavy: typeCounts.code_heavy / total,
      error_heavy: typeCounts.error_heavy / total,
      tool_heavy: typeCounts.tool_heavy / total,
      prose_heavy: typeCounts.prose_heavy / total,
      json_needle: typeCounts.json_needle / total
    };
  }

  /**
   * Calculates average closure depth for code content
   */
  private calculateAverageClosureDepth(candidates: Candidate[]): number {
    const depths = candidates
      .filter(c => c.kind === 'code' || c.kind === 'user_code')
      .map(c => this.calculateClosureDepth(c.text))
      .filter(d => d > 0);

    return depths.length > 0 ? depths.reduce((a, b) => a + b, 0) / depths.length : 0;
  }

  /**
   * Calculates average symbol length complexity
   */
  private calculateAverageSymbolLength(candidates: Candidate[]): number {
    const symbolLengths = candidates
      .map(c => this.extractSymbols(c.text))
      .flat()
      .map(s => s.length)
      .filter(l => l > 0);

    return symbolLengths.length > 0 ? symbolLengths.reduce((a, b) => a + b, 0) / symbolLengths.length : 0;
  }

  /**
   * Analyzes language distribution including code-switching
   */
  private analyzeLanguageDistribution(candidates: Candidate[]): LanguageProfile {
    let english = 0, chinese = 0, codeSwitch = 0;
    const programmingLanguages: Record<string, number> = {};

    candidates.forEach(candidate => {
      const langAnalysis = this.analyzeLanguage(candidate.text);
      
      if (langAnalysis.isEnglish) english++;
      if (langAnalysis.isChinese) chinese++;
      if (langAnalysis.isCodeSwitch) codeSwitch++;
      
      langAnalysis.programmingLanguages.forEach(lang => {
        programmingLanguages[lang] = (programmingLanguages[lang] || 0) + 1;
      });
    });

    const total = candidates.length || 1;
    
    // Normalize programming language counts
    Object.keys(programmingLanguages).forEach(lang => {
      programmingLanguages[lang] /= total;
    });

    return {
      english: english / total,
      chinese: chinese / total,
      code_switch: codeSwitch / total,
      programming_languages: programmingLanguages
    };
  }

  /**
   * Calculates KV prefix-Jaccard stability measure
   */
  private calculateKVStability(candidates: Candidate[]): number {
    // Extract key-value prefixes from candidates
    const kvPrefixes = candidates.map(c => this.extractKVPrefixes(c.text));
    
    if (kvPrefixes.length < 2) return 1.0;

    // Calculate pairwise Jaccard similarities
    const similarities: number[] = [];
    for (let i = 0; i < kvPrefixes.length - 1; i++) {
      for (let j = i + 1; j < kvPrefixes.length; j++) {
        similarities.push(this.calculateJaccardSimilarity(kvPrefixes[i], kvPrefixes[j]));
      }
    }

    return similarities.length > 0 ? similarities.reduce((a, b) => a + b, 0) / similarities.length : 1.0;
  }

  // ============================================================================
  // STATISTICAL ANALYSIS METHODS
  // ============================================================================

  /**
   * Performs paired bootstrap test for statistical significance
   */
  private async performPairedBootstrap(
    sample1: number[],
    sample2: number[],
    iterations: number
  ): Promise<{ p_value: number; confidence_interval: [number, number] }> {
    if (sample1.length !== sample2.length) {
      throw new Error('Sample sizes must be equal for paired bootstrap test');
    }

    // Calculate observed difference
    const observedDiff = this.calculateMeanDifference(sample1, sample2);
    
    // Bootstrap resampling
    const bootstrapDiffs: number[] = [];
    for (let i = 0; i < iterations; i++) {
      const indices = this.generateBootstrapIndices(sample1.length);
      const resample1 = indices.map(idx => sample1[idx]);
      const resample2 = indices.map(idx => sample2[idx]);
      bootstrapDiffs.push(this.calculateMeanDifference(resample1, resample2));
    }

    // Calculate p-value (two-tailed test)
    const nullDiffs = bootstrapDiffs.map(diff => diff - observedDiff);
    const pValue = (nullDiffs.filter(diff => Math.abs(diff) >= Math.abs(observedDiff)).length) / iterations;

    // Calculate 95% confidence interval
    bootstrapDiffs.sort((a, b) => a - b);
    const ciLower = bootstrapDiffs[Math.floor(iterations * 0.025)];
    const ciUpper = bootstrapDiffs[Math.floor(iterations * 0.975)];

    return {
      p_value: pValue,
      confidence_interval: [ciLower, ciUpper]
    };
  }

  /**
   * Applies Holm-Bonferroni correction for multiple testing
   */
  private applyHolmCorrection(pValue: number, totalTests: number): number {
    // For simplicity, using Bonferroni correction
    // In a full implementation, you would track all p-values and apply true Holm correction
    return Math.min(1.0, pValue * totalTests);
  }

  /**
   * Calculates Cohen's d effect size
   */
  private calculateCohensD(sample1: number[], sample2: number[]): number {
    const mean1 = sample1.reduce((a, b) => a + b, 0) / sample1.length;
    const mean2 = sample2.reduce((a, b) => a + b, 0) / sample2.length;
    
    const variance1 = sample1.reduce((a, b) => a + Math.pow(b - mean1, 2), 0) / (sample1.length - 1);
    const variance2 = sample2.reduce((a, b) => a + Math.pow(b - mean2, 2), 0) / (sample2.length - 1);
    
    const pooledStd = Math.sqrt((variance1 + variance2) / 2);
    
    return pooledStd === 0 ? 0 : (mean1 - mean2) / pooledStd;
  }

  // ============================================================================
  // UTILITY METHODS
  // ============================================================================

  /**
   * Generates unique slice ID from pairing keys
   */
  private generateSliceId(dataset: string, keepRatio: number, k: number, seed: number): string {
    const data = `${dataset}-${keepRatio}-${k}-${seed}`;
    return createHash('sha256').update(data).digest('hex').substring(0, 16);
  }

  /**
   * Generates policy ID from fingerprint data
   */
  private generatePolicyId(fingerprint: Omit<PolicyFingerprint, 'policy_id' | 'created_at' | 'validation_status'>): string {
    const data = JSON.stringify(fingerprint, Object.keys(fingerprint).sort());
    return createHash('sha256').update(data).digest('hex').substring(0, 16);
  }

  /**
   * Finds best competitor for given slice
   */
  private findBestCompetitor(sliceKey: string, competitorResults: CompetitorResults[]): CompetitorResults | null {
    return competitorResults
      .filter(r => `${r.dataset}-${r.keep_ratio}-${r.k}` === sliceKey)
      .sort((a, b) => b.macro_p_at_5 - a.macro_p_at_5)[0] || null;
  }

  /**
   * Calculates priority score for tuning queue
   */
  private calculatePriorityScore(
    deltaMap: GapRecord['delta_map'],
    rootCauseFeatures: GapRecord['root_cause_features']
  ): number {
    // Weighted combination of gap magnitude and improvement potential
    const performanceGapWeight = Math.abs(deltaMap.macro_p_at_5) * 0.4;
    const latencyGapWeight = Math.abs(deltaMap.latency_p95) * 0.3;
    const complexityWeight = rootCauseFeatures.entity_entropy * 0.2;
    const stabilityWeight = (1 - rootCauseFeatures.kv_stability) * 0.1;
    
    return performanceGapWeight + latencyGapWeight + complexityWeight + stabilityWeight;
  }

  /**
   * Estimates uplift potential for a gap
   */
  private estimateUpliftPotential(
    deltaMap: GapRecord['delta_map'],
    rootCauseFeatures: GapRecord['root_cause_features']
  ): number {
    // Heuristic estimate based on gap size and tuning complexity
    const baseUplift = Math.abs(deltaMap.macro_p_at_5) * 0.5; // Assume 50% of gap is closable
    const complexityMultiplier = 1 + (rootCauseFeatures.entity_entropy * 0.1);
    
    return baseUplift * complexityMultiplier;
  }

  // Helper methods for feature extraction (simplified implementations)
  private extractEntities(text: string): string[] {
    // Simple entity extraction using regex patterns
    const entityPatterns = [
      /\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b/g, // Proper nouns
      /\b\w+\.\w+\b/g, // Method calls
      /\b[A-Z_][A-Z0-9_]*\b/g // Constants
    ];
    
    return entityPatterns
      .flatMap(pattern => text.match(pattern) || [])
      .filter(entity => entity.length > 2);
  }

  private isCodeHeavy(candidate: Candidate): boolean {
    return candidate.kind === 'code' || candidate.kind === 'user_code' ||
           /```|function|class|def |import |#include/.test(candidate.text);
  }

  private isErrorHeavy(candidate: Candidate): boolean {
    return /error|exception|traceback|stack trace|failed|undefined/i.test(candidate.text);
  }

  private isToolHeavy(candidate: Candidate): boolean {
    return candidate.kind === 'tool_result' || /tool|command|executed|output/.test(candidate.text);
  }

  private isProseHeavy(candidate: Candidate): boolean {
    return candidate.kind === 'prose' && !/```|function|class/.test(candidate.text);
  }

  private containsJsonNeedle(candidate: Candidate): boolean {
    return /{[\s\S]*}/.test(candidate.text) && candidate.text.includes(':');
  }

  private calculateClosureDepth(text: string): number {
    const openBrackets = (text.match(/[\[{(]/g) || []).length;
    const closeBrackets = (text.match(/[\]})]/) || []).length;
    return Math.min(openBrackets, closeBrackets);
  }

  private extractSymbols(text: string): string[] {
    return text.match(/\b[a-zA-Z_][a-zA-Z0-9_]*\b/g) || [];
  }

  private analyzeLanguage(text: string): {
    isEnglish: boolean;
    isChinese: boolean;
    isCodeSwitch: boolean;
    programmingLanguages: string[];
  } {
    const hasEnglish = /[a-zA-Z]/.test(text);
    const hasChinese = /[\u4e00-\u9fff]/.test(text);
    const isCodeSwitch = hasEnglish && hasChinese;
    
    const programmingLanguages: string[] = [];
    const langPatterns = {
      'python': /def |import |from .* import|\.py\b/,
      'javascript': /function|const |let |var |\.js\b/,
      'typescript': /interface|type |\.ts\b/,
      'rust': /fn |let mut|use |\.rs\b/,
      'go': /func |package |import |\.go\b/
    };
    
    Object.entries(langPatterns).forEach(([lang, pattern]) => {
      if (pattern.test(text)) programmingLanguages.push(lang);
    });
    
    return { isEnglish: hasEnglish, isChinese: hasChinese, isCodeSwitch, programmingLanguages };
  }

  private extractKVPrefixes(text: string): Set<string> {
    const kvPatterns = [
      /"([^"]+)"\s*:\s*/g,  // JSON key-value
      /(\w+)\s*=\s*/g,      // Assignment
      /(\w+)\s*:\s*/g       // General key-value
    ];
    
    const prefixes = new Set<string>();
    kvPatterns.forEach(pattern => {
      let match;
      while ((match = pattern.exec(text)) !== null) {
        prefixes.add(match[1]);
      }
    });
    
    return prefixes;
  }

  private calculateJaccardSimilarity(set1: Set<string>, set2: Set<string>): number {
    const intersection = new Set([...set1].filter(x => set2.has(x)));
    const union = new Set([...set1, ...set2]);
    
    return union.size === 0 ? 1.0 : intersection.size / union.size;
  }

  private calculateMeanDifference(sample1: number[], sample2: number[]): number {
    const mean1 = sample1.reduce((a, b) => a + b, 0) / sample1.length;
    const mean2 = sample2.reduce((a, b) => a + b, 0) / sample2.length;
    return mean1 - mean2;
  }

  private generateBootstrapIndices(sampleSize: number): number[] {
    return Array.from({ length: sampleSize }, () => Math.floor(Math.random() * sampleSize));
  }

  private parseValidatorJsonl(jsonlLines: string[]): ValidationResult[] {
    return jsonlLines
      .filter(line => line.trim())
      .map(line => {
        try {
          return JSON.parse(line);
        } catch (error) {
          throw new Error(`Failed to parse JSONL line: ${line}`);
        }
      });
  }
}

// ============================================================================
// SUPPORTING TYPES AND INTERFACES
// ============================================================================

interface ValidationResult {
  dataset: string;
  keep_ratio: number;
  k: number;
  seed: number;
  macro_p_at_5: number;
  cost_per_query: number;
  latency_p95: number;
  latency_p99_p95_ratio: number;
  candidates: Candidate[];
  config: Config;
  sample_results: number[]; // Individual query results for statistical analysis
}

interface CompetitorResults {
  dataset: string;
  keep_ratio: number;
  k: number;
  competitor_name: string;
  macro_p_at_5: number;
  cost_per_query: number;
  latency_p95: number;
  latency_p99_p95_ratio: number;
  sample_results: number[];
}

interface PerformanceBaseline {
  slice_key: string;
  best_competitor_name: string;
  baseline_p_at_5: number;
  baseline_cost_per_query: number;
  baseline_latency_p95: number;
  sample_size: number;
  established_at: number;
}

/**
 * HTML Integration for GapBoard Visualization
 */
export class GapBoardHTMLRenderer {
  /**
   * Generates HTML visualization for gap analysis results
   */
  static generateGapBoardHTML(gapRecords: GapRecord[]): string {
    const sortedGaps = gapRecords.sort((a, b) => b.priority_score - a.priority_score);
    
    return `
    <!DOCTYPE html>
    <html>
    <head>
        <title>GapBoard v1 - Gap Analysis Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .gap-record { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
            .gap-record.high-priority { border-color: #ff4444; background-color: #fff5f5; }
            .gap-record.medium-priority { border-color: #ffaa00; background-color: #fff8f0; }
            .gap-record.low-priority { border-color: #44aa44; background-color: #f5fff5; }
            .delta-map { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin: 10px 0; }
            .metric { background: #f9f9f9; padding: 8px; border-radius: 3px; }
            .metric.negative { background: #ffe6e6; }
            .metric.positive { background: #e6ffe6; }
            .root-causes { margin: 10px 0; }
            .feature-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
            .policy-fingerprint { font-family: monospace; background: #f0f0f0; padding: 10px; margin: 5px 0; }
            .stats { font-size: 0.9em; color: #666; }
        </style>
    </head>
    <body>
        <h1>GapBoard v1 - Performance Gap Analysis</h1>
        <p>Identified ${gapRecords.length} statistically significant performance gaps</p>
        
        ${sortedGaps.map(gap => this.renderGapRecord(gap)).join('')}
    </body>
    </html>
    `;
  }

  private static renderGapRecord(gap: GapRecord): string {
    const priorityClass = gap.priority_score > 0.5 ? 'high-priority' : 
                         gap.priority_score > 0.2 ? 'medium-priority' : 'low-priority';

    return `
    <div class="gap-record ${priorityClass}">
        <h3>Gap: ${gap.slice_id} (${gap.dataset})</h3>
        <div class="metadata">
            <strong>Slice:</strong> keep_ratio=${gap.keep_ratio}, k=${gap.k}, seed=${gap.seed} |
            <strong>Priority:</strong> ${gap.priority_score.toFixed(3)} |
            <strong>Est. Uplift:</strong> ${gap.estimated_uplift.toFixed(3)}
        </div>
        
        <h4>Delta Map (vs Best Competitor)</h4>
        <div class="delta-map">
            <div class="metric ${gap.delta_map.macro_p_at_5 >= 0 ? 'positive' : 'negative'}">
                <strong>P@5 Delta:</strong> ${gap.delta_map.macro_p_at_5.toFixed(4)}
            </div>
            <div class="metric ${gap.delta_map.cost_per_query <= 0 ? 'positive' : 'negative'}">
                <strong>Cost Delta:</strong> ${gap.delta_map.cost_per_query.toFixed(4)}
            </div>
            <div class="metric ${gap.delta_map.latency_p95 <= 0 ? 'positive' : 'negative'}">
                <strong>Latency p95 Delta:</strong> ${gap.delta_map.latency_p95.toFixed(2)}ms
            </div>
            <div class="metric ${gap.delta_map.latency_p99_p95_ratio <= 2.5 ? 'positive' : 'negative'}">
                <strong>p99/p95 Ratio:</strong> ${gap.delta_map.latency_p99_p95_ratio.toFixed(2)}
            </div>
        </div>
        
        <h4>Root-Cause Features</h4>
        <div class="root-causes">
            <div class="feature-grid">
                <div><strong>Entity Entropy:</strong> ${gap.root_cause_features.entity_entropy.toFixed(3)}</div>
                <div><strong>Dup Rate:</strong> ${(gap.root_cause_features.dup_rate * 100).toFixed(1)}%</div>
                <div><strong>Closure Depth:</strong> ${gap.root_cause_features.closure_depth.toFixed(2)}</div>
                <div><strong>Symbol Complexity:</strong> ${gap.root_cause_features.symbol_length_avg.toFixed(1)}</div>
                <div><strong>KV Stability:</strong> ${(gap.root_cause_features.kv_stability * 100).toFixed(1)}%</div>
                <div><strong>Code Heavy:</strong> ${(gap.root_cause_features.type_mix.code_heavy * 100).toFixed(1)}%</div>
            </div>
        </div>
        
        <h4>Policy Fingerprint</h4>
        <div class="policy-fingerprint">
            λ=${gap.policy_fingerprint.lambda} μ=${gap.policy_fingerprint.mu} K2=${gap.policy_fingerprint.K2} r=${gap.policy_fingerprint.r}
            head_keep=${gap.policy_fingerprint.head_keep} window=${gap.policy_fingerprint.window_size} τ=${gap.policy_fingerprint.tau}
        </div>
        
        <div class="stats">
            <strong>Statistical Separation:</strong> 
            ${gap.statistical_separation.is_significant ? '✓ Significant' : '✗ Not Significant'} 
            (p=${gap.statistical_separation.p_value.toFixed(4)}, d=${gap.statistical_separation.effect_size.toFixed(3)}) |
            <strong>Status:</strong> ${gap.status} |
            <strong>Created:</strong> ${new Date(gap.created_at).toLocaleString()}
        </div>
    </div>
    `;
  }
}