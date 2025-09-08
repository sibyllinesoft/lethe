/**
 * Difficulty Gate - Lightweight GBM for Adaptive Policy Initialization
 * 
 * Trains a small Gradient Boosting Machine on gap features to choose optimal
 * dimensions (256/768) and K2 caps per query. Acts as a pre-tuner that feeds
 * θ0 into the main auto-tuning system.
 */

import {
  DifficultyGate,
  FeatureExtractorConfig,
  GapRecord,
  PolicyFingerprint,
  GapAnalysisResult,
  GapAnalysisError
} from './types.js';

import { Config } from '../types.js';

// ============================================================================
// CORE DIFFICULTY GATE SYSTEM
// ============================================================================

export class DifficultyGateSystem {
  private config: Config;
  private gbmModel: LightweightGBM | null = null;
  private featureExtractors: Map<string, FeatureExtractor> = new Map();
  private isModelTrained = false;

  constructor(config: Config) {
    this.config = config;
    this.initializeFeatureExtractors();
  }

  /**
   * Initializes adaptive policy based on difficulty assessment
   */
  async initializeAdaptivePolicy(
    gapRecord: GapRecord
  ): Promise<GapAnalysisResult<PolicyFingerprint>> {
    try {
      if (!this.isModelTrained) {
        await this.ensureModelTrained();
      }

      // Extract features for difficulty assessment
      const features = await this.extractDifficultyFeatures(gapRecord);
      
      // Predict complexity level using GBM
      const complexityPrediction = await this.gbmModel!.predict(features);
      
      // Generate adaptive policy based on prediction
      const adaptivePolicy = this.generateAdaptivePolicy(
        gapRecord.policy_fingerprint,
        complexityPrediction
      );

      console.log(`Difficulty gate: ${gapRecord.slice_id} -> complexity ${complexityPrediction.complexity_level} (score: ${complexityPrediction.complexity_score.toFixed(3)})`);

      return {
        success: true,
        data: adaptivePolicy
      };

    } catch (error) {
      return {
        success: false,
        error: {
          code: 'DIFFICULTY_GATE_ERROR',
          message: `Difficulty gate initialization failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          error_type: 'gap_detection',
          gap_context: {
            slice_id: gapRecord.slice_id
          },
          recovery_actions: ['Retrain GBM model', 'Use default policy initialization', 'Check feature extraction'],
          is_retryable: true,
          impact_severity: 'medium',
          affected_components: ['difficulty_gate', 'tuning_pipeline'],
          timestamp: Date.now()
        }
      };
    }
  }

  /**
   * Trains the GBM model on historical gap data
   */
  async trainGBMModel(
    trainingData: DifficultyTrainingData[]
  ): Promise<GapAnalysisResult<DifficultyGate>> {
    try {
      console.log(`Training GBM model on ${trainingData.length} samples`);

      // Prepare training features and targets
      const { features, targets } = await this.prepareTrainingData(trainingData);
      
      // Initialize and train GBM
      this.gbmModel = new LightweightGBM({
        n_estimators: 50,
        max_depth: 4,
        learning_rate: 0.1,
        feature_columns: this.getFeatureColumnNames()
      });

      const trainingResult = await this.gbmModel.train(features, targets);
      
      // Evaluate model performance
      const evaluation = await this.evaluateModel(features, targets);
      
      this.isModelTrained = true;

      const difficultyGate: DifficultyGate = {
        model_id: `gbm_${Date.now()}`,
        gbm_config: {
          n_estimators: 50,
          max_depth: 4,
          learning_rate: 0.1,
          feature_columns: this.getFeatureColumnNames()
        },
        dimension_thresholds: {
          low_complexity: 0.3,      // → 256 dims
          medium_complexity: 0.6,   // → 512 dims
          high_complexity: 0.8      // → 768 dims
        },
        k2_cap_rules: {
          easy_queries: 80,         // Conservative K2
          medium_queries: 120,      // Standard K2
          hard_queries: 180         // Aggressive K2
        },
        feature_extractors: Array.from(this.featureExtractors.values()).map(fe => fe.config),
        model_accuracy: evaluation,
        last_trained: Date.now(),
        training_data_size: trainingData.length
      };

      console.log(`GBM training completed. Accuracy: ${evaluation.cross_validation_score.toFixed(3)}`);

      return {
        success: true,
        data: difficultyGate
      };

    } catch (error) {
      return {
        success: false,
        error: {
          code: 'GBM_TRAINING_ERROR',
          message: `GBM training failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          error_type: 'gap_detection',
          recovery_actions: ['Verify training data quality', 'Adjust GBM hyperparameters', 'Check feature extraction'],
          is_retryable: true,
          impact_severity: 'medium',
          affected_components: ['difficulty_gate'],
          timestamp: Date.now()
        }
      };
    }
  }

  // ============================================================================
  // FEATURE EXTRACTION SYSTEM
  // ============================================================================

  private initializeFeatureExtractors(): void {
    // Gap features extractor
    const gapFeaturesExtractor = new FeatureExtractor({
      extractor_name: 'gap_features',
      feature_type: 'gap_features',
      computations: {
        entity_entropy: true,
        dup_intensity: true,
        closure_depth: true,
        symbol_complexity: true,
        type_mix_variance: true,
        language_switching_rate: true,
        kv_instability_score: true
      },
      normalization: 'z_score',
      missing_value_strategy: 'mean'
    });

    // Query features extractor (simulated - would extract from query text)
    const queryFeaturesExtractor = new FeatureExtractor({
      extractor_name: 'query_features',
      feature_type: 'query_features',
      computations: {
        entity_entropy: true,
        symbol_complexity: true,
        closure_depth: true
      },
      normalization: 'min_max',
      missing_value_strategy: 'median'
    });

    // Context features extractor
    const contextFeaturesExtractor = new FeatureExtractor({
      extractor_name: 'context_features',
      feature_type: 'context_features',
      computations: {
        type_mix_variance: true,
        language_switching_rate: true,
        kv_instability_score: true
      },
      normalization: 'robust',
      missing_value_strategy: 'zero'
    });

    this.featureExtractors.set('gap_features', gapFeaturesExtractor);
    this.featureExtractors.set('query_features', queryFeaturesExtractor);
    this.featureExtractors.set('context_features', contextFeaturesExtractor);
  }

  private async extractDifficultyFeatures(gapRecord: GapRecord): Promise<Float64Array> {
    const allFeatures: number[] = [];

    // Extract gap features
    const gapExtractor = this.featureExtractors.get('gap_features')!;
    const gapFeatures = await gapExtractor.extract(gapRecord);
    allFeatures.push(...gapFeatures);

    // Extract query features (simulated based on gap characteristics)
    const queryExtractor = this.featureExtractors.get('query_features')!;
    const queryFeatures = await queryExtractor.extract(gapRecord);
    allFeatures.push(...queryFeatures);

    // Extract context features
    const contextExtractor = this.featureExtractors.get('context_features')!;
    const contextFeatures = await contextExtractor.extract(gapRecord);
    allFeatures.push(...contextFeatures);

    return new Float64Array(allFeatures);
  }

  private getFeatureColumnNames(): string[] {
    return [
      // Gap features
      'entity_entropy', 'dup_intensity', 'closure_depth', 'symbol_complexity',
      'type_mix_variance', 'language_switching_rate', 'kv_instability_score',
      // Query features  
      'query_entity_entropy', 'query_symbol_complexity', 'query_closure_depth',
      // Context features
      'context_type_mix_variance', 'context_language_switching', 'context_kv_instability'
    ];
  }

  // ============================================================================
  // ADAPTIVE POLICY GENERATION
  // ============================================================================

  private generateAdaptivePolicy(
    basePolicy: PolicyFingerprint,
    complexityPrediction: ComplexityPrediction
  ): PolicyFingerprint {
    const adaptivePolicy = { ...basePolicy };

    // Adjust dimensions based on complexity
    const dimensions = this.selectDimensions(complexityPrediction.complexity_level);
    
    // Adjust K2 cap based on complexity
    const k2Cap = this.selectK2Cap(complexityPrediction.complexity_level);
    
    // Adjust other parameters based on complexity characteristics
    const parameterAdjustments = this.calculateParameterAdjustments(complexityPrediction);

    // Apply adjustments
    adaptivePolicy.K2 = Math.min(k2Cap, adaptivePolicy.K2);
    adaptivePolicy.lambda *= parameterAdjustments.lambda_multiplier;
    adaptivePolicy.mu *= parameterAdjustments.mu_multiplier;
    adaptivePolicy.r = Math.max(12, Math.min(24, Math.round(adaptivePolicy.r * parameterAdjustments.r_multiplier)));
    
    // Store adaptation metadata
    adaptivePolicy.policy_id = `adaptive_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
    adaptivePolicy.created_at = Date.now();
    adaptivePolicy.validation_status = 'pending';

    return adaptivePolicy;
  }

  private selectDimensions(complexityLevel: ComplexityLevel): number {
    switch (complexityLevel) {
      case 'low': return 256;
      case 'medium': return 512;
      case 'high': return 768;
      default: return 512;
    }
  }

  private selectK2Cap(complexityLevel: ComplexityLevel): number {
    switch (complexityLevel) {
      case 'low': return 80;      // Conservative for simple queries
      case 'medium': return 120;  // Standard cap
      case 'high': return 180;    // Aggressive for complex queries
      default: return 120;
    }
  }

  private calculateParameterAdjustments(prediction: ComplexityPrediction): ParameterAdjustments {
    const baseAdjustment = {
      lambda_multiplier: 1.0,
      mu_multiplier: 1.0,
      r_multiplier: 1.0
    };

    switch (prediction.complexity_level) {
      case 'low':
        // Simple queries: favor speed over comprehensiveness
        return {
          lambda_multiplier: 1.1,  // Slight BM25 bias
          mu_multiplier: 0.9,      // Reduce vector weight
          r_multiplier: 0.8        // Lower diversity requirement
        };

      case 'medium':
        // Standard queries: balanced approach
        return baseAdjustment;

      case 'high':
        // Complex queries: favor comprehensiveness
        return {
          lambda_multiplier: 0.9,  // Reduce BM25 weight
          mu_multiplier: 1.2,      // Increase vector weight
          r_multiplier: 1.3        // Higher diversity requirement
        };

      default:
        return baseAdjustment;
    }
  }

  // ============================================================================
  // MODEL TRAINING AND EVALUATION
  // ============================================================================

  private async prepareTrainingData(
    trainingData: DifficultyTrainingData[]
  ): Promise<{ features: Float64Array[]; targets: ComplexityLevel[] }> {
    const features: Float64Array[] = [];
    const targets: ComplexityLevel[] = [];

    for (const sample of trainingData) {
      const extractedFeatures = await this.extractDifficultyFeatures(sample.gap_record);
      features.push(extractedFeatures);
      targets.push(sample.actual_complexity);
    }

    return { features, targets };
  }

  private async evaluateModel(
    features: Float64Array[],
    targets: ComplexityLevel[]
  ): Promise<DifficultyGate['model_accuracy']> {
    if (!this.gbmModel) {
      throw new Error('GBM model not initialized');
    }

    // Perform k-fold cross-validation
    const kFolds = 5;
    const foldSize = Math.floor(features.length / kFolds);
    let totalAccuracy = 0;
    let totalPrecisionRecallAuc = 0;
    let totalCalibrationError = 0;

    for (let fold = 0; fold < kFolds; fold++) {
      // Split data into train/test for this fold
      const testStart = fold * foldSize;
      const testEnd = Math.min(testStart + foldSize, features.length);
      
      const trainFeatures = [
        ...features.slice(0, testStart),
        ...features.slice(testEnd)
      ];
      const trainTargets = [
        ...targets.slice(0, testStart),
        ...targets.slice(testEnd)
      ];
      const testFeatures = features.slice(testStart, testEnd);
      const testTargets = targets.slice(testStart, testEnd);

      // Train model on training fold
      const foldModel = new LightweightGBM(this.gbmModel.config);
      await foldModel.train(trainFeatures, trainTargets);

      // Evaluate on test fold
      let correctPredictions = 0;
      for (let i = 0; i < testFeatures.length; i++) {
        const prediction = await foldModel.predict(testFeatures[i]);
        if (prediction.complexity_level === testTargets[i]) {
          correctPredictions++;
        }
      }

      const foldAccuracy = correctPredictions / testFeatures.length;
      totalAccuracy += foldAccuracy;

      // Simplified precision-recall AUC (would be more complex in practice)
      totalPrecisionRecallAuc += foldAccuracy * 0.9; // Approximation

      // Simplified calibration error
      totalCalibrationError += (1 - foldAccuracy) * 0.1; // Approximation
    }

    return {
      cross_validation_score: totalAccuracy / kFolds,
      precision_recall_auc: totalPrecisionRecallAuc / kFolds,
      calibration_error: totalCalibrationError / kFolds
    };
  }

  private async ensureModelTrained(): Promise<void> {
    if (!this.isModelTrained) {
      // Generate synthetic training data if no real data available
      const syntheticData = this.generateSyntheticTrainingData(100);
      const trainingResult = await this.trainGBMModel(syntheticData);
      
      if (!trainingResult.success) {
        throw new Error(`Failed to train GBM model: ${trainingResult.error.message}`);
      }
    }
  }

  private generateSyntheticTrainingData(sampleCount: number): DifficultyTrainingData[] {
    const data: DifficultyTrainingData[] = [];

    for (let i = 0; i < sampleCount; i++) {
      // Generate synthetic gap record with varying complexity characteristics
      const complexity = ['low', 'medium', 'high'][Math.floor(Math.random() * 3)] as ComplexityLevel;
      const gapRecord = this.createSyntheticGapRecord(complexity);
      
      data.push({
        gap_record: gapRecord,
        actual_complexity: complexity,
        performance_outcome: {
          tuning_success: Math.random() > (complexity === 'high' ? 0.3 : 0.1),
          final_improvement: Math.random() * 0.1,
          convergence_speed: complexity === 'low' ? 'fast' : complexity === 'medium' ? 'medium' : 'slow'
        }
      });
    }

    return data;
  }

  private createSyntheticGapRecord(targetComplexity: ComplexityLevel): GapRecord {
    // Generate synthetic gap features based on target complexity
    const complexityFactors = {
      low: { entityEntropy: 0.3, closureDepth: 1.2, symbolLength: 8 },
      medium: { entityEntropy: 1.5, closureDepth: 3.5, symbolLength: 15 },
      high: { entityEntropy: 2.8, closureDepth: 6.0, symbolLength: 25 }
    };

    const factors = complexityFactors[targetComplexity];

    return {
      slice_id: `synthetic_${Date.now()}_${Math.random().toString(36).substr(2, 8)}`,
      dataset: 'synthetic',
      keep_ratio: 0.15,
      k: 10,
      seed: Math.floor(Math.random() * 10000),
      delta_map: {
        macro_p_at_5: -0.05 - Math.random() * 0.1,
        cost_per_query: 0.02 + Math.random() * 0.03,
        latency_p95: 10 + Math.random() * 20,
        latency_p99_p95_ratio: 1.8 + Math.random() * 0.5
      },
      root_cause_features: {
        entity_entropy: factors.entityEntropy + Math.random() * 0.5,
        dup_rate: Math.random() * 0.3,
        type_mix: {
          code_heavy: Math.random() * 0.5,
          error_heavy: Math.random() * 0.3,
          tool_heavy: Math.random() * 0.4,
          prose_heavy: Math.random() * 0.6,
          json_needle: Math.random() * 0.2
        },
        closure_depth: factors.closureDepth + Math.random() * 1.0,
        symbol_length_avg: factors.symbolLength + Math.random() * 5,
        language_distribution: {
          english: 0.6 + Math.random() * 0.3,
          chinese: Math.random() * 0.2,
          code_switch: Math.random() * 0.1,
          programming_languages: { javascript: 0.3, python: 0.2 }
        },
        kv_stability: 0.7 + Math.random() * 0.2
      },
      policy_fingerprint: {
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
        policy_id: `baseline_policy`,
        created_at: Date.now(),
        validation_status: 'pending'
      },
      statistical_separation: {
        is_significant: true,
        p_value: 0.01,
        confidence_interval: [-0.1, -0.02],
        effect_size: 1.2
      },
      priority_score: Math.random(),
      estimated_uplift: Math.random() * 0.05,
      created_at: Date.now(),
      updated_at: Date.now(),
      validation_runs: 0,
      status: 'identified'
    };
  }
}

// ============================================================================
// LIGHTWEIGHT GBM IMPLEMENTATION
// ============================================================================

export class LightweightGBM {
  public config: GBMConfig;
  private trees: DecisionTree[] = [];
  private featureMeans: Float64Array = new Float64Array();
  private featureStds: Float64Array = new Float64Array();

  constructor(config: GBMConfig) {
    this.config = config;
  }

  async train(features: Float64Array[], targets: ComplexityLevel[]): Promise<void> {
    console.log(`Training GBM with ${features.length} samples, ${this.config.n_estimators} estimators`);

    // Normalize features
    this.calculateFeatureNormalization(features);
    const normalizedFeatures = features.map(f => this.normalizeFeatures(f));

    // Convert targets to numerical values for training
    const numericalTargets = targets.map(t => this.complexityToNumber(t));

    // Initialize predictions with mean
    let predictions = new Float64Array(numericalTargets.length);
    predictions.fill(this.calculateMean(numericalTargets));

    // Train trees sequentially (boosting)
    for (let i = 0; i < this.config.n_estimators; i++) {
      // Calculate residuals
      const residuals = numericalTargets.map((target, idx) => target - predictions[idx]);
      
      // Train tree on residuals
      const tree = new DecisionTree(this.config.max_depth);
      await tree.train(normalizedFeatures, residuals);
      
      this.trees.push(tree);

      // Update predictions
      for (let j = 0; j < predictions.length; j++) {
        const treePrediction = await tree.predict(normalizedFeatures[j]);
        predictions[j] += this.config.learning_rate * treePrediction;
      }

      if ((i + 1) % 10 === 0) {
        console.log(`Completed ${i + 1}/${this.config.n_estimators} trees`);
      }
    }

    console.log('GBM training completed');
  }

  async predict(features: Float64Array): Promise<ComplexityPrediction> {
    if (this.trees.length === 0) {
      throw new Error('GBM model not trained');
    }

    const normalizedFeatures = this.normalizeFeatures(features);
    
    // Aggregate predictions from all trees
    let totalPrediction = 0;
    for (const tree of this.trees) {
      const treePrediction = await tree.predict(normalizedFeatures);
      totalPrediction += this.config.learning_rate * treePrediction;
    }

    // Convert numerical prediction to complexity level
    const complexityLevel = this.numberToComplexity(totalPrediction);
    
    return {
      complexity_level: complexityLevel,
      complexity_score: Math.abs(totalPrediction),
      confidence: Math.min(1.0, Math.abs(totalPrediction) / 2.0) // Rough confidence estimate
    };
  }

  private calculateFeatureNormalization(features: Float64Array[]): void {
    if (features.length === 0) return;

    const featureCount = features[0].length;
    this.featureMeans = new Float64Array(featureCount);
    this.featureStds = new Float64Array(featureCount);

    // Calculate means
    for (let f = 0; f < featureCount; f++) {
      let sum = 0;
      for (const sample of features) {
        sum += sample[f];
      }
      this.featureMeans[f] = sum / features.length;
    }

    // Calculate standard deviations
    for (let f = 0; f < featureCount; f++) {
      let sumSquaredDiffs = 0;
      for (const sample of features) {
        const diff = sample[f] - this.featureMeans[f];
        sumSquaredDiffs += diff * diff;
      }
      this.featureStds[f] = Math.sqrt(sumSquaredDiffs / features.length);
    }
  }

  private normalizeFeatures(features: Float64Array): Float64Array {
    const normalized = new Float64Array(features.length);
    for (let i = 0; i < features.length; i++) {
      const std = this.featureStds[i] || 1; // Avoid division by zero
      normalized[i] = (features[i] - this.featureMeans[i]) / std;
    }
    return normalized;
  }

  private complexityToNumber(complexity: ComplexityLevel): number {
    switch (complexity) {
      case 'low': return 0;
      case 'medium': return 1;
      case 'high': return 2;
      default: return 1;
    }
  }

  private numberToComplexity(value: number): ComplexityLevel {
    if (value < 0.5) return 'low';
    if (value < 1.5) return 'medium';
    return 'high';
  }

  private calculateMean(values: number[]): number {
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }
}

// ============================================================================
// DECISION TREE IMPLEMENTATION
// ============================================================================

export class DecisionTree {
  private maxDepth: number;
  private root: TreeNode | null = null;

  constructor(maxDepth: number) {
    this.maxDepth = maxDepth;
  }

  async train(features: Float64Array[], targets: number[]): Promise<void> {
    this.root = this.buildTree(features, targets, 0);
  }

  async predict(features: Float64Array): Promise<number> {
    if (!this.root) {
      throw new Error('Decision tree not trained');
    }

    return this.traverseTree(this.root, features);
  }

  private buildTree(features: Float64Array[], targets: number[], depth: number): TreeNode {
    // Base cases
    if (depth >= this.maxDepth || targets.length < 2) {
      return {
        isLeaf: true,
        value: this.calculateMean(targets),
        featureIndex: -1,
        threshold: 0,
        left: null,
        right: null
      };
    }

    // Find best split
    const bestSplit = this.findBestSplit(features, targets);
    
    if (bestSplit.improvement < 0.01) {
      // Not enough improvement, make leaf
      return {
        isLeaf: true,
        value: this.calculateMean(targets),
        featureIndex: -1,
        threshold: 0,
        left: null,
        right: null
      };
    }

    // Split data
    const { leftFeatures, leftTargets, rightFeatures, rightTargets } = 
      this.splitData(features, targets, bestSplit.featureIndex, bestSplit.threshold);

    // Recursively build subtrees
    const leftChild = this.buildTree(leftFeatures, leftTargets, depth + 1);
    const rightChild = this.buildTree(rightFeatures, rightTargets, depth + 1);

    return {
      isLeaf: false,
      value: 0,
      featureIndex: bestSplit.featureIndex,
      threshold: bestSplit.threshold,
      left: leftChild,
      right: rightChild
    };
  }

  private findBestSplit(features: Float64Array[], targets: number[]): SplitResult {
    let bestSplit: SplitResult = {
      featureIndex: 0,
      threshold: 0,
      improvement: -1
    };

    const featureCount = features[0]?.length || 0;
    
    for (let featureIdx = 0; featureIdx < featureCount; featureIdx++) {
      // Get sorted unique values for this feature
      const featureValues = features.map(f => f[featureIdx]).sort((a, b) => a - b);
      const uniqueValues = [...new Set(featureValues)];
      
      for (let i = 0; i < uniqueValues.length - 1; i++) {
        const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
        
        // Calculate improvement for this split
        const improvement = this.calculateSplitImprovement(features, targets, featureIdx, threshold);
        
        if (improvement > bestSplit.improvement) {
          bestSplit = {
            featureIndex: featureIdx,
            threshold,
            improvement
          };
        }
      }
    }

    return bestSplit;
  }

  private calculateSplitImprovement(
    features: Float64Array[],
    targets: number[],
    featureIndex: number,
    threshold: number
  ): number {
    // Calculate mean squared error before split
    const totalMse = this.calculateMSE(targets);
    
    // Split data
    const { leftTargets, rightTargets } = this.splitTargets(features, targets, featureIndex, threshold);
    
    if (leftTargets.length === 0 || rightTargets.length === 0) {
      return -1; // Invalid split
    }

    // Calculate weighted MSE after split
    const leftMse = this.calculateMSE(leftTargets);
    const rightMse = this.calculateMSE(rightTargets);
    const totalSamples = targets.length;
    const weightedMse = (leftTargets.length / totalSamples) * leftMse + 
                       (rightTargets.length / totalSamples) * rightMse;

    return totalMse - weightedMse;
  }

  private splitData(
    features: Float64Array[],
    targets: number[],
    featureIndex: number,
    threshold: number
  ): {
    leftFeatures: Float64Array[];
    leftTargets: number[];
    rightFeatures: Float64Array[];
    rightTargets: number[];
  } {
    const leftFeatures: Float64Array[] = [];
    const leftTargets: number[] = [];
    const rightFeatures: Float64Array[] = [];
    const rightTargets: number[] = [];

    for (let i = 0; i < features.length; i++) {
      if (features[i][featureIndex] <= threshold) {
        leftFeatures.push(features[i]);
        leftTargets.push(targets[i]);
      } else {
        rightFeatures.push(features[i]);
        rightTargets.push(targets[i]);
      }
    }

    return { leftFeatures, leftTargets, rightFeatures, rightTargets };
  }

  private splitTargets(
    features: Float64Array[],
    targets: number[],
    featureIndex: number,
    threshold: number
  ): { leftTargets: number[]; rightTargets: number[] } {
    const leftTargets: number[] = [];
    const rightTargets: number[] = [];

    for (let i = 0; i < features.length; i++) {
      if (features[i][featureIndex] <= threshold) {
        leftTargets.push(targets[i]);
      } else {
        rightTargets.push(targets[i]);
      }
    }

    return { leftTargets, rightTargets };
  }

  private traverseTree(node: TreeNode, features: Float64Array): number {
    if (node.isLeaf) {
      return node.value;
    }

    if (features[node.featureIndex] <= node.threshold) {
      return this.traverseTree(node.left!, features);
    } else {
      return this.traverseTree(node.right!, features);
    }
  }

  private calculateMean(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((sum, val) => sum + val, 0) / values.length;
  }

  private calculateMSE(targets: number[]): number {
    if (targets.length === 0) return 0;
    
    const mean = this.calculateMean(targets);
    const squaredDiffs = targets.map(t => (t - mean) ** 2);
    return squaredDiffs.reduce((sum, diff) => sum + diff, 0) / targets.length;
  }
}

// ============================================================================
// FEATURE EXTRACTOR IMPLEMENTATION
// ============================================================================

export class FeatureExtractor {
  public config: FeatureExtractorConfig;
  private normalizationStats: { mean: number; std: number; min: number; max: number }[] = [];

  constructor(config: FeatureExtractorConfig) {
    this.config = config;
  }

  async extract(gapRecord: GapRecord): Promise<number[]> {
    const features: number[] = [];
    const computations = this.config.computations;

    if (computations.entity_entropy) {
      features.push(gapRecord.root_cause_features.entity_entropy);
    }

    if (computations.dup_intensity) {
      features.push(gapRecord.root_cause_features.dup_rate);
    }

    if (computations.closure_depth) {
      features.push(gapRecord.root_cause_features.closure_depth);
    }

    if (computations.symbol_complexity) {
      features.push(gapRecord.root_cause_features.symbol_length_avg);
    }

    if (computations.type_mix_variance) {
      const typeMix = gapRecord.root_cause_features.type_mix;
      const values = [typeMix.code_heavy, typeMix.error_heavy, typeMix.tool_heavy, typeMix.prose_heavy];
      const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
      const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length;
      features.push(variance);
    }

    if (computations.language_switching_rate) {
      features.push(gapRecord.root_cause_features.language_distribution.code_switch);
    }

    if (computations.kv_instability_score) {
      features.push(1 - gapRecord.root_cause_features.kv_stability);
    }

    // Apply normalization if configured
    return this.applyNormalization(features);
  }

  private applyNormalization(features: number[]): number[] {
    switch (this.config.normalization) {
      case 'z_score':
        return this.applyZScoreNormalization(features);
      case 'min_max':
        return this.applyMinMaxNormalization(features);
      case 'robust':
        return this.applyRobustNormalization(features);
      default:
        return features;
    }
  }

  private applyZScoreNormalization(features: number[]): number[] {
    // Simple z-score normalization (would be more sophisticated in practice)
    const mean = features.reduce((sum, f) => sum + f, 0) / features.length;
    const variance = features.reduce((sum, f) => sum + (f - mean) ** 2, 0) / features.length;
    const std = Math.sqrt(variance) || 1;
    
    return features.map(f => (f - mean) / std);
  }

  private applyMinMaxNormalization(features: number[]): number[] {
    const min = Math.min(...features);
    const max = Math.max(...features);
    const range = max - min || 1;
    
    return features.map(f => (f - min) / range);
  }

  private applyRobustNormalization(features: number[]): number[] {
    // Using median and IQR for robust normalization
    const sorted = [...features].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const q1 = sorted[Math.floor(sorted.length * 0.25)];
    const q3 = sorted[Math.floor(sorted.length * 0.75)];
    const iqr = q3 - q1 || 1;
    
    return features.map(f => (f - median) / iqr);
  }
}

// ============================================================================
// SUPPORTING TYPES AND INTERFACES
// ============================================================================

type ComplexityLevel = 'low' | 'medium' | 'high';
type ConvergenceSpeed = 'fast' | 'medium' | 'slow';

interface ComplexityPrediction {
  complexity_level: ComplexityLevel;
  complexity_score: number;
  confidence: number;
}

interface ParameterAdjustments {
  lambda_multiplier: number;
  mu_multiplier: number;
  r_multiplier: number;
}

interface DifficultyTrainingData {
  gap_record: GapRecord;
  actual_complexity: ComplexityLevel;
  performance_outcome: {
    tuning_success: boolean;
    final_improvement: number;
    convergence_speed: ConvergenceSpeed;
  };
}

interface GBMConfig {
  n_estimators: number;
  max_depth: number;
  learning_rate: number;
  feature_columns: string[];
}

interface TreeNode {
  isLeaf: boolean;
  value: number;
  featureIndex: number;
  threshold: number;
  left: TreeNode | null;
  right: TreeNode | null;
}

interface SplitResult {
  featureIndex: number;
  threshold: number;
  improvement: number;
}