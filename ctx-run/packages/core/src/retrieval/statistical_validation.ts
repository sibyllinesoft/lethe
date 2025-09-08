/**
 * Statistical Validation Framework
 * 
 * Comprehensive statistical analysis and validation system for Lagrangian deployment.
 * Implements rigorous statistical testing for performance improvements and quality preservation.
 * 
 * Key Features:
 * - A/B testing framework with proper statistical power
 * - Bayesian update mechanisms for continuous learning
 * - Effect size estimation with confidence intervals
 * - Multiple testing correction (Bonferroni, FDR)
 * - Sequential testing for early stopping
 * - Promotion gate criteria with statistical significance
 * 
 * Target: Statistical validation of 85% latency reduction while maintaining +12.5% CBU
 */

import { z } from 'zod';

// Configuration for statistical validation
export const StatisticalValidationConfigSchema = z.object({
  // Test design parameters
  significance_level: z.number().min(0).max(1).default(0.05), // α = 0.05
  statistical_power: z.number().min(0).max(1).default(0.8), // 1-β = 0.8
  minimum_effect_size: z.number().min(0).default(0.1), // 10% minimum detectable effect
  
  // Sample size parameters
  minimum_sample_size_per_group: z.number().min(10).default(100),
  maximum_sample_size_per_group: z.number().min(100).default(10000),
  sequential_testing_interval: z.number().min(10).default(50), // Check every N samples
  
  // Multiple testing correction
  correction_method: z.enum(['bonferroni', 'fdr', 'holm']).default('fdr'),
  num_simultaneous_tests: z.number().min(1).default(4), // latency, quality, stability, efficiency
  
  // Bayesian parameters
  enable_bayesian_updates: z.boolean().default(true),
  prior_confidence: z.number().min(0).max(1).default(0.1), // Weak prior
  posterior_update_interval: z.number().min(5).default(20),
  credible_interval_level: z.number().min(0).max(1).default(0.95), // 95% CI
  
  // Promotion criteria
  latency_improvement_threshold: z.number().min(0).default(0.8), // 80% of target (85% → 68%)
  quality_preservation_threshold: z.number().min(0).default(0.95), // 95% quality preservation
  stability_score_threshold: z.number().min(0).max(1).default(0.9), // 90% stability
  confidence_threshold: z.number().min(0).max(1).default(0.95), // 95% confidence
  
  // Early stopping criteria
  enable_early_stopping: z.boolean().default(true),
  futility_boundary: z.number().min(0).max(1).default(0.1), // Stop if P(success) < 10%
  superiority_boundary: z.number().min(0).max(1).default(0.99), // Stop if P(success) > 99%
  
  // Monitoring and alerting
  enable_real_time_monitoring: z.boolean().default(true),
  monitoring_interval_minutes: z.number().min(1).default(5),
  alert_on_power_drop: z.boolean().default(true),
  minimum_power_threshold: z.number().min(0).max(1).default(0.7), // Alert if power < 70%
});

export type StatisticalValidationConfig = z.infer<typeof StatisticalValidationConfigSchema>;

// Sample data point for analysis
export interface SampleDataPoint {
  id: string;
  timestamp: number;
  group: 'control' | 'treatment';
  
  // Performance metrics
  latency_ms: number;
  quality_score: number;
  stability_score: number;
  efficiency_score: number;
  
  // Metadata
  query_type: string;
  candidate_count: number;
  token_budget: number;
  
  // Success indicators
  latency_target_met: boolean;
  quality_target_met: boolean;
  overall_success: boolean;
}

// Statistical test result
export interface StatisticalTestResult {
  test_name: string;
  metric: string;
  
  // Classical statistics
  test_statistic: number;
  p_value: number;
  adjusted_p_value: number;
  degrees_of_freedom?: number;
  
  // Effect size
  effect_size: number;
  cohens_d?: number;
  confidence_interval: [number, number];
  
  // Power analysis
  observed_power: number;
  required_sample_size: number;
  current_sample_size: number;
  
  // Decision
  significant: boolean;
  meets_promotion_criteria: boolean;
  recommendation: 'promote' | 'continue' | 'stop_futility' | 'investigate';
}

// Bayesian analysis result
export interface BayesianAnalysisResult {
  metric: string;
  
  // Posterior distribution parameters
  posterior_mean: number;
  posterior_std: number;
  credible_interval: [number, number];
  
  // Probabilities
  prob_improvement: number; // P(treatment > control)
  prob_meets_threshold: number; // P(improvement > threshold)
  prob_regression: number; // P(treatment < control)
  
  // Decision criteria
  bayes_factor: number;
  expected_loss: number;
  value_of_information: number;
  
  // Recommendation
  posterior_recommendation: 'strong_evidence' | 'moderate_evidence' | 'weak_evidence' | 'insufficient_evidence';
}

// Sequential testing result
export interface SequentialTestingResult {
  test_name: string;
  current_stage: number;
  
  // Boundaries
  superiority_boundary: number;
  futility_boundary: number;
  current_test_statistic: number;
  
  // Decision
  stop_for_superiority: boolean;
  stop_for_futility: boolean;
  continue_testing: boolean;
  
  // Projections
  estimated_completion_samples: number;
  probability_eventual_success: number;
  expected_effect_size: number;
}

// Comprehensive validation result
export interface ValidationSummary {
  overall_recommendation: 'promote' | 'continue_testing' | 'stop_futility' | 'rollback';
  confidence_score: number; // 0-1
  
  // Statistical evidence
  classical_tests: StatisticalTestResult[];
  bayesian_analysis: BayesianAnalysisResult[];
  sequential_results: SequentialTestingResult[];
  
  // Promotion criteria assessment
  promotion_criteria: {
    latency_improvement: { achieved: boolean; current: number; threshold: number };
    quality_preservation: { achieved: boolean; current: number; threshold: number };
    stability_score: { achieved: boolean; current: number; threshold: number };
    statistical_significance: { achieved: boolean; p_value: number; threshold: number };
  };
  
  // Meta-analysis
  sample_sizes: { control: number; treatment: number };
  test_duration_minutes: number;
  power_achieved: number;
  multiple_testing_penalty: number;
  
  // Risk assessment
  risk_assessment: {
    type_i_error_risk: number;
    type_ii_error_risk: number;
    false_discovery_risk: number;
    regression_risk: number;
  };
  
  // Next actions
  next_steps: string[];
  estimated_time_to_decision: number;
  required_additional_samples: number;
}

/**
 * Statistical Validation Engine
 * 
 * Implements comprehensive statistical testing for A/B experiments
 * with proper power analysis, multiple testing correction, and Bayesian updates.
 */
export class StatisticalValidationEngine {
  private config: StatisticalValidationConfig;
  private sample_data: SampleDataPoint[] = [];
  private test_start_time: number = Date.now();
  private sequential_stage: number = 0;
  
  // Cached analysis results
  private latest_validation?: ValidationSummary;
  private bayesian_priors: Map<string, {mean: number, precision: number}> = new Map();
  
  constructor(config: Partial<StatisticalValidationConfig> = {}) {
    this.config = StatisticalValidationConfigSchema.parse(config);
    
    // Initialize Bayesian priors
    this.initializeBayesianPriors();
    
    console.log('📊 Statistical Validation Engine initialized');
    console.log(`   Significance level: α = ${this.config.significance_level}`);
    console.log(`   Statistical power: 1-β = ${this.config.statistical_power}`);
    console.log(`   Minimum effect size: ${this.config.minimum_effect_size}`);
    console.log(`   Correction method: ${this.config.correction_method}`);
  }
  
  /**
   * Add sample data point
   */
  addSample(sample: SampleDataPoint): void {
    this.sample_data.push(sample);
    
    // Trigger sequential testing if enabled
    if (this.config.enable_early_stopping && 
        this.sample_data.length % this.config.sequential_testing_interval === 0) {
      this.performSequentialTesting();
    }
    
    // Update Bayesian posteriors
    if (this.config.enable_bayesian_updates && 
        this.sample_data.length % this.config.posterior_update_interval === 0) {
      this.updateBayesianPosteriors();
    }
  }
  
  /**
   * Perform comprehensive statistical validation
   */
  performValidation(): ValidationSummary {
    console.log('🧮 Performing comprehensive statistical validation...');
    
    // Separate control and treatment groups
    const control_samples = this.sample_data.filter(s => s.group === 'control');
    const treatment_samples = this.sample_data.filter(s => s.group === 'treatment');
    
    if (control_samples.length < 10 || treatment_samples.length < 10) {
      console.log('⚠️ Insufficient sample size for validation');
      return this.createInsufficientDataSummary();
    }
    
    // Perform classical statistical tests
    const classical_tests = this.performClassicalTests(control_samples, treatment_samples);
    
    // Perform Bayesian analysis
    const bayesian_analysis = this.performBayesianAnalysis(control_samples, treatment_samples);
    
    // Perform sequential testing analysis
    const sequential_results = this.performSequentialAnalysis(control_samples, treatment_samples);
    
    // Assess promotion criteria
    const promotion_criteria = this.assessPromotionCriteria(classical_tests, bayesian_analysis);
    
    // Calculate meta-analysis metrics
    const meta_analysis = this.calculateMetaAnalysis(control_samples, treatment_samples);
    
    // Perform risk assessment
    const risk_assessment = this.assessRisks(classical_tests, bayesian_analysis);
    
    // Make overall recommendation
    const overall_recommendation = this.makeOverallRecommendation(
      promotion_criteria,
      classical_tests,
      bayesian_analysis,
      sequential_results,
      risk_assessment
    );
    
    this.latest_validation = {
      overall_recommendation: overall_recommendation.recommendation,
      confidence_score: overall_recommendation.confidence,
      
      classical_tests,
      bayesian_analysis,
      sequential_results,
      
      promotion_criteria,
      
      sample_sizes: { 
        control: control_samples.length, 
        treatment: treatment_samples.length 
      },
      test_duration_minutes: (Date.now() - this.test_start_time) / 60000,
      power_achieved: meta_analysis.power_achieved,
      multiple_testing_penalty: meta_analysis.multiple_testing_penalty,
      
      risk_assessment,
      
      next_steps: overall_recommendation.next_steps,
      estimated_time_to_decision: overall_recommendation.estimated_time,
      required_additional_samples: overall_recommendation.additional_samples,
    };
    
    console.log(`✅ Statistical validation complete: ${overall_recommendation.recommendation} (${(overall_recommendation.confidence * 100).toFixed(1)}% confidence)`);
    
    return this.latest_validation;
  }
  
  /**
   * Perform classical statistical tests
   */
  private performClassicalTests(
    control: SampleDataPoint[], 
    treatment: SampleDataPoint[]
  ): StatisticalTestResult[] {
    const results: StatisticalTestResult[] = [];
    
    // Test latency improvement
    results.push(this.performTTest(
      'latency_improvement',
      'latency_ms',
      control.map(s => s.latency_ms),
      treatment.map(s => s.latency_ms),
      'less' // Treatment should have lower latency
    ));
    
    // Test quality preservation
    results.push(this.performTTest(
      'quality_preservation',
      'quality_score',
      control.map(s => s.quality_score),
      treatment.map(s => s.quality_score),
      'greater' // Treatment should have equal or higher quality
    ));
    
    // Test stability score
    results.push(this.performTTest(
      'stability_improvement',
      'stability_score',
      control.map(s => s.stability_score),
      treatment.map(s => s.stability_score),
      'greater'
    ));
    
    // Test efficiency score
    results.push(this.performTTest(
      'efficiency_improvement',
      'efficiency_score',
      control.map(s => s.efficiency_score),
      treatment.map(s => s.efficiency_score),
      'greater'
    ));
    
    // Apply multiple testing correction
    return this.applyMultipleTestingCorrection(results);
  }
  
  /**
   * Perform t-test between two groups
   */
  private performTTest(
    test_name: string,
    metric: string,
    control_values: number[],
    treatment_values: number[],
    alternative: 'two_sided' | 'greater' | 'less'
  ): StatisticalTestResult {
    // Calculate basic statistics
    const control_mean = this.mean(control_values);
    const treatment_mean = this.mean(treatment_values);
    const control_std = this.stddev(control_values);
    const treatment_std = this.stddev(treatment_values);
    
    const n1 = control_values.length;
    const n2 = treatment_values.length;
    
    // Pooled standard error
    const pooled_std = Math.sqrt(
      ((n1 - 1) * control_std * control_std + (n2 - 1) * treatment_std * treatment_std) / 
      (n1 + n2 - 2)
    );
    const standard_error = pooled_std * Math.sqrt(1/n1 + 1/n2);
    
    // T-statistic
    const t_statistic = (treatment_mean - control_mean) / standard_error;
    const degrees_of_freedom = n1 + n2 - 2;
    
    // P-value (simplified - would use proper t-distribution in production)
    let p_value = 2 * (1 - this.normalCDF(Math.abs(t_statistic)));
    if (alternative === 'greater') {
      p_value = 1 - this.normalCDF(t_statistic);
    } else if (alternative === 'less') {
      p_value = this.normalCDF(t_statistic);
    }
    
    // Effect size (Cohen's d)
    const cohens_d = (treatment_mean - control_mean) / pooled_std;
    const effect_size = Math.abs(cohens_d);
    
    // Confidence interval for effect size
    const margin_of_error = 1.96 * standard_error / pooled_std; // Approximate
    const confidence_interval: [number, number] = [
      cohens_d - margin_of_error,
      cohens_d + margin_of_error
    ];
    
    // Power analysis
    const observed_power = this.calculatePower(effect_size, n1, n2, this.config.significance_level);
    const required_sample_size = this.calculateRequiredSampleSize(
      this.config.minimum_effect_size,
      this.config.statistical_power,
      this.config.significance_level
    );
    
    // Decision
    const significant = p_value < this.config.significance_level;
    const meets_promotion_criteria = significant && 
      effect_size >= this.config.minimum_effect_size;
    
    let recommendation: StatisticalTestResult['recommendation'] = 'continue';
    if (meets_promotion_criteria) {
      recommendation = 'promote';
    } else if (n1 + n2 > required_sample_size * 2) {
      recommendation = 'stop_futility';
    } else if (p_value > 0.5) {
      recommendation = 'investigate';
    }
    
    return {
      test_name,
      metric,
      test_statistic: t_statistic,
      p_value,
      adjusted_p_value: p_value, // Will be adjusted by multiple testing correction
      degrees_of_freedom,
      effect_size,
      cohens_d,
      confidence_interval,
      observed_power,
      required_sample_size,
      current_sample_size: n1 + n2,
      significant,
      meets_promotion_criteria,
      recommendation,
    };
  }
  
  /**
   * Apply multiple testing correction
   */
  private applyMultipleTestingCorrection(results: StatisticalTestResult[]): StatisticalTestResult[] {
    const p_values = results.map(r => r.p_value);
    let adjusted_p_values: number[];
    
    switch (this.config.correction_method) {
      case 'bonferroni':
        adjusted_p_values = p_values.map(p => Math.min(1.0, p * p_values.length));
        break;
      case 'fdr':
        adjusted_p_values = this.benjaminiHochberg(p_values);
        break;
      case 'holm':
        adjusted_p_values = this.holmCorrection(p_values);
        break;
      default:
        adjusted_p_values = p_values;
    }
    
    return results.map((result, i) => ({
      ...result,
      adjusted_p_value: adjusted_p_values[i],
      significant: adjusted_p_values[i] < this.config.significance_level,
    }));
  }
  
  /**
   * Benjamini-Hochberg FDR correction
   */
  private benjaminiHochberg(p_values: number[]): number[] {
    const n = p_values.length;
    const sorted_indices = Array.from({length: n}, (_, i) => i)
      .sort((a, b) => p_values[a] - p_values[b]);
    
    const adjusted = new Array(n);
    
    for (let i = n - 1; i >= 0; i--) {
      const idx = sorted_indices[i];
      const rank = i + 1;
      const adjusted_p = p_values[idx] * n / rank;
      
      if (i === n - 1) {
        adjusted[idx] = Math.min(1.0, adjusted_p);
      } else {
        const next_idx = sorted_indices[i + 1];
        adjusted[idx] = Math.min(adjusted[next_idx], adjusted_p);
      }
    }
    
    return adjusted;
  }
  
  /**
   * Holm correction
   */
  private holmCorrection(p_values: number[]): number[] {
    const n = p_values.length;
    const sorted_indices = Array.from({length: n}, (_, i) => i)
      .sort((a, b) => p_values[a] - p_values[b]);
    
    const adjusted = new Array(n);
    
    for (let i = 0; i < n; i++) {
      const idx = sorted_indices[i];
      const multiplier = n - i;
      adjusted[idx] = Math.min(1.0, p_values[idx] * multiplier);
      
      if (i > 0) {
        const prev_idx = sorted_indices[i - 1];
        adjusted[idx] = Math.max(adjusted[idx], adjusted[prev_idx]);
      }
    }
    
    return adjusted;
  }
  
  /**
   * Perform Bayesian analysis
   */
  private performBayesianAnalysis(
    control: SampleDataPoint[], 
    treatment: SampleDataPoint[]
  ): BayesianAnalysisResult[] {
    const results: BayesianAnalysisResult[] = [];
    
    // Analyze each metric
    const metrics = ['latency_ms', 'quality_score', 'stability_score', 'efficiency_score'];
    
    for (const metric of metrics) {
      const control_values = control.map(s => s[metric as keyof SampleDataPoint] as number);
      const treatment_values = treatment.map(s => s[metric as keyof SampleDataPoint] as number);
      
      results.push(this.performBayesianTTest(metric, control_values, treatment_values));
    }
    
    return results;
  }
  
  /**
   * Perform Bayesian t-test
   */
  private performBayesianTTest(
    metric: string,
    control_values: number[],
    treatment_values: number[]
  ): BayesianAnalysisResult {
    // Get prior parameters
    const prior = this.bayesian_priors.get(metric) || { mean: 0, precision: this.config.prior_confidence };
    
    // Calculate sample statistics
    const control_mean = this.mean(control_values);
    const treatment_mean = this.mean(treatment_values);
    const control_var = this.variance(control_values);
    const treatment_var = this.variance(treatment_values);
    
    const n_control = control_values.length;
    const n_treatment = treatment_values.length;
    
    // Bayesian posterior for difference in means (simplified)
    const diff_observed = treatment_mean - control_mean;
    const diff_var = control_var / n_control + treatment_var / n_treatment;
    
    // Update posterior (normal-normal conjugacy)
    const posterior_precision = prior.precision + 1 / diff_var;
    const posterior_mean = (prior.mean * prior.precision + diff_observed / diff_var) / posterior_precision;
    const posterior_std = Math.sqrt(1 / posterior_precision);
    
    // Credible interval
    const z_score = 1.96; // 95% CI
    const credible_interval: [number, number] = [
      posterior_mean - z_score * posterior_std,
      posterior_mean + z_score * posterior_std
    ];
    
    // Probabilities of interest
    const prob_improvement = 1 - this.normalCDF((0 - posterior_mean) / posterior_std);
    const threshold = metric === 'latency_ms' ? -this.config.minimum_effect_size : this.config.minimum_effect_size;
    const prob_meets_threshold = metric === 'latency_ms' ? 
      this.normalCDF((threshold - posterior_mean) / posterior_std) :
      1 - this.normalCDF((threshold - posterior_mean) / posterior_std);
    const prob_regression = 1 - prob_improvement;
    
    // Bayes factor (simplified)
    const bayes_factor = this.calculateBayesFactor(posterior_mean, posterior_std, prior.mean, Math.sqrt(1/prior.precision));
    
    // Expected loss and value of information (simplified)
    const expected_loss = Math.abs(posterior_mean - threshold) * prob_regression;
    const value_of_information = posterior_std * posterior_std; // Uncertainty reduction
    
    // Recommendation based on probabilities
    let posterior_recommendation: BayesianAnalysisResult['posterior_recommendation'];
    if (prob_meets_threshold > 0.95) {
      posterior_recommendation = 'strong_evidence';
    } else if (prob_meets_threshold > 0.8) {
      posterior_recommendation = 'moderate_evidence';
    } else if (prob_meets_threshold > 0.6) {
      posterior_recommendation = 'weak_evidence';
    } else {
      posterior_recommendation = 'insufficient_evidence';
    }
    
    return {
      metric,
      posterior_mean,
      posterior_std,
      credible_interval,
      prob_improvement,
      prob_meets_threshold,
      prob_regression,
      bayes_factor,
      expected_loss,
      value_of_information,
      posterior_recommendation,
    };
  }
  
  /**
   * Initialize Bayesian priors
   */
  private initializeBayesianPriors(): void {
    // Weak priors centered around no effect
    this.bayesian_priors.set('latency_ms', { mean: 0, precision: this.config.prior_confidence });
    this.bayesian_priors.set('quality_score', { mean: 0, precision: this.config.prior_confidence });
    this.bayesian_priors.set('stability_score', { mean: 0, precision: this.config.prior_confidence });
    this.bayesian_priors.set('efficiency_score', { mean: 0, precision: this.config.prior_confidence });
  }
  
  /**
   * Update Bayesian posteriors
   */
  private updateBayesianPosteriors(): void {
    const control = this.sample_data.filter(s => s.group === 'control');
    const treatment = this.sample_data.filter(s => s.group === 'treatment');
    
    if (control.length < 5 || treatment.length < 5) return;
    
    // Update priors based on observed data (simplified)
    const metrics = ['latency_ms', 'quality_score', 'stability_score', 'efficiency_score'];
    
    for (const metric of metrics) {
      const control_values = control.map(s => s[metric as keyof SampleDataPoint] as number);
      const treatment_values = treatment.map(s => s[metric as keyof SampleDataPoint] as number);
      
      const diff_observed = this.mean(treatment_values) - this.mean(control_values);
      const diff_var = this.variance(treatment_values) / treatment_values.length + 
                      this.variance(control_values) / control_values.length;
      
      const current_prior = this.bayesian_priors.get(metric)!;
      const new_precision = current_prior.precision + 1 / diff_var;
      const new_mean = (current_prior.mean * current_prior.precision + diff_observed / diff_var) / new_precision;
      
      this.bayesian_priors.set(metric, { mean: new_mean, precision: new_precision });
    }
  }
  
  /**
   * Perform sequential testing analysis
   */
  private performSequentialAnalysis(
    control: SampleDataPoint[], 
    treatment: SampleDataPoint[]
  ): SequentialTestingResult[] {
    // This is a simplified implementation
    // In production, would implement proper sequential boundaries (O'Brien-Fleming, Pocock, etc.)
    
    const results: SequentialTestingResult[] = [];
    const total_samples = control.length + treatment.length;
    
    // Check primary endpoint (latency improvement)
    const latency_improvement = this.calculateLatencyImprovement(control, treatment);
    const improvement_z_score = latency_improvement / 0.1; // Simplified
    
    const superiority_boundary = 2.0; // Approximately p < 0.05
    const futility_boundary = -1.0;
    
    results.push({
      test_name: 'sequential_latency',
      current_stage: this.sequential_stage,
      superiority_boundary,
      futility_boundary,
      current_test_statistic: improvement_z_score,
      stop_for_superiority: improvement_z_score > superiority_boundary,
      stop_for_futility: improvement_z_score < futility_boundary,
      continue_testing: improvement_z_score >= futility_boundary && improvement_z_score <= superiority_boundary,
      estimated_completion_samples: this.config.maximum_sample_size_per_group * 2,
      probability_eventual_success: this.normalCDF(improvement_z_score),
      expected_effect_size: latency_improvement,
    });
    
    return results;
  }
  
  /**
   * Assess promotion criteria
   */
  private assessPromotionCriteria(
    classical_tests: StatisticalTestResult[],
    bayesian_analysis: BayesianAnalysisResult[]
  ): ValidationSummary['promotion_criteria'] {
    // Find relevant tests
    const latency_test = classical_tests.find(t => t.test_name === 'latency_improvement');
    const quality_test = classical_tests.find(t => t.test_name === 'quality_preservation');
    const stability_test = classical_tests.find(t => t.test_name === 'stability_improvement');
    
    const latency_bayes = bayesian_analysis.find(b => b.metric === 'latency_ms');
    const quality_bayes = bayesian_analysis.find(b => b.metric === 'quality_score');
    const stability_bayes = bayesian_analysis.find(b => b.metric === 'stability_score');
    
    return {
      latency_improvement: {
        achieved: (latency_test?.meets_promotion_criteria || false) && 
                 (latency_bayes?.prob_meets_threshold || 0) > this.config.confidence_threshold,
        current: latency_test?.effect_size || 0,
        threshold: this.config.latency_improvement_threshold,
      },
      quality_preservation: {
        achieved: (quality_test?.significant || false) && 
                 (quality_bayes?.prob_improvement || 0) > this.config.quality_preservation_threshold,
        current: quality_bayes?.posterior_mean || 0,
        threshold: this.config.quality_preservation_threshold,
      },
      stability_score: {
        achieved: (stability_test?.significant || false) && 
                 (stability_bayes?.prob_improvement || 0) > this.config.stability_score_threshold,
        current: stability_bayes?.posterior_mean || 0,
        threshold: this.config.stability_score_threshold,
      },
      statistical_significance: {
        achieved: classical_tests.every(t => t.significant),
        p_value: Math.max(...classical_tests.map(t => t.adjusted_p_value)),
        threshold: this.config.significance_level,
      },
    };
  }
  
  /**
   * Calculate meta-analysis metrics
   */
  private calculateMetaAnalysis(
    control: SampleDataPoint[], 
    treatment: SampleDataPoint[]
  ): {
    power_achieved: number;
    multiple_testing_penalty: number;
  } {
    const total_samples = control.length + treatment.length;
    const effect_size = this.calculateOverallEffectSize(control, treatment);
    
    const power_achieved = this.calculatePower(
      effect_size,
      control.length,
      treatment.length,
      this.config.significance_level
    );
    
    const multiple_testing_penalty = this.config.correction_method === 'bonferroni' ? 
      this.config.num_simultaneous_tests : 
      Math.sqrt(this.config.num_simultaneous_tests); // Approximate for FDR
    
    return {
      power_achieved,
      multiple_testing_penalty,
    };
  }
  
  /**
   * Assess statistical risks
   */
  private assessRisks(
    classical_tests: StatisticalTestResult[],
    bayesian_analysis: BayesianAnalysisResult[]
  ): ValidationSummary['risk_assessment'] {
    const max_p_value = Math.max(...classical_tests.map(t => t.adjusted_p_value));
    const min_power = Math.min(...classical_tests.map(t => t.observed_power));
    
    return {
      type_i_error_risk: max_p_value,
      type_ii_error_risk: 1 - min_power,
      false_discovery_risk: this.config.correction_method === 'fdr' ? this.config.significance_level : max_p_value,
      regression_risk: Math.max(...bayesian_analysis.map(b => b.prob_regression)),
    };
  }
  
  /**
   * Make overall recommendation
   */
  private makeOverallRecommendation(
    promotion_criteria: ValidationSummary['promotion_criteria'],
    classical_tests: StatisticalTestResult[],
    bayesian_analysis: BayesianAnalysisResult[],
    sequential_results: SequentialTestingResult[],
    risk_assessment: ValidationSummary['risk_assessment']
  ): {
    recommendation: ValidationSummary['overall_recommendation'];
    confidence: number;
    next_steps: string[];
    estimated_time: number;
    additional_samples: number;
  } {
    const all_criteria_met = Object.values(promotion_criteria).every(c => c.achieved);
    const strong_bayesian_evidence = bayesian_analysis.every(b => 
      b.posterior_recommendation === 'strong_evidence' || b.posterior_recommendation === 'moderate_evidence'
    );
    
    const should_stop_for_superiority = sequential_results.some(s => s.stop_for_superiority);
    const should_stop_for_futility = sequential_results.some(s => s.stop_for_futility);
    
    const high_risk = risk_assessment.type_i_error_risk > this.config.significance_level * 2 ||
                     risk_assessment.regression_risk > 0.2;
    
    let recommendation: ValidationSummary['overall_recommendation'];
    let confidence: number;
    const next_steps: string[] = [];
    let estimated_time = 0;
    let additional_samples = 0;
    
    if (should_stop_for_futility || high_risk) {
      recommendation = 'stop_futility';
      confidence = 0.9;
      next_steps.push('Stop testing - insufficient evidence of improvement');
      next_steps.push('Consider alternative optimization strategies');
    } else if (all_criteria_met && strong_bayesian_evidence && should_stop_for_superiority) {
      recommendation = 'promote';
      confidence = 0.95;
      next_steps.push('Proceed with full deployment');
      next_steps.push('Continue monitoring performance metrics');
    } else if (promotion_criteria.latency_improvement.achieved && promotion_criteria.statistical_significance.achieved) {
      recommendation = 'promote';
      confidence = 0.8;
      next_steps.push('Proceed with cautious deployment');
      next_steps.push('Enhance quality monitoring');
    } else {
      recommendation = 'continue_testing';
      confidence = 0.6;
      
      if (!promotion_criteria.latency_improvement.achieved) {
        next_steps.push('Continue collecting latency improvement data');
      }
      if (!promotion_criteria.quality_preservation.achieved) {
        next_steps.push('Monitor quality preservation closely');
      }
      if (!promotion_criteria.statistical_significance.achieved) {
        next_steps.push('Increase sample size for statistical power');
        additional_samples = Math.max(...classical_tests.map(t => t.required_sample_size)) - 
                           this.sample_data.length;
        estimated_time = additional_samples * 2; // Approximate minutes
      }
    }
    
    return {
      recommendation,
      confidence,
      next_steps,
      estimated_time,
      additional_samples,
    };
  }
  
  /**
   * Utility methods
   */
  
  private mean(values: number[]): number {
    return values.reduce((a, b) => a + b, 0) / values.length;
  }
  
  private variance(values: number[]): number {
    const m = this.mean(values);
    return values.reduce((sum, x) => sum + Math.pow(x - m, 2), 0) / (values.length - 1);
  }
  
  private stddev(values: number[]): number {
    return Math.sqrt(this.variance(values));
  }
  
  private normalCDF(x: number): number {
    // Approximation of standard normal CDF
    return 0.5 * (1 + this.erf(x / Math.sqrt(2)));
  }
  
  private erf(x: number): number {
    // Approximation of error function
    const a1 =  0.254829592;
    const a2 = -0.284496736;
    const a3 =  1.421413741;
    const a4 = -1.453152027;
    const a5 =  1.061405429;
    const p  =  0.3275911;
    
    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x);
    
    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    
    return sign * y;
  }
  
  private calculatePower(effect_size: number, n1: number, n2: number, alpha: number): number {
    // Simplified power calculation for two-sample t-test
    const df = n1 + n2 - 2;
    const ncp = effect_size * Math.sqrt((n1 * n2) / (n1 + n2)); // Non-centrality parameter
    const critical_t = 1.96; // Approximate for alpha = 0.05
    
    // Power = P(reject H0 | H1 is true)
    return 1 - this.normalCDF((critical_t - ncp) / Math.sqrt(1));
  }
  
  private calculateRequiredSampleSize(effect_size: number, power: number, alpha: number): number {
    // Simplified sample size calculation
    const z_alpha = 1.96; // For alpha = 0.05
    const z_beta = 0.84;  // For power = 0.8
    
    return Math.ceil(2 * Math.pow((z_alpha + z_beta) / effect_size, 2));
  }
  
  private calculateBayesFactor(
    posterior_mean: number, 
    posterior_std: number, 
    prior_mean: number, 
    prior_std: number
  ): number {
    // Simplified Bayes factor calculation
    const likelihood_h1 = Math.exp(-0.5 * Math.pow((posterior_mean - 0) / posterior_std, 2));
    const likelihood_h0 = Math.exp(-0.5 * Math.pow((prior_mean - 0) / prior_std, 2));
    
    return likelihood_h1 / likelihood_h0;
  }
  
  private calculateLatencyImprovement(control: SampleDataPoint[], treatment: SampleDataPoint[]): number {
    const control_mean = this.mean(control.map(s => s.latency_ms));
    const treatment_mean = this.mean(treatment.map(s => s.latency_ms));
    
    return (control_mean - treatment_mean) / control_mean;
  }
  
  private calculateOverallEffectSize(control: SampleDataPoint[], treatment: SampleDataPoint[]): number {
    // Composite effect size across all metrics
    const latency_effect = Math.abs(this.calculateLatencyImprovement(control, treatment));
    const quality_effect = Math.abs(
      this.mean(treatment.map(s => s.quality_score)) - this.mean(control.map(s => s.quality_score))
    ) / this.mean(control.map(s => s.quality_score));
    
    return Math.sqrt((latency_effect * latency_effect + quality_effect * quality_effect) / 2);
  }
  
  private createInsufficientDataSummary(): ValidationSummary {
    return {
      overall_recommendation: 'continue_testing',
      confidence_score: 0.1,
      classical_tests: [],
      bayesian_analysis: [],
      sequential_results: [],
      promotion_criteria: {
        latency_improvement: { achieved: false, current: 0, threshold: this.config.latency_improvement_threshold },
        quality_preservation: { achieved: false, current: 0, threshold: this.config.quality_preservation_threshold },
        stability_score: { achieved: false, current: 0, threshold: this.config.stability_score_threshold },
        statistical_significance: { achieved: false, p_value: 1.0, threshold: this.config.significance_level },
      },
      sample_sizes: { control: 0, treatment: 0 },
      test_duration_minutes: 0,
      power_achieved: 0,
      multiple_testing_penalty: 1,
      risk_assessment: {
        type_i_error_risk: 1.0,
        type_ii_error_risk: 1.0,
        false_discovery_risk: 1.0,
        regression_risk: 0.5,
      },
      next_steps: ['Collect more sample data', 'Ensure balanced group assignment'],
      estimated_time_to_decision: this.config.minimum_sample_size_per_group * 2,
      required_additional_samples: this.config.minimum_sample_size_per_group * 2,
    };
  }
  
  /**
   * Public API methods
   */
  
  /**
   * Get current validation status
   */
  getCurrentStatus(): {
    sample_count: number;
    latest_validation?: ValidationSummary;
    is_ready_for_decision: boolean;
    next_analysis_at: number;
  } {
    const is_ready = this.sample_data.length >= this.config.minimum_sample_size_per_group * 2;
    const next_analysis = this.config.sequential_testing_interval - 
      (this.sample_data.length % this.config.sequential_testing_interval);
    
    return {
      sample_count: this.sample_data.length,
      latest_validation: this.latest_validation,
      is_ready_for_decision: is_ready,
      next_analysis_at: next_analysis,
    };
  }
  
  /**
   * Reset validation state
   */
  reset(): void {
    this.sample_data = [];
    this.test_start_time = Date.now();
    this.sequential_stage = 0;
    this.latest_validation = undefined;
    this.initializeBayesianPriors();
    
    console.log('🔄 Statistical validation engine reset');
  }
  
  /**
   * Perform sequential testing check
   */
  private performSequentialTesting(): void {
    this.sequential_stage++;
    console.log(`📊 Sequential testing stage ${this.sequential_stage}: ${this.sample_data.length} samples`);
    
    // Trigger full validation if enough samples
    if (this.sample_data.length >= this.config.minimum_sample_size_per_group * 2) {
      this.performValidation();
    }
  }
}

// Default configuration optimized for deployment validation
export const DEFAULT_STATISTICAL_VALIDATION_CONFIG: StatisticalValidationConfig = 
  StatisticalValidationConfigSchema.parse({
    significance_level: 0.05,
    statistical_power: 0.8,
    minimum_effect_size: 0.1,
    minimum_sample_size_per_group: 100,
    correction_method: 'fdr',
    enable_bayesian_updates: true,
    enable_early_stopping: true,
    latency_improvement_threshold: 0.68, // 80% of 85% target
    quality_preservation_threshold: 0.95,
    stability_score_threshold: 0.9,
  });