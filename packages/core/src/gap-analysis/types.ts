/**
 * Gap Analysis Framework Types for Lethe's "Gap→Tune→Verify" System
 * 
 * This module defines the core types for analyzing performance gaps, conducting
 * counterfactual analysis, and implementing automated tuning with validation.
 */

import { Config, PerformanceMetrics, Result, LetheError } from '../types.js';

// ============================================================================
// CORE GAP ANALYSIS TYPES
// ============================================================================

/**
 * Policy fingerprint capturing all tuning parameters for reproducibility
 */
export interface PolicyFingerprint {
  // Ranking parameters
  lambda: number;           // BM25 weight adjustment (±20%)
  mu: number;              // Vector weight adjustment (±10%)
  K2: number;              // Cross-encoder K adjustment (±30%)
  r: number;               // DPP rank parameter {12,14,16,24}
  head_keep: number;       // Head retention rate (±4pp)
  
  // Window/stride parameters  
  window_size: number;
  stride: number;
  
  // Cross-encoder parameters
  ce_early_exit_rate: number;
  tau: number;             // Group-split threshold (±0.1)
  
  // Quality gates
  curvature_threshold: number;
  proxy_gap_max: number;   // Maximum allowable proxy gap
  
  // Metadata
  policy_id: string;
  created_at: number;
  validation_status: 'pending' | 'validated' | 'rejected';
}

/**
 * Slice-specific performance gap identification
 */
export interface GapRecord {
  // Slice identification
  slice_id: string;
  dataset: string;
  keep_ratio: number;      // Budget constraint: 0.08, 0.15, 0.30
  k: number;              // Retrieval depth
  seed: number;           // Reproducibility seed
  
  // Performance delta analysis
  delta_map: {
    macro_p_at_5: number;           // P@5 gap vs best competitor
    cost_per_query: number;         // Cost efficiency gap
    latency_p95: number;           // p95 latency difference
    latency_p99_p95_ratio: number; // Tail ratio constraint
  };
  
  // Root cause feature analysis
  root_cause_features: {
    entity_entropy: number;         // Information density measure
    dup_rate: number;              // Content duplication rate
    type_mix: TypeMixProfile;      // Distribution of content types
    closure_depth: number;         // Average closure depth
    symbol_length_avg: number;     // Average symbol complexity
    language_distribution: LanguageProfile;
    kv_stability: number;          // KV prefix-Jaccard stability
  };
  
  // Current policy state
  policy_fingerprint: PolicyFingerprint;
  
  // Statistical validation
  statistical_separation: {
    is_significant: boolean;       // Paired bootstrap significance
    p_value: number;              // Holm-corrected p-value
    confidence_interval: [number, number];
    effect_size: number;          // Cohen's d
  };
  
  // Gap priority and urgency
  priority_score: number;        // Ranking score for tuning queue
  estimated_uplift: number;      // Predicted improvement potential
  
  // Timestamps and metadata
  created_at: number;
  updated_at: number;
  validation_runs: number;
  status: 'identified' | 'tuning' | 'validated' | 'deployed';
}

/**
 * Content type distribution profile for root cause analysis
 */
export interface TypeMixProfile {
  code_heavy: number;       // Fraction of CODE-heavy content
  error_heavy: number;      // Fraction of ERROR-heavy content  
  tool_heavy: number;       // Fraction of TOOL-heavy content
  prose_heavy: number;      // Fraction of prose content
  json_needle: number;      // Fraction containing JSON needles
}

/**
 * Language distribution profile for multilingual analysis
 */
export interface LanguageProfile {
  english: number;
  chinese: number;
  code_switch: number;      // Mixed language content
  programming_languages: Record<string, number>;
}

// ============================================================================
// COUNTERFACTUAL ANALYSIS TYPES
// ============================================================================

/**
 * Off-policy evaluation using Importance-weighted Policy Sampling (IPS)
 */
export interface CounterfactualAnalysis {
  slice_id: string;
  base_policy: PolicyFingerprint;
  
  // Virtual knob perturbations (no LLM calls needed)
  perturbations: PolicyPerturbation[];
  
  // Counterfactual uplift estimates
  uplift_frontier: CounterfactualUplift[];
  
  // Validation constraints
  constraint_violations: ConstraintViolation[];
  
  // IPS-specific metrics
  importance_weights: {
    effective_sample_size: number;
    weight_variance: number;
    bias_estimate: number;
  };
  
  created_at: number;
}

/**
 * Individual policy perturbation for counterfactual analysis
 */
export interface PolicyPerturbation {
  perturbation_id: string;
  parameter_changes: Partial<PolicyFingerprint>;
  
  // Expected outcomes (no actual LLM execution)
  expected_delta_p_at_5: number;
  expected_delta_latency: number;
  
  // Confidence bounds
  confidence_interval_p_at_5: [number, number];
  confidence_interval_latency: [number, number];
  
  // Constraint satisfaction
  satisfies_constraints: boolean;
  violation_reasons: string[];
}

/**
 * Counterfactual uplift estimation results
 */
export interface CounterfactualUplift {
  policy_variant: PolicyFingerprint;
  
  // Performance predictions
  predicted_p_at_5_improvement: number;
  predicted_latency_change: number;
  predicted_cost_efficiency: number;
  
  // Risk assessment
  uncertainty_score: number;       // Prediction uncertainty
  downside_risk: number;          // Potential performance regression
  
  // Feasibility
  implementation_complexity: 'low' | 'medium' | 'high';
  estimated_validation_time: number; // Minutes for full validation
}

/**
 * Constraint violations for policy perturbations
 */
export interface ConstraintViolation {
  constraint_type: 'ECE' | 'curvature' | 'proxy_gap' | 'latency_ratio' | 'kv_stability';
  current_value: number;
  threshold: number;
  severity: 'warning' | 'error' | 'critical';
  mitigation_suggestions: string[];
}

// ============================================================================
// AUTO-TUNING FRAMEWORK TYPES  
// ============================================================================

/**
 * Bayesian + rule-based hybrid tuning configuration
 */
export interface AutoTuningProfile {
  profile_name: string;
  domain_specialization: DomainSpecialization;
  
  // Bayesian optimization settings
  bayesian_config: {
    n_trials: number;              // Default: 12
    acquisition_function: 'EI' | 'UCB' | 'PI';
    exploration_weight: number;
    initial_points: number;
  };
  
  // Rule-based biases and constraints
  rule_biases: DomainSpecificBiases;
  hard_constraints: TuningConstraints;
  
  // Risk management
  jensen_risk_adjustment: {
    alpha: number;                // Risk aversion parameter
    cvar_threshold: number;       // CVaR threshold for latency
  };
  
  // Validation requirements
  validation_config: ValidationConfig;
}

/**
 * Domain-specific specializations for different content types
 */
export interface DomainSpecialization {
  domain_type: 'code_error_gaps' | 'tool_json_needles' | 'multilingual_codeswitch' | 'general';
  
  // Domain-specific parameter preferences
  preferred_ranges: Partial<Record<keyof PolicyFingerprint, [number, number]>>;
  
  // Feature weights for this domain
  feature_importance_weights: Record<string, number>;
  
  // Success metric priorities
  metric_priorities: {
    p_at_5_weight: number;
    latency_weight: number;
    cost_efficiency_weight: number;
    stability_weight: number;
  };
}

/**
 * Domain-specific biases for different problem types
 */
export interface DomainSpecificBiases {
  // Code/ERROR gaps biases
  code_error?: {
    closure_strength_bias: number;    // Favor stronger closures
    r_preference: number;            // Preferred r value (e.g., 16)
    K2_boost_percent: number;        // K2 increase percentage
    lambda_adjustment: number;       // BM25 weight adjustment
    enable_summaries: boolean;
  };
  
  // Tool/JSON needles biases  
  tool_json?: {
    K2_boost_percent: number;        // Aggressive K2 boost (e.g., 25%)
    lambda_adjustment: number;       // BM25 adjustment  
    ce_early_exit_disabled: boolean; // Disable early exit for low K
    precision_over_recall: boolean;
  };
  
  // Multilingual/code-switch biases
  multilingual?: {
    re_isotonic_enabled: boolean;    // Enable re-isotonic calibration
    ce_early_exit_cap_widened: boolean;
    mu_adjustment: number;           // Vector weight adjustment
    r_preference: number;            // Preferred r value
  };
}

/**
 * Hard constraints for tuning parameter exploration
 */
export interface TuningConstraints {
  // Performance gates (same as current validation)
  p95_geq_avg: boolean;            // p95 >= average requirement
  p99_p95_ratio_max: number;       // p99/p95 <= 2.5
  ece_threshold: number;           // ECE <= 0.08
  proxy_gap_max: number;           // Proxy gap <= 0.5%
  
  // Parameter bounds
  lambda_bounds: [number, number]; // e.g., [-20%, +20%]
  mu_bounds: [number, number];     // e.g., [-10%, +10%]  
  K2_bounds: [number, number];     // e.g., [-30%, +30%]
  r_allowed_values: number[];      // e.g., [12, 14, 16, 24]
  head_keep_bounds: [number, number]; // e.g., [±4pp]
  
  // Stability constraints
  kv_prefix_jaccard_min: number;   // Minimum KV stability
  curvature_based_r_capping: boolean;
  
  // Group-split constraints
  tau_bounds: [number, number];    // e.g., [±0.1]
  ilp_usage_rate_max: number;      // Maximum ILP usage rate
}

/**
 * Validation configuration for tuned policies
 */
export interface ValidationConfig {
  // Paired replay requirements
  subset_size: number;             // e.g., M≈200 for quick validation
  full_matrix_size: number;        // Full validation size
  
  // Multi-budget validation
  budget_levels: number[];         // e.g., [8, 15, 30]
  
  // Statistical requirements  
  confidence_level: number;        // e.g., 0.95
  minimum_effect_size: number;     // Minimum meaningful improvement
  
  // Coverage requirements
  coverage_weighted_crps: boolean; // Enable CRPS coverage checks
  cross_domain_validation: boolean; // Validate across domains
}

// ============================================================================
// PROMOTION AND VALIDATION TYPES
// ============================================================================

/**
 * Promotion pipeline results for validated policies
 */
export interface PromotionResult {
  policy_id: string;
  source_gap: string;              // Original gap slice ID
  
  // Validation results
  gap_slice_validation: ValidationResult;
  union_set_validation: ValidationResult;
  cross_budget_validation: Record<number, ValidationResult>;
  
  // Performance improvements
  performance_gains: {
    p_at_5_improvement: number;
    latency_improvement: number;  
    cost_efficiency_gain: number;
    stability_score: number;
  };
  
  // Deployment readiness
  deployment_status: 'ready' | 'needs_review' | 'rejected';
  deployment_confidence: number;
  
  // Microsite integration
  pareto_front_annotation: ParetoFrontAnnotation;
  
  validation_timestamp: number;
  reviewer_notes?: string;
}

/**
 * Individual validation result for a specific test set
 */
export interface ValidationResult {
  test_set: string;
  sample_size: number;
  
  // Core metrics
  p_at_5_delta: number;
  p_at_5_ci: [number, number];
  latency_p95_delta: number;
  latency_p95_ci: [number, number];
  
  // Quality gates
  gates_passed: boolean;
  gate_violations: string[];
  
  // Statistical significance  
  is_significant: boolean;
  p_value: number;
  effect_size: number;
  
  // Detailed breakdown
  per_domain_results?: Record<string, Partial<ValidationResult>>;
}

/**
 * Pareto front annotation for microsite integration
 */
export interface ParetoFrontAnnotation {
  policy_label: string;            // e.g., "Tuned-v3 (Validated)"
  improvement_summary: string;     // Human-readable summary
  
  // Pareto coordinates
  cost_efficiency: number;
  performance_score: number;
  latency_score: number;
  
  // Visual markers
  marker_color: string;
  marker_size: number;
  highlight: boolean;
  
  // Detailed tooltip data
  tooltip_data: {
    domain_specialization: string;
    key_improvements: string[];
    validation_confidence: number;
    deployment_date: string;
  };
}

// ============================================================================
// DIFFICULTY GATING TYPES
// ============================================================================

/**
 * Lightweight GBM for adaptive policy initialization
 */
export interface DifficultyGate {
  model_id: string;
  
  // GBM configuration
  gbm_config: {
    n_estimators: number;
    max_depth: number;
    learning_rate: number;
    feature_columns: string[];
  };
  
  // Decision boundaries
  dimension_thresholds: {
    low_complexity: number;      // → 256 dims
    medium_complexity: number;   // → 512 dims  
    high_complexity: number;     // → 768 dims
  };
  
  k2_cap_rules: {
    easy_queries: number;        // Conservative K2 cap
    medium_queries: number;      // Standard K2 cap
    hard_queries: number;        // Aggressive K2 cap
  };
  
  // Feature extraction config
  feature_extractors: FeatureExtractorConfig[];
  
  model_accuracy: {
    cross_validation_score: number;
    precision_recall_auc: number;
    calibration_error: number;
  };
  
  last_trained: number;
  training_data_size: number;
}

/**
 * Feature extraction configuration for difficulty assessment
 */
export interface FeatureExtractorConfig {
  extractor_name: string;
  feature_type: 'gap_features' | 'query_features' | 'context_features';
  
  // Specific feature computations
  computations: {
    entity_entropy?: boolean;
    dup_intensity?: boolean;
    closure_depth?: boolean;
    symbol_complexity?: boolean;
    type_mix_variance?: boolean;
    language_switching_rate?: boolean;
    kv_instability_score?: boolean;
  };
  
  normalization: 'z_score' | 'min_max' | 'robust' | 'none';
  missing_value_strategy: 'mean' | 'median' | 'drop' | 'zero';
}

// ============================================================================
// SLICE MINING AND STRATIFICATION TYPES  
// ============================================================================

/**
 * Automatic slice mining and stratification results
 */
export interface SliceMiningResult {
  mining_run_id: string;
  
  // Stratification dimensions
  stratification_dimensions: {
    dataset_budget_k: Array<{
      dataset: string;
      keep_ratio: number;
      k: number;
      slice_count: number;
    }>;
    
    language_distribution: LanguageStratification;
    content_type_mix: ContentTypeStratification;
    complexity_bins: ComplexityBinStratification;
    stability_deciles: StabilityDecileStratification;
  };
  
  // Gap identification results
  identified_gaps: GapRecord[];
  
  // Statistical analysis
  statistical_summary: {
    total_slices_analyzed: number;
    significant_gaps_found: number;
    average_effect_size: number;
    multiple_testing_correction: 'holm' | 'bonferroni' | 'fdr';
  };
  
  // Prioritization
  tuning_queue: PrioritizedTuningQueue;
  
  mining_timestamp: number;
  computational_cost: {
    cpu_hours: number;
    memory_peak_gb: number;
    wall_clock_minutes: number;
  };
}

/**
 * Language-based stratification for multilingual analysis
 */
export interface LanguageStratification {
  pure_english: SliceGroup;
  pure_chinese: SliceGroup;
  code_switch_mixed: SliceGroup;
  programming_heavy: Record<string, SliceGroup>; // By programming language
}

/**
 * Content type stratification for domain-specific analysis  
 */
export interface ContentTypeStratification {
  code_heavy: SliceGroup;
  error_heavy: SliceGroup;
  tool_heavy: SliceGroup;
  json_needle: SliceGroup;
  prose_dominant: SliceGroup;
  mixed_content: SliceGroup;
}

/**
 * Complexity-based stratification using binning
 */
export interface ComplexityBinStratification {
  low_complexity: SliceGroup;     // Simple queries, shallow closures
  medium_complexity: SliceGroup;  // Standard complexity
  high_complexity: SliceGroup;    // Deep closures, high symbol density
  extreme_complexity: SliceGroup; // Edge cases, maximum complexity
}

/**
 * KV stability-based stratification using deciles
 */
export interface StabilityDecileStratification {
  deciles: SliceGroup[];          // 10 deciles of KV prefix-Jaccard stability
  unstable_outliers: SliceGroup;  // Bottom 5% most unstable
  highly_stable: SliceGroup;      // Top 5% most stable
}

/**
 * Individual slice group within stratification
 */
export interface SliceGroup {
  group_id: string;
  group_name: string;
  slice_ids: string[];
  
  // Aggregate statistics
  sample_size: number;
  performance_baseline: {
    mean_p_at_5: number;
    std_p_at_5: number;
    mean_latency_p95: number;
    std_latency_p95: number;
  };
  
  // Gap analysis summary
  gap_summary: {
    significant_gaps_count: number;
    average_gap_magnitude: number;
    priority_score_range: [number, number];
  };
  
  // Representative features
  feature_profile: {
    typical_entity_entropy: number;
    typical_dup_rate: number;
    typical_closure_depth: number;
    dominant_type_mix: TypeMixProfile;
  };
}

/**
 * Prioritized queue for tuning pipeline
 */
export interface PrioritizedTuningQueue {
  queue_items: TuningQueueItem[];
  total_estimated_time: number;    // Minutes for all items
  resource_requirements: {
    cpu_cores_needed: number;
    memory_gb_needed: number;
    gpu_required: boolean;
  };
  
  queue_created: number;
  expected_completion: number;
}

/**
 * Individual item in the tuning queue
 */
export interface TuningQueueItem {
  queue_position: number;
  gap_record_id: string;
  priority_score: number;
  
  // Estimated resource requirements
  estimated_tuning_time: number;  // Minutes
  estimated_validation_time: number;
  computational_complexity: 'low' | 'medium' | 'high';
  
  // Dependencies and constraints
  blocking_dependencies: string[]; // Other gap IDs that must complete first
  resource_constraints: string[];  // Special resource requirements
  
  // Expected outcomes
  predicted_improvement: {
    p_at_5_uplift: number;
    latency_improvement: number;
    confidence: number;
  };
  
  assigned_profile: string;        // Auto-tuning profile to use
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
}

// ============================================================================
// MONITORING AND OBSERVABILITY TYPES
// ============================================================================

/**
 * Comprehensive monitoring for the Gap→Tune→Verify system
 */
export interface GapAnalysisMonitoring {
  // System health
  system_health: {
    gap_mining_status: 'running' | 'idle' | 'error';
    counterfactual_engine_status: 'healthy' | 'degraded' | 'offline';
    tuning_pipeline_status: 'active' | 'backlogged' | 'maintenance';
    validation_system_status: 'operational' | 'limited' | 'offline';
  };
  
  // Performance metrics
  performance_metrics: {
    gaps_identified_per_hour: number;
    counterfactual_analyses_per_minute: number;
    tuning_jobs_completion_rate: number;
    validation_success_rate: number;
    
    // Latency metrics
    gap_detection_latency_p95: number;
    counterfactual_computation_latency_p95: number;
    tuning_iteration_latency_avg: number;
    end_to_end_pipeline_latency_p95: number;
  };
  
  // Quality metrics
  quality_metrics: {
    false_positive_gap_rate: number;
    counterfactual_accuracy: number;     // Actual vs predicted improvements
    tuning_success_rate: number;         // Fraction of tuning jobs that improve metrics
    validation_reliability: number;       // Consistency of validation results
    
    // Deployment metrics
    deployed_policy_performance: number; // Average performance gain in production
    policy_stability_score: number;      // How stable are deployed policies
  };
  
  // Resource utilization
  resource_utilization: {
    cpu_utilization_avg: number;
    memory_utilization_peak: number;
    gpu_utilization_avg: number;
    storage_usage_gb: number;
    
    // Cost tracking
    computational_cost_per_gap: number;
    validation_cost_per_policy: number;
    total_pipeline_cost_per_day: number;
  };
  
  // Alert conditions
  active_alerts: AlertCondition[];
  alert_history: AlertHistory[];
  
  last_updated: number;
}

/**
 * Alert condition for monitoring system
 */
export interface AlertCondition {
  alert_id: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  component: 'gap_mining' | 'counterfactual' | 'tuning' | 'validation' | 'deployment';
  
  condition_type: 'threshold_breach' | 'anomaly_detection' | 'system_failure' | 'performance_degradation';
  description: string;
  
  // Metrics that triggered the alert
  triggering_metrics: Record<string, number>;
  threshold_values: Record<string, number>;
  
  // Remediation
  suggested_actions: string[];
  auto_remediation_attempted: boolean;
  
  created_at: number;
  acknowledged_at?: number;
  resolved_at?: number;
}

/**
 * Historical alert tracking
 */
export interface AlertHistory {
  alert_id: string;
  occurrence_count: number;
  first_occurrence: number;
  last_occurrence: number;
  average_resolution_time: number;
  most_common_cause: string;
  
  // Trend analysis
  frequency_trend: 'increasing' | 'stable' | 'decreasing';
  severity_trend: 'escalating' | 'stable' | 'improving';
}

// ============================================================================
// RESULT TYPES AND ERROR HANDLING
// ============================================================================

/**
 * Standardized result type for Gap Analysis operations
 */
export type GapAnalysisResult<T> = Result<T, GapAnalysisError>;

/**
 * Specialized error types for Gap Analysis system
 */
export interface GapAnalysisError extends LetheError {
  error_type: 'gap_detection' | 'counterfactual_analysis' | 'tuning_failure' | 'validation_error' | 'deployment_error';
  
  // Context-specific error details
  gap_context?: {
    slice_id?: string;
    policy_id?: string;
    tuning_iteration?: number;
    validation_stage?: string;
  };
  
  // Recovery suggestions
  recovery_actions: string[];
  is_retryable: boolean;
  retry_delay_ms?: number;
  
  // Impact assessment
  impact_severity: 'low' | 'medium' | 'high' | 'critical';
  affected_components: string[];
}

/**
 * Batch operation result for handling multiple gap analysis operations
 */
export interface BatchGapAnalysisResult<T> {
  successful_operations: Array<{
    id: string;
    result: T;
    processing_time_ms: number;
  }>;
  
  failed_operations: Array<{
    id: string;
    error: GapAnalysisError;
    processing_time_ms: number;
  }>;
  
  // Aggregate statistics
  success_rate: number;
  total_processing_time_ms: number;
  average_operation_time_ms: number;
  
  batch_metadata: {
    batch_id: string;
    started_at: number;
    completed_at: number;
    total_operations: number;
  };
}

// ============================================================================
// CONFIGURATION AND INITIALIZATION TYPES
// ============================================================================

/**
 * Complete configuration for the Gap Analysis system
 */
export interface GapAnalysisConfig {
  // Core system configuration
  system: {
    enable_gap_mining: boolean;
    enable_counterfactual_analysis: boolean;
    enable_auto_tuning: boolean;
    enable_promotion_pipeline: boolean;
    enable_difficulty_gating: boolean;
  };
  
  // Mining configuration
  mining: {
    stratification_dimensions: string[];
    statistical_significance_threshold: number;
    multiple_testing_correction: 'holm' | 'bonferroni' | 'fdr';
    minimum_effect_size: number;
    mining_schedule_cron: string;
  };
  
  // Counterfactual analysis configuration
  counterfactual: {
    max_perturbations_per_gap: number;
    confidence_level: number;
    importance_sampling_method: 'naive' | 'self_normalized' | 'doubly_robust';
    variance_regularization: number;
  };
  
  // Auto-tuning configuration
  tuning: {
    max_concurrent_jobs: number;
    default_trial_count: number;
    timeout_per_trial_minutes: number;
    resource_limits: {
      cpu_cores_per_job: number;
      memory_gb_per_job: number;
      gpu_required: boolean;
    };
  };
  
  // Validation configuration
  validation: {
    paired_replay_sample_size: number;
    full_validation_sample_size: number;
    cross_validation_folds: number;
    bootstrap_iterations: number;
  };
  
  // Monitoring configuration
  monitoring: {
    metrics_collection_interval_seconds: number;
    alert_evaluation_interval_seconds: number;
    metric_retention_days: number;
    enable_detailed_tracing: boolean;
  };
  
  // Integration configuration
  integration: {
    microsite_webhook_url?: string;
    slack_notification_webhook?: string;
    email_notifications: string[];
    deployment_approval_required: boolean;
  };
}