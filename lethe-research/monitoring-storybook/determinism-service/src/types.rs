use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    pub slice_id: String,
    pub timestamp: DateTime<Utc>,
    pub result_hash: String,
    pub performance_metrics: PerformanceMetrics,
    pub invariants: InvariantChecks,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub duration_ms: u64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub p95_latency_ms: f64,
    pub throughput_ops_per_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantChecks {
    pub monotone_timestamps: bool,
    pub causal_ordering: bool,
    pub data_consistency: bool,
    pub structural_integrity: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterminismReport {
    pub slice_id: String,
    pub run_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub run1: ProcessingResult,
    pub run2: ProcessingResult,
    pub determinism_check: DeterminismCheck,
    pub performance_budget_check: PerformanceBudgetCheck,
    pub invariant_report: InvariantReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterminismCheck {
    pub is_deterministic: bool,
    pub hash_match: bool,
    pub timestamp_jitter_ms: u64,
    pub differences: Vec<String>,
    pub tolerance_met: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBudgetCheck {
    pub budget_met: bool,
    pub p95_latency_ms: f64,
    pub budget_threshold_ms: f64,
    pub performance_ratio: f64,
    pub sampling_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantReport {
    pub all_passed: bool,
    pub violations: Vec<InvariantViolation>,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantViolation {
    pub invariant_type: String,
    pub severity: ViolationSeverity,
    pub description: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub service_health: ServiceHealth,
    pub determinism_success_rate: f64,
    pub performance_budget_compliance: f64,
    pub active_tests: u64,
    pub total_replays: u64,
    pub last_check: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceHealth {
    Healthy,
    Degraded,
    Unhealthy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub determinism_success_rate: f64,
    pub avg_timestamp_jitter_ms: f64,
    pub p95_performance_budget: f64,
    pub invariant_violations_per_hour: f64,
    pub clock_skew_tolerance_ms: f64,
    pub total_tests_run: u64,
    pub uptime_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    pub success_rates: Vec<DataPoint>,
    pub performance_metrics: Vec<DataPoint>,
    pub invariant_violations: Vec<ViolationDataPoint>,
    pub clock_skew_tests: Vec<ClockSkewDataPoint>,
    pub system_health: SystemHealthData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationDataPoint {
    pub timestamp: DateTime<Utc>,
    pub violation_type: String,
    pub severity: ViolationSeverity,
    pub count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClockSkewDataPoint {
    pub timestamp: DateTime<Utc>,
    pub skew_ms: i64,
    pub tolerance_met: bool,
    pub test_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthData {
    pub overall_status: ServiceHealth,
    pub components: HashMap<String, ComponentHealth>,
    pub alerts: Vec<Alert>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub status: ServiceHealth,
    pub last_check: DateTime<Utc>,
    pub error_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: Uuid,
    pub severity: ViolationSeverity,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub acknowledged: bool,
}

// Error types
#[derive(Debug, thiserror::Error)]
pub enum ValidationError {
    #[error("Determinism validation failed: {0}")]
    DeterminismFailed(String),
    
    #[error("Performance budget exceeded: {actual}ms > {budget}ms")]
    PerformanceBudgetExceeded { actual: f64, budget: f64 },
    
    #[error("Invariant violation: {invariant} - {description}")]
    InvariantViolation { invariant: String, description: String },
    
    #[error("Clock skew tolerance exceeded: {skew}ms > {tolerance}ms")]
    ClockSkewExceeded { skew: u64, tolerance: u64 },
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Database error: {0}")]
    DatabaseError(#[from] sqlx::Error),
}

// V2 Transform Change types for learning loop closure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformChangeV2 {
    pub change_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub change_type: ChangeType,
    pub metadata: ChangeMetadata,
    pub before_state: Option<serde_json::Value>,
    pub after_state: Option<serde_json::Value>,
    pub performance_impact: Option<PerformanceImpact>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    Code,
    Error,
    Fix,
    Tool,
    Normalize,
    Rollback,
    HeadSummary,
    KvUpdate,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeMetadata {
    pub depth: u32,
    pub complexity_score: f64,
    pub edit_distance: Option<u32>,
    pub context_size: u32,
    pub causality_chain: Vec<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImpact {
    pub latency_delta_ms: f64,
    pub memory_delta_mb: f64,
    pub throughput_delta_percent: f64,
}

// V2 Feature extraction results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V2Features {
    pub error_fix_chains: u32,
    pub tool_normalize_chains: u32,
    pub rollback_occurred: bool,
    pub edit_depth: u32,
    pub change_entropy: f64,
    pub code_error_ratio: f64,
    pub late_head_edits: u32,
    pub kv_prefix_impact: f64,
}

// ΔU Training types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaUPrediction {
    pub predicted_utility_change: f64,
    pub confidence: f64,
    pub feature_weights: HashMap<String, f64>,
    pub isotonic_calibrated: bool,
    pub ips_adjusted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDatapoint {
    pub features: V2Features,
    pub ground_truth_utility: f64,
    pub timestamp: DateTime<Utc>,
    pub scenario_type: ScenarioType,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ScenarioType {
    Code,
    Prose,
    ToolResults,
    Mixed,
}

// λ/μ Controller types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaMuController {
    pub current_lambda: f64,
    pub current_mu: f64,
    pub target_k2: u32,
    pub difficulty_score: f64,
    pub hysteresis_state: HysteresisState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HysteresisState {
    pub alpha_shrink: f64,
    pub alpha_grow: f64,
    pub last_adjustment: DateTime<Utc>,
    pub consecutive_adjustments: u32,
}

// Configuration types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeterminismConfig {
    pub replay_interval_seconds: u64,
    pub tolerance_ms: u64,
    pub performance_budget_percent: f64,
    pub max_concurrent_replays: usize,
    pub clock_skew_test_interval_seconds: u64,
}

impl Default for DeterminismConfig {
    fn default() -> Self {
        Self {
            replay_interval_seconds: 3600, // Every hour
            tolerance_ms: 1,               // ≤1ms timestamp jitter
            performance_budget_percent: 2.0, // p95 ≤ 2%
            max_concurrent_replays: 10,
            clock_skew_test_interval_seconds: 900, // Every 15 minutes
        }
    }
}