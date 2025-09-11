use crate::{
    json_canon::CanonicalJson,
    types::*,
};
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, RwLock, Mutex, atomic::{AtomicBool, AtomicU64, Ordering}},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use tokio::time::timeout;

/// V1 Retirement Change Control System
/// Implements controlled migration from v1 to v2 with safety gates and rollback capability
#[derive(Debug)]
pub struct V1RetirementController {
    dual_write_manager: Arc<DualWriteManager>,
    circuit_breaker: Arc<V2InvariantBreaker>,
    kill_switch: Arc<KillSwitch>,
    retention_clock: Arc<RetentionClock>,
    audit_trail: Arc<Mutex<Vec<RetirementEvent>>>,
    metrics: Arc<Mutex<RetirementMetrics>>,
}

/// Dual-write manager for controlled migration
#[derive(Debug)]
pub struct DualWriteManager {
    pub write_mode: Arc<RwLock<WriteMode>>,
    pub reader_mode: Arc<RwLock<ReaderMode>>,
    pub v1_shards: Arc<RwLock<HashMap<String, V1Shard>>>,
    pub v2_shards: Arc<RwLock<HashMap<String, V2Shard>>>,
    pub watch_period: Duration,
    pub switch_timestamp: Arc<RwLock<Option<DateTime<Utc>>>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WriteMode {
    DualWrite,          // Write to both v1 and v2
    V2Only,             // Write only to v2
    EmergencyV1,        // Emergency fallback to v1
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReaderMode {
    V1Primary,          // Read from v1, fallback to v2
    V2Primary,          // Read from v2, fallback to v1
    V2Exclusive,        // Read only from v2
}

/// V1 shard representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V1Shard {
    pub shard_id: String,
    pub last_write_timestamp: DateTime<Utc>,
    pub record_count: u64,
    pub data_size_bytes: u64,
    pub retention_policy: RetentionPolicy,
    pub deletion_eligibility: DeletionEligibility,
    pub backup_status: BackupStatus,
}

/// V2 shard representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V2Shard {
    pub shard_id: String,
    pub migration_timestamp: DateTime<Utc>,
    pub verification_status: VerificationStatus,
    pub performance_metrics: V2PerformanceMetrics,
    pub integrity_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    pub retention_days: u32,
    pub compliance_requirements: Vec<ComplianceRequirement>,
    pub legal_hold: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeletionEligibility {
    Eligible,
    PendingCompliance,
    OnLegalHold,
    RecentActivity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackupStatus {
    Backed_up,
    BackupPending,
    BackupFailed,
    NoBackupRequired,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationStatus {
    Verified,
    Pending,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V2PerformanceMetrics {
    pub average_response_time_ms: f64,
    pub throughput_ops_per_sec: f64,
    pub error_rate: f64,
    pub data_integrity_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRequirement {
    pub standard: String,
    pub minimum_retention_days: u32,
    pub deletion_method: String,
}

/// Circuit breaker specifically for V2 invariant violations
#[derive(Debug)]
pub struct V2InvariantBreaker {
    pub state: Arc<RwLock<BreakerState>>,
    pub violation_threshold: u32,
    pub violation_window: Duration,
    pub violations: Arc<Mutex<VecDeque<InvariantViolation>>>,
    pub rollback_budget: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BreakerState {
    Closed,
    Open { opened_at: Instant },
    ForceRollback,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvariantViolation {
    pub violation_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub violation_type: ViolationType,
    pub severity: ViolationSeverity,
    pub affected_operations: Vec<String>,
    pub metrics_at_violation: ViolationMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    DeterminismBreach,
    PerformanceDegradation,
    DataInconsistency,
    SecurityViolation,
    MemoryLeak,
    TimeoutExceeded,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationMetrics {
    pub response_time_p99: f64,
    pub error_rate: f64,
    pub memory_usage_mb: f64,
    pub cpu_utilization: f64,
    pub queue_depth: u32,
}

/// Kill switch for emergency rollback
#[derive(Debug)]
pub struct KillSwitch {
    pub enabled: AtomicBool,
    pub emit_v1_flag: AtomicBool,
    pub rollback_budget_ms: AtomicU64,
    pub activation_timestamp: Arc<RwLock<Option<DateTime<Utc>>>>,
    pub rollback_plan: Arc<RwLock<RollbackPlan>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackPlan {
    pub steps: Vec<RollbackStep>,
    pub estimated_duration_ms: u64,
    pub validation_checks: Vec<ValidationCheck>,
    pub success_criteria: Vec<SuccessCriterion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackStep {
    pub step_id: String,
    pub description: String,
    pub action: RollbackAction,
    pub timeout_ms: u64,
    pub rollback_on_failure: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RollbackAction {
    SwitchToV1Readers,
    EnableV1Writers,
    DisableV2Processing,
    InvalidateV2Cache,
    RestoreV1Configuration,
    NotifyOperators,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCheck {
    pub check_id: String,
    pub description: String,
    pub success_condition: String,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessCriterion {
    pub criterion_id: String,
    pub metric: String,
    pub threshold: f64,
    pub direction: ThresholdDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThresholdDirection {
    Above,
    Below,
    Equal,
}

/// Retention clock for audit trail management
#[derive(Debug)]
pub struct RetentionClock {
    pub retention_policies: Arc<RwLock<HashMap<String, RetentionPolicy>>>,
    pub deletion_queue: Arc<Mutex<VecDeque<DeletionJob>>>,
    pub audit_log: Arc<Mutex<Vec<RetentionEvent>>>,
    pub clock_interval: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeletionJob {
    pub job_id: Uuid,
    pub scheduled_time: DateTime<Utc>,
    pub shard_id: String,
    pub data_category: String,
    pub compliance_approved: bool,
    pub backup_verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionEvent {
    pub event_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub event_type: RetentionEventType,
    pub shard_id: String,
    pub data_summary: DataSummary,
    pub compliance_status: ComplianceStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetentionEventType {
    DataCreated,
    DataDeleted,
    RetentionPolicyApplied,
    ComplianceReview,
    LegalHoldApplied,
    LegalHoldReleased,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSummary {
    pub record_count: u64,
    pub size_bytes: u64,
    pub data_types: Vec<String>,
    pub sensitivity_level: DataSensitivityLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSensitivityLevel {
    Public,
    Internal,
    Confidential,
    Restricted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceStatus {
    Compliant,
    NonCompliant,
    UnderReview,
    ExemptionGranted,
}

/// Retirement event tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetirementEvent {
    pub event_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub event_type: RetirementEventType,
    pub description: String,
    pub actor: String,
    pub affected_systems: Vec<String>,
    pub rollback_info: Option<RollbackInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetirementEventType {
    DualWriteStarted,
    ReaderSwitched,
    V1ShardDeleted,
    EmergencyRollback,
    KillSwitchActivated,
    ComplianceCheck,
    RetentionPolicyUpdated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackInfo {
    pub rollback_id: Uuid,
    pub trigger_reason: String,
    pub rollback_duration_ms: u64,
    pub success: bool,
    pub systems_affected: Vec<String>,
}

/// Retirement metrics tracking
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct RetirementMetrics {
    pub dual_write_duration_hours: f64,
    pub v1_shards_deleted: u32,
    pub v2_verification_success_rate: f64,
    pub emergency_rollbacks: u32,
    pub compliance_violations: u32,
    pub average_rollback_time_ms: f64,
    pub data_migration_completeness: f64,
}

impl V1RetirementController {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            dual_write_manager: Arc::new(DualWriteManager::new()?),
            circuit_breaker: Arc::new(V2InvariantBreaker::new()),
            kill_switch: Arc::new(KillSwitch::new()?),
            retention_clock: Arc::new(RetentionClock::new()?),
            audit_trail: Arc::new(Mutex::new(Vec::new())),
            metrics: Arc::new(Mutex::new(RetirementMetrics::default())),
        })
    }

    /// Start the dual-write shutdown sequence
    pub async fn start_dual_write_shutdown(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Step 1: Begin 24-hour watch period
        self.dual_write_manager.start_watch_period().await?;
        
        self.log_retirement_event(RetirementEvent {
            event_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type: RetirementEventType::DualWriteStarted,
            description: "24-hour watch period started for dual-write shutdown".to_string(),
            actor: "system".to_string(),
            affected_systems: vec!["dual_write_manager".to_string()],
            rollback_info: None,
        });

        // Step 2: Monitor V2 invariants during watch period
        self.monitor_v2_invariants_during_watch().await?;

        Ok(())
    }

    /// Execute reader switch after watch period
    pub async fn execute_reader_switch(&self) -> Result<(), Box<dyn std::error::Error>> {
        // Verify watch period completion
        if !self.dual_write_manager.is_watch_period_complete().await {
            return Err("Watch period not yet complete".into());
        }

        // Check V2 invariant violations
        if self.circuit_breaker.has_violations().await {
            return Err("V2 invariant violations detected during watch period".into());
        }

        // Switch readers to V2
        self.dual_write_manager.switch_readers_to_v2().await?;

        self.log_retirement_event(RetirementEvent {
            event_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type: RetirementEventType::ReaderSwitched,
            description: "Readers switched to V2 after successful watch period".to_string(),
            actor: "system".to_string(),
            affected_systems: vec!["reader_mode".to_string()],
            rollback_info: None,
        });

        Ok(())
    }

    /// Delete V1 shards with compliance checks
    pub async fn delete_v1_shards(&self) -> Result<DeletionSummary, Box<dyn std::error::Error>> {
        let mut deletion_summary = DeletionSummary {
            total_shards: 0,
            deleted_shards: 0,
            skipped_shards: 0,
            failed_deletions: 0,
            compliance_blocks: 0,
            total_data_deleted_mb: 0.0,
        };

        let v1_shards = self.dual_write_manager.v1_shards.read().unwrap().clone();
        deletion_summary.total_shards = v1_shards.len() as u32;

        for (shard_id, shard) in v1_shards {
            match self.evaluate_shard_deletion(&shard).await {
                DeletionDecision::Proceed => {
                    match self.delete_shard_safely(&shard_id, &shard).await {
                        Ok(deleted_mb) => {
                            deletion_summary.deleted_shards += 1;
                            deletion_summary.total_data_deleted_mb += deleted_mb;
                            
                            self.log_retirement_event(RetirementEvent {
                                event_id: Uuid::new_v4(),
                                timestamp: Utc::now(),
                                event_type: RetirementEventType::V1ShardDeleted,
                                description: format!("V1 shard {} successfully deleted", shard_id),
                                actor: "retention_controller".to_string(),
                                affected_systems: vec![shard_id.clone()],
                                rollback_info: None,
                            });
                        }
                        Err(_) => {
                            deletion_summary.failed_deletions += 1;
                        }
                    }
                }
                DeletionDecision::Skip(reason) => {
                    deletion_summary.skipped_shards += 1;
                    if reason.contains("compliance") {
                        deletion_summary.compliance_blocks += 1;
                    }
                }
            }
        }

        Ok(deletion_summary)
    }

    /// Activate kill switch for emergency rollback
    pub async fn activate_kill_switch(&self, reason: &str) -> Result<RollbackResult, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        
        // Set kill switch flags
        self.kill_switch.enabled.store(true, Ordering::SeqCst);
        self.kill_switch.emit_v1_flag.store(true, Ordering::SeqCst);
        
        *self.kill_switch.activation_timestamp.write().unwrap() = Some(Utc::now());

        // Execute rollback plan
        let rollback_result = self.execute_emergency_rollback().await?;

        let duration = start_time.elapsed();
        
        // Check if rollback was within budget
        let budget_ms = self.kill_switch.rollback_budget_ms.load(Ordering::SeqCst);
        let within_budget = duration.as_millis() as u64 <= budget_ms;

        self.log_retirement_event(RetirementEvent {
            event_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type: RetirementEventType::KillSwitchActivated,
            description: format!("Kill switch activated: {}", reason),
            actor: "emergency_system".to_string(),
            affected_systems: vec!["all_systems".to_string()],
            rollback_info: Some(RollbackInfo {
                rollback_id: Uuid::new_v4(),
                trigger_reason: reason.to_string(),
                rollback_duration_ms: duration.as_millis() as u64,
                success: rollback_result.success && within_budget,
                systems_affected: rollback_result.systems_affected.clone(),
            }),
        });

        Ok(RollbackResult {
            success: rollback_result.success && within_budget,
            duration_ms: duration.as_millis() as u64,
            within_budget,
            systems_affected: rollback_result.systems_affected,
            error_message: rollback_result.error_message,
        })
    }

    /// Get current retirement status
    pub async fn get_retirement_status(&self) -> RetirementStatus {
        let dual_write_mode = self.dual_write_manager.write_mode.read().unwrap().clone();
        let reader_mode = self.dual_write_manager.reader_mode.read().unwrap().clone();
        let kill_switch_active = self.kill_switch.enabled.load(Ordering::SeqCst);
        let circuit_breaker_state = self.circuit_breaker.state.read().unwrap().clone();
        
        let v1_shards = self.dual_write_manager.v1_shards.read().unwrap();
        let v1_shard_count = v1_shards.len() as u32;
        let total_v1_data_mb = v1_shards.values()
            .map(|shard| shard.data_size_bytes as f64 / (1024.0 * 1024.0))
            .sum();

        RetirementStatus {
            dual_write_mode,
            reader_mode,
            kill_switch_active,
            circuit_breaker_state,
            v1_shard_count,
            total_v1_data_mb,
            watch_period_remaining: self.dual_write_manager.get_watch_period_remaining().await,
            compliance_issues: self.get_compliance_issue_count().await,
            metrics: self.metrics.lock().unwrap().clone(),
        }
    }

    // Private implementation methods

    async fn monitor_v2_invariants_during_watch(&self) -> Result<(), Box<dyn std::error::Error>> {
        // This would run continuously during the watch period
        // monitoring V2 system health and invariants
        Ok(())
    }

    async fn evaluate_shard_deletion(&self, shard: &V1Shard) -> DeletionDecision {
        // Check legal hold
        if shard.retention_policy.legal_hold {
            return DeletionDecision::Skip("Legal hold active".to_string());
        }

        // Check backup status
        if !matches!(shard.backup_status, BackupStatus::Backed_up) {
            return DeletionDecision::Skip("Backup not completed".to_string());
        }

        // Check deletion eligibility
        match shard.deletion_eligibility {
            DeletionEligibility::Eligible => DeletionDecision::Proceed,
            DeletionEligibility::PendingCompliance => DeletionDecision::Skip("Compliance review pending".to_string()),
            DeletionEligibility::OnLegalHold => DeletionDecision::Skip("Legal hold".to_string()),
            DeletionEligibility::RecentActivity => DeletionDecision::Skip("Recent activity detected".to_string()),
        }
    }

    async fn delete_shard_safely(&self, shard_id: &str, shard: &V1Shard) -> Result<f64, Box<dyn std::error::Error>> {
        // Perform secure deletion
        let deleted_mb = shard.data_size_bytes as f64 / (1024.0 * 1024.0);
        
        // Remove from tracking
        self.dual_write_manager.v1_shards.write().unwrap().remove(shard_id);
        
        // Update retention clock
        self.retention_clock.log_deletion_event(shard_id, deleted_mb).await?;
        
        Ok(deleted_mb)
    }

    async fn execute_emergency_rollback(&self) -> Result<RollbackResult, Box<dyn std::error::Error>> {
        let rollback_plan = self.kill_switch.rollback_plan.read().unwrap().clone();
        let mut systems_affected = Vec::new();
        
        for step in rollback_plan.steps {
            match self.execute_rollback_step(&step).await {
                Ok(affected_systems) => {
                    systems_affected.extend(affected_systems);
                }
                Err(e) => {
                    return Ok(RollbackResult {
                        success: false,
                        duration_ms: 0,
                        within_budget: false,
                        systems_affected,
                        error_message: Some(e.to_string()),
                    });
                }
            }
        }

        Ok(RollbackResult {
            success: true,
            duration_ms: 0, // Will be set by caller
            within_budget: true, // Will be checked by caller
            systems_affected,
            error_message: None,
        })
    }

    async fn execute_rollback_step(&self, step: &RollbackStep) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        match &step.action {
            RollbackAction::SwitchToV1Readers => {
                *self.dual_write_manager.reader_mode.write().unwrap() = ReaderMode::V1Primary;
                Ok(vec!["reader_mode".to_string()])
            }
            RollbackAction::EnableV1Writers => {
                *self.dual_write_manager.write_mode.write().unwrap() = WriteMode::EmergencyV1;
                Ok(vec!["write_mode".to_string()])
            }
            RollbackAction::DisableV2Processing => {
                // Would disable V2 processing logic
                Ok(vec!["v2_processor".to_string()])
            }
            RollbackAction::InvalidateV2Cache => {
                // Would clear V2 caches
                Ok(vec!["v2_cache".to_string()])
            }
            RollbackAction::RestoreV1Configuration => {
                // Would restore V1 configuration
                Ok(vec!["configuration".to_string()])
            }
            RollbackAction::NotifyOperators => {
                // Would send operator notifications
                Ok(vec!["notification_system".to_string()])
            }
        }
    }

    async fn get_compliance_issue_count(&self) -> u32 {
        // Count active compliance issues
        let v1_shards = self.dual_write_manager.v1_shards.read().unwrap();
        v1_shards.values()
            .filter(|shard| matches!(shard.deletion_eligibility, DeletionEligibility::PendingCompliance))
            .count() as u32
    }

    fn log_retirement_event(&self, event: RetirementEvent) {
        self.audit_trail.lock().unwrap().push(event);
    }
}

impl DualWriteManager {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            write_mode: Arc::new(RwLock::new(WriteMode::DualWrite)),
            reader_mode: Arc::new(RwLock::new(ReaderMode::V1Primary)),
            v1_shards: Arc::new(RwLock::new(HashMap::new())),
            v2_shards: Arc::new(RwLock::new(HashMap::new())),
            watch_period: Duration::from_hours(24),
            switch_timestamp: Arc::new(RwLock::new(None)),
        })
    }

    async fn start_watch_period(&self) -> Result<(), Box<dyn std::error::Error>> {
        *self.switch_timestamp.write().unwrap() = Some(Utc::now());
        Ok(())
    }

    async fn is_watch_period_complete(&self) -> bool {
        if let Some(start_time) = *self.switch_timestamp.read().unwrap() {
            let elapsed = Utc::now() - start_time;
            elapsed > chrono::Duration::from_std(self.watch_period).unwrap()
        } else {
            false
        }
    }

    async fn switch_readers_to_v2(&self) -> Result<(), Box<dyn std::error::Error>> {
        *self.reader_mode.write().unwrap() = ReaderMode::V2Primary;
        Ok(())
    }

    async fn get_watch_period_remaining(&self) -> Option<Duration> {
        if let Some(start_time) = *self.switch_timestamp.read().unwrap() {
            let elapsed = Utc::now() - start_time;
            let elapsed_std = elapsed.to_std().ok()?;
            if elapsed_std < self.watch_period {
                Some(self.watch_period - elapsed_std)
            } else {
                Some(Duration::from_secs(0))
            }
        } else {
            None
        }
    }
}

impl V2InvariantBreaker {
    fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(BreakerState::Closed)),
            violation_threshold: 5,
            violation_window: Duration::from_minutes(10),
            violations: Arc::new(Mutex::new(VecDeque::new())),
            rollback_budget: Duration::from_minutes(10),
        }
    }

    async fn has_violations(&self) -> bool {
        let violations = self.violations.lock().unwrap();
        let now = Utc::now();
        let window_start = now - chrono::Duration::from_std(self.violation_window).unwrap();
        
        violations.iter()
            .filter(|v| v.timestamp >= window_start)
            .count() >= self.violation_threshold as usize
    }

    pub fn record_violation(&self, violation: InvariantViolation) {
        let mut violations = self.violations.lock().unwrap();
        violations.push_back(violation);
        
        // Clean old violations
        let now = Utc::now();
        let window_start = now - chrono::Duration::from_std(self.violation_window).unwrap();
        while let Some(front) = violations.front() {
            if front.timestamp < window_start {
                violations.pop_front();
            } else {
                break;
            }
        }
        
        // Check if we should open circuit breaker
        if violations.len() >= self.violation_threshold as usize {
            *self.state.write().unwrap() = BreakerState::Open { opened_at: Instant::now() };
        }
    }
}

impl KillSwitch {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            enabled: AtomicBool::new(false),
            emit_v1_flag: AtomicBool::new(false),
            rollback_budget_ms: AtomicU64::new(10 * 60 * 1000), // 10 minutes
            activation_timestamp: Arc::new(RwLock::new(None)),
            rollback_plan: Arc::new(RwLock::new(RollbackPlan::default_plan())),
        })
    }
}

impl RollbackPlan {
    fn default_plan() -> Self {
        Self {
            steps: vec![
                RollbackStep {
                    step_id: "switch_readers".to_string(),
                    description: "Switch readers back to V1".to_string(),
                    action: RollbackAction::SwitchToV1Readers,
                    timeout_ms: 30000,
                    rollback_on_failure: false,
                },
                RollbackStep {
                    step_id: "enable_v1_writers".to_string(),
                    description: "Enable V1 writers".to_string(),
                    action: RollbackAction::EnableV1Writers,
                    timeout_ms: 30000,
                    rollback_on_failure: false,
                },
                RollbackStep {
                    step_id: "disable_v2".to_string(),
                    description: "Disable V2 processing".to_string(),
                    action: RollbackAction::DisableV2Processing,
                    timeout_ms: 60000,
                    rollback_on_failure: true,
                },
            ],
            estimated_duration_ms: 120000, // 2 minutes
            validation_checks: vec![],
            success_criteria: vec![],
        }
    }
}

impl RetentionClock {
    fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            retention_policies: Arc::new(RwLock::new(HashMap::new())),
            deletion_queue: Arc::new(Mutex::new(VecDeque::new())),
            audit_log: Arc::new(Mutex::new(Vec::new())),
            clock_interval: Duration::from_hours(1),
        })
    }

    async fn log_deletion_event(&self, shard_id: &str, deleted_mb: f64) -> Result<(), Box<dyn std::error::Error>> {
        let event = RetentionEvent {
            event_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type: RetentionEventType::DataDeleted,
            shard_id: shard_id.to_string(),
            data_summary: DataSummary {
                record_count: 0, // Would be populated from shard metadata
                size_bytes: (deleted_mb * 1024.0 * 1024.0) as u64,
                data_types: vec!["certificates".to_string()],
                sensitivity_level: DataSensitivityLevel::Internal,
            },
            compliance_status: ComplianceStatus::Compliant,
        };

        self.audit_log.lock().unwrap().push(event);
        Ok(())
    }
}

// Supporting types

#[derive(Debug, Clone)]
pub enum DeletionDecision {
    Proceed,
    Skip(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeletionSummary {
    pub total_shards: u32,
    pub deleted_shards: u32,
    pub skipped_shards: u32,
    pub failed_deletions: u32,
    pub compliance_blocks: u32,
    pub total_data_deleted_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackResult {
    pub success: bool,
    pub duration_ms: u64,
    pub within_budget: bool,
    pub systems_affected: Vec<String>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetirementStatus {
    pub dual_write_mode: WriteMode,
    pub reader_mode: ReaderMode,
    pub kill_switch_active: bool,
    pub circuit_breaker_state: BreakerState,
    pub v1_shard_count: u32,
    pub total_v1_data_mb: f64,
    pub watch_period_remaining: Option<Duration>,
    pub compliance_issues: u32,
    pub metrics: RetirementMetrics,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_v1_retirement_controller_creation() {
        let controller = V1RetirementController::new().unwrap();
        let status = controller.get_retirement_status().await;
        
        assert_eq!(status.dual_write_mode, WriteMode::DualWrite);
        assert_eq!(status.reader_mode, ReaderMode::V1Primary);
        assert!(!status.kill_switch_active);
    }

    #[tokio::test]
    async fn test_dual_write_shutdown_sequence() {
        let controller = V1RetirementController::new().unwrap();
        
        // Start dual write shutdown
        controller.start_dual_write_shutdown().await.unwrap();
        
        let status = controller.get_retirement_status().await;
        assert!(status.watch_period_remaining.is_some());
    }

    #[tokio::test]
    async fn test_kill_switch_activation() {
        let controller = V1RetirementController::new().unwrap();
        
        let result = controller.activate_kill_switch("test emergency").await.unwrap();
        
        assert!(result.success);
        assert!(result.duration_ms < 10 * 60 * 1000); // Within 10 minute budget
        assert!(!result.systems_affected.is_empty());
        
        let status = controller.get_retirement_status().await;
        assert!(status.kill_switch_active);
    }

    #[test]
    fn test_v2_invariant_breaker() {
        let breaker = V2InvariantBreaker::new();
        
        // Record violations
        for i in 0..6 {
            let violation = InvariantViolation {
                violation_id: Uuid::new_v4(),
                timestamp: Utc::now(),
                violation_type: ViolationType::DeterminismBreach,
                severity: ViolationSeverity::High,
                affected_operations: vec![format!("operation_{}", i)],
                metrics_at_violation: ViolationMetrics {
                    response_time_p99: 1000.0,
                    error_rate: 0.05,
                    memory_usage_mb: 500.0,
                    cpu_utilization: 0.8,
                    queue_depth: 10,
                },
            };
            
            breaker.record_violation(violation);
        }
        
        // Circuit breaker should be open after threshold violations
        let state = breaker.state.read().unwrap();
        assert!(matches!(*state, BreakerState::Open { .. }));
    }

    #[tokio::test]
    async fn test_shard_deletion_evaluation() {
        let controller = V1RetirementController::new().unwrap();
        
        // Test shard that's eligible for deletion
        let eligible_shard = V1Shard {
            shard_id: "shard_1".to_string(),
            last_write_timestamp: Utc::now() - chrono::Duration::days(30),
            record_count: 1000,
            data_size_bytes: 1024 * 1024, // 1MB
            retention_policy: RetentionPolicy {
                retention_days: 7,
                compliance_requirements: vec![],
                legal_hold: false,
            },
            deletion_eligibility: DeletionEligibility::Eligible,
            backup_status: BackupStatus::Backed_up,
        };
        
        let decision = controller.evaluate_shard_deletion(&eligible_shard).await;
        assert!(matches!(decision, DeletionDecision::Proceed));
        
        // Test shard on legal hold
        let legal_hold_shard = V1Shard {
            deletion_eligibility: DeletionEligibility::OnLegalHold,
            ..eligible_shard
        };
        
        let decision = controller.evaluate_shard_deletion(&legal_hold_shard).await;
        assert!(matches!(decision, DeletionDecision::Skip(_)));
    }
}