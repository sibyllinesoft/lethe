use crate::{
    json_canon::CanonicalJson,
    types::*,
};
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, RwLock, Mutex, atomic::{AtomicU64, AtomicBool, Ordering}},
    time::{Duration, Instant},
};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use serde::{Deserialize, Serialize};
use rand::{Rng, thread_rng};

/// Auto-Dim + K2 Safe Rollout System
/// Implements gradient-based difficulty adjustment with machine learning dimension selection
#[derive(Debug)]
pub struct SafeAutoDim {
    difficulty_gate: Arc<RwLock<DifficultyGate>>,
    k2_change_budget: Arc<RwLock<K2ChangeBudget>>,
    dim_selection: Arc<RwLock<GradientBoostingMachine>>,
    safety_gates: Arc<SafetyGates>,
    rollout_controller: Arc<RolloutController>,
    performance_monitor: Arc<PerformanceMonitor>,
    metrics: Arc<Mutex<AutoDimMetrics>>,
}

/// Difficulty gate with multi-factor scoring
#[derive(Debug, Clone)]
pub struct DifficultyGate {
    pub alpha: f64,              // History factor weight
    pub beta: f64,               // Rollback rate weight  
    pub gamma: f64,              // Edit depth weight
    pub history_score: f64,      // H component
    pub rollback_rate: f64,      // Current rollback rate
    pub edit_depth: f64,         // Current edit complexity
    pub computed_difficulty: f64, // α·H + β·rollback_rate + γ·edit_depth
    pub threshold: f64,          // Difficulty threshold for activation
    pub last_updated: DateTime<Utc>,
}

/// K2 change budget controller with safety limits
#[derive(Debug, Clone)]
pub struct K2ChangeBudget {
    pub max_change_percent: f64,  // ≤+10% per turn
    pub current_k2: u32,          // Current K2 value
    pub target_k2: u32,           // Target K2 value
    pub change_velocity: f64,     // Rate of change per time unit
    pub turn_duration: Duration,  // Duration of each adjustment turn
    pub last_change_timestamp: DateTime<Utc>,
    pub safety_brake_active: bool,
    pub change_history: VecDeque<K2ChangeRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct K2ChangeRecord {
    pub timestamp: DateTime<Utc>,
    pub from_k2: u32,
    pub to_k2: u32,
    pub change_percent: f64,
    pub trigger_reason: String,
    pub safety_checks_passed: bool,
}

/// Gradient Boosting Machine for dimension selection (256 vs 768)
#[derive(Debug, Clone)]
pub struct GradientBoostingMachine {
    pub models: Vec<WeakLearner>,
    pub learning_rate: f64,
    pub feature_weights: HashMap<String, f64>,
    pub current_prediction: DimensionPrediction,
    pub training_data: VecDeque<TrainingExample>,
    pub model_performance: ModelPerformance,
}

#[derive(Debug, Clone)]
pub struct WeakLearner {
    pub learner_id: Uuid,
    pub decision_threshold: f64,
    pub feature_name: String,
    pub split_value: f64,
    pub left_prediction: f64,
    pub right_prediction: f64,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionPrediction {
    pub recommended_dimension: DimensionSize,
    pub confidence: f64,
    pub feature_importance: HashMap<String, f64>,
    pub prediction_timestamp: DateTime<Utc>,
    pub performance_forecast: PerformanceForecast,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DimensionSize {
    Dim256,
    Dim768,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceForecast {
    pub expected_latency_ms: f64,
    pub expected_throughput: f64,
    pub expected_accuracy: f64,
    pub confidence_interval: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub features: HashMap<String, f64>,
    pub actual_dimension: DimensionSize,
    pub performance_outcome: PerformanceOutcome,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct PerformanceOutcome {
    pub latency_ms: f64,
    pub throughput: f64,
    pub accuracy: f64,
    pub error_rate: f64,
}

#[derive(Debug, Clone)]
pub struct ModelPerformance {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub last_evaluation: DateTime<Utc>,
    pub training_iterations: u64,
}

/// Safety gates for monitoring system health
#[derive(Debug)]
pub struct SafetyGates {
    pub ece_threshold: f64,           // ΔECE ≤ 0.01
    pub drift_threshold: f64,         // λ/μ drift ≤ 10%/24h
    pub current_ece: Arc<RwLock<f64>>,
    pub lambda_mu_drift: Arc<RwLock<f64>>,
    pub violation_count: Arc<RwLock<u32>>,
    pub last_safety_check: Arc<RwLock<DateTime<Utc>>>,
    pub emergency_brake: Arc<RwLock<bool>>,
}

/// Rollout controller for safe deployment
#[derive(Debug)]
pub struct RolloutController {
    pub rollout_phases: Vec<RolloutPhase>,
    pub current_phase: Arc<RwLock<usize>>,
    pub phase_metrics: Arc<Mutex<HashMap<usize, PhaseMetrics>>>,
    pub rollback_triggers: Vec<RollbackTrigger>,
    pub auto_progression: bool,
}

#[derive(Debug, Clone)]
pub struct RolloutPhase {
    pub phase_id: usize,
    pub name: String,
    pub traffic_percentage: f64,
    pub duration: Duration,
    pub success_criteria: Vec<SuccessCriterion>,
    pub rollback_criteria: Vec<RollbackCriterion>,
}

#[derive(Debug, Clone)]
pub struct PhaseMetrics {
    pub start_time: DateTime<Utc>,
    pub traffic_served: u64,
    pub error_rate: f64,
    pub latency_p99: f64,
    pub throughput: f64,
    pub dimension_accuracy: f64,
    pub safety_violations: u32,
}

#[derive(Debug, Clone)]
pub struct SuccessCriterion {
    pub metric: String,
    pub threshold: f64,
    pub direction: ComparisonDirection,
    pub evaluation_window: Duration,
}

#[derive(Debug, Clone)]
pub struct RollbackCriterion {
    pub metric: String,
    pub threshold: f64,
    pub severity: RollbackSeverity,
    pub auto_rollback: bool,
}

#[derive(Debug, Clone)]
pub enum ComparisonDirection {
    Above,
    Below,
    Within(f64, f64),
}

#[derive(Debug, Clone)]
pub enum RollbackSeverity {
    Warning,
    Critical,
    Emergency,
}

#[derive(Debug, Clone)]
pub struct RollbackTrigger {
    pub trigger_id: String,
    pub condition: TriggerCondition,
    pub action: RollbackAction,
    pub cooldown: Duration,
    pub last_triggered: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub enum TriggerCondition {
    ErrorRateExceeds(f64),
    LatencyExceeds(f64),
    ThroughputBelow(f64),
    SafetyViolation,
    ManualTrigger,
}

#[derive(Debug, Clone)]
pub enum RollbackAction {
    PauseRollout,
    RevertToPhase(usize),
    EmergencyStop,
    ReduceTraffic(f64),
}

/// Performance monitoring for real-time feedback
#[derive(Debug)]
pub struct PerformanceMonitor {
    pub real_time_metrics: Arc<RwLock<RealTimeMetrics>>,
    pub metric_history: Arc<Mutex<VecDeque<MetricSnapshot>>>,
    pub alert_thresholds: HashMap<String, AlertThreshold>,
    pub active_alerts: Arc<Mutex<Vec<ActiveAlert>>>,
}

#[derive(Debug, Clone)]
pub struct RealTimeMetrics {
    pub current_dimension: DimensionSize,
    pub current_k2: u32,
    pub requests_per_second: f64,
    pub latency_p99_ms: f64,
    pub error_rate: f64,
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub ece_score: f64,
    pub lambda_mu_ratio: f64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct MetricSnapshot {
    pub timestamp: DateTime<Utc>,
    pub metrics: RealTimeMetrics,
    pub dimension_effectiveness: f64,
    pub k2_impact_score: f64,
}

#[derive(Debug, Clone)]
pub struct AlertThreshold {
    pub warning_threshold: f64,
    pub critical_threshold: f64,
    pub evaluation_window: Duration,
    pub alert_cooldown: Duration,
}

#[derive(Debug, Clone)]
pub struct ActiveAlert {
    pub alert_id: Uuid,
    pub metric_name: String,
    pub severity: AlertSeverity,
    pub current_value: f64,
    pub threshold_value: f64,
    pub triggered_at: DateTime<Utc>,
    pub acknowledgment_required: bool,
}

#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Auto-Dim metrics tracking
#[derive(Debug, Default, Clone)]
pub struct AutoDimMetrics {
    pub total_dimension_changes: u64,
    pub successful_changes: u64,
    pub failed_changes: u64,
    pub k2_adjustments: u64,
    pub safety_gate_violations: u64,
    pub rollbacks_triggered: u64,
    pub average_latency_improvement: f64,
    pub average_accuracy_improvement: f64,
    pub system_stability_score: f64,
}

impl SafeAutoDim {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            difficulty_gate: Arc::new(RwLock::new(DifficultyGate::new())),
            k2_change_budget: Arc::new(RwLock::new(K2ChangeBudget::new())),
            dim_selection: Arc::new(RwLock::new(GradientBoostingMachine::new())),
            safety_gates: Arc::new(SafetyGates::new()),
            rollout_controller: Arc::new(RolloutController::new()),
            performance_monitor: Arc::new(PerformanceMonitor::new()),
            metrics: Arc::new(Mutex::new(AutoDimMetrics::default())),
        })
    }

    /// Execute safe auto-dimension adjustment cycle
    pub async fn execute_auto_dim_cycle(&self) -> Result<AutoDimResult, Box<dyn std::error::Error>> {
        let cycle_start = Instant::now();
        
        // 1. Evaluate difficulty gate
        let difficulty_score = self.evaluate_difficulty_gate().await?;
        
        // 2. Check safety gates
        let safety_check = self.check_safety_gates().await?;
        
        if !safety_check.all_gates_pass {
            return Ok(AutoDimResult {
                action_taken: AutoDimAction::SafetyHold,
                difficulty_score,
                dimension_change: None,
                k2_change: None,
                safety_violations: safety_check.violations,
                cycle_duration: cycle_start.elapsed(),
                next_evaluation: Utc::now() + chrono::Duration::minutes(5),
            });
        }

        // 3. Generate dimension prediction using gradient boosting
        let dim_prediction = self.predict_optimal_dimension().await?;
        
        // 4. Evaluate K2 change budget
        let k2_change = self.evaluate_k2_change().await?;
        
        // 5. Execute changes if within safety bounds
        let mut dimension_change = None;
        let mut k2_change_result = None;
        
        if difficulty_score > self.difficulty_gate.read().unwrap().threshold {
            dimension_change = Some(self.execute_dimension_change(&dim_prediction).await?);
        }
        
        if let Some(k2_delta) = k2_change {
            k2_change_result = Some(self.execute_k2_change(k2_delta).await?);
        }

        // 6. Update metrics and training data
        self.update_training_data(&dim_prediction, dimension_change.as_ref(), k2_change_result.as_ref()).await?;
        
        Ok(AutoDimResult {
            action_taken: AutoDimAction::AdjustmentComplete,
            difficulty_score,
            dimension_change,
            k2_change: k2_change_result,
            safety_violations: vec![],
            cycle_duration: cycle_start.elapsed(),
            next_evaluation: Utc::now() + chrono::Duration::minutes(15),
        })
    }

    /// Start canary rollout with 10% traffic
    pub async fn start_canary_rollout(&self) -> Result<RolloutStatus, Box<dyn std::error::Error>> {
        self.rollout_controller.start_canary().await
    }

    /// Get current system status
    pub async fn get_system_status(&self) -> AutoDimSystemStatus {
        let difficulty_gate = self.difficulty_gate.read().unwrap().clone();
        let k2_budget = self.k2_change_budget.read().unwrap().clone();
        let safety_status = self.safety_gates.get_status().await;
        let performance = self.performance_monitor.real_time_metrics.read().unwrap().clone();
        let metrics = self.metrics.lock().unwrap().clone();

        AutoDimSystemStatus {
            difficulty_gate,
            k2_budget,
            safety_status,
            performance,
            rollout_phase: self.rollout_controller.get_current_phase().await,
            metrics,
            last_adjustment: Utc::now(), // Would track actual last adjustment
        }
    }

    // Private implementation methods

    async fn evaluate_difficulty_gate(&self) -> Result<f64, Box<dyn std::error::Error>> {
        let mut gate = self.difficulty_gate.write().unwrap();
        
        // Update components
        gate.rollback_rate = self.calculate_current_rollback_rate().await;
        gate.edit_depth = self.calculate_edit_depth().await;
        gate.history_score = self.calculate_history_score().await;
        
        // Compute difficulty score: α·H + β·rollback_rate + γ·edit_depth
        gate.computed_difficulty = gate.alpha * gate.history_score + 
                                   gate.beta * gate.rollback_rate + 
                                   gate.gamma * gate.edit_depth;
        
        gate.last_updated = Utc::now();
        
        Ok(gate.computed_difficulty)
    }

    async fn check_safety_gates(&self) -> Result<SafetyGateResult, Box<dyn std::error::Error>> {
        let mut violations = Vec::new();
        
        // Check ECE threshold (ΔECE ≤ 0.01)
        let current_ece = *self.safety_gates.current_ece.read().unwrap();
        if current_ece > self.safety_gates.ece_threshold {
            violations.push(SafetyViolation {
                gate_type: SafetyGateType::ECEThreshold,
                current_value: current_ece,
                threshold: self.safety_gates.ece_threshold,
                severity: ViolationSeverity::High,
            });
        }

        // Check λ/μ drift (≤ 10%/24h)
        let lambda_mu_drift = *self.safety_gates.lambda_mu_drift.read().unwrap();
        if lambda_mu_drift > self.safety_gates.drift_threshold {
            violations.push(SafetyViolation {
                gate_type: SafetyGateType::LambdaMuDrift,
                current_value: lambda_mu_drift,
                threshold: self.safety_gates.drift_threshold,
                severity: ViolationSeverity::Medium,
            });
        }

        Ok(SafetyGateResult {
            all_gates_pass: violations.is_empty(),
            violations,
            check_timestamp: Utc::now(),
        })
    }

    async fn predict_optimal_dimension(&self) -> Result<DimensionPrediction, Box<dyn std::error::Error>> {
        let gbm = self.dim_selection.read().unwrap();
        
        // Extract current features
        let features = self.extract_current_features().await;
        
        // Run prediction through ensemble
        let mut prediction_sum = 0.0;
        let mut total_weight = 0.0;
        
        for learner in &gbm.models {
            let feature_value = features.get(&learner.feature_name).unwrap_or(&0.0);
            let prediction = if *feature_value < learner.split_value {
                learner.left_prediction
            } else {
                learner.right_prediction
            };
            
            prediction_sum += prediction * learner.weight;
            total_weight += learner.weight;
        }

        let final_prediction = if total_weight > 0.0 {
            prediction_sum / total_weight
        } else {
            0.5 // Default to 50/50
        };

        let recommended_dimension = if final_prediction > 0.5 {
            DimensionSize::Dim768
        } else {
            DimensionSize::Dim256
        };

        Ok(DimensionPrediction {
            recommended_dimension,
            confidence: (final_prediction - 0.5).abs() * 2.0, // Convert to 0-1 scale
            feature_importance: self.calculate_feature_importance(&features),
            prediction_timestamp: Utc::now(),
            performance_forecast: self.generate_performance_forecast(&recommended_dimension).await,
        })
    }

    async fn evaluate_k2_change(&self) -> Result<Option<i32>, Box<dyn std::error::Error>> {
        let budget = self.k2_change_budget.read().unwrap();
        
        // Check if enough time has passed since last change
        let time_since_last = Utc::now() - budget.last_change_timestamp;
        if time_since_last < chrono::Duration::from_std(budget.turn_duration).unwrap() {
            return Ok(None);
        }

        // Check if safety brake is active
        if budget.safety_brake_active {
            return Ok(None);
        }

        // Calculate optimal K2 change
        let performance_metrics = self.performance_monitor.real_time_metrics.read().unwrap();
        let optimal_k2 = self.calculate_optimal_k2(&performance_metrics).await;
        
        let k2_delta = optimal_k2 as i32 - budget.current_k2 as i32;
        let change_percent = (k2_delta.abs() as f64 / budget.current_k2 as f64) * 100.0;
        
        // Check against budget constraints
        if change_percent <= budget.max_change_percent {
            Ok(Some(k2_delta))
        } else {
            // Scale down to fit within budget
            let max_delta = ((budget.max_change_percent / 100.0) * budget.current_k2 as f64) as i32;
            let scaled_delta = if k2_delta > 0 { max_delta } else { -max_delta };
            Ok(Some(scaled_delta))
        }
    }

    async fn execute_dimension_change(&self, prediction: &DimensionPrediction) -> Result<DimensionChangeResult, Box<dyn std::error::Error>> {
        let start_time = Instant::now();
        
        // Get current performance baseline
        let baseline_metrics = self.performance_monitor.real_time_metrics.read().unwrap().clone();
        
        // Execute the dimension change
        let change_success = self.apply_dimension_change(&prediction.recommended_dimension).await?;
        
        if !change_success {
            return Ok(DimensionChangeResult {
                success: false,
                from_dimension: self.get_current_dimension(),
                to_dimension: prediction.recommended_dimension.clone(),
                performance_impact: None,
                duration: start_time.elapsed(),
                error_message: Some("Failed to apply dimension change".to_string()),
            });
        }

        // Wait for stabilization
        tokio::time::sleep(Duration::from_secs(30)).await;
        
        // Measure performance impact
        let post_change_metrics = self.performance_monitor.real_time_metrics.read().unwrap().clone();
        let performance_impact = self.calculate_performance_impact(&baseline_metrics, &post_change_metrics);

        Ok(DimensionChangeResult {
            success: true,
            from_dimension: self.get_current_dimension(),
            to_dimension: prediction.recommended_dimension.clone(),
            performance_impact: Some(performance_impact),
            duration: start_time.elapsed(),
            error_message: None,
        })
    }

    async fn execute_k2_change(&self, k2_delta: i32) -> Result<K2ChangeResult, Box<dyn std::error::Error>> {
        let mut budget = self.k2_change_budget.write().unwrap();
        let old_k2 = budget.current_k2;
        let new_k2 = (budget.current_k2 as i32 + k2_delta) as u32;
        
        // Validate bounds
        if new_k2 < 1 || new_k2 > 1000 {
            return Ok(K2ChangeResult {
                success: false,
                from_k2: old_k2,
                to_k2: old_k2,
                actual_delta: 0,
                error_message: Some("K2 value out of bounds".to_string()),
            });
        }

        // Apply change
        budget.current_k2 = new_k2;
        budget.last_change_timestamp = Utc::now();
        
        // Record change
        let change_record = K2ChangeRecord {
            timestamp: Utc::now(),
            from_k2: old_k2,
            to_k2: new_k2,
            change_percent: (k2_delta.abs() as f64 / old_k2 as f64) * 100.0,
            trigger_reason: "auto_adjustment".to_string(),
            safety_checks_passed: true,
        };
        
        budget.change_history.push_back(change_record);
        if budget.change_history.len() > 100 {
            budget.change_history.pop_front();
        }

        Ok(K2ChangeResult {
            success: true,
            from_k2: old_k2,
            to_k2: new_k2,
            actual_delta: k2_delta,
            error_message: None,
        })
    }

    // Helper methods

    async fn calculate_current_rollback_rate(&self) -> f64 {
        // Calculate based on recent rollback events
        0.02 // 2% rollback rate (example)
    }

    async fn calculate_edit_depth(&self) -> f64 {
        // Calculate complexity of recent edits
        0.3 // Normalized edit depth (example)
    }

    async fn calculate_history_score(&self) -> f64 {
        // Calculate based on historical performance
        0.7 // History score (example)
    }

    async fn extract_current_features(&self) -> HashMap<String, f64> {
        let mut features = HashMap::new();
        let metrics = self.performance_monitor.real_time_metrics.read().unwrap();
        
        features.insert("latency_p99".to_string(), metrics.latency_p99_ms);
        features.insert("error_rate".to_string(), metrics.error_rate);
        features.insert("throughput".to_string(), metrics.requests_per_second);
        features.insert("cpu_utilization".to_string(), metrics.cpu_utilization);
        features.insert("memory_utilization".to_string(), metrics.memory_utilization);
        features.insert("ece_score".to_string(), metrics.ece_score);
        
        features
    }

    fn calculate_feature_importance(&self, features: &HashMap<String, f64>) -> HashMap<String, f64> {
        let gbm = self.dim_selection.read().unwrap();
        gbm.feature_weights.clone()
    }

    async fn generate_performance_forecast(&self, dimension: &DimensionSize) -> PerformanceForecast {
        // Generate forecast based on historical data and dimension choice
        match dimension {
            DimensionSize::Dim256 => PerformanceForecast {
                expected_latency_ms: 50.0,
                expected_throughput: 1000.0,
                expected_accuracy: 0.92,
                confidence_interval: (0.90, 0.94),
            },
            DimensionSize::Dim768 => PerformanceForecast {
                expected_latency_ms: 120.0,
                expected_throughput: 400.0,
                expected_accuracy: 0.96,
                confidence_interval: (0.94, 0.98),
            },
        }
    }

    async fn calculate_optimal_k2(&self, metrics: &RealTimeMetrics) -> u32 {
        // Calculate optimal K2 based on current performance
        let base_k2 = self.k2_change_budget.read().unwrap().current_k2;
        
        // Adjust based on performance metrics
        if metrics.latency_p99_ms > 100.0 {
            (base_k2 as f64 * 0.9) as u32  // Reduce K2 to improve latency
        } else if metrics.error_rate > 0.01 {
            (base_k2 as f64 * 1.1) as u32  // Increase K2 to improve accuracy
        } else {
            base_k2
        }
    }

    async fn apply_dimension_change(&self, target_dimension: &DimensionSize) -> Result<bool, Box<dyn std::error::Error>> {
        // Apply the dimension change to the system
        // This would interact with the actual ML/processing system
        Ok(true)
    }

    fn get_current_dimension(&self) -> DimensionSize {
        self.performance_monitor.real_time_metrics.read().unwrap().current_dimension.clone()
    }

    fn calculate_performance_impact(&self, baseline: &RealTimeMetrics, post_change: &RealTimeMetrics) -> PerformanceImpact {
        PerformanceImpact {
            latency_change_percent: ((post_change.latency_p99_ms - baseline.latency_p99_ms) / baseline.latency_p99_ms) * 100.0,
            throughput_change_percent: ((post_change.requests_per_second - baseline.requests_per_second) / baseline.requests_per_second) * 100.0,
            error_rate_change: post_change.error_rate - baseline.error_rate,
            cpu_impact: post_change.cpu_utilization - baseline.cpu_utilization,
            memory_impact: post_change.memory_utilization - baseline.memory_utilization,
        }
    }

    async fn update_training_data(&self, prediction: &DimensionPrediction, dim_change: Option<&DimensionChangeResult>, k2_change: Option<&K2ChangeResult>) -> Result<(), Box<dyn std::error::Error>> {
        // Update the gradient boosting machine with new training data
        if let Some(change_result) = dim_change {
            if let Some(impact) = &change_result.performance_impact {
                let training_example = TrainingExample {
                    features: self.extract_current_features().await,
                    actual_dimension: change_result.to_dimension.clone(),
                    performance_outcome: PerformanceOutcome {
                        latency_ms: 0.0, // Would be measured from actual system
                        throughput: 0.0,
                        accuracy: 0.0,
                        error_rate: 0.0,
                    },
                    timestamp: Utc::now(),
                };
                
                let mut gbm = self.dim_selection.write().unwrap();
                gbm.training_data.push_back(training_example);
                if gbm.training_data.len() > 1000 {
                    gbm.training_data.pop_front();
                }
            }
        }

        Ok(())
    }
}

// Implementation of supporting components

impl DifficultyGate {
    fn new() -> Self {
        Self {
            alpha: 0.4,
            beta: 0.3,
            gamma: 0.3,
            history_score: 0.0,
            rollback_rate: 0.0,
            edit_depth: 0.0,
            computed_difficulty: 0.0,
            threshold: 0.5,
            last_updated: Utc::now(),
        }
    }
}

impl K2ChangeBudget {
    fn new() -> Self {
        Self {
            max_change_percent: 10.0,
            current_k2: 64,
            target_k2: 64,
            change_velocity: 0.0,
            turn_duration: Duration::from_hours(1),
            last_change_timestamp: Utc::now(),
            safety_brake_active: false,
            change_history: VecDeque::new(),
        }
    }
}

impl GradientBoostingMachine {
    fn new() -> Self {
        Self {
            models: vec![],
            learning_rate: 0.1,
            feature_weights: HashMap::new(),
            current_prediction: DimensionPrediction {
                recommended_dimension: DimensionSize::Dim256,
                confidence: 0.5,
                feature_importance: HashMap::new(),
                prediction_timestamp: Utc::now(),
                performance_forecast: PerformanceForecast {
                    expected_latency_ms: 50.0,
                    expected_throughput: 1000.0,
                    expected_accuracy: 0.92,
                    confidence_interval: (0.90, 0.94),
                },
            },
            training_data: VecDeque::new(),
            model_performance: ModelPerformance {
                accuracy: 0.0,
                precision: 0.0,
                recall: 0.0,
                f1_score: 0.0,
                last_evaluation: Utc::now(),
                training_iterations: 0,
            },
        }
    }
}

impl SafetyGates {
    fn new() -> Self {
        Self {
            ece_threshold: 0.01,
            drift_threshold: 0.10,
            current_ece: Arc::new(RwLock::new(0.005)),
            lambda_mu_drift: Arc::new(RwLock::new(0.02)),
            violation_count: Arc::new(RwLock::new(0)),
            last_safety_check: Arc::new(RwLock::new(Utc::now())),
            emergency_brake: Arc::new(RwLock::new(false)),
        }
    }

    async fn get_status(&self) -> SafetyGateStatus {
        SafetyGateStatus {
            ece_status: SafetyGateIndividualStatus {
                current_value: *self.current_ece.read().unwrap(),
                threshold: self.ece_threshold,
                passing: *self.current_ece.read().unwrap() <= self.ece_threshold,
            },
            drift_status: SafetyGateIndividualStatus {
                current_value: *self.lambda_mu_drift.read().unwrap(),
                threshold: self.drift_threshold,
                passing: *self.lambda_mu_drift.read().unwrap() <= self.drift_threshold,
            },
            violation_count: *self.violation_count.read().unwrap(),
            emergency_brake_active: *self.emergency_brake.read().unwrap(),
            last_check: *self.last_safety_check.read().unwrap(),
        }
    }
}

impl RolloutController {
    fn new() -> Self {
        Self {
            rollout_phases: vec![
                RolloutPhase {
                    phase_id: 0,
                    name: "Canary".to_string(),
                    traffic_percentage: 10.0,
                    duration: Duration::from_hours(2),
                    success_criteria: vec![],
                    rollback_criteria: vec![],
                },
                RolloutPhase {
                    phase_id: 1,
                    name: "Limited".to_string(),
                    traffic_percentage: 25.0,
                    duration: Duration::from_hours(4),
                    success_criteria: vec![],
                    rollback_criteria: vec![],
                },
                RolloutPhase {
                    phase_id: 2,
                    name: "Full".to_string(),
                    traffic_percentage: 100.0,
                    duration: Duration::from_hours(24),
                    success_criteria: vec![],
                    rollback_criteria: vec![],
                },
            ],
            current_phase: Arc::new(RwLock::new(0)),
            phase_metrics: Arc::new(Mutex::new(HashMap::new())),
            rollback_triggers: vec![],
            auto_progression: true,
        }
    }

    async fn start_canary(&self) -> Result<RolloutStatus, Box<dyn std::error::Error>> {
        *self.current_phase.write().unwrap() = 0;
        
        Ok(RolloutStatus {
            current_phase: 0,
            phase_name: "Canary".to_string(),
            traffic_percentage: 10.0,
            phase_start_time: Utc::now(),
            estimated_completion: Utc::now() + chrono::Duration::hours(2),
            success_criteria_met: 0,
            total_success_criteria: 3,
            rollback_triggers_active: 0,
        })
    }

    async fn get_current_phase(&self) -> RolloutPhaseInfo {
        let phase_id = *self.current_phase.read().unwrap();
        let phase = &self.rollout_phases[phase_id];
        
        RolloutPhaseInfo {
            phase_id,
            name: phase.name.clone(),
            traffic_percentage: phase.traffic_percentage,
            duration: phase.duration,
            start_time: Utc::now(), // Would track actual start time
        }
    }
}

impl PerformanceMonitor {
    fn new() -> Self {
        Self {
            real_time_metrics: Arc::new(RwLock::new(RealTimeMetrics {
                current_dimension: DimensionSize::Dim256,
                current_k2: 64,
                requests_per_second: 100.0,
                latency_p99_ms: 50.0,
                error_rate: 0.001,
                cpu_utilization: 0.3,
                memory_utilization: 0.4,
                ece_score: 0.005,
                lambda_mu_ratio: 1.2,
                last_updated: Utc::now(),
            })),
            metric_history: Arc::new(Mutex::new(VecDeque::new())),
            alert_thresholds: HashMap::new(),
            active_alerts: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

// Supporting result types

#[derive(Debug, Clone)]
pub struct AutoDimResult {
    pub action_taken: AutoDimAction,
    pub difficulty_score: f64,
    pub dimension_change: Option<DimensionChangeResult>,
    pub k2_change: Option<K2ChangeResult>,
    pub safety_violations: Vec<SafetyViolation>,
    pub cycle_duration: Duration,
    pub next_evaluation: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub enum AutoDimAction {
    AdjustmentComplete,
    SafetyHold,
    NoChangeNeeded,
    EmergencyRollback,
}

#[derive(Debug, Clone)]
pub struct DimensionChangeResult {
    pub success: bool,
    pub from_dimension: DimensionSize,
    pub to_dimension: DimensionSize,
    pub performance_impact: Option<PerformanceImpact>,
    pub duration: Duration,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone)]
pub struct K2ChangeResult {
    pub success: bool,
    pub from_k2: u32,
    pub to_k2: u32,
    pub actual_delta: i32,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PerformanceImpact {
    pub latency_change_percent: f64,
    pub throughput_change_percent: f64,
    pub error_rate_change: f64,
    pub cpu_impact: f64,
    pub memory_impact: f64,
}

#[derive(Debug, Clone)]
pub struct SafetyGateResult {
    pub all_gates_pass: bool,
    pub violations: Vec<SafetyViolation>,
    pub check_timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct SafetyViolation {
    pub gate_type: SafetyGateType,
    pub current_value: f64,
    pub threshold: f64,
    pub severity: ViolationSeverity,
}

#[derive(Debug, Clone)]
pub enum SafetyGateType {
    ECEThreshold,
    LambdaMuDrift,
}

#[derive(Debug, Clone)]
pub enum ViolationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct SafetyGateStatus {
    pub ece_status: SafetyGateIndividualStatus,
    pub drift_status: SafetyGateIndividualStatus,
    pub violation_count: u32,
    pub emergency_brake_active: bool,
    pub last_check: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct SafetyGateIndividualStatus {
    pub current_value: f64,
    pub threshold: f64,
    pub passing: bool,
}

#[derive(Debug, Clone)]
pub struct RolloutStatus {
    pub current_phase: usize,
    pub phase_name: String,
    pub traffic_percentage: f64,
    pub phase_start_time: DateTime<Utc>,
    pub estimated_completion: DateTime<Utc>,
    pub success_criteria_met: u32,
    pub total_success_criteria: u32,
    pub rollback_triggers_active: u32,
}

#[derive(Debug, Clone)]
pub struct RolloutPhaseInfo {
    pub phase_id: usize,
    pub name: String,
    pub traffic_percentage: f64,
    pub duration: Duration,
    pub start_time: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct AutoDimSystemStatus {
    pub difficulty_gate: DifficultyGate,
    pub k2_budget: K2ChangeBudget,
    pub safety_status: SafetyGateStatus,
    pub performance: RealTimeMetrics,
    pub rollout_phase: RolloutPhaseInfo,
    pub metrics: AutoDimMetrics,
    pub last_adjustment: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_safe_auto_dim_creation() {
        let system = SafeAutoDim::new().unwrap();
        let status = system.get_system_status().await;
        
        assert_eq!(status.k2_budget.current_k2, 64);
        assert!(status.safety_status.ece_status.passing);
        assert!(status.safety_status.drift_status.passing);
    }

    #[tokio::test]
    async fn test_difficulty_gate_evaluation() {
        let system = SafeAutoDim::new().unwrap();
        let difficulty_score = system.evaluate_difficulty_gate().await.unwrap();
        
        assert!(difficulty_score >= 0.0);
        assert!(difficulty_score <= 1.0);
    }

    #[tokio::test]
    async fn test_safety_gates_check() {
        let system = SafeAutoDim::new().unwrap();
        let safety_result = system.check_safety_gates().await.unwrap();
        
        assert!(safety_result.all_gates_pass);
        assert!(safety_result.violations.is_empty());
    }

    #[tokio::test]
    async fn test_k2_change_budget() {
        let system = SafeAutoDim::new().unwrap();
        let k2_change = system.evaluate_k2_change().await.unwrap();
        
        // Should be None initially due to timing constraints
        assert!(k2_change.is_none());
    }

    #[tokio::test]
    async fn test_dimension_prediction() {
        let system = SafeAutoDim::new().unwrap();
        let prediction = system.predict_optimal_dimension().await.unwrap();
        
        assert!(matches!(prediction.recommended_dimension, DimensionSize::Dim256 | DimensionSize::Dim768));
        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
    }

    #[tokio::test]
    async fn test_canary_rollout() {
        let system = SafeAutoDim::new().unwrap();
        let rollout_status = system.start_canary_rollout().await.unwrap();
        
        assert_eq!(rollout_status.current_phase, 0);
        assert_eq!(rollout_status.phase_name, "Canary");
        assert_eq!(rollout_status.traffic_percentage, 10.0);
    }

    #[test]
    fn test_gradient_boosting_machine() {
        let gbm = GradientBoostingMachine::new();
        
        assert_eq!(gbm.learning_rate, 0.1);
        assert!(gbm.models.is_empty());
        assert!(gbm.training_data.is_empty());
    }

    #[test]
    fn test_safety_gate_violation_detection() {
        let safety_gates = SafetyGates::new();
        
        // Set ECE above threshold
        *safety_gates.current_ece.write().unwrap() = 0.02; // Above 0.01 threshold
        
        let rt = tokio::runtime::Runtime::new().unwrap();
        let status = rt.block_on(safety_gates.get_status());
        
        assert!(!status.ece_status.passing);
        assert!(status.drift_status.passing);
    }
}