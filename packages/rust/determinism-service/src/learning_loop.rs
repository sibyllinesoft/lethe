#![allow(dead_code, unused_imports, unused_variables)]

use crate::delta_u_training::DeltaUTrainer;
use crate::lambda_mu_controller::LambdaMuControllerImpl;
use crate::types::{ScenarioType, TrainingDatapoint, TransformChangeV2, V2Features};
use crate::v2_features::V2FeatureExtractor;
use chrono::{DateTime, Duration, Utc};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Learning Loop Closure - Integrates V2 features, ΔU training, and λ/μ controller
pub struct LearningLoopService {
    /// ΔU training system
    delta_u_trainer: Arc<RwLock<DeltaUTrainer>>,
    /// λ/μ controller with V2 integration
    lambda_mu_controller: Arc<RwLock<LambdaMuControllerImpl>>,
    /// Feature extractor
    feature_extractor: Arc<RwLock<V2FeatureExtractor>>,
    /// Recent transform changes for analysis
    recent_changes: Arc<RwLock<VecDeque<TransformChangeV2>>>,
    /// Performance metrics history
    performance_metrics: Arc<RwLock<VecDeque<PerformanceMetric>>>,
    /// A/B testing state
    ab_testing_state: Arc<RwLock<ABTestingState>>,
    /// Configuration
    config: LearningLoopConfig,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerformanceMetric {
    pub timestamp: DateTime<Utc>,
    pub p95_latency_ms: f64,
    pub throughput_ops_per_sec: f64,
    pub memory_usage_mb: f64,
    pub accuracy_score: f64,
    pub lambda: f64,
    pub mu: f64,
    pub context_size: u32,
}

#[derive(Debug, Clone)]
pub struct ABTestingState {
    pub current_variant: String,
    pub variants: HashMap<String, VariantConfig>,
    pub test_start_time: DateTime<Utc>,
    pub test_duration_hours: u32,
    pub metric_buckets: HashMap<String, Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct VariantConfig {
    pub use_v2_features: bool,
    pub use_isotonic_calibration: bool,
    pub use_ips_adjustment: bool,
    pub lambda_mu_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct LearningLoopConfig {
    /// Maximum number of recent changes to keep in memory
    pub max_recent_changes: usize,
    /// Maximum number of performance metrics to keep
    pub max_performance_history: usize,
    /// Minimum number of datapoints before retraining
    pub min_training_datapoints: usize,
    /// Retraining interval in hours
    pub retrain_interval_hours: u32,
    /// A/B test duration in hours
    pub ab_test_duration_hours: u32,
    /// Statistical significance threshold for A/B tests
    pub significance_threshold: f64,
}

impl Default for LearningLoopConfig {
    fn default() -> Self {
        Self {
            max_recent_changes: 1000,
            max_performance_history: 500,
            min_training_datapoints: 50,
            retrain_interval_hours: 24,
            ab_test_duration_hours: 48,
            significance_threshold: 0.05,
        }
    }
}

impl LearningLoopService {
    pub fn new(config: Option<LearningLoopConfig>) -> Self {
        let config = config.unwrap_or_default();

        // Initialize A/B testing variants
        let mut variants = HashMap::new();

        // Control variant - no V2 features
        variants.insert(
            "control".to_string(),
            VariantConfig {
                use_v2_features: false,
                use_isotonic_calibration: false,
                use_ips_adjustment: false,
                lambda_mu_enabled: false,
            },
        );

        // Treatment variant - full V2 features
        variants.insert(
            "v2_full".to_string(),
            VariantConfig {
                use_v2_features: true,
                use_isotonic_calibration: true,
                use_ips_adjustment: true,
                lambda_mu_enabled: true,
            },
        );

        // V2 features only variant
        variants.insert(
            "v2_features_only".to_string(),
            VariantConfig {
                use_v2_features: true,
                use_isotonic_calibration: false,
                use_ips_adjustment: false,
                lambda_mu_enabled: false,
            },
        );

        let ab_testing_state = ABTestingState {
            current_variant: "control".to_string(),
            variants,
            test_start_time: Utc::now(),
            test_duration_hours: config.ab_test_duration_hours,
            metric_buckets: HashMap::new(),
        };

        Self {
            delta_u_trainer: Arc::new(RwLock::new(DeltaUTrainer::new())),
            lambda_mu_controller: Arc::new(RwLock::new(LambdaMuControllerImpl::new(None))),
            feature_extractor: Arc::new(RwLock::new(V2FeatureExtractor::new())),
            recent_changes: Arc::new(RwLock::new(VecDeque::new())),
            performance_metrics: Arc::new(RwLock::new(VecDeque::new())),
            ab_testing_state: Arc::new(RwLock::new(ab_testing_state)),
            config,
        }
    }

    /// Process a batch of transform changes and update the learning loop
    pub async fn process_changes(
        &self,
        changes: Vec<TransformChangeV2>,
        scenario_type: ScenarioType,
    ) -> Result<LearningLoopResult, Box<dyn std::error::Error + Send + Sync>> {
        // Add changes to recent history
        {
            let mut recent = self.recent_changes.write().await;
            for change in &changes {
                recent.push_back(change.clone());
            }

            // Maintain size limit
            while recent.len() > self.config.max_recent_changes {
                recent.pop_front();
            }
        }

        // Get current A/B testing variant
        let current_variant = {
            let ab_state = self.ab_testing_state.read().await;
            ab_state.current_variant.clone()
        };

        // Process based on current variant
        let result = match current_variant.as_str() {
            "control" => {
                self.process_control_variant(&changes, scenario_type)
                    .await?
            }
            "v2_full" => {
                self.process_v2_full_variant(&changes, scenario_type)
                    .await?
            }
            "v2_features_only" => {
                self.process_v2_features_only(&changes, scenario_type)
                    .await?
            }
            _ => {
                warn!(
                    "Unknown A/B testing variant: {}, falling back to control",
                    current_variant
                );
                self.process_control_variant(&changes, scenario_type)
                    .await?
            }
        };

        // Record metrics for A/B testing
        self.record_ab_testing_metrics(&result, &current_variant)
            .await;

        // Check if we should retrain or switch variants
        self.check_maintenance_tasks().await?;

        Ok(result)
    }

    /// Process changes using control variant (no V2 features)
    async fn process_control_variant(
        &self,
        _changes: &[TransformChangeV2],
        _scenario_type: ScenarioType,
    ) -> Result<LearningLoopResult, Box<dyn std::error::Error + Send + Sync>> {
        // Simple baseline processing without V2 features
        let prediction_accuracy = 0.75; // Baseline accuracy
        let lambda = 0.12; // Fixed head keep ratio
        let mu = 1.0; // Fixed tail parameter
        let context_size = 2048; // Fixed context size

        Ok(LearningLoopResult {
            prediction_accuracy,
            lambda,
            mu,
            context_size,
            variant_used: "control".to_string(),
            v2_features_enabled: false,
            calibration_applied: false,
            ips_adjusted: false,
            computation_time_ms: 1.0, // Minimal computation
            feature_extraction: None,
        })
    }

    /// Process changes using full V2 feature set
    async fn process_v2_full_variant(
        &self,
        changes: &[TransformChangeV2],
        scenario_type: ScenarioType,
    ) -> Result<LearningLoopResult, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = std::time::Instant::now();

        // Extract V2 features
        let features = {
            let mut extractor = self.feature_extractor.write().await;
            extractor.extract_features(changes)
        };

        // Get ΔU prediction
        let prediction = {
            let trainer = self.delta_u_trainer.read().await;
            trainer.predict(&features, scenario_type)
        };

        // Update λ/μ controller
        let (lambda, mu) = {
            let mut controller = self.lambda_mu_controller.write().await;
            // Use latest performance metrics
            let latest_perf = self.get_latest_performance_metrics().await;
            controller.update_with_v2_features(
                changes,
                latest_perf.p95_latency_ms,
                latest_perf.memory_usage_mb,
            )
        };

        let context_size = {
            let controller = self.lambda_mu_controller.read().await;
            controller.get_state().target_k2
        };

        let computation_time = start_time.elapsed().as_secs_f64() * 1000.0;

        // Calculate prediction accuracy (would be measured against actual outcomes)
        let prediction_accuracy = prediction.as_ref().map(|p| p.confidence).unwrap_or(0.5);

        Ok(LearningLoopResult {
            prediction_accuracy,
            lambda,
            mu,
            context_size,
            variant_used: "v2_full".to_string(),
            v2_features_enabled: true,
            calibration_applied: prediction
                .as_ref()
                .map(|p| p.isotonic_calibrated)
                .unwrap_or(false),
            ips_adjusted: prediction.as_ref().map(|p| p.ips_adjusted).unwrap_or(false),
            computation_time_ms: computation_time,
            feature_extraction: Some(features),
        })
    }

    /// Process changes using V2 features only (no controller updates)
    async fn process_v2_features_only(
        &self,
        changes: &[TransformChangeV2],
        scenario_type: ScenarioType,
    ) -> Result<LearningLoopResult, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = std::time::Instant::now();

        // Extract V2 features
        let features = {
            let mut extractor = self.feature_extractor.write().await;
            extractor.extract_features(changes)
        };

        // Get ΔU prediction but don't update controller
        let prediction = {
            let trainer = self.delta_u_trainer.read().await;
            trainer.predict(&features, scenario_type)
        };

        let computation_time = start_time.elapsed().as_secs_f64() * 1000.0;

        let prediction_accuracy = prediction.as_ref().map(|p| p.confidence).unwrap_or(0.5);

        Ok(LearningLoopResult {
            prediction_accuracy,
            lambda: 0.12,       // Fixed
            mu: 1.0,            // Fixed
            context_size: 2048, // Fixed
            variant_used: "v2_features_only".to_string(),
            v2_features_enabled: true,
            calibration_applied: prediction
                .as_ref()
                .map(|p| p.isotonic_calibrated)
                .unwrap_or(false),
            ips_adjusted: prediction.as_ref().map(|p| p.ips_adjusted).unwrap_or(false),
            computation_time_ms: computation_time,
            feature_extraction: Some(features),
        })
    }

    /// Add training datapoint for continuous learning
    pub async fn add_training_datapoint(
        &self,
        features: V2Features,
        ground_truth_utility: f64,
        scenario_type: ScenarioType,
    ) {
        let datapoint = TrainingDatapoint {
            features,
            ground_truth_utility,
            timestamp: Utc::now(),
            scenario_type,
        };

        let mut trainer = self.delta_u_trainer.write().await;
        trainer.add_training_data(datapoint);

        debug!(
            "Added training datapoint, total: {}",
            trainer
                .get_training_stats()
                .get("training_data_count")
                .unwrap_or(&0.0)
        );
    }

    /// Record performance metrics
    pub async fn record_performance(&self, metric: PerformanceMetric) {
        let mut metrics = self.performance_metrics.write().await;
        metrics.push_back(metric);

        // Maintain size limit
        while metrics.len() > self.config.max_performance_history {
            metrics.pop_front();
        }
    }

    /// Get latest performance metrics (or default if none available)
    async fn get_latest_performance_metrics(&self) -> PerformanceMetric {
        let metrics = self.performance_metrics.read().await;
        metrics
            .back()
            .cloned()
            .unwrap_or_else(|| PerformanceMetric {
                timestamp: Utc::now(),
                p95_latency_ms: 100.0,
                throughput_ops_per_sec: 1000.0,
                memory_usage_mb: 500.0,
                accuracy_score: 0.8,
                lambda: 0.12,
                mu: 1.0,
                context_size: 2048,
            })
    }

    /// Record metrics for A/B testing analysis
    async fn record_ab_testing_metrics(&self, result: &LearningLoopResult, variant: &str) {
        let mut ab_state = self.ab_testing_state.write().await;

        // Record key metrics by variant
        let metrics_key = format!("{}_accuracy", variant);
        ab_state
            .metric_buckets
            .entry(metrics_key)
            .or_insert_with(Vec::new)
            .push(result.prediction_accuracy);

        let compute_key = format!("{}_compute_time", variant);
        ab_state
            .metric_buckets
            .entry(compute_key)
            .or_insert_with(Vec::new)
            .push(result.computation_time_ms);
    }

    /// Check if maintenance tasks need to be performed
    async fn check_maintenance_tasks(
        &self,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Check if we need to retrain the model
        let training_stats = {
            let trainer = self.delta_u_trainer.read().await;
            trainer.get_training_stats()
        };

        let datapoint_count = training_stats
            .get("training_data_count")
            .copied()
            .unwrap_or(0.0) as usize;

        if datapoint_count >= self.config.min_training_datapoints {
            // Retrain in background
            tokio::spawn({
                let trainer = Arc::clone(&self.delta_u_trainer);
                async move {
                    let mut trainer_guard = trainer.write().await;
                    match trainer_guard.train() {
                        Ok(_) => info!(
                            "Model retrained successfully with {} datapoints",
                            datapoint_count
                        ),
                        Err(e) => error!("Model retraining failed: {}", e),
                    }
                }
            });
        }

        // Check if A/B test should be rotated
        self.check_ab_test_rotation().await?;

        Ok(())
    }

    /// Check if A/B test should be rotated to next variant
    async fn check_ab_test_rotation(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut ab_state = self.ab_testing_state.write().await;
        let now = Utc::now();
        let test_elapsed = now - ab_state.test_start_time;

        if test_elapsed > Duration::hours(ab_state.test_duration_hours as i64) {
            // Rotate to next variant
            let current_variant = &ab_state.current_variant;
            let next_variant = match current_variant.as_str() {
                "control" => "v2_features_only",
                "v2_features_only" => "v2_full",
                "v2_full" => "control",
                _ => "control",
            };

            info!(
                "Rotating A/B test from {} to {}",
                current_variant, next_variant
            );

            ab_state.current_variant = next_variant.to_string();
            ab_state.test_start_time = now;
            ab_state.metric_buckets.clear(); // Reset metrics for new test
        }

        Ok(())
    }

    /// Get comprehensive learning loop metrics
    pub async fn get_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        // ΔU training metrics
        {
            let trainer = self.delta_u_trainer.read().await;
            let training_stats = trainer.get_training_stats();
            for (key, value) in training_stats {
                metrics.insert(format!("delta_u_{}", key), value);
            }
        }

        // λ/μ controller metrics
        {
            let controller = self.lambda_mu_controller.read().await;
            let controller_metrics = controller.get_metrics();
            for (key, value) in controller_metrics {
                metrics.insert(format!("controller_{}", key), value);
            }
        }

        // A/B testing metrics
        {
            let ab_state = self.ab_testing_state.read().await;
            metrics.insert(
                "ab_test_variant".to_string(),
                match ab_state.current_variant.as_str() {
                    "control" => 0.0,
                    "v2_features_only" => 1.0,
                    "v2_full" => 2.0,
                    _ => -1.0,
                },
            );

            let test_elapsed_hours = (Utc::now() - ab_state.test_start_time).num_hours();
            metrics.insert(
                "ab_test_elapsed_hours".to_string(),
                test_elapsed_hours as f64,
            );
        }

        // Performance metrics
        {
            let perf_metrics = self.performance_metrics.read().await;
            metrics.insert(
                "performance_history_count".to_string(),
                perf_metrics.len() as f64,
            );

            if let Some(latest) = perf_metrics.back() {
                metrics.insert("latest_p95_latency_ms".to_string(), latest.p95_latency_ms);
                metrics.insert("latest_accuracy".to_string(), latest.accuracy_score);
                metrics.insert("latest_memory_mb".to_string(), latest.memory_usage_mb);
            }
        }

        metrics
    }

    /// Get A/B testing results analysis
    pub async fn get_ab_test_results(&self) -> ABTestResults {
        let ab_state = self.ab_testing_state.read().await;
        let mut variant_results = HashMap::new();

        for (variant_metric, values) in &ab_state.metric_buckets {
            if !values.is_empty() {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance =
                    values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
                let std_dev = variance.sqrt();

                variant_results.insert(
                    variant_metric.clone(),
                    VariantResult {
                        mean,
                        std_dev,
                        count: values.len(),
                        confidence_interval_95: (
                            mean - 1.96 * std_dev / (values.len() as f64).sqrt(),
                            mean + 1.96 * std_dev / (values.len() as f64).sqrt(),
                        ),
                    },
                );
            }
        }

        let significant_difference = self.calculate_statistical_significance(&variant_results);

        ABTestResults {
            current_variant: ab_state.current_variant.clone(),
            test_start_time: ab_state.test_start_time,
            variant_results,
            significant_difference,
        }
    }

    /// Calculate statistical significance between variants
    fn calculate_statistical_significance(&self, results: &HashMap<String, VariantResult>) -> bool {
        // Simplified significance test - in practice would use proper t-test or Chi-square
        if let (Some(control), Some(treatment)) = (
            results.get("control_accuracy"),
            results.get("v2_full_accuracy"),
        ) {
            let diff = (treatment.mean - control.mean).abs();
            let pooled_se = (control.std_dev.powi(2) / control.count as f64
                + treatment.std_dev.powi(2) / treatment.count as f64)
                .sqrt();

            let t_stat = diff / pooled_se;
            // Rough threshold for significance
            t_stat > 1.96
        } else {
            false
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LearningLoopResult {
    pub prediction_accuracy: f64,
    pub lambda: f64,
    pub mu: f64,
    pub context_size: u32,
    pub variant_used: String,
    pub v2_features_enabled: bool,
    pub calibration_applied: bool,
    pub ips_adjusted: bool,
    pub computation_time_ms: f64,
    pub feature_extraction: Option<V2Features>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ABTestResults {
    pub current_variant: String,
    pub test_start_time: DateTime<Utc>,
    pub variant_results: HashMap<String, VariantResult>,
    pub significant_difference: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct VariantResult {
    pub mean: f64,
    pub std_dev: f64,
    pub count: usize,
    pub confidence_interval_95: (f64, f64),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChangeMetadata, ChangeType};
    use uuid::Uuid;

    fn create_test_change(change_type: ChangeType) -> TransformChangeV2 {
        TransformChangeV2 {
            change_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            change_type,
            metadata: ChangeMetadata {
                depth: 1,
                complexity_score: 1.0,
                edit_distance: Some(10),
                context_size: 100,
                causality_chain: vec![],
            },
            before_state: None,
            after_state: None,
            performance_impact: None,
        }
    }

    #[tokio::test]
    async fn test_learning_loop_initialization() {
        let service = LearningLoopService::new(None);
        let metrics = service.get_metrics().await;

        assert!(metrics.contains_key("delta_u_training_data_count"));
        assert!(metrics.contains_key("controller_current_lambda"));
        assert!(metrics.contains_key("ab_test_variant"));
    }

    #[tokio::test]
    async fn test_change_processing() {
        let service = LearningLoopService::new(None);

        let changes = vec![
            create_test_change(ChangeType::Code),
            create_test_change(ChangeType::Error),
            create_test_change(ChangeType::Fix),
        ];

        let result = service.process_changes(changes, ScenarioType::Code).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.variant_used, "control"); // Should start with control variant
        assert!(!result.v2_features_enabled); // Control variant doesn't use V2 features
    }

    #[tokio::test]
    async fn test_performance_recording() {
        let service = LearningLoopService::new(None);

        let metric = PerformanceMetric {
            timestamp: Utc::now(),
            p95_latency_ms: 95.0,
            throughput_ops_per_sec: 1200.0,
            memory_usage_mb: 450.0,
            accuracy_score: 0.85,
            lambda: 0.12,
            mu: 1.0,
            context_size: 2048,
        };

        service.record_performance(metric).await;

        let latest = service.get_latest_performance_metrics().await;
        assert_eq!(latest.p95_latency_ms, 95.0);
    }

    #[tokio::test]
    async fn test_ab_test_metrics_recording() {
        let service = LearningLoopService::new(None);

        let result = LearningLoopResult {
            prediction_accuracy: 0.85,
            lambda: 0.12,
            mu: 1.0,
            context_size: 2048,
            variant_used: "control".to_string(),
            v2_features_enabled: false,
            calibration_applied: false,
            ips_adjusted: false,
            computation_time_ms: 5.0,
            feature_extraction: None,
        };

        service.record_ab_testing_metrics(&result, "control").await;

        let ab_results = service.get_ab_test_results().await;
        assert!(ab_results.variant_results.contains_key("control_accuracy"));
    }
}
