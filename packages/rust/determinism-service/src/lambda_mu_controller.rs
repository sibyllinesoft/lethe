#![allow(dead_code, unused_imports, unused_variables)]

use crate::types::{HysteresisState, LambdaMuController, TransformChangeV2};
use crate::v2_features::V2FeatureExtractor;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// λ/μ Controller with V2 Feature Integration and Hysteresis
///
/// Controls context window management with:
/// - λ (lambda): Head retention parameter
/// - μ (mu): Tail management parameter
/// - K2: Target context window size
/// - Hysteresis: Prevents oscillation in parameter adjustments
pub struct LambdaMuControllerImpl {
    /// Current controller state
    state: LambdaMuController,
    /// Feature extractor for difficulty assessment
    feature_extractor: V2FeatureExtractor,
    /// Performance history for adaptive control
    performance_history: Vec<PerformanceSnapshot>,
    /// Configuration parameters
    config: ControllerConfig,
}

#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    pub timestamp: DateTime<Utc>,
    pub p95_latency_ms: f64,
    pub target_latency_ms: f64,
    pub context_size: u32,
    pub memory_usage_mb: f64,
}

#[derive(Debug, Clone)]
pub struct ControllerConfig {
    /// Base context size (K0)
    pub k0: u32,
    /// Difficulty scaling factor (κ)
    pub kappa: f64,
    /// Learning rate for μ adjustments (η)
    pub eta: f64,
    /// Growth hysteresis factor (smaller = more conservative growth)
    pub alpha_grow: f64,
    /// Shrink hysteresis factor (larger = more aggressive shrinking)
    pub alpha_shrink: f64,
    /// Target p95 latency in milliseconds
    pub target_p95_latency_ms: f64,
    /// Jaccard drop threshold for tail discipline
    pub jaccard_drop_threshold: f64,
    /// Head keep ratio (approximately 0.12 as specified)
    pub head_keep_ratio: f64,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            k0: 2048,
            kappa: 0.5,
            eta: 0.1,
            alpha_grow: 0.05,
            alpha_shrink: 0.15,
            target_p95_latency_ms: 100.0,
            jaccard_drop_threshold: 0.10,
            head_keep_ratio: 0.12,
        }
    }
}

impl LambdaMuControllerImpl {
    pub fn new(config: Option<ControllerConfig>) -> Self {
        let config = config.unwrap_or_default();
        let now = Utc::now();

        Self {
            state: LambdaMuController {
                current_lambda: config.head_keep_ratio,
                current_mu: 1.0,
                target_k2: config.k0,
                difficulty_score: 0.0,
                hysteresis_state: HysteresisState {
                    alpha_shrink: config.alpha_shrink,
                    alpha_grow: config.alpha_grow,
                    last_adjustment: now,
                    consecutive_adjustments: 0,
                },
            },
            feature_extractor: V2FeatureExtractor::new(),
            performance_history: Vec::new(),
            config,
        }
    }

    /// Update controller parameters based on V2 features and performance
    pub fn update_with_v2_features(
        &mut self,
        changes: &[TransformChangeV2],
        current_p95_latency: f64,
        current_memory_mb: f64,
    ) -> (f64, f64) {
        // Extract V2 features for difficulty assessment
        let features = self.feature_extractor.extract_features(changes);

        // Calculate difficulty score from V2 features
        let difficulty = self.calculate_difficulty_score(&features, changes);
        self.state.difficulty_score = difficulty;

        // Calculate target context size based on difficulty
        let new_target_k2 = self.calculate_target_k2(difficulty);
        self.state.target_k2 = new_target_k2;

        // Update performance history
        self.add_performance_snapshot(current_p95_latency, current_memory_mb);

        // Apply hysteresis-based μ adjustment
        let new_mu = self.adjust_mu_with_hysteresis(current_p95_latency);
        let new_lambda = self.adjust_lambda_with_kv_impact(&features, changes);

        self.state.current_lambda = new_lambda;
        self.state.current_mu = new_mu;

        info!(
            "Controller update: λ={:.3}, μ={:.3}, K2={}, difficulty={:.3}",
            new_lambda, new_mu, new_target_k2, difficulty
        );

        (new_lambda, new_mu)
    }

    /// Calculate difficulty score from V2 features
    fn calculate_difficulty_score(
        &self,
        features: &crate::types::V2Features,
        _changes: &[TransformChangeV2],
    ) -> f64 {
        // Multi-factor difficulty assessment
        let entropy_factor = 0.3 * features.change_entropy;
        let rollback_factor = if features.rollback_occurred { 0.4 } else { 0.0 };
        let depth_factor = 0.2 * (features.edit_depth as f64 / 10.0).min(1.0);
        let kv_instability_factor = 0.1 * features.kv_prefix_impact;

        let base_difficulty =
            entropy_factor + rollback_factor + depth_factor + kv_instability_factor;

        // Clamp difficulty to reasonable range
        base_difficulty.max(0.0).min(2.0)
    }

    /// Calculate target context size (K2) based on difficulty
    fn calculate_target_k2(&self, difficulty: f64) -> u32 {
        let scaling_factor = 1.0 + self.config.kappa * difficulty;
        let target = (self.config.k0 as f64 * scaling_factor) as u32;

        // Clamp to reasonable bounds
        target.max(1024).min(8192)
    }

    /// Adjust μ parameter with hysteresis to prevent oscillation
    fn adjust_mu_with_hysteresis(&mut self, current_p95_latency: f64) -> f64 {
        let target_latency = self.config.target_p95_latency_ms;
        let p95_ratio = current_p95_latency / target_latency;

        // Calculate adjustment magnitude with hysteresis
        let adjustment_magnitude = if p95_ratio > 1.0 {
            // Performance is worse than target - shrink more aggressively
            (p95_ratio - 1.0).min(self.state.hysteresis_state.alpha_shrink)
        } else {
            // Performance is better than target - grow more conservatively
            (1.0 - p95_ratio).min(self.state.hysteresis_state.alpha_grow)
        };

        // Apply exponential adjustment
        let adjustment_factor = if p95_ratio > 1.0 {
            // Shrink: reduce μ to tighten context management
            (-self.config.eta * adjustment_magnitude).exp()
        } else {
            // Grow: increase μ to relax context management
            (self.config.eta * adjustment_magnitude).exp()
        };

        let new_mu = self.state.current_mu * adjustment_factor;

        // Update hysteresis state
        self.update_hysteresis_state(adjustment_factor != 1.0);

        // Clamp μ to reasonable range
        new_mu.max(0.1).min(2.0)
    }

    /// Adjust λ parameter based on KV cache impact
    fn adjust_lambda_with_kv_impact(
        &self,
        features: &crate::types::V2Features,
        changes: &[TransformChangeV2],
    ) -> f64 {
        let base_lambda = self.config.head_keep_ratio;

        // Calculate prefix Jaccard drop
        let jaccard_drop = self.calculate_prefix_jaccard_drop(changes);

        if jaccard_drop > self.config.jaccard_drop_threshold {
            // High KV instability - be more conservative with head retention
            debug!(
                "High KV instability detected: {:.3}, reducing λ",
                jaccard_drop
            );
            return base_lambda * 0.9;
        }

        // Adjust based on late head edits
        if features.late_head_edits > 0 {
            // Late head edits indicate potential KV cache disruption
            let late_edit_penalty = (features.late_head_edits as f64 * 0.05).min(0.2);
            return (base_lambda - late_edit_penalty).max(0.05);
        }

        base_lambda
    }

    /// Calculate prefix Jaccard similarity drop (simplified implementation)
    fn calculate_prefix_jaccard_drop(&self, changes: &[TransformChangeV2]) -> f64 {
        if changes.len() < 2 {
            return 0.0;
        }

        // Count prefix-affecting changes
        let prefix_affecting_changes = changes
            .iter()
            .filter(|c| {
                matches!(
                    c.change_type,
                    crate::types::ChangeType::HeadSummary | crate::types::ChangeType::KvUpdate
                )
            })
            .count();

        let total_changes = changes.len();
        let instability_ratio = prefix_affecting_changes as f64 / total_changes as f64;

        // Simulate Jaccard drop - in real implementation this would analyze actual prefixes
        instability_ratio.min(1.0)
    }

    /// Update hysteresis state to prevent oscillation
    fn update_hysteresis_state(&mut self, adjustment_made: bool) {
        let now = Utc::now();

        if adjustment_made {
            self.state.hysteresis_state.consecutive_adjustments += 1;

            // Increase hysteresis if too many consecutive adjustments
            if self.state.hysteresis_state.consecutive_adjustments > 5 {
                self.state.hysteresis_state.alpha_grow *= 0.8; // More conservative growth
                self.state.hysteresis_state.alpha_shrink *= 1.1; // More aggressive shrinking
                warn!("High oscillation detected, adjusting hysteresis parameters");
            }
        } else {
            // Reset consecutive counter if no adjustment made
            self.state.hysteresis_state.consecutive_adjustments = 0;
        }

        self.state.hysteresis_state.last_adjustment = now;
    }

    /// Add performance snapshot to history
    fn add_performance_snapshot(&mut self, p95_latency: f64, memory_mb: f64) {
        let snapshot = PerformanceSnapshot {
            timestamp: Utc::now(),
            p95_latency_ms: p95_latency,
            target_latency_ms: self.config.target_p95_latency_ms,
            context_size: self.state.target_k2,
            memory_usage_mb: memory_mb,
        };

        self.performance_history.push(snapshot);

        // Keep only recent history (last 100 snapshots)
        if self.performance_history.len() > 100 {
            self.performance_history.remove(0);
        }
    }

    /// Get current controller state
    pub fn get_state(&self) -> &LambdaMuController {
        &self.state
    }

    /// Get controller performance metrics
    pub fn get_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        metrics.insert("current_lambda".to_string(), self.state.current_lambda);
        metrics.insert("current_mu".to_string(), self.state.current_mu);
        metrics.insert("target_k2".to_string(), self.state.target_k2 as f64);
        metrics.insert("difficulty_score".to_string(), self.state.difficulty_score);
        metrics.insert(
            "consecutive_adjustments".to_string(),
            self.state.hysteresis_state.consecutive_adjustments as f64,
        );

        if let Some(latest_perf) = self.performance_history.last() {
            metrics.insert(
                "latest_p95_latency_ms".to_string(),
                latest_perf.p95_latency_ms,
            );
            metrics.insert(
                "performance_ratio".to_string(),
                latest_perf.p95_latency_ms / latest_perf.target_latency_ms,
            );
        }

        metrics
    }

    /// Check if controller is stable (not oscillating)
    pub fn is_stable(&self) -> bool {
        self.state.hysteresis_state.consecutive_adjustments <= 3
    }

    /// Reset controller to default state
    pub fn reset(&mut self) {
        let now = Utc::now();
        self.state = LambdaMuController {
            current_lambda: self.config.head_keep_ratio,
            current_mu: 1.0,
            target_k2: self.config.k0,
            difficulty_score: 0.0,
            hysteresis_state: HysteresisState {
                alpha_shrink: self.config.alpha_shrink,
                alpha_grow: self.config.alpha_grow,
                last_adjustment: now,
                consecutive_adjustments: 0,
            },
        };
        self.performance_history.clear();
        info!("Controller reset to default state");
    }

    /// Get performance trend over recent history
    pub fn get_performance_trend(&self) -> Option<f64> {
        if self.performance_history.len() < 5 {
            return None;
        }

        let recent = &self.performance_history[self.performance_history.len() - 5..];
        let first_ratio = recent[0].p95_latency_ms / recent[0].target_latency_ms;
        let last_ratio = recent[4].p95_latency_ms / recent[4].target_latency_ms;

        Some(last_ratio - first_ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChangeMetadata, ChangeType, TransformChangeV2};
    use uuid::Uuid;

    fn create_test_change(change_type: ChangeType, depth: u32) -> TransformChangeV2 {
        TransformChangeV2 {
            change_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            change_type,
            metadata: ChangeMetadata {
                depth,
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

    #[test]
    fn test_controller_initialization() {
        let controller = LambdaMuControllerImpl::new(None);
        assert!(controller.state.current_lambda > 0.0);
        assert!(controller.state.current_mu > 0.0);
        assert!(controller.state.target_k2 > 0);
    }

    #[test]
    fn test_difficulty_calculation() {
        let mut controller = LambdaMuControllerImpl::new(None);

        // High difficulty scenario
        let high_difficulty_changes = vec![
            create_test_change(ChangeType::Error, 5),
            create_test_change(ChangeType::Rollback, 3),
            create_test_change(ChangeType::HeadSummary, 2),
        ];

        let (lambda, mu) =
            controller.update_with_v2_features(&high_difficulty_changes, 150.0, 500.0);

        // Should adjust parameters based on difficulty
        assert!(controller.state.difficulty_score > 0.0);
        assert!(controller.state.target_k2 >= controller.config.k0);
    }

    #[test]
    fn test_hysteresis_prevents_oscillation() {
        let mut controller = LambdaMuControllerImpl::new(None);

        let changes = vec![create_test_change(ChangeType::Code, 1)];

        // Simulate alternating performance conditions
        for i in 0..10 {
            let latency = if i % 2 == 0 { 80.0 } else { 120.0 };
            controller.update_with_v2_features(&changes, latency, 400.0);
        }

        // Should have high consecutive adjustments due to oscillation
        assert!(controller.state.hysteresis_state.consecutive_adjustments > 0);
    }

    #[test]
    fn test_kv_impact_adjustment() {
        let mut controller = LambdaMuControllerImpl::new(None);

        // Changes with high KV impact
        let kv_heavy_changes = vec![
            create_test_change(ChangeType::HeadSummary, 1),
            create_test_change(ChangeType::HeadSummary, 1),
            create_test_change(ChangeType::KvUpdate, 1),
        ];

        let initial_lambda = controller.state.current_lambda;
        let (new_lambda, _) = controller.update_with_v2_features(&kv_heavy_changes, 100.0, 400.0);

        // Lambda should be adjusted down due to KV instability
        assert!(new_lambda <= initial_lambda);
    }

    #[test]
    fn test_performance_trend_detection() {
        let mut controller = LambdaMuControllerImpl::new(None);
        let changes = vec![create_test_change(ChangeType::Code, 1)];

        // Add performance snapshots with improving trend
        for i in 0..5 {
            let latency = 150.0 - (i as f64 * 10.0); // Improving performance
            controller.update_with_v2_features(&changes, latency, 400.0);
        }

        let trend = controller.get_performance_trend();
        assert!(trend.is_some());
        assert!(trend.unwrap() < 0.0); // Negative trend = improving performance
    }
}
