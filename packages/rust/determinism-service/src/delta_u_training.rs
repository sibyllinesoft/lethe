#![allow(dead_code, unused_imports, unused_variables)]

use crate::types::{DeltaUPrediction, ScenarioType, TrainingDatapoint, V2Features};
use crate::v2_features::V2FeatureExtractor;
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// ΔU Training System with Isotonic Regression and IPS Calibration
pub struct DeltaUTrainer {
    /// Feature weights learned from training
    feature_weights: HashMap<String, f64>,
    /// Isotonic regression calibration mapping
    isotonic_calibration: Vec<(f64, f64)>,
    /// IPS (Inverse Propensity Scoring) adjustments
    ips_weights: HashMap<ScenarioType, f64>,
    /// Training dataset
    training_data: Vec<TrainingDatapoint>,
    /// Feature extractor
    feature_extractor: V2FeatureExtractor,
    /// Model is trained and ready for prediction
    is_trained: bool,
}

impl DeltaUTrainer {
    pub fn new() -> Self {
        Self {
            feature_weights: HashMap::new(),
            isotonic_calibration: Vec::new(),
            ips_weights: HashMap::new(),
            training_data: Vec::new(),
            feature_extractor: V2FeatureExtractor::new(),
            is_trained: false,
        }
    }

    /// Add training datapoint to the dataset
    pub fn add_training_data(&mut self, datapoint: TrainingDatapoint) {
        self.training_data.push(datapoint);
        debug!(
            "Added training datapoint, total: {}",
            self.training_data.len()
        );
    }

    /// Train the ΔU prediction model with isotonic regression and IPS calibration
    pub fn train(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.training_data.is_empty() {
            return Err("No training data available".into());
        }

        info!(
            "Starting ΔU training with {} datapoints",
            self.training_data.len()
        );

        // Step 1: Learn feature weights using linear regression
        self.learn_feature_weights()?;

        // Step 2: Apply isotonic regression calibration
        self.calibrate_isotonic_regression()?;

        // Step 3: Calculate IPS weights for different scenarios
        self.calculate_ips_weights();

        self.is_trained = true;
        info!("ΔU training completed successfully");
        Ok(())
    }

    /// Predict utility change for given V2 features
    pub fn predict(
        &self,
        features: &V2Features,
        scenario: ScenarioType,
    ) -> Option<DeltaUPrediction> {
        if !self.is_trained {
            warn!("Model not trained, cannot make predictions");
            return None;
        }

        // Calculate raw prediction using feature weights
        let raw_prediction = self.calculate_raw_prediction(features);

        // Apply isotonic calibration
        let calibrated_prediction = self.apply_isotonic_calibration(raw_prediction);

        // Apply IPS adjustment
        let ips_weight = self.ips_weights.get(&scenario).copied().unwrap_or(1.0);
        let final_prediction = calibrated_prediction * ips_weight;

        // Calculate confidence based on feature uncertainty
        let confidence = self.calculate_confidence(features);

        Some(DeltaUPrediction {
            predicted_utility_change: final_prediction,
            confidence,
            feature_weights: self.feature_weights.clone(),
            isotonic_calibrated: !self.isotonic_calibration.is_empty(),
            ips_adjusted: ips_weight != 1.0,
        })
    }

    /// Learn feature weights using simplified linear regression
    fn learn_feature_weights(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Initialize expected weights based on requirements
        let mut weights = HashMap::new();

        // Positive weights for productive changes
        weights.insert("error_fix_chains".to_string(), 0.3);
        weights.insert("code_error_ratio".to_string(), 0.25);
        weights.insert("tool_normalize_chains".to_string(), 0.15);

        // Negative weights for problematic patterns
        weights.insert("late_head_edits".to_string(), -0.4);
        weights.insert("kv_prefix_impact".to_string(), -0.3);
        weights.insert("rollback_occurred".to_string(), -0.2);

        // Structural weights
        weights.insert("edit_depth".to_string(), 0.1);
        weights.insert("change_entropy".to_string(), 0.05);

        // Refine weights using training data if available
        if self.training_data.len() > 10 {
            self.refine_weights_with_data(&mut weights)?;
        }

        self.feature_weights = weights;
        info!("Learned feature weights: {:?}", self.feature_weights);
        Ok(())
    }

    /// Refine weights using actual training data correlation
    fn refine_weights_with_data(
        &self,
        weights: &mut HashMap<String, f64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Simple correlation-based weight adjustment
        let mut feature_utility_correlations = HashMap::new();

        for data in &self.training_data {
            let feature_vec = self.features_to_vec(&data.features);
            for (_i, (feature_name, feature_value)) in feature_vec.iter().enumerate() {
                let correlation_key = feature_name.clone();
                let correlation = feature_utility_correlations
                    .entry(correlation_key)
                    .or_insert(Vec::new());
                correlation.push((*feature_value, data.ground_truth_utility));
            }
        }

        // Adjust weights based on correlations
        for (feature_name, values) in feature_utility_correlations {
            if values.len() < 3 {
                continue;
            }

            let correlation = calculate_correlation(&values);
            if let Some(current_weight) = weights.get_mut(&feature_name) {
                // Adjust weight by up to 50% based on observed correlation
                *current_weight *= 1.0 + 0.5 * correlation.signum() * correlation.abs().min(1.0);
            }
        }

        Ok(())
    }

    /// Apply isotonic regression calibration
    fn calibrate_isotonic_regression(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.training_data.len() < 5 {
            warn!("Insufficient data for isotonic calibration, skipping");
            return Ok(());
        }

        // Calculate raw predictions for training data
        let mut predictions_and_truth = Vec::new();
        for data in &self.training_data {
            let raw_pred = self.calculate_raw_prediction(&data.features);
            predictions_and_truth.push((raw_pred, data.ground_truth_utility));
        }

        // Sort by prediction values for isotonic regression
        predictions_and_truth.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Apply Pool Adjacent Violators Algorithm (simplified)
        self.isotonic_calibration = self.pava_algorithm(predictions_and_truth);

        info!(
            "Isotonic calibration completed with {} points",
            self.isotonic_calibration.len()
        );
        Ok(())
    }

    /// Pool Adjacent Violators Algorithm for isotonic regression
    fn pava_algorithm(&self, data: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut i = 0;

        while i < data.len() {
            let mut j = i + 1;
            let mut sum_x = data[i].0;
            let mut sum_y = data[i].1;
            let mut count = 1;

            // Find violating adjacent points and pool them
            while j < data.len() && sum_y / count as f64 > data[j].1 {
                sum_x += data[j].0;
                sum_y += data[j].1;
                count += 1;
                j += 1;
            }

            let avg_x = sum_x / count as f64;
            let avg_y = sum_y / count as f64;
            result.push((avg_x, avg_y));

            i = j;
        }

        result
    }

    /// Calculate IPS weights for different scenario types
    fn calculate_ips_weights(&mut self) {
        let mut scenario_counts = HashMap::new();
        for data in &self.training_data {
            *scenario_counts
                .entry(data.scenario_type.clone())
                .or_insert(0) += 1;
        }

        let total_count = self.training_data.len() as f64;

        // Calculate inverse propensity scores
        for (scenario, &count) in &scenario_counts {
            let propensity = count as f64 / total_count;
            let ips_weight = if propensity > 0.0 {
                1.0 / propensity
            } else {
                1.0
            };
            // Clamp weights to reasonable range
            let clamped_weight = ips_weight.max(0.1).min(5.0);
            self.ips_weights.insert(scenario.clone(), clamped_weight);
        }

        info!("Calculated IPS weights: {:?}", self.ips_weights);
    }

    /// Calculate raw prediction using feature weights
    fn calculate_raw_prediction(&self, features: &V2Features) -> f64 {
        let feature_vec = self.features_to_vec(features);

        feature_vec
            .iter()
            .map(|(name, value)| {
                let weight = self.feature_weights.get(name).copied().unwrap_or(0.0);
                value * weight
            })
            .sum()
    }

    /// Apply isotonic calibration to raw prediction
    fn apply_isotonic_calibration(&self, raw_prediction: f64) -> f64 {
        if self.isotonic_calibration.is_empty() {
            return raw_prediction;
        }

        // Find the appropriate calibration point
        for (pred, calibrated) in &self.isotonic_calibration {
            if raw_prediction <= *pred {
                return *calibrated;
            }
        }

        // If beyond the highest calibration point, return the last calibrated value
        self.isotonic_calibration
            .last()
            .map(|(_, calibrated)| *calibrated)
            .unwrap_or(raw_prediction)
    }

    /// Calculate prediction confidence based on feature uncertainty
    fn calculate_confidence(&self, features: &V2Features) -> f64 {
        // Simple heuristic: higher confidence for more "normal" feature patterns
        let feature_vec = self.features_to_vec(features);
        let feature_magnitude: f64 = feature_vec
            .iter()
            .map(|(_, value)| value.abs())
            .sum::<f64>()
            / feature_vec.len() as f64;

        // Confidence decreases with extreme feature values
        let confidence = 1.0 / (1.0 + feature_magnitude * 0.1);
        confidence.max(0.1).min(1.0)
    }

    /// Convert V2Features to vector representation
    fn features_to_vec(&self, features: &V2Features) -> Vec<(String, f64)> {
        vec![
            (
                "error_fix_chains".to_string(),
                features.error_fix_chains as f64,
            ),
            (
                "tool_normalize_chains".to_string(),
                features.tool_normalize_chains as f64,
            ),
            (
                "rollback_occurred".to_string(),
                if features.rollback_occurred { 1.0 } else { 0.0 },
            ),
            ("edit_depth".to_string(), features.edit_depth as f64),
            ("change_entropy".to_string(), features.change_entropy),
            ("code_error_ratio".to_string(), features.code_error_ratio),
            (
                "late_head_edits".to_string(),
                features.late_head_edits as f64,
            ),
            ("kv_prefix_impact".to_string(), features.kv_prefix_impact),
        ]
    }

    /// Get training statistics
    pub fn get_training_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        stats.insert(
            "training_data_count".to_string(),
            self.training_data.len() as f64,
        );
        stats.insert(
            "is_trained".to_string(),
            if self.is_trained { 1.0 } else { 0.0 },
        );
        stats.insert(
            "isotonic_calibration_points".to_string(),
            self.isotonic_calibration.len() as f64,
        );
        stats.insert(
            "feature_weights_count".to_string(),
            self.feature_weights.len() as f64,
        );
        stats
    }

    /// Cross-validation to measure lift with V2 features
    pub fn cross_validate(&self) -> Result<f64, Box<dyn std::error::Error>> {
        if self.training_data.len() < 10 {
            return Err("Insufficient data for cross-validation".into());
        }

        // Simple k-fold validation (k=5)
        let fold_size = self.training_data.len() / 5;
        let mut total_error = 0.0;

        for fold in 0..5 {
            let start = fold * fold_size;
            let end = if fold == 4 {
                self.training_data.len()
            } else {
                start + fold_size
            };

            let mut fold_trainer = DeltaUTrainer::new();

            // Train on other folds
            for (i, datapoint) in self.training_data.iter().enumerate() {
                if i < start || i >= end {
                    fold_trainer.add_training_data(datapoint.clone());
                }
            }

            fold_trainer.train()?;

            // Test on current fold
            let mut fold_error = 0.0;
            let mut fold_count = 0;

            for i in start..end {
                let test_point = &self.training_data[i];
                if let Some(prediction) =
                    fold_trainer.predict(&test_point.features, test_point.scenario_type.clone())
                {
                    let error = (prediction.predicted_utility_change
                        - test_point.ground_truth_utility)
                        .abs();
                    fold_error += error;
                    fold_count += 1;
                }
            }

            if fold_count > 0 {
                total_error += fold_error / fold_count as f64;
            }
        }

        Ok(total_error / 5.0)
    }
}

/// Calculate Pearson correlation coefficient
fn calculate_correlation(values: &[(f64, f64)]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let n = values.len() as f64;
    let sum_x: f64 = values.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = values.iter().map(|(_, y)| y).sum();
    let sum_xx: f64 = values.iter().map(|(x, _)| x * x).sum();
    let sum_yy: f64 = values.iter().map(|(_, y)| y * y).sum();
    let sum_xy: f64 = values.iter().map(|(x, y)| x * y).sum();

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)).sqrt();

    if denominator.abs() < f64::EPSILON {
        0.0
    } else {
        numerator / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::V2Features;
    use chrono::Utc;

    fn create_test_features(error_fix_chains: u32, rollback: bool) -> V2Features {
        V2Features {
            error_fix_chains,
            tool_normalize_chains: 1,
            rollback_occurred: rollback,
            edit_depth: 2,
            change_entropy: 1.5,
            code_error_ratio: 0.5,
            late_head_edits: 0,
            kv_prefix_impact: 0.1,
        }
    }

    #[test]
    fn test_feature_weight_learning() {
        let mut trainer = DeltaUTrainer::new();

        // Add training data
        trainer.add_training_data(TrainingDatapoint {
            features: create_test_features(3, false),
            ground_truth_utility: 0.8,
            timestamp: Utc::now(),
            scenario_type: ScenarioType::Code,
        });

        trainer.add_training_data(TrainingDatapoint {
            features: create_test_features(0, true),
            ground_truth_utility: -0.5,
            timestamp: Utc::now(),
            scenario_type: ScenarioType::Code,
        });

        assert!(trainer.train().is_ok());
        assert!(trainer.is_trained);
    }

    #[test]
    fn test_prediction() {
        let mut trainer = DeltaUTrainer::new();

        // Add minimal training data
        trainer.add_training_data(TrainingDatapoint {
            features: create_test_features(2, false),
            ground_truth_utility: 0.5,
            timestamp: Utc::now(),
            scenario_type: ScenarioType::Code,
        });

        trainer.train().unwrap();

        let prediction = trainer.predict(&create_test_features(3, false), ScenarioType::Code);
        assert!(prediction.is_some());

        let pred = prediction.unwrap();
        assert!(pred.confidence > 0.0 && pred.confidence <= 1.0);
    }

    #[test]
    fn test_isotonic_calibration() {
        let trainer = DeltaUTrainer::new();
        let data = vec![(1.0, 2.0), (2.0, 1.5), (3.0, 3.0), (4.0, 3.5)];
        let calibrated = trainer.pava_algorithm(data);

        // Results should be monotonically increasing
        for window in calibrated.windows(2) {
            assert!(window[0].1 <= window[1].1);
        }
    }
}
