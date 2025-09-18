#[cfg(test)]
mod tests {
    use crate::learning_loop::{LearningLoopService, PerformanceMetric};
    use crate::types::{ChangeMetadata, ChangeType, ScenarioType, TransformChangeV2, V2Features};
    use chrono::Utc;
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

    #[tokio::test]
    async fn test_end_to_end_learning_loop() {
        let service = LearningLoopService::new(None);

        // Step 1: Process some changes to gather data
        let changes = vec![
            create_test_change(ChangeType::Error, 2),
            create_test_change(ChangeType::Fix, 1),
            create_test_change(ChangeType::Code, 1),
        ];

        let result = service
            .process_changes(changes.clone(), ScenarioType::Code)
            .await;
        assert!(result.is_ok());
        let result = result.unwrap();

        // Should start with control variant
        assert_eq!(result.variant_used, "control");
        assert!(!result.v2_features_enabled);

        // Step 2: Add training data
        let features = V2Features {
            error_fix_chains: 1,
            tool_normalize_chains: 0,
            rollback_occurred: false,
            edit_depth: 2,
            change_entropy: 1.2,
            code_error_ratio: 0.67,
            late_head_edits: 0,
            kv_prefix_impact: 0.1,
        };

        service
            .add_training_datapoint(features, 0.8, ScenarioType::Code)
            .await;

        // Step 3: Record performance metrics
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

        // Step 4: Check metrics
        let metrics = service.get_metrics().await;
        assert!(metrics.contains_key("delta_u_training_data_count"));
        assert!(metrics.contains_key("controller_current_lambda"));
        assert!(metrics.contains_key("ab_test_variant"));

        // Should have training data count of 1
        assert_eq!(
            metrics
                .get("delta_u_training_data_count")
                .copied()
                .unwrap_or(0.0),
            1.0
        );

        // Step 5: Get A/B test results
        let ab_results = service.get_ab_test_results().await;
        assert_eq!(ab_results.current_variant, "control");
        assert!(ab_results.variant_results.contains_key("control_accuracy"));
    }

    #[tokio::test]
    async fn test_feature_extraction_integration() {
        let service = LearningLoopService::new(None);

        // Create a complex sequence of changes
        let changes = vec![
            create_test_change(ChangeType::Error, 3),
            create_test_change(ChangeType::Fix, 2),
            create_test_change(ChangeType::Tool, 1),
            create_test_change(ChangeType::Normalize, 1),
            create_test_change(ChangeType::Rollback, 4),
            create_test_change(ChangeType::HeadSummary, 1),
        ];

        // Process with code scenario
        let result = service.process_changes(changes, ScenarioType::Code).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.computation_time_ms > 0.0);
    }

    #[tokio::test]
    async fn test_controller_parameter_adjustment() {
        use crate::lambda_mu_controller::{ControllerConfig, LambdaMuControllerImpl};

        let config = ControllerConfig {
            target_p95_latency_ms: 100.0,
            ..Default::default()
        };

        let mut controller = LambdaMuControllerImpl::new(Some(config));

        // Test with high difficulty changes
        let high_difficulty_changes = vec![
            create_test_change(ChangeType::Error, 5),
            create_test_change(ChangeType::Rollback, 3),
            create_test_change(ChangeType::HeadSummary, 2),
        ];

        // Simulate poor performance
        let (lambda, mu) =
            controller.update_with_v2_features(&high_difficulty_changes, 150.0, 600.0);

        let state = controller.get_state();
        assert!(state.difficulty_score > 0.0);
        assert!(state.target_k2 >= 2048); // Should increase due to difficulty

        // Test with good performance - should grow more conservatively
        let (_lambda2, mu2) =
            controller.update_with_v2_features(&high_difficulty_changes, 80.0, 400.0);

        // μ should adjust based on performance
        assert_ne!(mu, mu2);
    }

    #[tokio::test]
    async fn test_training_system_integration() {
        use crate::delta_u_training::DeltaUTrainer;
        use crate::types::{TrainingDatapoint, V2Features};

        let mut trainer = DeltaUTrainer::new();

        // Add several training datapoints
        let features1 = V2Features {
            error_fix_chains: 2,
            tool_normalize_chains: 1,
            rollback_occurred: false,
            edit_depth: 3,
            change_entropy: 1.5,
            code_error_ratio: 0.8,
            late_head_edits: 0,
            kv_prefix_impact: 0.1,
        };

        let features2 = V2Features {
            error_fix_chains: 0,
            tool_normalize_chains: 0,
            rollback_occurred: true,
            edit_depth: 1,
            change_entropy: 0.5,
            code_error_ratio: 0.2,
            late_head_edits: 2,
            kv_prefix_impact: 0.4,
        };

        trainer.add_training_data(TrainingDatapoint {
            features: features1.clone(),
            ground_truth_utility: 0.9,
            timestamp: Utc::now(),
            scenario_type: ScenarioType::Code,
        });

        trainer.add_training_data(TrainingDatapoint {
            features: features2.clone(),
            ground_truth_utility: -0.3,
            timestamp: Utc::now(),
            scenario_type: ScenarioType::Code,
        });

        // Train the model
        assert!(trainer.train().is_ok());

        // Make predictions
        let prediction1 = trainer.predict(&features1, ScenarioType::Code);
        let prediction2 = trainer.predict(&features2, ScenarioType::Code);

        assert!(prediction1.is_some());
        assert!(prediction2.is_some());

        let pred1 = prediction1.unwrap();
        let pred2 = prediction2.unwrap();

        // Good features should predict higher utility than bad features
        assert!(pred1.predicted_utility_change > pred2.predicted_utility_change);

        // Both should have reasonable confidence
        assert!(pred1.confidence > 0.0 && pred1.confidence <= 1.0);
        assert!(pred2.confidence > 0.0 && pred2.confidence <= 1.0);
    }

    #[tokio::test]
    async fn test_performance_monitoring() {
        let service = LearningLoopService::new(None);

        // Record several performance metrics over time
        for i in 0..10 {
            let metric = PerformanceMetric {
                timestamp: Utc::now(),
                p95_latency_ms: 90.0 + i as f64,
                throughput_ops_per_sec: 1000.0 + i as f64 * 10.0,
                memory_usage_mb: 400.0 + i as f64 * 5.0,
                accuracy_score: 0.8 + i as f64 * 0.01,
                lambda: 0.12,
                mu: 1.0 + i as f64 * 0.01,
                context_size: 2048,
            };

            service.record_performance(metric).await;
        }

        let metrics = service.get_metrics().await;
        assert!(metrics.contains_key("performance_history_count"));
        assert_eq!(
            metrics
                .get("performance_history_count")
                .copied()
                .unwrap_or(0.0),
            10.0
        );

        // Latest metrics should be available
        assert!(metrics.contains_key("latest_p95_latency_ms"));
        assert!(metrics.contains_key("latest_accuracy"));
        assert!(metrics.contains_key("latest_memory_mb"));

        let latest_latency = metrics.get("latest_p95_latency_ms").copied().unwrap_or(0.0);
        assert!(latest_latency > 90.0); // Should be the last recorded value
    }

    #[tokio::test]
    async fn test_ab_testing_framework() {
        let service = LearningLoopService::new(None);

        // Simulate processing changes with multiple variants
        let changes = vec![create_test_change(ChangeType::Code, 1)];

        // Process changes multiple times to generate A/B test data
        for _ in 0..5 {
            let result = service
                .process_changes(changes.clone(), ScenarioType::Code)
                .await;
            assert!(result.is_ok());
        }

        let ab_results = service.get_ab_test_results().await;

        // Should have recorded metrics for the control variant
        assert!(!ab_results.variant_results.is_empty());

        // Should have accuracy and compute time metrics
        let control_accuracy_key = format!("{}_accuracy", ab_results.current_variant);
        let control_compute_key = format!("{}_compute_time", ab_results.current_variant);

        assert!(ab_results
            .variant_results
            .contains_key(&control_accuracy_key));
        assert!(ab_results
            .variant_results
            .contains_key(&control_compute_key));

        if let Some(accuracy_result) = ab_results.variant_results.get(&control_accuracy_key) {
            assert_eq!(accuracy_result.count, 5); // Should have 5 datapoints
            assert!(accuracy_result.mean > 0.0);
            assert!(accuracy_result.std_dev >= 0.0);
        }
    }
}
