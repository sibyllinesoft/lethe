use chrono::Utc;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use determinism_service::{
    config::Config, determinism::DeterminismSentinel, json_canon::CanonicalJson,
    learning_loop::LearningLoopService, performance::PerformanceBudgetEnforcer,
    testing::ClockSkewTester, types::*, v2_features::V2FeatureExtractor,
};
use rand::{
    distributions::{Alphanumeric, Uniform},
    thread_rng, Rng,
};
use std::{
    collections::HashMap,
    mem,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::runtime::Runtime;
use uuid::Uuid;

/// Comprehensive V2 Benchmark Matrix Configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub rust_refactor_enabled: bool,
    pub transform_v2_enabled: bool,
    pub scenarios: Vec<ScenarioType>,
    pub adversarial_buckets: Vec<AdversarialBucket>,
    pub performance_gates: PerformanceGates,
}

#[derive(Debug, Clone)]
pub struct PerformanceGates {
    pub max_ece_score: f64,              // ≤ 0.08
    pub p95_latency_baseline: f64,       // Must maintain ≥ baseline average
    pub max_p99_p95_ratio: f64,          // ≤ 2.5
    pub max_proxy_gap: f64,              // ≤ 0.5%
    pub pool_fingerprint_tolerance: f64, // For deterministic outputs
}

#[derive(Debug, Clone)]
pub enum AdversarialBucket {
    Pathological,       // Extremely long sequences, deep nesting, malformed data
    ResourceExhaustion, // Memory pressure, CPU saturation, disk I/O limits
    TimingAttacks,      // Clock skew, race conditions, timeout exploitation
    DataCorruption,     // Partial loss, bit flips, encoding issues
    Byzantine,          // Malicious inputs, coordinated attacks, split-brain conditions
}

#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub overall_status: PassFail,
    pub paired_results: PairedComparisonData,
    pub adversarial_results: AdversarialTestData,
    pub performance_gates: PerformanceGateStatus,
    pub v2_impact_metrics: V2FeatureImpactData,
    pub recommendations: Vec<ActionableRecommendation>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PassFail {
    Pass,
    Fail { reason: String },
}

impl BenchmarkConfig {
    pub fn production_ready() -> Self {
        Self {
            rust_refactor_enabled: true,
            transform_v2_enabled: true,
            scenarios: vec![
                ScenarioType::Code,
                ScenarioType::Prose,
                ScenarioType::ToolResults,
                ScenarioType::Mixed,
            ],
            adversarial_buckets: vec![
                AdversarialBucket::Pathological,
                AdversarialBucket::ResourceExhaustion,
                AdversarialBucket::TimingAttacks,
                AdversarialBucket::DataCorruption,
                AdversarialBucket::Byzantine,
            ],
            performance_gates: PerformanceGates {
                max_ece_score: 0.08,
                p95_latency_baseline: 100.0, // ms
                max_p99_p95_ratio: 2.5,
                max_proxy_gap: 0.5,
                pool_fingerprint_tolerance: 0.001,
            },
        }
    }
}

/// Core Paired Benchmark Matrix
pub fn bench_paired_matrix_v2_enabled(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = BenchmarkConfig::production_ready();

    for scenario in &config.scenarios {
        let mut group = c.benchmark_group(&format!("paired_matrix_{:?}", scenario));
        group.throughput(Throughput::Elements(1));

        // Rust refactor + V2 enabled configuration
        let app_config = Arc::new(Config::default());
        let sentinel = rt.block_on(async { DeterminismSentinel::new(app_config).await.unwrap() });

        let learning_loop = Arc::new(LearningLoopService::new(Some(
            ChannelConfig::with_v2_features(),
        )));

        let mut feature_extractor = V2FeatureExtractor::new();

        group.bench_function(
            BenchmarkId::new("v2_transform_processing", format!("{:?}", scenario)),
            |b| {
                b.to_async(&rt).iter(|| async {
                    // Generate test scenario data
                    let changes = generate_scenario_changes(&scenario, 50);
                    let features = feature_extractor.extract_features(&changes);

                    // Process through V2 learning loop
                    let result = learning_loop
                        .process_changes(changes, scenario.clone())
                        .await;

                    // Validate determinism with V2 enabled
                    let slice_id = format!("v2_test_{}", Uuid::new_v4());
                    let determinism_result = sentinel.replay_slice_twice(&slice_id).await;

                    black_box((features, result, determinism_result))
                });
            },
        );

        group.finish();
    }
}

/// Adversarial Test Buckets Implementation
pub fn bench_adversarial_buckets(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = BenchmarkConfig::production_ready();

    for bucket in &config.adversarial_buckets {
        let mut group = c.benchmark_group(&format!("adversarial_{:?}", bucket));

        let app_config = Arc::new(Config::default());
        let sentinel = rt.block_on(async { DeterminismSentinel::new(app_config).await.unwrap() });

        match bucket {
            AdversarialBucket::Pathological => {
                bench_pathological_inputs(&mut group, &rt, &sentinel);
            }
            AdversarialBucket::ResourceExhaustion => {
                bench_resource_exhaustion(&mut group, &rt, &sentinel);
            }
            AdversarialBucket::TimingAttacks => {
                bench_timing_attacks(&mut group, &rt, &sentinel);
            }
            AdversarialBucket::DataCorruption => {
                bench_data_corruption(&mut group, &rt, &sentinel);
            }
            AdversarialBucket::Byzantine => {
                bench_byzantine_scenarios(&mut group, &rt, &sentinel);
            }
        }

        group.finish();
    }
}

fn bench_pathological_inputs(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    rt: &Runtime,
    sentinel: &DeterminismSentinel,
) {
    // Extremely long sequences
    group.bench_function("extremely_long_sequences", |b| {
        b.to_async(rt).iter(|| async {
            let mut changes = Vec::with_capacity(10000);
            for i in 0..10000 {
                changes.push(create_pathological_change(i));
            }

            let result = sentinel.replay_slice_twice("pathological_long").await;
            black_box(result)
        });
    });

    // Deep nesting structures
    group.bench_function("deep_nesting", |b| {
        let deep_json = create_deeply_nested_json(100);
        let canonicalizer = CanonicalJson::new();

        b.iter(|| {
            let result = canonicalizer.serialize(&deep_json);
            black_box(result)
        });
    });

    // Malformed data handling
    group.bench_function("malformed_data", |b| {
        let malformed_inputs = generate_malformed_inputs(1000);

        b.to_async(rt).iter(|| async {
            let mut successful_parses = 0;
            for input in &malformed_inputs {
                if let Ok(_) = serde_json::from_str::<serde_json::Value>(input) {
                    successful_parses += 1;
                }
            }
            black_box(successful_parses)
        });
    });
}

fn bench_resource_exhaustion(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    rt: &Runtime,
    sentinel: &DeterminismSentinel,
) {
    // Memory pressure simulation
    group.bench_function("memory_pressure", |b| {
        b.to_async(rt).iter(|| async {
            // Allocate large amounts of memory to stress the system
            let mut large_vectors: Vec<Vec<u8>> = Vec::new();
            for _ in 0..100 {
                large_vectors.push(vec![0u8; 1024 * 1024]); // 1MB each
            }

            let result = sentinel.replay_slice_twice("memory_stress").await;

            // Clean up immediately
            drop(large_vectors);

            black_box(result)
        });
    });

    // CPU saturation
    group.bench_function("cpu_saturation", |b| {
        b.to_async(rt).iter(|| async {
            // Spawn multiple CPU-intensive tasks
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    tokio::spawn(async move {
                        let mut sum = 0u64;
                        for i in 0..1000000 {
                            sum = sum.wrapping_add(i);
                        }
                        sum
                    })
                })
                .collect();

            // Wait for CPU tasks while testing determinism
            let cpu_results = futures::future::join_all(handles);
            let determinism_result = sentinel.replay_slice_twice("cpu_stress");

            let (cpu_res, det_res) = tokio::join!(cpu_results, determinism_result);
            black_box((cpu_res, det_res))
        });
    });
}

fn bench_timing_attacks(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    rt: &Runtime,
    sentinel: &DeterminismSentinel,
) {
    // Clock skew simulation
    group.bench_function("clock_skew_attack", |b| {
        let clock_tester = ClockSkewTester::new(Duration::from_micros(100));

        b.to_async(rt).iter(|| async {
            // Simulate varying clock speeds
            tokio::time::sleep(Duration::from_nanos(thread_rng().gen_range(100..1000))).await;
            let result = clock_tester.validate_monotone_timestamps(100).await;
            black_box(result)
        });
    });

    // Race condition exploitation
    group.bench_function("race_conditions", |b| {
        b.to_async(rt).iter(|| async {
            // Concurrent access pattern designed to expose races
            let shared_sentinel = Arc::new(sentinel);
            let handles: Vec<_> = (0..10)
                .map(|i| {
                    let sentinel = shared_sentinel.clone();
                    tokio::spawn(async move {
                        sentinel.replay_slice_twice(&format!("race_{}", i)).await
                    })
                })
                .collect();

            let results = futures::future::join_all(handles).await;
            black_box(results)
        });
    });
}

fn bench_data_corruption(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    rt: &Runtime,
    _sentinel: &DeterminismSentinel,
) {
    // Bit flip simulation
    group.bench_function("bit_flips", |b| {
        let canonicalizer = CanonicalJson::new();

        b.iter(|| {
            let test_data = serde_json::json!({
                "test": "original data that will be corrupted",
                "numbers": vec![1, 2, 3, 4, 5]
            });

            // Simulate bit corruption in serialized data
            let mut serialized = canonicalizer.serialize(&test_data).unwrap();
            if !serialized.is_empty() {
                let corruption_idx = thread_rng().gen_range(0..serialized.len());
                unsafe {
                    let byte_ptr = serialized.as_mut_ptr().add(corruption_idx);
                    *byte_ptr ^= 1; // Flip a single bit
                }
            }

            // Attempt to deserialize corrupted data
            let result = serde_json::from_str::<serde_json::Value>(&serialized);
            black_box(result.is_ok())
        });
    });

    // Partial data loss
    group.bench_function("partial_loss", |b| {
        b.iter(|| {
            let original = "This is a test string that will be partially corrupted";
            let mut corrupted = original.to_string();

            // Remove random chunks
            let chunk_size = thread_rng().gen_range(1..10);
            let start_pos = thread_rng().gen_range(0..corrupted.len().saturating_sub(chunk_size));
            corrupted.drain(start_pos..start_pos + chunk_size);

            black_box(corrupted.len())
        });
    });
}

fn bench_byzantine_scenarios(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    rt: &Runtime,
    sentinel: &DeterminismSentinel,
) {
    // Malicious input patterns
    group.bench_function("malicious_inputs", |b| {
        let malicious_payloads = generate_malicious_payloads();

        b.to_async(rt).iter(|| async {
            let mut processed = 0;
            for payload in &malicious_payloads {
                // Test system resilience against malicious data
                if let Ok(_) = sentinel.replay_slice_twice(&payload.slice_id).await {
                    processed += 1;
                }
            }
            black_box(processed)
        });
    });

    // Coordinated attack simulation
    group.bench_function("coordinated_attacks", |b| {
        b.to_async(rt).iter(|| async {
            // Simulate multiple coordinated malicious requests
            let attack_patterns = vec![
                AttackPattern::ResourceDrain,
                AttackPattern::TimingExploitation,
                AttackPattern::StateCorruption,
            ];

            let mut results = Vec::new();
            for pattern in attack_patterns {
                let attack_result = execute_attack_pattern(pattern, sentinel).await;
                results.push(attack_result);
            }

            black_box(results)
        });
    });
}

/// Performance Gate Validation
pub fn bench_performance_gates(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let config = BenchmarkConfig::production_ready();

    let app_config = Arc::new(Config::default());
    let sentinel = rt.block_on(async { DeterminismSentinel::new(app_config).await.unwrap() });

    let performance_enforcer =
        PerformanceBudgetEnforcer::new(config.performance_gates.p95_latency_baseline / 100.0);

    c.bench_function("ece_score_validation", |b| {
        b.to_async(&rt).iter(|| async {
            // Generate calibration dataset for ECE calculation
            let predictions = generate_calibration_predictions(1000);
            let ece_score = calculate_ece(&predictions);

            let gate_passed = ece_score <= config.performance_gates.max_ece_score;
            black_box((ece_score, gate_passed))
        });
    });

    c.bench_function("latency_percentile_validation", |b| {
        b.to_async(&rt).iter(|| async {
            let mut latencies = Vec::new();

            // Collect 100 latency measurements
            for i in 0..100 {
                let start = Instant::now();
                let _ = sentinel
                    .replay_slice_twice(&format!("latency_test_{}", i))
                    .await;
                let elapsed = start.elapsed().as_millis() as f64;
                latencies.push(elapsed);
            }

            latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p95_idx = (latencies.len() as f64 * 0.95) as usize;
            let p99_idx = (latencies.len() as f64 * 0.99) as usize;

            let p95_latency = latencies[p95_idx.min(latencies.len() - 1)];
            let p99_latency = latencies[p99_idx.min(latencies.len() - 1)];

            let p99_p95_ratio = p99_latency / p95_latency;
            let p95_gate_passed = p95_latency >= config.performance_gates.p95_latency_baseline;
            let ratio_gate_passed = p99_p95_ratio <= config.performance_gates.max_p99_p95_ratio;

            black_box((
                p95_latency,
                p99_p95_ratio,
                p95_gate_passed,
                ratio_gate_passed,
            ))
        });
    });
}

/// V2 Feature Impact Analysis
pub fn bench_v2_feature_impact(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("v2_change_tracking_overhead", |b| {
        let mut extractor = V2FeatureExtractor::new();

        b.iter(|| {
            let changes = generate_scenario_changes(&ScenarioType::Mixed, 100);
            let features = extractor.extract_features(&changes);
            black_box(features)
        });
    });

    c.bench_function("v2_learning_loop_integration", |b| {
        let learning_loop = Arc::new(LearningLoopService::new(Some(
            ChannelConfig::with_v2_features(),
        )));

        b.to_async(&rt).iter(|| async {
            let changes = generate_scenario_changes(&ScenarioType::Code, 50);
            let result = learning_loop
                .process_changes(changes, ScenarioType::Code)
                .await;
            black_box(result)
        });
    });
}

// Helper functions and data generation

fn generate_scenario_changes(scenario: &ScenarioType, count: usize) -> Vec<TransformChangeV2> {
    let mut changes = Vec::with_capacity(count);
    let mut rng = thread_rng();

    for i in 0..count {
        let change_type = match scenario {
            ScenarioType::Code => {
                if rng.gen_bool(0.7) {
                    ChangeType::Code
                } else {
                    ChangeType::Fix
                }
            }
            ScenarioType::Prose => {
                if rng.gen_bool(0.8) {
                    ChangeType::HeadSummary
                } else {
                    ChangeType::KvUpdate
                }
            }
            ScenarioType::ToolResults => {
                if rng.gen_bool(0.6) {
                    ChangeType::Tool
                } else {
                    ChangeType::Normalize
                }
            }
            ScenarioType::Mixed => match rng.gen_range(0..6) {
                0 => ChangeType::Code,
                1 => ChangeType::Error,
                2 => ChangeType::Fix,
                3 => ChangeType::Tool,
                4 => ChangeType::HeadSummary,
                _ => ChangeType::KvUpdate,
            },
        };

        changes.push(TransformChangeV2 {
            change_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            change_type,
            metadata: ChangeMetadata {
                depth: i as u32 % 10,
                complexity_score: rng.gen_range(0.1..2.0),
                edit_distance: Some(rng.gen_range(1..100)),
                context_size: rng.gen_range(100..1000),
                causality_chain: vec![],
            },
            before_state: None,
            after_state: None,
            performance_impact: None,
        });
    }

    changes
}

fn create_pathological_change(index: usize) -> TransformChangeV2 {
    TransformChangeV2 {
        change_id: Uuid::new_v4(),
        timestamp: Utc::now(),
        change_type: ChangeType::Other(format!("pathological_{}", index)),
        metadata: ChangeMetadata {
            depth: (index % 1000) as u32,
            complexity_score: index as f64,
            edit_distance: Some((index * 17) as u32),
            context_size: (index * 23 + 12345) as u32,
            causality_chain: (0..index % 10).map(|_| Uuid::new_v4()).collect(),
        },
        before_state: Some(serde_json::json!({
            "massive_data": vec![index; index % 100 + 1]
        })),
        after_state: Some(serde_json::json!({
            "even_more_massive_data": vec![index * 2; index % 200 + 1]
        })),
        performance_impact: Some(PerformanceImpact {
            latency_delta_ms: index as f64 * 0.001,
            memory_delta_mb: index as f64 * 0.1,
            throughput_delta_percent: -(index as f64 * 0.01),
        }),
    }
}

fn create_deeply_nested_json(depth: usize) -> serde_json::Value {
    if depth == 0 {
        serde_json::json!("leaf")
    } else {
        serde_json::json!({
            "nested": create_deeply_nested_json(depth - 1),
            "data": format!("level_{}", depth),
            "array": vec![1, 2, 3, 4, 5]
        })
    }
}

fn generate_malformed_inputs(count: usize) -> Vec<String> {
    let mut inputs = Vec::with_capacity(count);
    let mut rng = thread_rng();

    for _ in 0..count {
        let input = match rng.gen_range(0..4) {
            0 => "{\"incomplete_json\":".to_string(),
            1 => "\"unterminated_string".to_string(),
            2 => "[1, 2, 3,]".to_string(), // Trailing comma
            _ => "invalid_json_completely".to_string(),
        };
        inputs.push(input);
    }

    inputs
}

#[derive(Clone)]
struct MaliciousPayload {
    slice_id: String,
    attack_vector: String,
}

fn generate_malicious_payloads() -> Vec<MaliciousPayload> {
    vec![
        MaliciousPayload {
            slice_id: "../../../etc/passwd".to_string(),
            attack_vector: "path_traversal".to_string(),
        },
        MaliciousPayload {
            slice_id: "'; DROP TABLE users; --".to_string(),
            attack_vector: "sql_injection".to_string(),
        },
        MaliciousPayload {
            slice_id: "<script>alert('xss')</script>".to_string(),
            attack_vector: "xss".to_string(),
        },
        MaliciousPayload {
            slice_id: "A".repeat(10000),
            attack_vector: "buffer_overflow".to_string(),
        },
    ]
}

#[derive(Clone)]
enum AttackPattern {
    ResourceDrain,
    TimingExploitation,
    StateCorruption,
}

async fn execute_attack_pattern(pattern: AttackPattern, sentinel: &DeterminismSentinel) -> bool {
    match pattern {
        AttackPattern::ResourceDrain => {
            // Simulate resource draining attack
            let mut handles = Vec::new();
            for i in 0..50 {
                let sentinel = sentinel.clone();
                handles.push(tokio::spawn(async move {
                    sentinel.replay_slice_twice(&format!("drain_{}", i)).await
                }));
            }

            // Check if system remains responsive
            futures::future::join_all(handles)
                .await
                .into_iter()
                .all(|result| result.is_ok())
        }
        AttackPattern::TimingExploitation => {
            // Simulate timing-based attack
            let start = Instant::now();
            let _result = sentinel.replay_slice_twice("timing_attack").await;
            let elapsed = start.elapsed();

            // Check for suspicious timing patterns
            elapsed.as_millis() < 10000 // Should complete within reasonable time
        }
        AttackPattern::StateCorruption => {
            // Simulate state corruption attempt
            let result1 = sentinel.replay_slice_twice("state_test_1").await;
            let result2 = sentinel.replay_slice_twice("state_test_2").await;

            // Verify state integrity maintained
            result1.is_ok() && result2.is_ok()
        }
    }
}

#[derive(Clone)]
struct CalibrationPrediction {
    confidence: f64,
    correct: bool,
}

fn generate_calibration_predictions(count: usize) -> Vec<CalibrationPrediction> {
    let mut rng = thread_rng();
    let mut predictions = Vec::with_capacity(count);

    for _ in 0..count {
        let confidence = rng.gen_range(0.0..1.0);
        // Simulate realistic calibration where higher confidence correlates with correctness
        let correct = rng.gen_bool(confidence * 0.8 + 0.1);
        predictions.push(CalibrationPrediction {
            confidence,
            correct,
        });
    }

    predictions
}

fn calculate_ece(predictions: &[CalibrationPrediction]) -> f64 {
    let num_bins = 10;
    let mut bins: Vec<Vec<&CalibrationPrediction>> = vec![Vec::new(); num_bins];

    // Assign predictions to confidence bins
    for pred in predictions {
        let bin_idx = ((pred.confidence * num_bins as f64) as usize).min(num_bins - 1);
        bins[bin_idx].push(pred);
    }

    // Calculate ECE
    let mut ece = 0.0;
    let total_predictions = predictions.len() as f64;

    for (bin_idx, bin) in bins.iter().enumerate() {
        if bin.is_empty() {
            continue;
        }

        let bin_confidence = (bin_idx as f64 + 0.5) / num_bins as f64;
        let bin_accuracy = bin.iter().filter(|pred| pred.correct).count() as f64 / bin.len() as f64;
        let bin_weight = bin.len() as f64 / total_predictions;

        ece += bin_weight * (bin_confidence - bin_accuracy).abs();
    }

    ece
}

// Placeholder types for compilation
#[derive(Clone)]
struct ChannelConfig;

impl ChannelConfig {
    fn with_v2_features() -> Self {
        Self
    }
}

#[derive(Debug, Clone)]
pub struct PairedComparisonData {
    pub scenarios_tested: usize,
    pub determinism_success_rate: f64,
    pub budget_compliance_rate: f64,
    pub closure_rate: f64,
}

#[derive(Debug, Clone)]
pub struct AdversarialTestData {
    pub buckets_tested: usize,
    pub resilience_score: f64,
    pub attack_mitigation_rate: f64,
}

#[derive(Debug, Clone)]
pub struct PerformanceGateStatus {
    pub ece_gate_passed: bool,
    pub p95_latency_gate_passed: bool,
    pub p99_p95_ratio_gate_passed: bool,
    pub proxy_gap_gate_passed: bool,
    pub pool_fingerprint_gate_passed: bool,
}

#[derive(Debug, Clone)]
pub struct V2FeatureImpactData {
    pub feature_extraction_overhead_ms: f64,
    pub learning_integration_overhead_ms: f64,
    pub accuracy_improvement_percent: f64,
}

#[derive(Debug, Clone)]
pub struct ActionableRecommendation {
    pub category: String,
    pub priority: String,
    pub description: String,
    pub estimated_impact: f64,
}

criterion_group!(
    benches,
    bench_paired_matrix_v2_enabled,
    bench_adversarial_buckets,
    bench_performance_gates,
    bench_v2_feature_impact
);
criterion_main!(benches);
