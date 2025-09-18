use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use determinism_service::{
    config::Config, determinism::DeterminismSentinel, json_canon::CanonicalJson,
    performance::PerformanceBudgetEnforcer, testing::ClockSkewTester,
};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::runtime::Runtime;

fn bench_canonical_json_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("canonical_json");

    let canonicalizer = CanonicalJson::new();

    // Test different data sizes
    for size in [10, 100, 1000, 10000].iter() {
        let mut data = HashMap::new();
        for i in 0..*size {
            data.insert(format!("key_{}", i), format!("value_{}", i));
        }

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("serialize", size), size, |b, _| {
            b.iter(|| {
                let _ = canonicalizer.serialize(&data);
            });
        });

        group.bench_with_input(BenchmarkId::new("hash", size), size, |b, _| {
            b.iter(|| {
                let _ = canonicalizer.hash_value(&data);
            });
        });
    }

    group.finish();
}

fn bench_determinism_replay(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let config = Arc::new(Config::default());
    let sentinel = rt.block_on(async { DeterminismSentinel::new(config).await.unwrap() });

    c.bench_function("determinism_replay_single", |b| {
        b.to_async(&rt).iter(|| async {
            let result = sentinel.replay_slice_twice("benchmark_slice").await;
            black_box(result)
        });
    });
}

fn bench_clock_skew_testing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let tester = ClockSkewTester::new(Duration::from_millis(1));

    c.bench_function("clock_skew_single_test", |b| {
        b.to_async(&rt).iter(|| async {
            let result = tester.validate_monotone_timestamps(100).await;
            black_box(result)
        });
    });
}

fn bench_performance_budget_enforcement(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let enforcer = PerformanceBudgetEnforcer::new(2.0);

    // Create test processing results
    let result1 = determinism_service::types::ProcessingResult {
        slice_id: "test".to_string(),
        timestamp: chrono::Utc::now(),
        result_hash: "hash1".to_string(),
        performance_metrics: determinism_service::types::PerformanceMetrics {
            duration_ms: 100,
            memory_usage_mb: 64.0,
            cpu_usage_percent: 15.0,
            p95_latency_ms: 1.5,
            throughput_ops_per_sec: 1000.0,
        },
        invariants: determinism_service::types::InvariantChecks {
            monotone_timestamps: true,
            causal_ordering: true,
            data_consistency: true,
            structural_integrity: true,
        },
        metadata: HashMap::new(),
    };

    let result2 = result1.clone();

    c.bench_function("performance_budget_validation", |b| {
        b.to_async(&rt).iter(|| async {
            let result = enforcer.validate_budget(&result1, &result2).await;
            black_box(result)
        });
    });
}

fn bench_json_canonicalization_edge_cases(c: &mut Criterion) {
    let canonicalizer = CanonicalJson::new();

    // Test with Unicode normalization
    let unicode_data = serde_json::json!({
        "café": "value1",      // NFC form
        "cafe\u{0301}": "value2",  // NFD form (should normalize to same as above)
        "test": "café"
    });

    c.bench_function("unicode_normalization", |b| {
        b.iter(|| {
            let _ = canonicalizer.serialize(&unicode_data);
        });
    });

    // Test with nested objects
    let nested_data = serde_json::json!({
        "level1": {
            "level2": {
                "level3": {
                    "data": vec![1, 2, 3, 4, 5],
                    "more_data": {
                        "x": 1.234567890123456789,
                        "y": null,
                        "z": true
                    }
                }
            }
        }
    });

    c.bench_function("nested_objects", |b| {
        b.iter(|| {
            let _ = canonicalizer.serialize(&nested_data);
        });
    });
}

fn bench_concurrent_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let config = Arc::new(Config::default());
    let sentinel = rt.block_on(async { Arc::new(DeterminismSentinel::new(config).await.unwrap()) });

    c.bench_function("concurrent_replays_10", |b| {
        b.to_async(&rt).iter(|| async {
            let sentinel = sentinel.clone();
            let handles: Vec<_> = (0..10)
                .map(|i| {
                    let sentinel = sentinel.clone();
                    tokio::spawn(async move {
                        sentinel
                            .replay_slice_twice(&format!("concurrent_slice_{}", i))
                            .await
                    })
                })
                .collect();

            for handle in handles {
                let _ = handle.await;
            }
        });
    });
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let canonicalizer = CanonicalJson::new();

    // Test memory efficiency with large arrays
    c.bench_function("large_array_canonicalization", |b| {
        let large_array: Vec<i32> = (0..10000).collect();
        b.iter(|| {
            let _ = canonicalizer.serialize(&large_array);
        });
    });

    // Test with repeated hash calculations
    c.bench_function("repeated_hash_calculations", |b| {
        let test_data = serde_json::json!({
            "repeated_data": "this is repeated data that should be hashed efficiently",
            "numbers": vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        });

        b.iter(|| {
            for _ in 0..100 {
                let _ = canonicalizer.hash_value(&test_data);
            }
        });
    });
}

criterion_group!(
    benches,
    bench_canonical_json_serialization,
    bench_determinism_replay,
    bench_clock_skew_testing,
    bench_performance_budget_enforcement,
    bench_json_canonicalization_edge_cases,
    bench_concurrent_operations,
    bench_memory_efficiency
);
criterion_main!(benches);
