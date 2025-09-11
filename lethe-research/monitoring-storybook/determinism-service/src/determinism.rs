use crate::{
    config::Config,
    json_canon::CanonicalJson,
    performance::PerformanceBudgetEnforcer,
    testing::ClockSkewTester,
    types::*,
};
use chrono::Utc;
use dashmap::DashMap;
use futures::future::join_all;
use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::{sync::RwLock, time::interval};
use tracing::{error, info, warn};
use uuid::Uuid;

pub struct DeterminismSentinel {
    config: Arc<Config>,
    canonical_json: Arc<CanonicalJson>,
    performance_enforcer: Arc<PerformanceBudgetEnforcer>,
    clock_skew_tester: Arc<ClockSkewTester>,
    metrics: Arc<SentinelMetrics>,
    active_replays: Arc<DashMap<String, Instant>>,
    replay_results: Arc<RwLock<Vec<DeterminismReport>>>,
}

pub struct SentinelMetrics {
    pub total_replays: AtomicU64,
    pub successful_replays: AtomicU64,
    pub failed_replays: AtomicU64,
    pub performance_budget_violations: AtomicU64,
    pub invariant_violations: AtomicU64,
    pub clock_skew_violations: AtomicU64,
    pub uptime_start: std::time::Instant,
}

impl Default for SentinelMetrics {
    fn default() -> Self {
        Self {
            total_replays: AtomicU64::new(0),
            successful_replays: AtomicU64::new(0),
            failed_replays: AtomicU64::new(0),
            performance_budget_violations: AtomicU64::new(0),
            invariant_violations: AtomicU64::new(0),
            clock_skew_violations: AtomicU64::new(0),
            uptime_start: std::time::Instant::now(),
        }
    }
}

impl DeterminismSentinel {
    pub async fn new(config: Arc<Config>) -> Result<Self, ValidationError> {
        let canonical_json = Arc::new(CanonicalJson::new());
        let performance_enforcer = Arc::new(PerformanceBudgetEnforcer::new(
            config.determinism.performance_budget_percent,
        ));
        let clock_skew_tester = Arc::new(ClockSkewTester::new(
            Duration::from_millis(config.determinism.tolerance_ms),
        ));

        Ok(Self {
            config,
            canonical_json,
            performance_enforcer,
            clock_skew_tester,
            metrics: Arc::new(SentinelMetrics::default()),
            active_replays: Arc::new(DashMap::new()),
            replay_results: Arc::new(RwLock::new(Vec::new())),
        })
    }

    pub async fn replay_slice_twice(&self, slice_id: &str) -> Result<DeterminismReport, ValidationError> {
        let run_id = Uuid::new_v4();
        let start_time = Instant::now();
        
        info!("Starting determinism replay for slice: {}", slice_id);
        self.active_replays.insert(slice_id.to_string(), start_time);
        
        // Increment total replays counter
        self.metrics.total_replays.fetch_add(1, Ordering::SeqCst);

        // Run the slice twice in parallel for better performance
        let (result1, result2) = tokio::join!(
            self.execute_slice(slice_id),
            self.execute_slice(slice_id)
        );

        let run1 = result1?;
        let run2 = result2?;

        // Perform determinism validation
        let determinism_check = self.assert_identity(&run1, &run2)?;
        
        // Validate performance budget
        let performance_check = self.performance_enforcer
            .validate_budget(&run1, &run2)
            .await?;

        // Validate invariants
        let invariant_report = self.validate_invariants(&[run1.clone(), run2.clone()]);

        let report = DeterminismReport {
            slice_id: slice_id.to_string(),
            run_id,
            timestamp: Utc::now(),
            run1,
            run2,
            determinism_check,
            performance_budget_check: performance_check,
            invariant_report,
        };

        // Update metrics based on results
        if report.determinism_check.is_deterministic {
            self.metrics.successful_replays.fetch_add(1, Ordering::SeqCst);
        } else {
            self.metrics.failed_replays.fetch_add(1, Ordering::SeqCst);
        }

        if !report.performance_budget_check.budget_met {
            self.metrics.performance_budget_violations.fetch_add(1, Ordering::SeqCst);
        }

        if !report.invariant_report.all_passed {
            self.metrics.invariant_violations.fetch_add(1, Ordering::SeqCst);
        }

        // Store the result
        {
            let mut results = self.replay_results.write().await;
            results.push(report.clone());
            
            // Keep only the last 1000 results
            if results.len() > 1000 {
                let len = results.len();
                results.drain(0..len - 1000);
            }
        }

        self.active_replays.remove(slice_id);
        
        info!(
            "Completed determinism replay for slice: {} in {:?}ms", 
            slice_id, 
            start_time.elapsed().as_millis()
        );

        Ok(report)
    }

    async fn execute_slice(&self, slice_id: &str) -> Result<ProcessingResult, ValidationError> {
        let start = Instant::now();
        
        // Simulate slice processing (in real implementation, this would call actual processing logic)
        let processing_result = self.simulate_slice_processing(slice_id).await?;
        
        let duration = start.elapsed();
        
        // Create canonical hash of the result
        let canonical_data = self.canonical_json.serialize(&processing_result)?;
        let result_hash = self.canonical_json.hash(&canonical_data);

        let performance_metrics = PerformanceMetrics {
            duration_ms: duration.as_millis() as u64,
            memory_usage_mb: self.get_memory_usage(),
            cpu_usage_percent: self.get_cpu_usage(),
            p95_latency_ms: duration.as_secs_f64() * 1000.0,
            throughput_ops_per_sec: 1000.0 / (duration.as_millis() as f64),
        };

        let invariants = InvariantChecks {
            monotone_timestamps: self.check_monotone_timestamps(&processing_result),
            causal_ordering: self.check_causal_ordering(&processing_result),
            data_consistency: self.check_data_consistency(&processing_result),
            structural_integrity: self.check_structural_integrity(&processing_result),
        };

        Ok(ProcessingResult {
            slice_id: slice_id.to_string(),
            timestamp: Utc::now(),
            result_hash,
            performance_metrics,
            invariants,
            metadata: processing_result,
        })
    }

    async fn simulate_slice_processing(&self, slice_id: &str) -> Result<std::collections::HashMap<String, serde_json::Value>, ValidationError> {
        use std::collections::HashMap;
        
        // In a real implementation, this would call the actual slice processing logic
        // For now, we simulate deterministic processing
        let mut result = HashMap::new();
        
        result.insert("slice_id".to_string(), serde_json::Value::String(slice_id.to_string()));
        result.insert("processed_at".to_string(), serde_json::Value::String(Utc::now().to_rfc3339()));
        result.insert("version".to_string(), serde_json::Value::String("1.0.0".to_string()));
        
        // Simulate some processing time
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        Ok(result)
    }

    pub fn assert_identity(&self, run1: &ProcessingResult, run2: &ProcessingResult) -> Result<DeterminismCheck, ValidationError> {
        let hash_match = run1.result_hash == run2.result_hash;
        
        // Calculate timestamp jitter
        let timestamp_diff = if run1.timestamp > run2.timestamp {
            run1.timestamp - run2.timestamp
        } else {
            run2.timestamp - run1.timestamp
        };
        let timestamp_jitter_ms = timestamp_diff.num_milliseconds().abs() as u64;
        
        let tolerance_met = timestamp_jitter_ms <= self.config.determinism.tolerance_ms;
        
        let mut differences = Vec::new();
        if !hash_match {
            differences.push("Result hashes do not match".to_string());
        }
        if !tolerance_met {
            differences.push(format!("Timestamp jitter {}ms exceeds tolerance {}ms", 
                timestamp_jitter_ms, self.config.determinism.tolerance_ms));
        }

        // Check performance metrics consistency
        let perf_diff = (run1.performance_metrics.duration_ms as f64 - run2.performance_metrics.duration_ms as f64).abs();
        if perf_diff > run1.performance_metrics.duration_ms as f64 * 0.1 {  // 10% tolerance
            differences.push(format!("Performance metrics differ by {:.2}ms", perf_diff));
        }

        let is_deterministic = hash_match && tolerance_met && differences.is_empty();

        Ok(DeterminismCheck {
            is_deterministic,
            hash_match,
            timestamp_jitter_ms,
            differences,
            tolerance_met,
        })
    }

    pub fn validate_invariants(&self, results: &[ProcessingResult]) -> InvariantReport {
        let mut violations = Vec::new();
        
        for result in results {
            if !result.invariants.monotone_timestamps {
                violations.push(InvariantViolation {
                    invariant_type: "monotone_timestamps".to_string(),
                    severity: ViolationSeverity::High,
                    description: "Timestamps are not monotonic".to_string(),
                    timestamp: Utc::now(),
                });
            }
            
            if !result.invariants.causal_ordering {
                violations.push(InvariantViolation {
                    invariant_type: "causal_ordering".to_string(),
                    severity: ViolationSeverity::Critical,
                    description: "Causal ordering violation detected".to_string(),
                    timestamp: Utc::now(),
                });
            }
            
            if !result.invariants.data_consistency {
                violations.push(InvariantViolation {
                    invariant_type: "data_consistency".to_string(),
                    severity: ViolationSeverity::High,
                    description: "Data consistency check failed".to_string(),
                    timestamp: Utc::now(),
                });
            }
            
            if !result.invariants.structural_integrity {
                violations.push(InvariantViolation {
                    invariant_type: "structural_integrity".to_string(),
                    severity: ViolationSeverity::Medium,
                    description: "Structural integrity check failed".to_string(),
                    timestamp: Utc::now(),
                });
            }
        }

        let all_passed = violations.is_empty();
        let score = if all_passed { 1.0 } else { 
            let critical_violations = violations.iter().filter(|v| matches!(v.severity, ViolationSeverity::Critical)).count();
            let high_violations = violations.iter().filter(|v| matches!(v.severity, ViolationSeverity::High)).count();
            let total_weight = critical_violations * 4 + high_violations * 2 + violations.len();
            1.0 - (total_weight as f64 / (results.len() * 10) as f64).min(1.0)
        };

        InvariantReport {
            all_passed,
            violations,
            score,
        }
    }

    pub async fn run_background_validation(&self) {
        let mut replay_interval = interval(Duration::from_secs(self.config.determinism.replay_interval_seconds));
        let mut clock_skew_interval = interval(Duration::from_secs(self.config.determinism.clock_skew_test_interval_seconds));

        loop {
            tokio::select! {
                _ = replay_interval.tick() => {
                    self.run_scheduled_replays().await;
                }
                _ = clock_skew_interval.tick() => {
                    self.run_clock_skew_tests().await;
                }
            }
        }
    }

    async fn run_scheduled_replays(&self) {
        info!("Running scheduled determinism replays");
        
        // Generate test slice IDs (in real implementation, these would come from actual data)
        let test_slices = vec!["test_slice_1", "test_slice_2", "test_slice_3"];
        
        let futures: Vec<_> = test_slices.into_iter()
            .map(|slice_id| self.replay_slice_twice(slice_id))
            .collect();

        let results = join_all(futures).await;
        
        let successful = results.iter().filter(|r| r.is_ok()).count();
        let failed = results.len() - successful;
        
        info!("Scheduled replays completed: {} successful, {} failed", successful, failed);
        
        for result in results {
            if let Err(e) = result {
                error!("Scheduled replay failed: {}", e);
            }
        }
    }

    async fn run_clock_skew_tests(&self) {
        info!("Running clock skew tolerance tests");
        
        match self.clock_skew_tester.run_skew_tests().await {
            Ok(results) => {
                info!("Clock skew tests completed: {} tests run", results.len());
                
                let violations = results.iter().filter(|r| !r.tolerance_met).count();
                if violations > 0 {
                    warn!("Clock skew violations detected: {}", violations);
                    self.metrics.clock_skew_violations.fetch_add(violations as u64, Ordering::SeqCst);
                }
            }
            Err(e) => {
                error!("Clock skew tests failed: {}", e);
            }
        }
    }

    pub async fn get_status(&self) -> SystemStatus {
        let total_replays = self.metrics.total_replays.load(Ordering::SeqCst);
        let successful_replays = self.metrics.successful_replays.load(Ordering::SeqCst);
        let performance_violations = self.metrics.performance_budget_violations.load(Ordering::SeqCst);

        let determinism_success_rate = if total_replays > 0 {
            successful_replays as f64 / total_replays as f64
        } else {
            1.0
        };

        let performance_budget_compliance = if total_replays > 0 {
            1.0 - (performance_violations as f64 / total_replays as f64)
        } else {
            1.0
        };

        let service_health = if determinism_success_rate >= 0.99 && performance_budget_compliance >= 0.98 {
            ServiceHealth::Healthy
        } else if determinism_success_rate >= 0.95 && performance_budget_compliance >= 0.90 {
            ServiceHealth::Degraded
        } else {
            ServiceHealth::Unhealthy
        };

        SystemStatus {
            service_health,
            determinism_success_rate,
            performance_budget_compliance,
            active_tests: self.active_replays.len() as u64,
            total_replays,
            last_check: Utc::now(),
        }
    }

    pub async fn get_metrics(&self) -> MetricsSnapshot {
        let total_replays = self.metrics.total_replays.load(Ordering::SeqCst);
        let successful_replays = self.metrics.successful_replays.load(Ordering::SeqCst);

        MetricsSnapshot {
            determinism_success_rate: if total_replays > 0 { 
                successful_replays as f64 / total_replays as f64 
            } else { 1.0 },
            avg_timestamp_jitter_ms: 0.5, // Would be calculated from actual data
            p95_performance_budget: 1.8,  // Would be calculated from actual data
            invariant_violations_per_hour: self.metrics.invariant_violations.load(Ordering::SeqCst) as f64,
            clock_skew_tolerance_ms: self.config.determinism.tolerance_ms as f64,
            total_tests_run: total_replays,
            uptime_seconds: self.metrics.uptime_start.elapsed().as_secs(),
        }
    }

    // Helper methods for invariant checks
    fn check_monotone_timestamps(&self, _result: &std::collections::HashMap<String, serde_json::Value>) -> bool {
        // In real implementation, check that timestamps are monotonically increasing
        true
    }

    fn check_causal_ordering(&self, _result: &std::collections::HashMap<String, serde_json::Value>) -> bool {
        // In real implementation, verify causal ordering constraints
        true
    }

    fn check_data_consistency(&self, _result: &std::collections::HashMap<String, serde_json::Value>) -> bool {
        // In real implementation, validate data consistency rules
        true
    }

    fn check_structural_integrity(&self, _result: &std::collections::HashMap<String, serde_json::Value>) -> bool {
        // In real implementation, check structural integrity of data
        true
    }

    fn get_memory_usage(&self) -> f64 {
        // In real implementation, get actual memory usage
        64.0 // MB
    }

    fn get_cpu_usage(&self) -> f64 {
        // In real implementation, get actual CPU usage
        15.0 // percent
    }
}