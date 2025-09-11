use crate::types::*;
use rand::Rng;
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Performance budget enforcement with dynamic sampling
/// Ensures system performance stays within defined budgets with adaptive monitoring
pub struct PerformanceBudgetEnforcer {
    budget_threshold_percent: f64,  // p95 threshold (e.g., 2.0%)
    performance_history: Arc<RwLock<VecDeque<PerformanceReading>>>,
    current_sampling_rate: Arc<AtomicUsize>, // Stored as percentage * 100 for atomic ops
    circuit_breaker: Arc<CircuitBreaker>,
    adaptive_sampler: Arc<AdaptiveSampler>,
}

/// Tracks performance readings over time
#[derive(Debug, Clone)]
struct PerformanceReading {
    timestamp: std::time::Instant,
    duration_ms: f64,
    operation_type: OperationType,
    sampling_rate: f64,
}

#[derive(Debug, Clone)]
enum OperationType {
    StructuralEdit,
    Diagnostic,
    Replay,
    Validation,
}

/// Circuit breaker for performance protection
pub struct CircuitBreaker {
    state: Mutex<CircuitState>,
    failure_threshold: usize,
    success_threshold: usize,
    timeout: Duration,
    failure_count: AtomicU64,
    last_failure_time: Mutex<Option<Instant>>,
}

#[derive(Debug, Clone)]
enum CircuitState {
    Closed,    // Normal operation
    Open,      // Circuit open, requests rejected
    HalfOpen,  // Testing if service recovered
}

impl Default for CircuitState {
    fn default() -> Self {
        CircuitState::Closed
    }
}

/// Adaptive sampling controller
pub struct AdaptiveSampler {
    min_sampling_rate: f64,
    max_sampling_rate: f64,
    adjustment_factor: f64,
    performance_window: Duration,
    load_tracker: LoadTracker,
}

/// Tracks system load for adaptive sampling decisions
struct LoadTracker {
    cpu_samples: Mutex<VecDeque<f64>>,
    memory_samples: Mutex<VecDeque<f64>>,
    request_rate_samples: Mutex<VecDeque<f64>>,
    window_size: usize,
}

impl PerformanceBudgetEnforcer {
    pub fn new(budget_threshold_percent: f64) -> Self {
        Self {
            budget_threshold_percent,
            performance_history: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            current_sampling_rate: Arc::new(AtomicUsize::new(10000)), // 100.00% as integer
            circuit_breaker: Arc::new(CircuitBreaker::new(
                5,                               // failure threshold
                3,                               // success threshold
                Duration::from_secs(30),         // timeout
            )),
            adaptive_sampler: Arc::new(AdaptiveSampler::new(
                0.01,                           // 1% minimum sampling
                1.0,                            // 100% maximum sampling
                0.1,                            // 10% adjustment factor
                Duration::from_secs(5 * 60),    // 5-minute window
            )),
        }
    }

    /// Validate performance budget for two processing results
    pub async fn validate_budget(
        &self,
        run1: &ProcessingResult,
        run2: &ProcessingResult,
    ) -> Result<PerformanceBudgetCheck, ValidationError> {
        let start_time = Instant::now();

        // Calculate p95 latency from both runs
        let latencies = vec![
            run1.performance_metrics.p95_latency_ms,
            run2.performance_metrics.p95_latency_ms,
        ];
        
        let p95_latency = self.calculate_p95(&latencies);
        let budget_threshold = self.budget_threshold_percent;
        let budget_met = p95_latency <= budget_threshold;

        // Record performance reading
        self.record_performance(
            run1.performance_metrics.duration_ms as f64,
            OperationType::Replay,
        ).await;

        // Check circuit breaker
        if !budget_met {
            self.circuit_breaker.record_failure().await;
        } else {
            self.circuit_breaker.record_success().await;
        }

        // Adjust sampling rate if needed
        let new_sampling_rate = self.adjust_sampling_rate(p95_latency).await;

        // Calculate performance ratio
        let performance_ratio = p95_latency / budget_threshold;

        let validation_duration = start_time.elapsed().as_millis() as f64;
        debug!(
            "Performance budget validation completed in {:.2}ms: budget_met={}, p95={:.2}ms, threshold={:.2}%",
            validation_duration, budget_met, p95_latency, budget_threshold
        );

        Ok(PerformanceBudgetCheck {
            budget_met,
            p95_latency_ms: p95_latency,
            budget_threshold_ms: budget_threshold,
            performance_ratio,
            sampling_rate: new_sampling_rate,
        })
    }

    /// Record a performance measurement
    async fn record_performance(&self, duration_ms: f64, operation_type: OperationType) {
        let current_rate = self.get_current_sampling_rate();
        
        let reading = PerformanceReading {
            timestamp: Instant::now(),
            duration_ms,
            operation_type,
            sampling_rate: current_rate,
        };

        let mut history = self.performance_history.write().await;
        
        // Maintain window size
        if history.len() >= 10000 {
            history.pop_front();
        }
        
        history.push_back(reading);
    }

    /// Calculate p95 latency from a set of measurements
    fn calculate_p95(&self, latencies: &[f64]) -> f64 {
        if latencies.is_empty() {
            return 0.0;
        }
        
        let mut sorted = latencies.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let index = (sorted.len() as f64 * 0.95).ceil() as usize - 1;
        sorted.get(index).copied().unwrap_or(0.0)
    }

    /// Adjust sampling rate based on performance
    async fn adjust_sampling_rate(&self, current_p95: f64) -> f64 {
        let threshold = self.budget_threshold_percent;
        let ratio = current_p95 / threshold;

        let new_rate = if ratio > 1.0 {
            // Performance exceeding budget - reduce sampling
            let reduction_factor = (ratio - 1.0).min(0.9); // Max 90% reduction
            let current_rate = self.get_current_sampling_rate();
            (current_rate * (1.0 - reduction_factor)).max(self.adaptive_sampler.min_sampling_rate)
        } else {
            // Performance within budget - potentially increase sampling
            let increase_factor = (1.0 - ratio).min(0.1); // Max 10% increase
            let current_rate = self.get_current_sampling_rate();
            (current_rate * (1.0 + increase_factor)).min(self.adaptive_sampler.max_sampling_rate)
        };

        // Consider system load
        let load_adjusted_rate = self.adaptive_sampler.adjust_for_load(new_rate).await;

        self.set_sampling_rate(load_adjusted_rate);

        if (load_adjusted_rate - self.get_current_sampling_rate()).abs() > 0.01 {
            info!(
                "Adjusted sampling rate: {:.1}% -> {:.1}% (p95: {:.2}ms, threshold: {:.2}%)",
                self.get_current_sampling_rate() * 100.0,
                load_adjusted_rate * 100.0,
                current_p95,
                threshold
            );
        }

        load_adjusted_rate
    }

    /// Get current sampling rate
    pub fn get_current_sampling_rate(&self) -> f64 {
        self.current_sampling_rate.load(Ordering::SeqCst) as f64 / 10000.0
    }

    /// Set sampling rate
    fn set_sampling_rate(&self, rate: f64) {
        let rate_int = (rate * 10000.0).round() as usize;
        self.current_sampling_rate.store(rate_int, Ordering::SeqCst);
    }

    /// Check if operation should be sampled
    pub fn should_sample(&self, operation_type: &OperationType) -> bool {
        match operation_type {
            OperationType::StructuralEdit => true, // 100% sampling for structural edits
            _ => {
                let rate = self.get_current_sampling_rate();
                let random_value: f64 = rand::thread_rng().gen();
                random_value < rate
            }
        }
    }

    /// Get performance statistics
    pub async fn get_performance_stats(&self) -> PerformanceStats {
        let history = self.performance_history.read().await;
        
        if history.is_empty() {
            return PerformanceStats::default();
        }

        let durations: Vec<f64> = history.iter().map(|r| r.duration_ms).collect();
        let p50 = self.calculate_percentile(&durations, 0.5);
        let p95 = self.calculate_percentile(&durations, 0.95);
        let p99 = self.calculate_percentile(&durations, 0.99);
        
        let avg_duration = durations.iter().sum::<f64>() / durations.len() as f64;
        let max_duration = durations.iter().fold(0.0_f64, |max, &val| max.max(val));
        let min_duration = durations.iter().fold(f64::INFINITY, |min, &val| min.min(val));

        // Count operations by type
        let mut structural_edits = 0;
        let mut diagnostics = 0;
        let mut replays = 0;
        let mut validations = 0;

        for reading in history.iter() {
            match reading.operation_type {
                OperationType::StructuralEdit => structural_edits += 1,
                OperationType::Diagnostic => diagnostics += 1,
                OperationType::Replay => replays += 1,
                OperationType::Validation => validations += 1,
            }
        }

        let circuit_state = self.circuit_breaker.get_state().await;
        let current_sampling_rate = self.get_current_sampling_rate();

        PerformanceStats {
            total_operations: history.len(),
            avg_duration_ms: avg_duration,
            p50_duration_ms: p50,
            p95_duration_ms: p95,
            p99_duration_ms: p99,
            max_duration_ms: max_duration,
            min_duration_ms: if min_duration == f64::INFINITY { 0.0 } else { min_duration },
            structural_edits,
            diagnostics,
            replays,
            validations,
            current_sampling_rate,
            circuit_breaker_state: circuit_state,
            budget_threshold_ms: self.budget_threshold_percent,
        }
    }

    fn calculate_percentile(&self, values: &[f64], percentile: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let index = (sorted.len() as f64 * percentile).ceil() as usize - 1;
        sorted.get(index).copied().unwrap_or(0.0)
    }
}

impl CircuitBreaker {
    pub fn new(failure_threshold: usize, success_threshold: usize, timeout: Duration) -> Self {
        Self {
            state: Mutex::new(CircuitState::Closed),
            failure_threshold,
            success_threshold,
            timeout,
            failure_count: AtomicU64::new(0),
            last_failure_time: Mutex::new(None),
        }
    }

    pub async fn record_failure(&self) {
        let count = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;
        *self.last_failure_time.lock().unwrap() = Some(Instant::now());

        let mut state = self.state.lock().unwrap();
        if count >= self.failure_threshold as u64 && matches!(*state, CircuitState::Closed) {
            *state = CircuitState::Open;
            warn!("Circuit breaker opened after {} failures", count);
        }
    }

    pub async fn record_success(&self) {
        let mut state = self.state.lock().unwrap();
        match *state {
            CircuitState::HalfOpen => {
                // Reset failure count and close circuit
                self.failure_count.store(0, Ordering::SeqCst);
                *state = CircuitState::Closed;
                info!("Circuit breaker closed after successful operation");
            }
            CircuitState::Open => {
                // Check if timeout period has passed
                if let Some(last_failure) = *self.last_failure_time.lock().unwrap() {
                    if last_failure.elapsed() >= self.timeout {
                        *state = CircuitState::HalfOpen;
                        info!("Circuit breaker moved to half-open state");
                    }
                }
            }
            CircuitState::Closed => {
                // Reset failure count on success
                self.failure_count.store(0, Ordering::SeqCst);
            }
        }
    }

    pub async fn can_execute(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        match *state {
            CircuitState::Closed | CircuitState::HalfOpen => true,
            CircuitState::Open => {
                // Check if timeout period has passed
                if let Some(last_failure) = *self.last_failure_time.lock().unwrap() {
                    if last_failure.elapsed() >= self.timeout {
                        *state = CircuitState::HalfOpen;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
        }
    }

    pub async fn get_state(&self) -> CircuitState {
        self.state.lock().unwrap().clone()
    }
}

impl AdaptiveSampler {
    pub fn new(
        min_sampling_rate: f64,
        max_sampling_rate: f64,
        adjustment_factor: f64,
        performance_window: Duration,
    ) -> Self {
        Self {
            min_sampling_rate,
            max_sampling_rate,
            adjustment_factor,
            performance_window,
            load_tracker: LoadTracker::new(100), // 100 sample window
        }
    }

    pub async fn adjust_for_load(&self, base_rate: f64) -> f64 {
        let load_metrics = self.load_tracker.get_metrics().await;
        
        // Reduce sampling under high load
        let load_factor = match load_metrics {
            LoadMetrics { cpu_usage, memory_usage, request_rate } => {
                let cpu_pressure = (cpu_usage - 70.0).max(0.0) / 30.0; // Scale 70-100% to 0-1
                let memory_pressure = (memory_usage - 80.0).max(0.0) / 20.0; // Scale 80-100% to 0-1
                let request_pressure = (request_rate - 100.0).max(0.0) / 100.0; // Scale 100+ RPS to 0+
                
                let max_pressure = cpu_pressure.max(memory_pressure).max(request_pressure.min(1.0));
                1.0 - (max_pressure * 0.8) // Reduce sampling by up to 80% under pressure
            }
        };

        let adjusted_rate = base_rate * load_factor;
        adjusted_rate.clamp(self.min_sampling_rate, self.max_sampling_rate)
    }
}

impl LoadTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            cpu_samples: Mutex::new(VecDeque::with_capacity(window_size)),
            memory_samples: Mutex::new(VecDeque::with_capacity(window_size)),
            request_rate_samples: Mutex::new(VecDeque::with_capacity(window_size)),
            window_size,
        }
    }

    pub async fn record_metrics(&self, cpu: f64, memory: f64, request_rate: f64) {
        self.add_sample(&self.cpu_samples, cpu);
        self.add_sample(&self.memory_samples, memory);
        self.add_sample(&self.request_rate_samples, request_rate);
    }

    fn add_sample(&self, samples: &Mutex<VecDeque<f64>>, value: f64) {
        let mut queue = samples.lock().unwrap();
        if queue.len() >= self.window_size {
            queue.pop_front();
        }
        queue.push_back(value);
    }

    pub async fn get_metrics(&self) -> LoadMetrics {
        let cpu_avg = self.calculate_average(&self.cpu_samples);
        let memory_avg = self.calculate_average(&self.memory_samples);
        let request_rate_avg = self.calculate_average(&self.request_rate_samples);

        LoadMetrics {
            cpu_usage: cpu_avg,
            memory_usage: memory_avg,
            request_rate: request_rate_avg,
        }
    }

    fn calculate_average(&self, samples: &Mutex<VecDeque<f64>>) -> f64 {
        let queue = samples.lock().unwrap();
        if queue.is_empty() {
            return 0.0;
        }
        queue.iter().sum::<f64>() / queue.len() as f64
    }
}

// Supporting types

#[derive(Debug, Default)]
pub struct PerformanceStats {
    pub total_operations: usize,
    pub avg_duration_ms: f64,
    pub p50_duration_ms: f64,
    pub p95_duration_ms: f64,
    pub p99_duration_ms: f64,
    pub max_duration_ms: f64,
    pub min_duration_ms: f64,
    pub structural_edits: usize,
    pub diagnostics: usize,
    pub replays: usize,
    pub validations: usize,
    pub current_sampling_rate: f64,
    pub circuit_breaker_state: CircuitState,
    pub budget_threshold_ms: f64,
}

#[derive(Debug)]
struct LoadMetrics {
    pub cpu_usage: f64,      // Percentage
    pub memory_usage: f64,   // Percentage  
    pub request_rate: f64,   // Requests per second
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_performance_budget_enforcer() {
        let enforcer = PerformanceBudgetEnforcer::new(2.0); // 2% threshold

        // Create test results
        let result1 = ProcessingResult {
            slice_id: "test".to_string(),
            timestamp: chrono::Utc::now(),
            result_hash: "hash1".to_string(),
            performance_metrics: PerformanceMetrics {
                duration_ms: 100,
                memory_usage_mb: 64.0,
                cpu_usage_percent: 15.0,
                p95_latency_ms: 1.5, // Within budget
                throughput_ops_per_sec: 1000.0,
            },
            invariants: InvariantChecks {
                monotone_timestamps: true,
                causal_ordering: true,
                data_consistency: true,
                structural_integrity: true,
            },
            metadata: std::collections::HashMap::new(),
        };

        let result2 = result1.clone();

        let check = enforcer.validate_budget(&result1, &result2).await;
        assert!(check.is_ok());
        
        let budget_check = check.unwrap();
        assert!(budget_check.budget_met);
        assert_eq!(budget_check.p95_latency_ms, 1.5);
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let breaker = CircuitBreaker::new(3, 2, Duration::from_millis(100));

        // Initially closed
        assert!(breaker.can_execute().await);

        // Record failures
        breaker.record_failure().await;
        breaker.record_failure().await;
        breaker.record_failure().await;

        // Should be open now
        assert!(!breaker.can_execute().await);

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Should be half-open
        assert!(breaker.can_execute().await);

        // Record success
        breaker.record_success().await;

        // Should be closed
        assert!(breaker.can_execute().await);
    }

    #[test]
    fn test_sampling_decision() {
        let enforcer = PerformanceBudgetEnforcer::new(2.0);
        
        // Structural edits should always be sampled
        assert!(enforcer.should_sample(&OperationType::StructuralEdit));
        
        // Other operations depend on sampling rate
        enforcer.set_sampling_rate(0.0); // 0% sampling
        assert!(!enforcer.should_sample(&OperationType::Diagnostic));
        
        enforcer.set_sampling_rate(1.0); // 100% sampling  
        assert!(enforcer.should_sample(&OperationType::Diagnostic));
    }
}