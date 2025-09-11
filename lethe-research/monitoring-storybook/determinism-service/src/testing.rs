use crate::types::*;
use chrono::{DateTime, Utc};
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicI64, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use tokio::time::{sleep, timeout};
use tracing::{debug, info, warn};

/// Clock skew and causality testing framework
/// Validates system behavior under various timing conditions
pub struct ClockSkewTester {
    tolerance: Duration,
    test_scenarios: Vec<ClockSkewScenario>,
    synthetic_clock: Arc<SyntheticClock>,
    causality_tracker: Arc<CausalityTracker>,
}

#[derive(Debug, Clone)]
struct ClockSkewScenario {
    name: String,
    skew_ms: i64,
    duration_seconds: u64,
    message_count: usize,
}

/// Synthetic clock for injecting artificial clock skew in testing
pub struct SyntheticClock {
    base_offset_ms: AtomicI64,
    drift_rate: AtomicI64, // nanoseconds per second
    last_update: Mutex<Instant>,
}

/// Tracks causal ordering violations
pub struct CausalityTracker {
    events: Mutex<VecDeque<CausalEvent>>,
    max_events: usize,
    violations: AtomicU64,
}

#[derive(Debug, Clone)]
struct CausalEvent {
    id: u64,
    timestamp: DateTime<Utc>,
    logical_time: u64,
    dependencies: Vec<u64>,
    source: String,
}

impl ClockSkewTester {
    pub fn new(tolerance: Duration) -> Self {
        let test_scenarios = vec![
            ClockSkewScenario {
                name: "minimal_skew".to_string(),
                skew_ms: 100,  // 100ms skew
                duration_seconds: 30,
                message_count: 50,
            },
            ClockSkewScenario {
                name: "moderate_skew".to_string(),
                skew_ms: 1000, // 1s skew
                duration_seconds: 60,
                message_count: 100,
            },
            ClockSkewScenario {
                name: "high_skew".to_string(),
                skew_ms: 5000, // 5s skew
                duration_seconds: 120,
                message_count: 200,
            },
            ClockSkewScenario {
                name: "negative_skew".to_string(),
                skew_ms: -2000, // -2s skew (clock behind)
                duration_seconds: 60,
                message_count: 100,
            },
            ClockSkewScenario {
                name: "clock_drift".to_string(),
                skew_ms: 0,
                duration_seconds: 300, // 5 minutes with drift
                message_count: 500,
            },
        ];

        Self {
            tolerance,
            test_scenarios,
            synthetic_clock: Arc::new(SyntheticClock::new()),
            causality_tracker: Arc::new(CausalityTracker::new(10000)),
        }
    }

    /// Run comprehensive clock skew tests
    pub async fn run_skew_tests(&self) -> Result<Vec<ClockSkewDataPoint>, ValidationError> {
        let mut all_results = Vec::new();
        
        info!("Starting comprehensive clock skew tests");

        for scenario in &self.test_scenarios {
            info!("Running clock skew scenario: {}", scenario.name);
            
            let results = self.run_single_scenario(scenario).await?;
            all_results.extend(results);
        }

        info!("Completed all clock skew tests: {} data points collected", all_results.len());
        Ok(all_results)
    }

    async fn run_single_scenario(&self, scenario: &ClockSkewScenario) -> Result<Vec<ClockSkewDataPoint>, ValidationError> {
        let mut results = Vec::new();
        
        // Configure synthetic clock for this scenario
        self.synthetic_clock.set_skew(Duration::from_millis(scenario.skew_ms.abs() as u64), scenario.skew_ms < 0);
        
        if scenario.name == "clock_drift" {
            // Set a drift rate of 1ms per second
            self.synthetic_clock.set_drift_rate(1_000_000); // 1ms in nanoseconds
        } else {
            self.synthetic_clock.set_drift_rate(0);
        }

        let test_start = Instant::now();
        let test_duration = Duration::from_secs(scenario.duration_seconds);
        
        // Generate test events with timing variations
        let mut message_interval = Duration::from_millis((scenario.duration_seconds * 1000) / scenario.message_count as u64);

        while test_start.elapsed() < test_duration {
            // Wait for next message time
            if let Ok(()) = timeout(Duration::from_millis(100), sleep(message_interval)).await {
                // Generate test event
                let event_result = self.generate_test_event(&scenario.name).await?;
                
                // Record the result
                let data_point = ClockSkewDataPoint {
                    timestamp: Utc::now(),
                    skew_ms: scenario.skew_ms,
                    tolerance_met: event_result.tolerance_met,
                    test_type: scenario.name.clone(),
                };
                
                results.push(data_point);
                
                // Check for causality violations
                if !event_result.causal_ordering_valid {
                    warn!("Causal ordering violation detected in scenario: {}", scenario.name);
                }
                
                // Add some randomness to message timing
                use rand::Rng;
                let jitter = rand::thread_rng().gen_range(-50..=50);
                message_interval = Duration::from_millis(
                    (message_interval.as_millis() as i64 + jitter).max(1) as u64
                );
            }
        }

        // Reset synthetic clock
        self.synthetic_clock.reset();
        
        Ok(results)
    }

    async fn generate_test_event(&self, scenario_name: &str) -> Result<TestEventResult, ValidationError> {
        let synthetic_time = self.synthetic_clock.now();
        let real_time = Utc::now();
        
        // Calculate actual skew
        let time_diff = if synthetic_time > real_time {
            synthetic_time - real_time
        } else {
            real_time - synthetic_time
        };
        
        let actual_skew_ms = time_diff.num_milliseconds().abs() as u64;
        let tolerance_met = actual_skew_ms <= self.tolerance.as_millis() as u64;
        
        // Create a causal event
        let event = CausalEvent {
            id: {
                use rand::Rng;
                rand::thread_rng().gen()
            },
            timestamp: synthetic_time,
            logical_time: self.get_logical_time(),
            dependencies: self.generate_dependencies(),
            source: scenario_name.to_string(),
        };
        
        // Validate causal ordering
        let causal_ordering_valid = self.causality_tracker.validate_event(&event);
        self.causality_tracker.add_event(event);
        
        Ok(TestEventResult {
            tolerance_met,
            actual_skew_ms,
            causal_ordering_valid,
        })
    }

    /// Test out-of-order message delivery simulation
    pub async fn test_out_of_order_delivery(&self) -> Result<Vec<CausalityTestResult>, ValidationError> {
        info!("Starting out-of-order message delivery tests");
        
        let mut results = Vec::new();
        let message_count = 100;
        let mut messages = Vec::new();
        
        // Generate a sequence of messages
        for i in 0..message_count {
            let message = TestMessage {
                id: i,
                timestamp: Utc::now(),
                logical_time: i as u64,
                content: format!("test_message_{}", i),
                dependencies: if i > 0 { vec![i - 1] } else { vec![] },
            };
            messages.push(message);
            
            // Small delay to ensure timestamp differences
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        
        // Shuffle messages to simulate out-of-order delivery
        use rand::seq::SliceRandom;
        let mut shuffled_messages = messages.clone();
        shuffled_messages.shuffle(&mut rand::thread_rng());
        
        // Process messages in shuffled order and check for violations
        for message in shuffled_messages {
            let result = self.validate_message_ordering(&message, &messages).await?;
            results.push(result);
        }
        
        info!("Completed out-of-order delivery tests: {} messages processed", results.len());
        Ok(results)
    }

    async fn validate_message_ordering(&self, message: &TestMessage, all_messages: &[TestMessage]) -> Result<CausalityTestResult, ValidationError> {
        let mut violations = Vec::new();
        
        // Check dependencies are satisfied
        for dep_id in &message.dependencies {
            if let Some(dep_message) = all_messages.iter().find(|m| m.id == *dep_id) {
                if message.timestamp < dep_message.timestamp {
                    violations.push(format!(
                        "Message {} has timestamp before dependency {}", 
                        message.id, dep_id
                    ));
                }
                if message.logical_time <= dep_message.logical_time {
                    violations.push(format!(
                        "Message {} has logical time not greater than dependency {}", 
                        message.id, dep_id
                    ));
                }
            } else {
                violations.push(format!(
                    "Message {} references missing dependency {}", 
                    message.id, dep_id
                ));
            }
        }
        
        let is_valid = violations.is_empty();
        Ok(CausalityTestResult {
            message_id: message.id,
            violations,
            is_valid,
            logical_time: message.logical_time,
            physical_time: message.timestamp,
        })
    }

    /// Validate monotone timestamps under stress
    pub async fn validate_monotone_timestamps(&self, event_count: usize) -> Result<MonotonicityTestResult, ValidationError> {
        info!("Starting monotone timestamp validation with {} events", event_count);
        
        let mut timestamps = Vec::new();
        let mut violations = 0;
        
        for i in 0..event_count {
            let timestamp = self.generate_monotone_timestamp().await?;
            
            // Check if this timestamp is monotonic
            if let Some(last_timestamp) = timestamps.last() {
                if timestamp <= *last_timestamp {
                    violations += 1;
                    debug!("Monotonicity violation at event {}: {} <= {}", i, timestamp, last_timestamp);
                }
            }
            
            timestamps.push(timestamp);
            
            // Add small delay to allow for natural progression
            tokio::time::sleep(Duration::from_micros(100)).await;
        }
        
        let violation_rate = violations as f64 / event_count as f64;
        
        Ok(MonotonicityTestResult {
            total_events: event_count,
            violations,
            violation_rate,
            is_monotonic: violations == 0,
            first_timestamp: timestamps.first().copied().unwrap_or_default(),
            last_timestamp: timestamps.last().copied().unwrap_or_default(),
        })
    }

    async fn generate_monotone_timestamp(&self) -> Result<DateTime<Utc>, ValidationError> {
        // Use synthetic clock which should maintain monotonicity
        Ok(self.synthetic_clock.now())
    }

    fn get_logical_time(&self) -> u64 {
        // Simple logical clock implementation
        static LOGICAL_CLOCK: AtomicU64 = AtomicU64::new(0);
        LOGICAL_CLOCK.fetch_add(1, Ordering::SeqCst)
    }

    fn generate_dependencies(&self) -> Vec<u64> {
        // Generate random dependencies for testing
        use rand::Rng;
        let dep_count = rand::thread_rng().gen_range(0..=3);
        (0..dep_count).map(|_| rand::thread_rng().gen_range(0..1000)).collect()
    }
}

impl SyntheticClock {
    pub fn new() -> Self {
        Self {
            base_offset_ms: AtomicI64::new(0),
            drift_rate: AtomicI64::new(0),
            last_update: Mutex::new(Instant::now()),
        }
    }

    pub fn set_skew(&self, skew: Duration, negative: bool) {
        let skew_ms = if negative {
            -(skew.as_millis() as i64)
        } else {
            skew.as_millis() as i64
        };
        self.base_offset_ms.store(skew_ms, Ordering::SeqCst);
    }

    pub fn set_drift_rate(&self, drift_ns_per_sec: i64) {
        self.drift_rate.store(drift_ns_per_sec, Ordering::SeqCst);
    }

    pub fn now(&self) -> DateTime<Utc> {
        let mut last_update = self.last_update.lock().unwrap();
        let now = Instant::now();
        let elapsed = now.duration_since(*last_update);
        *last_update = now;
        drop(last_update);

        let base_offset_ms = self.base_offset_ms.load(Ordering::SeqCst);
        let drift_rate = self.drift_rate.load(Ordering::SeqCst);

        // Calculate additional drift
        let drift_ms = (elapsed.as_secs() * (drift_rate / 1_000_000) as u64) as i64;
        let total_offset_ms = base_offset_ms + drift_ms;

        let real_time = Utc::now();
        real_time + chrono::Duration::milliseconds(total_offset_ms)
    }

    pub fn reset(&self) {
        self.base_offset_ms.store(0, Ordering::SeqCst);
        self.drift_rate.store(0, Ordering::SeqCst);
        *self.last_update.lock().unwrap() = Instant::now();
    }
}

impl CausalityTracker {
    pub fn new(max_events: usize) -> Self {
        Self {
            events: Mutex::new(VecDeque::new()),
            max_events,
            violations: AtomicU64::new(0),
        }
    }

    pub fn add_event(&self, event: CausalEvent) {
        let mut events = self.events.lock().unwrap();
        
        if events.len() >= self.max_events {
            events.pop_front();
        }
        
        events.push_back(event);
    }

    pub fn validate_event(&self, event: &CausalEvent) -> bool {
        let events = self.events.lock().unwrap();
        
        for dep_id in &event.dependencies {
            if let Some(dep_event) = events.iter().find(|e| e.id == *dep_id) {
                // Check temporal ordering
                if event.timestamp < dep_event.timestamp {
                    self.violations.fetch_add(1, Ordering::SeqCst);
                    return false;
                }
                
                // Check logical ordering
                if event.logical_time <= dep_event.logical_time {
                    self.violations.fetch_add(1, Ordering::SeqCst);
                    return false;
                }
            }
        }
        
        true
    }

    pub fn get_violation_count(&self) -> u64 {
        self.violations.load(Ordering::SeqCst)
    }
}

// Helper types for testing
#[derive(Debug)]
struct TestEventResult {
    tolerance_met: bool,
    actual_skew_ms: u64,
    causal_ordering_valid: bool,
}

#[derive(Debug, Clone)]
struct TestMessage {
    id: u64,
    timestamp: DateTime<Utc>,
    logical_time: u64,
    content: String,
    dependencies: Vec<u64>,
}

#[derive(Debug)]
pub struct CausalityTestResult {
    pub message_id: u64,
    pub violations: Vec<String>,
    pub is_valid: bool,
    pub logical_time: u64,
    pub physical_time: DateTime<Utc>,
}

#[derive(Debug)]
pub struct MonotonicityTestResult {
    pub total_events: usize,
    pub violations: usize,
    pub violation_rate: f64,
    pub is_monotonic: bool,
    pub first_timestamp: DateTime<Utc>,
    pub last_timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_synthetic_clock_skew() {
        let clock = SyntheticClock::new();
        let baseline = Utc::now();
        
        // Set 1 second positive skew
        clock.set_skew(Duration::from_secs(1), false);
        
        let skewed_time = clock.now();
        assert!(skewed_time > baseline);
        
        let diff = skewed_time - baseline;
        assert!(diff.num_milliseconds() >= 900); // Allow some tolerance
        assert!(diff.num_milliseconds() <= 1100);
    }

    #[tokio::test]
    async fn test_causality_tracker() {
        let tracker = CausalityTracker::new(100);
        
        let event1 = CausalEvent {
            id: 1,
            timestamp: Utc::now(),
            logical_time: 1,
            dependencies: vec![],
            source: "test".to_string(),
        };
        
        let event2 = CausalEvent {
            id: 2,
            timestamp: Utc::now() + chrono::Duration::milliseconds(100),
            logical_time: 2,
            dependencies: vec![1],
            source: "test".to_string(),
        };
        
        tracker.add_event(event1);
        assert!(tracker.validate_event(&event2));
    }

    #[tokio::test]
    async fn test_clock_skew_tester() {
        let tester = ClockSkewTester::new(Duration::from_millis(1000));
        
        // This should not panic and should return results
        let results = tester.run_skew_tests().await;
        assert!(results.is_ok());
        
        let data_points = results.unwrap();
        assert!(!data_points.is_empty());
    }
}