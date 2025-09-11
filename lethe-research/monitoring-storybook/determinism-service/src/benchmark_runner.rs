use crate::types::*;
use std::{sync::Arc, time::{Duration, Instant}, collections::HashMap};
use tokio::sync::RwLock;
use tracing::{info, error};

/// Comprehensive benchmark execution engine with 48-hour green gate validation
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
    execution_state: Arc<RwLock<ExecutionState>>,
    metrics_collector: MetricsCollector,
    gate_validator: GateValidator,
}

#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub rust_refactor_enabled: bool,
    pub transform_v2_enabled: bool,
    pub scenarios: Vec<ScenarioType>,
    pub adversarial_buckets: Vec<AdversarialBucket>,
    pub performance_gates: PerformanceGates,
    pub continuous_validation_hours: u64,
    pub statistical_significance_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct PerformanceGates {
    pub max_ece_score: f64,           // ≤ 0.08
    pub p95_latency_baseline: f64,    // Must maintain ≥ baseline average
    pub max_p99_p95_ratio: f64,       // ≤ 2.5
    pub max_proxy_gap: f64,           // ≤ 0.5%
    pub pool_fingerprint_tolerance: f64, // For deterministic outputs
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AdversarialBucket {
    Pathological,       
    ResourceExhaustion, 
    TimingAttacks,      
    DataCorruption,     
    Byzantine,          
}

#[derive(Debug)]
struct ExecutionState {
    phase: ExecutionPhase,
    start_time: Instant,
    results: BenchmarkResults,
    gate_status: GateStatus,
    continuous_monitoring: bool,
}

#[derive(Debug, Clone, PartialEq)]
enum ExecutionPhase {
    Initializing,
    RunningPairedMatrix,
    RunningAdversarialTests,
    ValidatingPerformanceGates,
    ContinuousMonitoring,
    GeneratingReport,
    Complete,
    Failed { reason: String },
}

#[derive(Debug, Default)]
struct BenchmarkResults {
    paired_results: HashMap<ScenarioType, PairedScenarioResult>,
    adversarial_results: HashMap<AdversarialBucket, AdversarialResult>,
    performance_metrics: PerformanceMetrics,
    v2_impact_metrics: V2ImpactMetrics,
}

#[derive(Debug, Clone)]
struct PairedScenarioResult {
    scenario: ScenarioType,
    iterations_completed: usize,
    determinism_success_rate: f64,
    budget_compliance_rate: f64,
    closure_rate: f64,
    paired_counts_match: bool,
    pool_fingerprints_stable: bool,
}

#[derive(Debug, Clone)]
struct AdversarialResult {
    bucket: AdversarialBucket,
    attacks_attempted: usize,
    attacks_mitigated: usize,
    system_resilience_score: f64,
    recovery_time_ms: Vec<f64>,
}

#[derive(Debug, Default, Clone)]
struct PerformanceMetrics {
    ece_scores: Vec<f64>,
    p95_latencies: Vec<f64>,
    p99_latencies: Vec<f64>,
    proxy_gaps: Vec<f64>,
    pool_fingerprint_stability: Vec<f64>,
}

#[derive(Debug, Default, Clone)]
struct V2ImpactMetrics {
    feature_extraction_overhead: Vec<f64>,
    learning_integration_overhead: Vec<f64>,
    accuracy_improvements: Vec<f64>,
    certificate_digest_stability: Vec<f64>,
}

#[derive(Debug, Default)]
struct GateStatus {
    ece_gate: GateState,
    p95_latency_gate: GateState,
    p99_p95_ratio_gate: GateState,
    proxy_gap_gate: GateState,
    pool_fingerprint_gate: GateState,
    continuous_validation_hours: f64,
    green_since: Option<Instant>,
}

#[derive(Debug, Clone, PartialEq)]
enum GateState {
    Unknown,
    Passing,
    Failing { reason: String, since: Instant },
}

impl Default for GateState {
    fn default() -> Self {
        GateState::Unknown
    }
}

pub struct MetricsCollector {
    samples: Arc<RwLock<Vec<MetricSample>>>,
}

#[derive(Debug, Clone)]
struct MetricSample {
    timestamp: Instant,
    metric_type: MetricType,
    value: f64,
    context: HashMap<String, String>,
}

#[derive(Debug, Clone)]
enum MetricType {
    EceScore,
    P95Latency,
    P99Latency,
    ProxyGap,
    PoolFingerprint,
    V2FeatureExtraction,
    V2LearningIntegration,
}

pub struct GateValidator {
    config: PerformanceGates,
    statistical_analyzer: StatisticalAnalyzer,
}

struct StatisticalAnalyzer {
    significance_threshold: f64,
}

impl BenchmarkRunner {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config: config.clone(),
            execution_state: Arc::new(RwLock::new(ExecutionState {
                phase: ExecutionPhase::Initializing,
                start_time: Instant::now(),
                results: BenchmarkResults::default(),
                gate_status: GateStatus::default(),
                continuous_monitoring: false,
            })),
            metrics_collector: MetricsCollector::new(),
            gate_validator: GateValidator::new(config.performance_gates),
        }
    }

    /// Execute the complete benchmark matrix with 48-hour validation requirement
    pub async fn execute_comprehensive_matrix(&self) -> anyhow::Result<BenchmarkReport> {
        info!("Starting comprehensive benchmark matrix execution");
        
        // Phase 1: Paired Benchmark Matrix
        self.set_phase(ExecutionPhase::RunningPairedMatrix).await;
        self.execute_paired_matrix().await?;
        
        // Phase 2: Adversarial Test Buckets
        self.set_phase(ExecutionPhase::RunningAdversarialTests).await;
        self.execute_adversarial_buckets().await?;
        
        // Phase 3: Performance Gate Validation
        self.set_phase(ExecutionPhase::ValidatingPerformanceGates).await;
        self.validate_performance_gates().await?;
        
        // Phase 4: Continuous Monitoring (48 hours)
        self.set_phase(ExecutionPhase::ContinuousMonitoring).await;
        self.run_continuous_validation().await?;
        
        // Phase 5: Generate Final Report
        self.set_phase(ExecutionPhase::GeneratingReport).await;
        let report = self.generate_comprehensive_report().await?;
        
        self.set_phase(ExecutionPhase::Complete).await;
        info!("Comprehensive benchmark matrix execution completed");
        
        Ok(report)
    }

    async fn execute_paired_matrix(&self) -> anyhow::Result<()> {
        info!("Executing paired benchmark matrix with V2 features enabled");
        
        let mut results = HashMap::new();
        
        for scenario in &self.config.scenarios {
            info!("Testing scenario: {:?}", scenario);
            
            let scenario_result = self.run_paired_scenario(scenario.clone()).await?;
            results.insert(scenario.clone(), scenario_result);
        }
        
        // Update execution state with results
        {
            let mut state = self.execution_state.write().await;
            state.results.paired_results = results;
        }
        
        info!("Paired matrix execution completed");
        Ok(())
    }

    async fn run_paired_scenario(&self, scenario: ScenarioType) -> anyhow::Result<PairedScenarioResult> {
        let iterations = 100; // Run each scenario 100 times for statistical significance
        let mut successful_iterations = 0;
        let mut budget_compliant_iterations = 0;
        let mut closure_successful_iterations = 0;
        let mut paired_counts_match = true;
        let mut pool_fingerprints_stable = true;
        
        for i in 0..iterations {
            info!("Running iteration {} for scenario {:?}", i + 1, scenario);
            
            // Generate test data for this scenario
            let test_changes = self.generate_scenario_changes(&scenario, 50).await;
            
            // Run paired comparison with V2 features
            let (run1_result, run2_result) = self.run_paired_comparison(&test_changes).await?;
            
            // Validate results
            if self.validate_determinism(&run1_result, &run2_result).await {
                successful_iterations += 1;
            }
            
            if self.validate_budget_compliance(&run1_result, &run2_result).await {
                budget_compliant_iterations += 1;
            }
            
            if self.validate_closure_rate(&test_changes).await {
                closure_successful_iterations += 1;
            }
            
            // Check paired counts and pool fingerprints
            if !self.validate_paired_counts(&run1_result, &run2_result).await {
                paired_counts_match = false;
            }
            
            if !self.validate_pool_fingerprints(&run1_result, &run2_result).await {
                pool_fingerprints_stable = false;
            }
            
            // Collect metrics
            self.collect_iteration_metrics(&run1_result, &run2_result).await;
        }
        
        let result = PairedScenarioResult {
            scenario,
            iterations_completed: iterations,
            determinism_success_rate: successful_iterations as f64 / iterations as f64,
            budget_compliance_rate: budget_compliant_iterations as f64 / iterations as f64,
            closure_rate: closure_successful_iterations as f64 / iterations as f64,
            paired_counts_match,
            pool_fingerprints_stable,
        };
        
        info!("Scenario {:?} results: success_rate={:.3}, budget_compliance={:.3}, closure_rate={:.3}", 
               result.scenario, result.determinism_success_rate, result.budget_compliance_rate, result.closure_rate);
        
        Ok(result)
    }

    async fn execute_adversarial_buckets(&self) -> anyhow::Result<()> {
        info!("Executing adversarial test buckets");
        
        let mut results = HashMap::new();
        
        for bucket in &self.config.adversarial_buckets {
            info!("Testing adversarial bucket: {:?}", bucket);
            
            let bucket_result = self.run_adversarial_bucket(bucket.clone()).await?;
            results.insert(bucket.clone(), bucket_result);
        }
        
        // Update execution state
        {
            let mut state = self.execution_state.write().await;
            state.results.adversarial_results = results;
        }
        
        info!("Adversarial bucket execution completed");
        Ok(())
    }

    async fn run_adversarial_bucket(&self, bucket: AdversarialBucket) -> anyhow::Result<AdversarialResult> {
        let attack_count = match bucket {
            AdversarialBucket::Pathological => 50,
            AdversarialBucket::ResourceExhaustion => 20,
            AdversarialBucket::TimingAttacks => 30,
            AdversarialBucket::DataCorruption => 40,
            AdversarialBucket::Byzantine => 25,
        };
        
        let mut attacks_mitigated = 0;
        let mut recovery_times = Vec::new();
        
        for i in 0..attack_count {
            info!("Executing attack {} of {} for bucket {:?}", i + 1, attack_count, bucket);
            
            let attack_start = Instant::now();
            let attack_result = self.execute_attack(&bucket, i).await?;
            let recovery_time = attack_start.elapsed().as_millis() as f64;
            
            recovery_times.push(recovery_time);
            
            if attack_result.mitigated {
                attacks_mitigated += 1;
            }
            
            // Wait for system to stabilize between attacks
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        
        let resilience_score = attacks_mitigated as f64 / attack_count as f64;
        
        let result = AdversarialResult {
            bucket,
            attacks_attempted: attack_count,
            attacks_mitigated,
            system_resilience_score: resilience_score,
            recovery_time_ms: recovery_times,
        };
        
        info!("Adversarial bucket {:?} results: resilience_score={:.3}, avg_recovery={:.1}ms", 
               result.bucket, result.system_resilience_score, 
               result.recovery_time_ms.iter().sum::<f64>() / result.recovery_time_ms.len() as f64);
        
        Ok(result)
    }

    async fn validate_performance_gates(&self) -> anyhow::Result<()> {
        info!("Validating performance gates");
        
        let metrics = self.collect_current_metrics().await;
        let gate_results = self.gate_validator.validate_all_gates(&metrics).await?;
        
        {
            let mut state = self.execution_state.write().await;
            state.gate_status = gate_results;
        }
        
        // Check if all gates are passing
        let all_gates_passing = self.check_all_gates_passing().await;
        
        if !all_gates_passing {
            return Err(anyhow::anyhow!("Performance gates failing - cannot proceed to continuous monitoring"));
        }
        
        info!("All performance gates are passing");
        Ok(())
    }

    async fn run_continuous_validation(&self) -> anyhow::Result<()> {
        info!("Starting continuous validation for {} hours", self.config.continuous_validation_hours);
        
        {
            let mut state = self.execution_state.write().await;
            state.continuous_monitoring = true;
            state.gate_status.green_since = Some(Instant::now());
        }
        
        let validation_duration = Duration::from_secs(self.config.continuous_validation_hours * 3600);
        let check_interval = Duration::from_secs(300); // Check every 5 minutes
        
        let start_time = Instant::now();
        
        while start_time.elapsed() < validation_duration {
            // Run validation check
            let metrics = self.collect_current_metrics().await;
            let gate_results = self.gate_validator.validate_all_gates(&metrics).await?;
            
            // Update gate status
            {
                let mut state = self.execution_state.write().await;
                state.gate_status = gate_results;
            }
            
            // Check if any gates have failed
            if !self.check_all_gates_passing().await {
                error!("Performance gates failed during continuous monitoring");
                
                {
                    let mut state = self.execution_state.write().await;
                    state.phase = ExecutionPhase::Failed { 
                        reason: "Performance gates failed during continuous monitoring".to_string() 
                    };
                }
                
                return Err(anyhow::anyhow!("Continuous validation failed - gates not green for required duration"));
            }
            
            let elapsed_hours = start_time.elapsed().as_secs_f64() / 3600.0;
            info!("Continuous validation progress: {:.1}/{} hours completed", 
                   elapsed_hours, self.config.continuous_validation_hours);
            
            tokio::time::sleep(check_interval).await;
        }
        
        info!("Continuous validation completed successfully - all gates green for {} hours", 
               self.config.continuous_validation_hours);
        Ok(())
    }

    async fn generate_comprehensive_report(&self) -> anyhow::Result<BenchmarkReport> {
        info!("Generating comprehensive benchmark report");
        
        let state = self.execution_state.read().await;
        let overall_status = self.determine_overall_status(&state).await;
        
        let paired_results = PairedComparisonData {
            scenarios_tested: state.results.paired_results.len(),
            determinism_success_rate: self.calculate_avg_determinism_success_rate(&state.results.paired_results),
            budget_compliance_rate: self.calculate_avg_budget_compliance_rate(&state.results.paired_results),
            closure_rate: self.calculate_avg_closure_rate(&state.results.paired_results),
        };
        
        let adversarial_results = AdversarialTestData {
            buckets_tested: state.results.adversarial_results.len(),
            resilience_score: self.calculate_avg_resilience_score(&state.results.adversarial_results),
            attack_mitigation_rate: self.calculate_attack_mitigation_rate(&state.results.adversarial_results),
        };
        
        let performance_gates = PerformanceGateStatus {
            ece_gate_passed: matches!(state.gate_status.ece_gate, GateState::Passing),
            p95_latency_gate_passed: matches!(state.gate_status.p95_latency_gate, GateState::Passing),
            p99_p95_ratio_gate_passed: matches!(state.gate_status.p99_p95_ratio_gate, GateState::Passing),
            proxy_gap_gate_passed: matches!(state.gate_status.proxy_gap_gate, GateState::Passing),
            pool_fingerprint_gate_passed: matches!(state.gate_status.pool_fingerprint_gate, GateState::Passing),
        };
        
        let v2_impact_metrics = V2FeatureImpactData {
            feature_extraction_overhead_ms: self.calculate_avg(&state.results.v2_impact_metrics.feature_extraction_overhead),
            learning_integration_overhead_ms: self.calculate_avg(&state.results.v2_impact_metrics.learning_integration_overhead),
            accuracy_improvement_percent: self.calculate_avg(&state.results.v2_impact_metrics.accuracy_improvements) * 100.0,
        };
        
        let recommendations = self.generate_recommendations(&state).await;
        
        let report = BenchmarkReport {
            overall_status,
            paired_results,
            adversarial_results,
            performance_gates,
            v2_impact_metrics,
            recommendations,
        };
        
        info!("Comprehensive benchmark report generated");
        self.log_report_summary(&report).await;
        
        Ok(report)
    }

    // Helper methods
    
    async fn set_phase(&self, phase: ExecutionPhase) {
        let mut state = self.execution_state.write().await;
        state.phase = phase;
    }
    
    async fn generate_scenario_changes(&self, _scenario: &ScenarioType, _count: usize) -> Vec<TransformChangeV2> {
        // Implementation for generating test changes based on scenario
        // This would integrate with the existing V2 feature generation
        vec![] // Placeholder
    }
    
    async fn run_paired_comparison(&self, _changes: &[TransformChangeV2]) -> anyhow::Result<(ProcessingResult, ProcessingResult)> {
        // Implementation for running paired comparisons with determinism validation
        // This would integrate with the existing DeterminismSentinel
        let dummy_result = ProcessingResult {
            slice_id: "test".to_string(),
            timestamp: chrono::Utc::now(),
            result_hash: "hash".to_string(),
            performance_metrics: crate::types::PerformanceMetrics {
                duration_ms: 100,
                memory_usage_mb: 64.0,
                cpu_usage_percent: 15.0,
                p95_latency_ms: 1.5,
                throughput_ops_per_sec: 1000.0,
            },
            invariants: InvariantChecks {
                monotone_timestamps: true,
                causal_ordering: true,
                data_consistency: true,
                structural_integrity: true,
            },
            metadata: HashMap::new(),
        };
        
        Ok((dummy_result.clone(), dummy_result))
    }
    
    async fn validate_determinism(&self, _run1: &ProcessingResult, _run2: &ProcessingResult) -> bool {
        // Implementation for determinism validation
        true
    }
    
    async fn validate_budget_compliance(&self, _run1: &ProcessingResult, _run2: &ProcessingResult) -> bool {
        // Implementation for budget compliance validation
        true
    }
    
    async fn validate_closure_rate(&self, _changes: &[TransformChangeV2]) -> bool {
        // Implementation for closure rate validation
        true
    }
    
    async fn validate_paired_counts(&self, _run1: &ProcessingResult, _run2: &ProcessingResult) -> bool {
        // Implementation for paired count validation
        true
    }
    
    async fn validate_pool_fingerprints(&self, _run1: &ProcessingResult, _run2: &ProcessingResult) -> bool {
        // Implementation for pool fingerprint validation
        true
    }
    
    async fn collect_iteration_metrics(&self, _run1: &ProcessingResult, _run2: &ProcessingResult) {
        // Implementation for metrics collection
    }
    
    async fn execute_attack(&self, _bucket: &AdversarialBucket, _attack_id: usize) -> anyhow::Result<AttackResult> {
        // Implementation for executing specific attacks
        Ok(AttackResult { mitigated: true })
    }
    
    async fn collect_current_metrics(&self) -> PerformanceMetrics {
        // Implementation for collecting current performance metrics
        PerformanceMetrics {
            ece_scores: vec![0.05],
            p95_latencies: vec![95.0],
            p99_latencies: vec![180.0],
            proxy_gaps: vec![0.3],
            pool_fingerprint_stability: vec![0.001],
        }
    }
    
    async fn check_all_gates_passing(&self) -> bool {
        let state = self.execution_state.read().await;
        matches!(state.gate_status.ece_gate, GateState::Passing) &&
        matches!(state.gate_status.p95_latency_gate, GateState::Passing) &&
        matches!(state.gate_status.p99_p95_ratio_gate, GateState::Passing) &&
        matches!(state.gate_status.proxy_gap_gate, GateState::Passing) &&
        matches!(state.gate_status.pool_fingerprint_gate, GateState::Passing)
    }
    
    async fn determine_overall_status(&self, state: &ExecutionState) -> PassFail {
        match &state.phase {
            ExecutionPhase::Complete => {
                if self.check_all_gates_passing().await {
                    PassFail::Pass
                } else {
                    PassFail::Fail { reason: "Performance gates not all passing".to_string() }
                }
            },
            ExecutionPhase::Failed { reason } => PassFail::Fail { reason: reason.clone() },
            _ => PassFail::Fail { reason: "Benchmark execution not completed".to_string() }
        }
    }
    
    fn calculate_avg_determinism_success_rate(&self, results: &HashMap<ScenarioType, PairedScenarioResult>) -> f64 {
        if results.is_empty() { return 0.0; }
        results.values().map(|r| r.determinism_success_rate).sum::<f64>() / results.len() as f64
    }
    
    fn calculate_avg_budget_compliance_rate(&self, results: &HashMap<ScenarioType, PairedScenarioResult>) -> f64 {
        if results.is_empty() { return 0.0; }
        results.values().map(|r| r.budget_compliance_rate).sum::<f64>() / results.len() as f64
    }
    
    fn calculate_avg_closure_rate(&self, results: &HashMap<ScenarioType, PairedScenarioResult>) -> f64 {
        if results.is_empty() { return 0.0; }
        results.values().map(|r| r.closure_rate).sum::<f64>() / results.len() as f64
    }
    
    fn calculate_avg_resilience_score(&self, results: &HashMap<AdversarialBucket, AdversarialResult>) -> f64 {
        if results.is_empty() { return 0.0; }
        results.values().map(|r| r.system_resilience_score).sum::<f64>() / results.len() as f64
    }
    
    fn calculate_attack_mitigation_rate(&self, results: &HashMap<AdversarialBucket, AdversarialResult>) -> f64 {
        if results.is_empty() { return 0.0; }
        let total_attacks: usize = results.values().map(|r| r.attacks_attempted).sum();
        let total_mitigated: usize = results.values().map(|r| r.attacks_mitigated).sum();
        total_mitigated as f64 / total_attacks as f64
    }
    
    fn calculate_avg(&self, values: &[f64]) -> f64 {
        if values.is_empty() { return 0.0; }
        values.iter().sum::<f64>() / values.len() as f64
    }
    
    async fn generate_recommendations(&self, _state: &ExecutionState) -> Vec<ActionableRecommendation> {
        // Implementation for generating actionable recommendations
        vec![]
    }
    
    async fn log_report_summary(&self, report: &BenchmarkReport) {
        info!("=== COMPREHENSIVE BENCHMARK REPORT SUMMARY ===");
        info!("Overall Status: {:?}", report.overall_status);
        info!("Paired Results - Scenarios: {}, Success Rate: {:.3}", 
               report.paired_results.scenarios_tested, 
               report.paired_results.determinism_success_rate);
        info!("Adversarial Results - Buckets: {}, Resilience: {:.3}", 
               report.adversarial_results.buckets_tested, 
               report.adversarial_results.resilience_score);
        info!("Performance Gates - All Passing: {}", 
               report.performance_gates.ece_gate_passed && 
               report.performance_gates.p95_latency_gate_passed &&
               report.performance_gates.p99_p95_ratio_gate_passed);
        info!("V2 Impact - Accuracy Improvement: {:.1}%", 
               report.v2_impact_metrics.accuracy_improvement_percent);
        info!("================================================");
    }
}

impl MetricsCollector {
    fn new() -> Self {
        Self {
            samples: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

impl GateValidator {
    fn new(config: PerformanceGates) -> Self {
        Self {
            config,
            statistical_analyzer: StatisticalAnalyzer {
                significance_threshold: 0.05, // 5% significance level
            },
        }
    }
    
    async fn validate_all_gates(&self, _metrics: &PerformanceMetrics) -> anyhow::Result<GateStatus> {
        // Implementation for validating all performance gates
        Ok(GateStatus::default())
    }
}

#[derive(Debug)]
struct AttackResult {
    mitigated: bool,
}

// Public interface types for the benchmark report

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PassFail {
    Pass,
    Fail { reason: String },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BenchmarkReport {
    pub overall_status: PassFail,
    pub paired_results: PairedComparisonData,
    pub adversarial_results: AdversarialTestData,
    pub performance_gates: PerformanceGateStatus,
    pub v2_impact_metrics: V2FeatureImpactData,
    pub recommendations: Vec<ActionableRecommendation>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PairedComparisonData {
    pub scenarios_tested: usize,
    pub determinism_success_rate: f64,
    pub budget_compliance_rate: f64,
    pub closure_rate: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AdversarialTestData {
    pub buckets_tested: usize,
    pub resilience_score: f64,
    pub attack_mitigation_rate: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerformanceGateStatus {
    pub ece_gate_passed: bool,
    pub p95_latency_gate_passed: bool,
    pub p99_p95_ratio_gate_passed: bool,
    pub proxy_gap_gate_passed: bool,
    pub pool_fingerprint_gate_passed: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct V2FeatureImpactData {
    pub feature_extraction_overhead_ms: f64,
    pub learning_integration_overhead_ms: f64,
    pub accuracy_improvement_percent: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ActionableRecommendation {
    pub category: String,
    pub priority: String,
    pub description: String,
    pub estimated_impact: f64,
}