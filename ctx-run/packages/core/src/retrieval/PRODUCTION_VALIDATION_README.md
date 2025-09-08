# Production Readiness Validation System

A sophisticated production validation framework for the Lethe retrieval system implementing three core mathematical proofs and comprehensive operational hardening.

## Overview

The Production Readiness Validation System ensures that Lethe retrieval deployments meet rigorous mathematical and operational standards before going to production. It implements:

1. **Three Core Mathematical Proofs**:
   - Dual Sanity Check (λ ↦ size monotonicity)
   - Out-of-Distribution Resilience (coverage-weighted CRPS)
   - Long-horizon Win Rate (hierarchical interleaving)

2. **Operational Hardening**:
   - Real-time monitoring with risk budget management
   - Chaos testing suite with automated recovery
   - DPP optimization with dynamic rank tuning
   - EmbeddingGemma-300M trial system with promotion gates

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                Production Readiness Orchestrator                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Validation     │  │  Monitoring     │  │  Hierarchical   │  │
│  │  System         │  │  System         │  │  Interleaving   │  │
│  │                 │  │                 │  │                 │  │
│  │  • Dual Sanity  │  │  • λ/size/CBU   │  │  • Multi-turn   │  │
│  │  • OOD Resil.   │  │  • CUSUM        │  │  • Statistical  │  │
│  │  • Long-horizon │  │  • Risk Budget  │  │  • Attribution  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐                      │
│  │  DPP            │  │  EmbeddingGemma │                      │
│  │  Optimization   │  │  Trial Engine   │                      │
│  │                 │  │                 │                      │
│  │  • Rank Tuning  │  │  • 7-day Canary │                      │
│  │  • ΔCBU/ms      │  │  • Promotion    │                      │
│  │  • Group Split  │  │  • Gates        │                      │
│  └─────────────────┘  └─────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage

```typescript
import { hybridRetrieval } from '@lethe/retrieval';

// Enable production validation
const results = await hybridRetrieval(queries, {
  db,
  embeddings,
  sessionId,
  config: {
    // Enable production validation
    production_validation: {
      enable_validation: true,
      enable_monitoring: true,
      
      // Quality gates
      dual_sanity_threshold: 0.005,  // <0.5% primal-dual gap
      ood_ece_threshold: 0.08,       // ≤8% calibration error
      win_rate_threshold: 0.80,      // ≥80% statistical power
      
      // Operational settings
      fail_fast_on_validation: false, // Continue on validation failures
      enable_chaos_testing: true,     // Enable fault injection
      risk_budget_threshold: 0.10     // 10% monthly error budget
    }
  }
});
```

### Advanced Configuration

```typescript
import { ProductionReadinessOrchestrator } from '@lethe/retrieval/production_orchestrator';

const orchestrator = new ProductionReadinessOrchestrator({
  session_id: 'prod-session-123',
  
  // Enable all subsystems
  enable_validation: true,
  enable_monitoring: true,
  enable_hierarchical_interleaving: true,
  enable_dpp_optimization: true,
  enable_embedding_gemma_trial: true,
  
  // Validation thresholds
  dual_sanity_threshold: 0.003,    // Stricter than default
  ood_ece_threshold: 0.06,         // Stricter calibration
  win_rate_threshold: 0.85,        // Higher power requirement
  
  // Monitoring configuration
  cusum_threshold: 2.5,            // Change detection sensitivity
  lambda_drift_bounds: [-0.1, 0.1], // Acceptable drift range
  risk_budget_threshold: 0.05,     // 5% monthly budget
  
  // Chaos testing
  enable_chaos_testing: true,
  chaos_scenarios: [
    'closure_cycle_injection',
    'rank_collapse_simulation',
    'kv_churn_spike'
  ],
  
  // DPP optimization
  dpp_config: {
    enable_rank_tuning: true,
    target_efficiency: 15.0,       // ΔCBU/ms target
    group_split_threshold: 0.7     // 70% contribution threshold
  },
  
  // EmbeddingGemma trial
  embedding_trial_config: {
    trial_duration_days: 7,
    promotion_threshold_cbu: 0.10, // ≥+10% ΔCBU/GB
    promotion_threshold_latency: 5 // ≥5ms p95 improvement
  }
});

// Run comprehensive assessment
const assessment = await orchestrator.assessProductionReadiness({
  query_text: "implement async retry logic",
  candidate_pool: retrievedCandidates,
  retrieval_config: config,
  system_metrics: {
    current_load: 0.65,
    memory_usage: 0.45,
    cpu_utilization: 0.50
  }
});

if (assessment.overall_readiness) {
  console.log('✅ Production deployment approved');
} else {
  console.log('❌ Production deployment blocked:', assessment.failing_components);
}
```

## Core Components

### 1. Production Validation System

Implements the three core mathematical proofs required for production readiness.

#### Dual Sanity Check

Validates λ ↦ size monotonicity with primal-dual gap analysis:

```typescript
const dualResult = await validationSystem.validateDualSanity({
  lambda_range: [0.1, 2.0],
  size_samples: 50,
  target_gap: 0.005
});

// Result structure
interface DualSanityResult {
  monotonicity_satisfied: boolean;
  primal_dual_gap: number;          // Target: <0.5%
  lambda: number;                   // Optimal λ value
  size_violation_count: number;
  critical_points: number[];
  gap_history: Array<{
    lambda: number;
    gap: number;
    timestamp: number;
  }>;
}
```

#### Out-of-Distribution Resilience

Coverage-weighted CRPS with Mondrian conformal prediction:

```typescript
const oodResult = await validationSystem.validateOODResilience({
  coverage_level: 0.90,
  mondrian_alpha: 0.10,
  crps_samples: 1000
});

// Result structure
interface OODResilienceResult {
  coverage_achieved: boolean;
  weighted_crps: number;            // Lower is better
  expected_calibration_error: number; // Target: ≤8%
  mondrian_coverage: number;        // Actual coverage
  conformal_intervals: Array<[number, number]>;
  ood_detection_accuracy: number;
}
```

#### Long-horizon Win Rate

Hierarchical interleaving for multi-turn attribution:

```typescript
const winRateResult = await validationSystem.validateLongHorizonWinRate({
  session_count: 1000,
  turns_per_session: 5,
  statistical_power: 0.80
});

// Result structure
interface LongHorizonWinRateResult {
  statistical_power_achieved: boolean;
  ndcg_improvement: number;         // Target: ≥+10%
  win_rate: number;                 // Target: ≥80%
  session_attribution: Array<{
    session_id: string;
    turn_attributions: number[];
    cluster_pair_decisions: string[];
  }>;
  power_analysis: {
    sample_size: number;
    effect_size: number;
    achieved_power: number;
  };
}
```

### 2. Production Monitoring System

Real-time monitoring with automated alerting and risk budget management:

```typescript
const monitoringSystem = new ProductionMonitoringSystem({
  session_id: 'monitoring-session',
  cusum_threshold: 2.5,
  lambda_drift_bounds: [-0.1, 0.1],
  risk_budget_threshold: 0.10
});

// Start monitoring
await monitoringSystem.startMonitoring();

// Get real-time metrics
const metrics = await monitoringSystem.getCurrentMetrics();
```

#### Key Metrics Tracked

- **λ/size/CBU Dashboard**: Real-time parameter tracking
- **CUSUM Monitoring**: Change detection with 2.5σ threshold
- **Risk Budget Ledger**: Monthly error budget tracking
- **Chaos Test Results**: Automated fault injection outcomes
- **Performance SLAs**: P95 latency, throughput, error rates

### 3. Hierarchical Interleaving Engine

Multi-turn attribution with statistical validation:

```typescript
const interleavingEngine = new HierarchicalInterleavingEngine({
  atom_level_interleaving: true,
  cluster_pair_sessions: true,
  statistical_power_target: 0.80
});

// Setup interleaving experiment
const experiment = await interleavingEngine.setupExperiment({
  session_id: 'experiment-123',
  baseline_system: 'current_production',
  test_system: 'candidate_deployment',
  target_sessions: 10000
});
```

### 4. DPP Optimization Engine

Dynamic rank tuning with ΔCBU/ms efficiency curves:

```typescript
const dppEngine = new DPPOptimizationEngine({
  enable_rank_tuning: true,
  target_efficiency: 15.0,
  group_split_threshold: 0.7
});

// Run optimization
const optimization = await dppEngine.optimizeDiversityRanking({
  candidate_embeddings: embeddings,
  efficiency_target: 15.0,
  max_iterations: 100
});
```

### 5. EmbeddingGemma Trial System

Automated canary deployment with promotion gates:

```typescript
const trialEngine = new EmbeddingGemmaTrialEngine({
  trial_duration_days: 7,
  promotion_threshold_cbu: 0.10,
  promotion_threshold_latency: 5
});

// Start canary trial
const trial = await trialEngine.startCanaryTrial({
  baseline_model: 'current_embeddings',
  candidate_model: 'EmbeddingGemma-300M'
});
```

## Quality Gates

The system enforces strict quality gates before production deployment:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Primal-dual Gap | <0.5% | λ optimization convergence |
| ECE (Expected Calibration Error) | ≤8% | Uncertainty calibration quality |
| ILP Incidence | ≤5% | Integer programming usage |
| Statistical Power | ≥80% | Multi-turn attribution confidence |
| nDCG@10 Improvement | ≥+10% | Quality improvement target |
| P95 Latency | <160ms | Performance requirement |
| ΔCBU/ms Efficiency | ≥15.0 | Resource efficiency target |

## Operational Runbooks

### Deployment Checklist

1. **Pre-deployment Validation**
   ```bash
   # Run full validation suite
   npm run validation:production
   
   # Check quality gates
   npm run quality:gates
   
   # Validate monitoring setup
   npm run monitoring:validate
   ```

2. **Canary Deployment**
   ```bash
   # Start 5% traffic canary
   npm run deploy:canary --traffic=5
   
   # Monitor for 2 hours
   npm run monitor:canary --duration=2h
   
   # Promote if successful
   npm run deploy:promote
   ```

3. **Full Deployment**
   ```bash
   # Deploy to production
   npm run deploy:production
   
   # Start monitoring
   npm run monitoring:start
   
   # Validate deployment
   npm run validation:post-deploy
   ```

### Troubleshooting Guide

#### Validation Failures

**Dual Sanity Check Failed (Gap >0.5%)**
```bash
# Check λ parameter stability
npm run debug:lambda-drift

# Validate monotonicity assumptions
npm run debug:monotonicity

# Regenerate calibration data
npm run recalibrate:lambda
```

**OOD Resilience Failed (ECE >8%)**
```bash
# Recalibrate uncertainty estimates
npm run recalibrate:uncertainty

# Check conformal prediction intervals
npm run debug:conformal

# Validate CRPS calculations
npm run validate:crps
```

**Win Rate Below Threshold (<80%)**
```bash
# Check statistical power
npm run debug:power-analysis

# Validate interleaving setup
npm run debug:interleaving

# Increase sample size
npm run increase:sample-size
```

#### Monitoring Alerts

**CUSUM Change Detection Triggered**
```bash
# Investigate parameter drift
npm run investigate:drift

# Check for system changes
npm run audit:changes

# Reset baseline if confirmed
npm run reset:baseline
```

**Risk Budget Exceeded**
```bash
# Check error patterns
npm run analyze:errors

# Implement mitigation
npm run mitigate:risks

# Update budget allocation
npm run update:risk-budget
```

## Performance Characteristics

### Benchmarks

| Operation | P50 | P95 | P99 |
|-----------|-----|-----|-----|
| Dual Sanity Validation | 85ms | 150ms | 200ms |
| OOD Resilience Check | 120ms | 180ms | 250ms |
| Win Rate Calculation | 200ms | 350ms | 500ms |
| Full Validation Suite | 450ms | 650ms | 850ms |

### Resource Usage

- **Memory**: 50-80MB baseline, 150MB during validation
- **CPU**: 10-15% baseline, 40-60% during validation
- **Disk I/O**: Minimal (logs and metrics storage)
- **Network**: Low (monitoring data transmission)

## Integration Examples

### With Existing Pipeline

```typescript
// Modify existing retrieval configuration
const config = {
  ...existingConfig,
  production_validation: {
    enable_validation: true,
    enable_monitoring: true,
    dual_sanity_threshold: 0.005,
    ood_ece_threshold: 0.08,
    win_rate_threshold: 0.80
  }
};

const results = await hybridRetrieval(queries, { db, embeddings, sessionId, config });
```

### With Custom Monitoring

```typescript
import { ProductionMonitoringSystem } from '@lethe/retrieval/monitoring_system';

const monitoring = new ProductionMonitoringSystem({
  session_id: 'custom-monitoring',
  alert_channels: ['slack', 'email', 'pagerduty'],
  custom_metrics: {
    business_kpis: ['user_satisfaction', 'task_completion_rate'],
    technical_metrics: ['cache_hit_rate', 'db_query_time']
  }
});

// Custom alert handlers
monitoring.onAlert('lambda_drift', async (alert) => {
  await notifyOnCall(alert);
  await triggerAutomaticMitigation(alert);
});
```

### A/B Testing Integration

```typescript
import { HierarchicalInterleavingEngine } from '@lethe/retrieval/hierarchical_interleaving';

const abTest = new HierarchicalInterleavingEngine({
  experiment_name: 'new_ranking_algorithm',
  traffic_split: { control: 0.5, treatment: 0.5 },
  statistical_power_target: 0.90,
  minimum_sample_size: 50000
});

const results = await abTest.runInterleaving({
  control_system: 'current_production',
  treatment_system: 'new_algorithm',
  evaluation_metrics: ['ndcg', 'user_satisfaction', 'task_completion']
});
```

## API Reference

### ProductionReadinessOrchestrator

Main orchestrator class for production validation.

```typescript
class ProductionReadinessOrchestrator {
  constructor(config: ProductionReadinessConfig);
  
  async assessProductionReadiness(
    input: ProductionReadinessInput
  ): Promise<ProductionReadinessAssessment>;
  
  async startContinuousMonitoring(): Promise<void>;
  async stopContinuousMonitoring(): Promise<void>;
  
  async runChaosTest(scenario: ChaosScenario): Promise<ChaosTestResult>;
  async generateDeploymentReport(): Promise<DeploymentReport>;
}
```

### Configuration Interfaces

```typescript
interface ProductionReadinessConfig {
  session_id: string;
  enable_validation: boolean;
  enable_monitoring: boolean;
  enable_hierarchical_interleaving: boolean;
  enable_dpp_optimization: boolean;
  enable_embedding_gemma_trial: boolean;
  
  // Thresholds
  dual_sanity_threshold: number;
  ood_ece_threshold: number;
  win_rate_threshold: number;
  
  // Operational settings
  fail_fast_on_validation?: boolean;
  enable_chaos_testing?: boolean;
  risk_budget_threshold?: number;
  
  // Advanced configuration
  cusum_threshold?: number;
  lambda_drift_bounds?: [number, number];
  chaos_scenarios?: ChaosScenario[];
  dpp_config?: DPPOptimizationConfig;
  embedding_trial_config?: EmbeddingTrialConfig;
}
```

## Contributing

### Development Setup

```bash
# Install dependencies
npm install

# Run tests
npm test

# Run validation suite
npm run test:validation

# Start development server
npm run dev
```

### Adding New Validation Proofs

1. Implement validation logic in `ProductionValidationSystem`
2. Add configuration options to `ProductionReadinessConfig`
3. Update orchestrator to include new validation
4. Add comprehensive tests
5. Update documentation

### Performance Optimization

- Profile validation performance with `npm run profile:validation`
- Monitor memory usage during validation
- Optimize mathematical computations for production scale
- Consider async/parallel execution for independent validations

## License

Part of the Lethe retrieval system. See main LICENSE file for details.