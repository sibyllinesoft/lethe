# Lethe Tuning Campaigns System

A production-ready system for running concrete optimization campaigns that builds on the existing Gap→Tune→Verify framework. This system implements sophisticated priority scoring, Bayesian optimization, and automated validation pipelines for systematic model optimization.

## Overview

The tuning campaigns system implements the complete workflow from the TODO specification:

1. **Priority Scoring**: Mathematical formula to rank optimization opportunities
2. **Campaign Management**: Systematic execution of 1-2 slice campaigns per budget tier with 12-18 BO trials
3. **Four Specific Campaigns**: Ready-to-run implementations for immediate deployment
4. **Validation Pipeline**: Comprehensive gates and guardrails with statistical significance testing
5. **Promotion Pipeline**: Automated validation and approval for production deployment
6. **Microsite Integration**: Automatic annotation of validated configurations
7. **Monitoring & Reporting**: Real-time tracking and comprehensive analytics

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Campaign Orchestrator                               │
├─────────────────┬───────────────────┬───────────────────┬──────────────────┤
│ Priority Scorer │ Campaign Manager  │ Validation        │ Microsite        │
│                 │                   │ Pipeline          │ Integration      │
│ • Mathematical  │ • Bayesian Opt   │ • Campaign Gates  │ • Auto Annotation│
│   scoring       │ • 12-18 trials    │ • Guardrails      │ • "Tuned-vX"     │
│ • Risk penalty  │ • Concurrent exec │ • Statistical     │ • Results pages  │
│ • Traffic weight│                   │   testing         │                  │
├─────────────────┼───────────────────┼───────────────────┼──────────────────┤
│                          Monitoring & Reporting                            │
│ • Real-time status • Resource tracking • Alert system • Analytics         │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Install dependencies
pip install scikit-optimize scipy pandas numpy statsmodels

# Navigate to tuning campaigns directory
cd /path/to/lethe-research/tuning_campaigns
```

## Quick Start

### 1. Two-Week Campaign Execution

Run the complete two-week campaign plan:

```bash
# Create execution plan and run priority analysis
python orchestrator.py plan --output-dir ./campaign_results

# Start campaign execution
python orchestrator.py start --output-dir ./campaign_results

# Monitor progress
python orchestrator.py status --output-dir ./campaign_results

# Generate final report
python orchestrator.py report --output-dir ./campaign_results
```

### 2. Individual Campaign Execution

```python
from tuning_campaigns import CampaignManager, CampaignFactory
from tuning_campaigns.specific_campaigns import create_demo_slice_candidates

# Create campaign manager
manager = CampaignManager(output_dir="./my_campaigns")

# Create specific campaign (e.g., Zh.QA @ 8%)
candidates = create_demo_slice_candidates()
spec = CampaignFactory.create_campaign_spec("zh_qa_8", candidates["zh_qa_8"])

# Execute campaign
campaign = manager.create_campaign(spec)
manager.start_campaign(campaign.campaign_id)

# Monitor progress
status = manager.get_campaign_status(campaign.campaign_id)
print(f"Progress: {status['progress']['trials_completed']}/{status['progress']['trials_total']}")
```

### 3. Priority Analysis Only

```python
from tuning_campaigns import PriorityScorer
from tuning_campaigns.specific_campaigns import create_demo_slice_candidates

# Initialize scorer
scorer = PriorityScorer(risk_penalty_weight=0.5)

# Score candidates
candidates = list(create_demo_slice_candidates().values())
priorities = scorer.score_all_candidates(candidates)

# View top priorities
for i, p in enumerate(priorities[:5]):
    print(f"{i+1}. {p.slice_candidate.slice_name}: {p.priority_score:.4f}")
```

## Four Specific Campaigns

### Week 1 Campaigns (Fast Wins, Low Risk)

#### 1. Zh.QA @ 8% - Code-Switch Fragility
- **Strategy**: re-isotonic, CE early-exit cap +20%, K2:+25%, r=16, λ:+5%
- **Gates**: ΔP@5≥+1.5pp with CI>0, p95∆≤+1ms, KV drop≤1pp
- **Focus**: Chinese QA performance for code-switch scenarios

```python
from tuning_campaigns import CampaignFactory
from tuning_campaigns.specific_campaigns import create_demo_slice_candidates

candidates = create_demo_slice_candidates()
zh_spec = CampaignFactory.create_campaign_spec("zh_qa_8", candidates["zh_qa_8"])
print(f"Knobs: {len(zh_spec.knob_spaces)}, Gates: {list(zh_spec.gates.keys())}")
```

#### 2. JSON/PassKey @ 15% - Fact Needles  
- **Strategy**: CE early-exit OFF for CE@k≤50, K2:+25%, λ:+5%, head_micro-summaries ON
- **Gates**: Same as Zh.QA + ECE×FACT bin ≤0.06
- **Focus**: Fact needle extraction from JSON/structured data

### Week 2 Campaigns (Harder, Higher ROI)

#### 3. Code.Debug @ 15% - Long Closures
- **Strategy**: stronger closures ON, head_keep +2-3pp, K2:+15%, r=16, τ=0.75, λ:+5%
- **Gates**: ILP_used≤10% and zero closure breaks
- **Focus**: Debugging for long code closures and complex control flow

#### 4. Retrieve.KV @ 30% - KV Stability
- **Strategy**: maintain head anchor, shrink W/stride before touching head, sinks=64-96, μ:+5%
- **Gates**: KV prefix-reuse ≥ baseline and p99/p95≤2.0
- **Focus**: KV cache stability for high-budget retrieval scenarios

## Priority Scoring Formula

The system uses a sophisticated mathematical formula to prioritize optimization opportunities:

```
score = (max(0,ΔP@5)/CI_width)² × S × T - ρ×R
```

Where:
- **ΔP@5**: (competitor−Lethe) on paired slice
- **CI_width**: paired bootstrap 95% width  
- **S**: counterfactual sensitivity (∂P/∂K2, ∂P/∂λ from IPS replays)
- **T**: tenant/traffic weight
- **R**: risk factors (KV-prefix drop, ECE drift, p99/p95 inflation)
- **ρ**: fixed penalty weight (default: 0.5)

## Validation Pipeline

### Campaign Gates

Each campaign has specific gates that must pass for promotion:

```python
# Example gates for Zh.QA campaign
gates = {
    "min_delta_p5": 0.015,  # ΔP@5≥+1.5pp
    "min_ci_confidence": 0.0,  # CI>0 (positive improvement)
    "max_latency_p95_delta": 1.0,  # p95∆≤+1ms
    "max_kv_drop": 0.01  # KV drop≤1pp
}
```

### Guardrails System

Comprehensive guardrails prevent production risks:

1. **Coverage-weighted CRPS checks** - Calibration drift detection
2. **KV-prefix Jaccard penalties** - Cache stability monitoring  
3. **Curvature-gated r increases** - Complexity cost control
4. **τ move caps with ILP monitoring** - Parameter stability

### Statistical Validation

- **Holm-Bonferroni correction** for multiple comparisons
- **Union non-degradation** across all datasets at 8/15/30%
- **Bootstrap confidence intervals** for effect size estimation

## Monitoring System

### Real-time Monitoring

```python
from tuning_campaigns import CampaignMonitor, CampaignReporter

# Start monitoring
monitor = CampaignMonitor(db_path="./monitoring.db")
monitor.start_monitoring(campaign_manager)

# Get health status
health = monitor.get_campaign_health(campaign_id)
print(f"Health: {health.overall_health}, Progress: {health.progress_percentage:.1f}%")

# Generate reports
reporter = CampaignReporter(monitor)
report = reporter.generate_campaign_summary_report(campaign_id)
```

### Alert System

Automatic alerts for:
- Trial duration exceeding thresholds
- Success rate below minimum
- Consecutive failures
- Resource utilization spikes
- Guardrail violations

## Microsite Integration

Validated configurations are automatically annotated with "Tuned-vX (Validated)" labels:

```python
from tuning_campaigns import MicrositeIntegrator

integrator = MicrositeIntegrator()

# Auto-annotate validated configuration
config_id = integrator.annotate_validated_configuration(campaign, promotion_decision)

# Update dashboard
integrator.update_campaign_dashboard(campaigns)

# Generate detailed reports
integrator.create_campaign_report(campaign, promotion_decision)
```

## Configuration

### Custom Evaluator Function

Replace the mock evaluator with your actual model evaluation:

```python
def my_evaluator(parameters: Dict[str, Any], spec: CampaignSpec) -> Dict[str, float]:
    # Your model evaluation logic here
    # Apply parameters to model configuration
    # Run evaluation on slice data
    # Return metrics dict
    
    return {
        "delta_p5": 0.025,  # Improvement vs baseline
        "latency_p95_delta": 0.8,  # Latency change in ms
        "kv_prefix_drop": 0.008,  # KV cache stability
        "ece_drift": 0.02,  # Calibration change
        # ... other metrics
    }

manager = CampaignManager(evaluator_function=my_evaluator)
```

### Custom Alert Thresholds

```python
thresholds = {
    "max_trial_duration_minutes": 30.0,
    "min_success_rate": 0.5,
    "max_memory_usage_gb": 16.0,
    "max_consecutive_failures": 2
}

monitor = CampaignMonitor(alert_thresholds=thresholds)
```

### Gap Analysis Integration

Load slice candidates from existing Gap→Tune→Verify analysis:

```python
orchestrator = CampaignOrchestrator(
    gap_analysis_path="./gap_analysis_results.json"
)
```

## API Reference

### Core Classes

- **`CampaignOrchestrator`** - Main orchestrator for full campaign execution
- **`PriorityScorer`** - Implements mathematical priority scoring
- **`CampaignManager`** - Manages individual campaign execution with BO
- **`CampaignValidator`** - Validates trials against campaign gates
- **`PromotionPipeline`** - Comprehensive promotion validation
- **`CampaignMonitor`** - Real-time monitoring and alerting
- **`MicrositeIntegrator`** - Integration with microsite frontend

### Campaign Specifications

- **`ZhQACampaign`** - Chinese QA code-switch optimization
- **`JSONPassKeyCampaign`** - JSON fact needle extraction  
- **`CodeDebugCampaign`** - Code debugging closure optimization
- **`RetrieveKVCampaign`** - KV cache stability for high budgets

## Examples

See the `examples/` directory for complete working examples:

- `basic_campaign.py` - Simple campaign execution
- `priority_analysis.py` - Comprehensive priority scoring
- `monitoring_example.py` - Real-time monitoring setup
- `validation_pipeline.py` - Custom validation configuration

## Troubleshooting

### Common Issues

1. **scikit-optimize not available**: Install with `pip install scikit-optimize`
2. **Database locked**: Ensure only one monitor instance is running
3. **Campaign stuck**: Check alert system and resource utilization
4. **Gates not passing**: Review gate thresholds and parameter ranges

### Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Manual Campaign Control

```python
# Stop campaign
manager.stop_campaign(campaign_id)

# Check detailed status
results = manager.get_campaign_results(campaign_id)

# Manual validation
validator = CampaignValidator()
gate_results = validator.validate_trial(trial, spec)
```

## Production Deployment

### Recommended Setup

1. **Resource Allocation**: 8+ CPU cores, 32GB+ RAM for concurrent campaigns
2. **Database**: PostgreSQL for production monitoring (replace SQLite)
3. **Microsite API**: Configure actual microsite endpoints  
4. **Evaluation Function**: Implement actual model evaluation pipeline
5. **Alert Integration**: Connect alerts to Slack/PagerDuty

### Security Considerations

- Validate all parameter inputs to prevent injection
- Secure database connections and API endpoints
- Monitor resource usage to prevent DoS conditions
- Log all campaign actions for audit trails

## Contributing

The system is designed for extensibility:

1. **New Campaigns**: Subclass campaign specifications
2. **Custom Gates**: Add validation functions to `CampaignValidator`
3. **Additional Guardrails**: Extend `Guardrails` class
4. **New Metrics**: Add metric calculations to monitoring
5. **Integration Points**: Extend `MicrositeIntegrator` for other frontends

## License

This implementation is part of the Lethe research project and follows the same licensing terms.