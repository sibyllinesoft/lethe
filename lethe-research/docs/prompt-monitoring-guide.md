# Lethe Prompt Monitoring Guide

Comprehensive guide to using the Lethe prompt monitoring and tracking system for research workflows.

## 🎯 Overview

The Lethe prompt monitoring system provides comprehensive tracking and analysis of prompt executions, enabling researchers to:

- **Track Performance**: Monitor execution times, response quality, and resource usage
- **Compare Variations**: A/B test different prompt versions and configurations
- **Analyze Trends**: Understand how prompts evolve and improve over time
- **Debug Issues**: Identify and analyze prompt execution errors
- **Generate Insights**: Create research reports with statistical analysis

## 🏗️ Architecture

The monitoring system integrates seamlessly with existing Lethe infrastructure:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Research      │    │   Monitoring    │    │   Analysis      │
│   Workflows     ├────┤   System        ├────┤   Framework     │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MLflow        │    │   SQLite        │    │   Streamlit     │
│   Experiments   │    │   Database      │    │   Dashboard     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Components

- **`PromptTracker`**: Core tracking engine with database integration
- **`PromptDashboard`**: Web-based visualization and analytics
- **`LethePromptMonitor`**: Integration layer for Lethe workflows
- **CLI Tools**: Command-line interface for monitoring operations

## 🚀 Quick Start

### 1. Basic Setup

```python
from src.monitoring import track_prompt, get_analytics

# Track a single prompt execution
model_config = {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 1000
}

with track_prompt(
    prompt_id="my_retrieval_prompt",
    prompt_text="What are the benefits of hybrid retrieval?",
    model_config=model_config
) as execution:
    
    # Your processing logic here
    response = process_prompt(execution.prompt_text)
    
    # Update execution with results
    execution.response_text = response
    execution.response_quality_score = calculate_quality(response)
```

### 2. Launch Dashboard

```bash
# Quick status check
python scripts/prompt_monitor.py status

# Launch web dashboard
python scripts/prompt_monitor.py dashboard

# View in browser: http://localhost:8501
```

### 3. Run Test Suite

```bash
# Generate test data and validate functionality
python test_prompt_monitoring.py
```

## 📊 Core Features

### Execution Tracking

Every prompt execution is automatically tracked with:

- **Timing Metrics**: Execution time, tokens per second
- **Quality Metrics**: Response quality scores, coherence, relevance  
- **Resource Metrics**: Memory usage, CPU utilization
- **Error Tracking**: Comprehensive error logging and analysis
- **Context**: Model configuration, environment, git commit

```python
# Example: Comprehensive tracking
with track_prompt(
    prompt_id="complex_analysis",
    prompt_text="Analyze this research paper...",
    model_config={"model": "gpt-4", "temperature": 0.3},
    experiment_tag="paper_analysis",
    conversation_turn=3
) as execution:
    
    # Processing happens here
    result = analyze_paper(execution.prompt_text)
    
    # Rich metadata capture
    execution.response_text = result["analysis"]
    execution.response_quality_score = result["confidence"]
    execution.coherence_score = result["coherence"]
    execution.memory_usage_mb = get_memory_usage()
```

### Before/After Analysis

Automatically detect and analyze changes between prompt executions:

```python
# Version comparison with change detection
from src.monitoring.dashboard import PromptDashboard

dashboard = PromptDashboard()
comparison = dashboard.get_before_after_comparison(execution_id)

print("Changes detected:")
for change in comparison["changes_detected"]:
    print(f"  • {change}")
```

### A/B Testing and Comparisons

Compare prompt variations with statistical analysis:

```python
from src.monitoring import compare_prompts

# Execute baseline and treatment
baseline_id = execute_prompt("original_prompt", baseline_config)
treatment_id = execute_prompt("improved_prompt", treatment_config)

# Statistical comparison
comparison = compare_prompts(
    baseline_id, treatment_id,
    notes="Testing improved prompt clarity"
)

print(f"Quality improvement: {comparison.quality_improvement:+.3f}")
print(f"Performance change: {comparison.performance_change_percent:+.1f}%")
```

## 🔧 Integration with Lethe Components

### Retrieval Pipeline Monitoring

```python
from src.monitoring.integration_examples import LethePromptMonitor

monitor = LethePromptMonitor()

# Monitor retrieval pipeline
with monitor.monitor_retrieval_prompt(
    query="What is information retrieval?",
    retrieval_config={
        "method": "hybrid",
        "alpha": 0.7,
        "rerank": True
    },
    experiment_name="hybrid_retrieval_study"
) as execution:
    
    # Your retrieval logic
    results = retrieval_pipeline.search(execution.prompt_text)
    
    # Results are automatically tracked
    execution.response_text = json.dumps(results[:5])
    execution.response_quality_score = calculate_ndcg(results)
```

### Fusion Component Tracking

```python
# Track fusion component performance
fusion_execution_id = monitor.track_fusion_execution(
    query=query,
    lexical_results=lexical_results,
    semantic_results=semantic_results, 
    fused_results=fused_results,
    fusion_params={"method": "rrf", "alpha": 0.6},
    performance_metrics={
        "latency_ms": 150.2,
        "memory_mb": 45.1,
        "ndcg_10": 0.87
    }
)
```

## 📈 Analytics and Reporting

### Prompt Performance Analysis

```python
from src.monitoring import get_analytics

# Get comprehensive analytics for a prompt
analytics = get_analytics("my_prompt_id")

print(f"Total executions: {analytics['total_executions']}")
print(f"Success rate: {analytics['success_rate']:.1f}%")
print(f"Performance trend: {analytics['performance_trend']}")
print(f"Quality trend: {analytics['quality_trend']}")
```

### Trend Analysis

The system automatically calculates trends for:

- **Performance**: Response time improvements/degradations
- **Quality**: Response quality changes over time
- **Reliability**: Success rate trends
- **Resource Usage**: Memory and CPU utilization patterns

### Research Reports

```python
# Generate research report
report = monitor.generate_research_report("hybrid_retrieval_experiment")

print(f"Experiment: {report['experiment_tag']}")
print(f"Key findings: {report['key_findings']}")
print(f"Recommendations: {report['recommendations']}")
```

## 🖥️ Dashboard Features

The web dashboard provides:

### Summary View
- Total executions, unique prompts, success rates
- Recent activity and performance trends
- High-level system health metrics

### Timeline Analysis
- Daily execution counts and performance trends
- Error rates and quality improvements over time
- Resource usage patterns

### Prompt Performance
- Bubble chart showing execution count vs performance
- Quality scores and success rates visualization
- Drill-down capabilities for detailed analysis

### Model Comparison
- Performance comparison across different models
- Quality score analysis by model type
- Resource usage patterns by model

### Detailed Execution View
- Complete execution metadata and timing
- Before/after change detection
- Full prompt and response text
- Error analysis and debugging information

## 🛠️ Command Line Interface

### Basic Operations

```bash
# System status
python scripts/prompt_monitor.py status

# List all prompts
python scripts/prompt_monitor.py list --limit 20

# Analyze specific prompt
python scripts/prompt_monitor.py analyze my_prompt_id --verbose

# Show execution details
python scripts/prompt_monitor.py show execution_id --verbose
```

### Comparison and Analysis

```bash
# Compare two executions
python scripts/prompt_monitor.py compare baseline_id treatment_id \
  --notes "Testing new retrieval method"

# Export data for external analysis
python scripts/prompt_monitor.py export --format csv
python scripts/prompt_monitor.py export --format json
```

### Maintenance

```bash
# Clean old data (older than 30 days)
python scripts/prompt_monitor.py cleanup --days 30 --vacuum

# Launch dashboard on custom port
python scripts/prompt_monitor.py dashboard --port 8502
```

## 📊 Data Schema

### Execution Records

Each prompt execution stores:

```sql
CREATE TABLE prompt_executions (
    execution_id TEXT PRIMARY KEY,
    prompt_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_parameters_json TEXT,
    timestamp TEXT NOT NULL,
    execution_time_ms REAL DEFAULT 0.0,
    response_text TEXT,
    response_length INTEGER DEFAULT 0,
    response_quality_score REAL,
    coherence_score REAL,
    relevance_score REAL,
    error_occurred BOOLEAN DEFAULT FALSE,
    error_message TEXT,
    memory_usage_mb REAL DEFAULT 0.0,
    ab_test_group TEXT,
    experiment_tag TEXT
    -- ... additional fields
);
```

### Comparison Records

Statistical comparisons between executions:

```sql
CREATE TABLE prompt_comparisons (
    comparison_id TEXT PRIMARY KEY,
    baseline_execution_id TEXT NOT NULL,
    treatment_execution_id TEXT NOT NULL,
    quality_improvement REAL,
    performance_change_percent REAL,
    length_change_percent REAL,
    p_value REAL,
    effect_size REAL,
    is_significant BOOLEAN DEFAULT FALSE,
    comparison_type TEXT DEFAULT 'A/B',
    notes TEXT
);
```

## 🔬 Research Workflow Integration

### Experimental Design

```python
# Design A/B test for prompt variations
def run_prompt_experiment(queries, baseline_prompt, treatment_prompt):
    results = []
    
    for query in queries:
        # Baseline execution
        with track_prompt(
            prompt_id="experiment_baseline",
            prompt_text=baseline_prompt.format(query=query),
            model_config=baseline_config,
            ab_test_group="baseline"
        ) as baseline:
            baseline_result = execute_baseline(baseline.prompt_text)
            baseline.response_text = baseline_result
            baseline.response_quality_score = evaluate_quality(baseline_result)
        
        # Treatment execution
        with track_prompt(
            prompt_id="experiment_treatment", 
            prompt_text=treatment_prompt.format(query=query),
            model_config=treatment_config,
            ab_test_group="treatment"
        ) as treatment:
            treatment_result = execute_treatment(treatment.prompt_text)
            treatment.response_text = treatment_result
            treatment.response_quality_score = evaluate_quality(treatment_result)
        
        # Compare results
        comparison = compare_prompts(
            baseline.execution_id, treatment.execution_id,
            notes=f"Query: {query}"
        )
        
        results.append(comparison)
    
    return results
```

### Statistical Analysis Integration

The monitoring system integrates with the unified analysis framework:

```python
from src.analysis_unified import UnifiedAnalysisFramework

# Export monitoring data for statistical analysis
framework = UnifiedAnalysisFramework()
monitoring_data = tracker.export_data("csv")

# Integrate with existing analysis pipeline
framework.load_experimental_data(monitoring_data)
results = framework.run_complete_analysis()
```

## 🎯 Best Practices

### 1. Consistent Prompt IDs

Use meaningful, consistent prompt IDs:

```python
# Good: Descriptive and consistent
prompt_id = f"retrieval_{method}_{domain}_{version}"

# Bad: Random or unclear
prompt_id = "test123"
```

### 2. Comprehensive Metadata

Include rich context for better analysis:

```python
with track_prompt(
    prompt_id=prompt_id,
    prompt_text=prompt_text,
    model_config=model_config,
    experiment_tag="paper_study_2024",
    conversation_turn=turn_number,
    baseline_execution_id=baseline_id  # For comparisons
) as execution:
    # ... processing
```

### 3. Quality Metrics

Implement consistent quality scoring:

```python
def calculate_response_quality(response, ground_truth):
    return {
        "overall_quality": calculate_overall_score(response, ground_truth),
        "coherence": calculate_coherence(response),
        "relevance": calculate_relevance(response, ground_truth),
        "completeness": calculate_completeness(response, ground_truth)
    }
```

### 4. Error Handling

Comprehensive error tracking:

```python
try:
    with track_prompt(...) as execution:
        result = complex_processing(execution.prompt_text)
        execution.response_text = result
        
except ProcessingError as e:
    # Error is automatically tracked
    logger.error(f"Processing failed: {e}")
    # Additional error context can be added
    execution.error_context = {"input_length": len(execution.prompt_text)}
```

## 🔍 Troubleshooting

### Common Issues

**Database Lock Errors**
```bash
# Check for long-running processes
python scripts/prompt_monitor.py status

# Compact database to resolve locks
python scripts/prompt_monitor.py cleanup --vacuum
```

**Dashboard Not Loading**
```bash
# Install required dependencies
pip install streamlit plotly pandas

# Check port availability
python scripts/prompt_monitor.py dashboard --port 8502
```

**Missing Data**
```bash
# Verify database exists and has data
python scripts/prompt_monitor.py list

# Check recent activity
python scripts/prompt_monitor.py status
```

### Performance Optimization

**Large Datasets**
- Use data export for analysis of large datasets
- Configure automatic cleanup for old executions
- Use database indexing for frequently queried fields

**Dashboard Performance**
- Limit timeline to recent periods for faster loading
- Use sampling for very large datasets
- Enable caching for dashboard components

## 📚 API Reference

### Core Classes

**`PromptTracker`**
- `track_execution(prompt_id, prompt_text, model_config, **kwargs)`
- `get_prompt_analytics(prompt_id)`
- `compare_executions(baseline_id, treatment_id, notes)`
- `export_data(format)`

**`PromptDashboard`** 
- `get_summary_stats()`
- `get_execution_timeline(days)`
- `get_prompt_performance(limit)`
- `get_before_after_comparison(execution_id)`

**`LethePromptMonitor`**
- `monitor_retrieval_prompt(query, retrieval_config, experiment_name)`
- `track_fusion_execution(...)`
- `compare_retrieval_methods(...)`
- `analyze_prompt_evolution(prompt_id)`

### Utility Functions

```python
# Convenience functions
from src.monitoring import (
    track_prompt,      # Context manager for tracking
    get_analytics,     # Get prompt analytics
    compare_prompts,   # Compare two executions
    get_prompt_tracker # Get global tracker instance
)
```

## 🤝 Contributing

### Adding New Metrics

1. Extend `PromptExecution` dataclass
2. Update database schema in `_init_database()`
3. Add visualization in dashboard
4. Update CLI to display new metrics

### Creating Custom Plugins

```python
from src.monitoring.prompt_tracker import PromptExecution

class CustomAnalysisPlugin:
    def analyze_execution(self, execution: PromptExecution):
        # Custom analysis logic
        return {"custom_metric": calculate_custom_metric(execution)}
```

---

**For additional support and advanced usage examples, see the integration examples in `src/monitoring/integration_examples.py` and run the comprehensive test suite with `python test_prompt_monitoring.py`.**