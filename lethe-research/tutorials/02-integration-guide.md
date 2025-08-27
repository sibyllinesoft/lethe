# Integration Guide: Lethe Workflows & MLflow

Learn how to integrate the prompt monitoring system with your existing Lethe research workflows and MLflow experiment tracking.

## 🎯 Overview

This guide covers:

- **Lethe Integration**: Seamless integration with existing research workflows
- **MLflow Integration**: Experiment tracking and model versioning
- **A/B Testing**: Systematic prompt comparison and optimization
- **Production Integration**: Monitoring in live research environments

## 🔗 Lethe Workflow Integration

### Basic Lethe Integration

The simplest way to add monitoring to existing Lethe workflows:

```python
# File: examples/lethe_basic_integration.py
from src.monitoring import LethePromptMonitor
from datetime import datetime

# Initialize the monitor
monitor = LethePromptMonitor(
    experiment_name="retrieval_optimization",
    project_id="lethe_research_2024"
)

# Example: Integrated retrieval workflow
def enhanced_retrieval_workflow(query, retrieval_config):
    """Enhanced retrieval with integrated monitoring."""
    
    # Start monitoring the retrieval process
    with monitor.track_prompt_execution(
        prompt_id="hybrid_retrieval",
        prompt_text=query,
        model_config=retrieval_config,
        stage="retrieval"
    ) as execution:
        
        # Your existing retrieval logic
        vector_results = perform_vector_search(query, k=10)
        keyword_results = perform_keyword_search(query, k=10)
        
        # Hybrid ranking
        ranked_results = hybrid_rank(vector_results, keyword_results)
        
        # Update monitoring with results
        execution.response_text = f"Retrieved {len(ranked_results)} results"
        execution.metadata.update({
            "vector_results_count": len(vector_results),
            "keyword_results_count": len(keyword_results),
            "final_results_count": len(ranked_results),
            "retrieval_method": "hybrid"
        })
        
        # Calculate relevance score
        relevance_score = calculate_retrieval_relevance(query, ranked_results)
        execution.response_quality_score = relevance_score
        execution.success = True
        
        return ranked_results

# Example usage
query = "What are the latest developments in hybrid retrieval systems?"
config = {
    "vector_model": "text-embedding-3-large",
    "keyword_weight": 0.3,
    "vector_weight": 0.7
}

results = enhanced_retrieval_workflow(query, config)
print(f"Retrieved {len(results)} results with monitoring")
```

### Multi-Stage Workflow Monitoring

Monitor complex workflows with multiple LLM interactions:

```python
# File: examples/lethe_multistage_integration.py
from src.monitoring import LethePromptMonitor
import asyncio

class EnhancedResearchWorkflow:
    def __init__(self):
        self.monitor = LethePromptMonitor(
            experiment_name="multistage_research",
            project_id="lethe_advanced"
        )
    
    async def run_research_pipeline(self, research_question):
        """Complete research pipeline with monitoring at each stage."""
        
        pipeline_id = f"research_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Stage 1: Query Analysis and Expansion
        expanded_queries = await self.analyze_and_expand_query(
            research_question, pipeline_id
        )
        
        # Stage 2: Multi-source Retrieval
        retrieved_docs = await self.retrieve_from_multiple_sources(
            expanded_queries, pipeline_id
        )
        
        # Stage 3: Synthesis and Analysis
        synthesis = await self.synthesize_findings(
            research_question, retrieved_docs, pipeline_id
        )
        
        # Stage 4: Quality Assessment and Refinement
        final_result = await self.assess_and_refine(
            synthesis, pipeline_id
        )
        
        return final_result
    
    async def analyze_and_expand_query(self, query, pipeline_id):
        """Stage 1: Query analysis and expansion with monitoring."""
        
        with self.monitor.track_prompt_execution(
            prompt_id=f"{pipeline_id}_query_analysis",
            prompt_text=f"Analyze and expand research query: {query}",
            model_config={"model": "gpt-4", "temperature": 0.3},
            stage="query_expansion",
            metadata={"pipeline_id": pipeline_id, "stage": 1}
        ) as execution:
            
            # Simulate query expansion logic
            expanded = await expand_research_query(query)
            
            execution.response_text = str(expanded)
            execution.metadata["expanded_query_count"] = len(expanded)
            execution.response_quality_score = 0.9
            execution.success = True
            
            return expanded
    
    async def retrieve_from_multiple_sources(self, queries, pipeline_id):
        """Stage 2: Multi-source retrieval with monitoring."""
        
        all_docs = []
        
        for i, query in enumerate(queries):
            with self.monitor.track_prompt_execution(
                prompt_id=f"{pipeline_id}_retrieval_{i}",
                prompt_text=query,
                model_config={"retrieval_method": "hybrid", "top_k": 20},
                stage="retrieval",
                metadata={"pipeline_id": pipeline_id, "stage": 2, "query_index": i}
            ) as execution:
                
                # Simulate retrieval
                docs = await retrieve_documents(query)
                all_docs.extend(docs)
                
                execution.response_text = f"Retrieved {len(docs)} documents"
                execution.metadata["documents_retrieved"] = len(docs)
                execution.response_quality_score = calculate_retrieval_quality(docs)
                execution.success = True
        
        return all_docs
    
    async def synthesize_findings(self, original_question, documents, pipeline_id):
        """Stage 3: Synthesis with monitoring."""
        
        with self.monitor.track_prompt_execution(
            prompt_id=f"{pipeline_id}_synthesis",
            prompt_text=f"Synthesize findings for: {original_question}",
            model_config={"model": "gpt-4", "temperature": 0.2, "max_tokens": 2000},
            stage="synthesis",
            metadata={
                "pipeline_id": pipeline_id,
                "stage": 3,
                "document_count": len(documents)
            }
        ) as execution:
            
            # Simulate synthesis
            synthesis = await synthesize_documents(original_question, documents)
            
            execution.response_text = synthesis
            execution.metadata.update({
                "synthesis_length": len(synthesis),
                "sources_cited": count_citations(synthesis)
            })
            execution.response_quality_score = assess_synthesis_quality(synthesis)
            execution.success = True
            
            return synthesis

# Usage
async def main():
    workflow = EnhancedResearchWorkflow()
    
    result = await workflow.run_research_pipeline(
        "How do hybrid retrieval systems compare to pure vector search?"
    )
    
    print("Research pipeline complete with full monitoring")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🧪 MLflow Integration

### Basic MLflow Setup

Integrate with MLflow for experiment tracking:

```python
# File: examples/mlflow_integration.py
import mlflow
from src.monitoring import track_prompt, get_prompt_tracker

# Configure MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("prompt_optimization")

def run_prompt_experiment(prompt_variant, model_config):
    """Run a prompt experiment with MLflow and monitoring integration."""
    
    with mlflow.start_run(run_name=f"prompt_v{prompt_variant}"):
        # Log MLflow parameters
        mlflow.log_params(model_config)
        mlflow.log_param("prompt_variant", prompt_variant)
        
        # Track with prompt monitoring
        with track_prompt(
            prompt_id=f"prompt_v{prompt_variant}",
            prompt_text=get_prompt_text(prompt_variant),
            model_config=model_config,
            tags=["mlflow", "experiment", f"variant-{prompt_variant}"],
            metadata={"mlflow_run_id": mlflow.active_run().info.run_id}
        ) as execution:
            
            # Run the experiment
            response = run_llm_call(execution.prompt_text, model_config)
            quality_score = evaluate_response(response)
            
            # Update monitoring
            execution.response_text = response
            execution.response_quality_score = quality_score
            execution.success = True
            
            # Log MLflow metrics
            mlflow.log_metrics({
                "quality_score": quality_score,
                "execution_time_ms": execution.execution_time_ms,
                "tokens_used": execution.tokens_used,
                "response_length": len(response)
            })
            
            # Log artifacts
            mlflow.log_text(response, "response.txt")
            mlflow.log_text(execution.prompt_text, "prompt.txt")
            
            return {
                "quality_score": quality_score,
                "execution_time": execution.execution_time_ms,
                "mlflow_run_id": mlflow.active_run().info.run_id,
                "monitoring_id": execution.execution_id
            }

# Run multiple experiments
experiments = [
    {"variant": 1, "config": {"model": "gpt-4", "temperature": 0.1}},
    {"variant": 2, "config": {"model": "gpt-4", "temperature": 0.7}},
    {"variant": 3, "config": {"model": "gpt-3.5-turbo", "temperature": 0.3}},
]

results = []
for exp in experiments:
    result = run_prompt_experiment(exp["variant"], exp["config"])
    results.append(result)
    print(f"Experiment {exp['variant']} - Quality: {result['quality_score']:.3f}")

# Analyze results across both systems
print("\n📊 Cross-system Analysis:")
tracker = get_prompt_tracker()
recent_executions = tracker.get_recent_executions(limit=len(experiments))

for execution in recent_executions:
    mlflow_id = execution['metadata'].get('mlflow_run_id', 'Unknown')
    print(f"Monitoring: {execution['execution_id']} ↔ MLflow: {mlflow_id}")
```

### Advanced MLflow Integration

Create a comprehensive integration with MLflow experiments:

```python
# File: examples/mlflow_advanced_integration.py
import mlflow
from mlflow.tracking import MlflowClient
from src.monitoring import LethePromptMonitor, PromptVersionManager
import pandas as pd

class MLflowPromptExperiment:
    """Advanced MLflow integration for prompt experiments."""
    
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.monitor = LethePromptMonitor(experiment_name)
        self.version_manager = PromptVersionManager()
        self.client = MlflowClient()
        
        # Set up MLflow experiment
        mlflow.set_experiment(experiment_name)
    
    def run_ab_test(self, prompt_versions, test_queries, model_config):
        """Run A/B test comparing multiple prompt versions."""
        
        results = {}
        
        for version_id, prompt_template in prompt_versions.items():
            version_results = []
            
            with mlflow.start_run(run_name=f"version_{version_id}"):
                # Log version metadata
                mlflow.log_param("prompt_version", version_id)
                mlflow.log_params(model_config)
                mlflow.log_text(prompt_template, f"prompt_template_v{version_id}.txt")
                
                for i, query in enumerate(test_queries):
                    # Format prompt with query
                    full_prompt = prompt_template.format(query=query)
                    
                    # Track with monitoring system
                    with self.monitor.track_prompt_execution(
                        prompt_id=f"ab_test_v{version_id}_q{i}",
                        prompt_text=full_prompt,
                        model_config=model_config,
                        stage="ab_testing",
                        metadata={
                            "prompt_version": version_id,
                            "query_index": i,
                            "mlflow_run_id": mlflow.active_run().info.run_id
                        }
                    ) as execution:
                        
                        # Run the actual LLM call
                        response = self.call_llm(full_prompt, model_config)
                        
                        # Evaluate response
                        metrics = self.evaluate_response(query, response)
                        
                        # Update monitoring
                        execution.response_text = response
                        execution.response_quality_score = metrics["quality_score"]
                        execution.success = True
                        execution.metadata.update(metrics)
                        
                        version_results.append({
                            "query_index": i,
                            "quality_score": metrics["quality_score"],
                            "execution_time": execution.execution_time_ms,
                            "tokens_used": execution.tokens_used,
                            **metrics
                        })
                
                # Aggregate metrics for this version
                avg_quality = sum(r["quality_score"] for r in version_results) / len(version_results)
                avg_time = sum(r["execution_time"] for r in version_results) / len(version_results)
                total_tokens = sum(r["tokens_used"] for r in version_results)
                
                # Log aggregated metrics to MLflow
                mlflow.log_metrics({
                    "avg_quality_score": avg_quality,
                    "avg_execution_time_ms": avg_time,
                    "total_tokens_used": total_tokens,
                    "queries_processed": len(test_queries)
                })
                
                results[version_id] = {
                    "avg_quality": avg_quality,
                    "avg_time": avg_time,
                    "total_tokens": total_tokens,
                    "individual_results": version_results,
                    "mlflow_run_id": mlflow.active_run().info.run_id
                }
        
        return results
    
    def analyze_experiment_results(self, results):
        """Analyze A/B test results using both MLflow and monitoring data."""
        
        print("📊 A/B Test Results Analysis")
        print("=" * 50)
        
        # Performance comparison
        performance_data = []
        for version_id, data in results.items():
            performance_data.append({
                "Version": version_id,
                "Avg Quality": data["avg_quality"],
                "Avg Time (ms)": data["avg_time"],
                "Total Tokens": data["total_tokens"]
            })
        
        df = pd.DataFrame(performance_data)
        print(df.to_string(index=False))
        
        # Statistical significance testing
        self.perform_significance_tests(results)
        
        # Cross-reference with monitoring data
        self.cross_reference_monitoring_data(results)
    
    def perform_significance_tests(self, results):
        """Perform statistical significance tests on A/B results."""
        from scipy import stats
        
        print("\n📈 Statistical Significance Analysis:")
        
        version_ids = list(results.keys())
        if len(version_ids) >= 2:
            for i in range(len(version_ids)):
                for j in range(i + 1, len(version_ids)):
                    v1, v2 = version_ids[i], version_ids[j]
                    
                    # Extract quality scores
                    scores1 = [r["quality_score"] for r in results[v1]["individual_results"]]
                    scores2 = [r["quality_score"] for r in results[v2]["individual_results"]]
                    
                    # T-test
                    t_stat, p_value = stats.ttest_ind(scores1, scores2)
                    
                    significance = "✅ Significant" if p_value < 0.05 else "❌ Not significant"
                    print(f"  {v1} vs {v2}: p-value = {p_value:.4f} ({significance})")
    
    def cross_reference_monitoring_data(self, results):
        """Cross-reference MLflow results with monitoring system data."""
        
        print("\n🔗 Cross-system Data Verification:")
        
        tracker = self.monitor.tracker
        
        for version_id, data in results.items():
            # Get monitoring data for this version
            executions = tracker.get_executions_by_tags([f"version-{version_id}"])
            
            if executions:
                monitoring_avg_quality = sum(e["response_quality_score"] or 0 for e in executions) / len(executions)
                mlflow_avg_quality = data["avg_quality"]
                
                diff = abs(monitoring_avg_quality - mlflow_avg_quality)
                match_status = "✅ Match" if diff < 0.001 else f"⚠️ Difference: {diff:.3f}"
                
                print(f"  Version {version_id}: {match_status}")
                print(f"    MLflow: {mlflow_avg_quality:.3f} | Monitoring: {monitoring_avg_quality:.3f}")

# Usage example
def main():
    experiment = MLflowPromptExperiment("prompt_optimization_advanced")
    
    # Define prompt versions to test
    prompt_versions = {
        "v1": "Answer the following question concisely: {query}",
        "v2": "Provide a detailed analysis of: {query}",
        "v3": "As an expert researcher, please address: {query}"
    }
    
    # Test queries
    test_queries = [
        "What are the advantages of hybrid search?",
        "How does vector similarity work in information retrieval?",
        "What are the limitations of keyword-based search?"
    ]
    
    # Model configuration
    model_config = {
        "model": "gpt-4",
        "temperature": 0.3,
        "max_tokens": 500
    }
    
    # Run A/B test
    results = experiment.run_ab_test(prompt_versions, test_queries, model_config)
    
    # Analyze results
    experiment.analyze_experiment_results(results)

if __name__ == "__main__":
    main()
```

## 🔄 A/B Testing Workflows

### Systematic Prompt Comparison

Create robust A/B testing workflows:

```python
# File: examples/ab_testing_workflow.py
from src.monitoring import PromptComparison, track_prompt
import random
from scipy import stats

class PromptABTester:
    """Systematic A/B testing for prompt optimization."""
    
    def __init__(self, test_name):
        self.test_name = test_name
        self.results = {"control": [], "variant": []}
    
    def run_ab_test(self, control_prompt, variant_prompt, test_cases, model_config):
        """Run A/B test with statistical rigor."""
        
        print(f"🧪 Starting A/B Test: {self.test_name}")
        print(f"📊 Test cases: {len(test_cases)}")
        
        # Randomize test case assignment
        random.shuffle(test_cases)
        midpoint = len(test_cases) // 2
        
        control_cases = test_cases[:midpoint]
        variant_cases = test_cases[midpoint:]
        
        # Run control group
        print(f"\n🅰️ Running Control Group ({len(control_cases)} cases)")
        for i, case in enumerate(control_cases):
            result = self.run_single_test(
                prompt=control_prompt.format(**case),
                case_data=case,
                group="control",
                case_index=i,
                model_config=model_config
            )
            self.results["control"].append(result)
        
        # Run variant group
        print(f"\n🅱️ Running Variant Group ({len(variant_cases)} cases)")
        for i, case in enumerate(variant_cases):
            result = self.run_single_test(
                prompt=variant_prompt.format(**case),
                case_data=case,
                group="variant",
                case_index=i,
                model_config=model_config
            )
            self.results["variant"].append(result)
        
        # Analyze results
        return self.analyze_results()
    
    def run_single_test(self, prompt, case_data, group, case_index, model_config):
        """Run a single test case with monitoring."""
        
        with track_prompt(
            prompt_id=f"{self.test_name}_{group}_{case_index}",
            prompt_text=prompt,
            model_config=model_config,
            tags=["ab_test", self.test_name, group],
            metadata={
                "test_name": self.test_name,
                "group": group,
                "case_index": case_index,
                **case_data
            }
        ) as execution:
            
            # Simulate LLM call
            response = simulate_llm_response(prompt, model_config)
            
            # Evaluate response
            metrics = evaluate_test_response(case_data, response)
            
            # Update execution
            execution.response_text = response
            execution.response_quality_score = metrics["quality_score"]
            execution.success = True
            execution.metadata.update(metrics)
            
            return {
                "execution_id": execution.execution_id,
                "quality_score": metrics["quality_score"],
                "execution_time": execution.execution_time_ms,
                "response_length": len(response),
                "case_data": case_data,
                **metrics
            }
    
    def analyze_results(self):
        """Perform statistical analysis of A/B test results."""
        
        print(f"\n📊 A/B Test Analysis: {self.test_name}")
        print("=" * 50)
        
        # Basic statistics
        control_scores = [r["quality_score"] for r in self.results["control"]]
        variant_scores = [r["quality_score"] for r in self.results["variant"]]
        
        control_mean = sum(control_scores) / len(control_scores)
        variant_mean = sum(variant_scores) / len(variant_scores)
        
        print(f"📈 Results Summary:")
        print(f"  Control Group:  {len(control_scores)} samples, mean = {control_mean:.3f}")
        print(f"  Variant Group:  {len(variant_scores)} samples, mean = {variant_mean:.3f}")
        print(f"  Difference:     {variant_mean - control_mean:+.3f} ({((variant_mean/control_mean - 1) * 100):+.1f}%)")
        
        # Statistical significance
        t_stat, p_value = stats.ttest_ind(control_scores, variant_scores)
        
        if p_value < 0.05:
            significance = "✅ Statistically significant"
            recommendation = "🚀 Deploy variant" if variant_mean > control_mean else "❌ Keep control"
        else:
            significance = "❌ Not statistically significant"
            recommendation = "🔄 Continue testing or refine variant"
        
        print(f"\n🧮 Statistical Analysis:")
        print(f"  T-statistic:    {t_stat:.3f}")
        print(f"  P-value:        {p_value:.4f}")
        print(f"  Significance:   {significance}")
        print(f"  Recommendation: {recommendation}")
        
        # Performance comparison
        control_time = sum(r["execution_time"] for r in self.results["control"]) / len(self.results["control"])
        variant_time = sum(r["execution_time"] for r in self.results["variant"]) / len(self.results["variant"])
        
        print(f"\n⚡ Performance Comparison:")
        print(f"  Control avg time: {control_time:.0f}ms")
        print(f"  Variant avg time: {variant_time:.0f}ms")
        print(f"  Time difference:  {variant_time - control_time:+.0f}ms")
        
        return {
            "control_mean": control_mean,
            "variant_mean": variant_mean,
            "improvement": variant_mean - control_mean,
            "improvement_percent": ((variant_mean/control_mean - 1) * 100),
            "p_value": p_value,
            "statistically_significant": p_value < 0.05,
            "recommendation": recommendation,
            "sample_sizes": {"control": len(control_scores), "variant": len(variant_scores)}
        }

# Example usage
def main():
    tester = PromptABTester("retrieval_prompt_optimization")
    
    # Define prompts to test
    control_prompt = "Find relevant documents for: {query}"
    variant_prompt = "As a research assistant, identify the most relevant documents for the query: {query}. Focus on authoritative sources and recent publications."
    
    # Test cases
    test_cases = [
        {"query": "machine learning in healthcare", "domain": "healthcare"},
        {"query": "quantum computing applications", "domain": "technology"},
        {"query": "climate change mitigation", "domain": "environment"},
        {"query": "sustainable energy solutions", "domain": "energy"},
        {"query": "artificial intelligence ethics", "domain": "ethics"},
        {"query": "blockchain in finance", "domain": "finance"}
    ]
    
    model_config = {
        "model": "gpt-4",
        "temperature": 0.3,
        "max_tokens": 200
    }
    
    # Run A/B test
    results = tester.run_ab_test(control_prompt, variant_prompt, test_cases, model_config)
    
    print(f"\n✅ A/B Test Complete!")
    if results["statistically_significant"]:
        print(f"🎉 Significant improvement of {results['improvement_percent']:.1f}%")
    else:
        print(f"📊 Continue testing - current sample size may be insufficient")

if __name__ == "__main__":
    main()
```

## 🚀 Production Integration

### Production Monitoring Setup

Set up monitoring for production research environments:

```python
# File: examples/production_monitoring.py
from src.monitoring import LethePromptMonitor
import logging
from contextlib import contextmanager

class ProductionMonitor:
    """Production-ready monitoring with error handling and performance optimization."""
    
    def __init__(self, service_name, environment="production"):
        self.monitor = LethePromptMonitor(
            experiment_name=f"{service_name}_{environment}",
            project_id=f"lethe_production_{service_name}"
        )
        self.logger = logging.getLogger(f"production_monitor_{service_name}")
        
        # Configure production logging
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    @contextmanager
    def track_production_prompt(self, prompt_id, prompt_text, model_config, 
                               user_id=None, session_id=None, **kwargs):
        """Production-safe prompt tracking with comprehensive error handling."""
        
        try:
            # Sanitize sensitive data
            safe_prompt = self.sanitize_prompt(prompt_text)
            safe_config = self.sanitize_config(model_config)
            
            with self.monitor.track_prompt_execution(
                prompt_id=prompt_id,
                prompt_text=safe_prompt,
                model_config=safe_config,
                stage="production",
                metadata={
                    "user_id": user_id,
                    "session_id": session_id,
                    "environment": "production",
                    **kwargs
                }
            ) as execution:
                
                self.logger.info(f"Started tracking: {prompt_id}")
                yield execution
                self.logger.info(f"Completed tracking: {prompt_id} - {execution.execution_time_ms}ms")
                
        except Exception as e:
            self.logger.error(f"Monitoring error for {prompt_id}: {str(e)}")
            # Create minimal execution record for error tracking
            error_execution = type('obj', (object,), {
                'execution_id': f"{prompt_id}_error",
                'success': False,
                'error_message': str(e),
                'execution_time_ms': 0
            })()
            yield error_execution
    
    def sanitize_prompt(self, prompt_text):
        """Remove or mask sensitive information from prompts."""
        # Implement your data sanitization logic
        # This is a basic example - customize for your needs
        import re
        
        # Remove potential PII patterns
        sanitized = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN-MASKED]', prompt_text)
        sanitized = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL-MASKED]', sanitized)
        
        return sanitized
    
    def sanitize_config(self, config):
        """Remove sensitive configuration data."""
        safe_config = config.copy()
        
        # Remove API keys and tokens
        sensitive_keys = ['api_key', 'token', 'secret', 'password']
        for key in sensitive_keys:
            if key in safe_config:
                safe_config[key] = '[REDACTED]'
        
        return safe_config
    
    def get_production_metrics(self, time_window_hours=24):
        """Get production metrics for monitoring dashboards."""
        
        from datetime import datetime, timedelta
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_window_hours)
        
        # Get metrics from monitoring system
        tracker = self.monitor.tracker
        executions = tracker.get_executions_by_date_range(start_time, end_time)
        
        if not executions:
            return {"error": "No data available for time window"}
        
        # Calculate key metrics
        total_requests = len(executions)
        successful_requests = sum(1 for e in executions if e.get('success', False))
        error_rate = (total_requests - successful_requests) / total_requests
        
        response_times = [e.get('execution_time_ms', 0) for e in executions if e.get('success')]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        quality_scores = [e.get('response_quality_score', 0) for e in executions 
                         if e.get('response_quality_score') is not None and e.get('success')]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            "time_window_hours": time_window_hours,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "error_rate": error_rate,
            "avg_response_time_ms": avg_response_time,
            "avg_quality_score": avg_quality,
            "p95_response_time": self.calculate_percentile(response_times, 95) if response_times else 0,
            "requests_per_hour": total_requests / time_window_hours
        }
    
    def calculate_percentile(self, values, percentile):
        """Calculate percentile of a list of values."""
        if not values:
            return 0
        
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]

# Example production service
class ProductionResearchService:
    def __init__(self):
        self.monitor = ProductionMonitor("research_service")
    
    async def process_research_query(self, query, user_id, session_id):
        """Production research query processing with monitoring."""
        
        with self.monitor.track_production_prompt(
            prompt_id="research_query",
            prompt_text=query,
            model_config={"model": "gpt-4", "temperature": 0.3},
            user_id=user_id,
            session_id=session_id,
            query_type="research"
        ) as execution:
            
            try:
                # Your production logic here
                results = await self.perform_research(query)
                
                # Update monitoring
                execution.response_text = f"Found {len(results)} results"
                execution.response_quality_score = self.assess_results_quality(results)
                execution.success = True
                
                return results
                
            except Exception as e:
                # Error is automatically tracked by the monitor
                self.monitor.logger.error(f"Research query failed: {str(e)}")
                raise
    
    def get_health_metrics(self):
        """Get service health metrics for monitoring."""
        return self.monitor.get_production_metrics(time_window_hours=1)

# Usage
async def main():
    service = ProductionResearchService()
    
    # Process query with monitoring
    try:
        results = await service.process_research_query(
            query="What are the latest developments in quantum computing?",
            user_id="user_123",
            session_id="session_456"
        )
        print(f"✅ Query processed successfully: {len(results)} results")
    except Exception as e:
        print(f"❌ Query failed: {str(e)}")
    
    # Get health metrics
    metrics = service.get_health_metrics()
    print(f"📊 Service Health: {metrics['error_rate']:.2%} error rate, {metrics['avg_response_time_ms']:.0f}ms avg response")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## 🎯 Next Steps

1. **Advanced Analytics Tutorial** → Learn statistical analysis and custom metrics
2. **Dashboard Deep Dive** → Master visualization and reporting features  
3. **Custom Extensions** → Build domain-specific monitoring capabilities
4. **Performance Optimization** → Scale monitoring for high-volume environments

## 💡 Best Practices

- **Tag Consistently**: Use standardized tags for filtering and analysis
- **Monitor Error Rates**: Track and alert on unusual error patterns
- **Version Control Prompts**: Use PromptVersionManager for systematic versioning
- **Sanitize Data**: Remove PII and sensitive information in production
- **Set Quality Thresholds**: Define and monitor quality score targets
- **Batch Analysis**: Use statistical methods for comparing prompt variants

---

Your monitoring system is now integrated with your workflows and ready for systematic prompt optimization!