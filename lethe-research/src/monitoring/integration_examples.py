#!/usr/bin/env python3
"""
Prompt Monitoring Integration Examples

Demonstrates how to integrate the prompt tracking system with existing
Lethe components and research workflows.
"""

import time
import json
from typing import Dict, List, Any, Optional
from contextlib import contextmanager

from .prompt_tracker import track_prompt, get_analytics, compare_prompts
from ..common.data_persistence import DataManager
from ..fusion.telemetry import FusionTelemetry


class LethePromptMonitor:
    """Enhanced prompt monitoring for Lethe research workflows."""
    
    def __init__(self):
        """Initialize with integration to existing Lethe infrastructure."""
        self.data_manager = DataManager()
    
    @contextmanager
    def monitor_retrieval_prompt(
        self,
        query: str,
        retrieval_config: Dict[str, Any],
        experiment_name: str = "lethe_retrieval"
    ):
        """Monitor retrieval prompts with before/after analysis."""
        
        # Create prompt ID from query and config
        prompt_id = f"retrieval_{hash(query + str(sorted(retrieval_config.items())))}"
        
        # Prepare model config for tracking
        model_config = {
            "model": retrieval_config.get("model", "lethe"),
            "version": retrieval_config.get("version", "1.0"),
            "parameters": retrieval_config
        }
        
        # Track the execution
        with track_prompt(
            prompt_id=prompt_id,
            prompt_text=query,
            model_config=model_config,
            experiment_tag=experiment_name,
            conversation_turn=1
        ) as execution:
            
            # Store original state for before/after comparison
            execution.prompt_variables = {
                "retrieval_method": retrieval_config.get("method", "hybrid"),
                "fusion_params": retrieval_config.get("fusion_params", {}),
                "rerank_enabled": retrieval_config.get("rerank", False)
            }
            
            yield execution
    
    def track_fusion_execution(
        self,
        query: str,
        lexical_results: List[Dict],
        semantic_results: List[Dict],
        fused_results: List[Dict],
        fusion_params: Dict[str, Any],
        performance_metrics: Dict[str, float]
    ) -> str:
        """Track fusion component execution with detailed metrics."""
        
        prompt_id = f"fusion_{hash(query)}"
        
        model_config = {
            "model": "lethe_fusion",
            "version": "1.0",
            "parameters": fusion_params
        }
        
        with track_prompt(
            prompt_id=prompt_id,
            prompt_text=query,
            model_config=model_config,
            experiment_tag="fusion_analysis"
        ) as execution:
            
            # Detailed before/after state
            execution.prompt_variables = {
                "lexical_count": len(lexical_results),
                "semantic_count": len(semantic_results),
                "fusion_method": fusion_params.get("method", "rrf"),
                "alpha": fusion_params.get("alpha", 0.5)
            }
            
            # Response analysis
            execution.response_text = json.dumps({
                "fused_results": fused_results[:5],  # Top 5 for analysis
                "result_count": len(fused_results)
            }, indent=2)
            
            execution.response_length = len(fused_results)
            execution.memory_usage_mb = performance_metrics.get("memory_mb", 0)
            execution.execution_time_ms = performance_metrics.get("latency_ms", 0)
            
            # Quality metrics from fusion
            execution.response_quality_score = performance_metrics.get("ndcg_10", 0)
            execution.coherence_score = performance_metrics.get("mrr", 0)
            
        return execution.execution_id
    
    def compare_retrieval_methods(
        self,
        query: str,
        baseline_config: Dict[str, Any],
        treatment_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two retrieval configurations with detailed analysis."""
        
        results = {
            "query": query,
            "baseline_execution": None,
            "treatment_execution": None,
            "comparison": None
        }
        
        # Execute baseline
        with self.monitor_retrieval_prompt(
            query, baseline_config, "baseline_comparison"
        ) as baseline_exec:
            # Simulate baseline execution
            time.sleep(0.1)  # Simulate processing
            baseline_exec.response_text = "Baseline retrieval results"
            baseline_exec.response_quality_score = 0.75
            results["baseline_execution"] = baseline_exec.execution_id
        
        # Execute treatment
        with self.monitor_retrieval_prompt(
            query, treatment_config, "treatment_comparison" 
        ) as treatment_exec:
            # Simulate treatment execution
            time.sleep(0.08)  # Simulate faster processing
            treatment_exec.response_text = "Treatment retrieval results"
            treatment_exec.response_quality_score = 0.82
            results["treatment_execution"] = treatment_exec.execution_id
        
        # Compare results
        comparison = compare_prompts(
            results["baseline_execution"],
            results["treatment_execution"],
            notes=f"Retrieval method comparison for query: {query[:50]}..."
        )
        
        results["comparison"] = {
            "quality_improvement": comparison.quality_improvement,
            "performance_change": comparison.performance_change_percent,
            "is_significant": comparison.is_significant
        }
        
        return results
    
    def analyze_prompt_evolution(self, prompt_id: str) -> Dict[str, Any]:
        """Analyze how a prompt has evolved over time."""
        analytics = get_analytics(prompt_id)
        
        evolution_analysis = {
            "prompt_id": prompt_id,
            "basic_analytics": analytics,
            "evolution_insights": []
        }
        
        if analytics.get("total_executions", 0) > 1:
            # Performance evolution
            perf_trend = analytics.get("performance_trend", "stable")
            evolution_analysis["evolution_insights"].append({
                "aspect": "performance",
                "trend": perf_trend,
                "recommendation": self._get_performance_recommendation(perf_trend)
            })
            
            # Quality evolution
            quality_trend = analytics.get("quality_trend")
            if quality_trend:
                evolution_analysis["evolution_insights"].append({
                    "aspect": "quality",
                    "trend": quality_trend,
                    "recommendation": self._get_quality_recommendation(quality_trend)
                })
        
        return evolution_analysis
    
    def _get_performance_recommendation(self, trend: str) -> str:
        """Get recommendation based on performance trend."""
        if trend == "improving":
            return "Performance is improving - continue current optimizations"
        elif trend == "declining":
            return "Performance is declining - investigate recent changes"
        else:
            return "Performance is stable - consider optimization opportunities"
    
    def _get_quality_recommendation(self, trend: str) -> str:
        """Get recommendation based on quality trend."""
        if trend == "improving":
            return "Quality is improving - document successful changes"
        elif trend == "declining":
            return "Quality is declining - review recent prompt modifications"
        else:
            return "Quality is stable - consider A/B testing new variations"
    
    def generate_research_report(self, experiment_tag: str) -> Dict[str, Any]:
        """Generate comprehensive research report for an experiment."""
        # This would integrate with the existing analysis framework
        report = {
            "experiment_tag": experiment_tag,
            "summary": "Prompt monitoring research report",
            "methodology": {
                "tracking_enabled": True,
                "metrics_collected": [
                    "execution_time", "response_length", "quality_scores",
                    "error_rates", "model_parameters"
                ],
                "comparison_methods": ["A/B testing", "time-series analysis"]
            },
            "key_findings": [],
            "recommendations": []
        }
        
        # Add integration with existing statistical analysis
        report["statistical_framework"] = {
            "hypothesis_testing": "Available via unified analysis framework",
            "confidence_intervals": "Bootstrap methods applied",
            "multiple_comparisons": "Bonferroni correction applied"
        }
        
        return report


class PromptVersionManager:
    """Manages prompt versions with automatic change detection."""
    
    def __init__(self):
        """Initialize version manager."""
        self.monitor = LethePromptMonitor()
    
    def create_prompt_version(
        self,
        prompt_id: str,
        prompt_text: str,
        version: str,
        change_description: str,
        created_by: str = "system"
    ) -> Dict[str, Any]:
        """Create a new prompt version with change tracking."""
        
        version_info = {
            "prompt_id": prompt_id,
            "version": version,
            "prompt_text": prompt_text,
            "change_description": change_description,
            "created_by": created_by,
            "changes_detected": []
        }
        
        # Detect changes from previous versions
        previous_analytics = get_analytics(prompt_id)
        
        if previous_analytics.get("total_executions", 0) > 0:
            # Compare with previous performance
            version_info["changes_detected"].extend([
                "New version created",
                f"Previous executions: {previous_analytics['total_executions']}",
                f"Previous success rate: {previous_analytics.get('success_rate', 0):.1f}%"
            ])
        
        return version_info
    
    def validate_prompt_change(
        self,
        prompt_id: str,
        old_text: str,
        new_text: str
    ) -> Dict[str, Any]:
        """Validate prompt changes and predict impact."""
        
        # Simple change detection
        changes = []
        
        if len(new_text) != len(old_text):
            length_change = len(new_text) - len(old_text)
            changes.append(f"Length changed by {length_change:+d} characters")
        
        # Word-level changes (simplified)
        old_words = set(old_text.lower().split())
        new_words = set(new_text.lower().split())
        
        added_words = new_words - old_words
        removed_words = old_words - new_words
        
        if added_words:
            changes.append(f"Added words: {', '.join(list(added_words)[:5])}")
        if removed_words:
            changes.append(f"Removed words: {', '.join(list(removed_words)[:5])}")
        
        validation = {
            "prompt_id": prompt_id,
            "changes_detected": changes,
            "impact_prediction": "Medium" if changes else "Low",
            "recommendation": "Test with small sample before full deployment" if changes else "Low risk change"
        }
        
        return validation


# Integration examples and usage patterns
def example_integration_workflow():
    """Example of integrating prompt monitoring with research workflow."""
    
    monitor = LethePromptMonitor()
    
    # Example 1: Monitor retrieval pipeline
    print("🔍 Example 1: Retrieval Pipeline Monitoring")
    
    query = "What are the benefits of hybrid retrieval systems?"
    baseline_config = {"method": "bm25", "k": 50}
    treatment_config = {"method": "hybrid", "alpha": 0.7, "k": 50}
    
    comparison_results = monitor.compare_retrieval_methods(
        query, baseline_config, treatment_config
    )
    
    print(f"Comparison Results: {comparison_results['comparison']}")
    
    # Example 2: Track fusion component
    print("\n🔧 Example 2: Fusion Component Tracking")
    
    fusion_execution_id = monitor.track_fusion_execution(
        query=query,
        lexical_results=[{"doc_id": f"lex_{i}", "score": 0.8 - i*0.1} for i in range(5)],
        semantic_results=[{"doc_id": f"sem_{i}", "score": 0.9 - i*0.1} for i in range(5)],
        fused_results=[{"doc_id": f"fused_{i}", "score": 0.95 - i*0.1} for i in range(5)],
        fusion_params={"method": "rrf", "alpha": 0.6},
        performance_metrics={"latency_ms": 150, "memory_mb": 45.2, "ndcg_10": 0.85}
    )
    
    print(f"Fusion execution tracked: {fusion_execution_id}")
    
    # Example 3: Analyze prompt evolution
    print("\n📈 Example 3: Prompt Evolution Analysis")
    
    evolution = monitor.analyze_prompt_evolution("retrieval_example")
    print(f"Evolution insights: {evolution}")
    
    # Example 4: Version management
    print("\n📝 Example 4: Prompt Version Management")
    
    version_mgr = PromptVersionManager()
    
    validation = version_mgr.validate_prompt_change(
        prompt_id="retrieval_example",
        old_text="What are the benefits of retrieval systems?",
        new_text="What are the key benefits and limitations of hybrid retrieval systems?"
    )
    
    print(f"Change validation: {validation}")


if __name__ == "__main__":
    example_integration_workflow()