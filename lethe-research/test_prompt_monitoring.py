#!/usr/bin/env python3
"""
Test Suite for Lethe Prompt Monitoring System

Comprehensive tests and demonstrations of the prompt tracking capabilities,
including integration with existing Lethe infrastructure.
"""

import json
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.monitoring import (
    PromptTracker, track_prompt, get_analytics, compare_prompts
)
from src.monitoring.integration_examples import LethePromptMonitor


def test_basic_prompt_tracking():
    """Test basic prompt tracking functionality."""
    print("🧪 Testing Basic Prompt Tracking")
    print("=" * 50)
    
    tracker = PromptTracker()
    
    # Test prompt execution tracking
    model_config = {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    with track_prompt(
        prompt_id="test_basic",
        prompt_text="What are the advantages of hybrid retrieval systems?",
        model_config=model_config,
        experiment_tag="basic_test"
    ) as execution:
        
        # Simulate processing time
        time.sleep(0.1)
        
        # Simulate response
        execution.response_text = "Hybrid retrieval systems combine lexical and semantic approaches..."
        execution.response_quality_score = 0.85
        execution.coherence_score = 0.9
    
    print(f"✅ Tracked execution: {execution.execution_id}")
    print(f"📝 Prompt ID: {execution.prompt_id}")
    print(f"⚡ Execution time: {execution.execution_time_ms:.1f}ms")
    print(f"🎯 Quality score: {execution.response_quality_score}")


def test_error_tracking():
    """Test error tracking capabilities."""
    print("\n🚨 Testing Error Tracking")
    print("=" * 50)
    
    model_config = {"model": "test-model", "temperature": 0.5}
    
    try:
        with track_prompt(
            prompt_id="test_error",
            prompt_text="This prompt will cause an error",
            model_config=model_config
        ) as execution:
            
            # Simulate an error
            raise ValueError("Simulated processing error")
            
    except ValueError as e:
        print(f"✅ Error tracked successfully: {e}")
        print(f"❌ Error recorded in execution: {execution.execution_id}")


def test_comparison_functionality():
    """Test prompt comparison capabilities."""
    print("\n⚖️ Testing Prompt Comparison")
    print("=" * 50)
    
    # Create baseline execution
    baseline_config = {"model": "baseline-model", "temperature": 0.3}
    treatment_config = {"model": "improved-model", "temperature": 0.7}
    
    prompt_text = "Explain the concept of information retrieval"
    
    baseline_id = None
    treatment_id = None
    
    # Baseline execution
    with track_prompt(
        prompt_id="comparison_test",
        prompt_text=prompt_text,
        model_config=baseline_config,
        ab_test_group="baseline"
    ) as baseline_exec:
        time.sleep(0.15)  # Slower baseline
        baseline_exec.response_text = "Short baseline response"
        baseline_exec.response_quality_score = 0.72
        baseline_id = baseline_exec.execution_id
    
    # Treatment execution  
    with track_prompt(
        prompt_id="comparison_test",
        prompt_text=prompt_text,
        model_config=treatment_config,
        ab_test_group="treatment"
    ) as treatment_exec:
        time.sleep(0.08)  # Faster treatment
        treatment_exec.response_text = "More comprehensive treatment response with better details"
        treatment_exec.response_quality_score = 0.89
        treatment_id = treatment_exec.execution_id
    
    # Compare executions
    comparison = compare_prompts(
        baseline_id, treatment_id,
        notes="Testing A/B comparison functionality"
    )
    
    print(f"📊 Comparison ID: {comparison.comparison_id}")
    print(f"🎯 Quality improvement: {comparison.quality_improvement:+.3f}")
    print(f"⚡ Performance change: {comparison.performance_change_percent:+.1f}%")
    print(f"📝 Length change: {comparison.length_change_percent:+.1f}%")


def test_analytics_and_trends():
    """Test analytics and trend analysis."""
    print("\n📈 Testing Analytics and Trends")
    print("=" * 50)
    
    # Generate multiple executions with trends
    prompt_id = "analytics_test"
    model_config = {"model": "trend-test", "temperature": 0.5}
    
    print("Generating test data with performance trend...")
    
    for i in range(10):
        with track_prompt(
            prompt_id=prompt_id,
            prompt_text=f"Test query {i}: What is machine learning?",
            model_config=model_config
        ) as execution:
            
            # Simulate improving performance over time
            base_time = 200 - (i * 15)  # Getting faster
            execution.execution_time_ms = base_time + random.uniform(-20, 20)
            
            # Simulate improving quality
            base_quality = 0.6 + (i * 0.03)  # Getting better
            execution.response_quality_score = base_quality + random.uniform(-0.05, 0.05)
            
            execution.response_text = f"Response {i} - quality improving over time"
            execution.memory_usage_mb = 50 + random.uniform(-5, 5)
    
    # Get analytics
    analytics = get_analytics(prompt_id)
    
    print(f"📊 Total executions: {analytics['total_executions']}")
    print(f"✅ Success rate: {analytics['success_rate']:.1f}%")
    print(f"⚡ Average time: {analytics['avg_execution_time_ms']:.1f}ms")
    print(f"📏 Average length: {analytics['avg_response_length']:.1f}")
    print(f"📈 Performance trend: {analytics['performance_trend']}")
    print(f"🎯 Quality trend: {analytics['quality_trend']}")


def test_integration_examples():
    """Test integration with Lethe components."""
    print("\n🔗 Testing Lethe Integration")
    print("=" * 50)
    
    monitor = LethePromptMonitor()
    
    # Test retrieval monitoring
    print("Testing retrieval method comparison...")
    
    query = "How do neural information retrieval systems work?"
    baseline_config = {"method": "bm25", "k": 50}
    treatment_config = {"method": "hybrid", "alpha": 0.6, "k": 50}
    
    results = monitor.compare_retrieval_methods(
        query, baseline_config, treatment_config
    )
    
    print(f"🔍 Query: {query[:50]}...")
    print(f"📊 Baseline execution: {results['baseline_execution'][:8]}...")
    print(f"📊 Treatment execution: {results['treatment_execution'][:8]}...")
    
    comparison = results['comparison']
    print(f"🎯 Quality improvement: {comparison['quality_improvement']:+.3f}")
    print(f"⚡ Performance change: {comparison['performance_change']:+.1f}%")
    
    # Test fusion tracking
    print("\nTesting fusion component tracking...")
    
    fusion_id = monitor.track_fusion_execution(
        query=query,
        lexical_results=[{"doc_id": f"lex_{i}", "score": 0.8 - i*0.1} for i in range(3)],
        semantic_results=[{"doc_id": f"sem_{i}", "score": 0.9 - i*0.1} for i in range(3)],
        fused_results=[{"doc_id": f"fused_{i}", "score": 0.95 - i*0.1} for i in range(5)],
        fusion_params={"method": "rrf", "alpha": 0.6},
        performance_metrics={
            "latency_ms": 120.5,
            "memory_mb": 42.3,
            "ndcg_10": 0.87,
            "mrr": 0.82
        }
    )
    
    print(f"🔧 Fusion execution tracked: {fusion_id[:8]}...")


def test_dashboard_data():
    """Generate diverse data for dashboard testing."""
    print("\n📊 Generating Dashboard Test Data")
    print("=" * 50)
    
    # Generate diverse prompts and executions
    prompt_types = [
        ("search_query", "search and retrieval tasks"),
        ("summarization", "text summarization tasks"),
        ("qa_system", "question answering tasks"),
        ("classification", "text classification tasks")
    ]
    
    models = [
        {"model": "gpt-4", "temperature": 0.3},
        {"model": "gpt-3.5-turbo", "temperature": 0.7},
        {"model": "claude-2", "temperature": 0.5},
        {"model": "lethe-hybrid", "temperature": 0.4}
    ]
    
    print("Generating diverse test data...")
    
    for prompt_type, description in prompt_types:
        for model in models:
            for i in range(random.randint(3, 8)):
                
                with track_prompt(
                    prompt_id=f"{prompt_type}_{model['model'].replace('-', '_')}",
                    prompt_text=f"Test prompt for {description} - variant {i}",
                    model_config=model,
                    experiment_tag=f"dashboard_test_{prompt_type}"
                ) as execution:
                    
                    # Simulate realistic execution patterns
                    base_time = {"gpt-4": 300, "gpt-3.5-turbo": 150, "claude-2": 250, "lethe-hybrid": 180}
                    execution.execution_time_ms = base_time.get(
                        model["model"], 200
                    ) + random.uniform(-50, 100)
                    
                    execution.response_length = random.randint(100, 1000)
                    execution.response_quality_score = random.uniform(0.6, 0.95)
                    execution.memory_usage_mb = random.uniform(30, 80)
                    
                    # Occasional errors
                    if random.random() < 0.05:
                        execution.error_occurred = True
                        execution.error_message = "Simulated error"
                        execution.error_type = "TestError"
                    else:
                        execution.response_text = f"Generated response for {description}"
    
    print(f"✅ Generated test data for dashboard")


def test_data_export():
    """Test data export functionality."""
    print("\n📤 Testing Data Export")
    print("=" * 50)
    
    tracker = PromptTracker()
    
    # Export in different formats
    formats = ["csv", "json"]
    
    for fmt in formats:
        try:
            filename = tracker.export_data(fmt)
            file_size = Path(filename).stat().st_size
            print(f"✅ Exported {fmt.upper()}: {filename} ({file_size} bytes)")
        except Exception as e:
            print(f"❌ Export failed for {fmt}: {e}")


def demonstrate_before_after_analysis():
    """Demonstrate before/after change analysis."""
    print("\n🔄 Demonstrating Before/After Analysis")
    print("=" * 50)
    
    # Version 1: Original prompt
    v1_config = {"model": "test-model", "version": "1.0", "temperature": 0.3}
    
    with track_prompt(
        prompt_id="version_demo",
        prompt_text="What is machine learning?",
        model_config=v1_config,
        prompt_version="1.0"
    ) as v1_exec:
        time.sleep(0.12)
        v1_exec.response_text = "Machine learning is a subset of AI..."
        v1_exec.response_quality_score = 0.75
        v1_execution_id = v1_exec.execution_id
    
    # Version 2: Improved prompt  
    v2_config = {"model": "test-model", "version": "2.0", "temperature": 0.5}
    
    with track_prompt(
        prompt_id="version_demo", 
        prompt_text="What is machine learning and how does it differ from traditional programming?",
        model_config=v2_config,
        prompt_version="2.0"
    ) as v2_exec:
        time.sleep(0.08)  # Optimized for speed
        v2_exec.response_text = "Machine learning is an advanced subset of artificial intelligence that enables systems to automatically learn and improve from experience without being explicitly programmed..."
        v2_exec.response_quality_score = 0.92
        v2_execution_id = v2_exec.execution_id
    
    # Compare versions
    comparison = compare_prompts(
        v1_execution_id, v2_execution_id,
        notes="Version comparison: original vs improved prompt"
    )
    
    print(f"📊 Version Comparison Results:")
    print(f"   🎯 Quality improvement: {comparison.quality_improvement:+.3f}")
    print(f"   ⚡ Performance change: {comparison.performance_change_percent:+.1f}%")
    print(f"   📝 Length change: {comparison.length_change_percent:+.1f}%")
    
    # Show the actual changes detected
    from src.monitoring.dashboard import PromptDashboard
    dashboard = PromptDashboard()
    
    before_after = dashboard.get_before_after_comparison(v2_execution_id)
    
    print(f"\n🔍 Detected Changes:")
    for change in before_after["changes_detected"]:
        print(f"   • {change}")


def main():
    """Run all tests and demonstrations."""
    print("🚀 Lethe Prompt Monitoring System - Test Suite")
    print("=" * 70)
    print(f"📅 Test started at: {datetime.now().isoformat()}")
    print()
    
    try:
        # Run all tests
        test_basic_prompt_tracking()
        test_error_tracking()
        test_comparison_functionality()
        test_analytics_and_trends()
        test_integration_examples()
        test_dashboard_data()
        test_data_export()
        demonstrate_before_after_analysis()
        
        print("\n" + "=" * 70)
        print("🎉 All tests completed successfully!")
        print()
        print("💡 Next steps:")
        print("   1. Run 'python scripts/prompt_monitor.py status' to see the results")
        print("   2. Run 'python scripts/prompt_monitor.py dashboard' to launch the web interface")
        print("   3. Use 'python scripts/prompt_monitor.py list' to see all tracked prompts")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())