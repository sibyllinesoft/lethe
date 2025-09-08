#!/usr/bin/env python3
"""
Comprehensive Benchmark Demo Script
===================================

Demonstrates the complete benchmarking pipeline with sample data.
Perfect for validation and testing without requiring full dataset downloads.
"""

import sys
import time
import json
from pathlib import Path

# Add benchmarks to Python path
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.config import BenchmarkConfig, EvaluationConfig, ReportingConfig
from benchmarks.orchestrator import BenchmarkOrchestrator
from benchmarks.datasets.base import DatasetSample
from benchmarks.competitors.registry import get_competitor_registry


def create_sample_datasets():
    """Create sample datasets for demonstration."""
    
    # Sample Chinese QA data
    zh_qa_samples = [
        DatasetSample(
            id="zh_qa_001",
            query="什么是机器学习？",
            context="机器学习是人工智能的一个分支，它使计算机系统能够自动学习和改进，而无需被明确编程。机器学习算法构建基于训练数据的数学模型，以便在没有明确编程的情况下做出预测或决策。" * 50,  # Long context
            answer="机器学习是人工智能的一个分支",
            context_length=1500,
            query_length=8,
            metadata={"task_type": "multilingual_qa", "language": "zh", "source": "demo"}
        ),
        DatasetSample(
            id="zh_qa_002", 
            query="深度学习的主要优势是什么？",
            context="深度学习是机器学习的子集，使用具有多个层的神经网络。它在图像识别、自然语言处理和语音识别等任务中表现出色。深度学习的主要优势包括自动特征提取、处理大量数据的能力，以及在复杂模式识别任务中的卓越性能。" * 40,
            answer="自动特征提取、处理大量数据的能力",
            context_length=1200,
            query_length=10,
            metadata={"task_type": "multilingual_qa", "language": "zh", "source": "demo"}
        )
    ]
    
    # Sample code debugging data
    code_debug_samples = [
        DatasetSample(
            id="code_debug_001",
            query="Why does this function return None instead of the expected result?",
            context="""
            def calculate_average(numbers):
                if not numbers:
                    return None
                total = sum(numbers)
                average = total / len(numbers)
                # Bug: missing return statement
            
            def process_data(data_list):
                results = []
                for item in data_list:
                    avg = calculate_average(item)
                    if avg is not None:
                        results.append(avg)
                return results
            """ * 20,  # Repeat for longer context
            answer="Missing return statement in calculate_average function",
            context_length=800,
            query_length=12,
            metadata={"task_type": "code_debugging", "programming_language": "python", "source": "demo"}
        )
    ]
    
    # Sample needle-in-haystack data
    passkey_samples = [
        DatasetSample(
            id="passkey_001",
            query="What is the pass key?",
            context="Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 100 + 
                   "The pass key is 42851. " + 
                   "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. " * 100,
            answer="42851",
            context_length=2500,
            query_length=5,
            metadata={"task_type": "needle_in_haystack", "passkey": "42851", "source": "demo"}
        )
    ]
    
    return {
        "demo_zh_qa": zh_qa_samples,
        "demo_code_debug": code_debug_samples, 
        "demo_passkey": passkey_samples
    }


def run_demo_benchmark():
    """Run a complete demo benchmark."""
    print("🚀 COMPREHENSIVE BENCHMARK DEMO")
    print("=" * 60)
    print("This demo shows the complete benchmarking pipeline")
    print("with sample data and mock competitors.")
    print()
    
    # Create demo configuration
    config = BenchmarkConfig(
        run_name="demo_comprehensive_benchmark",
        experiment_tags=["demo", "validation", "sample-data"],
        dry_run=True,  # Use mock data
        max_workers=2,
        enabled_competitors=["lethe_hybrid", "weaviate", "colbert_v2"],  # Subset for demo
        enabled_datasets=[]  # Will use synthetic data
    )
    
    # Override evaluation for faster demo
    config.evaluation = EvaluationConfig(
        keep_ratios=[0.15, 0.30],  # Only 2 budgets for speed
        statistical_testing={
            "bootstrap_iterations": 100,  # Reduced for demo
            "permutation_iterations": 100,
            "confidence_level": 0.95,
            "correction_method": "holm"
        }
    )
    
    print(f"📋 Demo Configuration:")
    print(f"   • Run Name: {config.run_name}")
    print(f"   • Competitors: {len(config.enabled_competitors)}")
    print(f"   • Keep Ratios: {config.evaluation.keep_ratios}")
    print(f"   • Statistical Iterations: {config.evaluation.statistical_testing['bootstrap_iterations']}")
    print()
    
    # Initialize orchestrator
    print("🏗️  Initializing benchmark orchestrator...")
    orchestrator = BenchmarkOrchestrator(config)
    
    # Create sample datasets
    print("📊 Creating sample datasets...")
    datasets = create_sample_datasets()
    print(f"   • Created {len(datasets)} datasets")
    for name, samples in datasets.items():
        print(f"     - {name}: {len(samples)} samples")
    print()
    
    # Get sample competitors  
    print("🤖 Initializing competitors...")
    competitor_registry = get_competitor_registry()
    competitors = []
    
    for comp_name in config.enabled_competitors:
        try:
            competitor = competitor_registry.get_competitor(comp_name)
            competitors.append(competitor)
            print(f"   ✅ {comp_name}: {competitor.__class__.__name__}")
        except Exception as e:
            print(f"   ❌ {comp_name}: {e}")
    
    print(f"   • Total competitors ready: {len(competitors)}")
    print()
    
    # Run evaluation
    print("🔬 Running evaluations...")
    start_time = time.time()
    
    try:
        evaluation_results = orchestrator.evaluation_engine.evaluate_all_competitors(
            competitors=competitors,
            datasets=datasets, 
            max_workers=config.max_workers
        )
        
        eval_time = time.time() - start_time
        print(f"   ✅ Evaluation completed in {eval_time:.1f}s")
        print(f"   • Total result sets: {len(evaluation_results)}")
        
        # Show sample results
        print("\n📈 Sample Results:")
        for key, results in list(evaluation_results.items())[:3]:
            result = results[0] if results else None
            if result:
                print(f"   • {key}:")
                print(f"     - Latency: {result.mean_latency_ms:.1f}ms")
                print(f"     - Precision@k: {result.precision_at_k:.3f}")
                print(f"     - Success Rate: {result.success_rate:.1%}")
        
    except Exception as e:
        print(f"   ❌ Evaluation failed: {e}")
        return 1
    
    # Run statistical analysis
    print("\n📊 Running statistical analysis...")
    try:
        statistical_comparisons = orchestrator.evaluation_engine.compare_competitors(
            results=evaluation_results,
            baseline_competitor="lethe_hybrid"
        )
        
        print(f"   ✅ Statistical analysis completed")
        print(f"   • Total comparisons: {len(statistical_comparisons)}")
        
        if statistical_comparisons:
            significant_count = sum(1 for comp in statistical_comparisons if comp.is_significant)
            print(f"   • Significant improvements: {significant_count}/{len(statistical_comparisons)}")
        
    except Exception as e:
        print(f"   ❌ Statistical analysis failed: {e}")
        statistical_comparisons = []
    
    # Generate reports
    print("\n📄 Generating reports...")
    try:
        results_dir = Path("demo_results")
        results_dir.mkdir(exist_ok=True)
        
        report_paths = orchestrator.report_generator.generate_comprehensive_report(
            evaluation_results=evaluation_results,
            statistical_comparisons=statistical_comparisons,
            datasets=datasets,
            competitor_configs={},  # Mock configs
            output_dir=results_dir
        )
        
        print("   ✅ Reports generated:")
        for report_type, path in report_paths.items():
            print(f"     - {report_type.upper()}: {path}")
        
    except Exception as e:
        print(f"   ❌ Report generation failed: {e}")
        return 1
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("✅ DEMO BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"📊 Total Duration: {total_time:.1f} seconds")
    print(f"📊 Datasets: {len(datasets)}")
    print(f"📊 Competitors: {len(competitors)}")
    print(f"📊 Evaluations: {sum(len(results) for results in evaluation_results.values())}")
    print(f"📊 Statistical Tests: {len(statistical_comparisons)}")
    print(f"📊 Reports Generated: {len(report_paths)}")
    print()
    print("🎯 This demonstrates the complete pipeline!")
    print("   For full benchmarks, use: python run_benchmark.py")
    print("=" * 60)
    
    return 0


def main():
    """Main demo entry point."""
    try:
        return run_demo_benchmark()
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())