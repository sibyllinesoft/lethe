#!/usr/bin/env python3
"""
Milestone 7: Publication-Ready Analysis CLI
Single command execution for complete publication pipeline.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.eval.milestone7_analysis import Milestone7AnalysisPipeline

def main():
    parser = argparse.ArgumentParser(
        description="Milestone 7: Publication-Ready Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with evaluation results
  python run_milestone7_analysis.py \\
    --metrics-file analysis/final_statistical_gatekeeper_results.json \\
    --train-data datasets/lethebench_agents_train.json \\
    --test-data datasets/lethebench_agents_test.json

  # Custom output directory
  python run_milestone7_analysis.py \\
    --metrics-file analysis/metrics.json \\
    --train-data datasets/train.json \\
    --test-data datasets/test.json \\
    --output-dir ./publication_results

  # Using existing evaluation data
  python run_milestone7_analysis.py \\
    --metrics-file analysis/final_statistical_gatekeeper_results.json \\
    --train-data datasets/lethebench \\
    --test-data datasets/lethebench \\
    --hardware-profile "Custom_Profile_Name"
        """
    )
    
    parser.add_argument("--metrics-file", type=Path, 
                       default=Path("analysis/final_statistical_gatekeeper_results.json"),
                       help="Path to metrics JSON from Milestone 6 evaluation")
    
    parser.add_argument("--train-data", type=Path,
                       default=Path("datasets/lethebench"),
                       help="Path to training dataset JSON file or directory")
    
    parser.add_argument("--test-data", type=Path,
                       default=Path("datasets/lethebench"), 
                       help="Path to test dataset JSON file or directory")
    
    parser.add_argument("--output-dir", type=Path, default=Path("./analysis"),
                       help="Output directory for analysis results (default: ./analysis)")
    
    parser.add_argument("--hardware-profile", type=str,
                       help="Custom hardware profile name (auto-detected if not provided)")
    
    parser.add_argument("--quick-test", action="store_true",
                       help="Run with synthetic data for quick validation")
    
    args = parser.parse_args()
    
    # Validate input files
    if not args.quick_test:
        if not args.metrics_file.exists():
            print(f"❌ Error: Metrics file not found: {args.metrics_file}")
            print("💡 Hint: Run Milestone 6 evaluation first or use --quick-test")
            sys.exit(1)
        
        # Handle dataset directory vs file
        train_data_file = _resolve_dataset_file(args.train_data, "train")
        test_data_file = _resolve_dataset_file(args.test_data, "test")
        
        if not train_data_file.exists():
            print(f"❌ Error: Training data not found: {train_data_file}")
            sys.exit(1)
        
        if not test_data_file.exists():
            print(f"❌ Error: Test data not found: {test_data_file}")
            sys.exit(1)
    else:
        # Use synthetic data for quick test
        train_data_file = _create_synthetic_dataset("train")
        test_data_file = _create_synthetic_dataset("test")
        
        if not args.metrics_file.exists():
            args.metrics_file = _create_synthetic_metrics()
    
    print("🚀 Starting Milestone 7: Publication-Ready Analysis Pipeline")
    print(f"📊 Metrics file: {args.metrics_file}")
    print(f"📁 Training data: {train_data_file}")
    print(f"📁 Test data: {test_data_file}")
    print(f"📁 Output directory: {args.output_dir}")
    
    # Initialize and run pipeline
    try:
        pipeline = Milestone7AnalysisPipeline(output_dir=args.output_dir)
        
        if args.hardware_profile:
            # Override hardware profile name
            pipeline.hardware_manager.profile_name_override = args.hardware_profile
        
        pipeline.run_complete_analysis(
            metrics_file=args.metrics_file,
            train_data_file=train_data_file,
            test_data_file=test_data_file
        )
        
        print("✅ Milestone 7 analysis completed successfully!")
        print(f"📁 Results available in: {pipeline.profile_dir}")
        print("\n📄 Generated outputs:")
        print("   📊 Tables: quality_metrics.csv/.tex, agent_metrics.csv/.tex, efficiency_metrics.csv/.tex")
        print("   📈 Plots: scalability, throughput, tradeoffs, scenario_breakdown")
        print("   🔍 Sanity Checks: validation reports and leakage analysis")
        print("   🖥️  Hardware Profile: organized by system configuration")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def _resolve_dataset_file(path: Path, split: str) -> Path:
    """Resolve dataset path to specific file"""
    
    if path.is_file():
        return path
    elif path.is_dir():
        # Try common naming patterns
        candidates = [
            path / f"lethebench_agents_{split}.json",
            path / f"lethebench_{split}.json", 
            path / f"{split}.json",
            path / "lethebench_agents.json",
            path / "lethebench.json"
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        # If no specific split file, use the general dataset
        general_file = path / "lethebench_agents.json"
        if general_file.exists():
            return general_file
        
        general_file = path / "lethebench.json"
        if general_file.exists():
            return general_file
    
    return path

def _create_synthetic_dataset(split: str) -> Path:
    """Create synthetic dataset for quick testing"""
    
    synthetic_data = []
    for i in range(50):  # Small dataset for quick testing
        item = {
            "query": f"Example query {i} for {split} split",
            "query_type": "simple_qa" if i % 5 == 0 else "multi_turn",
            "relevant_docs": [
                {"doc_id": f"doc_{i}_{j}", "content": f"Document {j} content", "relevance": 0.8}
                for j in range(3)
            ],
            "novelty_score": 0.3 + (i % 10) * 0.07,  # Varies from 0.3 to 0.9
            "planning_action": "EXPLORE" if i % 7 == 0 else "RETRIEVE",
            "split": split
        }
        synthetic_data.append(item)
    
    # Save synthetic data
    synthetic_file = Path(f"/tmp/synthetic_{split}_data.json")
    with open(synthetic_file, 'w') as f:
        json.dump(synthetic_data, f, indent=2)
    
    return synthetic_file

def _create_synthetic_metrics() -> Path:
    """Create synthetic metrics for quick testing"""
    
    synthetic_metrics = {
        "task5_metadata": {
            "completion_timestamp": "2025-08-25T14:00:00.000000",
            "task_name": "Synthetic Metrics for Milestone 7 Testing",
            "compliance_status": "SYNTHETIC"
        },
        "results": {
            "query_results": []
        },
        "statistical_analysis": {
            "methods_analyzed": [
                "baseline_bm25_only",
                "baseline_vector_only", 
                "baseline_bm25_vector_simple",
                "baseline_cross_encoder",
                "baseline_mmr",
                "baseline_faiss_ivf"
            ]
        }
    }
    
    # Add synthetic query results for sanity checks
    for i in range(100):
        query_result = {
            "query": f"synthetic query {i}",
            "query_type": "exact_match" if i % 10 == 0 else "general",
            "novelty_score": 0.2 + (i % 10) * 0.08,
            "planning_action": "EXPLORE" if i % 6 == 0 else "RETRIEVE",
            "retrieved_docs": [
                {"relevance_score": 0.95 if i % 10 == 0 else 0.6}  # Exact matches get high scores
            ]
        }
        synthetic_metrics["results"]["query_results"].append(query_result)
    
    # Save synthetic metrics
    synthetic_file = Path("/tmp/synthetic_metrics.json")
    with open(synthetic_file, 'w') as f:
        json.dump(synthetic_metrics, f, indent=2)
    
    return synthetic_file

if __name__ == "__main__":
    main()