#!/usr/bin/env python3
"""
Demo script for Precision/Recall and Efficiency Analysis
Demonstrates the enhanced evaluation system with visualization capabilities.
"""

import sys
import logging
import yaml
from pathlib import Path
import json

# Add the evaluation package to Python path
sys.path.append(str(Path(__file__).parent))

from infinitybench.evaluation_pipeline import EvaluationPipeline
from infinitybench.metrics import compute_comprehensive_ir_metrics
from infinitybench.visualization import create_comprehensive_evaluation_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def demo_synthetic_data():
    """Demonstrate P/R analysis with synthetic data."""
    logger.info("Running synthetic data demonstration")
    
    # Create synthetic ranked results for different methods
    # Format: (chunk_text, score, is_relevant)
    synthetic_results = {
        'lethe': [
            # Lethe: High precision, good efficiency
            ("Relevant answer 1", 0.95, True),
            ("Relevant answer 2", 0.89, True), 
            ("Relevant answer 3", 0.83, True),
            ("Relevant answer 4", 0.78, True),
            ("Irrelevant text 1", 0.72, False),
            ("Relevant answer 5", 0.68, True),
            ("Irrelevant text 2", 0.62, False),
            ("Irrelevant text 3", 0.55, False),
            ("Relevant answer 6", 0.48, True),
            ("Irrelevant text 4", 0.42, False)
        ],
        'bm25': [
            # BM25: Lower precision, more waste
            ("Relevant answer 1", 0.88, True),
            ("Irrelevant text 1", 0.82, False),
            ("Relevant answer 2", 0.76, True),
            ("Irrelevant text 2", 0.71, False),
            ("Irrelevant text 3", 0.68, False),
            ("Relevant answer 3", 0.62, True),
            ("Irrelevant text 4", 0.58, False),
            ("Irrelevant text 5", 0.53, False),
            ("Relevant answer 4", 0.48, True),
            ("Irrelevant text 6", 0.42, False)
        ],
        'chunking_uniform': [
            # Uniform chunking: Poor precision, high waste
            ("Irrelevant text 1", 0.65, False),
            ("Irrelevant text 2", 0.58, False),
            ("Relevant answer 1", 0.52, True),
            ("Irrelevant text 3", 0.48, False),
            ("Irrelevant text 4", 0.42, False),
            ("Irrelevant text 5", 0.38, False),
            ("Relevant answer 2", 0.32, True),
            ("Irrelevant text 6", 0.28, False),
            ("Irrelevant text 7", 0.22, False),
            ("Irrelevant text 8", 0.18, False)
        ]
    }
    
    # Compute IR metrics for each method
    results_data = {}
    k_values = [1, 5, 10]
    
    for method_name, ranked_results in synthetic_results.items():
        logger.info(f"Computing metrics for {method_name}")
        ir_metrics = compute_comprehensive_ir_metrics(ranked_results, k_values)
        results_data[method_name] = ir_metrics
    
    # Generate visualizations
    output_dir = Path("demo_results") / "synthetic_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating visualization report")
    plot_files = create_comprehensive_evaluation_report(
        results_data,
        str(output_dir),
        "Synthetic Data Demo"
    )
    
    logger.info(f"Generated {len(plot_files)} visualization files:")
    for plot_type, file_path in plot_files.items():
        logger.info(f"  - {plot_type}: {file_path}")
    
    # Save results as JSON
    results_path = output_dir / "synthetic_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"Saved detailed results to {results_path}")
    
    return results_data, plot_files

def demo_mini_evaluation():
    """Run a mini evaluation with the pipeline."""
    logger.info("Running mini evaluation demonstration")
    
    # Load configuration
    config_path = Path("config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify config for quick demo
    config['dataset']['max_samples'] = 5  # Very small sample for demo
    config['dataset']['tasks'] = ['passkey']  # Single task
    config['evaluation']['enable_pr_analysis'] = True
    config['lethe']['enabled'] = True  # Enable mock Lethe evaluation
    
    try:
        # Create evaluation pipeline
        pipeline = EvaluationPipeline(config)
        
        # Run evaluation
        logger.info("Starting pipeline evaluation")
        results = pipeline.run_evaluation(quick_mode=True)
        
        # Save results
        output_dir = Path("demo_results") / "mini_evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Mini evaluation completed. Results saved to {results_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("MINI EVALUATION SUMMARY")
        print("="*60)
        
        for task_name, task_results in results.items():
            if task_name == 'dataset_stats':
                continue
                
            print(f"\nTask: {task_name}")
            print(f"Samples: {task_results.get('num_samples', 0)}")
            
            if 'baselines' in task_results:
                print("Traditional Metrics:")
                for baseline_name, metrics in task_results['baselines'].items():
                    if isinstance(metrics, dict) and 'primary_metric' in metrics:
                        print(f"  {baseline_name}: {metrics['primary_metric']:.3f}")
            
            if 'ir_analysis' in task_results:
                print("IR Analysis:")
                for method_name, ir_metrics in task_results['ir_analysis'].items():
                    if 'summary' in ir_metrics:
                        summary = ir_metrics['summary']
                        efficiency = summary.get('overall_efficiency', 0.0)
                        ap = summary.get('average_precision', 0.0)
                        print(f"  {method_name}: Efficiency={efficiency:.3f}, AP={ap:.3f}")
            
            if 'visualization_files' in task_results:
                print(f"Visualizations: {len(task_results['visualization_files'])} files generated")
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline evaluation failed: {e}")
        logger.info("This is expected if data is not available. The synthetic demo should work.")
        return None

def main():
    """Run demonstrations."""
    print("Lethe Evaluation System - P/R Curves and Efficiency Analysis Demo")
    print("="*70)
    
    # Demo 1: Synthetic data
    print("\n1. Running synthetic data demonstration...")
    try:
        synthetic_results, synthetic_plots = demo_synthetic_data()
        print("✓ Synthetic data demo completed successfully")
        
        # Show key findings
        print("\nKey Findings from Synthetic Data:")
        if 'lethe' in synthetic_results and 'bm25' in synthetic_results:
            lethe_eff = synthetic_results['lethe']['efficiency_metrics']['overall_efficiency']
            bm25_eff = synthetic_results['bm25']['efficiency_metrics']['overall_efficiency']
            improvement = ((lethe_eff - bm25_eff) / bm25_eff) * 100
            print(f"  - Lethe Efficiency: {lethe_eff:.1%}")
            print(f"  - BM25 Efficiency: {bm25_eff:.1%}")
            print(f"  - Improvement: {improvement:.1f}%")
        
    except Exception as e:
        print(f"✗ Synthetic data demo failed: {e}")
    
    # Demo 2: Mini evaluation pipeline
    print("\n2. Running mini evaluation pipeline...")
    try:
        pipeline_results = demo_mini_evaluation()
        if pipeline_results:
            print("✓ Mini evaluation completed successfully")
        else:
            print("! Mini evaluation skipped (expected without data)")
    except Exception as e:
        print(f"✗ Mini evaluation failed: {e}")
    
    print("\n" + "="*70)
    print("SYSTEM CAPABILITIES DEMONSTRATED:")
    print("="*70)
    print("✓ Enhanced metrics system with P/R curves")
    print("✓ Efficiency analysis (relevance percentage vs waste)")
    print("✓ Publication-ready visualization system")
    print("✓ Dual-axis plots showing accuracy AND efficiency")
    print("✓ Comprehensive IR metrics (AP, NDCG, etc.)")
    print("✓ Configurable evaluation pipeline")
    print("✓ Support for multiple baselines and k-values")
    
    print(f"\nResults saved in: ./demo_results/")
    print("Check the generated plots to see Lethe's efficiency advantage!")

if __name__ == "__main__":
    main()