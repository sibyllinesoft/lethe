#!/usr/bin/env python3
"""
InfinityBench Evaluation Runner for Lethe
Academic-quality benchmark evaluation with statistical rigor.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import yaml
from datetime import datetime

from infinitybench import (
    InfinityBenchDataset,
    run_evaluation, 
    compute_statistical_analysis
)

def setup_logging(level=logging.INFO):
    """Setup comprehensive logging."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path):
    """Load evaluation configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def quick_test(config):
    """Run quick validation test."""
    print("🔬 Running InfinityBench Quick Test...")
    
    # Override config for quick test
    config['dataset']['max_samples'] = 5
    config['dataset']['tasks'] = ['passkey', 'kv_retrieval'] 
    config['evaluation']['bootstrap_samples'] = 100
    
    return run_evaluation(config, quick_mode=True)

def main():
    parser = argparse.ArgumentParser(description='InfinityBench Evaluation for Lethe')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    parser.add_argument('--quick-test', action='store_true', help='Run quick validation test')
    parser.add_argument('--tasks', help='Comma-separated list of tasks to evaluate')
    parser.add_argument('--output-dir', default='./results', help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override with command line arguments
        if args.tasks:
            config['dataset']['tasks'] = [t.strip() for t in args.tasks.split(',')]
        
        config['output']['results_dir'] = args.output_dir
        
        # Create output directory
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        if args.quick_test:
            results = quick_test(config)
            logger.info("✅ Quick test completed successfully!")
            print(json.dumps(results, indent=2))
        else:
            logger.info("🚀 Starting comprehensive InfinityBench evaluation...")
            
            # Run full evaluation
            results = run_evaluation(config)
            
            # Compute statistical analysis
            stats = compute_statistical_analysis(results, config)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = Path(args.output_dir) / f"infinitybench_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump({
                    'results': results,
                    'statistics': stats,
                    'config': config,
                    'timestamp': timestamp
                }, f, indent=2)
            
            logger.info(f"✅ Evaluation completed! Results saved to {results_file}")
            
            # Print summary
            print("\n📊 EVALUATION SUMMARY")
            print("=" * 50)
            for task, metrics in results.items():
                if isinstance(metrics, dict):
                    print(f"{task:20s}: {metrics.get('primary_metric', 'N/A'):.3f}")
            
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()