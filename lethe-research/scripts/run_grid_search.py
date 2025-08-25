#!/usr/bin/env python3
"""
Lethe Grid Search Execution Script
==================================

Executes comprehensive parameter grid search for Lethe system optimization.
Integrates with experiments/run.py for actual execution logic.

Usage:
    python run_grid_search.py --dataset dataset.json --output results/ --config config.yaml
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import yaml
from datetime import datetime
import multiprocessing as mp

# Add research directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from experiments.run import ExperimentController

def setup_logging(level: str) -> logging.Logger:
    """Configure logging with specified level."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

class GridSearchExecutor:
    """Executes Lethe parameter grid search."""
    
    def __init__(self, dataset_path: Path, output_dir: Path, config_path: Path,
                 ctx_run_path: Path, max_parallel: int, logger: logging.Logger):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.config_path = config_path
        self.ctx_run_path = ctx_run_path
        self.max_parallel = max_parallel
        self.logger = logger
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize experiment controller
        self.controller = ExperimentController(
            dataset_path=str(dataset_path),
            output_dir=str(output_dir),
            ctx_run_path=str(ctx_run_path),
            max_workers=max_parallel
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """Load grid search configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def validate_setup(self) -> None:
        """Validate that all required components are available."""
        self.logger.info("Validating grid search setup...")
        
        # Check dataset
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
            
        # Check ctx-run executable
        if not self.ctx_run_path.exists():
            raise FileNotFoundError(f"ctx-run executable not found: {self.ctx_run_path}")
        
        # Test dataset loading
        try:
            with open(self.dataset_path) as f:
                dataset = json.load(f)
            
            if not dataset.get('queries'):
                raise ValueError("Dataset contains no queries")
                
            if not dataset.get('examples'):
                raise ValueError("Dataset contains no examples")
                
            self.logger.info(f"Dataset validated: {len(dataset['queries'])} queries, "
                           f"{len(dataset['examples'])} examples")
            
        except Exception as e:
            raise ValueError(f"Invalid dataset format: {e}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("✅ Setup validation complete")
    
    def generate_grid_configurations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        grid_config = self.config.get('grid_search', {})
        
        # Default parameter grid if not specified
        if not grid_config.get('parameters'):
            grid_config['parameters'] = {
                'alpha': [0.3, 0.5, 0.7],
                'beta': [0.2, 0.4, 0.6],
                'chunk_size': [1000, 2000, 3000],
                'overlap': [200, 400, 600],
                'k_initial': [50, 100, 200],
                'k_final': [10, 20, 30],
                'diversify_pack_size': [3, 5, 7],
                'hyde_k': [1, 2, 3],
                'planning_strategy': ['exhaustive', 'focused', 'adaptive']
            }
        
        parameters = grid_config['parameters']
        
        # Generate all combinations
        configurations = []
        
        def generate_combinations(param_dict, current_config=None):
            if current_config is None:
                current_config = {}
            
            if not param_dict:
                configurations.append(current_config.copy())
                return
            
            param_name = next(iter(param_dict))
            param_values = param_dict[param_name]
            remaining_params = {k: v for k, v in param_dict.items() if k != param_name}
            
            for value in param_values:
                current_config[param_name] = value
                generate_combinations(remaining_params, current_config)
                del current_config[param_name]
        
        generate_combinations(parameters)
        
        # Add metadata to each configuration
        for i, config in enumerate(configurations):
            config['_config_id'] = f"lethe_grid_{i:04d}"
            config['_config_type'] = 'lethe_variant'
            config['_timestamp'] = datetime.utcnow().isoformat()
        
        self.logger.info(f"Generated {len(configurations)} grid configurations")
        return configurations
    
    def execute_grid_search(self) -> Dict[str, Any]:
        """Execute complete grid search."""
        self.logger.info("🔬 Starting Lethe parameter grid search...")
        
        # Validate setup
        self.validate_setup()
        
        # Generate configurations
        configurations = self.generate_grid_configurations()
        
        if not configurations:
            raise ValueError("No configurations generated for grid search")
        
        # Save configurations for reference
        config_file = self.output_dir / 'grid_configurations.json'
        with open(config_file, 'w') as f:
            json.dump(configurations, f, indent=2)
        
        self.logger.info(f"Saved configurations to: {config_file}")
        
        # Execute experiments
        results = self.controller.execute_experiment(configurations)
        
        # Save detailed results
        results_file = self.output_dir / 'grid_search_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved results to: {results_file}")
        
        # Generate summary
        summary = self._generate_summary(results)
        summary_file = self.output_dir / 'grid_search_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Generated summary: {summary_file}")
        
        return results
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of grid search results."""
        summary = {
            'metadata': {
                'total_configurations': len(results.get('experiments', [])),
                'completed_configurations': len([exp for exp in results.get('experiments', []) 
                                                if exp.get('status') == 'completed']),
                'failed_configurations': len([exp for exp in results.get('experiments', []) 
                                            if exp.get('status') == 'failed']),
                'total_runtime_seconds': results.get('metadata', {}).get('total_time', 0),
                'timestamp': datetime.utcnow().isoformat()
            },
            'best_configurations': {},
            'parameter_analysis': {},
            'performance_overview': {}
        }
        
        experiments = results.get('experiments', [])
        completed_experiments = [exp for exp in experiments if exp.get('status') == 'completed']
        
        if not completed_experiments:
            self.logger.warning("No completed experiments for summary generation")
            return summary
        
        # Find best configurations for each metric
        metrics = ['ndcg_at_10', 'recall_at_10', 'mrr_at_10', 'latency_p95', 'memory_peak']
        
        for metric in metrics:
            best_exp = None
            best_value = None
            
            for exp in completed_experiments:
                metric_value = exp.get('results', {}).get(metric)
                if metric_value is None:
                    continue
                
                # For latency and memory, lower is better
                if metric in ['latency_p95', 'memory_peak']:
                    if best_value is None or metric_value < best_value:
                        best_value = metric_value
                        best_exp = exp
                else:
                    # For quality metrics, higher is better
                    if best_value is None or metric_value > best_value:
                        best_value = metric_value
                        best_exp = exp
            
            if best_exp:
                summary['best_configurations'][metric] = {
                    'config_id': best_exp.get('config_id'),
                    'config': best_exp.get('config'),
                    'value': best_value,
                    'all_metrics': best_exp.get('results', {})
                }
        
        # Parameter analysis
        parameter_impacts = {}
        for param_name in ['alpha', 'beta', 'chunk_size', 'k_initial']:
            param_values = {}
            for exp in completed_experiments:
                param_val = exp.get('config', {}).get(param_name)
                if param_val is not None:
                    if param_val not in param_values:
                        param_values[param_val] = []
                    param_values[param_val].append(exp.get('results', {}).get('ndcg_at_10', 0))
            
            # Calculate average performance per parameter value
            param_averages = {}
            for val, scores in param_values.items():
                if scores:
                    param_averages[val] = sum(scores) / len(scores)
            
            parameter_impacts[param_name] = param_averages
        
        summary['parameter_analysis'] = parameter_impacts
        
        # Performance overview
        all_ndcg = [exp.get('results', {}).get('ndcg_at_10', 0) for exp in completed_experiments]
        all_latency = [exp.get('results', {}).get('latency_p95', 0) for exp in completed_experiments 
                      if exp.get('results', {}).get('latency_p95') is not None]
        
        summary['performance_overview'] = {
            'ndcg_at_10': {
                'min': min(all_ndcg) if all_ndcg else 0,
                'max': max(all_ndcg) if all_ndcg else 0,
                'avg': sum(all_ndcg) / len(all_ndcg) if all_ndcg else 0
            },
            'latency_p95': {
                'min': min(all_latency) if all_latency else 0,
                'max': max(all_latency) if all_latency else 0,
                'avg': sum(all_latency) / len(all_latency) if all_latency else 0
            }
        }
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Execute Lethe parameter grid search')
    parser.add_argument('--dataset', type=Path, required=True,
                       help='Path to LetheBench dataset JSON file')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for results')
    parser.add_argument('--config', type=Path, required=True,
                       help='Grid search configuration file')
    parser.add_argument('--ctx-run-path', type=Path, required=True,
                       help='Path to ctx-run CLI executable')
    parser.add_argument('--parallel', type=int, default=4,
                       help='Maximum parallel processes')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else args.log_level
    logger = setup_logging(log_level)
    
    try:
        # Create grid search executor
        executor = GridSearchExecutor(
            dataset_path=args.dataset,
            output_dir=args.output,
            config_path=args.config,
            ctx_run_path=args.ctx_run_path,
            max_parallel=args.parallel,
            logger=logger
        )
        
        # Execute grid search
        results = executor.execute_grid_search()
        
        logger.info("✅ Grid search completed successfully!")
        logger.info(f"Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"❌ Grid search failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()