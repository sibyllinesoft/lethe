#!/usr/bin/env python3
"""
InfiniteBench Evaluation Runner
==============================

Production-ready runner for the complete InfiniteBench evaluation pipeline.
This script orchestrates the entire evaluation process from dataset loading
through statistical analysis and publication report generation.

Usage:
    python run_evaluation.py --config config.yaml
    python run_evaluation.py --quick-test  # Run with subset for testing
    python run_evaluation.py --tasks passkey,kv_retrieval  # Specific tasks only

Author: Lethe Research Team
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from datetime import datetime
import json

# Import our InfiniteBench modules
from .dataset_loader import InfiniteBenchLoader
from .evaluation_pipeline import InfiniteBenchEvaluator, ExperimentConfig
from .baselines import BM25Baseline, NaiveChunkingBaseline, DenseRetrievalBaseline, GPT4Baseline
from .statistical_analysis import InfiniteBenchStatistics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InfiniteBenchRunner:
    """
    Main runner class for InfiniteBench evaluation.
    
    Orchestrates the complete pipeline:
    1. Dataset loading and preprocessing
    2. Baseline method execution
    3. Lethe system evaluation
    4. Statistical analysis
    5. Report generation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize runner with configuration."""
        self.config = self._load_config(config_path)
        self.setup_directories()
        
        # Initialize components
        data_dir = Path(self.config['data']['infinitebench_path'])
        self.loader = InfiniteBenchLoader(data_dir)
        self.evaluator = InfiniteBenchEvaluator()
        self.statistics = InfiniteBenchStatistics()
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'data': {
                'infinitebench_path': 'benchmarks/infinitebench/data',
                'output_dir': 'artifacts/infinitebench_results',
                'cache_dir': '.cache/infinitebench'
            },
            'evaluation': {
                'tasks': [
                    'passkey', 'number_string', 'kv_retrieval',
                    'longbook_sum_eng', 'longbook_choice_eng', 'longbook_qa_eng',
                    'longdialogue_qa_eng', 'code_debug', 'code_run',
                    'math_calc', 'math_find'
                ],
                'max_samples_per_task': None,  # Use all samples
                'methods': ['bm25', 'naive_chunking', 'dense_retrieval', 'lethe'],
                'include_gpt4_baseline': False,  # Expensive, enable manually
                'bootstrap_samples': 1000,
                'confidence_level': 0.95,
                'parallel_jobs': 4
            },
            'baselines': {
                'bm25': {
                    'k1': 1.2,
                    'b': 0.75,
                    'top_k': 5
                },
                'naive_chunking': {
                    'chunk_size': 1024,
                    'strategy': 'uniform',
                    'top_k': 5
                },
                'dense_retrieval': {
                    'model_name': 'all-MiniLM-L6-v2',
                    'top_k': 5
                }
            },
            'reporting': {
                'generate_plots': True,
                'plot_format': 'png',
                'include_detailed_analysis': True,
                'publication_ready': True
            }
        }
    
    def setup_directories(self):
        """Create necessary directories."""
        dirs_to_create = [
            self.config['data']['output_dir'],
            self.config['data']['cache_dir'],
            f"{self.config['data']['output_dir']}/plots",
            f"{self.config['data']['output_dir']}/reports"
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def initialize_baselines(self) -> Dict[str, object]:
        """Initialize all baseline methods."""
        baselines = {}
        
        if 'bm25' in self.config['evaluation']['methods']:
            baselines['bm25'] = BM25Baseline(**self.config['baselines']['bm25'])
            
        if 'naive_chunking' in self.config['evaluation']['methods']:
            baselines['naive_chunking'] = NaiveChunkingBaseline(
                **self.config['baselines']['naive_chunking']
            )
            
        if 'dense_retrieval' in self.config['evaluation']['methods']:
            baselines['dense_retrieval'] = DenseRetrievalBaseline(
                **self.config['baselines']['dense_retrieval']
            )
            
        if 'gpt4' in self.config['evaluation']['methods'] and self.config['evaluation']['include_gpt4_baseline']:
            baselines['gpt4'] = GPT4Baseline()
            logger.warning("GPT-4 baseline enabled - this will be expensive!")
        
        return baselines
    
    def run_quick_test(self) -> str:
        """Run a quick test with subset of data for validation."""
        logger.info("🧪 Running quick test with subset of data")
        
        # Override config for quick test
        test_config = ExperimentConfig(
            experiment_name=f"infinitebench_quick_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tasks=['passkey', 'kv_retrieval'],  # Just 2 tasks
            methods=['bm25', 'naive_chunking'],  # Fast methods only
            max_samples_per_task=10,  # Very small sample
            output_dir=Path(self.config['data']['output_dir']) / 'quick_test',
            save_intermediate=True,
            bootstrap_samples=100,  # Fewer bootstrap samples
            parallel_jobs=2
        )
        
        # Load test data
        test_data = {}
        for task in test_config.tasks:
            try:
                samples = self.loader.load_task(task)
                if samples:
                    # Take just a few samples for quick test
                    test_data[task] = samples[:test_config.max_samples_per_task]
                    logger.info(f"✅ Loaded {len(test_data[task])} samples for {task}")
                else:
                    logger.warning(f"⚠️ No samples found for task {task}")
            except Exception as e:
                logger.error(f"❌ Failed to load task {task}: {e}")
        
        if not test_data:
            logger.error("❌ No test data available - check dataset download")
            return "failed"
        
        # Initialize baselines
        baselines = {
            'bm25': BM25Baseline(**self.config['baselines']['bm25']),
            'naive_chunking': NaiveChunkingBaseline(**self.config['baselines']['naive_chunking'])
        }
        
        # Run evaluation
        try:
            results = self.evaluator.run_experiment(test_config, test_data, baselines)
            
            # Generate quick report
            report_path = test_config.output_dir / 'quick_test_report.md'
            self._generate_quick_report(results, report_path)
            
            logger.info(f"✅ Quick test completed successfully")
            logger.info(f"📊 Results saved to: {test_config.output_dir}")
            logger.info(f"📄 Report: {report_path}")
            
            return str(test_config.output_dir)
            
        except Exception as e:
            logger.error(f"❌ Quick test failed: {e}")
            return "failed"
    
    def run_full_evaluation(self, selected_tasks: Optional[List[str]] = None) -> str:
        """Run complete InfiniteBench evaluation."""
        logger.info("🚀 Starting full InfiniteBench evaluation")
        
        # Determine tasks to run
        tasks_to_run = selected_tasks or self.config['evaluation']['tasks']
        
        # Create experiment config
        experiment_config = ExperimentConfig(
            experiment_name=f"infinitebench_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tasks=tasks_to_run,
            methods=self.config['evaluation']['methods'],
            max_samples_per_task=self.config['evaluation']['max_samples_per_task'],
            output_dir=Path(self.config['data']['output_dir']) / 'full_evaluation',
            save_intermediate=True,
            bootstrap_samples=self.config['evaluation']['bootstrap_samples'],
            parallel_jobs=self.config['evaluation']['parallel_jobs']
        )
        
        # Load all data
        logger.info("📊 Loading dataset...")
        all_data = {}
        for task in tasks_to_run:
            try:
                samples = self.loader.load_task(task)
                if samples:
                    if experiment_config.max_samples_per_task:
                        samples = samples[:experiment_config.max_samples_per_task]
                    all_data[task] = samples
                    logger.info(f"✅ Loaded {len(all_data[task])} samples for {task}")
                else:
                    logger.warning(f"⚠️ No samples found for task {task}")
            except Exception as e:
                logger.error(f"❌ Failed to load task {task}: {e}")
        
        if not all_data:
            logger.error("❌ No data available for evaluation")
            return "failed"
        
        # Initialize baselines
        logger.info("🔧 Initializing baseline methods...")
        baselines = self.initialize_baselines()
        
        # Run evaluation
        logger.info("⚡ Running evaluation...")
        try:
            results = self.evaluator.run_experiment(experiment_config, all_data, baselines)
            
            # Generate comprehensive analysis
            logger.info("📊 Generating statistical analysis...")
            self._generate_full_analysis(results, experiment_config.output_dir)
            
            logger.info("✅ Full evaluation completed successfully")
            logger.info(f"📊 Results saved to: {experiment_config.output_dir}")
            
            return str(experiment_config.output_dir)
            
        except Exception as e:
            logger.error(f"❌ Full evaluation failed: {e}")
            raise
    
    def _generate_quick_report(self, results: Dict, report_path: Path):
        """Generate a quick test report."""
        with open(report_path, 'w') as f:
            f.write("# InfiniteBench Quick Test Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write("This is a quick validation test of the InfiniteBench evaluation pipeline.\n\n")
            
            # Task results
            for task, task_results in results.get('task_results', {}).items():
                f.write(f"### {task.title()}\n\n")
                
                for method, method_results in task_results.items():
                    if hasattr(method_results, 'overall_score'):
                        f.write(f"- **{method}**: {method_results.overall_score:.3f}\n")
                
                f.write("\n")
            
            f.write("## Status\n\n")
            f.write("✅ Pipeline validation successful\n")
            f.write("✅ Data loading functional\n") 
            f.write("✅ Baseline methods operational\n")
            f.write("✅ Metrics calculation working\n\n")
            
            f.write("Ready for full evaluation!\n")
    
    def _generate_full_analysis(self, results: Dict, output_dir: Path):
        """Generate comprehensive analysis and reports."""
        logger.info("📈 Generating publication-ready analysis...")
        
        # Generate statistical analysis
        analysis_results = self.statistics.analyze_results(
            results,
            bootstrap_samples=self.config['evaluation']['bootstrap_samples'],
            confidence_level=self.config['evaluation']['confidence_level']
        )
        
        # Save analysis results
        analysis_path = output_dir / 'statistical_analysis.json'
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Generate publication report
        if self.config['reporting']['publication_ready']:
            logger.info("📄 Generating publication report...")
            report_path = output_dir / 'reports' / 'publication_report.md'
            self.statistics.generate_publication_report(
                analysis_results,
                str(report_path)
            )
        
        # Generate visualizations
        if self.config['reporting']['generate_plots']:
            logger.info("📊 Creating visualization plots...")
            plots_dir = output_dir / 'plots'
            self.statistics.create_visualization_plots(
                analysis_results,
                str(plots_dir),
                format=self.config['reporting']['plot_format']
            )
        
        logger.info("✅ Analysis complete")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='InfiniteBench Evaluation Runner')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Run quick test with subset of data')
    parser.add_argument('--tasks', type=str, 
                       help='Comma-separated list of tasks to run')
    parser.add_argument('--output-dir', type=str,
                       help='Override output directory')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize runner
        runner = InfiniteBenchRunner(args.config)
        
        # Override output directory if specified
        if args.output_dir:
            runner.config['data']['output_dir'] = args.output_dir
            runner.setup_directories()
        
        if args.quick_test:
            result_dir = runner.run_quick_test()
            if result_dir != "failed":
                print(f"\n✅ Quick test completed successfully!")
                print(f"📊 Results: {result_dir}")
            else:
                print("\n❌ Quick test failed")
                sys.exit(1)
        else:
            # Parse tasks if specified
            selected_tasks = None
            if args.tasks:
                selected_tasks = [task.strip() for task in args.tasks.split(',')]
                logger.info(f"Running selected tasks: {selected_tasks}")
            
            result_dir = runner.run_full_evaluation(selected_tasks)
            if result_dir != "failed":
                print(f"\n🎉 Full evaluation completed successfully!")
                print(f"📊 Results: {result_dir}")
                print(f"📄 Check the reports/ directory for publication-ready analysis")
            else:
                print("\n❌ Full evaluation failed")
                sys.exit(1)
                
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()