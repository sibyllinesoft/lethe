#!/usr/bin/env python3
"""
Reproducibility Validation for Milestone 6 Evaluation

Ensures that evaluation results are reproducible within ±2% tolerance.
Validates deterministic behavior and consistent random seed handling.

Features:
- Multiple evaluation runs with same configuration
- Statistical comparison of results across runs
- Tolerance checking for key metrics
- Environment consistency validation
- Hardware profile standardization
"""

import json
import time
import logging
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import subprocess
import sys
import platform
import psutil
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ReproducibilityTest:
    """Single reproducibility test result"""
    metric_name: str
    run_1_value: float
    run_2_value: float
    absolute_diff: float
    relative_diff_percent: float
    within_tolerance: bool
    tolerance_percent: float

class EnvironmentValidator:
    """Validate environment consistency for reproducible evaluation"""
    
    def __init__(self):
        self.required_packages = [
            'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn',
            'scikit-learn', 'sentence-transformers', 'faiss-cpu'
        ]
        
    def validate_environment(self) -> Dict[str, Any]:
        """Validate that environment is suitable for reproducible evaluation"""
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'python_info': self._get_python_info(),
            'package_versions': self._get_package_versions(),
            'random_state_test': self._test_random_state(),
            'deterministic_computation_test': self._test_deterministic_computation(),
            'environment_hash': self._compute_environment_hash(),
            'validation_passed': True,
            'warnings': [],
            'errors': []
        }
        
        # Check for issues
        if not self._validate_package_versions(validation_results['package_versions']):
            validation_results['validation_passed'] = False
            validation_results['errors'].append("Missing or incompatible package versions")
        
        if not validation_results['random_state_test']['passed']:
            validation_results['validation_passed'] = False
            validation_results['errors'].append("Random state not deterministic")
        
        if not validation_results['deterministic_computation_test']['passed']:
            validation_results['validation_passed'] = False
            validation_results['errors'].append("Deterministic computation test failed")
        
        return validation_results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'platform': platform.platform(),
            'system': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'disk_free_gb': psutil.disk_usage('.').free / (1024**3)
        }
    
    def _get_python_info(self) -> Dict[str, Any]:
        """Get Python environment information"""
        return {
            'version': sys.version,
            'executable': sys.executable,
            'path': sys.path[:3],  # First few entries
            'hash_randomization': sys.hash_info.algorithm
        }
    
    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of required packages"""
        versions = {}
        
        for package in self.required_packages:
            try:
                if package == 'sentence-transformers':
                    import sentence_transformers
                    versions[package] = sentence_transformers.__version__
                elif package == 'faiss-cpu':
                    import faiss
                    versions['faiss'] = getattr(faiss, '__version__', 'unknown')
                else:
                    module = __import__(package)
                    versions[package] = getattr(module, '__version__', 'unknown')
            except ImportError:
                versions[package] = 'not_installed'
        
        return versions
    
    def _validate_package_versions(self, versions: Dict[str, str]) -> bool:
        """Validate that required packages are installed"""
        missing_packages = [pkg for pkg, version in versions.items() 
                           if version == 'not_installed']
        
        if missing_packages:
            logger.error(f"Missing packages: {missing_packages}")
            return False
        
        return True
    
    def _test_random_state(self) -> Dict[str, Any]:
        """Test that random state is deterministic"""
        
        # Test numpy random state
        np.random.seed(42)
        sample_1 = np.random.random(10)
        
        np.random.seed(42)
        sample_2 = np.random.random(10)
        
        numpy_deterministic = np.allclose(sample_1, sample_2)
        
        # Test Python random state
        import random
        random.seed(42)
        py_sample_1 = [random.random() for _ in range(10)]
        
        random.seed(42)
        py_sample_2 = [random.random() for _ in range(10)]
        
        python_deterministic = py_sample_1 == py_sample_2
        
        return {
            'numpy_deterministic': numpy_deterministic,
            'python_deterministic': python_deterministic,
            'passed': numpy_deterministic and python_deterministic
        }
    
    def _test_deterministic_computation(self) -> Dict[str, Any]:
        """Test that key computations are deterministic"""
        
        # Test matrix operations
        np.random.seed(42)
        matrix = np.random.random((100, 100))
        
        result_1 = np.linalg.svd(matrix, compute_uv=False)
        result_2 = np.linalg.svd(matrix, compute_uv=False)
        
        svd_deterministic = np.allclose(result_1, result_2)
        
        # Test sorting (should be stable)
        data = np.random.random(1000)
        indices_1 = np.argsort(data)
        indices_2 = np.argsort(data)
        
        sort_deterministic = np.array_equal(indices_1, indices_2)
        
        return {
            'svd_deterministic': svd_deterministic,
            'sort_deterministic': sort_deterministic,
            'passed': svd_deterministic and sort_deterministic
        }
    
    def _compute_environment_hash(self) -> str:
        """Compute hash of environment for comparison"""
        
        env_data = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'packages': self._get_package_versions()
        }
        
        env_string = json.dumps(env_data, sort_keys=True)
        return hashlib.md5(env_string.encode()).hexdigest()

class ReproducibilityValidator:
    """Validate reproducibility of evaluation results"""
    
    def __init__(self, tolerance_percent: float = 2.0):
        self.tolerance_percent = tolerance_percent
        self.environment_validator = EnvironmentValidator()
    
    def run_reproducibility_test(self,
                                dataset_path: Path,
                                output_dir: Path,
                                hardware_profile: str,
                                n_runs: int = 2,
                                quick_test: bool = True) -> Dict[str, Any]:
        """Run multiple evaluation runs and validate reproducibility"""
        
        logger.info(f"Starting reproducibility test with {n_runs} runs")
        
        # Validate environment first
        env_validation = self.environment_validator.validate_environment()
        
        if not env_validation['validation_passed']:
            logger.error("Environment validation failed:")
            for error in env_validation['errors']:
                logger.error(f"  - {error}")
            return {
                'reproducibility_passed': False,
                'environment_validation': env_validation,
                'error': 'Environment validation failed'
            }
        
        logger.info("Environment validation passed")
        
        # Run multiple evaluations
        run_results = []
        
        for run_idx in range(n_runs):
            logger.info(f"Starting run {run_idx + 1}/{n_runs}")
            
            run_output_dir = output_dir / f"reproducibility_run_{run_idx + 1}"
            run_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run evaluation
            try:
                run_result = self._run_single_evaluation(
                    dataset_path, run_output_dir, hardware_profile, quick_test
                )
                run_results.append(run_result)
                logger.info(f"Run {run_idx + 1} completed successfully")
                
            except Exception as e:
                logger.error(f"Run {run_idx + 1} failed: {e}")
                return {
                    'reproducibility_passed': False,
                    'environment_validation': env_validation,
                    'error': f'Run {run_idx + 1} failed: {e}'
                }
        
        # Compare results across runs
        comparison_results = self._compare_run_results(run_results)
        
        # Generate comprehensive report
        reproducibility_report = {
            'reproducibility_passed': comparison_results['within_tolerance'],
            'tolerance_percent': self.tolerance_percent,
            'n_runs': n_runs,
            'environment_validation': env_validation,
            'run_comparisons': comparison_results,
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(dataset_path),
            'hardware_profile': hardware_profile
        }
        
        # Save report
        report_file = output_dir / "reproducibility_report.json"
        with open(report_file, 'w') as f:
            json.dump(reproducibility_report, f, indent=2, default=str)
        
        logger.info(f"Reproducibility report saved to: {report_file}")
        
        # Log summary
        self._log_reproducibility_summary(reproducibility_report)
        
        return reproducibility_report
    
    def _run_single_evaluation(self,
                              dataset_path: Path,
                              output_dir: Path,
                              hardware_profile: str,
                              quick_test: bool) -> Dict[str, Any]:
        """Run a single evaluation and return results"""
        
        # Import the evaluation framework
        sys.path.append(str(Path(__file__).parent))
        try:
            from .milestone6_evaluation import Milestone6EvaluationFramework
        except ImportError:
            from milestone6_evaluation import Milestone6EvaluationFramework
        
        # Load dataset
        with open(dataset_path) as f:
            lethebench_data = json.load(f)
        
        # Create evaluation framework
        evaluator = Milestone6EvaluationFramework(
            output_dir=str(output_dir),
            hardware_profile=hardware_profile,
            lethebench_dataset=lethebench_data
        )
        
        # Prepare data
        from milestone6_evaluation import EvaluationQuery, RetrievalDocument
        
        queries_data = lethebench_data.get('queries', [])
        documents_data = lethebench_data.get('documents', [])
        
        if quick_test:
            queries_data = queries_data[:20]  # Small subset for testing
            documents_data = documents_data[:500]
        
        queries = [
            EvaluationQuery(
                query_id=q['query_id'],
                text=q['text'],
                domain=q.get('domain', 'agent_context'),
                complexity=q.get('complexity', 'medium'),
                relevance_judgments=q.get('relevance_judgments', {}),
                ground_truth_docs=q.get('ground_truth_docs', [])
            ) for q in queries_data
        ]
        
        documents = [
            RetrievalDocument(
                doc_id=d['doc_id'],
                content=d['content'],
                kind=d.get('kind', 'conversation_atom'),
                metadata=d.get('metadata', {})
            ) for d in documents_data
        ]
        
        # Run evaluation
        results = evaluator.run_complete_evaluation(documents, queries, k=100)
        
        # Extract key metrics for comparison
        key_metrics = self._extract_key_metrics(results)
        
        return {
            'full_results': results,
            'key_metrics': key_metrics,
            'output_dir': str(output_dir),
            'timestamp': datetime.now().isoformat()
        }
    
    def _extract_key_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract key metrics for reproducibility comparison"""
        
        key_metrics = {}
        
        # IR metrics
        if 'ir_metrics' in results and 'overall' in results['ir_metrics']:
            overall_metrics = results['ir_metrics']['overall']
            
            for baseline, baseline_metrics in overall_metrics.items():
                for metric in ['ndcg_10_mean', 'ndcg_20_mean', 'recall_10_mean', 'recall_20_mean', 'mrr_10_mean']:
                    if metric in baseline_metrics:
                        key_metrics[f'{baseline}_{metric}'] = baseline_metrics[metric]
        
        # Agent metrics
        if 'agent_metrics' in results and 'by_baseline' in results['agent_metrics']:
            agent_metrics = results['agent_metrics']['by_baseline']
            
            for baseline, baseline_metrics in agent_metrics.items():
                for metric in ['tool_result_recall_at_10_mean', 'action_argument_consistency_mean']:
                    if metric in baseline_metrics:
                        key_metrics[f'{baseline}_{metric}'] = baseline_metrics[metric]
        
        # Statistical results
        if 'statistical_results' in results and 'summary' in results['statistical_results']:
            stat_summary = results['statistical_results']['summary']
            for metric in ['total_tests_run', 'significant_tests']:
                if metric in stat_summary:
                    key_metrics[f'stats_{metric}'] = float(stat_summary[metric])
        
        return key_metrics
    
    def _compare_run_results(self, run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare results across multiple runs"""
        
        if len(run_results) < 2:
            return {
                'error': 'Need at least 2 runs for comparison',
                'within_tolerance': False
            }
        
        # Extract key metrics from all runs
        all_metrics = []
        for run_result in run_results:
            all_metrics.append(run_result['key_metrics'])
        
        # Pairwise comparisons
        comparison_tests = []
        
        # Compare run 1 vs run 2 (main comparison)
        metrics_1 = all_metrics[0]
        metrics_2 = all_metrics[1]
        
        common_metrics = set(metrics_1.keys()) & set(metrics_2.keys())
        
        for metric_name in common_metrics:
            value_1 = metrics_1[metric_name]
            value_2 = metrics_2[metric_name]
            
            test_result = self._compare_metric_values(
                metric_name, value_1, value_2, self.tolerance_percent
            )
            comparison_tests.append(test_result)
        
        # Summary statistics
        total_tests = len(comparison_tests)
        passed_tests = sum(1 for test in comparison_tests if test.within_tolerance)
        within_tolerance = (passed_tests / total_tests) >= 0.95 if total_tests > 0 else False
        
        # Detailed analysis
        failed_tests = [test for test in comparison_tests if not test.within_tolerance]
        
        return {
            'within_tolerance': within_tolerance,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
            'tolerance_percent': self.tolerance_percent,
            'comparison_tests': [
                {
                    'metric_name': test.metric_name,
                    'run_1_value': test.run_1_value,
                    'run_2_value': test.run_2_value,
                    'absolute_diff': test.absolute_diff,
                    'relative_diff_percent': test.relative_diff_percent,
                    'within_tolerance': test.within_tolerance
                }
                for test in comparison_tests
            ],
            'failed_tests': [
                {
                    'metric_name': test.metric_name,
                    'relative_diff_percent': test.relative_diff_percent,
                    'tolerance_percent': test.tolerance_percent
                }
                for test in failed_tests
            ]
        }
    
    def _compare_metric_values(self,
                              metric_name: str,
                              value_1: float,
                              value_2: float,
                              tolerance_percent: float) -> ReproducibilityTest:
        """Compare two metric values within tolerance"""
        
        absolute_diff = abs(value_1 - value_2)
        
        # Handle zero values
        if value_1 == 0 and value_2 == 0:
            relative_diff_percent = 0.0
        elif value_1 == 0 or value_2 == 0:
            # If one is zero, use absolute difference
            relative_diff_percent = absolute_diff * 100
        else:
            # Standard relative difference
            relative_diff_percent = (absolute_diff / abs(value_1)) * 100
        
        within_tolerance = relative_diff_percent <= tolerance_percent
        
        return ReproducibilityTest(
            metric_name=metric_name,
            run_1_value=value_1,
            run_2_value=value_2,
            absolute_diff=absolute_diff,
            relative_diff_percent=relative_diff_percent,
            within_tolerance=within_tolerance,
            tolerance_percent=tolerance_percent
        )
    
    def _log_reproducibility_summary(self, report: Dict[str, Any]) -> None:
        """Log summary of reproducibility test results"""
        
        logger.info("="*60)
        logger.info("REPRODUCIBILITY TEST RESULTS")
        logger.info("="*60)
        
        # Environment validation
        env_passed = report['environment_validation']['validation_passed']
        logger.info(f"Environment Validation: {'✅ PASSED' if env_passed else '❌ FAILED'}")
        
        if not env_passed:
            for error in report['environment_validation']['errors']:
                logger.error(f"  ❌ {error}")
        
        # Reproducibility results
        repro_passed = report['reproducibility_passed']
        logger.info(f"Reproducibility Test: {'✅ PASSED' if repro_passed else '❌ FAILED'}")
        
        if 'run_comparisons' in report:
            comp = report['run_comparisons']
            logger.info(f"Test Pass Rate: {comp['passed_tests']}/{comp['total_tests']} ({comp['pass_rate']:.1%})")
            logger.info(f"Tolerance: ±{comp['tolerance_percent']:.1f}%")
            
            # Failed tests
            if 'failed_tests' in comp and comp['failed_tests']:
                logger.warning("Failed Tests:")
                for test in comp['failed_tests'][:5]:  # Show first 5 failures
                    logger.warning(f"  ❌ {test['metric_name']}: "
                                 f"{test['relative_diff_percent']:.2f}% > {test['tolerance_percent']:.1f}%")
        
        logger.info("="*60)
        
        # Overall verdict
        if env_passed and repro_passed:
            logger.info("🎉 EVALUATION IS REPRODUCIBLE WITHIN TOLERANCE")
        else:
            logger.error("💥 EVALUATION FAILED REPRODUCIBILITY TEST")
        
        logger.info("="*60)

def main():
    """Main entry point for reproducibility validation"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate reproducibility of Milestone 6 evaluation"
    )
    parser.add_argument("--dataset", type=Path, required=True,
                       help="LetheBench-Agents dataset JSON file")
    parser.add_argument("--output-dir", type=Path, default="./reproducibility_test",
                       help="Output directory for test results")
    parser.add_argument("--hardware-profile", type=str,
                       default=f"{platform.system()}_{platform.machine()}",
                       help="Hardware profile identifier")
    parser.add_argument("--tolerance", type=float, default=2.0,
                       help="Tolerance percentage for reproducibility (default: 2.0%)")
    parser.add_argument("--n-runs", type=int, default=2,
                       help="Number of evaluation runs to compare (default: 2)")
    parser.add_argument("--quick-test", action="store_true",
                       help="Use quick test mode with reduced dataset")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "reproducibility_validation.log")
        ]
    )
    
    # Create validator
    validator = ReproducibilityValidator(tolerance_percent=args.tolerance)
    
    # Run test
    try:
        results = validator.run_reproducibility_test(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            hardware_profile=args.hardware_profile,
            n_runs=args.n_runs,
            quick_test=args.quick_test
        )
        
        # Return exit code based on results
        if results['reproducibility_passed']:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure
            
    except Exception as e:
        logger.error(f"Reproducibility validation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()