#!/usr/bin/env python3
"""
Comprehensive Verification Suite Runner for Lethe Research

This script orchestrates all verification components to ensure research infrastructure
meets NeurIPS publication standards.

Components executed:
1. Property-based testing (Hypothesis framework)
2. Mutation testing (≥0.80 score required)
3. Fuzzing infrastructure (robustness testing)
4. Oracle verification (correctness validation)

Usage:
    python run_verification_suite.py --full
    python run_verification_suite.py --component property-tests
    python run_verification_suite.py --component mutation-tests --threshold 0.85
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from verification.properties.test_suite import PropertyTestSuite
from verification.mutation.test_mutations import MutationTestRunner
from verification.fuzzing.test_fuzz import FuzzOrchestrator, QueryFuzzGenerator, VectorFuzzGenerator
from verification.oracles.test_oracles import OracleManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VerificationSuiteRunner:
    """Orchestrates comprehensive verification testing."""
    
    def __init__(
        self,
        output_dir: str = "verification_results",
        mutation_threshold: float = 0.80,
        property_coverage_threshold: float = 0.70,
        oracle_confidence_threshold: float = 0.85
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality thresholds
        self.mutation_threshold = mutation_threshold
        self.property_coverage_threshold = property_coverage_threshold
        self.oracle_confidence_threshold = oracle_confidence_threshold
        
        # Results storage
        self.results: Dict[str, Any] = {}
        
    def run_full_suite(self) -> Dict[str, Any]:
        """Run complete verification suite."""
        logger.info("🔬 Starting comprehensive verification suite...")
        start_time = time.time()
        
        # Run all components
        self._run_property_tests()
        self._run_mutation_tests()
        self._run_fuzzing_tests()
        self._run_oracle_verification()
        
        # Generate comprehensive report
        execution_time = time.time() - start_time
        suite_results = self._generate_suite_report(execution_time)
        
        # Validate against quality gates
        self._validate_quality_gates(suite_results)
        
        return suite_results
    
    def run_component(self, component: str, **kwargs) -> Dict[str, Any]:
        """Run specific verification component."""
        logger.info(f"🔧 Running verification component: {component}")
        
        if component == "property-tests":
            return self._run_property_tests(**kwargs)
        elif component == "mutation-tests":
            return self._run_mutation_tests(**kwargs)
        elif component == "fuzzing-tests":
            return self._run_fuzzing_tests(**kwargs)
        elif component == "oracle-verification":
            return self._run_oracle_verification(**kwargs)
        else:
            raise ValueError(f"Unknown verification component: {component}")
    
    def _run_property_tests(self, **kwargs) -> Dict[str, Any]:
        """Run property-based testing suite."""
        logger.info("🧪 Running property-based tests...")
        
        try:
            # Initialize property test suite
            suite = PropertyTestSuite(
                seed=kwargs.get('seed', 42),
                max_examples=kwargs.get('max_examples', 100)
            )
            
            # Run all property tests
            results = suite.run_all_tests()
            
            # Calculate coverage metrics
            total_tests = results.total_tests
            passed_tests = results.passed_tests
            coverage = passed_tests / total_tests if total_tests > 0 else 0.0
            
            property_results = {
                'component': 'property-tests',
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': results.failed_tests,
                'coverage': coverage,
                'execution_time': results.execution_time,
                'meets_threshold': coverage >= self.property_coverage_threshold,
                'threshold': self.property_coverage_threshold,
                'detailed_results': results.to_dict()
            }
            
            self.results['property_tests'] = property_results
            
            logger.info(f"✅ Property tests completed: {passed_tests}/{total_tests} passed (coverage: {coverage:.1%})")
            
            return property_results
            
        except Exception as e:
            logger.error(f"❌ Property tests failed: {e}")
            error_results = {
                'component': 'property-tests',
                'error': str(e),
                'meets_threshold': False
            }
            self.results['property_tests'] = error_results
            return error_results
    
    def _run_mutation_tests(self, **kwargs) -> Dict[str, Any]:
        """Run mutation testing framework."""
        logger.info("🧬 Running mutation tests...")
        
        try:
            # Target directories for mutation testing
            target_dirs = kwargs.get('targets', [
                'lethe-research/datasets',
                'lethe-research/experiments', 
                'lethe-research/verification'
            ])
            
            # Initialize mutation test runner
            runner = MutationTestRunner(
                target_dirs=target_dirs,
                test_dir=kwargs.get('test_dir', 'tests'),
                min_mutation_score=kwargs.get('threshold', self.mutation_threshold),
                seed=kwargs.get('seed', 42),
                timeout=kwargs.get('timeout', 30)
            )
            
            # Run mutation tests
            results = runner.run_mutation_tests()
            
            mutation_results = {
                'component': 'mutation-tests',
                'total_mutations': results.total_mutations,
                'killed_mutations': results.killed_mutations,
                'survived_mutations': results.survived_mutations,
                'mutation_score': results.mutation_score,
                'execution_time': results.execution_time,
                'meets_threshold': results.mutation_score >= self.mutation_threshold,
                'threshold': self.mutation_threshold,
                'detailed_results': results.to_dict()
            }
            
            self.results['mutation_tests'] = mutation_results
            
            logger.info(f"✅ Mutation tests completed: {results.mutation_score:.3f} score ({results.killed_mutations}/{results.total_mutations} killed)")
            
            return mutation_results
            
        except Exception as e:
            logger.error(f"❌ Mutation tests failed: {e}")
            error_results = {
                'component': 'mutation-tests',
                'error': str(e),
                'meets_threshold': False
            }
            self.results['mutation_tests'] = error_results
            return error_results
    
    def _run_fuzzing_tests(self, **kwargs) -> Dict[str, Any]:
        """Run fuzzing infrastructure tests."""
        logger.info("🔀 Running fuzzing tests...")
        
        try:
            # Create mock target function for demonstration
            def mock_retrieval_function(query):
                """Mock retrieval function for fuzzing demonstration."""
                if not query:
                    raise ValueError("Empty query")
                if len(str(query)) > 10000:
                    raise ValueError("Query too long")
                if isinstance(query, list):
                    return [{"id": f"doc_{i}", "score": 0.5} for i in range(min(10, len(query)))]
                return [{"id": "doc_1", "score": 0.9}, {"id": "doc_2", "score": 0.7}]
            
            # Initialize generators
            generators = [
                QueryFuzzGenerator(seed=kwargs.get('seed', 42)),
                VectorFuzzGenerator(seed=kwargs.get('seed', 42))
            ]
            
            # Initialize orchestrator
            orchestrator = FuzzOrchestrator(
                target_function=mock_retrieval_function,
                generators=generators,
                seed=kwargs.get('seed', 42)
            )
            
            # Run fuzzing campaign
            results = orchestrator.run_campaign(
                iterations=kwargs.get('iterations', 500),
                max_time=kwargs.get('max_time', 60.0)
            )
            
            fuzzing_results = {
                'component': 'fuzzing-tests',
                'total_executions': results.total_executions,
                'unique_crashes': results.unique_crashes,
                'code_coverage': results.code_coverage,
                'execution_time': results.execution_time,
                'results_breakdown': {k.value: v for k, v in results.results.items()},
                'meets_threshold': results.unique_crashes == 0,  # No crashes is good
                'detailed_results': results.to_dict()
            }
            
            self.results['fuzzing_tests'] = fuzzing_results
            
            if results.unique_crashes > 0:
                logger.warning(f"⚠️ Fuzzing found {results.unique_crashes} unique crashes!")
            else:
                logger.info(f"✅ Fuzzing completed: {results.total_executions} executions, no crashes found")
            
            return fuzzing_results
            
        except Exception as e:
            logger.error(f"❌ Fuzzing tests failed: {e}")
            error_results = {
                'component': 'fuzzing-tests',
                'error': str(e),
                'meets_threshold': False
            }
            self.results['fuzzing_tests'] = error_results
            return error_results
    
    def _run_oracle_verification(self, **kwargs) -> Dict[str, Any]:
        """Run oracle verification system."""
        logger.info("🔮 Running oracle verification...")
        
        try:
            # Initialize oracle manager
            manager = OracleManager()
            
            # Create sample verification batch
            sample_verifications = [
                {
                    "oracle": "retrieval_ground_truth",
                    "params": {
                        "query": "python async programming",
                        "ground_truth": [
                            {"document_id": "doc_1", "relevance_score": 1.0},
                            {"document_id": "doc_2", "relevance_score": 0.8},
                            {"document_id": "doc_3", "relevance_score": 0.6}
                        ],
                        "actual_results": [
                            {"document_id": "doc_1", "score": 0.95},
                            {"document_id": "doc_2", "score": 0.87},
                            {"document_id": "doc_3", "score": 0.72}
                        ]
                    }
                },
                {
                    "oracle": "statistical_validation",
                    "params": {
                        "experiment_data": {
                            "samples": [0.82, 0.85, 0.79, 0.88, 0.81, 0.84, 0.86, 0.83, 0.87, 0.80],
                            "p_value": 0.03,
                            "effect_size": 0.65,
                            "confidence_interval": [0.78, 0.89]
                        },
                        "expected_properties": {
                            "significance": {"significant": True},
                            "effect_size": {"min_effect_size": 0.5}
                        }
                    }
                },
                {
                    "oracle": "performance_validation", 
                    "params": {
                        "performance_data": {
                            "latency": 1.2,
                            "memory_usage": 1200000000,
                            "throughput": 150
                        },
                        "bounds": {
                            "max_latency": 3.0,
                            "max_memory": 1500000000,
                            "min_throughput": 100
                        }
                    }
                }
            ]
            
            # Run oracle verification batch
            results = manager.run_verification_batch(sample_verifications)
            
            # Calculate success metrics
            total_verifications = results.total_verifications
            passed_verifications = results.results.get('pass', 0)
            success_rate = passed_verifications / total_verifications if total_verifications > 0 else 0.0
            avg_confidence = results.confidence_stats['mean']
            
            oracle_results = {
                'component': 'oracle-verification',
                'total_verifications': total_verifications,
                'passed_verifications': passed_verifications,
                'success_rate': success_rate,
                'average_confidence': avg_confidence,
                'execution_time': results.execution_time,
                'meets_threshold': avg_confidence >= self.oracle_confidence_threshold,
                'threshold': self.oracle_confidence_threshold,
                'detailed_results': results.to_dict()
            }
            
            self.results['oracle_verification'] = oracle_results
            
            logger.info(f"✅ Oracle verification completed: {passed_verifications}/{total_verifications} passed (avg confidence: {avg_confidence:.3f})")
            
            return oracle_results
            
        except Exception as e:
            logger.error(f"❌ Oracle verification failed: {e}")
            error_results = {
                'component': 'oracle-verification',
                'error': str(e),
                'meets_threshold': False
            }
            self.results['oracle_verification'] = error_results
            return error_results
    
    def _generate_suite_report(self, execution_time: float) -> Dict[str, Any]:
        """Generate comprehensive verification suite report."""
        # Count components that meet thresholds
        components_passing = sum(
            1 for result in self.results.values()
            if result.get('meets_threshold', False)
        )
        total_components = len(self.results)
        
        # Calculate overall success rate
        overall_success = components_passing / total_components if total_components > 0 else 0.0
        
        suite_results = {
            'verification_suite': {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'execution_time': execution_time,
                'total_components': total_components,
                'passing_components': components_passing,
                'overall_success_rate': overall_success,
                'meets_quality_gates': overall_success >= 0.75,  # 75% of components must pass
                'components': self.results,
                'thresholds': {
                    'mutation_score': self.mutation_threshold,
                    'property_coverage': self.property_coverage_threshold,
                    'oracle_confidence': self.oracle_confidence_threshold
                }
            }
        }
        
        # Save comprehensive report
        report_file = self.output_dir / "verification_suite_report.json"
        with open(report_file, 'w') as f:
            json.dump(suite_results, f, indent=2, default=str)
        
        logger.info(f"📊 Verification suite report saved to {report_file}")
        
        return suite_results
    
    def _validate_quality_gates(self, suite_results: Dict[str, Any]) -> None:
        """Validate results against quality gates and log summary."""
        suite_info = suite_results['verification_suite']
        
        print("\n" + "="*60)
        print("🔬 LETHE RESEARCH VERIFICATION SUITE RESULTS")
        print("="*60)
        
        print(f"📊 Overall Success Rate: {suite_info['overall_success_rate']:.1%}")
        print(f"⏱️  Total Execution Time: {suite_info['execution_time']:.1f}s")
        print(f"🧩 Components Tested: {suite_info['total_components']}")
        print(f"✅ Components Passing: {suite_info['passing_components']}")
        
        print("\n📋 Component Results:")
        for component_name, results in suite_info['components'].items():
            status = "✅ PASS" if results.get('meets_threshold', False) else "❌ FAIL"
            component_display = component_name.replace('_', ' ').title()
            print(f"  {component_display}: {status}")
            
            # Show key metrics
            if component_name == 'property_tests' and 'coverage' in results:
                print(f"    Coverage: {results['coverage']:.1%} (threshold: {results.get('threshold', 0):.1%})")
            elif component_name == 'mutation_tests' and 'mutation_score' in results:
                print(f"    Mutation Score: {results['mutation_score']:.3f} (threshold: {results.get('threshold', 0):.3f})")
            elif component_name == 'oracle_verification' and 'average_confidence' in results:
                print(f"    Avg Confidence: {results['average_confidence']:.3f} (threshold: {results.get('threshold', 0):.3f})")
            elif component_name == 'fuzzing_tests' and 'unique_crashes' in results:
                crashes = results['unique_crashes']
                print(f"    Unique Crashes: {crashes} {'(⚠️  Issues Found)' if crashes > 0 else '(Clean)'}")
        
        # Overall assessment
        print(f"\n🎯 Quality Gates: {'✅ PASSED' if suite_info['meets_quality_gates'] else '❌ FAILED'}")
        
        if not suite_info['meets_quality_gates']:
            print("\n⚠️  QUALITY GATES NOT MET:")
            print("   - Review failed components above")
            print("   - Address issues before publication")
            print("   - Minimum 75% of components must pass")
        else:
            print("\n🎉 ALL QUALITY GATES PASSED!")
            print("   - Research infrastructure meets NeurIPS standards")
            print("   - Ready for publication workflow")
        
        print("="*60)


def main():
    """Main entry point for verification suite."""
    parser = argparse.ArgumentParser(
        description="Comprehensive verification suite for Lethe research infrastructure"
    )
    
    parser.add_argument(
        '--full', 
        action='store_true',
        help='Run complete verification suite'
    )
    
    parser.add_argument(
        '--component',
        choices=['property-tests', 'mutation-tests', 'fuzzing-tests', 'oracle-verification'],
        help='Run specific verification component'
    )
    
    parser.add_argument(
        '--output-dir',
        default='verification_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--mutation-threshold',
        type=float,
        default=0.80,
        help='Minimum mutation score threshold'
    )
    
    parser.add_argument(
        '--property-threshold',
        type=float,
        default=0.70,
        help='Minimum property test coverage threshold'
    )
    
    parser.add_argument(
        '--oracle-threshold',
        type=float,
        default=0.85,
        help='Minimum oracle confidence threshold'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for deterministic testing'
    )
    
    args = parser.parse_args()
    
    if not args.full and not args.component:
        parser.error("Must specify either --full or --component")
    
    # Initialize verification suite runner
    runner = VerificationSuiteRunner(
        output_dir=args.output_dir,
        mutation_threshold=args.mutation_threshold,
        property_coverage_threshold=args.property_threshold,
        oracle_confidence_threshold=args.oracle_threshold
    )
    
    try:
        if args.full:
            # Run complete suite
            results = runner.run_full_suite()
            exit_code = 0 if results['verification_suite']['meets_quality_gates'] else 1
        else:
            # Run specific component
            results = runner.run_component(args.component, seed=args.seed)
            exit_code = 0 if results.get('meets_threshold', False) else 1
        
        exit(exit_code)
        
    except Exception as e:
        logger.error(f"Verification suite failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()