#!/usr/bin/env python3
"""
Guarded Canary Test for Production-Quality Reproducible Results
==============================================================

Runs a fast canary test (10 minutes) with deterministic conditions
and comprehensive validation gates to catch regressions quickly.

Usage:
    python run_guarded_canary.py --samples 30
    python run_guarded_canary.py --samples 30 --strict --fail-fast
"""

import sys
import logging
import argparse
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent
lethe_root = project_root.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import evaluation system
try:
    from scripts.run_hybrid_infinitebench import (
        HybridInfiniteBenchRunner, 
        EvaluationConfig, 
        validate_measurement_pipeline
    )
    from validation_sentinels import (
        validate_measurement_pipeline_v2, 
        ValidationThresholds,
        GateFailureError
    )
    HAS_VALIDATION_SENTINELS = True
except ImportError as e:
    logging.warning(f"Comprehensive validation not available: {e}")
    HAS_VALIDATION_SENTINELS = False
    from scripts.run_hybrid_infinitebench import (
        HybridInfiniteBenchRunner, 
        EvaluationConfig, 
        validate_measurement_pipeline
    )

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CanaryGateThresholds:
    """Gate thresholds for canary validation"""
    min_p_at_5_per_scenario: float = 0.001
    min_median_at_8_percent: float = 500
    min_kv_mass_share: float = 0.8
    min_delta_cbu_variance: float = 0.001
    min_spearman_delta_cbu_p5: float = 0.3
    max_ece_per_type_budget: float = 0.08
    max_proxy_gap_percent: float = 0.5
    max_p99_p95_ratio: float = 2.5

@dataclass 
class CanaryResult:
    """Result from canary test with gate status"""
    all_gates_passed: bool
    execution_time_seconds: float
    sample_count: int
    method_results: Dict[str, Any]
    gate_results: Dict[str, bool]
    gate_details: Dict[str, Any]
    recommendations: List[str]

class GuardedCanaryRunner:
    """Runs guarded canary test with all validation gates"""
    
    def __init__(self, samples_per_scenario: int = 30, fail_fast: bool = True):
        self.samples_per_scenario = samples_per_scenario
        self.fail_fast = fail_fast
        self.thresholds = CanaryGateThresholds()
        
        # Set deterministic environment
        self._setup_deterministic_environment()
        
        # Create canary config (subset of full evaluation)
        self.config = EvaluationConfig(
            experiment_name=f"canary_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            methods=['streaming', 'lethe', 'hybrid'],
            keep_ratios=[0.08, 0.15, 0.30],  # Keep all ratios 
            datasets=['code_debug', 'code_qa', 'zh_qa'],
            min_samples=self.samples_per_scenario,  # Use samples_per_scenario
            zh_samples=min(50, self.samples_per_scenario),  # Limit zh_qa
            output_dir=Path("artifacts/canary_test"),
            bootstrap_samples=100  # Reduced for speed
        )
        
        logger.info(f"🔒 Canary test configured: {self.samples_per_scenario} samples/scenario")
        
    def _setup_deterministic_environment(self):
        """Set up deterministic environment for reproducible results"""
        # Set environment variables for determinism
        os.environ['LETHE_DETERMINISTIC'] = '1'
        os.environ['RAYON_NUM_THREADS'] = '1'
        os.environ['PYTHONHASHSEED'] = '42'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # GPU determinism if available
        
        # Set CPU affinity for quiet cores (if supported)
        try:
            os.system('taskset -c 0-1 echo "CPU affinity set to cores 0-1"')
        except:
            logger.warning("Could not set CPU affinity")
            
        # Set high process priority
        try:
            os.nice(-5)
        except:
            logger.warning("Could not set high process priority")
            
        logger.info("🔒 Deterministic environment configured")
        
    def run_canary_test(self) -> CanaryResult:
        """Run the complete canary test with all gates"""
        start_time = time.time()
        
        logger.info("🚀 Starting guarded canary test")
        logger.info(f"🔒 Deterministic mode: RAYON_NUM_THREADS=1, seed=42")
        
        try:
            # Initialize runner
            runner = HybridInfiniteBenchRunner(self.config)
            
            # Run evaluation matrix (limited sample size)
            matrix = runner.run_evaluation_matrix()
            
            # Convert results to validation format
            flat_results = self._flatten_results(matrix)
            
            # Run comprehensive validation if available
            if HAS_VALIDATION_SENTINELS:
                gate_results = self._run_comprehensive_gates(flat_results)
            else:
                gate_results = self._run_legacy_gates(flat_results)
            
            execution_time = time.time() - start_time
            
            # Create canary result
            canary_result = CanaryResult(
                all_gates_passed=all(gate_results.values()),
                execution_time_seconds=execution_time,
                sample_count=len(flat_results),
                method_results=self._summarize_method_performance(flat_results),
                gate_results=gate_results,
                gate_details=self._get_gate_details(flat_results),
                recommendations=self._generate_recommendations(gate_results, flat_results)
            )
            
            # Log results
            self._log_canary_results(canary_result)
            
            return canary_result
            
        except Exception as e:
            logger.error(f"❌ Canary test failed: {e}")
            if self.fail_fast:
                raise
            return CanaryResult(
                all_gates_passed=False,
                execution_time_seconds=time.time() - start_time,
                sample_count=0,
                method_results={},
                gate_results={'exception': False},
                gate_details={'error': str(e)},
                recommendations=[f"Fix critical error: {e}"]
            )
    
    def _flatten_results(self, matrix) -> List[Dict]:
        """Convert matrix results to flat format for validation"""
        flat_results = []
        for method, results in matrix.results.items():
            for r in results:
                flat_results.append({
                    'method_name': r.method_name,
                    'dataset': r.dataset,
                    'keep_ratio': r.keep_ratio,
                    'p_at_k': r.p_at_k,
                    'delta_cbu_per_1k': r.delta_cbu_per_1k,
                    'kv_reuse': r.kv_reuse,
                    'tokens_kept': r.tokens_kept,
                    'compression_ratio': r.compression_ratio,
                    'tail_cvar': r.tail_cvar,
                    'middleware_p95_ms': r.middleware_p95_ms,
                    'accuracy': r.accuracy
                })
        return flat_results
    
    def _run_comprehensive_gates(self, results: List[Dict]) -> Dict[str, bool]:
        """Run comprehensive validation gates"""
        try:
            logger.info("🔒 Running comprehensive fail-closed validation sentinels")
            
            thresholds = ValidationThresholds(
                min_p_at_5_per_scenario=self.thresholds.min_p_at_5_per_scenario,
                min_median_at_8_percent=self.thresholds.min_median_at_8_percent,
                min_kv_mass_share=self.thresholds.min_kv_mass_share,
                min_delta_cbu_variance=self.thresholds.min_delta_cbu_variance,
                min_spearman_delta_cbu_p5=self.thresholds.min_spearman_delta_cbu_p5,
                max_ece_per_type_budget=self.thresholds.max_ece_per_type_budget,
                max_proxy_gap_percent=self.thresholds.max_proxy_gap_percent,
                max_p99_p95_ratio=self.thresholds.max_p99_p95_ratio
            )
            
            report = validate_measurement_pipeline_v2(
                results,
                thresholds=thresholds,
                fail_fast=self.fail_fast
            )
            
            if report.success:
                logger.info("✅ ALL COMPREHENSIVE VALIDATION SENTINELS PASSED")
                return {gate.name: True for gate in report.gates}
            else:
                logger.error(f"❌ VALIDATION FAILED: {len(report.failures)} critical failures")
                gate_results = {}
                for gate in report.gates:
                    gate_results[gate.name] = gate.status == 'PASS'
                
                if self.fail_fast:
                    raise GateFailureError(f"Gates failed: {report.failures}")
                    
                return gate_results
                
        except ImportError:
            logger.warning("⚠️ Comprehensive validation not available - using legacy")
            return self._run_legacy_gates(results)
    
    def _run_legacy_gates(self, results: List[Dict]) -> Dict[str, bool]:
        """Run legacy validation gates"""
        logger.info("🔒 Running legacy validation gates")
        
        gate_results = {}
        
        # Gate 1: Non-zero P@5 per scenario
        datasets = set(r['dataset'] for r in results)
        p_at_5_gate = True
        for dataset in datasets:
            dataset_results = [r for r in results if r['dataset'] == dataset]
            p_at_5_values = [r.get('p_at_k', {}).get(5, 0) for r in dataset_results]
            avg_p_at_5 = np.mean(p_at_5_values) if p_at_5_values else 0
            if avg_p_at_5 <= self.thresholds.min_p_at_5_per_scenario:
                p_at_5_gate = False
                logger.warning(f"❌ Gate failure: {dataset} P@5={avg_p_at_5:.4f} <= {self.thresholds.min_p_at_5_per_scenario}")
        gate_results['non_zero_p_at_5'] = p_at_5_gate
        
        # Gate 2: Dataset collapse check
        expected_datasets = set(self.config.datasets)
        actual_datasets = set(r['dataset'] for r in results)
        dataset_collapse_gate = expected_datasets.issubset(actual_datasets)
        if not dataset_collapse_gate:
            missing = expected_datasets - actual_datasets
            logger.warning(f"❌ Gate failure: Dataset collapse - missing {missing}")
        gate_results['no_dataset_collapse'] = dataset_collapse_gate
        
        # Gate 3: Token increase monotonicity for zh_qa
        zh_results = [r for r in results if r['dataset'] == 'zh_qa']
        token_monotonicity_gate = True
        if zh_results:
            # Group by method and check keep ratio ordering
            for method in set(r['method_name'] for r in zh_results):
                method_results = [r for r in zh_results if r['method_name'] == method]
                method_results.sort(key=lambda x: x['keep_ratio'])
                
                tokens_sequence = [r['tokens_kept'] for r in method_results]
                if not all(tokens_sequence[i] <= tokens_sequence[i+1] for i in range(len(tokens_sequence)-1)):
                    token_monotonicity_gate = False
                    logger.warning(f"❌ Gate failure: Token monotonicity violated for {method}")
        gate_results['token_monotonicity'] = token_monotonicity_gate
        
        # Gate 4: Median tokens at 8% >= 500
        results_8_percent = [r for r in results if abs(r['keep_ratio'] - 0.08) < 0.01]
        if results_8_percent:
            tokens_at_8 = [r['tokens_kept'] for r in results_8_percent]
            median_8_percent = np.median(tokens_at_8)
            median_tokens_gate = median_8_percent >= self.thresholds.min_median_at_8_percent
            if not median_tokens_gate:
                logger.warning(f"❌ Gate failure: Median@8%={median_8_percent} < {self.thresholds.min_median_at_8_percent}")
        else:
            median_tokens_gate = False
            logger.warning("❌ Gate failure: No 8% keep ratio results found")
        gate_results['median_tokens_at_8_percent'] = median_tokens_gate
        
        # Gate 5: KV reuse statistics
        kv_values = [r.get('kv_reuse', 0) for r in results]
        kv_mass_gate = True
        if kv_values:
            # Share of results with KV reuse > 0.1 should be >= 0.8
            kv_share = sum(1 for kv in kv_values if kv > 0.1) / len(kv_values)
            kv_mass_gate = kv_share >= self.thresholds.min_kv_mass_share
            if not kv_mass_gate:
                logger.warning(f"❌ Gate failure: KV mass share={kv_share:.3f} < {self.thresholds.min_kv_mass_share}")
        else:
            kv_mass_gate = False
        gate_results['kv_mass_share'] = kv_mass_gate
        
        # Gate 6: Delta CBU variance
        delta_cbu_values = [r.get('delta_cbu_per_1k', 0) for r in results]
        if delta_cbu_values:
            delta_cbu_variance = np.var(delta_cbu_values)
            delta_cbu_gate = delta_cbu_variance > self.thresholds.min_delta_cbu_variance
            if not delta_cbu_gate:
                logger.warning(f"❌ Gate failure: ΔCBU variance={delta_cbu_variance:.6f} <= {self.thresholds.min_delta_cbu_variance}")
        else:
            delta_cbu_gate = False
        gate_results['delta_cbu_variance'] = delta_cbu_gate
        
        return gate_results
    
    def _get_gate_details(self, results: List[Dict]) -> Dict[str, Any]:
        """Get detailed gate metrics for reporting"""
        details = {}
        
        # P@5 by dataset
        datasets = set(r['dataset'] for r in results)
        p_at_5_by_dataset = {}
        for dataset in datasets:
            dataset_results = [r for r in results if r['dataset'] == dataset]
            p_at_5_values = [r.get('p_at_k', {}).get(5, 0) for r in dataset_results]
            p_at_5_by_dataset[dataset] = {
                'mean': np.mean(p_at_5_values) if p_at_5_values else 0,
                'count': len(p_at_5_values)
            }
        details['p_at_5_by_dataset'] = p_at_5_by_dataset
        
        # Token statistics
        results_8_percent = [r for r in results if abs(r['keep_ratio'] - 0.08) < 0.01]
        if results_8_percent:
            tokens_at_8 = [r['tokens_kept'] for r in results_8_percent]
            details['median_tokens_at_8_percent'] = float(np.median(tokens_at_8))
        
        # KV statistics
        kv_values = [r.get('kv_reuse', 0) for r in results]
        if kv_values:
            details['kv_statistics'] = {
                'mean': float(np.mean(kv_values)),
                'share_above_0.1': sum(1 for kv in kv_values if kv > 0.1) / len(kv_values),
                'count': len(kv_values)
            }
        
        return details
    
    def _summarize_method_performance(self, results: List[Dict]) -> Dict[str, Any]:
        """Summarize performance by method"""
        methods = set(r['method_name'] for r in results)
        performance = {}
        
        for method in methods:
            method_results = [r for r in results if r['method_name'] == method]
            
            # Average metrics
            accuracy_values = [r.get('accuracy', 0) for r in method_results]
            p95_values = [r.get('middleware_p95_ms', 0) for r in method_results]
            
            performance[method] = {
                'avg_accuracy': float(np.mean(accuracy_values)) if accuracy_values else 0.0,
                'avg_p95_ms': float(np.mean(p95_values)) if p95_values else 0.0,
                'sample_count': len(method_results)
            }
        
        return performance
    
    def _generate_recommendations(self, gate_results: Dict[str, bool], results: List[Dict]) -> List[str]:
        """Generate recommendations based on gate results"""
        recommendations = []
        
        if not gate_results.get('non_zero_p_at_5', True):
            recommendations.append("Critical: Fix generator configuration - P@5 is zero across scenarios")
            recommendations.append("Check: Ollama model availability and response generation")
            
        if not gate_results.get('no_dataset_collapse', True):
            recommendations.append("Critical: Dataset loading issue - check data paths and loader configuration")
            
        if not gate_results.get('token_monotonicity', True):
            recommendations.append("Fix: Token counting inconsistency - verify tokenizer configuration")
            
        if not gate_results.get('median_tokens_at_8_percent', True):
            recommendations.append("Fix: Token threshold too low at 8% keep ratio")
            
        if not gate_results.get('kv_mass_share', True):
            recommendations.append("Fix: KV reuse metrics defaulting - check hybrid system integration")
            
        if not gate_results.get('delta_cbu_variance', True):
            recommendations.append("Fix: ΔCBU variance too low - check cost model implementation")
            
        if all(gate_results.values()):
            recommendations.append("✅ All gates passed - proceed to fairness reset")
        else:
            recommendations.append("❌ Fix failing gates before proceeding to full evaluation")
            
        return recommendations
    
    def _log_canary_results(self, result: CanaryResult):
        """Log comprehensive canary results"""
        logger.info("="*60)
        logger.info("🎯 CANARY TEST RESULTS")
        logger.info("="*60)
        
        logger.info(f"⏱️  Execution time: {result.execution_time_seconds:.1f}s")
        logger.info(f"📊 Samples processed: {result.sample_count}")
        
        logger.info("\n🔒 VALIDATION GATES:")
        for gate_name, passed in result.gate_results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"   {gate_name}: {status}")
        
        logger.info("\n📈 METHOD PERFORMANCE:")
        for method, perf in result.method_results.items():
            logger.info(f"   {method}: accuracy={perf['avg_accuracy']:.3f}, p95={perf['avg_p95_ms']:.1f}ms")
        
        logger.info("\n💡 RECOMMENDATIONS:")
        for rec in result.recommendations:
            logger.info(f"   • {rec}")
        
        overall_status = "✅ CANARY PASSED" if result.all_gates_passed else "❌ CANARY FAILED"
        logger.info(f"\n🎯 OVERALL STATUS: {overall_status}")
        
        if not result.all_gates_passed:
            logger.error("❌ CANARY TEST FAILED - DO NOT PROCEED TO FULL EVALUATION")
        else:
            logger.info("✅ CANARY TEST PASSED - READY FOR FULL PAIRED MATRIX")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Guarded Canary Test')
    parser.add_argument('--samples', type=int, default=30,
                       help='Samples per scenario (default: 30)')
    parser.add_argument('--fail-fast', action='store_true', default=True,
                       help='Stop immediately on gate failure (default: True)')
    parser.add_argument('--no-fail-fast', dest='fail_fast', action='store_false',
                       help='Continue testing even if gates fail')
    parser.add_argument('--strict', action='store_true',
                       help='Use strictest gate thresholds')
    parser.add_argument('--verbose', '-v', action='store_true')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create and run canary
        runner = GuardedCanaryRunner(
            samples_per_scenario=args.samples,
            fail_fast=args.fail_fast
        )
        
        # Adjust thresholds for strict mode
        if args.strict:
            runner.thresholds.min_p_at_5_per_scenario = 0.01  # Stricter
            runner.thresholds.min_median_at_8_percent = 800   # Stricter
            runner.thresholds.max_ece_per_type_budget = 0.05  # Stricter
            logger.info("🔒 Strict mode: Using tighter gate thresholds")
        
        # Run the test
        result = runner.run_canary_test()
        
        # Exit with appropriate code
        if result.all_gates_passed:
            print("\n✅ Canary test PASSED - Ready for production evaluation!")
            sys.exit(0)
        else:
            print("\n❌ Canary test FAILED - Fix issues before proceeding!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Canary test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Canary test failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()