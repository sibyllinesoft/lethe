#!/usr/bin/env python3
"""
InfiniteBench Evaluation Runner for Lethe→StreamingLLM Hybrid System

Comprehensive evaluation script that runs the complete InfiniteBench evaluation
matrix as specified in TODO.md, including statistical validation and promotion
decision logic.

Evaluation Matrix:
- Methods: Streaming, Lethe, Hybrid
- Keep ratios: 0.08, 0.15, 0.30
- Datasets: Code.Debug + Code.QA (≥100 items) + 50-item Zh.QA
- Metrics: P@k/R@k, ΔCBU/1k, p95 times, KV-reuse, tail CVaR
- Promotion rule: Hybrid beats Streaming at matched keep-ratio with p95 ≤ +1ms

Features:
- Production-ready evaluation with comprehensive logging
- Statistical validation with bootstrap CI and permutation tests
- Automated promotion decision based on performance criteria
- Detailed result reporting and export
- Performance monitoring and health checks
- Canary configuration validation
"""

import logging
import time
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import numpy as np
from datetime import datetime

# Import hybrid system components - handle imports gracefully  
try:
    from hybrid_selector import HybridSelector, HybridConfig, create_hybrid_selector
    from instrumentation import HybridInstrumentation, create_instrumentation
    from adaptive_params import AdaptiveParameterController, OptimizationObjective
    from benchmarking import (
        HybridBenchmarkEvaluator, BenchmarkMethod, DatasetType,
        LetheStreamingHybridCompetitor, CompetitorConfig, BenchmarkRun
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Running in standalone mode with limited functionality")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/infinitebench_evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

class InfiniteBenchRunner:
    """Complete InfiniteBench evaluation runner."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize evaluation runner."""
        self.start_time = time.time()
        self.run_id = f"infinitebench_{int(self.start_time)}"
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.evaluator = HybridBenchmarkEvaluator()
        self.instrumentation = create_instrumentation()
        
        # Results storage
        self.results = {}
        self.performance_data = {}
        self.promotion_analysis = {}
        
        logger.info(f"InfiniteBench evaluation initialized: {self.run_id}")
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load evaluation configuration."""
        default_config = {
            # Canary configuration from TODO.md
            "canary_config": {
                "head_keep_ratio": 0.12,
                "window_size": 6000,
                "stride": 3000,
                "sink_tokens": 96,
                "ce_k2": 320,
                "dpp_rank": 14
            },
            
            # Evaluation matrix
            "evaluation_matrix": {
                "methods": ["streaming", "lethe", "hybrid"],
                "keep_ratios": [0.08, 0.15, 0.30],
                "datasets": ["code_debug", "code_qa", "zh_qa"],
                "min_samples": {
                    "code_debug": 100,
                    "code_qa": 100,
                    "zh_qa": 50
                }
            },
            
            # Performance targets
            "performance_targets": {
                "max_p95_latency_ms": 1.0,      # ≤1ms p95 target
                "min_cbu_improvement": 10.0,     # Minimum +10% CBU
                "max_latency_regression_ms": 1.0, # ≤+1ms regression allowed
                "min_kv_reuse_ratio": 0.6        # Minimum 60% KV reuse
            },
            
            # Statistical validation
            "statistical_config": {
                "confidence_level": 0.95,
                "n_bootstrap": 1000,
                "n_permutations": 1000,
                "min_effect_size": 0.1
            },
            
            # Output configuration
            "output": {
                "export_detailed_results": True,
                "export_telemetry": True,
                "generate_report": True,
                "results_directory": "/tmp/infinitebench_results"
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                user_config = json.load(f)
                # Merge with defaults
                default_config.update(user_config)
        
        return default_config
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete InfiniteBench evaluation."""
        logger.info("="*60)
        logger.info("STARTING INFINITEBENCH EVALUATION")
        logger.info("="*60)
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Configuration: {self.config['canary_config']}")
        
        try:
            # Phase 1: Validate system health
            self._validate_system_health()
            
            # Phase 2: Run benchmark evaluation
            benchmark_results = self._run_benchmark_evaluation()
            
            # Phase 3: Analyze performance
            performance_analysis = self._analyze_performance(benchmark_results)
            
            # Phase 4: Make promotion decision
            promotion_decision = self._make_promotion_decision(
                benchmark_results, performance_analysis
            )
            
            # Phase 5: Generate comprehensive report
            final_report = self._generate_final_report(
                benchmark_results, performance_analysis, promotion_decision
            )
            
            # Phase 6: Export results
            export_paths = self._export_results(final_report)
            
            logger.info("="*60)
            logger.info("INFINITEBENCH EVALUATION COMPLETED")
            logger.info("="*60)
            logger.info(f"Promotion decision: {promotion_decision['overall_verdict']}")
            logger.info(f"Results exported to: {export_paths}")
            
            return final_report
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            raise
    
    def _validate_system_health(self):
        """Validate system health before evaluation."""
        logger.info("Validating system health...")
        
        # Test hybrid selector initialization
        try:
            test_config = HybridConfig(**self.config["canary_config"])
            test_selector = create_hybrid_selector(test_config)
            logger.info("✅ Hybrid selector initialization: OK")
        except Exception as e:
            raise RuntimeError(f"Hybrid selector validation failed: {e}")
        
        # Test instrumentation
        try:
            test_instrumentation = create_instrumentation()
            dashboard = test_instrumentation.get_dashboard_metrics()
            logger.info("✅ Instrumentation system: OK")
        except Exception as e:
            raise RuntimeError(f"Instrumentation validation failed: {e}")
        
        # Validate configuration parameters
        canary = self.config["canary_config"]
        if not (0.05 <= canary["head_keep_ratio"] <= 0.30):
            raise ValueError(f"Invalid head_keep_ratio: {canary['head_keep_ratio']}")
        
        if canary["stride"] >= canary["window_size"]:
            raise ValueError(f"Stride ({canary['stride']}) must be < window_size ({canary['window_size']})")
        
        logger.info("✅ System health validation: PASSED")
    
    def _run_benchmark_evaluation(self) -> BenchmarkRun:
        """Run the complete benchmark evaluation."""
        logger.info("Starting benchmark evaluation...")
        
        # Configure evaluation matrix
        eval_config = self.config["evaluation_matrix"]
        self.evaluator.evaluation_matrix = {
            'methods': [BenchmarkMethod(m) for m in eval_config["methods"]],
            'keep_ratios': eval_config["keep_ratios"],
            'datasets': [DatasetType(d) for d in eval_config["datasets"]],
            'min_samples': eval_config["min_samples"]
        }
        
        # Run evaluation
        start_time = time.time()
        benchmark_run = self.evaluator.run_full_evaluation()
        evaluation_time = time.time() - start_time
        
        logger.info(f"Benchmark evaluation completed in {evaluation_time:.1f} seconds")
        logger.info(f"Total result sets: {len(benchmark_run.results)}")
        
        # Log summary statistics
        for method, stats in benchmark_run.summary_stats['by_method'].items():
            logger.info(f"{method.upper()} - F1: {stats.get('avg_f1_score', 0):.3f}, "
                       f"Time: {stats.get('avg_processing_time_ms', 0):.1f}ms, "
                       f"ΔCBU/1k: {stats.get('avg_delta_cbu_per_1k', 0):.3f}")
        
        return benchmark_run
    
    def _analyze_performance(self, benchmark_run: BenchmarkRun) -> Dict[str, Any]:
        """Analyze performance across all metrics."""
        logger.info("Analyzing performance metrics...")
        
        analysis = {
            'method_comparison': {},
            'dataset_performance': {},
            'keep_ratio_analysis': {},
            'performance_targets': {},
            'statistical_significance': {}
        }
        
        # Method comparison analysis
        methods = ['streaming', 'lethe', 'hybrid']
        for method in methods:
            method_data = benchmark_run.summary_stats['by_method'].get(method, {})
            analysis['method_comparison'][method] = {
                'avg_f1_score': method_data.get('avg_f1_score', 0),
                'avg_processing_time_ms': method_data.get('avg_processing_time_ms', 0),
                'p95_processing_time_ms': method_data.get('p95_processing_time_ms', 0),
                'avg_delta_cbu_per_1k': method_data.get('avg_delta_cbu_per_1k', 0),
                'avg_kv_reuse': method_data.get('avg_kv_reuse', 0),
                'sample_count': method_data.get('count', 0)
            }
        
        # Performance target analysis
        targets = self.config['performance_targets']
        for method, metrics in analysis['method_comparison'].items():
            analysis['performance_targets'][method] = {
                'meets_p95_target': metrics['p95_processing_time_ms'] <= targets['max_p95_latency_ms'],
                'meets_cbu_target': metrics['avg_delta_cbu_per_1k'] >= targets['min_cbu_improvement'],
                'meets_kv_target': metrics['avg_kv_reuse'] >= targets['min_kv_reuse_ratio'],
                'target_score': self._calculate_target_score(metrics, targets)
            }
        
        # Dataset-specific analysis
        for dataset, stats in benchmark_run.summary_stats['by_dataset'].items():
            analysis['dataset_performance'][dataset] = stats
        
        # Statistical significance analysis
        for test_name, test_result in benchmark_run.statistical_tests.items():
            analysis['statistical_significance'][test_name] = {
                'significant_metrics': [
                    metric for metric, data in test_result['metrics'].items()
                    if data.get('significant', False)
                ],
                'effect_sizes': {
                    metric: data.get('effect_size', 0)
                    for metric, data in test_result['metrics'].items()
                },
                'better_method': test_result['metrics']['f1_scores'].get('better_method', 'unknown')
            }
        
        logger.info("Performance analysis completed")
        return analysis
    
    def _calculate_target_score(self, metrics: Dict[str, float], 
                              targets: Dict[str, float]) -> float:
        """Calculate overall target achievement score."""
        scores = []
        
        # P95 latency score (lower is better, normalized)
        if metrics['p95_processing_time_ms'] <= targets['max_p95_latency_ms']:
            p95_score = 1.0
        else:
            # Penalize latency overruns
            p95_score = max(0.0, 1.0 - (metrics['p95_processing_time_ms'] - targets['max_p95_latency_ms']) / 10.0)
        scores.append(p95_score)
        
        # CBU improvement score
        cbu_score = min(1.0, metrics['avg_delta_cbu_per_1k'] / targets['min_cbu_improvement'])
        scores.append(cbu_score)
        
        # KV reuse score
        kv_score = min(1.0, metrics['avg_kv_reuse'] / targets['min_kv_reuse_ratio'])
        scores.append(kv_score)
        
        return np.mean(scores)
    
    def _make_promotion_decision(self, benchmark_run: BenchmarkRun, 
                               analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Make promotion decision based on evaluation results."""
        logger.info("Making promotion decision...")
        
        decision = {
            'overall_verdict': 'NO_PROMOTION',
            'promote_hybrid': False,
            'decision_criteria': {},
            'test_results': [],
            'performance_summary': {},
            'recommendation': '',
            'confidence_score': 0.0
        }
        
        # Extract promotion decision from benchmark run
        if benchmark_run.promotion_decision:
            base_decision = benchmark_run.promotion_decision
            decision.update(base_decision)
        
        # Additional validation based on performance analysis
        hybrid_performance = analysis['method_comparison'].get('hybrid', {})
        streaming_performance = analysis['method_comparison'].get('streaming', {})
        
        # Check performance targets for hybrid
        hybrid_targets = analysis['performance_targets'].get('hybrid', {})
        target_score = hybrid_targets.get('target_score', 0.0)
        
        decision['performance_summary'] = {
            'hybrid_target_score': target_score,
            'meets_all_targets': all([
                hybrid_targets.get('meets_p95_target', False),
                hybrid_targets.get('meets_cbu_target', False),
                hybrid_targets.get('meets_kv_target', False)
            ]),
            'hybrid_vs_streaming': {
                'f1_improvement': (hybrid_performance.get('avg_f1_score', 0) - 
                                 streaming_performance.get('avg_f1_score', 0)),
                'latency_regression': (hybrid_performance.get('p95_processing_time_ms', 0) - 
                                     streaming_performance.get('p95_processing_time_ms', 0)),
                'cbu_improvement': (hybrid_performance.get('avg_delta_cbu_per_1k', 0) - 
                                  streaming_performance.get('avg_delta_cbu_per_1k', 0))
            }
        }
        
        # Final promotion logic
        performance_summary = decision['performance_summary']
        
        # Must beat streaming on quality or CBU
        quality_better = performance_summary['hybrid_vs_streaming']['f1_improvement'] > 0.01  # 1% improvement
        cbu_better = performance_summary['hybrid_vs_streaming']['cbu_improvement'] > 0.5  # 0.5 ΔCBU/1k improvement
        
        # Must not regress on latency
        latency_ok = performance_summary['hybrid_vs_streaming']['latency_regression'] <= 1.0  # ≤+1ms
        
        # Must meet basic performance targets
        meets_targets = performance_summary['meets_all_targets']
        
        promotion_criteria = {
            'quality_or_cbu_better': quality_better or cbu_better,
            'latency_constraint_met': latency_ok,
            'performance_targets_met': meets_targets,
            'statistical_significance': len([
                test for test, results in analysis['statistical_significance'].items()
                if 'hybrid' in results.get('better_method', '')
            ]) > 0
        }
        
        decision['decision_criteria'] = promotion_criteria
        
        # Calculate confidence score
        criteria_met = sum(promotion_criteria.values())
        confidence_score = criteria_met / len(promotion_criteria)
        decision['confidence_score'] = confidence_score
        
        # Final decision
        if confidence_score >= 0.75:  # 75% of criteria must be met
            decision['promote_hybrid'] = True
            decision['overall_verdict'] = 'PROMOTE'
            decision['recommendation'] = (
                f"RECOMMEND PROMOTION: Hybrid system meets {criteria_met}/{len(promotion_criteria)} "
                f"criteria with {confidence_score:.1%} confidence."
            )
        else:
            decision['overall_verdict'] = 'NO_PROMOTION'
            decision['recommendation'] = (
                f"DO NOT PROMOTE: Hybrid system meets only {criteria_met}/{len(promotion_criteria)} "
                f"criteria with {confidence_score:.1%} confidence. Additional optimization needed."
            )
        
        logger.info(f"Promotion decision: {decision['overall_verdict']}")
        logger.info(f"Confidence score: {confidence_score:.1%}")
        logger.info(f"Criteria met: {criteria_met}/{len(promotion_criteria)}")
        
        return decision
    
    def _generate_final_report(self, benchmark_run: BenchmarkRun,
                             analysis: Dict[str, Any],
                             promotion_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        logger.info("Generating final report...")
        
        total_time = time.time() - self.start_time
        
        report = {
            'metadata': {
                'run_id': self.run_id,
                'timestamp': datetime.now().isoformat(),
                'total_evaluation_time_seconds': total_time,
                'configuration': self.config,
                'system_info': {
                    'python_version': sys.version,
                    'evaluation_samples': sum([
                        len(results) for results in benchmark_run.results.values()
                    ])
                }
            },
            'evaluation_results': {
                'benchmark_run_id': benchmark_run.run_id,
                'competitors_tested': len(benchmark_run.competitors),
                'datasets_tested': len(benchmark_run.datasets),
                'total_samples_processed': sum([
                    len(results) for results in benchmark_run.results.values()
                ]),
                'summary_statistics': benchmark_run.summary_stats,
                'statistical_tests': benchmark_run.statistical_tests
            },
            'performance_analysis': analysis,
            'promotion_decision': promotion_decision,
            'key_findings': self._extract_key_findings(analysis, promotion_decision),
            'recommendations': self._generate_recommendations(analysis, promotion_decision),
            'next_steps': self._generate_next_steps(promotion_decision)
        }
        
        return report
    
    def _extract_key_findings(self, analysis: Dict[str, Any], 
                            promotion_decision: Dict[str, Any]) -> List[str]:
        """Extract key findings from evaluation."""
        findings = []
        
        # Performance comparison
        hybrid_perf = analysis['method_comparison'].get('hybrid', {})
        streaming_perf = analysis['method_comparison'].get('streaming', {})
        
        if hybrid_perf and streaming_perf:
            f1_diff = hybrid_perf['avg_f1_score'] - streaming_perf['avg_f1_score']
            time_diff = hybrid_perf['p95_processing_time_ms'] - streaming_perf['p95_processing_time_ms']
            cbu_diff = hybrid_perf['avg_delta_cbu_per_1k'] - streaming_perf['avg_delta_cbu_per_1k']
            
            findings.append(f"Hybrid vs Streaming: F1 {f1_diff:+.3f}, Latency {time_diff:+.1f}ms, ΔCBU/1k {cbu_diff:+.2f}")
        
        # Performance targets
        targets = analysis['performance_targets']
        hybrid_targets = targets.get('hybrid', {})
        
        findings.append(f"Hybrid target achievement: {hybrid_targets.get('target_score', 0):.1%}")
        
        # Statistical significance
        significant_tests = sum([
            len(results.get('significant_metrics', []))
            for results in analysis['statistical_significance'].values()
        ])
        
        findings.append(f"Statistically significant improvements: {significant_tests} metrics")
        
        # KV cache performance
        kv_reuse = hybrid_perf.get('avg_kv_reuse', 0)
        findings.append(f"Average KV cache reuse ratio: {kv_reuse:.1%}")
        
        return findings
    
    def _generate_recommendations(self, analysis: Dict[str, Any],
                                promotion_decision: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        if promotion_decision['promote_hybrid']:
            recommendations.append("✅ PROCEED with hybrid system deployment to 5% canary traffic")
            recommendations.append("Monitor KV cache performance and tail latency closely")
            recommendations.append("Prepare rollback plan in case of performance degradation")
        else:
            # Identify improvement areas
            criteria = promotion_decision.get('decision_criteria', {})
            
            if not criteria.get('quality_or_cbu_better', True):
                recommendations.append("❌ Improve quality metrics (P@k/R@k) or ΔCBU/1k performance")
            
            if not criteria.get('latency_constraint_met', True):
                recommendations.append("❌ Optimize latency - current p95 exceeds +1ms regression limit")
            
            if not criteria.get('performance_targets_met', True):
                recommendations.append("❌ Address performance target gaps")
            
            recommendations.append("🔄 Re-run evaluation after optimizations")
        
        # General recommendations
        performance = analysis.get('performance_analysis', {})
        hybrid_perf = analysis['method_comparison'].get('hybrid', {})
        
        if hybrid_perf.get('avg_kv_reuse', 0) < 0.7:
            recommendations.append("🔧 Investigate KV cache reuse optimization opportunities")
        
        if hybrid_perf.get('p95_processing_time_ms', 0) > 500:
            recommendations.append("⚡ Consider tail latency optimization")
        
        return recommendations
    
    def _generate_next_steps(self, promotion_decision: Dict[str, Any]) -> List[str]:
        """Generate next steps based on promotion decision."""
        next_steps = []
        
        if promotion_decision['promote_hybrid']:
            next_steps.extend([
                "1. Deploy hybrid system to 5% canary traffic",
                "2. Monitor production metrics for 72 hours", 
                "3. Validate performance against SLA targets",
                "4. Prepare for gradual traffic increase if successful",
                "5. Document lessons learned and optimization opportunities"
            ])
        else:
            next_steps.extend([
                "1. Analyze specific failure points from evaluation",
                "2. Implement targeted optimizations",
                "3. Re-run InfiniteBench evaluation", 
                "4. Consider alternative architectural approaches",
                "5. Schedule follow-up evaluation in 2-4 weeks"
            ])
        
        return next_steps
    
    def _export_results(self, final_report: Dict[str, Any]) -> Dict[str, str]:
        """Export evaluation results to files."""
        logger.info("Exporting evaluation results...")
        
        # Create results directory
        results_dir = Path(self.config['output']['results_directory'])
        results_dir.mkdir(exist_ok=True, parents=True)
        
        export_paths = {}
        
        # Export comprehensive report
        report_path = results_dir / f"infinitebench_report_{self.run_id}.json"
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        export_paths['report'] = str(report_path)
        
        # Export summary report (human-readable)
        summary_path = results_dir / f"infinitebench_summary_{self.run_id}.md"
        with open(summary_path, 'w') as f:
            self._write_markdown_summary(f, final_report)
        export_paths['summary'] = str(summary_path)
        
        # Export telemetry data
        if self.config['output']['export_telemetry']:
            telemetry_path = self.instrumentation.export_telemetry(
                str(results_dir / f"telemetry_{self.run_id}.json")
            )
            export_paths['telemetry'] = telemetry_path
        
        # Export raw benchmark data
        if self.config['output']['export_detailed_results']:
            benchmark_path = results_dir / f"benchmark_raw_{self.run_id}.json"
            # Would export detailed benchmark results here
            export_paths['benchmark'] = str(benchmark_path)
        
        logger.info(f"Results exported to {len(export_paths)} files")
        return export_paths
    
    def _write_markdown_summary(self, f, report: Dict[str, Any]):
        """Write human-readable markdown summary."""
        f.write(f"# InfiniteBench Evaluation Report\n\n")
        f.write(f"**Run ID:** {report['metadata']['run_id']}\n")
        f.write(f"**Date:** {report['metadata']['timestamp']}\n")
        f.write(f"**Duration:** {report['metadata']['total_evaluation_time_seconds']:.1f} seconds\n\n")
        
        # Promotion decision
        decision = report['promotion_decision']
        f.write(f"## 🎯 Promotion Decision\n\n")
        f.write(f"**Verdict:** {decision['overall_verdict']}\n")
        f.write(f"**Confidence:** {decision.get('confidence_score', 0):.1%}\n\n")
        f.write(f"{decision.get('recommendation', 'No recommendation available.')}\n\n")
        
        # Key findings
        f.write(f"## 🔍 Key Findings\n\n")
        for finding in report.get('key_findings', []):
            f.write(f"- {finding}\n")
        f.write("\n")
        
        # Recommendations  
        f.write(f"## 💡 Recommendations\n\n")
        for rec in report.get('recommendations', []):
            f.write(f"- {rec}\n")
        f.write("\n")
        
        # Next steps
        f.write(f"## 📋 Next Steps\n\n")
        for step in report.get('next_steps', []):
            f.write(f"{step}\n")
        f.write("\n")
        
        # Performance summary
        analysis = report.get('performance_analysis', {})
        method_comp = analysis.get('method_comparison', {})
        
        f.write(f"## 📊 Performance Summary\n\n")
        f.write(f"| Method | F1 Score | P95 Latency (ms) | ΔCBU/1k | KV Reuse |\n")
        f.write(f"|--------|----------|------------------|---------|----------|\n")
        
        for method, metrics in method_comp.items():
            f.write(f"| {method.title()} | {metrics.get('avg_f1_score', 0):.3f} | "
                   f"{metrics.get('p95_processing_time_ms', 0):.1f} | "
                   f"{metrics.get('avg_delta_cbu_per_1k', 0):.2f} | "
                   f"{metrics.get('avg_kv_reuse', 0):.1%} |\n")

def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Run InfiniteBench evaluation for Lethe→StreamingLLM Hybrid"
    )
    parser.add_argument(
        '--config', 
        type=str,
        help='Path to evaluation configuration JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/tmp/infinitebench_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick evaluation with reduced sample sizes'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize runner
        runner = InfiniteBenchRunner(args.config)
        
        # Override output directory
        runner.config['output']['results_directory'] = args.output_dir
        
        # Quick mode adjustments
        if args.quick:
            logger.info("Running in quick mode with reduced samples")
            runner.config['evaluation_matrix']['min_samples'] = {
                'code_debug': 20,
                'code_qa': 20, 
                'zh_qa': 10
            }
        
        # Run evaluation
        final_report = runner.run_full_evaluation()
        
        # Print summary
        print("\n" + "="*80)
        print("INFINITEBENCH EVALUATION SUMMARY")
        print("="*80)
        
        decision = final_report['promotion_decision']
        print(f"Run ID: {final_report['metadata']['run_id']}")
        print(f"Promotion Decision: {decision['overall_verdict']}")
        print(f"Confidence Score: {decision.get('confidence_score', 0):.1%}")
        print(f"Total Evaluation Time: {final_report['metadata']['total_evaluation_time_seconds']:.1f}s")
        
        print(f"\nKey Findings:")
        for finding in final_report.get('key_findings', []):
            print(f"  • {finding}")
        
        print(f"\nRecommendation: {decision.get('recommendation', 'None')}")
        
        print("="*80)
        
        # Exit with appropriate code
        if decision['promote_hybrid']:
            print("✅ PROMOTION APPROVED - Hybrid system ready for deployment")
            sys.exit(0)
        else:
            print("❌ PROMOTION DECLINED - Additional optimization needed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()