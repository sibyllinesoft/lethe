#!/usr/bin/env python3
"""
Integration Tests and Validation for Lagrangian Latency Optimization System

Comprehensive test suite demonstrating the complete optimization system
achieving P95 ≤1ms latency target while maintaining CBU ≥10% quality.

Test Scenarios:
1. Baseline performance validation (current +6.8ms P95)
2. Optimization effectiveness demonstration
3. Quality preservation under aggressive optimization
4. Promotion criteria validation
5. System stability and monitoring tests
6. Real-world query workload simulation

Expected Outcomes:
- P95 latency reduction from 6.8ms → ≤1ms
- CBU improvement maintained ≥10% (current 12.5% baseline)
- Quality preservation ≥85% across all optimizations
- Promotion criteria met: ΔCBU/GB ≥ +10% OR P95 improvement ≥5ms
"""

import logging
import time
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our optimization system
from .integrated_latency_optimizer import (
    IntegratedLatencyOptimizer, IntegratedConfig, OptimizationMode
)
from .lagrangian_optimizer import LagrangianConfig, QueryComplexity
from .ce_early_exit import EarlyExitConfig, EarlyExitStrategy
from .performance_monitor import PerformanceConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestWorkload:
    """Test workload definition."""
    name: str
    queries: List[str]
    session_contexts: List[Dict[str, Any]]
    doc_candidates: List[List[Dict[str, Any]]]
    expected_complexity: List[QueryComplexity]
    performance_targets: Dict[str, float]

class LatencyOptimizationValidator:
    """
    Comprehensive validation system for latency optimization.
    
    Validates system performance against critical targets:
    - P95 latency ≤1ms (from current 6.8ms baseline)
    - CBU improvement ≥10% (maintain 12.5% current baseline)
    - Quality preservation ≥85%
    - System stability and monitoring effectiveness
    """
    
    def __init__(self):
        """Initialize validation system."""
        self.test_results: List[Dict[str, Any]] = []
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Create test configurations
        self.baseline_config = self._create_baseline_config()
        self.optimized_config = self._create_optimized_config()
        
        # Initialize optimizers
        self.baseline_optimizer = IntegratedLatencyOptimizer(self.baseline_config)
        self.optimized_optimizer = IntegratedLatencyOptimizer(self.optimized_config)
        
        logger.info("LatencyOptimizationValidator initialized")
    
    def _create_baseline_config(self) -> IntegratedConfig:
        """Create baseline configuration representing current system."""
        return IntegratedConfig(
            target_p95_latency_ms=6.8,  # Current baseline performance
            target_cbu_improvement=12.5,  # Current excellent CBU
            current_cbu_baseline=12.5,
            optimization_mode=OptimizationMode.QUALITY_FIRST,  # Conservative baseline
            auto_mode_switching=False,
            lagrangian_config=LagrangianConfig(
                target_latency_p95_ms=6.8,
                current_cbu_improvement=12.5,
                standard_embedding_dim=768,  # Full embeddings
                hard_embedding_dim=768,
                k1_range=(3000, 3000),  # Fixed high K1
                k2_range=(600, 600)     # Fixed high K2
            ),
            early_exit_config=EarlyExitConfig(
                strategy=EarlyExitStrategy.DISABLED,  # No early exit
                gain_per_token_base_threshold=0.0001,  # Very conservative
                posterior_confidence_min=0.95         # High confidence
            )
        )
    
    def _create_optimized_config(self) -> IntegratedConfig:
        """Create optimized configuration targeting ≤1ms P95."""
        return IntegratedConfig(
            target_p95_latency_ms=1.0,    # Aggressive target
            target_cbu_improvement=10.0,  # Minimum for promotion
            current_cbu_baseline=12.5,    # Maintain current quality
            optimization_mode=OptimizationMode.ADAPTIVE,  # Adaptive optimization
            auto_mode_switching=True,
            lagrangian_config=LagrangianConfig(
                target_latency_p95_ms=1.0,
                current_cbu_improvement=12.5,
                target_cbu_threshold=10.0,
                standard_embedding_dim=256,  # Matryoshka optimization
                hard_embedding_dim=768,      # Full dim only for hard queries
                k1_range=(1000, 3000),      # Dynamic K1
                k2_range=(200, 600),        # Dynamic K2
                dpp_rank_range=(12, 16),    # Reduced DPP rank
                group_split_tau=0.7         # Optimized group split
            ),
            early_exit_config=EarlyExitConfig(
                strategy=EarlyExitStrategy.CALIBRATED,  # Advanced early exit
                calibrated_prefix_min=150,
                calibrated_prefix_max=200,
                gain_per_token_base_threshold=0.001,
                lambda_multiplier_coupling=1.0,
                posterior_confidence_min=0.8
            )
        )
    
    def create_test_workloads(self) -> List[TestWorkload]:
        """Create comprehensive test workloads."""
        
        workloads = []
        
        # 1. Standard Query Workload (should use 256d Matryoshka)
        standard_queries = [
            "user authentication error",
            "database connection timeout", 
            "API rate limit exceeded",
            "file upload processing",
            "search results ranking"
        ]
        
        standard_contexts = [
            {"session_id": f"std_{i}", "turn_count": 1, "user_type": "standard"}
            for i in range(len(standard_queries))
        ]
        
        standard_candidates = [
            self._generate_mock_candidates(query, 20, "standard")
            for query in standard_queries
        ]
        
        workloads.append(TestWorkload(
            name="standard_queries",
            queries=standard_queries,
            session_contexts=standard_contexts,
            doc_candidates=standard_candidates,
            expected_complexity=[QueryComplexity.STANDARD] * len(standard_queries),
            performance_targets={"p95_latency": 0.8, "cbu_improvement": 11.0}
        ))
        
        # 2. Hard Query Workload (should use 768d embeddings)
        hard_queries = [
            "distributed system consensus algorithm byzantine fault tolerance implementation",
            "microservice orchestration kubernetes deployment yaml configuration troubleshooting advanced",
            "machine learning model hyperparameter optimization bayesian search cross-validation metrics",
            "cryptographic hash function collision resistance proof verification mathematical analysis",
            "real-time streaming data processing apache kafka partition rebalancing consumer lag"
        ]
        
        hard_contexts = [
            {
                "session_id": f"hard_{i}", 
                "turn_count": i+1, 
                "user_type": "expert",
                "previous_queries": hard_queries[:i] if i > 0 else []
            }
            for i in range(len(hard_queries))
        ]
        
        hard_candidates = [
            self._generate_mock_candidates(query, 35, "hard")
            for query in hard_queries
        ]
        
        workloads.append(TestWorkload(
            name="hard_queries",
            queries=hard_queries,
            session_contexts=hard_contexts,
            doc_candidates=hard_candidates,
            expected_complexity=[QueryComplexity.HARD] * len(hard_queries),
            performance_targets={"p95_latency": 1.2, "cbu_improvement": 13.0}
        ))
        
        # 3. Mixed Workload (realistic distribution)
        mixed_queries = standard_queries[:3] + hard_queries[:2] + [
            "simple error message",
            "complex distributed transaction rollback mechanism failure analysis"
        ]
        
        mixed_contexts = standard_contexts[:3] + hard_contexts[:2] + [
            {"session_id": "mix_6", "turn_count": 1, "user_type": "novice"},
            {"session_id": "mix_7", "turn_count": 4, "user_type": "expert"}
        ]
        
        mixed_candidates = standard_candidates[:3] + hard_candidates[:2] + [
            self._generate_mock_candidates("simple error", 15, "standard"),
            self._generate_mock_candidates("complex transaction", 30, "hard")
        ]
        
        mixed_expected_complexity = (
            [QueryComplexity.STANDARD] * 3 + 
            [QueryComplexity.HARD] * 2 + 
            [QueryComplexity.STANDARD, QueryComplexity.HARD]
        )
        
        workloads.append(TestWorkload(
            name="mixed_workload",
            queries=mixed_queries,
            session_contexts=mixed_contexts,
            doc_candidates=mixed_candidates,
            expected_complexity=mixed_expected_complexity,
            performance_targets={"p95_latency": 1.0, "cbu_improvement": 10.5}
        ))
        
        return workloads
    
    def _generate_mock_candidates(
        self, query: str, num_candidates: int, complexity_type: str
    ) -> List[Dict[str, Any]]:
        """Generate mock document candidates for testing."""
        
        candidates = []
        
        # Base score distribution depends on complexity
        if complexity_type == "standard":
            base_scores = np.random.exponential(0.3, num_candidates)  # More concentrated scores
        else:
            base_scores = np.random.gamma(2, 0.2, num_candidates)     # More distributed scores
        
        # Normalize scores
        base_scores = np.clip(base_scores, 0, 1)
        base_scores = np.sort(base_scores)[::-1]  # Sort descending
        
        for i in range(num_candidates):
            # Generate text based on complexity
            if complexity_type == "standard":
                text_length = np.random.randint(100, 500)  # Shorter texts
                text = f"Standard documentation for {query}. " * (text_length // 50)
            else:
                text_length = np.random.randint(500, 2000)  # Longer, complex texts
                text = (f"Advanced technical documentation covering {query} "
                       f"with detailed implementation examples and analysis. ") * (text_length // 100)
            
            candidates.append({
                'id': f'doc_{i}_{complexity_type}',
                'score': float(base_scores[i]),
                'text': text[:text_length],
                'type': complexity_type,
                'relevance': base_scores[i]
            })
        
        return candidates
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of the optimization system."""
        
        logger.info("Starting comprehensive latency optimization validation")
        
        validation_results = {
            'validation_start': datetime.now().isoformat(),
            'test_results': {},
            'performance_comparison': {},
            'promotion_analysis': {},
            'system_stability': {},
            'recommendations': []
        }
        
        try:
            # Create test workloads
            workloads = self.create_test_workloads()
            
            # Run baseline vs optimized comparisons
            for workload in workloads:
                logger.info(f"Testing workload: {workload.name}")
                
                baseline_results = await self._run_workload_test(
                    workload, self.baseline_optimizer, "baseline"
                )
                
                optimized_results = await self._run_workload_test(
                    workload, self.optimized_optimizer, "optimized"
                )
                
                workload_comparison = self._compare_workload_results(
                    baseline_results, optimized_results, workload.performance_targets
                )
                
                validation_results['test_results'][workload.name] = {
                    'baseline': baseline_results,
                    'optimized': optimized_results,
                    'comparison': workload_comparison
                }
            
            # Aggregate performance analysis
            validation_results['performance_comparison'] = self._analyze_aggregate_performance(
                validation_results['test_results']
            )
            
            # Promotion criteria analysis
            validation_results['promotion_analysis'] = self._analyze_promotion_criteria(
                validation_results['performance_comparison']
            )
            
            # System stability analysis
            validation_results['system_stability'] = self._analyze_system_stability()
            
            # Generate final recommendations
            validation_results['recommendations'] = self._generate_final_recommendations(
                validation_results
            )
            
            # Success assessment
            validation_results['validation_success'] = self._assess_validation_success(
                validation_results
            )
            
            logger.info("Comprehensive validation completed successfully")
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_results['error'] = str(e)
            validation_results['validation_success'] = False
        
        validation_results['validation_end'] = datetime.now().isoformat()
        return validation_results
    
    async def _run_workload_test(
        self,
        workload: TestWorkload,
        optimizer: IntegratedLatencyOptimizer,
        test_type: str
    ) -> Dict[str, Any]:
        """Run optimization test on a workload."""
        
        results = {
            'workload_name': workload.name,
            'test_type': test_type,
            'optimization_results': [],
            'performance_metrics': {},
            'query_classification': {}
        }
        
        # Track performance metrics
        latencies = []
        cbu_improvements = []
        quality_preservations = []
        computational_savings = []
        
        # Track query classification accuracy
        classification_correct = 0
        
        # Process each query in the workload
        for i, (query, context, candidates, expected_complexity) in enumerate(
            zip(workload.queries, workload.session_contexts, 
                workload.doc_candidates, workload.expected_complexity)
        ):
            
            # Add performance context with some variation
            perf_context = {
                'cpu_utilization': np.random.normal(0.6, 0.1),
                'memory_usage_mb': np.random.normal(2048, 200),
                'throughput_qps': np.random.normal(100, 10)
            }
            
            # Run optimization
            opt_result = optimizer.optimize_query(
                query, context, candidates, perf_context
            )
            
            # Record results
            results['optimization_results'].append({
                'query_index': i,
                'query': query[:100],  # Truncate for storage
                'predicted_complexity': opt_result.lagrangian_result.query_complexity.value,
                'expected_complexity': expected_complexity.value,
                'latency_ms': opt_result.total_latency_ms,
                'cbu_improvement': opt_result.total_cbu_improvement,
                'quality_preservation': opt_result.total_quality_preservation,
                'computational_savings': opt_result.total_computational_savings,
                'early_exit_used': opt_result.early_exit_triggered,
                'matryoshka_dim': opt_result.matryoshka_dimension,
                'k1_candidates': opt_result.k1_candidates,
                'k2_candidates': opt_result.k2_candidates,
                'promotion_ready': opt_result.promotion_criteria_met,
                'warnings': opt_result.warnings
            })
            
            # Track metrics
            latencies.append(opt_result.total_latency_ms)
            cbu_improvements.append(opt_result.total_cbu_improvement)
            quality_preservations.append(opt_result.total_quality_preservation)
            computational_savings.append(opt_result.total_computational_savings)
            
            # Check classification accuracy
            if opt_result.lagrangian_result.query_complexity == expected_complexity:
                classification_correct += 1
        
        # Compute aggregate metrics
        results['performance_metrics'] = {
            'mean_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'latency_std': np.std(latencies),
            'mean_cbu_improvement': np.mean(cbu_improvements),
            'min_cbu_improvement': np.min(cbu_improvements),
            'cbu_std': np.std(cbu_improvements),
            'mean_quality_preservation': np.mean(quality_preservations),
            'min_quality_preservation': np.min(quality_preservations),
            'mean_computational_savings': np.mean(computational_savings),
            'early_exit_rate': sum(1 for r in results['optimization_results'] if r['early_exit_used']) / len(results['optimization_results'])
        }
        
        results['query_classification'] = {
            'accuracy': classification_correct / len(workload.queries),
            'total_queries': len(workload.queries),
            'correct_classifications': classification_correct
        }
        
        return results
    
    def _compare_workload_results(
        self,
        baseline: Dict[str, Any],
        optimized: Dict[str, Any],
        targets: Dict[str, float]
    ) -> Dict[str, Any]:
        """Compare baseline vs optimized results for a workload."""
        
        baseline_perf = baseline['performance_metrics']
        optimized_perf = optimized['performance_metrics']
        
        # Compute improvements
        latency_improvement = baseline_perf['p95_latency_ms'] - optimized_perf['p95_latency_ms']
        latency_improvement_percent = (latency_improvement / baseline_perf['p95_latency_ms']) * 100
        
        cbu_delta = optimized_perf['mean_cbu_improvement'] - baseline_perf['mean_cbu_improvement']
        
        quality_delta = optimized_perf['mean_quality_preservation'] - baseline_perf['mean_quality_preservation']
        
        computational_savings = optimized_perf['mean_computational_savings']
        
        return {
            'latency_improvement': {
                'absolute_ms': latency_improvement,
                'percent': latency_improvement_percent,
                'target_met': optimized_perf['p95_latency_ms'] <= targets['p95_latency']
            },
            'cbu_performance': {
                'delta': cbu_delta,
                'optimized_value': optimized_perf['mean_cbu_improvement'],
                'target_met': optimized_perf['mean_cbu_improvement'] >= targets['cbu_improvement']
            },
            'quality_preservation': {
                'delta': quality_delta,
                'optimized_value': optimized_perf['mean_quality_preservation'],
                'acceptable': optimized_perf['min_quality_preservation'] >= 0.85
            },
            'computational_efficiency': {
                'savings': computational_savings,
                'early_exit_effectiveness': optimized['performance_metrics']['early_exit_rate']
            },
            'classification_accuracy': {
                'baseline': baseline['query_classification']['accuracy'],
                'optimized': optimized['query_classification']['accuracy']
            }
        }
    
    def _analyze_aggregate_performance(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze aggregate performance across all workloads."""
        
        # Aggregate metrics across all workloads
        all_optimized_latencies = []
        all_optimized_cbu = []
        all_optimized_quality = []
        all_baseline_latencies = []
        
        promotion_ready_count = 0
        total_queries = 0
        
        for workload_name, workload_results in test_results.items():
            optimized_results = workload_results['optimized']['optimization_results']
            baseline_results = workload_results['baseline']['optimization_results']
            
            for result in optimized_results:
                all_optimized_latencies.append(result['latency_ms'])
                all_optimized_cbu.append(result['cbu_improvement'])
                all_optimized_quality.append(result['quality_preservation'])
                if result['promotion_ready']:
                    promotion_ready_count += 1
                total_queries += 1
            
            for result in baseline_results:
                all_baseline_latencies.append(result['latency_ms'])
        
        # Compute aggregate metrics
        aggregate_p95_latency = np.percentile(all_optimized_latencies, 95)
        aggregate_mean_cbu = np.mean(all_optimized_cbu)
        aggregate_min_quality = np.min(all_optimized_quality)
        
        baseline_p95_latency = np.percentile(all_baseline_latencies, 95)
        
        # Critical target achievement
        p95_target_met = aggregate_p95_latency <= 1.0
        cbu_target_met = aggregate_mean_cbu >= 10.0
        quality_preserved = aggregate_min_quality >= 0.85
        
        # Promotion criteria
        latency_improvement = baseline_p95_latency - aggregate_p95_latency
        promotion_criteria_met = (
            (aggregate_mean_cbu >= 10.0) or  # CBU criterion
            (latency_improvement >= 5.0)     # P95 improvement criterion
        )
        
        return {
            'aggregate_metrics': {
                'p95_latency_ms': aggregate_p95_latency,
                'mean_cbu_improvement': aggregate_mean_cbu,
                'min_quality_preservation': aggregate_min_quality,
                'latency_improvement_vs_baseline': latency_improvement,
                'promotion_ready_rate': promotion_ready_count / total_queries
            },
            'target_achievement': {
                'p95_latency_target_met': p95_target_met,
                'cbu_target_met': cbu_target_met,
                'quality_preserved': quality_preserved,
                'all_targets_met': p95_target_met and cbu_target_met and quality_preserved
            },
            'promotion_criteria': {
                'criteria_met': promotion_criteria_met,
                'cbu_criterion': aggregate_mean_cbu >= 10.0,
                'latency_improvement_criterion': latency_improvement >= 5.0,
                'recommendation': 'PROMOTE' if promotion_criteria_met and quality_preserved else 'CONTINUE_OPTIMIZATION'
            },
            'performance_summary': {
                'baseline_p95_latency_ms': baseline_p95_latency,
                'optimized_p95_latency_ms': aggregate_p95_latency,
                'latency_reduction_percent': (latency_improvement / baseline_p95_latency) * 100,
                'cbu_maintained': aggregate_mean_cbu >= 10.0,
                'quality_impact': aggregate_min_quality,
                'total_queries_tested': total_queries
            }
        }
    
    def _analyze_promotion_criteria(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze promotion criteria in detail."""
        
        aggregate = performance_data['aggregate_metrics']
        targets = performance_data['target_achievement']
        promotion = performance_data['promotion_criteria']
        
        # Detailed promotion analysis
        promotion_score = 0
        promotion_factors = []
        
        # P95 latency achievement (weight: 40%)
        if targets['p95_latency_target_met']:
            promotion_score += 40
            promotion_factors.append(f"✅ P95 latency {aggregate['p95_latency_ms']:.2f}ms ≤ 1ms target")
        else:
            promotion_factors.append(f"❌ P95 latency {aggregate['p95_latency_ms']:.2f}ms > 1ms target")
        
        # CBU improvement (weight: 35%)
        if targets['cbu_target_met']:
            promotion_score += 35
            promotion_factors.append(f"✅ CBU {aggregate['mean_cbu_improvement']:.1f}% ≥ 10% target")
        else:
            promotion_factors.append(f"❌ CBU {aggregate['mean_cbu_improvement']:.1f}% < 10% target")
        
        # Quality preservation (weight: 20%)
        if targets['quality_preserved']:
            promotion_score += 20
            promotion_factors.append(f"✅ Quality {aggregate['min_quality_preservation']:.3f} ≥ 0.85 threshold")
        else:
            promotion_factors.append(f"❌ Quality {aggregate['min_quality_preservation']:.3f} < 0.85 threshold")
        
        # System stability (weight: 5%)
        if aggregate['promotion_ready_rate'] >= 0.8:
            promotion_score += 5
            promotion_factors.append(f"✅ Promotion ready rate {aggregate['promotion_ready_rate']:.1%} ≥ 80%")
        else:
            promotion_factors.append(f"❌ Promotion ready rate {aggregate['promotion_ready_rate']:.1%} < 80%")
        
        # Overall recommendation
        if promotion_score >= 80:
            recommendation = "PROMOTE TO PRODUCTION"
            confidence = "HIGH"
        elif promotion_score >= 60:
            recommendation = "PROMOTE TO CANARY"
            confidence = "MEDIUM"
        else:
            recommendation = "CONTINUE OPTIMIZATION"
            confidence = "LOW"
        
        return {
            'promotion_score': promotion_score,
            'promotion_factors': promotion_factors,
            'recommendation': recommendation,
            'confidence': confidence,
            'criteria_analysis': {
                'cbu_criterion_met': promotion['cbu_criterion'],
                'latency_improvement_met': promotion['latency_improvement_criterion'],
                'quality_threshold_met': targets['quality_preserved'],
                'system_stability': aggregate['promotion_ready_rate'] >= 0.8
            },
            'risk_assessment': self._assess_promotion_risks(aggregate, targets)
        }
    
    def _assess_promotion_risks(
        self, aggregate: Dict[str, Any], targets: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess risks associated with promotion."""
        
        risks = []
        risk_score = 0  # 0 = low risk, 100 = high risk
        
        # Latency risk
        if aggregate['p95_latency_ms'] > 1.2:
            risks.append("HIGH LATENCY RISK: P95 significantly above target")
            risk_score += 30
        elif aggregate['p95_latency_ms'] > 1.0:
            risks.append("MEDIUM LATENCY RISK: P95 slightly above target")
            risk_score += 15
        
        # Quality risk
        if aggregate['min_quality_preservation'] < 0.85:
            risks.append("QUALITY RISK: Minimum quality below threshold")
            risk_score += 25
        elif aggregate['min_quality_preservation'] < 0.9:
            risks.append("MINOR QUALITY RISK: Quality near threshold")
            risk_score += 10
        
        # CBU risk
        if aggregate['mean_cbu_improvement'] < 10.0:
            risks.append("CBU RISK: Below promotion threshold")
            risk_score += 20
        elif aggregate['mean_cbu_improvement'] < 11.0:
            risks.append("MINOR CBU RISK: Near promotion threshold")
            risk_score += 5
        
        # Stability risk
        if aggregate['promotion_ready_rate'] < 0.7:
            risks.append("STABILITY RISK: Low promotion ready rate")
            risk_score += 15
        
        risk_level = "LOW" if risk_score < 20 else "MEDIUM" if risk_score < 50 else "HIGH"
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'identified_risks': risks,
            'mitigation_required': risk_score >= 30
        }
    
    def _analyze_system_stability(self) -> Dict[str, Any]:
        """Analyze system stability and monitoring effectiveness."""
        
        # Analyze optimizer performance
        optimized_diagnostics = self.optimized_optimizer.get_system_status()
        baseline_diagnostics = self.baseline_optimizer.get_system_status()
        
        # Check monitoring system
        monitor_data = self.optimized_optimizer.performance_monitor.get_dashboard_data()
        
        return {
            'system_health': {
                'optimized_system': optimized_diagnostics['current_performance'],
                'baseline_system': baseline_diagnostics['current_performance'],
                'monitoring_active': monitor_data['status'] != 'no_data'
            },
            'component_status': optimized_diagnostics['component_status'],
            'optimization_trends': optimized_diagnostics['optimization_trends'],
            'alert_system': {
                'active_alerts': len(self.optimized_optimizer.performance_monitor.active_alerts),
                'alert_types': [alert.alert_level.value for alert in self.optimized_optimizer.performance_monitor.active_alerts]
            },
            'stability_assessment': {
                'lambda_stable': abs(optimized_diagnostics['component_status']['lagrangian_optimizer']['lambda_multiplier'] - 1.0) < 0.5,
                'no_critical_alerts': len([a for a in self.optimized_optimizer.performance_monitor.active_alerts if a.alert_level.name == 'CRITICAL']) == 0,
                'monitoring_functional': monitor_data['status'] != 'no_data'
            }
        }
    
    def _generate_final_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate final recommendations based on validation results."""
        
        recommendations = []
        
        performance = validation_results['performance_comparison']['aggregate_metrics']
        promotion = validation_results['promotion_analysis']
        
        # Primary recommendation
        if promotion['recommendation'] == "PROMOTE TO PRODUCTION":
            recommendations.append("🎯 RECOMMENDATION: PROMOTE TO PRODUCTION - All criteria met with high confidence")
        elif promotion['recommendation'] == "PROMOTE TO CANARY":
            recommendations.append("🔄 RECOMMENDATION: PROMOTE TO CANARY - Most criteria met, monitor closely")
        else:
            recommendations.append("⚠️ RECOMMENDATION: CONTINUE OPTIMIZATION - Key criteria not yet met")
        
        # Specific optimization suggestions
        if performance['p95_latency_ms'] > 1.0:
            recommendations.append(f"🚀 LATENCY: Further reduce P95 from {performance['p95_latency_ms']:.2f}ms to ≤1ms")
            recommendations.append("   - Consider more aggressive Matryoshka embedding optimization")
            recommendations.append("   - Increase early-exit aggressiveness for standard queries")
            recommendations.append("   - Further reduce DPP rank for computational efficiency")
        
        if performance['mean_cbu_improvement'] < 10.0:
            recommendations.append(f"📈 CBU: Improve CBU from {performance['mean_cbu_improvement']:.1f}% to ≥10%")
            recommendations.append("   - Maintain 768d embeddings for hard queries")
            recommendations.append("   - Optimize K1/K2 scheduling for better quality/speed balance")
            recommendations.append("   - Fine-tune λ multiplier for quality preservation")
        
        if performance['min_quality_preservation'] < 0.85:
            recommendations.append(f"🛡️ QUALITY: Improve quality preservation from {performance['min_quality_preservation']:.3f} to ≥0.85")
            recommendations.append("   - Reduce optimization aggressiveness for complex queries")
            recommendations.append("   - Implement quality-aware early-exit thresholds")
        
        # System recommendations
        recommendations.append("🔧 SYSTEM OPTIMIZATIONS:")
        recommendations.append("   - Continue monitoring λ-drift and dual gap stability")
        recommendations.append("   - Track CBU-elasticity smoothness around operating point")
        recommendations.append("   - Monitor KV prefix-reuse efficiency")
        
        # Next steps
        if promotion['recommendation'].startswith("PROMOTE"):
            recommendations.append("📋 NEXT STEPS:")
            recommendations.append("   - Deploy to canary environment with comprehensive monitoring")
            recommendations.append("   - Run A/B testing against current production baseline")
            recommendations.append("   - Monitor for performance regressions and quality preservation")
            recommendations.append("   - Prepare rollback procedures if metrics degrade")
        
        return recommendations
    
    def _assess_validation_success(self, validation_results: Dict[str, Any]) -> bool:
        """Assess overall validation success."""
        
        performance = validation_results['performance_comparison']
        promotion = validation_results['promotion_analysis']
        
        # Critical success criteria
        latency_success = performance['target_achievement']['p95_latency_target_met']
        cbu_success = performance['target_achievement']['cbu_target_met'] 
        quality_success = performance['target_achievement']['quality_preserved']
        promotion_ready = promotion['recommendation'] != "CONTINUE OPTIMIZATION"
        
        # Overall success requires meeting primary objectives
        return latency_success and (cbu_success or quality_success) and promotion_ready
    
    def export_validation_report(
        self, validation_results: Dict[str, Any], output_path: str = "lethe_optimization_validation_report.json"
    ) -> str:
        """Export comprehensive validation report."""
        
        try:
            with open(output_path, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
            
            logger.info(f"Validation report exported to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export validation report: {e}")
            raise
    
    def create_performance_visualization(
        self, validation_results: Dict[str, Any], output_dir: str = "."
    ) -> List[str]:
        """Create performance visualization charts."""
        
        try:
            plt.style.use('seaborn-v0_8')
            fig_paths = []
            
            # 1. Latency Comparison Chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Extract workload performance data
            workload_names = []
            baseline_p95 = []
            optimized_p95 = []
            cbu_improvements = []
            
            for workload_name, results in validation_results['test_results'].items():
                workload_names.append(workload_name.replace('_', ' ').title())
                baseline_p95.append(results['baseline']['performance_metrics']['p95_latency_ms'])
                optimized_p95.append(results['optimized']['performance_metrics']['p95_latency_ms'])
                cbu_improvements.append(results['optimized']['performance_metrics']['mean_cbu_improvement'])
            
            # P95 Latency comparison
            x_pos = np.arange(len(workload_names))
            width = 0.35
            
            ax1.bar(x_pos - width/2, baseline_p95, width, label='Baseline', color='skyblue')
            ax1.bar(x_pos + width/2, optimized_p95, width, label='Optimized', color='lightcoral')
            ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Target (1ms)')
            
            ax1.set_xlabel('Workload')
            ax1.set_ylabel('P95 Latency (ms)')
            ax1.set_title('P95 Latency: Baseline vs Optimized')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(workload_names)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # CBU Improvement
            ax2.bar(x_pos, cbu_improvements, color='lightgreen')
            ax2.axhline(y=10.0, color='red', linestyle='--', alpha=0.7, label='Target (10%)')
            
            ax2.set_xlabel('Workload')
            ax2.set_ylabel('CBU Improvement (%)')
            ax2.set_title('CBU Improvement by Workload')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(workload_names)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            fig_path = f"{output_dir}/latency_cbu_comparison.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            fig_paths.append(fig_path)
            plt.close()
            
            # 2. Performance Metrics Summary
            fig, ax = plt.subplots(figsize=(12, 8))
            
            metrics = ['P95 Latency Target', 'CBU Target', 'Quality Preservation', 'Promotion Ready']
            achievements = []
            
            perf_data = validation_results['performance_comparison']['target_achievement']
            achievements.append(1 if perf_data['p95_latency_target_met'] else 0)
            achievements.append(1 if perf_data['cbu_target_met'] else 0)
            achievements.append(1 if perf_data['quality_preserved'] else 0)
            achievements.append(1 if validation_results['promotion_analysis']['recommendation'] != "CONTINUE_OPTIMIZATION" else 0)
            
            colors = ['green' if a == 1 else 'red' for a in achievements]
            bars = ax.bar(metrics, achievements, color=colors, alpha=0.7)
            
            # Add percentage labels
            for bar, achievement in zip(bars, achievements):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{"✅" if achievement else "❌"}',
                       ha='center', va='bottom', fontsize=16)
            
            ax.set_ylim(0, 1.2)
            ax.set_ylabel('Achievement Status')
            ax.set_title('Optimization Targets Achievement Summary')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            fig_path = f"{output_dir}/targets_achievement.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            fig_paths.append(fig_path)
            plt.close()
            
            logger.info(f"Performance visualizations created: {fig_paths}")
            return fig_paths
            
        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}")
            return []


async def main():
    """Main validation execution."""
    
    print("🚀 Starting Lethe Lagrangian Latency Optimization Validation")
    print("=" * 80)
    print(f"Target: Reduce P95 latency from 6.8ms → ≤1ms while maintaining CBU ≥10%")
    print("=" * 80)
    
    # Initialize validator
    validator = LatencyOptimizationValidator()
    
    # Run comprehensive validation
    validation_results = await validator.run_comprehensive_validation()
    
    # Export results
    report_path = validator.export_validation_report(validation_results)
    
    # Create visualizations
    viz_paths = validator.create_performance_visualization(validation_results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 VALIDATION RESULTS SUMMARY")
    print("=" * 80)
    
    performance = validation_results['performance_comparison']['aggregate_metrics']
    promotion = validation_results['promotion_analysis']
    
    print(f"🎯 P95 Latency: {performance['p95_latency_ms']:.2f}ms (Target: ≤1ms)")
    print(f"📈 CBU Improvement: {performance['mean_cbu_improvement']:.1f}% (Target: ≥10%)")
    print(f"🛡️ Quality Preservation: {performance['min_quality_preservation']:.3f} (Target: ≥0.85)")
    print(f"🚀 Latency Improvement: {performance['latency_improvement_vs_baseline']:.2f}ms")
    print(f"✅ Promotion Ready Rate: {performance['promotion_ready_rate']:.1%}")
    
    print(f"\n🎖️ PROMOTION RECOMMENDATION: {promotion['recommendation']}")
    print(f"🏆 Promotion Score: {promotion['promotion_score']}/100")
    print(f"🔒 Confidence Level: {promotion['confidence']}")
    
    print("\n📋 KEY RECOMMENDATIONS:")
    for rec in validation_results['recommendations'][:5]:
        print(f"   {rec}")
    
    print(f"\n📄 Full Report: {report_path}")
    if viz_paths:
        print(f"📊 Visualizations: {', '.join(viz_paths)}")
    
    success = validation_results.get('validation_success', False)
    if success:
        print("\n🎉 VALIDATION SUCCESSFUL - OPTIMIZATION TARGETS ACHIEVED! 🎉")
    else:
        print("\n⚠️ VALIDATION INCOMPLETE - CONTINUE OPTIMIZATION REQUIRED")
    
    print("=" * 80)
    
    return validation_results


if __name__ == "__main__":
    # Run the validation
    results = asyncio.run(main())