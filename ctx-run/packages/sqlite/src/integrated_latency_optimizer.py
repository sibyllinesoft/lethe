#!/usr/bin/env python3
"""
Integrated Lagrangian Latency-Quality Optimization System

Master system that integrates all optimization components to achieve
the critical performance targets: P95 ≤1ms, CBU ≥10%, with comprehensive
monitoring and quality preservation.

Target Achievement:
- Current: +6.8ms P95 (vs ≤1ms target) 
- Current: +12.5% CBU (excellent, maintain)
- Goal: Optimize latency without sacrificing quality

Key Integration Points:
1. Lagrangian optimizer for strategy selection
2. CE early-exit for computational efficiency  
3. Performance monitoring for real-time adaptation
4. Quality preservation across all optimizations
5. Comprehensive diagnostics and alerting

Operational Requirements:
- Maintain dual diagnostics with <0.5% dual gap
- Alert on λ-drift ≤ ±15% violations
- Track CBU-elasticity smoothness
- Monitor KV prefix-reuse (≥10pp Jaccard drop detection)
- Promote only if ΔCBU/GB ≥ +10% or P95 improvement ≥5ms
"""

import logging
import time
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any, NamedTuple, Union
from enum import Enum
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Import our optimization components
from .lagrangian_optimizer import (
    LagrangianOptimizer, LagrangianConfig, OptimizationResult, 
    QueryComplexity, QueryFeatures
)
from .ce_early_exit import (
    CalibratedCEEarlyExit, EarlyExitConfig, EarlyExitResult,
    CandidateScore, EarlyExitStrategy
)
from .performance_monitor import (
    PerformanceMonitor, PerformanceConfig, MetricSnapshot,
    PerformanceAlert, AlertLevel
)

logger = logging.getLogger(__name__)

class OptimizationMode(Enum):
    """System optimization modes."""
    QUALITY_FIRST = "quality_first"      # Prioritize CBU over latency
    BALANCED = "balanced"                # Balance CBU and latency
    SPEED_FIRST = "speed_first"         # Prioritize latency over CBU
    ADAPTIVE = "adaptive"               # Adapt based on query characteristics

@dataclass
class IntegratedConfig:
    """Master configuration for integrated optimization system."""
    
    # Target performance criteria
    target_p95_latency_ms: float = 1.0     # Critical: ≤1ms target
    target_cbu_improvement: float = 10.0   # Minimum for promotion
    current_cbu_baseline: float = 12.5     # Current excellent performance
    
    # Optimization mode
    optimization_mode: OptimizationMode = OptimizationMode.ADAPTIVE
    auto_mode_switching: bool = True       # Allow automatic mode switching
    
    # Component configurations
    lagrangian_config: Optional[LagrangianConfig] = None
    early_exit_config: Optional[EarlyExitConfig] = None
    monitoring_config: Optional[PerformanceConfig] = None
    
    # Integration parameters
    quality_preservation_weight: float = 0.7    # Weight for quality in decisions
    latency_optimization_weight: float = 0.3    # Weight for latency optimization
    
    # Adaptation parameters
    adaptation_window_size: int = 50      # Window for adaptation decisions
    performance_trend_sensitivity: float = 0.1  # Sensitivity to performance trends
    
    # Safety bounds
    max_quality_degradation: float = 0.1   # Maximum allowed quality loss
    max_latency_increase: float = 2.0      # Maximum latency increase (ms)
    
    # Experimental features
    enable_experimental_optimizations: bool = False
    experimental_risk_tolerance: float = 0.05

@dataclass
class IntegratedOptimizationResult:
    """Comprehensive result from integrated optimization."""
    
    # Component results
    lagrangian_result: OptimizationResult
    early_exit_result: Optional[EarlyExitResult]
    
    # Integrated metrics
    total_latency_ms: float
    total_cbu_improvement: float
    total_computational_savings: float
    total_quality_preservation: float
    
    # Optimization decisions
    optimization_mode_used: OptimizationMode
    matryoshka_dimension: int
    k1_candidates: int
    k2_candidates: int
    dpp_rank: int
    early_exit_triggered: bool
    
    # System state
    lambda_multiplier: float
    dual_gap: float
    lambda_drift: float
    promotion_criteria_met: bool
    
    # Performance tracking
    stage_timings: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class IntegratedLatencyOptimizer:
    """
    Master optimization system integrating all components for latency-quality balance.
    
    Orchestrates Lagrangian optimization, CE early-exit, and performance monitoring
    to achieve critical performance targets while preserving system quality.
    
    Key Responsibilities:
    1. Strategy selection and parameter optimization
    2. Real-time performance monitoring and adaptation
    3. Quality preservation across all optimization stages
    4. Alert management and performance regression detection
    5. Comprehensive reporting and diagnostics
    6. Promotion criteria evaluation and recommendation
    
    Mathematical Framework:
    - Lagrangian optimization: L(λ) = CBU - λ×(latency - target)
    - Early exit: Stop when gain/token < λ×threshold with high confidence
    - Quality preservation: Maintain CBU ≥ baseline - max_degradation
    """
    
    def __init__(self, config: Optional[IntegratedConfig] = None):
        """Initialize integrated latency optimization system."""
        self.config = config or IntegratedConfig()
        
        # Initialize component configurations with integration
        self._initialize_component_configs()
        
        # Initialize core components
        self.lagrangian_optimizer = LagrangianOptimizer(self.config.lagrangian_config)
        self.early_exit_system = CalibratedCEEarlyExit(self.config.early_exit_config)
        self.performance_monitor = PerformanceMonitor(self.config.monitoring_config)
        
        # System state
        self.current_mode = self.config.optimization_mode
        self.optimization_history: List[IntegratedOptimizationResult] = []
        self.performance_trends: Dict[str, List[float]] = {
            'latency': [],
            'cbu': [],
            'quality': []
        }
        
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="lethe_opt")
        
        # Performance tracking
        self.total_queries_processed = 0
        self.promotion_recommendations: List[Dict[str, Any]] = []
        
        logger.info(
            f"IntegratedLatencyOptimizer initialized: "
            f"target P95={self.config.target_p95_latency_ms}ms, "
            f"CBU baseline={self.config.current_cbu_baseline}%, "
            f"mode={self.current_mode.value}"
        )
    
    def optimize_query(
        self,
        query: str,
        session_context: Dict[str, Any],
        doc_candidates: List[Dict[str, Any]],
        performance_context: Optional[Dict[str, Any]] = None
    ) -> IntegratedOptimizationResult:
        """
        Execute integrated optimization for a single query.
        
        Args:
            query: Input query text
            session_context: Session context and history
            doc_candidates: Document candidates for optimization
            performance_context: Current performance metrics
            
        Returns:
            IntegratedOptimizationResult with complete optimization data
        """
        optimization_start = time.time()
        stage_timings = {}
        
        try:
            self.total_queries_processed += 1
            
            # Stage 1: Lagrangian optimization for strategy selection
            with self._time_stage("lagrangian_optimization", stage_timings):
                lagrangian_result = self.lagrangian_optimizer.optimize_query(
                    query, session_context, performance_context
                )
            
            # Stage 2: Prepare candidates for CE early-exit
            with self._time_stage("candidate_preparation", stage_timings):
                candidate_scores = self._prepare_candidate_scores(
                    doc_candidates, lagrangian_result
                )
            
            # Stage 3: CE early-exit optimization (if beneficial)
            early_exit_result = None
            if (lagrangian_result.use_early_exit and 
                len(candidate_scores) >= self.config.early_exit_config.calibrated_prefix_min):
                
                with self._time_stage("early_exit_optimization", stage_timings):
                    early_exit_result = self.early_exit_system.process_candidates(
                        candidate_scores,
                        lagrangian_result.lambda_value,
                        self._extract_query_characteristics(query, session_context)
                    )
            
            # Stage 4: Compute integrated performance metrics
            with self._time_stage("performance_integration", stage_timings):
                integrated_result = self._compute_integrated_result(
                    lagrangian_result, early_exit_result, optimization_start, stage_timings
                )
            
            # Stage 5: Update performance monitoring
            with self._time_stage("performance_monitoring", stage_timings):
                self._update_performance_monitoring(integrated_result, performance_context)
            
            # Stage 6: Adaptation and mode switching
            with self._time_stage("system_adaptation", stage_timings):
                self._adapt_system_parameters(integrated_result)
            
            # Stage 7: Generate recommendations
            recommendations = self._generate_recommendations(integrated_result)
            integrated_result.recommendations = recommendations
            
            # Store result history
            self.optimization_history.append(integrated_result)
            
            # Log optimization summary
            logger.info(
                f"Integrated optimization complete: "
                f"latency={integrated_result.total_latency_ms:.2f}ms, "
                f"CBU={integrated_result.total_cbu_improvement:.1f}%, "
                f"mode={integrated_result.optimization_mode_used.value}, "
                f"promotion_ready={integrated_result.promotion_criteria_met}"
            )
            
            return integrated_result
            
        except Exception as e:
            logger.error(f"Integrated optimization failed: {e}")
            return self._create_fallback_result(query, optimization_start, str(e))
    
    def _initialize_component_configs(self):
        """Initialize component configurations with integration parameters."""
        
        # Lagrangian configuration
        if self.config.lagrangian_config is None:
            self.config.lagrangian_config = LagrangianConfig(
                target_latency_p95_ms=self.config.target_p95_latency_ms,
                current_cbu_improvement=self.config.current_cbu_baseline,
                target_cbu_threshold=self.config.target_cbu_improvement
            )
        
        # Early-exit configuration
        if self.config.early_exit_config is None:
            # More aggressive early-exit for speed-focused optimization
            strategy = EarlyExitStrategy.CALIBRATED
            if self.config.optimization_mode == OptimizationMode.SPEED_FIRST:
                strategy = EarlyExitStrategy.ADAPTIVE
            elif self.config.optimization_mode == OptimizationMode.QUALITY_FIRST:
                strategy = EarlyExitStrategy.THRESHOLD_BASED  # More conservative
                
            self.config.early_exit_config = EarlyExitConfig(
                strategy=strategy,
                gain_per_token_base_threshold=0.001,  # Aggressive for latency
                posterior_confidence_min=0.8,
                min_quality_preservation=self.config.current_cbu_baseline * 0.9
            )
        
        # Performance monitoring configuration
        if self.config.monitoring_config is None:
            self.config.monitoring_config = PerformanceConfig(
                target_p95_latency_ms=self.config.target_p95_latency_ms,
                target_cbu_improvement=self.config.target_cbu_improvement,
                promotion_cbu_delta_threshold=self.config.target_cbu_improvement,
                promotion_p95_improvement_threshold=5.0  # 5ms improvement threshold
            )
    
    def _prepare_candidate_scores(
        self, 
        doc_candidates: List[Dict[str, Any]], 
        lagrangian_result: OptimizationResult
    ) -> List[CandidateScore]:
        """Prepare document candidates for CE early-exit processing."""
        
        candidate_scores = []
        
        for i, candidate in enumerate(doc_candidates):
            # Extract candidate information
            doc_id = candidate.get('id', f'doc_{i}')
            raw_score = candidate.get('score', 0.0)
            text_length = len(candidate.get('text', ''))
            
            # Normalize score (0-1 range)
            normalized_score = min(1.0, max(0.0, raw_score))
            
            # Estimate token count
            token_count = max(50, int(text_length / 4))  # Rough tokenization estimate
            
            candidate_score = CandidateScore(
                candidate_id=doc_id,
                raw_score=raw_score,
                normalized_score=normalized_score,
                position=i,
                token_count=token_count
            )
            
            candidate_scores.append(candidate_score)
        
        return candidate_scores
    
    def _extract_query_characteristics(
        self, query: str, session_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract query characteristics for early-exit adaptation."""
        
        tokens = query.lower().split()
        unique_tokens = set(tokens)
        
        # Basic characteristics
        characteristics = {
            'query_length': len(tokens),
            'unique_token_ratio': len(unique_tokens) / max(len(tokens), 1),
            'avg_token_length': np.mean([len(token) for token in tokens]) if tokens else 0,
            'has_exact_identifiers': any(
                len(token) >= 8 and (token.isalnum() or '-' in token)
                for token in tokens
            ),
            'semantic_complexity': min(5.0, len(tokens) * 0.1 + len(unique_tokens) * 0.2)
        }
        
        # Session context influence
        if session_context:
            turn_count = session_context.get('turn_count', 1)
            characteristics['session_depth'] = min(10.0, turn_count * 0.5)
            
            # Historical complexity
            if 'previous_queries' in session_context:
                prev_queries = session_context['previous_queries']
                if prev_queries:
                    avg_prev_length = np.mean([len(q.split()) for q in prev_queries])
                    characteristics['complexity_trend'] = len(tokens) / max(avg_prev_length, 1)
        
        return characteristics
    
    def _compute_integrated_result(
        self,
        lagrangian_result: OptimizationResult,
        early_exit_result: Optional[EarlyExitResult],
        optimization_start: float,
        stage_timings: Dict[str, float]
    ) -> IntegratedOptimizationResult:
        """Compute integrated optimization result from all components."""
        
        # Base metrics from Lagrangian optimization
        total_latency = lagrangian_result.actual_latency_ms
        total_cbu = lagrangian_result.cbu_improvement
        total_savings = lagrangian_result.computational_savings
        total_quality = lagrangian_result.quality_preservation
        
        # Apply early-exit improvements if used
        if early_exit_result and early_exit_result.early_exit_triggered:
            # Early exit reduces latency and increases computational savings
            early_exit_latency_reduction = early_exit_result.computational_tokens_saved * 0.00001  # ms per token
            total_latency = max(0.1, total_latency - early_exit_latency_reduction)
            
            # Update computational savings
            total_savings = min(1.0, total_savings + (early_exit_result.computational_tokens_saved / 10000.0))
            
            # Quality preservation from early exit
            early_exit_quality = early_exit_result.quality_preservation_score
            total_quality = (total_quality + early_exit_quality) / 2.0
        
        # Check promotion criteria
        promotion_criteria_met = self._evaluate_promotion_criteria(
            total_latency, total_cbu, total_savings
        )
        
        # Generate warnings
        warnings = lagrangian_result.warnings.copy()
        if early_exit_result and early_exit_result.exit_decision:
            if early_exit_result.exit_decision.quality_preservation_estimate < 0.9:
                warnings.append("Early exit may impact quality preservation")
        
        if total_latency > self.config.target_p95_latency_ms:
            warnings.append(f"Latency {total_latency:.2f}ms exceeds target {self.config.target_p95_latency_ms}ms")
        
        if total_cbu < self.config.target_cbu_improvement:
            warnings.append(f"CBU {total_cbu:.1f}% below promotion threshold {self.config.target_cbu_improvement}%")
        
        return IntegratedOptimizationResult(
            lagrangian_result=lagrangian_result,
            early_exit_result=early_exit_result,
            total_latency_ms=total_latency,
            total_cbu_improvement=total_cbu,
            total_computational_savings=total_savings,
            total_quality_preservation=total_quality,
            optimization_mode_used=self.current_mode,
            matryoshka_dimension=lagrangian_result.embedding_dimension,
            k1_candidates=lagrangian_result.k1_candidates,
            k2_candidates=lagrangian_result.k2_candidates,
            dpp_rank=lagrangian_result.dpp_rank,
            early_exit_triggered=bool(early_exit_result and early_exit_result.early_exit_triggered),
            lambda_multiplier=lagrangian_result.lambda_value,
            dual_gap=lagrangian_result.dual_gap,
            lambda_drift=0.0,  # Computed separately
            promotion_criteria_met=promotion_criteria_met,
            stage_timings=stage_timings,
            warnings=warnings
        )
    
    def _evaluate_promotion_criteria(
        self, latency: float, cbu: float, savings: float
    ) -> bool:
        """Evaluate if current optimization meets promotion criteria."""
        
        # Criteria: ΔCBU/GB ≥ +10% OR P95 improvement ≥5ms
        cbu_threshold_met = cbu >= self.config.target_cbu_improvement
        
        # P95 improvement (current vs baseline)
        baseline_p95 = 6.8  # Current +6.8ms baseline
        p95_improvement = baseline_p95 - latency
        p95_threshold_met = p95_improvement >= 5.0
        
        return cbu_threshold_met or p95_threshold_met
    
    def _update_performance_monitoring(
        self,
        result: IntegratedOptimizationResult,
        performance_context: Optional[Dict[str, Any]]
    ):
        """Update performance monitoring with integrated result."""
        
        # Compute λ-drift from recent history
        lambda_drift = 0.0
        if len(self.optimization_history) >= 2:
            previous_lambda = self.optimization_history[-1].lambda_multiplier
            current_lambda = result.lambda_multiplier
            lambda_drift = (current_lambda - previous_lambda) / max(previous_lambda, 1e-6)
        
        # Extract system metrics
        cpu_utilization = performance_context.get('cpu_utilization', 0.0) if performance_context else 0.0
        memory_usage = performance_context.get('memory_usage_mb', 0.0) if performance_context else 0.0
        throughput = performance_context.get('throughput_qps', 0.0) if performance_context else 0.0
        
        # Record in performance monitor
        self.performance_monitor.record_optimization_result(
            lambda_multiplier=result.lambda_multiplier,
            p95_latency_ms=result.total_latency_ms,
            cbu_improvement=result.total_cbu_improvement,
            dual_gap=result.dual_gap,
            lambda_drift=lambda_drift,
            computational_savings=result.total_computational_savings,
            quality_preservation=result.total_quality_preservation,
            prefix_jaccard_similarity=result.lagrangian_result.prefix_jaccard_similarity,
            query_type=result.lagrangian_result.query_complexity.value,
            early_exit_used=result.early_exit_triggered,
            cpu_utilization=cpu_utilization,
            memory_usage_mb=memory_usage,
            throughput_qps=throughput
        )
        
        # Update performance trends
        self.performance_trends['latency'].append(result.total_latency_ms)
        self.performance_trends['cbu'].append(result.total_cbu_improvement)
        self.performance_trends['quality'].append(result.total_quality_preservation)
        
        # Keep trend windows manageable
        max_trend_size = self.config.adaptation_window_size * 2
        for trend_list in self.performance_trends.values():
            if len(trend_list) > max_trend_size:
                trend_list[:] = trend_list[-max_trend_size:]
    
    def _adapt_system_parameters(self, result: IntegratedOptimizationResult):
        """Adapt system parameters based on optimization results."""
        
        if not self.config.auto_mode_switching:
            return
        
        # Analyze recent performance trends
        if len(self.optimization_history) < self.config.adaptation_window_size:
            return
        
        recent_results = self.optimization_history[-self.config.adaptation_window_size:]
        
        # Compute trend metrics
        recent_latencies = [r.total_latency_ms for r in recent_results]
        recent_cbu = [r.total_cbu_improvement for r in recent_results]
        recent_quality = [r.total_quality_preservation for r in recent_results]
        
        latency_trend = np.polyfit(range(len(recent_latencies)), recent_latencies, 1)[0]
        cbu_trend = np.polyfit(range(len(recent_cbu)), recent_cbu, 1)[0]
        quality_trend = np.polyfit(range(len(recent_quality)), recent_quality, 1)[0]
        
        # Adaptation logic
        current_avg_latency = np.mean(recent_latencies[-10:])
        current_avg_cbu = np.mean(recent_cbu[-10:])
        
        # Mode switching logic
        if current_avg_latency > self.config.target_p95_latency_ms * 1.5:
            # Latency too high - switch to speed focus
            if self.current_mode != OptimizationMode.SPEED_FIRST:
                logger.info("Switching to SPEED_FIRST mode due to high latency")
                self.current_mode = OptimizationMode.SPEED_FIRST
                self._reconfigure_for_speed_focus()
        
        elif current_avg_cbu < self.config.target_cbu_improvement:
            # CBU too low - switch to quality focus
            if self.current_mode != OptimizationMode.QUALITY_FIRST:
                logger.info("Switching to QUALITY_FIRST mode due to low CBU")
                self.current_mode = OptimizationMode.QUALITY_FIRST
                self._reconfigure_for_quality_focus()
        
        elif (current_avg_latency <= self.config.target_p95_latency_ms and 
              current_avg_cbu >= self.config.target_cbu_improvement):
            # Both targets met - switch to balanced mode
            if self.current_mode != OptimizationMode.BALANCED:
                logger.info("Switching to BALANCED mode - targets achieved")
                self.current_mode = OptimizationMode.BALANCED
                self._reconfigure_for_balanced_mode()
    
    def _reconfigure_for_speed_focus(self):
        """Reconfigure system components for speed-first optimization."""
        
        # Adjust early-exit for more aggressive speed optimization
        self.early_exit_system.config.strategy = EarlyExitStrategy.ADAPTIVE
        self.early_exit_system.config.gain_per_token_base_threshold *= 1.5  # More aggressive
        self.early_exit_system.config.posterior_confidence_min = 0.7  # Lower confidence threshold
        
        # Adjust Lagrangian config for speed focus
        self.lagrangian_optimizer.config.target_latency_p95_ms = self.config.target_p95_latency_ms
        
        logger.info("Reconfigured for SPEED_FIRST optimization")
    
    def _reconfigure_for_quality_focus(self):
        """Reconfigure system components for quality-first optimization."""
        
        # Adjust early-exit for quality preservation
        self.early_exit_system.config.strategy = EarlyExitStrategy.THRESHOLD_BASED
        self.early_exit_system.config.gain_per_token_base_threshold *= 0.7  # More conservative
        self.early_exit_system.config.posterior_confidence_min = 0.9  # Higher confidence
        
        # Adjust Lagrangian config for quality focus
        self.lagrangian_optimizer.config.current_cbu_improvement = self.config.current_cbu_baseline
        
        logger.info("Reconfigured for QUALITY_FIRST optimization")
    
    def _reconfigure_for_balanced_mode(self):
        """Reconfigure system components for balanced optimization."""
        
        # Reset to balanced settings
        self.early_exit_system.config.strategy = EarlyExitStrategy.CALIBRATED
        self.early_exit_system.config.gain_per_token_base_threshold = 0.001  # Default
        self.early_exit_system.config.posterior_confidence_min = 0.8  # Balanced
        
        logger.info("Reconfigured for BALANCED optimization")
    
    def _generate_recommendations(self, result: IntegratedOptimizationResult) -> List[str]:
        """Generate recommendations based on optimization results."""
        
        recommendations = []
        
        # Latency recommendations
        if result.total_latency_ms > self.config.target_p95_latency_ms:
            latency_excess = result.total_latency_ms - self.config.target_p95_latency_ms
            if latency_excess > 2.0:
                recommendations.append("Consider more aggressive Matryoshka embedding reduction (256d→128d for standard queries)")
            if result.dpp_rank > 14:
                recommendations.append("Reduce DPP rank further to decrease computational overhead")
            if not result.early_exit_triggered:
                recommendations.append("Enable CE early-exit to reduce reranking overhead")
        
        # CBU recommendations
        if result.total_cbu_improvement < self.config.target_cbu_improvement:
            cbu_deficit = self.config.target_cbu_improvement - result.total_cbu_improvement
            if cbu_deficit > 5.0:
                recommendations.append("Increase embedding dimension for hard queries to preserve quality")
            if result.k2_candidates < 400:
                recommendations.append("Increase K2 candidate count to improve selection quality")
            if result.early_exit_triggered:
                recommendations.append("Reduce early-exit aggressiveness to preserve more candidates")
        
        # System health recommendations
        if result.dual_gap > 0.005:
            recommendations.append("λ-optimization showing instability - review convergence parameters")
        
        if result.total_quality_preservation < 0.85:
            recommendations.append("Quality preservation below threshold - reduce optimization aggressiveness")
        
        # Promotion readiness
        if result.promotion_criteria_met:
            recommendations.append("✅ PROMOTION READY: Meets criteria for canary→production deployment")
        else:
            recommendations.append("❌ Not promotion ready - continue optimization")
        
        return recommendations
    
    def _time_stage(self, stage_name: str, timings: Dict[str, float]):
        """Context manager for timing optimization stages."""
        return StageTimingContext(stage_name, timings)
    
    def _create_fallback_result(
        self, query: str, start_time: float, error: str
    ) -> IntegratedOptimizationResult:
        """Create fallback result when optimization fails."""
        
        # Create minimal Lagrangian result
        fallback_lagrangian = OptimizationResult(
            query_complexity=QueryComplexity.STANDARD,
            embedding_dimension=256,
            k1_candidates=1000,
            k2_candidates=200,
            dpp_rank=16,
            use_early_exit=False,
            group_split_threshold=0.7,
            actual_latency_ms=3.0,  # Conservative fallback
            cbu_improvement=8.0,    # Conservative fallback
            computational_savings=0.0,
            quality_preservation=1.0,
            lambda_value=1.0,
            dual_gap=0.0,
            orthogonal_mass_preserved=1.0,
            prefix_jaccard_similarity=0.7,
            optimization_time_ms=(time.time() - start_time) * 1000,
            warnings=[f"Optimization failed: {error}"]
        )
        
        return IntegratedOptimizationResult(
            lagrangian_result=fallback_lagrangian,
            early_exit_result=None,
            total_latency_ms=3.0,
            total_cbu_improvement=8.0,
            total_computational_savings=0.0,
            total_quality_preservation=1.0,
            optimization_mode_used=self.current_mode,
            matryoshka_dimension=256,
            k1_candidates=1000,
            k2_candidates=200,
            dpp_rank=16,
            early_exit_triggered=False,
            lambda_multiplier=1.0,
            dual_gap=0.0,
            lambda_drift=0.0,
            promotion_criteria_met=False,
            warnings=[f"Fallback mode due to error: {error}"]
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and diagnostics."""
        
        # Performance summary
        perf_summary = self.performance_monitor.get_performance_summary()
        
        # Recent optimization metrics
        recent_results = self.optimization_history[-10:] if self.optimization_history else []
        
        if recent_results:
            avg_latency = np.mean([r.total_latency_ms for r in recent_results])
            avg_cbu = np.mean([r.total_cbu_improvement for r in recent_results])
            avg_savings = np.mean([r.total_computational_savings for r in recent_results])
            promotion_ready_rate = sum(1 for r in recent_results if r.promotion_criteria_met) / len(recent_results)
        else:
            avg_latency = avg_cbu = avg_savings = promotion_ready_rate = 0.0
        
        return {
            'system_overview': {
                'total_queries_processed': self.total_queries_processed,
                'current_optimization_mode': self.current_mode.value,
                'auto_mode_switching': self.config.auto_mode_switching,
                'target_p95_latency_ms': self.config.target_p95_latency_ms,
                'target_cbu_improvement': self.config.target_cbu_improvement
            },
            'current_performance': {
                'avg_p95_latency_ms': avg_latency,
                'avg_cbu_improvement': avg_cbu,
                'avg_computational_savings': avg_savings,
                'promotion_ready_rate': promotion_ready_rate,
                'latency_target_met': avg_latency <= self.config.target_p95_latency_ms,
                'cbu_target_met': avg_cbu >= self.config.target_cbu_improvement
            },
            'component_status': {
                'lagrangian_optimizer': {
                    'active': True,
                    'lambda_multiplier': self.lagrangian_optimizer.state.lambda_multiplier,
                    'recent_cpu_utilization': self.lagrangian_optimizer.state.recent_cpu_utilization
                },
                'early_exit_system': {
                    'active': True,
                    'strategy': self.early_exit_system.config.strategy.value,
                    'isotonic_model_available': self.early_exit_system.isotonic_model is not None
                },
                'performance_monitor': {
                    'active': True,
                    'active_alerts': len(self.performance_monitor.active_alerts),
                    'monitoring_window': len(self.performance_monitor.real_time_metrics)
                }
            },
            'performance_monitoring': perf_summary,
            'optimization_trends': {
                'latency_trend': 'improving' if len(self.performance_trends['latency']) > 1 and 
                                self.performance_trends['latency'][-1] < self.performance_trends['latency'][0] else 'stable',
                'cbu_trend': 'improving' if len(self.performance_trends['cbu']) > 1 and 
                            self.performance_trends['cbu'][-1] > self.performance_trends['cbu'][0] else 'stable',
                'quality_trend': 'stable' if len(self.performance_trends['quality']) > 1 else 'unknown'
            }
        }
    
    def generate_promotion_report(self) -> Dict[str, Any]:
        """Generate comprehensive promotion readiness report."""
        
        if len(self.optimization_history) < 20:
            return {'status': 'insufficient_data', 'message': 'Need more optimization data for promotion analysis'}
        
        recent_results = self.optimization_history[-50:]
        
        # Compute promotion metrics
        latencies = [r.total_latency_ms for r in recent_results]
        cbu_scores = [r.total_cbu_improvement for r in recent_results]
        quality_scores = [r.total_quality_preservation for r in recent_results]
        
        # Statistical analysis
        p95_latency = np.percentile(latencies, 95)
        mean_cbu = np.mean(cbu_scores)
        min_quality = np.min(quality_scores)
        
        # Promotion criteria evaluation
        latency_criterion_met = p95_latency <= self.config.target_p95_latency_ms
        cbu_criterion_met = mean_cbu >= self.config.target_cbu_improvement
        
        # P95 improvement criterion (vs baseline)
        baseline_p95 = 6.8  # Current system baseline
        p95_improvement = baseline_p95 - p95_latency
        p95_improvement_criterion_met = p95_improvement >= 5.0
        
        # Overall promotion decision
        promotion_ready = latency_criterion_met and (cbu_criterion_met or p95_improvement_criterion_met)
        
        # Risk assessment
        risks = []
        if min_quality < 0.85:
            risks.append(f"Quality preservation risk: minimum {min_quality:.3f} < 0.85")
        if np.std(latencies) > 0.5:
            risks.append(f"Latency variability risk: σ={np.std(latencies):.3f}ms > 0.5ms")
        if np.std(cbu_scores) > 2.0:
            risks.append(f"CBU instability risk: σ={np.std(cbu_scores):.3f}% > 2.0%")
        
        return {
            'promotion_ready': promotion_ready,
            'criteria_evaluation': {
                'p95_latency': {
                    'current': p95_latency,
                    'target': self.config.target_p95_latency_ms,
                    'met': latency_criterion_met,
                    'improvement_vs_baseline': baseline_p95 - p95_latency
                },
                'cbu_improvement': {
                    'current': mean_cbu,
                    'target': self.config.target_cbu_improvement,
                    'met': cbu_criterion_met
                },
                'p95_improvement': {
                    'improvement': p95_improvement,
                    'threshold': 5.0,
                    'met': p95_improvement_criterion_met
                }
            },
            'quality_assessment': {
                'mean_quality_preservation': np.mean(quality_scores),
                'min_quality_preservation': min_quality,
                'quality_stable': np.std(quality_scores) < 0.05
            },
            'performance_stability': {
                'latency_volatility': np.std(latencies),
                'cbu_volatility': np.std(cbu_scores),
                'system_stable': len(risks) == 0
            },
            'recommendation': 'PROMOTE' if promotion_ready and len(risks) == 0 else 'CONTINUE_OPTIMIZATION',
            'risks': risks,
            'sample_size': len(recent_results),
            'confidence_level': min(1.0, len(recent_results) / 100.0)  # Higher confidence with more samples
        }


class StageTimingContext:
    """Context manager for timing optimization stages."""
    
    def __init__(self, stage_name: str, timings_dict: Dict[str, float]):
        self.stage_name = stage_name
        self.timings_dict = timings_dict
        self.start_time = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (time.time() - self.start_time) * 1000
        self.timings_dict[self.stage_name] = duration


def create_integrated_optimizer(config: Optional[IntegratedConfig] = None) -> IntegratedLatencyOptimizer:
    """Create integrated latency optimization system."""
    return IntegratedLatencyOptimizer(config)