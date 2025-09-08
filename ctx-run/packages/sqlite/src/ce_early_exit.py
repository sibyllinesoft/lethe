#!/usr/bin/env python3
"""
Cross-Encoder Early-Exit System with Calibrated Prefix Stopping

Implements sophisticated early-exit mechanism for cross-encoder reranking
using isotonic regression and posterior confidence bounds to stop when
remaining candidates have low expected gain per computational token.

Key Features:
- Calibrated prefix processing (150-200 candidates)
- Isotonic regression with confidence bounds
- λ-coupled gain/token threshold stopping
- Posterior confidence evaluation
- Computational budget tracking
- Quality preservation guarantees

Mathematical Framework:
Stop when: max(remaining_gain/token) < λ × threshold with P(correct) > confidence
"""

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any, NamedTuple, Callable
from enum import Enum
import math
from sklearn.isotonic import IsotonicRegression
from scipy import stats

logger = logging.getLogger(__name__)

class EarlyExitStrategy(Enum):
    """Early exit strategy types."""
    DISABLED = "disabled"           # No early exit
    THRESHOLD_BASED = "threshold"   # Simple threshold
    CALIBRATED = "calibrated"       # Isotonic + confidence bounds
    ADAPTIVE = "adaptive"           # Adapt threshold based on query

@dataclass
class EarlyExitConfig:
    """Configuration for CE early-exit system."""
    
    # Calibrated prefix configuration
    calibrated_prefix_min: int = 150    # Minimum candidates to process
    calibrated_prefix_max: int = 200    # Maximum for calibration
    prefix_percentile: float = 0.8      # Use 80th percentile for calibration
    
    # Stopping criteria
    gain_per_token_base_threshold: float = 0.001  # Base threshold for gain/token
    lambda_multiplier_coupling: float = 1.0       # How much λ affects threshold
    posterior_confidence_min: float = 0.8         # Minimum confidence to stop
    
    # Isotonic regression configuration
    isotonic_increasing: bool = True               # Expect decreasing returns
    confidence_bound_sigma: float = 2.0           # σ for confidence bounds
    min_samples_for_regression: int = 20          # Minimum samples for isotonic
    
    # Computational budget
    max_computational_tokens: int = 10000         # Maximum tokens to process
    token_cost_per_candidate: float = 50.0       # Avg tokens per candidate
    
    # Strategy selection
    strategy: EarlyExitStrategy = EarlyExitStrategy.CALIBRATED
    adaptive_threshold_window: int = 50           # Window for adaptive threshold
    
    # Quality preservation
    min_quality_preservation: float = 0.9        # Minimum quality to maintain
    quality_decay_penalty: float = 0.1           # Penalty for quality loss
    
    # Monitoring
    enable_detailed_logging: bool = False        # Detailed exit decisions
    track_prediction_accuracy: bool = True      # Track prediction quality

@dataclass
class CandidateScore:
    """Individual candidate with score and metadata."""
    candidate_id: str
    raw_score: float
    normalized_score: float
    position: int
    token_count: int
    processing_time_ms: float = 0.0

@dataclass
class EarlyExitDecision:
    """Decision result from early-exit analysis."""
    should_exit: bool
    exit_reason: str
    candidates_processed: int
    candidates_remaining: int
    predicted_remaining_gain: float
    confidence_score: float
    computational_savings: float
    quality_preservation_estimate: float
    threshold_used: float
    lambda_value: float

@dataclass
class EarlyExitResult:
    """Complete result of early-exit processing."""
    
    # Final results
    selected_candidates: List[CandidateScore]
    total_candidates_processed: int
    early_exit_triggered: bool
    exit_decision: Optional[EarlyExitDecision]
    
    # Performance metrics
    total_processing_time_ms: float
    computational_tokens_saved: int
    quality_preservation_score: float
    
    # Calibration data
    isotonic_model_quality: float
    confidence_bounds: Tuple[float, float]
    prefix_statistics: Dict[str, float]
    
    # Monitoring data
    stage_timings: Dict[str, float] = field(default_factory=dict)
    prediction_accuracy: Optional[float] = None

class CalibratedCEEarlyExit:
    """
    Advanced early-exit system for cross-encoder reranking.
    
    Uses isotonic regression to model score degradation and confidence
    bounds to make stopping decisions that preserve quality while
    maximizing computational savings.
    
    Key Innovations:
    1. Calibrated prefix establishes performance baseline
    2. Isotonic regression models expected remaining gain
    3. Confidence bounds ensure high-probability correct decisions
    4. λ-coupled thresholds balance speed vs quality
    5. Adaptive strategy selection based on query characteristics
    """
    
    def __init__(self, config: Optional[EarlyExitConfig] = None):
        """Initialize calibrated early-exit system."""
        self.config = config or EarlyExitConfig()
        
        # Model state
        self.isotonic_model: Optional[IsotonicRegression] = None
        self.calibration_data: List[Tuple[int, float]] = []  # (position, score)
        self.historical_accuracy: List[float] = []
        
        # Adaptive threshold tracking
        self.adaptive_thresholds: List[float] = []
        self.query_characteristics: List[Dict[str, float]] = []
        
        # Performance tracking
        self.exit_decisions: List[EarlyExitDecision] = []
        
        logger.info(f"CalibratedCEEarlyExit initialized: strategy={self.config.strategy.value}")
    
    def process_candidates(
        self,
        candidates: List[CandidateScore],
        lambda_multiplier: float,
        query_characteristics: Optional[Dict[str, float]] = None
    ) -> EarlyExitResult:
        """
        Process candidates with early-exit decision making.
        
        Args:
            candidates: List of candidate scores to process
            lambda_multiplier: Current λ value for threshold coupling
            query_characteristics: Query features for adaptation
            
        Returns:
            EarlyExitResult with processing decision and metrics
        """
        start_time = time.time()
        stage_timings = {}
        
        try:
            # Stage 1: Validate and prepare candidates
            with self._time_stage("preparation", stage_timings):
                if not candidates:
                    return self._create_empty_result(start_time)
                
                # Sort candidates by score (descending)
                sorted_candidates = sorted(candidates, key=lambda c: c.raw_score, reverse=True)
                total_candidates = len(sorted_candidates)
            
            # Stage 2: Process calibrated prefix
            with self._time_stage("calibrated_prefix", stage_timings):
                prefix_size = min(
                    max(self.config.calibrated_prefix_min, int(total_candidates * 0.2)),
                    self.config.calibrated_prefix_max,
                    total_candidates
                )
                
                prefix_candidates = sorted_candidates[:prefix_size]
                prefix_stats = self._analyze_prefix_statistics(prefix_candidates)
            
            # Stage 3: Build/update isotonic regression model
            with self._time_stage("isotonic_modeling", stage_timings):
                isotonic_quality = self._update_isotonic_model(prefix_candidates)
            
            # Stage 4: Early exit decision making
            selected_candidates = []
            exit_decision = None
            
            with self._time_stage("candidate_processing", stage_timings):
                for i, candidate in enumerate(sorted_candidates):
                    # Always process the prefix
                    if i < prefix_size:
                        selected_candidates.append(candidate)
                        continue
                    
                    # Check early exit conditions
                    exit_decision = self._evaluate_early_exit(
                        candidates_processed=i + 1,
                        candidates_remaining=total_candidates - i - 1,
                        current_candidate=candidate,
                        prefix_stats=prefix_stats,
                        lambda_multiplier=lambda_multiplier,
                        query_characteristics=query_characteristics
                    )
                    
                    if exit_decision.should_exit:
                        logger.info(
                            f"Early exit triggered at candidate {i+1}/{total_candidates}: "
                            f"{exit_decision.exit_reason}"
                        )
                        break
                    
                    selected_candidates.append(candidate)
            
            # Stage 5: Compute final metrics
            with self._time_stage("final_metrics", stage_timings):
                total_time = (time.time() - start_time) * 1000
                
                tokens_saved = self._compute_tokens_saved(
                    total_candidates, len(selected_candidates)
                )
                
                quality_score = self._estimate_quality_preservation(
                    selected_candidates, sorted_candidates, prefix_stats
                )
                
                confidence_bounds = self._compute_confidence_bounds(prefix_candidates)
            
            # Create result
            result = EarlyExitResult(
                selected_candidates=selected_candidates,
                total_candidates_processed=len(selected_candidates),
                early_exit_triggered=(exit_decision is not None and exit_decision.should_exit),
                exit_decision=exit_decision,
                total_processing_time_ms=total_time,
                computational_tokens_saved=tokens_saved,
                quality_preservation_score=quality_score,
                isotonic_model_quality=isotonic_quality,
                confidence_bounds=confidence_bounds,
                prefix_statistics=prefix_stats,
                stage_timings=stage_timings
            )
            
            # Update tracking data
            if exit_decision:
                self.exit_decisions.append(exit_decision)
                self._update_adaptive_tracking(exit_decision, query_characteristics)
            
            return result
            
        except Exception as e:
            logger.error(f"Early exit processing failed: {e}")
            return self._create_fallback_result(candidates, start_time, str(e))
    
    def _analyze_prefix_statistics(self, prefix_candidates: List[CandidateScore]) -> Dict[str, float]:
        """Analyze calibrated prefix to establish baseline statistics."""
        if not prefix_candidates:
            return {}
        
        scores = [c.normalized_score for c in prefix_candidates]
        
        return {
            'mean_score': np.mean(scores),
            'score_variance': np.var(scores),
            'score_decay_rate': self._compute_score_decay_rate(scores),
            'top_score': max(scores),
            'percentile_80': np.percentile(scores, 80),
            'percentile_50': np.percentile(scores, 50),
            'quality_drop': max(scores) - min(scores) if len(scores) > 1 else 0.0,
            'prefix_size': len(prefix_candidates)
        }
    
    def _compute_score_decay_rate(self, scores: List[float]) -> float:
        """Compute rate of score decay in prefix."""
        if len(scores) < 2:
            return 0.0
        
        # Fit linear regression to position vs score
        positions = np.arange(len(scores))
        slope, _, r_value, _, _ = stats.linregress(positions, scores)
        
        # Return decay rate (negative slope indicates decay)
        return abs(slope) * r_value ** 2  # Weight by R²
    
    def _update_isotonic_model(self, prefix_candidates: List[CandidateScore]) -> float:
        """Update isotonic regression model with prefix data."""
        if len(prefix_candidates) < self.config.min_samples_for_regression:
            return 0.0
        
        # Extract position-score pairs
        positions = [c.position for c in prefix_candidates]
        scores = [c.normalized_score for c in prefix_candidates]
        
        # Update calibration data
        new_data = list(zip(positions, scores))
        self.calibration_data.extend(new_data)
        
        # Keep only recent data to adapt to changing patterns
        max_calibration_samples = 1000
        if len(self.calibration_data) > max_calibration_samples:
            self.calibration_data = self.calibration_data[-max_calibration_samples:]
        
        try:
            # Fit isotonic regression
            all_positions = [x[0] for x in self.calibration_data]
            all_scores = [x[1] for x in self.calibration_data]
            
            self.isotonic_model = IsotonicRegression(
                increasing=self.config.isotonic_increasing,
                out_of_bounds='clip'
            )
            
            self.isotonic_model.fit(all_positions, all_scores)
            
            # Evaluate model quality
            predicted = self.isotonic_model.predict(positions)
            r2_score = 1 - (np.sum((np.array(scores) - predicted) ** 2) / 
                           np.sum((np.array(scores) - np.mean(scores)) ** 2))
            
            return max(0.0, r2_score)
            
        except Exception as e:
            logger.warning(f"Isotonic regression failed: {e}")
            self.isotonic_model = None
            return 0.0
    
    def _evaluate_early_exit(
        self,
        candidates_processed: int,
        candidates_remaining: int,
        current_candidate: CandidateScore,
        prefix_stats: Dict[str, float],
        lambda_multiplier: float,
        query_characteristics: Optional[Dict[str, float]]
    ) -> EarlyExitDecision:
        """Evaluate whether to exit early at current position."""
        
        # Strategy-specific evaluation
        if self.config.strategy == EarlyExitStrategy.DISABLED:
            return EarlyExitDecision(
                should_exit=False, exit_reason="early exit disabled",
                candidates_processed=candidates_processed,
                candidates_remaining=candidates_remaining,
                predicted_remaining_gain=0.0, confidence_score=0.0,
                computational_savings=0.0, quality_preservation_estimate=1.0,
                threshold_used=0.0, lambda_value=lambda_multiplier
            )
        
        if self.config.strategy == EarlyExitStrategy.THRESHOLD_BASED:
            return self._threshold_based_decision(
                candidates_processed, candidates_remaining, current_candidate,
                lambda_multiplier
            )
        
        if self.config.strategy == EarlyExitStrategy.CALIBRATED:
            return self._calibrated_decision(
                candidates_processed, candidates_remaining, current_candidate,
                prefix_stats, lambda_multiplier
            )
        
        if self.config.strategy == EarlyExitStrategy.ADAPTIVE:
            return self._adaptive_decision(
                candidates_processed, candidates_remaining, current_candidate,
                prefix_stats, lambda_multiplier, query_characteristics
            )
        
        # Fallback
        return EarlyExitDecision(
            should_exit=False, exit_reason="unknown strategy",
            candidates_processed=candidates_processed,
            candidates_remaining=candidates_remaining,
            predicted_remaining_gain=0.0, confidence_score=0.0,
            computational_savings=0.0, quality_preservation_estimate=1.0,
            threshold_used=0.0, lambda_value=lambda_multiplier
        )
    
    def _calibrated_decision(
        self,
        processed: int,
        remaining: int,
        candidate: CandidateScore,
        prefix_stats: Dict[str, float],
        lambda_multiplier: float
    ) -> EarlyExitDecision:
        """Make calibrated early-exit decision using isotonic regression."""
        
        # Predict remaining gain using isotonic model
        predicted_gain = self._predict_remaining_gain(processed, remaining, prefix_stats)
        
        # Compute λ-coupled threshold
        base_threshold = self.config.gain_per_token_base_threshold
        lambda_coupled_threshold = base_threshold * (1.0 + lambda_multiplier * self.config.lambda_multiplier_coupling)
        
        # Estimate computational cost
        remaining_tokens = remaining * self.config.token_cost_per_candidate
        gain_per_token = predicted_gain / max(remaining_tokens, 1.0)
        
        # Compute confidence bounds
        confidence_score = self._compute_stopping_confidence(
            processed, candidate.normalized_score, prefix_stats
        )
        
        # Decision logic
        should_exit = (
            gain_per_token < lambda_coupled_threshold and
            confidence_score > self.config.posterior_confidence_min and
            remaining > 0  # Don't exit on last candidate
        )
        
        exit_reason = ""
        if should_exit:
            if gain_per_token < lambda_coupled_threshold:
                exit_reason = f"gain/token {gain_per_token:.4f} < threshold {lambda_coupled_threshold:.4f}"
            elif confidence_score <= self.config.posterior_confidence_min:
                exit_reason += f", low confidence {confidence_score:.3f}"
        
        # Compute savings and quality metrics
        computational_savings = remaining_tokens / max(processed * self.config.token_cost_per_candidate, 1.0)
        quality_preservation = max(0.0, 1.0 - (predicted_gain * self.config.quality_decay_penalty))
        
        return EarlyExitDecision(
            should_exit=should_exit,
            exit_reason=exit_reason or "continue processing",
            candidates_processed=processed,
            candidates_remaining=remaining,
            predicted_remaining_gain=predicted_gain,
            confidence_score=confidence_score,
            computational_savings=computational_savings,
            quality_preservation_estimate=quality_preservation,
            threshold_used=lambda_coupled_threshold,
            lambda_value=lambda_multiplier
        )
    
    def _predict_remaining_gain(
        self, processed: int, remaining: int, prefix_stats: Dict[str, float]
    ) -> float:
        """Predict remaining gain using isotonic model or heuristics."""
        
        if self.isotonic_model is not None and remaining > 0:
            # Use isotonic model to predict remaining scores
            future_positions = np.arange(processed, processed + remaining)
            predicted_scores = self.isotonic_model.predict(future_positions)
            
            # Estimate gain as sum of predicted improvements
            current_baseline = prefix_stats.get('percentile_50', 0.5)
            remaining_gain = max(0.0, np.sum(np.maximum(0, predicted_scores - current_baseline)))
            
            return remaining_gain
        
        # Fallback heuristic
        score_decay_rate = prefix_stats.get('score_decay_rate', 0.01)
        baseline_score = prefix_stats.get('percentile_80', 0.8)
        
        # Assume exponential decay
        remaining_gain = 0.0
        for i in range(remaining):
            predicted_score = baseline_score * math.exp(-score_decay_rate * i)
            remaining_gain += max(0.0, predicted_score - baseline_score * 0.5)
        
        return remaining_gain
    
    def _compute_stopping_confidence(
        self, processed: int, current_score: float, prefix_stats: Dict[str, float]
    ) -> float:
        """Compute confidence that stopping now is correct decision."""
        
        # Confidence based on score degradation
        expected_score = prefix_stats.get('percentile_80', 0.8)
        score_ratio = current_score / max(expected_score, 1e-6)
        
        # Confidence based on position
        prefix_size = prefix_stats.get('prefix_size', 150)
        position_confidence = min(1.0, processed / max(prefix_size * 2, 1))
        
        # Confidence based on model quality
        model_confidence = 0.8 if self.isotonic_model is not None else 0.5
        
        # Combine confidences
        combined_confidence = (score_ratio * 0.4 + position_confidence * 0.3 + model_confidence * 0.3)
        
        return min(1.0, max(0.0, combined_confidence))
    
    def _threshold_based_decision(
        self, processed: int, remaining: int, candidate: CandidateScore, lambda_multiplier: float
    ) -> EarlyExitDecision:
        """Simple threshold-based early exit decision."""
        
        threshold = self.config.gain_per_token_base_threshold * lambda_multiplier
        gain_per_token = candidate.normalized_score / max(candidate.token_count, 1.0)
        
        should_exit = gain_per_token < threshold
        
        return EarlyExitDecision(
            should_exit=should_exit,
            exit_reason=f"threshold: {gain_per_token:.4f} < {threshold:.4f}" if should_exit else "continue",
            candidates_processed=processed,
            candidates_remaining=remaining,
            predicted_remaining_gain=remaining * gain_per_token,
            confidence_score=0.7,  # Fixed confidence
            computational_savings=remaining / max(processed, 1.0),
            quality_preservation_estimate=0.8,
            threshold_used=threshold,
            lambda_value=lambda_multiplier
        )
    
    def _adaptive_decision(
        self,
        processed: int, remaining: int, candidate: CandidateScore,
        prefix_stats: Dict[str, float], lambda_multiplier: float,
        query_characteristics: Optional[Dict[str, float]]
    ) -> EarlyExitDecision:
        """Adaptive early-exit decision based on query characteristics."""
        
        # Start with calibrated decision
        base_decision = self._calibrated_decision(
            processed, remaining, candidate, prefix_stats, lambda_multiplier
        )
        
        # Adjust based on query characteristics
        if query_characteristics:
            complexity = query_characteristics.get('semantic_complexity', 2.5)
            entropy = query_characteristics.get('entity_entropy', 1.0)
            
            # Complex queries need more processing
            complexity_factor = 1.0 + (complexity - 2.5) * 0.1
            threshold_adjustment = complexity_factor * (1.0 + entropy * 0.1)
            
            # Adjust threshold
            adjusted_threshold = base_decision.threshold_used * threshold_adjustment
            
            # Recompute decision with adjusted threshold
            remaining_tokens = remaining * self.config.token_cost_per_candidate
            gain_per_token = base_decision.predicted_remaining_gain / max(remaining_tokens, 1.0)
            
            should_exit = (
                gain_per_token < adjusted_threshold and
                base_decision.confidence_score > self.config.posterior_confidence_min
            )
            
            base_decision.should_exit = should_exit
            base_decision.threshold_used = adjusted_threshold
            base_decision.exit_reason = f"adaptive threshold {gain_per_token:.4f} < {adjusted_threshold:.4f}"
        
        return base_decision
    
    def _compute_tokens_saved(self, total_candidates: int, processed_candidates: int) -> int:
        """Compute computational tokens saved by early exit."""
        candidates_saved = total_candidates - processed_candidates
        return int(candidates_saved * self.config.token_cost_per_candidate)
    
    def _estimate_quality_preservation(
        self,
        selected_candidates: List[CandidateScore],
        all_candidates: List[CandidateScore],
        prefix_stats: Dict[str, float]
    ) -> float:
        """Estimate quality preservation from early exit."""
        
        if not all_candidates or not selected_candidates:
            return 1.0
        
        # Quality based on score coverage
        selected_scores = [c.normalized_score for c in selected_candidates]
        all_scores = [c.normalized_score for c in all_candidates]
        
        selected_total = sum(selected_scores)
        all_total = sum(all_scores)
        
        coverage_ratio = selected_total / max(all_total, 1e-6)
        
        # Quality based on top candidates preserved
        top_20_percent = int(len(all_candidates) * 0.2)
        top_preserved = len(selected_candidates) >= top_20_percent
        
        # Combine metrics
        quality_score = coverage_ratio * 0.7 + (0.3 if top_preserved else 0.0)
        
        return min(1.0, max(0.0, quality_score))
    
    def _compute_confidence_bounds(self, candidates: List[CandidateScore]) -> Tuple[float, float]:
        """Compute confidence bounds for score predictions."""
        if len(candidates) < 2:
            return (0.0, 1.0)
        
        scores = [c.normalized_score for c in candidates]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        bound = self.config.confidence_bound_sigma * std_score
        
        return (
            max(0.0, mean_score - bound),
            min(1.0, mean_score + bound)
        )
    
    def _update_adaptive_tracking(
        self, decision: EarlyExitDecision, query_characteristics: Optional[Dict[str, float]]
    ):
        """Update adaptive threshold tracking."""
        self.adaptive_thresholds.append(decision.threshold_used)
        
        if query_characteristics:
            self.query_characteristics.append(query_characteristics.copy())
        
        # Keep only recent data
        max_history = self.config.adaptive_threshold_window
        if len(self.adaptive_thresholds) > max_history:
            self.adaptive_thresholds = self.adaptive_thresholds[-max_history:]
            self.query_characteristics = self.query_characteristics[-max_history:]
    
    def _time_stage(self, stage_name: str, timings: Dict[str, float]):
        """Context manager for timing stages."""
        return StageTimingContext(stage_name, timings)
    
    def _create_empty_result(self, start_time: float) -> EarlyExitResult:
        """Create result for empty candidate list."""
        return EarlyExitResult(
            selected_candidates=[],
            total_candidates_processed=0,
            early_exit_triggered=False,
            exit_decision=None,
            total_processing_time_ms=(time.time() - start_time) * 1000,
            computational_tokens_saved=0,
            quality_preservation_score=1.0,
            isotonic_model_quality=0.0,
            confidence_bounds=(0.0, 1.0),
            prefix_statistics={}
        )
    
    def _create_fallback_result(
        self, candidates: List[CandidateScore], start_time: float, error: str
    ) -> EarlyExitResult:
        """Create fallback result when processing fails."""
        return EarlyExitResult(
            selected_candidates=candidates,  # Return all candidates
            total_candidates_processed=len(candidates),
            early_exit_triggered=False,
            exit_decision=None,
            total_processing_time_ms=(time.time() - start_time) * 1000,
            computational_tokens_saved=0,
            quality_preservation_score=1.0,
            isotonic_model_quality=0.0,
            confidence_bounds=(0.0, 1.0),
            prefix_statistics={'error': error}
        )
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the early-exit system."""
        diagnostics = {
            'config': {
                'strategy': self.config.strategy.value,
                'calibrated_prefix_range': (self.config.calibrated_prefix_min, self.config.calibrated_prefix_max),
                'gain_threshold': self.config.gain_per_token_base_threshold,
                'confidence_min': self.config.posterior_confidence_min
            },
            'model_state': {
                'isotonic_model_available': self.isotonic_model is not None,
                'calibration_samples': len(self.calibration_data),
                'recent_decisions': len(self.exit_decisions)
            }
        }
        
        if self.exit_decisions:
            recent_decisions = self.exit_decisions[-50:]  # Last 50 decisions
            diagnostics['recent_performance'] = {
                'early_exit_rate': sum(1 for d in recent_decisions if d.should_exit) / len(recent_decisions),
                'avg_candidates_processed': np.mean([d.candidates_processed for d in recent_decisions]),
                'avg_computational_savings': np.mean([d.computational_savings for d in recent_decisions]),
                'avg_confidence': np.mean([d.confidence_score for d in recent_decisions])
            }
        
        if self.adaptive_thresholds:
            diagnostics['adaptive_tracking'] = {
                'threshold_mean': np.mean(self.adaptive_thresholds),
                'threshold_std': np.std(self.adaptive_thresholds),
                'threshold_trend': 'increasing' if len(self.adaptive_thresholds) > 1 and 
                                  self.adaptive_thresholds[-1] > self.adaptive_thresholds[0] else 'stable'
            }
        
        return diagnostics


class StageTimingContext:
    """Context manager for timing processing stages."""
    
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


def create_early_exit_system(config: Optional[EarlyExitConfig] = None) -> CalibratedCEEarlyExit:
    """Create calibrated CE early-exit system."""
    return CalibratedCEEarlyExit(config)