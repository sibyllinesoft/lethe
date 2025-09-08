#!/usr/bin/env python3
"""
Campaign Validation and Promotion Pipeline
==========================================

Implements comprehensive validation including:
1. Campaign-specific gates and validation
2. Promotion pipeline with paired, budget-matched full matrix validation
3. Comprehensive guardrails: coverage-weighted CRPS, KV-prefix penalties, curvature-gated increases
4. Holm-corrected significance testing
5. Union non-degradation across all datasets at 8/15/30%
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from scipy import stats
from statsmodels.stats.multitest import multipletests
import json
import logging
from pathlib import Path
from datetime import datetime
import warnings

# Integration with existing framework
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from campaign_manager import Campaign, Trial, CampaignSpec
from analysis.metrics import MetricsCalculator, EvaluationMetrics, StatisticalComparator

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation check"""
    check_name: str
    passed: bool
    value: float
    threshold: float
    description: str
    severity: str = "error"  # error, warning, info
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GuardrailViolation:
    """Guardrail violation details"""
    guardrail_name: str
    violation_type: str
    current_value: float
    limit_value: float
    severity: str
    recommendation: str
    affected_metrics: List[str] = field(default_factory=list)

@dataclass
class PromotionDecision:
    """Final promotion decision with rationale"""
    campaign_id: str
    trial_id: str
    decision: str  # "approve", "reject", "conditional"
    confidence_score: float
    
    # Validation results
    gate_results: List[ValidationResult]
    guardrail_violations: List[GuardrailViolation]
    statistical_tests: Dict[str, Any]
    
    # Cross-dataset validation
    dataset_results: Dict[str, Dict[str, Any]]
    union_non_degradation: bool
    
    # Decision rationale
    rationale: str
    conditions: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)

class CampaignValidator:
    """Validates individual campaigns against their gates"""
    
    def __init__(self):
        self.metrics_calculator = MetricsCalculator()
    
    def validate_trial(self, trial: Trial, spec: CampaignSpec) -> List[ValidationResult]:
        """Validate single trial against campaign gates"""
        results = []
        
        for gate_name, gate_spec in spec.gates.items():
            result = self._evaluate_gate(trial, gate_name, gate_spec)
            results.append(result)
        
        return results
    
    def _evaluate_gate(self, trial: Trial, gate_name: str, gate_spec: Any) -> ValidationResult:
        """Evaluate a specific gate"""
        try:
            if gate_name == "min_delta_p5":
                value = trial.metrics.get("delta_p5", 0.0)
                passed = value >= gate_spec
                return ValidationResult(
                    check_name=gate_name,
                    passed=passed,
                    value=value,
                    threshold=gate_spec,
                    description=f"P@5 improvement must be ≥ {gate_spec:.3f}",
                    severity="error"
                )
            
            elif gate_name == "min_ci_confidence":
                # Check if confidence interval is positive (improvement > 0)
                ci_lower = trial.metrics.get("ci_lower", -1.0)
                passed = ci_lower > gate_spec
                return ValidationResult(
                    check_name=gate_name,
                    passed=passed,
                    value=ci_lower,
                    threshold=gate_spec,
                    description=f"Confidence interval lower bound must be > {gate_spec}",
                    severity="error"
                )
            
            elif gate_name == "max_latency_p95_delta":
                value = trial.metrics.get("latency_p95_delta", 0.0)
                passed = value <= gate_spec
                return ValidationResult(
                    check_name=gate_name,
                    passed=passed,
                    value=value,
                    threshold=gate_spec,
                    description=f"P95 latency increase must be ≤ {gate_spec}ms",
                    severity="error"
                )
            
            elif gate_name == "max_kv_drop":
                value = trial.metrics.get("kv_prefix_drop", 0.0)
                passed = value <= gate_spec
                return ValidationResult(
                    check_name=gate_name,
                    passed=passed,
                    value=value,
                    threshold=gate_spec,
                    description=f"KV prefix drop must be ≤ {gate_spec:.2%}",
                    severity="error"
                )
            
            elif gate_name == "max_ece_fact_bin":
                value = trial.metrics.get("ece_fact_bin", 0.0)
                passed = value <= gate_spec
                return ValidationResult(
                    check_name=gate_name,
                    passed=passed,
                    value=value,
                    threshold=gate_spec,
                    description=f"ECE×FACT bin must be ≤ {gate_spec:.3f}",
                    severity="error"
                )
            
            elif gate_name == "max_ilp_used":
                value = trial.metrics.get("ilp_usage", 0.0)
                passed = value <= gate_spec
                return ValidationResult(
                    check_name=gate_name,
                    passed=passed,
                    value=value,
                    threshold=gate_spec,
                    description=f"ILP usage must be ≤ {gate_spec:.1%}",
                    severity="error"
                )
            
            elif gate_name == "max_closure_breaks":
                value = trial.metrics.get("closure_breaks", 0)
                passed = value <= gate_spec
                return ValidationResult(
                    check_name=gate_name,
                    passed=passed,
                    value=value,
                    threshold=gate_spec,
                    description=f"Closure breaks must be ≤ {gate_spec}",
                    severity="error"
                )
            
            elif gate_name == "min_kv_prefix_reuse":
                value = trial.metrics.get("kv_prefix_reuse_ratio", 1.0)
                passed = value >= gate_spec
                return ValidationResult(
                    check_name=gate_name,
                    passed=passed,
                    value=value,
                    threshold=gate_spec,
                    description=f"KV prefix reuse ratio must be ≥ {gate_spec}",
                    severity="error"
                )
            
            elif gate_name == "max_p99_p95_ratio":
                value = trial.metrics.get("p99_p95_ratio", 2.0)
                passed = value <= gate_spec
                return ValidationResult(
                    check_name=gate_name,
                    passed=passed,
                    value=value,
                    threshold=gate_spec,
                    description=f"P99/P95 ratio must be ≤ {gate_spec}",
                    severity="error"
                )
            
            else:
                # Generic numeric gate
                value = trial.metrics.get(gate_name, 0.0)
                if isinstance(gate_spec, dict):
                    min_val = gate_spec.get("min", float("-inf"))
                    max_val = gate_spec.get("max", float("inf"))
                    passed = min_val <= value <= max_val
                    description = f"{gate_name} must be in [{min_val}, {max_val}]"
                else:
                    passed = value >= gate_spec
                    description = f"{gate_name} must be ≥ {gate_spec}"
                
                return ValidationResult(
                    check_name=gate_name,
                    passed=passed,
                    value=value,
                    threshold=gate_spec,
                    description=description,
                    severity="error"
                )
        
        except Exception as e:
            logger.error(f"Failed to evaluate gate {gate_name}: {str(e)}")
            return ValidationResult(
                check_name=gate_name,
                passed=False,
                value=0.0,
                threshold=gate_spec,
                description=f"Gate evaluation failed: {str(e)}",
                severity="error"
            )

class Guardrails:
    """Implements comprehensive guardrails system"""
    
    def __init__(self):
        self.violation_history: Dict[str, List[GuardrailViolation]] = {}
    
    def check_all_guardrails(self, 
                           trial: Trial, 
                           spec: CampaignSpec,
                           baseline_metrics: Optional[Dict[str, float]] = None) -> List[GuardrailViolation]:
        """Check all guardrails for a trial"""
        violations = []
        
        # Coverage-weighted CRPS checks
        violations.extend(self._check_crps_coverage(trial, spec))
        
        # KV-prefix Jaccard penalties
        violations.extend(self._check_kv_jaccard_penalties(trial, spec))
        
        # Curvature-gated r increases
        violations.extend(self._check_curvature_gated_increases(trial, spec))
        
        # τ move caps with ILP monitoring
        violations.extend(self._check_tau_move_caps(trial, spec))
        
        # Memory and performance guardrails
        violations.extend(self._check_performance_guardrails(trial, spec, baseline_metrics))
        
        # Calibration guardrails  
        violations.extend(self._check_calibration_guardrails(trial, spec))
        
        return violations
    
    def _check_crps_coverage(self, trial: Trial, spec: CampaignSpec) -> List[GuardrailViolation]:
        """Check coverage-weighted CRPS for calibration drift"""
        violations = []
        
        crps_score = trial.metrics.get("crps_coverage_weighted", 0.0)
        crps_threshold = spec.validator_fences.get("max_crps_coverage", 0.10)
        
        if crps_score > crps_threshold:
            violations.append(GuardrailViolation(
                guardrail_name="CRPS Coverage",
                violation_type="calibration_drift",
                current_value=crps_score,
                limit_value=crps_threshold,
                severity="high",
                recommendation="Re-isotonic calibration required before trusting IPS deltas",
                affected_metrics=["delta_p5", "ece_score", "calibration_error"]
            ))
        
        return violations
    
    def _check_kv_jaccard_penalties(self, trial: Trial, spec: CampaignSpec) -> List[GuardrailViolation]:
        """Check KV-prefix Jaccard penalties for cache stability"""
        violations = []
        
        jaccard_drop = trial.metrics.get("kv_prefix_jaccard_drop", 0.0)
        jaccard_threshold = spec.validator_fences.get("max_kv_jaccard_drop", 0.05)
        
        if jaccard_drop > jaccard_threshold:
            violations.append(GuardrailViolation(
                guardrail_name="KV Jaccard Stability",
                violation_type="cache_instability",
                current_value=jaccard_drop,
                limit_value=jaccard_threshold,
                severity="medium",
                recommendation="Reduce streaming tail changes (W, s, sinks) to preserve KV reuse",
                affected_metrics=["kv_prefix_reuse", "memory_efficiency", "cache_hit_rate"]
            ))
        
        return violations
    
    def _check_curvature_gated_increases(self, trial: Trial, spec: CampaignSpec) -> List[GuardrailViolation]:
        """Check curvature-gated DPP rank increases"""
        violations = []
        
        r_value = trial.parameters.get("r_dpp", 16)
        curvature_spike = trial.metrics.get("curvature_spike_detected", False)
        
        if r_value > 16 and not curvature_spike:
            violations.append(GuardrailViolation(
                guardrail_name="DPP Rank Gating",
                violation_type="unjustified_complexity_increase",
                current_value=r_value,
                limit_value=16,
                severity="medium",
                recommendation="Only increase r when measured curvature spikes justify O(r²) cost",
                affected_metrics=["computational_cost", "memory_usage", "latency"]
            ))
        
        return violations
    
    def _check_tau_move_caps(self, trial: Trial, spec: CampaignSpec) -> List[GuardrailViolation]:
        """Check τ move caps with ILP monitoring"""
        violations = []
        
        tau_value = trial.parameters.get("tau_group_split", 0.75)
        baseline_tau = 0.75  # Default baseline
        tau_move = abs(tau_value - baseline_tau)
        max_tau_move = spec.validator_fences.get("max_tau_move", 0.1)
        
        if tau_move > max_tau_move:
            violations.append(GuardrailViolation(
                guardrail_name="Tau Move Cap",
                violation_type="excessive_parameter_change",
                current_value=tau_move,
                limit_value=max_tau_move,
                severity="medium",
                recommendation=f"Cap τ moves to ±{max_tau_move} to control ILP incidence",
                affected_metrics=["ilp_usage", "group_split_stability"]
            ))
        
        # Check ILP usage if τ was moved
        if tau_move > 0:
            ilp_usage = trial.metrics.get("ilp_usage", 0.0)
            max_ilp = spec.validator_fences.get("max_ilp_threshold", 0.10)
            
            if ilp_usage > max_ilp:
                violations.append(GuardrailViolation(
                    guardrail_name="ILP Usage Monitor",
                    violation_type="ilp_overflow",
                    current_value=ilp_usage,
                    limit_value=max_ilp,
                    severity="high",
                    recommendation="Alert: ILP_used > 10% - revert τ changes",
                    affected_metrics=["inference_stability", "completion_quality"]
                ))
        
        return violations
    
    def _check_performance_guardrails(self, 
                                    trial: Trial, 
                                    spec: CampaignSpec,
                                    baseline_metrics: Optional[Dict[str, float]]) -> List[GuardrailViolation]:
        """Check performance-related guardrails"""
        violations = []
        
        # P99/P95 ratio check
        p99_p95_ratio = trial.metrics.get("p99_p95_ratio", 2.0)
        max_ratio = spec.validator_fences.get("max_p99_p95_ratio", 2.5)
        
        if p99_p95_ratio > max_ratio:
            violations.append(GuardrailViolation(
                guardrail_name="Latency Tail Control",
                violation_type="latency_tail_inflation",
                current_value=p99_p95_ratio,
                limit_value=max_ratio,
                severity="medium",
                recommendation="Investigate latency tail causes - may indicate resource contention",
                affected_metrics=["p99_latency", "user_experience"]
            ))
        
        # Memory growth monitoring
        if baseline_metrics:
            memory_current = trial.metrics.get("memory_mb", 0.0)
            memory_baseline = baseline_metrics.get("memory_mb", memory_current)
            memory_growth = (memory_current - memory_baseline) / memory_baseline if memory_baseline > 0 else 0
            max_growth = spec.validator_fences.get("max_memory_growth", 0.20)
            
            if memory_growth > max_growth:
                violations.append(GuardrailViolation(
                    guardrail_name="Memory Growth Control",
                    violation_type="memory_inflation",
                    current_value=memory_growth,
                    limit_value=max_growth,
                    severity="high",
                    recommendation=f"Memory growth > {max_growth:.0%} - investigate memory leaks",
                    affected_metrics=["memory_usage", "system_stability"]
                ))
        
        return violations
    
    def _check_calibration_guardrails(self, trial: Trial, spec: CampaignSpec) -> List[GuardrailViolation]:
        """Check calibration and consistency guardrails"""
        violations = []
        
        # ECE drift check
        ece_drift = trial.metrics.get("ece_drift", 0.0)
        max_ece_drift = spec.validator_fences.get("max_ece_drift", 0.08)
        
        if ece_drift > max_ece_drift:
            violations.append(GuardrailViolation(
                guardrail_name="ECE Drift Control",
                violation_type="calibration_drift",
                current_value=ece_drift,
                limit_value=max_ece_drift,
                severity="high",
                recommendation="ECE drift too high - recalibration required",
                affected_metrics=["calibration_error", "confidence_reliability"]
            ))
        
        # Proxy gap check
        proxy_gap = trial.metrics.get("proxy_gap", 0.0)
        max_proxy_gap = spec.validator_fences.get("max_proxy_gap", 0.005)
        
        if proxy_gap > max_proxy_gap:
            violations.append(GuardrailViolation(
                guardrail_name="Proxy Gap Control",
                violation_type="metric_divergence",
                current_value=proxy_gap,
                limit_value=max_proxy_gap,
                severity="medium",
                recommendation="Proxy metrics diverging from ground truth - validate evaluation pipeline",
                affected_metrics=["evaluation_reliability", "metric_consistency"]
            ))
        
        return violations

class PromotionPipeline:
    """Implements full promotion validation pipeline"""
    
    def __init__(self):
        self.validator = CampaignValidator()
        self.guardrails = Guardrails()
        self.statistical_comparator = StatisticalComparator()
    
    def evaluate_for_promotion(self, 
                             campaign: Campaign,
                             baseline_metrics: Dict[str, float],
                             cross_dataset_results: Dict[str, Dict[str, Any]]) -> PromotionDecision:
        """Comprehensive promotion evaluation"""
        
        best_trial = campaign.best_trial
        if not best_trial:
            return PromotionDecision(
                campaign_id=campaign.campaign_id,
                trial_id="none",
                decision="reject",
                confidence_score=0.0,
                gate_results=[],
                guardrail_violations=[],
                statistical_tests={},
                dataset_results={},
                union_non_degradation=False,
                rationale="No successful trials found",
                next_steps=["Review campaign parameters and gates"]
            )
        
        # 1. Validate against campaign gates
        gate_results = self.validator.validate_trial(best_trial, campaign.spec)
        gates_passed = all(result.passed for result in gate_results)
        
        # 2. Check guardrails
        guardrail_violations = self.guardrails.check_all_guardrails(
            best_trial, campaign.spec, baseline_metrics
        )
        critical_violations = [v for v in guardrail_violations if v.severity == "high"]
        
        # 3. Statistical testing with Holm correction
        statistical_tests = self._perform_statistical_tests(
            best_trial, baseline_metrics, cross_dataset_results
        )
        
        # 4. Cross-dataset union non-degradation check
        union_non_degradation = self._check_union_non_degradation(
            cross_dataset_results, baseline_metrics
        )
        
        # 5. Make promotion decision
        decision, confidence_score, rationale, conditions, next_steps = self._make_promotion_decision(
            gates_passed, critical_violations, statistical_tests, union_non_degradation
        )
        
        return PromotionDecision(
            campaign_id=campaign.campaign_id,
            trial_id=best_trial.trial_id,
            decision=decision,
            confidence_score=confidence_score,
            gate_results=gate_results,
            guardrail_violations=guardrail_violations,
            statistical_tests=statistical_tests,
            dataset_results=cross_dataset_results,
            union_non_degradation=union_non_degradation,
            rationale=rationale,
            conditions=conditions,
            next_steps=next_steps
        )
    
    def _perform_statistical_tests(self, 
                                  trial: Trial, 
                                  baseline_metrics: Dict[str, float],
                                  cross_dataset_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical tests with Holm correction"""
        
        # Collect p-values from multiple comparisons
        p_values = []
        test_names = []
        test_results = {}
        
        # Primary metric test (P@5)
        delta_p5 = trial.metrics.get("delta_p5", 0.0)
        p_value_p5 = self._compute_significance_test(
            delta_p5, trial.metrics.get("delta_p5_std", 0.01)
        )
        p_values.append(p_value_p5)
        test_names.append("delta_p5")
        test_results["delta_p5"] = {
            "p_value": p_value_p5,
            "effect_size": delta_p5,
            "significant": p_value_p5 < 0.05
        }
        
        # Latency non-degradation test
        latency_delta = trial.metrics.get("latency_p95_delta", 0.0)
        p_value_latency = self._compute_significance_test(
            latency_delta, trial.metrics.get("latency_p95_delta_std", 1.0),
            one_tailed=True
        )
        p_values.append(p_value_latency)
        test_names.append("latency_non_degradation")
        test_results["latency_non_degradation"] = {
            "p_value": p_value_latency,
            "effect_size": latency_delta,
            "significant": p_value_latency < 0.05
        }
        
        # Cross-dataset consistency tests
        for dataset_name, dataset_result in cross_dataset_results.items():
            dataset_p_value = dataset_result.get("p_value", 1.0)
            p_values.append(dataset_p_value)
            test_names.append(f"dataset_{dataset_name}")
            test_results[f"dataset_{dataset_name}"] = {
                "p_value": dataset_p_value,
                "effect_size": dataset_result.get("delta_p5", 0.0),
                "significant": dataset_p_value < 0.05
            }
        
        # Apply Holm-Bonferroni correction
        if p_values:
            rejected, corrected_p_values, alpha_sidak, alpha_bonf = multipletests(
                p_values, alpha=0.05, method="holm"
            )
            
            for i, (test_name, corrected_p, is_significant) in enumerate(
                zip(test_names, corrected_p_values, rejected)
            ):
                test_results[test_name]["corrected_p_value"] = corrected_p
                test_results[test_name]["holm_significant"] = is_significant
        
        return {
            "individual_tests": test_results,
            "holm_correction_applied": True,
            "overall_significant": any(test_results[name]["holm_significant"] for name in test_names)
        }
    
    def _compute_significance_test(self, 
                                  effect: float, 
                                  std_error: float, 
                                  one_tailed: bool = False) -> float:
        """Compute significance test p-value"""
        if std_error <= 0:
            return 1.0
        
        z_score = effect / std_error
        
        if one_tailed:
            p_value = 1 - stats.norm.cdf(abs(z_score))
        else:
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return p_value
    
    def _check_union_non_degradation(self, 
                                    cross_dataset_results: Dict[str, Dict[str, Any]],
                                    baseline_metrics: Dict[str, float]) -> bool:
        """Check union non-degradation across all datasets at 8/15/30%"""
        
        budget_tiers = [8, 15, 30]
        
        for budget_tier in budget_tiers:
            tier_datasets = {
                name: result for name, result in cross_dataset_results.items()
                if result.get("budget_tier") == budget_tier
            }
            
            if not tier_datasets:
                continue
            
            # Check that no dataset shows significant degradation
            for dataset_name, dataset_result in tier_datasets.items():
                delta_p5 = dataset_result.get("delta_p5", 0.0)
                ci_lower = dataset_result.get("ci_lower", delta_p5)
                
                # Non-degradation means CI lower bound > -threshold  
                degradation_threshold = 0.005  # -0.5pp maximum acceptable degradation
                
                if ci_lower < -degradation_threshold:
                    logger.warning(
                        f"Union non-degradation violated: {dataset_name} at {budget_tier}% "
                        f"shows degradation (CI lower: {ci_lower:.4f})"
                    )
                    return False
        
        return True
    
    def _make_promotion_decision(self, 
                               gates_passed: bool,
                               critical_violations: List[GuardrailViolation],
                               statistical_tests: Dict[str, Any],
                               union_non_degradation: bool) -> Tuple[str, float, str, List[str], List[str]]:
        """Make final promotion decision"""
        
        conditions = []
        next_steps = []
        confidence_components = []
        
        # Gate compliance
        if not gates_passed:
            return (
                "reject",
                0.0,
                "Campaign gates not met",
                [],
                ["Review gate failures and adjust parameters"]
            )
        confidence_components.append(0.3)  # 30% for gate compliance
        
        # Critical guardrail violations
        if critical_violations:
            violation_names = [v.guardrail_name for v in critical_violations]
            return (
                "reject", 
                0.2,
                f"Critical guardrail violations: {', '.join(violation_names)}",
                [],
                [v.recommendation for v in critical_violations]
            )
        confidence_components.append(0.2)  # 20% for no critical violations
        
        # Statistical significance
        overall_significant = statistical_tests.get("overall_significant", False)
        if not overall_significant:
            conditions.append("Statistical significance borderline - require additional validation")
            confidence_components.append(0.1)  # 10% if not significant
        else:
            confidence_components.append(0.3)  # 30% for significance
        
        # Union non-degradation
        if not union_non_degradation:
            return (
                "reject",
                0.3, 
                "Union non-degradation check failed across datasets",
                [],
                ["Investigate cross-dataset degradation and adjust parameters"]
            )
        confidence_components.append(0.2)  # 20% for union non-degradation
        
        # Calculate confidence score
        confidence_score = sum(confidence_components)
        
        # Final decision logic
        if confidence_score >= 0.8:
            decision = "approve"
            rationale = "All validation criteria passed with high confidence"
        elif confidence_score >= 0.6:
            decision = "conditional" 
            rationale = "Most criteria passed but some concerns remain"
            conditions.extend([
                "Monitor performance closely in production",
                "Prepare rollback plan",
                "Gradual rollout recommended"
            ])
        else:
            decision = "reject"
            rationale = "Insufficient confidence for production promotion"
            next_steps.extend([
                "Review validation failures",
                "Adjust campaign parameters",
                "Consider additional trials"
            ])
        
        return decision, confidence_score, rationale, conditions, next_steps

if __name__ == "__main__":
    # Test validation pipeline
    import logging
    logging.basicConfig(level=logging.INFO)
    
    from campaign_manager import Trial, CampaignSpec, KnobSpace
    from priority_scoring import SliceCandidate
    
    # Create test trial
    trial = Trial(
        trial_id="test_trial_001",
        campaign_id="test_campaign",
        trial_number=1,
        parameters={
            "lambda_hybrid": 0.05,
            "K2_multiplier": 1.2,
            "r_dpp": 16
        },
        metrics={
            "delta_p5": 0.025,  # 2.5pp improvement
            "delta_p5_std": 0.008,
            "latency_p95_delta": 0.8,  # Small latency increase
            "latency_p95_delta_std": 0.3,
            "kv_prefix_drop": 0.008,  # Small KV drop
            "ece_drift": 0.02,
            "p99_p95_ratio": 1.8,
            "memory_mb": 520,
            "ci_lower": 0.010,
            "ci_upper": 0.040
        }
    )
    
    # Create test campaign spec
    candidate = SliceCandidate(
        slice_name="test@15%", budget_tier=15, domain="test", complexity="medium",
        lethe_p5=0.70, competitor_p5=0.85, ci_width=0.02,
        sensitivity_k2=0.1, sensitivity_lambda=0.08, sensitivity_mu=0.05,
        sensitivity_r=0.06, sensitivity_tau=0.04,
        traffic_weight=1.0, tenant_weight=1.0,
        kv_prefix_drop_risk=0.02, ece_drift_risk=0.01, 
        latency_inflation_risk=0.03, complexity_risk=0.1,
        sample_size=200, last_updated="2025-01-15"
    )
    
    spec = CampaignSpec(
        name="Test Campaign",
        slice_candidate=candidate,
        knob_spaces=[],
        gates={
            "min_delta_p5": 0.015,
            "max_latency_p95_delta": 1.0,
            "max_kv_drop": 0.01
        },
        validator_fences={
            "max_ece_drift": 0.08,
            "max_p99_p95_ratio": 2.5
        }
    )
    
    # Test validation
    validator = CampaignValidator()
    gate_results = validator.validate_trial(trial, spec)
    
    print("=== Gate Validation Results ===")
    for result in gate_results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status}: {result.description}")
        print(f"  Value: {result.value:.4f}, Threshold: {result.threshold}")
    
    # Test guardrails
    guardrails = Guardrails()
    violations = guardrails.check_all_guardrails(trial, spec)
    
    print(f"\n=== Guardrail Check Results ===")
    if violations:
        for violation in violations:
            print(f"VIOLATION: {violation.guardrail_name}")
            print(f"  Type: {violation.violation_type}")
            print(f"  Severity: {violation.severity}")
            print(f"  Value: {violation.current_value:.4f}, Limit: {violation.limit_value:.4f}")
            print(f"  Recommendation: {violation.recommendation}")
    else:
        print("No guardrail violations detected")
    
    print(f"\nValidation Summary:")
    print(f"  Gates passed: {all(r.passed for r in gate_results)}")
    print(f"  Critical violations: {len([v for v in violations if v.severity == 'high'])}")
    print(f"  Total violations: {len(violations)}")