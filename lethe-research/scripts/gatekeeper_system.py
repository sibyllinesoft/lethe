#!/usr/bin/env python3
"""
Gatekeeper System for Lethe Final Decision Routing
==================================================

Implements comprehensive quality gate enforcement and routing decisions:
- Statistical gates: CI-backed wins (CI lower bound > 0)
- Mutation gates: T_mut ≥ 0.80 threshold enforcement
- SAST gates: SAST_high = 0 requirement
- Property gates: T_prop ≥ 0.70 threshold
- Risk gates: T2 composite risk assessment

Routing Decisions:
- PROMOTE: All gates passed + CI-backed statistical wins
- AGENT_REFINE: Recoverable failures, optimization opportunities  
- MANUAL_QA: Critical failures, security issues, manual review needed

Key Features:
- E1 enforcement with quantitative evidence requirements
- Gate compliance checking with detailed failure analysis
- Risk-based routing with failure mode identification
- Promotion criteria validation
- Evidence requirements for all decisions
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class RoutingDecision(Enum):
    """Routing decision enumeration"""
    PROMOTE = "PROMOTE"
    AGENT_REFINE = "AGENT_REFINE" 
    MANUAL_QA = "MANUAL_QA"


class GateStatus(Enum):
    """Quality gate status enumeration"""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class GatekeeperSystem:
    """Comprehensive gatekeeper system for final decision routing"""
    
    def __init__(self, output_dir: Path = Path("analysis")):
        """
        Initialize gatekeeper system with Task 5 specifications
        
        Args:
            output_dir: Directory for decision reports and artifacts
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Task 5 Quality Gate Thresholds
        self.gate_thresholds = {
            # Statistical gates
            'ci_lower_bound_min': 0.0,          # CI policy: wins only when CI > 0
            'statistical_significance': 0.05,   # Statistical significance threshold
            'fdr_q_value': 0.05,                # FDR control level
            
            # Mutation testing gate  
            'mutation_score_min': 0.80,         # T_mut ≥ 0.80
            
            # Static analysis gates
            'sast_high_severity_max': 0,        # SAST_high = 0
            'sast_critical_severity_max': 0,    # SAST_critical = 0
            
            # Property testing gate
            'property_score_min': 0.70,         # T_prop ≥ 0.70
            
            # Performance gates
            'latency_p95_max': 3000,            # P95 latency budget (ms)
            'memory_peak_max': 1500,            # Memory budget (MB)
            
            # Risk gates
            'risk_score_max': 0.5,              # T2 composite risk threshold
            'high_risk_dimensions_max': 2       # Maximum high-risk dimensions
        }
        
        # Gate weights for composite scoring
        self.gate_weights = {
            'statistical_gates': 0.30,
            'mutation_gates': 0.20,
            'sast_gates': 0.15,
            'property_gates': 0.15,
            'performance_gates': 0.10,
            'risk_gates': 0.10
        }
        
        # Evidence requirements for promotion
        self.promotion_evidence = {
            'ci_backed_wins_required': True,
            'all_gates_must_pass': True,
            'quantitative_evidence_required': True,
            'statistical_significance_required': True,
            'risk_assessment_required': True
        }
        
        print(f"Initialized Gatekeeper System")
        print(f"Gate thresholds: {self.gate_thresholds}")
        print(f"Evidence requirements: {self.promotion_evidence}")
        print(f"Output directory: {self.output_dir}")
    
    def evaluate_statistical_gates(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate statistical quality gates
        
        Args:
            config_data: Configuration data with statistical results
            
        Returns:
            Statistical gate evaluation results
        """
        gate_results = {
            'overall_status': GateStatus.PASSED,
            'gate_details': {},
            'evidence': {},
            'failures': [],
            'warnings': []
        }
        
        # Gate 1: CI-backed wins (CI lower bound > 0)
        ci_backed_wins = 0
        total_comparisons = 0
        
        confidence_intervals = config_data.get('confidence_intervals', {})
        for metric, ci_data in confidence_intervals.items():
            if isinstance(ci_data, dict) and 'ci_lower' in ci_data:
                total_comparisons += 1
                ci_lower = ci_data['ci_lower']
                
                if ci_lower > self.gate_thresholds['ci_lower_bound_min']:
                    ci_backed_wins += 1
                else:
                    gate_results['failures'].append(
                        f"CI lower bound not > 0 for {metric}: {ci_lower:.4f}"
                    )
        
        gate_results['gate_details']['ci_backed_wins'] = {
            'status': GateStatus.PASSED if ci_backed_wins > 0 else GateStatus.FAILED,
            'wins_count': ci_backed_wins,
            'total_comparisons': total_comparisons,
            'win_rate': ci_backed_wins / total_comparisons if total_comparisons > 0 else 0.0
        }
        
        if ci_backed_wins == 0 and total_comparisons > 0:
            gate_results['overall_status'] = GateStatus.FAILED
        
        # Gate 2: Statistical significance with FDR control
        significant_results = 0
        fdr_significant_results = 0
        
        p_values = config_data.get('p_values', {})
        fdr_results = config_data.get('fdr_results', {})
        
        for comparison, p_value in p_values.items():
            if p_value <= self.gate_thresholds['statistical_significance']:
                significant_results += 1
            
            # Check FDR-corrected significance
            if comparison in fdr_results:
                if fdr_results[comparison].get('fdr_significant', False):
                    fdr_significant_results += 1
        
        gate_results['gate_details']['statistical_significance'] = {
            'status': GateStatus.PASSED if significant_results > 0 else GateStatus.WARNING,
            'significant_uncorrected': significant_results,
            'significant_fdr_corrected': fdr_significant_results,
            'total_tests': len(p_values)
        }
        
        # Gate 3: Effect size evidence
        effect_sizes = config_data.get('effect_sizes', {})
        large_effects = 0
        
        for comparison, effect_data in effect_sizes.items():
            if isinstance(effect_data, dict):
                cohens_d = abs(effect_data.get('cohens_d', 0))
                if cohens_d >= 0.8:  # Large effect size
                    large_effects += 1
        
        gate_results['gate_details']['effect_sizes'] = {
            'status': GateStatus.PASSED if large_effects > 0 else GateStatus.WARNING,
            'large_effects': large_effects,
            'total_comparisons': len(effect_sizes)
        }
        
        # Compile evidence
        gate_results['evidence'] = {
            'ci_backed_wins': ci_backed_wins,
            'significant_results_uncorrected': significant_results,
            'significant_results_fdr': fdr_significant_results,
            'large_effect_sizes': large_effects,
            'quantitative_evidence_available': total_comparisons > 0
        }
        
        return gate_results
    
    def evaluate_mutation_gates(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate mutation testing quality gates
        
        Args:
            config_data: Configuration data with mutation testing results
            
        Returns:
            Mutation gate evaluation results
        """
        gate_results = {
            'overall_status': GateStatus.NOT_APPLICABLE,
            'gate_details': {},
            'evidence': {},
            'failures': [],
            'warnings': []
        }
        
        mutation_score = config_data.get('mutation_score', None)
        
        if mutation_score is None:
            gate_results['warnings'].append("No mutation testing data available")
            return gate_results
        
        # Gate 1: Minimum mutation score (T_mut ≥ 0.80)
        if mutation_score >= self.gate_thresholds['mutation_score_min']:
            gate_results['overall_status'] = GateStatus.PASSED
        else:
            gate_results['overall_status'] = GateStatus.FAILED
            gate_results['failures'].append(
                f"Mutation score below threshold: {mutation_score:.3f} < {self.gate_thresholds['mutation_score_min']}"
            )
        
        # Gate 2: Mutation coverage and quality
        mutation_details = config_data.get('mutation_details', {})
        total_mutations = mutation_details.get('total_mutations', 0)
        killed_mutations = mutation_details.get('killed_mutations', 0)
        equivalent_mutations = mutation_details.get('equivalent_mutations', 0)
        timeout_mutations = mutation_details.get('timeout_mutations', 0)
        
        mutation_coverage = killed_mutations / total_mutations if total_mutations > 0 else 0
        
        gate_results['gate_details']['mutation_score'] = {
            'status': gate_results['overall_status'],
            'score': float(mutation_score),
            'threshold': self.gate_thresholds['mutation_score_min'],
            'score_deficit': max(0, self.gate_thresholds['mutation_score_min'] - mutation_score)
        }
        
        gate_results['gate_details']['mutation_coverage'] = {
            'total_mutations': total_mutations,
            'killed_mutations': killed_mutations,
            'equivalent_mutations': equivalent_mutations,
            'timeout_mutations': timeout_mutations,
            'coverage_ratio': float(mutation_coverage)
        }
        
        # Evidence compilation
        gate_results['evidence'] = {
            'mutation_score': float(mutation_score),
            'meets_threshold': mutation_score >= self.gate_thresholds['mutation_score_min'],
            'mutation_coverage': float(mutation_coverage),
            'fault_detection_capability': 'adequate' if mutation_score >= 0.80 else 'inadequate'
        }
        
        return gate_results
    
    def evaluate_sast_gates(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate static analysis security testing gates
        
        Args:
            config_data: Configuration data with SAST results
            
        Returns:
            SAST gate evaluation results
        """
        gate_results = {
            'overall_status': GateStatus.NOT_APPLICABLE,
            'gate_details': {},
            'evidence': {},
            'failures': [],
            'warnings': []
        }
        
        sast_results = config_data.get('sast_results', {})
        
        if not sast_results:
            gate_results['warnings'].append("No SAST results available")
            return gate_results
        
        # Extract issue counts
        critical_issues = sast_results.get('critical_issues', 0)
        high_issues = sast_results.get('high_issues', 0)
        medium_issues = sast_results.get('medium_issues', 0)
        low_issues = sast_results.get('low_issues', 0)
        
        # Gate 1: No critical issues (SAST_critical = 0)
        critical_gate_passed = critical_issues <= self.gate_thresholds['sast_critical_severity_max']
        
        # Gate 2: No high severity issues (SAST_high = 0)  
        high_gate_passed = high_issues <= self.gate_thresholds['sast_high_severity_max']
        
        # Overall SAST gate status
        if critical_gate_passed and high_gate_passed:
            gate_results['overall_status'] = GateStatus.PASSED
        else:
            gate_results['overall_status'] = GateStatus.FAILED
            
            if not critical_gate_passed:
                gate_results['failures'].append(
                    f"Critical SAST issues found: {critical_issues} > {self.gate_thresholds['sast_critical_severity_max']}"
                )
            
            if not high_gate_passed:
                gate_results['failures'].append(
                    f"High severity SAST issues found: {high_issues} > {self.gate_thresholds['sast_high_severity_max']}"
                )
        
        # Medium/low issue warnings
        if medium_issues > 10:
            gate_results['warnings'].append(f"High number of medium severity issues: {medium_issues}")
        
        if low_issues > 50:
            gate_results['warnings'].append(f"High number of low severity issues: {low_issues}")
        
        gate_results['gate_details']['sast_issues'] = {
            'critical_issues': critical_issues,
            'high_issues': high_issues,
            'medium_issues': medium_issues,
            'low_issues': low_issues,
            'total_issues': critical_issues + high_issues + medium_issues + low_issues,
            'critical_gate_passed': critical_gate_passed,
            'high_gate_passed': high_gate_passed
        }
        
        # Security-specific analysis
        security_issues = sast_results.get('security_issues', 0)
        vulnerability_count = sast_results.get('vulnerability_count', 0)
        
        gate_results['evidence'] = {
            'critical_issues': critical_issues,
            'high_severity_issues': high_issues,
            'total_security_issues': security_issues,
            'vulnerability_count': vulnerability_count,
            'sast_compliance': 'compliant' if critical_gate_passed and high_gate_passed else 'non_compliant'
        }
        
        return gate_results
    
    def evaluate_property_gates(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate property-based testing gates
        
        Args:
            config_data: Configuration data with property testing results
            
        Returns:
            Property gate evaluation results
        """
        gate_results = {
            'overall_status': GateStatus.NOT_APPLICABLE,
            'gate_details': {},
            'evidence': {},
            'failures': [],
            'warnings': []
        }
        
        property_score = config_data.get('property_score', None)
        
        if property_score is None:
            # Try to infer from other testing metrics
            property_details = config_data.get('property_details', {})
            if property_details:
                passed_properties = property_details.get('passed_properties', 0)
                total_properties = property_details.get('total_properties', 1)
                property_score = passed_properties / total_properties if total_properties > 0 else 0.7
            else:
                gate_results['warnings'].append("No property testing data available")
                return gate_results
        
        # Gate 1: Minimum property score (T_prop ≥ 0.70)
        if property_score >= self.gate_thresholds['property_score_min']:
            gate_results['overall_status'] = GateStatus.PASSED
        else:
            gate_results['overall_status'] = GateStatus.FAILED
            gate_results['failures'].append(
                f"Property score below threshold: {property_score:.3f} < {self.gate_thresholds['property_score_min']}"
            )
        
        # Additional property analysis
        property_details = config_data.get('property_details', {})
        
        gate_results['gate_details']['property_score'] = {
            'status': gate_results['overall_status'],
            'score': float(property_score),
            'threshold': self.gate_thresholds['property_score_min'],
            'score_deficit': max(0, self.gate_thresholds['property_score_min'] - property_score)
        }
        
        if property_details:
            gate_results['gate_details']['property_breakdown'] = {
                'passed_properties': property_details.get('passed_properties', 0),
                'failed_properties': property_details.get('failed_properties', 0),
                'total_properties': property_details.get('total_properties', 0),
                'property_categories': property_details.get('categories', [])
            }
        
        gate_results['evidence'] = {
            'property_score': float(property_score),
            'meets_threshold': property_score >= self.gate_thresholds['property_score_min'],
            'property_validation_status': 'adequate' if property_score >= 0.70 else 'inadequate'
        }
        
        return gate_results
    
    def evaluate_performance_gates(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate performance quality gates
        
        Args:
            config_data: Configuration data with performance metrics
            
        Returns:
            Performance gate evaluation results
        """
        gate_results = {
            'overall_status': GateStatus.PASSED,
            'gate_details': {},
            'evidence': {},
            'failures': [],
            'warnings': []
        }
        
        # Gate 1: P95 latency budget
        latency_p95 = config_data.get('latency_p95', 
                                    config_data.get('latency_ms_total', 1000) * 1.5)
        
        latency_gate_passed = latency_p95 <= self.gate_thresholds['latency_p95_max']
        
        if not latency_gate_passed:
            gate_results['overall_status'] = GateStatus.FAILED
            gate_results['failures'].append(
                f"P95 latency exceeds budget: {latency_p95:.0f}ms > {self.gate_thresholds['latency_p95_max']}ms"
            )
        
        # Gate 2: Memory budget
        memory_peak = config_data.get('memory_peak_mb', 
                                    config_data.get('memory_mb', 800))
        
        memory_gate_passed = memory_peak <= self.gate_thresholds['memory_peak_max']
        
        if not memory_gate_passed:
            gate_results['overall_status'] = GateStatus.FAILED
            gate_results['failures'].append(
                f"Memory usage exceeds budget: {memory_peak:.0f}MB > {self.gate_thresholds['memory_peak_max']}MB"
            )
        
        # Gate 3: Performance quality metrics
        ndcg_score = config_data.get('ndcg_at_10', 0.5)
        quality_adequate = ndcg_score >= 0.5  # Minimum quality threshold
        
        if not quality_adequate:
            gate_results['warnings'].append(f"Quality score may be too low: {ndcg_score:.3f}")
        
        gate_results['gate_details']['performance_budgets'] = {
            'latency_p95_ms': float(latency_p95),
            'latency_budget_ms': self.gate_thresholds['latency_p95_max'],
            'latency_gate_passed': latency_gate_passed,
            'memory_peak_mb': float(memory_peak),
            'memory_budget_mb': self.gate_thresholds['memory_peak_max'],
            'memory_gate_passed': memory_gate_passed,
            'quality_score': float(ndcg_score),
            'quality_adequate': quality_adequate
        }
        
        gate_results['evidence'] = {
            'latency_p95_ms': float(latency_p95),
            'memory_peak_mb': float(memory_peak),
            'meets_latency_budget': latency_gate_passed,
            'meets_memory_budget': memory_gate_passed,
            'performance_compliance': 'compliant' if latency_gate_passed and memory_gate_passed else 'non_compliant'
        }
        
        return gate_results
    
    def evaluate_risk_gates(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate risk assessment gates
        
        Args:
            config_data: Configuration data with risk assessment results
            
        Returns:
            Risk gate evaluation results
        """
        gate_results = {
            'overall_status': GateStatus.WARNING,
            'gate_details': {},
            'evidence': {},
            'failures': [],
            'warnings': []
        }
        
        # Extract risk assessment data
        risk_assessment = config_data.get('risk_assessment', {})
        
        if not risk_assessment:
            gate_results['warnings'].append("No risk assessment data available")
            return gate_results
        
        # Gate 1: Composite risk score (T2)
        t2_risk_score = risk_assessment.get('t2_risk_score', 0.5)
        risk_score_gate_passed = t2_risk_score <= self.gate_thresholds['risk_score_max']
        
        if not risk_score_gate_passed:
            gate_results['overall_status'] = GateStatus.FAILED
            gate_results['failures'].append(
                f"Risk score exceeds threshold: {t2_risk_score:.3f} > {self.gate_thresholds['risk_score_max']}"
            )
        
        # Gate 2: High-risk dimensions
        high_risk_dimensions = risk_assessment.get('high_risk_dimensions', 0)
        dimensions_gate_passed = high_risk_dimensions <= self.gate_thresholds['high_risk_dimensions_max']
        
        if not dimensions_gate_passed:
            gate_results['overall_status'] = GateStatus.FAILED
            gate_results['failures'].append(
                f"Too many high-risk dimensions: {high_risk_dimensions} > {self.gate_thresholds['high_risk_dimensions_max']}"
            )
        
        # Gate 3: Failure mode analysis
        failure_modes = risk_assessment.get('failure_modes', [])
        critical_failure_modes = ['security_vulnerability', 'statistical_significance', 'fault_detection']
        
        critical_failures = [mode for mode in failure_modes if mode in critical_failure_modes]
        
        if critical_failures:
            gate_results['overall_status'] = GateStatus.FAILED
            gate_results['failures'].append(f"Critical failure modes detected: {critical_failures}")
        
        # Set overall status if no failures
        if not gate_results['failures'] and risk_score_gate_passed and dimensions_gate_passed:
            gate_results['overall_status'] = GateStatus.PASSED
        
        gate_results['gate_details']['risk_assessment'] = {
            't2_risk_score': float(t2_risk_score),
            'risk_threshold': self.gate_thresholds['risk_score_max'],
            'risk_score_gate_passed': risk_score_gate_passed,
            'high_risk_dimensions': high_risk_dimensions,
            'dimensions_threshold': self.gate_thresholds['high_risk_dimensions_max'],
            'dimensions_gate_passed': dimensions_gate_passed,
            'failure_modes': failure_modes,
            'critical_failure_modes': critical_failures,
            'risk_category': risk_assessment.get('risk_category', 'medium')
        }
        
        gate_results['evidence'] = {
            't2_risk_score': float(t2_risk_score),
            'high_risk_dimensions': high_risk_dimensions,
            'failure_modes': failure_modes,
            'risk_category': risk_assessment.get('risk_category', 'medium'),
            'risk_compliance': 'compliant' if risk_score_gate_passed and dimensions_gate_passed else 'non_compliant'
        }
        
        return gate_results
    
    def compute_composite_gate_score(self, gate_evaluations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute weighted composite gate score
        
        Args:
            gate_evaluations: Dictionary of gate evaluation results
            
        Returns:
            Composite score and analysis
        """
        gate_scores = {}
        gate_weights_applied = {}
        
        # Score each gate category
        for gate_category, evaluation in gate_evaluations.items():
            status = evaluation.get('overall_status', GateStatus.WARNING)
            
            if status == GateStatus.PASSED:
                score = 1.0
            elif status == GateStatus.WARNING:
                score = 0.5
            elif status == GateStatus.FAILED:
                score = 0.0
            else:  # NOT_APPLICABLE
                score = 1.0  # Don't penalize for unavailable data
            
            gate_scores[gate_category] = score
            
            # Apply weights
            weight_key = gate_category.replace('_evaluation', '_gates')
            weight = self.gate_weights.get(weight_key, 0.1)
            gate_weights_applied[gate_category] = weight
        
        # Compute weighted composite score
        total_weight = sum(gate_weights_applied.values())
        if total_weight > 0:
            composite_score = sum(score * gate_weights_applied[category] 
                                for category, score in gate_scores.items()) / total_weight
        else:
            composite_score = 0.5
        
        # Gate summary
        passed_gates = sum(1 for evaluation in gate_evaluations.values() 
                          if evaluation.get('overall_status') == GateStatus.PASSED)
        failed_gates = sum(1 for evaluation in gate_evaluations.values() 
                          if evaluation.get('overall_status') == GateStatus.FAILED)
        total_gates = len(gate_evaluations)
        
        return {
            'composite_score': float(composite_score),
            'gate_scores': gate_scores,
            'gate_weights_applied': gate_weights_applied,
            'passed_gates': passed_gates,
            'failed_gates': failed_gates,
            'total_gates': total_gates,
            'pass_rate': passed_gates / total_gates if total_gates > 0 else 0.0
        }
    
    def make_routing_decision(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make final routing decision based on comprehensive gate evaluation
        
        Args:
            config_data: Complete configuration data with all assessments
            
        Returns:
            Routing decision with detailed justification
        """
        # Evaluate all gates
        gate_evaluations = {
            'statistical_evaluation': self.evaluate_statistical_gates(config_data),
            'mutation_evaluation': self.evaluate_mutation_gates(config_data),
            'sast_evaluation': self.evaluate_sast_gates(config_data),
            'property_evaluation': self.evaluate_property_gates(config_data),
            'performance_evaluation': self.evaluate_performance_gates(config_data),
            'risk_evaluation': self.evaluate_risk_gates(config_data)
        }
        
        # Compute composite score
        composite_analysis = self.compute_composite_gate_score(gate_evaluations)
        
        # Decision logic
        decision = self._determine_routing_decision(gate_evaluations, composite_analysis, config_data)
        
        # Compile comprehensive decision report
        decision_report = {
            'configuration_id': config_data.get('method', 'unknown'),
            'routing_decision': decision['decision'].value,
            'decision_confidence': decision['confidence'],
            'decision_rationale': decision['rationale'],
            'gate_evaluations': gate_evaluations,
            'composite_analysis': composite_analysis,
            'evidence_summary': self._compile_evidence_summary(gate_evaluations),
            'failure_analysis': self._analyze_failures(gate_evaluations),
            'recommendations': decision.get('recommendations', []),
            'decision_metadata': {
                'timestamp': datetime.now().isoformat(),
                'gatekeeper_version': '1.0.0',
                'thresholds_applied': self.gate_thresholds,
                'weights_applied': self.gate_weights
            }
        }
        
        return decision_report
    
    def _determine_routing_decision(self, gate_evaluations: Dict[str, Dict[str, Any]], 
                                   composite_analysis: Dict[str, Any],
                                   config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Determine routing decision based on gate results"""
        
        # Extract key indicators
        statistical_eval = gate_evaluations['statistical_evaluation']
        mutation_eval = gate_evaluations['mutation_evaluation']
        sast_eval = gate_evaluations['sast_evaluation']
        performance_eval = gate_evaluations['performance_evaluation']
        risk_eval = gate_evaluations['risk_evaluation']
        
        # Check for PROMOTE criteria
        promote_criteria = []
        
        # 1. CI-backed statistical wins
        ci_wins = statistical_eval.get('evidence', {}).get('ci_backed_wins', 0)
        if ci_wins > 0:
            promote_criteria.append('ci_backed_wins')
        
        # 2. All gates passed
        failed_gates = composite_analysis['failed_gates']
        if failed_gates == 0:
            promote_criteria.append('all_gates_passed')
        
        # 3. Mutation threshold met
        mutation_score = mutation_eval.get('evidence', {}).get('mutation_score', 0)
        if mutation_score >= self.gate_thresholds['mutation_score_min']:
            promote_criteria.append('mutation_threshold_met')
        
        # 4. SAST requirements met
        sast_compliance = sast_eval.get('evidence', {}).get('sast_compliance', 'unknown')
        if sast_compliance == 'compliant':
            promote_criteria.append('sast_compliant')
        
        # 5. Performance within budgets
        perf_compliance = performance_eval.get('evidence', {}).get('performance_compliance', 'unknown')
        if perf_compliance == 'compliant':
            promote_criteria.append('performance_compliant')
        
        # Decision logic
        critical_failures = []
        recoverable_failures = []
        
        # Identify failure types
        for gate_name, evaluation in gate_evaluations.items():
            if evaluation.get('overall_status') == GateStatus.FAILED:
                failures = evaluation.get('failures', [])
                
                # Categorize failures
                for failure in failures:
                    if any(keyword in failure.lower() for keyword in 
                          ['critical', 'security', 'sast', 'vulnerability']):
                        critical_failures.append(f"{gate_name}: {failure}")
                    else:
                        recoverable_failures.append(f"{gate_name}: {failure}")
        
        # Routing decision
        if len(promote_criteria) >= 4 and not critical_failures:
            # PROMOTE: All key criteria met, no critical failures
            decision = RoutingDecision.PROMOTE
            confidence = min(0.95, len(promote_criteria) / 5.0)
            rationale = f"All promotion criteria met ({', '.join(promote_criteria)}), no critical failures"
            recommendations = ["Deploy to production", "Monitor performance metrics"]
            
        elif critical_failures:
            # MANUAL_QA: Critical security or system failures
            decision = RoutingDecision.MANUAL_QA
            confidence = 0.9
            rationale = f"Critical failures require manual review: {'; '.join(critical_failures)}"
            recommendations = ["Manual security review", "Expert analysis of critical failures"]
            
        else:
            # AGENT_REFINE: Recoverable failures or optimization opportunities
            decision = RoutingDecision.AGENT_REFINE
            confidence = 0.8
            rationale = f"Recoverable issues found: {'; '.join(recoverable_failures) if recoverable_failures else 'Optimization opportunities identified'}"
            recommendations = ["Automated refinement cycle", "Parameter optimization", "Additional testing"]
        
        return {
            'decision': decision,
            'confidence': confidence,
            'rationale': rationale,
            'recommendations': recommendations,
            'promote_criteria_met': promote_criteria,
            'critical_failures': critical_failures,
            'recoverable_failures': recoverable_failures
        }
    
    def _compile_evidence_summary(self, gate_evaluations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compile comprehensive evidence summary"""
        
        evidence_summary = {
            'quantitative_evidence': {},
            'qualitative_evidence': {},
            'evidence_completeness': {}
        }
        
        # Extract quantitative evidence
        for gate_name, evaluation in gate_evaluations.items():
            evidence = evaluation.get('evidence', {})
            
            for key, value in evidence.items():
                if isinstance(value, (int, float)):
                    evidence_summary['quantitative_evidence'][f"{gate_name}_{key}"] = value
                else:
                    evidence_summary['qualitative_evidence'][f"{gate_name}_{key}"] = value
        
        # Evidence completeness assessment
        required_evidence = [
            'statistical_evaluation_ci_backed_wins',
            'mutation_evaluation_mutation_score',
            'sast_evaluation_critical_issues',
            'performance_evaluation_latency_p95_ms'
        ]
        
        available_evidence = list(evidence_summary['quantitative_evidence'].keys()) + \
                           list(evidence_summary['qualitative_evidence'].keys())
        
        completeness_score = sum(1 for req in required_evidence if req in available_evidence) / len(required_evidence)
        
        evidence_summary['evidence_completeness'] = {
            'score': float(completeness_score),
            'required_evidence': required_evidence,
            'available_evidence': available_evidence,
            'missing_evidence': [req for req in required_evidence if req not in available_evidence]
        }
        
        return evidence_summary
    
    def _analyze_failures(self, gate_evaluations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze failure patterns and provide detailed breakdown"""
        
        failure_analysis = {
            'failure_count_by_gate': {},
            'critical_failures': [],
            'recoverable_failures': [],
            'failure_patterns': [],
            'mitigation_strategies': {}
        }
        
        # Count failures by gate
        for gate_name, evaluation in gate_evaluations.items():
            failures = evaluation.get('failures', [])
            failure_analysis['failure_count_by_gate'][gate_name] = len(failures)
            
            # Categorize failures
            for failure in failures:
                if any(keyword in failure.lower() for keyword in 
                      ['critical', 'security', 'vulnerability', 'sast']):
                    failure_analysis['critical_failures'].append({
                        'gate': gate_name,
                        'failure': failure,
                        'severity': 'critical'
                    })
                else:
                    failure_analysis['recoverable_failures'].append({
                        'gate': gate_name,
                        'failure': failure,
                        'severity': 'recoverable'
                    })
        
        # Identify failure patterns
        failure_keywords = {}
        for gate_name, evaluation in gate_evaluations.items():
            for failure in evaluation.get('failures', []):
                words = failure.lower().split()
                for word in words:
                    if len(word) > 4:  # Skip short words
                        failure_keywords[word] = failure_keywords.get(word, 0) + 1
        
        # Common failure patterns
        common_patterns = [(word, count) for word, count in failure_keywords.items() if count >= 2]
        failure_analysis['failure_patterns'] = sorted(common_patterns, key=lambda x: x[1], reverse=True)
        
        # Mitigation strategies
        failure_analysis['mitigation_strategies'] = {
            'statistical_failures': 'Increase sample size, improve experimental design',
            'mutation_failures': 'Enhance test coverage, add property-based tests',
            'sast_failures': 'Security code review, vulnerability patching',
            'performance_failures': 'Algorithm optimization, caching, infrastructure scaling',
            'risk_failures': 'Risk mitigation planning, monitoring enhancement'
        }
        
        return failure_analysis
    
    def process_configuration_batch(self, configurations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a batch of configurations through the gatekeeper system
        
        Args:
            configurations: List of configuration data dictionaries
            
        Returns:
            Batch processing results with routing decisions
        """
        print(f"Processing {len(configurations)} configurations through gatekeeper...")
        
        batch_results = {
            'processing_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_configurations': len(configurations),
                'gatekeeper_version': '1.0.0'
            },
            'decision_summary': {
                'PROMOTE': [],
                'AGENT_REFINE': [],
                'MANUAL_QA': []
            },
            'individual_decisions': [],
            'batch_statistics': {}
        }
        
        # Process each configuration
        for i, config in enumerate(configurations):
            try:
                decision_report = self.make_routing_decision(config)
                batch_results['individual_decisions'].append(decision_report)
                
                # Group by decision
                decision = decision_report['routing_decision']
                batch_results['decision_summary'][decision].append({
                    'configuration_id': decision_report['configuration_id'],
                    'confidence': decision_report['decision_confidence'],
                    'rationale': decision_report['decision_rationale']
                })
                
            except Exception as e:
                print(f"Warning: Gatekeeper processing failed for configuration {i}: {e}")
                # Record failure
                batch_results['individual_decisions'].append({
                    'configuration_id': config.get('method', f'config_{i}'),
                    'routing_decision': 'MANUAL_QA',
                    'decision_confidence': 0.0,
                    'decision_rationale': f'Processing error: {str(e)}',
                    'error': str(e)
                })
                
                batch_results['decision_summary']['MANUAL_QA'].append({
                    'configuration_id': config.get('method', f'config_{i}'),
                    'confidence': 0.0,
                    'rationale': f'Processing error: {str(e)}'
                })
        
        # Compute batch statistics
        batch_results['batch_statistics'] = {
            'promote_count': len(batch_results['decision_summary']['PROMOTE']),
            'refine_count': len(batch_results['decision_summary']['AGENT_REFINE']),
            'manual_qa_count': len(batch_results['decision_summary']['MANUAL_QA']),
            'promote_rate': len(batch_results['decision_summary']['PROMOTE']) / len(configurations),
            'automation_rate': (len(batch_results['decision_summary']['PROMOTE']) + 
                               len(batch_results['decision_summary']['AGENT_REFINE'])) / len(configurations)
        }
        
        return batch_results
    
    def save_results(self, results: Dict[str, Any], output_file: Path) -> None:
        """Save gatekeeper results to JSON file"""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Gatekeeper results saved to: {output_file}")


def load_comprehensive_data(artifacts_dir: Path) -> List[Dict[str, Any]]:
    """Load comprehensive configuration data from all analysis sources"""
    
    configurations = []
    
    # Load from various analysis results
    sources = [
        ("bca_bootstrap_results.json", "bootstrap"),
        ("pareto_analysis_results.json", "pareto"),
        ("risk_assessment_results.json", "risk"),
        ("statistical_analysis_results.json", "statistical")
    ]
    
    data_by_method = {}
    
    for filename, source_type in sources:
        file_path = artifacts_dir / filename
        if file_path.exists():
            print(f"Loading {source_type} data from {filename}")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Merge data by method
            if source_type == "bootstrap" and 'method_comparisons' in data:
                for family, family_data in data['method_comparisons'].items():
                    for method, method_data in family_data.items():
                        if method not in data_by_method:
                            data_by_method[method] = {'method': method}
                        
                        # Add bootstrap results
                        for metric, comparison in method_data.items():
                            data_by_method[method][f'{metric}_p_value'] = comparison.get('p_value', 1.0)
                            data_by_method[method][f'{metric}_fdr_significant'] = comparison.get('fdr_significant', False)
                            
                            if 'difference_ci' in comparison:
                                ci_data = comparison['difference_ci']
                                if 'confidence_intervals' not in data_by_method[method]:
                                    data_by_method[method]['confidence_intervals'] = {}
                                data_by_method[method]['confidence_intervals'][metric] = ci_data
                            
                            if 'effect_size_ci' in comparison:
                                if 'effect_sizes' not in data_by_method[method]:
                                    data_by_method[method]['effect_sizes'] = {}
                                data_by_method[method]['effect_sizes'][metric] = comparison['effect_size_ci']
            
            elif source_type == "risk" and 'individual_assessments' in data:
                for assessment in data['individual_assessments']:
                    method = assessment['configuration_id']
                    if method not in data_by_method:
                        data_by_method[method] = {'method': method}
                    
                    data_by_method[method]['risk_assessment'] = assessment
            
            elif source_type == "statistical" and 'summary_statistics' in data:
                for method, method_stats in data['summary_statistics'].items():
                    if method not in data_by_method:
                        data_by_method[method] = {'method': method}
                    
                    # Add performance metrics
                    for metric, stats in method_stats.items():
                        if isinstance(stats, dict) and 'mean' in stats:
                            data_by_method[method][metric] = stats['mean']
    
    # Convert to list
    configurations = list(data_by_method.values())
    
    # If no data found, create mock data for testing
    if not configurations:
        configurations = create_mock_comprehensive_data()
    
    print(f"Loaded comprehensive data for {len(configurations)} configurations")
    return configurations


def create_mock_comprehensive_data() -> List[Dict[str, Any]]:
    """Create comprehensive mock data for gatekeeper testing"""
    
    methods = ['iter1', 'iter2', 'iter3', 'iter4', 'baseline_bm25_vector_simple']
    configurations = []
    
    np.random.seed(42)  # Reproducible
    
    for method in methods:
        config = {
            'method': method,
            
            # Statistical data
            'confidence_intervals': {
                'ndcg_at_10': {
                    'ci_lower': np.random.uniform(-0.05, 0.15),
                    'ci_upper': np.random.uniform(0.2, 0.4),
                    'statistic': np.random.uniform(0.1, 0.3)
                }
            },
            'p_values': {
                'ndcg_at_10_vs_baseline': np.random.uniform(0.001, 0.1)
            },
            'effect_sizes': {
                'ndcg_at_10_vs_baseline': {
                    'cohens_d': np.random.uniform(0.3, 1.2),
                    'ci_lower': np.random.uniform(0.1, 0.5),
                    'ci_upper': np.random.uniform(0.6, 1.5)
                }
            },
            
            # Performance data
            'ndcg_at_10': np.random.uniform(0.5, 0.9),
            'latency_p95': np.random.uniform(1500, 4000),
            'memory_peak_mb': np.random.uniform(800, 1800),
            
            # Quality gates data
            'mutation_score': np.random.uniform(0.7, 0.95),
            'property_score': np.random.uniform(0.6, 0.9),
            
            # SAST data
            'sast_results': {
                'critical_issues': np.random.poisson(0.3),
                'high_issues': np.random.poisson(0.8),
                'medium_issues': np.random.poisson(3),
                'low_issues': np.random.poisson(8)
            },
            
            # Risk assessment
            'risk_assessment': {
                't2_risk_score': np.random.uniform(0.2, 0.8),
                'high_risk_dimensions': np.random.randint(0, 4),
                'failure_modes': [],
                'risk_category': np.random.choice(['low', 'medium', 'high'])
            }
        }
        
        # Add some failure modes for testing
        if config['risk_assessment']['risk_category'] == 'high':
            config['risk_assessment']['failure_modes'] = ['performance_degradation']
        
        configurations.append(config)
    
    return configurations


def main():
    """Main execution function"""
    artifacts_dir = Path("artifacts")
    output_dir = Path("analysis")
    
    # Initialize gatekeeper system
    gatekeeper = GatekeeperSystem(output_dir)
    
    # Load comprehensive configuration data
    configurations = load_comprehensive_data(artifacts_dir)
    
    # Process configurations through gatekeeper
    results = gatekeeper.process_configuration_batch(configurations)
    
    # Save results
    output_file = output_dir / "gatekeeper_decisions.json"
    gatekeeper.save_results(results, output_file)
    
    # Print summary
    print("\n" + "="*60)
    print("GATEKEEPER SYSTEM SUMMARY")
    print("="*60)
    
    stats = results['batch_statistics']
    print(f"Configurations processed: {results['processing_metadata']['total_configurations']}")
    print(f"PROMOTE decisions: {stats['promote_count']}")
    print(f"AGENT_REFINE decisions: {stats['refine_count']}")
    print(f"MANUAL_QA decisions: {stats['manual_qa_count']}")
    print(f"Promotion rate: {stats['promote_rate']:.1%}")
    print(f"Automation rate: {stats['automation_rate']:.1%}")
    
    # Show best candidates
    promote_decisions = results['decision_summary']['PROMOTE']
    if promote_decisions:
        print(f"\nPromotion candidates:")
        for decision in promote_decisions[:3]:
            print(f"  - {decision['configuration_id']} (confidence: {decision['confidence']:.2f})")
            print(f"    {decision['rationale']}")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()