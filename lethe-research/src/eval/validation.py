#!/usr/bin/env python3
"""
Task 3 Anti-Fraud Validation System
===================================

Comprehensive validation framework to prevent fraud indicators and ensure
baseline integrity in IR evaluation.

Features:
- Non-empty result validation with detailed diagnostics
- Smoke test framework with configurable thresholds  
- Statistical fraud detection (identical scores, suspicious distributions)
- Budget parity enforcement and monitoring
- Result provenance tracking and audit trails

Key Anti-Fraud Checks:
1. Non-empty results: All baselines must return non-empty retrieved_docs
2. Valid document IDs: No empty, null, or malformed document identifiers
3. Score validity: No NaN, Inf, or suspicious score patterns
4. Statistical plausibility: Score distributions must be reasonable
5. Budget compliance: Compute costs within ±5% parity
6. Reproducibility: Consistent results across runs with same parameters
"""

import numpy as np
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path
import hashlib

from .baselines import BaselineResult, EvaluationQuery, RetrievalResult

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation check"""
    check_name: str
    passed: bool
    details: Dict[str, Any]
    severity: str  # 'ERROR', 'WARNING', 'INFO'
    message: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class FraudReport:
    """Comprehensive fraud detection report"""
    baseline_name: str
    total_queries: int
    validation_results: List[ValidationResult]
    overall_status: str  # 'PASSED', 'FAILED', 'WARNING'
    fraud_indicators: List[str]
    confidence_score: float  # 0.0 = definite fraud, 1.0 = definitely clean
    recommendations: List[str]

class ValidationCheck:
    """Base class for validation checks"""
    
    def __init__(self, name: str, severity: str = 'ERROR'):
        self.name = name
        self.severity = severity
        
    def validate(self, 
                baseline_name: str,
                query: EvaluationQuery,
                result: BaselineResult) -> ValidationResult:
        """Override in subclasses"""
        raise NotImplementedError

class NonEmptyResultsCheck(ValidationCheck):
    """Validate that baseline returns non-empty results"""
    
    def __init__(self):
        super().__init__("NonEmptyResults", "ERROR")
        
    def validate(self, 
                baseline_name: str,
                query: EvaluationQuery, 
                result: BaselineResult) -> ValidationResult:
        
        details = {
            "query_id": query.query_id,
            "retrieved_count": len(result.retrieved_docs),
            "candidate_count": result.candidate_count
        }
        
        if not result.retrieved_docs:
            return ValidationResult(
                check_name=self.name,
                passed=False,
                details=details,
                severity=self.severity,
                message=f"Baseline {baseline_name} returned empty results for query {query.query_id}"
            )
            
        return ValidationResult(
            check_name=self.name,
            passed=True,
            details=details,
            severity="INFO",
            message=f"Baseline {baseline_name} returned {len(result.retrieved_docs)} results"
        )

class ValidDocumentIDsCheck(ValidationCheck):
    """Validate document IDs are valid and non-empty"""
    
    def __init__(self):
        super().__init__("ValidDocumentIDs", "ERROR")
        
    def validate(self,
                baseline_name: str,
                query: EvaluationQuery,
                result: BaselineResult) -> ValidationResult:
        
        invalid_doc_ids = []
        
        for i, doc_id in enumerate(result.retrieved_docs):
            if not doc_id or not isinstance(doc_id, str) or doc_id.strip() == "":
                invalid_doc_ids.append(f"Position {i}: '{doc_id}'")
                
        details = {
            "query_id": query.query_id,
            "total_docs": len(result.retrieved_docs),
            "invalid_count": len(invalid_doc_ids),
            "invalid_docs": invalid_doc_ids[:5]  # Show first 5
        }
        
        if invalid_doc_ids:
            return ValidationResult(
                check_name=self.name,
                passed=False,
                details=details,
                severity=self.severity,
                message=f"Baseline {baseline_name} has {len(invalid_doc_ids)} invalid document IDs"
            )
            
        return ValidationResult(
            check_name=self.name,
            passed=True,
            details=details,
            severity="INFO",
            message="All document IDs are valid"
        )

class ValidScoresCheck(ValidationCheck):
    """Validate relevance scores are valid numbers"""
    
    def __init__(self):
        super().__init__("ValidScores", "ERROR")
        
    def validate(self,
                baseline_name: str,
                query: EvaluationQuery,
                result: BaselineResult) -> ValidationResult:
        
        scores = result.relevance_scores
        
        # Check for NaN/Inf
        nan_count = sum(1 for s in scores if np.isnan(s))
        inf_count = sum(1 for s in scores if np.isinf(s))
        
        # Check for reasonable range (scores should typically be positive)
        negative_count = sum(1 for s in scores if s < 0)
        zero_count = sum(1 for s in scores if s == 0.0)
        
        details = {
            "query_id": query.query_id,
            "total_scores": len(scores),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "negative_count": negative_count,
            "zero_count": zero_count,
            "score_range": [float(min(scores)), float(max(scores))] if scores else [0, 0],
            "unique_scores": len(set(scores))
        }
        
        issues = []
        if nan_count > 0:
            issues.append(f"{nan_count} NaN scores")
        if inf_count > 0:
            issues.append(f"{inf_count} Inf scores")
            
        if issues:
            return ValidationResult(
                check_name=self.name,
                passed=False,
                details=details,
                severity=self.severity,
                message=f"Score validation failed: {', '.join(issues)}"
            )
            
        return ValidationResult(
            check_name=self.name,
            passed=True,
            details=details,
            severity="INFO",
            message="All scores are valid numbers"
        )

class ScoreDistributionCheck(ValidationCheck):
    """Check for suspicious score distributions (fraud indicator)"""
    
    def __init__(self):
        super().__init__("ScoreDistribution", "WARNING")
        
    def validate(self,
                baseline_name: str, 
                query: EvaluationQuery,
                result: BaselineResult) -> ValidationResult:
        
        scores = result.relevance_scores
        
        if len(scores) < 2:
            return ValidationResult(
                check_name=self.name,
                passed=True,
                details={"message": "Too few scores to analyze"},
                severity="INFO",
                message="Insufficient scores for distribution analysis"
            )
            
        unique_scores = len(set(scores))
        total_scores = len(scores)
        uniqueness_ratio = unique_scores / total_scores
        
        # Check for identical scores (suspicious)
        identical_scores = unique_scores == 1
        
        # Check for very low variance (suspicious)
        score_std = np.std(scores)
        score_mean = np.mean(scores)
        coefficient_of_variation = score_std / abs(score_mean) if abs(score_mean) > 1e-10 else 0
        
        details = {
            "query_id": query.query_id,
            "unique_scores": unique_scores,
            "total_scores": total_scores,
            "uniqueness_ratio": uniqueness_ratio,
            "score_std": score_std,
            "coefficient_of_variation": coefficient_of_variation,
            "identical_scores": identical_scores
        }
        
        # Suspicious patterns
        warnings = []
        if identical_scores and total_scores > 1:
            warnings.append("All scores are identical")
        if uniqueness_ratio < 0.3 and total_scores > 10:
            warnings.append(f"Low score diversity ({uniqueness_ratio:.1%})")
        if coefficient_of_variation < 0.01 and score_mean > 0:
            warnings.append("Suspiciously low score variance")
            
        if warnings:
            return ValidationResult(
                check_name=self.name,
                passed=False,
                details=details,
                severity=self.severity,
                message=f"Suspicious score distribution: {', '.join(warnings)}"
            )
            
        return ValidationResult(
            check_name=self.name,
            passed=True,
            details=details,
            severity="INFO",
            message="Score distribution appears normal"
        )

class BudgetParityCheck(ValidationCheck):
    """Check compute budget parity against baseline"""
    
    def __init__(self, baseline_budget: Optional[float] = None, tolerance: float = 0.05):
        super().__init__("BudgetParity", "WARNING")
        self.baseline_budget = baseline_budget
        self.tolerance = tolerance
        
    def validate(self,
                baseline_name: str,
                query: EvaluationQuery,
                result: BaselineResult) -> ValidationResult:
        
        if self.baseline_budget is None:
            return ValidationResult(
                check_name=self.name,
                passed=True,
                details={"message": "No baseline budget set"},
                severity="INFO",
                message="Budget parity check skipped - no baseline"
            )
            
        flops = result.flops_estimate
        deviation = abs(flops - self.baseline_budget) / self.baseline_budget if self.baseline_budget > 0 else 0
        
        details = {
            "query_id": query.query_id,
            "baseline_budget": self.baseline_budget,
            "method_flops": flops,
            "deviation": deviation,
            "tolerance": self.tolerance,
            "within_budget": deviation <= self.tolerance
        }
        
        if deviation > self.tolerance:
            return ValidationResult(
                check_name=self.name,
                passed=False,
                details=details,
                severity=self.severity,
                message=f"Budget parity violation: {deviation:.1%} deviation (limit: {self.tolerance:.1%})"
            )
            
        return ValidationResult(
            check_name=self.name,
            passed=True,
            details=details,
            severity="INFO",
            message=f"Budget compliant: {deviation:.1%} deviation"
        )

class ReproducibilityCheck(ValidationCheck):
    """Check for consistent results across runs"""
    
    def __init__(self):
        super().__init__("Reproducibility", "WARNING")
        self.previous_results: Dict[str, Dict[str, Any]] = {}
        
    def validate(self,
                baseline_name: str,
                query: EvaluationQuery, 
                result: BaselineResult) -> ValidationResult:
        
        # Create result fingerprint
        result_key = f"{baseline_name}_{query.query_id}"
        current_fingerprint = {
            "doc_ids_hash": hashlib.md5(
                json.dumps(result.retrieved_docs, sort_keys=True).encode()
            ).hexdigest(),
            "scores_hash": hashlib.md5(
                json.dumps([round(s, 6) for s in result.relevance_scores], sort_keys=True).encode()
            ).hexdigest(),
            "timestamp": result.timestamp
        }
        
        details = {
            "query_id": query.query_id,
            "current_run": current_fingerprint,
            "has_previous_run": result_key in self.previous_results
        }
        
        if result_key not in self.previous_results:
            self.previous_results[result_key] = current_fingerprint
            return ValidationResult(
                check_name=self.name,
                passed=True,
                details=details,
                severity="INFO",
                message="First run - baseline established"
            )
        
        previous = self.previous_results[result_key]
        docs_match = current_fingerprint["doc_ids_hash"] == previous["doc_ids_hash"]
        scores_match = current_fingerprint["scores_hash"] == previous["scores_hash"]
        
        details.update({
            "previous_run": previous,
            "docs_match": docs_match,
            "scores_match": scores_match,
            "fully_reproducible": docs_match and scores_match
        })
        
        if not (docs_match and scores_match):
            return ValidationResult(
                check_name=self.name,
                passed=False,
                details=details,
                severity=self.severity,
                message="Results differ from previous run - reproducibility concern"
            )
            
        return ValidationResult(
            check_name=self.name,
            passed=True,
            details=details,
            severity="INFO",
            message="Results consistent with previous run"
        )

class ComprehensiveValidator:
    """Main validation orchestrator with all checks"""
    
    def __init__(self, baseline_budget: Optional[float] = None):
        self.checks = [
            NonEmptyResultsCheck(),
            ValidDocumentIDsCheck(),
            ValidScoresCheck(),
            ScoreDistributionCheck(),
            BudgetParityCheck(baseline_budget),
            ReproducibilityCheck()
        ]
        
        self.validation_log: List[ValidationResult] = []
        self.fraud_indicators: Dict[str, List[str]] = defaultdict(list)
        
    def validate_result(self,
                       baseline_name: str,
                       query: EvaluationQuery,
                       result: BaselineResult) -> List[ValidationResult]:
        """Run all validation checks on a single result"""
        
        validations = []
        
        for check in self.checks:
            try:
                validation = check.validate(baseline_name, query, result)
                validations.append(validation)
                self.validation_log.append(validation)
                
                # Track fraud indicators
                if not validation.passed and validation.severity == 'ERROR':
                    self.fraud_indicators[baseline_name].append(validation.message)
                    
            except Exception as e:
                logger.error(f"Validation check {check.name} failed: {e}")
                # Create error validation result
                error_validation = ValidationResult(
                    check_name=check.name,
                    passed=False,
                    details={"error": str(e)},
                    severity="ERROR",
                    message=f"Validation check failed: {e}"
                )
                validations.append(error_validation)
                self.validation_log.append(error_validation)
                
        return validations
        
    def generate_fraud_report(self, baseline_name: str) -> FraudReport:
        """Generate comprehensive fraud report for a baseline"""
        
        baseline_validations = [
            v for v in self.validation_log 
            if v.details.get('query_id') and baseline_name in str(v.message)
        ]
        
        if not baseline_validations:
            return FraudReport(
                baseline_name=baseline_name,
                total_queries=0,
                validation_results=[],
                overall_status="UNKNOWN",
                fraud_indicators=[],
                confidence_score=0.5,
                recommendations=["No validation data available"]
            )
            
        # Analyze validation results
        total_queries = len(set(v.details.get('query_id', '') for v in baseline_validations))
        
        error_count = sum(1 for v in baseline_validations if not v.passed and v.severity == 'ERROR')
        warning_count = sum(1 for v in baseline_validations if not v.passed and v.severity == 'WARNING')
        
        # Determine overall status
        if error_count > 0:
            overall_status = "FAILED"
        elif warning_count > total_queries * 0.2:  # More than 20% warnings
            overall_status = "WARNING"
        else:
            overall_status = "PASSED"
            
        # Calculate confidence score
        total_checks = len(baseline_validations)
        passed_checks = sum(1 for v in baseline_validations if v.passed)
        confidence_score = passed_checks / total_checks if total_checks > 0 else 0.0
        
        # Fraud indicators
        fraud_indicators = self.fraud_indicators.get(baseline_name, [])
        
        # Generate recommendations
        recommendations = []
        if error_count > 0:
            recommendations.append("Critical errors found - baseline may be broken")
        if warning_count > 0:
            recommendations.append("Suspicious patterns detected - manual review recommended")
        if confidence_score < 0.8:
            recommendations.append("Low confidence score - thorough validation needed")
        if not recommendations:
            recommendations.append("Baseline appears clean - no major issues detected")
            
        return FraudReport(
            baseline_name=baseline_name,
            total_queries=total_queries,
            validation_results=baseline_validations,
            overall_status=overall_status,
            fraud_indicators=fraud_indicators,
            confidence_score=confidence_score,
            recommendations=recommendations
        )
        
    def get_summary_report(self) -> Dict[str, Any]:
        """Get summary validation report across all baselines"""
        
        baseline_names = set()
        for validation in self.validation_log:
            # Extract baseline name from message (simple heuristic)
            for word in validation.message.split():
                if 'baseline' in validation.message.lower():
                    continue
                baseline_names.add(word)
                break
                
        fraud_reports = {}
        for baseline_name in baseline_names:
            fraud_reports[baseline_name] = self.generate_fraud_report(baseline_name)
            
        # Overall summary
        total_validations = len(self.validation_log)
        passed_validations = sum(1 for v in self.validation_log if v.passed)
        
        failed_baselines = [
            name for name, report in fraud_reports.items()
            if report.overall_status == "FAILED"
        ]
        
        return {
            "validation_summary": {
                "total_validations": total_validations,
                "passed_validations": passed_validations,
                "success_rate": passed_validations / total_validations if total_validations > 0 else 0,
                "baselines_analyzed": len(fraud_reports),
                "failed_baselines": failed_baselines
            },
            "fraud_reports": {name: report for name, report in fraud_reports.items()},
            "recommendations": self._generate_overall_recommendations(fraud_reports)
        }
        
    def _generate_overall_recommendations(self, fraud_reports: Dict[str, FraudReport]) -> List[str]:
        """Generate overall recommendations based on all fraud reports"""
        
        recommendations = []
        
        failed_count = sum(1 for r in fraud_reports.values() if r.overall_status == "FAILED")
        warning_count = sum(1 for r in fraud_reports.values() if r.overall_status == "WARNING")
        
        if failed_count > 0:
            recommendations.append(f"{failed_count} baseline(s) failed validation - immediate investigation required")
            
        if warning_count > 0:
            recommendations.append(f"{warning_count} baseline(s) have warnings - review recommended")
            
        avg_confidence = np.mean([r.confidence_score for r in fraud_reports.values()])
        if avg_confidence < 0.7:
            recommendations.append(f"Low average confidence ({avg_confidence:.2f}) - comprehensive review needed")
            
        if not recommendations:
            recommendations.append("All baselines passed validation - evaluation appears clean")
            
        return recommendations

def create_smoke_test_queries() -> List[EvaluationQuery]:
    """Create standard smoke test queries for baseline validation"""
    
    queries = [
        EvaluationQuery(
            query_id="smoke_001",
            text="machine learning algorithms",
            domain="general",
            complexity="simple"
        ),
        EvaluationQuery(
            query_id="smoke_002", 
            text="how to implement neural networks in python",
            domain="technical",
            complexity="medium"
        ),
        EvaluationQuery(
            query_id="smoke_003",
            text="best practices for software architecture design patterns",
            domain="technical",
            complexity="complex"
        ),
        EvaluationQuery(
            query_id="smoke_004",
            text="covid vaccine effectiveness",
            domain="medical",
            complexity="medium"
        ),
        EvaluationQuery(
            query_id="smoke_005",
            text="climate change impact on agriculture",
            domain="scientific", 
            complexity="complex"
        )
    ]
    
    return queries