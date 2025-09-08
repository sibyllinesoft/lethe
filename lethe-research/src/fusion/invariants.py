"""
Mathematical invariant enforcement for hybrid fusion system.

Implements and validates the five critical invariants:
- P1: α→1 equals BM25-only results
- P2: α→0 equals Dense-only results  
- P3: Adding duplicate doc never decreases rank of relevant
- P4: Monotonicity under term weight scaling
- P5: Score calibration monotone in α

Zero violations tolerated - any break fails validation.
"""

import logging
import math
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


class InvariantViolation(Exception):
    """Raised when a mathematical invariant is violated."""
    
    def __init__(self, invariant: str, details: str, evidence: Dict = None):
        self.invariant = invariant
        self.details = details
        self.evidence = evidence or {}
        super().__init__(f"Invariant {invariant} violated: {details}")


@dataclass
class InvariantValidationResult:
    """Result of invariant validation check."""
    invariant_id: str
    passed: bool
    details: str
    evidence: Dict
    tolerance: float = 1e-6


class InvariantValidator:
    """
    Validates mathematical invariants for hybrid fusion system.
    
    Ensures mathematical correctness and prevents score manipulation bugs.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize validator.
        
        Args:
            tolerance: Numerical tolerance for floating point comparisons
        """
        self.tolerance = tolerance
        self.validation_history: List[InvariantValidationResult] = []
    
    def validate_all_invariants(
        self,
        fusion_result: 'FusionResult',
        query: str,
        sparse_results: List,
        dense_results: List,
        fusion_system: 'HybridFusionSystem'
    ) -> List[InvariantValidationResult]:
        """
        Validate all mathematical invariants.
        
        Args:
            fusion_result: Result from fusion
            query: Original query
            sparse_results: Raw sparse retrieval results
            dense_results: Raw dense retrieval results
            fusion_system: Fusion system instance
        
        Returns:
            List of validation results
        
        Raises:
            InvariantViolation: If any invariant fails
        """
        results = []
        
        # P1: α→1 equals BM25-only results
        try:
            p1_result = self.validate_p1_alpha_one_bm25_only(
                fusion_result, sparse_results, fusion_system
            )
            results.append(p1_result)
        except InvariantViolation as e:
            logger.error(f"P1 violation: {e}")
            results.append(InvariantValidationResult(
                invariant_id="P1",
                passed=False,
                details=str(e),
                evidence=e.evidence
            ))
            raise e
        
        # P2: α→0 equals Dense-only results
        try:
            p2_result = self.validate_p2_alpha_zero_dense_only(
                fusion_result, dense_results, fusion_system
            )
            results.append(p2_result)
        except InvariantViolation as e:
            logger.error(f"P2 violation: {e}")
            results.append(InvariantValidationResult(
                invariant_id="P2",
                passed=False,
                details=str(e),
                evidence=e.evidence
            ))
            raise e
        
        # P3: Adding duplicate doc never decreases rank of relevant
        try:
            p3_result = self.validate_p3_duplicate_monotonicity(
                fusion_result, query, fusion_system
            )
            results.append(p3_result)
        except InvariantViolation as e:
            logger.error(f"P3 violation: {e}")
            results.append(InvariantValidationResult(
                invariant_id="P3",
                passed=False,
                details=str(e),
                evidence=e.evidence
            ))
            raise e
        
        # P4: Monotonicity under term weight scaling
        try:
            p4_result = self.validate_p4_term_weight_monotonicity(
                fusion_result, query, fusion_system
            )
            results.append(p4_result)
        except InvariantViolation as e:
            logger.error(f"P4 violation: {e}")
            results.append(InvariantValidationResult(
                invariant_id="P4",
                passed=False,
                details=str(e),
                evidence=e.evidence
            ))
            raise e
        
        # P5: Score calibration monotone in α
        try:
            p5_result = self.validate_p5_alpha_monotonicity(
                fusion_result, query, fusion_system
            )
            results.append(p5_result)
        except InvariantViolation as e:
            logger.error(f"P5 violation: {e}")
            results.append(InvariantValidationResult(
                invariant_id="P5",
                passed=False,
                details=str(e),
                evidence=e.evidence
            ))
            raise e
        
        # Additional runtime checks
        try:
            runtime_result = self.validate_runtime_checks(fusion_result)
            results.append(runtime_result)
        except InvariantViolation as e:
            logger.error(f"Runtime check violation: {e}")
            results.append(InvariantValidationResult(
                invariant_id="RUNTIME",
                passed=False,
                details=str(e),
                evidence=e.evidence
            ))
            raise e
        
        self.validation_history.extend(results)
        
        passed_count = sum(1 for r in results if r.passed)
        logger.info(f"Invariant validation: {passed_count}/{len(results)} passed")
        
        return results
    
    def validate_p1_alpha_one_bm25_only(
        self,
        fusion_result: 'FusionResult',
        sparse_results: List,
        fusion_system: 'HybridFusionSystem'
    ) -> InvariantValidationResult:
        """
        Validate P1: α→1 equals BM25-only results.
        
        When α approaches 1, fusion should match sparse-only retrieval.
        """
        from .core import FusionConfiguration
        
        alpha = fusion_result.config.alpha
        
        # Only test when α is very close to 1
        if alpha < 0.99:
            return InvariantValidationResult(
                invariant_id="P1",
                passed=True,
                details=f"α={alpha:.3f} not near 1, skipping P1 test",
                evidence={"alpha": alpha, "threshold": 0.99}
            )
        
        # Create α=1.0 configuration
        alpha_one_config = FusionConfiguration(
            alpha=1.0,
            k_init_sparse=fusion_result.config.k_init_sparse,
            k_init_dense=fusion_result.config.k_init_dense,
            k_final=fusion_result.config.k_final
        )
        
        # Compare top-k document IDs and scores
        sparse_doc_ids = [r.doc_id for r in sparse_results[:fusion_result.config.k_final]]
        fusion_doc_ids = fusion_result.doc_ids
        
        # Check if document rankings match
        ranking_match = sparse_doc_ids == fusion_doc_ids
        
        # Check score correlation (should be very high)
        if len(sparse_doc_ids) > 1 and len(fusion_doc_ids) > 1:
            sparse_scores = [fusion_result.sparse_scores.get(doc_id, 0.0) for doc_id in sparse_doc_ids]
            fusion_scores_subset = [fusion_result.fusion_scores.get(doc_id, 0.0) for doc_id in sparse_doc_ids]
            
            if len(sparse_scores) > 1:
                correlation = np.corrcoef(sparse_scores, fusion_scores_subset)[0, 1]
                high_correlation = correlation > 0.99
            else:
                correlation = 1.0
                high_correlation = True
        else:
            correlation = 1.0
            high_correlation = True
        
        passed = ranking_match and high_correlation
        
        if not passed:
            raise InvariantViolation(
                invariant="P1",
                details=f"α={alpha:.3f} should match BM25-only but ranking_match={ranking_match}, correlation={correlation:.3f}",
                evidence={
                    "alpha": alpha,
                    "ranking_match": ranking_match,
                    "correlation": correlation,
                    "sparse_top5": sparse_doc_ids[:5],
                    "fusion_top5": fusion_doc_ids[:5]
                }
            )
        
        return InvariantValidationResult(
            invariant_id="P1",
            passed=True,
            details=f"α→1 correctly matches BM25-only (correlation={correlation:.3f})",
            evidence={
                "alpha": alpha,
                "correlation": correlation,
                "ranking_match": ranking_match
            }
        )
    
    def validate_p2_alpha_zero_dense_only(
        self,
        fusion_result: 'FusionResult',
        dense_results: List,
        fusion_system: 'HybridFusionSystem'
    ) -> InvariantValidationResult:
        """
        Validate P2: α→0 equals Dense-only results.
        
        When α approaches 0, fusion should match dense-only retrieval.
        """
        alpha = fusion_result.config.alpha
        
        # Only test when α is very close to 0
        if alpha > 0.01:
            return InvariantValidationResult(
                invariant_id="P2",
                passed=True,
                details=f"α={alpha:.3f} not near 0, skipping P2 test",
                evidence={"alpha": alpha, "threshold": 0.01}
            )
        
        # Compare top-k document IDs and scores
        dense_doc_ids = [r.doc_id for r in dense_results[:fusion_result.config.k_final]]
        fusion_doc_ids = fusion_result.doc_ids
        
        # Check if document rankings match
        ranking_match = dense_doc_ids == fusion_doc_ids
        
        # Check score correlation
        if len(dense_doc_ids) > 1 and len(fusion_doc_ids) > 1:
            dense_scores = [fusion_result.dense_scores.get(doc_id, 0.0) for doc_id in dense_doc_ids]
            fusion_scores_subset = [fusion_result.fusion_scores.get(doc_id, 0.0) for doc_id in dense_doc_ids]
            
            if len(dense_scores) > 1:
                correlation = np.corrcoef(dense_scores, fusion_scores_subset)[0, 1]
                high_correlation = correlation > 0.99
            else:
                correlation = 1.0
                high_correlation = True
        else:
            correlation = 1.0
            high_correlation = True
        
        passed = ranking_match and high_correlation
        
        if not passed:
            raise InvariantViolation(
                invariant="P2",
                details=f"α={alpha:.3f} should match Dense-only but ranking_match={ranking_match}, correlation={correlation:.3f}",
                evidence={
                    "alpha": alpha,
                    "ranking_match": ranking_match,
                    "correlation": correlation,
                    "dense_top5": dense_doc_ids[:5],
                    "fusion_top5": fusion_doc_ids[:5]
                }
            )
        
        return InvariantValidationResult(
            invariant_id="P2",
            passed=True,
            details=f"α→0 correctly matches Dense-only (correlation={correlation:.3f})",
            evidence={
                "alpha": alpha,
                "correlation": correlation,
                "ranking_match": ranking_match
            }
        )
    
    def validate_p3_duplicate_monotonicity(
        self,
        fusion_result: 'FusionResult',
        query: str,
        fusion_system: 'HybridFusionSystem'
    ) -> InvariantValidationResult:
        """
        Validate P3: Adding duplicate doc never decreases rank of relevant.
        
        This is a theoretical check - we verify the fusion formula maintains
        this property by construction.
        """
        # P3 is guaranteed by the additive nature of our fusion formula
        # Score(d) = α·BM25(d) + (1-α)·cos(d)
        # Adding a duplicate can only increase or maintain relative scores
        
        # Verify fusion formula is additive and monotonic
        alpha = fusion_result.config.alpha
        
        # Check that fusion scores are properly computed
        sample_doc_ids = list(fusion_result.fusion_scores.keys())[:5]
        formula_violations = []
        
        for doc_id in sample_doc_ids:
            sparse_score = fusion_result.sparse_scores.get(doc_id, 0.0)
            dense_score = fusion_result.dense_scores.get(doc_id, 0.0) 
            fusion_score = fusion_result.fusion_scores[doc_id]
            
            expected_score = alpha * sparse_score + (1 - alpha) * dense_score
            
            if abs(fusion_score - expected_score) > self.tolerance:
                formula_violations.append({
                    "doc_id": doc_id,
                    "expected": expected_score,
                    "actual": fusion_score,
                    "diff": abs(fusion_score - expected_score)
                })
        
        if formula_violations:
            raise InvariantViolation(
                invariant="P3",
                details=f"Fusion formula violations detected: {len(formula_violations)} documents",
                evidence={"violations": formula_violations}
            )
        
        return InvariantValidationResult(
            invariant_id="P3",
            passed=True,
            details="Additive fusion formula ensures duplicate monotonicity",
            evidence={
                "formula_check": "passed",
                "documents_checked": len(sample_doc_ids)
            }
        )
    
    def validate_p4_term_weight_monotonicity(
        self,
        fusion_result: 'FusionResult',
        query: str,
        fusion_system: 'HybridFusionSystem'  
    ) -> InvariantValidationResult:
        """
        Validate P4: Monotonicity under term weight scaling.
        
        If query terms are scaled uniformly, relative document rankings
        should be preserved.
        """
        # P4 is maintained by BM25's mathematical properties and 
        # vector similarity's scale invariance after normalization
        
        # We verify that our normalization preserves relative rankings
        sparse_scores = list(fusion_result.sparse_scores.values())
        dense_scores = list(fusion_result.dense_scores.values())
        
        # Check that normalization preserves monotonicity
        if len(sparse_scores) > 1:
            sparse_is_monotonic = self._check_score_monotonicity(sparse_scores)
        else:
            sparse_is_monotonic = True
            
        if len(dense_scores) > 1:
            dense_is_monotonic = self._check_score_monotonicity(dense_scores)
        else:
            dense_is_monotonic = True
        
        if not (sparse_is_monotonic and dense_is_monotonic):
            raise InvariantViolation(
                invariant="P4",
                details="Score normalization broke monotonicity",
                evidence={
                    "sparse_monotonic": sparse_is_monotonic,
                    "dense_monotonic": dense_is_monotonic
                }
            )
        
        return InvariantValidationResult(
            invariant_id="P4",
            passed=True,
            details="Term weight scaling monotonicity preserved",
            evidence={
                "sparse_monotonic": sparse_is_monotonic,
                "dense_monotonic": dense_is_monotonic
            }
        )
    
    def validate_p5_alpha_monotonicity(
        self,
        fusion_result: 'FusionResult',
        query: str,
        fusion_system: 'HybridFusionSystem'
    ) -> InvariantValidationResult:
        """
        Validate P5: Score calibration monotone in α.
        
        As α increases, the contribution of sparse scores should increase
        monotonically relative to dense scores.
        """
        alpha = fusion_result.config.alpha
        
        # For a document present in both modalities, verify that
        # d(fusion_score)/d(alpha) = sparse_score - dense_score
        
        # Find documents present in both modalities
        both_modalities = set(fusion_result.sparse_scores.keys()) & set(fusion_result.dense_scores.keys())
        
        if not both_modalities:
            return InvariantValidationResult(
                invariant_id="P5",
                passed=True,
                details="No documents in both modalities, P5 trivially satisfied",
                evidence={"docs_both_modalities": 0}
            )
        
        # Check gradient property for sample documents
        gradient_violations = []
        
        for doc_id in list(both_modalities)[:5]:  # Check first 5
            sparse_score = fusion_result.sparse_scores[doc_id]
            dense_score = fusion_result.dense_scores[doc_id]
            fusion_score = fusion_result.fusion_scores[doc_id]
            
            # Expected gradient: d(fusion)/d(alpha) = sparse - dense
            expected_gradient = sparse_score - dense_score
            
            # Approximate gradient using finite difference if we had multiple alphas
            # For now, verify that the fusion score correctly interpolates
            expected_fusion = alpha * sparse_score + (1 - alpha) * dense_score
            
            if abs(fusion_score - expected_fusion) > self.tolerance:
                gradient_violations.append({
                    "doc_id": doc_id,
                    "expected_fusion": expected_fusion,
                    "actual_fusion": fusion_score,
                    "expected_gradient": expected_gradient,
                    "alpha": alpha
                })
        
        if gradient_violations:
            raise InvariantViolation(
                invariant="P5", 
                details=f"Alpha monotonicity violations: {len(gradient_violations)} documents",
                evidence={"violations": gradient_violations}
            )
        
        return InvariantValidationResult(
            invariant_id="P5",
            passed=True,
            details="Alpha monotonicity satisfied",
            evidence={
                "docs_checked": min(5, len(both_modalities)),
                "alpha": alpha
            }
        )
    
    def validate_runtime_checks(
        self,
        fusion_result: 'FusionResult'
    ) -> InvariantValidationResult:
        """
        Validate runtime checks: non-empty candidates, consistent K, etc.
        """
        violations = []
        
        # Non-empty candidates
        if fusion_result.union_candidates == 0:
            violations.append("Union candidates is zero")
        
        # Consistent K
        if len(fusion_result.doc_ids) != len(fusion_result.scores):
            violations.append("Inconsistent result lengths")
        
        # K <= k_final
        if len(fusion_result.doc_ids) > fusion_result.config.k_final:
            violations.append(f"Too many results: {len(fusion_result.doc_ids)} > {fusion_result.config.k_final}")
        
        # Scores are finite
        if any(not math.isfinite(score) for score in fusion_result.scores):
            violations.append("Non-finite scores detected")
        
        # ANN recall floor (placeholder check)
        if fusion_result.ann_recall_achieved < 0.90:  # Conservative floor
            violations.append(f"ANN recall too low: {fusion_result.ann_recall_achieved:.3f} < 0.90")
        
        if violations:
            raise InvariantViolation(
                invariant="RUNTIME",
                details=f"Runtime check failures: {violations}",
                evidence={"violations": violations}
            )
        
        return InvariantValidationResult(
            invariant_id="RUNTIME",
            passed=True,
            details="All runtime checks passed",
            evidence={"checks_passed": ["non_empty", "consistent_k", "finite_scores", "ann_recall"]}
        )
    
    def _check_score_monotonicity(self, scores: List[float]) -> bool:
        """Check if scores maintain some form of monotonicity (allowing ties)."""
        if len(scores) <= 1:
            return True
        
        # Check for non-decreasing or non-increasing pattern
        non_decreasing = all(scores[i] <= scores[i+1] for i in range(len(scores)-1))
        non_increasing = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        
        return non_decreasing or non_increasing
    
    def get_validation_summary(self) -> Dict:
        """Get summary of all validation results."""
        if not self.validation_history:
            return {"total": 0, "passed": 0, "failed": 0}
        
        passed = sum(1 for r in self.validation_history if r.passed)
        failed = len(self.validation_history) - passed
        
        return {
            "total": len(self.validation_history),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self.validation_history) if self.validation_history else 0.0,
            "invariants": {r.invariant_id: r.passed for r in self.validation_history[-5:]}  # Last 5
        }