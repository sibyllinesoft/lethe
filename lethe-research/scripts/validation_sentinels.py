#!/usr/bin/env python3
"""
Comprehensive Validation System with Fail-Closed Sentinels
=========================================================

Validates three critical measurement pipes for research evaluation system:
1. ΔCBU Computation Pipeline 
2. Token Accounting Pipeline
3. KV-Reuse Measurement Pipeline

All sentinels are FAIL-CLOSED - any trip immediately stops execution.
"""

import sys
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import scipy.stats
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationThresholds:
    """Configuration thresholds for validation sentinels."""
    
    # ΔCBU Validation 
    delta_cbu_variance_epsilon: float = 1e-3
    delta_cbu_correlation_min: float = 0.3
    
    # Token Accounting
    zh_qa_min_tokens_at_8pct: int = 500
    compression_ratio_min: float = 0.07
    compression_ratio_max: float = 0.09
    tiny_token_cluster_threshold: int = 10  # Values {4,5,6,...,10}
    
    # KV-Reuse Validation
    prefix_jaccard_nonzero_min: float = 0.8  # 80% of samples should have >0.1 jaccard
    prefix_jaccard_threshold: float = 0.1
    
    # Dataset-specific medians
    code_debug_median_min: float = 0.7
    code_qa_median_target: float = 0.6
    zh_qa_median_target: float = 0.5

@dataclass
class ValidationFailure:
    """Represents a validation failure."""
    sentinel_name: str
    check_name: str
    expected: Any
    actual: Any
    message: str
    severity: str = "CRITICAL"  # CRITICAL stops execution
    
    def __str__(self) -> str:
        return f"[{self.severity}] {self.sentinel_name}.{self.check_name}: {self.message}"

@dataclass
class ValidationReport:
    """Complete validation report."""
    timestamp: str
    total_records: int
    passed_sentinels: List[str] = field(default_factory=list)
    failed_sentinels: List[str] = field(default_factory=list)
    failures: List[ValidationFailure] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    success: bool = False
    
    def add_failure(self, failure: ValidationFailure):
        """Add a validation failure."""
        self.failures.append(failure)
        if failure.sentinel_name not in self.failed_sentinels:
            self.failed_sentinels.append(failure.sentinel_name)
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(f"[WARNING] {message}")
    
    def mark_sentinel_passed(self, sentinel_name: str):
        """Mark a sentinel as passed."""
        if sentinel_name not in self.failed_sentinels:
            self.passed_sentinels.append(sentinel_name)

class ValidationSentinels:
    """Comprehensive validation system with fail-closed sentinels."""
    
    def __init__(self, thresholds: ValidationThresholds = None):
        self.thresholds = thresholds or ValidationThresholds()
        self.report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            total_records=0
        )
    
    def validate_pipeline(self, results: List[Dict[str, Any]]) -> ValidationReport:
        """
        Validate the complete measurement pipeline.
        
        Args:
            results: List of evaluation results from hybrid evaluation
            
        Returns:
            ValidationReport with success=True only if ALL sentinels pass
        """
        logger.info("🔒 Starting fail-closed validation sentinels")
        
        self.report.total_records = len(results)
        
        # Fail fast on empty results
        if not results:
            self.report.add_failure(ValidationFailure(
                sentinel_name="PreCheck",
                check_name="EmptyResults",
                expected=">0 records",
                actual="0 records",
                message="No evaluation results provided - pipeline completely broken"
            ))
            return self._finalize_report()
        
        # Run each sentinel - any failure immediately fails the entire pipeline
        try:
            self._validate_delta_cbu_sentinel(results)
            if self.report.failures:
                return self._finalize_report()
            
            self._validate_token_accounting_sentinel(results)
            if self.report.failures:
                return self._finalize_report()
            
            self._validate_kv_reuse_sentinel(results)
            if self.report.failures:
                return self._finalize_report()
            
            # If we get here, all sentinels passed
            self.report.success = True
            logger.info("✅ ALL VALIDATION SENTINELS PASSED")
            
        except Exception as e:
            self.report.add_failure(ValidationFailure(
                sentinel_name="SystemError",
                check_name="UnexpectedError",
                expected="Normal execution",
                actual=f"Exception: {e}",
                message=f"Validation system crashed: {e}"
            ))
        
        return self._finalize_report()
    
    def _validate_delta_cbu_sentinel(self, results: List[Dict[str, Any]]):
        """
        ΔCBU Validation Sentinel (fail-closed if any trip):
        - std(delta_cbu_per_1k | system,dataset,keep_ratio) > ε
        - Pearson/Spearman correlation(ΔCBU, macro-P@5) > 0.3 across slices  
        - eval_ok=true only if all invariants pass
        - Reject if any scenario has constant ΔCBU across systems
        - If V2 payload missing → drop row, don't zero-fill
        """
        logger.info("🔍 ΔCBU Validation Sentinel")
        
        # Extract ΔCBU data
        delta_cbu_data = []
        p_at_5_data = []
        
        for result in results:
            # Check for V2 payload - skip if missing (don't zero-fill)
            if not self._has_v2_payload(result):
                self.report.add_warning(f"Dropping row due to missing V2 payload: {result.get('method_name', 'unknown')}")
                continue
            
            delta_cbu = result.get('delta_cbu_per_1k', 0.0)
            p_at_5 = result.get('p_at_k', {}).get(5, 0.0)
            
            if delta_cbu == 0.0:
                self.report.add_failure(ValidationFailure(
                    sentinel_name="ΔCBU_Sentinel",
                    check_name="ZeroDeltaCBU",
                    expected=">0",
                    actual=delta_cbu,
                    message=f"ΔCBU is zero for {result.get('method_name', 'unknown')} - computation pipe broken"
                ))
                return
            
            delta_cbu_data.append(delta_cbu)
            p_at_5_data.append(p_at_5)
        
        if len(delta_cbu_data) < 3:
            self.report.add_failure(ValidationFailure(
                sentinel_name="ΔCBU_Sentinel", 
                check_name="InsufficientData",
                expected="≥3 valid measurements",
                actual=len(delta_cbu_data),
                message="Not enough valid ΔCBU measurements for statistical validation"
            ))
            return
        
        # Check 1: Variance across scenarios
        delta_cbu_std = np.std(delta_cbu_data)
        if delta_cbu_std <= self.thresholds.delta_cbu_variance_epsilon:
            self.report.add_failure(ValidationFailure(
                sentinel_name="ΔCBU_Sentinel",
                check_name="InsufficientVariance", 
                expected=f"std > {self.thresholds.delta_cbu_variance_epsilon}",
                actual=delta_cbu_std,
                message=f"ΔCBU variance too low ({delta_cbu_std:.6f}) - likely constant values across systems"
            ))
            return
        
        # Check 2: Correlation with P@5
        if len(p_at_5_data) == len(delta_cbu_data) and len(p_at_5_data) > 2:
            # Remove any zero P@5 values that might break correlation
            valid_pairs = [(d, p) for d, p in zip(delta_cbu_data, p_at_5_data) if p > 0]
            
            if len(valid_pairs) >= 3:
                valid_delta, valid_p5 = zip(*valid_pairs)
                
                # Pearson correlation
                pearson_r, pearson_p = scipy.stats.pearsonr(valid_delta, valid_p5)
                spearman_r, spearman_p = scipy.stats.spearmanr(valid_delta, valid_p5)
                
                if abs(pearson_r) < self.thresholds.delta_cbu_correlation_min and abs(spearman_r) < self.thresholds.delta_cbu_correlation_min:
                    self.report.add_failure(ValidationFailure(
                        sentinel_name="ΔCBU_Sentinel",
                        check_name="WeakCorrelation",
                        expected=f"correlation > {self.thresholds.delta_cbu_correlation_min}",
                        actual=f"Pearson: {pearson_r:.3f}, Spearman: {spearman_r:.3f}",
                        message="ΔCBU shows no correlation with P@5 - measurement pipe disconnected"
                    ))
                    return
        
        # Check 3: Constant ΔCBU detection across systems
        system_groups = defaultdict(list)
        for result in results:
            if self._has_v2_payload(result):
                system = result.get('method_name', 'unknown')
                delta_cbu = result.get('delta_cbu_per_1k', 0.0)
                system_groups[system].append(delta_cbu)
        
        for system, values in system_groups.items():
            if len(values) > 1:
                system_std = np.std(values)
                if system_std < self.thresholds.delta_cbu_variance_epsilon / 10:  # Even stricter for within-system
                    self.report.add_failure(ValidationFailure(
                        sentinel_name="ΔCBU_Sentinel",
                        check_name="ConstantSystemValues",
                        expected=f"std > {self.thresholds.delta_cbu_variance_epsilon/10}",
                        actual=system_std,
                        message=f"System {system} has constant ΔCBU values - computation not varying with conditions"
                    ))
                    return
        
        self.report.mark_sentinel_passed("ΔCBU_Sentinel")
        logger.info(f"✅ ΔCBU Sentinel PASSED (std={delta_cbu_std:.6f}, correlation validated)")
    
    def _validate_token_accounting_sentinel(self, results: List[Dict[str, Any]]):
        """
        Token Accounting Validation Sentinel:
        - Monotonicity: median(tokens_kept@30%) > median@15% > median@8%)
        - Sanity check (zh_qa): median(tokens_kept@8%) > 500
        - Compression ratios: 0.07 < median(tokens_kept/tokens_in@8%) < 0.09
        - Reject if "tokens_kept" clustered at tiny values {4,5,6,8,...}
        """
        logger.info("🔍 Token Accounting Validation Sentinel")
        
        # Group by keep_ratio
        ratio_groups = defaultdict(list)
        zh_qa_groups = defaultdict(list)
        
        for result in results:
            keep_ratio = result.get('keep_ratio', 0.0)
            tokens_kept = result.get('tokens_kept', 0)
            dataset = result.get('dataset', '')
            
            if tokens_kept <= 0:
                self.report.add_failure(ValidationFailure(
                    sentinel_name="TokenAccounting_Sentinel",
                    check_name="ZeroTokensKept",
                    expected=">0 tokens",
                    actual=tokens_kept,
                    message=f"Zero tokens kept for {result.get('method_name', 'unknown')} - token accounting broken"
                ))
                return
            
            ratio_groups[keep_ratio].append(tokens_kept)
            
            if 'zh_qa' in dataset.lower():
                zh_qa_groups[keep_ratio].append(tokens_kept)
        
        # Check 1: Monotonicity across keep ratios
        expected_ratios = [0.08, 0.15, 0.30]
        medians = {}
        
        for ratio in expected_ratios:
            if ratio in ratio_groups and ratio_groups[ratio]:
                medians[ratio] = np.median(ratio_groups[ratio])
            else:
                self.report.add_failure(ValidationFailure(
                    sentinel_name="TokenAccounting_Sentinel",
                    check_name="MissingKeepRatio",
                    expected=f"Data for keep_ratio={ratio}",
                    actual="No data",
                    message=f"Missing data for keep_ratio {ratio} - incomplete evaluation"
                ))
                return
        
        # Monotonicity check
        if not (medians[0.08] < medians[0.15] < medians[0.30]):
            self.report.add_failure(ValidationFailure(
                sentinel_name="TokenAccounting_Sentinel",
                check_name="MonotonicityViolation",
                expected="tokens_kept@8% < @15% < @30%",
                actual=f"@8%:{medians[0.08]}, @15%:{medians[0.15]}, @30%:{medians[0.30]}",
                message="Token accounting not monotonic - keep_ratio not controlling token retention"
            ))
            return
        
        # Check 2: zh_qa sanity check
        if 0.08 in zh_qa_groups and zh_qa_groups[0.08]:
            zh_qa_median_8pct = np.median(zh_qa_groups[0.08])
            if zh_qa_median_8pct < self.thresholds.zh_qa_min_tokens_at_8pct:
                self.report.add_failure(ValidationFailure(
                    sentinel_name="TokenAccounting_Sentinel",
                    check_name="ZhQaSanityFailure",
                    expected=f"> {self.thresholds.zh_qa_min_tokens_at_8pct} tokens",
                    actual=zh_qa_median_8pct,
                    message=f"zh_qa median tokens@8% too low ({zh_qa_median_8pct}) - window/sink confusion?"
                ))
                return
        
        # Check 3: Tiny token clustering detection
        all_tokens = [token for tokens_list in ratio_groups.values() for token in tokens_list]
        tiny_values = [t for t in all_tokens if t <= self.thresholds.tiny_token_cluster_threshold]
        
        if len(tiny_values) > len(all_tokens) * 0.5:  # More than 50% are tiny values
            self.report.add_failure(ValidationFailure(
                sentinel_name="TokenAccounting_Sentinel", 
                check_name="TinyTokenClustering",
                expected=f"< 50% values ≤ {self.thresholds.tiny_token_cluster_threshold}",
                actual=f"{len(tiny_values)}/{len(all_tokens)} ({len(tiny_values)/len(all_tokens)*100:.1f}%)",
                message="Tokens clustered at tiny values - likely accounting error or wrong field"
            ))
            return
        
        self.report.mark_sentinel_passed("TokenAccounting_Sentinel")
        logger.info(f"✅ Token Accounting Sentinel PASSED (monotonicity: {medians[0.08]:.0f} < {medians[0.15]:.0f} < {medians[0.30]:.0f})")
    
    def _validate_kv_reuse_sentinel(self, results: List[Dict[str, Any]]):
        """
        KV-Reuse Validation Sentinel:
        - Non-zero mass: share(prefix_jaccard>0.1) ≥ 0.8 in each scenario
        - Dataset-specific medians: Code.Debug ≥0.7, Code.QA ~0.6, Zh.QA ~0.5
        - No universal zeros (arranger must be wired)
        """
        logger.info("🔍 KV-Reuse Validation Sentinel")
        
        kv_reuse_values = []
        dataset_groups = defaultdict(list)
        
        for result in results:
            kv_reuse = result.get('kv_reuse', 0.0)
            dataset = result.get('dataset', '')
            
            kv_reuse_values.append(kv_reuse)
            dataset_groups[dataset].append(kv_reuse)
        
        # Check 1: Universal zeros detection
        if all(kv == 0.0 for kv in kv_reuse_values):
            self.report.add_failure(ValidationFailure(
                sentinel_name="KVReuse_Sentinel",
                check_name="UniversalZeros", 
                expected="Some non-zero KV reuse values",
                actual="All zeros",
                message="All KV reuse values are zero - arranger not wired or completely broken"
            ))
            return
        
        # Check 2: Non-zero mass in each scenario
        nonzero_count = sum(1 for kv in kv_reuse_values if kv > self.thresholds.prefix_jaccard_threshold)
        nonzero_ratio = nonzero_count / len(kv_reuse_values) if kv_reuse_values else 0.0
        
        if nonzero_ratio < self.thresholds.prefix_jaccard_nonzero_min:
            self.report.add_failure(ValidationFailure(
                sentinel_name="KVReuse_Sentinel",
                check_name="InsufficientNonZeroMass",
                expected=f"≥ {self.thresholds.prefix_jaccard_nonzero_min*100:.0f}% > {self.thresholds.prefix_jaccard_threshold}",
                actual=f"{nonzero_ratio*100:.1f}%",
                message=f"Too few samples with meaningful KV reuse - arranger likely broken"
            ))
            return
        
        # Check 3: Dataset-specific median checks
        dataset_checks = {
            'code_debug': self.thresholds.code_debug_median_min,
            'code_qa': self.thresholds.code_qa_median_target,  
            'zh_qa': self.thresholds.zh_qa_median_target
        }
        
        for dataset, expected_median in dataset_checks.items():
            # Find matching datasets (flexible matching)
            matching_values = []
            for ds_key, values in dataset_groups.items():
                if dataset.replace('_', '').lower() in ds_key.replace('_', '').lower():
                    matching_values.extend(values)
            
            if matching_values:
                actual_median = np.median(matching_values)
                
                # For code_debug, enforce minimum. For others, check reasonable range
                if dataset == 'code_debug':
                    if actual_median < expected_median:
                        self.report.add_failure(ValidationFailure(
                            sentinel_name="KVReuse_Sentinel",
                            check_name=f"DatasetMedian_{dataset}",
                            expected=f"≥ {expected_median}",
                            actual=actual_median,
                            message=f"{dataset} KV reuse median too low - expected high reuse for debug scenarios"
                        ))
                        return
                else:
                    # For other datasets, check within reasonable range (±0.2 of target)
                    if abs(actual_median - expected_median) > 0.2:
                        self.report.add_warning(
                            f"{dataset} KV reuse median ({actual_median:.3f}) far from expected ({expected_median:.3f})"
                        )
        
        self.report.mark_sentinel_passed("KVReuse_Sentinel")
        logger.info(f"✅ KV-Reuse Sentinel PASSED ({nonzero_ratio*100:.1f}% non-zero, medians validated)")
    
    def _has_v2_payload(self, result: Dict[str, Any]) -> bool:
        """Check if result has V2 payload markers."""
        # Look for V2-specific fields or markers
        v2_indicators = [
            'delta_cbu_per_1k',
            'kv_reuse', 
            'tail_cvar',
            'eval_ok'
        ]
        
        return any(indicator in result for indicator in v2_indicators)
    
    def _finalize_report(self) -> ValidationReport:
        """Finalize the validation report."""
        if self.report.failures:
            self.report.success = False
            logger.error(f"❌ VALIDATION FAILED - {len(self.report.failures)} critical failures")
            
            for failure in self.report.failures:
                logger.error(f"  {failure}")
        else:
            self.report.success = True
            logger.info(f"✅ ALL SENTINELS PASSED - {len(self.report.passed_sentinels)} sentinels validated")
        
        return self.report

def validate_measurement_pipeline_v2(results: List[Dict[str, Any]], 
                                   thresholds: ValidationThresholds = None,
                                   fail_fast: bool = True) -> ValidationReport:
    """
    Enhanced validation function with comprehensive fail-closed sentinels.
    
    Args:
        results: Evaluation results to validate
        thresholds: Custom validation thresholds
        fail_fast: If True, stop on first failure (default)
        
    Returns:
        ValidationReport with success=True only if ALL sentinels pass
        
    Raises:
        SystemExit: If fail_fast=True and validation fails
    """
    validator = ValidationSentinels(thresholds)
    report = validator.validate_pipeline(results)
    
    if fail_fast and not report.success:
        logger.error("🚨 VALIDATION FAILED - STOPPING EXECUTION")
        logger.error("Pipeline has critical measurement failures that would invalidate results")
        
        print("\n" + "="*80)
        print("CRITICAL VALIDATION FAILURES DETECTED")
        print("="*80)
        
        for failure in report.failures:
            print(f"❌ {failure}")
        
        print(f"\nFailed Sentinels: {', '.join(report.failed_sentinels)}")
        print("Fix these issues before generating any metrics or claims.")
        print("="*80)
        
        sys.exit(1)
    
    return report

def generate_validation_summary(report: ValidationReport, output_path: Path = None) -> str:
    """Generate human-readable validation summary."""
    
    summary = []
    summary.append("# Measurement Pipeline Validation Report")
    summary.append(f"Generated: {report.timestamp}")
    summary.append(f"Records Validated: {report.total_records}")
    summary.append("")
    
    if report.success:
        summary.append("## ✅ VALIDATION PASSED")
        summary.append("All critical measurement pipes are functioning correctly.")
        summary.append("")
        summary.append("### Passed Sentinels:")
        for sentinel in report.passed_sentinels:
            summary.append(f"- ✅ {sentinel}")
    else:
        summary.append("## ❌ VALIDATION FAILED")
        summary.append("Critical measurement pipe failures detected.")
        summary.append("")
        summary.append("### Failed Sentinels:")
        for sentinel in report.failed_sentinels:
            summary.append(f"- ❌ {sentinel}")
        
        summary.append("")
        summary.append("### Critical Failures:")
        for failure in report.failures:
            summary.append(f"- **{failure.check_name}**: {failure.message}")
            summary.append(f"  - Expected: {failure.expected}")
            summary.append(f"  - Actual: {failure.actual}")
    
    if report.warnings:
        summary.append("")
        summary.append("### Warnings:")
        for warning in report.warnings:
            summary.append(f"- ⚠️ {warning}")
    
    summary.append("")
    summary.append("## Summary")
    summary.append(f"- **Total Sentinels**: {len(report.passed_sentinels) + len(report.failed_sentinels)}")
    summary.append(f"- **Passed**: {len(report.passed_sentinels)}")
    summary.append(f"- **Failed**: {len(report.failed_sentinels)}")
    summary.append(f"- **Critical Failures**: {len(report.failures)}")
    summary.append(f"- **Warnings**: {len(report.warnings)}")
    
    summary_text = "\n".join(summary)
    
    if output_path:
        output_path.write_text(summary_text)
        logger.info(f"Validation summary saved to {output_path}")
    
    return summary_text

# Export for use in main evaluation pipeline
__all__ = [
    'ValidationSentinels',
    'ValidationThresholds', 
    'ValidationReport',
    'ValidationFailure',
    'validate_measurement_pipeline_v2',
    'generate_validation_summary'
]

if __name__ == '__main__':
    # Demo/test usage
    print("Validation Sentinels Module")
    print("Import this module and call validate_measurement_pipeline_v2(results)")