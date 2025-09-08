#!/usr/bin/env python3
"""
Lethe Fail-Closed Validator
===========================

Validates benchmark results with strict statistical integrity checks.
Fails closed on any violation - no partial results allowed.
"""

import json
import sys
import numpy as np
from typing import Dict, List, Tuple, Any


class LetheValidator:
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.violations = []
    
    def validate_results(self, results_file: str) -> Tuple[bool, List[str]]:
        """Validate results file with fail-closed policy"""
        try:
            with open(results_file) as f:
                data = json.load(f)
            
            # Reset violations
            self.violations = []
            
            # Run all validation checks
            self._validate_ci_integrity(data)
            self._validate_paired_aggregation(data) 
            self._validate_pool_consistency(data)
            self._validate_statistical_significance(data)
            self._validate_fairness_invariants(data)
            
            # Fail closed on any violation
            if self.violations:
                return False, self.violations
            
            return True, ["All validations passed"]
            
        except Exception as e:
            return False, [f"Validation failed with error: {e}"]
    
    def _validate_ci_integrity(self, data: Dict):
        """Ensure all confidence intervals bracket their means"""
        for system_name, system_data in data.get("systems", {}).items():
            relevance = system_data.get("relevance_score", 0)
            ci = system_data.get("bootstrap_ci", [0, 1])
            
            if len(ci) != 2:
                self.violations.append(f"Invalid CI format for {system_name}")
                continue
            
            ci_lower, ci_upper = ci
            if not (ci_lower <= relevance <= ci_upper):
                self.violations.append(
                    f"CI integrity violation: {system_name} mean {relevance:.3f} "
                    f"not in CI [{ci_lower:.3f}, {ci_upper:.3f}]"
                )
    
    def _validate_paired_aggregation(self, data: Dict):
        """Validate paired aggregation consistency"""
        paired_counts = []
        for system_data in data.get("systems", {}).values():
            paired_counts.append(system_data.get("paired_slices", 0))
        
        if len(set(paired_counts)) > 1:
            self.violations.append(
                f"Paired aggregation violation: inconsistent slice counts {set(paired_counts)}"
            )
    
    def _validate_pool_consistency(self, data: Dict):
        """Validate pool fingerprint consistency"""
        pool_fingerprints = []
        for system_data in data.get("systems", {}).values():
            fingerprint = system_data.get("pool_fingerprint")
            if fingerprint:
                pool_fingerprints.append(fingerprint)
        
        if len(set(pool_fingerprints)) > 1:
            self.violations.append(
                f"Pool consistency violation: multiple pool fingerprints {set(pool_fingerprints)}"
            )
    
    def _validate_statistical_significance(self, data: Dict):
        """Validate statistical significance of results"""
        for system_name, system_data in data.get("systems", {}).items():
            p_value = system_data.get("p_value_vs_lethe")
            if p_value is not None and p_value > 0.05:
                self.violations.append(
                    f"Statistical significance violation: {system_name} p-value {p_value:.4f} > 0.05"
                )
    
    def _validate_fairness_invariants(self, data: Dict):
        """Validate fairness invariants"""
        # Check for reasonable latency ratios
        for system_name, system_data in data.get("systems", {}).items():
            avg_latency = system_data.get("latency_ms", 0)
            p95_latency = system_data.get("p95_latency_ms", 0)
            
            if p95_latency < avg_latency:
                self.violations.append(
                    f"Fairness violation: {system_name} P95 < avg latency"
                )
            
            # Check for reasonable P99/P95 ratios
            p99_latency = system_data.get("p99_latency_ms", p95_latency * 1.5)
            if p99_latency / p95_latency > 3.0:
                self.violations.append(
                    f"Fairness violation: {system_name} P99/P95 ratio too high"
                )


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Lethe Fail-Closed Validator")
    parser.add_argument("results_file", help="Results file to validate")
    parser.add_argument("--strict", action="store_true", help="Enable strict mode")
    
    args = parser.parse_args()
    
    validator = LetheValidator(strict_mode=args.strict)
    is_valid, messages = validator.validate_results(args.results_file)
    
    if is_valid:
        print("✅ VALIDATION PASSED")
        for msg in messages:
            print(f"   {msg}")
        sys.exit(0)
    else:
        print("❌ VALIDATION FAILED")
        for violation in messages:
            print(f"   {violation}")
        sys.exit(1)


if __name__ == "__main__":
    main()
