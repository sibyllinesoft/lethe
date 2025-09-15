#!/usr/bin/env python3
"""
Postprocess validation script for validator reports.
Hard assertions that MUST pass for CI/CD pipeline success.
"""

import sys
import os
import pandas as pd
import json
import hashlib
from pathlib import Path
from enhanced_html_generator import RECALL_METRIC, LETHE_ENGINE_ADAPTER_ID


def assert_recall_metric():
    """Hard test: RECALL_METRIC must equal 'score'"""
    assert RECALL_METRIC == "score", \
        f"CRITICAL: RECALL_METRIC must be 'score', got '{RECALL_METRIC}'"
    print("✅ RECALL_METRIC == 'score'")


def assert_lethe_engine_in_csv(csv_path):
    """Hard test: LETHE_ENGINE_ADAPTER_ID must exist in CSV"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    adapters = set(df['adapter'].unique())
    
    assert LETHE_ENGINE_ADAPTER_ID in adapters, \
        f"CRITICAL: LETHE_ENGINE_ADAPTER_ID '{LETHE_ENGINE_ADAPTER_ID}' not found in CSV. Available: {sorted(adapters)}"
    print(f"✅ LETHE_ENGINE_ADAPTER_ID '{LETHE_ENGINE_ADAPTER_ID}' found in CSV")


def assert_all_budgets_in_html(html_path):
    """Hard test: Charts must include all 3 budgets (8%, 15%, 30%)"""
    if not os.path.exists(html_path):
        raise FileNotFoundError(f"HTML file not found: {html_path}")
    
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    required_budgets = ["8%", "15%", "30%"]
    missing_budgets = []
    
    for budget in required_budgets:
        if f"{budget} budget" not in html_content:
            missing_budgets.append(budget)
    
    assert not missing_budgets, \
        f"CRITICAL: Missing budget charts: {missing_budgets}. Required: {required_budgets}"
    print("✅ All budget charts present (8%, 15%, 30%)")


def assert_lethe_engine_label_in_html(html_path):
    """Hard test: 'Lethe Engine' label must be present in HTML"""
    if not os.path.exists(html_path):
        raise FileNotFoundError(f"HTML file not found: {html_path}")
    
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    assert "Lethe Engine" in html_content, \
        "CRITICAL: 'Lethe Engine' label not found in HTML"
    print("✅ 'Lethe Engine' label present in HTML")


def assert_manifest_sha_matches(html_path, manifest_path):
    """Hard test: Manifest SHA in HTML must match actual file"""
    if not os.path.exists(html_path):
        raise FileNotFoundError(f"HTML file not found: {html_path}")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    # Calculate actual manifest SHA
    with open(manifest_path, 'rb') as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()[:16]
    
    # Check if it's in HTML
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    assert actual_sha in html_content, \
        f"CRITICAL: Manifest SHA '{actual_sha}' not found in HTML"
    print(f"✅ Manifest SHA '{actual_sha}' matches HTML")


def run_all_hard_tests(csv_path, html_path, manifest_path):
    """Run all hard tests that MUST pass"""
    print("🔍 Running postprocess validation (hard tests)...")
    print(f"  CSV: {csv_path}")
    print(f"  HTML: {html_path}")
    print(f"  Manifest: {manifest_path}")
    print()
    
    try:
        # Run all assertions
        assert_recall_metric()
        assert_lethe_engine_in_csv(csv_path)
        assert_all_budgets_in_html(html_path)
        assert_lethe_engine_label_in_html(html_path)
        assert_manifest_sha_matches(html_path, manifest_path)
        
        print()
        print("🎯 ALL HARD TESTS PASSED! ✅")
        print("Validator report meets all critical requirements.")
        return True
        
    except (AssertionError, FileNotFoundError) as e:
        print()
        print(f"💥 HARD TEST FAILED: {e}")
        print("❌ VALIDATION FAILURE - Report does not meet critical requirements")
        return False
    except Exception as e:
        print()
        print(f"⚠️  UNEXPECTED ERROR: {e}")
        return False


def main():
    """Main entry point"""
    if len(sys.argv) != 4:
        print("Usage: python postprocess_validation.py <csv_path> <html_path> <manifest_path>")
        print()
        print("Example:")
        print("  python postprocess_validation.py metrics.csv validator_report.html signed_manifest.json")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    html_path = sys.argv[2]
    manifest_path = sys.argv[3]
    
    success = run_all_hard_tests(csv_path, html_path, manifest_path)
    
    # Exit with appropriate code for CI/CD
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()