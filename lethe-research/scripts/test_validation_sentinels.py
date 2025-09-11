#!/usr/bin/env python3
"""
Test script for validation sentinels - demonstrates fail-closed behavior
"""

import sys
import json
from pathlib import Path
from validation_sentinels import (
    validate_measurement_pipeline_v2, 
    ValidationThresholds,
    generate_validation_summary
)

def create_test_data_broken():
    """Create test data that should FAIL validation"""
    return [
        # All accuracy = 0.0 and other issues
        {
            'method_name': 'streaming',
            'dataset': 'code_debug',
            'keep_ratio': 0.08,
            'p_at_k': {5: 0.0, 10: 0.0},  # Universal zeros - should fail
            'delta_cbu_per_1k': 0.01,     # Constant values - should fail
            'kv_reuse': 0.0,              # Universal zeros - should fail
            'tokens_kept': 5,             # Tiny cluster - should fail
            'compression_ratio': 0.05,
            'tail_cvar': 0.0,
            'middleware_p95_ms': 200.0
        },
        {
            'method_name': 'lethe',
            'dataset': 'code_debug', 
            'keep_ratio': 0.08,
            'p_at_k': {5: 0.0, 10: 0.0},  # Universal zeros
            'delta_cbu_per_1k': 0.01,     # Same as above - no variance
            'kv_reuse': 0.0,              # Universal zeros
            'tokens_kept': 6,             # Tiny cluster
            'compression_ratio': 0.05,
            'tail_cvar': 0.0,
            'middleware_p95_ms': 250.0
        },
        {
            'method_name': 'hybrid',
            'dataset': 'zh_qa',
            'keep_ratio': 0.08,
            'p_at_k': {5: 0.0, 10: 0.0},  # Universal zeros
            'delta_cbu_per_1k': 0.01,     # Same again - no variance
            'kv_reuse': 0.0,              # Universal zeros
            'tokens_kept': 50,            # zh_qa too low - should fail
            'compression_ratio': 0.05,
            'tail_cvar': 0.0,
            'middleware_p95_ms': 300.0
        }
    ]

def create_test_data_good():
    """Create test data that should PASS validation"""
    return [
        {
            'method_name': 'streaming',
            'dataset': 'code_debug',
            'keep_ratio': 0.08,
            'p_at_k': {5: 0.65, 10: 0.72},
            'delta_cbu_per_1k': 0.012,    # Varied values
            'kv_reuse': 0.75,             # High for code_debug
            'tokens_kept': 800,           # Reasonable values
            'compression_ratio': 0.08,
            'tail_cvar': 0.15,
            'middleware_p95_ms': 200.0
        },
        {
            'method_name': 'streaming',
            'dataset': 'code_debug',
            'keep_ratio': 0.15,
            'p_at_k': {5: 0.68, 10: 0.75},
            'delta_cbu_per_1k': 0.018,    # Higher with more tokens
            'kv_reuse': 0.73,
            'tokens_kept': 1200,          # Monotonic increase
            'compression_ratio': 0.08,
            'tail_cvar': 0.12,
            'middleware_p95_ms': 180.0
        },
        {
            'method_name': 'streaming', 
            'dataset': 'code_debug',
            'keep_ratio': 0.30,
            'p_at_k': {5: 0.72, 10: 0.78},
            'delta_cbu_per_1k': 0.025,    # Highest with most tokens
            'kv_reuse': 0.71,
            'tokens_kept': 2400,          # Monotonic increase
            'compression_ratio': 0.08,
            'tail_cvar': 0.08,
            'middleware_p95_ms': 160.0
        },
        {
            'method_name': 'lethe',
            'dataset': 'code_qa',
            'keep_ratio': 0.08,
            'p_at_k': {5: 0.58, 10: 0.64},
            'delta_cbu_per_1k': 0.015,    # Different system, different values
            'kv_reuse': 0.62,             # Lower for code_qa vs debug
            'tokens_kept': 750,
            'compression_ratio': 0.08,
            'tail_cvar': 0.18,
            'middleware_p95_ms': 220.0
        },
        {
            'method_name': 'hybrid',
            'dataset': 'zh_qa',
            'keep_ratio': 0.08,
            'p_at_k': {5: 0.45, 10: 0.52},
            'delta_cbu_per_1k': 0.020,    # Different again - creates variance
            'kv_reuse': 0.48,             # Expected for zh_qa
            'tokens_kept': 650,           # Above zh_qa threshold
            'compression_ratio': 0.08,
            'tail_cvar': 0.22,
            'middleware_p95_ms': 280.0
        }
    ]

def test_validation_system():
    """Test the validation system with both good and bad data"""
    
    print("🧪 Testing Validation Sentinels System")
    print("="*60)
    
    # Test 1: Broken data should fail
    print("\n1️⃣ Testing with BROKEN data (should fail):")
    broken_data = create_test_data_broken()
    
    try:
        report = validate_measurement_pipeline_v2(
            broken_data, 
            fail_fast=False  # Don't exit, just report
        )
        
        if report.success:
            print("❌ ERROR: Broken data passed validation!")
        else:
            print(f"✅ CORRECT: Broken data failed as expected ({len(report.failures)} failures)")
            for failure in report.failures[:3]:  # Show first 3
                print(f"   - {failure.sentinel_name}: {failure.message}")
                
    except Exception as e:
        print(f"❌ Validation system crashed: {e}")
    
    # Test 2: Good data should pass
    print("\n2️⃣ Testing with GOOD data (should pass):")
    good_data = create_test_data_good()
    
    try:
        report = validate_measurement_pipeline_v2(
            good_data,
            fail_fast=False
        )
        
        if report.success:
            print(f"✅ CORRECT: Good data passed validation ({len(report.passed_sentinels)} sentinels)")
            for sentinel in report.passed_sentinels:
                print(f"   - ✅ {sentinel}")
        else:
            print(f"❌ ERROR: Good data failed validation ({len(report.failures)} failures)")
            for failure in report.failures:
                print(f"   - {failure}")
                
    except Exception as e:
        print(f"❌ Validation system crashed: {e}")
    
    # Test 3: Save validation report
    print("\n3️⃣ Generating validation report:")
    
    try:
        output_dir = Path("artifacts/validation_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary = generate_validation_summary(
            report, 
            output_dir / "validation_test_report.md"
        )
        
        print(f"📄 Validation report saved to {output_dir / 'validation_test_report.md'}")
        
        # Also test with your actual evaluation results if they exist
        artifacts_dir = Path("artifacts/hybrid_evaluation")
        if artifacts_dir.exists():
            json_files = list(artifacts_dir.glob("hybrid_evaluation_*.json"))
            if json_files:
                latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
                print(f"\n4️⃣ Testing with actual evaluation results from {latest_file.name}:")
                
                with open(latest_file) as f:
                    actual_data = json.load(f)
                
                # Extract results in the format expected by validation
                flat_results = []
                for method, results in actual_data.get('results', {}).items():
                    for r in results:
                        flat_results.append(r)
                
                if flat_results:
                    report = validate_measurement_pipeline_v2(
                        flat_results,
                        fail_fast=False
                    )
                    
                    if report.success:
                        print("✅ Actual evaluation results PASSED validation")
                    else:
                        print(f"❌ Actual evaluation results FAILED validation ({len(report.failures)} issues)")
                        for failure in report.failures[:5]:  # Show first 5
                            print(f"   - {failure.sentinel_name}: {failure.message}")
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
    
    print("\n" + "="*60)
    print("🔒 Validation Sentinels Test Complete")

if __name__ == '__main__':
    test_validation_system()