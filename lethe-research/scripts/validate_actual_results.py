#!/usr/bin/env python3
"""
Test validation sentinels on actual evaluation results
"""

import json
import sys
from pathlib import Path
from validation_sentinels import validate_measurement_pipeline_v2, generate_validation_summary

def validate_actual_results():
    """Test validation on the actual evaluation results"""
    
    # Find the most recent evaluation results
    results_dir = Path("artifacts/hybrid_evaluation")
    if not results_dir.exists():
        print("❌ No evaluation results found")
        return
    
    json_files = list(results_dir.glob("hybrid_evaluation_*.json"))
    if not json_files:
        print("❌ No hybrid evaluation JSON files found")
        return
    
    # Get the most recent file
    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    print(f"🔍 Testing validation on: {latest_file.name}")
    
    try:
        # Load the actual results
        with open(latest_file) as f:
            data = json.load(f)
        
        # Extract results in the format expected by validation
        flat_results = []
        for method, results in data.get('results', {}).items():
            for r in results:
                flat_results.append(r)
        
        print(f"📊 Loaded {len(flat_results)} evaluation records")
        
        # Run validation
        print("\n" + "="*60)
        print("RUNNING COMPREHENSIVE VALIDATION ON ACTUAL RESULTS")
        print("="*60)
        
        report = validate_measurement_pipeline_v2(
            flat_results,
            fail_fast=False  # Don't exit, show all failures
        )
        
        # Generate detailed report
        validation_dir = Path("artifacts/validation_actual")
        validation_dir.mkdir(parents=True, exist_ok=True)
        
        summary = generate_validation_summary(
            report, 
            validation_dir / f"validation_report_{latest_file.stem}.md"
        )
        
        print("\n" + "="*60)
        if report.success:
            print("✅ VALIDATION PASSED - All measurement pipes working correctly")
        else:
            print("❌ VALIDATION FAILED - Critical measurement pipe failures detected")
            print(f"\nFailed Sentinels: {', '.join(report.failed_sentinels)}")
            print(f"Critical Failures: {len(report.failures)}")
            
            print("\nTop 5 Critical Issues:")
            for i, failure in enumerate(report.failures[:5], 1):
                print(f"{i}. {failure.sentinel_name}: {failure.message}")
                print(f"   Expected: {failure.expected}")
                print(f"   Actual: {failure.actual}")
                print()
        
        print(f"📄 Detailed report saved to: {validation_dir / f'validation_report_{latest_file.stem}.md'}")
        print("="*60)
        
        # Show a preview of the data that's causing issues
        if not report.success:
            print("\n🔍 Sample of problematic data:")
            for result in flat_results[:3]:
                print(f"- {result.get('method_name', 'unknown')}: accuracy={result.get('accuracy', 'N/A')}, "
                      f"p@5={result.get('p_at_k', {}).get(5, 'N/A')}, "
                      f"delta_cbu={result.get('delta_cbu_per_1k', 'N/A')}, "
                      f"kv_reuse={result.get('kv_reuse', 'N/A')}, "
                      f"tokens={result.get('tokens_kept', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error validating results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    validate_actual_results()