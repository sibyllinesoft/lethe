#!/usr/bin/env python3
"""
Debug script to investigate canary validation issues.
"""

import sys
import json
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluation import ExpandedEvaluationSuite

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_canary_gates():
    """Debug the gate validation issues."""
    logger.info("Debugging canary validation issues")
    
    # Create minimal suite for testing
    suite = ExpandedEvaluationSuite(
        datasets=["infinitebench_qa"],
        budget_ratios=[0.15],
        K_values=[5],
        seeds=[1],
        adapter_filter=["last_k_turns_5", "bm25_lucene", "sliding_window_2048"],
        output_dir="debug_results"
    )
    
    # Run canary and capture detailed results
    canary_result = suite.run_canary_validation()
    
    print("=== CANARY VALIDATION DEBUG RESULTS ===")
    print(f"Success: {canary_result['success']}")
    print(f"Total evaluations: {canary_result.get('total_evaluations', 0)}")
    print(f"Failed gates: {canary_result.get('failed_gates', 0)}")
    
    # Print detailed gate results
    if 'gate_results' in canary_result:
        print("\nGate Results:")
        for gate_result in canary_result['gate_results']:
            status = gate_result['status']
            print(f"  {gate_result['gate_name']}: {status}")
            print(f"    Message: {gate_result['message']}")
            if 'details' in gate_result:
                print(f"    Details: {gate_result['details']}")
            print()
    
    # Check adapter validation details
    if 'adapter_validation' in canary_result:
        print("Adapter Validation:")
        adapter_val = canary_result['adapter_validation']
        for method_id, stats in adapter_val.items():
            success_rate = stats['success_rate']
            print(f"  {method_id}: {success_rate:.1%} success rate ({stats['valid_evaluations']}/{stats['total_evaluations']})")
            if stats['error_count'] > 0:
                print(f"    Errors: {stats['sample_errors']}")
    
    # Save detailed results for inspection
    debug_file = Path("debug_results") / "canary_debug.json"
    debug_file.parent.mkdir(exist_ok=True)
    with open(debug_file, 'w') as f:
        json.dump(canary_result, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {debug_file}")
    
    return canary_result

if __name__ == "__main__":
    debug_canary_gates()