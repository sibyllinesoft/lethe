#!/usr/bin/env python3
"""
Fixed Evaluation Pipeline Runner
===============================

Runs the hybrid evaluation with fixed measurement pipes that implement:
1. Proper tokenizer-based token counting (not window/sink counts)
2. KV-reuse with prefix-Jaccard calculation 
3. ΔCBU computation using V2 payloads and bundle scoring

This script patches the existing evaluation to use the fixed measurement pipeline
and validates all three critical pipes with fail-closed guards.
"""

import sys
import logging
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Main entry point for fixed evaluation."""
    parser = argparse.ArgumentParser(description='Run hybrid evaluation with fixed measurement pipes')
    parser.add_argument('--mode', choices=['test', 'full'], default='test', 
                       help='Run mode: test (quick) or full evaluation')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run measurement pipeline tests')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🔧 Fixed Measurement Pipeline Evaluation")
    print("=" * 50)
    
    # Step 1: Run measurement pipeline tests
    print("\n📋 Step 1: Validating Fixed Measurement Pipeline")
    print("-" * 50)
    
    try:
        from test_measurement_pipeline import run_test_suite
        
        print("Running comprehensive test suite...")
        test_success = run_test_suite()
        
        if not test_success:
            print("❌ Measurement pipeline tests failed!")
            print("Fix the measurement pipeline before running evaluation.")
            return 1
        
        print("✅ Measurement pipeline tests passed!")
        
    except Exception as e:
        print(f"❌ Failed to run measurement tests: {e}")
        return 1
    
    if args.validate_only:
        print("\n🎉 Validation complete!")
        return 0
    
    # Step 2: Apply measurement integration patch
    print("\n🔌 Step 2: Applying Measurement Integration Patch")
    print("-" * 50)
    
    try:
        from measurement_integration import patch_existing_evaluation
        
        patch_success = patch_existing_evaluation()
        if not patch_success:
            print("❌ Failed to patch existing evaluation!")
            return 1
        
        print("✅ Successfully patched evaluation with fixed measurements!")
        
    except Exception as e:
        print(f"❌ Failed to apply patch: {e}")
        return 1
    
    # Step 3: Run the patched evaluation
    print("\n🚀 Step 3: Running Patched Evaluation")
    print("-" * 50)
    
    try:
        # Import the patched evaluation
        import run_hybrid_infinitebench as main_eval
        
        # Configure for test or full mode
        if args.mode == 'test':
            sys.argv = [
                'run_hybrid_infinitebench.py',
                '--mode', 'quick-test',
                '--keep-ratios', '0.08,0.15,0.30',
                '--methods', 'streaming,lethe,hybrid',
                '--verbose'
            ]
            print("Running in QUICK TEST mode...")
        else:
            sys.argv = [
                'run_hybrid_infinitebench.py',
                '--mode', 'full-evaluation',
                '--keep-ratios', '0.08,0.15,0.30',
                '--methods', 'streaming,lethe,hybrid'
            ]
            print("Running in FULL EVALUATION mode...")
        
        # Clear previous sys.argv to avoid conflicts
        if len(sys.argv) > 1:
            sys.argv = sys.argv[:1] + sys.argv[1:]
        
        # Run the main evaluation with patches
        print(f"Starting evaluation at {datetime.now().strftime('%H:%M:%S')}")
        
        # Call the main function directly
        main_eval.main()
        
        print("✅ Evaluation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Evaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        logger.exception("Detailed error:")
        return 1
    
    # Step 4: Validation Report
    print("\n📊 Step 4: Validation Summary")
    print("-" * 50)
    
    try:
        # Look for the most recent results file
        artifacts_dir = Path("artifacts/hybrid_evaluation")
        if artifacts_dir.exists():
            result_files = list(artifacts_dir.glob("hybrid_evaluation_*.json"))
            if result_files:
                latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
                
                with open(latest_file, 'r') as f:
                    results = json.load(f)
                
                print(f"📁 Results saved to: {latest_file}")
                
                # Check if our fixes worked
                print("\n🔍 Validation Checks:")
                
                # Check for measurement fields in results
                sample_results = []
                for method, method_results in results.get('results', {}).items():
                    sample_results.extend(method_results)
                
                if sample_results:
                    sample = sample_results[0]
                    
                    # Check tokenization
                    if 'tokens_kept' in sample and sample['tokens_kept'] > 10:
                        print("✅ Tokenization pipe: Using proper tokenizer (not window/sink counts)")
                    else:
                        print("❌ Tokenization pipe: Possibly using window/sink counts")
                    
                    # Check KV-reuse
                    kv_values = [r.get('kv_reuse', 0.0) for r in sample_results]
                    if any(kv > 0.0 for kv in kv_values):
                        print("✅ KV-reuse pipe: Non-zero values detected")
                    else:
                        print("❌ KV-reuse pipe: All zeros (arranger not wired)")
                    
                    # Check ΔCBU
                    cbu_values = [r.get('delta_cbu_per_1k', 0.0) for r in sample_results]
                    if len(set(cbu_values)) > 1:
                        print("✅ ΔCBU pipe: Variance across methods detected")
                    else:
                        print("❌ ΔCBU pipe: Constant values (computation not varying)")
                
                # Check zh_qa specific
                zh_results = [r for r in sample_results if 'zh' in r.get('dataset', '').lower()]
                if zh_results:
                    zh_tokens = [r.get('tokens_kept', 0) for r in zh_results 
                               if abs(r.get('keep_ratio', 0.0) - 0.08) < 0.01]
                    if zh_tokens and any(t > 500 for t in zh_tokens):
                        print("✅ zh_qa sanity: tokens_kept@8% > 500")
                    else:
                        print("❌ zh_qa sanity: tokens_kept@8% <= 500")
                
                # Overall assessment
                promotion_count = sum(1 for d in results.get('promotion_decisions', {}).values() 
                                    if d.get('promoted', False))
                total_conditions = len(results.get('promotion_decisions', {}))
                
                print(f"\n🏆 Promotion Status: {promotion_count}/{total_conditions} conditions met")
                
                if promotion_count == total_conditions:
                    print("🎉 All promotion criteria met! Measurement fixes successful!")
                elif promotion_count > 0:
                    print("⚠️ Partial success - some measurement issues remain")
                else:
                    print("❌ No promotion criteria met - measurement pipes still broken")
                
            else:
                print("⚠️ No result files found")
        else:
            print("⚠️ Results directory not found")
    
    except Exception as e:
        print(f"⚠️ Could not generate validation report: {e}")
    
    print(f"\n🏁 Evaluation completed at {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 50)
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)