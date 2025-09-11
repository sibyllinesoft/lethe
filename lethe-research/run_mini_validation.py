#!/usr/bin/env python3
"""
Mini validation test: Run 50 samples to verify non-zero accuracy
"""

import sys
import os
sys.path.append('benchmarks/infinitebench/src')

from eval_utils import get_results

def run_mini_validation():
    print("🔍 Running mini validation with 50 samples...")
    
    # Test with Code.Debug dataset, 50 samples max
    try:
        results = get_results(
            tasks=['code_debug'],
            max_samples=50,
            model_api='ollama'  # Use ollama API
        )
        
        if results:
            accuracies = []
            for result in results:
                if 'accuracy' in result:
                    accuracies.append(result['accuracy'])
                    print(f"Sample accuracy: {result['accuracy']:.3f}")
            
            if accuracies:
                avg_accuracy = sum(accuracies) / len(accuracies)
                print(f"\n🎯 Average accuracy: {avg_accuracy:.3f}")
                
                if avg_accuracy > 0:
                    print("✅ SUCCESS: Non-zero accuracy achieved!")
                    print("🚀 Ready for full matrix evaluation")
                    return True
                else:
                    print("❌ FAIL: Still getting zero accuracy")
                    return False
            else:
                print("❌ No accuracy results found")
                return False
        else:
            print("❌ No results returned")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_mini_validation()
    sys.exit(0 if success else 1)