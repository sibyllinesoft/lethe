#!/usr/bin/env python3
"""
Quick accuracy test to verify fix
"""

import requests
import json

def test_mini_evaluation():
    print("🔍 Quick accuracy test...")
    
    # Test with a simple code debug scenario
    test_cases = [
        {
            "code": """def repack_carchive():
    print('this has a bug')
    return None

def normal_function():
    return True""",
            "expected": "repack_carchive"
        },
        {
            "code": """def safe_function():
    return "ok"

def buggy_parser():
    # obvious bug here
    return 1/0""",
            "expected": "buggy_parser"
        }
    ]
    
    correct = 0
    total = len(test_cases)
    
    for i, case in enumerate(test_cases):
        prompt = f"""You are evaluating code to find bugs. Which function contains a bug in this code:

{case['code']}

Answer with only the function name that has the bug:"""

        payload = {
            "model": "gemma3:27b",
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                answer = data['response'].strip()
                print(f"Test {i+1}: Expected '{case['expected']}', Got '{answer}'")
                
                if case['expected'] in answer:
                    correct += 1
                    print("  ✅ CORRECT")
                else:
                    print("  ❌ INCORRECT")
            else:
                print(f"  ❌ API Error: {response.status_code}")
        except Exception as e:
            print(f"  ❌ Exception: {e}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n🎯 Quick test accuracy: {accuracy:.1%} ({correct}/{total})")
    
    if accuracy > 0:
        print("✅ SUCCESS: Non-zero accuracy confirmed!")
        print("🚀 Model fix verified - ready for full evaluation")
        return True
    else:
        print("❌ FAIL: Still getting zero accuracy")
        return False

if __name__ == "__main__":
    test_mini_evaluation()