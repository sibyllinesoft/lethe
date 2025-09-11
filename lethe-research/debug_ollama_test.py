#!/usr/bin/env python3
"""
URGENT: Test Ollama API to see if it's actually generating responses
"""

import requests
import json
import sys

def test_ollama_api():
    """Test direct Ollama API call"""
    print("🔍 Testing Ollama API directly...")
    
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": "gemma3:27b",
        "prompt": "What is the name of the function that has a bug in this code:\n\ndef repack_carchive():\n    print('bug here')\n    return None\n\nAnswer with just the function name:",
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"✅ Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Response keys: {list(data.keys())}")
            
            if 'response' in data:
                response_text = data['response'].strip()
                print(f"✅ Response: '{response_text}'")
                print(f"✅ Response length: {len(response_text)}")
                
                # Check if it contains the expected function name
                if 'repack_carchive' in response_text:
                    print("🎯 GOOD: Response contains expected function name")
                else:
                    print("❌ BAD: Response does not contain expected function name")
                    
                return response_text
            else:
                print("❌ No 'response' field in API response")
                print(f"Full response: {data}")
        else:
            print(f"❌ API call failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        
    return None

def test_evaluation_scoring():
    """Test the evaluation scoring logic"""
    print("\n🔍 Testing evaluation scoring logic...")
    
    # Test cases
    test_cases = [
        {
            "response": "repack_carchive",
            "expected": "repack_carchive", 
            "should_match": True
        },
        {
            "response": "The function repack_carchive has a bug",
            "expected": "repack_carchive",
            "should_match": True
        },
        {
            "response": "some_other_function",
            "expected": "repack_carchive", 
            "should_match": False
        }
    ]
    
    for i, test in enumerate(test_cases):
        # Simple substring match (what we probably need)
        simple_match = test["expected"] in test["response"]
        
        # Exact match (what might be failing)
        exact_match = test["response"].strip() == test["expected"]
        
        print(f"\nTest {i+1}:")
        print(f"  Response: '{test['response']}'")
        print(f"  Expected: '{test['expected']}'")
        print(f"  Simple match: {simple_match} (should be {test['should_match']})")
        print(f"  Exact match: {exact_match}")

if __name__ == "__main__":
    print("🚨 URGENT: Debugging evaluation accuracy=0.000 issue")
    
    # Test Ollama API
    response = test_ollama_api()
    
    # Test scoring logic
    test_evaluation_scoring()
    
    if response:
        print(f"\n🎯 Ollama is working. Response: '{response}'")
        print("🔍 Issue might be in evaluation scoring logic or data flow")
    else:
        print("\n❌ Ollama API not responding properly")
        print("🔍 This could be the root cause")