#!/usr/bin/env python3
"""
Simple test script to verify the improved LLM generation function works
"""
import sys
import logging
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the improved function
from scripts.run_hybrid_infinitebench import generate_llm_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_code_debug_task():
    """Test the improved code debug functionality"""
    
    # Example from InfiniteBench - simplified
    query = "Which funtion has deliberate error?"
    
    # Mock context with function that has error
    context = """
def repack_carchive(filename):
    '''Repack a pyarmor archive file'''
    try:
        import struct
        import tempfile
        import os
        
        # This function has a deliberate error - missing return statement
        with open(filename, 'rb') as f:
            data = f.read()
        
        # Process data
        processed = data[10:]  # Skip header
        
        # DELIBERATE ERROR: Missing return statement
        # Should return processed data
        
    except Exception as e:
        return None

def working_function():
    '''This function works correctly'''
    return "Hello World"

def another_function():
    '''Another working function'''
    x = 1 + 2
    return x * 3
    """
    
    expected_answer = ["repack_carchive"]
    
    logger.info("Testing improved code debug task...")
    logger.info(f"Query: {query}")
    logger.info(f"Expected: {expected_answer}")
    
    response = generate_llm_response(query, context)
    
    logger.info(f"LLM Response: '{response}'")
    
    # Test matching logic
    expected_items = [str(item).lower().strip() for item in expected_answer]
    actual_normalized = response.lower().strip() if response else ""
    actual_normalized = actual_normalized.replace('"', '').replace("'", '').replace("`", "")
    actual_normalized = actual_normalized.replace("function ", "").replace("def ", "")
    actual_normalized = actual_normalized.split("(")[0]
    actual_normalized = actual_normalized.split(":")[0]
    actual_normalized = actual_normalized.split()[0] if actual_normalized.split() else actual_normalized
    
    logger.info(f"Normalized response: '{actual_normalized}'")
    logger.info(f"Expected items: {expected_items}")
    
    # Check accuracy
    accuracy = 0.0
    if expected_items and actual_normalized:
        for expected_item in expected_items:
            if expected_item == actual_normalized:
                accuracy = 1.0
                break
        
        if accuracy == 0.0:
            for expected_item in expected_items:
                if expected_item and len(expected_item) > 2:
                    if expected_item in actual_normalized or actual_normalized in expected_item:
                        accuracy = 0.8
                        break
    
    logger.info(f"Accuracy: {accuracy}")
    
    return accuracy > 0

def test_code_run_task():
    """Test code execution task"""
    query = "What will be the output when calling test_function(3)?"
    
    context = """
def test_function(x):
    result = x * 2 + 5
    return result
    """
    
    expected = "11"
    
    logger.info("Testing code run task...")
    logger.info(f"Query: {query}")
    logger.info(f"Expected: {expected}")
    
    response = generate_llm_response(query, context)
    logger.info(f"LLM Response: '{response}'")
    
    return expected.strip() in response.strip()

def test_zh_qa_task():
    """Test Chinese QA task"""
    query = "作者的名字是什么？"
    
    context = """
这本书是由张三写的。张三是一位著名的作家，他写了很多畅销书。
"""
    
    expected = "张三"
    
    logger.info("Testing Chinese QA task...")
    logger.info(f"Query: {query}")
    logger.info(f"Expected: {expected}")
    
    response = generate_llm_response(query, context)
    logger.info(f"LLM Response: '{response}'")
    
    return expected in response

if __name__ == "__main__":
    logger.info("🧪 Testing improved LLM generation function...")
    
    results = []
    
    try:
        results.append(("Code Debug", test_code_debug_task()))
    except Exception as e:
        logger.error(f"Code Debug test failed: {e}")
        results.append(("Code Debug", False))
    
    try:
        results.append(("Code Run", test_code_run_task()))
    except Exception as e:
        logger.error(f"Code Run test failed: {e}")
        results.append(("Code Run", False))
    
    try:
        results.append(("Chinese QA", test_zh_qa_task()))
    except Exception as e:
        logger.error(f"Chinese QA test failed: {e}")
        results.append(("Chinese QA", False))
    
    logger.info("\n📊 Test Results:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! LLM generation improvements working.")
    else:
        logger.warning("⚠️ Some tests failed. Check implementation.")