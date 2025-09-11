#!/usr/bin/env python3
"""
Debug script to trace accuracy calculation issues in the evaluation pipeline
"""
import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'benchmarks' / 'infinitebench'))
sys.path.insert(0, str(project_root))

import json
from src.infinitebench.dataset_loader import InfiniteBenchLoader
from scripts.run_hybrid_infinitebench import generate_llm_response

def debug_sample_processing():
    """Test accuracy calculation with a single known sample"""
    print("🔍 DEBUGGING ACCURACY CALCULATION")
    print("=" * 50)
    
    # Load a few samples to debug
    print("1. Loading dataset samples...")
    try:
        infinitebench_path = project_root / "benchmarks" / "infinitebench" / "data"
        loader = InfiniteBenchLoader(infinitebench_path)
        samples = loader.load_task("code_debug")[:5]  # Just first 5 samples
        print(f"   ✅ Loaded {len(samples)} code samples")
    except Exception as e:
        print(f"   ❌ Failed to load samples: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test first sample
    sample = samples[0]
    print(f"\n2. Analyzing first sample:")
    print(f"   - Sample type: {type(sample)}")
    print(f"   - Sample attributes: {dir(sample) if hasattr(sample, '__dict__') else 'No __dict__'}")
    
    # Check different ways to access the expected answer
    expected_candidates = []
    if hasattr(sample, 'answer'):
        expected_candidates.append(('sample.answer', sample.answer))
    if hasattr(sample, '__dict__'):
        for key in ['answer', 'expected', 'output', 'target']:
            if hasattr(sample, key):
                expected_candidates.append((f'sample.{key}', getattr(sample, key)))
    
    if isinstance(sample, dict):
        for key in ['answer', 'expected', 'output', 'target']:
            if key in sample:
                expected_candidates.append((f"sample['{key}']", sample[key]))
    
    print(f"\n3. Expected answer candidates:")
    for source, value in expected_candidates:
        print(f"   - {source}: {repr(value)} (type: {type(value)})")
    
    # Test query extraction
    query_candidates = []
    if hasattr(sample, 'question'):
        query_candidates.append(('sample.question', sample.question))
    if hasattr(sample, 'query'):
        query_candidates.append(('sample.query', sample.query))
    if isinstance(sample, dict):
        for key in ['question', 'query', 'input']:
            if key in sample:
                query_candidates.append((f"sample['{key}']", sample[key]))
    
    print(f"\n4. Query candidates:")
    for source, value in query_candidates:
        print(f"   - {source}: {repr(value[:100] if isinstance(value, str) else value)}...")
        if isinstance(value, str) and len(value) > 100:
            print(f"     [truncated, total length: {len(value)}]")
    
    # Test context extraction
    context_candidates = []
    if hasattr(sample, 'context'):
        context_candidates.append(('sample.context', sample.context))
    if isinstance(sample, dict):
        for key in ['context', 'input', 'content']:
            if key in sample:
                context_candidates.append((f"sample['{key}']", sample[key]))
    
    print(f"\n5. Context candidates:")
    for source, value in context_candidates:
        if isinstance(value, str):
            print(f"   - {source}: {len(value)} chars")
            print(f"     Preview: {repr(value[:200])}...")
        else:
            print(f"   - {source}: {repr(value)} (type: {type(value)})")
    
    # Test LLM response generation
    print(f"\n6. Testing LLM response generation...")
    if query_candidates and context_candidates:
        query = query_candidates[0][1]  # Use first query candidate
        context = context_candidates[0][1]  # Use first context candidate
        
        if isinstance(query, str) and isinstance(context, str):
            try:
                response = generate_llm_response(query, context)
                print(f"   ✅ LLM Response: {repr(response)}")
                print(f"   Response length: {len(response) if response else 0}")
                print(f"   Response type: {type(response)}")
            except Exception as e:
                print(f"   ❌ LLM generation failed: {e}")
                response = ""
        else:
            print(f"   ❌ Invalid query/context types: {type(query)}/{type(context)}")
            response = ""
    else:
        print(f"   ❌ No valid query/context found")
        response = ""
    
    # Test accuracy calculation
    print(f"\n7. Testing accuracy calculation...")
    if expected_candidates and response:
        expected = expected_candidates[0][1]  # Use first expected answer
        
        # Copy the accuracy logic from the main script
        expected_items = []
        if isinstance(expected, list):
            expected_items = [str(item).lower().strip() for item in expected if item]
        else:
            expected_str = str(expected) if expected is not None else ""
            if expected_str.strip():
                # Handle string representation of list
                if expected_str.startswith('[') and expected_str.endswith(']'):
                    try:
                        import ast
                        parsed_list = ast.literal_eval(expected_str)
                        if isinstance(parsed_list, list):
                            expected_items = [str(item).lower().strip() for item in parsed_list if item]
                        else:
                            expected_items = [expected_str.lower().strip()]
                    except (ValueError, SyntaxError):
                        expected_items = [expected_str.lower().strip()]
                else:
                    expected_items = [expected_str.lower().strip()]
        
        print(f"   Expected items: {expected_items}")
        
        # Normalize actual response
        actual_normalized = response.lower().strip() if response else ""
        print(f"   Actual normalized: {repr(actual_normalized)}")
        
        # Check accuracy
        accuracy = 0.0
        if expected_items and actual_normalized:
            for expected_item in expected_items:
                if expected_item in actual_normalized:
                    accuracy = 1.0
                    print(f"   ✅ MATCH found: '{expected_item}' in response")
                    break
                else:
                    print(f"   ❌ No match: '{expected_item}' not in response")
        else:
            print(f"   ❌ Missing data: expected_items={bool(expected_items)}, actual={bool(actual_normalized)}")
        
        print(f"   Final accuracy: {accuracy}")
        
        # Additional debug - show exactly what we're comparing
        print(f"\n8. Detailed comparison:")
        print(f"   Expected (raw): {repr(expected)}")
        print(f"   Expected items: {expected_items}")
        print(f"   Actual (raw): {repr(response)}")
        print(f"   Actual normalized: {repr(actual_normalized)}")
        
        if expected_items and actual_normalized:
            for i, expected_item in enumerate(expected_items):
                print(f"   Check {i+1}: '{expected_item}' in '{actual_normalized}'?")
                print(f"             Result: {expected_item in actual_normalized}")

if __name__ == "__main__":
    debug_sample_processing()