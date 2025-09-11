#!/usr/bin/env python3
"""
Debug script to investigate zero accuracy issue.
Tests a single code_debug sample to see what responses are generated.
"""
import sys
import json
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.infinitebench.dataset_loader import InfiniteBenchLoader

def test_single_sample():
    """Test what happens with a single sample"""
    print("🔍 Loading single code_debug sample...")
    
    # Load dataset
    infinitebench_path = project_root / "benchmarks" / "infinitebench" / "data"
    loader = InfiniteBenchLoader(infinitebench_path)
    
    # Load first sample
    samples = loader.load_task('code_debug')
    if not samples:
        print("❌ No samples loaded!")
        return
    
    sample = samples[0]
    print(f"📋 Sample ID: {getattr(sample, 'id', 'N/A')}")
    
    # Check available attributes
    attrs = [attr for attr in dir(sample) if not attr.startswith('_')]
    print(f"📝 Available attributes: {attrs}")
    
    # Try different attribute names
    query = getattr(sample, 'input', None) or getattr(sample, 'question', None) or getattr(sample, 'query', None)
    answer = getattr(sample, 'answer', None) or getattr(sample, 'expected', None) or getattr(sample, 'output', None)
    context = getattr(sample, 'context', None) or getattr(sample, 'input', None)
    
    print(f"❓ Question: {query}")
    print(f"✅ Expected Answer: {answer}")
    print(f"📊 Context Length: {len(context) if context else 0} characters")
    
    # Test retrieval without LLM generation
    from scripts.run_hybrid_infinitebench import StreamingLLMBaseline
    
    competitor = StreamingLLMBaseline({})
    competitor.initialize()
    
    print("\n🔧 Testing retrieval process...")
    retrieval_result = competitor.retrieve(
        query=query,
        context=context, 
        max_tokens=4000
    )
    
    print(f"📤 Retrieval Response: '{retrieval_result.response}'")
    print(f"📝 Context Used Length: {len(retrieval_result.context_used)} chars")
    print(f"⚡ Processing Time: {retrieval_result.processing_time_ms:.1f}ms")
    
    # Analyze the matching logic - handle list format properly
    if isinstance(answer, list):
        expected_items = [str(item).lower().strip() for item in answer if item]
    else:
        expected_items = [str(answer).lower().strip()] if answer else []
    actual_normalized = retrieval_result.response.lower().strip() if retrieval_result.response else ""
    
    print(f"\n🎯 Matching Analysis:")
    print(f"   Expected items: {expected_items}")
    print(f"   Actual response: '{actual_normalized}'")
    print(f"   Match found: {any(expected_item in actual_normalized for expected_item in expected_items)}")
    
    # The key insight: NO LLM IS BEING CALLED!
    print(f"\n🚨 ROOT CAUSE IDENTIFIED:")
    print(f"   • Retrieval systems only select relevant context chunks")
    print(f"   • No LLM is called to generate actual answers")
    print(f"   • Response field is empty: '{retrieval_result.response}'")
    print(f"   • This causes 100% accuracy failure across all methods!")

if __name__ == '__main__':
    test_single_sample()