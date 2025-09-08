#!/usr/bin/env python3
"""
Simple test script to verify our real Lethe service works with the benchmark framework.
"""

import sys
import time
import json
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "benchmarks"))

from benchmarks.competitors.lethe_baseline import LetheHybridCompetitor


def test_real_lethe():
    """Test the real Lethe service using the benchmark competitor."""
    print("🚀 Testing Real Lethe Service Connection...")
    
    # Initialize the Lethe competitor
    competitor = LetheHybridCompetitor(
        name="lethe_real_test",
        api_endpoint="http://localhost:8094"
    )
    
    # Test queries from the LetheBench dataset
    test_queries = [
        {
            "query": "How do I migrate from SVN to Kubernetes with minimal user disruption?",
            "context": "This is a comprehensive research context about migration strategies, best practices, system migration, data transfer, compatibility assessment, rollback planning, monitoring and troubleshooting guidance for migrating systems with minimal user disruption in enterprise environments",
            "keep_ratio": 1.0
        },
        {
            "query": "How do I optimize Docker performance for transactional processing?",
            "context": "Performance optimization context with Docker configuration tuning memory allocation connection pooling cache size resource scaling horizontal vertical monitoring setup",
            "keep_ratio": 0.8
        }
    ]
    
    print(f"🔗 Connecting to Lethe service at: {competitor.api_endpoint}")
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n📝 Test Query {i}: {test_case['query'][:50]}...")
        
        start_time = time.time()
        
        try:
            # Call the competitor's retrieve method
            result = competitor.retrieve(
                query=test_case['query'],
                context=test_case['context'],
                keep_ratio=test_case['keep_ratio'],
                k=5
            )
            
            latency = (time.time() - start_time) * 1000
            
            print(f"✅ Success!")
            print(f"   📊 Results: {len(result.doc_ids)} documents")
            print(f"   📈 Scores: {[round(s, 3) for s in result.scores[:3]]}")
            print(f"   🔢 Tokens: {result.tokens_retrieved}")
            print(f"   ⏱️  Latency: {latency:.1f}ms")
            print(f"   🎯 Exact matches: {result.exact_matches}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            continue
    
    print(f"\n🎉 Real Lethe service test completed!")


if __name__ == "__main__":
    test_real_lethe()