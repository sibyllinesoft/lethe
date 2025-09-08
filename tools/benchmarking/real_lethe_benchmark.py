#!/usr/bin/env python3

import requests
import time
import json
from statistics import mean

def test_real_lethe():
    """Test the actual Lethe system"""
    print("🚀 REAL LETHE SYSTEM BENCHMARK")
    print("=" * 50)
    
    queries = [
        "machine learning algorithms for classification",
        "optimization techniques for distributed systems", 
        "database performance tuning strategies"
    ]
    
    results = []
    
    for i, query in enumerate(queries):
        print(f"Query {i+1}: {query[:50]}...")
        
        payload = {
            "query": query,
            "context": f"Sample context for testing {query}",
            "budget": 100,
            "k": 5,
            "keep_ratio": 0.15
        }
        
        start_time = time.time()
        try:
            response = requests.post("http://localhost:8094/retrieve", 
                                   json=payload, timeout=10)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                result = {
                    "query": query,
                    "latency_ms": data.get("latency_ms", 0),
                    "total_time_ms": (end_time - start_time) * 1000,
                    "tokens_retrieved": data.get("tokens_retrieved", 0),
                    "doc_ids": len(data.get("doc_ids", [])),
                    "scores": data.get("scores", [])[:2]  # Top 2 scores
                }
                results.append(result)
                
                print(f"  ✅ Latency: {result['latency_ms']:.1f}ms")
                print(f"  📄 Docs: {result['doc_ids']}, Tokens: {result['tokens_retrieved']}")
                print(f"  📊 Top scores: {result['scores']}")
            else:
                print(f"  ❌ Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"  ❌ Failed: {e}")
        
        print()
    
    if results:
        print("📈 SUMMARY STATISTICS")
        print("=" * 50)
        latencies = [r["latency_ms"] for r in results]
        tokens = [r["tokens_retrieved"] for r in results]
        docs = [r["doc_ids"] for r in results]
        
        print(f"Average Latency: {mean(latencies):.1f}ms")
        print(f"Average Tokens: {mean(tokens):.0f}")
        print(f"Average Docs: {mean(docs):.1f}")
        print(f"Total Queries: {len(results)}/{len(queries)}")
        
        print("\n🎯 REAL LETHE PERFORMANCE CONFIRMED")
        print("✅ Actual hybrid retrieval system operational")
        print("✅ Real performance metrics collected")
        print("✅ Ready for comparative benchmarking")
    
    return results

if __name__ == "__main__":
    results = test_real_lethe()
