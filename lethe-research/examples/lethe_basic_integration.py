#!/usr/bin/env python3
"""
Basic Lethe Integration Example

This script demonstrates how to integrate prompt monitoring into existing
Lethe research workflows with minimal changes to existing code.
"""

import time
import random
from datetime import datetime
from src.monitoring import LethePromptMonitor

def simulate_vector_search(query, k=10):
    """Simulate vector search results."""
    time.sleep(0.1)  # Simulate search time
    
    # Generate mock results
    results = []
    for i in range(k):
        results.append({
            "id": f"doc_{i}",
            "score": random.uniform(0.6, 0.95),
            "title": f"Document {i} about {query[:20]}...",
            "snippet": f"This document discusses {query.lower()} and related concepts."
        })
    
    return sorted(results, key=lambda x: x["score"], reverse=True)

def simulate_keyword_search(query, k=10):
    """Simulate keyword search results."""
    time.sleep(0.05)  # Simulate search time
    
    # Generate mock results
    results = []
    for i in range(k):
        results.append({
            "id": f"kw_doc_{i}",
            "score": random.uniform(0.5, 0.85),
            "title": f"Keyword Document {i} for {query[:15]}",
            "snippet": f"Keyword-based result for '{query}' with relevant content."
        })
    
    return sorted(results, key=lambda x: x["score"], reverse=True)

def hybrid_rank(vector_results, keyword_results, vector_weight=0.7, keyword_weight=0.3):
    """Combine vector and keyword search results with hybrid ranking."""
    
    # Normalize scores
    max_vector_score = max(r["score"] for r in vector_results) if vector_results else 1.0
    max_keyword_score = max(r["score"] for r in keyword_results) if keyword_results else 1.0
    
    # Create combined ranking
    combined_results = {}
    
    # Add vector results
    for result in vector_results:
        doc_id = result["id"]
        normalized_score = result["score"] / max_vector_score
        combined_results[doc_id] = {
            **result,
            "combined_score": normalized_score * vector_weight,
            "vector_score": result["score"],
            "keyword_score": 0.0
        }
    
    # Add keyword results
    for result in keyword_results:
        doc_id = result["id"]
        normalized_score = result["score"] / max_keyword_score
        
        if doc_id in combined_results:
            # Document found in both searches
            combined_results[doc_id]["combined_score"] += normalized_score * keyword_weight
            combined_results[doc_id]["keyword_score"] = result["score"]
        else:
            # Keyword-only document
            combined_results[doc_id] = {
                **result,
                "combined_score": normalized_score * keyword_weight,
                "vector_score": 0.0,
                "keyword_score": result["score"]
            }
    
    # Sort by combined score
    ranked_results = sorted(
        combined_results.values(),
        key=lambda x: x["combined_score"],
        reverse=True
    )
    
    return ranked_results

def calculate_retrieval_relevance(query, results):
    """Calculate relevance score for retrieval results."""
    if not results:
        return 0.0
    
    # Simple relevance calculation based on:
    # - Number of results
    # - Average scores
    # - Coverage (both vector and keyword results)
    
    avg_combined_score = sum(r["combined_score"] for r in results) / len(results)
    
    # Bonus for having both vector and keyword matches
    mixed_results = sum(1 for r in results if r["vector_score"] > 0 and r["keyword_score"] > 0)
    diversity_bonus = min(0.1, mixed_results / len(results) * 0.1)
    
    # Penalty for too few results
    result_count_factor = min(1.0, len(results) / 10.0)
    
    relevance_score = (avg_combined_score + diversity_bonus) * result_count_factor
    
    return min(1.0, relevance_score)

def enhanced_retrieval_workflow(query, retrieval_config):
    """Enhanced retrieval workflow with integrated monitoring."""
    
    # Initialize the monitor
    monitor = LethePromptMonitor(
        experiment_name="retrieval_optimization",
        project_id="lethe_research_2024"
    )
    
    # Start monitoring the retrieval process
    with monitor.track_prompt_execution(
        prompt_id="hybrid_retrieval",
        prompt_text=query,
        model_config=retrieval_config,
        stage="retrieval",
        tags=["hybrid", "retrieval", "lethe-integration"]
    ) as execution:
        
        print(f"🔍 Processing query: '{query}'")
        
        # Perform vector search
        print("  📊 Running vector search...")
        vector_results = simulate_vector_search(query, k=retrieval_config.get("vector_k", 10))
        
        # Perform keyword search
        print("  🔤 Running keyword search...")
        keyword_results = simulate_keyword_search(query, k=retrieval_config.get("keyword_k", 10))
        
        # Hybrid ranking
        print("  🔄 Performing hybrid ranking...")
        ranked_results = hybrid_rank(
            vector_results, 
            keyword_results,
            vector_weight=retrieval_config.get("vector_weight", 0.7),
            keyword_weight=retrieval_config.get("keyword_weight", 0.3)
        )
        
        # Take top N results
        final_results = ranked_results[:retrieval_config.get("final_k", 5)]
        
        # Update monitoring with results
        execution.response_text = f"Retrieved {len(final_results)} results using hybrid approach"
        execution.metadata.update({
            "vector_results_count": len(vector_results),
            "keyword_results_count": len(keyword_results),
            "final_results_count": len(final_results),
            "retrieval_method": "hybrid",
            "config": retrieval_config
        })
        
        # Calculate and update relevance score
        relevance_score = calculate_retrieval_relevance(query, final_results)
        execution.response_quality_score = relevance_score
        execution.tokens_used = len(query.split())  # Simulate token usage
        execution.success = True
        
        print(f"  ✅ Retrieved {len(final_results)} results (relevance: {relevance_score:.3f})")
        
        return final_results

def batch_retrieval_experiment(queries, base_config):
    """Run batch retrieval experiments with different configurations."""
    
    print("🧪 Batch Retrieval Experiment")
    print("=" * 40)
    
    # Different configuration variants to test
    config_variants = [
        {"name": "vector_heavy", "vector_weight": 0.8, "keyword_weight": 0.2},
        {"name": "balanced", "vector_weight": 0.6, "keyword_weight": 0.4},
        {"name": "keyword_heavy", "vector_weight": 0.4, "keyword_weight": 0.6},
    ]
    
    results = []
    
    for variant in config_variants:
        print(f"\n📋 Testing configuration: {variant['name']}")
        
        # Merge base config with variant
        test_config = {**base_config, **variant}
        
        variant_results = []
        
        for i, query in enumerate(queries):
            print(f"  Query {i+1}: {query[:50]}...")
            
            retrieval_results = enhanced_retrieval_workflow(query, test_config)
            
            variant_results.append({
                "query": query,
                "results_count": len(retrieval_results),
                "top_score": retrieval_results[0]["combined_score"] if retrieval_results else 0.0,
                "avg_score": sum(r["combined_score"] for r in retrieval_results) / len(retrieval_results) if retrieval_results else 0.0
            })
        
        # Calculate variant summary
        avg_results_count = sum(r["results_count"] for r in variant_results) / len(variant_results)
        avg_top_score = sum(r["top_score"] for r in variant_results) / len(variant_results)
        avg_avg_score = sum(r["avg_score"] for r in variant_results) / len(variant_results)
        
        results.append({
            "config": variant,
            "avg_results_count": avg_results_count,
            "avg_top_score": avg_top_score,
            "avg_avg_score": avg_avg_score,
            "individual_results": variant_results
        })
        
        print(f"    📊 Summary: {avg_results_count:.1f} results avg, {avg_top_score:.3f} top score avg")
    
    return results

def analyze_experiment_results(results):
    """Analyze batch experiment results."""
    
    print("\n📈 Experiment Analysis")
    print("=" * 30)
    
    # Compare configurations
    print("Configuration Comparison:")
    for result in results:
        config_name = result["config"]["name"]
        vector_weight = result["config"]["vector_weight"]
        keyword_weight = result["config"]["keyword_weight"]
        avg_score = result["avg_avg_score"]
        
        print(f"  {config_name}: V={vector_weight:.1f}, K={keyword_weight:.1f} → Avg Score: {avg_score:.3f}")
    
    # Find best configuration
    best_config = max(results, key=lambda x: x["avg_avg_score"])
    print(f"\n🏆 Best Configuration: {best_config['config']['name']}")
    print(f"   Vector Weight: {best_config['config']['vector_weight']}")
    print(f"   Keyword Weight: {best_config['config']['keyword_weight']}")
    print(f"   Average Score: {best_config['avg_avg_score']:.3f}")
    
    return best_config

def main():
    """Main function demonstrating Lethe integration."""
    
    print("🚀 Lethe Basic Integration Example")
    print("=" * 50)
    
    # Single query example
    print("\n📝 Single Query Example:")
    
    query = "What are the latest developments in hybrid retrieval systems?"
    config = {
        "vector_model": "text-embedding-3-large",
        "vector_k": 10,
        "keyword_k": 8,
        "final_k": 5,
        "vector_weight": 0.7,
        "keyword_weight": 0.3
    }
    
    results = enhanced_retrieval_workflow(query, config)
    
    print(f"\n🎯 Results for: '{query}'")
    for i, result in enumerate(results[:3], 1):
        print(f"  {i}. {result['title']} (score: {result['combined_score']:.3f})")
    
    # Batch experiment example
    print(f"\n📝 Batch Experiment Example:")
    
    test_queries = [
        "How do hybrid retrieval systems compare to pure vector search?",
        "What are the advantages of combining keyword and semantic search?",
        "Which retrieval method works best for academic research?",
        "How to optimize retrieval performance for domain-specific queries?"
    ]
    
    base_config = {
        "vector_model": "text-embedding-3-large",
        "vector_k": 12,
        "keyword_k": 10,
        "final_k": 6
    }
    
    experiment_results = batch_retrieval_experiment(test_queries, base_config)
    best_config = analyze_experiment_results(experiment_results)
    
    print(f"\n💡 Recommendations:")
    print(f"  1. Use '{best_config['config']['name']}' configuration for similar queries")
    print(f"  2. Monitor results through the dashboard: python scripts/prompt_monitor.py dashboard")
    print(f"  3. Analyze trends with: python examples/quick_analytics.py")
    print(f"  4. Export data for further analysis: python scripts/prompt_monitor.py export")

if __name__ == "__main__":
    main()