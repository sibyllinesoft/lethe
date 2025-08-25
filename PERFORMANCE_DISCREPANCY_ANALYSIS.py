#!/usr/bin/env python3
"""
CRITICAL ANALYSIS: Lethe Performance Claims Discrepancy Investigation
====================================================================

URGENT ISSUE: README claims 0.837 F1 score for BM25+Vector method
ACTUAL RESULTS: Need to validate what the measured NDCG@10 actually is

This script investigates the root cause of this discrepancy.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Add the research directory to path
research_dir = Path('/home/nathan/Projects/lethe/lethe-research')
sys.path.append(str(research_dir))

from analysis.metrics import MetricsCalculator, load_results_from_json

def analyze_performance_discrepancy():
    """Investigate the performance claims discrepancy"""
    
    print("=" * 80)
    print("CRITICAL ANALYSIS: Performance Claims Discrepancy Investigation")  
    print("=" * 80)
    
    # Load the BM25+Vector results (the method claimed to have 0.837 F1)
    results_file = research_dir / "artifacts/20250823_022745/baseline_results/bm25_vector_simple_results.json"
    
    if not results_file.exists():
        print(f"ERROR: Results file not found: {results_file}")
        return
    
    print(f"Loading results from: {results_file}")
    results = load_results_from_json(str(results_file))
    
    print(f"Loaded {len(results)} query results")
    print()
    
    # Initialize metrics calculator  
    calculator = MetricsCalculator()
    
    # Compute comprehensive metrics
    print("Computing comprehensive metrics...")
    metrics = calculator.compute_all_metrics(results, "bm25_vector_simple")
    
    print("\n" + "="*60)
    print("PERFORMANCE METRICS ANALYSIS")
    print("="*60)
    
    # Display key metrics
    print(f"Configuration: {metrics.config_name}")
    print(f"Number of queries: {metrics.n_queries}")
    print()
    
    print("QUALITY METRICS:")
    print(f"  NDCG@5:  {metrics.ndcg_at_k.get(5, 'N/A'):.4f}")
    print(f"  NDCG@10: {metrics.ndcg_at_k.get(10, 'N/A'):.4f}")  
    print(f"  NDCG@20: {metrics.ndcg_at_k.get(20, 'N/A'):.4f}")
    print()
    
    print("PRECISION/RECALL METRICS:")
    print(f"  Precision@5:  {metrics.precision_at_k.get(5, 'N/A'):.4f}")
    print(f"  Precision@10: {metrics.precision_at_k.get(10, 'N/A'):.4f}")
    print(f"  Recall@5:     {metrics.recall_at_k.get(5, 'N/A'):.4f}")
    print(f"  Recall@10:    {metrics.recall_at_k.get(10, 'N/A'):.4f}")
    print(f"  Recall@20:    {metrics.recall_at_k.get(20, 'N/A'):.4f}")
    print()
    
    print("RANKING METRICS:")
    print(f"  MRR@10: {metrics.mrr_at_k.get(10, 'N/A'):.4f}")
    print()
    
    # Investigate F1 score calculation
    print("F1 SCORE INVESTIGATION:")
    print("=" * 40)
    
    # Manual F1 calculation at different k values
    for k in [5, 10, 20]:
        precision_k = metrics.precision_at_k.get(k, 0)
        recall_k = metrics.recall_at_k.get(k, 0)
        
        if precision_k + recall_k > 0:
            f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
        else:
            f1_k = 0.0
            
        print(f"  F1@{k}: {f1_k:.4f} (P={precision_k:.4f}, R={recall_k:.4f})")
    
    print()
    
    # Analyze individual query performance
    print("INDIVIDUAL QUERY ANALYSIS (First 5 queries):")
    print("=" * 50)
    
    for i, result in enumerate(results[:5]):
        print(f"\nQuery {i+1} ({result.query_id}):")
        print(f"  Domain: {result.domain}")
        print(f"  Ground truth docs: {len(result.ground_truth_docs)}")
        print(f"  Retrieved docs: {len(result.retrieved_docs)}")
        
        # Calculate precision@10 for this query
        retrieved_at_10 = set(result.retrieved_docs[:10])
        relevant_docs = set(result.ground_truth_docs)
        
        hits = len(retrieved_at_10 & relevant_docs)
        precision_10 = hits / min(10, len(result.retrieved_docs)) if result.retrieved_docs else 0
        recall_10 = hits / len(relevant_docs) if relevant_docs else 0
        
        print(f"  Hits@10: {hits}")
        print(f"  Precision@10: {precision_10:.4f}")
        print(f"  Recall@10: {recall_10:.4f}")
        
        if precision_10 + recall_10 > 0:
            f1_10 = 2 * (precision_10 * recall_10) / (precision_10 + recall_10)
            print(f"  F1@10: {f1_10:.4f}")
        else:
            print(f"  F1@10: 0.0000")
    
    print("\n" + "="*60)
    print("DISCREPANCY ANALYSIS")
    print("="*60)
    
    # Compare claimed vs measured performance
    claimed_f1 = 0.837
    measured_f1_10 = metrics.precision_at_k.get(10, 0)
    measured_recall_10 = metrics.recall_at_k.get(10, 0) 
    
    if measured_f1_10 + measured_recall_10 > 0:
        actual_f1_10 = 2 * (measured_f1_10 * measured_recall_10) / (measured_f1_10 + measured_recall_10)
    else:
        actual_f1_10 = 0.0
    
    measured_ndcg_10 = metrics.ndcg_at_k.get(10, 0)
    
    print(f"CLAIMED Performance (from README):")
    print(f"  BM25+Vector F1: {claimed_f1:.3f}")
    print()
    
    print(f"MEASURED Performance (actual results):")
    print(f"  F1@10: {actual_f1_10:.4f}")
    print(f"  NDCG@10: {measured_ndcg_10:.4f}")
    print()
    
    print(f"DISCREPANCY:")
    print(f"  F1 difference: {abs(claimed_f1 - actual_f1_10):.4f}")
    print(f"  Relative error: {abs(claimed_f1 - actual_f1_10)/claimed_f1*100:.1f}%")
    
    if abs(claimed_f1 - actual_f1_10) > 0.1:
        print("  ⚠️  MAJOR DISCREPANCY DETECTED!")
        print("     The claimed F1 score is significantly different from measured results.")
    
    print("\n" + "="*60)
    print("ROOT CAUSE INVESTIGATION")
    print("="*60)
    
    # Check if this is a domain-specific issue
    domain_performance = {}
    for result in results:
        domain = result.domain
        if domain not in domain_performance:
            domain_performance[domain] = []
        
        # Calculate F1@10 for this query
        retrieved_at_10 = set(result.retrieved_docs[:10])
        relevant_docs = set(result.ground_truth_docs)
        
        hits = len(retrieved_at_10 & relevant_docs)
        precision_10 = hits / min(10, len(result.retrieved_docs)) if result.retrieved_docs else 0
        recall_10 = hits / len(relevant_docs) if relevant_docs else 0
        
        if precision_10 + recall_10 > 0:
            f1_10 = 2 * (precision_10 * recall_10) / (precision_10 + recall_10)
        else:
            f1_10 = 0.0
            
        domain_performance[domain].append(f1_10)
    
    print("Performance by domain:")
    for domain, scores in domain_performance.items():
        avg_f1 = np.mean(scores) if scores else 0
        print(f"  {domain}: F1@10 = {avg_f1:.4f} ({len(scores)} queries)")
    
    # Check if results file represents all expected domains
    print(f"\nDomains in results: {list(domain_performance.keys())}")
    print(f"Expected domains: ['code', 'prose', 'tool']")
    
    if not all(domain in domain_performance for domain in ['code', 'prose', 'tool']):
        print("  ⚠️  MISSING DOMAINS! Results may not include all expected domains.")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    recommendations = []
    
    if abs(claimed_f1 - actual_f1_10) > 0.1:
        recommendations.append("1. IMMEDIATE: Update README with correct performance numbers")
        recommendations.append("2. Investigate if claimed numbers are from a different evaluation")
        recommendations.append("3. Check if F1 calculation method differs from implementation")
    
    if measured_ndcg_10 < 0.5:
        recommendations.append("4. Performance appears suboptimal - consider baseline tuning")
    
    if not all(domain in domain_performance for domain in ['code', 'prose', 'tool']):
        recommendations.append("5. Ensure evaluation includes all three target domains")
        
    if len(results) < 100:
        recommendations.append("6. Scale up dataset - current sample size may be insufficient")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"  {rec}")
    
    if not recommendations:
        print("  ✓ No immediate issues identified")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    
    print("1. Verify metric calculation methodology")
    print("2. Check if claimed numbers are from separate evaluation")
    print("3. Update README with accurate performance claims")
    print("4. Scale dataset to improve statistical power")
    print("5. Run comprehensive evaluation across all baselines")
    
    return {
        'claimed_f1': claimed_f1,
        'measured_f1_10': actual_f1_10,
        'measured_ndcg_10': measured_ndcg_10,
        'discrepancy': abs(claimed_f1 - actual_f1_10),
        'n_queries': len(results),
        'domain_performance': domain_performance,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    results = analyze_performance_discrepancy()
    
    print(f"\nAnalysis complete. Critical discrepancy of {results['discrepancy']:.3f} detected.")
    if results['discrepancy'] > 0.1:
        print("⚠️  URGENT: Performance claims require immediate correction!")