#!/usr/bin/env python3
"""
Simple Experiment Runner - V1 Baselines
=======================================

Minimal experiment runner to test V1 baselines without MLflow dependencies.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add project modules to path
sys.path.append(str(Path(__file__).parent.parent))
from scripts.baseline_implementations import BaselineEvaluator, Document, Query, QueryResult

def load_dataset(split: str = "dev") -> tuple[List[Document], List[Query]]:
    """Load LetheBench dataset for evaluation (simplified)"""
    dataset_path = Path("datasets/test_run/lethebench_v3.0.0")
    
    # Load queries directly without complex schema validation
    queries_file = dataset_path / "splits" / f"{split}.jsonl"
    print(f"Looking for queries at: {queries_file.absolute()}")
    if not queries_file.exists():
        raise FileNotFoundError(f"Split file not found: {queries_file}")
    
    queries = []
    documents = []
    doc_set = set()
    
    with open(queries_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                query_data = json.loads(line)
                
                # Extract ground truth documents
                ground_truth_docs = query_data.get("ground_truth_docs", [])
                ground_truth_doc_ids = []
                
                # Process ground truth documents  
                for gt_doc in ground_truth_docs:
                    doc_id = gt_doc.get("doc_id")
                    if doc_id and doc_id not in doc_set:
                        doc = Document(
                            doc_id=doc_id,
                            content=gt_doc.get("content", ""),
                            kind=gt_doc.get("doc_type", "text"),
                            metadata=gt_doc.get("metadata", {})
                        )
                        documents.append(doc)
                        doc_set.add(doc_id)
                    if doc_id:
                        ground_truth_doc_ids.append(doc_id)
                
                # Convert to evaluation format using raw JSON data
                query = Query(
                    query_id=query_data.get("query_id", f"query_{line_num}"),
                    text=query_data.get("query_text", ""),
                    session_id=query_data.get("session_id", f"session_{line_num}"),
                    domain=query_data.get("domain", "unknown"),
                    complexity=query_data.get("complexity", "medium"),
                    ground_truth_docs=ground_truth_doc_ids
                )
                queries.append(query)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(queries)} queries and {len(documents)} documents from {split} split")
    return documents, queries

def run_v1_baselines(split: str = "dev", quick: bool = False):
    """Run V1 baseline experiments"""
    print(f"Running V1 Baselines on {split} split (quick={quick})")
    
    # Load dataset
    documents, queries = load_dataset(split)
    
    # Limit queries for quick test
    if quick:
        queries = queries[:10]
        print(f"Quick mode: limiting to {len(queries)} queries")
    
    # Setup baseline evaluator
    db_path = f"/tmp/lethe_v1_{split}.db"
    evaluator = BaselineEvaluator(db_path)
    
    # Run evaluation
    start_time = time.time()
    results = evaluator.evaluate_all_baselines(documents, queries, k=10)
    total_time = time.time() - start_time
    
    # Save results
    output_dir = Path("../artifacts/experiments/v1_baselines")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "experiment": "v1_baselines",
        "split": split,
        "quick_mode": quick,
        "total_queries": len(queries),
        "total_documents": len(documents),
        "total_time_seconds": total_time,
        "baselines": {}
    }
    
    for baseline_name, baseline_results in results.items():
        if baseline_results:
            avg_latency = sum(r["latency_ms"] for r in baseline_results) / len(baseline_results)
            avg_memory = sum(r["memory_mb"] for r in baseline_results) / len(baseline_results)
            
            summary["baselines"][baseline_name] = {
                "queries_completed": len(baseline_results),
                "avg_latency_ms": avg_latency,
                "avg_memory_mb": avg_memory,
                "success_rate": len(baseline_results) / len(queries)
            }
        
        # Save detailed results
        output_file = output_dir / f"{baseline_name}_{split}_results.json"
        with open(output_file, 'w') as f:
            json.dump(baseline_results, f, indent=2)
        print(f"Saved {len(baseline_results)} results to {output_file}")
    
    # Save summary
    summary_file = output_dir / f"v1_baselines_{split}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved experiment summary to {summary_file}")
    
    # Print results
    print("\nV1 Baseline Results:")
    print("=" * 50)
    for baseline_name, stats in summary["baselines"].items():
        print(f"{baseline_name:20s}: {stats['queries_completed']:3d} queries, "
              f"{stats['avg_latency_ms']:6.1f}ms avg latency, "
              f"{stats['success_rate']*100:5.1f}% success")
    print(f"\nTotal experiment time: {total_time:.1f}s")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run V1 Baseline Experiments")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev",
                        help="Dataset split to evaluate on")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick test with limited queries")
    
    args = parser.parse_args()
    
    try:
        results = run_v1_baselines(split=args.split, quick=args.quick)
        print(f"\nV1 Baselines completed successfully!")
        
    except Exception as e:
        print(f"Error running V1 baselines: {e}")
        sys.exit(1)