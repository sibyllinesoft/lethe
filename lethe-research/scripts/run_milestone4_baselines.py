#!/usr/bin/env python3
"""
Milestone 4 Baseline Evaluation Runner
====================================

Single command execution interface for all six baseline implementations.

Usage:
    python scripts/run_milestone4_baselines.py --datasets datasets/lethebench --output results/milestone4_baselines.json

Features:
- Automatic dataset discovery and loading
- Progress monitoring with timing estimates
- Comprehensive validation and smoke testing  
- JSON output with complete experimental metadata
- Hardware profiling and reproducibility tracking
- Budget parity enforcement across all baselines
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add the research modules to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.eval.milestone4_baselines import (
    Milestone4BaselineEvaluator, 
    RetrievalDocument, 
    EvaluationQuery,
    create_baseline_evaluator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('milestone4_baselines.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_lethebench_dataset(dataset_path: Path) -> tuple[List[RetrievalDocument], List[EvaluationQuery]]:
    """
    Load LetheBench dataset in the expected format.
    
    Args:
        dataset_path: Path to dataset directory containing documents.jsonl and queries.jsonl
        
    Returns:
        Tuple of (documents, queries)
    """
    logger.info(f"Loading LetheBench dataset from {dataset_path}")
    
    # Look for standard filenames
    docs_file = dataset_path / "documents.jsonl"
    queries_file = dataset_path / "queries.jsonl"
    
    # Alternative locations
    if not docs_file.exists():
        docs_file = dataset_path / "corpus.jsonl"
    if not docs_file.exists():
        docs_file = dataset_path / "splits" / "train.jsonl"
    
    if not queries_file.exists():
        queries_file = dataset_path / "splits" / "dev.jsonl"
    if not queries_file.exists():
        queries_file = dataset_path / "test_queries.jsonl"
    
    if not docs_file.exists():
        raise FileNotFoundError(f"No documents file found in {dataset_path}")
    if not queries_file.exists():
        raise FileNotFoundError(f"No queries file found in {dataset_path}")
    
    # Load documents
    documents = []
    with open(docs_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                doc_data = json.loads(line)
                
                # Flexible document format handling
                doc_id = doc_data.get("doc_id", doc_data.get("id", f"doc_{line_num}"))
                content = doc_data.get("content", doc_data.get("text", doc_data.get("body", "")))
                kind = doc_data.get("kind", doc_data.get("type", "text"))
                
                if content.strip():  # Only add non-empty documents
                    documents.append(RetrievalDocument(
                        doc_id=str(doc_id),
                        content=content,
                        kind=kind,
                        metadata=doc_data.get("metadata", {})
                    ))
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed document line {line_num}: {e}")
                continue
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # Load queries  
    queries = []
    with open(queries_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                query_data = json.loads(line)
                
                # Flexible query format handling
                query_id = query_data.get("query_id", query_data.get("id", f"query_{line_num}"))
                text = query_data.get("text", query_data.get("query", ""))
                session_id = query_data.get("session_id", "")
                domain = query_data.get("domain", "general")
                complexity = query_data.get("complexity", "medium")
                
                # Ground truth handling
                ground_truth = query_data.get("ground_truth_docs", [])
                if isinstance(ground_truth, list):
                    ground_truth_docs = [str(doc) for doc in ground_truth]
                else:
                    ground_truth_docs = []
                
                relevance_judgments = query_data.get("relevance_judgments", {})
                
                if text.strip():  # Only add non-empty queries
                    queries.append(EvaluationQuery(
                        query_id=str(query_id),
                        text=text,
                        session_id=session_id,
                        domain=domain,
                        complexity=complexity,
                        ground_truth_docs=ground_truth_docs,
                        relevance_judgments={str(k): v for k, v in relevance_judgments.items()}
                    ))
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed query line {line_num}: {e}")
                continue
    
    logger.info(f"Loaded {len(queries)} queries")
    
    if not documents:
        raise ValueError("No valid documents loaded")
    if not queries:
        raise ValueError("No valid queries loaded")
    
    return documents, queries

def create_baseline_config(args) -> Dict[str, Any]:
    """Create baseline configuration from command line arguments"""
    config = {
        "db_path": ":memory:",  # Use in-memory SQLite for speed
        "model_name": args.embedding_model,
        "alpha": args.alpha,
        "lambda": args.mmr_lambda,
        "num_expansions": args.doc2query_expansions,
        "rerank_k": args.rerank_k,
        "k1": 1.2,  # BM25 parameter
        "b": 0.75,  # BM25 parameter
        "ef_construction": 200,  # HNSW build parameter
        "ef_search": 50,  # HNSW search parameter
        "expansion_model": args.doc2query_model,
        "rerank_model": args.rerank_model,
        "index_type": "hnsw"
    }
    
    return config

def print_progress_summary(evaluator, results: Dict[str, List[Any]], start_time: float):
    """Print progress summary during evaluation"""
    elapsed = time.time() - start_time
    total_queries = sum(len(r) for r in results.values()) if results else 0
    
    print(f"\n📊 Progress Summary (after {elapsed:.1f}s)")
    print(f"{'Baseline':<20} {'Queries':<8} {'Avg Latency':<12} {'Status'}")
    print("-" * 60)
    
    for baseline_name in evaluator.baselines.keys():
        if baseline_name in results:
            baseline_results = results[baseline_name]
            queries_done = len(baseline_results)
            avg_latency = sum(r.latency_ms for r in baseline_results) / len(baseline_results) if baseline_results else 0
            status = "✅ Complete" if queries_done > 0 else "⏳ Running"
        else:
            queries_done = 0
            avg_latency = 0
            status = "⏳ Pending"
        
        print(f"{baseline_name:<20} {queries_done:<8} {avg_latency:<12.1f} {status}")
    
    print()

def main():
    parser = argparse.ArgumentParser(
        description="Run Milestone 4 baseline evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python scripts/run_milestone4_baselines.py --dataset datasets/lethebench --output results/baselines.json

  # Custom configuration
  python scripts/run_milestone4_baselines.py \\
    --dataset datasets/lethebench \\
    --output results/baselines.json \\
    --k 50 \\
    --embedding-model all-MiniLM-L12-v2 \\
    --alpha 0.6 \\
    --mmr-lambda 0.8

  # Quick test run
  python scripts/run_milestone4_baselines.py \\
    --dataset datasets/lethebench \\
    --output results/quick_test.json \\
    --max-queries 10
        """
    )
    
    # Required arguments
    parser.add_argument("--dataset", type=Path, required=True, 
                       help="Path to LetheBench dataset directory")
    parser.add_argument("--output", type=Path, required=True,
                       help="Output path for results JSON")
    
    # Retrieval parameters
    parser.add_argument("--k", type=int, default=100,
                       help="Top-K results to retrieve (default: 100)")
    parser.add_argument("--max-queries", type=int, default=None,
                       help="Limit number of queries for testing (default: all)")
    
    # Model configuration
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                       help="Sentence transformer model for embeddings")
    parser.add_argument("--doc2query-model", default="doc2query/msmarco-t5-base-v1",
                       help="T5 model for doc2query expansion")
    parser.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-2-v2",
                       help="Cross-encoder model for reranking")
    
    # Baseline-specific parameters
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Static fusion weight for hybrid baseline (default: 0.5)")
    parser.add_argument("--mmr-lambda", type=float, default=0.7,
                       help="MMR diversity parameter (default: 0.7)")
    parser.add_argument("--doc2query-expansions", type=int, default=3,
                       help="Number of doc2query expansions (default: 3)")
    parser.add_argument("--rerank-k", type=int, default=100,
                       help="Number of candidates to rerank (default: 100)")
    
    # Execution options
    parser.add_argument("--config", type=Path,
                       help="JSON config file (overrides CLI args)")
    parser.add_argument("--no-smoke-tests", action="store_true",
                       help="Skip smoke tests for faster execution")
    parser.add_argument("--skip-baselines", nargs="+", 
                       choices=["bm25_only", "vector_only", "hybrid_static", 
                               "mmr_diversity", "doc2query_expansion", "crossencoder_rerank"],
                       help="Baselines to skip")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.dataset.exists():
        print(f"❌ Dataset path does not exist: {args.dataset}")
        return 1
    
    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load dataset
        print("🔍 Loading LetheBench dataset...")
        documents, queries = load_lethebench_dataset(args.dataset)
        
        # Limit queries for testing
        if args.max_queries:
            queries = queries[:args.max_queries]
            print(f"📝 Limited to {len(queries)} queries for testing")
        
        # Create configuration
        if args.config and args.config.exists():
            with open(args.config) as f:
                config = json.load(f)
        else:
            config = create_baseline_config(args)
        
        print(f"🏗️  Configuration: {json.dumps(config, indent=2)}")
        
        # Create evaluator
        print("⚙️  Initializing baseline evaluator...")
        evaluator = create_baseline_evaluator()
        evaluator.config = config
        
        # Remove skipped baselines
        if args.skip_baselines:
            for baseline_name in args.skip_baselines:
                if baseline_name in evaluator.baselines:
                    del evaluator.baselines[baseline_name]
                    print(f"⏭️  Skipped baseline: {baseline_name}")
        
        print(f"📊 Will evaluate {len(evaluator.baselines)} baselines on {len(queries)} queries")
        
        # Build indices
        print("🏗️  Building indices (this may take several minutes)...")
        index_start_time = time.time()
        evaluator.build_all_indices(documents)
        index_time = time.time() - index_start_time
        print(f"✅ Index building complete in {index_time:.1f}s")
        
        # Run evaluation
        print("🚀 Starting baseline evaluation...")
        eval_start_time = time.time()
        
        results = evaluator.evaluate_all_baselines(queries, args.k)
        
        eval_time = time.time() - eval_start_time
        print(f"✅ Evaluation complete in {eval_time:.1f}s")
        
        # Print final summary
        print_progress_summary(evaluator, results, eval_start_time)
        
        # Save results
        print(f"💾 Saving results to {args.output}...")
        evaluator.save_results(results, args.output)
        
        # Print summary statistics
        print("\n📈 Summary Statistics:")
        total_queries = sum(len(r) for r in results.values())
        total_time = index_time + eval_time
        
        print(f"  Total baselines: {len(results)}")
        print(f"  Total query evaluations: {total_queries}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average time per query: {total_time/total_queries:.2f}s")
        
        # Budget parity report
        budget_report = evaluator.budget_tracker.get_budget_report()
        if budget_report.get("baseline_budget"):
            print(f"\n💰 Budget Parity Report:")
            baseline_budget = budget_report["baseline_budget"]
            print(f"  Reference budget: {baseline_budget:.2e} FLOPs")
            
            for method, stats in budget_report["methods"].items():
                deviation = stats["deviation_from_baseline"]
                status = "✅" if stats["parity_compliant"] else "❌"
                print(f"  {method}: {deviation:.1%} deviation {status}")
        
        # Validation report
        validation_report = evaluator.fraud_validator.get_validation_report()
        print(f"\n🔍 Validation Report:")
        for method, stats in validation_report.get("methods", {}).items():
            success_rate = stats["success_rate"]
            status = "✅" if success_rate >= 0.8 else "❌"
            print(f"  {method}: {success_rate:.1%} validation success {status}")
        
        print(f"\n✅ Milestone 4 baseline evaluation complete!")
        print(f"📄 Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        print(f"\n❌ Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())