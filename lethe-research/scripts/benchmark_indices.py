#!/usr/bin/env python3
"""
Production-Grade Index Benchmarking Harness

Comprehensive performance measurement system for BM25 and ANN indices
with recall curve generation and budget parity validation.

Usage:
    python benchmark_indices.py --config config.yaml --dataset msmarco-passage-dev
    python benchmark_indices.py --config config.yaml --recall-curves --ann-sweep
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retriever import (
    RetrieverConfig,
    TimingHarness,
    PerformanceProfiler, 
    create_bm25_retriever,
    create_ann_retriever,
    DenseEmbeddingManager,
    IndexRegistry,
    generate_recall_curves
)

logger = logging.getLogger(__name__)

class IndexBenchmarkHarness:
    """
    Comprehensive benchmarking harness for IR indices.
    
    Provides end-to-end latency measurement, recall evaluation,
    and budget parity validation across different index types.
    """
    
    def __init__(self, config: RetrieverConfig):
        """
        Initialize benchmark harness.
        
        Args:
            config: Complete retriever configuration
        """
        self.config = config
        self.indices_dir = Path(config.system.indices_dir)
        
        # Initialize components
        self.registry = IndexRegistry(self.indices_dir)
        self.profiler = PerformanceProfiler()
        
        # Setup logging
        self._setup_logging()
        
        # Benchmark configuration
        self.cold_cycles = 50
        self.warm_cycles = 500
        
    def _setup_logging(self):
        """Configure logging for benchmarking."""
        level = getattr(logging, self.config.log_level.upper())
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.indices_dir / "benchmark.log")
            ]
        )
        
    def benchmark_all_indices(self,
                             datasets: Optional[List[str]] = None,
                             index_types: Optional[List[str]] = None,
                             generate_recall_curves: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark all specified indices.
        
        Args:
            datasets: Dataset names to benchmark
            index_types: Index types to benchmark  
            generate_recall_curves: Whether to generate ANN recall curves
            
        Returns:
            Comprehensive benchmarking results
        """
        
        datasets = datasets or self.config.datasets
        index_types = index_types or ["bm25", "hnsw", "ivf_pq"]
        
        logger.info(f"Benchmarking indices for datasets: {datasets}")
        logger.info(f"Index types: {index_types}")
        logger.info(f"Generate recall curves: {generate_recall_curves}")
        
        results = {}
        
        for dataset_name in datasets:
            logger.info(f"\n{'='*60}")
            logger.info(f"Benchmarking dataset: {dataset_name}")  
            logger.info(f"{'='*60}")
            
            dataset_results = {}
            
            # Load queries and ground truth
            queries, ground_truth = self._load_evaluation_data(dataset_name)
            if not queries:
                logger.error(f"No queries found for dataset: {dataset_name}")
                continue
                
            logger.info(f"Loaded {len(queries)} queries")
            
            # Benchmark each index type
            for index_type in index_types:
                try:
                    logger.info(f"\nBenchmarking {index_type} index...")
                    
                    if index_type == "bm25":
                        result = self._benchmark_bm25(dataset_name, queries, ground_truth)
                    elif index_type in ["hnsw", "ivf_pq"]:
                        result = self._benchmark_ann(dataset_name, index_type, queries, ground_truth, generate_recall_curves)
                    else:
                        logger.warning(f"Unknown index type: {index_type}")
                        continue
                        
                    dataset_results[index_type] = result
                    logger.info(f"Successfully benchmarked {index_type}")
                    
                except Exception as e:
                    logger.error(f"Failed to benchmark {index_type}: {e}", exc_info=True)
                    dataset_results[index_type] = {"error": str(e)}
                    
            results[dataset_name] = dataset_results
            
        # Export benchmark report
        self._export_benchmark_report(results)
        
        # Validate budget parity
        self._validate_budget_parity(results)
        
        return results
        
    def _load_evaluation_data(self, dataset_name: str) -> Tuple[List[str], Optional[List[List[str]]]]:
        """Load queries and ground truth for evaluation."""
        
        # Load queries
        queries_path = Path(f"datasets/{dataset_name}/queries.jsonl")
        queries = []
        
        if queries_path.exists():
            with open(queries_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    query_text = data.get('text', data.get('query', ''))
                    if query_text:
                        queries.append(query_text)
        else:
            logger.warning(f"Queries file not found: {queries_path}")
            # Create synthetic queries for testing
            queries = [
                "information retrieval system",
                "machine learning algorithms", 
                "deep neural networks",
                "natural language processing",
                "computer vision applications"
            ] * 20  # 100 synthetic queries
            logger.info(f"Using {len(queries)} synthetic queries")
            
        # Load ground truth (for recall evaluation)
        ground_truth = None
        qrels_path = Path(f"datasets/{dataset_name}/qrels.jsonl")
        
        if qrels_path.exists():
            ground_truth = []
            with open(qrels_path, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    relevant_docs = data.get('relevant_docs', [])
                    ground_truth.append(relevant_docs)
                    
            logger.info(f"Loaded ground truth for {len(ground_truth)} queries")
        else:
            logger.warning(f"Ground truth file not found: {qrels_path}")
            
        return queries, ground_truth
        
    def _benchmark_bm25(self,
                       dataset_name: str,
                       queries: List[str],
                       ground_truth: Optional[List[List[str]]]) -> Dict[str, Any]:
        """Benchmark BM25 index."""
        
        # Load BM25 index
        index_path = self.indices_dir / dataset_name / "bm25"
        
        if not index_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {index_path}")
            
        # Create timing harness
        harness = self.profiler.create_harness(
            "bm25_benchmark",
            cold_cycles=self.cold_cycles,
            warm_cycles=self.warm_cycles,
            gc_between_runs=True
        )
        
        # Create retriever
        retriever = create_bm25_retriever(index_path, self.config.bm25, harness)
        
        # Benchmark search performance
        logger.info("Benchmarking BM25 search performance...")
        
        def search_queries():
            for query in queries:
                results = retriever.search(query, k=1000)
                
        profile = harness.benchmark_function(search_queries, "bm25_batch_search")
        
        # Calculate per-query metrics
        per_query_latency = profile.latency_mean / len(queries)
        throughput = len(queries) / (profile.total_duration_sec / profile.warm_cycles)
        
        # Get index statistics
        index_stats = retriever.get_index_stats()
        
        result = {
            "performance": {
                "latency_p50_ms": profile.latency_p50,
                "latency_p95_ms": profile.latency_p95, 
                "latency_p99_ms": profile.latency_p99,
                "per_query_latency_ms": per_query_latency,
                "throughput_queries_per_sec": throughput,
                "memory_mean_mb": profile.memory_mean,
                "memory_peak_mb": profile.memory_peak
            },
            "index_stats": index_stats.__dict__ if index_stats else {},
            "num_queries": len(queries),
            "warm_cycles": profile.warm_cycles,
            "total_duration_sec": profile.total_duration_sec
        }
        
        return result
        
    def _benchmark_ann(self,
                      dataset_name: str,
                      index_type: str,
                      queries: List[str], 
                      ground_truth: Optional[List[List[str]]],
                      generate_curves: bool) -> Dict[str, Any]:
        """Benchmark ANN index (HNSW or IVF-PQ)."""
        
        # Load ANN index
        index_path = self.indices_dir / dataset_name / "dense" / index_type
        
        if not index_path.exists():
            raise FileNotFoundError(f"{index_type.upper()} index not found: {index_path}")
            
        # Create timing harness
        harness = self.profiler.create_harness(
            f"{index_type}_benchmark",
            cold_cycles=self.cold_cycles,
            warm_cycles=self.warm_cycles,
            gc_between_runs=True
        )
        
        # Create retriever
        retriever = create_ann_retriever(index_path, index_type, harness)
        
        # Generate query embeddings
        query_vectors = self._generate_query_embeddings(queries, dataset_name)
        
        # Benchmark with default parameters
        logger.info(f"Benchmarking {index_type.upper()} search performance...")
        
        def search_queries():
            retriever.search(query_vectors, k=1000)
            
        profile = harness.benchmark_function(search_queries, f"{index_type}_batch_search")
        
        # Calculate metrics
        per_query_latency = profile.latency_mean / len(queries)
        throughput = len(queries) / (profile.total_duration_sec / profile.warm_cycles)
        
        result = {
            "performance": {
                "latency_p50_ms": profile.latency_p50,
                "latency_p95_ms": profile.latency_p95,
                "latency_p99_ms": profile.latency_p99,
                "per_query_latency_ms": per_query_latency,
                "throughput_queries_per_sec": throughput,
                "memory_mean_mb": profile.memory_mean,
                "memory_peak_mb": profile.memory_peak
            },
            "num_queries": len(queries),
            "query_vector_dim": query_vectors.shape[1],
            "warm_cycles": profile.warm_cycles,
            "total_duration_sec": profile.total_duration_sec
        }
        
        # Generate recall curves if requested
        if generate_curves and ground_truth:
            logger.info(f"Generating recall curves for {index_type.upper()}...")
            recall_curves = self._generate_recall_curves(
                retriever, query_vectors, ground_truth, index_type
            )
            result["recall_curves"] = [curve.to_dict() for curve in recall_curves]
            
        return result
        
    def _generate_query_embeddings(self,
                                  queries: List[str],
                                  dataset_name: str) -> np.ndarray:
        """Generate embeddings for query texts."""
        
        # Initialize embedding manager
        embeddings_dir = self.indices_dir / dataset_name / "embeddings"
        
        embedding_manager = DenseEmbeddingManager(
            config=self.config.embeddings,
            cache_dir=embeddings_dir
        )
        
        # Encode queries
        logger.info("Encoding query embeddings...")
        query_vectors = embedding_manager.encode_texts(queries, show_progress=True)
        
        logger.info(f"Generated query embeddings: {query_vectors.shape}")
        return query_vectors
        
    def _generate_recall_curves(self,
                               retriever,
                               query_vectors: np.ndarray,
                               ground_truth: List[List[str]],
                               index_type: str):
        """Generate recall curves for ANN index."""
        
        if index_type == "hnsw":
            parameter_configs = {
                "efSearch": self.config.hnsw.ef_search_values
            }
        elif index_type == "ivf_pq":
            parameter_configs = {
                "nprobe": self.config.ivf_pq.nprobe_values
            }
        else:
            return []
            
        return generate_recall_curves(
            retriever=retriever,
            query_vectors=query_vectors,
            ground_truth=ground_truth,
            parameter_configs=parameter_configs,
            k_values=[10, 100, 1000]
        )
        
    def _validate_budget_parity(self, results: Dict[str, Dict[str, Any]]):
        """Validate budget parity constraints (±5% compute/FLOPs)."""
        
        logger.info("\nValidating budget parity constraints...")
        
        flops_variance = self.config.system.flops_budget_variance
        
        for dataset_name, dataset_results in results.items():
            logger.info(f"Budget validation for {dataset_name}:")
            
            # Extract latency metrics
            latencies = {}
            for index_type, result in dataset_results.items():
                if "error" not in result and "performance" in result:
                    latencies[index_type] = result["performance"]["per_query_latency_ms"]
                    
            if len(latencies) < 2:
                logger.warning("Need at least 2 index types for budget comparison")
                continue
                
            # Compare all pairs
            index_types = list(latencies.keys())
            for i in range(len(index_types)):
                for j in range(i + 1, len(index_types)):
                    type1, type2 = index_types[i], index_types[j]
                    lat1, lat2 = latencies[type1], latencies[type2]
                    
                    # Calculate relative difference
                    relative_diff = abs(lat1 - lat2) / min(lat1, lat2)
                    within_budget = relative_diff <= flops_variance
                    
                    status = "✓" if within_budget else "✗"
                    logger.info(f"  {type1} vs {type2}: {lat1:.2f}ms vs {lat2:.2f}ms "
                               f"({relative_diff:.1%} diff) {status}")
                               
    def _export_benchmark_report(self, results: Dict[str, Dict[str, Any]]):
        """Export comprehensive benchmark report."""
        
        report_file = self.indices_dir / "benchmark_report.json"
        
        # Generate summary statistics
        total_benchmarks = 0
        successful_benchmarks = 0
        total_latency = 0.0
        total_throughput = 0.0
        
        for dataset_results in results.values():
            for result in dataset_results.values():
                total_benchmarks += 1
                
                if "error" not in result and "performance" in result:
                    successful_benchmarks += 1
                    total_latency += result["performance"]["per_query_latency_ms"]
                    total_throughput += result["performance"]["throughput_queries_per_sec"]
                    
        # Create comprehensive report
        report = {
            "summary": {
                "total_benchmarks": total_benchmarks,
                "successful_benchmarks": successful_benchmarks,
                "success_rate": successful_benchmarks / total_benchmarks if total_benchmarks > 0 else 0,
                "avg_latency_ms": total_latency / successful_benchmarks if successful_benchmarks > 0 else 0,
                "avg_throughput_qps": total_throughput / successful_benchmarks if successful_benchmarks > 0 else 0
            },
            "configuration": {
                "cold_cycles": self.cold_cycles,
                "warm_cycles": self.warm_cycles,
                "system_config": self.config.system.__dict__
            },
            "results": results,
            "performance_profiles": self.profiler.export_profiles()
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Benchmark report exported: {report_file}")
        logger.info(f"Summary: {successful_benchmarks}/{total_benchmarks} successful benchmarks")

def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="Benchmark production IR indices")
    
    parser.add_argument("--config", "-c",
                       type=str,
                       default="config/retriever_config.yaml",
                       help="Configuration file path")
    
    parser.add_argument("--dataset", "-d", 
                       type=str,
                       help="Dataset name (or 'all' for all configured)")
    
    parser.add_argument("--indices", "-i",
                       type=str,
                       default="bm25,hnsw,ivf_pq",
                       help="Comma-separated list of index types")
    
    parser.add_argument("--recall-curves",
                       action="store_true",
                       help="Generate recall curves for ANN indices")
    
    parser.add_argument("--ann-sweep",
                       action="store_true",
                       help="Perform parameter sweep for ANN indices")
    
    parser.add_argument("--cold-cycles",
                       type=int,
                       default=50,
                       help="Number of cold cycles for warm-up")
    
    parser.add_argument("--warm-cycles", 
                       type=int,
                       default=500,
                       help="Number of warm cycles for measurement")
    
    parser.add_argument("--debug",
                       action="store_true",
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        if Path(args.config).exists():
            config = RetrieverConfig.from_yaml(args.config)
        else:
            logger.warning(f"Config file not found: {args.config}, using defaults")
            config = RetrieverConfig()
            
        if args.debug:
            config.debug = True
            config.log_level = "DEBUG"
            
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
        
    # Prepare datasets and index types
    if args.dataset == "all" or args.dataset is None:
        datasets = config.datasets
    else:
        datasets = [args.dataset]
        
    index_types = [t.strip() for t in args.indices.split(",")]
    
    # Create and run benchmark harness
    try:
        harness = IndexBenchmarkHarness(config)
        harness.cold_cycles = args.cold_cycles
        harness.warm_cycles = args.warm_cycles
        
        results = harness.benchmark_all_indices(
            datasets=datasets,
            index_types=index_types,
            generate_recall_curves=args.recall_curves or args.ann_sweep
        )
        
        logger.info("Benchmarking completed successfully!")
        
        # Print summary
        successful = sum(1 for dataset_results in results.values()
                        for result in dataset_results.values() 
                        if "error" not in result)
        total = sum(len(dataset_results) for dataset_results in results.values())
        
        print(f"\nBenchmark Summary:")
        print(f"  Successful: {successful}/{total}")
        print(f"  Cold cycles: {args.cold_cycles}")
        print(f"  Warm cycles: {args.warm_cycles}")
        
        if successful < total:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Benchmarking failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()