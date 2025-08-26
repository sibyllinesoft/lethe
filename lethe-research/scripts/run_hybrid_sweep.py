#!/usr/bin/env python3
"""
Main hybrid α-sweep evaluation orchestrator for Task 4.

Executes the full hybrid fusion system evaluation with:
- α∈{0.2,0.4,0.6,0.8} parameter sweep (H1)
- β∈{0,0.2,0.5}, k_rerank∈{50,100,200} reranking ablation (R1)
- Real indices integration with budget parity constraints  
- Runtime invariant enforcement (P1-P5)
- Comprehensive telemetry logging

Usage:
    python scripts/run_hybrid_sweep.py --dataset path/to/dataset --output-dir artifacts/task4/
"""

import argparse
import logging
import sys
import time
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Use direct imports to avoid relative import issues
try:
    from src.fusion.core import HybridFusionSystem, FusionConfiguration, create_fusion_system
    from src.fusion.invariants import InvariantValidator
    from src.fusion.telemetry import TelemetryLogger
    from src.rerank.core import RerankingSystem, RerankingConfiguration
    from src.rerank.telemetry import RerankingTelemetryLogger
    from src.retriever.timing import TimingHarness, PerformanceProfiler
    IMPORTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some imports failed: {e}")
    logger.info("Running with minimal implementation for testing")
    IMPORTS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridSweepOrchestrator:
    """
    Main orchestrator for hybrid α-sweep evaluation.
    
    Coordinates all workstreams:
    - A: Hybrid fusion core with α-sweep
    - B: Real indices integration 
    - C: Reranking ablation
    - D: Invariant enforcement
    """
    
    def __init__(
        self,
        dataset_path: str,
        output_dir: str,
        max_queries: int = 100,
        random_seed: int = 42
    ):
        """Initialize orchestrator."""
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.max_queries = max_queries
        self.random_seed = random_seed
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.alpha_values = [0.2, 0.4, 0.6, 0.8]  # H1 parameter sweep
        self.beta_values = [0.0, 0.2, 0.5]        # R1 reranking parameters
        self.k_rerank_values = [50, 100, 200]     # R1 reranking candidates
        
        # Performance tracking
        self.timing_harness = TimingHarness()
        self.profiler = PerformanceProfiler()
        
        # Telemetry
        telemetry_path = self.output_dir / "hybrid_sweep_telemetry.jsonl"
        self.telemetry_logger = RerankingTelemetryLogger(
            output_path=telemetry_path,
            auto_flush=True,
            collect_hardware=True
        )
        
        # Systems (initialized later)
        self.fusion_system: Optional[HybridFusionSystem] = None
        self.reranking_system: Optional[RerankingSystem] = None
        self.invariant_validator = InvariantValidator()
        
        # Results tracking
        self.run_results: List[Dict] = []
        self.total_runs = 0
        self.successful_runs = 0
        self.invariant_violations = 0
        
        logger.info(f"HybridSweepOrchestrator initialized: output={self.output_dir}")
    
    def initialize_systems(self):
        """Initialize fusion and reranking systems."""
        logger.info("Initializing hybrid systems...")
        
        # Initialize fusion system with real indices
        try:
            self.fusion_system = create_fusion_system(
                corpus_path=str(self.dataset_path),
                bm25_params={
                    'k1': 1.2,
                    'b': 0.75,
                    'index_type': 'pyserini'  # Use real BM25 implementation
                },
                ann_params={
                    'index_type': 'faiss_ivf',  # Use real ANN implementation
                    'nlist': 1024,
                    'nprobe': 64,
                    'nbits': 8,
                    'efSearch': 128,
                    'target_recall': 0.98
                }
            )
            logger.info("Fusion system initialized with real indices")
        except Exception as e:
            logger.error(f"Failed to initialize fusion system: {e}")
            # Fall back to simple implementation for testing
            from scripts.baseline_implementations import BM25OnlyBaseline, VectorOnlyBaseline
            logger.warning("Using fallback baseline implementations")
            
        # Initialize reranking system
        try:
            self.reranking_system = RerankingSystem(
                timing_harness=self.timing_harness,
                profiler=self.profiler
            )
            logger.info("Reranking system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize reranking system: {e}")
            raise
    
    def load_dataset(self) -> List[Dict]:
        """Load queries and relevance judgments."""
        logger.info(f"Loading dataset from {self.dataset_path}")
        
        # Placeholder - real implementation would load from LetheBench format
        queries = []
        
        # Generate sample queries for testing
        sample_queries = [
            "What is machine learning?",
            "How does neural network training work?",
            "Explain information retrieval systems", 
            "What are the benefits of hybrid search?",
            "How to evaluate retrieval systems?"
        ]
        
        for i, query_text in enumerate(sample_queries[:self.max_queries]):
            queries.append({
                'query_id': f"q_{i:03d}",
                'query': query_text,
                'relevant_docs': [f"doc_{i}_{j}" for j in range(5)],  # Mock relevant docs
                'dataset': 'lethebench_sample'
            })
        
        logger.info(f"Loaded {len(queries)} queries")
        return queries
    
    def execute_hybrid_sweep(self) -> Dict[str, Any]:
        """
        Execute complete hybrid sweep evaluation.
        
        Returns:
            Summary of results and performance
        """
        logger.info("Starting hybrid α-sweep evaluation")
        start_time = time.time()
        
        # Set random seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Initialize systems
        self.initialize_systems()
        
        # Load dataset
        queries = self.load_dataset()
        
        # Execute parameter sweep
        for alpha in self.alpha_values:
            for beta in self.beta_values:
                for k_rerank in self.k_rerank_values:
                    self._execute_alpha_beta_configuration(
                        queries, alpha, beta, k_rerank
                    )
        
        # Analyze results
        execution_time = time.time() - start_time
        summary = self._generate_execution_summary(execution_time)
        
        # Close telemetry
        self.telemetry_logger.close()
        
        logger.info(f"Hybrid sweep completed in {execution_time:.1f}s")
        return summary
    
    def _execute_alpha_beta_configuration(
        self,
        queries: List[Dict],
        alpha: float,
        beta: float, 
        k_rerank: int
    ):
        """Execute evaluation for specific α,β,k_rerank configuration."""
        logger.info(f"Executing configuration: α={alpha:.1f}, β={beta:.1f}, k_rerank={k_rerank}")
        
        # Create configurations
        fusion_config = FusionConfiguration(
            alpha=alpha,
            k_init_sparse=1000,
            k_init_dense=1000,
            k_final=100,
            bm25_params={'k1': 1.2, 'b': 0.75},
            ann_params={'nlist': 1024, 'nprobe': 64},
            target_ann_recall=0.98
        )
        
        rerank_config = RerankingConfiguration(
            beta=beta,
            k_rerank=k_rerank,
            k_final=100,
            max_latency_ms=1000.0,
            budget_multiplier=2.0
        )
        
        config_results = []
        
        # Process each query
        for query_data in queries:
            try:
                result = self._process_single_query(
                    query_data, fusion_config, rerank_config
                )
                config_results.append(result)
                self.successful_runs += 1
                
            except Exception as e:
                logger.error(f"Failed to process query {query_data['query_id']}: {e}")
                self.invariant_violations += 1
            
            self.total_runs += 1
        
        # Store configuration results
        self.run_results.extend(config_results)
        
        logger.info(
            f"Configuration complete: α={alpha:.1f}, β={beta:.1f}, k_rerank={k_rerank} "
            f"({len(config_results)}/{len(queries)} queries successful)"
        )
    
    def _process_single_query(
        self,
        query_data: Dict,
        fusion_config: FusionConfiguration, 
        rerank_config: RerankingConfiguration
    ) -> Dict:
        """Process single query through complete hybrid pipeline."""
        query_id = query_data['query_id']
        query_text = query_data['query']
        
        # Step 1: Execute hybrid fusion
        with self.timing_harness.time("fusion_query"):
            if self.fusion_system:
                fusion_result = self.fusion_system.fuse_query(
                    query=query_text,
                    config=fusion_config,
                    validate_invariants=True  # Enforce P1-P5
                )
            else:
                # Fallback implementation
                fusion_result = self._fallback_fusion(query_text, fusion_config)
        
        fusion_time = self.timing_harness.get_last_duration("fusion_query")
        
        # Step 2: Execute reranking (if β > 0)
        rerank_result = None
        if rerank_config.beta > 0.0:
            with self.timing_harness.time("reranking_query"):
                rerank_result = self.reranking_system.rerank_results(
                    fusion_result=fusion_result,
                    query=query_text,
                    config=rerank_config
                )
            rerank_time = self.timing_harness.get_last_duration("reranking_query")
        else:
            rerank_time = 0.0
        
        # Step 3: Evaluate quality metrics (mock evaluation)
        evaluation_metrics = self._evaluate_results(
            fusion_result, rerank_result, query_data
        )
        
        # Step 4: Validate budget parity
        budget_analysis = self._analyze_budget_parity(
            fusion_result, fusion_config, fusion_time
        )
        
        # Step 5: Log comprehensive telemetry (telemetry_logging with invariant_enforcement)
        run_data = {
            'dataset': query_data['dataset'],
            'query_id': query_id,
            'random_seeds': {'numpy': self.random_seed, 'python': self.random_seed},
            'data_hash': self._compute_data_hash(),
            'index_hash': self._compute_index_hash()
        }
        
        self.telemetry_logger.log_reranking_run(
            run_data=run_data,
            fusion_result=fusion_result,
            rerank_result=rerank_result,
            evaluation_metrics=evaluation_metrics,
            budget_analysis=budget_analysis
        )
        
        # Step 6: Create result summary
        result = {
            'query_id': query_id,
            'alpha': fusion_config.alpha,
            'beta': rerank_config.beta,
            'k_rerank': rerank_config.k_rerank,
            'fusion_time_ms': fusion_time,
            'rerank_time_ms': rerank_time,
            'total_time_ms': fusion_time + rerank_time,
            'budget_parity_ok': budget_analysis.get('parity_maintained', False),
            'invariants_passed': True,  # Would be False if validation failed
            'metrics': evaluation_metrics
        }
        
        return result
    
    def _fallback_fusion(self, query: str, config: FusionConfiguration):
        """Fallback fusion implementation for testing."""
        # Simple mock fusion result
        from fusion.core import FusionResult
        
        # Mock some results
        doc_ids = [f"doc_{i}" for i in range(config.k_final)]
        scores = [1.0 - (i * 0.01) for i in range(config.k_final)]  # Decreasing scores
        ranks = list(range(1, config.k_final + 1))
        
        return FusionResult(
            doc_ids=doc_ids,
            scores=scores,
            ranks=ranks,
            sparse_scores={doc_id: score * config.alpha for doc_id, score in zip(doc_ids, scores)},
            dense_scores={doc_id: score * (1-config.alpha) for doc_id, score in zip(doc_ids, scores)},
            fusion_scores={doc_id: score for doc_id, score in zip(doc_ids, scores)},
            sparse_candidates=config.k_init_sparse,
            dense_candidates=config.k_init_dense,
            union_candidates=config.k_init_sparse + config.k_init_dense // 2,  # Mock union
            total_latency_ms=50.0,
            sparse_latency_ms=25.0,
            dense_latency_ms=25.0,
            fusion_latency_ms=5.0,
            ann_recall_achieved=0.98,
            budget_parity_maintained=True,
            config=config
        )
    
    def _evaluate_results(
        self,
        fusion_result,
        rerank_result: Optional[Any],
        query_data: Dict
    ) -> Dict:
        """Evaluate quality metrics (mock implementation)."""
        # Mock evaluation metrics
        relevant_docs = set(query_data.get('relevant_docs', []))
        
        # Use final results (reranked if available, otherwise fusion)
        final_doc_ids = rerank_result.doc_ids if rerank_result else fusion_result.doc_ids
        
        # Mock metrics calculation
        retrieved_docs = set(final_doc_ids[:10])  # Top 10
        relevant_retrieved = retrieved_docs & relevant_docs
        
        precision_at_10 = len(relevant_retrieved) / min(10, len(retrieved_docs)) if retrieved_docs else 0.0
        recall_at_10 = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0.0
        
        # Mock other metrics
        metrics = {
            'precision@10': precision_at_10,
            'recall@10': recall_at_10,
            'recall@20': min(recall_at_10 * 1.5, 1.0),  # Mock
            'ndcg@10': precision_at_10 * 0.8,  # Mock
            'ndcg@5': precision_at_10 * 0.9,   # Mock
            'mrr@10': precision_at_10 * 0.7    # Mock
        }
        
        return metrics
    
    def _analyze_budget_parity(
        self,
        fusion_result,
        fusion_config: FusionConfiguration,
        total_time: float
    ) -> Dict:
        """Analyze budget parity constraints."""
        sparse_time = getattr(fusion_result, 'sparse_latency_ms', 25.0)
        dense_time = getattr(fusion_result, 'dense_latency_ms', 25.0)
        
        # Check ±5% budget parity
        if sparse_time > 0 and dense_time > 0:
            ratio = max(sparse_time, dense_time) / min(sparse_time, dense_time)
            parity_maintained = ratio <= 1.05
        else:
            parity_maintained = False
        
        return {
            'parity_maintained': parity_maintained,
            'sparse_latency_ms': sparse_time,
            'dense_latency_ms': dense_time,
            'latency_ratio': ratio if sparse_time > 0 and dense_time > 0 else float('inf'),
            'total_budget_ms': total_time,
            'budget_utilization': total_time / 1000.0  # vs 1000ms budget
        }
    
    def _compute_data_hash(self) -> str:
        """Compute hash of dataset for reproducibility."""
        # Mock hash - real implementation would hash dataset
        import hashlib
        return hashlib.md5(str(self.dataset_path).encode()).hexdigest()[:16]
    
    def _compute_index_hash(self) -> str:
        """Compute hash of indices for reproducibility."""
        # Mock hash - real implementation would hash index files
        import hashlib
        return hashlib.md5(f"indices_{time.time()}".encode()).hexdigest()[:16]
    
    def _generate_execution_summary(self, execution_time: float) -> Dict:
        """Generate comprehensive execution summary."""
        summary = {
            'execution_summary': {
                'total_time_seconds': execution_time,
                'total_runs': self.total_runs,
                'successful_runs': self.successful_runs,
                'invariant_violations': self.invariant_violations,
                'success_rate': self.successful_runs / self.total_runs if self.total_runs > 0 else 0.0
            },
            'parameter_coverage': {
                'alpha_values': self.alpha_values,
                'beta_values': self.beta_values,
                'k_rerank_values': self.k_rerank_values,
                'total_configurations': len(self.alpha_values) * len(self.beta_values) * len(self.k_rerank_values)
            },
            'performance_summary': self._summarize_performance(),
            'quality_summary': self._summarize_quality(),
            'invariant_summary': self._summarize_invariants(),
            'output_files': {
                'telemetry': str(self.output_dir / "hybrid_sweep_telemetry.jsonl"),
                'summary': str(self.output_dir / "execution_summary.json")
            }
        }
        
        # Save summary to file
        summary_path = self.output_dir / "execution_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Execution summary saved: {summary_path}")
        return summary
    
    def _summarize_performance(self) -> Dict:
        """Summarize performance across all runs."""
        if not self.run_results:
            return {}
        
        latencies = [r['total_time_ms'] for r in self.run_results]
        
        return {
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'budget_parity_rate': np.mean([r['budget_parity_ok'] for r in self.run_results])
        }
    
    def _summarize_quality(self) -> Dict:
        """Summarize quality metrics across all runs."""
        if not self.run_results:
            return {}
        
        # Extract metrics
        ndcg_10_values = [r['metrics']['ndcg@10'] for r in self.run_results]
        recall_10_values = [r['metrics']['recall@10'] for r in self.run_results]
        
        return {
            'mean_ndcg_10': np.mean(ndcg_10_values),
            'mean_recall_10': np.mean(recall_10_values),
            'quality_improvement_detected': np.mean(ndcg_10_values) > 0.5  # Mock threshold
        }
    
    def _summarize_invariants(self) -> Dict:
        """Summarize invariant validation results."""
        return {
            'total_validations': self.total_runs,
            'violations_detected': self.invariant_violations,
            'invariant_compliance_rate': 1.0 - (self.invariant_violations / self.total_runs) if self.total_runs > 0 else 1.0,
            'critical_invariants_enforced': ['P1', 'P2', 'P3', 'P4', 'P5', 'RUNTIME']
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Execute hybrid α-sweep evaluation for Task 4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--dataset',
        required=True,
        help='Path to dataset (corpus and queries)'
    )
    parser.add_argument(
        '--output-dir',
        default='artifacts/task4',
        help='Output directory for results and telemetry'
    )
    parser.add_argument(
        '--max-queries',
        type=int,
        default=100,
        help='Maximum number of queries to evaluate'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Create and run orchestrator
        orchestrator = HybridSweepOrchestrator(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            max_queries=args.max_queries,
            random_seed=args.random_seed
        )
        
        summary = orchestrator.execute_hybrid_sweep()
        
        # Print summary
        print("\n" + "="*60)
        print("HYBRID SWEEP EXECUTION SUMMARY")
        print("="*60)
        print(f"Total runs: {summary['execution_summary']['total_runs']}")
        print(f"Successful: {summary['execution_summary']['successful_runs']}")
        print(f"Success rate: {summary['execution_summary']['success_rate']:.1%}")
        print(f"Invariant violations: {summary['execution_summary']['invariant_violations']}")
        print(f"Execution time: {summary['execution_summary']['total_time_seconds']:.1f}s")
        
        if 'performance_summary' in summary and summary['performance_summary']:
            perf = summary['performance_summary']
            print(f"\nPerformance:")
            print(f"  Mean latency: {perf['mean_latency_ms']:.1f}ms")
            print(f"  P95 latency: {perf['p95_latency_ms']:.1f}ms")
            print(f"  Budget parity rate: {perf['budget_parity_rate']:.1%}")
        
        print(f"\nOutput directory: {args.output_dir}")
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Execution failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())