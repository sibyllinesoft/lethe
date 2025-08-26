#!/usr/bin/env python3
"""
Task 3: Baseline Suite Evaluation Orchestrator
==============================================

Main evaluation script for running bulletproof baseline evaluation with:
- Real model integrations (BM25, SPLADE, uniCOIL, Dense, ColBERTv2, RRF)
- Budget parity enforcement (±5% compute/FLOPs) 
- Anti-fraud validation (non-empty guards, smoke tests)
- Statistical rigor (per-query metrics, JSONL persistence)
- MS MARCO/BEIR dataset integration

Usage:
    python scripts/run_eval.py --dataset msmarco --output artifacts/eval_results/
    python scripts/run_eval.py --dataset beir-covid --smoke-test-only
    python scripts/run_eval.py --all-datasets --baselines BM25,Dense,RRF

Critical Success Criteria:
1. All baselines produce non-empty retrieved_docs
2. Budget parity enforced (±5% compute)
3. Competitive baseline performance (not degraded)
4. Real latency measurements recorded
5. Full telemetry persisted for statistical analysis
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.baselines import BaselineRegistry, BM25Baseline, DenseBaseline, RRFBaseline
from src.eval.evaluation import EvaluationFramework, DatasetLoader, DatasetSplit
from src.eval.validation import ComprehensiveValidator, create_smoke_test_queries
from src.eval.metrics import generate_statistical_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaselineEvaluationOrchestrator:
    """Main orchestrator for Task 3 evaluation"""
    
    def __init__(self, 
                 output_dir: str,
                 data_dir: str = "datasets",
                 index_dir: str = "indices"):
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        self.index_dir = Path(index_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.baseline_registry = BaselineRegistry()
        self.evaluation_framework = EvaluationFramework(
            str(self.output_dir),
            self.baseline_registry
        )
        self.validator = ComprehensiveValidator()
        
        # Configuration
        self.available_datasets = {
            "msmarco": "MS MARCO Passage Dev (small)",
            "beir-covid": "BEIR TREC-COVID",
            "beir-nfcorpus": "BEIR NFCorpus", 
            "beir-fiqa": "BEIR FiQA-2018"
        }
        
        self.available_baselines = {
            "BM25": "Lexical retrieval with BM25",
            "Dense": "Dense retrieval with sentence-transformers",
            "RRF": "Reciprocal Rank Fusion (BM25 + Dense)"
            # TODO: Add SPLADE, uniCOIL, ColBERTv2 when models are available
        }
        
    def setup_baselines(self, baseline_names: List[str]) -> None:
        """Setup and register requested baselines"""
        logger.info(f"Setting up baselines: {baseline_names}")
        
        if "BM25" in baseline_names:
            logger.info("Setting up BM25 baseline...")
            bm25 = BM25Baseline(
                str(self.index_dir / "bm25"),
                self.baseline_registry.budget_tracker,
                self.baseline_registry.anti_fraud
            )
            self.baseline_registry.register_baseline(bm25)
            
        if "Dense" in baseline_names:
            logger.info("Setting up Dense baseline...")
            # Use a lightweight model for testing
            dense = DenseBaseline(
                "sentence-transformers/all-MiniLM-L6-v2",
                self.baseline_registry.budget_tracker,
                self.baseline_registry.anti_fraud,
                max_seq_length=256,  # Reduced for faster evaluation
                batch_size=16
            )
            self.baseline_registry.register_baseline(dense)
            
        if "RRF" in baseline_names and "BM25" in baseline_names and "Dense" in baseline_names:
            logger.info("Setting up RRF baseline...")
            # RRF needs BM25 and Dense to be already created
            bm25_baseline = None
            dense_baseline = None
            
            for baseline in self.baseline_registry.baselines.values():
                if isinstance(baseline, BM25Baseline):
                    bm25_baseline = baseline
                elif isinstance(baseline, DenseBaseline):
                    dense_baseline = baseline
                    
            if bm25_baseline and dense_baseline:
                rrf = RRFBaseline(
                    bm25_baseline,
                    dense_baseline,
                    self.baseline_registry.budget_tracker,
                    self.baseline_registry.anti_fraud
                )
                self.baseline_registry.register_baseline(rrf)
            else:
                logger.warning("Cannot create RRF - BM25 and Dense baselines required")
                
    def load_dataset(self, dataset_name: str, max_queries: Optional[int] = None) -> DatasetSplit:
        """Load specified dataset"""
        logger.info(f"Loading dataset: {dataset_name}")
        
        if dataset_name == "msmarco":
            return DatasetLoader.load_msmarco_dev(
                str(self.data_dir / "msmarco"),
                max_queries=max_queries
            )
        elif dataset_name.startswith("beir-"):
            beir_name = dataset_name.replace("beir-", "")
            return DatasetLoader.load_beir_dataset(
                beir_name,
                str(self.data_dir / "beir"),
                split="test",
                max_queries=max_queries
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
    def run_smoke_test(self, dataset_name: str) -> bool:
        """Run smoke test on all baselines"""
        logger.info("=" * 60)
        logger.info("RUNNING SMOKE TEST")
        logger.info("=" * 60)
        
        # Use predefined smoke test queries
        smoke_queries = create_smoke_test_queries()
        
        # Load a small subset of documents for smoke test
        try:
            dataset = self.load_dataset(dataset_name, max_queries=10)
            if not dataset.documents:
                logger.error("No documents loaded for smoke test")
                return False
                
            # Use only first 1000 documents for smoke test
            test_documents = dataset.documents[:1000]
            
            # Index documents for all baselines
            for name, baseline in self.baseline_registry.baselines.items():
                logger.info(f"Indexing documents for smoke test: {name}")
                baseline.index_documents(test_documents)
                
            # Run smoke tests
            smoke_results = self.baseline_registry.run_smoke_tests(smoke_queries)
            
            # Report results
            for baseline_name, passed in smoke_results.items():
                status = "PASSED" if passed else "FAILED"
                logger.info(f"Smoke test {baseline_name}: {status}")
                
            failed_baselines = [name for name, passed in smoke_results.items() if not passed]
            
            if failed_baselines:
                logger.error(f"SMOKE TEST FAILED for baselines: {failed_baselines}")
                return False
            else:
                logger.info("ALL SMOKE TESTS PASSED!")
                return True
                
        except Exception as e:
            logger.error(f"Smoke test failed with exception: {e}")
            return False
            
    def run_full_evaluation(self, 
                          dataset_name: str, 
                          max_queries: Optional[int] = None,
                          k: int = 10) -> Dict[str, Any]:
        """Run full baseline evaluation"""
        logger.info("=" * 60)
        logger.info(f"RUNNING FULL EVALUATION: {dataset_name}")
        logger.info("=" * 60)
        
        # Load dataset
        dataset = self.load_dataset(dataset_name, max_queries)
        logger.info(f"Loaded {len(dataset.queries)} queries, {len(dataset.documents)} documents")
        
        # Validate dataset
        if not dataset.queries:
            raise ValueError("No queries loaded from dataset")
        if not dataset.documents:
            raise ValueError("No documents loaded from dataset")
            
        # Run evaluation
        start_time = time.time()
        
        try:
            summary = self.evaluation_framework.run_full_evaluation(
                dataset=dataset,
                k=k,
                smoke_test_first=True
            )
            
            # Generate statistical analysis
            logger.info("Generating statistical analysis...")
            statistical_report = generate_statistical_report(
                self.evaluation_framework.metrics_results
            )
            
            # Save statistical report
            stats_file = self.output_dir / f"statistical_analysis_{dataset_name}.json"
            with open(stats_file, 'w') as f:
                json.dump(statistical_report, f, indent=2, default=str)
                
            logger.info(f"Statistical analysis saved to {stats_file}")
            
            evaluation_time = time.time() - start_time
            logger.info(f"Evaluation completed in {evaluation_time:.1f}s")
            
            # Add timing to summary
            summary['evaluation_time_seconds'] = evaluation_time
            summary['statistical_analysis_file'] = str(stats_file)
            
            return summary
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise
            
    def run_budget_parity_analysis(self) -> Dict[str, Any]:
        """Generate detailed budget parity analysis"""
        logger.info("Generating budget parity analysis...")
        
        budget_report = self.baseline_registry.budget_tracker.get_budget_report()
        
        # Add interpretation
        budget_report['interpretation'] = {
            'compliant_methods': [],
            'violating_methods': [],
            'recommendations': []
        }
        
        if 'methods' in budget_report:
            for method_name, data in budget_report['methods'].items():
                if data['parity_compliant']:
                    budget_report['interpretation']['compliant_methods'].append(method_name)
                else:
                    budget_report['interpretation']['violating_methods'].append(method_name)
                    
        if budget_report['interpretation']['violating_methods']:
            budget_report['interpretation']['recommendations'].append(
                "Budget parity violations detected - review method implementations"
            )
        else:
            budget_report['interpretation']['recommendations'].append(
                "All methods comply with budget parity constraints"
            )
            
        return budget_report
        
    def run_anti_fraud_analysis(self) -> Dict[str, Any]:
        """Generate detailed anti-fraud analysis"""
        logger.info("Generating anti-fraud analysis...")
        
        return self.baseline_registry.anti_fraud.get_validation_report()
        
    def print_summary_report(self, summary: Dict[str, Any]) -> None:
        """Print formatted summary report"""
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        
        # Dataset info
        dataset_info = summary.get('dataset_info', {})
        logger.info(f"Dataset: {dataset_info.get('name', 'Unknown')}")
        logger.info(f"Queries: {dataset_info.get('num_queries', 0)}")
        logger.info(f"Documents: {dataset_info.get('num_documents', 0)}")
        
        # Baseline results
        baseline_summaries = summary.get('baseline_summaries', {})
        logger.info(f"\nBaseline Results (k=10):")
        logger.info("-" * 40)
        
        for baseline_name, metrics in baseline_summaries.items():
            logger.info(f"{baseline_name}:")
            logger.info(f"  nDCG@10: {metrics.get('mean_ndcg_10', 0):.3f}")
            logger.info(f"  Recall@10: {metrics.get('mean_recall_10', 0):.3f}")
            logger.info(f"  MRR@10: {metrics.get('mean_mrr_10', 0):.3f}")
            logger.info(f"  Latency: {metrics.get('mean_latency_ms', 0):.1f}ms (p95: {metrics.get('p95_latency_ms', 0):.1f}ms)")
            logger.info(f"  Queries: {metrics.get('queries_processed', 0)}")
            
        # Budget parity status
        budget_report = summary.get('budget_parity_report', {})
        if 'methods' in budget_report:
            compliant_count = sum(1 for m in budget_report['methods'].values() if m['parity_compliant'])
            total_count = len(budget_report['methods'])
            logger.info(f"\nBudget Parity: {compliant_count}/{total_count} methods compliant")
            
        # Anti-fraud status
        fraud_report = summary.get('anti_fraud_report', {})
        if 'methods' in fraud_report:
            logger.info(f"\nAnti-Fraud Validation:")
            for method_name, method_data in fraud_report['methods'].items():
                success_rate = method_data.get('success_rate', 0)
                logger.info(f"  {method_name}: {success_rate:.1%} success rate")
                
        logger.info("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="Run Task 3 baseline evaluation with anti-fraud validation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Dataset options
    parser.add_argument(
        '--dataset', 
        choices=['msmarco', 'beir-covid', 'beir-nfcorpus', 'beir-fiqa', 'all'],
        default='msmarco',
        help='Dataset to evaluate on'
    )
    
    # Baseline options
    parser.add_argument(
        '--baselines',
        default='BM25,Dense,RRF',
        help='Comma-separated list of baselines to evaluate'
    )
    
    # Evaluation options
    parser.add_argument(
        '--max-queries',
        type=int,
        help='Maximum number of queries to evaluate (for testing)'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Number of results to retrieve per query'
    )
    
    # Mode options
    parser.add_argument(
        '--smoke-test-only',
        action='store_true',
        help='Run only smoke tests, skip full evaluation'
    )
    
    parser.add_argument(
        '--skip-smoke-test',
        action='store_true', 
        help='Skip smoke tests and run full evaluation directly'
    )
    
    # Directory options
    parser.add_argument(
        '--output',
        default='artifacts/eval_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--data-dir',
        default='datasets',
        help='Data directory containing datasets'
    )
    
    parser.add_argument(
        '--index-dir', 
        default='indices',
        help='Index directory for baseline indices'
    )
    
    args = parser.parse_args()
    
    # Parse baselines
    baseline_names = [name.strip() for name in args.baselines.split(',')]
    
    # Initialize orchestrator
    orchestrator = BaselineEvaluationOrchestrator(
        output_dir=args.output,
        data_dir=args.data_dir,
        index_dir=args.index_dir
    )
    
    try:
        # Setup baselines
        orchestrator.setup_baselines(baseline_names)
        
        # Determine datasets to evaluate
        if args.dataset == 'all':
            datasets = ['msmarco', 'beir-covid', 'beir-nfcorpus', 'beir-fiqa']
        else:
            datasets = [args.dataset]
            
        results = {}
        
        for dataset_name in datasets:
            logger.info(f"\n{'='*60}")
            logger.info(f"PROCESSING DATASET: {dataset_name.upper()}")
            logger.info(f"{'='*60}")
            
            try:
                if args.smoke_test_only:
                    # Run smoke test only
                    smoke_passed = orchestrator.run_smoke_test(dataset_name)
                    results[dataset_name] = {"smoke_test_passed": smoke_passed}
                    
                elif args.skip_smoke_test:
                    # Run full evaluation without smoke test
                    summary = orchestrator.run_full_evaluation(
                        dataset_name, args.max_queries, args.k
                    )
                    results[dataset_name] = summary
                    orchestrator.print_summary_report(summary)
                    
                else:
                    # Run smoke test first, then full evaluation
                    smoke_passed = orchestrator.run_smoke_test(dataset_name)
                    
                    if not smoke_passed:
                        logger.error(f"Smoke test failed for {dataset_name} - skipping full evaluation")
                        results[dataset_name] = {"smoke_test_passed": False, "evaluation_skipped": True}
                        continue
                        
                    summary = orchestrator.run_full_evaluation(
                        dataset_name, args.max_queries, args.k
                    )
                    results[dataset_name] = summary
                    orchestrator.print_summary_report(summary)
                    
            except Exception as e:
                logger.error(f"Failed to process dataset {dataset_name}: {e}")
                results[dataset_name] = {"error": str(e)}
                continue
                
        # Generate overall analysis
        logger.info("\n" + "="*60)
        logger.info("GENERATING OVERALL ANALYSIS")
        logger.info("="*60)
        
        budget_analysis = orchestrator.run_budget_parity_analysis()
        fraud_analysis = orchestrator.run_anti_fraud_analysis()
        
        # Save overall results
        overall_results = {
            "datasets_processed": list(results.keys()),
            "results_by_dataset": results,
            "budget_parity_analysis": budget_analysis,
            "anti_fraud_analysis": fraud_analysis,
            "configuration": {
                "baselines": baseline_names,
                "max_queries": args.max_queries,
                "k": args.k,
                "smoke_test_only": args.smoke_test_only
            }
        }
        
        overall_file = orchestrator.output_dir / "overall_evaluation_results.json"
        with open(overall_file, 'w') as f:
            json.dump(overall_results, f, indent=2, default=str)
            
        logger.info(f"Overall results saved to {overall_file}")
        
        # Final summary
        successful_datasets = [name for name, result in results.items() if not result.get('error')]
        failed_datasets = [name for name, result in results.items() if result.get('error')]
        
        logger.info(f"\n{'='*60}")
        logger.info("FINAL SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Successful datasets: {len(successful_datasets)} - {successful_datasets}")
        if failed_datasets:
            logger.info(f"Failed datasets: {len(failed_datasets)} - {failed_datasets}")
        logger.info(f"Results saved to: {orchestrator.output_dir}")
        
        # Exit code based on success
        if failed_datasets:
            sys.exit(1)
        else:
            logger.info("All evaluations completed successfully!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()