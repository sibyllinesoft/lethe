#!/usr/bin/env python3
"""
Task 3 Evaluation Framework
==========================

Comprehensive evaluation framework for baseline suite with statistical rigor.

Features:
- MS MARCO/BEIR dataset integration
- Standard IR metrics (nDCG@{10,5}, Recall@{10,20}, MRR@10)
- Per-query metrics for statistical analysis
- JSONL persistence for full telemetry
- Hardware profiling and reproducibility tracking

Key Components:
- MetricsCalculator: Compute nDCG, Recall, MRR, etc.
- DatasetLoader: Load MS MARCO/BEIR with ground truth
- EvaluationFramework: Orchestrate full evaluation pipeline
- ResultsPersistence: JSONL export with full telemetry
"""

import json
import time
import logging
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import gc

from .baselines import (
    BaselineRetrieverV2, BaselineResult, EvaluationQuery, 
    RetrievalDocument, BaselineRegistry
)

logger = logging.getLogger(__name__)

@dataclass 
class MetricsResult:
    """Individual metrics result for a single query"""
    query_id: str
    baseline_name: str
    
    # Ranking metrics
    ndcg_10: float
    ndcg_5: float
    recall_10: float
    recall_20: float
    mrr_10: float
    
    # Additional metrics
    precision_10: float
    map_score: float
    
    # Query characteristics
    num_relevant: int
    num_retrieved: int
    query_length: int
    
    # Performance metrics
    latency_ms: float
    memory_mb: float
    flops_estimate: int

@dataclass
class DatasetSplit:
    """Dataset split with queries and relevance judgments"""
    name: str
    queries: List[EvaluationQuery]
    documents: List[RetrievalDocument] 
    relevance_judgments: Dict[str, Dict[str, int]]  # query_id -> {doc_id: relevance}
    metadata: Dict[str, Any] = field(default_factory=dict)

class MetricsCalculator:
    """Compute standard IR evaluation metrics"""
    
    @staticmethod
    def dcg_at_k(relevances: List[int], k: int) -> float:
        """Compute Discounted Cumulative Gain at k"""
        relevances = relevances[:k]
        if not relevances:
            return 0.0
            
        dcg = relevances[0]
        for i, rel in enumerate(relevances[1:], 2):
            dcg += rel / np.log2(i)
            
        return dcg
    
    @staticmethod
    def ndcg_at_k(relevances: List[int], k: int) -> float:
        """Compute Normalized Discounted Cumulative Gain at k"""
        dcg = MetricsCalculator.dcg_at_k(relevances, k)
        
        # Ideal DCG (sort relevances in descending order)
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = MetricsCalculator.dcg_at_k(ideal_relevances, k)
        
        if idcg == 0:
            return 0.0
            
        return dcg / idcg
    
    @staticmethod
    def recall_at_k(relevant_retrieved: int, total_relevant: int, k: int = None) -> float:
        """Compute Recall at k"""
        if total_relevant == 0:
            return 0.0
        return relevant_retrieved / total_relevant
    
    @staticmethod
    def precision_at_k(relevant_retrieved: int, k: int) -> float:
        """Compute Precision at k"""
        if k == 0:
            return 0.0
        return relevant_retrieved / k
    
    @staticmethod
    def mrr_at_k(relevances: List[int], k: int) -> float:
        """Compute Mean Reciprocal Rank at k"""
        relevances = relevances[:k]
        for i, rel in enumerate(relevances, 1):
            if rel > 0:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def average_precision(relevances: List[int]) -> float:
        """Compute Average Precision"""
        if not relevances:
            return 0.0
            
        relevant_count = 0
        ap_sum = 0.0
        
        for i, rel in enumerate(relevances, 1):
            if rel > 0:
                relevant_count += 1
                ap_sum += relevant_count / i
                
        total_relevant = sum(1 for r in relevances if r > 0)
        if total_relevant == 0:
            return 0.0
            
        return ap_sum / total_relevant
    
    @classmethod
    def compute_metrics(cls, 
                       retrieved_docs: List[str],
                       relevance_judgments: Dict[str, int],
                       k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Compute all standard metrics for a single query.
        
        Args:
            retrieved_docs: List of retrieved document IDs in rank order
            relevance_judgments: Dict mapping doc_id to relevance score
            k_values: List of k values to compute metrics for
            
        Returns:
            Dictionary with all computed metrics
        """
        # Map retrieved docs to relevance scores
        relevances = [relevance_judgments.get(doc_id, 0) for doc_id in retrieved_docs]
        
        # Count relevant documents
        total_relevant = sum(1 for score in relevance_judgments.values() if score > 0)
        
        metrics = {}
        
        # Compute metrics for each k
        for k in k_values:
            relevances_at_k = relevances[:k]
            relevant_at_k = sum(1 for r in relevances_at_k if r > 0)
            
            metrics[f'ndcg_{k}'] = cls.ndcg_at_k(relevances, k)
            metrics[f'recall_{k}'] = cls.recall_at_k(relevant_at_k, total_relevant, k)
            metrics[f'precision_{k}'] = cls.precision_at_k(relevant_at_k, k)
            metrics[f'mrr_{k}'] = cls.mrr_at_k(relevances, k)
            
        # Overall metrics
        metrics['map'] = cls.average_precision(relevances)
        metrics['num_relevant'] = total_relevant
        metrics['num_retrieved'] = len(retrieved_docs)
        
        return metrics

class DatasetLoader:
    """Load and prepare evaluation datasets (MS MARCO, BEIR)"""
    
    @staticmethod
    def load_msmarco_dev(data_dir: str, max_queries: Optional[int] = None) -> DatasetSplit:
        """Load MS MARCO passage dev set"""
        data_path = Path(data_dir)
        
        logger.info(f"Loading MS MARCO dev from {data_path}")
        
        # Load queries
        queries_file = data_path / "queries.dev.small.tsv"
        queries = []
        
        if queries_file.exists():
            with open(queries_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_queries and i >= max_queries:
                        break
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        qid, qtext = parts[0], parts[1]
                        query = EvaluationQuery(
                            query_id=qid,
                            text=qtext,
                            domain="passage_retrieval",
                            complexity="medium"
                        )
                        queries.append(query)
        else:
            logger.warning(f"Queries file not found: {queries_file}")
            
        # Load relevance judgments
        qrels_file = data_path / "qrels.dev.small.tsv"
        relevance_judgments = defaultdict(dict)
        
        if qrels_file.exists():
            with open(qrels_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        qid, _, doc_id, relevance = parts
                        relevance_judgments[qid][doc_id] = int(relevance)
        else:
            logger.warning(f"Qrels file not found: {qrels_file}")
            
        # Load documents (collection)
        docs_file = data_path / "collection.tsv"
        documents = []
        
        if docs_file.exists():
            logger.info("Loading document collection...")
            with open(docs_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i % 100000 == 0:
                        logger.info(f"Loaded {i} documents...")
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        doc_id, content = parts[0], parts[1]
                        doc = RetrievalDocument(
                            doc_id=doc_id,
                            content=content,
                            kind="passage",
                            metadata={"source": "msmarco"}
                        )
                        documents.append(doc)
        else:
            logger.warning(f"Collection file not found: {docs_file}")
            
        # Update queries with relevance judgments
        for query in queries:
            if query.query_id in relevance_judgments:
                query.relevance_judgments = dict(relevance_judgments[query.query_id])
                query.ground_truth_docs = [
                    doc_id for doc_id, rel in query.relevance_judgments.items() if rel > 0
                ]
                
        logger.info(f"Loaded MS MARCO: {len(queries)} queries, {len(documents)} documents")
        
        return DatasetSplit(
            name="msmarco_dev",
            queries=queries,
            documents=documents,
            relevance_judgments=dict(relevance_judgments),
            metadata={
                "dataset": "ms_marco_passage",
                "split": "dev_small",
                "max_queries": max_queries
            }
        )
    
    @staticmethod
    def load_beir_dataset(dataset_name: str, 
                         data_dir: str, 
                         split: str = "test",
                         max_queries: Optional[int] = None) -> DatasetSplit:
        """Load BEIR dataset (TREC-COVID, NFCorpus, FiQA-2018)"""
        data_path = Path(data_dir) / dataset_name
        
        logger.info(f"Loading BEIR {dataset_name} {split} from {data_path}")
        
        # BEIR datasets have standard structure
        queries_file = data_path / f"queries.jsonl"
        corpus_file = data_path / f"corpus.jsonl"
        qrels_file = data_path / f"qrels/{split}.tsv"
        
        # Load queries
        queries = []
        if queries_file.exists():
            with open(queries_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if max_queries and i >= max_queries:
                        break
                    data = json.loads(line.strip())
                    query = EvaluationQuery(
                        query_id=data['_id'],
                        text=data['text'],
                        domain=dataset_name,
                        complexity="medium"
                    )
                    queries.append(query)
        else:
            logger.warning(f"Queries file not found: {queries_file}")
            
        # Load corpus
        documents = []
        if corpus_file.exists():
            with open(corpus_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i % 10000 == 0:
                        logger.info(f"Loaded {i} documents...")
                    data = json.loads(line.strip())
                    
                    # Combine title and text
                    content = data.get('title', '') + ' ' + data.get('text', '')
                    content = content.strip()
                    
                    doc = RetrievalDocument(
                        doc_id=data['_id'],
                        content=content,
                        kind="document",
                        metadata={
                            "source": dataset_name,
                            "title": data.get('title', ''),
                            "text": data.get('text', '')
                        }
                    )
                    documents.append(doc)
        else:
            logger.warning(f"Corpus file not found: {corpus_file}")
            
        # Load relevance judgments
        relevance_judgments = defaultdict(dict)
        if qrels_file.exists():
            with open(qrels_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        qid, _, doc_id, relevance = parts
                        relevance_judgments[qid][doc_id] = int(relevance)
        else:
            logger.warning(f"Qrels file not found: {qrels_file}")
            
        # Update queries with relevance judgments
        for query in queries:
            if query.query_id in relevance_judgments:
                query.relevance_judgments = dict(relevance_judgments[query.query_id])
                query.ground_truth_docs = [
                    doc_id for doc_id, rel in query.relevance_judgments.items() if rel > 0
                ]
                
        logger.info(f"Loaded {dataset_name}: {len(queries)} queries, {len(documents)} documents")
        
        return DatasetSplit(
            name=f"{dataset_name}_{split}",
            queries=queries,
            documents=documents,
            relevance_judgments=dict(relevance_judgments),
            metadata={
                "dataset": dataset_name,
                "split": split,
                "max_queries": max_queries
            }
        )

class ResultsPersistence:
    """Handle JSONL persistence and telemetry export"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create run metadata
        self.run_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self.run_metadata = {
            "run_id": self.run_id,
            "timestamp": time.time(),
            "start_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "task": "task3_baseline_evaluation"
        }
        
        logger.info(f"Results will be saved to {self.output_dir} (run_id: {self.run_id})")
        
    def save_baseline_results(self, 
                            baseline_name: str,
                            results: List[BaselineResult],
                            dataset_name: str) -> None:
        """Save baseline results to JSONL"""
        filename = f"{baseline_name}_{dataset_name}_{self.run_id}.jsonl"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            for result in results:
                # Convert to dict and add run metadata
                result_dict = asdict(result)
                result_dict['run_id'] = self.run_id
                result_dict['dataset'] = dataset_name
                
                f.write(json.dumps(result_dict) + '\n')
                
        logger.info(f"Saved {len(results)} results to {filepath}")
        
    def save_metrics_results(self,
                           metrics_results: List[MetricsResult],
                           dataset_name: str) -> None:
        """Save computed metrics to JSONL"""
        filename = f"metrics_{dataset_name}_{self.run_id}.jsonl" 
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            for metrics in metrics_results:
                # Convert to dict and add run metadata
                metrics_dict = asdict(metrics)
                metrics_dict['run_id'] = self.run_id
                metrics_dict['dataset'] = dataset_name
                
                f.write(json.dumps(metrics_dict) + '\n')
                
        logger.info(f"Saved {len(metrics_results)} metrics to {filepath}")
        
    def save_summary_report(self, 
                          summary: Dict[str, Any],
                          dataset_name: str) -> None:
        """Save evaluation summary report"""
        filename = f"summary_{dataset_name}_{self.run_id}.json"
        filepath = self.output_dir / filename
        
        # Add run metadata
        summary['run_metadata'] = self.run_metadata
        summary['dataset'] = dataset_name
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        logger.info(f"Saved summary report to {filepath}")
        
    def get_run_manifest(self) -> Dict[str, Any]:
        """Get manifest of all files created in this run"""
        files = list(self.output_dir.glob(f"*_{self.run_id}.*"))
        
        return {
            "run_id": self.run_id,
            "output_dir": str(self.output_dir),
            "files_created": [str(f.name) for f in files],
            "total_files": len(files),
            "manifest_created": time.strftime('%Y-%m-%d %H:%M:%S')
        }

class EvaluationFramework:
    """Main evaluation orchestrator"""
    
    def __init__(self, 
                 output_dir: str,
                 baseline_registry: BaselineRegistry):
        self.baseline_registry = baseline_registry
        self.persistence = ResultsPersistence(output_dir)
        self.metrics_calculator = MetricsCalculator()
        
        # Evaluation state
        self.current_dataset: Optional[DatasetSplit] = None
        self.evaluation_results: Dict[str, List[BaselineResult]] = {}
        self.metrics_results: List[MetricsResult] = []
        
    def run_full_evaluation(self, 
                          dataset: DatasetSplit,
                          k: int = 10,
                          smoke_test_first: bool = True) -> Dict[str, Any]:
        """
        Run complete baseline evaluation on dataset.
        
        Args:
            dataset: Dataset split to evaluate on
            k: Number of results to retrieve per query
            smoke_test_first: Run smoke tests before full evaluation
            
        Returns:
            Summary report of evaluation
        """
        logger.info(f"Starting evaluation on {dataset.name} with {len(dataset.queries)} queries")
        
        self.current_dataset = dataset
        start_time = time.time()
        
        # Index all baselines first
        logger.info("Indexing documents for all baselines...")
        for name, baseline in self.baseline_registry.baselines.items():
            if not baseline.is_indexed:
                logger.info(f"Indexing {name}...")
                baseline.index_documents(dataset.documents)
                
        # Set compute budget baseline (use BM25)
        if "BM25" in self.baseline_registry.baselines:
            sample_query = dataset.queries[0] if dataset.queries else None
            if sample_query:
                bm25_baseline = self.baseline_registry.baselines["BM25"]
                baseline_flops = bm25_baseline.estimate_flops(sample_query, k)
                self.baseline_registry.budget_tracker.set_baseline_budget(baseline_flops)
        
        # Run smoke tests
        if smoke_test_first:
            smoke_queries = dataset.queries[:5]  # Use first 5 queries for smoke test
            smoke_results = self.baseline_registry.run_smoke_tests(smoke_queries)
            
            failed_baselines = [name for name, passed in smoke_results.items() if not passed]
            if failed_baselines:
                logger.error(f"Smoke tests failed for baselines: {failed_baselines}")
                raise RuntimeError("Smoke tests failed - aborting evaluation")
                
            logger.info("All smoke tests passed!")
        
        # Run evaluation on each baseline
        for baseline_name, baseline in self.baseline_registry.baselines.items():
            logger.info(f"Evaluating {baseline_name}...")
            baseline_results = self._evaluate_baseline(baseline, dataset.queries, k)
            
            self.evaluation_results[baseline_name] = baseline_results
            
            # Save results immediately
            self.persistence.save_baseline_results(
                baseline_name, baseline_results, dataset.name)
                
            logger.info(f"Completed {baseline_name}: {len(baseline_results)} queries processed")
            
        # Compute metrics for all baselines
        logger.info("Computing evaluation metrics...")
        self._compute_all_metrics(dataset)
        
        # Save metrics
        self.persistence.save_metrics_results(self.metrics_results, dataset.name)
        
        # Generate summary report
        summary = self._generate_summary_report(dataset)
        self.persistence.save_summary_report(summary, dataset.name)
        
        total_time = time.time() - start_time
        logger.info(f"Evaluation completed in {total_time:.1f}s")
        
        return summary
        
    def _evaluate_baseline(self, 
                          baseline: BaselineRetrieverV2,
                          queries: List[EvaluationQuery],
                          k: int) -> List[BaselineResult]:
        """Evaluate single baseline on all queries"""
        results = []
        
        for i, query in enumerate(queries, 1):
            if i % 50 == 0:
                logger.info(f"  Query {i}/{len(queries)}")
                
            try:
                # Run evaluation with telemetry
                result = baseline.evaluate_with_telemetry(query, k)
                results.append(result)
                
                # Memory management for large evaluations
                if i % 100 == 0:
                    gc.collect()
                    
            except Exception as e:
                logger.error(f"Error evaluating {baseline.name} on query {query.query_id}: {e}")
                continue
                
        return results
        
    def _compute_all_metrics(self, dataset: DatasetSplit) -> None:
        """Compute evaluation metrics for all baseline results"""
        self.metrics_results = []
        
        for baseline_name, results in self.evaluation_results.items():
            for result in results:
                # Get relevance judgments for this query
                query_judgments = dataset.relevance_judgments.get(result.query_id, {})
                
                if not query_judgments:
                    logger.warning(f"No relevance judgments for query {result.query_id}")
                    continue
                
                # Compute metrics
                metrics_dict = self.metrics_calculator.compute_metrics(
                    result.retrieved_docs, 
                    query_judgments
                )
                
                # Create MetricsResult
                metrics_result = MetricsResult(
                    query_id=result.query_id,
                    baseline_name=baseline_name,
                    ndcg_10=metrics_dict.get('ndcg_10', 0.0),
                    ndcg_5=metrics_dict.get('ndcg_5', 0.0),
                    recall_10=metrics_dict.get('recall_10', 0.0),
                    recall_20=metrics_dict.get('recall_20', 0.0),
                    mrr_10=metrics_dict.get('mrr_10', 0.0),
                    precision_10=metrics_dict.get('precision_10', 0.0),
                    map_score=metrics_dict.get('map', 0.0),
                    num_relevant=metrics_dict.get('num_relevant', 0),
                    num_retrieved=metrics_dict.get('num_retrieved', 0),
                    query_length=len(result.query_text.split()),
                    latency_ms=result.latency_ms,
                    memory_mb=result.memory_mb,
                    flops_estimate=result.flops_estimate
                )
                
                self.metrics_results.append(metrics_result)
                
    def _generate_summary_report(self, dataset: DatasetSplit) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        
        # Aggregate metrics by baseline
        baseline_summaries = {}
        
        for baseline_name in self.baseline_registry.baselines.keys():
            baseline_metrics = [m for m in self.metrics_results if m.baseline_name == baseline_name]
            
            if not baseline_metrics:
                continue
                
            # Compute mean metrics
            summary = {
                "queries_processed": len(baseline_metrics),
                "mean_ndcg_10": np.mean([m.ndcg_10 for m in baseline_metrics]),
                "mean_ndcg_5": np.mean([m.ndcg_5 for m in baseline_metrics]),  
                "mean_recall_10": np.mean([m.recall_10 for m in baseline_metrics]),
                "mean_recall_20": np.mean([m.recall_20 for m in baseline_metrics]),
                "mean_mrr_10": np.mean([m.mrr_10 for m in baseline_metrics]),
                "mean_precision_10": np.mean([m.precision_10 for m in baseline_metrics]),
                "mean_map": np.mean([m.map_score for m in baseline_metrics]),
                
                # Performance metrics
                "mean_latency_ms": np.mean([m.latency_ms for m in baseline_metrics]),
                "p95_latency_ms": np.percentile([m.latency_ms for m in baseline_metrics], 95),
                "mean_memory_mb": np.mean([m.memory_mb for m in baseline_metrics]),
                "mean_flops": np.mean([m.flops_estimate for m in baseline_metrics])
            }
            
            baseline_summaries[baseline_name] = summary
            
        # Overall summary
        report = {
            "dataset_info": {
                "name": dataset.name,
                "num_queries": len(dataset.queries),
                "num_documents": len(dataset.documents),
                "metadata": dataset.metadata
            },
            "evaluation_config": {
                "k": 10,  # Hardcoded for now
                "baselines_evaluated": list(self.baseline_registry.baselines.keys()),
                "smoke_tests_enabled": True
            },
            "baseline_summaries": baseline_summaries,
            "budget_parity_report": self.baseline_registry.budget_tracker.get_budget_report(),
            "anti_fraud_report": self.baseline_registry.anti_fraud.get_validation_report(),
            "run_info": self.persistence.run_metadata,
            "file_manifest": self.persistence.get_run_manifest()
        }
        
        return report