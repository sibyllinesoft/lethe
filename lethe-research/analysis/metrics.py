#!/usr/bin/env python3
"""
Lethe Research Metrics Implementation
====================================

Comprehensive evaluation metrics for the Lethe hybrid retrieval system.
Supports all hypothesis testing requirements with statistical rigor.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import ndcg_score
import json
import warnings
from collections import defaultdict

@dataclass
class QueryResult:
    """Single query evaluation result"""
    query_id: str
    session_id: str
    domain: str
    complexity: str
    ground_truth_docs: List[str]
    retrieved_docs: List[str]
    relevance_scores: List[float]
    latency_ms: float
    memory_mb: float
    entities_covered: List[str]
    contradictions: List[str]
    timestamp: str

@dataclass 
class EvaluationMetrics:
    """Complete evaluation metrics for a single configuration"""
    config_name: str
    
    # H1: Quality metrics
    ndcg_at_k: Dict[int, float]
    recall_at_k: Dict[int, float] 
    mrr_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    
    # H2: Efficiency metrics
    latency_percentiles: Dict[int, float]  # P50, P90, P95, P99
    memory_stats: Dict[str, float]  # peak_mb, avg_mb
    throughput_qps: float
    
    # H3: Coverage metrics
    coverage_at_n: Dict[int, float]
    diversity_indices: Dict[str, float]
    uniqueness_score: float
    
    # H4: Consistency metrics
    contradiction_rate: float
    hallucination_score: float
    consistency_index: float
    
    # Statistical metadata
    n_queries: int
    confidence_intervals: Dict[str, Tuple[float, float]]
    effect_sizes: Dict[str, float]

class MetricsCalculator:
    """Main class for computing all evaluation metrics"""
    
    def __init__(self, bootstrap_samples: int = 1000, confidence_level: float = 0.95):
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def compute_all_metrics(self, results: List[QueryResult], config_name: str) -> EvaluationMetrics:
        """Compute comprehensive metrics for a configuration"""
        
        return EvaluationMetrics(
            config_name=config_name,
            ndcg_at_k=self._compute_ndcg(results),
            recall_at_k=self._compute_recall(results),
            mrr_at_k=self._compute_mrr(results),
            precision_at_k=self._compute_precision(results),
            latency_percentiles=self._compute_latency_percentiles(results),
            memory_stats=self._compute_memory_stats(results),
            throughput_qps=self._compute_throughput(results),
            coverage_at_n=self._compute_coverage(results),
            diversity_indices=self._compute_diversity(results),
            uniqueness_score=self._compute_uniqueness(results),
            contradiction_rate=self._compute_contradiction_rate(results),
            hallucination_score=self._compute_hallucination_score(results),
            consistency_index=self._compute_consistency_index(results),
            n_queries=len(results),
            confidence_intervals=self._compute_confidence_intervals(results),
            effect_sizes={}  # Computed during comparison
        )
    
    def _compute_ndcg(self, results: List[QueryResult]) -> Dict[int, float]:
        """Compute NDCG@k for multiple k values"""
        ndcg_scores = {}
        k_values = [5, 10, 20]
        
        for k in k_values:
            scores = []
            for result in results:
                if len(result.retrieved_docs) == 0:
                    scores.append(0.0)
                    continue
                    
                # Create relevance array for retrieved docs
                relevance = []
                for doc in result.retrieved_docs[:k]:
                    if doc in result.ground_truth_docs:
                        # Use provided relevance score or default to 1.0
                        idx = min(len(result.relevance_scores) - 1, len(relevance))
                        relevance.append(result.relevance_scores[idx] if result.relevance_scores else 1.0)
                    else:
                        relevance.append(0.0)
                
                if len(relevance) == 0 or sum(relevance) == 0:
                    scores.append(0.0)
                else:
                    # Compute NDCG@k
                    try:
                        ndcg = ndcg_score([relevance], [relevance], k=min(k, len(relevance)))
                        scores.append(ndcg)
                    except:
                        scores.append(0.0)
            
            ndcg_scores[k] = np.mean(scores) if scores else 0.0
        
        return ndcg_scores
    
    def _compute_recall(self, results: List[QueryResult]) -> Dict[int, float]:
        """Compute Recall@k for multiple k values"""
        recall_scores = {}
        k_values = [5, 10, 20]
        
        for k in k_values:
            scores = []
            for result in results:
                if len(result.ground_truth_docs) == 0:
                    continue
                    
                retrieved_at_k = set(result.retrieved_docs[:k])
                relevant_docs = set(result.ground_truth_docs)
                
                recall = len(retrieved_at_k & relevant_docs) / len(relevant_docs)
                scores.append(recall)
            
            recall_scores[k] = np.mean(scores) if scores else 0.0
            
        return recall_scores
    
    def _compute_mrr(self, results: List[QueryResult]) -> Dict[int, float]:
        """Compute Mean Reciprocal Rank@k"""
        mrr_scores = {}
        k_values = [10]
        
        for k in k_values:
            rr_scores = []
            for result in results:
                rr = 0.0
                for i, doc in enumerate(result.retrieved_docs[:k], 1):
                    if doc in result.ground_truth_docs:
                        rr = 1.0 / i
                        break
                rr_scores.append(rr)
            
            mrr_scores[k] = np.mean(rr_scores) if rr_scores else 0.0
            
        return mrr_scores
    
    def _compute_precision(self, results: List[QueryResult]) -> Dict[int, float]:
        """Compute Precision@k for multiple k values"""
        precision_scores = {}
        k_values = [5, 10]
        
        for k in k_values:
            scores = []
            for result in results:
                if len(result.retrieved_docs) == 0:
                    scores.append(0.0)
                    continue
                    
                retrieved_at_k = set(result.retrieved_docs[:k])
                relevant_docs = set(result.ground_truth_docs)
                
                precision = len(retrieved_at_k & relevant_docs) / min(k, len(result.retrieved_docs))
                scores.append(precision)
            
            precision_scores[k] = np.mean(scores) if scores else 0.0
            
        return precision_scores
    
    def _compute_latency_percentiles(self, results: List[QueryResult]) -> Dict[int, float]:
        """Compute latency percentiles"""
        latencies = [r.latency_ms for r in results]
        if not latencies:
            return {p: 0.0 for p in [50, 90, 95, 99]}
            
        return {
            p: np.percentile(latencies, p) 
            for p in [50, 90, 95, 99]
        }
    
    def _compute_memory_stats(self, results: List[QueryResult]) -> Dict[str, float]:
        """Compute memory usage statistics"""
        memory_values = [r.memory_mb for r in results]
        if not memory_values:
            return {"peak_mb": 0.0, "avg_mb": 0.0}
            
        return {
            "peak_mb": np.max(memory_values),
            "avg_mb": np.mean(memory_values)
        }
    
    def _compute_throughput(self, results: List[QueryResult]) -> float:
        """Compute queries per second throughput"""
        if not results:
            return 0.0
            
        total_time_s = sum(r.latency_ms for r in results) / 1000.0
        return len(results) / total_time_s if total_time_s > 0 else 0.0
    
    def _compute_coverage(self, results: List[QueryResult]) -> Dict[int, float]:
        """Compute entity coverage@N metrics"""
        coverage_scores = {}
        n_values = [10, 20, 50]
        
        for n in n_values:
            scores = []
            for result in results:
                # Coverage = unique entities covered / total entities in session
                if len(result.entities_covered) == 0:
                    scores.append(0.0)
                else:
                    coverage = min(len(set(result.entities_covered[:n])), n) / n
                    scores.append(coverage)
            
            coverage_scores[n] = np.mean(scores) if scores else 0.0
            
        return coverage_scores
    
    def _compute_diversity(self, results: List[QueryResult]) -> Dict[str, float]:
        """Compute diversity indices (Shannon entropy, Simpson index)"""
        # Shannon entropy of entity distribution
        all_entities = []
        for result in results:
            all_entities.extend(result.entities_covered)
        
        if not all_entities:
            return {"shannon_entropy": 0.0, "simpson_index": 0.0}
        
        entity_counts = defaultdict(int)
        for entity in all_entities:
            entity_counts[entity] += 1
        
        total = len(all_entities)
        probs = [count / total for count in entity_counts.values()]
        
        # Shannon entropy
        shannon = -sum(p * np.log2(p) for p in probs if p > 0)
        
        # Simpson index (1 - sum of squares of probabilities)  
        simpson = 1 - sum(p ** 2 for p in probs)
        
        return {
            "shannon_entropy": shannon,
            "simpson_index": simpson
        }
    
    def _compute_uniqueness(self, results: List[QueryResult]) -> float:
        """Compute result uniqueness (1 - average Jaccard similarity)"""
        if len(results) < 2:
            return 1.0
            
        similarities = []
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                set_i = set(results[i].retrieved_docs)
                set_j = set(results[j].retrieved_docs)
                
                if len(set_i | set_j) == 0:
                    similarities.append(0.0)
                else:
                    jaccard = len(set_i & set_j) / len(set_i | set_j)
                    similarities.append(jaccard)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        return 1.0 - avg_similarity
    
    def _compute_contradiction_rate(self, results: List[QueryResult]) -> float:
        """Compute percentage of queries with contradictions"""
        if not results:
            return 0.0
            
        with_contradictions = sum(1 for r in results if len(r.contradictions) > 0)
        return with_contradictions / len(results)
    
    def _compute_hallucination_score(self, results: List[QueryResult]) -> float:
        """Compute hallucination score (placeholder - requires human annotation)"""
        # This would be computed from human annotations in practice
        # For now, return a placeholder based on contradiction rate
        contradiction_rate = self._compute_contradiction_rate(results)
        return contradiction_rate * 0.5  # Approximate relationship
    
    def _compute_consistency_index(self, results: List[QueryResult]) -> float:
        """Compute consistency index across similar queries"""
        # Placeholder implementation - would use semantic similarity in practice
        if len(results) < 2:
            return 1.0
            
        # Use entity overlap as proxy for consistency
        entity_sets = [set(r.entities_covered) for r in results]
        similarities = []
        
        for i in range(len(entity_sets)):
            for j in range(i + 1, len(entity_sets)):
                if len(entity_sets[i] | entity_sets[j]) == 0:
                    similarities.append(0.0)
                else:
                    jaccard = len(entity_sets[i] & entity_sets[j]) / len(entity_sets[i] | entity_sets[j])
                    similarities.append(jaccard)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _compute_confidence_intervals(self, results: List[QueryResult]) -> Dict[str, Tuple[float, float]]:
        """Compute bootstrap confidence intervals for key metrics"""
        if len(results) < 10:  # Need minimum sample size
            return {}
        
        cis = {}
        
        # Bootstrap CI for NDCG@10
        ndcg_10_samples = []
        for _ in range(self.bootstrap_samples):
            sample = np.random.choice(results, len(results), replace=True)
            ndcg_10 = self._compute_ndcg(sample)[10]
            ndcg_10_samples.append(ndcg_10)
        
        lower = np.percentile(ndcg_10_samples, (self.alpha/2) * 100)
        upper = np.percentile(ndcg_10_samples, (1 - self.alpha/2) * 100)
        cis["ndcg_at_10"] = (lower, upper)
        
        # Bootstrap CI for P95 latency
        latency_samples = []
        for _ in range(self.bootstrap_samples):
            sample = np.random.choice(results, len(results), replace=True)
            p95 = np.percentile([r.latency_ms for r in sample], 95)
            latency_samples.append(p95)
        
        lower = np.percentile(latency_samples, (self.alpha/2) * 100)
        upper = np.percentile(latency_samples, (1 - self.alpha/2) * 100)
        cis["latency_p95"] = (lower, upper)
        
        return cis

class StatisticalComparator:
    """Statistical comparison between configurations"""
    
    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level
    
    def compare_configurations(self, 
                             baseline_metrics: EvaluationMetrics,
                             treatment_metrics: EvaluationMetrics,
                             baseline_results: List[QueryResult],
                             treatment_results: List[QueryResult]) -> Dict[str, Any]:
        """Comprehensive statistical comparison"""
        
        comparison = {
            "baseline": baseline_metrics.config_name,
            "treatment": treatment_metrics.config_name,
            "n_baseline": len(baseline_results),
            "n_treatment": len(treatment_results),
            "tests": {},
            "effect_sizes": {},
            "practical_significance": {}
        }
        
        # Compare NDCG@10
        baseline_ndcg = [self._compute_single_ndcg(r, k=10) for r in baseline_results]
        treatment_ndcg = [self._compute_single_ndcg(r, k=10) for r in treatment_results]
        
        comparison["tests"]["ndcg_at_10"] = self._wilcoxon_test(baseline_ndcg, treatment_ndcg)
        comparison["effect_sizes"]["ndcg_at_10"] = self._cohens_d(baseline_ndcg, treatment_ndcg)
        
        # Compare P95 latency
        baseline_latency = [r.latency_ms for r in baseline_results]
        treatment_latency = [r.latency_ms for r in treatment_results]
        
        comparison["tests"]["latency_p95"] = self._wilcoxon_test(baseline_latency, treatment_latency)
        comparison["effect_sizes"]["latency_p95"] = self._cohens_d(baseline_latency, treatment_latency)
        
        # Compare contradiction rate
        baseline_contradictions = [len(r.contradictions) > 0 for r in baseline_results]
        treatment_contradictions = [len(r.contradictions) > 0 for r in treatment_results]
        
        comparison["tests"]["contradiction_rate"] = self._proportion_test(
            baseline_contradictions, treatment_contradictions)
        
        return comparison
    
    def _compute_single_ndcg(self, result: QueryResult, k: int) -> float:
        """Helper to compute NDCG for single result"""
        if len(result.retrieved_docs) == 0:
            return 0.0
            
        relevance = []
        for doc in result.retrieved_docs[:k]:
            if doc in result.ground_truth_docs:
                relevance.append(1.0)
            else:
                relevance.append(0.0)
        
        if sum(relevance) == 0:
            return 0.0
            
        try:
            return ndcg_score([relevance], [relevance], k=min(k, len(relevance)))
        except:
            return 0.0
    
    def _wilcoxon_test(self, baseline: List[float], treatment: List[float]) -> Dict[str, float]:
        """Wilcoxon signed-rank test"""
        try:
            statistic, p_value = stats.wilcoxon(treatment, baseline, alternative='greater')
            return {
                "statistic": float(statistic),
                "p_value": float(p_value),
                "significant": p_value < self.alpha
            }
        except Exception as e:
            return {"error": str(e), "significant": False}
    
    def _cohens_d(self, baseline: List[float], treatment: List[float]) -> float:
        """Cohen's d effect size"""
        try:
            mean_diff = np.mean(treatment) - np.mean(baseline)
            pooled_std = np.sqrt((np.var(baseline) + np.var(treatment)) / 2)
            return mean_diff / pooled_std if pooled_std > 0 else 0.0
        except:
            return 0.0
    
    def _proportion_test(self, baseline: List[bool], treatment: List[bool]) -> Dict[str, float]:
        """Test for difference in proportions"""
        try:
            prop_baseline = np.mean(baseline)
            prop_treatment = np.mean(treatment)
            
            # Simple z-test for proportions
            n1, n2 = len(baseline), len(treatment)
            p_pooled = (sum(baseline) + sum(treatment)) / (n1 + n2)
            
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
            z_stat = (prop_treatment - prop_baseline) / se if se > 0 else 0
            p_value = 1 - stats.norm.cdf(abs(z_stat))
            
            return {
                "z_statistic": float(z_stat),
                "p_value": float(p_value),
                "significant": p_value < self.alpha,
                "prop_baseline": float(prop_baseline),
                "prop_treatment": float(prop_treatment)
            }
        except Exception as e:
            return {"error": str(e), "significant": False}

def load_results_from_json(filepath: str) -> List[QueryResult]:
    """Load evaluation results from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    results = []
    for item in data:
        result = QueryResult(
            query_id=item["query_id"],
            session_id=item["session_id"], 
            domain=item["domain"],
            complexity=item["complexity"],
            ground_truth_docs=item["ground_truth_docs"],
            retrieved_docs=item["retrieved_docs"],
            relevance_scores=item.get("relevance_scores", []),
            latency_ms=item["latency_ms"],
            memory_mb=item["memory_mb"],
            entities_covered=item.get("entities_covered", []),
            contradictions=item.get("contradictions", []),
            timestamp=item["timestamp"]
        )
        results.append(result)
    
    return results

def save_metrics_to_json(metrics: EvaluationMetrics, filepath: str):
    """Save metrics to JSON file"""
    # Convert to serializable format
    metrics_dict = {
        "config_name": metrics.config_name,
        "ndcg_at_k": metrics.ndcg_at_k,
        "recall_at_k": metrics.recall_at_k,
        "mrr_at_k": metrics.mrr_at_k,
        "precision_at_k": metrics.precision_at_k,
        "latency_percentiles": metrics.latency_percentiles,
        "memory_stats": metrics.memory_stats,
        "throughput_qps": metrics.throughput_qps,
        "coverage_at_n": metrics.coverage_at_n,
        "diversity_indices": metrics.diversity_indices,
        "uniqueness_score": metrics.uniqueness_score,
        "contradiction_rate": metrics.contradiction_rate,
        "hallucination_score": metrics.hallucination_score,
        "consistency_index": metrics.consistency_index,
        "n_queries": metrics.n_queries,
        "confidence_intervals": metrics.confidence_intervals,
        "effect_sizes": metrics.effect_sizes
    }
    
    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

if __name__ == "__main__":
    print("Lethe Research Metrics Implementation")
    print("=====================================")
    print("This module provides comprehensive evaluation metrics")
    print("for the Lethe hybrid retrieval system research.")
    print("\nUsage:")
    print("  from metrics import MetricsCalculator, StatisticalComparator")
    print("  calculator = MetricsCalculator()")
    print("  metrics = calculator.compute_all_metrics(results, 'config_name')")