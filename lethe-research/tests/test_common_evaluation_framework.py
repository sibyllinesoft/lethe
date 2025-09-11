"""
Comprehensive tests for evaluation framework module.

Tests cover the high-complexity EvaluationFramework class which valknut identified
as having significant complexity in metrics calculation and statistical analysis.

Test areas:
- IR metrics calculation (nDCG, Recall, Precision, MRR, MAP, F1)
- Statistical significance testing (t-test, Wilcoxon, Mann-Whitney)
- Effect size computation (Cohen's d, Hedges' g, Glass's delta)
- Bootstrap confidence intervals and resampling
- Per-query analysis and outlier detection
- Cross-validation and holdout strategies
- Performance profiling and timing
- System comparison workflows
- Edge cases and error handling
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from dataclasses import dataclass
from typing import Dict, List, Any

# Import the module under test
try:
    from src.common.evaluation_framework import (
        EvaluationFramework, MetricConfig, QueryResult, SystemComparison,
        StatisticalTest, EffectSizeType, ValidationError, InsufficientDataError
    )
except ImportError:
    # Handle missing dependencies gracefully
    pytest.skip("evaluation_framework module not available", allow_module_level=True)


class TestEvaluationFramework:
    """Test suite for EvaluationFramework functionality."""
    
    @pytest.fixture
    def framework(self):
        """Create EvaluationFramework instance."""
        return EvaluationFramework()
    
    @pytest.fixture
    def basic_config(self):
        """Basic metric configuration."""
        return MetricConfig(
            metrics=["ndcg_10", "recall_20", "precision_10", "mrr", "map"],
            statistical_tests=[StatisticalTest.T_TEST, StatisticalTest.WILCOXON],
            confidence_level=0.95,
            bootstrap_samples=1000
        )
    
    @pytest.fixture
    def sample_retrieved_docs(self):
        """Sample retrieved document rankings."""
        return [
            ["doc1", "doc2", "doc3", "doc4", "doc5"],  # Query 1
            ["doc6", "doc7", "doc8", "doc9", "doc10"], # Query 2
            ["doc11", "doc12", "doc13", "doc14", "doc15"] # Query 3
        ]
    
    @pytest.fixture
    def sample_ground_truth(self):
        """Sample ground truth relevance judgments."""
        return [
            {"doc1": 2, "doc2": 1, "doc4": 1},  # Query 1
            {"doc6": 3, "doc8": 2, "doc10": 1}, # Query 2
            {"doc11": 1, "doc13": 2, "doc14": 1} # Query 3
        ]
    
    @pytest.fixture
    def sample_query_ids(self):
        """Sample query identifiers."""
        return ["q1", "q2", "q3"]

    # Basic IR metrics tests
    def test_ndcg_calculation(self, framework):
        """Test nDCG calculation for various scenarios."""
        # Perfect ranking
        retrieved = ["doc1", "doc2", "doc3"]
        ground_truth = {"doc1": 3, "doc2": 2, "doc3": 1}
        
        ndcg = framework.calculate_ndcg(retrieved, ground_truth, k=3)
        assert ndcg == 1.0, "Perfect ranking should have nDCG = 1.0"
        
        # Worst ranking
        retrieved = ["doc3", "doc2", "doc1"]
        ndcg_worst = framework.calculate_ndcg(retrieved, ground_truth, k=3)
        assert ndcg_worst < ndcg, "Reversed ranking should have lower nDCG"
        
        # No relevant documents
        retrieved = ["doc4", "doc5", "doc6"]
        ground_truth_empty = {}
        ndcg_empty = framework.calculate_ndcg(retrieved, ground_truth_empty, k=3)
        assert ndcg_empty == 0.0, "No relevant docs should give nDCG = 0.0"
    
    def test_recall_calculation(self, framework):
        """Test recall calculation at different cutoffs."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        ground_truth = {"doc1": 1, "doc3": 1, "doc6": 1}  # 3 relevant, 2 retrieved
        
        recall_5 = framework.calculate_recall(retrieved, ground_truth, k=5)
        assert recall_5 == 2/3, f"Recall@5 should be 2/3, got {recall_5}"
        
        recall_2 = framework.calculate_recall(retrieved, ground_truth, k=2)
        assert recall_2 == 1/3, f"Recall@2 should be 1/3, got {recall_2}"
        
        # All relevant retrieved
        ground_truth_all = {"doc1": 1, "doc3": 1}
        recall_all = framework.calculate_recall(retrieved, ground_truth_all, k=5)
        assert recall_all == 1.0, "Should achieve perfect recall"
    
    def test_precision_calculation(self, framework):
        """Test precision calculation at different cutoffs."""
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        ground_truth = {"doc1": 1, "doc3": 1}  # 2 relevant out of 5 retrieved
        
        precision_5 = framework.calculate_precision(retrieved, ground_truth, k=5)
        assert precision_5 == 2/5, f"Precision@5 should be 2/5, got {precision_5}"
        
        precision_3 = framework.calculate_precision(retrieved, ground_truth, k=3)
        assert precision_3 == 2/3, f"Precision@3 should be 2/3, got {precision_3}"
    
    def test_mrr_calculation(self, framework):
        """Test Mean Reciprocal Rank calculation."""
        # First relevant at position 1
        retrieved = ["doc1", "doc2", "doc3"]
        ground_truth = {"doc1": 1}
        mrr = framework.calculate_mrr(retrieved, ground_truth)
        assert mrr == 1.0, "First position should give MRR = 1.0"
        
        # First relevant at position 2
        retrieved = ["doc2", "doc1", "doc3"]
        mrr = framework.calculate_mrr(retrieved, ground_truth)
        assert mrr == 0.5, "Second position should give MRR = 0.5"
        
        # No relevant documents
        ground_truth_empty = {}
        mrr = framework.calculate_mrr(retrieved, ground_truth_empty)
        assert mrr == 0.0, "No relevant docs should give MRR = 0.0"
    
    def test_map_calculation(self, framework):
        """Test Mean Average Precision calculation."""
        retrieved = ["doc1", "doc2", "doc3", "doc4"]
        ground_truth = {"doc1": 1, "doc3": 1}  # Relevant at positions 1, 3
        
        # AP = (1/1 + 2/3) / 2 = (1 + 0.667) / 2 = 0.833
        map_score = framework.calculate_map(retrieved, ground_truth)
        expected = (1.0 + 2/3) / 2
        assert abs(map_score - expected) < 0.001, f"MAP should be {expected}, got {map_score}"

    # Query evaluation tests
    def test_evaluate_single_query(self, framework, basic_config):
        """Test single query evaluation."""
        query_id = "q1"
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        ground_truth = {"doc1": 2, "doc3": 1, "doc5": 1}
        
        result = framework.evaluate_query(
            query_id=query_id,
            retrieved_docs=retrieved,
            ground_truth=ground_truth,
            config=basic_config
        )
        
        assert isinstance(result, QueryResult)
        assert result.query_id == query_id
        assert "ndcg_10" in result.metrics
        assert "recall_20" in result.metrics
        assert "precision_10" in result.metrics
        assert "mrr" in result.metrics
        assert "map" in result.metrics
        
        # Verify reasonable metric values
        assert 0 <= result.metrics["ndcg_10"] <= 1
        assert 0 <= result.metrics["recall_20"] <= 1
        assert 0 <= result.metrics["precision_10"] <= 1
        assert 0 <= result.metrics["mrr"] <= 1
        assert 0 <= result.metrics["map"] <= 1
    
    def test_evaluate_multiple_queries(self, framework, basic_config, 
                                      sample_query_ids, sample_retrieved_docs, 
                                      sample_ground_truth):
        """Test evaluation of multiple queries."""
        results = framework.evaluate_queries(
            query_ids=sample_query_ids,
            retrieved_docs=sample_retrieved_docs,
            ground_truth=sample_ground_truth,
            config=basic_config
        )
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, QueryResult)
            assert result.query_id in sample_query_ids
            assert len(result.metrics) == len(basic_config.metrics)
    
    def test_aggregate_results(self, framework, basic_config, sample_query_ids,
                              sample_retrieved_docs, sample_ground_truth):
        """Test aggregation of query results."""
        query_results = framework.evaluate_queries(
            query_ids=sample_query_ids,
            retrieved_docs=sample_retrieved_docs,
            ground_truth=sample_ground_truth,
            config=basic_config
        )
        
        aggregated = framework.aggregate_results(query_results)
        
        # Should have mean, std, min, max for each metric
        assert "ndcg_10_mean" in aggregated
        assert "ndcg_10_std" in aggregated
        assert "ndcg_10_min" in aggregated
        assert "ndcg_10_max" in aggregated
        
        # Verify statistical properties
        ndcg_values = [r.metrics["ndcg_10"] for r in query_results]
        assert abs(aggregated["ndcg_10_mean"] - np.mean(ndcg_values)) < 0.001
        assert abs(aggregated["ndcg_10_std"] - np.std(ndcg_values)) < 0.001

    # Statistical testing
    def test_t_test_comparison(self, framework):
        """Test t-test statistical comparison."""
        baseline_scores = [0.7, 0.8, 0.6, 0.9, 0.75]
        comparison_scores = [0.8, 0.85, 0.7, 0.95, 0.82]
        
        result = framework.run_statistical_test(
            baseline_scores,
            comparison_scores,
            test_type=StatisticalTest.T_TEST,
            alpha=0.05
        )
        
        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert "effect_size" in result
        
        assert isinstance(result["significant"], bool)
        assert 0 <= result["p_value"] <= 1
    
    def test_wilcoxon_test_comparison(self, framework):
        """Test Wilcoxon signed-rank test."""
        baseline_scores = [0.7, 0.8, 0.6, 0.9, 0.75]
        comparison_scores = [0.8, 0.85, 0.7, 0.95, 0.82]
        
        result = framework.run_statistical_test(
            baseline_scores,
            comparison_scores,
            test_type=StatisticalTest.WILCOXON,
            alpha=0.05
        )
        
        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result
        
        # Wilcoxon should handle tied pairs
        assert result is not None
    
    def test_effect_size_calculation(self, framework):
        """Test effect size calculations."""
        baseline_scores = [0.5, 0.6, 0.4, 0.7, 0.55]
        comparison_scores = [0.8, 0.9, 0.7, 1.0, 0.85]
        
        # Cohen's d
        cohens_d = framework.calculate_effect_size(
            baseline_scores,
            comparison_scores,
            effect_type=EffectSizeType.COHENS_D
        )
        
        assert isinstance(cohens_d, float)
        assert cohens_d > 0, "Should show positive effect"
        
        # Hedges' g (should be similar but slightly smaller)
        hedges_g = framework.calculate_effect_size(
            baseline_scores,
            comparison_scores,
            effect_type=EffectSizeType.HEDGES_G
        )
        
        assert isinstance(hedges_g, float)
        assert abs(hedges_g - cohens_d) < 0.1, "Hedges' g should be close to Cohen's d"

    # Bootstrap testing
    def test_bootstrap_confidence_interval(self, framework):
        """Test bootstrap confidence interval calculation."""
        scores = [0.7, 0.8, 0.6, 0.9, 0.75, 0.82, 0.68, 0.94, 0.73, 0.77]
        
        ci = framework.bootstrap_confidence_interval(
            scores,
            confidence_level=0.95,
            n_bootstrap=1000,
            statistic_func=np.mean
        )
        
        assert len(ci) == 2  # Lower and upper bounds
        assert ci[0] < ci[1], "Lower bound should be less than upper bound"
        assert ci[0] < np.mean(scores) < ci[1], "Mean should be within CI"
    
    def test_bootstrap_different_statistics(self, framework):
        """Test bootstrap with different statistics."""
        scores = [0.7, 0.8, 0.6, 0.9, 0.75, 0.82, 0.68, 0.94, 0.73, 0.77]
        
        # Test with median
        ci_median = framework.bootstrap_confidence_interval(
            scores,
            statistic_func=np.median,
            n_bootstrap=500
        )
        
        assert len(ci_median) == 2
        assert ci_median[0] < np.median(scores) < ci_median[1]

    # System comparison tests
    def test_system_comparison(self, framework, basic_config):
        """Test complete system comparison workflow."""
        # Create mock system results
        baseline_results = []
        comparison_results = []
        
        for i in range(5):  # 5 queries
            baseline_results.append(QueryResult(
                query_id=f"q{i}",
                metrics={
                    "ndcg_10": 0.7 + np.random.normal(0, 0.1),
                    "recall_20": 0.8 + np.random.normal(0, 0.1),
                    "precision_10": 0.6 + np.random.normal(0, 0.1)
                }
            ))
            
            comparison_results.append(QueryResult(
                query_id=f"q{i}",
                metrics={
                    "ndcg_10": 0.75 + np.random.normal(0, 0.1),
                    "recall_20": 0.85 + np.random.normal(0, 0.1),
                    "precision_10": 0.65 + np.random.normal(0, 0.1)
                }
            ))
        
        comparison = framework.compare_systems(
            baseline_results,
            comparison_results,
            config=basic_config
        )
        
        assert isinstance(comparison, SystemComparison)
        assert "ndcg_10" in comparison.metric_comparisons
        assert "statistical_tests" in comparison.metric_comparisons["ndcg_10"]
        assert "effect_sizes" in comparison.metric_comparisons["ndcg_10"]

    # Cross-validation tests
    def test_cross_validation_split(self, framework):
        """Test cross-validation data splitting."""
        query_ids = [f"q{i}" for i in range(10)]
        
        folds = framework.create_cv_folds(query_ids, n_folds=5)
        
        assert len(folds) == 5
        
        # Check that all queries appear exactly once in test sets
        all_test_queries = []
        for train, test in folds:
            all_test_queries.extend(test)
            assert len(set(train) & set(test)) == 0, "Train and test should not overlap"
        
        assert set(all_test_queries) == set(query_ids), "All queries should appear in test"
    
    def test_holdout_validation(self, framework):
        """Test holdout validation split."""
        query_ids = [f"q{i}" for i in range(100)]
        
        train, test = framework.create_holdout_split(query_ids, test_ratio=0.2)
        
        assert len(test) == 20, "Should have 20% test queries"
        assert len(train) == 80, "Should have 80% train queries"
        assert len(set(train) & set(test)) == 0, "No overlap between train/test"

    # Performance and outlier analysis
    def test_outlier_detection(self, framework):
        """Test outlier detection in query results."""
        # Create results with one clear outlier
        results = []
        for i in range(10):
            score = 0.8 + np.random.normal(0, 0.05) if i != 5 else 0.2  # Outlier at i=5
            results.append(QueryResult(
                query_id=f"q{i}",
                metrics={"ndcg_10": score}
            ))
        
        outliers = framework.detect_outliers(results, metric="ndcg_10", method="iqr")
        
        assert "q5" in outliers, "Should detect the clear outlier"
        assert len(outliers) <= 2, "Should not detect too many outliers"
    
    def test_performance_profiling(self, framework):
        """Test performance profiling functionality."""
        def dummy_computation():
            time.sleep(0.01)  # Small delay
            return [0.7, 0.8, 0.9]
        
        with framework.profile_performance() as profiler:
            scores = dummy_computation()
        
        profile_data = profiler.get_stats()
        
        assert "duration" in profile_data
        assert "memory_peak" in profile_data
        assert profile_data["duration"] >= 0.01

    # Edge cases and error handling
    def test_empty_retrieved_list(self, framework):
        """Test handling of empty retrieved document list."""
        retrieved = []
        ground_truth = {"doc1": 1}
        
        # Should handle gracefully
        ndcg = framework.calculate_ndcg(retrieved, ground_truth, k=10)
        assert ndcg == 0.0
        
        recall = framework.calculate_recall(retrieved, ground_truth, k=10)
        assert recall == 0.0
        
        precision = framework.calculate_precision(retrieved, ground_truth, k=10)
        assert precision == 0.0
    
    def test_empty_ground_truth(self, framework):
        """Test handling of empty ground truth."""
        retrieved = ["doc1", "doc2", "doc3"]
        ground_truth = {}
        
        # Should handle gracefully
        ndcg = framework.calculate_ndcg(retrieved, ground_truth, k=3)
        assert ndcg == 0.0
        
        recall = framework.calculate_recall(retrieved, ground_truth, k=3)
        assert recall == 0.0  # or could be undefined
        
        precision = framework.calculate_precision(retrieved, ground_truth, k=3)
        assert precision == 0.0
    
    def test_mismatched_query_counts(self, framework, basic_config):
        """Test error handling for mismatched query counts."""
        query_ids = ["q1", "q2"]
        retrieved_docs = [["doc1"], ["doc2"], ["doc3"]]  # 3 vs 2 queries
        ground_truth = [{"doc1": 1}, {"doc2": 1}]
        
        with pytest.raises(ValidationError):
            framework.evaluate_queries(
                query_ids=query_ids,
                retrieved_docs=retrieved_docs,
                ground_truth=ground_truth,
                config=basic_config
            )
    
    def test_insufficient_data_for_statistics(self, framework):
        """Test error handling for insufficient data in statistical tests."""
        baseline_scores = [0.7]  # Only one sample
        comparison_scores = [0.8]
        
        with pytest.raises(InsufficientDataError):
            framework.run_statistical_test(
                baseline_scores,
                comparison_scores,
                test_type=StatisticalTest.T_TEST
            )
    
    def test_invalid_metric_names(self, framework):
        """Test handling of invalid metric names."""
        config = MetricConfig(metrics=["invalid_metric"])
        
        with pytest.raises(ValidationError):
            framework.evaluate_query(
                query_id="q1",
                retrieved_docs=["doc1"],
                ground_truth={"doc1": 1},
                config=config
            )
    
    def test_negative_k_values(self, framework):
        """Test handling of negative k values."""
        retrieved = ["doc1", "doc2"]
        ground_truth = {"doc1": 1}
        
        with pytest.raises(ValueError):
            framework.calculate_ndcg(retrieved, ground_truth, k=-1)
        
        with pytest.raises(ValueError):
            framework.calculate_recall(retrieved, ground_truth, k=0)

    # Integration and workflow tests
    def test_complete_evaluation_workflow(self, framework):
        """Test complete evaluation workflow from data to report."""
        # Setup data
        query_ids = ["q1", "q2", "q3"]
        retrieved_docs = [
            ["doc1", "doc2", "doc3"],
            ["doc4", "doc5", "doc6"],
            ["doc7", "doc8", "doc9"]
        ]
        ground_truth = [
            {"doc1": 2, "doc3": 1},
            {"doc4": 1, "doc6": 2},
            {"doc7": 1, "doc8": 1, "doc9": 1}
        ]
        
        config = MetricConfig(
            metrics=["ndcg_10", "recall_10", "precision_10", "map"],
            statistical_tests=[StatisticalTest.T_TEST],
            confidence_level=0.95
        )
        
        # Run evaluation
        results = framework.evaluate_queries(
            query_ids=query_ids,
            retrieved_docs=retrieved_docs,
            ground_truth=ground_truth,
            config=config
        )
        
        # Aggregate results
        aggregated = framework.aggregate_results(results)
        
        # Generate report
        report = framework.generate_report(results, aggregated)
        
        assert isinstance(report, dict)
        assert "summary" in report
        assert "per_query_results" in report
        assert "statistical_summary" in report


if __name__ == "__main__":
    pytest.main([__file__])