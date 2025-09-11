"""
Comprehensive tests for common data structures module.

Tests cover:
- PerformanceMetrics aggregation and calculations
- QueryInfo initialization and validation
- DocumentInfo with embeddings handling
- RetrievalResult data integrity
- EvaluationResult with legacy conversions
- AggregatedResults statistical calculations
- Edge cases and error conditions
"""

import pytest
import time
import statistics
import numpy as np
from dataclasses import fields
from unittest.mock import Mock

from src.common.data_structures import (
    PerformanceMetrics,
    QueryInfo,
    DocumentInfo,
    RetrievalResult,
    EvaluationResult,
    AggregatedResults
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics functionality."""
    
    def test_basic_creation(self):
        """Test basic PerformanceMetrics creation."""
        metrics = PerformanceMetrics(
            latency_ms=100.5,
            memory_mb=64.2,
            cpu_percent=75.3
        )
        
        assert metrics.latency_ms == 100.5
        assert metrics.memory_mb == 64.2
        assert metrics.cpu_percent == 75.3
        assert metrics.memory_peak_mb == 64.2  # Should default to memory_mb
        assert metrics.measurements_count == 1
        assert isinstance(metrics.timestamp, float)
        
    def test_default_values(self):
        """Test PerformanceMetrics with default values."""
        metrics = PerformanceMetrics()
        
        assert metrics.latency_ms == 0.0
        assert metrics.memory_mb == 0.0
        assert metrics.cpu_percent == 0.0
        assert metrics.memory_peak_mb == 0.0
        assert metrics.flops_estimate is None
        assert metrics.execution_context is None
        assert metrics.measurements_count == 1
        
    def test_post_init_memory_peak(self):
        """Test __post_init__ sets memory_peak_mb correctly."""
        # When memory_peak_mb is None, it should be set to memory_mb
        metrics = PerformanceMetrics(memory_mb=50.0)
        assert metrics.memory_peak_mb == 50.0
        
        # When memory_peak_mb is provided, it should be preserved
        metrics = PerformanceMetrics(memory_mb=50.0, memory_peak_mb=75.0)
        assert metrics.memory_peak_mb == 75.0
        
    def test_add_measurement_basic(self):
        """Test adding measurements for aggregation."""
        metrics1 = PerformanceMetrics(
            latency_ms=100.0,
            memory_mb=50.0,
            cpu_percent=60.0,
            measurements_count=1
        )
        
        metrics2 = PerformanceMetrics(
            latency_ms=200.0,
            memory_mb=70.0,
            cpu_percent=80.0,
            measurements_count=1
        )
        
        metrics1.add_measurement(metrics2)
        
        # Should compute weighted averages
        assert metrics1.latency_ms == 150.0  # (100*1 + 200*1) / 2
        assert metrics1.memory_mb == 60.0    # (50*1 + 70*1) / 2
        assert metrics1.cpu_percent == 70.0  # (60*1 + 80*1) / 2
        assert metrics1.measurements_count == 2
        
    def test_add_measurement_weighted(self):
        """Test adding measurements with different counts."""
        metrics1 = PerformanceMetrics(
            latency_ms=100.0,
            memory_mb=50.0,
            cpu_percent=60.0,
            measurements_count=3
        )
        
        metrics2 = PerformanceMetrics(
            latency_ms=200.0,
            memory_mb=70.0,
            cpu_percent=80.0,
            measurements_count=2
        )
        
        metrics1.add_measurement(metrics2)
        
        # Should compute weighted averages based on measurement counts
        expected_latency = (100.0 * 3 + 200.0 * 2) / 5  # 140.0
        expected_memory = (50.0 * 3 + 70.0 * 2) / 5      # 58.0
        expected_cpu = (60.0 * 3 + 80.0 * 2) / 5         # 68.0
        
        assert metrics1.latency_ms == expected_latency
        assert metrics1.memory_mb == expected_memory
        assert metrics1.cpu_percent == expected_cpu
        assert metrics1.measurements_count == 5
        
    def test_add_measurement_peak_memory(self):
        """Test peak memory handling in add_measurement."""
        metrics1 = PerformanceMetrics(
            memory_mb=50.0,
            memory_peak_mb=60.0
        )
        
        metrics2 = PerformanceMetrics(
            memory_mb=40.0,
            memory_peak_mb=80.0  # Higher peak
        )
        
        metrics1.add_measurement(metrics2)
        
        # Should take max of peak values
        assert metrics1.memory_peak_mb == 80.0
        
        # Test with None values
        metrics3 = PerformanceMetrics(memory_mb=30.0, memory_peak_mb=None)
        metrics4 = PerformanceMetrics(memory_mb=35.0, memory_peak_mb=40.0)
        
        # Should handle None gracefully
        metrics3.add_measurement(metrics4)
        assert metrics3.memory_peak_mb == 40.0
        
    def test_custom_fields(self):
        """Test custom fields like flops_estimate and execution_context."""
        metrics = PerformanceMetrics(
            latency_ms=50.0,
            flops_estimate=1000000,
            execution_context="test_environment"
        )
        
        assert metrics.flops_estimate == 1000000
        assert metrics.execution_context == "test_environment"


class TestQueryInfo:
    """Test QueryInfo functionality."""
    
    def test_basic_creation(self):
        """Test basic QueryInfo creation."""
        query = QueryInfo(
            query_id="q1",
            text="What is machine learning?",
            domain="AI",
            complexity="high"
        )
        
        assert query.query_id == "q1"
        assert query.text == "What is machine learning?"
        assert query.domain == "AI"
        assert query.complexity == "high"
        assert query.query_length == 4  # Should be calculated in __post_init__
        assert query.session_id is None
        assert len(query.ground_truth_docs) == 0
        assert len(query.relevance_judgments) == 0
        
    def test_default_values(self):
        """Test QueryInfo with default values."""
        query = QueryInfo(query_id="q1", text="test query")
        
        assert query.domain == "general"
        assert query.complexity == "medium"
        assert query.session_id is None
        assert query.ground_truth_docs == []
        assert query.relevance_judgments == {}
        assert query.metadata == {}
        
    def test_query_length_calculation(self):
        """Test automatic query length calculation."""
        # Normal text
        query = QueryInfo(query_id="q1", text="This is a test query")
        assert query.query_length == 5
        
        # Empty text
        query_empty = QueryInfo(query_id="q2", text="")
        assert query_empty.query_length == 0
        
        # Single word
        query_single = QueryInfo(query_id="q3", text="test")
        assert query_single.query_length == 1
        
        # Explicit query_length should be preserved
        query_explicit = QueryInfo(
            query_id="q4", 
            text="test", 
            query_length=10
        )
        assert query_explicit.query_length == 10  # Should not be overwritten
        
    def test_with_ground_truth(self):
        """Test QueryInfo with ground truth data."""
        ground_truth_docs = ["doc1", "doc2", "doc3"]
        relevance_judgments = {"doc1": 2, "doc2": 1, "doc3": 0}
        
        query = QueryInfo(
            query_id="q1",
            text="test query",
            ground_truth_docs=ground_truth_docs,
            relevance_judgments=relevance_judgments
        )
        
        assert query.ground_truth_docs == ground_truth_docs
        assert query.relevance_judgments == relevance_judgments
        
    def test_with_metadata(self):
        """Test QueryInfo with additional metadata."""
        metadata = {
            "source": "test_suite",
            "difficulty": "hard",
            "keywords": ["ML", "AI"]
        }
        
        query = QueryInfo(
            query_id="q1",
            text="test query",
            metadata=metadata
        )
        
        assert query.metadata == metadata


class TestDocumentInfo:
    """Test DocumentInfo functionality."""
    
    def test_basic_creation(self):
        """Test basic DocumentInfo creation."""
        doc = DocumentInfo(
            doc_id="doc1",
            content="This is a test document."
        )
        
        assert doc.doc_id == "doc1"
        assert doc.content == "This is a test document."
        assert doc.kind == "text"  # Default value
        assert doc.embedding is None
        assert doc.metadata == {}
        
    def test_with_embedding(self):
        """Test DocumentInfo with numpy embedding."""
        embedding = np.array([0.1, 0.2, 0.3, 0.4])
        
        doc = DocumentInfo(
            doc_id="doc1",
            content="test",
            embedding=embedding
        )
        
        assert doc.embedding is not None
        np.testing.assert_array_equal(doc.embedding, embedding)
        
    def test_different_kinds(self):
        """Test DocumentInfo with different document kinds."""
        kinds = ["text", "code", "tool_output"]
        
        for kind in kinds:
            doc = DocumentInfo(
                doc_id=f"doc_{kind}",
                content="test content",
                kind=kind
            )
            assert doc.kind == kind
            
    def test_with_metadata(self):
        """Test DocumentInfo with metadata."""
        metadata = {
            "source": "wikipedia",
            "timestamp": "2024-01-01",
            "length": 1000
        }
        
        doc = DocumentInfo(
            doc_id="doc1",
            content="test",
            metadata=metadata
        )
        
        assert doc.metadata == metadata


class TestRetrievalResult:
    """Test RetrievalResult functionality."""
    
    def test_basic_creation(self):
        """Test basic RetrievalResult creation."""
        result = RetrievalResult(
            doc_id="doc1",
            score=0.85,
            rank=1
        )
        
        assert result.doc_id == "doc1"
        assert result.score == 0.85
        assert result.rank == 1
        assert result.content is None
        assert result.kind is None
        assert result.metadata == {}
        
    def test_with_optional_fields(self):
        """Test RetrievalResult with optional fields."""
        metadata = {"source": "bm25", "normalized": True}
        
        result = RetrievalResult(
            doc_id="doc1",
            score=0.75,
            rank=2,
            content="This is the document content",
            kind="text",
            metadata=metadata
        )
        
        assert result.content == "This is the document content"
        assert result.kind == "text"
        assert result.metadata == metadata


class TestEvaluationResult:
    """Test EvaluationResult functionality."""
    
    def create_sample_evaluation_result(self):
        """Create a sample EvaluationResult for testing."""
        performance = PerformanceMetrics(
            latency_ms=150.5,
            memory_mb=45.2,
            cpu_percent=60.0
        )
        
        ranking_metrics = {
            "ndcg_10": 0.75,
            "recall_10": 0.80,
            "precision_10": 0.70
        }
        
        retrieval_results = [
            RetrievalResult("doc1", 0.95, 1),
            RetrievalResult("doc2", 0.85, 2),
            RetrievalResult("doc3", 0.75, 3)
        ]
        
        return EvaluationResult(
            query_id="q1",
            system_name="hybrid_system",
            query_text="test query",
            retrieved_docs=["doc1", "doc2", "doc3"],
            retrieval_results=retrieval_results,
            performance=performance,
            ranking_metrics=ranking_metrics
        )
        
    def test_basic_creation(self):
        """Test basic EvaluationResult creation."""
        result = self.create_sample_evaluation_result()
        
        assert result.query_id == "q1"
        assert result.system_name == "hybrid_system"
        assert result.query_text == "test query"
        assert len(result.retrieved_docs) == 3
        assert len(result.retrieval_results) == 3
        assert result.performance is not None
        assert len(result.ranking_metrics) == 3
        assert result.validation_passed == True
        
    def test_ranking_metric_methods(self):
        """Test ranking metric getter and setter methods."""
        result = self.create_sample_evaluation_result()
        
        # Test get_ranking_metric
        assert result.get_ranking_metric("ndcg_10") == 0.75
        assert result.get_ranking_metric("nonexistent", 0.5) == 0.5
        
        # Test set_ranking_metric
        result.set_ranking_metric("map_score", 0.65)
        assert result.ranking_metrics["map_score"] == 0.65
        
    def test_performance_summary(self):
        """Test performance summary extraction."""
        result = self.create_sample_evaluation_result()
        summary = result.get_performance_summary()
        
        expected_keys = ["latency_ms", "memory_mb", "cpu_percent", "flops_estimate"]
        for key in expected_keys:
            assert key in summary
            
        assert summary["latency_ms"] == 150.5
        assert summary["memory_mb"] == 45.2
        assert summary["cpu_percent"] == 60.0
        
        # Test with no performance metrics
        result_no_perf = EvaluationResult(query_id="q1", system_name="test")
        assert result_no_perf.get_performance_summary() == {}
        
    def test_from_baseline_result(self):
        """Test conversion from legacy BaselineResult."""
        # Create mock baseline result
        mock_baseline = Mock()
        mock_baseline.query_id = "q1"
        mock_baseline.baseline_name = "bm25"
        mock_baseline.query_text = "test query"
        mock_baseline.retrieved_docs = ["doc1", "doc2"]
        mock_baseline.relevance_scores = [0.9, 0.8]
        mock_baseline.ranks = [1, 2]
        mock_baseline.latency_ms = 100.0
        mock_baseline.memory_mb = 32.0
        mock_baseline.cpu_percent = 50.0
        mock_baseline.non_empty_validated = True
        mock_baseline.smoke_test_passed = True
        mock_baseline.candidate_count = 1000
        
        result = EvaluationResult.from_baseline_result(mock_baseline)
        
        assert result.query_id == "q1"
        assert result.system_name == "bm25"
        assert result.query_text == "test query"
        assert result.retrieved_docs == ["doc1", "doc2"]
        assert len(result.retrieval_results) == 2
        assert result.performance.latency_ms == 100.0
        assert result.validation_passed == True
        assert result.candidate_count == 1000
        
    def test_from_metrics_result(self):
        """Test conversion from legacy MetricsResult."""
        # Create mock metrics result
        mock_metrics = Mock()
        mock_metrics.query_id = "q2"
        mock_metrics.baseline_name = "dense"
        mock_metrics.latency_ms = 75.0
        mock_metrics.memory_mb = 28.0
        mock_metrics.ndcg_10 = 0.82
        mock_metrics.recall_10 = 0.85
        mock_metrics.precision_10 = 0.78
        mock_metrics.num_relevant = 5
        mock_metrics.num_retrieved = 10
        mock_metrics.query_length = 4
        
        result = EvaluationResult.from_metrics_result(mock_metrics)
        
        assert result.query_id == "q2"
        assert result.system_name == "dense"
        assert result.performance.latency_ms == 75.0
        assert result.ranking_metrics["ndcg_10"] == 0.82
        assert result.ranking_metrics["recall_10"] == 0.85
        assert result.ranking_metrics["precision_10"] == 0.78
        assert result.metadata["num_relevant"] == 5
        assert result.metadata["num_retrieved"] == 10


class TestAggregatedResults:
    """Test AggregatedResults functionality."""
    
    def create_sample_results(self):
        """Create sample EvaluationResult instances for testing."""
        results = []
        
        for i in range(3):
            performance = PerformanceMetrics(
                latency_ms=100.0 + i * 10,
                memory_mb=40.0 + i * 5,
                cpu_percent=50.0 + i * 5
            )
            
            ranking_metrics = {
                "ndcg_10": 0.7 + i * 0.05,
                "recall_10": 0.6 + i * 0.1
            }
            
            result = EvaluationResult(
                query_id=f"q{i+1}",
                system_name="test_system",
                performance=performance,
                ranking_metrics=ranking_metrics
            )
            results.append(result)
            
        return results
        
    def test_basic_creation(self):
        """Test basic AggregatedResults creation."""
        aggregated = AggregatedResults(system_name="test_system")
        
        assert aggregated.system_name == "test_system"
        assert len(aggregated.individual_results) == 0
        assert aggregated.query_count == 0
        assert len(aggregated.mean_metrics) == 0
        
    def test_add_result_and_aggregation(self):
        """Test adding results and automatic aggregation."""
        aggregated = AggregatedResults(system_name="test_system")
        results = self.create_sample_results()
        
        # Add results one by one
        for result in results:
            aggregated.add_result(result)
            
        assert aggregated.query_count == 3
        assert len(aggregated.individual_results) == 3
        
        # Check aggregated metrics
        assert "ndcg_10" in aggregated.mean_metrics
        assert "recall_10" in aggregated.mean_metrics
        
        # Verify calculations
        expected_mean_ndcg = statistics.mean([0.7, 0.75, 0.8])
        expected_mean_recall = statistics.mean([0.6, 0.7, 0.8])
        
        assert aggregated.mean_metrics["ndcg_10"] == pytest.approx(expected_mean_ndcg)
        assert aggregated.mean_metrics["recall_10"] == pytest.approx(expected_mean_recall)
        
    def test_statistical_calculations(self):
        """Test statistical calculations (mean, std, median)."""
        aggregated = AggregatedResults(system_name="test_system")
        results = self.create_sample_results()
        
        for result in results:
            aggregated.add_result(result)
            
        # Test standard deviation calculation
        ndcg_values = [0.7, 0.75, 0.8]
        expected_std = statistics.stdev(ndcg_values)
        assert aggregated.std_metrics["ndcg_10"] == pytest.approx(expected_std)
        
        # Test median calculation
        expected_median = statistics.median(ndcg_values)
        assert aggregated.median_metrics["ndcg_10"] == pytest.approx(expected_median)
        
        # Test with single result (std should be 0)
        single_aggregated = AggregatedResults(system_name="single")
        single_aggregated.add_result(results[0])
        assert single_aggregated.std_metrics["ndcg_10"] == 0.0
        
    def test_performance_aggregation(self):
        """Test performance metrics aggregation."""
        aggregated = AggregatedResults(system_name="test_system")
        results = self.create_sample_results()
        
        for result in results:
            aggregated.add_result(result)
            
        # Should have aggregated performance metrics
        assert aggregated.mean_performance is not None
        
        # Check aggregated values
        expected_mean_latency = statistics.mean([100.0, 110.0, 120.0])
        expected_mean_memory = statistics.mean([40.0, 45.0, 50.0])
        expected_mean_cpu = statistics.mean([50.0, 55.0, 60.0])
        
        assert aggregated.mean_performance.latency_ms == pytest.approx(expected_mean_latency)
        assert aggregated.mean_performance.memory_mb == pytest.approx(expected_mean_memory)
        assert aggregated.mean_performance.cpu_percent == pytest.approx(expected_mean_cpu)
        assert aggregated.mean_performance.measurements_count == 3
        
    def test_empty_aggregation(self):
        """Test aggregation with no results."""
        aggregated = AggregatedResults(system_name="empty")
        
        # Should handle empty case gracefully
        aggregated._recalculate_aggregates()
        
        assert len(aggregated.mean_metrics) == 0
        assert aggregated.mean_performance is None
        
    def test_results_with_missing_performance(self):
        """Test aggregation when some results have no performance metrics."""
        aggregated = AggregatedResults(system_name="test_system")
        
        # Create result without performance
        result_no_perf = EvaluationResult(
            query_id="q1",
            system_name="test_system",
            ranking_metrics={"ndcg_10": 0.8}
        )
        
        # Create result with performance
        result_with_perf = EvaluationResult(
            query_id="q2", 
            system_name="test_system",
            performance=PerformanceMetrics(latency_ms=100.0),
            ranking_metrics={"ndcg_10": 0.7}
        )
        
        aggregated.add_result(result_no_perf)
        aggregated.add_result(result_with_perf)
        
        # Should aggregate ranking metrics from both
        assert aggregated.mean_metrics["ndcg_10"] == pytest.approx(0.75)
        
        # Should only aggregate performance from results that have it
        assert aggregated.mean_performance is not None
        assert aggregated.mean_performance.latency_ms == 100.0
        assert aggregated.mean_performance.measurements_count == 1


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_performance_metrics_with_zero_values(self):
        """Test PerformanceMetrics with zero and negative values."""
        metrics = PerformanceMetrics(
            latency_ms=0.0,
            memory_mb=-5.0,  # Negative memory (unusual but should be handled)
            cpu_percent=0.0
        )
        
        assert metrics.latency_ms == 0.0
        assert metrics.memory_mb == -5.0
        assert metrics.memory_peak_mb == -5.0
        
    def test_query_info_with_special_characters(self):
        """Test QueryInfo with special characters and Unicode."""
        query = QueryInfo(
            query_id="q_special",
            text="What is AI? 中文查询 🤖",
            domain="unicode_test"
        )
        
        assert "中文查询" in query.text
        assert "🤖" in query.text
        assert query.query_length == 5  # "What is AI? 中文查询 🤖" splits to 5 tokens
        
    def test_document_info_with_large_embedding(self):
        """Test DocumentInfo with large numpy arrays."""
        large_embedding = np.random.randn(1000)  # Large embedding
        
        doc = DocumentInfo(
            doc_id="large_doc",
            content="test",
            embedding=large_embedding
        )
        
        assert doc.embedding.shape == (1000,)
        np.testing.assert_array_equal(doc.embedding, large_embedding)
        
    def test_evaluation_result_with_empty_lists(self):
        """Test EvaluationResult with empty lists."""
        result = EvaluationResult(
            query_id="empty_q",
            system_name="empty_system",
            retrieved_docs=[],
            retrieval_results=[]
        )
        
        assert len(result.retrieved_docs) == 0
        assert len(result.retrieval_results) == 0
        assert result.get_performance_summary() == {}
        
    def test_aggregated_results_with_inconsistent_metrics(self):
        """Test AggregatedResults when individual results have different metrics."""
        aggregated = AggregatedResults(system_name="inconsistent")
        
        # Result with only ndcg_10
        result1 = EvaluationResult(
            query_id="q1",
            system_name="inconsistent",
            ranking_metrics={"ndcg_10": 0.8}
        )
        
        # Result with only recall_10
        result2 = EvaluationResult(
            query_id="q2",
            system_name="inconsistent", 
            ranking_metrics={"recall_10": 0.7}
        )
        
        aggregated.add_result(result1)
        aggregated.add_result(result2)
        
        # Should handle missing metrics gracefully
        assert "ndcg_10" in aggregated.mean_metrics
        assert "recall_10" in aggregated.mean_metrics
        assert aggregated.mean_metrics["ndcg_10"] == 0.8  # Only one value
        assert aggregated.mean_metrics["recall_10"] == 0.7  # Only one value


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])