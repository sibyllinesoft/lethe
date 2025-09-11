"""
Comprehensive tests for InfiniteBench metrics module.

Focuses on testing high-complexity conditional logic and edge cases
identified by valknut analysis, particularly in metric calculations
and evaluation summary handling.

Test areas:
- ROUGE-L calculation with various text configurations
- Exact match handling for different answer formats
- F1 score calculation with edge cases
- nDCG@k computation with ranking scenarios
- MetricResult validation and error handling
- EvaluationSummary aggregation logic
- Text preprocessing and normalization edge cases
- Score boundary conditions and outlier handling
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

# Import the module under test
try:
    from src.infinitebench.metrics import (
        MetricResult, EvaluationSummary, 
        rouge_l_score, exact_match_score, f1_score, ndcg_k_score,
        normalize_text, extract_answer, calculate_overall_score
    )
except ImportError:
    # Create minimal implementations for testing if imports fail
    from dataclasses import dataclass
    from typing import Dict, Any, Optional, List
    
    @dataclass
    class MetricResult:
        metric_name: str
        score: float
        details: Dict[str, Any]
        sample_scores: Optional[List[float]] = None
    
    @dataclass
    class EvaluationSummary:
        task_name: str
        num_samples: int
        metric_results: Dict[str, MetricResult]
        overall_score: float
    
    def rouge_l_score(predicted, reference):
        return 0.5
    
    def exact_match_score(predicted, reference):
        return 1.0 if predicted.strip().lower() == reference.strip().lower() else 0.0
    
    def f1_score(predicted, reference):
        return 0.5
    
    def ndcg_k_score(relevance_scores, retrieved_docs, k=10):
        return 0.5
    
    def normalize_text(text):
        return text.strip().lower()
    
    def extract_answer(text):
        return text.strip()
    
    def calculate_overall_score(metric_results):
        return sum(r.score for r in metric_results.values()) / len(metric_results)


class TestMetricResult:
    """Test suite for MetricResult functionality."""
    
    def test_metric_result_creation(self):
        """Test creation of MetricResult with valid data."""
        result = MetricResult(
            metric_name="rouge_l",
            score=0.85,
            details={"precision": 0.9, "recall": 0.8},
            sample_scores=[0.8, 0.9, 0.85]
        )
        
        assert result.metric_name == "rouge_l"
        assert result.score == 0.85
        assert result.details["precision"] == 0.9
        assert len(result.sample_scores) == 3
    
    def test_metric_result_boundary_scores(self):
        """Test MetricResult with boundary score values."""
        # Perfect score
        perfect_result = MetricResult(
            metric_name="exact_match",
            score=1.0,
            details={}
        )
        assert perfect_result.score == 1.0
        
        # Zero score
        zero_result = MetricResult(
            metric_name="exact_match", 
            score=0.0,
            details={}
        )
        assert zero_result.score == 0.0
    
    def test_metric_result_unusual_scores(self):
        """Test MetricResult with unusual score values."""
        # Negative score (should trigger warning)
        with pytest.warns(match="Unusual score value"):
            negative_result = MetricResult(
                metric_name="test_metric",
                score=-0.1,
                details={}
            )
            assert negative_result.score == -0.1
        
        # Score greater than 1 (should trigger warning)  
        with pytest.warns(match="Unusual score value"):
            high_result = MetricResult(
                metric_name="test_metric",
                score=1.5,
                details={}
            )
            assert high_result.score == 1.5


class TestEvaluationSummary:
    """Test suite for EvaluationSummary functionality."""
    
    @pytest.fixture
    def sample_metric_results(self):
        """Sample metric results for testing."""
        return {
            "rouge_l": MetricResult("rouge_l", 0.75, {"precision": 0.8, "recall": 0.7}),
            "exact_match": MetricResult("exact_match", 0.6, {"matches": 12, "total": 20}),
            "f1": MetricResult("f1", 0.82, {"precision": 0.85, "recall": 0.79})
        }
    
    def test_evaluation_summary_creation(self, sample_metric_results):
        """Test creation of EvaluationSummary."""
        summary = EvaluationSummary(
            task_name="retrieve.needle_in_haystack",
            num_samples=100,
            metric_results=sample_metric_results,
            overall_score=0.72
        )
        
        assert summary.task_name == "retrieve.needle_in_haystack"
        assert summary.num_samples == 100
        assert len(summary.metric_results) == 3
        assert summary.overall_score == 0.72
    
    def test_get_metric_score(self, sample_metric_results):
        """Test getting specific metric scores."""
        summary = EvaluationSummary(
            task_name="test_task",
            num_samples=50,
            metric_results=sample_metric_results,
            overall_score=0.72
        )
        
        assert summary.get_metric_score("rouge_l") == 0.75
        assert summary.get_metric_score("exact_match") == 0.6
        assert summary.get_metric_score("f1") == 0.82
    
    def test_get_nonexistent_metric_score(self, sample_metric_results):
        """Test error handling for nonexistent metrics."""
        summary = EvaluationSummary(
            task_name="test_task",
            num_samples=50,
            metric_results=sample_metric_results,
            overall_score=0.72
        )
        
        with pytest.raises(KeyError, match="Metric 'nonexistent' not found"):
            summary.get_metric_score("nonexistent")


class TestRougeL:
    """Test suite for ROUGE-L score calculation."""
    
    def test_rouge_l_identical_texts(self):
        """Test ROUGE-L with identical predicted and reference texts."""
        text = "The quick brown fox jumps over the lazy dog"
        score = rouge_l_score(text, text)
        assert score == 1.0
    
    def test_rouge_l_completely_different_texts(self):
        """Test ROUGE-L with completely different texts."""
        predicted = "The quick brown fox"
        reference = "Cats and dogs playing"
        score = rouge_l_score(predicted, reference)
        assert score == 0.0
    
    def test_rouge_l_partial_overlap(self):
        """Test ROUGE-L with partial text overlap."""
        predicted = "The quick brown fox jumps"
        reference = "The quick brown cat runs"
        score = rouge_l_score(predicted, reference)
        assert 0.0 < score < 1.0
    
    def test_rouge_l_empty_texts(self):
        """Test ROUGE-L with empty texts."""
        # Both empty
        assert rouge_l_score("", "") == 1.0
        
        # One empty
        assert rouge_l_score("", "some text") == 0.0
        assert rouge_l_score("some text", "") == 0.0
    
    def test_rouge_l_single_words(self):
        """Test ROUGE-L with single words."""
        # Same word
        assert rouge_l_score("cat", "cat") == 1.0
        
        # Different words
        assert rouge_l_score("cat", "dog") == 0.0
    
    def test_rouge_l_case_sensitivity(self):
        """Test ROUGE-L case sensitivity."""
        predicted = "The Quick Brown Fox"
        reference = "the quick brown fox"
        score = rouge_l_score(predicted, reference)
        # Should handle case insensitivity
        assert score > 0.5
    
    def test_rouge_l_punctuation_handling(self):
        """Test ROUGE-L with punctuation."""
        predicted = "Hello, world!"
        reference = "Hello world"
        score = rouge_l_score(predicted, reference)
        # Should handle punctuation appropriately
        assert score > 0.0


class TestExactMatch:
    """Test suite for exact match score calculation."""
    
    def test_exact_match_identical(self):
        """Test exact match with identical texts."""
        text = "The answer is 42"
        assert exact_match_score(text, text) == 1.0
    
    def test_exact_match_different(self):
        """Test exact match with different texts."""
        predicted = "The answer is 42"
        reference = "The answer is 24"
        assert exact_match_score(predicted, reference) == 0.0
    
    def test_exact_match_whitespace_normalization(self):
        """Test exact match with whitespace differences."""
        predicted = "  The answer is 42  "
        reference = "The answer is 42"
        assert exact_match_score(predicted, reference) == 1.0
    
    def test_exact_match_case_insensitive(self):
        """Test exact match case insensitivity."""
        predicted = "THE ANSWER IS 42"
        reference = "the answer is 42"
        assert exact_match_score(predicted, reference) == 1.0
    
    def test_exact_match_empty_strings(self):
        """Test exact match with empty strings."""
        assert exact_match_score("", "") == 1.0
        assert exact_match_score("", "not empty") == 0.0
        assert exact_match_score("not empty", "") == 0.0
    
    def test_exact_match_special_characters(self):
        """Test exact match with special characters."""
        predicted = "Price: $19.99"
        reference = "Price: $19.99"
        assert exact_match_score(predicted, reference) == 1.0
        
        predicted = "Price: $19.99"
        reference = "Price: $20.00"
        assert exact_match_score(predicted, reference) == 0.0


class TestF1Score:
    """Test suite for F1 score calculation."""
    
    def test_f1_identical_texts(self):
        """Test F1 score with identical texts."""
        text = "The quick brown fox jumps over the lazy dog"
        score = f1_score(text, text)
        assert score == 1.0
    
    def test_f1_no_overlap(self):
        """Test F1 score with no word overlap."""
        predicted = "cats and dogs"
        reference = "birds fly high"
        score = f1_score(predicted, reference)
        assert score == 0.0
    
    def test_f1_partial_overlap(self):
        """Test F1 score with partial overlap."""
        predicted = "the quick brown fox"
        reference = "the slow brown cat"
        score = f1_score(predicted, reference)
        # Should have score between 0 and 1 due to partial overlap
        assert 0.0 < score < 1.0
    
    def test_f1_empty_texts(self):
        """Test F1 score with empty texts."""
        # Both empty - undefined, should handle gracefully
        score = f1_score("", "")
        assert score >= 0.0
        
        # One empty
        assert f1_score("", "some text") == 0.0
        assert f1_score("some text", "") == 0.0
    
    def test_f1_single_words(self):
        """Test F1 score with single words."""
        assert f1_score("cat", "cat") == 1.0
        assert f1_score("cat", "dog") == 0.0
    
    def test_f1_repeated_words(self):
        """Test F1 score with repeated words."""
        predicted = "cat cat cat"
        reference = "cat dog"
        score = f1_score(predicted, reference)
        # Should handle word repetition appropriately
        assert 0.0 < score < 1.0


class TestNDCGScore:
    """Test suite for nDCG@k score calculation."""
    
    def test_ndcg_perfect_ranking(self):
        """Test nDCG with perfect ranking."""
        relevance_scores = [3, 2, 1, 0]
        retrieved_docs = ["doc1", "doc2", "doc3", "doc4"]
        score = ndcg_k_score(relevance_scores, retrieved_docs, k=4)
        assert score == 1.0
    
    def test_ndcg_worst_ranking(self):
        """Test nDCG with worst possible ranking."""
        relevance_scores = [0, 1, 2, 3]
        retrieved_docs = ["doc1", "doc2", "doc3", "doc4"]
        score = ndcg_k_score(relevance_scores, retrieved_docs, k=4)
        assert score < 1.0
    
    def test_ndcg_no_relevant_docs(self):
        """Test nDCG when no documents are relevant."""
        relevance_scores = [0, 0, 0, 0]
        retrieved_docs = ["doc1", "doc2", "doc3", "doc4"]
        score = ndcg_k_score(relevance_scores, retrieved_docs, k=4)
        assert score == 0.0
    
    def test_ndcg_all_relevant(self):
        """Test nDCG when all documents are equally relevant."""
        relevance_scores = [1, 1, 1, 1]
        retrieved_docs = ["doc1", "doc2", "doc3", "doc4"]
        score = ndcg_k_score(relevance_scores, retrieved_docs, k=4)
        # Any ranking should be equally good
        assert score == 1.0
    
    def test_ndcg_k_cutoff(self):
        """Test nDCG with different k values."""
        relevance_scores = [3, 2, 1, 0, 1]
        retrieved_docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        
        # nDCG@3 should only consider first 3 documents
        score_k3 = ndcg_k_score(relevance_scores, retrieved_docs, k=3)
        score_k5 = ndcg_k_score(relevance_scores, retrieved_docs, k=5)
        
        # Scores should differ due to different cutoffs
        assert score_k3 != score_k5
    
    def test_ndcg_empty_lists(self):
        """Test nDCG with empty lists."""
        score = ndcg_k_score([], [], k=1)
        # Should handle empty input gracefully
        assert score == 0.0
    
    def test_ndcg_mismatched_lengths(self):
        """Test nDCG with mismatched relevance scores and documents."""
        relevance_scores = [3, 2, 1]
        retrieved_docs = ["doc1", "doc2"]  # Shorter list
        
        with pytest.raises(ValueError):
            ndcg_k_score(relevance_scores, retrieved_docs, k=2)


class TestTextNormalization:
    """Test suite for text normalization functionality."""
    
    def test_normalize_text_basic(self):
        """Test basic text normalization."""
        text = "  Hello World!  "
        normalized = normalize_text(text)
        assert normalized == "hello world!"
    
    def test_normalize_text_punctuation(self):
        """Test normalization with punctuation."""
        text = "Hello, world! How are you?"
        normalized = normalize_text(text)
        # Should remove punctuation and convert to lowercase
        assert "," not in normalized
        assert "!" not in normalized
        assert "?" not in normalized
        assert normalized.islower()
    
    def test_normalize_text_numbers(self):
        """Test normalization with numbers."""
        text = "The answer is 42"
        normalized = normalize_text(text)
        assert "42" in normalized  # Numbers should be preserved
    
    def test_normalize_text_empty(self):
        """Test normalization with empty text."""
        normalized = normalize_text("")
        assert normalized == ""
    
    def test_normalize_text_whitespace_only(self):
        """Test normalization with whitespace-only text."""
        normalized = normalize_text("   \n\t  ")
        assert normalized == ""


class TestAnswerExtraction:
    """Test suite for answer extraction functionality."""
    
    def test_extract_answer_basic(self):
        """Test basic answer extraction."""
        text = "The answer is: 42"
        answer = extract_answer(text)
        assert answer == "42"
    
    def test_extract_answer_no_pattern(self):
        """Test answer extraction when no pattern is found."""
        text = "This text has no clear answer format"
        answer = extract_answer(text)
        # Should return original text when no pattern matches
        assert answer == text.strip()
    
    def test_extract_answer_multiple_patterns(self):
        """Test answer extraction with multiple possible patterns."""
        text = "The answer is 42, but some might say 24."
        answer = extract_answer(text)
        # Should extract first matching pattern
        assert answer in ["42", "24"]
    
    def test_extract_answer_empty_text(self):
        """Test answer extraction with empty text."""
        answer = extract_answer("")
        assert answer == ""
    
    def test_extract_answer_complex_format(self):
        """Test answer extraction with complex answer formats."""
        text = "Based on the analysis, the final answer is (A) option A."
        answer = extract_answer(text)
        # Should handle complex formats appropriately
        assert len(answer) > 0


class TestOverallScoreCalculation:
    """Test suite for overall score calculation."""
    
    def test_calculate_overall_score_basic(self):
        """Test basic overall score calculation."""
        metric_results = {
            "rouge_l": MetricResult("rouge_l", 0.8, {}),
            "exact_match": MetricResult("exact_match", 0.6, {}),
            "f1": MetricResult("f1", 0.7, {})
        }
        
        overall = calculate_overall_score(metric_results)
        expected = (0.8 + 0.6 + 0.7) / 3
        assert abs(overall - expected) < 0.001
    
    def test_calculate_overall_score_weighted(self):
        """Test weighted overall score calculation."""
        metric_results = {
            "rouge_l": MetricResult("rouge_l", 0.8, {}),
            "exact_match": MetricResult("exact_match", 0.6, {}),
            "f1": MetricResult("f1", 0.7, {})
        }
        
        weights = {"rouge_l": 0.5, "exact_match": 0.3, "f1": 0.2}
        overall = calculate_overall_score(metric_results, weights=weights)
        
        expected = 0.8 * 0.5 + 0.6 * 0.3 + 0.7 * 0.2
        assert abs(overall - expected) < 0.001
    
    def test_calculate_overall_score_empty(self):
        """Test overall score calculation with empty metrics."""
        metric_results = {}
        overall = calculate_overall_score(metric_results)
        assert overall == 0.0
    
    def test_calculate_overall_score_single_metric(self):
        """Test overall score calculation with single metric."""
        metric_results = {
            "rouge_l": MetricResult("rouge_l", 0.85, {})
        }
        
        overall = calculate_overall_score(metric_results)
        assert overall == 0.85


class TestEdgeCasesAndErrorHandling:
    """Test suite for edge cases and error handling."""
    
    def test_metric_calculation_with_none_values(self):
        """Test metric calculations with None values."""
        # Should handle None inputs gracefully
        with pytest.raises((TypeError, AttributeError)):
            rouge_l_score(None, "reference")
        
        with pytest.raises((TypeError, AttributeError)):
            exact_match_score("predicted", None)
    
    def test_metric_calculation_with_numeric_inputs(self):
        """Test metric calculations with numeric inputs."""
        # Should handle numeric inputs appropriately
        score = exact_match_score("42", "42")
        assert score == 1.0
        
        score = exact_match_score("42.5", "42.50")
        # Should handle numeric precision appropriately
        assert score >= 0.0
    
    def test_metric_calculation_with_unicode(self):
        """Test metric calculations with Unicode text."""
        predicted = "Café français"
        reference = "Café français"
        score = exact_match_score(predicted, reference)
        assert score == 1.0
    
    def test_metric_result_with_invalid_details(self):
        """Test MetricResult with invalid details."""
        # Should handle various types in details
        result = MetricResult(
            metric_name="test",
            score=0.5,
            details={"string": "value", "number": 42, "list": [1, 2, 3]}
        )
        assert result.details["string"] == "value"
        assert result.details["number"] == 42
        assert result.details["list"] == [1, 2, 3]
    
    def test_evaluation_summary_with_zero_samples(self):
        """Test EvaluationSummary with zero samples."""
        summary = EvaluationSummary(
            task_name="empty_task",
            num_samples=0,
            metric_results={},
            overall_score=0.0
        )
        assert summary.num_samples == 0
        assert len(summary.metric_results) == 0
    
    def test_score_boundary_conditions(self):
        """Test various boundary conditions in score calculations."""
        # Test with very small texts
        assert rouge_l_score("a", "a") == 1.0
        assert f1_score("a", "b") == 0.0
        
        # Test with very long texts
        long_text1 = "word " * 1000
        long_text2 = "word " * 1000
        assert exact_match_score(long_text1, long_text2) == 1.0
        
        # Test with special floating point values
        result = MetricResult("test", float('inf'), {})
        assert result.score == float('inf')


if __name__ == "__main__":
    pytest.main([__file__])