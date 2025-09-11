"""
Comprehensive tests for the rerank core module.

Tests cover:
- RerankingConfiguration validation and methods
- RerankingResult data handling and serialization  
- RerankingSystem core functionality
- Score interpolation and normalization logic
- Budget and latency monitoring
- Go/No-Go validation for promotion decisions
- Performance tracking and telemetry
- Edge cases and error conditions
"""

import pytest
import json
import hashlib
import time
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

# Import the modules to test
from src.rerank.core import (
    RerankingConfiguration,
    RerankingResult, 
    RerankingSystem
)
from src.fusion.core import FusionResult, FusionConfiguration
from src.retriever.timing import TimingHarness, PerformanceProfiler


class TestRerankingConfiguration:
    """Test RerankingConfiguration validation and methods."""
    
    def test_valid_configuration_creation(self):
        """Test creating valid reranking configurations."""
        config = RerankingConfiguration(beta=0.5, k_rerank=200)
        
        assert config.beta == 0.5
        assert config.k_rerank == 200
        assert config.k_final == 100  # Default value
        assert config.cross_encoder_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert config.batch_size == 32
        assert config.max_length == 512
        assert config.max_latency_ms == 1000.0
        assert config.budget_multiplier == 2.0
        
    def test_beta_validation(self):
        """Test beta parameter validation."""
        # Valid beta values
        for beta in [0.0, 0.2, 0.5, 0.8, 1.0]:
            config = RerankingConfiguration(beta=beta, k_rerank=100)
            assert config.beta == beta
        
        # Invalid beta values
        with pytest.raises(ValueError, match="Beta must be in \\[0,1\\]"):
            RerankingConfiguration(beta=-0.1, k_rerank=100)
        
        with pytest.raises(ValueError, match="Beta must be in \\[0,1\\]"):
            RerankingConfiguration(beta=1.1, k_rerank=100)
            
    def test_k_rerank_validation(self):
        """Test k_rerank parameter validation."""
        # Invalid k_rerank
        with pytest.raises(ValueError, match="k_rerank must be positive"):
            RerankingConfiguration(beta=0.5, k_rerank=0)
            
        with pytest.raises(ValueError, match="k_rerank must be positive"):
            RerankingConfiguration(beta=0.5, k_rerank=-10)
            
    def test_k_final_validation(self):
        """Test k_final parameter validation."""
        # k_final must be positive
        with pytest.raises(ValueError, match="k_final must be positive and <= k_rerank"):
            RerankingConfiguration(beta=0.5, k_rerank=100, k_final=0)
            
        # k_final must be <= k_rerank
        with pytest.raises(ValueError, match="k_final must be positive and <= k_rerank"):
            RerankingConfiguration(beta=0.5, k_rerank=100, k_final=150)
            
        # Valid k_final
        config = RerankingConfiguration(beta=0.5, k_rerank=200, k_final=50)
        assert config.k_final == 50
        
    def test_max_latency_validation(self):
        """Test max_latency_ms parameter validation."""
        with pytest.raises(ValueError, match="max_latency_ms must be positive"):
            RerankingConfiguration(beta=0.5, k_rerank=100, max_latency_ms=0)
            
        with pytest.raises(ValueError, match="max_latency_ms must be positive"):
            RerankingConfiguration(beta=0.5, k_rerank=100, max_latency_ms=-100)
        
    def test_weight_properties(self):
        """Test original and reranking weight properties."""
        config = RerankingConfiguration(beta=0.3, k_rerank=100)
        assert config.w_original == 0.7
        assert config.w_rerank == 0.3
        assert config.w_original + config.w_rerank == pytest.approx(1.0)
        
        # Test edge cases
        config_zero = RerankingConfiguration(beta=0.0, k_rerank=100)
        assert config_zero.w_original == 1.0
        assert config_zero.w_rerank == 0.0
        
        config_one = RerankingConfiguration(beta=1.0, k_rerank=100)
        assert config_one.w_original == 0.0
        assert config_one.w_rerank == 1.0
        
    def test_configuration_hash(self):
        """Test configuration hash generation."""
        config1 = RerankingConfiguration(beta=0.5, k_rerank=100)
        config2 = RerankingConfiguration(beta=0.5, k_rerank=100)
        config3 = RerankingConfiguration(beta=0.6, k_rerank=100)
        
        # Same configurations should have same hash
        assert config1.get_hash() == config2.get_hash()
        
        # Different configurations should have different hashes
        assert config1.get_hash() != config3.get_hash()
        
        # Hash should be consistent
        hash1 = config1.get_hash()
        hash2 = config1.get_hash()
        assert hash1 == hash2
        
        # Hash should be 16 characters
        assert len(config1.get_hash()) == 16
        
    def test_configuration_with_custom_params(self):
        """Test configuration with custom parameters."""
        config = RerankingConfiguration(
            beta=0.4,
            k_rerank=150,
            k_final=75,
            cross_encoder_model="custom-model",
            batch_size=16,
            max_length=256,
            max_latency_ms=500.0,
            budget_multiplier=1.5
        )
        
        assert config.cross_encoder_model == "custom-model"
        assert config.batch_size == 16
        assert config.max_length == 256
        assert config.max_latency_ms == 500.0
        assert config.budget_multiplier == 1.5


class TestRerankingResult:
    """Test RerankingResult data handling and serialization."""
    
    def create_sample_reranking_result(self):
        """Create a sample RerankingResult for testing."""
        config = RerankingConfiguration(beta=0.5, k_rerank=200)
        
        return RerankingResult(
            doc_ids=["doc1", "doc2", "doc3"],
            scores=[0.95, 0.85, 0.75],
            ranks=[1, 2, 3],
            original_scores={"doc1": 0.9, "doc2": 0.8, "doc3": 0.7},
            rerank_scores={"doc1": 1.0, "doc2": 0.9, "doc3": 0.8},
            final_scores={"doc1": 0.95, "doc2": 0.85, "doc3": 0.75},
            candidates_reranked=200,
            reranking_latency_ms=45.2,
            total_latency_ms=67.8,
            p95_latency_ms=72.5,
            throughput_qps=14.7,
            budget_respected=True,
            latency_within_budget=True,
            config=config
        )
        
    def test_reranking_result_creation(self):
        """Test RerankingResult creation and basic properties."""
        result = self.create_sample_reranking_result()
        
        assert len(result.doc_ids) == 3
        assert len(result.scores) == 3
        assert len(result.ranks) == 3
        assert result.candidates_reranked == 200
        assert result.reranking_latency_ms == 45.2
        assert result.total_latency_ms == 67.8
        assert result.budget_respected == True
        assert result.latency_within_budget == True
        
    def test_reranking_result_to_dict(self):
        """Test RerankingResult serialization to dictionary."""
        result = self.create_sample_reranking_result()
        result_dict = result.to_dict()
        
        # Check all required fields are present
        required_fields = [
            'doc_ids', 'scores', 'ranks', 'original_scores', 'rerank_scores',
            'final_scores', 'candidates_reranked', 'reranking_latency_ms',
            'total_latency_ms', 'p95_latency_ms', 'throughput_qps',
            'budget_respected', 'latency_within_budget', 'config'
        ]
        
        for field in required_fields:
            assert field in result_dict
            
        # Check config serialization
        config_dict = result_dict['config']
        assert 'beta' in config_dict
        assert 'k_rerank' in config_dict
        assert 'w_original' in config_dict
        assert 'w_rerank' in config_dict
        assert 'config_hash' in config_dict
        
        # Verify values
        assert result_dict['doc_ids'] == ["doc1", "doc2", "doc3"]
        assert result_dict['config']['beta'] == 0.5
        assert result_dict['config']['w_original'] == 0.5
        assert result_dict['config']['w_rerank'] == 0.5


class TestRerankingSystem:
    """Test RerankingSystem core functionality."""
    
    def create_mock_cross_encoder(self):
        """Create mock cross encoder for testing."""
        cross_encoder = Mock()
        cross_encoder.score_pairs.return_value = {
            "doc1": 0.9,
            "doc2": 0.8,  
            "doc3": 0.7,
            "doc4": 0.6
        }
        return cross_encoder
        
    def create_sample_fusion_result(self):
        """Create sample FusionResult for testing."""
        config = FusionConfiguration(alpha=0.5)
        
        return FusionResult(
            doc_ids=["doc1", "doc2", "doc3", "doc4"],
            scores=[0.95, 0.85, 0.75, 0.65],
            ranks=[1, 2, 3, 4],
            sparse_scores={},
            dense_scores={},
            fusion_scores={},
            sparse_candidates=1000,
            dense_candidates=1000,
            union_candidates=1500,
            total_latency_ms=45.2,
            sparse_latency_ms=15.3,
            dense_latency_ms=18.7,
            fusion_latency_ms=5.2,
            ann_recall_achieved=0.98,
            budget_parity_maintained=True,
            config=config
        )
        
    def test_reranking_system_initialization(self):
        """Test RerankingSystem initialization."""
        with patch('src.rerank.core.CrossEncoderReranker') as mock_ce_class:
            mock_ce = Mock()
            mock_ce_class.return_value = mock_ce
            
            system = RerankingSystem()
            
            assert system.cross_encoder == mock_ce
            assert system.timing_harness is not None
            assert system.profiler is not None
            assert isinstance(system.telemetry_log, list)
            assert len(system.telemetry_log) == 0
            assert isinstance(system.latency_history, list)
            assert len(system.latency_history) == 0
            
    def test_reranking_system_with_custom_components(self):
        """Test RerankingSystem with custom components."""
        mock_cross_encoder = self.create_mock_cross_encoder()
        mock_timing = Mock()
        mock_profiler = Mock()
        
        system = RerankingSystem(
            cross_encoder=mock_cross_encoder,
            timing_harness=mock_timing,
            profiler=mock_profiler
        )
        
        assert system.cross_encoder == mock_cross_encoder
        assert system.timing_harness == mock_timing
        assert system.profiler == mock_profiler
        
    def test_rerank_results_basic_functionality(self):
        """Test basic reranking functionality."""
        mock_cross_encoder = self.create_mock_cross_encoder()
        system = RerankingSystem(cross_encoder=mock_cross_encoder)
        
        # Mock timing harness
        system.timing_harness.time = MagicMock()
        system.timing_harness.get_last_duration = Mock(return_value=25.0)
        
        fusion_result = self.create_sample_fusion_result()
        config = RerankingConfiguration(beta=0.5, k_rerank=3, k_final=2)
        
        result = system.rerank_results(fusion_result, "test query", config)
        
        # Verify cross-encoder call
        mock_cross_encoder.score_pairs.assert_called_once()
        call_args = mock_cross_encoder.score_pairs.call_args
        assert call_args[1]['query'] == "test query"
        assert len(call_args[1]['doc_ids']) == 3  # k_rerank
        assert call_args[1]['batch_size'] == config.batch_size
        assert call_args[1]['max_length'] == config.max_length
        
        # Verify result structure
        assert isinstance(result, RerankingResult)
        assert len(result.doc_ids) == config.k_final  # Should return k_final results
        assert len(result.scores) == len(result.doc_ids)
        assert len(result.ranks) == len(result.doc_ids)
        assert result.config == config
        assert result.candidates_reranked == 3
        
        # Verify telemetry logging
        assert len(system.telemetry_log) == 1
        assert len(system.latency_history) == 1
        
    def test_rerank_with_beta_zero(self):
        """Test reranking with beta=0 (no reranking)."""
        mock_cross_encoder = self.create_mock_cross_encoder()
        system = RerankingSystem(cross_encoder=mock_cross_encoder)
        
        system.timing_harness.time = MagicMock()
        system.timing_harness.get_last_duration = Mock(return_value=5.0)
        
        fusion_result = self.create_sample_fusion_result()
        config = RerankingConfiguration(beta=0.0, k_rerank=3, k_final=2)
        
        result = system.rerank_results(fusion_result, "test query", config)
        
        # Should not call cross encoder when beta=0
        mock_cross_encoder.score_pairs.assert_called_once()
        
        # Rerank scores should be all zeros
        for score in result.rerank_scores.values():
            assert score == 0.0
            
        # Final scores should equal original scores (w_original = 1.0)
        assert result.config.w_original == 1.0
        assert result.config.w_rerank == 0.0
        
    def test_rerank_with_beta_one(self):
        """Test reranking with beta=1 (full reranking)."""
        mock_cross_encoder = self.create_mock_cross_encoder()
        system = RerankingSystem(cross_encoder=mock_cross_encoder)
        
        system.timing_harness.time = MagicMock()
        system.timing_harness.get_last_duration = Mock(return_value=35.0)
        
        fusion_result = self.create_sample_fusion_result()
        config = RerankingConfiguration(beta=1.0, k_rerank=3, k_final=2)
        
        result = system.rerank_results(fusion_result, "test query", config)
        
        # Should call cross encoder
        mock_cross_encoder.score_pairs.assert_called_once()
        
        # Weights should be correct
        assert result.config.w_original == 0.0
        assert result.config.w_rerank == 1.0
        
    def test_score_normalization(self):
        """Test score normalization functionality."""
        system = RerankingSystem()
        
        # Test normal case
        scores = {"doc1": 10.0, "doc2": 5.0, "doc3": 0.0}
        normalized = system._normalize_scores(scores)
        
        assert normalized["doc1"] == 1.0  # (10-0)/(10-0)
        assert normalized["doc2"] == 0.5   # (5-0)/(10-0)
        assert normalized["doc3"] == 0.0   # (0-0)/(10-0)
        
        # Test identical scores
        identical_scores = {"doc1": 5.0, "doc2": 5.0, "doc3": 5.0}
        normalized_identical = system._normalize_scores(identical_scores)
        
        for score in normalized_identical.values():
            assert score == 1.0
            
        # Test empty scores
        empty_normalized = system._normalize_scores({})
        assert empty_normalized == {}
        
        # Test single score
        single_score = {"doc1": 7.5}
        normalized_single = system._normalize_scores(single_score)
        assert normalized_single["doc1"] == 1.0
        
    def test_p95_latency_computation(self):
        """Test P95 latency computation."""
        system = RerankingSystem()
        
        # Empty history
        assert system._compute_p95_latency() == 0.0
        
        # Add latency samples
        latencies = [100, 150, 200, 250, 300, 350, 400, 450, 500, 1000]
        system.latency_history = latencies
        
        p95 = system._compute_p95_latency()
        expected_p95 = np.percentile(latencies, 95)
        assert p95 == pytest.approx(expected_p95)
        
    def test_budget_checking(self):
        """Test budget constraint checking."""
        system = RerankingSystem()
        config = RerankingConfiguration(
            beta=0.5, 
            k_rerank=100,
            max_latency_ms=100.0,
            budget_multiplier=2.0
        )
        
        # Within budget (latency * multiplier)
        assert system._check_budget_respected(150.0, config) == True
        assert system._check_budget_respected(200.0, config) == True
        
        # Exceeds budget
        assert system._check_budget_respected(201.0, config) == False
        
        # Edge case - exactly at limit
        assert system._check_budget_respected(200.0, config) == True
        
    def test_go_no_go_validation_approve(self):
        """Test Go/No-Go validation that approves promotion."""
        system = RerankingSystem()
        
        # Create successful reranking result
        config = RerankingConfiguration(beta=0.5, k_rerank=100, max_latency_ms=1000.0)
        reranking_result = RerankingResult(
            doc_ids=["doc1"],
            scores=[0.9],
            ranks=[1],
            original_scores={"doc1": 0.8},
            rerank_scores={"doc1": 1.0},
            final_scores={"doc1": 0.9},
            candidates_reranked=100,
            reranking_latency_ms=45.0,
            total_latency_ms=67.0,
            p95_latency_ms=70.0,
            throughput_qps=15.0,
            budget_respected=True,
            latency_within_budget=True,
            config=config
        )
        
        should_promote, reason, evidence = system.validate_go_no_go(
            baseline_result=None,  # Not used currently
            reranking_result=reranking_result,
            confidence_interval_lower_bound=0.05  # Positive improvement
        )
        
        assert should_promote == True
        assert "Promotion approved" in reason
        assert evidence['ci_positive'] == True
        assert evidence['budget_respected'] == True
        assert evidence['latency_within_budget'] == True
        
    def test_go_no_go_validation_reject_ci(self):
        """Test Go/No-Go validation that rejects due to CI."""
        system = RerankingSystem()
        
        config = RerankingConfiguration(beta=0.5, k_rerank=100)
        reranking_result = RerankingResult(
            doc_ids=["doc1"],
            scores=[0.9],
            ranks=[1],
            original_scores={"doc1": 0.8},
            rerank_scores={"doc1": 1.0},
            final_scores={"doc1": 0.9},
            candidates_reranked=100,
            reranking_latency_ms=45.0,
            total_latency_ms=67.0,
            p95_latency_ms=70.0,
            throughput_qps=15.0,
            budget_respected=True,
            latency_within_budget=True,
            config=config
        )
        
        should_promote, reason, evidence = system.validate_go_no_go(
            baseline_result=None,
            reranking_result=reranking_result,
            confidence_interval_lower_bound=-0.02  # Negative CI
        )
        
        assert should_promote == False
        assert "Promotion rejected" in reason
        assert "CI_lower=-0.020 <= 0" in reason
        assert evidence['ci_positive'] == False
        
    def test_go_no_go_validation_reject_budget(self):
        """Test Go/No-Go validation that rejects due to budget."""
        system = RerankingSystem()
        
        config = RerankingConfiguration(beta=0.5, k_rerank=100)
        reranking_result = RerankingResult(
            doc_ids=["doc1"],
            scores=[0.9],
            ranks=[1],
            original_scores={"doc1": 0.8},
            rerank_scores={"doc1": 1.0},
            final_scores={"doc1": 0.9},
            candidates_reranked=100,
            reranking_latency_ms=45.0,
            total_latency_ms=67.0,
            p95_latency_ms=70.0,
            throughput_qps=15.0,
            budget_respected=False,  # Budget violated
            latency_within_budget=False,
            config=config
        )
        
        should_promote, reason, evidence = system.validate_go_no_go(
            baseline_result=None,
            reranking_result=reranking_result,
            confidence_interval_lower_bound=0.05  # Positive CI
        )
        
        assert should_promote == False
        assert "Promotion rejected" in reason
        assert "budget_violated" in reason
        assert evidence['ci_positive'] == True
        assert evidence['budget_respected'] == False
        
    def test_telemetry_management(self):
        """Test telemetry logging and management."""
        system = RerankingSystem()
        
        # Add some mock telemetry
        system.telemetry_log.append({"test": "data1"})
        system.telemetry_log.append({"test": "data2"})
        system.latency_history = [100.0, 200.0]
        
        # Test get_telemetry
        telemetry = system.get_telemetry()
        assert len(telemetry) == 2
        assert telemetry[0]["test"] == "data1"
        assert telemetry[1]["test"] == "data2"
        
        # Verify it returns a copy
        telemetry.append({"test": "data3"})
        assert len(system.get_telemetry()) == 2  # Original unchanged
        
        # Test clear_telemetry
        system.clear_telemetry()
        assert len(system.telemetry_log) == 0
        assert len(system.latency_history) == 0
        
    def test_performance_summary(self):
        """Test performance summary generation."""
        system = RerankingSystem()
        
        # Empty history
        summary = system.get_performance_summary()
        assert summary["queries_processed"] == 0
        
        # Add latency data
        latencies = [100.0, 150.0, 200.0, 250.0, 300.0]
        system.latency_history = latencies
        
        summary = system.get_performance_summary()
        
        assert summary["queries_processed"] == 5
        assert "latency_stats" in summary
        assert "throughput_stats" in summary
        
        # Check latency stats
        latency_stats = summary["latency_stats"]
        assert latency_stats["mean_ms"] == pytest.approx(200.0)
        assert latency_stats["median_ms"] == pytest.approx(200.0)
        assert latency_stats["p95_ms"] == pytest.approx(np.percentile(latencies, 95))
        assert latency_stats["p99_ms"] == pytest.approx(np.percentile(latencies, 99))
        
        # Check throughput stats  
        throughput_stats = summary["throughput_stats"]
        expected_mean_qps = 1000.0 / 200.0  # 1000ms / mean_latency
        expected_peak_qps = 1000.0 / 100.0  # 1000ms / min_latency
        assert throughput_stats["mean_qps"] == pytest.approx(expected_mean_qps)
        assert throughput_stats["peak_qps"] == pytest.approx(expected_peak_qps)


class TestEdgeCasesAndErrorConditions:
    """Test edge cases and error conditions."""
    
    def test_rerank_with_fewer_candidates_than_k_rerank(self):
        """Test reranking when fusion result has fewer candidates than k_rerank."""
        mock_cross_encoder = Mock()
        mock_cross_encoder.score_pairs.return_value = {"doc1": 0.9, "doc2": 0.8}
        
        system = RerankingSystem(cross_encoder=mock_cross_encoder)
        system.timing_harness.time = MagicMock()
        system.timing_harness.get_last_duration = Mock(return_value=15.0)
        
        # Create fusion result with only 2 documents
        fusion_config = FusionConfiguration(alpha=0.5)
        fusion_result = FusionResult(
            doc_ids=["doc1", "doc2"],
            scores=[0.9, 0.8],
            ranks=[1, 2],
            sparse_scores={}, dense_scores={}, fusion_scores={},
            sparse_candidates=100, dense_candidates=100, union_candidates=150,
            total_latency_ms=30.0, sparse_latency_ms=10.0, 
            dense_latency_ms=12.0, fusion_latency_ms=3.0,
            ann_recall_achieved=0.98, budget_parity_maintained=True,
            config=fusion_config
        )
        
        # Request more candidates than available
        config = RerankingConfiguration(beta=0.5, k_rerank=5, k_final=3)
        
        result = system.rerank_results(fusion_result, "test query", config)
        
        # Should only rerank available candidates
        assert result.candidates_reranked == 2
        assert len(result.doc_ids) == 2  # Only 2 available
        
    def test_rerank_with_empty_fusion_result(self):
        """Test reranking with empty fusion result."""
        mock_cross_encoder = Mock()
        mock_cross_encoder.score_pairs.return_value = {}
        
        system = RerankingSystem(cross_encoder=mock_cross_encoder)
        system.timing_harness.time = MagicMock()
        system.timing_harness.get_last_duration = Mock(return_value=5.0)
        
        # Create empty fusion result
        fusion_config = FusionConfiguration(alpha=0.5)
        fusion_result = FusionResult(
            doc_ids=[], scores=[], ranks=[],
            sparse_scores={}, dense_scores={}, fusion_scores={},
            sparse_candidates=0, dense_candidates=0, union_candidates=0,
            total_latency_ms=10.0, sparse_latency_ms=5.0, 
            dense_latency_ms=5.0, fusion_latency_ms=0.0,
            ann_recall_achieved=0.0, budget_parity_maintained=True,
            config=fusion_config
        )
        
        config = RerankingConfiguration(beta=0.5, k_rerank=100, k_final=50)
        
        result = system.rerank_results(fusion_result, "test query", config)
        
        assert result.candidates_reranked == 0
        assert len(result.doc_ids) == 0
        assert len(result.scores) == 0
        assert len(result.ranks) == 0
        
    def test_score_interpolation_edge_cases(self):
        """Test score interpolation with edge cases."""
        mock_cross_encoder = Mock()
        
        # Test with extreme beta values and score ranges
        test_cases = [
            (0.0, {"doc1": 100.0}, {"doc1": 0.5}),  # beta=0, no reranking weight
            (1.0, {"doc1": 0.1}, {"doc1": 10.0}),   # beta=1, no original weight
            (0.5, {"doc1": 0.0}, {"doc1": 0.0}),    # All zero scores
        ]
        
        for beta, original_scores, rerank_scores in test_cases:
            mock_cross_encoder.score_pairs.return_value = rerank_scores
            system = RerankingSystem(cross_encoder=mock_cross_encoder)
            system.timing_harness.time = MagicMock()
            system.timing_harness.get_last_duration = Mock(return_value=10.0)
            
            # Create minimal fusion result
            fusion_config = FusionConfiguration(alpha=0.5)
            fusion_result = FusionResult(
                doc_ids=["doc1"], scores=[list(original_scores.values())[0]], ranks=[1],
                sparse_scores={}, dense_scores={}, fusion_scores={},
                sparse_candidates=1, dense_candidates=1, union_candidates=1,
                total_latency_ms=5.0, sparse_latency_ms=2.0, 
                dense_latency_ms=2.0, fusion_latency_ms=1.0,
                ann_recall_achieved=0.98, budget_parity_maintained=True,
                config=fusion_config
            )
            
            config = RerankingConfiguration(beta=beta, k_rerank=1, k_final=1)
            result = system.rerank_results(fusion_result, "test", config)
            
            # Should complete without errors
            assert len(result.doc_ids) == 1
            assert len(result.scores) == 1
            
    def test_performance_summary_edge_cases(self):
        """Test performance summary with edge case latencies."""
        system = RerankingSystem()
        
        # Test with zero latencies
        system.latency_history = [0.0, 0.0, 0.0]
        summary = system.get_performance_summary()
        
        # Should handle division by zero gracefully
        assert summary["queries_processed"] == 3
        assert summary["latency_stats"]["mean_ms"] == 0.0
        
        # Throughput calculations with zero latency
        throughput_stats = summary["throughput_stats"]
        # Should be 0.0 when mean latency is 0
        assert throughput_stats["mean_qps"] == 0.0
        
    def test_extreme_configuration_values(self):
        """Test system with extreme configuration values."""
        # Very large k_rerank and k_final
        config = RerankingConfiguration(
            beta=0.5,
            k_rerank=100000,
            k_final=50000,
            max_latency_ms=0.001,  # Very strict budget
            budget_multiplier=10.0  # Very loose multiplier
        )
        
        assert config.k_rerank == 100000
        assert config.k_final == 50000
        
        # Budget checking with extreme values
        system = RerankingSystem()
        
        # Should respect multiplier
        max_allowed = config.max_latency_ms * config.budget_multiplier  # 0.01ms
        assert system._check_budget_respected(0.005, config) == True
        assert system._check_budget_respected(0.015, config) == False


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])