"""
Comprehensive tests for the fusion core module.

Tests cover:
- FusionConfiguration validation and methods
- FusionResult data handling and serialization
- HybridFusionSystem core functionality
- Score normalization and fusion logic
- Error conditions and edge cases
- Performance and timing functionality
"""

import pytest
import json
import hashlib
import time
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

# Import the modules to test
from src.fusion.core import (
    FusionConfiguration, 
    FusionResult, 
    HybridFusionSystem,
    create_fusion_system
)
from src.retriever.timing import TimingHarness, PerformanceProfiler


class TestFusionConfiguration:
    """Test FusionConfiguration validation and methods."""
    
    def test_valid_configuration_creation(self):
        """Test creating valid fusion configurations."""
        config = FusionConfiguration(alpha=0.5)
        assert config.alpha == 0.5
        assert config.k_init_sparse == 1000
        assert config.k_init_dense == 1000
        assert config.k_final == 100
        
    def test_alpha_validation(self):
        """Test alpha parameter validation."""
        # Valid alpha values
        for alpha in [0.0, 0.2, 0.5, 0.8, 1.0]:
            config = FusionConfiguration(alpha=alpha)
            assert config.alpha == alpha
        
        # Invalid alpha values
        with pytest.raises(ValueError, match="Alpha must be in \\[0,1\\]"):
            FusionConfiguration(alpha=-0.1)
        
        with pytest.raises(ValueError, match="Alpha must be in \\[0,1\\]"):
            FusionConfiguration(alpha=1.1)
            
    def test_k_init_validation(self):
        """Test k_init parameter validation."""
        # Invalid k_init_sparse
        with pytest.raises(ValueError, match="k_init parameters must be positive"):
            FusionConfiguration(alpha=0.5, k_init_sparse=0)
            
        with pytest.raises(ValueError, match="k_init parameters must be positive"):
            FusionConfiguration(alpha=0.5, k_init_sparse=-10)
            
        # Invalid k_init_dense
        with pytest.raises(ValueError, match="k_init parameters must be positive"):
            FusionConfiguration(alpha=0.5, k_init_dense=0)
    
    def test_k_final_validation(self):
        """Test k_final parameter validation."""
        # k_final must be positive
        with pytest.raises(ValueError, match="k_final must be positive"):
            FusionConfiguration(alpha=0.5, k_final=0)
            
        # k_final must be <= min(k_init_sparse, k_init_dense)
        with pytest.raises(ValueError, match="k_final must be positive and <= min"):
            FusionConfiguration(alpha=0.5, k_init_sparse=100, k_init_dense=200, k_final=150)
            
        # Valid k_final
        config = FusionConfiguration(alpha=0.5, k_init_sparse=100, k_init_dense=200, k_final=50)
        assert config.k_final == 50
        
    def test_weight_properties(self):
        """Test sparse and dense weight properties."""
        config = FusionConfiguration(alpha=0.3)
        assert config.w_sparse == 0.3
        assert config.w_dense == 0.7
        assert config.w_sparse + config.w_dense == pytest.approx(1.0)
        
        # Test edge cases
        config_zero = FusionConfiguration(alpha=0.0)
        assert config_zero.w_sparse == 0.0
        assert config_zero.w_dense == 1.0
        
        config_one = FusionConfiguration(alpha=1.0)
        assert config_one.w_sparse == 1.0
        assert config_one.w_dense == 0.0
        
    def test_configuration_hash(self):
        """Test configuration hash generation."""
        config1 = FusionConfiguration(alpha=0.5)
        config2 = FusionConfiguration(alpha=0.5)
        config3 = FusionConfiguration(alpha=0.6)
        
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
        """Test configuration with custom BM25 and ANN parameters."""
        bm25_params = {"k1": 1.5, "b": 0.75}
        ann_params = {"n_trees": 10, "metric": "cosine"}
        
        config = FusionConfiguration(
            alpha=0.4,
            bm25_params=bm25_params,
            ann_params=ann_params,
            target_ann_recall=0.95
        )
        
        assert config.bm25_params == bm25_params
        assert config.ann_params == ann_params
        assert config.target_ann_recall == 0.95


class TestFusionResult:
    """Test FusionResult data handling and serialization."""
    
    def create_sample_fusion_result(self):
        """Create a sample FusionResult for testing."""
        config = FusionConfiguration(alpha=0.5)
        
        return FusionResult(
            doc_ids=["doc1", "doc2", "doc3"],
            scores=[0.95, 0.85, 0.75],
            ranks=[1, 2, 3],
            sparse_scores={"doc1": 0.9, "doc2": 0.7, "doc3": 0.6},
            dense_scores={"doc1": 0.8, "doc2": 0.9, "doc3": 0.8},
            fusion_scores={"doc1": 0.85, "doc2": 0.8, "doc3": 0.7},
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
        
    def test_fusion_result_creation(self):
        """Test FusionResult creation and basic properties."""
        result = self.create_sample_fusion_result()
        
        assert len(result.doc_ids) == 3
        assert len(result.scores) == 3
        assert len(result.ranks) == 3
        assert result.sparse_candidates == 1000
        assert result.dense_candidates == 1000
        assert result.union_candidates == 1500
        
    def test_fusion_result_to_dict(self):
        """Test FusionResult serialization to dictionary."""
        result = self.create_sample_fusion_result()
        result_dict = result.to_dict()
        
        # Check all required fields are present
        required_fields = [
            'doc_ids', 'scores', 'ranks', 'sparse_scores', 'dense_scores',
            'fusion_scores', 'sparse_candidates', 'dense_candidates', 
            'union_candidates', 'total_latency_ms', 'sparse_latency_ms',
            'dense_latency_ms', 'fusion_latency_ms', 'ann_recall_achieved',
            'budget_parity_maintained', 'config'
        ]
        
        for field in required_fields:
            assert field in result_dict
            
        # Check config serialization
        config_dict = result_dict['config']
        assert 'alpha' in config_dict
        assert 'k_init_sparse' in config_dict
        assert 'w_sparse' in config_dict
        assert 'w_dense' in config_dict
        assert 'config_hash' in config_dict
        
        # Verify values
        assert result_dict['doc_ids'] == ["doc1", "doc2", "doc3"]
        assert result_dict['config']['alpha'] == 0.5
        assert result_dict['config']['w_sparse'] == 0.5
        assert result_dict['config']['w_dense'] == 0.5


class TestHybridFusionSystem:
    """Test HybridFusionSystem core functionality."""
    
    def create_mock_retrievers(self):
        """Create mock retrievers for testing."""
        # Mock RetrievalResult class
        @dataclass
        class MockRetrievalResult:
            doc_id: str
            score: float
            
        sparse_retriever = Mock()
        dense_retriever = Mock()
        
        # Configure mock sparse retriever
        sparse_results = [
            MockRetrievalResult("doc1", 2.5),
            MockRetrievalResult("doc2", 2.0),
            MockRetrievalResult("doc3", 1.5),
            MockRetrievalResult("doc4", 1.0)
        ]
        sparse_retriever.retrieve.return_value = sparse_results
        
        # Configure mock dense retriever
        dense_results = [
            MockRetrievalResult("doc2", 0.95),
            MockRetrievalResult("doc3", 0.90),
            MockRetrievalResult("doc5", 0.85),
            MockRetrievalResult("doc1", 0.80)
        ]
        dense_retriever.retrieve.return_value = dense_results
        
        return sparse_retriever, dense_retriever
        
    def test_hybrid_fusion_system_initialization(self):
        """Test HybridFusionSystem initialization."""
        system = HybridFusionSystem()
        
        assert system.sparse_retriever is None
        assert system.dense_retriever is None
        assert system.timing_harness is not None
        assert system.profiler is not None
        assert isinstance(system.telemetry_log, list)
        assert len(system.telemetry_log) == 0
        
    def test_set_retrievers(self):
        """Test setting retrievers after initialization."""
        system = HybridFusionSystem()
        sparse_retriever, dense_retriever = self.create_mock_retrievers()
        
        system.set_retrievers(sparse_retriever, dense_retriever)
        
        assert system.sparse_retriever == sparse_retriever
        assert system.dense_retriever == dense_retriever
        
    def test_fuse_query_without_retrievers(self):
        """Test fusion query fails without retrievers."""
        system = HybridFusionSystem()
        config = FusionConfiguration(alpha=0.5)
        
        with pytest.raises(ValueError, match="Retrievers must be set before fusion"):
            system.fuse_query("test query", config)
            
    def test_fuse_query_basic_functionality(self):
        """Test basic fusion query functionality."""
        system = HybridFusionSystem()
        sparse_retriever, dense_retriever = self.create_mock_retrievers()
        system.set_retrievers(sparse_retriever, dense_retriever)
        
        config = FusionConfiguration(alpha=0.6, k_final=3)
        
        # Mock the timing harness context manager
        mock_context = MagicMock()
        mock_context.__enter__ = Mock()
        mock_context.__exit__ = Mock(return_value=None)
        system.timing_harness.time = Mock(return_value=mock_context)
        system.timing_harness.get_last_duration = Mock(return_value=10.0)
        
        with patch.object(system, '_validate_fusion_invariants'):
            result = system.fuse_query("test query", config, validate_invariants=False)
        
        # Verify retriever calls
        sparse_retriever.retrieve.assert_called_once_with("test query", k=1000)
        dense_retriever.retrieve.assert_called_once_with("test query", k=1000)
        
        # Verify result structure
        assert isinstance(result, FusionResult)
        assert len(result.doc_ids) <= config.k_final
        assert len(result.scores) == len(result.doc_ids)
        assert len(result.ranks) == len(result.doc_ids)
        assert result.config == config
        
        # Verify telemetry logging
        assert len(system.telemetry_log) == 1
        
    def test_score_normalization(self):
        """Test score normalization functionality."""
        system = HybridFusionSystem()
        
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
        
    def test_budget_parity_check(self):
        """Test budget parity checking."""
        system = HybridFusionSystem()
        
        # Within 5% - should pass
        assert system._check_budget_parity(100.0, 105.0) == True
        assert system._check_budget_parity(105.0, 100.0) == True
        
        # Exactly 5% - should pass
        assert system._check_budget_parity(100.0, 105.0) == True
        
        # Beyond 5% - should fail
        assert system._check_budget_parity(100.0, 110.0) == False
        assert system._check_budget_parity(110.0, 100.0) == False
        
        # Zero latency cases
        assert system._check_budget_parity(0.0, 100.0) == False
        assert system._check_budget_parity(100.0, 0.0) == False
        
    def test_ann_recall_computation(self):
        """Test ANN recall computation (placeholder implementation)."""
        system = HybridFusionSystem()
        
        # Currently returns placeholder value
        recall = system._compute_ann_recall([], 0.98)
        assert recall == 0.98  # Placeholder value
        
    def test_telemetry_management(self):
        """Test telemetry logging and management."""
        system = HybridFusionSystem()
        
        # Add some mock telemetry
        system.telemetry_log.append({"test": "data1"})
        system.telemetry_log.append({"test": "data2"})
        
        # Test get_telemetry
        telemetry = system.get_telemetry()
        assert len(telemetry) == 2
        assert telemetry[0]["test"] == "data1"
        assert telemetry[1]["test"] == "data2"
        
        # Verify it returns a copy
        telemetry.append({"test": "data3"})
        assert len(system.telemetry_log) == 2  # Original unchanged
        
        # Test clear_telemetry
        system.clear_telemetry()
        assert len(system.telemetry_log) == 0
        
    def test_fusion_scoring_logic(self):
        """Test the core fusion scoring logic."""
        system = HybridFusionSystem()
        sparse_retriever, dense_retriever = self.create_mock_retrievers()
        system.set_retrievers(sparse_retriever, dense_retriever)
        
        # Use different alpha values to test scoring
        configs = [
            FusionConfiguration(alpha=0.0),  # Pure dense
            FusionConfiguration(alpha=1.0),  # Pure sparse
            FusionConfiguration(alpha=0.5),  # Balanced
        ]
        
        for config in configs:
            # Mock the timing harness context manager
            mock_context = MagicMock()
            mock_context.__enter__ = Mock()
            mock_context.__exit__ = Mock(return_value=None)
            system.timing_harness.time = Mock(return_value=mock_context)
            system.timing_harness.get_last_duration = Mock(return_value=12.0)
            
            with patch.object(system, '_validate_fusion_invariants'):
                result = system.fuse_query("test query", config, validate_invariants=False)
            
            # Verify fusion scores are computed correctly
            assert len(result.fusion_scores) > 0
            
            # All fusion scores should be non-negative
            for score in result.fusion_scores.values():
                assert score >= 0.0
                
            # Top results should have highest scores
            if len(result.doc_ids) > 1:
                for i in range(len(result.scores) - 1):
                    assert result.scores[i] >= result.scores[i + 1]


class TestEdgeCasesAndErrorConditions:
    """Test edge cases and error conditions."""
    
    def test_fusion_with_no_candidates(self):
        """Test fusion when retrievers return no results."""
        system = HybridFusionSystem()
        
        # Create empty result retrievers
        sparse_retriever = Mock()
        dense_retriever = Mock()
        sparse_retriever.retrieve.return_value = []
        dense_retriever.retrieve.return_value = []
        
        system.set_retrievers(sparse_retriever, dense_retriever)
        config = FusionConfiguration(alpha=0.5)
        
        # Mock the timing harness context manager
        mock_context = MagicMock()
        mock_context.__enter__ = Mock()
        mock_context.__exit__ = Mock(return_value=None)
        system.timing_harness.time = Mock(return_value=mock_context)
        system.timing_harness.get_last_duration = Mock(return_value=5.0)
        
        with patch.object(system, '_validate_fusion_invariants'):
            result = system.fuse_query("test query", config, validate_invariants=False)
        
        assert len(result.doc_ids) == 0
        assert len(result.scores) == 0
        assert result.union_candidates == 0
        
    def test_fusion_with_large_k_final(self):
        """Test fusion when k_final exceeds available candidates."""
        system = HybridFusionSystem()
        
        @dataclass
        class MockRetrievalResult:
            doc_id: str
            score: float
        
        # Create retrievers with limited results
        sparse_retriever = Mock()
        dense_retriever = Mock()
        
        sparse_results = [MockRetrievalResult("doc1", 1.0)]
        dense_results = [MockRetrievalResult("doc2", 1.0)]
        
        sparse_retriever.retrieve.return_value = sparse_results
        dense_retriever.retrieve.return_value = dense_results
        
        system.set_retrievers(sparse_retriever, dense_retriever)
        
        # Request more results than available
        config = FusionConfiguration(alpha=0.5, k_final=10)
        
        # Mock the timing harness context manager
        mock_context = MagicMock()
        mock_context.__enter__ = Mock()
        mock_context.__exit__ = Mock(return_value=None)
        system.timing_harness.time = Mock(return_value=mock_context)
        system.timing_harness.get_last_duration = Mock(return_value=8.0)
        
        with patch.object(system, '_validate_fusion_invariants'):
            result = system.fuse_query("test query", config, validate_invariants=False)
        
        # Should return only available candidates
        assert len(result.doc_ids) == 2  # doc1 and doc2
        assert result.union_candidates == 2


class TestCreateFusionSystem:
    """Test the create_fusion_system factory function."""
    
    @patch('src.fusion.core.create_bm25_retriever')
    @patch('src.fusion.core.create_ann_retriever')
    def test_create_fusion_system_basic(self, mock_ann_creator, mock_bm25_creator):
        """Test basic fusion system creation."""
        # Setup mocks
        mock_sparse = Mock()
        mock_dense = Mock()
        mock_bm25_creator.return_value = mock_sparse
        mock_ann_creator.return_value = mock_dense
        
        # Create system
        system = create_fusion_system("/path/to/corpus")
        
        # Verify factory calls
        mock_bm25_creator.assert_called_once_with(corpus_path="/path/to/corpus")
        mock_ann_creator.assert_called_once_with(
            corpus_path="/path/to/corpus",
            embeddings_path=None
        )
        
        # Verify system setup
        assert isinstance(system, HybridFusionSystem)
        assert system.sparse_retriever == mock_sparse
        assert system.dense_retriever == mock_dense
        
    @patch('src.fusion.core.create_bm25_retriever')
    @patch('src.fusion.core.create_ann_retriever')
    def test_create_fusion_system_with_params(self, mock_ann_creator, mock_bm25_creator):
        """Test fusion system creation with custom parameters."""
        mock_sparse = Mock()
        mock_dense = Mock()
        mock_bm25_creator.return_value = mock_sparse
        mock_ann_creator.return_value = mock_dense
        
        bm25_params = {"k1": 1.5, "b": 0.75}
        ann_params = {"n_trees": 20, "metric": "cosine"}
        
        system = create_fusion_system(
            corpus_path="/path/to/corpus",
            embeddings_path="/path/to/embeddings",
            bm25_params=bm25_params,
            ann_params=ann_params
        )
        
        # Verify factory calls with parameters
        mock_bm25_creator.assert_called_once_with(
            corpus_path="/path/to/corpus",
            k1=1.5,
            b=0.75
        )
        mock_ann_creator.assert_called_once_with(
            corpus_path="/path/to/corpus",
            embeddings_path="/path/to/embeddings",
            n_trees=20,
            metric="cosine"
        )


class TestPerformanceAndTiming:
    """Test performance measurement and timing functionality."""
    
    def test_timing_integration(self):
        """Test timing harness integration in fusion system."""
        system = HybridFusionSystem()
        
        # Create mock timing harness with real-like behavior
        timing_harness = Mock()
        timing_harness.time.return_value.__enter__ = Mock()
        timing_harness.time.return_value.__exit__ = Mock()
        timing_harness.get_last_duration.return_value = 15.5
        
        system.timing_harness = timing_harness
        
        sparse_retriever = Mock()
        dense_retriever = Mock()
        sparse_retriever.retrieve.return_value = []
        dense_retriever.retrieve.return_value = []
        
        system.set_retrievers(sparse_retriever, dense_retriever)
        config = FusionConfiguration(alpha=0.5)
        
        with patch.object(system, '_validate_fusion_invariants'):
            result = system.fuse_query("test query", config, validate_invariants=False)
        
        # Verify timing calls were made
        expected_timing_calls = [
            "sparse_retrieval",
            "dense_retrieval", 
            "score_normalization",
            "fusion_scoring"
        ]
        
        for call_name in expected_timing_calls:
            timing_harness.time.assert_any_call(call_name)
            
        # Verify latency measurements in result
        assert result.sparse_latency_ms == 15.5
        assert result.dense_latency_ms == 15.5
        assert result.fusion_latency_ms == 15.5
        assert result.total_latency_ms > 0


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])