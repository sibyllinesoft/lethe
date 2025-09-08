#!/usr/bin/env python3
"""
Comprehensive Test Suite for Milestone 3: Hybrid Retrieval + Rerank + Diversification

Tests all components and integration scenarios including:
- Adaptive planning policy decisions
- Entity-based diversification with session-IDF
- Exact identifier matching guarantees  
- Multi-entity query coverage
- Performance targets (sub-200ms)
- Optional cross-encoder reranking
"""

import unittest
import time
from typing import List, Dict, Tuple, Set
import tempfile
import json
from unittest.mock import Mock, MagicMock, patch

# Import components to test
from .planning import (
    AdaptivePlanningEngine, PlanningConfiguration, PlanningStrategy, QueryFeatures
)
from .diversification import (
    EntityDiversificationEngine, DiversificationConfig, Entity
)
from .reranker import (
    LightweightCrossEncoderReranker, RerankingConfig
)
from .hybrid_retrieval import (
    EnhancedHybridRetrievalSystem, HybridRetrievalConfig
)

class TestAdaptivePlanningEngine(unittest.TestCase):
    """Test adaptive planning policy with VERIFY/EXPLORE/EXPLOIT strategies."""
    
    def setUp(self):
        """Set up test planning engine."""
        self.config = PlanningConfiguration(
            tau_verify_idf=8.0,
            tau_entity_overlap=0.3,
            tau_novelty=0.1,
            alpha_verify=0.7,
            alpha_explore=0.3,
            alpha_exploit=0.5
        )
        self.engine = AdaptivePlanningEngine(self.config)
        
        # Add some session context
        self.engine.update_session_context(
            "test_session",
            entities={"function_name", "class_name", "error_code"},
            tools={"grep", "find", "git"}
        )
    
    def test_verify_strategy_selection(self):
        """Test VERIFY strategy for high-precision queries."""
        # High IDF + entity overlap + identifiers -> VERIFY
        term_idfs = {"function_name": 10.0, "call": 5.0, "error": 3.0}
        
        result = self.engine.plan_retrieval(
            query="function_name() call error in class_name",
            session_id="test_session", 
            turn_idx=5,
            term_idfs=term_idfs
        )
        
        self.assertEqual(result.strategy, PlanningStrategy.VERIFY)
        self.assertEqual(result.alpha, 0.7)  # BM25-heavy
        self.assertEqual(result.ef_search, 50)  # Lower recall, faster
        self.assertGreater(result.confidence, 0.5)
        self.assertIn("High precision needed", result.reasoning)
    
    def test_explore_strategy_selection(self):
        """Test EXPLORE strategy for novel queries."""
        # Low entity overlap + no tool overlap -> EXPLORE
        result = self.engine.plan_retrieval(
            query="new concept never seen before",
            session_id="new_session",  # No history
            turn_idx=1,
            term_idfs={"new": 2.0, "concept": 3.0}
        )
        
        self.assertEqual(result.strategy, PlanningStrategy.EXPLORE)
        self.assertEqual(result.alpha, 0.3)  # Vector-heavy
        self.assertEqual(result.ef_search, 200)  # Higher recall
        self.assertIn("Novel query needs exploration", result.reasoning)
    
    def test_exploit_strategy_selection(self):
        """Test EXPLOIT strategy for balanced queries."""
        # Moderate conditions -> EXPLOIT
        term_idfs = {"some": 4.0, "function": 6.0}
        
        result = self.engine.plan_retrieval(
            query="some function call with error",
            session_id="test_session",
            turn_idx=3,
            term_idfs=term_idfs
        )
        
        self.assertEqual(result.strategy, PlanningStrategy.EXPLOIT)
        self.assertEqual(result.alpha, 0.5)  # Balanced
        self.assertEqual(result.ef_search, 100)
        self.assertIn("Balanced approach", result.reasoning)
    
    def test_feature_extraction(self):
        """Test comprehensive feature extraction."""
        term_idfs = {"function_name": 8.0, "error": 5.0}
        
        features = self.engine.extract_features(
            query="function_name() threw TypeError in class_name.method",
            session_id="test_session",
            turn_idx=5,
            term_idfs=term_idfs
        )
        
        self.assertEqual(features.max_idf, 8.0)
        self.assertTrue(features.has_code)  # function_name()
        self.assertTrue(features.has_error)  # TypeError
        self.assertTrue(features.has_identifier)  # function_name, class_name
        self.assertTrue(features.tool_overlap)  # Should detect some overlap
        self.assertGreater(features.entity_overlap_jaccard, 0.0)


class TestEntityDiversificationEngine(unittest.TestCase):
    """Test entity-based diversification with session-IDF weighting."""
    
    def setUp(self):
        """Set up test diversification engine."""
        self.config = DiversificationConfig(
            max_tokens=1000,
            max_docs=10
        )
        self.engine = EntityDiversificationEngine(self.config)
        
        # Set up test documents and entities
        self._setup_test_data()
    
    def _setup_test_data(self):
        """Create test documents with entities."""
        test_entities = [
            ("function_a", "id", 5.0),
            ("class_b", "id", 4.0), 
            ("error_x", "error", 8.0),
            ("file_y", "file", 3.0)
        ]
        
        # Doc 1: function_a + class_b
        self.engine.update_entity_database(
            "doc1", 
            "def function_a(): class_b.method()", 
            [("function_a", "id", 5.0), ("class_b", "id", 4.0)],
            20
        )
        
        # Doc 2: error_x + function_a
        self.engine.update_entity_database(
            "doc2",
            "function_a threw error_x",
            [("function_a", "id", 5.0), ("error_x", "error", 8.0)],
            15
        )
        
        # Doc 3: file_y + class_b
        self.engine.update_entity_database(
            "doc3",
            "class_b defined in file_y",
            [("class_b", "id", 4.0), ("file_y", "file", 3.0)],
            25
        )
        
        # Doc 4: only error_x (high weight)
        self.engine.update_entity_database(
            "doc4",
            "Critical error_x occurred",
            [("error_x", "error", 8.0)],
            10
        )
    
    def test_exact_identifier_matching(self):
        """Test exact identifier matches are guaranteed inclusion."""
        query = "function_a() error occurred"  # Should match function_a
        candidates = ["doc1", "doc2", "doc3", "doc4"]
        scores = [0.8, 0.9, 0.6, 0.7]
        
        result = self.engine.diversify_selection(query, candidates, scores)
        
        # Should include docs with function_a (doc1, doc2) as exact matches
        self.assertGreater(result.exact_matches_count, 0)
        self.assertIn("doc1", result.doc_ids)  # Contains function_a
        self.assertIn("doc2", result.doc_ids)  # Contains function_a
    
    def test_entity_coverage_maximization(self):
        """Test entity coverage objective function."""
        query = "show me diverse examples"  # No exact matches
        candidates = ["doc1", "doc2", "doc3", "doc4"] 
        scores = [0.7, 0.8, 0.6, 0.9]
        
        result = self.engine.diversify_selection(query, candidates, scores)
        
        # Should maximize entity coverage
        self.assertGreater(result.objective_value, 0.0)
        self.assertGreater(len(result.entity_coverage), 1)  # Multiple entities covered
        
        # High-weight entities should be preferred
        self.assertIn("error_x", result.entity_coverage)  # Highest weight entity
    
    def test_budget_enforcement(self):
        """Test token and document budget enforcement."""
        query = "diverse query"
        candidates = ["doc1", "doc2", "doc3", "doc4"]
        scores = [0.8, 0.7, 0.6, 0.5]
        
        result = self.engine.diversify_selection(query, candidates, scores, enforce_budget=True)
        
        # Should respect budget constraints
        self.assertLessEqual(result.tokens_used, self.config.max_tokens)
        self.assertLessEqual(len(result.doc_ids), self.config.max_docs)
    
    def test_diminishing_returns(self):
        """Test per-entity contribution capping."""
        # Add multiple docs with same entity
        for i in range(5, 10):
            self.engine.update_entity_database(
                f"doc{i}",
                "Another error_x occurrence",
                [("error_x", "error", 8.0)],
                10
            )
        
        query = "error_x examples"
        candidates = [f"doc{i}" for i in range(1, 10)]
        scores = [0.8] * len(candidates)
        
        result = self.engine.diversify_selection(query, candidates, scores)
        
        # Should not select all docs with error_x due to diminishing returns
        error_x_docs = sum(1 for doc_id in result.doc_ids 
                          if "error_x" in self.engine.doc_entities.get(doc_id, set()))
        self.assertLess(error_x_docs, len([d for d in candidates if "error_x" in self.engine.doc_entities.get(d, set())]))
    
    def test_multi_entity_query_coverage(self):
        """Test coverage of multiple entities in complex queries."""
        query = "function_a in class_b throws error_x from file_y"
        candidates = ["doc1", "doc2", "doc3", "doc4"]
        scores = [0.8, 0.7, 0.6, 0.5]
        
        result = self.engine.diversify_selection(query, candidates, scores)
        
        # Should cover multiple distinct entities
        distinct_entities = set()
        for doc_id in result.doc_ids:
            distinct_entities.update(self.engine.doc_entities.get(doc_id, set()))
        
        self.assertGreaterEqual(len(distinct_entities), 3)  # At least 3 different entities
        
        # Verify specific entity coverage
        expected_entities = {"function_a", "class_b", "error_x", "file_y"}
        covered_entities = distinct_entities & expected_entities
        self.assertGreaterEqual(len(covered_entities), 2)  # Cover at least 2 target entities


class TestLightweightCrossEncoderReranker(unittest.TestCase):
    """Test optional cross-encoder reranking system."""
    
    def setUp(self):
        """Set up test reranker."""
        self.config = RerankingConfig(
            enabled=True,
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            silent_fallback=True,
            top_k_rerank=10,
            batch_size=4
        )
        self.reranker = LightweightCrossEncoderReranker(self.config)
    
    def test_reranking_disabled_by_default(self):
        """Test that reranking is OFF by default."""
        default_config = RerankingConfig()
        default_reranker = LightweightCrossEncoderReranker(default_config)
        
        self.assertFalse(default_config.enabled)
        self.assertFalse(default_reranker.is_available())
    
    @patch('src.reranker.SENTENCE_TRANSFORMERS_AVAILABLE', False)
    def test_graceful_fallback_no_model(self):
        """Test graceful fallback when model unavailable."""
        query = "test query"
        doc_ids = ["doc1", "doc2", "doc3"]
        doc_texts = ["text1", "text2", "text3"]
        scores = [0.8, 0.7, 0.6]
        
        result = self.reranker.rerank(query, doc_ids, doc_texts, scores)
        
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.doc_ids, doc_ids)  # Unchanged order
        self.assertEqual(result.final_scores, scores)  # Original scores
        self.assertEqual(result.candidates_reranked, 0)
    
    def test_top_k_limitation(self):
        """Test that only top-K candidates are reranked for efficiency."""
        query = "test query"
        doc_ids = [f"doc{i}" for i in range(20)]  # More than top_k_rerank
        doc_texts = [f"text {i}" for i in range(20)]
        scores = [0.9 - i*0.01 for i in range(20)]  # Decreasing scores
        
        # Mock the model to avoid actual inference
        with patch.object(self.reranker, 'model_loaded', True), \
             patch.object(self.reranker, 'model') as mock_model:
            
            mock_model.predict.return_value = [0.95 - i*0.05 for i in range(self.config.top_k_rerank)]
            
            result = self.reranker.rerank(query, doc_ids, doc_texts, scores)
            
            # Should only rerank top-K
            self.assertEqual(result.candidates_reranked, self.config.top_k_rerank)
            self.assertFalse(result.fallback_used)
    
    def test_batch_processing(self):
        """Test efficient batch processing."""
        query = "test query"
        doc_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        doc_texts = ["text1", "text2", "text3", "text4", "text5"]
        scores = [0.8, 0.7, 0.6, 0.5, 0.4]
        
        with patch.object(self.reranker, 'model_loaded', True), \
             patch.object(self.reranker, 'model') as mock_model:
            
            mock_model.predict.return_value = [0.9, 0.8, 0.85, 0.6, 0.7]
            
            result = self.reranker.rerank(query, doc_ids, doc_texts, scores)
            
            # Should process in batches (batch_size=4, so 2 batches for 5 docs)
            expected_batches = (len(doc_ids) + self.config.batch_size - 1) // self.config.batch_size
            self.assertEqual(result.batch_count, expected_batches)


class TestIntegratedHybridRetrievalSystem(unittest.TestCase):
    """Test complete integrated system with all Milestone 3 components."""
    
    def setUp(self):
        """Set up integrated test system."""
        # Mock retrievers
        self.mock_sparse_retriever = Mock()
        self.mock_dense_retriever = Mock()
        
        # Configure system
        self.config = HybridRetrievalConfig(
            target_latency_ms=200.0,
            enable_reranking=False,  # Start disabled per requirements
            enable_diversification=True,
            enable_exact_matching=True
        )
        
        self.system = EnhancedHybridRetrievalSystem(
            config=self.config,
            sparse_retriever=self.mock_sparse_retriever,
            dense_retriever=self.mock_dense_retriever
        )
    
    def test_end_to_end_retrieval_pipeline(self):
        """Test complete end-to-end retrieval pipeline."""
        # Setup mock retriever responses
        mock_sparse_results = [
            Mock(doc_id="doc1", score=0.8),
            Mock(doc_id="doc2", score=0.7),
            Mock(doc_id="doc3", score=0.6)
        ]
        mock_dense_results = [
            Mock(doc_id="doc2", score=0.9),
            Mock(doc_id="doc3", score=0.8),
            Mock(doc_id="doc4", score=0.7)
        ]
        
        self.mock_sparse_retriever.retrieve.return_value = mock_sparse_results
        self.mock_dense_retriever.retrieve.return_value = mock_dense_results
        
        # Test data
        query = "function_call() error in class_method"
        session_id = "test_session"
        session_entities = [
            ("function_call", "id", 6.0),
            ("class_method", "id", 5.0),
            ("error", "error", 7.0)
        ]
        doc_texts = {
            "doc1": "def function_call(): pass",
            "doc2": "class_method threw error", 
            "doc3": "another function_call example",
            "doc4": "unrelated content"
        }
        term_idfs = {"function_call": 8.0, "error": 6.0, "class_method": 5.0}
        
        # Execute retrieval
        result = self.system.retrieve(
            query=query,
            session_id=session_id,
            turn_idx=3,
            session_entities=session_entities,
            doc_texts=doc_texts,
            term_idfs=term_idfs
        )
        
        # Verify pipeline execution
        self.assertIsNotNone(result.planning_result)
        self.assertIsNotNone(result.diversification_result)
        self.assertIsNone(result.reranking_result)  # Disabled by default
        
        # Verify results
        self.assertGreater(len(result.doc_ids), 0)
        self.assertEqual(len(result.doc_ids), len(result.scores))
        
        # Verify exact matches
        self.assertGreater(result.exact_matches_included, 0)
        
        # Verify entity diversity
        self.assertGreater(result.entity_diversity_score, 0.0)
    
    def test_performance_target_monitoring(self):
        """Test sub-200ms latency monitoring."""
        # Setup fast mock responses
        self.mock_sparse_retriever.retrieve.return_value = [Mock(doc_id="doc1", score=0.8)]
        self.mock_dense_retriever.retrieve.return_value = [Mock(doc_id="doc1", score=0.9)]
        
        result = self.system.retrieve(
            query="fast query",
            session_id="perf_test",
            turn_idx=1
        )
        
        # Should meet latency target
        self.assertLess(result.total_latency_ms, self.config.target_latency_ms)
        
        # Should have stage timings
        self.assertIn("planning", result.stage_latencies_ms)
        self.assertIn("fusion", result.stage_latencies_ms)
        self.assertIn("diversification", result.stage_latencies_ms)
    
    def test_adaptive_alpha_weighting(self):
        """Test that planning policy affects fusion α values."""
        # Setup mocks
        self.mock_sparse_retriever.retrieve.return_value = [Mock(doc_id="doc1", score=0.8)]
        self.mock_dense_retriever.retrieve.return_value = [Mock(doc_id="doc1", score=0.9)]
        
        # High-precision query should use VERIFY strategy (α=0.7)
        result = self.system.retrieve(
            query="specific_function_name() exact_error_code",
            session_id="test_session", 
            turn_idx=5,
            term_idfs={"specific_function_name": 12.0, "exact_error_code": 10.0}
        )
        
        # Should use VERIFY strategy with high α (BM25-heavy)
        self.assertEqual(result.planning_result.strategy, PlanningStrategy.VERIFY)
        self.assertEqual(result.planning_result.alpha, 0.7)
    
    def test_graceful_degradation(self):
        """Test graceful degradation when components fail."""
        # Force an error in the main pipeline
        self.mock_sparse_retriever.retrieve.side_effect = Exception("Retriever failed")
        
        # Should fall back gracefully
        result = self.system.retrieve(
            query="test query with failure",
            session_id="test_session",
            turn_idx=1
        )
        
        # Should still return a result (even if empty)
        self.assertIsInstance(result, type(self.system.retrieve.__annotations__['return']))
    
    def test_multi_entity_query_comprehensive(self):
        """Test comprehensive multi-entity query handling."""
        # Setup diverse mock results
        mock_sparse_results = [Mock(doc_id=f"doc{i}", score=0.9-i*0.1) for i in range(8)]
        mock_dense_results = [Mock(doc_id=f"doc{i}", score=0.8-i*0.1) for i in range(2, 10)]
        
        self.mock_sparse_retriever.retrieve.return_value = mock_sparse_results
        self.mock_dense_retriever.retrieve.return_value = mock_dense_results
        
        # Complex multi-entity query
        query = "authenticate_user() in UserManager throws AuthError from auth_service.py when validate_token() fails"
        
        session_entities = [
            ("authenticate_user", "id", 8.0),
            ("UserManager", "id", 7.0),
            ("AuthError", "error", 9.0),
            ("auth_service.py", "file", 5.0),
            ("validate_token", "id", 6.0)
        ]
        
        doc_texts = {f"doc{i}": f"Document {i} contains some entities" for i in range(10)}
        
        result = self.system.retrieve(
            query=query,
            session_id="multi_entity_session",
            turn_idx=7,
            session_entities=session_entities,
            doc_texts=doc_texts,
            term_idfs={"authenticate_user": 8.0, "AuthError": 9.0, "validate_token": 6.0}
        )
        
        # Should handle multiple entities effectively
        self.assertGreater(len(result.diversification_result.entity_coverage), 0)
        self.assertGreater(result.entity_diversity_score, 0.0)
        
        # Should include exact matches for identifiers
        self.assertGreater(result.exact_matches_included, 0)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for Milestone 3 system."""
    
    def setUp(self):
        """Set up performance test system."""
        self.system = self._create_mock_system()
    
    def _create_mock_system(self):
        """Create system with fast mock components."""
        mock_sparse = Mock()
        mock_dense = Mock()
        
        # Fast mock responses
        mock_sparse.retrieve.return_value = [Mock(doc_id=f"doc{i}", score=0.9-i*0.1) for i in range(50)]
        mock_dense.retrieve.return_value = [Mock(doc_id=f"doc{i}", score=0.8-i*0.1) for i in range(50)]
        
        config = HybridRetrievalConfig(target_latency_ms=200.0)
        return EnhancedHybridRetrievalSystem(config, mock_sparse, mock_dense)
    
    def test_latency_target_achievement(self):
        """Test that system achieves sub-200ms latency target."""
        queries = [
            "simple query",
            "function_name() error handling",
            "complex multi entity query with class_name.method() and error_type",
            "exact identifier match for specific_function_call()",
            "exploratory query about new concepts and ideas"
        ]
        
        latencies = []
        
        for query in queries:
            start_time = time.time()
            
            result = self.system.retrieve(
                query=query,
                session_id="perf_test",
                turn_idx=1
            )
            
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
            
            # Individual query should meet target
            self.assertLess(latency, 200.0, f"Query '{query}' exceeded latency target: {latency:.1f}ms")
        
        # Average latency should be well under target
        avg_latency = sum(latencies) / len(latencies)
        self.assertLess(avg_latency, 150.0, f"Average latency too high: {avg_latency:.1f}ms")
        
        print(f"Performance Results:")
        print(f"  Average latency: {avg_latency:.1f}ms")
        print(f"  Max latency: {max(latencies):.1f}ms")
        print(f"  Min latency: {min(latencies):.1f}ms")


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)