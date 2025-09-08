#!/usr/bin/env python3
"""
Milestone 3 Example Usage and Integration Guide

Demonstrates how to use the complete hybrid retrieval system with
adaptive planning, entity diversification, and optional reranking.

Example scenarios:
1. Basic setup and configuration
2. Agent conversation retrieval
3. Multi-entity query handling
4. Performance monitoring and optimization
5. Custom configuration and tuning
"""

import logging
import time
from typing import List, Dict, Tuple
from pathlib import Path

# Import Milestone 3 components
from .config import SystemConfig, PerformanceProfile, create_default_config
from .hybrid_retrieval import EnhancedHybridRetrievalSystem, HybridRetrievalConfig
from .planning import PlanningStrategy
from .diversification import DiversificationResult

# Mock retrievers for example (replace with real implementations)
class MockBM25Retriever:
    """Mock BM25 retriever for examples."""
    
    def retrieve(self, query: str, k: int = 100):
        """Mock retrieval results."""
        from collections import namedtuple
        Result = namedtuple('Result', ['doc_id', 'score'])
        
        # Simulate relevant results based on query
        if "function_name" in query.lower():
            return [
                Result("doc_func_1", 0.95),
                Result("doc_func_2", 0.87),
                Result("doc_code_1", 0.76)
            ]
        elif "error" in query.lower():
            return [
                Result("doc_error_1", 0.92),
                Result("doc_debug_1", 0.84),
                Result("doc_trace_1", 0.71)
            ]
        else:
            return [
                Result(f"doc_{i}", 0.8 - i*0.1) for i in range(min(k, 10))
            ]

class MockANNRetriever:
    """Mock vector retriever for examples."""
    
    def retrieve(self, query: str, k: int = 100):
        """Mock vector similarity results."""
        from collections import namedtuple
        Result = namedtuple('Result', ['doc_id', 'score'])
        
        # Different relevance patterns than BM25
        if "concept" in query.lower() or "idea" in query.lower():
            return [
                Result("doc_concept_1", 0.89),
                Result("doc_theory_1", 0.82),
                Result("doc_abstract_1", 0.74)
            ]
        else:
            return [
                Result(f"doc_{i+5}", 0.85 - i*0.08) for i in range(min(k, 10))
            ]


class Milestone3ExampleRunner:
    """Complete example runner demonstrating all Milestone 3 features."""
    
    def __init__(self):
        """Initialize example runner."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create mock retrievers
        self.sparse_retriever = MockBM25Retriever()
        self.dense_retriever = MockANNRetriever()
        
        # Sample document database
        self.doc_texts = {
            "doc_func_1": "def function_name(args): implementation here",
            "doc_func_2": "function_name called in class_name.method()",
            "doc_code_1": "class_name contains function_name implementation",
            "doc_error_1": "function_name threw TypeError: invalid argument",
            "doc_debug_1": "Debug trace for function_name error_code_123",
            "doc_trace_1": "Stack trace shows function_name -> class_name.method",
            "doc_concept_1": "Conceptual overview of the system architecture",
            "doc_theory_1": "Theoretical foundations and abstract principles",
            "doc_abstract_1": "High-level design patterns and methodologies",
        }
        
        # Sample session entities (would come from entity extraction)
        self.session_entities = [
            ("function_name", "id", 8.0),
            ("class_name", "id", 6.0),
            ("TypeError", "error", 9.0),
            ("error_code_123", "error", 7.0),
            ("method", "id", 5.0),
            ("system", "concept", 4.0),
            ("architecture", "concept", 5.0)
        ]
    
    def example_1_basic_setup(self):
        """Example 1: Basic system setup with default configuration."""
        self.logger.info("=== Example 1: Basic Setup ===")
        
        # Create default balanced configuration
        config = create_default_config()
        
        # Initialize system
        system = EnhancedHybridRetrievalSystem(
            config=config.hybrid_retrieval,
            sparse_retriever=self.sparse_retriever,
            dense_retriever=self.dense_retriever
        )
        
        # Simple query
        query = "function_name implementation"
        result = system.retrieve(
            query=query,
            session_id="example_session",
            turn_idx=1,
            session_entities=self.session_entities,
            doc_texts=self.doc_texts
        )
        
        # Display results
        self.logger.info(f"Query: {query}")
        self.logger.info(f"Strategy: {result.planning_result.strategy.value}")
        self.logger.info(f"Alpha: {result.planning_result.alpha}")
        self.logger.info(f"Retrieved {len(result.doc_ids)} documents")
        self.logger.info(f"Exact matches: {result.exact_matches_included}")
        self.logger.info(f"Entity diversity: {result.entity_diversity_score:.2f}")
        self.logger.info(f"Total latency: {result.total_latency_ms:.1f}ms")
        
        return result
    
    def example_2_performance_profiles(self):
        """Example 2: Different performance profiles."""
        self.logger.info("=== Example 2: Performance Profiles ===")
        
        queries = [
            "function_name error handling",
            "explore new architecture concepts",
            "class_name.method() TypeError debugging"
        ]
        
        profiles = [PerformanceProfile.FAST, PerformanceProfile.BALANCED, PerformanceProfile.QUALITY]
        
        results = {}
        
        for profile in profiles:
            self.logger.info(f"\n--- Testing {profile.value.upper()} profile ---")
            
            config = SystemConfig(profile=profile)
            system = EnhancedHybridRetrievalSystem(
                config=config.hybrid_retrieval,
                sparse_retriever=self.sparse_retriever,
                dense_retriever=self.dense_retriever
            )
            
            profile_results = []
            
            for query in queries:
                result = system.retrieve(
                    query=query,
                    session_id=f"{profile.value}_session",
                    turn_idx=1,
                    session_entities=self.session_entities,
                    doc_texts=self.doc_texts
                )
                
                profile_results.append({
                    'query': query,
                    'strategy': result.planning_result.strategy.value,
                    'latency_ms': result.total_latency_ms,
                    'doc_count': len(result.doc_ids),
                    'exact_matches': result.exact_matches_included,
                    'diversity_score': result.entity_diversity_score
                })
                
                self.logger.info(
                    f"  {query[:30]}... -> {result.planning_result.strategy.value} "
                    f"({result.total_latency_ms:.1f}ms, {len(result.doc_ids)} docs)"
                )
            
            results[profile.value] = profile_results
        
        # Compare performance
        self.logger.info("\n--- Performance Comparison ---")
        for profile_name, profile_results in results.items():
            avg_latency = sum(r['latency_ms'] for r in profile_results) / len(profile_results)
            avg_diversity = sum(r['diversity_score'] for r in profile_results) / len(profile_results)
            self.logger.info(
                f"{profile_name.upper()}: avg_latency={avg_latency:.1f}ms, "
                f"avg_diversity={avg_diversity:.2f}"
            )
        
        return results
    
    def example_3_multi_entity_queries(self):
        """Example 3: Complex multi-entity queries."""
        self.logger.info("=== Example 3: Multi-Entity Queries ===")
        
        # Complex queries covering multiple entities
        complex_queries = [
            "function_name() in class_name throws TypeError with error_code_123",
            "debug system architecture for function_name method calls", 
            "TypeError handling in class_name.method() implementation",
            "trace error_code_123 through function_name call stack"
        ]
        
        config = create_default_config()
        system = EnhancedHybridRetrievalSystem(
            config=config.hybrid_retrieval,
            sparse_retriever=self.sparse_retriever,
            dense_retriever=self.dense_retriever
        )
        
        for i, query in enumerate(complex_queries):
            self.logger.info(f"\n--- Query {i+1}: {query} ---")
            
            result = system.retrieve(
                query=query,
                session_id="multi_entity_session",
                turn_idx=i+1,
                session_entities=self.session_entities,
                doc_texts=self.doc_texts,
                term_idfs={
                    "function_name": 8.0,
                    "class_name": 6.0, 
                    "TypeError": 9.0,
                    "error_code_123": 7.0,
                    "method": 5.0,
                    "system": 4.0,
                    "architecture": 5.0
                }
            )
            
            self.logger.info(f"Strategy: {result.planning_result.strategy.value}")
            self.logger.info(f"Planning confidence: {result.planning_result.confidence:.2f}")
            self.logger.info(f"Exact matches guaranteed: {result.exact_matches_included}")
            self.logger.info(f"Entity diversity score: {result.entity_diversity_score:.2f}")
            self.logger.info(f"Total latency: {result.total_latency_ms:.1f}ms")
            
            # Show entity coverage
            entity_coverage = result.diversification_result.entity_coverage
            if entity_coverage:
                self.logger.info("Entity coverage:")
                for entity, count in sorted(entity_coverage.items(), key=lambda x: x[1], reverse=True):
                    self.logger.info(f"  {entity}: {count} documents")
            
            # Show stage timings
            self.logger.info("Stage timings:")
            for stage, timing in result.stage_latencies_ms.items():
                self.logger.info(f"  {stage}: {timing:.1f}ms")
    
    def example_4_adaptive_strategies(self):
        """Example 4: Demonstrating adaptive strategy selection."""
        self.logger.info("=== Example 4: Adaptive Strategy Selection ===")
        
        # Queries designed to trigger different strategies
        strategy_queries = [
            # VERIFY: High IDF + entity overlap + identifiers
            ("specific_function_name() exact_error_code", "VERIFY", 
             {"specific_function_name": 12.0, "exact_error_code": 10.0}),
            
            # EXPLORE: Novel concepts + no entity overlap
            ("new distributed system paradigm concepts", "EXPLORE",
             {"new": 2.0, "distributed": 3.0, "paradigm": 2.5}),
            
            # EXPLOIT: Moderate conditions
            ("function error handling best practices", "EXPLOIT",
             {"function": 6.0, "error": 5.0, "handling": 4.0})
        ]
        
        config = create_default_config() 
        system = EnhancedHybridRetrievalSystem(
            config=config.hybrid_retrieval,
            sparse_retriever=self.sparse_retriever,
            dense_retriever=self.dense_retriever
        )
        
        for query, expected_strategy, term_idfs in strategy_queries:
            self.logger.info(f"\n--- Testing {expected_strategy} query ---")
            self.logger.info(f"Query: {query}")
            
            result = system.retrieve(
                query=query,
                session_id="adaptive_session",
                turn_idx=1,
                session_entities=self.session_entities,
                doc_texts=self.doc_texts,
                term_idfs=term_idfs
            )
            
            actual_strategy = result.planning_result.strategy.value.upper()
            self.logger.info(f"Expected: {expected_strategy}, Actual: {actual_strategy}")
            self.logger.info(f"Alpha (BM25 weight): {result.planning_result.alpha}")
            self.logger.info(f"EF Search: {result.planning_result.ef_search}")
            self.logger.info(f"Reasoning: {result.planning_result.reasoning}")
            
            # Verify strategy selection
            if actual_strategy == expected_strategy:
                self.logger.info("✓ Correct strategy selected")
            else:
                self.logger.warning("⚠ Unexpected strategy selected")
    
    def example_5_performance_monitoring(self):
        """Example 5: Performance monitoring and optimization."""
        self.logger.info("=== Example 5: Performance Monitoring ===")
        
        config = create_default_config()
        config.hybrid_retrieval.enable_performance_monitoring = True
        
        system = EnhancedHybridRetrievalSystem(
            config=config.hybrid_retrieval,
            sparse_retriever=self.sparse_retriever,
            dense_retriever=self.dense_retriever
        )
        
        # Run multiple queries to gather performance data
        test_queries = [
            "function_name implementation details",
            "class_name.method() error handling", 
            "system architecture overview",
            "TypeError debugging strategies",
            "error_code_123 resolution steps"
        ]
        
        latencies = []
        
        for i, query in enumerate(test_queries):
            start_time = time.time()
            
            result = system.retrieve(
                query=query,
                session_id="performance_session",
                turn_idx=i+1,
                session_entities=self.session_entities,
                doc_texts=self.doc_texts
            )
            
            latencies.append(result.total_latency_ms)
            
            self.logger.info(
                f"Query {i+1}: {result.total_latency_ms:.1f}ms "
                f"({result.planning_result.strategy.value})"
            )
        
        # Performance statistics
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)
        
        self.logger.info(f"\n--- Performance Summary ---")
        self.logger.info(f"Average latency: {avg_latency:.1f}ms")
        self.logger.info(f"Max latency: {max_latency:.1f}ms")
        self.logger.info(f"Min latency: {min_latency:.1f}ms")
        self.logger.info(f"Target: {config.hybrid_retrieval.target_latency_ms:.1f}ms")
        
        if avg_latency < config.hybrid_retrieval.target_latency_ms:
            self.logger.info("✓ Meeting performance targets")
        else:
            self.logger.warning("⚠ Exceeding performance targets")
        
        # System performance stats
        perf_stats = system.get_performance_stats()
        self.logger.info(f"\nSystem Statistics:")
        self.logger.info(f"  Total queries: {perf_stats['total_queries']}")
        self.logger.info(f"  Average latency: {perf_stats['avg_latency_ms']:.1f}ms")
        self.logger.info(f"  Exact match rate: {perf_stats['exact_match_rate']:.1%}")
        self.logger.info(f"  Diversification rate: {perf_stats['diversification_rate']:.1%}")
    
    def example_6_custom_configuration(self):
        """Example 6: Custom configuration and tuning."""
        self.logger.info("=== Example 6: Custom Configuration ===")
        
        # Create custom configuration
        from .planning import PlanningConfiguration
        from .diversification import DiversificationConfig
        from .reranker import RerankingConfig
        
        custom_config = HybridRetrievalConfig(
            target_latency_ms=150.0,  # Tighter target
            enable_reranking=False,   # Keep disabled per requirements
            enable_diversification=True,
            planning_config=PlanningConfiguration(
                tau_verify_idf=7.0,    # Custom threshold
                alpha_verify=0.8,      # More BM25-heavy for VERIFY
                alpha_explore=0.2,     # More vector-heavy for EXPLORE
                history_window=15      # Custom context window
            ),
            diversification_config=DiversificationConfig(
                max_tokens=6000,       # Custom token budget
                max_docs=75,           # Custom document limit
                max_entity_contribution=0.8  # Lower diminishing returns cap
            )
        )
        
        system = EnhancedHybridRetrievalSystem(
            config=custom_config,
            sparse_retriever=self.sparse_retriever,
            dense_retriever=self.dense_retriever
        )
        
        # Test with custom configuration
        query = "function_name() TypeError in class_name.method()"
        
        result = system.retrieve(
            query=query,
            session_id="custom_session", 
            turn_idx=1,
            session_entities=self.session_entities,
            doc_texts=self.doc_texts,
            term_idfs={"function_name": 9.0, "TypeError": 8.0, "method": 6.0}
        )
        
        self.logger.info(f"Custom configuration results:")
        self.logger.info(f"  Query: {query}")
        self.logger.info(f"  Strategy: {result.planning_result.strategy.value}")
        self.logger.info(f"  Alpha: {result.planning_result.alpha} (custom value)")
        self.logger.info(f"  Latency: {result.total_latency_ms:.1f}ms (target: {custom_config.target_latency_ms}ms)")
        self.logger.info(f"  Documents: {len(result.doc_ids)} (max: {custom_config.diversification_config.max_docs})")
        self.logger.info(f"  Tokens: {result.final_token_count} (max: {custom_config.diversification_config.max_tokens})")
    
    def run_all_examples(self):
        """Run all examples in sequence."""
        self.logger.info("🚀 Starting Milestone 3 Hybrid Retrieval System Examples")
        self.logger.info("=" * 60)
        
        try:
            self.example_1_basic_setup()
            self.example_2_performance_profiles()
            self.example_3_multi_entity_queries()
            self.example_4_adaptive_strategies()
            self.example_5_performance_monitoring()
            self.example_6_custom_configuration()
            
            self.logger.info("\n" + "=" * 60)
            self.logger.info("✅ All examples completed successfully!")
            self.logger.info("\nKey Milestone 3 Features Demonstrated:")
            self.logger.info("  ✓ Adaptive Planning (VERIFY/EXPLORE/EXPLOIT)")
            self.logger.info("  ✓ Hybrid Fusion with α-weighting")
            self.logger.info("  ✓ Entity-based Diversification")
            self.logger.info("  ✓ Exact Identifier Matching")
            self.logger.info("  ✓ Sub-200ms Performance Targets")
            self.logger.info("  ✓ Optional Cross-encoder Reranking (disabled by default)")
            self.logger.info("  ✓ Comprehensive Configuration System")
            
        except Exception as e:
            self.logger.error(f"Example failed: {e}")
            raise


def main():
    """Main example runner."""
    runner = Milestone3ExampleRunner()
    runner.run_all_examples()


if __name__ == "__main__":
    main()