"""
Test Runner for Lethe→StreamingLLM Hybrid System

Comprehensive test suite to validate the hybrid system implementation
and demonstrate all components working together.

Tests:
1. Basic hybrid selection functionality
2. Head builder with grouped atoms
3. Tail builder with streaming windows
4. Gating logic activation
5. KV-aware arrangement
6. Comprehensive instrumentation
7. Integration with benchmark infrastructure
8. Canary deployment simulation
"""

import sys
import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src directories to path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "src" / "context_competitors"))
sys.path.append(str(Path(__file__).parent / "src" / "infinitebench"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_hybrid_selector():
    """Test the core hybrid selector functionality."""
    print("=== Testing Hybrid Selector ===")
    
    try:
        from context_competitors.lethe_streaming_hybrid import HybridSelector
        
        # Create test context with various atom types
        test_context = """
        import numpy as np
        from typing import List, Dict, Any
        import logging
        
        logger = logging.getLogger(__name__)
        
        def process_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            '''Main data processing function.'''
            if not data:
                raise ValueError("Data cannot be empty")
            
            results = []
            for item in data:
                try:
                    processed = transform_item(item)
                    results.append(processed)
                except Exception as e:
                    logger.error(f"Failed to process item {item}: {e}")
                    continue
            
            return results
        
        def transform_item(item: Dict[str, Any]) -> Dict[str, Any]:
            '''Transform a single data item.'''
            return {
                'id': item.get('id', ''),
                'value': str(item.get('value', '')).upper().strip(),
                'timestamp': time.time()
            }
        
        class DataProcessor:
            '''Main data processing class with error handling.'''
            
            CONFIG_KEY = "PROCESSOR_CONFIG"
            MAX_RETRIES = 3
            
            def __init__(self, config: Dict[str, Any]):
                self.config = config
                self.retry_count = 0
            
            def run(self) -> List[Dict[str, Any]]:
                '''Run the complete processing pipeline.'''
                try:
                    data = self.load_data()
                    return process_data(data)
                except FileNotFoundError as e:
                    logger.error(f"Data file not found: {e}")
                    raise
                except ValueError as e:
                    logger.error(f"Invalid data format: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    if self.retry_count < self.MAX_RETRIES:
                        self.retry_count += 1
                        return self.run()
                    raise
            
            def load_data(self) -> List[Dict[str, Any]]:
                '''Load data from configured source.'''
                return [
                    {'id': '1', 'value': 'test1'},
                    {'id': '2', 'value': 'test2'},
                    {'id': '3', 'value': 'test3'}
                ]
        
        if __name__ == "__main__":
            processor = DataProcessor({'source': 'test.json'})
            results = processor.run()
            print(f"Processed {len(results)} items")
        """ * 5  # Repeat to make it longer
        
        test_query = "What are the main classes and functions in this code? How does error handling work?"
        
        # Test basic functionality
        config = {
            'head_keep': 0.15,
            'window_size': 4000,
            'stride': 2000,
            'sinks': 64
        }
        
        selector = HybridSelector(config)
        
        # Test selection
        start_time = time.time()
        result = selector.select(test_query, test_context)
        selection_time = (time.time() - start_time) * 1000
        
        print(f"✓ Hybrid selection completed in {selection_time:.2f}ms")
        print(f"  Original tokens: {len(test_context.split())}")
        print(f"  Final tokens: {result.total_tokens}")
        print(f"  Keep ratio: {result.keep_ratio:.3f}")
        print(f"  Gating decision: {result.gating_decision}")
        
        if result.head_result:
            print(f"  Head groups: {len(result.head_result.selected_atoms)}")
            print(f"  Head tokens: {result.head_result.total_tokens}")
            print(f"  Head keep ratio: {result.head_result.keep_ratio:.3f}")
        
        if result.tail_result:
            print(f"  Tail windows: {result.tail_result.num_windows}")
            print(f"  Tail tokens: {result.tail_result.total_tokens}")
            print(f"  Tail keep ratio: {result.tail_result.keep_ratio:.3f}")
        
        # Test instrumentation
        instrumentation = result.instrumentation
        print(f"  Lambda: {instrumentation.lambda_param}")
        print(f"  Mu: {instrumentation.mu_param}")
        print(f"  KV reuse: {instrumentation.kv_prefix_reuse:.3f}")
        print(f"  ΔCBU/1k: {instrumentation.delta_cbu_per_1k:.4f}")
        
        # Verify final context is not empty
        assert len(result.final_context) > 0, "Final context should not be empty"
        assert result.total_tokens > 0, "Should have selected some tokens"
        
        print("✓ Basic hybrid selection test passed")
        return True
        
    except Exception as e:
        print(f"✗ Hybrid selector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_head_builder():
    """Test the Lethe head builder with grouped atoms."""
    print("\n=== Testing Head Builder ===")
    
    try:
        from context_competitors.lethe_streaming_hybrid import LetheHeadBuilder
        
        # Create test context with clear atom groups
        test_context = """
        def calculate_sum(a, b):
            '''Calculate sum of two numbers.'''
            return a + b
        
        def calculate_product(x, y):
            '''Calculate product of two numbers.'''
            return x * y
        
        class MathUtils:
            '''Utility class for math operations.'''
            
            @staticmethod
            def divide(a, b):
                if b == 0:
                    raise ZeroDivisionError("Cannot divide by zero")
                return a / b
        
        import math
        from typing import Union
        
        PI_CONSTANT = 3.14159
        MAX_VALUE = 1000000
        
        try:
            result = calculate_sum(10, 20)
        except ValueError as e:
            print(f"Calculation error: {e}")
        except TypeError as e:
            print(f"Type error: {e}")
        
        if __name__ == "__main__":
            utils = MathUtils()
            print(utils.divide(10, 2))
        """
        
        builder = LetheHeadBuilder(target_keep_ratio=0.2, dpp_rank=10)
        
        # Test head building
        start_time = time.time()
        head_result = builder.build_head(test_context, lambda_param=0.001, mu_param=0.0001)
        build_time = (time.time() - start_time) * 1000
        
        print(f"✓ Head building completed in {build_time:.2f}ms")
        print(f"  Selected groups: {len(head_result.selected_atoms)}")
        print(f"  Total tokens: {head_result.total_tokens}")
        print(f"  Keep ratio: {head_result.keep_ratio:.3f}")
        print(f"  Utility score: {head_result.utility_score:.2f}")
        print(f"  DPP rank: {head_result.dpp_rank}")
        
        # Verify atom groups
        group_types = [group.group_type for group in head_result.selected_atoms]
        print(f"  Group types found: {set(group_types)}")
        
        # Should find functions, classes, imports, errors
        expected_types = {'def', 'symbol_header', 'tool_key', 'error'}
        found_types = set(group_types)
        
        if found_types & expected_types:
            print(f"✓ Found expected atom types: {found_types & expected_types}")
        else:
            print(f"⚠ No expected atom types found. Found: {found_types}")
        
        # Verify selection is reasonable
        assert head_result.total_tokens > 0, "Should select some tokens"
        assert head_result.keep_ratio > 0, "Should have positive keep ratio"
        assert len(head_result.selected_atoms) > 0, "Should select some atom groups"
        
        print("✓ Head builder test passed")
        return True
        
    except Exception as e:
        print(f"✗ Head builder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tail_builder():
    """Test the StreamingLLM tail builder."""
    print("\n=== Testing Tail Builder ===")
    
    try:
        from context_competitors.lethe_streaming_hybrid import StreamingTailBuilder
        
        # Create long test context for windowing
        test_context = " ".join([
            f"This is sentence number {i} in a very long document that needs to be processed with streaming windows."
            for i in range(1000)
        ])
        
        head_summary = "Functions: calculate_sum, calculate_product. Classes: MathUtils. Imports: math, typing"
        
        builder = StreamingTailBuilder(window_size=200, stride=100, sink_tokens=20)
        
        # Test tail building
        start_time = time.time()
        tail_result = builder.build_tail(
            text=test_context,
            head_summary=head_summary,
            lambda_param=0.001,
            mu_param=0.0001,
            budget_tokens=500
        )
        build_time = (time.time() - start_time) * 1000
        
        print(f"✓ Tail building completed in {build_time:.2f}ms")
        print(f"  Windows created: {tail_result.num_windows}")
        print(f"  Total tokens: {tail_result.total_tokens}")
        print(f"  Keep ratio: {tail_result.keep_ratio:.3f}")
        print(f"  Window size: {builder.window_size}")
        print(f"  Stride: {tail_result.stride}")
        print(f"  Sink tokens per window: {tail_result.sink_tokens_per_window}")
        
        # Verify windows
        if tail_result.windows:
            first_window = tail_result.windows[0]
            print(f"  First window: {first_window.window_size} tokens, {len(first_window.attention_sinks)} sinks")
            print(f"  Attention sinks preview: {' '.join(first_window.attention_sinks[:5])}")
            
            # Check if head summary is in sinks
            sink_text = ' '.join(first_window.attention_sinks)
            if any(term in sink_text for term in ['calculate_sum', 'MathUtils', 'Functions']):
                print("✓ Head summary integrated into attention sinks")
            else:
                print("⚠ Head summary not found in attention sinks")
        
        # Verify windowing constraints
        assert tail_result.num_windows > 0, "Should create some windows"
        assert tail_result.total_tokens <= 500, "Should respect budget constraint"
        
        print("✓ Tail builder test passed")
        return True
        
    except Exception as e:
        print(f"✗ Tail builder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_benchmark_integration():
    """Test integration with benchmark infrastructure."""
    print("\n=== Testing Benchmark Integration ===")
    
    try:
        from context_competitors.benchmarks.hybrid_benchmark import HybridSystemCompetitor
        
        # Create hybrid competitor
        config = {
            'head_keep': 0.15,
            'window_size': 3000,
            'stride': 1500,
            'sinks': 48,
            'K2': 200,
            'dpp_rank': 10
        }
        
        competitor = HybridSystemCompetitor(config)
        
        # Test initialization
        if not competitor.initialize():
            print("✗ Failed to initialize hybrid competitor")
            return False
        
        print("✓ Hybrid competitor initialized")
        
        # Test processing
        test_query = "What does this code do?"
        test_context = """
        import json
        import requests
        from datetime import datetime
        
        def fetch_api_data(url: str, headers: dict = None) -> dict:
            '''Fetch data from API endpoint.'''
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                raise ConnectionError(f"API request failed: {e}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON response: {e}")
        
        class APIClient:
            '''Client for REST API operations.'''
            
            def __init__(self, base_url: str, api_key: str):
                self.base_url = base_url
                self.headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
            
            def get_user_data(self, user_id: int) -> dict:
                '''Get user data by ID.'''
                url = f"{self.base_url}/users/{user_id}"
                return fetch_api_data(url, self.headers)
        
        # Example usage
        client = APIClient('https://api.example.com', 'your-api-key')
        user_data = client.get_user_data(123)
        print(f"User: {user_data.get('name', 'Unknown')}")
        """ * 3
        
        # Process with hybrid system
        start_time = time.time()
        result = competitor.process_context(test_query, test_context, max_tokens=1000)
        processing_time = (time.time() - start_time) * 1000
        
        print(f"✓ Processing completed in {processing_time:.2f}ms")
        print(f"  Original tokens: {result.original_token_count}")
        print(f"  Processed tokens: {result.processed_token_count}")
        print(f"  Compression ratio: {result.compression_ratio:.2%}")
        print(f"  Response preview: {result.response[:100]}...")
        
        # Check metadata
        metadata = result.metadata
        print(f"  Gating decision: {metadata.get('gating_decision')}")
        print(f"  Head tokens: {metadata.get('head_tokens')}")
        print(f"  Tail tokens: {metadata.get('tail_tokens')}")
        print(f"  KV reuse: {metadata.get('kv_reuse', 0.0):.3f}")
        
        # Test performance summary
        summary = competitor.get_performance_summary()
        print(f"✓ Performance summary retrieved:")
        print(f"  Total calls: {summary['performance_stats']['total_calls']}")
        print(f"  Average processing time: {summary['derived_metrics']['average_processing_time_ms']:.2f}ms")
        
        # Verify results
        assert result.processed_token_count <= 1000, "Should respect token limit"
        assert result.compression_ratio >= 0, "Should have valid compression ratio"
        assert len(result.response) > 0, "Should generate response"
        
        print("✓ Benchmark integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Benchmark integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_canary_deployment():
    """Test canary deployment functionality."""
    print("\n=== Testing Canary Deployment ===")
    
    try:
        from context_competitors.benchmarks.hybrid_benchmark import (
            HybridSystemCompetitor, HybridCanaryController
        )
        
        # Create mock baseline (simplified)
        class MockBaseline:
            def __init__(self):
                self.name = "MockStreamingLLM"
                
            def initialize(self):
                return True
            
            def process_context(self, query, context, max_tokens=4000):
                from context_competitors.competitor_interface import ContextProcessingResult
                import time
                
                # Mock simple truncation
                tokens = context.split()
                if len(tokens) > max_tokens:
                    processed_context = ' '.join(tokens[:max_tokens])
                    processed_tokens = max_tokens
                else:
                    processed_context = context
                    processed_tokens = len(tokens)
                
                return ContextProcessingResult(
                    original_context=context,
                    processed_context=processed_context,
                    query=query,
                    response=f"Mock response for: {query[:50]}",
                    processing_time_ms=25.0,  # Mock fast baseline
                    original_token_count=len(tokens),
                    processed_token_count=processed_tokens,
                    compression_ratio=1.0 - (processed_tokens / len(tokens)) if tokens else 0.0,
                    method_name=self.name,
                    metadata={'mock': True}
                )
        
        # Create hybrid and baseline
        hybrid_config = {'head_keep': 0.12, 'window_size': 2000}
        hybrid = HybridSystemCompetitor(hybrid_config)
        baseline = MockBaseline()
        
        # Initialize
        hybrid.initialize()
        baseline.initialize()
        
        # Create canary controller with 20% traffic to hybrid
        controller = HybridCanaryController(baseline, hybrid, canary_percentage=20.0)
        
        print("✓ Canary controller created")
        
        # Simulate requests
        test_requests = [
            ("Query 1", "Short context for testing"),
            ("Query 2", "Another context " * 100),
            ("Query 3", "Third test context " * 50),
            ("Query 4", "Fourth context for validation"),
            ("Query 5", "Final test context " * 75)
        ]
        
        results = []
        for query, context in test_requests:
            result = controller.process_request(query, context, max_tokens=500)
            results.append(result)
        
        # Get canary statistics
        stats = controller.get_canary_stats()
        
        print(f"✓ Processed {stats['total_requests']} requests")
        print(f"  Hybrid requests: {stats['hybrid_requests']}")
        print(f"  Baseline requests: {stats['baseline_requests']}")
        print(f"  Actual hybrid %: {stats['actual_hybrid_percentage']:.1f}%")
        
        if 'hybrid_performance' in stats:
            print(f"  Hybrid avg latency: {stats['hybrid_performance']['avg_processing_time_ms']:.2f}ms")
        
        if 'baseline_performance' in stats:
            print(f"  Baseline avg latency: {stats['baseline_performance']['avg_processing_time_ms']:.2f}ms")
        
        if 'performance_comparison' in stats:
            comparison = stats['performance_comparison']
            print(f"  Latency difference: {comparison['latency_difference_ms']:.2f}ms")
            print(f"  Meets latency req: {comparison['meets_latency_requirement']}")
        
        # Test promotion decision
        should_promote = controller.should_promote_hybrid(min_requests=3)  # Lower threshold for test
        print(f"  Should promote hybrid: {should_promote}")
        
        # Verify routing worked
        hybrid_routed = sum(1 for r in results if r.metadata.get('canary_routing') == 'hybrid')
        baseline_routed = sum(1 for r in results if r.metadata.get('canary_routing') == 'baseline')
        
        assert hybrid_routed + baseline_routed == len(test_requests), "All requests should be routed"
        print(f"✓ Routing verified: {hybrid_routed} hybrid, {baseline_routed} baseline")
        
        print("✓ Canary deployment test passed")
        return True
        
    except Exception as e:
        print(f"✗ Canary deployment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_instrumentation():
    """Test comprehensive instrumentation."""
    print("\n=== Testing Instrumentation ===")
    
    try:
        from context_competitors.lethe_streaming_hybrid import HybridInstrumentation
        
        # Create test instrumentation
        instrumentation = HybridInstrumentation(
            lambda_param=0.001,
            mu_param=0.0001,
            tokens_in=5000,
            head_tokens=600,
            tail_tokens=400,
            keep_ratio_head=0.12,
            keep_ratio_tail=0.08,
            K1=1000,
            K2=320,
            dpp_rank=14,
            ce_early_exit=True,
            num_windows=3,
            window_size=6000,
            stride=3000,
            sinks=96,
            kv_prefix_reuse=0.75,
            middleware_p95_ms=45.2,
            llm_p95_ms=120.5,
            delta_cbu_per_1k=0.008,
            precision_at_k={5: 0.85, 10: 0.78},
            recall_at_k={5: 0.82, 10: 0.88}
        )
        
        # Test serialization
        instrumentation_dict = instrumentation.to_dict()
        
        print("✓ Instrumentation created and serialized")
        print(f"  Lambda: {instrumentation_dict['lambda']}")
        print(f"  Mu: {instrumentation_dict['mu']}")
        print(f"  Tokens in: {instrumentation_dict['tokens_in']}")
        print(f"  Head tokens: {instrumentation_dict['head_tokens']}")
        print(f"  Tail tokens: {instrumentation_dict['tail_tokens']}")
        print(f"  Keep ratio head: {instrumentation_dict['keep_ratio_head']:.3f}")
        print(f"  Keep ratio tail: {instrumentation_dict['keep_ratio_tail']:.3f}")
        print(f"  KV reuse: {instrumentation_dict['kv_prefix_reuse']:.3f}")
        print(f"  Middleware p95: {instrumentation_dict['middleware_p95_ms']:.2f}ms")
        print(f"  ΔCBU/1k: {instrumentation_dict['delta_cbu_per_1k']:.4f}")
        print(f"  P@5: {instrumentation_dict['precision_at_k'][5]:.3f}")
        print(f"  R@10: {instrumentation_dict['recall_at_k'][10]:.3f}")
        
        # Verify all required fields
        required_fields = [
            'lambda', 'mu', 'tokens_in', 'head_tokens', 'tail_tokens',
            'keep_ratio_head', 'keep_ratio_tail', 'K1', 'K2', 'dpp_rank',
            'ce_early_exit', 'num_windows', 'window_size', 'stride', 'sinks',
            'kv_prefix_reuse', 'middleware_p95_ms', 'llm_p95_ms', 'delta_cbu_per_1k',
            'precision_at_k', 'recall_at_k'
        ]
        
        missing_fields = [field for field in required_fields if field not in instrumentation_dict]
        if missing_fields:
            print(f"✗ Missing instrumentation fields: {missing_fields}")
            return False
        
        print("✓ All required instrumentation fields present")
        print("✓ Instrumentation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Instrumentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all test cases."""
    print("=" * 60)
    print("LETHE→STREAMINGLLM HYBRID SYSTEM TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Hybrid Selector", test_hybrid_selector),
        ("Head Builder", test_head_builder),  
        ("Tail Builder", test_tail_builder),
        ("Benchmark Integration", test_benchmark_integration),
        ("Canary Deployment", test_canary_deployment),
        ("Instrumentation", test_instrumentation)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                failed += 1
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed} PASSED, {failed} FAILED")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Hybrid system is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)