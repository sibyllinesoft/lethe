#!/usr/bin/env python3
"""
Complete Lethe→StreamingLLM Hybrid System Test
==============================================

This script tests the complete hybrid system as implemented per TODO.md:
- Verifies all components work together
- Tests the minimal integration pseudocode 
- Validates instrumentation and monitoring
- Demonstrates the evaluation workflow

Usage:
    python test_hybrid_complete.py --mode integration-test
    python test_hybrid_complete.py --mode demo
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any
import json
import time

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from src.context_competitors.lethe_streaming_hybrid import (
        HybridSelector, LetheStreamingHybridCompetitor,
        HybridResult, LetheHeadBuilder, StreamingTailBuilder, 
        KVAwareArranger, AdvancedInstrumentationLogger
    )
    print("✅ Successfully imported hybrid system components")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_minimal_integration_pseudocode():
    """Test the minimal integration pseudocode from TODO.md lines 34-44."""
    print("\n🧪 Testing Minimal Integration Pseudocode")
    print("=" * 50)
    
    # Sample context and query
    test_context = """
    def calculate_fibonacci(n):
        if n <= 1:
            return n
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
    
    class DataProcessor:
        def __init__(self, config):
            self.config = config
            self.cache = {}
        
        def process_data(self, data):
            if data in self.cache:
                return self.cache[data]
            
            result = self._expensive_operation(data)
            self.cache[data] = result
            return result
    
    ValueError: Invalid input data format
    The data must be in JSON format with required fields: id, timestamp, value
    
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    CONFIG_PATH = '/etc/myapp/config.yaml'
    MAX_RETRY_COUNT = 3
    """
    
    test_query = "How do I fix the ValueError and optimize the fibonacci function?"
    
    # Test parameters from TODO.md canary configuration
    lambda_param = 0.001  # Token budget constraint
    mu_param = 0.0001     # Compute budget constraint
    
    print(f"📝 Test Context: {len(test_context.split())} tokens")
    print(f"❓ Query: {test_query}")
    print(f"⚙️ Parameters: λ={lambda_param}, μ={mu_param}")
    
    try:
        # Initialize hybrid selector with canary configuration
        selector = HybridSelector({
            'head_keep': 0.12,
            'window_size': 6000, 
            'stride': 3000,
            'sinks': 96,
            'K2': 320,
            'dpp_rank': 14
        })
        
        print("✅ Hybrid selector initialized")
        
        # Execute the pseudocode logic:
        # H = LETHE_SELECT(blob, λ, μ)
        print("🔄 Step 1: Lethe head selection...")
        head_result = selector.head_builder.build_head(test_context, lambda_param, mu_param)
        print(f"   Head tokens: {head_result.total_tokens}, keep ratio: {head_result.keep_ratio:.3f}")
        
        # Check gating conditions: CODE_INTENT && ENTROPY_HIGH && BUDGET_OK
        print("🔄 Step 2: Gating decision...")
        should_stream, entropy = selector.entropy_analyzer.should_enable_streaming(
            test_context, head_result.keep_ratio
        )
        print(f"   Entity entropy: {entropy:.3f}, should stream: {should_stream}")
        
        tail_result = None
        if should_stream:
            print("🔄 Step 3: StreamingLLM tail processing...")
            head_summary = " ".join([
                f"{group.group_type}: {' '.join(group.atoms[:2])}"
                for group in head_result.selected_atoms[:3]
            ])
            
            budget_tokens = len(test_context.split()) - head_result.total_tokens
            if budget_tokens > 1000:  # Minimum budget for streaming
                tail_result = selector.tail_builder.build_tail(
                    test_context, head_summary, lambda_param, mu_param, budget_tokens
                )
                print(f"   Tail tokens: {tail_result.total_tokens}, windows: {tail_result.num_windows}")
        
        # S = H ∪ T
        print("🔄 Step 4: KV-aware arrangement...")
        final_context = selector.kv_arranger.arrange_for_kv_optimization(head_result, tail_result)
        
        print(f"✅ Final context: {len(final_context)} characters")
        print(f"📊 Total tokens: {len(final_context.split())}")
        
        # Test complete hybrid selection
        print("\n🔄 Step 5: Complete hybrid selection...")
        hybrid_result = selector.select(test_query, test_context, lambda_param, mu_param)
        
        print("\n📈 Hybrid Results:")
        print(f"   Gating decision: {hybrid_result.gating_decision}")
        print(f"   Total tokens: {hybrid_result.total_tokens}")
        print(f"   Keep ratio: {hybrid_result.keep_ratio:.3f}")
        print(f"   Processing time: {hybrid_result.processing_time_ms:.1f}ms")
        
        # Advanced instrumentation metrics
        inst = hybrid_result.instrumentation
        print(f"   KV reuse: {inst.kv_prefix_reuse:.3f}")
        print(f"   Primal-dual gap: {inst.primal_dual_gap:.4f}")
        print(f"   Tail CVaR: {inst.tail_cvar_95:.2f}")
        
        print("\n✅ Minimal integration pseudocode test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_competitor_integration():
    """Test the competitor integration for benchmarking."""
    print("\n🧪 Testing Competitor Integration")
    print("=" * 50)
    
    try:
        # Initialize hybrid competitor
        competitor = LetheStreamingHybridCompetitor()
        
        if not competitor.initialize():
            print("❌ Competitor initialization failed")
            return False
        
        print("✅ Hybrid competitor initialized")
        
        # Test context processing
        test_query = "What is the main algorithm complexity issue?"
        test_context = """
        The recursive fibonacci implementation has exponential time complexity O(2^n)
        because it recalculates the same subproblems multiple times. This can be 
        optimized using dynamic programming to achieve O(n) time complexity.
        
        def fibonacci_optimized(n, memo={}):
            if n in memo:
                return memo[n]
            if n <= 1:
                return n
            memo[n] = fibonacci_optimized(n-1, memo) + fibonacci_optimized(n-2, memo)
            return memo[n]
        """
        
        print(f"📝 Processing context: {len(test_context.split())} tokens")
        
        result = competitor.process_context(test_query, test_context, max_tokens=200)
        
        print("📊 Processing Results:")
        print(f"   Original tokens: {result.original_token_count}")
        print(f"   Processed tokens: {result.processed_token_count}")
        print(f"   Compression ratio: {result.compression_ratio:.3f}")
        print(f"   Processing time: {result.processing_time_ms:.1f}ms")
        
        # Check metadata for hybrid-specific metrics
        metadata = result.metadata or {}
        print(f"   Gating decision: {metadata.get('gating_decision', 'unknown')}")
        print(f"   Head tokens: {metadata.get('head_tokens', 0)}")
        print(f"   Tail tokens: {metadata.get('tail_tokens', 0)}")
        print(f"   KV reuse: {metadata.get('kv_reuse', 0.0):.3f}")
        
        # Get hybrid stats
        stats = competitor.get_hybrid_stats()
        monitoring = stats.get('monitoring_summary', {})
        
        print("\n📈 Advanced Monitoring:")
        evt_data = monitoring.get('evt_tail_model', {})
        print(f"   ξ parameter: {evt_data.get('xi_parameter', 0.0):.4f}")
        print(f"   Tail CVaR: {evt_data.get('tail_cvar_95', 0.0):.2f}")
        
        gap_data = monitoring.get('primal_dual_gap', {})
        print(f"   Current gap: {gap_data.get('current_gap', 0.0):.4f}")
        print(f"   Converged: {gap_data.get('converged', False)}")
        
        # Test adaptive adjustments
        adjustments = competitor.apply_adaptive_adjustments()
        if adjustments:
            print(f"   Adaptive adjustments: {adjustments}")
        
        print("\n✅ Competitor integration test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_demo():
    """Run a comprehensive demo of the hybrid system."""
    print("\n🎬 Lethe→StreamingLLM Hybrid System Demo")
    print("=" * 60)
    
    # Demo context simulating a real code debugging scenario
    demo_context = """
    import asyncio
    import aiohttp
    import logging
    from typing import Dict, List, Optional
    from dataclasses import dataclass
    import json
    
    @dataclass
    class APIResponse:
        status_code: int
        data: Optional[Dict]
        error: Optional[str] = None
    
    class WebAPIClient:
        def __init__(self, base_url: str, timeout: int = 30):
            self.base_url = base_url
            self.timeout = timeout
            self.session = None
            
        async def __aenter__(self):
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
            return self
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if self.session:
                await self.session.close()
        
        async def get_user(self, user_id: int) -> APIResponse:
            try:
                async with self.session.get(f"{self.base_url}/users/{user_id}") as response:
                    if response.status == 200:
                        data = await response.json()
                        return APIResponse(status_code=200, data=data)
                    else:
                        error_text = await response.text()
                        return APIResponse(status_code=response.status, data=None, error=error_text)
                        
            except asyncio.TimeoutError:
                return APIResponse(status_code=408, data=None, error="Request timeout")
            except aiohttp.ClientError as e:
                return APIResponse(status_code=500, data=None, error=str(e))
    
    # The problem: Memory leak in long-running service
    class UserService:
        def __init__(self):
            self.cache = {}  # This grows unbounded!
            self.api_client = WebAPIClient("https://api.example.com")
            
        async def get_user_info(self, user_id: int) -> Optional[Dict]:
            # Check cache first
            if user_id in self.cache:
                logging.info(f"Cache hit for user {user_id}")
                return self.cache[user_id]
            
            # Fetch from API
            async with self.api_client as client:
                response = await client.get_user(user_id)
                
            if response.status_code == 200:
                # Store in cache (MEMORY LEAK!)
                self.cache[user_id] = response.data
                return response.data
            else:
                logging.error(f"Failed to fetch user {user_id}: {response.error}")
                return None
    
    # Error logs showing the memory issue
    ERROR: Memory usage critical: 8.2GB used (95% of available)
    WARNING: Large cache detected in UserService: 1.2M entries
    ERROR: OutOfMemoryError in worker process 3
    CRITICAL: Service becoming unresponsive due to GC pressure
    
    # Configuration that might help
    MAX_CACHE_SIZE = 10000
    CACHE_TTL_SECONDS = 3600
    ENABLE_CACHE_METRICS = True
    
    # Additional context about the system
    This is a high-traffic user service handling 100k requests/minute.
    The service runs in Kubernetes with 16GB memory limit per pod.
    Cache hit rate is currently 85% which is helping performance.
    But memory usage grows continuously until pod restart required.
    """
    
    demo_query = "How do I fix the memory leak in the UserService while maintaining good cache performance?"
    
    print(f"🎯 Query: {demo_query}")
    print(f"📄 Context: {len(demo_context.split())} tokens")
    
    # Initialize hybrid selector with canary config
    selector = HybridSelector({
        'head_keep': 0.12,
        'window_size': 6000,
        'stride': 3000, 
        'sinks': 96,
        'K2': 320,
        'dpp_rank': 14
    })
    
    # Run hybrid selection
    start_time = time.time()
    result = selector.select(demo_query, demo_context, lambda_param=0.15, mu_param=0.05)
    end_time = time.time()
    
    print(f"\n📊 Hybrid Processing Results:")
    print(f"   Processing time: {(end_time - start_time) * 1000:.1f}ms")
    print(f"   Gating decision: {result.gating_decision}")
    print(f"   Keep ratio: {result.keep_ratio:.3f}")
    print(f"   Total tokens: {result.total_tokens}")
    
    if result.head_result:
        print(f"\n🧠 Head (Lethe) Results:")
        print(f"   Tokens: {result.head_result.total_tokens}")
        print(f"   Groups selected: {len(result.head_result.selected_atoms)}")
        for i, group in enumerate(result.head_result.selected_atoms[:3]):
            print(f"   Group {i+1}: {group.group_type} ({group.total_tokens} tokens)")
    
    if result.tail_result:
        print(f"\n🔄 Tail (StreamingLLM) Results:")
        print(f"   Tokens: {result.tail_result.total_tokens}")
        print(f"   Windows: {result.tail_result.num_windows}")
        print(f"   Stride: {result.tail_result.stride}")
    
    # Advanced instrumentation
    inst = result.instrumentation
    print(f"\n🔬 Advanced Instrumentation:")
    print(f"   λ parameter: {inst.lambda_param}")
    print(f"   μ parameter: {inst.mu_param}")
    print(f"   KV prefix reuse: {inst.kv_prefix_reuse:.3f}")
    print(f"   Primal-dual gap: {inst.primal_dual_gap:.4f}")
    print(f"   Tail CVaR₀.₉₅: {inst.tail_cvar_95:.2f}ms")
    print(f"   CE early exit: {inst.ce_early_exit}")
    print(f"   DPP rank: {inst.dpp_rank}")
    
    # Show context preview
    print(f"\n📝 Processed Context Preview:")
    preview = result.final_context[:500] + "..." if len(result.final_context) > 500 else result.final_context
    print(preview)
    
    # Adaptive recommendations
    adjustments = selector.get_adaptive_adjustments()
    if adjustments['actions_recommended']:
        print(f"\n🔧 Adaptive Recommendations:")
        for recommendation in adjustments['actions_recommended']:
            print(f"   • {recommendation}")
    else:
        print(f"\n✅ System operating within optimal parameters")
    
    print(f"\n🎉 Demo completed successfully!")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Hybrid System Complete Test')
    parser.add_argument('--mode', choices=['integration-test', 'demo'], 
                       default='integration-test', help='Test mode')
    
    args = parser.parse_args()
    
    success = True
    
    if args.mode == 'integration-test':
        print("🧪 Running Integration Test Suite")
        print("=" * 60)
        
        # Test 1: Minimal integration pseudocode
        success &= test_minimal_integration_pseudocode()
        
        # Test 2: Competitor integration
        success &= test_competitor_integration()
        
        if success:
            print("\n✅ ALL INTEGRATION TESTS PASSED")
            print("🚀 Hybrid system is fully operational and ready for evaluation!")
        else:
            print("\n❌ SOME TESTS FAILED")
            print("🔧 Please check the implementation and try again")
    
    elif args.mode == 'demo':
        run_demo()
    
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())