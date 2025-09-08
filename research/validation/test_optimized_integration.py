#!/usr/bin/env python3
"""
Test script to validate HybridOptimizerSystem integration with InfiniteBench evaluation.
"""

import sys
import logging
import time
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "ctx-run" / "packages" / "sqlite" / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_optimized_system_import():
    """Test that optimized system can be imported."""
    try:
        from hybrid_optimizations import HybridOptimizerSystem, OptimizationConfig
        from hybrid_selector import HybridConfig
        logger.info("✅ Successfully imported HybridOptimizerSystem")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import HybridOptimizerSystem: {e}")
        return False

def test_benchmarking_integration():
    """Test that benchmarking system can use optimized components."""
    try:
        from benchmarking import LetheStreamingHybridCompetitor, BenchmarkMethod, CompetitorConfig
        
        # Create optimized competitor configuration
        config = CompetitorConfig(
            method=BenchmarkMethod.HYBRID,
            keep_ratio=0.15,
            config_params={
                'window_size': 6000,
                'stride': 3000,
                'sink_tokens': 96,
                'dpp_rank': 14,
                'ce_k2': 320,
                'use_optimizations': True
            }
        )
        
        # Initialize competitor
        competitor = LetheStreamingHybridCompetitor(BenchmarkMethod.HYBRID, config)
        
        # Test initialization
        if not competitor.initialize():
            logger.error("❌ Failed to initialize hybrid competitor")
            return False
            
        logger.info("✅ Successfully created and initialized optimized hybrid competitor")
        
        # Check if optimizer is available
        if hasattr(competitor, 'hybrid_optimizer'):
            logger.info("✅ HybridOptimizerSystem is available in competitor")
        else:
            logger.warning("⚠️ HybridOptimizerSystem not available, using fallback")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import benchmarking components: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error during benchmarking integration test: {e}")
        return False

def test_processing_performance():
    """Test processing performance with sample content."""
    try:
        from benchmarking import LetheStreamingHybridCompetitor, BenchmarkMethod, CompetitorConfig, BenchmarkSample, DatasetType
        
        # Create test content
        test_content = """
def hybrid_selection_main(content, config):
    '''Main hybrid selection function with optimizations.'''
    atoms = extract_atoms_optimized(content)
    return process_hybrid_optimized(atoms, config)

Error: Failed to process selection with latency regression
    at line 42 in hybrid_selector.py
TypeError: unsupported operation in gating logic
PerformanceError: p95 latency exceeded target (150ms > 100ms)

@tool
def search_api_optimized(query):
    '''Optimized API search with caching.'''
    API_KEY = "test_key"
    return call_cached_endpoint(query)

# Processing context for optimization
The system handles both stable head content and streaming tail content
using optimized Lethe DPP selection and enhanced StreamingLLM windowing.
Performance targets: p95 <50ms (optimized from 100ms), KV reuse >80% (optimized from 60%).

class OptimizedProcessor:
    def __init__(self):
        self.cache = LRUCache(maxsize=1000)
        self.pattern_matcher = CachedPatternMatcher()
        self.entity_extractor = OptimizedEntityExtractor()
        
    def process(self, content):
        return self.optimized_processing_pipeline(content)
"""
        
        # Create competitor with optimizations
        config = CompetitorConfig(
            method=BenchmarkMethod.HYBRID,
            keep_ratio=0.30,
            config_params={
                'window_size': 6000,
                'stride': 3000,
                'sink_tokens': 96,
                'dpp_rank': 14,
                'ce_k2': 320
            }
        )
        
        competitor = LetheStreamingHybridCompetitor(BenchmarkMethod.HYBRID, config)
        competitor.initialize()
        
        # Create test sample
        sample = BenchmarkSample(
            sample_id="optimization_test_001",
            dataset=DatasetType.CODE_DEBUG,
            input_text=test_content,
            reference_answer="Optimize hybrid selection performance and reduce latency regression",
            metadata={'test_type': 'optimization_validation'}
        )
        
        # Process and measure performance
        start_time = time.perf_counter()
        result = competitor.process_sample(sample)
        end_time = time.perf_counter()
        
        processing_time_ms = (end_time - start_time) * 1000
        
        # Validate results
        logger.info(f"🔍 Processing Results:")
        logger.info(f"   Tokens kept: {result.tokens_kept}")
        logger.info(f"   Keep ratio: {result.keep_ratio:.3f}")
        logger.info(f"   Processing time: {processing_time_ms:.2f}ms")
        logger.info(f"   F1 score: {result.f1_score:.3f}")
        logger.info(f"   KV reuse ratio: {result.kv_reuse_ratio:.3f}")
        logger.info(f"   Head tokens: {result.head_tokens}")
        logger.info(f"   Tail tokens: {result.tail_tokens}")
        logger.info(f"   Number of windows: {result.num_windows}")
        
        # Performance validation
        if processing_time_ms < 100:  # Target: significant improvement from baseline
            logger.info(f"✅ Processing time within target ({processing_time_ms:.2f}ms < 100ms)")
        else:
            logger.warning(f"⚠️ Processing time above target ({processing_time_ms:.2f}ms >= 100ms)")
        
        if result.kv_reuse_ratio > 0.6:  # Target: good KV cache reuse
            logger.info(f"✅ KV reuse ratio meets target ({result.kv_reuse_ratio:.3f} > 0.6)")
        else:
            logger.warning(f"⚠️ KV reuse ratio below target ({result.kv_reuse_ratio:.3f} <= 0.6)")
            
        if result.tokens_kept > 0:
            logger.info(f"✅ Successfully processed content ({result.tokens_kept} tokens)")
        else:
            logger.error(f"❌ No tokens were processed")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during processing performance test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evaluation_script_compatibility():
    """Test compatibility with evaluation script interface."""
    try:
        from benchmarking import LetheStreamingHybridCompetitor, BenchmarkMethod, CompetitorConfig
        
        # Test the interface used by evaluation script
        config = CompetitorConfig(
            method=BenchmarkMethod.HYBRID,
            keep_ratio=0.15,
            config_params={'window_size': 6000, 'stride': 3000}
        )
        
        competitor = LetheStreamingHybridCompetitor(BenchmarkMethod.HYBRID, config)
        competitor.initialize()
        
        # Test process_context method (used by evaluation script)
        result = competitor.process_context(
            query="What is the main error in this code?",
            context="def test(): return undefined_variable",
            max_tokens=1000
        )
        
        # Validate result structure
        assert hasattr(result, 'accuracy_score'), "Missing accuracy_score attribute"
        assert hasattr(result, 'response'), "Missing response attribute"  
        assert hasattr(result, 'processed_token_count'), "Missing processed_token_count attribute"
        assert hasattr(result, 'metadata'), "Missing metadata attribute"
        
        logger.info(f"✅ Evaluation script compatibility validated")
        logger.info(f"   Accuracy score: {result.accuracy_score:.3f}")
        logger.info(f"   Processed tokens: {result.processed_token_count}")
        logger.info(f"   Response: {result.response}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during evaluation script compatibility test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    logger.info("🚀 Starting HybridOptimizerSystem integration tests...")
    
    tests = [
        ("Optimized System Import", test_optimized_system_import),
        ("Benchmarking Integration", test_benchmarking_integration),
        ("Processing Performance", test_processing_performance),
        ("Evaluation Script Compatibility", test_evaluation_script_compatibility)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                logger.info(f"✅ PASSED: {test_name}")
                passed += 1
            else:
                logger.error(f"❌ FAILED: {test_name}")
                failed += 1
        except Exception as e:
            logger.error(f"❌ FAILED: {test_name} - Exception: {e}")
            failed += 1
    
    logger.info(f"\n📊 Test Results:")
    logger.info(f"   Passed: {passed}")
    logger.info(f"   Failed: {failed}")
    logger.info(f"   Total: {passed + failed}")
    
    if failed == 0:
        logger.info("🎉 All integration tests passed! The optimized system is ready for evaluation.")
        return 0
    else:
        logger.error("💥 Some integration tests failed. Please address issues before running evaluation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())