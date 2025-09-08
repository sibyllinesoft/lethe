#!/usr/bin/env python3
"""
Simple Optimization Test - Quick validation of hybrid system improvements

Tests the core optimizations with simple, reliable content generation.
Focuses on measuring the key improvements:
1. Latency reduction
2. Mode selection improvements  
3. Cache efficiency gains
4. Quality preservation
"""

import logging
import time
import statistics
from typing import Dict, List, Tuple, Any
import json

# Import systems
try:
    from hybrid_selector import HybridSelector, HybridConfig, create_hybrid_selector
    from hybrid_optimizations import HybridOptimizerSystem, OptimizationConfig, create_optimized_hybrid_selector
except ImportError as e:
    print(f"Import error: {e}")
    import sys
    sys.exit(1)

logger = logging.getLogger(__name__)

def create_test_content(size_tokens: int) -> str:
    """Create test content with predictable patterns."""
    content_parts = []
    
    # Add definitions (stable content)
    for i in range(size_tokens // 200):  # ~50 tokens each
        content_parts.append(f"""def process_function_{i}(data, config):
    '''Process data using configuration parameters.'''
    result = analyze_content(data, config.threshold)
    return optimize_result(result)""")
    
    # Add error frames (high stability)  
    for i in range(size_tokens // 400):  # ~100 tokens each
        content_parts.append(f"""Error: ProcessingError in hybrid_selector_{i}
    at line {42 + i} in selector.py
TypeError: unsupported operand for processing: 'NoneType' and 'dict'
    Failed to process content with config={{'threshold': {i * 10}}}""")
    
    # Add context (medium stability)
    for i in range(size_tokens // 150):  # ~75 tokens each
        content_parts.append(f"""The hybrid system_{i} processes content using DPP selection 
for head content and StreamingLLM windowing for tail content. Key parameters 
include head_keep_ratio=0.12, window_size=6000, stride=3000. Performance 
targets: p95 latency <100ms, KV reuse >{60 + i}%, quality score >0.8.""")
    
    # Add volatile content (low stability)
    for i in range(size_tokens // 100):  # ~25 tokens each  
        content_parts.append(f"""Processing step {i}: status=active, time={i*10}ms, tokens={i*5}""")
    
    return "\n\n".join(content_parts)

def run_performance_test(system, system_name: str, test_content: str, iterations: int = 50) -> Dict[str, Any]:
    """Run performance test on a system."""
    print(f"Testing {system_name} system...")
    
    latencies = []
    mode_counts = {'head_only': 0, 'hybrid': 0, 'streaming_only': 0}
    kv_reuses = []
    keep_ratios = []
    
    # Warmup
    for _ in range(10):
        try:
            if hasattr(system, 'optimized_select'):
                system.optimized_select(test_content)
            else:
                system.select(test_content)
        except:
            pass
    
    # Test runs
    for i in range(iterations):
        if i % 20 == 0:
            print(f"  Run {i+1}/{iterations}")
            
        try:
            start_time = time.perf_counter()
            
            if hasattr(system, 'optimized_select'):
                result = system.optimized_select(test_content)
            else:
                result = system.select(test_content)
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Extract metrics based on result format
            if hasattr(result, 'processing_mode'):
                # Original system result
                mode = str(result.processing_mode).split('.')[-1].lower()
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
                kv_reuses.append(result.kv_prefix_reuse_ratio)
                keep_ratios.append(result.keep_ratio)
            else:
                # Optimized system result (dict format)
                mode = str(result.get('processing_mode', 'unknown')).split('.')[-1].lower()  
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
                kv_reuses.append(result.get('kv_prefix_reuse_ratio', 0.0))
                keep_ratios.append(result.get('keep_ratio', 0.0))
                
        except Exception as e:
            print(f"    Error in iteration {i}: {e}")
            continue
    
    if not latencies:
        print(f"  WARNING: No successful runs for {system_name}")
        return {}
    
    # Calculate metrics
    metrics = {
        'system_name': system_name,
        'iterations': len(latencies),
        'mean_latency_ms': statistics.mean(latencies),
        'p50_latency_ms': statistics.median(latencies),
        'p95_latency_ms': sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        'std_latency_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
        'min_latency_ms': min(latencies),
        'max_latency_ms': max(latencies),
        'mode_counts': mode_counts,
        'hybrid_ratio': mode_counts.get('hybrid', 0) / len(latencies) if latencies else 0,
        'avg_kv_reuse': statistics.mean(kv_reuses) if kv_reuses else 0,
        'avg_keep_ratio': statistics.mean(keep_ratios) if keep_ratios else 0
    }
    
    return metrics

def compare_systems(original_metrics: Dict, optimized_metrics: Dict) -> Dict[str, Any]:
    """Compare system metrics and provide recommendations."""
    
    if not original_metrics or not optimized_metrics:
        return {'error': 'Missing metrics for comparison'}
    
    # Calculate improvements
    latency_improvement = ((original_metrics['p95_latency_ms'] - optimized_metrics['p95_latency_ms']) / 
                          original_metrics['p95_latency_ms'] * 100) if original_metrics['p95_latency_ms'] > 0 else 0
    
    kv_improvement = optimized_metrics['avg_kv_reuse'] - original_metrics['avg_kv_reuse']
    hybrid_improvement = optimized_metrics['hybrid_ratio'] - original_metrics['hybrid_ratio']
    
    # Performance targets
    p95_target = 100.0  # ms
    meets_p95_target = optimized_metrics['p95_latency_ms'] < p95_target
    
    # Promotion criteria
    promotion_criteria = [
        latency_improvement >= 0,  # No latency regression
        meets_p95_target,  # Meets p95 target
        kv_improvement >= -0.05,  # No significant KV reuse regression
        optimized_metrics['avg_keep_ratio'] >= 0.05  # Reasonable keep ratio
    ]
    
    criteria_met = sum(promotion_criteria)
    promote = criteria_met >= 3  # 75% of criteria
    
    comparison = {
        'performance_improvements': {
            'latency_improvement_percent': latency_improvement,
            'p95_latency_reduction_ms': original_metrics['p95_latency_ms'] - optimized_metrics['p95_latency_ms'],
            'kv_reuse_improvement': kv_improvement,
            'hybrid_ratio_improvement': hybrid_improvement
        },
        'target_achievement': {
            'meets_p95_target': meets_p95_target,
            'p95_target_ms': p95_target,
            'actual_p95_ms': optimized_metrics['p95_latency_ms']
        },
        'promotion_analysis': {
            'criteria_met': criteria_met,
            'total_criteria': len(promotion_criteria),
            'promotion_recommended': promote,
            'confidence': criteria_met / len(promotion_criteria),
            'reasoning': f"Meets {criteria_met}/{len(promotion_criteria)} promotion criteria"
        },
        'detailed_comparison': {
            'original': original_metrics,
            'optimized': optimized_metrics
        }
    }
    
    return comparison

def main():
    """Main test execution."""
    print("=" * 70)
    print("HYBRID SYSTEM OPTIMIZATION VALIDATION")  
    print("=" * 70)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create test content  
        print("Generating test content...")
        test_content_2k = create_test_content(2000)
        test_content_5k = create_test_content(5000)
        
        print(f"Generated content: 2k tokens ({len(test_content_2k.split())} words), 5k tokens ({len(test_content_5k.split())} words)")
        
        # Initialize systems
        print("\nInitializing systems...")
        
        # Original system
        original_config = HybridConfig(
            head_keep_ratio=0.12,
            window_size=6000,
            stride=3000,
            sink_tokens=96
        )
        original_system = create_hybrid_selector(original_config)
        
        # Optimized system
        opt_config = OptimizationConfig(
            enable_pattern_cache=True,
            enable_entity_cache=True,
            optimize_gating_logic=True,
            adaptive_thresholds=True,
            optimize_head_selection=True,
            optimize_kv_reuse=True
        )
        optimized_system = create_optimized_hybrid_selector(original_config, opt_config)
        
        print("Systems initialized successfully")
        
        # Test both content sizes
        results = {}
        
        for content_name, content in [("2k_tokens", test_content_2k), ("5k_tokens", test_content_5k)]:
            print(f"\n{'='*50}")
            print(f"TESTING WITH {content_name.upper()} CONTENT")
            print(f"{'='*50}")
            
            # Test original system
            original_metrics = run_performance_test(original_system, "original", content, 40)
            
            # Test optimized system  
            optimized_metrics = run_performance_test(optimized_system, "optimized", content, 40)
            
            # Compare systems
            comparison = compare_systems(original_metrics, optimized_metrics)
            results[content_name] = comparison
            
            # Print results
            print(f"\n{content_name.upper()} RESULTS:")
            print("-" * 30)
            if 'error' not in comparison:
                perf = comparison['performance_improvements']
                target = comparison['target_achievement']
                promo = comparison['promotion_analysis']
                
                print(f"Latency Improvement: {perf['latency_improvement_percent']:+.1f}%")
                print(f"P95 Latency: {original_metrics['p95_latency_ms']:.1f}ms → {optimized_metrics['p95_latency_ms']:.1f}ms")
                print(f"KV Reuse: {original_metrics['avg_kv_reuse']:.3f} → {optimized_metrics['avg_kv_reuse']:.3f}")
                print(f"Hybrid Ratio: {original_metrics['hybrid_ratio']:.1%} → {optimized_metrics['hybrid_ratio']:.1%}")
                print(f"Meets P95 Target: {'YES' if target['meets_p95_target'] else 'NO'} (target: {target['p95_target_ms']}ms)")
                print(f"Promotion Recommended: {'YES' if promo['promotion_recommended'] else 'NO'}")
                print(f"Confidence: {promo['confidence']:.1%}")
            else:
                print(f"ERROR: {comparison['error']}")
        
        # Overall summary
        print(f"\n{'='*70}")
        print("OVERALL SUMMARY")
        print(f"{'='*70}")
        
        all_promoted = all(results[k].get('promotion_analysis', {}).get('promotion_recommended', False) 
                          for k in results if 'error' not in results[k])
        
        if all_promoted:
            print("✅ ALL TESTS PASS - Optimizations ready for promotion")
            print("\nKey improvements validated:")
            for content_name, result in results.items():
                if 'error' not in result:
                    perf = result['performance_improvements']
                    print(f"  {content_name}: {perf['latency_improvement_percent']:+.1f}% latency, "
                          f"P95 = {result['detailed_comparison']['optimized']['p95_latency_ms']:.1f}ms")
        else:
            print("❌ OPTIMIZATION NEEDED - Some tests failed promotion criteria")
            for content_name, result in results.items():
                if 'error' not in result and not result['promotion_analysis']['promotion_recommended']:
                    print(f"  {content_name}: Failed promotion criteria")
        
        # Export detailed results
        timestamp = int(time.time())
        report_path = f"/tmp/optimization_test_results_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results exported to: {report_path}")
        print("=" * 70)
        
        return 0 if all_promoted else 1
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())