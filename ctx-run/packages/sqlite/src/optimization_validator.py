#!/usr/bin/env python3
"""
Optimization Validation and Benchmarking Suite

Validates hybrid system optimizations against performance targets with realistic workloads.
Tests the critical issues from canary evaluation:

1. Latency Regression: Measure p95 latency improvements
2. Mode Selection Logic: Validate improved gating decisions  
3. Quality Recovery: Compare before/after quality metrics
4. Performance Targets: Validate against <100ms p95 target

Features:
- Realistic content generation matching production patterns
- Before/after optimization comparison with statistical significance
- InfiniteBench-style evaluation with quality metrics
- Continuous performance validation
- Automated promotion/regression detection
"""

import logging
import time
import sys
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass, field
import numpy as np
import json
from collections import defaultdict, deque

# Import both original and optimized systems
try:
    from hybrid_selector import HybridSelector, HybridConfig, create_hybrid_selector
    from hybrid_optimizations import (
        HybridOptimizerSystem, OptimizationConfig, 
        create_optimized_hybrid_selector
    )
    from instrumentation import HybridInstrumentation, create_instrumentation
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)

@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics."""
    
    # Performance metrics
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    std_latency_ms: float
    
    # Throughput metrics
    operations_per_second: float
    total_operations: int
    total_time_seconds: float
    
    # Quality metrics
    hybrid_mode_ratio: float
    average_keep_ratio: float
    average_kv_reuse: float
    average_head_tokens: float
    average_tail_tokens: float
    
    # System metrics
    memory_usage_mb: float
    cpu_utilization_percent: float
    error_rate: float

@dataclass
class ComparisonResult:
    """Result of comparing original vs optimized system."""
    
    original_metrics: ValidationMetrics
    optimized_metrics: ValidationMetrics
    
    # Improvement calculations
    latency_improvement_percent: float
    throughput_improvement_percent: float
    quality_improvement_percent: float
    
    # Statistical significance
    latency_p_value: float
    statistically_significant: bool
    
    # Target achievement
    meets_p95_target: bool
    meets_throughput_target: bool
    meets_quality_target: bool
    
    # Recommendation
    promotion_recommended: bool
    confidence_score: float
    recommendation_reason: str

class RealisticContentGenerator:
    """Generate realistic content patterns for validation."""
    
    def __init__(self):
        self.code_patterns = [
            "def {name}(self, {params}):\n    '''{docstring}'''\n    {body}\n    return {result}",
            "class {name}:\n    '''{docstring}'''\n    def __init__(self, {params}):\n        {body}",
            "async def {name}({params}):\n    '''{docstring}'''\n    try:\n        {body}\n    except Exception as err:\n        logger.error(f'Error: {{err}}')",
        ]
        
        self.error_patterns = [
            "Error: {error_type} in {location}\n    at {file}:{line}\n{message}",
            "Exception: {error_type}\nTraceback (most recent call last):\n  File '{file}', line {line}\n{message}",
            "TypeError: {message}\n    in function {function}\n    expected {expected}, got {actual}",
        ]
        
        self.context_patterns = [
            "The {system} processes {content_type} using {method} for {purpose}. Key parameters include {params}.",
            "Implementation details: {details}. Performance targets: {targets}. Current status: {status}.",
            "Configuration settings: {settings}. Optimization features: {features}. Known limitations: {limitations}.",
        ]
        
        self.entity_pools = {
            'names': ['process_content', 'hybrid_selector', 'atom_extractor', 'head_builder', 'tail_builder'],
            'params': ['config', 'tokens', 'budget', 'threshold', 'window_size', 'stride'],
            'errors': ['ValueError', 'TypeError', 'RuntimeError', 'ConnectionError', 'TimeoutError'],
            'systems': ['hybrid system', 'streaming processor', 'context manager', 'cache optimizer'],
            'methods': ['DPP selection', 'windowing', 'attention sinks', 'KV caching', 'entropy analysis']
        }
    
    def generate_content(self, target_tokens: int, content_mix: Dict[str, float] = None) -> str:
        """Generate realistic content with specified token count and content mix."""
        content_mix = content_mix or {
            'code': 0.3,
            'errors': 0.2, 
            'context': 0.4,
            'volatile': 0.1
        }
        
        sections = []
        current_tokens = 0
        target_per_type = {k: int(target_tokens * v) for k, v in content_mix.items()}
        
        # Generate code sections
        for _ in range(target_per_type['code'] // 100):  # ~100 tokens per code section
            code_content = self._generate_code_section()
            sections.append(code_content)
            current_tokens += len(code_content.split())
        
        # Generate error sections  
        for _ in range(target_per_type['errors'] // 50):  # ~50 tokens per error
            error_content = self._generate_error_section()
            sections.append(error_content)
            current_tokens += len(error_content.split())
        
        # Generate context sections
        for _ in range(target_per_type['context'] // 80):  # ~80 tokens per context
            context_content = self._generate_context_section()
            sections.append(context_content)
            current_tokens += len(context_content.split())
        
        # Add volatile content to reach target
        while current_tokens < target_tokens * 0.95:  # Within 5% of target
            volatile_content = self._generate_volatile_section()
            sections.append(volatile_content)
            current_tokens += len(volatile_content.split())
        
        return "\n\n".join(sections)
    
    def _generate_code_section(self) -> str:
        """Generate code section."""
        import random
        
        pattern = random.choice(self.code_patterns)
        name = random.choice(self.entity_pools['names'])
        params = ', '.join(random.sample(self.entity_pools['params'], 2))
        body = f"result = process_{random.choice(self.entity_pools['methods']).replace(' ', '_')}({random.choice(self.entity_pools['params'])})"
        result = random.choice(self.entity_pools['params'])
        
        docstring = f"Process {name} with {params}"
        
        return pattern.format(
            name=name,
            params=params,
            docstring=docstring,
            body=body,
            result=result
        )
    
    def _generate_error_section(self) -> str:
        """Generate error section."""
        import random
        
        pattern = random.choice(self.error_patterns)
        error_type = random.choice(self.entity_pools['errors'])
        location = random.choice(self.entity_pools['names'])
        file = f"{random.choice(self.entity_pools['names'])}.py"
        line = random.randint(10, 500)
        message = f"Failed to {random.choice(self.entity_pools['methods'])} due to invalid {random.choice(self.entity_pools['params'])}"
        
        return pattern.format(
            error_type=error_type,
            location=location,
            file=file,
            line=line,
            message=message,
            function=random.choice(self.entity_pools['names']),
            expected='str',
            actual='NoneType'
        )
    
    def _generate_context_section(self) -> str:
        """Generate context section."""
        import random
        
        pattern = random.choice(self.context_patterns)
        system = random.choice(self.entity_pools['systems'])
        content_type = 'hybrid content'
        method = random.choice(self.entity_pools['methods'])
        purpose = 'optimal performance'
        params = ', '.join([f"{p}={random.randint(1, 1000)}" for p in random.sample(self.entity_pools['params'], 3)])
        
        return pattern.format(
            system=system,
            content_type=content_type,
            method=method,
            purpose=purpose,
            params=params,
            details=f"Uses {method} with {random.choice(self.entity_pools['params'])} optimization",
            targets=f"p95 <100ms, KV reuse >60%, tokens <{random.randint(1000, 10000)}",
            status=random.choice(['active', 'optimizing', 'monitoring', 'stable']),
            settings=params,
            features=', '.join(random.sample(['caching', 'windowing', 'selection', 'arrangement'], 2)),
            limitations=f"Maximum {random.choice(self.entity_pools['params'])} of {random.randint(100, 5000)}"
        )
    
    def _generate_volatile_section(self) -> str:
        """Generate volatile/repetitive content."""
        import random
        
        base_text = f"Processing step {random.randint(1, 20)}: {random.choice(self.entity_pools['methods'])} in progress"
        variations = [
            f"{base_text} with {random.choice(self.entity_pools['params'])}={random.randint(1, 100)}",
            f"{base_text} - status: {random.choice(['active', 'pending', 'complete'])}",
            f"{base_text} - time: {random.randint(1, 1000)}ms",
        ]
        
        return "\n".join([random.choice(variations) for _ in range(random.randint(3, 8))])

class ValidationTestSuite:
    """Comprehensive validation test suite."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Test configuration
        self.test_iterations = self.config.get('test_iterations', 100)
        self.warmup_iterations = self.config.get('warmup_iterations', 20)
        self.content_sizes = self.config.get('content_sizes', [1000, 5000, 10000, 20000])
        self.performance_target_p95_ms = self.config.get('performance_target_p95_ms', 100.0)
        
        # Content generator
        self.content_generator = RealisticContentGenerator()
        
        # Systems under test
        self.original_system = None
        self.optimized_system = None
        
        # Instrumentation
        self.instrumentation = create_instrumentation()
        
        logger.info("ValidationTestSuite initialized")
    
    def initialize_systems(self):
        """Initialize both original and optimized systems."""
        logger.info("Initializing systems for comparison...")
        
        # Create original system
        original_config = HybridConfig(
            head_keep_ratio=0.12,
            window_size=6000,
            stride=3000,
            sink_tokens=96,
            ce_k2=320,
            dpp_rank=14
        )
        self.original_system = create_hybrid_selector(original_config)
        
        # Create optimized system
        optimization_config = OptimizationConfig(
            enable_pattern_cache=True,
            enable_entity_cache=True,
            enable_kv_hash_cache=True,
            optimize_gating_logic=True,
            adaptive_thresholds=True,
            optimize_head_selection=True,
            optimize_tail_windowing=True,
            optimize_kv_reuse=True
        )
        self.optimized_system = create_optimized_hybrid_selector(
            original_config, optimization_config
        )
        
        logger.info("Systems initialized successfully")
    
    def run_comprehensive_validation(self) -> ComparisonResult:
        """Run comprehensive validation comparing original vs optimized."""
        logger.info("Starting comprehensive validation...")
        
        self.initialize_systems()
        
        # Test original system
        logger.info("Testing original system...")
        original_metrics = self._test_system(self.original_system, "original")
        
        # Test optimized system
        logger.info("Testing optimized system...")  
        optimized_metrics = self._test_system(self.optimized_system, "optimized")
        
        # Compare results
        comparison = self._compare_systems(original_metrics, optimized_metrics)
        
        logger.info("Comprehensive validation completed")
        return comparison
    
    def _test_system(self, system, system_name: str) -> ValidationMetrics:
        """Test a system with comprehensive metrics collection."""
        logger.info(f"Testing {system_name} system...")
        
        all_latencies = []
        all_throughputs = []
        mode_counts = defaultdict(int)
        keep_ratios = []
        kv_reuses = []
        head_tokens = []
        tail_tokens = []
        errors = 0
        
        total_start_time = time.perf_counter()
        
        # Test across different content sizes
        for content_size in self.content_sizes:
            logger.info(f"  Testing with {content_size} token content...")
            
            # Generate test content
            test_content = self.content_generator.generate_content(
                content_size,
                {'code': 0.25, 'errors': 0.15, 'context': 0.45, 'volatile': 0.15}  # Mix for hybrid mode
            )
            
            # Warmup runs
            for _ in range(self.warmup_iterations):
                try:
                    if hasattr(system, 'optimized_select'):
                        system.optimized_select(test_content)
                    else:
                        system.select(test_content)
                except Exception:
                    pass
            
            # Actual test runs
            size_latencies = []
            size_throughputs = []
            
            for iteration in range(self.test_iterations):
                try:
                    start_time = time.perf_counter()
                    
                    if hasattr(system, 'optimized_select'):
                        result = system.optimized_select(test_content)
                    else:
                        result = system.select(test_content)
                    
                    end_time = time.perf_counter()
                    
                    # Collect metrics
                    latency_ms = (end_time - start_time) * 1000
                    throughput = len(test_content.split()) / (end_time - start_time)  # tokens/sec
                    
                    size_latencies.append(latency_ms)
                    size_throughputs.append(throughput)
                    
                    # Extract result metrics
                    if hasattr(result, 'processing_mode'):
                        mode_counts[result.processing_mode] += 1
                        keep_ratios.append(result.keep_ratio)
                        kv_reuses.append(result.kv_prefix_reuse_ratio)
                        
                        if result.head_selection:
                            head_tokens.append(result.head_selection.total_tokens)
                        if result.tail_selection:
                            tail_tokens.append(result.tail_selection.total_tokens)
                    else:
                        # Handle optimized system result format
                        processing_mode = result.get('processing_mode', 'unknown')
                        mode_counts[processing_mode] += 1
                        keep_ratios.append(result.get('keep_ratio', 0.0))
                        kv_reuses.append(result.get('kv_prefix_reuse_ratio', 0.0))
                        
                        if result.get('head_selection'):
                            head_tokens.append(result['head_selection'].total_tokens)
                        if result.get('tail_selection'):
                            tail_tokens.append(result['tail_selection'].total_tokens)
                    
                except Exception as e:
                    logger.error(f"Error in {system_name} test iteration: {e}")
                    errors += 1
                    continue
            
            all_latencies.extend(size_latencies)
            all_throughputs.extend(size_throughputs)
        
        total_end_time = time.perf_counter()
        total_time = total_end_time - total_start_time
        
        # Calculate comprehensive metrics
        if all_latencies:
            mean_latency = statistics.mean(all_latencies)
            p50_latency = statistics.median(all_latencies)
            p95_latency = np.percentile(all_latencies, 95)
            p99_latency = np.percentile(all_latencies, 99)
            std_latency = statistics.stdev(all_latencies) if len(all_latencies) > 1 else 0.0
        else:
            mean_latency = p50_latency = p95_latency = p99_latency = std_latency = 0.0
        
        if all_throughputs:
            mean_throughput = statistics.mean(all_throughputs)
        else:
            mean_throughput = 0.0
        
        total_operations = len(all_latencies) + errors
        hybrid_ratio = mode_counts.get('hybrid', 0) / max(1, total_operations)
        
        metrics = ValidationMetrics(
            mean_latency_ms=mean_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            std_latency_ms=std_latency,
            operations_per_second=total_operations / total_time if total_time > 0 else 0.0,
            total_operations=total_operations,
            total_time_seconds=total_time,
            hybrid_mode_ratio=hybrid_ratio,
            average_keep_ratio=statistics.mean(keep_ratios) if keep_ratios else 0.0,
            average_kv_reuse=statistics.mean(kv_reuses) if kv_reuses else 0.0,
            average_head_tokens=statistics.mean(head_tokens) if head_tokens else 0.0,
            average_tail_tokens=statistics.mean(tail_tokens) if tail_tokens else 0.0,
            memory_usage_mb=0.0,  # Would measure with tracemalloc
            cpu_utilization_percent=0.0,  # Would measure with psutil
            error_rate=errors / max(1, total_operations)
        )
        
        logger.info(f"{system_name.capitalize()} system metrics:")
        logger.info(f"  Mean latency: {mean_latency:.2f}ms")
        logger.info(f"  P95 latency: {p95_latency:.2f}ms")
        logger.info(f"  Throughput: {mean_throughput:.1f} tokens/sec")
        logger.info(f"  Hybrid ratio: {hybrid_ratio:.1%}")
        logger.info(f"  Error rate: {errors / max(1, total_operations):.1%}")
        
        return metrics
    
    def _compare_systems(self, original: ValidationMetrics, 
                        optimized: ValidationMetrics) -> ComparisonResult:
        """Compare original vs optimized system metrics."""
        logger.info("Comparing system performance...")
        
        # Calculate improvements
        latency_improvement = ((original.p95_latency_ms - optimized.p95_latency_ms) / 
                              original.p95_latency_ms * 100) if original.p95_latency_ms > 0 else 0.0
        
        throughput_improvement = ((optimized.operations_per_second - original.operations_per_second) / 
                                 original.operations_per_second * 100) if original.operations_per_second > 0 else 0.0
        
        # Simple quality score based on multiple factors
        original_quality = (original.hybrid_mode_ratio * 0.3 + 
                           original.average_kv_reuse * 0.4 + 
                           (1 - original.error_rate) * 0.3)
        optimized_quality = (optimized.hybrid_mode_ratio * 0.3 + 
                            optimized.average_kv_reuse * 0.4 + 
                            (1 - optimized.error_rate) * 0.3)
        
        quality_improvement = ((optimized_quality - original_quality) / 
                              original_quality * 100) if original_quality > 0 else 0.0
        
        # Statistical significance (simplified)
        # In practice, would use proper statistical tests
        latency_diff = abs(original.p95_latency_ms - optimized.p95_latency_ms)
        latency_combined_std = (original.std_latency_ms + optimized.std_latency_ms) / 2
        statistically_significant = latency_diff > (2 * latency_combined_std)  # Rough 2-sigma test
        
        # Target achievement
        meets_p95_target = optimized.p95_latency_ms < self.performance_target_p95_ms
        meets_throughput_target = throughput_improvement > 5.0  # >5% improvement
        meets_quality_target = quality_improvement > 2.0  # >2% improvement
        
        # Promotion decision
        promotion_criteria = [
            meets_p95_target,
            latency_improvement > 0,  # Any latency improvement
            optimized.error_rate <= original.error_rate,  # No regression in errors
            quality_improvement > -5.0  # No significant quality regression
        ]
        
        criteria_met = sum(promotion_criteria)
        confidence_score = criteria_met / len(promotion_criteria)
        promotion_recommended = confidence_score >= 0.75  # 75% of criteria
        
        # Recommendation reasoning
        if promotion_recommended:
            recommendation_reason = f"Meets {criteria_met}/{len(promotion_criteria)} criteria. "
            if latency_improvement > 10:
                recommendation_reason += "Significant latency improvement. "
            if meets_p95_target:
                recommendation_reason += "Achieves p95 target. "
            recommendation_reason += "Ready for promotion."
        else:
            recommendation_reason = f"Meets only {criteria_met}/{len(promotion_criteria)} criteria. "
            if not meets_p95_target:
                recommendation_reason += f"P95 latency ({optimized.p95_latency_ms:.1f}ms) exceeds target ({self.performance_target_p95_ms}ms). "
            if latency_improvement < 0:
                recommendation_reason += "Latency regression detected. "
            if quality_improvement < -5:
                recommendation_reason += "Quality regression detected. "
            recommendation_reason += "Additional optimization needed."
        
        comparison = ComparisonResult(
            original_metrics=original,
            optimized_metrics=optimized,
            latency_improvement_percent=latency_improvement,
            throughput_improvement_percent=throughput_improvement,
            quality_improvement_percent=quality_improvement,
            latency_p_value=0.05 if statistically_significant else 0.1,  # Simplified
            statistically_significant=statistically_significant,
            meets_p95_target=meets_p95_target,
            meets_throughput_target=meets_throughput_target,
            meets_quality_target=meets_quality_target,
            promotion_recommended=promotion_recommended,
            confidence_score=confidence_score,
            recommendation_reason=recommendation_reason
        )
        
        return comparison
    
    def generate_validation_report(self, comparison: ComparisonResult) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        logger.info("Generating validation report...")
        
        report = {
            "validation_summary": {
                "promotion_recommended": comparison.promotion_recommended,
                "confidence_score": comparison.confidence_score,
                "recommendation_reason": comparison.recommendation_reason,
                "meets_performance_targets": {
                    "p95_latency": comparison.meets_p95_target,
                    "throughput": comparison.meets_throughput_target,
                    "quality": comparison.meets_quality_target
                }
            },
            "performance_comparison": {
                "latency_improvement_percent": comparison.latency_improvement_percent,
                "throughput_improvement_percent": comparison.throughput_improvement_percent,
                "quality_improvement_percent": comparison.quality_improvement_percent,
                "statistically_significant": comparison.statistically_significant
            },
            "original_system_metrics": {
                "p95_latency_ms": comparison.original_metrics.p95_latency_ms,
                "mean_latency_ms": comparison.original_metrics.mean_latency_ms,
                "operations_per_second": comparison.original_metrics.operations_per_second,
                "hybrid_mode_ratio": comparison.original_metrics.hybrid_mode_ratio,
                "average_kv_reuse": comparison.original_metrics.average_kv_reuse,
                "error_rate": comparison.original_metrics.error_rate
            },
            "optimized_system_metrics": {
                "p95_latency_ms": comparison.optimized_metrics.p95_latency_ms,
                "mean_latency_ms": comparison.optimized_metrics.mean_latency_ms,
                "operations_per_second": comparison.optimized_metrics.operations_per_second,
                "hybrid_mode_ratio": comparison.optimized_metrics.hybrid_mode_ratio,
                "average_kv_reuse": comparison.optimized_metrics.average_kv_reuse,
                "error_rate": comparison.optimized_metrics.error_rate
            },
            "optimization_impact": {
                "absolute_latency_reduction_ms": (comparison.original_metrics.p95_latency_ms - 
                                                comparison.optimized_metrics.p95_latency_ms),
                "absolute_throughput_increase": (comparison.optimized_metrics.operations_per_second - 
                                               comparison.original_metrics.operations_per_second),
                "kv_reuse_improvement": (comparison.optimized_metrics.average_kv_reuse - 
                                       comparison.original_metrics.average_kv_reuse),
                "hybrid_mode_improvement": (comparison.optimized_metrics.hybrid_mode_ratio - 
                                          comparison.original_metrics.hybrid_mode_ratio)
            },
            "test_configuration": {
                "test_iterations": self.test_iterations,
                "content_sizes_tested": self.content_sizes,
                "performance_target_p95_ms": self.performance_target_p95_ms,
                "total_operations": (comparison.original_metrics.total_operations + 
                                   comparison.optimized_metrics.total_operations)
            }
        }
        
        return report
    
    def export_report(self, report: Dict[str, Any], filepath: Optional[str] = None) -> str:
        """Export validation report to file."""
        if not filepath:
            timestamp = int(time.time())
            filepath = f"/tmp/optimization_validation_report_{timestamp}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Validation report exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            raise

def main():
    """Main validation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Hybrid System Optimizations")
    parser.add_argument('--iterations', type=int, default=100, help='Test iterations per content size')
    parser.add_argument('--warmup', type=int, default=20, help='Warmup iterations')
    parser.add_argument('--sizes', nargs='+', type=int, default=[1000, 5000, 10000, 20000], 
                       help='Content sizes to test')
    parser.add_argument('--target-p95', type=float, default=100.0, help='P95 latency target in ms')
    parser.add_argument('--output', type=str, help='Output file path for report')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize validation suite
        config = {
            'test_iterations': args.iterations,
            'warmup_iterations': args.warmup,
            'content_sizes': args.sizes,
            'performance_target_p95_ms': args.target_p95
        }
        
        validator = ValidationTestSuite(config)
        
        print("=" * 80)
        print("HYBRID SYSTEM OPTIMIZATION VALIDATION")
        print("=" * 80)
        print(f"Test iterations: {args.iterations}")
        print(f"Content sizes: {args.sizes}")
        print(f"P95 target: {args.target_p95}ms")
        print()
        
        # Run comprehensive validation
        print("Running comprehensive validation...")
        comparison_result = validator.run_comprehensive_validation()
        
        # Generate report
        print("Generating validation report...")
        report = validator.generate_validation_report(comparison_result)
        
        # Export report
        output_path = args.output or f"/tmp/optimization_validation_{int(time.time())}.json"
        validator.export_report(report, output_path)
        
        # Print summary
        print("\n" + "=" * 80)
        print("OPTIMIZATION VALIDATION SUMMARY")
        print("=" * 80)
        
        summary = report["validation_summary"]
        print(f"Promotion Recommended: {'YES' if summary['promotion_recommended'] else 'NO'}")
        print(f"Confidence Score: {summary['confidence_score']:.1%}")
        print(f"P95 Target Met: {'YES' if summary['meets_performance_targets']['p95_latency'] else 'NO'}")
        
        performance = report["performance_comparison"]
        print(f"\nPerformance Improvements:")
        print(f"  Latency: {performance['latency_improvement_percent']:+.1f}%")
        print(f"  Throughput: {performance['throughput_improvement_percent']:+.1f}%")
        print(f"  Quality: {performance['quality_improvement_percent']:+.1f}%")
        
        original = report["original_system_metrics"]
        optimized = report["optimized_system_metrics"]
        print(f"\nOriginal vs Optimized:")
        print(f"  P95 Latency: {original['p95_latency_ms']:.1f}ms → {optimized['p95_latency_ms']:.1f}ms")
        print(f"  Throughput: {original['operations_per_second']:.1f} → {optimized['operations_per_second']:.1f} ops/sec")
        print(f"  KV Reuse: {original['average_kv_reuse']:.1%} → {optimized['average_kv_reuse']:.1%}")
        print(f"  Hybrid Mode: {original['hybrid_mode_ratio']:.1%} → {optimized['hybrid_mode_ratio']:.1%}")
        
        print(f"\nRecommendation: {summary['recommendation_reason']}")
        print(f"Full report: {output_path}")
        print("=" * 80)
        
        # Exit with status based on recommendation
        if summary['promotion_recommended']:
            print("✅ Optimizations validated - ready for promotion")
            sys.exit(0)
        else:
            print("❌ Optimizations need improvement - not ready for promotion")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()