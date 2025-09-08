#!/usr/bin/env python3
"""
Performance Profiler for Lethe→StreamingLLM Hybrid System

Comprehensive profiling tool to identify latency bottlenecks and optimization opportunities.
Focuses on the critical performance issues identified in canary evaluation:

1. Latency Regression: p95 latency >100ms target  
2. Mode Selection Logic: Hybrid vs streaming decision efficiency
3. Quality Recovery: Performance degradation analysis
4. Computational Overhead: Real-time processing bottlenecks

Features:
- Micro-benchmarking with statistical significance
- Component-level latency breakdown
- Memory allocation profiling  
- Cache efficiency analysis
- Bottleneck identification with optimization recommendations
"""

import logging
import time
import sys
import cProfile
import pstats
import io
import tracemalloc
import statistics
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import json

# Import hybrid system components
try:
    from hybrid_selector import HybridSelector, HybridConfig, create_hybrid_selector
    from instrumentation import HybridInstrumentation, create_instrumentation
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)

@dataclass
class ProfiledOperation:
    """Result of a profiled operation."""
    operation_name: str
    execution_time_ms: float
    memory_delta_mb: float
    cpu_time_ms: float
    function_calls: int
    
    # Component breakdown
    component_times: Dict[str, float]
    memory_allocations: Dict[str, float]
    
    # Statistical measures
    samples_count: int
    mean_time_ms: float
    std_time_ms: float
    p50_time_ms: float
    p95_time_ms: float
    p99_time_ms: float

@dataclass  
class PerformanceBottleneck:
    """Identified performance bottleneck."""
    component: str
    severity: str  # "critical", "high", "medium", "low"
    time_percentage: float
    absolute_time_ms: float
    description: str
    optimization_recommendations: List[str]
    estimated_improvement: str

class ComponentTimer:
    """Context manager for timing components."""
    
    def __init__(self, profiler: 'PerformanceProfiler', component_name: str):
        self.profiler = profiler
        self.component_name = component_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed_ms = (time.perf_counter() - self.start_time) * 1000
            self.profiler.record_component_time(self.component_name, elapsed_ms)

class PerformanceProfiler:
    """Advanced performance profiler for hybrid system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Profiling configuration
        self.enable_memory_profiling = self.config.get('enable_memory_profiling', True)
        self.enable_cpu_profiling = self.config.get('enable_cpu_profiling', True) 
        self.sample_size = self.config.get('sample_size', 100)
        self.warmup_runs = self.config.get('warmup_runs', 10)
        
        # Results storage
        self.profiled_operations: List[ProfiledOperation] = []
        self.component_times: Dict[str, List[float]] = defaultdict(list)
        self.bottlenecks: List[PerformanceBottleneck] = []
        
        # Test data
        self.test_content = self._generate_test_content()
        
        logger.info("PerformanceProfiler initialized")
    
    def _generate_test_content(self) -> str:
        """Generate representative test content for profiling."""
        content_parts = [
            # Definitions
            """def hybrid_selector_main(content, config):
    '''Main hybrid selection function.'''
    atoms = extract_atoms(content)
    head = build_head(atoms, config.head_budget)
    tail = build_tail(remaining_content, config.tail_budget) 
    return arrange_final_content(head, tail)""",
            
            # Error frames  
            """Error: Failed to process hybrid selection
    at hybrid_selector.py:245 in select()
    at atom_extractor.py:156 in extract_atoms()
TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'
    tokens = head_tokens + tail_tokens
ValueError: invalid literal for int() with base 10: 'abc'""",
            
            # Tool references
            """@tool
def search_codebase(query: str) -> List[str]:
    '''Search through codebase for patterns.'''
    API_KEY = os.environ['SEARCH_API_KEY']
    endpoint = "https://search.api.com/v1/search"
    return tool_call(endpoint, query)""",
            
            # Code blocks and documentation
            """```python
class HybridSelector:
    '''Lethe→StreamingLLM hybrid selector implementation.
    
    Combines stable head selection using Lethe DPP with
    StreamingLLM windowed tail processing.
    '''
    
    def __init__(self, config: HybridConfig):
        self.config = config
        self.atom_extractor = AtomExtractor(config)
        self.head_builder = HeadBuilder(config)
        self.tail_builder = TailBuilder(config)
```""",
            
            # Context with entities
            """The hybrid system processes context by extracting atoms, then applies
DPP-based diversification for head selection. Key parameters include:
- head_keep_ratio: proportion for stable content (typically 0.12)
- window_size: StreamingLLM window size (6000 tokens)
- stride: sliding window stride (3000 tokens) 
- sink_tokens: attention sink allocation (96 tokens)

Performance targets: p95 latency <100ms, KV reuse >60%, ΔCBU/1k >10%""",
            
            # Volatile content (repeated patterns)
            """Processing step 1: atom extraction complete
Processing step 2: head building in progress  
Processing step 3: tail windowing active
Processing step 4: KV arrangement optimization
Processing step 5: final content assembly
Processing step 6: quality metrics calculation""" * 10
        ]
        
        return "\n\n".join(content_parts)
    
    def record_component_time(self, component: str, time_ms: float):
        """Record timing for a component."""
        self.component_times[component].append(time_ms)
    
    def profile_hybrid_selection(self, num_runs: Optional[int] = None) -> ProfiledOperation:
        """Profile complete hybrid selection process."""
        num_runs = num_runs or self.sample_size
        
        logger.info(f"Profiling hybrid selection ({num_runs} runs)...")
        
        # Create hybrid selector
        config = HybridConfig(
            head_keep_ratio=0.12,
            window_size=6000,
            stride=3000,
            sink_tokens=96,
            ce_k2=320,
            dpp_rank=14
        )
        selector = create_hybrid_selector(config)
        
        # Warmup runs
        logger.info(f"Warmup runs ({self.warmup_runs})...")
        for _ in range(self.warmup_runs):
            selector.select(self.test_content)
        
        # Profiling runs
        execution_times = []
        memory_deltas = []
        
        for run in range(num_runs):
            if run % 20 == 0:
                logger.info(f"Profiling run {run+1}/{num_runs}")
            
            # Memory profiling
            if self.enable_memory_profiling:
                tracemalloc.start()
                start_memory = tracemalloc.get_traced_memory()[0]
            
            # CPU profiling for detailed breakdown
            if self.enable_cpu_profiling and run == 0:  # Profile first run for details
                pr = cProfile.Profile()
                pr.enable()
            
            # Time the operation
            with self.timer('hybrid_selection'):
                start_time = time.perf_counter()
                
                with self.timer('content_extraction'):
                    # This will be captured by hybrid selector internally
                    pass
                
                result = selector.select(self.test_content)
                
                end_time = time.perf_counter()
            
            execution_time_ms = (end_time - start_time) * 1000
            execution_times.append(execution_time_ms)
            
            # Memory measurement
            memory_delta_mb = 0.0
            if self.enable_memory_profiling:
                end_memory = tracemalloc.get_traced_memory()[0] 
                memory_delta_mb = (end_memory - start_memory) / (1024 * 1024)
                memory_deltas.append(memory_delta_mb)
                tracemalloc.stop()
            
            # CPU profiling details
            if self.enable_cpu_profiling and run == 0:
                pr.disable()
                
                # Analyze CPU profile
                stats_stream = io.StringIO()
                ps = pstats.Stats(pr, stream=stats_stream)
                ps.sort_stats('cumulative')
                
                # Extract function call stats
                total_calls = ps.total_calls
                
                # Store detailed breakdown for first run
                self.component_times['total_profile'].append(execution_time_ms)
        
        # Calculate statistics
        mean_time = statistics.mean(execution_times)
        std_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
        p50_time = statistics.median(execution_times)
        p95_time = np.percentile(execution_times, 95)
        p99_time = np.percentile(execution_times, 99)
        
        mean_memory = statistics.mean(memory_deltas) if memory_deltas else 0.0
        
        # Component time analysis
        component_breakdown = {}
        for component, times in self.component_times.items():
            if times:
                component_breakdown[component] = {
                    'mean_ms': statistics.mean(times),
                    'p95_ms': np.percentile(times, 95),
                    'count': len(times)
                }
        
        operation = ProfiledOperation(
            operation_name="hybrid_selection",
            execution_time_ms=mean_time,
            memory_delta_mb=mean_memory,
            cpu_time_ms=mean_time,  # Simplified
            function_calls=total_calls if self.enable_cpu_profiling else 0,
            component_times=component_breakdown,
            memory_allocations={},  # Would extract from memory profiler
            samples_count=num_runs,
            mean_time_ms=mean_time,
            std_time_ms=std_time,
            p50_time_ms=p50_time,
            p95_time_ms=p95_time,
            p99_time_ms=p99_time
        )
        
        self.profiled_operations.append(operation)
        
        logger.info(f"Hybrid selection profiling complete:")
        logger.info(f"  Mean time: {mean_time:.2f}ms")
        logger.info(f"  P95 time: {p95_time:.2f}ms")
        logger.info(f"  P99 time: {p99_time:.2f}ms") 
        logger.info(f"  Memory usage: {mean_memory:.2f}MB")
        
        return operation
    
    def profile_component_breakdown(self) -> Dict[str, ProfiledOperation]:
        """Profile individual components for detailed breakdown."""
        logger.info("Profiling component breakdown...")
        
        components = {
            'atom_extraction': self._profile_atom_extraction,
            'head_building': self._profile_head_building, 
            'tail_building': self._profile_tail_building,
            'kv_arrangement': self._profile_kv_arrangement,
            'gating_decision': self._profile_gating_decision
        }
        
        component_profiles = {}
        
        for component_name, profile_func in components.items():
            logger.info(f"Profiling {component_name}...")
            profile = profile_func()
            component_profiles[component_name] = profile
            
            logger.info(f"  {component_name}: {profile.mean_time_ms:.2f}ms mean, "
                       f"{profile.p95_time_ms:.2f}ms p95")
        
        return component_profiles
    
    def _profile_atom_extraction(self) -> ProfiledOperation:
        """Profile atom extraction component."""
        from hybrid_selector import AtomExtractor
        
        config = HybridConfig()
        extractor = AtomExtractor(config)
        
        times = []
        for _ in range(self.sample_size):
            start_time = time.perf_counter()
            atoms = extractor.extract_atoms(self.test_content)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        
        return self._create_operation_profile("atom_extraction", times)
    
    def _profile_head_building(self) -> ProfiledOperation:
        """Profile head building component."""
        from hybrid_selector import HeadBuilder, AtomExtractor
        
        config = HybridConfig()
        extractor = AtomExtractor(config)
        head_builder = HeadBuilder(config)
        
        # Extract atoms once
        atoms = extractor.extract_atoms(self.test_content)
        budget = int(sum(atom.tokens for atom in atoms) * config.head_keep_ratio)
        
        times = []
        for _ in range(self.sample_size):
            start_time = time.perf_counter()
            head_selection = head_builder.build_head(atoms, budget)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        
        return self._create_operation_profile("head_building", times)
    
    def _profile_tail_building(self) -> ProfiledOperation:
        """Profile tail building component.""" 
        from hybrid_selector import TailBuilder
        
        config = HybridConfig()
        tail_builder = TailBuilder(config)
        
        # Use portion of content for tail
        remaining_content = self.test_content[1000:]  # Skip first 1000 chars
        head_digest = "HEAD_CONTEXT"
        budget = 12000
        
        times = []
        for _ in range(self.sample_size):
            start_time = time.perf_counter()
            tail_selection = tail_builder.build_tail(remaining_content, head_digest, budget)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        
        return self._create_operation_profile("tail_building", times)
    
    def _profile_kv_arrangement(self) -> ProfiledOperation:
        """Profile KV-aware arrangement component."""
        from hybrid_selector import HybridSelector, AtomExtractor, HeadBuilder, TailBuilder
        
        config = HybridConfig()
        selector = HybridSelector(config)
        
        # Get head and tail selections
        extractor = AtomExtractor(config)  
        atoms = extractor.extract_atoms(self.test_content)
        
        head_builder = HeadBuilder(config)
        head_budget = int(sum(atom.tokens for atom in atoms) * config.head_keep_ratio)
        head_selection = head_builder.build_head(atoms, head_budget)
        
        tail_builder = TailBuilder(config)
        remaining_content = self.test_content[1000:]
        tail_selection = tail_builder.build_tail(remaining_content, head_selection.head_digest, 12000)
        
        times = []
        for _ in range(self.sample_size):
            start_time = time.perf_counter()
            arrangement = selector._create_kv_aware_arrangement(head_selection, tail_selection)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        
        return self._create_operation_profile("kv_arrangement", times)
    
    def _profile_gating_decision(self) -> ProfiledOperation:
        """Profile gating decision logic."""
        from hybrid_selector import HybridSelector, AtomExtractor
        
        config = HybridConfig()
        selector = HybridSelector(config)
        extractor = AtomExtractor(config)
        
        # Extract atoms once
        atoms = extractor.extract_atoms(self.test_content)
        total_tokens = sum(atom.tokens for atom in atoms)
        
        times = []
        for _ in range(self.sample_size):
            start_time = time.perf_counter()
            gating_decision = selector._make_gating_decision(atoms, None, total_tokens)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        
        return self._create_operation_profile("gating_decision", times)
    
    def _create_operation_profile(self, name: str, times: List[float]) -> ProfiledOperation:
        """Create ProfiledOperation from timing data."""
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        p50_time = statistics.median(times)
        p95_time = np.percentile(times, 95)
        p99_time = np.percentile(times, 99)
        
        return ProfiledOperation(
            operation_name=name,
            execution_time_ms=mean_time,
            memory_delta_mb=0.0,  # Not measured for component profiles
            cpu_time_ms=mean_time,
            function_calls=0,
            component_times={},
            memory_allocations={},
            samples_count=len(times),
            mean_time_ms=mean_time,
            std_time_ms=std_time,
            p50_time_ms=p50_time,
            p95_time_ms=p95_time,
            p99_time_ms=p99_time
        )
    
    def timer(self, component_name: str) -> ComponentTimer:
        """Context manager for timing components."""
        return ComponentTimer(self, component_name)
    
    def identify_bottlenecks(self, operation: ProfiledOperation) -> List[PerformanceBottleneck]:
        """Identify performance bottlenecks from profiling data."""
        logger.info("Analyzing bottlenecks...")
        
        bottlenecks = []
        
        # Check overall latency
        if operation.p95_time_ms > 100:  # Target is <100ms p95
            severity = "critical" if operation.p95_time_ms > 200 else "high"
            bottlenecks.append(PerformanceBottleneck(
                component="overall_latency",
                severity=severity,
                time_percentage=100.0,
                absolute_time_ms=operation.p95_time_ms,
                description=f"P95 latency ({operation.p95_time_ms:.1f}ms) exceeds target (<100ms)",
                optimization_recommendations=[
                    "Implement caching for frequently accessed patterns",
                    "Optimize algorithm complexity in hot paths", 
                    "Consider parallel processing for independent operations",
                    "Profile memory allocations for optimization opportunities"
                ],
                estimated_improvement="20-40% latency reduction"
            ))
        
        # Analyze component breakdown
        total_component_time = sum(
            times.get('mean_ms', 0) for times in operation.component_times.values()
        )
        
        if total_component_time > 0:
            for component, times in operation.component_times.items():
                component_time = times.get('mean_ms', 0)
                percentage = (component_time / total_component_time) * 100
                
                if percentage > 30:  # Component taking >30% of time
                    severity = "high" if percentage > 50 else "medium"
                    
                    recommendations = self._get_component_recommendations(component)
                    
                    bottlenecks.append(PerformanceBottleneck(
                        component=component,
                        severity=severity,
                        time_percentage=percentage,
                        absolute_time_ms=component_time,
                        description=f"{component} takes {percentage:.1f}% of execution time",
                        optimization_recommendations=recommendations,
                        estimated_improvement=f"{int(percentage * 0.3)}-{int(percentage * 0.5)}% reduction"
                    ))
        
        # Memory usage analysis
        if operation.memory_delta_mb > 50:  # >50MB per operation
            bottlenecks.append(PerformanceBottleneck(
                component="memory_usage",
                severity="medium",
                time_percentage=0.0,
                absolute_time_ms=0.0,
                description=f"High memory usage ({operation.memory_delta_mb:.1f}MB per operation)",
                optimization_recommendations=[
                    "Implement object pooling for frequently allocated objects",
                    "Use memory-efficient data structures",
                    "Consider streaming processing for large datasets",
                    "Profile memory allocations with tracemalloc"
                ],
                estimated_improvement="30-50% memory reduction"
            ))
        
        # Statistical analysis 
        if operation.std_time_ms > operation.mean_time_ms * 0.5:  # High variance
            bottlenecks.append(PerformanceBottleneck(
                component="performance_consistency",
                severity="medium", 
                time_percentage=0.0,
                absolute_time_ms=operation.std_time_ms,
                description=f"High performance variance (σ={operation.std_time_ms:.1f}ms)",
                optimization_recommendations=[
                    "Investigate intermittent performance issues",
                    "Consider JIT warmup effects",
                    "Analyze system resource contention",
                    "Implement performance monitoring and alerting"
                ],
                estimated_improvement="More consistent performance"
            ))
        
        self.bottlenecks.extend(bottlenecks)
        
        logger.info(f"Identified {len(bottlenecks)} bottlenecks")
        for bottleneck in bottlenecks:
            logger.warning(f"  {bottleneck.severity.upper()}: {bottleneck.component} - {bottleneck.description}")
        
        return bottlenecks
    
    def _get_component_recommendations(self, component: str) -> List[str]:
        """Get optimization recommendations for specific components."""
        recommendations = {
            'atom_extraction': [
                "Cache regex patterns to avoid recompilation",
                "Use more efficient string splitting/parsing algorithms", 
                "Consider parallel processing for independent content blocks",
                "Optimize content type classification logic"
            ],
            'head_building': [
                "Implement caching for DPP kernel computations",
                "Optimize atom grouping and selection algorithms",
                "Use more efficient data structures for atom storage",
                "Consider approximate algorithms for large atom sets"
            ],
            'tail_building': [
                "Optimize window creation and sliding logic",
                "Cache attention sink computations",
                "Use more efficient entropy calculation methods",
                "Consider streaming window processing"
            ],
            'kv_arrangement': [
                "Cache KV prefix hash computations",
                "Optimize content arrangement algorithms",
                "Use more efficient string concatenation methods",
                "Consider lazy evaluation for arrangement"
            ],
            'gating_decision': [
                "Cache entity extraction and entropy calculations",
                "Optimize accept rate and entity entropy computations",
                "Use faster statistical computation methods",
                "Consider decision tree optimization"
            ]
        }
        
        return recommendations.get(component, [
            "Profile component in detail to identify specific bottlenecks",
            "Consider algorithmic optimizations",
            "Look for caching opportunities",
            "Evaluate data structure efficiency"
        ])
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        logger.info("Generating optimization report...")
        
        if not self.profiled_operations:
            return {"error": "No profiling data available"}
        
        main_operation = self.profiled_operations[0]  # hybrid_selection
        bottlenecks = self.identify_bottlenecks(main_operation)
        
        # Performance targets analysis
        target_p95_ms = 100.0
        current_p95_ms = main_operation.p95_time_ms
        p95_gap_ms = current_p95_ms - target_p95_ms
        
        # Priority analysis
        critical_bottlenecks = [b for b in bottlenecks if b.severity == "critical"]
        high_bottlenecks = [b for b in bottlenecks if b.severity == "high"]
        
        # Estimated improvement potential
        total_improvement_potential = sum(
            bottleneck.time_percentage * 0.4 for bottleneck in bottlenecks  # 40% improvement per bottleneck
        )
        
        report = {
            "summary": {
                "current_p95_latency_ms": current_p95_ms,
                "target_p95_latency_ms": target_p95_ms,
                "performance_gap_ms": p95_gap_ms,
                "gap_percentage": (p95_gap_ms / target_p95_ms) * 100,
                "critical_bottlenecks": len(critical_bottlenecks),
                "high_priority_bottlenecks": len(high_bottlenecks),
                "total_bottlenecks": len(bottlenecks),
                "estimated_improvement_potential": f"{total_improvement_potential:.1f}%"
            },
            "performance_metrics": {
                "mean_latency_ms": main_operation.mean_time_ms,
                "std_latency_ms": main_operation.std_time_ms,
                "p50_latency_ms": main_operation.p50_time_ms,
                "p95_latency_ms": main_operation.p95_time_ms,
                "p99_latency_ms": main_operation.p99_time_ms,
                "memory_usage_mb": main_operation.memory_delta_mb,
                "samples_count": main_operation.samples_count
            },
            "component_breakdown": {
                component: {
                    "mean_time_ms": times.get('mean_ms', 0),
                    "p95_time_ms": times.get('p95_ms', 0),
                    "time_percentage": (times.get('mean_ms', 0) / main_operation.mean_time_ms) * 100
                }
                for component, times in main_operation.component_times.items()
            },
            "bottlenecks": [
                {
                    "component": b.component,
                    "severity": b.severity,
                    "description": b.description,
                    "time_percentage": b.time_percentage,
                    "absolute_time_ms": b.absolute_time_ms,
                    "optimization_recommendations": b.optimization_recommendations,
                    "estimated_improvement": b.estimated_improvement
                }
                for b in sorted(bottlenecks, key=lambda x: x.time_percentage, reverse=True)
            ],
            "optimization_roadmap": self._generate_optimization_roadmap(bottlenecks),
            "next_steps": [
                "Implement highest-impact optimizations first",
                "Set up continuous performance monitoring",
                "Establish performance regression testing", 
                "Create performance budget and SLA tracking",
                "Re-run profiling after each optimization"
            ]
        }
        
        return report
    
    def _generate_optimization_roadmap(self, bottlenecks: List[PerformanceBottleneck]) -> Dict[str, List[str]]:
        """Generate optimization roadmap by priority."""
        roadmap = {
            "immediate": [],  # Critical issues
            "short_term": [], # High priority issues  
            "medium_term": [], # Medium priority issues
            "long_term": []   # Low priority and architectural changes
        }
        
        for bottleneck in bottlenecks:
            recommendations = bottleneck.optimization_recommendations
            
            if bottleneck.severity == "critical":
                roadmap["immediate"].extend([
                    f"{bottleneck.component}: {rec}" for rec in recommendations[:2]
                ])
            elif bottleneck.severity == "high":
                roadmap["short_term"].extend([
                    f"{bottleneck.component}: {rec}" for rec in recommendations[:2] 
                ])
            elif bottleneck.severity == "medium":
                roadmap["medium_term"].extend([
                    f"{bottleneck.component}: {rec}" for rec in recommendations[:1]
                ])
            else:
                roadmap["long_term"].extend([
                    f"{bottleneck.component}: {rec}" for rec in recommendations[:1]
                ])
        
        return roadmap
    
    def export_report(self, filepath: Optional[str] = None) -> str:
        """Export optimization report to file."""
        report = self.generate_optimization_report()
        
        if not filepath:
            timestamp = int(time.time())
            filepath = f"/tmp/performance_optimization_report_{timestamp}.json"
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Performance report exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            raise

def main():
    """Main profiling entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile Lethe→StreamingLLM Hybrid Performance")
    parser.add_argument('--samples', type=int, default=100, help='Number of samples for profiling')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup runs')
    parser.add_argument('--output', type=str, help='Output file path for report')
    parser.add_argument('--components', action='store_true', help='Profile individual components')
    parser.add_argument('--memory', action='store_true', help='Enable memory profiling')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize profiler
        config = {
            'sample_size': args.samples,
            'warmup_runs': args.warmup,
            'enable_memory_profiling': args.memory,
            'enable_cpu_profiling': True
        }
        
        profiler = PerformanceProfiler(config)
        
        print("=" * 80)
        print("LETHE→STREAMINGLM HYBRID PERFORMANCE PROFILER")
        print("=" * 80)
        print(f"Sample size: {args.samples}")
        print(f"Warmup runs: {args.warmup}")
        print(f"Memory profiling: {args.memory}")
        print(f"Component profiling: {args.components}")
        print()
        
        # Profile main operation
        print("Profiling hybrid selection...")
        main_profile = profiler.profile_hybrid_selection()
        
        # Profile components if requested
        component_profiles = {}
        if args.components:
            print("Profiling individual components...")
            component_profiles = profiler.profile_component_breakdown()
        
        # Generate report
        print("Generating optimization report...")
        report = profiler.generate_optimization_report()
        
        # Export report
        output_path = args.output or f"/tmp/performance_report_{int(time.time())}.json"
        profiler.export_report(output_path)
        
        # Print summary
        print("\n" + "=" * 80)
        print("PERFORMANCE ANALYSIS SUMMARY")  
        print("=" * 80)
        
        summary = report["summary"]
        print(f"Current P95 Latency: {summary['current_p95_latency_ms']:.1f}ms")
        print(f"Target P95 Latency: {summary['target_p95_latency_ms']:.1f}ms")
        print(f"Performance Gap: {summary['performance_gap_ms']:.1f}ms ({summary['gap_percentage']:.1f}%)")
        print(f"Critical Bottlenecks: {summary['critical_bottlenecks']}")
        print(f"High Priority Bottlenecks: {summary['high_priority_bottlenecks']}")
        print(f"Estimated Improvement Potential: {summary['estimated_improvement_potential']}")
        
        print("\nTop Bottlenecks:")
        for bottleneck in report["bottlenecks"][:3]:
            print(f"  • {bottleneck['component']}: {bottleneck['description']}")
            print(f"    Severity: {bottleneck['severity']}, Time: {bottleneck['time_percentage']:.1f}%")
        
        print(f"\nFull report exported to: {output_path}")
        print("=" * 80)
        
        # Exit with status based on performance gap
        if summary['performance_gap_ms'] > 50:  # >50ms gap is critical
            print("❌ CRITICAL: Performance gap >50ms - immediate optimization required")
            sys.exit(1)
        elif summary['performance_gap_ms'] > 10:  # >10ms gap needs attention
            print("⚠️  WARNING: Performance gap >10ms - optimization recommended") 
            sys.exit(2)
        else:
            print("✅ Performance within acceptable range")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\nProfiling interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Profiling failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()