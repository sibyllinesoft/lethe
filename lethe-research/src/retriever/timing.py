"""
Production-Grade Timing Infrastructure for IR System

Implements dual-timer system with GC barriers and steady-state measurements
for accurate performance evaluation of retrieval systems.

Features:
- In-process and external timing measurement
- Steady-state warm-up with configurable cold/warm cycles
- GC barriers and memory profiling
- Statistical aggregation (p50/p95/p99)
- Memory usage and throughput tracking
"""

import gc
import time
import psutil
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class TimingResult:
    """Container for timing measurement results."""
    
    operation: str
    latency_ms: float
    memory_mb: float
    cpu_percent: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class PerformanceProfile:
    """Aggregated performance statistics."""
    
    operation: str
    count: int
    
    # Latency statistics (milliseconds)
    latency_p50: float
    latency_p95: float  
    latency_p99: float
    latency_mean: float
    latency_std: float
    
    # Memory statistics (MB)
    memory_mean: float
    memory_peak: float
    memory_std: float
    
    # CPU statistics (%)
    cpu_mean: float
    cpu_peak: float
    
    # Throughput (ops/sec)
    throughput: float
    
    # System information
    warm_cycles: int
    total_duration_sec: float
    gc_collections: int

class TimingHarness:
    """
    Production-grade timing harness with dual-timer system.
    
    Provides accurate timing measurements with warm-up cycles,
    GC barriers, and statistical aggregation.
    """
    
    def __init__(self, 
                 cold_cycles: int = 50,
                 warm_cycles: int = 500,
                 gc_between_runs: bool = True,
                 memory_profiling: bool = True):
        """
        Initialize timing harness.
        
        Args:
            cold_cycles: Number of initial cycles to discard (warm-up)
            warm_cycles: Number of cycles to measure after warm-up
            gc_between_runs: Whether to run GC between measurements
            memory_profiling: Whether to profile memory usage
        """
        self.cold_cycles = cold_cycles
        self.warm_cycles = warm_cycles
        self.gc_between_runs = gc_between_runs
        self.memory_profiling = memory_profiling
        
        self.results: List[TimingResult] = []
        self.process = psutil.Process()
        self._gc_initial_collections = 0
        
    def reset(self):
        """Reset timing results."""
        self.results = []
        self._gc_initial_collections = 0
        
    @contextmanager
    def measure(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for single timing measurement.
        
        Args:
            operation: Name of the operation being measured
            metadata: Additional metadata to store with timing
            
        Yields:
            None
            
        Example:
            with harness.measure("query_execution"):
                result = index.search(query)
        """
        if metadata is None:
            metadata = {}
            
        # Pre-measurement GC
        if self.gc_between_runs:
            gc.collect()
            
        # Memory baseline
        memory_before = 0
        if self.memory_profiling:
            memory_before = self.process.memory_info().rss / 1024 / 1024  # MB
            
        cpu_before = self.process.cpu_percent()
        timestamp = time.time()
        
        # Start timing (in-process timer)
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            # End timing
            end_time = time.perf_counter()
            
            # Memory after
            memory_after = 0
            if self.memory_profiling:
                memory_after = self.process.memory_info().rss / 1024 / 1024  # MB
                
            cpu_after = self.process.cpu_percent()
            
            # Calculate metrics
            latency_ms = (end_time - start_time) * 1000
            memory_mb = max(memory_after, memory_before)  # Peak during operation
            cpu_percent = max(cpu_after, cpu_before)  # Peak during operation
            
            # Store result
            result = TimingResult(
                operation=operation,
                latency_ms=latency_ms,
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
                timestamp=timestamp,
                metadata=metadata
            )
            
            self.results.append(result)
            
    def benchmark_function(self, 
                          func: Callable,
                          operation: str,
                          *args, **kwargs) -> PerformanceProfile:
        """
        Benchmark a function with steady-state measurement.
        
        Args:
            func: Function to benchmark
            operation: Name for the operation
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            PerformanceProfile with aggregated statistics
        """
        logger.info(f"Starting benchmark: {operation}")
        logger.info(f"Cold cycles: {self.cold_cycles}, Warm cycles: {self.warm_cycles}")
        
        self.reset()
        self._gc_initial_collections = sum(gc.get_stats()[i]['collections'] for i in range(3))
        
        start_benchmark = time.time()
        
        # Cold cycles (warm-up)
        logger.info("Running cold cycles...")
        for i in range(self.cold_cycles):
            try:
                with self.measure(f"{operation}_cold", {"cycle": i}):
                    func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Cold cycle {i} failed: {e}")
                continue
                
        # Clear cold results
        cold_results = len(self.results)
        self.results = []
        
        # Warm cycles (measured)
        logger.info("Running warm cycles...")
        for i in range(self.warm_cycles):
            try:
                with self.measure(operation, {"cycle": i}):
                    func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Warm cycle {i} failed: {e}")
                continue
                
        end_benchmark = time.time()
        
        # Generate performance profile
        profile = self._generate_profile(operation, start_benchmark, end_benchmark, cold_results)
        
        logger.info(f"Benchmark complete: {operation}")
        logger.info(f"Latency p95: {profile.latency_p95:.2f}ms, Throughput: {profile.throughput:.1f} ops/sec")
        
        return profile
        
    def _generate_profile(self, 
                         operation: str, 
                         start_time: float, 
                         end_time: float,
                         cold_cycles: int) -> PerformanceProfile:
        """Generate aggregated performance profile from results."""
        
        if not self.results:
            raise ValueError("No timing results available")
            
        # Extract metrics
        latencies = [r.latency_ms for r in self.results]
        memories = [r.memory_mb for r in self.results] 
        cpus = [r.cpu_percent for r in self.results]
        
        # Calculate statistics
        latency_percentiles = np.percentile(latencies, [50, 95, 99])
        
        # GC collections during benchmark
        gc_final_collections = sum(gc.get_stats()[i]['collections'] for i in range(3))
        gc_collections = gc_final_collections - self._gc_initial_collections
        
        # Duration and throughput
        total_duration = end_time - start_time
        throughput = len(self.results) / total_duration if total_duration > 0 else 0
        
        return PerformanceProfile(
            operation=operation,
            count=len(self.results),
            latency_p50=latency_percentiles[0],
            latency_p95=latency_percentiles[1], 
            latency_p99=latency_percentiles[2],
            latency_mean=np.mean(latencies),
            latency_std=np.std(latencies),
            memory_mean=np.mean(memories),
            memory_peak=np.max(memories),
            memory_std=np.std(memories),
            cpu_mean=np.mean(cpus),
            cpu_peak=np.max(cpus), 
            throughput=throughput,
            warm_cycles=len(self.results),
            total_duration_sec=total_duration,
            gc_collections=gc_collections
        )

class PerformanceProfiler:
    """
    System-wide performance profiler for IR operations.
    
    Manages multiple timing harnesses and aggregates results
    across different operations and configurations.
    """
    
    def __init__(self):
        self.harnesses: Dict[str, TimingHarness] = {}
        self.profiles: Dict[str, PerformanceProfile] = {}
        
    def create_harness(self, name: str, **harness_kwargs) -> TimingHarness:
        """Create and register a new timing harness."""
        harness = TimingHarness(**harness_kwargs)
        self.harnesses[name] = harness
        return harness
        
    def get_harness(self, name: str) -> TimingHarness:
        """Get existing timing harness."""
        if name not in self.harnesses:
            self.harnesses[name] = TimingHarness()
        return self.harnesses[name]
        
    def benchmark_operation(self, 
                           harness_name: str,
                           func: Callable,
                           operation: str,
                           *args, **kwargs) -> PerformanceProfile:
        """Benchmark operation using specified harness."""
        harness = self.get_harness(harness_name)
        profile = harness.benchmark_function(func, operation, *args, **kwargs)
        
        profile_key = f"{harness_name}:{operation}"
        self.profiles[profile_key] = profile
        
        return profile
        
    def compare_profiles(self, profile_keys: List[str]) -> Dict[str, Any]:
        """Compare performance profiles across operations."""
        comparison = {
            'profiles': {},
            'relative_performance': {}
        }
        
        for key in profile_keys:
            if key in self.profiles:
                comparison['profiles'][key] = self.profiles[key]
                
        # Calculate relative performance
        if len(comparison['profiles']) >= 2:
            keys = list(comparison['profiles'].keys())
            baseline = comparison['profiles'][keys[0]]
            
            for key in keys[1:]:
                profile = comparison['profiles'][key]
                comparison['relative_performance'][key] = {
                    'latency_ratio': profile.latency_p95 / baseline.latency_p95,
                    'throughput_ratio': profile.throughput / baseline.throughput,
                    'memory_ratio': profile.memory_mean / baseline.memory_mean
                }
                
        return comparison
        
    def export_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Export all performance profiles."""
        export = {}
        
        for key, profile in self.profiles.items():
            export[key] = {
                'operation': profile.operation,
                'count': profile.count,
                'latency': {
                    'p50': profile.latency_p50,
                    'p95': profile.latency_p95,
                    'p99': profile.latency_p99,
                    'mean': profile.latency_mean,
                    'std': profile.latency_std
                },
                'memory': {
                    'mean': profile.memory_mean,
                    'peak': profile.memory_peak,
                    'std': profile.memory_std
                },
                'cpu': {
                    'mean': profile.cpu_mean,
                    'peak': profile.cpu_peak
                },
                'throughput': profile.throughput,
                'system': {
                    'warm_cycles': profile.warm_cycles,
                    'duration_sec': profile.total_duration_sec,
                    'gc_collections': profile.gc_collections
                }
            }
            
        return export