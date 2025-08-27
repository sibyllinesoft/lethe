"""
Common Timing Utilities

Provides standardized timing measurement patterns to eliminate redundant timing code
throughout the codebase. Builds on the existing TimingHarness but adds simple utilities
for common use cases.

Usage:
    from common.timing import TimingContext, measure_latency, LatencyTracker

    # Simple context manager for timing
    with TimingContext("operation") as timer:
        # do work
        pass
    print(f"Latency: {timer.latency_ms:.2f}ms")

    # Function decorator
    @measure_latency
    def slow_function():
        time.sleep(0.1)
    
    result, latency_ms = slow_function()

    # Latency tracking across operations
    tracker = LatencyTracker()
    with tracker.measure("query"):
        # query execution
        pass
    
    stats = tracker.get_statistics()
"""

import time
import logging
import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)

# Import the existing timing infrastructure if available
# Note: This may fail due to torch dependencies in the retriever module
HAS_TIMING_HARNESS = False
TimingHarness = None
TimingResult = None
PerformanceProfile = None

try:
    from ..retriever.timing import TimingHarness, TimingResult, PerformanceProfile
    HAS_TIMING_HARNESS = True
except (ImportError, OSError) as e:
    logger.debug(f"TimingHarness not available: {e}")
    HAS_TIMING_HARNESS = False


@dataclass
class SimpleTimingResult:
    """Lightweight timing result for simple measurements."""
    
    operation: str
    latency_ms: float
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        return f"{self.operation}: {self.latency_ms:.2f}ms"


class TimingContext:
    """Simple context manager for timing operations."""
    
    def __init__(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        self.operation = operation
        self.metadata = metadata or {}
        self.start_time = 0.0
        self.end_time = 0.0
        self.latency_ms = 0.0
        self.timestamp = 0.0
    
    def __enter__(self) -> 'TimingContext':
        self.timestamp = time.time()
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.latency_ms = (self.end_time - self.start_time) * 1000.0
    
    def get_result(self) -> SimpleTimingResult:
        """Get timing result."""
        return SimpleTimingResult(
            operation=self.operation,
            latency_ms=self.latency_ms,
            timestamp=self.timestamp,
            metadata=self.metadata
        )


def measure_latency(func: Optional[Callable] = None, *, operation_name: Optional[str] = None):
    """
    Decorator to measure function execution latency.
    
    Args:
        func: Function to wrap (when used as @measure_latency)
        operation_name: Custom name for the operation (defaults to function name)
    
    Returns:
        Tuple of (function_result, latency_ms)
        
    Usage:
        @measure_latency
        def my_function():
            return "result"
        
        result, latency_ms = my_function()
        
        # Or with custom name
        @measure_latency(operation_name="custom_op")
        def my_function():
            return "result"
    """
    def decorator(f: Callable) -> Callable:
        op_name = operation_name or f.__name__
        
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            with TimingContext(op_name) as timer:
                result = f(*args, **kwargs)
            return result, timer.latency_ms
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


class LatencyTracker:
    """Tracks latencies across multiple operations for statistical analysis."""
    
    def __init__(self):
        self.measurements: Dict[str, List[float]] = defaultdict(list)
        self.last_measurement: Optional[SimpleTimingResult] = None
    
    @contextmanager
    def measure(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager to measure and track latency."""
        with TimingContext(operation, metadata) as timer:
            yield timer
        
        self.measurements[operation].append(timer.latency_ms)
        self.last_measurement = timer.get_result()
    
    def record_latency(self, operation: str, latency_ms: float):
        """Manually record a latency measurement."""
        self.measurements[operation].append(latency_ms)
        self.last_measurement = SimpleTimingResult(
            operation=operation,
            latency_ms=latency_ms,
            timestamp=time.time()
        )
    
    def get_statistics(self, operation: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Get statistical summary of latencies.
        
        Args:
            operation: Specific operation to analyze (None for all operations)
            
        Returns:
            Dictionary with statistics for each operation
        """
        if operation:
            operations = [operation] if operation in self.measurements else []
        else:
            operations = list(self.measurements.keys())
        
        stats = {}
        
        for op in operations:
            latencies = self.measurements[op]
            if not latencies:
                continue
            
            stats[op] = {
                'count': len(latencies),
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'std': statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
                'min': min(latencies),
                'max': max(latencies),
                'p95': statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
                'p99': statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies)
            }
        
        return stats
    
    def clear(self, operation: Optional[str] = None):
        """Clear measurements."""
        if operation:
            self.measurements[operation].clear()
        else:
            self.measurements.clear()
            self.last_measurement = None
    
    def get_last_measurement(self) -> Optional[SimpleTimingResult]:
        """Get the most recent timing measurement."""
        return self.last_measurement


# Global tracker instance for convenience
_global_tracker = LatencyTracker()


def track_latency(operation: str):
    """Convenience function to track latency globally."""
    return _global_tracker.measure(operation)


def get_global_stats() -> Dict[str, Dict[str, float]]:
    """Get global latency statistics."""
    return _global_tracker.get_statistics()


def clear_global_stats():
    """Clear global latency statistics."""
    _global_tracker.clear()


# Integration with existing timing infrastructure
if HAS_TIMING_HARNESS:
    class EnhancedTimingContext(TimingContext):
        """Enhanced timing context that uses TimingHarness when available."""
        
        def __init__(self, operation: str, 
                     harness: Optional[TimingHarness] = None,
                     metadata: Optional[Dict[str, Any]] = None):
            super().__init__(operation, metadata)
            self.harness = harness
            self._harness_context = None
        
        def __enter__(self) -> 'EnhancedTimingContext':
            if self.harness:
                self._harness_context = self.harness.measure(self.operation, self.metadata)
                self._harness_context.__enter__()
            
            return super().__enter__()
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._harness_context:
                self._harness_context.__exit__(exc_type, exc_val, exc_tb)
                # Extract latency from harness results
                if self.harness.results:
                    self.latency_ms = self.harness.results[-1].latency_ms
            else:
                super().__exit__(exc_type, exc_val, exc_tb)


# Convenience functions for common timing patterns
def time_operation(operation: str, func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """Time a function call and return result with latency."""
    with TimingContext(operation) as timer:
        result = func(*args, **kwargs)
    return result, timer.latency_ms


def benchmark_function(func: Callable, iterations: int = 100, 
                      warmup: int = 10) -> Dict[str, float]:
    """Simple function benchmarking."""
    # Warmup
    for _ in range(warmup):
        try:
            func()
        except Exception:
            pass
    
    # Measurements
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            func()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
        except Exception:
            continue
    
    if not latencies:
        return {'error': 'All iterations failed'}
    
    return {
        'count': len(latencies),
        'mean': statistics.mean(latencies),
        'median': statistics.median(latencies),
        'std': statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        'min': min(latencies),
        'max': max(latencies),
        'p95': statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
    }


# Utility for replacing common timing patterns
def replace_timing_pattern(operation: str):
    """
    Context manager to replace the common pattern:
        start_time = time.time()
        # ... operation ...
        latency_ms = (time.time() - start_time) * 1000
    
    Usage:
        with replace_timing_pattern("query_execution") as timer:
            # ... operation ...
            pass
        latency_ms = timer.latency_ms
    """
    return TimingContext(operation)


# Export commonly used components
__all__ = [
    'TimingContext',
    'SimpleTimingResult',
    'LatencyTracker', 
    'measure_latency',
    'track_latency',
    'time_operation',
    'benchmark_function',
    'replace_timing_pattern',
    'get_global_stats',
    'clear_global_stats'
]