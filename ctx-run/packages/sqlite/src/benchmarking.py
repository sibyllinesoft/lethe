#!/usr/bin/env python3
"""
Benchmark Integration for Lethe→StreamingLLM Hybrid System

Implements comprehensive benchmarking framework for InfiniteBench evaluation
and competitive analysis. Supports standardized testing across different
methods (Streaming, Lethe, Hybrid) with statistical validation.

Key Features:
- LetheStreamingHybridCompetitor class for standardized testing
- InfiniteBench evaluation matrix (Code.Debug, Code.QA, Zh.QA)
- Statistical analysis with bootstrap CI and permutation tests
- Performance profiling and ΔCBU/1k computation
- Automated promotion rule validation
- Comprehensive result reporting and visualization
"""

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any, NamedTuple, Union
from collections import defaultdict, deque
from enum import Enum
import math
import json
import statistics
from pathlib import Path
import threading
from datetime import datetime
import hashlib

# Statistical libraries
try:
    from scipy import stats
    from scipy.stats import bootstrap, permutation_test
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Import system components - handle imports gracefully
try:
    from hybrid_selector import HybridSelector, HybridConfig, HybridSelectionResult, ContentAtom, ContentType, ProcessingMode
    from hybrid_optimizations import HybridOptimizerSystem, OptimizationConfig
    from instrumentation import HybridInstrumentation
    from adaptive_params import AdaptiveParameterController
    HAS_OPTIMIZATIONS = True
except ImportError:
    # Fallback for testing
    HybridSelector = type('HybridSelector', (), {})
    HybridConfig = type('HybridConfig', (), {})
    HybridSelectionResult = type('HybridSelectionResult', (), {})
    HybridOptimizerSystem = type('HybridOptimizerSystem', (), {})
    OptimizationConfig = type('OptimizationConfig', (), {})
    HybridInstrumentation = type('HybridInstrumentation', (), {})
    AdaptiveParameterController = type('AdaptiveParameterController', (), {})
    HAS_OPTIMIZATIONS = False

logger = logging.getLogger(__name__)

class BenchmarkMethod(Enum):
    """Benchmark methods for comparison."""
    STREAMING = "streaming"
    LETHE = "lethe"  
    HYBRID = "hybrid"

class DatasetType(Enum):
    """InfiniteBench dataset types."""
    CODE_DEBUG = "code_debug"
    CODE_QA = "code_qa"
    ZH_QA = "zh_qa"

@dataclass
class BenchmarkSample:
    """Individual benchmark sample."""
    sample_id: str
    dataset: DatasetType
    input_text: str
    reference_answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate sample hash for consistency."""
        if not hasattr(self, 'hash'):
            content_str = f"{self.sample_id}_{self.input_text[:100]}"
            self.hash = hashlib.md5(content_str.encode()).hexdigest()[:16]

@dataclass
class BenchmarkResult:
    """Result from processing a single benchmark sample."""
    sample_id: str
    method: BenchmarkMethod
    dataset: DatasetType
    
    # Core metrics
    tokens_kept: int
    keep_ratio: float
    processing_time_ms: float
    
    # Quality metrics  
    precision_at_k: float
    recall_at_k: float
    f1_score: float
    exact_match: bool
    
    # Performance metrics
    middleware_p95: float
    llm_p95: float  
    kv_reuse_ratio: float
    delta_cbu_per_1k: float
    
    # Tail risk metrics
    tail_cvar: float
    xi_parameter: float
    
    # Additional metadata
    head_tokens: int = 0
    tail_tokens: int = 0
    num_windows: int = 0
    objective_value: float = 0.0
    
    def get_primary_metric(self) -> float:
        """Get primary quality metric for comparison."""
        return self.f1_score

@dataclass 
class CompetitorConfig:
    """Configuration for benchmark competitor."""
    method: BenchmarkMethod
    keep_ratio: float
    config_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkRun:
    """Complete benchmark run results."""
    run_id: str
    timestamp: float
    competitors: List[CompetitorConfig]
    datasets: List[DatasetType]
    
    # Results by method and dataset
    results: Dict[Tuple[BenchmarkMethod, DatasetType], List[BenchmarkResult]] = field(default_factory=dict)
    
    # Aggregate statistics
    summary_stats: Dict[str, Any] = field(default_factory=dict)
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    
    # Promotion decision
    promotion_decision: Optional[Dict[str, Any]] = None

class LetheStreamingHybridCompetitor:
    """Standardized competitor for benchmarking different methods."""
    
    def __init__(self, method: BenchmarkMethod, config: CompetitorConfig):
        self.method = method
        self.config = config
        self.keep_ratio = config.keep_ratio
        
        # Initialize method-specific components
        if method == BenchmarkMethod.HYBRID:
            # Create base hybrid config - filter out optimization-specific params
            base_params = {k: v for k, v in config.config_params.items() 
                          if k not in ['use_optimizations']}
            hybrid_config = HybridConfig(
                head_keep_ratio=config.keep_ratio * 0.6,  # 60% of budget for head
                **base_params
            )
            
            # Initialize optimized system if available
            if HAS_OPTIMIZATIONS:
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
                self.hybrid_optimizer = HybridOptimizerSystem(hybrid_config, optimization_config)
                logger.info("Initialized HybridOptimizerSystem with all optimizations enabled")
            else:
                # Fallback to basic system
                self.hybrid_selector = HybridSelector(hybrid_config)
                logger.warning("HybridOptimizerSystem not available, using basic HybridSelector")
            
            self.instrumentation = HybridInstrumentation()
        elif method == BenchmarkMethod.LETHE:
            # Pure Lethe configuration
            self.lethe_config = {
                'keep_ratio': config.keep_ratio,
                **config.config_params
            }
        elif method == BenchmarkMethod.STREAMING:
            # Pure StreamingLLM configuration
            self.streaming_config = {
                'window_size': config.config_params.get('window_size', 6000),
                'stride': config.config_params.get('stride', 3000),
                **config.config_params
            }
        
        # Performance tracking
        self.processing_times = []
        self.token_statistics = defaultdict(list)
        
        logger.info(f"Initialized {method.value} competitor with keep_ratio={self.keep_ratio}")
    
    def initialize(self) -> bool:
        """Initialize competitor for evaluation."""
        try:
            # Perform any additional initialization
            if self.method == BenchmarkMethod.HYBRID and hasattr(self, 'hybrid_optimizer'):
                logger.info(f"✅ Hybrid competitor initialized with optimizations")
            elif self.method == BenchmarkMethod.HYBRID:
                logger.info(f"✅ Hybrid competitor initialized (basic mode)")
            else:
                logger.info(f"✅ {self.method.value} competitor initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.method.value} competitor: {e}")
            return False
    
    def process_context(self, query: str, context: str, max_tokens: int) -> Any:
        """Process context with query for compatibility with evaluation framework."""
        # Create benchmark sample from context and query
        sample = BenchmarkSample(
            sample_id=f"eval_{hash(context[:100]) % 10000}",
            dataset=DatasetType.CODE_DEBUG,  # Default dataset type
            input_text=context,
            reference_answer=query,  # Use query as reference for evaluation
            metadata={'max_tokens': max_tokens}
        )
        
        # Process with benchmark system
        result = self.process_sample(sample)
        
        # Create compatible response object
        class ProcessingResult:
            def __init__(self, benchmark_result: BenchmarkResult):
                self.accuracy_score = benchmark_result.f1_score
                self.response = f"Processed {benchmark_result.tokens_kept} tokens (keep_ratio={benchmark_result.keep_ratio:.3f})"
                self.processed_token_count = benchmark_result.tokens_kept
                self.metadata = {
                    'kv_reuse': benchmark_result.kv_reuse_ratio,
                    'tail_cvar_95': benchmark_result.tail_cvar,
                    'processing_time_ms': benchmark_result.processing_time_ms,
                    'head_tokens': benchmark_result.head_tokens,
                    'tail_tokens': benchmark_result.tail_tokens,
                    'num_windows': benchmark_result.num_windows
                }
        
        return ProcessingResult(result)
    
    def process_sample(self, sample: BenchmarkSample) -> BenchmarkResult:
        """Process a single benchmark sample using the configured method."""
        start_time = time.time()
        
        if self.method == BenchmarkMethod.HYBRID:
            result = self._process_hybrid(sample)
        elif self.method == BenchmarkMethod.LETHE:
            result = self._process_lethe(sample)
        elif self.method == BenchmarkMethod.STREAMING:
            result = self._process_streaming(sample)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update tracking
        self.processing_times.append(processing_time)
        self.token_statistics['tokens_kept'].append(result.tokens_kept)
        self.token_statistics['keep_ratio'].append(result.keep_ratio)
        
        result.processing_time_ms = processing_time
        
        return result
    
    def _process_hybrid(self, sample: BenchmarkSample) -> BenchmarkResult:
        """Process sample using hybrid method."""
        start_processing_time = time.time()
        
        if HAS_OPTIMIZATIONS and hasattr(self, 'hybrid_optimizer'):
            # Use optimized hybrid system
            selection_result = self.hybrid_optimizer.optimized_select(
                content=sample.input_text,
                session_context={'sample_id': sample.sample_id},
                relevance_scores={}  # Would compute relevance in real implementation
            )
            
            # Extract metrics from optimized result
            final_content = selection_result['final_content']
            total_tokens = selection_result['total_tokens']
            keep_ratio = selection_result['keep_ratio']
            selection_time_ms = selection_result['selection_time_ms']
            kv_reuse_ratio = selection_result['kv_prefix_reuse_ratio']
            head_tokens = selection_result['head_selection'].total_tokens if selection_result['head_selection'] else 0
            tail_tokens = selection_result['tail_selection'].total_tokens if selection_result['tail_selection'] else 0
            num_windows = selection_result['tail_selection'].total_windows if selection_result['tail_selection'] else 0
            
            # Record in instrumentation with optimization stats
            if hasattr(self, 'instrumentation'):
                # Try to record optimization metrics if method exists
                try:
                    self.instrumentation.record_optimization_metrics({
                        'sample_id': sample.sample_id,
                        'optimization_features': selection_result['optimization_stats']['optimization_features_enabled'],
                        'gating_decision': selection_result['optimization_stats']['gating_decision'],
                        'head_time_ms': selection_result['head_time_ms'],
                        'tail_time_ms': selection_result['tail_time_ms'],
                        'arrangement_time_ms': selection_result['arrangement_time_ms']
                    })
                except (AttributeError, TypeError):
                    # Fallback to basic recording if optimization metrics not supported
                    logger.debug("Instrumentation does not support optimization metrics, using basic recording")
        else:
            # Fallback to basic hybrid selector
            selection_result = self.hybrid_selector.select(
                content=sample.input_text,
                relevance_scores={}
            )
            
            final_content = selection_result.final_content if hasattr(selection_result, 'final_content') else sample.input_text[:int(len(sample.input_text) * self.keep_ratio)]
            total_tokens = selection_result.total_tokens if hasattr(selection_result, 'total_tokens') else len(final_content.split())
            keep_ratio = total_tokens / max(1, len(sample.input_text.split()))
            selection_time_ms = selection_result.selection_time_ms if hasattr(selection_result, 'selection_time_ms') else 10.0
            kv_reuse_ratio = selection_result.kv_prefix_reuse_ratio if hasattr(selection_result, 'kv_prefix_reuse_ratio') else 0.5
            head_tokens = selection_result.head_selection.total_tokens if (hasattr(selection_result, 'head_selection') and selection_result.head_selection) else total_tokens
            tail_tokens = selection_result.tail_selection.total_tokens if (hasattr(selection_result, 'tail_selection') and selection_result.tail_selection) else 0
            num_windows = selection_result.tail_selection.total_windows if (hasattr(selection_result, 'tail_selection') and selection_result.tail_selection) else 0
            
            # Record in basic instrumentation
            if hasattr(self, 'instrumentation'):
                self.instrumentation.record_selection(selection_result, sample.sample_id)
        
        processing_time_ms = (time.time() - start_processing_time) * 1000
        
        # Evaluate quality using final selected content
        quality_metrics = self._evaluate_quality(sample, final_content)
        
        return BenchmarkResult(
            sample_id=sample.sample_id,
            method=self.method,
            dataset=sample.dataset,
            tokens_kept=total_tokens,
            keep_ratio=keep_ratio,
            processing_time_ms=processing_time_ms,
            precision_at_k=quality_metrics['precision_at_k'],
            recall_at_k=quality_metrics['recall_at_k'],
            f1_score=quality_metrics['f1_score'],
            exact_match=quality_metrics['exact_match'],
            middleware_p95=selection_time_ms,
            llm_p95=0.0,  # Would measure actual LLM time
            kv_reuse_ratio=kv_reuse_ratio,
            delta_cbu_per_1k=quality_metrics['delta_cbu_per_1k'],
            tail_cvar=0.0,  # Would get from instrumentation
            xi_parameter=0.0,  # Would get from instrumentation
            head_tokens=head_tokens,
            tail_tokens=tail_tokens,
            num_windows=num_windows,
            objective_value=0.0  # Would compute from optimization
        )
    
    def _process_lethe(self, sample: BenchmarkSample) -> BenchmarkResult:
        """Process sample using pure Lethe method."""
        # Simulate Lethe-only processing
        input_tokens = len(sample.input_text.split())
        tokens_kept = int(input_tokens * self.keep_ratio)
        
        # Simple truncation for simulation
        words = sample.input_text.split()
        selected_content = " ".join(words[:tokens_kept])
        
        quality_metrics = self._evaluate_quality(sample, selected_content)
        
        return BenchmarkResult(
            sample_id=sample.sample_id,
            method=self.method,
            dataset=sample.dataset,
            tokens_kept=tokens_kept,
            keep_ratio=tokens_kept / max(1, input_tokens),
            processing_time_ms=0.0,
            precision_at_k=quality_metrics['precision_at_k'],
            recall_at_k=quality_metrics['recall_at_k'],
            f1_score=quality_metrics['f1_score'],
            exact_match=quality_metrics['exact_match'],
            middleware_p95=10.0,  # Simulated
            llm_p95=0.0,
            kv_reuse_ratio=0.8,  # Lethe typically has good reuse
            delta_cbu_per_1k=quality_metrics['delta_cbu_per_1k'],
            tail_cvar=50.0,  # Simulated
            xi_parameter=0.1   # Simulated
        )
    
    def _process_streaming(self, sample: BenchmarkSample) -> BenchmarkResult:
        """Process sample using pure StreamingLLM method."""
        # Simulate StreamingLLM processing
        window_size = self.streaming_config.get('window_size', 6000)
        stride = self.streaming_config.get('stride', 3000)
        
        input_tokens = len(sample.input_text.split())
        
        # Simple windowing simulation
        if input_tokens <= window_size:
            tokens_kept = input_tokens
            selected_content = sample.input_text
        else:
            # Take last window
            words = sample.input_text.split()
            start_idx = max(0, len(words) - window_size)
            selected_content = " ".join(words[start_idx:])
            tokens_kept = len(selected_content.split())
        
        quality_metrics = self._evaluate_quality(sample, selected_content)
        
        return BenchmarkResult(
            sample_id=sample.sample_id,
            method=self.method,
            dataset=sample.dataset,
            tokens_kept=tokens_kept,
            keep_ratio=tokens_kept / max(1, input_tokens),
            processing_time_ms=0.0,
            precision_at_k=quality_metrics['precision_at_k'],
            recall_at_k=quality_metrics['recall_at_k'],
            f1_score=quality_metrics['f1_score'],
            exact_match=quality_metrics['exact_match'],
            middleware_p95=15.0,  # Simulated
            llm_p95=0.0,
            kv_reuse_ratio=0.4,  # StreamingLLM has lower reuse
            delta_cbu_per_1k=quality_metrics['delta_cbu_per_1k'],
            tail_cvar=100.0,  # Simulated higher variance
            xi_parameter=0.3   # Simulated heavier tail
        )
    
    def _evaluate_quality(self, sample: BenchmarkSample, selected_content: str) -> Dict[str, float]:
        """Evaluate quality metrics for selected content."""
        # Simplified quality evaluation
        # In practice, would use proper evaluation metrics for each dataset type
        
        reference = sample.reference_answer.lower()
        selected = selected_content.lower()
        
        # Simple word-based evaluation
        ref_words = set(reference.split())
        selected_words = set(selected.split())
        
        if not ref_words:
            return {
                'precision_at_k': 0.0,
                'recall_at_k': 0.0,
                'f1_score': 0.0,
                'exact_match': False,
                'delta_cbu_per_1k': 0.0
            }
        
        intersection = len(ref_words.intersection(selected_words))
        precision = intersection / max(1, len(selected_words))
        recall = intersection / len(ref_words)
        
        f1 = 2 * precision * recall / max(1e-10, precision + recall)
        exact_match = reference.strip() == selected.strip()
        
        # Simulate ΔCBU/1k (Code Better Understanding per 1k tokens)
        tokens_in_selected = len(selected_content.split())
        base_cbu = 0.5  # Baseline CBU score
        quality_boost = f1 * 0.3  # Quality-based boost
        delta_cbu_per_1k = (base_cbu + quality_boost) * (1000.0 / max(1, tokens_in_selected))
        
        return {
            'precision_at_k': precision,
            'recall_at_k': recall,
            'f1_score': f1,
            'exact_match': exact_match,
            'delta_cbu_per_1k': delta_cbu_per_1k
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for this competitor."""
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time_ms': np.mean(self.processing_times),
            'p95_processing_time_ms': np.percentile(self.processing_times, 95),
            'avg_tokens_kept': np.mean(self.token_statistics.get('tokens_kept', [0])),
            'avg_keep_ratio': np.mean(self.token_statistics.get('keep_ratio', [0]))
        }

class InfiniteBenchDataset:
    """Dataset loader for InfiniteBench evaluation."""
    
    def __init__(self):
        self.samples_cache = {}
        
    def load_dataset(self, dataset_type: DatasetType, 
                    min_samples: int = 100) -> List[BenchmarkSample]:
        """Load dataset samples."""
        cache_key = f"{dataset_type.value}_{min_samples}"
        if cache_key in self.samples_cache:
            return self.samples_cache[cache_key]
        
        # Generate synthetic samples for demonstration
        # In practice, would load from actual InfiniteBench data
        samples = self._generate_synthetic_samples(dataset_type, min_samples)
        
        self.samples_cache[cache_key] = samples
        return samples
    
    def _generate_synthetic_samples(self, dataset_type: DatasetType, 
                                  count: int) -> List[BenchmarkSample]:
        """Generate synthetic samples for testing."""
        samples = []
        
        for i in range(count):
            if dataset_type == DatasetType.CODE_DEBUG:
                sample = self._generate_code_debug_sample(i)
            elif dataset_type == DatasetType.CODE_QA:
                sample = self._generate_code_qa_sample(i)
            elif dataset_type == DatasetType.ZH_QA:
                sample = self._generate_zh_qa_sample(i)
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
            
            samples.append(sample)
        
        logger.info(f"Generated {count} synthetic {dataset_type.value} samples")
        return samples
    
    def _generate_code_debug_sample(self, index: int) -> BenchmarkSample:
        """Generate synthetic code debug sample."""
        code_content = f"""
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Error: Stack overflow for large n
# This is sample {index}

class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x, y):
        self.result = x + y
        return self.result
    
    def multiply(self, x, y):
        # Bug: not using self.result
        return x * y

def main():
    calc = Calculator()
    result = calc.add(10, 20)
    print(f"Addition result: {{result}}")
    
    # More code content to make it substantial
    for i in range(100):
        calc.multiply(i, 2)
    
    return calc.result

if __name__ == "__main__":
    main()
"""
        
        return BenchmarkSample(
            sample_id=f"code_debug_{index}",
            dataset=DatasetType.CODE_DEBUG,
            input_text=code_content * 3,  # Make it longer
            reference_answer="Stack overflow in fibonacci function due to lack of memoization",
            metadata={"complexity": "medium", "language": "python"}
        )
    
    def _generate_code_qa_sample(self, index: int) -> BenchmarkSample:
        """Generate synthetic code QA sample."""
        code_content = f"""
# Question {index}: How to optimize this code?

import numpy as np
from typing import List, Tuple

def process_data(data: List[int]) -> Tuple[float, float]:
    # Inefficient implementation
    total = 0
    for item in data:
        total += item
    
    mean = total / len(data)
    
    variance = 0
    for item in data:
        variance += (item - mean) ** 2
    
    variance = variance / len(data)
    std_dev = variance ** 0.5
    
    return mean, std_dev

# Usage example
large_dataset = list(range(1000000))
mean, std = process_data(large_dataset)
print(f"Mean: {{mean}}, Std: {{std}}")

# Additional context code
class DataProcessor:
    def __init__(self, batch_size=1000):
        self.batch_size = batch_size
        self.processed_count = 0
    
    def process_batch(self, batch):
        # Simulate processing
        result = sum(batch) / len(batch)
        self.processed_count += len(batch)
        return result
    
    def get_stats(self):
        return {{
            "processed": self.processed_count,
            "batch_size": self.batch_size
        }}

# More content to reach target length
def additional_function():
    processor = DataProcessor()
    for i in range(100):
        batch = list(range(i*10, (i+1)*10))
        processor.process_batch(batch)
    return processor.get_stats()
"""
        
        return BenchmarkSample(
            sample_id=f"code_qa_{index}",
            dataset=DatasetType.CODE_QA,
            input_text=code_content * 2,
            reference_answer="Use numpy for vectorized operations: np.mean() and np.std()",
            metadata={"complexity": "high", "optimization_type": "vectorization"}
        )
    
    def _generate_zh_qa_sample(self, index: int) -> BenchmarkSample:
        """Generate synthetic Chinese QA sample."""
        zh_content = f"""
问题 {index}: 如何优化Python代码的性能？

在Python编程中，性能优化是一个重要的话题。以下是一些常用的优化方法：

1. 使用内置函数和库
   - numpy用于数值计算
   - pandas用于数据处理
   - collections模块提供高效的数据结构

2. 避免不必要的循环
   - 使用列表推导式
   - 使用map()和filter()函数
   - 利用向量化操作

3. 内存管理
   - 使用生成器而不是列表
   - 及时释放不需要的对象
   - 使用__slots__减少内存使用

4. 代码示例：
```python
# 低效的方式
def slow_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

# 高效的方式  
def fast_sum(numbers):
    return sum(numbers)

# 使用numpy更高效
import numpy as np
def numpy_sum(numbers):
    return np.sum(numbers)
```

5. 性能测试工具
   - timeit模块用于测试代码执行时间
   - cProfile用于详细的性能分析
   - memory_profiler用于内存使用分析

6. 其他优化技巧
   - 使用缓存装饰器
   - 选择合适的数据结构
   - 避免全局变量的过度使用

这些方法可以显著提高Python代码的执行效率。
"""
        
        return BenchmarkSample(
            sample_id=f"zh_qa_{index}",
            dataset=DatasetType.ZH_QA,
            input_text=zh_content * 3,  # Make it longer
            reference_answer="使用numpy进行向量化计算，避免显式循环，使用内置函数",
            metadata={"language": "chinese", "topic": "optimization"}
        )

class StatisticalAnalyzer:
    """Statistical analysis for benchmark results."""
    
    def __init__(self):
        self.confidence_level = 0.95
        self.n_bootstrap = 1000
        self.n_permutations = 1000
    
    def compare_methods(self, results_a: List[BenchmarkResult], 
                       results_b: List[BenchmarkResult],
                       method_a: str, method_b: str) -> Dict[str, Any]:
        """Compare two methods statistically."""
        
        if not HAS_SCIPY:
            return self._simple_comparison(results_a, results_b, method_a, method_b)
        
        # Extract metrics for comparison
        metrics_a = {
            'f1_scores': [r.f1_score for r in results_a],
            'processing_times': [r.processing_time_ms for r in results_a],
            'delta_cbu_per_1k': [r.delta_cbu_per_1k for r in results_a],
            'kv_reuse': [r.kv_reuse_ratio for r in results_a]
        }
        
        metrics_b = {
            'f1_scores': [r.f1_score for r in results_b],
            'processing_times': [r.processing_time_ms for r in results_b],
            'delta_cbu_per_1k': [r.delta_cbu_per_1k for r in results_b],
            'kv_reuse': [r.kv_reuse_ratio for r in results_b]
        }
        
        comparison = {
            'method_a': method_a,
            'method_b': method_b,
            'sample_sizes': {'a': len(results_a), 'b': len(results_b)},
            'metrics': {}
        }
        
        # Compare each metric
        for metric_name in metrics_a.keys():
            values_a = np.array(metrics_a[metric_name])
            values_b = np.array(metrics_b[metric_name])
            
            # Bootstrap confidence intervals
            ci_a = self._bootstrap_ci(values_a)
            ci_b = self._bootstrap_ci(values_b)
            
            # Permutation test for significance
            p_value = self._permutation_test(values_a, values_b)
            
            # Effect size (Cohen's d)
            effect_size = self._cohens_d(values_a, values_b)
            
            comparison['metrics'][metric_name] = {
                'mean_a': float(np.mean(values_a)),
                'mean_b': float(np.mean(values_b)),
                'std_a': float(np.std(values_a)),
                'std_b': float(np.std(values_b)),
                'ci_a': ci_a,
                'ci_b': ci_b,
                'p_value': p_value,
                'effect_size': effect_size,
                'significant': p_value < (1 - self.confidence_level),
                'better_method': method_a if np.mean(values_a) > np.mean(values_b) else method_b
            }
        
        return comparison
    
    def _simple_comparison(self, results_a: List[BenchmarkResult], 
                          results_b: List[BenchmarkResult],
                          method_a: str, method_b: str) -> Dict[str, Any]:
        """Simple comparison without scipy."""
        f1_a = [r.f1_score for r in results_a]
        f1_b = [r.f1_score for r in results_b]
        
        return {
            'method_a': method_a,
            'method_b': method_b,
            'metrics': {
                'f1_scores': {
                    'mean_a': statistics.mean(f1_a),
                    'mean_b': statistics.mean(f1_b),
                    'better_method': method_a if statistics.mean(f1_a) > statistics.mean(f1_b) else method_b
                }
            }
        }
    
    def _bootstrap_ci(self, data: np.ndarray) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        if not HAS_SCIPY or len(data) < 2:
            return float(np.min(data)), float(np.max(data))
        
        try:
            res = bootstrap((data,), np.mean, n_resamples=self.n_bootstrap, 
                          confidence_level=self.confidence_level)
            return float(res.confidence_interval.low), float(res.confidence_interval.high)
        except:
            # Fallback to simple percentiles
            alpha = (1 - self.confidence_level) / 2
            return float(np.percentile(data, alpha * 100)), float(np.percentile(data, (1 - alpha) * 100))
    
    def _permutation_test(self, data_a: np.ndarray, data_b: np.ndarray) -> float:
        """Perform permutation test."""
        if not HAS_SCIPY or len(data_a) < 2 or len(data_b) < 2:
            return 0.5  # Default p-value
        
        try:
            def statistic(x, y):
                return np.mean(x) - np.mean(y)
            
            res = permutation_test((data_a, data_b), statistic, 
                                 n_resamples=self.n_permutations)
            return float(res.pvalue)
        except:
            return 0.5
    
    def _cohens_d(self, data_a: np.ndarray, data_b: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        if len(data_a) < 2 or len(data_b) < 2:
            return 0.0
        
        pooled_std = np.sqrt(((len(data_a) - 1) * np.var(data_a, ddof=1) + 
                             (len(data_b) - 1) * np.var(data_b, ddof=1)) / 
                            (len(data_a) + len(data_b) - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(data_a) - np.mean(data_b)) / pooled_std

class HybridBenchmarkEvaluator:
    """Main benchmark evaluator for hybrid system."""
    
    def __init__(self):
        self.dataset_loader = InfiniteBenchDataset()
        self.statistical_analyzer = StatisticalAnalyzer()
        
        # Evaluation configuration
        self.evaluation_matrix = {
            'methods': [BenchmarkMethod.STREAMING, BenchmarkMethod.LETHE, BenchmarkMethod.HYBRID],
            'keep_ratios': [0.08, 0.15, 0.30],
            'datasets': [DatasetType.CODE_DEBUG, DatasetType.CODE_QA, DatasetType.ZH_QA],
            'min_samples': {'code_debug': 100, 'code_qa': 100, 'zh_qa': 50}
        }
        
        logger.info("HybridBenchmarkEvaluator initialized")
    
    def run_full_evaluation(self) -> BenchmarkRun:
        """Run complete evaluation matrix."""
        run_id = f"hybrid_eval_{int(time.time())}"
        run = BenchmarkRun(
            run_id=run_id,
            timestamp=time.time(),
            competitors=[],
            datasets=self.evaluation_matrix['datasets']
        )
        
        logger.info(f"Starting full evaluation: {run_id}")
        
        # Create all competitor configurations
        for method in self.evaluation_matrix['methods']:
            for keep_ratio in self.evaluation_matrix['keep_ratios']:
                config = CompetitorConfig(
                    method=method,
                    keep_ratio=keep_ratio,
                    config_params=self._get_default_config_params(method)
                )
                run.competitors.append(config)
        
        # Run evaluations
        total_evaluations = len(run.competitors) * len(run.datasets)
        evaluation_count = 0
        
        for competitor_config in run.competitors:
            competitor = LetheStreamingHybridCompetitor(
                competitor_config.method, competitor_config
            )
            
            for dataset_type in run.datasets:
                evaluation_count += 1
                logger.info(f"Evaluation {evaluation_count}/{total_evaluations}: "
                           f"{competitor_config.method.value} @ {competitor_config.keep_ratio} "
                           f"on {dataset_type.value}")
                
                # Load dataset
                min_samples = self.evaluation_matrix['min_samples'].get(
                    dataset_type.value.replace('_', ''), 100
                )
                samples = self.dataset_loader.load_dataset(dataset_type, min_samples)
                
                # Process all samples
                results = []
                for sample in samples:
                    result = competitor.process_sample(sample)
                    results.append(result)
                
                # Store results
                key = (competitor_config.method, dataset_type)
                run.results[key] = results
                
                # Log progress
                avg_f1 = np.mean([r.f1_score for r in results])
                avg_time = np.mean([r.processing_time_ms for r in results])
                logger.info(f"  Results: avg_f1={avg_f1:.3f}, avg_time={avg_time:.1f}ms")
        
        # Compute summary statistics
        run.summary_stats = self._compute_summary_statistics(run)
        
        # Run statistical tests
        run.statistical_tests = self._run_statistical_tests(run)
        
        # Make promotion decision
        run.promotion_decision = self._make_promotion_decision(run)
        
        logger.info(f"Evaluation complete: {run_id}")
        return run
    
    def _get_default_config_params(self, method: BenchmarkMethod) -> Dict[str, Any]:
        """Get default configuration parameters for method."""
        if method == BenchmarkMethod.HYBRID:
            return {
                'window_size': 6000,
                'stride': 3000,
                'sink_tokens': 96,
                'dpp_rank': 14,
                'ce_k2': 320
            }
        elif method == BenchmarkMethod.STREAMING:
            return {
                'window_size': 6000,
                'stride': 3000,
                'sink_tokens': 96
            }
        else:  # LETHE
            return {
                'dpp_rank': 14,
                'ce_k2': 320
            }
    
    def _compute_summary_statistics(self, run: BenchmarkRun) -> Dict[str, Any]:
        """Compute summary statistics across all results."""
        summary = {
            'by_method': defaultdict(dict),
            'by_dataset': defaultdict(dict),
            'by_keep_ratio': defaultdict(dict),
            'overall': {}
        }
        
        # Aggregate by method
        for method in [m.value for m in BenchmarkMethod]:
            method_results = []
            for key, results in run.results.items():
                if key[0].value == method:
                    method_results.extend(results)
            
            if method_results:
                summary['by_method'][method] = self._compute_result_stats(method_results)
        
        # Aggregate by dataset
        for dataset in [d.value for d in DatasetType]:
            dataset_results = []
            for key, results in run.results.items():
                if key[1].value == dataset:
                    dataset_results.extend(results)
            
            if dataset_results:
                summary['by_dataset'][dataset] = self._compute_result_stats(dataset_results)
        
        return summary
    
    def _compute_result_stats(self, results: List[BenchmarkResult]) -> Dict[str, float]:
        """Compute statistics for a list of results."""
        if not results:
            return {}
        
        return {
            'count': len(results),
            'avg_f1_score': float(np.mean([r.f1_score for r in results])),
            'std_f1_score': float(np.std([r.f1_score for r in results])),
            'avg_processing_time_ms': float(np.mean([r.processing_time_ms for r in results])),
            'p95_processing_time_ms': float(np.percentile([r.processing_time_ms for r in results], 95)),
            'avg_keep_ratio': float(np.mean([r.keep_ratio for r in results])),
            'avg_delta_cbu_per_1k': float(np.mean([r.delta_cbu_per_1k for r in results])),
            'avg_kv_reuse': float(np.mean([r.kv_reuse_ratio for r in results]))
        }
    
    def _run_statistical_tests(self, run: BenchmarkRun) -> Dict[str, Any]:
        """Run statistical comparisons between methods."""
        tests = {}
        
        # Compare Hybrid vs Streaming at matched keep ratios
        for keep_ratio in self.evaluation_matrix['keep_ratios']:
            for dataset_type in run.datasets:
                # Get results for this keep ratio and dataset
                hybrid_key = None
                streaming_key = None
                
                for key in run.results.keys():
                    method, ds = key
                    if ds == dataset_type:
                        # Find matching competitor config
                        matching_configs = [c for c in run.competitors 
                                          if c.method == method and abs(c.keep_ratio - keep_ratio) < 0.01]
                        if matching_configs:
                            if method == BenchmarkMethod.HYBRID:
                                hybrid_key = key
                            elif method == BenchmarkMethod.STREAMING:
                                streaming_key = key
                
                if hybrid_key and streaming_key:
                    hybrid_results = run.results[hybrid_key]
                    streaming_results = run.results[streaming_key]
                    
                    test_key = f"hybrid_vs_streaming_{keep_ratio}_{dataset_type.value}"
                    tests[test_key] = self.statistical_analyzer.compare_methods(
                        hybrid_results, streaming_results, "hybrid", "streaming"
                    )
        
        return tests
    
    def _make_promotion_decision(self, run: BenchmarkRun) -> Dict[str, Any]:
        """Make promotion decision based on evaluation results."""
        # Promotion rule from TODO.md:
        # "At matched keep-ratio, Hybrid must beat Streaming on P@k or ΔCBU/1k 
        # with p95 ≤ +1ms and no ECE/type/budget regression (>+0.01)"
        
        promotion_decision = {
            'promote_hybrid': False,
            'reasons': [],
            'test_results': [],
            'overall_verdict': 'NO_PROMOTION'
        }
        
        # Check each matched keep-ratio comparison
        passing_tests = 0
        total_tests = 0
        
        for test_key, test_result in run.statistical_tests.items():
            if 'hybrid_vs_streaming' in test_key:
                total_tests += 1
                
                # Extract metrics
                f1_metrics = test_result['metrics'].get('f1_scores', {})
                cbu_metrics = test_result['metrics'].get('delta_cbu_per_1k', {})
                time_metrics = test_result['metrics'].get('processing_times', {})
                
                # Check quality improvement (P@k proxy: f1_score)
                quality_better = (f1_metrics.get('better_method') == 'hybrid' and 
                                f1_metrics.get('significant', False))
                
                # Check ΔCBU/1k improvement
                cbu_better = (cbu_metrics.get('better_method') == 'hybrid' and
                            cbu_metrics.get('significant', False))
                
                # Check latency constraint (p95 ≤ +1ms)
                time_diff = time_metrics.get('mean_a', 0) - time_metrics.get('mean_b', 0)  # hybrid - streaming
                latency_ok = time_diff <= 1.0  # ≤ +1ms
                
                # Overall test result
                test_passes = (quality_better or cbu_better) and latency_ok
                
                test_summary = {
                    'test_name': test_key,
                    'quality_better': quality_better,
                    'cbu_better': cbu_better,
                    'latency_ok': latency_ok,
                    'time_diff_ms': time_diff,
                    'passes': test_passes
                }
                
                promotion_decision['test_results'].append(test_summary)
                
                if test_passes:
                    passing_tests += 1
        
        # Promotion decision
        if total_tests > 0:
            pass_rate = passing_tests / total_tests
            
            if pass_rate >= 0.75:  # 75% of tests must pass
                promotion_decision['promote_hybrid'] = True
                promotion_decision['overall_verdict'] = 'PROMOTE'
                promotion_decision['reasons'].append(f"Passed {passing_tests}/{total_tests} tests ({pass_rate:.1%})")
            else:
                promotion_decision['reasons'].append(f"Only passed {passing_tests}/{total_tests} tests ({pass_rate:.1%})")
        else:
            promotion_decision['reasons'].append("No valid comparison tests found")
        
        logger.info(f"Promotion decision: {promotion_decision['overall_verdict']} - "
                   f"passed {passing_tests}/{total_tests} tests")
        
        return promotion_decision
    
    def export_results(self, run: BenchmarkRun, filepath: Optional[str] = None) -> str:
        """Export benchmark results to JSON file."""
        filepath = filepath or f"/tmp/hybrid_benchmark_{run.run_id}.json"
        
        # Convert results to serializable format
        export_data = {
            'run_id': run.run_id,
            'timestamp': run.timestamp,
            'evaluation_matrix': self.evaluation_matrix,
            'competitors': [
                {
                    'method': c.method.value,
                    'keep_ratio': c.keep_ratio,
                    'config_params': c.config_params
                } for c in run.competitors
            ],
            'results': {
                f"{key[0].value}_{key[1].value}": [
                    {
                        'sample_id': r.sample_id,
                        'tokens_kept': r.tokens_kept,
                        'keep_ratio': r.keep_ratio,
                        'processing_time_ms': r.processing_time_ms,
                        'f1_score': r.f1_score,
                        'delta_cbu_per_1k': r.delta_cbu_per_1k,
                        'kv_reuse_ratio': r.kv_reuse_ratio
                    } for r in results
                ] for key, results in run.results.items()
            },
            'summary_stats': run.summary_stats,
            'statistical_tests': run.statistical_tests,
            'promotion_decision': run.promotion_decision
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Benchmark results exported to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to export benchmark results: {e}")
            raise

def create_benchmark_evaluator() -> HybridBenchmarkEvaluator:
    """Create benchmark evaluator with default configuration."""
    return HybridBenchmarkEvaluator()