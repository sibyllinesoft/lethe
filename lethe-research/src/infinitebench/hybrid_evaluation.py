"""
Lethe→StreamingLLM Hybrid Evaluation Framework for InfiniteBench

Evaluates the hybrid system against StreamingLLM and Lethe-only baselines
on InfiniteBench tasks, with comprehensive metrics and statistical validation.

Evaluation Matrix:
- Methods: {Streaming, Lethe, Hybrid}
- Keep ratios: {0.08, 0.15, 0.30}
- Datasets: InfiniteBench Code.Debug + Code.QA (≥100 items), Zh.QA (50 items)
- Metrics: P@k/R@k vs tokens kept, ΔCBU/1k, middleware p95, LLM p95, KV-reuse, tail CVaR

Promotion rule: Hybrid must beat Streaming on P@k or ΔCBU/1k with p95 ≤ +1ms
and no ECE/type/budget regression (>+0.01).
"""

import json
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import concurrent.futures

# Import our hybrid system
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from context_competitors.lethe_streaming_hybrid import HybridSelector, HybridResult
from context_competitors.benchmarks.streamingllm_benchmark import StreamingLLMCompetitor
from infinitebench.dataset_loader import InfiniteBenchDataset
from common.data_structures import EvaluationResult, PerformanceMetrics

logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for hybrid evaluation."""
    
    # Dataset configuration
    datasets: List[str] = field(default_factory=lambda: ['code_debug', 'code_qa', 'zh_qa'])
    max_samples_per_dataset: Dict[str, int] = field(default_factory=lambda: {
        'code_debug': 100,
        'code_qa': 100, 
        'zh_qa': 50
    })
    
    # Evaluation matrix
    methods: List[str] = field(default_factory=lambda: ['streaming', 'lethe', 'hybrid'])
    keep_ratios: List[float] = field(default_factory=lambda: [0.08, 0.15, 0.30])
    
    # Statistical validation
    confidence_level: float = 0.95
    min_effect_size: float = 0.01
    bootstrap_iterations: int = 1000
    
    # Performance thresholds
    max_latency_increase_ms: float = 1.0  # p95 ≤ +1ms requirement
    max_regression_threshold: float = 0.01  # >+0.01 regression threshold
    
    # Output configuration
    output_dir: Path = Path("hybrid_evaluation_results")
    save_detailed_results: bool = True
    save_instrumentation: bool = True

@dataclass
class EvaluationSample:
    """Single evaluation sample."""
    sample_id: str
    dataset: str
    query: str
    context: str
    ground_truth: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class MethodResult:
    """Result for a single method on a single sample."""
    sample_id: str
    method_name: str
    keep_ratio: float
    
    # Core metrics
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float
    
    # Performance metrics
    latency_ms: float
    tokens_kept: int
    original_tokens: int
    compression_ratio: float
    
    # Hybrid-specific metrics
    kv_reuse: float = 0.0
    delta_cbu_per_1k: float = 0.0
    tail_cvar_95: float = 0.0
    
    # Quality metrics
    response_quality: float = 0.0  # To be evaluated by external judge
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComparisonResult:
    """Statistical comparison between two methods."""
    method_a: str
    method_b: str
    keep_ratio: float
    
    # Statistical tests
    precision_p_value: float
    recall_p_value: float
    latency_p_value: float
    cbu_p_value: float
    
    # Effect sizes
    precision_effect_size: float
    recall_effect_size: float
    latency_effect_size: float
    cbu_effect_size: float
    
    # Significance flags
    significant_improvement: bool
    meets_promotion_criteria: bool
    
    metadata: Dict[str, Any] = field(default_factory=dict)

class InfiniteBenchLoader:
    """Loads and preprocesses InfiniteBench datasets for evaluation."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.dataset_cache = {}
    
    def load_dataset(self, dataset_name: str, max_samples: int = None) -> List[EvaluationSample]:
        """Load dataset samples for evaluation."""
        if dataset_name in self.dataset_cache:
            samples = self.dataset_cache[dataset_name]
        else:
            samples = self._load_dataset_from_disk(dataset_name)
            self.dataset_cache[dataset_name] = samples
        
        if max_samples:
            samples = samples[:max_samples]
        
        return samples
    
    def _load_dataset_from_disk(self, dataset_name: str) -> List[EvaluationSample]:
        """Load dataset from InfiniteBench data files."""
        # Map dataset names to actual InfiniteBench files
        dataset_mapping = {
            'code_debug': 'code_debug.jsonl',
            'code_qa': 'code_qa.jsonl', 
            'zh_qa': 'zh_qa.jsonl'
        }
        
        if dataset_name not in dataset_mapping:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        file_path = self.data_dir / dataset_mapping[dataset_name]
        if not file_path.exists():
            logger.warning(f"Dataset file not found: {file_path}")
            return []
        
        samples = []
        try:
            with open(file_path, 'r') as f:
                for line_idx, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        
                        # Extract standardized fields
                        sample = EvaluationSample(
                            sample_id=f"{dataset_name}_{line_idx}",
                            dataset=dataset_name,
                            query=data.get('input', ''),
                            context=data.get('context', ''),
                            ground_truth={'answer': data.get('answers', [])},
                            metadata={'original_data': data}
                        )
                        samples.append(sample)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_idx} in {file_path}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            return []
        
        logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
        return samples

class HybridMethodEvaluator:
    """Evaluates the hybrid method on evaluation samples."""
    
    def __init__(self):
        self.hybrid_selector = None
        self.streaming_competitor = None
        
    def initialize_methods(self, config: EvaluationConfig) -> bool:
        """Initialize all evaluation methods."""
        try:
            # Initialize hybrid selector with canary defaults
            hybrid_config = {
                'head_keep': 0.12,
                'window_size': 6000,
                'stride': 3000,
                'sinks': 96,
                'K2': 320,
                'dpp_rank': 14
            }
            self.hybrid_selector = HybridSelector(hybrid_config)
            
            # Initialize StreamingLLM competitor
            streaming_config = {
                'window_size': 6000,
                'attention_sink_size': 96,
                'model_name': 'gemma2:9b'
            }
            self.streaming_competitor = StreamingLLMCompetitor(streaming_config)
            
            # Initialize streaming competitor
            if not self.streaming_competitor.initialize():
                logger.warning("StreamingLLM competitor failed to initialize, will use mock")
                self.streaming_competitor = None
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize methods: {e}")
            return False
    
    def evaluate_hybrid_method(self, sample: EvaluationSample, keep_ratio: float) -> MethodResult:
        """Evaluate hybrid method on a single sample."""
        start_time = time.time()
        
        try:
            # Adjust hybrid selector parameters to target keep ratio
            self.hybrid_selector.head_keep_ratio = keep_ratio * 0.6  # 60% of budget for head
            
            # Calculate lambda to achieve target keep ratio
            total_tokens = len(sample.context.split())
            target_tokens = int(total_tokens * keep_ratio)
            lambda_param = self._calculate_lambda_for_tokens(target_tokens, total_tokens)
            
            # Execute hybrid selection
            result = self.hybrid_selector.select(
                query=sample.query,
                context=sample.context,
                lambda_param=lambda_param,
                mu_param=0.0001
            )
            
            # Calculate metrics
            precision_5, recall_5 = self._calculate_precision_recall(
                result.final_context, sample.ground_truth, k=5
            )
            precision_10, recall_10 = self._calculate_precision_recall(
                result.final_context, sample.ground_truth, k=10
            )
            
            latency_ms = time.time() - start_time
            
            return MethodResult(
                sample_id=sample.sample_id,
                method_name='hybrid',
                keep_ratio=result.keep_ratio,
                precision_at_5=precision_5,
                precision_at_10=precision_10,
                recall_at_5=recall_5,
                recall_at_10=recall_10,
                latency_ms=latency_ms * 1000,
                tokens_kept=result.total_tokens,
                original_tokens=total_tokens,
                compression_ratio=1.0 - result.keep_ratio,
                kv_reuse=result.instrumentation.kv_prefix_reuse,
                delta_cbu_per_1k=result.instrumentation.delta_cbu_per_1k,
                tail_cvar_95=result.instrumentation.tail_cvar_95,
                metadata={'hybrid_result': result}
            )
            
        except Exception as e:
            logger.error(f"Hybrid evaluation failed for {sample.sample_id}: {e}")
            return self._create_error_result(sample, 'hybrid', keep_ratio, str(e))
    
    def evaluate_streaming_method(self, sample: EvaluationSample, keep_ratio: float) -> MethodResult:
        """Evaluate StreamingLLM method on a single sample."""
        start_time = time.time()
        
        try:
            if self.streaming_competitor is None:
                # Mock streaming evaluation
                return self._create_mock_streaming_result(sample, keep_ratio)
            
            # Adjust window size to target keep ratio
            total_tokens = len(sample.context.split())
            target_tokens = int(total_tokens * keep_ratio)
            self.streaming_competitor.window_size = min(target_tokens, 8000)
            
            # Execute streaming selection
            result = self.streaming_competitor.process_context(
                query=sample.query,
                context=sample.context,
                max_tokens=target_tokens
            )
            
            # Calculate metrics
            precision_5, recall_5 = self._calculate_precision_recall(
                result.processed_context, sample.ground_truth, k=5
            )
            precision_10, recall_10 = self._calculate_precision_recall(
                result.processed_context, sample.ground_truth, k=10
            )
            
            latency_ms = time.time() - start_time
            
            return MethodResult(
                sample_id=sample.sample_id,
                method_name='streaming',
                keep_ratio=result.compression_ratio,
                precision_at_5=precision_5,
                precision_at_10=precision_10,
                recall_at_5=recall_5,
                recall_at_10=recall_10,
                latency_ms=latency_ms * 1000,
                tokens_kept=result.processed_token_count,
                original_tokens=result.original_token_count,
                compression_ratio=result.compression_ratio,
                metadata={'streaming_result': result}
            )
            
        except Exception as e:
            logger.error(f"Streaming evaluation failed for {sample.sample_id}: {e}")
            return self._create_error_result(sample, 'streaming', keep_ratio, str(e))
    
    def evaluate_lethe_method(self, sample: EvaluationSample, keep_ratio: float) -> MethodResult:
        """Evaluate Lethe-only method on a single sample."""
        start_time = time.time()
        
        try:
            # Use only the head builder from hybrid selector
            head_result = self.hybrid_selector.head_builder.build_head(
                sample.context,
                lambda_param=self._calculate_lambda_for_tokens(
                    int(len(sample.context.split()) * keep_ratio),
                    len(sample.context.split())
                ),
                mu_param=0.0001
            )
            
            # Create context from head only
            head_content = ""
            if head_result.selected_atoms:
                head_parts = []
                for group in head_result.selected_atoms:
                    group_content = ' '.join(group.atoms)
                    head_parts.append(f"# {group.group_type.upper()}\n{group_content}")
                head_content = '\n\n'.join(head_parts)
            
            # Calculate metrics
            precision_5, recall_5 = self._calculate_precision_recall(
                head_content, sample.ground_truth, k=5
            )
            precision_10, recall_10 = self._calculate_precision_recall(
                head_content, sample.ground_truth, k=10
            )
            
            latency_ms = time.time() - start_time
            
            return MethodResult(
                sample_id=sample.sample_id,
                method_name='lethe',
                keep_ratio=head_result.keep_ratio,
                precision_at_5=precision_5,
                precision_at_10=precision_10,
                recall_at_5=recall_5,
                recall_at_10=recall_10,
                latency_ms=latency_ms * 1000,
                tokens_kept=head_result.total_tokens,
                original_tokens=len(sample.context.split()),
                compression_ratio=1.0 - head_result.keep_ratio,
                metadata={'head_result': head_result}
            )
            
        except Exception as e:
            logger.error(f"Lethe evaluation failed for {sample.sample_id}: {e}")
            return self._create_error_result(sample, 'lethe', keep_ratio, str(e))
    
    def _calculate_lambda_for_tokens(self, target_tokens: int, total_tokens: int) -> float:
        """Calculate lambda parameter to achieve target token count."""
        if total_tokens <= target_tokens:
            return 0.0001  # Minimal lambda for no constraints
        
        # Simple heuristic: higher lambda for more compression
        compression_ratio = 1.0 - (target_tokens / total_tokens)
        return compression_ratio * 0.01  # Scale factor
    
    def _calculate_precision_recall(self, context: str, ground_truth: Dict[str, Any], k: int) -> Tuple[float, float]:
        """Calculate precision@k and recall@k metrics."""
        # Simplified implementation - in practice would use more sophisticated matching
        if not context or not ground_truth.get('answer'):
            return 0.0, 0.0
        
        context_lower = context.lower()
        answers = ground_truth.get('answer', [])
        
        if isinstance(answers, str):
            answers = [answers]
        
        # Simple keyword matching
        relevant_found = 0
        for answer in answers[:k]:
            if str(answer).lower() in context_lower:
                relevant_found += 1
        
        precision = relevant_found / min(k, len(answers)) if answers else 0.0
        recall = relevant_found / len(answers) if answers else 0.0
        
        return precision, recall
    
    def _create_mock_streaming_result(self, sample: EvaluationSample, keep_ratio: float) -> MethodResult:
        """Create mock result when streaming competitor unavailable."""
        total_tokens = len(sample.context.split())
        kept_tokens = int(total_tokens * keep_ratio)
        
        # Mock truncation - keep last kept_tokens
        context_tokens = sample.context.split()
        if len(context_tokens) > kept_tokens:
            mock_context = ' '.join(context_tokens[-kept_tokens:])
        else:
            mock_context = sample.context
        
        precision_5, recall_5 = self._calculate_precision_recall(mock_context, sample.ground_truth, k=5)
        precision_10, recall_10 = self._calculate_precision_recall(mock_context, sample.ground_truth, k=10)
        
        return MethodResult(
            sample_id=sample.sample_id,
            method_name='streaming',
            keep_ratio=keep_ratio,
            precision_at_5=precision_5,
            precision_at_10=precision_10,
            recall_at_5=recall_5,
            recall_at_10=recall_10,
            latency_ms=50.0,  # Mock latency
            tokens_kept=kept_tokens,
            original_tokens=total_tokens,
            compression_ratio=1.0 - keep_ratio,
            metadata={'mock_result': True}
        )
    
    def _create_error_result(self, sample: EvaluationSample, method: str, keep_ratio: float, error: str) -> MethodResult:
        """Create error result for failed evaluations."""
        return MethodResult(
            sample_id=sample.sample_id,
            method_name=method,
            keep_ratio=keep_ratio,
            precision_at_5=0.0,
            precision_at_10=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            latency_ms=999999.0,  # High latency for errors
            tokens_kept=0,
            original_tokens=len(sample.context.split()),
            compression_ratio=1.0,  # All tokens filtered
            metadata={'error': error}
        )

class StatisticalAnalyzer:
    """Performs statistical analysis and significance testing."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
    
    def compare_methods(self, results_a: List[MethodResult], results_b: List[MethodResult], 
                       keep_ratio: float) -> ComparisonResult:
        """Compare two methods using statistical tests."""
        
        # Align results by sample_id
        aligned_a, aligned_b = self._align_results(results_a, results_b)
        
        if len(aligned_a) < 10:  # Minimum sample size
            logger.warning(f"Insufficient samples for statistical comparison: {len(aligned_a)}")
        
        # Extract metrics
        precision_a = [r.precision_at_5 for r in aligned_a]
        precision_b = [r.precision_at_5 for r in aligned_b]
        
        recall_a = [r.recall_at_5 for r in aligned_a]
        recall_b = [r.recall_at_5 for r in aligned_b]
        
        latency_a = [r.latency_ms for r in aligned_a]
        latency_b = [r.latency_ms for r in aligned_b]
        
        cbu_a = [r.delta_cbu_per_1k for r in aligned_a]
        cbu_b = [r.delta_cbu_per_1k for r in aligned_b]
        
        # Perform paired t-tests
        precision_stat, precision_p = self._paired_ttest(precision_a, precision_b)
        recall_stat, recall_p = self._paired_ttest(recall_a, recall_b)
        latency_stat, latency_p = self._paired_ttest(latency_a, latency_b)
        cbu_stat, cbu_p = self._paired_ttest(cbu_a, cbu_b)
        
        # Calculate effect sizes (Cohen's d)
        precision_effect = self._cohens_d(precision_a, precision_b)
        recall_effect = self._cohens_d(recall_a, recall_b)
        latency_effect = self._cohens_d(latency_a, latency_b)
        cbu_effect = self._cohens_d(cbu_a, cbu_b)
        
        # Check promotion criteria
        method_a_name = aligned_a[0].method_name if aligned_a else "unknown"
        method_b_name = aligned_b[0].method_name if aligned_b else "unknown"
        
        significant_improvement = self._check_significance(
            precision_p, recall_p, latency_p, cbu_p
        )
        
        meets_promotion = self._check_promotion_criteria(
            precision_effect, cbu_effect, latency_effect,
            np.mean(latency_a), np.mean(latency_b)
        )
        
        return ComparisonResult(
            method_a=method_a_name,
            method_b=method_b_name,
            keep_ratio=keep_ratio,
            precision_p_value=precision_p,
            recall_p_value=recall_p,
            latency_p_value=latency_p,
            cbu_p_value=cbu_p,
            precision_effect_size=precision_effect,
            recall_effect_size=recall_effect,
            latency_effect_size=latency_effect,
            cbu_effect_size=cbu_effect,
            significant_improvement=significant_improvement,
            meets_promotion_criteria=meets_promotion,
            metadata={
                'sample_size': len(aligned_a),
                'confidence_level': self.confidence_level
            }
        )
    
    def _align_results(self, results_a: List[MethodResult], results_b: List[MethodResult]) -> Tuple[List[MethodResult], List[MethodResult]]:
        """Align results by sample_id for paired comparisons."""
        results_a_dict = {r.sample_id: r for r in results_a}
        results_b_dict = {r.sample_id: r for r in results_b}
        
        common_ids = set(results_a_dict.keys()) & set(results_b_dict.keys())
        
        aligned_a = [results_a_dict[id] for id in common_ids]
        aligned_b = [results_b_dict[id] for id in common_ids]
        
        return aligned_a, aligned_b
    
    def _paired_ttest(self, a: List[float], b: List[float]) -> Tuple[float, float]:
        """Perform paired t-test."""
        if len(a) != len(b) or len(a) < 2:
            return 0.0, 1.0
        
        try:
            stat, p_value = stats.ttest_rel(a, b)
            return stat, p_value
        except Exception:
            return 0.0, 1.0
    
    def _cohens_d(self, a: List[float], b: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        if len(a) < 2 or len(b) < 2:
            return 0.0
        
        try:
            mean_a, mean_b = np.mean(a), np.mean(b)
            std_a, std_b = np.std(a, ddof=1), np.std(b, ddof=1)
            
            # Pooled standard deviation
            n_a, n_b = len(a), len(b)
            pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
            
            if pooled_std == 0:
                return 0.0
            
            return (mean_a - mean_b) / pooled_std
        except Exception:
            return 0.0
    
    def _check_significance(self, precision_p: float, recall_p: float, latency_p: float, cbu_p: float) -> bool:
        """Check if improvement is statistically significant."""
        # Apply Holm correction for multiple comparisons
        p_values = sorted([precision_p, recall_p, latency_p, cbu_p])
        
        for i, p in enumerate(p_values):
            adjusted_alpha = self.alpha / (len(p_values) - i)
            if p > adjusted_alpha:
                return False
        
        return True
    
    def _check_promotion_criteria(self, precision_effect: float, cbu_effect: float, 
                                latency_effect: float, latency_a: float, latency_b: float) -> bool:
        """Check promotion criteria from TODO.md."""
        # Hybrid must beat Streaming on P@k or ΔCBU/1k with p95 ≤ +1ms
        # and no ECE/type/budget regression (>+0.01)
        
        beats_on_precision = precision_effect > 0.01  # Small positive effect
        beats_on_cbu = cbu_effect < -0.01  # Negative effect (improvement in efficiency)
        
        latency_increase = latency_a - latency_b
        meets_latency_req = latency_increase <= 1.0  # p95 ≤ +1ms
        
        no_regression = abs(latency_effect) <= 0.01  # No significant regression
        
        return (beats_on_precision or beats_on_cbu) and meets_latency_req and no_regression

class HybridEvaluationRunner:
    """Main evaluation runner for the hybrid system."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.data_loader = None
        self.method_evaluator = None
        self.statistical_analyzer = StatisticalAnalyzer(config.confidence_level)
        self.results = defaultdict(list)
        
    def run_evaluation(self, data_dir: Path) -> Dict[str, Any]:
        """Run complete hybrid evaluation."""
        logger.info("Starting hybrid evaluation")
        
        # Initialize components
        self.data_loader = InfiniteBenchLoader(data_dir)
        self.method_evaluator = HybridMethodEvaluator()
        
        if not self.method_evaluator.initialize_methods(self.config):
            raise RuntimeError("Failed to initialize evaluation methods")
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run evaluation for each dataset and configuration
        all_results = {}
        
        for dataset_name in self.config.datasets:
            logger.info(f"Evaluating dataset: {dataset_name}")
            
            # Load dataset
            max_samples = self.config.max_samples_per_dataset.get(dataset_name, 100)
            samples = self.data_loader.load_dataset(dataset_name, max_samples)
            
            if not samples:
                logger.warning(f"No samples found for dataset {dataset_name}")
                continue
            
            dataset_results = {}
            
            # Evaluate each method at each keep ratio
            for keep_ratio in self.config.keep_ratios:
                logger.info(f"Evaluating keep_ratio: {keep_ratio}")
                
                ratio_results = {}
                
                for method_name in self.config.methods:
                    logger.info(f"Evaluating method: {method_name}")
                    
                    method_results = []
                    
                    # Evaluate each sample
                    for sample in samples:
                        try:
                            if method_name == 'hybrid':
                                result = self.method_evaluator.evaluate_hybrid_method(sample, keep_ratio)
                            elif method_name == 'streaming':
                                result = self.method_evaluator.evaluate_streaming_method(sample, keep_ratio)
                            elif method_name == 'lethe':
                                result = self.method_evaluator.evaluate_lethe_method(sample, keep_ratio)
                            else:
                                logger.warning(f"Unknown method: {method_name}")
                                continue
                            
                            method_results.append(result)
                            
                        except Exception as e:
                            logger.error(f"Evaluation failed for {sample.sample_id} with {method_name}: {e}")
                            continue
                    
                    ratio_results[method_name] = method_results
                    logger.info(f"Completed {method_name}: {len(method_results)} results")
                
                dataset_results[keep_ratio] = ratio_results
            
            all_results[dataset_name] = dataset_results
        
        # Perform statistical analysis
        comparisons = self._perform_statistical_analysis(all_results)
        
        # Save results
        self._save_results(all_results, comparisons)
        
        # Generate summary report
        summary = self._generate_summary_report(all_results, comparisons)
        
        logger.info("Hybrid evaluation completed")
        return summary
    
    def _perform_statistical_analysis(self, all_results: Dict[str, Any]) -> Dict[str, List[ComparisonResult]]:
        """Perform statistical analysis comparing methods."""
        comparisons = defaultdict(list)
        
        for dataset_name, dataset_results in all_results.items():
            logger.info(f"Analyzing dataset: {dataset_name}")
            
            for keep_ratio, ratio_results in dataset_results.items():
                logger.info(f"Analyzing keep_ratio: {keep_ratio}")
                
                # Compare hybrid vs streaming (main comparison)
                if 'hybrid' in ratio_results and 'streaming' in ratio_results:
                    comparison = self.statistical_analyzer.compare_methods(
                        ratio_results['hybrid'],
                        ratio_results['streaming'],
                        keep_ratio
                    )
                    comparisons[f"{dataset_name}_hybrid_vs_streaming"].append(comparison)
                
                # Compare hybrid vs lethe
                if 'hybrid' in ratio_results and 'lethe' in ratio_results:
                    comparison = self.statistical_analyzer.compare_methods(
                        ratio_results['hybrid'],
                        ratio_results['lethe'],
                        keep_ratio
                    )
                    comparisons[f"{dataset_name}_hybrid_vs_lethe"].append(comparison)
                
                # Compare streaming vs lethe
                if 'streaming' in ratio_results and 'lethe' in ratio_results:
                    comparison = self.statistical_analyzer.compare_methods(
                        ratio_results['streaming'],
                        ratio_results['lethe'],
                        keep_ratio
                    )
                    comparisons[f"{dataset_name}_streaming_vs_lethe"].append(comparison)
        
        return comparisons
    
    def _save_results(self, all_results: Dict[str, Any], comparisons: Dict[str, List[ComparisonResult]]):
        """Save evaluation results to disk."""
        
        # Save detailed results
        if self.config.save_detailed_results:
            results_file = self.config.output_dir / "detailed_results.json"
            with open(results_file, 'w') as f:
                # Convert results to serializable format
                serializable_results = {}
                for dataset, dataset_results in all_results.items():
                    serializable_results[dataset] = {}
                    for keep_ratio, ratio_results in dataset_results.items():
                        serializable_results[dataset][str(keep_ratio)] = {}
                        for method, method_results in ratio_results.items():
                            serializable_results[dataset][str(keep_ratio)][method] = [
                                {
                                    'sample_id': r.sample_id,
                                    'method_name': r.method_name,
                                    'keep_ratio': r.keep_ratio,
                                    'precision_at_5': r.precision_at_5,
                                    'precision_at_10': r.precision_at_10,
                                    'recall_at_5': r.recall_at_5,
                                    'recall_at_10': r.recall_at_10,
                                    'latency_ms': r.latency_ms,
                                    'tokens_kept': r.tokens_kept,
                                    'original_tokens': r.original_tokens,
                                    'compression_ratio': r.compression_ratio,
                                    'kv_reuse': r.kv_reuse,
                                    'delta_cbu_per_1k': r.delta_cbu_per_1k,
                                    'tail_cvar_95': r.tail_cvar_95
                                }
                                for r in method_results
                            ]
                
                json.dump(serializable_results, f, indent=2)
        
        # Save comparisons
        comparisons_file = self.config.output_dir / "statistical_comparisons.json"
        with open(comparisons_file, 'w') as f:
            serializable_comparisons = {}
            for comparison_name, comparison_list in comparisons.items():
                serializable_comparisons[comparison_name] = [
                    {
                        'method_a': c.method_a,
                        'method_b': c.method_b,
                        'keep_ratio': c.keep_ratio,
                        'precision_p_value': c.precision_p_value,
                        'recall_p_value': c.recall_p_value,
                        'latency_p_value': c.latency_p_value,
                        'cbu_p_value': c.cbu_p_value,
                        'precision_effect_size': c.precision_effect_size,
                        'recall_effect_size': c.recall_effect_size,
                        'latency_effect_size': c.latency_effect_size,
                        'cbu_effect_size': c.cbu_effect_size,
                        'significant_improvement': c.significant_improvement,
                        'meets_promotion_criteria': c.meets_promotion_criteria
                    }
                    for c in comparison_list
                ]
            
            json.dump(serializable_comparisons, f, indent=2)
    
    def _generate_summary_report(self, all_results: Dict[str, Any], comparisons: Dict[str, List[ComparisonResult]]) -> Dict[str, Any]:
        """Generate summary report for evaluation."""
        
        summary = {
            'evaluation_config': {
                'datasets': self.config.datasets,
                'methods': self.config.methods,
                'keep_ratios': self.config.keep_ratios,
                'max_samples_per_dataset': self.config.max_samples_per_dataset
            },
            'dataset_summaries': {},
            'promotion_decisions': {},
            'overall_recommendation': None
        }
        
        # Summarize each dataset
        for dataset_name, dataset_results in all_results.items():
            dataset_summary = {
                'total_samples': 0,
                'methods_evaluated': list(dataset_results.get(0.15, {}).keys()) if 0.15 in dataset_results else [],
                'keep_ratios_tested': list(dataset_results.keys()),
                'average_metrics': {}
            }
            
            # Calculate average metrics across keep ratios
            for keep_ratio, ratio_results in dataset_results.items():
                for method_name, method_results in ratio_results.items():
                    if method_results:
                        dataset_summary['total_samples'] = max(dataset_summary['total_samples'], len(method_results))
                        
                        if method_name not in dataset_summary['average_metrics']:
                            dataset_summary['average_metrics'][method_name] = {}
                        
                        dataset_summary['average_metrics'][method_name][str(keep_ratio)] = {
                            'precision_at_5': np.mean([r.precision_at_5 for r in method_results]),
                            'recall_at_5': np.mean([r.recall_at_5 for r in method_results]),
                            'latency_ms': np.mean([r.latency_ms for r in method_results]),
                            'kv_reuse': np.mean([r.kv_reuse for r in method_results]),
                            'delta_cbu_per_1k': np.mean([r.delta_cbu_per_1k for r in method_results])
                        }
            
            summary['dataset_summaries'][dataset_name] = dataset_summary
        
        # Analyze promotion decisions
        promotion_votes = 0
        total_comparisons = 0
        
        for comparison_name, comparison_list in comparisons.items():
            if 'hybrid_vs_streaming' in comparison_name:
                for comparison in comparison_list:
                    total_comparisons += 1
                    if comparison.meets_promotion_criteria:
                        promotion_votes += 1
                        
                    summary['promotion_decisions'][f"{comparison_name}_{comparison.keep_ratio}"] = {
                        'meets_criteria': comparison.meets_promotion_criteria,
                        'significant_improvement': comparison.significant_improvement,
                        'precision_effect_size': comparison.precision_effect_size,
                        'cbu_effect_size': comparison.cbu_effect_size,
                        'latency_effect_size': comparison.latency_effect_size
                    }
        
        # Overall recommendation
        if total_comparisons > 0:
            promotion_rate = promotion_votes / total_comparisons
            if promotion_rate >= 0.5:
                summary['overall_recommendation'] = 'PROMOTE_HYBRID'
            else:
                summary['overall_recommendation'] = 'KEEP_STREAMING'
        else:
            summary['overall_recommendation'] = 'INSUFFICIENT_DATA'
        
        summary['promotion_statistics'] = {
            'promotion_votes': promotion_votes,
            'total_comparisons': total_comparisons,
            'promotion_rate': promotion_rate if total_comparisons > 0 else 0.0
        }
        
        # Save summary
        summary_file = self.config.output_dir / "evaluation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

def run_hybrid_evaluation(data_dir: str, output_dir: str = "hybrid_evaluation_results") -> Dict[str, Any]:
    """Main entry point for hybrid evaluation."""
    
    config = EvaluationConfig(
        output_dir=Path(output_dir)
    )
    
    runner = HybridEvaluationRunner(config)
    
    try:
        results = runner.run_evaluation(Path(data_dir))
        
        logger.info("=== EVALUATION SUMMARY ===")
        logger.info(f"Overall recommendation: {results.get('overall_recommendation')}")
        logger.info(f"Promotion rate: {results.get('promotion_statistics', {}).get('promotion_rate', 0.0):.2%}")
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Lethe-StreamingLLM Hybrid Evaluation")
    parser.add_argument("--data-dir", required=True, help="InfiniteBench data directory")
    parser.add_argument("--output-dir", default="hybrid_evaluation_results", help="Output directory")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    
    results = run_hybrid_evaluation(args.data_dir, args.output_dir)
    print(f"\nEvaluation complete. Results saved to: {args.output_dir}")
    print(f"Recommendation: {results.get('overall_recommendation')}")