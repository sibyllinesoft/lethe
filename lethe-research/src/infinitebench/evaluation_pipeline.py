"""
InfiniteBench Evaluation Pipeline
===============================

Comprehensive evaluation pipeline for running InfiniteBench experiments
with Lethe and baseline methods. Includes experiment management, result
tracking, and integration with statistical analysis frameworks.

Author: Lethe Research Team
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from .dataset_loader import InfiniteBenchLoader, InfiniteBenchSample
from .baselines import BaselineMethod, BM25Baseline, NaiveChunkingBaseline, DenseRetrievalBaseline
from .metrics import InfiniteBenchMetrics, EvaluationSummary

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for InfiniteBench experiments."""
    
    # Dataset configuration
    tasks: List[str]  # List of task names to evaluate
    
    # Method configuration  
    baseline_methods: List[str]  # List of baseline method names
    
    max_samples_per_task: Optional[int] = None
    data_split: str = "test"
    include_lethe: bool = True
    
    # Evaluation configuration
    max_context_tokens: int = 32000
    max_answer_tokens: int = 1000
    evaluation_timeout_seconds: int = 300
    
    # Experiment metadata
    experiment_name: str = "infinitebench_evaluation"
    description: str = ""
    random_seed: int = 42
    
    # Output configuration
    output_dir: Path = Path("results/infinitebench")
    save_predictions: bool = True
    save_intermediate: bool = True

@dataclass
class MethodResult:
    """Result from evaluating a single method on a task."""
    
    method_name: str
    task_name: str
    evaluation_summary: EvaluationSummary
    predictions: List[str]
    processing_times: List[float]
    errors: List[str]
    metadata: Dict[str, Any]

@dataclass
class ExperimentResult:
    """Complete results from an InfiniteBench experiment."""
    
    config: ExperimentConfig
    method_results: List[MethodResult]
    start_time: str
    end_time: str
    total_duration_seconds: float
    summary_statistics: Dict[str, Any]

class InfiniteBenchEvaluator:
    """
    Main evaluation pipeline for InfiniteBench experiments.
    
    Features:
    - Multi-task evaluation across all 12 InfiniteBench tasks
    - Parallel execution for faster evaluation
    - Integration with Lethe and baseline methods
    - Comprehensive result tracking and analysis
    - Statistical analysis integration
    - Academic publication reporting
    """
    
    def __init__(self, 
                 data_dir: Union[str, Path],
                 lethe_pipeline: Optional[Callable] = None):
        """
        Initialize InfiniteBench evaluator.
        
        Args:
            data_dir: Directory containing InfiniteBench data
            lethe_pipeline: Optional Lethe pipeline function for evaluation
        """
        self.data_dir = Path(data_dir)
        self.loader = InfiniteBenchLoader(self.data_dir)
        self.metrics = InfiniteBenchMetrics()
        self.lethe_pipeline = lethe_pipeline
    
    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a complete InfiniteBench experiment.
        
        Args:
            config: Experiment configuration
            
        Returns:
            ExperimentResult with comprehensive results
        """
        logger.info(f"Starting InfiniteBench experiment: {config.experiment_name}")
        start_time = time.time()
        
        # Create output directory
        config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        logger.info("Loading InfiniteBench dataset...")
        task_datasets = self._load_experiment_data(config)
        
        # Initialize methods
        logger.info("Initializing evaluation methods...")
        methods = self._initialize_methods(config)
        
        # Run evaluation for each method and task
        logger.info(f"Running evaluation on {len(methods)} methods and {len(task_datasets)} tasks...")
        method_results = []
        
        total_evaluations = len(methods) * len(task_datasets)
        completed = 0
        
        for method in methods:
            for task_name, samples in task_datasets.items():
                try:
                    logger.info(f"Evaluating {method.name} on {task_name} ({completed+1}/{total_evaluations})")
                    
                    result = self._evaluate_method_on_task(
                        method=method,
                        task_name=task_name,
                        samples=samples,
                        config=config
                    )
                    method_results.append(result)
                    
                    # Save intermediate results
                    if config.save_intermediate:
                        self._save_intermediate_result(result, config.output_dir)
                    
                    completed += 1
                    logger.info(f"Completed {method.name} on {task_name}: {result.evaluation_summary.overall_score:.3f}")
                    
                except Exception as e:
                    logger.error(f"Failed to evaluate {method.name} on {task_name}: {e}")
                    logger.error(traceback.format_exc())
        
        # Calculate experiment summary
        end_time = time.time()
        duration = end_time - start_time
        
        summary_stats = self._calculate_experiment_summary(method_results)
        
        # Create final result
        experiment_result = ExperimentResult(
            config=config,
            method_results=method_results,
            start_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
            end_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time)),
            total_duration_seconds=duration,
            summary_statistics=summary_stats
        )
        
        # Save final results
        self._save_experiment_result(experiment_result, config.output_dir)
        
        logger.info(f"Experiment completed in {duration:.1f} seconds")
        logger.info(f"Results saved to: {config.output_dir}")
        
        return experiment_result
    
    def _load_experiment_data(self, config: ExperimentConfig) -> Dict[str, List[InfiniteBenchSample]]:
        """Load dataset for experiment based on configuration."""
        
        task_datasets = {}
        
        for task_name in config.tasks:
            try:
                samples = self.loader.load_task(task_name, split=config.data_split)
                
                # Limit samples if specified
                if config.max_samples_per_task and len(samples) > config.max_samples_per_task:
                    import random
                    random.seed(config.random_seed)
                    samples = random.sample(samples, config.max_samples_per_task)
                
                task_datasets[task_name] = samples
                logger.info(f"Loaded {len(samples)} samples for task '{task_name}'")
                
            except Exception as e:
                logger.error(f"Failed to load task '{task_name}': {e}")
        
        return task_datasets
    
    def _initialize_methods(self, config: ExperimentConfig) -> List[BaselineMethod]:
        """Initialize evaluation methods based on configuration."""
        
        methods = []
        
        # Add baseline methods
        for method_name in config.baseline_methods:
            if method_name.lower() == "bm25":
                methods.append(BM25Baseline())
            elif method_name.lower() == "naive_first":
                methods.append(NaiveChunkingBaseline(strategy="first"))
            elif method_name.lower() == "naive_random":
                methods.append(NaiveChunkingBaseline(strategy="random"))
            elif method_name.lower() == "naive_uniform":
                methods.append(NaiveChunkingBaseline(strategy="uniform"))
            elif method_name.lower() == "dense_retrieval":
                methods.append(DenseRetrievalBaseline())
            else:
                logger.warning(f"Unknown baseline method: {method_name}")
        
        # Add Lethe if requested
        if config.include_lethe and self.lethe_pipeline:
            methods.append(LetheMethod(self.lethe_pipeline))
        
        return methods
    
    def _evaluate_method_on_task(self, 
                                method: BaselineMethod,
                                task_name: str,
                                samples: List[InfiniteBenchSample],
                                config: ExperimentConfig) -> MethodResult:
        """Evaluate a single method on a task."""
        
        predictions = []
        processing_times = []
        errors = []
        
        for sample in samples:
            try:
                start_time = time.time()
                
                # Get query (question if available, otherwise use context directly for retrieval tasks)
                query = sample.question or "Extract the relevant information"
                
                # Run method
                if isinstance(method, LetheMethod):
                    # Special handling for Lethe
                    prediction = method.retrieve(query, sample.context, config.max_answer_tokens)
                    prediction = prediction.context_used  # Use retrieved context as prediction
                else:
                    # Baseline method
                    result = method.retrieve(query, sample.context, config.max_context_tokens)
                    
                    # For retrieval tasks, use the retrieved context
                    # For Q&A tasks, we would need to add another step to generate answers
                    prediction = result.context_used
                
                processing_time = (time.time() - start_time) * 1000
                
                predictions.append(prediction)
                processing_times.append(processing_time)
                
            except Exception as e:
                error_msg = f"Error processing sample {sample.id}: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)
                
                # Add empty prediction to maintain alignment
                predictions.append("")
                processing_times.append(0.0)
        
        # Extract references for evaluation
        references = [sample.answer or "" for sample in samples]
        contexts = [sample.context for sample in samples]
        
        # Evaluate predictions
        evaluation_summary = self.metrics.evaluate_task(
            predictions=predictions,
            references=references,
            task_name=task_name,
            contexts=contexts
        )
        
        return MethodResult(
            method_name=method.name,
            task_name=task_name,
            evaluation_summary=evaluation_summary,
            predictions=predictions,
            processing_times=processing_times,
            errors=errors,
            metadata={
                "num_samples": len(samples),
                "avg_processing_time_ms": sum(processing_times) / len(processing_times) if processing_times else 0,
                "num_errors": len([e for e in errors if e]),
                "error_rate": len([e for e in errors if e]) / len(samples) if samples else 0
            }
        )
    
    def _calculate_experiment_summary(self, method_results: List[MethodResult]) -> Dict[str, Any]:
        """Calculate summary statistics for the experiment."""
        
        if not method_results:
            return {}
        
        # Group results by method and task
        results_by_method = {}
        results_by_task = {}
        
        for result in method_results:
            if result.method_name not in results_by_method:
                results_by_method[result.method_name] = []
            results_by_method[result.method_name].append(result)
            
            if result.task_name not in results_by_task:
                results_by_task[result.task_name] = []
            results_by_task[result.task_name].append(result)
        
        # Calculate method averages
        method_averages = {}
        for method_name, results in results_by_method.items():
            scores = [r.evaluation_summary.overall_score for r in results]
            method_averages[method_name] = {
                "mean_score": sum(scores) / len(scores),
                "num_tasks": len(scores),
                "scores": scores
            }
        
        # Calculate task averages  
        task_averages = {}
        for task_name, results in results_by_task.items():
            scores = [r.evaluation_summary.overall_score for r in results]
            task_averages[task_name] = {
                "mean_score": sum(scores) / len(scores),
                "num_methods": len(scores),
                "scores": scores
            }
        
        # Overall statistics
        all_scores = [r.evaluation_summary.overall_score for r in method_results]
        overall_stats = {
            "mean_score": sum(all_scores) / len(all_scores),
            "min_score": min(all_scores),
            "max_score": max(all_scores),
            "num_evaluations": len(all_scores)
        }
        
        return {
            "method_averages": method_averages,
            "task_averages": task_averages,
            "overall_statistics": overall_stats
        }
    
    def _save_intermediate_result(self, result: MethodResult, output_dir: Path):
        """Save intermediate result for a method-task combination."""
        
        filename = f"{result.method_name}_{result.task_name}_result.json"
        filepath = output_dir / "intermediate" / filename
        filepath.parent.mkdir(exist_ok=True)
        
        # Convert to serializable format
        result_dict = {
            "method_name": result.method_name,
            "task_name": result.task_name,
            "overall_score": result.evaluation_summary.overall_score,
            "metric_scores": {
                name: metric.score 
                for name, metric in result.evaluation_summary.metric_results.items()
            },
            "num_samples": result.evaluation_summary.num_samples,
            "metadata": result.metadata,
            "errors": result.errors
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
    
    def _save_experiment_result(self, experiment_result: ExperimentResult, output_dir: Path):
        """Save complete experiment results."""
        
        # Save main results file
        results_file = output_dir / f"{experiment_result.config.experiment_name}_results.json"
        
        # Convert to serializable format
        results_dict = {
            "config": asdict(experiment_result.config),
            "start_time": experiment_result.start_time,
            "end_time": experiment_result.end_time, 
            "duration_seconds": experiment_result.total_duration_seconds,
            "summary_statistics": experiment_result.summary_statistics,
            "method_results": []
        }
        
        # Convert method results
        for result in experiment_result.method_results:
            method_dict = {
                "method_name": result.method_name,
                "task_name": result.task_name,
                "overall_score": result.evaluation_summary.overall_score,
                "metric_scores": {
                    name: metric.score 
                    for name, metric in result.evaluation_summary.metric_results.items()
                },
                "num_samples": result.evaluation_summary.num_samples,
                "metadata": result.metadata
            }
            results_dict["method_results"].append(method_dict)
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save predictions if requested
        if experiment_result.config.save_predictions:
            predictions_dir = output_dir / "predictions"
            predictions_dir.mkdir(exist_ok=True)
            
            for result in experiment_result.method_results:
                pred_file = predictions_dir / f"{result.method_name}_{result.task_name}_predictions.json"
                pred_data = {
                    "method_name": result.method_name,
                    "task_name": result.task_name,
                    "predictions": result.predictions,
                    "processing_times": result.processing_times
                }
                
                with open(pred_file, 'w') as f:
                    json.dump(pred_data, f, indent=2)
        
        logger.info(f"Experiment results saved to {results_file}")

class LetheMethod(BaselineMethod):
    """Wrapper for Lethe pipeline to fit into baseline evaluation framework."""
    
    def __init__(self, lethe_pipeline: Callable):
        super().__init__("Lethe")
        self.pipeline = lethe_pipeline
    
    def retrieve(self, query: str, context: str, max_tokens: int = 4000):
        """Use Lethe pipeline for retrieval."""
        import time
        start_time = time.time()
        
        try:
            # Call Lethe pipeline
            result = self.pipeline(query, context, max_tokens=max_tokens)
            
            processing_time = (time.time() - start_time) * 1000
            
            return type('RetrievalResult', (), {
                'query_id': '',
                'retrieved_chunks': [(result, 1.0)],
                'context_used': result,
                'processing_time_ms': processing_time,
                'metadata': {'method': 'Lethe', 'tokens_used': self.count_tokens(result)}
            })()
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            return type('RetrievalResult', (), {
                'query_id': '',
                'retrieved_chunks': [('', 0.0)],
                'context_used': '',
                'processing_time_ms': processing_time,
                'metadata': {'method': 'Lethe', 'error': str(e)}
            })()

def main():
    """Example usage of InfiniteBench evaluation pipeline."""
    
    # Configuration for a small test experiment
    config = ExperimentConfig(
        experiment_name="infinitebench_test",
        description="Test experiment with subset of tasks and baselines",
        tasks=["kv_retrieval", "longbook_qa_eng"],
        max_samples_per_task=10,
        baseline_methods=["bm25", "naive_first"],
        include_lethe=False,  # Would need actual Lethe pipeline
        output_dir=Path("results/infinitebench_test")
    )
    
    # Initialize evaluator
    data_dir = Path("benchmarks/infinitebench/data")
    evaluator = InfiniteBenchEvaluator(data_dir)
    
    # Run experiment
    try:
        result = evaluator.run_experiment(config)
        
        print("\nExperiment Results Summary:")
        print(f"Experiment: {result.config.experiment_name}")
        print(f"Duration: {result.total_duration_seconds:.1f} seconds")
        print(f"Total evaluations: {len(result.method_results)}")
        
        print("\nMethod Performance:")
        for method_name, stats in result.summary_statistics["method_averages"].items():
            print(f"  {method_name}: {stats['mean_score']:.3f} (avg across {stats['num_tasks']} tasks)")
        
        print("\nTask Difficulty:")
        for task_name, stats in result.summary_statistics["task_averages"].items():
            print(f"  {task_name}: {stats['mean_score']:.3f} (avg across {stats['num_methods']} methods)")
            
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()