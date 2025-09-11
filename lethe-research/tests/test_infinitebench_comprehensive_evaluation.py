"""
Comprehensive tests for InfiniteBench evaluation orchestrator.

Tests cover the high-complexity ComprehensiveEvaluationOrchestrator which
valknut identified as having significant complexity in evaluation coordination.

Test areas:
- Baseline method orchestration and coordination
- Extended task management and execution
- External benchmark integration
- Publication protocol compliance
- Statistical analysis and reporting
- Asynchronous evaluation workflows
- Error handling and recovery
- Performance monitoring and optimization
- Configuration validation and management
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass

# Import the module under test
try:
    from src.infinitebench.comprehensive_evaluation import (
        ComprehensiveEvaluationOrchestrator,
        EvaluationResults, EvaluationError
    )
    from src.infinitebench.comprehensive_baselines import (
        ComprehensiveConfig, ComprehensiveBaselineMethod
    )
    from src.infinitebench.publication_protocol import (
        EvaluationProtocol, PublicationEvaluator
    )
except ImportError:
    # Handle missing dependencies gracefully
    pytest.skip("infinitebench modules not available", allow_module_level=True)


class TestComprehensiveEvaluationOrchestrator:
    """Test suite for ComprehensiveEvaluationOrchestrator functionality."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for evaluation outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def mock_baseline_config(self):
        """Mock comprehensive baseline configuration."""
        return ComprehensiveConfig(
            baseline_families=["lethe", "traditional_rag", "context_compression", "learned_sparse"],
            enable_hybrid_methods=True,
            max_sequence_length=32000,
            batch_size=4,
            enable_gpu_acceleration=False,
            evaluation_timeout=300
        )
    
    @pytest.fixture
    def mock_evaluation_protocol(self):
        """Mock evaluation protocol configuration."""
        protocol = Mock(spec=EvaluationProtocol)
        protocol.statistical_tests = ["t_test", "wilcoxon"]
        protocol.confidence_level = 0.95
        protocol.multiple_runs = 3
        protocol.validation_strategy = "cross_validation"
        return protocol
    
    @pytest.fixture
    def orchestrator(self, mock_baseline_config, mock_evaluation_protocol, temp_output_dir):
        """Create orchestrator instance for testing."""
        return ComprehensiveEvaluationOrchestrator(
            baseline_config=mock_baseline_config,
            evaluation_protocol=mock_evaluation_protocol,
            output_dir=temp_output_dir
        )
    
    @pytest.fixture
    def sample_tasks(self):
        """Sample evaluation tasks."""
        return [
            "retrieve.needle_in_haystack",
            "retrieve.number_string",
            "code.debug_hard",
            "math.single_equation",
            "qa.hotpotqa"
        ]
    
    @pytest.fixture
    def mock_baseline_methods(self):
        """Mock baseline methods for testing."""
        methods = {}
        method_names = [
            "lethe_hybrid", "lethe_semantic", "lethe_lexical",
            "bm25_baseline", "tfidf_baseline",
            "llmlingua", "selective_context",
            "splade", "colbert"
        ]
        
        for name in method_names:
            method = Mock(spec=ComprehensiveBaselineMethod)
            method.name = name
            method.evaluate = AsyncMock(return_value={
                "accuracy": 0.75 + hash(name) % 100 / 400,  # Vary scores
                "latency_ms": 100 + hash(name) % 50,
                "memory_mb": 50 + hash(name) % 30
            })
            methods[name] = method
        
        return methods

    # Initialization and configuration tests
    def test_orchestrator_initialization(self, mock_baseline_config, 
                                        mock_evaluation_protocol, temp_output_dir):
        """Test orchestrator initialization with various configurations."""
        orchestrator = ComprehensiveEvaluationOrchestrator(
            baseline_config=mock_baseline_config,
            evaluation_protocol=mock_evaluation_protocol,
            output_dir=temp_output_dir
        )
        
        assert orchestrator.baseline_config == mock_baseline_config
        assert orchestrator.evaluation_protocol == mock_evaluation_protocol
        assert orchestrator.output_dir == temp_output_dir
        assert orchestrator._baseline_methods == {}
        assert orchestrator._task_registry == {}
    
    def test_configuration_validation(self, mock_evaluation_protocol, temp_output_dir):
        """Test configuration validation on initialization."""
        # Invalid baseline config
        invalid_config = ComprehensiveConfig(
            baseline_families=[],  # Empty families
            max_sequence_length=0,  # Invalid length
            batch_size=-1  # Invalid batch size
        )
        
        with pytest.raises(ValueError, match="baseline families cannot be empty"):
            ComprehensiveEvaluationOrchestrator(
                baseline_config=invalid_config,
                evaluation_protocol=mock_evaluation_protocol,
                output_dir=temp_output_dir
            )
    
    def test_output_directory_creation(self, mock_baseline_config, 
                                      mock_evaluation_protocol, temp_output_dir):
        """Test automatic output directory creation."""
        nonexistent_dir = temp_output_dir / "new_evaluation"
        
        orchestrator = ComprehensiveEvaluationOrchestrator(
            baseline_config=mock_baseline_config,
            evaluation_protocol=mock_evaluation_protocol,
            output_dir=nonexistent_dir
        )
        
        assert nonexistent_dir.exists()
        assert (nonexistent_dir / "logs").exists()
        assert (nonexistent_dir / "results").exists()

    # Baseline method management tests
    @patch('src.infinitebench.comprehensive_baselines.ComprehensiveBaselineFactory')
    def test_baseline_method_registration(self, mock_factory, orchestrator):
        """Test registration and initialization of baseline methods."""
        mock_methods = {
            "lethe_hybrid": Mock(),
            "bm25_baseline": Mock(),
            "llmlingua": Mock()
        }
        mock_factory.create_all_baselines.return_value = mock_methods
        
        orchestrator.initialize_baseline_methods()
        
        assert len(orchestrator._baseline_methods) == 3
        for name, method in mock_methods.items():
            assert orchestrator._baseline_methods[name] == method
    
    @patch('src.infinitebench.comprehensive_baselines.ComprehensiveBaselineFactory')
    def test_baseline_method_filtering(self, mock_factory, orchestrator):
        """Test filtering baseline methods by configuration."""
        all_methods = {
            "lethe_hybrid": Mock(),
            "bm25_baseline": Mock(),
            "splade": Mock(),
            "colbert": Mock()
        }
        mock_factory.create_all_baselines.return_value = all_methods
        
        # Configure to only use specific families
        orchestrator.baseline_config.baseline_families = ["lethe", "traditional_rag"]
        orchestrator.baseline_config.method_filter = ["lethe_hybrid", "bm25_baseline"]
        
        orchestrator.initialize_baseline_methods()
        
        # Should only include filtered methods
        assert len(orchestrator._baseline_methods) == 2
        assert "lethe_hybrid" in orchestrator._baseline_methods
        assert "bm25_baseline" in orchestrator._baseline_methods

    # Task management tests
    @patch('src.infinitebench.extended_tasks.ExtendedTaskFactory')
    def test_task_registration(self, mock_task_factory, orchestrator, sample_tasks):
        """Test task registration and initialization."""
        mock_tasks = {}
        for task_name in sample_tasks:
            task = Mock()
            task.name = task_name
            task.difficulty = "medium"
            task.categories = ["retrieval", "reasoning"]
            mock_tasks[task_name] = task
        
        mock_task_factory.create_tasks.return_value = mock_tasks
        
        orchestrator.initialize_tasks(sample_tasks)
        
        assert len(orchestrator._task_registry) == len(sample_tasks)
        for task_name in sample_tasks:
            assert task_name in orchestrator._task_registry
    
    def test_task_filtering_by_difficulty(self, orchestrator):
        """Test filtering tasks by difficulty level."""
        # Mock tasks with different difficulties
        mock_tasks = {
            "easy_task": Mock(name="easy_task", difficulty="easy"),
            "medium_task": Mock(name="medium_task", difficulty="medium"),
            "hard_task": Mock(name="hard_task", difficulty="hard")
        }
        orchestrator._task_registry = mock_tasks
        
        # Filter by difficulty
        filtered_tasks = orchestrator.filter_tasks_by_difficulty("medium")
        
        assert len(filtered_tasks) == 1
        assert "medium_task" in filtered_tasks
    
    def test_task_filtering_by_category(self, orchestrator):
        """Test filtering tasks by category."""
        mock_tasks = {
            "retrieval_task": Mock(name="retrieval_task", categories=["retrieval"]),
            "reasoning_task": Mock(name="reasoning_task", categories=["reasoning"]),
            "mixed_task": Mock(name="mixed_task", categories=["retrieval", "reasoning"])
        }
        orchestrator._task_registry = mock_tasks
        
        # Filter by category
        retrieval_tasks = orchestrator.filter_tasks_by_category("retrieval")
        
        assert len(retrieval_tasks) == 2
        assert "retrieval_task" in retrieval_tasks
        assert "mixed_task" in retrieval_tasks

    # Evaluation execution tests
    @pytest.mark.asyncio
    async def test_single_method_evaluation(self, orchestrator, mock_baseline_methods):
        """Test evaluation of a single baseline method."""
        orchestrator._baseline_methods = mock_baseline_methods
        task_name = "retrieve.needle_in_haystack"
        method_name = "lethe_hybrid"
        
        # Mock task
        mock_task = Mock()
        mock_task.evaluate = AsyncMock(return_value={
            "accuracy": 0.85,
            "retrieval_precision": 0.9,
            "latency_ms": 120
        })
        orchestrator._task_registry = {task_name: mock_task}
        
        result = await orchestrator.evaluate_method_on_task(method_name, task_name)
        
        assert result["method"] == method_name
        assert result["task"] == task_name
        assert "accuracy" in result["metrics"]
        assert "latency_ms" in result["metrics"]
    
    @pytest.mark.asyncio
    async def test_batch_evaluation(self, orchestrator, mock_baseline_methods, sample_tasks):
        """Test batch evaluation of multiple methods on multiple tasks."""
        orchestrator._baseline_methods = mock_baseline_methods
        
        # Mock tasks
        mock_tasks = {}
        for task_name in sample_tasks:
            task = Mock()
            task.evaluate = AsyncMock(return_value={
                "accuracy": 0.7 + hash(task_name) % 100 / 500,
                "latency_ms": 100 + hash(task_name) % 50
            })
            mock_tasks[task_name] = task
        orchestrator._task_registry = mock_tasks
        
        # Select subset for testing
        test_methods = ["lethe_hybrid", "bm25_baseline", "llmlingua"]
        test_tasks = sample_tasks[:3]
        
        results = await orchestrator.run_comprehensive_evaluation(
            methods=test_methods,
            tasks=test_tasks
        )
        
        assert len(results) == len(test_methods) * len(test_tasks)
        
        # Verify structure of results
        for result in results:
            assert "method" in result
            assert "task" in result
            assert "metrics" in result
            assert result["method"] in test_methods
            assert result["task"] in test_tasks
    
    @pytest.mark.asyncio
    async def test_evaluation_with_timeout(self, orchestrator, mock_baseline_methods):
        """Test evaluation timeout handling."""
        orchestrator._baseline_methods = mock_baseline_methods
        
        # Mock slow task
        slow_task = Mock()
        slow_task.evaluate = AsyncMock(side_effect=asyncio.TimeoutError())
        orchestrator._task_registry = {"slow_task": slow_task}
        
        orchestrator.baseline_config.evaluation_timeout = 1  # 1 second timeout
        
        with pytest.raises(EvaluationError, match="timeout"):
            await orchestrator.evaluate_method_on_task("lethe_hybrid", "slow_task")
    
    @pytest.mark.asyncio
    async def test_evaluation_error_handling(self, orchestrator, mock_baseline_methods):
        """Test evaluation error handling and recovery."""
        orchestrator._baseline_methods = mock_baseline_methods
        
        # Mock failing task
        failing_task = Mock()
        failing_task.evaluate = AsyncMock(side_effect=Exception("Task failed"))
        orchestrator._task_registry = {"failing_task": failing_task}
        
        # Should handle gracefully with error logging
        result = await orchestrator.evaluate_method_on_task(
            "lethe_hybrid", "failing_task", handle_errors=True
        )
        
        assert result["status"] == "error"
        assert "Task failed" in result["error_message"]

    # Statistical analysis tests
    def test_results_aggregation(self, orchestrator):
        """Test aggregation of evaluation results."""
        # Mock evaluation results
        raw_results = [
            {"method": "lethe_hybrid", "task": "task1", "metrics": {"accuracy": 0.85}},
            {"method": "lethe_hybrid", "task": "task2", "metrics": {"accuracy": 0.90}},
            {"method": "bm25_baseline", "task": "task1", "metrics": {"accuracy": 0.75}},
            {"method": "bm25_baseline", "task": "task2", "metrics": {"accuracy": 0.80}}
        ]
        
        aggregated = orchestrator.aggregate_results(raw_results)
        
        # Should group by method
        assert "lethe_hybrid" in aggregated
        assert "bm25_baseline" in aggregated
        
        # Should compute statistics
        lethe_stats = aggregated["lethe_hybrid"]["accuracy"]
        assert lethe_stats["mean"] == 0.875  # (0.85 + 0.90) / 2
        assert lethe_stats["std"] > 0
        assert lethe_stats["min"] == 0.85
        assert lethe_stats["max"] == 0.90
    
    @patch('src.infinitebench.publication_protocol.PublicationEvaluator')
    def test_statistical_significance_testing(self, mock_evaluator, orchestrator):
        """Test statistical significance testing between methods."""
        mock_pub_evaluator = Mock()
        mock_pub_evaluator.compare_methods.return_value = {
            "t_test": {"p_value": 0.02, "significant": True},
            "wilcoxon": {"p_value": 0.03, "significant": True},
            "effect_size": {"cohens_d": 0.8, "magnitude": "large"}
        }
        mock_evaluator.return_value = mock_pub_evaluator
        
        results_a = [0.80, 0.85, 0.82, 0.87, 0.84]
        results_b = [0.70, 0.75, 0.72, 0.77, 0.74]
        
        comparison = orchestrator.compare_methods("method_a", "method_b", results_a, results_b)
        
        assert comparison["significant"]
        assert comparison["p_value"] < 0.05
        assert comparison["effect_size"]["magnitude"] == "large"
    
    def test_confidence_interval_calculation(self, orchestrator):
        """Test confidence interval calculation for results."""
        scores = [0.80, 0.85, 0.82, 0.87, 0.84, 0.81, 0.88, 0.83, 0.86, 0.79]
        
        ci = orchestrator.calculate_confidence_interval(scores, confidence_level=0.95)
        
        assert len(ci) == 2  # Lower and upper bounds
        assert ci[0] < ci[1]
        
        # Mean should be within confidence interval
        mean_score = sum(scores) / len(scores)
        assert ci[0] <= mean_score <= ci[1]

    # External benchmark integration tests
    @patch('src.infinitebench.external_benchmarks.ExternalBenchmarkFactory')
    def test_external_benchmark_integration(self, mock_benchmark_factory, orchestrator):
        """Test integration with external benchmarks."""
        mock_benchmarks = {
            "longbench_v2": Mock(name="longbench_v2"),
            "l_eval": Mock(name="l_eval"),
            "ruler": Mock(name="ruler")
        }
        
        for benchmark in mock_benchmarks.values():
            benchmark.evaluate = AsyncMock(return_value={
                "accuracy": 0.78,
                "task_specific_metrics": {"f1": 0.82, "precision": 0.85}
            })
        
        mock_benchmark_factory.create_benchmarks.return_value = mock_benchmarks
        
        orchestrator.initialize_external_benchmarks(["longbench_v2", "l_eval"])
        
        assert len(orchestrator._external_benchmarks) == 2
        assert "longbench_v2" in orchestrator._external_benchmarks

    # Performance monitoring tests
    def test_performance_tracking(self, orchestrator):
        """Test performance metrics tracking during evaluation."""
        # Mock performance data
        performance_data = {
            "total_evaluation_time": 3600,  # 1 hour
            "average_task_time": 120,       # 2 minutes per task
            "memory_peak_mb": 2048,         # 2GB peak
            "gpu_utilization": 0.85
        }
        
        orchestrator._performance_tracker = performance_data
        
        metrics = orchestrator.get_performance_metrics()
        
        assert metrics["total_evaluation_time"] == 3600
        assert metrics["tasks_per_minute"] > 0
        assert metrics["memory_efficiency"] > 0
    
    @pytest.mark.asyncio
    async def test_resource_monitoring(self, orchestrator):
        """Test resource monitoring during evaluation."""
        with patch('psutil.Process') as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
            mock_process.return_value.cpu_percent.return_value = 75.0
            
            async with orchestrator.monitor_resources() as monitor:
                # Simulate some work
                await asyncio.sleep(0.1)
            
            resource_stats = monitor.get_stats()
            
            assert "peak_memory_mb" in resource_stats
            assert "avg_cpu_percent" in resource_stats
            assert resource_stats["peak_memory_mb"] == 100.0

    # Report generation tests
    def test_html_report_generation(self, orchestrator, temp_output_dir):
        """Test HTML report generation."""
        # Mock aggregated results
        aggregated_results = {
            "lethe_hybrid": {
                "accuracy": {"mean": 0.85, "std": 0.05, "min": 0.80, "max": 0.90},
                "latency_ms": {"mean": 120, "std": 20, "min": 100, "max": 150}
            },
            "bm25_baseline": {
                "accuracy": {"mean": 0.75, "std": 0.08, "min": 0.65, "max": 0.85},
                "latency_ms": {"mean": 80, "std": 15, "min": 60, "max": 100}
            }
        }
        
        report_path = orchestrator.generate_html_report(
            aggregated_results,
            output_path=temp_output_dir / "evaluation_report.html"
        )
        
        assert report_path.exists()
        assert report_path.suffix == ".html"
        
        # Check content contains key elements
        content = report_path.read_text()
        assert "lethe_hybrid" in content
        assert "bm25_baseline" in content
        assert "accuracy" in content
    
    def test_json_results_export(self, orchestrator, temp_output_dir):
        """Test JSON results export."""
        results = [
            {"method": "lethe_hybrid", "task": "task1", "metrics": {"accuracy": 0.85}},
            {"method": "bm25_baseline", "task": "task1", "metrics": {"accuracy": 0.75}}
        ]
        
        export_path = orchestrator.export_results_json(
            results,
            output_path=temp_output_dir / "results.json"
        )
        
        assert export_path.exists()
        
        # Verify JSON structure
        with export_path.open() as f:
            exported_data = json.load(f)
        
        assert len(exported_data) == 2
        assert exported_data[0]["method"] == "lethe_hybrid"
        assert exported_data[1]["method"] == "bm25_baseline"
    
    def test_publication_ready_report(self, orchestrator, temp_output_dir):
        """Test publication-ready report generation."""
        # Mock comprehensive results
        results = {
            "methods": ["lethe_hybrid", "bm25_baseline", "llmlingua"],
            "tasks": ["retrieve.needle", "code.debug", "qa.hotpot"],
            "statistical_comparisons": {
                ("lethe_hybrid", "bm25_baseline"): {
                    "p_value": 0.02,
                    "effect_size": 0.8,
                    "significant": True
                }
            },
            "performance_analysis": {
                "total_evaluation_time": 3600,
                "tasks_completed": 45,
                "success_rate": 0.95
            }
        }
        
        report_path = orchestrator.generate_publication_report(
            results,
            output_dir=temp_output_dir
        )
        
        assert report_path.exists()
        assert (temp_output_dir / "figures").exists()
        assert (temp_output_dir / "tables").exists()

    # Edge cases and error handling
    def test_empty_method_list_handling(self, orchestrator):
        """Test handling of empty method list."""
        with pytest.raises(ValueError, match="No methods provided"):
            orchestrator.run_comprehensive_evaluation(methods=[], tasks=["task1"])
    
    def test_empty_task_list_handling(self, orchestrator):
        """Test handling of empty task list."""
        with pytest.raises(ValueError, match="No tasks provided"):
            orchestrator.run_comprehensive_evaluation(methods=["method1"], tasks=[])
    
    def test_nonexistent_method_handling(self, orchestrator, mock_baseline_methods):
        """Test handling of nonexistent methods."""
        orchestrator._baseline_methods = mock_baseline_methods
        
        with pytest.raises(ValueError, match="Method.*not found"):
            orchestrator.run_comprehensive_evaluation(
                methods=["nonexistent_method"],
                tasks=["task1"]
            )
    
    def test_partial_failure_handling(self, orchestrator, mock_baseline_methods):
        """Test handling of partial evaluation failures."""
        orchestrator._baseline_methods = mock_baseline_methods
        
        # Mix of successful and failing tasks
        mock_tasks = {
            "success_task": Mock(evaluate=AsyncMock(return_value={"accuracy": 0.8})),
            "fail_task": Mock(evaluate=AsyncMock(side_effect=Exception("Failed")))
        }
        orchestrator._task_registry = mock_tasks
        
        # Should complete successfully overall
        results = orchestrator.run_comprehensive_evaluation(
            methods=["lethe_hybrid"],
            tasks=["success_task", "fail_task"],
            handle_errors=True
        )
        
        # Should have results for successful task and error info for failed task
        assert len(results) == 2
        success_result = next(r for r in results if r["task"] == "success_task")
        fail_result = next(r for r in results if r["task"] == "fail_task")
        
        assert success_result["status"] == "success"
        assert fail_result["status"] == "error"

    # Configuration and workflow tests
    def test_evaluation_pipeline_configuration(self, orchestrator):
        """Test configuration of evaluation pipeline."""
        pipeline_config = {
            "parallel_workers": 4,
            "retry_failed": True,
            "save_intermediate": True,
            "enable_profiling": True
        }
        
        orchestrator.configure_pipeline(**pipeline_config)
        
        assert orchestrator._pipeline_config["parallel_workers"] == 4
        assert orchestrator._pipeline_config["retry_failed"] is True
    
    @pytest.mark.asyncio
    async def test_complete_evaluation_workflow(self, orchestrator, mock_baseline_methods, temp_output_dir):
        """Test complete end-to-end evaluation workflow."""
        # Setup
        orchestrator._baseline_methods = mock_baseline_methods
        
        mock_tasks = {
            "task1": Mock(evaluate=AsyncMock(return_value={"accuracy": 0.85})),
            "task2": Mock(evaluate=AsyncMock(return_value={"accuracy": 0.78}))
        }
        orchestrator._task_registry = mock_tasks
        
        # Run complete workflow
        final_results = await orchestrator.run_complete_evaluation(
            methods=["lethe_hybrid", "bm25_baseline"],
            tasks=["task1", "task2"],
            generate_reports=True
        )
        
        # Verify results
        assert "raw_results" in final_results
        assert "aggregated_results" in final_results
        assert "statistical_comparisons" in final_results
        assert "performance_metrics" in final_results
        
        # Verify reports were generated
        assert (temp_output_dir / "evaluation_report.html").exists()
        assert (temp_output_dir / "results.json").exists()


if __name__ == "__main__":
    pytest.main([__file__])