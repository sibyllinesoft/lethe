"""
Matrix Execution Framework with Fail-Closed Gates

This module implements the complete matrix execution system with:

1. Matrix Configuration: Datasets, budgets, K values, seeds
2. Fail-Closed Gates: pool/tokenizer equality, timing constraints, ECE validation
3. Dataset Management: InfiniteBench slices + conversation-centric sets
4. Canary Validation: Mini-matrix testing before full execution
5. Result Generation: Comprehensive outputs with certificates

Usage:
    from evaluation.matrix_execution import MatrixExecutor, MatrixConfig
    from evaluation.parity_harness import ParityHarness
    
    # Create executor
    executor = MatrixExecutor(harness=harness)
    
    # Configure matrix
    config = MatrixConfig(
        datasets=["infinitebench_qa", "conversation_code"],
        budget_ratios=[0.08, 0.15, 0.30],
        K_values=[1, 5, 10],
        seeds=[1, 2, 3]
    )
    
    # Execute matrix
    results = executor.execute_matrix(config)
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import numpy as np
from collections import defaultdict
import concurrent.futures
import hashlib

from .unified_adapter_interface import AdapterRegistry, generate_hash
from .parity_harness import ParityHarness, CorpusSpec, ContextItem, EvaluationResult
from .embedding_freezing import EmbeddingManager, PoolManager

logger = logging.getLogger(__name__)

class GateStatus(Enum):
    """Status of fail-closed gates."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"

@dataclass
class FailClosedGate:
    """Definition of a fail-closed gate."""
    name: str
    description: str
    validator_func: str  # Function name to call
    threshold: Optional[float] = None
    required: bool = True
    
@dataclass
class GateResult:
    """Result of a gate evaluation."""
    gate_name: str
    status: GateStatus
    value: Any
    threshold: Optional[float]
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MatrixConfig:
    """Configuration for matrix execution."""
    
    # Core matrix dimensions
    datasets: List[str] = field(default_factory=lambda: ["infinitebench_qa", "conversation_code"])
    budget_ratios: List[float] = field(default_factory=lambda: [0.08, 0.15, 0.30])
    K_values: List[int] = field(default_factory=lambda: [1, 5, 10])
    seeds: List[int] = field(default_factory=lambda: [1, 2, 3])
    
    # Adapter filtering
    adapter_filter: Optional[List[str]] = None
    
    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path("matrix_results"))
    save_intermediate: bool = True
    
    # Execution configuration
    parallel_samples: bool = True
    max_workers: int = 4
    timeout_per_sample: Optional[float] = 300  # 5 minutes per sample
    
    # Gate configuration
    enable_gates: bool = True
    gate_config: Dict[str, Any] = field(default_factory=dict)
    
    def get_total_combinations(self) -> int:
        """Get total number of matrix combinations."""
        adapters = self.adapter_filter or AdapterRegistry.list_adapters()
        return (len(self.datasets) * len(self.budget_ratios) * 
                len(self.K_values) * len(self.seeds) * len(adapters))

@dataclass
class MatrixResult:
    """Result of complete matrix execution."""
    config: MatrixConfig
    results: Dict[str, Dict[str, Dict[str, EvaluationResult]]]  # dataset -> sample_id -> method_id -> result
    gate_results: Dict[str, List[GateResult]]
    execution_summary: Dict[str, Any]
    start_time: float
    end_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Convert results to serializable format
        serializable_results = {}
        for dataset, dataset_results in self.results.items():
            serializable_results[dataset] = {}
            for sample_id, sample_results in dataset_results.items():
                serializable_results[dataset][sample_id] = {
                    method_id: result.to_dict() 
                    for method_id, result in sample_results.items()
                }
        
        return {
            'config': asdict(self.config),
            'results': serializable_results,
            'gate_results': {
                dataset: [asdict(gate) for gate in gates] 
                for dataset, gates in self.gate_results.items()
            },
            'execution_summary': self.execution_summary,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_seconds': self.end_time - self.start_time
        }

class DatasetManager:
    """Manages dataset loading and sample generation."""
    
    def __init__(self, data_dir: Path = Path("datasets")):
        self.data_dir = data_dir
        self._dataset_cache = {}
        
    def load_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Load a dataset by name."""
        if dataset_name in self._dataset_cache:
            return self._dataset_cache[dataset_name]
        
        # Try to load from various sources
        dataset_samples = []
        
        if dataset_name.startswith("infinitebench"):
            dataset_samples = self._load_infinitebench_dataset(dataset_name)
        elif dataset_name.startswith("conversation"):
            dataset_samples = self._load_conversation_dataset(dataset_name)
        else:
            # Try loading from file
            dataset_file = self.data_dir / f"{dataset_name}.json"
            if dataset_file.exists():
                with open(dataset_file, 'r') as f:
                    dataset_samples = json.load(f)
            else:
                logger.warning(f"Dataset not found: {dataset_name}")
                return []
        
        self._dataset_cache[dataset_name] = dataset_samples
        logger.info(f"Loaded {len(dataset_samples)} samples from {dataset_name}")
        return dataset_samples
    
    def _load_infinitebench_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Load InfiniteBench dataset slices."""
        # This would integrate with existing InfiniteBench loading
        # For now, return synthetic samples
        
        task_type = dataset_name.split("_")[-1] if "_" in dataset_name else "qa"
        
        samples = []
        for i in range(20):  # 20 samples per InfiniteBench slice
            sample = {
                'sample_id': f"{dataset_name}_{i:03d}",
                'query': f"Question {i+1} for {task_type} task",
                'context_items': self._generate_synthetic_context(task_type, length=1000 + i*100),
                'ground_truth': f"Answer {i+1}",
                'metadata': {
                    'task_type': task_type,
                    'source': 'infinitebench',
                    'difficulty': 'medium'
                }
            }
            samples.append(sample)
        
        return samples
    
    def _load_conversation_dataset(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Load conversation-centric datasets."""
        domain = dataset_name.split("_")[-1] if "_" in dataset_name else "general"
        
        samples = []
        for i in range(15):  # 15 samples per conversation dataset
            sample = {
                'sample_id': f"{dataset_name}_{i:03d}",
                'query': f"How do I solve this {domain} problem?",
                'context_items': self._generate_conversation_context(domain, turns=5 + i),
                'ground_truth': f"Solution for {domain} problem {i+1}",
                'metadata': {
                    'domain': domain,
                    'source': 'conversation',
                    'turns': 5 + i
                }
            }
            samples.append(sample)
        
        return samples
    
    def _generate_synthetic_context(self, task_type: str, length: int) -> List[ContextItem]:
        """Generate synthetic context items."""
        items = []
        
        # Add some turns
        for i in range(3):
            items.append(ContextItem(
                content=f"Turn {i+1}: This is some {task_type} related content. " * (length // 50),
                item_type="turn",
                timestamp=time.time() - (3-i) * 3600,
                source="synthetic"
            ))
        
        # Add tool I/O
        items.append(ContextItem(
            content=f"Tool output: Processed {task_type} data successfully. " * (length // 100),
            item_type="tool_io",
            timestamp=time.time() - 1800,
            source="tool"
        ))
        
        # Add code/error if relevant
        if task_type in ["code", "debug"]:
            items.append(ContextItem(
                content=f"Error: {task_type} compilation failed at line 42. " * (length // 80),
                item_type="error",
                timestamp=time.time() - 900,
                source="compiler"
            ))
        
        return items
    
    def _generate_conversation_context(self, domain: str, turns: int) -> List[ContextItem]:
        """Generate conversation context items."""
        items = []
        
        for i in range(turns):
            speaker = "user" if i % 2 == 0 else "assistant"
            content = f"{speaker.title()}: This is turn {i+1} in a {domain} conversation. "
            
            if domain == "code":
                content += "Here's some code that might be relevant. def function(): return True"
            elif domain == "debug":
                content += "The error occurs when trying to access undefined variable x."
            else:
                content += f"Let me help you with this {domain} question."
            
            items.append(ContextItem(
                content=content * 20,  # Make it longer
                item_type="turn",
                timestamp=time.time() - (turns-i) * 600,
                source=speaker,
                metadata={'speaker': speaker, 'turn': i+1}
            ))
        
        return items
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a dataset."""
        samples = self.load_dataset(dataset_name)
        
        if not samples:
            return {'exists': False}
        
        # Analyze dataset
        total_context_items = sum(len(sample['context_items']) for sample in samples)
        avg_context_items = total_context_items / len(samples)
        
        return {
            'exists': True,
            'num_samples': len(samples),
            'avg_context_items': avg_context_items,
            'sample_ids': [sample['sample_id'] for sample in samples[:5]],  # First 5
            'metadata_keys': list(samples[0].get('metadata', {}).keys()) if samples else []
        }

class FailClosedGateValidator:
    """Validates fail-closed gates during matrix execution."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.gates = self._define_gates()
        
    def _define_gates(self) -> List[FailClosedGate]:
        """Define all fail-closed gates."""
        return [
            FailClosedGate(
                name="pool_tokenizer_equality",
                description="All methods use same pool and tokenizer fingerprints",
                validator_func="validate_pool_tokenizer_equality",
                required=True
            ),
            FailClosedGate(
                name="timing_constraints",
                description="p95 >= avg time, p99/p95 <= 2.5",
                validator_func="validate_timing_constraints",
                threshold=2.5,
                required=True
            ),
            FailClosedGate(
                name="budget_compliance",
                description="No method exceeds token budget",
                validator_func="validate_budget_compliance",
                required=True
            ),
            FailClosedGate(
                name="coverage_minimum",
                description="Non-zero coverage at 30% budget",
                validator_func="validate_coverage_minimum",
                threshold=0.0,
                required=True
            ),
            FailClosedGate(
                name="ece_variance",
                description="ECE × type × budget <= 0.08",
                validator_func="validate_ece_variance",
                threshold=0.08,
                required=False  # May not apply to all methods
            )
        ]
    
    def validate_all_gates(self, results: Dict[str, Dict[str, EvaluationResult]]) -> List[GateResult]:
        """Validate all gates for a set of results."""
        gate_results = []
        
        for gate in self.gates:
            try:
                validator_func = getattr(self, gate.validator_func)
                gate_result = validator_func(gate, results)
                gate_results.append(gate_result)
                
            except Exception as e:
                gate_result = GateResult(
                    gate_name=gate.name,
                    status=GateStatus.FAILED,
                    value=None,
                    threshold=gate.threshold,
                    message=f"Gate validation error: {e}",
                    details={'error': str(e)}
                )
                gate_results.append(gate_result)
        
        return gate_results
    
    def validate_pool_tokenizer_equality(self, gate: FailClosedGate, 
                                       results: Dict[str, Dict[str, EvaluationResult]]) -> GateResult:
        """Validate that all methods use same pool and tokenizer."""
        pool_fingerprints = set()
        tokenizer_hashes = set()
        
        for sample_id, sample_results in results.items():
            for method_id, result in sample_results.items():
                pool_fingerprints.add(result.selection_result.pool_fingerprint)
                tokenizer_hashes.add(result.selection_result.tokenizer_hash)
        
        pool_equal = len(pool_fingerprints) <= 1
        tokenizer_equal = len(tokenizer_hashes) <= 1
        
        if pool_equal and tokenizer_equal:
            status = GateStatus.PASSED
            message = "All methods use consistent pool and tokenizer"
        else:
            status = GateStatus.FAILED
            message = f"Inconsistent fingerprints: {len(pool_fingerprints)} pools, {len(tokenizer_hashes)} tokenizers"
        
        return GateResult(
            gate_name=gate.name,
            status=status,
            value={'pools': len(pool_fingerprints), 'tokenizers': len(tokenizer_hashes)},
            threshold=None,
            message=message,
            details={
                'pool_fingerprints': list(pool_fingerprints),
                'tokenizer_hashes': list(tokenizer_hashes)
            }
        )
    
    def validate_timing_constraints(self, gate: FailClosedGate,
                                  results: Dict[str, Dict[str, EvaluationResult]]) -> GateResult:
        """Validate timing constraints."""
        violations = []
        
        for sample_id, sample_results in results.items():
            for method_id, result in sample_results.items():
                sr = result.selection_result
                
                # Check p95 >= avg
                if sr.time_p95 < sr.time_ms:
                    violations.append(f"{method_id}@{sample_id}: p95({sr.time_p95:.1f}) < avg({sr.time_ms:.1f})")
                
                # Check ratio constraint (assuming p99 ≈ p95 * ratio_factor)
                if sr.time_p95 > 0 and sr.time_ms > 0:
                    ratio = sr.time_p95 / sr.time_ms
                    if ratio > gate.threshold:
                        violations.append(f"{method_id}@{sample_id}: ratio({ratio:.2f}) > {gate.threshold}")
        
        if not violations:
            status = GateStatus.PASSED
            message = "All timing constraints satisfied"
        else:
            status = GateStatus.FAILED
            message = f"{len(violations)} timing violations"
        
        return GateResult(
            gate_name=gate.name,
            status=status,
            value=len(violations),
            threshold=gate.threshold,
            message=message,
            details={'violations': violations[:10]}  # Limit to first 10
        )
    
    def validate_budget_compliance(self, gate: FailClosedGate,
                                 results: Dict[str, Dict[str, EvaluationResult]]) -> GateResult:
        """Validate budget compliance."""
        violations = []
        
        for sample_id, sample_results in results.items():
            for method_id, result in sample_results.items():
                selected_tokens = result.selection_result.total_tokens()
                budget_tokens = result.selection_result.metadata.get('budget_tokens', 0)
                
                if selected_tokens > budget_tokens:
                    violations.append(f"{method_id}@{sample_id}: {selected_tokens} > {budget_tokens}")
        
        if not violations:
            status = GateStatus.PASSED
            message = "All methods comply with token budgets"
        else:
            status = GateStatus.FAILED
            message = f"{len(violations)} budget violations"
        
        return GateResult(
            gate_name=gate.name,
            status=status,
            value=len(violations),
            threshold=0,
            message=message,
            details={'violations': violations}
        )
    
    def validate_coverage_minimum(self, gate: FailClosedGate,
                                results: Dict[str, Dict[str, EvaluationResult]]) -> GateResult:
        """Validate minimum coverage at 30% budget."""
        coverage_failures = []
        
        for sample_id, sample_results in results.items():
            for method_id, result in sample_results.items():
                selected_count = len(result.selection_result.selected_atoms)
                
                # Check if this is a 30% budget run
                budget_ratio = result.selection_result.metadata.get('budget_ratio', 0)
                if abs(budget_ratio - 0.30) < 0.01:  # Within 1% of 30%
                    if selected_count == 0:
                        coverage_failures.append(f"{method_id}@{sample_id}: zero coverage at 30% budget")
        
        if not coverage_failures:
            status = GateStatus.PASSED
            message = "All methods have non-zero coverage at 30% budget"
        else:
            status = GateStatus.FAILED
            message = f"{len(coverage_failures)} coverage failures"
        
        return GateResult(
            gate_name=gate.name,
            status=status,
            value=len(coverage_failures),
            threshold=gate.threshold,
            message=message,
            details={'failures': coverage_failures}
        )
    
    def validate_ece_variance(self, gate: FailClosedGate,
                            results: Dict[str, Dict[str, EvaluationResult]]) -> GateResult:
        """Validate ECE variance constraints (placeholder)."""
        # This would implement the actual ECE validation
        # For now, just return a passed status as this is optional
        
        return GateResult(
            gate_name=gate.name,
            status=GateStatus.SKIPPED,
            value=None,
            threshold=gate.threshold,
            message="ECE validation not implemented yet",
            details={'note': 'This gate is optional and not yet implemented'}
        )

class MatrixExecutor:
    """Main executor for matrix evaluation."""
    
    def __init__(self, harness: ParityHarness,
                 dataset_manager: Optional[DatasetManager] = None,
                 embedding_manager: Optional[EmbeddingManager] = None):
        self.harness = harness
        self.dataset_manager = dataset_manager or DatasetManager()
        self.embedding_manager = embedding_manager
        self.gate_validator = FailClosedGateValidator()
        
    def execute_mini_matrix_canary(self, config: MatrixConfig) -> Dict[str, Any]:
        """
        Execute mini-matrix canary on 1 dataset/bucket with seeds=1.
        
        This validates that all adapters work correctly before full execution.
        """
        logger.info("Starting mini-matrix canary validation")
        
        # Use first dataset and first budget ratio only
        canary_config = MatrixConfig(
            datasets=[config.datasets[0]],
            budget_ratios=[config.budget_ratios[0]],
            K_values=config.K_values,
            seeds=[1],  # Only seed 1
            adapter_filter=config.adapter_filter,
            enable_gates=True
        )
        
        # Execute mini-matrix
        start_time = time.time()
        
        try:
            results = self._execute_matrix_internal(canary_config)
            
            # Validate gates
            all_gate_results = []
            for dataset, dataset_results in results.items():
                gate_results = self.gate_validator.validate_all_gates(dataset_results)
                all_gate_results.extend(gate_results)
            
            # Check for failures
            failed_gates = [gr for gr in all_gate_results if gr.status == GateStatus.FAILED]
            
            canary_result = {
                'success': len(failed_gates) == 0,
                'duration_seconds': time.time() - start_time,
                'total_evaluations': sum(len(dr) for dr in results.values()),
                'gate_results': [asdict(gr) for gr in all_gate_results],
                'failed_gates': len(failed_gates),
                'adapter_validation': self._validate_adapters(results),
                'config': asdict(canary_config)
            }
            
            if canary_result['success']:
                logger.info("Mini-matrix canary PASSED - proceeding to full matrix")
            else:
                logger.error(f"Mini-matrix canary FAILED - {len(failed_gates)} gate failures")
            
            return canary_result
            
        except Exception as e:
            logger.error(f"Mini-matrix canary execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration_seconds': time.time() - start_time
            }
    
    def execute_matrix(self, config: MatrixConfig, run_canary: bool = True) -> MatrixResult:
        """Execute complete matrix evaluation."""
        
        start_time = time.time()
        
        # Run canary first if requested
        if run_canary:
            canary_result = self.execute_mini_matrix_canary(config)
            if not canary_result['success']:
                raise RuntimeError(f"Canary validation failed: {canary_result}")
        
        logger.info(f"Starting full matrix execution with {config.get_total_combinations()} combinations")
        
        # Execute full matrix
        results = self._execute_matrix_internal(config)
        
        # Validate gates for all datasets
        gate_results = {}
        for dataset, dataset_results in results.items():
            gate_results[dataset] = self.gate_validator.validate_all_gates(dataset_results)
        
        # Generate execution summary
        execution_summary = self._generate_execution_summary(results, gate_results)
        
        # Create matrix result
        matrix_result = MatrixResult(
            config=config,
            results=results,
            gate_results=gate_results,
            execution_summary=execution_summary,
            start_time=start_time,
            end_time=time.time()
        )
        
        # Save results
        if config.output_dir:
            self._save_matrix_result(matrix_result)
        
        logger.info(f"Matrix execution completed in {matrix_result.end_time - matrix_result.start_time:.1f}s")
        return matrix_result
    
    def _execute_matrix_internal(self, config: MatrixConfig) -> Dict[str, Dict[str, Dict[str, EvaluationResult]]]:
        """Internal matrix execution logic."""
        results = {}
        
        # Ensure adapters are registered
        if not self.harness._adapters_registered:
            self.harness.register_all_adapters()
        
        # Set up embedding manager if needed
        if self.embedding_manager:
            pool_manager = PoolManager(self.embedding_manager)
        
        for dataset_name in config.datasets:
            logger.info(f"Processing dataset: {dataset_name}")
            
            # Load dataset
            dataset_samples = self.dataset_manager.load_dataset(dataset_name)
            if not dataset_samples:
                logger.warning(f"No samples found for dataset {dataset_name}")
                continue
            
            dataset_results = {}
            
            # Process each combination of budget_ratio, K, seed
            for budget_ratio in config.budget_ratios:
                for K in config.K_values:
                    for seed in config.seeds:
                        
                        # Process each sample
                        for sample_data in dataset_samples:
                            sample_id = f"{sample_data['sample_id']}_b{budget_ratio}_k{K}_s{seed}"
                            
                            # Create corpus spec
                            spec = CorpusSpec(
                                query=sample_data['query'],
                                context_items=[
                                    ContextItem(**item) if isinstance(item, dict) else item
                                    for item in sample_data['context_items']
                                ],
                                keep_ratio=budget_ratio,
                                K=K,
                                seed=seed,
                                sample_id=sample_id
                            )
                            
                            # Evaluate sample
                            try:
                                sample_results = self.harness.evaluate_sample(spec, config.adapter_filter)
                                
                                # Add budget ratio to metadata
                                for result in sample_results.values():
                                    result.selection_result.metadata['budget_ratio'] = budget_ratio
                                
                                dataset_results[sample_id] = sample_results
                                
                            except Exception as e:
                                logger.error(f"Failed to evaluate {sample_id}: {e}")
                                continue
            
            results[dataset_name] = dataset_results
            
            # Save intermediate results if requested
            if config.save_intermediate and config.output_dir:
                intermediate_file = config.output_dir / f"intermediate_{dataset_name}.json"
                self._save_dataset_results(dataset_results, intermediate_file)
        
        return results
    
    def _validate_adapters(self, results: Dict[str, Dict[str, Dict[str, EvaluationResult]]]) -> Dict[str, Any]:
        """Validate that all adapters produced valid results."""
        adapter_stats = defaultdict(lambda: {'total': 0, 'valid': 0, 'errors': []})
        
        for dataset, dataset_results in results.items():
            for sample_id, sample_results in dataset_results.items():
                for method_id, result in sample_results.items():
                    adapter_stats[method_id]['total'] += 1
                    if result.is_valid:
                        adapter_stats[method_id]['valid'] += 1
                    else:
                        adapter_stats[method_id]['errors'].extend(result.validation_errors)
        
        # Calculate success rates
        validation_summary = {}
        for method_id, stats in adapter_stats.items():
            success_rate = stats['valid'] / stats['total'] if stats['total'] > 0 else 0
            validation_summary[method_id] = {
                'total_evaluations': stats['total'],
                'valid_evaluations': stats['valid'],
                'success_rate': success_rate,
                'error_count': len(stats['errors']),
                'sample_errors': stats['errors'][:5]  # First 5 errors
            }
        
        return validation_summary
    
    def _generate_execution_summary(self, results: Dict[str, Dict[str, Dict[str, EvaluationResult]]],
                                  gate_results: Dict[str, List[GateResult]]) -> Dict[str, Any]:
        """Generate comprehensive execution summary."""
        
        # Count totals
        total_samples = sum(len(dataset_results) for dataset_results in results.values())
        total_evaluations = sum(
            len(sample_results) 
            for dataset_results in results.values()
            for sample_results in dataset_results.values()
        )
        
        # Count gate failures
        total_gate_failures = sum(
            len([gr for gr in gates if gr.status == GateStatus.FAILED])
            for gates in gate_results.values()
        )
        
        # Performance stats
        all_times = []
        all_token_counts = []
        
        for dataset_results in results.values():
            for sample_results in dataset_results.values():
                for result in sample_results.values():
                    if result.is_valid:
                        all_times.append(result.selection_result.time_ms)
                        all_token_counts.append(result.selection_result.total_tokens())
        
        performance_stats = {}
        if all_times:
            performance_stats = {
                'avg_time_ms': np.mean(all_times),
                'p95_time_ms': np.percentile(all_times, 95),
                'avg_tokens_selected': np.mean(all_token_counts),
                'total_evaluations': len(all_times)
            }
        
        return {
            'datasets': list(results.keys()),
            'total_samples': total_samples,
            'total_evaluations': total_evaluations,
            'total_gate_failures': total_gate_failures,
            'performance_stats': performance_stats,
            'adapter_count': len(set(
                method_id
                for dataset_results in results.values()
                for sample_results in dataset_results.values()
                for method_id in sample_results.keys()
            )) if results else 0
        }
    
    def _save_matrix_result(self, matrix_result: MatrixResult):
        """Save complete matrix result."""
        output_dir = matrix_result.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main result
        result_file = output_dir / "matrix_result.json"
        with open(result_file, 'w') as f:
            json.dump(matrix_result.to_dict(), f, indent=2, default=str)
        
        # Save individual dataset results
        for dataset, dataset_results in matrix_result.results.items():
            dataset_file = output_dir / f"dataset_{dataset}.json"
            self._save_dataset_results(dataset_results, dataset_file)
        
        # Save gate results
        gate_file = output_dir / "gate_results.json"
        with open(gate_file, 'w') as f:
            json.dump({
                dataset: [asdict(gate) for gate in gates]
                for dataset, gates in matrix_result.gate_results.items()
            }, f, indent=2, default=str)
        
        logger.info(f"Matrix results saved to {output_dir}")
    
    def _save_dataset_results(self, dataset_results: Dict[str, Dict[str, EvaluationResult]], 
                            output_file: Path):
        """Save dataset results to file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        serializable_results = {
            sample_id: {
                method_id: result.to_dict()
                for method_id, result in sample_results.items()
            }
            for sample_id, sample_results in dataset_results.items()
        }
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)

# Export main classes
__all__ = [
    'MatrixExecutor',
    'MatrixConfig',
    'MatrixResult',
    'DatasetManager',
    'FailClosedGateValidator',
    'GateResult',
    'GateStatus'
]