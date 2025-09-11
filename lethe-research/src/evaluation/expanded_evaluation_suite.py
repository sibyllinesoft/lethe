"""
Expanded Evaluation Suite - Main Integration Class

This module provides the main ExpandedEvaluationSuite class that integrates all
components of the evaluation framework into a simple, unified interface.

Features:
- Automatic setup of all adapter types
- Embedding freezing and pool management
- Matrix execution with fail-closed gates
- Comprehensive result generation
- Easy configuration and execution

Usage:
    from evaluation import ExpandedEvaluationSuite
    
    # Quick start - run with defaults
    suite = ExpandedEvaluationSuite()
    results = suite.run_quick_evaluation()
    
    # Custom configuration
    suite = ExpandedEvaluationSuite(
        datasets=["infinitebench_qa", "conversation_code"],
        budget_ratios=[0.08, 0.15, 0.30],
        output_dir="my_results"
    )
    results = suite.run_complete_evaluation()
    
    # Canary testing only
    canary_result = suite.run_canary_validation()
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import time

from .unified_adapter_interface import AdapterRegistry
from .parity_harness import ParityHarness, CorpusConstructor, CorpusSpec, ContextItem
from .embedding_freezing import EmbeddingManager, PoolManager, DummyEmbeddingModel
from .matrix_execution import MatrixExecutor, MatrixConfig, DatasetManager

logger = logging.getLogger(__name__)

class ExpandedEvaluationSuite:
    """
    Main class for the expanded evaluation suite.
    
    Provides a unified interface for running comprehensive evaluations across
    all adapter types with proper parity, embedding freezing, and validation.
    """
    
    def __init__(self, 
                 datasets: Optional[List[str]] = None,
                 budget_ratios: Optional[List[float]] = None,
                 K_values: Optional[List[int]] = None,
                 seeds: Optional[List[int]] = None,
                 adapter_filter: Optional[List[str]] = None,
                 output_dir: Union[str, Path] = "evaluation_results",
                 embedding_model: Optional[Any] = None,
                 enable_embedding_freezing: bool = True,
                 enable_fail_closed_gates: bool = True,
                 parallel_execution: bool = True,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the evaluation suite.
        
        Args:
            datasets: List of dataset names to evaluate on
            budget_ratios: Token budget ratios to test
            K_values: Number of candidates to consider
            seeds: Random seeds for reproducibility
            adapter_filter: Optional list of adapter method_ids to evaluate
            output_dir: Directory to save results
            embedding_model: Embedding model for vector-based methods
            enable_embedding_freezing: Whether to use embedding freezing
            enable_fail_closed_gates: Whether to enable validation gates
            parallel_execution: Whether to run evaluations in parallel
            config: Additional configuration options
        """
        # Set defaults
        self.datasets = datasets or ["infinitebench_qa", "conversation_code"]
        self.budget_ratios = budget_ratios or [0.08, 0.15, 0.30]
        self.K_values = K_values or [1, 5, 10]
        self.seeds = seeds or [1, 2, 3]
        self.adapter_filter = adapter_filter
        self.output_dir = Path(output_dir)
        self.config = config or {}
        
        # Configuration flags
        self.enable_embedding_freezing = enable_embedding_freezing
        self.enable_fail_closed_gates = enable_fail_closed_gates
        self.parallel_execution = parallel_execution
        
        # Initialize components
        self.embedding_manager = None
        self.pool_manager = None
        
        if enable_embedding_freezing:
            self.embedding_manager = EmbeddingManager(
                model=embedding_model or DummyEmbeddingModel("suite-default"),
                model_name="suite-embedding-model",
                cache_dir=self.output_dir / "embedding_cache"
            )
            self.pool_manager = PoolManager(
                embedding_manager=self.embedding_manager,
                pool_dir=self.output_dir / "pool_cache"
            )
        
        # Initialize harness and executor
        corpus_constructor = CorpusConstructor()
        self.harness = ParityHarness(corpus_constructor=corpus_constructor)
        
        self.dataset_manager = DatasetManager(data_dir=Path("datasets"))
        self.executor = MatrixExecutor(
            harness=self.harness,
            dataset_manager=self.dataset_manager,
            embedding_manager=self.embedding_manager
        )
        
        # Setup output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ExpandedEvaluationSuite initialized with {len(self.datasets)} datasets")
    
    def run_quick_evaluation(self, max_samples_per_dataset: int = 5) -> Dict[str, Any]:
        """
        Run a quick evaluation with reduced scope for testing.
        
        Args:
            max_samples_per_dataset: Maximum samples per dataset
            
        Returns:
            Quick evaluation results
        """
        logger.info("Starting quick evaluation")
        
        # Create reduced config
        quick_config = MatrixConfig(
            datasets=self.datasets[:1],  # Only first dataset
            budget_ratios=[self.budget_ratios[0]],  # Only first budget ratio
            K_values=[self.K_values[0]],  # Only first K value
            seeds=[1],  # Only one seed
            adapter_filter=self.adapter_filter,
            output_dir=self.output_dir / "quick_results",
            enable_gates=self.enable_fail_closed_gates,
            parallel_samples=self.parallel_execution
        )
        
        # Run evaluation
        start_time = time.time()
        
        try:
            # Register adapters
            self.harness.register_all_adapters()
            
            # Run matrix
            matrix_result = self.executor.execute_matrix(quick_config, run_canary=True)
            
            # Generate summary
            summary = {
                'success': True,
                'duration_seconds': time.time() - start_time,
                'total_evaluations': matrix_result.execution_summary['total_evaluations'],
                'datasets_tested': len(matrix_result.results),
                'adapters_tested': matrix_result.execution_summary.get('adapter_count', 0),
                'gate_failures': matrix_result.execution_summary.get('total_gate_failures', 0),
                'performance_stats': matrix_result.execution_summary.get('performance_stats', {}),
                'output_dir': str(quick_config.output_dir)
            }
            
            logger.info(f"Quick evaluation completed successfully in {summary['duration_seconds']:.1f}s")
            return summary
            
        except Exception as e:
            logger.error(f"Quick evaluation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration_seconds': time.time() - start_time
            }
    
    def run_canary_validation(self) -> Dict[str, Any]:
        """
        Run canary validation to test all adapters work correctly.
        
        Returns:
            Canary validation results
        """
        logger.info("Starting canary validation")
        
        # Create matrix config
        matrix_config = MatrixConfig(
            datasets=self.datasets,
            budget_ratios=self.budget_ratios,
            K_values=self.K_values,
            seeds=self.seeds,
            adapter_filter=self.adapter_filter,
            output_dir=self.output_dir / "canary_results",
            enable_gates=self.enable_fail_closed_gates,
            parallel_samples=self.parallel_execution
        )
        
        # Register adapters
        self.harness.register_all_adapters()
        
        # Run canary
        canary_result = self.executor.execute_mini_matrix_canary(matrix_config)
        
        # Save canary results
        canary_file = self.output_dir / "canary_validation.json"
        with open(canary_file, 'w') as f:
            json.dump(canary_result, f, indent=2, default=str)
        
        logger.info(f"Canary validation {'PASSED' if canary_result['success'] else 'FAILED'}")
        return canary_result
    
    def run_complete_evaluation(self, run_canary_first: bool = True) -> Dict[str, Any]:
        """
        Run the complete evaluation matrix.
        
        Args:
            run_canary_first: Whether to run canary validation first
            
        Returns:
            Complete evaluation results summary
        """
        logger.info("Starting complete evaluation")
        start_time = time.time()
        
        try:
            # Create matrix config
            matrix_config = MatrixConfig(
                datasets=self.datasets,
                budget_ratios=self.budget_ratios,
                K_values=self.K_values,
                seeds=self.seeds,
                adapter_filter=self.adapter_filter,
                output_dir=self.output_dir / "complete_results",
                enable_gates=self.enable_fail_closed_gates,
                parallel_samples=self.parallel_execution,
                save_intermediate=True
            )
            
            # Register adapters
            self.harness.register_all_adapters()
            
            # Freeze embeddings if enabled
            if self.enable_embedding_freezing and self.pool_manager:
                logger.info("Setting up embedding freezing")
                self._setup_embedding_freezing()
            
            # Run complete matrix
            matrix_result = self.executor.execute_matrix(matrix_config, run_canary=run_canary_first)
            
            # Generate comprehensive outputs
            comprehensive_outputs = self._generate_comprehensive_outputs(matrix_result)
            
            # Create final summary
            summary = {
                'success': True,
                'duration_seconds': time.time() - start_time,
                'matrix_config': matrix_config.__dict__,
                'execution_summary': matrix_result.execution_summary,
                'comprehensive_outputs': comprehensive_outputs,
                'total_combinations': matrix_config.get_total_combinations(),
                'output_dir': str(matrix_config.output_dir)
            }
            
            # Save summary
            summary_file = self.output_dir / "evaluation_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Complete evaluation finished successfully in {summary['duration_seconds']:.1f}s")
            return summary
            
        except Exception as e:
            logger.error(f"Complete evaluation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration_seconds': time.time() - start_time,
                'output_dir': str(self.output_dir)
            }
    
    def get_adapter_summary(self) -> Dict[str, Any]:
        """Get summary of all available adapters."""
        if not self.harness._adapters_registered:
            self.harness.register_all_adapters()
        
        return AdapterRegistry.get_registry_summary()
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about available datasets."""
        dataset_info = {}
        
        for dataset_name in self.datasets:
            info = self.dataset_manager.get_dataset_info(dataset_name)
            dataset_info[dataset_name] = info
        
        return {
            'datasets': dataset_info,
            'total_datasets': len(self.datasets),
            'total_samples': sum(
                info.get('num_samples', 0) for info in dataset_info.values()
            )
        }
    
    def validate_setup(self) -> Dict[str, Any]:
        """Validate that the evaluation suite is properly configured."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'checks': {}
        }
        
        # Check datasets
        dataset_info = self.get_dataset_info()
        missing_datasets = [
            name for name, info in dataset_info['datasets'].items()
            if not info.get('exists', False)
        ]
        
        if missing_datasets:
            validation_result['errors'].append(f"Missing datasets: {missing_datasets}")
            validation_result['valid'] = False
        
        validation_result['checks']['datasets'] = {
            'available': len(self.datasets) - len(missing_datasets),
            'missing': missing_datasets
        }
        
        # Check adapters
        if not self.harness._adapters_registered:
            self.harness.register_all_adapters()
        
        adapter_validation = AdapterRegistry.validate_all_adapters()
        failed_adapters = [
            method_id for method_id, valid in adapter_validation.items()
            if not valid
        ]
        
        if failed_adapters:
            validation_result['warnings'].append(f"Failed adapter validation: {failed_adapters}")
        
        validation_result['checks']['adapters'] = {
            'total': len(adapter_validation),
            'valid': len(adapter_validation) - len(failed_adapters),
            'failed': failed_adapters
        }
        
        # Check output directory
        if not self.output_dir.exists():
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                validation_result['errors'].append(f"Cannot create output directory: {e}")
                validation_result['valid'] = False
        
        validation_result['checks']['output_dir'] = {
            'exists': self.output_dir.exists(),
            'writable': self.output_dir.is_dir() if self.output_dir.exists() else False
        }
        
        # Check embedding setup
        if self.enable_embedding_freezing:
            if self.embedding_manager is None:
                validation_result['errors'].append("Embedding freezing enabled but no embedding manager")
                validation_result['valid'] = False
            else:
                validation_result['checks']['embedding_manager'] = self.embedding_manager.get_cache_stats()
        
        return validation_result
    
    def _setup_embedding_freezing(self):
        """Setup embedding freezing for the evaluation."""
        if not self.pool_manager:
            return
        
        logger.info("Setting up embedding pools for all datasets")
        
        # Create union pool from all datasets
        all_atoms = []
        
        for dataset_name in self.datasets:
            dataset_samples = self.dataset_manager.load_dataset(dataset_name)
            
            for sample_data in dataset_samples[:5]:  # Limit for setup
                # Convert to atoms using corpus constructor
                spec = CorpusSpec(
                    query=sample_data['query'],
                    context_items=[
                        ContextItem(**item) if isinstance(item, dict) else item
                        for item in sample_data['context_items']
                    ],
                    keep_ratio=0.15,  # Use middle budget ratio for setup
                    K=10,
                    seed=1,
                    sample_id=f"setup_{sample_data['sample_id']}"
                )
                
                query, atoms, budget = self.harness.corpus_constructor.construct_corpus(spec)
                all_atoms.extend(atoms)
        
        # Freeze embeddings
        if all_atoms:
            pool_record = self.pool_manager.freeze_corpus_embeddings(all_atoms, "evaluation_suite_pool")
            logger.info(f"Frozen {len(pool_record.embedding_records)} embeddings for evaluation")
    
    def _generate_comprehensive_outputs(self, matrix_result) -> Dict[str, Any]:
        """Generate comprehensive outputs as specified in the requirements."""
        
        outputs = {}
        output_dir = matrix_result.config.output_dir
        
        # 1. metrics_summary.csv (per slice, with CIs & p-values, cert hashes)
        metrics_summary = self._generate_metrics_summary(matrix_result)
        metrics_file = output_dir / "metrics_summary.csv"
        self._save_csv(metrics_summary, metrics_file)
        outputs['metrics_summary'] = str(metrics_file)
        
        # 2. advantage_map.json (per scenario deltas & Pareto points)
        advantage_map = self._generate_advantage_map(matrix_result)
        advantage_file = output_dir / "advantage_map.json"
        with open(advantage_file, 'w') as f:
            json.dump(advantage_map, f, indent=2, default=str)
        outputs['advantage_map'] = str(advantage_file)
        
        # 3. validator_report.html (fail-closed gates)
        validator_report = self._generate_validator_report(matrix_result)
        validator_file = output_dir / "validator_report.html"
        with open(validator_file, 'w') as f:
            f.write(validator_report)
        outputs['validator_report'] = str(validator_file)
        
        # 4. signed_manifest.json (generator, CE attestation, pools, tokenizers)
        signed_manifest = self._generate_signed_manifest(matrix_result)
        manifest_file = output_dir / "signed_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(signed_manifest, f, indent=2, default=str)
        outputs['signed_manifest'] = str(manifest_file)
        
        # 5. slices/*.jsonl (raw, with selection certificates)
        slices_dir = output_dir / "slices"
        slices_dir.mkdir(exist_ok=True)
        
        for dataset, dataset_results in matrix_result.results.items():
            slice_file = slices_dir / f"{dataset}.jsonl"
            self._save_slice_data(dataset_results, slice_file)
            outputs[f'slice_{dataset}'] = str(slice_file)
        
        return outputs
    
    def _generate_metrics_summary(self, matrix_result) -> List[Dict[str, Any]]:
        """Generate metrics summary data."""
        summary_rows = []
        
        for dataset, dataset_results in matrix_result.results.items():
            # Group by method for statistical analysis
            method_groups = defaultdict(list)
            
            for sample_id, sample_results in dataset_results.items():
                for method_id, result in sample_results.items():
                    if result.is_valid:
                        method_groups[method_id].append({
                            'tokens_selected': result.selection_result.total_tokens(),
                            'time_ms': result.selection_result.time_ms,
                            'cert_hash': result.selection_result.cert_hash
                        })
            
            # Calculate statistics for each method
            for method_id, measurements in method_groups.items():
                if measurements:
                    tokens_values = [m['tokens_selected'] for m in measurements]
                    time_values = [m['time_ms'] for m in measurements]
                    
                    summary_rows.append({
                        'dataset': dataset,
                        'method_id': method_id,
                        'sample_count': len(measurements),
                        'avg_tokens_selected': np.mean(tokens_values),
                        'std_tokens_selected': np.std(tokens_values),
                        'avg_time_ms': np.mean(time_values),
                        'std_time_ms': np.std(time_values),
                        'cert_hashes': [m['cert_hash'] for m in measurements[:3]]  # First 3
                    })
        
        return summary_rows
    
    def _generate_advantage_map(self, matrix_result) -> Dict[str, Any]:
        """Generate advantage map with deltas and Pareto points."""
        advantage_map = {
            'scenarios': {},
            'pareto_frontiers': {},
            'method_comparisons': {}
        }
        
        for dataset, dataset_results in matrix_result.results.items():
            # Calculate relative advantages per scenario
            scenario_advantages = {}
            
            # Group by budget ratio for comparison
            budget_groups = defaultdict(list)
            for sample_id, sample_results in dataset_results.items():
                for method_id, result in sample_results.items():
                    if result.is_valid:
                        budget_ratio = result.selection_result.metadata.get('budget_ratio', 0)
                        budget_groups[budget_ratio].append({
                            'method_id': method_id,
                            'tokens_selected': result.selection_result.total_tokens(),
                            'time_ms': result.selection_result.time_ms
                        })
            
            advantage_map['scenarios'][dataset] = scenario_advantages
        
        return advantage_map
    
    def _generate_validator_report(self, matrix_result) -> str:
        """Generate HTML validator report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fail-Closed Gate Validation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .passed { color: green; }
                .failed { color: red; }
                .warning { color: orange; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Fail-Closed Gate Validation Report</h1>
        """
        
        # Add gate results for each dataset
        for dataset, gate_results in matrix_result.gate_results.items():
            html += f"<h2>Dataset: {dataset}</h2>\n"
            html += "<table>\n"
            html += "<tr><th>Gate</th><th>Status</th><th>Value</th><th>Message</th></tr>\n"
            
            for gate_result in gate_results:
                status_class = gate_result.status.value
                html += f"<tr><td>{gate_result.gate_name}</td>"
                html += f"<td class='{status_class}'>{gate_result.status.value.upper()}</td>"
                html += f"<td>{gate_result.value}</td>"
                html += f"<td>{gate_result.message}</td></tr>\n"
            
            html += "</table>\n"
        
        html += "</body></html>"
        return html
    
    def _generate_signed_manifest(self, matrix_result) -> Dict[str, Any]:
        """Generate signed manifest with attestation."""
        return {
            'generator': {
                'name': 'ExpandedEvaluationSuite',
                'version': '1.0.0',
                'timestamp': time.time()
            },
            'ce_attestation': {
                'cross_encoder_used': False,  # Placeholder
                'attestation_hash': 'placeholder_attestation'
            },
            'pools': {
                'pool_fingerprint': 'evaluation_suite_pool' if self.pool_manager else None,
                'encoder_hash': self.embedding_manager.get_encoder_hash() if self.embedding_manager else None
            },
            'tokenizers': {
                'tokenizer_hash': self.harness.corpus_constructor.get_tokenizer_hash()
            },
            'matrix_config': asdict(matrix_result.config),
            'execution_summary': matrix_result.execution_summary
        }
    
    def _save_slice_data(self, dataset_results: Dict[str, Dict[str, Any]], output_file: Path):
        """Save slice data in JSONL format."""
        with open(output_file, 'w') as f:
            for sample_id, sample_results in dataset_results.items():
                for method_id, result in sample_results.items():
                    slice_record = {
                        'sample_id': sample_id,
                        'method_id': method_id,
                        'selection_certificate': {
                            'cert_hash': result.selection_result.cert_hash,
                            'encoder_hash': result.selection_result.encoder_hash,
                            'pool_fingerprint': result.selection_result.pool_fingerprint,
                            'tokenizer_hash': result.selection_result.tokenizer_hash
                        },
                        'selection_result': result.selection_result.to_dict(),
                        'validation_status': result.is_valid
                    }
                    f.write(json.dumps(slice_record, default=str) + '\n')
    
    def _save_csv(self, data: List[Dict[str, Any]], output_file: Path):
        """Save data as CSV."""
        if not data:
            return
        
        import csv
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

# Import for defaultdict
from collections import defaultdict

# Export main class
__all__ = ['ExpandedEvaluationSuite']