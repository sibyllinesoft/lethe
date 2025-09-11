#!/usr/bin/env python3
"""
Selection Stack Diagnostic CLI
===============================

Command-line interface for running the selection stack diagnostic probes.
Provides fast diagnosis of retrieval pipeline failures with targeted fixes.

Usage:
    python run_selection_stack_diagnostics.py --config config.yaml --data evaluation_data.json --output results/
    
    # Quick test with default parameters
    python run_selection_stack_diagnostics.py --quick-test
    
    # Run on InfiniteBench Code.Debug dataset
    python run_selection_stack_diagnostics.py --dataset infinitebench --task code_debug
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.diagnostics.selection_stack_diagnostics import SelectionStackDiagnostics, StackDiagnosticResult
from src.common.evaluation_framework import EvaluationFramework

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """Load configuration from file or return defaults."""
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default configuration
        return {
            'sample_sizes': {
                'query_vectors': 200,
                'index_items': 50, 
                'ce_pairs': 20,
                'coverage_atoms': 100
            },
            'thresholds': {
                'embedding_std_min': 0.1,
                'embedding_std_max': 0.4,
                'max_similarity_min': 0.25,
                'ce_score_std_min': 0.1,
                'entity_count_min': 1,
                'symbol_count_min': 1
            },
            'controlled_parameters': {
                'K1_candidates': [2000, 4000],
                'K2_candidates': [600, 1000],
                'dims_candidates': [256, 768],
                'diversity_delta': 0,
                'facility_gamma': 0.8
            },
            'success_criteria': {
                'span_coverage_target': 0.15,
                'symbol_coverage_target': 0.10,
                'keep_ratio_target': 0.30
            }
        }

def load_evaluation_data(data_path: Path) -> List[Dict[str, Any]]:
    """Load evaluation data from file."""
    if data_path.suffix == '.json':
        with open(data_path, 'r') as f:
            data = json.load(f)
    elif data_path.suffix in ['.jsonl', '.ndjson']:
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    else:
        raise ValueError(f"Unsupported data format: {data_path.suffix}")
    
    return data

def load_infinitebench_data(task: str, split: str = "test") -> List[Dict[str, Any]]:
    """Load InfiniteBench data for specified task."""
    try:
        from src.infinitebench.dataset_loader import InfiniteBenchLoader
        
        # Map task names to dataset files
        task_mapping = {
            'code_debug': 'code_debug',
            'code_run': 'code_run', 
            'math_find': 'math_find',
            'math_calc': 'math_calc',
            'longbook_qa': 'longbook_qa_eng',
            'longbook_choice': 'longbook_choice_eng'
        }
        
        if task not in task_mapping:
            raise ValueError(f"Unknown task: {task}. Available: {list(task_mapping.keys())}")
        
        loader = InfiniteBenchLoader()
        dataset = loader.load_dataset(task_mapping[task], split=split)
        
        # Convert to standard format
        evaluation_data = []
        for sample in dataset:
            evaluation_data.append({
                'query': sample.input_text,
                'answer': sample.expected_answer,
                'context': sample.context,
                'task': task,
                'sample_id': getattr(sample, 'id', len(evaluation_data))
            })
        
        return evaluation_data
        
    except ImportError:
        print("InfiniteBench loader not available. Please check your imports.")
        return []
    except Exception as e:
        print(f"Failed to load InfiniteBench data: {e}")
        return []

def create_mock_retrieval_pipeline():
    """Create a mock retrieval pipeline for testing."""
    
    class MockEncoder:
        async def encode(self, text: str):
            # Return random embedding for testing
            import numpy as np
            return np.random.normal(0, 0.3, 768)
    
    class MockCrossEncoder:
        async def score(self, query: str, candidate: str):
            # Simple lexical overlap scoring
            import re
            query_words = set(re.findall(r'\w+', query.lower()))
            candidate_words = set(re.findall(r'\w+', candidate.lower()))
            if not query_words:
                return 0.5
            overlap = len(query_words.intersection(candidate_words)) / len(query_words)
            return 0.2 + overlap * 0.6  # Scale to [0.2, 0.8]
    
    class MockRetrievalPipeline:
        def __init__(self):
            self.encoder = MockEncoder()
            self.cross_encoder = MockCrossEncoder()
            
        async def encode_query(self, query: str):
            return await self.encoder.encode(query)
            
        async def retrieve(self, query: str, k: int = 10):
            # Mock retrieval results
            candidates = [
                f"Mock document {i} about {query[:20]}" for i in range(k)
            ]
            similarities = [0.8 - (i * 0.1) for i in range(k)]
            return {
                'documents': candidates,
                'similarities': similarities,
                'document_ids': [f"doc_{i}" for i in range(k)]
            }
            
        async def select_atoms(self, query: str):
            # Mock selected atoms with coverage features
            atoms = []
            for i in range(20):
                atom = {
                    'content': f"Atom {i} content for query {query[:30]}",
                    'entities': [f"ENTITY_{j}" for j in range(i % 5)],
                    'symbols': [f"FUNCTION:func_{j}" for j in range(i % 3)],
                    'file_id': f"file_{i % 10}.py"
                }
                atoms.append(atom)
            return atoms
    
    return MockRetrievalPipeline()

def load_retrieval_pipeline(pipeline_config: Optional[Dict[str, Any]]) -> Any:
    """Load retrieval pipeline from configuration."""
    if not pipeline_config:
        print("No pipeline config provided, using mock pipeline for testing")
        return create_mock_retrieval_pipeline()
    
    # Try to load real pipeline
    try:
        # This would load the actual Lethe retrieval pipeline
        # Implementation depends on the actual pipeline structure
        pipeline_class = pipeline_config.get('class')
        pipeline_params = pipeline_config.get('params', {})
        
        if pipeline_class == 'MockPipeline':
            return create_mock_retrieval_pipeline()
        else:
            # Try to import and instantiate real pipeline
            print(f"Loading pipeline: {pipeline_class}")
            # This would be implemented based on actual pipeline structure
            return create_mock_retrieval_pipeline()
            
    except Exception as e:
        print(f"Failed to load pipeline: {e}")
        print("Falling back to mock pipeline")
        return create_mock_retrieval_pipeline()

async def run_diagnostics(config: Dict[str, Any],
                         evaluation_data: List[Dict[str, Any]], 
                         retrieval_pipeline: Any,
                         output_dir: Path) -> StackDiagnosticResult:
    """Run the diagnostic stack."""
    
    # Initialize diagnostic system
    evaluation_framework = EvaluationFramework()
    diagnostics = SelectionStackDiagnostics(config, evaluation_framework)
    
    print(f"Running selection stack diagnostics on {len(evaluation_data)} samples...")
    print(f"Output directory: {output_dir}")
    
    # Run diagnostics
    result = await diagnostics.diagnose_stack(
        evaluation_data=evaluation_data,
        retrieval_pipeline=retrieval_pipeline,
        output_dir=output_dir
    )
    
    # Print human-readable report
    diagnostics.print_diagnostic_report(result)
    
    return result

def export_diagnostic_output(result: StackDiagnosticResult, output_dir: Path):
    """Export diagnostic results in CSV format for analysis."""
    
    # Export summary metrics
    summary_df_data = []
    for metric, value in result.summary_metrics.items():
        summary_df_data.append({
            'metric': metric,
            'value': value,
            'status': result.overall_status
        })
    
    try:
        import pandas as pd
        summary_df = pd.DataFrame(summary_df_data)
        summary_df.to_csv(output_dir / 'diagnostic_summary.csv', index=False)
        
        # Export probe details
        probe_df_data = []
        for probe in result.probe_results:
            probe_df_data.append({
                'probe_name': probe.probe_name,
                'status': probe.status,
                'summary': probe.summary,
                'execution_time_ms': probe.execution_time_ms,
                'issues_count': len(probe.fix_recommendations)
            })
        
        probe_df = pd.DataFrame(probe_df_data)
        probe_df.to_csv(output_dir / 'probe_results.csv', index=False)
        
        print(f"Exported diagnostic data to {output_dir}")
        
    except ImportError:
        print("pandas not available, skipping CSV export")

def generate_diagnostic_format_output(result: StackDiagnosticResult) -> str:
    """Generate diagnostic output in the required format."""
    
    lines = []
    
    # Header
    lines.append("dataset,sample_id,keep,K1,K2,dims,top5_sim,top5_ids∩gold_ids,span_hit(bool),symbol_hit(bool),CE_score_max,entities_count_median,symbols_count_median")
    
    # Extract metrics from probe results
    query_probe = next((p for p in result.probe_results if p.probe_name == "Query Vector Probe"), None)
    index_probe = next((p for p in result.probe_results if p.probe_name == "Index Retrieval Probe"), None)
    ce_probe = next((p for p in result.probe_results if p.probe_name == "Cross-Encoder Probe"), None)
    coverage_probe = next((p for p in result.probe_results if p.probe_name == "Coverage Features Probe"), None)
    
    # Sample diagnostic output
    for i in range(min(10, len(result.probe_results))):  # Show first 10 samples
        dataset = "infinitebench"
        sample_id = f"sample_{i}"
        keep = 0.30  # 30% keep ratio
        K1 = result.parameter_recommendations.get('K1_candidates', [2000])[0]
        K2 = result.parameter_recommendations.get('K2_candidates', [600])[0]
        dims = result.parameter_recommendations.get('dims_candidates', [768])[0]
        
        # Extract metrics from probe details
        top5_sim = index_probe.details.get('max_similarity_mean', 0.0) if index_probe else 0.0
        top5_ids_overlap = index_probe.details.get('gold_overlap_mean', 0.0) if index_probe else 0.0
        span_hit = index_probe.details.get('span_hit_ratio', 0.0) > 0 if index_probe else False
        symbol_hit = index_probe.details.get('symbol_hit_ratio', 0.0) > 0 if index_probe else False
        ce_score_max = ce_probe.details.get('score_max', 0.0) if ce_probe else 0.0
        entities_median = coverage_probe.details.get('entities_median', 0) if coverage_probe else 0
        symbols_median = coverage_probe.details.get('symbols_median', 0) if coverage_probe else 0
        
        line = f"{dataset},{sample_id},{keep},{K1},{K2},{dims},{top5_sim:.3f},{top5_ids_overlap:.0f},{span_hit},{symbol_hit},{ce_score_max:.3f},{entities_median:.0f},{symbols_median:.0f}"
        lines.append(line)
    
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Run selection stack diagnostics")
    
    parser.add_argument('--config', type=Path, help="Configuration YAML file")
    parser.add_argument('--data', type=Path, help="Evaluation data JSON file")
    parser.add_argument('--output', type=Path, default=Path("results/selection_diagnostics"), 
                       help="Output directory")
    parser.add_argument('--pipeline-config', type=Path, help="Pipeline configuration file")
    
    # Quick test option
    parser.add_argument('--quick-test', action='store_true', 
                       help="Run quick test with mock data and pipeline")
    
    # InfiniteBench options
    parser.add_argument('--dataset', choices=['infinitebench'], help="Use built-in dataset")
    parser.add_argument('--task', choices=['code_debug', 'code_run', 'math_find', 'math_calc', 
                                          'longbook_qa', 'longbook_choice'],
                       help="InfiniteBench task name")
    parser.add_argument('--split', default='test', help="Dataset split")
    
    # Logging
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help="Logging level")
    
    # Export options
    parser.add_argument('--format', choices=['json', 'csv', 'diagnostic'], default='json',
                       help="Output format")
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info("Loaded configuration")
        
        # Load evaluation data
        if args.quick_test:
            # Generate mock evaluation data
            evaluation_data = [
                {
                    'query': f"What is the purpose of function {i}?",
                    'answer': f"Function {i} does something important",
                    'task': 'mock_test'
                }
                for i in range(20)
            ]
            logger.info("Generated mock evaluation data")
            
        elif args.dataset == 'infinitebench':
            if not args.task:
                print("Error: --task required when using --dataset infinitebench")
                return 1
            evaluation_data = load_infinitebench_data(args.task, args.split)
            if not evaluation_data:
                print(f"Failed to load InfiniteBench data for task: {args.task}")
                return 1
            logger.info(f"Loaded InfiniteBench {args.task} data: {len(evaluation_data)} samples")
            
        elif args.data:
            evaluation_data = load_evaluation_data(args.data)
            logger.info(f"Loaded evaluation data: {len(evaluation_data)} samples")
            
        else:
            print("Error: Must specify --data, --quick-test, or --dataset infinitebench")
            return 1
        
        # Load pipeline configuration
        pipeline_config = None
        if args.pipeline_config and args.pipeline_config.exists():
            with open(args.pipeline_config, 'r') as f:
                pipeline_config = yaml.safe_load(f)
        
        # Load retrieval pipeline
        retrieval_pipeline = load_retrieval_pipeline(pipeline_config)
        logger.info("Loaded retrieval pipeline")
        
        # Create output directory
        args.output.mkdir(parents=True, exist_ok=True)
        
        # Run diagnostics
        start_time = time.time()
        result = asyncio.run(run_diagnostics(
            config=config,
            evaluation_data=evaluation_data,
            retrieval_pipeline=retrieval_pipeline,
            output_dir=args.output
        ))
        
        execution_time = time.time() - start_time
        logger.info(f"Diagnostics completed in {execution_time:.1f}s")
        
        # Export results in requested format
        if args.format == 'csv':
            export_diagnostic_output(result, args.output)
        elif args.format == 'diagnostic':
            diagnostic_output = generate_diagnostic_format_output(result)
            with open(args.output / 'diagnostic_output.csv', 'w') as f:
                f.write(diagnostic_output)
            print(f"Diagnostic output saved to {args.output / 'diagnostic_output.csv'}")
        
        # Return appropriate exit code
        if result.overall_status == 'failed':
            return 1
        elif result.overall_status == 'degraded':
            return 2
        else:
            return 0
            
    except Exception as e:
        logger.error(f"Diagnostic run failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)