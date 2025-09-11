"""
Selection Stack Diagnostic System
==================================

Fast, targeted diagnostic probes for identifying exact failure points in the
Lethe retrieval selection pipeline (S0→S1→S2→CBU). Each probe systematically
validates a specific layer to pinpoint why coverage is 0.0%.

4 Fast Probes:
1. S1 Query Vector Sanity (query embeddings not constant/broken)
2. S1 Index/Space Audit (are we retrieving anything relevant?) 
3. S2 Pair Feeding Sanity (cross-encoder sees query+candidate correctly)
4. Coverage Features Present (entity/symbol extraction working)

Provides definitive diagnosis with minimal surgical fixes.
"""

import numpy as np
import pandas as pd
import hashlib
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import time

from .probe_query_vectors import QueryVectorProbe
from .probe_index_retrieval import IndexRetrievalProbe  
from .probe_cross_encoder import CrossEncoderProbe
from .probe_coverage_features import CoverageFeaturesProbe
from ..common.evaluation_framework import EvaluationFramework

logger = logging.getLogger(__name__)

@dataclass
class ProbeResult:
    """Result from a single diagnostic probe."""
    probe_name: str
    status: str  # 'pass', 'fail', 'warning'
    summary: str
    details: Dict[str, Any]
    fix_recommendations: List[str]
    execution_time_ms: float

@dataclass  
class StackDiagnosticResult:
    """Complete diagnostic result for the selection stack."""
    overall_status: str  # 'healthy', 'degraded', 'failed'
    failure_layer: Optional[str]  # 'S1', 'S2', 'CBU' or None if healthy
    probe_results: List[ProbeResult]
    summary_metrics: Dict[str, float]
    recommended_fixes: List[str]
    parameter_recommendations: Dict[str, Any]
    execution_time_ms: float

class SelectionStackDiagnostics:
    """
    Main coordinator for selection stack diagnostic probes.
    
    Systematically probes each layer:
    - S0: Input validation and preprocessing 
    - S1: Dense retrieval (query embeddings + index search)
    - S2: Cross-encoder reranking
    - CBU: Coverage-based utility selection
    
    Provides definitive diagnosis of failure point with targeted fixes.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 evaluation_framework: Optional[EvaluationFramework] = None):
        """
        Initialize diagnostic system.
        
        Args:
            config: Configuration for probes and parameters
            evaluation_framework: Optional evaluation framework for metrics
        """
        self.config = config or self._default_config()
        self.evaluation_framework = evaluation_framework or EvaluationFramework()
        
        # Initialize probes
        self.query_probe = QueryVectorProbe(self.config.get('query_probe', {}))
        self.index_probe = IndexRetrievalProbe(self.config.get('index_probe', {}))
        self.ce_probe = CrossEncoderProbe(self.config.get('ce_probe', {}))
        self.coverage_probe = CoverageFeaturesProbe(self.config.get('coverage_probe', {}))
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for diagnostic system."""
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
                'diversity_delta': 0,  # Disable DPP temporarily
                'facility_gamma': 0.8  # Emphasize facility-location
            },
            'success_criteria': {
                'span_coverage_target': 0.15,  # 15% minimum
                'symbol_coverage_target': 0.10,  # 10% minimum
                'keep_ratio_target': 0.30  # 30% keep ratio test
            }
        }
    
    async def diagnose_stack(self, 
                           evaluation_data: List[Dict[str, Any]],
                           retrieval_pipeline: Any,
                           output_dir: Optional[Path] = None) -> StackDiagnosticResult:
        """
        Run complete diagnostic scan of the selection stack.
        
        Args:
            evaluation_data: List of evaluation samples
            retrieval_pipeline: Lethe retrieval pipeline instance
            output_dir: Optional directory for diagnostic outputs
            
        Returns:
            Complete diagnostic result with failure analysis
        """
        start_time = time.time()
        
        self.logger.info("Starting selection stack diagnostic scan...")
        self.logger.info(f"Evaluating {len(evaluation_data)} samples")
        
        # Initialize results
        probe_results = []
        overall_status = 'healthy'
        failure_layer = None
        recommended_fixes = []
        
        try:
            # Probe 1: S1 Query Vector Sanity Check
            self.logger.info("Running Probe 1: S1 Query Vector Sanity Check")
            query_result = await self._run_probe_with_timeout(
                self.query_probe.diagnose_query_vectors,
                evaluation_data,
                retrieval_pipeline,
                "Query Vector Probe"
            )
            probe_results.append(query_result)
            
            if query_result.status == 'fail':
                overall_status = 'failed'
                failure_layer = 'S1_vectors'
                recommended_fixes.extend(query_result.fix_recommendations)
                
            # Probe 2: S1 Index/Space Audit  
            self.logger.info("Running Probe 2: S1 Index/Space Audit")
            index_result = await self._run_probe_with_timeout(
                self.index_probe.diagnose_index_retrieval,
                evaluation_data,
                retrieval_pipeline,
                "Index Retrieval Probe"
            )
            probe_results.append(index_result)
            
            if index_result.status == 'fail':
                overall_status = 'failed' 
                failure_layer = failure_layer or 'S1_index'
                recommended_fixes.extend(index_result.fix_recommendations)
                
            # Probe 3: S2 Cross-Encoder Pair Feeding
            self.logger.info("Running Probe 3: S2 Cross-Encoder Pair Feeding")
            ce_result = await self._run_probe_with_timeout(
                self.ce_probe.diagnose_cross_encoder,
                evaluation_data,
                retrieval_pipeline,
                "Cross-Encoder Probe"
            )
            probe_results.append(ce_result)
            
            if ce_result.status == 'fail':
                overall_status = 'failed'
                failure_layer = failure_layer or 'S2_reranking'
                recommended_fixes.extend(ce_result.fix_recommendations)
                
            # Probe 4: Coverage Features Validation
            self.logger.info("Running Probe 4: Coverage Features Validation") 
            coverage_result = await self._run_probe_with_timeout(
                self.coverage_probe.diagnose_coverage_features,
                evaluation_data,
                retrieval_pipeline,
                "Coverage Features Probe"
            )
            probe_results.append(coverage_result)
            
            if coverage_result.status == 'fail':
                overall_status = 'failed'
                failure_layer = failure_layer or 'CBU_features'
                recommended_fixes.extend(coverage_result.fix_recommendations)
                
            # Generate summary metrics
            summary_metrics = self._compute_summary_metrics(probe_results)
            
            # Generate parameter recommendations
            param_recommendations = self._generate_parameter_recommendations(probe_results)
            
            # Determine overall health
            if overall_status == 'healthy':
                if any(r.status == 'warning' for r in probe_results):
                    overall_status = 'degraded'
                    
        except Exception as e:
            self.logger.error(f"Diagnostic scan failed: {e}")
            overall_status = 'failed'
            failure_layer = 'system_error'
            recommended_fixes.append(f"System error during diagnostics: {str(e)}")
            
        execution_time = (time.time() - start_time) * 1000
        
        result = StackDiagnosticResult(
            overall_status=overall_status,
            failure_layer=failure_layer, 
            probe_results=probe_results,
            summary_metrics=summary_metrics,
            recommended_fixes=recommended_fixes,
            parameter_recommendations=param_recommendations,
            execution_time_ms=execution_time
        )
        
        # Save results if output directory provided
        if output_dir:
            await self._save_diagnostic_results(result, output_dir)
            
        self.logger.info(f"Diagnostic scan complete in {execution_time:.1f}ms")
        self.logger.info(f"Overall status: {overall_status}")
        if failure_layer:
            self.logger.warning(f"Failure detected in: {failure_layer}")
            
        return result
    
    async def _run_probe_with_timeout(self,
                                    probe_func,
                                    evaluation_data,
                                    retrieval_pipeline,
                                    probe_name: str,
                                    timeout_seconds: int = 300) -> ProbeResult:
        """Run a single probe with timeout protection."""
        start_time = time.time()
        
        try:
            # Run probe with timeout
            import asyncio
            result = await asyncio.wait_for(
                probe_func(evaluation_data, retrieval_pipeline),
                timeout=timeout_seconds
            )
            return result
            
        except asyncio.TimeoutError:
            execution_time = (time.time() - start_time) * 1000
            return ProbeResult(
                probe_name=probe_name,
                status='fail',
                summary=f'Probe timed out after {timeout_seconds}s',
                details={'timeout_seconds': timeout_seconds},
                fix_recommendations=[f'Increase timeout or optimize {probe_name}'],
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            return ProbeResult(
                probe_name=probe_name,
                status='fail',
                summary=f'Probe failed: {str(e)}',
                details={'error': str(e)},
                fix_recommendations=[f'Fix {probe_name} implementation: {str(e)}'],
                execution_time_ms=execution_time
            )
    
    def _compute_summary_metrics(self, probe_results: List[ProbeResult]) -> Dict[str, float]:
        """Compute summary metrics across all probes."""
        total_time = sum(r.execution_time_ms for r in probe_results)
        pass_rate = sum(1 for r in probe_results if r.status == 'pass') / len(probe_results)
        
        # Extract key metrics from probe details
        metrics = {
            'total_execution_time_ms': total_time,
            'probe_pass_rate': pass_rate,
            'probes_total': len(probe_results),
            'probes_passed': sum(1 for r in probe_results if r.status == 'pass'),
            'probes_failed': sum(1 for r in probe_results if r.status == 'fail'),
            'probes_warnings': sum(1 for r in probe_results if r.status == 'warning')
        }
        
        # Add probe-specific metrics
        for result in probe_results:
            if result.probe_name == 'Query Vector Probe':
                metrics.update({
                    'query_embedding_std': result.details.get('avg_per_dim_std', 0.0),
                    'query_cosine_self_sim': result.details.get('avg_cosine_self_sim', 0.0)
                })
            elif result.probe_name == 'Index Retrieval Probe':
                metrics.update({
                    'max_similarity_mean': result.details.get('max_similarity_mean', 0.0),
                    'relevant_items_found': result.details.get('relevant_items_found', 0)
                })
            elif result.probe_name == 'Cross-Encoder Probe':
                metrics.update({
                    'ce_score_std': result.details.get('score_std', 0.0),
                    'ce_score_range': result.details.get('score_range', 0.0)
                })
            elif result.probe_name == 'Coverage Features Probe':
                metrics.update({
                    'entities_count_median': result.details.get('entities_median', 0),
                    'symbols_count_median': result.details.get('symbols_median', 0)
                })
                
        return metrics
    
    def _generate_parameter_recommendations(self, probe_results: List[ProbeResult]) -> Dict[str, Any]:
        """Generate optimized parameter recommendations based on probe results."""
        recommendations = {}
        
        # Default recommendations
        recommendations.update(self.config['controlled_parameters'])
        
        # Adjust based on probe results
        for result in probe_results:
            if result.probe_name == 'Query Vector Probe' and result.status != 'pass':
                # If query vectors are problematic, try different encoder dims
                recommendations['dims_candidates'] = [768]  # Force higher dimensionality
                
            elif result.probe_name == 'Index Retrieval Probe' and result.status != 'pass':
                # If retrieval is weak, increase K1
                recommendations['K1_candidates'] = [4000, 6000]
                
            elif result.probe_name == 'Cross-Encoder Probe' and result.status != 'pass':
                # If CE is problematic, increase K2 for more diversity
                recommendations['K2_candidates'] = [1000, 1500]
                
        return recommendations
    
    async def _save_diagnostic_results(self, result: StackDiagnosticResult, output_dir: Path):
        """Save diagnostic results to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main result as JSON
        result_dict = {
            'overall_status': result.overall_status,
            'failure_layer': result.failure_layer,
            'summary_metrics': result.summary_metrics,
            'recommended_fixes': result.recommended_fixes,
            'parameter_recommendations': result.parameter_recommendations,
            'execution_time_ms': result.execution_time_ms,
            'probe_results': [
                {
                    'probe_name': r.probe_name,
                    'status': r.status,
                    'summary': r.summary,
                    'details': r.details,
                    'fix_recommendations': r.fix_recommendations,
                    'execution_time_ms': r.execution_time_ms
                }
                for r in result.probe_results
            ]
        }
        
        result_file = output_dir / 'selection_stack_diagnostic.json'
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
            
        self.logger.info(f"Diagnostic results saved to {result_file}")
        
        # Save detailed probe outputs
        for probe_result in result.probe_results:
            probe_file = output_dir / f"{probe_result.probe_name.lower().replace(' ', '_')}_details.json"
            with open(probe_file, 'w') as f:
                json.dump(probe_result.details, f, indent=2)
    
    def print_diagnostic_report(self, result: StackDiagnosticResult):
        """Print human-readable diagnostic report."""
        print("\n" + "="*80)
        print("LETHE SELECTION STACK DIAGNOSTIC REPORT")
        print("="*80)
        
        print(f"\nOverall Status: {result.overall_status.upper()}")
        if result.failure_layer:
            print(f"Failure Layer: {result.failure_layer}")
        
        print(f"\nExecution Time: {result.execution_time_ms:.1f}ms")
        print(f"Probes Run: {len(result.probe_results)}")
        
        # Summary metrics
        print("\nSummary Metrics:")
        for metric, value in result.summary_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
        
        # Probe results
        print("\nProbe Results:")
        for probe in result.probe_results:
            status_symbol = "✓" if probe.status == 'pass' else "⚠" if probe.status == 'warning' else "✗"
            print(f"  {status_symbol} {probe.probe_name}: {probe.summary}")
            
        # Recommendations
        if result.recommended_fixes:
            print("\nRecommended Fixes:")
            for i, fix in enumerate(result.recommended_fixes, 1):
                print(f"  {i}. {fix}")
                
        # Parameter recommendations
        print("\nParameter Recommendations:")
        for param, value in result.parameter_recommendations.items():
            print(f"  {param}: {value}")
        
        print("\n" + "="*80)