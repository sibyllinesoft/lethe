"""
Reranking-specific telemetry system.

Extends the core telemetry with reranking-specific metrics.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

from ..fusion.telemetry import FusionTelemetry, TelemetryLogger

logger = logging.getLogger(__name__)


@dataclass
class RerankingTelemetry(FusionTelemetry):
    """Extended telemetry for reranking operations."""
    
    # Reranking-specific parameters
    cross_encoder_model: Optional[str] = None
    reranking_batch_size: int = 32
    reranking_max_length: int = 512
    
    # Reranking performance
    cross_encoder_inference_time_ms: float = 0.0
    reranking_preprocessing_time_ms: float = 0.0
    score_interpolation_time_ms: float = 0.0
    
    # Quality metrics
    score_correlation_original_vs_rerank: Optional[float] = None
    rank_correlation_kendall_tau: Optional[float] = None
    
    # Budget tracking
    budget_exceeded: bool = False
    budget_utilization_ratio: float = 0.0


class RerankingTelemetryLogger(TelemetryLogger):
    """Specialized telemetry logger for reranking experiments."""
    
    def log_reranking_run(
        self,
        run_data: Dict,
        fusion_result: Optional['FusionResult'] = None,
        rerank_result: Optional['RerankingResult'] = None,
        evaluation_metrics: Optional[Dict] = None,
        cross_encoder_stats: Optional[Dict] = None,
        budget_analysis: Optional[Dict] = None
    ):
        """
        Log a complete reranking run with extended telemetry.
        """
        # Create extended telemetry object  
        telemetry = RerankingTelemetry(
            run_id=self._generate_run_id(run_data),
            timestamp=time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime()),
            dataset=run_data.get('dataset', 'unknown'),
            query_id=run_data.get('query_id')
        )
        
        # Fill base data
        if fusion_result:
            self._populate_fusion_data(telemetry, fusion_result)
        
        # Fill reranking-specific data
        if rerank_result:
            self._populate_reranking_data(telemetry, rerank_result)
            self._populate_reranking_extended_data(telemetry, rerank_result, cross_encoder_stats)
        
        # Fill evaluation metrics
        if evaluation_metrics:
            self._populate_evaluation_metrics(telemetry, evaluation_metrics)
        
        # Fill budget analysis
        if budget_analysis:
            self._populate_budget_analysis(telemetry, budget_analysis)
        
        # Add reproducibility data (commit_sha, random_seeds)
        self._populate_reproducibility_data(telemetry, run_data)
        
        # Add environment data
        if self.collect_hardware:
            self._populate_environment_data(telemetry)
        
        # Write to JSONL
        self.file_handle.write(telemetry.to_jsonl_entry() + '\n')
        
        if self.auto_flush:
            self.file_handle.flush()
        
        logger.debug(f"Reranking telemetry logged: run_id={telemetry.run_id}")
    
    def _populate_reranking_extended_data(
        self,
        telemetry: RerankingTelemetry, 
        rerank_result: 'RerankingResult',
        cross_encoder_stats: Optional[Dict]
    ):
        """Populate reranking-specific telemetry data."""
        config = rerank_result.config
        
        # Cross-encoder configuration
        telemetry.cross_encoder_model = config.cross_encoder_model
        telemetry.reranking_batch_size = config.batch_size
        telemetry.reranking_max_length = config.max_length
        
        # Performance breakdown (p50_latency_ms, p95_latency_ms)
        if cross_encoder_stats:
            telemetry.cross_encoder_inference_time_ms = cross_encoder_stats.get(
                'inference_time_ms', 0.0
            )
            telemetry.reranking_preprocessing_time_ms = cross_encoder_stats.get(
                'preprocessing_time_ms', 0.0
            )
            telemetry.score_interpolation_time_ms = cross_encoder_stats.get(
                'interpolation_time_ms', 0.0
            )
        
        # Quality metrics
        original_scores = list(rerank_result.original_scores.values())
        rerank_scores = list(rerank_result.rerank_scores.values())
        
        if len(original_scores) > 1 and len(rerank_scores) > 1:
            try:
                import numpy as np
                from scipy.stats import kendalltau
                
                # Score correlation
                correlation = np.corrcoef(original_scores, rerank_scores)[0, 1]
                telemetry.score_correlation_original_vs_rerank = float(correlation)
                
                # Rank correlation
                tau, _ = kendalltau(original_scores, rerank_scores)
                telemetry.rank_correlation_kendall_tau = float(tau)
                
            except ImportError:
                logger.warning("scipy not available for correlation computation")
            except Exception as e:
                logger.warning(f"Failed to compute correlations: {e}")
    
    def _populate_budget_analysis(self, telemetry: RerankingTelemetry, budget_analysis: Dict):
        """Populate budget analysis data."""
        telemetry.budget_exceeded = budget_analysis.get('budget_exceeded', False)
        telemetry.budget_utilization_ratio = budget_analysis.get('utilization_ratio', 0.0)


def analyze_reranking_telemetry(telemetry_path: str) -> Dict:
    """
    Analyze reranking telemetry data for insights.
    
    Args:
        telemetry_path: Path to JSONL telemetry file
        
    Returns:
        Analysis results
    """
    import json
    from pathlib import Path
    import numpy as np
    
    entries = []
    
    try:
        with open(telemetry_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    entries.append(data)
    except FileNotFoundError:
        logger.error(f"Telemetry file not found: {telemetry_path}")
        return {}
    
    if not entries:
        return {}
    
    analysis = {
        'total_runs': len(entries),
        'datasets': list(set(e.get('dataset', 'unknown') for e in entries)),
        'alpha_values': sorted(list(set(e.get('alpha', 0.0) for e in entries))),
        'beta_values': sorted(list(set(e.get('beta', 0.0) for e in entries))),
        'performance': {},
        'quality': {},
        'budget': {}
    }
    
    # Performance analysis
    latencies = [e.get('total_latency_ms', 0) for e in entries if e.get('total_latency_ms', 0) > 0]
    if latencies:
        analysis['performance'] = {
            'mean_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_range_qps': [
                min(e.get('throughput_qps', 0) for e in entries),
                max(e.get('throughput_qps', 0) for e in entries)
            ]
        }
    
    # Quality analysis
    correlations = [e.get('score_correlation_original_vs_rerank') for e in entries 
                   if e.get('score_correlation_original_vs_rerank') is not None]
    if correlations:
        analysis['quality']['score_correlations'] = {
            'mean': np.mean(correlations),
            'std': np.std(correlations),
            'range': [min(correlations), max(correlations)]
        }
    
    # Budget analysis
    budget_exceeded_count = sum(1 for e in entries if e.get('budget_exceeded', False))
    analysis['budget'] = {
        'budget_violation_rate': budget_exceeded_count / len(entries),
        'mean_budget_utilization': np.mean([
            e.get('budget_utilization_ratio', 0) for e in entries
        ])
    }
    
    # Alpha-Beta performance breakdown
    alpha_beta_groups = {}
    for entry in entries:
        alpha = entry.get('alpha', 0.0)
        beta = entry.get('beta', 0.0)
        key = f"α={alpha:.1f}_β={beta:.1f}"
        
        if key not in alpha_beta_groups:
            alpha_beta_groups[key] = []
        alpha_beta_groups[key].append(entry)
    
    analysis['alpha_beta_breakdown'] = {}
    for key, group in alpha_beta_groups.items():
        group_latencies = [e.get('total_latency_ms', 0) for e in group if e.get('total_latency_ms', 0) > 0]
        if group_latencies:
            analysis['alpha_beta_breakdown'][key] = {
                'count': len(group),
                'mean_latency_ms': np.mean(group_latencies),
                'p95_latency_ms': np.percentile(group_latencies, 95)
            }
    
    return analysis