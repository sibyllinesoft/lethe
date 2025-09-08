"""
Comprehensive telemetry logging system for hybrid fusion.

Captures all required metrics for reproducibility:
- Per-run metrics: dataset, seeds, α/β, k_init/K, index params
- Performance: p50/p95 latency, throughput, memory, ANN recall
- Quality: nDCG@{10,5}, Recall@{10,20}, MRR@10
- Reproducibility: commit SHA, hashes, seeds
"""

import logging
import json
import time
import hashlib
import psutil
import os
import subprocess
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FusionTelemetry:
    """Comprehensive telemetry data for a single fusion run."""
    
    # Run identification
    run_id: str
    timestamp: str
    dataset: str
    query_id: Optional[str] = None
    
    # Configuration parameters
    alpha: float = 0.0
    beta: float = 0.0
    k_init_sparse: int = 1000
    k_init_dense: int = 1000
    k_final: int = 100
    k_rerank: Optional[int] = None
    
    # Index parameters (for budget parity)
    bm25_params: Dict = None
    ann_params: Dict = None
    efSearch: Optional[int] = None
    nlist: Optional[int] = None
    nprobe: Optional[int] = None
    nbits: Optional[int] = None
    
    # Performance metrics
    total_latency_ms: float = 0.0
    sparse_latency_ms: float = 0.0
    dense_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    throughput_qps: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Candidate counts
    sparse_candidates: int = 0
    dense_candidates: int = 0
    union_candidates: int = 0
    final_results: int = 0
    
    # Quality metrics
    ann_recall_at_1k: float = 0.0
    budget_parity_maintained: bool = False
    
    # Evaluation metrics (if available)
    ndcg_at_10: Optional[float] = None
    ndcg_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None
    recall_at_20: Optional[float] = None
    mrr_at_10: Optional[float] = None
    
    # Invariant validation
    invariants_passed: int = 0
    invariant_violations: List[str] = None
    
    # Reproducibility data
    commit_sha: Optional[str] = None
    data_hash: Optional[str] = None
    index_hash: Optional[str] = None
    model_checkpoint_sha: Optional[str] = None
    random_seeds: Dict = None
    
    # Environment
    python_version: Optional[str] = None
    hardware_info: Dict = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.bm25_params is None:
            self.bm25_params = {}
        if self.ann_params is None:
            self.ann_params = {}
        if self.invariant_violations is None:
            self.invariant_violations = []
        if self.random_seeds is None:
            self.random_seeds = {}
        if self.hardware_info is None:
            self.hardware_info = {}
    
    def to_jsonl_entry(self) -> str:
        """Convert to JSONL format for logging."""
        return json.dumps(asdict(self), separators=(',', ':'))
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FusionTelemetry':
        """Create from dictionary."""
        return cls(**data)


class TelemetryLogger:
    """
    Comprehensive telemetry logger with JSONL output.
    
    Handles all metrics collection and persistent logging for reproducibility.
    """
    
    def __init__(
        self,
        output_path: Union[str, Path],
        auto_flush: bool = True,
        collect_hardware: bool = True
    ):
        """
        Initialize telemetry logger.
        
        Args:
            output_path: Path to JSONL output file
            auto_flush: Whether to flush after each write
            collect_hardware: Whether to collect hardware info
        """
        self.output_path = Path(output_path)
        self.auto_flush = auto_flush
        self.collect_hardware = collect_hardware
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open file for writing
        self.file_handle = open(self.output_path, 'a', encoding='utf-8')
        
        # Cache static info
        self._cached_hardware = None
        self._cached_commit_sha = None
        
        logger.info(f"TelemetryLogger initialized: {self.output_path}")
    
    def log_fusion_run(
        self,
        run_data: Dict,
        fusion_result: Optional['FusionResult'] = None,
        rerank_result: Optional['RerankingResult'] = None,
        evaluation_metrics: Optional[Dict] = None,
        invariant_results: Optional[List] = None
    ):
        """
        Log a complete fusion run with all telemetry.
        
        Args:
            run_data: Basic run information (dataset, query, etc.)
            fusion_result: Result from fusion system
            rerank_result: Optional result from reranking
            evaluation_metrics: Optional evaluation metrics
            invariant_results: Optional invariant validation results
        """
        start_time = time.time()
        
        # Create telemetry object
        telemetry = FusionTelemetry(
            run_id=self._generate_run_id(run_data),
            timestamp=time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime()),
            dataset=run_data.get('dataset', 'unknown'),
            query_id=run_data.get('query_id')
        )
        
        # Fill fusion data
        if fusion_result:
            self._populate_fusion_data(telemetry, fusion_result)
        
        # Fill reranking data
        if rerank_result:
            self._populate_reranking_data(telemetry, rerank_result)
        
        # Fill evaluation metrics
        if evaluation_metrics:
            self._populate_evaluation_metrics(telemetry, evaluation_metrics)
        
        # Fill invariant results
        if invariant_results:
            self._populate_invariant_results(telemetry, invariant_results)
        
        # Add reproducibility data
        self._populate_reproducibility_data(telemetry, run_data)
        
        # Add environment data
        if self.collect_hardware:
            self._populate_environment_data(telemetry)
        
        # Write to JSONL
        self.file_handle.write(telemetry.to_jsonl_entry() + '\n')
        
        if self.auto_flush:
            self.file_handle.flush()
        
        log_time = (time.time() - start_time) * 1000
        logger.debug(f"Telemetry logged in {log_time:.1f}ms: run_id={telemetry.run_id}")
    
    def _generate_run_id(self, run_data: Dict) -> str:
        """Generate unique run ID."""
        timestamp = str(int(time.time() * 1000))  # millisecond timestamp
        data_key = json.dumps(run_data, sort_keys=True)
        hash_part = hashlib.md5(data_key.encode()).hexdigest()[:8]
        return f"{timestamp}_{hash_part}"
    
    def _populate_fusion_data(self, telemetry: FusionTelemetry, fusion_result: 'FusionResult'):
        """Populate telemetry with fusion result data."""
        config = fusion_result.config
        
        # Configuration
        telemetry.alpha = config.alpha
        telemetry.k_init_sparse = config.k_init_sparse
        telemetry.k_init_dense = config.k_init_dense
        telemetry.k_final = config.k_final
        telemetry.bm25_params = config.bm25_params.copy()
        telemetry.ann_params = config.ann_params.copy()
        
        # Performance
        telemetry.total_latency_ms = fusion_result.total_latency_ms
        telemetry.sparse_latency_ms = fusion_result.sparse_latency_ms
        telemetry.dense_latency_ms = fusion_result.dense_latency_ms
        telemetry.fusion_latency_ms = fusion_result.fusion_latency_ms
        
        # Candidates
        telemetry.sparse_candidates = fusion_result.sparse_candidates
        telemetry.dense_candidates = fusion_result.dense_candidates
        telemetry.union_candidates = fusion_result.union_candidates
        telemetry.final_results = len(fusion_result.doc_ids)
        
        # Quality
        telemetry.ann_recall_at_1k = fusion_result.ann_recall_achieved
        telemetry.budget_parity_maintained = fusion_result.budget_parity_maintained
    
    def _populate_reranking_data(self, telemetry: FusionTelemetry, rerank_result: 'RerankingResult'):
        """Populate telemetry with reranking result data."""
        config = rerank_result.config
        
        # Configuration
        telemetry.beta = config.beta
        telemetry.k_rerank = config.k_rerank
        
        # Performance
        telemetry.rerank_latency_ms = rerank_result.reranking_latency_ms
        telemetry.total_latency_ms = rerank_result.total_latency_ms  # Override with final
        telemetry.p95_latency_ms = rerank_result.p95_latency_ms
        telemetry.throughput_qps = rerank_result.throughput_qps
        
        # Update final results count
        telemetry.final_results = len(rerank_result.doc_ids)
    
    def _populate_evaluation_metrics(self, telemetry: FusionTelemetry, metrics: Dict):
        """Populate telemetry with evaluation metrics."""
        telemetry.ndcg_at_10 = metrics.get('ndcg@10')
        telemetry.ndcg_at_5 = metrics.get('ndcg@5')
        telemetry.recall_at_10 = metrics.get('recall@10')
        telemetry.recall_at_20 = metrics.get('recall@20')
        telemetry.mrr_at_10 = metrics.get('mrr@10')
    
    def _populate_invariant_results(self, telemetry: FusionTelemetry, results: List):
        """Populate telemetry with invariant validation results."""
        passed_count = sum(1 for r in results if r.passed)
        telemetry.invariants_passed = passed_count
        
        violations = [r.invariant_id for r in results if not r.passed]
        telemetry.invariant_violations = violations
    
    def _populate_reproducibility_data(self, telemetry: FusionTelemetry, run_data: Dict):
        """Populate telemetry with reproducibility data."""
        # Commit SHA
        if not self._cached_commit_sha:
            self._cached_commit_sha = self._get_git_commit_sha()
        telemetry.commit_sha = self._cached_commit_sha
        
        # Data and index hashes
        telemetry.data_hash = run_data.get('data_hash')
        telemetry.index_hash = run_data.get('index_hash')
        telemetry.model_checkpoint_sha = run_data.get('model_checkpoint_sha')
        
        # Random seeds
        telemetry.random_seeds = run_data.get('random_seeds', {})
    
    def _populate_environment_data(self, telemetry: FusionTelemetry):
        """Populate telemetry with environment data."""
        import sys
        
        # Python version
        telemetry.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Hardware info (cached)
        if not self._cached_hardware:
            self._cached_hardware = self._collect_hardware_info()
        telemetry.hardware_info = self._cached_hardware
        
        # Memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        telemetry.memory_usage_mb = memory_info.rss / (1024 * 1024)
    
    def _get_git_commit_sha(self) -> Optional[str]:
        """Get current Git commit SHA."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.warning(f"Failed to get Git commit SHA: {e}")
        
        return None
    
    def _collect_hardware_info(self) -> Dict:
        """Collect hardware information."""
        info = {}
        
        try:
            # CPU info
            info['cpu_count'] = psutil.cpu_count()
            info['cpu_count_logical'] = psutil.cpu_count(logical=True)
            
            # Memory info
            memory = psutil.virtual_memory()
            info['memory_total_gb'] = round(memory.total / (1024**3), 1)
            
            # Platform info
            import platform
            info['platform'] = platform.platform()
            info['architecture'] = platform.architecture()[0]
            
            # GPU info (if available)
            try:
                import torch
                if torch.cuda.is_available():
                    info['gpu_available'] = True
                    info['gpu_count'] = torch.cuda.device_count()
                    info['gpu_name'] = torch.cuda.get_device_name(0)
                else:
                    info['gpu_available'] = False
            except ImportError:
                info['gpu_available'] = False
            
        except Exception as e:
            logger.warning(f"Failed to collect hardware info: {e}")
            info['error'] = str(e)
        
        return info
    
    def compute_latency_percentiles(self, latencies: List[float]) -> Dict[str, float]:
        """Compute latency percentiles for telemetry."""
        if not latencies:
            return {'p50': 0.0, 'p95': 0.0, 'p99': 0.0}
        
        latency_array = np.array(latencies)
        
        return {
            'p50': float(np.percentile(latency_array, 50)),
            'p95': float(np.percentile(latency_array, 95)),
            'p99': float(np.percentile(latency_array, 99)),
            'mean': float(np.mean(latency_array)),
            'std': float(np.std(latency_array))
        }
    
    def flush(self):
        """Flush file buffer."""
        self.file_handle.flush()
    
    def close(self):
        """Close telemetry logger."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
        logger.info(f"TelemetryLogger closed: {self.output_path}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TelemetryAnalyzer:
    """
    Analyzer for telemetry data.
    
    Provides utilities to process and analyze logged telemetry.
    """
    
    def __init__(self, telemetry_path: Union[str, Path]):
        """Initialize analyzer."""
        self.telemetry_path = Path(telemetry_path)
        
    def load_telemetry(self) -> List[FusionTelemetry]:
        """Load all telemetry entries from JSONL file."""
        entries = []
        
        try:
            with open(self.telemetry_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        entry = FusionTelemetry.from_dict(data)
                        entries.append(entry)
        except FileNotFoundError:
            logger.warning(f"Telemetry file not found: {self.telemetry_path}")
        except Exception as e:
            logger.error(f"Failed to load telemetry: {e}")
        
        return entries
    
    def analyze_performance(self, entries: List[FusionTelemetry]) -> Dict:
        """Analyze performance metrics across runs."""
        if not entries:
            return {}
        
        latencies = [e.total_latency_ms for e in entries if e.total_latency_ms > 0]
        throughputs = [e.throughput_qps for e in entries if e.throughput_qps > 0]
        memory_usage = [e.memory_usage_mb for e in entries if e.memory_usage_mb > 0]
        
        analysis = {}
        
        if latencies:
            analysis['latency'] = {
                'mean_ms': np.mean(latencies),
                'median_ms': np.median(latencies),
                'p95_ms': np.percentile(latencies, 95),
                'p99_ms': np.percentile(latencies, 99),
                'std_ms': np.std(latencies)
            }
        
        if throughputs:
            analysis['throughput'] = {
                'mean_qps': np.mean(throughputs),
                'median_qps': np.median(throughputs),
                'max_qps': np.max(throughputs)
            }
        
        if memory_usage:
            analysis['memory'] = {
                'mean_mb': np.mean(memory_usage),
                'max_mb': np.max(memory_usage),
                'std_mb': np.std(memory_usage)
            }
        
        return analysis
    
    def analyze_by_alpha(self, entries: List[FusionTelemetry]) -> Dict:
        """Analyze performance by alpha parameter."""
        alpha_groups = {}
        
        for entry in entries:
            alpha = entry.alpha
            if alpha not in alpha_groups:
                alpha_groups[alpha] = []
            alpha_groups[alpha].append(entry)
        
        analysis = {}
        for alpha, group_entries in alpha_groups.items():
            analysis[alpha] = self.analyze_performance(group_entries)
            analysis[alpha]['count'] = len(group_entries)
        
        return analysis