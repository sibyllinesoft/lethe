"""
InfinityBench Integration for Lethe Evaluation
Academic-quality long-context evaluation framework.
"""

from .dataset_loader import InfinityBenchDataset
from .metrics import compute_metrics
from .baselines import BM25Baseline, NaiveChunkingBaseline
from .evaluation_pipeline import run_evaluation
from .statistical_analysis import compute_statistical_analysis

__version__ = "1.0.0"
__all__ = [
    "InfinityBenchDataset", 
    "compute_metrics", 
    "BM25Baseline",
    "NaiveChunkingBaseline", 
    "run_evaluation",
    "compute_statistical_analysis"
]