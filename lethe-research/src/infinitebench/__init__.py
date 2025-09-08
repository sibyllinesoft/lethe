"""
InfiniteBench Integration for Lethe Framework
===========================================

This module provides a comprehensive integration of the InfiniteBench dataset
for evaluating long-context retrieval systems. It includes:

- Dataset loading and preprocessing utilities
- Evaluation metrics (ROUGE-L, EM, F1, nDCG@k)
- Baseline comparison implementations
- Statistical analysis integration
- Academic publication reporting

Authors: Lethe Research Team
Version: 1.0.0
License: MIT
"""

from .dataset_loader import InfiniteBenchLoader
from .evaluation_pipeline import InfiniteBenchEvaluator
from .baselines import BM25Baseline, DenseRetrievalBaseline, NaiveChunkingBaseline
from .metrics import InfiniteBenchMetrics
from .statistical_analysis import InfiniteBenchStatistics

__all__ = [
    'InfiniteBenchLoader',
    'InfiniteBenchEvaluator', 
    'BM25Baseline',
    'DenseRetrievalBaseline',
    'NaiveChunkingBaseline',
    'InfiniteBenchMetrics',
    'InfiniteBenchStatistics'
]

__version__ = "1.0.0"