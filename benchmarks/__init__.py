#!/usr/bin/env python3
"""
Comprehensive Benchmarking Infrastructure for Lethe-Hybrid
===========================================================

This package implements a complete benchmarking system comparing Lethe-Hybrid
against 5 categories of open-source retrieval leaders:

1. Hybrid Vector DBs: Weaviate, Milvus, Vespa, OpenSearch
2. Learned Sparse/Late-Interaction: SPLADE v2, ColBERTv2, RAGatouille
3. Open Rerankers: BGE-reranker-large/v2-m3, MonoT5
4. Code Search & Graph: Zoekt, livegrep, GraphRAG
5. Long-Context Algorithms: StreamingLLM, LongNet, BGE-M3

Features:
- Docker orchestration for all competitor systems
- Fair evaluation with matched budgets (8%, 15%, 30% keep_ratio)
- Statistical rigor with paired bootstrap + permutation testing
- Marketing-ready HTML reports with interactive charts
- Complete reproducibility with JSONL logs and config snapshots
"""

__version__ = "1.0.0"
__author__ = "Lethe Research Team"

from .orchestrator import BenchmarkOrchestrator
from .config import BenchmarkConfig
from .competitors import CompetitorRegistry
from .datasets import DatasetRegistry
from .evaluation import EvaluationEngine
from .reporting import ReportGenerator

__all__ = [
    "BenchmarkOrchestrator",
    "BenchmarkConfig", 
    "CompetitorRegistry",
    "DatasetRegistry",
    "EvaluationEngine",
    "ReportGenerator"
]