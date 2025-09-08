"""
Lethe Production-Grade Information Retrieval System

This module provides production-ready implementations of BM25 and ANN retrievers
with comprehensive timing infrastructure and metadata management.

Components:
- BM25 retriever using Anserini/PySerini
- ANN retriever with HNSW and IVF-PQ indices using FAISS
- Dense embeddings management with model checkpoints
- Dual-timer system with GC barriers for accurate performance measurement
- Index metadata export and reproducibility features
"""

from .timing import TimingHarness, PerformanceProfiler, PerformanceProfile
from .metadata import IndexMetadata, MetadataManager
from .bm25 import BM25Retriever, BM25IndexBuilder
from .ann import ANNRetriever, ANNIndexBuilder
from .embeddings import DenseEmbeddingManager
from .config import RetrieverConfig

__version__ = "1.0.0"
__all__ = [
    "TimingHarness",
    "PerformanceProfiler",
    "PerformanceProfile", 
    "IndexMetadata",
    "MetadataManager",
    "BM25Retriever",
    "BM25IndexBuilder",
    "ANNRetriever", 
    "ANNIndexBuilder",
    "DenseEmbeddingManager",
    "RetrieverConfig"
]