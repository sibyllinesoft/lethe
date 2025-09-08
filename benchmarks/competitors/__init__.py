#!/usr/bin/env python3
"""
Competitor Integration System
=============================

Unified interfaces for all 5 competitor categories:

1. Hybrid Vector DBs: Weaviate, Milvus, Vespa, OpenSearch
2. Learned Sparse/Late-Interaction: SPLADE v2, ColBERTv2, RAGatouille  
3. Open Rerankers: BGE-reranker-large/v2-m3, MonoT5
4. Code Search & Graph: Zoekt, livegrep, GraphRAG
5. Long-Context Algorithms: StreamingLLM, LongNet, BGE-M3

Each competitor provides consistent interfaces for fair evaluation
with vendor-recommended configurations.
"""

from .registry import CompetitorRegistry
from .base import BaseCompetitor, CompetitorResult, CompetitorMetrics
from .hybrid_vector_db import *
from .learned_sparse import *
from .rerankers import *
from .code_search import *  
from .long_context import *
from .lethe_baseline import LetheHybridCompetitor

__all__ = [
    "CompetitorRegistry",
    "BaseCompetitor",
    "CompetitorResult", 
    "CompetitorMetrics",
    # Hybrid Vector DBs
    "WeaviateCompetitor",
    "MilvusCompetitor", 
    "VespaCompetitor",
    "OpenSearchCompetitor",
    # Learned Sparse
    "SpladeV2Competitor",
    "ColBERTV2Competitor",
    "RAGatouilleCompetitor",
    # Rerankers
    "BGERerankerCompetitor",
    "BGEM3RerankerCompetitor", 
    "MonoT5Competitor",
    # Code Search
    "ZoektCompetitor",
    "LivegrepCompetitor",
    "GraphRAGCompetitor",
    # Long Context
    "StreamingLLMCompetitor",
    "LongNetCompetitor",
    "BGEM3BaselineCompetitor",
    # Lethe Baseline
    "LetheHybridCompetitor"
]