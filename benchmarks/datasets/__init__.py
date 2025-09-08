#!/usr/bin/env python3
"""
Dataset Integration System
===========================

Unified interface for loading and validating benchmark datasets from multiple sources:
- InfiniteBench (core evaluation suite)
- RULER (retrieval and multi-hop reasoning) 
- LongBench-v2 (extended evaluation)
- BABILong (distributed facts evaluation)

All loaders provide consistent interfaces with length statistics and validation.
"""

from .registry import DatasetRegistry
from .base import BaseDatasetLoader, DatasetSample, DatasetMetrics
from .infinitebench import *
from .ruler import RulerLoader
from .longbench import LongBenchV2Loader  
from .babilong import BABILongLoader

__all__ = [
    "DatasetRegistry",
    "BaseDatasetLoader", 
    "DatasetSample",
    "DatasetMetrics",
    # InfiniteBench loaders
    "ZhQALoader",
    "RetrievePasskeyLoader", 
    "RetrieveKVLoader",
    "RetrieveNumberLoader",
    "CodeDebugLoader",
    "CodeQALoader", 
    "EnQALoader",
    # External loaders
    "RulerLoader",
    "LongBenchV2Loader",
    "BABILongLoader"
]