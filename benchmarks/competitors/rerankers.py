#!/usr/bin/env python3
"""
Open Reranker Competitors
=========================

Implementations for open-source reranking systems:
- BGE-reranker-large (multilingual cross-encoder)
- BGE-M3 (multilingual multi-mode reranker)
- MonoT5 (T5-based reranking)
"""

import logging
from .mock import MockCompetitor

logger = logging.getLogger(__name__)

class BGERerankerCompetitor(MockCompetitor):
    """BGE reranker large competitor."""
    pass

class BGEM3RerankerCompetitor(MockCompetitor):
    """BGE-M3 multilingual reranker competitor."""
    pass

class MonoT5Competitor(MockCompetitor):
    """MonoT5 reranker competitor."""
    pass