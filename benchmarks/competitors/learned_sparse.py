#!/usr/bin/env python3
"""
Learned Sparse/Late-Interaction Competitors
===========================================

Implementations for learned sparse and late-interaction systems:
- SPLADE v2 (sparse lexical expansion)
- ColBERT v2 (token-level late interaction)
- RAGatouille (ColBERT wrapper)
"""

import logging
from .mock import MockCompetitor

logger = logging.getLogger(__name__)

class SpladeV2Competitor(MockCompetitor):
    """SPLADE v2 learned sparse retrieval competitor."""
    pass

class ColBERTV2Competitor(MockCompetitor):
    """ColBERT v2 late interaction competitor."""
    pass

class RAGatouilleCompetitor(MockCompetitor):
    """RAGatouille ColBERT wrapper competitor."""
    pass