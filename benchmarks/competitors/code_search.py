#!/usr/bin/env python3
"""
Code Search & Graph Competitors
===============================

Implementations for code search and graph-based systems:
- Zoekt (fast trigram code search)
- livegrep (regex code search)
- GraphRAG (graph-structured retrieval)
"""

import logging
from .mock import MockCompetitor

logger = logging.getLogger(__name__)

class ZoektCompetitor(MockCompetitor):
    """Zoekt trigram code search competitor."""
    pass

class LivegrepCompetitor(MockCompetitor):
    """livegrep regex code search competitor."""
    pass

class GraphRAGCompetitor(MockCompetitor):
    """GraphRAG graph-structured competitor."""
    pass