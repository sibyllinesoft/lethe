#!/usr/bin/env python3
"""
Long-Context Algorithm Competitors
==================================

Implementations for long-context processing algorithms:
- StreamingLLM (attention sinks + sliding window)
- LongNet (dilated attention)
- BGE-M3 (baseline embedding model)
"""

import logging
from .mock import MockCompetitor

logger = logging.getLogger(__name__)

class StreamingLLMCompetitor(MockCompetitor):
    """StreamingLLM attention sink competitor."""
    pass

class LongNetCompetitor(MockCompetitor):
    """LongNet dilated attention competitor."""
    pass

class BGEM3BaselineCompetitor(MockCompetitor):
    """BGE-M3 baseline embedding competitor."""
    pass