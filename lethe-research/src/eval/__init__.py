"""
Lethe IR System - Evaluation Framework
=====================================

Task 3 Implementation: Baseline Suite Evaluation with Budget Parity

This module provides bulletproof baseline evaluation with:
- Real model integrations (SPLADE, uniCOIL, ColBERTv2, Dense)
- Budget parity enforcement (±5% compute/FLOPs)
- Anti-fraud validation (non-empty guards, smoke tests)
- Statistical rigor (per-query metrics, JSONL persistence)
- MS MARCO/BEIR integration

Key Components:
- BaselineRetriever: Abstract base with parity enforcement
- RealModelBaselines: SPLADE, uniCOIL, ColBERTv2, Dense implementations
- EvaluationFramework: Metrics computation and persistence
- AntiFreudValidator: Non-empty guards and smoke testing
"""

from .baselines import *
from .evaluation import *
from .metrics import *
from .validation import *

__version__ = "1.0.0"
__author__ = "Lethe Research Team"