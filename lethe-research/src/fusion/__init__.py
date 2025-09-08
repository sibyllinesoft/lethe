"""
Hybrid fusion system for Lethe IR research.

This module implements the α-sweep fusion mechanism that combines sparse (BM25) 
and dense (vector) retrieval with strict mathematical invariants and budget constraints.

Core fusion formula: Score(d) = w_s·BM25 + w_d·cos, where w_s,w_d≥0, w_s+w_d=1; α = w_s
"""

from .core import HybridFusionSystem, FusionConfiguration, FusionResult
from .invariants import InvariantValidator, InvariantViolation
from .telemetry import FusionTelemetry, TelemetryLogger

__all__ = [
    'HybridFusionSystem',
    'FusionConfiguration', 
    'FusionResult',
    'InvariantValidator',
    'InvariantViolation',
    'FusionTelemetry',
    'TelemetryLogger'
]