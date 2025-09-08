"""
Cross-encoder reranking system for Lethe hybrid IR.

Implements reranking ablation with:
- Parameters: β∈{0,0.2,0.5}, k_rerank∈{50,100,200}
- Cross-encoder integration with public checkpoints
- Budget-aware latency constraints
- Full telemetry logging
"""

from .core import RerankingSystem, RerankingConfiguration, RerankingResult
from .cross_encoder import CrossEncoderReranker
from .telemetry import RerankingTelemetry

__all__ = [
    'RerankingSystem',
    'RerankingConfiguration', 
    'RerankingResult',
    'CrossEncoderReranker',
    'RerankingTelemetry'
]