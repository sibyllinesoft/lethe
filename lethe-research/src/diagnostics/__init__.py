"""
Diagnostic "Ladder of Proofs" Framework
=====================================

Systematic validation system for InfiniteBench evaluation pipeline components
without relying on LLM API calls. Each rung of the ladder validates different
aspects of the pipeline to identify the root cause of low accuracy.

Rungs:
0. Scoring sanity (gold-echo, normalizer probes, random baseline)
1. Retrieval/selection coverage (SpanCoverage, SymbolCoverage, IDOverlap)
2. Extractive baselines (no LLM, answer from text)
3. Oracle bounds (upper bounds and isolation analysis)
4. Curriculum & ablations (make hardness legible)
5. Keep-ratio & K2 curves (where selection fails)

Selection Stack Diagnostics:
Fast targeted probes for identifying exact failure points in the
Lethe retrieval selection pipeline (S0→S1→S2→CBU).
"""

from .ladder_runner import DiagnosticLadderRunner
from .coverage_analyzer import CoverageAnalyzer
from .extractive_baselines import ExtractionBaselines
from .oracle_bounds import OracleBoundsCalculator
from .sample_ledger import SampleLedger
from .selection_stack_diagnostics import SelectionStackDiagnostics
from .probe_query_vectors import QueryVectorProbe
from .probe_index_retrieval import IndexRetrievalProbe
from .probe_cross_encoder import CrossEncoderProbe
from .probe_coverage_features import CoverageFeaturesProbe

# Cross-encoder comprehensive debugging system
from .ce_attestation import CrossEncoderAttestationSystem
from .ce_synthetic_tests import CrossEncoderSyntheticTester
from .ce_input_debugging import CrossEncoderInputDebugger
from .ce_head_validation import CrossEncoderHeadValidator
from .ce_safe_mode import CrossEncoderSafeMode, SafeModeConfig

__all__ = [
    'DiagnosticLadderRunner',
    'CoverageAnalyzer', 
    'ExtractionBaselines',
    'OracleBoundsCalculator',
    'SampleLedger',
    'SelectionStackDiagnostics',
    'QueryVectorProbe',
    'IndexRetrievalProbe',
    'CrossEncoderProbe',
    'CoverageFeaturesProbe',
    # Cross-encoder debugging components
    'CrossEncoderAttestationSystem',
    'CrossEncoderSyntheticTester',
    'CrossEncoderInputDebugger',
    'CrossEncoderHeadValidator',
    'CrossEncoderSafeMode',
    'SafeModeConfig'
]