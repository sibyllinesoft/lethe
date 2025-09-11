"""
Expanded Evaluation Suite - Complete Framework

This package implements the comprehensive evaluation framework with:
- Context-pruning adapters (heuristics, library compressors, code lexical)
- RAG/search stacks (BM25, Vector, Hybrid, Rerankers)  
- Long-context baselines (sliding window, StreamingLLM, full context)
- Unified parity harness for fair comparison
- Embedding freezing and pool fingerprinting
- Matrix execution with fail-closed gates
- Comprehensive result generation

Usage:
    from evaluation import ExpandedEvaluationSuite
    
    # Create and run evaluation suite
    suite = ExpandedEvaluationSuite()
    results = suite.run_complete_evaluation()
"""

from .unified_adapter_interface import (
    BaseAdapter, AdapterRegistry, AdapterType, Atom, SelectionResult,
    SelectionCertificate, generate_hash, validate_selection_result
)

from .context_pruning_adapters import (
    LastKTurnsAdapter, RecencyEntityAdapter, TFIDFSpansAdapter,
    SlidingWindowAdapter, CodeLexicalAdapter
)

from .rag_search_adapters import (
    BM25Adapter, VectorAdapter, HybridAdapter, RerankerAdapter
)

from .long_context_adapters import (
    SlidingWindowBaseline, StreamingLLMAdapter, FullContextAdapter,
    AdaptiveContextAdapter
)

from .parity_harness import (
    ParityHarness, CorpusConstructor, CorpusSpec, ContextItem, EvaluationResult
)

from .embedding_freezing import (
    EmbeddingManager, PoolManager, EmbeddingRecord, PoolRecord,
    DummyEmbeddingModel
)

from .matrix_execution import (
    MatrixExecutor, MatrixConfig, MatrixResult, DatasetManager,
    FailClosedGateValidator, GateResult, GateStatus
)

# Main integration class
from .expanded_evaluation_suite import ExpandedEvaluationSuite

__all__ = [
    # Core interfaces
    'BaseAdapter', 'AdapterRegistry', 'AdapterType', 'Atom', 'SelectionResult',
    'SelectionCertificate', 'generate_hash', 'validate_selection_result',
    
    # Context-pruning adapters
    'LastKTurnsAdapter', 'RecencyEntityAdapter', 'TFIDFSpansAdapter',
    'SlidingWindowAdapter', 'CodeLexicalAdapter',
    
    # RAG/search adapters
    'BM25Adapter', 'VectorAdapter', 'HybridAdapter', 'RerankerAdapter',
    
    # Long-context adapters
    'SlidingWindowBaseline', 'StreamingLLMAdapter', 'FullContextAdapter',
    'AdaptiveContextAdapter',
    
    # Parity harness
    'ParityHarness', 'CorpusConstructor', 'CorpusSpec', 'ContextItem', 'EvaluationResult',
    
    # Embedding management
    'EmbeddingManager', 'PoolManager', 'EmbeddingRecord', 'PoolRecord',
    'DummyEmbeddingModel',
    
    # Matrix execution
    'MatrixExecutor', 'MatrixConfig', 'MatrixResult', 'DatasetManager',
    'FailClosedGateValidator', 'GateResult', 'GateStatus',
    
    # Main suite
    'ExpandedEvaluationSuite'
]