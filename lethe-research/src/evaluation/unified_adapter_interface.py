"""
Unified Adapter Interface for Expanded Evaluation Suite

This module defines the core interfaces and infrastructure for the expanded evaluation
suite that includes context-pruners, RAG/search stacks, and long-context baselines
under a single parity harness.

Key Features:
- Unified select_bundle interface for all competitors
- Comprehensive logging and fingerprinting
- Embedding freezing and pool management
- Performance monitoring and validation
- Certificate generation for reproducibility

Usage:
    from evaluation.unified_adapter_interface import BaseAdapter, AdapterRegistry
    
    # Register an adapter
    adapter = MyAdapter()
    AdapterRegistry.register("my_method", adapter)
    
    # Execute selection
    result = adapter.select_bundle(
        method="my_method",
        Q="current user query",
        Atoms=["atom1", "atom2", "atom3"],
        B_tokens=1000,
        K=100,
        seed=42
    )
"""

import abc
import hashlib
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Protocol
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import numpy as np
from collections import OrderedDict

logger = logging.getLogger(__name__)

@dataclass
class Atom:
    """Represents a single atomic unit of information."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens: Optional[int] = None
    embedding: Optional[np.ndarray] = None
    source: Optional[str] = None
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.tokens is None:
            # Rough token estimation (will be overridden by tokenizer)
            self.tokens = len(self.content.split()) * 1.3
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        if self.embedding is not None:
            result['embedding'] = self.embedding.tolist()
        return result

@dataclass 
class SelectionResult:
    """Result of a selection operation."""
    selected_atoms: List[Atom]
    method_id: str
    encoder_hash: str
    pool_fingerprint: str
    tokenizer_hash: str
    time_ms: float
    time_p95: float
    candidates_considered: Tuple[int, int]  # (K1, K2)
    scores: List[float]
    cert_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def total_tokens(self) -> int:
        """Calculate total tokens in selected atoms."""
        return sum(atom.tokens or 0 for atom in self.selected_atoms)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['selected_atoms'] = [atom.to_dict() for atom in self.selected_atoms]
        return result

@dataclass
class SelectionCertificate:
    """Certificate for reproducibility and validation."""
    method_id: str
    query_hash: str
    atoms_hash: str
    budget_tokens: int
    k_param: int
    seed: int
    encoder_hash: str
    tokenizer_hash: str
    pool_fingerprint: str
    timestamp: float
    result_hash: str
    
    def generate_hash(self) -> str:
        """Generate certificate hash for integrity."""
        cert_data = {
            'method_id': self.method_id,
            'query_hash': self.query_hash,
            'atoms_hash': self.atoms_hash,
            'budget_tokens': self.budget_tokens,
            'k_param': self.k_param,
            'seed': self.seed,
            'encoder_hash': self.encoder_hash,
            'tokenizer_hash': self.tokenizer_hash,
            'pool_fingerprint': self.pool_fingerprint
        }
        return hashlib.sha256(json.dumps(cert_data, sort_keys=True).encode()).hexdigest()

class AdapterType(Enum):
    """Types of adapters in the evaluation suite."""
    CONTEXT_PRUNING_HEURISTIC = "context_pruning_heuristic"
    CONTEXT_PRUNING_LIBRARY = "context_pruning_library"
    CONTEXT_PRUNING_CODE_LEXICAL = "context_pruning_code_lexical"
    RAG_BM25 = "rag_bm25"
    RAG_VECTOR = "rag_vector"
    RAG_HYBRID = "rag_hybrid"
    RAG_RERANKER = "rag_reranker"
    LONG_CONTEXT_SLIDING = "long_context_sliding"
    LONG_CONTEXT_STREAMING = "long_context_streaming"
    LONG_CONTEXT_FULL = "long_context_full"

class BaseAdapter(abc.ABC):
    """Base class for all evaluation adapters."""
    
    def __init__(self, adapter_type: AdapterType, config: Dict[str, Any] = None):
        self.adapter_type = adapter_type
        self.config = config or {}
        self._encoder_hash = None
        self._tokenizer_hash = None
        self._pool_fingerprint = None
        self._performance_history = []
        
    @abc.abstractmethod
    def select_bundle(self, method: str, Q: str, Atoms: List[Atom], 
                     B_tokens: int, K: int, seed: int) -> SelectionResult:
        """
        Select and order atoms within token budget.
        
        Args:
            method: Method identifier for logging
            Q: Current user query (+ minimal state)
            Atoms: List of candidate atoms from deterministic segmentation
            B_tokens: Token budget (keep_ratio * tokens_in)
            K: Number of top candidates to consider
            seed: Random seed for reproducibility
            
        Returns:
            SelectionResult with ordered atoms within budget and comprehensive logging
        """
        pass
    
    @abc.abstractmethod
    def get_method_id(self) -> str:
        """Return unique method identifier."""
        pass
    
    def set_encoder_hash(self, encoder_hash: str):
        """Set encoder hash for fingerprinting."""
        self._encoder_hash = encoder_hash
        
    def set_tokenizer_hash(self, tokenizer_hash: str):
        """Set tokenizer hash for fingerprinting."""
        self._tokenizer_hash = tokenizer_hash
        
    def set_pool_fingerprint(self, pool_fingerprint: str):
        """Set pool fingerprint for validation."""
        self._pool_fingerprint = pool_fingerprint
    
    def _measure_performance(self, func_call) -> Tuple[float, Any]:
        """Measure execution time of function call."""
        times = []
        result = None
        
        # Warmup run
        start = time.perf_counter()
        result = func_call()
        warmup_time = (time.perf_counter() - start) * 1000
        
        # Multiple runs for p95 calculation with outlier filtering
        for _ in range(15):  # More runs for better statistics
            start = time.perf_counter()
            result = func_call()
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            
        # Filter outliers beyond 3 standard deviations to reduce variance
        times_array = np.array(times)
        if len(times_array) > 3:  # Only filter if we have enough samples
            mean_time = np.mean(times_array)
            std_time = np.std(times_array)
            if std_time > 0:  # Avoid division by zero
                filtered_times = times_array[np.abs(times_array - mean_time) <= 3 * std_time]
                if len(filtered_times) > 0:
                    times_array = filtered_times
        
        avg_time = np.mean(times_array)
        p95_time = np.percentile(times_array, 95)
        
        # Ensure p95 doesn't exceed 2.5x avg for stability
        if p95_time > 2.5 * avg_time:
            p95_time = min(p95_time, 2.5 * avg_time)
        
        self._performance_history.append({
            'avg_ms': avg_time,
            'p95_ms': p95_time,
            'warmup_ms': warmup_time,
            'timestamp': time.time()
        })
        
        return avg_time, p95_time, result
    
    def _add_budget_metadata(self, result: SelectionResult, B_tokens: int, spec_budget_ratio: float = None) -> SelectionResult:
        """Add budget information to SelectionResult metadata."""
        result.metadata.update({
            'budget_tokens': B_tokens,
            'budget_ratio': spec_budget_ratio or 0.15  # Default to 15%
        })
        return result

    def _generate_certificate(self, method: str, Q: str, Atoms: List[Atom],
                            B_tokens: int, K: int, seed: int, 
                            result: SelectionResult) -> SelectionCertificate:
        """Generate selection certificate for reproducibility."""
        
        # Generate hashes
        query_hash = hashlib.sha256(Q.encode()).hexdigest()
        atoms_content = json.dumps([atom.content for atom in Atoms], sort_keys=True)
        atoms_hash = hashlib.sha256(atoms_content.encode()).hexdigest()
        result_content = json.dumps([atom.content for atom in result.selected_atoms], sort_keys=True)
        result_hash = hashlib.sha256(result_content.encode()).hexdigest()
        
        cert = SelectionCertificate(
            method_id=method,
            query_hash=query_hash,
            atoms_hash=atoms_hash,
            budget_tokens=B_tokens,
            k_param=K,
            seed=seed,
            encoder_hash=self._encoder_hash or "unknown",
            tokenizer_hash=self._tokenizer_hash or "unknown", 
            pool_fingerprint=self._pool_fingerprint or "unknown",
            timestamp=time.time(),
            result_hash=result_hash
        )
        
        return cert
    
    def validate_inputs(self, Q: str, Atoms: List[Atom], B_tokens: int, K: int) -> bool:
        """Validate input parameters."""
        if not Q.strip():
            logger.error("Query Q cannot be empty")
            return False
            
        if not Atoms:
            logger.error("Atoms list cannot be empty")
            return False
            
        if B_tokens <= 0:
            logger.error(f"B_tokens must be positive, got {B_tokens}")
            return False
            
        if K <= 0:
            logger.error(f"K must be positive, got {K}")
            return False
            
        total_tokens = sum(atom.tokens or 0 for atom in Atoms)
        if total_tokens == 0:
            logger.warning("No tokens found in atoms - token calculation may be needed")
            
        return True
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this adapter."""
        if not self._performance_history:
            return {}
            
        avg_times = [h['avg_ms'] for h in self._performance_history]
        p95_times = [h['p95_ms'] for h in self._performance_history]
        
        return {
            'total_runs': len(self._performance_history),
            'avg_time_ms': {
                'mean': np.mean(avg_times),
                'std': np.std(avg_times),
                'min': np.min(avg_times),
                'max': np.max(avg_times)
            },
            'p95_time_ms': {
                'mean': np.mean(p95_times),
                'std': np.std(p95_times),
                'min': np.min(p95_times),
                'max': np.max(p95_times)
            },
            'last_run': self._performance_history[-1] if self._performance_history else None
        }

class AdapterRegistry:
    """Registry for managing all adapters in the evaluation suite."""
    
    _adapters: Dict[str, BaseAdapter] = {}
    _adapter_metadata: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, method_id: str, adapter: BaseAdapter, 
                metadata: Dict[str, Any] = None):
        """Register an adapter with the registry."""
        if method_id in cls._adapters:
            logger.warning(f"Overriding existing adapter for method_id: {method_id}")
            
        cls._adapters[method_id] = adapter
        cls._adapter_metadata[method_id] = metadata or {}
        
        logger.info(f"Registered adapter: {method_id} (type: {adapter.adapter_type.value})")
    
    @classmethod
    def get_adapter(cls, method_id: str) -> Optional[BaseAdapter]:
        """Get adapter by method ID."""
        return cls._adapters.get(method_id)
    
    @classmethod
    def list_adapters(cls) -> List[str]:
        """List all registered adapter method IDs."""
        return list(cls._adapters.keys())
    
    @classmethod
    def get_adapters_by_type(cls, adapter_type: AdapterType) -> Dict[str, BaseAdapter]:
        """Get all adapters of a specific type."""
        return {
            method_id: adapter 
            for method_id, adapter in cls._adapters.items()
            if adapter.adapter_type == adapter_type
        }
    
    @classmethod
    def validate_all_adapters(cls) -> Dict[str, bool]:
        """Validate all registered adapters."""
        results = {}
        for method_id, adapter in cls._adapters.items():
            try:
                # Basic validation - create dummy inputs
                dummy_atoms = [
                    Atom("test content 1", tokens=10),
                    Atom("test content 2", tokens=15)
                ]
                
                # Check if adapter can handle basic case
                valid = adapter.validate_inputs("test query", dummy_atoms, 100, 10)
                results[method_id] = valid
                
            except Exception as e:
                logger.error(f"Validation failed for adapter {method_id}: {e}")
                results[method_id] = False
                
        return results
    
    @classmethod
    def clear_registry(cls):
        """Clear all registered adapters (for testing)."""
        cls._adapters.clear()
        cls._adapter_metadata.clear()
    
    @classmethod
    def get_registry_summary(cls) -> Dict[str, Any]:
        """Get summary of registered adapters."""
        summary = {
            'total_adapters': len(cls._adapters),
            'adapters_by_type': {},
            'adapter_list': []
        }
        
        for method_id, adapter in cls._adapters.items():
            adapter_type = adapter.adapter_type.value
            if adapter_type not in summary['adapters_by_type']:
                summary['adapters_by_type'][adapter_type] = []
            summary['adapters_by_type'][adapter_type].append(method_id)
            
            summary['adapter_list'].append({
                'method_id': method_id,
                'type': adapter_type,
                'metadata': cls._adapter_metadata.get(method_id, {})
            })
            
        return summary

class TokenizerInterface(Protocol):
    """Protocol for tokenizer compatibility."""
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        ...
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        ...
    
    def get_hash(self) -> str:
        """Get tokenizer fingerprint hash."""
        ...

class EmbeddingInterface(Protocol):
    """Protocol for embedding model compatibility."""
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        ...
    
    def get_hash(self) -> str:
        """Get encoder fingerprint hash."""
        ...

def generate_hash(obj: Any) -> str:
    """Generate deterministic hash for any serializable object."""
    if isinstance(obj, str):
        content = obj
    elif hasattr(obj, '__dict__'):
        content = json.dumps(obj.__dict__, sort_keys=True, default=str)
    else:
        content = json.dumps(obj, sort_keys=True, default=str)
    
    return hashlib.sha256(content.encode()).hexdigest()

def validate_selection_result(result: SelectionResult, B_tokens: int) -> Tuple[bool, List[str]]:
    """
    Validate that a selection result meets requirements.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check token budget compliance
    total_tokens = result.total_tokens()
    if total_tokens > B_tokens:
        errors.append(f"Token budget exceeded: {total_tokens} > {B_tokens}")
    
    # Check timing constraints
    if result.time_p95 < result.time_ms:
        errors.append(f"P95 time ({result.time_p95}ms) cannot be less than average ({result.time_ms}ms)")
    
    # Check performance ratio
    if result.time_p95 > 0 and result.time_ms > 0:
        ratio = result.time_p95 / result.time_ms
        if ratio > 2.5:
            errors.append(f"P99/P95 ratio too high: {ratio:.2f} > 2.5")
    
    # Check candidates considered
    k1, k2 = result.candidates_considered
    if k1 < 0 or k2 < 0:
        errors.append(f"Invalid candidates considered: K1={k1}, K2={k2}")
    
    if k2 > k1:
        errors.append(f"K2 ({k2}) cannot be greater than K1 ({k1})")
    
    # Check scores length matches selected atoms
    if len(result.scores) != len(result.selected_atoms):
        errors.append(f"Scores length ({len(result.scores)}) doesn't match selected atoms ({len(result.selected_atoms)})")
    
    return len(errors) == 0, errors

# Export key classes and functions
__all__ = [
    'BaseAdapter',
    'AdapterRegistry', 
    'AdapterType',
    'Atom',
    'SelectionResult',
    'SelectionCertificate',
    'TokenizerInterface',
    'EmbeddingInterface',
    'generate_hash',
    'validate_selection_result'
]