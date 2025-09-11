"""
Long-Context Baseline Adapters for Evaluation Suite

This module implements long-context baseline methods as unified adapters:

1. Sliding window (naïve): Simple sliding window with fixed size
2. StreamingLLM (windowed attention): Attention-based window management
3. Full context upper bound: Use all available context up to model limits

These serve as baselines to compare against context-pruning and RAG methods.

Usage:
    from evaluation.long_context_adapters import SlidingWindowBaseline, StreamingLLMAdapter
    from evaluation.unified_adapter_interface import AdapterRegistry
    
    # Register adapters
    AdapterRegistry.register("sliding_window", SlidingWindowBaseline(window_size=2048))
    AdapterRegistry.register("streaming_llm", StreamingLLMAdapter(cache_size=256))
    AdapterRegistry.register("full_context", FullContextAdapter())
    
    # Use via registry
    adapter = AdapterRegistry.get_adapter("sliding_window")
    result = adapter.select_bundle(method, Q, Atoms, B_tokens, K, seed)
"""

import time
import random
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import deque

from .unified_adapter_interface import (
    BaseAdapter, AdapterType, Atom, SelectionResult, generate_hash
)

logger = logging.getLogger(__name__)

class SlidingWindowBaseline(BaseAdapter):
    """Naïve sliding window baseline that keeps most recent atoms."""
    
    def __init__(self, window_size: int = 2048, overlap_ratio: float = 0.1,
                 config: Dict[str, Any] = None):
        super().__init__(AdapterType.LONG_CONTEXT_SLIDING, config)
        self.window_size = window_size  # Window size in tokens
        self.overlap_ratio = overlap_ratio  # Overlap between windows
        
    def get_method_id(self) -> str:
        return f"sliding_window_size{self.window_size}_overlap{self.overlap_ratio}"
    
    def select_bundle(self, method: str, Q: str, Atoms: List[Atom], 
                     B_tokens: int, K: int, seed: int) -> SelectionResult:
        """Select atoms using sliding window approach."""
        
        if not self.validate_inputs(Q, Atoms, B_tokens, K):
            raise ValueError("Invalid inputs to SlidingWindowBaseline")
        
        random.seed(seed)
        
        # Calculate effective window size (constrained by budget)
        effective_window_size = min(self.window_size, B_tokens)
        
        # Start from the most recent atoms (end of list)
        selected_atoms = []
        total_tokens = 0
        scores = []
        
        # Work backwards from most recent
        for i in range(len(Atoms) - 1, -1, -1):
            atom = Atoms[i]
            atom_tokens = atom.tokens or 0
            
            if total_tokens + atom_tokens <= effective_window_size:
                selected_atoms.insert(0, atom)  # Insert at beginning to maintain order
                total_tokens += atom_tokens
                # Score by recency (higher for more recent)
                scores.insert(0, len(Atoms) - i)
                
                if len(selected_atoms) >= K:
                    break
            else:
                # Can't fit this atom, stop if we have any atoms
                if selected_atoms:
                    break
                # If no atoms selected yet and this one is too big, skip it
                continue
        
        avg_time, p95_time, _ = self._measure_performance(
            lambda: self._select_sliding_window(Atoms, effective_window_size, K)
        )
        
        cert = self._generate_certificate(method, Q, Atoms, B_tokens, K, seed,
                                        SelectionResult([], method, "", "", "", 0, 0, (0, 0), [], ""))
        
        return SelectionResult(
            selected_atoms=selected_atoms,
            method_id=method,
            encoder_hash=self._encoder_hash or "n/a",
            pool_fingerprint=self._pool_fingerprint or "n/a",
            tokenizer_hash=self._tokenizer_hash or "sliding_window_tokenizer",
            time_ms=avg_time,
            time_p95=p95_time,
            candidates_considered=(len(Atoms), len(selected_atoms)),
            scores=scores,
            cert_hash=cert.generate_hash(),
            metadata={'window_size': self.window_size, 'overlap_ratio': self.overlap_ratio}
        )
    
    def _select_sliding_window(self, atoms: List[Atom], window_size: int, K: int) -> List[Atom]:
        """Internal selection logic for performance measurement."""
        selected = []
        total_tokens = 0
        
        for i in range(len(atoms) - 1, -1, -1):
            atom = atoms[i]
            atom_tokens = atom.tokens or 0
            
            if total_tokens + atom_tokens <= window_size:
                selected.insert(0, atom)
                total_tokens += atom_tokens
                
                if len(selected) >= K:
                    break
            else:
                if selected:
                    break
                continue
        
        return selected

class StreamingLLMAdapter(BaseAdapter):
    """StreamingLLM-style adapter with attention-based window management."""
    
    def __init__(self, cache_size: int = 256, sink_tokens: int = 4,
                 window_size: int = 2048, config: Dict[str, Any] = None):
        super().__init__(AdapterType.LONG_CONTEXT_STREAMING, config)
        self.cache_size = cache_size  # Size of attention cache
        self.sink_tokens = sink_tokens  # Number of initial tokens to always keep
        self.window_size = window_size  # Total window size
        self._attention_cache = deque(maxlen=cache_size)
        
    def get_method_id(self) -> str:
        return f"streaming_llm_cache{self.cache_size}_sink{self.sink_tokens}_window{self.window_size}"
    
    def select_bundle(self, method: str, Q: str, Atoms: List[Atom], 
                     B_tokens: int, K: int, seed: int) -> SelectionResult:
        """Select atoms using StreamingLLM-style attention management."""
        
        if not self.validate_inputs(Q, Atoms, B_tokens, K):
            raise ValueError("Invalid inputs to StreamingLLMAdapter")
        
        random.seed(seed)
        
        # Calculate effective window size
        effective_window_size = min(self.window_size, B_tokens)
        
        # Select sink tokens (initial important atoms)
        sink_atoms = Atoms[:min(self.sink_tokens, len(Atoms))]
        remaining_atoms = Atoms[self.sink_tokens:]
        
        # Calculate tokens used by sink atoms
        sink_token_count = sum(atom.tokens or 0 for atom in sink_atoms)
        remaining_budget = effective_window_size - sink_token_count
        
        if remaining_budget <= 0:
            # Only sink atoms fit
            selected_atoms = sink_atoms
            scores = [1.0] * len(sink_atoms)  # All sink atoms have max score
        else:
            # Add atoms from the end (most recent) using streaming strategy
            streaming_atoms = []
            streaming_tokens = 0
            
            # Work backwards from most recent, simulating streaming attention
            for i in range(len(remaining_atoms) - 1, -1, -1):
                atom = remaining_atoms[i]
                atom_tokens = atom.tokens or 0
                
                if streaming_tokens + atom_tokens <= remaining_budget:
                    streaming_atoms.insert(0, atom)  # Maintain order
                    streaming_tokens += atom_tokens
                    
                    if len(streaming_atoms) >= (K - len(sink_atoms)):
                        break
                else:
                    # Apply attention-based eviction strategy
                    if self._should_evict_for_new_atom(atom, streaming_atoms):
                        # Find least important atom to evict
                        evict_idx = self._find_eviction_candidate(streaming_atoms)
                        if evict_idx is not None:
                            evicted = streaming_atoms.pop(evict_idx)
                            streaming_tokens -= (evicted.tokens or 0)
                            
                            # Try to add new atom
                            if streaming_tokens + atom_tokens <= remaining_budget:
                                streaming_atoms.insert(0, atom)
                                streaming_tokens += atom_tokens
            
            # Combine sink and streaming atoms
            selected_atoms = sink_atoms + streaming_atoms
            
            # Generate scores (sink atoms get highest scores)
            scores = ([1.0] * len(sink_atoms) + 
                     [0.5 + 0.4 * (i / max(len(streaming_atoms), 1)) 
                      for i in range(len(streaming_atoms))])
        
        avg_time, p95_time, _ = self._measure_performance(
            lambda: self._select_streaming(Atoms, effective_window_size, K)
        )
        
        cert = self._generate_certificate(method, Q, Atoms, B_tokens, K, seed,
                                        SelectionResult([], method, "", "", "", 0, 0, (0, 0), [], ""))
        
        return SelectionResult(
            selected_atoms=selected_atoms,
            method_id=method,
            encoder_hash=self._encoder_hash or "n/a",
            pool_fingerprint=self._pool_fingerprint or "n/a",
            tokenizer_hash=self._tokenizer_hash or "streaming_tokenizer",
            time_ms=avg_time,
            time_p95=p95_time,
            candidates_considered=(len(Atoms), len(selected_atoms)),
            scores=scores,
            cert_hash=cert.generate_hash(),
            metadata={'cache_size': self.cache_size, 'sink_tokens': self.sink_tokens}
        )
    
    def _should_evict_for_new_atom(self, new_atom: Atom, current_atoms: List[Atom]) -> bool:
        """Decide whether to evict an existing atom for a new one."""
        # Simple heuristic: evict if new atom seems more important
        # In a real implementation, this would use attention scores
        
        if not current_atoms:
            return True
        
        # Score based on content length and recency
        new_score = self._compute_importance_score(new_atom, position=0)  # Most recent
        
        min_score = float('inf')
        for i, atom in enumerate(current_atoms):
            score = self._compute_importance_score(atom, position=i)
            min_score = min(min_score, score)
        
        return new_score > min_score
    
    def _find_eviction_candidate(self, atoms: List[Atom]) -> Optional[int]:
        """Find the best candidate atom to evict."""
        if not atoms:
            return None
        
        min_score = float('inf')
        min_idx = 0
        
        for i, atom in enumerate(atoms):
            score = self._compute_importance_score(atom, position=i)
            if score < min_score:
                min_score = score
                min_idx = i
        
        return min_idx
    
    def _compute_importance_score(self, atom: Atom, position: int) -> float:
        """Compute importance score for an atom."""
        # Combine content-based and position-based scores
        content_score = len(atom.content) / 1000.0  # Longer content is more important
        recency_score = 1.0 / (position + 1)  # More recent is more important
        
        return content_score + recency_score
    
    def _select_streaming(self, atoms: List[Atom], window_size: int, K: int) -> List[Atom]:
        """Internal selection logic for performance measurement."""
        sink_atoms = atoms[:min(self.sink_tokens, len(atoms))]
        remaining_atoms = atoms[self.sink_tokens:]
        
        sink_token_count = sum(atom.tokens or 0 for atom in sink_atoms)
        remaining_budget = window_size - sink_token_count
        
        if remaining_budget <= 0:
            return sink_atoms
        
        streaming_atoms = []
        streaming_tokens = 0
        
        for i in range(len(remaining_atoms) - 1, -1, -1):
            atom = remaining_atoms[i]
            atom_tokens = atom.tokens or 0
            
            if streaming_tokens + atom_tokens <= remaining_budget:
                streaming_atoms.insert(0, atom)
                streaming_tokens += atom_tokens
                
                if len(streaming_atoms) >= (K - len(sink_atoms)):
                    break
        
        return sink_atoms + streaming_atoms

class FullContextAdapter(BaseAdapter):
    """Full context upper bound adapter that uses all available context."""
    
    def __init__(self, max_context_tokens: int = 32768, config: Dict[str, Any] = None):
        super().__init__(AdapterType.LONG_CONTEXT_FULL, config)
        self.max_context_tokens = max_context_tokens
        
    def get_method_id(self) -> str:
        return f"full_context_max{self.max_context_tokens}"
    
    def select_bundle(self, method: str, Q: str, Atoms: List[Atom], 
                     B_tokens: int, K: int, seed: int) -> SelectionResult:
        """Select all atoms within budget and context limits."""
        
        if not self.validate_inputs(Q, Atoms, B_tokens, K):
            raise ValueError("Invalid inputs to FullContextAdapter")
        
        random.seed(seed)
        
        # Use the smaller of budget and max context
        effective_limit = min(B_tokens, self.max_context_tokens)
        
        # Take atoms in order until we hit limits
        selected_atoms = []
        total_tokens = 0
        scores = []
        
        for i, atom in enumerate(Atoms):
            atom_tokens = atom.tokens or 0
            
            # Check both token budget and K limit
            if (total_tokens + atom_tokens <= effective_limit and 
                len(selected_atoms) < K):
                
                selected_atoms.append(atom)
                total_tokens += atom_tokens
                # Score by order (maintaining original sequence importance)
                scores.append(1.0 - (i / max(len(Atoms), 1)))
            else:
                break
        
        # If we couldn't fit even one atom due to size, try to fit largest possible subset
        if not selected_atoms and Atoms:
            # Find atoms that fit individually
            for atom in Atoms:
                if (atom.tokens or 0) <= effective_limit:
                    selected_atoms.append(atom)
                    scores.append(1.0)
                    break
        
        avg_time, p95_time, _ = self._measure_performance(
            lambda: self._select_full_context(Atoms, effective_limit, K)
        )
        
        cert = self._generate_certificate(method, Q, Atoms, B_tokens, K, seed,
                                        SelectionResult([], method, "", "", "", 0, 0, (0, 0), [], ""))
        
        return SelectionResult(
            selected_atoms=selected_atoms,
            method_id=method,
            encoder_hash=self._encoder_hash or "n/a",
            pool_fingerprint=self._pool_fingerprint or "n/a",
            tokenizer_hash=self._tokenizer_hash or "full_context_tokenizer",
            time_ms=avg_time,
            time_p95=p95_time,
            candidates_considered=(len(Atoms), len(selected_atoms)),
            scores=scores,
            cert_hash=cert.generate_hash(),
            metadata={'max_context_tokens': self.max_context_tokens}
        )
    
    def _select_full_context(self, atoms: List[Atom], token_limit: int, K: int) -> List[Atom]:
        """Internal selection logic for performance measurement."""
        selected = []
        total_tokens = 0
        
        for atom in atoms:
            atom_tokens = atom.tokens or 0
            
            if (total_tokens + atom_tokens <= token_limit and 
                len(selected) < K):
                selected.append(atom)
                total_tokens += atom_tokens
            else:
                break
        
        return selected

class AdaptiveContextAdapter(BaseAdapter):
    """Adaptive context management that adjusts strategy based on content."""
    
    def __init__(self, strategies: List[str] = None, 
                 adaptation_threshold: float = 0.7,
                 config: Dict[str, Any] = None):
        super().__init__(AdapterType.LONG_CONTEXT_STREAMING, config)  # Use streaming type
        
        self.strategies = strategies or ["sliding_window", "streaming", "full_context"]
        self.adaptation_threshold = adaptation_threshold
        
        # Initialize sub-adapters
        self.sliding_adapter = SlidingWindowBaseline()
        self.streaming_adapter = StreamingLLMAdapter()  
        self.full_adapter = FullContextAdapter()
        
    def get_method_id(self) -> str:
        return f"adaptive_context_threshold{self.adaptation_threshold}"
    
    def select_bundle(self, method: str, Q: str, Atoms: List[Atom], 
                     B_tokens: int, K: int, seed: int) -> SelectionResult:
        """Adaptively select context management strategy based on content."""
        
        if not self.validate_inputs(Q, Atoms, B_tokens, K):
            raise ValueError("Invalid inputs to AdaptiveContextAdapter")
        
        random.seed(seed)
        
        # Analyze content to choose strategy
        chosen_strategy = self._choose_strategy(Q, Atoms, B_tokens)
        
        # Delegate to chosen adapter
        if chosen_strategy == "sliding_window":
            result = self.sliding_adapter.select_bundle(method, Q, Atoms, B_tokens, K, seed)
        elif chosen_strategy == "streaming":
            result = self.streaming_adapter.select_bundle(method, Q, Atoms, B_tokens, K, seed)
        elif chosen_strategy == "full_context":
            result = self.full_adapter.select_bundle(method, Q, Atoms, B_tokens, K, seed)
        else:
            # Default to sliding window
            result = self.sliding_adapter.select_bundle(method, Q, Atoms, B_tokens, K, seed)
        
        # Update metadata to include chosen strategy
        result.metadata.update({
            'chosen_strategy': chosen_strategy,
            'adaptation_threshold': self.adaptation_threshold
        })
        
        return result
    
    def _choose_strategy(self, query: str, atoms: List[Atom], budget: int) -> str:
        """Choose the best strategy based on content analysis."""
        total_tokens = sum(atom.tokens or 0 for atom in atoms)
        
        # If everything fits in budget, use full context
        if total_tokens <= budget:
            return "full_context"
        
        # Analyze query complexity
        query_complexity = self._analyze_query_complexity(query)
        
        # Analyze content diversity
        content_diversity = self._analyze_content_diversity(atoms)
        
        # Choose strategy based on analysis
        if query_complexity > self.adaptation_threshold and content_diversity > self.adaptation_threshold:
            # Complex query with diverse content - use streaming for better attention
            return "streaming"
        elif content_diversity < 0.3:
            # Homogeneous content - sliding window is sufficient
            return "sliding_window"
        else:
            # Default to streaming for better long-range dependencies
            return "streaming"
    
    def _analyze_query_complexity(self, query: str) -> float:
        """Analyze query complexity (0 to 1 scale)."""
        # Simple heuristics for query complexity
        complexity_indicators = [
            len(query.split()) > 10,  # Long query
            '?' in query,  # Question
            any(word in query.lower() for word in ['compare', 'analyze', 'explain', 'why', 'how']),
            len([c for c in query if c.isupper()]) > 3,  # Technical terms
        ]
        
        return sum(complexity_indicators) / len(complexity_indicators)
    
    def _analyze_content_diversity(self, atoms: List[Atom]) -> float:
        """Analyze content diversity (0 to 1 scale)."""
        if len(atoms) < 2:
            return 0.0
        
        # Simple diversity measure based on vocabulary overlap
        all_words = set()
        atom_words = []
        
        for atom in atoms:
            words = set(atom.content.lower().split())
            atom_words.append(words)
            all_words.update(words)
        
        if not all_words:
            return 0.0
        
        # Calculate average pairwise Jaccard distance
        distances = []
        for i in range(len(atom_words)):
            for j in range(i + 1, len(atom_words)):
                words_i, words_j = atom_words[i], atom_words[j]
                intersection = len(words_i.intersection(words_j))
                union = len(words_i.union(words_j))
                
                if union > 0:
                    jaccard_sim = intersection / union
                    distances.append(1 - jaccard_sim)  # Convert to distance
        
        return np.mean(distances) if distances else 0.0

# Export all adapter classes
__all__ = [
    'SlidingWindowBaseline',
    'StreamingLLMAdapter',
    'FullContextAdapter', 
    'AdaptiveContextAdapter'
]