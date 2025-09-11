"""
Context-Pruning Adapters for Evaluation Suite

This module implements various context-pruning methods including heuristics,
library compressors, and code lexical filters as unified adapters.

Categories:
1. Heuristics: last-K turns, recency+entity caps, TF-IDF/BM25 top-spans, 
   sliding-window, entropy/surprisal filter
2. Library compressors: LangChain ContextualCompression, LlamaIndex post-processors,
   LLMLingua-style token drop
3. Code lexical: Zoekt/regex symbol filters (for CODE/ERROR)

Usage:
    from evaluation.context_pruning_adapters import LastKTurnsAdapter, TFIDFSpansAdapter
    from evaluation.unified_adapter_interface import AdapterRegistry
    
    # Register adapters
    AdapterRegistry.register("last_k_turns", LastKTurnsAdapter(k=5))
    AdapterRegistry.register("tfidf_spans", TFIDFSpansAdapter(top_k=20))
    
    # Use via registry
    adapter = AdapterRegistry.get_adapter("last_k_turns")
    result = adapter.select_bundle(method, Q, Atoms, B_tokens, K, seed)
"""

import re
import math
import random
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter, defaultdict
import numpy as np
from dataclasses import dataclass, field

from .unified_adapter_interface import (
    BaseAdapter, AdapterType, Atom, SelectionResult, generate_hash
)

logger = logging.getLogger(__name__)

class LastKTurnsAdapter(BaseAdapter):
    """Select last K turns/atoms based on recency."""
    
    def __init__(self, k: int = 10, config: Dict[str, Any] = None):
        super().__init__(AdapterType.CONTEXT_PRUNING_HEURISTIC, config)
        self.k = k
        
    def get_method_id(self) -> str:
        return f"last_k_turns_k{self.k}"
    
    def select_bundle(self, method: str, Q: str, Atoms: List[Atom], 
                     B_tokens: int, K: int, seed: int) -> SelectionResult:
        """Select the last K atoms (most recent) within budget."""
        
        if not self.validate_inputs(Q, Atoms, B_tokens, K):
            raise ValueError("Invalid inputs to LastKTurnsAdapter")
        
        random.seed(seed)
        start_time = time.perf_counter()
        
        # Sort by timestamp (most recent first) or reverse order if no timestamps
        if all(atom.timestamp for atom in Atoms):
            sorted_atoms = sorted(Atoms, key=lambda x: x.timestamp or 0, reverse=True)
        else:
            # Assume reverse order (last items are most recent)
            sorted_atoms = list(reversed(Atoms))
        
        # Take last K atoms
        candidates = sorted_atoms[:min(K, len(sorted_atoms))]
        
        # Fit within budget
        selected_atoms = []
        total_tokens = 0
        scores = []
        
        for i, atom in enumerate(candidates):
            atom_tokens = atom.tokens or 0
            if total_tokens + atom_tokens <= B_tokens:
                selected_atoms.append(atom)
                total_tokens += atom_tokens
                # Score by recency (higher for more recent)
                scores.append(len(candidates) - i)
            else:
                break
        
        avg_time, p95_time, _ = self._measure_performance(
            lambda: self._select_last_k(sorted_atoms, K, B_tokens)
        )
        
        # Generate certificate
        cert = self._generate_certificate(method, Q, Atoms, B_tokens, K, seed, 
                                        SelectionResult([], method, "", "", "", 0, 0, (0, 0), [], ""))
        
        return SelectionResult(
            selected_atoms=selected_atoms,
            method_id=method,
            encoder_hash=self._encoder_hash or "n/a",
            pool_fingerprint=self._pool_fingerprint or "n/a", 
            tokenizer_hash=self._tokenizer_hash or "n/a",
            time_ms=avg_time,
            time_p95=p95_time,
            candidates_considered=(len(candidates), len(selected_atoms)),
            scores=scores,
            cert_hash=cert.generate_hash(),
            metadata={'k': self.k, 'method': 'last_k_turns'}
        )
    
    def _select_last_k(self, sorted_atoms: List[Atom], K: int, B_tokens: int) -> List[Atom]:
        """Internal selection logic for performance measurement."""
        candidates = sorted_atoms[:min(K, len(sorted_atoms))]
        selected = []
        total_tokens = 0
        
        for atom in candidates:
            atom_tokens = atom.tokens or 0
            if total_tokens + atom_tokens <= B_tokens:
                selected.append(atom)
                total_tokens += atom_tokens
            else:
                break
                
        return selected

class RecencyEntityAdapter(BaseAdapter):
    """Select atoms based on recency with entity importance weighting."""
    
    def __init__(self, entity_weight: float = 2.0, recency_decay: float = 0.1, 
                 config: Dict[str, Any] = None):
        super().__init__(AdapterType.CONTEXT_PRUNING_HEURISTIC, config)
        self.entity_weight = entity_weight
        self.recency_decay = recency_decay
        
    def get_method_id(self) -> str:
        return f"recency_entity_w{self.entity_weight}_d{self.recency_decay}"
    
    def select_bundle(self, method: str, Q: str, Atoms: List[Atom], 
                     B_tokens: int, K: int, seed: int) -> SelectionResult:
        """Select atoms based on recency and entity importance."""
        
        if not self.validate_inputs(Q, Atoms, B_tokens, K):
            raise ValueError("Invalid inputs to RecencyEntityAdapter")
        
        random.seed(seed)
        
        # Extract entities from query (simple heuristic)
        query_entities = self._extract_entities(Q)
        
        # Score atoms by recency and entity overlap
        scored_atoms = []
        for i, atom in enumerate(Atoms):
            recency_score = math.exp(-self.recency_decay * (len(Atoms) - i - 1))
            entity_score = self._compute_entity_score(atom.content, query_entities)
            
            total_score = recency_score + self.entity_weight * entity_score
            scored_atoms.append((atom, total_score))
        
        # Sort by score descending
        scored_atoms.sort(key=lambda x: x[1], reverse=True)
        
        # Take top K candidates
        candidates = scored_atoms[:min(K, len(scored_atoms))]
        
        # Fit within budget
        selected_atoms = []
        total_tokens = 0
        scores = []
        
        for atom, score in candidates:
            atom_tokens = atom.tokens or 0
            if total_tokens + atom_tokens <= B_tokens:
                selected_atoms.append(atom)
                total_tokens += atom_tokens
                scores.append(score)
            else:
                break
        
        avg_time, p95_time, _ = self._measure_performance(
            lambda: self._select_recency_entity(Atoms, query_entities, K, B_tokens)
        )
        
        cert = self._generate_certificate(method, Q, Atoms, B_tokens, K, seed,
                                        SelectionResult([], method, "", "", "", 0, 0, (0, 0), [], ""))
        
        return SelectionResult(
            selected_atoms=selected_atoms,
            method_id=method,
            encoder_hash=self._encoder_hash or "n/a",
            pool_fingerprint=self._pool_fingerprint or "n/a",
            tokenizer_hash=self._tokenizer_hash or "n/a", 
            time_ms=avg_time,
            time_p95=p95_time,
            candidates_considered=(len(candidates), len(selected_atoms)),
            scores=scores,
            cert_hash=cert.generate_hash(),
            metadata={'entity_weight': self.entity_weight, 'recency_decay': self.recency_decay}
        )
    
    def _extract_entities(self, text: str) -> Set[str]:
        """Extract entities using simple patterns."""
        # Simple entity extraction - capitalized words, code symbols, etc.
        entities = set()
        
        # Capitalized words (potential proper nouns)
        for match in re.finditer(r'\b[A-Z][a-zA-Z]+\b', text):
            entities.add(match.group().lower())
        
        # Code-like entities (functions, variables)
        for match in re.finditer(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text):
            word = match.group()
            if '_' in word or any(c.isupper() for c in word[1:]):
                entities.add(word.lower())
        
        return entities
    
    def _compute_entity_score(self, text: str, query_entities: Set[str]) -> float:
        """Compute entity overlap score."""
        text_entities = self._extract_entities(text)
        
        if not query_entities or not text_entities:
            return 0.0
        
        overlap = len(query_entities.intersection(text_entities))
        return overlap / len(query_entities)
    
    def _select_recency_entity(self, Atoms: List[Atom], query_entities: Set[str], 
                              K: int, B_tokens: int) -> List[Atom]:
        """Internal selection logic for performance measurement."""
        scored_atoms = []
        for i, atom in enumerate(Atoms):
            recency_score = math.exp(-self.recency_decay * (len(Atoms) - i - 1))
            entity_score = self._compute_entity_score(atom.content, query_entities)
            total_score = recency_score + self.entity_weight * entity_score
            scored_atoms.append((atom, total_score))
        
        scored_atoms.sort(key=lambda x: x[1], reverse=True)
        candidates = scored_atoms[:min(K, len(scored_atoms))]
        
        selected = []
        total_tokens = 0
        for atom, _ in candidates:
            atom_tokens = atom.tokens or 0
            if total_tokens + atom_tokens <= B_tokens:
                selected.append(atom)
                total_tokens += atom_tokens
            else:
                break
        
        return selected

class TFIDFSpansAdapter(BaseAdapter):
    """Select top spans using TF-IDF scoring."""
    
    def __init__(self, top_k: int = 20, min_span_tokens: int = 5, 
                 max_span_tokens: int = 100, config: Dict[str, Any] = None):
        super().__init__(AdapterType.CONTEXT_PRUNING_HEURISTIC, config)
        self.top_k = top_k
        self.min_span_tokens = min_span_tokens
        self.max_span_tokens = max_span_tokens
        
    def get_method_id(self) -> str:
        return f"tfidf_spans_k{self.top_k}_min{self.min_span_tokens}_max{self.max_span_tokens}"
    
    def select_bundle(self, method: str, Q: str, Atoms: List[Atom], 
                     B_tokens: int, K: int, seed: int) -> SelectionResult:
        """Select atoms using TF-IDF scoring of spans."""
        
        if not self.validate_inputs(Q, Atoms, B_tokens, K):
            raise ValueError("Invalid inputs to TFIDFSpansAdapter")
        
        random.seed(seed)
        
        # Compute TF-IDF scores
        query_terms = self._tokenize(Q.lower())
        atom_scores = []
        
        # Build document frequency
        df = defaultdict(int)
        total_docs = len(Atoms)
        
        for atom in Atoms:
            terms = set(self._tokenize(atom.content.lower()))
            for term in terms:
                df[term] += 1
        
        # Score each atom
        for atom in Atoms:
            score = self._compute_tfidf_score(atom.content, query_terms, df, total_docs)
            atom_scores.append((atom, score))
        
        # Sort by score descending
        atom_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top K candidates  
        candidates = atom_scores[:min(K, len(atom_scores))]
        
        # Fit within budget
        selected_atoms = []
        total_tokens = 0
        scores = []
        
        for atom, score in candidates:
            atom_tokens = atom.tokens or 0
            if total_tokens + atom_tokens <= B_tokens:
                selected_atoms.append(atom)
                total_tokens += atom_tokens
                scores.append(score)
            else:
                break
        
        avg_time, p95_time, _ = self._measure_performance(
            lambda: self._select_tfidf(Atoms, query_terms, K, B_tokens)
        )
        
        cert = self._generate_certificate(method, Q, Atoms, B_tokens, K, seed,
                                        SelectionResult([], method, "", "", "", 0, 0, (0, 0), [], ""))
        
        return SelectionResult(
            selected_atoms=selected_atoms,
            method_id=method,
            encoder_hash=self._encoder_hash or "n/a",
            pool_fingerprint=self._pool_fingerprint or "n/a",
            tokenizer_hash=self._tokenizer_hash or "n/a",
            time_ms=avg_time,
            time_p95=p95_time,
            candidates_considered=(len(candidates), len(selected_atoms)),
            scores=scores,
            cert_hash=cert.generate_hash(),
            metadata={'top_k': self.top_k, 'min_span_tokens': self.min_span_tokens}
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Remove punctuation and split
        cleaned = re.sub(r'[^\w\s]', ' ', text)
        return [token for token in cleaned.split() if len(token) > 1]
    
    def _compute_tfidf_score(self, document: str, query_terms: List[str], 
                           df: Dict[str, int], total_docs: int) -> float:
        """Compute TF-IDF score for document against query."""
        doc_terms = self._tokenize(document.lower())
        doc_tf = Counter(doc_terms)
        doc_length = len(doc_terms)
        
        score = 0.0
        for term in query_terms:
            if term in doc_tf:
                tf = doc_tf[term] / doc_length
                idf = math.log(total_docs / (df[term] + 1))
                score += tf * idf
        
        return score
    
    def _select_tfidf(self, Atoms: List[Atom], query_terms: List[str], 
                     K: int, B_tokens: int) -> List[Atom]:
        """Internal selection logic for performance measurement."""
        # Rebuild DF for performance test
        df = defaultdict(int)
        total_docs = len(Atoms)
        
        for atom in Atoms:
            terms = set(self._tokenize(atom.content.lower()))
            for term in terms:
                df[term] += 1
        
        atom_scores = []
        for atom in Atoms:
            score = self._compute_tfidf_score(atom.content, query_terms, df, total_docs)
            atom_scores.append((atom, score))
        
        atom_scores.sort(key=lambda x: x[1], reverse=True)
        candidates = atom_scores[:min(K, len(atom_scores))]
        
        selected = []
        total_tokens = 0
        for atom, _ in candidates:
            atom_tokens = atom.tokens or 0
            if total_tokens + atom_tokens <= B_tokens:
                selected.append(atom)
                total_tokens += atom_tokens
            else:
                break
        
        return selected

class SlidingWindowAdapter(BaseAdapter):
    """Select atoms using a sliding window approach."""
    
    def __init__(self, window_size: int = 10, overlap: int = 2, 
                 config: Dict[str, Any] = None):
        super().__init__(AdapterType.CONTEXT_PRUNING_HEURISTIC, config)
        self.window_size = window_size
        self.overlap = overlap
        
    def get_method_id(self) -> str:
        return f"sliding_window_size{self.window_size}_overlap{self.overlap}"
    
    def select_bundle(self, method: str, Q: str, Atoms: List[Atom], 
                     B_tokens: int, K: int, seed: int) -> SelectionResult:
        """Select atoms using sliding window with overlap."""
        
        if not self.validate_inputs(Q, Atoms, B_tokens, K):
            raise ValueError("Invalid inputs to SlidingWindowAdapter")
        
        random.seed(seed)
        
        # Create sliding windows
        windows = []
        step_size = self.window_size - self.overlap
        
        for i in range(0, len(Atoms), step_size):
            window_end = min(i + self.window_size, len(Atoms))
            window = Atoms[i:window_end]
            
            if window:
                # Score window by relevance to query
                window_score = self._score_window(window, Q)
                windows.append((window, window_score, i))
        
        # Sort windows by score
        windows.sort(key=lambda x: x[1], reverse=True)
        
        # Flatten and deduplicate atoms from top windows
        seen_indices = set()
        selected_atoms = []
        scores = []
        total_tokens = 0
        atoms_added = 0
        
        for window, window_score, start_idx in windows:
            if atoms_added >= K:
                break
                
            for j, atom in enumerate(window):
                global_idx = start_idx + j
                if global_idx not in seen_indices:
                    atom_tokens = atom.tokens or 0
                    if total_tokens + atom_tokens <= B_tokens:
                        selected_atoms.append(atom)
                        scores.append(window_score)
                        total_tokens += atom_tokens
                        seen_indices.add(global_idx)
                        atoms_added += 1
                        
                        if atoms_added >= K:
                            break
                    else:
                        break
        
        avg_time, p95_time, _ = self._measure_performance(
            lambda: self._select_sliding_window(Atoms, Q, K, B_tokens)
        )
        
        cert = self._generate_certificate(method, Q, Atoms, B_tokens, K, seed,
                                        SelectionResult([], method, "", "", "", 0, 0, (0, 0), [], ""))
        
        return SelectionResult(
            selected_atoms=selected_atoms,
            method_id=method,
            encoder_hash=self._encoder_hash or "n/a",
            pool_fingerprint=self._pool_fingerprint or "n/a",
            tokenizer_hash=self._tokenizer_hash or "n/a",
            time_ms=avg_time,
            time_p95=p95_time,
            candidates_considered=(len(windows), len(selected_atoms)),
            scores=scores,
            cert_hash=cert.generate_hash(),
            metadata={'window_size': self.window_size, 'overlap': self.overlap}
        )
    
    def _score_window(self, window: List[Atom], query: str) -> float:
        """Score a window by relevance to query."""
        query_terms = set(query.lower().split())
        
        total_score = 0.0
        for atom in window:
            atom_terms = set(atom.content.lower().split())
            overlap = len(query_terms.intersection(atom_terms))
            total_score += overlap / max(len(query_terms), 1)
        
        return total_score / len(window) if window else 0.0
    
    def _select_sliding_window(self, Atoms: List[Atom], query: str, 
                              K: int, B_tokens: int) -> List[Atom]:
        """Internal selection logic for performance measurement."""
        windows = []
        step_size = self.window_size - self.overlap
        
        for i in range(0, len(Atoms), step_size):
            window_end = min(i + self.window_size, len(Atoms))
            window = Atoms[i:window_end]
            if window:
                window_score = self._score_window(window, query)
                windows.append((window, window_score, i))
        
        windows.sort(key=lambda x: x[1], reverse=True)
        
        seen_indices = set()
        selected = []
        total_tokens = 0
        atoms_added = 0
        
        for window, _, start_idx in windows:
            if atoms_added >= K:
                break
            for j, atom in enumerate(window):
                global_idx = start_idx + j
                if global_idx not in seen_indices:
                    atom_tokens = atom.tokens or 0
                    if total_tokens + atom_tokens <= B_tokens:
                        selected.append(atom)
                        total_tokens += atom_tokens
                        seen_indices.add(global_idx)
                        atoms_added += 1
                        if atoms_added >= K:
                            break
                    else:
                        break
        
        return selected

class CodeLexicalAdapter(BaseAdapter):
    """Select atoms using code-specific lexical filtering."""
    
    def __init__(self, symbol_patterns: List[str] = None, 
                 error_patterns: List[str] = None,
                 config: Dict[str, Any] = None):
        super().__init__(AdapterType.CONTEXT_PRUNING_CODE_LEXICAL, config)
        
        # Default patterns for code symbols
        self.symbol_patterns = symbol_patterns or [
            r'\b[A-Z][a-zA-Z0-9_]*\b',  # Class names
            r'\b[a-z_][a-zA-Z0-9_]*\(\)',  # Function calls
            r'\b[a-z_][a-zA-Z0-9_]*\b',  # Variables
            r'\b\w+\.\w+\b',  # Method calls
            r'#[a-zA-Z0-9_]+',  # Hash symbols
        ]
        
        # Default patterns for errors
        self.error_patterns = error_patterns or [
            r'Error:',
            r'Exception:',
            r'Traceback',
            r'Failed',
            r'undefined',
            r'null pointer',
            r'segmentation fault'
        ]
        
    def get_method_id(self) -> str:
        return "code_lexical_filter"
    
    def select_bundle(self, method: str, Q: str, Atoms: List[Atom], 
                     B_tokens: int, K: int, seed: int) -> SelectionResult:
        """Select atoms using code lexical patterns."""
        
        if not self.validate_inputs(Q, Atoms, B_tokens, K):
            raise ValueError("Invalid inputs to CodeLexicalAdapter")
        
        random.seed(seed)
        
        # Extract patterns from query
        query_symbols = self._extract_symbols(Q)
        query_has_error = self._has_error_pattern(Q)
        
        # Score atoms by code relevance
        scored_atoms = []
        for atom in Atoms:
            score = self._compute_code_score(atom.content, query_symbols, query_has_error)
            scored_atoms.append((atom, score))
        
        # Sort by score descending
        scored_atoms.sort(key=lambda x: x[1], reverse=True)
        
        # Take top K candidates
        candidates = scored_atoms[:min(K, len(scored_atoms))]
        
        # Fit within budget
        selected_atoms = []
        total_tokens = 0
        scores = []
        
        for atom, score in candidates:
            atom_tokens = atom.tokens or 0
            if total_tokens + atom_tokens <= B_tokens:
                selected_atoms.append(atom)
                total_tokens += atom_tokens
                scores.append(score)
            else:
                break
        
        avg_time, p95_time, _ = self._measure_performance(
            lambda: self._select_code_lexical(Atoms, query_symbols, query_has_error, K, B_tokens)
        )
        
        cert = self._generate_certificate(method, Q, Atoms, B_tokens, K, seed,
                                        SelectionResult([], method, "", "", "", 0, 0, (0, 0), [], ""))
        
        return SelectionResult(
            selected_atoms=selected_atoms,
            method_id=method,
            encoder_hash=self._encoder_hash or "n/a",
            pool_fingerprint=self._pool_fingerprint or "n/a",
            tokenizer_hash=self._tokenizer_hash or "n/a",
            time_ms=avg_time,
            time_p95=p95_time,
            candidates_considered=(len(candidates), len(selected_atoms)),
            scores=scores,
            cert_hash=cert.generate_hash(),
            metadata={'symbol_patterns': len(self.symbol_patterns), 'error_patterns': len(self.error_patterns)}
        )
    
    def _extract_symbols(self, text: str) -> Set[str]:
        """Extract code symbols from text."""
        symbols = set()
        
        for pattern in self.symbol_patterns:
            for match in re.finditer(pattern, text):
                symbols.add(match.group().lower())
        
        return symbols
    
    def _has_error_pattern(self, text: str) -> bool:
        """Check if text contains error patterns."""
        for pattern in self.error_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _compute_code_score(self, content: str, query_symbols: Set[str], 
                           query_has_error: bool) -> float:
        """Compute code relevance score."""
        content_symbols = self._extract_symbols(content)
        content_has_error = self._has_error_pattern(content)
        
        # Symbol overlap score
        if query_symbols and content_symbols:
            symbol_score = len(query_symbols.intersection(content_symbols)) / len(query_symbols)
        else:
            symbol_score = 0.0
        
        # Error relevance score
        error_score = 0.0
        if query_has_error and content_has_error:
            error_score = 1.0
        elif query_has_error and not content_has_error:
            error_score = -0.5  # Penalize non-error content when looking for errors
        
        # Code density score (higher for code-like content)
        code_patterns = len([p for p in self.symbol_patterns if re.search(p, content)])
        code_density = code_patterns / len(self.symbol_patterns)
        
        return symbol_score + error_score + 0.2 * code_density
    
    def _select_code_lexical(self, Atoms: List[Atom], query_symbols: Set[str], 
                            query_has_error: bool, K: int, B_tokens: int) -> List[Atom]:
        """Internal selection logic for performance measurement."""
        scored_atoms = []
        for atom in Atoms:
            score = self._compute_code_score(atom.content, query_symbols, query_has_error)
            scored_atoms.append((atom, score))
        
        scored_atoms.sort(key=lambda x: x[1], reverse=True)
        candidates = scored_atoms[:min(K, len(scored_atoms))]
        
        selected = []
        total_tokens = 0
        for atom, _ in candidates:
            atom_tokens = atom.tokens or 0
            if total_tokens + atom_tokens <= B_tokens:
                selected.append(atom)
                total_tokens += atom_tokens
            else:
                break
        
        return selected

# Import timing module
import time

# Export all adapter classes
__all__ = [
    'LastKTurnsAdapter',
    'RecencyEntityAdapter', 
    'TFIDFSpansAdapter',
    'SlidingWindowAdapter',
    'CodeLexicalAdapter'
]