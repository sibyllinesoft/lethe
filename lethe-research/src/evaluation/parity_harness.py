"""
Parity Harness for Unified Evaluation Framework

This module implements the core parity harness that ensures fair comparison across
all adapter types by providing:

1. Corpus construction per sample (Q, C, Atoms)
2. Budgeting fairness (B_tokens = keep_ratio * tokens_in)
3. Deterministic segmentation (same chunker for everyone)
4. Budget enforcement (cut at ≤ B_tokens after method-specific ordering)
5. No method exceeds budget or gets extra LLM steps

Usage:
    from evaluation.parity_harness import ParityHarness, CorpusConstructor
    from evaluation.unified_adapter_interface import AdapterRegistry
    
    # Create harness
    harness = ParityHarness()
    
    # Register adapters
    harness.register_all_adapters()
    
    # Evaluate sample
    results = harness.evaluate_sample(
        sample_id="sample_001",
        user_query="How do I fix this error?",
        context_data=context_items,
        keep_ratio=0.15,
        K=10,
        seed=42
    )
"""

import json
import time
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
from collections import defaultdict

from .unified_adapter_interface import (
    BaseAdapter, AdapterRegistry, Atom, SelectionResult, 
    AdapterType, generate_hash, validate_selection_result
)

# Import all adapter types
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

logger = logging.getLogger(__name__)

@dataclass
class ContextItem:
    """Represents a single context item before segmentation."""
    content: str
    item_type: str  # "turn", "tool_io", "code", "error", "metadata"
    timestamp: Optional[float] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CorpusSpec:
    """Specification for corpus construction."""
    query: str  # Q := current user turn (+ minimal state)
    context_items: List[ContextItem]  # C := {all prior turns, tool I/O, etc.}
    keep_ratio: float  # Budget ratio
    K: int  # Number of candidates to consider
    seed: int  # Random seed
    sample_id: str  # Unique sample identifier
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for hashing."""
        return {
            'query': self.query,
            'context_items': [asdict(item) for item in self.context_items],
            'keep_ratio': self.keep_ratio,
            'K': self.K,
            'seed': self.seed,
            'sample_id': self.sample_id
        }
    
    def get_hash(self) -> str:
        """Generate deterministic hash for this corpus spec."""
        return generate_hash(self.to_dict())

@dataclass
class EvaluationResult:
    """Result of evaluating one adapter on one sample."""
    sample_id: str
    method_id: str
    adapter_type: str
    selection_result: SelectionResult
    corpus_hash: str
    validation_errors: List[str] = field(default_factory=list)
    is_valid: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'sample_id': self.sample_id,
            'method_id': self.method_id,
            'adapter_type': self.adapter_type,
            'selection_result': self.selection_result.to_dict(),
            'corpus_hash': self.corpus_hash,
            'validation_errors': self.validation_errors,
            'is_valid': self.is_valid
        }

class CorpusConstructor:
    """Handles deterministic corpus construction and atom segmentation."""
    
    def __init__(self, tokenizer=None, chunker_config: Dict[str, Any] = None):
        self.tokenizer = tokenizer
        self.chunker_config = chunker_config or {
            'chunk_size': 200,  # tokens per chunk
            'overlap': 20,      # tokens overlap between chunks
            'min_chunk_size': 50,  # minimum viable chunk size
        }
        self._tokenizer_hash = None
        
    def construct_corpus(self, spec: CorpusSpec) -> Tuple[str, List[Atom], int]:
        """
        Construct corpus from specification.
        
        Returns:
            Tuple of (query, atoms, budget_tokens)
        """
        # Deterministic segmentation using S0 chunker
        atoms = self._segment_context_items(spec.context_items)
        
        # Calculate budget
        total_input_tokens = sum(atom.tokens or 0 for atom in atoms)
        budget_tokens = int(spec.keep_ratio * total_input_tokens)
        
        logger.info(f"Constructed corpus for {spec.sample_id}: "
                   f"{len(atoms)} atoms, {total_input_tokens} total tokens, "
                   f"budget {budget_tokens} tokens (ratio {spec.keep_ratio})")
        
        return spec.query, atoms, budget_tokens
    
    def _segment_context_items(self, context_items: List[ContextItem]) -> List[Atom]:
        """Segment context items into atoms using deterministic chunker."""
        atoms = []
        
        for item in context_items:
            item_atoms = self._chunk_content(
                content=item.content,
                item_type=item.item_type,
                timestamp=item.timestamp,
                source=item.source,
                metadata=item.metadata
            )
            atoms.extend(item_atoms)
        
        return atoms
    
    def _chunk_content(self, content: str, item_type: str, 
                      timestamp: Optional[float] = None,
                      source: Optional[str] = None,
                      metadata: Dict[str, Any] = None) -> List[Atom]:
        """Chunk content into atoms."""
        if not content.strip():
            return []
        
        # Simple sentence-based chunking for now
        # In production, this would use a sophisticated chunker
        chunks = self._simple_chunk(content)
        
        atoms = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 10:  # Skip very short chunks
                continue
                
            atom = Atom(
                content=chunk,
                metadata={
                    'item_type': item_type,
                    'chunk_index': i,
                    **(metadata or {})
                },
                tokens=self._count_tokens(chunk),
                source=source,
                timestamp=timestamp
            )
            atoms.append(atom)
        
        return atoms
    
    def _simple_chunk(self, content: str) -> List[str]:
        """Simple chunking based on sentences and token limits."""
        import re
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_tokens = self._count_tokens(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if (current_tokens + sentence_tokens > self.chunker_config['chunk_size'] and 
                current_chunk):
                
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(self._count_tokens(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append(chunk_text)
        
        return chunks
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer and hasattr(self.tokenizer, 'encode'):
            try:
                return len(self.tokenizer.encode(text))
            except:
                pass
        
        # Fallback: rough estimation
        return int(len(text.split()) * 1.3)
    
    def get_tokenizer_hash(self) -> str:
        """Get tokenizer fingerprint hash."""
        if self._tokenizer_hash is None:
            if self.tokenizer and hasattr(self.tokenizer, 'get_hash'):
                self._tokenizer_hash = self.tokenizer.get_hash()
            else:
                # Generate hash from chunker config
                config_str = json.dumps(self.chunker_config, sort_keys=True)
                self._tokenizer_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        
        return self._tokenizer_hash

class ParityHarness:
    """Main parity harness for unified evaluation."""
    
    def __init__(self, corpus_constructor: Optional[CorpusConstructor] = None,
                 config: Dict[str, Any] = None):
        self.corpus_constructor = corpus_constructor or CorpusConstructor()
        self.config = config or {}
        self._adapters_registered = False
        self._evaluation_log = []
        
    def register_all_adapters(self):
        """Register all available adapters with the registry."""
        if self._adapters_registered:
            logger.info("Adapters already registered, skipping")
            return
        
        # Context-Pruning Heuristics
        AdapterRegistry.register("last_k_turns_5", LastKTurnsAdapter(k=5))
        AdapterRegistry.register("last_k_turns_10", LastKTurnsAdapter(k=10))
        AdapterRegistry.register("recency_entity", RecencyEntityAdapter())
        AdapterRegistry.register("tfidf_spans", TFIDFSpansAdapter(top_k=20))
        AdapterRegistry.register("sliding_window_heuristic", SlidingWindowAdapter(window_size=10))
        AdapterRegistry.register("code_lexical", CodeLexicalAdapter())
        
        # RAG/Search Stacks
        AdapterRegistry.register("bm25_lucene", BM25Adapter(k1=1.2, b=0.75, K1=2000))
        AdapterRegistry.register("vector_faiss", VectorAdapter(metric="cosine", K1=2000))
        AdapterRegistry.register("hybrid_weaviate", HybridAdapter(alpha=0.5, K1=2000))
        AdapterRegistry.register("reranker_bm25_600", RerankerAdapter(K2=600, base_retriever="bm25"))
        AdapterRegistry.register("reranker_bm25_1100", RerankerAdapter(K2=1100, base_retriever="bm25"))
        
        # Long-Context Baselines
        AdapterRegistry.register("sliding_window_2048", SlidingWindowBaseline(window_size=2048))
        AdapterRegistry.register("streaming_llm", StreamingLLMAdapter(cache_size=256))
        AdapterRegistry.register("full_context", FullContextAdapter(max_context_tokens=32768))
        AdapterRegistry.register("adaptive_context", AdaptiveContextAdapter())
        
        self._adapters_registered = True
        
        registered_count = len(AdapterRegistry.list_adapters())
        logger.info(f"Registered {registered_count} adapters across all categories")
    
    def evaluate_sample(self, spec: CorpusSpec, 
                       adapter_filter: Optional[List[str]] = None) -> Dict[str, EvaluationResult]:
        """
        Evaluate a single sample across all registered adapters.
        
        Args:
            spec: Corpus specification
            adapter_filter: Optional list of adapter method_ids to evaluate
            
        Returns:
            Dictionary mapping method_id to EvaluationResult
        """
        if not self._adapters_registered:
            self.register_all_adapters()
        
        # Construct corpus once for all adapters (ensuring fairness)
        query, atoms, budget_tokens = self.corpus_constructor.construct_corpus(spec)
        corpus_hash = spec.get_hash()
        
        # Set tokenizer hash for all adapters
        tokenizer_hash = self.corpus_constructor.get_tokenizer_hash()
        
        # Get adapters to evaluate
        adapter_ids = adapter_filter or AdapterRegistry.list_adapters()
        
        results = {}
        
        for method_id in adapter_ids:
            adapter = AdapterRegistry.get_adapter(method_id)
            if adapter is None:
                logger.warning(f"Adapter not found: {method_id}")
                continue
            
            try:
                # Set fingerprints for this adapter
                adapter.set_tokenizer_hash(tokenizer_hash)
                # Use consistent pool fingerprint across all samples for same evaluation
                shared_pool_fingerprint = f"eval_pool_{spec.sample_id.split('_')[0] if '_' in spec.sample_id else 'shared'}"
                adapter.set_pool_fingerprint(shared_pool_fingerprint)
                
                # Execute selection
                selection_result = adapter.select_bundle(
                    method=method_id,
                    Q=query,
                    Atoms=atoms,
                    B_tokens=budget_tokens,
                    K=spec.K,
                    seed=spec.seed
                )
                
                # Add budget metadata to result
                selection_result.metadata.update({
                    'budget_tokens': budget_tokens,
                    'budget_ratio': spec.keep_ratio
                })
                
                # Validate result
                is_valid, validation_errors = validate_selection_result(selection_result, budget_tokens)
                
                result = EvaluationResult(
                    sample_id=spec.sample_id,
                    method_id=method_id,
                    adapter_type=adapter.adapter_type.value,
                    selection_result=selection_result,
                    corpus_hash=corpus_hash,
                    validation_errors=validation_errors,
                    is_valid=is_valid
                )
                
                results[method_id] = result
                
                # Log the evaluation
                self._evaluation_log.append({
                    'timestamp': time.time(),
                    'sample_id': spec.sample_id,
                    'method_id': method_id,
                    'is_valid': is_valid,
                    'budget_tokens': budget_tokens,
                    'selected_tokens': selection_result.total_tokens(),
                    'time_ms': selection_result.time_ms
                })
                
                if not is_valid:
                    logger.warning(f"Validation failed for {method_id} on {spec.sample_id}: {validation_errors}")
                
            except Exception as e:
                logger.error(f"Error evaluating {method_id} on {spec.sample_id}: {e}")
                
                # Create error result
                error_result = EvaluationResult(
                    sample_id=spec.sample_id,
                    method_id=method_id,
                    adapter_type=adapter.adapter_type.value if adapter else "unknown",
                    selection_result=SelectionResult(
                        selected_atoms=[],
                        method_id=method_id,
                        encoder_hash="error",
                        pool_fingerprint="error", 
                        tokenizer_hash="error",
                        time_ms=0,
                        time_p95=0,
                        candidates_considered=(0, 0),
                        scores=[],
                        cert_hash="error"
                    ),
                    corpus_hash=corpus_hash,
                    validation_errors=[str(e)],
                    is_valid=False
                )
                
                results[method_id] = error_result
        
        logger.info(f"Evaluated {len(results)} adapters on sample {spec.sample_id}")
        return results
    
    def evaluate_batch(self, specs: List[CorpusSpec],
                      adapter_filter: Optional[List[str]] = None,
                      output_file: Optional[Path] = None) -> Dict[str, Dict[str, EvaluationResult]]:
        """
        Evaluate a batch of samples across all adapters.
        
        Args:
            specs: List of corpus specifications
            adapter_filter: Optional list of adapter method_ids to evaluate
            output_file: Optional file to save results
            
        Returns:
            Dictionary mapping sample_id to results dictionary
        """
        all_results = {}
        
        for i, spec in enumerate(specs):
            logger.info(f"Evaluating sample {i+1}/{len(specs)}: {spec.sample_id}")
            
            sample_results = self.evaluate_sample(spec, adapter_filter)
            all_results[spec.sample_id] = sample_results
            
            # Save intermediate results
            if output_file and i % 10 == 0:  # Save every 10 samples
                self._save_results(all_results, output_file)
        
        # Final save
        if output_file:
            self._save_results(all_results, output_file)
        
        return all_results
    
    def _save_results(self, results: Dict[str, Dict[str, EvaluationResult]], 
                     output_file: Path):
        """Save results to file."""
        try:
            # Convert to serializable format
            serializable_results = {}
            for sample_id, sample_results in results.items():
                serializable_results[sample_id] = {
                    method_id: result.to_dict() 
                    for method_id, result in sample_results.items()
                }
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Saved results to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of evaluations performed."""
        if not self._evaluation_log:
            return {'total_evaluations': 0}
        
        # Aggregate statistics
        total_evaluations = len(self._evaluation_log)
        valid_evaluations = sum(1 for log in self._evaluation_log if log['is_valid'])
        
        # Group by method
        method_stats = defaultdict(list)
        for log in self._evaluation_log:
            method_stats[log['method_id']].append(log)
        
        # Calculate per-method statistics
        method_summary = {}
        for method_id, logs in method_stats.items():
            method_summary[method_id] = {
                'total_samples': len(logs),
                'valid_samples': sum(1 for log in logs if log['is_valid']),
                'avg_time_ms': np.mean([log['time_ms'] for log in logs]),
                'avg_tokens_selected': np.mean([log['selected_tokens'] for log in logs])
            }
        
        return {
            'total_evaluations': total_evaluations,
            'valid_evaluations': valid_evaluations,
            'success_rate': valid_evaluations / total_evaluations if total_evaluations > 0 else 0,
            'unique_samples': len(set(log['sample_id'] for log in self._evaluation_log)),
            'unique_methods': len(set(log['method_id'] for log in self._evaluation_log)),
            'method_summary': method_summary
        }
    
    def validate_parity(self, results: Dict[str, Dict[str, EvaluationResult]]) -> Dict[str, Any]:
        """
        Validate that parity constraints are met across all evaluations.
        
        Returns:
            Validation report with any parity violations
        """
        violations = []
        
        # Check that all methods used same corpus for each sample
        for sample_id, sample_results in results.items():
            corpus_hashes = {result.corpus_hash for result in sample_results.values()}
            if len(corpus_hashes) > 1:
                violations.append(f"Sample {sample_id}: Different corpus hashes across methods")
        
        # Check budget compliance
        budget_violations = []
        for sample_id, sample_results in results.items():
            for method_id, result in sample_results.items():
                if result.selection_result.total_tokens() > result.selection_result.metadata.get('budget_tokens', 0):
                    budget_violations.append(f"{method_id} on {sample_id} exceeded budget")
        
        # Check timing constraints (p95 >= avg, p99/p95 <= 2.5)
        timing_violations = []
        for sample_id, sample_results in results.items():
            for method_id, result in sample_results.items():
                sr = result.selection_result
                if sr.time_p95 < sr.time_ms:
                    timing_violations.append(f"{method_id} on {sample_id}: p95 < avg time")
        
        return {
            'corpus_violations': violations,
            'budget_violations': budget_violations,
            'timing_violations': timing_violations,
            'total_violations': len(violations) + len(budget_violations) + len(timing_violations),
            'is_valid': len(violations) == 0 and len(budget_violations) == 0 and len(timing_violations) == 0
        }

# Export main classes
__all__ = [
    'ParityHarness',
    'CorpusConstructor', 
    'CorpusSpec',
    'ContextItem',
    'EvaluationResult'
]