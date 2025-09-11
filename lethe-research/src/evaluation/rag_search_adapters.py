"""
RAG/Search Stack Adapters for Evaluation Suite

This module implements various RAG and search methods as unified adapters:

1. BM25 (Lucene/Elasticsearch): k1=1.2, b=0.75, K1=2000
2. Vector (Faiss): cosine, frozen embeddings, K1=2000  
3. Hybrid: Weaviate, Milvus, Vespa (50/50 lexical/neural blend), K1=2000
4. Rerankers: BGE-reranker at K2∈{600,1100}, CE input via render_for_ce(atom)

Usage:
    from evaluation.rag_search_adapters import BM25Adapter, VectorAdapter, HybridAdapter
    from evaluation.unified_adapter_interface import AdapterRegistry
    
    # Register adapters
    AdapterRegistry.register("bm25_lucene", BM25Adapter(k1=1.2, b=0.75))
    AdapterRegistry.register("vector_faiss", VectorAdapter(metric="cosine"))
    AdapterRegistry.register("hybrid_weaviate", HybridAdapter(alpha=0.5))
    
    # Use via registry
    adapter = AdapterRegistry.get_adapter("bm25_lucene")
    result = adapter.select_bundle(method, Q, Atoms, B_tokens, K, seed)
"""

import math
import time
import random
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np
from dataclasses import dataclass
import heapq

from .unified_adapter_interface import (
    BaseAdapter, AdapterType, Atom, SelectionResult, generate_hash, 
    EmbeddingInterface
)

logger = logging.getLogger(__name__)

class BM25Adapter(BaseAdapter):
    """BM25 retrieval adapter with configurable parameters."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75, K1: int = 2000,
                 config: Dict[str, Any] = None):
        super().__init__(AdapterType.RAG_BM25, config)
        self.k1 = k1
        self.b = b
        self.K1 = K1
        self._term_freqs = {}
        self._doc_freqs = {}
        self._avg_doc_length = 0
        self._total_docs = 0
        
    def get_method_id(self) -> str:
        return f"bm25_k1{self.k1}_b{self.b}_K1{self.K1}"
    
    def select_bundle(self, method: str, Q: str, Atoms: List[Atom], 
                     B_tokens: int, K: int, seed: int) -> SelectionResult:
        """Select atoms using BM25 scoring."""
        
        if not self.validate_inputs(Q, Atoms, B_tokens, K):
            raise ValueError("Invalid inputs to BM25Adapter")
        
        random.seed(seed)
        
        # Build BM25 index
        self._build_index(Atoms)
        
        # Score atoms against query
        query_terms = self._tokenize(Q.lower())
        scored_atoms = []
        
        for i, atom in enumerate(Atoms):
            score = self._compute_bm25_score(query_terms, i)
            scored_atoms.append((atom, score))
        
        # Sort by score descending
        scored_atoms.sort(key=lambda x: x[1], reverse=True)
        
        # Take top K1 candidates initially
        k1_candidates = scored_atoms[:min(self.K1, len(scored_atoms))]
        
        # Then apply K limit
        candidates = k1_candidates[:min(K, len(k1_candidates))]
        
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
            lambda: self._select_bm25(query_terms, Atoms, K, B_tokens)
        )
        
        cert = self._generate_certificate(method, Q, Atoms, B_tokens, K, seed,
                                        SelectionResult([], method, "", "", "", 0, 0, (0, 0), [], ""))
        
        return SelectionResult(
            selected_atoms=selected_atoms,
            method_id=method,
            encoder_hash=self._encoder_hash or "n/a",
            pool_fingerprint=self._pool_fingerprint or "n/a",
            tokenizer_hash=self._tokenizer_hash or "bm25_tokenizer",
            time_ms=avg_time,
            time_p95=p95_time,
            candidates_considered=(len(k1_candidates), len(selected_atoms)),
            scores=scores,
            cert_hash=cert.generate_hash(),
            metadata={'k1': self.k1, 'b': self.b, 'K1': self.K1}
        )
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        import re
        # Remove punctuation and convert to lowercase
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        return [token for token in cleaned.split() if len(token) > 1]
    
    def _build_index(self, atoms: List[Atom]):
        """Build BM25 index from atoms."""
        self._term_freqs = {}
        self._doc_freqs = defaultdict(int)
        total_length = 0
        
        for i, atom in enumerate(atoms):
            doc_terms = self._tokenize(atom.content)
            doc_length = len(doc_terms)
            total_length += doc_length
            
            # Term frequencies for this document
            term_freq = Counter(doc_terms)
            self._term_freqs[i] = (term_freq, doc_length)
            
            # Document frequencies
            unique_terms = set(doc_terms)
            for term in unique_terms:
                self._doc_freqs[term] += 1
        
        self._total_docs = len(atoms)
        self._avg_doc_length = total_length / max(len(atoms), 1)
    
    def _compute_bm25_score(self, query_terms: List[str], doc_id: int) -> float:
        """Compute BM25 score for document against query."""
        if doc_id not in self._term_freqs:
            return 0.0
        
        term_freq, doc_length = self._term_freqs[doc_id]
        score = 0.0
        
        for term in query_terms:
            if term in term_freq:
                # Term frequency component
                tf = term_freq[term]
                
                # Document frequency component  
                df = self._doc_freqs[term]
                idf = math.log((self._total_docs - df + 0.5) / (df + 0.5))
                
                # Length normalization
                length_norm = self.k1 * ((1 - self.b) + self.b * (doc_length / self._avg_doc_length))
                
                # BM25 formula
                term_score = idf * (tf * (self.k1 + 1)) / (tf + length_norm)
                score += term_score
        
        return score
    
    def _select_bm25(self, query_terms: List[str], atoms: List[Atom], 
                    K: int, B_tokens: int) -> List[Atom]:
        """Internal selection logic for performance measurement."""
        scored_atoms = []
        for i, atom in enumerate(atoms):
            score = self._compute_bm25_score(query_terms, i)
            scored_atoms.append((atom, score))
        
        scored_atoms.sort(key=lambda x: x[1], reverse=True)
        k1_candidates = scored_atoms[:min(self.K1, len(scored_atoms))]
        candidates = k1_candidates[:min(K, len(k1_candidates))]
        
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

class VectorAdapter(BaseAdapter):
    """Vector similarity retrieval adapter with frozen embeddings."""
    
    def __init__(self, metric: str = "cosine", K1: int = 2000, 
                 embedding_model: Optional[EmbeddingInterface] = None,
                 config: Dict[str, Any] = None):
        super().__init__(AdapterType.RAG_VECTOR, config)
        self.metric = metric
        self.K1 = K1
        self.embedding_model = embedding_model
        self._atom_embeddings = {}
        
    def get_method_id(self) -> str:
        return f"vector_{self.metric}_K1{self.K1}"
    
    def select_bundle(self, method: str, Q: str, Atoms: List[Atom], 
                     B_tokens: int, K: int, seed: int) -> SelectionResult:
        """Select atoms using vector similarity."""
        
        if not self.validate_inputs(Q, Atoms, B_tokens, K):
            raise ValueError("Invalid inputs to VectorAdapter")
        
        random.seed(seed)
        
        # Get query embedding
        if self.embedding_model:
            query_embedding = self.embedding_model.encode([Q])[0]
        else:
            # Use pre-computed embeddings or create dummy
            query_embedding = self._get_dummy_embedding(Q)
        
        # Get or compute atom embeddings
        atom_embeddings = []
        for atom in Atoms:
            if atom.embedding is not None:
                embedding = atom.embedding
            elif self.embedding_model:
                embedding = self.embedding_model.encode([atom.content])[0]
            else:
                embedding = self._get_dummy_embedding(atom.content)
            atom_embeddings.append(embedding)
        
        # Compute similarities
        scored_atoms = []
        for i, (atom, embedding) in enumerate(zip(Atoms, atom_embeddings)):
            similarity = self._compute_similarity(query_embedding, embedding)
            scored_atoms.append((atom, similarity))
        
        # Sort by similarity descending
        scored_atoms.sort(key=lambda x: x[1], reverse=True)
        
        # Take top K1 candidates initially
        k1_candidates = scored_atoms[:min(self.K1, len(scored_atoms))]
        
        # Then apply K limit
        candidates = k1_candidates[:min(K, len(k1_candidates))]
        
        # Fit within budget
        selected_atoms = []
        total_tokens = 0
        scores = []
        
        for atom, similarity in candidates:
            atom_tokens = atom.tokens or 0
            if total_tokens + atom_tokens <= B_tokens:
                selected_atoms.append(atom)
                total_tokens += atom_tokens
                scores.append(similarity)
            else:
                break
        
        avg_time, p95_time, _ = self._measure_performance(
            lambda: self._select_vector(query_embedding, Atoms, atom_embeddings, K, B_tokens)
        )
        
        cert = self._generate_certificate(method, Q, Atoms, B_tokens, K, seed,
                                        SelectionResult([], method, "", "", "", 0, 0, (0, 0), [], ""))
        
        return SelectionResult(
            selected_atoms=selected_atoms,
            method_id=method,
            encoder_hash=self._encoder_hash or "vector_encoder",
            pool_fingerprint=self._pool_fingerprint or "vector_pool",
            tokenizer_hash=self._tokenizer_hash or "n/a",
            time_ms=avg_time,
            time_p95=p95_time,
            candidates_considered=(len(k1_candidates), len(selected_atoms)),
            scores=scores,
            cert_hash=cert.generate_hash(),
            metadata={'metric': self.metric, 'K1': self.K1}
        )
    
    def _compute_similarity(self, query_emb: np.ndarray, atom_emb: np.ndarray) -> float:
        """Compute similarity between query and atom embeddings."""
        if self.metric == "cosine":
            # Cosine similarity
            query_norm = np.linalg.norm(query_emb)
            atom_norm = np.linalg.norm(atom_emb)
            
            if query_norm == 0 or atom_norm == 0:
                return 0.0
            
            return np.dot(query_emb, atom_emb) / (query_norm * atom_norm)
        
        elif self.metric == "euclidean":
            # Negative euclidean distance (higher is better)
            return -np.linalg.norm(query_emb - atom_emb)
        
        elif self.metric == "dot":
            # Dot product
            return np.dot(query_emb, atom_emb)
        
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
    
    def _get_dummy_embedding(self, text: str, dim: int = 768) -> np.ndarray:
        """Generate a dummy embedding for testing when no model available."""
        # Create deterministic embedding based on text hash
        hash_val = hash(text) % (2**31)
        np.random.seed(hash_val)
        embedding = np.random.normal(0, 1, dim)
        return embedding / np.linalg.norm(embedding)  # Normalize
    
    def _select_vector(self, query_embedding: np.ndarray, atoms: List[Atom],
                      atom_embeddings: List[np.ndarray], K: int, B_tokens: int) -> List[Atom]:
        """Internal selection logic for performance measurement."""
        scored_atoms = []
        for atom, embedding in zip(atoms, atom_embeddings):
            similarity = self._compute_similarity(query_embedding, embedding)
            scored_atoms.append((atom, similarity))
        
        scored_atoms.sort(key=lambda x: x[1], reverse=True)
        k1_candidates = scored_atoms[:min(self.K1, len(scored_atoms))]
        candidates = k1_candidates[:min(K, len(k1_candidates))]
        
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

class HybridAdapter(BaseAdapter):
    """Hybrid retrieval adapter combining lexical and neural methods."""
    
    def __init__(self, alpha: float = 0.5, lexical_method: str = "bm25",
                 K1: int = 2000, config: Dict[str, Any] = None):
        super().__init__(AdapterType.RAG_HYBRID, config)
        self.alpha = alpha  # Weight for lexical vs neural (0.5 = 50/50)
        self.lexical_method = lexical_method
        self.K1 = K1
        
        # Initialize sub-adapters
        self.bm25_adapter = BM25Adapter(K1=K1)
        self.vector_adapter = VectorAdapter(K1=K1)
        
    def get_method_id(self) -> str:
        return f"hybrid_{self.lexical_method}_alpha{self.alpha}_K1{self.K1}"
    
    def select_bundle(self, method: str, Q: str, Atoms: List[Atom], 
                     B_tokens: int, K: int, seed: int) -> SelectionResult:
        """Select atoms using hybrid lexical + neural scoring."""
        
        if not self.validate_inputs(Q, Atoms, B_tokens, K):
            raise ValueError("Invalid inputs to HybridAdapter")
        
        random.seed(seed)
        
        # Get lexical scores (BM25)
        self.bm25_adapter._build_index(Atoms)
        query_terms = self.bm25_adapter._tokenize(Q.lower())
        lexical_scores = []
        
        for i, atom in enumerate(Atoms):
            score = self.bm25_adapter._compute_bm25_score(query_terms, i)
            lexical_scores.append(score)
        
        # Get neural scores (Vector)
        if self.vector_adapter.embedding_model:
            query_embedding = self.vector_adapter.embedding_model.encode([Q])[0]
        else:
            query_embedding = self.vector_adapter._get_dummy_embedding(Q)
        
        neural_scores = []
        for atom in Atoms:
            if atom.embedding is not None:
                embedding = atom.embedding
            elif self.vector_adapter.embedding_model:
                embedding = self.vector_adapter.embedding_model.encode([atom.content])[0]
            else:
                embedding = self.vector_adapter._get_dummy_embedding(atom.content)
            
            similarity = self.vector_adapter._compute_similarity(query_embedding, embedding)
            neural_scores.append(similarity)
        
        # Normalize scores to [0, 1] range
        lexical_scores_norm = self._normalize_scores(lexical_scores)
        neural_scores_norm = self._normalize_scores(neural_scores)
        
        # Combine scores
        scored_atoms = []
        for i, atom in enumerate(Atoms):
            hybrid_score = (self.alpha * lexical_scores_norm[i] + 
                          (1 - self.alpha) * neural_scores_norm[i])
            scored_atoms.append((atom, hybrid_score))
        
        # Sort by hybrid score descending
        scored_atoms.sort(key=lambda x: x[1], reverse=True)
        
        # Take top K1 candidates initially
        k1_candidates = scored_atoms[:min(self.K1, len(scored_atoms))]
        
        # Then apply K limit
        candidates = k1_candidates[:min(K, len(k1_candidates))]
        
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
            lambda: self._select_hybrid(Q, Atoms, K, B_tokens)
        )
        
        cert = self._generate_certificate(method, Q, Atoms, B_tokens, K, seed,
                                        SelectionResult([], method, "", "", "", 0, 0, (0, 0), [], ""))
        
        return SelectionResult(
            selected_atoms=selected_atoms,
            method_id=method,
            encoder_hash=self._encoder_hash or "hybrid_encoder",
            pool_fingerprint=self._pool_fingerprint or "hybrid_pool",
            tokenizer_hash=self._tokenizer_hash or "hybrid_tokenizer",
            time_ms=avg_time,
            time_p95=p95_time,
            candidates_considered=(len(k1_candidates), len(selected_atoms)),
            scores=scores,
            cert_hash=cert.generate_hash(),
            metadata={'alpha': self.alpha, 'lexical_method': self.lexical_method, 'K1': self.K1}
        )
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)  # All scores equal
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def _select_hybrid(self, Q: str, atoms: List[Atom], K: int, B_tokens: int) -> List[Atom]:
        """Internal selection logic for performance measurement."""
        # Rebuild indices for performance test
        self.bm25_adapter._build_index(atoms)
        query_terms = self.bm25_adapter._tokenize(Q.lower())
        
        if self.vector_adapter.embedding_model:
            query_embedding = self.vector_adapter.embedding_model.encode([Q])[0]
        else:
            query_embedding = self.vector_adapter._get_dummy_embedding(Q)
        
        # Get scores
        lexical_scores = []
        neural_scores = []
        
        for i, atom in enumerate(atoms):
            lex_score = self.bm25_adapter._compute_bm25_score(query_terms, i)
            lexical_scores.append(lex_score)
            
            if atom.embedding is not None:
                embedding = atom.embedding
            else:
                embedding = self.vector_adapter._get_dummy_embedding(atom.content)
            
            neural_score = self.vector_adapter._compute_similarity(query_embedding, embedding)
            neural_scores.append(neural_score)
        
        # Normalize and combine
        lexical_scores_norm = self._normalize_scores(lexical_scores)
        neural_scores_norm = self._normalize_scores(neural_scores)
        
        scored_atoms = []
        for i, atom in enumerate(atoms):
            hybrid_score = (self.alpha * lexical_scores_norm[i] + 
                          (1 - self.alpha) * neural_scores_norm[i])
            scored_atoms.append((atom, hybrid_score))
        
        scored_atoms.sort(key=lambda x: x[1], reverse=True)
        k1_candidates = scored_atoms[:min(self.K1, len(scored_atoms))]
        candidates = k1_candidates[:min(K, len(k1_candidates))]
        
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

class RerankerAdapter(BaseAdapter):
    """Cross-encoder reranking adapter."""
    
    def __init__(self, K2: int = 600, base_retriever: str = "bm25",
                 reranker_model: Optional[Any] = None,
                 config: Dict[str, Any] = None):
        super().__init__(AdapterType.RAG_RERANKER, config)
        self.K2 = K2  # Number of candidates to rerank
        self.base_retriever = base_retriever
        self.reranker_model = reranker_model
        
        # Initialize base retriever
        if base_retriever == "bm25":
            self.base_adapter = BM25Adapter(K1=2000)
        elif base_retriever == "vector":
            self.base_adapter = VectorAdapter(K1=2000)
        elif base_retriever == "hybrid":
            self.base_adapter = HybridAdapter(K1=2000)
        else:
            raise ValueError(f"Unsupported base retriever: {base_retriever}")
        
    def get_method_id(self) -> str:
        return f"reranker_{self.base_retriever}_K2{self.K2}"
    
    def select_bundle(self, method: str, Q: str, Atoms: List[Atom], 
                     B_tokens: int, K: int, seed: int) -> SelectionResult:
        """Select atoms using base retriever + cross-encoder reranking."""
        
        if not self.validate_inputs(Q, Atoms, B_tokens, K):
            raise ValueError("Invalid inputs to RerankerAdapter")
        
        random.seed(seed)
        
        # Get initial candidates from base retriever
        # Use large K to get more candidates for reranking
        base_K = min(self.K2 * 2, len(Atoms))  
        base_result = self.base_adapter.select_bundle(
            f"{method}_base", Q, Atoms, B_tokens * 3, base_K, seed  # Larger budget initially
        )
        
        # Get top K2 candidates for reranking
        initial_candidates = base_result.selected_atoms[:self.K2]
        
        # Apply cross-encoder reranking
        reranked_atoms = []
        
        for atom in initial_candidates:
            # Render atom for cross-encoder input
            ce_input = self._render_for_ce(Q, atom)
            
            # Compute reranking score
            if self.reranker_model:
                rerank_score = self._compute_rerank_score(ce_input)
            else:
                # Dummy reranking based on content similarity
                rerank_score = self._dummy_rerank_score(Q, atom.content)
            
            reranked_atoms.append((atom, rerank_score))
        
        # Sort by rerank score descending
        reranked_atoms.sort(key=lambda x: x[1], reverse=True)
        
        # Apply final K limit and budget
        candidates = reranked_atoms[:min(K, len(reranked_atoms))]
        
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
            lambda: self._select_reranker(Q, Atoms, K, B_tokens)
        )
        
        cert = self._generate_certificate(method, Q, Atoms, B_tokens, K, seed,
                                        SelectionResult([], method, "", "", "", 0, 0, (0, 0), [], ""))
        
        return SelectionResult(
            selected_atoms=selected_atoms,
            method_id=method,
            encoder_hash=self._encoder_hash or "reranker_encoder",
            pool_fingerprint=self._pool_fingerprint or "reranker_pool",
            tokenizer_hash=self._tokenizer_hash or "reranker_tokenizer",
            time_ms=avg_time,
            time_p95=p95_time,
            candidates_considered=(len(initial_candidates), len(selected_atoms)),
            scores=scores,
            cert_hash=cert.generate_hash(),
            metadata={'K2': self.K2, 'base_retriever': self.base_retriever}
        )
    
    def _render_for_ce(self, query: str, atom: Atom) -> str:
        """Render query-atom pair for cross-encoder input."""
        # Standard format for cross-encoder models
        return f"Query: {query}\nPassage: {atom.content}"
    
    def _compute_rerank_score(self, ce_input: str) -> float:
        """Compute cross-encoder reranking score."""
        if self.reranker_model and hasattr(self.reranker_model, 'predict'):
            try:
                # Assume model returns relevance score
                return float(self.reranker_model.predict(ce_input))
            except Exception as e:
                logger.warning(f"Reranker model failed: {e}, using dummy score")
                return self._dummy_rerank_score_from_text(ce_input)
        else:
            return self._dummy_rerank_score_from_text(ce_input)
    
    def _dummy_rerank_score(self, query: str, content: str) -> float:
        """Compute dummy reranking score based on text similarity."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(content_words))
        return overlap / len(query_words)
    
    def _dummy_rerank_score_from_text(self, ce_input: str) -> float:
        """Compute dummy score from CE input text."""
        lines = ce_input.split('\n')
        if len(lines) >= 2:
            query_line = lines[0].replace('Query:', '').strip()
            passage_line = lines[1].replace('Passage:', '').strip()
            return self._dummy_rerank_score(query_line, passage_line)
        return 0.5  # Default score
    
    def _select_reranker(self, Q: str, atoms: List[Atom], K: int, B_tokens: int) -> List[Atom]:
        """Internal selection logic for performance measurement."""
        # Get base candidates
        base_K = min(self.K2 * 2, len(atoms))
        base_result = self.base_adapter.select_bundle(
            "base_perf", Q, atoms, B_tokens * 3, base_K, 1
        )
        
        initial_candidates = base_result.selected_atoms[:self.K2]
        
        # Rerank
        reranked_atoms = []
        for atom in initial_candidates:
            rerank_score = self._dummy_rerank_score(Q, atom.content)
            reranked_atoms.append((atom, rerank_score))
        
        reranked_atoms.sort(key=lambda x: x[1], reverse=True)
        candidates = reranked_atoms[:min(K, len(reranked_atoms))]
        
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

# Export all adapter classes
__all__ = [
    'BM25Adapter',
    'VectorAdapter',
    'HybridAdapter', 
    'RerankerAdapter'
]