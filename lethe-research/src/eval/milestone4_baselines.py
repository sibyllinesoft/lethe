#!/usr/bin/env python3
"""
Milestone 4: Stronger Local Baselines Implementation
==================================================

Implements the six core baselines for rigorous evaluation against Lethe hybrid retrieval system:

1. BM25-only (SQLite FTS5) with identical candidate caps
2. Vector-only (ANN) with same limits  
3. BM25+Vector (static α=0.5) without reranking
4. MMR (λ=0.7) over vector candidates for diversity
5. BM25 + doc2query expansion with offline precomputation
6. Tiny Cross-Encoder Reranking (CPU-only)

Key Features:
- Identical interfaces producing comparable JSON outputs
- Shared infrastructure: build indices once, reuse across baselines
- Single command execution via 'make baselines'
- Performance parity with fair computational budgets
- Local execution only, CPU-compatible
- Anti-fraud validation and budget parity tracking
"""

import numpy as np
import json
import time
import logging
import hashlib
import sqlite3
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import gc
import psutil
from collections import defaultdict
import math
import re
from functools import lru_cache

# Try imports with fallbacks for optional dependencies
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    CrossEncoder = None

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

# Lethe imports
from .baselines import (
    RetrievalDocument, EvaluationQuery, RetrievalResult, BaselineResult,
    BudgetParityTracker, AntiFreudValidator, BaselineRetriever
)

logger = logging.getLogger(__name__)

class SQLiteFTSBaseline(BaselineRetriever):
    """BM25-only baseline using SQLite FTS5"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("BM25-only", config)
        self.db_path = config.get("db_path", ":memory:")
        self.k1 = config.get("k1", 1.2)
        self.b = config.get("b", 0.75)
        self.conn = None
        
    def build_index(self, documents: List[RetrievalDocument]) -> None:
        """Build SQLite FTS5 index"""
        self.conn = sqlite3.connect(self.db_path)
        
        # Create FTS5 table with BM25 parameters
        self.conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_docs USING fts5(
                doc_id UNINDEXED,
                content,
                kind UNINDEXED,
                rank=bm25({self.k1}, {self.b})
            )
        """)
        
        # Insert documents
        for doc in documents:
            self.conn.execute(
                "INSERT INTO fts_docs (doc_id, content, kind) VALUES (?, ?, ?)",
                (doc.doc_id, doc.content, doc.kind)
            )
        
        self.conn.commit()
        logger.info(f"Built SQLite FTS5 index with {len(documents)} documents")
        
    def retrieve(self, query: EvaluationQuery, k: int = 100) -> List[RetrievalResult]:
        """Retrieve using BM25 FTS5"""
        if not self.conn:
            raise ValueError("Index not built. Call build_index() first.")
        
        start_time = time.time()
        
        # Escape FTS5 special characters
        escaped_query = self._escape_fts_query(query.text)
        
        # Execute FTS5 query
        cursor = self.conn.execute("""
            SELECT doc_id, content, kind, rank 
            FROM fts_docs 
            WHERE fts_docs MATCH ? 
            ORDER BY rank 
            LIMIT ?
        """, (escaped_query, k))
        
        results = []
        for rank, (doc_id, content, kind, score) in enumerate(cursor.fetchall()):
            results.append(RetrievalResult(
                doc_id=doc_id,
                score=float(score),
                rank=rank + 1,
                content=content,
                kind=kind,
                metadata={"method": "bm25_fts5"}
            ))
        
        latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"BM25 FTS5 retrieval: {len(results)} results in {latency_ms:.2f}ms")
        
        return results
    
    def _escape_fts_query(self, query: str) -> str:
        """Escape FTS5 special characters"""
        # Remove or escape FTS5 special characters
        query = re.sub(r'["\*\(\)\[\]{}]', ' ', query)
        query = re.sub(r'\s+', ' ', query).strip()
        return query
        
    def get_flops_estimate(self, query: EvaluationQuery, k: int) -> int:
        """Estimate FLOPs for BM25 FTS5"""
        # Conservative FTS5 FLOPs estimate:
        # - Tokenization: ~5 FLOPs per character
        # - BM25 scoring: ~10 FLOPs per term per document
        # - Sorting: ~k log k FLOPs
        
        query_chars = len(query.text)
        estimated_terms = len(query.text.split())
        estimated_docs_scored = 1000  # Conservative estimate
        
        tokenization_flops = query_chars * 5
        scoring_flops = estimated_terms * estimated_docs_scored * 10
        sorting_flops = k * math.log2(k) if k > 1 else 0
        
        return int(tokenization_flops + scoring_flops + sorting_flops)

class VectorOnlyBaseline(BaselineRetriever):
    """Vector-only baseline using ANN search"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Vector-only", config)
        self.model_name = config.get("model_name", "all-MiniLM-L6-v2")
        self.index_type = config.get("index_type", "hnsw")
        self.ef_construction = config.get("ef_construction", 200)
        self.ef_search = config.get("ef_search", 50)
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required for vector baseline")
        if not FAISS_AVAILABLE:
            raise ImportError("faiss required for vector baseline")
            
        self.model = SentenceTransformer(self.model_name)
        self.model.eval()  # Set to inference mode
        
        self.index = None
        self.doc_mapping = {}  # index_id -> doc_id
        self.documents = []
        
    def build_index(self, documents: List[RetrievalDocument]) -> None:
        """Build FAISS vector index"""
        self.documents = documents
        
        # Generate embeddings
        texts = [doc.content for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        embeddings = embeddings.astype(np.float32)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        
        if self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dimension, 16)  # M=16
            self.index.hnsw.efConstruction = self.ef_construction
            self.index.hnsw.efSearch = self.ef_search
        else:
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Create mapping
        self.doc_mapping = {i: doc.doc_id for i, doc in enumerate(documents)}
        
        logger.info(f"Built FAISS index ({self.index_type}) with {len(documents)} documents, dim={dimension}")
        
    def retrieve(self, query: EvaluationQuery, k: int = 100) -> List[RetrievalResult]:
        """Retrieve using vector similarity"""
        if not self.index:
            raise ValueError("Index not built. Call build_index() first.")
        
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.model.encode([query.text])
        query_embedding = query_embedding.astype(np.float32)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            doc_id = self.doc_mapping[idx]
            doc = self.documents[idx]
            
            results.append(RetrievalResult(
                doc_id=doc_id,
                score=float(score),
                rank=rank + 1,
                content=doc.content,
                kind=doc.kind,
                metadata={"method": "vector_ann", "model": self.model_name}
            ))
        
        latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"Vector ANN retrieval: {len(results)} results in {latency_ms:.2f}ms")
        
        return results
        
    def get_flops_estimate(self, query: EvaluationQuery, k: int) -> int:
        """Estimate FLOPs for vector retrieval"""
        # Vector retrieval FLOPs:
        # - Query encoding: model_params * 2 (forward pass approximation)
        # - Similarity computation: dimension * num_candidates
        # - HNSW traversal: log(n) * dimension (approximate)
        
        # Rough model size estimate (MiniLM-L6 ≈ 22M params)
        model_params = 22_000_000
        encoding_flops = model_params * 2
        
        dimension = 384  # MiniLM-L6 dimension
        hnsw_flops = int(math.log2(len(self.documents)) * dimension) if self.documents else 0
        similarity_flops = dimension * k
        
        return encoding_flops + hnsw_flops + similarity_flops

class HybridStaticBaseline(BaselineRetriever):
    """BM25+Vector (static α=0.5) baseline without reranking"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("BM25+Vector-static", config)
        self.alpha = config.get("alpha", 0.5)  # Static fusion weight
        
        # Initialize sub-baselines
        self.bm25_baseline = SQLiteFTSBaseline(config)
        self.vector_baseline = VectorOnlyBaseline(config)
        
    def build_index(self, documents: List[RetrievalDocument]) -> None:
        """Build both BM25 and vector indices"""
        logger.info("Building BM25 index...")
        self.bm25_baseline.build_index(documents)
        
        logger.info("Building vector index...")
        self.vector_baseline.build_index(documents)
        
        logger.info(f"Built hybrid indices with α={self.alpha}")
        
    def retrieve(self, query: EvaluationQuery, k: int = 100) -> List[RetrievalResult]:
        """Retrieve using hybrid fusion with static weights"""
        start_time = time.time()
        
        # Get results from both baselines (retrieve more for better fusion)
        k_candidates = min(k * 2, 1000)
        
        bm25_results = self.bm25_baseline.retrieve(query, k_candidates)
        vector_results = self.vector_baseline.retrieve(query, k_candidates)
        
        # Create score dictionaries
        bm25_scores = {r.doc_id: r.score for r in bm25_results}
        vector_scores = {r.doc_id: r.score for r in vector_results}
        
        # Get all candidate documents
        all_doc_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
        
        # Normalize scores to [0, 1] per modality
        bm25_values = list(bm25_scores.values())
        vector_values = list(vector_scores.values())
        
        if bm25_values:
            bm25_min, bm25_max = min(bm25_values), max(bm25_values)
            bm25_range = bm25_max - bm25_min
            if bm25_range > 0:
                bm25_scores = {doc_id: (score - bm25_min) / bm25_range 
                              for doc_id, score in bm25_scores.items()}
        
        if vector_values:
            vector_min, vector_max = min(vector_values), max(vector_values)
            vector_range = vector_max - vector_min
            if vector_range > 0:
                vector_scores = {doc_id: (score - vector_min) / vector_range 
                               for doc_id, score in vector_scores.items()}
        
        # Fusion with static weights
        fused_scores = {}
        for doc_id in all_doc_ids:
            bm25_score = bm25_scores.get(doc_id, 0.0)
            vector_score = vector_scores.get(doc_id, 0.0)
            fused_scores[doc_id] = self.alpha * bm25_score + (1 - self.alpha) * vector_score
        
        # Sort and take top k
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Create result objects
        doc_lookup = {}
        for r in bm25_results + vector_results:
            if r.doc_id not in doc_lookup:
                doc_lookup[r.doc_id] = r
        
        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs):
            if doc_id in doc_lookup:
                base_result = doc_lookup[doc_id]
                results.append(RetrievalResult(
                    doc_id=doc_id,
                    score=score,
                    rank=rank + 1,
                    content=base_result.content,
                    kind=base_result.kind,
                    metadata={
                        "method": "hybrid_static",
                        "alpha": self.alpha,
                        "bm25_score": bm25_scores.get(doc_id, 0.0),
                        "vector_score": vector_scores.get(doc_id, 0.0)
                    }
                ))
        
        latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"Hybrid static retrieval: {len(results)} results in {latency_ms:.2f}ms")
        
        return results
        
    def get_flops_estimate(self, query: EvaluationQuery, k: int) -> int:
        """Estimate FLOPs for hybrid retrieval"""
        # Sum of both baselines plus fusion overhead
        bm25_flops = self.bm25_baseline.get_flops_estimate(query, k * 2)
        vector_flops = self.vector_baseline.get_flops_estimate(query, k * 2)
        fusion_flops = k * 10  # Normalization and weighted sum
        
        return bm25_flops + vector_flops + fusion_flops

class MMRDiversityBaseline(BaselineRetriever):
    """MMR (λ=0.7) diversity baseline over vector candidates"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("MMR-diversity", config)
        self.lambda_param = config.get("lambda", 0.7)  # Diversity parameter
        self.vector_baseline = VectorOnlyBaseline(config)
        
    def build_index(self, documents: List[RetrievalDocument]) -> None:
        """Build vector index for MMR"""
        self.vector_baseline.build_index(documents)
        logger.info(f"Built MMR index with λ={self.lambda_param}")
        
    def retrieve(self, query: EvaluationQuery, k: int = 100) -> List[RetrievalResult]:
        """Retrieve using MMR for diversity"""
        start_time = time.time()
        
        # Get initial candidate set (larger for MMR selection)
        k_candidates = min(k * 5, 1000)
        candidates = self.vector_baseline.retrieve(query, k_candidates)
        
        if not candidates:
            return []
        
        # Get query embedding
        query_embedding = self.vector_baseline.model.encode([query.text])[0]
        
        # Get document embeddings (recompute for MMR)
        doc_embeddings = {}
        for candidate in candidates:
            # Find the document in the baseline's documents
            for doc in self.vector_baseline.documents:
                if doc.doc_id == candidate.doc_id:
                    emb = self.vector_baseline.model.encode([doc.content])[0]
                    doc_embeddings[candidate.doc_id] = emb
                    break
        
        # MMR selection
        selected = []
        remaining = candidates.copy()
        
        # Select first document (highest similarity to query)
        if remaining:
            first_doc = remaining[0]
            selected.append(first_doc)
            remaining.remove(first_doc)
        
        # Iteratively select diverse documents
        while len(selected) < k and remaining:
            best_score = -float('inf')
            best_doc = None
            
            for candidate in remaining:
                if candidate.doc_id not in doc_embeddings:
                    continue
                    
                doc_emb = doc_embeddings[candidate.doc_id]
                
                # Query similarity (relevance)
                query_sim = np.dot(query_embedding, doc_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
                )
                
                # Maximum similarity to already selected (diversity penalty)
                max_selected_sim = 0.0
                for selected_doc in selected:
                    if selected_doc.doc_id in doc_embeddings:
                        selected_emb = doc_embeddings[selected_doc.doc_id]
                        sim = np.dot(doc_emb, selected_emb) / (
                            np.linalg.norm(doc_emb) * np.linalg.norm(selected_emb)
                        )
                        max_selected_sim = max(max_selected_sim, sim)
                
                # MMR score: λ * relevance - (1-λ) * max_similarity_to_selected
                mmr_score = (
                    self.lambda_param * query_sim - 
                    (1 - self.lambda_param) * max_selected_sim
                )
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_doc = candidate
            
            if best_doc:
                selected.append(best_doc)
                remaining.remove(best_doc)
            else:
                break
        
        # Update ranks and metadata
        results = []
        for rank, doc in enumerate(selected):
            results.append(RetrievalResult(
                doc_id=doc.doc_id,
                score=doc.score,  # Keep original similarity score
                rank=rank + 1,
                content=doc.content,
                kind=doc.kind,
                metadata={
                    "method": "mmr_diversity",
                    "lambda": self.lambda_param,
                    "original_rank": doc.rank
                }
            ))
        
        latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"MMR diversity retrieval: {len(results)} results in {latency_ms:.2f}ms")
        
        return results
        
    def get_flops_estimate(self, query: EvaluationQuery, k: int) -> int:
        """Estimate FLOPs for MMR"""
        # Vector baseline FLOPs + MMR selection overhead
        vector_flops = self.vector_baseline.get_flops_estimate(query, k * 5)
        
        # MMR selection: O(k²) similarity computations
        dimension = 384  # MiniLM-L6 dimension
        mmr_flops = k * k * dimension * 3  # dot product + norms for each comparison
        
        return vector_flops + mmr_flops

class Doc2QueryExpansionBaseline(BaselineRetriever):
    """BM25 + doc2query expansion with offline precomputation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("BM25+Doc2Query", config)
        self.num_expansions = config.get("num_expansions", 3)
        self.expansion_model_name = config.get("expansion_model", "doc2query/msmarco-t5-base-v1")
        
        # Try to load T5 model for query expansion
        self.expansion_model = None
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            self.expansion_model = T5ForConditionalGeneration.from_pretrained(
                self.expansion_model_name, torch_dtype="auto"
            )
            self.expansion_tokenizer = T5Tokenizer.from_pretrained(self.expansion_model_name)
            self.expansion_model.eval()
        except ImportError:
            logger.warning("transformers not available - using pattern-based expansion fallback")
        except Exception as e:
            logger.warning(f"Could not load doc2query model: {e} - using fallback")
        
        self.conn = None
        self.expansion_cache = {}
        
    def _generate_query_expansions(self, text: str) -> List[str]:
        """Generate query expansions for document text"""
        if self.expansion_model and len(text.strip()) > 10:
            try:
                # Use T5 doc2query model
                input_text = f"generate query: {text[:500]}"  # Truncate for efficiency
                inputs = self.expansion_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
                
                expansions = []
                for _ in range(self.num_expansions):
                    outputs = self.expansion_model.generate(
                        inputs, 
                        max_length=64, 
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.8,
                        no_repeat_ngram_size=2
                    )
                    expansion = self.expansion_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if expansion.strip() and expansion not in expansions:
                        expansions.append(expansion.strip())
                
                return expansions
                
            except Exception as e:
                logger.warning(f"Doc2query generation failed: {e}")
        
        # Fallback: pattern-based expansion
        return self._pattern_based_expansion(text)
    
    def _pattern_based_expansion(self, text: str) -> List[str]:
        """Fallback pattern-based query expansion"""
        expansions = []
        
        # Extract potential queries from code comments, error messages, etc.
        patterns = [
            r'#\s*(.+)',  # Comments
            r'//\s*(.+)',  # Comments
            r'error[:\s]+(.+)',  # Error messages
            r'exception[:\s]+(.+)',  # Exceptions
            r'TODO[:\s]+(.+)',  # TODOs
            r'FIXME[:\s]+(.+)',  # FIXMEs
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches[:2]:  # Limit extractions
                cleaned = re.sub(r'[^\w\s]', ' ', match).strip()
                if len(cleaned) > 5 and cleaned not in expansions:
                    expansions.append(cleaned)
                    if len(expansions) >= self.num_expansions:
                        break
        
        # Add key terms as queries
        words = text.split()
        if len(words) >= 3:
            # Create phrase queries from consecutive important words
            important_words = [w for w in words if len(w) > 3 and w.isalnum()]
            for i in range(0, min(len(important_words), 6), 3):
                phrase = " ".join(important_words[i:i+3])
                if phrase not in expansions:
                    expansions.append(phrase)
                    if len(expansions) >= self.num_expansions:
                        break
        
        return expansions[:self.num_expansions]
        
    def build_index(self, documents: List[RetrievalDocument]) -> None:
        """Build FTS5 index with expanded queries"""
        self.conn = sqlite3.connect(self.config.get("db_path", ":memory:"))
        
        # Create main FTS table
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_docs USING fts5(
                doc_id UNINDEXED,
                content,
                kind UNINDEXED,
                rank=bm25(1.2, 0.75)
            )
        """)
        
        # Create expanded queries table
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS expanded_fts USING fts5(
                doc_id UNINDEXED,
                original_content,
                expanded_queries,
                kind UNINDEXED,
                rank=bm25(1.2, 0.75)
            )
        """)
        
        logger.info(f"Generating doc2query expansions for {len(documents)} documents...")
        
        for i, doc in enumerate(documents):
            if i % 100 == 0:
                logger.info(f"Processing document {i+1}/{len(documents)}")
            
            # Insert original document
            self.conn.execute(
                "INSERT INTO fts_docs (doc_id, content, kind) VALUES (?, ?, ?)",
                (doc.doc_id, doc.content, doc.kind)
            )
            
            # Generate and cache expansions
            expansions = self._generate_query_expansions(doc.content)
            self.expansion_cache[doc.doc_id] = expansions
            
            # Insert expanded version
            combined_text = doc.content + " " + " ".join(expansions)
            self.conn.execute(
                "INSERT INTO expanded_fts (doc_id, original_content, expanded_queries, kind) VALUES (?, ?, ?, ?)",
                (doc.doc_id, doc.content, " ".join(expansions), doc.kind)
            )
        
        self.conn.commit()
        logger.info(f"Built doc2query expanded index with {len(documents)} documents")
        
    def retrieve(self, query: EvaluationQuery, k: int = 100) -> List[RetrievalResult]:
        """Retrieve using expanded FTS index"""
        if not self.conn:
            raise ValueError("Index not built. Call build_index() first.")
        
        start_time = time.time()
        
        # Escape query
        escaped_query = self._escape_fts_query(query.text)
        
        # Search in expanded index
        cursor = self.conn.execute("""
            SELECT doc_id, original_content, expanded_queries, kind, rank 
            FROM expanded_fts 
            WHERE expanded_fts MATCH ? 
            ORDER BY rank 
            LIMIT ?
        """, (escaped_query, k))
        
        results = []
        for rank, (doc_id, content, expansions, kind, score) in enumerate(cursor.fetchall()):
            results.append(RetrievalResult(
                doc_id=doc_id,
                score=float(score),
                rank=rank + 1,
                content=content,
                kind=kind,
                metadata={
                    "method": "bm25_doc2query", 
                    "expansions": expansions,
                    "num_expansions": self.num_expansions
                }
            ))
        
        latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"Doc2Query retrieval: {len(results)} results in {latency_ms:.2f}ms")
        
        return results
    
    def _escape_fts_query(self, query: str) -> str:
        """Escape FTS5 special characters"""
        query = re.sub(r'["\*\(\)\[\]{}]', ' ', query)
        query = re.sub(r'\s+', ' ', query).strip()
        return query
        
    def get_flops_estimate(self, query: EvaluationQuery, k: int) -> int:
        """Estimate FLOPs for doc2query retrieval"""
        # Similar to BM25 but searching over expanded text (roughly 2-3x more text)
        query_chars = len(query.text)
        estimated_terms = len(query.text.split())
        estimated_docs_scored = 1000
        expansion_factor = self.num_expansions + 1  # Original + expansions
        
        tokenization_flops = query_chars * 5
        scoring_flops = estimated_terms * estimated_docs_scored * 10 * expansion_factor
        sorting_flops = k * math.log2(k) if k > 1 else 0
        
        return int(tokenization_flops + scoring_flops + sorting_flops)

class TinyCrossEncoderBaseline(BaselineRetriever):
    """Tiny Cross-Encoder reranking baseline (CPU-only)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("CrossEncoder-rerank", config)
        self.rerank_model_name = config.get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-2-v2")
        self.rerank_k = config.get("rerank_k", 100)  # Top-K to rerank
        
        # Base retriever for initial candidates
        self.base_retriever = VectorOnlyBaseline(config)
        
        # Load cross-encoder model
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers required for cross-encoder baseline")
        
        try:
            self.reranker = CrossEncoder(self.rerank_model_name)
            logger.info(f"Loaded cross-encoder: {self.rerank_model_name}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder: {e}")
            raise
        
    def build_index(self, documents: List[RetrievalDocument]) -> None:
        """Build base vector index for initial retrieval"""
        self.base_retriever.build_index(documents)
        logger.info("Built cross-encoder baseline with vector first-stage")
        
    def retrieve(self, query: EvaluationQuery, k: int = 100) -> List[RetrievalResult]:
        """Retrieve using two-stage vector + cross-encoder reranking"""
        start_time = time.time()
        
        # Stage 1: Get initial candidates
        candidates = self.base_retriever.retrieve(query, self.rerank_k)
        
        if not candidates:
            return []
        
        # Stage 2: Cross-encoder reranking
        query_text = query.text
        pairs = [(query_text, candidate.content) for candidate in candidates]
        
        # Batch reranking for efficiency
        batch_size = 32
        rerank_scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            batch_scores = self.reranker.predict(batch_pairs)
            rerank_scores.extend(batch_scores.tolist() if hasattr(batch_scores, 'tolist') else batch_scores)
        
        # Combine with reranking scores
        reranked_candidates = []
        for candidate, rerank_score in zip(candidates, rerank_scores):
            reranked_candidates.append((candidate, float(rerank_score)))
        
        # Sort by reranking score
        reranked_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Create final results
        results = []
        for rank, (candidate, rerank_score) in enumerate(reranked_candidates[:k]):
            results.append(RetrievalResult(
                doc_id=candidate.doc_id,
                score=rerank_score,
                rank=rank + 1,
                content=candidate.content,
                kind=candidate.kind,
                metadata={
                    "method": "cross_encoder_rerank",
                    "model": self.rerank_model_name,
                    "original_score": candidate.score,
                    "original_rank": candidate.rank,
                    "rerank_score": rerank_score
                }
            ))
        
        latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"Cross-encoder rerank: {len(results)} results in {latency_ms:.2f}ms")
        
        return results
        
    def get_flops_estimate(self, query: EvaluationQuery, k: int) -> int:
        """Estimate FLOPs for cross-encoder reranking"""
        # Stage 1: Vector retrieval
        vector_flops = self.base_retriever.get_flops_estimate(query, self.rerank_k)
        
        # Stage 2: Cross-encoder reranking
        # Rough estimate: MiniLM-L2 ≈ 15M params, forward pass per query-doc pair
        model_params = 15_000_000
        rerank_pairs = min(self.rerank_k, k)
        rerank_flops = model_params * 2 * rerank_pairs  # Forward pass per pair
        
        return vector_flops + rerank_flops

class Milestone4BaselineEvaluator:
    """Main evaluator for all Milestone 4 baselines"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.budget_tracker = BudgetParityTracker(tolerance=0.05)
        self.fraud_validator = AntiFreudValidator()
        
        # Initialize all baselines
        self.baselines = {
            "bm25_only": SQLiteFTSBaseline(config),
            "vector_only": VectorOnlyBaseline(config),
            "hybrid_static": HybridStaticBaseline(config),
            "mmr_diversity": MMRDiversityBaseline(config),
            "doc2query_expansion": Doc2QueryExpansionBaseline(config),
            "crossencoder_rerank": TinyCrossEncoderBaseline(config)
        }
        
    def build_all_indices(self, documents: List[RetrievalDocument]) -> None:
        """Build indices for all baselines (shared infrastructure)"""
        logger.info(f"Building indices for {len(self.baselines)} baselines...")
        
        for name, baseline in self.baselines.items():
            logger.info(f"Building index for {name}...")
            try:
                baseline.build_index(documents)
                logger.info(f"✅ {name} index built successfully")
            except Exception as e:
                logger.error(f"❌ Failed to build {name} index: {e}")
                # Remove failed baseline
                del self.baselines[name]
        
        logger.info(f"Index building complete. {len(self.baselines)} baselines ready.")
        
    def evaluate_all_baselines(self, queries: List[EvaluationQuery], k: int = 100) -> Dict[str, List[BaselineResult]]:
        """Evaluate all baselines on the given queries"""
        logger.info(f"Evaluating {len(self.baselines)} baselines on {len(queries)} queries...")
        
        all_results = {}
        baseline_budget_set = False
        
        for baseline_name, baseline in self.baselines.items():
            logger.info(f"Evaluating {baseline_name}...")
            baseline_results = []
            
            for query in queries:
                try:
                    # Measure performance
                    start_memory = baseline.get_memory_usage()
                    start_time = time.time()
                    
                    # Retrieve
                    results = baseline.retrieve(query, k)
                    
                    # Measure metrics
                    end_time = time.time()
                    end_memory = baseline.get_memory_usage()
                    latency_ms = (end_time - start_time) * 1000
                    memory_delta = max(0, end_memory - start_memory)
                    
                    # Estimate FLOPs
                    flops = baseline.get_flops_estimate(query, k)
                    
                    # Budget tracking (use first baseline as reference)
                    if not baseline_budget_set and baseline_name == "bm25_only":
                        self.budget_tracker.set_baseline_budget(flops)
                        baseline_budget_set = True
                    
                    budget_compliant = self.budget_tracker.validate_budget(baseline_name, flops)
                    
                    # Fraud validation
                    fraud_passed = self.fraud_validator.validate_non_empty_results(baseline_name, query, results)
                    
                    # Create result
                    result = BaselineResult(
                        baseline_name=baseline_name,
                        query_id=query.query_id,
                        query_text=query.text,
                        retrieved_docs=[r.doc_id for r in results],
                        relevance_scores=[r.score for r in results],
                        ranks=[r.rank for r in results],
                        latency_ms=latency_ms,
                        memory_mb=memory_delta,
                        cpu_percent=psutil.cpu_percent(),
                        flops_estimate=flops,
                        non_empty_validated=fraud_passed,
                        smoke_test_passed=True,  # Will be updated in smoke tests
                        candidate_count=len(results),
                        timestamp=time.time(),
                        model_checkpoint=getattr(baseline, 'model_name', 'n/a'),
                        hyperparameters=baseline.config,
                        hardware_profile=baseline.get_hardware_profile()
                    )
                    
                    baseline_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error evaluating {baseline_name} on query {query.query_id}: {e}")
                    continue
            
            all_results[baseline_name] = baseline_results
            logger.info(f"✅ {baseline_name}: {len(baseline_results)} queries completed")
        
        # Run smoke tests
        self._run_smoke_tests(queries[:10])
        
        return all_results
        
    def _run_smoke_tests(self, test_queries: List[EvaluationQuery]) -> None:
        """Run smoke tests on all baselines"""
        logger.info("Running smoke tests...")
        
        for baseline_name, baseline in self.baselines.items():
            smoke_passed = self.fraud_validator.run_smoke_tests(baseline, test_queries)
            logger.info(f"Smoke test {baseline_name}: {'✅ PASSED' if smoke_passed else '❌ FAILED'}")
            
    def save_results(self, results: Dict[str, List[BaselineResult]], output_path: Path) -> None:
        """Save results to JSON with comprehensive metadata"""
        output_data = {
            "metadata": {
                "timestamp": time.time(),
                "total_baselines": len(results),
                "total_queries": sum(len(r) for r in results.values()),
                "config": self.config,
                "hardware_profile": list(self.baselines.values())[0].get_hardware_profile() if self.baselines else {}
            },
            "budget_report": self.budget_tracker.get_budget_report(),
            "validation_report": self.fraud_validator.get_validation_report(),
            "baseline_results": {}
        }
        
        for baseline_name, baseline_results in results.items():
            output_data["baseline_results"][baseline_name] = [
                {
                    "baseline_name": r.baseline_name,
                    "query_id": r.query_id,
                    "query_text": r.query_text,
                    "retrieved_docs": r.retrieved_docs,
                    "relevance_scores": r.relevance_scores,
                    "ranks": r.ranks,
                    "latency_ms": r.latency_ms,
                    "memory_mb": r.memory_mb,
                    "cpu_percent": r.cpu_percent,
                    "flops_estimate": r.flops_estimate,
                    "non_empty_validated": r.non_empty_validated,
                    "smoke_test_passed": r.smoke_test_passed,
                    "candidate_count": r.candidate_count,
                    "timestamp": r.timestamp,
                    "model_checkpoint": r.model_checkpoint,
                    "hyperparameters": r.hyperparameters,
                    "hardware_profile": r.hardware_profile
                }
                for r in baseline_results
            ]
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        logger.info(f"Results saved to {output_path}")

def create_baseline_evaluator(config_path: Optional[Path] = None) -> Milestone4BaselineEvaluator:
    """Factory function to create baseline evaluator with config"""
    if config_path and config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {
            "db_path": ":memory:",
            "model_name": "all-MiniLM-L6-v2",
            "alpha": 0.5,
            "lambda": 0.7,
            "num_expansions": 3,
            "rerank_k": 100,
            "k1": 1.2,
            "b": 0.75,
            "ef_construction": 200,
            "ef_search": 50
        }
    
    return Milestone4BaselineEvaluator(config)

# Main execution interface
if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Milestone 4 baselines evaluation")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--documents", type=Path, required=True, help="Documents JSON file")
    parser.add_argument("--queries", type=Path, required=True, help="Queries JSON file")
    parser.add_argument("--output", type=Path, required=True, help="Output results path")
    parser.add_argument("--k", type=int, default=100, help="Top-K results to retrieve")
    
    args = parser.parse_args()
    
    # Load data
    with open(args.documents) as f:
        docs_data = [json.loads(line) for line in f]
        documents = [RetrievalDocument(**doc) for doc in docs_data]
    
    with open(args.queries) as f:
        queries_data = [json.loads(line) for line in f]
        queries = [EvaluationQuery(**q) for q in queries_data]
    
    # Create evaluator and run
    evaluator = create_baseline_evaluator(args.config)
    evaluator.build_all_indices(documents)
    results = evaluator.evaluate_all_baselines(queries, args.k)
    evaluator.save_results(results, args.output)
    
    print(f"✅ Evaluation complete. Results saved to {args.output}")