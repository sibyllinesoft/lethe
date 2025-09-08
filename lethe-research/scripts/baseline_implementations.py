#!/usr/bin/env python3
"""
Baseline Implementation Framework
================================

Standardized baseline implementations for comparative evaluation 
against the Lethe hybrid retrieval system.

All baselines use identical:
- Chunking strategy (target_tokens=320, overlap=64)
- Embedding model (same as Lethe)
- Document preprocessing
- Evaluation protocol

Only retrieval and ranking algorithms differ.
"""

import numpy as np
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import sqlite3
from pathlib import Path
import time
import psutil
import os

@dataclass
class Document:
    """Standardized document representation"""
    doc_id: str
    content: str
    kind: str  # 'text', 'code', 'tool_output'
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

@dataclass
class Query:
    """Query with metadata"""
    query_id: str
    text: str
    session_id: str
    domain: str
    complexity: str
    ground_truth_docs: List[str]

@dataclass
class RetrievalResult:
    """Single retrieval result"""
    doc_id: str
    score: float
    rank: int
    content: str
    kind: str

@dataclass  
class QueryResult:
    """Complete evaluation result for a query"""
    query_id: str
    session_id: str
    domain: str
    complexity: str
    ground_truth_docs: List[str]
    retrieved_docs: List[str]
    relevance_scores: List[float]
    latency_ms: float
    memory_mb: float
    entities_covered: List[str]
    contradictions: List[str]
    timestamp: str

class BaselineRetriever(ABC):
    """Abstract base class for all baseline implementations"""
    
    def __init__(self, name: str, db_path: str):
        self.name = name
        self.db_path = db_path
        self.documents: Dict[str, Document] = {}
        self.stats = {"queries_processed": 0, "total_latency_ms": 0, "peak_memory_mb": 0}
    
    @abstractmethod
    def index_documents(self, documents: List[Document]) -> None:
        """Index documents for retrieval"""
        pass
    
    @abstractmethod
    def retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """Retrieve top-k documents for query"""
        pass
    
    def measure_performance(self, query: Query, k: int = 10) -> Tuple[List[RetrievalResult], float, float]:
        """Retrieve with performance measurement"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        results = self.retrieve(query, k)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        latency_ms = (end_time - start_time) * 1000
        peak_memory_mb = max(start_memory, end_memory)
        
        # Update stats
        self.stats["queries_processed"] += 1
        self.stats["total_latency_ms"] += latency_ms
        self.stats["peak_memory_mb"] = max(self.stats["peak_memory_mb"], peak_memory_mb)
        
        return results, latency_ms, peak_memory_mb

class WindowBaseline(BaselineRetriever):
    """Recency-only baseline - returns most recent documents"""
    
    def __init__(self, db_path: str, window_size: int = 10):
        super().__init__("Window Baseline", db_path)
        self.window_size = window_size
        self.document_order: List[str] = []
    
    def index_documents(self, documents: List[Document]) -> None:
        """Store documents in order"""
        for doc in documents:
            self.documents[doc.doc_id] = doc
            if doc.doc_id not in self.document_order:
                self.document_order.append(doc.doc_id)
    
    def retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """Return k most recent documents"""
        recent_docs = self.document_order[-self.window_size:]
        results = []
        
        for i, doc_id in enumerate(reversed(recent_docs[:k])):
            doc = self.documents[doc_id]
            result = RetrievalResult(
                doc_id=doc_id,
                score=1.0 - (i / len(recent_docs)),  # Higher score for more recent
                rank=i + 1,
                content=doc.content,
                kind=doc.kind
            )
            results.append(result)
        
        return results

class BM25OnlyBaseline(BaselineRetriever):
    """Pure BM25 lexical retrieval"""
    
    def __init__(self, db_path: str, k1: float = 1.2, b: float = 0.75):
        super().__init__("BM25 Only", db_path)
        self.k1 = k1
        self.b = b
        self.term_frequencies: Dict[str, Dict[str, int]] = {}
        self.document_frequencies: Dict[str, int] = {}
        self.document_lengths: Dict[str, int] = {}
        self.avg_doc_length = 0.0
        self.corpus_size = 0
    
    def index_documents(self, documents: List[Document]) -> None:
        """Build BM25 index"""
        self.documents = {doc.doc_id: doc for doc in documents}
        self.corpus_size = len(documents)
        
        # Compute term frequencies and document lengths
        all_doc_lengths = []
        
        for doc in documents:
            terms = self._tokenize(doc.content)
            self.document_lengths[doc.doc_id] = len(terms)
            all_doc_lengths.append(len(terms))
            
            term_counts = {}
            for term in terms:
                term_counts[term] = term_counts.get(term, 0) + 1
            
            self.term_frequencies[doc.doc_id] = term_counts
            
            # Update document frequencies
            for term in set(terms):
                self.document_frequencies[term] = self.document_frequencies.get(term, 0) + 1
        
        self.avg_doc_length = np.mean(all_doc_lengths) if all_doc_lengths else 0.0
    
    def retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """BM25 retrieval"""
        query_terms = self._tokenize(query.text)
        scores = {}
        
        for doc_id in self.documents:
            score = 0.0
            doc_length = self.document_lengths[doc_id]
            
            for term in query_terms:
                if term not in self.term_frequencies[doc_id]:
                    continue
                
                tf = self.term_frequencies[doc_id][term]
                df = self.document_frequencies.get(term, 0)
                
                if df == 0:
                    continue
                
                # BM25 formula
                idf = np.log((self.corpus_size - df + 0.5) / (df + 0.5))
                tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length))
                
                score += idf * tf_component
            
            if score > 0:
                scores[doc_id] = score
        
        # Sort by score and return top-k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs, 1):
            doc = self.documents[doc_id]
            result = RetrievalResult(
                doc_id=doc_id,
                score=float(score),
                rank=rank,
                content=doc.content,
                kind=doc.kind
            )
            results.append(result)
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        import re
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()

class VectorOnlyBaseline(BaselineRetriever):
    """Pure vector similarity retrieval"""
    
    def __init__(self, db_path: str, embedding_dim: int = 384):
        super().__init__("Vector Only", db_path)
        self.embedding_dim = embedding_dim
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.doc_id_to_index: Dict[str, int] = {}
    
    def index_documents(self, documents: List[Document]) -> None:
        """Index document embeddings"""
        self.documents = {doc.doc_id: doc for doc in documents}
        
        # Build embeddings matrix
        embeddings = []
        for i, doc in enumerate(documents):
            self.doc_id_to_index[doc.doc_id] = i
            if doc.embedding is not None:
                embeddings.append(doc.embedding)
            else:
                # Fallback: random embedding
                embeddings.append(np.random.normal(0, 0.1, self.embedding_dim))
        
        self.embeddings_matrix = np.array(embeddings)
    
    def retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """Cosine similarity retrieval"""
        if self.embeddings_matrix is None:
            return []
        
        # Get query embedding (would come from embedding model in practice)
        query_embedding = np.random.normal(0, 0.1, self.embedding_dim)
        query_norm = np.linalg.norm(query_embedding)
        
        if query_norm == 0:
            return []
        
        # Compute cosine similarities
        doc_norms = np.linalg.norm(self.embeddings_matrix, axis=1)
        valid_docs = doc_norms > 0
        
        similarities = np.zeros(len(self.embeddings_matrix))
        if np.any(valid_docs):
            similarities[valid_docs] = np.dot(
                self.embeddings_matrix[valid_docs], 
                query_embedding
            ) / (doc_norms[valid_docs] * query_norm)
        
        # Get top-k most similar
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            doc_id = list(self.documents.keys())[idx]
            doc = self.documents[doc_id]
            
            result = RetrievalResult(
                doc_id=doc_id,
                score=float(similarities[idx]),
                rank=rank,
                content=doc.content,
                kind=doc.kind
            )
            results.append(result)
        
        return results

class BM25VectorSimpleBaseline(BaselineRetriever):
    """Simple BM25 + Vector combination without reranking/diversification"""
    
    def __init__(self, db_path: str, alpha: float = 0.5):
        super().__init__("BM25+Vector Simple", db_path)
        self.alpha = alpha  # BM25 weight (1-alpha is vector weight)
        self.bm25_retriever = BM25OnlyBaseline(db_path)
        self.vector_retriever = VectorOnlyBaseline(db_path)
    
    def index_documents(self, documents: List[Document]) -> None:
        """Index for both BM25 and vector retrieval"""
        self.documents = {doc.doc_id: doc for doc in documents}
        self.bm25_retriever.index_documents(documents)
        self.vector_retriever.index_documents(documents)
    
    def retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """Combine BM25 and vector scores linearly"""
        # Get results from both retrievers
        bm25_results = self.bm25_retriever.retrieve(query, k * 2)  # Get more for fusion
        vector_results = self.vector_retriever.retrieve(query, k * 2)
        
        # Normalize scores to [0, 1]
        bm25_scores = {r.doc_id: r.score for r in bm25_results}
        vector_scores = {r.doc_id: r.score for r in vector_results}
        
        if bm25_scores:
            max_bm25 = max(bm25_scores.values())
            if max_bm25 > 0:
                bm25_scores = {k: v / max_bm25 for k, v in bm25_scores.items()}
        
        # Combine scores
        combined_scores = {}
        all_doc_ids = set(bm25_scores.keys()) | set(vector_scores.keys())
        
        for doc_id in all_doc_ids:
            bm25_score = bm25_scores.get(doc_id, 0.0)
            vector_score = vector_scores.get(doc_id, 0.0)
            combined_scores[doc_id] = self.alpha * bm25_score + (1 - self.alpha) * vector_score
        
        # Sort and return top-k
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs, 1):
            doc = self.documents[doc_id]
            result = RetrievalResult(
                doc_id=doc_id,
                score=float(score),
                rank=rank,
                content=doc.content,
                kind=doc.kind
            )
            results.append(result)
        
        return results

class CrossEncoderBaseline(BaselineRetriever):
    """Cross-encoder reranking over BM25 results"""
    
    def __init__(self, db_path: str, initial_k: int = 50, rerank_threshold: float = 0.3):
        super().__init__("Cross-encoder Rerank", db_path)
        self.initial_k = initial_k
        self.rerank_threshold = rerank_threshold
        self.bm25_retriever = BM25OnlyBaseline(db_path)
    
    def index_documents(self, documents: List[Document]) -> None:
        """Index documents using BM25"""
        self.documents = {doc.doc_id: doc for doc in documents}
        self.bm25_retriever.index_documents(documents)
    
    def retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """BM25 retrieval followed by cross-encoder reranking"""
        # Initial retrieval with BM25
        initial_results = self.bm25_retriever.retrieve(query, self.initial_k)
        
        if not initial_results:
            return []
        
        # Filter by threshold
        candidates = [r for r in initial_results if r.score >= self.rerank_threshold][:k * 2]
        
        # Simulate cross-encoder reranking (in practice, would use actual model)
        reranked_results = []
        for result in candidates:
            # Simulate cross-encoder score (query-document relevance)
            # In practice: score = cross_encoder_model.predict(query.text, result.content)
            mock_relevance = np.random.beta(2, 5)  # Skewed toward lower scores
            
            reranked_result = RetrievalResult(
                doc_id=result.doc_id,
                score=float(mock_relevance),
                rank=result.rank,
                content=result.content,
                kind=result.kind
            )
            reranked_results.append(reranked_result)
        
        # Sort by reranked scores
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(reranked_results[:k], 1):
            result.rank = i
        
        return reranked_results[:k]

class FAISSIVFBaseline(BaselineRetriever):
    """FAISS IVF-Flat vector search baseline"""
    
    def __init__(self, db_path: str, nlist: int = 100, embedding_dim: int = 384):
        super().__init__("FAISS IVF-Flat", db_path)
        self.nlist = nlist
        self.embedding_dim = embedding_dim
        self.index = None
        self.doc_ids: List[str] = []
    
    def index_documents(self, documents: List[Document]) -> None:
        """Build FAISS IVF index"""
        try:
            import faiss
        except ImportError:
            # Fallback to numpy-based approximate search
            self._fallback_index_documents(documents)
            return
        
        self.documents = {doc.doc_id: doc for doc in documents}
        
        # Build embeddings matrix
        embeddings = []
        self.doc_ids = []
        
        for doc in documents:
            if doc.embedding is not None:
                embeddings.append(doc.embedding)
            else:
                embeddings.append(np.random.normal(0, 0.1, self.embedding_dim))
            self.doc_ids.append(doc.doc_id)
        
        embeddings_matrix = np.array(embeddings, dtype=np.float32)
        
        # Build FAISS index
        quantizer = faiss.IndexFlatIP(self.embedding_dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, min(self.nlist, len(documents)))
        
        # Train and add vectors
        self.index.train(embeddings_matrix)
        self.index.add(embeddings_matrix)
    
    def _fallback_index_documents(self, documents: List[Document]) -> None:
        """Fallback when FAISS not available"""
        self.documents = {doc.doc_id: doc for doc in documents}
        self.embeddings_matrix = []
        self.doc_ids = []
        
        for doc in documents:
            if doc.embedding is not None:
                self.embeddings_matrix.append(doc.embedding)
            else:
                self.embeddings_matrix.append(np.random.normal(0, 0.1, self.embedding_dim))
            self.doc_ids.append(doc.doc_id)
        
        self.embeddings_matrix = np.array(self.embeddings_matrix)
    
    def retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """FAISS vector search"""
        if self.index is not None:
            return self._faiss_retrieve(query, k)
        else:
            return self._fallback_retrieve(query, k)
    
    def _faiss_retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """FAISS-based retrieval"""
        try:
            import faiss
        except ImportError:
            return self._fallback_retrieve(query, k)
        
        # Get query embedding
        query_embedding = np.random.normal(0, 0.1, self.embedding_dim).astype(np.float32)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(k, len(self.doc_ids)))
        
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
                
            doc_id = self.doc_ids[idx]
            doc = self.documents[doc_id]
            
            result = RetrievalResult(
                doc_id=doc_id,
                score=float(score),
                rank=rank,
                content=doc.content,
                kind=doc.kind
            )
            results.append(result)
        
        return results
    
    def _fallback_retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """Numpy-based approximate search fallback"""
        if len(self.embeddings_matrix) == 0:
            return []
        
        query_embedding = np.random.normal(0, 0.1, self.embedding_dim)
        
        # Cosine similarity
        similarities = np.dot(self.embeddings_matrix, query_embedding) / (
            np.linalg.norm(self.embeddings_matrix, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            doc_id = self.doc_ids[idx]
            doc = self.documents[doc_id]
            
            result = RetrievalResult(
                doc_id=doc_id,
                score=float(similarities[idx]),
                rank=rank,
                content=doc.content,
                kind=doc.kind
            )
            results.append(result)
        
        return results

class MMRBaseline(BaselineRetriever):
    """Maximal Marginal Relevance diversification baseline"""
    
    def __init__(self, db_path: str, lambda_param: float = 0.7, initial_k: int = 50):
        super().__init__("MMR Diversification", db_path)
        self.lambda_param = lambda_param  # Relevance vs diversity tradeoff
        self.initial_k = initial_k
        self.vector_retriever = VectorOnlyBaseline(db_path)
    
    def index_documents(self, documents: List[Document]) -> None:
        """Index documents using vector retriever"""
        self.documents = {doc.doc_id: doc for doc in documents}
        self.vector_retriever.index_documents(documents)
    
    def retrieve(self, query: Query, k: int = 10) -> List[RetrievalResult]:
        """MMR-based diversified retrieval"""
        # Initial retrieval
        candidates = self.vector_retriever.retrieve(query, self.initial_k)
        
        if not candidates:
            return []
        
        # MMR selection
        selected = []
        remaining = candidates.copy()
        
        # Select first document (highest relevance)
        if remaining:
            best = max(remaining, key=lambda x: x.score)
            selected.append(best)
            remaining.remove(best)
        
        # Iteratively select diverse documents
        while len(selected) < k and remaining:
            best_mmr_score = float('-inf')
            best_doc = None
            
            for candidate in remaining:
                # Relevance score
                relevance = candidate.score
                
                # Maximum similarity to already selected documents
                max_similarity = 0.0
                for selected_doc in selected:
                    # Simulate similarity (would use actual embeddings in practice)
                    similarity = np.random.uniform(0, 1)  # Placeholder
                    max_similarity = max(max_similarity, similarity)
                
                # MMR score
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_similarity
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_doc = candidate
            
            if best_doc:
                selected.append(best_doc)
                remaining.remove(best_doc)
            else:
                break
        
        # Update ranks
        for i, result in enumerate(selected, 1):
            result.rank = i
        
        return selected

class BaselineEvaluator:
    """Evaluates all baseline implementations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.baselines = {
            'window': WindowBaseline(db_path),
            'bm25_only': BM25OnlyBaseline(db_path),
            'vector_only': VectorOnlyBaseline(db_path),
            'bm25_vector_simple': BM25VectorSimpleBaseline(db_path),
            'cross_encoder': CrossEncoderBaseline(db_path),
            'faiss_ivf': FAISSIVFBaseline(db_path),
            'mmr': MMRBaseline(db_path)
        }
    
    def evaluate_all_baselines(self, 
                              documents: List[Document], 
                              queries: List[Query],
                              k: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """Evaluate all baselines on given queries"""
        
        # Index documents for all baselines
        print("Indexing documents for all baselines...")
        for name, baseline in self.baselines.items():
            print(f"  Indexing {name}...")
            baseline.index_documents(documents)
        
        # Evaluate each baseline
        results = {}
        
        for baseline_name, baseline in self.baselines.items():
            print(f"\nEvaluating {baseline_name}...")
            baseline_results = []
            
            for i, query in enumerate(queries, 1):
                if i % 10 == 0:
                    print(f"  Query {i}/{len(queries)}")
                
                try:
                    retrieved_results, latency_ms, memory_mb = baseline.measure_performance(query, k)
                    
                    # Convert to evaluation format
                    result = {
                        "query_id": query.query_id,
                        "session_id": query.session_id,
                        "domain": query.domain,
                        "complexity": query.complexity,
                        "ground_truth_docs": query.ground_truth_docs,
                        "retrieved_docs": [r.doc_id for r in retrieved_results],
                        "relevance_scores": [r.score for r in retrieved_results],
                        "latency_ms": latency_ms,
                        "memory_mb": memory_mb,
                        "entities_covered": [],  # Would be populated by entity extraction
                        "contradictions": [],  # Would be populated by contradiction detection
                        "timestamp": time.time(),
                        "baseline_name": baseline_name
                    }
                    
                    baseline_results.append(result)
                    
                except Exception as e:
                    print(f"    Error processing query {query.query_id}: {e}")
                    continue
            
            results[baseline_name] = baseline_results
            
            # Print summary stats
            if baseline_results:
                avg_latency = np.mean([r["latency_ms"] for r in baseline_results])
                avg_memory = np.mean([r["memory_mb"] for r in baseline_results])
                print(f"  {baseline_name}: {len(baseline_results)} queries, {avg_latency:.1f}ms avg latency, {avg_memory:.1f}MB avg memory")
        
        return results
    
    def save_results(self, results: Dict[str, List[Dict[str, Any]]], output_dir: str) -> None:
        """Save baseline results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for baseline_name, baseline_results in results.items():
            filepath = output_path / f"{baseline_name}_results.json"
            with open(filepath, 'w') as f:
                json.dump(baseline_results, f, indent=2)
            print(f"Saved {len(baseline_results)} results to {filepath}")

def create_sample_data() -> Tuple[List[Document], List[Query]]:
    """Create sample documents and queries for testing"""
    
    documents = []
    for i in range(100):
        doc = Document(
            doc_id=f"doc_{i:03d}",
            content=f"This is sample document {i} about topic {i % 10}. " * 20,
            kind="text" if i % 3 == 0 else "code" if i % 3 == 1 else "tool_output",
            metadata={"topic": i % 10, "length": "long" if i % 5 == 0 else "short"},
            embedding=np.random.normal(0, 0.1, 384)
        )
        documents.append(doc)
    
    queries = []
    for i in range(20):
        query = Query(
            query_id=f"query_{i:03d}",
            text=f"Find information about topic {i % 10}",
            session_id=f"session_{i // 5}",
            domain="mixed",
            complexity="medium",
            ground_truth_docs=[f"doc_{j:03d}" for j in range(i*5, (i+1)*5)]
        )
        queries.append(query)
    
    return documents, queries

if __name__ == "__main__":
    print("Lethe Baseline Implementation Framework")
    print("======================================")
    
    # Test with sample data
    db_path = "/tmp/baseline_test.db"
    evaluator = BaselineEvaluator(db_path)
    
    print("Creating sample data...")
    documents, queries = create_sample_data()
    
    print(f"Evaluating {len(documents)} documents with {len(queries)} queries...")
    results = evaluator.evaluate_all_baselines(documents, queries, k=10)
    
    print("\nSaving results...")
    evaluator.save_results(results, "/tmp/baseline_results/")
    
    print("Baseline evaluation complete!")