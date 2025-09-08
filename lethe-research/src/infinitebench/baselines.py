"""
InfiniteBench Baseline Implementations
====================================

Comprehensive baseline methods for comparison with Lethe on InfiniteBench tasks.
Includes BM25, naive chunking, dense retrieval, and other standard approaches
for long-context retrieval evaluation.

Author: Lethe Research Team
"""

import re
import math
import random
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np
import tiktoken
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Result from a retrieval baseline method."""
    
    query_id: Union[int, str]
    retrieved_chunks: List[Tuple[str, float]]  # (chunk_text, score) 
    context_used: str
    processing_time_ms: float
    metadata: Dict[str, Any]

class BaselineMethod(ABC):
    """Abstract base class for baseline retrieval methods."""
    
    def __init__(self, name: str):
        self.name = name
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    @abstractmethod
    def retrieve(self, query: str, context: str, max_tokens: int = 4000) -> RetrievalResult:
        """Retrieve relevant information for a query from long context."""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to maximum token count."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)

class BM25Baseline(BaselineMethod):
    """
    BM25 baseline for long-context retrieval.
    
    Chunks the long context and uses BM25 scoring to retrieve 
    the most relevant chunks for answering the query.
    """
    
    def __init__(self, 
                 chunk_size: int = 512,
                 chunk_overlap: int = 50,
                 k1: float = 1.2,
                 b: float = 0.75):
        """
        Initialize BM25 baseline.
        
        Args:
            chunk_size: Size of each chunk in tokens
            chunk_overlap: Overlap between chunks in tokens
            k1: BM25 parameter k1 (term frequency saturation)
            b: BM25 parameter b (length normalization)
        """
        super().__init__("BM25")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k1 = k1
        self.b = b
    
    def retrieve(self, query: str, context: str, max_tokens: int = 4000) -> RetrievalResult:
        """Retrieve using BM25 scoring."""
        import time
        start_time = time.time()
        
        # Chunk the context
        chunks = self._chunk_text(context)
        
        if not chunks:
            return RetrievalResult(
                query_id="",
                retrieved_chunks=[],
                context_used="",
                processing_time_ms=(time.time() - start_time) * 1000,
                metadata={"num_chunks": 0, "method": "BM25"}
            )
        
        # Calculate BM25 scores for each chunk
        query_terms = self._tokenize(query)
        scored_chunks = []
        
        # Precompute document statistics
        doc_lengths = [len(self._tokenize(chunk)) for chunk in chunks]
        avg_doc_length = sum(doc_lengths) / len(doc_lengths)
        term_doc_freq = self._compute_term_doc_frequencies(chunks)
        
        for i, chunk in enumerate(chunks):
            score = self._bm25_score(query_terms, chunk, doc_lengths[i], 
                                   avg_doc_length, term_doc_freq, len(chunks))
            scored_chunks.append((chunk, score))
        
        # Sort by score and select top chunks within token limit
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        selected_chunks = []
        total_tokens = 0
        
        for chunk, score in scored_chunks:
            chunk_tokens = self.count_tokens(chunk)
            if total_tokens + chunk_tokens <= max_tokens:
                selected_chunks.append((chunk, score))
                total_tokens += chunk_tokens
            else:
                break
        
        # Combine selected chunks
        context_used = "\n\n".join([chunk for chunk, _ in selected_chunks])
        
        processing_time = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            query_id="",
            retrieved_chunks=selected_chunks,
            context_used=context_used,
            processing_time_ms=processing_time,
            metadata={
                "num_chunks_total": len(chunks),
                "num_chunks_selected": len(selected_chunks),
                "total_tokens_used": total_tokens,
                "bm25_params": {"k1": self.k1, "b": self.b},
                "method": "BM25"
            }
        )
    
    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text with sliding window approach."""
        tokens = self.encoding.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if end == len(tokens):
                break
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Lowercase and split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
    
    def _compute_term_doc_frequencies(self, chunks: List[str]) -> Dict[str, int]:
        """Compute document frequency for each term."""
        term_doc_freq = defaultdict(int)
        
        for chunk in chunks:
            chunk_terms = set(self._tokenize(chunk))
            for term in chunk_terms:
                term_doc_freq[term] += 1
        
        return dict(term_doc_freq)
    
    def _bm25_score(self, query_terms: List[str], doc: str, doc_length: int,
                   avg_doc_length: float, term_doc_freq: Dict[str, int], 
                   num_docs: int) -> float:
        """Calculate BM25 score for a document."""
        doc_terms = self._tokenize(doc)
        doc_term_freq = Counter(doc_terms)
        
        score = 0.0
        
        for term in query_terms:
            if term not in doc_term_freq:
                continue
            
            # Term frequency in document
            tf = doc_term_freq[term]
            
            # Document frequency and IDF
            df = term_doc_freq.get(term, 0)
            if df == 0:
                continue
            
            idf = math.log((num_docs - df + 0.5) / (df + 0.5))
            
            # BM25 formula
            tf_component = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * (doc_length / avg_doc_length))
            )
            
            score += idf * tf_component
        
        return score

class NaiveChunkingBaseline(BaselineMethod):
    """
    Naive chunking baseline that simply takes the first N tokens
    or uses simple heuristics to select chunks.
    """
    
    def __init__(self, 
                 strategy: str = "first",
                 chunk_size: int = 512):
        """
        Initialize naive chunking baseline.
        
        Args:
            strategy: Strategy for selection ("first", "random", "uniform")
            chunk_size: Size of chunks in tokens
        """
        super().__init__(f"NaiveChunking-{strategy}")
        self.strategy = strategy
        self.chunk_size = chunk_size
    
    def retrieve(self, query: str, context: str, max_tokens: int = 4000) -> RetrievalResult:
        """Retrieve using naive chunking strategy."""
        import time
        start_time = time.time()
        
        if self.strategy == "first":
            # Simply take the first max_tokens
            context_used = self.truncate_to_tokens(context, max_tokens)
            chunks = [(context_used, 1.0)]  # Score of 1.0 for selected chunk
            
        elif self.strategy == "random":
            # Split into chunks and randomly select
            all_chunks = self._chunk_text(context)
            selected_chunks = self._select_random_chunks(all_chunks, max_tokens)
            chunks = [(chunk, 1.0/len(selected_chunks)) for chunk in selected_chunks]
            context_used = "\n\n".join(selected_chunks)
            
        elif self.strategy == "uniform":
            # Take chunks uniformly distributed across the context
            all_chunks = self._chunk_text(context)
            selected_chunks = self._select_uniform_chunks(all_chunks, max_tokens)
            chunks = [(chunk, 1.0/len(selected_chunks)) for chunk in selected_chunks]
            context_used = "\n\n".join(selected_chunks)
            
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        processing_time = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            query_id="",
            retrieved_chunks=chunks,
            context_used=context_used,
            processing_time_ms=processing_time,
            metadata={
                "strategy": self.strategy,
                "num_chunks_selected": len(chunks),
                "total_tokens_used": self.count_tokens(context_used),
                "method": "NaiveChunking"
            }
        )
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        tokens = self.encoding.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    def _select_random_chunks(self, chunks: List[str], max_tokens: int) -> List[str]:
        """Randomly select chunks within token limit."""
        random.seed(42)  # For reproducibility
        
        selected = []
        total_tokens = 0
        shuffled_chunks = chunks.copy()
        random.shuffle(shuffled_chunks)
        
        for chunk in shuffled_chunks:
            chunk_tokens = self.count_tokens(chunk)
            if total_tokens + chunk_tokens <= max_tokens:
                selected.append(chunk)
                total_tokens += chunk_tokens
            else:
                break
        
        return selected
    
    def _select_uniform_chunks(self, chunks: List[str], max_tokens: int) -> List[str]:
        """Select chunks uniformly distributed across document."""
        if not chunks:
            return []
        
        selected = []
        total_tokens = 0
        
        # Calculate step size to distribute evenly
        num_chunks_possible = min(len(chunks), max_tokens // self.chunk_size + 1)
        if num_chunks_possible <= 1:
            return chunks[:1]
        
        step = max(1, len(chunks) // num_chunks_possible)
        
        for i in range(0, len(chunks), step):
            chunk = chunks[i]
            chunk_tokens = self.count_tokens(chunk)
            
            if total_tokens + chunk_tokens <= max_tokens:
                selected.append(chunk)
                total_tokens += chunk_tokens
            else:
                break
        
        return selected

class DenseRetrievalBaseline(BaselineMethod):
    """
    Dense retrieval baseline using sentence transformers.
    
    Note: This is a simplified implementation. In practice, you would
    use pre-trained dense retrieval models like DPR, ANCE, etc.
    """
    
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        """
        Initialize dense retrieval baseline.
        
        Args:
            model_name: Name of sentence transformer model
            chunk_size: Size of chunks in tokens
            chunk_overlap: Overlap between chunks
        """
        super().__init__("DenseRetrieval")
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize model (lazy loading)
        self._model = None
    
    def _load_model(self):
        """Lazily load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded sentence transformer: {self.model_name}")
            except ImportError:
                logger.error("sentence-transformers not available. Install with: pip install sentence-transformers")
                raise
            except Exception as e:
                logger.error(f"Failed to load model {self.model_name}: {e}")
                raise
    
    def retrieve(self, query: str, context: str, max_tokens: int = 4000) -> RetrievalResult:
        """Retrieve using dense embeddings and cosine similarity."""
        import time
        start_time = time.time()
        
        self._load_model()
        
        # Chunk the context
        chunks = self._chunk_text(context)
        
        if not chunks:
            return RetrievalResult(
                query_id="",
                retrieved_chunks=[],
                context_used="",
                processing_time_ms=(time.time() - start_time) * 1000,
                metadata={"num_chunks": 0, "method": "DenseRetrieval"}
            )
        
        # Encode query and chunks
        query_embedding = self._model.encode([query], convert_to_tensor=True)
        chunk_embeddings = self._model.encode(chunks, convert_to_tensor=True)
        
        # Calculate cosine similarities
        similarities = self._model.similarity(query_embedding, chunk_embeddings)[0]
        
        # Create scored chunks
        scored_chunks = [(chunks[i], float(similarities[i])) for i in range(len(chunks))]
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Select top chunks within token limit
        selected_chunks = []
        total_tokens = 0
        
        for chunk, score in scored_chunks:
            chunk_tokens = self.count_tokens(chunk)
            if total_tokens + chunk_tokens <= max_tokens:
                selected_chunks.append((chunk, score))
                total_tokens += chunk_tokens
            else:
                break
        
        # Combine selected chunks
        context_used = "\n\n".join([chunk for chunk, _ in selected_chunks])
        
        processing_time = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            query_id="",
            retrieved_chunks=selected_chunks,
            context_used=context_used,
            processing_time_ms=processing_time,
            metadata={
                "model_name": self.model_name,
                "num_chunks_total": len(chunks),
                "num_chunks_selected": len(selected_chunks),
                "total_tokens_used": total_tokens,
                "avg_similarity": np.mean([score for _, score in selected_chunks]),
                "method": "DenseRetrieval"
            }
        )
    
    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text with overlapping windows."""
        tokens = self.encoding.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if end == len(tokens):
                break
            start += self.chunk_size - self.chunk_overlap
        
        return chunks

class GPT4Baseline(BaselineMethod):
    """
    GPT-4 baseline for direct comparison.
    
    This baseline sends the entire context (up to token limit) to GPT-4
    and asks it to answer the question directly.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4-1106-preview",
                 max_context_tokens: int = 120000):
        """
        Initialize GPT-4 baseline.
        
        Args:
            api_key: OpenAI API key
            model: GPT-4 model variant to use
            max_context_tokens: Maximum context tokens to send to GPT-4
        """
        super().__init__(f"GPT4-{model}")
        self.api_key = api_key
        self.model = model
        self.max_context_tokens = max_context_tokens
        
        # Initialize OpenAI client (lazy loading)
        self._client = None
    
    def _load_client(self):
        """Lazily load OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
                logger.info(f"Initialized OpenAI client for {self.model}")
            except ImportError:
                logger.error("openai package not available. Install with: pip install openai")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {e}")
                raise
    
    def retrieve(self, query: str, context: str, max_tokens: int = 4000) -> RetrievalResult:
        """Use GPT-4 to directly answer from context."""
        import time
        start_time = time.time()
        
        self._load_client()
        
        # Truncate context to model limits
        context_truncated = self.truncate_to_tokens(context, self.max_context_tokens - 1000)  # Leave room for prompt
        
        # Create prompt
        prompt = f"""Based on the following context, please answer the question.

Context:
{context_truncated}

Question: {query}

Answer:"""
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.0
            )
            
            answer = response.choices[0].message.content.strip()
            
            processing_time = (time.time() - start_time) * 1000
            
            return RetrievalResult(
                query_id="",
                retrieved_chunks=[(answer, 1.0)],  # The answer itself as a "chunk"
                context_used=context_truncated,
                processing_time_ms=processing_time,
                metadata={
                    "model": self.model,
                    "context_tokens": self.count_tokens(context_truncated),
                    "answer_tokens": self.count_tokens(answer),
                    "api_call_successful": True,
                    "method": "GPT4"
                }
            )
            
        except Exception as e:
            logger.error(f"GPT-4 API call failed: {e}")
            
            processing_time = (time.time() - start_time) * 1000
            
            return RetrievalResult(
                query_id="",
                retrieved_chunks=[("", 0.0)],
                context_used="",
                processing_time_ms=processing_time,
                metadata={
                    "model": self.model,
                    "error": str(e),
                    "api_call_successful": False,
                    "method": "GPT4"
                }
            )

def main():
    """Example usage of baseline methods."""
    
    # Example context and query
    context = """
    The history of artificial intelligence (AI) began in antiquity, with myths, stories and rumors of 
    artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern 
    AI were planted by classical philosophers who attempted to describe the process of human thinking as 
    the mechanical manipulation of symbols. This work culminated in the invention of the programmable 
    digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning.
    
    The field of AI research was born at a Dartmouth College conference in 1956. Attendees Allen Newell, 
    Herbert Simon, John McCarthy, Marvin Minsky and Arthur Samuel became the founders and leaders of AI 
    research. They and their students produced programs that the press described as "astonishing": 
    computers were learning checkers strategies, solving word problems in algebra, proving logical theorems 
    and speaking English.
    
    By the middle of the 1960s, research in the U.S. was heavily funded by the Department of Defense and 
    laboratories had been established around the world. AI's founders were optimistic about the future: 
    Herbert Simon predicted, "machines will be capable, within twenty years, of doing any work a man can do." 
    Marvin Minsky agreed, writing, "within a generation... the problem of creating 'artificial intelligence' 
    will substantially be solved."
    
    However, they had underestimated the difficulty of the problems involved. Both the U.S. and British 
    governments cut off exploratory research in response to the criticism of Sir James Lighthill and ongoing 
    pressure from the U.S. Congress to fund more productive projects. Minsky's and Papert's book Perceptrons 
    was understood to prove that artificial neural networks would never be useful for solving real-world tasks, 
    thus discrediting the approach altogether.
    """
    
    query = "When was the field of AI research born?"
    
    print("Testing InfiniteBench Baselines")
    print("=" * 50)
    
    # Test BM25 baseline
    print("\n1. BM25 Baseline:")
    bm25 = BM25Baseline()
    result = bm25.retrieve(query, context, max_tokens=500)
    print(f"   Processing time: {result.processing_time_ms:.1f}ms")
    print(f"   Chunks selected: {result.metadata['num_chunks_selected']}")
    print(f"   Context preview: {result.context_used[:200]}...")
    
    # Test naive chunking baseline
    print("\n2. Naive Chunking Baseline (first):")
    naive_first = NaiveChunkingBaseline(strategy="first")
    result = naive_first.retrieve(query, context, max_tokens=500)
    print(f"   Processing time: {result.processing_time_ms:.1f}ms")
    print(f"   Tokens used: {result.metadata['total_tokens_used']}")
    print(f"   Context preview: {result.context_used[:200]}...")
    
    # Test dense retrieval baseline (if sentence-transformers available)
    try:
        print("\n3. Dense Retrieval Baseline:")
        dense = DenseRetrievalBaseline()
        result = dense.retrieve(query, context, max_tokens=500)
        print(f"   Processing time: {result.processing_time_ms:.1f}ms")
        print(f"   Chunks selected: {result.metadata['num_chunks_selected']}")
        print(f"   Avg similarity: {result.metadata['avg_similarity']:.3f}")
    except Exception as e:
        print(f"   Dense retrieval not available: {e}")
    
    print("\nBaseline testing complete!")

if __name__ == "__main__":
    main()