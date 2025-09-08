"""
Baseline Methods for InfinityBench Evaluation
Standard baselines for academic comparison with ranked result support.
"""

import random
from typing import List, Dict, Any, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
import logging

logger = logging.getLogger(__name__)

class BM25Baseline:
    """BM25 baseline for information retrieval."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75, chunk_size: int = 512, 
                 chunk_overlap: int = 50):
        self.k1 = k1
        self.b = b
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            if chunk_words:
                chunks.append(' '.join(chunk_words))
                
        return chunks if chunks else [text]
    
    def retrieve_ranked_results(self, context: str, question: str, 
                              max_results: int = 100) -> List[Tuple[str, float]]:
        """Retrieve and rank all chunks with scores."""
        # Chunk the context
        chunks = self.chunk_text(context)
        
        # Tokenize chunks for BM25
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        
        # Build BM25 index
        bm25 = BM25Okapi(tokenized_chunks)
        
        # Query tokenization
        query_tokens = question.lower().split()
        
        # Get scores for all chunks
        scores = bm25.get_scores(query_tokens)
        
        # Create ranked results with scores
        ranked_results = [(chunks[i], scores[i]) for i in range(len(chunks))]
        
        # Sort by score descending
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        return ranked_results[:max_results]
        
    def retrieve_and_answer(self, context: str, question: str, top_k: int = 5) -> str:
        """Retrieve relevant chunks and generate answer."""
        # Get ranked results
        ranked_results = self.retrieve_ranked_results(context, question, top_k)
        
        # Extract top chunks for answer generation
        retrieved_chunks = [chunk for chunk, score in ranked_results if score > 0]
        
        if not retrieved_chunks:
            # Fallback to first chunk if no matches
            chunks = self.chunk_text(context)
            retrieved_chunks = chunks[:1]
            
        # Combine top chunks as answer context
        combined_context = ' '.join(retrieved_chunks)
        
        # Simple heuristic: look for answer patterns
        return self._extract_answer(combined_context, question)
        
    def _extract_answer(self, context: str, question: str) -> str:
        """Simple answer extraction heuristic."""
        # This is a placeholder - in practice you'd use an LM
        sentences = context.split('.')
        
        # Return first sentence that contains question keywords
        question_words = set(question.lower().split())
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            if len(question_words & sentence_words) >= 2:
                return sentence.strip()
                
        # Fallback to first sentence
        return sentences[0].strip() if sentences else "No answer found"

class NaiveChunkingBaseline:
    """Naive chunking baseline with different strategies."""
    
    def __init__(self, chunk_size: int = 1024, max_chunks: int = 10):
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
        
    def chunk_text(self, text: str, strategy: str = "uniform") -> List[str]:
        """Chunk text using specified strategy."""
        words = text.split()
        
        if strategy == "uniform":
            # Uniform chunking
            chunks = []
            for i in range(0, len(words), self.chunk_size):
                chunk_words = words[i:i + self.chunk_size]
                if chunk_words:
                    chunks.append(' '.join(chunk_words))
            return chunks[:self.max_chunks]
            
        elif strategy == "first":
            # Take first N chunks
            chunk = ' '.join(words[:self.chunk_size * self.max_chunks])
            return [chunk]
            
        elif strategy == "random":
            # Random sampling of chunks
            if len(words) <= self.chunk_size:
                return [text]
                
            num_possible_chunks = len(words) // self.chunk_size
            selected_chunks = min(self.max_chunks, num_possible_chunks)
            
            chunk_indices = random.sample(range(num_possible_chunks), selected_chunks)
            chunks = []
            
            for idx in sorted(chunk_indices):
                start = idx * self.chunk_size
                end = start + self.chunk_size
                chunk_words = words[start:end]
                chunks.append(' '.join(chunk_words))
                
            return chunks
            
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def retrieve_ranked_results(self, context: str, question: str, 
                              strategy: str = "uniform", 
                              max_results: int = 100) -> List[Tuple[str, float]]:
        """Retrieve and rank chunks with simple similarity scores."""
        chunks = self.chunk_text(context, strategy)
        
        # Simple scoring based on keyword overlap
        question_words = set(question.lower().split())
        
        ranked_results = []
        for chunk in chunks:
            chunk_words = set(chunk.lower().split())
            
            # Simple Jaccard similarity as score
            intersection = len(question_words & chunk_words)
            union = len(question_words | chunk_words)
            
            if union > 0:
                score = intersection / union
            else:
                score = 0.0
            
            # Add some randomness for uniform baseline variety
            if strategy == "random":
                score += random.uniform(0, 0.1)
            elif strategy == "first":
                # Earlier chunks get slightly higher scores
                position_bonus = (len(chunks) - chunks.index(chunk)) / len(chunks) * 0.1
                score += position_bonus
                
            ranked_results.append((chunk, score))
        
        # Sort by score descending
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_results[:max_results]
            
    def retrieve_and_answer(self, context: str, question: str, strategy: str = "uniform") -> str:
        """Retrieve using naive chunking and return simple answer."""
        chunks = self.chunk_text(context, strategy)
        
        # Simple heuristic: return first chunk
        if chunks:
            return self._extract_answer(chunks[0], question)
        else:
            return "No answer found"
            
    def _extract_answer(self, context: str, question: str) -> str:
        """Simple answer extraction heuristic."""
        # Similar to BM25 baseline
        sentences = context.split('.')
        question_words = set(question.lower().split())
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            if len(question_words & sentence_words) >= 1:
                return sentence.strip()
                
        return sentences[0].strip() if sentences else "No answer found"

def evaluate_relevance(prediction: str, reference: str, threshold: float = 0.3) -> bool:
    """
    Determine if a prediction is relevant to the reference answer.
    
    Uses F1 score as relevance indicator with configurable threshold.
    """
    from .metrics import f1_score
    
    f1 = f1_score(prediction, reference)
    return f1 >= threshold

def run_baseline_evaluation(baseline, samples: List[Dict], baseline_name: str) -> List[str]:
    """Run baseline evaluation on samples."""
    logger.info(f"Running {baseline_name} baseline on {len(samples)} samples")
    
    predictions = []
    
    for i, sample in enumerate(samples):
        try:
            if hasattr(baseline, 'retrieve_and_answer'):
                prediction = baseline.retrieve_and_answer(
                    sample['context'], 
                    sample['question']
                )
            else:
                # Fallback for other baseline types
                prediction = "No answer found"
                
            predictions.append(str(prediction))
            
            if i % 10 == 0:
                logger.debug(f"Processed {i+1}/{len(samples)} samples")
                
        except Exception as e:
            logger.warning(f"Error processing sample {i}: {e}")
            predictions.append("Error")
            
    logger.info(f"Completed {baseline_name} baseline evaluation")
    return predictions

def run_ranked_baseline_evaluation(
    baseline, 
    samples: List[Dict], 
    baseline_name: str,
    max_results: int = 100,
    relevance_threshold: float = 0.3
) -> Tuple[List[str], List[List[Tuple[str, float, bool]]]]:
    """
    Run baseline evaluation with ranked results and relevance assessment.
    
    Returns:
        Tuple of (predictions, ranked_results_with_relevance)
        where ranked_results_with_relevance is a list of (chunk, score, is_relevant) tuples
    """
    logger.info(f"Running ranked {baseline_name} baseline on {len(samples)} samples")
    
    predictions = []
    all_ranked_results = []
    
    for i, sample in enumerate(samples):
        try:
            # Get ranked results if available
            if hasattr(baseline, 'retrieve_ranked_results'):
                ranked_results = baseline.retrieve_ranked_results(
                    sample['context'], 
                    sample['question'],
                    max_results=max_results
                )
                
                # Assess relevance for each result
                ranked_with_relevance = []
                for chunk, score in ranked_results:
                    is_relevant = evaluate_relevance(chunk, sample['answer'], relevance_threshold)
                    ranked_with_relevance.append((chunk, score, is_relevant))
                
                all_ranked_results.append(ranked_with_relevance)
            else:
                # Fallback: create single result
                all_ranked_results.append([("No ranked results available", 0.0, False)])
            
            # Get prediction for traditional evaluation
            if hasattr(baseline, 'retrieve_and_answer'):
                prediction = baseline.retrieve_and_answer(
                    sample['context'], 
                    sample['question']
                )
            else:
                prediction = "No answer found"
                
            predictions.append(str(prediction))
            
            if i % 10 == 0:
                logger.debug(f"Processed {i+1}/{len(samples)} samples")
                
        except Exception as e:
            logger.warning(f"Error processing sample {i}: {e}")
            predictions.append("Error")
            all_ranked_results.append([("Error", 0.0, False)])
            
    logger.info(f"Completed ranked {baseline_name} baseline evaluation")
    return predictions, all_ranked_results