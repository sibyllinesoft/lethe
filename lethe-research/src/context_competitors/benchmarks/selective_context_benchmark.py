"""
Selective Context Pruning Benchmark.

Selective Context uses learned importance scoring to select the most relevant
context segments while maintaining task performance.

Paper: "Selective Context: On How to Train and Deploy Long Context LLMs"
Approach: Sentence-level importance scoring with efficient content selection
"""

import time
import logging
from typing import Dict, Any, List
import re
import json
import requests

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from competitor_interface import ContextManagementCompetitor, ContextProcessingResult

logger = logging.getLogger(__name__)


class SelectiveContextCompetitor(ContextManagementCompetitor):
    """Selective Context importance-based pruning competitor."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("SelectiveContext", config)
        self.model_name = config.get("model_name", "gemma2:9b") if config else "gemma2:9b"
        self.ollama_url = config.get("ollama_url", "http://localhost:11434") if config else "http://localhost:11434"
        self.sentence_model = None
    
    def initialize(self) -> bool:
        """Initialize Selective Context with sentence transformer and Ollama LLM."""
        try:
            # Test connection to Ollama first
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code != 200:
                logger.error(f"Cannot connect to Ollama at {self.ollama_url}")
                return False
            
            # Check if model is available
            models = response.json().get('models', [])
            available_models = [model['name'] for model in models]
            
            if self.model_name not in available_models:
                logger.error(f"Model {self.model_name} not found in Ollama. Available: {available_models}")
                return False
            
            # Initialize sentence transformer for relevance scoring (optional)
            try:
                from sentence_transformers import SentenceTransformer
                sentence_model_name = self.config.get("sentence_model", "all-MiniLM-L6-v2") if self.config else "all-MiniLM-L6-v2"
                self.sentence_model = SentenceTransformer(sentence_model_name)
                logger.info(f"SelectiveContext initialized with sentence model: {sentence_model_name}")
            except ImportError:
                logger.warning("sentence-transformers not available, using keyword-based relevance scoring")
                self.sentence_model = None
            
            self._initialized = True
            logger.info(f"SelectiveContext initialized with model: {self.model_name}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize SelectiveContext: {e}")
            return False
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for selective processing."""
        # Simple sentence splitting - could be enhanced with spaCy/NLTK
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _score_sentence_relevance(self, sentences: List[str], query: str) -> List[float]:
        """Score sentence relevance using sentence transformers."""
        try:
            # Encode query and sentences
            query_embedding = self.sentence_model.encode([query])
            sentence_embeddings = self.sentence_model.encode(sentences)
            
            # Calculate cosine similarity scores
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]
            
            return similarities.tolist()
            
        except ImportError:
            logger.warning("scikit-learn not available, using simple keyword matching")
            # Fallback to simple keyword matching
            query_words = set(query.lower().split())
            scores = []
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words.intersection(sentence_words))
                scores.append(overlap / max(len(query_words), 1))
            return scores
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Simple word-based tokenization as approximation."""
        # Split on whitespace and punctuation
        import re
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
    
    def _generate_with_ollama(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate response using Ollama API."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": max_new_tokens
            }
        }
        
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            return f"Error: {str(e)}"
    
    def process_context(self, query: str, context: str, max_tokens: int = 4000) -> ContextProcessingResult:
        """Process context using selective importance-based pruning."""
        if not self._initialized:
            raise RuntimeError("SelectiveContext not initialized")
        
        start_time = time.time()
        original_tokens = len(context.split())
        
        try:
            # Split context into sentences
            sentences = self._split_into_sentences(context)
            if not sentences:
                sentences = [context]  # Fallback if no sentence splits
            
            # Score sentence relevance to query
            relevance_scores = self._score_sentence_relevance(sentences, query)
            
            # Sort sentences by relevance score
            sentence_score_pairs = list(zip(sentences, relevance_scores))
            sentence_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Select top sentences within token budget
            selected_sentences = []
            current_tokens = 0
            query_tokens = len(self._tokenize_text("Query: " + query + "\nAnswer:"))
            available_tokens = max_tokens - query_tokens - 50  # Reserve tokens for response generation
            
            for sentence, score in sentence_score_pairs:
                sentence_tokens = len(self._tokenize_text(sentence))
                if current_tokens + sentence_tokens <= available_tokens:
                    selected_sentences.append((sentence, score))
                    current_tokens += sentence_tokens
                else:
                    break
            
            # Sort selected sentences back to original order for coherence
            sentence_positions = {sentence: i for i, sentence in enumerate(sentences)}
            selected_sentences.sort(key=lambda x: sentence_positions.get(x[0], 0))
            
            # Create processed context
            processed_context = ". ".join([sentence for sentence, _ in selected_sentences])
            full_text = processed_context + "\n\nQuery: " + query + "\nAnswer:"
            
            # Generate response using Ollama API
            response = self._generate_with_ollama(full_text, 100)
            
            processing_time = (time.time() - start_time) * 1000
            processed_tokens = len(processed_context.split())
            
            # Calculate metrics
            sentences_kept = len(selected_sentences)
            sentences_total = len(sentences)
            avg_relevance_score = sum(score for _, score in selected_sentences) / max(len(selected_sentences), 1)
            
            return ContextProcessingResult(
                original_context=context,
                processed_context=processed_context,
                query=query,
                response=response,
                processing_time_ms=processing_time,
                original_token_count=original_tokens,
                processed_token_count=processed_tokens,
                compression_ratio=1.0 - (float(processed_tokens) / float(original_tokens)) if original_tokens > 0 else 1.0,
                method_name=self.name,
                metadata={
                    "approach": "selective_context_pruning",
                    "sentence_model": self.sentence_model.__class__.__name__ if self.sentence_model else "keyword_based",
                    "model": self.model_name,
                    "ollama_url": self.ollama_url,
                    "sentences_total": sentences_total,
                    "sentences_kept": sentences_kept,
                    "sentence_retention_ratio": sentences_kept / max(sentences_total, 1),
                    "avg_relevance_score": avg_relevance_score,
                    "selection_strategy": "relevance_based",
                    "max_context_tokens": available_tokens
                }
            )
            
        except Exception as e:
            logger.error(f"SelectiveContext processing failed: {e}")
            return ContextProcessingResult(
                original_context=context,
                processed_context="",
                query=query,
                response=f"Error: {str(e)}",
                processing_time_ms=(time.time() - start_time) * 1000,
                original_token_count=original_tokens,
                processed_token_count=0,
                compression_ratio=1.0,  # All context was filtered out = 100% compression
                method_name=self.name,
                metadata={"error": str(e)}
            )
    
    def get_installation_requirements(self) -> List[str]:
        """Get SelectiveContext installation requirements."""
        return [
            "requests>=2.20.0",
            "sentence-transformers>=2.0.0",  # Optional for better relevance scoring
            "scikit-learn>=1.0.0"  # Optional for cosine similarity
        ]
    
    def cleanup(self):
        """Clean up SelectiveContext resources."""
        if self.sentence_model is not None:
            del self.sentence_model
            self.sentence_model = None
        # No other resources to clean up for Ollama API client