"""
H2O (Heavy-Hitter Oracle) Context Pruning Benchmark.

H2O dynamically evicts key-value cache to enable infinite-length inputs 
while maintaining conversation ability via attention mechanisms.

Paper: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models"
GitHub: https://github.com/FMInference/H2O
"""

import time
import logging
from typing import Dict, Any, List
import json
import requests

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from competitor_interface import ContextManagementCompetitor, ContextProcessingResult

logger = logging.getLogger(__name__)


class H2OCompetitor(ContextManagementCompetitor):
    """H2O attention-based context pruning competitor."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("H2O", config)
        self.model_name = config.get("model_name", "gemma2:9b") if config else "gemma2:9b"
        self.ollama_url = config.get("ollama_url", "http://localhost:11434") if config else "http://localhost:11434"
    
    def initialize(self) -> bool:
        """Initialize H2O with Ollama model."""
        try:
            # Test connection to Ollama
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
            
            self._initialized = True
            logger.info(f"H2O initialized with model: {self.model_name}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize H2O: {e}")
            return False
    
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
        """Process context using H2O attention-based pruning."""
        if not self._initialized:
            raise RuntimeError("H2O not initialized")
        
        start_time = time.time()
        original_tokens = len(context.split())
        
        try:
            # Tokenize full context + query
            full_text = context + "\n\nQuery: " + query + "\nAnswer:"
            context_tokens = self._tokenize_text(context)
            full_tokens = self._tokenize_text(full_text)
            
            # Simulate H2O heavy-hitter eviction policy
            # In practice, this would analyze attention patterns during inference
            if len(full_tokens) > max_tokens:
                # Keep most recent tokens + estimated "heavy hitters" (key information)
                recent_tokens = max_tokens // 4  # Keep 25% recent
                heavy_hitter_tokens = max_tokens - recent_tokens  # 75% for important content
                
                # Simple heuristic: keep tokens around query-relevant keywords
                query_keywords = set(query.lower().split())
                
                # Score tokens based on query relevance (simplified heavy-hitter detection)
                token_scores = []
                for i, token in enumerate(context_tokens):
                    score = 0
                    for keyword in query_keywords:
                        if keyword in token.lower():
                            score += 10  # High relevance
                        if i < len(context_tokens) * 0.1 or i > len(context_tokens) * 0.9:
                            score += 1  # Positional bias (beginning/end important)
                    token_scores.append((i, score, token))
                
                # Select heavy hitters + recent context
                heavy_hitters = sorted(token_scores, key=lambda x: x[1], reverse=True)[:heavy_hitter_tokens]
                recent_context = context_tokens[-recent_tokens:] if len(context_tokens) > recent_tokens else context_tokens
                
                # Combine and create pruned input
                selected_tokens = [token for _, _, token in heavy_hitters]
                pruned_tokens = selected_tokens + recent_context
                
                # Re-create input with query
                pruned_context = " ".join(pruned_tokens)
                pruned_text = pruned_context + "\n\nQuery: " + query + "\nAnswer:"
            else:
                pruned_text = full_text
                pruned_context = context
            
            # Generate response using Ollama API
            response = self._generate_with_ollama(pruned_text, 100)
            
            processing_time = (time.time() - start_time) * 1000
            processed_tokens = len(pruned_context.split())
            
            return ContextProcessingResult(
                original_context=context,
                processed_context=pruned_context,
                query=query,
                response=response,
                processing_time_ms=processing_time,
                original_token_count=original_tokens,
                processed_token_count=processed_tokens,
                compression_ratio=1.0 - (float(processed_tokens) / float(original_tokens)) if original_tokens > 0 else 1.0,
                method_name=self.name,
                metadata={
                    "approach": "attention_pruning",
                    "model": self.model_name,
                    "ollama_url": self.ollama_url,
                    "max_length": max_tokens,
                    "heavy_hitter_ratio": 0.75,
                    "recent_context_ratio": 0.25
                }
            )
            
        except Exception as e:
            logger.error(f"H2O processing failed: {e}")
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
        """Get H2O installation requirements."""
        return [
            "requests>=2.20.0"
        ]
    
    def cleanup(self):
        """Clean up H2O resources."""
        # No resources to clean up for Ollama API client
        pass