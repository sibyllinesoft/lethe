"""
StreamingLLM Context Window Extension Benchmark.

StreamingLLM enables LLMs to work with infinite-length inputs by maintaining
attention sinks and using sliding window attention mechanisms.

Paper: "Efficient Streaming Language Models with Attention Sinks"
GitHub: https://github.com/mit-han-lab/streaming-llm
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


class StreamingLLMCompetitor(ContextManagementCompetitor):
    """StreamingLLM sliding window attention competitor."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("StreamingLLM", config)
        self.model_name = config.get("model_name", "gemma2:9b") if config else "gemma2:9b"
        self.ollama_url = config.get("ollama_url", "http://localhost:11434") if config else "http://localhost:11434"
        self.window_size = config.get("window_size", 2048) if config else 2048
        self.attention_sink_size = config.get("attention_sink_size", 4) if config else 4
    
    def initialize(self) -> bool:
        """Initialize StreamingLLM with Ollama model."""
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
            logger.info(f"StreamingLLM initialized with model={self.model_name}, window_size={self.window_size}, attention_sink_size={self.attention_sink_size}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize StreamingLLM: {e}")
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
        """Process context using StreamingLLM sliding window attention."""
        if not self._initialized:
            raise RuntimeError("StreamingLLM not initialized")
        
        start_time = time.time()
        original_tokens = len(context.split())
        
        try:
            # Tokenize full context + query
            full_text = context + "\n\nQuery: " + query + "\nAnswer:"
            full_tokens = self._tokenize_text(full_text)
            
            # Apply StreamingLLM windowing strategy
            if len(full_tokens) > self.window_size:
                # StreamingLLM: Keep attention sinks + sliding window
                attention_sinks = full_tokens[:self.attention_sink_size]  # First few tokens as attention sinks
                
                # Calculate sliding window size
                available_window = self.window_size - self.attention_sink_size
                
                # Keep most recent tokens within the sliding window
                recent_start = max(self.attention_sink_size, len(full_tokens) - available_window)
                sliding_window = full_tokens[recent_start:]
                
                # Combine attention sinks + sliding window
                if recent_start > self.attention_sink_size:
                    # Gap exists between sinks and window - this is the key StreamingLLM insight
                    streamed_tokens = attention_sinks + sliding_window
                else:
                    # No gap, just truncate normally
                    streamed_tokens = full_tokens[:self.window_size]
                
                processed_text = " ".join(streamed_tokens)
            else:
                processed_text = full_text
            
            # Generate response using Ollama API
            response = self._generate_with_ollama(processed_text, 100)
            
            processing_time = (time.time() - start_time) * 1000
            processed_tokens = len(processed_text.split())
            
            # Calculate effective compression considering streaming window
            effective_compression = 1.0 - (float(processed_tokens) / float(original_tokens)) if original_tokens > 0 else 1.0
            
            return ContextProcessingResult(
                original_context=context,
                processed_context=processed_text,
                query=query,
                response=response,
                processing_time_ms=processing_time,
                original_token_count=original_tokens,
                processed_token_count=processed_tokens,
                compression_ratio=effective_compression,
                method_name=self.name,
                metadata={
                    "approach": "sliding_window_attention",
                    "model": self.model_name,
                    "ollama_url": self.ollama_url,
                    "window_size": self.window_size,
                    "attention_sink_size": self.attention_sink_size,
                    "streaming_enabled": original_tokens > self.window_size,
                    "effective_window_usage": processed_tokens / self.window_size
                }
            )
            
        except Exception as e:
            logger.error(f"StreamingLLM processing failed: {e}")
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
        """Get StreamingLLM installation requirements."""
        return [
            "requests>=2.20.0"
        ]
    
    def cleanup(self):
        """Clean up StreamingLLM resources."""
        # No resources to clean up for Ollama API client
        pass