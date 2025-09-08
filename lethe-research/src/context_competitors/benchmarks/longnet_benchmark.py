"""
LongNet Hierarchical Attention Benchmark.

LongNet scales sequence length to more than 1 billion tokens using 
dilated attention patterns that approximate hierarchical attention.

Paper: "LongNet: Scaling Transformers to 1,000,000,000 Tokens"
Approach: Dilated attention with exponentially increasing dilation rates
"""

import time
import logging
from typing import Dict, Any, List
import math
import json
import requests

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from competitor_interface import ContextManagementCompetitor, ContextProcessingResult

logger = logging.getLogger(__name__)


class LongNetCompetitor(ContextManagementCompetitor):
    """LongNet dilated attention scaling competitor."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("LongNet", config)
        self.model_name = config.get("model_name", "gemma2:9b") if config else "gemma2:9b"
        self.ollama_url = config.get("ollama_url", "http://localhost:11434") if config else "http://localhost:11434"
        self.segment_len = config.get("segment_len", 512) if config else 512
        self.dilated_ratios = config.get("dilated_ratios", [1, 2, 4, 8]) if config else [1, 2, 4, 8]
    
    def initialize(self) -> bool:
        """Initialize LongNet with Ollama model."""
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
            logger.info(f"LongNet initialized with model={self.model_name}, segment_len={self.segment_len}, dilated_ratios={self.dilated_ratios}")
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize LongNet: {e}")
            return False
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Simple word-based tokenization as approximation."""
        # Split on whitespace and punctuation
        import re
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
    
    def _apply_dilated_attention_sampling(self, context: str, query: str, max_tokens: int) -> str:
        """
        Apply LongNet-style dilated attention sampling to context.
        
        Simulates hierarchical attention by sampling context at different granularities:
        - Layer 1: Full attention within local segments (ratio=1)
        - Layer 2: Dilated attention every 2nd token (ratio=2)  
        - Layer 3: Dilated attention every 4th token (ratio=4)
        - etc.
        """
        context_tokens = self._tokenize_text(context)
        
        if len(context_tokens) <= max_tokens:
            return context  # No dilution needed
        
        # Reserve tokens for query and response
        query_tokens = len(self._tokenize_text("Query: " + query + "\nAnswer:"))
        available_tokens = max_tokens - query_tokens - 50
        
        # Apply hierarchical dilated sampling
        sampled_tokens = []
        
        # Always include some recent context (highest resolution)
        recent_window = min(self.segment_len, available_tokens // 2)
        sampled_tokens.extend(context_tokens[-recent_window:])
        remaining_budget = available_tokens - len(sampled_tokens)
        
        # Apply dilated attention patterns to earlier context
        if remaining_budget > 0:
            earlier_context = context_tokens[:-recent_window] if len(context_tokens) > recent_window else []
            
            for ratio in self.dilated_ratios:
                if remaining_budget <= 0:
                    break
                
                # Sample every ratio-th token from earlier context
                dilated_samples = []
                for i in range(0, len(earlier_context), ratio):
                    if len(dilated_samples) >= remaining_budget:
                        break
                    dilated_samples.append(earlier_context[i])
                
                # Add samples to beginning of our sampled tokens
                tokens_to_add = min(len(dilated_samples), remaining_budget)
                sampled_tokens = dilated_samples[:tokens_to_add] + sampled_tokens
                remaining_budget -= tokens_to_add
                
                # Update earlier context for next dilution level
                if ratio > 1:
                    earlier_context = earlier_context[::ratio][tokens_to_add:]
        
        # Convert back to text
        processed_context = " ".join(sampled_tokens)
        return processed_context
    
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
        """Process context using LongNet dilated attention simulation."""
        if not self._initialized:
            raise RuntimeError("LongNet not initialized")
        
        start_time = time.time()
        original_tokens = len(context.split())
        
        try:
            # Apply dilated attention sampling
            processed_context = self._apply_dilated_attention_sampling(context, query, max_tokens)
            
            # Create full input with query
            full_text = processed_context + "\n\nQuery: " + query + "\nAnswer:"
            
            # Generate response using Ollama API
            response = self._generate_with_ollama(full_text, 100)
            
            processing_time = (time.time() - start_time) * 1000
            processed_tokens = len(processed_context.split())
            
            # Calculate dilated attention metrics
            effective_ratio = max(self.dilated_ratios) if original_tokens > max_tokens else 1
            hierarchical_levels = len([r for r in self.dilated_ratios if r <= (original_tokens / self.segment_len)])
            
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
                    "approach": "dilated_hierarchical_attention",
                    "model": self.model_name,
                    "ollama_url": self.ollama_url,
                    "segment_len": self.segment_len,
                    "dilated_ratios": self.dilated_ratios,
                    "max_dilated_ratio": effective_ratio,
                    "hierarchical_levels_used": hierarchical_levels,
                    "requires_long_sequence": original_tokens > max_tokens,
                    "effective_resolution": f"1/{effective_ratio}" if effective_ratio > 1 else "full"
                }
            )
            
        except Exception as e:
            logger.error(f"LongNet processing failed: {e}")
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
        """Get LongNet installation requirements."""
        return [
            "requests>=2.20.0"
        ]
    
    def cleanup(self):
        """Clean up LongNet resources."""
        # No resources to clean up for Ollama API client
        pass