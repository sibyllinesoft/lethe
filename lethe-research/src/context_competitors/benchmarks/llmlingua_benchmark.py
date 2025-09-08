"""
LLMLingua/LLMLingua-2 Context Compression Benchmark.

LLMLingua uses small language models to compress prompts while preserving
key information for downstream LLM tasks.

Paper: "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models"
GitHub: https://github.com/microsoft/LLMLingua
"""

import time
import logging
from typing import Dict, Any, List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from competitor_interface import ContextManagementCompetitor, ContextProcessingResult

logger = logging.getLogger(__name__)


class LLMLinguaCompetitor(ContextManagementCompetitor):
    """LLMLingua prompt compression competitor."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("LLMLingua", config)
        self.compressor = None
        self.device = config.get("device", "cpu") if config else "cpu"
    
    def initialize(self) -> bool:
        """Initialize LLMLingua compressor."""
        try:
            from llmlingua import PromptCompressor
            
            # Initialize with default small model
            self.compressor = PromptCompressor(
                model_name=self.config.get("model_name", "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"),
                device_map=self.device,
                use_llmlingua2=True  # Use LLMLingua-2 by default
            )
            
            self._initialized = True
            logger.info(f"LLMLingua initialized with device: {self.device}")
            return True
            
        except ImportError:
            logger.error("LLMLingua not installed. Run: pip install llmlingua")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize LLMLingua: {e}")
            return False
    
    def process_context(self, query: str, context: str, max_tokens: int = 4000) -> ContextProcessingResult:
        """Process context using LLMLingua compression."""
        if not self._initialized:
            raise RuntimeError("LLMLingua not initialized")
        
        start_time = time.time()
        original_tokens = len(context.split())
        
        try:
            # Compress the context while preserving query relevance
            compressed_result = self.compressor.compress_prompt(
                context,
                rate=max(0.1, min(0.8, max_tokens / original_tokens)),  # Adaptive compression rate
                query=query,
                use_sentence_level_filter=True,
                use_context_level_filter=True,
                use_token_level_filter=True,
                force_tokens=query.split(),  # Ensure query terms are preserved
            )
            
            processed_context = compressed_result["compressed_prompt"]
            
            # Generate response (placeholder - would use actual LLM)
            response = f"LLMLingua processed response to: {query} (context compressed {compressed_result['rate']:.2f}x)"
            
            processing_time = (time.time() - start_time) * 1000
            processed_tokens = len(processed_context.split())
            
            return ContextProcessingResult(
                original_context=context,
                processed_context=processed_context,
                query=query,
                response=response,
                processing_time_ms=processing_time,
                original_token_count=original_tokens,
                processed_token_count=processed_tokens,
                compression_ratio=compressed_result.get("rate", 1.0 - (float(processed_tokens) / float(original_tokens))) if original_tokens > 0 else 1.0,
                method_name=self.name,
                metadata={
                    "approach": "prompt_compression",
                    "model": self.compressor.model_name if hasattr(self.compressor, 'model_name') else "unknown",
                    "device": self.device,
                    "compression_rate": compressed_result.get("rate", 0),
                    "origin_tokens": compressed_result.get("origin_tokens", original_tokens),
                    "compressed_tokens": compressed_result.get("compressed_tokens", processed_tokens)
                }
            )
            
        except Exception as e:
            logger.error(f"LLMLingua processing failed: {e}")
            # Return error result
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
        """Get LLMLingua installation requirements."""
        return [
            "llmlingua>=0.2.0",
            "torch>=1.8.0",
            "transformers>=4.20.0",
            "accelerate>=0.20.0",
            "sentencepiece>=0.1.90"
        ]
    
    def cleanup(self):
        """Clean up LLMLingua resources."""
        if self.compressor is not None:
            # Clear model from GPU if loaded
            if hasattr(self.compressor, 'model'):
                del self.compressor.model
            del self.compressor
            self.compressor = None
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass