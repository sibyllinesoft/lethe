#!/usr/bin/env python3
"""
Mock Competitor Implementation
==============================

Mock implementations for testing when real competitor systems aren't available.
"""

import logging
import time
import random
import numpy as np
from typing import Dict, List, Any

from .base import BaseCompetitor, CompetitorResult

logger = logging.getLogger(__name__)


class MockCompetitor(BaseCompetitor):
    """Mock competitor for testing purposes."""
    
    def __init__(
        self,
        name: str,
        api_endpoint: str = "http://localhost:8080",
        config_params: Dict[str, Any] = None,
        **kwargs
    ):
        """Initialize mock competitor."""
        super().__init__(
            name=name,
            api_endpoint=api_endpoint,
            config_params=config_params or {},
            **kwargs
        )
        
        # Set deterministic seed based on name for consistent results
        self.random_seed = hash(name) % 2**32
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        logger.info(f"Initialized mock competitor: {name}")
    
    def retrieve(
        self,
        query: str,
        context: str,
        keep_ratio: float,
        k: int = 100
    ) -> CompetitorResult:
        """Mock retrieval implementation."""
        
        start_time = time.time()
        
        # Simulate processing time based on competitor type
        processing_delay = self._get_processing_delay(keep_ratio)
        time.sleep(processing_delay / 1000.0)  # Convert ms to seconds
        
        # Generate mock results
        doc_ids, scores, tokens_kept = self._generate_mock_results(
            query, context, keep_ratio, k
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Count mock exact matches
        exact_matches = self._count_mock_exact_matches(query, doc_ids)
        
        return CompetitorResult(
            doc_ids=doc_ids,
            scores=scores,
            latency_ms=latency_ms,
            tokens_retrieved=sum(len(doc_id.split()) for doc_id in doc_ids),  # Mock token count
            exact_matches=exact_matches,
            keep_ratio=keep_ratio,
            tokens_kept=tokens_kept,
            original_context_tokens=len(context.split()),
            competitor_name=self.name,
            config_params=self.config_params,
            success=True
        )
    
    def _get_processing_delay(self, keep_ratio: float) -> float:
        """Get processing delay based on competitor characteristics."""
        
        # Base delay varies by competitor type
        base_delays = {
            "weaviate": 80,
            "milvus": 70,
            "vespa": 60,
            "opensearch": 90,
            "splade_v2": 150,
            "colbert_v2": 120,
            "ragatouille": 140,
            "bge_reranker_large": 200,
            "bge_m3_reranker": 180,
            "monot5": 250,
            "zoekt": 30,
            "livegrep": 25,
            "graphrag": 300,
            "streaming_llm": 400,
            "longnet": 350,
            "bge_m3_baseline": 100,
            "lethe_hybrid": 120
        }
        
        base_delay = base_delays.get(self.name, 100)
        
        # Add some variation based on keep_ratio (more processing for lower ratios)
        ratio_factor = 1 + (1 - keep_ratio) * 0.5
        
        # Add random variation (±20%)
        variation = np.random.uniform(0.8, 1.2)
        
        return base_delay * ratio_factor * variation
    
    def _generate_mock_results(
        self,
        query: str,
        context: str,
        keep_ratio: float,
        k: int
    ) -> tuple:
        """Generate mock retrieval results."""
        
        # Set seed based on query for consistent results
        query_seed = hash(query) % 2**32
        np.random.seed(query_seed)
        
        # Generate number of documents (varies by system)
        num_docs = min(k, np.random.randint(5, 50))
        
        # Generate document IDs
        doc_ids = [f"doc_{i:04d}" for i in range(num_docs)]
        
        # Generate scores based on competitor characteristics
        scores = self._generate_mock_scores(num_docs)
        
        # Calculate tokens kept
        total_tokens = len(context.split())
        tokens_kept = int(total_tokens * keep_ratio)
        
        return doc_ids, scores, tokens_kept
    
    def _generate_mock_scores(self, num_docs: int) -> List[float]:
        """Generate mock relevance scores."""
        
        # Different score distributions for different competitor types
        if "reranker" in self.name:
            # Rerankers tend to have more confident scores
            scores = np.random.beta(3, 2, num_docs)
        elif "vector" in self.name or "embedding" in self.name:
            # Vector systems have cosine-like scores
            scores = np.random.normal(0.7, 0.15, num_docs)
            scores = np.clip(scores, 0.1, 1.0)
        elif "bm25" in self.name or "sparse" in self.name:
            # Sparse systems have long-tail distributions
            scores = np.random.exponential(0.3, num_docs)
            scores = scores / max(scores)  # Normalize
        else:
            # Default uniform distribution
            scores = np.random.uniform(0.1, 0.9, num_docs)
        
        # Sort in descending order (as expected from retrievers)
        scores = sorted(scores, reverse=True)
        
        return scores
    
    def _count_mock_exact_matches(self, query: str, doc_ids: List[str]) -> int:
        """Count mock exact matches based on query."""
        
        # Simple heuristic: longer queries are more likely to have exact matches
        match_probability = min(0.8, len(query.split()) / 20.0)
        
        matches = 0
        for doc_id in doc_ids:
            if np.random.random() < match_probability:
                matches += 1
        
        return matches
    
    def health_check(self) -> bool:
        """Mock health check - always returns True."""
        return True
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get mock system information."""
        
        # System info varies by competitor type
        system_infos = {
            "weaviate": {
                "system": "Weaviate Vector Database",
                "version": "1.22.4",
                "search_type": "hybrid",
                "fusion_method": "rankedFusion"
            },
            "milvus": {
                "system": "Milvus Vector Database", 
                "version": "2.3.2",
                "search_type": "hybrid",
                "embedding_support": ["dense", "sparse"]
            },
            "splade_v2": {
                "system": "SPLADE v2",
                "version": "2.2.0",
                "model": "naver/splade-cocondenser-ensembledistil",
                "search_type": "learned_sparse"
            },
            "colbert_v2": {
                "system": "ColBERT v2",
                "version": "2.0",
                "model": "colbert-ir/colbertv2.0", 
                "search_type": "late_interaction"
            },
            "lethe_hybrid": {
                "system": "Lethe-Hybrid",
                "version": "1.0.0",
                "search_type": "adaptive_hybrid",
                "planning": "enabled"
            }
        }
        
        base_info = system_infos.get(self.name, {
            "system": f"Mock {self.name}",
            "version": "1.0.0",
            "search_type": "mock"
        })
        
        base_info.update({
            "config": self.config_params,
            "mock": True,
            "random_seed": self.random_seed
        })
        
        return base_info