#!/usr/bin/env python3
"""
Competitor Registry
==================

Central registry for all competitor implementations with automatic discovery.
"""

import logging
from typing import Dict, Type, Optional, List
from .base import BaseCompetitor

logger = logging.getLogger(__name__)


class CompetitorRegistry:
    """Registry for all competitor implementations."""
    
    def __init__(self):
        """Initialize competitor registry."""
        self._competitors: Dict[str, Type[BaseCompetitor]] = {}
        self._register_default_competitors()
    
    def _register_default_competitors(self):
        """Register all built-in competitors."""
        # Import and register competitors
        try:
            # Lethe baseline
            from .lethe_baseline import LetheHybridCompetitor
            self.register("lethe_hybrid", LetheHybridCompetitor)
            
            # Hybrid Vector DBs
            from .hybrid_vector_db import (
                WeaviateCompetitor, MilvusCompetitor, 
                VespaCompetitor, OpenSearchCompetitor
            )
            self.register("weaviate", WeaviateCompetitor)
            self.register("milvus", MilvusCompetitor)
            self.register("vespa", VespaCompetitor)
            self.register("opensearch", OpenSearchCompetitor)
            
            # Learned Sparse/Late-Interaction
            from .learned_sparse import (
                SpladeV2Competitor, ColBERTV2Competitor, RAGatouilleCompetitor
            )
            self.register("splade_v2", SpladeV2Competitor)
            self.register("colbert_v2", ColBERTV2Competitor)
            self.register("ragatouille", RAGatouilleCompetitor)
            
            # Rerankers
            from .rerankers import (
                BGERerankerCompetitor, BGEM3RerankerCompetitor, MonoT5Competitor
            )
            self.register("bge_reranker_large", BGERerankerCompetitor)
            self.register("bge_m3_reranker", BGEM3RerankerCompetitor)
            self.register("monot5", MonoT5Competitor)
            
            # Code Search & Graph
            from .code_search import (
                ZoektCompetitor, LivegrepCompetitor, GraphRAGCompetitor
            )
            self.register("zoekt", ZoektCompetitor)
            self.register("livegrep", LivegrepCompetitor)
            self.register("graphrag", GraphRAGCompetitor)
            
            # Long Context
            from .long_context import (
                StreamingLLMCompetitor, LongNetCompetitor, BGEM3BaselineCompetitor
            )
            self.register("streaming_llm", StreamingLLMCompetitor)
            self.register("longnet", LongNetCompetitor)
            self.register("bge_m3_baseline", BGEM3BaselineCompetitor)
            
        except ImportError as e:
            logger.warning(f"Some competitor implementations not available: {e}")
            # Register mock competitors for testing
            self._register_mock_competitors()
        
        logger.info(f"Registered {len(self._competitors)} competitor implementations")
    
    def _register_mock_competitors(self):
        """Register mock competitors for testing when real ones aren't available."""
        from .mock import MockCompetitor
        
        mock_names = [
            "weaviate", "milvus", "vespa", "opensearch",
            "splade_v2", "colbert_v2", "ragatouille", 
            "bge_reranker_large", "bge_m3_reranker", "monot5",
            "zoekt", "livegrep", "graphrag",
            "streaming_llm", "longnet", "bge_m3_baseline",
            "lethe_hybrid"
        ]
        
        for name in mock_names:
            self.register(name, MockCompetitor)
            logger.debug(f"Registered mock competitor: {name}")
    
    def register(self, name: str, competitor_class: Type[BaseCompetitor]):
        """Register a competitor implementation."""
        if not issubclass(competitor_class, BaseCompetitor):
            raise ValueError(f"Competitor class must inherit from BaseCompetitor")
        
        self._competitors[name] = competitor_class
        logger.debug(f"Registered competitor: {name}")
    
    def get_competitor(
        self, 
        name: str,
        config_params: Optional[Dict] = None,
        **kwargs
    ) -> BaseCompetitor:
        """Get a configured competitor instance."""
        if name not in self._competitors:
            raise ValueError(f"Unknown competitor: {name}. Available: {self.list_competitors()}")
        
        competitor_class = self._competitors[name]
        
        # Import config if not provided
        if config_params is None:
            from ..config import get_competitor_config
            try:
                config = get_competitor_config(name)
                config_params = config.config_params
                kwargs.update({
                    'api_endpoint': config.api_endpoint,
                    'timeout_seconds': config.timeout_seconds,
                    'max_retries': config.max_retries
                })
            except Exception as e:
                logger.warning(f"Could not load config for {name}: {e}")
                config_params = {}
        
        return competitor_class(
            name=name,
            config_params=config_params,
            **kwargs
        )
    
    def list_competitors(self) -> List[str]:
        """List all registered competitor names."""
        return sorted(self._competitors.keys())
    
    def get_competitors_by_category(self) -> Dict[str, List[str]]:
        """Group competitors by category."""
        from ..config import COMPETITOR_CONFIGS
        
        categories = {}
        for name in self._competitors.keys():
            if name in COMPETITOR_CONFIGS:
                category = COMPETITOR_CONFIGS[name].category
                if category not in categories:
                    categories[category] = []
                categories[category].append(name)
            else:
                # Default category for unknown competitors
                if "other" not in categories:
                    categories["other"] = []
                categories["other"].append(name)
        
        return categories
    
    def validate_competitor(self, name: str) -> Dict[str, any]:
        """Validate a competitor implementation."""
        if name not in self._competitors:
            return {"valid": False, "error": f"Unknown competitor: {name}"}
        
        try:
            # Try to instantiate competitor
            competitor = self.get_competitor(name)
            
            # Check required methods
            required_methods = ["retrieve", "health_check", "get_system_info"]
            missing_methods = []
            
            for method in required_methods:
                if not hasattr(competitor, method):
                    missing_methods.append(method)
            
            if missing_methods:
                return {
                    "valid": False,
                    "error": f"Missing required methods: {missing_methods}"
                }
            
            return {"valid": True, "class": competitor.__class__.__name__}
            
        except Exception as e:
            return {"valid": False, "error": str(e)}


# Global registry instance
_registry = CompetitorRegistry()

def get_competitor_registry() -> CompetitorRegistry:
    """Get the global competitor registry."""
    return _registry

def get_competitor(name: str, **kwargs) -> BaseCompetitor:
    """Convenience function to get a competitor."""
    return _registry.get_competitor(name, **kwargs)

def list_available_competitors() -> List[str]:
    """List all available competitors."""
    return _registry.list_competitors()