#!/usr/bin/env python3
"""
Benchmarking module for lethe-research
Provides access to the optimized benchmarking system classes.
"""

# Import from the main benchmarking system
import sys
from pathlib import Path

# Add paths to find the benchmarking system
lethe_root = Path(__file__).parent.parent
sqlite_src_path = lethe_root / "packages" / "lethe-monitor" / "packages" / "sqlite" / "src"
sys.path.insert(0, str(sqlite_src_path))

try:
    from benchmarking import (
        BenchmarkMethod,
        CompetitorConfig, 
        LetheStreamingHybridCompetitor
    )
    
    # Re-export for convenience
    __all__ = [
        'BenchmarkMethod',
        'CompetitorConfig',
        'LetheStreamingHybridCompetitor'
    ]
    
except ImportError as e:
    # Fallback definitions if the main system isn't available
    from enum import Enum
    from dataclasses import dataclass, field
    from typing import Dict, Any
    
    class BenchmarkMethod(Enum):
        """Benchmark methods for comparison."""
        STREAMING = "streaming"
        LETHE = "lethe"  
        HYBRID = "hybrid"
    
    @dataclass 
    class CompetitorConfig:
        """Configuration for benchmark competitor."""
        method: BenchmarkMethod
        keep_ratio: float
        config_params: Dict[str, Any] = field(default_factory=dict)
    
    class LetheStreamingHybridCompetitor:
        """Fallback competitor implementation."""
        def __init__(self, method: BenchmarkMethod, config: CompetitorConfig):
            self.method = method
            self.config = config
            self.keep_ratio = config.keep_ratio
        
        def initialize(self) -> bool:
            return True
        
        def retrieve(self, query: str, context: str, max_tokens: int = 4000):
            """Fallback retrieve method"""
            from dataclasses import dataclass
            
            @dataclass
            class RetrievalResult:
                query_id: str
                retrieved_chunks: list
                context_used: str
                processing_time_ms: float
                metadata: dict
            
            # Simple fallback implementation
            if len(context) <= max_tokens:
                context_used = context
            else:
                context_used = context[:max_tokens]
            
            return RetrievalResult(
                query_id=str(hash(query)),
                retrieved_chunks=[(context_used, 1.0)],
                context_used=context_used,
                processing_time_ms=0.0,
                metadata={"method": self.method.value, "fallback": True}
            )
    
    __all__ = [
        'BenchmarkMethod',
        'CompetitorConfig', 
        'LetheStreamingHybridCompetitor'
    ]