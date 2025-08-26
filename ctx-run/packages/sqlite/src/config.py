#!/usr/bin/env python3
"""
Comprehensive Configuration System for Milestone 3

Provides centralized configuration management with sensible defaults,
environment variable overrides, and validation for all components.

Key Features:
- Hierarchical configuration with environment overrides
- Component-specific configs with inheritance
- Performance tuning profiles (fast, balanced, quality)
- Validation and type checking
- Configuration persistence and loading
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from enum import Enum

from .planning import PlanningConfiguration
from .diversification import DiversificationConfig
from .reranker import RerankingConfig
from .hybrid_retrieval import HybridRetrievalConfig

logger = logging.getLogger(__name__)

class PerformanceProfile(Enum):
    """Performance vs quality trade-off profiles."""
    FAST = "fast"           # Sub-100ms target, minimal processing
    BALANCED = "balanced"   # Sub-200ms target, full pipeline
    QUALITY = "quality"     # No time limit, maximum quality

@dataclass
class SystemConfig:
    """Master system configuration with all components."""
    
    # Performance profile
    profile: PerformanceProfile = PerformanceProfile.BALANCED
    
    # Target hardware specification
    hardware_profile: str = "commodity"  # commodity, high-end, server
    
    # Component configurations
    planning: Optional[PlanningConfiguration] = None
    diversification: Optional[DiversificationConfig] = None  
    reranking: Optional[RerankingConfig] = None
    hybrid_retrieval: Optional[HybridRetrievalConfig] = None
    
    # Global settings
    log_level: str = "INFO"
    enable_telemetry: bool = True
    cache_dir: Optional[str] = None
    
    # Environment settings
    max_memory_mb: int = 2048        # Memory limit
    max_concurrent_queries: int = 10  # Concurrency limit
    enable_gpu: bool = False         # Force CPU-only by default
    
    def __post_init__(self):
        """Initialize component configs based on profile if not provided."""
        if self.planning is None:
            self.planning = self._create_planning_config()
        if self.diversification is None:
            self.diversification = self._create_diversification_config()
        if self.reranking is None:
            self.reranking = self._create_reranking_config()
        if self.hybrid_retrieval is None:
            self.hybrid_retrieval = self._create_hybrid_retrieval_config()
            
        # Apply environment overrides
        self._apply_environment_overrides()
    
    def _create_planning_config(self) -> PlanningConfiguration:
        """Create planning config based on performance profile."""
        if self.profile == PerformanceProfile.FAST:
            return PlanningConfiguration(
                tau_verify_idf=6.0,      # Lower threshold for faster decisions
                tau_entity_overlap=0.2,   
                tau_novelty=0.15,
                history_window=5          # Smaller context window
            )
        elif self.profile == PerformanceProfile.QUALITY:
            return PlanningConfiguration(
                tau_verify_idf=10.0,     # Higher threshold for precision
                tau_entity_overlap=0.4,
                tau_novelty=0.05,
                history_window=20         # Larger context window
            )
        else:  # BALANCED
            return PlanningConfiguration()  # Use defaults
    
    def _create_diversification_config(self) -> DiversificationConfig:
        """Create diversification config based on performance profile."""
        if self.profile == PerformanceProfile.FAST:
            return DiversificationConfig(
                max_tokens=4000,          # Smaller budget for speed
                max_docs=50,
                exact_match_patterns=[   # Fewer patterns for speed
                    r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(',
                    r'\b[A-Z][a-zA-Z0-9_]*\.[a-zA-Z0-9_]+',
                ]
            )
        elif self.profile == PerformanceProfile.QUALITY:
            return DiversificationConfig(
                max_tokens=16000,         # Larger budget for quality
                max_docs=200,
                exact_match_patterns=[   # More comprehensive patterns
                    r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(',
                    r'\b[A-Z][a-zA-Z0-9_]*\.[a-zA-Z0-9_]+',
                    r'\b[a-zA-Z0-9_]+_[a-zA-Z0-9_]+\b',
                    r'\b[A-Z][a-z]+[A-Z][a-z]*\b',
                    r'/[a-zA-Z0-9/_.-]+\.[a-z]{1,4}',  # File paths
                    r'\b[A-Z]+_[A-Z_]+\b',             # Constants
                ]
            )
        else:  # BALANCED
            return DiversificationConfig()  # Use defaults
    
    def _create_reranking_config(self) -> RerankingConfig:
        """Create reranking config based on performance profile."""
        if self.profile == PerformanceProfile.FAST:
            return RerankingConfig(
                enabled=False,            # Disabled for speed
                top_k_rerank=50,
                batch_size=16
            )
        elif self.profile == PerformanceProfile.QUALITY:
            return RerankingConfig(
                enabled=True,             # Enabled for quality
                model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",  # Larger model
                top_k_rerank=200,
                batch_size=8              # Smaller batches for larger model
            )
        else:  # BALANCED
            return RerankingConfig(
                enabled=False             # OFF by default per requirements
            )
    
    def _create_hybrid_retrieval_config(self) -> HybridRetrievalConfig:
        """Create hybrid retrieval config based on performance profile."""
        if self.profile == PerformanceProfile.FAST:
            return HybridRetrievalConfig(
                target_latency_ms=100.0,
                enable_reranking=False,
                enable_diversification=True,
                strict_latency_enforcement=True,
                planning_config=self.planning,
                diversification_config=self.diversification,
                reranking_config=self.reranking
            )
        elif self.profile == PerformanceProfile.QUALITY:
            return HybridRetrievalConfig(
                target_latency_ms=1000.0,  # Relaxed for quality
                enable_reranking=True,
                enable_diversification=True,
                strict_latency_enforcement=False,
                planning_config=self.planning,
                diversification_config=self.diversification,
                reranking_config=self.reranking
            )
        else:  # BALANCED
            return HybridRetrievalConfig(
                planning_config=self.planning,
                diversification_config=self.diversification, 
                reranking_config=self.reranking
            )
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides."""
        # Global overrides
        if "LETHE_LOG_LEVEL" in os.environ:
            self.log_level = os.environ["LETHE_LOG_LEVEL"]
        
        if "LETHE_MAX_MEMORY_MB" in os.environ:
            try:
                self.max_memory_mb = int(os.environ["LETHE_MAX_MEMORY_MB"])
            except ValueError:
                logger.warning("Invalid LETHE_MAX_MEMORY_MB value")
        
        if "LETHE_ENABLE_GPU" in os.environ:
            self.enable_gpu = os.environ["LETHE_ENABLE_GPU"].lower() in ("true", "1", "yes")
        
        # Performance profile override
        if "LETHE_PROFILE" in os.environ:
            try:
                self.profile = PerformanceProfile(os.environ["LETHE_PROFILE"])
                # Recreate component configs with new profile
                self.planning = self._create_planning_config()
                self.diversification = self._create_diversification_config()
                self.reranking = self._create_reranking_config()
                self.hybrid_retrieval = self._create_hybrid_retrieval_config()
            except ValueError:
                logger.warning(f"Invalid LETHE_PROFILE: {os.environ['LETHE_PROFILE']}")
        
        # Component-specific overrides
        self._apply_component_overrides()
    
    def _apply_component_overrides(self):
        """Apply component-specific environment overrides."""
        # Planning overrides
        if "LETHE_ALPHA_VERIFY" in os.environ:
            try:
                self.planning.alpha_verify = float(os.environ["LETHE_ALPHA_VERIFY"])
            except (ValueError, AttributeError):
                logger.warning("Invalid LETHE_ALPHA_VERIFY value")
        
        if "LETHE_ENABLE_RERANKING" in os.environ:
            enable = os.environ["LETHE_ENABLE_RERANKING"].lower() in ("true", "1", "yes")
            self.reranking.enabled = enable
            self.hybrid_retrieval.enable_reranking = enable
        
        # Target latency override
        if "LETHE_TARGET_LATENCY_MS" in os.environ:
            try:
                target = float(os.environ["LETHE_TARGET_LATENCY_MS"])
                self.hybrid_retrieval.target_latency_ms = target
            except (ValueError, AttributeError):
                logger.warning("Invalid LETHE_TARGET_LATENCY_MS value")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'profile': self.profile.value,
            'hardware_profile': self.hardware_profile,
            'log_level': self.log_level,
            'enable_telemetry': self.enable_telemetry,
            'cache_dir': self.cache_dir,
            'max_memory_mb': self.max_memory_mb,
            'max_concurrent_queries': self.max_concurrent_queries,
            'enable_gpu': self.enable_gpu,
            'planning': asdict(self.planning) if self.planning else None,
            'diversification': asdict(self.diversification) if self.diversification else None,
            'reranking': asdict(self.reranking) if self.reranking else None,
            'hybrid_retrieval': asdict(self.hybrid_retrieval) if self.hybrid_retrieval else None
        }
    
    def save(self, file_path: Union[str, Path]):
        """Save configuration to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'SystemConfig':
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct component configs
        config = cls()
        
        # Basic fields
        if 'profile' in data:
            config.profile = PerformanceProfile(data['profile'])
        if 'hardware_profile' in data:
            config.hardware_profile = data['hardware_profile']
        if 'log_level' in data:
            config.log_level = data['log_level']
        
        # Component configs would need custom deserialization
        # (Simplified for this example)
        
        logger.info(f"Configuration loaded from {file_path}")
        return config
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Memory validation
        if self.max_memory_mb < 512:
            errors.append("max_memory_mb too low (minimum 512MB)")
        
        # Component validation
        if self.hybrid_retrieval:
            if self.hybrid_retrieval.target_latency_ms <= 0:
                errors.append("target_latency_ms must be positive")
        
        if self.planning:
            if not (0 <= self.planning.alpha_verify <= 1):
                errors.append("alpha_verify must be in [0, 1]")
        
        return errors
    
    def optimize_for_hardware(self, hardware_type: str):
        """Optimize configuration for specific hardware."""
        if hardware_type == "low_memory":
            self.max_memory_mb = 1024
            self.diversification.max_tokens = 4000
            self.diversification.max_docs = 50
            self.planning.history_window = 5
            
        elif hardware_type == "high_performance":
            self.max_memory_mb = 8192
            self.diversification.max_tokens = 16000
            self.diversification.max_docs = 200
            self.planning.history_window = 20
            self.enable_gpu = True
            
        elif hardware_type == "server":
            self.max_memory_mb = 16384
            self.max_concurrent_queries = 50
            self.enable_gpu = True
            
        logger.info(f"Configuration optimized for {hardware_type}")


def create_default_config(profile: PerformanceProfile = PerformanceProfile.BALANCED) -> SystemConfig:
    """Create default system configuration."""
    return SystemConfig(profile=profile)

def create_fast_config() -> SystemConfig:
    """Create configuration optimized for speed (sub-100ms target)."""
    return SystemConfig(profile=PerformanceProfile.FAST)

def create_quality_config() -> SystemConfig:
    """Create configuration optimized for quality."""
    return SystemConfig(profile=PerformanceProfile.QUALITY)

def load_config_from_env() -> SystemConfig:
    """Load configuration with environment overrides applied."""
    config = SystemConfig()
    
    # Validate after environment overrides
    errors = config.validate()
    if errors:
        logger.warning(f"Configuration validation errors: {errors}")
    
    return config

# Pre-defined configuration templates
CONFIG_TEMPLATES = {
    "development": SystemConfig(
        profile=PerformanceProfile.FAST,
        log_level="DEBUG",
        enable_telemetry=True,
        max_memory_mb=1024
    ),
    
    "production": SystemConfig(
        profile=PerformanceProfile.BALANCED,
        log_level="INFO", 
        enable_telemetry=True,
        max_memory_mb=4096,
        max_concurrent_queries=20
    ),
    
    "research": SystemConfig(
        profile=PerformanceProfile.QUALITY,
        log_level="INFO",
        enable_telemetry=True,
        max_memory_mb=8192,
        enable_gpu=False  # Ensure reproducibility
    )
}

def get_config_template(template_name: str) -> SystemConfig:
    """Get pre-defined configuration template."""
    if template_name not in CONFIG_TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(CONFIG_TEMPLATES.keys())}")
    
    return CONFIG_TEMPLATES[template_name]