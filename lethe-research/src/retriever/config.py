"""
Configuration Management for Production IR System

Centralized configuration for BM25 and ANN indices with validation
and parameter management for reproducible experiments.

Features:
- Typed configuration classes
- Parameter validation and constraints  
- Environment-based configuration loading
- Index-specific parameter sets
- Budget constraint validation
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import yaml
import os
import math

@dataclass
class BM25Config:
    """Configuration for BM25 indices."""
    
    # Core BM25 parameters
    k1: float = 0.9
    b: float = 0.4
    
    # Index settings
    stemmer: str = "porter"  # porter, krovetz, none
    stopwords: bool = True
    lowercase: bool = True
    
    # Build settings
    ram_buffer_size: int = 2048  # MB
    max_merge_count: int = 8
    
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        if not (0.1 <= self.k1 <= 5.0):
            errors.append(f"k1 must be in range (0.1, 5.0), got {self.k1}")
            
        if not (0.0 <= self.b <= 1.0):
            errors.append(f"b must be in range (0.0, 1.0), got {self.b}")
            
        if self.ram_buffer_size <= 0:
            errors.append(f"ram_buffer_size must be positive, got {self.ram_buffer_size}")
            
        return errors

@dataclass
class HNSWConfig:
    """Configuration for HNSW indices."""
    
    # Build parameters
    m: int = 16  # Number of bi-directional links
    ef_construction: int = 200  # Size of dynamic candidate list during construction
    max_m: int = 16  # Maximum number of bi-directional links for every new element
    ml: float = 1 / math.log(2.0)  # Level generation parameter
    
    # Search parameters  
    ef_search_values: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    target_recall: float = 0.98  # Target recall@1000
    
    # Memory and performance
    max_elements: int = 1000000
    memory_limit_gb: float = 4.0
    
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        if not (4 <= self.m <= 64):
            errors.append(f"m must be in range (4, 64), got {self.m}")
            
        if not (10 <= self.ef_construction <= 2000):
            errors.append(f"ef_construction must be in range (10, 2000), got {self.ef_construction}")
            
        if self.max_m < self.m:
            errors.append(f"max_m ({self.max_m}) must be >= m ({self.m})")
            
        if self.target_recall < 0.0 or self.target_recall > 1.0:
            errors.append(f"target_recall must be in [0, 1], got {self.target_recall}")
            
        return errors

@dataclass
class IVFPQConfig:
    """Configuration for IVF-PQ indices."""
    
    # IVF parameters
    nlist_values: List[int] = field(default_factory=lambda: [1000, 4000, 16000])
    nprobe_values: List[int] = field(default_factory=lambda: [1, 4, 16, 64])
    
    # PQ parameters  
    nbits_values: List[int] = field(default_factory=lambda: [6, 8, 10])
    m_pq: int = 64  # Number of subquantizers
    
    # Training parameters
    training_sample_size: int = 1000000
    max_training_iterations: int = 25
    
    # Search parameters
    target_recall: float = 0.98
    
    # Budget constraints
    memory_limit_gb: float = 2.0
    compute_flops_budget: int = 1000000  # Approximate FLOPs per query
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        if not all(n > 0 for n in self.nlist_values):
            errors.append("All nlist_values must be positive")
            
        if not all(n > 0 for n in self.nprobe_values):
            errors.append("All nprobe_values must be positive") 
            
        if not all(b in [6, 8, 10, 12] for b in self.nbits_values):
            errors.append("nbits_values must be in [6, 8, 10, 12]")
            
        if self.m_pq <= 0 or self.m_pq % 8 != 0:
            errors.append("m_pq must be positive and divisible by 8")
            
        return errors

@dataclass
class EmbeddingConfig:
    """Configuration for dense embeddings."""
    
    # Model settings
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_cache_dir: str = "./models"
    
    # Encoding settings
    batch_size: int = 32
    max_length: int = 512
    normalize_embeddings: bool = True
    
    # Performance settings
    device: str = "cuda"  # cuda, cpu, auto
    fp16: bool = True
    dataloader_num_workers: int = 4
    
    # Persistence
    embeddings_cache_dir: str = "./embeddings"
    cache_embeddings: bool = True
    
    def validate(self) -> List[str]:
        """Validate configuration parameters."""
        errors = []
        
        if self.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {self.batch_size}")
            
        if self.max_length <= 0:
            errors.append(f"max_length must be positive, got {self.max_length}")
            
        if self.device not in ["cuda", "cpu", "auto"]:
            errors.append(f"device must be cuda/cpu/auto, got {self.device}")
            
        return errors

@dataclass 
class SystemConfig:
    """System-level configuration."""
    
    # Hardware profile
    cpu_cores: int = 32
    memory_gb: int = 128
    gpu_memory_gb: Optional[int] = 40  # A100
    
    # Performance settings
    concurrency: int = 1  # For latency measurements
    warm_cache: bool = True
    
    # Directories
    indices_dir: str = "./indices"
    data_dir: str = "./datasets" 
    models_dir: str = "./models"
    
    # Budget constraints (±5% compute/FLOPs parity)
    flops_budget_variance: float = 0.05
    
    def validate(self) -> List[str]:
        """Validate system configuration."""
        errors = []
        
        if self.cpu_cores <= 0:
            errors.append("cpu_cores must be positive")
            
        if self.memory_gb <= 0:
            errors.append("memory_gb must be positive")
            
        if self.concurrency <= 0:
            errors.append("concurrency must be positive")
            
        return errors

@dataclass
class RetrieverConfig:
    """Master configuration for the IR system."""
    
    # Component configurations
    bm25: BM25Config = field(default_factory=BM25Config)
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)
    ivf_pq: IVFPQConfig = field(default_factory=IVFPQConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    # Dataset configurations
    datasets: List[str] = field(default_factory=lambda: [
        "msmarco-passage-dev", 
        "trec-covid", 
        "nfcorpus", 
        "fiqa-2018"
    ])
    
    # Global settings
    debug: bool = False
    log_level: str = "INFO"
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'RetrieverConfig':
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
            
        return cls.from_dict(data)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrieverConfig':
        """Create configuration from dictionary."""
        # Create component configs
        bm25_config = BM25Config(**data.get('bm25', {}))
        hnsw_config = HNSWConfig(**data.get('hnsw', {}))
        ivf_pq_config = IVFPQConfig(**data.get('ivf_pq', {}))
        embeddings_config = EmbeddingConfig(**data.get('embeddings', {}))
        system_config = SystemConfig(**data.get('system', {}))
        
        # Create master config
        return cls(
            bm25=bm25_config,
            hnsw=hnsw_config,
            ivf_pq=ivf_pq_config,
            embeddings=embeddings_config,
            system=system_config,
            datasets=data.get('datasets', cls().datasets),
            debug=data.get('debug', False),
            log_level=data.get('log_level', 'INFO')
        )
        
    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        
        # Convert to dictionary
        data = self.to_dict()
        
        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'bm25': asdict(self.bm25),
            'hnsw': asdict(self.hnsw),
            'ivf_pq': asdict(self.ivf_pq),
            'embeddings': asdict(self.embeddings),
            'system': asdict(self.system),
            'datasets': self.datasets,
            'debug': self.debug,
            'log_level': self.log_level
        }
        
    def validate(self) -> List[str]:
        """Validate entire configuration."""
        all_errors = []
        
        # Validate component configs
        all_errors.extend([f"BM25: {err}" for err in self.bm25.validate()])
        all_errors.extend([f"HNSW: {err}" for err in self.hnsw.validate()])  
        all_errors.extend([f"IVF-PQ: {err}" for err in self.ivf_pq.validate()])
        all_errors.extend([f"Embeddings: {err}" for err in self.embeddings.validate()])
        all_errors.extend([f"System: {err}" for err in self.system.validate()])
        
        # Cross-component validation
        if not self.datasets:
            all_errors.append("At least one dataset must be specified")
            
        return all_errors
        
    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset-specific configuration."""
        base_config = {
            'bm25_params': asdict(self.bm25),
            'hnsw_params': asdict(self.hnsw), 
            'ivf_pq_params': asdict(self.ivf_pq),
            'embedding_params': asdict(self.embeddings)
        }
        
        # Dataset-specific overrides could go here
        # For now, return base config
        return base_config

# Math functions used in configuration

# Environment-based configuration loading
def load_config_from_env() -> RetrieverConfig:
    """Load configuration with environment variable overrides."""
    
    # Start with defaults
    config = RetrieverConfig()
    
    # Override from environment variables
    if os.getenv('LETHE_BM25_K1'):
        config.bm25.k1 = float(os.getenv('LETHE_BM25_K1'))
        
    if os.getenv('LETHE_BM25_B'):
        config.bm25.b = float(os.getenv('LETHE_BM25_B'))
        
    if os.getenv('LETHE_MODEL_NAME'):
        config.embeddings.model_name = os.getenv('LETHE_MODEL_NAME')
        
    if os.getenv('LETHE_DEVICE'):
        config.embeddings.device = os.getenv('LETHE_DEVICE')
        
    if os.getenv('LETHE_DEBUG'):
        config.debug = os.getenv('LETHE_DEBUG').lower() == 'true'
        
    if os.getenv('LETHE_LOG_LEVEL'):
        config.log_level = os.getenv('LETHE_LOG_LEVEL')
        
    return config

def create_default_config_file(output_path: Union[str, Path]) -> None:
    """Create a default configuration file."""
    config = RetrieverConfig()
    config.to_yaml(output_path)
    print(f"Default configuration saved to: {output_path}")