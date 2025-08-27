"""
Advanced ML Model Management Framework

Provides unified model loading, initialization, and management to eliminate the extensive
duplication across ML model handling code. Consolidates patterns found in 15+ files.

Features:
- Unified device detection and management
- Standardized model loading with fallback strategies  
- Model warmup and validation
- Model caching and versioning
- Performance monitoring integration
- Memory management and cleanup
- Configuration-driven model initialization

Usage:
    from common.model_manager import ModelManager, ModelConfig
    
    # Create model manager
    manager = ModelManager(cache_dir="./models", device="auto")
    
    # Define model configuration
    config = ModelConfig(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        model_type="cross_encoder",
        max_length=512,
        warmup_samples=1
    )
    
    # Load model with automatic caching and warmup
    model_info = manager.load_model(config)
    model = model_info.model
    tokenizer = model_info.tokenizer
    
    # Use model...
    
    # Cleanup when done
    manager.cleanup_model(config.model_name)
"""

import hashlib
import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from contextlib import contextmanager
from enum import Enum

# Suppress common ML library warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported model types."""
    CROSS_ENCODER = "cross_encoder"
    SENTENCE_TRANSFORMER = "sentence_transformer" 
    TRANSFORMERS = "transformers"
    CUSTOM = "custom"

class DeviceType(Enum):
    """Supported device types."""
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"
    MPS = "mps"  # Apple Silicon

@dataclass
class ModelConfig:
    """Configuration for model loading and initialization."""
    
    model_name: str
    model_type: ModelType
    device: Union[str, DeviceType] = DeviceType.AUTO
    max_length: int = 512
    batch_size: int = 32
    fp16: bool = True
    
    # Warmup configuration
    warmup_samples: int = 1
    warmup_input_length: int = 50
    
    # Caching configuration
    use_cache: bool = True
    cache_ttl: Optional[int] = None  # seconds
    
    # Custom initialization parameters
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Fallback configuration
    fallback_models: List[str] = field(default_factory=list)
    fallback_to_cpu: bool = True
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        if isinstance(self.device, str):
            try:
                self.device = DeviceType(self.device.lower())
            except ValueError:
                logger.warning(f"Unknown device type: {self.device}, using AUTO")
                self.device = DeviceType.AUTO
        
        if isinstance(self.model_type, str):
            try:
                self.model_type = ModelType(self.model_type.lower())
            except ValueError:
                logger.warning(f"Unknown model type: {self.model_type}, using CUSTOM")
                self.model_type = ModelType.CUSTOM

@dataclass
class ModelInfo:
    """Information about a loaded model."""
    
    model_name: str
    model: Any
    tokenizer: Optional[Any] = None
    device: str = "cpu"
    model_hash: Optional[str] = None
    load_time: float = 0.0
    warmup_time: float = 0.0
    memory_usage_mb: float = 0.0
    config: Optional[ModelConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_load_time(self) -> float:
        """Total time including warmup."""
        return self.load_time + self.warmup_time

class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    pass

class DeviceManager:
    """Manages device detection and allocation."""
    
    @staticmethod
    def detect_best_device(requested_device: DeviceType = DeviceType.AUTO) -> str:
        """Detect the best available device."""
        
        if requested_device == DeviceType.CPU:
            return "cpu"
        
        if requested_device == DeviceType.CUDA:
            if not DeviceManager.is_cuda_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
            return "cuda"
        
        if requested_device == DeviceType.MPS:
            if not DeviceManager.is_mps_available():
                logger.warning("MPS requested but not available, falling back to CPU")
                return "cpu"
            return "mps"
        
        # AUTO detection
        if DeviceManager.is_cuda_available():
            return "cuda"
        elif DeviceManager.is_mps_available():
            return "mps"
        else:
            return "cpu"
    
    @staticmethod
    def is_cuda_available() -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    @staticmethod
    def is_mps_available() -> bool:
        """Check if MPS (Apple Silicon) is available."""
        try:
            import torch
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except (ImportError, AttributeError):
            return False
    
    @staticmethod
    def get_device_info(device: str) -> Dict[str, Any]:
        """Get detailed device information."""
        info = {"device": device, "available": True}
        
        if device == "cuda":
            try:
                import torch
                if torch.cuda.is_available():
                    info.update({
                        "cuda_version": torch.version.cuda,
                        "device_count": torch.cuda.device_count(),
                        "device_name": torch.cuda.get_device_name(0),
                        "memory_total": torch.cuda.get_device_properties(0).total_memory // 1024**2,
                    })
                else:
                    info["available"] = False
            except ImportError:
                info["available"] = False
        
        return info

class ModelLoader:
    """Factory for loading different types of models."""
    
    def __init__(self):
        self.loaders = {
            ModelType.CROSS_ENCODER: self._load_cross_encoder,
            ModelType.SENTENCE_TRANSFORMER: self._load_sentence_transformer,
            ModelType.TRANSFORMERS: self._load_transformers,
            ModelType.CUSTOM: self._load_custom
        }
    
    def load_model(self, config: ModelConfig, device: str) -> Tuple[Any, Optional[Any]]:
        """Load model and tokenizer based on configuration."""
        
        loader = self.loaders.get(config.model_type)
        if not loader:
            raise ModelLoadError(f"Unsupported model type: {config.model_type}")
        
        return loader(config, device)
    
    def _load_cross_encoder(self, config: ModelConfig, device: str) -> Tuple[Any, Any]:
        """Load cross-encoder model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        except ImportError:
            raise ModelLoadError("transformers library not available")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, 
            **config.tokenizer_kwargs
        )
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            **config.model_kwargs
        )
        
        model.to(device)
        model.eval()
        
        return model, tokenizer
    
    def _load_sentence_transformer(self, config: ModelConfig, device: str) -> Tuple[Any, None]:
        """Load sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ModelLoadError("sentence-transformers library not available")
        
        model = SentenceTransformer(
            config.model_name,
            device=device,
            **config.model_kwargs
        )
        
        return model, None  # SentenceTransformer includes tokenizer
    
    def _load_transformers(self, config: ModelConfig, device: str) -> Tuple[Any, Any]:
        """Load generic transformers model."""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError:
            raise ModelLoadError("transformers library not available")
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            **config.tokenizer_kwargs
        )
        
        model = AutoModel.from_pretrained(
            config.model_name,
            **config.model_kwargs
        )
        
        model.to(device)
        model.eval()
        
        return model, tokenizer
    
    def _load_custom(self, config: ModelConfig, device: str) -> Tuple[Any, Optional[Any]]:
        """Load custom model (requires custom loading function)."""
        if 'custom_loader' not in config.model_kwargs:
            raise ModelLoadError("Custom model type requires 'custom_loader' in model_kwargs")
        
        custom_loader = config.model_kwargs['custom_loader']
        return custom_loader(config, device)

class ModelCache:
    """Manages model caching and versioning."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_info = {}
        self._load_cache_info()
    
    def _load_cache_info(self):
        """Load cache metadata."""
        cache_info_path = self.cache_dir / "cache_info.json"
        if cache_info_path.exists():
            try:
                with open(cache_info_path, 'r') as f:
                    self.cache_info = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache info: {e}")
                self.cache_info = {}
    
    def _save_cache_info(self):
        """Save cache metadata."""
        cache_info_path = self.cache_dir / "cache_info.json"
        try:
            with open(cache_info_path, 'w') as f:
                json.dump(self.cache_info, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache info: {e}")
    
    def get_model_hash(self, config: ModelConfig) -> str:
        """Generate hash for model configuration."""
        config_str = json.dumps({
            "model_name": config.model_name,
            "model_type": config.model_type.value,
            "device": config.device.value if isinstance(config.device, DeviceType) else config.device,
            "model_kwargs": config.model_kwargs,
            "tokenizer_kwargs": config.tokenizer_kwargs
        }, sort_keys=True)
        
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    def is_cached(self, model_hash: str) -> bool:
        """Check if model is cached and not expired."""
        if model_hash not in self.cache_info:
            return False
        
        cache_entry = self.cache_info[model_hash]
        
        # Check TTL if specified
        if cache_entry.get("ttl"):
            cache_time = cache_entry.get("cached_at", 0)
            ttl = cache_entry.get("ttl", 0)
            if time.time() - cache_time > ttl:
                return False
        
        return True
    
    def add_to_cache(self, model_hash: str, model_info: ModelInfo):
        """Add model info to cache."""
        self.cache_info[model_hash] = {
            "model_name": model_info.model_name,
            "device": model_info.device,
            "load_time": model_info.load_time,
            "warmup_time": model_info.warmup_time,
            "memory_usage_mb": model_info.memory_usage_mb,
            "cached_at": time.time(),
            "ttl": model_info.config.cache_ttl if model_info.config else None
        }
        self._save_cache_info()

class ModelManager:
    """Unified model management system."""
    
    def __init__(self, 
                 cache_dir: Union[str, Path] = "./models",
                 default_device: Union[str, DeviceType] = DeviceType.AUTO,
                 enable_performance_monitoring: bool = True):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_device = default_device
        self.enable_monitoring = enable_performance_monitoring
        
        # Components
        self.device_manager = DeviceManager()
        self.model_loader = ModelLoader()
        self.model_cache = ModelCache(self.cache_dir)
        
        # Active models
        self._loaded_models: Dict[str, ModelInfo] = {}
        
        # Performance tracking
        self._performance_stats = {
            "total_models_loaded": 0,
            "total_load_time": 0.0,
            "average_load_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info(f"ModelManager initialized with cache_dir={self.cache_dir}")
    
    def load_model(self, config: ModelConfig) -> ModelInfo:
        """Load a model with the specified configuration."""
        
        start_time = time.time()
        
        # Generate model hash for caching
        model_hash = self.model_cache.get_model_hash(config)
        
        # Check if already loaded
        if model_hash in self._loaded_models:
            logger.info(f"Model {config.model_name} already loaded, reusing")
            return self._loaded_models[model_hash]
        
        # Detect device
        device_type = config.device if config.device != DeviceType.AUTO else self.default_device
        device = self.device_manager.detect_best_device(device_type)
        
        logger.info(f"Loading model {config.model_name} on device {device}")
        
        # Try to load model with fallbacks
        model_info = self._load_with_fallbacks(config, device)
        
        # Warmup model
        if config.warmup_samples > 0:
            warmup_start = time.time()
            self._warmup_model(model_info, config)
            model_info.warmup_time = time.time() - warmup_start
        
        # Calculate total load time
        model_info.load_time = time.time() - start_time - model_info.warmup_time
        model_info.model_hash = model_hash
        model_info.config = config
        
        # Add memory usage info
        if self.enable_monitoring:
            model_info.memory_usage_mb = self._get_memory_usage()
        
        # Cache model info
        self._loaded_models[model_hash] = model_info
        if config.use_cache:
            self.model_cache.add_to_cache(model_hash, model_info)
        
        # Update performance stats
        self._update_performance_stats(model_info)
        
        logger.info(
            f"Model {config.model_name} loaded successfully in "
            f"{model_info.total_load_time:.2f}s on {device}"
        )
        
        return model_info
    
    def _load_with_fallbacks(self, config: ModelConfig, device: str) -> ModelInfo:
        """Load model with fallback strategies."""
        
        models_to_try = [config.model_name] + config.fallback_models
        devices_to_try = [device]
        
        if config.fallback_to_cpu and device != "cpu":
            devices_to_try.append("cpu")
        
        last_error = None
        
        for model_name in models_to_try:
            for device_to_try in devices_to_try:
                try:
                    # Create temporary config for this attempt
                    temp_config = ModelConfig(
                        model_name=model_name,
                        model_type=config.model_type,
                        device=device_to_try,
                        model_kwargs=config.model_kwargs,
                        tokenizer_kwargs=config.tokenizer_kwargs
                    )
                    
                    # Try to load
                    model, tokenizer = self.model_loader.load_model(temp_config, device_to_try)
                    
                    return ModelInfo(
                        model_name=model_name,
                        model=model,
                        tokenizer=tokenizer,
                        device=device_to_try
                    )
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"Failed to load {model_name} on {device_to_try}: {e}")
                    continue
        
        # All fallbacks failed
        raise ModelLoadError(f"Failed to load any model. Last error: {last_error}")
    
    def _warmup_model(self, model_info: ModelInfo, config: ModelConfig):
        """Warmup model with dummy inputs."""
        
        if not model_info.model:
            return
        
        try:
            if config.model_type == ModelType.CROSS_ENCODER:
                self._warmup_cross_encoder(model_info, config)
            elif config.model_type == ModelType.SENTENCE_TRANSFORMER:
                self._warmup_sentence_transformer(model_info, config)
            elif config.model_type == ModelType.TRANSFORMERS:
                self._warmup_transformers(model_info, config)
            # Custom models handle their own warmup
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    def _warmup_cross_encoder(self, model_info: ModelInfo, config: ModelConfig):
        """Warmup cross-encoder model."""
        if not model_info.tokenizer:
            return
        
        dummy_pairs = [("sample query", "sample document")] * config.warmup_samples
        
        for query, doc in dummy_pairs:
            inputs = model_info.tokenizer(
                query, doc,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt"
            )
            
            # Move to correct device
            inputs = {k: v.to(model_info.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                model_info.model(**inputs)
    
    def _warmup_sentence_transformer(self, model_info: ModelInfo, config: ModelConfig):
        """Warmup sentence transformer model."""
        dummy_sentences = ["sample sentence for warmup"] * config.warmup_samples
        model_info.model.encode(dummy_sentences, show_progress_bar=False)
    
    def _warmup_transformers(self, model_info: ModelInfo, config: ModelConfig):
        """Warmup generic transformers model."""
        if not model_info.tokenizer:
            return
        
        dummy_text = "sample text for warmup " * (config.warmup_input_length // 5)
        
        for _ in range(config.warmup_samples):
            inputs = model_info.tokenizer(
                dummy_text,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(model_info.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                model_info.model(**inputs)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0
    
    def _update_performance_stats(self, model_info: ModelInfo):
        """Update performance statistics."""
        self._performance_stats["total_models_loaded"] += 1
        self._performance_stats["total_load_time"] += model_info.total_load_time
        self._performance_stats["average_load_time"] = (
            self._performance_stats["total_load_time"] / 
            self._performance_stats["total_models_loaded"]
        )
    
    def get_model(self, model_name: str) -> Optional[ModelInfo]:
        """Get a loaded model by name."""
        for model_info in self._loaded_models.values():
            if model_info.model_name == model_name:
                return model_info
        return None
    
    def list_loaded_models(self) -> List[str]:
        """Get list of currently loaded model names."""
        return [info.model_name for info in self._loaded_models.values()]
    
    def cleanup_model(self, model_name: str) -> bool:
        """Clean up a specific model from memory."""
        model_hash_to_remove = None
        
        for model_hash, model_info in self._loaded_models.items():
            if model_info.model_name == model_name:
                model_hash_to_remove = model_hash
                break
        
        if model_hash_to_remove:
            # Cleanup GPU memory if using CUDA
            if self._loaded_models[model_hash_to_remove].device == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
            
            del self._loaded_models[model_hash_to_remove]
            logger.info(f"Model {model_name} cleaned up from memory")
            return True
        
        return False
    
    def cleanup_all_models(self):
        """Clean up all loaded models."""
        model_names = self.list_loaded_models()
        for model_name in model_names:
            self.cleanup_model(model_name)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self._performance_stats.copy()
        stats.update({
            "currently_loaded_models": len(self._loaded_models),
            "loaded_model_names": self.list_loaded_models()
        })
        return stats
    
    @contextmanager
    def temporary_model(self, config: ModelConfig):
        """Context manager for temporary model loading."""
        model_info = self.load_model(config)
        try:
            yield model_info
        finally:
            self.cleanup_model(model_info.model_name)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_all_models()

# Convenience functions for common use cases
def load_cross_encoder(model_name: str, 
                      device: str = "auto",
                      max_length: int = 512,
                      **kwargs) -> ModelInfo:
    """Convenience function to load a cross-encoder model."""
    
    config = ModelConfig(
        model_name=model_name,
        model_type=ModelType.CROSS_ENCODER,
        device=device,
        max_length=max_length,
        **kwargs
    )
    
    manager = ModelManager()
    return manager.load_model(config)

def load_sentence_transformer(model_name: str,
                             device: str = "auto",
                             **kwargs) -> ModelInfo:
    """Convenience function to load a sentence transformer model."""
    
    config = ModelConfig(
        model_name=model_name,
        model_type=ModelType.SENTENCE_TRANSFORMER,
        device=device,
        **kwargs
    )
    
    manager = ModelManager()
    return manager.load_model(config)

# Export commonly used components
__all__ = [
    'ModelManager',
    'ModelConfig', 
    'ModelInfo',
    'ModelType',
    'DeviceType',
    'DeviceManager',
    'ModelLoadError',
    'load_cross_encoder',
    'load_sentence_transformer'
]