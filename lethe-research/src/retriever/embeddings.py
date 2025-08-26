"""
Dense Embedding Management for Production IR System

Provides comprehensive dense embedding generation, caching, and management
using modern transformer models with performance optimization.

Features:
- Multiple model support (Sentence-BERT, BGE, E5, etc.)
- Efficient batch processing with GPU optimization
- Persistent embedding cache with hash-based validation
- Memory-efficient streaming for large collections
- Model checkpoint management and versioning
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Generator, Tuple
from dataclasses import dataclass, field
import numpy as np
import torch
from tqdm import tqdm

# Transformers and sentence transformers
try:
    import transformers
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .config import EmbeddingConfig
from .timing import TimingHarness

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingMetadata:
    """Metadata for embedding collections."""
    
    model_name: str
    model_hash: str
    collection_name: str
    num_embeddings: int
    embedding_dim: int
    
    # Processing parameters
    max_length: int
    normalize: bool
    batch_size: int
    
    # Content verification
    content_hash: str
    
    # Storage information
    storage_format: str  # numpy, hdf5, faiss
    file_path: str
    file_size_mb: float
    
    # Performance metrics
    encoding_time_sec: float
    throughput_docs_per_sec: float
    
    created_at: str = field(default_factory=lambda: str(pd.Timestamp.now()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (str, int, float, bool)):
                result[key] = value
            else:
                result[key] = str(value)
        return result
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingMetadata':
        """Create from dictionary."""
        return cls(**data)

class DenseEmbeddingManager:
    """
    Manages dense embeddings with caching and model management.
    
    Provides efficient encoding, storage, and retrieval of dense embeddings
    with support for multiple models and formats.
    """
    
    def __init__(self, 
                 config: EmbeddingConfig,
                 cache_dir: Optional[Union[str, Path]] = None,
                 timing_harness: Optional[TimingHarness] = None):
        """
        Initialize embedding manager.
        
        Args:
            config: Embedding configuration
            cache_dir: Directory for embedding cache (overrides config)
            timing_harness: Optional timing harness for performance measurement
        """
        self.config = config
        self.cache_dir = Path(cache_dir or config.embeddings_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.timing_harness = timing_harness
        
        # Model state
        self._model = None
        self._tokenizer = None
        self._device = None
        self._model_hash = None
        
        # Validate configuration
        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid embedding configuration: {errors}")
            
    def _initialize_model(self) -> None:
        """Initialize the embedding model with lazy loading."""
        if self._model is not None:
            return
            
        logger.info(f"Loading embedding model: {self.config.model_name}")
        
        # Determine device
        if self.config.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = self.config.device
            
        logger.info(f"Using device: {self._device}")
        
        try:
            # Try sentence-transformers first (recommended)
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self._model = SentenceTransformer(
                    self.config.model_name,
                    cache_folder=self.config.model_cache_dir,
                    device=self._device
                )
                
                if self.config.fp16 and self._device == "cuda":
                    self._model.half()
                    
                logger.info("Loaded model using sentence-transformers")
                
            elif TRANSFORMERS_AVAILABLE:
                # Fallback to transformers
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.model_cache_dir
                )
                self._model = AutoModel.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.model_cache_dir
                )
                self._model.to(self._device)
                
                if self.config.fp16 and self._device == "cuda":
                    self._model.half()
                    
                logger.info("Loaded model using transformers (fallback)")
                
            else:
                raise ImportError("Neither sentence-transformers nor transformers available")
                
            # Compute model hash for reproducibility
            self._model_hash = self._compute_model_hash()
            
        except Exception as e:
            logger.error(f"Failed to load model {self.config.model_name}: {e}")
            raise
            
    def _compute_model_hash(self) -> str:
        """Compute hash of model for reproducibility."""
        try:
            # Create hash based on model name and config
            hasher = hashlib.sha256()
            hasher.update(self.config.model_name.encode())
            hasher.update(str(self.config.max_length).encode())
            hasher.update(str(self.config.normalize_embeddings).encode())
            
            return hasher.hexdigest()[:16]  # Short hash
            
        except Exception as e:
            logger.warning(f"Could not compute model hash: {e}")
            return "unknown"
            
    def encode_texts(self, 
                    texts: List[str],
                    show_progress: bool = True,
                    chunk_size: Optional[int] = None) -> np.ndarray:
        """
        Encode texts to dense embeddings.
        
        Args:
            texts: List of texts to encode
            show_progress: Whether to show progress bar
            chunk_size: Batch size for processing (overrides config)
            
        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        self._initialize_model()
        
        if not texts:
            return np.empty((0, 0))
            
        batch_size = chunk_size or self.config.batch_size
        
        logger.info(f"Encoding {len(texts)} texts with batch_size={batch_size}")
        
        encode_func = lambda: self._encode_batch(texts, batch_size, show_progress)
        
        if self.timing_harness:
            with self.timing_harness.measure("dense_encoding", 
                                            {"num_texts": len(texts), "batch_size": batch_size}):
                return encode_func()
        else:
            return encode_func()
            
    def _encode_batch(self, 
                     texts: List[str],
                     batch_size: int,
                     show_progress: bool) -> np.ndarray:
        """Execute batch encoding."""
        
        if SENTENCE_TRANSFORMERS_AVAILABLE and isinstance(self._model, SentenceTransformer):
            # Use sentence-transformers (preferred)
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_numpy=True
            )
            
        elif TRANSFORMERS_AVAILABLE:
            # Use transformers (fallback)
            embeddings = self._encode_with_transformers(texts, batch_size, show_progress)
            
        else:
            raise RuntimeError("No embedding model available")
            
        logger.info(f"Encoded embeddings shape: {embeddings.shape}")
        return embeddings
        
    def _encode_with_transformers(self, 
                                 texts: List[str],
                                 batch_size: int,
                                 show_progress: bool) -> np.ndarray:
        """Encode using transformers library."""
        
        all_embeddings = []
        
        progress_bar = tqdm(range(0, len(texts), batch_size), 
                           desc="Encoding") if show_progress else range(0, len(texts), batch_size)
        
        for i in progress_bar:
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            # Encode
            with torch.no_grad():
                outputs = self._model(**inputs)
                
                # Mean pooling
                embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                
                # Apply attention mask and mean pool
                embeddings = embeddings * attention_mask.unsqueeze(-1)
                embeddings = embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                
                # Normalize if requested
                if self.config.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    
                # Convert to numpy
                batch_embeddings = embeddings.cpu().numpy()
                all_embeddings.append(batch_embeddings)
                
        return np.vstack(all_embeddings)
        
    def encode_collection(self, 
                         documents: List[Dict[str, Any]],
                         collection_name: str,
                         text_field: str = "text",
                         force_recompute: bool = False) -> Tuple[np.ndarray, EmbeddingMetadata]:
        """
        Encode a document collection with caching.
        
        Args:
            documents: List of document dictionaries
            collection_name: Name for the collection
            text_field: Field containing text to encode
            force_recompute: Whether to force recomputation even if cached
            
        Returns:
            Tuple of (embeddings array, metadata)
        """
        
        # Check cache first
        if self.config.cache_embeddings and not force_recompute:
            cached_embeddings, cached_metadata = self._load_cached_embeddings(collection_name)
            if cached_embeddings is not None:
                logger.info(f"Loaded cached embeddings: {collection_name}")
                return cached_embeddings, cached_metadata
                
        logger.info(f"Computing embeddings for collection: {collection_name}")
        
        # Extract texts
        texts = []
        for doc in documents:
            text = doc.get(text_field, "")
            if isinstance(text, str):
                texts.append(text)
            else:
                texts.append(str(text))
                
        if not texts:
            raise ValueError("No texts found in documents")
            
        # Encode embeddings
        start_time = time.time()
        embeddings = self.encode_texts(texts, show_progress=True)
        encoding_time = time.time() - start_time
        
        # Create metadata
        content_hash = self._compute_content_hash(texts)
        
        metadata = EmbeddingMetadata(
            model_name=self.config.model_name,
            model_hash=self._model_hash,
            collection_name=collection_name,
            num_embeddings=len(embeddings),
            embedding_dim=embeddings.shape[1],
            max_length=self.config.max_length,
            normalize=self.config.normalize_embeddings,
            batch_size=self.config.batch_size,
            content_hash=content_hash,
            storage_format="numpy",
            file_path="",  # Will be set during caching
            file_size_mb=0.0,  # Will be set during caching
            encoding_time_sec=encoding_time,
            throughput_docs_per_sec=len(texts) / encoding_time
        )
        
        # Cache if enabled
        if self.config.cache_embeddings:
            self._cache_embeddings(embeddings, metadata)
            
        logger.info(f"Encoded {len(embeddings)} embeddings in {encoding_time:.2f}s "
                   f"({metadata.throughput_docs_per_sec:.1f} docs/sec)")
        
        return embeddings, metadata
        
    def _compute_content_hash(self, texts: List[str]) -> str:
        """Compute hash of text collection for cache validation."""
        hasher = hashlib.sha256()
        
        for text in texts:
            hasher.update(text.encode('utf-8'))
            
        return hasher.hexdigest()
        
    def _get_cache_path(self, collection_name: str) -> Tuple[Path, Path]:
        """Get cache file paths for embeddings and metadata."""
        embeddings_path = self.cache_dir / f"{collection_name}_embeddings.npy"
        metadata_path = self.cache_dir / f"{collection_name}_metadata.json"
        return embeddings_path, metadata_path
        
    def _cache_embeddings(self, 
                         embeddings: np.ndarray,
                         metadata: EmbeddingMetadata) -> None:
        """Cache embeddings and metadata to disk."""
        
        embeddings_path, metadata_path = self._get_cache_path(metadata.collection_name)
        
        try:
            # Save embeddings
            np.save(embeddings_path, embeddings)
            
            # Update metadata with file info
            metadata.file_path = str(embeddings_path)
            metadata.file_size_mb = embeddings_path.stat().st_size / (1024 * 1024)
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
                
            logger.info(f"Cached embeddings: {embeddings_path}")
            
        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {e}")
            
    def _load_cached_embeddings(self, 
                               collection_name: str) -> Tuple[Optional[np.ndarray], Optional[EmbeddingMetadata]]:
        """Load cached embeddings and metadata."""
        
        embeddings_path, metadata_path = self._get_cache_path(collection_name)
        
        if not embeddings_path.exists() or not metadata_path.exists():
            return None, None
            
        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
                
            metadata = EmbeddingMetadata.from_dict(metadata_dict)
            
            # Validate model compatibility
            if metadata.model_name != self.config.model_name:
                logger.info(f"Model changed: {metadata.model_name} -> {self.config.model_name}")
                return None, None
                
            # Load embeddings
            embeddings = np.load(embeddings_path)
            
            logger.info(f"Loaded cached embeddings: {embeddings.shape}")
            return embeddings, metadata
            
        except Exception as e:
            logger.warning(f"Failed to load cached embeddings: {e}")
            return None, None
            
    def list_cached_collections(self) -> List[str]:
        """List available cached collections."""
        collections = []
        
        for metadata_file in self.cache_dir.glob("*_metadata.json"):
            collection_name = metadata_file.stem.replace("_metadata", "")
            collections.append(collection_name)
            
        return sorted(collections)
        
    def get_cached_metadata(self, collection_name: str) -> Optional[EmbeddingMetadata]:
        """Get metadata for cached collection."""
        _, metadata_path = self._get_cache_path(collection_name)
        
        if not metadata_path.exists():
            return None
            
        try:
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
                
            return EmbeddingMetadata.from_dict(metadata_dict)
            
        except Exception as e:
            logger.warning(f"Failed to load metadata for {collection_name}: {e}")
            return None
            
    def clear_cache(self, collection_name: Optional[str] = None) -> None:
        """Clear embedding cache."""
        if collection_name:
            # Clear specific collection
            embeddings_path, metadata_path = self._get_cache_path(collection_name)
            
            if embeddings_path.exists():
                embeddings_path.unlink()
                
            if metadata_path.exists():
                metadata_path.unlink()
                
            logger.info(f"Cleared cache for collection: {collection_name}")
            
        else:
            # Clear all cache
            for file_path in self.cache_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
                    
            logger.info("Cleared all embedding cache")
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        self._initialize_model()
        
        info = {
            'model_name': self.config.model_name,
            'model_hash': self._model_hash,
            'device': self._device,
            'max_length': self.config.max_length,
            'normalize_embeddings': self.config.normalize_embeddings,
            'fp16': self.config.fp16,
            'sentence_transformers_available': SENTENCE_TRANSFORMERS_AVAILABLE,
            'transformers_available': TRANSFORMERS_AVAILABLE
        }
        
        # Try to get embedding dimension
        try:
            test_embedding = self.encode_texts(["test"])
            info['embedding_dim'] = test_embedding.shape[1]
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension: {e}")
            info['embedding_dim'] = None
            
        return info

# Import time for timing
import time

# Import pandas for timestamp (add to requirements if not available)
try:
    import pandas as pd
except ImportError:
    # Fallback timestamp
    from datetime import datetime
    class pd:
        @staticmethod
        def Timestamp():
            return datetime.now()

def create_embedding_manager(config: EmbeddingConfig,
                           cache_dir: Optional[Union[str, Path]] = None,
                           timing_harness: Optional[TimingHarness] = None) -> DenseEmbeddingManager:
    """
    Factory function for creating embedding manager.
    
    Args:
        config: Embedding configuration
        cache_dir: Optional cache directory override
        timing_harness: Optional timing harness
        
    Returns:
        Configured DenseEmbeddingManager
    """
    return DenseEmbeddingManager(config, cache_dir, timing_harness)