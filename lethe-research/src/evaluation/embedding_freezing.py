"""
Embedding Freezing and Pool Fingerprinting System

This module implements the embedding freezing system that ensures:
1. One embedding model across all methods
2. Baked hashes for pool/tokenizer fingerprints
3. Precomputed frozen union pool for rerankers
4. Record encoder_hash and pool_fingerprint for all methods

Key Features:
- Deterministic embedding computation and caching
- Pool fingerprinting for reproducibility
- Frozen embeddings shared across all vector-based methods
- Hash-based integrity checking

Usage:
    from evaluation.embedding_freezing import EmbeddingManager, PoolManager
    
    # Create managers
    embedding_mgr = EmbeddingManager(model_name="all-MiniLM-L6-v2")
    pool_mgr = PoolManager(embedding_manager=embedding_mgr)
    
    # Freeze embeddings for a corpus
    pool_mgr.freeze_corpus_embeddings(atoms)
    
    # Get fingerprints
    encoder_hash = embedding_mgr.get_encoder_hash()
    pool_fingerprint = pool_mgr.get_pool_fingerprint()
"""

import json
import hashlib
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field, asdict
import time

from .unified_adapter_interface import Atom, EmbeddingInterface, generate_hash

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingRecord:
    """Record of a computed embedding."""
    content_hash: str
    embedding: np.ndarray
    model_hash: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'content_hash': self.content_hash,
            'embedding': self.embedding.tolist(),
            'model_hash': self.model_hash,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingRecord':
        """Create from dictionary."""
        return cls(
            content_hash=data['content_hash'],
            embedding=np.array(data['embedding']),
            model_hash=data['model_hash'],
            timestamp=data['timestamp'],
            metadata=data.get('metadata', {})
        )

@dataclass
class PoolRecord:
    """Record of an embedding pool."""
    pool_id: str
    encoder_hash: str
    tokenizer_hash: str
    atom_hashes: List[str]
    pool_fingerprint: str
    embedding_records: List[EmbeddingRecord]
    created_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.""" 
        return {
            'pool_id': self.pool_id,
            'encoder_hash': self.encoder_hash,
            'tokenizer_hash': self.tokenizer_hash,
            'atom_hashes': self.atom_hashes,
            'pool_fingerprint': self.pool_fingerprint,
            'embedding_records': [record.to_dict() for record in self.embedding_records],
            'created_at': self.created_at,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PoolRecord':
        """Create from dictionary."""
        return cls(
            pool_id=data['pool_id'],
            encoder_hash=data['encoder_hash'],
            tokenizer_hash=data['tokenizer_hash'],
            atom_hashes=data['atom_hashes'],
            pool_fingerprint=data['pool_fingerprint'],
            embedding_records=[EmbeddingRecord.from_dict(rec) for rec in data['embedding_records']],
            created_at=data['created_at'],
            metadata=data.get('metadata', {})
        )

class DummyEmbeddingModel:
    """Dummy embedding model for testing when no real model is available."""
    
    def __init__(self, model_name: str = "dummy-model", dimension: int = 768):
        self.model_name = model_name
        self.dimension = dimension
        self._model_hash = None
        
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        embeddings = []
        for text in texts:
            # Create deterministic embedding based on text hash
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            # Use first 16 chars of hash as seed
            seed = int(text_hash[:16], 16) % (2**31)
            np.random.seed(seed)
            
            # Generate normalized random embedding
            embedding = np.random.normal(0, 1, self.dimension)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def get_hash(self) -> str:
        """Get model fingerprint hash."""
        if self._model_hash is None:
            model_data = {
                'model_name': self.model_name,
                'dimension': self.dimension,
                'type': 'dummy'
            }
            self._model_hash = generate_hash(model_data)
        return self._model_hash

class EmbeddingManager:
    """Manages embedding computation and caching."""
    
    def __init__(self, model: Optional[EmbeddingInterface] = None,
                 model_name: str = "default",
                 cache_dir: Optional[Path] = None):
        self.model = model or DummyEmbeddingModel(model_name)
        self.model_name = model_name
        self.cache_dir = cache_dir or Path("embedding_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._embedding_cache = {}  # content_hash -> embedding
        self._model_hash = None
        
        # Load cache if exists
        self._load_cache()
    
    def get_encoder_hash(self) -> str:
        """Get encoder fingerprint hash."""
        if self._model_hash is None:
            if hasattr(self.model, 'get_hash'):
                self._model_hash = self.model.get_hash()
            else:
                # Generate hash from model name and type
                model_data = {
                    'model_name': self.model_name,
                    'type': type(self.model).__name__
                }
                self._model_hash = generate_hash(model_data)
        
        return self._model_hash
    
    def compute_embeddings(self, texts: List[str], 
                          use_cache: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Compute embeddings for texts.
        
        Returns:
            Tuple of (embeddings array, content_hashes)
        """
        embeddings = []
        content_hashes = []
        texts_to_compute = []
        compute_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            content_hash = generate_hash(text)
            content_hashes.append(content_hash)
            
            if use_cache and content_hash in self._embedding_cache:
                embeddings.append(self._embedding_cache[content_hash])
            else:
                embeddings.append(None)  # Placeholder
                texts_to_compute.append(text)
                compute_indices.append(i)
        
        # Compute missing embeddings
        if texts_to_compute:
            logger.info(f"Computing {len(texts_to_compute)} new embeddings")
            new_embeddings = self.model.encode(texts_to_compute)
            
            # Store in cache and fill placeholders
            for idx, (compute_idx, text, embedding) in enumerate(zip(compute_indices, texts_to_compute, new_embeddings)):
                content_hash = content_hashes[compute_idx]
                self._embedding_cache[content_hash] = embedding
                embeddings[compute_idx] = embedding
                
                # Create embedding record
                record = EmbeddingRecord(
                    content_hash=content_hash,
                    embedding=embedding,
                    model_hash=self.get_encoder_hash(),
                    timestamp=time.time()
                )
                
                # Save to disk cache
                self._save_embedding_to_cache(record)
        
        return np.array(embeddings), content_hashes
    
    def compute_atom_embeddings(self, atoms: List[Atom]) -> List[Atom]:
        """
        Compute embeddings for atoms and update them in-place.
        
        Returns:
            Updated atoms with embeddings
        """
        texts = [atom.content for atom in atoms]
        embeddings, content_hashes = self.compute_embeddings(texts)
        
        # Update atoms with embeddings
        for i, atom in enumerate(atoms):
            atom.embedding = embeddings[i]
            if 'content_hash' not in atom.metadata:
                atom.metadata['content_hash'] = content_hashes[i]
        
        return atoms
    
    def _load_cache(self):
        """Load embedding cache from disk."""
        cache_file = self.cache_dir / f"embeddings_{self.get_encoder_hash()}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                for content_hash, embedding_data in cache_data.items():
                    self._embedding_cache[content_hash] = np.array(embedding_data['embedding'])
                
                logger.info(f"Loaded {len(self._embedding_cache)} embeddings from cache")
                
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
    
    def _save_embedding_to_cache(self, record: EmbeddingRecord):
        """Save individual embedding record to cache."""
        cache_file = self.cache_dir / f"embeddings_{self.get_encoder_hash()}.json"
        
        # Load existing cache
        cache_data = {}
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load existing cache: {e}")
        
        # Add new record
        cache_data[record.content_hash] = {
            'embedding': record.embedding.tolist(),
            'model_hash': record.model_hash,
            'timestamp': record.timestamp
        }
        
        # Save back to disk
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cached_embeddings': len(self._embedding_cache),
            'model_hash': self.get_encoder_hash(),
            'cache_dir': str(self.cache_dir)
        }

class PoolManager:
    """Manages embedding pools and fingerprinting."""
    
    def __init__(self, embedding_manager: EmbeddingManager,
                 tokenizer_hash: str = "default",
                 pool_dir: Optional[Path] = None):
        self.embedding_manager = embedding_manager
        self.tokenizer_hash = tokenizer_hash
        self.pool_dir = pool_dir or Path("pool_cache")
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        
        self._current_pool = None
        self._pool_fingerprint = None
    
    def freeze_corpus_embeddings(self, atoms: List[Atom], 
                                pool_id: Optional[str] = None) -> PoolRecord:
        """
        Freeze embeddings for a corpus of atoms.
        
        Args:
            atoms: List of atoms to freeze embeddings for
            pool_id: Optional pool identifier
            
        Returns:
            PoolRecord with frozen embeddings
        """
        # Generate pool ID if not provided
        if pool_id is None:
            atom_contents = [atom.content for atom in atoms]
            pool_data = {
                'atoms': atom_contents,
                'encoder_hash': self.embedding_manager.get_encoder_hash(),
                'tokenizer_hash': self.tokenizer_hash
            }
            pool_id = generate_hash(pool_data)[:16]
        
        # Check if pool already exists
        existing_pool = self._load_pool(pool_id)
        if existing_pool is not None:
            logger.info(f"Using existing pool: {pool_id}")
            self._current_pool = existing_pool
            return existing_pool
        
        # Compute embeddings for all atoms
        logger.info(f"Freezing embeddings for {len(atoms)} atoms")
        atoms_with_embeddings = self.embedding_manager.compute_atom_embeddings(atoms)
        
        # Create embedding records
        embedding_records = []
        atom_hashes = []
        
        for atom in atoms_with_embeddings:
            content_hash = atom.metadata.get('content_hash', generate_hash(atom.content))
            atom_hashes.append(content_hash)
            
            record = EmbeddingRecord(
                content_hash=content_hash,
                embedding=atom.embedding,
                model_hash=self.embedding_manager.get_encoder_hash(),
                timestamp=time.time(),
                metadata={'tokens': atom.tokens, 'source': atom.source}
            )
            embedding_records.append(record)
        
        # Generate pool fingerprint
        pool_fingerprint_data = {
            'pool_id': pool_id,
            'encoder_hash': self.embedding_manager.get_encoder_hash(),
            'tokenizer_hash': self.tokenizer_hash,
            'atom_hashes': sorted(atom_hashes)  # Sort for consistency
        }
        pool_fingerprint = generate_hash(pool_fingerprint_data)
        
        # Create pool record
        pool_record = PoolRecord(
            pool_id=pool_id,
            encoder_hash=self.embedding_manager.get_encoder_hash(),
            tokenizer_hash=self.tokenizer_hash,
            atom_hashes=atom_hashes,
            pool_fingerprint=pool_fingerprint,
            embedding_records=embedding_records,
            created_at=time.time(),
            metadata={'num_atoms': len(atoms)}
        )
        
        # Save pool
        self._save_pool(pool_record)
        self._current_pool = pool_record
        self._pool_fingerprint = pool_fingerprint
        
        logger.info(f"Frozen pool {pool_id} with fingerprint {pool_fingerprint[:16]}")
        return pool_record
    
    def get_pool_fingerprint(self) -> Optional[str]:
        """Get current pool fingerprint."""
        return self._pool_fingerprint
    
    def get_frozen_embeddings(self, content_hashes: List[str]) -> Optional[np.ndarray]:
        """
        Get frozen embeddings for given content hashes.
        
        Args:
            content_hashes: List of content hashes to get embeddings for
            
        Returns:
            Array of embeddings if all found, None otherwise
        """
        if self._current_pool is None:
            return None
        
        # Build lookup map
        embedding_map = {
            record.content_hash: record.embedding 
            for record in self._current_pool.embedding_records
        }
        
        # Check if all hashes are available
        embeddings = []
        for content_hash in content_hashes:
            if content_hash not in embedding_map:
                logger.warning(f"Content hash not found in frozen pool: {content_hash}")
                return None
            embeddings.append(embedding_map[content_hash])
        
        return np.array(embeddings)
    
    def create_union_pool(self, pool_ids: List[str]) -> PoolRecord:
        """
        Create a union pool from multiple existing pools.
        
        Args:
            pool_ids: List of pool IDs to union
            
        Returns:
            New PoolRecord containing union of all pools
        """
        # Load all pools
        pools = []
        for pool_id in pool_ids:
            pool = self._load_pool(pool_id)
            if pool is None:
                raise ValueError(f"Pool not found: {pool_id}")
            pools.append(pool)
        
        # Verify encoder/tokenizer compatibility
        encoder_hashes = set(pool.encoder_hash for pool in pools)
        tokenizer_hashes = set(pool.tokenizer_hash for pool in pools)
        
        if len(encoder_hashes) > 1:
            raise ValueError(f"Incompatible encoder hashes in pools: {encoder_hashes}")
        if len(tokenizer_hashes) > 1:
            raise ValueError(f"Incompatible tokenizer hashes in pools: {tokenizer_hashes}")
        
        # Merge embedding records (deduplicate by content hash)
        merged_records = {}
        merged_atom_hashes = []
        
        for pool in pools:
            for record in pool.embedding_records:
                if record.content_hash not in merged_records:
                    merged_records[record.content_hash] = record
                    merged_atom_hashes.append(record.content_hash)
        
        # Create union pool ID
        union_data = {
            'pool_ids': sorted(pool_ids),
            'encoder_hash': list(encoder_hashes)[0],
            'tokenizer_hash': list(tokenizer_hashes)[0]
        }
        union_pool_id = f"union_{generate_hash(union_data)[:16]}"
        
        # Generate union pool fingerprint
        pool_fingerprint_data = {
            'pool_id': union_pool_id,
            'encoder_hash': list(encoder_hashes)[0],
            'tokenizer_hash': list(tokenizer_hashes)[0],
            'atom_hashes': sorted(merged_atom_hashes)
        }
        union_fingerprint = generate_hash(pool_fingerprint_data)
        
        # Create union pool record
        union_pool = PoolRecord(
            pool_id=union_pool_id,
            encoder_hash=list(encoder_hashes)[0],
            tokenizer_hash=list(tokenizer_hashes)[0],
            atom_hashes=merged_atom_hashes,
            pool_fingerprint=union_fingerprint,
            embedding_records=list(merged_records.values()),
            created_at=time.time(),
            metadata={
                'source_pools': pool_ids,
                'num_atoms': len(merged_records),
                'type': 'union'
            }
        )
        
        # Save union pool
        self._save_pool(union_pool)
        
        logger.info(f"Created union pool {union_pool_id} from {len(pool_ids)} source pools")
        return union_pool
    
    def _load_pool(self, pool_id: str) -> Optional[PoolRecord]:
        """Load pool from disk."""
        pool_file = self.pool_dir / f"pool_{pool_id}.json"
        
        if pool_file.exists():
            try:
                with open(pool_file, 'r') as f:
                    pool_data = json.load(f)
                return PoolRecord.from_dict(pool_data)
            except Exception as e:
                logger.warning(f"Failed to load pool {pool_id}: {e}")
        
        return None
    
    def _save_pool(self, pool_record: PoolRecord):
        """Save pool to disk."""
        pool_file = self.pool_dir / f"pool_{pool_record.pool_id}.json"
        
        try:
            with open(pool_file, 'w') as f:
                json.dump(pool_record.to_dict(), f, indent=2, default=str)
            logger.debug(f"Saved pool to {pool_file}")
        except Exception as e:
            logger.error(f"Failed to save pool: {e}")
    
    def list_pools(self) -> List[str]:
        """List all available pool IDs."""
        pool_files = list(self.pool_dir.glob("pool_*.json"))
        pool_ids = []
        
        for pool_file in pool_files:
            pool_id = pool_file.stem.replace("pool_", "")
            pool_ids.append(pool_id)
        
        return sorted(pool_ids)
    
    def get_pool_info(self, pool_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific pool."""
        pool = self._load_pool(pool_id)
        if pool is None:
            return None
        
        return {
            'pool_id': pool.pool_id,
            'encoder_hash': pool.encoder_hash,
            'tokenizer_hash': pool.tokenizer_hash,
            'pool_fingerprint': pool.pool_fingerprint,
            'num_atoms': len(pool.atom_hashes),
            'num_embeddings': len(pool.embedding_records),
            'created_at': pool.created_at,
            'metadata': pool.metadata
        }
    
    def validate_pool_integrity(self, pool_id: str) -> Dict[str, Any]:
        """Validate integrity of a pool."""
        pool = self._load_pool(pool_id)
        if pool is None:
            return {'valid': False, 'error': 'Pool not found'}
        
        errors = []
        
        # Check that atom_hashes match embedding_records
        record_hashes = set(record.content_hash for record in pool.embedding_records)
        atom_hashes = set(pool.atom_hashes)
        
        if record_hashes != atom_hashes:
            errors.append("Mismatch between atom_hashes and embedding_records")
        
        # Check that all embeddings have same model hash
        model_hashes = set(record.model_hash for record in pool.embedding_records)
        if len(model_hashes) > 1:
            errors.append(f"Multiple model hashes in pool: {model_hashes}")
        
        # Verify pool fingerprint
        pool_fingerprint_data = {
            'pool_id': pool.pool_id,
            'encoder_hash': pool.encoder_hash,
            'tokenizer_hash': pool.tokenizer_hash,
            'atom_hashes': sorted(pool.atom_hashes)
        }
        expected_fingerprint = generate_hash(pool_fingerprint_data)
        
        if expected_fingerprint != pool.pool_fingerprint:
            errors.append("Pool fingerprint mismatch")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'pool_id': pool.pool_id,
            'fingerprint': pool.pool_fingerprint
        }

# Export main classes
__all__ = [
    'EmbeddingManager',
    'PoolManager',
    'EmbeddingRecord',
    'PoolRecord',
    'DummyEmbeddingModel'
]