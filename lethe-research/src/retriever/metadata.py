"""
Index Metadata Management for Production IR System

Provides comprehensive metadata tracking for BM25 and ANN indices
including build parameters, content hashes, performance characteristics,
and reproducibility information.

Features:
- Index parameter persistence and validation
- Content hashing for reproducibility
- Performance metrics storage
- Recall curve data management
- Index statistics tracking
"""

import json
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np

@dataclass
class IndexStats:
    """Statistics for an index."""
    
    # Basic statistics
    num_documents: int
    num_terms: int
    total_postings: int
    avg_doc_length: float
    
    # Collection statistics
    collection_size_mb: float
    index_size_mb: float
    compression_ratio: float
    
    # Build statistics
    build_time_sec: float
    memory_used_mb: float
    cpu_time_sec: float
    
    # Additional metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class RecallCurve:
    """Recall curve data for ANN indices."""
    
    parameter_name: str  # e.g., "efSearch", "nprobe"
    parameter_values: List[Union[int, float]]
    recall_at_k: Dict[int, List[float]]  # k -> [recall values]
    latency_ms: List[float]  # latency for each parameter value
    memory_mb: List[float]  # memory usage for each parameter value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'parameter_name': self.parameter_name,
            'parameter_values': self.parameter_values,
            'recall_at_k': self.recall_at_k,
            'latency_ms': self.latency_ms,
            'memory_mb': self.memory_mb
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecallCurve':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class IndexMetadata:
    """Comprehensive metadata for an index."""
    
    # Identity
    index_type: str  # "bm25", "hnsw", "ivf_pq"
    index_name: str
    dataset_name: str
    
    # Build configuration
    build_params: Dict[str, Any]
    model_params: Optional[Dict[str, Any]] = None  # For dense indices
    
    # Content verification
    content_hash: str = ""
    parameter_hash: str = ""
    
    # Statistics
    stats: Optional[IndexStats] = None
    
    # Performance data
    recall_curves: List[RecallCurve] = field(default_factory=list)
    
    # System information
    build_environment: Dict[str, Any] = field(default_factory=dict)
    
    # Paths
    index_path: str = ""
    config_path: str = ""
    
    # Versioning
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert recall curves to dictionaries
        result['recall_curves'] = [curve.to_dict() for curve in self.recall_curves]
        return result
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndexMetadata':
        """Create from dictionary."""
        # Convert recall curves from dictionaries
        recall_curves = [RecallCurve.from_dict(curve_data) 
                        for curve_data in data.get('recall_curves', [])]
        
        # Handle stats field
        stats_data = data.get('stats')
        stats = IndexStats(**stats_data) if stats_data else None
        
        # Create metadata object
        data_copy = data.copy()
        data_copy['recall_curves'] = recall_curves
        data_copy['stats'] = stats
        
        return cls(**data_copy)

class MetadataManager:
    """
    Manages index metadata persistence and retrieval.
    
    Handles storage, loading, and validation of index metadata
    with support for content hashing and reproducibility.
    """
    
    def __init__(self, metadata_dir: Union[str, Path]):
        """
        Initialize metadata manager.
        
        Args:
            metadata_dir: Directory to store metadata files
        """
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
    def save_metadata(self, metadata: IndexMetadata) -> Path:
        """
        Save index metadata to disk.
        
        Args:
            metadata: Index metadata to save
            
        Returns:
            Path to saved metadata file
        """
        # Generate filename
        filename = f"{metadata.dataset_name}_{metadata.index_type}_{metadata.index_name}.meta"
        filepath = self.metadata_dir / filename
        
        # Update paths in metadata
        metadata.config_path = str(filepath)
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2, default=self._json_serializer)
            
        return filepath
        
    def load_metadata(self, 
                     dataset_name: str,
                     index_type: str, 
                     index_name: str) -> Optional[IndexMetadata]:
        """
        Load index metadata from disk.
        
        Args:
            dataset_name: Name of dataset
            index_type: Type of index (bm25, hnsw, ivf_pq)
            index_name: Name of specific index
            
        Returns:
            Loaded metadata or None if not found
        """
        filename = f"{dataset_name}_{index_type}_{index_name}.meta"
        filepath = self.metadata_dir / filename
        
        if not filepath.exists():
            return None
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        return IndexMetadata.from_dict(data)
        
    def list_metadata(self, 
                     dataset_name: Optional[str] = None,
                     index_type: Optional[str] = None) -> List[IndexMetadata]:
        """
        List available metadata files.
        
        Args:
            dataset_name: Filter by dataset name
            index_type: Filter by index type
            
        Returns:
            List of available metadata objects
        """
        metadata_list = []
        
        for filepath in self.metadata_dir.glob("*.meta"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                metadata = IndexMetadata.from_dict(data)
                
                # Apply filters
                if dataset_name and metadata.dataset_name != dataset_name:
                    continue
                if index_type and metadata.index_type != index_type:
                    continue
                    
                metadata_list.append(metadata)
                
            except Exception as e:
                print(f"Warning: Could not load metadata from {filepath}: {e}")
                continue
                
        return metadata_list
        
    def compute_content_hash(self, 
                           data: Any, 
                           hash_algo: str = "sha256") -> str:
        """
        Compute hash of content for reproducibility.
        
        Args:
            data: Content to hash (string, bytes, or serializable object)
            hash_algo: Hash algorithm to use
            
        Returns:
            Hex digest of hash
        """
        if isinstance(data, str):
            content = data.encode('utf-8')
        elif isinstance(data, bytes):
            content = data
        else:
            # Serialize object deterministically
            content = json.dumps(data, sort_keys=True, default=self._json_serializer).encode('utf-8')
            
        hasher = hashlib.new(hash_algo)
        hasher.update(content)
        return hasher.hexdigest()
        
    def compute_file_hash(self, filepath: Union[str, Path]) -> str:
        """Compute hash of file contents."""
        filepath = Path(filepath)
        
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
                
        return hasher.hexdigest()
        
    def validate_index_integrity(self, metadata: IndexMetadata) -> Dict[str, bool]:
        """
        Validate index integrity using stored hashes.
        
        Args:
            metadata: Index metadata to validate
            
        Returns:
            Dictionary of validation results
        """
        results = {
            'metadata_exists': bool(metadata),
            'index_path_exists': False,
            'content_hash_valid': False,
            'parameter_hash_valid': False
        }
        
        if not metadata:
            return results
            
        # Check index path exists
        if metadata.index_path and Path(metadata.index_path).exists():
            results['index_path_exists'] = True
            
        # Validate content hash if available
        if metadata.content_hash and metadata.index_path:
            try:
                current_hash = self.compute_file_hash(metadata.index_path)
                results['content_hash_valid'] = (current_hash == metadata.content_hash)
            except Exception:
                results['content_hash_valid'] = False
                
        # Validate parameter hash if available  
        if metadata.parameter_hash and metadata.build_params:
            try:
                current_hash = self.compute_content_hash(metadata.build_params)
                results['parameter_hash_valid'] = (current_hash == metadata.parameter_hash)
            except Exception:
                results['parameter_hash_valid'] = False
                
        return results
        
    def add_recall_curve(self, 
                        metadata: IndexMetadata,
                        curve: RecallCurve) -> None:
        """Add recall curve to metadata."""
        metadata.recall_curves.append(curve)
        
    def get_recall_curves(self, 
                         metadata: IndexMetadata,
                         parameter_name: Optional[str] = None) -> List[RecallCurve]:
        """Get recall curves from metadata, optionally filtered by parameter."""
        if parameter_name:
            return [curve for curve in metadata.recall_curves 
                   if curve.parameter_name == parameter_name]
        return metadata.recall_curves
        
    def export_recall_data(self, 
                          metadata: IndexMetadata,
                          output_path: Union[str, Path]) -> None:
        """Export recall curve data to CSV format."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for CSV export: pip install pandas")
        
        output_path = Path(output_path)
        
        for curve in metadata.recall_curves:
            # Create DataFrame for this curve
            data_rows = []
            
            for i, param_val in enumerate(curve.parameter_values):
                row = {
                    'parameter_name': curve.parameter_name,
                    'parameter_value': param_val,
                    'latency_ms': curve.latency_ms[i] if i < len(curve.latency_ms) else None,
                    'memory_mb': curve.memory_mb[i] if i < len(curve.memory_mb) else None
                }
                
                # Add recall@k values
                for k, recall_values in curve.recall_at_k.items():
                    if i < len(recall_values):
                        row[f'recall@{k}'] = recall_values[i]
                        
                data_rows.append(row)
                
            # Save to CSV
            df = pd.DataFrame(data_rows)
            filename = f"{metadata.index_name}_{curve.parameter_name}_recall.csv"
            df.to_csv(output_path / filename, index=False)
            
    def _json_serializer(self, obj):
        """Custom JSON serializer for special types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

class IndexRegistry:
    """
    Registry for managing multiple indices and their metadata.
    
    Provides high-level interface for index discovery and management
    across different datasets and index types.
    """
    
    def __init__(self, indices_dir: Union[str, Path]):
        """
        Initialize index registry.
        
        Args:
            indices_dir: Root directory containing indices
        """
        self.indices_dir = Path(indices_dir)
        self.metadata_managers = {}
        
    def get_metadata_manager(self, dataset_name: str) -> MetadataManager:
        """Get metadata manager for dataset."""
        if dataset_name not in self.metadata_managers:
            metadata_dir = self.indices_dir / dataset_name / "metadata"
            self.metadata_managers[dataset_name] = MetadataManager(metadata_dir)
            
        return self.metadata_managers[dataset_name]
        
    def register_index(self, metadata: IndexMetadata) -> None:
        """Register an index in the registry."""
        manager = self.get_metadata_manager(metadata.dataset_name)
        manager.save_metadata(metadata)
        
    def find_indices(self, 
                    dataset_name: Optional[str] = None,
                    index_type: Optional[str] = None) -> List[IndexMetadata]:
        """Find indices matching criteria."""
        results = []
        
        if dataset_name:
            datasets = [dataset_name]
        else:
            # Find all datasets
            datasets = [d.name for d in self.indices_dir.iterdir() 
                       if d.is_dir() and (d / "metadata").exists()]
            
        for dataset in datasets:
            try:
                manager = self.get_metadata_manager(dataset)
                indices = manager.list_metadata(dataset_name=dataset, index_type=index_type)
                results.extend(indices)
            except Exception as e:
                print(f"Warning: Error accessing dataset {dataset}: {e}")
                continue
                
        return results
        
    def validate_all_indices(self) -> Dict[str, Dict[str, bool]]:
        """Validate integrity of all registered indices."""
        validation_results = {}
        
        all_indices = self.find_indices()
        
        for metadata in all_indices:
            key = f"{metadata.dataset_name}/{metadata.index_type}/{metadata.index_name}"
            manager = self.get_metadata_manager(metadata.dataset_name)
            validation_results[key] = manager.validate_index_integrity(metadata)
            
        return validation_results