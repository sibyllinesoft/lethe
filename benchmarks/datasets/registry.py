#!/usr/bin/env python3
"""
Dataset Registry
================

Central registry for all benchmark datasets with automatic loader discovery
and validation.
"""

import logging
from typing import Dict, List, Type, Optional, Tuple, Any
from pathlib import Path
from .base import BaseDatasetLoader, DatasetSample, DatasetMetrics
from .infinitebench import *
from .ruler import RulerLoader
from .longbench import LongBenchV2Loader
from .babilong import BABILongLoader

logger = logging.getLogger(__name__)


class DatasetRegistry:
    """Registry for all available dataset loaders."""
    
    def __init__(self):
        """Initialize dataset registry with all available loaders."""
        self._loaders: Dict[str, Type[BaseDatasetLoader]] = {}
        self._register_default_loaders()
    
    def _register_default_loaders(self):
        """Register all built-in dataset loaders."""
        # InfiniteBench loaders
        self.register("infinitebench_zh_qa", ZhQALoader)
        self.register("infinitebench_retrieve_passkey", RetrievePasskeyLoader)
        self.register("infinitebench_retrieve_kv", RetrieveKVLoader)
        self.register("infinitebench_retrieve_number", RetrieveNumberLoader)
        self.register("infinitebench_code_debug", CodeDebugLoader)
        self.register("infinitebench_code_qa", CodeQALoader)
        self.register("infinitebench_en_qa", EnQALoader)
        
        # External loaders
        self.register("ruler", RulerLoader)
        self.register("longbench_v2", LongBenchV2Loader)
        self.register("babilong", BABILongLoader)
        
        logger.info(f"Registered {len(self._loaders)} dataset loaders")
    
    def register(self, name: str, loader_class: Type[BaseDatasetLoader]):
        """Register a dataset loader."""
        if not issubclass(loader_class, BaseDatasetLoader):
            raise ValueError(f"Loader class must inherit from BaseDatasetLoader")
        
        self._loaders[name] = loader_class
        logger.debug(f"Registered dataset loader: {name}")
    
    def get_loader(
        self,
        name: str,
        data_path: str,
        max_samples: Optional[int] = None,
        validate_samples: bool = True
    ) -> BaseDatasetLoader:
        """Get a configured dataset loader instance."""
        if name not in self._loaders:
            raise ValueError(f"Unknown dataset: {name}. Available: {self.list_datasets()}")
        
        loader_class = self._loaders[name]
        return loader_class(
            data_path=data_path,
            max_samples=max_samples,
            validate_samples=validate_samples
        )
    
    def load_dataset(
        self,
        name: str,
        data_path: str,
        max_samples: Optional[int] = None,
        validate_samples: bool = True
    ) -> Tuple[List[DatasetSample], DatasetMetrics]:
        """Load a complete dataset."""
        loader = self.get_loader(name, data_path, max_samples, validate_samples)
        return loader.load()
    
    def list_datasets(self) -> List[str]:
        """List all registered dataset names."""
        return sorted(self._loaders.keys())
    
    def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """Get information about a dataset loader.""" 
        if name not in self._loaders:
            raise ValueError(f"Unknown dataset: {name}")
        
        loader_class = self._loaders[name]
        
        # Create a temporary instance to get info
        try:
            temp_loader = loader_class(data_path="/tmp/dummy")
            info = temp_loader.get_info()
            info["class_name"] = loader_class.__name__
            info["module"] = loader_class.__module__
            return info
        except Exception as e:
            logger.warning(f"Could not get info for {name}: {e}")
            return {
                "class_name": loader_class.__name__,
                "module": loader_class.__module__,
                "error": str(e)
            }
    
    def list_by_source(self) -> Dict[str, List[str]]:
        """Group datasets by source."""
        sources = {}
        for name in self._loaders:
            if "infinitebench" in name:
                source = "infinitebench"
            elif name == "ruler":
                source = "ruler" 
            elif name == "longbench_v2":
                source = "longbench"
            elif name == "babilong":
                source = "babilong"
            else:
                source = "other"
            
            if source not in sources:
                sources[source] = []
            sources[source].append(name)
        
        return sources
    
    def validate_data_paths(self, data_paths: Dict[str, str]) -> Dict[str, bool]:
        """Validate that data paths exist for given datasets."""
        results = {}
        for dataset_name, data_path in data_paths.items():
            if dataset_name not in self._loaders:
                results[dataset_name] = False
                continue
            
            path = Path(data_path)
            results[dataset_name] = path.exists() and path.is_file()
        
        return results
    
    def get_length_statistics(
        self,
        datasets: Dict[str, str],
        max_samples_per_dataset: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """Get length statistics for multiple datasets (for budgeting)."""
        stats = {}
        
        for name, data_path in datasets.items():
            try:
                logger.info(f"Computing length statistics for {name}")
                _, metrics = self.load_dataset(
                    name=name,
                    data_path=data_path,
                    max_samples=max_samples_per_dataset,
                    validate_samples=True
                )
                
                stats[name] = {
                    "mean_context_length": metrics.mean_context_length,
                    "median_context_length": metrics.median_context_length,
                    "p95_context_length": metrics.p95_context_length,
                    "p99_context_length": metrics.p99_context_length,
                    "total_samples": metrics.total_samples,
                    "validation_rate": metrics.validation_success_rate
                }
            except Exception as e:
                logger.error(f"Failed to compute stats for {name}: {e}")
                stats[name] = {"error": str(e)}
        
        return stats


# Global registry instance
_registry = DatasetRegistry()

def get_dataset_registry() -> DatasetRegistry:
    """Get the global dataset registry."""
    return _registry

def load_dataset(
    name: str,
    data_path: str,
    max_samples: Optional[int] = None,
    validate_samples: bool = True
) -> Tuple[List[DatasetSample], DatasetMetrics]:
    """Convenience function to load a dataset."""
    return _registry.load_dataset(name, data_path, max_samples, validate_samples)

def list_available_datasets() -> List[str]:
    """List all available datasets."""
    return _registry.list_datasets()