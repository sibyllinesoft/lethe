#!/usr/bin/env python3
"""
Base Dataset Loading Infrastructure
===================================

Provides consistent interfaces and validation for all benchmark datasets.
"""

import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterator, Tuple
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DatasetSample:
    """A single sample from a benchmark dataset."""
    
    # Core fields (present in all datasets)
    id: str
    query: str
    context: str
    answer: str
    
    # Length information
    context_length: int
    query_length: int
    
    # Optional fields
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Compute derived fields."""
        if self.context_length <= 0:
            self.context_length = len(self.context.split())
        if self.query_length <= 0:  
            self.query_length = len(self.query.split())
    
    @property
    def total_length(self) -> int:
        """Total tokens in sample."""
        return self.context_length + self.query_length
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'query': self.query,
            'context': self.context,
            'answer': self.answer,
            'context_length': self.context_length,
            'query_length': self.query_length,
            'metadata': self.metadata
        }


@dataclass
class DatasetMetrics:
    """Statistics and metadata for a loaded dataset."""
    
    # Basic counts
    total_samples: int
    
    # Length statistics
    mean_context_length: float
    median_context_length: float
    p95_context_length: float
    p99_context_length: float
    
    mean_query_length: float
    median_query_length: float
    
    # Length distribution
    length_histogram: Dict[str, int]  # bucketed lengths
    
    # Quality metrics
    valid_samples: int
    invalid_samples: int
    validation_errors: List[str]
    
    # Source metadata
    source_name: str
    loader_version: str
    data_path: str
    
    def __post_init__(self):
        """Validate metrics."""
        if self.total_samples != self.valid_samples + self.invalid_samples:
            logger.warning(f"Sample counts inconsistent: total={self.total_samples}, valid={self.valid_samples}, invalid={self.invalid_samples}")
    
    @property
    def validation_success_rate(self) -> float:
        """Fraction of samples that passed validation."""
        if self.total_samples == 0:
            return 0.0
        return self.valid_samples / self.total_samples
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'total_samples': self.total_samples,
            'mean_context_length': self.mean_context_length,
            'median_context_length': self.median_context_length,
            'p95_context_length': self.p95_context_length,
            'p99_context_length': self.p99_context_length,
            'mean_query_length': self.mean_query_length,
            'median_query_length': self.median_query_length,
            'length_histogram': self.length_histogram,
            'valid_samples': self.valid_samples,
            'invalid_samples': self.invalid_samples,
            'validation_success_rate': self.validation_success_rate,
            'validation_errors': self.validation_errors[:10],  # Limit error list
            'source_name': self.source_name,
            'loader_version': self.loader_version,
            'data_path': self.data_path
        }


class BaseDatasetLoader(ABC):
    """Base class for all dataset loaders."""
    
    def __init__(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
        validate_samples: bool = True
    ):
        """Initialize dataset loader."""
        self.data_path = Path(data_path)
        self.max_samples = max_samples
        self.validate_samples = validate_samples
        
        # Loader metadata
        self.loader_name = self.__class__.__name__
        self.loader_version = "1.0.0"
        
        # Validation state
        self._validation_errors: List[str] = []
        self._samples_processed = 0
        self._samples_valid = 0
    
    @abstractmethod
    def load_samples(self) -> Iterator[DatasetSample]:
        """Load dataset samples. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_expected_fields(self) -> List[str]:
        """Get list of expected fields in raw data."""
        pass
    
    def load(self) -> Tuple[List[DatasetSample], DatasetMetrics]:
        """Load complete dataset with validation and metrics."""
        logger.info(f"Loading dataset from {self.data_path}")
        
        samples = []
        context_lengths = []
        query_lengths = []
        
        # Load and validate samples
        for sample in self.load_samples():
            self._samples_processed += 1
            
            if self.validate_samples and not self._validate_sample(sample):
                continue
                
            samples.append(sample)
            context_lengths.append(sample.context_length)
            query_lengths.append(sample.query_length)
            
            self._samples_valid += 1
            
            # Check max samples limit
            if self.max_samples and len(samples) >= self.max_samples:
                logger.info(f"Reached max samples limit: {self.max_samples}")
                break
        
        if not samples:
            raise ValueError(f"No valid samples loaded from {self.data_path}")
        
        # Compute statistics
        metrics = self._compute_metrics(samples, context_lengths, query_lengths)
        
        logger.info(
            f"Loaded {len(samples)} samples from {self.data_path} "
            f"(validation rate: {metrics.validation_success_rate:.1%})"
        )
        
        return samples, metrics
    
    def _validate_sample(self, sample: DatasetSample) -> bool:
        """Validate a single sample."""
        errors = []
        
        # Check required fields
        if not sample.id:
            errors.append("Missing sample ID")
        if not sample.query.strip():
            errors.append("Empty query") 
        if not sample.context.strip():
            errors.append("Empty context")
        if not sample.answer.strip():
            errors.append("Empty answer")
            
        # Check length consistency
        if sample.context_length <= 0:
            errors.append(f"Invalid context length: {sample.context_length}")
        if sample.query_length <= 0:
            errors.append(f"Invalid query length: {sample.query_length}")
        
        # Check reasonable bounds
        if sample.context_length > 1_000_000:  # 1M tokens seems unreasonable
            errors.append(f"Context too long: {sample.context_length} tokens")
        if sample.query_length > 10_000:  # 10K token queries seem unreasonable  
            errors.append(f"Query too long: {sample.query_length} tokens")
        
        # Store errors
        if errors:
            error_msg = f"Sample {sample.id}: " + "; ".join(errors)
            self._validation_errors.append(error_msg)
            return False
            
        return True
    
    def _compute_metrics(
        self,
        samples: List[DatasetSample],
        context_lengths: List[int],
        query_lengths: List[int]
    ) -> DatasetMetrics:
        """Compute dataset statistics."""
        
        # Length statistics
        ctx_array = np.array(context_lengths)
        query_array = np.array(query_lengths)
        
        # Create length histogram (log-scale buckets)
        length_histogram = {}
        for length in context_lengths:
            bucket = self._get_length_bucket(length)
            length_histogram[bucket] = length_histogram.get(bucket, 0) + 1
        
        return DatasetMetrics(
            total_samples=self._samples_processed,
            mean_context_length=float(np.mean(ctx_array)),
            median_context_length=float(np.median(ctx_array)),
            p95_context_length=float(np.percentile(ctx_array, 95)),
            p99_context_length=float(np.percentile(ctx_array, 99)),
            mean_query_length=float(np.mean(query_array)),
            median_query_length=float(np.median(query_array)),
            length_histogram=length_histogram,
            valid_samples=self._samples_valid,
            invalid_samples=self._samples_processed - self._samples_valid,
            validation_errors=self._validation_errors.copy(),
            source_name=self.data_path.stem,
            loader_version=self.loader_version,
            data_path=str(self.data_path)
        )
    
    def _get_length_bucket(self, length: int) -> str:
        """Get histogram bucket for a given length."""
        if length < 1000:
            return "0-1k"
        elif length < 5000:
            return "1k-5k"
        elif length < 10000:
            return "5k-10k"
        elif length < 50000:
            return "10k-50k" 
        elif length < 100000:
            return "50k-100k"
        else:
            return "100k+"
    
    def _load_jsonl(self, path: Path) -> Iterator[Dict[str, Any]]:
        """Helper to load JSONL files."""
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num} in {path}: {e}")
                    continue
    
    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Helper to load JSON files."""
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_info(self) -> Dict[str, Any]:
        """Get loader information."""
        return {
            'loader_name': self.loader_name,
            'loader_version': self.loader_version,
            'data_path': str(self.data_path),
            'max_samples': self.max_samples,
            'validate_samples': self.validate_samples,
            'expected_fields': self.get_expected_fields()
        }