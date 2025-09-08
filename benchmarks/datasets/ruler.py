#!/usr/bin/env python3
"""
RULER Dataset Loader  
====================

Loader for NVIDIA's RULER benchmark for long-context evaluation:
- Retrieval tasks (single and multi-hop)
- Variable-tracking and aggregation
- Length sweep validation  
- Multi-hop tracing

Official repo: https://github.com/NVIDIA/RULER
"""

import logging
from typing import Iterator, List, Dict, Any
from .base import BaseDatasetLoader, DatasetSample

logger = logging.getLogger(__name__)


class RulerLoader(BaseDatasetLoader):
    """RULER benchmark dataset loader."""
    
    def get_expected_fields(self) -> List[str]:
        return ["task", "query", "context", "answer", "length", "subtask_type"]
    
    def load_samples(self) -> Iterator[DatasetSample]:
        """Load RULER evaluation samples.""" 
        for item in self._load_jsonl(self.data_path):
            # RULER has varied field names, normalize them
            task_name = item.get("task", "unknown")
            subtask = item.get("subtask_type", item.get("subtype", "unknown"))
            
            # Try different query field names
            query = (item.get("query") or 
                    item.get("question") or 
                    item.get("input") or "")
            
            # Try different context field names  
            context = (item.get("context") or
                      item.get("passage") or
                      item.get("document") or "")
            
            # Try different answer field names
            answer = (item.get("answer") or
                     item.get("target") or 
                     item.get("expected") or "")
            
            if not all([query.strip(), context.strip(), answer.strip()]):
                logger.warning(f"Missing required fields in RULER sample: {item.keys()}")
                continue
            
            # Generate sample ID
            sample_id = (item.get("id") or 
                        f"ruler_{task_name}_{subtask}_{hash(query)}")
            
            # Get or compute length
            context_length = item.get("length", len(context.split()))
            
            # Extract task-specific metadata
            metadata = {
                "task_type": "ruler_evaluation",
                "ruler_task": task_name,
                "subtask_type": subtask,
                "source": "ruler",
                "original_length": item.get("length")
            }
            
            # Add task-specific fields
            if "needle" in task_name.lower():
                metadata["task_category"] = "needle_in_haystack"
                # Extract needle information if present
                if "needle" in item:
                    metadata["needle_content"] = item["needle"]
            
            elif "multi_hop" in task_name.lower():
                metadata["task_category"] = "multi_hop_reasoning"
                if "hops" in item:
                    metadata["hop_count"] = item["hops"]
            
            elif "variable_tracking" in task_name.lower():
                metadata["task_category"] = "variable_tracking"
                if "variables" in item:
                    metadata["variable_count"] = len(item["variables"])
            
            elif "aggregation" in task_name.lower():
                metadata["task_category"] = "aggregation"
                if "operation" in item:
                    metadata["aggregation_operation"] = item["operation"]
            
            else:
                metadata["task_category"] = "other"
            
            # Add any additional metadata from the original item
            for key, value in item.items():
                if key not in ["task", "query", "question", "input", 
                              "context", "passage", "document",
                              "answer", "target", "expected", "id", "length"]:
                    metadata[f"ruler_{key}"] = value
            
            yield DatasetSample(
                id=str(sample_id),
                query=query,
                context=context,
                answer=answer,
                context_length=context_length,
                query_length=len(query.split()),
                metadata=metadata
            )