#!/usr/bin/env python3
"""
LongBench-v2 Dataset Loader
===========================

Loader for LongBench-v2 extended evaluation benchmark.
Covers various long-context tasks with extended evaluation protocols.

Official repo: https://github.com/THUDM/LongBench
"""

import logging
from typing import Iterator, List, Dict, Any
from .base import BaseDatasetLoader, DatasetSample

logger = logging.getLogger(__name__)


class LongBenchV2Loader(BaseDatasetLoader):
    """LongBench-v2 dataset loader."""
    
    def get_expected_fields(self) -> List[str]:
        return ["id", "context", "input", "answers", "length", "dataset", "_id"]
    
    def load_samples(self) -> Iterator[DatasetSample]:
        """Load LongBench-v2 samples."""
        for item in self._load_jsonl(self.data_path):
            # LongBench uses specific field naming
            sample_id = (item.get("_id") or 
                        item.get("id") or 
                        f"longbench_{hash(item.get('input', ''))}")
            
            # Query field 
            query = item.get("input", "")
            
            # Context field
            context = item.get("context", "")
            
            # Answer field (may be list or string)
            answers = item.get("answers", item.get("answer", ""))
            if isinstance(answers, list):
                # Take first answer for consistency  
                answer = answers[0] if answers else ""
            else:
                answer = str(answers)
            
            if not all([query.strip(), context.strip(), answer.strip()]):
                logger.warning(f"Missing required fields in LongBench sample: {sample_id}")
                continue
            
            # Dataset/task information
            dataset_name = item.get("dataset", "unknown")
            category = self._infer_category(dataset_name, query, context)
            
            # Length information
            context_length = item.get("length", len(context.split()))
            
            # Extract language if available
            language = self._infer_language(context, query)
            
            metadata = {
                "task_type": "longbench_evaluation",
                "dataset_name": dataset_name,
                "category": category,
                "language": language,
                "source": "longbench_v2",
                "original_length": item.get("length")
            }
            
            # Add all available answers if multiple
            if isinstance(item.get("answers"), list) and len(item["answers"]) > 1:
                metadata["all_answers"] = item["answers"]
            
            # Add any additional metadata
            for key, value in item.items():
                if key not in ["_id", "id", "input", "context", "answers", "answer", "length", "dataset"]:
                    metadata[f"longbench_{key}"] = value
            
            yield DatasetSample(
                id=str(sample_id),
                query=query,
                context=context,
                answer=answer,
                context_length=context_length,
                query_length=len(query.split()),
                metadata=metadata
            )
    
    def _infer_category(self, dataset_name: str, query: str, context: str) -> str:
        """Infer task category from dataset name and content."""
        dataset_lower = dataset_name.lower()
        
        if any(word in dataset_lower for word in ["qa", "question"]):
            return "question_answering"
        elif any(word in dataset_lower for word in ["summarization", "summary"]):
            return "summarization"  
        elif any(word in dataset_lower for word in ["code", "programming"]):
            return "code_understanding"
        elif any(word in dataset_lower for word in ["math", "calculation"]):
            return "mathematical_reasoning"
        elif any(word in dataset_lower for word in ["passage", "reading"]):
            return "reading_comprehension"
        elif "needle" in dataset_lower:
            return "needle_in_haystack"
        elif "classification" in dataset_lower:
            return "classification"
        else:
            return "general"
    
    def _infer_language(self, context: str, query: str) -> str:
        """Infer language from content."""
        text = (context + " " + query)[:1000].lower()  # Sample for language detection
        
        # Simple heuristic language detection
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        if chinese_chars > 10:
            return "zh"
        
        # Could add more language detection logic here
        return "en"  # Default to English