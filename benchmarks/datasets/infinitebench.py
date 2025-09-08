#!/usr/bin/env python3
"""
InfiniteBench Dataset Loaders
=============================

Official loaders for all InfiniteBench tasks:
- Zh.QA (multilingual QA)
- Retrieve.PassKey (needle-in-haystack)
- Retrieve.KV (key-value retrieval) 
- Retrieve.Number (number retrieval)
- Code.Debug (repository-scale debugging)
- Code.QA (code understanding)
- En.QA (English QA)

Uses official data formats and statistics for transparency.
"""

import logging
import re
from typing import Iterator, List, Dict, Any
from .base import BaseDatasetLoader, DatasetSample

logger = logging.getLogger(__name__)


class ZhQALoader(BaseDatasetLoader):
    """InfiniteBench Chinese QA dataset loader."""
    
    def get_expected_fields(self) -> List[str]:
        return ["id", "context", "input", "answer", "length"]
    
    def load_samples(self) -> Iterator[DatasetSample]:
        """Load Chinese QA samples."""
        for item in self._load_jsonl(self.data_path):
            # Validate expected fields
            required_fields = ["context", "input", "answer"]
            if not all(field in item for field in required_fields):
                logger.warning(f"Missing required fields in sample: {item.keys()}")
                continue
            
            # Extract fields with InfiniteBench naming
            sample_id = item.get("id", f"zh_qa_{hash(item['input'])}")
            context = item["context"]
            query = item["input"]  # InfiniteBench uses "input" for query
            answer = item["answer"]
            
            # Use provided length or compute
            context_length = item.get("length", len(context.split()))
            
            yield DatasetSample(
                id=str(sample_id),
                query=query,
                context=context,
                answer=answer,
                context_length=context_length,
                query_length=len(query.split()),
                metadata={
                    "task_type": "multilingual_qa",
                    "language": "zh",
                    "source": "infinitebench",
                    "original_length": item.get("length")
                }
            )


class RetrievePasskeyLoader(BaseDatasetLoader):
    """InfiniteBench passkey retrieval (needle-in-haystack) loader."""
    
    def get_expected_fields(self) -> List[str]:
        return ["id", "context", "input", "answer", "length"]
    
    def load_samples(self) -> Iterator[DatasetSample]:
        """Load passkey retrieval samples."""
        for item in self._load_jsonl(self.data_path):
            required_fields = ["context", "input", "answer"]
            if not all(field in item for field in required_fields):
                continue
            
            sample_id = item.get("id", f"passkey_{hash(item['input'])}")
            context = item["context"]
            query = item["input"]
            answer = item["answer"]
            
            # Extract passkey from context for validation
            passkey_match = re.search(r'The pass key is (\d+)', context)
            expected_passkey = passkey_match.group(1) if passkey_match else None
            
            context_length = item.get("length", len(context.split()))
            
            yield DatasetSample(
                id=str(sample_id),
                query=query,
                context=context,
                answer=answer,
                context_length=context_length,
                query_length=len(query.split()),
                metadata={
                    "task_type": "needle_in_haystack",
                    "passkey": expected_passkey,
                    "source": "infinitebench",
                    "original_length": item.get("length")
                }
            )


class RetrieveKVLoader(BaseDatasetLoader):
    """InfiniteBench key-value retrieval loader."""
    
    def get_expected_fields(self) -> List[str]:
        return ["id", "context", "input", "answer", "length"]
    
    def load_samples(self) -> Iterator[DatasetSample]:
        """Load key-value retrieval samples.""" 
        for item in self._load_jsonl(self.data_path):
            required_fields = ["context", "input", "answer"]
            if not all(field in item for field in required_fields):
                continue
                
            sample_id = item.get("id", f"kv_{hash(item['input'])}")
            context = item["context"]
            query = item["input"]
            answer = item["answer"]
            
            # Extract key from query for metadata
            key_match = re.search(r'key is "([^"]+)"', query)
            lookup_key = key_match.group(1) if key_match else None
            
            context_length = item.get("length", len(context.split()))
            
            yield DatasetSample(
                id=str(sample_id),
                query=query,
                context=context,
                answer=answer,
                context_length=context_length,
                query_length=len(query.split()),
                metadata={
                    "task_type": "key_value_retrieval",
                    "lookup_key": lookup_key,
                    "source": "infinitebench",
                    "original_length": item.get("length")
                }
            )


class RetrieveNumberLoader(BaseDatasetLoader):
    """InfiniteBench number retrieval loader."""
    
    def get_expected_fields(self) -> List[str]:
        return ["id", "context", "input", "answer", "length"]
    
    def load_samples(self) -> Iterator[DatasetSample]:
        """Load number retrieval samples."""
        for item in self._load_jsonl(self.data_path):
            required_fields = ["context", "input", "answer"]
            if not all(field in item for field in required_fields):
                continue
                
            sample_id = item.get("id", f"number_{hash(item['input'])}")
            context = item["context"] 
            query = item["input"]
            answer = item["answer"]
            
            # Extract target number from query
            number_match = re.search(r'What is the (\d+)\w* number', query)
            target_position = number_match.group(1) if number_match else None
            
            context_length = item.get("length", len(context.split()))
            
            yield DatasetSample(
                id=str(sample_id),
                query=query,
                context=context,
                answer=answer,
                context_length=context_length,
                query_length=len(query.split()),
                metadata={
                    "task_type": "number_retrieval",
                    "target_position": target_position,
                    "source": "infinitebench",
                    "original_length": item.get("length")
                }
            )


class CodeDebugLoader(BaseDatasetLoader):
    """InfiniteBench code debugging loader."""
    
    def get_expected_fields(self) -> List[str]:
        return ["id", "context", "input", "answer", "length"]
    
    def load_samples(self) -> Iterator[DatasetSample]:
        """Load code debugging samples."""
        for item in self._load_jsonl(self.data_path):
            required_fields = ["context", "input", "answer"]
            if not all(field in item for field in required_fields):
                continue
                
            sample_id = item.get("id", f"code_debug_{hash(item['input'])}")
            context = item["context"]  # Full repository context
            query = item["input"]      # Debug question
            answer = item["answer"]    # Expected fix/explanation
            
            # Extract programming language if available
            lang_indicators = {
                'python': ['.py', 'def ', 'import ', 'print('],
                'javascript': ['.js', 'function ', 'const ', 'console.log'],
                'java': ['.java', 'public class', 'System.out'],
                'cpp': ['.cpp', '#include', 'std::', 'cout <<'],
                'rust': ['.rs', 'fn ', 'let ', 'println!']
            }
            
            detected_lang = None
            context_lower = context.lower()
            for lang, indicators in lang_indicators.items():
                if any(indicator in context_lower for indicator in indicators):
                    detected_lang = lang
                    break
            
            context_length = item.get("length", len(context.split()))
            
            yield DatasetSample(
                id=str(sample_id),
                query=query,
                context=context,
                answer=answer,
                context_length=context_length,
                query_length=len(query.split()),
                metadata={
                    "task_type": "code_debugging",
                    "programming_language": detected_lang,
                    "source": "infinitebench",
                    "original_length": item.get("length")
                }
            )


class CodeQALoader(BaseDatasetLoader):
    """InfiniteBench code QA loader."""
    
    def get_expected_fields(self) -> List[str]:
        return ["id", "context", "input", "answer", "length"]
    
    def load_samples(self) -> Iterator[DatasetSample]:
        """Load code QA samples."""
        for item in self._load_jsonl(self.data_path):
            required_fields = ["context", "input", "answer"]
            if not all(field in item for field in required_fields):
                continue
                
            sample_id = item.get("id", f"code_qa_{hash(item['input'])}")
            context = item["context"]
            query = item["input"]
            answer = item["answer"]
            
            # Count code files in context (rough heuristic)
            file_patterns = [r'# File: ', r'// File: ', r'/\* File: ']
            file_count = sum(len(re.findall(pattern, context)) for pattern in file_patterns)
            
            context_length = item.get("length", len(context.split()))
            
            yield DatasetSample(
                id=str(sample_id),
                query=query,
                context=context,
                answer=answer,
                context_length=context_length,
                query_length=len(query.split()),
                metadata={
                    "task_type": "code_qa",
                    "estimated_file_count": file_count,
                    "source": "infinitebench", 
                    "original_length": item.get("length")
                }
            )


class EnQALoader(BaseDatasetLoader):
    """InfiniteBench English QA loader."""
    
    def get_expected_fields(self) -> List[str]:
        return ["id", "context", "input", "answer", "length"]
    
    def load_samples(self) -> Iterator[DatasetSample]:
        """Load English QA samples."""
        for item in self._load_jsonl(self.data_path):
            required_fields = ["context", "input", "answer"]
            if not all(field in item for field in required_fields):
                continue
                
            sample_id = item.get("id", f"en_qa_{hash(item['input'])}")
            context = item["context"]
            query = item["input"]
            answer = item["answer"]
            
            # Detect document type (novel, article, etc.)
            doc_type = "unknown"
            context_start = context.lower()[:500]
            if "chapter" in context_start:
                doc_type = "novel"
            elif "abstract:" in context_start or "introduction:" in context_start:
                doc_type = "academic_paper"
            elif "article" in context_start or "news" in context_start:
                doc_type = "news_article"
            
            context_length = item.get("length", len(context.split()))
            
            yield DatasetSample(
                id=str(sample_id),
                query=query,
                context=context,
                answer=answer,
                context_length=context_length,
                query_length=len(query.split()),
                metadata={
                    "task_type": "english_qa",
                    "document_type": doc_type,
                    "language": "en",
                    "source": "infinitebench",
                    "original_length": item.get("length")
                }
            )


# Export all loaders
__all__ = [
    "ZhQALoader",
    "RetrievePasskeyLoader", 
    "RetrieveKVLoader",
    "RetrieveNumberLoader",
    "CodeDebugLoader",
    "CodeQALoader",
    "EnQALoader"
]