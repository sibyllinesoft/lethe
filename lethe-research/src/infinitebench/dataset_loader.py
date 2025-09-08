"""
InfiniteBench Dataset Loader
===========================

Comprehensive dataset loading and preprocessing utilities for the InfiniteBench
long-context evaluation dataset. Supports all 12 task types with proper 
train/test splits and preprocessing.

Author: Lethe Research Team
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
import random
from abc import ABC, abstractmethod
import tiktoken

logger = logging.getLogger(__name__)

@dataclass
class InfiniteBenchSample:
    """Single sample from InfiniteBench dataset."""
    
    id: Union[int, str]
    context: str
    question: Optional[str] = None
    answer: Optional[str] = None
    options: Optional[List[str]] = None
    task_type: Optional[str] = None
    length: Optional[int] = None
    
    def __post_init__(self):
        """Calculate context length if not provided."""
        if self.length is None:
            # Use tiktoken for accurate token counting
            encoding = tiktoken.get_encoding("cl100k_base")
            self.length = len(encoding.encode(self.context))

@dataclass
class TaskMetadata:
    """Metadata for each InfiniteBench task."""
    
    name: str
    description: str
    metric: str
    num_samples: int
    avg_length: float
    task_type: str  # retrieval, code, math, novel, dialogue
    language: str   # en, zh
    eval_method: str  # exact_match, rouge_l, accuracy

class InfiniteBenchLoader:
    """
    Comprehensive loader for InfiniteBench dataset with support for all 12 tasks.
    
    Features:
    - Automatic download and caching
    - Proper train/test splits
    - Task-specific preprocessing
    - Token counting and length analysis
    - Statistical reporting
    """
    
    TASK_CONFIGS = {
        "passkey": TaskMetadata(
            name="passkey",
            description="Passkey retrieval in long context",
            metric="accuracy",
            num_samples=590,
            avg_length=122900,
            task_type="retrieval",
            language="en",
            eval_method="exact_match"
        ),
        "number_string": TaskMetadata(
            name="number_string", 
            description="Number string locating task",
            metric="accuracy",
            num_samples=590,
            avg_length=122900,
            task_type="retrieval",
            language="en", 
            eval_method="exact_match"
        ),
        "kv_retrieval": TaskMetadata(
            name="kv_retrieval",
            description="Key-value retrieval from JSON",
            metric="accuracy", 
            num_samples=500,
            avg_length=89000,
            task_type="retrieval",
            language="en",
            eval_method="exact_match"
        ),
        "longbook_sum_eng": TaskMetadata(
            name="longbook_sum_eng",
            description="Long book summarization (English)",
            metric="rouge_l",
            num_samples=103,
            avg_length=171500,
            task_type="novel",
            language="en", 
            eval_method="rouge_l"
        ),
        "longbook_choice_eng": TaskMetadata(
            name="longbook_choice_eng", 
            description="Long book multiple choice (English)",
            metric="accuracy",
            num_samples=229,
            avg_length=171500,
            task_type="novel",
            language="en",
            eval_method="exact_match"
        ),
        "longbook_qa_eng": TaskMetadata(
            name="longbook_qa_eng",
            description="Long book Q&A (English)",
            metric="f1", 
            num_samples=351,
            avg_length=171500,
            task_type="novel",
            language="en",
            eval_method="exact_match"
        ),
        "longbook_qa_chn": TaskMetadata(
            name="longbook_qa_chn",
            description="Long book Q&A (Chinese)",
            metric="f1",
            num_samples=189, 
            avg_length=171500,
            task_type="novel",
            language="zh",
            eval_method="exact_match"
        ),
        "longdialogue_qa_eng": TaskMetadata(
            name="longdialogue_qa_eng",
            description="Long dialogue Q&A (English)", 
            metric="f1",
            num_samples=200,
            avg_length=110000,
            task_type="dialogue",
            language="en",
            eval_method="exact_match"
        ),
        "code_debug": TaskMetadata(
            name="code_debug",
            description="Code debugging task",
            metric="accuracy",
            num_samples=394, 
            avg_length=114200,
            task_type="code",
            language="en",
            eval_method="exact_match"
        ),
        "code_run": TaskMetadata(
            name="code_run",
            description="Code execution task",
            metric="accuracy",
            num_samples=400,
            avg_length=75000,
            task_type="code", 
            language="en",
            eval_method="exact_match"
        ),
        "math_calc": TaskMetadata(
            name="math_calc",
            description="Mathematical calculation", 
            metric="accuracy",
            num_samples=400,
            avg_length=190000,
            task_type="math",
            language="en",
            eval_method="exact_match"
        ),
        "math_find": TaskMetadata(
            name="math_find",
            description="Mathematical finding task",
            metric="accuracy",
            num_samples=400,
            avg_length=190000, 
            task_type="math",
            language="en",
            eval_method="exact_match"
        )
    }
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the InfiniteBench loader.
        
        Args:
            data_dir: Directory containing the InfiniteBench data files
        """
        self.data_dir = Path(data_dir)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def load_task(self, task_name: str, split: str = "all") -> List[InfiniteBenchSample]:
        """
        Load a specific task from InfiniteBench.
        
        Args:
            task_name: Name of the task (e.g., 'kv_retrieval', 'longbook_qa_eng')
            split: Data split to load ('all', 'train', 'test', 'dev')
            
        Returns:
            List of InfiniteBenchSample objects
        """
        if task_name not in self.TASK_CONFIGS:
            raise ValueError(f"Unknown task: {task_name}. Available tasks: {list(self.TASK_CONFIGS.keys())}")
        
        data_file = self.data_dir / f"{task_name}.jsonl"
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        logger.info(f"Loading task '{task_name}' from {data_file}")
        
        samples = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Create sample with task-specific processing
                    sample = self._process_sample(data, task_name)
                    samples.append(sample)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON on line {line_num}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
        
        logger.info(f"Loaded {len(samples)} samples for task '{task_name}'")
        
        # Apply split if requested
        if split != "all":
            samples = self._apply_split(samples, split)
        
        return samples
    
    def _process_sample(self, data: Dict[str, Any], task_name: str) -> InfiniteBenchSample:
        """Process raw JSON data into InfiniteBenchSample."""
        
        # Extract common fields
        sample_id = data.get('id', data.get('index', ''))
        context = data.get('context', data.get('input', ''))
        answer = data.get('answer', data.get('output', data.get('target', '')))
        
        # Task-specific processing
        if task_name in ['longbook_choice_eng']:
            # Multiple choice tasks
            options = data.get('options', [])
            question = data.get('question', '')
        elif task_name in ['longbook_qa_eng', 'longbook_qa_chn', 'longdialogue_qa_eng']:
            # Q&A tasks
            question = data.get('question', data.get('input', ''))
            options = None
        else:
            # Other tasks (retrieval, code, math)
            question = data.get('input', data.get('question', ''))
            options = None
        
        return InfiniteBenchSample(
            id=sample_id,
            context=context,
            question=question,
            answer=str(answer) if answer is not None else None,
            options=options,
            task_type=task_name,
            length=None  # Will be calculated in __post_init__
        )
    
    def _apply_split(self, samples: List[InfiniteBenchSample], split: str) -> List[InfiniteBenchSample]:
        """Apply train/test/dev split to samples."""
        
        # Use deterministic split based on sample IDs for reproducibility
        random.seed(42)
        shuffled_samples = samples.copy()
        random.shuffle(shuffled_samples)
        
        total = len(shuffled_samples)
        
        if split == "train":
            return shuffled_samples[:int(0.7 * total)]
        elif split == "test":  
            return shuffled_samples[int(0.7 * total):int(0.9 * total)]
        elif split == "dev":
            return shuffled_samples[int(0.9 * total):]
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def load_all_tasks(self, split: str = "all") -> Dict[str, List[InfiniteBenchSample]]:
        """
        Load all available tasks.
        
        Args:
            split: Data split to load for all tasks
            
        Returns:
            Dictionary mapping task names to sample lists
        """
        all_tasks = {}
        
        for task_name in self.TASK_CONFIGS.keys():
            try:
                samples = self.load_task(task_name, split)
                all_tasks[task_name] = samples
            except Exception as e:
                logger.warning(f"Failed to load task '{task_name}': {e}")
        
        return all_tasks
    
    def get_task_metadata(self, task_name: str) -> TaskMetadata:
        """Get metadata for a specific task."""
        if task_name not in self.TASK_CONFIGS:
            raise ValueError(f"Unknown task: {task_name}")
        return self.TASK_CONFIGS[task_name]
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive dataset statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            "total_tasks": len(self.TASK_CONFIGS),
            "tasks_by_type": {},
            "tasks_by_language": {},
            "tasks_by_metric": {},
            "total_samples": 0,
            "avg_length_by_task": {},
            "length_distribution": {}
        }
        
        # Aggregate statistics from metadata
        for task_name, metadata in self.TASK_CONFIGS.items():
            # By task type
            if metadata.task_type not in stats["tasks_by_type"]:
                stats["tasks_by_type"][metadata.task_type] = []
            stats["tasks_by_type"][metadata.task_type].append(task_name)
            
            # By language
            if metadata.language not in stats["tasks_by_language"]:
                stats["tasks_by_language"][metadata.language] = []
            stats["tasks_by_language"][metadata.language].append(task_name)
            
            # By metric
            if metadata.metric not in stats["tasks_by_metric"]:
                stats["tasks_by_metric"][metadata.metric] = []
            stats["tasks_by_metric"][metadata.metric].append(task_name)
            
            # Sample counts and lengths
            stats["total_samples"] += metadata.num_samples
            stats["avg_length_by_task"][task_name] = metadata.avg_length
        
        return stats
    
    def create_evaluation_subset(self, 
                               max_samples_per_task: int = 50,
                               min_length: int = 50000,
                               max_length: int = 200000) -> Dict[str, List[InfiniteBenchSample]]:
        """
        Create a smaller evaluation subset for faster experimentation.
        
        Args:
            max_samples_per_task: Maximum samples per task
            min_length: Minimum context length
            max_length: Maximum context length
            
        Returns:
            Dictionary of filtered task samples
        """
        subset = {}
        
        for task_name in self.TASK_CONFIGS.keys():
            try:
                samples = self.load_task(task_name)
                
                # Filter by length
                filtered = [
                    s for s in samples 
                    if min_length <= s.length <= max_length
                ]
                
                # Limit number of samples
                if len(filtered) > max_samples_per_task:
                    random.seed(42)
                    filtered = random.sample(filtered, max_samples_per_task)
                
                subset[task_name] = filtered
                logger.info(f"Task '{task_name}': {len(filtered)}/{len(samples)} samples in subset")
                
            except Exception as e:
                logger.warning(f"Failed to create subset for task '{task_name}': {e}")
        
        return subset

def main():
    """Example usage of InfiniteBenchLoader."""
    
    # Initialize loader
    data_dir = Path("benchmarks/infinitebench/data")
    loader = InfiniteBenchLoader(data_dir)
    
    # Print dataset statistics
    stats = loader.get_dataset_statistics()
    print("Dataset Statistics:")
    print(f"Total tasks: {stats['total_tasks']}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Tasks by type: {stats['tasks_by_type']}")
    print(f"Tasks by language: {stats['tasks_by_language']}")
    
    # Load a specific task
    kv_samples = loader.load_task("kv_retrieval")
    print(f"\nKV Retrieval task: {len(kv_samples)} samples")
    if kv_samples:
        sample = kv_samples[0]
        print(f"Sample ID: {sample.id}")
        print(f"Context length: {sample.length} tokens")
        print(f"Answer: {sample.answer}")
    
    # Create evaluation subset
    subset = loader.create_evaluation_subset(max_samples_per_task=10)
    print(f"\nEvaluation subset created with {sum(len(samples) for samples in subset.values())} total samples")

if __name__ == "__main__":
    main()