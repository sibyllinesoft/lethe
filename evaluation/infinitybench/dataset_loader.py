"""
InfinityBench Dataset Loader
Handles loading and preprocessing of the 12 InfinityBench tasks.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tiktoken
from datasets import load_dataset

logger = logging.getLogger(__name__)

class InfinityBenchDataset:
    """InfinityBench dataset loader with comprehensive task support."""
    
    TASK_CONFIGS = {
        'passkey': {'metric': 'exact_match', 'type': 'retrieval'},
        'number_string': {'metric': 'exact_match', 'type': 'retrieval'}, 
        'kv_retrieval': {'metric': 'exact_match', 'type': 'retrieval'},
        'longbook_qa_eng': {'metric': 'f1', 'type': 'qa'},
        'longdialogue_qa_eng': {'metric': 'f1', 'type': 'qa'},
        'math_find': {'metric': 'exact_match', 'type': 'reasoning'},
        'math_calc': {'metric': 'exact_match', 'type': 'reasoning'},
        'code_run': {'metric': 'exact_match', 'type': 'code'},
        'code_debug': {'metric': 'exact_match', 'type': 'code'},
        'longbook_sum_eng': {'metric': 'rouge_l', 'type': 'summarization'},
        'longbook_choice_eng': {'metric': 'accuracy', 'type': 'classification'},
        'longbook_qa_chn': {'metric': 'f1', 'type': 'qa'}
    }
    
    def __init__(self, data_dir: str = "./data/infinitybench", max_samples: Optional[int] = None):
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tasks = {}
        
    def load_task(self, task_name: str) -> List[Dict]:
        """Load a specific InfinityBench task."""
        if task_name not in self.TASK_CONFIGS:
            raise ValueError(f"Unknown task: {task_name}")
            
        logger.info(f"Loading task: {task_name}")
        
        try:
            # Load from Hugging Face datasets
            dataset = load_dataset("xinrongzhang2022/InfiniteBench", task_name, split="test")
            
            samples = []
            for i, example in enumerate(dataset):
                if self.max_samples and i >= self.max_samples:
                    break
                    
                # Standardize format
                sample = {
                    'id': example.get('id', f"{task_name}_{i}"),
                    'task': task_name,
                    'context': example['context'],
                    'question': example['input'],
                    'answer': example['answer'],
                    'context_length': len(self.tokenizer.encode(example['context'])),
                    'question_length': len(self.tokenizer.encode(example['input']))
                }
                
                samples.append(sample)
                
            logger.info(f"Loaded {len(samples)} samples for {task_name}")
            self.tasks[task_name] = samples
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load task {task_name}: {e}")
            raise
            
    def load_all_tasks(self, task_names: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """Load all specified tasks."""
        if task_names is None:
            task_names = list(self.TASK_CONFIGS.keys())
            
        all_tasks = {}
        for task_name in task_names:
            all_tasks[task_name] = self.load_task(task_name)
            
        return all_tasks
    
    def get_task_stats(self, task_name: str) -> Dict:
        """Get statistics for a loaded task."""
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} not loaded")
            
        samples = self.tasks[task_name]
        context_lengths = [s['context_length'] for s in samples]
        
        return {
            'task_name': task_name,
            'num_samples': len(samples),
            'avg_context_length': sum(context_lengths) / len(context_lengths),
            'max_context_length': max(context_lengths),
            'min_context_length': min(context_lengths),
            'metric_type': self.TASK_CONFIGS[task_name]['metric'],
            'task_type': self.TASK_CONFIGS[task_name]['type']
        }
        
    def get_all_stats(self) -> Dict[str, Dict]:
        """Get statistics for all loaded tasks."""
        return {task: self.get_task_stats(task) for task in self.tasks.keys()}
        
    def create_evaluation_split(self, task_name: str, test_size: float = 0.2, 
                              random_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
        """Create train/test split for evaluation."""
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} not loaded")
            
        import random
        random.seed(random_seed)
        
        samples = self.tasks[task_name].copy()
        random.shuffle(samples)
        
        split_idx = int(len(samples) * (1 - test_size))
        
        return samples[:split_idx], samples[split_idx:]
        
    def save_task(self, task_name: str, filepath: str):
        """Save task data to JSON."""
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} not loaded")
            
        with open(filepath, 'w') as f:
            json.dump(self.tasks[task_name], f, indent=2)
            
    def load_from_file(self, task_name: str, filepath: str):
        """Load task data from JSON file."""
        with open(filepath, 'r') as f:
            self.tasks[task_name] = json.load(f)