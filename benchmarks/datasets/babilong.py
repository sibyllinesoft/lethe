#!/usr/bin/env python3
"""
BABILong Dataset Loader
=======================

Loader for BABILong benchmark - distributed facts evaluation.
Tests long-context reasoning with distributed factual information.

Official repo: https://github.com/booydar/babilong  
"""

import logging
from typing import Iterator, List, Dict, Any
from .base import BaseDatasetLoader, DatasetSample

logger = logging.getLogger(__name__)


class BABILongLoader(BaseDatasetLoader):
    """BABILong dataset loader."""
    
    def get_expected_fields(self) -> List[str]:
        return ["id", "story", "question", "answer", "task_id", "supporting_facts"]
    
    def load_samples(self) -> Iterator[DatasetSample]:
        """Load BABILong samples."""
        for item in self._load_jsonl(self.data_path):
            # BABILong field mapping
            sample_id = item.get("id", f"babilong_{hash(item.get('question', ''))}")
            
            # Story is the context, question is the query
            context = item.get("story", "")
            query = item.get("question", "")
            answer = item.get("answer", "")
            
            if not all([query.strip(), context.strip(), answer.strip()]):
                logger.warning(f"Missing required fields in BABILong sample: {sample_id}")
                continue
            
            # Task information
            task_id = item.get("task_id", "unknown")
            task_name = self._get_task_name(task_id)
            
            # Supporting facts for analysis
            supporting_facts = item.get("supporting_facts", [])
            
            context_length = len(context.split())
            
            # Extract task-specific metadata
            metadata = {
                "task_type": "babilong_evaluation", 
                "task_id": task_id,
                "task_name": task_name,
                "supporting_fact_count": len(supporting_facts),
                "supporting_facts": supporting_facts,
                "source": "babilong"
            }
            
            # Analyze story structure if possible
            if context:
                sentences = context.split('.')
                metadata["story_sentence_count"] = len([s for s in sentences if s.strip()])
                
                # Look for fact distribution patterns
                if supporting_facts:
                    fact_positions = []
                    for fact in supporting_facts:
                        # Find where supporting facts appear in the story
                        fact_pos = context.find(str(fact))
                        if fact_pos >= 0:
                            # Convert to relative position (0.0 to 1.0)
                            relative_pos = fact_pos / len(context)
                            fact_positions.append(relative_pos)
                    
                    if fact_positions:
                        metadata["fact_positions"] = fact_positions
                        metadata["fact_distribution_span"] = max(fact_positions) - min(fact_positions)
            
            # Add any additional fields
            for key, value in item.items():
                if key not in ["id", "story", "question", "answer", "task_id", "supporting_facts"]:
                    metadata[f"babilong_{key}"] = value
            
            yield DatasetSample(
                id=str(sample_id),
                query=query,
                context=context,
                answer=answer,
                context_length=context_length,
                query_length=len(query.split()),
                metadata=metadata
            )
    
    def _get_task_name(self, task_id: Any) -> str:
        """Map task ID to human-readable name."""
        task_map = {
            1: "single_supporting_fact",
            2: "two_supporting_facts", 
            3: "three_supporting_facts",
            4: "two_arg_relations",
            5: "three_arg_relations",
            6: "yes_no_questions",
            7: "counting",
            8: "lists_sets",
            9: "simple_negation",
            10: "indefinite_knowledge",
            11: "basic_coreference",
            12: "conjunction",
            13: "compound_coreference",
            14: "time_reasoning",
            15: "basic_deduction",
            16: "basic_induction",
            17: "positional_reasoning",
            18: "size_reasoning",
            19: "path_finding",
            20: "agents_motivations"
        }
        
        try:
            task_num = int(task_id)
            return task_map.get(task_num, f"task_{task_num}")
        except (ValueError, TypeError):
            return str(task_id)