"""
InfiniteBench Evaluation Metrics
==============================

Comprehensive metrics implementation for evaluating long-context retrieval
systems on the InfiniteBench dataset. Includes standard metrics like ROUGE-L,
Exact Match, F1, and nDCG@k for academic evaluation.

Author: Lethe Research Team
"""

import re
import string
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import Counter
from dataclasses import dataclass
# from rouge import Rouge  # Implemented manually below
from sklearn.metrics import ndcg_score
import logging

logger = logging.getLogger(__name__)

@dataclass
class MetricResult:
    """Container for metric evaluation results."""
    
    metric_name: str
    score: float
    details: Dict[str, Any]
    sample_scores: Optional[List[float]] = None
    
    def __post_init__(self):
        """Validate metric result."""
        if not 0 <= self.score <= 1:
            logger.warning(f"Unusual score value for {self.metric_name}: {self.score}")

@dataclass  
class EvaluationSummary:
    """Summary of evaluation results across multiple metrics."""
    
    task_name: str
    num_samples: int
    metric_results: Dict[str, MetricResult]
    overall_score: float
    
    def get_metric_score(self, metric_name: str) -> float:
        """Get score for specific metric."""
        if metric_name not in self.metric_results:
            raise KeyError(f"Metric '{metric_name}' not found")
        return self.metric_results[metric_name].score

class InfiniteBenchMetrics:
    """
    Comprehensive metrics implementation for InfiniteBench evaluation.
    
    Supports:
    - Exact Match (EM) 
    - F1 Score
    - ROUGE-L
    - nDCG@k
    - Accuracy
    - Custom long-context metrics
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        pass  # No external dependencies needed
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _rouge_l_single(self, prediction: str, reference: str) -> Tuple[float, float, float]:
        """Calculate ROUGE-L for a single prediction-reference pair."""
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        if not pred_tokens and not ref_tokens:
            return 1.0, 1.0, 1.0
        if not pred_tokens or not ref_tokens:
            return 0.0, 0.0, 0.0
        
        lcs_len = self._lcs_length(pred_tokens, ref_tokens)
        
        precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0
        recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return f1, precision, recall
    
    def exact_match(self, predictions: List[str], references: List[str]) -> MetricResult:
        """
        Calculate Exact Match score.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            MetricResult with EM score
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        sample_scores = []
        for pred, ref in zip(predictions, references):
            # Normalize both strings
            pred_norm = self._normalize_text(pred)
            ref_norm = self._normalize_text(ref)
            
            score = 1.0 if pred_norm == ref_norm else 0.0
            sample_scores.append(score)
        
        overall_score = np.mean(sample_scores)
        
        return MetricResult(
            metric_name="exact_match",
            score=overall_score,
            details={
                "num_exact_matches": sum(sample_scores),
                "total_samples": len(sample_scores),
                "match_rate": overall_score
            },
            sample_scores=sample_scores
        )
    
    def f1_score(self, predictions: List[str], references: List[str]) -> MetricResult:
        """
        Calculate token-level F1 score.
        
        Args:
            predictions: List of predicted answers
            references: List of reference answers
            
        Returns:
            MetricResult with F1 score
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        sample_scores = []
        precision_scores = []
        recall_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = set(self._tokenize(pred))
            ref_tokens = set(self._tokenize(ref))
            
            if len(pred_tokens) == 0 and len(ref_tokens) == 0:
                # Both empty - perfect match
                f1 = 1.0
                precision = 1.0
                recall = 1.0
            elif len(pred_tokens) == 0 or len(ref_tokens) == 0:
                # One empty - no match
                f1 = 0.0
                precision = 0.0
                recall = 0.0
            else:
                # Calculate F1
                intersection = pred_tokens & ref_tokens
                precision = len(intersection) / len(pred_tokens)
                recall = len(intersection) / len(ref_tokens)
                
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * precision * recall / (precision + recall)
            
            sample_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        overall_score = np.mean(sample_scores)
        
        return MetricResult(
            metric_name="f1_score",
            score=overall_score,
            details={
                "precision": np.mean(precision_scores),
                "recall": np.mean(recall_scores),
                "f1": overall_score,
                "num_samples": len(sample_scores)
            },
            sample_scores=sample_scores
        )
    
    def rouge_l(self, predictions: List[str], references: List[str]) -> MetricResult:
        """
        Calculate ROUGE-L score.
        
        Args:
            predictions: List of predicted answers/summaries
            references: List of reference answers/summaries
            
        Returns:
            MetricResult with ROUGE-L score
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        sample_scores = []
        precision_scores = []
        recall_scores = []
        
        for pred, ref in zip(predictions, references):
            try:
                rouge_score, precision, recall = self._rouge_l_single(pred.strip(), ref.strip())
            except Exception as e:
                logger.warning(f"ROUGE calculation failed: {e}")
                rouge_score, precision, recall = 0.0, 0.0, 0.0
            
            sample_scores.append(rouge_score)
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        overall_score = np.mean(sample_scores)
        
        return MetricResult(
            metric_name="rouge_l",
            score=overall_score,
            details={
                "rouge_l_f": overall_score,
                "rouge_l_p": np.mean(precision_scores),
                "rouge_l_r": np.mean(recall_scores),
                "num_samples": len(sample_scores)
            },
            sample_scores=sample_scores
        )
    
    def ndcg_at_k(self, 
                  predictions: List[List[Tuple[str, float]]],  
                  references: List[List[str]],
                  k: int = 10) -> MetricResult:
        """
        Calculate nDCG@k for ranked retrieval results.
        
        Args:
            predictions: List of ranked (document, score) tuples per query
            references: List of relevant documents per query  
            k: Cut-off rank for nDCG calculation
            
        Returns:
            MetricResult with nDCG@k score
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        sample_scores = []
        
        for pred_ranked, ref_relevant in zip(predictions, references):
            # Convert to relevance scores
            y_true = []
            y_score = []
            
            for doc, score in pred_ranked[:k]:
                relevance = 1.0 if doc in ref_relevant else 0.0
                y_true.append(relevance)
                y_score.append(score)
            
            if not y_true or max(y_true) == 0:
                # No relevant documents found
                ndcg = 0.0
            else:
                try:
                    ndcg = ndcg_score([y_true], [y_score], k=k)
                except Exception as e:
                    logger.warning(f"nDCG calculation failed: {e}")
                    ndcg = 0.0
            
            sample_scores.append(ndcg)
        
        overall_score = np.mean(sample_scores)
        
        return MetricResult(
            metric_name=f"ndcg_at_{k}",
            score=overall_score,
            details={
                "k": k,
                "ndcg": overall_score,
                "num_queries": len(sample_scores),
                "queries_with_relevant": sum(1 for score in sample_scores if score > 0)
            },
            sample_scores=sample_scores
        )
    
    def accuracy(self, predictions: List[str], references: List[str]) -> MetricResult:
        """
        Calculate classification accuracy.
        
        Args:
            predictions: List of predicted labels
            references: List of true labels
            
        Returns:
            MetricResult with accuracy score
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        sample_scores = []
        for pred, ref in zip(predictions, references):
            # Normalize for comparison
            pred_norm = str(pred).strip().lower()
            ref_norm = str(ref).strip().lower()
            
            score = 1.0 if pred_norm == ref_norm else 0.0
            sample_scores.append(score)
        
        overall_score = np.mean(sample_scores)
        
        return MetricResult(
            metric_name="accuracy",
            score=overall_score,
            details={
                "correct": sum(sample_scores),
                "total": len(sample_scores),
                "accuracy": overall_score
            },
            sample_scores=sample_scores
        )
    
    def long_context_metrics(self,
                           predictions: List[str],
                           references: List[str], 
                           contexts: List[str]) -> Dict[str, MetricResult]:
        """
        Calculate comprehensive long-context specific metrics.
        
        Args:
            predictions: List of predictions
            references: List of references
            contexts: List of input contexts
            
        Returns:
            Dictionary of metric results
        """
        results = {}
        
        # Standard metrics
        results["exact_match"] = self.exact_match(predictions, references)
        results["f1_score"] = self.f1_score(predictions, references) 
        results["rouge_l"] = self.rouge_l(predictions, references)
        
        # Context-specific metrics
        context_coverage_scores = []
        answer_position_scores = []
        
        for pred, ref, context in zip(predictions, references, contexts):
            # Context coverage: How much of the answer appears in context
            if ref and context:
                ref_tokens = set(self._tokenize(ref))
                context_tokens = set(self._tokenize(context))
                
                if ref_tokens:
                    coverage = len(ref_tokens & context_tokens) / len(ref_tokens)
                    context_coverage_scores.append(coverage)
                
                # Answer position: Relative position of answer in context
                ref_norm = self._normalize_text(ref)
                context_norm = self._normalize_text(context)
                
                if ref_norm in context_norm:
                    pos = context_norm.find(ref_norm)
                    relative_pos = pos / len(context_norm) if context_norm else 0.5
                    answer_position_scores.append(relative_pos)
        
        # Context coverage metric
        if context_coverage_scores:
            results["context_coverage"] = MetricResult(
                metric_name="context_coverage",
                score=np.mean(context_coverage_scores),
                details={
                    "mean_coverage": np.mean(context_coverage_scores),
                    "std_coverage": np.std(context_coverage_scores),
                    "num_samples": len(context_coverage_scores)
                },
                sample_scores=context_coverage_scores
            )
        
        # Answer position metric
        if answer_position_scores:
            results["answer_position"] = MetricResult(
                metric_name="answer_position",
                score=np.mean(answer_position_scores),
                details={
                    "mean_position": np.mean(answer_position_scores),
                    "std_position": np.std(answer_position_scores),
                    "num_samples": len(answer_position_scores)
                },
                sample_scores=answer_position_scores
            )
        
        return results
    
    def evaluate_task(self,
                     predictions: List[str],
                     references: List[str],
                     task_name: str,
                     contexts: Optional[List[str]] = None) -> EvaluationSummary:
        """
        Evaluate predictions for a specific InfiniteBench task.
        
        Args:
            predictions: Model predictions
            references: Ground truth references
            task_name: Name of the InfiniteBench task
            contexts: Optional context strings for context-specific metrics
            
        Returns:
            EvaluationSummary with all relevant metrics
        """
        metric_results = {}
        
        # Task-specific primary metric
        if task_name in ['passkey', 'number_string', 'kv_retrieval', 
                        'longbook_choice_eng', 'code_debug', 'code_run',
                        'math_calc', 'math_find']:
            # Tasks that use exact match/accuracy
            metric_results["accuracy"] = self.accuracy(predictions, references)
            metric_results["exact_match"] = self.exact_match(predictions, references)
            primary_metric = "exact_match"
            
        elif task_name in ['longbook_qa_eng', 'longbook_qa_chn', 'longdialogue_qa_eng']:
            # Q&A tasks that use F1
            metric_results["f1_score"] = self.f1_score(predictions, references)
            metric_results["exact_match"] = self.exact_match(predictions, references)
            primary_metric = "f1_score"
            
        elif task_name in ['longbook_sum_eng']:
            # Summarization tasks that use ROUGE-L
            metric_results["rouge_l"] = self.rouge_l(predictions, references)
            metric_results["f1_score"] = self.f1_score(predictions, references)
            primary_metric = "rouge_l"
        else:
            # Default to comprehensive evaluation
            metric_results["exact_match"] = self.exact_match(predictions, references)
            metric_results["f1_score"] = self.f1_score(predictions, references)
            metric_results["rouge_l"] = self.rouge_l(predictions, references)
            primary_metric = "f1_score"
        
        # Add context-specific metrics if contexts provided
        if contexts:
            context_metrics = self.long_context_metrics(predictions, references, contexts)
            metric_results.update(context_metrics)
        
        # Overall score is the primary metric for this task
        overall_score = metric_results[primary_metric].score
        
        return EvaluationSummary(
            task_name=task_name,
            num_samples=len(predictions),
            metric_results=metric_results,
            overall_score=overall_score
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if not text:
            return []
        
        # Simple whitespace tokenization after normalization
        normalized = self._normalize_text(text)
        return normalized.split()

def main():
    """Example usage of InfiniteBenchMetrics."""
    
    metrics = InfiniteBenchMetrics()
    
    # Example predictions and references
    predictions = [
        "The answer is 42",
        "Machine learning is a subset of AI", 
        "The capital of France is Paris"
    ]
    
    references = [
        "42", 
        "Machine learning is part of artificial intelligence",
        "Paris is the capital of France"
    ]
    
    contexts = [
        "In the book, the ultimate answer to life, universe and everything is 42.",
        "Artificial intelligence encompasses machine learning as a key component.",
        "France is a European country with Paris as its capital city."
    ]
    
    # Calculate individual metrics
    em_result = metrics.exact_match(predictions, references)
    print(f"Exact Match: {em_result.score:.3f}")
    
    f1_result = metrics.f1_score(predictions, references) 
    print(f"F1 Score: {f1_result.score:.3f}")
    
    rouge_result = metrics.rouge_l(predictions, references)
    print(f"ROUGE-L: {rouge_result.score:.3f}")
    
    # Long-context metrics
    long_metrics = metrics.long_context_metrics(predictions, references, contexts)
    print(f"Context Coverage: {long_metrics['context_coverage'].score:.3f}")
    
    # Task evaluation
    summary = metrics.evaluate_task(predictions, references, "longbook_qa_eng", contexts)
    print(f"\nTask Evaluation Summary:")
    print(f"Task: {summary.task_name}")
    print(f"Overall Score: {summary.overall_score:.3f}")
    for metric_name, result in summary.metric_results.items():
        print(f"{metric_name}: {result.score:.3f}")

if __name__ == "__main__":
    main()