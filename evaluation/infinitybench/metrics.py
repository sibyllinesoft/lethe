"""
Evaluation Metrics for InfinityBench
Standard metrics with academic rigor, enhanced with P/R curves and efficiency metrics.
"""

import string
import re
from typing import List, Dict, Any, Union, Tuple, Optional
from collections import Counter
import numpy as np
from rouge_score import rouge_scorer

def normalize_text(text: str) -> str:
    """Normalize text for fair comparison."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def exact_match(prediction: str, reference: str) -> float:
    """Compute exact match score."""
    pred_norm = normalize_text(prediction)
    ref_norm = normalize_text(reference)
    return float(pred_norm == ref_norm)

def f1_score(prediction: str, reference: str) -> float:
    """Compute F1 score between prediction and reference."""
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
        
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    
    # Calculate precision and recall
    overlap = sum((pred_counter & ref_counter).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
        
    return 2 * precision * recall / (precision + recall)

def rouge_l_score(prediction: str, reference: str) -> float:
    """Compute ROUGE-L score."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores['rougeL'].fmeasure

def accuracy_score(predictions: List[str], references: List[str]) -> float:
    """Compute accuracy for classification tasks."""
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
        
    correct = sum(1 for p, r in zip(predictions, references) if normalize_text(p) == normalize_text(r))
    return correct / len(predictions)

def ndcg_at_k(relevance_scores: List[float], k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain at k."""
    def dcg(scores: List[float], k: int) -> float:
        return sum(score / np.log2(i + 2) for i, score in enumerate(scores[:k]))
    
    actual_dcg = dcg(relevance_scores, k)
    ideal_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = dcg(ideal_scores, k)
    
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def compute_metrics(predictions: List[str], references: List[str], 
                   metric_types: List[str]) -> Dict[str, float]:
    """Compute multiple metrics for evaluation."""
    results = {}
    
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have same length")
    
    for metric in metric_types:
        if metric == 'exact_match':
            scores = [exact_match(p, r) for p, r in zip(predictions, references)]
            results['exact_match'] = np.mean(scores)
            
        elif metric == 'f1':
            scores = [f1_score(p, r) for p, r in zip(predictions, references)]
            results['f1'] = np.mean(scores)
            
        elif metric == 'rouge_l':
            scores = [rouge_l_score(p, r) for p, r in zip(predictions, references)]
            results['rouge_l'] = np.mean(scores)
            
        elif metric == 'accuracy':
            results['accuracy'] = accuracy_score(predictions, references)
            
        elif metric == 'ndcg_10':
            # For NDCG, assume binary relevance based on exact match
            relevance = [1.0 if exact_match(p, r) else 0.0 for p, r in zip(predictions, references)]
            results['ndcg_10'] = ndcg_at_k(relevance, k=10)
            
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    return results

# ============================================================================
# PRECISION/RECALL AND EFFICIENCY METRICS
# ============================================================================

def precision_at_k(relevant_results: List[bool], k: int) -> float:
    """Compute precision at k."""
    if k == 0:
        return 0.0
    
    top_k_results = relevant_results[:k]
    if not top_k_results:
        return 0.0
    
    return sum(top_k_results) / len(top_k_results)

def recall_at_k(relevant_results: List[bool], k: int, total_relevant: int) -> float:
    """Compute recall at k."""
    if total_relevant == 0:
        return 0.0
    
    top_k_results = relevant_results[:k]
    return sum(top_k_results) / total_relevant

def efficiency_at_k(relevant_results: List[bool], k: int) -> float:
    """Compute efficiency (relevance percentage) at k."""
    return precision_at_k(relevant_results, k)

def waste_percentage_at_k(relevant_results: List[bool], k: int) -> float:
    """Compute waste percentage (irrelevant results) at k."""
    return 1.0 - efficiency_at_k(relevant_results, k)

def compute_precision_recall_curves(
    ranked_results: List[Tuple[str, float, bool]], 
    k_values: List[int] = None
) -> Dict[str, List[float]]:
    """
    Compute precision and recall curves for ranked results.
    
    Args:
        ranked_results: List of (result, score, is_relevant) tuples, sorted by score desc
        k_values: List of k values to compute metrics for
    
    Returns:
        Dictionary with precision, recall, and efficiency curves
    """
    if k_values is None:
        k_values = [1, 5, 10, 20, 50, 100]
    
    # Extract relevance indicators
    relevance_list = [is_relevant for _, _, is_relevant in ranked_results]
    total_relevant = sum(relevance_list)
    
    # Compute metrics at each k
    precisions = []
    recalls = []
    efficiencies = []
    waste_percentages = []
    
    for k in k_values:
        precision = precision_at_k(relevance_list, k)
        recall = recall_at_k(relevance_list, k, total_relevant)
        efficiency = efficiency_at_k(relevance_list, k)
        waste = waste_percentage_at_k(relevance_list, k)
        
        precisions.append(precision)
        recalls.append(recall)
        efficiencies.append(efficiency)
        waste_percentages.append(waste)
    
    return {
        'k_values': k_values,
        'precision': precisions,
        'recall': recalls,
        'efficiency': efficiencies,
        'waste_percentage': waste_percentages,
        'total_relevant': total_relevant,
        'total_results': len(ranked_results)
    }

def compute_interpolated_precision_recall(
    ranked_results: List[Tuple[str, float, bool]], 
    recall_points: List[float] = None
) -> Dict[str, List[float]]:
    """
    Compute interpolated precision-recall curve.
    
    Args:
        ranked_results: List of (result, score, is_relevant) tuples
        recall_points: Recall points for interpolation (0.0 to 1.0)
    
    Returns:
        Dictionary with interpolated precision and recall values
    """
    if recall_points is None:
        recall_points = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    relevance_list = [is_relevant for _, _, is_relevant in ranked_results]
    total_relevant = sum(relevance_list)
    
    if total_relevant == 0:
        return {
            'recall_points': recall_points,
            'interpolated_precision': [0.0] * len(recall_points)
        }
    
    # Compute precision and recall at each position
    precisions = []
    recalls = []
    
    for i in range(1, len(relevance_list) + 1):
        precision = precision_at_k(relevance_list, i)
        recall = recall_at_k(relevance_list, i, total_relevant)
        precisions.append(precision)
        recalls.append(recall)
    
    # Interpolate precision at standard recall points
    interpolated_precisions = []
    
    for target_recall in recall_points:
        # Find maximum precision at recall >= target_recall
        max_precision = 0.0
        
        for i, recall in enumerate(recalls):
            if recall >= target_recall:
                max_precision = max(max_precision, precisions[i])
        
        interpolated_precisions.append(max_precision)
    
    return {
        'recall_points': recall_points,
        'interpolated_precision': interpolated_precisions
    }

def compute_average_precision(ranked_results: List[Tuple[str, float, bool]]) -> float:
    """Compute Average Precision (AP) for ranked results."""
    relevance_list = [is_relevant for _, _, is_relevant in ranked_results]
    total_relevant = sum(relevance_list)
    
    if total_relevant == 0:
        return 0.0
    
    ap_sum = 0.0
    relevant_count = 0
    
    for i, is_relevant in enumerate(relevance_list):
        if is_relevant:
            relevant_count += 1
            precision = relevant_count / (i + 1)
            ap_sum += precision
    
    return ap_sum / total_relevant

def compute_efficiency_metrics(
    ranked_results: List[Tuple[str, float, bool]], 
    k_values: List[int] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive efficiency metrics.
    
    Args:
        ranked_results: List of (result, score, is_relevant) tuples
        k_values: List of k values for evaluation
    
    Returns:
        Dictionary with efficiency metrics
    """
    if k_values is None:
        k_values = [1, 5, 10, 20, 50, 100]
    
    relevance_list = [is_relevant for _, _, is_relevant in ranked_results]
    
    # Basic efficiency metrics
    efficiency_metrics = {
        'efficiency_at_k': {},
        'waste_percentage_at_k': {},
        'cumulative_relevant_found': {},
        'total_relevant': sum(relevance_list),
        'total_results': len(ranked_results)
    }
    
    cumulative_relevant = 0
    for k in k_values:
        if k <= len(relevance_list):
            # Count relevant results up to position k
            cumulative_relevant = sum(relevance_list[:k])
            
            efficiency = efficiency_at_k(relevance_list, k)
            waste = waste_percentage_at_k(relevance_list, k)
            
            efficiency_metrics['efficiency_at_k'][f'k_{k}'] = efficiency
            efficiency_metrics['waste_percentage_at_k'][f'k_{k}'] = waste
            efficiency_metrics['cumulative_relevant_found'][f'k_{k}'] = cumulative_relevant
        else:
            # k exceeds available results
            efficiency_metrics['efficiency_at_k'][f'k_{k}'] = 0.0
            efficiency_metrics['waste_percentage_at_k'][f'k_{k}'] = 1.0
            efficiency_metrics['cumulative_relevant_found'][f'k_{k}'] = cumulative_relevant
    
    # Compute efficiency ratio (what percentage of results are relevant)
    if len(relevance_list) > 0:
        overall_efficiency = sum(relevance_list) / len(relevance_list)
    else:
        overall_efficiency = 0.0
    
    efficiency_metrics['overall_efficiency'] = overall_efficiency
    efficiency_metrics['overall_waste'] = 1.0 - overall_efficiency
    
    return efficiency_metrics

def compute_comprehensive_ir_metrics(
    ranked_results: List[Tuple[str, float, bool]], 
    k_values: List[int] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive information retrieval metrics.
    
    Args:
        ranked_results: List of (result, score, is_relevant) tuples
        k_values: List of k values for evaluation
    
    Returns:
        Dictionary with all IR and efficiency metrics
    """
    if k_values is None:
        k_values = [1, 5, 10, 20, 50, 100]
    
    # Compute all metrics
    pr_curves = compute_precision_recall_curves(ranked_results, k_values)
    interpolated_pr = compute_interpolated_precision_recall(ranked_results)
    efficiency_metrics = compute_efficiency_metrics(ranked_results, k_values)
    avg_precision = compute_average_precision(ranked_results)
    
    # Compute NDCG for completeness
    relevance_scores = [1.0 if is_relevant else 0.0 for _, _, is_relevant in ranked_results]
    ndcg_scores = {}
    for k in k_values:
        if k <= len(relevance_scores):
            ndcg_scores[f'ndcg_{k}'] = ndcg_at_k(relevance_scores, k)
    
    return {
        'precision_recall_curves': pr_curves,
        'interpolated_precision_recall': interpolated_pr,
        'efficiency_metrics': efficiency_metrics,
        'average_precision': avg_precision,
        'ndcg_scores': ndcg_scores,
        'summary': {
            'total_results': len(ranked_results),
            'total_relevant': sum(1 for _, _, is_relevant in ranked_results if is_relevant),
            'overall_precision': pr_curves['precision'][0] if pr_curves['precision'] else 0.0,
            'overall_efficiency': efficiency_metrics['overall_efficiency'],
            'average_precision': avg_precision
        }
    }

def compute_task_metrics(task_name: str, predictions: List[str], 
                        references: List[str]) -> Dict[str, float]:
    """Compute appropriate metrics for a specific task."""
    try:
        from .dataset_loader import InfinityBenchDataset
        task_config = InfinityBenchDataset.TASK_CONFIGS.get(task_name, {})
        primary_metric = task_config.get('metric', 'f1')
    except ImportError:
        # Fallback if dataset_loader dependencies aren't available
        primary_metric = 'f1'
    
    # Define metric sets based on task type
    if primary_metric == 'exact_match':
        metrics = ['exact_match', 'f1']
    elif primary_metric == 'f1':
        metrics = ['f1', 'exact_match', 'rouge_l']
    elif primary_metric == 'rouge_l':
        metrics = ['rouge_l', 'f1']
    elif primary_metric == 'accuracy':
        metrics = ['accuracy', 'exact_match']
    else:
        metrics = ['f1', 'exact_match', 'rouge_l']
    
    results = compute_metrics(predictions, references, metrics)
    results['primary_metric'] = results[primary_metric]
    
    return results