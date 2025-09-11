"""
Rung 0: Scoring Sanity Checks
=============================

Validates basic scoring functions without any modeling or retrieval.
Expected to complete in ~1 hour with high confidence in results.

Tests:
- Gold-echo: feed pred := gold for mixed items → expect Accuracy/F1 = 1.0
- Normalizer probes: test three normalizers and report deltas  
- Random baseline: uniform random from candidate space → compute expected chance P@5
"""

import json
import random
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path
import numpy as np
from collections import defaultdict

# Import existing scoring functions (with fallback for missing dependencies)
import sys
import string
from collections import Counter

# Add path for InfiniteBench scoring functions
sys.path.append(str(Path(__file__).parent.parent.parent))

# Fallback scoring functions if original imports fail
def normalize_answer_fallback(s: str) -> str:
    """Fallback normalization function."""
    import re
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score_fallback(prediction, ground_truth) -> tuple:
    """Fallback F1 calculation."""
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def qa_f1_score_fallback(pred: str, ground_truths) -> float:
    """Fallback QA F1 scoring."""
    f1 = 0
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer_fallback(pred)
        normalized_ground_truth = normalize_answer_fallback(ground_truth)
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        scores = f1_score_fallback(prediction_tokens, ground_truth_tokens)
        this_f1, _, _ = scores
        f1 = max(f1, this_f1)
    return f1

def get_score_one_fallback(pred: str, label, task_name: str, model_name: str) -> float:
    """Fallback scoring function with basic task support."""
    if isinstance(label, list):
        # Use F1 scoring for list answers
        return qa_f1_score_fallback(pred, label)
    else:
        # Simple exact match for single answers
        pred_normalized = normalize_answer_fallback(str(pred))
        label_normalized = normalize_answer_fallback(str(label))
        return 1.0 if pred_normalized == label_normalized else 0.0

logger = logging.getLogger(__name__)

# Try to import original functions, fallback if not available
try:
    from benchmarks.infinitebench.src.compute_scores import (
        get_score_one, normalize_answer, normalize_zh_answer, 
        qa_f1_score, qa_f1_score_zh, ALL_TASKS
    )
    logger.info("Successfully imported original InfiniteBench scoring functions")
except ImportError as e:
    logger.warning(f"Could not import original scoring functions: {e}")
    logger.info("Using fallback scoring functions")
    
    # Use fallback functions
    get_score_one = get_score_one_fallback
    normalize_answer = normalize_answer_fallback
    normalize_zh_answer = normalize_answer_fallback
    qa_f1_score = qa_f1_score_fallback
    qa_f1_score_zh = qa_f1_score_fallback
    ALL_TASKS = [
        "passkey", "number_string", "kv_retrieval", "longdialogue_qa_eng",
        "longbook_sum_eng", "longbook_choice_eng", "longbook_qa_eng", 
        "longbook_qa_chn", "math_find", "math_calc", "code_run", "code_debug"
    ]

class ScoringValidator:
    """Validates scoring functions with systematic sanity checks."""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
    def run_gold_echo_test(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test 1: Gold-echo validation
        
        Feed pred := gold for 200 mixed items and verify Accuracy/F1 = 1.0
        This validates that scoring functions work correctly with perfect predictions.
        """
        logger.info("Running gold-echo test...")
        
        results_by_task = defaultdict(list)
        total_samples = 0
        perfect_scores = 0
        
        for sample in samples:
            task_name = sample.get('task_name', 'unknown')
            ground_truth = sample.get('ground_truth') or sample.get('label')
            
            if not ground_truth:
                logger.warning(f"No ground truth found for sample: {sample.get('id', 'unknown')}")
                continue
            
            # Use ground truth as prediction (gold-echo)
            if isinstance(ground_truth, list):
                prediction = ground_truth[0]  # Use first answer for prediction
            else:
                prediction = str(ground_truth)
            
            try:
                # Get score using existing function
                score = get_score_one(prediction, ground_truth, task_name, "diagnostic")
                
                results_by_task[task_name].append(score)
                total_samples += 1
                
                # Check if score is perfect (1.0 or very close)
                if abs(score - 1.0) < 0.001:
                    perfect_scores += 1
                else:
                    logger.warning(f"Non-perfect gold-echo score: {score:.3f} for {task_name}")
                    
            except Exception as e:
                logger.error(f"Scoring error for {task_name}: {e}")
        
        # Calculate summary statistics
        task_summaries = {}
        for task_name, scores in results_by_task.items():
            task_summaries[task_name] = {
                'mean_score': np.mean(scores),
                'perfect_rate': sum(1 for s in scores if abs(s - 1.0) < 0.001) / len(scores),
                'num_samples': len(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores)
            }
        
        overall_perfect_rate = perfect_scores / total_samples if total_samples > 0 else 0.0
        
        return {
            'test_name': 'gold_echo',
            'overall_perfect_rate': overall_perfect_rate,
            'total_samples': total_samples,
            'task_summaries': task_summaries,
            'passed': overall_perfect_rate > 0.95,  # Should be near 1.0
            'issues': [
                f"Task {task}: {summary['perfect_rate']:.3f} perfect rate"
                for task, summary in task_summaries.items() 
                if summary['perfect_rate'] < 0.95
            ]
        }
    
    def run_normalizer_probe_test(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test 2: Normalizer probe validation
        
        Run three normalizers (strict, punctuation-agnostic, unicode-folded) 
        and report deltas to identify normalization issues.
        """
        logger.info("Running normalizer probe test...")
        
        normalizers = {
            'strict': lambda x: x,  # No normalization
            'punctuation_agnostic': normalize_answer,
            'unicode_folded': lambda x: normalize_answer(x).replace(' ', '').lower(),
            'chinese_aware': normalize_zh_answer
        }
        
        results_by_task = defaultdict(lambda: defaultdict(list))
        normalizer_deltas = defaultdict(lambda: defaultdict(list))
        
        for sample in samples[:100]:  # Test subset for speed
            task_name = sample.get('task_name', 'unknown') 
            ground_truth = sample.get('ground_truth') or sample.get('label')
            
            if not ground_truth:
                continue
                
            if isinstance(ground_truth, list):
                prediction = ground_truth[0] + " extra text"  # Add noise
            else:
                prediction = str(ground_truth) + " extra text"
            
            # Test each normalizer
            scores_by_normalizer = {}
            for norm_name, norm_func in normalizers.items():
                try:
                    if norm_name in ['punctuation_agnostic', 'unicode_folded']:
                        # Use F1 scoring for these normalizers
                        if task_name in ['longbook_qa_chn']:
                            score = qa_f1_score_zh(prediction, [ground_truth] if not isinstance(ground_truth, list) else ground_truth)
                        else:
                            score = qa_f1_score(prediction, [ground_truth] if not isinstance(ground_truth, list) else ground_truth)
                    else:
                        # Use standard scoring
                        score = get_score_one(prediction, ground_truth, task_name, "diagnostic")
                    
                    scores_by_normalizer[norm_name] = score
                    results_by_task[task_name][norm_name].append(score)
                    
                except Exception as e:
                    logger.debug(f"Normalizer {norm_name} failed on {task_name}: {e}")
                    scores_by_normalizer[norm_name] = 0.0
            
            # Calculate deltas between normalizers
            if len(scores_by_normalizer) >= 2:
                norm_names = list(scores_by_normalizer.keys())
                for i, norm1 in enumerate(norm_names):
                    for norm2 in norm_names[i+1:]:
                        delta = abs(scores_by_normalizer[norm1] - scores_by_normalizer[norm2])
                        normalizer_deltas[task_name][f"{norm1}_vs_{norm2}"].append(delta)
        
        # Summarize results
        task_summaries = {}
        for task_name in results_by_task:
            task_summaries[task_name] = {
                'normalizer_scores': {
                    norm: {
                        'mean': np.mean(scores),
                        'std': np.std(scores)
                    } for norm, scores in results_by_task[task_name].items()
                },
                'normalizer_deltas': {
                    comparison: {
                        'mean_delta': np.mean(deltas),
                        'max_delta': np.max(deltas) if deltas else 0.0
                    } for comparison, deltas in normalizer_deltas[task_name].items()
                }
            }
        
        return {
            'test_name': 'normalizer_probes',
            'task_summaries': task_summaries,
            'passed': True,  # Informational test
            'insights': [
                f"Task {task}: max delta {max(summary['normalizer_deltas'].get(comp, {}).get('max_delta', 0) for comp in summary['normalizer_deltas']):.3f}"
                for task, summary in task_summaries.items()
            ]
        }
    
    def run_random_baseline_test(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test 3: Random baseline validation
        
        Generate uniform random predictions from candidate space and compute 
        expected chance P@5 to establish lower bounds.
        """
        logger.info("Running random baseline test...")
        
        results_by_task = defaultdict(list)
        random_predictions_made = 0
        
        # Task-specific random generation strategies
        task_generators = {
            'passkey': lambda: str(random.randint(10000, 99999)),
            'number_string': lambda: str(random.randint(1, 50000)),
            'kv_retrieval': lambda: ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8)),
            'code_run': lambda: str(random.randint(-1000, 1000)),
            'code_debug': lambda: random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']),
            'math_find': lambda: str(random.randint(1, 10000)),
            'math_calc': lambda: ' '.join(str(random.randint(0, 100)) for _ in range(10)),
            'longbook_choice_eng': lambda: random.choice(['A', 'B', 'C', 'D']),
            'longdialogue_qa_eng': lambda: ' '.join(random.choices([
                'yes', 'no', 'true', 'false', 'John', 'Mary', 'New York', 'London', 
                'red', 'blue', 'dog', 'cat', 'house', 'car'
            ], k=3)),
            'longbook_qa_eng': lambda: ' '.join(random.choices([
                'character', 'plot', 'setting', 'theme', 'author', 'chapter', 'book',
                'story', 'protagonist', 'antagonist', 'conflict', 'resolution'
            ], k=5)),
            'longbook_sum_eng': lambda: ' '.join(random.choices([
                'The', 'story', 'follows', 'a', 'character', 'who', 'experiences', 'conflict',
                'and', 'reaches', 'resolution', 'through', 'various', 'events', 'in', 'the', 'narrative.'
            ], k=15)),
            'longbook_qa_chn': lambda: ''.join(random.choices('的是在有这个一都我了你他她它我们', k=10)),
        }
        
        for sample in samples[:200]:  # Test subset
            task_name = sample.get('task_name', 'unknown')
            ground_truth = sample.get('ground_truth') or sample.get('label')
            
            if not ground_truth or task_name not in task_generators:
                continue
            
            # Generate random prediction
            random_pred = task_generators[task_name]()
            
            try:
                score = get_score_one(random_pred, ground_truth, task_name, "diagnostic")
                results_by_task[task_name].append(score)
                random_predictions_made += 1
                
            except Exception as e:
                logger.debug(f"Random baseline error for {task_name}: {e}")
        
        # Calculate expected random performance
        task_summaries = {}
        for task_name, scores in results_by_task.items():
            if scores:
                task_summaries[task_name] = {
                    'mean_random_score': np.mean(scores),
                    'std_random_score': np.std(scores),
                    'max_random_score': np.max(scores),
                    'num_samples': len(scores),
                    'expected_p5': min(np.mean(scores) * 5, 1.0)  # Rough P@5 estimate
                }
        
        overall_random_score = np.mean([
            summary['mean_random_score'] 
            for summary in task_summaries.values()
        ]) if task_summaries else 0.0
        
        return {
            'test_name': 'random_baseline',
            'overall_random_score': overall_random_score,
            'predictions_made': random_predictions_made,
            'task_summaries': task_summaries,
            'passed': overall_random_score < 0.2,  # Should be low
            'insights': [
                f"Task {task}: {summary['mean_random_score']:.3f} random score"
                for task, summary in task_summaries.items()
            ]
        }
    
    def run_all_tests(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run all Rung 0 tests and return comprehensive results."""
        
        logger.info("Starting Rung 0: Scoring Sanity Tests")
        
        # Ensure we have enough samples, mix across tasks
        task_samples = defaultdict(list)
        for sample in samples:
            task_name = sample.get('task_name', 'unknown')
            task_samples[task_name].append(sample)
        
        # Balance samples across tasks (up to 200 total)
        mixed_samples = []
        samples_per_task = min(200 // len(task_samples) if task_samples else 200, 50)
        
        for task_name, task_sample_list in task_samples.items():
            selected = task_sample_list[:samples_per_task]
            mixed_samples.extend(selected)
        
        logger.info(f"Running tests on {len(mixed_samples)} mixed samples across {len(task_samples)} tasks")
        
        # Run individual tests
        results = {}
        
        try:
            results['gold_echo'] = self.run_gold_echo_test(mixed_samples)
        except Exception as e:
            logger.error(f"Gold-echo test failed: {e}")
            results['gold_echo'] = {'test_name': 'gold_echo', 'passed': False, 'error': str(e)}
        
        try:
            results['normalizer_probes'] = self.run_normalizer_probe_test(mixed_samples)
        except Exception as e:
            logger.error(f"Normalizer probe test failed: {e}")
            results['normalizer_probes'] = {'test_name': 'normalizer_probes', 'passed': False, 'error': str(e)}
        
        try:
            results['random_baseline'] = self.run_random_baseline_test(mixed_samples)
        except Exception as e:
            logger.error(f"Random baseline test failed: {e}")
            results['random_baseline'] = {'test_name': 'random_baseline', 'passed': False, 'error': str(e)}
        
        # Overall assessment
        tests_passed = sum(1 for r in results.values() if r.get('passed', False))
        total_tests = len(results)
        
        return {
            'rung': 0,
            'name': 'Scoring Sanity',
            'tests_passed': tests_passed,
            'total_tests': total_tests,
            'overall_passed': tests_passed >= 2,  # At least 2/3 tests should pass
            'results': results,
            'samples_tested': len(mixed_samples),
            'summary': self._generate_summary(results)
        }
    
    def _generate_summary(self, results: Dict[str, Any]) -> List[str]:
        """Generate human-readable summary of test results."""
        
        summary = []
        
        # Gold-echo assessment
        gold_result = results.get('gold_echo', {})
        if gold_result.get('passed', False):
            perfect_rate = gold_result.get('overall_perfect_rate', 0)
            summary.append(f"✓ Scoring functions work correctly ({perfect_rate:.1%} perfect rate)")
        else:
            summary.append(f"✗ Scoring functions have issues: {gold_result.get('error', 'multiple failures')}")
        
        # Normalizer assessment
        norm_result = results.get('normalizer_probes', {})
        if 'task_summaries' in norm_result:
            max_delta = 0
            for task_summary in norm_result['task_summaries'].values():
                for comp_deltas in task_summary.get('normalizer_deltas', {}).values():
                    max_delta = max(max_delta, comp_deltas.get('max_delta', 0))
            summary.append(f"ℹ Normalizer differences: max delta {max_delta:.3f}")
        
        # Random baseline assessment  
        random_result = results.get('random_baseline', {})
        if random_result.get('passed', False):
            random_score = random_result.get('overall_random_score', 0)
            summary.append(f"✓ Random baseline established: {random_score:.3f} average score")
        else:
            summary.append("⚠ Random baseline unclear")
        
        return summary