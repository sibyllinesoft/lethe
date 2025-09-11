"""
Rung 3: Oracle Bounds and Isolation Analysis
===========================================

Calculates upper bounds and isolates different sources of performance loss.
This helps distinguish between selection failures and generation failures.

Methods:
- Oracle-context: Replace Lethe head with gold-bearing atoms, run extractive scorer
- Oracle-selector: Force-include atoms containing gold symbols; keep budget fixed  
- Ceiling gap: OracleExtractive - Extractive isolates selection loss
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
from collections import defaultdict
import random

logger = logging.getLogger(__name__)

class OracleBoundsCalculator:
    """Calculate oracle upper bounds for diagnostic analysis."""
    
    def __init__(self, seed: int = 42):
        """Initialize oracle bounds calculator."""
        self.seed = seed
        random.seed(seed)
        
    def create_oracle_context(self, 
                             all_atoms: List[str], 
                             gold_answers: List[str], 
                             budget: int = 10) -> Tuple[List[str], Dict[str, Any]]:
        """
        Oracle-context: Replace selection with gold-bearing atoms.
        
        Args:
            all_atoms: Complete list of available atoms
            gold_answers: Ground truth answers
            budget: Maximum number of atoms to select
            
        Returns:
            Tuple of (oracle_selected_atoms, metadata)
        """
        if not all_atoms or not gold_answers:
            return [], {'gold_bearing_atoms': 0, 'total_atoms': len(all_atoms)}
        
        # Find atoms that contain gold answers
        gold_bearing_atoms = []
        gold_bearing_info = []
        
        for i, atom in enumerate(all_atoms):
            atom_lower = atom.lower()
            for answer in gold_answers:
                answer_lower = str(answer).lower()
                if answer_lower and answer_lower in atom_lower:
                    gold_bearing_atoms.append(atom)
                    gold_bearing_info.append({
                        'atom_index': i,
                        'matched_answer': answer,
                        'atom_preview': atom[:100] + "..." if len(atom) > 100 else atom
                    })
                    break  # Don't double-count atoms
        
        # If we have fewer gold-bearing atoms than budget, fill with random atoms
        oracle_atoms = gold_bearing_atoms[:budget]
        
        if len(oracle_atoms) < budget:
            non_gold_atoms = [atom for atom in all_atoms if atom not in gold_bearing_atoms]
            additional_needed = budget - len(oracle_atoms)
            
            if non_gold_atoms:
                additional_atoms = random.sample(
                    non_gold_atoms, 
                    min(additional_needed, len(non_gold_atoms))
                )
                oracle_atoms.extend(additional_atoms)
        
        metadata = {
            'gold_bearing_atoms': len(gold_bearing_atoms),
            'total_atoms': len(all_atoms),
            'gold_bearing_info': gold_bearing_info,
            'oracle_atoms_selected': len(oracle_atoms),
            'budget': budget,
            'gold_coverage_rate': len(gold_bearing_atoms) / len(all_atoms) if all_atoms else 0.0
        }
        
        return oracle_atoms, metadata
    
    def create_oracle_selector(self, 
                              originally_selected_atoms: List[str], 
                              all_atoms: List[str], 
                              gold_answers: List[str], 
                              budget: int = None) -> Tuple[List[str], Dict[str, Any]]:
        """
        Oracle-selector: Force-include gold-bearing atoms while keeping budget fixed.
        
        Args:
            originally_selected_atoms: Atoms selected by original method
            all_atoms: Complete list of available atoms  
            gold_answers: Ground truth answers
            budget: Budget constraint (defaults to length of originally_selected)
            
        Returns:
            Tuple of (oracle_selected_atoms, metadata)
        """
        if budget is None:
            budget = len(originally_selected_atoms)
        
        if not all_atoms or not gold_answers or budget <= 0:
            return originally_selected_atoms, {'modifications_made': 0}
        
        # Find gold-bearing atoms not in original selection
        gold_bearing_atoms = []
        for atom in all_atoms:
            atom_lower = atom.lower()
            for answer in gold_answers:
                answer_lower = str(answer).lower()
                if answer_lower and answer_lower in atom_lower:
                    gold_bearing_atoms.append(atom)
                    break
        
        # Find which gold-bearing atoms are missing from original selection
        originally_selected_set = set(originally_selected_atoms)
        missing_gold_atoms = [atom for atom in gold_bearing_atoms if atom not in originally_selected_set]
        
        if not missing_gold_atoms:
            # No improvements to make
            return originally_selected_atoms, {
                'modifications_made': 0,
                'gold_already_included': len([atom for atom in originally_selected_atoms if atom in gold_bearing_atoms])
            }
        
        # Create oracle selection by replacing worst atoms with gold atoms
        oracle_selection = originally_selected_atoms.copy()
        modifications_made = 0
        
        # Replace from the end (assuming atoms are ranked by relevance)
        for i, missing_gold in enumerate(missing_gold_atoms):
            if len(oracle_selection) + i < budget:
                oracle_selection.append(missing_gold)
                modifications_made += 1
            elif modifications_made < len(oracle_selection):
                # Replace least relevant atom (from the end)
                replace_index = len(oracle_selection) - 1 - modifications_made
                if replace_index >= 0:
                    oracle_selection[replace_index] = missing_gold
                    modifications_made += 1
        
        # Trim to budget
        oracle_selection = oracle_selection[:budget]
        
        metadata = {
            'modifications_made': modifications_made,
            'original_budget': len(originally_selected_atoms),
            'final_budget': len(oracle_selection),
            'missing_gold_atoms_found': len(missing_gold_atoms),
            'gold_atoms_now_included': len([atom for atom in oracle_selection if atom in gold_bearing_atoms])
        }
        
        return oracle_selection, metadata
    
    def compute_oracle_bounds_for_sample(self, 
                                        sample: Dict[str, Any], 
                                        all_atoms: List[str], 
                                        originally_selected_atoms: List[str], 
                                        extractive_evaluator) -> Dict[str, Any]:
        """
        Compute all oracle bounds for a single sample.
        
        Args:
            sample: Sample data with ground truth
            all_atoms: Complete atom set for this sample
            originally_selected_atoms: Atoms selected by original method
            extractive_evaluator: Extractive baseline evaluator instance
            
        Returns:
            Dict with oracle bound results
        """
        sample_id = sample.get('id', 'unknown')
        task_name = sample.get('task_name', 'unknown')
        ground_truth = sample.get('ground_truth') or sample.get('label')
        
        if isinstance(ground_truth, list):
            gold_answers = [str(x) for x in ground_truth]
        else:
            gold_answers = [str(ground_truth)] if ground_truth else []
        
        if not gold_answers:
            return {'error': 'No ground truth available'}
        
        results = {
            'sample_id': sample_id,
            'task_name': task_name,
            'gold_answers': gold_answers,
            'original_selection_size': len(originally_selected_atoms),
            'total_atoms_available': len(all_atoms)
        }
        
        # Import scoring function with fallback
        try:
            from benchmarks.infinitebench.src.compute_scores import get_score_one
        except ImportError:
            # Use fallback from rung0_scoring_sanity
            from .rung0_scoring_sanity import get_score_one
        
        # 1. Oracle-context bound
        try:
            oracle_context_atoms, oracle_context_meta = self.create_oracle_context(
                all_atoms, gold_answers, budget=len(originally_selected_atoms)
            )
            
            # Run extractive evaluation on oracle context
            oracle_context_predictions = []
            
            # Try each extractive method
            extractive_methods = {
                'regex': lambda: extractive_evaluator.extract_regex_patterns(
                    ' '.join(oracle_context_atoms), task_name
                ),
                'top_span': lambda: extractive_evaluator.extract_top_span_heuristic(
                    oracle_context_atoms, gold_answers, {'coverage': True, 'covering_atoms': [{'atom_index': 0}]}
                )
            }
            
            oracle_context_scores = {}
            for method_name, method_func in extractive_methods.items():
                try:
                    prediction = method_func()
                    if prediction:
                        score = get_score_one(prediction, ground_truth, task_name, "oracle")
                        oracle_context_scores[method_name] = score
                except Exception as e:
                    logger.debug(f"Oracle context {method_name} failed: {e}")
            
            results['oracle_context'] = {
                'metadata': oracle_context_meta,
                'scores': oracle_context_scores,
                'best_score': max(oracle_context_scores.values()) if oracle_context_scores else 0.0
            }
            
        except Exception as e:
            logger.error(f"Oracle context computation failed for {sample_id}: {e}")
            results['oracle_context'] = {'error': str(e)}
        
        # 2. Oracle-selector bound
        try:
            oracle_selector_atoms, oracle_selector_meta = self.create_oracle_selector(
                originally_selected_atoms, all_atoms, gold_answers
            )
            
            oracle_selector_scores = {}
            for method_name, method_func in extractive_methods.items():
                try:
                    prediction = method_func()
                    if prediction:
                        score = get_score_one(prediction, ground_truth, task_name, "oracle")
                        oracle_selector_scores[method_name] = score
                except Exception as e:
                    logger.debug(f"Oracle selector {method_name} failed: {e}")
            
            results['oracle_selector'] = {
                'metadata': oracle_selector_meta,
                'scores': oracle_selector_scores,
                'best_score': max(oracle_selector_scores.values()) if oracle_selector_scores else 0.0
            }
            
        except Exception as e:
            logger.error(f"Oracle selector computation failed for {sample_id}: {e}")
            results['oracle_selector'] = {'error': str(e)}
        
        # 3. Baseline extractive performance (for comparison)
        try:
            baseline_scores = {}
            for method_name, method_func in extractive_methods.items():
                try:
                    prediction = method_func()
                    if prediction:
                        score = get_score_one(prediction, ground_truth, task_name, "baseline")
                        baseline_scores[method_name] = score
                except Exception as e:
                    logger.debug(f"Baseline {method_name} failed: {e}")
            
            results['baseline_extractive'] = {
                'scores': baseline_scores,
                'best_score': max(baseline_scores.values()) if baseline_scores else 0.0
            }
            
        except Exception as e:
            logger.error(f"Baseline extractive computation failed for {sample_id}: {e}")
            results['baseline_extractive'] = {'error': str(e)}
        
        return results
    
    def analyze_oracle_gaps(self, oracle_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze gaps between different oracle bounds and baseline performance.
        
        Args:
            oracle_results: List of per-sample oracle results
            
        Returns:
            Dict with gap analysis
        """
        if not oracle_results:
            return {}
        
        # Extract scores for analysis
        oracle_context_scores = []
        oracle_selector_scores = []
        baseline_extractive_scores = []
        
        results_by_task = defaultdict(lambda: defaultdict(list))
        
        for result in oracle_results:
            task_name = result.get('task_name', 'unknown')
            
            # Oracle context scores
            oracle_context = result.get('oracle_context', {})
            if 'best_score' in oracle_context:
                score = oracle_context['best_score']
                oracle_context_scores.append(score)
                results_by_task[task_name]['oracle_context'].append(score)
            
            # Oracle selector scores
            oracle_selector = result.get('oracle_selector', {})
            if 'best_score' in oracle_selector:
                score = oracle_selector['best_score']
                oracle_selector_scores.append(score)
                results_by_task[task_name]['oracle_selector'].append(score)
            
            # Baseline extractive scores
            baseline_extractive = result.get('baseline_extractive', {})
            if 'best_score' in baseline_extractive:
                score = baseline_extractive['best_score']
                baseline_extractive_scores.append(score)
                results_by_task[task_name]['baseline_extractive'].append(score)
        
        # Calculate overall gaps
        overall_analysis = {}
        
        if oracle_context_scores and baseline_extractive_scores:
            oracle_context_mean = np.mean(oracle_context_scores)
            baseline_mean = np.mean(baseline_extractive_scores)
            
            overall_analysis['ceiling_gap'] = {
                'oracle_context_mean': oracle_context_mean,
                'baseline_extractive_mean': baseline_mean,
                'absolute_gap': oracle_context_mean - baseline_mean,
                'relative_gap': (oracle_context_mean - baseline_mean) / baseline_mean if baseline_mean > 0 else float('inf'),
                'samples': len(oracle_context_scores)
            }
        
        if oracle_selector_scores and baseline_extractive_scores:
            oracle_selector_mean = np.mean(oracle_selector_scores)
            baseline_mean = np.mean(baseline_extractive_scores)
            
            overall_analysis['selection_gap'] = {
                'oracle_selector_mean': oracle_selector_mean,
                'baseline_extractive_mean': baseline_mean,
                'absolute_gap': oracle_selector_mean - baseline_mean,
                'relative_gap': (oracle_selector_mean - baseline_mean) / baseline_mean if baseline_mean > 0 else float('inf'),
                'samples': len(oracle_selector_scores)
            }
        
        # Task-specific analysis
        task_analysis = {}
        for task_name, task_results in results_by_task.items():
            task_analysis[task_name] = {}
            
            if 'oracle_context' in task_results and 'baseline_extractive' in task_results:
                oracle_mean = np.mean(task_results['oracle_context'])
                baseline_mean = np.mean(task_results['baseline_extractive'])
                
                task_analysis[task_name]['ceiling_gap'] = {
                    'oracle_mean': oracle_mean,
                    'baseline_mean': baseline_mean,
                    'absolute_gap': oracle_mean - baseline_mean,
                    'samples': len(task_results['oracle_context'])
                }
            
            if 'oracle_selector' in task_results and 'baseline_extractive' in task_results:
                oracle_mean = np.mean(task_results['oracle_selector'])
                baseline_mean = np.mean(task_results['baseline_extractive'])
                
                task_analysis[task_name]['selection_gap'] = {
                    'oracle_mean': oracle_mean,
                    'baseline_mean': baseline_mean,
                    'absolute_gap': oracle_mean - baseline_mean,
                    'samples': len(task_results['oracle_selector'])
                }
        
        return {
            'overall_analysis': overall_analysis,
            'task_analysis': task_analysis,
            'total_samples': len(oracle_results)
        }
    
    def diagnose_performance_gaps(self, gap_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate diagnostic insights based on oracle gap analysis.
        
        Args:
            gap_analysis: Results from analyze_oracle_gaps
            
        Returns:
            List of diagnostic insights
        """
        insights = []
        
        overall = gap_analysis.get('overall_analysis', {})
        
        # Analyze ceiling gap (oracle context vs baseline)
        ceiling_gap = overall.get('ceiling_gap', {})
        if ceiling_gap:
            absolute_gap = ceiling_gap.get('absolute_gap', 0)
            relative_gap = ceiling_gap.get('relative_gap', 0)
            
            if absolute_gap > 0.3:
                insights.append(f"🔴 LARGE CEILING GAP: Oracle context achieves {absolute_gap:.2f} higher score. Major room for improvement in selection.")
            elif absolute_gap > 0.1:
                insights.append(f"🟡 MODERATE CEILING GAP: Oracle context {absolute_gap:.2f} higher. Some selection issues.")
            else:
                insights.append(f"✅ Small ceiling gap ({absolute_gap:.2f}). Selection is working reasonably well.")
        
        # Analyze selection gap (oracle selector vs baseline)  
        selection_gap = overall.get('selection_gap', {})
        if selection_gap:
            absolute_gap = selection_gap.get('absolute_gap', 0)
            
            if absolute_gap > 0.2:
                insights.append(f"🔴 SELECTION ISSUE: Forcing gold atoms improves score by {absolute_gap:.2f}. Retrieval is missing key content.")
            elif absolute_gap > 0.05:
                insights.append(f"🟡 Minor selection issue: {absolute_gap:.2f} improvement possible from better retrieval.")
            else:
                insights.append(f"✅ Selection appears adequate (gap: {absolute_gap:.2f}).")
        
        # Task-specific insights
        task_analysis = gap_analysis.get('task_analysis', {})
        for task_name, task_gaps in task_analysis.items():
            ceiling_gap = task_gaps.get('ceiling_gap', {})
            if ceiling_gap and ceiling_gap.get('absolute_gap', 0) > 0.4:
                insights.append(f"🔴 Task '{task_name}': Severe ceiling gap ({ceiling_gap['absolute_gap']:.2f}). Selection completely failing.")
        
        return insights