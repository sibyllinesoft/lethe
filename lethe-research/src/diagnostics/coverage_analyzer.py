"""
Rung 1: Retrieval/Selection Coverage Analysis 
==========================================

Analyzes coverage of gold answers in retrieved/selected atoms without LLM calls.
Key metrics:
- SpanCoverage@K: does any selected atom contain gold answer string?
- SymbolCoverage@K: does any selected atom contain canonical function/symbol ID?
- IDOverlap@{1,5}: retrieval-only P@k for tasks with ID ground truth
- Entity coverage curve: coverage vs keep ratio
"""

import re
import logging
from typing import Dict, List, Any, Set, Optional, Tuple
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class CoverageAnalyzer:
    """Analyzes retrieval coverage without requiring LLM generation."""
    
    def __init__(self, case_sensitive: bool = False):
        """Initialize coverage analyzer."""
        self.case_sensitive = case_sensitive
        
    def compute_span_coverage_at_k(self, 
                                  selected_atoms: List[str], 
                                  gold_answers: List[str], 
                                  k: int) -> Dict[str, Any]:
        """
        SpanCoverage@K: Does any of the top-K selected atoms contain a gold answer string?
        
        Args:
            selected_atoms: List of retrieved text atoms/chunks
            gold_answers: List of possible correct answers
            k: Number of top atoms to consider
            
        Returns:
            Dict with coverage results and metadata
        """
        if not selected_atoms or not gold_answers:
            return {
                'coverage': False,
                'coverage_rate': 0.0,
                'covering_atoms': [],
                'matched_answers': [],
                'k': k,
                'total_atoms': len(selected_atoms)
            }
        
        # Consider only top-k atoms
        top_k_atoms = selected_atoms[:k]
        
        covering_atoms = []
        matched_answers = []
        
        for i, atom in enumerate(top_k_atoms):
            atom_text = atom if self.case_sensitive else atom.lower()
            
            for answer in gold_answers:
                answer_text = str(answer) if self.case_sensitive else str(answer).lower()
                
                if answer_text and answer_text in atom_text:
                    covering_atoms.append({
                        'atom_index': i,
                        'atom_preview': atom[:200] + "..." if len(atom) > 200 else atom,
                        'matched_answer': answer
                    })
                    matched_answers.append(answer)
                    break  # Move to next atom once we find a match
        
        has_coverage = len(covering_atoms) > 0
        
        return {
            'coverage': has_coverage,
            'coverage_rate': 1.0 if has_coverage else 0.0,
            'covering_atoms': covering_atoms,
            'matched_answers': list(set(matched_answers)),
            'k': k,
            'total_atoms': len(selected_atoms),
            'atoms_checked': len(top_k_atoms)
        }
    
    def compute_symbol_coverage_at_k(self, 
                                   selected_atoms: List[str], 
                                   gold_symbols: List[str], 
                                   k: int) -> Dict[str, Any]:
        """
        SymbolCoverage@K: Does any selected atom contain canonical function/symbol ID?
        Particularly useful for code tasks.
        
        Args:
            selected_atoms: List of retrieved code/text atoms  
            gold_symbols: List of target function/variable/symbol names
            k: Number of top atoms to consider
            
        Returns:
            Dict with symbol coverage results
        """
        if not selected_atoms or not gold_symbols:
            return {
                'coverage': False,
                'coverage_rate': 0.0,
                'covering_atoms': [],
                'matched_symbols': [],
                'k': k,
                'total_atoms': len(selected_atoms)
            }
        
        top_k_atoms = selected_atoms[:k]
        
        covering_atoms = []
        matched_symbols = []
        
        for i, atom in enumerate(top_k_atoms):
            for symbol in gold_symbols:
                symbol_name = str(symbol)
                
                # Multiple symbol matching strategies
                patterns = [
                    # Exact word boundary match
                    rf'\b{re.escape(symbol_name)}\b',
                    # Function definition pattern  
                    rf'def\s+{re.escape(symbol_name)}\s*\(',
                    # Class definition pattern
                    rf'class\s+{re.escape(symbol_name)}\s*[:(]',
                    # Variable assignment
                    rf'{re.escape(symbol_name)}\s*=',
                    # Import statement
                    rf'from\s+\w+\s+import\s+.*{re.escape(symbol_name)}'
                ]
                
                for pattern in patterns:
                    if re.search(pattern, atom, re.IGNORECASE if not self.case_sensitive else 0):
                        covering_atoms.append({
                            'atom_index': i,
                            'atom_preview': atom[:200] + "..." if len(atom) > 200 else atom,
                            'matched_symbol': symbol_name,
                            'pattern_used': pattern
                        })
                        matched_symbols.append(symbol_name)
                        break
                
                if covering_atoms and covering_atoms[-1]['matched_symbol'] == symbol_name:
                    break  # Move to next atom
        
        has_coverage = len(covering_atoms) > 0
        
        return {
            'coverage': has_coverage,
            'coverage_rate': 1.0 if has_coverage else 0.0,
            'covering_atoms': covering_atoms,
            'matched_symbols': list(set(matched_symbols)),
            'k': k,
            'total_atoms': len(selected_atoms),
            'atoms_checked': len(top_k_atoms)
        }
    
    def compute_id_overlap_at_k(self, 
                               selected_ids: List[str], 
                               gold_ids: List[str], 
                               k_values: List[int] = [1, 5]) -> Dict[str, Any]:
        """
        IDOverlap@K: "Retrieval-only P@k" for tasks with ID ground truth.
        Measures precision at k for retrieved IDs vs gold IDs.
        
        Args:
            selected_ids: List of retrieved document/chunk IDs
            gold_ids: List of ground truth relevant IDs  
            k_values: List of k values to compute precision for
            
        Returns:
            Dict with precision@k results
        """
        if not selected_ids or not gold_ids:
            return {k: 0.0 for k in k_values}
        
        gold_id_set = set(str(gid) for gid in gold_ids)
        results = {}
        
        for k in k_values:
            top_k_ids = selected_ids[:k]
            relevant_in_top_k = sum(1 for sid in top_k_ids if str(sid) in gold_id_set)
            
            precision_at_k = relevant_in_top_k / k if k > 0 else 0.0
            
            results[f'precision_at_{k}'] = precision_at_k
            results[f'relevant_retrieved_at_{k}'] = relevant_in_top_k
            results[f'total_retrieved_at_{k}'] = min(k, len(selected_ids))
        
        results.update({
            'total_relevant': len(gold_id_set),
            'total_retrieved': len(selected_ids),
            'gold_ids': list(gold_id_set),
            'selected_ids_preview': selected_ids[:10]
        })
        
        return results
    
    def compute_entity_coverage_curve(self, 
                                    all_atoms: List[str], 
                                    gold_answers: List[str],
                                    keep_ratios: List[float] = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]) -> Dict[str, Any]:
        """
        Entity Coverage Curve: Coverage vs keep ratio.
        Shows how coverage changes as we keep more/fewer atoms.
        
        Args:
            all_atoms: Complete list of available atoms
            gold_answers: Ground truth answers to find
            keep_ratios: Ratios of atoms to keep (e.g., 0.1 = top 10%)
            
        Returns:
            Dict with coverage curve data
        """
        if not all_atoms or not gold_answers:
            return {
                'curve_points': [],
                'max_coverage': 0.0,
                'optimal_keep_ratio': 0.0
            }
        
        curve_points = []
        
        for keep_ratio in sorted(keep_ratios):
            num_atoms_to_keep = max(1, int(len(all_atoms) * keep_ratio))
            selected_atoms = all_atoms[:num_atoms_to_keep]
            
            coverage_result = self.compute_span_coverage_at_k(
                selected_atoms, gold_answers, num_atoms_to_keep
            )
            
            curve_points.append({
                'keep_ratio': keep_ratio,
                'num_atoms': num_atoms_to_keep,
                'coverage_rate': coverage_result['coverage_rate'],
                'matched_answers': len(coverage_result['matched_answers']),
                'total_gold_answers': len(gold_answers)
            })
        
        # Find optimal keep ratio (lowest ratio with maximum coverage)
        max_coverage = max(point['coverage_rate'] for point in curve_points) if curve_points else 0.0
        optimal_points = [p for p in curve_points if p['coverage_rate'] == max_coverage]
        optimal_keep_ratio = min(p['keep_ratio'] for p in optimal_points) if optimal_points else 0.0
        
        return {
            'curve_points': curve_points,
            'max_coverage': max_coverage,
            'optimal_keep_ratio': optimal_keep_ratio,
            'total_atoms': len(all_atoms),
            'total_gold_answers': len(gold_answers)
        }
    
    def analyze_sample_coverage(self, 
                              sample: Dict[str, Any], 
                              selected_atoms: List[str],
                              k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, Any]:
        """
        Comprehensive coverage analysis for a single sample.
        
        Args:
            sample: Sample data with ground truth
            selected_atoms: Retrieved/selected atoms for this sample
            k_values: Different k values to analyze
            
        Returns:
            Dict with all coverage metrics
        """
        # Extract ground truth answers
        ground_truth = sample.get('ground_truth') or sample.get('label') or sample.get('answer')
        if isinstance(ground_truth, str):
            gold_answers = [ground_truth]
        elif isinstance(ground_truth, list):
            gold_answers = [str(x) for x in ground_truth]
        else:
            gold_answers = [str(ground_truth)] if ground_truth else []
        
        # Extract symbols for code tasks
        task_name = sample.get('task_name', '')
        gold_symbols = []
        
        if 'code' in task_name.lower():
            # Try to extract function/variable names from context or answers
            for answer in gold_answers:
                # Simple heuristic: look for identifiers
                symbols = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', str(answer))
                gold_symbols.extend(symbols[:3])  # Limit to first 3
        
        # Extract IDs if available
        selected_ids = []
        gold_ids = []
        
        if 'retrieval' in task_name.lower() or 'id' in str(sample.get('metadata', {})).lower():
            # Try to extract document IDs from atoms (this would need domain-specific logic)
            selected_ids = [f"atom_{i}" for i in range(len(selected_atoms))]  # Placeholder
        
        # Compute all coverage metrics
        results = {
            'sample_id': sample.get('id', 'unknown'),
            'task_name': task_name,
            'gold_answers': gold_answers,
            'gold_symbols': gold_symbols,
            'num_selected_atoms': len(selected_atoms),
            'coverage_metrics': {}
        }
        
        # Span coverage at different k values
        for k in k_values:
            span_coverage = self.compute_span_coverage_at_k(selected_atoms, gold_answers, k)
            results['coverage_metrics'][f'span_coverage_at_{k}'] = span_coverage
            
            if gold_symbols:
                symbol_coverage = self.compute_symbol_coverage_at_k(selected_atoms, gold_symbols, k)  
                results['coverage_metrics'][f'symbol_coverage_at_{k}'] = symbol_coverage
        
        # ID overlap if IDs available
        if selected_ids and gold_ids:
            id_overlap = self.compute_id_overlap_at_k(selected_ids, gold_ids, [1, 5])
            results['coverage_metrics']['id_overlap'] = id_overlap
        
        # Coverage curve
        if len(selected_atoms) > 5:  # Only if we have enough atoms
            coverage_curve = self.compute_entity_coverage_curve(selected_atoms, gold_answers)
            results['coverage_metrics']['coverage_curve'] = coverage_curve
        
        return results
    
    def aggregate_coverage_results(self, sample_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate coverage results across multiple samples.
        
        Args:
            sample_results: List of per-sample coverage results
            
        Returns:
            Dict with aggregated statistics
        """
        if not sample_results:
            return {}
        
        # Group by task
        results_by_task = defaultdict(list)
        for result in sample_results:
            task_name = result.get('task_name', 'unknown')
            results_by_task[task_name].append(result)
        
        # Aggregate statistics
        aggregated = {
            'total_samples': len(sample_results),
            'tasks_analyzed': list(results_by_task.keys()),
            'task_summaries': {}
        }
        
        for task_name, task_results in results_by_task.items():
            task_summary = {
                'num_samples': len(task_results),
                'span_coverage_rates': {},
                'symbol_coverage_rates': {},
                'avg_atoms_per_sample': np.mean([r['num_selected_atoms'] for r in task_results])
            }
            
            # Aggregate span coverage rates
            k_values = [1, 5, 10, 20]
            for k in k_values:
                coverage_rates = []
                for result in task_results:
                    metric_key = f'span_coverage_at_{k}'
                    if metric_key in result['coverage_metrics']:
                        coverage_rates.append(result['coverage_metrics'][metric_key]['coverage_rate'])
                
                if coverage_rates:
                    task_summary['span_coverage_rates'][f'at_{k}'] = {
                        'mean': np.mean(coverage_rates),
                        'std': np.std(coverage_rates),
                        'samples': len(coverage_rates)
                    }
            
            # Aggregate symbol coverage rates (for code tasks)
            for k in k_values:
                symbol_rates = []
                for result in task_results:
                    metric_key = f'symbol_coverage_at_{k}'
                    if metric_key in result['coverage_metrics']:
                        symbol_rates.append(result['coverage_metrics'][metric_key]['coverage_rate'])
                
                if symbol_rates:
                    task_summary['symbol_coverage_rates'][f'at_{k}'] = {
                        'mean': np.mean(symbol_rates),
                        'std': np.std(symbol_rates),
                        'samples': len(symbol_rates)
                    }
            
            aggregated['task_summaries'][task_name] = task_summary
        
        # Overall statistics
        all_span_coverage_at_5 = []
        for task_results in results_by_task.values():
            for result in task_results:
                if 'span_coverage_at_5' in result['coverage_metrics']:
                    all_span_coverage_at_5.append(result['coverage_metrics']['span_coverage_at_5']['coverage_rate'])
        
        if all_span_coverage_at_5:
            aggregated['overall_span_coverage_at_5'] = {
                'mean': np.mean(all_span_coverage_at_5),
                'std': np.std(all_span_coverage_at_5),
                'samples': len(all_span_coverage_at_5)
            }
        
        return aggregated
    
    def diagnose_coverage_issues(self, aggregated_results: Dict[str, Any]) -> List[str]:
        """
        Generate diagnostic insights based on coverage analysis.
        
        Args:
            aggregated_results: Aggregated coverage statistics
            
        Returns:
            List of diagnostic insights
        """
        insights = []
        
        # Check overall coverage rates
        overall_coverage = aggregated_results.get('overall_span_coverage_at_5', {})
        if overall_coverage:
            mean_coverage = overall_coverage['mean']
            if mean_coverage < 0.1:
                insights.append(f"🔴 CRITICAL: Very low span coverage at k=5 ({mean_coverage:.1%}). Selection/retrieval is failing.")
            elif mean_coverage < 0.3:
                insights.append(f"🟡 MODERATE: Low span coverage at k=5 ({mean_coverage:.1%}). Retrieval may need tuning.")
            else:
                insights.append(f"✅ Span coverage at k=5 looks reasonable ({mean_coverage:.1%}).")
        
        # Check task-specific issues
        task_summaries = aggregated_results.get('task_summaries', {})
        for task_name, summary in task_summaries.items():
            span_at_5 = summary.get('span_coverage_rates', {}).get('at_5', {})
            if span_at_5:
                mean_cov = span_at_5['mean']
                if mean_cov < 0.1:
                    insights.append(f"🔴 Task '{task_name}': Critical coverage failure ({mean_cov:.1%} at k=5)")
                elif mean_cov > 0.8:
                    insights.append(f"✅ Task '{task_name}': Good coverage ({mean_cov:.1%} at k=5)")
        
        # Check symbol coverage for code tasks
        for task_name, summary in task_summaries.items():
            if 'code' in task_name.lower():
                symbol_at_5 = summary.get('symbol_coverage_rates', {}).get('at_5', {})
                if symbol_at_5:
                    mean_sym = symbol_at_5['mean']
                    if mean_sym < 0.2:
                        insights.append(f"🟡 Code task '{task_name}': Poor symbol coverage ({mean_sym:.1%} at k=5)")
        
        return insights