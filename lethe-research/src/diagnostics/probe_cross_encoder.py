"""
Probe 3: S2 Cross-Encoder Pair Feeding Sanity
==============================================

Validates that the cross-encoder reranker is receiving proper query+candidate 
pairs and producing meaningful score distributions. Tests 20 CE inputs:

- Log CE inputs **verbatim**: CE([question_text], [atom_text_snippet]) → score
- Expected: scores spread (std>0.1) and correlate with lexical overlap  
- Red flags: every score ≈ constant (0.5 or 0.0) → CE given empty/identical sides or wrong tokenizer

Tests cross-encoder wiring, input formatting, and score meaningfulness.
"""

import numpy as np
import pandas as pd
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class CrossEncoderStats:
    """Statistics for cross-encoder analysis."""
    ce_inputs: List[Dict[str, Any]]  # Raw CE input pairs
    ce_scores: List[float]  # Output scores
    lexical_overlaps: List[float]  # Lexical overlap ratios
    query_lengths: List[int]  # Query text lengths
    candidate_lengths: List[int]  # Candidate text lengths
    empty_query_count: int
    empty_candidate_count: int
    identical_pair_count: int

class CrossEncoderProbe:
    """
    Probe 3: Validates S2 cross-encoder pair feeding and scoring.
    
    Checks for common failure modes:
    - Empty or malformed query/candidate pairs
    - Constant scores (broken CE or input processing)
    - Wrong tokenizer or input format
    - No correlation with lexical similarity
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sample_size = config.get('sample_size', 20)
        
        # Thresholds for pass/fail
        self.min_score_std = config.get('min_score_std', 0.1)
        self.max_empty_ratio = config.get('max_empty_ratio', 0.1)  
        self.min_lexical_correlation = config.get('min_lexical_correlation', 0.2)
        self.max_constant_score_ratio = config.get('max_constant_score_ratio', 0.8)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
    async def diagnose_cross_encoder(self, 
                                   evaluation_data: List[Dict[str, Any]], 
                                   retrieval_pipeline: Any) -> 'ProbeResult':
        """
        Run S2 cross-encoder pair feeding sanity check.
        
        Args:
            evaluation_data: List of evaluation samples
            retrieval_pipeline: Lethe retrieval pipeline instance
            
        Returns:
            ProbeResult with pass/fail status and diagnostics
        """
        from .selection_stack_diagnostics import ProbeResult
        
        start_time = time.time()
        
        try:
            # Sample queries and get retrieval candidates
            sample_data = self._sample_queries(evaluation_data, self.sample_size)
            
            # Generate query-candidate pairs for CE analysis
            ce_pairs = await self._generate_ce_pairs(sample_data, retrieval_pipeline)
            
            if not ce_pairs:
                return ProbeResult(
                    probe_name="Cross-Encoder Probe",
                    status="fail",
                    summary="No CE pairs could be generated",
                    details={"error": "No retrieval candidates found"},
                    fix_recommendations=["Check retrieval pipeline - no candidates generated"],
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Run cross-encoder on pairs
            ce_results = await self._run_cross_encoder_analysis(ce_pairs, retrieval_pipeline)
            
            # Analyze CE performance
            stats = self._analyze_ce_results(ce_pairs, ce_results)
            
            # Determine pass/fail status  
            status, issues, fixes = self._evaluate_ce_stats(stats)
            
            # Generate detailed analysis
            details = self._generate_detailed_analysis(stats, ce_pairs, ce_results)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Log key findings
            self._log_findings(stats, status, issues)
            
            return ProbeResult(
                probe_name="Cross-Encoder Probe",
                status=status,
                summary=f"Cross-encoder {status}: {len(issues)} issues found" if issues else f"Cross-encoder {status}",
                details=details,
                fix_recommendations=fixes,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Cross-encoder probe failed: {e}")
            execution_time = (time.time() - start_time) * 1000
            
            return ProbeResult(
                probe_name="Cross-Encoder Probe",
                status="fail",
                summary=f"Probe failed with error: {str(e)}",
                details={"error": str(e)},
                fix_recommendations=[f"Fix cross-encoder probe: {str(e)}"],
                execution_time_ms=execution_time
            )
    
    def _sample_queries(self, evaluation_data: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
        """Sample queries for CE analysis."""
        if len(evaluation_data) <= sample_size:
            return evaluation_data
            
        return np.random.choice(evaluation_data, size=sample_size, replace=False).tolist()
    
    async def _generate_ce_pairs(self, 
                               sample_data: List[Dict[str, Any]], 
                               retrieval_pipeline: Any) -> List[Dict[str, Any]]:
        """Generate query-candidate pairs for CE analysis."""
        pairs = []
        
        for i, sample in enumerate(sample_data):
            query_text = self._extract_query_text(sample)
            if not query_text:
                continue
                
            try:
                # Get retrieval candidates
                candidates = await self._get_retrieval_candidates(query_text, retrieval_pipeline)
                
                # Create pairs with top candidates
                for j, candidate in enumerate(candidates[:5]):  # Top 5 candidates per query
                    candidate_text = self._extract_candidate_text(candidate)
                    
                    pair = {
                        'pair_id': f"query_{i}_cand_{j}",
                        'query_text': query_text,
                        'candidate_text': candidate_text,
                        'query_length': len(query_text) if query_text else 0,
                        'candidate_length': len(candidate_text) if candidate_text else 0,
                        'is_empty_query': not query_text or len(query_text.strip()) == 0,
                        'is_empty_candidate': not candidate_text or len(candidate_text.strip()) == 0,
                        'is_identical': query_text == candidate_text if query_text and candidate_text else False,
                        'lexical_overlap': self._compute_lexical_overlap(query_text, candidate_text)
                    }
                    
                    pairs.append(pair)
                    
            except Exception as e:
                self.logger.warning(f"Failed to generate pairs for query {i}: {e}")
                continue
                
        return pairs
    
    def _extract_query_text(self, sample: Dict[str, Any]) -> Optional[str]:
        """Extract query text from sample."""
        query_fields = ['query', 'question', 'input', 'text', 'prompt']
        
        for field in query_fields:
            if field in sample and sample[field]:
                return str(sample[field])
                
        if 'sample' in sample:
            for field in query_fields:
                if field in sample['sample'] and sample['sample'][field]:
                    return str(sample['sample'][field])
        
        return None
    
    async def _get_retrieval_candidates(self, query_text: str, retrieval_pipeline: Any) -> List[Any]:
        """Get retrieval candidates for the query."""
        try:
            # Try different methods to get candidates
            if hasattr(retrieval_pipeline, 'retrieve'):
                results = await retrieval_pipeline.retrieve(query_text, k=10)
                return self._extract_candidates_from_results(results)
                
            elif hasattr(retrieval_pipeline, 'search'):
                results = await retrieval_pipeline.search(query_text, k=10)
                return self._extract_candidates_from_results(results)
                
            elif hasattr(retrieval_pipeline, 'get_candidates'):
                return await retrieval_pipeline.get_candidates(query_text, k=10)
                
            else:
                self.logger.warning("No candidate retrieval method found")
                return []
                
        except Exception as e:
            self.logger.warning(f"Candidate retrieval failed: {e}")
            return []
    
    def _extract_candidates_from_results(self, results: Any) -> List[Any]:
        """Extract candidate documents from retrieval results."""
        if not results:
            return []
            
        if isinstance(results, dict):
            return results.get('documents', results.get('docs', []))
        elif isinstance(results, list):
            # Extract documents from result objects
            candidates = []
            for result in results:
                if hasattr(result, 'document'):
                    candidates.append(result.document)
                elif hasattr(result, 'content'):
                    candidates.append(result.content)
                else:
                    candidates.append(result)
            return candidates
        else:
            return [results] if results else []
    
    def _extract_candidate_text(self, candidate: Any) -> str:
        """Extract text from candidate document."""
        if isinstance(candidate, str):
            return candidate
        elif isinstance(candidate, dict):
            # Try common text fields
            text_fields = ['text', 'content', 'body', 'passage', 'document']
            for field in text_fields:
                if field in candidate and candidate[field]:
                    return str(candidate[field])
            # Fallback to full dict string
            return str(candidate)
        elif hasattr(candidate, 'text'):
            return str(candidate.text)
        elif hasattr(candidate, 'content'):
            return str(candidate.content)
        else:
            return str(candidate) if candidate else ""
    
    def _compute_lexical_overlap(self, query_text: str, candidate_text: str) -> float:
        """Compute lexical overlap ratio between query and candidate."""
        if not query_text or not candidate_text:
            return 0.0
            
        # Simple word-based overlap
        query_words = set(re.findall(r'\b\w+\b', query_text.lower()))
        candidate_words = set(re.findall(r'\b\w+\b', candidate_text.lower()))
        
        if not query_words:
            return 0.0
            
        overlap = len(query_words.intersection(candidate_words))
        return overlap / len(query_words)
    
    async def _run_cross_encoder_analysis(self, 
                                        ce_pairs: List[Dict[str, Any]], 
                                        retrieval_pipeline: Any) -> List[Dict[str, Any]]:
        """Run cross-encoder on query-candidate pairs."""
        results = []
        
        for pair in ce_pairs:
            try:
                # Get CE score using different possible methods
                score = await self._get_ce_score(
                    pair['query_text'], 
                    pair['candidate_text'], 
                    retrieval_pipeline
                )
                
                result = {
                    'pair_id': pair['pair_id'],
                    'ce_score': score,
                    'score_valid': score is not None and not np.isnan(score),
                    'input_query': pair['query_text'][:100],  # Truncate for logging
                    'input_candidate': pair['candidate_text'][:100],  # Truncate for logging
                    'processing_time_ms': 0  # Could add timing if needed
                }
                
                # Log verbatim CE input for first few pairs
                if len(results) < 5:
                    self.logger.info(f"CE Input: query='{pair['query_text'][:50]}...', "
                                   f"candidate='{pair['candidate_text'][:50]}...' → score={score:.3f}")
                
                results.append(result)
                
            except Exception as e:
                self.logger.warning(f"CE scoring failed for pair {pair['pair_id']}: {e}")
                results.append({
                    'pair_id': pair['pair_id'],
                    'ce_score': None,
                    'score_valid': False,
                    'error': str(e)
                })
                
        return results
    
    async def _get_ce_score(self, query_text: str, candidate_text: str, retrieval_pipeline: Any) -> Optional[float]:
        """Get cross-encoder score for query-candidate pair."""
        try:
            # Try different methods to access cross-encoder
            if hasattr(retrieval_pipeline, 'cross_encoder'):
                ce = retrieval_pipeline.cross_encoder
                return await self._score_with_ce(ce, query_text, candidate_text)
                
            elif hasattr(retrieval_pipeline, 'reranker'):
                reranker = retrieval_pipeline.reranker
                return await self._score_with_ce(reranker, query_text, candidate_text)
                
            elif hasattr(retrieval_pipeline, 'score_pair'):
                return await retrieval_pipeline.score_pair(query_text, candidate_text)
                
            else:
                # Try to find CE in components
                ce = self._find_cross_encoder_in_pipeline(retrieval_pipeline)
                if ce:
                    return await self._score_with_ce(ce, query_text, candidate_text)
                    
                self.logger.warning("No cross-encoder found in pipeline")
                return None
                
        except Exception as e:
            self.logger.warning(f"CE scoring failed: {e}")
            return None
    
    def _find_cross_encoder_in_pipeline(self, pipeline: Any) -> Any:
        """Find cross-encoder component in pipeline."""
        ce_attrs = ['cross_encoder', 'reranker', 'scorer', 'ce']
        
        for attr in ce_attrs:
            if hasattr(pipeline, attr):
                return getattr(pipeline, attr)
                
        # Try nested components
        if hasattr(pipeline, 'components'):
            for component in pipeline.components:
                for attr in ce_attrs:
                    if hasattr(component, attr):
                        return getattr(component, attr)
                        
        return None
    
    async def _score_with_ce(self, ce: Any, query_text: str, candidate_text: str) -> Optional[float]:
        """Score query-candidate pair with cross-encoder."""
        try:
            # Try different scoring methods
            if hasattr(ce, 'score'):
                result = await ce.score(query_text, candidate_text)
                return float(result) if result is not None else None
                
            elif hasattr(ce, 'predict'):
                result = await ce.predict(query_text, candidate_text)
                return float(result) if result is not None else None
                
            elif hasattr(ce, '__call__'):
                result = await ce(query_text, candidate_text)
                return float(result) if result is not None else None
                
            elif hasattr(ce, 'encode'):
                # Some CEs use encode for pairs
                result = await ce.encode([[query_text, candidate_text]])
                if isinstance(result, (list, np.ndarray)) and len(result) > 0:
                    return float(result[0])
                    
            else:
                self.logger.warning(f"CE has no known scoring method: {type(ce)}")
                return None
                
        except Exception as e:
            self.logger.warning(f"CE scoring method failed: {e}")
            return None
    
    def _analyze_ce_results(self, 
                          ce_pairs: List[Dict[str, Any]], 
                          ce_results: List[Dict[str, Any]]) -> CrossEncoderStats:
        """Analyze cross-encoder results for quality metrics."""
        
        # Extract valid results
        valid_results = [r for r in ce_results if r.get('score_valid', False)]
        
        ce_inputs = []
        ce_scores = []
        lexical_overlaps = []
        query_lengths = []
        candidate_lengths = []
        empty_query_count = 0
        empty_candidate_count = 0
        identical_pair_count = 0
        
        for pair, result in zip(ce_pairs, ce_results):
            # Track input characteristics
            ce_inputs.append({
                'query_text': pair['query_text'][:200],  # Truncate for storage
                'candidate_text': pair['candidate_text'][:200],
                'ce_score': result.get('ce_score')
            })
            
            if result.get('score_valid', False):
                ce_scores.append(result['ce_score'])
                lexical_overlaps.append(pair['lexical_overlap'])
                
            query_lengths.append(pair['query_length'])
            candidate_lengths.append(pair['candidate_length'])
            
            if pair['is_empty_query']:
                empty_query_count += 1
            if pair['is_empty_candidate']:
                empty_candidate_count += 1
            if pair['is_identical']:
                identical_pair_count += 1
        
        return CrossEncoderStats(
            ce_inputs=ce_inputs,
            ce_scores=ce_scores,
            lexical_overlaps=lexical_overlaps,
            query_lengths=query_lengths,
            candidate_lengths=candidate_lengths,
            empty_query_count=empty_query_count,
            empty_candidate_count=empty_candidate_count,
            identical_pair_count=identical_pair_count
        )
    
    def _evaluate_ce_stats(self, stats: CrossEncoderStats) -> Tuple[str, List[str], List[str]]:
        """Evaluate CE statistics to determine pass/fail status."""
        issues = []
        fixes = []
        
        total_pairs = len(stats.ce_inputs)
        
        # Check for empty inputs
        empty_query_ratio = stats.empty_query_count / total_pairs if total_pairs > 0 else 1.0
        empty_candidate_ratio = stats.empty_candidate_count / total_pairs if total_pairs > 0 else 1.0
        
        if empty_query_ratio > self.max_empty_ratio:
            issues.append(f"High empty query ratio: {empty_query_ratio:.1%}")
            fixes.append("Check query extraction - queries are empty or malformed")
            
        if empty_candidate_ratio > self.max_empty_ratio:
            issues.append(f"High empty candidate ratio: {empty_candidate_ratio:.1%}")
            fixes.append("Check candidate retrieval - candidates are empty or malformed")
        
        # Check identical pairs
        identical_ratio = stats.identical_pair_count / total_pairs if total_pairs > 0 else 0.0
        if identical_ratio > 0.1:  # More than 10% identical
            issues.append(f"High identical pair ratio: {identical_ratio:.1%}")
            fixes.append("Query and candidate texts are identical - check pair generation")
        
        # Check score validity and distribution
        if not stats.ce_scores:
            issues.append("No valid CE scores generated")
            fixes.append("Cross-encoder is not producing scores - check CE implementation")
        else:
            # Check score spread
            score_std = np.std(stats.ce_scores)
            if score_std < self.min_score_std:
                issues.append(f"CE scores have low variance: {score_std:.3f}")
                fixes.append("CE scores are constant - check tokenizer, model weights, or input format")
            
            # Check for constant scores
            unique_scores = len(set(np.round(stats.ce_scores, 3)))
            if unique_scores == 1:
                issues.append(f"All CE scores are identical: {stats.ce_scores[0]:.3f}")
                fixes.append("CE is returning constant score - check model and input processing")
            elif unique_scores / len(stats.ce_scores) < 0.2:  # Less than 20% unique scores
                issues.append(f"CE scores lack diversity: {unique_scores} unique values")
                fixes.append("CE scores are too similar - check model sensitivity and inputs")
            
            # Check correlation with lexical overlap
            if len(stats.lexical_overlaps) > 5:  # Need enough samples
                correlation = np.corrcoef(stats.ce_scores, stats.lexical_overlaps)[0, 1]
                if not np.isnan(correlation) and correlation < self.min_lexical_correlation:
                    issues.append(f"Low CE-lexical correlation: {correlation:.3f}")
                    fixes.append("CE scores don't correlate with lexical overlap - check CE training or inputs")
        
        # Check score range
        if stats.ce_scores:
            score_range = max(stats.ce_scores) - min(stats.ce_scores)
            if score_range < 0.1:
                issues.append(f"CE score range too narrow: {score_range:.3f}")
                fixes.append("CE scores have minimal range - check model sensitivity")
        
        # Determine status
        if not issues:
            status = "pass"
        elif len(issues) <= 2 and stats.ce_scores:
            status = "warning"
        else:
            status = "fail"
            
        return status, issues, fixes
    
    def _generate_detailed_analysis(self, 
                                  stats: CrossEncoderStats,
                                  ce_pairs: List[Dict[str, Any]], 
                                  ce_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed analysis for reporting."""
        
        valid_scores = [s for s in stats.ce_scores if not np.isnan(s)]
        
        analysis = {
            'pairs_analyzed': len(stats.ce_inputs),
            'valid_scores': len(valid_scores),
            'valid_score_ratio': len(valid_scores) / len(stats.ce_inputs) if stats.ce_inputs else 0.0,
            
            # Input quality metrics
            'empty_query_count': stats.empty_query_count,
            'empty_candidate_count': stats.empty_candidate_count,
            'identical_pair_count': stats.identical_pair_count,
            'avg_query_length': float(np.mean(stats.query_lengths)) if stats.query_lengths else 0.0,
            'avg_candidate_length': float(np.mean(stats.candidate_lengths)) if stats.candidate_lengths else 0.0,
            
            # Score statistics
            'score_mean': float(np.mean(valid_scores)) if valid_scores else None,
            'score_std': float(np.std(valid_scores)) if valid_scores else None,
            'score_min': float(np.min(valid_scores)) if valid_scores else None,
            'score_max': float(np.max(valid_scores)) if valid_scores else None,
            'score_range': float(np.max(valid_scores) - np.min(valid_scores)) if valid_scores else None,
            'unique_scores': len(set(np.round(valid_scores, 3))) if valid_scores else 0,
            
            # Correlation with lexical overlap
            'lexical_correlation': float(np.corrcoef(stats.ce_scores, stats.lexical_overlaps)[0, 1]) 
                if len(stats.ce_scores) > 1 and len(stats.lexical_overlaps) > 1 else None,
            
            # Sample inputs for inspection
            'sample_inputs': [
                {
                    'query': inp['query_text'][:100],
                    'candidate': inp['candidate_text'][:100],
                    'score': inp['ce_score']
                }
                for inp in stats.ce_inputs[:10]  # First 10 samples
            ]
        }
        
        return analysis
    
    def _log_findings(self, stats: CrossEncoderStats, status: str, issues: List[str]):
        """Log key findings from the probe."""
        self.logger.info(f"Cross-Encoder Probe: {status.upper()}")
        self.logger.info(f"Analyzed {len(stats.ce_inputs)} CE pairs")
        self.logger.info(f"Valid scores: {len(stats.ce_scores)}")
        
        if stats.ce_scores:
            self.logger.info(f"Score mean: {np.mean(stats.ce_scores):.3f}")
            self.logger.info(f"Score std: {np.std(stats.ce_scores):.3f}")
            self.logger.info(f"Score range: {max(stats.ce_scores) - min(stats.ce_scores):.3f}")
        
        self.logger.info(f"Empty queries: {stats.empty_query_count}")
        self.logger.info(f"Empty candidates: {stats.empty_candidate_count}")
        
        if issues:
            self.logger.warning(f"Issues found: {', '.join(issues)}")
        else:
            self.logger.info("No issues detected in cross-encoder")