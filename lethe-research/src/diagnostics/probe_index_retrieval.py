"""
Probe 2: S1 Index/Space Audit  
==============================

Validates that the index search is retrieving relevant items and that
query/index embeddings are properly aligned. Tests 50 items and checks:

- For each query, print (top5 ids, similarities)
- Expected: max similarity ≥0.25 for code/debug tasks
- If all similarities ~0.0, indicates encoder/index mismatch
- Verify stored document vectors' model_id hash matches query encoder
- Check K1 actually returns >0 items (not filtered by quotas)

Red flags: all similarities ~0.0, empty results, model hash mismatch
→ encoder/index mismatch or wrong retrieval configuration
"""

import numpy as np
import pandas as pd
import hashlib
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class RetrievalStats:
    """Statistics for retrieval analysis."""
    query_results: List[Dict[str, Any]]  # Per-query retrieval results
    similarities_by_query: List[List[float]]  # Top K similarities per query
    retrieved_ids_by_query: List[List[str]]  # Top K document IDs per query
    model_hashes: Set[str]  # Model hashes found in index
    empty_result_count: int
    avg_max_similarity: float
    relevant_items_found: int

class IndexRetrievalProbe:
    """
    Probe 2: Validates S1 index retrieval and embedding space alignment.
    
    Checks for common failure modes:
    - Encoder/index mismatch (different models)
    - Poor retrieval performance (all similarities ~0)
    - Empty retrieval results (quota/filtering issues)
    - Wrong retrieval configuration
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sample_size = config.get('sample_size', 50)
        self.top_k = config.get('top_k', 5)
        
        # Thresholds for pass/fail
        self.min_max_similarity = config.get('min_max_similarity', 0.25)
        self.min_relevant_items = config.get('min_relevant_items', 10)  # Minimum relevant items across all queries
        self.max_empty_ratio = config.get('max_empty_ratio', 0.1)  # Max 10% empty results
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
    async def diagnose_index_retrieval(self, 
                                     evaluation_data: List[Dict[str, Any]], 
                                     retrieval_pipeline: Any) -> 'ProbeResult':
        """
        Run S1 index/space audit.
        
        Args:
            evaluation_data: List of evaluation samples
            retrieval_pipeline: Lethe retrieval pipeline instance
            
        Returns:
            ProbeResult with pass/fail status and diagnostics
        """
        from .selection_stack_diagnostics import ProbeResult
        
        start_time = time.time()
        
        try:
            # Sample queries for analysis
            sample_data = self._sample_queries(evaluation_data, self.sample_size)
            
            # Run retrieval for each query
            retrieval_results = await self._run_retrieval_tests(sample_data, retrieval_pipeline)
            
            # Analyze retrieval performance
            stats = self._analyze_retrieval_results(retrieval_results)
            
            # Check model hash alignment
            model_alignment = await self._check_model_alignment(retrieval_pipeline)
            stats.model_hashes = model_alignment.get('index_model_hashes', set())
            
            # Determine pass/fail status
            status, issues, fixes = self._evaluate_retrieval_stats(stats, model_alignment)
            
            # Generate detailed analysis
            details = self._generate_detailed_analysis(stats, model_alignment, retrieval_results)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Log key findings
            self._log_findings(stats, status, issues)
            
            return ProbeResult(
                probe_name="Index Retrieval Probe",
                status=status,
                summary=f"Index retrieval {status}: {len(issues)} issues found" if issues else f"Index retrieval {status}",
                details=details,
                fix_recommendations=fixes,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Index retrieval probe failed: {e}")
            execution_time = (time.time() - start_time) * 1000
            
            return ProbeResult(
                probe_name="Index Retrieval Probe",
                status="fail", 
                summary=f"Probe failed with error: {str(e)}",
                details={"error": str(e)},
                fix_recommendations=[f"Fix index retrieval probe: {str(e)}"],
                execution_time_ms=execution_time
            )
    
    def _sample_queries(self, evaluation_data: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
        """Sample queries for retrieval analysis."""
        if len(evaluation_data) <= sample_size:
            return evaluation_data
            
        # Stratified sampling by dataset if available
        return np.random.choice(evaluation_data, size=sample_size, replace=False).tolist()
    
    async def _run_retrieval_tests(self, 
                                 sample_data: List[Dict[str, Any]], 
                                 retrieval_pipeline: Any) -> List[Dict[str, Any]]:
        """Run retrieval tests for sampled queries."""
        results = []
        
        for i, sample in enumerate(sample_data):
            query_text = self._extract_query_text(sample)
            if not query_text:
                continue
                
            try:
                # Run retrieval with different K values to test
                retrieval_result = await self._retrieve_with_pipeline(
                    query_text, retrieval_pipeline, k=self.top_k
                )
                
                if retrieval_result:
                    result = {
                        'query_id': f"query_{i}",
                        'query_text': query_text[:100] + "..." if len(query_text) > 100 else query_text,
                        'retrieved_count': len(retrieval_result.get('documents', [])),
                        'top_similarities': retrieval_result.get('similarities', [])[:self.top_k],
                        'top_document_ids': retrieval_result.get('document_ids', [])[:self.top_k],
                        'max_similarity': max(retrieval_result.get('similarities', [0])) if retrieval_result.get('similarities') else 0.0,
                        'gold_overlap': self._compute_gold_overlap(sample, retrieval_result),
                        'has_span_hit': self._check_span_hit(sample, retrieval_result),
                        'has_symbol_hit': self._check_symbol_hit(sample, retrieval_result)
                    }
                else:
                    result = {
                        'query_id': f"query_{i}",
                        'query_text': query_text[:100] + "..." if len(query_text) > 100 else query_text,
                        'retrieved_count': 0,
                        'top_similarities': [],
                        'top_document_ids': [],
                        'max_similarity': 0.0,
                        'gold_overlap': 0,
                        'has_span_hit': False,
                        'has_symbol_hit': False
                    }
                
                results.append(result)
                
                # Log sample results for inspection
                if i < 10:  # Log first 10 for debugging
                    self.logger.info(f"Sample {i}: top5_ids={result['top_document_ids']}, "
                                   f"sims={[f'{s:.3f}' for s in result['top_similarities']]}")
                    
            except Exception as e:
                self.logger.warning(f"Retrieval failed for query {i}: {e}")
                results.append({
                    'query_id': f"query_{i}",
                    'query_text': query_text[:100],
                    'retrieved_count': 0,
                    'error': str(e)
                })
                
        return results
    
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
    
    async def _retrieve_with_pipeline(self, 
                                    query_text: str, 
                                    retrieval_pipeline: Any, 
                                    k: int = 5) -> Optional[Dict[str, Any]]:
        """Run retrieval using the pipeline."""
        try:
            # Try different methods to run retrieval
            if hasattr(retrieval_pipeline, 'retrieve'):
                results = await retrieval_pipeline.retrieve(query_text, k=k)
                return self._normalize_retrieval_results(results)
                
            elif hasattr(retrieval_pipeline, 'search'):
                results = await retrieval_pipeline.search(query_text, k=k)  
                return self._normalize_retrieval_results(results)
                
            elif hasattr(retrieval_pipeline, 'query'):
                results = await retrieval_pipeline.query(query_text, k=k)
                return self._normalize_retrieval_results(results)
                
            else:
                # Try to find retrieval method in components
                retrieval_method = self._find_retrieval_method(retrieval_pipeline)
                if retrieval_method:
                    results = await retrieval_method(query_text, k=k)
                    return self._normalize_retrieval_results(results)
                    
                self.logger.warning("No retrieval method found in pipeline")
                return None
                
        except Exception as e:
            self.logger.warning(f"Retrieval failed: {e}")
            return None
    
    def _find_retrieval_method(self, pipeline: Any) -> Optional[callable]:
        """Find retrieval method in pipeline components."""
        retrieval_methods = ['retrieve', 'search', 'query', 'get_similar']
        
        for method_name in retrieval_methods:
            if hasattr(pipeline, method_name):
                return getattr(pipeline, method_name)
                
        # Try nested components
        if hasattr(pipeline, 'components'):
            for component in pipeline.components:
                for method_name in retrieval_methods:
                    if hasattr(component, method_name):
                        return getattr(component, method_name)
                        
        return None
    
    def _normalize_retrieval_results(self, results: Any) -> Dict[str, Any]:
        """Normalize retrieval results to standard format."""
        if results is None:
            return {'documents': [], 'similarities': [], 'document_ids': []}
            
        # Handle different result formats
        if isinstance(results, dict):
            return {
                'documents': results.get('documents', results.get('docs', [])),
                'similarities': results.get('similarities', results.get('scores', results.get('distances', []))),
                'document_ids': results.get('document_ids', results.get('ids', []))
            }
            
        elif isinstance(results, list):
            # List of result objects
            docs = []
            sims = []
            ids = []
            
            for result in results:
                if hasattr(result, 'document'):
                    docs.append(result.document)
                elif hasattr(result, 'content'):
                    docs.append(result.content)
                    
                if hasattr(result, 'similarity'):
                    sims.append(result.similarity)
                elif hasattr(result, 'score'):
                    sims.append(result.score)
                elif hasattr(result, 'distance'):
                    # Convert distance to similarity
                    sims.append(1.0 / (1.0 + result.distance))
                    
                if hasattr(result, 'id'):
                    ids.append(result.id)
                elif hasattr(result, 'doc_id'):
                    ids.append(result.doc_id)
                else:
                    ids.append(f"doc_{len(ids)}")
                    
            return {
                'documents': docs,
                'similarities': sims,
                'document_ids': ids
            }
            
        else:
            # Single result or unknown format
            return {
                'documents': [results] if results else [],
                'similarities': [1.0] if results else [],
                'document_ids': ['doc_0'] if results else []
            }
    
    def _compute_gold_overlap(self, sample: Dict[str, Any], retrieval_result: Dict[str, Any]) -> int:
        """Compute overlap with gold/ground truth documents."""
        # Try to find gold document IDs in sample
        gold_ids = set()
        
        # Common fields for gold standard
        gold_fields = ['gold_ids', 'relevant_ids', 'ground_truth', 'answer_docs']
        
        for field in gold_fields:
            if field in sample and sample[field]:
                if isinstance(sample[field], list):
                    gold_ids.update(str(id) for id in sample[field])
                else:
                    gold_ids.add(str(sample[field]))
        
        if not gold_ids:
            return 0
            
        # Check overlap with retrieved IDs
        retrieved_ids = set(str(id) for id in retrieval_result.get('document_ids', []))
        overlap = len(gold_ids.intersection(retrieved_ids))
        
        return overlap
    
    def _check_span_hit(self, sample: Dict[str, Any], retrieval_result: Dict[str, Any]) -> bool:
        """Check if retrieval includes documents with answer spans."""
        # Try to find answer text in retrieved documents
        answer_text = sample.get('answer', sample.get('target', ''))
        if not answer_text:
            return False
            
        documents = retrieval_result.get('documents', [])
        for doc in documents:
            doc_text = str(doc) if doc else ''
            if answer_text.lower() in doc_text.lower():
                return True
                
        return False
    
    def _check_symbol_hit(self, sample: Dict[str, Any], retrieval_result: Dict[str, Any]) -> bool:
        """Check if retrieval includes documents with relevant code symbols."""
        # Look for code-related keywords in documents
        code_keywords = ['function', 'class', 'def ', 'import', 'return', 'if ', 'for ', 'while ']
        
        documents = retrieval_result.get('documents', [])
        for doc in documents:
            doc_text = str(doc).lower() if doc else ''
            if any(keyword in doc_text for keyword in code_keywords):
                return True
                
        return False
    
    def _analyze_retrieval_results(self, retrieval_results: List[Dict[str, Any]]) -> RetrievalStats:
        """Analyze retrieval results for quality metrics."""
        
        similarities_by_query = []
        retrieved_ids_by_query = []
        max_similarities = []
        empty_count = 0
        relevant_count = 0
        
        for result in retrieval_results:
            if 'error' in result:
                empty_count += 1
                continue
                
            sims = result.get('top_similarities', [])
            ids = result.get('top_document_ids', [])
            max_sim = result.get('max_similarity', 0.0)
            
            similarities_by_query.append(sims)
            retrieved_ids_by_query.append(ids)
            max_similarities.append(max_sim)
            
            if result.get('retrieved_count', 0) == 0:
                empty_count += 1
            else:
                # Count as relevant if max similarity above threshold or has hits
                if (max_sim >= self.min_max_similarity or 
                    result.get('has_span_hit', False) or 
                    result.get('has_symbol_hit', False) or
                    result.get('gold_overlap', 0) > 0):
                    relevant_count += 1
        
        avg_max_similarity = np.mean(max_similarities) if max_similarities else 0.0
        
        return RetrievalStats(
            query_results=retrieval_results,
            similarities_by_query=similarities_by_query,
            retrieved_ids_by_query=retrieved_ids_by_query,
            model_hashes=set(),  # Will be filled by model alignment check
            empty_result_count=empty_count,
            avg_max_similarity=avg_max_similarity,
            relevant_items_found=relevant_count
        )
    
    async def _check_model_alignment(self, retrieval_pipeline: Any) -> Dict[str, Any]:
        """Check if query encoder and index use the same model."""
        alignment_info = {
            'query_encoder_hash': None,
            'index_model_hashes': set(),
            'alignment_status': 'unknown'
        }
        
        try:
            # Get query encoder hash
            query_encoder = self._find_query_encoder(retrieval_pipeline)
            if query_encoder:
                encoder_hash = self._get_model_hash(query_encoder)
                alignment_info['query_encoder_hash'] = encoder_hash
            
            # Get index model hashes
            index_hashes = await self._get_index_model_hashes(retrieval_pipeline)
            alignment_info['index_model_hashes'] = index_hashes
            
            # Check alignment
            if (alignment_info['query_encoder_hash'] and 
                alignment_info['query_encoder_hash'] in index_hashes):
                alignment_info['alignment_status'] = 'aligned'
            elif alignment_info['query_encoder_hash'] and index_hashes:
                alignment_info['alignment_status'] = 'misaligned'
            else:
                alignment_info['alignment_status'] = 'unknown'
                
        except Exception as e:
            self.logger.warning(f"Model alignment check failed: {e}")
            alignment_info['error'] = str(e)
            
        return alignment_info
    
    def _find_query_encoder(self, pipeline: Any) -> Any:
        """Find query encoder in pipeline."""
        encoder_attrs = ['encoder', 'query_encoder', 'embedding_model']
        
        for attr in encoder_attrs:
            if hasattr(pipeline, attr):
                return getattr(pipeline, attr)
                
        return None
    
    def _get_model_hash(self, model: Any) -> Optional[str]:
        """Get hash identifier for a model."""
        try:
            # Try common model identification methods
            if hasattr(model, 'model_name'):
                return hashlib.sha256(str(model.model_name).encode()).hexdigest()[:16]
            elif hasattr(model, 'name'):
                return hashlib.sha256(str(model.name).encode()).hexdigest()[:16]  
            elif hasattr(model, 'config'):
                config_str = str(model.config)
                return hashlib.sha256(config_str.encode()).hexdigest()[:16]
            else:
                # Fallback: hash model class and parameters
                model_str = str(type(model)) + str(getattr(model, '__dict__', {}))
                return hashlib.sha256(model_str.encode()).hexdigest()[:16]
        except Exception:
            return None
    
    async def _get_index_model_hashes(self, pipeline: Any) -> Set[str]:
        """Get model hashes from index metadata."""
        hashes = set()
        
        try:
            # Try to access index
            if hasattr(pipeline, 'index'):
                index = pipeline.index
                
                # Look for model metadata
                if hasattr(index, 'metadata'):
                    metadata = index.metadata
                    if 'model_hash' in metadata:
                        hashes.add(metadata['model_hash'])
                    elif 'encoder_hash' in metadata:
                        hashes.add(metadata['encoder_hash'])
                
                # Try to get from index configuration
                if hasattr(index, 'config'):
                    config_hash = self._get_model_hash(index)
                    if config_hash:
                        hashes.add(config_hash)
                        
        except Exception as e:
            self.logger.warning(f"Failed to get index model hashes: {e}")
            
        return hashes
    
    def _evaluate_retrieval_stats(self, 
                                stats: RetrievalStats, 
                                model_alignment: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
        """Evaluate retrieval statistics to determine pass/fail."""
        issues = []
        fixes = []
        
        # Check empty results
        empty_ratio = stats.empty_result_count / len(stats.query_results) if stats.query_results else 1.0
        if empty_ratio > self.max_empty_ratio:
            issues.append(f"High empty result ratio: {empty_ratio:.1%}")
            fixes.append("Check retrieval configuration and index availability")
        
        # Check average max similarity
        if stats.avg_max_similarity < self.min_max_similarity:
            issues.append(f"Low average max similarity: {stats.avg_max_similarity:.3f}")
            fixes.append("Check encoder/index alignment and embedding quality")
        
        # Check relevant items found
        if stats.relevant_items_found < self.min_relevant_items:
            issues.append(f"Few relevant items found: {stats.relevant_items_found}")
            fixes.append("Increase K1, check retrieval parameters, or improve embeddings")
        
        # Check model alignment
        if model_alignment.get('alignment_status') == 'misaligned':
            issues.append("Query encoder and index use different models")
            fixes.append("Ensure query encoder matches index encoder - rebuild index if needed")
        elif model_alignment.get('alignment_status') == 'unknown':
            issues.append("Cannot verify query/index encoder alignment")
            fixes.append("Add model hash metadata to index and verify encoder consistency")
        
        # Check for all-zero similarities
        all_sims = [sim for sims in stats.similarities_by_query for sim in sims]
        if all_sims and max(all_sims) < 0.01:
            issues.append("All similarities near zero - possible encoder mismatch")
            fixes.append("Verify same encoder used for queries and index embeddings")
        
        # Determine status
        if not issues:
            status = "pass"
        elif len(issues) <= 2 and stats.relevant_items_found > 0:
            status = "warning"
        else:
            status = "fail"
            
        return status, issues, fixes
    
    def _generate_detailed_analysis(self, 
                                  stats: RetrievalStats,
                                  model_alignment: Dict[str, Any], 
                                  retrieval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed analysis for reporting."""
        
        # Compute additional statistics
        all_max_sims = [r.get('max_similarity', 0.0) for r in retrieval_results if 'error' not in r]
        all_overlaps = [r.get('gold_overlap', 0) for r in retrieval_results if 'error' not in r]
        span_hits = sum(1 for r in retrieval_results if r.get('has_span_hit', False))
        symbol_hits = sum(1 for r in retrieval_results if r.get('has_symbol_hit', False))
        
        return {
            'queries_tested': len(stats.query_results),
            'empty_results': stats.empty_result_count,
            'empty_result_ratio': stats.empty_result_count / len(stats.query_results) if stats.query_results else 1.0,
            
            # Similarity statistics
            'max_similarity_mean': float(np.mean(all_max_sims)) if all_max_sims else 0.0,
            'max_similarity_std': float(np.std(all_max_sims)) if all_max_sims else 0.0,
            'max_similarity_min': float(np.min(all_max_sims)) if all_max_sims else 0.0,
            'max_similarity_max': float(np.max(all_max_sims)) if all_max_sims else 0.0,
            
            # Relevance metrics
            'relevant_items_found': stats.relevant_items_found,
            'relevance_ratio': stats.relevant_items_found / len(stats.query_results) if stats.query_results else 0.0,
            'gold_overlap_mean': float(np.mean(all_overlaps)) if all_overlaps else 0.0,
            'span_hit_count': span_hits,
            'span_hit_ratio': span_hits / len(retrieval_results) if retrieval_results else 0.0,
            'symbol_hit_count': symbol_hits,
            'symbol_hit_ratio': symbol_hits / len(retrieval_results) if retrieval_results else 0.0,
            
            # Model alignment
            'query_encoder_hash': model_alignment.get('query_encoder_hash'),
            'index_model_hashes': list(model_alignment.get('index_model_hashes', [])),
            'alignment_status': model_alignment.get('alignment_status'),
            
            # Sample results for inspection
            'sample_results': [
                {
                    'query_id': r['query_id'],
                    'query_text': r.get('query_text', '')[:50],
                    'max_similarity': r.get('max_similarity', 0.0),
                    'top_similarities': r.get('top_similarities', [])[:3],
                    'retrieved_count': r.get('retrieved_count', 0),
                    'gold_overlap': r.get('gold_overlap', 0)
                }
                for r in retrieval_results[:10]  # First 10 samples
                if 'error' not in r
            ]
        }
    
    def _log_findings(self, stats: RetrievalStats, status: str, issues: List[str]):
        """Log key findings from the probe."""
        self.logger.info(f"Index Retrieval Probe: {status.upper()}")
        self.logger.info(f"Tested {len(stats.query_results)} queries")
        self.logger.info(f"Empty results: {stats.empty_result_count} ({stats.empty_result_count/len(stats.query_results)*100:.1f}%)")
        self.logger.info(f"Average max similarity: {stats.avg_max_similarity:.3f}")
        self.logger.info(f"Relevant items found: {stats.relevant_items_found}")
        
        if issues:
            self.logger.warning(f"Issues found: {', '.join(issues)}")
        else:
            self.logger.info("No issues detected in index retrieval")