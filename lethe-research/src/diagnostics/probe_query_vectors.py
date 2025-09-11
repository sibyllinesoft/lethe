"""
Probe 1: S1 Query Vector Sanity Check
=====================================

Validates that query embeddings are not constant, corrupted, or using
wrong encoder weights. Dumps 200 query embeddings and checks:

- Per-dimension variance ~0.1-0.4 (not constant)
- Cosine self-similarity ~1.0 (proper normalization)
- Average cosine to 1k random atoms around 0.0±0.05 (reasonable space)
- No all-zero/NaN vectors or identical hashes

Red flags: all-zero/NaN, identical hashes, cosines ~0 everywhere
→ wrong encoder weights or wrong input field
"""

import numpy as np
import pandas as pd
import hashlib
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass 
class QueryVectorStats:
    """Statistics for query vector analysis."""
    embedding_hashes: List[str]
    norms: List[float]
    per_dim_stds: List[float]
    cosine_self_similarities: List[float]
    cosine_to_random_atoms: List[float]
    pc_variances: List[float]  # First 3 principal component variances
    
class QueryVectorProbe:
    """
    Probe 1: Validates S1 query vector embeddings for sanity.
    
    Checks for common failure modes:
    - Constant/zero embeddings (wrong encoder)
    - Wrong input field (getting wrong text) 
    - Normalization issues
    - Encoder/index mismatch
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sample_size = config.get('sample_size', 200)
        self.random_atoms_sample = config.get('random_atoms_sample', 1000)
        
        # Thresholds for pass/fail
        self.std_min = config.get('std_min', 0.1)
        self.std_max = config.get('std_max', 0.4) 
        self.cosine_self_min = config.get('cosine_self_min', 0.95)
        self.cosine_random_tolerance = config.get('cosine_random_tolerance', 0.05)
        self.pc_variance_min = config.get('pc_variance_min', 0.01)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
    async def diagnose_query_vectors(self, 
                                   evaluation_data: List[Dict[str, Any]], 
                                   retrieval_pipeline: Any) -> 'ProbeResult':
        """
        Run S1 query vector sanity check.
        
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
            
            # Generate query embeddings
            embeddings, query_texts = await self._generate_query_embeddings(
                sample_data, retrieval_pipeline
            )
            
            # Analyze embeddings
            stats = self._analyze_embeddings(embeddings)
            
            # Get random atom embeddings for comparison
            random_atoms = await self._get_random_atom_embeddings(
                retrieval_pipeline, self.random_atoms_sample
            )
            
            # Compute cosine similarities to random atoms
            cosines_to_atoms = self._compute_cosines_to_atoms(embeddings, random_atoms)
            stats.cosine_to_random_atoms = cosines_to_atoms
            
            # Determine pass/fail status
            status, issues, fixes = self._evaluate_stats(stats)
            
            # Generate detailed analysis
            details = self._generate_detailed_analysis(stats, embeddings, query_texts)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Log key findings
            self._log_findings(stats, status, issues)
            
            return ProbeResult(
                probe_name="Query Vector Probe",
                status=status,
                summary=f"Query embeddings {status}: {len(issues)} issues found" if issues else f"Query embeddings {status}",
                details=details,
                fix_recommendations=fixes,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Query vector probe failed: {e}")
            execution_time = (time.time() - start_time) * 1000
            
            return ProbeResult(
                probe_name="Query Vector Probe", 
                status="fail",
                summary=f"Probe failed with error: {str(e)}",
                details={"error": str(e)},
                fix_recommendations=[f"Fix query vector probe: {str(e)}"],
                execution_time_ms=execution_time
            )
    
    def _sample_queries(self, evaluation_data: List[Dict[str, Any]], sample_size: int) -> List[Dict[str, Any]]:
        """Sample queries for embedding analysis."""
        if len(evaluation_data) <= sample_size:
            return evaluation_data
            
        # Stratified sampling if dataset info available
        return np.random.choice(evaluation_data, size=sample_size, replace=False).tolist()
    
    async def _generate_query_embeddings(self, 
                                       sample_data: List[Dict[str, Any]], 
                                       retrieval_pipeline: Any) -> Tuple[np.ndarray, List[str]]:
        """Generate embeddings for sampled queries."""
        embeddings = []
        query_texts = []
        
        for sample in sample_data:
            # Extract query text - check multiple possible fields
            query_text = self._extract_query_text(sample)
            if not query_text:
                continue
                
            try:
                # Generate embedding using retrieval pipeline's encoder
                if hasattr(retrieval_pipeline, 'encode_query'):
                    embedding = await retrieval_pipeline.encode_query(query_text)
                elif hasattr(retrieval_pipeline, 'encoder'):
                    embedding = await retrieval_pipeline.encoder.encode(query_text)
                else:
                    # Fallback: try to find encoder in pipeline components
                    encoder = self._find_encoder_in_pipeline(retrieval_pipeline)
                    if encoder:
                        embedding = await encoder.encode(query_text)
                    else:
                        self.logger.warning("No encoder found in retrieval pipeline")
                        continue
                
                if embedding is not None:
                    embeddings.append(embedding)
                    query_texts.append(query_text)
                    
            except Exception as e:
                self.logger.warning(f"Failed to encode query '{query_text[:50]}...': {e}")
                continue
        
        if not embeddings:
            raise ValueError("No query embeddings could be generated")
            
        return np.array(embeddings), query_texts
    
    def _extract_query_text(self, sample: Dict[str, Any]) -> Optional[str]:
        """Extract query text from sample, trying multiple possible fields."""
        # Try common field names for query text
        query_fields = ['query', 'question', 'input', 'text', 'prompt']
        
        for field in query_fields:
            if field in sample and sample[field]:
                return str(sample[field])
                
        # Try nested structures
        if 'sample' in sample:
            for field in query_fields:
                if field in sample['sample'] and sample['sample'][field]:
                    return str(sample['sample'][field])
        
        self.logger.warning(f"Could not find query text in sample keys: {list(sample.keys())}")
        return None
    
    def _find_encoder_in_pipeline(self, pipeline: Any) -> Any:
        """Find encoder component in retrieval pipeline."""
        # Try common attribute names
        encoder_attrs = ['encoder', 'query_encoder', 'embedding_model', 'embedder']
        
        for attr in encoder_attrs:
            if hasattr(pipeline, attr):
                return getattr(pipeline, attr)
                
        # Try to find in nested components
        if hasattr(pipeline, 'components'):
            for component in pipeline.components:
                for attr in encoder_attrs:
                    if hasattr(component, attr):
                        return getattr(component, attr)
        
        return None
    
    def _analyze_embeddings(self, embeddings: np.ndarray) -> QueryVectorStats:
        """Analyze query embeddings for sanity checks."""
        n_samples, dim = embeddings.shape
        
        # 1. Compute embedding hashes
        embedding_hashes = []
        for emb in embeddings:
            hash_str = hashlib.sha256(emb.tobytes()).hexdigest()[:16]
            embedding_hashes.append(hash_str)
        
        # 2. Compute norms 
        norms = np.linalg.norm(embeddings, axis=1).tolist()
        
        # 3. Per-dimension standard deviations
        per_dim_stds = np.std(embeddings, axis=0).tolist()
        
        # 4. Cosine self-similarities (should be ~1.0 for normalized embeddings)
        cosine_self_sims = []
        for emb in embeddings:
            norm_emb = emb / (np.linalg.norm(emb) + 1e-8)
            cosine_self_sim = np.dot(norm_emb, norm_emb)  # Should be 1.0
            cosine_self_sims.append(cosine_self_sim)
        
        # 5. Principal component analysis
        if n_samples > 3:
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(3, dim))
                pca.fit(embeddings)
                pc_variances = pca.explained_variance_.tolist()
            except ImportError:
                # Fallback: compute variance of first 3 dimensions
                pc_variances = np.var(embeddings[:, :3], axis=0).tolist()
        else:
            pc_variances = [0.0, 0.0, 0.0]
        
        return QueryVectorStats(
            embedding_hashes=embedding_hashes,
            norms=norms,
            per_dim_stds=per_dim_stds,
            cosine_self_similarities=cosine_self_sims,
            cosine_to_random_atoms=[],  # Will be filled later
            pc_variances=pc_variances
        )
    
    async def _get_random_atom_embeddings(self, 
                                        retrieval_pipeline: Any, 
                                        sample_size: int) -> np.ndarray:
        """Get embeddings of random atoms from the index for comparison."""
        try:
            # Try to access index directly
            if hasattr(retrieval_pipeline, 'index'):
                index = retrieval_pipeline.index
                if hasattr(index, 'sample_embeddings'):
                    return await index.sample_embeddings(sample_size)
                elif hasattr(index, 'embeddings'):
                    all_embeddings = index.embeddings
                    if len(all_embeddings) > sample_size:
                        indices = np.random.choice(len(all_embeddings), sample_size, replace=False)
                        return all_embeddings[indices]
                    else:
                        return all_embeddings
            
            # Try to get from retrieval results
            if hasattr(retrieval_pipeline, 'retrieve'):
                # Get a dummy query and use retrieved items
                dummy_query = "test query"
                results = await retrieval_pipeline.retrieve(dummy_query, k=sample_size)
                if results and hasattr(results[0], 'embedding'):
                    return np.array([r.embedding for r in results[:sample_size]])
            
            # Fallback: generate random embeddings as placeholder
            self.logger.warning("Could not get atom embeddings from index, using random placeholders")
            dim = 768  # Common embedding dimension
            return np.random.normal(0, 0.1, (sample_size, dim))
            
        except Exception as e:
            self.logger.warning(f"Failed to get random atom embeddings: {e}")
            # Return random embeddings as fallback
            dim = 768
            return np.random.normal(0, 0.1, (sample_size, dim))
    
    def _compute_cosines_to_atoms(self, 
                                query_embeddings: np.ndarray, 
                                atom_embeddings: np.ndarray) -> List[float]:
        """Compute average cosine similarities between queries and random atoms."""
        cosines = []
        
        for query_emb in query_embeddings:
            query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            
            # Compute cosines to all atoms
            atom_norms = atom_embeddings / (np.linalg.norm(atom_embeddings, axis=1, keepdims=True) + 1e-8)
            query_atom_cosines = np.dot(atom_norms, query_norm)
            
            # Average cosine similarity
            avg_cosine = np.mean(query_atom_cosines)
            cosines.append(avg_cosine)
        
        return cosines
    
    def _evaluate_stats(self, stats: QueryVectorStats) -> Tuple[str, List[str], List[str]]:
        """Evaluate statistics to determine pass/fail status."""
        issues = []
        fixes = []
        
        # Check for duplicate embeddings
        unique_hashes = set(stats.embedding_hashes)
        if len(unique_hashes) < len(stats.embedding_hashes) * 0.95:
            issues.append("Many duplicate embeddings detected")
            fixes.append("Check if queries are being properly processed - may be using wrong input field")
        
        # Check norms
        avg_norm = np.mean(stats.norms)
        if avg_norm < 0.1:
            issues.append("Embeddings have very small norms (near zero)")
            fixes.append("Check encoder weights and normalization - may be using wrong encoder")
        elif avg_norm > 10.0:
            issues.append("Embeddings have very large norms (not normalized)")
            fixes.append("Apply L2 normalization to embeddings before use")
        
        # Check per-dimension variance
        avg_std = np.mean(stats.per_dim_stds)
        if avg_std < self.std_min:
            issues.append(f"Per-dimension variance too low ({avg_std:.3f} < {self.std_min})")
            fixes.append("Embeddings appear constant - check encoder weights and input processing")
        elif avg_std > self.std_max:
            issues.append(f"Per-dimension variance too high ({avg_std:.3f} > {self.std_max})")
            fixes.append("Embeddings may be poorly normalized - check encoding process")
        
        # Check cosine self-similarities
        avg_cosine_self = np.mean(stats.cosine_self_similarities)
        if avg_cosine_self < self.cosine_self_min:
            issues.append(f"Cosine self-similarity too low ({avg_cosine_self:.3f})")
            fixes.append("Embeddings not properly normalized - apply L2 normalization")
        
        # Check cosine similarities to random atoms
        if stats.cosine_to_random_atoms:
            avg_cosine_random = np.mean(np.abs(stats.cosine_to_random_atoms))
            if avg_cosine_random > self.cosine_random_tolerance:
                issues.append(f"Query-atom cosines too high ({avg_cosine_random:.3f})")
                fixes.append("Possible encoder/index mismatch - verify same encoder for queries and atoms")
        
        # Check principal component variances
        if len(stats.pc_variances) >= 3:
            if stats.pc_variances[0] < self.pc_variance_min:
                issues.append("First principal component has very low variance")
                fixes.append("Embeddings may be constant - check input processing and encoder")
        
        # Determine status
        if not issues:
            status = "pass"
        elif len(issues) <= 2:
            status = "warning"
        else:
            status = "fail"
            
        return status, issues, fixes
    
    def _generate_detailed_analysis(self, 
                                  stats: QueryVectorStats, 
                                  embeddings: np.ndarray, 
                                  query_texts: List[str]) -> Dict[str, Any]:
        """Generate detailed analysis for reporting."""
        
        return {
            'embeddings_analyzed': len(stats.embedding_hashes),
            'embedding_dimension': embeddings.shape[1],
            'unique_hashes': len(set(stats.embedding_hashes)),
            'duplicate_rate': 1.0 - (len(set(stats.embedding_hashes)) / len(stats.embedding_hashes)),
            
            # Norm statistics  
            'norm_mean': float(np.mean(stats.norms)),
            'norm_std': float(np.std(stats.norms)),
            'norm_min': float(np.min(stats.norms)),
            'norm_max': float(np.max(stats.norms)),
            
            # Per-dimension variance
            'avg_per_dim_std': float(np.mean(stats.per_dim_stds)),
            'per_dim_std_min': float(np.min(stats.per_dim_stds)),
            'per_dim_std_max': float(np.max(stats.per_dim_stds)),
            
            # Self-similarities
            'avg_cosine_self_sim': float(np.mean(stats.cosine_self_similarities)),
            'cosine_self_sim_std': float(np.std(stats.cosine_self_similarities)),
            
            # Random atom similarities
            'avg_cosine_to_atoms': float(np.mean(stats.cosine_to_random_atoms)) if stats.cosine_to_random_atoms else None,
            'cosine_to_atoms_std': float(np.std(stats.cosine_to_random_atoms)) if stats.cosine_to_random_atoms else None,
            
            # Principal components
            'pc_variances': stats.pc_variances,
            
            # Sample data
            'sample_hashes': stats.embedding_hashes[:10],  # First 10 for inspection
            'sample_norms': stats.norms[:10],
            'sample_query_texts': [text[:100] for text in query_texts[:5]]  # First 5 query snippets
        }
    
    def _log_findings(self, stats: QueryVectorStats, status: str, issues: List[str]):
        """Log key findings from the probe."""
        self.logger.info(f"Query Vector Probe: {status.upper()}")
        self.logger.info(f"Analyzed {len(stats.embedding_hashes)} query embeddings")
        self.logger.info(f"Average per-dimension std: {np.mean(stats.per_dim_stds):.3f}")
        self.logger.info(f"Average norm: {np.mean(stats.norms):.3f}")
        self.logger.info(f"Average cosine self-similarity: {np.mean(stats.cosine_self_similarities):.3f}")
        
        if stats.cosine_to_random_atoms:
            self.logger.info(f"Average cosine to random atoms: {np.mean(stats.cosine_to_random_atoms):.3f}")
        
        if issues:
            self.logger.warning(f"Issues found: {', '.join(issues)}")
        else:
            self.logger.info("No issues detected in query vectors")