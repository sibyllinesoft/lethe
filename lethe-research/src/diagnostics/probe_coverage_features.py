"""
Probe 4: Coverage Features Validation  
=====================================

Validates that entity and symbol extraction is working for coverage-based
utility (CBU) selection. Tests selected atoms and checks:

- For selected atoms, log entities_count, symbols_count, file_ids
- Expected: medians >0 for code datasets (dozens typical)  
- If zeros, entity/symbol extraction wasn't run or feature flag is off

Ensures CBU has proper features for coverage optimization.
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import time

logger = logging.getLogger(__name__)

@dataclass
class CoverageFeatureStats:
    """Statistics for coverage feature analysis."""
    atoms_analyzed: int
    entities_counts: List[int]
    symbols_counts: List[int]  
    file_ids: List[str]
    unique_files: Set[str]
    atoms_with_entities: int
    atoms_with_symbols: int
    atoms_with_file_ids: int
    entity_types: Counter
    symbol_types: Counter

class CoverageFeaturesProbe:
    """
    Probe 4: Validates coverage features (entities, symbols, files) are extracted.
    
    Checks for common failure modes:
    - Entity extraction disabled or failing
    - Symbol extraction disabled or failing  
    - Missing file ID mapping
    - Feature extraction not run during indexing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sample_size = config.get('sample_size', 100)
        
        # Thresholds for pass/fail (for code datasets)
        self.min_entity_median = config.get('min_entity_median', 1)
        self.min_symbol_median = config.get('min_symbol_median', 1)
        self.min_coverage_ratio = config.get('min_coverage_ratio', 0.5)  # 50% atoms should have features
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
    async def diagnose_coverage_features(self, 
                                       evaluation_data: List[Dict[str, Any]], 
                                       retrieval_pipeline: Any) -> 'ProbeResult':
        """
        Run coverage features validation.
        
        Args:
            evaluation_data: List of evaluation samples
            retrieval_pipeline: Lethe retrieval pipeline instance
            
        Returns:
            ProbeResult with pass/fail status and diagnostics
        """
        from .selection_stack_diagnostics import ProbeResult
        
        start_time = time.time()
        
        try:
            # Get selected atoms from retrieval pipeline
            selected_atoms = await self._get_selected_atoms(evaluation_data, retrieval_pipeline)
            
            if not selected_atoms:
                return ProbeResult(
                    probe_name="Coverage Features Probe",
                    status="fail",
                    summary="No atoms found for coverage analysis",
                    details={"error": "No selected atoms available"},
                    fix_recommendations=["Check atom selection pipeline - no atoms found"],
                    execution_time_ms=(time.time() - start_time) * 1000
                )
            
            # Analyze coverage features
            stats = self._analyze_coverage_features(selected_atoms)
            
            # Determine pass/fail status
            status, issues, fixes = self._evaluate_coverage_stats(stats)
            
            # Generate detailed analysis
            details = self._generate_detailed_analysis(stats, selected_atoms)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Log key findings
            self._log_findings(stats, status, issues)
            
            return ProbeResult(
                probe_name="Coverage Features Probe",
                status=status,
                summary=f"Coverage features {status}: {len(issues)} issues found" if issues else f"Coverage features {status}",
                details=details,
                fix_recommendations=fixes,
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            self.logger.error(f"Coverage features probe failed: {e}")
            execution_time = (time.time() - start_time) * 1000
            
            return ProbeResult(
                probe_name="Coverage Features Probe",
                status="fail",
                summary=f"Probe failed with error: {str(e)}",
                details={"error": str(e)},
                fix_recommendations=[f"Fix coverage features probe: {str(e)}"],
                execution_time_ms=execution_time
            )
    
    async def _get_selected_atoms(self, 
                                evaluation_data: List[Dict[str, Any]], 
                                retrieval_pipeline: Any) -> List[Dict[str, Any]]:
        """Get selected atoms from retrieval pipeline for coverage analysis."""
        atoms = []
        
        # Sample some queries to get selected atoms
        sample_queries = evaluation_data[:min(20, len(evaluation_data))]
        
        for sample in sample_queries:
            query_text = self._extract_query_text(sample)
            if not query_text:
                continue
                
            try:
                # Get atoms through different possible methods
                query_atoms = await self._get_atoms_for_query(query_text, retrieval_pipeline)
                atoms.extend(query_atoms)
                
                # Limit total atoms analyzed
                if len(atoms) >= self.sample_size:
                    break
                    
            except Exception as e:
                self.logger.warning(f"Failed to get atoms for query: {e}")
                continue
        
        # If we still don't have enough, try to get from index directly
        if len(atoms) < self.sample_size // 2:
            try:
                index_atoms = await self._get_atoms_from_index(retrieval_pipeline)
                atoms.extend(index_atoms)
            except Exception as e:
                self.logger.warning(f"Failed to get atoms from index: {e}")
        
        return atoms[:self.sample_size]  # Limit to sample size
    
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
    
    async def _get_atoms_for_query(self, query_text: str, retrieval_pipeline: Any) -> List[Dict[str, Any]]:
        """Get selected atoms for a specific query."""
        atoms = []
        
        try:
            # Try different methods to get selected atoms
            if hasattr(retrieval_pipeline, 'select_atoms'):
                result = await retrieval_pipeline.select_atoms(query_text)
                atoms = self._normalize_atoms(result)
                
            elif hasattr(retrieval_pipeline, 'get_selected_atoms'):
                result = await retrieval_pipeline.get_selected_atoms(query_text)
                atoms = self._normalize_atoms(result)
                
            elif hasattr(retrieval_pipeline, 'retrieve'):
                # Get retrieval results and extract atoms
                results = await retrieval_pipeline.retrieve(query_text, k=50)
                atoms = self._extract_atoms_from_results(results)
                
            else:
                # Try to find selection method in components
                selection_method = self._find_selection_method(retrieval_pipeline)
                if selection_method:
                    result = await selection_method(query_text)
                    atoms = self._normalize_atoms(result)
                    
        except Exception as e:
            self.logger.warning(f"Atom selection failed: {e}")
            
        return atoms
    
    async def _get_atoms_from_index(self, retrieval_pipeline: Any) -> List[Dict[str, Any]]:
        """Get atoms directly from index for analysis."""
        atoms = []
        
        try:
            if hasattr(retrieval_pipeline, 'index'):
                index = retrieval_pipeline.index
                
                # Try different methods to sample atoms from index
                if hasattr(index, 'sample_atoms'):
                    atoms = await index.sample_atoms(self.sample_size)
                elif hasattr(index, 'get_all_atoms'):
                    all_atoms = await index.get_all_atoms()
                    if len(all_atoms) > self.sample_size:
                        indices = np.random.choice(len(all_atoms), self.sample_size, replace=False)
                        atoms = [all_atoms[i] for i in indices]
                    else:
                        atoms = all_atoms
                elif hasattr(index, 'atoms'):
                    all_atoms = index.atoms
                    if len(all_atoms) > self.sample_size:
                        atoms = np.random.choice(all_atoms, self.sample_size, replace=False).tolist()
                    else:
                        atoms = all_atoms
                        
        except Exception as e:
            self.logger.warning(f"Failed to get atoms from index: {e}")
            
        return self._normalize_atoms(atoms)
    
    def _find_selection_method(self, pipeline: Any) -> Optional[callable]:
        """Find atom selection method in pipeline."""
        selection_methods = ['select_atoms', 'get_selected_atoms', 'select', 'choose_atoms']
        
        for method_name in selection_methods:
            if hasattr(pipeline, method_name):
                return getattr(pipeline, method_name)
                
        # Try nested components
        if hasattr(pipeline, 'components'):
            for component in pipeline.components:
                for method_name in selection_methods:
                    if hasattr(component, method_name):
                        return getattr(component, method_name)
        
        return None
    
    def _normalize_atoms(self, atoms: Any) -> List[Dict[str, Any]]:
        """Normalize atoms to standard format."""
        if not atoms:
            return []
            
        normalized = []
        
        if isinstance(atoms, dict) and 'atoms' in atoms:
            atoms = atoms['atoms']
        
        if not isinstance(atoms, list):
            atoms = [atoms]
            
        for atom in atoms:
            if isinstance(atom, dict):
                normalized.append(atom)
            elif hasattr(atom, '__dict__'):
                # Convert object to dict
                normalized.append(vars(atom))
            else:
                # Create basic atom structure
                normalized.append({
                    'content': str(atom),
                    'entities': [],
                    'symbols': [],
                    'file_id': None
                })
                
        return normalized
    
    def _extract_atoms_from_results(self, results: Any) -> List[Dict[str, Any]]:
        """Extract atoms from retrieval results."""
        atoms = []
        
        if isinstance(results, dict):
            # Results might contain atoms or documents
            if 'atoms' in results:
                atoms = results['atoms']
            elif 'documents' in results:
                atoms = results['documents']  
            elif 'items' in results:
                atoms = results['items']
                
        elif isinstance(results, list):
            atoms = results
            
        return self._normalize_atoms(atoms)
    
    def _analyze_coverage_features(self, atoms: List[Dict[str, Any]]) -> CoverageFeatureStats:
        """Analyze coverage features in selected atoms."""
        
        entities_counts = []
        symbols_counts = []
        file_ids = []
        unique_files = set()
        atoms_with_entities = 0
        atoms_with_symbols = 0  
        atoms_with_file_ids = 0
        entity_types = Counter()
        symbol_types = Counter()
        
        for atom in atoms:
            # Extract entities
            entities = self._extract_entities(atom)
            entities_count = len(entities) if entities else 0
            entities_counts.append(entities_count)
            
            if entities_count > 0:
                atoms_with_entities += 1
                # Count entity types
                for entity in entities:
                    entity_type = self._get_entity_type(entity)
                    if entity_type:
                        entity_types[entity_type] += 1
            
            # Extract symbols  
            symbols = self._extract_symbols(atom)
            symbols_count = len(symbols) if symbols else 0
            symbols_counts.append(symbols_count)
            
            if symbols_count > 0:
                atoms_with_symbols += 1
                # Count symbol types
                for symbol in symbols:
                    symbol_type = self._get_symbol_type(symbol)
                    if symbol_type:
                        symbol_types[symbol_type] += 1
            
            # Extract file ID
            file_id = self._extract_file_id(atom)
            if file_id:
                file_ids.append(file_id)
                unique_files.add(file_id)
                atoms_with_file_ids += 1
            else:
                file_ids.append("")
        
        return CoverageFeatureStats(
            atoms_analyzed=len(atoms),
            entities_counts=entities_counts,
            symbols_counts=symbols_counts,
            file_ids=file_ids,
            unique_files=unique_files,
            atoms_with_entities=atoms_with_entities,
            atoms_with_symbols=atoms_with_symbols,
            atoms_with_file_ids=atoms_with_file_ids,
            entity_types=entity_types,
            symbol_types=symbol_types
        )
    
    def _extract_entities(self, atom: Dict[str, Any]) -> List[Any]:
        """Extract entities from atom."""
        # Try different possible entity fields
        entity_fields = ['entities', 'named_entities', 'ner', 'entities_list']
        
        for field in entity_fields:
            if field in atom and atom[field]:
                entities = atom[field]
                if isinstance(entities, list):
                    return entities
                elif isinstance(entities, dict):
                    # Flatten dict values
                    all_entities = []
                    for entities_list in entities.values():
                        if isinstance(entities_list, list):
                            all_entities.extend(entities_list)
                    return all_entities
                else:
                    return [entities]
        
        # Try to extract from content if entities not explicitly stored
        content = atom.get('content', atom.get('text', ''))
        if content:
            return self._extract_entities_from_content(content)
            
        return []
    
    def _extract_symbols(self, atom: Dict[str, Any]) -> List[Any]:
        """Extract code symbols from atom."""
        # Try different possible symbol fields
        symbol_fields = ['symbols', 'code_symbols', 'functions', 'classes', 'symbols_list']
        
        for field in symbol_fields:
            if field in atom and atom[field]:
                symbols = atom[field]
                if isinstance(symbols, list):
                    return symbols
                elif isinstance(symbols, dict):
                    # Flatten dict values
                    all_symbols = []
                    for symbols_list in symbols.values():
                        if isinstance(symbols_list, list):
                            all_symbols.extend(symbols_list)
                    return all_symbols
                else:
                    return [symbols]
        
        # Try to extract from content if symbols not explicitly stored
        content = atom.get('content', atom.get('text', ''))
        if content:
            return self._extract_symbols_from_content(content)
            
        return []
    
    def _extract_file_id(self, atom: Dict[str, Any]) -> Optional[str]:
        """Extract file ID from atom."""
        # Try different possible file ID fields
        file_fields = ['file_id', 'file_path', 'filename', 'source_file', 'document_id']
        
        for field in file_fields:
            if field in atom and atom[field]:
                return str(atom[field])
                
        return None
    
    def _extract_entities_from_content(self, content: str) -> List[str]:
        """Simple entity extraction from content using regex patterns."""
        import re
        
        entities = []
        
        # Simple patterns for common entity types
        patterns = {
            'URL': r'https?://[^\s]+',
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE': r'\b\d{3}-\d{3}-\d{4}\b|\(\d{3}\)\s*\d{3}-\d{4}',
            'DATE': r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
            'NUMBER': r'\b\d+\.\d+\b|\b\d+\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, content)
            entities.extend([f"{entity_type}:{match}" for match in matches])
            
        return entities[:50]  # Limit to avoid too many
    
    def _extract_symbols_from_content(self, content: str) -> List[str]:
        """Simple symbol extraction from code content."""
        import re
        
        symbols = []
        
        # Code symbol patterns
        patterns = {
            'FUNCTION': r'\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            'CLASS': r'\bclass\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            'VARIABLE': r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=',
            'IMPORT': r'\bfrom\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+import|\bimport\s+([a-zA-Z_][a-zA-Z0-9_.]*)'
        }
        
        for symbol_type, pattern in patterns.items():
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle multiple capture groups
                    for m in match:
                        if m:
                            symbols.append(f"{symbol_type}:{m}")
                else:
                    symbols.append(f"{symbol_type}:{match}")
                    
        return symbols[:50]  # Limit to avoid too many
    
    def _get_entity_type(self, entity: Any) -> Optional[str]:
        """Get entity type from entity object."""
        if isinstance(entity, str):
            # Entity stored as "TYPE:value"
            if ':' in entity:
                return entity.split(':', 1)[0]
            else:
                return 'ENTITY'
        elif isinstance(entity, dict):
            return entity.get('type', entity.get('label', 'ENTITY'))
        else:
            return 'ENTITY'
    
    def _get_symbol_type(self, symbol: Any) -> Optional[str]:
        """Get symbol type from symbol object."""
        if isinstance(symbol, str):
            # Symbol stored as "TYPE:value"
            if ':' in symbol:
                return symbol.split(':', 1)[0]
            else:
                return 'SYMBOL'
        elif isinstance(symbol, dict):
            return symbol.get('type', symbol.get('kind', 'SYMBOL'))
        else:
            return 'SYMBOL'
    
    def _evaluate_coverage_stats(self, stats: CoverageFeatureStats) -> Tuple[str, List[str], List[str]]:
        """Evaluate coverage statistics to determine pass/fail status."""
        issues = []
        fixes = []
        
        if stats.atoms_analyzed == 0:
            issues.append("No atoms available for analysis")
            fixes.append("Check atom selection and indexing pipeline")
            return "fail", issues, fixes
        
        # Check entity extraction
        entities_median = np.median(stats.entities_counts) if stats.entities_counts else 0
        if entities_median < self.min_entity_median:
            issues.append(f"Low entity median: {entities_median}")
            fixes.append("Enable entity extraction or check NER pipeline")
        
        entity_coverage_ratio = stats.atoms_with_entities / stats.atoms_analyzed
        if entity_coverage_ratio < self.min_coverage_ratio:
            issues.append(f"Low entity coverage: {entity_coverage_ratio:.1%}")
            fixes.append("Improve entity extraction coverage - many atoms have no entities")
        
        # Check symbol extraction
        symbols_median = np.median(stats.symbols_counts) if stats.symbols_counts else 0
        if symbols_median < self.min_symbol_median:
            issues.append(f"Low symbol median: {symbols_median}")
            fixes.append("Enable code symbol extraction or check symbol parser")
        
        symbol_coverage_ratio = stats.atoms_with_symbols / stats.atoms_analyzed
        if symbol_coverage_ratio < self.min_coverage_ratio:
            issues.append(f"Low symbol coverage: {symbol_coverage_ratio:.1%}")
            fixes.append("Improve symbol extraction coverage - many atoms have no symbols")
        
        # Check file ID mapping
        file_coverage_ratio = stats.atoms_with_file_ids / stats.atoms_analyzed
        if file_coverage_ratio < 0.8:  # 80% should have file IDs
            issues.append(f"Low file ID coverage: {file_coverage_ratio:.1%}")
            fixes.append("Ensure file ID mapping during indexing - needed for coverage tracking")
        
        # Check diversity
        if len(stats.unique_files) < 2 and stats.atoms_with_file_ids > 10:
            issues.append(f"Low file diversity: {len(stats.unique_files)} unique files")
            fixes.append("Atoms from too few files - check dataset diversity")
        
        # Determine status
        if not issues:
            status = "pass"
        elif (len(issues) <= 2 and 
              entities_median > 0 and 
              symbols_median > 0 and
              file_coverage_ratio > 0.5):
            status = "warning"
        else:
            status = "fail"
            
        return status, issues, fixes
    
    def _generate_detailed_analysis(self, 
                                  stats: CoverageFeatureStats,
                                  atoms: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate detailed analysis for reporting."""
        
        return {
            'atoms_analyzed': stats.atoms_analyzed,
            
            # Entity statistics
            'entities_median': float(np.median(stats.entities_counts)) if stats.entities_counts else 0.0,
            'entities_mean': float(np.mean(stats.entities_counts)) if stats.entities_counts else 0.0,
            'entities_std': float(np.std(stats.entities_counts)) if stats.entities_counts else 0.0,
            'entities_max': int(max(stats.entities_counts)) if stats.entities_counts else 0,
            'atoms_with_entities': stats.atoms_with_entities,
            'entity_coverage_ratio': stats.atoms_with_entities / stats.atoms_analyzed if stats.atoms_analyzed > 0 else 0.0,
            'top_entity_types': dict(stats.entity_types.most_common(10)),
            
            # Symbol statistics  
            'symbols_median': float(np.median(stats.symbols_counts)) if stats.symbols_counts else 0.0,
            'symbols_mean': float(np.mean(stats.symbols_counts)) if stats.symbols_counts else 0.0,
            'symbols_std': float(np.std(stats.symbols_counts)) if stats.symbols_counts else 0.0,
            'symbols_max': int(max(stats.symbols_counts)) if stats.symbols_counts else 0,
            'atoms_with_symbols': stats.atoms_with_symbols,
            'symbol_coverage_ratio': stats.atoms_with_symbols / stats.atoms_analyzed if stats.atoms_analyzed > 0 else 0.0,
            'top_symbol_types': dict(stats.symbol_types.most_common(10)),
            
            # File statistics
            'atoms_with_file_ids': stats.atoms_with_file_ids,
            'file_coverage_ratio': stats.atoms_with_file_ids / stats.atoms_analyzed if stats.atoms_analyzed > 0 else 0.0,
            'unique_files': len(stats.unique_files),
            'sample_file_ids': list(stats.unique_files)[:20],  # First 20 file IDs
            
            # Sample atoms for inspection
            'sample_atoms': [
                {
                    'entities_count': len(self._extract_entities(atom)),
                    'symbols_count': len(self._extract_symbols(atom)),
                    'file_id': self._extract_file_id(atom),
                    'content_length': len(str(atom.get('content', atom.get('text', ''))))
                }
                for atom in atoms[:10]  # First 10 atoms
            ]
        }
    
    def _log_findings(self, stats: CoverageFeatureStats, status: str, issues: List[str]):
        """Log key findings from the probe."""
        self.logger.info(f"Coverage Features Probe: {status.upper()}")
        self.logger.info(f"Analyzed {stats.atoms_analyzed} atoms")
        
        if stats.entities_counts:
            self.logger.info(f"Entities median: {np.median(stats.entities_counts)}")
            self.logger.info(f"Atoms with entities: {stats.atoms_with_entities} ({stats.atoms_with_entities/stats.atoms_analyzed*100:.1f}%)")
        
        if stats.symbols_counts:
            self.logger.info(f"Symbols median: {np.median(stats.symbols_counts)}")
            self.logger.info(f"Atoms with symbols: {stats.atoms_with_symbols} ({stats.atoms_with_symbols/stats.atoms_analyzed*100:.1f}%)")
        
        self.logger.info(f"Atoms with file IDs: {stats.atoms_with_file_ids} ({stats.atoms_with_file_ids/stats.atoms_analyzed*100:.1f}%)")
        self.logger.info(f"Unique files: {len(stats.unique_files)}")
        
        if issues:
            self.logger.warning(f"Issues found: {', '.join(issues)}")
        else:
            self.logger.info("No issues detected in coverage features")