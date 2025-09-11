#!/usr/bin/env python3
"""
Extract Evaluation Data for Diagnostic Analysis
==============================================

This script extracts sample-level evaluation data from InfiniteBench evaluations
and converts it to the diagnostic ledger format for running the 5-rung diagnostic ladder.

Usage:
    python scripts/extract_evaluation_data.py --method hybrid --keep_ratio 0.08 --dataset code --max_samples 50
    python scripts/extract_evaluation_data.py --from_results artifacts/hybrid_evaluation/results.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
import random

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import with more flexible error handling
try:
    from infinitebench.dataset_loader import InfiniteBenchLoader
    from infinitebench.hybrid_evaluation import EvaluationSample, HybridMethodEvaluator
except ImportError as e:
    logger.warning(f"Failed to import infinitebench modules: {e}")
    # Define minimal classes for testing
    from dataclasses import dataclass
    @dataclass
    class EvaluationSample:
        sample_id: str
        dataset: str
        query: str
        context: str
        ground_truth: Dict[str, Any]
        metadata: Dict[str, Any] = None
        
    class InfiniteBenchLoader:
        def __init__(self, data_dir):
            self.data_dir = data_dir
            
    class HybridMethodEvaluator:
        def __init__(self):
            pass

from diagnostics.sample_ledger import SampleLedger, SampleLedgerEntry

try:
    from context_competitors.lethe_streaming_hybrid import HybridSelector, HybridResult
except ImportError:
    logger.warning("HybridSelector not available, using mock implementation")
    class HybridSelector:
        def __init__(self, config):
            self.head_keep_ratio = 0.12
        def select_context(self, query, atoms, lambda_diversity):
            # Return first N atoms as mock selection
            target_count = min(len(atoms), int(len(atoms) * 0.1))
            return type('MockResult', (), {'selected_atoms': atoms[:target_count]})()
    class HybridResult:
        pass

logger = logging.getLogger(__name__)

@dataclass
class ExtractionConfig:
    """Configuration for data extraction."""
    method: str = "hybrid"  # hybrid, lethe, streaming
    keep_ratio: float = 0.08
    dataset: str = "code"
    max_samples: int = 50
    random_seed: int = 42
    output_dir: Path = Path("diagnostic_extraction_data")
    save_detailed_atoms: bool = True
    save_predictions: bool = True

class EvaluationDataExtractor:
    """Extract sample-level data from evaluation runs for diagnostic analysis."""
    
    def __init__(self, config: ExtractionConfig):
        """Initialize extractor with configuration."""
        self.config = config
        self.data_dir = Path("benchmarks/infinitebench/data")
        
        # Initialize components
        self.loader = InfiniteBenchLoader(self.data_dir)
        self.evaluator = HybridMethodEvaluator()
        
        # Initialize method-specific components
        self._initialize_method_components()
        
        # Setup output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _initialize_method_components(self):
        """Initialize components for the specified method."""
        if self.config.method in ["hybrid", "lethe"]:
            # Initialize hybrid selector
            hybrid_config = {
                'head_keep': 0.12,
                'window_size': 6000,
                'stride': 3000,
                'sinks': 96,
                'K2': 320,
                'dpp_rank': 14
            }
            self.hybrid_selector = HybridSelector(hybrid_config)
            
    def extract_from_live_evaluation(self) -> List[SampleLedgerEntry]:
        """Extract data from a live evaluation run."""
        logger.info(f"Starting live evaluation data extraction")
        logger.info(f"Method: {self.config.method}, Keep ratio: {self.config.keep_ratio}")
        logger.info(f"Dataset: {self.config.dataset}, Max samples: {self.config.max_samples}")
        
        # Load samples
        samples = self._load_samples()
        if not samples:
            logger.error("No samples loaded")
            return []
        
        logger.info(f"Loaded {len(samples)} samples for extraction")
        
        # Set random seed
        random.seed(self.config.random_seed)
        
        # Limit samples if specified
        if self.config.max_samples and len(samples) > self.config.max_samples:
            samples = random.sample(samples, self.config.max_samples)
            logger.info(f"Randomly selected {len(samples)} samples for analysis")
        
        # Extract data from each sample
        ledger_entries = []
        for i, sample in enumerate(samples):
            logger.info(f"Processing sample {i+1}/{len(samples)}: {sample.sample_id}")
            
            try:
                entry = self._extract_sample_data(sample)
                if entry:
                    ledger_entries.append(entry)
                    logger.debug(f"Successfully extracted data for {sample.sample_id}")
                else:
                    logger.warning(f"Failed to extract data for {sample.sample_id}")
                    
            except Exception as e:
                logger.error(f"Error processing sample {sample.sample_id}: {e}")
                continue
        
        logger.info(f"Successfully extracted data for {len(ledger_entries)} samples")
        return ledger_entries
    
    def _load_samples(self) -> List[EvaluationSample]:
        """Load evaluation samples from dataset."""
        # Map dataset names
        dataset_mapping = {
            'code': ['code_debug', 'code_qa'],
            'zh_qa': ['zh_qa']
        }
        
        if self.config.dataset in dataset_mapping:
            dataset_names = dataset_mapping[self.config.dataset]
        else:
            dataset_names = [self.config.dataset]
        
        all_samples = []
        for dataset_name in dataset_names:
            file_path = self.data_dir / f"{dataset_name}.jsonl"
            if not file_path.exists():
                logger.warning(f"Dataset file not found: {file_path}")
                continue
                
            try:
                with open(file_path, 'r') as f:
                    for line_idx, line in enumerate(f):
                        try:
                            data = json.loads(line.strip())
                            
                            sample = EvaluationSample(
                                sample_id=f"{dataset_name}_{line_idx}",
                                dataset=dataset_name,
                                query=data.get('input', ''),
                                context=data.get('context', ''),
                                ground_truth={'answers': data.get('answers', [])},
                                metadata={'original_data': data}
                            )
                            all_samples.append(sample)
                            
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse line {line_idx} in {file_path}: {e}")
                            continue
                            
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_name}: {e}")
                continue
        
        return all_samples
    
    def _extract_sample_data(self, sample: EvaluationSample) -> Optional[SampleLedgerEntry]:
        """Extract diagnostic data from a single sample."""
        start_time = time.time()
        
        try:
            # Get gold answers
            gold_answers = sample.ground_truth.get('answers', [])
            if isinstance(gold_answers, str):
                gold_answers = [gold_answers]
            elif not isinstance(gold_answers, list):
                gold_answers = [str(gold_answers)]
            
            # Process with hybrid method to get selected atoms
            selected_atoms = []
            predicted_answer = ""
            
            if self.config.method in ["hybrid", "lethe"]:
                # Use hybrid selector to get selected atoms
                atoms = self._segment_context_into_atoms(sample.context)
                
                if atoms:
                    # Adjust parameters for target keep ratio
                    original_head_keep = self.hybrid_selector.head_keep_ratio
                    self.hybrid_selector.head_keep_ratio = self.config.keep_ratio * 0.6
                    
                    try:
                        # Calculate lambda to achieve target compression
                        target_tokens = int(len(atoms) * self.config.keep_ratio)
                        optimal_lambda = self._find_optimal_lambda(atoms, target_tokens)
                        
                        # Select atoms with hybrid method
                        result = self.hybrid_selector.select_context(
                            query=sample.query,
                            atoms=atoms,
                            lambda_diversity=optimal_lambda
                        )
                        
                        if result and hasattr(result, 'selected_atoms'):
                            selected_atoms = result.selected_atoms[:target_tokens]
                        else:
                            # Fallback to first N atoms
                            selected_atoms = atoms[:target_tokens]
                            
                    finally:
                        # Restore original head_keep
                        self.hybrid_selector.head_keep_ratio = original_head_keep
                
                # For now, use simple extraction as predicted answer
                predicted_answer = self._extract_simple_answer(selected_atoms, sample.query)
            
            else:  # streaming method
                # For streaming, just use the full context with windowing
                atoms = self._segment_context_into_atoms(sample.context)
                window_size = int(len(atoms) * self.config.keep_ratio)
                selected_atoms = atoms[:window_size]  # Simple head selection
                predicted_answer = self._extract_simple_answer(selected_atoms, sample.query)
            
            # Analyze atom coverage
            spans_present, symbols_present = self._analyze_atom_coverage(
                selected_atoms, gold_answers
            )
            
            # Create coverage flags
            coverage_flags = self._calculate_coverage_metrics(
                spans_present, symbols_present, k=5
            )
            
            # Create ledger entry
            entry = SampleLedgerEntry(
                dataset=sample.dataset,
                sample_id=sample.sample_id,
                keep_ratio=self.config.keep_ratio,
                k=5,  # Default k value
                seed=self.config.random_seed,
                gold_answers=gold_answers,
                selected_atoms=selected_atoms,
                spans_present=spans_present,
                symbols_present=symbols_present,
                extractive_pred=predicted_answer,
                extractive_score=self._calculate_extractive_score(predicted_answer, gold_answers),
                coverage_flags=coverage_flags,
                cert_hash=self._calculate_cert_hash(sample.sample_id, selected_atoms, gold_answers),
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                processing_time_ms=(time.time() - start_time) * 1000,
                errors=[]
            )
            
            return entry
            
        except Exception as e:
            logger.error(f"Error extracting sample data for {sample.sample_id}: {e}")
            return None
    
    def _segment_context_into_atoms(self, context: str) -> List[str]:
        """Segment context into atoms (sentences/paragraphs)."""
        if not context:
            return []
        
        # Simple sentence segmentation for now
        sentences = []
        for paragraph in context.split('\n\n'):
            if paragraph.strip():
                # Split by sentence endings but keep reasonable length
                para_sentences = paragraph.replace('. ', '.|').split('|')
                sentences.extend([s.strip() for s in para_sentences if s.strip()])
        
        # If no sentence splitting worked, split by lines
        if not sentences:
            sentences = [line.strip() for line in context.split('\n') if line.strip()]
        
        return sentences if sentences else [context]
    
    def _find_optimal_lambda(self, atoms: List[str], target_tokens: int) -> float:
        """Find lambda value to achieve target token count."""
        # Simple binary search for lambda
        low, high = 0.0, 2.0
        best_lambda = 1.0
        
        for _ in range(10):  # Binary search iterations
            mid = (low + high) / 2
            
            # Mock evaluation of atom selection with this lambda
            # In real implementation, this would use the actual hybrid selector
            estimated_tokens = int(len(atoms) * (1.0 - mid * 0.3))  # Mock formula
            
            if abs(estimated_tokens - target_tokens) < 10:  # Close enough
                best_lambda = mid
                break
            elif estimated_tokens > target_tokens:
                low = mid
            else:
                high = mid
        
        return best_lambda
    
    def _analyze_atom_coverage(self, atoms: List[str], gold_answers: List[str]) -> Tuple[List[bool], List[bool]]:
        """Analyze which atoms contain gold answer spans/symbols."""
        spans_present = []
        symbols_present = []
        
        for atom in atoms:
            atom_lower = atom.lower()
            
            # Check for span presence (substring match)
            span_found = any(
                gold.lower() in atom_lower 
                for gold in gold_answers 
                if len(gold) > 3  # Avoid matching tiny strings
            )
            spans_present.append(span_found)
            
            # Check for symbol presence (word-level match)
            atom_words = set(atom_lower.split())
            symbol_found = any(
                any(word in atom_words for word in gold.lower().split())
                for gold in gold_answers
            )
            symbols_present.append(symbol_found)
        
        return spans_present, symbols_present
    
    def _calculate_coverage_metrics(self, spans_present: List[bool], symbols_present: List[bool], k: int) -> Dict[str, Any]:
        """Calculate coverage metrics for k atoms."""
        total_atoms = len(spans_present)
        k_atoms = min(k, total_atoms)
        
        if k_atoms == 0:
            return {
                "SpanCoverage@K": 0.0,
                "SymbolCoverage@K": 0.0,
                "SpanDensity": 0.0,
                "SymbolDensity": 0.0
            }
        
        # Coverage@K: at least one positive in first K
        span_coverage_k = any(spans_present[:k_atoms])
        symbol_coverage_k = any(symbols_present[:k_atoms])
        
        # Density: fraction of positives in first K
        span_density = sum(spans_present[:k_atoms]) / k_atoms
        symbol_density = sum(symbols_present[:k_atoms]) / k_atoms
        
        return {
            "SpanCoverage@K": float(span_coverage_k),
            "SymbolCoverage@K": float(symbol_coverage_k),
            "SpanDensity": span_density,
            "SymbolDensity": symbol_density,
            "K": k_atoms,
            "TotalAtoms": total_atoms
        }
    
    def _extract_simple_answer(self, atoms: List[str], query: str) -> str:
        """Extract simple answer from selected atoms."""
        if not atoms:
            return ""
        
        # Simple heuristic: concatenate first few atoms and extract key phrases
        combined_text = " ".join(atoms[:3])
        
        # Look for common answer patterns
        answer_patterns = [
            r"answer is (.+?)(?:\.|$)",
            r"result is (.+?)(?:\.|$)",
            r"solution is (.+?)(?:\.|$)",
            r"equals? (.+?)(?:\.|$)",
        ]
        
        import re
        for pattern in answer_patterns:
            match = re.search(pattern, combined_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: return first non-trivial sentence
        sentences = combined_text.split('. ')
        for sentence in sentences:
            if len(sentence.split()) > 3:  # At least 4 words
                return sentence.strip()
        
        return combined_text[:100].strip()  # First 100 chars as fallback
    
    def _calculate_extractive_score(self, predicted: str, gold_answers: List[str]) -> float:
        """Calculate extractive matching score."""
        if not predicted or not gold_answers:
            return 0.0
        
        predicted_lower = predicted.lower().strip()
        
        # Check for exact matches
        for gold in gold_answers:
            if predicted_lower == gold.lower().strip():
                return 1.0
        
        # Check for substring matches
        best_score = 0.0
        for gold in gold_answers:
            gold_lower = gold.lower().strip()
            if gold_lower in predicted_lower or predicted_lower in gold_lower:
                # Calculate overlap ratio
                intersection = len(set(gold_lower.split()) & set(predicted_lower.split()))
                union = len(set(gold_lower.split()) | set(predicted_lower.split()))
                if union > 0:
                    score = intersection / union
                    best_score = max(best_score, score)
        
        return best_score
    
    def _calculate_cert_hash(self, sample_id: str, atoms: List[str], gold_answers: List[str]) -> str:
        """Calculate certification hash for data integrity."""
        data_str = f"{sample_id}|{len(atoms)}|{len(gold_answers)}|{hash(tuple(atoms[:5]))}"
        return hashlib.md5(data_str.encode()).hexdigest()[:16]
    
    def save_ledger_entries(self, entries: List[SampleLedgerEntry], ledger_db_path: Optional[Path] = None):
        """Save ledger entries to database."""
        if not entries:
            logger.warning("No entries to save")
            return
        
        if not ledger_db_path:
            ledger_db_path = self.config.output_dir / "extracted_data_ledger.db"
        
        # Initialize ledger
        ledger = SampleLedger(ledger_db_path)
        
        # Save entries
        saved_count = 0
        for entry in entries:
            try:
                success = ledger.add_entry(entry)
                if success:
                    saved_count += 1
                else:
                    logger.warning(f"Failed to save entry for {entry.sample_id}")
            except Exception as e:
                logger.error(f"Error saving entry for {entry.sample_id}: {e}")
        
        logger.info(f"Saved {saved_count}/{len(entries)} entries to ledger: {ledger_db_path}")
        
        # Also save as JSON for debugging
        json_path = self.config.output_dir / "extracted_data.json"
        try:
            with open(json_path, 'w') as f:
                json.dump([entry.__dict__ for entry in entries], f, indent=2, default=str)
            logger.info(f"Also saved entries as JSON: {json_path}")
        except Exception as e:
            logger.warning(f"Failed to save JSON backup: {e}")

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )

def main():
    parser = argparse.ArgumentParser(description="Extract evaluation data for diagnostic analysis")
    parser.add_argument("--method", choices=["hybrid", "lethe", "streaming"], default="hybrid",
                       help="Evaluation method to use")
    parser.add_argument("--keep_ratio", type=float, default=0.08,
                       help="Context keep ratio for evaluation")
    parser.add_argument("--dataset", choices=["code", "zh_qa"], default="code",
                       help="Dataset to evaluate on")
    parser.add_argument("--max_samples", type=int, default=50,
                       help="Maximum number of samples to process")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for sample selection")
    parser.add_argument("--output_dir", type=Path, default="diagnostic_extraction_data",
                       help="Output directory for extracted data")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Create extraction config
    config = ExtractionConfig(
        method=args.method,
        keep_ratio=args.keep_ratio,
        dataset=args.dataset,
        max_samples=args.max_samples,
        random_seed=args.random_seed,
        output_dir=args.output_dir
    )
    
    # Run extraction
    logger.info("Starting evaluation data extraction")
    extractor = EvaluationDataExtractor(config)
    
    try:
        entries = extractor.extract_from_live_evaluation()
        
        if entries:
            extractor.save_ledger_entries(entries)
            logger.info(f"✅ Extraction completed successfully")
            logger.info(f"📊 Extracted data for {len(entries)} samples")
            logger.info(f"💾 Results saved to: {config.output_dir}")
        else:
            logger.error("❌ No data extracted")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())