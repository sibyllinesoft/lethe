#!/usr/bin/env python3
"""
Simple Evaluation Data Extractor for Diagnostic Analysis
========================================================

This script directly loads InfiniteBench data and creates sample entries
for the diagnostic ladder system without complex evaluation dependencies.

Usage:
    python scripts/extract_simple_evaluation_data.py --dataset code --max_samples 10
"""

import argparse
import json
import logging
import sys
import time
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logger = logging.getLogger(__name__)

@dataclass
class SampleLedgerEntry:
    """Complete diagnostic entry for a single sample."""
    
    # Primary key fields
    dataset: str
    sample_id: str  
    keep_ratio: float
    k: int
    seed: int
    
    # Ground truth data
    gold_answers: List[str]
    
    # Selection results
    selected_atoms: List[str]
    spans_present: List[bool]  # Does atom contain gold span?
    symbols_present: List[bool]  # Does atom contain gold symbol?
    
    # Extractive predictions
    extractive_pred: str
    extractive_score: float
    
    # Coverage flags
    coverage_flags: Dict[str, Any]  # SpanCoverage@K, SymbolCoverage@K, etc.
    
    # Data integrity
    cert_hash: str  # Hash of all fields for validation
    
    # Metadata
    timestamp: str
    processing_time_ms: float
    errors: List[str] = field(default_factory=list)

class SimpleDataExtractor:
    """Simple data extractor that works directly with InfiniteBench JSON files."""
    
    def __init__(self, data_dir: Path = Path("benchmarks/infinitebench/data")):
        """Initialize extractor with data directory."""
        self.data_dir = data_dir
        
    def extract_samples(self, dataset: str, max_samples: int = 50, 
                       keep_ratio: float = 0.08, random_seed: int = 42) -> List[SampleLedgerEntry]:
        """Extract samples from InfiniteBench data and create diagnostic entries."""
        logger.info(f"Extracting samples from {dataset} dataset")
        
        # Load raw data
        raw_samples = self._load_raw_samples(dataset)
        if not raw_samples:
            logger.error(f"No samples loaded from {dataset}")
            return []
        
        logger.info(f"Loaded {len(raw_samples)} raw samples")
        
        # Limit samples
        random.seed(random_seed)
        if max_samples and len(raw_samples) > max_samples:
            raw_samples = random.sample(raw_samples, max_samples)
            logger.info(f"Selected {len(raw_samples)} samples for analysis")
        
        # Process each sample
        ledger_entries = []
        for i, raw_sample in enumerate(raw_samples):
            logger.info(f"Processing sample {i+1}/{len(raw_samples)}")
            
            try:
                entry = self._create_ledger_entry(raw_sample, dataset, keep_ratio, random_seed)
                if entry:
                    ledger_entries.append(entry)
                    
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue
        
        logger.info(f"Successfully created {len(ledger_entries)} ledger entries")
        return ledger_entries
    
    def _load_raw_samples(self, dataset: str) -> List[Dict[str, Any]]:
        """Load raw samples from InfiniteBench JSON files."""
        # Map dataset names to files
        dataset_files = {
            'code': ['code_debug.jsonl', 'code_run.jsonl'],
            'zh_qa': ['longbook_qa_chn.jsonl']
        }
        
        if dataset not in dataset_files:
            logger.error(f"Unknown dataset: {dataset}")
            return []
        
        all_samples = []
        for filename in dataset_files[dataset]:
            filepath = self.data_dir / filename
            
            if not filepath.exists():
                logger.warning(f"File not found: {filepath}")
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line_idx, line in enumerate(f):
                        line = line.strip()
                        if line:
                            try:
                                sample = json.loads(line)
                                sample['_file'] = filename
                                sample['_line_idx'] = line_idx
                                all_samples.append(sample)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Invalid JSON at line {line_idx} in {filename}: {e}")
                                
            except Exception as e:
                logger.error(f"Error reading {filepath}: {e}")
                continue
        
        return all_samples
    
    def _create_ledger_entry(self, raw_sample: Dict[str, Any], dataset: str, 
                           keep_ratio: float, seed: int) -> Optional[SampleLedgerEntry]:
        """Create a ledger entry from raw sample data."""
        start_time = time.time()
        
        try:
            # Extract basic fields
            sample_id = f"{dataset}_{raw_sample.get('_file', 'unknown')}_{raw_sample.get('_line_idx', 0)}"
            
            # Get query and context
            query = raw_sample.get('input', '')
            context = raw_sample.get('context', '')
            
            if not query or not context:
                logger.warning(f"Sample {sample_id} missing query or context")
                return None
            
            # Get gold answers (check both 'answer' and 'answers' fields)
            gold_answers = raw_sample.get('answer', raw_sample.get('answers', []))
            if isinstance(gold_answers, str):
                gold_answers = [gold_answers]
            elif not isinstance(gold_answers, list):
                # Handle non-list types (int, float, etc.) by converting to string
                gold_answers = [str(gold_answers)]
            
            if not gold_answers:
                logger.warning(f"Sample {sample_id} has no gold answers")
                return None
            
            # Segment context into atoms
            atoms = self._segment_context(context)
            if not atoms:
                logger.warning(f"Sample {sample_id} produced no atoms")
                return None
            
            # Apply simple selection (mock hybrid method)
            selected_atoms = self._simple_atom_selection(atoms, query, keep_ratio)
            
            # Analyze coverage
            spans_present, symbols_present = self._analyze_coverage(selected_atoms, gold_answers)
            
            # Create simple extractive prediction
            extractive_pred = self._simple_extraction(selected_atoms, query)
            extractive_score = self._calculate_extractive_score(extractive_pred, gold_answers)
            
            # Calculate coverage metrics
            coverage_flags = self._calculate_coverage_metrics(spans_present, symbols_present, k=5)
            
            # Create certification hash
            cert_hash = self._calculate_cert_hash(sample_id, selected_atoms, gold_answers)
            
            # Create ledger entry
            entry = SampleLedgerEntry(
                dataset=dataset,
                sample_id=sample_id,
                keep_ratio=keep_ratio,
                k=5,
                seed=seed,
                gold_answers=gold_answers,
                selected_atoms=selected_atoms,
                spans_present=spans_present,
                symbols_present=symbols_present,
                extractive_pred=extractive_pred,
                extractive_score=extractive_score,
                coverage_flags=coverage_flags,
                cert_hash=cert_hash,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                processing_time_ms=(time.time() - start_time) * 1000,
                errors=[]
            )
            
            return entry
            
        except Exception as e:
            logger.error(f"Error creating ledger entry: {e}")
            return None
    
    def _segment_context(self, context: str) -> List[str]:
        """Segment context into atoms (sentences/paragraphs)."""
        if not context:
            return []
        
        # Try paragraph segmentation first
        paragraphs = [p.strip() for p in context.split('\n\n') if p.strip()]
        
        if len(paragraphs) > 5:  # If we have good paragraph segmentation
            return paragraphs
        
        # Fall back to sentence segmentation
        import re
        sentences = re.split(r'[.!?]+', context)
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]
        
        if sentences:
            return sentences
        
        # Final fallback: line segmentation
        lines = [line.strip() for line in context.split('\n') if line.strip()]
        return lines if lines else [context]
    
    def _simple_atom_selection(self, atoms: List[str], query: str, keep_ratio: float) -> List[str]:
        """Simple atom selection based on keep ratio."""
        target_count = max(1, int(len(atoms) * keep_ratio))
        
        # Simple heuristic: prefer atoms that contain query words
        query_words = set(query.lower().split())
        
        # Score atoms by query word overlap
        scored_atoms = []
        for i, atom in enumerate(atoms):
            atom_words = set(atom.lower().split())
            score = len(query_words & atom_words)
            scored_atoms.append((score, i, atom))
        
        # Sort by score (descending), then by position (ascending)
        scored_atoms.sort(key=lambda x: (-x[0], x[1]))
        
        # Select top atoms
        selected = [atom for _, _, atom in scored_atoms[:target_count]]
        
        return selected
    
    def _analyze_coverage(self, atoms: List[str], gold_answers: List[str]) -> Tuple[List[bool], List[bool]]:
        """Analyze which atoms contain gold answer information."""
        spans_present = []
        symbols_present = []
        
        for atom in atoms:
            atom_lower = atom.lower()
            
            # Check for span presence (substring match)
            span_found = False
            for gold in gold_answers:
                if len(gold) > 3 and gold.lower() in atom_lower:
                    span_found = True
                    break
            spans_present.append(span_found)
            
            # Check for symbol presence (word overlap)
            atom_words = set(atom_lower.split())
            symbol_found = False
            for gold in gold_answers:
                gold_words = set(gold.lower().split())
                if gold_words & atom_words:  # Non-empty intersection
                    symbol_found = True
                    break
            symbols_present.append(symbol_found)
        
        return spans_present, symbols_present
    
    def _calculate_coverage_metrics(self, spans_present: List[bool], 
                                  symbols_present: List[bool], k: int) -> Dict[str, Any]:
        """Calculate coverage metrics."""
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
    
    def _simple_extraction(self, atoms: List[str], query: str) -> str:
        """Simple extractive answer prediction."""
        if not atoms:
            return ""
        
        # Combine first few atoms
        combined = " ".join(atoms[:3])
        
        # Look for common answer patterns
        import re
        patterns = [
            r"(?:answer|result|solution|output)(?:\s*is)?\s*:?\s*([^.!?\n]+)",
            r"(?:equals?|=)\s*([^.!?\n]+)",
            r"(?:the\s+)?(?:correct\s+)?(?:answer\s+)?(?:is\s+)?([A-Z](?:\.[A-Z])*|\d+(?:\.\d+)?|[a-zA-Z0-9_-]+)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, combined, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        # Fallback: extract first meaningful phrase
        sentences = combined.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if 5 <= len(sentence.split()) <= 15:  # Reasonable answer length
                return sentence
        
        return combined[:100].strip()
    
    def _calculate_extractive_score(self, predicted: str, gold_answers: List[str]) -> float:
        """Calculate extractive matching score."""
        if not predicted or not gold_answers:
            return 0.0
        
        predicted_lower = predicted.lower().strip()
        
        # Check exact matches
        for gold in gold_answers:
            if predicted_lower == gold.lower().strip():
                return 1.0
        
        # Check substring matches
        best_score = 0.0
        for gold in gold_answers:
            gold_lower = gold.lower().strip()
            
            # Bidirectional substring check
            if gold_lower in predicted_lower or predicted_lower in gold_lower:
                # Jaccard similarity
                pred_words = set(predicted_lower.split())
                gold_words = set(gold_lower.split())
                
                intersection = len(pred_words & gold_words)
                union = len(pred_words | gold_words)
                
                if union > 0:
                    score = intersection / union
                    best_score = max(best_score, score)
        
        return best_score
    
    def _calculate_cert_hash(self, sample_id: str, atoms: List[str], gold_answers: List[str]) -> str:
        """Calculate certification hash."""
        data_str = f"{sample_id}|{len(atoms)}|{len(gold_answers)}|{hash(tuple(atoms[:3]))}"
        return hashlib.md5(data_str.encode()).hexdigest()[:16]

def save_to_json(entries: List[SampleLedgerEntry], output_file: Path):
    """Save entries to JSON file."""
    try:
        with open(output_file, 'w') as f:
            json.dump([entry.__dict__ for entry in entries], f, indent=2, default=str)
        logger.info(f"Saved {len(entries)} entries to {output_file}")
    except Exception as e:
        logger.error(f"Error saving to JSON: {e}")

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def main():
    parser = argparse.ArgumentParser(description="Extract simple evaluation data for diagnostics")
    parser.add_argument("--dataset", choices=["code", "zh_qa"], default="code",
                       help="Dataset to extract from")
    parser.add_argument("--max_samples", type=int, default=50,
                       help="Maximum samples to process")
    parser.add_argument("--keep_ratio", type=float, default=0.08,
                       help="Context keep ratio for atom selection")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=Path, default="simple_extraction_data",
                       help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info("🚀 Starting simple data extraction...")
        
        extractor = SimpleDataExtractor()
        entries = extractor.extract_samples(
            dataset=args.dataset,
            max_samples=args.max_samples,
            keep_ratio=args.keep_ratio,
            random_seed=args.random_seed
        )
        
        if entries:
            # Save to JSON
            output_file = args.output_dir / f"extracted_samples_{args.dataset}.json"
            save_to_json(entries, output_file)
            
            logger.info("✅ Extraction completed successfully!")
            logger.info(f"📊 Extracted {len(entries)} samples")
            logger.info(f"💾 Saved to: {output_file}")
            
            # Print sample statistics
            span_coverage = sum(1 for e in entries if e.coverage_flags.get("SpanCoverage@K", 0) > 0)
            symbol_coverage = sum(1 for e in entries if e.coverage_flags.get("SymbolCoverage@K", 0) > 0)
            extractive_success = sum(1 for e in entries if e.extractive_score > 0.1)
            
            logger.info(f"📈 Coverage statistics:")
            logger.info(f"   SpanCoverage@5: {span_coverage}/{len(entries)} ({span_coverage/len(entries)*100:.1f}%)")
            logger.info(f"   SymbolCoverage@5: {symbol_coverage}/{len(entries)} ({symbol_coverage/len(entries)*100:.1f}%)")
            logger.info(f"   Extractive success (>0.1): {extractive_success}/{len(entries)} ({extractive_success/len(entries)*100:.1f}%)")
            
        else:
            logger.error("❌ No samples extracted")
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Extraction failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())