#!/usr/bin/env python3
"""
Simple Diagnostic Runner for Real InfiniteBench Data
==================================================

This script runs the complete 5-rung diagnostic ladder on real InfiniteBench
evaluation data to provide definitive diagnosis of the 0.000 accuracy issue.

Usage:
    python scripts/run_simple_diagnostic.py --data_file simple_extraction_data/extracted_samples_code.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logger = logging.getLogger(__name__)

class SimpleDiagnosticRunner:
    """Run complete diagnostic ladder on extracted JSON data."""
    
    def __init__(self, output_dir: Path = Path("simple_diagnostic_results")):
        """Initialize diagnostic runner."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_extracted_data(self, data_file: Path) -> List[Dict[str, Any]]:
        """Load extracted sample data from JSON file."""
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} samples from {data_file}")
            return data
        except Exception as e:
            logger.error(f"Failed to load data from {data_file}: {e}")
            return []
    
    def analyze_coverage_patterns(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze coverage patterns across all samples (Rung 1: Coverage Analysis)."""
        logger.info("🔍 RUNG 1: Coverage Analysis")
        
        total_samples = len(samples)
        span_coverage_count = 0
        symbol_coverage_count = 0
        extractive_success_count = 0
        
        coverage_details = []
        
        for sample in samples:
            coverage_flags = sample.get('coverage_flags', {})
            span_coverage = coverage_flags.get('SpanCoverage@K', 0.0) > 0
            symbol_coverage = coverage_flags.get('SymbolCoverage@K', 0.0) > 0
            extractive_score = sample.get('extractive_score', 0.0)
            
            if span_coverage:
                span_coverage_count += 1
            if symbol_coverage:
                symbol_coverage_count += 1
            if extractive_score > 0.1:
                extractive_success_count += 1
                
            coverage_details.append({
                'sample_id': sample.get('sample_id'),
                'span_coverage': span_coverage,
                'symbol_coverage': symbol_coverage,
                'extractive_score': extractive_score,
                'selected_atoms_count': len(sample.get('selected_atoms', [])),
                'gold_answers': sample.get('gold_answers', [])
            })
        
        coverage_analysis = {
            'total_samples': total_samples,
            'span_coverage_rate': span_coverage_count / total_samples if total_samples > 0 else 0.0,
            'symbol_coverage_rate': symbol_coverage_count / total_samples if total_samples > 0 else 0.0,
            'extractive_success_rate': extractive_success_count / total_samples if total_samples > 0 else 0.0,
            'coverage_details': coverage_details
        }
        
        logger.info(f"   Span Coverage: {span_coverage_count}/{total_samples} ({coverage_analysis['span_coverage_rate']*100:.1f}%)")
        logger.info(f"   Symbol Coverage: {symbol_coverage_count}/{total_samples} ({coverage_analysis['symbol_coverage_rate']*100:.1f}%)")
        logger.info(f"   Extractive Success: {extractive_success_count}/{total_samples} ({coverage_analysis['extractive_success_rate']*100:.1f}%)")
        
        return coverage_analysis
    
    def analyze_selection_quality(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze atom selection quality (Rung 2: Selection Analysis)."""
        logger.info("🔍 RUNG 2: Selection Analysis")
        
        selection_stats = {
            'avg_atoms_selected': 0,
            'atom_selection_distribution': {},
            'context_compression_ratios': [],
            'selection_patterns': []
        }
        
        total_atoms_selected = 0
        total_samples = len(samples)
        
        for sample in samples:
            selected_atoms = sample.get('selected_atoms', [])
            num_selected = len(selected_atoms)
            total_atoms_selected += num_selected
            
            # Estimate original context size (this is approximate)
            estimated_original_size = num_selected / sample.get('keep_ratio', 0.08)
            compression_ratio = num_selected / estimated_original_size if estimated_original_size > 0 else 0
            selection_stats['context_compression_ratios'].append(compression_ratio)
            
            selection_stats['selection_patterns'].append({
                'sample_id': sample.get('sample_id'),
                'atoms_selected': num_selected,
                'compression_ratio': compression_ratio,
                'first_atom_preview': selected_atoms[0][:100] if selected_atoms else "No atoms selected"
            })
        
        selection_stats['avg_atoms_selected'] = total_atoms_selected / total_samples if total_samples > 0 else 0
        
        logger.info(f"   Average atoms selected: {selection_stats['avg_atoms_selected']:.1f}")
        logger.info(f"   Average compression ratio: {sum(selection_stats['context_compression_ratios'])/len(selection_stats['context_compression_ratios']):.3f}")
        
        return selection_stats
    
    def analyze_extraction_failures(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze answer extraction failures (Rung 3: Extraction Analysis)."""
        logger.info("🔍 RUNG 3: Extraction Analysis")
        
        extraction_analysis = {
            'total_extractions': len(samples),
            'successful_extractions': 0,
            'failed_extractions': 0,
            'extraction_patterns': [],
            'common_failures': {}
        }
        
        for sample in samples:
            extractive_pred = sample.get('extractive_pred', '')
            extractive_score = sample.get('extractive_score', 0.0)
            gold_answers = sample.get('gold_answers', [])
            
            is_success = extractive_score > 0.1
            if is_success:
                extraction_analysis['successful_extractions'] += 1
            else:
                extraction_analysis['failed_extractions'] += 1
            
            extraction_analysis['extraction_patterns'].append({
                'sample_id': sample.get('sample_id'),
                'prediction': extractive_pred[:100] if extractive_pred else "Empty prediction",
                'score': extractive_score,
                'gold_answers': gold_answers,
                'success': is_success
            })
        
        logger.info(f"   Successful extractions: {extraction_analysis['successful_extractions']}/{len(samples)}")
        logger.info(f"   Failed extractions: {extraction_analysis['failed_extractions']}/{len(samples)}")
        
        return extraction_analysis
    
    def analyze_gold_answer_patterns(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze gold answer patterns and formats (Rung 4: Gold Standard Analysis)."""
        logger.info("🔍 RUNG 4: Gold Standard Analysis")
        
        gold_analysis = {
            'answer_types': {},
            'answer_lengths': [],
            'answer_patterns': []
        }
        
        for sample in samples:
            gold_answers = sample.get('gold_answers', [])
            
            for answer in gold_answers:
                answer_str = str(answer)
                answer_type = type(answer).__name__
                
                gold_analysis['answer_types'][answer_type] = gold_analysis['answer_types'].get(answer_type, 0) + 1
                gold_analysis['answer_lengths'].append(len(answer_str))
                
            gold_analysis['answer_patterns'].append({
                'sample_id': sample.get('sample_id'),
                'answers': gold_answers,
                'answer_count': len(gold_answers)
            })
        
        avg_length = sum(gold_analysis['answer_lengths']) / len(gold_analysis['answer_lengths']) if gold_analysis['answer_lengths'] else 0
        
        logger.info(f"   Answer types: {gold_analysis['answer_types']}")
        logger.info(f"   Average answer length: {avg_length:.1f}")
        
        return gold_analysis
    
    def generate_diagnosis(self, coverage_analysis: Dict[str, Any], 
                          selection_analysis: Dict[str, Any],
                          extraction_analysis: Dict[str, Any],
                          gold_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate definitive diagnosis (Rung 5: Root Cause Diagnosis)."""
        logger.info("🔍 RUNG 5: Root Cause Diagnosis")
        
        diagnosis = {
            'primary_issue': '',
            'confidence': 0.0,
            'evidence': [],
            'recommended_actions': [],
            'secondary_issues': []
        }
        
        # Analyze the evidence
        span_coverage_rate = coverage_analysis['span_coverage_rate']
        symbol_coverage_rate = coverage_analysis['symbol_coverage_rate']
        extractive_success_rate = coverage_analysis['extractive_success_rate']
        
        logger.info(f"   Span coverage rate: {span_coverage_rate*100:.1f}%")
        logger.info(f"   Symbol coverage rate: {symbol_coverage_rate*100:.1f}%")
        logger.info(f"   Extractive success rate: {extractive_success_rate*100:.1f}%")
        
        # Determine primary issue based on the failure pattern
        if span_coverage_rate < 0.1 and symbol_coverage_rate < 0.1:
            diagnosis['primary_issue'] = 'SELECTION_FAILURE'
            diagnosis['confidence'] = 0.95
            diagnosis['evidence'].append(f"Extremely low span coverage ({span_coverage_rate*100:.1f}%) and symbol coverage ({symbol_coverage_rate*100:.1f}%)")
            diagnosis['evidence'].append("This indicates the hybrid retrieval system is not selecting relevant context atoms")
            diagnosis['recommended_actions'].append("1. Fix context atom selection - increase keep_ratio or improve hybrid scoring")
            diagnosis['recommended_actions'].append("2. Debug the hybrid method's relevance scoring algorithm")
            diagnosis['recommended_actions'].append("3. Verify InfiniteBench context segmentation is working correctly")
            
        elif symbol_coverage_rate > 0.3 and extractive_success_rate < 0.1:
            diagnosis['primary_issue'] = 'EXTRACTION_FAILURE'
            diagnosis['confidence'] = 0.85
            diagnosis['evidence'].append(f"Decent symbol coverage ({symbol_coverage_rate*100:.1f}%) but poor extraction ({extractive_success_rate*100:.1f}%)")
            diagnosis['evidence'].append("This indicates relevant content is present but answer extraction is failing")
            diagnosis['recommended_actions'].append("1. Fix answer extraction patterns - improve regex and heuristics")
            diagnosis['recommended_actions'].append("2. Debug the extractive prediction algorithm")
            diagnosis['recommended_actions'].append("3. Verify gold answer format compatibility")
            
        elif extractive_success_rate > 0.3:
            diagnosis['primary_issue'] = 'SCORING_MISMATCH'
            diagnosis['confidence'] = 0.70
            diagnosis['evidence'].append(f"Reasonable extractive success ({extractive_success_rate*100:.1f}%) suggests extraction works")
            diagnosis['evidence'].append("The issue may be in how predictions are scored against gold answers")
            diagnosis['recommended_actions'].append("1. Debug the scoring function - check exact match vs similarity")
            diagnosis['recommended_actions'].append("2. Verify gold answer format matches prediction format")
            diagnosis['recommended_actions'].append("3. Consider fuzzy matching for numerical answers")
            
        else:
            diagnosis['primary_issue'] = 'SYSTEMATIC_FAILURE'
            diagnosis['confidence'] = 0.80
            diagnosis['evidence'].append("Multiple failure modes detected across the pipeline")
            diagnosis['evidence'].append("This suggests fundamental issues with the evaluation setup")
            diagnosis['recommended_actions'].append("1. InfiniteBench is legitimately challenging - verify this is expected")
            diagnosis['recommended_actions'].append("2. Check if the hybrid method parameters are appropriate for this data")
            diagnosis['recommended_actions'].append("3. Compare against known baseline performance")
        
        logger.info(f"   🎯 PRIMARY DIAGNOSIS: {diagnosis['primary_issue']} (confidence: {diagnosis['confidence']*100:.1f}%)")
        
        return diagnosis
    
    def run_complete_diagnostic(self, data_file: Path) -> Dict[str, Any]:
        """Run the complete 5-rung diagnostic ladder."""
        logger.info("🚀 Starting Complete Diagnostic Analysis")
        logger.info("="*60)
        
        start_time = time.time()
        
        # Load data
        samples = self.load_extracted_data(data_file)
        if not samples:
            logger.error("No samples loaded - cannot run diagnostics")
            return {}
        
        # Run 5-rung diagnostic ladder
        results = {
            'metadata': {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'data_file': str(data_file),
                'sample_count': len(samples),
                'analysis_duration_seconds': 0
            }
        }
        
        # Rung 1: Coverage Analysis
        results['coverage_analysis'] = self.analyze_coverage_patterns(samples)
        
        # Rung 2: Selection Analysis  
        results['selection_analysis'] = self.analyze_selection_quality(samples)
        
        # Rung 3: Extraction Analysis
        results['extraction_analysis'] = self.analyze_extraction_failures(samples)
        
        # Rung 4: Gold Standard Analysis
        results['gold_analysis'] = self.analyze_gold_answer_patterns(samples)
        
        # Rung 5: Root Cause Diagnosis
        results['diagnosis'] = self.generate_diagnosis(
            results['coverage_analysis'],
            results['selection_analysis'], 
            results['extraction_analysis'],
            results['gold_analysis']
        )
        
        results['metadata']['analysis_duration_seconds'] = time.time() - start_time
        
        # Save results
        results_file = self.output_dir / f"diagnostic_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("="*60)
        logger.info("🎯 DIAGNOSTIC COMPLETE")
        logger.info(f"📊 Analysis Duration: {results['metadata']['analysis_duration_seconds']:.2f} seconds")
        logger.info(f"💾 Results saved to: {results_file}")
        
        # Print summary
        self.print_diagnostic_summary(results)
        
        return results
    
    def print_diagnostic_summary(self, results: Dict[str, Any]):
        """Print a human-readable summary of the diagnostic results."""
        print("\n" + "="*60)
        print("🏥 DIAGNOSTIC SUMMARY")
        print("="*60)
        
        diagnosis = results.get('diagnosis', {})
        coverage = results.get('coverage_analysis', {})
        
        print(f"📊 SAMPLE ANALYSIS:")
        print(f"   • Total Samples: {results['metadata']['sample_count']}")
        print(f"   • Span Coverage: {coverage.get('span_coverage_rate', 0)*100:.1f}%")
        print(f"   • Symbol Coverage: {coverage.get('symbol_coverage_rate', 0)*100:.1f}%") 
        print(f"   • Extractive Success: {coverage.get('extractive_success_rate', 0)*100:.1f}%")
        
        print(f"\n🎯 PRIMARY DIAGNOSIS:")
        print(f"   • Issue: {diagnosis.get('primary_issue', 'UNKNOWN')}")
        print(f"   • Confidence: {diagnosis.get('confidence', 0)*100:.1f}%")
        
        print(f"\n📋 EVIDENCE:")
        for evidence in diagnosis.get('evidence', []):
            print(f"   • {evidence}")
            
        print(f"\n🔧 RECOMMENDED ACTIONS:")
        for i, action in enumerate(diagnosis.get('recommended_actions', []), 1):
            print(f"   {action}")
            
        print("="*60)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def main():
    parser = argparse.ArgumentParser(description="Run simple diagnostic ladder on extracted data")
    parser.add_argument("--data_file", type=Path, required=True,
                       help="Path to extracted JSON data file")
    parser.add_argument("--output_dir", type=Path, default="simple_diagnostic_results",
                       help="Output directory for results")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    try:
        runner = SimpleDiagnosticRunner(args.output_dir)
        results = runner.run_complete_diagnostic(args.data_file)
        
        if results:
            logger.info("✅ Diagnostic analysis completed successfully!")
            return 0
        else:
            logger.error("❌ Diagnostic analysis failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Diagnostic analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())