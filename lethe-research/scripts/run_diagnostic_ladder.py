#!/usr/bin/env python3
"""
Diagnostic Ladder CLI Entry Point
===============================

Command-line interface for running the complete diagnostic ladder of proofs.
Provides systematic validation of InfiniteBench evaluation pipeline components.

Usage:
    python scripts/run_diagnostic_ladder.py --samples_file data.jsonl --output_dir results/
    python scripts/run_diagnostic_ladder.py --quick --max_samples 50
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diagnostics.ladder_runner import DiagnosticLadderRunner, DiagnosticConfig
from diagnostics.sample_ledger import SampleLedger

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('diagnostic_ladder.log')
        ]
    )

def load_samples_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load samples from JSONL file."""
    samples = []
    
    if not file_path.exists():
        raise FileNotFoundError(f"Samples file not found: {file_path}")
    
    try:
        if file_path.suffix == '.jsonl':
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            sample = json.loads(line)
                            samples.append(sample)
                        except json.JSONDecodeError as e:
                            logging.warning(f"Skipping invalid JSON at line {line_num}: {e}")
        
        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                else:
                    samples = [data]
        
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    except Exception as e:
        logging.error(f"Failed to load samples from {file_path}: {e}")
        raise
    
    logging.info(f"Loaded {len(samples)} samples from {file_path}")
    return samples

def load_atom_mappings(atoms_file: Optional[Path]) -> Dict[str, List[str]]:
    """Load selected atoms per sample from file."""
    if not atoms_file or not atoms_file.exists():
        logging.warning("No atoms file provided or file not found")
        return {}
    
    try:
        with open(atoms_file, 'r') as f:
            atom_data = json.load(f)
        
        # Handle different formats
        if isinstance(atom_data, dict):
            if 'selected_atoms_per_sample' in atom_data:
                return atom_data['selected_atoms_per_sample']
            else:
                return atom_data
        
        logging.warning(f"Unexpected atoms file format: {type(atom_data)}")
        return {}
        
    except Exception as e:
        logging.error(f"Failed to load atoms from {atoms_file}: {e}")
        return {}

def create_diagnostic_config(args) -> DiagnosticConfig:
    """Create diagnostic configuration from command line arguments."""
    
    # Parse keep ratios
    keep_ratios = None
    if args.keep_ratios:
        try:
            keep_ratios = [float(x) for x in args.keep_ratios.split(',')]
        except ValueError as e:
            logging.error(f"Invalid keep_ratios format: {e}")
            sys.exit(1)
    
    # Parse k values
    k_values = None
    if args.k_values:
        try:
            k_values = [int(x) for x in args.k_values.split(',')]
        except ValueError as e:
            logging.error(f"Invalid k_values format: {e}")
            sys.exit(1)
    
    return DiagnosticConfig(
        max_samples_per_rung=args.max_samples,
        mixed_task_sampling=not args.no_mixed_sampling,
        random_seed=args.seed,
        
        rung0_enable=not args.skip_rung0,
        rung1_k_values=k_values,
        rung2_enable_all_extractors=not args.minimal_extractors,
        rung3_oracle_budget=args.oracle_budget,
        rung4_curriculum_enable=not args.skip_curriculum,
        rung5_keep_ratios=keep_ratios,
        
        fail_fast=not args.no_fail_fast,
        parallel_samples=args.parallel,
        
        output_dir=Path(args.output_dir),
        save_intermediate=not args.no_save_intermediate,
        ledger_db_path=Path(args.ledger_db) if args.ledger_db else None
    )

def print_diagnostic_summary(result):
    """Print human-readable diagnostic summary."""
    
    print("\n" + "="*60)
    print("DIAGNOSTIC LADDER SUMMARY")
    print("="*60)
    
    print(f"\nExecution Time: {result.total_duration_seconds:.1f} seconds")
    print(f"Samples Processed: {result.samples_processed}")
    print(f"Ledger Entries Created: {result.ledger_entries_created}")
    
    print(f"\nPRIMARY DIAGNOSIS:")
    print(f"  {result.primary_diagnosis}")
    
    print(f"\nDECISION TRAIL:")
    for i, decision in enumerate(result.decision_trail, 1):
        print(f"  {i}. {decision}")
    
    print(f"\nRECOMMENDED ACTIONS:")
    for action in result.recommended_actions:
        print(f"  {action}")
    
    print(f"\nRUNG TIMINGS:")
    for rung_num, timing in result.rung_timings.items():
        print(f"  Rung {rung_num}: {timing:.1f}s")
    
    print(f"\nRUNG RESULTS SUMMARY:")
    for rung_num, rung_result in result.rung_results.items():
        print(f"  Rung {rung_num}: ", end="")
        
        if rung_num == 0:  # Scoring sanity
            passed = rung_result.get('overall_passed', False)
            tests_passed = rung_result.get('tests_passed', 0)
            total_tests = rung_result.get('total_tests', 0)
            print(f"{'PASS' if passed else 'FAIL'} ({tests_passed}/{total_tests} tests)")
        
        elif rung_num == 1:  # Coverage analysis
            coverage = rung_result.get('overall_span_coverage_at_5', {})
            mean_coverage = coverage.get('mean', 0.0)
            print(f"SpanCoverage@5 = {mean_coverage:.1%}")
        
        elif rung_num == 2:  # Extractive baselines
            performance = rung_result.get('overall_performance', {})
            macro_p5 = performance.get('macro_p5', 0.0)
            print(f"Extractive P@5 = {macro_p5:.3f}")
        
        elif rung_num == 3:  # Oracle bounds
            gap_analysis = rung_result.get('gap_analysis', {})
            ceiling_gap = gap_analysis.get('overall_analysis', {}).get('ceiling_gap', {})
            if ceiling_gap:
                abs_gap = ceiling_gap.get('absolute_gap', 0.0)
                print(f"Ceiling gap = {abs_gap:.3f}")
            else:
                print("No gap analysis available")
        
        else:
            print("Analysis completed")
    
    print("\n" + "="*60)

def main():
    """Main CLI entry point."""
    
    parser = argparse.ArgumentParser(
        description="Run diagnostic ladder of proofs for InfiniteBench evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full diagnostic on evaluation results
  python scripts/run_diagnostic_ladder.py --samples_file results/samples.jsonl --atoms_file results/selected_atoms.json
  
  # Quick diagnostic with reduced samples
  python scripts/run_diagnostic_ladder.py --quick --max_samples 50 --samples_file data.jsonl
  
  # Focus on specific rungs
  python scripts/run_diagnostic_ladder.py --skip_rung0 --samples_file data.jsonl
  
  # Custom configuration
  python scripts/run_diagnostic_ladder.py --samples_file data.jsonl --keep_ratios "0.1,0.2,0.3" --k_values "1,5,10"
        """)
    
    # Input files
    parser.add_argument('--samples_file', type=Path, required=True,
                       help='JSONL file containing evaluation samples with ground truth')
    parser.add_argument('--atoms_file', type=Path,
                       help='JSON file containing selected atoms per sample')
    parser.add_argument('--all_atoms_file', type=Path,
                       help='JSON file containing all available atoms per sample')
    
    # Sample selection
    parser.add_argument('--max_samples', type=int, default=200,
                       help='Maximum samples per rung (default: 200)')
    parser.add_argument('--no_mixed_sampling', action='store_true',
                       help='Disable mixed task sampling')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    # Rung configuration
    parser.add_argument('--skip_rung0', action='store_true',
                       help='Skip Rung 0 (scoring sanity checks)')
    parser.add_argument('--k_values', type=str, default='1,5,10,20',
                       help='Comma-separated k values for coverage analysis (default: 1,5,10,20)')
    parser.add_argument('--minimal_extractors', action='store_true',
                       help='Use minimal set of extractors for Rung 2')
    parser.add_argument('--oracle_budget', type=int, default=10,
                       help='Budget for oracle context selection (default: 10)')
    parser.add_argument('--skip_curriculum', action='store_true',
                       help='Skip Rung 4 (curriculum analysis)')
    parser.add_argument('--keep_ratios', type=str, default='0.05,0.1,0.15,0.2,0.3,0.5',
                       help='Comma-separated keep ratios for Rung 5 (default: 0.05,0.1,0.15,0.2,0.3,0.5)')
    
    # Execution options
    parser.add_argument('--no_fail_fast', action='store_true',
                       help='Continue execution even after critical failures')
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing of samples')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='diagnostic_results',
                       help='Output directory for results (default: diagnostic_results)')
    parser.add_argument('--no_save_intermediate', action='store_true',
                       help='Skip saving intermediate rung results')
    parser.add_argument('--ledger_db', type=str,
                       help='Path to ledger database file (default: output_dir/diagnostic_ledger.db)')
    
    # Logging and convenience
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output - only final summary')
    parser.add_argument('--quick', action='store_true',
                       help='Quick diagnostic: max_samples=50, skip_curriculum=True, minimal_extractors=True')
    
    args = parser.parse_args()
    
    # Apply quick mode settings
    if args.quick:
        args.max_samples = min(args.max_samples, 50)
        args.skip_curriculum = True
        args.minimal_extractors = True
    
    # Setup logging
    if not args.quiet:
        setup_logging(args.verbose)
    
    try:
        # Load input data
        logging.info("Loading evaluation samples...")
        samples = load_samples_from_file(args.samples_file)
        
        if not samples:
            logging.error("No samples loaded - cannot proceed")
            sys.exit(1)
        
        # Load atom mappings if provided
        selected_atoms_per_sample = load_atom_mappings(args.atoms_file)
        all_atoms_per_sample = load_atom_mappings(args.all_atoms_file)
        
        # Create diagnostic configuration
        config = create_diagnostic_config(args)
        
        logging.info(f"Starting diagnostic ladder with config: {config}")
        
        # Run diagnostic ladder
        runner = DiagnosticLadderRunner(config)
        result = runner.run_complete_ladder(
            samples=samples,
            selected_atoms_per_sample=selected_atoms_per_sample,
            all_atoms_per_sample=all_atoms_per_sample
        )
        
        # Save results
        runner.save_final_result(result)
        
        # Print summary
        if not args.quiet:
            print_diagnostic_summary(result)
        
        # Exit with appropriate code
        if "CRITICAL" in result.primary_diagnosis or "FAILURE" in result.primary_diagnosis:
            sys.exit(2)  # Critical failure
        elif "WARNING" in result.primary_diagnosis:
            sys.exit(1)  # Warning
        else:
            sys.exit(0)  # Success
            
    except KeyboardInterrupt:
        logging.info("Diagnostic ladder interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        logging.error(f"Diagnostic ladder failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()