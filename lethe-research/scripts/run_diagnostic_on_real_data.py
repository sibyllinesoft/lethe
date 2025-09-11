#!/usr/bin/env python3
"""
Run Diagnostic Ladder on Real InfiniteBench Data
===============================================

This script runs the complete 5-rung diagnostic ladder on real InfiniteBench
evaluation data to provide definitive diagnosis of the 0.000 accuracy issue.

Usage:
    # Extract data and run diagnostics in one step
    python scripts/run_diagnostic_on_real_data.py --method hybrid --keep_ratio 0.08 --max_samples 50
    
    # Run diagnostics on existing extracted data
    python scripts/run_diagnostic_on_real_data.py --ledger_db diagnostic_extraction_data/extracted_data_ledger.db
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from diagnostics.ladder_runner import DiagnosticLadderRunner, DiagnosticConfig
from diagnostics.sample_ledger import SampleLedger
from scripts.extract_evaluation_data import EvaluationDataExtractor, ExtractionConfig

logger = logging.getLogger(__name__)

class RealDataDiagnosticRunner:
    """Run complete diagnostic ladder on real InfiniteBench evaluation data."""
    
    def __init__(self, output_dir: Path = Path("real_data_diagnostics")):
        """Initialize diagnostic runner for real data."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_complete_diagnostic_pipeline(self, 
                                       extraction_config: Optional[ExtractionConfig] = None,
                                       ledger_db_path: Optional[Path] = None,
                                       diagnostic_config: Optional[DiagnosticConfig] = None) -> Dict[str, Any]:
        """
        Run the complete pipeline: extract data (if needed) + run diagnostics.
        
        Args:
            extraction_config: Configuration for data extraction (if needed)
            ledger_db_path: Path to existing ledger DB (skip extraction if provided)
            diagnostic_config: Configuration for diagnostic ladder
            
        Returns:
            Complete diagnostic results with decision
        """
        start_time = time.time()
        
        # Step 1: Extract data if needed
        if ledger_db_path and ledger_db_path.exists():
            logger.info(f"Using existing ledger data: {ledger_db_path}")
        else:
            if not extraction_config:
                raise ValueError("Must provide extraction_config if no ledger_db_path given")
            
            logger.info("🔍 Step 1: Extracting evaluation data...")
            ledger_db_path = self._extract_evaluation_data(extraction_config)
            
            if not ledger_db_path or not ledger_db_path.exists():
                raise RuntimeError("Data extraction failed")
        
        # Step 2: Verify ledger data
        logger.info("📊 Step 2: Verifying ledger data...")
        sample_count = self._verify_ledger_data(ledger_db_path)
        
        if sample_count == 0:
            raise RuntimeError("No samples found in ledger database")
        
        logger.info(f"✅ Verified {sample_count} samples in ledger")
        
        # Step 3: Run diagnostic ladder
        logger.info("🔬 Step 3: Running 5-rung diagnostic ladder...")
        
        if not diagnostic_config:
            diagnostic_config = DiagnosticConfig(
                max_samples_per_rung=min(200, sample_count),
                ledger_db_path=ledger_db_path,
                output_dir=self.output_dir,
                fail_fast=False,  # Want complete analysis
                save_intermediate=True
            )
        
        ladder_runner = DiagnosticLadderRunner(diagnostic_config)
        diagnostic_result = ladder_runner.run_complete_ladder()
        
        # Step 4: Generate comprehensive report
        logger.info("📝 Step 4: Generating comprehensive diagnostic report...")
        report = self._generate_comprehensive_report(
            diagnostic_result, 
            extraction_config,
            sample_count,
            time.time() - start_time
        )
        
        # Save results
        results_file = self.output_dir / "complete_diagnostic_results.json"
        with open(results_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"💾 Complete results saved to: {results_file}")
        
        # Print decision summary
        self._print_decision_summary(report)
        
        return report
    
    def _extract_evaluation_data(self, config: ExtractionConfig) -> Path:
        """Extract evaluation data and return ledger DB path."""
        # Set output directory relative to our diagnostic output
        config.output_dir = self.output_dir / "extraction_data"
        
        # Run extraction
        extractor = EvaluationDataExtractor(config)
        entries = extractor.extract_from_live_evaluation()
        
        if not entries:
            raise RuntimeError("No evaluation data extracted")
        
        # Save to ledger
        ledger_db_path = config.output_dir / "extracted_data_ledger.db"
        extractor.save_ledger_entries(entries, ledger_db_path)
        
        logger.info(f"✅ Extracted {len(entries)} samples to ledger: {ledger_db_path}")
        return ledger_db_path
    
    def _verify_ledger_data(self, ledger_db_path: Path) -> int:
        """Verify ledger data integrity and return sample count."""
        try:
            ledger = SampleLedger(ledger_db_path)
            
            # Get sample statistics
            entries = ledger.get_all_entries()
            
            if not entries:
                logger.error("No entries found in ledger database")
                return 0
            
            # Verify data integrity
            datasets = set()
            keep_ratios = set()
            methods = set()
            
            for entry in entries:
                datasets.add(entry.dataset)
                keep_ratios.add(entry.keep_ratio)
                
                # Verify required fields
                if not entry.sample_id:
                    logger.warning(f"Entry missing sample_id")
                if not entry.gold_answers:
                    logger.warning(f"Entry {entry.sample_id} missing gold_answers")
                if not entry.selected_atoms:
                    logger.warning(f"Entry {entry.sample_id} missing selected_atoms")
            
            logger.info(f"Ledger contains:")
            logger.info(f"  - {len(entries)} sample entries")
            logger.info(f"  - Datasets: {sorted(datasets)}")
            logger.info(f"  - Keep ratios: {sorted(keep_ratios)}")
            
            return len(entries)
            
        except Exception as e:
            logger.error(f"Error verifying ledger data: {e}")
            return 0
    
    def _generate_comprehensive_report(self, 
                                     diagnostic_result: Any,
                                     extraction_config: Optional[ExtractionConfig],
                                     sample_count: int,
                                     total_time: float) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report."""
        
        # Convert diagnostic result to dict if needed
        if hasattr(diagnostic_result, '__dict__'):
            diagnostic_dict = asdict(diagnostic_result)
        else:
            diagnostic_dict = diagnostic_result
        
        report = {
            "experiment_metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_duration_seconds": total_time,
                "samples_analyzed": sample_count,
                "extraction_config": asdict(extraction_config) if extraction_config else None,
                "purpose": "Definitive diagnosis of InfiniteBench 0.000 accuracy issue"
            },
            
            "diagnostic_results": diagnostic_dict,
            
            "decision_analysis": self._analyze_diagnostic_decision(diagnostic_dict),
            
            "actionable_recommendations": self._generate_actionable_recommendations(diagnostic_dict),
            
            "technical_summary": {
                "primary_failure_mode": diagnostic_dict.get("primary_diagnosis", "Unknown"),
                "confidence_level": "High" if sample_count >= 50 else "Medium",
                "critical_rungs": self._identify_critical_rungs(diagnostic_dict),
                "repair_priority": diagnostic_dict.get("recommended_actions", [])
            }
        }
        
        return report
    
    def _analyze_diagnostic_decision(self, diagnostic_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the diagnostic decision logic."""
        rung_results = diagnostic_dict.get("rung_results", {})
        
        analysis = {
            "decision_logic_trace": [],
            "rung_performance": {},
            "failure_point_analysis": {},
            "confidence_assessment": {}
        }
        
        # Analyze each rung
        for rung_num in range(6):  # Rungs 0-5
            rung_key = str(rung_num)
            if rung_key in rung_results:
                rung_data = rung_results[rung_key]
                
                analysis["rung_performance"][rung_num] = {
                    "status": "pass" if rung_data.get("success", False) else "fail",
                    "key_metrics": self._extract_key_metrics(rung_data, rung_num),
                    "issues_found": rung_data.get("issues_found", [])
                }
                
                # Decision logic trace
                if rung_num == 1:  # SpanCoverage@K
                    coverage = rung_data.get("span_coverage_5", 0.0)
                    analysis["decision_logic_trace"].append(
                        f"Rung 1 SpanCoverage@5: {coverage:.3f} " +
                        ("→ selection/retrieval OK" if coverage > 0.1 else "→ FIX selection/retrieval")
                    )
                
                elif rung_num == 2:  # Extractive vs LLM comparison
                    extractive_p5 = rung_data.get("extractive_p5", 0.0)
                    llm_p5 = rung_data.get("llm_p5", 0.0) if "llm_p5" in rung_data else 0.0
                    
                    analysis["decision_logic_trace"].append(
                        f"Rung 2 Extractive P@5: {extractive_p5:.3f}, LLM P@5: {llm_p5:.3f} " +
                        ("→ generation/format OK" if abs(extractive_p5 - llm_p5) < 0.2 else "→ FIX generation/format")
                    )
        
        return analysis
    
    def _extract_key_metrics(self, rung_data: Dict[str, Any], rung_num: int) -> Dict[str, Any]:
        """Extract key metrics for each rung."""
        key_metrics = {}
        
        if rung_num == 0:  # Scoring sanity
            key_metrics = {
                "exact_matches": rung_data.get("exact_matches", 0),
                "partial_matches": rung_data.get("partial_matches", 0),
                "normalization_issues": rung_data.get("normalization_issues", 0)
            }
        
        elif rung_num == 1:  # Coverage analysis
            key_metrics = {
                "span_coverage_5": rung_data.get("span_coverage_5", 0.0),
                "span_coverage_10": rung_data.get("span_coverage_10", 0.0),
                "symbol_coverage_5": rung_data.get("symbol_coverage_5", 0.0),
                "atom_utilization": rung_data.get("atom_utilization", 0.0)
            }
        
        elif rung_num == 2:  # Extractive baselines
            key_metrics = {
                "extractive_p5": rung_data.get("extractive_p5", 0.0),
                "extractive_p10": rung_data.get("extractive_p10", 0.0),
                "mean_extractive_score": rung_data.get("mean_extractive_score", 0.0)
            }
        
        elif rung_num == 3:  # Oracle bounds
            key_metrics = {
                "oracle_upper_bound": rung_data.get("oracle_upper_bound", 0.0),
                "achievable_ceiling": rung_data.get("achievable_ceiling", 0.0),
                "optimal_atom_count": rung_data.get("optimal_atom_count", 0)
            }
        
        return key_metrics
    
    def _identify_critical_rungs(self, diagnostic_dict: Dict[str, Any]) -> List[int]:
        """Identify which rungs show critical failures."""
        critical_rungs = []
        rung_results = diagnostic_dict.get("rung_results", {})
        
        for rung_num in range(6):
            rung_key = str(rung_num)
            if rung_key in rung_results:
                rung_data = rung_results[rung_key]
                
                # Define critical failure conditions per rung
                is_critical = False
                
                if rung_num == 1:  # Coverage analysis
                    span_coverage = rung_data.get("span_coverage_5", 0.0)
                    is_critical = span_coverage < 0.1  # Less than 10% coverage
                
                elif rung_num == 2:  # Extractive baselines  
                    extractive_p5 = rung_data.get("extractive_p5", 0.0)
                    is_critical = extractive_p5 < 0.05  # Less than 5% precision
                
                elif rung_num == 3:  # Oracle bounds
                    oracle_bound = rung_data.get("oracle_upper_bound", 0.0)
                    is_critical = oracle_bound < 0.2  # Less than 20% achievable
                
                if is_critical:
                    critical_rungs.append(rung_num)
        
        return critical_rungs
    
    def _generate_actionable_recommendations(self, diagnostic_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific, actionable recommendations based on diagnostic results."""
        recommendations = []
        rung_results = diagnostic_dict.get("rung_results", {})
        primary_diagnosis = diagnostic_dict.get("primary_diagnosis", "")
        
        # Rung 1: Selection/Retrieval Issues
        if "1" in rung_results:
            span_coverage = rung_results["1"].get("span_coverage_5", 0.0)
            if span_coverage < 0.1:
                recommendations.append({
                    "priority": "CRITICAL",
                    "category": "Selection/Retrieval",
                    "issue": f"SpanCoverage@5 = {span_coverage:.3f} (< 0.1 threshold)",
                    "action": "Fix Lethe context selection algorithm",
                    "specific_steps": [
                        "Increase keep_ratio from 0.08 to 0.15+",
                        "Tune DPP diversity parameters (lambda_diversity)",
                        "Verify atom segmentation preserves answer spans",
                        "Check if answers are being split across atom boundaries"
                    ],
                    "expected_improvement": "SpanCoverage@5 should improve to >0.3",
                    "test_command": "python scripts/run_diagnostic_on_real_data.py --keep_ratio 0.15"
                })
        
        # Rung 2: Generation/Formatting Issues
        if "2" in rung_results:
            extractive_p5 = rung_results["2"].get("extractive_p5", 0.0)
            if extractive_p5 > 0.1:  # Extractive works but LLM doesn't
                recommendations.append({
                    "priority": "HIGH",
                    "category": "Generation/Formatting",
                    "issue": f"Extractive P@5 = {extractive_p5:.3f} but LLM P@5 ≈ 0.0",
                    "action": "Fix answer extraction/normalization pipeline",
                    "specific_steps": [
                        "Debug LLM response parsing in hybrid_evaluation.py",
                        "Check answer normalization in infinitebench/metrics.py",
                        "Verify prompt templates include proper answer formatting",
                        "Test LLM streaming vs completion modes"
                    ],
                    "expected_improvement": "LLM P@5 should match Extractive P@5",
                    "test_command": "python scripts/simple_evaluation_diagnostics.py --debug_generation"
                })
        
        # Rung 3: Lethe Configuration Issues
        if "3" in rung_results:
            oracle_bound = rung_results["3"].get("oracle_upper_bound", 0.0)
            achievable = rung_results["3"].get("achievable_ceiling", 0.0)
            if oracle_bound > 0.5 and achievable < 0.2:  # High potential, low achievement
                recommendations.append({
                    "priority": "MEDIUM",
                    "category": "Lethe Configuration",
                    "issue": f"Oracle bound = {oracle_bound:.3f} but achievable = {achievable:.3f}",
                    "action": "Optimize Lethe selection parameters",
                    "specific_steps": [
                        "Tune DPP rank parameter (current: 14)",
                        "Adjust window_size and stride parameters",
                        "Experiment with different atom segmentation strategies",
                        "Test semantic vs lexical similarity weighting"
                    ],
                    "expected_improvement": "Achievable ceiling should approach oracle bound",
                    "test_command": "python scripts/tune_lethe_parameters.py"
                })
        
        # Model Limitation Assessment
        if all(rung_results.get(str(i), {}).get("success", False) for i in range(1, 4)):
            # If early rungs pass but performance is still low
            recommendations.append({
                "priority": "LOW",
                "category": "Model Limitations", 
                "issue": "All diagnostic rungs pass but accuracy remains low",
                "action": "Accept 'InfiniteBench is hard' conclusion",
                "specific_steps": [
                    "Document that diagnostic ladder shows no pipeline bugs",
                    "Consider stronger foundation models (GPT-4, Claude-3)",
                    "Experiment with few-shot prompting strategies",
                    "Report baseline results for InfiniteBench difficulty assessment"
                ],
                "expected_improvement": "N/A - this validates the evaluation is working correctly",
                "test_command": "python scripts/generate_baseline_report.py"
            })
        
        # Sort by priority
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 4))
        
        return recommendations
    
    def _print_decision_summary(self, report: Dict[str, Any]):
        """Print a clear decision summary to console."""
        print("\n" + "="*80)
        print("🔬 INFINITEBENCH DIAGNOSTIC LADDER DECISION SUMMARY")
        print("="*80)
        
        primary_diagnosis = report.get("diagnostic_results", {}).get("primary_diagnosis", "Unknown")
        samples_analyzed = report.get("experiment_metadata", {}).get("samples_analyzed", 0)
        
        print(f"📊 Samples Analyzed: {samples_analyzed}")
        print(f"🎯 Primary Diagnosis: {primary_diagnosis}")
        
        recommendations = report.get("actionable_recommendations", [])
        if recommendations:
            print(f"\n🚨 CRITICAL ACTIONS REQUIRED:")
            for rec in recommendations:
                if rec["priority"] in ["CRITICAL", "HIGH"]:
                    print(f"   • {rec['category']}: {rec['action']}")
                    print(f"     Issue: {rec['issue']}")
                    print(f"     Test: {rec['test_command']}")
                    print()
        
        critical_rungs = report.get("technical_summary", {}).get("critical_rungs", [])
        if critical_rungs:
            print(f"❌ Critical Failures in Rungs: {critical_rungs}")
        else:
            print(f"✅ All diagnostic rungs passed - Model may be at capability limit")
        
        print("\n📁 Full results saved to:", report.get("results_file", "complete_diagnostic_results.json"))
        print("="*80)

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def main():
    parser = argparse.ArgumentParser(description="Run diagnostic ladder on real InfiniteBench data")
    
    # Data extraction options
    parser.add_argument("--method", choices=["hybrid", "lethe", "streaming"], default="hybrid",
                       help="Evaluation method to use for extraction")
    parser.add_argument("--keep_ratio", type=float, default=0.08,
                       help="Context keep ratio for evaluation")
    parser.add_argument("--dataset", choices=["code", "zh_qa"], default="code",
                       help="Dataset to evaluate on")
    parser.add_argument("--max_samples", type=int, default=50,
                       help="Maximum number of samples to process")
    
    # Existing data options
    parser.add_argument("--ledger_db", type=Path,
                       help="Path to existing ledger database (skip extraction)")
    
    # Output options
    parser.add_argument("--output_dir", type=Path, default="real_data_diagnostics",
                       help="Output directory for diagnostic results")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    try:
        logger.info("🚀 Starting comprehensive InfiniteBench diagnostic analysis...")
        
        runner = RealDataDiagnosticRunner(args.output_dir)
        
        # Setup extraction config if needed
        extraction_config = None
        if not args.ledger_db:
            extraction_config = ExtractionConfig(
                method=args.method,
                keep_ratio=args.keep_ratio,
                dataset=args.dataset,
                max_samples=args.max_samples
            )
        
        # Run complete pipeline
        report = runner.run_complete_diagnostic_pipeline(
            extraction_config=extraction_config,
            ledger_db_path=args.ledger_db
        )
        
        logger.info("✅ Diagnostic analysis completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Diagnostic analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())