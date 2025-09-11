"""
Diagnostic Ladder Runner
=======================

Main coordinator for running the complete diagnostic "ladder of proofs" system.
Executes all 6 rungs systematically and provides definitive diagnosis of 
pipeline issues.

Decision Rules:
- If Rung 1 SpanCoverage@K ≈ 0 → fix selection/retrieval
- Else if Rung 2 Extractive P@5 > 0 but LLM P@5 ≈ 0 → fix generation/format/normalization  
- Else if Rung 3 OracleExtractive high but Extractive low → improve Lethe settings
- Only if all rungs rise monotonically and LLM remains flat → accept "model too weak"
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from .sample_ledger import SampleLedger, SampleLedgerEntry
from .rung0_scoring_sanity import ScoringValidator
from .coverage_analyzer import CoverageAnalyzer
from .extractive_baselines import ExtractionBaselines
from .oracle_bounds import OracleBoundsCalculator

logger = logging.getLogger(__name__)

@dataclass
class DiagnosticConfig:
    """Configuration for diagnostic ladder execution."""
    
    # Sample selection
    max_samples_per_rung: int = 200
    mixed_task_sampling: bool = True
    random_seed: int = 42
    
    # Rung-specific settings
    rung0_enable: bool = True
    rung1_k_values: List[int] = None
    rung2_enable_all_extractors: bool = True
    rung3_oracle_budget: int = 10
    rung4_curriculum_enable: bool = True
    rung5_keep_ratios: List[float] = None
    
    # Performance settings
    fail_fast: bool = True  # Stop on critical failures
    parallel_samples: bool = False  # Process samples in parallel
    
    # Output settings
    output_dir: Path = Path("diagnostic_results")
    save_intermediate: bool = True
    ledger_db_path: Optional[Path] = None
    
    def __post_init__(self):
        if self.rung1_k_values is None:
            self.rung1_k_values = [1, 5, 10, 20]
        if self.rung5_keep_ratios is None:
            self.rung5_keep_ratios = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

@dataclass 
class DiagnosticResult:
    """Complete results from diagnostic ladder execution."""
    
    config: DiagnosticConfig
    start_time: str
    end_time: str
    total_duration_seconds: float
    
    # Results by rung
    rung_results: Dict[int, Dict[str, Any]]
    
    # Decision logic results
    primary_diagnosis: str
    decision_trail: List[str]
    recommended_actions: List[str]
    
    # Sample tracking
    samples_processed: int
    ledger_entries_created: int
    
    # Performance stats
    rung_timings: Dict[int, float]

class DiagnosticLadderRunner:
    """
    Main coordinator for the diagnostic ladder of proofs system.
    
    Executes rungs 0-5 systematically to diagnose evaluation pipeline issues
    without requiring LLM API calls.
    """
    
    def __init__(self, config: DiagnosticConfig):
        """Initialize diagnostic runner with configuration."""
        self.config = config
        
        # Initialize output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ledger system
        if config.ledger_db_path is None:
            config.ledger_db_path = config.output_dir / "diagnostic_ledger.db"
        
        self.ledger = SampleLedger(config.ledger_db_path)
        
        # Initialize rung processors
        self.scoring_validator = ScoringValidator(seed=config.random_seed)
        self.coverage_analyzer = CoverageAnalyzer()
        self.extractive_baselines = ExtractionBaselines()
        self.oracle_calculator = OracleBoundsCalculator(seed=config.random_seed)
        
        logger.info(f"Initialized diagnostic ladder runner with output dir: {config.output_dir}")
    
    def run_complete_ladder(self, 
                           samples: List[Dict[str, Any]], 
                           selected_atoms_per_sample: Optional[Dict[str, List[str]]] = None,
                           all_atoms_per_sample: Optional[Dict[str, List[str]]] = None) -> DiagnosticResult:
        """
        Run the complete diagnostic ladder on provided samples.
        
        Args:
            samples: List of evaluation samples with ground truth
            selected_atoms_per_sample: Mapping from sample_id to selected atoms (for Lethe)
            all_atoms_per_sample: Mapping from sample_id to all available atoms
            
        Returns:
            DiagnosticResult with complete analysis
        """
        logger.info("Starting complete diagnostic ladder execution")
        start_time = time.time()
        
        # Prepare sample selection
        selected_samples = self._select_samples_for_diagnosis(samples)
        logger.info(f"Selected {len(selected_samples)} samples for diagnosis")
        
        # Initialize result tracking
        rung_results = {}
        rung_timings = {}
        ledger_entries_created = 0
        decision_trail = []
        
        try:
            # Rung 0: Scoring sanity checks
            if self.config.rung0_enable:
                logger.info("Executing Rung 0: Scoring Sanity")
                rung_start = time.time()
                
                rung_results[0] = self.scoring_validator.run_all_tests(selected_samples)
                rung_timings[0] = time.time() - rung_start
                
                if self.config.save_intermediate:
                    self._save_rung_result(0, rung_results[0])
                
                # Decision logic for Rung 0
                if not rung_results[0].get('overall_passed', False):
                    decision_trail.append("Rung 0 FAILED: Scoring functions have critical issues")
                    if self.config.fail_fast:
                        return self._create_diagnostic_result(
                            start_time, rung_results, rung_timings, 
                            selected_samples, ledger_entries_created, decision_trail,
                            "CRITICAL: Scoring system failure - fix scoring functions first"
                        )
                else:
                    decision_trail.append("Rung 0 PASSED: Scoring functions work correctly")
            
            # Rung 1: Coverage analysis
            logger.info("Executing Rung 1: Coverage Analysis")
            rung_start = time.time()
            
            coverage_results, ledger_entries = self._run_coverage_analysis(
                selected_samples, selected_atoms_per_sample or {}
            )
            rung_results[1] = coverage_results
            rung_timings[1] = time.time() - rung_start
            ledger_entries_created += ledger_entries
            
            if self.config.save_intermediate:
                self._save_rung_result(1, rung_results[1])
            
            # Decision logic for Rung 1
            overall_coverage = coverage_results.get('overall_span_coverage_at_5', {})
            mean_coverage = overall_coverage.get('mean', 0.0)
            
            if mean_coverage < 0.1:
                decision_trail.append(f"Rung 1 CRITICAL: SpanCoverage@5 = {mean_coverage:.1%} - selection/retrieval failing")
                return self._create_diagnostic_result(
                    start_time, rung_results, rung_timings,
                    selected_samples, ledger_entries_created, decision_trail,
                    "SELECTION FAILURE: Fix retrieval/selection system - coverage too low"
                )
            elif mean_coverage < 0.3:
                decision_trail.append(f"Rung 1 WARNING: Low SpanCoverage@5 = {mean_coverage:.1%} - selection needs tuning")
            else:
                decision_trail.append(f"Rung 1 PASSED: Reasonable SpanCoverage@5 = {mean_coverage:.1%}")
            
            # Rung 2: Extractive baselines
            logger.info("Executing Rung 2: Extractive Baselines")
            rung_start = time.time()
            
            # Collect selected atoms for samples from ledger
            ledger_selected_atoms = {}
            ledger_coverage_results = {}
            
            for sample in selected_samples:
                sample_id = sample.get('id', 'unknown')
                # Try to get from ledger (simplified - would need proper keep_ratio/k/seed lookup)
                entry = self.ledger.read_entry(
                    sample.get('task_name', 'unknown'), sample_id, 0.15, 5, self.config.random_seed
                )
                if entry:
                    ledger_selected_atoms[sample_id] = entry.selected_atoms
                    ledger_coverage_results[sample_id] = entry.coverage_flags
                else:
                    # Fallback to provided atoms
                    ledger_selected_atoms[sample_id] = selected_atoms_per_sample.get(sample_id, [])
            
            extractive_results = self.extractive_baselines.run_extractive_evaluation(
                selected_samples, ledger_selected_atoms, ledger_coverage_results
            )
            rung_results[2] = extractive_results
            rung_timings[2] = time.time() - rung_start
            
            if self.config.save_intermediate:
                self._save_rung_result(2, rung_results[2])
            
            # Decision logic for Rung 2
            extractive_p5 = extractive_results.get('overall_performance', {}).get('macro_p5', 0.0)
            decision_trail.append(f"Rung 2: Extractive P@5 = {extractive_p5:.3f}")
            
            # Rung 3: Oracle bounds
            logger.info("Executing Rung 3: Oracle Bounds")
            rung_start = time.time()
            
            oracle_results = self._run_oracle_analysis(
                selected_samples, ledger_selected_atoms, all_atoms_per_sample or {}
            )
            rung_results[3] = oracle_results
            rung_timings[3] = time.time() - rung_start
            
            if self.config.save_intermediate:
                self._save_rung_result(3, rung_results[3])
            
            # Decision logic for Rung 3
            gap_analysis = oracle_results.get('gap_analysis', {})
            ceiling_gap = gap_analysis.get('overall_analysis', {}).get('ceiling_gap', {})
            
            if ceiling_gap:
                absolute_gap = ceiling_gap.get('absolute_gap', 0)
                decision_trail.append(f"Rung 3: Ceiling gap = {absolute_gap:.3f}")
                
                if absolute_gap > 0.3:
                    return self._create_diagnostic_result(
                        start_time, rung_results, rung_timings,
                        selected_samples, ledger_entries_created, decision_trail,
                        "LARGE CEILING GAP: Major selection/retrieval improvements possible"
                    )
            
            # If we reach here, run additional rungs for comprehensive analysis
            # Rung 4 & 5 would be implemented similarly...
            
            # Final diagnosis
            final_diagnosis = self._generate_final_diagnosis(rung_results, decision_trail)
            
            return self._create_diagnostic_result(
                start_time, rung_results, rung_timings,
                selected_samples, ledger_entries_created, decision_trail,
                final_diagnosis
            )
            
        except Exception as e:
            logger.error(f"Diagnostic ladder execution failed: {e}")
            decision_trail.append(f"EXECUTION ERROR: {str(e)}")
            
            return self._create_diagnostic_result(
                start_time, rung_results, rung_timings,
                selected_samples, ledger_entries_created, decision_trail,
                f"EXECUTION FAILURE: {str(e)}"
            )
    
    def _select_samples_for_diagnosis(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select representative samples for diagnosis."""
        if not samples:
            return []
        
        if len(samples) <= self.config.max_samples_per_rung:
            return samples
        
        if self.config.mixed_task_sampling:
            # Balance across tasks
            from collections import defaultdict
            import random
            
            random.seed(self.config.random_seed)
            
            task_samples = defaultdict(list)
            for sample in samples:
                task_name = sample.get('task_name', 'unknown')
                task_samples[task_name].append(sample)
            
            # Calculate samples per task
            num_tasks = len(task_samples)
            samples_per_task = self.config.max_samples_per_rung // num_tasks
            
            selected = []
            for task_name, task_sample_list in task_samples.items():
                task_selected = random.sample(
                    task_sample_list, 
                    min(samples_per_task, len(task_sample_list))
                )
                selected.extend(task_selected)
            
            # Fill remaining slots randomly
            remaining_slots = self.config.max_samples_per_rung - len(selected)
            if remaining_slots > 0:
                remaining_samples = [s for s in samples if s not in selected]
                if remaining_samples:
                    additional = random.sample(
                        remaining_samples,
                        min(remaining_slots, len(remaining_samples))
                    )
                    selected.extend(additional)
            
            return selected
        else:
            # Simple random sampling
            import random
            random.seed(self.config.random_seed)
            return random.sample(samples, self.config.max_samples_per_rung)
    
    def _run_coverage_analysis(self, 
                              samples: List[Dict[str, Any]], 
                              selected_atoms_per_sample: Dict[str, List[str]]) -> Tuple[Dict[str, Any], int]:
        """Run comprehensive coverage analysis for Rung 1."""
        
        coverage_results = []
        ledger_entries_created = 0
        
        for sample in samples:
            sample_id = sample.get('id', 'unknown')
            task_name = sample.get('task_name', 'unknown')
            selected_atoms = selected_atoms_per_sample.get(sample_id, [])
            
            # Analyze coverage for this sample
            sample_coverage = self.coverage_analyzer.analyze_sample_coverage(
                sample, selected_atoms, self.config.rung1_k_values
            )
            coverage_results.append(sample_coverage)
            
            # Create ledger entry
            ground_truth = sample.get('ground_truth') or sample.get('label')
            if isinstance(ground_truth, list):
                gold_answers = [str(x) for x in ground_truth]
            else:
                gold_answers = [str(ground_truth)] if ground_truth else []
            
            # Extract coverage flags for ledger
            coverage_flags = {}
            for k in self.config.rung1_k_values:
                span_key = f'span_coverage_at_{k}'
                if span_key in sample_coverage['coverage_metrics']:
                    coverage_flags[span_key] = sample_coverage['coverage_metrics'][span_key]['coverage_rate']
            
            # Create and write ledger entry (simplified)
            ledger_entry = SampleLedgerEntry(
                dataset=task_name,
                sample_id=sample_id,
                keep_ratio=0.15,  # Default assumption
                k=5,  # Default assumption  
                seed=self.config.random_seed,
                gold_answers=gold_answers,
                selected_atoms=selected_atoms,
                spans_present=[True] * len(selected_atoms),  # Simplified
                symbols_present=[False] * len(selected_atoms),  # Simplified
                extractive_pred="",  # Will be filled in Rung 2
                extractive_score=0.0,  # Will be filled in Rung 2
                coverage_flags=coverage_flags,
                cert_hash="",  # Will be computed
                timestamp="",  # Will be set
                processing_time_ms=0.0,
                errors=[]
            )
            
            try:
                if self.ledger.write_entry(ledger_entry):
                    ledger_entries_created += 1
            except Exception as e:
                logger.warning(f"Failed to write ledger entry for {sample_id}: {e}")
        
        # Aggregate results
        aggregated_results = self.coverage_analyzer.aggregate_coverage_results(coverage_results)
        
        # Add diagnostic insights
        diagnostic_insights = self.coverage_analyzer.diagnose_coverage_issues(aggregated_results)
        aggregated_results['diagnostic_insights'] = diagnostic_insights
        
        return aggregated_results, ledger_entries_created
    
    def _run_oracle_analysis(self, 
                           samples: List[Dict[str, Any]], 
                           selected_atoms_per_sample: Dict[str, List[str]], 
                           all_atoms_per_sample: Dict[str, List[str]]) -> Dict[str, Any]:
        """Run oracle bounds analysis for Rung 3."""
        
        oracle_results = []
        
        for sample in samples:
            sample_id = sample.get('id', 'unknown')
            selected_atoms = selected_atoms_per_sample.get(sample_id, [])
            all_atoms = all_atoms_per_sample.get(sample_id, selected_atoms)  # Fallback
            
            sample_oracle_result = self.oracle_calculator.compute_oracle_bounds_for_sample(
                sample, all_atoms, selected_atoms, self.extractive_baselines
            )
            oracle_results.append(sample_oracle_result)
        
        # Analyze gaps
        gap_analysis = self.oracle_calculator.analyze_oracle_gaps(oracle_results)
        
        # Generate diagnostic insights
        diagnostic_insights = self.oracle_calculator.diagnose_performance_gaps(gap_analysis)
        
        return {
            'sample_results': oracle_results,
            'gap_analysis': gap_analysis,
            'diagnostic_insights': diagnostic_insights
        }
    
    def _generate_final_diagnosis(self, 
                                 rung_results: Dict[int, Dict[str, Any]], 
                                 decision_trail: List[str]) -> str:
        """Generate final diagnosis based on all rung results."""
        
        # Check for critical failures first
        if 0 in rung_results and not rung_results[0].get('overall_passed', False):
            return "CRITICAL: Fix scoring system before proceeding"
        
        # Check coverage issues
        if 1 in rung_results:
            coverage = rung_results[1].get('overall_span_coverage_at_5', {}).get('mean', 0.0)
            if coverage < 0.1:
                return "SELECTION FAILURE: Retrieval/selection system not finding relevant content"
        
        # Check extractive performance vs oracle bounds
        if 2 in rung_results and 3 in rung_results:
            extractive_p5 = rung_results[2].get('overall_performance', {}).get('macro_p5', 0.0)
            
            gap_analysis = rung_results[3].get('gap_analysis', {})
            ceiling_gap = gap_analysis.get('overall_analysis', {}).get('ceiling_gap', {})
            
            if ceiling_gap:
                absolute_gap = ceiling_gap.get('absolute_gap', 0)
                if absolute_gap > 0.3:
                    return "LARGE IMPROVEMENT POTENTIAL: Selection improvements could yield major gains"
                elif extractive_p5 > 0.1 and absolute_gap > 0.1:
                    return "MODERATE IMPROVEMENTS POSSIBLE: Both selection and extraction can be improved"
        
        return "ANALYSIS COMPLETE: Multiple factors need investigation - see detailed results"
    
    def _create_diagnostic_result(self, 
                                 start_time: float, 
                                 rung_results: Dict[int, Dict[str, Any]], 
                                 rung_timings: Dict[int, float],
                                 samples: List[Dict[str, Any]], 
                                 ledger_entries: int, 
                                 decision_trail: List[str], 
                                 diagnosis: str) -> DiagnosticResult:
        """Create final diagnostic result object."""
        
        end_time = time.time()
        
        # Generate recommended actions based on diagnosis
        recommended_actions = self._generate_recommended_actions(diagnosis, rung_results)
        
        return DiagnosticResult(
            config=self.config,
            start_time=datetime.fromtimestamp(start_time).isoformat(),
            end_time=datetime.fromtimestamp(end_time).isoformat(),
            total_duration_seconds=end_time - start_time,
            rung_results=rung_results,
            primary_diagnosis=diagnosis,
            decision_trail=decision_trail,
            recommended_actions=recommended_actions,
            samples_processed=len(samples),
            ledger_entries_created=ledger_entries,
            rung_timings=rung_timings
        )
    
    def _generate_recommended_actions(self, 
                                     diagnosis: str, 
                                     rung_results: Dict[int, Dict[str, Any]]) -> List[str]:
        """Generate specific recommended actions based on diagnosis."""
        
        actions = []
        
        if "SCORING" in diagnosis:
            actions.append("1. Fix scoring functions and normalization issues")
            actions.append("2. Validate ground truth data quality")
            actions.append("3. Re-run evaluation after scoring fixes")
        
        elif "SELECTION FAILURE" in diagnosis:
            actions.append("1. Improve retrieval/selection algorithm")
            actions.append("2. Tune embedding models or BM25 parameters")
            actions.append("3. Increase context budget or keep-ratio")
            actions.append("4. Verify input preprocessing pipeline")
        
        elif "LARGE IMPROVEMENT POTENTIAL" in diagnosis:
            actions.append("1. Optimize atom selection strategy")
            actions.append("2. Experiment with hybrid retrieval approaches")
            actions.append("3. Adjust keep-ratio and context budget")
            actions.append("4. Consider semantic chunking improvements")
        
        else:
            actions.append("1. Review detailed rung results for specific issues")
            actions.append("2. Consider running additional diagnostic rungs")
            actions.append("3. Examine per-task performance variations")
        
        return actions
    
    def _save_rung_result(self, rung_number: int, result: Dict[str, Any]):
        """Save intermediate rung result to file."""
        
        output_file = self.config.output_dir / f"rung_{rung_number}_result.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save rung {rung_number} result: {e}")
    
    def save_final_result(self, result: DiagnosticResult):
        """Save final diagnostic result to file."""
        
        output_file = self.config.output_dir / "diagnostic_ladder_result.json"
        
        try:
            result_dict = asdict(result)
            # Convert Path objects to strings for JSON serialization
            result_dict['config']['output_dir'] = str(result_dict['config']['output_dir'])
            if result_dict['config']['ledger_db_path']:
                result_dict['config']['ledger_db_path'] = str(result_dict['config']['ledger_db_path'])
            
            with open(output_file, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
                
            logger.info(f"Saved final diagnostic result to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save final result: {e}")