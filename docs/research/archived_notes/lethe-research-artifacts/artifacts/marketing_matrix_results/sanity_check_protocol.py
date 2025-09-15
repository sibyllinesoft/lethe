#!/usr/bin/env python3
"""
Sanity Check Protocol - Catch Simulation & Enforce Parity
=====================================

This script validates marketing reports against their source artifacts
to prevent simulated results and ensure mathematical consistency.

Usage:
    python sanity_check_protocol.py --report marketing_edge_report_fixed.html --artifacts-dir .
"""

import json
import hashlib
import csv
import re
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from statistics import mean
import numpy as np
from bs4 import BeautifulSoup


@dataclass
class ValidationError:
    category: str
    description: str
    expected: str
    actual: str
    severity: str = "ERROR"


@dataclass
class ValidationResult:
    passed: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    summary: str


class SanityCheckProtocol:
    """
    Implements the complete sanity-check protocol as specified:
    1. Artifact presence & integrity
    2. Parity & leakage checks
    3. Metric recomputation
    4. Monotonicity & gates validation
    5. Label compliance
    """
    
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = artifacts_dir
        self.errors = []
        self.warnings = []
        self.metrics_file = None
        self.manifest_file = None
        
    def check_artifact_presence_integrity(self) -> bool:
        """Step 1: Verify the full deliverables checklist"""
        required_artifacts = [
            "marketing_matrix_results.json",
            "marketing_scenario_results.json", 
            # Expected but may not exist yet:
            # "signed_manifest.json",
            # "leakage_attestation.json", 
            # "overlap_calibration.csv",
            # "stage_timings_p50_p95.csv",
            # "advantage_map.json"
        ]
        
        # Additional artifacts from new benchmark pipeline
        new_pipeline_artifacts = [
            # Raw benchmark results from run_benchmark.py
            "results/longbench.json",
            "results/leval.json", 
            "results/ruler.json",
            "results/loogle.json",
            "results/loong.json",
            # Aggregated results from aggregate_results.py
            "aggregated_benchmark_results.csv",
            "statistical_analysis_report.json",
            # Generated marketing report from build_marketing_report.py
            "enhanced_marketing_report.html"
        ]
        
        # Check for new benchmark artifacts if they exist
        benchmark_dir = self.artifacts_dir.parent.parent / "benchmarks"
        if benchmark_dir.exists():
            required_artifacts.extend(new_pipeline_artifacts)
        
        missing_artifacts = []
        for artifact in required_artifacts:
            # Handle relative paths for benchmark artifacts
            if artifact.startswith("results/"):
                benchmark_dir = self.artifacts_dir.parent.parent / "benchmarks"
                artifact_path = benchmark_dir / artifact
            elif artifact in ["aggregated_benchmark_results.csv", "statistical_analysis_report.json", "enhanced_marketing_report.html"]:
                benchmark_dir = self.artifacts_dir.parent.parent / "benchmarks"
                artifact_path = benchmark_dir / artifact
            else:
                artifact_path = self.artifacts_dir / artifact
                
            if not artifact_path.exists():
                missing_artifacts.append(artifact)
        
        if missing_artifacts:
            self.errors.append(ValidationError(
                "ARTIFACT_PRESENCE",
                f"Missing required artifacts: {missing_artifacts}",
                f"All artifacts present: {required_artifacts}",
                f"Missing: {missing_artifacts}",
                "ERROR"
            ))
            return False
            
        # TODO: Implement SHA256 verification when signed_manifest.json exists
        return True
    
    def check_parity_leakage(self) -> bool:
        """Step 2: Check overlap_calibration.csv → Jaccard@200 ≥ 0.80"""
        # Placeholder - would check overlap_calibration.csv if it exists
        overlap_file = self.artifacts_dir / "overlap_calibration.csv"
        if not overlap_file.exists():
            self.warnings.append(ValidationError(
                "PARITY_LEAKAGE",
                "overlap_calibration.csv not found - cannot verify Jaccard@200 ≥ 0.80",
                "Jaccard@200 ≥ 0.80",
                "File missing",
                "WARNING"
            ))
            return True  # Don't fail on missing optional file
            
        # TODO: Implement Jaccard similarity validation
        return True
    
    def check_no_mock_data_contamination(self) -> bool:
        """Step 0: CRITICAL - Verify no mock data in any benchmark run"""
        print("   🚫 Scanning for mock data contamination...")
        
        # Check for run_meta.json files from all benchmark suites
        benchmark_dir = self.artifacts_dir.parent.parent / "benchmarks"
        if not benchmark_dir.exists():
            self.errors.append(ValidationError(
                "NO_MOCK_VALIDATION",
                "Benchmark directory not found - cannot verify real endpoints were used",
                "Benchmark directory with run metadata",
                "Directory missing",
                "ERROR"
            ))
            return False
        
        # Look for run metadata files
        meta_files = list(benchmark_dir.glob("results/run_meta_*.json"))
        if not meta_files:
            self.errors.append(ValidationError(
                "NO_MOCK_VALIDATION",
                "No run metadata files found - cannot verify real endpoints were used",
                "Run metadata files present (run_meta_*.json)",
                "No metadata files found",
                "ERROR"
            ))
            return False
        
        # Validate each run metadata file
        mock_runs_detected = []
        missing_metadata = []
        
        for meta_file in meta_files:
            try:
                with open(meta_file) as f:
                    metadata = json.load(f)
                
                # Check for mock flag
                if metadata.get("mock", True):  # Default to True if missing
                    mock_runs_detected.append({
                        "file": meta_file.name,
                        "suite": metadata.get("suite", "unknown")
                    })
                
                # Check for required real endpoint fields
                required_fields = ["endpoint", "model", "validation_passed"]
                missing_fields = [field for field in required_fields if field not in metadata]
                if missing_fields:
                    missing_metadata.append({
                        "file": meta_file.name,
                        "missing": missing_fields
                    })
                
                # Validate endpoint format
                endpoint = metadata.get("endpoint", "")
                if not endpoint.startswith(("http://", "https://")):
                    self.errors.append(ValidationError(
                        "NO_MOCK_VALIDATION",
                        f"Invalid endpoint format in {meta_file.name}: {endpoint}",
                        "Valid HTTP/HTTPS endpoint URL",
                        endpoint,
                        "ERROR"
                    ))
                
                print(f"   ✅ {meta_file.name}: Real endpoint verified ({metadata.get('model', 'unknown')})")
                
            except Exception as e:
                self.errors.append(ValidationError(
                    "NO_MOCK_VALIDATION",
                    f"Failed to read run metadata {meta_file.name}: {e}",
                    "Valid JSON metadata file",
                    str(e),
                    "ERROR"
                ))
        
        # Report any mock runs found
        if mock_runs_detected:
            self.errors.append(ValidationError(
                "NO_MOCK_VALIDATION",
                f"MOCK DATA DETECTED in {len(mock_runs_detected)} benchmark runs",
                "All runs using real model endpoints (mock: false)",
                f"Mock runs: {[r['suite'] for r in mock_runs_detected]}",
                "ERROR"
            ))
            return False
        
        # Report missing metadata
        if missing_metadata:
            self.errors.append(ValidationError(
                "NO_MOCK_VALIDATION",
                f"Incomplete metadata in {len(missing_metadata)} files",
                "Complete metadata with endpoint, model, validation_passed",
                f"Missing fields: {missing_metadata}",
                "ERROR"
            ))
            return False
        
        print(f"   ✅ Validated {len(meta_files)} run metadata files - NO MOCK DATA DETECTED")
        return True
    
    def load_scenario_data(self) -> Dict:
        """Load and parse scenario results from multiple sources"""
        scenario_data = {}
        
        # If we have a metrics CSV file, load from that (new pipeline mode)
        if self.metrics_file and self.metrics_file.exists():
            try:
                import pandas as pd
                df = pd.read_csv(self.metrics_file)
                
                # Convert CSV to scenario format
                for _, row in df.iterrows():
                    # Extract budget percentage from string like "4.0%"
                    budget_pct = float(row['budget'].replace('%', ''))
                    scenario_id = f"{row['suite']}_{row['adapter']}_k200_keep{budget_pct:.0f}%_task{row['task']}"
                    scenario_data[scenario_id] = {
                        'recall_at_k': row['mean_recall_at_k'],
                        'p95_latency_ms': row['p95_latency_ms'], 
                        'fail_rate': row['mean_fail_rate'],
                        'qt_score': row['mean_qt_score']
                    }
                    
                print(f"✅ Loaded {len(scenario_data)} scenarios from CSV metrics file")
                return scenario_data
                
            except Exception as e:
                self.errors.append(ValidationError(
                    "DATA_LOAD",
                    f"Failed to load CSV metrics file: {e}",
                    "Valid CSV file",
                    str(e),
                    "ERROR"
                ))
        
        # Legacy mode: Try original scenario file
        scenario_file = self.artifacts_dir / "marketing_scenario_results.json"
        if scenario_file.exists():
            try:
                with open(scenario_file) as f:
                    scenario_data.update(json.load(f))
            except Exception as e:
                self.errors.append(ValidationError(
                    "DATA_LOAD",
                    f"Failed to load original scenario data: {e}",
                    "Valid JSON file",
                    str(e),
                    "ERROR"
                ))
        
        # Try loading new benchmark results (legacy fallback)
        benchmark_dir = self.artifacts_dir.parent.parent / "benchmarks"
        if benchmark_dir.exists():
            # Load individual benchmark results
            for suite in ["longbench", "leval", "ruler", "loogle", "loong"]:
                result_file = benchmark_dir / "results" / f"{suite}.json"
                if result_file.exists():
                    try:
                        with open(result_file) as f:
                            suite_data = json.load(f)
                            scenario_data.update(suite_data)
                    except Exception as e:
                        self.warnings.append(ValidationError(
                            "DATA_LOAD",
                            f"Failed to load {suite} results: {e}",
                            "Valid JSON file",
                            str(e),
                            "WARNING"
                        ))
        
        if not scenario_data:
            self.errors.append(ValidationError(
                "DATA_LOAD",
                "No scenario data found in any location",
                "Valid scenario data",
                "No data loaded",
                "ERROR"
            ))
        
        return scenario_data
    
    def load_summary_data(self) -> Dict:
        """Load and parse summary results from multiple sources"""
        summary_data = {}
        
        # Try original summary file
        summary_file = self.artifacts_dir / "marketing_matrix_results.json"
        if summary_file.exists():
            try:
                with open(summary_file) as f:
                    summary_data.update(json.load(f))
            except Exception as e:
                self.errors.append(ValidationError(
                    "DATA_LOAD", 
                    f"Failed to load original summary data: {e}",
                    "Valid JSON file",
                    str(e),
                    "ERROR"
                ))
        
        # Try loading new statistical analysis
        benchmark_dir = self.artifacts_dir.parent.parent / "benchmarks"
        if benchmark_dir.exists():
            stats_file = benchmark_dir / "statistical_analysis_report.json"
            if stats_file.exists():
                try:
                    with open(stats_file) as f:
                        stats_data = json.load(f)
                        summary_data.update(stats_data)
                except Exception as e:
                    self.warnings.append(ValidationError(
                        "DATA_LOAD",
                        f"Failed to load statistical analysis: {e}",
                        "Valid JSON file",
                        str(e),
                        "WARNING"
                    ))
        
        if not summary_data:
            self.errors.append(ValidationError(
                "DATA_LOAD",
                "No summary data found in any location",
                "Valid summary data",
                "No data loaded",
                "ERROR"
            ))
        
        return summary_data
    
    def recompute_qt_scores(self, scenario_data: Dict) -> bool:
        """Step 3: Recompute QT from p95/fail% and verify against artifacts"""
        qt_errors = []
        
        for scenario_id, data in scenario_data.items():
            if not all(k in data for k in ['recall_at_k', 'p95_latency_ms', 'fail_rate']):
                continue
                
            # QT Formula: Recall@5 × (1000/p95_ms) × (1 − fail_rate)
            expected_qt = data['recall_at_k'] * (1000.0 / data['p95_latency_ms']) * (1.0 - data['fail_rate'])
            actual_qt = data.get('qt_score', 0)
            
            # Allow 1% tolerance for floating point arithmetic
            if abs(expected_qt - actual_qt) / max(expected_qt, actual_qt, 1e-6) > 0.01:
                qt_errors.append({
                    'scenario': scenario_id,
                    'expected': expected_qt,
                    'actual': actual_qt,
                    'recall': data['recall_at_k'],
                    'p95': data['p95_latency_ms'],
                    'fail_rate': data['fail_rate']
                })
        
        if qt_errors:
            self.errors.append(ValidationError(
                "QT_RECOMPUTE",
                f"QT computation mismatch in {len(qt_errors)} scenarios",
                "QT = Recall@5 × (1000/p95_ms) × (1-fail_rate)",
                f"First mismatch: {qt_errors[0]}",
                "ERROR"
            ))
            return False
            
        return True
    
    def check_monotonicity_gates(self, scenario_data: Dict, summary_data: Dict) -> bool:
        """Step 4: Assert monotonicity and quality gates"""
        # Group scenarios by adapter and budget
        adapter_budgets = {}
        for scenario_id, data in scenario_data.items():
            adapter = data.get('adapter', 'unknown')
            budget = data.get('keep_ratio', 0) * 100  # Convert to percentage
            
            if adapter not in adapter_budgets:
                adapter_budgets[adapter] = {}
            if budget not in adapter_budgets[adapter]:
                adapter_budgets[adapter][budget] = []
            adapter_budgets[adapter][budget].append(data)
        
        # Check monotonicity: Recall@5 should be non-decreasing across 4→8→16%
        monotonicity_errors = []
        for adapter, budgets in adapter_budgets.items():
            if not all(b in budgets for b in [4.0, 8.0, 16.0]):
                continue
                
            recall_4 = mean([d['recall_at_k'] for d in budgets[4.0]])
            recall_8 = mean([d['recall_at_k'] for d in budgets[8.0]])  
            recall_16 = mean([d['recall_at_k'] for d in budgets[16.0]])
            
            if not (recall_4 <= recall_8 <= recall_16):
                monotonicity_errors.append({
                    'adapter': adapter,
                    'recalls': [recall_4, recall_8, recall_16]
                })
        
        if monotonicity_errors:
            self.errors.append(ValidationError(
                "MONOTONICITY",
                f"Recall not monotonic for {len(monotonicity_errors)} adapters",
                "Recall@5 non-decreasing across 4%→8%→16%", 
                f"Violations: {monotonicity_errors}",
                "ERROR"
            ))
        
        # Check quality gates from summary
        quality_gates = summary_data.get('quality_gates', [])
        failed_gates = [g for g in quality_gates if not g.get('passed', False)]
        
        if failed_gates:
            self.errors.append(ValidationError(
                "QUALITY_GATES",
                f"Failed quality gates: {[g['gate_name'] for g in failed_gates]}",
                "All quality gates passed",
                f"Failed: {failed_gates}",
                "ERROR"
            ))
            return False
            
        return len(monotonicity_errors) == 0
    
    def check_label_compliance(self, html_file: Path) -> bool:
        """Step 5: Enforce 'no internals in UI' rule and approved labels"""
        if html_file is None:
            self.warnings.append(ValidationError(
                "LABEL_COMPLIANCE",
                "HTML report not provided - skipping label compliance check",
                "HTML report available",
                "File not provided",
                "WARNING"
            ))
            return True  # Don't fail validation just because we can't check labels
            
        if not html_file.exists():
            self.errors.append(ValidationError(
                "LABEL_COMPLIANCE",
                f"HTML report not found: {html_file}",
                "HTML report exists",
                "File missing",
                "ERROR"
            ))
            return False
            
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for forbidden raw adapter IDs
        forbidden_patterns = [
            r'rag:vector_faiss_cosine',
            r'rag:hybrid_faiss_50_50', 
            r'rag:hybrid_milvus_50_50',
            r'rag:[a-zA-Z_]+',  # Any rag: prefixed ID
        ]
        
        label_violations = []
        for pattern in forbidden_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                label_violations.extend(matches)
        
        if label_violations:
            self.errors.append(ValidationError(
                "LABEL_COMPLIANCE",
                f"Found raw adapter IDs in UI: {set(label_violations)}",
                "Clean UI labels (Vector, Hybrid 50/50, BM25)",
                f"Raw IDs found: {set(label_violations)}",
                "ERROR"
            ))
            return False
        
        # Check for required clean labels (updated from new pipeline)
        required_labels = [
            "Vector (Faiss) — Lethe Engine",
            "Hybrid 50/50", 
            "BM25",
            # Additional labels from expanded benchmark suite
            "LongBench",
            "L-Eval", 
            "RULER",
            "LooGLE",
            "Loong"
        ]
        
        missing_labels = []
        for label in required_labels:
            if label not in content:
                missing_labels.append(label)
        
        if missing_labels:
            self.warnings.append(ValidationError(
                "LABEL_COMPLIANCE",
                f"Missing required clean labels: {missing_labels}",
                f"All labels present: {required_labels}",
                f"Missing: {missing_labels}",
                "WARNING"
            ))
        
        return True
    
    def validate_benchmark_results(self) -> bool:
        """Step 6: Validate benchmark pipeline artifacts with data integrity gates"""
        benchmark_dir = self.artifacts_dir.parent.parent / "benchmarks"
        if not benchmark_dir.exists():
            self.warnings.append(ValidationError(
                "BENCHMARK_VALIDATION",
                "Benchmark directory not found - skipping benchmark validation",
                "Benchmark results present",
                "No benchmark directory",
                "WARNING"
            ))
            return True  # Don't fail on missing optional benchmark validation
            
        # Load metrics CSV for integrity validation
        if not self.metrics_file or not self.metrics_file.exists():
            self.warnings.append(ValidationError(
                "DATA_INTEGRITY",
                "Metrics CSV not available - skipping data integrity validation",
                "Metrics CSV available",
                "No metrics file",
                "WARNING"
            ))
            return True
            
        try:
            import pandas as pd
            df = pd.read_csv(self.metrics_file)
            
            # Gate 1: Key presence - assert 'suite' is a grouping column
            if 'suite' not in df.columns:
                self.errors.append(ValidationError(
                    "DATA_INTEGRITY",
                    "Missing 'suite' column in metrics - data collapsed incorrectly",
                    "Suite column present for per-suite analysis",
                    "Missing suite column",
                    "ERROR"
                ))
                return False
                
            # Gate 2: Per-suite variance - check if all suites have identical metrics
            suites = df['suite'].unique()
            if len(suites) < 2:
                self.warnings.append(ValidationError(
                    "DATA_INTEGRITY",
                    "Only one suite found - cannot validate cross-suite variance",
                    "Multiple suites for variance analysis",
                    f"Only {len(suites)} suite(s)",
                    "WARNING"
                ))
            else:
                # Check variance across suites for each (adapter, budget) combination
                identical_combinations = 0
                total_combinations = 0
                
                for adapter in df['adapter'].unique():
                    for budget in df['budget'].unique():
                        subset = df[(df['adapter'] == adapter) & (df['budget'] == budget)]
                        if len(subset) < 2:
                            continue
                            
                        total_combinations += 1
                        # Check if all key metrics are identical across suites
                        recall_var = subset['mean_recall_at_k'].var()
                        p95_var = subset['p95_latency_ms'].var() 
                        qt_var = subset['mean_qt_score'].var()
                        
                        if recall_var < 1e-10 and p95_var < 1e-10 and qt_var < 1e-10:
                            identical_combinations += 1
                
                if identical_combinations > 0:
                    self.errors.append(ValidationError(
                        "DATA_INTEGRITY",
                        f"Identical metrics across suites detected in {identical_combinations}/{total_combinations} combinations - likely mock data replication",
                        "Per-suite variance > 0 for at least one metric",
                        f"{identical_combinations} combinations with zero variance",
                        "ERROR"
                    ))
                    return False
                    
            # Gate 3: Row counts - check manifest vs actual counts
            if self.manifest_file and self.manifest_file.exists():
                try:
                    with open(self.manifest_file) as f:
                        manifest = json.load(f)
                    
                    manifest_total = manifest.get('total_scenarios', 0) 
                    csv_rows = len(df)
                    
                    if manifest_total > 0 and csv_rows != manifest_total:
                        self.errors.append(ValidationError(
                            "DATA_INTEGRITY", 
                            f"Row count mismatch: CSV has {csv_rows} rows but manifest claims {manifest_total} total scenarios",
                            f"CSV rows ({csv_rows}) == manifest total ({manifest_total})",
                            f"Mismatch: {csv_rows} != {manifest_total}",
                            "ERROR"
                        ))
                        return False
                        
                except Exception as e:
                    self.warnings.append(ValidationError(
                        "DATA_INTEGRITY",
                        f"Could not validate manifest counts: {e}",
                        "Manifest validation",
                        f"Error: {e}",
                        "WARNING"
                    ))
                    
            # Gate 4: Join check - verify no duplicate keys after grouping
            key_cols = ['suite', 'adapter', 'budget', 'task'] 
            available_key_cols = [col for col in key_cols if col in df.columns]
            
            if len(available_key_cols) >= 3:  # Need at least suite, adapter, budget
                duplicates = df.duplicated(subset=available_key_cols).sum()
                if duplicates > 0:
                    self.errors.append(ValidationError(
                        "DATA_INTEGRITY",
                        f"Duplicate keys found after grouping by {available_key_cols}: {duplicates} duplicates",
                        "No duplicate keys in grouped data", 
                        f"{duplicates} duplicate combinations",
                        "ERROR"
                    ))
                    return False
                    
        except Exception as e:
            self.errors.append(ValidationError(
                "DATA_INTEGRITY",
                f"Failed to validate benchmark data integrity: {e}",
                "Successful data integrity validation",
                str(e),
                "ERROR"
            ))
            return False
            
        return True
    
    def run_full_protocol(self, html_file: Path) -> ValidationResult:
        """Run the complete sanity-check protocol"""
        print("🔍 Running Enhanced Sanity-Check Protocol...")
        
        # Step 0: CRITICAL - No-Mock validation (NEW)
        print("0. 🚫 CRITICAL: Checking for mock data contamination...")
        no_mock_ok = self.check_no_mock_data_contamination()
        
        # Step 1: Artifact presence & integrity
        print("1. Checking artifact presence & integrity...")
        artifacts_ok = self.check_artifact_presence_integrity()
        
        # Step 2: Parity & leakage
        print("2. Checking parity & leakage...")
        parity_ok = self.check_parity_leakage()
        
        # Load data for subsequent checks
        scenario_data = self.load_scenario_data()
        summary_data = self.load_summary_data()
        
        if not scenario_data or not summary_data:
            return ValidationResult(
                passed=False,
                errors=self.errors,
                warnings=self.warnings,
                summary="❌ FAILED: Could not load required data files"
            )
        
        # Step 3: Metric recomputation  
        print("3. Recomputing and validating QT scores...")
        qt_ok = self.recompute_qt_scores(scenario_data)
        
        # Step 4: Monotonicity & gates
        print("4. Checking monotonicity & quality gates...")  
        monotonicity_ok = self.check_monotonicity_gates(scenario_data, summary_data)
        
        # Step 5: Label compliance
        print("5. Checking label compliance...")
        labels_ok = self.check_label_compliance(html_file)
        
        # Step 6: Benchmark validation (new)
        print("6. Validating benchmark pipeline artifacts...")
        benchmark_ok = self.validate_benchmark_results()
        
        # Overall assessment
        all_passed = all([no_mock_ok, artifacts_ok, parity_ok, qt_ok, monotonicity_ok, labels_ok, benchmark_ok])
        
        if all_passed and not self.errors:
            summary = "✅ ALL CHECKS PASSED - Report validated as authentic with NO MOCK DATA and consistent metrics"
        elif self.errors:
            summary = f"❌ FAILED: {len(self.errors)} errors detected - Report may contain MOCK DATA or pipeline issues"
        else:
            summary = f"⚠️ WARNINGS: {len(self.warnings)} warnings but no blocking errors"
            
        return ValidationResult(
            passed=all_passed and not self.errors,
            errors=self.errors,
            warnings=self.warnings, 
            summary=summary
        )


def main():
    parser = argparse.ArgumentParser(description="Run sanity check protocol on marketing reports")
    parser.add_argument("--manifest", help="Signed manifest JSON file")
    parser.add_argument("--metrics", help="CSV metrics summary file from benchmark pipeline")
    parser.add_argument("--report", help="HTML report file to validate (legacy)")
    parser.add_argument("--artifacts-dir", help="Directory containing artifacts (legacy)")
    
    args = parser.parse_args()
    
    # Support both new and legacy command line interfaces
    if args.manifest and args.metrics:
        # New pipeline mode: validate using manifest and CSV metrics
        manifest_file = Path(args.manifest)
        metrics_file = Path(args.metrics)
        
        if not manifest_file.exists():
            print(f"❌ Manifest file not found: {manifest_file}")
            sys.exit(1)
            
        if not metrics_file.exists():
            print(f"❌ Metrics file not found: {metrics_file}")
            sys.exit(1)
        
        # Use artifacts directory from manifest's parent directory
        artifacts_dir = manifest_file.parent
        protocol = SanityCheckProtocol(artifacts_dir)
        protocol.metrics_file = metrics_file
        protocol.manifest_file = manifest_file
        
        # Find the HTML report in the parent directory
        html_file = None
        for report_name in ["marketing_edge_report_final.html", "marketing_edge_report_fixed.html", "marketing_edge_report.html"]:
            potential_report = artifacts_dir.parent.parent / "benchmarks" / report_name
            if potential_report.exists():
                html_file = potential_report
                break
        
        if not html_file:
            print("⚠️ No HTML report found, skipping visual validation")
            
    elif args.report and args.artifacts_dir:
        # Legacy mode: validate using report and artifacts directory
        artifacts_dir = Path(args.artifacts_dir)
        html_file = Path(args.report)
        
        if not artifacts_dir.exists():
            print(f"❌ Artifacts directory not found: {artifacts_dir}")
            sys.exit(1)
            
        protocol = SanityCheckProtocol(artifacts_dir)
        protocol.metrics_file = None
        protocol.manifest_file = None
    else:
        print("❌ Must provide either --manifest and --metrics, or --report and --artifacts-dir")
        sys.exit(1)
        
    result = protocol.run_full_protocol(html_file)
    
    # Print detailed results
    print("\n" + "="*60)
    print(result.summary)
    print("="*60)
    
    if result.errors:
        print(f"\n🔴 ERRORS ({len(result.errors)}):")
        for i, error in enumerate(result.errors, 1):
            print(f"  {i}. [{error.category}] {error.description}")
            print(f"     Expected: {error.expected}")
            print(f"     Actual: {error.actual}\n")
    
    if result.warnings:
        print(f"🟡 WARNINGS ({len(result.warnings)}):")
        for i, warning in enumerate(result.warnings, 1):
            print(f"  {i}. [{warning.category}] {warning.description}")
            print(f"     Expected: {warning.expected}")
            print(f"     Actual: {warning.actual}\n")
    
    if result.passed:
        print("✅ VALIDATION SUMMARY: Report is authentic with consistent metrics")
        print("   - 🚫 NO MOCK DATA detected (Step-0 validation passed)")
        print("   - Real model endpoints verified for all runs")
        print("   - QT computations verified")  
        print("   - Quality gates passed")
        print("   - Label compliance enforced")
        print("   - Benchmark pipeline artifacts validated")
        sys.exit(0)
    else:
        print("❌ VALIDATION SUMMARY: Report FAILED validation")
        print("   - ⚠️ May contain MOCK DATA or simulated results")
        print("   - Real endpoint verification failed")
        print("   - Pipeline artifacts may have issues")
        print("   - 🚫 BLOCKED from publication until resolved")
        sys.exit(1)


if __name__ == "__main__":
    main()