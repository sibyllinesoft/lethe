#!/usr/bin/env python3
"""
Comprehensive test suite for HTML validator report generator.
Enforces all critical requirements to prevent regressions.
"""

import pytest
import pandas as pd
import json
import hashlib
import os
import tempfile
from pathlib import Path
from bs4 import BeautifulSoup
from enhanced_html_generator import generate_enhanced_validator_report, RECALL_METRIC, LETHE_ENGINE_ADAPTER_ID


class TestValidatorReportCriticalRequirements:
    """Test suite for critical validator report requirements"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create sample CSV data with required columns
        self.sample_csv_data = [
            {
                'adapter': LETHE_ENGINE_ADAPTER_ID,
                'dataset': 'Conv-Set-A', 
                'k_value': 5.0,
                'keep_percentage': 0.08,
                'metric': 'score',
                'mean': 0.82
            },
            {
                'adapter': 'selector:random_within_type',
                'dataset': 'Conv-Set-A',
                'k_value': 5.0, 
                'keep_percentage': 0.15,
                'metric': 'score',
                'mean': 0.65
            },
            {
                'adapter': LETHE_ENGINE_ADAPTER_ID,
                'dataset': 'Conv-Set-A',
                'k_value': 5.0,
                'keep_percentage': 0.30,
                'metric': 'score', 
                'mean': 0.89
            }
        ]
        
        # Create test files
        self.csv_path = self.temp_path / "metrics.csv"
        self.advantage_map_path = self.temp_path / "advantage_map.json"
        self.manifest_path = self.temp_path / "signed_manifest.json"
        self.leakage_path = self.temp_path / "leakage_attestation.json"
        self.output_path = self.temp_path / "test_report.html"
        
        # Create CSV
        pd.DataFrame(self.sample_csv_data).to_csv(self.csv_path, index=False)
        
        # Create advantage map
        advantage_map = {
            "advantage_matrix": {
                LETHE_ENGINE_ADAPTER_ID: {"some_dataset": 0.15},
                "selector:random_within_type": {"some_dataset": -0.05}
            }
        }
        with open(self.advantage_map_path, 'w') as f:
            json.dump(advantage_map, f)
        
        # Create manifest
        manifest = {
            "run_id": "test_run_123",
            "generator": "test_generator",
            "timestamp": "2024-01-01T12:00:00Z"
        }
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f)
            
        # Create leakage attestation
        with open(self.leakage_path, 'w') as f:
            json.dump({"status": "clean"}, f)

    def test_recall_metric_assertion(self):
        """HARD TEST: Assert RECALL_METRIC == 'score'"""
        assert RECALL_METRIC == "score", f"RECALL_METRIC must be 'score', got '{RECALL_METRIC}'"
    
    def test_lethe_engine_adapter_id_in_csv(self):
        """HARD TEST: Assert LETHE_ENGINE_ADAPTER_ID exists in CSV data"""
        df = pd.read_csv(self.csv_path)
        adapters_in_csv = set(df['adapter'].unique())
        assert LETHE_ENGINE_ADAPTER_ID in adapters_in_csv, \
            f"LETHE_ENGINE_ADAPTER_ID '{LETHE_ENGINE_ADAPTER_ID}' not found in CSV adapters: {adapters_in_csv}"
    
    def test_all_budget_charts_present(self):
        """HARD TEST: Assert charts include all 3 budgets (8%, 15%, 30%)"""
        # Generate report
        output_file, validation_results = generate_enhanced_validator_report(
            str(self.csv_path),
            str(self.advantage_map_path), 
            str(self.manifest_path),
            str(self.leakage_path),
            str(self.output_path)
        )
        
        # Parse HTML
        with open(self.output_path, 'r') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all chart titles
        chart_titles = [h3.get_text() for h3 in soup.find_all('h3')]
        
        # Check for all required budgets
        required_budgets = ["8%", "15%", "30%"]
        for budget in required_budgets:
            budget_found = any(budget in title for title in chart_titles)
            assert budget_found, f"Budget '{budget}' chart not found in HTML. Charts: {chart_titles}"
    
    def test_lethe_engine_label_present(self):
        """HARD TEST: Assert 'Lethe Engine' label is present in HTML"""
        # Generate report
        output_file, validation_results = generate_enhanced_validator_report(
            str(self.csv_path),
            str(self.advantage_map_path),
            str(self.manifest_path), 
            str(self.leakage_path),
            str(self.output_path)
        )
        
        # Check HTML content
        with open(self.output_path, 'r') as f:
            html_content = f.read()
        
        assert "Lethe Engine" in html_content, "HTML must contain 'Lethe Engine' label"
        
        # Also check in parsed structure
        soup = BeautifulSoup(html_content, 'html.parser')
        lethe_elements = soup.find_all(string=lambda text: text and "Lethe Engine" in text)
        assert len(lethe_elements) > 0, "No 'Lethe Engine' text elements found in HTML structure"
    
    def test_manifest_sha_in_html_matches_file(self):
        """HARD TEST: Assert manifest SHA in HTML matches actual file"""
        # Calculate expected SHA
        with open(self.manifest_path, 'rb') as f:
            expected_sha = hashlib.sha256(f.read()).hexdigest()[:16]
        
        # Generate report
        output_file, validation_results = generate_enhanced_validator_report(
            str(self.csv_path),
            str(self.advantage_map_path),
            str(self.manifest_path),
            str(self.leakage_path), 
            str(self.output_path)
        )
        
        # Check HTML content
        with open(self.output_path, 'r') as f:
            html_content = f.read()
        
        assert expected_sha in html_content, \
            f"Expected manifest SHA '{expected_sha}' not found in HTML"
    
    def test_end_to_end_report_generation(self):
        """INTEGRATION TEST: Full end-to-end report generation"""
        # This should not raise any exceptions
        output_file, validation_results = generate_enhanced_validator_report(
            str(self.csv_path),
            str(self.advantage_map_path), 
            str(self.manifest_path),
            str(self.leakage_path),
            str(self.output_path)
        )
        
        # Verify file was created
        assert os.path.exists(output_file), f"Output file {output_file} was not created"
        
        # Verify it's not empty
        assert os.path.getsize(output_file) > 1000, "Generated HTML file is suspiciously small"
        
        # Verify it's valid HTML
        with open(output_file, 'r') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        assert soup.find('html') is not None, "Generated file is not valid HTML"
        assert soup.find('title') is not None, "HTML missing title tag"
        
        # Verify validation results
        assert validation_results['validation_passed'], f"Validation failed: {validation_results}"

    def test_quality_gates_section_present(self):
        """TEST: Verify quality gates section is present"""
        output_file, validation_results = generate_enhanced_validator_report(
            str(self.csv_path),
            str(self.advantage_map_path),
            str(self.manifest_path),
            str(self.leakage_path),
            str(self.output_path)
        )
        
        with open(self.output_path, 'r') as f:
            html_content = f.read()
        
        assert "Quality Gates" in html_content, "HTML must contain Quality Gates section"
        assert "Placebo baseline" in html_content, "Quality gates must include placebo baseline check"
        assert "Budget monotonicity" in html_content, "Quality gates must include budget monotonicity check"

    def test_methodology_section_completeness(self):
        """TEST: Verify methodology section includes required elements"""
        output_file, validation_results = generate_enhanced_validator_report(
            str(self.csv_path), 
            str(self.advantage_map_path),
            str(self.manifest_path),
            str(self.leakage_path),
            str(self.output_path)
        )
        
        with open(self.output_path, 'r') as f:
            html_content = f.read()
        
        # Required methodology elements
        required_elements = [
            "Methodology",
            "Baseline:",
            "Metric:",
            "Statistical Framework:",
            "Sample Size:"
        ]
        
        for element in required_elements:
            assert element in html_content, f"Methodology section missing required element: {element}"

    def test_provenance_banner_completeness(self):
        """TEST: Verify provenance banner includes all metadata"""
        output_file, validation_results = generate_enhanced_validator_report(
            str(self.csv_path),
            str(self.advantage_map_path), 
            str(self.manifest_path),
            str(self.leakage_path),
            str(self.output_path)
        )
        
        with open(self.output_path, 'r') as f:
            html_content = f.read()
        
        # Required provenance elements
        provenance_elements = [
            "Powered by Lethe",
            "commit=",
            "manifest_sha=", 
            "run_id=test_run_123",
            "generator=test_generator",
            "datasets=",
            "leakage_attestation=✓"
        ]
        
        for element in provenance_elements:
            assert element in html_content, f"Provenance banner missing required element: {element}"


def validate_report_requirements(html_path, csv_path, manifest_path):
    """
    Standalone validation function that can be called from generate_enhanced_validator_report()
    Returns dict with validation results
    """
    results = {
        'recall_metric_correct': RECALL_METRIC == "score",
        'lethe_engine_in_csv': False,
        'all_budgets_present': False,
        'lethe_engine_label_present': False,
        'manifest_sha_matches': False,
        'html_file_valid': False
    }
    
    try:
        # Check CSV
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            results['lethe_engine_in_csv'] = LETHE_ENGINE_ADAPTER_ID in df['adapter'].unique()
        
        # Check HTML file  
        if os.path.exists(html_path):
            results['html_file_valid'] = os.path.getsize(html_path) > 1000
            
            with open(html_path, 'r') as f:
                html_content = f.read()
            
            # Check budgets
            budgets_found = [budget for budget in ["8%", "15%", "30%"] 
                           if f"{budget} budget" in html_content]
            results['all_budgets_present'] = len(budgets_found) == 3
            
            # Check Lethe Engine label
            results['lethe_engine_label_present'] = "Lethe Engine" in html_content
            
            # Check manifest SHA
            if os.path.exists(manifest_path):
                with open(manifest_path, 'rb') as f:
                    expected_sha = hashlib.sha256(f.read()).hexdigest()[:16]
                results['manifest_sha_matches'] = expected_sha in html_content
        
    except Exception as e:
        results['validation_error'] = str(e)
    
    return results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])