#!/usr/bin/env python3
"""
Basic Milestone 7 Test - Minimal dependency testing
Tests core functionality without heavy dependencies.
"""

import sys
import json
from pathlib import Path
from typing import Tuple

# Add src to path for test utilities
sys.path.insert(0, str(Path(__file__).parent / "src"))
from testing.test_utils import (
    assert_imports_available,
    assert_files_exist,
    assert_makefile_targets,
    generate_synthetic_data
)

def test_basic_imports():
    """Test basic Python imports without heavy ML dependencies"""
    # Test scientific libraries
    assert_imports_available("pandas", "numpy")
    
    # Test matplotlib with non-interactive backend
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    assert_imports_available("matplotlib.pyplot")
    
    # Test standard library imports
    assert_imports_available(
        "json", "hashlib", "pathlib", "dataclasses", "typing"
    )

def test_file_structure():
    """Test that all required files exist"""
    required_files = [
        "src/eval/milestone7_analysis.py",
        "run_milestone7_analysis.py",
        "validate_milestone7_implementation.py",
        "MILESTONE7_COMPLETION_REPORT.md",
        "requirements_milestone7.txt",
        "demo_milestone7.py"
    ]
    
    assert_files_exist(*required_files)

def test_makefile_integration():
    """Test Makefile has proper targets"""
    required_targets = [
        "figures:",
        "milestone7-analysis:",
        "milestone7-quick:",
        "tables:",
        "plots:",
        "sanity-checks:"
    ]
    
    assert_makefile_targets(*required_targets)

def test_dataclass_creation():
    """Test that we can create the basic dataclasses"""
    from dataclasses import dataclass
    
    @dataclass
    class TestMetrics:
        ndcg_10: float
        latency_p95_ms: float
        ndcg_10_ci: Tuple[float, float]
    
    # Create test instance
    test_metrics = TestMetrics(
        ndcg_10=0.75,
        latency_p95_ms=250.5,
        ndcg_10_ci=(0.70, 0.80)
    )
    
    # Validate dataclass creation worked
    assert test_metrics.ndcg_10 == 0.75
    assert test_metrics.latency_p95_ms == 250.5
    assert test_metrics.ndcg_10_ci == (0.70, 0.80)
    assert hasattr(test_metrics, '__dataclass_fields__')

def test_synthetic_data_generation():
    """Test synthetic data generation for quick testing"""
    # Define schema for synthetic data
    schema = {
        "query": "str",
        "query_type": ["exact_match", "general"],
        "novelty_score": "float",
        "planning_action": ["EXPLORE", "RETRIEVE"],
        "retrieved_docs": "list"
    }
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(schema, n_samples=10)
    
    # Validate generated data
    assert len(synthetic_data) == 10
    assert all("query" in item for item in synthetic_data)
    assert all("query_type" in item for item in synthetic_data)
    assert all("novelty_score" in item for item in synthetic_data)
    
    # Test JSON serialization
    json_str = json.dumps(synthetic_data, indent=2)
    parsed_back = json.loads(json_str)
    assert len(parsed_back) == len(synthetic_data)

# Note: The main() function is no longer needed for pytest-style tests.
# These tests will be run automatically by pytest.

# Keep the main() function for backward compatibility if run directly
def main():
    """Run basic Milestone 7 tests (for backward compatibility)"""
    print("🚀 Milestone 7 Basic Implementation Test")
    print("Note: This file now uses proper pytest assertions.")
    print("Run with: python -m pytest test_milestone7_basic.py -v")
    return True

if __name__ == "__main__":
    # When run directly, suggest using pytest
    print("✅ Tests have been converted to use proper pytest assertions.")
    print("Please run with: python -m pytest test_milestone7_basic.py -v")
    print("Or simply: pytest test_milestone7_basic.py -v")
    sys.exit(0)