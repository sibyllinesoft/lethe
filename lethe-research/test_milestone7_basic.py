#!/usr/bin/env python3
"""
Basic Milestone 7 Test - Minimal dependency testing
Tests core functionality without heavy dependencies.
"""

import sys
import json
from pathlib import Path

def test_basic_imports():
    """Test basic Python imports without heavy ML dependencies"""
    
    try:
        import pandas as pd
        import numpy as np
        print("✅ pandas and numpy imported successfully")
        
        # Test basic matplotlib without display
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        print("✅ matplotlib imported successfully")
        
        import json
        import hashlib
        from pathlib import Path
        from dataclasses import dataclass, asdict
        from typing import Dict, List, Tuple, Optional, Any
        print("✅ Standard library imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic import failed: {str(e)}")
        return False

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
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def test_makefile_integration():
    """Test Makefile has proper targets"""
    
    try:
        with open("Makefile") as f:
            content = f.read()
        
        required_targets = [
            "figures:",
            "milestone7-analysis:",
            "milestone7-quick:",
            "tables:",
            "plots:",
            "sanity-checks:"
        ]
        
        missing = []
        for target in required_targets:
            if target not in content:
                missing.append(target)
        
        if missing:
            print(f"❌ Missing Makefile targets: {missing}")
            return False
        else:
            print("✅ All Makefile targets present")
            return True
            
    except Exception as e:
        print(f"❌ Makefile test failed: {str(e)}")
        return False

def test_dataclass_creation():
    """Test that we can create the basic dataclasses"""
    
    try:
        from dataclasses import dataclass
        from typing import Tuple
        
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
        
        print("✅ Dataclass creation successful")
        print(f"   Sample metric: nDCG@10 = {test_metrics.ndcg_10}")
        return True
        
    except Exception as e:
        print(f"❌ Dataclass test failed: {str(e)}")
        return False

def test_synthetic_data_generation():
    """Test synthetic data generation for quick testing"""
    
    try:
        # Create minimal synthetic dataset
        synthetic_data = []
        for i in range(10):
            item = {
                "query": f"test query {i}",
                "query_type": "exact_match" if i % 3 == 0 else "general",
                "novelty_score": 0.3 + (i % 5) * 0.15,
                "planning_action": "EXPLORE" if i % 4 == 0 else "RETRIEVE",
                "retrieved_docs": [
                    {"relevance_score": 0.9 if i % 3 == 0 else 0.6}
                ]
            }
            synthetic_data.append(item)
        
        # Test JSON serialization
        json_str = json.dumps(synthetic_data, indent=2)
        parsed_back = json.loads(json_str)
        
        print("✅ Synthetic data generation successful")
        print(f"   Generated {len(synthetic_data)} test items")
        return True
        
    except Exception as e:
        print(f"❌ Synthetic data test failed: {str(e)}")
        return False

def main():
    """Run basic Milestone 7 tests"""
    
    print("🚀 Milestone 7 Basic Implementation Test")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Basic Imports", test_basic_imports),
        ("Makefile Integration", test_makefile_integration), 
        ("Dataclass Creation", test_dataclass_creation),
        ("Synthetic Data Generation", test_synthetic_data_generation)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {str(e)}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All basic tests passed!")
        print("\nNext steps:")
        print("1. Install requirements: pip install -r requirements_milestone7.txt")
        print("2. Run: make milestone7-quick")
        print("3. Check: make analysis-summary")
        return True
    else:
        print(f"\n❌ {total - passed} tests failed")
        print("Please fix issues before proceeding")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)