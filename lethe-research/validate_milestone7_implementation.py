#!/usr/bin/env python3
"""
Milestone 7 Implementation Validation
Comprehensive testing of publication-ready analysis pipeline.
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

def validate_file_structure() -> Tuple[bool, List[str]]:
    """Validate all required files are present"""
    
    required_files = [
        "src/eval/milestone7_analysis.py",
        "run_milestone7_analysis.py",
        "Makefile"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    success = len(missing_files) == 0
    messages = [f"✅ All required files present"] if success else [f"❌ Missing files: {missing_files}"]
    
    return success, messages

def validate_imports() -> Tuple[bool, List[str]]:
    """Test that all imports work correctly"""
    
    messages = []
    
    try:
        # Test main analysis pipeline imports
        sys.path.insert(0, "src")
        from src.eval.milestone7_analysis import (
            PublicationMetrics,
            PublicationTableGenerator,
            PublicationPlotGenerator, 
            SanityCheckValidator,
            HardwareProfileManager,
            Milestone7AnalysisPipeline
        )
        messages.append("✅ All core analysis classes imported successfully")
        
        # Test required dependencies
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats
        import hashlib
        import psutil
        messages.append("✅ All required dependencies available")
        
        return True, messages
        
    except ImportError as e:
        return False, [f"❌ Import error: {str(e)}"]
    except Exception as e:
        return False, [f"❌ Unexpected error: {str(e)}"]

def validate_makefile_targets() -> Tuple[bool, List[str]]:
    """Validate Makefile targets are properly defined"""
    
    messages = []
    
    try:
        with open("Makefile") as f:
            makefile_content = f.read()
        
        # Check for required targets
        required_targets = [
            "figures:",
            "milestone7-analysis:",
            "milestone7-quick:",
            "tables:",
            "plots:",
            "sanity-checks:",
            "clean-analysis:",
            "analysis-summary:"
        ]
        
        missing_targets = []
        for target in required_targets:
            if target not in makefile_content:
                missing_targets.append(target)
        
        if missing_targets:
            return False, [f"❌ Missing Makefile targets: {missing_targets}"]
        
        messages.append("✅ All required Makefile targets present")
        
        # Check for proper milestone 7 integration
        if "MILESTONE 7: PUBLICATION-READY ANALYSIS PIPELINE" in makefile_content:
            messages.append("✅ Milestone 7 section properly integrated")
        else:
            messages.append("⚠️  Milestone 7 section header not found")
        
        return True, messages
        
    except Exception as e:
        return False, [f"❌ Error reading Makefile: {str(e)}"]

def test_quick_analysis_run() -> Tuple[bool, List[str]]:
    """Test the quick analysis run with synthetic data"""
    
    messages = []
    
    try:
        # Run quick test
        python_cmd = "python3" if subprocess.run(["which", "python3"], capture_output=True).returncode == 0 else "python"
        result = subprocess.run([
            python_cmd, "run_milestone7_analysis.py",
            "--quick-test",
            "--output-dir", "./test_analysis_output"
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            messages.append("✅ Quick analysis run completed successfully")
            
            # Check output structure
            output_dir = Path("./test_analysis_output")
            if output_dir.exists():
                messages.append("✅ Output directory created")
                
                # Check for expected subdirectories
                expected_dirs = ["hardware_profiles"]
                for expected_dir in expected_dirs:
                    dir_path = output_dir / expected_dir
                    if dir_path.exists():
                        messages.append(f"✅ {expected_dir}/ directory created")
                    else:
                        messages.append(f"⚠️  {expected_dir}/ directory not found")
                
                # Check for generated files
                generated_files = list(output_dir.rglob("*"))
                file_count = len([f for f in generated_files if f.is_file()])
                messages.append(f"📁 Generated {file_count} files in output directory")
                
            else:
                messages.append("❌ Output directory not created")
                return False, messages
            
            return True, messages
        else:
            messages.append(f"❌ Quick analysis run failed with return code {result.returncode}")
            messages.append(f"STDOUT: {result.stdout}")
            messages.append(f"STDERR: {result.stderr}")
            return False, messages
            
    except subprocess.TimeoutExpired:
        return False, ["❌ Quick analysis run timed out (>5 minutes)"]
    except Exception as e:
        return False, [f"❌ Error running quick analysis: {str(e)}"]

def validate_class_functionality() -> Tuple[bool, List[str]]:
    """Test core class functionality with minimal data"""
    
    messages = []
    
    try:
        sys.path.insert(0, "src")
        from src.eval.milestone7_analysis import (
            PublicationMetrics,
            PublicationTableGenerator,
            HardwareProfileManager
        )
        
        # Test PublicationMetrics dataclass
        test_metrics = PublicationMetrics(
            ndcg_10=0.75, ndcg_20=0.80, recall_10=0.60, recall_20=0.70, mrr_10=0.65,
            tool_result_recall_10=0.45, action_consistency_score=0.85, 
            loop_exit_rate=0.95, provenance_precision=0.90,
            latency_p50_ms=120.5, latency_p95_ms=350.2, memory_peak_mb=512.0, qps=25.5,
            entity_coverage=0.88, scenario_coverage=0.92,
            ndcg_10_ci=(0.70, 0.80), latency_p95_ci=(300, 400), memory_peak_ci=(450, 600)
        )
        messages.append("✅ PublicationMetrics dataclass creation successful")
        
        # Test HardwareProfileManager
        hw_manager = HardwareProfileManager(Path("./test_hw_output"))
        profile = hw_manager.get_hardware_profile()
        messages.append(f"✅ Hardware profile detection: {profile.name}")
        
        # Test directory creation
        profile_dir = hw_manager.get_profile_output_dir()
        if profile_dir.exists():
            messages.append("✅ Hardware profile directory created")
        
        return True, messages
        
    except Exception as e:
        return False, [f"❌ Class functionality test failed: {str(e)}"]

def cleanup_test_files() -> None:
    """Clean up test output files"""
    
    import shutil
    
    test_dirs = [
        "./test_analysis_output",
        "./test_hw_output",
    ]
    
    for test_dir in test_dirs:
        dir_path = Path(test_dir)
        if dir_path.exists():
            shutil.rmtree(dir_path)
    
    # Clean up synthetic files
    synthetic_files = Path("/tmp").glob("synthetic_*.json")
    for file_path in synthetic_files:
        file_path.unlink(missing_ok=True)

def generate_validation_report(results: Dict[str, Tuple[bool, List[str]]]) -> str:
    """Generate comprehensive validation report"""
    
    report = []
    report.append("=" * 80)
    report.append("MILESTONE 7 IMPLEMENTATION VALIDATION REPORT")
    report.append("=" * 80)
    report.append("")
    
    overall_success = True
    total_tests = len(results)
    passed_tests = 0
    
    for test_name, (success, messages) in results.items():
        report.append(f"🔍 {test_name.replace('_', ' ').title()}")
        report.append("-" * 50)
        
        if success:
            report.append("✅ PASSED")
            passed_tests += 1
        else:
            report.append("❌ FAILED")
            overall_success = False
        
        for message in messages:
            report.append(f"   {message}")
        
        report.append("")
    
    report.append("=" * 80)
    report.append("SUMMARY")
    report.append("=" * 80)
    report.append(f"Tests Passed: {passed_tests}/{total_tests}")
    report.append(f"Overall Status: {'✅ PASSED' if overall_success else '❌ FAILED'}")
    
    if overall_success:
        report.append("")
        report.append("🎉 Milestone 7 implementation is ready for use!")
        report.append("")
        report.append("Next steps:")
        report.append("1. Run 'make milestone7-quick' for quick test with synthetic data")
        report.append("2. Run 'make figures' to generate publication-ready outputs")
        report.append("3. Check analysis/hardware_profiles/ for organized results")
    else:
        report.append("")
        report.append("❌ Implementation issues detected. Please review failed tests above.")
    
    return "\n".join(report)

def main():
    """Run complete Milestone 7 implementation validation"""
    
    print("🚀 Starting Milestone 7 Implementation Validation...")
    print("=" * 80)
    
    # Define all validation tests
    validation_tests = {
        "file_structure": validate_file_structure,
        "imports": validate_imports,
        "makefile_targets": validate_makefile_targets,
        "class_functionality": validate_class_functionality,
        "quick_analysis_run": test_quick_analysis_run,
    }
    
    # Run all validation tests
    results = {}
    for test_name, test_func in validation_tests.items():
        print(f"🔍 Running {test_name.replace('_', ' ')} validation...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            results[test_name] = (False, [f"❌ Test execution failed: {str(e)}"])
        print("")
    
    # Generate and display report
    report = generate_validation_report(results)
    print(report)
    
    # Save report to file
    report_file = Path("milestone7_validation_report.txt")
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"\n📄 Full validation report saved to: {report_file}")
    
    # Cleanup test files
    cleanup_test_files()
    
    # Return appropriate exit code
    overall_success = all(success for success, _ in results.values())
    sys.exit(0 if overall_success else 1)

if __name__ == "__main__":
    main()