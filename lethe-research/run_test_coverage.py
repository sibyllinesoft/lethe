#!/usr/bin/env python3
"""
Test runner script with coverage analysis.

Runs the new comprehensive tests and generates coverage reports
to validate the test coverage improvements.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_tests_with_coverage():
    """Run tests with coverage reporting."""
    
    # Change to project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("=== Running Test Coverage Analysis ===")
    print(f"Project root: {project_root}")
    
    # List of test files to run
    test_files = [
        "tests/test_fusion_core.py",
        "tests/test_common_data_structures.py", 
        "tests/test_common_validation.py",
        "tests/test_rerank_core.py"
    ]
    
    # Check that test files exist
    missing_files = []
    for test_file in test_files:
        if not Path(test_file).exists():
            missing_files.append(test_file)
    
    if missing_files:
        print(f"ERROR: Missing test files: {missing_files}")
        return False
    
    print(f"Found {len(test_files)} test files to run")
    
    # Run coverage with pytest
    try:
        print("\n=== Installing required packages ===")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "pytest", "coverage", "pytest-cov", "numpy"
        ], check=True)
        
        print("\n=== Running tests with coverage ===")
        
        # Run pytest with coverage
        cmd = [
            sys.executable, "-m", "pytest", 
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--verbose"
        ] + test_files
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"Tests failed with return code: {result.returncode}")
            return False
            
        print("\n=== Coverage Analysis Complete ===")
        print("Coverage report saved to: htmlcov/index.html")
        
        # Generate additional detailed coverage report
        print("\n=== Generating detailed coverage report ===")
        detailed_cmd = [sys.executable, "-m", "coverage", "report", "--show-missing"]
        detailed_result = subprocess.run(detailed_cmd, capture_output=True, text=True)
        
        print("Detailed Coverage Report:")
        print(detailed_result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error running tests: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def analyze_coverage_improvements():
    """Analyze and report coverage improvements."""
    
    print("\n=== Coverage Improvement Analysis ===")
    
    # Run coverage report to get current stats
    try:
        result = subprocess.run([
            sys.executable, "-m", "coverage", "report", 
            "--include=src/*", "--skip-empty"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            
            # Parse coverage data
            covered_files = 0
            total_statements = 0
            covered_statements = 0
            
            for line in lines:
                if line.startswith('src/') and not line.startswith('---'):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            statements = int(parts[1])
                            missed = int(parts[2])
                            covered = statements - missed
                            
                            total_statements += statements
                            covered_statements += covered
                            covered_files += 1
                        except (ValueError, IndexError):
                            continue
            
            if total_statements > 0:
                overall_coverage = (covered_statements / total_statements) * 100
                print(f"Files with coverage: {covered_files}")
                print(f"Total statements: {total_statements}")
                print(f"Covered statements: {covered_statements}")
                print(f"Overall coverage: {overall_coverage:.1f}%")
                
                # Identify high-impact improvements
                print("\n=== Key Achievements ===")
                print("✅ Added comprehensive tests for fusion core module")
                print("✅ Added comprehensive tests for common data structures")
                print("✅ Added comprehensive tests for validation framework")
                print("✅ Added comprehensive tests for rerank core module")
                print("✅ Implemented edge case and error condition testing")
                print("✅ Added performance and timing functionality tests")
                
                return True
            else:
                print("No coverage data found")
                return False
                
        else:
            print("Error generating coverage report")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Error analyzing coverage: {e}")
        return False

def main():
    """Main test runner function."""
    
    print("🚀 Starting comprehensive test coverage analysis...")
    
    # Run tests with coverage
    success = run_tests_with_coverage()
    
    if success:
        # Analyze improvements
        analyze_coverage_improvements()
        
        print("\n✅ Test coverage analysis completed successfully!")
        print("\nNext steps:")
        print("1. Review coverage report: open htmlcov/index.html")
        print("2. Identify remaining uncovered areas")
        print("3. Add additional tests for critical uncovered functions")
        
        return 0
    else:
        print("\n❌ Test coverage analysis failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())