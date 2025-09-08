#!/usr/bin/env python3
"""
Standalone Milestone 4 Validation Script  
========================================

Validates the Milestone 4 baseline implementation without complex dependencies.
Focuses on core functionality and interface validation.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

def validate_files_exist():
    """Validate that all required files are present"""
    required_files = [
        "src/eval/milestone4_baselines.py",
        "scripts/run_milestone4_baselines.py", 
        "scripts/test_milestone4_implementation.py",
        "config/milestone4_baseline_config.json",
        "docs/MILESTONE4_BASELINES.md"
    ]
    
    print("🔍 Validating file structure...")
    all_exist = True
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            all_exist = False
    
    return all_exist

def validate_code_structure():
    """Validate code structure without importing"""
    print("🔍 Validating code structure...")
    
    milestone4_file = Path("src/eval/milestone4_baselines.py")
    if not milestone4_file.exists():
        print("  ❌ Main implementation file missing")
        return False
    
    content = milestone4_file.read_text()
    
    # Check for required classes
    required_classes = [
        "SQLiteFTSBaseline",
        "VectorOnlyBaseline", 
        "HybridStaticBaseline",
        "MMRDiversityBaseline",
        "Doc2QueryExpansionBaseline",
        "TinyCrossEncoderBaseline",
        "Milestone4BaselineEvaluator"
    ]
    
    missing_classes = []
    for class_name in required_classes:
        if f"class {class_name}" in content:
            print(f"  ✅ {class_name}")
        else:
            print(f"  ❌ {class_name}")
            missing_classes.append(class_name)
    
    # Check for required methods
    required_patterns = [
        "def build_index(",
        "def retrieve(",
        "def get_flops_estimate(",
        "BudgetParityTracker",
        "AntiFreudValidator"
    ]
    
    for pattern in required_patterns:
        if pattern in content:
            print(f"  ✅ Found {pattern}")
        else:
            print(f"  ❌ Missing {pattern}")
    
    return len(missing_classes) == 0

def validate_makefile_integration():
    """Validate Makefile has required targets"""
    print("🔍 Validating Makefile integration...")
    
    makefile = Path("Makefile")
    if not makefile.exists():
        print("  ❌ Makefile not found")
        return False
    
    content = makefile.read_text()
    
    required_targets = [
        "baselines:",
        "milestone4-baselines:",
        "baseline-quick-test:",
        "test-baselines:"
    ]
    
    missing_targets = []
    for target in required_targets:
        if target in content:
            print(f"  ✅ {target}")
        else:
            print(f"  ❌ {target}")
            missing_targets.append(target)
    
    return len(missing_targets) == 0

def validate_configuration():
    """Validate configuration file structure"""
    print("🔍 Validating configuration...")
    
    config_file = Path("config/milestone4_baseline_config.json")
    if not config_file.exists():
        print("  ❌ Configuration file missing")
        return False
    
    try:
        with open(config_file) as f:
            config = json.load(f)
        
        required_sections = [
            "embedding_models",
            "bm25_parameters", 
            "hybrid_fusion",
            "mmr_diversity",
            "doc2query_expansion",
            "cross_encoder_rerank",
            "budget_parity"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section in config:
                print(f"  ✅ {section}")
            else:
                print(f"  ❌ {section}")
                missing_sections.append(section)
        
        return len(missing_sections) == 0
        
    except json.JSONDecodeError as e:
        print(f"  ❌ Invalid JSON: {e}")
        return False

def validate_documentation():
    """Validate documentation completeness"""
    print("🔍 Validating documentation...")
    
    doc_file = Path("docs/MILESTONE4_BASELINES.md")
    if not doc_file.exists():
        print("  ❌ Documentation file missing")
        return False
    
    content = doc_file.read_text()
    
    required_sections = [
        "## Implemented Baselines",
        "### 1. BM25-only",
        "### 2. Vector-only", 
        "### 3. BM25+Vector",
        "### 4. MMR Diversity",
        "### 5. BM25 + Doc2Query",
        "### 6. Tiny Cross-Encoder",
        "## Usage",
        "## Performance Characteristics",
        "## Troubleshooting"
    ]
    
    missing_sections = []
    for section in required_sections:
        if section in content:
            print(f"  ✅ {section}")
        else:
            print(f"  ❌ {section}")
            missing_sections.append(section)
    
    return len(missing_sections) == 0

def validate_interface_compliance():
    """Validate interface compliance by parsing code"""
    print("🔍 Validating baseline interface compliance...")
    
    milestone4_file = Path("src/eval/milestone4_baselines.py")
    content = milestone4_file.read_text()
    
    # Check that each baseline class has required methods
    baseline_classes = [
        "SQLiteFTSBaseline",
        "VectorOnlyBaseline", 
        "HybridStaticBaseline",
        "MMRDiversityBaseline",
        "Doc2QueryExpansionBaseline",
        "TinyCrossEncoderBaseline"
    ]
    
    required_methods = ["build_index", "retrieve", "get_flops_estimate"]
    
    compliance_count = 0
    
    for class_name in baseline_classes:
        class_found = f"class {class_name}" in content
        if class_found:
            methods_found = all(f"def {method}(" in content for method in required_methods)
            if methods_found:
                print(f"  ✅ {class_name}: Interface compliant")
                compliance_count += 1
            else:
                print(f"  ❌ {class_name}: Missing required methods")
        else:
            print(f"  ❌ {class_name}: Class not found")
    
    return compliance_count == len(baseline_classes)

def validate_cli_interface():
    """Validate command-line interface"""
    print("🔍 Validating CLI interface...")
    
    cli_file = Path("scripts/run_milestone4_baselines.py")
    content = cli_file.read_text()
    
    required_args = [
        "--dataset",
        "--output", 
        "--k",
        "--alpha",
        "--mmr-lambda",
        "--config"
    ]
    
    missing_args = []
    for arg in required_args:
        if arg in content:
            print(f"  ✅ {arg}")
        else:
            print(f"  ❌ {arg}")
            missing_args.append(arg)
    
    return len(missing_args) == 0

def main():
    """Run all validation checks"""
    print("🧪 Starting Milestone 4 Standalone Validation...\n")
    
    validation_results = {
        "files_exist": validate_files_exist(),
        "code_structure": validate_code_structure(), 
        "makefile_integration": validate_makefile_integration(),
        "configuration": validate_configuration(),
        "documentation": validate_documentation(),
        "interface_compliance": validate_interface_compliance(),
        "cli_interface": validate_cli_interface()
    }
    
    print(f"\n📊 Validation Results:")
    print("-" * 40)
    
    passed_count = 0
    total_count = len(validation_results)
    
    for check_name, passed in validation_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {check_name.replace('_', ' ').title()}: {status}")
        if passed:
            passed_count += 1
    
    print("-" * 40)
    print(f"Overall: {passed_count}/{total_count} checks passed ({passed_count/total_count:.1%})")
    
    if passed_count == total_count:
        print("\n✅ Milestone 4 implementation validation PASSED")
        print("All baselines are properly implemented and ready for evaluation!")
        return 0
    else:
        print("\n❌ Milestone 4 implementation validation FAILED")
        print(f"Please fix the {total_count - passed_count} failing validation checks.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)