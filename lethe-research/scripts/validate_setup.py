#!/usr/bin/env python3
"""
Research Framework Validation Script
===================================

Comprehensive validation of the Lethe research infrastructure to ensure
all components are properly configured and functional before running
expensive experiments.

This script validates:
1. Environment setup and dependencies
2. Data availability and formats
3. Baseline implementations
4. Metric calculations
5. Statistical analysis pipeline
6. Output generation

Usage: python3 validate_setup.py [--verbose] [--fix-issues]
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import traceback

# Color constants for terminal output
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

class ValidationResult:
    def __init__(self, name: str, passed: bool, message: str, details: Optional[str] = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details
    
    def __str__(self):
        status = f"{Colors.GREEN}✅ PASS{Colors.NC}" if self.passed else f"{Colors.RED}❌ FAIL{Colors.NC}"
        result = f"{status} {self.name}: {self.message}"
        if self.details and not self.passed:
            result += f"\n   Details: {self.details}"
        return result

class ResearchValidator:
    """Validates the complete research framework"""
    
    def __init__(self, verbose: bool = False, fix_issues: bool = False):
        self.verbose = verbose
        self.fix_issues = fix_issues
        self.results: List[ValidationResult] = []
        
        # Detect paths
        script_dir = Path(__file__).parent
        self.research_dir = script_dir.parent
        self.project_dir = self.research_dir.parent
        self.ctx_run_dir = self.project_dir / "ctx-run"
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled"""
        if self.verbose:
            color = {
                "INFO": Colors.BLUE,
                "SUCCESS": Colors.GREEN,
                "WARNING": Colors.YELLOW,
                "ERROR": Colors.RED
            }.get(level, Colors.NC)
            print(f"{color}[{level}]{Colors.NC} {message}")
    
    def add_result(self, name: str, passed: bool, message: str, details: Optional[str] = None):
        """Add a validation result"""
        result = ValidationResult(name, passed, message, details)
        self.results.append(result)
        if self.verbose:
            print(result)
    
    def validate_environment(self):
        """Validate system environment and dependencies"""
        self.log("Validating environment...", "INFO")
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            self.add_result("Python Version", True, f"Python {python_version.major}.{python_version.minor} OK")
        else:
            self.add_result("Python Version", False, f"Python {python_version.major}.{python_version.minor} < 3.8")
        
        # Check required Python packages
        required_packages = [
            ("numpy", "numpy"),
            ("pandas", "pandas"), 
            ("scipy", "scipy"),
            ("sklearn", "scikit-learn"),
            ("yaml", "PyYAML")
        ]
        
        missing_packages = []
        for import_name, package_name in required_packages:
            try:
                __import__(import_name)
                self.add_result(f"Package: {package_name}", True, "Installed")
            except ImportError:
                self.add_result(f"Package: {package_name}", False, "Not installed")
                missing_packages.append(package_name)
        
        # Check system commands
        required_commands = ["node", "npm", "git"]
        for cmd in required_commands:
            try:
                result = subprocess.run([cmd, "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    version = result.stdout.strip().split('\n')[0]
                    self.add_result(f"Command: {cmd}", True, f"Available ({version})")
                else:
                    self.add_result(f"Command: {cmd}", False, "Not working")
            except FileNotFoundError:
                self.add_result(f"Command: {cmd}", False, "Not found")
    
    def validate_project_structure(self):
        """Validate project directory structure"""
        self.log("Validating project structure...", "INFO")
        
        # Check key directories
        required_dirs = [
            (self.ctx_run_dir, "ctx-run implementation"),
            (self.research_dir / "datasets", "datasets directory"),
            (self.research_dir / "experiments", "experiments directory"),
            (self.research_dir / "scripts", "scripts directory"),
            (self.research_dir / "analysis", "analysis directory"),
            (self.research_dir / "paper", "paper directory")
        ]
        
        for dir_path, description in required_dirs:
            if dir_path.exists():
                self.add_result(f"Directory: {description}", True, f"Exists at {dir_path}")
            else:
                self.add_result(f"Directory: {description}", False, f"Missing: {dir_path}")
        
        # Check key files
        required_files = [
            (self.research_dir / "experiments" / "hypothesis_framework.json", "hypothesis framework"),
            (self.research_dir / "experiments" / "grid_config.yaml", "grid configuration"),
            (self.research_dir / "analysis" / "metrics.py", "metrics implementation"),
            (self.research_dir / "scripts" / "baseline_implementations.py", "baseline implementations"),
            (self.research_dir / "paper" / "template.tex", "paper template"),
            (self.project_dir / "lethe_version.json", "environment snapshot")
        ]
        
        for file_path, description in required_files:
            if file_path.exists():
                self.add_result(f"File: {description}", True, f"Exists ({file_path.stat().st_size} bytes)")
            else:
                self.add_result(f"File: {description}", False, f"Missing: {file_path}")
    
    def validate_ctx_run(self):
        """Validate ctx-run implementation"""
        self.log("Validating ctx-run implementation...", "INFO")
        
        # Check if ctx-run builds
        try:
            os.chdir(self.ctx_run_dir)
            result = subprocess.run(["npm", "run", "build"], capture_output=True, text=True)
            if result.returncode == 0:
                self.add_result("ctx-run Build", True, "Builds successfully")
            else:
                self.add_result("ctx-run Build", False, f"Build failed: {result.stderr}")
        except Exception as e:
            self.add_result("ctx-run Build", False, f"Build error: {str(e)}")
        
        # Check CLI executable
        cli_path = self.ctx_run_dir / "packages" / "cli" / "dist" / "index.js"
        if cli_path.exists():
            self.add_result("CLI Executable", True, f"Found at {cli_path}")
        else:
            self.add_result("CLI Executable", False, f"Missing: {cli_path}")
            return
        
        # Test ctx-run initialization and diagnostics
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Test init
                result = subprocess.run([
                    "node", str(cli_path), "init", temp_dir
                ], capture_output=True, text=True, cwd=temp_dir)
                
                if result.returncode == 0:
                    self.add_result("ctx-run Init", True, "Initialization works")
                else:
                    self.add_result("ctx-run Init", False, f"Init failed: {result.stderr}")
                    return
                
                # Test diagnostics
                result = subprocess.run([
                    "node", str(cli_path), "diagnose"
                ], capture_output=True, text=True, cwd=temp_dir)
                
                if result.returncode == 0:
                    self.add_result("ctx-run Diagnostics", True, "Diagnostics pass")
                else:
                    self.add_result("ctx-run Diagnostics", False, f"Diagnostics failed: {result.stderr}")
                    
            except Exception as e:
                self.add_result("ctx-run Testing", False, f"Testing error: {str(e)}")
    
    def validate_configuration_files(self):
        """Validate configuration file formats"""
        self.log("Validating configuration files...", "INFO")
        
        # Validate hypothesis framework JSON
        hypothesis_path = self.research_dir / "experiments" / "hypothesis_framework.json"
        if hypothesis_path.exists():
            try:
                with open(hypothesis_path, 'r') as f:
                    data = json.load(f)
                
                required_keys = ["hypotheses", "grid_parameters", "evaluation_conditions"]
                missing_keys = [k for k in required_keys if k not in data]
                
                if not missing_keys:
                    self.add_result("Hypothesis Framework", True, f"Valid JSON with {len(data)} sections")
                else:
                    self.add_result("Hypothesis Framework", False, f"Missing keys: {missing_keys}")
                    
            except Exception as e:
                self.add_result("Hypothesis Framework", False, f"Invalid JSON: {str(e)}")
        
        # Validate grid config YAML
        grid_config_path = self.research_dir / "experiments" / "grid_config.yaml"
        if grid_config_path.exists():
            try:
                import yaml
                with open(grid_config_path, 'r') as f:
                    data = yaml.safe_load(f)
                
                required_sections = ["parameters", "baselines", "metrics"]
                missing_sections = [s for s in required_sections if s not in data]
                
                if not missing_sections:
                    param_count = len(data.get("parameters", {}))
                    baseline_count = len(data.get("baselines", {}))
                    self.add_result("Grid Configuration", True, f"Valid YAML ({param_count} params, {baseline_count} baselines)")
                else:
                    self.add_result("Grid Configuration", False, f"Missing sections: {missing_sections}")
                    
            except Exception as e:
                self.add_result("Grid Configuration", False, f"Invalid YAML: {str(e)}")
    
    def validate_baseline_implementations(self):
        """Validate baseline implementation code"""
        self.log("Validating baseline implementations...", "INFO")
        
        baseline_script = self.research_dir / "scripts" / "baseline_implementations.py"
        if not baseline_script.exists():
            self.add_result("Baseline Script", False, "baseline_implementations.py not found")
            return
        
        try:
            # Try to import the module
            sys.path.insert(0, str(self.research_dir / "scripts"))
            import baseline_implementations
            
            # Check for required classes
            required_classes = [
                "WindowBaseline", "BM25OnlyBaseline", "VectorOnlyBaseline",
                "BM25VectorSimpleBaseline", "CrossEncoderBaseline", 
                "FAISSIVFBaseline", "MMRBaseline", "BaselineEvaluator"
            ]
            
            missing_classes = []
            for class_name in required_classes:
                if hasattr(baseline_implementations, class_name):
                    self.add_result(f"Baseline: {class_name}", True, "Implementation found")
                else:
                    self.add_result(f"Baseline: {class_name}", False, "Missing implementation")
                    missing_classes.append(class_name)
            
            if not missing_classes:
                self.add_result("Baseline Implementations", True, f"All {len(required_classes)} baselines implemented")
            else:
                self.add_result("Baseline Implementations", False, f"Missing: {missing_classes}")
                
        except Exception as e:
            self.add_result("Baseline Implementations", False, f"Import error: {str(e)}")
    
    def validate_metrics_implementation(self):
        """Validate metrics calculation code"""
        self.log("Validating metrics implementation...", "INFO")
        
        metrics_script = self.research_dir / "analysis" / "metrics.py"
        if not metrics_script.exists():
            self.add_result("Metrics Script", False, "metrics.py not found")
            return
        
        try:
            sys.path.insert(0, str(self.research_dir / "analysis"))
            import metrics
            
            # Check for required classes and functions
            required_components = [
                ("MetricsCalculator", "class"),
                ("StatisticalComparator", "class"),
                ("QueryResult", "class"),
                ("EvaluationMetrics", "class"),
                ("load_results_from_json", "function"),
                ("save_metrics_to_json", "function")
            ]
            
            for component_name, component_type in required_components:
                if hasattr(metrics, component_name):
                    self.add_result(f"Metrics: {component_name}", True, f"{component_type} found")
                else:
                    self.add_result(f"Metrics: {component_name}", False, f"Missing {component_type}")
            
            # Test basic functionality
            calculator = metrics.MetricsCalculator()
            self.add_result("Metrics Calculator", True, "Can instantiate calculator")
            
        except Exception as e:
            self.add_result("Metrics Implementation", False, f"Import error: {str(e)}")
    
    def validate_scripts_executability(self):
        """Validate that scripts are executable and properly formatted"""
        self.log("Validating script executability...", "INFO")
        
        scripts_to_check = [
            "run_full_evaluation.sh",
            "create_dataset.sh", 
            "run_grid_search.sh",
            "evaluate_baselines.sh",
            "generate_paper.sh"
        ]
        
        for script_name in scripts_to_check:
            script_path = self.research_dir / "scripts" / script_name
            
            if not script_path.exists():
                self.add_result(f"Script: {script_name}", False, "Not found")
                continue
            
            # Check if executable
            if os.access(script_path, os.X_OK):
                self.add_result(f"Script: {script_name}", True, "Executable")
            else:
                self.add_result(f"Script: {script_name}", False, "Not executable")
                
                if self.fix_issues:
                    try:
                        script_path.chmod(0o755)
                        self.log(f"Fixed permissions for {script_name}", "SUCCESS")
                    except Exception as e:
                        self.log(f"Failed to fix permissions for {script_name}: {e}", "ERROR")
    
    def validate_sample_data_creation(self):
        """Validate that we can create sample data for testing"""
        self.log("Validating sample data creation...", "INFO")
        
        try:
            # Test baseline implementations with sample data
            sys.path.insert(0, str(self.research_dir / "scripts"))
            import baseline_implementations
            
            # Create sample data
            documents, queries = baseline_implementations.create_sample_data()
            
            if len(documents) > 0 and len(queries) > 0:
                self.add_result("Sample Data Creation", True, f"Created {len(documents)} docs, {len(queries)} queries")
            else:
                self.add_result("Sample Data Creation", False, "Empty sample data")
            
            # Test a simple baseline
            with tempfile.NamedTemporaryFile() as tmp_db:
                baseline = baseline_implementations.WindowBaseline(tmp_db.name)
                baseline.index_documents(documents[:10])
                
                results = baseline.retrieve(queries[0], k=5)
                if len(results) > 0:
                    self.add_result("Baseline Execution", True, f"Retrieved {len(results)} results")
                else:
                    self.add_result("Baseline Execution", False, "No results returned")
                    
        except Exception as e:
            self.add_result("Sample Data Testing", False, f"Error: {str(e)}")
    
    def validate_paper_template(self):
        """Validate LaTeX paper template"""
        self.log("Validating paper template...", "INFO")
        
        template_path = self.research_dir / "paper" / "template.tex"
        if not template_path.exists():
            self.add_result("Paper Template", False, "template.tex not found")
            return
        
        try:
            with open(template_path, 'r') as f:
                content = f.read()
            
            # Check for required placeholders
            required_placeholders = [
                "{{NDCG_IMPROVEMENT}}", "{{LATENCY_P95}}", "{{COVERAGE_20}}",
                "{{CONTRADICTION_REDUCTION}}", "{{LETHE_NDCG}}", "{{LETHE_RECALL}}"
            ]
            
            missing_placeholders = [p for p in required_placeholders if p not in content]
            
            if not missing_placeholders:
                placeholder_count = len([p for p in required_placeholders if p in content])
                self.add_result("Paper Template", True, f"Valid template with {placeholder_count} placeholders")
            else:
                self.add_result("Paper Template", False, f"Missing placeholders: {missing_placeholders}")
            
            # Check basic LaTeX structure
            if "\\documentclass" in content and "\\begin{document}" in content:
                self.add_result("LaTeX Structure", True, "Valid LaTeX document structure")
            else:
                self.add_result("LaTeX Structure", False, "Invalid LaTeX structure")
                
        except Exception as e:
            self.add_result("Paper Template", False, f"Error reading template: {str(e)}")
    
    def run_all_validations(self):
        """Run all validation checks"""
        print(f"{Colors.CYAN}🔍 Lethe Research Framework Validation{Colors.NC}")
        print(f"{Colors.CYAN}======================================={Colors.NC}")
        print()
        
        validations = [
            self.validate_environment,
            self.validate_project_structure,
            self.validate_ctx_run,
            self.validate_configuration_files,
            self.validate_baseline_implementations,
            self.validate_metrics_implementation,
            self.validate_scripts_executability,
            self.validate_sample_data_creation,
            self.validate_paper_template
        ]
        
        for validation_func in validations:
            try:
                validation_func()
            except Exception as e:
                self.add_result(validation_func.__name__, False, f"Validation error: {str(e)}")
                if self.verbose:
                    traceback.print_exc()
        
        # Print summary
        print(f"\n{Colors.CYAN}Validation Summary{Colors.NC}")
        print(f"{Colors.CYAN}================={Colors.NC}")
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        if not self.verbose:
            # Print all results if not in verbose mode
            for result in self.results:
                print(result)
        
        print(f"\n{Colors.CYAN}Results: {passed}/{total} checks passed{Colors.NC}")
        
        if passed == total:
            print(f"{Colors.GREEN}🎉 All validations passed! Research framework is ready.{Colors.NC}")
            return 0
        else:
            failed = total - passed
            print(f"{Colors.RED}❌ {failed} validation(s) failed. Please address issues before running experiments.{Colors.NC}")
            
            # Show fix suggestions
            print(f"\n{Colors.YELLOW}💡 Suggested fixes:{Colors.NC}")
            for result in self.results:
                if not result.passed and "missing" in result.message.lower():
                    print(f"  - Install/create: {result.name}")
                elif not result.passed and "not executable" in result.message.lower():
                    print(f"  - chmod +x for: {result.name}")
            
            return 1

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate Lethe research framework setup",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose output")
    parser.add_argument("--fix-issues", action="store_true",
                       help="Attempt to fix simple issues automatically")
    
    args = parser.parse_args()
    
    validator = ResearchValidator(verbose=args.verbose, fix_issues=args.fix_issues)
    return validator.run_all_validations()

if __name__ == "__main__":
    sys.exit(main())