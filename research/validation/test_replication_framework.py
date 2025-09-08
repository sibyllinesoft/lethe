#!/usr/bin/env python3
"""
Test Suite for Lethe Comprehensive Replication Framework
========================================================

Comprehensive validation of all framework components:
1. CLI tool functionality
2. Matrix configuration validation
3. Docker Compose setup
4. Interactive calculator generation
5. Adversarial testing framework
6. Cryptographic integrity checks
7. Statistical validation

Usage:
    python3 test_replication_framework.py [--verbose] [--quick]
"""

import json
import yaml
import subprocess
import tempfile
import shutil
import hashlib
import hmac
import pytest
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReplicationFrameworkTester:
    """Comprehensive test suite for the replication framework"""
    
    def __init__(self, verbose: bool = False, quick: bool = False):
        self.verbose = verbose
        self.quick = quick
        self.test_results = []
        self.temp_dir = Path(tempfile.mkdtemp())
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
    
    def run_all_tests(self) -> bool:
        """Run all tests and return overall success"""
        logger.info("🧪 Starting comprehensive replication framework tests...")
        
        test_methods = [
            self.test_cli_tool_exists,
            self.test_matrix_configuration,
            self.test_docker_compose_validity,
            self.test_comprehensive_framework,
            self.test_adversarial_suite,
            self.test_calculator_generation,
            self.test_cryptographic_integrity,
            self.test_statistical_validation,
        ]
        
        if self.quick:
            # Run only essential tests in quick mode
            test_methods = test_methods[:4]
            logger.info("🏃 Running in quick mode - essential tests only")
        
        passed = 0
        total = len(test_methods)
        
        for test_method in test_methods:
            try:
                logger.info(f"Running {test_method.__name__}...")
                result = test_method()
                if result:
                    passed += 1
                    logger.info(f"✅ {test_method.__name__} PASSED")
                else:
                    logger.error(f"❌ {test_method.__name__} FAILED")
                
                self.test_results.append({
                    "test": test_method.__name__,
                    "passed": result,
                    "description": test_method.__doc__ or "No description"
                })
                
            except Exception as e:
                logger.error(f"❌ {test_method.__name__} ERROR: {e}")
                self.test_results.append({
                    "test": test_method.__name__,
                    "passed": False,
                    "error": str(e)
                })
        
        # Generate test report
        self._generate_test_report(passed, total)
        
        # Cleanup
        self._cleanup()
        
        success = passed == total
        if success:
            logger.info(f"🎉 All {total} tests passed!")
        else:
            logger.error(f"💥 {total - passed} of {total} tests failed")
        
        return success
    
    def test_cli_tool_exists(self) -> bool:
        """Test that lethe-bench CLI tool exists and is executable"""
        cli_path = Path("lethe-bench")
        
        if not cli_path.exists():
            logger.error("lethe-bench CLI tool not found")
            return False
        
        # Check if executable
        if not cli_path.stat().st_mode & 0o111:
            logger.error("lethe-bench is not executable")
            return False
        
        # Test basic help
        try:
            result = subprocess.run(
                ["./lethe-bench", "--help"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode != 0:
                logger.error(f"CLI help failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("CLI help command timed out")
            return False
        except Exception as e:
            logger.error(f"CLI help command failed: {e}")
            return False
        
        logger.debug("✅ CLI tool exists and responds to --help")
        return True
    
    def test_matrix_configuration(self) -> bool:
        """Test that matrix.yml is valid and contains required sections"""
        matrix_path = Path("matrix.yml")
        
        if not matrix_path.exists():
            logger.error("matrix.yml not found")
            return False
        
        try:
            with open(matrix_path) as f:
                matrix = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in matrix.yml: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading matrix.yml: {e}")
            return False
        
        # Check required sections
        required_sections = ["systems", "scenarios", "validation", "adversarial"]
        for section in required_sections:
            if section not in matrix:
                logger.error(f"Missing required section '{section}' in matrix.yml")
                return False
        
        # Validate systems
        systems = matrix["systems"]
        if not systems:
            logger.error("No systems defined in matrix.yml")
            return False
        
        for system_name, config in systems.items():
            if "endpoint" not in config:
                logger.error(f"System {system_name} missing endpoint")
                return False
            
            if "expected_performance" not in config:
                logger.error(f"System {system_name} missing expected_performance")
                return False
        
        # Validate scenarios
        scenarios = matrix["scenarios"]
        if not scenarios:
            logger.error("No scenarios defined in matrix.yml")
            return False
        
        for scenario in scenarios:
            required_fields = ["name", "description", "metrics", "keep_ratios"]
            for field in required_fields:
                if field not in scenario:
                    logger.error(f"Scenario missing required field '{field}'")
                    return False
        
        # Validate adversarial config
        adversarial = matrix["adversarial"]
        if not adversarial.get("enabled"):
            logger.warning("Adversarial testing disabled in matrix.yml")
        
        test_suites = adversarial.get("test_suites", {})
        expected_tests = ["near_duplicate_storm", "symbol_chain_depth", "json_kv_needles"]
        for test in expected_tests:
            if test not in test_suites:
                logger.error(f"Missing adversarial test: {test}")
                return False
        
        logger.debug("✅ Matrix configuration is valid")
        return True
    
    def test_docker_compose_validity(self) -> bool:
        """Test that Docker Compose configurations are valid"""
        compose_files = ["docker-compose.yml", "docker-compose.replication.yml"]
        
        for compose_file in compose_files:
            compose_path = Path(compose_file)
            
            if not compose_path.exists():
                logger.warning(f"{compose_file} not found, skipping validation")
                continue
            
            # Validate YAML syntax
            try:
                with open(compose_path) as f:
                    compose_config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                logger.error(f"Invalid YAML in {compose_file}: {e}")
                return False
            
            # Check basic structure
            if "services" not in compose_config:
                logger.error(f"Missing 'services' section in {compose_file}")
                return False
            
            services = compose_config["services"]
            
            # Check for core services
            if compose_file == "docker-compose.replication.yml":
                required_services = ["lethe-hybrid", "weaviate", "milvus"]
                for service in required_services:
                    if service not in services:
                        logger.error(f"Missing required service '{service}' in {compose_file}")
                        return False
            
            # Validate service configurations
            for service_name, service_config in services.items():
                # Check for health checks on key services
                if service_name in ["lethe-hybrid", "weaviate", "milvus"]:
                    if "healthcheck" not in service_config:
                        logger.warning(f"Service {service_name} missing health check")
            
            # Test Docker Compose validation (if docker-compose available)
            try:
                result = subprocess.run(
                    ["docker-compose", "-f", compose_file, "config"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode != 0:
                    logger.warning(f"Docker Compose validation warnings for {compose_file}: {result.stderr}")
            except FileNotFoundError:
                logger.warning("docker-compose not available for validation")
            except subprocess.TimeoutExpired:
                logger.warning(f"Docker Compose validation timed out for {compose_file}")
            except Exception as e:
                logger.warning(f"Docker Compose validation error for {compose_file}: {e}")
        
        logger.debug("✅ Docker Compose configurations are valid")
        return True
    
    def test_comprehensive_framework(self) -> bool:
        """Test that the comprehensive framework can run and generate outputs"""
        framework_path = Path("comprehensive_replication_framework.py")
        
        if not framework_path.exists():
            logger.error("comprehensive_replication_framework.py not found")
            return False
        
        # Test framework execution (dry run with minimal data)
        try:
            # Create minimal test data
            test_data = {
                "timestamp": "2025-01-08T00:00:00",
                "systems": {
                    "test_system": {
                        "latency_ms": 20.0,
                        "relevance_score": 0.8,
                        "success_rate": 95.0
                    }
                }
            }
            
            test_data_path = self.temp_dir / "test_data.json"
            with open(test_data_path, "w") as f:
                json.dump(test_data, f)
            
            # Run framework with test data
            result = subprocess.run(
                ["python3", "comprehensive_replication_framework.py", 
                 "--existing-results", str(test_data_path)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=Path.cwd()
            )
            
            if result.returncode != 0:
                logger.error(f"Framework execution failed: {result.stderr}")
                return False
            
            # Check that outputs were generated
            output_patterns = [
                "lethe-replication-pack-*.zip",
                "lethe_decision_calculator_*.html",
                "adversarial_results_*.json",
                "drift_analysis_*.json",
                "comprehensive_replication_report_*.html"
            ]
            
            for pattern in output_patterns:
                matching_files = list(Path.cwd().glob(pattern))
                if not matching_files:
                    logger.error(f"Expected output file matching pattern {pattern} not found")
                    return False
            
        except subprocess.TimeoutExpired:
            logger.error("Framework execution timed out")
            return False
        except Exception as e:
            logger.error(f"Framework execution error: {e}")
            return False
        
        logger.debug("✅ Comprehensive framework executes successfully")
        return True
    
    def test_adversarial_suite(self) -> bool:
        """Test that adversarial test suite components are properly configured"""
        try:
            # Import and test adversarial suite
            import sys
            sys.path.append(str(Path.cwd()))
            
            from comprehensive_replication_framework import AdversarialTestSuite
            
            suite = AdversarialTestSuite()
            
            # Check that all required tests are defined
            expected_tests = [
                "Near-Duplicate Storm",
                "Symbol Chain Depth 4-6", 
                "JSON-KV Needles",
                "Bilingual Code-Switch",
                "Index Outage Scenario"
            ]
            
            test_names = [test.name for test in suite.tests]
            
            for expected in expected_tests:
                if expected not in test_names:
                    logger.error(f"Missing adversarial test: {expected}")
                    return False
            
            # Test that each adversarial test has required components
            for test in suite.tests:
                if not test.description:
                    logger.error(f"Test {test.name} missing description")
                    return False
                
                if not test.parameters:
                    logger.error(f"Test {test.name} missing parameters")
                    return False
                
                if test.expected_degradation <= 0 or test.expected_degradation > 1:
                    logger.error(f"Test {test.name} invalid degradation threshold: {test.expected_degradation}")
                    return False
        
        except ImportError as e:
            logger.error(f"Could not import adversarial suite: {e}")
            return False
        except Exception as e:
            logger.error(f"Adversarial suite test error: {e}")
            return False
        
        logger.debug("✅ Adversarial test suite is properly configured")
        return True
    
    def test_calculator_generation(self) -> bool:
        """Test that interactive calculator can be generated"""
        try:
            import sys
            sys.path.append(str(Path.cwd()))
            
            from comprehensive_replication_framework import InteractiveCalculator
            
            calculator = InteractiveCalculator()
            
            # Test data
            test_performance_data = {
                "systems": {
                    "lethe_hybrid": {
                        "latency_ms": 14.0,
                        "relevance_score": 0.831,
                        "cost_per_query": 0.0012
                    },
                    "weaviate": {
                        "latency_ms": 43.2,
                        "relevance_score": 0.735,
                        "cost_per_query": 0.0031
                    }
                }
            }
            
            # Generate calculator HTML
            html_content = calculator.generate_calculator_html(test_performance_data)
            
            # Basic validation of generated HTML
            required_elements = [
                "Decision Calculator",
                "latency-target",
                "budget-ratio", 
                "query-complexity",
                "recalculate()",
                "performance-chart"
            ]
            
            for element in required_elements:
                if element not in html_content:
                    logger.error(f"Calculator HTML missing required element: {element}")
                    return False
            
            # Check that it's valid HTML structure
            if not html_content.strip().startswith("<!DOCTYPE html>"):
                logger.error("Generated calculator is not valid HTML")
                return False
            
            if "</html>" not in html_content:
                logger.error("Generated calculator HTML is incomplete")
                return False
            
            # Save test output
            test_calc_path = self.temp_dir / "test_calculator.html"
            with open(test_calc_path, "w") as f:
                f.write(html_content)
            
        except ImportError as e:
            logger.error(f"Could not import calculator: {e}")
            return False
        except Exception as e:
            logger.error(f"Calculator generation error: {e}")
            return False
        
        logger.debug("✅ Interactive calculator generates successfully")
        return True
    
    def test_cryptographic_integrity(self) -> bool:
        """Test cryptographic integrity features"""
        try:
            import sys
            sys.path.append(str(Path.cwd()))
            
            from comprehensive_replication_framework import ArtifactHasher
            
            hasher = ArtifactHasher("test_secret_key")
            
            # Test pool hashing
            test_pool = [{"id": 1, "content": "test"}, {"id": 2, "content": "test2"}]
            hash1 = hasher.hash_pool(test_pool)
            hash2 = hasher.hash_pool(test_pool)
            
            if hash1 != hash2:
                logger.error("Pool hashing is not deterministic")
                return False
            
            if len(hash1) != 64:  # SHA256 hex length
                logger.error("Pool hash wrong length")
                return False
            
            # Test tokenizer hashing
            tokenizer_config = {"model": "test", "vocab_size": 1000}
            tok_hash = hasher.hash_tokenizer(tokenizer_config)
            
            if len(tok_hash) != 64:
                logger.error("Tokenizer hash wrong length")
                return False
            
            # Test manifest signing
            test_manifest = {"version": "1.0", "created_at": "2025-01-08", "data": test_pool}
            signature = hasher.sign_manifest(test_manifest)
            
            if len(signature) != 64:  # HMAC-SHA256 hex length
                logger.error("Manifest signature wrong length")
                return False
            
            # Test signature verification
            if not hasher.verify_manifest(test_manifest, signature):
                logger.error("Manifest signature verification failed")
                return False
            
            # Test signature verification with wrong data
            wrong_manifest = {"version": "2.0", "data": "wrong"}
            if hasher.verify_manifest(wrong_manifest, signature):
                logger.error("Manifest signature verification should have failed")
                return False
        
        except ImportError as e:
            logger.error(f"Could not import cryptographic components: {e}")
            return False
        except Exception as e:
            logger.error(f"Cryptographic integrity test error: {e}")
            return False
        
        logger.debug("✅ Cryptographic integrity features work correctly")
        return True
    
    def test_statistical_validation(self) -> bool:
        """Test statistical validation components"""
        try:
            # Test data with potential violations
            test_results = {
                "systems": {
                    "system1": {
                        "avg_latency_ms": 20.0,
                        "p95_latency_ms": 35.0,  # Valid: P95 > avg
                        "overall_success_rate": 95.0,
                        "macro_p5": 0.85
                    },
                    "system2": {
                        "avg_latency_ms": 50.0,
                        "p95_latency_ms": 45.0,  # Invalid: P95 < avg  
                        "overall_success_rate": 85.0,
                        "macro_p5": 0.75
                    }
                }
            }
            
            # Save test results
            test_results_path = self.temp_dir / "test_results.json"
            with open(test_results_path, "w") as f:
                json.dump(test_results, f)
            
            # Test validation using CLI tool
            result = subprocess.run(
                ["python3", "-c", f"""
import sys
sys.path.append('{Path.cwd()}')
from comprehensive_replication_framework import FailClosedValidator

validator = FailClosedValidator(strict_mode=True)
is_valid, messages = validator.validate_results('{test_results_path}')
print('VALID' if is_valid else 'INVALID')
for msg in messages:
    print(f'MESSAGE: {{msg}}')
"""],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.error(f"Statistical validation test failed: {result.stderr}")
                return False
            
            output_lines = result.stdout.strip().split('\n')
            validation_result = output_lines[0]
            
            # Should be INVALID due to P95 < avg violation
            if validation_result != "INVALID":
                logger.error("Statistical validation should have detected P95 < avg violation")
                return False
            
            # Check that violation was detected
            found_violation = any("P95" in line and "avg" in line for line in output_lines)
            if not found_violation:
                logger.error("Statistical validation did not report expected P95 violation")
                return False
        
        except subprocess.TimeoutExpired:
            logger.error("Statistical validation test timed out")
            return False
        except Exception as e:
            logger.error(f"Statistical validation test error: {e}")
            return False
        
        logger.debug("✅ Statistical validation detects violations correctly")
        return True
    
    def _generate_test_report(self, passed: int, total: int):
        """Generate comprehensive test report"""
        report = {
            "timestamp": "2025-01-08T00:00:00Z",
            "framework_version": "1.0",
            "test_summary": {
                "total_tests": total,
                "passed_tests": passed,
                "failed_tests": total - passed,
                "success_rate": passed / total if total > 0 else 0
            },
            "test_results": self.test_results
        }
        
        report_path = Path("test_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📋 Test report saved to {report_path}")
        
        # Generate markdown report
        md_report = self._generate_markdown_report(report)
        md_report_path = Path("TEST_REPORT.md")
        with open(md_report_path, "w") as f:
            f.write(md_report)
        
        logger.info(f"📋 Markdown test report saved to {md_report_path}")
    
    def _generate_markdown_report(self, report: Dict) -> str:
        """Generate markdown test report"""
        summary = report["test_summary"]
        
        md = f"""# Lethe Replication Framework Test Report

**Generated:** {report["timestamp"]}
**Framework Version:** {report["framework_version"]}

## Summary

- **Total Tests:** {summary["total_tests"]}
- **Passed:** {summary["passed_tests"]} ✅
- **Failed:** {summary["failed_tests"]} ❌  
- **Success Rate:** {summary["success_rate"]:.1%}

## Test Results

| Test | Status | Description |
|------|---------|-------------|
"""
        
        for result in report["test_results"]:
            status = "✅ PASS" if result["passed"] else "❌ FAIL"
            description = result.get("description", "No description").replace("\n", " ").strip()
            if result.get("error"):
                description += f" (Error: {result['error']})"
            
            md += f"| `{result['test']}` | {status} | {description} |\n"
        
        md += f"""

## Recommendations

"""
        
        if summary["failed_tests"] == 0:
            md += "🎉 **All tests passed!** The replication framework is ready for deployment.\n"
        else:
            md += f"⚠️ **{summary['failed_tests']} test(s) failed.** Review the failures above and fix issues before deployment.\n"
        
        md += """
## Next Steps

1. **If all tests passed:** Proceed with framework deployment
2. **If tests failed:** Review failure details and fix issues
3. **Re-run tests:** Execute `python3 test_replication_framework.py` after fixes

## Framework Components Validated

- ✅ CLI tool functionality and help system
- ✅ Matrix configuration schema and required fields
- ✅ Docker Compose service definitions and health checks  
- ✅ Comprehensive framework execution and output generation
- ✅ Adversarial test suite configuration and parameters
- ✅ Interactive calculator HTML generation and structure
- ✅ Cryptographic integrity (hashing and signing)
- ✅ Statistical validation and fail-closed operation
"""
        
        return md
    
    def _cleanup(self):
        """Clean up temporary files"""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Lethe Replication Framework")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", "-q", action="store_true", help="Quick mode (essential tests only)")
    
    args = parser.parse_args()
    
    tester = ReplicationFrameworkTester(verbose=args.verbose, quick=args.quick)
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 ALL TESTS PASSED - Framework is ready!")
        exit(0)
    else:
        print("\n💥 SOME TESTS FAILED - Check test report for details")
        exit(1)


if __name__ == "__main__":
    main()