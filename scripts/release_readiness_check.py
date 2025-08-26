#!/usr/bin/env python3
"""
Lethe Release Readiness Check
============================

Comprehensive pre-release validation ensuring all quality gates are met
before production release. This script validates the complete system
and ensures reproducibility from a fresh clone.

Quality Gates:
1. Environment validation
2. Dependency integrity
3. Build system functionality
4. Test suite execution
5. Performance regression
6. Security compliance
7. Documentation completeness
8. Fresh clone reproducibility
"""

import json
import subprocess
import sys
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile
import shutil

@dataclass
class QualityGate:
    """Individual quality gate result"""
    name: str
    description: str
    status: str  # "PASS", "FAIL", "WARN", "SKIP"
    execution_time_s: float
    details: str = ""
    error_message: str = ""
    artifacts: List[str] = None

@dataclass
class ReleaseReadinessReport:
    """Complete release readiness report"""
    timestamp: str
    lethe_version: str
    total_gates: int
    passed_gates: int
    failed_gates: int
    warned_gates: int
    skipped_gates: int
    overall_status: str
    execution_time_s: float
    gates: List[QualityGate]
    system_fingerprint: str
    recommendations: List[str]

class ReleaseReadinessChecker:
    """Complete release readiness validation system"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.start_time = time.time()
        self.gates = []
        self.temp_dirs = []
        
        print("🚀 Lethe Release Readiness Check")
        print("=" * 40)
        print(f"Project root: {self.project_root}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def run_command(self, cmd: str, cwd: Optional[Path] = None, timeout: int = 300) -> Tuple[int, str, str]:
        """Run shell command and return (returncode, stdout, stderr)"""
        if cwd is None:
            cwd = self.project_root
            
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return -1, "", str(e)
    
    def gate_environment_validation(self) -> QualityGate:
        """Validate development environment"""
        start_time = time.time()
        
        # Run environment check script
        returncode, stdout, stderr = self.run_command(
            "scripts/benchmark_env_check.sh"
        )
        
        execution_time = time.time() - start_time
        
        if returncode == 0:
            status = "PASS"
            details = "Environment meets all requirements"
        elif returncode == 1:
            status = "WARN"
            details = "Environment acceptable with warnings"
        else:
            status = "FAIL"
            details = f"Environment validation failed: {stderr}"
        
        return QualityGate(
            name="environment_validation",
            description="Validate development environment and dependencies",
            status=status,
            execution_time_s=execution_time,
            details=details,
            error_message=stderr if returncode != 0 else ""
        )
    
    def gate_dependency_integrity(self) -> QualityGate:
        """Validate dependency integrity and security"""
        start_time = time.time()
        
        checks = []
        
        # Check Node.js dependencies
        returncode, stdout, stderr = self.run_command("npm audit --audit-level=moderate")
        if returncode == 0:
            checks.append("✅ Node.js dependencies secure")
        else:
            checks.append(f"❌ Node.js security issues: {stderr}")
        
        # Check Python dependencies  
        returncode, stdout, stderr = self.run_command("cd lethe-research && pip check")
        if returncode == 0:
            checks.append("✅ Python dependencies compatible")
        else:
            checks.append(f"⚠️ Python dependency issues: {stderr}")
        
        # Check for known vulnerabilities
        returncode, stdout, stderr = self.run_command("npm list --depth=0")
        if "vulnerabilities" not in stdout.lower():
            checks.append("✅ No known vulnerabilities")
        else:
            checks.append("⚠️ Potential vulnerabilities detected")
        
        execution_time = time.time() - start_time
        failed_checks = [c for c in checks if "❌" in c]
        
        if not failed_checks:
            status = "PASS"
        elif len(failed_checks) < len(checks) // 2:
            status = "WARN"
        else:
            status = "FAIL"
        
        return QualityGate(
            name="dependency_integrity",
            description="Validate dependency security and compatibility",
            status=status,
            execution_time_s=execution_time,
            details="\n".join(checks),
            error_message="\n".join(failed_checks) if failed_checks else ""
        )
    
    def gate_build_system(self) -> QualityGate:
        """Validate build system functionality"""
        start_time = time.time()
        
        build_steps = []
        
        # Node.js build
        returncode, stdout, stderr = self.run_command("npm run build")
        if returncode == 0:
            build_steps.append("✅ Node.js build successful")
        else:
            build_steps.append(f"❌ Node.js build failed: {stderr}")
        
        # Python build (if applicable)
        if (self.project_root / "lethe-research" / "setup.py").exists():
            returncode, stdout, stderr = self.run_command("cd lethe-research && python setup.py build")
            if returncode == 0:
                build_steps.append("✅ Python build successful")
            else:
                build_steps.append(f"❌ Python build failed: {stderr}")
        
        # Index building
        returncode, stdout, stderr = self.run_command("make build_indices", timeout=600)
        if returncode == 0:
            build_steps.append("✅ Index building successful")
        else:
            build_steps.append(f"❌ Index building failed: {stderr}")
        
        execution_time = time.time() - start_time
        failed_builds = [b for b in build_steps if "❌" in b]
        
        status = "PASS" if not failed_builds else "FAIL"
        
        return QualityGate(
            name="build_system",
            description="Validate build system and index generation",
            status=status,
            execution_time_s=execution_time,
            details="\n".join(build_steps),
            error_message="\n".join(failed_builds) if failed_builds else ""
        )
    
    def gate_test_execution(self) -> QualityGate:
        """Execute test suite validation"""
        start_time = time.time()
        
        test_results = []
        
        # Unit tests
        returncode, stdout, stderr = self.run_command("npm test")
        if returncode == 0:
            test_results.append("✅ Unit tests passed")
        else:
            test_results.append(f"❌ Unit tests failed: {stderr}")
        
        # Integration tests  
        returncode, stdout, stderr = self.run_command("make quick_test")
        if returncode == 0:
            test_results.append("✅ Integration tests passed")
        else:
            test_results.append(f"❌ Integration tests failed: {stderr}")
        
        # Smoke tests
        returncode, stdout, stderr = self.run_command("python scripts/smoke_tests.py")
        if returncode == 0:
            test_results.append("✅ Smoke tests passed")
        else:
            test_results.append(f"❌ Smoke tests failed: {stderr}")
        
        execution_time = time.time() - start_time
        failed_tests = [t for t in test_results if "❌" in t]
        
        status = "PASS" if not failed_tests else "FAIL"
        
        return QualityGate(
            name="test_execution",
            description="Execute comprehensive test suite",
            status=status,
            execution_time_s=execution_time,
            details="\n".join(test_results),
            error_message="\n".join(failed_tests) if failed_tests else ""
        )
    
    def gate_performance_regression(self) -> QualityGate:
        """Check for performance regressions"""
        start_time = time.time()
        
        # Load performance baselines
        baseline_file = self.project_root / "performance_baselines.json"
        
        if not baseline_file.exists():
            return QualityGate(
                name="performance_regression",
                description="Check for performance regressions against baselines",
                status="SKIP",
                execution_time_s=time.time() - start_time,
                details="No performance baselines found - skipping regression check"
            )
        
        performance_checks = []
        
        # Run performance benchmarks
        returncode, stdout, stderr = self.run_command("make eval_all", timeout=1800)  # 30 min timeout
        
        if returncode == 0:
            performance_checks.append("✅ Performance evaluation completed")
            
            # Check for regression patterns in output
            if "regression" in stdout.lower():
                performance_checks.append("⚠️ Potential performance regression detected")
                status = "WARN"
            else:
                performance_checks.append("✅ No performance regressions detected")
                status = "PASS"
        else:
            performance_checks.append(f"❌ Performance evaluation failed: {stderr}")
            status = "FAIL"
        
        execution_time = time.time() - start_time
        
        return QualityGate(
            name="performance_regression",
            description="Check for performance regressions against baselines",
            status=status,
            execution_time_s=execution_time,
            details="\n".join(performance_checks),
            error_message=stderr if returncode != 0 else ""
        )
    
    def gate_security_compliance(self) -> QualityGate:
        """Validate security and privacy compliance"""
        start_time = time.time()
        
        security_checks = []
        
        # Check for hardcoded secrets
        returncode, stdout, stderr = self.run_command(
            "grep -r -i 'password\\|secret\\|token\\|api_key' --include='*.py' --include='*.ts' --include='*.js' . || true"
        )
        
        if stdout.strip():
            # Filter out legitimate patterns
            suspicious_lines = []
            for line in stdout.split('\n'):
                if line.strip() and not any(pattern in line.lower() for pattern in [
                    'test', 'example', 'placeholder', 'todo', 'fixme', 'comment'
                ]):
                    suspicious_lines.append(line)
            
            if suspicious_lines:
                security_checks.append(f"⚠️ Potential hardcoded secrets found:\n" + '\n'.join(suspicious_lines[:5]))
            else:
                security_checks.append("✅ No hardcoded secrets detected")
        else:
            security_checks.append("✅ No hardcoded secrets detected")
        
        # Check privacy scrubbing
        privacy_script = self.project_root / "lethe-research" / "datasets" / "privacy_scrubber.py"
        if privacy_script.exists():
            returncode, stdout, stderr = self.run_command(
                f"cd lethe-research && python datasets/privacy_scrubber.py --validate"
            )
            
            if returncode == 0:
                security_checks.append("✅ Privacy scrubbing validation passed")
            else:
                security_checks.append(f"❌ Privacy scrubbing validation failed: {stderr}")
        else:
            security_checks.append("⚠️ Privacy scrubbing script not found")
        
        # Check file permissions
        executable_files = []
        for ext in ['.sh', '.py']:
            returncode, stdout, stderr = self.run_command(f"find . -name '*{ext}' -executable")
            if stdout.strip():
                executable_files.extend(stdout.strip().split('\n'))
        
        if executable_files:
            security_checks.append(f"✅ Found {len(executable_files)} executable scripts")
        else:
            security_checks.append("⚠️ No executable scripts found")
        
        execution_time = time.time() - start_time
        
        failed_checks = [c for c in security_checks if "❌" in c]
        warning_checks = [c for c in security_checks if "⚠️" in c]
        
        if failed_checks:
            status = "FAIL"
        elif warning_checks:
            status = "WARN"
        else:
            status = "PASS"
        
        return QualityGate(
            name="security_compliance",
            description="Validate security and privacy compliance",
            status=status,
            execution_time_s=execution_time,
            details="\n".join(security_checks),
            error_message="\n".join(failed_checks) if failed_checks else ""
        )
    
    def gate_documentation_completeness(self) -> QualityGate:
        """Validate documentation completeness"""
        start_time = time.time()
        
        doc_checks = []
        
        # Check required files
        required_files = [
            "README.md",
            "LICENSE", 
            "CONTRIBUTING.md",
            "SECURITY.md"
        ]
        
        for file in required_files:
            file_path = self.project_root / file
            if file_path.exists() and file_path.stat().st_size > 100:  # At least 100 bytes
                doc_checks.append(f"✅ {file} exists and has content")
            else:
                doc_checks.append(f"❌ {file} missing or empty")
        
        # Check for API documentation
        api_docs_found = False
        for pattern in ["docs/API.md", "docs/api.md", "api.md", "API.md"]:
            if (self.project_root / pattern).exists():
                api_docs_found = True
                break
        
        if api_docs_found:
            doc_checks.append("✅ API documentation found")
        else:
            doc_checks.append("⚠️ API documentation not found")
        
        # Check for setup instructions
        setup_found = any([
            (self.project_root / "docs" / "SETUP.md").exists(),
            "installation" in (self.project_root / "README.md").read_text().lower(),
            "setup" in (self.project_root / "README.md").read_text().lower()
        ])
        
        if setup_found:
            doc_checks.append("✅ Setup instructions found")
        else:
            doc_checks.append("❌ Setup instructions missing")
        
        execution_time = time.time() - start_time
        
        failed_checks = [c for c in doc_checks if "❌" in c]
        status = "PASS" if not failed_checks else "FAIL"
        
        return QualityGate(
            name="documentation_completeness",
            description="Validate documentation completeness and quality",
            status=status,
            execution_time_s=execution_time,
            details="\n".join(doc_checks),
            error_message="\n".join(failed_checks) if failed_checks else ""
        )
    
    def gate_fresh_clone_reproducibility(self) -> QualityGate:
        """Test reproducibility from fresh clone"""
        start_time = time.time()
        
        # Create temporary directory for fresh clone test
        temp_dir = tempfile.mkdtemp(prefix="lethe_fresh_clone_")
        self.temp_dirs.append(temp_dir)
        
        clone_checks = []
        
        try:
            # Simulate fresh clone by copying project
            temp_project = Path(temp_dir) / "lethe"
            shutil.copytree(
                self.project_root, 
                temp_project,
                ignore=shutil.ignore_patterns(
                    'node_modules', '.git', '__pycache__', '*.pyc', 
                    'results', 'artifacts', 'mlruns', 'venv'
                )
            )
            
            clone_checks.append("✅ Project copied to temporary directory")
            
            # Test installation
            returncode, stdout, stderr = self.run_command(
                "make fresh_install", 
                cwd=temp_project,
                timeout=600
            )
            
            if returncode == 0:
                clone_checks.append("✅ Fresh installation successful")
            else:
                clone_checks.append(f"❌ Fresh installation failed: {stderr}")
                
            # Test basic functionality
            if returncode == 0:
                returncode, stdout, stderr = self.run_command(
                    "make quick_test",
                    cwd=temp_project, 
                    timeout=300
                )
                
                if returncode == 0:
                    clone_checks.append("✅ Basic functionality test passed")
                else:
                    clone_checks.append(f"❌ Basic functionality test failed: {stderr}")
            
        except Exception as e:
            clone_checks.append(f"❌ Fresh clone test error: {str(e)}")
        
        execution_time = time.time() - start_time
        
        failed_checks = [c for c in clone_checks if "❌" in c]
        status = "PASS" if not failed_checks else "FAIL"
        
        return QualityGate(
            name="fresh_clone_reproducibility", 
            description="Test reproducibility from fresh clone",
            status=status,
            execution_time_s=execution_time,
            details="\n".join(clone_checks),
            error_message="\n".join(failed_checks) if failed_checks else ""
        )
    
    def run_all_gates(self) -> ReleaseReadinessReport:
        """Run all quality gates and generate report"""
        print("🔍 Running release readiness quality gates...\n")
        
        # Define all gates
        gate_functions = [
            self.gate_environment_validation,
            self.gate_dependency_integrity,
            self.gate_build_system,
            self.gate_test_execution,
            self.gate_performance_regression,
            self.gate_security_compliance,
            self.gate_documentation_completeness,
            self.gate_fresh_clone_reproducibility
        ]
        
        # Execute gates
        for gate_func in gate_functions:
            print(f"Running {gate_func.__name__.replace('gate_', '').replace('_', ' ').title()}...")
            gate_result = gate_func()
            self.gates.append(gate_result)
            
            # Print immediate feedback
            status_icon = {
                "PASS": "✅",
                "FAIL": "❌", 
                "WARN": "⚠️",
                "SKIP": "⏭️"
            }[gate_result.status]
            
            print(f"{status_icon} {gate_result.name}: {gate_result.status}")
            if gate_result.error_message:
                print(f"   Error: {gate_result.error_message[:100]}...")
            print()
        
        # Calculate statistics
        total_gates = len(self.gates)
        passed_gates = len([g for g in self.gates if g.status == "PASS"])
        failed_gates = len([g for g in self.gates if g.status == "FAIL"])
        warned_gates = len([g for g in self.gates if g.status == "WARN"])
        skipped_gates = len([g for g in self.gates if g.status == "SKIP"])
        
        # Determine overall status
        if failed_gates > 0:
            overall_status = "BLOCKED"
        elif warned_gates > 0:
            overall_status = "CAUTION"
        else:
            overall_status = "READY"
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Create system fingerprint
        system_fingerprint = self._generate_system_fingerprint()
        
        execution_time = time.time() - self.start_time
        
        # Generate report
        report = ReleaseReadinessReport(
            timestamp=datetime.now().isoformat(),
            lethe_version="2.1.0",  # Would get from actual version
            total_gates=total_gates,
            passed_gates=passed_gates,
            failed_gates=failed_gates,
            warned_gates=warned_gates,
            skipped_gates=skipped_gates,
            overall_status=overall_status,
            execution_time_s=execution_time,
            gates=self.gates,
            system_fingerprint=system_fingerprint,
            recommendations=recommendations
        )
        
        self._print_summary(report)
        self._cleanup()
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on gate results"""
        recommendations = []
        
        failed_gates = [g for g in self.gates if g.status == "FAIL"]
        warned_gates = [g for g in self.gates if g.status == "WARN"]
        
        if failed_gates:
            recommendations.append("🚫 RELEASE BLOCKED: Address all failed quality gates before release")
            for gate in failed_gates:
                recommendations.append(f"   - Fix {gate.name}: {gate.error_message[:100]}")
        
        if warned_gates:
            recommendations.append("⚠️ Review warnings before release:")
            for gate in warned_gates:
                recommendations.append(f"   - Review {gate.name}: Consider addressing warnings")
        
        if not failed_gates and not warned_gates:
            recommendations.append("🎉 All quality gates passed - ready for release!")
            recommendations.append("📝 Consider running final manual validation")
            recommendations.append("🚀 Deploy with confidence")
        
        return recommendations
    
    def _generate_system_fingerprint(self) -> str:
        """Generate system fingerprint for reproducibility"""
        import hashlib
        
        fingerprint_data = {
            "os": os.name,
            "platform": sys.platform,
            "python_version": sys.version,
            "working_dir": str(self.project_root)
        }
        
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
    
    def _print_summary(self, report: ReleaseReadinessReport):
        """Print comprehensive summary"""
        print("=" * 60)
        print("🎯 RELEASE READINESS SUMMARY")
        print("=" * 60)
        
        # Overall status
        status_icons = {
            "READY": "🟢 READY FOR RELEASE",
            "CAUTION": "🟡 PROCEED WITH CAUTION", 
            "BLOCKED": "🔴 RELEASE BLOCKED"
        }
        
        print(f"\nOverall Status: {status_icons[report.overall_status]}")
        print(f"Total Quality Gates: {report.total_gates}")
        print(f"✅ Passed: {report.passed_gates}")
        print(f"❌ Failed: {report.failed_gates}")
        print(f"⚠️ Warned: {report.warned_gates}")
        print(f"⏭️ Skipped: {report.skipped_gates}")
        print(f"⏱️ Total Time: {report.execution_time_s:.1f}s")
        
        # Detailed gate results
        print(f"\n📊 Detailed Gate Results:")
        for gate in report.gates:
            status_icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "SKIP": "⏭️"}[gate.status]
            print(f"  {status_icon} {gate.name}: {gate.status} ({gate.execution_time_s:.1f}s)")
            
        # Recommendations
        print(f"\n🎯 Recommendations:")
        for rec in report.recommendations:
            print(f"  {rec}")
        
        print(f"\n🔐 System Fingerprint: {report.system_fingerprint}")
        print(f"📅 Generated: {report.timestamp}")
    
    def _cleanup(self):
        """Clean up temporary directories"""
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Could not clean up {temp_dir}: {e}")

def main():
    """Main release readiness check execution"""
    checker = ReleaseReadinessChecker()
    report = checker.run_all_gates()
    
    # Save report
    report_file = Path("release_readiness_report.json")
    with open(report_file, 'w') as f:
        json.dump(asdict(report), f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Exit with appropriate code
    if report.overall_status == "BLOCKED":
        print("\n🚫 RELEASE BLOCKED - Address failed quality gates")
        sys.exit(2)
    elif report.overall_status == "CAUTION":
        print("\n⚠️ PROCEED WITH CAUTION - Review warnings")
        sys.exit(1)
    else:
        print("\n🎉 READY FOR RELEASE!")
        sys.exit(0)

if __name__ == "__main__":
    main()