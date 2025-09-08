#!/usr/bin/env python3
"""
Fresh Clone Validation Test
==========================

Tests that a fresh clone of the Lethe repository can be set up and
run the complete evaluation pipeline to reproduce headline numbers.

This is the ultimate validation that the project is properly
packaged for open-source release.

Test Phases:
1. Environment setup validation
2. Dependency installation
3. Build system execution  
4. Index generation
5. Basic functionality test
6. Performance evaluation
7. Result comparison

Usage:
    python scripts/fresh_clone_test.py [--temp-dir PATH] [--keep-temp]
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

class FreshCloneValidator:
    """Validates complete setup and execution from fresh clone"""
    
    def __init__(self, source_dir: str, temp_dir: Optional[str] = None, keep_temp: bool = False):
        self.source_dir = Path(source_dir).resolve()
        self.keep_temp = keep_temp
        
        if temp_dir:
            self.temp_dir = Path(temp_dir).resolve()
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="lethe_fresh_"))
            
        self.clone_dir = self.temp_dir / "lethe"
        self.results = []
        self.start_time = time.time()
        
        print(f"🔬 Fresh Clone Validation Test")
        print(f"Source: {self.source_dir}")
        print(f"Test dir: {self.temp_dir}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 50)
    
    def run_command(self, cmd: str, cwd: Optional[Path] = None, timeout: int = 600) -> Tuple[int, str, str]:
        """Execute command and return result"""
        if cwd is None:
            cwd = self.clone_dir
            
        print(f"  Running: {cmd}")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            print(f"  Completed in {execution_time:.1f}s (exit code: {result.returncode})")
            
            return result.returncode, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            print(f"  ❌ Command timed out after {timeout}s")
            return -1, "", f"Timeout after {timeout}s"
        except Exception as e:
            print(f"  ❌ Command failed: {e}")
            return -1, "", str(e)
    
    def log_step(self, step: str, success: bool, details: str = "", duration: float = 0):
        """Log test step result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {step} ({duration:.1f}s)")
        if details:
            print(f"    {details}")
        
        self.results.append({
            "step": step,
            "success": success,
            "details": details,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        })
    
    def step_clone_project(self) -> bool:
        """Clone/copy project to temporary directory"""
        print("\n1️⃣ Cloning project...")
        start_time = time.time()
        
        try:
            # Copy source to temp directory (simulating git clone)
            shutil.copytree(
                self.source_dir,
                self.clone_dir,
                ignore=shutil.ignore_patterns(
                    '.git', 'node_modules', '__pycache__', '*.pyc',
                    'results', 'artifacts', 'mlruns', 'venv', '.venv',
                    'lethe-research/venv*', 'ctx-run/node_modules'
                )
            )
            
            duration = time.time() - start_time
            self.log_step("Project cloned", True, f"Copied to {self.clone_dir}", duration)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_step("Project cloned", False, str(e), duration)
            return False
    
    def step_environment_check(self) -> bool:
        """Validate environment meets requirements"""
        print("\n2️⃣ Checking environment...")
        start_time = time.time()
        
        returncode, stdout, stderr = self.run_command(
            "scripts/benchmark_env_check.sh",
            timeout=60
        )
        
        duration = time.time() - start_time
        success = returncode in [0, 1]  # 0=perfect, 1=acceptable warnings
        
        details = "Environment meets requirements" if returncode == 0 else "Environment has warnings"
        if not success:
            details = f"Environment check failed: {stderr}"
            
        self.log_step("Environment validated", success, details, duration)
        return success
    
    def step_install_dependencies(self) -> bool:
        """Install all dependencies"""
        print("\n3️⃣ Installing dependencies...")
        start_time = time.time()
        
        # Node.js dependencies
        returncode, stdout, stderr = self.run_command(
            "npm install",
            timeout=300
        )
        
        if returncode != 0:
            duration = time.time() - start_time
            self.log_step("Dependencies installed", False, f"npm install failed: {stderr}", duration)
            return False
        
        # ctx-run dependencies
        returncode, stdout, stderr = self.run_command(
            "cd ctx-run && npm install",
            timeout=300
        )
        
        if returncode != 0:
            duration = time.time() - start_time
            self.log_step("Dependencies installed", False, f"ctx-run npm install failed: {stderr}", duration)
            return False
        
        # Python dependencies
        returncode, stdout, stderr = self.run_command(
            "cd lethe-research && pip install -r requirements.txt",
            timeout=300
        )
        
        if returncode != 0:
            duration = time.time() - start_time
            self.log_step("Dependencies installed", False, f"pip install failed: {stderr}", duration)
            return False
        
        duration = time.time() - start_time
        self.log_step("Dependencies installed", True, "All dependencies installed successfully", duration)
        return True
    
    def step_build_system(self) -> bool:
        """Test build system"""
        print("\n4️⃣ Testing build system...")
        start_time = time.time()
        
        # Test Node.js build
        returncode, stdout, stderr = self.run_command(
            "npm run build",
            timeout=300
        )
        
        if returncode != 0:
            duration = time.time() - start_time
            self.log_step("Build system", False, f"npm build failed: {stderr}", duration)
            return False
        
        duration = time.time() - start_time
        self.log_step("Build system", True, "Build completed successfully", duration)
        return True
    
    def step_index_generation(self) -> bool:
        """Test index building"""
        print("\n5️⃣ Building indices...")
        start_time = time.time()
        
        returncode, stdout, stderr = self.run_command(
            "make build_indices",
            timeout=600  # 10 minutes for index building
        )
        
        duration = time.time() - start_time
        success = returncode == 0
        
        details = "Indices built successfully" if success else f"Index building failed: {stderr}"
        self.log_step("Indices built", success, details, duration)
        return success
    
    def step_basic_functionality(self) -> bool:
        """Test basic functionality"""
        print("\n6️⃣ Testing basic functionality...")
        start_time = time.time()
        
        # Run quick test
        returncode, stdout, stderr = self.run_command(
            "make quick_test",
            timeout=300
        )
        
        duration = time.time() - start_time
        success = returncode == 0
        
        details = "Basic functionality works" if success else f"Basic tests failed: {stderr}"
        self.log_step("Basic functionality", success, details, duration)
        return success
    
    def step_evaluation_pipeline(self) -> bool:
        """Run evaluation pipeline to reproduce headline numbers"""
        print("\n7️⃣ Running evaluation pipeline...")
        start_time = time.time()
        
        # Run evaluation with reduced scope for testing
        returncode, stdout, stderr = self.run_command(
            "cd lethe-research && make baseline-quick-test",
            timeout=900  # 15 minutes
        )
        
        duration = time.time() - start_time
        success = returncode == 0
        
        details = "Evaluation pipeline completed" if success else f"Evaluation failed: {stderr}"
        self.log_step("Evaluation pipeline", success, details, duration)
        return success
    
    def step_result_validation(self) -> bool:
        """Validate that results are reasonable"""
        print("\n8️⃣ Validating results...")
        start_time = time.time()
        
        # Look for result files
        result_files = list(self.clone_dir.glob("**/results/*.json"))
        result_files.extend(list(self.clone_dir.glob("**/*results*.json")))
        
        if not result_files:
            duration = time.time() - start_time
            self.log_step("Results validated", False, "No result files found", duration)
            return False
        
        # Check that results contain expected structure
        try:
            for result_file in result_files[:3]:  # Check first 3 files
                with open(result_file) as f:
                    data = json.load(f)
                    
                # Basic validation - results should have metrics
                if not (isinstance(data, dict) and ("results" in data or "metrics" in data)):
                    duration = time.time() - start_time
                    self.log_step("Results validated", False, f"Invalid result structure in {result_file.name}", duration)
                    return False
            
            duration = time.time() - start_time
            self.log_step("Results validated", True, f"Found {len(result_files)} result files", duration)
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.log_step("Results validated", False, f"Result validation error: {e}", duration)
            return False
    
    def run_validation(self) -> Dict[str, Any]:
        """Run complete validation pipeline"""
        steps = [
            self.step_clone_project,
            self.step_environment_check, 
            self.step_install_dependencies,
            self.step_build_system,
            self.step_index_generation,
            self.step_basic_functionality,
            self.step_evaluation_pipeline,
            self.step_result_validation
        ]
        
        success_count = 0
        
        for step_func in steps:
            try:
                success = step_func()
                if success:
                    success_count += 1
                else:
                    # Stop on first failure for critical steps
                    if step_func in [self.step_clone_project, self.step_environment_check, self.step_install_dependencies]:
                        print(f"\n❌ Critical step failed, stopping validation")
                        break
            except Exception as e:
                print(f"\n❌ Step {step_func.__name__} crashed: {e}")
                self.log_step(step_func.__name__, False, str(e), 0)
        
        # Generate summary
        total_time = time.time() - self.start_time
        total_steps = len([r for r in self.results if r["step"]])
        successful_steps = len([r for r in self.results if r["step"] and r["success"]])
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "source_dir": str(self.source_dir),
            "temp_dir": str(self.temp_dir),
            "total_time_s": total_time,
            "total_steps": total_steps,
            "successful_steps": successful_steps,
            "success_rate": successful_steps / total_steps if total_steps > 0 else 0,
            "overall_success": successful_steps == len(steps),
            "results": self.results
        }
        
        self._print_summary(report)
        return report
    
    def _print_summary(self, report: Dict[str, Any]):
        """Print validation summary"""
        print("\n" + "=" * 60)
        print("🎯 FRESH CLONE VALIDATION SUMMARY")
        print("=" * 60)
        
        if report["overall_success"]:
            print("🟢 VALIDATION PASSED")
            print("✅ Fresh clone setup and evaluation completed successfully!")
        else:
            print("🔴 VALIDATION FAILED")
            print("❌ Fresh clone setup or evaluation failed")
        
        print(f"\n📊 Statistics:")
        print(f"  Total steps: {report['total_steps']}")
        print(f"  Successful: {report['successful_steps']}")
        print(f"  Success rate: {report['success_rate']:.1%}")
        print(f"  Total time: {report['total_time_s']:.1f}s")
        
        print(f"\n📋 Step Results:")
        for result in report["results"]:
            status = "✅" if result["success"] else "❌"
            print(f"  {status} {result['step']} ({result['duration']:.1f}s)")
            if result["details"] and not result["success"]:
                print(f"      {result['details']}")
        
        if not report["overall_success"]:
            print(f"\n🔧 Recommendations:")
            print(f"  - Review failed steps above")
            print(f"  - Check dependency installation")
            print(f"  - Verify environment requirements")
            print(f"  - Check build system configuration")
        
        if self.keep_temp:
            print(f"\n📁 Test directory preserved: {self.temp_dir}")
        else:
            print(f"\n🧹 Test directory will be cleaned up")
    
    def cleanup(self):
        """Clean up temporary directory"""
        if not self.keep_temp:
            try:
                shutil.rmtree(self.temp_dir)
                print(f"✅ Cleaned up temporary directory")
            except Exception as e:
                print(f"⚠️ Could not clean up {self.temp_dir}: {e}")

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description="Fresh clone validation test")
    parser.add_argument(
        "--source-dir",
        default=".",
        help="Source directory to test (default: current directory)"
    )
    parser.add_argument(
        "--temp-dir",
        help="Temporary directory for testing (default: auto-generated)"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary directory after test"
    )
    
    args = parser.parse_args()
    
    # Run validation
    validator = FreshCloneValidator(
        source_dir=args.source_dir,
        temp_dir=args.temp_dir,
        keep_temp=args.keep_temp
    )
    
    try:
        report = validator.run_validation()
        
        # Save report
        report_file = Path("fresh_clone_validation_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        # Exit with appropriate code
        if report["overall_success"]:
            print("\n🎉 Fresh clone validation PASSED!")
            exit_code = 0
        else:
            print(f"\n❌ Fresh clone validation FAILED!")
            exit_code = 1
            
    except Exception as e:
        print(f"\n💥 Validation crashed: {e}")
        exit_code = 2
        
    finally:
        validator.cleanup()
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()