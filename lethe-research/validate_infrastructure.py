#!/usr/bin/env python3
"""
Comprehensive Infrastructure Validation Script
Validates the complete Lethe Research infrastructure orchestration
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def run_command(cmd: List[str], timeout: int = 300, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command with timeout and logging"""
    print(f"🔄 Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=check)
        if result.returncode != 0:
            print(f"⚠️  Command failed with code {result.returncode}")
            print(f"   stdout: {result.stdout[:200]}...")
            print(f"   stderr: {result.stderr[:200]}...")
        return result
    except subprocess.TimeoutExpired:
        print(f"⏰ Command timed out after {timeout} seconds")
        raise
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {e}")
        raise


def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists and report"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} missing: {filepath}")
        return False


def validate_docker_setup() -> bool:
    """Validate Docker and Docker Compose setup"""
    print("\n🐳 Validating Docker setup...")
    
    checks_passed = 0
    total_checks = 4
    
    # Check Docker installation
    try:
        result = run_command(['docker', '--version'])
        print(f"✅ Docker version: {result.stdout.strip()}")
        checks_passed += 1
    except Exception:
        print("❌ Docker not available")
    
    # Check Docker daemon
    try:
        run_command(['docker', 'info'], timeout=10)
        print("✅ Docker daemon is running")
        checks_passed += 1
    except Exception:
        print("❌ Docker daemon not accessible")
    
    # Check Docker Compose
    try:
        result = run_command(['docker-compose', '--version'])
        print(f"✅ Docker Compose version: {result.stdout.strip()}")
        checks_passed += 1
    except Exception:
        print("❌ Docker Compose not available")
    
    # Check for Dockerfile
    if check_file_exists('infra/Dockerfile', 'Multi-stage Dockerfile'):
        checks_passed += 1
    
    success = checks_passed == total_checks
    print(f"🐳 Docker validation: {checks_passed}/{total_checks} checks passed")
    return success


def validate_infrastructure_files() -> bool:
    """Validate presence of infrastructure files"""
    print("\n🏗️ Validating infrastructure files...")
    
    required_files = [
        ('infra/docker-compose.yml', 'Docker Compose orchestration'),
        ('infra/Dockerfile', 'Container build definition'),
        ('infra/requirements.txt', 'Python dependencies'),
        ('infra/package.json', 'Node.js dependencies'),
        ('.github/workflows/ci-full.yml', 'CI/CD pipeline'),
        ('.github/workflows/validation.yml', 'Quality gate validation'),
        ('scripts/record_env.py', 'Environment recording'),
        ('scripts/spinup_smoke.sh', 'Smoke testing suite'),
        ('scripts/sign_transcript.py', 'Boot transcript signing'),
        ('scripts/bundle_artifact.sh', 'Artifact bundling'),
        ('scripts/apply_gates.py', 'Quality gate enforcement')
    ]
    
    checks_passed = 0
    for filepath, description in required_files:
        if check_file_exists(filepath, description):
            checks_passed += 1
    
    total_checks = len(required_files)
    success = checks_passed == total_checks
    print(f"🏗️ Infrastructure files: {checks_passed}/{total_checks} found")
    return success


def validate_security_config() -> bool:
    """Validate security configuration"""
    print("\n🔒 Validating security configuration...")
    
    security_files = [
        ('infra/security/semgrep-rules/custom-security.yml', 'Custom security rules'),
        ('infra/config/smoke-tests.json', 'Smoke test configuration')
    ]
    
    checks_passed = 0
    for filepath, description in security_files:
        if check_file_exists(filepath, description):
            checks_passed += 1
    
    # Check security in Docker Compose
    if os.path.exists('infra/docker-compose.yml'):
        with open('infra/docker-compose.yml') as f:
            compose_content = f.read()
        
        security_features = [
            ('no-new-privileges:true', 'No new privileges'),
            ('cap_drop:', 'Capability dropping'),
            ('read_only: true', 'Read-only containers'),
            ('USER ', 'Non-root user')
        ]
        
        for feature, description in security_features:
            if feature in compose_content:
                print(f"✅ {description} configured")
                checks_passed += 1
            else:
                print(f"⚠️  {description} not found in compose")
    
    total_checks = len(security_files) + len(security_features)
    print(f"🔒 Security validation: {checks_passed}/{total_checks} features found")
    return checks_passed >= (total_checks * 0.8)  # Allow 80% pass rate


def test_environment_recording() -> bool:
    """Test environment recording functionality"""
    print("\n📋 Testing environment recording...")
    
    try:
        # Test environment recording
        result = run_command(['python3', 'scripts/record_env.py', '--output=test-manifest.json'], timeout=60)
        
        if os.path.exists('test-manifest.json'):
            with open('test-manifest.json') as f:
                manifest = json.load(f)
            
            required_sections = ['version', 'environment', 'dependencies', 'toolchain', 'environment_digest']
            missing_sections = [section for section in required_sections if section not in manifest]
            
            if not missing_sections:
                print("✅ Environment manifest generated successfully")
                # Cleanup
                os.remove('test-manifest.json')
                return True
            else:
                print(f"❌ Missing sections in manifest: {missing_sections}")
                return False
        else:
            print("❌ Environment manifest file not created")
            return False
            
    except Exception as e:
        print(f"❌ Environment recording failed: {e}")
        return False


def test_smoke_test_script() -> bool:
    """Test smoke test script functionality"""
    print("\n💨 Testing smoke test script...")
    
    try:
        # Test smoke test validation only
        result = run_command(['bash', 'scripts/spinup_smoke.sh', '--validate-only'], timeout=120)
        
        if result.returncode == 0:
            print("✅ Smoke test validation passed")
            return True
        else:
            print("❌ Smoke test validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Smoke test script failed: {e}")
        return False


def test_quality_gates() -> bool:
    """Test quality gate enforcement"""
    print("\n🚦 Testing quality gate enforcement...")
    
    try:
        # Test dataset quality gates (with relaxed requirements for testing)
        result = run_command([
            'python3', 'scripts/apply_gates.py',
            '--type=dataset', 
            '--min-queries=1',
            '--min-domains=1',
            '--min-iaa-kappa=0.1'
        ], timeout=60, check=False)
        
        if result.returncode in [0, 1]:  # Allow both pass and fail
            print("✅ Quality gate system functional")
            return True
        else:
            print("❌ Quality gate system error")
            return False
            
    except Exception as e:
        print(f"❌ Quality gate testing failed: {e}")
        return False


def test_artifact_bundling() -> bool:
    """Test artifact bundling functionality"""
    print("\n📦 Testing artifact bundling...")
    
    try:
        # Test bundle creation (dry run)
        result = run_command([
            'bash', 'scripts/bundle_artifact.sh',
            '--version=test',
            '--no-quality-gates',
            '--include-tests=false',
            '--output=test-bundle.tar.gz'
        ], timeout=180, check=False)
        
        success = result.returncode == 0
        
        # Cleanup test bundle if created
        if os.path.exists('test-bundle.tar.gz'):
            os.remove('test-bundle.tar.gz')
        if os.path.exists('test-bundle.tar.gz.sha256'):
            os.remove('test-bundle.tar.gz.sha256')
        if os.path.exists('test-bundle.tar.gz.md5'):
            os.remove('test-bundle.tar.gz.md5')
        
        # Cleanup bundle directory
        import shutil
        bundle_dirs = [d for d in os.listdir('.') if d.startswith('lethe-research-artifact-')]
        for bundle_dir in bundle_dirs:
            if os.path.isdir(bundle_dir):
                shutil.rmtree(bundle_dir)
        
        if success:
            print("✅ Artifact bundling functional")
            return True
        else:
            print("❌ Artifact bundling failed")
            return False
            
    except Exception as e:
        print(f"❌ Artifact bundling test failed: {e}")
        return False


def validate_ci_cd_config() -> bool:
    """Validate CI/CD configuration"""
    print("\n🔄 Validating CI/CD configuration...")
    
    checks_passed = 0
    total_checks = 0
    
    # Check GitHub Actions workflows
    workflows = [
        '.github/workflows/ci-full.yml',
        '.github/workflows/validation.yml'
    ]
    
    for workflow_file in workflows:
        total_checks += 1
        if os.path.exists(workflow_file):
            print(f"✅ Workflow found: {workflow_file}")
            checks_passed += 1
            
            # Basic validation of workflow structure
            with open(workflow_file) as f:
                content = f.read()
            
            required_elements = ['on:', 'jobs:', 'runs-on:', 'steps:']
            for element in required_elements:
                if element in content:
                    checks_passed += 1
                total_checks += 1
        else:
            print(f"❌ Workflow missing: {workflow_file}")
    
    success = checks_passed >= (total_checks * 0.9)  # 90% pass rate
    print(f"🔄 CI/CD validation: {checks_passed}/{total_checks} checks passed")
    return success


def generate_validation_report(results: Dict[str, bool]) -> Dict:
    """Generate comprehensive validation report"""
    report = {
        "validation_timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_status": "passed" if all(results.values()) else "failed",
        "results": results,
        "summary": {
            "total_validations": len(results),
            "passed_validations": sum(1 for result in results.values() if result),
            "failed_validations": sum(1 for result in results.values() if not result)
        }
    }
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Validate Lethe Research infrastructure orchestration')
    parser.add_argument('--output', '-o', help='Output file for validation report')
    parser.add_argument('--skip-docker', action='store_true', help='Skip Docker-related tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    print("🏗️  Lethe Research Infrastructure Validation")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Define validation steps
    validation_steps = [
        ('infrastructure_files', validate_infrastructure_files),
        ('security_config', validate_security_config),
        ('environment_recording', test_environment_recording),
        ('smoke_testing', test_smoke_test_script),
        ('quality_gates', test_quality_gates),
        ('artifact_bundling', test_artifact_bundling),
        ('ci_cd_config', validate_ci_cd_config)
    ]
    
    if not args.skip_docker:
        validation_steps.insert(0, ('docker_setup', validate_docker_setup))
    
    # Run validations
    results = {}
    for step_name, step_func in validation_steps:
        try:
            print(f"\n{'='*50}")
            result = step_func()
            results[step_name] = result
            
            if result:
                print(f"✅ {step_name.replace('_', ' ').title()} validation PASSED")
            else:
                print(f"❌ {step_name.replace('_', ' ').title()} validation FAILED")
                
        except KeyboardInterrupt:
            print(f"\n❌ Validation interrupted by user")
            sys.exit(130)
        except Exception as e:
            print(f"❌ Validation error in {step_name}: {e}")
            results[step_name] = False
    
    # Generate final report
    report = generate_validation_report(results)
    
    print(f"\n{'='*50}")
    print("📊 VALIDATION SUMMARY")
    print(f"{'='*50}")
    
    overall_status = report["overall_status"]
    status_icon = "✅" if overall_status == "passed" else "❌"
    
    print(f"Overall Status: {status_icon} {overall_status.upper()}")
    print(f"Passed: {report['summary']['passed_validations']}/{report['summary']['total_validations']}")
    
    if overall_status == "failed":
        failed_steps = [name for name, result in results.items() if not result]
        print(f"\nFailed validations:")
        for step in failed_steps:
            print(f"  ❌ {step.replace('_', ' ').title()}")
    
    # Save report if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Validation report saved to: {args.output}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if overall_status == "passed":
        print("\n🎉 Infrastructure validation PASSED!")
        print("✅ Your Lethe Research infrastructure is ready for deployment!")
        sys.exit(0)
    else:
        print("\n💥 Infrastructure validation FAILED!")
        print("❌ Please address the failed validations before deployment.")
        sys.exit(1)


if __name__ == '__main__':
    main()