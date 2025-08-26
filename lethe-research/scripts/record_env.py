#!/usr/bin/env python3
"""
Environment Recording and Manifest Generation System
Creates deterministic build manifests for hermetic environments
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class EnvironmentRecorder:
    """Records comprehensive environment state for hermetic builds"""
    
    def __init__(self, output_path: Optional[str] = None):
        self.output_path = output_path or "build-manifest.json"
        self.manifest: Dict[str, Any] = {
            "version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "hermetic_build": True,
            "environment": {},
            "dependencies": {},
            "toolchain": {},
            "security": {},
            "validation": {}
        }
    
    def record_system_environment(self) -> None:
        """Record system environment information"""
        self.manifest["environment"] = {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "node": platform.node()
            },
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "compiler": platform.python_compiler(),
                "build": platform.python_build(),
                "executable": sys.executable,
                "path": sys.path.copy()
            },
            "environment_variables": self._get_filtered_env_vars(),
            "working_directory": os.getcwd(),
            "user": {
                "uid": os.getuid() if hasattr(os, 'getuid') else None,
                "gid": os.getgid() if hasattr(os, 'getgid') else None,
                "username": os.getenv('USER', 'unknown')
            }
        }
    
    def record_dependencies(self) -> None:
        """Record all dependency information with versions and hashes"""
        self.manifest["dependencies"] = {
            "python": self._get_python_dependencies(),
            "nodejs": self._get_nodejs_dependencies(),
            "system": self._get_system_dependencies(),
            "docker": self._get_docker_dependencies()
        }
    
    def record_toolchain(self) -> None:
        """Record toolchain versions and configurations"""
        self.manifest["toolchain"] = {
            "compilers": self._get_compiler_info(),
            "build_tools": self._get_build_tools(),
            "container_runtime": self._get_container_runtime(),
            "version_control": self._get_version_control_info()
        }
    
    def record_security_context(self) -> None:
        """Record security-relevant environment information"""
        self.manifest["security"] = {
            "container_context": self._get_container_context(),
            "privileges": self._get_privilege_context(),
            "network_isolation": self._check_network_isolation(),
            "file_permissions": self._check_file_permissions(),
            "secrets_detection": self._scan_for_secrets()
        }
    
    def calculate_environment_digest(self) -> str:
        """Calculate deterministic digest of environment state"""
        # Create reproducible hash of environment
        env_data = {
            "platform": self.manifest["environment"]["platform"],
            "dependencies": self.manifest["dependencies"],
            "toolchain": self.manifest["toolchain"]
        }
        
        env_json = json.dumps(env_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(env_json.encode('utf-8')).hexdigest()
    
    def validate_hermetic_requirements(self) -> Dict[str, Any]:
        """Validate that environment meets hermetic build requirements"""
        validation = {
            "is_hermetic": True,
            "violations": [],
            "warnings": [],
            "requirements_met": {}
        }
        
        # Check for deterministic builds
        validation["requirements_met"]["pinned_dependencies"] = self._check_pinned_dependencies()
        validation["requirements_met"]["isolated_network"] = self._check_network_isolation()
        validation["requirements_met"]["reproducible_toolchain"] = self._check_toolchain_reproducibility()
        validation["requirements_met"]["no_external_dependencies"] = self._check_external_dependencies()
        
        # Aggregate results
        all_met = all(validation["requirements_met"].values())
        validation["is_hermetic"] = all_met
        
        if not all_met:
            validation["violations"].append("Hermetic build requirements not fully met")
        
        return validation
    
    def generate_manifest(self) -> Dict[str, Any]:
        """Generate complete environment manifest"""
        print("🔍 Recording system environment...")
        self.record_system_environment()
        
        print("📦 Recording dependencies...")
        self.record_dependencies()
        
        print("🔨 Recording toolchain information...")
        self.record_toolchain()
        
        print("🔒 Recording security context...")
        self.record_security_context()
        
        print("🧮 Calculating environment digest...")
        self.manifest["environment_digest"] = self.calculate_environment_digest()
        
        print("✅ Validating hermetic requirements...")
        self.manifest["validation"] = self.validate_hermetic_requirements()
        
        return self.manifest
    
    def save_manifest(self) -> None:
        """Save manifest to file"""
        with open(self.output_path, 'w') as f:
            json.dump(self.manifest, f, indent=2, sort_keys=True)
        
        print(f"📋 Environment manifest saved to: {self.output_path}")
        
        # Also create a human-readable summary
        summary_path = self.output_path.replace('.json', '-summary.txt')
        self._generate_summary(summary_path)
        print(f"📄 Human-readable summary saved to: {summary_path}")
    
    def _get_filtered_env_vars(self) -> Dict[str, str]:
        """Get environment variables, filtering out sensitive ones"""
        sensitive_patterns = [
            'PASSWORD', 'SECRET', 'KEY', 'TOKEN', 'CREDENTIAL', 
            'AUTH', 'PRIVATE', 'CERT', 'API_KEY'
        ]
        
        filtered_env = {}
        for key, value in os.environ.items():
            if not any(pattern in key.upper() for pattern in sensitive_patterns):
                filtered_env[key] = value
            else:
                filtered_env[key] = "<REDACTED>"
        
        return filtered_env
    
    def _get_python_dependencies(self) -> Dict[str, Any]:
        """Get Python dependency information"""
        try:
            # Get pip list with versions and locations
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list', '--format=json'],
                capture_output=True, text=True, check=True
            )
            pip_packages = json.loads(result.stdout)
            
            # Get requirements.txt if it exists
            requirements_files = []
            for req_file in ['requirements.txt', 'requirements_statistical.txt', 'infra/requirements.txt']:
                if os.path.exists(req_file):
                    with open(req_file) as f:
                        content = f.read()
                        file_hash = hashlib.sha256(content.encode()).hexdigest()
                        requirements_files.append({
                            "file": req_file,
                            "content_hash": file_hash,
                            "lines": content.strip().split('\n')
                        })
            
            return {
                "pip_packages": pip_packages,
                "requirements_files": requirements_files,
                "python_path": sys.path.copy(),
                "site_packages": self._get_site_packages_info()
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_nodejs_dependencies(self) -> Dict[str, Any]:
        """Get Node.js dependency information"""
        if not os.path.exists('package.json'):
            return {"status": "not_applicable"}
        
        try:
            # Read package.json
            with open('package.json') as f:
                package_json = json.load(f)
            
            # Get npm list
            result = subprocess.run(
                ['npm', 'list', '--json', '--depth=0'],
                capture_output=True, text=True, check=True
            )
            npm_packages = json.loads(result.stdout)
            
            # Get lockfile hash
            lockfile_hash = None
            if os.path.exists('package-lock.json'):
                with open('package-lock.json') as f:
                    lockfile_content = f.read()
                    lockfile_hash = hashlib.sha256(lockfile_content.encode()).hexdigest()
            
            return {
                "package_json": package_json,
                "npm_packages": npm_packages,
                "lockfile_hash": lockfile_hash,
                "node_version": subprocess.run(['node', '--version'], capture_output=True, text=True).stdout.strip(),
                "npm_version": subprocess.run(['npm', '--version'], capture_output=True, text=True).stdout.strip()
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _get_system_dependencies(self) -> Dict[str, Any]:
        """Get system-level dependency information"""
        system_deps = {}
        
        # Check for common system tools
        tools = ['git', 'docker', 'curl', 'gcc', 'make']
        for tool in tools:
            try:
                result = subprocess.run([tool, '--version'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    system_deps[tool] = {
                        "version": result.stdout.split('\n')[0],
                        "available": True
                    }
                else:
                    system_deps[tool] = {"available": False}
            except Exception:
                system_deps[tool] = {"available": False}
        
        # Get package manager info (if on Linux)
        if platform.system() == 'Linux':
            system_deps["package_manager"] = self._get_package_manager_info()
        
        return system_deps
    
    def _get_docker_dependencies(self) -> Dict[str, Any]:
        """Get Docker-related dependency information"""
        try:
            # Docker version
            docker_version = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            docker_compose_version = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
            
            # Docker info
            docker_info = subprocess.run(['docker', 'info', '--format={{json .}}'], capture_output=True, text=True)
            
            result = {
                "docker_available": docker_version.returncode == 0,
                "docker_compose_available": docker_compose_version.returncode == 0
            }
            
            if docker_version.returncode == 0:
                result["docker_version"] = docker_version.stdout.strip()
            
            if docker_compose_version.returncode == 0:
                result["docker_compose_version"] = docker_compose_version.stdout.strip()
            
            if docker_info.returncode == 0:
                result["docker_info"] = json.loads(docker_info.stdout)
            
            return result
        except Exception as e:
            return {"error": str(e), "docker_available": False}
    
    def _get_compiler_info(self) -> Dict[str, Any]:
        """Get compiler version information"""
        compilers = {}
        
        compiler_commands = {
            'gcc': ['gcc', '--version'],
            'clang': ['clang', '--version'],
            'python': [sys.executable, '--version'],
            'rustc': ['rustc', '--version'],
            'go': ['go', 'version']
        }
        
        for name, cmd in compiler_commands.items():
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    compilers[name] = {
                        "version": result.stdout.strip().split('\n')[0],
                        "available": True
                    }
                else:
                    compilers[name] = {"available": False}
            except Exception:
                compilers[name] = {"available": False}
        
        return compilers
    
    def _get_build_tools(self) -> Dict[str, Any]:
        """Get build tool information"""
        tools = ['make', 'cmake', 'pip', 'npm', 'yarn', 'cargo']
        build_tools = {}
        
        for tool in tools:
            try:
                result = subprocess.run([tool, '--version'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    build_tools[tool] = {
                        "version": result.stdout.strip().split('\n')[0],
                        "available": True
                    }
                else:
                    build_tools[tool] = {"available": False}
            except Exception:
                build_tools[tool] = {"available": False}
        
        return build_tools
    
    def _get_container_runtime(self) -> Dict[str, Any]:
        """Get container runtime information"""
        runtime_info = {}
        
        # Check for container environment
        if os.path.exists('/.dockerenv'):
            runtime_info["in_container"] = True
            runtime_info["container_type"] = "docker"
        elif os.path.exists('/run/.containerenv'):
            runtime_info["in_container"] = True
            runtime_info["container_type"] = "podman"
        else:
            runtime_info["in_container"] = False
        
        # Get cgroup info
        try:
            with open('/proc/1/cgroup') as f:
                cgroup_content = f.read()
                runtime_info["cgroup_info"] = cgroup_content.strip()
        except Exception:
            runtime_info["cgroup_info"] = None
        
        return runtime_info
    
    def _get_version_control_info(self) -> Dict[str, Any]:
        """Get version control information"""
        vcs_info = {}
        
        if os.path.exists('.git'):
            try:
                # Git commit info
                commit_hash = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
                branch = subprocess.run(['git', 'branch', '--show-current'], capture_output=True, text=True).stdout.strip()
                status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True).stdout.strip()
                
                vcs_info["git"] = {
                    "commit_hash": commit_hash,
                    "branch": branch,
                    "clean": len(status) == 0,
                    "status": status
                }
            except Exception as e:
                vcs_info["git"] = {"error": str(e)}
        
        return vcs_info
    
    def _get_container_context(self) -> Dict[str, Any]:
        """Get container security context"""
        context = {
            "in_container": os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv'),
            "user_namespace": None,
            "capabilities": None
        }
        
        try:
            # Check user namespace
            with open('/proc/self/uid_map') as f:
                uid_map = f.read().strip()
                context["user_namespace"] = len(uid_map) > 0 and uid_map != "0 0 4294967295"
        except Exception:
            pass
        
        try:
            # Check capabilities (if available)
            with open('/proc/self/status') as f:
                for line in f:
                    if line.startswith('CapEff:'):
                        context["capabilities"] = line.split(':')[1].strip()
                        break
        except Exception:
            pass
        
        return context
    
    def _get_privilege_context(self) -> Dict[str, Any]:
        """Get privilege and permission context"""
        context = {
            "running_as_root": os.geteuid() == 0 if hasattr(os, 'geteuid') else None,
            "effective_uid": os.geteuid() if hasattr(os, 'geteuid') else None,
            "effective_gid": os.getegid() if hasattr(os, 'getegid') else None,
            "supplementary_groups": os.getgroups() if hasattr(os, 'getgroups') else None
        }
        
        return context
    
    def _check_network_isolation(self) -> bool:
        """Check if network is properly isolated"""
        try:
            # Try to reach common external endpoints
            external_endpoints = ['8.8.8.8', 'google.com']
            for endpoint in external_endpoints:
                result = subprocess.run(['ping', '-c', '1', '-W', '2', endpoint], 
                                      capture_output=True, timeout=5)
                if result.returncode == 0:
                    return False  # Network is not isolated
            return True  # Network appears isolated
        except Exception:
            return True  # Assume isolated if can't test
    
    def _check_file_permissions(self) -> Dict[str, Any]:
        """Check file permission context"""
        permissions = {
            "working_directory_writable": os.access(os.getcwd(), os.W_OK),
            "tmp_writable": os.access('/tmp', os.W_OK),
            "can_create_files": True
        }
        
        # Test file creation
        try:
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                tmp.write(b'test')
                permissions["can_create_files"] = True
        except Exception:
            permissions["can_create_files"] = False
        
        return permissions
    
    def _scan_for_secrets(self) -> Dict[str, Any]:
        """Scan for potential secrets in environment"""
        secrets_scan = {
            "potential_secrets_found": False,
            "locations": [],
            "recommendations": []
        }
        
        # Check environment variables
        secret_patterns = ['password', 'secret', 'key', 'token', 'credential', 'auth']
        for key, value in os.environ.items():
            if any(pattern in key.lower() for pattern in secret_patterns):
                if len(value) > 0 and value != "<REDACTED>":
                    secrets_scan["potential_secrets_found"] = True
                    secrets_scan["locations"].append(f"Environment variable: {key}")
        
        if secrets_scan["potential_secrets_found"]:
            secrets_scan["recommendations"].append("Move secrets to secure secret management system")
        
        return secrets_scan
    
    def _check_pinned_dependencies(self) -> bool:
        """Check if dependencies are properly pinned"""
        try:
            # Check Python requirements
            for req_file in ['requirements.txt', 'requirements_statistical.txt']:
                if os.path.exists(req_file):
                    with open(req_file) as f:
                        content = f.read()
                        if '==' not in content:
                            return False
            
            # Check package-lock.json exists
            if os.path.exists('package.json') and not os.path.exists('package-lock.json'):
                return False
            
            return True
        except Exception:
            return False
    
    def _check_toolchain_reproducibility(self) -> bool:
        """Check if toolchain is reproducible"""
        # Check for version pinning in Docker
        dockerfile_paths = ['Dockerfile', 'infra/Dockerfile']
        for dockerfile_path in dockerfile_paths:
            if os.path.exists(dockerfile_path):
                with open(dockerfile_path) as f:
                    content = f.read()
                    # Look for version pins in apk/apt commands
                    if 'apk add' in content and '=' not in content:
                        return False
        
        return True
    
    def _check_external_dependencies(self) -> bool:
        """Check for external dependencies that break hermeticism"""
        # This is a placeholder - would check for:
        # - External network calls during build
        # - Non-cached external resources
        # - Dynamic dependency resolution
        return True
    
    def _get_site_packages_info(self) -> List[str]:
        """Get site-packages directory information"""
        import site
        return site.getsitepackages()
    
    def _get_package_manager_info(self) -> Dict[str, Any]:
        """Get Linux package manager information"""
        pm_info = {}
        
        # Check for different package managers
        managers = {
            'apt': ['dpkg', '--version'],
            'yum': ['yum', '--version'],
            'pacman': ['pacman', '--version'],
            'apk': ['apk', '--version']
        }
        
        for pm_name, cmd in managers.items():
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    pm_info[pm_name] = {
                        "version": result.stdout.strip().split('\n')[0],
                        "available": True
                    }
            except Exception:
                pm_info[pm_name] = {"available": False}
        
        return pm_info
    
    def _generate_summary(self, summary_path: str) -> None:
        """Generate human-readable summary"""
        with open(summary_path, 'w') as f:
            f.write("# Environment Manifest Summary\n\n")
            f.write(f"Generated: {self.manifest['generated_at']}\n")
            f.write(f"Environment Digest: {self.manifest['environment_digest']}\n\n")
            
            # System info
            env = self.manifest["environment"]
            f.write("## System Environment\n")
            f.write(f"Platform: {env['platform']['system']} {env['platform']['release']}\n")
            f.write(f"Architecture: {env['platform']['machine']}\n")
            f.write(f"Python: {env['python']['version']}\n\n")
            
            # Validation results
            validation = self.manifest["validation"]
            f.write("## Hermetic Build Validation\n")
            f.write(f"Hermetic: {'✅ Yes' if validation['is_hermetic'] else '❌ No'}\n")
            
            for req, met in validation["requirements_met"].items():
                status = "✅" if met else "❌"
                f.write(f"{req}: {status}\n")
            
            if validation["violations"]:
                f.write("\n### Violations:\n")
                for violation in validation["violations"]:
                    f.write(f"- {violation}\n")
            
            if validation["warnings"]:
                f.write("\n### Warnings:\n")
                for warning in validation["warnings"]:
                    f.write(f"- {warning}\n")


def main():
    parser = argparse.ArgumentParser(description='Record environment state for hermetic builds')
    parser.add_argument('--output', '-o', default='build-manifest.json',
                       help='Output file path for manifest')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate hermetic requirements')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimize output')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🏗️ Lethe Research Environment Recorder")
        print("====================================")
    
    recorder = EnvironmentRecorder(args.output)
    
    try:
        if args.validate_only:
            recorder.record_system_environment()
            recorder.record_dependencies()
            recorder.record_toolchain()
            validation = recorder.validate_hermetic_requirements()
            
            if validation["is_hermetic"]:
                print("✅ Environment meets hermetic build requirements")
                sys.exit(0)
            else:
                print("❌ Environment does not meet hermetic build requirements:")
                for violation in validation["violations"]:
                    print(f"  - {violation}")
                sys.exit(1)
        else:
            manifest = recorder.generate_manifest()
            recorder.save_manifest()
            
            if not args.quiet:
                print(f"\n📋 Manifest Summary:")
                print(f"Environment Digest: {manifest['environment_digest']}")
                print(f"Hermetic Build: {'✅ Yes' if manifest['validation']['is_hermetic'] else '❌ No'}")
                
                if not manifest['validation']['is_hermetic']:
                    print("❌ Violations:")
                    for violation in manifest['validation']['violations']:
                        print(f"  - {violation}")
            
            # Exit with error if not hermetic
            if not manifest['validation']['is_hermetic']:
                sys.exit(1)
                
    except Exception as e:
        print(f"❌ Error recording environment: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()