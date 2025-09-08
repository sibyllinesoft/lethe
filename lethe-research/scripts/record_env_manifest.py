#!/usr/bin/env python3
"""
Environment Manifest Recording Script
Records complete environment state with cryptographic hashes
Part of Lethe Hermetic Infrastructure (B0)
"""

import json
import hashlib
import subprocess
import sys
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata
import socket
import getpass

def get_git_info():
    """Get current git commit and repository information"""
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], 
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        remote_url = subprocess.check_output(
            ['git', 'config', '--get', 'remote.origin.url'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        is_dirty = len(subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL
        )) > 0
        
        return {
            "commit_hash": commit_hash,
            "branch": branch,
            "remote_url": remote_url,
            "is_dirty": is_dirty,
            "short_hash": commit_hash[:8]
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "commit_hash": "unknown",
            "branch": "unknown", 
            "remote_url": "unknown",
            "is_dirty": True,
            "short_hash": "unknown"
        }

def get_python_environment():
    """Get comprehensive Python environment information"""
    installed_packages = {}
    try:
        for dist in importlib_metadata.distributions():
            requires = []
            if hasattr(dist, 'requires') and dist.requires:
                requires = list(dist.requires)
            
            installed_packages[dist.metadata['Name']] = {
                "version": dist.version,
                "location": str(dist.locate_file('')),
                "requires": requires
            }
    except Exception as e:
        print(f"Warning: Could not enumerate packages: {e}", file=sys.stderr)
    
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "compiler": platform.python_compiler(),
        "build": platform.python_build(),
        "executable": sys.executable,
        "path": sys.path,
        "prefix": sys.prefix,
        "exec_prefix": sys.exec_prefix,
        "installed_packages": installed_packages
    }

def get_system_information():
    """Get system hardware and OS information"""
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "architecture": platform.architecture(),
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count()
    }

def get_environment_variables():
    """Get relevant environment variables (filtered for security)"""
    # Only include safe environment variables
    safe_env_prefixes = [
        'PATH', 'PYTHONPATH', 'HOME', 'USER', 'SHELL',
        'LANG', 'LC_', 'TZ', 'CUDA_', 'OMP_', 'MKL_',
        'BLAS_', 'LAPACK_', 'NODE_', 'NPM_'
    ]
    
    filtered_env = {}
    for key, value in os.environ.items():
        if any(key.startswith(prefix) for prefix in safe_env_prefixes):
            filtered_env[key] = value
    
    return filtered_env

def get_cuda_information():
    """Get CUDA and GPU information if available"""
    cuda_info = {"available": False}
    
    try:
        # Try nvcc first
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            cuda_info["nvcc_version"] = result.stdout.strip()
            cuda_info["available"] = True
    except FileNotFoundError:
        pass
    
    try:
        # Try nvidia-smi
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', 
                               '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            cuda_info["nvidia_smi"] = result.stdout.strip()
            cuda_info["available"] = True
    except FileNotFoundError:
        pass
    
    return cuda_info

def get_node_environment():
    """Get Node.js environment information if available"""
    node_info = {"available": False}
    
    try:
        node_version = subprocess.check_output(
            ['node', '--version'], stderr=subprocess.DEVNULL
        ).decode().strip()
        
        npm_version = subprocess.check_output(
            ['npm', '--version'], stderr=subprocess.DEVNULL
        ).decode().strip()
        
        node_info = {
            "available": True,
            "node_version": node_version,
            "npm_version": npm_version
        }
        
        # Try to get package.json if it exists
        if os.path.exists('package.json'):
            with open('package.json', 'r') as f:
                node_info["package_json"] = json.load(f)
                
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return node_info

def calculate_file_hash(filepath):
    """Calculate SHA-256 hash of a file"""
    hasher = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Warning: Could not hash {filepath}: {e}", file=sys.stderr)
        return None

def get_critical_file_hashes():
    """Get hashes of critical configuration and dependency files"""
    critical_files = [
        'requirements.txt',
        'requirements_clean.txt', 
        'requirements_minimal.txt',
        'package.json',
        'package-lock.json',
        'Dockerfile',
        'docker-compose.yml',
        'pyproject.toml',
        'setup.py',
        'environment.yml',
        'Pipfile',
        'Pipfile.lock'
    ]
    
    file_hashes = {}
    for filename in critical_files:
        if os.path.exists(filename):
            file_hash = calculate_file_hash(filename)
            if file_hash:
                file_hashes[filename] = {
                    "sha256": file_hash,
                    "size": os.path.getsize(filename),
                    "mtime": os.path.getmtime(filename)
                }
    
    return file_hashes

def create_environment_manifest():
    """Create comprehensive environment manifest"""
    manifest = {
        "manifest_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "script": __file__,
            "version": "1.0.0"
        },
        "git_info": get_git_info(),
        "system": get_system_information(),
        "python": get_python_environment(),
        "environment_variables": get_environment_variables(),
        "cuda": get_cuda_information(),
        "nodejs": get_node_environment(),
        "critical_files": get_critical_file_hashes()
    }
    
    # Add manifest hash (excluding itself)
    manifest_str = json.dumps(manifest, sort_keys=True, indent=2)
    manifest["manifest_hash"] = hashlib.sha256(
        manifest_str.encode('utf-8')
    ).hexdigest()
    
    return manifest

def save_manifest(manifest, output_path):
    """Save manifest to file with proper formatting"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    
    print(f"Environment manifest saved to: {output_path}")
    print(f"Manifest hash: {manifest['manifest_hash']}")
    
    return str(output_path)

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = "artifacts/boot_env.json"
    
    print("Recording environment manifest...")
    
    try:
        manifest = create_environment_manifest()
        manifest_path = save_manifest(manifest, output_path)
        
        print(f"\nEnvironment Summary:")
        print(f"  Python: {manifest['python']['version']}")
        print(f"  System: {manifest['system']['platform']}")
        print(f"  Git: {manifest['git_info']['short_hash']} ({manifest['git_info']['branch']})")
        print(f"  CUDA: {'Available' if manifest['cuda']['available'] else 'Not Available'}")
        print(f"  Node.js: {'Available' if manifest['nodejs']['available'] else 'Not Available'}")
        print(f"  Packages: {len(manifest['python']['installed_packages'])}")
        print(f"  Critical Files: {len(manifest['critical_files'])}")
        
        return manifest_path
        
    except Exception as e:
        print(f"Error creating environment manifest: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()