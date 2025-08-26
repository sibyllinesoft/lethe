#!/usr/bin/env python3
"""
Asset Hash Generator and Validator
Generates and validates cryptographic hashes for all datasets and models
Part of Lethe Hermetic Infrastructure (B1)
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
from datetime import datetime
import argparse

class AssetHasher:
    """Handles cryptographic hashing and validation of assets"""
    
    def __init__(self):
        self.hash_algorithms = ["sha256", "md5"]
        self.processed_files = 0
        self.total_size = 0
        
    def calculate_file_hashes(self, filepath: Path, algorithms: List[str] = None) -> Dict[str, str]:
        """Calculate multiple hashes for a single file"""
        if algorithms is None:
            algorithms = self.hash_algorithms
            
        hashers = {alg: hashlib.new(alg) for alg in algorithms}
        
        try:
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(65536), b""):  # 64KB chunks
                    for hasher in hashers.values():
                        hasher.update(chunk)
                        
            return {alg: hasher.hexdigest() for alg, hasher in hashers.items()}
            
        except Exception as e:
            print(f"Error hashing {filepath}: {e}")
            return {}
    
    def hash_directory(self, directory: Path, exclude_patterns: List[str] = None) -> Dict:
        """Hash all files in a directory recursively"""
        if exclude_patterns is None:
            exclude_patterns = [
                "*.pyc", "__pycache__", ".git", "*.tmp", "*.temp",
                "*.log", ".DS_Store", "Thumbs.db"
            ]
        
        directory_info = {
            "path": str(directory),
            "total_files": 0,
            "total_size": 0,
            "files": {},
            "directories": {},
            "directory_hash": None
        }
        
        if not directory.exists():
            print(f"Directory does not exist: {directory}")
            return directory_info
            
        print(f"Hashing directory: {directory}")
        
        # Collect all files
        all_files = []
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                # Check exclude patterns
                skip = False
                for pattern in exclude_patterns:
                    if file_path.match(pattern):
                        skip = True
                        break
                        
                if not skip:
                    all_files.append(file_path)
        
        directory_info["total_files"] = len(all_files)
        
        # Hash all files with progress
        for i, file_path in enumerate(all_files):
            relative_path = file_path.relative_to(directory)
            
            # Progress indicator
            if i % 100 == 0 or i == len(all_files) - 1:
                print(f"  Progress: {i+1}/{len(all_files)} files")
            
            file_hashes = self.calculate_file_hashes(file_path)
            if file_hashes:
                file_size = file_path.stat().st_size
                directory_info["total_size"] += file_size
                
                directory_info["files"][str(relative_path)] = {
                    "size": file_size,
                    "modified": file_path.stat().st_mtime,
                    "hashes": file_hashes
                }
                
        # Calculate directory hash (hash of all file hashes)
        directory_info["directory_hash"] = self.calculate_directory_hash(directory_info["files"])
        
        print(f"  Completed: {directory_info['total_files']} files, {directory_info['total_size']} bytes")
        return directory_info
    
    def calculate_directory_hash(self, files_dict: Dict) -> str:
        """Calculate a single hash representing the entire directory"""
        # Create a sorted list of file paths and their primary hashes
        sorted_items = []
        for file_path, file_info in sorted(files_dict.items()):
            primary_hash = file_info["hashes"].get("sha256", "")
            sorted_items.append(f"{file_path}:{primary_hash}")
        
        # Hash the concatenated string
        content = "\n".join(sorted_items)
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def hash_benchmarks(self, benchmarks_dir: Path) -> Dict:
        """Hash all benchmark datasets"""
        benchmarks_info = {
            "type": "benchmarks",
            "base_path": str(benchmarks_dir),
            "datasets": {},
            "manifest_hash": None
        }
        
        if not benchmarks_dir.exists():
            print(f"Benchmarks directory does not exist: {benchmarks_dir}")
            return benchmarks_info
            
        print(f"\n=== Hashing Benchmark Datasets ===")
        
        # Hash each dataset subdirectory
        for dataset_dir in benchmarks_dir.iterdir():
            if dataset_dir.is_dir() and not dataset_dir.name.startswith('.'):
                print(f"\nHashing benchmark dataset: {dataset_dir.name}")
                benchmarks_info["datasets"][dataset_dir.name] = self.hash_directory(dataset_dir)
        
        # Calculate overall benchmarks hash
        benchmarks_content = json.dumps(benchmarks_info["datasets"], sort_keys=True)
        benchmarks_info["manifest_hash"] = hashlib.sha256(benchmarks_content.encode('utf-8')).hexdigest()
        
        return benchmarks_info
    
    def hash_models(self, models_dir: Path) -> Dict:
        """Hash all model checkpoints"""
        models_info = {
            "type": "models", 
            "base_path": str(models_dir),
            "models": {},
            "manifest_hash": None
        }
        
        if not models_dir.exists():
            print(f"Models directory does not exist: {models_dir}")
            return models_info
            
        print(f"\n=== Hashing Model Checkpoints ===")
        
        # Hash each model subdirectory
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir() and not model_dir.name.startswith('.'):
                print(f"\nHashing model: {model_dir.name}")
                models_info["models"][model_dir.name] = self.hash_directory(model_dir)
        
        # Calculate overall models hash
        models_content = json.dumps(models_info["models"], sort_keys=True)
        models_info["manifest_hash"] = hashlib.sha256(models_content.encode('utf-8')).hexdigest()
        
        return models_info
    
    def validate_hashes(self, manifest_file: Path) -> bool:
        """Validate assets against existing hash manifest"""
        if not manifest_file.exists():
            print(f"Hash manifest not found: {manifest_file}")
            return False
            
        print(f"Validating assets against: {manifest_file}")
        
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        validation_results = {
            "total_files": 0,
            "validated_files": 0,
            "failed_files": 0,
            "missing_files": 0,
            "errors": []
        }
        
        # Validate benchmarks
        if "benchmarks" in manifest:
            for dataset_name, dataset_info in manifest["benchmarks"]["datasets"].items():
                base_path = Path(dataset_info["path"])
                
                for file_path, file_info in dataset_info["files"].items():
                    validation_results["total_files"] += 1
                    full_path = base_path / file_path
                    
                    if not full_path.exists():
                        validation_results["missing_files"] += 1
                        validation_results["errors"].append(f"Missing file: {full_path}")
                        continue
                    
                    # Recalculate hash
                    current_hashes = self.calculate_file_hashes(full_path)
                    expected_hash = file_info["hashes"].get("sha256", "")
                    current_hash = current_hashes.get("sha256", "")
                    
                    if current_hash == expected_hash:
                        validation_results["validated_files"] += 1
                    else:
                        validation_results["failed_files"] += 1
                        validation_results["errors"].append(
                            f"Hash mismatch: {full_path}\n  Expected: {expected_hash}\n  Current: {current_hash}"
                        )
        
        # Validate models
        if "models" in manifest:
            for model_name, model_info in manifest["models"]["models"].items():
                base_path = Path(model_info["path"])
                
                for file_path, file_info in model_info["files"].items():
                    validation_results["total_files"] += 1
                    full_path = base_path / file_path
                    
                    if not full_path.exists():
                        validation_results["missing_files"] += 1
                        validation_results["errors"].append(f"Missing file: {full_path}")
                        continue
                    
                    # Recalculate hash
                    current_hashes = self.calculate_file_hashes(full_path)
                    expected_hash = file_info["hashes"].get("sha256", "")
                    current_hash = current_hashes.get("sha256", "")
                    
                    if current_hash == expected_hash:
                        validation_results["validated_files"] += 1
                    else:
                        validation_results["failed_files"] += 1
                        validation_results["errors"].append(
                            f"Hash mismatch: {full_path}\n  Expected: {expected_hash}\n  Current: {current_hash}"
                        )
        
        # Print validation summary
        print(f"\n=== Validation Results ===")
        print(f"Total files: {validation_results['total_files']}")
        print(f"Validated: {validation_results['validated_files']}")
        print(f"Failed: {validation_results['failed_files']}")
        print(f"Missing: {validation_results['missing_files']}")
        
        if validation_results['errors']:
            print(f"\nErrors:")
            for error in validation_results['errors'][:10]:  # Show first 10 errors
                print(f"  {error}")
            if len(validation_results['errors']) > 10:
                print(f"  ... and {len(validation_results['errors']) - 10} more errors")
        
        return validation_results['failed_files'] == 0 and validation_results['missing_files'] == 0
    
    def create_asset_manifest(self, benchmarks_dir: Path, models_dir: Path, output_file: Path) -> Dict:
        """Create comprehensive asset manifest with all hashes"""
        manifest = {
            "manifest_version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "generator": {
                "script": __file__,
                "version": "1.0.0"
            },
            "hash_algorithms": self.hash_algorithms,
            "benchmarks": self.hash_benchmarks(benchmarks_dir),
            "models": self.hash_models(models_dir)
        }
        
        # Calculate overall manifest hash
        manifest_content = json.dumps(
            {k: v for k, v in manifest.items() if k != "manifest_hash"}, 
            sort_keys=True
        )
        manifest["manifest_hash"] = hashlib.sha256(manifest_content.encode('utf-8')).hexdigest()
        
        # Save manifest
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        
        print(f"\n=== Asset Manifest Created ===")
        print(f"Manifest saved to: {output_file}")
        print(f"Manifest hash: {manifest['manifest_hash']}")
        
        # Summary
        total_datasets = len(manifest["benchmarks"]["datasets"])
        total_models = len(manifest["models"]["models"])
        total_files = sum(info["total_files"] for info in manifest["benchmarks"]["datasets"].values())
        total_files += sum(info["total_files"] for info in manifest["models"]["models"].values())
        total_size = sum(info["total_size"] for info in manifest["benchmarks"]["datasets"].values())
        total_size += sum(info["total_size"] for info in manifest["models"]["models"].values())
        
        print(f"Datasets: {total_datasets}")
        print(f"Models: {total_models}")
        print(f"Total files: {total_files}")
        print(f"Total size: {total_size / (1024*1024*1024):.2f} GB")
        
        return manifest

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Hash and validate IR assets")
    parser.add_argument("--benchmarks", default="datasets/benchmarks", 
                       help="Benchmarks directory path")
    parser.add_argument("--models", default="models/checkpoints",
                       help="Models directory path")
    parser.add_argument("--output", default="artifacts/asset_hashes.json",
                       help="Output manifest file")
    parser.add_argument("--validate", help="Validate against existing manifest file")
    
    args = parser.parse_args()
    
    hasher = AssetHasher()
    
    if args.validate:
        # Validation mode
        success = hasher.validate_hashes(Path(args.validate))
        sys.exit(0 if success else 1)
    else:
        # Generation mode
        print("Generating asset hash manifest...")
        
        benchmarks_dir = Path(args.benchmarks)
        models_dir = Path(args.models)
        output_file = Path(args.output)
        
        try:
            manifest = hasher.create_asset_manifest(benchmarks_dir, models_dir, output_file)
            print("Asset hashing completed successfully!")
        except Exception as e:
            print(f"Error creating asset manifest: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()