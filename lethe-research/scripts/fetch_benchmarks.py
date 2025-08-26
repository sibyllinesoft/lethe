#!/usr/bin/env python3
"""
Benchmark Dataset Fetcher
Downloads and prepares standard IR benchmarks for evaluation
Part of Lethe Hermetic Infrastructure (B1)
"""

import json
import os
import requests
import gzip
import tarfile
import zipfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
import sys
from datetime import datetime

# Standard benchmark configurations
BENCHMARKS_CONFIG = {
    "ms_marco_passage_dev": {
        "description": "MS MARCO Passage Retrieval Development Set",
        "urls": {
            "queries": "https://msmarco.blob.core.windows.net/msmarcoranking/queries.dev.small.tsv",
            "qrels": "https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.small.tsv",
            "collection": "https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz"
        },
        "expected_hashes": {
            "queries": "sha256:expected_hash_here",  # To be updated with actual hashes
            "qrels": "sha256:expected_hash_here",
            "collection": "sha256:expected_hash_here"
        },
        "format": "tsv",
        "size_mb": 2800,
        "num_queries": 6980,
        "num_docs": 8841823
    },
    
    "beir_trec_covid": {
        "description": "BEIR TREC-COVID dataset",
        "urls": {
            "dataset": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip"
        },
        "expected_hashes": {
            "dataset": "sha256:expected_hash_here"
        },
        "format": "jsonl",
        "size_mb": 275,
        "num_queries": 50,
        "num_docs": 171332
    },
    
    "beir_nfcorpus": {
        "description": "BEIR NFCorpus dataset", 
        "urls": {
            "dataset": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"
        },
        "expected_hashes": {
            "dataset": "sha256:expected_hash_here"
        },
        "format": "jsonl",
        "size_mb": 5,
        "num_queries": 323,
        "num_docs": 3633
    },
    
    "beir_fiqa": {
        "description": "BEIR FiQA-2018 dataset",
        "urls": {
            "dataset": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
        },
        "expected_hashes": {
            "dataset": "sha256:expected_hash_here"
        },
        "format": "jsonl", 
        "size_mb": 15,
        "num_queries": 648,
        "num_docs": 57638
    }
}

class BenchmarkFetcher:
    """Handles downloading and validation of benchmark datasets"""
    
    def __init__(self, cache_dir: str = "datasets/benchmarks"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.download_log = []
        
    def calculate_file_hash(self, filepath: Path, algorithm: str = "sha256") -> str:
        """Calculate cryptographic hash of file"""
        hash_func = hashlib.new(algorithm)
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
                
        return hash_func.hexdigest()
    
    def download_file(self, url: str, output_path: Path, expected_hash: Optional[str] = None) -> bool:
        """Download file with progress tracking and hash validation"""
        print(f"Downloading {url}...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rProgress: {progress:.1f}%", end='', flush=True)
            
            print(f"\nDownloaded to {output_path}")
            
            # Validate hash if provided
            if expected_hash and expected_hash != "sha256:expected_hash_here":
                algorithm, expected = expected_hash.split(":", 1)
                actual_hash = self.calculate_file_hash(output_path, algorithm)
                
                if actual_hash != expected:
                    print(f"Hash mismatch! Expected: {expected}, Got: {actual_hash}")
                    return False
                else:
                    print(f"Hash validated: {actual_hash}")
            else:
                # Calculate and store hash for future validation
                actual_hash = self.calculate_file_hash(output_path)
                print(f"Calculated hash: sha256:{actual_hash}")
                
                # Log for manifest
                self.download_log.append({
                    "url": url,
                    "file": str(output_path),
                    "hash": f"sha256:{actual_hash}",
                    "size": output_path.stat().st_size,
                    "timestamp": datetime.now().isoformat()
                })
            
            return True
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """Extract compressed archive"""
        print(f"Extracting {archive_path}...")
        
        try:
            extract_to.mkdir(parents=True, exist_ok=True)
            
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            elif archive_path.suffix == '.gz':
                # Handle .gz files
                output_name = archive_path.stem
                with gzip.open(archive_path, 'rb') as gz_file:
                    with open(extract_to / output_name, 'wb') as out_file:
                        out_file.write(gz_file.read())
            else:
                print(f"Unsupported archive format: {archive_path.suffix}")
                return False
                
            print(f"Extracted to {extract_to}")
            return True
            
        except Exception as e:
            print(f"Error extracting {archive_path}: {e}")
            return False
    
    def fetch_ms_marco_passage_dev(self) -> bool:
        """Fetch MS MARCO Passage Development Set"""
        dataset_name = "ms_marco_passage_dev"
        config = BENCHMARKS_CONFIG[dataset_name]
        output_dir = self.cache_dir / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        success = True
        
        # Download queries
        queries_path = output_dir / "queries.dev.small.tsv"
        if not queries_path.exists():
            success &= self.download_file(
                config["urls"]["queries"], 
                queries_path,
                config["expected_hashes"]["queries"]
            )
        
        # Download qrels
        qrels_path = output_dir / "qrels.dev.small.tsv"
        if not qrels_path.exists():
            success &= self.download_file(
                config["urls"]["qrels"],
                qrels_path, 
                config["expected_hashes"]["qrels"]
            )
        
        # Download and extract collection
        collection_path = output_dir / "collectionandqueries.tar.gz"
        if not collection_path.exists():
            success &= self.download_file(
                config["urls"]["collection"],
                collection_path,
                config["expected_hashes"]["collection"]
            )
            
        # Extract collection
        if success and collection_path.exists():
            success &= self.extract_archive(collection_path, output_dir / "collection")
            
        return success
    
    def fetch_beir_dataset(self, dataset_name: str) -> bool:
        """Fetch BEIR dataset"""
        if dataset_name not in BENCHMARKS_CONFIG:
            print(f"Unknown dataset: {dataset_name}")
            return False
            
        config = BENCHMARKS_CONFIG[dataset_name]
        output_dir = self.cache_dir / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download dataset zip
        dataset_path = output_dir / f"{dataset_name}.zip"
        if not dataset_path.exists():
            success = self.download_file(
                config["urls"]["dataset"],
                dataset_path,
                config["expected_hashes"]["dataset"]
            )
            if not success:
                return False
        
        # Extract dataset
        extract_dir = output_dir / "data"
        if not extract_dir.exists():
            success = self.extract_archive(dataset_path, extract_dir)
            if not success:
                return False
                
        return True
    
    def validate_dataset(self, dataset_name: str) -> bool:
        """Validate dataset integrity and format"""
        dataset_dir = self.cache_dir / dataset_name
        
        if not dataset_dir.exists():
            print(f"Dataset directory not found: {dataset_dir}")
            return False
            
        config = BENCHMARKS_CONFIG[dataset_name]
        
        # Basic validation - check expected files exist
        if dataset_name == "ms_marco_passage_dev":
            required_files = ["queries.dev.small.tsv", "qrels.dev.small.tsv"]
            for filename in required_files:
                file_path = dataset_dir / filename
                if not file_path.exists():
                    print(f"Missing required file: {file_path}")
                    return False
        else:
            # BEIR datasets
            data_dir = dataset_dir / "data"
            if not data_dir.exists():
                print(f"Missing data directory: {data_dir}")
                return False
                
        print(f"Dataset {dataset_name} validation passed")
        return True
    
    def create_benchmark_manifest(self) -> Dict:
        """Create manifest of all downloaded benchmarks"""
        manifest = {
            "manifest_version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "benchmarks": {},
            "download_log": self.download_log
        }
        
        for dataset_name, config in BENCHMARKS_CONFIG.items():
            dataset_dir = self.cache_dir / dataset_name
            if dataset_dir.exists():
                manifest["benchmarks"][dataset_name] = {
                    "description": config["description"],
                    "format": config["format"],
                    "expected_size_mb": config["size_mb"],
                    "num_queries": config["num_queries"],
                    "num_docs": config["num_docs"],
                    "path": str(dataset_dir),
                    "validated": self.validate_dataset(dataset_name),
                    "files": []
                }
                
                # Add file hashes
                for file_path in dataset_dir.rglob("*"):
                    if file_path.is_file():
                        file_hash = self.calculate_file_hash(file_path)
                        manifest["benchmarks"][dataset_name]["files"].append({
                            "path": str(file_path.relative_to(dataset_dir)),
                            "size": file_path.stat().st_size,
                            "sha256": file_hash
                        })
        
        return manifest

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        cache_dir = sys.argv[1]
    else:
        cache_dir = "datasets/benchmarks"
    
    print("Fetching standard IR benchmarks...")
    
    fetcher = BenchmarkFetcher(cache_dir)
    
    # Fetch all benchmarks
    success = True
    
    print("\n=== Fetching MS MARCO Passage Dev ===")
    success &= fetcher.fetch_ms_marco_passage_dev()
    
    print("\n=== Fetching BEIR TREC-COVID ===")
    success &= fetcher.fetch_beir_dataset("beir_trec_covid")
    
    print("\n=== Fetching BEIR NFCorpus ===")
    success &= fetcher.fetch_beir_dataset("beir_nfcorpus")
    
    print("\n=== Fetching BEIR FiQA-2018 ===")
    success &= fetcher.fetch_beir_dataset("beir_fiqa")
    
    if success:
        # Create and save manifest
        manifest = fetcher.create_benchmark_manifest()
        manifest_path = Path(cache_dir) / "benchmark_manifest.json"
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            
        print(f"\n=== Benchmark Fetch Complete ===")
        print(f"Manifest saved to: {manifest_path}")
        print(f"Datasets fetched: {len(manifest['benchmarks'])}")
        
        for name, info in manifest['benchmarks'].items():
            status = "✓" if info['validated'] else "✗"
            print(f"  {status} {name}: {info['num_queries']} queries, {info['num_docs']} docs")
            
    else:
        print("\n=== Errors occurred during benchmark fetch ===")
        sys.exit(1)

if __name__ == "__main__":
    main()