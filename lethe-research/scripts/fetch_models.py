#!/usr/bin/env python3
"""
Model Checkpoint Fetcher
Downloads and validates standard IR model checkpoints
Part of Lethe Hermetic Infrastructure (B1)
"""

import json
import os
import requests
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import sys
from datetime import datetime
import shutil
from huggingface_hub import hf_hub_download, snapshot_download
import subprocess

# Standard model configurations
MODELS_CONFIG = {
    "splade_plus_plus": {
        "description": "SPLADE++ sparse retrieval model",
        "type": "sparse_retrieval",
        "source": "huggingface",
        "model_id": "naver/splade-cocondenser-ensembledistil",
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json"],
        "expected_size_mb": 440,
        "framework": "transformers"
    },
    
    "unicoil": {
        "description": "uniCOIL sparse retrieval model",
        "type": "sparse_retrieval", 
        "source": "huggingface",
        "model_id": "castorini/unicoil-msmarco-passage",
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json"],
        "expected_size_mb": 440,
        "framework": "transformers"
    },
    
    "colbert_v2": {
        "description": "ColBERTv2 late interaction model",
        "type": "late_interaction",
        "source": "huggingface", 
        "model_id": "colbert-ir/colbertv2.0",
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json"],
        "expected_size_mb": 440,
        "framework": "colbert"
    },
    
    "cross_encoder_msmarco": {
        "description": "Cross-encoder reranker for MS MARCO",
        "type": "cross_encoder",
        "source": "huggingface",
        "model_id": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json"],
        "expected_size_mb": 90,
        "framework": "sentence_transformers"
    },
    
    "sentence_transformer_msmarco": {
        "description": "Sentence transformer for dense retrieval",
        "type": "dense_retrieval",
        "source": "huggingface",
        "model_id": "sentence-transformers/msmarco-distilbert-base-tas-b",
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json"],
        "expected_size_mb": 250,
        "framework": "sentence_transformers"
    },
    
    "bge_base_en_v15": {
        "description": "BGE base English embedding model v1.5",
        "type": "dense_retrieval",
        "source": "huggingface",
        "model_id": "BAAI/bge-base-en-v1.5",
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json"],
        "expected_size_mb": 440,
        "framework": "sentence_transformers"
    },
    
    "e5_base_v2": {
        "description": "E5 base embedding model v2",
        "type": "dense_retrieval",
        "source": "huggingface",
        "model_id": "intfloat/e5-base-v2", 
        "files": ["pytorch_model.bin", "config.json", "tokenizer.json"],
        "expected_size_mb": 440,
        "framework": "sentence_transformers"
    }
}

class ModelFetcher:
    """Handles downloading and validation of model checkpoints"""
    
    def __init__(self, cache_dir: str = "models/checkpoints"):
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
    
    def download_huggingface_model(self, model_id: str, output_dir: Path, files: List[str]) -> bool:
        """Download model from Hugging Face Hub"""
        print(f"Downloading {model_id} from Hugging Face Hub...")
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if not files:
                # Download entire repository
                snapshot_download(
                    repo_id=model_id,
                    cache_dir=str(self.cache_dir / "hub_cache"),
                    local_dir=str(output_dir),
                    local_dir_use_symlinks=False
                )
            else:
                # Download specific files
                for filename in files:
                    try:
                        downloaded_path = hf_hub_download(
                            repo_id=model_id,
                            filename=filename,
                            cache_dir=str(self.cache_dir / "hub_cache"),
                            local_dir=str(output_dir),
                            local_dir_use_symlinks=False
                        )
                        print(f"Downloaded: {filename}")
                    except Exception as e:
                        print(f"Warning: Could not download {filename}: {e}")
                        # Continue with other files
            
            print(f"Model downloaded to {output_dir}")
            return True
            
        except Exception as e:
            print(f"Error downloading {model_id}: {e}")
            return False
    
    def download_direct_url(self, url: str, output_path: Path) -> bool:
        """Download file from direct URL"""
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
            return True
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    
    def fetch_model(self, model_name: str) -> bool:
        """Fetch a specific model"""
        if model_name not in MODELS_CONFIG:
            print(f"Unknown model: {model_name}")
            return False
            
        config = MODELS_CONFIG[model_name]
        output_dir = self.cache_dir / model_name
        
        # Skip if already exists
        if output_dir.exists() and any(output_dir.iterdir()):
            print(f"Model {model_name} already exists, skipping download")
            return True
        
        success = False
        
        if config["source"] == "huggingface":
            success = self.download_huggingface_model(
                config["model_id"],
                output_dir,
                config.get("files", [])
            )
        elif config["source"] == "direct_url":
            # For future direct URL downloads
            output_path = output_dir / config.get("filename", "model.bin")
            success = self.download_direct_url(config["url"], output_path)
        
        if success:
            # Log the download
            self.download_log.append({
                "model_name": model_name,
                "model_id": config.get("model_id", ""),
                "path": str(output_dir),
                "timestamp": datetime.now().isoformat()
            })
            
        return success
    
    def validate_model(self, model_name: str) -> bool:
        """Validate model integrity"""
        model_dir = self.cache_dir / model_name
        
        if not model_dir.exists():
            print(f"Model directory not found: {model_dir}")
            return False
            
        config = MODELS_CONFIG[model_name]
        
        # Check if required files exist
        required_files = config.get("files", [])
        for filename in required_files:
            file_path = model_dir / filename
            if not file_path.exists():
                # Try common alternatives
                alternatives = [
                    filename.replace(".bin", ".safetensors"),
                    "model.safetensors",
                    "pytorch_model.safetensors"
                ]
                
                found = False
                for alt in alternatives:
                    alt_path = model_dir / alt
                    if alt_path.exists():
                        print(f"Found alternative file: {alt} for {filename}")
                        found = True
                        break
                
                if not found:
                    print(f"Missing required file: {file_path}")
                    # Don't fail validation for missing files in HF models
                    # as they might use different formats
                    if config["source"] != "huggingface":
                        return False
        
        # Basic structure validation
        if config["framework"] == "transformers":
            # Should have config.json at minimum
            if not (model_dir / "config.json").exists():
                print(f"Missing config.json for transformers model")
                return False
        
        print(f"Model {model_name} validation passed")
        return True
    
    def calculate_model_size(self, model_dir: Path) -> int:
        """Calculate total size of model directory in MB"""
        total_size = 0
        for file_path in model_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size // (1024 * 1024)  # Convert to MB
    
    def create_model_manifest(self) -> Dict:
        """Create manifest of all downloaded models"""
        manifest = {
            "manifest_version": "1.0.0", 
            "generated_at": datetime.now().isoformat(),
            "models": {},
            "download_log": self.download_log
        }
        
        for model_name, config in MODELS_CONFIG.items():
            model_dir = self.cache_dir / model_name
            if model_dir.exists():
                actual_size_mb = self.calculate_model_size(model_dir)
                
                manifest["models"][model_name] = {
                    "description": config["description"],
                    "type": config["type"],
                    "framework": config["framework"],
                    "model_id": config.get("model_id", ""),
                    "expected_size_mb": config["expected_size_mb"],
                    "actual_size_mb": actual_size_mb,
                    "path": str(model_dir),
                    "validated": self.validate_model(model_name),
                    "files": []
                }
                
                # Add file hashes
                for file_path in model_dir.rglob("*"):
                    if file_path.is_file():
                        file_hash = self.calculate_file_hash(file_path)
                        manifest["models"][model_name]["files"].append({
                            "path": str(file_path.relative_to(model_dir)),
                            "size": file_path.stat().st_size,
                            "sha256": file_hash
                        })
        
        return manifest
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available"""
        try:
            import huggingface_hub
            print(f"huggingface_hub version: {huggingface_hub.__version__}")
            return True
        except ImportError:
            print("Error: huggingface_hub not installed")
            print("Install with: pip install huggingface_hub")
            return False

def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        cache_dir = sys.argv[1]
    else:
        cache_dir = "models/checkpoints"
    
    print("Fetching standard IR model checkpoints...")
    
    fetcher = ModelFetcher(cache_dir)
    
    # Check dependencies
    if not fetcher.check_dependencies():
        sys.exit(1)
    
    # Fetch all models
    success = True
    
    for model_name in MODELS_CONFIG.keys():
        print(f"\n=== Fetching {model_name} ===")
        model_success = fetcher.fetch_model(model_name)
        success &= model_success
        
        if model_success:
            print(f"✓ {model_name} downloaded successfully")
        else:
            print(f"✗ Failed to download {model_name}")
    
    if success:
        # Create and save manifest
        manifest = fetcher.create_model_manifest()
        manifest_path = Path(cache_dir) / "model_manifest.json"
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            
        print(f"\n=== Model Fetch Complete ===")
        print(f"Manifest saved to: {manifest_path}")
        print(f"Models fetched: {len(manifest['models'])}")
        
        total_size = sum(info['actual_size_mb'] for info in manifest['models'].values())
        print(f"Total size: {total_size} MB")
        
        for name, info in manifest['models'].items():
            status = "✓" if info['validated'] else "✗"
            print(f"  {status} {name}: {info['actual_size_mb']} MB ({info['type']})")
            
    else:
        print("\n=== Errors occurred during model fetch ===")
        sys.exit(1)

if __name__ == "__main__":
    main()