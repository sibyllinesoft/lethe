#!/usr/bin/env python3
"""
Core System Test for Production IR System

Tests the core infrastructure components without heavy dependencies
like transformers/torch to validate implementation structure.

Usage:
    python test_core_system.py
"""

import os
import sys
import json
import tempfile
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_core_imports():
    """Test that all core modules can be imported."""
    print("Testing core imports...")
    
    try:
        from retriever.core_only import (
            TimingHarness, PerformanceProfiler,
            IndexMetadata, MetadataManager, IndexStats,
            RetrieverConfig, BM25Config, HNSWConfig, IVFPQConfig
        )
        print("✓ Core modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_timing_harness():
    """Test the timing harness functionality."""
    print("Testing timing harness...")
    
    try:
        from retriever.core_only import TimingHarness, PerformanceProfiler
        
        harness = TimingHarness(cold_cycles=3, warm_cycles=5)
        
        def dummy_operation():
            # Simulate some work
            return sum(i**2 for i in range(1000))
            
        # Test single measurement
        with harness.measure("dummy_op"):
            dummy_operation()
            
        # Test benchmark
        profile = harness.benchmark_function(dummy_operation, "dummy_benchmark")
        
        assert profile.count > 0
        assert profile.latency_p50 > 0
        assert profile.throughput > 0
        
        print(f"✓ Timing harness: {profile.count} measurements, "
              f"p95={profile.latency_p95:.2f}ms, "
              f"throughput={profile.throughput:.1f} ops/sec")
        return True
    except Exception as e:
        print(f"❌ Timing harness error: {e}")
        return False

def test_configuration():
    """Test configuration system."""
    print("Testing configuration...")
    
    try:
        from retriever.core_only import RetrieverConfig, BM25Config
        
        # Test default configuration
        config = RetrieverConfig()
        errors = config.validate()
        
        if errors:
            print(f"❌ Default config validation failed: {errors}")
            return False
        
        # Test configuration serialization
        config_dict = config.to_dict()
        config2 = RetrieverConfig.from_dict(config_dict)
        
        assert config.bm25.k1 == config2.bm25.k1
        assert config.hnsw.m == config2.hnsw.m
        assert config.system.cpu_cores == config2.system.cpu_cores
        
        print("✓ Configuration: validation and serialization working")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_metadata_system():
    """Test metadata management system."""
    print("Testing metadata system...")
    
    try:
        from retriever.core_only import IndexMetadata, MetadataManager, IndexStats
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create metadata manager
            manager = MetadataManager(temp_dir)
            
            # Create sample metadata
            stats = IndexStats(
                num_documents=1000,
                num_terms=10000, 
                total_postings=50000,
                avg_doc_length=100.0,
                collection_size_mb=10.5,
                index_size_mb=8.3,
                compression_ratio=0.79,
                build_time_sec=45.2,
                memory_used_mb=512.0,
                cpu_time_sec=42.1
            )
            
            metadata = IndexMetadata(
                index_type="test",
                index_name="test_index",
                dataset_name="test_dataset",
                build_params={"param1": "value1"},
                stats=stats
            )
            
            # Test save/load cycle
            filepath = manager.save_metadata(metadata)
            assert filepath.exists()
            
            loaded_metadata = manager.load_metadata("test_dataset", "test", "test_index")
            assert loaded_metadata is not None
            assert loaded_metadata.dataset_name == "test_dataset"
            assert loaded_metadata.stats.num_documents == 1000
            
            print("✓ Metadata: save/load cycle working")
            return True
    except Exception as e:
        print(f"❌ Metadata error: {e}")
        return False

def test_file_structure():
    """Test that required files are present."""
    print("Testing file structure...")
    
    root = Path(__file__).parent.parent
    required_files = [
        "src/retriever/__init__.py",
        "src/retriever/timing.py", 
        "src/retriever/metadata.py",
        "src/retriever/config.py",
        "src/retriever/bm25.py",
        "src/retriever/ann.py", 
        "src/retriever/embeddings.py",
        "scripts/build_indices.py",
        "scripts/benchmark_indices.py",
        "config/retriever_config.yaml",
        "requirements_ir.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not (root / file_path).exists():
            missing_files.append(file_path)
            
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files present")
        return True

def test_config_file():
    """Test that configuration file is valid.""" 
    print("Testing configuration file...")
    
    try:
        from retriever.core_only import RetrieverConfig
        
        config_path = Path(__file__).parent.parent / "config" / "retriever_config.yaml"
        
        if not config_path.exists():
            print(f"❌ Config file not found: {config_path}")
            return False
            
        config = RetrieverConfig.from_yaml(config_path)
        errors = config.validate()
        
        if errors:
            print(f"❌ Config file validation failed: {errors}")
            return False
            
        # Check some expected values
        assert config.bm25.k1 == 0.9
        assert config.hnsw.m == 16
        assert "msmarco-passage-dev" in config.datasets
        
        print("✓ Configuration file valid")
        return True
    except Exception as e:
        print(f"❌ Config file error: {e}")
        return False

def test_script_syntax():
    """Test that main scripts have valid syntax."""
    print("Testing script syntax...")
    
    root = Path(__file__).parent.parent
    scripts = [
        "scripts/build_indices.py",
        "scripts/benchmark_indices.py"
    ]
    
    try:
        for script_path in scripts:
            full_path = root / script_path
            with open(full_path, 'r') as f:
                code = f.read()
            compile(code, str(full_path), 'exec')
            
        print("✓ All scripts have valid syntax")
        return True
    except SyntaxError as e:
        print(f"❌ Syntax error in {e.filename}: {e}")
        return False
    except Exception as e:
        print(f"❌ Script error: {e}")
        return False

def run_all_tests():
    """Run all core system tests."""
    print("🧪 Running Core IR System Tests")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_core_imports,
        test_configuration,
        test_metadata_system, 
        test_timing_harness,
        test_config_file,
        test_script_syntax
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print()  # Add spacing after failed test
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 50)
    
    if passed == total:
        print(f"✅ All {total} core tests passed!")
        print("\n🚀 Core IR system implementation is ready!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements_ir.txt") 
        print("  2. Build indices: python scripts/build_indices.py --config config/retriever_config.yaml")
        print("  3. Benchmark: python scripts/benchmark_indices.py --config config/retriever_config.yaml")
        return True
    else:
        print(f"❌ {passed}/{total} tests passed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)