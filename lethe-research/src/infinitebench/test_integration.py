#!/usr/bin/env python3
"""
InfiniteBench Integration Test
=============================

Quick validation script to test the InfiniteBench integration components
without running a full evaluation.

Usage:
    python test_integration.py
"""

import sys
import logging
from pathlib import Path
from typing import List

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported successfully."""
    logger.info("🧪 Testing imports...")
    
    try:
        from . import (
            InfiniteBenchLoader,
            InfiniteBenchEvaluator, 
            BM25Baseline,
            NaiveChunkingBaseline,
            DenseRetrievalBaseline,
            InfiniteBenchMetrics,
            InfiniteBenchStatistics
        )
        logger.info("✅ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False

def test_dataset_loader():
    """Test dataset loader functionality."""
    logger.info("🧪 Testing dataset loader...")
    
    try:
        from .dataset_loader import InfiniteBenchLoader, TaskMetadata
        
        # Test with a non-existent directory first
        try:
            loader = InfiniteBenchLoader("fake/path")
            logger.error("❌ Should have failed with non-existent path")
            return False
        except FileNotFoundError:
            logger.info("✅ Correctly handles non-existent data directory")
        
        # Test metadata access
        loader = InfiniteBenchLoader("benchmarks/infinitebench/data")
        
        # Check task configurations
        assert len(loader.TASK_CONFIGS) == 12, "Should have 12 task configurations"
        
        # Test metadata retrieval
        passkey_metadata = loader.get_task_metadata("passkey")
        assert passkey_metadata.name == "passkey"
        assert passkey_metadata.metric == "accuracy"
        
        logger.info("✅ Dataset loader basic functionality works")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset loader test failed: {e}")
        return False

def test_metrics():
    """Test metrics calculation."""
    logger.info("🧪 Testing metrics...")
    
    try:
        from .metrics import InfiniteBenchMetrics
        
        metrics = InfiniteBenchMetrics()
        
        # Test exact match
        predictions = ["hello world", "test answer", "42"]
        references = ["hello world", "different answer", "42"]
        
        em_result = metrics.exact_match(predictions, references)
        assert em_result.score == 2/3, f"Expected EM score 0.667, got {em_result.score}"
        
        # Test F1 score
        f1_result = metrics.f1_score(predictions, references)
        assert 0 <= f1_result.score <= 1, f"F1 score should be between 0 and 1, got {f1_result.score}"
        
        # Test ROUGE-L
        rouge_result = metrics.rouge_l(predictions, references)
        assert 0 <= rouge_result.score <= 1, f"ROUGE-L should be between 0 and 1, got {rouge_result.score}"
        
        logger.info("✅ Metrics calculation works")
        return True
        
    except Exception as e:
        logger.error(f"❌ Metrics test failed: {e}")
        return False

def test_baselines():
    """Test baseline method initialization."""
    logger.info("🧪 Testing baseline methods...")
    
    try:
        from .baselines import BM25Baseline, NaiveChunkingBaseline
        
        # Test BM25
        bm25 = BM25Baseline(k1=1.2, b=0.75, top_k=5)
        assert bm25.k1 == 1.2
        assert bm25.b == 0.75
        
        # Test Naive Chunking
        naive = NaiveChunkingBaseline(chunk_size=1024, strategy="uniform", top_k=5)
        assert naive.chunk_size == 1024
        assert naive.strategy == "uniform"
        
        logger.info("✅ Baseline method initialization works")
        return True
        
    except Exception as e:
        logger.error(f"❌ Baseline test failed: {e}")
        return False

def test_evaluation_pipeline():
    """Test evaluation pipeline setup."""
    logger.info("🧪 Testing evaluation pipeline...")
    
    try:
        from .evaluation_pipeline import InfiniteBenchEvaluator, ExperimentConfig
        
        evaluator = InfiniteBenchEvaluator()
        
        # Test experiment config creation
        config = ExperimentConfig(
            experiment_name="test_experiment",
            tasks=["passkey", "kv_retrieval"],
            methods=["bm25"],
            max_samples_per_task=10,
            output_dir=Path("test_output"),
            bootstrap_samples=100
        )
        
        assert config.experiment_name == "test_experiment"
        assert len(config.tasks) == 2
        assert config.bootstrap_samples == 100
        
        logger.info("✅ Evaluation pipeline setup works")
        return True
        
    except Exception as e:
        logger.error(f"❌ Evaluation pipeline test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading."""
    logger.info("🧪 Testing configuration...")
    
    try:
        import yaml
        
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ['data', 'evaluation', 'baselines', 'reporting']
            for section in required_sections:
                assert section in config, f"Missing required config section: {section}"
            
            # Check specific values
            assert 'tasks' in config['evaluation']
            assert len(config['evaluation']['tasks']) > 0
            
            logger.info("✅ Configuration loading works")
            return True
        else:
            logger.warning("⚠️ Config file not found, skipping configuration test")
            return True
            
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

def test_data_availability():
    """Test if dataset files are available."""
    logger.info("🧪 Testing data availability...")
    
    try:
        data_dir = Path("benchmarks/infinitebench/data")
        
        if not data_dir.exists():
            logger.warning(f"⚠️ Data directory not found: {data_dir}")
            logger.info("   Run dataset download first: bash scripts/download_dataset.sh")
            return True  # Not a failure, just not set up yet
        
        # Check for some expected files
        expected_files = [
            "passkey.jsonl",
            "kv_retrieval.jsonl", 
            "longbook_qa_eng.jsonl"
        ]
        
        available_files = []
        for file_name in expected_files:
            file_path = data_dir / file_name
            if file_path.exists():
                available_files.append(file_name)
                
        if available_files:
            logger.info(f"✅ Found {len(available_files)} data files: {available_files}")
        else:
            logger.warning("⚠️ No expected data files found - dataset download may be incomplete")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Data availability test failed: {e}")
        return False

def run_all_tests():
    """Run all integration tests."""
    logger.info("🚀 Starting InfiniteBench integration tests...")
    
    tests = [
        ("Imports", test_imports),
        ("Dataset Loader", test_dataset_loader), 
        ("Metrics", test_metrics),
        ("Baselines", test_baselines),
        ("Evaluation Pipeline", test_evaluation_pipeline),
        ("Configuration", test_configuration),
        ("Data Availability", test_data_availability),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            failed += 1
    
    logger.info(f"\n🏁 Test Results:")
    logger.info(f"✅ Passed: {passed}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📊 Success Rate: {passed}/{passed + failed} ({100 * passed / (passed + failed):.1f}%)")
    
    if failed == 0:
        logger.info("🎉 All tests passed! The integration is ready to use.")
        return True
    else:
        logger.error(f"💥 {failed} tests failed. Please fix issues before running evaluation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)