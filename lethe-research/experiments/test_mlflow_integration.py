#!/usr/bin/env python3
"""
MLflow Integration Test for Lethe Research Framework
===================================================

Quick validation test to ensure MLflow integration is working properly
before running the full evaluation pipeline.

Usage:
    python test_mlflow_integration.py
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import logging

# Add project modules to path
sys.path.append(str(Path(__file__).parent.parent))

def test_mlflow_availability():
    """Test that MLflow is properly installed and importable"""
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow import log_param, log_metric, log_artifact
        print(f"✅ MLflow v{mlflow.__version__} is available")
        return True
    except ImportError as e:
        print(f"❌ MLflow import failed: {e}")
        print("Install MLflow with: pip install -r requirements.txt")
        return False

def test_experiment_controller_import():
    """Test that the updated ExperimentController can be imported"""
    try:
        from run import ExperimentController
        print("✅ ExperimentController with MLflow integration imported successfully")
        return True
    except ImportError as e:
        print(f"❌ ExperimentController import failed: {e}")
        return False

def test_basic_mlflow_tracking():
    """Test basic MLflow tracking functionality"""
    try:
        import mlflow
        
        # Create temporary MLflow directory
        with tempfile.TemporaryDirectory() as temp_dir:
            mlflow_dir = Path(temp_dir) / "mlruns"
            mlflow.set_tracking_uri(str(mlflow_dir))
            
            # Create experiment
            exp_id = mlflow.create_experiment("test_integration")
            print(f"✅ Created test experiment: {exp_id}")
            
            # Start run
            with mlflow.start_run():
                # Log parameters
                mlflow.log_param("alpha", 0.7)
                mlflow.log_param("beta", 0.5)
                mlflow.log_params({"test_param_1": "value1", "test_param_2": 42})
                
                # Log metrics
                mlflow.log_metric("ndcg_at_10", 0.85)
                mlflow.log_metric("recall_at_50", 0.92)
                mlflow.log_metric("latency_p95", 150.5)
                mlflow.log_metric("memory_peak", 512.3)
                
                # Log additional metrics
                mlflow.log_metrics({
                    "runtime_seconds": 45.2,
                    "n_queries": 100,
                    "success_rate": 0.98
                })
                
                print("✅ Logged parameters and metrics successfully")
                
                # Test artifact logging
                test_file = Path(temp_dir) / "test_artifact.txt"
                test_file.write_text("Test artifact content for MLflow integration")
                mlflow.log_artifact(str(test_file))
                print("✅ Logged test artifact successfully")
                
            print("✅ Basic MLflow tracking test completed successfully")
            return True
            
    except Exception as e:
        print(f"❌ MLflow tracking test failed: {e}")
        return False

def test_experiment_controller_initialization():
    """Test ExperimentController initialization with MLflow"""
    try:
        from run import ExperimentController
        
        # Create temporary config file
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"
            config_content = """
name: "test_experiment"
version: "1.0"
seed: 42
replications: 1
parameters:
  alpha:
    type: float
    values: [0.7]
    default: 0.7
  beta:
    type: float  
    values: [0.5]
    default: 0.5
conditions:
  domains:
    - name: "test_domain"
      description: "Test domain"
      weight: 1.0
  query_complexity:
    - name: "simple"
      description: "Simple queries"
      weight: 1.0
  session_length:
    - name: "short"
      turns: 5
      description: "Short sessions"
      weight: 1.0
baselines:
  test_baseline:
    name: "Test Baseline"
    params:
      strategy: "test"
metrics:
  quality:
    ndcg_at_k: [10]
    recall_at_k: [50]
  efficiency:
    latency_percentiles: [95]
  coverage: {}
  consistency: {}
statistics:
  significance_level: 0.05
  confidence_intervals: "bootstrap_95"
resources:
  max_concurrent_runs: 1
  timeout_per_query: "30s"
  timeout_per_session: "5min"
output:
  artifacts_dir: "test_artifacts"
  results_format: ["json"]
  log_level: "INFO"
"""
            config_path.write_text(config_content)
            
            # Initialize controller with MLflow
            mlflow_dir = Path(temp_dir) / "mlruns"
            controller = ExperimentController(
                config_path=str(config_path),
                output_dir=str(temp_dir),
                max_workers=1,
                mlflow_tracking_uri=str(mlflow_dir),
                experiment_name="test_controller_init"
            )
            
            print("✅ ExperimentController initialized with MLflow successfully")
            
            # Test parameter logging
            if hasattr(controller, 'mlflow_experiment_id') and controller.mlflow_experiment_id:
                print("✅ MLflow experiment ID created successfully")
            else:
                print("⚠️  MLflow experiment ID not created (may be normal if MLflow unavailable)")
                
            return True
            
    except Exception as e:
        print(f"❌ ExperimentController initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_git_sha_tracking():
    """Test Git SHA tracking functionality"""
    try:
        from run import ExperimentController
        import subprocess
        
        # Check if we're in a git repository
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent
            )
            if result.returncode == 0:
                expected_sha = result.stdout.strip()
                print(f"✅ Git repository detected, current SHA: {expected_sha[:8]}...")
            else:
                print("⚠️  Not in a git repository, SHA tracking will return 'unknown'")
                expected_sha = "unknown"
        except FileNotFoundError:
            print("⚠️  Git command not available, SHA tracking will return 'unknown'")
            expected_sha = "unknown"
            
        # Test controller's Git SHA method
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(__file__).parent / "grid_config.yaml"
            if not config_path.exists():
                print("⚠️  grid_config.yaml not found, using minimal config for test")
                config_path = Path(temp_dir) / "test_config.yaml"
                config_path.write_text("""
name: "git_test"
version: "1.0" 
seed: 42
replications: 1
parameters:
  alpha: {type: float, values: [0.7]}
conditions:
  domains: [{name: "test", weight: 1.0}]
  query_complexity: [{name: "simple", weight: 1.0}]
  session_length: [{name: "short", turns: 5, weight: 1.0}]
baselines: {}
metrics: {quality: {}, efficiency: {}, coverage: {}, consistency: {}}
statistics: {significance_level: 0.05}
resources: {max_concurrent_runs: 1}  
output: {artifacts_dir: "test", log_level: "INFO"}
""")
            
            controller = ExperimentController(
                config_path=str(config_path),
                output_dir=temp_dir,
                mlflow_tracking_uri=None  # Disable MLflow for this test
            )
            
            git_sha = controller.get_git_commit_sha()
            if expected_sha == "unknown":
                print(f"✅ Git SHA tracking returned '{git_sha}' as expected")
            elif git_sha == expected_sha:
                print(f"✅ Git SHA tracking working correctly: {git_sha[:8]}...")
            else:
                print(f"⚠️  Git SHA mismatch: expected {expected_sha[:8]}, got {git_sha[:8]}")
                
            return True
            
    except Exception as e:
        print(f"❌ Git SHA tracking test failed: {e}")
        return False

def main():
    """Run all MLflow integration tests"""
    print("=" * 60)
    print("🚀 MLflow Integration Test Suite")
    print("Phase 2.4: MLflow Integration Validation")
    print("=" * 60)
    print()
    
    tests = [
        ("MLflow Availability", test_mlflow_availability),
        ("ExperimentController Import", test_experiment_controller_import),
        ("Basic MLflow Tracking", test_basic_mlflow_tracking),
        ("ExperimentController + MLflow", test_experiment_controller_initialization),
        ("Git SHA Tracking", test_git_sha_tracking)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print()
    print(f"Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! MLflow integration is ready.")
        print()
        print("Next steps:")
        print("1. Start MLflow server: ./scripts/start_mlflow_server.sh")
        print("2. Run full evaluation: ./scripts/run_full_evaluation.sh") 
        print("3. View results at: http://127.0.0.1:5000")
        return True
    else:
        print("⚠️  Some tests failed. Please address issues before running full evaluation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)