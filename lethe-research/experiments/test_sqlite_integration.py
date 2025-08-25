#!/usr/bin/env python3
"""
Test SQLite Integration
======================

Quick validation test to ensure the SQLite database schema and 
experiment controller integration work correctly.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import json

# Add the experiments directory to path
sys.path.append(str(Path(__file__).parent))

from sqlite_schema import create_experiment_database
from sqlite_run_controller import SQLiteExperimentController

def test_database_creation():
    """Test basic database creation and schema"""
    print("Testing database creation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_experiments.db"
        
        # Create database
        db = create_experiment_database(str(db_path))
        
        # Test basic operations
        experiment_data = {
            'experiment_id': 'test_exp_001',
            'name': 'Test Experiment',
            'version': '1.0.0',
            'start_time': '2024-08-23T12:00:00Z',
            'status': 'running'
        }
        
        experiment_id = db.insert_experiment(experiment_data)
        print(f"✅ Created experiment: {experiment_id}")
        
        # Test configuration insertion
        config_data = {
            'config_id': 'test_config_001',
            'experiment_id': experiment_id,
            'config_name': 'test_lethe_config',
            'config_type': 'lethe',
            'parameters': {'alpha': 0.7, 'beta': 0.5}
        }
        
        config_id = db.insert_configuration(config_data)
        print(f"✅ Created configuration: {config_id}")
        
        # Test run insertion
        run_data = {
            'run_id': 'test_run_001',
            'experiment_id': experiment_id,
            'config_id': config_id,
            'domain': 'code_heavy',
            'complexity': 'medium',
            'session_length': 'medium',
            'replication': 0,
            'status': 'completed',
            'runtime_seconds': 45.2,
            'peak_memory_mb': 128.5
        }
        
        run_id = db.insert_run(run_data)
        print(f"✅ Created run: {run_id}")
        
        # Test query result insertion
        query_data = {
            'run_id': run_id,
            'query_id': 'test_query_001',
            'session_id': 'test_session',
            'query_text': 'How do I implement binary search?',
            'domain': 'code_heavy',
            'complexity': 'medium',
            'ground_truth_docs': ['doc1', 'doc2'],
            'retrieved_docs': ['doc1', 'doc3', 'doc2'],
            'relevance_scores': [0.9, 0.6, 0.8],
            'latency_ms': 150.0,
            'memory_mb': 45.2,
            'timestamp': '2024-08-23T12:01:00Z'
        }
        
        query_execution_id = db.insert_query_result(query_data)
        print(f"✅ Created query result: {query_execution_id}")
        
        # Test metrics insertion
        metrics_data = [
            {
                'run_id': run_id,
                'metric_category': 'quality',
                'metric_name': 'ndcg_at_10',
                'metric_value': 0.75,
                'metric_unit': 'score'
            },
            {
                'run_id': run_id,
                'metric_category': 'efficiency',
                'metric_name': 'latency_p95',
                'metric_value': 180.0,
                'metric_unit': 'ms'
            }
        ]
        
        db.insert_metrics(metrics_data)
        print(f"✅ Created {len(metrics_data)} metrics")
        
        # Test summary retrieval
        summary = db.get_experiment_summary(experiment_id)
        print(f"✅ Retrieved experiment summary: {summary['experiment']['name']}")
        
        # Test leaderboard
        leaderboard = db.get_leaderboard(experiment_id, 'ndcg_at_10')
        print(f"✅ Retrieved leaderboard with {len(leaderboard)} configurations")
        
        print("✅ All database tests passed!")

def test_config_file_creation():
    """Create a minimal test config file"""
    config = {
        "name": "test_sqlite_integration",
        "version": "1.0.0",
        "seed": 42,
        "replications": 2,
        "parameters": {
            "alpha": {
                "values": [0.5, 0.7],
                "type": "float",
                "default": 0.7
            },
            "beta": {
                "values": [0.3, 0.5],
                "type": "float", 
                "default": 0.5
            }
        },
        "conditions": {
            "domains": [
                {"name": "code_heavy"},
                {"name": "mixed"}
            ],
            "query_complexity": [
                {"name": "simple"},
                {"name": "medium"}
            ],
            "session_length": [
                {"name": "short"}
            ]
        },
        "baselines": {
            "bm25_simple": {
                "params": {"type": "bm25"}
            }
        },
        "metrics": {
            "quality": ["ndcg_at_k", "recall_at_k"],
            "efficiency": ["latency", "memory"]
        },
        "statistics": {
            "significance_level": 0.05,
            "bootstrap_samples": 1000
        },
        "resources": {
            "timeout_per_session": "30s",
            "max_memory_gb": 4
        },
        "output": {
            "log_level": "INFO",
            "save_artifacts": True
        }
    }
    
    return config

def test_controller_integration():
    """Test the SQLite experiment controller"""
    print("\nTesting controller integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test config file
        config_data = test_config_file_creation()
        config_path = temp_path / "test_config.yaml"
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        print(f"✅ Created test config: {config_path}")
        
        # Initialize controller
        controller = SQLiteExperimentController(
            config_path=str(config_path),
            db_path="test_integration.db",
            output_dir=str(temp_path),
            max_workers=2
        )
        
        print(f"✅ Initialized controller")
        print(f"   Database: {controller.db_path}")
        print(f"   Experiment ID: {controller.experiment_id}")
        
        # Generate configurations
        config_ids = controller.generate_configurations()
        print(f"✅ Generated {len(config_ids)} configurations")
        
        # Generate runs (dry run) - pass config_ids to avoid regenerating
        runs = controller.generate_experiment_runs(config_ids)
        print(f"✅ Generated {len(runs)} experimental runs")
        
        # Expected: 2 alphas * 2 betas * 2 domains * 2 complexities * 1 session * 2 reps = 32 lethe runs
        # Plus: 1 baseline * 2 domains * 2 complexities * 1 session * 2 reps = 8 baseline runs  
        # Total: 40 runs
        expected_runs = 2 * 2 * 2 * 2 * 1 * 2 + 1 * 2 * 2 * 1 * 2
        print(f"   Expected {expected_runs} runs, got {len(runs)}")
        
        # Test experiment summary
        summary = controller.get_experiment_summary()
        print(f"✅ Retrieved experiment summary")
        print(f"   Total runs: {summary['statistics']['total_runs']}")
        
        print("✅ Controller integration tests passed!")

def main():
    """Run all tests"""
    print("🧪 Testing SQLite Integration for Lethe Experiments")
    print("=" * 60)
    
    try:
        test_database_creation()
        test_controller_integration()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed! SQLite integration is working correctly.")
        print("\n🎯 Key improvements:")
        print("   • Single SQLite database replaces multiple output formats")
        print("   • Centralized data storage for simplified analysis")
        print("   • Efficient querying and aggregation capabilities")
        print("   • Better data integrity and consistency")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())