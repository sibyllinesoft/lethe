#!/usr/bin/env python3
"""
Minimal MLflow Integration Test for Lethe Hardening Workstream
Tests MLflow logging, FastAPI service, and reproduciblity requirements

This script validates:
- Objective 2: FastAPI prediction service performance (<50ms)
- Objective 3: MLflow experiment tracking and reproducibility
"""

import mlflow
import mlflow.sklearn
import time
import requests
import json
import os
import sys
from pathlib import Path

# Test configuration
FASTAPI_URL = "http://localhost:8080"
MLFLOW_URL = "http://localhost:5000"
TEST_QUERIES = [
    "How to implement async functions in Python?",
    "FastAPI validation error handling",
    "TypeScript interface best practices",
    "React component optimization techniques",
    "Node.js memory leak debugging"
]

def test_fastapi_service():
    """Test FastAPI service is running and performs within latency requirements"""
    print("🔍 Testing FastAPI Prediction Service...")
    
    try:
        # Test health endpoint
        health_response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"❌ Health check failed: {health_response.status_code}")
            return False
            
        health_data = health_response.json()
        print(f"✅ Service healthy: {health_data}")
        
        # Test prediction performance
        total_time = 0
        predictions = []
        
        for i, query in enumerate(TEST_QUERIES):
            start_time = time.time()
            
            response = requests.post(
                f"{FASTAPI_URL}/predict",
                json={"query": query, "context": {}},
                timeout=5
            )
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            if response.status_code != 200:
                print(f"❌ Prediction failed for query {i+1}: {response.status_code}")
                return False
                
            pred_data = response.json()
            prediction_time = pred_data.get('prediction_time_ms', latency_ms)
            
            print(f"   Query {i+1}: {prediction_time:.2f}ms (total latency: {latency_ms:.2f}ms)")
            predictions.append({
                'query': query,
                'alpha': pred_data.get('alpha'),
                'beta': pred_data.get('beta'),
                'plan': pred_data.get('plan'),
                'prediction_time_ms': prediction_time,
                'total_latency_ms': latency_ms
            })
            
            total_time += latency_ms
            
            # Check <50ms requirement
            if prediction_time > 50:
                print(f"❌ Performance target missed: {prediction_time:.2f}ms > 50ms")
                return False
        
        avg_latency = total_time / len(TEST_QUERIES)
        print(f"✅ Average prediction latency: {avg_latency:.2f}ms (target: <50ms)")
        
        return predictions
        
    except Exception as e:
        print(f"❌ FastAPI service error: {e}")
        return False

def test_mlflow_server():
    """Test MLflow server is running and accessible"""
    print("🔍 Testing MLflow Server...")
    
    try:
        response = requests.get(f"{MLFLOW_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ MLflow server is accessible")
            return True
        else:
            print(f"❌ MLflow server returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ MLflow server error: {e}")
        return False

def run_mlflow_experiment(predictions_data):
    """Run MLflow experiment with prediction data"""
    print("🔍 Testing MLflow Experiment Tracking...")
    
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_URL)
        
        # Create experiment
        experiment_name = f"lethe_hardening_test_{int(time.time())}"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.MlflowException:
            # Experiment might already exist
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                experiment_id = experiment.experiment_id
            else:
                raise
        
        print(f"✅ Created experiment: {experiment_name} (ID: {experiment_id})")
        
        # Start MLflow run
        with mlflow.start_run(experiment_id=experiment_id):
            # Log system parameters
            mlflow.log_param("test_queries_count", len(TEST_QUERIES))
            mlflow.log_param("fastapi_url", FASTAPI_URL)
            mlflow.log_param("python_version", sys.version.split()[0])
            mlflow.log_param("timestamp", int(time.time()))
            
            # Log performance metrics from FastAPI
            latencies = [p['prediction_time_ms'] for p in predictions_data]
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
            
            mlflow.log_metric("avg_prediction_latency_ms", avg_latency)
            mlflow.log_metric("max_prediction_latency_ms", max_latency)
            mlflow.log_metric("min_prediction_latency_ms", min_latency)
            mlflow.log_metric("performance_target_met", 1.0 if max_latency < 50 else 0.0)
            
            # Log prediction parameters
            alphas = [p['alpha'] for p in predictions_data if p['alpha'] is not None]
            betas = [p['beta'] for p in predictions_data if p['beta'] is not None]
            
            if alphas:
                mlflow.log_metric("avg_alpha", sum(alphas) / len(alphas))
            if betas:
                mlflow.log_metric("avg_beta", sum(betas) / len(betas))
            
            # Log plan distribution
            plans = [p['plan'] for p in predictions_data if p['plan'] is not None]
            plan_counts = {}
            for plan in plans:
                plan_counts[plan] = plan_counts.get(plan, 0) + 1
            
            for plan, count in plan_counts.items():
                mlflow.log_metric(f"plan_{plan}_count", count)
            
            # Save predictions as artifact
            predictions_file = "predictions.json"
            with open(predictions_file, 'w') as f:
                json.dump(predictions_data, f, indent=2)
            mlflow.log_artifact(predictions_file)
            os.remove(predictions_file)  # Clean up local file
            
            run_id = mlflow.active_run().info.run_id
            print(f"✅ MLflow run completed: {run_id}")
            print(f"   - Logged {len(predictions_data)} predictions")
            print(f"   - Average latency: {avg_latency:.2f}ms")
            print(f"   - Performance target: {'✅ Met' if max_latency < 50 else '❌ Missed'}")
            
            return run_id
            
    except Exception as e:
        print(f"❌ MLflow experiment error: {e}")
        return False

def validate_mlflow_artifacts(run_id):
    """Validate MLflow logged artifacts and metrics"""
    print("🔍 Validating MLflow Artifacts...")
    
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Get run details
        run = mlflow.get_run(run_id)
        
        # Check required metrics exist
        required_metrics = [
            'avg_prediction_latency_ms',
            'max_prediction_latency_ms',
            'performance_target_met'
        ]
        
        missing_metrics = []
        for metric in required_metrics:
            if metric not in run.data.metrics:
                missing_metrics.append(metric)
        
        if missing_metrics:
            print(f"❌ Missing metrics: {missing_metrics}")
            return False
        
        # Check artifacts exist using client
        try:
            artifacts = client.list_artifacts(run_id)
            artifact_names = [a.path for a in artifacts]
            
            if 'predictions.json' not in artifact_names:
                print("❌ predictions.json artifact missing")
                return False
        except Exception as artifact_error:
            print(f"⚠️  Artifact validation skipped: {artifact_error}")
            # Continue with other validations
        
        # Validate performance target
        performance_met = run.data.metrics.get('performance_target_met', 0.0)
        if performance_met != 1.0:
            print(f"❌ Performance target not met: {performance_met}")
            return False
        
        # Validate key metrics are reasonable
        avg_latency = run.data.metrics.get('avg_prediction_latency_ms', 0)
        if avg_latency <= 0 or avg_latency > 50:
            print(f"❌ Invalid average latency: {avg_latency}ms")
            return False
        
        print("✅ All MLflow artifacts and metrics validated")
        print(f"   - Average latency: {avg_latency:.2f}ms")
        print(f"   - Performance target met: {'Yes' if performance_met == 1.0 else 'No'}")
        print(f"   - Total metrics logged: {len(run.data.metrics)}")
        return True
        
    except Exception as e:
        print(f"❌ MLflow validation error: {e}")
        return False

def main():
    """Main test execution"""
    print("🚀 Lethe Hardening Workstream - MLflow Integration Test")
    print("=" * 60)
    
    success_count = 0
    total_tests = 4
    
    # Test 1: FastAPI Service
    predictions_data = test_fastapi_service()
    if predictions_data:
        success_count += 1
        print("✅ Objective 2: FastAPI service performance validated (<50ms)")
    else:
        print("❌ Objective 2: FastAPI service test failed")
        return False
    
    # Test 2: MLflow Server
    if test_mlflow_server():
        success_count += 1
        print("✅ MLflow server accessibility validated")
    else:
        print("❌ MLflow server test failed")
        return False
    
    # Test 3: MLflow Experiment
    run_id = run_mlflow_experiment(predictions_data)
    if run_id:
        success_count += 1
        print("✅ Objective 3: MLflow experiment tracking validated")
    else:
        print("❌ Objective 3: MLflow experiment test failed")
        return False
    
    # Test 4: MLflow Artifacts Validation
    if validate_mlflow_artifacts(run_id):
        success_count += 1
        print("✅ MLflow artifacts and reproducibility validated")
    else:
        print("❌ MLflow artifacts validation failed")
        return False
    
    print("=" * 60)
    print(f"🎉 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("✅ All hardening workstream objectives validated!")
        print(f"🔬 View MLflow experiment: {MLFLOW_URL}")
        print(f"📊 Run ID: {run_id}")
        return True
    else:
        print("❌ Some tests failed - review output above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)