#!/usr/bin/env python3
"""
Test script for FastAPI ML Prediction Service
Workstream A, Phase 2.1: Verification of HTTP service functionality
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import pytest


async def test_health_endpoint():
    """Test the health check endpoint."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://127.0.0.1:8080/health", timeout=5.0)
            assert response.status_code == 200
            
            health = response.json()
            assert health["status"] == "healthy"
            assert "version" in health
            assert "uptime_seconds" in health
            
            print("✅ Health endpoint working")
            return True
            
        except httpx.ConnectError:
            print("❌ Service not running on port 8080")
            return False
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False


async def test_prediction_endpoint():
    """Test the prediction endpoint with various queries."""
    test_cases = [
        {
            "query": "How to implement binary search in Python?",
            "context": {"bm25_top1": 0.85, "ann_top1": 0.72},
            "expected_plan": "explore"  # How-to question
        },
        {
            "query": "React component not rendering TypeError undefined",
            "context": {"bm25_top1": 0.45, "ann_top1": 0.38},
            "expected_plan": "verify"  # Error debugging
        },
        {
            "query": "getUserById function implementation",
            "context": {"bm25_top1": 0.92, "ann_top1": 0.88},
            "expected_plan": "exploit"  # Code-heavy query
        }
    ]
    
    async with httpx.AsyncClient() as client:
        for i, test_case in enumerate(test_cases, 1):
            try:
                start_time = time.time()
                response = await client.post(
                    "http://127.0.0.1:8080/predict",
                    json=test_case,
                    timeout=10.0
                )
                request_time = (time.time() - start_time) * 1000
                
                assert response.status_code == 200
                prediction = response.json()
                
                # Validate response structure
                assert "alpha" in prediction
                assert "beta" in prediction
                assert "plan" in prediction
                assert "prediction_time_ms" in prediction
                assert "model_loaded" in prediction
                
                # Validate parameter ranges
                assert 0 <= prediction["alpha"] <= 1
                assert 0 <= prediction["beta"] <= 1
                assert prediction["plan"] in ["explore", "verify", "exploit"]
                
                # Performance check
                assert request_time < 1000, f"Request took {request_time:.1f}ms (>1000ms)"
                assert prediction["prediction_time_ms"] < 500, f"Prediction took {prediction['prediction_time_ms']:.1f}ms (>500ms)"
                
                print(f"✅ Test case {i}: {prediction['plan']} plan, {prediction['prediction_time_ms']:.1f}ms")
                
            except Exception as e:
                print(f"❌ Test case {i} failed: {e}")
                return False
    
    print("✅ All prediction tests passed")
    return True


async def test_models_info_endpoint():
    """Test the models info endpoint."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://127.0.0.1:8080/models/info", timeout=5.0)
            assert response.status_code == 200
            
            info = response.json()
            assert "models_loaded" in info
            
            if info["models_loaded"]:
                print("✅ Models loaded successfully")
            else:
                print("⚠️  Models not loaded, fallback mode active")
            
            return True
            
        except Exception as e:
            print(f"❌ Models info test failed: {e}")
            return False


async def run_all_tests():
    """Run all service tests."""
    print("🧪 Testing Lethe ML Prediction Service")
    print("=" * 50)
    
    # Test health endpoint
    health_ok = await test_health_endpoint()
    if not health_ok:
        print("\n❌ Service appears to be down. Start it with: ./start_service.sh")
        return False
    
    # Test prediction endpoint
    prediction_ok = await test_prediction_endpoint()
    
    # Test models info endpoint
    info_ok = await test_models_info_endpoint()
    
    print("\n" + "=" * 50)
    if health_ok and prediction_ok and info_ok:
        print("🎉 All tests passed! Service is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check service logs for details.")
        return False


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        # Run with pytest for CI/CD
        pytest.main([__file__, "-v"])
    else:
        # Run standalone tests
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()