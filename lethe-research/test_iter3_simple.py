#!/usr/bin/env python3
"""
Simple test to validate Iteration 3 ML integration works end-to-end
"""

import json
import subprocess
import time
from pathlib import Path

def test_simple_ml_integration():
    """Test basic ML integration with a single query."""
    
    print("🧪 Testing Iteration 3 ML Integration")
    
    ctx_run_path = Path(__file__).parent.parent / "ctx-run"
    
    # Simple test script
    js_code = '''
    const { enhancedQuery } = require('./packages/core/dist/index.js');
    const { migrate } = require('./packages/sqlite/dist/index.js');
    const Database = require('better-sqlite3');

    async function testML() {
        const db = new Database(':memory:');
        migrate(db);

        const mockEmbeddings = {
            embed: async (text) => Array(384).fill(0),
            dimension: 384
        };

        try {
            const result = await enhancedQuery("TypeScript async error handling", {
                db,
                embeddings: mockEmbeddings,
                sessionId: "test-ml",
                enableQueryUnderstanding: false,
                enableHyde: false,
                enableSummarization: false,
                enablePlanSelection: true,
                enableMLPrediction: true,
                mlConfig: {
                    fusion_dynamic: true,
                    plan_learned: true
                },
                debug: true
            });

            console.log(JSON.stringify({
                success: true,
                plan: result.plan.plan,
                reasoning: result.plan.reasoning,
                mlPrediction: result.mlPrediction,
                totalTime: result.duration.total
            }));

        } catch (error) {
            console.log(JSON.stringify({
                success: false,
                error: error.message,
                stack: error.stack
            }));
        }

        db.close();
    }

    testML().catch(console.error);
    '''
    
    try:
        result = subprocess.run(
            ['node', '-e', js_code],
            cwd=ctx_run_path,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            try:
                # Extract JSON from output (it may contain debug logs)
                stdout_lines = result.stdout.strip().split('\n')
                json_line = None
                for line in reversed(stdout_lines):  # Start from the end to find the JSON result
                    if line.startswith('{') and 'success' in line:
                        json_line = line
                        break
                
                if not json_line:
                    raise ValueError("No JSON result found in output")
                    
                response = json.loads(json_line)
                
                if response.get('success'):
                    print("✅ ML Integration Test PASSED")
                    print(f"   Plan: {response.get('plan', 'unknown')}")
                    print(f"   Reasoning: {response.get('reasoning', 'N/A')}")
                    
                    ml_pred = response.get('mlPrediction')
                    if ml_pred:
                        print(f"   ML Alpha: {ml_pred.get('alpha', 'N/A')}")
                        print(f"   ML Beta: {ml_pred.get('beta', 'N/A')}")
                        print(f"   ML Plan: {ml_pred.get('predicted_plan', 'N/A')}")
                        print(f"   Prediction Time: {ml_pred.get('prediction_time_ms', 0)}ms")
                    
                    print(f"   Total Time: {response.get('totalTime', 0)}ms")
                    return True
                else:
                    print("❌ ML Integration Test FAILED")
                    print(f"   Error: {response.get('error', 'Unknown')}")
                    return False
                    
            except json.JSONDecodeError:
                print("❌ Failed to parse response")
                print(f"   stdout: {result.stdout}")
                print(f"   stderr: {result.stderr}")
                return False
        else:
            print("❌ Node.js process failed")
            print(f"   Return code: {result.returncode}")
            print(f"   stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


if __name__ == "__main__":
    success = test_simple_ml_integration()
    exit(0 if success else 1)