#!/usr/bin/env python3
"""
Iteration 3: Integration Test for Dynamic Fusion & Learned Planning

This test validates the complete ML-enhanced pipeline including:
1. Model training and loading
2. TypeScript integration via subprocess
3. Dynamic parameter prediction
4. Learned plan selection
5. Performance constraints validation
"""

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any
import statistics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Iteration3IntegrationTest:
    """Complete integration test suite for Iteration 3."""
    
    def __init__(self):
        self.test_results = {
            'timestamp': time.time(),
            'test_type': 'iteration_3_integration',
            'ml_models': {
                'trained': False,
                'loaded': False,
                'performance_metrics': {}
            },
            'typescript_integration': {
                'success': False,
                'test_queries': []
            },
            'performance_validation': {
                'latency_tests': [],
                'quality_gates': {}
            },
            'overall_success': False
        }
        
    def run_complete_test(self) -> Dict[str, Any]:
        """Run the complete Iteration 3 integration test suite."""
        
        logger.info("🚀 Starting Iteration 3: Dynamic Fusion & Learned Planning Integration Test")
        
        try:
            # Step 1: Train and validate ML models
            logger.info("Step 1: ML Model Training & Validation")
            model_success = self.test_ml_training()
            self.test_results['ml_models']['trained'] = model_success
            
            if not model_success:
                logger.error("ML model training failed, aborting integration test")
                return self.test_results
            
            # Step 2: Test TypeScript integration
            logger.info("Step 2: TypeScript Integration Testing")  
            ts_success = self.test_typescript_integration()
            self.test_results['typescript_integration']['success'] = ts_success
            
            # Step 3: Performance validation
            logger.info("Step 3: Performance & Quality Gate Validation")
            perf_success = self.test_performance_constraints()
            
            # Step 4: End-to-end validation
            logger.info("Step 4: End-to-End ML Pipeline Validation")
            e2e_success = self.test_end_to_end_pipeline()
            
            # Overall success determination
            overall_success = model_success and ts_success and perf_success and e2e_success
            self.test_results['overall_success'] = overall_success
            
            if overall_success:
                logger.info("✅ Iteration 3 integration test PASSED")
            else:
                logger.warning("⚠️ Iteration 3 integration test had issues")
                
        except Exception as e:
            logger.error(f"Integration test failed with error: {e}")
            self.test_results['error'] = str(e)
            
        return self.test_results
    
    def test_ml_training(self) -> bool:
        """Test ML model training and basic functionality."""
        
        try:
            # Import and run training
            from experiments.iter3_ml_training import train_iteration3_models
            
            logger.info("Training ML models...")
            training_results = train_iteration3_models()
            
            # Validate training results
            fusion_results = training_results.get('training_results', {}).get('fusion_model', {})
            plan_results = training_results.get('training_results', {}).get('plan_selector', {})
            validation_results = training_results.get('validation', {})
            
            # Check model performance thresholds
            fusion_success = (
                fusion_results.get('alpha_mse', float('inf')) < 0.1 and
                fusion_results.get('beta_mse', float('inf')) < 0.1
            )
            
            plan_success = plan_results.get('accuracy', 0) > 0.3  # Basic threshold
            
            loading_success = validation_results.get('load_success', False)
            speed_success = validation_results.get('speed_requirement_met', False)
            
            self.test_results['ml_models']['performance_metrics'] = {
                'fusion_alpha_mse': fusion_results.get('alpha_mse'),
                'fusion_beta_mse': fusion_results.get('beta_mse'),
                'plan_accuracy': plan_results.get('accuracy'),
                'avg_prediction_time_ms': validation_results.get('avg_prediction_time_ms'),
                'max_prediction_time_ms': validation_results.get('max_prediction_time_ms'),
                'load_success': loading_success,
                'speed_requirement_met': speed_success
            }
            
            success = fusion_success and plan_success and loading_success and speed_success
            
            if success:
                logger.info(f"✅ ML training successful - Fusion MSE: α={fusion_results.get('alpha_mse', 0):.4f}, β={fusion_results.get('beta_mse', 0):.4f}, Plan accuracy: {plan_results.get('accuracy', 0):.3f}")
            else:
                logger.error(f"❌ ML training failed validation - Fusion: {fusion_success}, Plan: {plan_success}, Loading: {loading_success}, Speed: {speed_success}")
                
            return success
            
        except Exception as e:
            logger.error(f"ML training test failed: {e}")
            return False
    
    def test_typescript_integration(self) -> bool:
        """Test TypeScript integration with ML predictions."""
        
        test_queries = [
            "async error handling in TypeScript",
            "React component optimization techniques", 
            "database indexing and performance tuning",
            "authentication middleware implementation",
            "microservices communication patterns"
        ]
        
        ctx_run_path = Path(__file__).parent.parent / "ctx-run"
        
        if not ctx_run_path.exists():
            logger.error(f"ctx-run directory not found at {ctx_run_path}")
            return False
        
        successful_tests = 0
        
        for query in test_queries:
            try:
                # Create TypeScript test script for ML integration
                js_code = f"""
                const {{ enhancedQuery }} = require('./packages/core/dist/index.js');
                const {{ migrate }} = require('./packages/sqlite/dist/index.js');
                const Database = require('better-sqlite3');
                
                async function testMLIntegration() {{
                    const db = new Database(':memory:');
                    migrate(db);
                    
                    // Mock embeddings
                    const mockEmbeddings = {{
                        embed: async (text) => Array(384).fill(0),
                        dimension: 384
                    }};
                    
                    try {{
                        const result = await enhancedQuery("{query}", {{
                            db,
                            embeddings: mockEmbeddings,
                            sessionId: "iter3-test-session",
                            enableQueryUnderstanding: true,
                            enableHyde: false,
                            enableSummarization: false,
                            enablePlanSelection: true,
                            // Iteration 3: Enable ML features
                            enableMLPrediction: true,
                            mlConfig: {{
                                fusion_dynamic: true,
                                plan_learned: true
                            }}
                        }});
                        
                        console.log(JSON.stringify({{
                            success: true,
                            query: "{query}",
                            duration: result.duration,
                            plan: result.plan,
                            mlPrediction: result.mlPrediction,
                            debug: result.debug
                        }}));
                    }} catch (error) {{
                        console.log(JSON.stringify({{
                            success: false,
                            error: error.message,
                            query: "{query}"
                        }}));
                    }}
                    
                    db.close();
                }}
                
                testMLIntegration().catch(console.error);
                """
                
                # Run TypeScript test
                result = subprocess.run(
                    ['node', '-e', js_code],
                    cwd=ctx_run_path,
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                
                if result.returncode == 0:
                    try:
                        response = json.loads(result.stdout.strip())
                        
                        if response.get('success', False):
                            # Validate ML integration
                            ml_prediction = response.get('mlPrediction')
                            has_ml_results = (
                                ml_prediction and 
                                'alpha' in ml_prediction and 
                                'beta' in ml_prediction and
                                'predicted_plan' in ml_prediction
                            )
                            
                            if has_ml_results:
                                logger.info(f"✅ ML integration test passed for: {query[:30]}...")
                                successful_tests += 1
                                
                                self.test_results['typescript_integration']['test_queries'].append({
                                    'query': query,
                                    'success': True,
                                    'ml_alpha': ml_prediction.get('alpha'),
                                    'ml_beta': ml_prediction.get('beta'),
                                    'ml_plan': ml_prediction.get('predicted_plan'),
                                    'prediction_time_ms': ml_prediction.get('prediction_time_ms'),
                                    'total_time_ms': response.get('duration', {}).get('total', 0)
                                })
                            else:
                                logger.warning(f"⚠️ No ML predictions for: {query[:30]}...")
                        else:
                            logger.warning(f"⚠️ TypeScript test failed for: {query[:30]}...")
                            
                    except json.JSONDecodeError:
                        logger.error(f"❌ Could not parse TypeScript response for: {query[:30]}...")
                        
                else:
                    logger.error(f"❌ TypeScript process failed for: {query[:30]}... - {result.stderr}")
                    
            except Exception as e:
                logger.error(f"TypeScript integration test error for {query[:30]}...: {e}")
        
        success_rate = successful_tests / len(test_queries)
        success = success_rate >= 0.8  # 80% success rate threshold
        
        logger.info(f"TypeScript integration: {successful_tests}/{len(test_queries)} tests passed ({success_rate:.1%})")
        
        return success
    
    def test_performance_constraints(self) -> bool:
        """Test performance constraints and quality gates."""
        
        try:
            from experiments.iter3_prediction import create_production_predictor
            
            # Create predictor and test performance
            predictor = create_production_predictor('models')
            
            # Test model loading time
            load_time = predictor.get_load_time()
            load_success = load_time <= 200  # 200ms budget
            
            # Test prediction speed with various queries
            test_queries = [
                "simple query",
                "async error handling in TypeScript with complex debugging",
                "how to implement authentication middleware for microservices architecture",
                "performance optimization techniques for React components in large applications",
                "database indexing strategies for high-throughput OLTP systems"
            ]
            
            prediction_times = []
            
            for query in test_queries:
                start_time = time.time()
                result = predictor.predict_all(query)
                prediction_time = (time.time() - start_time) * 1000
                
                prediction_times.append(prediction_time)
                
                # Validate prediction format
                has_valid_results = (
                    'alpha' in result and
                    'beta' in result and
                    'plan' in result and
                    0.3 <= result['alpha'] <= 1.5 and
                    0.3 <= result['beta'] <= 1.5 and
                    result['plan'] in ['explore', 'verify', 'exploit']
                )
                
                if not has_valid_results:
                    logger.error(f"Invalid prediction results for query: {query[:30]}...")
                    return False
            
            # Performance statistics
            avg_prediction_time = statistics.mean(prediction_times)
            max_prediction_time = max(prediction_times)
            p95_prediction_time = sorted(prediction_times)[int(0.95 * len(prediction_times))]
            
            # Performance constraints validation
            speed_success = max_prediction_time <= 50  # 50ms individual prediction budget
            avg_speed_success = avg_prediction_time <= 25  # Average should be even faster
            
            self.test_results['performance_validation'] = {
                'load_time_ms': load_time,
                'load_success': load_success,
                'avg_prediction_time_ms': avg_prediction_time,
                'max_prediction_time_ms': max_prediction_time,
                'p95_prediction_time_ms': p95_prediction_time,
                'speed_success': speed_success,
                'avg_speed_success': avg_speed_success,
                'prediction_times': prediction_times
            }
            
            overall_success = load_success and speed_success and avg_speed_success
            
            if overall_success:
                logger.info(f"✅ Performance validation passed - Load: {load_time:.1f}ms, Avg predict: {avg_prediction_time:.1f}ms, Max: {max_prediction_time:.1f}ms")
            else:
                logger.error(f"❌ Performance validation failed - Load: {load_success}, Speed: {speed_success}, Avg: {avg_speed_success}")
                
            return overall_success
            
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            return False
    
    def test_end_to_end_pipeline(self) -> bool:
        """Test complete end-to-end ML-enhanced pipeline."""
        
        try:
            # Test queries representing different complexity levels
            test_cases = [
                {
                    'query': 'TypeScript async await error handling',
                    'expected_alpha_range': (0.6, 1.0),  # Code queries should favor lexical
                    'expected_plan': ['exploit', 'explore'],  # Likely plans
                    'complexity': 'medium'
                },
                {
                    'query': 'undefined reference exception stack trace debugging',
                    'expected_alpha_range': (0.4, 0.8),
                    'expected_plan': ['verify', 'explore'],  # Error queries
                    'complexity': 'high'
                },
                {
                    'query': 'best practices guide',
                    'expected_alpha_range': (0.3, 0.7),  # General queries favor semantic
                    'expected_plan': ['explore', 'exploit'],
                    'complexity': 'low'
                }
            ]
            
            from experiments.iter3_prediction import create_production_predictor
            predictor = create_production_predictor('models')
            
            successful_cases = 0
            
            for test_case in test_cases:
                query = test_case['query']
                
                # Get ML prediction
                result = predictor.predict_all(query)
                
                # Validate alpha/beta ranges
                alpha_in_range = (
                    test_case['expected_alpha_range'][0] <= result['alpha'] <= test_case['expected_alpha_range'][1]
                )
                
                # Validate plan prediction
                plan_valid = result['plan'] in test_case['expected_plan']
                
                # Validate prediction speed
                speed_ok = result.get('prediction_time_ms', float('inf')) <= 50
                
                # Validate parameter clamping
                params_clamped = (
                    0.3 <= result['alpha'] <= 1.5 and
                    0.3 <= result['beta'] <= 1.5
                )
                
                case_success = alpha_in_range and plan_valid and speed_ok and params_clamped
                
                if case_success:
                    successful_cases += 1
                    logger.info(f"✅ E2E test passed for {test_case['complexity']} query: α={result['alpha']:.3f}, β={result['beta']:.3f}, plan={result['plan']}")
                else:
                    logger.warning(f"⚠️ E2E test issues for {test_case['complexity']} query: α_range={alpha_in_range}, plan={plan_valid}, speed={speed_ok}, clamp={params_clamped}")
            
            success_rate = successful_cases / len(test_cases)
            pipeline_success = success_rate >= 0.8
            
            logger.info(f"End-to-end pipeline: {successful_cases}/{len(test_cases)} tests passed ({success_rate:.1%})")
            
            return pipeline_success
            
        except Exception as e:
            logger.error(f"End-to-end pipeline test failed: {e}")
            return False


def main():
    """Run the complete Iteration 3 integration test."""
    
    test_suite = Iteration3IntegrationTest()
    results = test_suite.run_complete_test()
    
    # Save results
    timestamp = int(time.time())
    results_path = f"artifacts/iter3_integration_test_{timestamp}.json"
    
    Path("artifacts").mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Integration test results saved to {results_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("ITERATION 3: DYNAMIC FUSION & LEARNED PLANNING")
    print("Integration Test Summary")
    print("="*60)
    
    print(f"📊 ML Models Trained: {'✅' if results['ml_models']['trained'] else '❌'}")
    print(f"🔧 TypeScript Integration: {'✅' if results['typescript_integration']['success'] else '❌'}")
    print(f"⚡ Performance Validation: {'✅' if results.get('performance_validation', {}).get('load_success', False) else '❌'}")
    print(f"🎯 Overall Success: {'✅' if results['overall_success'] else '❌'}")
    
    if results['overall_success']:
        print("\n🎉 Iteration 3 implementation is ready for evaluation!")
        
        # Print key metrics
        ml_metrics = results['ml_models']['performance_metrics']
        if ml_metrics:
            print(f"\n📈 Key Performance Metrics:")
            print(f"   • Model Load Time: {results.get('performance_validation', {}).get('load_time_ms', 0):.1f}ms")
            print(f"   • Avg Prediction Time: {ml_metrics.get('avg_prediction_time_ms', 0):.1f}ms")
            print(f"   • Plan Accuracy: {ml_metrics.get('plan_accuracy', 0):.1%}")
            print(f"   • Fusion MSE: α={ml_metrics.get('fusion_alpha_mse', 0):.4f}, β={ml_metrics.get('fusion_beta_mse', 0):.4f}")
    else:
        print("\n⚠️ Iteration 3 implementation needs attention before evaluation.")
        
    return results['overall_success']


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)