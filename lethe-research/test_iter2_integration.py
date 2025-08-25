#!/usr/bin/env python3
"""
Test Iteration 2 integration end-to-end
"""

import asyncio
import json
import logging
import sys
import os

# Add experiments directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))

from run_iter2 import Iteration2Experiment

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_iteration2_integration():
    """Test the complete Iteration 2 pipeline"""
    
    logger.info("🚀 Testing Iteration 2: Query Understanding Integration")
    
    try:
        # Initialize the experiment
        config_path = "experiments/iter1_dev.yaml"
        experiment = Iteration2Experiment(config_path)
        
        # Test queries that should benefit from query understanding
        test_queries = [
            {
                "query_id": "test_001",
                "query": "How do I handle async errors in TypeScript?",
                "domain": "code",
                "complexity": "medium"
            },
            {
                "query_id": "test_002", 
                "query": "Explain React hooks and their best practices",
                "domain": "mixed",
                "complexity": "medium"
            },
            {
                "query_id": "test_003",
                "query": "Database optimization techniques for performance",
                "domain": "mixed", 
                "complexity": "high"
            }
        ]
        
        logger.info(f"Running {len(test_queries)} test queries...")
        
        results = []
        for query_data in test_queries:
            logger.info(f"Processing: {query_data['query']}")
            
            try:
                result = await experiment.run_single_query(query_data)
                results.append(result)
                
                # Log key metrics
                logger.info(f"  ✅ Completed in {result['latency_ms']:.1f}ms")
                logger.info(f"  📊 Memory: {result['memory_mb']}MB")
                logger.info(f"  🔄 Rewrite success: {result.get('rewrite_success', False)}")
                logger.info(f"  🔀 Decompose success: {result.get('decompose_success', False)}")
                logger.info(f"  🖥️  TypeScript integration: {result.get('typescript_integration', False)}")
                
                # Quality gates check
                gates = result.get('quality_gates', {})
                logger.info(f"  🎯 Quality gates:")
                logger.info(f"     Latency OK: {gates.get('latency_within_budget', False)}")
                logger.info(f"     No rewrite failures: {not gates.get('rewrite_failure', True)}")
                logger.info(f"     No JSON parse errors: {not gates.get('json_parse_error', True)}")
                
            except Exception as e:
                logger.error(f"  ❌ Query failed: {e}")
                results.append({
                    **query_data,
                    "error": str(e),
                    "latency_ms": 0,
                    "success": False
                })
            
            print()
        
        # Calculate aggregate metrics
        successful_queries = [r for r in results if r.get('success', True)]
        total_queries = len(results)
        success_rate = len(successful_queries) / total_queries if total_queries > 0 else 0
        
        if successful_queries:
            avg_latency = sum(r['latency_ms'] for r in successful_queries) / len(successful_queries)
            p95_latency = sorted([r['latency_ms'] for r in successful_queries])[int(0.95 * len(successful_queries))] if len(successful_queries) > 1 else avg_latency
            avg_memory = sum(r['memory_mb'] for r in successful_queries) / len(successful_queries)
            
            rewrite_success_rate = sum(1 for r in successful_queries if r.get('rewrite_success', False)) / len(successful_queries)
            decompose_success_rate = sum(1 for r in successful_queries if r.get('decompose_success', False)) / len(successful_queries)
            typescript_usage = sum(1 for r in successful_queries if r.get('typescript_integration', False)) / len(successful_queries)
        else:
            avg_latency = p95_latency = avg_memory = 0
            rewrite_success_rate = decompose_success_rate = typescript_usage = 0
        
        logger.info("📊 Iteration 2 Summary:")
        logger.info(f"   Success rate: {success_rate:.1%}")
        logger.info(f"   Average latency: {avg_latency:.1f}ms")
        logger.info(f"   P95 latency: {p95_latency:.1f}ms")
        logger.info(f"   Average memory: {avg_memory:.1f}MB")
        logger.info(f"   Rewrite success rate: {rewrite_success_rate:.1%}")
        logger.info(f"   Decompose success rate: {decompose_success_rate:.1%}")
        logger.info(f"   TypeScript integration: {typescript_usage:.1%}")
        
        # Quality gates evaluation
        logger.info("🎯 Quality Gates Assessment:")
        latency_ok = p95_latency <= 3500
        memory_ok = avg_memory <= 1500  # 1500MB budget
        rewrite_failure_rate = 1 - rewrite_success_rate
        
        logger.info(f"   ✅ Latency P95 ≤ 3500ms: {latency_ok} ({p95_latency:.1f}ms)")
        logger.info(f"   ✅ Memory ≤ 1500MB: {memory_ok} ({avg_memory:.1f}MB)")
        logger.info(f"   ✅ Rewrite failure rate ≤ 5%: {rewrite_failure_rate <= 0.05} ({rewrite_failure_rate:.1%})")
        
        # Save results
        output_path = "artifacts/iter2_integration_test_results.json"
        os.makedirs("artifacts", exist_ok=True)
        
        test_summary = {
            "iteration": "iteration_2_query_understanding",
            "timestamp": asyncio.get_event_loop().time(),
            "summary": {
                "total_queries": total_queries,
                "successful_queries": len(successful_queries),
                "success_rate": success_rate,
                "avg_latency_ms": avg_latency,
                "p95_latency_ms": p95_latency,
                "avg_memory_mb": avg_memory,
                "rewrite_success_rate": rewrite_success_rate,
                "decompose_success_rate": decompose_success_rate,
                "typescript_integration_rate": typescript_usage
            },
            "quality_gates": {
                "latency_within_budget": latency_ok,
                "memory_within_budget": memory_ok,
                "rewrite_failure_rate_ok": rewrite_failure_rate <= 0.05,
                "overall_pass": latency_ok and memory_ok and rewrite_failure_rate <= 0.05
            },
            "detailed_results": results
        }
        
        with open(output_path, 'w') as f:
            json.dump(test_summary, f, indent=2)
            
        logger.info(f"📄 Results saved to: {output_path}")
        
        if test_summary["quality_gates"]["overall_pass"]:
            logger.info("🎉 Iteration 2 implementation PASSED all quality gates!")
            return True
        else:
            logger.warning("⚠️ Iteration 2 implementation did not pass all quality gates")
            return False
            
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_iteration2_integration())
    sys.exit(0 if success else 1)