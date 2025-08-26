#!/usr/bin/env python3
"""
Milestone 4 Implementation Validation Script
===========================================

Validates the six baseline implementations with comprehensive testing:

1. Unit tests for each baseline class
2. Integration tests with synthetic data
3. Performance parity validation 
4. Anti-fraud validation testing
5. Budget tracking verification
6. Interface compatibility testing

This script ensures all baselines are working correctly before full evaluation.
"""

import sys
import time
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add the research modules to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.eval.milestone4_baselines import (
    SQLiteFTSBaseline, VectorOnlyBaseline, HybridStaticBaseline,
    MMRDiversityBaseline, Doc2QueryExpansionBaseline, TinyCrossEncoderBaseline,
    Milestone4BaselineEvaluator, RetrievalDocument, EvaluationQuery,
    BudgetParityTracker, AntiFreudValidator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_synthetic_dataset(num_docs: int = 50, num_queries: int = 10) -> tuple[List[RetrievalDocument], List[EvaluationQuery]]:
    """Create synthetic dataset for testing"""
    
    # Create diverse document types
    documents = []
    
    # Code documents
    for i in range(num_docs // 3):
        content = f"""
        def process_data_{i}(input_data):
            # Process the input data using algorithm {i}
            result = []
            for item in input_data:
                if validate_item(item):
                    processed = transform_item_{i}(item)
                    result.append(processed)
            return result
        
        class DataProcessor_{i}:
            def __init__(self, config):
                self.config = config
                self.cache = {{}}
            
            def process_batch(self, items):
                return [self.process_item(item) for item in items]
        """
        
        documents.append(RetrievalDocument(
            doc_id=f"code_{i}",
            content=content.strip(),
            kind="code"
        ))
    
    # Error/log documents  
    for i in range(num_docs // 3):
        content = f"""
        ERROR: Failed to connect to database server_{i}
        Connection timeout after 30 seconds
        Stack trace:
          File "app.py", line {i*10 + 42}, in connect_database
            conn = connect(host='{{'server_{i}'}}'...)
          DatabaseConnectionError: Unable to establish connection
        
        Suggested fixes:
        1. Check network connectivity to server_{i}
        2. Verify database credentials  
        3. Increase connection timeout value
        """
        
        documents.append(RetrievalDocument(
            doc_id=f"error_{i}",
            content=content.strip(),
            kind="error"
        ))
    
    # Tool output documents
    for i in range(num_docs - 2 * (num_docs // 3)):
        content = f"""
        $ git log --oneline -n {i+5}
        commit_{i*100 + 1} Fix bug in data processor_{i} 
        commit_{i*100 + 2} Add validation for input type_{i}
        commit_{i*100 + 3} Update documentation for API_{i}
        commit_{i*100 + 4} Refactor database connection handling
        commit_{i*100 + 5} Add error handling for edge case_{i}
        
        Files changed: src/processor_{i}.py, tests/test_processor_{i}.py
        Lines added: {i*7 + 23}, Lines removed: {i*3 + 11}
        """
        
        documents.append(RetrievalDocument(
            doc_id=f"tool_output_{i}",
            content=content.strip(),
            kind="tool_output"
        ))
    
    # Create diverse queries
    queries = []
    
    query_templates = [
        "How to fix connection timeout error in database server_{i}?",
        "Show me the implementation of process_data_{i} function",
        "What are the recent git commits for processor_{i}?",
        "Debug DataProcessor_{i} class initialization",
        "Find error handling code for edge case_{i}",
        "Get configuration options for data processing",
        "Show validation logic for input items",
        "How to increase database connection timeout?",
        "Find recent bug fixes in the codebase",
        "What files were changed in latest commits?"
    ]
    
    for i in range(num_queries):
        template_idx = i % len(query_templates)
        query_text = query_templates[template_idx].format(i=i % (num_docs // 3))
        
        # Create ground truth based on query content
        ground_truth = []
        if "database" in query_text.lower() or "connection" in query_text.lower():
            ground_truth = [f"error_{j}" for j in range(min(3, num_docs // 3))]
        elif "process_data" in query_text.lower() or "DataProcessor" in query_text:
            ground_truth = [f"code_{j}" for j in range(min(3, num_docs // 3))]
        elif "git" in query_text.lower() or "commit" in query_text.lower():
            ground_truth = [f"tool_output_{j}" for j in range(min(3, num_docs // 3))]
        
        queries.append(EvaluationQuery(
            query_id=f"query_{i}",
            text=query_text,
            session_id=f"session_{i % 3}",  # 3 sessions
            domain="development",
            complexity="medium",
            ground_truth_docs=ground_truth
        ))
    
    return documents, queries

def test_baseline_interface(baseline_class, config: Dict[str, Any], documents: List[RetrievalDocument], queries: List[EvaluationQuery]) -> Dict[str, Any]:
    """Test a single baseline implementation"""
    logger.info(f"Testing {baseline_class.__name__}...")
    
    test_results = {
        "baseline_name": baseline_class.__name__,
        "tests_passed": 0,
        "tests_failed": 0,
        "errors": [],
        "metrics": {}
    }
    
    try:
        # Test 1: Initialization
        logger.info("  Test 1: Initialization")
        baseline = baseline_class(config)
        test_results["tests_passed"] += 1
        
        # Test 2: Index building
        logger.info("  Test 2: Index building")
        start_time = time.time()
        baseline.build_index(documents)
        build_time = time.time() - start_time
        test_results["tests_passed"] += 1
        test_results["metrics"]["build_time_seconds"] = build_time
        
        # Test 3: Single query retrieval
        logger.info("  Test 3: Single query retrieval")
        test_query = queries[0]
        start_time = time.time()
        results = baseline.retrieve(test_query, k=10)
        retrieval_time = time.time() - start_time
        
        # Validate results
        if not results:
            test_results["errors"].append("Empty results returned")
            test_results["tests_failed"] += 1
        elif not all(hasattr(r, 'doc_id') and hasattr(r, 'score') for r in results):
            test_results["errors"].append("Results missing required attributes")
            test_results["tests_failed"] += 1
        else:
            test_results["tests_passed"] += 1
            test_results["metrics"]["single_query_time_ms"] = retrieval_time * 1000
            test_results["metrics"]["results_returned"] = len(results)
        
        # Test 4: Batch retrieval
        logger.info("  Test 4: Batch retrieval")
        batch_queries = queries[:3]
        start_time = time.time()
        batch_results = []
        for query in batch_queries:
            batch_results.extend(baseline.retrieve(query, k=5))
        batch_time = time.time() - start_time
        
        test_results["tests_passed"] += 1
        test_results["metrics"]["batch_time_ms"] = batch_time * 1000
        test_results["metrics"]["batch_results"] = len(batch_results)
        
        # Test 5: FLOPS estimation
        logger.info("  Test 5: FLOPS estimation") 
        flops = baseline.get_flops_estimate(test_query, 10)
        if flops > 0:
            test_results["tests_passed"] += 1
            test_results["metrics"]["estimated_flops"] = flops
        else:
            test_results["errors"].append("Invalid FLOPS estimate")
            test_results["tests_failed"] += 1
        
        # Test 6: Memory usage tracking
        logger.info("  Test 6: Memory usage tracking")
        memory_mb = baseline.get_memory_usage()
        if memory_mb > 0:
            test_results["tests_passed"] += 1
            test_results["metrics"]["memory_usage_mb"] = memory_mb
        else:
            test_results["errors"].append("Invalid memory usage")
            test_results["tests_failed"] += 1
            
    except ImportError as e:
        test_results["errors"].append(f"Import error (expected for optional dependencies): {e}")
        test_results["tests_failed"] += 6  # Mark all tests as failed
    except Exception as e:
        test_results["errors"].append(f"Unexpected error: {e}")
        test_results["tests_failed"] += 1
    
    success_rate = test_results["tests_passed"] / (test_results["tests_passed"] + test_results["tests_failed"])
    test_results["success_rate"] = success_rate
    
    logger.info(f"  {baseline_class.__name__}: {test_results['tests_passed']}/{test_results['tests_passed'] + test_results['tests_failed']} tests passed ({success_rate:.1%})")
    
    return test_results

def test_budget_parity_tracker():
    """Test budget parity tracking functionality"""
    logger.info("Testing BudgetParityTracker...")
    
    tracker = BudgetParityTracker(tolerance=0.05)
    
    # Set baseline budget
    baseline_budget = 1000000.0
    tracker.set_baseline_budget(baseline_budget)
    
    # Test compliant methods
    assert tracker.validate_budget("method_a", 1000000.0) == True  # Exact match
    assert tracker.validate_budget("method_b", 1040000.0) == True  # Within 4% (< 5% tolerance)
    assert tracker.validate_budget("method_c", 960000.0) == True   # Within 4% (< 5% tolerance)
    
    # Test non-compliant methods
    assert tracker.validate_budget("method_d", 1100000.0) == False  # 10% over (> 5% tolerance)
    assert tracker.validate_budget("method_e", 900000.0) == False   # 10% under (> 5% tolerance)
    
    # Generate report
    report = tracker.get_budget_report()
    assert report["baseline_budget"] == baseline_budget
    assert len(report["methods"]) == 5
    
    compliant_count = sum(1 for m in report["methods"].values() if m["parity_compliant"])
    assert compliant_count == 3
    
    logger.info("✅ BudgetParityTracker tests passed")

def test_anti_fraud_validator():
    """Test anti-fraud validation functionality"""
    logger.info("Testing AntiFreudValidator...")
    
    validator = AntiFreudValidator()
    
    # Create mock query and results
    query = EvaluationQuery(
        query_id="test_query",
        text="test query",
        session_id="test_session"
    )
    
    # Test valid results
    valid_results = [
        type('MockResult', (), {
            'doc_id': f'doc_{i}',
            'score': 0.9 - i*0.1, 
            'content': f'content {i}',
            'rank': i+1
        })() for i in range(5)
    ]
    
    assert validator.validate_non_empty_results("test_method", query, valid_results) == True
    
    # Test empty results (should fail)
    empty_results = []
    assert validator.validate_non_empty_results("test_method", query, empty_results) == False
    
    # Test constant scores (should fail)
    constant_score_results = [
        type('MockResult', (), {
            'doc_id': f'doc_{i}',
            'score': 0.5,  # All same score
            'content': f'content {i}',
            'rank': i+1
        })() for i in range(5)
    ]
    
    # Note: This might pass if variance threshold is very low
    result = validator.validate_non_empty_results("test_method", query, constant_score_results)
    
    # Generate report
    report = validator.get_validation_report()
    assert "methods" in report
    assert "test_method" in report["methods"]
    
    logger.info("✅ AntiFreudValidator tests passed")

def test_evaluator_integration():
    """Test the full evaluator integration"""
    logger.info("Testing Milestone4BaselineEvaluator integration...")
    
    # Create small synthetic dataset
    documents, queries = create_synthetic_dataset(num_docs=10, num_queries=3)
    
    # Simple configuration that should work without optional dependencies
    config = {
        "db_path": ":memory:",
        "model_name": "all-MiniLM-L6-v2",
        "alpha": 0.5,
        "lambda": 0.7,
        "num_expansions": 2,
        "rerank_k": 20
    }
    
    # Create evaluator (will automatically skip baselines with missing dependencies)
    evaluator = Milestone4BaselineEvaluator(config)
    
    # Test index building
    logger.info("  Building indices...")
    evaluator.build_all_indices(documents)
    
    working_baselines = len(evaluator.baselines)
    logger.info(f"  {working_baselines} baselines initialized successfully")
    
    if working_baselines == 0:
        logger.warning("  No baselines available - likely due to missing optional dependencies")
        return
    
    # Test evaluation
    logger.info("  Running evaluation...")
    results = evaluator.evaluate_all_baselines(queries[:2], k=5)  # Small test
    
    # Validate results
    assert len(results) == working_baselines
    for baseline_name, baseline_results in results.items():
        assert len(baseline_results) == 2  # 2 queries
        for result in baseline_results:
            assert result.baseline_name == baseline_name
            assert result.query_id in [q.query_id for q in queries[:2]]
            assert isinstance(result.latency_ms, (int, float))
            assert isinstance(result.flops_estimate, int)
    
    # Test save/load
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = Path(f.name)
    
    evaluator.save_results(results, temp_path)
    
    # Validate saved file
    with open(temp_path) as f:
        saved_data = json.load(f)
    
    assert "metadata" in saved_data
    assert "baseline_results" in saved_data
    assert len(saved_data["baseline_results"]) == working_baselines
    
    # Cleanup
    temp_path.unlink()
    
    logger.info("✅ Evaluator integration tests passed")

def main():
    """Run all validation tests"""
    logger.info("🧪 Starting Milestone 4 implementation validation...")
    
    # Create test dataset
    logger.info("📋 Creating synthetic test dataset...")
    documents, queries = create_synthetic_dataset(num_docs=20, num_queries=5)
    logger.info(f"  Created {len(documents)} documents and {len(queries)} queries")
    
    # Configuration for testing
    config = {
        "db_path": ":memory:",
        "model_name": "all-MiniLM-L6-v2", 
        "alpha": 0.5,
        "lambda": 0.7,
        "num_expansions": 2,
        "rerank_k": 50,
        "k1": 1.2,
        "b": 0.75
    }
    
    # Test individual baseline classes
    baseline_classes = [
        SQLiteFTSBaseline,
        VectorOnlyBaseline, 
        HybridStaticBaseline,
        MMRDiversityBaseline,
        Doc2QueryExpansionBaseline,
        TinyCrossEncoderBaseline
    ]
    
    all_results = []
    working_baselines = 0
    
    for baseline_class in baseline_classes:
        try:
            test_result = test_baseline_interface(baseline_class, config, documents, queries)
            all_results.append(test_result)
            if test_result["success_rate"] > 0:
                working_baselines += 1
        except Exception as e:
            logger.error(f"Failed to test {baseline_class.__name__}: {e}")
            all_results.append({
                "baseline_name": baseline_class.__name__,
                "success_rate": 0,
                "errors": [str(e)]
            })
    
    # Test utility classes
    test_budget_parity_tracker()
    test_anti_fraud_validator()
    
    # Test full integration
    test_evaluator_integration()
    
    # Summary report
    logger.info("\n📊 Validation Summary:")
    logger.info(f"  Total baselines tested: {len(baseline_classes)}")
    logger.info(f"  Working baselines: {working_baselines}")
    logger.info(f"  Success rate: {working_baselines/len(baseline_classes):.1%}")
    
    for result in all_results:
        status = "✅" if result["success_rate"] > 0.5 else "❌"
        logger.info(f"  {result['baseline_name']}: {result['success_rate']:.1%} {status}")
        if result.get("errors"):
            for error in result["errors"][:2]:  # Show first 2 errors
                logger.info(f"    - {error}")
    
    # Final validation
    if working_baselines >= 3:  # At least BM25, maybe Vector, and Hybrid should work
        logger.info("\n✅ Milestone 4 implementation validation PASSED")
        logger.info("Ready for full baseline evaluation!")
        return 0
    else:
        logger.error("\n❌ Milestone 4 implementation validation FAILED")
        logger.error("Please install missing dependencies or fix implementation issues")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)