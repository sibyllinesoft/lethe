#!/usr/bin/env python3
"""
Test/Demo Script for Expanded Evaluation Suite

This script demonstrates and validates the complete expanded evaluation suite
implementation including all adapter types, parity harness, embedding freezing,
and matrix execution with fail-closed gates.

Usage:
    python test_expanded_evaluation_suite.py --mode [demo|test|canary|full]
    
    demo:   Quick demonstration of key features
    test:   Comprehensive testing of all components
    canary: Run canary validation only
    full:   Run complete evaluation matrix
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluation import (
    ExpandedEvaluationSuite, AdapterRegistry, 
    CorpusSpec, ContextItem
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_adapter_registration():
    """Demonstrate adapter registration and validation."""
    logger.info("=== Demo: Adapter Registration ===")
    
    # Create suite and register adapters
    suite = ExpandedEvaluationSuite(output_dir="demo_results")
    
    # Get adapter summary
    adapter_summary = suite.get_adapter_summary()
    
    print(f"Total adapters registered: {adapter_summary['total_adapters']}")
    print(f"Adapters by type:")
    for adapter_type, adapters in adapter_summary['adapters_by_type'].items():
        print(f"  {adapter_type}: {len(adapters)} adapters")
        for adapter in adapters[:3]:  # Show first 3
            print(f"    - {adapter}")
    
    # Validate setup
    validation = suite.validate_setup()
    print(f"\nSetup validation: {'PASSED' if validation['valid'] else 'FAILED'}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    
    return validation['valid']

def demo_single_evaluation():
    """Demonstrate single sample evaluation."""
    logger.info("=== Demo: Single Sample Evaluation ===")
    
    # Create sample corpus spec
    context_items = [
        ContextItem(
            content="This is the first context item about Python programming. It contains information about functions and classes.",
            item_type="turn",
            timestamp=time.time() - 3600,
            source="user"
        ),
        ContextItem(
            content="def hello_world():\n    print('Hello, World!')\n\nThis function demonstrates basic Python syntax.",
            item_type="code",
            timestamp=time.time() - 1800,
            source="assistant"
        ),
        ContextItem(
            content="Error: NameError at line 42. The variable 'undefined_var' is not defined. This error occurs when trying to use a variable that hasn't been declared.",
            item_type="error",
            timestamp=time.time() - 900,
            source="interpreter"
        )
    ]
    
    spec = CorpusSpec(
        query="How do I fix the NameError in my Python code?",
        context_items=context_items,
        keep_ratio=0.15,
        K=5,
        seed=42,
        sample_id="demo_sample_001"
    )
    
    # Create suite and evaluate
    suite = ExpandedEvaluationSuite(
        datasets=["demo"],  # Won't be used for single evaluation
        output_dir="demo_results"
    )
    
    # Register adapters
    suite.harness.register_all_adapters()
    
    # Evaluate with a few selected adapters
    test_adapters = ["last_k_turns_5", "bm25_lucene", "sliding_window_2048"]
    results = suite.harness.evaluate_sample(spec, adapter_filter=test_adapters)
    
    print(f"Evaluated {len(results)} adapters on demo sample:")
    for method_id, result in results.items():
        sr = result.selection_result
        print(f"  {method_id}:")
        print(f"    Selected atoms: {len(sr.selected_atoms)}")
        print(f"    Total tokens: {sr.total_tokens()}")
        print(f"    Time: {sr.time_ms:.1f}ms (p95: {sr.time_p95:.1f}ms)")
        print(f"    Valid: {result.is_valid}")
        if result.validation_errors:
            print(f"    Errors: {result.validation_errors}")
    
    return len(results) > 0

def test_adapter_types():
    """Test all adapter types individually."""
    logger.info("=== Test: All Adapter Types ===")
    
    # Create test atoms
    test_atoms = [
        ContextItem(
            content=f"Test content {i}: This is sample text for testing adapter functionality. " * 10,
            item_type="turn",
            timestamp=time.time() - i * 100,
            source="test"
        ) for i in range(20)
    ]
    
    spec = CorpusSpec(
        query="Test query for adapter validation",
        context_items=test_atoms,
        keep_ratio=0.20,
        K=10,
        seed=1,
        sample_id="test_adapter_validation"
    )
    
    suite = ExpandedEvaluationSuite(output_dir="test_results")
    suite.harness.register_all_adapters()
    
    # Test each adapter type
    adapter_types = [
        "context_pruning_heuristic",
        "rag_bm25", 
        "rag_vector",
        "rag_hybrid",
        "long_context_sliding"
    ]
    
    results_by_type = {}
    
    for adapter_type in adapter_types:
        type_adapters = [
            method_id for method_id, adapter in AdapterRegistry._adapters.items()
            if adapter.adapter_type.value == adapter_type
        ]
        
        if type_adapters:
            test_adapter = type_adapters[0]  # Test first adapter of this type
            try:
                results = suite.harness.evaluate_sample(spec, adapter_filter=[test_adapter])
                results_by_type[adapter_type] = {
                    'adapter': test_adapter,
                    'success': len(results) > 0 and list(results.values())[0].is_valid,
                    'result': list(results.values())[0] if results else None
                }
                logger.info(f"✓ {adapter_type} test passed")
            except Exception as e:
                results_by_type[adapter_type] = {
                    'adapter': test_adapter,
                    'success': False,
                    'error': str(e)
                }
                logger.error(f"✗ {adapter_type} test failed: {e}")
    
    # Summary
    successful_types = sum(1 for result in results_by_type.values() if result['success'])
    print(f"\nAdapter type testing: {successful_types}/{len(adapter_types)} types passed")
    
    for adapter_type, result in results_by_type.items():
        status = "✓" if result['success'] else "✗"
        print(f"  {status} {adapter_type}: {result['adapter']}")
        if not result['success'] and 'error' in result:
            print(f"    Error: {result['error']}")
    
    return successful_types == len(adapter_types)

def test_canary_validation():
    """Test canary validation functionality.""" 
    logger.info("=== Test: Canary Validation ===")
    
    suite = ExpandedEvaluationSuite(
        datasets=["infinitebench_qa"],  # Use one dataset
        budget_ratios=[0.15],           # One budget ratio
        K_values=[5],                   # One K value
        seeds=[1],                      # One seed
        adapter_filter=["last_k_turns_5", "bm25_lucene", "sliding_window_2048"],  # Few adapters
        output_dir="canary_test_results"
    )
    
    # Run canary validation
    canary_result = suite.run_canary_validation()
    
    print(f"Canary validation: {'PASSED' if canary_result['success'] else 'FAILED'}")
    print(f"Duration: {canary_result['duration_seconds']:.1f}s")
    print(f"Total evaluations: {canary_result.get('total_evaluations', 0)}")
    print(f"Failed gates: {canary_result.get('failed_gates', 0)}")
    
    if not canary_result['success']:
        print("Gate failures:")
        for gate_result in canary_result.get('gate_results', []):
            if gate_result['status'] == 'failed':
                print(f"  - {gate_result['gate_name']}: {gate_result['message']}")
    
    return canary_result['success']

def run_quick_evaluation():
    """Run quick evaluation to test end-to-end functionality."""
    logger.info("=== Test: Quick Evaluation ===")
    
    suite = ExpandedEvaluationSuite(
        datasets=["infinitebench_qa", "conversation_code"],
        budget_ratios=[0.08, 0.15],
        K_values=[1, 5],
        seeds=[1, 2],
        adapter_filter=["last_k_turns_5", "bm25_lucene", "sliding_window_2048", "vector_faiss"],
        output_dir="quick_eval_results",
        enable_embedding_freezing=True,
        enable_fail_closed_gates=True
    )
    
    # Run quick evaluation
    result = suite.run_quick_evaluation()
    
    print(f"Quick evaluation: {'PASSED' if result['success'] else 'FAILED'}")
    print(f"Duration: {result['duration_seconds']:.1f}s")
    print(f"Total evaluations: {result.get('total_evaluations', 0)}")
    print(f"Datasets tested: {result.get('datasets_tested', 0)}")
    print(f"Adapters tested: {result.get('adapters_tested', 0)}")
    print(f"Gate failures: {result.get('gate_failures', 0)}")
    
    if result.get('performance_stats'):
        perf = result['performance_stats']
        print(f"Performance: {perf.get('avg_time_ms', 0):.1f}ms avg, {perf.get('avg_tokens_selected', 0):.0f} tokens avg")
    
    return result['success']

def run_full_evaluation():
    """Run full evaluation matrix (for comprehensive testing)."""
    logger.info("=== Full Evaluation Matrix ===")
    
    suite = ExpandedEvaluationSuite(
        datasets=["infinitebench_qa", "conversation_code"],
        budget_ratios=[0.08, 0.15, 0.30],
        K_values=[1, 5, 10],
        seeds=[1, 2, 3],
        adapter_filter=None,  # All adapters
        output_dir="full_eval_results",
        enable_embedding_freezing=True,
        enable_fail_closed_gates=True,
        parallel_execution=True
    )
    
    # Run complete evaluation
    result = suite.run_complete_evaluation(run_canary_first=True)
    
    print(f"Full evaluation: {'PASSED' if result['success'] else 'FAILED'}")
    print(f"Duration: {result['duration_seconds']:.1f}s")
    print(f"Total combinations: {result.get('total_combinations', 0)}")
    
    if result['success']:
        summary = result['execution_summary']
        print(f"Total evaluations: {summary.get('total_evaluations', 0)}")
        print(f"Total samples: {summary.get('total_samples', 0)}")
        print(f"Gate failures: {summary.get('total_gate_failures', 0)}")
        print(f"Adapter count: {summary.get('adapter_count', 0)}")
        
        outputs = result.get('comprehensive_outputs', {})
        print(f"Generated outputs:")
        for output_type, output_path in outputs.items():
            print(f"  - {output_type}: {output_path}")
    
    return result['success']

def main():
    """Main test/demo execution."""
    parser = argparse.ArgumentParser(description="Test/Demo Expanded Evaluation Suite")
    parser.add_argument('--mode', choices=['demo', 'test', 'canary', 'full'], 
                       default='demo', help='Execution mode')
    
    args = parser.parse_args()
    
    logger.info(f"Starting expanded evaluation suite {args.mode} mode")
    start_time = time.time()
    
    success = True
    
    try:
        if args.mode == 'demo':
            print("=== EXPANDED EVALUATION SUITE DEMONSTRATION ===\n")
            
            # Run demo components
            success &= demo_adapter_registration()
            print()
            success &= demo_single_evaluation()
            
        elif args.mode == 'test':
            print("=== EXPANDED EVALUATION SUITE COMPREHENSIVE TESTING ===\n")
            
            # Run all tests
            success &= demo_adapter_registration()
            print()
            success &= test_adapter_types()
            print()
            success &= test_canary_validation()
            print()
            success &= run_quick_evaluation()
            
        elif args.mode == 'canary':
            print("=== CANARY VALIDATION ONLY ===\n")
            success &= test_canary_validation()
            
        elif args.mode == 'full':
            print("=== FULL EVALUATION MATRIX ===\n")
            success &= run_full_evaluation()
        
        # Final summary
        duration = time.time() - start_time
        status = "PASSED" if success else "FAILED"
        
        print(f"\n{'='*60}")
        print(f"EXPANDED EVALUATION SUITE {args.mode.upper()} MODE: {status}")
        print(f"Total duration: {duration:.1f} seconds")
        print(f"{'='*60}")
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        print(f"\nEXPANDED EVALUATION SUITE {args.mode.upper()} MODE: FAILED")
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())