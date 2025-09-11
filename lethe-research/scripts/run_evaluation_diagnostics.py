#!/usr/bin/env python3
"""
Standalone Evaluation Diagnostics Script
========================================

This script runs comprehensive diagnostics on the existing evaluation pipeline
to isolate exactly where the accuracy=0.000 problem is occurring.

Usage:
    python scripts/run_evaluation_diagnostics.py [--task code_debug] [--samples 50]

The script will:
1. Load real evaluation data from the infinitebench dataset
2. Test the three diagnostic tiers to isolate failures
3. Provide specific recommendations for fixing the identified issues

Author: Lethe Research Team
"""

import sys
import logging
import argparse
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.eval.evaluation_diagnostics import EvaluationDiagnostics
from src.infinitebench.dataset_loader import InfiniteBenchLoader
from src.infinitebench.metrics import InfiniteBenchMetrics

logger = logging.getLogger(__name__)

def mock_streaming_method(sample):
    """
    Mock streaming method that simulates the current evaluation pipeline.
    This will help us test the diagnostic system itself.
    """
    # Simulate the streaming method returning some text
    context = sample.get('context', '')
    question = sample.get('question', '')
    
    # Simulate text generation (very basic)
    if 'debug' in question.lower():
        response_text = "The bug is in line 42 of the function."
    elif 'what' in question.lower():
        response_text = "The answer is 42."
    elif question:
        response_text = f"Based on the context, {question.split()[-1] if question.split() else 'unknown'}"
    else:
        response_text = "No clear answer found in the provided context."
    
    return type('MockResponse', (), {
        'text': response_text,
        'stop_reason': 'length',
        'finish_reason': 'stop',
        'tokens': response_text.split(),
        'processing_time_ms': 150.0
    })()

def load_real_evaluation_data(data_dir: Path, task_name: str, max_samples: int = 50):
    """Load real evaluation data for diagnostics."""
    logger.info(f"Loading {task_name} data from {data_dir}")
    
    loader = InfiniteBenchLoader(data_dir)
    
    try:
        samples = loader.load_task(task_name)
        if len(samples) > max_samples:
            samples = samples[:max_samples]
        
        logger.info(f"Loaded {len(samples)} samples for task {task_name}")
        
        # Convert to diagnostic format
        diagnostic_samples = []
        for sample in samples:
            diagnostic_samples.append({
                'id': sample.id,
                'question': sample.question,
                'context': sample.context,
                'answer': sample.answer,
                'language': getattr(sample, 'language', 'en')
            })
        
        return diagnostic_samples
        
    except Exception as e:
        logger.error(f"Failed to load {task_name}: {e}")
        return []

def run_pipeline_diagnostics(task_name: str = "code_debug", max_samples: int = 50):
    """Run comprehensive diagnostics on the evaluation pipeline."""
    
    # Initialize paths
    data_dir = Path("benchmarks/infinitebench/data")
    if not data_dir.exists():
        data_dir = Path("../benchmarks/infinitebench/data")
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return False
    
    # Load evaluation data
    samples = load_real_evaluation_data(data_dir, task_name, max_samples)
    if not samples:
        logger.error(f"No samples loaded for task {task_name}")
        return False
    
    # Initialize diagnostics
    diagnostics = EvaluationDiagnostics(data_dir)
    
    # Run comprehensive diagnostic
    logger.info(f"🚀 Starting comprehensive diagnostic for {task_name}")
    
    try:
        result = diagnostics.run_comprehensive_diagnostic(
            method_fn=mock_streaming_method,
            samples=samples,
            task_type=task_name,
            experiment_name=f"pipeline_diagnostic_{task_name}"
        )
        
        # Print diagnostic report
        diagnostics.print_diagnostic_report(result)
        
        # Save detailed results
        output_dir = Path("diagnostic_results")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / f"pipeline_diagnostic_{task_name}.json"
        diagnostics.save_diagnostic_report(result, output_path)
        
        # Generate specific recommendations based on results
        print("\n" + "="*80)
        print("🔧 SPECIFIC REPAIR RECOMMENDATIONS")
        print("="*80)
        
        if not result.tier_results["gold_echo"].success:
            print("❌ CRITICAL: Scorer/Normalization BROKEN")
            print("   → Check normalize_answer() function in evaluation pipeline")
            print("   → Verify Unicode normalization (NFKC) is working correctly")
            print("   → Test token F1 computation with simple examples")
            print("   → Look for text encoding issues in answer comparison")
            print()
        
        if not result.tier_results["raw_capture"].success:
            print("⚠️ Generation pipeline issues detected")
            print("   → Verify model is properly initialized")
            print("   → Check if streaming method is returning text correctly")
            print("   → Validate context truncation and tokenization")
            print("   → Monitor for timeout or connection issues")
            print()
        
        if result.tier_results["id_space"].details.get("applicable", False):
            if not result.tier_results["id_space"].success:
                print("⚠️ ID namespace issues detected")
                print("   → Check retrieval index document ID format")
                print("   → Verify predicted IDs match gold ID schema")
                print("   → Test retrieval method with known documents")
                print()
        
        # Summary recommendation
        if all(r.success for r in result.tier_results.values()):
            print("✅ All diagnostic tiers passed!")
            print("   The evaluation infrastructure appears healthy.")
            print("   The accuracy=0.000 issue may be in a different component.")
            print("   Consider checking:")
            print("   → Data loading and sample preparation")
            print("   → Answer field extraction from samples") 
            print("   → Metric aggregation and reporting")
        else:
            print("❌ Critical issues found - fix in priority order:")
            for i, priority in enumerate(result.repair_priority, 1):
                print(f"   {i}. {priority}")
        
        print("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point for diagnostic script."""
    parser = argparse.ArgumentParser(description="Run evaluation pipeline diagnostics")
    parser.add_argument("--task", default="code_debug", 
                       help="Task to diagnose (default: code_debug)")
    parser.add_argument("--samples", type=int, default=50,
                       help="Number of samples to test (default: 50)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info(f"🧪 Evaluation Pipeline Diagnostics")
    logger.info(f"Task: {args.task}, Samples: {args.samples}")
    
    # Run diagnostics
    success = run_pipeline_diagnostics(args.task, args.samples)
    
    if success:
        logger.info("✅ Diagnostic completed successfully")
        print("\n📁 Detailed results saved to: diagnostic_results/")
        sys.exit(0)
    else:
        logger.error("❌ Diagnostic failed")
        sys.exit(1)

if __name__ == "__main__":
    main()