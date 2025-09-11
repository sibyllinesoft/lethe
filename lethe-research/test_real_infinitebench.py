#!/usr/bin/env python3
"""
Test the improved LLM generation on real InfiniteBench samples
"""
import sys
import logging
from pathlib import Path
import json

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import dependencies
from scripts.run_hybrid_infinitebench import generate_llm_response
from src.infinitebench.dataset_loader import InfiniteBenchLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_real_samples():
    """Test with actual InfiniteBench samples"""
    
    # Load real data
    infinitebench_path = project_root / "benchmarks" / "infinitebench" / "data"
    loader = InfiniteBenchLoader(infinitebench_path)
    
    results = []
    
    # Test code_debug samples
    logger.info("Testing code_debug samples...")
    code_debug_samples = loader.load_task('code_debug')
    if code_debug_samples:
        for i, sample in enumerate(code_debug_samples[:3]):  # Test first 3
            query = sample.question
            context = sample.context
            expected = sample.answer
            
            logger.info(f"\n🔍 CODE DEBUG SAMPLE {i+1}:")
            logger.info(f"Query: {query[:100]}...")
            logger.info(f"Expected: {expected}")
            logger.info(f"Context length: {len(context)} chars")
            
            response = generate_llm_response(query, context)
            logger.info(f"LLM Response: '{response}'")
            
            # Apply same normalization as in the main script
            expected_items = []
            if isinstance(expected, list):
                expected_items = [str(item).lower().strip() for item in expected if item]
            else:
                expected_str = str(expected) if expected is not None else ""
                if expected_str.strip():
                    if expected_str.startswith('[') and expected_str.endswith(']'):
                        try:
                            import ast
                            parsed_list = ast.literal_eval(expected_str)
                            if isinstance(parsed_list, list):
                                expected_items = [str(item).lower().strip() for item in parsed_list if item]
                            else:
                                expected_items = [expected_str.lower().strip()]
                        except (ValueError, SyntaxError):
                            expected_items = [expected_str.lower().strip()]
                    else:
                        expected_items = [expected_str.lower().strip()]
            
            actual_normalized = response.lower().strip() if response else ""
            actual_normalized = actual_normalized.replace('"', '').replace("'", '').replace("`", "")
            actual_normalized = actual_normalized.replace("function ", "").replace("def ", "")
            actual_normalized = actual_normalized.split("(")[0]
            actual_normalized = actual_normalized.split(":")[0]
            actual_normalized = actual_normalized.split()[0] if actual_normalized.split() else actual_normalized
            
            accuracy = 0.0
            if expected_items and actual_normalized:
                for expected_item in expected_items:
                    if expected_item == actual_normalized:
                        accuracy = 1.0
                        break
                
                if accuracy == 0.0:
                    for expected_item in expected_items:
                        if expected_item and len(expected_item) > 2:
                            if expected_item in actual_normalized or actual_normalized in expected_item:
                                accuracy = 0.8
                                break
            
            logger.info(f"Expected items: {expected_items}")
            logger.info(f"Normalized response: '{actual_normalized}'")
            logger.info(f"Accuracy: {accuracy}")
            
            results.append({
                'task': 'code_debug',
                'sample': i+1,
                'accuracy': accuracy,
                'expected': expected,
                'response': response
            })
    
    # Test code_run samples
    logger.info("\n" + "="*50)
    logger.info("Testing code_run samples...")
    code_run_samples = loader.load_task('code_run')
    if code_run_samples:
        for i, sample in enumerate(code_run_samples[:2]):  # Test first 2
            query = sample.question
            context = sample.context
            expected = sample.answer
            
            logger.info(f"\n🔍 CODE RUN SAMPLE {i+1}:")
            logger.info(f"Query: {query[:100]}...")
            logger.info(f"Expected: {expected}")
            logger.info(f"Context length: {len(context)} chars")
            
            response = generate_llm_response(query, context)
            logger.info(f"LLM Response: '{response}'")
            
            # Simple substring check for code run
            accuracy = 0.0
            if expected and response:
                expected_str = str(expected).lower().strip()
                response_str = response.lower().strip()
                if expected_str in response_str or response_str in expected_str:
                    accuracy = 1.0
                    
            logger.info(f"Accuracy: {accuracy}")
            
            results.append({
                'task': 'code_run',
                'sample': i+1,
                'accuracy': accuracy,
                'expected': expected,
                'response': response
            })
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📊 SUMMARY:")
    
    task_accuracies = {}
    for result in results:
        task = result['task']
        if task not in task_accuracies:
            task_accuracies[task] = []
        task_accuracies[task].append(result['accuracy'])
    
    overall_accuracy = 0.0
    total_samples = 0
    
    for task, accuracies in task_accuracies.items():
        avg_accuracy = sum(accuracies) / len(accuracies)
        logger.info(f"{task}: {avg_accuracy:.1%} accuracy ({len(accuracies)} samples)")
        overall_accuracy += sum(accuracies)
        total_samples += len(accuracies)
    
    if total_samples > 0:
        overall_accuracy /= total_samples
        logger.info(f"Overall: {overall_accuracy:.1%} accuracy ({total_samples} samples)")
        
        if overall_accuracy > 0:
            logger.info("🎉 SUCCESS: Non-zero accuracy achieved!")
            logger.info("   The improved LLM generation is working on real InfiniteBench data.")
        else:
            logger.warning("❌ Still getting zero accuracy - may need further improvements")
    
    return results

if __name__ == "__main__":
    logger.info("🧪 Testing improved LLM generation on real InfiniteBench data...")
    
    try:
        results = test_real_samples()
        
        # Save results
        with open("test_infinitebench_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info("Results saved to test_infinitebench_results.json")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()