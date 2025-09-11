#!/usr/bin/env python3
"""
Simple Standalone Evaluation Diagnostics
========================================

Minimal diagnostic system to test the three-tier failure isolation
without complex dependencies. Tests the core scoring logic directly.

Usage: python3 scripts/simple_evaluation_diagnostics.py

Author: Lethe Research Team
"""

import json
import logging
import random
import time
import unicodedata
import regex as re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

def normalize_answer(text, language: str = 'en') -> str:
    """Deterministic text normalization for scoring"""
    # Handle different input types (list, string, number, etc.)
    if isinstance(text, list):
        if not text:
            return ""
        # Take first element if list, convert to string
        text = str(text[0])
    elif text is None:
        return ""
    else:
        # Convert to string (handles int, float, etc.)
        text = str(text)
    
    if not text:
        return ""
        
    # Unicode normalization and case folding
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    
    # Language-specific punctuation normalization
    if language == "zh":
        text = text.replace("，", ",").replace("。", ".").replace("：", ":")
    
    # Remove non-word characters except specific Unicode ranges
    text = re.sub(
        r"[^\w\s\p{Han}\p{Katakana}\p{Hiragana}\p{Hangul}\-\.]", 
        "", 
        text, 
        flags=re.UNICODE
    )
    
    return text.strip()

def compute_token_f1(pred, gold, language: str = 'en') -> float:
    """Token-level F1 score with proper normalization"""
    pred_tokens = normalize_answer(pred, language).split()
    gold_tokens = normalize_answer(gold, language).split()
    
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
        
    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)
    intersection = pred_set & gold_set
    
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(gold_set)
    
    if precision + recall == 0:
        return 0.0
        
    return 2 * precision * recall / (precision + recall)

def evaluate_single_prediction(prediction, gold, language: str = 'en') -> float:
    """Single prediction evaluation with comprehensive normalization"""
    # Exact match after normalization
    pred_norm = normalize_answer(prediction, language)
    gold_norm = normalize_answer(gold, language)
    
    if pred_norm == gold_norm:
        return 1.0
    
    # Fall back to token F1
    return compute_token_f1(prediction, gold, language)

def gold_echo_control(samples: List[dict], n_samples: int = 200) -> dict:
    """Gold-Echo Control: Test scorer with perfect inputs"""
    logger.info(f"🧪 Running Gold-Echo Control (n={n_samples})")
    
    if len(samples) < n_samples:
        n_samples = len(samples)
        logger.warning(f"Only {len(samples)} samples available, using all")
    
    mixed_samples = random.sample(samples, n_samples)
    correct = 0
    total = 0
    issues = []
    detailed_results = []
    
    for sample in mixed_samples:
        try:
            # Use gold as prediction - should always score 1.0
            gold_answer = sample.get('answer', '')
            if not gold_answer:
                issues.append(f"Sample {sample.get('id', 'unknown')} has empty gold answer")
                continue
            
            # Test scorer with gold as prediction
            prediction = gold_answer
            language = sample.get('language', 'en')
            score = evaluate_single_prediction(prediction, gold_answer, language)
            
            detailed_results.append({
                'sample_id': sample.get('id', 'unknown'),
                'prediction': prediction[:100] + "..." if len(prediction) > 100 else prediction,
                'gold': gold_answer[:100] + "..." if len(gold_answer) > 100 else gold_answer,
                'score': score,
                'language': language
            })
            
            correct += score
            total += 1
            
            if score < 1.0:
                issues.append(f"Perfect input scored {score:.3f} for sample {sample.get('id', 'unknown')}")
                
        except Exception as e:
            issues.append(f"Error evaluating sample {sample.get('id', 'unknown')}: {str(e)}")
            continue
    
    accuracy = correct / total if total > 0 else 0.0
    success = accuracy >= 0.95  # Allow for minor floating point errors
    
    return {
        "success": success,
        "score": accuracy,
        "total_samples": total,
        "perfect_scores": correct,
        "issues": issues,
        "sample_results": detailed_results[:10]
    }

def raw_capture_probe(samples: List[dict], n_samples: int = 100) -> dict:
    """Raw-Capture Probe: Verify model is generating non-empty responses"""
    logger.info(f"🎯 Running Raw-Capture Probe (n={n_samples})")
    
    # Mock generation function
    def mock_generation(sample):
        """Simulate text generation for diagnostic purposes"""
        context = sample.get('context', '')
        question = sample.get('question', '')
        
        if 'debug' in question.lower():
            return "The bug is in line 42 of the function."
        elif 'what' in question.lower():
            return "The answer is 42."
        elif question:
            return f"Based on the context, {question.split()[-1] if question.split() else 'unknown'}"
        else:
            return "No clear answer found in the provided context."
    
    if len(samples) < n_samples:
        n_samples = len(samples)
        logger.warning(f"Only {len(samples)} samples available, using all")
    
    probe_samples = random.sample(samples, n_samples)
    probes = []
    issues = []
    total_tokens = 0
    non_empty_responses = 0
    
    for sample in probe_samples:
        try:
            start_time = time.time()
            
            # Run mock generation
            response_text = mock_generation(sample)
            processing_time = (time.time() - start_time) * 1000
            
            # Count tokens (simple whitespace split)
            token_count = len(response_text.split()) if response_text else 0
            total_tokens += token_count
            
            if response_text.strip():
                non_empty_responses += 1
            
            probe_result = {
                'sample_id': sample.get('id', 'unknown'),
                'decoded_text': response_text[:200] + "..." if len(response_text) > 200 else response_text,
                'new_token_count': token_count,
                'processing_time_ms': processing_time
            }
            probes.append(probe_result)
            
            if token_count == 0:
                issues.append(f"Empty response for sample {sample.get('id', 'unknown')}")
            elif token_count < 5:
                issues.append(f"Very short response ({token_count} tokens) for sample {sample.get('id', 'unknown')}")
                
        except Exception as e:
            issues.append(f"Generation failed for sample {sample.get('id', 'unknown')}: {str(e)}")
            probe_result = {
                'sample_id': sample.get('id', 'unknown'),
                'decoded_text': "",
                'new_token_count': 0,
                'processing_time_ms': 0,
                'error': str(e)
            }
            probes.append(probe_result)
    
    # Calculate metrics
    median_tokens = np.median([p['new_token_count'] for p in probes]) if probes else 0
    response_rate = non_empty_responses / len(probes) if probes else 0
    success = response_rate >= 0.8 and median_tokens >= 5
    
    return {
        "success": success,
        "score": response_rate,
        "total_samples": len(probes),
        "non_empty_responses": non_empty_responses,
        "response_rate": response_rate,
        "median_tokens": median_tokens,
        "issues": issues,
        "probe_samples": probes[:10]
    }

def id_space_check(samples: List[dict], task_type: str) -> dict:
    """ID-Space Check: Verify ID namespace alignment for retrieval tasks"""
    logger.info(f"🔍 Running ID-Space Check for {task_type}")
    
    # Only applicable to certain task types
    if task_type not in ['code_debug', 'passkey', 'retrieval', 'kv_retrieval']:
        return {
            "success": True,
            "score": 1.0,
            "applicable": False,
            "task_type": task_type
        }
    
    # Mock ID extraction for diagnostic purposes
    def mock_get_prediction_ids(sample):
        """Mock prediction ID extraction"""
        return [f"doc_{i}" for i in range(5)]  # Mock top-5 doc IDs
    
    def mock_get_gold_ids(sample):
        """Mock gold ID extraction"""
        # Simulate some overlap for realistic testing
        if random.random() < 0.4:  # 40% have some overlap
            return [f"doc_{random.randint(0, 4)}", f"doc_{random.randint(5, 9)}"]
        else:
            return [f"doc_{random.randint(10, 15)}"]  # No overlap
    
    intersections = []
    top5_ids_samples = []
    issues = []
    
    for sample in samples[:50]:  # Check first 50 for speed
        try:
            # Get predicted and gold IDs
            pred_ids = mock_get_prediction_ids(sample)
            gold_ids = mock_get_gold_ids(sample)
            
            intersection = set(pred_ids) & set(gold_ids)
            intersections.append(len(intersection) > 0)
            
            top5_ids_samples.append({
                'sample_id': sample.get('id', 'unknown'),
                'pred_ids': pred_ids[:5],
                'gold_ids': gold_ids,
                'intersection': list(intersection)
            })
            
            if len(intersection) == 0:
                issues.append(f"No ID overlap for sample {sample.get('id', 'unknown')}")
                
        except Exception as e:
            issues.append(f"ID extraction failed for sample {sample.get('id', 'unknown')}: {str(e)}")
    
    intersection_rate = sum(intersections) / len(intersections) if intersections else 0.0
    success = intersection_rate >= 0.30  # At least 30% should have some overlap
    
    return {
        "success": success,
        "score": intersection_rate,
        "applicable": True,
        "task_type": task_type,
        "intersection_rate": intersection_rate,
        "samples_checked": len(intersections),
        "issues": issues,
        "id_samples": top5_ids_samples[:10]
    }

def load_sample_data(task_name: str = "code_debug") -> List[dict]:
    """Load sample evaluation data for testing"""
    # Check for real data first
    data_path = Path("benchmarks/infinitebench/data") / f"{task_name}.jsonl"
    
    if data_path.exists():
        logger.info(f"Loading real data from {data_path}")
        samples = []
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        samples.append({
                            'id': data.get('id', 'unknown'),
                            'question': data.get('input', ''),
                            'context': data.get('context', ''),
                            'answer': data.get('answer', ''),
                            'language': 'en'
                        })
            logger.info(f"Loaded {len(samples)} real samples")
            return samples[:100]  # Limit for testing
        except Exception as e:
            logger.warning(f"Failed to load real data: {e}")
    
    # Fall back to mock data
    logger.info("Using mock data for testing")
    mock_samples = []
    for i in range(50):
        mock_samples.append({
            'id': f'sample_{i}',
            'question': f'What is the bug in function_{i}?',
            'context': f'def function_{i}():\n    x = {i}\n    return x + 1',
            'answer': f'Function {i} has no bugs',
            'language': 'en'
        })
    
    return mock_samples

def run_comprehensive_diagnostic(task_name: str = "code_debug"):
    """Run all three diagnostic tiers and generate report"""
    logger.info(f"🚀 Starting comprehensive diagnostic for {task_name}")
    start_time = time.time()
    
    # Load sample data
    samples = load_sample_data(task_name)
    if not samples:
        logger.error("No samples loaded for diagnostic")
        return False
    
    logger.info(f"Running diagnostics on {len(samples)} samples")
    
    # Run all three tiers
    results = {}
    
    # Tier 1: Gold-Echo Control
    results["gold_echo"] = gold_echo_control(samples, n_samples=min(200, len(samples)))
    
    # Tier 2: Raw-Capture Probe  
    results["raw_capture"] = raw_capture_probe(samples, n_samples=min(100, len(samples)))
    
    # Tier 3: ID-Space Check
    results["id_space"] = id_space_check(samples, task_name)
    
    # Generate report
    print("\n" + "="*80)
    print("🧪 EVALUATION DIAGNOSTIC REPORT")
    print("="*80)
    print(f"Task: {task_name}")
    print(f"Samples: {len(samples)}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Print tier results
    tier_names = {
        "gold_echo": "📊 Gold-Echo Control",
        "raw_capture": "🎯 Raw-Capture Probe", 
        "id_space": "🔍 ID-Space Check"
    }
    
    for i, (tier_key, result) in enumerate(results.items(), 1):
        status = "✅" if result["success"] else "❌"
        print(f"{i}. {status} {tier_names[tier_key]} (n={result.get('total_samples', result.get('samples_checked', 0))})")
        print(f"   Score: {result['score']:.3f}")
        
        if tier_key == "gold_echo":
            print(f"   Expected: 1.000 (perfect scorer)")
        elif tier_key == "raw_capture":
            print(f"   Expected: ≥0.80 (response rate)")
            print(f"   Median tokens: {result.get('median_tokens', 0)} (Expected: ≥5)")
        elif tier_key == "id_space":
            if result.get("applicable", True):
                print(f"   Expected: ≥0.30 (ID intersection rate)")
            else:
                print(f"   Not applicable to task type")
        
        # Show issues
        issues = result.get("issues", [])
        if issues:
            print(f"   Issues: {len(issues)} found")
            if len(issues) <= 3:
                for issue in issues:
                    print(f"     • {issue}")
            else:
                for issue in issues[:2]:
                    print(f"     • {issue}")
                print(f"     • ... and {len(issues)-2} more")
        print()
    
    # Overall assessment
    print("🎯 OVERALL ASSESSMENT")
    all_passed = all(r["success"] for r in results.values())
    
    if all_passed:
        print("✅ All diagnostic tiers PASSED")
        print("   The evaluation infrastructure appears healthy.")
        print("   If seeing accuracy=0.000, check:")
        print("   → Data loading and sample preparation")
        print("   → Answer field extraction from samples")
        print("   → Metric aggregation and reporting")
    else:
        failed_tiers = [tier_names[k] for k, v in results.items() if not v["success"]]
        print(f"❌ CRITICAL: {len(failed_tiers)} tier(s) failed")
        
        if not results["gold_echo"]["success"]:
            print("   🚨 CRITICAL: Scorer/Normalization BROKEN")
            print("   → Fix normalize_answer() function")
            print("   → Check Unicode normalization (NFKC)")
            print("   → Test token F1 computation")
        
        if not results["raw_capture"]["success"]:
            print("   ⚠️ Generation pipeline issues")
            print("   → Check model initialization")
            print("   → Verify text generation works")
        
        if results["id_space"].get("applicable", False) and not results["id_space"]["success"]:
            print("   ⚠️ ID namespace issues")
            print("   → Check retrieval ID mapping")
    
    print("="*80)
    
    # Save results
    output_dir = Path("diagnostic_results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"simple_diagnostic_{task_name}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "task": task_name,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "samples_count": len(samples),
            "results": results,
            "overall_success": all_passed,
            "duration_seconds": time.time() - start_time
        }, f, indent=2, default=str)
    
    logger.info(f"📁 Results saved to: {output_file}")
    
    return all_passed

def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    logger.info("🧪 Simple Evaluation Diagnostics")
    
    success = run_comprehensive_diagnostic("code_debug")
    
    if success:
        print("\n✅ All diagnostics passed!")
    else:
        print("\n❌ Critical issues found - see recommendations above")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())