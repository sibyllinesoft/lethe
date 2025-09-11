#!/usr/bin/env python3
"""
Production Guards Test Script
============================

Standalone test script for the production guards system that doesn't depend
on complex module imports. Tests the core guard functionality with mock data.

Usage:
    python scripts/test_production_guards.py
"""

import json
import logging
import numpy as np
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_datasets() -> Dict[str, List[Dict]]:
    """Create test datasets with some intentional duplicates"""
    
    datasets = {}
    
    # Training set
    train_samples = []
    for i in range(100):
        sample = {
            'id': f"train_{i:03d}",
            'text': f"Training sample {i} with unique content about topic {i % 10}",
            'question': f"What is the main topic of sample {i}?",
            'answer': f"Topic {i % 10}",
            'domain': 'train'
        }
        train_samples.append(sample)
    
    # Add some exact duplicates
    train_samples.append(train_samples[0])  # Exact duplicate
    train_samples.append(train_samples[5])  # Another exact duplicate
    
    datasets['train'] = train_samples
    
    # Dev set with some near-duplicates
    dev_samples = []
    for i in range(50):
        # Create some near-duplicates by slightly modifying training samples
        if i < 5:
            base_sample = train_samples[i]
            sample = {
                'id': f"dev_{i:03d}",
                'text': base_sample['text'].replace('Training', 'Development'),  # Near duplicate
                'question': base_sample['question'],
                'answer': base_sample['answer'],
                'domain': 'dev'
            }
        else:
            sample = {
                'id': f"dev_{i:03d}",
                'text': f"Development sample {i} with different content about topic {i % 5}",
                'question': f"What is discussed in dev sample {i}?",
                'answer': f"Development topic {i % 5}",
                'domain': 'dev'
            }
        dev_samples.append(sample)
    
    datasets['dev'] = dev_samples
    
    # Test set
    test_samples = []
    for i in range(30):
        sample = {
            'id': f"test_{i:03d}",
            'text': f"Test sample {i} content for evaluation purposes",
            'question': f"What is test sample {i} about?",
            'answer': f"Test content {i}",
            'domain': 'test'
        }
        test_samples.append(sample)
    
    datasets['test'] = test_samples
    
    return datasets

def create_test_rag_pool() -> List[Dict]:
    """Create test RAG pool with some overlaps"""
    
    pool = []
    
    for i in range(200):
        doc = {
            'id': f"rag_doc_{i:04d}",
            'content': f"RAG document {i} containing information about subject {i % 20}",
            'title': f"Document {i}",
            'type': ['passage', 'document'][i % 2],
            'metadata': {
                'length': 100 + (i % 50),
                'source': 'rag_corpus'
            }
        }
        pool.append(doc)
    
    return pool

def create_mock_evaluation_results() -> Dict[str, Any]:
    """Create mock evaluation results for testing"""
    
    results = {}
    
    # Mock results by method and budget
    methods = ['bm25', 'dense_retrieval', 'lethe_hybrid', 'placebo_random']
    budgets = ['8%', '15%', '30%']
    
    for method in methods:
        results[method] = {}
        for budget in budgets:
            # Generate mock scores with some realistic patterns
            base_score = {
                'bm25': 0.3,
                'dense_retrieval': 0.5,
                'lethe_hybrid': 0.7,
                'placebo_random': 0.1
            }[method]
            
            # Higher budget = slightly better performance
            budget_multiplier = 0.8 + (float(budget.rstrip('%')) / 100) * 0.4
            
            scores = []
            for _ in range(50):  # 50 samples per condition
                score = base_score * budget_multiplier + random.gauss(0, 0.05)
                score = max(0, min(1, score))  # Clamp to [0, 1]
                scores.append(score)
            
            results[method][budget] = scores
    
    return results

def test_production_guards():
    """Test the production guards system"""
    
    logger.info("🧪 Testing Production Guards System")
    
    try:
        # Import the production guards module
        from src.eval.production_guards import run_production_guards
        
        # Create test data
        logger.info("Creating test datasets...")
        datasets = create_test_datasets()
        rag_pool = create_test_rag_pool()
        evaluation_results = create_mock_evaluation_results()
        
        # Mock hashes
        pool_hash = "test_pool_hash_12345"
        tokenizer_hash = "test_tokenizer_hash_67890"
        
        # Run production guards
        logger.info("Running production guards...")
        guard_report = run_production_guards(
            datasets=datasets,
            rag_pool=rag_pool,
            evaluation_results=evaluation_results,
            pool_hash=pool_hash,
            tokenizer_hash=tokenizer_hash
        )
        
        # Print results
        print("\n" + "="*60)
        print("🛡️ PRODUCTION GUARDS TEST RESULTS")
        print("="*60)
        
        print(f"\nOverall Status: {guard_report.get('overall_status', 'UNKNOWN')}")
        
        print(f"\n📊 Leakage Analysis:")
        leakage_analysis = guard_report.get('leakage_analysis', {})
        print(f"   Datasets analyzed: {leakage_analysis.get('train_samples', 0)} train, {leakage_analysis.get('dev_samples', 0)} dev, {leakage_analysis.get('test_samples', 0)} test")
        print(f"   RAG pool size: {leakage_analysis.get('rag_pool_samples', 0)}")
        print(f"   Exact duplicates found: {len(leakage_analysis.get('exact_duplicates', {}))}")
        print(f"   Near duplicates found: {len(leakage_analysis.get('near_duplicates', {}))}")
        
        print(f"\n🔒 Attestations:")
        attestations = guard_report.get('attestations', {})
        for key, value in attestations.items():
            status = "✅" if value else "❌"
            print(f"   {status} {key}: {value}")
        
        print(f"\n⚠️ Issues Found:")
        critical_failures = guard_report.get('critical_failures', [])
        if critical_failures:
            for failure in critical_failures:
                print(f"   ❌ {failure}")
        else:
            print("   ✅ No critical failures")
        
        warnings = guard_report.get('warnings', [])
        if warnings:
            for warning in warnings:
                print(f"   ⚠️ {warning}")
        else:
            print("   ✅ No warnings")
        
        print(f"\n🧪 Invariance Tests:")
        invariance_tests = guard_report.get('invariance_tests', [])
        for test in invariance_tests:
            status = "✅" if test['passed'] else "❌"
            print(f"   {status} {test['test_name']}: {'PASSED' if test['passed'] else 'FAILED'}")
        
        print(f"\n⚡ Power Analysis:")
        power_analysis = guard_report.get('power_analysis', [])
        if power_analysis:
            conclusive_count = sum(1 for analysis in power_analysis if analysis['conclusive'])
            total_count = len(power_analysis)
            print(f"   Conclusive conditions: {conclusive_count}/{total_count}")
            
            sample_size_recs = guard_report.get('sample_size_recommendations', {})
            if sample_size_recs.get('status') == 'expansion_needed':
                print(f"   ⚠️ Sample size expansion recommended for {sample_size_recs.get('inconclusive_conditions', 0)} conditions")
            else:
                print(f"   ✅ Current sample sizes are sufficient")
        
        print("\n" + "="*60)
        
        # Test passed if no critical failures
        if critical_failures:
            logger.error("❌ Production guards test FAILED")
            return False
        else:
            logger.info("✅ Production guards test PASSED")
            return True
            
    except ImportError as e:
        logger.error(f"Failed to import production guards: {e}")
        return False
    except Exception as e:
        logger.error(f"Production guards test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_jaccard_distribution():
    """Test Jaccard distribution binning"""
    
    logger.info("🎯 Testing Jaccard distribution analysis")
    
    # Generate test similarity scores
    np.random.seed(42)
    scores = np.concatenate([
        np.random.uniform(0.0, 0.1, 1000),  # Mostly dissimilar
        np.random.uniform(0.1, 0.3, 200),   # Somewhat similar
        np.random.uniform(0.8, 1.0, 50),    # Very similar (potential duplicates)
    ])
    
    # Test binning function
    try:
        from src.eval.production_guards import LeakageDetector
        
        detector = LeakageDetector()
        bins = detector._create_jaccard_bins(scores.tolist())
        
        print("\n📊 Jaccard Distribution Test:")
        total_scores = sum(bins.values())
        for bin_range, count in bins.items():
            percentage = (count / total_scores) * 100 if total_scores > 0 else 0
            print(f"   {bin_range}: {count:4d} ({percentage:5.1f}%)")
        
        # Validate that we have the expected high similarity cluster
        high_similarity = bins['0.8-0.9'] + bins['0.9-1.0']
        if high_similarity >= 45:  # Should have ~50 high similarity scores
            logger.info("✅ Jaccard distribution test PASSED")
            return True
        else:
            logger.warning(f"⚠️ Expected ~50 high similarity scores, got {high_similarity}")
            return False
            
    except Exception as e:
        logger.error(f"Jaccard distribution test failed: {e}")
        return False

def test_minhash_similarity():
    """Test MinHash similarity estimation"""
    
    logger.info("🔗 Testing MinHash similarity estimation")
    
    try:
        from src.eval.production_guards import MinHashDeduplicator
        
        deduplicator = MinHashDeduplicator(num_hashes=64, shingle_size=3)
        
        # Test with similar texts
        text1 = "This is a sample document for testing similarity detection"
        text2 = "This is a sample document for testing duplicate detection"  # Very similar
        text3 = "Completely different content about unrelated topics"  # Different
        
        sig1 = deduplicator.compute_minhash(text1)
        sig2 = deduplicator.compute_minhash(text2)
        sig3 = deduplicator.compute_minhash(text3)
        
        sim_1_2 = deduplicator.estimate_jaccard(sig1, sig2)
        sim_1_3 = deduplicator.estimate_jaccard(sig1, sig3)
        
        print(f"\n🔗 MinHash Similarity Test:")
        print(f"   Text 1 vs Text 2 (similar): {sim_1_2:.3f}")
        print(f"   Text 1 vs Text 3 (different): {sim_1_3:.3f}")
        
        # Validate that similar texts have higher similarity
        if sim_1_2 > sim_1_3 and sim_1_2 > 0.3:
            logger.info("✅ MinHash similarity test PASSED")
            return True
        else:
            logger.warning(f"⚠️ Expected sim_1_2 > sim_1_3 and sim_1_2 > 0.3, got {sim_1_2:.3f} vs {sim_1_3:.3f}")
            return False
            
    except Exception as e:
        logger.error(f"MinHash similarity test failed: {e}")
        return False

def main():
    """Run all production guard tests"""
    
    logger.info("🚀 Starting Production Guards Test Suite")
    
    tests = [
        ("MinHash Similarity", test_minhash_similarity),
        ("Jaccard Distribution", test_jaccard_distribution),
        ("Production Guards System", test_production_guards),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print final summary
    print(f"\n{'='*60}")
    print("🎯 PRODUCTION GUARDS TEST SUITE SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status:10} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All production guard tests PASSED!")
        sys.exit(0)
    else:
        logger.error(f"❌ {total - passed} tests FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()