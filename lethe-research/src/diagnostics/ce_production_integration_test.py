"""
CE Production Integration Test

Tests the exact issue found: cross-encoder receiving doc_ids without documents,
causing it to score against generic "Document {doc_id}" text instead of real content.
"""

import logging
import time
from typing import Dict, List, Optional
import numpy as np

from ..rerank.cross_encoder import CrossEncoderReranker

logger = logging.getLogger(__name__)


class CEProductionIntegrationTest:
    """Test CE production integration issue."""
    
    def __init__(self):
        self.cross_encoder = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    
    def test_production_vs_standalone(self, query: str = "machine learning algorithms") -> Dict:
        """Test CE with and without documents parameter."""
        
        # Test data
        doc_ids = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]
        
        # Real document content (what should be used)
        real_documents = {
            "doc_1": "Machine learning algorithms are computational methods that learn patterns from data",
            "doc_2": "Deep neural networks use multiple layers to extract hierarchical features", 
            "doc_3": "Random forests combine multiple decision trees for robust predictions",
            "doc_4": "Support vector machines find optimal hyperplanes for classification",
            "doc_5": "Clustering algorithms group similar data points without supervision"
        }
        
        # Test 1: Production bug - no documents parameter (uses fallback)
        print(f"🔍 Testing PRODUCTION BUG: score_pairs without documents")
        start_time = time.time()
        
        production_scores = self.cross_encoder.score_pairs(
            query=query,
            doc_ids=doc_ids,
            # documents=real_documents,  # MISSING! This is the bug
            batch_size=32,
            max_length=512
        )
        
        production_time = time.time() - start_time
        
        # Test 2: Correct usage - with documents parameter
        print(f"🔍 Testing CORRECT USAGE: score_pairs with documents")
        start_time = time.time()
        
        correct_scores = self.cross_encoder.score_pairs(
            query=query,
            doc_ids=doc_ids,
            documents=real_documents,  # PRESENT! This is correct
            batch_size=32,
            max_length=512
        )
        
        correct_time = time.time() - start_time
        
        # Analyze the difference
        production_values = list(production_scores.values())
        correct_values = list(correct_scores.values())
        
        production_std = np.std(production_values)
        correct_std = np.std(correct_values)
        
        production_range = max(production_values) - min(production_values)
        correct_range = max(correct_values) - min(correct_values)
        
        results = {
            "query": query,
            "production_bug": {
                "scores": production_scores,
                "std": production_std,
                "range": production_range,
                "mean": np.mean(production_values),
                "time_ms": production_time * 1000,
                "uses_fallback": True,
                "content_type": "Generic 'Document {doc_id}' text"
            },
            "correct_usage": {
                "scores": correct_scores,
                "std": correct_std,
                "range": correct_range, 
                "mean": np.mean(correct_values),
                "time_ms": correct_time * 1000,
                "uses_fallback": False,
                "content_type": "Real document content"
            },
            "comparison": {
                "std_ratio": correct_std / production_std if production_std > 0 else float('inf'),
                "range_ratio": correct_range / production_range if production_range > 0 else float('inf'),
                "score_correlation": np.corrcoef(production_values, correct_values)[0, 1] if len(production_values) > 1 else 0.0,
                "explains_flat_scores": production_std < 0.1,  # Flat if std < 0.1
                "issue_severity": "CRITICAL" if production_std < 0.1 else "MODERATE"
            }
        }
        
        return results
    
    def demonstrate_fix(self) -> str:
        """Demonstrate the fix needed in production code."""
        
        fix_code = '''
# CURRENT BUGGY CODE in src/rerank/core.py line 200-205:
rerank_scores = self.cross_encoder.score_pairs(
    query=query,
    doc_ids=candidate_docs,
    batch_size=config.batch_size,
    max_length=config.max_length
    # documents=??? <-- MISSING!
)

# FIXED CODE:
# Need to extract documents from fusion_result.candidates or similar
documents = {}
for doc_id in candidate_docs:
    # Extract actual document content
    documents[doc_id] = get_document_content(doc_id)  # implement this

rerank_scores = self.cross_encoder.score_pairs(
    query=query,
    doc_ids=candidate_docs,
    documents=documents,  # <-- ADDED!
    batch_size=config.batch_size,
    max_length=config.max_length
)
'''
        return fix_code
    
    def run_test(self) -> Dict:
        """Run complete integration test."""
        
        print("=" * 60)
        print("🔬 CE PRODUCTION INTEGRATION TEST")
        print("=" * 60)
        
        # Test multiple queries
        test_queries = [
            "machine learning algorithms",
            "neural network architecture", 
            "database optimization techniques",
            "distributed systems design",
            "quantum computing applications"
        ]
        
        all_results = []
        
        for i, query in enumerate(test_queries):
            print(f"\n📝 Test {i+1}/5: '{query}'")
            result = self.test_production_vs_standalone(query)
            all_results.append(result)
            
            # Print key metrics
            prod_std = result["production_bug"]["std"]
            correct_std = result["correct_usage"]["std"]
            
            print(f"  Production bug std: {prod_std:.6f}")
            print(f"  Correct usage std:  {correct_std:.6f}")
            print(f"  Improvement ratio:  {correct_std/prod_std:.1f}x" if prod_std > 0 else "  Improvement: ∞")
            
            if prod_std < 0.1:
                print(f"  ❌ CONFIRMS FLAT SCORING BUG!")
            else:
                print(f"  ✅ No flat scoring detected")
        
        # Summary
        avg_prod_std = np.mean([r["production_bug"]["std"] for r in all_results])
        avg_correct_std = np.mean([r["correct_usage"]["std"] for r in all_results])
        
        summary = {
            "test_results": all_results,
            "summary": {
                "avg_production_std": avg_prod_std,
                "avg_correct_std": avg_correct_std,
                "avg_improvement_ratio": avg_correct_std / avg_prod_std if avg_prod_std > 0 else float('inf'),
                "confirms_bug": avg_prod_std < 0.1,
                "tests_run": len(test_queries)
            },
            "recommended_fix": self.demonstrate_fix()
        }
        
        print("\n" + "=" * 60)
        print("📊 SUMMARY")
        print("=" * 60)
        print(f"Average production std: {avg_prod_std:.6f}")
        print(f"Average correct std:    {avg_correct_std:.6f}")
        print(f"Average improvement:    {avg_correct_std/avg_prod_std:.1f}x" if avg_prod_std > 0 else "∞")
        
        if avg_prod_std < 0.1:
            print("🎯 CONFIRMED: Missing documents parameter causes flat scoring!")
            print("🔧 FIX: Add documents parameter to score_pairs call in rerank/core.py")
        else:
            print("✅ No systematic flat scoring detected")
        
        return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    tester = CEProductionIntegrationTest()
    results = tester.run_test()
    
    # Save results
    import json
    with open("ce_production_integration_test_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to ce_production_integration_test_results.json")