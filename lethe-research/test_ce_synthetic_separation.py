"""
Synthetic separation test for CE fix validation.

Tests that CE can properly separate:
1. Query + identical content → HIGH logit
2. Query + disjoint content → LOW logit  
3. Ordering is correct and std > 0.2
"""

import logging
import sys
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.rerank.cross_encoder import CrossEncoderReranker
from src.rerank.content_renderer import ContentRenderer, CEGuards

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_synthetic_separation():
    """Test CE synthetic separation with fixed content rendering."""
    
    print("🧪 SYNTHETIC SEPARATION TEST")
    print("=" * 50)
    
    # Initialize components
    ce = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    renderer = ContentRenderer()
    guards = CEGuards()
    
    # Test query
    query = "machine learning algorithms"
    
    # Test pairs with expected ordering: identical > partial > disjoint
    test_pairs = [
        {
            "type": "identical",
            "query": query,
            "doc": "machine learning algorithms",
            "expected": "HIGH"
        },
        {
            "type": "partial", 
            "query": query,
            "doc": "machine learning algorithms are computational methods for pattern recognition",
            "expected": "HIGH"
        },
        {
            "type": "disjoint",
            "query": query,
            "doc": "xyz uvw abc def random unrelated text",
            "expected": "LOW"
        }
    ]
    
    # Score all pairs
    print(f"📝 Query: '{query}'")
    print(f"📊 Testing {len(test_pairs)} synthetic pairs...")
    
    results = []
    doc_ids = []
    documents = {}
    
    for i, pair in enumerate(test_pairs):
        doc_id = f"doc_{i}"
        doc_ids.append(doc_id)
        documents[doc_id] = pair["doc"]
        
        print(f"  {pair['type']:>10}: '{pair['doc'][:50]}...' (expect {pair['expected']})")
    
    # Call CE with documents (fixed version)
    scores = ce.score_pairs(
        query=query,
        doc_ids=doc_ids,
        documents=documents  # CRITICAL: passing actual content
    )
    
    # Extract logits/scores
    logits = [scores[doc_id] for doc_id in doc_ids]
    score_types = [pair["type"] for pair in test_pairs]
    
    # Analyze results
    print(f"\n📊 RESULTS:")
    for i, (doc_id, score, pair_type) in enumerate(zip(doc_ids, logits, score_types)):
        print(f"  {pair_type:>10}: logit = {score:>8.3f}")
    
    # Compute statistics
    std = np.std(logits)
    score_range = max(logits) - min(logits)
    
    print(f"\n📈 STATISTICS:")
    print(f"  Mean:   {np.mean(logits):>8.3f}")
    print(f"  Std:    {std:>8.3f}")
    print(f"  Range:  {score_range:>8.3f}")
    print(f"  Min:    {min(logits):>8.3f}")
    print(f"  Max:    {max(logits):>8.3f}")
    
    # Validate with guards
    variance_check = guards.validate_score_variance(logits)
    
    print(f"\n🛡️ GUARD VALIDATION:")
    print(f"  Std >= {guards.min_std}: {'✅' if variance_check['std'] >= guards.min_std else '❌'} ({variance_check['std']:.3f})")
    print(f"  Range >= {guards.min_range}: {'✅' if variance_check['range'] >= guards.min_range else '❌'} ({variance_check['range']:.3f})")
    print(f"  Overall valid: {'✅' if variance_check['valid'] else '❌'}")
    
    if not variance_check['valid']:
        print(f"  Issues: {variance_check['issues']}")
    
    # Check ordering (identical should be highest, disjoint should be lowest)
    identical_idx = next(i for i, p in enumerate(test_pairs) if p["type"] == "identical")
    disjoint_idx = next(i for i, p in enumerate(test_pairs) if p["type"] == "disjoint")
    
    identical_score = logits[identical_idx]
    disjoint_score = logits[disjoint_idx]
    
    ordering_correct = identical_score > disjoint_score
    
    print(f"\n🎯 ORDERING CHECK:")
    print(f"  Identical > Disjoint: {'✅' if ordering_correct else '❌'}")
    print(f"    Identical: {identical_score:.3f}")
    print(f"    Disjoint:  {disjoint_score:.3f}")
    print(f"    Margin:    {identical_score - disjoint_score:.3f}")
    
    # Overall test result
    test_passed = variance_check['valid'] and ordering_correct
    
    print(f"\n🏆 SYNTHETIC SEPARATION TEST: {'✅ PASSED' if test_passed else '❌ FAILED'}")
    
    if test_passed:
        print("✅ CE can properly differentiate between relevant and irrelevant content")
        print("✅ Score variance indicates healthy model operation")
        print("✅ Ready for real-pair testing")
    else:
        print("❌ CE synthetic separation failed")
        if not variance_check['valid']:
            print("   - Score variance too low (flat scoring)")
        if not ordering_correct:
            print("   - Ordering incorrect (relevance signal broken)")
    
    return test_passed, {
        "std": std,
        "range": score_range,
        "logits": logits,
        "ordering_correct": ordering_correct,
        "variance_valid": variance_check['valid']
    }


if __name__ == "__main__":
    test_passed, results = test_synthetic_separation()
    sys.exit(0 if test_passed else 1)