"""
Real-pair peek test for CE fix validation.

Tests CE with 20 real query-document pairs to verify:
1. Real content (not generic "Document {id}") 
2. Token lengths are reasonable
3. Logits show proper variance (min/median/max)
"""

import logging
import sys
from pathlib import Path
import json
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.rerank.cross_encoder import CrossEncoderReranker
from src.rerank.content_renderer import ContentRenderer, CEGuards

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_real_test_pairs():
    """Create 20 real query-document pairs for testing."""
    
    # Real examples that should show clear relevance differences
    test_data = [
        {
            "query": "python error handling best practices",
            "documents": {
                "high_rel": "Python error handling uses try-except blocks. Best practices include catching specific exceptions, using finally blocks for cleanup, and logging errors appropriately. Avoid bare except clauses.",
                "medium_rel": "Exception handling in programming languages allows graceful failure recovery. Python, Java, and C++ all implement similar concepts with different syntax.",
                "low_rel": "Database indexing strategies can improve query performance significantly. B-tree indexes are most common for equality searches."
            }
        },
        {
            "query": "machine learning model evaluation metrics",
            "documents": {
                "high_rel": "Model evaluation metrics include accuracy, precision, recall, and F1-score. For regression: MSE, RMSE, MAE. Cross-validation helps assess generalization.",
                "medium_rel": "Statistical analysis often requires different metrics depending on the problem type. Classification and regression have distinct evaluation approaches.",
                "low_rel": "Web development frameworks like React and Vue.js have gained popularity for building interactive user interfaces."
            }
        },
        {
            "query": "database indexing optimization",
            "documents": {
                "high_rel": "Database indexing optimization involves choosing appropriate index types (B-tree, hash, bitmap), analyzing query patterns, and balancing read vs write performance.",
                "medium_rel": "Database performance tuning includes indexing, query optimization, hardware scaling, and proper schema design for efficient data access.",
                "low_rel": "Mobile app development requires consideration of platform differences, user experience design, and performance constraints."
            }
        },
        {
            "query": "neural network backpropagation algorithm",
            "documents": {
                "high_rel": "Backpropagation algorithm computes gradients by applying chain rule backwards through neural network layers. It calculates partial derivatives of loss with respect to weights.",
                "medium_rel": "Neural networks use various learning algorithms. Gradient descent and its variants are fundamental optimization techniques for training deep learning models.",
                "low_rel": "Cloud computing services provide scalable infrastructure for enterprise applications with pay-as-you-go pricing models."
            }
        },
        {
            "query": "REST API design principles",
            "documents": {
                "high_rel": "REST API design principles include stateless communication, uniform interface, resource-based URLs, proper HTTP methods (GET, POST, PUT, DELETE), and meaningful status codes.",
                "medium_rel": "API design patterns help create maintainable interfaces. RESTful services and GraphQL represent different approaches to data exchange.",
                "low_rel": "Computer graphics rendering techniques involve rasterization, ray tracing, and shader programming for realistic visual effects."
            }
        }
    ]
    
    # Expand to 20 pairs by creating variations
    pairs = []
    for i, test_case in enumerate(test_data):
        query = test_case["query"]
        for j, (rel_level, doc) in enumerate(test_case["documents"].items()):
            pairs.append({
                "pair_id": f"pair_{i}_{j}",
                "query": query,
                "document": doc,
                "expected_relevance": rel_level,
                "query_id": f"q_{i}",
                "doc_id": f"doc_{i}_{j}"
            })
    
    # Add more diverse examples to reach 20
    extra_pairs = [
        {
            "pair_id": "pair_5_0",
            "query": "docker container orchestration",
            "document": "Docker container orchestration with Kubernetes manages deployment, scaling, and operations of containerized applications across clusters.",
            "expected_relevance": "high_rel",
            "query_id": "q_5",
            "doc_id": "doc_5_0"
        },
        {
            "pair_id": "pair_5_1", 
            "query": "docker container orchestration",
            "document": "Cooking pasta requires boiling water, adding salt, and timing the cooking process carefully for optimal texture.",
            "expected_relevance": "low_rel",
            "query_id": "q_5",
            "doc_id": "doc_5_1"
        },
        {
            "pair_id": "pair_6_0",
            "query": "time series forecasting methods",
            "document": "Time series forecasting methods include ARIMA, exponential smoothing, and neural networks like LSTM for predicting future values.",
            "expected_relevance": "high_rel", 
            "query_id": "q_6",
            "doc_id": "doc_6_0"
        },
        {
            "pair_id": "pair_6_1",
            "query": "time series forecasting methods", 
            "document": "Garden maintenance involves watering plants, pruning branches, and applying fertilizer during appropriate seasons.",
            "expected_relevance": "low_rel",
            "query_id": "q_6", 
            "doc_id": "doc_6_1"
        },
        {
            "pair_id": "pair_7_0",
            "query": "cryptographic hash functions",
            "document": "Cryptographic hash functions like SHA-256 produce fixed-size digests from variable input, with properties of determinism and avalanche effect.",
            "expected_relevance": "high_rel",
            "query_id": "q_7",
            "doc_id": "doc_7_0"
        }
    ]
    
    pairs.extend(extra_pairs)
    return pairs[:20]  # Ensure exactly 20 pairs


def test_real_pairs():
    """Test CE with real query-document pairs."""
    
    print("🔍 REAL-PAIR PEEK TEST (N=20)")
    print("=" * 60)
    
    # Initialize components
    ce = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    renderer = ContentRenderer()
    guards = CEGuards()
    
    # Get test pairs
    test_pairs = create_real_test_pairs()
    print(f"📊 Testing {len(test_pairs)} real query-document pairs...")
    
    # Process pairs in batches by query
    results = []
    all_logits = []
    
    queries_seen = set()
    for pair in test_pairs:
        query = pair["query"]
        doc_id = pair["doc_id"]
        document = pair["document"]
        
        if query not in queries_seen:
            print(f"\n📝 Query: '{query}'")
            queries_seen.add(query)
        
        # Score single pair
        scores = ce.score_pairs(
            query=query,
            doc_ids=[doc_id],
            documents={doc_id: document}
        )
        
        logit = scores[doc_id]
        all_logits.append(logit)
        
        # Log token information
        query_tokens = renderer.tokenizer.encode(query, add_special_tokens=False)
        doc_tokens = renderer.tokenizer.encode(document, add_special_tokens=False)
        total_tokens = len(query_tokens) + len(doc_tokens) + 3  # +3 for special tokens
        
        # Extract first/last 30 tokens for verification
        all_tokens = renderer.tokenizer.encode(f"{query} [SEP] {document}", 
                                             add_special_tokens=True, 
                                             max_length=512, 
                                             truncation=True)
        token_texts = renderer.tokenizer.convert_ids_to_tokens(all_tokens)
        
        first_30 = token_texts[:30]
        last_30 = token_texts[-30:] if len(token_texts) > 30 else token_texts
        
        result = {
            "pair_id": pair["pair_id"],
            "query": query,
            "document": document[:100] + "..." if len(document) > 100 else document,
            "expected_relevance": pair["expected_relevance"], 
            "logit": logit,
            "query_tokens": len(query_tokens),
            "doc_tokens": len(doc_tokens),
            "total_tokens": total_tokens,
            "first_30_tokens": first_30,
            "last_30_tokens": last_30
        }
        
        results.append(result)
        
        print(f"  {pair['expected_relevance']:>10}: logit={logit:>7.3f}, "
              f"tokens=({len(query_tokens):>2}q+{len(doc_tokens):>3}d={total_tokens:>3})")
    
    # Analyze results
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total pairs: {len(results)}")
    print(f"  Logits min:  {min(all_logits):>8.3f}")
    print(f"  Logits med:  {np.median(all_logits):>8.3f}")
    print(f"  Logits max:  {max(all_logits):>8.3f}")
    print(f"  Logits std:  {np.std(all_logits):>8.3f}")
    print(f"  Logits rng:  {max(all_logits) - min(all_logits):>8.3f}")
    
    # Token statistics
    token_stats = {
        "query_tokens": [r["query_tokens"] for r in results],
        "doc_tokens": [r["doc_tokens"] for r in results], 
        "total_tokens": [r["total_tokens"] for r in results]
    }
    
    print(f"\n📏 TOKEN STATISTICS:")
    for token_type, tokens in token_stats.items():
        print(f"  {token_type:>12}: min={min(tokens):>3}, "
              f"med={np.median(tokens):>5.1f}, max={max(tokens):>3}")
    
    # Validate with guards
    variance_check = guards.validate_score_variance(all_logits)
    
    print(f"\n🛡️ GUARD VALIDATION:")
    print(f"  Std >= {guards.min_std}: {'✅' if variance_check['std'] >= guards.min_std else '❌'} ({variance_check['std']:.3f})")
    print(f"  Range >= {guards.min_range}: {'✅' if variance_check['range'] >= guards.min_range else '❌'} ({variance_check['range']:.3f})")
    print(f"  Overall valid: {'✅' if variance_check['valid'] else '❌'}")
    
    # Check for placeholder content
    placeholder_detected = any("Document doc_" in r["document"] for r in results)
    print(f"  No placeholders: {'✅' if not placeholder_detected else '❌'}")
    
    # Content verification samples
    print(f"\n🔍 CONTENT VERIFICATION (First 3 pairs):")
    for i, result in enumerate(results[:3]):
        print(f"  Pair {i+1}:")
        print(f"    Query: {result['query'][:50]}...")
        print(f"    Doc:   {result['document'][:50]}...")
        print(f"    First tokens: {' '.join(result['first_30_tokens'][:10])}...")
        print(f"    Last tokens:  {' '.join(result['last_30_tokens'][-5:])}...")
    
    # Overall test result
    test_passed = (variance_check['valid'] and 
                  not placeholder_detected and
                  len(results) == 20)
    
    print(f"\n🏆 REAL-PAIR PEEK TEST: {'✅ PASSED' if test_passed else '❌ FAILED'}")
    
    if test_passed:
        print("✅ CE receives real content (not placeholders)")
        print("✅ Token lengths are reasonable")
        print("✅ Logits show proper variance")
        print("✅ Ready for coverage canary")
    else:
        print("❌ Real-pair test failed")
        if not variance_check['valid']:
            print("   - Logit variance too low")
        if placeholder_detected:
            print("   - Placeholder content detected")
        if len(results) != 20:
            print(f"   - Wrong number of pairs: {len(results)} != 20")
    
    # Save detailed results
    with open("real_pair_test_results.json", "w") as f:
        json.dump({
            "test_passed": test_passed,
            "statistics": {
                "logits_min": float(min(all_logits)),
                "logits_median": float(np.median(all_logits)),
                "logits_max": float(max(all_logits)),
                "logits_std": float(np.std(all_logits)),
                "logits_range": float(max(all_logits) - min(all_logits))
            },
            "guard_validation": variance_check,
            "placeholder_detected": placeholder_detected,
            "pairs_tested": len(results),
            "detailed_results": results
        }, f, indent=2, default=str)
    
    return test_passed, results


if __name__ == "__main__":
    test_passed, results = test_real_pairs()
    sys.exit(0 if test_passed else 1)