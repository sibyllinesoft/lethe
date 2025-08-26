# Lethe Full Grid Experimental Results

Generated: 2025-08-25 08:40:12

## Executive Summary

- **Total Configurations Tested**: 24
- **Success Rate**: 100.0%
- **Duration**: 0.0 minutes
- **Quick Mode**: Yes

**Best Performing Variant**: V2_iter1 (nDCG@10: 1.000)

## Variant Results

### V2_iter1: Core Hybrid Retrieval

- **Configurations Tested**: 12
- **Success Rate**: 100.0%
- **Best nDCG@10**: 1.000
- **Average Latency P95**: 0.7ms
- **Best Parameters**: {'alpha': 0.1, 'k_initial': 20, 'k_final': 10, 'chunk_size': 256, 'chunk_overlap': 32}

- **Improvement over Baseline**: +0.550 (122.2%)
- **Meets Target**: Yes

### V3_iter2: Query Understanding & Reranking

- **Configurations Tested**: 12
- **Success Rate**: 100.0%
- **Best nDCG@10**: 1.000
- **Average Latency P95**: 0.9ms
- **Best Parameters**: {'beta': 0.0, 'k_rerank': 10, 'query_rewrite': 'none', 'hyde_num_docs': 1, 'decompose_max_subqueries': 2, 'retrieval_stages': 'single'}

- **Improvement over Baseline**: +0.550 (122.2%)
- **Meets Target**: Yes

## Performance Analysis

- **Latency P95 Range**: 0.5 - 1.3ms
- **Average Latency P95**: 0.8ms

## Next Steps

1. Analyze best configurations for production deployment
2. Run extended evaluation on full dataset
3. Conduct statistical significance testing
4. Prepare final results for publication
