# Lethe→StreamingLLM Hybrid Evaluation Report

Generated: 2025-09-10 21:15:25

## Evaluation Configuration

- **Methods**: streaming, lethe, hybrid
- **Keep Ratios**: 0.08, 0.15, 0.30
- **Datasets**: code_debug, code_qa, zh_qa
- **Promotion Threshold**: ≤+1.0ms p95

## Results Summary

| Method | Keep Ratio | Dataset | Accuracy | P@5 | P95 (ms) | ΔCBU/1k | KV Reuse |
|--------|------------|---------|----------|-----|----------|---------|----------|
| streaming | 0.08 | code | 0.000 | 0.000 | 190.2 | 0.0102 | 0.000 |
| streaming | 0.08 | zh_qa | 0.000 | 0.000 | 1642.9 | 0.0102 | 0.000 |
| streaming | 0.15 | code | 0.000 | 0.000 | 210.8 | 0.0095 | 0.000 |
| streaming | 0.15 | zh_qa | 0.000 | 0.000 | 1262.8 | 0.0095 | 0.000 |
| streaming | 0.30 | code | 0.000 | 0.000 | 148.3 | 0.0080 | 0.000 |
| streaming | 0.30 | zh_qa | 0.000 | 0.000 | 1440.7 | 0.0080 | 0.000 |
| lethe | 0.08 | code | 0.000 | 0.000 | 195.5 | 0.0102 | 0.000 |
| lethe | 0.08 | zh_qa | 0.000 | 0.000 | 1587.9 | 0.0102 | 0.000 |
| lethe | 0.15 | code | 0.000 | 0.000 | 259.3 | 0.0095 | 0.000 |
| lethe | 0.15 | zh_qa | 0.000 | 0.000 | 2388.7 | 0.0095 | 0.000 |
| lethe | 0.30 | code | 0.000 | 0.000 | 290.9 | 0.0080 | 0.000 |
| lethe | 0.30 | zh_qa | 0.000 | 0.000 | 2206.6 | 0.0080 | 0.000 |
| hybrid | 0.08 | code | 0.000 | 0.000 | 168.1 | 0.0102 | 0.000 |
| hybrid | 0.08 | zh_qa | 0.000 | 0.000 | 1445.6 | 0.0102 | 0.000 |
| hybrid | 0.15 | code | 0.000 | 0.000 | 121.5 | 0.0095 | 0.000 |
| hybrid | 0.15 | zh_qa | 0.000 | 0.000 | 1622.0 | 0.0095 | 0.000 |
| hybrid | 0.30 | code | 0.000 | 0.000 | 150.6 | 0.0080 | 0.000 |
| hybrid | 0.30 | zh_qa | 0.000 | 0.000 | 1744.2 | 0.0080 | 0.000 |

## Promotion Analysis

### Keep Ratio 0.08

**Status**: 🔴 **NOT PROMOTED**

- **Performance Improvement**: False
- **Latency Constraint**: True
- **Details**: P@5: +0.000, ΔCBU: +0.000, Δp95: -22.1ms

### Keep Ratio 0.15

**Status**: 🔴 **NOT PROMOTED**

- **Performance Improvement**: False
- **Latency Constraint**: True
- **Details**: P@5: +0.000, ΔCBU: +0.000, Δp95: -89.3ms

### Keep Ratio 0.30

**Status**: 🔴 **NOT PROMOTED**

- **Performance Improvement**: False
- **Latency Constraint**: False
- **Details**: P@5: +0.000, ΔCBU: +0.000, Δp95: +2.3ms

## Statistical Analysis

### Keep Ratio 0.08

- **streaming**: 0.000 (95% CI: [0.000, 0.000])
- **lethe**: 0.000 (95% CI: [0.000, 0.000])
- **hybrid**: 0.000 (95% CI: [0.000, 0.000])

### Keep Ratio 0.15

- **streaming**: 0.000 (95% CI: [0.000, 0.000])
- **lethe**: 0.000 (95% CI: [0.000, 0.000])
- **hybrid**: 0.000 (95% CI: [0.000, 0.000])

### Keep Ratio 0.30

- **streaming**: 0.000 (95% CI: [0.000, 0.000])
- **lethe**: 0.000 (95% CI: [0.000, 0.000])
- **hybrid**: 0.000 (95% CI: [0.000, 0.000])

## Conclusion

❌ **Hybrid system does not meet promotion criteria.**
Further optimization required before production deployment.
