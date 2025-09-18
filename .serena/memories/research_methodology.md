# Research Methodology & Standards

## Research Hypotheses
**H1 (Quality)**: Hybrid retrieval beats baselines on nDCG@k, Recall@k, MRR@10
**H2 (Efficiency)**: <3s latency, <1.5GB memory under load
**H3 (Coverage)**: Diversification increases coverage@N and entity diversity
**H4 (Adaptivity)**: Adaptive planning reduces contradiction rates and improves consistency

## Experimental Design
- **Dataset**: LetheBench with 139 queries across 3 domains (api, web, cli)
- **Baselines**: 7 competitive methods (window, BM25-only, vector-only, cross-encoder, FAISS, MMR, BM25+Vector)
- **Grid Search**: 9 parameters optimized across 63 configurations
- **Metrics**: 20+ evaluation metrics with statistical validation

## Statistical Rigor Standards
- **Bootstrap confidence intervals** (1000 samples, 95% CI)
- **Multiple comparison correction** (Bonferroni)
- **Effect size reporting** (Cohen's d, Hedges' g)
- **Significance testing** (α = 0.05, power = 0.8)
- **Fixed seeds** for reproducibility
- **Fraud-proof validation** with automated sanity checks

## Quality Gates
- All 52 validation checks must pass
- Statistical significance required for claims
- Complete reproducibility with version control
- Open-source implementation available

## Paper Requirements
- **NeurIPS 2025 format**: 9 pages + references + appendix
- **Automated table generation** from experimental results
- **Mathematical notation** for all algorithms
- **Publication-ready figures** with proper captions
- **Comprehensive appendix** with full technical details