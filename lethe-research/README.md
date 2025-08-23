# Lethe Research Infrastructure

This directory contains the comprehensive research framework for the Lethe academic paper development (NeurIPS submission).

## 🎯 Research Mission

Systematic evaluation of Lethe's hybrid retrieval system through rigorous experimental design, demonstrating superior performance across 4 key hypotheses:

- **H1** (Quality): Hybrid retrieval beats baselines on nDCG@k, Recall@k, MRR@10
- **H2** (Efficiency): <3s latency, <1.5GB memory under load
- **H3** (Robustness): Diversification increases coverage@N 
- **H4** (Adaptivity): Adaptive planning reduces hallucination rates

## 📁 Directory Structure

```
lethe-research/
├── datasets/           # LetheBench construction and storage
├── experiments/        # Grid search and evaluation configurations
├── artifacts/          # Raw experimental outputs and logs
├── paper/             # LaTeX source, figures, tables
├── scripts/           # Automation, runners, and orchestration
└── analysis/          # Statistical analysis and visualization
```

## 🔬 Experimental Framework

### Baseline Implementation Status
**Research Freeze**: `research-freeze-v1` (SHA: 5cda28f)
**Environment**: Node.js v20.18.1, WASM vector backend, Ollama available
**Health Status**: All components passing diagnostics

### Grid Configuration Variables
- α/β weighting parameters (0.1-0.9 range)
- Chunk size/overlap (128-512 tokens, 16-128 overlap)
- Reranking parameters (top-k, similarity thresholds)
- Diversification pack sizes (5-50 results)
- Planning strategies (adaptive vs fixed)
- Backend configurations (WASM vs native)

### Baseline Implementations Required
1. **Window Baseline**: Recency-only retrieval
2. **BM25-only**: Pure lexical search
3. **Vector-only**: Pure semantic search  
4. **BM25+Vector**: No rerank/diversify
5. **Cross-encoder**: Rerank over BM25 results
6. **FAISS IVF-Flat**: Alternative RAG system
7. **MMR Alternative**: Different diversification approach

### Evaluation Metrics
- **Quality**: nDCG@k, Recall@k, MRR@10
- **Coverage**: Entity coverage@N, topic diversity
- **Efficiency**: P95 latency, memory consumption
- **Robustness**: Contradiction detection rates
- **Statistical**: Bootstrap confidence intervals, effect sizes

## 🚀 One-Command Automation

**Complete Pipeline Execution**:
```bash
./scripts/run_full_evaluation.sh
```

**Individual Components**:
```bash
./scripts/create_dataset.sh      # LetheBench construction
./scripts/run_grid_search.sh     # Parameter optimization
./scripts/evaluate_baselines.sh  # Comparative evaluation
./scripts/generate_paper.sh      # LaTeX compilation
```

## 📊 Statistical Rigor

- **Reproducibility**: Fixed seeds, pinned dependencies
- **Significance**: Bootstrap CIs, permutation tests
- **Effect Sizes**: Cohen's d, confidence intervals
- **Multiple Comparisons**: Bonferroni correction
- **Fraud Prevention**: Automated sanity checks

## 🔄 Development Workflow

1. **Stage 1**: Dataset construction (LetheBench)
2. **Stage 2**: Baseline implementation 
3. **Stage 3**: Grid search optimization
4. **Stage 4**: Evaluation and analysis
5. **Stage 5**: Paper generation and submission

**Status**: Infrastructure setup complete ✅
**Next**: Dataset construction and baseline implementation

---

*Infrastructure established: 2025-08-23*  
*Research freeze: research-freeze-v1*  
*Target venue: NeurIPS 2025*