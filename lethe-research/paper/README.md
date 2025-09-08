# Lethe NeurIPS 2025 Paper

This directory contains the complete NeurIPS 2025 research paper for the Lethe project.

## Files

- `lethe_neurips2025.tex` - Complete LaTeX source for the paper
- `neurips_2025.sty` - NeurIPS 2025 style file (minimal version)
- `template.tex` - Original template with placeholders
- `README.md` - This file

## Paper Overview

**Title**: "Lethe: Local-First Conversational Context Packing with Hybrid Retrieval and Adaptive Planning"

**Abstract**: We present Lethe, a local-first conversational context packing system that combines hybrid retrieval with adaptive planning for long-context AI applications. Unlike existing RAG systems that depend on cloud infrastructure, Lethe operates entirely on-device using transformers.js and dynamically adapts its search strategy based on conversation state.

## Key Contributions

1. **Local-First Architecture**: Complete operation using transformers.js with graceful degradation
2. **Per-Session DF/IDF**: Conversation-specific term importance calculation
3. **Hybrid Retrieval Fusion**: Dynamic weighting of BM25 and vector similarity
4. **Entity-Based Diversification**: Submodular optimization for conversational coverage
5. **Adaptive Planning**: Three-mode strategy selection (explore/verify/exploit)

## Experimental Results

- **Dataset**: LetheBench with 703 datapoints across 4 conversation domains (chatty_prose, code_heavy, mixed, tool_results)
- **Baselines**: 7 competitive baseline methods + 4 Lethe iterations evaluated (11 total)
- **Quality**: 77.7% NDCG@10 improvement over best baseline (0.917 vs 0.516 BM25+Vector)
- **Efficiency**: Statistical significance p < 0.001 after Bonferroni correction
- **Coverage**: 133.3% Coverage@N improvement over best baseline (0.770 vs 0.330)

## Compilation

To compile the paper (requires LaTeX):

```bash
pdflatex lethe_neurips2025.tex
bibtex lethe_neurips2025
pdflatex lethe_neurips2025.tex
pdflatex lethe_neurips2025.tex
```

Note: The paper includes inline bibliography and should compile without external .bib files.

## Technical Details

- **Architecture**: Local-first with transformers.js
- **Chunking**: 320 tokens with 64 token overlap
- **Planning**: Rule-based system with entity recurrence analysis
- **Hybrid Fusion**: Plan-specific α weights (Verify: 0.7, Explore: 0.3, Exploit: 0.5)
- **Diversification**: Submodular entity coverage optimization

## Statistical Validation

- **Rigorous Testing**: Pairwise Mann-Whitney U tests with Bonferroni correction
- **Effect Sizes**: Cohen's d calculated for all comparisons with medium-to-large effects
- **Significance**: All key improvements significant at p < 0.001 after multiple testing correction
- **Confidence Intervals**: 95% confidence intervals reported for all performance metrics
- **Dataset Balance**: 79.7% synthetic + 20.3% human-labeled data across 4 balanced domains

## Reproducibility

All experimental code and datasets are available with complete reproducibility:
- `/datasets/` - LetheBench dataset construction (703 queries, 4 domains)
- `/experiments/` - Evaluation framework with 11 methods tested
- `/scripts/` - Baseline implementations and statistical analysis
- `/artifacts/` - Complete experimental results with statistical validation
- `/paper/tables/` - LaTeX tables with publication-ready results and confidence intervals

## Citation

```bibtex
@inproceedings{lethe2025,
  title={Lethe: Local-First Conversational Context Packing with Hybrid Retrieval and Adaptive Planning},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```