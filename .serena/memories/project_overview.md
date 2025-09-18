# Lethe Project Overview

## Purpose
The Lethe project is a comprehensive research initiative for developing and evaluating a local-first conversational context packing system with hybrid retrieval and adaptive planning. The goal is to generate a publication-ready NeurIPS 2025 research paper.

## Project Structure
- **Main System**: ctx-run - NPX context manager for AI conversation history search
- **Research Infrastructure**: Complete framework in `lethe-research/` directory
- **Paper**: NeurIPS 2025 format LaTeX paper targeting 9 pages + appendix

## Key Components
1. **Local-First Architecture**: Complete operation using transformers.js 
2. **Per-Session DF/IDF**: Conversation-specific term importance
3. **Hybrid Retrieval Fusion**: Dynamic BM25 + vector similarity weighting
4. **Entity-Based Diversification**: Submodular optimization for coverage
5. **Adaptive Planning**: Three-mode strategy (explore/verify/exploit)

## Research Status
- **Infrastructure**: Complete ✅ (52/52 validation checks passing)
- **Dataset**: LetheBench with 139 queries from 3 domains
- **Baselines**: 7 competitive methods implemented 
- **Experimental Results**: Available in artifacts directory
- **Paper Template**: Existing LaTeX structure with NeurIPS 2025 format

## Target Paper Sections
1. Abstract (150-200 words) with measurable claims
2. Introduction with numbered contributions
3. Related Work (RAG, local-first, conversational IR, diversification)
4. Method (architecture, algorithms, mathematical notation)
5. LetheBench Dataset description
6. Experimental Setup and Results
7. Analysis with statistical significance
8. Limitations & Ethics
9. Conclusion with future work
10. Complete Appendix with technical details