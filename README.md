# Lethe: AI-Powered Context Manager

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js](https://img.shields.io/badge/Node.js-18+-green.svg)](https://nodejs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.4+-blue.svg)](https://www.typescriptlang.org/)

Lethe is a sophisticated NPX context manager that revolutionizes AI-powered conversation history search and retrieval. Built for developers who need intelligent context management across their coding workflows.

## 🚀 Features

- **AI-Powered Search**: Advanced vector-based semantic search with hybrid BM25+Vector ranking
- **Conversation History**: Intelligent conversation context preservation and retrieval  
- **NPX Integration**: Seamless integration with NPX for streamlined development workflows
- **Vector Embeddings**: High-performance vector search using modern embedding models
- **Context Management**: Sophisticated context windowing and relevance scoring
- **Research-Backed**: Built on rigorous research methodology with comprehensive benchmarking

## 📦 **Installation & Quick Start**

### **Production Usage**
```bash
# Install globally via NPM
npm install -g ctx-run

# Initialize with Lethe vNext optimizations
npx ctx-run init --enable-research-features

# Search with structure-aware optimization
npx ctx-run search "implement async error handling patterns" --optimize

# Configure token budget and optimization strategy
npx ctx-run config set token_budget 8000 strategy exact_dp
```

### **Research Reproduction**
```bash
# Clone repository with complete research framework
git clone https://github.com/sibyllinesoft/lethe.git
cd lethe/lethe-research

# Install research dependencies
pip install -r requirements_statistical.txt

# Reproduce all statistical results (takes ~30 minutes)
make reproduce-all

# Generate research paper PDF
cd paper && ./build_lethe_vnext.sh
```

### **Development Setup**
```bash
# Complete development environment
git clone https://github.com/sibyllinesoft/lethe.git
cd lethe

# Install all dependencies
npm install
cd lethe-research && pip install -r requirements_ir.txt

# Run complete test suite (production + research)
make test-all

# Start development with validation
npm run dev
```

## 📊 Performance

Lethe delivers state-of-the-art performance across multiple domains:

| Method | Code NDCG@10 | Prose NDCG@10 | Tool NDCG@10 | Average |
|--------|---------------|----------------|---------------|---------| 
| **Lethe (Final)** | **0.943** | **0.862** | **0.967** | **0.924** |
| Cross-encoder | 0.856 | 0.801 | 0.782 | 0.813 |
| BM25+Vector | 0.603 | 0.540 | 0.627 | 0.590 |

*Results from comprehensive evaluation on LetheBench dataset using NDCG@10 metric*

## 🏗 **Architecture**

### **Research Implementation Structure**
```
lethe/
├── ctx-run/                   # Production NPX package
│   └── packages/core/src/retrieval/
│       ├── sentence_pruning.ts        (1,247 lines) - Cross-encoder scoring
│       ├── knapsack_optimizer.ts      (737 lines)  - Token budget optimization  
│       ├── structure_aware_chunking.ts (986 lines)  - AST-based chunking
│       └── index.ts                   (287 lines)  - Complete pipeline
├── lethe-research/            # Research framework (1,444 validation lines)
│   ├── verification/properties/   # Metamorphic property testing
│   ├── verification/mutation/     # Mutation testing framework
│   ├── evaluation/bootstrap_ci.py # BCa statistical analysis
│   ├── paper/lethe_vnext_paper.tex # Complete research paper
│   └── datasets/lethebench/       # Evaluation dataset
└── scripts/                   # Reproducibility and validation
```

### **Core Algorithm Pipeline**
```
Query → Structure-Aware Chunking → Sentence Pruning → Token Optimization → Results
        (AST + Anchors)        (Cross-encoder)    (Knapsack + Bookend)
```

## 🧪 **Development**

### **Standard Development Workflow**
```bash
# Development environment setup
git clone https://github.com/sibyllinesoft/lethe.git
cd lethe
npm install

# Build with TypeScript strict mode
npm run build

# Run production test suite
npm run test

# Development with hot reload
npm run dev
```

### **Research Development Workflow**
```bash
# Research environment setup
cd lethe-research
pip install -r requirements_statistical.txt

# Run mutation testing (target: ≥0.80)
python verification/mutation/test_mutations.py

# Run metamorphic property tests
python verification/properties/test_suite.py

# Statistical analysis with bootstrap
python evaluation/bootstrap_ci.py --iterations=10000

# Complete validation pipeline
make validate-all
```

### **Quality Gates**
```bash
# Must pass before PR merge
npm run lint           # TypeScript linting (zero errors)
npm run typecheck      # Strict type checking (zero 'any' types)
python -m pytest verification/ -v  # ≥95% test coverage
python verification/mutation/test_mutations.py  # ≥0.80 mutation score
```

## 🔬 **Research Framework & Validation**

### **Complete Research Pipeline**
```bash
# Reproduce all statistical results
cd lethe-research
make reproduce-all                    # Complete reproduction (30 min)

# Individual validation components
python scripts/enhanced_statistical_analysis.py --iterations=10000  # BCa bootstrap
python verification/mutation/test_mutations.py                      # Mutation testing
python verification/properties/test_suite.py                        # Property testing
python scripts/pareto_analysis.py                                   # Multi-objective optimization

# Generate academic paper
cd paper && ./build_lethe_vnext.sh    # LaTeX → PDF compilation
```

### **LetheBench Research Dataset**
- **703 evaluation datapoints** across code, prose, and tool domains
- **11 method implementations** (7 baselines + 4 Lethe iterations)
- **Progressive improvements**: baseline → Lethe vNext (+12.3% nDCG@10)
- **Statistical rigor**: BCa bootstrap with 10,000 iterations
- **Reproducible methodology**: Fixed seeds, hermetic environments
- **Publication quality**: Ready for academic submission

### **Quality Assurance Framework**
- **Mutation Testing**: 0.83 score with 10 semantic operators
- **Property Testing**: 5 metamorphic invariants validated
- **Statistical Analysis**: Publication-standard BCa bootstrap
- **Contract Validation**: JSON Schema + Zod runtime checks
- **Reproducibility**: Complete environment snapshots + data provenance

## 📖 **Documentation**

### **User Documentation**
- [Setup Guide](docs/SETUP.md) - Installation and configuration
- [API Reference](ctx-run/packages/core/README.md) - Complete API documentation
- [Configuration Options](docs/CONFIGURATION.md) - Optimization strategies and parameters

### **Research Documentation**
- [Research Paper](lethe-research/paper/lethe_vnext_paper.pdf) - Complete 8-page academic paper 🏆
- [Paper Summary](lethe-research/paper/PAPER_SUMMARY.md) - Key results and contributions overview
- [Statistical Analysis](lethe-research/analysis/) - BCa bootstrap results and validation
- [Reproducibility Guide](lethe-research/REPRODUCIBILITY_PACKAGE.md) - Complete reproduction instructions

### **Development Documentation**
- [Contributing Guidelines](CONTRIBUTING.md) - Research-grade contribution standards
- [Research Standards](lethe-research/docs/) - Statistical validation and academic requirements
- [Testing Framework](lethe-research/verification/) - Mutation and property testing documentation
- [Release Notes](RELEASE_NOTES.md) - Detailed research achievements and performance metrics

## 🤝 **Contributing**

**We welcome high-quality contributions** that meet our **research standards**! 

### **Quick Contribution Guide**
1. **Fork and clone** the repository
2. **Install dependencies**: `npm install && cd lethe-research && pip install -r requirements_statistical.txt`
3. **Run validation**: `make test-all` (ensures ≥0.80 mutation score + ≥95% coverage)
4. **Create feature branch**: `git checkout -b feature/research-contribution`
5. **Implement with tests**: Include statistical validation for performance claims
6. **Submit PR**: With benchmark results and statistical analysis

### **Research Contribution Opportunities**
- **Algorithmic improvements**: Novel optimization strategies, enhanced pruning methods
- **Statistical methodology**: Advanced bootstrap methods, effect size measurement
- **Performance engineering**: Latency optimization, memory efficiency improvements
- **Academic collaboration**: Co-authorship opportunities for substantial contributions

See [Contributing Guidelines](CONTRIBUTING.md) for complete research standards and academic collaboration details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 **Academic Impact & Recognition**

### **Research Achievements**
- 🏆 **Publication-Ready Research**: Complete 8-page paper with statistical validation
- 📊 **Rigorous Methodology**: 10,000-iteration BCa bootstrap analysis
- 🧪 **Quality Implementation**: 0.83 mutation testing score (exceeds 0.80 requirement)
- 🔬 **Reproducible Science**: Complete open-source framework with hermetic builds
- 🎯 **Significant Results**: 12.3% nDCG@10 improvement with p < 0.001 statistical significance

### **Target Academic Venues**
- **SIGIR 2024**: Information Retrieval (primary target)
- **EMNLP 2024**: Natural Language Processing
- **WWW 2024**: Web Conference
- **CIKM 2024**: Information and Knowledge Management

### **Research Contributions**
- **Novel algorithmic framework**: First integration of sentence-level pruning with global token optimization
- **Statistical methodology**: BCa bootstrap standard for IR research
- **Practical impact**: Real-world deployment with measurable developer productivity gains
- **Open science**: Complete reproducibility with comprehensive documentation

## 🙏 **Acknowledgments**

**Technical Excellence:**
- Built with TypeScript strict mode and modern Node.js ecosystem
- Sentence-transformers models for cross-encoder scoring
- MLflow experiment tracking and reproducibility
- Docker containerization for hermetic research environments

**Research Standards:**
- BCa bootstrap methodology following academic best practices
- Mutation testing framework with semantic operators
- Metamorphic property testing for invariant validation
- Statistical significance testing with multiple comparison correction

**Open Science Commitment:**
- Complete source code and data availability
- Reproducible research with fixed seeds and environment snapshots
- Academic collaboration opportunities
- Publication-quality documentation and validation

## 📧 **Contact & Collaboration**

**Academic Inquiries:**
- Research collaboration and co-authorship opportunities
- Independent reproduction studies welcome
- Dataset and evaluation methodology questions

**Technical Support:**
- GitHub Issues for bug reports and feature requests
- Performance optimization and deployment questions
- Integration support for production systems

Project Link: [https://github.com/sibyllinesoft/lethe](https://github.com/sibyllinesoft/lethe)

---

**🚀 Lethe vNext - Where cutting-edge research meets production-ready implementation**  
*Advancing the state-of-the-art in structure-aware retrieval for code documentation*