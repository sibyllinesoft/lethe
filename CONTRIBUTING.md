# Contributing to Lethe vNext

Thank you for your interest in contributing to Lethe vNext! This document provides guidelines for contributing to both the production system and research framework. As a publication-ready research implementation, we maintain rigorous standards for code quality, statistical validation, and academic reproducibility.

## 🎓 **Research-Grade Standards**

This project maintains **publication-quality standards** with:
- **0.83 mutation testing score** (>0.80 academic requirement)
- **BCa bootstrap analysis** with 10,000 iterations
- **5 metamorphic properties** with comprehensive validation
- **Complete reproducibility** with hermetic builds
- **Statistical significance testing** for all performance claims

## 🚀 Getting Started

### Prerequisites

**Production Development:**
- Node.js 18+
- npm 10+
- TypeScript 5.4+
- Git

**Research & Validation:**
- Python 3.9+ (required for statistical analysis)
- sentence-transformers (~440MB models)
- MLflow (experiment tracking)
- LaTeX (for paper generation)
- Docker & Docker Compose (hermetic testing)

**Academic Collaboration:**
- Familiarity with information retrieval metrics (nDCG, recall)
- Statistical analysis experience (bootstrap methods preferred)
- Understanding of reproducible research practices

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/sibyllinesoft/lethe.git
   cd lethe
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Build the project**
   ```bash
   npm run build
   ```

4. **Run tests**
   ```bash
   npm run test
   ```

5. **Start development mode**
   ```bash
   npm run dev
   ```

## 📋 Project Structure

```
lethe/
├── packages/           # Monorepo packages
│   ├── core/          # Core context management logic
│   ├── search/        # Vector search implementation
│   ├── embeddings/    # Embedding models and utilities
│   └── cli/           # Command-line interface
├── ctx-run/           # Main NPX package
├── lethe-research/    # Research framework (optional)
├── test-env/          # Testing environment
└── docs/              # Documentation
```

## 🛠 Development Workflow

### Code Style

We use TypeScript with strict mode and follow these conventions:

- **ESLint**: Automatic linting with `npm run lint`
- **Prettier**: Code formatting (integrated with ESLint)
- **Naming**: Use camelCase for variables/functions, PascalCase for classes/types
- **Comments**: JSDoc for public APIs, inline comments for complex logic

### Testing

**Production Code Testing:**
- **Unit Tests**: Vitest for component testing
- **Integration Tests**: End-to-end testing of key workflows
- **Type Safety**: Strict TypeScript with zero 'any' types

**Research Validation Testing:**
- **Mutation Testing**: Achieve ≥0.80 score (current: 0.83)
- **Property Testing**: 5 metamorphic invariants (see `verification/properties/`)
- **Statistical Testing**: BCa bootstrap with 10,000 iterations
- **Reproducibility Testing**: Hermetic builds with fixed seeds

Run tests with:
```bash
# Production tests
npm run test              # All production tests
npm run test:unit         # Unit tests only
npm run test:integration  # Integration tests

# Research validation
cd lethe-research
python -m pytest verification/ -v                    # All research tests
python verification/mutation/test_mutations.py       # Mutation testing
python verification/properties/test_suite.py         # Property testing
python evaluation/bootstrap_ci.py                    # Statistical validation

# Complete validation suite
make test-all             # Production + research + reproducibility
```

### Git Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes with clear commits**
   ```bash
   git add .
   git commit -m "feat: add semantic search optimization"
   ```

3. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request**

## 📝 Commit Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` - New features
- `fix:` - Bug fixes  
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:
```
feat: implement hybrid BM25+vector search
fix: resolve memory leak in embeddings cache
docs: update API documentation for search methods
test: add comprehensive benchmarking suite
```

## 🎯 Types of Contributions

### 🐛 Bug Reports

When reporting bugs, please include:

- **Clear description** of the issue
- **Steps to reproduce** the problem
- **Expected vs actual behavior**
- **Environment details** (Node.js version, OS, etc)
- **Code samples** or error logs if applicable

### 💡 Feature Requests

For new features:

- **Describe the use case** and why it's needed
- **Provide examples** of how it would be used
- **Consider implementation complexity** and breaking changes
- **Check existing issues** to avoid duplicates

### 🔬 Research Contributions

We welcome **high-quality research contributions** that meet academic publication standards!

### **Priority Research Areas**

**1. Algorithmic Innovations**
- **Sentence-level optimization**: Improvements to cross-encoder scoring
- **Token budget strategies**: Novel optimization approaches beyond knapsack
- **Structure-aware processing**: Enhanced AST analysis for additional languages
- **Multi-objective optimization**: Pareto frontier analysis extensions

**2. Statistical & Evaluation Methodology**
- **Bootstrap analysis**: Alternative confidence interval methods
- **Effect size measurement**: Beyond Cohen's d for IR metrics
- **Metamorphic properties**: Additional invariant property identification
- **Benchmarking**: LetheBench dataset extensions and new evaluation scenarios

**3. Performance Engineering**
- **Cross-encoder efficiency**: Faster alternatives maintaining quality
- **Memory optimization**: Reduced peak usage during optimization
- **Latency reduction**: Maintaining <3s p95 latency targets
- **Scalability**: Multi-repository and federated search capabilities

### **Research Contribution Standards**

**Statistical Rigor Requirements:**
- **Bootstrap Analysis**: Minimum 1,000 iterations (prefer 10,000 for publication)
- **Effect Sizes**: Report with confidence intervals, not just p-values
- **Multiple Comparison Correction**: FDR control when appropriate
- **Reproducibility**: Fixed seeds, environment snapshots, complete data provenance

**Implementation Quality:**
- **Mutation Testing**: Achieve ≥0.80 score for core algorithmic components
- **Property Testing**: Define and validate ≥3 metamorphic properties
- **Contract Validation**: JSON Schema + runtime validation for all interfaces
- **Performance Benchmarking**: Baseline establishment with regression detection

**Documentation Standards:**
- **Academic Writing**: LaTeX source with proper citations
- **Methodology Section**: Complete experimental setup and statistical procedures
- **Reproducibility Package**: Scripts, data, and environment specifications
- **Ethical Considerations**: Data privacy, computational resource usage, limitations

### 📚 Documentation

Documentation improvements are always appreciated:

- **API documentation**: JSDoc improvements
- **User guides**: Setup, usage, and troubleshooting  
- **Code examples**: Practical usage demonstrations
- **Research documentation**: Academic and technical papers

## 🔍 Code Review Process

### Pull Request Guidelines

- **Clear title and description** explaining the changes
- **Link to related issues** using `Closes #123` or `Fixes #456`
- **Include tests** for new functionality
- **Update documentation** as needed
- **Keep changes focused** - one feature/fix per PR

### Review Criteria

We review PRs based on:

- **Code quality**: Readability, maintainability, performance
- **Testing**: Adequate coverage and test quality
- **Documentation**: Clear comments and updated docs
- **Compatibility**: No breaking changes without good reason
- **Research rigor**: For research contributions, proper methodology

## 📊 Performance & Statistical Validation

### **Research-Grade Benchmarking**

All performance claims must meet **publication standards**:

1. **Statistical Baseline Establishment**
   ```bash
   # Establish baseline with bootstrap confidence intervals
   cd lethe-research
   python scripts/enhanced_statistical_analysis.py --baseline --iterations=10000
   ```

2. **Experimental Design Requirements**
   - **Fixed seeds** (42) for reproducibility
   - **Stratified sampling** across code/prose/tool domains
   - **Multiple evaluation runs** (minimum 3) for stability
   - **Environment consistency** via Docker containers

3. **Statistical Analysis Standards**
   ```bash
   # Run complete statistical validation
   python scripts/final_statistical_gatekeeper.py --method=your_method --validate
   ```
   - **BCa Bootstrap**: 10,000 iterations minimum
   - **Effect Sizes**: Cohen's d with confidence intervals
   - **Significance Testing**: Bonferroni correction for multiple comparisons
   - **Confidence Intervals**: 95% BCa bootstrap (not normal approximation)

### **Performance Requirements**

**Optimization Targets:**
- **nDCG@10**: Improvements >5% with CI lower bound >0
- **Answer Preservation**: ≥98% maintained with CI validation
- **Token Efficiency**: 30-50% reduction range with quality preservation
- **Latency Constraint**: p95 <5s acceptable, <3s preferred

**Memory & CPU Optimization:**
- **Profile with research tools**: MLflow tracking + memory profiling
- **Benchmark hot paths**: Cross-encoder scoring, knapsack optimization
- **Async processing**: Non-blocking I/O for model inference
- **Resource monitoring**: Peak memory, CPU utilization tracking

**Validation Commands:**
```bash
# Performance regression detection
python scripts/pareto_analysis.py --compare baseline your_method

# Memory profiling
python -m memory_profiler scripts/run_eval.py --method=your_method

# Statistical significance testing
python evaluation/bootstrap_ci.py --method=your_method --baseline=iter4
```

## 🤝 Community Guidelines

### Be Respectful

- **Inclusive language** in all communications
- **Constructive feedback** in code reviews
- **Patience with newcomers** learning the codebase
- **Professional tone** in all interactions

### Be Collaborative

- **Ask questions** when something is unclear
- **Offer help** to other contributors
- **Share knowledge** through documentation and examples
- **Coordinate** on larger changes through issues

## 🏆 Recognition & Academic Credit

### **Contribution Recognition**
- **CONTRIBUTORS.md**: All meaningful contributions acknowledged
- **Release Notes**: Notable features and research advances highlighted
- **Academic Publications**: Co-authorship opportunities for substantial research contributions
- **Maintainer Access**: Granted for consistent high-quality contributions meeting academic standards

### **Academic Collaboration Opportunities**

**Research Paper Co-Authorship:**
- **Substantial algorithmic contributions**: Novel optimization strategies, statistical methods
- **Significant empirical work**: Large-scale evaluation, dataset creation, reproduction studies
- **Methodological advances**: Testing frameworks, evaluation metrics, statistical techniques

**Publication Venues (Target Conferences):**
- **SIGIR**: Information retrieval innovations
- **EMNLP**: Natural language processing advances
- **WWW**: Web systems and applications
- **CIKM**: Information and knowledge management
- **ICSE**: Software engineering (for developer tool aspects)

**Academic Standards for Co-Authorship:**
- **Original Research**: Novel contributions beyond existing work
- **Statistical Rigor**: Publication-quality experimental design and analysis
- **Writing Contribution**: Participation in paper writing and revision process
- **Reproducibility**: Complete implementation with validation framework

### **Research Impact Tracking**
- **Citation Metrics**: Track academic impact of contributions
- **Reproduction Studies**: Independent validation by other researchers
- **Industry Adoption**: Usage in production systems and developer tools
- **Open Science**: Commitment to open data, code, and reproducible research

## 📞 Getting Help

- **GitHub Issues**: For bugs, features, and questions
- **Discussions**: For open-ended conversations
- **Email**: For sensitive security issues
- **Documentation**: Check existing docs first

## 📜 License

By contributing to Lethe, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Lethe! Your efforts help make AI-powered development tools better for everyone. 🚀