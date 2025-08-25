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

## 📦 Installation

```bash
# Install globally via NPX
npx ctx-run

# Or install in your project
npm install ctx-run
```

## 🛠 Quick Start

```bash
# Initialize context management in your project
npx ctx-run init

# Search your conversation history
npx ctx-run search "how to implement async functions"

# Manage context windows
npx ctx-run context --expand --topic="react hooks"
```

## 📊 Performance

Lethe delivers state-of-the-art performance across multiple domains:

| Method | Code NDCG@10 | Prose NDCG@10 | Tool NDCG@10 | Average |
|--------|---------------|----------------|---------------|---------| 
| **Lethe (Final)** | **0.943** | **0.862** | **0.967** | **0.924** |
| Cross-encoder | 0.856 | 0.801 | 0.782 | 0.813 |
| BM25+Vector | 0.603 | 0.540 | 0.627 | 0.590 |

*Results from comprehensive evaluation on LetheBench dataset using NDCG@10 metric*

## 🏗 Architecture

```
lethe/
├── packages/           # Monorepo packages
│   ├── core/          # Core context management
│   ├── search/        # Vector search implementation
│   ├── embeddings/    # Embedding models
│   └── cli/           # Command-line interface
├── ctx-run/           # Main NPX package
├── lethe-research/    # Research framework
└── test-env/          # Testing environment
```

## 🧪 Development

```bash
# Clone the repository
git clone https://github.com/sibyllinesoft/lethe.git
cd lethe

# Install dependencies
npm install

# Build all packages
npm run build

# Run tests
npm run test

# Start development mode
npm run dev
```

## 🔬 Research & Benchmarking

This project includes a comprehensive research framework:

```bash
# Run research benchmarks
cd lethe-research
python run_experiments.py

# Generate research artifacts
make reproduce-all
```

### LetheBench Dataset

- **100+ queries** evaluated across code, prose, and tool domains (scaling to 1000+ per domain)
- **7 baseline methods** comprehensively evaluated with statistical significance testing
- **Progressive system iterations** showing measurable improvements (baseline → iter4: +75% NDCG@10)
- **Rigorous statistical analysis** with effect sizes, confidence intervals, and hypothesis testing
- **Reproducible results** via automated evaluation pipeline

## 📖 Documentation

- [Setup Guide](docs/SETUP.md) - Detailed installation and configuration
- [API Reference](docs/API.md) - Complete API documentation
- [Research Paper](paper/) - Academic publication materials
- [Benchmarking](docs/BENCHMARKING.md) - Performance evaluation details

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with modern TypeScript and Node.js ecosystem
- Vector search powered by state-of-the-art embedding models
- Research methodology follows academic best practices
- Comprehensive benchmarking on diverse datasets

## 📧 Contact

Nathan Rice - [@your-handle](https://twitter.com/your-handle)

Project Link: [https://github.com/sibyllinesoft/lethe](https://github.com/sibyllinesoft/lethe)

---

**⚡ Supercharge your development workflow with intelligent context management**