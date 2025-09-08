# Comprehensive Retrieval Benchmarking System

A complete benchmarking infrastructure for comparing Lethe-Hybrid against 5 categories of open-source retrieval leaders with statistical rigor and marketing-ready reports.

## 🚀 One-Shot Execution

```bash
# Quick start (dry run for validation)
python run_benchmark.py --dry-run

# Full comprehensive benchmark 
python run_benchmark.py
```

## 📊 What Gets Benchmarked

### Competitor Categories (16+ Systems)
- **Hybrid Vector DBs**: Weaviate, Milvus, Vespa, OpenSearch
- **Learned Sparse/Late-Interaction**: SPLADE v2, ColBERT v2, RAGatouille
- **Open Rerankers**: BGE-reranker-large/v2-m3, MonoT5
- **Code Search & Graph**: Zoekt, livegrep, GraphRAG  
- **Long-Context Algorithms**: StreamingLLM, LongNet, BGE-M3

### Datasets (10+ Tasks)
- **InfiniteBench Core**: Zh.QA, Retrieve.PassKey/KV/Number, Code.Debug/QA, En.QA
- **External Stress**: RULER, LongBench-v2, BABILong

### Evaluation Protocol
- **Matched Budgets**: 8%, 15%, 30% keep_ratio across all systems
- **Statistical Rigor**: Bootstrap + permutation testing with Holm correction
- **Vendor-Fair Configs**: Each system's documented optimal settings
- **Complete Reproducibility**: JSONL logs, config snapshots, container versions

## 📋 Generated Outputs

### Marketing-Ready Reports
- **Interactive HTML Report**: Scenario cards, advantage maps, statistical analysis
- **CSV Summary**: All metrics for further analysis  
- **JSON Export**: Complete results with statistical comparisons
- **Raw Data Package**: JSONL logs, configurations, container manifests

### Key Report Sections
- **Scenario Cards**: "Multilingual QA @15% keep: Lethe-Hybrid beats BGE-M3 by +12% ΔCBU/1k at -45ms"
- **Advantage Map**: Interactive heatmap of competitive performance
- **Honest Assessment**: "When NOT to use Lethe" failure bucket analysis
- **Competitor Strengths**: What each system does best with documentation links
- **Statistical Analysis**: Bootstrap confidence intervals with significance testing

## 🏗️ Infrastructure

### Docker Orchestration
- **35+ Services**: All competitors in isolated containers with resource limits
- **Health Monitoring**: Automatic health checks and restart policies
- **Resource Management**: 8GB RAM, 4 CPU cores per competitor
- **Network Isolation**: Dedicated benchmark network with service discovery

### Quality Assurance
- **Configuration Validation**: Pre-flight checks for all requirements
- **Statistical Validation**: Multiple comparison correction and effect size thresholds
- **Result Verification**: Automated consistency checks and outlier detection
- **Error Recovery**: Graceful degradation and partial result preservation

## 📁 Project Structure

```
benchmarks/
├── __init__.py                 # Package exports
├── __main__.py                 # CLI entry point
├── config.py                   # Configuration management
├── orchestrator.py             # Main execution engine
├── evaluation.py               # Statistical evaluation engine
├── reporting.py                # Marketing-ready report generation
├── docker-compose.benchmark.yml # All competitor services
├── datasets/                   # Dataset loading system
│   ├── base.py                # Common dataset interfaces
│   ├── infinitebench.py       # InfiniteBench loaders
│   ├── ruler.py               # RULER dataset loader
│   ├── longbench.py           # LongBench-v2 loader
│   ├── babilong.py            # BABILong loader
│   └── registry.py            # Dataset registry
└── competitors/                # Competitor implementations
    ├── base.py                # Common competitor interfaces
    ├── registry.py            # Competitor registry
    ├── lethe_baseline.py      # Lethe-Hybrid implementation
    ├── hybrid_vector_db.py    # Vector database competitors
    ├── learned_sparse.py      # Sparse retrieval competitors
    ├── rerankers.py           # Reranking competitors
    ├── code_search.py         # Code search competitors
    ├── long_context.py        # Long-context competitors
    └── mock.py                # Mock implementations for testing
```

## 🚀 Quick Start Guide

### Prerequisites
- Docker & Docker Compose
- Python 3.9+ with pip
- 32GB+ RAM recommended (16GB minimum)
- 100GB+ free disk space

### Installation
```bash
# Clone repository
git clone https://github.com/lethe-research/lethe.git
cd lethe

# Install dependencies
pip install -r requirements-benchmark.txt

# Validate configuration
python run_benchmark.py --validate-only
```

### Data Preparation
```bash
# Download benchmark datasets
mkdir -p data/{infinitebench,ruler,longbench_v2,babilong}

# InfiniteBench
wget -O data/infinitebench/zh_qa.jsonl https://github.com/OpenBMB/InfiniteBench/releases/download/v1.0/zh_qa.jsonl

# RULER  
git clone https://github.com/NVIDIA/RULER data/ruler

# LongBench-v2
git clone https://github.com/THUDM/LongBench data/longbench_v2

# BABILong
git clone https://github.com/booydar/babilong data/babilong
```

### Execution Options

```bash
# Full benchmark (2-6 hours depending on hardware)
python run_benchmark.py

# Specific competitors only
python run_benchmark.py --competitors weaviate colbert_v2 lethe_hybrid

# Specific datasets only  
python run_benchmark.py --datasets infinitebench_zh_qa ruler

# Custom configuration
python run_benchmark.py --config my_benchmark_config.yaml

# Dry run for testing (5 minutes)
python run_benchmark.py --dry-run
```

## 🔬 Statistical Methodology

### Fair Evaluation Protocol
- **Budget Matching**: All systems get identical token budgets (8%, 15%, 30% of context)
- **Consistent Tokenization**: Whitespace-based tokenization for cross-system fairness
- **Vendor-Fair Configs**: Each system uses its documented optimal configuration
- **Resource Limits**: Identical hardware constraints (8GB RAM, 4 CPU cores)

### Statistical Rigor
- **Bootstrap Confidence Intervals**: 1000 iterations for effect size estimation
- **Permutation Testing**: 1000 iterations for significance testing
- **Multiple Comparison Correction**: Holm step-down method for family-wise error rate
- **Effect Size Thresholds**: Cohen's d > 0.1 for practical significance
- **Power Analysis**: Sufficient sample sizes for reliable statistical inference

### Metrics Computed
- **Retrieval Quality**: Precision@k, Recall@k, Exact Match Rate
- **Efficiency**: ΔCBU/1k (Context Budget Unit efficiency), Token utilization
- **Performance**: P95/P99 latency, Memory usage, CPU utilization  
- **Diversity**: Entity coverage, Semantic diversity scores
- **Reliability**: Success rate, Error rate, Timeout frequency

## 📊 Example Results

### Sample Scenario Card
```
🎯 Multilingual QA @15% keep
Best Open Source: BGE-M3 Reranker  
Lethe-Hybrid Result: +12% ΔCBU/1k at -45ms (p < 0.01)
Statistical: Cohen's d = 0.34, 95% CI [0.18, 0.51]
```

### Sample Advantage Map
```
              Lethe  Weaviate  ColBERT  BGE-M3  SPLADE
ZH QA           0      -0.12    +0.08   -0.23   -0.18
Code Debug      0      +0.05    -0.15   +0.11   -0.08  
Passkey         0      -0.31    +0.24   -0.16   +0.06
Multi-hop       0      +0.18    -0.09   +0.03   -0.21
```

## 🔧 Configuration

### Key Configuration Options
```yaml
# Competitor selection (empty = all)
enabled_competitors: ["lethe_hybrid", "weaviate", "colbert_v2"]

# Dataset selection (empty = all)  
enabled_datasets: ["infinitebench_zh_qa", "ruler"]

# Evaluation budgets
evaluation:
  keep_ratios: [0.08, 0.15, 0.30]
  statistical_testing:
    bootstrap_iterations: 1000
    confidence_level: 0.95
```

### Resource Configuration
```yaml
infrastructure:
  container_memory_limit: "8g"
  container_cpu_limit: "4.0" 
  max_parallel_containers: 4
  api_timeout_seconds: 300
```

## 🐛 Troubleshooting

### Common Issues
1. **Docker out of space**: `docker system prune -a`
2. **Port conflicts**: Check ports 8080-8094 are free
3. **Memory issues**: Reduce `max_parallel_containers` or increase system RAM
4. **Dataset missing**: Download from official sources (see Data Preparation)

### Debug Mode
```bash
# Enable debug logging
python run_benchmark.py --log-level DEBUG

# Preserve failed containers for inspection  
# Set preserve_failed_containers: true in config

# Check specific container logs
docker logs benchmark_weaviate
```

## 📄 License & Citation

```bibtex
@software{lethe_benchmark_2024,
  title = {Comprehensive Retrieval Benchmarking System},
  author = {Lethe Research Team},
  year = {2024},
  url = {https://github.com/lethe-research/lethe}
}
```

## 🤝 Contributing

1. **Add New Competitors**: Implement `BaseCompetitor` interface in `competitors/`
2. **Add New Datasets**: Implement `BaseDatasetLoader` interface in `datasets/`  
3. **Extend Metrics**: Add new metrics in `evaluation.py`
4. **Improve Reports**: Enhance templates in `reporting.py`

## 🆘 Support

- **Documentation**: See `docs/` directory for detailed API documentation
- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and community support
- **Email**: benchmark-support@lethe-research.org