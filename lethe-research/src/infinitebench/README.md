# InfiniteBench Integration for Lethe

A comprehensive evaluation framework integrating the InfiniteBench dataset with the Lethe retrieval system for academic-quality long-context evaluation.

## 🎯 Overview

This framework provides production-ready integration between Lethe and the InfiniteBench dataset, enabling:

- **Academic Credibility**: Replace internal smoke tests with recognized benchmark results
- **Comprehensive Evaluation**: All 12 InfiniteBench tasks with 200K+ token contexts
- **Statistical Rigor**: BCa bootstrap confidence intervals and significance testing
- **Publication-Ready Reports**: Automated generation of academic-quality analysis
- **Baseline Comparisons**: BM25, naive chunking, dense retrieval, and GPT-4 baselines

## 📊 Dataset Coverage

### Supported Tasks (12 total)

| Task | Type | Avg Length | Samples | Metric | Description |
|------|------|------------|---------|--------|-------------|
| `passkey` | Retrieval | 122.9K | 590 | Accuracy | Passkey retrieval in long context |
| `number_string` | Retrieval | 122.9K | 590 | Accuracy | Number string locating task |
| `kv_retrieval` | Retrieval | 89.0K | 500 | Accuracy | Key-value retrieval from JSON |
| `longbook_sum_eng` | Novel | 171.5K | 103 | ROUGE-L | Long book summarization (English) |
| `longbook_choice_eng` | Novel | 171.5K | 229 | Accuracy | Long book multiple choice (English) |
| `longbook_qa_eng` | Novel | 171.5K | 351 | F1 Score | Long book Q&A (English) |
| `longbook_qa_chn` | Novel | 171.5K | 189 | F1 Score | Long book Q&A (Chinese) |
| `longdialogue_qa_eng` | Dialogue | 110.0K | 200 | F1 Score | Long dialogue Q&A (English) |
| `code_debug` | Code | 114.2K | 394 | Accuracy | Code debugging task |
| `code_run` | Code | 75.0K | 400 | Accuracy | Code execution task |
| `math_calc` | Math | 190.0K | 400 | Accuracy | Mathematical calculation |
| `math_find` | Math | 190.0K | 400 | Accuracy | Mathematical finding task |

### Evaluation Metrics

- **Exact Match (EM)**: Perfect string matching after normalization
- **F1 Score**: Token-level F1 with precision and recall
- **ROUGE-L**: Longest common subsequence for summarization
- **nDCG@k**: Normalized discounted cumulative gain for ranking
- **Accuracy**: Classification accuracy with normalization

## 🚀 Quick Start

### 1. Installation

Ensure you have the required dependencies:

```bash
pip install tiktoken rouge-score scikit-learn numpy pandas matplotlib seaborn
pip install sentence-transformers  # For dense retrieval baseline
```

### 2. Dataset Setup

Download the InfiniteBench dataset:

```bash
cd lethe-research
git clone https://huggingface.co/datasets/OpenBMB/InfiniteBench benchmarks/infinitebench
cd benchmarks/infinitebench
bash scripts/download_dataset.sh
```

### 3. Quick Test

Run a validation test to ensure everything works:

```bash
cd src/infinitebench
python run_evaluation.py --quick-test
```

### 4. Full Evaluation

Run the complete evaluation:

```bash
python run_evaluation.py --config config.yaml
```

### 5. Custom Evaluation

Run specific tasks only:

```bash
python run_evaluation.py --tasks passkey,kv_retrieval,longbook_qa_eng
```

## 📋 Configuration

The evaluation is configured via `config.yaml`. Key sections:

### Task Selection
```yaml
evaluation:
  tasks:
    - "passkey"
    - "kv_retrieval" 
    - "longbook_qa_eng"
    # ... more tasks
```

### Method Configuration
```yaml
evaluation:
  methods:
    - "bm25"              # BM25 baseline
    - "naive_chunking"    # Naive chunking
    - "dense_retrieval"   # Dense retrieval
    - "lethe"             # Your Lethe system
```

### Statistical Parameters
```yaml
evaluation:
  bootstrap_samples: 1000    # For confidence intervals
  confidence_level: 0.95     # 95% confidence intervals
  parallel_jobs: 4           # Parallel processing
```

## 🏗️ Architecture

### Core Components

```
src/infinitebench/
├── __init__.py              # Package exports
├── dataset_loader.py        # Dataset loading and preprocessing
├── metrics.py              # Evaluation metrics implementation
├── baselines.py            # Baseline method implementations
├── evaluation_pipeline.py  # Main evaluation orchestration
├── statistical_analysis.py # Statistical analysis and reporting
├── run_evaluation.py       # Main runner script
├── config.yaml            # Configuration file
└── README.md              # This documentation
```

### Data Flow

```
InfiniteBench Dataset → DataLoader → Evaluation Pipeline → Statistical Analysis → Publication Report
                                          ↑
                            Baselines + Lethe System
```

### Integration Points

- **Dataset Loading**: `InfiniteBenchLoader` handles all 12 task types
- **Evaluation Pipeline**: `InfiniteBenchEvaluator` orchestrates experiments
- **Statistical Analysis**: Integrates with existing BCa bootstrap framework
- **Metrics**: Task-specific metric selection and computation

## 📊 Output Structure

After running evaluation, you'll find:

```
artifacts/infinitebench_results/
├── full_evaluation_YYYYMMDD_HHMMSS/
│   ├── experiment_config.json
│   ├── results.json
│   ├── statistical_analysis.json
│   ├── reports/
│   │   ├── publication_report.md
│   │   ├── performance_by_task.csv
│   │   └── significance_matrix.csv
│   └── plots/
│       ├── performance_comparison.png
│       ├── task_breakdown.png
│       └── confidence_intervals.png
```

### Key Output Files

- **`publication_report.md`**: Camera-ready academic report
- **`performance_by_task.csv`**: Detailed performance breakdown
- **`significance_matrix.csv`**: Statistical significance testing
- **`statistical_analysis.json`**: Raw statistical analysis data

## 🔬 Statistical Analysis

### Bootstrap Confidence Intervals

The framework uses BCa (Bias-Corrected and Accelerated) bootstrap to compute robust confidence intervals:

```python
# 95% confidence intervals for all metrics
# Bias correction for small sample sizes
# Acceleration correction for skewed distributions
```

### Significance Testing

Pairwise comparisons between methods:

- **Welch's t-test**: For normally distributed metrics
- **Mann-Whitney U**: For non-parametric comparisons
- **Bootstrap difference**: For robust difference testing
- **Effect size**: Cohen's d for practical significance

### Multiple Comparison Correction

- **Bonferroni correction**: Conservative family-wise error rate control
- **False Discovery Rate**: More powerful multiple testing correction

## 🎯 Baseline Methods

### BM25 Baseline
```python
# Classical sparse retrieval
BM25Baseline(k1=1.2, b=0.75, top_k=5)
```

### Naive Chunking
```python
# Simple chunking strategies
NaiveChunkingBaseline(chunk_size=1024, strategy="uniform", top_k=5)
```

### Dense Retrieval
```python  
# Sentence transformer embeddings
DenseRetrievalBaseline(model_name="all-MiniLM-L6-v2", top_k=5)
```

### GPT-4 Baseline (Optional)
```python
# Direct GPT-4 comparison (expensive)
GPT4Baseline(model="gpt-4", temperature=0.0)
```

## 🔧 Advanced Usage

### Custom Method Integration

To integrate your own retrieval method:

```python
class YourMethod:
    def retrieve(self, query: str, context: str, top_k: int = 5) -> List[str]:
        # Your retrieval implementation
        return retrieved_chunks
    
    def generate_answer(self, query: str, chunks: List[str]) -> str:
        # Your answer generation
        return answer

# Register with evaluator
evaluator.register_method("your_method", YourMethod())
```

### Custom Metrics

Add task-specific metrics:

```python
class CustomMetric:
    def calculate(self, predictions: List[str], references: List[str]) -> float:
        # Your metric implementation
        return score

metrics.register_metric("custom_metric", CustomMetric())
```

### Performance Tuning

For large-scale evaluation:

```yaml
performance:
  max_memory_gb: 32         # Increase memory limit
  batch_size: 16            # Larger batches
  parallel_jobs: 8          # More parallelism
  timeout_seconds: 600      # Longer timeouts
```

## 📈 Publication Support

### Academic Report Generation

The framework generates publication-ready reports including:

- **Performance tables**: LaTeX-formatted results tables
- **Statistical significance**: Comprehensive significance testing
- **Effect size analysis**: Practical significance assessment
- **Confidence intervals**: Robust uncertainty quantification

### Citation Information

```bibtex
@dataset{infinitebench2024,
  title={InfiniteBench: Extending Long Context Evaluation Beyond 100K Tokens},
  author={Zhang, Xinrong and Chen, Yingfa and Hu, Shengding and others},
  year={2024},
  publisher={OpenBMB}
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Dataset Download Timeout**
   ```bash
   # Manual download if timeout occurs
   cd benchmarks/infinitebench
   wget -c [URL] -O data/[filename].jsonl
   ```

2. **Memory Issues**
   ```yaml
   # Reduce batch size in config.yaml
   performance:
     batch_size: 4
     max_memory_gb: 8
   ```

3. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Debugging

Enable debug logging:

```bash
python run_evaluation.py --verbose
```

Check logs:
```bash
tail -f evaluation.log
```

## 🚨 Important Notes

### Performance Considerations

- **Memory Usage**: Some tasks require significant memory (200K+ tokens)
- **Computation Time**: Full evaluation can take several hours
- **GPT-4 Costs**: GPT-4 baseline is expensive, use judiciously
- **Parallel Processing**: Adjust `parallel_jobs` based on your system

### Quality Assurance

- **Reproducibility**: Fixed random seeds for consistent results
- **Validation**: Automatic output format validation
- **Error Handling**: Comprehensive error handling and recovery
- **Intermediate Saving**: Results saved incrementally

## 📚 References

- [InfiniteBench Paper](https://arxiv.org/abs/2402.13718)
- [OpenBMB/InfiniteBench Repository](https://github.com/OpenBMB/InfiniteBench)
- [Lethe Documentation](../../../README.md)

## 🤝 Contributing

To contribute improvements:

1. Add new baseline methods to `baselines.py`
2. Implement additional metrics in `metrics.py`
3. Enhance statistical analysis in `statistical_analysis.py`
4. Update this documentation

## 📄 License

This integration follows the same license as the Lethe project. The InfiniteBench dataset has its own license terms.