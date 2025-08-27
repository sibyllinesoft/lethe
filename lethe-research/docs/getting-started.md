# Getting Started with Lethe Research

This guide will help you get up and running with the Lethe research framework in under 10 minutes.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (tested with 3.13)
- Git for version control
- 8GB+ RAM recommended
- Optional: Node.js 20+ for full system components

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd lethe-research

# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Test the unified analysis framework
python test_unified_analysis.py
```

Expected output:
```
✅ Successfully imported UnifiedAnalysisFramework
🧪 TESTING UNIFIED ANALYSIS FRAMEWORK
✅ Framework initialized successfully
✅ Loaded 500 rows of data
✅ Analysis completed - Plugins run: 3, Successful: 3
🎉 UNIFIED FRAMEWORK TEST COMPLETED SUCCESSFULLY
```

### 3. Run Your First Analysis

#### Option A: Use Existing Experimental Data
```bash
# Run analysis on existing artifacts
python src/analysis_unified.py --artifacts-dir artifacts/ --output-dir results/
```

#### Option B: Generate Synthetic Data
```bash
# Create synthetic test data and analyze
python -c "
from src.analysis_unified import UnifiedAnalysisFramework
import pandas as pd
import numpy as np

# Create synthetic data
np.random.seed(42)
data = []
for method in ['baseline_bm25', 'lethe_hybrid']:
    for _ in range(100):
        data.append({
            'method': method,
            'ndcg_10': np.random.normal(0.7 + (0.1 if 'lethe' in method else 0), 0.1),
            'latency_ms_total': np.random.normal(2000, 500),
            'memory_mb': np.random.normal(1000, 200)
        })

df = pd.DataFrame(data)
df.to_csv('sample_data.csv', index=False)

# Run analysis
framework = UnifiedAnalysisFramework()
framework.load_experimental_data('.')
results = framework.run_complete_analysis()
print('Analysis complete! Check analysis/ directory for outputs.')
"
```

## 📊 Understanding the Output

After running the analysis, you'll find:

```
analysis/
├── unified_analysis_results.json    # Raw analysis results
├── figures/                         # Generated visualizations
│   ├── progression_figure.pdf      # Performance progression
│   └── pareto_frontier.png         # Multi-objective analysis
└── tables/                         # LaTeX tables for publication
    ├── main_results_table.tex      # Primary results
    └── statistical_tests_summary.csv
```

## 🎯 Next Steps

### Explore the Analysis Framework
```bash
# Interactive exploration
python -c "
from src.analysis_unified import UnifiedAnalysisFramework
framework = UnifiedAnalysisFramework()
help(framework)  # See available methods
"
```

### Run Legacy Analysis (for comparison)
```bash
# Traditional fragmented approach
python scripts/enhanced_statistical_analysis.py
python scripts/pareto_analysis.py
python scripts/generate_figures.py
```

### Examine Real Experimental Data
```bash
# Browse available experiments
ls artifacts/*/
```

## 🔧 Configuration

### Basic Configuration
Create `config.json`:
```json
{
  "baseline_methods": ["baseline_bm25", "baseline_vector"],
  "lethe_iterations": ["lethe_iter_1", "lethe_iter_2", "lethe_iter_3"],
  "primary_metrics": ["ndcg_at_10", "recall_at_50", "latency_ms_total"],
  "significance_level": 0.05,
  "bootstrap_samples": 1000
}
```

Use with:
```bash
python src/analysis_unified.py --config config.json
```

### Advanced Configuration
```python
from src.analysis_unified import AnalysisConfig

config = AnalysisConfig(
    bootstrap_samples=10000,  # Higher precision
    significance_level=0.01,  # Stricter significance
    multiple_comparison_method='fdr',  # FDR correction
    hypotheses=['H1_quality_improvement', 'H2_efficiency_maintained']
)
```

## 🐛 Troubleshooting

### Common Issues

**Import Error**: `ModuleNotFoundError: No module named 'src.analysis_unified'`
```bash
# Make sure you're in the project root directory
cd lethe-research
python src/analysis_unified.py
```

**Missing Dependencies**: Install requirements
```bash
pip install -r requirements.txt
```

**No Data Found**: Check artifacts directory
```bash
ls artifacts/
# Should contain .csv or .json files with experimental data
```

**Memory Issues**: Reduce bootstrap samples
```python
config = AnalysisConfig(bootstrap_samples=100)  # Default is 1000
```

### Getting Help

1. **Check Documentation**: Browse `docs/` directory
2. **Run Tests**: `python test_unified_analysis.py`
3. **Validate Setup**: `python validate_infrastructure.py`
4. **Review Examples**: Check `examples/` directory

## 📈 Performance Tips

### For Large Datasets
- Use `bootstrap_samples=100` for faster analysis
- Process data in chunks if memory constrained
- Consider running analysis on subsets first

### For Production Use
- Set `bootstrap_samples=10000` for publication-quality confidence intervals
- Use `multiple_comparison_method='bonferroni'` for conservative significance testing
- Enable all hypothesis tests for comprehensive analysis

## 🎯 What's Next?

- **Explore Architecture**: Read [Architecture Guide](architecture.md)
- **Understand Experiments**: See [Experimental Design](experimental-design.md)
- **Dive into Code**: Check [API Reference](api-reference.md)
- **Contribute**: Read [Contributing Guidelines](contributing.md)

---

**Need help?** Check the [Troubleshooting Guide](troubleshooting.md) or open an issue.