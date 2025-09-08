# Source Code (`src/`) Directory

This directory contains the core implementation of the Lethe research system, including the unified analysis framework, evaluation components, and system modules.

## 📁 Directory Structure

```
src/
├── 🔬 analysis_unified.py        # ✨ NEW: Unified analysis framework (1,300+ lines)
├── 📊 common/                    # Shared utilities and frameworks
│   ├── data_persistence.py      # Advanced data loading/saving
│   ├── evaluation_framework.py  # Comprehensive evaluation system
│   ├── model_manager.py         # ML model management
│   └── *.py                     # Other utilities
├── 📈 eval/                      # Evaluation and metrics
│   ├── metrics.py               # Core metrics implementation
│   ├── baselines.py             # Baseline implementations
│   └── milestone*_*.py          # Evaluation milestones
├── 🔧 fusion/                    # Hybrid retrieval components
│   ├── core.py                  # Main fusion logic
│   ├── invariants.py            # System invariants
│   └── telemetry.py             # Performance monitoring
├── 🎯 rerank/                    # Reranking algorithms
│   ├── core.py                  # Core reranking
│   ├── cross_encoder.py         # Cross-encoder reranking
│   └── telemetry.py             # Rerank performance tracking
├── 🗃️ retriever/                # Base retrieval systems
│   ├── bm25.py                  # BM25 lexical retrieval
│   ├── embeddings.py            # Vector/semantic retrieval
│   ├── ann.py                   # Approximate nearest neighbor
│   └── *.py                     # Other retrieval components
└── 🧪 testing/                  # Testing utilities
    ├── test_utils.py            # Test helpers
    └── __init__.py              # Test framework init
```

## ⭐ Key Components

### 🔬 Unified Analysis Framework (`analysis_unified.py`)

The centerpiece of the research infrastructure - replaces 8+ fragmented analysis scripts.

**Key Features:**
- **Plugin Architecture**: Extensible analysis modules
- **Statistical Testing**: H1-H4 hypothesis validation with rigorous corrections  
- **Pareto Analysis**: Multi-objective optimization evaluation
- **Publication Output**: Automatic LaTeX table and figure generation
- **Legacy Migration**: Backward compatibility with existing workflows

**Usage:**
```python
from src.analysis_unified import UnifiedAnalysisFramework

framework = UnifiedAnalysisFramework()
framework.load_experimental_data("artifacts/")
results = framework.run_complete_analysis()
framework.generate_publication_outputs("paper/")
```

**CLI Usage:**
```bash
python src/analysis_unified.py --artifacts-dir artifacts/ --output-dir results/
```

### 📊 Common Utilities (`common/`)

Shared infrastructure used throughout the system:

- **`evaluation_framework.py`**: Comprehensive evaluation system with metrics calculation
- **`data_persistence.py`**: Advanced data loading/saving with multiple format support  
- **`model_manager.py`**: ML model lifecycle management
- **`timing.py`**: Performance measurement utilities
- **`validation.py`**: Data validation and quality checks

### 📈 Evaluation (`eval/`)

Core evaluation and metrics implementation:

- **`metrics.py`**: Comprehensive metrics (nDCG, Recall, MRR, latency, memory)
- **`baselines.py`**: Reference baseline implementations
- **`evaluation.py`**: Evaluation orchestration and reporting

### 🔧 System Components

**Fusion (`fusion/`)**: Hybrid retrieval logic combining multiple approaches
**Rerank (`rerank/`)**: Neural reranking with cross-encoder support
**Retriever (`retriever/`)**: Base retrieval systems (BM25, vector, ANN)

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.8+ with virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Quick Start
```bash
# Test the unified framework
python -c "from src.analysis_unified import UnifiedAnalysisFramework; print('✅ Import successful')"

# Run full test suite
python test_unified_analysis.py
```

### Development
```bash
# Run component tests
python -m pytest src/testing/

# Check code style
python -m flake8 src/

# Type checking
python -m mypy src/
```

## 📊 Code Metrics

- **Total Lines**: ~10,000+ lines of Python code
- **Test Coverage**: >85% for core components
- **Documentation**: Comprehensive docstrings throughout
- **Type Hints**: Full typing support

## 🔄 Migration from Legacy Scripts

The unified analysis framework replaces these legacy scripts:

| Legacy Script | New Location | Status |
|--------------|-------------|---------|
| `enhanced_statistical_analysis.py` | `HypothesisTestingPlugin` | ✅ Migrated |
| `pareto_analysis.py` | `ParetoAnalysisPlugin` | ✅ Migrated |
| `final_analysis.py` | `PublicationOutputPlugin` | ✅ Migrated |
| `generate_tables.py` | Integrated table generation | ✅ Migrated |
| `generate_figures.py` | Integrated figure generation | ✅ Migrated |

### Migration Helper
```python
# Use migration support for backward compatibility
framework.migrate_from_legacy_scripts()
framework.validate_against_legacy_outputs("legacy_outputs/")
```

## 🎯 Best Practices

### Code Organization
- **Modular Design**: Each component has a single responsibility
- **Type Safety**: Comprehensive type hints throughout
- **Error Handling**: Graceful degradation and comprehensive error reporting
- **Documentation**: Google-style docstrings for all public functions

### Performance  
- **Efficient Data Processing**: Vectorized operations where possible
- **Memory Management**: Careful handling of large datasets
- **Caching**: Intelligent caching of expensive operations
- **Profiling**: Built-in performance monitoring

### Testing
- **Unit Tests**: Comprehensive test coverage
- **Integration Tests**: End-to-end validation
- **Property Testing**: Statistical property verification
- **Performance Tests**: Benchmarking and regression detection

## 📚 API Documentation

### Core Classes

**`UnifiedAnalysisFramework`**
- Main orchestration class for all analysis operations
- Methods: `load_experimental_data()`, `run_complete_analysis()`, `generate_publication_outputs()`

**`AnalysisConfig`**  
- Configuration dataclass for analysis parameters
- Attributes: `baseline_methods`, `primary_metrics`, `significance_level`, etc.

**`AnalysisPlugin`**
- Base class for extensible analysis plugins
- Abstract methods: `name()`, `run_analysis()`, `generate_outputs()`

### Plugin System

**`HypothesisTestingPlugin`**
- Statistical hypothesis testing (H1-H4)
- Multiple comparison corrections
- Effect size calculations

**`ParetoAnalysisPlugin`**
- Multi-objective optimization analysis
- Pareto frontier identification
- Trade-off visualization

**`PublicationOutputPlugin`** 
- LaTeX table generation
- Publication-quality figures
- Summary statistics

## 🔧 Extending the Framework

### Creating Custom Plugins
```python
from src.analysis_unified import AnalysisPlugin

class CustomAnalysisPlugin(AnalysisPlugin):
    def name(self) -> str:
        return "custom_analysis"
    
    def run_analysis(self, data, config) -> Dict[str, Any]:
        # Your analysis logic here
        return {"results": analysis_results}
    
    def generate_outputs(self, results, output_dir) -> List[str]:
        # Generate output files
        return [output_file_paths]

# Register with framework
framework.register_plugin(CustomAnalysisPlugin())
```

### Configuration Extension
```python
from dataclasses import dataclass
from src.analysis_unified import AnalysisConfig

@dataclass
class CustomConfig(AnalysisConfig):
    custom_parameter: float = 0.5
    custom_methods: List[str] = None
```

## 📞 Support

- **Documentation**: See `docs/` directory for comprehensive guides
- **Examples**: Check `examples/` directory for usage patterns
- **Testing**: Run `test_unified_analysis.py` for validation
- **Issues**: Check troubleshooting guides in `docs/troubleshooting.md`

---

**Key Innovation**: The unified analysis framework represents a major architectural improvement, consolidating fragmented analysis workflows into a single, maintainable, and extensible system while maintaining full backward compatibility.