# Lethe Research Scripts Directory

This directory contains evaluation scripts for the Lethe research project, including both the **legacy pipeline scripts** and the **new unified analysis framework**.

## ✨ New: Unified Analysis Framework

**Recommended approach:** Use the unified analysis framework instead of fragmented legacy scripts:

```bash
# Run complete analysis with unified framework
python ../src/analysis_unified.py --artifacts-dir ../artifacts/ --output-dir ../results/

# Or use interactively
python -c "from src.analysis_unified import UnifiedAnalysisFramework; 
framework = UnifiedAnalysisFramework(); 
framework.load_experimental_data('../artifacts/'); 
results = framework.run_complete_analysis()"
```

**Benefits of Unified Framework:**
- Single coherent analysis pipeline (replaces 8+ fragmented scripts)
- Plugin architecture for extensibility
- Enhanced statistical testing with proper corrections
- Automatic LaTeX table and figure generation
- Better error handling and progress reporting

## 🔄 Legacy Pipeline (For Reference)

## 🚀 One-Command Execution

To run the complete evaluation pipeline:

```bash
./run_full_evaluation.sh
```

This will execute the entire research pipeline from dataset creation to paper generation.

## 📊 Pipeline Overview

The evaluation pipeline consists of the following stages:

1. **Dataset Creation** (`create_lethebench.py`) - Creates LetheBench evaluation dataset
2. **Grid Search** (`run_grid_search.py`) - Executes Lethe parameter optimization  
3. **Baseline Evaluation** (`baseline_implementations.py`) - Evaluates 7 competitive baselines
4. **Statistical Analysis** (`run_analysis.py`) - Comprehensive hypothesis testing
5. **Visualization** (`generate_figures.py`) - Publication-quality figures
6. **Paper Generation** (`generate_paper.py`) - LaTeX paper with results
7. **Validation** (`validate_results.py`) - Results quality assurance

## 🔧 Individual Script Usage

### Dataset Creation
```bash
./create_dataset.sh [OUTPUT_PATH] [EXAMPLES_DIR] [CONFIG_PATH]
```

### Grid Search
```bash
./run_grid_search.sh [DATASET_PATH] [OUTPUT_DIR] [CONFIG_PATH] [MAX_PARALLEL]
```

### Baseline Evaluation  
```bash
./evaluate_baselines.sh [DATASET_PATH] [OUTPUT_DIR] [CONFIG_PATH] [MAX_PARALLEL]
```

## 🎯 Key Features

- **Statistical Rigor**: Bootstrap confidence intervals, Holm-Bonferroni correction
- **Fraud-Proofing**: Placebo tests, query shuffling, random embeddings validation
- **Publication Ready**: LaTeX paper generation with automated results integration
- **Reproducible**: Complete environment capture and validation framework
- **Scalable**: Parallel execution with configurable worker limits

## 📁 Output Structure

The pipeline generates a complete research artifact directory:

```
artifacts/
├── datasets/           # LetheBench evaluation dataset
├── baselines/          # Baseline system results
├── lethe_runs/         # Lethe grid search results
├── analysis/           # Statistical analysis outputs
├── figures/            # Publication-quality visualizations
├── paper/              # LaTeX paper and compilation
└── logs/               # Execution logs and debugging
```

## ⚙️ Configuration

Primary configuration file: `../experiments/grid_config.yaml`

Key configuration sections:
- **Grid Search Parameters**: 9 key hyperparameters with factorial design
- **Baseline Systems**: 7 competitive retrieval approaches  
- **Evaluation Metrics**: Quality, efficiency, coverage, consistency
- **Statistical Testing**: Hypothesis framework (H1-H4) with effect sizes
- **Fraud-Proofing**: 13 validation checks for result reliability

## 🔬 Hypothesis Testing Framework

The pipeline tests 4 core hypotheses:

- **H1 (Quality)**: Lethe achieves superior retrieval quality
- **H2 (Efficiency)**: Lethe maintains acceptable computational efficiency
- **H3 (Robustness)**: Lethe demonstrates consistent cross-domain performance
- **H4 (Adaptivity)**: Lethe adapts effectively to different query contexts

## 📈 Expected Runtime

- **Full Pipeline**: 4-8 hours (depending on grid size and parallelism)
- **Dataset Creation**: 10-30 minutes
- **Grid Search**: 2-5 hours  
- **Baseline Evaluation**: 1-2 hours
- **Analysis & Visualization**: 15-30 minutes
- **Paper Generation**: 5-10 minutes

## 🛡️ Quality Assurance

The pipeline includes comprehensive validation:

- **Data Integrity**: Format validation and completeness checks
- **Statistical Validation**: Significance testing and effect size verification
- **Publication Readiness**: Automated assessment with scoring rubric
- **Reproducibility**: Environment capture and replication validation

## 📋 Requirements

### System Requirements
- **OS**: Linux/macOS with Bash
- **Memory**: 8GB+ recommended
- **Storage**: 5GB+ free space
- **CPU**: Multi-core recommended for parallelism

### Software Requirements
- **Python 3.8+** with packages: numpy, pandas, scipy, scikit-learn, matplotlib, seaborn
- **Node.js 16+** (for ctx-run dependency)
- **Git** (for environment tracking)
- **LaTeX** (optional, for PDF compilation)

### Verification
Run the setup validation:
```bash
python validate_setup.py
```

## 🚨 Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `MAX_PARALLEL` or `grid_search.parameters` scope
2. **Timeout Errors**: Increase timeout values in configuration
3. **Missing Dependencies**: Run setup validation and install missing packages
4. **Disk Space**: Ensure 5GB+ available, enable compression if needed

### Debug Mode
```bash
LOG_LEVEL=DEBUG ./run_full_evaluation.sh
```

### Manual Recovery
If pipeline fails, individual stages can be run separately:
```bash
# Resume from specific stage
SKIP_DATASET=true SKIP_BASELINES=true ./run_full_evaluation.sh
```

## 🔄 Migration from Legacy Scripts

The unified analysis framework replaces these legacy scripts:

| Legacy Script | Unified Framework Component | Migration Status |
|--------------|----------------------------|------------------|
| `enhanced_statistical_analysis.py` | `HypothesisTestingPlugin` | ✅ Fully migrated |
| `pareto_analysis.py` | `ParetoAnalysisPlugin` | ✅ Fully migrated |
| `final_analysis.py` | `PublicationOutputPlugin` | ✅ Fully migrated |
| `generate_figures.py` | Integrated figure generation | ✅ Enhanced |
| `generate_tables.py` | Integrated table generation | ✅ Enhanced |
| `run_analysis.py` | Main framework orchestrator | ✅ Replaced |

### Migration Helper Commands

```bash
# Validate unified framework against legacy outputs
python -c "
from src.analysis_unified import UnifiedAnalysisFramework
framework = UnifiedAnalysisFramework()
framework.migrate_from_legacy_scripts()
framework.validate_against_legacy_outputs('legacy_outputs/')
"

# Test unified framework
python ../test_unified_analysis.py
```

## 📞 Support

For issues or questions:

1. **New Unified Framework**: See `../src/README.md` and `../docs/getting-started.md`
2. **Legacy Pipeline Issues**: Check troubleshooting section above
3. **Migration Questions**: Review execution logs in `artifacts/*/logs/`
4. **Validation**: Run `python validate_results.py --results-dir artifacts/latest/`
5. **Full Documentation**: Consult `../README.md`

---

**Generated by the Lethe Research Framework** 🚀