# Unified Scientific Analysis Framework

## 🎯 Overview

The Unified Scientific Analysis Framework consolidates all fragmented analysis pipelines into a single, comprehensive system for scientific analysis and publication-ready output generation. This framework successfully replaces the following scattered scripts:

- `enhanced_statistical_analysis.py` → HypothesisTestingPlugin
- `pareto_analysis.py` → ParetoAnalysisPlugin  
- `final_analysis.py` → PublicationOutputPlugin
- `generate_tables.py` → Integrated table generation
- `generate_figures.py` → Integrated figure generation

## 🏗️ Architecture

### Core Framework Components

1. **UnifiedAnalysisFramework** - Main orchestration class
2. **Plugin System** - Extensible analysis modules
3. **Integrated Components** - Leverages existing analysis infrastructure
4. **Migration Support** - Backward compatibility with legacy scripts

### Plugin Architecture

```python
# Three default plugins handle all analysis needs
HypothesisTestingPlugin:    # Statistical significance testing
  - H1-H4 hypothesis evaluation
  - Pairwise method comparisons  
  - Multiple comparison corrections
  - Effect size calculations
  - Bootstrap confidence intervals

ParetoAnalysisPlugin:       # Multi-objective optimization
  - Pareto frontier identification
  - Domination analysis
  - Trade-off visualization
  - Objective correlation analysis

PublicationOutputPlugin:    # Publication-ready outputs
  - LaTeX table generation
  - Publication-quality figures
  - Summary statistics
  - Consolidated reporting
```

## 🔗 Integration with Existing Components

The framework seamlessly integrates with existing analysis infrastructure:

- **✅ MetricsCalculator** - Advanced metrics computation
- **✅ StatisticalComparator** - Enhanced statistical analysis
- **✅ DataManager** - Advanced data persistence 
- **✅ EvaluationFramework** - Comprehensive evaluation

## 🚀 Quick Start

### Basic Usage

```python
from src.analysis_unified import UnifiedAnalysisFramework

# Initialize framework
framework = UnifiedAnalysisFramework()

# Load experimental data
framework.load_experimental_data("artifacts/")

# Run complete analysis
results = framework.run_complete_analysis()

# Generate publication outputs
output_files = framework.generate_publication_outputs("paper/")
```

### Command Line Interface

```bash
# Run with default settings
python src/analysis_unified.py

# Custom configuration
python src/analysis_unified.py --artifacts-dir experiments/ --output-dir results/

# With configuration file
python src/analysis_unified.py --config analysis_config.json
```

## 📊 Features

### Comprehensive Statistical Analysis

- **Hypothesis Testing**: H1-H4 evaluation with rigorous statistical tests
- **Effect Sizes**: Cohen's d with interpretive guidelines
- **Multiple Comparisons**: Bonferroni, Holm, FDR corrections
- **Bootstrap CI**: Confidence intervals via bootstrap resampling
- **Non-parametric Tests**: Mann-Whitney U for robust comparisons

### Multi-Objective Optimization

- **Pareto Frontiers**: Identify optimal trade-off solutions
- **Domination Analysis**: Systematic method comparison
- **Objective Correlations**: Understanding metric relationships
- **Trade-off Visualization**: Publication-quality Pareto plots

### Publication-Ready Outputs

- **LaTeX Tables**: Professional statistical result formatting
- **High-Quality Figures**: Publication-standard visualizations
- **Summary Statistics**: Comprehensive method comparisons
- **Consolidated Reports**: Single-source analysis documentation

## 🔄 Migration from Legacy Scripts

### Automatic Migration Support

```python
# Migrate from existing fragmented scripts
migration_results = framework.migrate_from_legacy_scripts()

# Validate against legacy outputs  
validation = framework.validate_against_legacy_outputs("legacy_outputs/")
```

### Legacy Script Mapping

| Legacy Script | Unified Framework | Status |
|--------------|------------------|---------|
| `enhanced_statistical_analysis.py` | `HypothesisTestingPlugin` | ✅ Migrated |
| `pareto_analysis.py` | `ParetoAnalysisPlugin` | ✅ Migrated |
| `final_analysis.py` | `PublicationOutputPlugin` | ✅ Migrated |
| `generate_tables.py` | Integrated table generation | ✅ Migrated |
| `generate_figures.py` | Integrated figure generation | ✅ Migrated |

## 🎛️ Configuration

### Analysis Configuration

```python
config = AnalysisConfig(
    baseline_methods=['baseline_bm25', 'baseline_vector'],
    lethe_iterations=['lethe_iter_1', 'lethe_iter_2', 'lethe_iter_3'],
    primary_metrics=['ndcg_at_10', 'recall_at_50', 'latency_ms_total'],
    significance_level=0.05,
    bootstrap_samples=1000,
    multiple_comparison_method='bonferroni'
)
```

### Directory Structure

```
project/
├── artifacts/          # Experimental data (CSV, JSON, JSONL)
├── analysis/          # Analysis outputs and cache
├── paper/
│   ├── figures/      # Generated figures
│   └── tables/       # Generated tables
└── src/
    └── analysis_unified.py  # Main framework
```

## 📈 Example Output

### Statistical Analysis Results

```json
{
  "statistical_tests": [
    {
      "test_name": "Mann-Whitney U",
      "baseline_method": "baseline_bm25",
      "comparison_method": "lethe_iter_3", 
      "metric": "ndcg_at_10",
      "p_value": 0.0001,
      "significant": true,
      "effect_size": 0.8,
      "effect_size_interpretation": "Large effect"
    }
  ],
  "hypothesis_conclusions": {
    "H1_quality_improvement": {
      "supported": true,
      "confidence": "High",
      "evidence": "Strong statistical significance across all metrics"
    }
  }
}
```

### Pareto Analysis Results

```json
{
  "pareto_solutions": [
    {
      "method": "lethe_iter_3",
      "objectives": {
        "ndcg_at_10": 0.92,
        "latency_ms_total": 145,
        "memory_mb": 220
      },
      "is_pareto_optimal": true,
      "pareto_rank": 1
    }
  ]
}
```

## 🧪 Testing

Run the comprehensive test suite to validate integration:

```bash
python test_unified_analysis.py
```

Expected output:
```
✅ Framework initialized successfully
✅ Loaded 500 rows of data  
✅ Analysis completed - Plugins run: 3, Successful: 3
✅ Migration completed with 3 result sets
✅ Integration validation: All components available
🎉 UNIFIED FRAMEWORK TEST COMPLETED SUCCESSFULLY
```

## 📝 Benefits of Unification

### Before: Fragmented Analysis
- 8+ separate analysis scripts
- Duplicated logic and dependencies
- Inconsistent output formats
- Manual coordination required
- Difficult maintenance and updates

### After: Unified Framework  
- Single comprehensive system
- Consistent plugin architecture
- Standardized outputs
- Automated coordination
- Easy extension and maintenance
- Backward compatibility maintained

## 🔧 Extending the Framework

### Creating Custom Plugins

```python
class CustomAnalysisPlugin(AnalysisPlugin):
    def name(self) -> str:
        return "custom_analysis"
    
    def run_analysis(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        # Custom analysis logic
        return {"custom_results": analysis_results}
    
    def generate_outputs(self, results: Dict[str, Any], output_dir: Path) -> List[str]:
        # Generate custom outputs
        return [output_file_paths]

# Register with framework
framework.register_plugin(CustomAnalysisPlugin())
```

## 📋 Requirements

- Python 3.8+
- pandas, numpy, scipy
- matplotlib, seaborn
- scikit-learn
- Existing analysis components (MetricsCalculator, etc.)

## 🎯 Next Steps

The unified framework is production-ready and can immediately replace all fragmented analysis scripts. Key advantages:

1. **Consistency**: Standardized analysis across all research
2. **Maintainability**: Single codebase vs. 8+ scattered scripts  
3. **Extensibility**: Easy to add new analysis methods
4. **Quality**: Comprehensive testing and validation
5. **Migration**: Seamless transition from legacy scripts

The fragmented scientific analysis section has been successfully unified into a robust, extensible framework that maintains all existing functionality while providing significant architectural improvements.