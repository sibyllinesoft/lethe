# Selection Stack Diagnostics System

## Overview

The Selection Stack Diagnostics system provides fast, targeted diagnosis of Lethe's retrieval selection pipeline (S0→S1→S2→CBU) to identify exact failure points when coverage metrics are 0.0%. The system uses 4 specialized probes to systematically validate each layer of the pipeline.

## Architecture

```
S0: Input Processing → S1: Dense Retrieval → S2: Cross-Encoder Reranking → CBU: Coverage-Based Utility
     │                     │                      │                            │
     └── Probe 1 ──────────┘                      └── Probe 3 ─────────────────┘
                             └── Probe 2 ─────────┘                            │
                                                                               └── Probe 4
```

## 4 Fast Probes

### Probe 1: S1 Query Vector Sanity Check
**Target**: Query embeddings not constant/broken
- **Tests**: 200 query embeddings for variance, norms, self-similarity
- **Expected**: per-dim std ~0.1–0.4, cosine self-sim ~1.0, avg cosine to random atoms ~0.0±0.05
- **Red Flags**: all-zero/NaN, identical hashes, cosines ~0 everywhere
- **Indicates**: Wrong encoder weights or wrong input field

### Probe 2: S1 Index/Space Audit  
**Target**: Index search retrieving relevant items
- **Tests**: 50 queries with top-5 retrieval results
- **Expected**: max similarity ≥0.25 for code/debug tasks
- **Red Flags**: all similarities ~0.0, empty results, model hash mismatch
- **Indicates**: Encoder/index mismatch or wrong retrieval configuration

### Probe 3: S2 Cross-Encoder Pair Feeding
**Target**: Cross-encoder receiving proper query+candidate pairs
- **Tests**: 20 CE inputs with verbatim logging
- **Expected**: score spread (std>0.1) and correlation with lexical overlap
- **Red Flags**: constant scores (~0.5 or 0.0), empty/identical pairs
- **Indicates**: CE tokenizer issues, wrong input format, or broken model

### Probe 4: Coverage Features Validation
**Target**: Entity/symbol extraction working for CBU
- **Tests**: Selected atoms for entities, symbols, file IDs
- **Expected**: medians >0 for code datasets (dozens typical)
- **Red Flags**: all zeros in feature counts
- **Indicates**: Feature extraction disabled or not run during indexing

## Diagnostic Output Format

Each probe generates structured output in the required format:

```csv
dataset,sample_id,keep,K1,K2,dims,top5_sim,top5_ids∩gold_ids,span_hit(bool),symbol_hit(bool),CE_score_max,entities_count_median,symbols_count_median
infinitebench,sample_0,0.30,4000,1000,768,0.800,2,True,True,0.629,2,1
```

## Usage

### Command Line Interface

```bash
# Quick test with mock data
python scripts/run_selection_stack_diagnostics.py --quick-test

# Run on specific evaluation data  
python scripts/run_selection_stack_diagnostics.py --data evaluation_data.json --output results/

# Run on InfiniteBench Code.Debug dataset
python scripts/run_selection_stack_diagnostics.py --dataset infinitebench --task code_debug

# Generate diagnostic format output
python scripts/run_selection_stack_diagnostics.py --quick-test --format diagnostic
```

### Programmatic Usage

```python
from src.diagnostics import SelectionStackDiagnostics
from src.common.evaluation_framework import EvaluationFramework

# Initialize system
config = {...}  # Configuration dict
framework = EvaluationFramework()
diagnostics = SelectionStackDiagnostics(config, framework)

# Run diagnosis
result = await diagnostics.diagnose_stack(
    evaluation_data=data,
    retrieval_pipeline=pipeline,
    output_dir=Path("results/")
)

# Check results
if result.overall_status == 'failed':
    print(f"Failure in layer: {result.failure_layer}")
    print(f"Recommended fixes: {result.recommended_fixes}")
```

## Configuration

```yaml
sample_sizes:
  query_vectors: 200      # Query embeddings to analyze
  index_items: 50         # Retrieval results to test  
  ce_pairs: 20           # Cross-encoder pairs to test
  coverage_atoms: 100    # Atoms to check for features

thresholds:
  embedding_std_min: 0.1      # Minimum per-dimension variance
  embedding_std_max: 0.4      # Maximum per-dimension variance  
  max_similarity_min: 0.25    # Minimum acceptable max similarity
  ce_score_std_min: 0.1       # Minimum CE score variance
  entity_count_min: 1         # Minimum entities per atom
  symbol_count_min: 1         # Minimum symbols per atom

controlled_parameters:
  K1_candidates: [2000, 4000]   # Dense retrieval candidates
  K2_candidates: [600, 1000]    # Reranking candidates
  dims_candidates: [256, 768]   # Embedding dimensions to test
  diversity_delta: 0            # Disable DPP temporarily
  facility_gamma: 0.8           # Emphasize facility-location

success_criteria:
  span_coverage_target: 0.15    # 15% minimum span coverage
  symbol_coverage_target: 0.10  # 10% minimum symbol coverage  
  keep_ratio_target: 0.30       # 30% keep ratio test
```

## Interpreting Results

### Status Levels
- **PASS**: All probes successful, pipeline healthy
- **WARNING**: Minor issues detected, pipeline degraded but functional
- **FAILED**: Critical issues found, specific layer identified

### Common Failure Patterns

#### S1 Vector Issues
- **Zero/constant embeddings**: Wrong encoder loaded or input field incorrect
- **Poor normalization**: Embeddings not L2-normalized  
- **Hash mismatch**: Query encoder differs from index encoder

#### S1 Index Issues  
- **Low similarities**: Encoder/index mismatch, wrong model weights
- **Empty results**: Retrieval configuration broken, K1 filtering too aggressive
- **No relevant items**: Index doesn't contain relevant documents for queries

#### S2 Cross-Encoder Issues
- **Constant scores**: Broken tokenizer, wrong input format, or model weights
- **No score variance**: All inputs identical, poor model sensitivity
- **Empty pairs**: Query or candidate extraction failing

#### CBU Feature Issues
- **Zero entity counts**: Entity extraction disabled or failing
- **Zero symbol counts**: Code symbol extraction not run during indexing
- **Missing file IDs**: File mapping not preserved during atom creation

## Minimal Surgical Fixes

Based on probe results, the system recommends targeted fixes:

1. **Input Field Correctness**: Enforce `sample.question` everywhere
2. **Encoder/Index Alignment**: Stamp `encoder_hash` in both index and runtime
3. **Normalization Mode**: Make both unit-norm consistently  
4. **CE Wiring**: Assert non-empty pair texts, ensure exact atom text
5. **Entity/Symbol Extraction**: Turn on code symbol extractor

## Integration with Evaluation Pipeline

The diagnostics integrate seamlessly with existing evaluation infrastructure:

- **Data Loading**: Works with InfiniteBench loader and custom datasets
- **Pipeline Compatibility**: Supports any retrieval pipeline with standard interfaces
- **Output Integration**: Generates files compatible with existing analysis tools
- **Performance Monitoring**: Tracks execution time and resource usage

## Success Criteria

The system validates that:
- **SpanCoverage ≥ 10–20%** and **SymbolCoverage ≥ 10%** on Code.Debug at **30% keep**
- Anything >0 at 15% keep confirms the pipeline is functional
- Query embeddings have proper variance and normalization
- Cross-encoder produces meaningful score distributions
- Coverage features are extracted and available for selection

## Files Created

- `src/diagnostics/selection_stack_diagnostics.py` - Main coordinator
- `src/diagnostics/probe_query_vectors.py` - S1 query vector analysis
- `src/diagnostics/probe_index_retrieval.py` - S1 index audit  
- `src/diagnostics/probe_cross_encoder.py` - S2 pair feeding analysis
- `src/diagnostics/probe_coverage_features.py` - Coverage feature validation
- `scripts/run_selection_stack_diagnostics.py` - CLI entry point

The system provides definitive diagnosis of selection pipeline failures with targeted fixes, enabling rapid debugging of 0.0% coverage issues in the Lethe retrieval system.