# Benchmark Hardware Profile & Reproducibility Requirements

## Hardware Target Profile

**All reported benchmark numbers in this project are measured on the following standardized hardware profile:**

### Primary Target: MacBook Air M2
- **CPU**: Apple M2 (8-core: 4 performance + 4 efficiency cores)
- **Memory**: 16 GB unified memory
- **Storage**: SSD (minimum 256GB available)
- **OS**: macOS Sonoma 14.x or later
- **Architecture**: arm64

### Software Environment
- **Node.js**: v20.11.0 (LTS)
- **SQLite**: 3.42.0 or later
- **Python**: 3.11.x (for data processing scripts)
- **Git**: 2.40+ with LFS support

### Reproducibility Standards

#### Deterministic Builds
- **Global random seed**: `42` (used across all components)
- **HNSW index seed**: `123` (for consistent vector index builds)
- **Dataset generation seed**: `456` (for synthetic data)
- **Evaluation seed**: `789` (for train/test splits)

#### Environment Variables
```bash
export LETHE_RANDOM_SEED=42
export LETHE_HNSW_SEED=123
export LETHE_DATASET_SEED=456
export LETHE_EVAL_SEED=789
export NODE_ENV=benchmark
export UV_THREADPOOL_SIZE=8
```

#### Version Pinning
All dependencies are locked to specific versions in:
- `package-lock.json` (Node.js dependencies)
- `requirements-lock.txt` (Python dependencies)
- `lockfile-benchmark.json` (Complete environment snapshot)

### Measurement Standards

#### Latency Reporting
- **End-to-end measurements**: Include full request parsing, index lookup, result formatting
- **Cold vs Warm**: First query after process start (cold) vs subsequent queries (warm)
- **Percentiles**: P50, P95, P99 over minimum 1000 samples
- **Concurrency**: Measured at 1, 5, 10 concurrent clients
- **Corpus sizes**: Tested at 1k, 10k, 100k atoms

#### Memory Reporting
- **Peak RSS**: Maximum resident set size during operation
- **Index sizes**: On-disk storage requirements for FTS5 and HNSW indices
- **Working set**: Memory usage during active query processing

#### Quality Metrics
- **Statistical significance**: Bootstrap confidence intervals (n=1000)
- **Effect sizes**: Cohen's d with 95% CI
- **Multiple comparisons**: Bonferroni correction applied
- **Reproducibility tolerance**: ±2% variation across runs

### Validation Requirements

Before any benchmark execution, the environment must pass validation via:
```bash
./scripts/benchmark_env_check.sh
```

This script verifies:
- Hardware specifications match target profile
- Software versions are correctly pinned
- Random seeds are properly configured
- Required disk space is available
- Network dependencies are disabled for local-first execution

### Alternative Hardware Profiles

**Note**: If measurements are taken on alternative hardware, they must be clearly labeled and reported in separate result files under `results/HARDWARE_PROFILE/`. Mixing hardware profiles within the same result table is strictly prohibited.

#### Supported Alternative: Windows Intel Laptop
- **CPU**: Intel i5-1240P or equivalent (12-core)
- **Memory**: 16 GB DDR4/DDR5
- **OS**: Windows 11 22H2+
- **Results path**: `results/WIN_INTEL_I5_16GB/`

### Compliance Declaration

By following these standards, we ensure:
1. **Reproducible builds** across identical hardware configurations
2. **Honest latency reporting** with proper cold/warm distinctions
3. **Statistical rigor** in quality comparisons
4. **Local-first execution** without cloud dependencies
5. **Hardware-specific results** clearly labeled and separated

All reported numbers in the paper correspond to the **MacBook Air M2** profile unless explicitly noted otherwise.