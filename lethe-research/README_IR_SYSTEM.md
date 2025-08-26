# Production-Grade Information Retrieval System

This directory contains a complete production-ready implementation of BM25 and ANN (Approximate Nearest Neighbor) retrieval systems with comprehensive timing infrastructure for the Lethe hybrid IR research project.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install IR-specific dependencies
pip install -r requirements_ir.txt

# Optional: Install with GPU support
pip install faiss-gpu  # Instead of faiss-cpu
```

### 2. Build Indices

```bash
# Build all indices for all datasets
python scripts/build_indices.py --config config/retriever_config.yaml

# Build specific indices for specific dataset
python scripts/build_indices.py --config config/retriever_config.yaml \
    --dataset msmarco-passage-dev --indices bm25,hnsw

# Build with custom output directory
python scripts/build_indices.py --config config/retriever_config.yaml \
    --output-dir ./custom_indices --debug
```

### 3. Benchmark Performance

```bash
# Benchmark all indices
python scripts/benchmark_indices.py --config config/retriever_config.yaml

# Generate recall curves for ANN indices
python scripts/benchmark_indices.py --config config/retriever_config.yaml \
    --recall-curves --ann-sweep

# Custom benchmark parameters
python scripts/benchmark_indices.py --config config/retriever_config.yaml \
    --dataset msmarco-passage-dev --cold-cycles 50 --warm-cycles 500
```

## 📋 System Architecture

### Core Components

#### 1. **BM25 Retriever** (`src/retriever/bm25.py`)
- **Production Implementation**: Uses PySerini/Anserini for real Lucene-based BM25
- **Fallback Support**: rank-bm25 when PySerini unavailable  
- **Statistics Export**: Vocabulary size, postings, average document length
- **Parameter Management**: k1, b values with validation

#### 2. **ANN Retriever** (`src/retriever/ann.py`)
- **HNSW Indices**: FAISS implementation with efSearch parameter sweeps
- **IVF-PQ Indices**: Quantized indices with (nlist, nprobe, nbits) optimization
- **Recall Curves**: Automated generation targeting ≥0.98@1k recall
- **Budget Constraints**: ±5% compute/FLOPs parity enforcement

#### 3. **Dense Embeddings** (`src/retriever/embeddings.py`)
- **Model Support**: Sentence-BERT, BGE, E5 via transformers
- **Efficient Processing**: GPU/CPU optimization, FP16, batch processing
- **Persistent Caching**: Hash-validated embedding storage
- **Memory Management**: Streaming for large collections

#### 4. **Timing Harness** (`src/retriever/timing.py`)
- **Dual-Timer System**: In-process + external timing measurement
- **Steady-State Warm-up**: Configurable cold/warm cycles (default: 50/500)
- **GC Barriers**: Explicit garbage collection between runs
- **Statistical Reporting**: p50/p95/p99 latencies, throughput, memory usage

#### 5. **Metadata Management** (`src/retriever/metadata.py`)
- **Index Parameters**: Complete build configuration persistence
- **Content Hashing**: SHA-256 validation for reproducibility
- **Recall Curves**: Parameter sweep results storage
- **Performance Metrics**: Build times, index sizes, compression ratios

### Directory Structure

```
indices/
├── {DATASET}/
│   ├── bm25/
│   │   ├── index.faiss              # BM25 index files
│   │   └── params.meta              # Build parameters
│   ├── dense/
│   │   ├── hnsw/
│   │   │   ├── index.faiss          # HNSW index
│   │   │   ├── doc_ids.npy          # Document ID mapping
│   │   │   └── params.meta          # HNSW parameters
│   │   └── ivf_pq/
│   │       ├── index.faiss          # IVF-PQ index
│   │       ├── doc_ids.npy          # Document ID mapping  
│   │       └── params.meta          # IVF-PQ parameters
│   ├── embeddings/
│   │   ├── {collection}_embeddings.npy   # Dense vectors
│   │   └── {collection}_metadata.json    # Embedding metadata
│   └── metadata/
│       ├── {dataset}_bm25_{name}.meta    # Index metadata
│       └── {dataset}_dense_{name}.meta   # ANN metadata
```

## ⚙️ Configuration

The system uses YAML configuration files with comprehensive validation:

### Key Configuration Sections

```yaml
# BM25 parameters
bm25:
  k1: 0.9           # Term frequency saturation
  b: 0.4            # Document length normalization
  stemmer: "porter" # Stemming algorithm
  
# HNSW parameters  
hnsw:
  m: 16                           # Number of connections
  ef_construction: 200            # Build quality
  ef_search_values: [64,128,256,512]  # Search quality sweep
  target_recall: 0.98             # Target recall@1000
  
# IVF-PQ parameters
ivf_pq:
  nlist_values: [1000,4000,16000]     # Cluster counts
  nprobe_values: [1,4,16,64]          # Search clusters
  nbits_values: [6,8,10]              # Quantization bits
  m_pq: 64                            # Subquantizers
  
# Dense embedding configuration
embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 32
  normalize_embeddings: true
  device: "auto"                      # cuda/cpu/auto
  
# System configuration
system:
  cpu_cores: 32                       # CPU_32C_128G profile
  memory_gb: 128
  gpu_memory_gb: 40                   # A100_40G profile
  concurrency: 1                      # Latency measurement
  flops_budget_variance: 0.05         # ±5% budget parity
```

## 🔬 Performance Measurement

### Timing Infrastructure

The system implements a **dual-timer system** with production-grade measurement:

1. **In-Process Timer**: `time.perf_counter()` for microsecond precision
2. **External Validation**: Process-level CPU and memory monitoring
3. **Steady-State Protocol**: 
   - Cold cycles (default: 50) - discarded for warm-up
   - Warm cycles (default: 500) - measured for statistics
4. **GC Barriers**: Explicit `gc.collect()` between measurements
5. **Statistical Aggregation**: p50/p95/p99 latencies, memory profiling

### Budget Parity Validation

The system enforces **±5% compute/FLOPs parity** across indices:

```python
# Validate query-time compute budget
def validate_budget_constraints(query_flops, budget_flops, variance=0.05):
    lower_bound = budget_flops * (1 - variance)
    upper_bound = budget_flops * (1 + variance)
    return lower_bound <= query_flops <= upper_bound
```

### Recall Curve Generation

For ANN indices, the system automatically generates recall curves:

- **HNSW**: efSearch parameter sweep [64, 128, 256, 512]
- **IVF-PQ**: nprobe parameter sweep [1, 4, 16, 64]
- **Target Metrics**: Recall@1000 ≥ 0.98
- **Performance Tracking**: Latency and memory for each parameter setting

## 📊 Output Formats

### Index Metadata (`params.meta`)

```json
{
  "index_type": "hnsw",
  "build_params": {
    "m": 16,
    "ef_construction": 200,
    "max_m": 16
  },
  "stats": {
    "num_documents": 8841823,
    "index_size_mb": 2847.3,
    "build_time_sec": 1247.8,
    "memory_used_mb": 4096.0
  },
  "content_hash": "a7b8c9d...",
  "created_at": "2025-01-15T10:30:00Z"
}
```

### Recall Curve Data

```json
{
  "parameter_name": "efSearch",
  "parameter_values": [64, 128, 256, 512],
  "recall_at_k": {
    "10": [0.89, 0.94, 0.97, 0.98],
    "100": [0.91, 0.96, 0.98, 0.99],
    "1000": [0.93, 0.97, 0.99, 0.99]
  },
  "latency_ms": [12.3, 18.7, 28.1, 42.5],
  "memory_mb": [2847, 2847, 2847, 2847]
}
```

### Performance Profiles

```json
{
  "operation": "hnsw_search",
  "count": 500,
  "latency_p50": 15.2,
  "latency_p95": 23.8,
  "latency_p99": 31.4,
  "throughput": 47.3,
  "memory_peak_mb": 2891.2,
  "gc_collections": 3
}
```

## 🎯 Quality Assurance

### Index Validation

The system provides comprehensive index validation:

```python
# Validate index integrity
validation_results = registry.validate_all_indices()
# Returns: metadata_exists, index_path_exists, content_hash_valid, parameter_hash_valid
```

### Performance Benchmarks

Standard performance benchmarks with configurable parameters:

- **Latency**: p50/p95/p99 query response times
- **Throughput**: Queries per second under load  
- **Memory**: Peak and average memory usage
- **Accuracy**: Recall@k for ANN indices

### Reproducibility Features

- **Content Hashing**: SHA-256 validation of input data
- **Parameter Hashing**: Build configuration verification
- **Environment Capture**: System information and dependencies
- **Model Versioning**: Transformer model checkpoint tracking

## 🔧 Development and Extension

### Adding New Index Types

1. **Create Retriever Class**: Inherit from base retriever interface
2. **Implement Builder**: Follow `ANNIndexBuilder` pattern
3. **Add Configuration**: Extend `RetrieverConfig` with new parameters
4. **Register Factory**: Add to `create_*_retriever` functions

### Custom Models

```python
# Add new embedding model
config = EmbeddingConfig(
    model_name="your-custom-model",
    batch_size=64,
    max_length=256
)

embedding_manager = DenseEmbeddingManager(config)
```

### Advanced Timing

```python
# Custom timing harness
harness = TimingHarness(
    cold_cycles=100,      # More warm-up
    warm_cycles=1000,     # More measurements
    gc_between_runs=True, # Memory isolation
    memory_profiling=True # Detailed memory tracking
)
```

## 🚨 Hardware Requirements

### Minimum Requirements
- **CPU**: 8 cores, 16GB RAM
- **Storage**: 10GB for indices
- **Python**: 3.8+ with pip

### Recommended Configuration
- **CPU**: 32 cores, 128GB RAM (CPU_32C_128G profile)
- **GPU**: A100 40GB for dense embedding generation
- **Storage**: 100GB SSD for large-scale datasets
- **Network**: High-bandwidth for model downloads

## 📚 Dataset Support

Currently supported BEIR datasets:

- **MS MARCO Passage Dev**: Large-scale passage ranking
- **TREC-COVID**: Scientific literature retrieval
- **NFCorpus**: Nutrition and health information
- **FiQA-2018**: Financial question answering

### Adding New Datasets

1. Place collection in `datasets/{name}/collection.jsonl`
2. Add queries in `datasets/{name}/queries.jsonl`  
3. Add ground truth in `datasets/{name}/qrels.jsonl`
4. Update configuration `datasets` list

## 🔍 Troubleshooting

### Common Issues

**PySerini Installation**: 
```bash
pip install pyserini
# If fails, use fallback:
pip install rank-bm25
```

**FAISS GPU Issues**:
```bash
pip install faiss-cpu  # Fallback to CPU
# Check GPU availability:
python -c "import faiss; print(faiss.StandardGpuResources())"
```

**Memory Issues**:
- Reduce `batch_size` in embedding config
- Increase system `memory_gb` setting
- Use `fp16=true` for GPU models

**Performance Issues**:
- Check `warm_cache=true` in system config
- Verify `concurrency=1` for latency measurements
- Monitor with `memory_profiling=true`

### Debug Mode

```bash
python scripts/build_indices.py --debug --config config/retriever_config.yaml
```

Enables detailed logging, timing breakdowns, and intermediate result validation.

---

## 📄 Implementation Summary

This implementation provides a **production-grade IR system** meeting all Task 2 requirements:

✅ **Real BM25 Indices**: PySerini/Anserini-based with collection statistics  
✅ **ANN Engineering**: FAISS HNSW/IVF-PQ with parameter sweeps  
✅ **Dense Embeddings**: Multi-model support with efficient caching  
✅ **Timing Harness**: Dual-timer system with steady-state measurement  
✅ **Index Metadata**: Comprehensive parameter and performance export  
✅ **Budget Parity**: ±5% compute/FLOPs constraint validation  
✅ **Recall Curves**: Automated ANN parameter optimization  

The system eliminates all synthetic components from previous implementations and provides real, measurable performance characteristics suitable for production deployment and research evaluation.