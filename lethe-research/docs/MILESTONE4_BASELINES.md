# Milestone 4: Stronger Local Baselines

## Overview

This document describes the implementation of six stronger local baseline systems for rigorous evaluation against the Lethe hybrid retrieval system. All baselines operate with identical computational budgets and produce comparable outputs for fair comparison.

## Implemented Baselines

### 1. BM25-only (SQLite FTS5)
- **Implementation**: `SQLiteFTSBaseline`
- **Description**: Pure lexical search using SQLite's FTS5 with BM25 scoring
- **Parameters**: k1=1.2, b=0.75 (standard BM25 parameters)
- **Advantages**: Fast, interpretable, good for exact keyword matches
- **Use Cases**: Technical documentation, code search, exact terminology

### 2. Vector-only (ANN)
- **Implementation**: `VectorOnlyBaseline`  
- **Description**: Dense vector search using sentence transformers + FAISS
- **Model**: all-MiniLM-L6-v2 (384-dim embeddings)
- **Index**: HNSW with M=16, efConstruction=200, efSearch=50
- **Advantages**: Semantic similarity, handles synonyms and paraphrases
- **Use Cases**: Conceptual queries, natural language questions

### 3. BM25+Vector (Static α=0.5)
- **Implementation**: `HybridStaticBaseline`
- **Description**: Static fusion of BM25 and vector scores without reranking
- **Fusion**: score = 0.5 * normalized_BM25 + 0.5 * normalized_vector
- **Advantages**: Combines lexical and semantic signals
- **Use Cases**: General-purpose retrieval with balanced precision/recall

### 4. MMR Diversity (λ=0.7)
- **Implementation**: `MMRDiversityBaseline`
- **Description**: Maximal Marginal Relevance for result diversification
- **Algorithm**: Iteratively select documents maximizing λ*relevance - (1-λ)*similarity_to_selected
- **Advantages**: Reduces redundancy, increases topic coverage
- **Use Cases**: Exploratory search, broad topic coverage

### 5. BM25 + Doc2Query Expansion
- **Implementation**: `Doc2QueryExpansionBaseline`
- **Description**: BM25 over documents augmented with generated query expansions
- **Model**: doc2query/msmarco-t5-base-v1 (with pattern-based fallback)
- **Process**: Generate 3-5 likely queries per document offline, search expanded index
- **Advantages**: Improved recall through vocabulary mismatch reduction
- **Use Cases**: Technical documentation, FAQ retrieval

### 6. Tiny Cross-Encoder Reranking
- **Implementation**: `TinyCrossEncoderBaseline`
- **Description**: Two-stage retrieval with vector first-stage + cross-encoder reranking
- **Model**: cross-encoder/ms-marco-MiniLM-L-2-v2
- **Process**: Vector retrieval → rerank top-100 candidates → return top-k
- **Advantages**: High precision through interaction modeling
- **Use Cases**: High-precision applications, final answer selection

## Key Features

### Identical Interfaces
All baselines implement the same `BaselineRetriever` interface:
```python
class BaselineRetriever(ABC):
    def build_index(self, documents: List[RetrievalDocument]) -> None
    def retrieve(self, query: EvaluationQuery, k: int) -> List[RetrievalResult]  
    def get_flops_estimate(self, query: EvaluationQuery, k: int) -> int
```

### Shared Infrastructure  
- **Single index build**: Documents processed once, reused across baselines
- **Consistent data formats**: RetrievalDocument, EvaluationQuery, RetrievalResult
- **Unified configuration**: JSON config file controls all parameters

### Performance Parity
- **Budget tracking**: FLOPs estimation with ±5% tolerance across baselines
- **Fair comparison**: Same candidate limits (k=100, expansion up to 1000)
- **Hardware profiling**: Consistent measurement across all methods

### Anti-Fraud Validation
- **Non-empty results**: Prevent baselines returning empty results
- **Score variance**: Detect constant dummy scores
- **Smoke testing**: Validate baselines on representative queries
- **Content validation**: Ensure results contain actual document content

## Usage

### Single Command Execution
```bash
# Run all baselines on LetheBench
make baselines

# Quick test with limited queries  
make baseline-quick-test

# Test implementation before full run
make test-baselines
```

### Detailed Command Line
```bash
python scripts/run_milestone4_baselines.py \
    --dataset datasets/lethebench \
    --output results/milestone4_baselines.json \
    --k 100 \
    --alpha 0.5 \
    --mmr-lambda 0.7
```

### Configuration File
```bash
python scripts/run_milestone4_baselines.py \
    --dataset datasets/lethebench \
    --output results/baselines.json \
    --config config/milestone4_baseline_config.json
```

## Output Format

Results are saved as comprehensive JSON with:

```json
{
  "metadata": {
    "timestamp": 1234567890.0,
    "total_baselines": 6,
    "total_queries": 139,
    "config": {...},
    "hardware_profile": {...}
  },
  "budget_report": {
    "baseline_budget": 1000000.0,
    "tolerance": 0.05,
    "methods": {
      "bm25_only": {
        "mean_flops": 1000000.0,
        "deviation_from_baseline": 0.0,
        "parity_compliant": true
      }
    }
  },
  "validation_report": {...},
  "baseline_results": {
    "bm25_only": [
      {
        "baseline_name": "bm25_only",
        "query_id": "query_001",
        "retrieved_docs": ["doc_123", "doc_456"],
        "relevance_scores": [0.95, 0.87],
        "latency_ms": 23.4,
        "flops_estimate": 987654,
        "non_empty_validated": true
      }
    ]
  }
}
```

## Dependencies

### Required (Core Functionality)
- `sqlite3` (built into Python)
- `numpy`
- `psutil` (memory/CPU monitoring)

### Optional (Enhanced Baselines)
- `sentence-transformers` (vector and cross-encoder baselines)
- `faiss-cpu` (vector indexing)
- `transformers` (doc2query expansion)

### Installation
```bash
# Core dependencies
pip install numpy psutil

# Optional dependencies for full functionality
pip install sentence-transformers faiss-cpu transformers torch
```

**Note**: Baselines with missing dependencies are automatically skipped with warnings.

## Performance Characteristics

### Computational Complexity
- **BM25**: O(|q| * |D|) where |q|=query terms, |D|=documents
- **Vector**: O(|q| * d + log|D| * d) where d=embedding dimension  
- **Hybrid**: O(BM25 + Vector + fusion)
- **MMR**: O(Vector + k² * d) for k results
- **Doc2Query**: O(BM25 * expansion_factor)
- **CrossEncoder**: O(Vector + k * model_params)

### Expected Latency (per query)
- **BM25**: 10-50ms
- **Vector**: 50-200ms
- **Hybrid**: 100-300ms  
- **MMR**: 200-500ms
- **Doc2Query**: 50-150ms
- **CrossEncoder**: 500-2000ms

### Memory Usage
- **BM25**: ~10MB + index size
- **Vector**: ~100MB + embeddings (4 * |D| * dim bytes)
- **Others**: Combination of above

## Validation and Testing

### Unit Tests
```bash
python scripts/test_milestone4_implementation.py
```

Tests each baseline individually:
- Interface compliance
- Index building
- Query retrieval  
- FLOPS estimation
- Memory tracking

### Integration Tests
- Full evaluator workflow
- Budget parity tracking
- Anti-fraud validation
- Result serialization

### Smoke Tests
- Run baselines on representative queries
- Validate non-empty, meaningful results
- Check score distributions
- Verify content quality

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ImportError: No module named 'sentence_transformers'
   ```
   - Install optional dependencies or skip affected baselines

2. **Memory Errors**
   ```
   MemoryError: Unable to allocate array
   ```
   - Reduce batch size or use smaller embedding model

3. **Empty Results**
   ```
   WARNING: Empty results returned
   ```
   - Check dataset format and query preprocessing

4. **Budget Parity Violations**
   ```
   WARNING: Budget parity violation: method_x used 1.2e6 FLOPs
   ```
   - Indicates computational fairness issue, investigate method

### Debugging Commands

```bash
# Test specific baseline
make baseline-bm25-only

# Run with verbose logging
python scripts/run_milestone4_baselines.py --dataset datasets/lethebench --output results/debug.json --max-queries 5

# Check implementation
python -c "from src.eval.milestone4_baselines import *; print('Import successful')"
```

## Extending the Framework

### Adding New Baselines

1. **Inherit from BaselineRetriever**:
```python
class MyBaseline(BaselineRetriever):
    def __init__(self, config):
        super().__init__("MyBaseline", config)
    
    def build_index(self, documents): 
        # Build index
        pass
    
    def retrieve(self, query, k):
        # Return List[RetrievalResult]
        pass
        
    def get_flops_estimate(self, query, k):
        # Return estimated FLOPs
        pass
```

2. **Register in Evaluator**:
```python
self.baselines["my_baseline"] = MyBaseline(config)
```

3. **Add to Makefile**:
```makefile
baseline-my-method:
    python scripts/run_milestone4_baselines.py \
        --skip-baselines [others] \
        --output results/my_baseline_test.json
```

### Custom Configurations

Create custom config JSON files in `config/` directory and reference with `--config` flag.

## References

- **BM25**: Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond"
- **Dense Retrieval**: Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering"
- **MMR**: Carbonell & Goldstein, "The Use of MMR, Diversity-Based Reranking for Reordering Documents"
- **Doc2Query**: Nogueira et al., "Document Expansion by Query Prediction"
- **Cross-Encoders**: Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"

## Compliance with Milestone 4 Requirements

✅ **BM25-only (SQLite FTS5)** with identical candidate caps  
✅ **Vector-only (ANN)** with same limits  
✅ **BM25+Vector (static α=0.5)** without rerank  
✅ **MMR (λ=0.7)** over vector candidates for diversity baseline  
✅ **BM25 + doc2query expansion** with offline precomputation  
✅ **Tiny Cross-Encoder Rerank** (CPU-only)  
✅ **Identical interfaces** producing comparable JSON outputs  
✅ **Shared infrastructure**: build indices once, reuse across baselines  
✅ **Single command execution** via `make baselines`  
✅ **Performance parity** with fair computational budgets  
✅ **Local execution** only, CPU-compatible  
✅ **Anti-fraud validation** and budget parity tracking