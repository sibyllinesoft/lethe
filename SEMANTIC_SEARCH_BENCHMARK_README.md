# 🔍 Semantic Search Benchmarking System

A comprehensive benchmarking system for evaluating semantic search methods using the InfinityBench zh.qa dataset (2M token stress test). This system compares three retrieval approaches: ChromaDB vector search, Lethe context-aware retrieval, and simple truncation methods.

## 🚀 Features

- **InfinityBench zh.qa Dataset**: Uses Chinese QA with 2M+ token contexts for stress testing
- **Three Comparison Methods**:
  - **ChromaDB**: Vector similarity search with sentence-transformers embeddings
  - **Lethe**: Context-aware semantic retrieval (placeholder implementation)  
  - **Truncation**: First 120k tokens + direct Ollama query
- **Comprehensive Testing**: ROC curves at k=1,5,10,20,50 for proper evaluation
- **Publication-Quality Visualizations**: Performance plots, precision/recall curves, query time analysis
- **GPU Acceleration**: CUDA support for embeddings and inference
- **Robust Error Handling**: Comprehensive validation and fallback systems

## 📋 System Requirements

- **Python 3.11+**
- **CUDA-capable GPU** (for optimal performance)
- **Ollama** with a large language model (tested with Gemma 3 27B)
- **16GB+ RAM** (recommended for full dataset)

## 🛠️ Installation

### Dependencies
```bash
# Install via pip with system override (if needed)
pip3 install --break-system-packages -r semantic_search_requirements.txt
```

### Verify ChromaDB Installation
```bash
python3 test_chromadb.py
```
This runs comprehensive tests including:
- Basic ChromaDB functionality
- Metadata handling and filtering
- Custom embeddings with sentence transformers
- Persistence across client restarts
- Large batch operations (1000+ documents)
- Multilingual support (Chinese, English, etc.)
- Performance benchmarking

## 📊 Usage

### Quick Test (3 samples)
```bash
python3 run_small_benchmark.py
```

### Full Benchmark (All samples)
```bash
python3 semantic_search_benchmark.py
```

### Custom Configuration
```python
from semantic_search_benchmark import SemanticSearchBenchmark
import asyncio

async def custom_benchmark():
    benchmark = SemanticSearchBenchmark(
        data_dir="./custom_data",
        chroma_dir="./custom_chroma", 
        max_samples=50,  # Limit samples for testing
        ollama_model="your-model"
    )
    
    await benchmark.run_comprehensive_benchmark()

asyncio.run(custom_benchmark())
```

## 🏗️ Architecture

### Component Overview
```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   InfinityBench     │    │   ChromaDB Vector   │    │   Ollama LLM        │
│   Dataset Loader    │───▶│   Search Index      │───▶│   Generation        │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
           │                           │                           │
           ▼                           ▼                           ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Sentence          │    │   Metadata &        │    │   Performance       │
│   Transformers      │    │   Chunk Management  │    │   Metrics           │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

### Data Processing Pipeline
1. **Dataset Loading**: InfinityBench zh.qa (Chinese QA with 600k+ tokens per sample)
2. **Context Chunking**: Split long contexts into 1000-token overlapping chunks
3. **Vector Embedding**: Generate embeddings using multilingual sentence transformers
4. **Index Creation**: Store in ChromaDB with metadata for efficient retrieval
5. **Query Processing**: Search, rank, and generate answers using Ollama
6. **Evaluation**: Calculate precision/recall at different k values

## 📈 Evaluation Metrics

### Precision@K and Recall@K
- Measures retrieval effectiveness at different cutoffs
- k values: 1, 5, 10, 20, 50

### Query Performance
- **Query Time**: Time to retrieve relevant chunks
- **Generation Time**: Time for LLM to generate answers
- **Total Latency**: End-to-end response time

### Scalability Analysis
- Performance vs context length
- Memory usage patterns  
- Throughput under load

## 🎯 Key Results Structure

### Generated Files
```
benchmark_data/
├── results/
│   ├── chroma_results.json          # ChromaDB benchmark results
│   ├── lethe_results.json           # Lethe benchmark results  
│   ├── truncation_results.json      # Truncation benchmark results
│   └── all_results.csv              # Combined analysis dataset
└── visualizations/
    ├── precision_at_k.png           # Precision@K comparison
    ├── recall_at_k.png              # Recall@K comparison
    ├── query_times.png              # Performance comparison
    └── performance_vs_length.png    # Scalability analysis
```

### Sample Results
```json
{
  "method": "chroma",
  "sample_id": "sample_0", 
  "k": 10,
  "query_time": 0.245,
  "generation_time": 8.123,
  "retrieved_relevant": 7,
  "precision_at_k": 0.7,
  "recall_at_k": 0.85,
  "context_length": 611376
}
```

## 🔧 Configuration Options

### SemanticSearchBenchmark Parameters
- `data_dir`: Directory for storing benchmark data and results
- `chroma_dir`: ChromaDB persistence directory  
- `max_samples`: Limit number of samples (None for all)
- `ollama_model`: Ollama model name (e.g., "gemma3:27b")

### Chunking Parameters
- `chunk_size`: Token count per chunk (default: 1000)
- `overlap`: Overlapping tokens between chunks (default: 200)

### Model Settings
- **Sentence Transformer**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Vector Space**: Cosine similarity
- **Generation Parameters**: Temperature=0.0, max_tokens=500

## 🚨 Stress Testing Capabilities

### 2M Token Handling
- **Dataset**: InfinityBench zh.qa with 600k+ tokens per context
- **Chunking**: Efficient processing of ultra-long documents  
- **Memory Management**: Batch processing for large datasets
- **Performance**: GPU acceleration for embedding generation

### Scalability Validation
- Tests with 1000+ document collections
- Multi-gigabyte index handling
- Concurrent query processing
- Memory efficiency monitoring

## 🐛 Troubleshooting

### Common Issues

#### ChromaDB Installation
```bash
# If ChromaDB fails to install
pip3 install --break-system-packages chromadb

# Test installation
python3 -c "import chromadb; print('ChromaDB OK')"
```

#### Ollama Connection
```bash
# Check Ollama service
ollama list

# Test model availability  
ollama run gemma3:27b "Test prompt"
```

#### Memory Issues
```bash
# Monitor memory usage during benchmarking
watch -n 1 'free -h && nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
```

#### Dataset Loading Errors
- Ensure stable internet connection for HuggingFace datasets
- Check available disk space (>10GB recommended)
- Verify dataset access permissions

### Performance Optimization

#### GPU Acceleration
```python
# Verify CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name()}")
```

#### Batch Size Tuning
```python
# Adjust batch sizes based on available GPU memory
benchmark = SemanticSearchBenchmark(...)
# Modify batch_size in create_chroma_index() method
```

## 📚 Technical References

### Key Dependencies
- **ChromaDB 1.0.20+**: Vector database for similarity search
- **Sentence Transformers 5.1.0+**: Multilingual embeddings
- **Datasets 4.0.0+**: HuggingFace dataset access
- **Ollama 0.5.3+**: Local LLM inference
- **PyTorch**: GPU acceleration for embeddings

### Dataset Citation
```bibtex
@article{zhang2024infinitybench,
  title={InfinityBench: Extending Long Context Evaluation Beyond 100K Tokens},
  author={Zhang, Xinrong and others},
  journal={arXiv preprint arXiv:2402.13718},
  year={2024}
}
```

## 🎯 Future Enhancements

- **Real Lethe Integration**: Replace placeholder with actual Lethe retrieval
- **Additional Datasets**: Support for English and other language variants
- **Advanced Metrics**: BLEU, ROUGE, and semantic similarity scores
- **Distributed Processing**: Multi-GPU and multi-node scaling
- **Interactive Dashboard**: Real-time monitoring and analysis
- **A/B Testing Framework**: Systematic method comparison

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review generated log files (`semantic_search_benchmark.log`)
3. Validate system requirements and dependencies
4. Test individual components with provided validation scripts

The system is designed for robustness and provides comprehensive error reporting to help diagnose and resolve issues quickly.