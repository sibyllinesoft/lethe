# 🔬 LLM Context Management Research Benchmarks

This directory contains benchmarking implementations for competing LLM context management and pruning research projects, providing apples-to-apples comparisons with Lethe.

## 🎯 **Purpose**

Compare Lethe's context management approach against:
- **Context Compression**: LLMLingua, LongLLMLingua, Selective Context
- **Context Pruning**: H2O, StreamingLLM attention mechanisms  
- **Position Management**: Lost in the Middle mitigation strategies
- **Hierarchical Attention**: LongNet, Landmark Attention approaches

## 🏗️ **Architecture**

**Modular Design:**
```
src/context_competitors/
├── README.md                    # This file
├── competitor_runner.py         # Main orchestrator (robust to script changes)
├── competitor_interface.py      # Common interface for all competitors
├── benchmarks/                  # Individual competitor implementations
│   ├── llmlingua_benchmark.py
│   ├── h2o_benchmark.py  
│   ├── selective_context_benchmark.py
│   ├── streamingllm_benchmark.py
│   └── longnet_benchmark.py
└── results/                     # Benchmark outputs
    ├── comparison_results.json
    └── performance_metrics.csv
```

**Key Features:**
- ✅ **Isolated**: Self-contained competitor implementations  
- ✅ **Modular**: Add/remove competitors by adding/deleting scripts
- ✅ **Robust**: Parent runner discovers competitors automatically
- ✅ **Comparable**: Common interface ensures fair comparison

## 🚀 **Usage**

```bash
# Run all available competitors
python3 -m src.context_competitors.competitor_runner

# Run specific competitors  
python3 -m src.context_competitors.competitor_runner --competitors llmlingua h2o

# Generate comparison report
python3 -m src.context_competitors.competitor_runner --report-only
```

## 📊 **Benchmarking Protocol**

**Common Evaluation Tasks:**
- **Long Context QA**: InfiniteBench tasks (PassKey, KV retrieval)
- **Context Compression**: Maintain performance while reducing tokens
- **Position Robustness**: Performance across different context positions
- **Scalability**: Performance vs context length (1K → 1M+ tokens)

**Metrics Tracked:**
- **Performance**: Accuracy, F1, exact match on QA tasks
- **Efficiency**: Context reduction ratio, processing latency  
- **Scalability**: Performance degradation with context length
- **Resource Usage**: Memory, compute requirements

## 🔧 **Adding New Competitors**

1. **Create benchmark script** in `benchmarks/` directory
2. **Implement common interface** from `competitor_interface.py`  
3. **Add installation requirements** to competitor's docstring
4. **Parent runner auto-discovers** new competitors

No need to modify the main runner - it's designed to be robust to script additions/removals!