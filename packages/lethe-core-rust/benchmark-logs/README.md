# Benchmark Logs Directory

This directory contains structured logs from Lethe benchmarking runs. **All files in this directory are gitignored** to prevent sensitive LLM interaction data from being committed.

## Directory Structure

```
benchmark-logs/
├── runs/           # Individual benchmark run logs, organized by timestamp
│   └── YYYY-MM-DD_HH-MM-SS_<benchmark_name>/
│       ├── metadata.json      # Run configuration and metadata
│       ├── requests.jsonl     # All LLM request/response logs
│       ├── performance.jsonl  # Performance metrics per operation
│       └── summary.json       # Run summary and final metrics
├── analysis/       # Post-processing analysis results
│   └── comparative_analysis_<timestamp>.json
├── summary/        # Daily/weekly summary reports
│   └── <date>_summary.json
└── providers/      # Provider-specific logs for debugging
    ├── openai/
    ├── anthropic/
    └── ollama/
```

## Log Formats

### Request Log Format (requests.jsonl)
Each line contains a JSON object representing a single LLM interaction:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "event": "benchmark_llm_request",
  "benchmark_id": "lethe_vs_competitors_2024_01_15",
  "benchmark_name": "retrieval_accuracy",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "query_id": "query_001",
  "performance": {
    "request_duration_ms": 1234,
    "tokens_used": 150,
    "response_time_ms": 1200
  },
  "request_data": {
    "query": "...",
    "context": "...",
    "parameters": {...}
  },
  "response_data": {
    "content": "...",
    "finish_reason": "stop"
  },
  "lethe_transform": {
    "enabled": true,
    "changes_applied": [...],
    "transform_duration_ms": 5
  }
}
```

### Performance Log Format (performance.jsonl)
Performance metrics collected during benchmark execution:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "event": "benchmark_performance",
  "benchmark_id": "lethe_vs_competitors_2024_01_15",
  "operation": "embedding_generation",
  "duration_ms": 234,
  "throughput_ops_per_sec": 4.27,
  "memory_usage_mb": 45.2,
  "cpu_usage_percent": 23.1
}
```

## Usage

Benchmark logs are automatically created when running benchmarks with logging enabled:

```bash
cargo run --bin lethe-cli benchmark all --log-level detailed
cargo run --bin lethe-cli benchmark query --count 100 --enable-proxy-logging
```

## Analysis

Use the provided analysis utilities to process logs:

```bash
cargo run --bin lethe-cli analyze-benchmarks --run-dir benchmark-logs/runs/2024-01-15_10-30-00_retrieval_accuracy
```