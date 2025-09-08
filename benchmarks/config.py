#!/usr/bin/env python3
"""
Benchmark Configuration Management
==================================

Defines comprehensive configuration for all benchmark runs, competitor systems,
datasets, and evaluation protocols with vendor-fair defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import yaml


@dataclass
class CompetitorConfig:
    """Configuration for a single competitor system."""
    
    name: str
    category: str  # hybrid_vector_db, learned_sparse, reranker, code_search, long_context
    docker_image: str
    api_endpoint: str
    config_params: Dict[str, Any]
    budget_ratios: List[float] = field(default_factory=lambda: [0.08, 0.15, 0.30])
    timeout_seconds: int = 300
    max_retries: int = 3
    vendor_documentation_url: str = ""
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.name or not self.category:
            raise ValueError("name and category are required")
        
        valid_categories = {
            "hybrid_vector_db", "learned_sparse", "reranker", 
            "code_search", "long_context", "baseline"
        }
        if self.category not in valid_categories:
            raise ValueError(f"Invalid category: {self.category}")


@dataclass 
class DatasetConfig:
    """Configuration for a benchmark dataset."""
    
    name: str
    source: str  # infinitebench, ruler, babilong, custom
    loader_class: str
    data_path: str
    official_loader_url: str = ""
    max_samples: Optional[int] = None
    length_stats_required: bool = True
    multilingual: bool = False
    
    # Expected format validation
    expected_fields: List[str] = field(default_factory=lambda: ["query", "context", "answer"])
    
    def __post_init__(self):
        """Validate dataset configuration."""
        if not all([self.name, self.source, self.loader_class, self.data_path]):
            raise ValueError("All dataset fields are required")


@dataclass
class EvaluationConfig:
    """Configuration for evaluation protocols."""
    
    # Budget constraints
    keep_ratios: List[float] = field(default_factory=lambda: [0.08, 0.15, 0.30])
    
    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        "precision_at_k", "recall_at_k", "delta_cbu_per_1k", 
        "middleware_p95", "middleware_p99", "kv_reuse_rate",
        "tail_cvar", "exact_match", "entity_diversity"
    ])
    
    # Statistical testing
    statistical_testing: Dict[str, Any] = field(default_factory=lambda: {
        "bootstrap_iterations": 1000,
        "permutation_iterations": 1000,
        "confidence_level": 0.95,
        "correction_method": "holm",  # Multiple comparisons correction
        "effect_size_threshold": 0.1
    })
    
    # Performance profiling
    latency_percentiles: List[float] = field(default_factory=lambda: [0.5, 0.95, 0.99])
    memory_profiling: bool = True
    cpu_profiling: bool = True
    
    # Quality gates
    quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "min_recall_at_10": 0.1,
        "max_latency_p99_ms": 5000.0,
        "max_memory_gb": 16.0
    })


@dataclass
class InfrastructureConfig:
    """Configuration for benchmark infrastructure."""
    
    # Docker settings
    docker_compose_path: str = "docker-compose.benchmark.yml"
    container_registry: str = "ghcr.io/lethe-research"
    max_parallel_containers: int = 4
    container_memory_limit: str = "8g"
    container_cpu_limit: str = "4.0"
    
    # Storage
    results_dir: str = "benchmark_results"
    logs_dir: str = "benchmark_logs"
    cache_dir: str = "benchmark_cache"
    
    # Networking
    api_timeout_seconds: int = 300
    retry_backoff_seconds: float = 2.0
    
    # Resource monitoring
    monitor_resources: bool = True
    resource_check_interval_seconds: int = 30
    
    # Cleanup
    auto_cleanup_containers: bool = True
    preserve_failed_containers: bool = True


@dataclass 
class ReportingConfig:
    """Configuration for result reporting and visualization."""
    
    # Output formats
    generate_html: bool = True
    generate_pdf: bool = True
    generate_json: bool = True
    generate_csv: bool = True
    
    # HTML report features
    interactive_charts: bool = True
    scenario_cards: bool = True
    advantage_map: bool = True
    raw_data_links: bool = True
    
    # Marketing content
    include_competitor_strengths: bool = True
    include_failure_buckets: bool = True  # When NOT to use Lethe
    include_config_snippets: bool = True
    vendor_documentation_links: bool = True
    
    # Statistical presentation
    show_confidence_intervals: bool = True
    show_effect_sizes: bool = True
    show_significance_tests: bool = True
    highlight_improvements: bool = True
    
    # Report customization
    company_branding: bool = True
    custom_css_path: Optional[str] = None
    logo_path: Optional[str] = None


@dataclass
class BenchmarkConfig:
    """Master configuration for comprehensive benchmarking."""
    
    # Component configurations
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    infrastructure: InfrastructureConfig = field(default_factory=InfrastructureConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    
    # System settings
    random_seed: int = 42
    log_level: str = "INFO"
    max_workers: int = 4
    
    # Experiment control
    run_name: str = ""
    experiment_tags: List[str] = field(default_factory=list)
    dry_run: bool = False
    
    # Competitor and dataset selections
    enabled_competitors: List[str] = field(default_factory=list)  # Empty = all
    enabled_datasets: List[str] = field(default_factory=list)     # Empty = all
    
    def __post_init__(self):
        """Initialize derived settings."""
        if not self.run_name:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"comprehensive_benchmark_{timestamp}"
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'BenchmarkConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Convert nested dicts to dataclass instances
        if 'evaluation' in data:
            data['evaluation'] = EvaluationConfig(**data['evaluation'])
        if 'infrastructure' in data:
            data['infrastructure'] = InfrastructureConfig(**data['infrastructure'])
        if 'reporting' in data:
            data['reporting'] = ReportingConfig(**data['reporting'])
            
        return cls(**data)
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        # Convert dataclass instances to dicts for serialization
        data = {
            'evaluation': self.evaluation.__dict__,
            'infrastructure': self.infrastructure.__dict__,
            'reporting': self.reporting.__dict__,
            'random_seed': self.random_seed,
            'log_level': self.log_level,
            'max_workers': self.max_workers,
            'run_name': self.run_name,
            'experiment_tags': self.experiment_tags,
            'dry_run': self.dry_run,
            'enabled_competitors': self.enabled_competitors,
            'enabled_datasets': self.enabled_datasets
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)


# Vendor-fair default configurations for all competitors
COMPETITOR_CONFIGS = {
    # Hybrid Vector DBs
    "weaviate": CompetitorConfig(
        name="weaviate",
        category="hybrid_vector_db", 
        docker_image="semitechnologies/weaviate:1.22.4",
        api_endpoint="http://localhost:8080/v1",
        config_params={
            "hybrid_alpha": 0.7,  # Vector vs BM25F weight
            "hybrid_fusion_type": "rankedFusion",
            "vectorizer": "text2vec-openai",
            "properties": ["content"],
            "bm25f_properties": ["title^2", "content"]
        },
        vendor_documentation_url="https://docs.weaviate.io/weaviate/search/hybrid"
    ),
    
    "milvus": CompetitorConfig(
        name="milvus", 
        category="hybrid_vector_db",
        docker_image="milvusdb/milvus:v2.3.2",
        api_endpoint="http://localhost:19530",
        config_params={
            "hybrid_search_reranker": "WeightedRanker",
            "dense_weight": 0.7,
            "sparse_weight": 0.3,
            "embedding_model": "BAAI/bge-m3",
            "sparse_embedding_function": "BM25EmbeddingFunction"
        },
        vendor_documentation_url="https://milvus.io/docs/hybrid_search_with_milvus.md"
    ),
    
    "vespa": CompetitorConfig(
        name="vespa",
        category="hybrid_vector_db",
        docker_image="vespaengine/vespa:8.247.17", 
        api_endpoint="http://localhost:8080",
        config_params={
            "ranking_profile": "hybrid",
            "first_phase": "bm25(title) + bm25(content)",
            "second_phase": "closeness(field, embedding)",
            "match_features": ["bm25(title)", "bm25(content)", "closeness(field,embedding)"]
        },
        vendor_documentation_url="https://docs.vespa.ai/en/reference/bm25.html"
    ),
    
    "opensearch": CompetitorConfig(
        name="opensearch",
        category="hybrid_vector_db",
        docker_image="opensearchproject/opensearch:2.11.0",
        api_endpoint="http://localhost:9200", 
        config_params={
            "search_type": "hybrid",
            "query": {
                "hybrid": {
                    "queries": [
                        {"match": {"content": "{query}"}},
                        {"knn": {"field": "content_vector", "query_vector": "{embedding}", "k": 100}}
                    ]
                }
            },
            "normalization_technique": "min_max",
            "combination_technique": "arithmetic_mean"
        },
        vendor_documentation_url="https://docs.opensearch.org/latest/vector-search/"
    ),
    
    # Learned Sparse/Late-Interaction
    "splade_v2": CompetitorConfig(
        name="splade_v2",
        category="learned_sparse",
        docker_image="naver/splade:v2.2.0",
        api_endpoint="http://localhost:8081",
        config_params={
            "model_name": "naver/splade-cocondenser-ensembledistil",
            "sparse_type": "splade_pp",
            "agg": "max",
            "regularization_alpha": 0.0008
        },
        vendor_documentation_url="https://github.com/naver/splade"
    ),
    
    "colbert_v2": CompetitorConfig(
        name="colbert_v2", 
        category="learned_sparse",
        docker_image="stanford/colbert:v2.0",
        api_endpoint="http://localhost:8082",
        config_params={
            "model_name": "colbert-ir/colbertv2.0",
            "index_name": "collection",
            "similarity": "cosine",
            "query_maxlen": 32,
            "doc_maxlen": 180,
            "kmeans_niters": 4
        },
        vendor_documentation_url="https://github.com/stanford-futuredata/ColBERT"
    ),
    
    "ragatouille": CompetitorConfig(
        name="ragatouille",
        category="learned_sparse", 
        docker_image="bclavie/ragatouille:0.0.7",
        api_endpoint="http://localhost:8083",
        config_params={
            "model_name": "colbert-ir/colbertv2.0",
            "index_name": "benchmark_collection",
            "k": 100,
            "use_gpu": True
        },
        vendor_documentation_url="https://github.com/bclavie/RAGatouille"
    ),
    
    # Open Rerankers
    "bge_reranker_large": CompetitorConfig(
        name="bge_reranker_large",
        category="reranker",
        docker_image="flagopen/bge-reranker:large-v1.5",
        api_endpoint="http://localhost:8084",
        config_params={
            "model_name": "BAAI/bge-reranker-large", 
            "max_length": 512,
            "batch_size": 32,
            "normalize": True
        },
        vendor_documentation_url="https://huggingface.co/BAAI/bge-reranker-large"
    ),
    
    "bge_m3_reranker": CompetitorConfig(
        name="bge_m3_reranker",
        category="reranker",
        docker_image="flagopen/bge-m3:v1.0",
        api_endpoint="http://localhost:8085", 
        config_params={
            "model_name": "BAAI/bge-m3",
            "use_fp16": True,
            "max_passage_length": 8192,
            "weights_for_different_modes": [0.4, 0.2, 0.4]  # dense, sparse, colbert
        },
        vendor_documentation_url="https://huggingface.co/BAAI/bge-m3"
    ),
    
    "monot5": CompetitorConfig(
        name="monot5",
        category="reranker",
        docker_image="castorini/monot5:3b",
        api_endpoint="http://localhost:8086",
        config_params={
            "model_name": "castorini/monot5-3b-msmarco-10k",
            "tokenizer_name": "t5-3b", 
            "max_length": 512,
            "batch_size": 8
        },
        vendor_documentation_url="https://github.com/castorini/rank_llm"
    ),
    
    # Code Search & Graph  
    "zoekt": CompetitorConfig(
        name="zoekt",
        category="code_search",
        docker_image="sourcegraph/zoekt-indexserver:3.4.0",
        api_endpoint="http://localhost:8087",
        config_params={
            "index_type": "trigram",
            "max_matches": 1000,
            "shard_max_match": 100000,
            "index_options": {
                "large_files": True,
                "symbol_search": True,
                "ctags": True
            }
        },
        vendor_documentation_url="https://github.com/sourcegraph/zoekt"
    ),
    
    "livegrep": CompetitorConfig(
        name="livegrep", 
        category="code_search",
        docker_image="livegrep/livegrep:latest",
        api_endpoint="http://localhost:8088",
        config_params={
            "index_type": "codesearch",
            "max_matches": 1000,
            "context_lines": 3,
            "regex_timeout": 10
        },
        vendor_documentation_url="https://github.com/livegrep/livegrep"
    ),
    
    "graphrag": CompetitorConfig(
        name="graphrag",
        category="code_search", 
        docker_image="microsoft/graphrag:0.1.0",
        api_endpoint="http://localhost:8089",
        config_params={
            "llm_model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "chunk_size": 1200,
            "chunk_overlap": 100,
            "community_level": 2
        },
        vendor_documentation_url="https://microsoft.github.io/graphrag/"
    ),
    
    # Long-Context Algorithms
    "streaming_llm": CompetitorConfig(
        name="streaming_llm",
        category="long_context",
        docker_image="mit-han-lab/streaming-llm:latest", 
        api_endpoint="http://localhost:8090",
        config_params={
            "model_name": "meta-llama/Llama-2-7b-chat-hf",
            "attention_sink_size": 4,
            "recent_size": 2000, 
            "cache_size": 2048
        },
        vendor_documentation_url="https://github.com/mit-han-lab/streaming-llm"
    ),
    
    "longnet": CompetitorConfig(
        name="longnet",
        category="long_context",
        docker_image="microsoft/longnet:1.0",
        api_endpoint="http://localhost:8091",
        config_params={
            "model_name": "microsoft/DialoGPT-medium",
            "dilated_attention_pattern": [1, 2, 4, 8, 16, 32], 
            "segment_size": 2048,
            "max_position": 1000000
        },
        vendor_documentation_url="https://arxiv.org/pdf/2307.02486"
    ),
    
    "bge_m3_baseline": CompetitorConfig(
        name="bge_m3_baseline",
        category="long_context",
        docker_image="flagopen/bge-m3:v1.0",
        api_endpoint="http://localhost:8092",
        config_params={
            "model_name": "BAAI/bge-m3",
            "pooling_method": "cls",
            "normalize_embeddings": True,
            "max_length": 8192,
            "use_fp16": True
        },
        vendor_documentation_url="https://huggingface.co/BAAI/bge-m3"
    ),
    
    # Lethe baseline for comparison
    "lethe_hybrid": CompetitorConfig(
        name="lethe_hybrid",
        category="baseline",
        docker_image="local/lethe-hybrid:latest",
        api_endpoint="http://localhost:8094",
        config_params={
            "planning_strategy": "adaptive",
            "fusion_alpha": "dynamic",
            "diversification_enabled": True,
            "reranking_enabled": False,  # Per requirements
            "target_latency_ms": 200
        },
        vendor_documentation_url="https://github.com/lethe-research/lethe"
    )
}


# Dataset configurations for comprehensive evaluation
DATASET_CONFIGS = {
    # InfiniteBench Core
    "infinitebench_zh_qa": DatasetConfig(
        name="infinitebench_zh_qa",
        source="infinitebench",
        loader_class="benchmarks.datasets.infinitebench.ZhQALoader",
        data_path="data/infinitebench/zh_qa.jsonl",
        official_loader_url="https://github.com/OpenBMB/InfiniteBench",
        multilingual=True,
        expected_fields=["query", "context", "answer", "length"]
    ),
    
    "infinitebench_retrieve_passkey": DatasetConfig(
        name="infinitebench_retrieve_passkey", 
        source="infinitebench",
        loader_class="benchmarks.datasets.infinitebench.RetrievePasskeyLoader",
        data_path="data/infinitebench/retrieve_passkey.jsonl",
        official_loader_url="https://github.com/OpenBMB/InfiniteBench",
        expected_fields=["query", "context", "passkey", "length"]
    ),
    
    "infinitebench_retrieve_kv": DatasetConfig(
        name="infinitebench_retrieve_kv",
        source="infinitebench", 
        loader_class="benchmarks.datasets.infinitebench.RetrieveKVLoader",
        data_path="data/infinitebench/retrieve_kv.jsonl",
        official_loader_url="https://github.com/OpenBMB/InfiniteBench",
        expected_fields=["query", "context", "key", "value", "length"]
    ),
    
    "infinitebench_retrieve_number": DatasetConfig(
        name="infinitebench_retrieve_number",
        source="infinitebench",
        loader_class="benchmarks.datasets.infinitebench.RetrieveNumberLoader", 
        data_path="data/infinitebench/retrieve_number.jsonl",
        official_loader_url="https://github.com/OpenBMB/InfiniteBench",
        expected_fields=["query", "context", "number", "length"]
    ),
    
    "infinitebench_code_debug": DatasetConfig(
        name="infinitebench_code_debug",
        source="infinitebench",
        loader_class="benchmarks.datasets.infinitebench.CodeDebugLoader",
        data_path="data/infinitebench/code_debug.jsonl", 
        official_loader_url="https://github.com/OpenBMB/InfiniteBench",
        expected_fields=["query", "context", "error", "fix", "length"]
    ),
    
    "infinitebench_code_qa": DatasetConfig(
        name="infinitebench_code_qa",
        source="infinitebench",
        loader_class="benchmarks.datasets.infinitebench.CodeQALoader",
        data_path="data/infinitebench/code_qa.jsonl",
        official_loader_url="https://github.com/OpenBMB/InfiniteBench", 
        expected_fields=["query", "context", "answer", "length"]
    ),
    
    "infinitebench_en_qa": DatasetConfig(
        name="infinitebench_en_qa",
        source="infinitebench",
        loader_class="benchmarks.datasets.infinitebench.EnQALoader",
        data_path="data/infinitebench/en_qa.jsonl",
        official_loader_url="https://github.com/OpenBMB/InfiniteBench",
        expected_fields=["query", "context", "answer", "length"]
    ),
    
    # External Stress Testing
    "ruler": DatasetConfig(
        name="ruler",
        source="ruler", 
        loader_class="benchmarks.datasets.ruler.RulerLoader",
        data_path="data/ruler/tasks.jsonl",
        official_loader_url="https://github.com/NVIDIA/RULER",
        expected_fields=["query", "context", "answer", "task_type", "length"]
    ),
    
    "longbench_v2": DatasetConfig(
        name="longbench_v2",
        source="longbench",
        loader_class="benchmarks.datasets.longbench.LongBenchV2Loader", 
        data_path="data/longbench_v2/tasks.jsonl",
        official_loader_url="https://github.com/THUDM/LongBench",
        expected_fields=["query", "context", "answer", "category", "length"]
    ),
    
    "babilong": DatasetConfig(
        name="babilong", 
        source="babilong",
        loader_class="benchmarks.datasets.babilong.BABILongLoader",
        data_path="data/babilong/tasks.jsonl",
        official_loader_url="https://github.com/booydar/babilong",
        expected_fields=["query", "context", "answer", "task_id", "length"]
    )
}


def get_default_config() -> BenchmarkConfig:
    """Get default benchmark configuration."""
    return BenchmarkConfig(
        run_name="comprehensive_benchmark_default",
        experiment_tags=["comprehensive", "fair-evaluation", "production"],
        log_level="INFO",
        max_workers=4
    )


def get_competitor_config(name: str) -> CompetitorConfig:
    """Get configuration for a specific competitor."""
    if name not in COMPETITOR_CONFIGS:
        raise ValueError(f"Unknown competitor: {name}. Available: {list(COMPETITOR_CONFIGS.keys())}")
    return COMPETITOR_CONFIGS[name]


def get_dataset_config(name: str) -> DatasetConfig:
    """Get configuration for a specific dataset."""
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[name]


def list_competitors_by_category() -> Dict[str, List[str]]:
    """List all competitors organized by category."""
    categories = {}
    for name, config in COMPETITOR_CONFIGS.items():
        if config.category not in categories:
            categories[config.category] = []
        categories[config.category].append(name)
    return categories


def list_datasets_by_source() -> Dict[str, List[str]]:
    """List all datasets organized by source."""
    sources = {}
    for name, config in DATASET_CONFIGS.items():
        if config.source not in sources:
            sources[config.source] = []
        sources[config.source].append(name)
    return sources