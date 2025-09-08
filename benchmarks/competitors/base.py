#!/usr/bin/env python3
"""
Base Competitor Interface
=========================

Provides consistent interfaces for all competitor systems with
standardized result formats and performance monitoring.
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import requests
import docker
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CompetitorResult:
    """Standardized result from a competitor system."""
    
    # Core retrieval results
    doc_ids: List[str]
    scores: List[float]
    
    # Performance metrics
    latency_ms: float
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Quality metrics
    tokens_retrieved: int = 0
    exact_matches: int = 0
    
    # Budget compliance
    keep_ratio: float = 0.0
    tokens_kept: int = 0
    original_context_tokens: int = 0
    
    # System metadata
    competitor_name: str = ""
    config_params: Dict[str, Any] = field(default_factory=dict)
    api_version: str = ""
    
    # Error information
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate and compute derived fields."""
        if self.doc_ids and self.scores:
            if len(self.doc_ids) != len(self.scores):
                raise ValueError(f"Mismatched doc_ids ({len(self.doc_ids)}) and scores ({len(self.scores)})")
        
        # Compute keep ratio if not provided
        if self.original_context_tokens > 0 and self.tokens_kept > 0:
            self.keep_ratio = self.tokens_kept / self.original_context_tokens
    
    @property
    def result_count(self) -> int:
        """Number of results returned."""
        return len(self.doc_ids)
    
    @property
    def mean_score(self) -> float:
        """Mean relevance score."""
        return sum(self.scores) / len(self.scores) if self.scores else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'doc_ids': self.doc_ids,
            'scores': self.scores,
            'latency_ms': self.latency_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'tokens_retrieved': self.tokens_retrieved,
            'exact_matches': self.exact_matches,
            'keep_ratio': self.keep_ratio,
            'tokens_kept': self.tokens_kept,
            'original_context_tokens': self.original_context_tokens,
            'competitor_name': self.competitor_name,
            'config_params': self.config_params,
            'api_version': self.api_version,
            'success': self.success,
            'error_message': self.error_message,
            'result_count': self.result_count,
            'mean_score': self.mean_score
        }


@dataclass 
class CompetitorMetrics:
    """Performance and quality metrics for a competitor."""
    
    # Latency metrics
    mean_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Throughput metrics  
    queries_per_second: float
    
    # Resource usage
    mean_memory_mb: float
    peak_memory_mb: float
    mean_cpu_percent: float
    
    # Quality metrics
    mean_exact_matches: float
    mean_keep_ratio: float
    success_rate: float
    
    # System stability
    error_count: int
    timeout_count: int
    total_queries: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'mean_latency_ms': self.mean_latency_ms,
            'p95_latency_ms': self.p95_latency_ms,
            'p99_latency_ms': self.p99_latency_ms,
            'queries_per_second': self.queries_per_second,
            'mean_memory_mb': self.mean_memory_mb,
            'peak_memory_mb': self.peak_memory_mb,
            'mean_cpu_percent': self.mean_cpu_percent,
            'mean_exact_matches': self.mean_exact_matches,
            'mean_keep_ratio': self.mean_keep_ratio,
            'success_rate': self.success_rate,
            'error_count': self.error_count,
            'timeout_count': self.timeout_count,
            'total_queries': self.total_queries
        }


class BaseCompetitor(ABC):
    """Base class for all competitor implementations."""
    
    def __init__(
        self,
        name: str,
        api_endpoint: str,
        config_params: Dict[str, Any],
        timeout_seconds: int = 300,
        max_retries: int = 3
    ):
        """Initialize base competitor."""
        self.name = name
        self.api_endpoint = api_endpoint
        self.config_params = config_params.copy()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        
        # Performance tracking
        self.results_cache: List[CompetitorResult] = []
        self.docker_client = docker.from_env()
        
        logger.info(f"Initialized competitor: {name}")
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        context: str,
        keep_ratio: float,
        k: int = 100
    ) -> CompetitorResult:
        """Execute retrieval with budget constraints."""
        pass
    
    @abstractmethod 
    def health_check(self) -> bool:
        """Check if competitor system is healthy and responding."""
        pass
    
    @abstractmethod
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and version details."""
        pass
    
    def start_container(self, docker_image: str, ports: Dict[str, int] = None) -> str:
        """Start Docker container for competitor system."""
        try:
            # Check if container already running
            existing_containers = self.docker_client.containers.list(
                filters={"label": f"competitor={self.name}"}
            )
            
            for container in existing_containers:
                if container.status == "running":
                    logger.info(f"Container for {self.name} already running: {container.id[:12]}")
                    return container.id
                else:
                    logger.info(f"Removing stale container: {container.id[:12]}")
                    container.remove(force=True)
            
            # Start new container
            logger.info(f"Starting container for {self.name}: {docker_image}")
            
            container_config = {
                "image": docker_image,
                "labels": {"competitor": self.name, "benchmark": "lethe"},
                "detach": True,
                "name": f"benchmark_{self.name}",
                "remove": False,  # Keep for debugging
                "mem_limit": "8g",
                "cpus": "4.0"
            }
            
            if ports:
                container_config["ports"] = ports
            
            # Add competitor-specific environment variables
            env_vars = self._get_container_env()
            if env_vars:
                container_config["environment"] = env_vars
            
            container = self.docker_client.containers.run(**container_config)
            
            # Wait for container to be ready
            self._wait_for_ready(container.id, timeout=120)
            
            logger.info(f"Container started successfully: {container.id[:12]}")
            return container.id
            
        except Exception as e:
            logger.error(f"Failed to start container for {self.name}: {e}")
            raise
    
    def stop_container(self):
        """Stop and remove competitor container."""
        try:
            containers = self.docker_client.containers.list(
                filters={"label": f"competitor={self.name}"}
            )
            
            for container in containers:
                logger.info(f"Stopping container: {container.id[:12]}")
                container.stop(timeout=30)
                container.remove()
                
        except Exception as e:
            logger.warning(f"Error stopping container for {self.name}: {e}")
    
    def _get_container_env(self) -> Dict[str, str]:
        """Get environment variables for container."""
        # Override in subclasses for system-specific env vars
        return {}
    
    def _wait_for_ready(self, container_id: str, timeout: int = 120):
        """Wait for container to be ready to accept requests."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if self.health_check():
                    logger.info(f"Container {container_id[:12]} is ready")
                    return
            except Exception:
                pass
            
            time.sleep(2)
        
        raise TimeoutError(f"Container {container_id[:12]} not ready after {timeout}s")
    
    def _make_request(
        self,
        endpoint: str,
        method: str = "POST",
        json_data: Dict[str, Any] = None,
        params: Dict[str, Any] = None
    ) -> requests.Response:
        """Make HTTP request with retries and error handling."""
        url = f"{self.api_endpoint.rstrip('/')}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    json=json_data,
                    params=params,
                    timeout=self.timeout_seconds,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    return response
                elif response.status_code >= 500 and attempt < self.max_retries:
                    logger.warning(f"Server error {response.status_code}, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.Timeout:
                if attempt < self.max_retries:
                    logger.warning(f"Request timeout, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    logger.warning(f"Request error: {e}, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise
        
        raise RuntimeError(f"Max retries exceeded for {url}")
    
    def _compute_budget_tokens(self, context: str, keep_ratio: float) -> int:
        """Compute token budget from context and keep ratio."""
        # Simple whitespace tokenization for cross-system consistency
        total_tokens = len(context.split())
        budget_tokens = int(total_tokens * keep_ratio)
        return max(budget_tokens, 1)  # Always keep at least 1 token
    
    def _extract_top_k_tokens(self, context: str, doc_ids: List[str], k_tokens: int) -> Tuple[str, int]:
        """Extract top k tokens from context based on retrieved documents."""
        # Simple implementation - take first k_tokens from context
        # In practice, would use document boundaries and relevance scores
        tokens = context.split()
        if len(tokens) <= k_tokens:
            return context, len(tokens)
        
        # Take first k_tokens 
        selected_tokens = tokens[:k_tokens]
        extracted_context = " ".join(selected_tokens)
        
        return extracted_context, len(selected_tokens)
    
    def compute_metrics(self) -> CompetitorMetrics:
        """Compute performance metrics from cached results."""
        if not self.results_cache:
            raise ValueError("No results available for metrics computation")
        
        successful_results = [r for r in self.results_cache if r.success]
        
        if not successful_results:
            # All queries failed
            return CompetitorMetrics(
                mean_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                queries_per_second=0.0,
                mean_memory_mb=0.0,
                peak_memory_mb=0.0,
                mean_cpu_percent=0.0,
                mean_exact_matches=0.0,
                mean_keep_ratio=0.0,
                success_rate=0.0,
                error_count=len(self.results_cache),
                timeout_count=0,
                total_queries=len(self.results_cache)
            )
        
        # Compute latency percentiles
        latencies = [r.latency_ms for r in successful_results]
        latencies.sort()
        
        n = len(latencies)
        p95_idx = int(0.95 * n)
        p99_idx = int(0.99 * n)
        
        # Compute other metrics
        mean_latency = sum(latencies) / n
        p95_latency = latencies[p95_idx] if p95_idx < n else latencies[-1]
        p99_latency = latencies[p99_idx] if p99_idx < n else latencies[-1]
        
        mean_memory = sum(r.memory_usage_mb for r in successful_results) / n
        peak_memory = max(r.memory_usage_mb for r in successful_results) if successful_results else 0.0
        mean_cpu = sum(r.cpu_usage_percent for r in successful_results) / n
        
        mean_exact_matches = sum(r.exact_matches for r in successful_results) / n
        mean_keep_ratio = sum(r.keep_ratio for r in successful_results) / n
        
        success_rate = len(successful_results) / len(self.results_cache)
        error_count = len(self.results_cache) - len(successful_results)
        
        # Estimate QPS (rough approximation)
        total_time_sec = sum(r.latency_ms for r in successful_results) / 1000.0
        queries_per_second = n / total_time_sec if total_time_sec > 0 else 0.0
        
        return CompetitorMetrics(
            mean_latency_ms=mean_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            queries_per_second=queries_per_second,
            mean_memory_mb=mean_memory,
            peak_memory_mb=peak_memory,
            mean_cpu_percent=mean_cpu,
            mean_exact_matches=mean_exact_matches,
            mean_keep_ratio=mean_keep_ratio,
            success_rate=success_rate,
            error_count=error_count,
            timeout_count=0,  # Could track this separately
            total_queries=len(self.results_cache)
        )
    
    def clear_cache(self):
        """Clear results cache."""
        self.results_cache.clear()
        logger.debug(f"Cleared results cache for {self.name}")
    
    def get_cached_results(self) -> List[CompetitorResult]:
        """Get all cached results."""
        return self.results_cache.copy()
    
    def __enter__(self):
        """Context manager entry - could start containers here."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.stop_container()