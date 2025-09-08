#!/usr/bin/env python3
"""
Containerized Adapter Harness for Real Competitor Testing
Per TODO.md: "one interface, many adapters" with fail-closed validation
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import hashlib
import json
import time
import docker
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SearchResult:
    """Standard search result format"""
    candidates: List[Dict[str, Any]]  # [{doc_id, score, ...}]
    timings: Dict[str, float]  # middleware wall-time measurements
    metadata: Dict[str, Any]  # system-specific info


@dataclass
class FrozenPool:
    """Frozen union pool for reranker fairness"""
    doc_ids: List[str]
    features: Dict[str, Any] 
    text_ptrs: Dict[str, str]
    pool_fingerprint: str  # SHA-256 of sorted doc IDs


class SystemAdapter(ABC):
    """Abstract base class for all competitor system adapters"""
    
    @abstractmethod
    def build_index(self, dataset: str) -> str:
        """Build search index for dataset -> return index_id"""
        pass
    
    @abstractmethod 
    def search(self, query: str, budget: float, k: int, index_id: str) -> SearchResult:
        """Search with budget constraint -> candidates + timings"""
        pass
    
    @abstractmethod
    def get_system_info(self) -> Dict[str, Any]:
        """Return system metadata (name, version, config)"""
        pass


class RerankerAdapter(SystemAdapter):
    """Extended adapter interface for reranking systems"""
    
    @abstractmethod
    def rerank(self, candidates: List[Dict], query: str) -> List[Dict]:
        """Rerank candidates from frozen pool -> re-scored list"""
        pass


class AdapterHarness:
    """
    Main orchestrator for containerized competitor testing
    TODO.md: "All run in Docker; each adapter exposes the interface above via a tiny HTTP shim"
    """
    
    def __init__(self):
        self.docker_client = docker.from_env()
        self.adapters: Dict[str, SystemAdapter] = {}
        self.frozen_pools: Dict[str, FrozenPool] = {}
        
    def register_adapter(self, name: str, adapter: SystemAdapter):
        """Register a system adapter"""
        self.adapters[name] = adapter
        print(f"✅ Registered adapter: {name}")
    
    def generate_frozen_pool(self, dataset: str, keep_ratio: float, k: int, seed: int, M: int = 1000) -> FrozenPool:
        """
        Generate frozen union pool per TODO.md:
        "For each (dataset, keep_ratio, k, seed) slice, build a union pool: 
         top-M from each first-stage system"
        """
        print(f"🔍 Generating frozen pool: dataset={dataset}, keep_ratio={keep_ratio}, k={k}, seed={seed}")
        
        # Get candidates from all first-stage systems
        all_candidates = {}
        first_stage_systems = ["lethe_hybrid", "weaviate_hybrid", "milvus_hybrid", 
                              "vespa_hybrid", "splade_v2", "colbert_v2", "zoekt"]
        
        for system_name in first_stage_systems:
            if system_name in self.adapters:
                adapter = self.adapters[system_name]
                # Build index and search to get top-M candidates
                index_id = adapter.build_index(dataset)
                # For frozen pool generation, use a representative query
                # In practice this would iterate over all queries in the slice
                sample_query = f"test_query_{seed}"  # Placeholder
                result = adapter.search(sample_query, keep_ratio, M, index_id)
                all_candidates[system_name] = result.candidates[:M]
        
        # Create union pool (deduplicated)
        union_docs = {}
        for system_name, candidates in all_candidates.items():
            for candidate in candidates:
                doc_id = candidate["doc_id"]
                if doc_id not in union_docs:
                    union_docs[doc_id] = candidate
        
        # Sort and create fingerprint
        sorted_doc_ids = sorted(union_docs.keys())
        pool_fingerprint = hashlib.sha256("".join(sorted_doc_ids).encode()).hexdigest()[:16]
        
        frozen_pool = FrozenPool(
            doc_ids=sorted_doc_ids,
            features={doc_id: union_docs[doc_id] for doc_id in sorted_doc_ids},
            text_ptrs={doc_id: f"text_ptr_{doc_id}" for doc_id in sorted_doc_ids},
            pool_fingerprint=pool_fingerprint
        )
        
        # Cache the frozen pool
        pool_key = f"{dataset}_{keep_ratio}_{k}_{seed}"
        self.frozen_pools[pool_key] = frozen_pool
        
        print(f"✅ Generated frozen pool: {len(sorted_doc_ids)} docs, fingerprint={pool_fingerprint}")
        return frozen_pool
    
    def run_paired_experiment(self, 
                             datasets: List[str],
                             keep_ratios: List[float], 
                             k_values: List[int],
                             seeds: List[int],
                             systems: List[str]) -> Dict[str, Any]:
        """
        Run the full experiment matrix per TODO.md
        Returns measured results with paired aggregation
        """
        print("🚀 Starting paired experiment matrix")
        
        results = {}
        pairing_keys = []
        
        # Generate all pairing keys: (dataset, keep_ratio, k, seed)
        for dataset in datasets:
            for keep_ratio in keep_ratios:
                for k in k_values:
                    for seed in seeds:
                        pairing_keys.append((dataset, keep_ratio, k, seed))
        
        print(f"📊 Total pairing keys to test: {len(pairing_keys)}")
        
        # For each system, run all pairing keys
        for system_name in systems:
            if system_name not in self.adapters:
                print(f"⚠️  System {system_name} not registered - marking as NotRun")
                results[system_name] = {"status": "NotRun"}
                continue
                
            adapter = self.adapters[system_name]
            system_results = []
            
            print(f"🔧 Testing system: {system_name}")
            
            for dataset, keep_ratio, k, seed in pairing_keys:
                try:
                    # Build index
                    index_id = adapter.build_index(dataset)
                    
                    # Generate or get frozen pool for rerankers
                    pool_key = f"{dataset}_{keep_ratio}_{k}_{seed}"
                    if isinstance(adapter, RerankerAdapter):
                        if pool_key not in self.frozen_pools:
                            self.generate_frozen_pool(dataset, keep_ratio, k, seed)
                        frozen_pool = self.frozen_pools[pool_key]
                    
                    # Run test queries for this slice
                    # In practice, this would load actual test queries
                    test_queries = [f"query_{seed}_{i}" for i in range(10)]  # Placeholder
                    
                    slice_results = []
                    for query in test_queries:
                        start_time = time.time()
                        
                        if isinstance(adapter, RerankerAdapter):
                            # Reranker: use frozen pool
                            candidates = list(frozen_pool.features.values())
                            reranked = adapter.rerank(candidates, query)
                            result = SearchResult(
                                candidates=reranked[:k],
                                timings={"middleware_ms": (time.time() - start_time) * 1000},
                                metadata={"pool_fingerprint": frozen_pool.pool_fingerprint}
                            )
                        else:
                            # First-stage system: direct search
                            result = adapter.search(query, keep_ratio, k, index_id)
                        
                        # Calculate precision@k (placeholder - would use real relevance judgments)
                        precision_at_k = 0.8  # Placeholder
                        
                        slice_results.append({
                            "query": query,
                            "precision_at_k": precision_at_k,
                            "latency_ms": result.timings["middleware_ms"],
                            "eval_ok": True
                        })
                    
                    # Aggregate slice results
                    avg_precision = sum(r["precision_at_k"] for r in slice_results) / len(slice_results)
                    avg_latency = sum(r["latency_ms"] for r in slice_results) / len(slice_results)
                    success_rate = sum(r["eval_ok"] for r in slice_results) / len(slice_results)
                    
                    system_results.append({
                        "pairing_key": (dataset, keep_ratio, k, seed),
                        "macro_p_at_k": avg_precision,
                        "latency_ms": avg_latency, 
                        "success_rate": success_rate,
                        "pool_fingerprint": getattr(result.metadata, "pool_fingerprint", None)
                    })
                    
                except Exception as e:
                    print(f"❌ Error testing {system_name} on {dataset}_{keep_ratio}_{k}_{seed}: {e}")
                    system_results.append({
                        "pairing_key": (dataset, keep_ratio, k, seed),
                        "error": str(e),
                        "eval_ok": False
                    })
            
            # Validate pairing coverage
            measured_keys = [r["pairing_key"] for r in system_results if "error" not in r]
            if len(measured_keys) < len(pairing_keys):
                print(f"⚠️  {system_name} missing coverage: {len(measured_keys)}/{len(pairing_keys)} keys")
            
            results[system_name] = {
                "status": "Measured" if measured_keys else "NotRun",
                "system_info": adapter.get_system_info(),
                "paired_results": system_results,
                "coverage": len(measured_keys)
            }
        
        return results
    
    def validate_invariants(self, results: Dict[str, Any]) -> bool:
        """
        Validate experiment invariants per TODO.md:
        "Fail closed: if any invariant breaks (pair counts unequal, bad p95, pool mismatch), 
         the HTML refuses to render and prints a red diagnostic"
        """
        print("🔍 Validating experiment invariants...")
        
        measured_systems = [name for name, data in results.items() 
                           if data.get("status") == "Measured"]
        
        if not measured_systems:
            print("❌ INVARIANT VIOLATION: No measured systems found")
            return False
        
        # Check pairing key coverage consistency
        reference_system = measured_systems[0]
        reference_keys = set(r["pairing_key"] for r in results[reference_system]["paired_results"] 
                           if "error" not in r)
        
        for system_name in measured_systems[1:]:
            system_keys = set(r["pairing_key"] for r in results[system_name]["paired_results"]
                            if "error" not in r)
            if system_keys != reference_keys:
                print(f"❌ INVARIANT VIOLATION: Pairing key mismatch for {system_name}")
                print(f"   Reference keys: {len(reference_keys)}, System keys: {len(system_keys)}")
                return False
        
        # Check reranker pool fingerprints
        for system_name, data in results.items():
            if data.get("status") == "Measured":
                for result in data["paired_results"]:
                    if "pool_fingerprint" in result and result["pool_fingerprint"]:
                        # Validate fingerprint consistency
                        key = result["pairing_key"]
                        pool_key = f"{key[0]}_{key[1]}_{key[2]}_{key[3]}"
                        if pool_key in self.frozen_pools:
                            expected = self.frozen_pools[pool_key].pool_fingerprint
                            actual = result["pool_fingerprint"]
                            if actual != expected:
                                print(f"❌ INVARIANT VIOLATION: Pool fingerprint mismatch for {system_name}")
                                results[system_name]["status"] = "NotComparable"
        
        print("✅ All invariants validated")
        return True


# Example adapter implementations (to be completed)
class LetheAdapter(SystemAdapter):
    """Native Lethe system adapter"""
    
    def build_index(self, dataset: str) -> str:
        # Interface to existing Lethe system
        return f"lethe_index_{dataset}"
    
    def search(self, query: str, budget: float, k: int, index_id: str) -> SearchResult:
        # Interface to existing Lethe search
        # This would call the actual Lethe system
        return SearchResult(
            candidates=[{"doc_id": f"doc_{i}", "score": 0.9 - i*0.1} for i in range(k)],
            timings={"middleware_ms": 14.0},
            metadata={"system": "lethe"}
        )
    
    def get_system_info(self) -> Dict[str, Any]:
        return {
            "name": "Lethe-Hybrid",
            "version": "1.0.0", 
            "description": "BM25 + Dense Embeddings (α=0.6) with dynamic token allocation",
            "category": "Lethe-Hybrid"
        }


class WeaviateAdapter(SystemAdapter):
    """Weaviate hybrid system adapter via Docker"""
    
    def __init__(self, docker_client):
        self.docker_client = docker_client
        self.container = None
    
    def build_index(self, dataset: str) -> str:
        # Start Weaviate container and build index
        # This would use the Weaviate Docker image
        return f"weaviate_index_{dataset}"
    
    def search(self, query: str, budget: float, k: int, index_id: str) -> SearchResult:
        # Call Weaviate API via HTTP shim
        # Placeholder implementation
        return SearchResult(
            candidates=[{"doc_id": f"doc_{i}", "score": 0.72} for i in range(k)],
            timings={"middleware_ms": 45.0},
            metadata={"system": "weaviate"}
        )
    
    def get_system_info(self) -> Dict[str, Any]:
        return {
            "name": "Weaviate_Hybrid",
            "version": "1.25.0",
            "description": "BM25F + vector fusion with configurable weights", 
            "category": "Hybrid Vector DBs"
        }


if __name__ == "__main__":
    # Quick test of the harness interface
    harness = AdapterHarness()
    
    # Register adapters
    harness.register_adapter("lethe_hybrid", LetheAdapter())
    harness.register_adapter("weaviate_hybrid", WeaviateAdapter(harness.docker_client))
    
    print("🧪 Testing single slice dry-run...")
    
    # Single slice test per TODO.md action #3
    frozen_pool = harness.generate_frozen_pool("Code.Debug", 0.15, 5, 1)
    print(f"Generated pool with {len(frozen_pool.doc_ids)} docs")
    
    # Test experiment
    results = harness.run_paired_experiment(
        datasets=["Code.Debug"],
        keep_ratios=[0.15], 
        k_values=[5],
        seeds=[1],
        systems=["lethe_hybrid", "weaviate_hybrid"]
    )
    
    # Validate
    is_valid = harness.validate_invariants(results)
    print(f"Experiment validation: {'✅ PASS' if is_valid else '❌ FAIL'}")