#!/usr/bin/env python3
"""
Test Script for Production IR System

Validates the implementation of BM25 and ANN retrievers with
sample data to ensure everything works correctly.

Usage:
    python test_ir_system.py
"""

import os
import sys
import json
import tempfile
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retriever import (
    RetrieverConfig,
    TimingHarness,
    BM25IndexBuilder,
    ANNIndexBuilder, 
    DenseEmbeddingManager,
    create_bm25_retriever,
    create_ann_retriever,
    HNSWConfig,
    IVFPQConfig,
    EmbeddingConfig
)

def create_sample_documents(n=1000):
    """Create sample documents for testing."""
    documents = []
    
    topics = [
        "machine learning algorithms and deep neural networks",
        "information retrieval systems and search engines", 
        "natural language processing and text analysis",
        "computer vision and image recognition techniques",
        "data mining and knowledge discovery methods",
        "artificial intelligence and automated reasoning",
        "database systems and query optimization",
        "distributed computing and parallel processing",
        "software engineering and system design",
        "cybersecurity and network protection"
    ]
    
    for i in range(n):
        topic = topics[i % len(topics)]
        doc = {
            "id": f"doc_{i:04d}",
            "text": f"{topic} example document {i}. This document contains relevant information about {topic.split()[0]} and related concepts."
        }
        documents.append(doc)
        
    return documents

def create_sample_queries():
    """Create sample queries for testing."""
    return [
        "machine learning neural networks",
        "information retrieval search",
        "natural language processing",
        "computer vision recognition", 
        "data mining knowledge",
        "artificial intelligence reasoning",
        "database query optimization",
        "distributed parallel computing",
        "software system design",
        "cybersecurity network security"
    ]

def test_timing_harness():
    """Test the timing harness functionality."""
    print("Testing timing harness...")
    
    harness = TimingHarness(cold_cycles=5, warm_cycles=10)
    
    def dummy_operation():
        # Simulate some work
        x = sum(i**2 for i in range(1000))
        return x
        
    # Test single measurement
    with harness.measure("dummy_op"):
        dummy_operation()
        
    # Test benchmark
    profile = harness.benchmark_function(dummy_operation, "dummy_benchmark")
    
    assert profile.count > 0
    assert profile.latency_p50 > 0
    assert profile.throughput > 0
    
    print(f"✓ Timing harness: {profile.count} measurements, "
          f"p95={profile.latency_p95:.2f}ms, "
          f"throughput={profile.throughput:.1f} ops/sec")

def test_embeddings():
    """Test dense embedding generation."""
    print("Testing embeddings...")
    
    config = EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=8,
        device="cpu",  # Force CPU for testing
        cache_embeddings=False  # Disable caching for test
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = DenseEmbeddingManager(config, cache_dir=temp_dir)
        
        # Test text encoding
        texts = ["hello world", "machine learning", "information retrieval"]
        embeddings = manager.encode_texts(texts, show_progress=False)
        
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] > 0
        assert isinstance(embeddings, np.ndarray)
        
        print(f"✓ Embeddings: {embeddings.shape} vectors generated")
        
        # Test collection encoding
        documents = [{"text": text, "id": f"doc_{i}"} for i, text in enumerate(texts)]
        coll_embeddings, metadata = manager.encode_collection(
            documents, "test_collection", force_recompute=True
        )
        
        assert coll_embeddings.shape == embeddings.shape
        assert metadata.num_embeddings == 3
        
        print(f"✓ Collection encoding: {metadata.num_embeddings} docs, "
              f"dim={metadata.embedding_dim}")

def test_bm25_index():
    """Test BM25 index building and retrieval."""
    print("Testing BM25 index...")
    
    # Create sample data
    documents = create_sample_documents(100)
    queries = create_sample_queries()[:3]  # Just a few for testing
    
    with tempfile.TemporaryDirectory() as temp_dir:
        index_dir = Path(temp_dir) / "bm25_test"
        
        # Build index
        builder = BM25IndexBuilder(config=RetrieverConfig().bm25)
        metadata = builder.build_index(
            documents=documents,
            index_path=index_dir, 
            dataset_name="test_dataset"
        )
        
        assert metadata.stats.num_documents == 100
        assert index_dir.exists()
        
        print(f"✓ BM25 build: {metadata.stats.num_documents} docs, "
              f"{metadata.stats.index_size_mb:.1f}MB")
        
        # Test retrieval
        retriever = create_bm25_retriever(index_dir)
        
        for query in queries:
            results = retriever.search(query, k=10)
            assert len(results) > 0
            assert all(hasattr(r, 'doc_id') and hasattr(r, 'score') for r in results)
            
        print(f"✓ BM25 search: {len(queries)} queries processed")

def test_ann_index():
    """Test ANN index building and retrieval."""
    print("Testing ANN index...")
    
    # Create sample data
    documents = create_sample_documents(100)
    queries = create_sample_queries()[:3]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate embeddings
        config = EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2",
                               device="cpu", batch_size=8)
        manager = DenseEmbeddingManager(config, cache_dir=temp_dir)
        
        vectors, metadata = manager.encode_collection(documents, "test_vectors")
        doc_ids = [doc["id"] for doc in documents]
        
        print(f"✓ Generated embeddings: {vectors.shape}")
        
        # Test HNSW
        hnsw_dir = Path(temp_dir) / "hnsw_test"
        hnsw_config = HNSWConfig(m=8, ef_construction=50)  # Small for testing
        hnsw_builder = ANNIndexBuilder("hnsw", hnsw_config)
        
        hnsw_metadata = hnsw_builder.build_index(
            vectors=vectors,
            doc_ids=doc_ids,
            index_path=hnsw_dir,
            dataset_name="test_dataset"
        )
        
        assert hnsw_metadata.stats.num_documents == 100
        assert hnsw_dir.exists()
        
        print(f"✓ HNSW build: {hnsw_metadata.stats.num_documents} vectors, "
              f"{hnsw_metadata.stats.index_size_mb:.1f}MB")
        
        # Test HNSW retrieval
        hnsw_retriever = create_ann_retriever(hnsw_dir, "hnsw")
        
        query_vectors = manager.encode_texts(queries, show_progress=False)
        results = hnsw_retriever.search(query_vectors, k=10)
        
        assert len(results) == len(queries)
        assert all(len(query_results) > 0 for query_results in results)
        
        print(f"✓ HNSW search: {len(queries)} queries, "
              f"{len(results[0])} results per query")

def test_configuration():
    """Test configuration loading and validation."""
    print("Testing configuration...")
    
    # Test default configuration
    config = RetrieverConfig()
    errors = config.validate()
    
    assert len(errors) == 0, f"Default config validation failed: {errors}"
    
    # Test configuration serialization
    config_dict = config.to_dict()
    config2 = RetrieverConfig.from_dict(config_dict)
    
    assert config.bm25.k1 == config2.bm25.k1
    assert config.hnsw.m == config2.hnsw.m
    assert config.system.cpu_cores == config2.system.cpu_cores
    
    print("✓ Configuration: validation and serialization working")

def run_all_tests():
    """Run all tests."""
    print("🧪 Running IR System Tests")
    print("=" * 50)
    
    try:
        test_timing_harness()
        test_configuration()
        test_embeddings() 
        test_bm25_index()
        
        # Skip ANN test if FAISS not available
        try:
            import faiss
            test_ann_index()
        except ImportError:
            print("⚠️  Skipping ANN tests (FAISS not available)")
            
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)