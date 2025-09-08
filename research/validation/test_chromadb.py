#!/usr/bin/env python3
"""
ChromaDB Comprehensive Testing and Validation Script

This script thoroughly tests ChromaDB functionality to ensure it works correctly
before running the semantic search benchmarks.

Tests include:
1. Basic ChromaDB installation and import
2. Client initialization and collection management
3. Document embedding and storage
4. Query functionality and similarity search
5. Metadata handling and filtering
6. Persistence and data integrity
7. Performance benchmarking with small dataset
"""

import logging
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChromaDBTester:
    """Comprehensive ChromaDB testing and validation."""
    
    def __init__(self):
        self.temp_dir = None
        self.client = None
        self.sentence_model = None
        self.test_results = {}
        
    def setup(self):
        """Set up test environment."""
        logger.info("Setting up ChromaDB test environment...")
        
        # Create temporary directory for testing
        self.temp_dir = Path(tempfile.mkdtemp(prefix="chromadb_test_"))
        logger.info(f"Using temporary directory: {self.temp_dir}")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.temp_dir))
        
        # Initialize sentence transformer
        logger.info("Loading sentence transformer model...")
        self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        
        logger.info("Setup completed successfully")
        
    def test_basic_functionality(self) -> bool:
        """Test basic ChromaDB operations."""
        logger.info("Testing basic ChromaDB functionality...")
        
        try:
            # Create collection
            collection = self.client.create_collection(
                name="test_basic",
                metadata={"description": "Basic functionality test"}
            )
            
            # Add documents
            test_docs = [
                "The quick brown fox jumps over the lazy dog.",
                "Artificial intelligence is transforming modern technology.",
                "Machine learning algorithms can process vast amounts of data.",
                "Natural language processing enables computers to understand human language.",
                "Deep learning networks are inspired by the human brain."
            ]
            
            ids = [f"doc_{i}" for i in range(len(test_docs))]
            
            collection.add(
                documents=test_docs,
                ids=ids
            )
            
            # Verify documents were added
            count = collection.count()
            assert count == len(test_docs), f"Expected {len(test_docs)} docs, got {count}"
            
            # Test query
            results = collection.query(
                query_texts=["machine learning artificial intelligence"],
                n_results=3
            )
            
            assert len(results['documents'][0]) == 3, "Query should return 3 documents"
            assert 'distances' in results, "Results should include distances"
            
            # Clean up
            self.client.delete_collection("test_basic")
            
            self.test_results['basic_functionality'] = True
            logger.info("✅ Basic functionality test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Basic functionality test failed: {e}")
            self.test_results['basic_functionality'] = False
            return False
    
    def test_metadata_handling(self) -> bool:
        """Test metadata storage and filtering."""
        logger.info("Testing metadata handling...")
        
        try:
            collection = self.client.create_collection("test_metadata")
            
            # Add documents with metadata
            documents = [
                "This is a research paper about machine learning.",
                "This is a blog post about cooking recipes.",
                "This is a news article about politics.",
                "This is a tutorial about programming."
            ]
            
            metadata = [
                {"type": "research", "category": "ml", "year": 2023},
                {"type": "blog", "category": "food", "year": 2024},
                {"type": "news", "category": "politics", "year": 2024},
                {"type": "tutorial", "category": "programming", "year": 2023}
            ]
            
            ids = [f"meta_doc_{i}" for i in range(len(documents))]
            
            collection.add(
                documents=documents,
                metadatas=metadata,
                ids=ids
            )
            
            # Test metadata filtering
            results = collection.query(
                query_texts=["learning"],
                n_results=10,
                where={"type": "research"}
            )
            
            assert len(results['documents'][0]) == 1, "Should find 1 research document"
            assert results['metadatas'][0][0]['type'] == 'research', "Metadata should match filter"
            
            # Test complex metadata filtering
            results = collection.query(
                query_texts=["information"],
                n_results=10,
                where={"year": {"$eq": 2024}}
            )
            
            assert all(meta['year'] == 2024 for meta in results['metadatas'][0]), "All results should be from 2024"
            
            # Clean up
            self.client.delete_collection("test_metadata")
            
            self.test_results['metadata_handling'] = True
            logger.info("✅ Metadata handling test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Metadata handling test failed: {e}")
            self.test_results['metadata_handling'] = False
            return False
    
    def test_custom_embeddings(self) -> bool:
        """Test custom embeddings with sentence transformers."""
        logger.info("Testing custom embeddings...")
        
        try:
            collection = self.client.create_collection(
                name="test_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Create custom embeddings
            documents = [
                "The weather is sunny today.",
                "It's raining heavily outside.",
                "Machine learning models require training data.",
                "Neural networks can solve complex problems."
            ]
            
            # Generate embeddings using sentence transformer
            embeddings = self.sentence_model.encode(documents).tolist()
            ids = [f"emb_doc_{i}" for i in range(len(documents))]
            
            # Add with custom embeddings
            collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=ids
            )
            
            # Test query with custom query embedding
            query_text = "What's the weather like?"
            query_embedding = self.sentence_model.encode([query_text]).tolist()
            
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=2
            )
            
            # Should find weather-related documents
            retrieved_docs = results['documents'][0]
            weather_docs = [doc for doc in retrieved_docs if 'weather' in doc.lower() or 'rain' in doc.lower()]
            assert len(weather_docs) > 0, "Should find weather-related documents"
            
            # Clean up
            self.client.delete_collection("test_embeddings")
            
            self.test_results['custom_embeddings'] = True
            logger.info("✅ Custom embeddings test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Custom embeddings test failed: {e}")
            self.test_results['custom_embeddings'] = False
            return False
    
    def test_persistence(self) -> bool:
        """Test data persistence across client restarts."""
        logger.info("Testing persistence...")
        
        try:
            # First client - add data
            collection = self.client.create_collection("test_persistence")
            
            test_docs = [
                "Persistence test document one.",
                "Persistence test document two.",
                "Persistence test document three."
            ]
            
            ids = ["persist_1", "persist_2", "persist_3"]
            
            collection.add(
                documents=test_docs,
                ids=ids
            )
            
            # Close first client
            del self.client
            
            # Create new client with same directory
            self.client = chromadb.PersistentClient(path=str(self.temp_dir))
            
            # Try to access the collection
            collection = self.client.get_collection("test_persistence")
            
            # Verify data is still there
            count = collection.count()
            assert count == 3, f"Expected 3 documents after restart, got {count}"
            
            # Test query still works
            results = collection.query(
                query_texts=["persistence test"],
                n_results=3
            )
            
            assert len(results['documents'][0]) == 3, "Should still be able to query all documents"
            
            # Clean up
            self.client.delete_collection("test_persistence")
            
            self.test_results['persistence'] = True
            logger.info("✅ Persistence test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Persistence test failed: {e}")
            self.test_results['persistence'] = False
            return False
    
    def test_large_batch_operations(self) -> bool:
        """Test handling of large batch operations."""
        logger.info("Testing large batch operations...")
        
        try:
            collection = self.client.create_collection("test_large_batch")
            
            # Generate larger dataset
            num_docs = 1000
            documents = [f"This is test document number {i} with some content about topic {i % 10}." for i in range(num_docs)]
            ids = [f"batch_doc_{i}" for i in range(num_docs)]
            metadata = [{"batch": i // 100, "topic": i % 10} for i in range(num_docs)]
            
            # Add in batches to test batch processing
            batch_size = 100
            
            start_time = time.time()
            
            for i in range(0, num_docs, batch_size):
                end_idx = min(i + batch_size, num_docs)
                
                collection.add(
                    documents=documents[i:end_idx],
                    ids=ids[i:end_idx],
                    metadatas=metadata[i:end_idx]
                )
            
            add_time = time.time() - start_time
            
            # Verify all documents were added
            count = collection.count()
            assert count == num_docs, f"Expected {num_docs} documents, got {count}"
            
            # Test large query
            start_time = time.time()
            results = collection.query(
                query_texts=["test document topic"],
                n_results=50,
                where={"batch": {"$eq": 5}}
            )
            query_time = time.time() - start_time
            
            # Should find documents from batch 5 (documents 500-599)
            assert len(results['documents'][0]) <= 50, "Should not exceed requested n_results"
            assert all(meta['batch'] == 5 for meta in results['metadatas'][0]), "All results should be from batch 5"
            
            logger.info(f"Batch operations performance: Add={add_time:.2f}s, Query={query_time:.2f}s")
            
            # Clean up
            self.client.delete_collection("test_large_batch")
            
            self.test_results['large_batch_operations'] = True
            logger.info("✅ Large batch operations test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Large batch operations test failed: {e}")
            self.test_results['large_batch_operations'] = False
            return False
    
    def test_multilingual_support(self) -> bool:
        """Test multilingual document support (important for zh.qa dataset)."""
        logger.info("Testing multilingual support...")
        
        try:
            collection = self.client.create_collection("test_multilingual")
            
            # Mixed language documents
            documents = [
                "This is an English document about machine learning.",
                "这是一份关于人工智能的中文文档。",  # Chinese
                "Ceci est un document français sur l'informatique.",  # French
                "これは日本語のコンピューターサイエンスに関する文書です。",  # Japanese
                "Este es un documento en español sobre tecnología."  # Spanish
            ]
            
            ids = [f"lang_doc_{i}" for i in range(len(documents))]
            languages = ["en", "zh", "fr", "ja", "es"]
            metadata = [{"language": lang} for lang in languages]
            
            collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadata
            )
            
            # Test query in Chinese
            results = collection.query(
                query_texts=["人工智能 机器学习"],  # AI and machine learning in Chinese
                n_results=3
            )
            
            assert len(results['documents'][0]) == 3, "Should return 3 documents"
            
            # Test filtering by language
            results = collection.query(
                query_texts=["technology computer"],
                n_results=10,
                where={"language": "zh"}
            )
            
            # Should find the Chinese document
            assert len(results['documents'][0]) == 1, "Should find 1 Chinese document"
            assert results['metadatas'][0][0]['language'] == 'zh', "Should be Chinese document"
            
            # Clean up
            self.client.delete_collection("test_multilingual")
            
            self.test_results['multilingual_support'] = True
            logger.info("✅ Multilingual support test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Multilingual support test failed: {e}")
            self.test_results['multilingual_support'] = False
            return False
    
    def test_performance_benchmarks(self) -> bool:
        """Run basic performance benchmarks."""
        logger.info("Running performance benchmarks...")
        
        try:
            collection = self.client.create_collection("test_performance")
            
            # Generate test data
            num_docs = 500
            documents = [f"Performance test document {i} containing various keywords like technology, science, research, innovation, development, analysis, system, method, approach, solution." for i in range(num_docs)]
            ids = [f"perf_doc_{i}" for i in range(num_docs)]
            
            # Benchmark document addition
            start_time = time.time()
            collection.add(documents=documents, ids=ids)
            add_time = time.time() - start_time
            
            # Benchmark queries
            query_times = []
            
            test_queries = [
                "technology innovation",
                "research analysis method",
                "system development approach",
                "science solution technology",
                "performance benchmark test"
            ]
            
            for query in test_queries:
                start_time = time.time()
                results = collection.query(
                    query_texts=[query],
                    n_results=10
                )
                query_time = time.time() - start_time
                query_times.append(query_time)
                
                # Verify results
                assert len(results['documents'][0]) == 10, f"Query should return 10 results for '{query}'"
            
            avg_query_time = np.mean(query_times)
            
            # Performance thresholds (reasonable for testing)
            add_rate = num_docs / add_time  # docs per second
            
            logger.info(f"Performance metrics:")
            logger.info(f"  Document addition: {add_time:.2f}s ({add_rate:.0f} docs/sec)")
            logger.info(f"  Average query time: {avg_query_time:.3f}s")
            logger.info(f"  Query times range: {min(query_times):.3f}s - {max(query_times):.3f}s")
            
            # Basic performance assertions (not too strict for testing)
            assert add_rate > 10, f"Addition rate too slow: {add_rate} docs/sec"
            assert avg_query_time < 1.0, f"Average query time too slow: {avg_query_time}s"
            
            # Clean up
            self.client.delete_collection("test_performance")
            
            self.test_results['performance_benchmarks'] = True
            logger.info("✅ Performance benchmarks test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance benchmarks test failed: {e}")
            self.test_results['performance_benchmarks'] = False
            return False
    
    def cleanup(self):
        """Clean up test environment."""
        logger.info("Cleaning up test environment...")
        
        try:
            # Delete any remaining collections
            collections = self.client.list_collections()
            for collection in collections:
                logger.info(f"Deleting collection: {collection.name}")
                self.client.delete_collection(collection.name)
        except Exception as e:
            logger.warning(f"Error during collection cleanup: {e}")
        
        # Clean up temporary directory
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
            logger.info(f"Removed temporary directory: {self.temp_dir}")
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all ChromaDB tests."""
        logger.info("🧪 Starting ChromaDB comprehensive testing...")
        logger.info("=" * 60)
        
        try:
            # Setup
            self.setup()
            
            # Run tests
            tests = [
                ("Basic Functionality", self.test_basic_functionality),
                ("Metadata Handling", self.test_metadata_handling),
                ("Custom Embeddings", self.test_custom_embeddings),
                ("Persistence", self.test_persistence),
                ("Large Batch Operations", self.test_large_batch_operations),
                ("Multilingual Support", self.test_multilingual_support),
                ("Performance Benchmarks", self.test_performance_benchmarks)
            ]
            
            passed = 0
            total = len(tests)
            
            for test_name, test_func in tests:
                logger.info(f"\n🔬 Running test: {test_name}")
                success = test_func()
                if success:
                    passed += 1
            
            # Summary
            logger.info("\n" + "=" * 60)
            logger.info(f"🎯 TEST SUMMARY: {passed}/{total} tests passed")
            
            if passed == total:
                logger.info("✅ All tests passed! ChromaDB is ready for benchmarking.")
            else:
                logger.warning(f"⚠️ {total - passed} test(s) failed. Check logs for details.")
            
            # Detailed results
            logger.info("\nDetailed Results:")
            for test_name, result in self.test_results.items():
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"  {test_name}: {status}")
            
            return self.test_results
            
        except Exception as e:
            logger.error(f"Testing failed with exception: {e}")
            raise
        finally:
            # Always cleanup
            self.cleanup()


def main():
    """Main function to run ChromaDB tests."""
    tester = ChromaDBTester()
    
    try:
        results = tester.run_all_tests()
        
        # Return appropriate exit code
        if all(results.values()):
            print("\n🚀 ChromaDB is ready for semantic search benchmarking!")
            exit(0)
        else:
            print("\n❌ ChromaDB testing failed. Please fix issues before benchmarking.")
            exit(1)
            
    except Exception as e:
        print(f"\n💥 Testing crashed: {e}")
        exit(1)


if __name__ == "__main__":
    main()