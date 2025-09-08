#!/usr/bin/env python3
"""
Small-scale test of the semantic search benchmark system.
This runs a quick test with a limited dataset to validate the system before full benchmarking.
"""

import asyncio
import logging
from semantic_search_benchmark import SemanticSearchBenchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Run small-scale benchmark test."""
    print("🧪 Running Small-Scale Semantic Search Benchmark Test")
    print("=" * 60)
    
    # Initialize with very limited samples for testing
    benchmark = SemanticSearchBenchmark(
        data_dir="./test_benchmark_data",
        chroma_dir="./test_chroma_db", 
        max_samples=3,  # Very small for testing
        ollama_model="gemma3:27b"  # Using the correct model name
    )
    
    try:
        logger.info("Setting up components...")
        await benchmark.setup_components()
        
        logger.info("Loading small dataset...")
        dataset = await benchmark.load_infinitybench_dataset()
        
        if not dataset:
            logger.error("No data loaded!")
            return
        
        logger.info(f"Loaded {len(dataset)} samples for testing")
        for i, sample in enumerate(dataset):
            logger.info(f"Sample {i+1}: {sample['context_length']:,} tokens")
        
        logger.info("Creating ChromaDB index...")
        await benchmark.create_chroma_index()
        
        logger.info("Running ChromaDB benchmark (k=1,5,10)...")
        chroma_results = await benchmark.benchmark_chroma_retrieval([1, 5, 10])
        
        logger.info("Running Lethe benchmark (k=1,5,10)...")
        lethe_results = await benchmark.benchmark_lethe_retrieval([1, 5, 10])
        
        logger.info("Running Truncation benchmark...")
        truncation_results = await benchmark.benchmark_truncation_method()
        
        # Save results
        benchmark.save_results()
        
        # Generate visualizations
        benchmark.generate_visualizations()
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ SMALL BENCHMARK TEST COMPLETED!")
        print(f"📊 ChromaDB results: {len(chroma_results)}")
        print(f"📊 Lethe results: {len(lethe_results)}")
        print(f"📊 Truncation results: {len(truncation_results)}")
        print(f"📁 Results saved to: {benchmark.data_dir}/results/")
        print(f"📈 Visualizations saved to: {benchmark.data_dir}/visualizations/")
        
        # Show sample results
        if chroma_results:
            sample_result = chroma_results[0]
            print(f"\nSample ChromaDB result:")
            print(f"  Query time: {sample_result['query_time']:.3f}s")
            print(f"  Generation time: {sample_result['generation_time']:.3f}s")
            print(f"  Precision@{sample_result['k']}: {sample_result['precision_at_k']:.3f}")
        
        print("\n🚀 System is ready for full-scale benchmarking!")
        
    except Exception as e:
        logger.error(f"Small benchmark test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())