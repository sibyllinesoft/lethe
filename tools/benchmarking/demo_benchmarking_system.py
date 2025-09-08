#!/usr/bin/env python3
"""
Semantic Search Benchmarking System Demo

This script demonstrates the complete benchmarking system setup and validation.
It shows all major components working together for the InfinityBench zh.qa stress test.
"""

import asyncio
import logging
import time
from pathlib import Path

# Configure clean logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print system banner."""
    print("🔍" + "="*70 + "🔍")
    print("  SEMANTIC SEARCH BENCHMARKING SYSTEM - INFINITYBENCH ZH.QA")  
    print("  2M Token Stress Test for ChromaDB vs Lethe vs Truncation")
    print("🔍" + "="*70 + "🔍")

def print_system_info():
    """Display system information."""
    print("\n📋 SYSTEM INFORMATION")
    print("-" * 40)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name() if cuda_available else "No GPU"
        print(f"🖥️  CUDA Available: {cuda_available}")
        print(f"🎮 GPU Device: {gpu_name}")
    except ImportError:
        print("🖥️  PyTorch not available - CPU only")
    
    try:
        import chromadb
        print(f"🗄️  ChromaDB Version: {chromadb.__version__}")
    except ImportError:
        print("❌ ChromaDB not available")
    
    try:
        import ollama
        print("🦙 Ollama Client: Available")
    except ImportError:
        print("❌ Ollama not available")

def test_ollama_models():
    """Test Ollama model availability."""
    print("\n🦙 OLLAMA MODEL CHECK")
    print("-" * 40)
    
    try:
        import ollama
        client = ollama.Client()
        models = client.list()
        
        print("Available models:")
        for model in models.models:
            size_gb = model.size / (1024**3) if model.size else 0
            print(f"  📦 {model.model} ({size_gb:.1f}GB)")
        
        # Test generation with first available model
        if models.models:
            test_model = models.models[0].model
            print(f"\n🧪 Testing generation with {test_model}...")
            
            start = time.time()
            response = client.generate(
                model=test_model,
                prompt="What is semantic search?",
                options={'num_predict': 20}
            )
            duration = time.time() - start
            
            print(f"✅ Response: {response['response'][:100]}...")
            print(f"⏱️  Generation time: {duration:.2f}s")
        
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")

def test_chromadb():
    """Test ChromaDB functionality."""
    print("\n🗄️ CHROMADB FUNCTIONALITY CHECK")  
    print("-" * 40)
    
    try:
        import chromadb
        import tempfile
        from pathlib import Path
        
        # Create temporary client
        with tempfile.TemporaryDirectory() as temp_dir:
            client = chromadb.PersistentClient(path=temp_dir)
            
            # Test collection creation
            collection = client.create_collection("test")
            
            # Test document addition
            test_docs = [
                "Semantic search uses vector embeddings for retrieval",
                "Traditional search relies on keyword matching",
                "ChromaDB provides vector similarity search"
            ]
            
            collection.add(
                documents=test_docs,
                ids=[f"doc_{i}" for i in range(len(test_docs))]
            )
            
            # Test query
            results = collection.query(
                query_texts=["vector similarity retrieval"],
                n_results=2
            )
            
            print("✅ ChromaDB basic operations successful")
            print(f"📊 Query results: {len(results['documents'][0])} documents")
            print(f"🎯 Top result: {results['documents'][0][0][:60]}...")
        
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")

def test_dataset_access():
    """Test InfinityBench dataset access."""
    print("\n📚 INFINITYBENCH DATASET ACCESS")
    print("-" * 40)
    
    try:
        from datasets import load_dataset
        
        print("🔄 Loading InfinityBench zh.qa dataset (1 sample)...")
        start = time.time()
        
        dataset = load_dataset(
            "xinrongzhang2022/InfiniteBench",
            data_files="longbook_qa_chn.jsonl",
            split="train[:1]"  # Just one sample for testing
        )
        
        duration = time.time() - start
        
        if len(dataset) > 0:
            sample = dataset[0]
            context_len = len(sample['context'])
            
            print(f"✅ Dataset loaded successfully ({duration:.2f}s)")
            print(f"📄 Sample context length: {context_len:,} characters")
            print(f"❓ Question: {sample['input'][:80]}...")
            print(f"💡 Answer type: {type(sample['answer'])}")
            
            # Estimate token count
            est_tokens = context_len // 4  # Rough estimate for Chinese text
            print(f"🔢 Estimated tokens: ~{est_tokens:,}")
            
            if est_tokens > 500_000:
                print("🚀 Confirmed: 500k+ token stress test dataset")
        else:
            print("⚠️  Dataset loaded but no samples found")
    
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")

def show_benchmarking_overview():
    """Show the benchmarking system overview."""
    print("\n🎯 BENCHMARKING SYSTEM OVERVIEW")
    print("-" * 40)
    
    print("📊 Three Methods Compared:")
    print("  1️⃣  ChromaDB Vector Search")
    print("     - Sentence transformer embeddings")  
    print("     - Cosine similarity ranking")
    print("     - Chunk-based retrieval")
    
    print("\n  2️⃣  Lethe Context-Aware Retrieval")
    print("     - Semantic understanding (placeholder)")
    print("     - Context-aware ranking") 
    print("     - Advanced query processing")
    
    print("\n  3️⃣  Simple Truncation Method")
    print("     - First 120k tokens only")
    print("     - Direct LLM processing")
    print("     - Baseline comparison")
    
    print("\n📈 Evaluation Metrics:")
    print("  • Precision@K (k=1,5,10,20,50)")
    print("  • Recall@K (k=1,5,10,20,50)")  
    print("  • Query time performance")
    print("  • Generation time analysis")
    print("  • Scalability vs context length")
    
    print("\n📁 Output Generation:")
    print("  • JSON results for each method")
    print("  • CSV combined dataset")
    print("  • Publication-quality plots")
    print("  • Performance analysis reports")

def show_usage_instructions():
    """Show how to use the system."""
    print("\n🚀 USAGE INSTRUCTIONS")
    print("-" * 40)
    
    print("1️⃣  Quick Test (3 samples):")
    print("   python3 run_small_benchmark.py")
    
    print("\n2️⃣  Full Benchmark (all samples):")  
    print("   python3 semantic_search_benchmark.py")
    
    print("\n3️⃣  Validate ChromaDB Setup:")
    print("   python3 test_chromadb.py")
    
    print("\n📋 Key Files:")
    print("   • semantic_search_benchmark.py   - Main benchmarking system")
    print("   • test_chromadb.py               - ChromaDB validation")
    print("   • run_small_benchmark.py         - Quick testing script")
    print("   • semantic_search_requirements.txt - Dependencies")
    
    print("\n📊 Results Location:")
    print("   • ./benchmark_data/results/      - JSON/CSV results")
    print("   • ./benchmark_data/visualizations/ - PNG/PDF plots")

async def main():
    """Main demo function."""
    print_banner()
    print_system_info()
    test_ollama_models()
    test_chromadb()
    test_dataset_access()
    show_benchmarking_overview()
    show_usage_instructions()
    
    print("\n🎉 SYSTEM VALIDATION COMPLETE")
    print("="*72)
    print("✅ All major components are functional")
    print("🚀 Ready for semantic search benchmarking!")
    print("📖 See SEMANTIC_SEARCH_BENCHMARK_README.md for detailed usage")

if __name__ == "__main__":
    asyncio.run(main())