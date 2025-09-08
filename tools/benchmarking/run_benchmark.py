#!/usr/bin/env python3
"""
Comprehensive Benchmark Execution Script
========================================

One-shot execution of the complete benchmarking pipeline.
"""

import sys
import logging
import json
import time
from pathlib import Path

# Add benchmarks to Python path
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.config import BenchmarkConfig
from benchmarks.orchestrator import BenchmarkOrchestrator


def setup_logging():
    """Setup comprehensive logging."""
    log_dir = Path("benchmark_logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = int(time.time())
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / f"benchmark_{timestamp}.log"),
            logging.FileHandler(log_dir / "benchmark_latest.log")  # Always current
        ]
    )


def main():
    """Execute comprehensive benchmark."""
    print("=" * 80)
    print("COMPREHENSIVE RETRIEVAL BENCHMARKING SYSTEM")
    print("=" * 80)
    print("Comparing Lethe-Hybrid against 5 categories of open-source leaders:")
    print("• Hybrid Vector DBs: Weaviate, Milvus, Vespa, OpenSearch")
    print("• Learned Sparse: SPLADE v2, ColBERT v2, RAGatouille")
    print("• Rerankers: BGE-reranker-large/v2-m3, MonoT5")
    print("• Code Search: Zoekt, livegrep, GraphRAG")
    print("• Long-Context: StreamingLLM, LongNet, BGE-M3")
    print()
    print("Datasets: InfiniteBench + RULER + LongBench-v2 + BABILong")
    print("Statistical Rigor: Bootstrap + Permutation + Holm Correction")
    print("=" * 80)
    print()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config_path = Path("benchmark_config.yaml")
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            logger.info("Please ensure benchmark_config.yaml is in the current directory")
            return 1
        
        logger.info("Loading benchmark configuration...")
        config = BenchmarkConfig.from_yaml(config_path)
        
        # Initialize orchestrator
        logger.info("Initializing benchmark orchestrator...")
        orchestrator = BenchmarkOrchestrator(config)
        
        # Validate configuration
        logger.info("Validating configuration and environment...")
        validation_results = orchestrator.validate_configuration()
        
        if not validation_results["config_valid"]:
            logger.error("Configuration validation failed:")
            for error in validation_results["errors"]:
                logger.error(f"  • {error}")
            return 1
        
        if validation_results["warnings"]:
            logger.warning("Configuration warnings:")
            for warning in validation_results["warnings"]:
                logger.warning(f"  • {warning}")
        
        # Show status
        status = orchestrator.get_status()
        logger.info("Benchmark Status:")
        logger.info(f"  • Run Name: {status['run_name']}")
        logger.info(f"  • Results Directory: {status['results_dir']}")
        logger.info(f"  • Dry Run: {status['dry_run']}")
        logger.info(f"  • Competitors: {len(status['enabled_competitors'])}")
        logger.info(f"  • Datasets: {len(status['enabled_datasets'])}")
        logger.info(f"  • Docker Available: {status['docker_available']}")
        
        # Execute comprehensive benchmark
        logger.info("Starting comprehensive benchmark execution...")
        print("\nExecuting benchmark - this may take several hours for full evaluation...")
        
        results = orchestrator.run_comprehensive_benchmark()
        
        # Print results summary
        print("\n" + "=" * 80)
        if results["success"]:
            print("✅ COMPREHENSIVE BENCHMARK COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print(f"📊 Total Duration: {results['total_duration_seconds']:.1f} seconds")
            print(f"📊 Datasets Evaluated: {results.get('datasets_evaluated', 'N/A')}")
            print(f"📊 Competitors Evaluated: {results.get('competitors_evaluated', 'N/A')}")
            print(f"📊 Total Evaluations: {results.get('total_evaluations', 'N/A')}")
            print(f"📊 Statistical Comparisons: {results.get('statistical_comparisons', 'N/A')}")
            
            print("\n📋 Generated Reports:")
            for report_type, path in results.get('report_paths', {}).items():
                print(f"  • {report_type.upper()}: {path}")
            
            print(f"\n🎯 Results saved in: {Path(config.infrastructure.results_dir) / config.run_name}")
            
        else:
            print("❌ BENCHMARK EXECUTION FAILED")
            print("=" * 80)
            print(f"Error: {results.get('error', 'Unknown error')}")
            print(f"Duration before failure: {results['total_duration_seconds']:.1f} seconds")
            return 1
        
        print("=" * 80)
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by user")
        print("\n⚠️  Benchmark interrupted - cleaning up...")
        return 130  # Standard exit code for Ctrl+C
        
    except Exception as e:
        logger.error(f"Benchmark execution failed: {e}")
        print(f"\n❌ Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)