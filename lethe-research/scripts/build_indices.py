#!/usr/bin/env python3
"""
Production-Grade Index Building Orchestrator

Main script for building real BM25 and ANN indices with comprehensive
metadata export and performance measurement.

Usage:
    python build_indices.py --config config.yaml --dataset msmarco-passage-dev
    python build_indices.py --config config.yaml --dataset all --indices bm25,hnsw
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retriever import (
    RetrieverConfig, 
    TimingHarness, 
    PerformanceProfiler,
    BM25IndexBuilder, 
    ANNIndexBuilder, 
    DenseEmbeddingManager,
    IndexRegistry,
    MetadataManager,
    HNSWConfig,
    IVFPQConfig
)

logger = logging.getLogger(__name__)

class IndexBuildOrchestrator:
    """
    Orchestrates the building of all index types with comprehensive
    timing and metadata management.
    """
    
    def __init__(self, config: RetrieverConfig):
        """
        Initialize the orchestrator.
        
        Args:
            config: Complete retriever configuration
        """
        self.config = config
        self.indices_dir = Path(config.system.indices_dir)
        self.indices_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.registry = IndexRegistry(self.indices_dir)
        self.profiler = PerformanceProfiler()
        self.embedding_manager = None
        
        # Setup logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging for the build process."""
        level = getattr(logging, self.config.log_level.upper())
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.indices_dir / "build.log")
            ]
        )
        
    def build_all_indices(self,
                         datasets: Optional[List[str]] = None,
                         index_types: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Build all specified indices for all datasets.
        
        Args:
            datasets: List of dataset names (None for all configured)
            index_types: List of index types (None for all: bm25, hnsw, ivf_pq)
            
        Returns:
            Dictionary of build results
        """
        
        datasets = datasets or self.config.datasets
        index_types = index_types or ["bm25", "hnsw", "ivf_pq"]
        
        logger.info(f"Building indices for datasets: {datasets}")
        logger.info(f"Index types: {index_types}")
        
        results = {}
        
        for dataset_name in datasets:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing dataset: {dataset_name}")
            logger.info(f"{'='*60}")
            
            dataset_results = {}
            
            # Load dataset
            documents = self._load_dataset(dataset_name)
            if not documents:
                logger.error(f"Failed to load dataset: {dataset_name}")
                continue
                
            logger.info(f"Loaded {len(documents)} documents")
            
            # Build each index type
            for index_type in index_types:
                try:
                    logger.info(f"\nBuilding {index_type} index for {dataset_name}...")
                    
                    if index_type == "bm25":
                        result = self._build_bm25_index(documents, dataset_name)
                    elif index_type == "hnsw":
                        result = self._build_hnsw_index(documents, dataset_name)  
                    elif index_type == "ivf_pq":
                        result = self._build_ivf_pq_index(documents, dataset_name)
                    else:
                        logger.warning(f"Unknown index type: {index_type}")
                        continue
                        
                    dataset_results[index_type] = result
                    logger.info(f"Successfully built {index_type} index")
                    
                except Exception as e:
                    logger.error(f"Failed to build {index_type} index: {e}", exc_info=True)
                    dataset_results[index_type] = {"error": str(e)}
                    
            results[dataset_name] = dataset_results
            
        # Export final report
        self._export_build_report(results)
        
        return results
        
    def _load_dataset(self, dataset_name: str) -> Optional[List[Dict[str, Any]]]:
        """Load dataset documents."""
        
        # Define dataset paths (this would normally come from configuration)
        dataset_paths = {
            "msmarco-passage-dev": "datasets/msmarco-passage-dev/collection.jsonl",
            "trec-covid": "datasets/trec-covid/collection.jsonl", 
            "nfcorpus": "datasets/nfcorpus/collection.jsonl",
            "fiqa-2018": "datasets/fiqa-2018/collection.jsonl"
        }
        
        dataset_path = Path(dataset_paths.get(dataset_name, f"datasets/{dataset_name}/collection.jsonl"))
        
        if not dataset_path.exists():
            logger.error(f"Dataset file not found: {dataset_path}")
            return None
            
        documents = []
        
        try:
            with open(dataset_path, 'r') as f:
                for line_num, line in enumerate(tqdm(f, desc=f"Loading {dataset_name}"), 1):
                    line = line.strip()
                    if line:
                        try:
                            doc = json.loads(line)
                            documents.append(doc)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON at line {line_num}: {e}")
                            continue
                            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            return None
            
        return documents
        
    def _build_bm25_index(self, 
                         documents: List[Dict[str, Any]],
                         dataset_name: str) -> Dict[str, Any]:
        """Build BM25 index for dataset."""
        
        # Create output directory
        index_dir = self.indices_dir / dataset_name / "bm25"
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timing harness
        harness = self.profiler.create_harness("bm25_build", 
                                             cold_cycles=5, 
                                             warm_cycles=1,  # Single build
                                             gc_between_runs=True)
        
        # Create BM25 builder
        builder = BM25IndexBuilder(self.config.bm25)
        
        # Build index with timing
        def build_func():
            return builder.build_index(
                documents=documents,
                index_path=index_dir,
                dataset_name=dataset_name,
                progress_callback=None  # Could add progress callback
            )
            
        # Execute build with timing
        logger.info("Building BM25 index...")
        metadata = harness.benchmark_function(build_func, "bm25_index_build")[0]  # Get first result
        
        # Register index
        self.registry.register_index(metadata)
        
        # Export index parameters
        self._export_index_params(metadata, index_dir)
        
        return {
            "metadata": metadata.to_dict(),
            "build_time_sec": metadata.stats.build_time_sec if metadata.stats else 0,
            "index_size_mb": metadata.stats.index_size_mb if metadata.stats else 0,
            "index_path": str(index_dir)
        }
        
    def _build_hnsw_index(self,
                         documents: List[Dict[str, Any]],
                         dataset_name: str) -> Dict[str, Any]:
        """Build HNSW index for dataset."""
        
        # Create output directory
        index_dir = self.indices_dir / dataset_name / "dense" / "hnsw"
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate embeddings first
        vectors, doc_ids = self._generate_embeddings(documents, dataset_name)
        
        # Create timing harness
        harness = self.profiler.create_harness("hnsw_build",
                                             cold_cycles=1,
                                             warm_cycles=1,
                                             gc_between_runs=True)
        
        # Create HNSW builder
        builder = ANNIndexBuilder("hnsw", self.config.hnsw)
        
        # Build index with timing
        def build_func():
            return builder.build_index(
                vectors=vectors,
                doc_ids=doc_ids,
                index_path=index_dir,
                dataset_name=dataset_name
            )
            
        # Execute build with timing
        logger.info("Building HNSW index...")
        metadata = harness.benchmark_function(build_func, "hnsw_index_build")[0]
        
        # Register index
        self.registry.register_index(metadata)
        
        # Export index parameters
        self._export_index_params(metadata, index_dir)
        
        return {
            "metadata": metadata.to_dict(), 
            "build_time_sec": metadata.stats.build_time_sec if metadata.stats else 0,
            "index_size_mb": metadata.stats.index_size_mb if metadata.stats else 0,
            "index_path": str(index_dir),
            "num_vectors": len(vectors),
            "vector_dim": vectors.shape[1]
        }
        
    def _build_ivf_pq_index(self,
                           documents: List[Dict[str, Any]],
                           dataset_name: str) -> Dict[str, Any]:
        """Build IVF-PQ index for dataset."""
        
        # Create output directory
        index_dir = self.indices_dir / dataset_name / "dense" / "ivf_pq"
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate embeddings (reuse if available)
        vectors, doc_ids = self._generate_embeddings(documents, dataset_name)
        
        # Create timing harness
        harness = self.profiler.create_harness("ivf_pq_build",
                                             cold_cycles=1,
                                             warm_cycles=1,
                                             gc_between_runs=True)
        
        # Create IVF-PQ builder  
        builder = ANNIndexBuilder("ivf_pq", self.config.ivf_pq)
        
        # Build index with timing
        def build_func():
            return builder.build_index(
                vectors=vectors,
                doc_ids=doc_ids,
                index_path=index_dir,
                dataset_name=dataset_name
            )
            
        # Execute build with timing
        logger.info("Building IVF-PQ index...")
        metadata = harness.benchmark_function(build_func, "ivf_pq_index_build")[0]
        
        # Register index
        self.registry.register_index(metadata)
        
        # Export index parameters
        self._export_index_params(metadata, index_dir)
        
        return {
            "metadata": metadata.to_dict(),
            "build_time_sec": metadata.stats.build_time_sec if metadata.stats else 0,
            "index_size_mb": metadata.stats.index_size_mb if metadata.stats else 0, 
            "index_path": str(index_dir),
            "num_vectors": len(vectors),
            "vector_dim": vectors.shape[1]
        }
        
    def _generate_embeddings(self,
                           documents: List[Dict[str, Any]],
                           dataset_name: str) -> tuple:
        """Generate dense embeddings for documents."""
        
        if self.embedding_manager is None:
            embeddings_dir = self.indices_dir / dataset_name / "embeddings"
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            
            harness = self.profiler.create_harness("embedding_generation",
                                                 cold_cycles=1,
                                                 warm_cycles=1)
            
            self.embedding_manager = DenseEmbeddingManager(
                config=self.config.embeddings,
                cache_dir=embeddings_dir,
                timing_harness=harness
            )
            
        logger.info("Generating dense embeddings...")
        
        # Encode collection
        embeddings, metadata = self.embedding_manager.encode_collection(
            documents=documents,
            collection_name=f"{dataset_name}_embeddings",
            text_field="text"  # Assume documents have "text" field
        )
        
        # Extract document IDs
        doc_ids = []
        for doc in documents:
            doc_id = doc.get("id", doc.get("docid", str(len(doc_ids))))
            doc_ids.append(doc_id)
            
        logger.info(f"Generated embeddings: {embeddings.shape}")
        
        return embeddings, doc_ids
        
    def _export_index_params(self, metadata, index_dir: Path):
        """Export index parameters to params.meta file."""
        
        params_file = index_dir / "params.meta"
        
        params_data = {
            "index_type": metadata.index_type,
            "build_params": metadata.build_params,
            "model_params": metadata.model_params,
            "content_hash": metadata.content_hash,
            "parameter_hash": metadata.parameter_hash,
            "build_environment": metadata.build_environment,
            "created_at": metadata.created_at
        }
        
        with open(params_file, 'w') as f:
            json.dump(params_data, f, indent=2)
            
        logger.info(f"Exported parameters: {params_file}")
        
    def _export_build_report(self, results: Dict[str, Dict[str, Any]]):
        """Export comprehensive build report."""
        
        report_file = self.indices_dir / "build_report.json"
        
        # Generate summary statistics
        total_indices = 0
        successful_builds = 0
        failed_builds = 0
        total_build_time = 0.0
        total_index_size = 0.0
        
        for dataset_name, dataset_results in results.items():
            for index_type, result in dataset_results.items():
                total_indices += 1
                
                if "error" in result:
                    failed_builds += 1
                else:
                    successful_builds += 1
                    total_build_time += result.get("build_time_sec", 0)
                    total_index_size += result.get("index_size_mb", 0)
                    
        # Create comprehensive report
        report = {
            "summary": {
                "total_indices": total_indices,
                "successful_builds": successful_builds,
                "failed_builds": failed_builds,
                "success_rate": successful_builds / total_indices if total_indices > 0 else 0,
                "total_build_time_sec": total_build_time,
                "total_index_size_mb": total_index_size
            },
            "configuration": self.config.to_dict(),
            "results": results,
            "performance_profiles": self.profiler.export_profiles()
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Build report exported: {report_file}")
        logger.info(f"Summary: {successful_builds}/{total_indices} successful builds, "
                   f"Total time: {total_build_time:.1f}s, "
                   f"Total size: {total_index_size:.1f}MB")

def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(description="Build production-grade IR indices")
    
    parser.add_argument("--config", "-c", 
                       type=str, 
                       default="config/retriever_config.yaml",
                       help="Configuration file path")
    
    parser.add_argument("--dataset", "-d",
                       type=str,
                       help="Dataset name (or 'all' for all configured datasets)")
    
    parser.add_argument("--indices", "-i",
                       type=str,
                       default="bm25,hnsw,ivf_pq",
                       help="Comma-separated list of index types to build")
    
    parser.add_argument("--output-dir", "-o",
                       type=str,
                       help="Output directory for indices (overrides config)")
    
    parser.add_argument("--force", "-f",
                       action="store_true",
                       help="Force rebuild even if indices exist")
    
    parser.add_argument("--debug",
                       action="store_true", 
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        if Path(args.config).exists():
            config = RetrieverConfig.from_yaml(args.config)
        else:
            logger.warning(f"Config file not found: {args.config}, using defaults")
            config = RetrieverConfig()
            
        # Apply command line overrides
        if args.output_dir:
            config.system.indices_dir = args.output_dir
            
        if args.debug:
            config.debug = True
            config.log_level = "DEBUG"
            
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
        
    # Validate configuration
    errors = config.validate()
    if errors:
        logger.error(f"Configuration validation failed: {errors}")
        sys.exit(1)
        
    # Prepare datasets and index types
    if args.dataset == "all" or args.dataset is None:
        datasets = config.datasets
    else:
        datasets = [args.dataset]
        
    index_types = [t.strip() for t in args.indices.split(",")]
    
    # Create and run orchestrator
    try:
        orchestrator = IndexBuildOrchestrator(config)
        results = orchestrator.build_all_indices(datasets, index_types)
        
        logger.info("Index building completed successfully!")
        
        # Print summary
        successful = sum(1 for dataset_results in results.values() 
                        for result in dataset_results.values()
                        if "error" not in result)
        total = sum(len(dataset_results) for dataset_results in results.values())
        
        print(f"\nBuild Summary:")
        print(f"  Successful: {successful}/{total}")
        print(f"  Output directory: {config.system.indices_dir}")
        
        if successful < total:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Index building failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()