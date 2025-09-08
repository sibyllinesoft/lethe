#!/usr/bin/env python3
"""
Benchmark Orchestrator
======================

Main orchestration engine for comprehensive benchmarking.
Coordinates all systems for fair, reproducible evaluation.
"""

import logging
import time
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import json
import docker

from .config import BenchmarkConfig, COMPETITOR_CONFIGS, DATASET_CONFIGS
from .datasets.registry import get_dataset_registry, DatasetRegistry
from .competitors.registry import get_competitor_registry, CompetitorRegistry
from .evaluation import EvaluationEngine
from .reporting import ReportGenerator

logger = logging.getLogger(__name__)


class BenchmarkOrchestrator:
    """Main orchestrator for comprehensive benchmarking."""
    
    def __init__(self, config: BenchmarkConfig):
        """Initialize benchmark orchestrator."""
        self.config = config
        
        # Component engines
        self.dataset_registry = get_dataset_registry()
        self.competitor_registry = get_competitor_registry()
        self.evaluation_engine = EvaluationEngine(config.evaluation)
        self.report_generator = ReportGenerator(config.reporting)
        
        # Docker client for container management
        self.docker_client = docker.from_env()
        
        # Results storage
        self.results_dir = Path(config.infrastructure.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"BenchmarkOrchestrator initialized: {config.run_name}")
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Execute the complete benchmarking pipeline."""
        
        logger.info("Starting comprehensive benchmark execution")
        start_time = time.time()
        
        try:
            # Step 1: Validate environment and data
            self._validate_environment()
            
            # Step 2: Load datasets
            datasets = self._load_datasets()
            
            # Step 3: Start competitor systems
            competitors = self._start_competitors()
            
            # Step 4: Execute evaluations
            evaluation_results = self._run_evaluations(competitors, datasets)
            
            # Step 5: Perform statistical analysis
            statistical_comparisons = self._run_statistical_analysis(evaluation_results)
            
            # Step 6: Generate reports
            report_paths = self._generate_reports(evaluation_results, statistical_comparisons, datasets)
            
            # Step 7: Cleanup
            self._cleanup_competitors(competitors)
            
            total_time = time.time() - start_time
            
            # Final summary
            summary = {
                "run_name": self.config.run_name,
                "total_duration_seconds": total_time,
                "datasets_evaluated": len(datasets),
                "competitors_evaluated": len(competitors),
                "total_evaluations": sum(len(results) for results in evaluation_results.values()),
                "statistical_comparisons": len(statistical_comparisons),
                "report_paths": report_paths,
                "success": True
            }
            
            # Save summary
            with open(self.results_dir / "benchmark_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Comprehensive benchmark completed successfully in {total_time:.1f}s")
            return summary
            
        except Exception as e:
            logger.error(f"Benchmark execution failed: {e}")
            
            # Try to cleanup on failure
            try:
                self._cleanup_competitors()
            except Exception as cleanup_error:
                logger.error(f"Cleanup failed: {cleanup_error}")
            
            # Return failure summary
            return {
                "run_name": self.config.run_name,
                "success": False,
                "error": str(e),
                "total_duration_seconds": time.time() - start_time
            }
    
    def _validate_environment(self):
        """Validate environment requirements."""
        logger.info("Validating environment...")
        
        # Check Docker availability
        try:
            self.docker_client.ping()
            logger.info("Docker connection validated")
        except Exception as e:
            raise RuntimeError(f"Docker not available: {e}")
        
        # Check disk space
        disk_usage = subprocess.run(['df', '-h', str(self.results_dir)], 
                                   capture_output=True, text=True)
        logger.info(f"Disk usage: {disk_usage.stdout.splitlines()[-1]}")
        
        # Check available datasets
        if self.config.enabled_datasets:
            missing_datasets = []
            for dataset_name in self.config.enabled_datasets:
                if dataset_name not in DATASET_CONFIGS:
                    missing_datasets.append(dataset_name)
            
            if missing_datasets:
                raise ValueError(f"Unknown datasets: {missing_datasets}")
        
        # Check available competitors
        if self.config.enabled_competitors:
            missing_competitors = []
            for competitor_name in self.config.enabled_competitors:
                if competitor_name not in COMPETITOR_CONFIGS:
                    missing_competitors.append(competitor_name)
            
            if missing_competitors:
                raise ValueError(f"Unknown competitors: {missing_competitors}")
        
        logger.info("Environment validation completed")
    
    def _load_datasets(self) -> Dict[str, List[Any]]:
        """Load all benchmark datasets."""
        logger.info("Loading benchmark datasets...")
        
        # Determine which datasets to load
        if self.config.enabled_datasets:
            datasets_to_load = self.config.enabled_datasets
        else:
            datasets_to_load = list(DATASET_CONFIGS.keys())
        
        loaded_datasets = {}
        
        for dataset_name in datasets_to_load:
            logger.info(f"Loading dataset: {dataset_name}")
            
            try:
                dataset_config = DATASET_CONFIGS[dataset_name]
                
                # Check if data path exists
                data_path = Path(dataset_config.data_path)
                if not data_path.exists():
                    logger.warning(f"Dataset file not found: {data_path}")
                    logger.info(f"You may need to download from: {dataset_config.official_loader_url}")
                    continue
                
                # Load dataset with size limits for testing
                max_samples = dataset_config.max_samples
                if self.config.dry_run:
                    max_samples = min(max_samples or 100, 10)  # Limit to 10 samples for dry run
                
                samples, metrics = self.dataset_registry.load_dataset(
                    name=dataset_name,
                    data_path=str(data_path),
                    max_samples=max_samples,
                    validate_samples=True
                )
                
                loaded_datasets[dataset_name] = samples
                logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
                
                # Log dataset statistics
                logger.info(f"Dataset stats - Mean length: {metrics.mean_context_length:.0f}, "
                          f"P95: {metrics.p95_context_length:.0f}, "
                          f"Validation rate: {metrics.validation_success_rate:.1%}")
                
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_name}: {e}")
                if not self.config.dry_run:
                    raise
        
        if not loaded_datasets:
            raise ValueError("No datasets successfully loaded")
        
        logger.info(f"Successfully loaded {len(loaded_datasets)} datasets")
        return loaded_datasets
    
    def _start_competitors(self) -> List[Any]:
        """Start all competitor systems.""" 
        logger.info("Starting competitor systems...")
        
        # Determine which competitors to start
        if self.config.enabled_competitors:
            competitors_to_start = self.config.enabled_competitors
        else:
            competitors_to_start = list(COMPETITOR_CONFIGS.keys())
        
        if self.config.dry_run:
            # For dry run, only start a subset
            competitors_to_start = competitors_to_start[:2]
            logger.info(f"Dry run mode: limiting to {competitors_to_start}")
        
        # Start Docker Compose services
        if not self.config.dry_run:
            self._start_docker_services(competitors_to_start)
        
        # Initialize competitor clients
        competitors = []
        for competitor_name in competitors_to_start:
            try:
                logger.info(f"Initializing competitor: {competitor_name}")
                
                competitor = self.competitor_registry.get_competitor(competitor_name)
                
                # Wait for service to be ready
                if not self.config.dry_run:
                    self._wait_for_competitor_ready(competitor, timeout=120)
                
                competitors.append(competitor)
                logger.info(f"Competitor {competitor_name} ready")
                
            except Exception as e:
                logger.error(f"Failed to start competitor {competitor_name}: {e}")
                if not self.config.dry_run:
                    raise
        
        logger.info(f"Successfully started {len(competitors)} competitors")
        return competitors
    
    def _start_docker_services(self, competitor_names: List[str]):
        """Start Docker Compose services for competitors."""
        logger.info("Starting Docker Compose services...")
        
        # Map competitor names to service names
        service_mapping = {
            'weaviate': 'weaviate',
            'milvus': 'milvus',
            'vespa': 'vespa',
            'opensearch': 'opensearch',
            'splade_v2': 'splade-v2',
            'colbert_v2': 'colbert-v2', 
            'ragatouille': 'ragatouille',
            'bge_reranker_large': 'bge-reranker-large',
            'bge_m3_reranker': 'bge-m3-reranker',
            'monot5': 'monot5',
            'zoekt': ['zoekt-indexserver', 'zoekt-webserver'],
            'livegrep': 'livegrep',
            'graphrag': 'graphrag',
            'streaming_llm': 'streaming-llm',
            'longnet': 'longnet',
            'bge_m3_baseline': 'bge-m3-baseline',
            'lethe_hybrid': 'lethe-hybrid'
        }
        
        # Get services to start
        services_to_start = []
        for competitor_name in competitor_names:
            if competitor_name in service_mapping:
                service = service_mapping[competitor_name]
                if isinstance(service, list):
                    services_to_start.extend(service)
                else:
                    services_to_start.append(service)
        
        if not services_to_start:
            logger.warning("No Docker services to start")
            return
        
        # Start services with docker-compose
        compose_file = Path(__file__).parent / "docker-compose.benchmark.yml"
        
        cmd = [
            "docker-compose", 
            "-f", str(compose_file),
            "up", "-d"
        ] + services_to_start
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Docker services started successfully")
            logger.debug(f"Docker compose output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Docker services: {e}")
            logger.error(f"Stderr: {e.stderr}")
            raise
    
    def _wait_for_competitor_ready(self, competitor: Any, timeout: int = 120):
        """Wait for competitor to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if competitor.health_check():
                    return
            except Exception:
                pass
            
            time.sleep(5)
        
        raise TimeoutError(f"Competitor {competitor.name} not ready after {timeout}s")
    
    def _run_evaluations(self, competitors: List[Any], datasets: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Run all evaluations.""" 
        logger.info("Starting comprehensive evaluation...")
        
        if self.config.dry_run:
            # For dry run, use minimal evaluation
            logger.info("Dry run mode: using simplified evaluation")
            return self._run_dry_run_evaluations(competitors, datasets)
        
        # Run full evaluation with all competitors and datasets
        evaluation_results = self.evaluation_engine.evaluate_all_competitors(
            competitors=competitors,
            datasets=datasets,
            max_workers=self.config.max_workers
        )
        
        # Save raw results
        output_path = self.results_dir / self.config.run_name
        self.evaluation_engine.save_results(
            results=evaluation_results,
            comparisons=[],  # Will be computed next
            output_path=output_path
        )
        
        logger.info(f"Evaluation completed: {len(evaluation_results)} result sets")
        return evaluation_results
    
    def _run_dry_run_evaluations(self, competitors: List[Any], datasets: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Run simplified evaluations for dry run."""
        results = {}
        
        # Pick one dataset and one budget ratio
        dataset_name = list(datasets.keys())[0]
        dataset_samples = datasets[dataset_name][:5]  # Only 5 samples
        keep_ratio = self.config.evaluation.keep_ratios[0]  # Only first budget
        
        for competitor in competitors[:2]:  # Only first 2 competitors
            try:
                logger.info(f"Dry run evaluation: {competitor.name} on {dataset_name}")
                
                result = self.evaluation_engine.evaluate_competitor(
                    competitor=competitor,
                    dataset=dataset_samples,
                    dataset_name=dataset_name,
                    keep_ratio=keep_ratio,
                    k=10  # Smaller k for dry run
                )
                
                key = f"{competitor.name}_{dataset_name}"
                results[key] = [result]
                
            except Exception as e:
                logger.error(f"Dry run evaluation failed for {competitor.name}: {e}")
        
        return results
    
    def _run_statistical_analysis(self, evaluation_results: Dict[str, List[Any]]) -> List[Any]:
        """Run statistical comparisons."""
        logger.info("Performing statistical analysis...")
        
        if self.config.dry_run:
            logger.info("Dry run mode: skipping statistical analysis")
            return []
        
        try:
            statistical_comparisons = self.evaluation_engine.compare_competitors(
                results=evaluation_results,
                baseline_competitor="lethe_hybrid"
            )
            
            logger.info(f"Statistical analysis completed: {len(statistical_comparisons)} comparisons")
            return statistical_comparisons
            
        except Exception as e:
            logger.error(f"Statistical analysis failed: {e}")
            return []
    
    def _generate_reports(
        self, 
        evaluation_results: Dict[str, List[Any]], 
        statistical_comparisons: List[Any],
        datasets: Dict[str, List[Any]]
    ) -> Dict[str, str]:
        """Generate all reports."""
        logger.info("Generating reports...")
        
        output_dir = self.results_dir / self.config.run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Generate comprehensive report
            report_paths = self.report_generator.generate_comprehensive_report(
                evaluation_results=evaluation_results,
                statistical_comparisons=statistical_comparisons,
                datasets=datasets,
                competitor_configs=COMPETITOR_CONFIGS,
                output_dir=output_dir
            )
            
            logger.info(f"Reports generated: {list(report_paths.keys())}")
            return report_paths
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {}
    
    def _cleanup_competitors(self, competitors: List[Any] = None):
        """Cleanup competitor systems."""
        logger.info("Cleaning up competitor systems...")
        
        # Stop individual competitors
        if competitors:
            for competitor in competitors:
                try:
                    if hasattr(competitor, 'stop_container'):
                        competitor.stop_container()
                except Exception as e:
                    logger.warning(f"Failed to stop competitor {competitor.name}: {e}")
        
        # Stop Docker Compose services
        if not self.config.dry_run and self.config.infrastructure.auto_cleanup_containers:
            self._stop_docker_services()
        
        logger.info("Cleanup completed")
    
    def _stop_docker_services(self):
        """Stop all Docker Compose services."""
        try:
            compose_file = Path(__file__).parent / "docker-compose.benchmark.yml"
            
            cmd = ["docker-compose", "-f", str(compose_file), "down"]
            
            if not self.config.infrastructure.preserve_failed_containers:
                cmd.extend(["-v", "--remove-orphans"])  # Remove volumes and orphans
            
            logger.info(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Docker services stopped successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stop Docker services: {e}")
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate benchmark configuration."""
        validation_results = {
            "config_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check dataset availability
        for dataset_name in (self.config.enabled_datasets or DATASET_CONFIGS.keys()):
            if dataset_name not in DATASET_CONFIGS:
                validation_results["errors"].append(f"Unknown dataset: {dataset_name}")
                validation_results["config_valid"] = False
                continue
            
            dataset_config = DATASET_CONFIGS[dataset_name]
            data_path = Path(dataset_config.data_path)
            
            if not data_path.exists():
                validation_results["warnings"].append(
                    f"Dataset file not found: {data_path} "
                    f"(download from: {dataset_config.official_loader_url})"
                )
        
        # Check competitor availability  
        for competitor_name in (self.config.enabled_competitors or COMPETITOR_CONFIGS.keys()):
            if competitor_name not in COMPETITOR_CONFIGS:
                validation_results["errors"].append(f"Unknown competitor: {competitor_name}")
                validation_results["config_valid"] = False
        
        # Check Docker availability
        try:
            self.docker_client.ping()
        except Exception as e:
            validation_results["errors"].append(f"Docker not available: {e}")
            validation_results["config_valid"] = False
        
        return validation_results
    
    def get_status(self) -> Dict[str, Any]:
        """Get current benchmark status."""
        return {
            "run_name": self.config.run_name,
            "results_dir": str(self.results_dir),
            "dry_run": self.config.dry_run,
            "enabled_competitors": self.config.enabled_competitors or list(COMPETITOR_CONFIGS.keys()),
            "enabled_datasets": self.config.enabled_datasets or list(DATASET_CONFIGS.keys()),
            "docker_available": self._check_docker_available()
        }
    
    def _check_docker_available(self) -> bool:
        """Check if Docker is available."""
        try:
            self.docker_client.ping()
            return True
        except Exception:
            return False


def main():
    """Command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Retrieval Benchmarking")
    parser.add_argument("--config", type=str, help="Configuration YAML file")
    parser.add_argument("--dry-run", action="store_true", help="Run with minimal data for testing")
    parser.add_argument("--validate-only", action="store_true", help="Only validate configuration")
    parser.add_argument("--competitors", nargs="+", help="Specific competitors to evaluate")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to use")
    parser.add_argument("--run-name", type=str, help="Custom run name")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"benchmark_{int(time.time())}.log")
        ]
    )
    
    # Load configuration
    if args.config:
        config = BenchmarkConfig.from_yaml(args.config)
    else:
        config = BenchmarkConfig()
    
    # Apply command line overrides
    if args.dry_run:
        config.dry_run = True
    if args.competitors:
        config.enabled_competitors = args.competitors
    if args.datasets:
        config.enabled_datasets = args.datasets
    if args.run_name:
        config.run_name = args.run_name
    
    # Initialize orchestrator
    orchestrator = BenchmarkOrchestrator(config)
    
    if args.validate_only:
        # Validation only
        validation_results = orchestrator.validate_configuration()
        print(json.dumps(validation_results, indent=2))
        sys.exit(0 if validation_results["config_valid"] else 1)
    
    # Run benchmark
    try:
        results = orchestrator.run_comprehensive_benchmark()
        print(json.dumps(results, indent=2))
        sys.exit(0 if results["success"] else 1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()