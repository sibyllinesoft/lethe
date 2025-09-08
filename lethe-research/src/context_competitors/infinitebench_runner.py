#!/usr/bin/env python3
"""
InfiniteBench Context Management Competitor Evaluation.

Runs all context management competitors against real InfiniteBench datasets
for comprehensive evaluation of context processing approaches.
"""

import os
import sys
import json
import importlib
import importlib.util
import argparse
import logging
import traceback
import time
from pathlib import Path
from typing import List, Dict, Any, Type
from datetime import datetime

from .competitor_interface import ContextManagementCompetitor, ContextProcessingResult, LetheCompetitor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InfiniteBenchCompetitorRunner:
    """
    Run context management competitors against InfiniteBench datasets.
    """
    
    def __init__(self, results_dir: str = None, infinitebench_data_dir: str = None):
        """Initialize InfiniteBench competitor runner."""
        self.project_root = Path(__file__).parent
        self.benchmarks_dir = self.project_root / "benchmarks"
        self.results_dir = Path(results_dir) if results_dir else (self.project_root / "results")
        
        # InfiniteBench data directory
        if infinitebench_data_dir:
            self.infinitebench_data_dir = Path(infinitebench_data_dir)
        else:
            # Try to find InfiniteBench data automatically
            possible_paths = [
                Path(__file__).parent.parent.parent / "benchmarks" / "infinitebench" / "data",
                Path(__file__).parent.parent / "infinitebench" / "data",
                Path("./benchmarks/infinitebench/data"),
                Path("./data")
            ]
            self.infinitebench_data_dir = None
            for path in possible_paths:
                if path.exists() and (path / "longbook_qa_chn.jsonl").exists():
                    self.infinitebench_data_dir = path
                    break
        
        # Create directories
        self.benchmarks_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        self.competitors: Dict[str, Type[ContextManagementCompetitor]] = {}
    
    def discover_competitors(self) -> List[str]:
        """Discover available competitor implementations."""
        discovered = []
        
        # Always include Lethe
        self.competitors['lethe'] = LetheCompetitor
        discovered.append('lethe')
        
        # Auto-discover benchmark scripts
        if not self.benchmarks_dir.exists():
            logger.warning(f"Benchmarks directory not found: {self.benchmarks_dir}")
            return discovered
        
        for benchmark_file in self.benchmarks_dir.glob("*_benchmark.py"):
            try:
                # Add the competitor directory to sys.path temporarily
                competitors_dir = str(self.benchmarks_dir.parent)
                if competitors_dir not in sys.path:
                    sys.path.insert(0, competitors_dir)
                
                # Import the module
                module_name = benchmark_file.stem
                spec = importlib.util.spec_from_file_location(module_name, benchmark_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find competitor class
                competitor_class = None
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type):
                        try:
                            # Check if it's a subclass by checking the method resolution order
                            base_class_names = [base.__name__ for base in attr.__mro__]
                            if ('ContextManagementCompetitor' in base_class_names and 
                                attr.__name__ != 'ContextManagementCompetitor'):
                                competitor_class = attr
                                break
                        except Exception:
                            continue
                
                if competitor_class:
                    competitor_name = module_name.replace('_benchmark', '')
                    self.competitors[competitor_name] = competitor_class
                    discovered.append(competitor_name)
                    logger.info(f"Discovered competitor: {competitor_name}")
                else:
                    logger.warning(f"No competitor class found in {benchmark_file}")
                    
            except Exception as e:
                logger.error(f"Failed to load {benchmark_file}: {e}")
                logger.debug(traceback.format_exc())
        
        return discovered
    
    def load_infinitebench_dataset(self, dataset_name: str, max_samples: int = 10) -> List[Dict[str, Any]]:
        """Load samples from InfiniteBench dataset."""
        if not self.infinitebench_data_dir:
            raise FileNotFoundError(f"InfiniteBench data directory not found. Please specify with --data-dir")
        
        dataset_file = self.infinitebench_data_dir / f"{dataset_name}.jsonl"
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_file}")
        
        samples = []
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {i+1} in {dataset_name}: {e}")
                    continue
        
        logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
        return samples
    
    def run_infinitebench_evaluation(self, dataset_name: str = "longbook_qa_chn", 
                                   competitor_names: List[str] = None, 
                                   max_samples: int = 10) -> Dict[str, Any]:
        """
        Run InfiniteBench evaluation for specified competitors.
        
        Args:
            dataset_name: InfiniteBench dataset to use (e.g., 'longbook_qa_chn')
            competitor_names: List of competitor names to test (None = all available)
            max_samples: Maximum number of samples to test per competitor
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation results
        """
        # Discover available competitors
        available_competitors = self.discover_competitors()
        
        # Use specified competitors or all available
        if competitor_names:
            competitors_to_test = [name for name in competitor_names if name in available_competitors]
            missing = set(competitor_names) - set(competitors_to_test)
            if missing:
                logger.warning(f"Requested competitors not found: {missing}")
        else:
            competitors_to_test = available_competitors
        
        logger.info(f"Testing competitors: {competitors_to_test}")
        
        # Load InfiniteBench dataset
        try:
            samples = self.load_infinitebench_dataset(dataset_name, max_samples)
        except FileNotFoundError as e:
            logger.error(f"Failed to load dataset: {e}")
            return {"error": str(e)}
        
        # Run evaluation
        results = {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "samples_tested": len(samples),
            "competitors_tested": competitors_to_test,
            "results": {}
        }
        
        for competitor_name in competitors_to_test:
            logger.info(f"\\n=== Testing {competitor_name} on {dataset_name} ===")
            
            try:
                # Initialize competitor
                competitor_class = self.competitors[competitor_name]
                competitor = competitor_class()
                
                # Check availability
                if not competitor.is_available():
                    logger.warning(f"{competitor_name} dependencies not available")
                    results["results"][competitor_name] = {
                        "status": "unavailable",
                        "error": "Dependencies not installed",
                        "requirements": competitor.get_installation_requirements()
                    }
                    continue
                
                # Initialize
                if not competitor.initialize():
                    logger.error(f"Failed to initialize {competitor_name}")
                    results["results"][competitor_name] = {
                        "status": "initialization_failed",
                        "error": "Initialization failed"
                    }
                    continue
                
                # Run evaluation on samples
                competitor_results = []
                total_time = 0
                successful_samples = 0
                
                for i, sample in enumerate(samples):
                    logger.info(f"  Processing sample {i+1}/{len(samples)}...")
                    
                    try:
                        # Extract context and query from InfiniteBench format
                        context = sample.get('context', '')
                        query = sample.get('input', '') or sample.get('question', '')
                        expected_answer = sample.get('answer', '') or sample.get('answers', [])
                        
                        # Estimate max tokens (2M tokens ≈ 8M characters)
                        context_length = len(context)
                        max_tokens = min(4000, max(1000, context_length // 4))
                        
                        start_time = time.time()
                        result = competitor.process_context(
                            query=query,
                            context=context, 
                            max_tokens=max_tokens
                        )
                        
                        processing_time = (time.time() - start_time) * 1000
                        total_time += processing_time
                        successful_samples += 1
                        
                        # Calculate basic accuracy (simplified)
                        response_text = result.response.lower() if result.response else ""
                        expected_text = str(expected_answer).lower() if expected_answer else ""
                        
                        # Simple accuracy check (could be improved with more sophisticated matching)
                        accuracy = 1.0 if expected_text in response_text or any(
                            str(ans).lower() in response_text for ans in (expected_answer if isinstance(expected_answer, list) else [expected_answer])
                        ) else 0.0
                        
                        competitor_results.append({
                            "sample_id": i,
                            "context_length_chars": len(context),
                            "query_length_chars": len(query),
                            "processing_time_ms": processing_time,
                            "original_tokens": result.original_token_count,
                            "processed_tokens": result.processed_token_count,
                            "compression_ratio": result.compression_ratio,
                            "response_length": len(result.response),
                            "accuracy": accuracy,
                            "metadata": result.metadata
                        })
                        
                    except Exception as e:
                        logger.error(f"  Failed sample {i+1}: {e}")
                        competitor_results.append({
                            "sample_id": i,
                            "error": str(e),
                            "status": "failed"
                        })
                
                # Calculate summary statistics
                successful_results = [r for r in competitor_results if "error" not in r]
                if successful_results:
                    avg_time = sum(r["processing_time_ms"] for r in successful_results) / len(successful_results)
                    avg_compression = sum(r["compression_ratio"] for r in successful_results) / len(successful_results)
                    avg_accuracy = sum(r["accuracy"] for r in successful_results) / len(successful_results)
                else:
                    avg_time = avg_compression = avg_accuracy = 0.0
                
                results["results"][competitor_name] = {
                    "status": "completed",
                    "samples_processed": len(successful_results),
                    "samples_failed": len(competitor_results) - len(successful_results),
                    "avg_processing_time_ms": avg_time,
                    "avg_compression_ratio": avg_compression,
                    "avg_accuracy": avg_accuracy,
                    "total_time_ms": total_time,
                    "sample_results": competitor_results,
                    "requirements": competitor.get_installation_requirements()
                }
                
                # Cleanup
                competitor.cleanup()
                
            except Exception as e:
                logger.error(f"Failed to test {competitor_name}: {e}")
                logger.debug(traceback.format_exc())
                results["results"][competitor_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Save results
        self.save_infinitebench_results(results, dataset_name)
        return results
    
    def save_infinitebench_results(self, results: Dict[str, Any], dataset_name: str):
        """Save InfiniteBench evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"infinitebench_{dataset_name}_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {results_file}")
    
    def generate_infinitebench_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable InfiniteBench evaluation report."""
        report = []
        report.append(f"# InfiniteBench Context Management Evaluation Report")
        report.append(f"Dataset: {results['dataset']}")
        report.append(f"Generated: {results['timestamp']}")
        report.append(f"Samples tested: {results['samples_tested']}")
        report.append(f"Competitors tested: {len(results['results'])}")
        report.append("")
        
        # Summary table
        report.append("## Performance Summary")
        report.append("| Competitor | Status | Avg Time (ms) | Compression | Accuracy | Samples |")
        report.append("|------------|--------|---------------|-------------|----------|---------|")
        
        for competitor_name, competitor_results in results["results"].items():
            if competitor_results["status"] == "completed":
                avg_time = competitor_results["avg_processing_time_ms"]
                avg_compression = competitor_results["avg_compression_ratio"]
                avg_accuracy = competitor_results["avg_accuracy"]
                samples_processed = competitor_results["samples_processed"]
                report.append(f"| {competitor_name} | ✅ Complete | {avg_time:.1f} | {avg_compression:.3f} | {avg_accuracy:.2f} | {samples_processed} |")
            else:
                status = competitor_results["status"]
                report.append(f"| {competitor_name} | ❌ {status} | - | - | - | 0 |")
        
        report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        for competitor_name, competitor_results in results["results"].items():
            report.append(f"### {competitor_name}")
            
            if competitor_results["status"] != "completed":
                report.append(f"Status: {competitor_results['status']}")
                if "error" in competitor_results:
                    report.append(f"Error: {competitor_results['error']}")
                report.append("")
                continue
            
            report.append(f"- **Samples processed**: {competitor_results['samples_processed']}")
            report.append(f"- **Average processing time**: {competitor_results['avg_processing_time_ms']:.1f}ms")
            report.append(f"- **Average compression ratio**: {competitor_results['avg_compression_ratio']:.3f}")
            report.append(f"- **Average accuracy**: {competitor_results['avg_accuracy']:.2f}")
            report.append(f"- **Total processing time**: {competitor_results['total_time_ms']:.1f}ms")
            report.append("")
        
        return "\\n".join(report)


def main():
    """Main entry point for InfiniteBench competitor evaluation."""
    parser = argparse.ArgumentParser(description="InfiniteBench Context Management Competitor Evaluation")
    parser.add_argument("--dataset", default="longbook_qa_chn", help="InfiniteBench dataset to use")
    parser.add_argument("--competitors", nargs="*", help="Specific competitors to test")
    parser.add_argument("--samples", type=int, default=10, help="Maximum samples to test per competitor")
    parser.add_argument("--data-dir", help="InfiniteBench data directory")
    parser.add_argument("--results-dir", help="Directory to store results")
    
    args = parser.parse_args()
    
    runner = InfiniteBenchCompetitorRunner(
        results_dir=args.results_dir,
        infinitebench_data_dir=args.data_dir
    )
    
    # Run InfiniteBench evaluation
    logger.info(f"Starting InfiniteBench evaluation on {args.dataset}")
    results = runner.run_infinitebench_evaluation(
        dataset_name=args.dataset,
        competitor_names=args.competitors,
        max_samples=args.samples
    )
    
    # Generate and display report
    if "error" not in results:
        report = runner.generate_infinitebench_report(results)
        print("\\n" + "="*80)
        print(report)
    else:
        print(f"Error: {results['error']}")


if __name__ == "__main__":
    main()