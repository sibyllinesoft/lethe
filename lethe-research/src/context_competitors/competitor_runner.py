#!/usr/bin/env python3
"""
Robust LLM Context Management Research Benchmarking Runner.

Automatically discovers and runs competitor implementations for fair comparison
with Lethe. Designed to be robust to addition/removal of benchmark scripts.

Usage:
    python3 -m src.context_competitors.competitor_runner
    python3 -m src.context_competitors.competitor_runner --competitors llmlingua h2o
    python3 -m src.context_competitors.competitor_runner --report-only
"""

import os
import sys
import json
import importlib
import importlib.util
import argparse
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Any, Type
from datetime import datetime

from .competitor_interface import ContextManagementCompetitor, ContextProcessingResult, LetheCompetitor


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CompetitorBenchmarkRunner:
    """
    Robust runner for LLM context management competitor benchmarks.
    
    Features:
    - Auto-discovers competitor implementations
    - Robust to script additions/removals
    - Fair comparison protocol
    - Comprehensive result reporting
    """
    
    def __init__(self, results_dir: str = None):
        """Initialize benchmark runner."""
        self.project_root = Path(__file__).parent
        self.benchmarks_dir = self.project_root / "benchmarks"
        self.results_dir = Path(results_dir) if results_dir else (self.project_root / "results")
        
        # Create directories
        self.benchmarks_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        self.competitors: Dict[str, Type[ContextManagementCompetitor]] = {}
        self.test_cases = []
        
    def discover_competitors(self) -> List[str]:
        """
        Automatically discover competitor implementations.
        
        Returns:
            List[str]: Names of discovered competitors
        """
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
                module_classes = []
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type):
                        module_classes.append(f"{attr_name}: {attr}")
                        try:
                            # Check if it's a subclass by checking the method resolution order or base class names
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
                    logger.debug(f"Available classes in {module_name}: {module_classes}")
                    logger.debug(f"ContextManagementCompetitor available: {hasattr(module, 'ContextManagementCompetitor')}")
                    
            except Exception as e:
                logger.error(f"Failed to load {benchmark_file}: {e}")
                logger.debug(traceback.format_exc())
        
        return discovered
    
    def create_test_cases(self) -> List[Dict[str, Any]]:
        """Create standardized test cases for all competitors."""
        test_cases = [
            {
                "name": "short_qa",
                "query": "What is the main topic discussed?",
                "context": "Artificial intelligence has revolutionized many industries. Machine learning algorithms can process vast amounts of data to identify patterns and make predictions. Deep learning, a subset of machine learning, uses neural networks with multiple layers to solve complex problems. These technologies have applications in healthcare, finance, transportation, and entertainment.",
                "max_tokens": 100,
                "expected_type": "topic_identification"
            },
            {
                "name": "medium_context",
                "query": "What are the key benefits mentioned?",
                "context": "Cloud computing has transformed how businesses operate. " * 100 + " The main benefits include scalability, cost reduction, improved collaboration, enhanced security, and automatic updates. Companies can scale resources up or down based on demand, reducing infrastructure costs significantly.",
                "max_tokens": 500,
                "expected_type": "benefit_extraction"
            },
            {
                "name": "long_context_needle",
                "query": "What is the secret code?",
                "context": self._generate_needle_haystack_context("The secret code is ALPHA-7839", 2000),
                "max_tokens": 1000,
                "expected_type": "needle_in_haystack"
            },
            {
                "name": "very_long_context",
                "query": "Summarize the main findings about climate change impacts.",
                "context": self._generate_climate_research_context(5000),
                "max_tokens": 2000,
                "expected_type": "summarization"
            }
        ]
        
        return test_cases
    
    def _generate_needle_haystack_context(self, needle: str, target_length: int) -> str:
        """Generate needle-in-haystack test context."""
        haystack = "This is filler text that provides context but doesn't contain the answer. " * 50
        
        # Insert needle at random position
        import random
        needle_pos = random.randint(len(haystack) // 4, 3 * len(haystack) // 4)
        
        context = haystack[:needle_pos] + " " + needle + " " + haystack[needle_pos:]
        
        # Extend to target length
        while len(context.split()) < target_length:
            context += " " + haystack
        
        return " ".join(context.split()[:target_length])
    
    def _generate_climate_research_context(self, target_length: int) -> str:
        """Generate realistic climate research context."""
        base_text = """Climate change research has revealed significant impacts on global ecosystems. 
        Temperature increases have led to ice cap melting, sea level rise, and altered precipitation patterns. 
        Agricultural systems face challenges from changing weather patterns, affecting food security globally. 
        Biodiversity loss accelerates as species struggle to adapt to rapid environmental changes. 
        Economic impacts include infrastructure damage from extreme weather events and adaptation costs. 
        Renewable energy adoption has increased, but transition speed remains critical for emission reduction goals. 
        International cooperation through climate agreements aims to limit global temperature increase to 1.5°C. 
        Scientific consensus indicates immediate action is necessary to prevent irreversible environmental damage."""
        
        # Repeat and extend to target length
        context = base_text
        while len(context.split()) < target_length:
            context += " " + base_text
        
        return " ".join(context.split()[:target_length])
    
    def run_benchmarks(self, competitor_names: List[str] = None) -> Dict[str, Any]:
        """
        Run benchmarks for specified competitors.
        
        Args:
            competitor_names: List of competitor names to test (None = all available)
            
        Returns:
            Dict[str, Any]: Comprehensive benchmark results
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
        
        # Create test cases
        test_cases = self.create_test_cases()
        
        # Run benchmarks
        results = {
            "timestamp": datetime.now().isoformat(),
            "competitors_tested": competitors_to_test,
            "test_cases": len(test_cases),
            "results": {}
        }
        
        for competitor_name in competitors_to_test:
            logger.info(f"\\n=== Testing {competitor_name} ===")
            
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
                
                # Run test cases
                competitor_results = []
                for test_case in test_cases:
                    logger.info(f"  Running {test_case['name']}...")
                    
                    try:
                        result = competitor.process_context(
                            query=test_case["query"],
                            context=test_case["context"], 
                            max_tokens=test_case["max_tokens"]
                        )
                        
                        # Convert to serializable format
                        competitor_results.append({
                            "test_case": test_case["name"],
                            "processing_time_ms": result.processing_time_ms,
                            "original_tokens": result.original_token_count,
                            "processed_tokens": result.processed_token_count,
                            "compression_ratio": result.compression_ratio,
                            "response_length": len(result.response),
                            "accuracy_score": result.accuracy_score,
                            "metadata": result.metadata
                        })
                        
                    except Exception as e:
                        logger.error(f"  Failed {test_case['name']}: {e}")
                        competitor_results.append({
                            "test_case": test_case["name"],
                            "error": str(e),
                            "status": "failed"
                        })
                
                results["results"][competitor_name] = {
                    "status": "completed",
                    "test_results": competitor_results,
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
        self.save_results(results)
        return results
    
    def save_results(self, results: Dict[str, Any]):
        """Save benchmark results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"competitor_benchmark_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
        # Also save as latest
        latest_file = self.results_dir / "latest_results.json"
        with open(latest_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def generate_report(self, results_file: str = None) -> str:
        """Generate human-readable comparison report."""
        if results_file:
            with open(results_file, 'r') as f:
                results = json.load(f)
        else:
            latest_file = self.results_dir / "latest_results.json"
            if not latest_file.exists():
                return "No results found. Run benchmarks first."
            with open(latest_file, 'r') as f:
                results = json.load(f)
        
        report = []
        report.append("# LLM Context Management Competitor Benchmark Report")
        report.append(f"Generated: {results['timestamp']}")
        report.append(f"Competitors tested: {len(results['results'])}")
        report.append("")
        
        # Summary table
        report.append("## Performance Summary")
        report.append("| Competitor | Status | Avg Processing Time | Avg Compression Ratio |")
        report.append("|------------|--------|-------------------|---------------------|")
        
        for competitor_name, competitor_results in results["results"].items():
            if competitor_results["status"] == "completed":
                test_results = competitor_results["test_results"]
                avg_time = sum(r.get("processing_time_ms", 0) for r in test_results) / len(test_results)
                avg_compression = sum(r.get("compression_ratio", 0) for r in test_results) / len(test_results)
                report.append(f"| {competitor_name} | ✅ Complete | {avg_time:.1f}ms | {avg_compression:.2f} |")
            else:
                status = competitor_results["status"]
                report.append(f"| {competitor_name} | ❌ {status} | - | - |")
        
        report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        for competitor_name, competitor_results in results["results"].items():
            report.append(f"### {competitor_name}")
            
            if competitor_results["status"] != "completed":
                report.append(f"Status: {competitor_results['status']}")
                if "error" in competitor_results:
                    report.append(f"Error: {competitor_results['error']}")
                if "requirements" in competitor_results:
                    report.append(f"Requirements: {', '.join(competitor_results['requirements'])}")
                report.append("")
                continue
            
            test_results = competitor_results["test_results"]
            for result in test_results:
                if "error" not in result:
                    report.append(f"**{result['test_case']}**: {result['processing_time_ms']:.1f}ms, "
                                f"{result['compression_ratio']:.2f} compression, "
                                f"{result['response_length']} chars response")
                else:
                    report.append(f"**{result['test_case']}**: Failed - {result['error']}")
            report.append("")
        
        return "\\n".join(report)


def main():
    """Main entry point for competitor benchmarking."""
    parser = argparse.ArgumentParser(description="LLM Context Management Competitor Benchmarks")
    parser.add_argument("--competitors", nargs="*", help="Specific competitors to test")
    parser.add_argument("--results-dir", help="Directory to store results")
    parser.add_argument("--report-only", action="store_true", help="Generate report from existing results")
    parser.add_argument("--report-file", help="Specific results file for report generation")
    
    args = parser.parse_args()
    
    runner = CompetitorBenchmarkRunner(results_dir=args.results_dir)
    
    if args.report_only:
        report = runner.generate_report(args.report_file)
        print(report)
        return
    
    # Run benchmarks
    logger.info("Starting LLM Context Management Competitor Benchmarks")
    results = runner.run_benchmarks(competitor_names=args.competitors)
    
    # Generate and display report
    report = runner.generate_report()
    print("\\n" + "="*80)
    print(report)


if __name__ == "__main__":
    main()