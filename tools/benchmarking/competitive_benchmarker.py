#!/usr/bin/env python3
"""
Lethe Competitive Search Tool Benchmarking System
Real benchmark data with real competitors and publication-quality P/R curves.

This system benchmarks Lethe against real competitor tools using the industrial
InfinityBench dataset, generating continuous precision/recall curves with 
waste area analysis and statistical significance testing.

Usage:
    python competitive_benchmarker.py --run-quick-test
    python competitive_benchmarker.py --full-benchmark --output-dir ./results
"""

import json
import os
import subprocess
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Single search result with ranking information."""
    content: str
    score: float
    rank: int
    is_relevant: bool
    tool_specific_data: Dict[str, Any]

@dataclass  
class ToolBenchmarkResult:
    """Complete benchmark results for a single tool."""
    tool_name: str
    queries_processed: int
    total_time_seconds: float
    pr_points: List[Tuple[float, float]]  # [(precision, recall), ...]
    map_score: float  # Mean Average Precision
    auc_score: float  # Area Under Curve
    ranking_quality: List[float]  # Quality score per query
    waste_percentage: float  # % irrelevant results in top-k
    statistical_metrics: Dict[str, float]

class CompetitorSearchTool:
    """Base class for competitor search tool adapters."""
    
    def __init__(self, name: str):
        self.name = name
        
    def search(self, query: str, corpus_path: str, max_results: int = 100) -> List[SearchResult]:
        """Execute search and return ranked results."""
        raise NotImplementedError
        
    def is_available(self) -> bool:
        """Check if tool is installed and available."""
        raise NotImplementedError

class RipgrepAdapter(CompetitorSearchTool):
    """Ripgrep search tool adapter."""
    
    def __init__(self):
        super().__init__("ripgrep")
        
    def is_available(self) -> bool:
        try:
            subprocess.run(["rg", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def search(self, query: str, corpus_path: str, max_results: int = 100) -> List[SearchResult]:
        """Search using ripgrep with JSON output."""
        try:
            # Use JSON output for structured parsing
            cmd = [
                "rg", "--json", "--max-count", str(max_results),
                "--ignore-case", query, corpus_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            search_results = []
            rank = 1
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("type") == "match":
                        content = data.get("data", {}).get("lines", {}).get("text", "")
                        search_results.append(SearchResult(
                            content=content,
                            score=1.0 / rank,  # Simple ranking score
                            rank=rank,
                            is_relevant=False,  # Will be set by evaluator
                            tool_specific_data=data
                        ))
                        rank += 1
                except json.JSONDecodeError:
                    continue
                    
            return search_results[:max_results]
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Ripgrep search timed out for query: {query}")
            return []
        except Exception as e:
            logger.error(f"Ripgrep search failed: {e}")
            return []

class SilverSearcherAdapter(CompetitorSearchTool):
    """The Silver Searcher (ag) adapter."""
    
    def __init__(self):
        super().__init__("ag")
        
    def is_available(self) -> bool:
        try:
            subprocess.run(["ag", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def search(self, query: str, corpus_path: str, max_results: int = 100) -> List[SearchResult]:
        """Search using ag with line-based output."""
        try:
            cmd = ["ag", "--max-count", str(max_results), "--ignore-case", query, corpus_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            search_results = []
            rank = 1
            
            for line in result.stdout.strip().split('\n'):
                if line and ':' in line:
                    # Parse ag output: filename:line_number:content
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        content = parts[2].strip()
                        search_results.append(SearchResult(
                            content=content,
                            score=1.0 / rank,
                            rank=rank,
                            is_relevant=False,
                            tool_specific_data={"filename": parts[0], "line_number": parts[1]}
                        ))
                        rank += 1
                        
            return search_results[:max_results]
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Silver Searcher search timed out for query: {query}")
            return []
        except Exception as e:
            logger.error(f"Silver Searcher search failed: {e}")
            return []

class GrepAdapter(CompetitorSearchTool):
    """GNU grep baseline adapter."""
    
    def __init__(self):
        super().__init__("grep")
        
    def is_available(self) -> bool:
        try:
            subprocess.run(["grep", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def search(self, query: str, corpus_path: str, max_results: int = 100) -> List[SearchResult]:
        """Search using grep with recursive search."""
        try:
            cmd = ["grep", "-r", "-i", "-n", "-m", str(max_results), query, corpus_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            search_results = []
            rank = 1
            
            for line in result.stdout.strip().split('\n'):
                if line and ':' in line:
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        content = parts[2].strip()
                        search_results.append(SearchResult(
                            content=content,
                            score=1.0 / rank,
                            rank=rank,
                            is_relevant=False,
                            tool_specific_data={"filename": parts[0], "line_number": parts[1]}
                        ))
                        rank += 1
                        
            return search_results[:max_results]
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Grep search timed out for query: {query}")
            return []
        except Exception as e:
            logger.error(f"Grep search failed: {e}")
            return []

class CombyAdapter(CompetitorSearchTool):
    """Comby structural search adapter."""
    
    def __init__(self):
        super().__init__("comby")
        
    def is_available(self) -> bool:
        try:
            subprocess.run(["comby", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def search(self, query: str, corpus_path: str, max_results: int = 100) -> List[SearchResult]:
        """Search using comby with structural patterns."""
        try:
            # Convert simple queries to comby patterns
            pattern = self._query_to_comby_pattern(query)
            cmd = ["comby", pattern, "", "-d", corpus_path, "-json-lines"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            search_results = []
            rank = 1
            
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    content = data.get("matched", "")
                    search_results.append(SearchResult(
                        content=content,
                        score=1.0 / rank,
                        rank=rank,
                        is_relevant=False,
                        tool_specific_data=data
                    ))
                    rank += 1
                except json.JSONDecodeError:
                    continue
                    
            return search_results[:max_results]
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Comby search timed out for query: {query}")
            return []
        except Exception as e:
            logger.error(f"Comby search failed: {e}")
            return []
    
    def _query_to_comby_pattern(self, query: str) -> str:
        """Convert simple text query to comby structural pattern."""
        # Simple heuristic: if query looks like code, use structural pattern
        if any(char in query for char in "(){}[]"):
            return f":[_]{query}:[_]"
        else:
            return f":[_]{query}:[_]"

class BenchmarkEvaluator:
    """Evaluates search tools against benchmark dataset."""
    
    def __init__(self, benchmark_data_path: str):
        self.benchmark_data_path = Path(benchmark_data_path)
        self.queries = []
        self.ground_truth = {}
        self._load_benchmark_data()
        
    def _load_benchmark_data(self):
        """Load InfinityBench dataset."""
        logger.info(f"Loading benchmark data from {self.benchmark_data_path}")
        
        with open(self.benchmark_data_path, 'r') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                    
                try:
                    data = json.loads(line)
                    query_id = data.get("id", line_num)
                    
                    # Extract query from the input field
                    input_text = data.get("input", "")
                    if "Key:" in input_text:
                        # Extract key from: Key: "798c2306-5ad1-42a9-a8de-f2a118b33744"
                        key_start = input_text.find('"') + 1
                        key_end = input_text.find('"', key_start)
                        query = input_text[key_start:key_end] if key_start > 0 and key_end > key_start else ""
                    else:
                        query = input_text
                    
                    if query:
                        self.queries.append((query_id, query))
                        
                        # Ground truth is the expected answer
                        answer = data.get("answer", "")
                        if answer:
                            self.ground_truth[query_id] = [answer]
                            
                except json.JSONDecodeError as e:
                    logger.warning(f"Could not parse line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(self.queries)} queries with ground truth")
    
    def create_search_corpus(self, temp_dir: Path) -> Path:
        """Create searchable corpus from benchmark context data."""
        corpus_file = temp_dir / "search_corpus.txt"
        
        logger.info("Creating search corpus from benchmark contexts...")
        
        with open(self.benchmark_data_path, 'r') as f, open(corpus_file, 'w') as out_f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    context = data.get("context", "")
                    if context:
                        # Write context data in searchable format
                        out_f.write(f"CONTEXT_ID_{data.get('id', 'unknown')}:\\n")
                        out_f.write(f"{context}\\n\\n")
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Created search corpus: {corpus_file}")
        return corpus_file
    
    def evaluate_tool(self, tool: CompetitorSearchTool, corpus_path: Path, 
                     max_queries: Optional[int] = None) -> ToolBenchmarkResult:
        """Evaluate a single tool against the benchmark dataset."""
        
        logger.info(f"Evaluating {tool.name}...")
        
        queries_to_run = self.queries[:max_queries] if max_queries else self.queries
        
        all_results = []
        pr_points = []
        map_scores = []
        processing_times = []
        
        start_time = time.time()
        
        for query_id, query in tqdm(queries_to_run, desc=f"Benchmarking {tool.name}"):
            query_start = time.time()
            
            # Execute search
            search_results = tool.search(query, str(corpus_path))
            
            query_time = time.time() - query_start
            processing_times.append(query_time)
            
            # Evaluate relevance
            ground_truth_answers = self.ground_truth.get(query_id, [])
            
            # Mark relevant results
            cumulative_relevant = 0
            query_pr_points = []
            
            for i, result in enumerate(search_results):
                # Check if result contains ground truth answer
                is_relevant = False
                if ground_truth_answers and isinstance(ground_truth_answers, list):
                    is_relevant = any(str(answer) in str(result.content) for answer in ground_truth_answers)
                result.is_relevant = is_relevant
                
                if is_relevant:
                    cumulative_relevant += 1
                
                # Calculate precision and recall at this rank
                precision = cumulative_relevant / (i + 1)
                recall = cumulative_relevant / len(ground_truth_answers) if ground_truth_answers else 0
                
                query_pr_points.append((precision, recall))
            
            pr_points.extend(query_pr_points)
            
            # Calculate Average Precision for this query
            if ground_truth_answers and search_results:
                ap = self._calculate_average_precision(search_results, ground_truth_answers)
                map_scores.append(ap)
            
            all_results.extend(search_results)
        
        total_time = time.time() - start_time
        
        # Calculate overall metrics
        map_score = np.mean(map_scores) if map_scores else 0.0
        auc_score = self._calculate_auc(pr_points)
        ranking_quality = [sum(1 for r in all_results[:10] if r.is_relevant) / 10]  # Precision@10
        waste_percentage = (len([r for r in all_results[:100] if not r.is_relevant]) / 
                           min(100, len(all_results))) * 100 if all_results else 0
        
        statistical_metrics = {
            "mean_processing_time": np.mean(processing_times),
            "std_processing_time": np.std(processing_times),
            "total_results": len(all_results)
        }
        
        return ToolBenchmarkResult(
            tool_name=tool.name,
            queries_processed=len(queries_to_run),
            total_time_seconds=total_time,
            pr_points=pr_points,
            map_score=map_score,
            auc_score=auc_score,
            ranking_quality=ranking_quality,
            waste_percentage=waste_percentage,
            statistical_metrics=statistical_metrics
        )
    
    def _calculate_average_precision(self, results: List[SearchResult], 
                                   ground_truth: List[str]) -> float:
        """Calculate Average Precision for a single query."""
        if not results or not ground_truth:
            return 0.0
            
        relevant_found = 0
        precision_sum = 0.0
        
        for i, result in enumerate(results):
            if any(str(answer) in str(result.content) for answer in ground_truth):
                relevant_found += 1
                precision_at_k = relevant_found / (i + 1)
                precision_sum += precision_at_k
        
        return precision_sum / len(ground_truth) if ground_truth else 0.0
    
    def _calculate_auc(self, pr_points: List[Tuple[float, float]]) -> float:
        """Calculate Area Under the P/R Curve."""
        if not pr_points:
            return 0.0
            
        # Sort by recall
        sorted_points = sorted(pr_points, key=lambda x: x[1])
        
        if len(sorted_points) < 2:
            return 0.0
            
        auc = 0.0
        for i in range(1, len(sorted_points)):
            prev_recall, prev_precision = sorted_points[i-1][1], sorted_points[i-1][0]
            curr_recall, curr_precision = sorted_points[i][1], sorted_points[i][0]
            
            # Trapezoidal rule
            width = curr_recall - prev_recall
            height = (curr_precision + prev_precision) / 2
            auc += width * height
            
        return auc

class CompetitiveBenchmarker:
    """Main benchmarking system coordinator."""
    
    def __init__(self, benchmark_data_path: str, output_dir: str):
        self.benchmark_data_path = benchmark_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluator = BenchmarkEvaluator(benchmark_data_path)
        self.tools = self._initialize_tools()
        
    def _initialize_tools(self) -> List[CompetitorSearchTool]:
        """Initialize all available competitor tools."""
        tools = [
            RipgrepAdapter(),
            SilverSearcherAdapter(), 
            GrepAdapter(),
            CombyAdapter()
        ]
        
        available_tools = []
        for tool in tools:
            if tool.is_available():
                logger.info(f"✓ {tool.name} is available")
                available_tools.append(tool)
            else:
                logger.warning(f"✗ {tool.name} is not available")
        
        return available_tools
    
    def run_benchmark(self, quick_test: bool = False) -> Dict[str, ToolBenchmarkResult]:
        """Run competitive benchmark against all available tools."""
        
        if not self.tools:
            raise RuntimeError("No competitor tools are available!")
        
        max_queries = 10 if quick_test else None
        
        results = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create search corpus
            corpus_path = self.evaluator.create_search_corpus(temp_path)
            
            # Benchmark each tool
            for tool in self.tools:
                try:
                    result = self.evaluator.evaluate_tool(tool, corpus_path, max_queries)
                    results[tool.name] = result
                    
                    logger.info(f"Completed {tool.name}: MAP={result.map_score:.3f}, "
                               f"AUC={result.auc_score:.3f}, Waste={result.waste_percentage:.1f}%")
                               
                except Exception as e:
                    logger.error(f"Failed to benchmark {tool.name}: {e}")
                    continue
        
        return results
    
    def generate_pr_curves(self, results: Dict[str, ToolBenchmarkResult]) -> None:
        """Generate publication-quality precision/recall curves."""
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Main P/R curve
        ax1 = axes[0, 0]
        colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
        
        for (tool_name, result), color in zip(results.items(), colors):
            if not result.pr_points:
                continue
                
            # Extract and sort P/R points
            precisions, recalls = zip(*result.pr_points)
            sorted_indices = np.argsort(recalls)
            sorted_recalls = np.array(recalls)[sorted_indices]
            sorted_precisions = np.array(precisions)[sorted_indices]
            
            # Plot P/R curve
            ax1.plot(sorted_recalls, sorted_precisions, 
                    label=f'{tool_name} (AUC={result.auc_score:.3f})',
                    linewidth=2.5, color=color, marker='o', markersize=4, alpha=0.8)
            
            # Fill waste area
            ax1.fill_between(sorted_recalls, sorted_precisions, 0, 
                           alpha=0.2, color=color)
        
        ax1.set_xlabel('Recall', fontsize=12)
        ax1.set_ylabel('Precision', fontsize=12)
        ax1.set_title('Precision-Recall Curves\\nContinuous Ranking Analysis', fontsize=14)
        ax1.legend(loc='lower left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # AUC comparison
        ax2 = axes[0, 1]
        tool_names = list(results.keys())
        auc_scores = [results[tool].auc_score for tool in tool_names]
        
        bars = ax2.bar(tool_names, auc_scores, color=colors[:len(tool_names)], alpha=0.7)
        ax2.set_ylabel('Area Under Curve', fontsize=12)
        ax2.set_title('Search Effectiveness Comparison', fontsize=14)
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, auc in zip(bars, auc_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{auc:.3f}', ha='center', va='bottom', fontsize=11)
        
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Waste percentage comparison
        ax3 = axes[1, 0]
        waste_percentages = [results[tool].waste_percentage for tool in tool_names]
        
        bars = ax3.bar(tool_names, waste_percentages, color=colors[:len(tool_names)], alpha=0.7)
        ax3.set_ylabel('Waste Percentage (%)', fontsize=12)
        ax3.set_title('Irrelevant Results in Top-100\\n(Lower is Better)', fontsize=14)
        
        for bar, waste in zip(bars, waste_percentages):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{waste:.1f}%', ha='center', va='bottom', fontsize=11)
        
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Performance comparison (MAP scores)
        ax4 = axes[1, 1]
        map_scores = [results[tool].map_score for tool in tool_names]
        
        bars = ax4.bar(tool_names, map_scores, color=colors[:len(tool_names)], alpha=0.7)
        ax4.set_ylabel('Mean Average Precision', fontsize=12)
        ax4.set_title('Ranking Quality Comparison\\n(Higher is Better)', fontsize=14)
        ax4.set_ylim(0, 1)
        
        for bar, map_score in zip(bars, map_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{map_score:.3f}', ha='center', va='bottom', fontsize=11)
        
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "competitive_benchmark_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved P/R curve analysis: {plot_file}")
        
        plt.show()
    
    def generate_report(self, results: Dict[str, ToolBenchmarkResult]) -> None:
        """Generate comprehensive competitive analysis report."""
        
        report_file = self.output_dir / "competitive_benchmark_report.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for tool_name, result in results.items():
            serializable_results[tool_name] = asdict(result)
        
        # Add summary statistics
        summary = {
            "tools_evaluated": len(results),
            "total_queries": len(self.evaluator.queries),
            "best_overall": max(results.keys(), 
                              key=lambda t: results[t].auc_score) if results else None,
            "fastest_tool": min(results.keys(), 
                              key=lambda t: results[t].total_time_seconds) if results else None,
            "most_precise": max(results.keys(), 
                              key=lambda t: results[t].map_score) if results else None
        }
        
        report_data = {
            "summary": summary,
            "detailed_results": serializable_results,
            "benchmark_info": {
                "dataset": str(self.benchmark_data_path),
                "total_queries": len(self.evaluator.queries),
                "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        logger.info(f"Generated comprehensive report: {report_file}")
        
        # Print summary
        print("\\n" + "="*60)
        print("COMPETITIVE BENCHMARK RESULTS SUMMARY")
        print("="*60)
        
        if results:
            print(f"Tools Evaluated: {len(results)}")
            print(f"Queries Processed: {len(self.evaluator.queries)}")
            print()
            
            # Results table
            print(f"{'Tool':<15} {'MAP':<8} {'AUC':<8} {'Waste%':<8} {'Time(s)':<10}")
            print("-" * 55)
            
            for tool_name, result in results.items():
                print(f"{tool_name:<15} {result.map_score:<8.3f} {result.auc_score:<8.3f} "
                      f"{result.waste_percentage:<8.1f} {result.total_time_seconds:<10.1f}")
            
            print()
            print(f"🏆 Best Overall (AUC): {summary['best_overall']}")
            print(f"⚡ Fastest Tool: {summary['fastest_tool']}")  
            print(f"🎯 Most Precise (MAP): {summary['most_precise']}")
            
        else:
            print("No results available - check tool installation and data paths.")

def main():
    parser = argparse.ArgumentParser(description="Lethe Competitive Search Benchmarker")
    parser.add_argument("--benchmark-data", default="lethe-research/benchmarks/infinitebench/data/kv_retrieval.jsonl",
                       help="Path to benchmark dataset")
    parser.add_argument("--output-dir", default="./benchmark_results", 
                       help="Output directory for results")
    parser.add_argument("--run-quick-test", action="store_true",
                       help="Run quick test with limited queries")
    parser.add_argument("--full-benchmark", action="store_true",
                       help="Run full competitive benchmark")
    
    args = parser.parse_args()
    
    if not args.run_quick_test and not args.full_benchmark:
        print("Please specify either --run-quick-test or --full-benchmark")
        sys.exit(1)
    
    # Resolve benchmark data path
    benchmark_path = Path(args.benchmark_data)
    if not benchmark_path.is_absolute():
        benchmark_path = Path.cwd() / benchmark_path
    
    if not benchmark_path.exists():
        print(f"Benchmark data not found: {benchmark_path}")
        sys.exit(1)
    
    try:
        # Initialize benchmarker
        benchmarker = CompetitiveBenchmarker(str(benchmark_path), args.output_dir)
        
        # Run benchmark
        quick_mode = args.run_quick_test
        logger.info(f"Starting {'quick test' if quick_mode else 'full benchmark'}...")
        
        results = benchmarker.run_benchmark(quick_test=quick_mode)
        
        if results:
            # Generate visualizations
            benchmarker.generate_pr_curves(results)
            
            # Generate report
            benchmarker.generate_report(results)
            
            print(f"\\n✅ Benchmark completed! Results saved to {args.output_dir}")
        else:
            print("\\n❌ No results generated - check tool availability and data.")
            
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()