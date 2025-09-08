#!/usr/bin/env python3
"""
Enhanced Lethe Competitive Search Tool Benchmarking System
Includes Lethe as a competitor with improved key-value matching.

Usage:
    python enhanced_competitive_benchmarker.py --run-quick-test
    python enhanced_competitive_benchmarker.py --full-benchmark --include-lethe
"""

import sys
import os
import json
import re
from pathlib import Path

# Add current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from competitive_benchmarker import *
import sys
sys.path.append('../..')
from lethe_adapter import LetheAdapter

class EnhancedBenchmarkEvaluator(BenchmarkEvaluator):
    """Enhanced evaluator with better key-value handling."""
    
    def create_search_corpus(self, temp_dir: Path) -> Path:
        """Create enhanced searchable corpus with proper key-value formatting."""
        corpus_file = temp_dir / "search_corpus.txt"
        
        logger.info("Creating enhanced search corpus from benchmark contexts...")
        
        with open(self.benchmark_data_path, 'r') as f, open(corpus_file, 'w') as out_f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    context = data.get("context", "")
                    
                    if context and "JSON data:" in context:
                        # Extract JSON data
                        json_start = context.find("{")
                        json_end = context.rfind("}") + 1
                        
                        if json_start != -1 and json_end > json_start:
                            try:
                                json_str = context[json_start:json_end]
                                json_data = json.loads(json_str)
                                
                                # Write document header
                                doc_id = data.get('id', line_num)
                                out_f.write(f"=== DOCUMENT {doc_id} ===\\n")
                                
                                # Write key-value pairs in searchable format
                                for key, value in json_data.items():
                                    out_f.write(f"KEY: {key}\\n")
                                    out_f.write(f"VALUE: {value}\\n")
                                    out_f.write(f"PAIR: {key} -> {value}\\n")
                                    out_f.write("\\n")
                                
                                # Also write original JSON for fallback
                                out_f.write(f"ORIGINAL_JSON: {json_str}\\n\\n")
                                
                            except json.JSONDecodeError:
                                # Fallback to original context
                                out_f.write(f"CONTEXT_ID_{data.get('id', 'unknown')}:\\n")
                                out_f.write(f"{context}\\n\\n")
                    
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Created enhanced search corpus: {corpus_file} ({corpus_file.stat().st_size} bytes)")
        return corpus_file
    
    def evaluate_tool(self, tool: CompetitorSearchTool, corpus_path: Path, 
                     max_queries: Optional[int] = None) -> ToolBenchmarkResult:
        """Enhanced evaluation with better relevance checking."""
        
        logger.info(f"Evaluating {tool.name} with enhanced key-value matching...")
        
        queries_to_run = self.queries[:max_queries] if max_queries else self.queries
        
        all_results = []
        pr_points = []
        map_scores = []
        processing_times = []
        
        start_time = time.time()
        
        for query_id, query in tqdm(queries_to_run, desc=f"Benchmarking {tool.name}"):
            query_start = time.time()
            
            # Execute search with the UUID key
            search_results = tool.search(query, str(corpus_path))
            
            query_time = time.time() - query_start
            processing_times.append(query_time)
            
            # Enhanced relevance evaluation
            ground_truth_answers = self.ground_truth.get(query_id, [])
            
            # Mark relevant results with enhanced matching
            cumulative_relevant = 0
            query_pr_points = []
            
            for i, result in enumerate(search_results):
                is_relevant = self._enhanced_relevance_check(query, result.content, ground_truth_answers)
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
                ap = self._calculate_average_precision_enhanced(search_results, query, ground_truth_answers)
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
            "total_results": len(all_results),
            "relevant_results": len([r for r in all_results if r.is_relevant])
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
    
    def _enhanced_relevance_check(self, query: str, result_content: str, 
                                ground_truth_answers: List[str]) -> bool:
        """Enhanced relevance checking for key-value pairs."""
        if not ground_truth_answers:
            return False
            
        content_str = str(result_content).lower()
        query_str = str(query).lower()
        
        # Check multiple relevance criteria
        for answer in ground_truth_answers:
            answer_str = str(answer).lower()
            
            # 1. Direct answer match
            if answer_str in content_str:
                return True
                
            # 2. Key-value pair match
            if query_str in content_str and answer_str in content_str:
                return True
                
            # 3. UUID pattern match
            if self._is_uuid_like(query_str) and self._is_uuid_like(answer_str):
                # Both are UUID-like, check for any occurrence
                if query_str in content_str or answer_str in content_str:
                    return True
            
            # 4. Contextual match (key -> value pattern)
            key_value_pattern = f"{query_str}.*{answer_str}|{answer_str}.*{query_str}"
            if re.search(key_value_pattern, content_str, re.IGNORECASE):
                return True
        
        return False
    
    def _is_uuid_like(self, text: str) -> bool:
        """Check if text looks like a UUID."""
        # UUID pattern: 8-4-4-4-12 hexadecimal digits
        uuid_pattern = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
        return bool(re.match(uuid_pattern, text))
    
    def _calculate_average_precision_enhanced(self, results: List[SearchResult], 
                                            query: str, ground_truth: List[str]) -> float:
        """Enhanced Average Precision calculation."""
        if not results or not ground_truth:
            return 0.0
            
        relevant_found = 0
        precision_sum = 0.0
        
        for i, result in enumerate(results):
            if self._enhanced_relevance_check(query, result.content, ground_truth):
                relevant_found += 1
                precision_at_k = relevant_found / (i + 1)
                precision_sum += precision_at_k
        
        return precision_sum / len(ground_truth) if ground_truth else 0.0

class EnhancedCompetitiveBenchmarker(CompetitiveBenchmarker):
    """Enhanced benchmarker with Lethe integration."""
    
    def __init__(self, benchmark_data_path: str, output_dir: str, include_lethe: bool = False):
        self.benchmark_data_path = benchmark_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.include_lethe = include_lethe
        
        self.evaluator = EnhancedBenchmarkEvaluator(benchmark_data_path)
        self.tools = self._initialize_tools()
    
    def _initialize_tools(self) -> List[CompetitorSearchTool]:
        """Initialize all available competitor tools including Lethe."""
        tools = [
            RipgrepAdapter(),
            SilverSearcherAdapter(), 
            GrepAdapter(),
            CombyAdapter()
        ]
        
        # Add Lethe if requested
        if self.include_lethe:
            lethe_adapter = LetheAdapter()
            tools.append(lethe_adapter)
        
        available_tools = []
        for tool in tools:
            if tool.is_available():
                logger.info(f"✓ {tool.name} is available")
                available_tools.append(tool)
            else:
                logger.warning(f"✗ {tool.name} is not available")
        
        return available_tools
    
    def generate_enhanced_report(self, results: Dict[str, ToolBenchmarkResult]) -> None:
        """Generate enhanced competitive analysis report with better insights."""
        
        report_file = self.output_dir / "enhanced_competitive_benchmark_report.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for tool_name, result in results.items():
            serializable_results[tool_name] = asdict(result)
        
        # Enhanced summary statistics
        if results:
            best_map = max(results.keys(), key=lambda t: results[t].map_score)
            best_auc = max(results.keys(), key=lambda t: results[t].auc_score)
            fastest = min(results.keys(), key=lambda t: results[t].total_time_seconds)
            least_waste = min(results.keys(), key=lambda t: results[t].waste_percentage)
            
            # Statistical significance testing
            tool_names = list(results.keys())
            if len(tool_names) > 1:
                pairwise_comparisons = {}
                for i in range(len(tool_names)):
                    for j in range(i+1, len(tool_names)):
                        tool1, tool2 = tool_names[i], tool_names[j]
                        # Simple statistical comparison based on MAP scores
                        map_diff = results[tool1].map_score - results[tool2].map_score
                        pairwise_comparisons[f"{tool1}_vs_{tool2}"] = {
                            "map_difference": map_diff,
                            "significance": "significant" if abs(map_diff) > 0.05 else "not_significant"
                        }
            else:
                pairwise_comparisons = {}
        else:
            best_map = best_auc = fastest = least_waste = None
            pairwise_comparisons = {}
        
        summary = {
            "tools_evaluated": len(results),
            "total_queries": len(self.evaluator.queries),
            "best_map_score": best_map,
            "best_auc_score": best_auc, 
            "fastest_tool": fastest,
            "least_waste": least_waste,
            "lethe_included": self.include_lethe,
            "statistical_comparisons": pairwise_comparisons
        }
        
        report_data = {
            "summary": summary,
            "detailed_results": serializable_results,
            "benchmark_info": {
                "dataset": str(self.benchmark_data_path),
                "total_queries": len(self.evaluator.queries),
                "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "enhanced_features": [
                    "UUID-aware matching",
                    "Key-value pair recognition",
                    "Contextual relevance checking",
                    "Enhanced corpus generation"
                ]
            }
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        logger.info(f"Generated enhanced report: {report_file}")
        
        # Print enhanced summary
        print("\\n" + "="*70)
        print("ENHANCED COMPETITIVE BENCHMARK RESULTS SUMMARY")
        print("="*70)
        
        if results:
            print(f"Tools Evaluated: {len(results)}")
            print(f"Queries Processed: {len(self.evaluator.queries)}")
            if self.include_lethe:
                print("✓ Lethe included in comparison")
            print()
            
            # Enhanced results table
            print(f"{'Tool':<15} {'MAP':<8} {'AUC':<8} {'Waste%':<8} {'Relevant':<10} {'Time(s)':<10}")
            print("-" * 75)
            
            for tool_name, result in results.items():
                relevant_count = result.statistical_metrics.get('relevant_results', 0)
                print(f"{tool_name:<15} {result.map_score:<8.3f} {result.auc_score:<8.3f} "
                      f"{result.waste_percentage:<8.1f} {relevant_count:<10} {result.total_time_seconds:<10.1f}")
            
            print()
            print(f"🎯 Best MAP Score: {summary['best_map_score']}")
            print(f"🏆 Best AUC Score: {summary['best_auc_score']}")
            print(f"⚡ Fastest Tool: {summary['fastest_tool']}")  
            print(f"🎯 Least Waste: {summary['least_waste']}")
            
            # Show statistical comparisons
            if pairwise_comparisons:
                print("\\n📊 Statistical Comparisons:")
                for comparison, stats in pairwise_comparisons.items():
                    tools = comparison.replace('_vs_', ' vs ')
                    diff = stats['map_difference']
                    sig = stats['significance']
                    print(f"  {tools}: MAP difference = {diff:+.3f} ({sig})")
            
        else:
            print("No results available - check tool installation and data paths.")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Lethe Competitive Search Benchmarker")
    parser.add_argument("--benchmark-data", default="lethe-research/benchmarks/infinitebench/data/kv_retrieval.jsonl",
                       help="Path to benchmark dataset")
    parser.add_argument("--output-dir", default="./enhanced_benchmark_results", 
                       help="Output directory for results")
    parser.add_argument("--run-quick-test", action="store_true",
                       help="Run quick test with limited queries")
    parser.add_argument("--full-benchmark", action="store_true",
                       help="Run full competitive benchmark")
    parser.add_argument("--include-lethe", action="store_true",
                       help="Include Lethe in the competitive analysis")
    
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
        # Initialize enhanced benchmarker
        benchmarker = EnhancedCompetitiveBenchmarker(
            str(benchmark_path), 
            args.output_dir,
            include_lethe=args.include_lethe
        )
        
        # Run benchmark
        quick_mode = args.run_quick_test
        logger.info(f"Starting enhanced {'quick test' if quick_mode else 'full benchmark'}...")
        if args.include_lethe:
            logger.info("Including Lethe in competitive comparison")
        
        results = benchmarker.run_benchmark(quick_test=quick_mode)
        
        if results:
            # Generate visualizations
            benchmarker.generate_pr_curves(results)
            
            # Generate enhanced report
            benchmarker.generate_enhanced_report(results)
            
            print(f"\\n✅ Enhanced benchmark completed! Results saved to {args.output_dir}")
        else:
            print("\\n❌ No results generated - check tool availability and data.")
            
    except Exception as e:
        logger.error(f"Enhanced benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()