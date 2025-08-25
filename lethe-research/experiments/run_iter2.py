#!/usr/bin/env python3
"""
Iteration 2 Experiment Runner

Integrates query understanding components with the main Lethe pipeline
for comprehensive evaluation against Iteration 1 baseline.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

from query_understanding import QueryUnderstandingPipeline, create_iter2_config, QueryUnderstandingConfig
from run import LetheExperiment, ExperimentConfig
from score import compute_metrics

logger = logging.getLogger(__name__)


class Iteration2Experiment(LetheExperiment):
    """Extended experiment runner for Iteration 2 with query understanding."""
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        
        # Load Iteration 2 specific config
        self.iter2_config = self._load_iter2_config()
        self.query_pipeline = QueryUnderstandingPipeline(self.iter2_config)
        
    def _load_iter2_config(self) -> QueryUnderstandingConfig:
        """Load Iteration 2 query understanding configuration."""
        
        iter2_config_path = Path("experiments/iter2_query_understanding.yaml")
        
        if not iter2_config_path.exists():
            logger.warning(f"Iter2 config not found at {iter2_config_path}, using defaults")
            return QueryUnderstandingConfig()
            
        with open(iter2_config_path) as f:
            config_data = yaml.safe_load(f)
            
        # Extract parameters section  
        params = config_data.get("parameters", {})
        
        # Convert to QueryUnderstandingConfig
        return create_iter2_config({
            "query_rewrite": params.get("query_rewrite", {}).get("values", [True])[0],
            "query_decompose": params.get("query_decompose", {}).get("values", [True])[0], 
            "hyde_enabled": params.get("hyde_enabled", {}).get("values", [True])[0],
            "llm_model": params.get("llm_model", {}).get("values", ["gpt-4o-mini"])[0],
            "max_subqueries": params.get("max_subqueries", {}).get("values", [3])[0],
            "rewrite_strategy": params.get("rewrite_strategy", {}).get("values", ["both"])[0]
        })
    
    async def preprocess_query(self, query: str, domain: str = "mixed") -> Dict[str, Any]:
        """Preprocess query through understanding pipeline."""
        
        start_time = time.time()
        
        try:
            processed = await self.query_pipeline.process_query(query, domain)
            
            preprocessing_info = {
                "original_query": query,
                "processed_query": processed.rewritten_query or query,
                "subqueries": processed.subqueries or [],
                "hyde_documents": processed.hyde_documents or [],
                "preprocessing_time_ms": processed.processing_time_ms,
                "llm_calls_made": processed.llm_calls_made
            }
            
            return preprocessing_info
            
        except Exception as e:
            logger.error(f"Query preprocessing failed: {e}")
            
            # Fallback to original query
            return {
                "original_query": query,
                "processed_query": query, 
                "subqueries": [],
                "hyde_documents": [],
                "preprocessing_time_ms": (time.time() - start_time) * 1000,
                "llm_calls_made": 0,
                "error": str(e)
            }
    
    async def run_single_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run single query with Iteration 2 enhancements."""
        
        query = query_data["query"]
        domain = query_data.get("domain", "mixed")
        
        # Step 1: Query understanding preprocessing
        preprocessing = await self.preprocess_query(query, domain)
        
        # Step 2: Run retrieval with processed query
        processed_query = preprocessing["processed_query"]
        subqueries = preprocessing["subqueries"] 
        hyde_docs = preprocessing["hyde_documents"]
        
        # Combine query variants for enhanced retrieval
        retrieval_queries = [processed_query]
        if subqueries:
            retrieval_queries.extend(subqueries)
        if hyde_docs:
            retrieval_queries.extend(hyde_docs)
            
        # Run retrieval pipeline
        start_time = time.time()
        
        # For now, use primary processed query (can be enhanced later)
        retrieval_results = await self.retrieve_documents(processed_query, domain)
        
        retrieval_time_ms = (time.time() - start_time) * 1000
        
        # Combine timing
        total_time_ms = preprocessing["preprocessing_time_ms"] + retrieval_time_ms
        
        # Format results with Iteration 2 metrics
        result = {
            **query_data,
            "retrieved_docs": retrieval_results["documents"], 
            "relevance_scores": retrieval_results["scores"],
            "latency_ms": total_time_ms,
            "memory_mb": retrieval_results.get("memory_mb", 0),
            "preprocessing": preprocessing,
            "retrieval_time_ms": retrieval_time_ms,
            
            # Iteration 2 specific metrics
            "query_understanding": retrieval_results.get("query_understanding", {}),
            "typescript_integration": retrieval_results.get("typescript_integration", False),
            "rewrite_success": preprocessing.get("processed_query") != preprocessing.get("original_query"),
            "decompose_success": len(preprocessing.get("subqueries", [])) > 0,
            "llm_calls_made": preprocessing.get("llm_calls_made", 0),
            "preprocessing_errors": preprocessing.get("error", None),
            
            # Quality gate metrics for Iteration 2
            "quality_gates": {
                "latency_within_budget": total_time_ms <= 3500,  # 3500ms budget
                "rewrite_failure": not preprocessing.get("processed_query") or preprocessing.get("error"),
                "json_parse_error": "JSON" in str(preprocessing.get("error", "")),
                "hyde_skip": False  # Would be determined by actual HyDE usage
            },
            
            "entities_covered": [],  # TODO: Implement entity tracking
            "contradictions": [],     # TODO: Implement contradiction detection
            "timestamp": time.time()
        }
        
        return result
        
    async def retrieve_documents(self, query: str, domain: str) -> Dict[str, Any]:
        """Retrieve documents using the main Lethe pipeline with TypeScript integration."""
        
        try:
            import subprocess
            import json
            import tempfile
            import os
            
            # Path to the ctx-run system
            ctx_run_path = "/home/nathan/Projects/lethe/ctx-run"
            
            # Create a test query using the TypeScript system
            # This uses the CLI interface to the enhanced query system
            cmd = [
                "node",
                "-e",
                f"""
                const {{ enhancedQuery }} = require('./packages/core/dist/index.js');
                const {{ migrate }} = require('./packages/sqlite/dist/index.js');
                const Database = require('better-sqlite3');
                
                async function runQuery() {{
                    const db = new Database(':memory:');
                    migrate(db);
                    
                    // Set up Iteration 2 configuration
                    const {{ upsertConfig }} = require('./packages/sqlite/dist/index.js');
                    upsertConfig(db, 'plan', {{
                        query_rewrite: true,
                        query_decompose: true
                    }});
                    
                    upsertConfig(db, 'timeouts', {{
                        rewrite_ms: 1500,
                        decompose_ms: 2000
                    }});
                    
                    // Mock embeddings for testing
                    const mockEmbeddings = {{
                        embed: async (text) => Array(384).fill(0),
                        dimension: 384
                    }};
                    
                    try {{
                        const result = await enhancedQuery("{query}", {{
                            db,
                            embeddings: mockEmbeddings,
                            sessionId: "test-session",
                            enableQueryUnderstanding: true,
                            enableHyde: false, // Disable to focus on query understanding
                            enableSummarization: false,
                            enablePlanSelection: false,
                            recentTurns: []
                        }});
                        
                        console.log(JSON.stringify({{
                            success: true,
                            queryUnderstanding: result.queryUnderstanding,
                            duration: result.duration,
                            debug: result.debug
                        }}));
                    }} catch (error) {{
                        console.log(JSON.stringify({{
                            success: false,
                            error: error.message,
                            query: "{query}"
                        }}));
                    }}
                    
                    db.close();
                }}
                
                runQuery().catch(console.error);
                """
            ]
            
            # Run the command
            result = subprocess.run(
                cmd, 
                cwd=ctx_run_path,
                capture_output=True,
                text=True,
                timeout=10  # 10 second timeout
            )
            
            if result.returncode == 0:
                try:
                    response_data = json.loads(result.stdout.strip())
                    
                    if response_data.get("success", False):
                        qu_result = response_data.get("queryUnderstanding", {})
                        
                        return {
                            "documents": [],  # Would come from actual retrieval
                            "scores": [],
                            "memory_mb": 100,  # Estimated for Iteration 2
                            "query_understanding": qu_result,
                            "processing_time_ms": response_data.get("duration", {}).get("queryUnderstanding", 0),
                            "typescript_integration": True
                        }
                    else:
                        logger.warning(f"TypeScript query processing failed: {response_data.get('error', 'unknown')}")
                        
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse TypeScript response: {result.stdout}")
                    
            else:
                logger.warning(f"TypeScript process failed: {result.stderr}")
                
        except Exception as e:
            logger.warning(f"TypeScript integration failed: {e}")
        
        # Fallback to Python implementation
        return {
            "documents": [],  # Document IDs
            "scores": [],     # Relevance scores  
            "memory_mb": 50,   # Placeholder memory usage
            "typescript_integration": False
        }
        
    async def run_grid_search(self, grid_params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run grid search over Iteration 2 parameters."""
        
        results = []
        
        for i, params in enumerate(grid_params):
            logger.info(f"Running configuration {i+1}/{len(grid_params)}: {params}")
            
            # Create configuration for this grid cell
            config = create_iter2_config(params)
            self.query_pipeline = QueryUnderstandingPipeline(config)
            
            # Run evaluation on test queries
            config_results = await self.evaluate_configuration(params)
            
            results.append({
                "config_id": i,
                "parameters": params,
                "results": config_results
            })
            
        return results
        
    async def evaluate_configuration(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single parameter configuration."""
        
        # Load test queries (placeholder)
        test_queries = self._load_test_queries()
        
        query_results = []
        for query_data in test_queries:
            result = await self.run_single_query(query_data)
            query_results.append(result)
            
        # Compute metrics
        metrics = compute_metrics(query_results)
        
        return {
            "metrics": metrics,
            "query_results": query_results,
            "config": params
        }
        
    def _load_test_queries(self) -> List[Dict[str, Any]]:
        """Load test queries for evaluation."""
        
        # Placeholder - load from LetheBench dataset
        return [
            {
                "query_id": f"query_{i:03d}",
                "query": f"Test query {i}",
                "domain": "mixed",
                "complexity": "medium",
                "ground_truth_docs": [f"doc_{j:03d}" for j in range(i*5, (i+1)*5)]
            }
            for i in range(20)  # 20 test queries
        ]


def generate_iter2_grid() -> List[Dict[str, Any]]:
    """Generate grid search parameters for Iteration 2."""
    
    grid_params = []
    
    # Core combinations
    for query_rewrite in [True, False]:
        for query_decompose in [True, False]:
            for hyde_enabled in [True, False]:
                
                # Skip all-false configuration (equivalent to Iteration 1)
                if not any([query_rewrite, query_decompose, hyde_enabled]):
                    continue
                    
                base_config = {
                    "query_rewrite": query_rewrite,
                    "query_decompose": query_decompose, 
                    "hyde_enabled": hyde_enabled,
                    "llm_model": "gpt-4o-mini"
                }
                
                # Add conditional parameters
                if query_decompose:
                    for max_subqueries in [2, 3]:
                        config = base_config.copy()
                        config["max_subqueries"] = max_subqueries
                        grid_params.append(config)
                else:
                    grid_params.append(base_config)
                    
    return grid_params


async def run_iteration2_experiment():
    """Main entry point for Iteration 2 experiments."""
    
    logger.info("Starting Iteration 2: Query Understanding experiment")
    
    # Initialize experiment
    config_path = "experiments/iter1_dev.yaml"  # Base config from Iteration 1
    experiment = Iteration2Experiment(config_path)
    
    # Generate grid search parameters
    grid_params = generate_iter2_grid()
    logger.info(f"Running grid search with {len(grid_params)} configurations")
    
    # Run experiments
    results = await experiment.run_grid_search(grid_params)
    
    # Save results
    timestamp = int(time.time())
    results_path = f"artifacts/iter2_results_{timestamp}.json"
    
    os.makedirs("artifacts", exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Results saved to {results_path}")
    
    # Analyze and report best configuration
    best_config = analyze_iter2_results(results)
    logger.info(f"Best configuration: {best_config}")
    
    return results


def analyze_iter2_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze Iteration 2 results and identify best configuration."""
    
    best_config = None
    best_ndcg = 0
    
    for result in results:
        metrics = result["results"]["metrics"]
        ndcg = metrics.get("ndcg_at_10", 0)
        
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_config = result["parameters"]
            
    return {
        "parameters": best_config,
        "ndcg_at_10": best_ndcg,
        "improvement_over_baseline": best_ndcg  # TODO: Compare with Iteration 1 baseline
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Iteration 2 experiments")
    parser.add_argument("--single-query", help="Test single query")
    parser.add_argument("--domain", default="mixed", help="Query domain")
    parser.add_argument("--full-grid", action="store_true", help="Run full grid search")
    
    args = parser.parse_args()
    
    if args.single_query:
        # Test single query
        async def test_single():
            config_path = "experiments/iter1_dev.yaml"
            experiment = Iteration2Experiment(config_path)
            
            query_data = {
                "query": args.single_query,
                "domain": args.domain,
                "query_id": "test_001"
            }
            
            result = await experiment.run_single_query(query_data)
            print(json.dumps(result, indent=2))
            
        asyncio.run(test_single())
        
    elif args.full_grid:
        # Run full experiment
        asyncio.run(run_iteration2_experiment())
        
    else:
        print("Use --single-query or --full-grid")