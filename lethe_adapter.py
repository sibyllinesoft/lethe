#!/usr/bin/env python3
"""
Lethe Search Tool Adapter for Competitive Benchmarking
Integrates Lethe as a competitor in the benchmarking suite.
"""

import json
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Any
import sys
sys.path.append(str(Path(__file__).parent / 'tools' / 'benchmarking'))
from competitive_benchmarker import CompetitorSearchTool, SearchResult

logger = logging.getLogger(__name__)

class LetheAdapter(CompetitorSearchTool):
    """Lethe search system adapter for competitive benchmarking."""
    
    def __init__(self, lethe_binary_path: str = None):
        super().__init__("Lethe")
        self.lethe_binary = lethe_binary_path or self._find_lethe_binary()
        
    def _find_lethe_binary(self) -> str:
        """Find Lethe binary in the project."""
        possible_paths = [
            "./packages/cli/dist/cli.js",
            "./ctx-run/packages/cli/dist/index.js", 
            "./lethe",
            "lethe"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return str(Path(path).resolve())
        
        # Try to find in PATH
        try:
            result = subprocess.run(["which", "lethe"], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except:
            pass
            
        return "node packages/cli/dist/index.js"  # Fallback
        
    def is_available(self) -> bool:
        """Check if Lethe is available."""
        try:
            if self.lethe_binary.endswith(".js"):
                cmd = ["node", self.lethe_binary, "--help"]
            else:
                cmd = [self.lethe_binary, "--help"]
                
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            return result.returncode == 0
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def search(self, query: str, corpus_path: str, max_results: int = 100) -> List[SearchResult]:
        """Search using Lethe system."""
        try:
            # Create temporary directory for Lethe indexing
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Initialize Lethe workspace
                workspace_path = temp_path / "lethe_workspace"
                workspace_path.mkdir()
                
                # Index the corpus
                self._index_corpus(corpus_path, workspace_path)
                
                # Execute search
                search_results = self._execute_search(query, workspace_path, max_results)
                
                return search_results
                
        except Exception as e:
            logger.error(f"Lethe search failed: {e}")
            return []
    
    def _index_corpus(self, corpus_path: str, workspace_path: Path) -> None:
        """Index corpus data with Lethe."""
        try:
            # Create Lethe config
            config = {
                "indexing": {
                    "sources": [str(corpus_path)],
                    "chunk_size": 1000,
                    "overlap": 100
                },
                "retrieval": {
                    "method": "hybrid", 
                    "k": 100
                },
                "embeddings": {
                    "model": "all-MiniLM-L6-v2",
                    "cache": True
                }
            }
            
            config_file = workspace_path / "config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Run indexing
            if self.lethe_binary.endswith(".js"):
                cmd = ["node", self.lethe_binary, "index", "--config", str(config_file)]
            else:
                cmd = [self.lethe_binary, "index", "--config", str(config_file)]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                                  cwd=workspace_path)
            
            if result.returncode != 0:
                logger.warning(f"Lethe indexing warning: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logger.warning("Lethe indexing timed out")
        except Exception as e:
            logger.error(f"Lethe indexing failed: {e}")
    
    def _execute_search(self, query: str, workspace_path: Path, max_results: int) -> List[SearchResult]:
        """Execute search query with Lethe."""
        try:
            if self.lethe_binary.endswith(".js"):
                cmd = ["node", self.lethe_binary, "search", query, "--limit", str(max_results), "--json"]
            else:
                cmd = [self.lethe_binary, "search", query, "--limit", str(max_results), "--json"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                                  cwd=workspace_path)
            
            if result.returncode != 0:
                logger.warning(f"Lethe search warning: {result.stderr}")
                return []
            
            # Parse Lethe JSON output
            search_results = []
            
            try:
                output_data = json.loads(result.stdout)
                results_list = output_data.get("results", []) if isinstance(output_data, dict) else output_data
                
                for rank, item in enumerate(results_list[:max_results], 1):
                    content = item.get("content", item.get("text", str(item)))
                    score = float(item.get("score", item.get("similarity", 1.0 / rank)))
                    
                    search_results.append(SearchResult(
                        content=content,
                        score=score,
                        rank=rank,
                        is_relevant=False,  # Will be set by evaluator
                        tool_specific_data=item
                    ))
                    
            except json.JSONDecodeError:
                # Fallback: parse line-by-line output
                for rank, line in enumerate(result.stdout.strip().split('\n'), 1):
                    if line.strip():
                        search_results.append(SearchResult(
                            content=line.strip(),
                            score=1.0 / rank,
                            rank=rank,
                            is_relevant=False,
                            tool_specific_data={"raw_line": line}
                        ))
            
            return search_results
            
        except subprocess.TimeoutExpired:
            logger.warning("Lethe search timed out")
            return []
        except Exception as e:
            logger.error(f"Lethe search execution failed: {e}")
            return []

# Integration with main benchmarker
if __name__ == "__main__":
    # Test Lethe adapter standalone
    adapter = LetheAdapter()
    
    print(f"Lethe binary: {adapter.lethe_binary}")
    print(f"Lethe available: {adapter.is_available()}")
    
    if adapter.is_available():
        # Create test corpus
        test_corpus = "/tmp/test_corpus.txt"
        with open(test_corpus, 'w') as f:
            f.write("Key: test-key-123\\nValue: test-value-456\\n")
            f.write("Another key: abc-def-ghi\\nAnother value: xyz-uvw\\n")
        
        # Test search
        results = adapter.search("test-key-123", test_corpus, max_results=10)
        print(f"Search results: {len(results)}")
        for result in results:
            print(f"  {result.rank}: {result.content[:100]}... (score: {result.score:.3f})")
    else:
        print("Lethe is not available for testing")