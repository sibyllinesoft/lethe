#!/usr/bin/env python3
"""
Iteration 2: Query Understanding Components

Implements LLM-based query processing including:
- Query rewriting and expansion 
- Query decomposition into sub-queries
- Enhanced HyDE (Hypothetical Document Embedding)
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

@dataclass
class QueryUnderstandingConfig:
    """Configuration for query understanding components."""
    
    # Feature toggles
    query_rewrite: bool = True
    query_decompose: bool = True  
    hyde_enabled: bool = True
    
    # LLM settings
    llm_model: str = "gpt-4o-mini"
    llm_timeout_ms: int = 5000
    max_tokens: int = 512
    temperature: float = 0.1
    
    # Query processing
    max_subqueries: int = 3
    rewrite_strategy: str = "both"  # expand, clarify, both
    hyde_num_docs: int = 2
    
    # Quality controls
    min_query_length: int = 5
    max_query_length: int = 500
    similarity_threshold: float = 0.8


@dataclass 
class ProcessedQuery:
    """Result of query understanding pipeline."""
    
    original_query: str
    rewritten_query: Optional[str] = None
    subqueries: List[str] = None
    hyde_documents: List[str] = None
    processing_time_ms: float = 0
    llm_calls_made: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query, 
            "subqueries": self.subqueries or [],
            "hyde_documents": self.hyde_documents or [],
            "processing_time_ms": self.processing_time_ms,
            "llm_calls_made": self.llm_calls_made
        }


class LLMClient:
    """Unified LLM client supporting OpenAI and Anthropic."""
    
    def __init__(self, model: str, timeout_ms: int = 5000):
        self.model = model
        self.timeout_seconds = timeout_ms / 1000
        
        if model.startswith("gpt"):
            self.client = AsyncOpenAI()
            self.provider = "openai"
        elif model.startswith("claude"):
            self.client = AsyncAnthropic()
            self.provider = "anthropic"
        else:
            raise ValueError(f"Unsupported model: {model}")
            
    async def complete(self, prompt: str, max_tokens: int = 512, 
                      temperature: float = 0.1) -> str:
        """Get completion from LLM with timeout."""
        
        try:
            if self.provider == "openai":
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature
                    ),
                    timeout=self.timeout_seconds
                )
                return response.choices[0].message.content.strip()
                
            elif self.provider == "anthropic":
                response = await asyncio.wait_for(
                    self.client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt}]
                    ),
                    timeout=self.timeout_seconds
                )
                return response.content[0].text.strip()
                
        except asyncio.TimeoutError:
            logger.warning(f"LLM timeout after {self.timeout_seconds}s")
            raise
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise


class QueryRewriter:
    """Handles query rewriting and expansion."""
    
    def __init__(self, llm_client: LLMClient, strategy: str = "both"):
        self.llm = llm_client
        self.strategy = strategy
        
    async def rewrite_query(self, query: str, domain: str = "mixed") -> str:
        """Rewrite query for improved retrieval."""
        
        if self.strategy == "expand":
            prompt = self._get_expand_prompt(query, domain)
        elif self.strategy == "clarify":
            prompt = self._get_clarify_prompt(query, domain)
        else:  # both
            prompt = self._get_combined_prompt(query, domain)
            
        try:
            rewritten = await self.llm.complete(prompt)
            return self._validate_rewrite(query, rewritten)
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return query  # Fallback to original
            
    def _get_expand_prompt(self, query: str, domain: str) -> str:
        return f"""
Expand this search query to improve information retrieval by adding relevant keywords and synonyms.
Keep the core intent but make it more comprehensive.

Domain: {domain}
Original query: {query}

Expanded query:"""

    def _get_clarify_prompt(self, query: str, domain: str) -> str:
        return f"""
Clarify this search query by making ambiguous terms more specific and precise.
Maintain the original meaning but reduce ambiguity.

Domain: {domain}  
Original query: {query}

Clarified query:"""
        
    def _get_combined_prompt(self, query: str, domain: str) -> str:
        return f"""
Improve this search query by both expanding with relevant terms and clarifying ambiguous language.
Make it more comprehensive while maintaining precision.

Domain: {domain}
Original query: {query}

Improved query:"""
        
    def _validate_rewrite(self, original: str, rewritten: str) -> str:
        """Validate rewritten query meets quality standards."""
        
        # Basic validation
        if not rewritten or len(rewritten) < 5:
            return original
            
        # Length check
        if len(rewritten) > 500:
            return original
            
        # Similarity check (simple word overlap)
        orig_words = set(original.lower().split())
        rewrite_words = set(rewritten.lower().split())
        overlap = len(orig_words & rewrite_words) / len(orig_words)
        
        if overlap < 0.3:  # Too different
            return original
            
        return rewritten


class QueryDecomposer:
    """Decomposes complex queries into focused sub-queries."""
    
    def __init__(self, llm_client: LLMClient, max_subqueries: int = 3):
        self.llm = llm_client
        self.max_subqueries = max_subqueries
        
    async def decompose_query(self, query: str, domain: str = "mixed") -> List[str]:
        """Break complex query into focused sub-queries."""
        
        # Skip decomposition for simple queries
        if len(query.split()) < 5:
            return [query]
            
        prompt = f"""
Break this complex search query into {self.max_subqueries} focused sub-queries.
Each sub-query should target a specific aspect of the original question.

Domain: {domain}
Complex query: {query}

Sub-queries (one per line):
1."""

        try:
            response = await self.llm.complete(prompt, max_tokens=256)
            subqueries = self._parse_subqueries(response)
            
            # Validation
            if not subqueries or len(subqueries) > self.max_subqueries:
                return [query]  # Fallback
                
            return subqueries
            
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [query]
            
    def _parse_subqueries(self, response: str) -> List[str]:
        """Parse LLM response into list of sub-queries."""
        
        lines = response.strip().split('\n')
        subqueries = []
        
        for line in lines:
            # Remove numbering and clean up
            clean_line = line.strip()
            if clean_line:
                # Remove leading numbers/bullets
                import re
                clean_line = re.sub(r'^\d+\.?\s*', '', clean_line)
                clean_line = re.sub(r'^[-•]\s*', '', clean_line)
                
                if len(clean_line) > 5:
                    subqueries.append(clean_line)
                    
        return subqueries[:self.max_subqueries]


class HyDEGenerator:
    """Generates hypothetical documents for enhanced retrieval."""
    
    def __init__(self, llm_client: LLMClient, num_docs: int = 2):
        self.llm = llm_client
        self.num_docs = num_docs
        
    async def generate_documents(self, query: str, domain: str = "mixed") -> List[str]:
        """Generate hypothetical documents that would answer the query."""
        
        prompt = f"""
Generate {self.num_docs} brief hypothetical documents that would perfectly answer this search query.
Each document should be 2-3 sentences and contain the key information being sought.

Domain: {domain}
Query: {query}

Document 1:"""

        try:
            response = await self.llm.complete(prompt, max_tokens=400)
            documents = self._parse_documents(response)
            
            # Ensure we have valid documents
            if not documents:
                return []
                
            return documents[:self.num_docs]
            
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return []
            
    def _parse_documents(self, response: str) -> List[str]:
        """Parse LLM response into list of documents."""
        
        # Split by document markers
        import re
        doc_pattern = r'Document \d+:'
        parts = re.split(doc_pattern, response)
        
        documents = []
        for part in parts[1:]:  # Skip first empty part
            doc = part.strip()
            if len(doc) > 20:  # Minimum length
                documents.append(doc)
                
        return documents


class QueryUnderstandingPipeline:
    """Main pipeline orchestrating all query understanding components."""
    
    def __init__(self, config: QueryUnderstandingConfig):
        self.config = config
        self.llm_client = LLMClient(config.llm_model, config.llm_timeout_ms)
        
        # Initialize components
        self.rewriter = QueryRewriter(self.llm_client, config.rewrite_strategy)
        self.decomposer = QueryDecomposer(self.llm_client, config.max_subqueries)
        self.hyde_generator = HyDEGenerator(self.llm_client, config.hyde_num_docs)
        
    async def process_query(self, query: str, domain: str = "mixed") -> ProcessedQuery:
        """Run complete query understanding pipeline."""
        
        start_time = time.time()
        llm_calls = 0
        
        result = ProcessedQuery(original_query=query)
        
        try:
            # Step 1: Query rewriting
            if self.config.query_rewrite:
                result.rewritten_query = await self.rewriter.rewrite_query(query, domain)
                llm_calls += 1
                
            # Use rewritten query for downstream processing
            working_query = result.rewritten_query or query
            
            # Step 2: Query decomposition  
            if self.config.query_decompose:
                result.subqueries = await self.decomposer.decompose_query(working_query, domain)
                llm_calls += 1
                
            # Step 3: HyDE document generation
            if self.config.hyde_enabled:
                result.hyde_documents = await self.hyde_generator.generate_documents(working_query, domain)
                llm_calls += 1
                
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            
        # Record metrics
        result.processing_time_ms = (time.time() - start_time) * 1000
        result.llm_calls_made = llm_calls
        
        return result
    
    async def batch_process(self, queries: List[Tuple[str, str]]) -> List[ProcessedQuery]:
        """Process multiple queries efficiently."""
        
        tasks = [
            self.process_query(query, domain) 
            for query, domain in queries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Query {i} failed: {result}")
                processed_results.append(ProcessedQuery(original_query=queries[i][0]))
            else:
                processed_results.append(result)
                
        return processed_results


# Evaluation utilities

async def evaluate_query_understanding(config: QueryUnderstandingConfig, 
                                     test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate query understanding pipeline performance."""
    
    pipeline = QueryUnderstandingPipeline(config)
    
    results = []
    processing_times = []
    llm_call_counts = []
    
    for query_data in test_queries:
        query = query_data["query"]
        domain = query_data.get("domain", "mixed")
        
        processed = await pipeline.process_query(query, domain)
        
        results.append({
            "query_id": query_data.get("query_id", "unknown"),
            **processed.to_dict()
        })
        
        processing_times.append(processed.processing_time_ms)
        llm_call_counts.append(processed.llm_calls_made)
        
    # Compute metrics
    metrics = {
        "total_queries": len(test_queries),
        "avg_processing_time_ms": np.mean(processing_times),
        "p95_processing_time_ms": np.percentile(processing_times, 95),
        "avg_llm_calls": np.mean(llm_call_counts),
        "total_llm_calls": sum(llm_call_counts),
        "success_rate": len([r for r in results if r["rewritten_query"] or r["subqueries"]]) / len(results)
    }
    
    return {
        "config": config.__dict__,
        "metrics": metrics,
        "detailed_results": results
    }


# Integration with main experiment framework

def create_iter2_config(grid_params: Dict[str, Any]) -> QueryUnderstandingConfig:
    """Create config from grid search parameters."""
    
    return QueryUnderstandingConfig(
        query_rewrite=grid_params.get("query_rewrite", True),
        query_decompose=grid_params.get("query_decompose", True),
        hyde_enabled=grid_params.get("hyde_enabled", True),
        llm_model=grid_params.get("llm_model", "gpt-4o-mini"),
        max_subqueries=grid_params.get("max_subqueries", 3),
        rewrite_strategy=grid_params.get("rewrite_strategy", "both"),
        llm_timeout_ms=grid_params.get("llm_timeout_ms", 5000)
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test query understanding pipeline")
    parser.add_argument("--query", required=True, help="Query to process")
    parser.add_argument("--domain", default="mixed", help="Query domain")
    parser.add_argument("--config", help="Config file path")
    
    args = parser.parse_args()
    
    # Default config for testing
    config = QueryUnderstandingConfig()
    
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config_data = json.load(f)
            config = QueryUnderstandingConfig(**config_data)
    
    async def main():
        pipeline = QueryUnderstandingPipeline(config)
        result = await pipeline.process_query(args.query, args.domain)
        
        print(json.dumps(result.to_dict(), indent=2))
        
    asyncio.run(main())