#!/usr/bin/env python3
"""
Code Domain Builder for LetheBench
==================================

Specialized builder for generating high-quality code-heavy queries that represent
realistic programming questions and scenarios. This builder creates queries that
test conversational retrieval systems on technical programming content.

Features:
- Realistic programming query patterns
- Multi-language coverage (Python, JavaScript, Go, Rust, Java, C++)
- Diverse complexity levels with appropriate code examples
- Ground truth code documentation and examples
- Cross-reference capabilities between concepts
"""

import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from schema import (
    QueryRecord, GroundTruthDocument, QueryMetadata, 
    ComplexityLevel, DomainType, ValidationResult
)


@dataclass
class CodePattern:
    """Code pattern for generating realistic queries"""
    name: str
    template: str
    complexity: ComplexityLevel
    languages: List[str]
    concepts: List[str]
    example_code: str
    difficulty_keywords: List[str]


class CodeDomainBuilder:
    """
    Builder for code-heavy domain queries with realistic programming scenarios
    """
    
    def __init__(self, seed: int = 42, logger: Optional[logging.Logger] = None):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize programming language patterns
        self.programming_languages = [
            "Python", "JavaScript", "TypeScript", "Go", "Rust", "Java", 
            "C++", "C", "Swift", "Kotlin", "Ruby", "PHP"
        ]
        
        # Initialize code patterns by complexity
        self.code_patterns = self._initialize_code_patterns()
        
        # Code concepts and their relationships
        self.code_concepts = self._initialize_code_concepts()
        
        # Framework and library mappings
        self.frameworks = self._initialize_frameworks()
        
    def _initialize_code_patterns(self) -> Dict[ComplexityLevel, List[CodePattern]]:
        """Initialize comprehensive code patterns for query generation"""
        
        patterns = {
            ComplexityLevel.SIMPLE: [
                CodePattern(
                    name="basic_syntax",
                    template="How do I {action} in {language}?",
                    complexity=ComplexityLevel.SIMPLE,
                    languages=["Python", "JavaScript", "Java", "C++"],
                    concepts=["syntax", "basic_operations", "variables"],
                    example_code="# Example: variables and basic operations\nx = 10\ny = 20\nresult = x + y\nprint(result)",
                    difficulty_keywords=["basic", "simple", "how to", "syntax"]
                ),
                CodePattern(
                    name="function_definition",
                    template="Show me how to define a {function_type} function in {language}",
                    complexity=ComplexityLevel.SIMPLE,
                    languages=["Python", "JavaScript", "Go", "Rust"],
                    concepts=["functions", "parameters", "return_values"],
                    example_code="def calculate_sum(a, b):\n    return a + b\n\nresult = calculate_sum(5, 3)",
                    difficulty_keywords=["function", "define", "parameters"]
                ),
                CodePattern(
                    name="basic_data_structures",
                    template="What's the best way to work with {data_structure} in {language}?",
                    complexity=ComplexityLevel.SIMPLE,
                    languages=["Python", "Java", "JavaScript", "C++"],
                    concepts=["arrays", "lists", "dictionaries", "maps"],
                    example_code="# Working with lists\ndata = [1, 2, 3, 4, 5]\ndata.append(6)\nprint(data)",
                    difficulty_keywords=["array", "list", "dictionary", "basic"]
                ),
                CodePattern(
                    name="loop_iteration",
                    template="How do I iterate over {iterable} in {language}?",
                    complexity=ComplexityLevel.SIMPLE,
                    languages=["Python", "JavaScript", "Go", "Java"],
                    concepts=["loops", "iteration", "control_flow"],
                    example_code="for item in items:\n    print(item)",
                    difficulty_keywords=["loop", "iterate", "for", "while"]
                ),
                CodePattern(
                    name="conditional_logic",
                    template="What's the syntax for {conditional_type} statements in {language}?",
                    complexity=ComplexityLevel.SIMPLE,
                    languages=["Python", "JavaScript", "Java", "C++", "Go"],
                    concepts=["conditionals", "if_statements", "boolean_logic"],
                    example_code="if x > 0:\n    print('positive')\nelif x < 0:\n    print('negative')\nelse:\n    print('zero')",
                    difficulty_keywords=["if", "else", "condition", "boolean"]
                )
            ],
            
            ComplexityLevel.MEDIUM: [
                CodePattern(
                    name="error_handling",
                    template="How should I handle {error_type} errors when {scenario} in {language}?",
                    complexity=ComplexityLevel.MEDIUM,
                    languages=["Python", "JavaScript", "Java", "Go", "Rust"],
                    concepts=["exception_handling", "error_propagation", "try_catch"],
                    example_code="try:\n    result = risky_operation()\n    return result\nexcept ValueError as e:\n    logger.error(f'Invalid value: {e}')\n    return None",
                    difficulty_keywords=["exception", "error", "try", "catch", "handle"]
                ),
                CodePattern(
                    name="async_programming",
                    template="What's the best approach for {async_operation} using {async_pattern} in {language}?",
                    complexity=ComplexityLevel.MEDIUM,
                    languages=["Python", "JavaScript", "TypeScript", "C#", "Rust"],
                    concepts=["async_await", "promises", "futures", "concurrency"],
                    example_code="async def fetch_data(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            return await response.json()",
                    difficulty_keywords=["async", "await", "promise", "concurrent", "parallel"]
                ),
                CodePattern(
                    name="data_processing",
                    template="How can I efficiently {operation} {data_type} data in {language}?",
                    complexity=ComplexityLevel.MEDIUM,
                    languages=["Python", "JavaScript", "Java", "Scala"],
                    concepts=["data_transformation", "filtering", "mapping", "aggregation"],
                    example_code="def process_data(items):\n    return [\n        transform_item(item) \n        for item in items \n        if meets_criteria(item)\n    ]",
                    difficulty_keywords=["process", "transform", "filter", "map", "aggregate"]
                ),
                CodePattern(
                    name="class_design",
                    template="How should I design a {class_type} class in {language} that {requirement}?",
                    complexity=ComplexityLevel.MEDIUM,
                    languages=["Python", "Java", "C++", "C#", "TypeScript"],
                    concepts=["object_oriented", "inheritance", "polymorphism", "encapsulation"],
                    example_code="class DataProcessor:\n    def __init__(self, config):\n        self._config = config\n        self._cache = {}\n    \n    def process(self, data):\n        if data.id in self._cache:\n            return self._cache[data.id]\n        result = self._transform(data)\n        self._cache[data.id] = result\n        return result",
                    difficulty_keywords=["class", "object", "inheritance", "method", "property"]
                ),
                CodePattern(
                    name="algorithm_implementation",
                    template="What's an efficient way to implement {algorithm} for {use_case} in {language}?",
                    complexity=ComplexityLevel.MEDIUM,
                    languages=["Python", "Java", "C++", "Go"],
                    concepts=["algorithms", "data_structures", "time_complexity", "optimization"],
                    example_code="def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
                    difficulty_keywords=["algorithm", "search", "sort", "optimization", "complexity"]
                )
            ],
            
            ComplexityLevel.COMPLEX: [
                CodePattern(
                    name="system_design",
                    template="How would you architect a {system_type} system in {language} that handles {requirements}?",
                    complexity=ComplexityLevel.COMPLEX,
                    languages=["Python", "Java", "Go", "Rust", "C++"],
                    concepts=["system_architecture", "scalability", "distributed_systems", "microservices"],
                    example_code="class MessageQueue:\n    def __init__(self, max_size=1000):\n        self._queue = asyncio.Queue(maxsize=max_size)\n        self._subscribers = set()\n        self._metrics = MessageMetrics()\n    \n    async def publish(self, message):\n        await self._queue.put(message)\n        await self._notify_subscribers(message)\n        self._metrics.increment_published()",
                    difficulty_keywords=["architecture", "scalability", "distributed", "microservices", "system"]
                ),
                CodePattern(
                    name="performance_optimization",
                    template="How can I optimize {performance_aspect} in my {language} application when dealing with {constraint}?",
                    complexity=ComplexityLevel.COMPLEX,
                    languages=["Python", "Java", "C++", "Rust", "Go"],
                    concepts=["performance_tuning", "profiling", "memory_management", "caching"],
                    example_code="from functools import lru_cache\nimport cProfile\n\nclass OptimizedProcessor:\n    def __init__(self):\n        self._cache = LRUCache(maxsize=1000)\n        self._pool = ThreadPoolExecutor(max_workers=4)\n    \n    @lru_cache(maxsize=256)\n    def expensive_computation(self, data):\n        return complex_algorithm(data)",
                    difficulty_keywords=["performance", "optimization", "profiling", "memory", "cache"]
                ),
                CodePattern(
                    name="concurrent_programming",
                    template="What's the best strategy for {concurrency_goal} in {language} while avoiding {pitfall}?",
                    complexity=ComplexityLevel.COMPLEX,
                    languages=["Python", "Java", "Go", "Rust", "C++"],
                    concepts=["concurrency", "threading", "synchronization", "deadlocks", "race_conditions"],
                    example_code="import asyncio\nfrom asyncio import Semaphore\n\nclass ConcurrentProcessor:\n    def __init__(self, max_concurrent=10):\n        self._semaphore = Semaphore(max_concurrent)\n        self._results = asyncio.Queue()\n    \n    async def process_batch(self, items):\n        tasks = [self._process_item(item) for item in items]\n        await asyncio.gather(*tasks)",
                    difficulty_keywords=["concurrent", "threading", "synchronization", "deadlock", "race"]
                ),
                CodePattern(
                    name="database_optimization",
                    template="How should I optimize database operations in {language} for {scenario} with {constraint}?",
                    complexity=ComplexityLevel.COMPLEX,
                    languages=["Python", "Java", "Go", "C#"],
                    concepts=["database_optimization", "query_performance", "connection_pooling", "transactions"],
                    example_code="class DatabaseManager:\n    def __init__(self, connection_pool):\n        self._pool = connection_pool\n        self._query_cache = QueryCache()\n    \n    async def execute_optimized_query(self, query, params):\n        cache_key = self._generate_cache_key(query, params)\n        if cached := self._query_cache.get(cache_key):\n            return cached\n        \n        async with self._pool.acquire() as conn:\n            result = await conn.execute(query, params)\n            self._query_cache.set(cache_key, result)\n            return result",
                    difficulty_keywords=["database", "query", "optimization", "connection", "transaction"]
                ),
                CodePattern(
                    name="testing_strategy",
                    template="What testing strategy would you recommend for {component_type} in {language} that {complexity_factor}?",
                    complexity=ComplexityLevel.COMPLEX,
                    languages=["Python", "JavaScript", "Java", "Go"],
                    concepts=["testing_strategies", "mocking", "integration_testing", "test_driven_development"],
                    example_code="import pytest\nfrom unittest.mock import Mock, patch\n\nclass TestComplexComponent:\n    @pytest.fixture\n    def component(self):\n        return ComplexComponent(Mock(), Mock())\n    \n    @patch('external_service.api_call')\n    def test_complex_workflow(self, mock_api, component):\n        mock_api.return_value = {'status': 'success'}\n        result = component.process_workflow(test_data)\n        assert result.is_successful()",
                    difficulty_keywords=["testing", "mock", "integration", "unit", "strategy"]
                )
            ]
        }
        
        return patterns
    
    def _initialize_code_concepts(self) -> Dict[str, Dict[str, Any]]:
        """Initialize code concepts and their relationships"""
        return {
            "data_structures": {
                "keywords": ["array", "list", "stack", "queue", "tree", "graph", "hash_table"],
                "related": ["algorithms", "memory_management", "performance"],
                "examples": {
                    "Python": "list, dict, set, tuple",
                    "JavaScript": "Array, Object, Set, Map",
                    "Java": "ArrayList, HashMap, LinkedList",
                    "C++": "vector, map, unordered_map, stack"
                }
            },
            "algorithms": {
                "keywords": ["sort", "search", "traversal", "optimization", "dynamic_programming"],
                "related": ["data_structures", "complexity_analysis", "performance"],
                "examples": {
                    "sorting": "quicksort, mergesort, heapsort",
                    "searching": "binary_search, depth_first_search, breadth_first_search",
                    "optimization": "dynamic_programming, greedy_algorithms"
                }
            },
            "concurrency": {
                "keywords": ["threading", "async", "parallel", "synchronization", "deadlock"],
                "related": ["performance", "system_programming", "error_handling"],
                "examples": {
                    "Python": "asyncio, threading, multiprocessing",
                    "JavaScript": "async/await, Promise, Web Workers",
                    "Java": "ExecutorService, CompletableFuture, synchronized"
                }
            },
            "web_development": {
                "keywords": ["http", "api", "rest", "graphql", "authentication", "middleware"],
                "related": ["databases", "security", "performance"],
                "examples": {
                    "Python": "Flask, Django, FastAPI",
                    "JavaScript": "Express, React, Next.js",
                    "Java": "Spring Boot, Jersey"
                }
            },
            "databases": {
                "keywords": ["sql", "nosql", "orm", "migration", "indexing", "transaction"],
                "related": ["performance", "concurrency", "data_modeling"],
                "examples": {
                    "SQL": "PostgreSQL, MySQL, SQLite",
                    "NoSQL": "MongoDB, Redis, Cassandra",
                    "ORM": "SQLAlchemy, Mongoose, Hibernate"
                }
            }
        }
    
    def _initialize_frameworks(self) -> Dict[str, List[str]]:
        """Initialize framework mappings by language"""
        return {
            "Python": ["Django", "Flask", "FastAPI", "Pytest", "SQLAlchemy", "Pandas", "NumPy"],
            "JavaScript": ["React", "Vue", "Express", "Jest", "Lodash", "Axios"],
            "TypeScript": ["Angular", "NestJS", "TypeORM", "React", "Express"],
            "Java": ["Spring Boot", "Hibernate", "JUnit", "Maven", "Gradle"],
            "Go": ["Gin", "Echo", "Gorm", "Testify", "Cobra"],
            "Rust": ["Rocket", "Actix-web", "Serde", "Tokio", "Diesel"],
            "C++": ["Boost", "Qt", "POCO", "Google Test"],
            "C#": [".NET Core", "Entity Framework", "XUnit", "ASP.NET"],
            "PHP": ["Laravel", "Symfony", "PHPUnit", "Composer"],
            "Ruby": ["Rails", "Sinatra", "RSpec", "Bundler"]
        }
    
    def generate_queries(self, count: int, complexity_distribution: Optional[Dict[str, float]] = None) -> List[QueryRecord]:
        """
        Generate realistic code domain queries
        
        Args:
            count: Number of queries to generate
            complexity_distribution: Distribution of complexity levels (default: balanced)
        
        Returns:
            List of QueryRecord objects with realistic programming content
        """
        if complexity_distribution is None:
            complexity_distribution = {
                ComplexityLevel.SIMPLE: 0.4,
                ComplexityLevel.MEDIUM: 0.4, 
                ComplexityLevel.COMPLEX: 0.2
            }
        
        queries = []
        
        for i in range(count):
            # Deterministic complexity assignment
            complexity_rng = np.random.RandomState(self.seed + i)
            complexity = complexity_rng.choice(
                list(complexity_distribution.keys()),
                p=list(complexity_distribution.values())
            )
            
            # Generate query based on complexity
            query = self._generate_single_query(i, complexity)
            queries.append(query)
            
        self.logger.info(f"Generated {len(queries)} code domain queries")
        return queries
    
    def _generate_single_query(self, query_index: int, complexity: ComplexityLevel) -> QueryRecord:
        """Generate a single realistic code query"""
        
        # Create deterministic RNG for this query
        query_rng = np.random.RandomState(self.seed + query_index + 1000)
        
        # Select pattern for this complexity level
        patterns = self.code_patterns[complexity]
        pattern = query_rng.choice(patterns)
        
        # Generate query text from pattern
        query_text = self._populate_query_template(pattern, query_rng)
        
        # Generate ground truth documents
        ground_truth_docs = self._generate_ground_truth_docs(
            pattern, query_text, query_rng, query_index
        )
        
        # Calculate quality score based on pattern and content
        quality_score = self._calculate_code_quality(query_text, pattern, ground_truth_docs)
        
        # Create metadata
        metadata = QueryMetadata(
            creation_seed=self.seed + query_index + 1000,
            query_index=query_index,
            template_used=pattern.name,
            length_chars=len(query_text),
            complexity_score={"simple": 1, "medium": 2, "complex": 3}[str(complexity)],
            quality_score=quality_score,
            n_ground_truth_docs=len(ground_truth_docs),
            token_count=len(query_text.split()),
            has_code_blocks=bool(query_rng.random() > 0.3),  # 70% chance of code blocks
        )
        
        # Generate query record
        query_id = f"code_heavy_query_{query_index:06d}"
        
        # Generate content hash to match schema validation
        hash_content = {
            "query_id": query_id,
            "query_text": query_text,
            "domain": DomainType.CODE_HEAVY,
            "complexity": complexity,
            "ground_truth_docs": sorted([doc.doc_id for doc in ground_truth_docs])
        }
        content_json = json.dumps(hash_content, sort_keys=True, separators=(',', ':'), default=str)
        content_hash = hashlib.sha256(content_json.encode()).hexdigest()
        
        return QueryRecord(
            query_id=query_id,
            domain=DomainType.CODE_HEAVY,
            complexity=complexity,
            session_id=f"code_heavy_session_{query_index // 10:04d}",
            turn_index=query_rng.randint(1, 8),
            query_text=query_text,
            ground_truth_docs=ground_truth_docs,
            metadata=metadata,
            content_hash=content_hash,
            creation_timestamp=datetime.now(timezone.utc)
        )
    
    def _populate_query_template(self, pattern: CodePattern, rng: np.random.RandomState) -> str:
        """Populate query template with realistic values"""
        
        template = pattern.template
        
        # Language selection
        if "{language}" in template:
            language = rng.choice(pattern.languages)
            template = template.replace("{language}", language)
        
        # Pattern-specific replacements
        replacements = {
            "action": ["create a variable", "iterate through arrays", "handle exceptions", 
                      "define functions", "work with classes", "manage memory"],
            "function_type": ["recursive", "async", "pure", "generator", "decorator", "callback"],
            "data_structure": ["arrays", "hash maps", "linked lists", "trees", "graphs", "stacks"],
            "iterable": ["arrays", "objects", "maps", "lists", "collections", "ranges"],
            "conditional_type": ["if-else", "switch-case", "ternary", "pattern matching"],
            "error_type": ["network", "validation", "parsing", "database", "authentication"],
            "scenario": ["making API calls", "parsing JSON", "reading files", "database operations"],
            "async_operation": ["API requests", "file I/O", "database queries", "image processing"],
            "async_pattern": ["promises", "async/await", "callbacks", "futures"],
            "operation": ["filter", "transform", "aggregate", "sort", "group", "validate"],
            "data_type": ["JSON", "CSV", "XML", "binary", "streaming", "time-series"],
            "class_type": ["singleton", "factory", "builder", "observer", "adapter"],
            "requirement": ["handles validation", "supports caching", "is thread-safe", "manages state"],
            "algorithm": ["binary search", "quicksort", "BFS", "dynamic programming", "hash table"],
            "use_case": ["sorted arrays", "large datasets", "real-time processing", "memory constraints"],
            "system_type": ["messaging", "caching", "load balancing", "monitoring", "authentication"],
            "requirements": ["high throughput", "fault tolerance", "real-time updates", "data consistency"],
            "performance_aspect": ["memory usage", "response time", "throughput", "CPU utilization"],
            "constraint": ["limited memory", "high concurrency", "large datasets", "strict latency"],
            "concurrency_goal": ["parallel processing", "resource sharing", "task coordination"],
            "pitfall": ["deadlocks", "race conditions", "memory leaks", "starvation"],
            "scenario": ["high-frequency queries", "large transactions", "concurrent updates"],
            "component_type": ["API services", "data processors", "message handlers", "authentication"],
            "complexity_factor": ["involves external dependencies", "handles complex state", "requires mocking"]
        }
        
        # Apply replacements
        for placeholder, options in replacements.items():
            if f"{{{placeholder}}}" in template:
                value = rng.choice(options)
                template = template.replace(f"{{{placeholder}}}", value)
        
        # Add complexity-based elaboration
        if pattern.complexity == ComplexityLevel.COMPLEX:
            elaborations = [
                " Please include performance considerations and potential trade-offs.",
                " I need to understand both the implementation details and architectural implications.", 
                " Can you provide examples with error handling and edge cases?",
                " Include discussion of scalability and maintainability concerns."
            ]
            template += rng.choice(elaborations)
        elif pattern.complexity == ComplexityLevel.MEDIUM:
            elaborations = [
                " Please provide working examples.",
                " I'd like to understand best practices for this.",
                " Can you explain the key concepts involved?",
                " What are the common pitfalls to avoid?"
            ]
            template += rng.choice(elaborations)
        
        return template
    
    def _generate_ground_truth_docs(
        self, 
        pattern: CodePattern, 
        query_text: str, 
        rng: np.random.RandomState,
        query_index: int
    ) -> List[GroundTruthDocument]:
        """Generate realistic ground truth documents for code queries"""
        
        # Determine number of documents based on complexity
        n_docs = {
            ComplexityLevel.SIMPLE: rng.randint(2, 4),
            ComplexityLevel.MEDIUM: rng.randint(3, 6),
            ComplexityLevel.COMPLEX: rng.randint(4, 8)
        }[pattern.complexity]
        
        docs = []
        
        for doc_idx in range(n_docs):
            doc_id = f"code_heavy_doc_{query_index:06d}_{doc_idx:02d}"
            
            # Generate document content based on type
            doc_type = rng.choice(["code_example", "documentation", "tutorial", "reference"])
            content = self._generate_code_document_content(pattern, doc_type, rng)
            
            # Calculate relevance score
            relevance_score = 0.6 + rng.random() * 0.4  # Between 0.6 and 1.0
            
            # Create content hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            doc = GroundTruthDocument(
                doc_id=doc_id,
                content=content,
                relevance_score=relevance_score,
                doc_type=doc_type,
                content_hash=content_hash,
                metadata={
                    "pattern_name": pattern.name,
                    "languages": pattern.languages,
                    "concepts": pattern.concepts
                }
            )
            docs.append(doc)
        
        return docs
    
    def _generate_code_document_content(
        self, 
        pattern: CodePattern, 
        doc_type: str, 
        rng: np.random.RandomState
    ) -> str:
        """Generate realistic code document content"""
        
        if doc_type == "code_example":
            return f"""# {pattern.name.replace('_', ' ').title()} Example

```{rng.choice(pattern.languages).lower()}
{pattern.example_code}
```

This example demonstrates {rng.choice(pattern.concepts)} in practice.
Key concepts: {', '.join(pattern.concepts[:3])}

## Usage Notes
- Consider {rng.choice(['performance', 'memory usage', 'error handling', 'maintainability'])} implications
- Test with {rng.choice(['edge cases', 'large datasets', 'concurrent access', 'invalid input'])}
- Follow {rng.choice(['SOLID principles', 'DRY principles', 'clean code practices'])}
"""
            
        elif doc_type == "documentation":
            return f"""# {pattern.name.replace('_', ' ').title()} Documentation

## Overview
This section covers {pattern.name.replace('_', ' ')} patterns and best practices.

## Supported Languages
{', '.join(pattern.languages)}

## Key Concepts
{chr(10).join(f'- {concept.replace("_", " ").title()}' for concept in pattern.concepts)}

## Implementation Guidelines
1. Always validate input parameters
2. Handle error conditions gracefully  
3. Consider performance implications
4. Write comprehensive tests
5. Document public APIs

## Common Pitfalls
- Not handling {rng.choice(['null values', 'edge cases', 'concurrent access'])}
- Ignoring {rng.choice(['memory leaks', 'performance bottlenecks', 'error propagation'])}
- Poor {rng.choice(['naming conventions', 'code organization', 'test coverage'])}

## Related Topics
{', '.join(pattern.concepts)}
"""
            
        elif doc_type == "tutorial":
            return f"""# Step-by-Step Tutorial: {pattern.name.replace('_', ' ').title()}

## Prerequisites
- Basic knowledge of {rng.choice(pattern.languages)}
- Understanding of {rng.choice(pattern.concepts).replace('_', ' ')}

## Step 1: Setup
First, let's set up the basic structure:

```{rng.choice(pattern.languages).lower()}
{pattern.example_code.split(chr(10))[0]}
```

## Step 2: Implementation
Now we implement the core functionality:

```{rng.choice(pattern.languages).lower()}
{chr(10).join(pattern.example_code.split(chr(10))[1:3])}
```

## Step 3: Testing
Add comprehensive tests:

```{rng.choice(pattern.languages).lower()}
# Test cases for {pattern.name}
test_basic_functionality()
test_edge_cases() 
test_error_conditions()
```

## Next Steps
- Explore {rng.choice(pattern.concepts).replace('_', ' ')} patterns
- Consider {rng.choice(['performance optimization', 'scalability improvements', 'security hardening'])}
- Review {rng.choice(['related algorithms', 'design patterns', 'best practices'])}
"""
            
        else:  # reference
            return f"""# {pattern.name.replace('_', ' ').title()} Reference

## Quick Reference
{pattern.name.replace('_', ' ').title()} is used for {rng.choice(pattern.concepts).replace('_', ' ')}.

## Syntax
```{rng.choice(pattern.languages).lower()}
{pattern.example_code.split(chr(10))[0]}
```

## Parameters
- Input: {rng.choice(['data structure', 'configuration object', 'callback function', 'primitive value'])}
- Output: {rng.choice(['processed data', 'boolean result', 'new instance', 'void'])}

## Complexity
- Time: O({rng.choice(['1', 'log n', 'n', 'n log n', 'n²'])})
- Space: O({rng.choice(['1', 'log n', 'n'])})

## Examples
See {pattern.name}_examples.{rng.choice(['py', 'js', 'java', 'cpp'])} for detailed examples.

## See Also
- {rng.choice(pattern.concepts).replace('_', ' ').title()}
- {rng.choice(['Best Practices', 'Performance Guidelines', 'Error Handling'])}
- {rng.choice(['Related Algorithms', 'Design Patterns', 'Testing Strategies'])}
"""
    
    def _calculate_code_quality(
        self, 
        query_text: str, 
        pattern: CodePattern, 
        ground_truth_docs: List[GroundTruthDocument]
    ) -> float:
        """Calculate quality score for code domain query"""
        
        score = 0.0
        
        # Length appropriateness (20-500 chars for code queries)
        length = len(query_text)
        if 30 <= length <= 300:
            score += 0.25
        elif 20 <= length <= 500:
            score += 0.15
        
        # Complexity appropriateness
        complexity_scores = {
            ComplexityLevel.SIMPLE: 0.2,
            ComplexityLevel.MEDIUM: 0.25,
            ComplexityLevel.COMPLEX: 0.2  # Complex queries are harder to get right
        }
        score += complexity_scores[pattern.complexity]
        
        # Programming specificity
        programming_keywords = [
            "function", "class", "algorithm", "implement", "code", "syntax",
            "debug", "optimize", "refactor", "test", "api", "library", "framework"
        ]
        if any(keyword in query_text.lower() for keyword in programming_keywords):
            score += 0.2
        
        # Ground truth appropriateness 
        n_docs = len(ground_truth_docs)
        if pattern.complexity == ComplexityLevel.SIMPLE and 2 <= n_docs <= 4:
            score += 0.15
        elif pattern.complexity == ComplexityLevel.MEDIUM and 3 <= n_docs <= 6:
            score += 0.15
        elif pattern.complexity == ComplexityLevel.COMPLEX and 4 <= n_docs <= 8:
            score += 0.15
        else:
            score += 0.05
        
        # Pattern quality bonus
        if pattern.name in ["async_programming", "performance_optimization", "system_design"]:
            score += 0.1  # Bonus for advanced patterns
        
        # Language specificity
        languages = set(pattern.languages)
        modern_languages = {"Python", "JavaScript", "TypeScript", "Go", "Rust"}
        if languages & modern_languages:
            score += 0.05
        
        return min(1.0, max(0.0, score))
    
    def validate_queries(self, queries: List[QueryRecord]) -> ValidationResult:
        """Validate generated code domain queries"""
        
        errors = []
        warnings = []
        query_errors = {}
        
        # Check domain consistency
        non_code_queries = [q for q in queries if q.domain != DomainType.CODE_HEAVY]
        if non_code_queries:
            errors.append(f"Found {len(non_code_queries)} non-code queries")
        
        # Check for programming-specific content
        programming_patterns = [
            r'\b(function|class|method|algorithm|code|implement|debug)\b',
            r'\b(python|javascript|java|rust|go|cpp)\b', 
            r'\b(api|framework|library|syntax)\b'
        ]
        
        for query in queries:
            query_programming_score = sum(
                1 for pattern in programming_patterns
                if __import__('re').search(pattern, query.query_text.lower())
            )
            
            if query_programming_score == 0:
                query_errors[query.query_id] = ["Query lacks programming-specific content"]
                
        # Check complexity distribution
        complexity_counts = {}
        for query in queries:
            complexity_counts[query.complexity] = complexity_counts.get(query.complexity, 0) + 1
        
        total_queries = len(queries)
        if total_queries > 0:
            simple_ratio = complexity_counts.get(ComplexityLevel.SIMPLE, 0) / total_queries
            if simple_ratio > 0.6:
                warnings.append(f"High ratio of simple queries: {simple_ratio:.2%}")
        
        # Check quality scores
        if queries:
            avg_quality = sum(q.metadata.quality_score for q in queries) / len(queries)
            if avg_quality < 0.75:
                warnings.append(f"Average quality score below threshold: {avg_quality:.3f}")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            query_errors=query_errors,
            statistics={
                "total_queries": len(queries),
                "avg_quality_score": avg_quality if queries else 0,
                "programming_content_ratio": (len(queries) - len(query_errors)) / len(queries) if queries else 0
            },
            recommendations=[
                "Ensure all queries contain programming-specific terminology",
                "Balance complexity distribution for realistic scenarios",
                "Include diverse programming languages and frameworks",
                "Add code examples to ground truth documents"
            ]
        )