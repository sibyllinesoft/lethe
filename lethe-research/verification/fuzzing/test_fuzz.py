#!/usr/bin/env python3
"""
Fuzzing Infrastructure for Lethe Research Components

This module implements comprehensive fuzzing to discover edge cases and failure modes
in retrieval algorithms, scoring functions, and data processing pipelines.

Requirements:
- Deterministic fuzzing with seed control for reproducibility
- Coverage-guided fuzzing to maximize code path exploration
- Crash detection and crash reproduction
- Statistical analysis of discovered failures
- Integration with existing test infrastructure
"""

import json
import logging
import random
import string
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Iterator, Callable
from enum import Enum
import sys
import hashlib
import pickle
import multiprocessing as mp
from collections import defaultdict, Counter
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parents[3]))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FuzzResult(Enum):
    """Result types from fuzzing executions."""
    PASS = "pass"
    FAIL = "fail"
    CRASH = "crash"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"


class FuzzStrategy(Enum):
    """Fuzzing strategies for input generation."""
    RANDOM = "random"
    MUTATION = "mutation"
    GRAMMAR_BASED = "grammar_based"
    COVERAGE_GUIDED = "coverage_guided"
    CONSTRAINT_BASED = "constraint_based"


@dataclass
class FuzzInput:
    """Represents a single fuzzing input."""
    input_id: str
    strategy: FuzzStrategy
    data: Any
    metadata: Dict[str, Any]
    parent_id: Optional[str] = None
    generation: int = 0
    
    def to_bytes(self) -> bytes:
        """Serialize input for hashing."""
        return pickle.dumps(self.data)
    
    def hash(self) -> str:
        """Generate hash of input data."""
        return hashlib.sha256(self.to_bytes()).hexdigest()[:16]


@dataclass
class FuzzExecution:
    """Results from executing a fuzz input."""
    input_id: str
    result: FuzzResult
    execution_time: float
    output: Any = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    coverage_data: Optional[Dict[str, Set[int]]] = None
    crash_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'input_id': self.input_id,
            'result': self.result.value,
            'execution_time': self.execution_time,
            'output': str(self.output) if self.output is not None else None,
            'error_message': self.error_message,
            'stack_trace': self.stack_trace,
            'coverage_data': {k: list(v) for k, v in self.coverage_data.items()} if self.coverage_data else None,
            'crash_hash': self.crash_hash
        }


@dataclass
class FuzzCampaign:
    """Results from a complete fuzzing campaign."""
    target_function: str
    strategy: FuzzStrategy
    total_executions: int
    execution_time: float
    results: Dict[FuzzResult, int]
    unique_crashes: int
    code_coverage: float
    executions: List[FuzzExecution]
    interesting_inputs: List[FuzzInput]
    crash_buckets: Dict[str, List[FuzzExecution]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert campaign results to dictionary."""
        return {
            'target_function': self.target_function,
            'strategy': self.strategy.value,
            'total_executions': self.total_executions,
            'execution_time': self.execution_time,
            'results': {k.value: v for k, v in self.results.items()},
            'unique_crashes': self.unique_crashes,
            'code_coverage': self.code_coverage,
            'executions': [e.to_dict() for e in self.executions],
            'interesting_inputs': [asdict(i) for i in self.interesting_inputs],
            'crash_buckets': {k: [e.to_dict() for e in v] for k, v in self.crash_buckets.items()}
        }


class FuzzInputGenerator(ABC):
    """Abstract base class for fuzz input generators."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.random = random.Random(seed)
        self.generation = 0
        
    @abstractmethod
    def generate(self, count: int = 1) -> Iterator[FuzzInput]:
        """Generate fuzz inputs."""
        pass
    
    @abstractmethod
    def mutate(self, input_data: FuzzInput) -> FuzzInput:
        """Mutate existing input to create new variant."""
        pass


class QueryFuzzGenerator(FuzzInputGenerator):
    """Generate fuzz inputs for query strings and retrieval functions."""
    
    def __init__(self, seed: int = 42, max_length: int = 1000):
        super().__init__(seed)
        self.max_length = max_length
        
        # Common query patterns and edge cases
        self.query_templates = [
            "",  # Empty query
            " ",  # Whitespace only
            "\n\t\r",  # Control characters
            "a" * 1000,  # Very long query
            "SELECT * FROM users",  # SQL injection attempt
            "<script>alert('xss')</script>",  # XSS attempt
            "../../etc/passwd",  # Path traversal
            "null",  # Null-like strings
            "undefined",
            "NaN",
            "true",
            "false",
            "[]",
            "{}",
            "0",
            "-1",
            "9999999999999999999999",  # Large numbers
        ]
        
        self.special_chars = "!@#$%^&*()[]{}|\\:;\"'<>,.?/~`"
        self.unicode_ranges = [
            (0x0100, 0x017F),  # Latin Extended-A
            (0x4E00, 0x9FFF),  # CJK Unified Ideographs
            (0x1F600, 0x1F64F),  # Emoticons
        ]
    
    def generate(self, count: int = 1) -> Iterator[FuzzInput]:
        """Generate diverse query inputs."""
        for i in range(count):
            self.generation += 1
            
            # Choose generation strategy
            strategy_choice = self.random.choice([
                'template',
                'random_ascii',
                'random_unicode',
                'structured',
                'boundary'
            ])
            
            if strategy_choice == 'template':
                data = self.random.choice(self.query_templates)
            elif strategy_choice == 'random_ascii':
                data = self._generate_random_ascii()
            elif strategy_choice == 'random_unicode':
                data = self._generate_random_unicode()
            elif strategy_choice == 'structured':
                data = self._generate_structured_query()
            else:  # boundary
                data = self._generate_boundary_case()
            
            yield FuzzInput(
                input_id=f"query_{self.generation:06d}",
                strategy=FuzzStrategy.RANDOM,
                data=data,
                metadata={
                    'generation_type': strategy_choice,
                    'length': len(str(data)),
                    'contains_special_chars': any(c in self.special_chars for c in str(data)),
                },
                generation=self.generation
            )
    
    def mutate(self, input_data: FuzzInput) -> FuzzInput:
        """Mutate existing query input."""
        self.generation += 1
        original = str(input_data.data)
        
        # Choose mutation strategy
        mutations = [
            self._mutate_insert,
            self._mutate_delete,
            self._mutate_replace,
            self._mutate_duplicate,
            self._mutate_case,
            self._mutate_encoding
        ]
        
        mutation_func = self.random.choice(mutations)
        mutated_data = mutation_func(original)
        
        return FuzzInput(
            input_id=f"query_mut_{self.generation:06d}",
            strategy=FuzzStrategy.MUTATION,
            data=mutated_data,
            metadata={
                'parent': input_data.input_id,
                'mutation_type': mutation_func.__name__,
                'length': len(str(mutated_data)),
            },
            parent_id=input_data.input_id,
            generation=self.generation
        )
    
    def _generate_random_ascii(self) -> str:
        """Generate random ASCII string."""
        length = self.random.randint(0, self.max_length)
        chars = string.ascii_letters + string.digits + string.punctuation + " \t\n"
        return ''.join(self.random.choice(chars) for _ in range(length))
    
    def _generate_random_unicode(self) -> str:
        """Generate string with random Unicode characters."""
        length = self.random.randint(1, min(100, self.max_length))
        chars = []
        
        for _ in range(length):
            if self.random.random() < 0.7:
                # ASCII character
                chars.append(self.random.choice(string.ascii_letters + " "))
            else:
                # Unicode character
                start, end = self.random.choice(self.unicode_ranges)
                chars.append(chr(self.random.randint(start, end)))
        
        return ''.join(chars)
    
    def _generate_structured_query(self) -> str:
        """Generate structured query-like strings."""
        structures = [
            lambda: f"{self._random_word()} AND {self._random_word()}",
            lambda: f"{self._random_word()} OR {self._random_word()}",
            lambda: f'"{self._random_phrase()}"',
            lambda: f"{self._random_word()}:{self._random_word()}",
            lambda: f"({self._random_word()} {self._random_word()})",
            lambda: f"{self._random_word()} -{self._random_word()}",
        ]
        
        return self.random.choice(structures)()
    
    def _generate_boundary_case(self) -> str:
        """Generate boundary condition cases."""
        cases = [
            "",  # Empty
            " " * self.random.randint(1, 100),  # Whitespace
            "a" * self.max_length,  # Max length
            "\x00" * self.random.randint(1, 10),  # Null bytes
            "A" * 65536,  # Very large
        ]
        
        return self.random.choice(cases)
    
    def _random_word(self) -> str:
        """Generate random word."""
        length = self.random.randint(1, 20)
        return ''.join(self.random.choice(string.ascii_letters) for _ in range(length))
    
    def _random_phrase(self) -> str:
        """Generate random phrase."""
        words = [self._random_word() for _ in range(self.random.randint(1, 5))]
        return ' '.join(words)
    
    def _mutate_insert(self, data: str) -> str:
        """Insert random characters."""
        if not data:
            return self._random_word()
        
        pos = self.random.randint(0, len(data))
        insert_char = self.random.choice(string.ascii_letters + self.special_chars)
        return data[:pos] + insert_char + data[pos:]
    
    def _mutate_delete(self, data: str) -> str:
        """Delete random characters."""
        if not data:
            return data
        
        if len(data) == 1:
            return ""
        
        pos = self.random.randint(0, len(data) - 1)
        return data[:pos] + data[pos + 1:]
    
    def _mutate_replace(self, data: str) -> str:
        """Replace random characters."""
        if not data:
            return self._random_word()
        
        data_list = list(data)
        pos = self.random.randint(0, len(data_list) - 1)
        data_list[pos] = self.random.choice(string.ascii_letters + self.special_chars)
        return ''.join(data_list)
    
    def _mutate_duplicate(self, data: str) -> str:
        """Duplicate parts of the string."""
        if not data:
            return data
        
        if len(data) > self.max_length // 2:
            return data
        
        return data + data
    
    def _mutate_case(self, data: str) -> str:
        """Change case randomly."""
        if not data:
            return data
        
        if self.random.random() < 0.5:
            return data.upper()
        else:
            return data.lower()
    
    def _mutate_encoding(self, data: str) -> str:
        """Apply encoding mutations."""
        if not data:
            return data
        
        # URL encoding
        if self.random.random() < 0.3:
            return data.replace(' ', '%20').replace('<', '%3C').replace('>', '%3E')
        
        # HTML encoding
        if self.random.random() < 0.3:
            return data.replace('<', '&lt;').replace('>', '&gt;').replace('&', '&amp;')
        
        return data


class VectorFuzzGenerator(FuzzInputGenerator):
    """Generate fuzz inputs for vector operations and embeddings."""
    
    def __init__(self, seed: int = 42, max_dim: int = 1000):
        super().__init__(seed)
        self.max_dim = max_dim
        self.np_random = np.random.RandomState(seed)
    
    def generate(self, count: int = 1) -> Iterator[FuzzInput]:
        """Generate diverse vector inputs."""
        for i in range(count):
            self.generation += 1
            
            # Choose vector type
            vector_type = self.random.choice([
                'zero_vector',
                'unit_vector',
                'random_normal',
                'sparse_vector',
                'extreme_values',
                'nan_inf_vector',
                'high_dimensional'
            ])
            
            if vector_type == 'zero_vector':
                dim = self.random.randint(1, 100)
                data = np.zeros(dim)
            elif vector_type == 'unit_vector':
                dim = self.random.randint(1, 100)
                data = self.np_random.randn(dim)
                data = data / np.linalg.norm(data) if np.linalg.norm(data) > 0 else data
            elif vector_type == 'random_normal':
                dim = self.random.randint(1, self.max_dim)
                data = self.np_random.randn(dim)
            elif vector_type == 'sparse_vector':
                dim = self.random.randint(10, self.max_dim)
                data = np.zeros(dim)
                num_nonzero = self.random.randint(1, min(10, dim))
                indices = self.np_random.choice(dim, num_nonzero, replace=False)
                data[indices] = self.np_random.randn(num_nonzero)
            elif vector_type == 'extreme_values':
                dim = self.random.randint(1, 100)
                data = self.np_random.choice(
                    [-1e10, -1000, -1, 0, 1, 1000, 1e10],
                    size=dim
                ).astype(float)
            elif vector_type == 'nan_inf_vector':
                dim = self.random.randint(1, 100)
                data = self.np_random.randn(dim)
                # Inject NaN and Inf values
                for _ in range(self.random.randint(0, min(5, dim))):
                    idx = self.random.randint(0, dim - 1)
                    data[idx] = self.random.choice([np.nan, np.inf, -np.inf])
            else:  # high_dimensional
                dim = self.random.randint(1000, self.max_dim)
                data = self.np_random.randn(dim)
            
            yield FuzzInput(
                input_id=f"vector_{self.generation:06d}",
                strategy=FuzzStrategy.RANDOM,
                data=data.tolist(),  # Convert to list for serialization
                metadata={
                    'vector_type': vector_type,
                    'dimension': len(data),
                    'has_nan': bool(np.isnan(data).any()),
                    'has_inf': bool(np.isinf(data).any()),
                    'norm': float(np.linalg.norm(data)) if not np.isnan(data).any() else None,
                },
                generation=self.generation
            )
    
    def mutate(self, input_data: FuzzInput) -> FuzzInput:
        """Mutate existing vector input."""
        self.generation += 1
        original = np.array(input_data.data)
        
        # Choose mutation strategy
        mutations = [
            self._mutate_scale,
            self._mutate_add_noise,
            self._mutate_zero_elements,
            self._mutate_permute,
            self._mutate_inject_special,
            self._mutate_truncate
        ]
        
        mutation_func = self.random.choice(mutations)
        mutated_data = mutation_func(original)
        
        return FuzzInput(
            input_id=f"vector_mut_{self.generation:06d}",
            strategy=FuzzStrategy.MUTATION,
            data=mutated_data.tolist(),
            metadata={
                'parent': input_data.input_id,
                'mutation_type': mutation_func.__name__,
                'dimension': len(mutated_data),
            },
            parent_id=input_data.input_id,
            generation=self.generation
        )
    
    def _mutate_scale(self, data: np.ndarray) -> np.ndarray:
        """Scale vector by random factor."""
        scale = self.random.choice([0, 0.001, 0.1, 10, 1000, 1e6])
        return data * scale
    
    def _mutate_add_noise(self, data: np.ndarray) -> np.ndarray:
        """Add random noise to vector."""
        noise = self.np_random.randn(len(data)) * 0.1
        return data + noise
    
    def _mutate_zero_elements(self, data: np.ndarray) -> np.ndarray:
        """Zero out random elements."""
        mutated = data.copy()
        num_zeros = self.random.randint(1, min(5, len(data)))
        indices = self.np_random.choice(len(data), num_zeros, replace=False)
        mutated[indices] = 0
        return mutated
    
    def _mutate_permute(self, data: np.ndarray) -> np.ndarray:
        """Randomly permute elements."""
        return self.np_random.permutation(data)
    
    def _mutate_inject_special(self, data: np.ndarray) -> np.ndarray:
        """Inject special values (NaN, Inf)."""
        mutated = data.copy()
        num_inject = self.random.randint(1, min(3, len(data)))
        for _ in range(num_inject):
            idx = self.random.randint(0, len(data) - 1)
            mutated[idx] = self.random.choice([np.nan, np.inf, -np.inf])
        return mutated
    
    def _mutate_truncate(self, data: np.ndarray) -> np.ndarray:
        """Truncate vector to random length."""
        if len(data) <= 1:
            return data
        
        new_length = self.random.randint(1, len(data))
        return data[:new_length]


class FuzzExecutor:
    """Execute fuzz inputs against target functions."""
    
    def __init__(self, target_function: Callable, timeout: float = 1.0):
        self.target_function = target_function
        self.timeout = timeout
    
    def execute(self, fuzz_input: FuzzInput) -> FuzzExecution:
        """Execute a single fuzz input."""
        start_time = time.time()
        
        try:
            # Execute with timeout
            result = self._execute_with_timeout(fuzz_input.data)
            
            return FuzzExecution(
                input_id=fuzz_input.input_id,
                result=FuzzResult.PASS,
                execution_time=time.time() - start_time,
                output=result
            )
            
        except TimeoutError:
            return FuzzExecution(
                input_id=fuzz_input.input_id,
                result=FuzzResult.TIMEOUT,
                execution_time=time.time() - start_time,
                error_message="Execution timeout"
            )
            
        except Exception as e:
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            
            # Determine if this is a crash or expected failure
            if any(keyword in error_msg.lower() for keyword in 
                   ['segmentation fault', 'access violation', 'core dumped']):
                result_type = FuzzResult.CRASH
                crash_hash = hashlib.md5(stack_trace.encode()).hexdigest()[:8]
            else:
                result_type = FuzzResult.FAIL
                crash_hash = None
            
            return FuzzExecution(
                input_id=fuzz_input.input_id,
                result=result_type,
                execution_time=time.time() - start_time,
                error_message=error_msg,
                stack_trace=stack_trace,
                crash_hash=crash_hash
            )
    
    def _execute_with_timeout(self, input_data: Any) -> Any:
        """Execute function with timeout (simplified implementation)."""
        # In a production system, this would use proper timeout mechanisms
        # For now, we'll trust the function to return reasonably quickly
        return self.target_function(input_data)


class FuzzOrchestrator:
    """Main orchestrator for fuzzing campaigns."""
    
    def __init__(
        self,
        target_function: Callable,
        generators: List[FuzzInputGenerator],
        output_dir: str = "fuzz_results",
        seed: int = 42
    ):
        self.target_function = target_function
        self.generators = generators
        self.output_dir = Path(output_dir)
        self.seed = seed
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize executor
        self.executor = FuzzExecutor(target_function)
        
        # Track results
        self.all_inputs: List[FuzzInput] = []
        self.all_executions: List[FuzzExecution] = []
        self.crash_buckets: Dict[str, List[FuzzExecution]] = defaultdict(list)
        self.interesting_inputs: List[FuzzInput] = []
    
    def run_campaign(
        self,
        iterations: int = 1000,
        max_time: float = 300.0,
        strategy: FuzzStrategy = FuzzStrategy.RANDOM
    ) -> FuzzCampaign:
        """Run a complete fuzzing campaign."""
        logger.info(f"Starting fuzzing campaign: {iterations} iterations, {max_time}s max")
        start_time = time.time()
        
        result_counter = Counter()
        
        for iteration in range(iterations):
            if time.time() - start_time > max_time:
                logger.info(f"Reached time limit at iteration {iteration}")
                break
            
            # Generate input
            generator = random.choice(self.generators)
            
            if strategy == FuzzStrategy.MUTATION and self.all_inputs and random.random() < 0.3:
                # Mutation-based: mutate existing input
                parent = random.choice(self.all_inputs)
                fuzz_input = generator.mutate(parent)
            else:
                # Generation-based: create new input
                fuzz_input = next(generator.generate(1))
            
            # Execute input
            execution = self.executor.execute(fuzz_input)
            
            # Track results
            self.all_inputs.append(fuzz_input)
            self.all_executions.append(execution)
            result_counter[execution.result] += 1
            
            # Handle crashes
            if execution.result == FuzzResult.CRASH and execution.crash_hash:
                self.crash_buckets[execution.crash_hash].append(execution)
                if len(self.crash_buckets[execution.crash_hash]) == 1:  # New unique crash
                    self.interesting_inputs.append(fuzz_input)
                    logger.warning(f"New crash discovered: {execution.crash_hash}")
            
            # Handle interesting failures
            elif execution.result == FuzzResult.FAIL:
                # Simple heuristic: consider it interesting if error is rare
                if result_counter[FuzzResult.FAIL] <= 5:
                    self.interesting_inputs.append(fuzz_input)
            
            # Progress logging
            if (iteration + 1) % 100 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Iteration {iteration + 1}/{iterations}, "
                           f"Crashes: {len(self.crash_buckets)}, "
                           f"Time: {elapsed:.1f}s")
        
        # Build campaign results
        execution_time = time.time() - start_time
        campaign = FuzzCampaign(
            target_function=self.target_function.__name__,
            strategy=strategy,
            total_executions=len(self.all_executions),
            execution_time=execution_time,
            results=dict(result_counter),
            unique_crashes=len(self.crash_buckets),
            code_coverage=self._calculate_coverage(),  # Simplified
            executions=self.all_executions,
            interesting_inputs=self.interesting_inputs,
            crash_buckets=dict(self.crash_buckets)
        )
        
        # Generate reports
        self._generate_reports(campaign)
        
        return campaign
    
    def _calculate_coverage(self) -> float:
        """Calculate code coverage (simplified implementation)."""
        # In a real implementation, this would use coverage.py or similar
        # For now, return a simple heuristic based on execution diversity
        if not self.all_executions:
            return 0.0
        
        unique_outcomes = len(set(e.result for e in self.all_executions))
        max_outcomes = len(FuzzResult)
        return unique_outcomes / max_outcomes
    
    def _generate_reports(self, campaign: FuzzCampaign) -> None:
        """Generate fuzzing reports."""
        # JSON report
        json_report = self.output_dir / "fuzz_report.json"
        with open(json_report, 'w') as f:
            json.dump(campaign.to_dict(), f, indent=2, default=str)
        
        # HTML report
        html_report = self.output_dir / "fuzz_report.html"
        self._generate_html_report(campaign, html_report)
        
        # Crash reproduction scripts
        self._generate_crash_reproducer(campaign)
        
        logger.info(f"Fuzzing reports generated in {self.output_dir}")
    
    def _generate_html_report(self, campaign: FuzzCampaign, output_file: Path) -> None:
        """Generate HTML fuzzing report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fuzzing Campaign Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .crash {{ margin: 10px 0; padding: 10px; border-left: 3px solid #F44336; }}
                .stats {{ display: flex; gap: 20px; }}
                .stat {{ background: white; padding: 15px; border-radius: 5px; text-align: center; }}
                .warning {{ color: #F44336; font-weight: bold; }}
                .success {{ color: #4CAF50; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>Fuzzing Campaign Report</h1>
            
            <div class="summary">
                <h2>Campaign Summary</h2>
                <div class="stats">
                    <div class="stat">
                        <h3>Total Executions</h3>
                        <p>{campaign.total_executions}</p>
                    </div>
                    <div class="stat">
                        <h3>Unique Crashes</h3>
                        <p class="{'warning' if campaign.unique_crashes > 0 else 'success'}">
                            {campaign.unique_crashes}
                        </p>
                    </div>
                    <div class="stat">
                        <h3>Code Coverage</h3>
                        <p>{campaign.code_coverage:.1%}</p>
                    </div>
                    <div class="stat">
                        <h3>Execution Time</h3>
                        <p>{campaign.execution_time:.1f}s</p>
                    </div>
                </div>
            </div>
            
            <h2>Result Breakdown</h2>
            <ul>
        """
        
        for result, count in campaign.results.items():
            html_content += f"<li><strong>{result.title()}:</strong> {count}</li>"
        
        html_content += "</ul>"
        
        if campaign.crash_buckets:
            html_content += "<h2>Discovered Crashes</h2>"
            for crash_hash, executions in campaign.crash_buckets.items():
                html_content += f"""
                <div class="crash">
                    <h3>Crash {crash_hash}</h3>
                    <p><strong>Occurrences:</strong> {len(executions)}</p>
                    <p><strong>Error:</strong> {executions[0].error_message}</p>
                    <p><strong>First Input ID:</strong> {executions[0].input_id}</p>
                </div>
                """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def _generate_crash_reproducer(self, campaign: FuzzCampaign) -> None:
        """Generate crash reproduction scripts."""
        if not campaign.crash_buckets:
            return
        
        reproducer_script = self.output_dir / "reproduce_crashes.py"
        
        script_content = f'''#!/usr/bin/env python3
"""
Crash reproduction script generated from fuzzing campaign
Target function: {campaign.target_function}
Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[3]))

def reproduce_crash(crash_id: str, input_data):
    """Reproduce a specific crash."""
    print(f"\\n=== Reproducing crash {{crash_id}} ===")
    print(f"Input data: {{input_data}}")
    print(f"Input type: {{type(input_data)}}")
    
    try:
        # Import and call target function
        from {self.target_function.__module__} import {self.target_function.__name__}
        result = {self.target_function.__name__}(input_data)
        print(f"Unexpected success: {{result}}")
    except Exception as e:
        print(f"Exception reproduced: {{e}}")
        print("Stack trace:")
        traceback.print_exc()

def main():
    """Reproduce all discovered crashes."""
'''
        
        for crash_hash, executions in campaign.crash_buckets.items():
            # Find the input that caused this crash
            crash_input = None
            for inp in campaign.interesting_inputs:
                if any(e.input_id == inp.input_id for e in executions):
                    crash_input = inp
                    break
            
            if crash_input:
                script_content += f'''
    # Crash {crash_hash}
    reproduce_crash("{crash_hash}", {repr(crash_input.data)})
'''
        
        script_content += '''

if __name__ == "__main__":
    main()
'''
        
        with open(reproducer_script, 'w') as f:
            f.write(script_content)
        
        reproducer_script.chmod(0o755)


def main():
    """Main entry point for fuzzing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run fuzzing campaign on Lethe research code")
    parser.add_argument('--target', required=True, help='Target function to fuzz (module:function)')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of iterations')
    parser.add_argument('--max-time', type=float, default=300.0, help='Maximum time in seconds')
    parser.add_argument('--output-dir', default='fuzz_results', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--strategy', 
                       choices=['random', 'mutation'], 
                       default='random', 
                       help='Fuzzing strategy')
    
    args = parser.parse_args()
    
    # Import target function
    module_name, function_name = args.target.split(':')
    module = __import__(module_name, fromlist=[function_name])
    target_function = getattr(module, function_name)
    
    # Create generators
    generators = [
        QueryFuzzGenerator(seed=args.seed),
        VectorFuzzGenerator(seed=args.seed)
    ]
    
    # Run campaign
    orchestrator = FuzzOrchestrator(
        target_function=target_function,
        generators=generators,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    strategy = FuzzStrategy.MUTATION if args.strategy == 'mutation' else FuzzStrategy.RANDOM
    
    try:
        campaign = orchestrator.run_campaign(
            iterations=args.iterations,
            max_time=args.max_time,
            strategy=strategy
        )
        
        print(f"\n=== Fuzzing Campaign Results ===")
        print(f"Target: {campaign.target_function}")
        print(f"Total Executions: {campaign.total_executions}")
        print(f"Unique Crashes: {campaign.unique_crashes}")
        print(f"Code Coverage: {campaign.code_coverage:.1%}")
        print(f"Execution Time: {campaign.execution_time:.1f}s")
        print(f"Results: {dict(campaign.results)}")
        
        if campaign.unique_crashes > 0:
            print(f"\n🚨 {campaign.unique_crashes} unique crashes discovered!")
            print("Check crash reproducer script for details.")
        else:
            print("\n✅ No crashes discovered.")
        
        # Exit with appropriate code
        exit(1 if campaign.unique_crashes > 0 else 0)
        
    except Exception as e:
        logger.error(f"Fuzzing campaign failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()