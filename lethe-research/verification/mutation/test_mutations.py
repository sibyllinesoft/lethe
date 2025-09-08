#!/usr/bin/env python3
"""
Mutation Testing Framework for Lethe Research Infrastructure

This module implements comprehensive mutation testing to achieve ≥0.80 mutation score
for critical research components. Targets core retrieval algorithms, scoring functions,
and statistical analysis methods.

Requirements:
- Mutation score ≥ 0.80 for all critical paths
- Deterministic mutation generation with seed control
- Comprehensive reporting with survival analysis
- Integration with existing test infrastructure
"""

import ast
import copy
import hashlib
import json
import logging
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
import sys
import os

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parents[3]))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MutationType(Enum):
    """Types of mutations to apply to source code."""
    ARITHMETIC_OPERATOR = "arithmetic_operator"
    RELATIONAL_OPERATOR = "relational_operator" 
    LOGICAL_OPERATOR = "logical_operator"
    CONSTANT_REPLACEMENT = "constant_replacement"
    VARIABLE_REPLACEMENT = "variable_replacement"
    RETURN_VALUE = "return_value"
    CONDITION_BOUNDARY = "condition_boundary"
    LOOP_BOUNDARY = "loop_boundary"
    EXCEPTION_HANDLING = "exception_handling"
    METHOD_CALL = "method_call"


@dataclass
class Mutation:
    """Represents a single code mutation."""
    mutation_id: str
    mutation_type: MutationType
    file_path: str
    line_number: int
    column_number: int
    original_code: str
    mutated_code: str
    description: str
    killed: bool = False
    execution_time: float = 0.0
    error_message: Optional[str] = None


@dataclass
class MutationResult:
    """Results from running mutation tests."""
    total_mutations: int
    killed_mutations: int
    survived_mutations: int
    mutation_score: float
    execution_time: float
    mutations: List[Mutation]
    coverage_map: Dict[str, Set[int]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        return {
            'total_mutations': self.total_mutations,
            'killed_mutations': self.killed_mutations,
            'survived_mutations': self.survived_mutations,
            'mutation_score': self.mutation_score,
            'execution_time': self.execution_time,
            'mutations': [
                {
                    'mutation_id': m.mutation_id,
                    'mutation_type': m.mutation_type.value,
                    'file_path': m.file_path,
                    'line_number': m.line_number,
                    'column_number': m.column_number,
                    'original_code': m.original_code,
                    'mutated_code': m.mutated_code,
                    'description': m.description,
                    'killed': m.killed,
                    'execution_time': m.execution_time,
                    'error_message': m.error_message
                }
                for m in self.mutations
            ],
            'coverage_map': {k: list(v) for k, v in self.coverage_map.items()}
        }


class MutationOperators:
    """Collection of mutation operators for different code constructs."""
    
    ARITHMETIC_OPS = {
        ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, 
        ast.Div: ast.Mult, ast.Mod: ast.Mult, ast.FloorDiv: ast.Div,
        ast.Pow: ast.Mult
    }
    
    RELATIONAL_OPS = {
        ast.Lt: ast.Le, ast.Le: ast.Lt, ast.Gt: ast.Ge,
        ast.Ge: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq
    }
    
    LOGICAL_OPS = {
        ast.And: ast.Or, ast.Or: ast.And
    }
    
    UNARY_OPS = {
        ast.Not: None, ast.UAdd: ast.USub, ast.USub: ast.UAdd
    }


class MutationGenerator(ast.NodeTransformer):
    """Generates mutations for Python AST nodes."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.random = random.Random(seed)
        self.mutations: List[Tuple[ast.AST, str, str]] = []
        self.current_file: str = ""
        
    def generate_mutations(self, source_code: str, file_path: str) -> List[Mutation]:
        """Generate all possible mutations for source code."""
        self.current_file = file_path
        self.mutations = []
        
        try:
            tree = ast.parse(source_code, filename=file_path)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return []
        
        # Generate mutations by visiting AST nodes
        self.visit(tree)
        
        mutations = []
        for i, (node, original, mutated) in enumerate(self.mutations):
            mutation_id = self._generate_mutation_id(file_path, node, i)
            mutations.append(Mutation(
                mutation_id=mutation_id,
                mutation_type=self._get_mutation_type(node),
                file_path=file_path,
                line_number=getattr(node, 'lineno', 0),
                column_number=getattr(node, 'col_offset', 0),
                original_code=original,
                mutated_code=mutated,
                description=f"Mutate {original} to {mutated}"
            ))
        
        return mutations
    
    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        """Visit binary operations for arithmetic/relational mutations."""
        self.generic_visit(node)
        
        op_type = type(node.op)
        if op_type in MutationOperators.ARITHMETIC_OPS:
            new_op = MutationOperators.ARITHMETIC_OPS[op_type]()
            original = ast.unparse(node.op) if hasattr(ast, 'unparse') else str(op_type.__name__)
            mutated = ast.unparse(new_op) if hasattr(ast, 'unparse') else str(type(new_op).__name__)
            self.mutations.append((node, original, mutated))
        
        return node
    
    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        """Visit comparison operations for relational mutations."""
        self.generic_visit(node)
        
        for i, op in enumerate(node.ops):
            op_type = type(op)
            if op_type in MutationOperators.RELATIONAL_OPS:
                new_op = MutationOperators.RELATIONAL_OPS[op_type]()
                original = ast.unparse(op) if hasattr(ast, 'unparse') else str(op_type.__name__)
                mutated = ast.unparse(new_op) if hasattr(ast, 'unparse') else str(type(new_op).__name__)
                self.mutations.append((node, original, mutated))
        
        return node
    
    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        """Visit boolean operations for logical mutations."""
        self.generic_visit(node)
        
        op_type = type(node.op)
        if op_type in MutationOperators.LOGICAL_OPS:
            new_op = MutationOperators.LOGICAL_OPS[op_type]()
            original = "and" if op_type == ast.And else "or"
            mutated = "and" if type(new_op) == ast.And else "or"
            self.mutations.append((node, original, mutated))
        
        return node
    
    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        """Visit constants for value mutations."""
        self.generic_visit(node)
        
        if isinstance(node.value, (int, float)):
            if node.value == 0:
                mutated_value = 1
            elif node.value == 1:
                mutated_value = 0
            else:
                mutated_value = node.value + 1
            
            original = str(node.value)
            mutated = str(mutated_value)
            self.mutations.append((node, original, mutated))
        elif isinstance(node.value, bool):
            mutated_value = not node.value
            original = str(node.value)
            mutated = str(mutated_value)
            self.mutations.append((node, original, mutated))
        elif isinstance(node.value, str) and node.value:
            mutated_value = ""
            original = f'"{node.value}"'
            mutated = '""'
            self.mutations.append((node, original, mutated))
        
        return node
    
    def visit_Return(self, node: ast.Return) -> ast.AST:
        """Visit return statements for return value mutations."""
        self.generic_visit(node)
        
        if node.value:
            original = ast.unparse(node.value) if hasattr(ast, 'unparse') else "return_value"
            # Common return value mutations
            mutations = ["None", "True", "False", "0", "1", "[]", "{}", '""']
            for mutated in mutations:
                self.mutations.append((node, f"return {original}", f"return {mutated}"))
        
        return node
    
    def visit_If(self, node: ast.If) -> ast.AST:
        """Visit if statements for condition mutations."""
        self.generic_visit(node)
        
        # Negate condition
        original = ast.unparse(node.test) if hasattr(ast, 'unparse') else "condition"
        mutated = f"not ({original})"
        self.mutations.append((node, f"if {original}", f"if {mutated}"))
        
        # Always true/false conditions
        self.mutations.append((node, f"if {original}", "if True"))
        self.mutations.append((node, f"if {original}", "if False"))
        
        return node
    
    def visit_For(self, node: ast.For) -> ast.AST:
        """Visit for loops for boundary mutations."""
        self.generic_visit(node)
        
        # Skip first/last iteration mutations would go here
        # This is a simplified version - full implementation would modify loop ranges
        
        return node
    
    def visit_While(self, node: ast.While) -> ast.AST:
        """Visit while loops for condition mutations."""
        self.generic_visit(node)
        
        original = ast.unparse(node.test) if hasattr(ast, 'unparse') else "condition"
        mutated = f"not ({original})"
        self.mutations.append((node, f"while {original}", f"while {mutated}"))
        
        return node
    
    def _generate_mutation_id(self, file_path: str, node: ast.AST, index: int) -> str:
        """Generate unique mutation ID."""
        content = f"{file_path}:{getattr(node, 'lineno', 0)}:{index}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def _get_mutation_type(self, node: ast.AST) -> MutationType:
        """Determine mutation type based on AST node."""
        if isinstance(node, ast.BinOp):
            return MutationType.ARITHMETIC_OPERATOR
        elif isinstance(node, ast.Compare):
            return MutationType.RELATIONAL_OPERATOR
        elif isinstance(node, ast.BoolOp):
            return MutationType.LOGICAL_OPERATOR
        elif isinstance(node, ast.Constant):
            return MutationType.CONSTANT_REPLACEMENT
        elif isinstance(node, ast.Return):
            return MutationType.RETURN_VALUE
        elif isinstance(node, ast.If):
            return MutationType.CONDITION_BOUNDARY
        elif isinstance(node, (ast.For, ast.While)):
            return MutationType.LOOP_BOUNDARY
        else:
            return MutationType.METHOD_CALL


class MutationTester:
    """Executes mutation tests against test suite."""
    
    def __init__(self, test_command: str = "python -m pytest", timeout: int = 30):
        self.test_command = test_command
        self.timeout = timeout
        self.original_test_results: Optional[subprocess.CompletedProcess] = None
    
    def run_original_tests(self, test_dir: str) -> bool:
        """Run original test suite to establish baseline."""
        logger.info("Running original test suite...")
        
        try:
            result = subprocess.run(
                f"{self.test_command} {test_dir}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=Path(test_dir).parent
            )
            
            self.original_test_results = result
            success = result.returncode == 0
            
            if success:
                logger.info("Original tests passed")
            else:
                logger.error(f"Original tests failed: {result.stderr}")
            
            return success
        except subprocess.TimeoutExpired:
            logger.error("Original tests timed out")
            return False
        except Exception as e:
            logger.error(f"Error running original tests: {e}")
            return False
    
    def test_mutation(self, mutation: Mutation, source_code: str, test_dir: str) -> bool:
        """Test a single mutation by applying it and running tests."""
        start_time = time.time()
        
        try:
            # Apply mutation to source code
            mutated_code = self._apply_mutation(source_code, mutation)
            
            # Write mutated code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(mutated_code)
                temp_file_path = temp_file.name
            
            # Backup original file
            original_path = Path(mutation.file_path)
            backup_path = original_path.with_suffix('.py.backup')
            
            if original_path.exists():
                original_path.rename(backup_path)
            
            # Copy mutated file to original location
            Path(temp_file_path).rename(original_path)
            
            try:
                # Run tests
                result = subprocess.run(
                    f"{self.test_command} {test_dir}",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=Path(test_dir).parent
                )
                
                # Mutation is killed if tests fail
                killed = result.returncode != 0
                
                if not killed:
                    logger.debug(f"Mutation survived: {mutation.description}")
                    mutation.error_message = "Tests still passed with mutation"
                else:
                    logger.debug(f"Mutation killed: {mutation.description}")
                
                mutation.killed = killed
                
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout testing mutation: {mutation.description}")
                mutation.killed = True
                mutation.error_message = "Test execution timeout"
            
            finally:
                # Restore original file
                if backup_path.exists():
                    original_path.unlink(missing_ok=True)
                    backup_path.rename(original_path)
                
                # Clean up temp file
                Path(temp_file_path).unlink(missing_ok=True)
        
        except Exception as e:
            logger.error(f"Error testing mutation {mutation.mutation_id}: {e}")
            mutation.killed = True
            mutation.error_message = str(e)
        
        mutation.execution_time = time.time() - start_time
        return mutation.killed
    
    def _apply_mutation(self, source_code: str, mutation: Mutation) -> str:
        """Apply mutation to source code."""
        # This is a simplified implementation
        # A full implementation would use AST manipulation to precisely apply mutations
        lines = source_code.split('\n')
        
        if 1 <= mutation.line_number <= len(lines):
            line = lines[mutation.line_number - 1]
            if mutation.original_code in line:
                mutated_line = line.replace(mutation.original_code, mutation.mutated_code, 1)
                lines[mutation.line_number - 1] = mutated_line
        
        return '\n'.join(lines)


class MutationTestRunner:
    """Main class for running comprehensive mutation tests."""
    
    def __init__(
        self,
        target_dirs: List[str],
        test_dir: str,
        output_dir: str = "mutation_results",
        seed: int = 42,
        min_mutation_score: float = 0.80,
        test_command: str = "python -m pytest -v",
        timeout: int = 30
    ):
        self.target_dirs = [Path(d) for d in target_dirs]
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.min_mutation_score = min_mutation_score
        self.test_command = test_command
        self.timeout = timeout
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.mutation_generator = MutationGenerator(seed)
        self.mutation_tester = MutationTester(test_command, timeout)
    
    def run_mutation_tests(self, parallel: bool = False) -> MutationResult:
        """Run complete mutation testing process."""
        logger.info("Starting mutation testing...")
        start_time = time.time()
        
        # Step 1: Verify original tests pass
        if not self.mutation_tester.run_original_tests(str(self.test_dir)):
            raise RuntimeError("Original tests must pass before running mutation tests")
        
        # Step 2: Generate all mutations
        all_mutations = self._generate_all_mutations()
        logger.info(f"Generated {len(all_mutations)} mutations")
        
        # Step 3: Filter mutations using coverage (if available)
        covered_mutations = self._filter_by_coverage(all_mutations)
        logger.info(f"Testing {len(covered_mutations)} covered mutations")
        
        # Step 4: Execute mutations
        killed_count = 0
        for i, mutation in enumerate(covered_mutations, 1):
            logger.info(f"Testing mutation {i}/{len(covered_mutations)}: {mutation.description}")
            
            source_code = self._read_source_file(mutation.file_path)
            if source_code:
                killed = self.mutation_tester.test_mutation(mutation, source_code, str(self.test_dir))
                if killed:
                    killed_count += 1
        
        # Step 5: Calculate results
        execution_time = time.time() - start_time
        mutation_score = killed_count / len(covered_mutations) if covered_mutations else 0.0
        
        result = MutationResult(
            total_mutations=len(covered_mutations),
            killed_mutations=killed_count,
            survived_mutations=len(covered_mutations) - killed_count,
            mutation_score=mutation_score,
            execution_time=execution_time,
            mutations=covered_mutations,
            coverage_map={}  # Would be populated by coverage analysis
        )
        
        # Step 6: Generate reports
        self._generate_reports(result)
        
        # Step 7: Validate against threshold
        if mutation_score < self.min_mutation_score:
            logger.warning(
                f"Mutation score {mutation_score:.3f} below threshold {self.min_mutation_score:.3f}"
            )
        else:
            logger.info(f"Mutation score {mutation_score:.3f} meets threshold")
        
        return result
    
    def _generate_all_mutations(self) -> List[Mutation]:
        """Generate mutations for all target files."""
        all_mutations = []
        
        for target_dir in self.target_dirs:
            if not target_dir.exists():
                logger.warning(f"Target directory does not exist: {target_dir}")
                continue
            
            python_files = list(target_dir.rglob("*.py"))
            for py_file in python_files:
                if py_file.name.startswith("test_"):
                    continue  # Skip test files
                
                source_code = self._read_source_file(str(py_file))
                if source_code:
                    mutations = self.mutation_generator.generate_mutations(
                        source_code, str(py_file)
                    )
                    all_mutations.extend(mutations)
                    logger.debug(f"Generated {len(mutations)} mutations for {py_file}")
        
        return all_mutations
    
    def _filter_by_coverage(self, mutations: List[Mutation]) -> List[Mutation]:
        """Filter mutations to only include covered code (simplified)."""
        # In a full implementation, this would use coverage.py to identify
        # which lines are executed by the test suite
        return mutations
    
    def _read_source_file(self, file_path: str) -> Optional[str]:
        """Read source file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None
    
    def _generate_reports(self, result: MutationResult) -> None:
        """Generate mutation testing reports."""
        # JSON report
        json_report = self.output_dir / "mutation_report.json"
        with open(json_report, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # HTML report
        html_report = self.output_dir / "mutation_report.html"
        self._generate_html_report(result, html_report)
        
        # CSV report for statistical analysis
        csv_report = self.output_dir / "mutations.csv"
        self._generate_csv_report(result, csv_report)
        
        logger.info(f"Reports generated in {self.output_dir}")
    
    def _generate_html_report(self, result: MutationResult, output_file: Path) -> None:
        """Generate HTML mutation testing report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mutation Testing Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .mutation {{ margin: 10px 0; padding: 10px; border-left: 3px solid #ccc; }}
                .killed {{ border-left-color: #4CAF50; }}
                .survived {{ border-left-color: #F44336; }}
                .score {{ font-size: 24px; font-weight: bold; }}
                .threshold-met {{ color: #4CAF50; }}
                .threshold-not-met {{ color: #F44336; }}
            </style>
        </head>
        <body>
            <h1>Mutation Testing Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Mutations:</strong> {result.total_mutations}</p>
                <p><strong>Killed Mutations:</strong> {result.killed_mutations}</p>
                <p><strong>Survived Mutations:</strong> {result.survived_mutations}</p>
                <p><strong>Execution Time:</strong> {result.execution_time:.2f} seconds</p>
                <p><strong>Mutation Score:</strong> 
                    <span class="score {'threshold-met' if result.mutation_score >= self.min_mutation_score else 'threshold-not-met'}">
                        {result.mutation_score:.3f}
                    </span>
                    (Threshold: {self.min_mutation_score:.2f})
                </p>
            </div>
            
            <h2>Mutations</h2>
        """
        
        for mutation in result.mutations:
            status_class = "killed" if mutation.killed else "survived"
            status_text = "KILLED" if mutation.killed else "SURVIVED"
            
            html_content += f"""
            <div class="mutation {status_class}">
                <h3>{mutation.mutation_id} - {status_text}</h3>
                <p><strong>Type:</strong> {mutation.mutation_type.value}</p>
                <p><strong>File:</strong> {mutation.file_path}</p>
                <p><strong>Line:</strong> {mutation.line_number}</p>
                <p><strong>Change:</strong> {mutation.original_code} → {mutation.mutated_code}</p>
                <p><strong>Time:</strong> {mutation.execution_time:.3f}s</p>
                {f'<p><strong>Error:</strong> {mutation.error_message}</p>' if mutation.error_message else ''}
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def _generate_csv_report(self, result: MutationResult, output_file: Path) -> None:
        """Generate CSV report for statistical analysis."""
        import csv
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'mutation_id', 'mutation_type', 'file_path', 'line_number',
                'original_code', 'mutated_code', 'killed', 'execution_time', 'error_message'
            ])
            
            for mutation in result.mutations:
                writer.writerow([
                    mutation.mutation_id,
                    mutation.mutation_type.value,
                    mutation.file_path,
                    mutation.line_number,
                    mutation.original_code,
                    mutation.mutated_code,
                    mutation.killed,
                    mutation.execution_time,
                    mutation.error_message or ''
                ])


def main():
    """Main entry point for mutation testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run mutation tests on Lethe research code")
    parser.add_argument(
        '--targets',
        nargs='+',
        default=['lethe-research/datasets', 'lethe-research/experiments', 'lethe-research/verification'],
        help='Target directories to mutate'
    )
    parser.add_argument(
        '--test-dir',
        default='tests',
        help='Test directory'
    )
    parser.add_argument(
        '--output-dir',
        default='mutation_results',
        help='Output directory for reports'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.80,
        help='Minimum mutation score threshold'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for deterministic mutations'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Test timeout in seconds'
    )
    
    args = parser.parse_args()
    
    runner = MutationTestRunner(
        target_dirs=args.targets,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        min_mutation_score=args.threshold,
        seed=args.seed,
        timeout=args.timeout
    )
    
    try:
        result = runner.run_mutation_tests()
        
        print(f"\n=== Mutation Testing Results ===")
        print(f"Total Mutations: {result.total_mutations}")
        print(f"Killed: {result.killed_mutations}")
        print(f"Survived: {result.survived_mutations}")
        print(f"Mutation Score: {result.mutation_score:.3f}")
        print(f"Threshold: {args.threshold:.2f}")
        print(f"Status: {'PASS' if result.mutation_score >= args.threshold else 'FAIL'}")
        print(f"Execution Time: {result.execution_time:.2f}s")
        
        # Exit with appropriate code
        exit(0 if result.mutation_score >= args.threshold else 1)
        
    except Exception as e:
        logger.error(f"Mutation testing failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()