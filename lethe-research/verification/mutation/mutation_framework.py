#!/usr/bin/env python3
"""
Mutation Testing Framework for Lethe vNext
==========================================

Implements comprehensive mutation testing to achieve ≥0.80 mutation score
for critical retrieval algorithms including sentence_prune, knapsack_pack,
and bookend_linearize.

This framework generates semantic mutants that test the robustness of
the core algorithms against subtle bugs that could affect research validity.
"""

import ast
import copy
import hashlib
import importlib.util
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Callable, Tuple
import json
import sys

@dataclass
class MutationOperator:
    """Represents a mutation operator that can be applied to code"""
    name: str
    description: str
    pattern: str
    replacement: str
    applies_to: List[str]  # Code patterns this applies to
    risk_level: str  # 'low', 'medium', 'high'
    
@dataclass
class Mutant:
    """Represents a specific mutant of the original code"""
    mutant_id: str
    operator: MutationOperator
    file_path: str
    line_number: int
    column_number: int
    original_code: str
    mutated_code: str
    killed: bool = False
    detected_by: List[str] = None
    execution_time: float = 0.0
    
@dataclass
class MutationTestResult:
    """Results of mutation testing"""
    total_mutants: int
    killed_mutants: int
    survived_mutants: int
    mutation_score: float
    execution_time: float
    mutants_by_operator: Dict[str, int]
    killed_by_operator: Dict[str, int]
    detailed_results: List[Mutant]

class MutationGenerator:
    """Generates semantic mutants for Lethe core algorithms"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.operators = self._define_mutation_operators()
        
    def _define_mutation_operators(self) -> List[MutationOperator]:
        """Define mutation operators specific to retrieval algorithms"""
        return [
            # Arithmetic operators
            MutationOperator(
                name="arithmetic_replacement",
                description="Replace arithmetic operators (+, -, *, /)",
                pattern="BinOp",
                replacement="different_binop",
                applies_to=["scoring", "aggregation", "ranking"],
                risk_level="high"
            ),
            
            # Comparison operators
            MutationOperator(
                name="comparison_replacement", 
                description="Replace comparison operators (>, <, >=, <=, ==, !=)",
                pattern="Compare",
                replacement="different_compare",
                applies_to=["threshold_check", "ranking", "filtering"],
                risk_level="high"
            ),
            
            # Boolean operators
            MutationOperator(
                name="boolean_replacement",
                description="Replace boolean operators (and, or, not)",
                pattern="BoolOp",
                replacement="different_boolop",
                applies_to=["conditions", "filtering", "group_rules"],
                risk_level="high"
            ),
            
            # Constant mutations
            MutationOperator(
                name="constant_replacement",
                description="Replace numeric constants with +1, -1, 0",
                pattern="Constant",
                replacement="modified_constant",
                applies_to=["thresholds", "indices", "sizes"],
                risk_level="medium"
            ),
            
            # List/array access mutations
            MutationOperator(
                name="index_replacement",
                description="Replace array indices (first/last element access)",
                pattern="Subscript",
                replacement="different_index",
                applies_to=["ranking", "selection", "ordering"],
                risk_level="high"
            ),
            
            # Function call mutations
            MutationOperator(
                name="function_replacement",
                description="Replace built-in functions (max->min, sum->len, etc.)",
                pattern="Call",
                replacement="different_function",
                applies_to=["aggregation", "ranking", "scoring"],
                risk_level="high"
            ),
            
            # Slice mutations
            MutationOperator(
                name="slice_replacement",
                description="Modify slice operations (start, stop, step)",
                pattern="Slice",
                replacement="modified_slice",
                applies_to=["truncation", "selection", "windowing"],
                risk_level="medium"
            ),
            
            # Loop mutations
            MutationOperator(
                name="loop_boundary",
                description="Modify loop boundaries (off-by-one errors)",
                pattern="For",
                replacement="modified_range",
                applies_to=["iteration", "processing", "aggregation"],
                risk_level="high"
            ),
            
            # Return value mutations
            MutationOperator(
                name="return_mutation",
                description="Modify return values (empty list, None, etc.)",
                pattern="Return", 
                replacement="different_return",
                applies_to=["result_generation", "error_handling"],
                risk_level="medium"
            ),
            
            # Variable mutations (specific to retrieval)
            MutationOperator(
                name="scoring_mutation",
                description="Mutate scoring logic (weights, combinations)",
                pattern="scoring_logic",
                replacement="modified_scoring",
                applies_to=["sentence_scoring", "chunk_scoring", "fusion"],
                risk_level="high"
            ),
        ]
    
    def generate_mutants(self, source_file: Path, target_functions: List[str] = None) -> List[Mutant]:
        """
        Generate all possible mutants for a source file.
        
        Args:
            source_file: Path to Python source file
            target_functions: Optional list of function names to target
            
        Returns:
            List of generated mutants
        """
        self.logger.info(f"Generating mutants for {source_file}")
        
        with open(source_file, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Parse AST
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            self.logger.error(f"Failed to parse {source_file}: {e}")
            return []
        
        mutants = []
        mutant_counter = 0
        
        # Walk through AST and apply operators
        for node in ast.walk(tree):
            for operator in self.operators:
                candidate_mutants = self._apply_operator(
                    node, operator, source_code, source_file, target_functions
                )
                for mutant in candidate_mutants:
                    mutant.mutant_id = f"{source_file.stem}_{mutant_counter:04d}"
                    mutants.append(mutant)
                    mutant_counter += 1
        
        self.logger.info(f"Generated {len(mutants)} mutants for {source_file}")
        return mutants
    
    def _apply_operator(self, node: ast.AST, operator: MutationOperator, 
                       source_code: str, source_file: Path, 
                       target_functions: List[str] = None) -> List[Mutant]:
        """Apply a mutation operator to an AST node"""
        mutants = []
        
        # Skip if we're targeting specific functions and this node isn't in them
        if target_functions and not self._is_in_target_function(node, source_code, target_functions):
            return mutants
            
        if operator.name == "arithmetic_replacement" and isinstance(node, ast.BinOp):
            mutants.extend(self._mutate_arithmetic(node, operator, source_code, source_file))
            
        elif operator.name == "comparison_replacement" and isinstance(node, ast.Compare):
            mutants.extend(self._mutate_comparison(node, operator, source_code, source_file))
            
        elif operator.name == "boolean_replacement" and isinstance(node, ast.BoolOp):
            mutants.extend(self._mutate_boolean(node, operator, source_code, source_file))
            
        elif operator.name == "constant_replacement" and isinstance(node, (ast.Constant, ast.Num)):
            mutants.extend(self._mutate_constant(node, operator, source_code, source_file))
            
        elif operator.name == "index_replacement" and isinstance(node, ast.Subscript):
            mutants.extend(self._mutate_index(node, operator, source_code, source_file))
            
        elif operator.name == "function_replacement" and isinstance(node, ast.Call):
            mutants.extend(self._mutate_function_call(node, operator, source_code, source_file))
            
        elif operator.name == "slice_replacement" and isinstance(node, ast.Slice):
            mutants.extend(self._mutate_slice(node, operator, source_code, source_file))
            
        return mutants
    
    def _is_in_target_function(self, node: ast.AST, source_code: str, target_functions: List[str]) -> bool:
        """Check if node is within one of the target functions"""
        # Simplified implementation - in practice, would track function scope
        return True  # For now, apply to all nodes
    
    def _mutate_arithmetic(self, node: ast.BinOp, operator: MutationOperator, 
                          source_code: str, source_file: Path) -> List[Mutant]:
        """Generate arithmetic operator mutations"""
        mutants = []
        original_op = type(node.op).__name__
        
        # Define replacement mappings
        replacements = {
            'Add': ['Sub', 'Mult', 'Div'],
            'Sub': ['Add', 'Mult', 'Div'], 
            'Mult': ['Add', 'Sub', 'Div'],
            'Div': ['Add', 'Sub', 'Mult'],
        }
        
        if original_op in replacements:
            for replacement_op in replacements[original_op]:
                mutant = self._create_mutant_from_node(
                    node, operator, source_code, source_file,
                    original_op, replacement_op
                )
                if mutant:
                    mutants.append(mutant)
        
        return mutants
    
    def _mutate_comparison(self, node: ast.Compare, operator: MutationOperator,
                          source_code: str, source_file: Path) -> List[Mutant]:
        """Generate comparison operator mutations"""
        mutants = []
        
        # Only handle single comparisons for simplicity
        if len(node.ops) != 1:
            return mutants
            
        original_op = type(node.ops[0]).__name__
        
        replacements = {
            'Lt': ['Gt', 'LtE', 'GtE', 'Eq', 'NotEq'],
            'Gt': ['Lt', 'LtE', 'GtE', 'Eq', 'NotEq'],
            'LtE': ['Lt', 'Gt', 'GtE', 'Eq', 'NotEq'], 
            'GtE': ['Lt', 'Gt', 'LtE', 'Eq', 'NotEq'],
            'Eq': ['NotEq', 'Lt', 'Gt'],
            'NotEq': ['Eq', 'Lt', 'Gt'],
        }
        
        if original_op in replacements:
            for replacement_op in replacements[original_op]:
                mutant = self._create_mutant_from_node(
                    node, operator, source_code, source_file,
                    original_op, replacement_op
                )
                if mutant:
                    mutants.append(mutant)
        
        return mutants
    
    def _mutate_boolean(self, node: ast.BoolOp, operator: MutationOperator,
                       source_code: str, source_file: Path) -> List[Mutant]:
        """Generate boolean operator mutations"""
        mutants = []
        original_op = type(node.op).__name__
        
        replacements = {
            'And': ['Or'],
            'Or': ['And'],
        }
        
        if original_op in replacements:
            for replacement_op in replacements[original_op]:
                mutant = self._create_mutant_from_node(
                    node, operator, source_code, source_file,
                    original_op, replacement_op
                )
                if mutant:
                    mutants.append(mutant)
        
        return mutants
    
    def _mutate_constant(self, node: ast.AST, operator: MutationOperator,
                        source_code: str, source_file: Path) -> List[Mutant]:
        """Generate constant mutations"""
        mutants = []
        
        # Get original value
        if hasattr(node, 'value'):
            original_value = node.value
        elif hasattr(node, 'n'):  # Python < 3.8
            original_value = node.n
        else:
            return mutants
        
        # Only mutate numbers
        if isinstance(original_value, (int, float)):
            # Generate mutations: +1, -1, 0, *2, /2
            mutations = [
                original_value + 1,
                original_value - 1, 
                0,
                original_value * 2,
                original_value / 2 if original_value != 0 else 1
            ]
            
            for mutated_value in mutations:
                if mutated_value != original_value:
                    mutant = self._create_mutant_from_node(
                        node, operator, source_code, source_file,
                        str(original_value), str(mutated_value)
                    )
                    if mutant:
                        mutants.append(mutant)
        
        return mutants
    
    def _mutate_index(self, node: ast.Subscript, operator: MutationOperator,
                     source_code: str, source_file: Path) -> List[Mutant]:
        """Generate index access mutations"""
        mutants = []
        
        # Handle simple index access
        if isinstance(node.slice, ast.Constant) or isinstance(node.slice, ast.Num):
            # Change index by ±1
            original = self._get_source_segment(node, source_code)
            if '[0]' in original:
                mutant = self._create_mutant_from_node(
                    node, operator, source_code, source_file,
                    '[0]', '[1]'
                )
                if mutant:
                    mutants.append(mutant)
            elif '[-1]' in original:
                mutant = self._create_mutant_from_node(
                    node, operator, source_code, source_file,
                    '[-1]', '[0]'
                )
                if mutant:
                    mutants.append(mutant)
        
        return mutants
    
    def _mutate_function_call(self, node: ast.Call, operator: MutationOperator,
                             source_code: str, source_file: Path) -> List[Mutant]:
        """Generate function call mutations"""
        mutants = []
        
        # Get function name
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            
            # Define critical function replacements for retrieval algorithms
            replacements = {
                'max': ['min', 'sum', 'len'],
                'min': ['max', 'sum', 'len'],
                'sum': ['max', 'min', 'len'],
                'len': ['sum'],
                'sorted': ['reversed'],
                'reversed': ['sorted'],
                'abs': ['identity'],
                'round': ['int', 'float'],
            }
            
            if func_name in replacements:
                for replacement in replacements[func_name]:
                    mutant = self._create_mutant_from_node(
                        node, operator, source_code, source_file,
                        func_name, replacement
                    )
                    if mutant:
                        mutants.append(mutant)
        
        return mutants
    
    def _mutate_slice(self, node: ast.Slice, operator: MutationOperator,
                     source_code: str, source_file: Path) -> List[Mutant]:
        """Generate slice mutations"""
        # Implement slice mutations (start:end:step modifications)
        return []  # Simplified for now
    
    def _create_mutant_from_node(self, node: ast.AST, operator: MutationOperator,
                                source_code: str, source_file: Path,
                                original_pattern: str, replacement_pattern: str) -> Optional[Mutant]:
        """Create a mutant by modifying the source code"""
        
        # Get node location
        if not hasattr(node, 'lineno') or not hasattr(node, 'col_offset'):
            return None
        
        # Get the original source segment
        lines = source_code.split('\n')
        if node.lineno > len(lines):
            return None
            
        original_line = lines[node.lineno - 1]
        
        # Simple text replacement (in practice, would use more sophisticated AST manipulation)
        if original_pattern in original_line:
            mutated_line = original_line.replace(original_pattern, replacement_pattern, 1)
            mutated_code = '\n'.join(lines[:node.lineno-1] + [mutated_line] + lines[node.lineno:])
            
            return Mutant(
                mutant_id="",  # Will be set by caller
                operator=operator,
                file_path=str(source_file),
                line_number=node.lineno,
                column_number=node.col_offset,
                original_code=original_line,
                mutated_code=mutated_line,
                detected_by=[]
            )
        
        return None
    
    def _get_source_segment(self, node: ast.AST, source_code: str) -> str:
        """Get the source code segment for an AST node"""
        lines = source_code.split('\n')
        if hasattr(node, 'lineno') and node.lineno <= len(lines):
            return lines[node.lineno - 1]
        return ""

class MutationTester:
    """Executes mutation testing and evaluates test suite quality"""
    
    def __init__(self, test_command: str = "python -m pytest"):
        self.logger = logging.getLogger(__name__)
        self.test_command = test_command
        self.generator = MutationGenerator()
    
    def run_mutation_testing(self, source_files: List[Path], 
                           test_files: List[Path],
                           target_functions: List[str] = None,
                           output_dir: Path = None) -> MutationTestResult:
        """
        Run complete mutation testing process.
        
        Args:
            source_files: List of source files to mutate
            test_files: List of test files to run
            target_functions: Optional list of functions to target
            output_dir: Optional directory to save results
            
        Returns:
            MutationTestResult with detailed analysis
        """
        self.logger.info("Starting mutation testing...")
        start_time = time.time()
        
        # Generate all mutants
        all_mutants = []
        for source_file in source_files:
            mutants = self.generator.generate_mutants(source_file, target_functions)
            all_mutants.extend(mutants)
        
        self.logger.info(f"Generated {len(all_mutants)} mutants total")
        
        # Run baseline tests to ensure they pass
        if not self._run_baseline_tests(test_files):
            raise RuntimeError("Baseline tests are failing - cannot proceed with mutation testing")
        
        # Execute mutation testing
        killed_mutants = 0
        survived_mutants = []
        mutants_by_operator = {}
        killed_by_operator = {}
        
        for i, mutant in enumerate(all_mutants):
            if i % 50 == 0:
                self.logger.info(f"Processing mutant {i+1}/{len(all_mutants)}")
            
            # Track by operator
            op_name = mutant.operator.name
            mutants_by_operator[op_name] = mutants_by_operator.get(op_name, 0) + 1
            
            # Test if mutant is killed
            start_time_mutant = time.time()
            is_killed, detected_by = self._test_mutant(mutant, test_files)
            mutant.execution_time = time.time() - start_time_mutant
            mutant.killed = is_killed
            mutant.detected_by = detected_by
            
            if is_killed:
                killed_mutants += 1
                killed_by_operator[op_name] = killed_by_operator.get(op_name, 0) + 1
            else:
                survived_mutants.append(mutant)
        
        total_time = time.time() - start_time
        mutation_score = killed_mutants / len(all_mutants) if all_mutants else 0.0
        
        result = MutationTestResult(
            total_mutants=len(all_mutants),
            killed_mutants=killed_mutants,
            survived_mutants=len(survived_mutants),
            mutation_score=mutation_score,
            execution_time=total_time,
            mutants_by_operator=mutants_by_operator,
            killed_by_operator=killed_by_operator,
            detailed_results=all_mutants
        )
        
        # Save results
        if output_dir:
            self._save_results(result, output_dir)
        
        self.logger.info(f"Mutation testing completed: {mutation_score:.3f} score ({killed_mutants}/{len(all_mutants)} killed)")
        return result
    
    def _run_baseline_tests(self, test_files: List[Path]) -> bool:
        """Run tests to ensure they pass without mutations"""
        self.logger.info("Running baseline tests...")
        
        try:
            result = subprocess.run(
                self.test_command.split() + [str(f) for f in test_files],
                capture_output=True,
                text=True,
                timeout=120
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            self.logger.error("Baseline tests timed out")
            return False
        except Exception as e:
            self.logger.error(f"Failed to run baseline tests: {e}")
            return False
    
    def _test_mutant(self, mutant: Mutant, test_files: List[Path]) -> Tuple[bool, List[str]]:
        """
        Test if a mutant is killed by the test suite.
        
        Returns:
            (is_killed, detected_by_tests)
        """
        # Create temporary file with mutated code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(mutant.mutated_code)
            temp_file_path = Path(temp_file.name)
        
        try:
            # Replace original file temporarily and run tests
            original_path = Path(mutant.file_path)
            backup_content = original_path.read_text()
            
            # Write mutated code to original file
            original_path.write_text(mutant.mutated_code)
            
            # Run tests
            try:
                result = subprocess.run(
                    self.test_command.split() + [str(f) for f in test_files],
                    capture_output=True,
                    text=True,
                    timeout=30  # Shorter timeout for individual mutants
                )
                
                # Mutant is killed if tests fail
                is_killed = result.returncode != 0
                detected_by = self._extract_failing_tests(result.stdout + result.stderr) if is_killed else []
                
            except subprocess.TimeoutExpired:
                # Timeout counts as killed (infinite loop detection)
                is_killed = True
                detected_by = ["timeout"]
                
            finally:
                # Restore original file
                original_path.write_text(backup_content)
            
        except Exception as e:
            self.logger.error(f"Error testing mutant {mutant.mutant_id}: {e}")
            is_killed = False
            detected_by = []
        
        finally:
            # Clean up temporary file
            temp_file_path.unlink(missing_ok=True)
        
        return is_killed, detected_by
    
    def _extract_failing_tests(self, test_output: str) -> List[str]:
        """Extract names of failing tests from test output"""
        failing_tests = []
        lines = test_output.split('\n')
        
        for line in lines:
            if 'FAILED' in line and '::' in line:
                # Extract test name (simplified)
                parts = line.split('::')
                if len(parts) >= 2:
                    test_name = parts[-1].split()[0]
                    failing_tests.append(test_name)
        
        return failing_tests
    
    def _save_results(self, result: MutationTestResult, output_dir: Path):
        """Save mutation testing results to files"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary = {
            "mutation_score": result.mutation_score,
            "total_mutants": result.total_mutants,
            "killed_mutants": result.killed_mutants,
            "survived_mutants": result.survived_mutants,
            "execution_time": result.execution_time,
            "mutants_by_operator": result.mutants_by_operator,
            "killed_by_operator": result.killed_by_operator
        }
        
        with open(output_dir / "mutation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        detailed = []
        for mutant in result.detailed_results:
            detailed.append({
                "mutant_id": mutant.mutant_id,
                "operator": mutant.operator.name,
                "file_path": mutant.file_path,
                "line_number": mutant.line_number,
                "original_code": mutant.original_code,
                "mutated_code": mutant.mutated_code,
                "killed": mutant.killed,
                "detected_by": mutant.detected_by,
                "execution_time": mutant.execution_time
            })
        
        with open(output_dir / "mutation_details.json", 'w') as f:
            json.dump(detailed, f, indent=2)
        
        # Save survived mutants (need attention)
        survived = [m for m in result.detailed_results if not m.killed]
        with open(output_dir / "survived_mutants.json", 'w') as f:
            json.dump([{
                "mutant_id": m.mutant_id,
                "operator": m.operator.name,
                "file_path": m.file_path,
                "line_number": m.line_number,
                "original_code": m.original_code,
                "mutated_code": m.mutated_code
            } for m in survived], f, indent=2)

def main():
    """Main entry point for mutation testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mutation Testing for Lethe vNext")
    parser.add_argument("--source-files", nargs="+", required=True,
                       help="Source files to mutate")
    parser.add_argument("--test-files", nargs="+", required=True,
                       help="Test files to run")
    parser.add_argument("--target-functions", nargs="*",
                       help="Specific functions to target")
    parser.add_argument("--output-dir", type=str, default="./mutation_output",
                       help="Output directory for results")
    parser.add_argument("--test-command", type=str, default="python -m pytest",
                       help="Command to run tests")
    parser.add_argument("--target-score", type=float, default=0.80,
                       help="Target mutation score")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Convert paths
    source_files = [Path(f) for f in args.source_files]
    test_files = [Path(f) for f in args.test_files]
    output_dir = Path(args.output_dir)
    
    # Validate files exist
    for f in source_files + test_files:
        if not f.exists():
            print(f"Error: File not found: {f}")
            return 1
    
    # Run mutation testing
    tester = MutationTester(test_command=args.test_command)
    
    try:
        result = tester.run_mutation_testing(
            source_files=source_files,
            test_files=test_files,
            target_functions=args.target_functions,
            output_dir=output_dir
        )
        
        # Print results
        print(f"\n🧬 Mutation Testing Results:")
        print(f"  📊 Mutation Score: {result.mutation_score:.3f} ({result.killed_mutants}/{result.total_mutants})")
        print(f"  ⏱️  Execution Time: {result.execution_time:.1f}s")
        print(f"  💀 Killed Mutants: {result.killed_mutants}")
        print(f"  🧟 Survived Mutants: {result.survived_mutants}")
        
        if result.survived_mutants > 0:
            print(f"\n⚠️  {result.survived_mutants} mutants survived - test suite may need improvement")
        
        # Check if target score achieved
        if result.mutation_score >= args.target_score:
            print(f"✅ Target mutation score achieved: {result.mutation_score:.3f} >= {args.target_score:.3f}")
            return 0
        else:
            print(f"❌ Target mutation score NOT achieved: {result.mutation_score:.3f} < {args.target_score:.3f}")
            return 1
            
    except Exception as e:
        logging.error(f"Mutation testing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())