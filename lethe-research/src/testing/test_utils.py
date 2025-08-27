#!/usr/bin/env python3
"""
Test Utilities for Lethe Research Project

Provides proper assertion-based utilities to replace return-based test patterns.
Includes helpers for import checking, file existence, Makefile targets, and common test patterns.

Features:
- Import availability checking with detailed error messages
- File existence validation with path resolution
- Makefile target validation
- Test dataclass creation utilities
- Synthetic data generation for testing
- Proper pytest assertions throughout
"""

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Type
from dataclasses import dataclass, field, make_dataclass
import numpy as np
import pandas as pd

# Optional pytest import
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    pytest = None


class TestAssertionError(AssertionError):
    """Custom assertion error for test utilities"""
    pass


def assert_imports_available(*modules: str) -> None:
    """
    Assert that all specified modules can be imported.
    
    Args:
        *modules: Module names to check for availability
        
    Raises:
        TestAssertionError: If any module cannot be imported
        
    Example:
        assert_imports_available('numpy', 'pandas', 'scipy')
    """
    missing_modules = []
    import_errors = {}
    
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            missing_modules.append(module_name)
            import_errors[module_name] = str(e)
    
    if missing_modules:
        error_details = []
        for module in missing_modules:
            error_details.append(f"  - {module}: {import_errors[module]}")
        
        raise TestAssertionError(
            f"Failed to import {len(missing_modules)} required modules:\n" +
            "\n".join(error_details)
        )


def assert_module_has_attributes(module_name: str, *attributes: str) -> None:
    """
    Assert that a module has all specified attributes.
    
    Args:
        module_name: Name of the module to check
        *attributes: Attribute names to verify
        
    Raises:
        TestAssertionError: If module cannot be imported or attributes are missing
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise TestAssertionError(f"Cannot import module '{module_name}': {e}")
    
    missing_attributes = []
    for attr in attributes:
        if not hasattr(module, attr):
            missing_attributes.append(attr)
    
    if missing_attributes:
        raise TestAssertionError(
            f"Module '{module_name}' missing attributes: {missing_attributes}"
        )


def assert_files_exist(*file_paths: Union[str, Path]) -> None:
    """
    Assert that all specified files exist.
    
    Args:
        *file_paths: File paths to check for existence
        
    Raises:
        TestAssertionError: If any file does not exist
        
    Example:
        assert_files_exist('config.yaml', 'src/main.py', Path('data/test.csv'))
    """
    missing_files = []
    
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            missing_files.append(str(path))
        elif not path.is_file():
            missing_files.append(f"{path} (exists but is not a file)")
    
    if missing_files:
        raise TestAssertionError(
            f"Missing files ({len(missing_files)}):\n" +
            "\n".join(f"  - {f}" for f in missing_files)
        )


def assert_directories_exist(*dir_paths: Union[str, Path]) -> None:
    """
    Assert that all specified directories exist.
    
    Args:
        *dir_paths: Directory paths to check for existence
        
    Raises:
        TestAssertionError: If any directory does not exist
    """
    missing_dirs = []
    
    for dir_path in dir_paths:
        path = Path(dir_path)
        if not path.exists():
            missing_dirs.append(str(path))
        elif not path.is_dir():
            missing_dirs.append(f"{path} (exists but is not a directory)")
    
    if missing_dirs:
        raise TestAssertionError(
            f"Missing directories ({len(missing_dirs)}):\n" +
            "\n".join(f"  - {d}" for d in missing_dirs)
        )


def assert_makefile_targets(*targets: str, makefile_path: Union[str, Path] = "Makefile") -> None:
    """
    Assert that all specified targets exist in a Makefile.
    
    Args:
        *targets: Target names to check for
        makefile_path: Path to the Makefile (default: "Makefile")
        
    Raises:
        TestAssertionError: If Makefile doesn't exist or targets are missing
    """
    makefile = Path(makefile_path)
    
    if not makefile.exists():
        raise TestAssertionError(f"Makefile not found at: {makefile}")
    
    try:
        content = makefile.read_text()
    except Exception as e:
        raise TestAssertionError(f"Cannot read Makefile: {e}")
    
    # Extract targets (lines that start with word characters followed by colon)
    import re
    target_pattern = re.compile(r'^([a-zA-Z0-9_-]+):', re.MULTILINE)
    found_targets = set(target_pattern.findall(content))
    
    missing_targets = []
    for target in targets:
        if target not in found_targets:
            missing_targets.append(target)
    
    if missing_targets:
        available_targets = sorted(found_targets)
        raise TestAssertionError(
            f"Missing Makefile targets: {missing_targets}\n" +
            f"Available targets: {available_targets}"
        )


def assert_makefile_target_runs(target: str, makefile_path: Union[str, Path] = "Makefile",
                               timeout: int = 30) -> str:
    """
    Assert that a Makefile target runs successfully.
    
    Args:
        target: Target name to run
        makefile_path: Path to the Makefile (default: "Makefile")
        timeout: Timeout in seconds (default: 30)
        
    Returns:
        Command output as string
        
    Raises:
        TestAssertionError: If target fails to run
    """
    makefile = Path(makefile_path)
    makefile_dir = makefile.parent
    
    try:
        result = subprocess.run(
            ["make", "-f", str(makefile.name), target],
            cwd=makefile_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode != 0:
            raise TestAssertionError(
                f"Makefile target '{target}' failed with return code {result.returncode}\n" +
                f"STDOUT:\n{result.stdout}\n" +
                f"STDERR:\n{result.stderr}"
            )
        
        return result.stdout
        
    except subprocess.TimeoutExpired:
        raise TestAssertionError(f"Makefile target '{target}' timed out after {timeout}s")
    except FileNotFoundError:
        raise TestAssertionError("'make' command not found - ensure GNU Make is installed")


def create_test_dataclass(class_name: str, **field_definitions) -> Type:
    """
    Create a test dataclass dynamically.
    
    Args:
        class_name: Name for the dataclass
        **field_definitions: Field definitions as name=type pairs
        
    Returns:
        Dynamically created dataclass type
        
    Example:
        TestData = create_test_dataclass(
            'TestData',
            name=str,
            value=int,
            items=(List[str], field(default_factory=list))
        )
    """
    fields = []
    for name, type_def in field_definitions.items():
        if isinstance(type_def, tuple) and len(type_def) == 2:
            # Handle (type, field()) pattern
            type_hint, field_def = type_def
            fields.append((name, type_hint, field_def))
        else:
            # Handle simple type
            fields.append((name, type_def))
    
    return make_dataclass(class_name, fields)


def generate_synthetic_data(schema: Dict[str, Any], n_samples: int = 100,
                          random_state: Optional[int] = 42) -> pd.DataFrame:
    """
    Generate synthetic data for testing based on a schema.
    
    Args:
        schema: Dictionary defining column types and constraints
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic data
        
    Example:
        schema = {
            'user_id': {'type': 'int', 'min': 1, 'max': 1000},
            'score': {'type': 'float', 'min': 0.0, 'max': 1.0},
            'category': {'type': 'categorical', 'values': ['A', 'B', 'C']},
            'description': {'type': 'string', 'length': 10}
        }
        df = generate_synthetic_data(schema, n_samples=50)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    data = {}
    
    for column, config in schema.items():
        col_type = config.get('type', 'float')
        
        if col_type == 'int':
            min_val = config.get('min', 0)
            max_val = config.get('max', 100)
            data[column] = np.random.randint(min_val, max_val + 1, size=n_samples)
            
        elif col_type == 'float':
            min_val = config.get('min', 0.0)
            max_val = config.get('max', 1.0)
            data[column] = np.random.uniform(min_val, max_val, size=n_samples)
            
        elif col_type == 'categorical':
            values = config.get('values', ['A', 'B', 'C'])
            data[column] = np.random.choice(values, size=n_samples)
            
        elif col_type == 'string':
            length = config.get('length', 5)
            chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            data[column] = [''.join(np.random.choice(list(chars), size=length)) 
                          for _ in range(n_samples)]
            
        elif col_type == 'bool':
            prob = config.get('prob', 0.5)
            data[column] = np.random.choice([True, False], size=n_samples, p=[prob, 1-prob])
            
        else:
            raise ValueError(f"Unsupported column type: {col_type}")
    
    return pd.DataFrame(data)


def assert_dataframe_structure(df: pd.DataFrame, expected_schema: Dict[str, Any]) -> None:
    """
    Assert that a DataFrame matches expected structure and constraints.
    
    Args:
        df: DataFrame to validate
        expected_schema: Schema with column types and constraints
        
    Raises:
        TestAssertionError: If DataFrame doesn't match schema
    """
    # Check columns exist
    missing_cols = set(expected_schema.keys()) - set(df.columns)
    if missing_cols:
        raise TestAssertionError(f"Missing columns: {missing_cols}")
    
    # Check column types and constraints
    for column, config in expected_schema.items():
        col_data = df[column]
        expected_type = config.get('type')
        
        if expected_type == 'int':
            if not pd.api.types.is_integer_dtype(col_data):
                raise TestAssertionError(f"Column '{column}' should be integer type")
                
            if 'min' in config and col_data.min() < config['min']:
                raise TestAssertionError(f"Column '{column}' has values below minimum {config['min']}")
                
            if 'max' in config and col_data.max() > config['max']:
                raise TestAssertionError(f"Column '{column}' has values above maximum {config['max']}")
        
        elif expected_type == 'float':
            if not pd.api.types.is_numeric_dtype(col_data):
                raise TestAssertionError(f"Column '{column}' should be numeric type")
        
        elif expected_type == 'categorical':
            expected_values = set(config.get('values', []))
            actual_values = set(col_data.unique())
            invalid_values = actual_values - expected_values
            if invalid_values:
                raise TestAssertionError(
                    f"Column '{column}' has invalid values: {invalid_values}"
                )


def assert_arrays_close(arr1: np.ndarray, arr2: np.ndarray, 
                       rtol: float = 1e-5, atol: float = 1e-8,
                       array_names: Tuple[str, str] = ("array1", "array2")) -> None:
    """
    Assert that two numpy arrays are element-wise close within tolerance.
    
    Args:
        arr1, arr2: Arrays to compare
        rtol: Relative tolerance
        atol: Absolute tolerance
        array_names: Names for the arrays in error messages
        
    Raises:
        TestAssertionError: If arrays are not close
    """
    try:
        np.testing.assert_allclose(arr1, arr2, rtol=rtol, atol=atol)
    except AssertionError as e:
        raise TestAssertionError(
            f"Arrays '{array_names[0]}' and '{array_names[1]}' are not close:\n{e}"
        )


def assert_deterministic_computation(func, *args, n_runs: int = 3, **kwargs) -> Any:
    """
    Assert that a function produces deterministic results across multiple runs.
    
    Args:
        func: Function to test
        *args: Arguments to pass to function
        n_runs: Number of times to run function
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        The result from the first run
        
    Raises:
        TestAssertionError: If results are not consistent
    """
    results = []
    
    for i in range(n_runs):
        # Reset random state if numpy is being used
        if hasattr(np.random, 'seed'):
            np.random.seed(42)
        
        result = func(*args, **kwargs)
        results.append(result)
    
    # Compare all results to the first one
    first_result = results[0]
    for i, result in enumerate(results[1:], 1):
        if isinstance(first_result, np.ndarray):
            if not np.allclose(first_result, result):
                raise TestAssertionError(
                    f"Function is not deterministic: run 0 vs run {i} differ"
                )
        elif isinstance(first_result, (list, tuple)):
            if first_result != result:
                raise TestAssertionError(
                    f"Function is not deterministic: run 0 vs run {i} differ"
                )
        else:
            if first_result != result:
                raise TestAssertionError(
                    f"Function is not deterministic: run 0 vs run {i} differ"
                )
    
    return first_result


def assert_performance_within_bounds(func, *args, max_time_seconds: float = 1.0,
                                   n_runs: int = 3, **kwargs) -> Tuple[Any, float]:
    """
    Assert that a function completes within time bounds.
    
    Args:
        func: Function to test
        *args: Arguments to pass to function
        max_time_seconds: Maximum allowed execution time
        n_runs: Number of runs to average timing over
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        Tuple of (result, average_time_seconds)
        
    Raises:
        TestAssertionError: If function takes too long
    """
    import time
    
    times = []
    result = None
    
    for _ in range(n_runs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = sum(times) / len(times)
    max_time = max(times)
    
    if max_time > max_time_seconds:
        raise TestAssertionError(
            f"Function exceeded time limit: {max_time:.4f}s > {max_time_seconds}s "
            f"(average: {avg_time:.4f}s)"
        )
    
    return result, avg_time


def assert_no_side_effects(func, *args, check_globals: bool = True, **kwargs) -> Any:
    """
    Assert that a function has no observable side effects.
    
    Args:
        func: Function to test
        *args: Arguments to pass to function
        check_globals: Whether to check for global variable modifications
        **kwargs: Keyword arguments to pass to function
        
    Returns:
        Function result
        
    Raises:
        TestAssertionError: If side effects are detected
    """
    # Capture initial state
    initial_globals = dict(func.__globals__) if check_globals else None
    
    # Run function
    result = func(*args, **kwargs)
    
    # Check for global modifications
    if check_globals and initial_globals is not None:
        current_globals = dict(func.__globals__)
        if initial_globals != current_globals:
            added = set(current_globals.keys()) - set(initial_globals.keys())
            modified = {k for k in initial_globals.keys() 
                       if k in current_globals and initial_globals[k] != current_globals[k]}
            removed = set(initial_globals.keys()) - set(current_globals.keys())
            
            side_effects = []
            if added:
                side_effects.append(f"Added globals: {added}")
            if modified:
                side_effects.append(f"Modified globals: {modified}")
            if removed:
                side_effects.append(f"Removed globals: {removed}")
            
            if side_effects:
                raise TestAssertionError(
                    f"Function has side effects:\n" + "\n".join(side_effects)
                )
    
    return result


# Convenience pytest fixtures that can be imported
# @pytest.fixture - Requires pytest installation
def synthetic_data_generator():
    """Fixture providing synthetic data generation utility"""
    return generate_synthetic_data


# @pytest.fixture - Requires pytest installation
def temp_test_files(tmp_path):
    """Fixture providing temporary test files"""
    def _create_files(*file_paths):
        created_files = []
        for file_path in file_paths:
            full_path = tmp_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(f"Test content for {file_path}")
            created_files.append(full_path)
        return created_files
    
    return _create_files


if __name__ == "__main__":
    # Simple self-test
    print("Testing test utilities...")
    
    # Test import checking
    try:
        assert_imports_available('os', 'sys', 'pathlib')
        print("✓ Import checking works")
    except TestAssertionError as e:
        print(f"✗ Import checking failed: {e}")
    
    # Test synthetic data generation
    try:
        schema = {
            'id': {'type': 'int', 'min': 1, 'max': 100},
            'score': {'type': 'float', 'min': 0.0, 'max': 1.0},
            'category': {'type': 'categorical', 'values': ['A', 'B', 'C']}
        }
        df = generate_synthetic_data(schema, n_samples=10)
        assert_dataframe_structure(df, schema)
        print("✓ Synthetic data generation works")
    except Exception as e:
        print(f"✗ Synthetic data generation failed: {e}")
    
    # Test deterministic computation
    try:
        def test_func(x):
            return x * 2
        
        result = assert_deterministic_computation(test_func, 5)
        assert result == 10
        print("✓ Deterministic computation checking works")
    except Exception as e:
        print(f"✗ Deterministic computation checking failed: {e}")
    
    print("Test utilities self-test complete!")