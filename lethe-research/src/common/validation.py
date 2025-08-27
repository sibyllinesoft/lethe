"""
Common Configuration Validation Framework

Provides standardized validation patterns to eliminate redundant validation code
across the codebase. Supports both method-based and exception-based validation.

Usage:
    from common.validation import ConfigValidator, ValidationError, validate_range

    # Method-based validation (returns List[str])
    validator = ConfigValidator()
    validator.validate_range("alpha", alpha, 0.0, 1.0)
    validator.validate_positive("batch_size", batch_size)
    errors = validator.get_errors()

    # Exception-based validation (raises ValidationError) 
    validate_range("alpha", alpha, 0.0, 1.0, raise_on_error=True)
    validate_positive("batch_size", batch_size, raise_on_error=True)
"""

from typing import List, Any, Optional, Union, Callable, Dict, Set
from dataclasses import dataclass
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Exception raised for configuration validation errors."""
    
    def __init__(self, field_name: str, value: Any, message: str):
        self.field_name = field_name
        self.value = value
        self.message = message
        super().__init__(f"{field_name}: {message} (got {value})")


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    
    field_name: str
    is_valid: bool
    error_message: Optional[str] = None
    value: Optional[Any] = None
    
    def __bool__(self) -> bool:
        return self.is_valid


class ConfigValidator:
    """Accumulates validation errors for batch validation."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.field_results: Dict[str, ValidationResult] = {}
    
    def add_error(self, field_name: str, message: str, value: Any = None):
        """Add a validation error."""
        error_msg = f"{field_name}: {message}" + (f" (got {value})" if value is not None else "")
        self.errors.append(error_msg)
        self.field_results[field_name] = ValidationResult(field_name, False, message, value)
        
    def add_success(self, field_name: str, value: Any):
        """Add a successful validation."""
        self.field_results[field_name] = ValidationResult(field_name, True, None, value)
    
    def get_errors(self) -> List[str]:
        """Get all accumulated errors."""
        return self.errors.copy()
    
    def is_valid(self) -> bool:
        """Check if all validations passed."""
        return len(self.errors) == 0
    
    def get_result(self, field_name: str) -> Optional[ValidationResult]:
        """Get result for a specific field."""
        return self.field_results.get(field_name)
    
    def clear(self):
        """Clear all errors and results."""
        self.errors.clear()
        self.field_results.clear()
    
    # Range validation
    def validate_range(self, field_name: str, value: Union[int, float], 
                      min_val: Union[int, float], max_val: Union[int, float],
                      inclusive: bool = True) -> bool:
        """Validate that value is within range."""
        if inclusive:
            is_valid = min_val <= value <= max_val
            range_desc = f"[{min_val}, {max_val}]"
        else:
            is_valid = min_val < value < max_val
            range_desc = f"({min_val}, {max_val})"
        
        if is_valid:
            self.add_success(field_name, value)
            return True
        else:
            self.add_error(field_name, f"must be in range {range_desc}", value)
            return False
    
    # Positivity validation
    def validate_positive(self, field_name: str, value: Union[int, float], 
                         allow_zero: bool = False) -> bool:
        """Validate that value is positive."""
        if allow_zero:
            is_valid = value >= 0
            constraint = ">= 0"
        else:
            is_valid = value > 0
            constraint = "> 0"
        
        if is_valid:
            self.add_success(field_name, value)
            return True
        else:
            self.add_error(field_name, f"must be {constraint}", value)
            return False
    
    # Choice validation
    def validate_choice(self, field_name: str, value: Any, 
                       valid_choices: Union[List, Set, tuple]) -> bool:
        """Validate that value is in allowed choices."""
        is_valid = value in valid_choices
        
        if is_valid:
            self.add_success(field_name, value)
            return True
        else:
            choices_str = "/".join(str(c) for c in valid_choices)
            self.add_error(field_name, f"must be one of {choices_str}", value)
            return False
    
    # List validation
    def validate_list_not_empty(self, field_name: str, value: List) -> bool:
        """Validate that list is not empty."""
        is_valid = isinstance(value, list) and len(value) > 0
        
        if is_valid:
            self.add_success(field_name, value)
            return True
        else:
            self.add_error(field_name, "must be a non-empty list", value)
            return False
    
    def validate_list_all_positive(self, field_name: str, values: List[Union[int, float]]) -> bool:
        """Validate that all values in list are positive."""
        if not isinstance(values, list):
            self.add_error(field_name, "must be a list", values)
            return False
        
        negative_values = [v for v in values if v <= 0]
        if not negative_values:
            self.add_success(field_name, values)
            return True
        else:
            self.add_error(field_name, f"all values must be positive, found: {negative_values}")
            return False
    
    def validate_list_choices(self, field_name: str, values: List, 
                             valid_choices: Union[List, Set, tuple]) -> bool:
        """Validate that all values in list are valid choices."""
        if not isinstance(values, list):
            self.add_error(field_name, "must be a list", values)
            return False
        
        invalid_values = [v for v in values if v not in valid_choices]
        if not invalid_values:
            self.add_success(field_name, values)
            return True
        else:
            choices_str = "/".join(str(c) for c in valid_choices)
            self.add_error(field_name, f"all values must be in {choices_str}, found invalid: {invalid_values}")
            return False
    
    # Path validation
    def validate_path_exists(self, field_name: str, path: Union[str, Path], 
                           must_be_file: bool = False, must_be_dir: bool = False) -> bool:
        """Validate that path exists."""
        path_obj = Path(path)
        
        if not path_obj.exists():
            self.add_error(field_name, f"path does not exist: {path}")
            return False
        
        if must_be_file and not path_obj.is_file():
            self.add_error(field_name, f"must be a file: {path}")
            return False
            
        if must_be_dir and not path_obj.is_dir():
            self.add_error(field_name, f"must be a directory: {path}")
            return False
        
        self.add_success(field_name, path)
        return True
    
    # Divisibility validation
    def validate_divisible_by(self, field_name: str, value: int, divisor: int) -> bool:
        """Validate that value is divisible by divisor."""
        is_valid = isinstance(value, int) and value % divisor == 0
        
        if is_valid:
            self.add_success(field_name, value)
            return True
        else:
            self.add_error(field_name, f"must be divisible by {divisor}", value)
            return False
    
    # Comparison validation  
    def validate_greater_than(self, field_name: str, value: Union[int, float], 
                             other_field: str, other_value: Union[int, float]) -> bool:
        """Validate that value > other_value."""
        is_valid = value > other_value
        
        if is_valid:
            self.add_success(field_name, value)
            return True
        else:
            self.add_error(field_name, f"must be > {other_field} ({other_value})", value)
            return False


# Standalone validation functions for exception-based validation
def validate_range(field_name: str, value: Union[int, float], 
                  min_val: Union[int, float], max_val: Union[int, float],
                  inclusive: bool = True, raise_on_error: bool = False) -> bool:
    """Validate range with optional exception raising."""
    if inclusive:
        is_valid = min_val <= value <= max_val
        range_desc = f"[{min_val}, {max_val}]"
    else:
        is_valid = min_val < value < max_val
        range_desc = f"({min_val}, {max_val})"
    
    if not is_valid and raise_on_error:
        raise ValidationError(field_name, value, f"must be in range {range_desc}")
    
    return is_valid


def validate_positive(field_name: str, value: Union[int, float], 
                     allow_zero: bool = False, raise_on_error: bool = False) -> bool:
    """Validate positive with optional exception raising."""
    if allow_zero:
        is_valid = value >= 0
        constraint = ">= 0"
    else:
        is_valid = value > 0
        constraint = "> 0"
    
    if not is_valid and raise_on_error:
        raise ValidationError(field_name, value, f"must be {constraint}")
    
    return is_valid


def validate_choice(field_name: str, value: Any, 
                   valid_choices: Union[List, Set, tuple],
                   raise_on_error: bool = False) -> bool:
    """Validate choice with optional exception raising."""
    is_valid = value in valid_choices
    
    if not is_valid and raise_on_error:
        choices_str = "/".join(str(c) for c in valid_choices)
        raise ValidationError(field_name, value, f"must be one of {choices_str}")
    
    return is_valid


# Common validation patterns for ML/IR systems
class MLConfigValidator(ConfigValidator):
    """Specialized validator for ML configuration patterns."""
    
    def validate_ml_parameter_range(self, field_name: str, value: float) -> bool:
        """Validate typical ML parameter ranges (0.0 to 2.0)."""
        return self.validate_range(field_name, value, 0.0, 2.0)
    
    def validate_alpha_beta_pair(self, alpha: float, beta: float) -> bool:
        """Validate alpha/beta parameter pair for fusion."""
        valid = True
        valid &= self.validate_range("alpha", alpha, 0.0, 1.0)
        valid &= self.validate_range("beta", beta, 0.0, 1.0)
        
        # Additional constraint: alpha + beta should be reasonable
        if valid and (alpha + beta) > 2.0:
            self.add_error("alpha+beta", f"sum should be <= 2.0", alpha + beta)
            valid = False
        
        return valid
    
    def validate_budget_constraints(self, k_init: int, k_final: int) -> bool:
        """Validate retrieval budget constraints."""
        valid = True
        valid &= self.validate_positive("k_init", k_init)
        valid &= self.validate_positive("k_final", k_final)
        valid &= self.validate_greater_than("k_init", k_init, "k_final", k_final)
        
        return valid
    
    def validate_embedding_config(self, batch_size: int, max_length: int, device: str) -> bool:
        """Validate embedding configuration."""
        valid = True
        valid &= self.validate_positive("batch_size", batch_size)
        valid &= self.validate_positive("max_length", max_length)
        valid &= self.validate_choice("device", device, ["cuda", "cpu", "auto"])
        
        return valid


class SystemConfigValidator(ConfigValidator):
    """Specialized validator for system configuration patterns."""
    
    def validate_hardware_constraints(self, cpu_cores: int, memory_gb: int) -> bool:
        """Validate hardware resource constraints."""
        valid = True
        valid &= self.validate_positive("cpu_cores", cpu_cores)
        valid &= self.validate_positive("memory_gb", memory_gb)
        
        # Sanity checks for reasonable hardware
        if cpu_cores > 256:
            self.add_error("cpu_cores", "unusually high, check configuration", cpu_cores)
            valid = False
            
        if memory_gb > 2048:  # 2TB RAM
            self.add_error("memory_gb", "unusually high, check configuration", memory_gb)
            valid = False
            
        return valid
    
    def validate_performance_budgets(self, latency_ms: float, memory_mb: float) -> bool:
        """Validate performance budget constraints."""
        valid = True
        valid &= self.validate_positive("latency_ms", latency_ms)
        valid &= self.validate_positive("memory_mb", memory_mb)
        
        return valid


# Export commonly used validators
__all__ = [
    'ValidationError',
    'ValidationResult', 
    'ConfigValidator',
    'MLConfigValidator',
    'SystemConfigValidator',
    'validate_range',
    'validate_positive', 
    'validate_choice'
]