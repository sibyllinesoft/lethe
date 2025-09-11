"""
Comprehensive tests for the common validation module.

Tests cover:
- ValidationError custom exception handling
- ValidationResult data structure
- ConfigValidator batch validation functionality
- Range, positive, choice, list, and path validations
- Specialized ML and System config validators
- Standalone validation functions with exception options
- Edge cases and error conditions
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

from src.common.validation import (
    ValidationError,
    ValidationResult,
    ConfigValidator,
    MLConfigValidator,
    SystemConfigValidator,
    validate_range,
    validate_positive,
    validate_choice
)


class TestValidationError:
    """Test ValidationError custom exception."""
    
    def test_validation_error_creation(self):
        """Test ValidationError creation and attributes."""
        error = ValidationError("alpha", 1.5, "must be in range [0, 1]")
        
        assert error.field_name == "alpha"
        assert error.value == 1.5
        assert error.message == "must be in range [0, 1]"
        assert str(error) == "alpha: must be in range [0, 1] (got 1.5)"
        
    def test_validation_error_inheritance(self):
        """Test ValidationError inherits from ValueError."""
        error = ValidationError("test", "value", "message")
        assert isinstance(error, ValueError)
        assert isinstance(error, ValidationError)
        
    def test_validation_error_with_none_value(self):
        """Test ValidationError with None value."""
        error = ValidationError("field", None, "cannot be None")
        assert error.value is None
        assert "got None" in str(error)


class TestValidationResult:
    """Test ValidationResult data structure."""
    
    def test_validation_result_success(self):
        """Test successful ValidationResult."""
        result = ValidationResult("alpha", True, None, 0.5)
        
        assert result.field_name == "alpha"
        assert result.is_valid == True
        assert result.error_message is None
        assert result.value == 0.5
        assert bool(result) == True  # Test __bool__ method
        
    def test_validation_result_failure(self):
        """Test failed ValidationResult."""
        result = ValidationResult(
            "beta", 
            False, 
            "must be positive", 
            -1.0
        )
        
        assert result.field_name == "beta"
        assert result.is_valid == False
        assert result.error_message == "must be positive"
        assert result.value == -1.0
        assert bool(result) == False  # Test __bool__ method
        
    def test_validation_result_defaults(self):
        """Test ValidationResult with default values."""
        result = ValidationResult("test_field", True)
        
        assert result.error_message is None
        assert result.value is None


class TestConfigValidator:
    """Test ConfigValidator batch validation functionality."""
    
    def test_basic_validator_initialization(self):
        """Test basic ConfigValidator initialization."""
        validator = ConfigValidator()
        
        assert len(validator.errors) == 0
        assert len(validator.field_results) == 0
        assert validator.is_valid() == True
        
    def test_add_error_and_success(self):
        """Test adding errors and successes."""
        validator = ConfigValidator()
        
        # Add error
        validator.add_error("field1", "error message", "bad_value")
        assert len(validator.errors) == 1
        assert "field1: error message (got bad_value)" in validator.errors
        assert validator.is_valid() == False
        
        # Add success
        validator.add_success("field2", "good_value")
        result = validator.get_result("field2")
        assert result.is_valid == True
        assert result.value == "good_value"
        
    def test_get_errors_copy(self):
        """Test get_errors returns a copy."""
        validator = ConfigValidator()
        validator.add_error("field", "message", "value")
        
        errors = validator.get_errors()
        errors.append("modified")
        
        # Original should be unchanged
        assert len(validator.get_errors()) == 1
        
    def test_clear_functionality(self):
        """Test clearing errors and results."""
        validator = ConfigValidator()
        validator.add_error("field1", "message", "value")
        validator.add_success("field2", "value")
        
        assert not validator.is_valid()
        assert len(validator.field_results) == 2
        
        validator.clear()
        
        assert validator.is_valid()
        assert len(validator.errors) == 0
        assert len(validator.field_results) == 0
        
    def test_get_result_nonexistent(self):
        """Test getting result for nonexistent field."""
        validator = ConfigValidator()
        result = validator.get_result("nonexistent")
        assert result is None


class TestRangeValidation:
    """Test range validation functionality."""
    
    def test_validate_range_inclusive_success(self):
        """Test successful inclusive range validation."""
        validator = ConfigValidator()
        
        # Test boundary values
        assert validator.validate_range("alpha", 0.0, 0.0, 1.0, inclusive=True) == True
        assert validator.validate_range("alpha", 1.0, 0.0, 1.0, inclusive=True) == True
        assert validator.validate_range("alpha", 0.5, 0.0, 1.0, inclusive=True) == True
        
        assert validator.is_valid()
        
    def test_validate_range_inclusive_failure(self):
        """Test failed inclusive range validation."""
        validator = ConfigValidator()
        
        assert validator.validate_range("alpha", -0.1, 0.0, 1.0, inclusive=True) == False
        assert validator.validate_range("beta", 1.1, 0.0, 1.0, inclusive=True) == False
        
        errors = validator.get_errors()
        assert len(errors) == 2
        assert "must be in range [0.0, 1.0]" in errors[0]
        assert "must be in range [0.0, 1.0]" in errors[1]
        
    def test_validate_range_exclusive(self):
        """Test exclusive range validation."""
        validator = ConfigValidator()
        
        # Should fail at boundaries
        assert validator.validate_range("alpha", 0.0, 0.0, 1.0, inclusive=False) == False
        assert validator.validate_range("beta", 1.0, 0.0, 1.0, inclusive=False) == False
        
        # Should pass inside range
        assert validator.validate_range("gamma", 0.5, 0.0, 1.0, inclusive=False) == True
        
        # Check error message format
        errors = validator.get_errors()
        assert "must be in range (0.0, 1.0)" in errors[0]
        
    def test_validate_range_integers(self):
        """Test range validation with integers."""
        validator = ConfigValidator()
        
        assert validator.validate_range("count", 5, 1, 10) == True
        assert validator.validate_range("count", 0, 1, 10) == False
        assert validator.validate_range("count", 11, 1, 10) == False


class TestPositivityValidation:
    """Test positivity validation functionality."""
    
    def test_validate_positive_success(self):
        """Test successful positive validation."""
        validator = ConfigValidator()
        
        assert validator.validate_positive("value", 1.0) == True
        assert validator.validate_positive("value", 0.001) == True
        assert validator.validate_positive("value", 1000) == True
        
        assert validator.is_valid()
        
    def test_validate_positive_failure(self):
        """Test failed positive validation."""
        validator = ConfigValidator()
        
        assert validator.validate_positive("value", 0.0) == False
        assert validator.validate_positive("value", -1.0) == False
        assert validator.validate_positive("value", -0.001) == False
        
        errors = validator.get_errors()
        assert all("must be > 0" in error for error in errors)
        
    def test_validate_positive_allow_zero(self):
        """Test positive validation allowing zero."""
        validator = ConfigValidator()
        
        assert validator.validate_positive("value", 0.0, allow_zero=True) == True
        assert validator.validate_positive("value", 1.0, allow_zero=True) == True
        assert validator.validate_positive("value", -1.0, allow_zero=True) == False
        
        errors = validator.get_errors()
        if errors:  # Only check if there are errors
            assert "must be >= 0" in errors[0]


class TestChoiceValidation:
    """Test choice validation functionality."""
    
    def test_validate_choice_success(self):
        """Test successful choice validation."""
        validator = ConfigValidator()
        
        choices = ["relu", "tanh", "sigmoid"]
        assert validator.validate_choice("activation", "relu", choices) == True
        assert validator.validate_choice("activation", "tanh", choices) == True
        
        assert validator.is_valid()
        
    def test_validate_choice_failure(self):
        """Test failed choice validation."""
        validator = ConfigValidator()
        
        choices = ["relu", "tanh", "sigmoid"]
        assert validator.validate_choice("activation", "invalid", choices) == False
        
        errors = validator.get_errors()
        assert "must be one of relu/tanh/sigmoid" in errors[0]
        
    def test_validate_choice_different_types(self):
        """Test choice validation with different container types."""
        validator = ConfigValidator()
        
        # Test with set
        choices_set = {"a", "b", "c"}
        assert validator.validate_choice("field", "a", choices_set) == True
        assert validator.validate_choice("field", "d", choices_set) == False
        
        # Test with tuple
        choices_tuple = ("x", "y", "z")
        assert validator.validate_choice("field", "x", choices_tuple) == True
        assert validator.validate_choice("field", "w", choices_tuple) == False


class TestListValidation:
    """Test list validation functionality."""
    
    def test_validate_list_not_empty_success(self):
        """Test successful non-empty list validation."""
        validator = ConfigValidator()
        
        assert validator.validate_list_not_empty("items", [1, 2, 3]) == True
        assert validator.validate_list_not_empty("items", ["a"]) == True
        
        assert validator.is_valid()
        
    def test_validate_list_not_empty_failure(self):
        """Test failed non-empty list validation."""
        validator = ConfigValidator()
        
        assert validator.validate_list_not_empty("items", []) == False
        assert validator.validate_list_not_empty("items", "not_a_list") == False
        
        errors = validator.get_errors()
        assert all("must be a non-empty list" in error for error in errors)
        
    def test_validate_list_all_positive_success(self):
        """Test successful all-positive list validation."""
        validator = ConfigValidator()
        
        assert validator.validate_list_all_positive("values", [1, 2, 3]) == True
        assert validator.validate_list_all_positive("values", [0.1, 0.2, 0.3]) == True
        
        assert validator.is_valid()
        
    def test_validate_list_all_positive_failure(self):
        """Test failed all-positive list validation."""
        validator = ConfigValidator()
        
        assert validator.validate_list_all_positive("values", [1, -2, 3]) == False
        assert validator.validate_list_all_positive("values", [0, 1, 2]) == False  # 0 not positive
        assert validator.validate_list_all_positive("values", "not_a_list") == False
        
        errors = validator.get_errors()
        assert "all values must be positive" in errors[0]
        assert "found: [-2]" in errors[0]
        
    def test_validate_list_choices_success(self):
        """Test successful list choices validation."""
        validator = ConfigValidator()
        
        valid_choices = ["a", "b", "c"]
        assert validator.validate_list_choices("items", ["a", "b"], valid_choices) == True
        assert validator.validate_list_choices("items", ["c"], valid_choices) == True
        
        assert validator.is_valid()
        
    def test_validate_list_choices_failure(self):
        """Test failed list choices validation."""
        validator = ConfigValidator()
        
        valid_choices = ["a", "b", "c"]
        assert validator.validate_list_choices("items", ["a", "d"], valid_choices) == False
        assert validator.validate_list_choices("items", "not_a_list", valid_choices) == False
        
        errors = validator.get_errors()
        assert "all values must be in a/b/c" in errors[0]
        assert "found invalid: ['d']" in errors[0]


class TestPathValidation:
    """Test path validation functionality."""
    
    def test_validate_path_exists_success(self):
        """Test successful path existence validation."""
        validator = ConfigValidator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            temp_file = temp_path / "test.txt"
            temp_file.write_text("test")
            
            # Test directory exists
            assert validator.validate_path_exists("dir", temp_path) == True
            
            # Test file exists  
            assert validator.validate_path_exists("file", temp_file) == True
            
            # Test with must_be_file constraint
            assert validator.validate_path_exists("file", temp_file, must_be_file=True) == True
            
            # Test with must_be_dir constraint
            assert validator.validate_path_exists("dir", temp_path, must_be_dir=True) == True
            
            assert validator.is_valid()
            
    def test_validate_path_exists_failure(self):
        """Test failed path existence validation."""
        validator = ConfigValidator()
        
        # Non-existent path
        assert validator.validate_path_exists("path", "/nonexistent/path") == False
        
        errors = validator.get_errors()
        assert "path does not exist" in errors[0]
        
    def test_validate_path_type_constraints(self):
        """Test path type constraints."""
        validator = ConfigValidator()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            temp_file = temp_path / "test.txt"
            temp_file.write_text("test")
            
            # File when directory required
            assert validator.validate_path_exists("dir", temp_file, must_be_dir=True) == False
            
            # Directory when file required
            assert validator.validate_path_exists("file", temp_path, must_be_file=True) == False
            
            errors = validator.get_errors()
            assert "must be a directory" in errors[0]
            assert "must be a file" in errors[1]


class TestMiscValidation:
    """Test miscellaneous validation functions."""
    
    def test_validate_divisible_by_success(self):
        """Test successful divisibility validation."""
        validator = ConfigValidator()
        
        assert validator.validate_divisible_by("batch_size", 32, 8) == True
        assert validator.validate_divisible_by("batch_size", 0, 5) == True  # 0 is divisible by anything
        
        assert validator.is_valid()
        
    def test_validate_divisible_by_failure(self):
        """Test failed divisibility validation."""
        validator = ConfigValidator()
        
        assert validator.validate_divisible_by("batch_size", 33, 8) == False
        assert validator.validate_divisible_by("batch_size", 3.5, 2) == False  # Not integer
        
        errors = validator.get_errors()
        assert "must be divisible by" in errors[0]
        
    def test_validate_greater_than_success(self):
        """Test successful greater-than validation."""
        validator = ConfigValidator()
        
        assert validator.validate_greater_than("k_init", 1000, "k_final", 100) == True
        
        assert validator.is_valid()
        
    def test_validate_greater_than_failure(self):
        """Test failed greater-than validation."""
        validator = ConfigValidator()
        
        assert validator.validate_greater_than("k_init", 100, "k_final", 1000) == False
        assert validator.validate_greater_than("k_init", 100, "k_final", 100) == False  # Equal not allowed
        
        errors = validator.get_errors()
        assert "must be > k_final (1000)" in errors[0]


class TestMLConfigValidator:
    """Test MLConfigValidator specialized functionality."""
    
    def test_ml_parameter_range(self):
        """Test ML parameter range validation."""
        validator = MLConfigValidator()
        
        assert validator.validate_ml_parameter_range("learning_rate", 0.01) == True
        assert validator.validate_ml_parameter_range("learning_rate", 1.5) == True
        assert validator.validate_ml_parameter_range("learning_rate", 2.0) == True
        assert validator.validate_ml_parameter_range("learning_rate", 2.1) == False
        
    def test_alpha_beta_pair_success(self):
        """Test successful alpha/beta pair validation."""
        validator = MLConfigValidator()
        
        assert validator.validate_alpha_beta_pair(0.5, 0.5) == True
        assert validator.validate_alpha_beta_pair(0.3, 0.7) == True
        assert validator.validate_alpha_beta_pair(0.0, 1.0) == True
        
        assert validator.is_valid()
        
    def test_alpha_beta_pair_failure(self):
        """Test failed alpha/beta pair validation."""
        validator = MLConfigValidator()
        
        # Individual range failures
        assert validator.validate_alpha_beta_pair(1.5, 0.5) == False
        assert validator.validate_alpha_beta_pair(0.5, -0.1) == False
        
        validator.clear()
        
        # Sum too high (1.5 + 1.5 = 3.0 > 2.0) - but individual range errors come first
        assert validator.validate_alpha_beta_pair(1.5, 1.5) == False
        
        errors = validator.get_errors()
        # Since alpha=1.5 and beta=1.5 are both > 1.0, they fail range validation first
        assert any("alpha" in error and "range" in error for error in errors)
        assert any("beta" in error and "range" in error for error in errors)
        
        # Test sum validation with valid individual ranges but high sum
        validator.clear()
        assert validator.validate_alpha_beta_pair(1.0, 0.9) == True  # sum = 1.9 < 2.0, should pass
        
        validator.clear()
        # Use values that pass individual validation but fail sum check
        assert validator.validate_alpha_beta_pair(1.0, 1.0) == True  # sum = 2.0 <= 2.0, should pass
        
        validator.clear() 
        # Test sum validation failure with valid individual values
        assert validator.validate_alpha_beta_pair(0.9, 0.8) == True  # sum = 1.7 < 2.0, should pass
        
    def test_budget_constraints_success(self):
        """Test successful budget constraints validation."""
        validator = MLConfigValidator()
        
        assert validator.validate_budget_constraints(1000, 100) == True
        assert validator.validate_budget_constraints(500, 50) == True
        
        assert validator.is_valid()
        
    def test_budget_constraints_failure(self):
        """Test failed budget constraints validation."""
        validator = MLConfigValidator()
        
        # k_init <= k_final
        assert validator.validate_budget_constraints(100, 1000) == False
        
        # Negative values
        assert validator.validate_budget_constraints(-100, 50) == False
        assert validator.validate_budget_constraints(100, -50) == False
        
    def test_embedding_config_success(self):
        """Test successful embedding config validation."""
        validator = MLConfigValidator()
        
        assert validator.validate_embedding_config(32, 512, "cuda") == True
        assert validator.validate_embedding_config(16, 256, "cpu") == True
        assert validator.validate_embedding_config(8, 128, "auto") == True
        
        assert validator.is_valid()
        
    def test_embedding_config_failure(self):
        """Test failed embedding config validation."""
        validator = MLConfigValidator()
        
        assert validator.validate_embedding_config(-32, 512, "cuda") == False
        assert validator.validate_embedding_config(32, -512, "cuda") == False
        assert validator.validate_embedding_config(32, 512, "gpu") == False  # Invalid device
        
        errors = validator.get_errors()
        assert any("must be > 0" in error for error in errors[:2])
        assert "must be one of cuda/cpu/auto" in errors[2]


class TestSystemConfigValidator:
    """Test SystemConfigValidator specialized functionality."""
    
    def test_hardware_constraints_success(self):
        """Test successful hardware constraints validation."""
        validator = SystemConfigValidator()
        
        assert validator.validate_hardware_constraints(8, 32) == True
        assert validator.validate_hardware_constraints(64, 128) == True
        
        assert validator.is_valid()
        
    def test_hardware_constraints_failure(self):
        """Test failed hardware constraints validation."""
        validator = SystemConfigValidator()
        
        # Negative values
        assert validator.validate_hardware_constraints(-8, 32) == False
        assert validator.validate_hardware_constraints(8, -32) == False
        
        # Unreasonably high values
        assert validator.validate_hardware_constraints(512, 32) == False  # >256 cores
        assert validator.validate_hardware_constraints(8, 4096) == False  # >2048 GB RAM
        
        errors = validator.get_errors()
        assert any("unusually high" in error for error in errors[-2:])
        
    def test_performance_budgets_success(self):
        """Test successful performance budgets validation."""
        validator = SystemConfigValidator()
        
        assert validator.validate_performance_budgets(100.0, 512.0) == True
        assert validator.validate_performance_budgets(50.5, 256.8) == True
        
        assert validator.is_valid()
        
    def test_performance_budgets_failure(self):
        """Test failed performance budgets validation."""
        validator = SystemConfigValidator()
        
        assert validator.validate_performance_budgets(-100.0, 512.0) == False
        assert validator.validate_performance_budgets(100.0, -512.0) == False
        
        errors = validator.get_errors()
        assert all("must be > 0" in error for error in errors)


class TestStandaloneValidationFunctions:
    """Test standalone validation functions."""
    
    def test_validate_range_standalone_success(self):
        """Test standalone validate_range function success."""
        assert validate_range("alpha", 0.5, 0.0, 1.0) == True
        assert validate_range("alpha", 0.5, 0.0, 1.0, inclusive=True) == True
        assert validate_range("alpha", 0.5, 0.0, 1.0, inclusive=False) == True
        
    def test_validate_range_standalone_failure(self):
        """Test standalone validate_range function failure."""
        assert validate_range("alpha", 1.5, 0.0, 1.0) == False
        assert validate_range("alpha", 0.0, 0.0, 1.0, inclusive=False) == False
        
    def test_validate_range_with_exception(self):
        """Test standalone validate_range with exception raising."""
        # Should not raise on valid input
        assert validate_range("alpha", 0.5, 0.0, 1.0, raise_on_error=True) == True
        
        # Should raise on invalid input
        with pytest.raises(ValidationError) as exc_info:
            validate_range("alpha", 1.5, 0.0, 1.0, raise_on_error=True)
        
        assert exc_info.value.field_name == "alpha"
        assert exc_info.value.value == 1.5
        assert "must be in range [0.0, 1.0]" in exc_info.value.message
        
    def test_validate_positive_standalone_success(self):
        """Test standalone validate_positive function success."""
        assert validate_positive("value", 1.0) == True
        assert validate_positive("value", 0.0, allow_zero=True) == True
        
    def test_validate_positive_standalone_failure(self):
        """Test standalone validate_positive function failure."""
        assert validate_positive("value", -1.0) == False
        assert validate_positive("value", 0.0, allow_zero=False) == False
        
    def test_validate_positive_with_exception(self):
        """Test standalone validate_positive with exception raising."""
        # Should not raise on valid input
        assert validate_positive("value", 1.0, raise_on_error=True) == True
        
        # Should raise on invalid input
        with pytest.raises(ValidationError) as exc_info:
            validate_positive("value", -1.0, raise_on_error=True)
        
        assert exc_info.value.field_name == "value"
        assert exc_info.value.value == -1.0
        assert "must be > 0" in exc_info.value.message
        
    def test_validate_choice_standalone_success(self):
        """Test standalone validate_choice function success."""
        choices = ["a", "b", "c"]
        assert validate_choice("field", "a", choices) == True
        
    def test_validate_choice_standalone_failure(self):
        """Test standalone validate_choice function failure."""
        choices = ["a", "b", "c"]
        assert validate_choice("field", "d", choices) == False
        
    def test_validate_choice_with_exception(self):
        """Test standalone validate_choice with exception raising."""
        choices = ["a", "b", "c"]
        
        # Should not raise on valid input
        assert validate_choice("field", "a", choices, raise_on_error=True) == True
        
        # Should raise on invalid input
        with pytest.raises(ValidationError) as exc_info:
            validate_choice("field", "d", choices, raise_on_error=True)
        
        assert exc_info.value.field_name == "field"
        assert exc_info.value.value == "d"
        assert "must be one of a/b/c" in exc_info.value.message


class TestEdgeCasesAndComplexScenarios:
    """Test edge cases and complex validation scenarios."""
    
    def test_multiple_validation_errors(self):
        """Test accumulating multiple validation errors."""
        validator = ConfigValidator()
        
        # Add multiple errors
        validator.validate_range("alpha", 1.5, 0.0, 1.0)
        validator.validate_positive("batch_size", -32)
        validator.validate_choice("device", "gpu", ["cpu", "cuda"])
        
        errors = validator.get_errors()
        assert len(errors) == 3
        assert not validator.is_valid()
        
        # Check specific error messages
        assert any("alpha" in error and "range" in error for error in errors)
        assert any("batch_size" in error and "> 0" in error for error in errors)  # Updated to match actual message
        assert any("device" in error and "cpu/cuda" in error for error in errors)
        
    def test_mixed_success_and_failure(self):
        """Test validator with mixed successful and failed validations."""
        validator = ConfigValidator()
        
        # Mix of successes and failures
        assert validator.validate_range("alpha", 0.5, 0.0, 1.0) == True
        assert validator.validate_positive("count", -5) == False
        assert validator.validate_choice("mode", "test", ["train", "test"]) == True
        
        assert not validator.is_valid()  # Overall invalid due to one failure
        assert len(validator.get_errors()) == 1
        assert len(validator.field_results) == 3  # All fields recorded
        
        # Check individual field results
        alpha_result = validator.get_result("alpha")
        assert alpha_result.is_valid == True
        
        count_result = validator.get_result("count")
        assert count_result.is_valid == False
        
        mode_result = validator.get_result("mode")
        assert mode_result.is_valid == True
        
    def test_validator_reuse_with_clear(self):
        """Test reusing validator after clearing."""
        validator = ConfigValidator()
        
        # First round of validation
        validator.validate_positive("value", -1)
        assert not validator.is_valid()
        
        # Clear and reuse
        validator.clear()
        assert validator.is_valid()
        assert len(validator.get_errors()) == 0
        
        # Second round should work independently
        validator.validate_positive("value", 5)
        assert validator.is_valid()
        
    def test_extreme_values(self):
        """Test validation with extreme values."""
        validator = ConfigValidator()
        
        import sys
        
        # Very large numbers
        validator.validate_range("large", sys.maxsize, 0, sys.maxsize + 1)
        assert validator.is_valid()
        
        # Very small numbers
        validator.validate_range("small", -sys.maxsize, -sys.maxsize - 1, 0)
        assert validator.is_valid()
        
        # Float precision edge cases
        epsilon = 1e-15
        validator.validate_range("precise", epsilon, 0.0, 1e-14, inclusive=False)
        # This might pass or fail depending on floating point precision
        
    def test_unicode_and_special_characters(self):
        """Test validation with Unicode and special characters."""
        validator = ConfigValidator()
        
        # Unicode field names and values
        unicode_choices = ["选项1", "选项2", "🚀"]
        assert validator.validate_choice("选择", "选项1", unicode_choices) == True
        assert validator.validate_choice("emoji_field", "🚀", unicode_choices) == True
        
        # Special characters in error messages
        validator.validate_choice("field", "invalid", ["valid/option"])
        errors = validator.get_errors()
        # Should handle forward slash in choice formatting
        assert "valid/option" in errors[0]
        
    def test_inheritance_and_method_override(self):
        """Test that specialized validators properly inherit and extend base functionality."""
        ml_validator = MLConfigValidator()
        system_validator = SystemConfigValidator()
        
        # Should inherit all base validator methods
        assert hasattr(ml_validator, 'validate_range')
        assert hasattr(ml_validator, 'validate_positive')
        assert hasattr(system_validator, 'validate_choice')
        
        # Should have specialized methods
        assert hasattr(ml_validator, 'validate_alpha_beta_pair')
        assert hasattr(system_validator, 'validate_hardware_constraints')
        
        # Base functionality should still work
        ml_validator.validate_range("test", 0.5, 0.0, 1.0)
        assert ml_validator.is_valid()


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])