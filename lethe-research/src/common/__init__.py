"""
Common utilities and shared components across the Lethe IR system.

This package contains reusable utilities that eliminate redundant code patterns
throughout the codebase, including:
- Configuration validation framework
- Timing measurement utilities  
- Logging setup patterns
- Path handling utilities

Usage:
    from common.validation import ConfigValidator, validate_range
    from common.timing import TimingContext, measure_latency
"""

# Re-export commonly used components
from .validation import (
    ValidationError,
    ConfigValidator, 
    MLConfigValidator,
    SystemConfigValidator,
    validate_range,
    validate_positive,
    validate_choice
)

from .timing import (
    TimingContext,
    SimpleTimingResult,
    LatencyTracker,
    measure_latency,
    track_latency,
    time_operation,
    replace_timing_pattern
)

from .paths import (
    ensure_dir,
    safe_path,
    validate_file_exists,
    validate_dir_exists,
    ensure_file_dir,
    find_files,
    safe_write_text,
    safe_read_text
)

from .logging import (
    get_logger,
    setup_logging,
    LoggingContext,
    log_performance,
    PerformanceLogger,
    configure_ml_logging
)

# Import advanced frameworks
from .model_manager import (
    ModelManager,
    ModelConfig,
    ModelInfo,
    ModelType,
    DeviceType,
    load_cross_encoder,
    load_sentence_transformer
)

from .data_persistence import (
    DataManager,
    DataConfig,
    DataFormat,
    load_json,
    save_json,
    load_jsonl,
    stream_jsonl
)

from .evaluation_framework import (
    EvaluationFramework,
    MetricConfig,
    QueryResult,
    SystemResult,
    ComparisonResult,
    evaluate_single_query,
    compare_two_systems
)

from .data_structures import (
    PerformanceMetrics,
    QueryInfo,
    DocumentInfo,
    RetrievalResult,
    EvaluationResult,
    AggregatedResults
)

__version__ = "1.0.0"

__all__ = [
    # Validation
    'ValidationError',
    'ConfigValidator',
    'MLConfigValidator', 
    'SystemConfigValidator',
    'validate_range',
    'validate_positive',
    'validate_choice',
    
    # Timing
    'TimingContext',
    'SimpleTimingResult', 
    'LatencyTracker',
    'measure_latency',
    'track_latency',
    'time_operation',
    'replace_timing_pattern',
    
    # Paths
    'ensure_dir',
    'safe_path',
    'validate_file_exists',
    'validate_dir_exists',
    'ensure_file_dir',
    'find_files',
    'safe_write_text',
    'safe_read_text',
    
    # Logging
    'get_logger',
    'setup_logging',
    'LoggingContext', 
    'log_performance',
    'PerformanceLogger',
    'configure_ml_logging'
]