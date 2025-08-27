"""
Common Logging Utilities

Provides standardized logging setup patterns to eliminate redundant logging 
configuration throughout the codebase.

Usage:
    from common.logging import get_logger, setup_logging, LoggingContext

    # Get a logger (replaces logging.getLogger(__name__))
    logger = get_logger(__name__)

    # Setup consistent logging format
    setup_logging(level="INFO", format_style="detailed")

    # Temporary logging level change
    with LoggingContext(level="DEBUG"):
        logger.debug("This will be shown")
"""

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime

# Default logging formats
SIMPLE_FORMAT = "%(levelname)s - %(name)s - %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s"
PRODUCTION_FORMAT = "%(asctime)s [%(levelname)8s] %(name)s: %(message)s (%(filename)s:%(lineno)d)"

FORMAT_STYLES = {
    "simple": SIMPLE_FORMAT,
    "detailed": DETAILED_FORMAT,
    "production": PRODUCTION_FORMAT
}


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get logger with optional level setting.
    
    Replaces the common pattern: logger = logging.getLogger(__name__)
    
    Args:
        name: Logger name (typically __name__)
        level: Optional logging level to set
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if level:
        logger.setLevel(getattr(logging, level.upper()))
    
    return logger


def setup_logging(level: Union[str, int] = "INFO",
                 format_style: str = "detailed",
                 custom_format: Optional[str] = None,
                 log_file: Optional[str] = None,
                 console: bool = True) -> None:
    """
    Setup consistent logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_style: Format style ("simple", "detailed", "production") 
        custom_format: Custom format string (overrides format_style)
        log_file: Optional file to log to
        console: If True, also log to console
    """
    # Convert string level to logging constant
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Get format string
    if custom_format:
        format_str = custom_format
    else:
        format_str = FORMAT_STYLES.get(format_style, DETAILED_FORMAT)
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set root level
    root_logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(format_str)
    
    # Add console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


@contextmanager
def LoggingContext(level: Union[str, int], 
                  logger: Optional[logging.Logger] = None):
    """
    Temporarily change logging level within a context.
    
    Args:
        level: Temporary logging level
        logger: Specific logger to modify (default: root logger)
        
    Usage:
        with LoggingContext("DEBUG"):
            logger.debug("This will be shown even if root level is INFO")
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    target_logger = logger or logging.getLogger()
    original_level = target_logger.level
    
    target_logger.setLevel(level)
    try:
        yield
    finally:
        target_logger.setLevel(original_level)


def log_performance(logger: logging.Logger, 
                   operation: str,
                   start_time: float,
                   metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Log performance information for an operation.
    
    Args:
        logger: Logger to use
        operation: Name of the operation
        start_time: Start time from time.time()
        metadata: Optional additional information to log
    """
    duration_ms = (time.time() - start_time) * 1000
    
    log_msg = f"{operation} completed in {duration_ms:.2f}ms"
    
    if metadata:
        metadata_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
        log_msg += f" ({metadata_str})"
    
    logger.info(log_msg)


def log_error_with_context(logger: logging.Logger,
                          error: Exception,
                          operation: str,
                          context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log error with contextual information.
    
    Args:
        logger: Logger to use
        error: Exception that occurred
        operation: Operation that failed
        context: Optional context information
    """
    error_msg = f"{operation} failed: {type(error).__name__}: {error}"
    
    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        error_msg += f" (context: {context_str})"
    
    logger.error(error_msg, exc_info=True)


class PerformanceLogger:
    """Logger that tracks performance metrics across operations."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_times: Dict[str, float] = {}
        self.metrics: Dict[str, list] = {}
    
    def start(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        """End timing an operation and log result."""
        if operation not in self.start_times:
            self.logger.warning(f"No start time recorded for operation: {operation}")
            return 0.0
        
        start_time = self.start_times.pop(operation)
        duration_ms = (time.time() - start_time) * 1000
        
        # Track metrics
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration_ms)
        
        # Log performance
        log_performance(self.logger, operation, start_time, metadata)
        
        return duration_ms
    
    @contextmanager
    def measure(self, operation: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for measuring operation performance."""
        self.start(operation)
        try:
            yield
        finally:
            self.end(operation, metadata)
    
    def get_statistics(self, operation: str) -> Optional[Dict[str, float]]:
        """Get performance statistics for an operation."""
        if operation not in self.metrics:
            return None
        
        times = self.metrics[operation]
        return {
            'count': len(times),
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'latest': times[-1] if times else 0.0
        }


def create_timestamped_logger(name: str, 
                             log_dir: str = "./logs",
                             level: str = "INFO") -> logging.Logger:
    """
    Create logger that writes to timestamped file.
    
    Args:
        name: Logger name
        log_dir: Directory to write logs to
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = name.replace(".", "_").replace("/", "_")
    log_filename = f"{clean_name}_{timestamp}.log"
    
    log_path = Path(log_dir) / log_filename
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Add file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(PRODUCTION_FORMAT)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized, writing to: {log_path}")
    
    return logger


def silence_noisy_loggers(*logger_names: str, level: str = "WARNING") -> None:
    """
    Silence noisy third-party loggers.
    
    Args:
        *logger_names: Names of loggers to silence
        level: Level to set them to
    """
    level_int = getattr(logging, level.upper())
    
    for name in logger_names:
        logging.getLogger(name).setLevel(level_int)


def configure_ml_logging() -> None:
    """Configure logging for ML libraries to reduce noise."""
    # Common noisy ML/data science loggers
    noisy_loggers = [
        "urllib3.connectionpool",
        "requests.packages.urllib3.connectionpool", 
        "matplotlib.font_manager",
        "PIL.PngImagePlugin",
        "transformers.tokenization_utils",
        "transformers.file_utils",
        "sentence_transformers",
        "torch",
        "tensorflow"
    ]
    
    silence_noisy_loggers(*noisy_loggers, level="WARNING")


# Export commonly used functions
__all__ = [
    'get_logger',
    'setup_logging', 
    'LoggingContext',
    'log_performance',
    'log_error_with_context',
    'PerformanceLogger',
    'create_timestamped_logger',
    'silence_noisy_loggers',
    'configure_ml_logging'
]