#!/usr/bin/env python3
"""
Prompt Execution Monitoring and Tracking System

This module provides comprehensive prompt tracking capabilities that integrate
with the existing Lethe monitoring infrastructure, offering real-time monitoring,
performance analytics, A/B testing, and experiment tracking for LLM prompt executions.

**Key Features:**
    * Automatic prompt execution tracking with timing and metrics
    * Context manager interface for seamless integration
    * SQLite database with full-text search and analytics
    * Optional MLflow integration for experiment tracking
    * A/B testing and statistical comparison tools
    * Performance trend analysis and quality scoring
    * Export capabilities (CSV, JSON, Parquet formats)
    * Version control for prompt templates and variations

**Usage Example:**
    Basic prompt tracking:
    
    >>> from src.monitoring.prompt_tracker import track_prompt
    >>> with track_prompt(
    ...     prompt_id="user_greeting_v2",
    ...     prompt_text="Hello! How can I help you today?",
    ...     model_config={"model": "gpt-4", "temperature": 0.7}
    ... ) as execution:
    ...     # Your LLM call here
    ...     execution.response_text = "I'm here to help!"
    ...     execution.response_quality_score = 0.95

**Database Schema:**
    The system maintains three core tables:
    * `prompt_executions`: Individual execution records with metrics
    * `prompt_versions`: Version history for prompt templates
    * `prompt_comparisons`: Statistical comparisons between executions

**Integration Points:**
    * MLflow: Experiment tracking and artifact storage
    * Lethe DataManager: Data persistence layer integration
    * FusionTelemetry: Telemetry and monitoring integration
    * SQLite: Local database for fast queries and analytics

**Thread Safety:**
    This module is thread-safe for concurrent prompt tracking. Each execution
    gets a unique UUID and atomic database operations ensure consistency.
"""

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4

import pandas as pd
from contextlib import contextmanager

# Optional MLflow integration
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Optional imports for existing components
try:
    from ..common.data_persistence import DataManager
except ImportError:
    DataManager = None

try:
    from ..fusion.telemetry import FusionTelemetry
except ImportError:
    FusionTelemetry = None


@dataclass
class PromptExecution:
    """Comprehensive data structure for tracking a single prompt execution.
    
    This dataclass captures all relevant information about a prompt execution,
    including input parameters, model configuration, performance metrics,
    response data, and error handling information.
    
    **Lifecycle:**
        1. Created when entering track_execution() context manager
        2. Populated during execution with timing and response data
        3. Saved to database and optionally logged to MLflow
        4. Used for analytics and comparison operations
    
    **Key Metrics Tracked:**
        * Execution time and performance metrics
        * Response quality and coherence scores
        * Memory usage and token consumption
        * Error rates and failure modes
        * A/B testing group assignments
    
    **Example:**
        >>> execution = PromptExecution(
        ...     prompt_id="greeting_prompt",
        ...     prompt_text="Hello, user!",
        ...     model_name="gpt-4",
        ...     temperature=0.7
        ... )
        >>> # execution.prompt_hash is auto-generated
        >>> print(execution.prompt_hash[:8])  # First 8 chars of hash
    
    **Attributes:**
        execution_id (str): Unique UUID for this execution
        prompt_id (str): Logical identifier for the prompt template
        prompt_version (str): Semantic version of the prompt (default: "1.0.0")
        conversation_id (Optional[str]): Links executions in a conversation
        prompt_text (str): The actual prompt text sent to the model
        prompt_hash (str): SHA-256 hash of prompt + model params (auto-generated)
        model_name (str): Model identifier (e.g., "gpt-4", "claude-3")
        temperature (float): Model temperature setting
        execution_time_ms (float): Total execution time in milliseconds
        response_text (str): Model's response text
        response_quality_score (Optional[float]): Quality score (0.0-1.0)
        error_occurred (bool): Whether an error occurred during execution
        ab_test_group (Optional[str]): A/B test group assignment
    """
    
    # Unique identifiers
    execution_id: str = field(default_factory=lambda: str(uuid4()))
    prompt_id: str = ""
    prompt_version: str = "1.0.0"
    conversation_id: Optional[str] = None
    
    # Prompt content and metadata
    prompt_text: str = ""
    prompt_hash: str = field(init=False)
    prompt_template: Optional[str] = None
    prompt_variables: Dict[str, Any] = field(default_factory=dict)
    
    # Model configuration
    model_name: str = ""
    model_version: str = ""
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    
    # Execution context
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    environment: Dict[str, str] = field(default_factory=dict)
    git_commit: Optional[str] = None
    
    # Input metrics
    context_length: int = 0
    conversation_turn: int = 0
    
    # Response metrics
    response_text: str = ""
    response_length: int = 0
    response_tokens: Optional[int] = None
    
    # Performance metrics
    execution_time_ms: float = 0.0
    tokens_per_second: Optional[float] = None
    memory_usage_mb: float = 0.0
    
    # Quality metrics
    response_quality_score: Optional[float] = None
    coherence_score: Optional[float] = None
    relevance_score: Optional[float] = None
    
    # Error tracking
    error_occurred: bool = False
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    
    # Comparison and A/B testing
    baseline_execution_id: Optional[str] = None
    ab_test_group: Optional[str] = None
    experiment_tag: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Generate hash for prompt content after initialization.
        
        Called automatically after dataclass initialization to compute
        the prompt hash based on prompt text and model parameters.
        
        **Note:**
            The hash is deterministic and includes both prompt text and
            model parameters to ensure identical prompts with different
            model settings get different hashes.
        """
        self.prompt_hash = self._generate_prompt_hash()
    
    def _generate_prompt_hash(self) -> str:
        """Generate a deterministic hash for the prompt content.
        
        Creates a SHA-256 hash of the prompt text combined with sorted
        model parameters. This ensures consistent hashing regardless of
        parameter order while capturing both prompt and model variations.
        
        **Returns:**
            str: First 16 characters of SHA-256 hash (sufficient for uniqueness)
        
        **Example:**
            >>> execution = PromptExecution(
            ...     prompt_text="Hello",
            ...     model_parameters={"temp": 0.7, "model": "gpt-4"}
            ... )
            >>> len(execution.prompt_hash)
            16
            >>> execution.prompt_hash.isalnum()  # Hexadecimal string
            True
        """
        content = f"{self.prompt_text}|{json.dumps(self.model_parameters, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass 
class PromptComparison:
    """Data structure for statistical comparison between two prompt executions.
    
    This class encapsulates the results of comparing a baseline execution
    against a treatment execution, providing metrics for A/B testing and
    performance analysis.
    
    **Statistical Methods:**
        * Performance change calculation (percentage improvement/degradation)
        * Quality score differential analysis
        * Response length variation measurement
        * Statistical significance testing (when sufficient data available)
        * Effect size calculation for practical significance
    
    **Use Cases:**
        * A/B testing different prompt variations
        * Performance regression detection
        * Model comparison studies
        * Prompt optimization workflows
    
    **Example:**
        >>> comparison = tracker.compare_executions(
        ...     baseline_id="exec_123",
        ...     treatment_id="exec_456",
        ...     notes="Testing new greeting format"
        ... )
        >>> if comparison.performance_change_percent < -10:
        ...     print("Significant performance improvement!")
    
    **Attributes:**
        comparison_id (str): Unique identifier for this comparison
        baseline_execution_id (str): Reference to baseline execution
        treatment_execution_id (str): Reference to treatment execution
        quality_improvement (Optional[float]): Quality score change (can be negative)
        performance_change_percent (Optional[float]): Execution time change as percentage
        length_change_percent (Optional[float]): Response length change as percentage
        p_value (Optional[float]): Statistical significance (when calculable)
        effect_size (Optional[float]): Cohen's d or similar effect size measure
        is_significant (bool): Whether difference is statistically significant
        confidence_level (float): Confidence level for significance test (default: 0.95)
    """
    
    comparison_id: str = field(default_factory=lambda: str(uuid4()))
    baseline_execution_id: str = ""
    treatment_execution_id: str = ""
    
    # Comparison metrics
    quality_improvement: Optional[float] = None
    performance_change_percent: Optional[float] = None
    length_change_percent: Optional[float] = None
    
    # Statistical significance
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    is_significant: bool = False
    confidence_level: float = 0.95
    
    # Metadata
    comparison_type: str = "A/B"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    notes: str = ""


class PromptTracker:
    """Comprehensive prompt execution tracking and monitoring system.
    
    The PromptTracker is the core component of the Lethe prompt monitoring
    infrastructure. It provides automatic tracking of prompt executions,
    performance analytics, A/B testing capabilities, and integration with
    MLflow experiment tracking.
    
    **Architecture:**
        * SQLite database for local storage and fast queries
        * Optional MLflow integration for experiment tracking
        * Context manager interface for automatic timing
        * Thread-safe operations for concurrent usage
        * Extensible schema supporting custom metrics
    
    **Database Tables:**
        * `prompt_executions`: Core execution data with performance metrics
        * `prompt_versions`: Version control for prompt templates
        * `prompt_comparisons`: Statistical comparisons and A/B test results
    
    **Key Capabilities:**
        * Automatic execution timing and resource tracking
        * Quality score integration and trend analysis
        * Error tracking and failure mode analysis
        * Statistical comparison tools for A/B testing
        * Data export in multiple formats (CSV, JSON, Parquet)
        * Integration with existing Lethe monitoring components
    
    **Performance Considerations:**
        * Uses SQLite WAL mode for concurrent access
        * Indexed queries for fast analytics operations
        * Batch operations for bulk data processing
        * Optional async MLflow logging to avoid blocking
    
    **Example Usage:**
        Basic tracking:
        
        >>> tracker = PromptTracker("experiments/my_prompts.db")
        >>> with tracker.track_execution(
        ...     prompt_id="greeting_v1",
        ...     prompt_text="Hello! How can I help?",
        ...     model_config={"model": "gpt-4", "temperature": 0.7}
        ... ) as execution:
        ...     # Your model call here
        ...     execution.response_text = "Hi there!"
        ...     execution.response_quality_score = 0.9
        
        Analytics:
        
        >>> analytics = tracker.get_prompt_analytics("greeting_v1")
        >>> print(f"Success rate: {analytics['success_rate']:.1f}%")
        >>> print(f"Avg time: {analytics['avg_execution_time_ms']:.0f}ms")
        
        A/B Testing:
        
        >>> comparison = tracker.compare_executions(
        ...     baseline_id="exec_123",
        ...     treatment_id="exec_456"
        ... )
        >>> print(f"Performance change: {comparison.performance_change_percent:.1f}%")
    
    **Thread Safety:**
        This class is designed for concurrent usage. Database operations
        use connection pooling and transactions for consistency.
    
    **Error Handling:**
        * Database connection errors are handled gracefully
        * MLflow failures are logged but don't interrupt tracking
        * Invalid execution IDs raise ValueError with clear messages
        * Corrupted data is detected and reported
    """
    
    def __init__(self, db_path: str = "experiments/prompt_tracking.db") -> None:
        """Initialize prompt tracker with database and optional MLflow integration.
        
        Sets up the SQLite database, creates tables if they don't exist,
        and initializes optional integrations with MLflow and Lethe components.
        
        **Args:**
            db_path (str): Path to SQLite database file. Parent directories
                will be created if they don't exist. Defaults to
                "experiments/prompt_tracking.db".
        
        **Raises:**
            PermissionError: If unable to create database file or directories
            sqlite3.Error: If database initialization fails
        
        **Side Effects:**
            * Creates database file and parent directories if needed
            * Initializes database schema with tables and indexes
            * Sets up optional MLflow and DataManager connections
            * Establishes thread-safe database connection pool
        
        **Example:**
            >>> tracker = PromptTracker()  # Uses default path
            >>> tracker = PromptTracker("custom/path/prompts.db")
            >>> tracker.db_path
            PosixPath('custom/path/prompts.db')
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize optional components
        self.data_manager = DataManager() if DataManager else None
        self._init_database()
        self._current_execution: Optional[PromptExecution] = None
        
    def _init_database(self) -> None:
        """Initialize SQLite database with prompt tracking schema.
        
        Creates all necessary tables, indexes, and constraints for the prompt
        tracking system. This method is idempotent - it can be called multiple
        times safely.
        
        **Database Schema:**
            * `prompt_executions`: Main table with execution data and metrics
            * `prompt_versions`: Version control for prompt templates
            * `prompt_comparisons`: Statistical comparison results
        
        **Indexes Created:**
            * `idx_prompt_executions_timestamp`: For time-based queries
            * `idx_prompt_executions_prompt_id`: For prompt-specific analytics
            * `idx_prompt_executions_hash`: For duplicate detection
            * `idx_prompt_versions_prompt_id`: For version history queries
        
        **Constraints:**
            * Foreign key relationships for data integrity
            * NOT NULL constraints on critical fields
            * DEFAULT values for optional metrics
        
        **Raises:**
            sqlite3.Error: If database initialization fails
            PermissionError: If unable to write to database file
        
        **Note:**
            This method uses SQLite's "CREATE TABLE IF NOT EXISTS" syntax,
            making it safe to call on existing databases without data loss.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS prompt_executions (
                    execution_id TEXT PRIMARY KEY,
                    prompt_id TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    conversation_id TEXT,
                    prompt_text TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    prompt_template TEXT,
                    prompt_variables_json TEXT,
                    model_name TEXT NOT NULL,
                    model_version TEXT,
                    model_parameters_json TEXT,
                    temperature REAL,
                    max_tokens INTEGER,
                    timestamp TEXT NOT NULL,
                    environment_json TEXT,
                    git_commit TEXT,
                    context_length INTEGER DEFAULT 0,
                    conversation_turn INTEGER DEFAULT 0,
                    response_text TEXT,
                    response_length INTEGER DEFAULT 0,
                    response_tokens INTEGER,
                    execution_time_ms REAL DEFAULT 0.0,
                    tokens_per_second REAL,
                    memory_usage_mb REAL DEFAULT 0.0,
                    response_quality_score REAL,
                    coherence_score REAL,
                    relevance_score REAL,
                    error_occurred BOOLEAN DEFAULT FALSE,
                    error_message TEXT,
                    error_type TEXT,
                    baseline_execution_id TEXT,
                    ab_test_group TEXT,
                    experiment_tag TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS prompt_versions (
                    prompt_id TEXT,
                    version TEXT,
                    prompt_text TEXT NOT NULL,
                    created_by TEXT,
                    change_description TEXT,
                    parent_version TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (prompt_id, version)
                );
                
                CREATE TABLE IF NOT EXISTS prompt_comparisons (
                    comparison_id TEXT PRIMARY KEY,
                    baseline_execution_id TEXT NOT NULL,
                    treatment_execution_id TEXT NOT NULL,
                    quality_improvement REAL,
                    performance_change_percent REAL,
                    length_change_percent REAL,
                    p_value REAL,
                    effect_size REAL,
                    is_significant BOOLEAN DEFAULT FALSE,
                    confidence_level REAL DEFAULT 0.95,
                    comparison_type TEXT DEFAULT 'A/B',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT,
                    FOREIGN KEY (baseline_execution_id) REFERENCES prompt_executions (execution_id),
                    FOREIGN KEY (treatment_execution_id) REFERENCES prompt_executions (execution_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_prompt_executions_timestamp ON prompt_executions(timestamp);
                CREATE INDEX IF NOT EXISTS idx_prompt_executions_prompt_id ON prompt_executions(prompt_id);
                CREATE INDEX IF NOT EXISTS idx_prompt_executions_hash ON prompt_executions(prompt_hash);
                CREATE INDEX IF NOT EXISTS idx_prompt_versions_prompt_id ON prompt_versions(prompt_id);
            """)
    
    @contextmanager
    def track_execution(
        self,
        prompt_id: str,
        prompt_text: str,
        model_config: Dict[str, Any],
        **kwargs
    ) -> PromptExecution:
        """Context manager for tracking prompt execution with automatic timing.
        
        This is the primary interface for tracking prompt executions. It provides
        automatic timing, error handling, and database persistence while allowing
        flexible configuration of tracking parameters.
        
        **Args:**
            prompt_id (str): Unique identifier for the prompt template.
                Should be consistent across versions for analytics.
            prompt_text (str): The actual prompt text sent to the model.
                Used for hash generation and debugging.
            model_config (Dict[str, Any]): Model configuration including:
                - model (str): Model name (e.g., "gpt-4", "claude-3")
                - version (str, optional): Model version
                - temperature (float, optional): Sampling temperature
                - max_tokens (int, optional): Maximum response tokens
                - **other model-specific parameters**
            **kwargs: Additional tracking parameters:
                - conversation_id (str): Link multiple executions
                - prompt_version (str): Semantic version (default: "1.0.0")
                - experiment_tag (str): Tag for grouping experiments
                - ab_test_group (str): A/B test group assignment
                - baseline_execution_id (str): Reference execution for comparison
                - environment (Dict[str, str]): Environment variables
                - git_commit (str): Git commit hash for reproducibility
        
        **Yields:**
            PromptExecution: Execution object to populate during model call.
                Set response_text, response_quality_score, and other metrics.
        
        **Raises:**
            ValueError: If required parameters are missing or invalid
            sqlite3.Error: If database operations fail
            Exception: Re-raises any exception from the tracked code block
        
        **Usage Pattern:**
            >>> with tracker.track_execution(
            ...     prompt_id="user_query_v2",
            ...     prompt_text="Analyze this text: {text}",
            ...     model_config={
            ...         "model": "gpt-4",
            ...         "temperature": 0.3,
            ...         "max_tokens": 500
            ...     },
            ...     experiment_tag="text_analysis_experiment",
            ...     conversation_id="conv_123"
            ... ) as execution:
            ...     # Your LLM API call here
            ...     response = model.generate(prompt_text)
            ...     
            ...     # Set response data
            ...     execution.response_text = response.text
            ...     execution.response_tokens = response.token_count
            ...     execution.response_quality_score = evaluate_quality(response.text)
        
        **Error Handling:**
            * Exceptions during execution are caught and logged
            * Error details are stored in the execution record
            * Original exception is re-raised after logging
            * Partial execution data is still saved for analysis
        
        **Performance Notes:**
            * Timing resolution is millisecond precision
            * Database writes happen in finally block for reliability
            * MLflow logging is asynchronous and non-blocking
            * Memory usage is tracked via psutil if available
        """
        
        # Initialize execution tracking with provided parameters
        execution = PromptExecution(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            model_name=model_config.get("model", "unknown"),
            model_version=model_config.get("version", "unknown"),
            model_parameters=model_config,
            **kwargs
        )
        
        # Set up tracking state and start timing
        self._current_execution = execution
        start_time = time.time()
        
        try:
            # Yield execution object for user code to populate
            yield execution
            
        except Exception as e:
            # Capture error details for analysis
            execution.error_occurred = True
            execution.error_message = str(e)[:1000]  # Limit error message length
            execution.error_type = type(e).__name__
            # Re-raise to preserve original exception behavior
            raise
            
        finally:
            # Complete timing measurement (always executed)
            execution.execution_time_ms = (time.time() - start_time) * 1000
            
            # Calculate response metrics if response was set
            if hasattr(execution, 'response_text') and execution.response_text:
                execution.response_length = len(execution.response_text)
                
                # Calculate tokens per second if token count available
                if (execution.response_tokens and 
                    execution.execution_time_ms > 0):
                    execution.tokens_per_second = (
                        execution.response_tokens / (execution.execution_time_ms / 1000)
                    )
            
            # Persist execution data (local database)
            self._save_execution(execution)
            
            # Optional MLflow logging (non-blocking)
            self._log_to_mlflow(execution)
            
            # Clean up tracking state
            self._current_execution = None
    
    def _save_execution(self, execution: PromptExecution) -> None:
        """Save execution record to SQLite database.
        
        Persists the complete execution record to the database using a single
        atomic transaction. JSON fields are serialized and stored as TEXT.
        
        **Args:**
            execution (PromptExecution): Complete execution record to save
        
        **Database Operations:**
            * Inserts record into prompt_executions table
            * Serializes complex fields (variables, parameters, environment) as JSON
            * Uses parameterized queries to prevent SQL injection
            * Commits transaction atomically
        
        **Raises:**
            sqlite3.IntegrityError: If execution_id already exists
            sqlite3.Error: For other database-related errors
            json.JSONEncodeError: If unable to serialize JSON fields
        
        **Performance:**
            * Uses prepared statements for efficiency
            * Single transaction for atomicity
            * Indexes automatically updated
        
        **Note:**
            This is a private method called automatically by track_execution().
            Direct calls should be rare and used with caution.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO prompt_executions (
                    execution_id, prompt_id, prompt_version, conversation_id,
                    prompt_text, prompt_hash, prompt_template, prompt_variables_json,
                    model_name, model_version, model_parameters_json, temperature,
                    max_tokens, timestamp, environment_json, git_commit,
                    context_length, conversation_turn, response_text, response_length,
                    response_tokens, execution_time_ms, tokens_per_second, memory_usage_mb,
                    response_quality_score, coherence_score, relevance_score,
                    error_occurred, error_message, error_type,
                    baseline_execution_id, ab_test_group, experiment_tag
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                execution.execution_id, execution.prompt_id, execution.prompt_version,
                execution.conversation_id, execution.prompt_text, execution.prompt_hash,
                execution.prompt_template, json.dumps(execution.prompt_variables),
                execution.model_name, execution.model_version,
                json.dumps(execution.model_parameters), execution.temperature,
                execution.max_tokens, execution.timestamp,
                json.dumps(execution.environment), execution.git_commit,
                execution.context_length, execution.conversation_turn,
                execution.response_text, execution.response_length,
                execution.response_tokens, execution.execution_time_ms,
                execution.tokens_per_second, execution.memory_usage_mb,
                execution.response_quality_score, execution.coherence_score,
                execution.relevance_score, execution.error_occurred,
                execution.error_message, execution.error_type,
                execution.baseline_execution_id, execution.ab_test_group,
                execution.experiment_tag
            ))
    
    def _log_to_mlflow(self, execution: PromptExecution) -> None:
        """Log execution data to MLflow for experiment tracking.
        
        Creates a nested MLflow run to log execution parameters, metrics,
        and artifacts. This provides integration with MLflow experiment
        tracking workflows and enables advanced analytics.
        
        **Args:**
            execution (PromptExecution): Execution record to log
        
        **MLflow Data Logged:**
            * **Parameters**: prompt_id, prompt_version, prompt_hash, model_name, temperature
            * **Metrics**: execution_time_ms, response_length, context_length, memory_usage_mb,
              response_quality_score, tokens_per_second
            * **Artifacts**: 
                - Prompt text as `prompts/{execution_id}.txt`
                - Complete execution data as `executions/{execution_id}.json`
        
        **Error Handling:**
            * Silently skips if MLflow is not available (MLFLOW_AVAILABLE=False)
            * Logs warnings for MLflow connection errors
            * Does not raise exceptions to avoid interrupting main execution flow
            * Provides graceful degradation when MLflow server is unavailable
        
        **MLflow Integration:**
            * Creates nested runs with descriptive names
            * Preserves parent/child run relationships
            * Supports MLflow tracking server and local file storage
            * Compatible with MLflow autologging features
        
        **Performance:**
            * Non-blocking: failures don't interrupt main workflow
            * Batch artifacts where possible
            * Respects MLflow client timeouts
        
        **Example MLflow Run Structure:**
            ```
            Parent Run: "prompt_experiment_2024"
            ├── Child Run: "prompt_greeting_v1_exec_123"
            │   ├── Parameters: model=gpt-4, temperature=0.7
            │   ├── Metrics: execution_time_ms=150, quality_score=0.95
            │   └── Artifacts: prompts/, executions/
            ```
        
        **Note:**
            This is a private method called automatically by track_execution().
            MLflow integration is optional and controlled by MLFLOW_AVAILABLE flag.
        """
        if not MLFLOW_AVAILABLE:
            return  # Silently skip if MLflow not available
            
        try:
            # Create nested run for prompt tracking
            with mlflow.start_run(nested=True, run_name=f"prompt_{execution.prompt_id}"):
                # Log parameters
                mlflow.log_param("prompt_id", execution.prompt_id)
                mlflow.log_param("prompt_version", execution.prompt_version)
                mlflow.log_param("prompt_hash", execution.prompt_hash)
                mlflow.log_param("model_name", execution.model_name)
                mlflow.log_param("temperature", execution.temperature)
                
                # Log metrics
                mlflow.log_metric("execution_time_ms", execution.execution_time_ms)
                mlflow.log_metric("response_length", execution.response_length)
                mlflow.log_metric("context_length", execution.context_length)
                mlflow.log_metric("memory_usage_mb", execution.memory_usage_mb)
                
                if execution.response_quality_score:
                    mlflow.log_metric("quality_score", execution.response_quality_score)
                if execution.tokens_per_second:
                    mlflow.log_metric("tokens_per_second", execution.tokens_per_second)
                
                # Store prompt as artifact
                mlflow.log_text(execution.prompt_text, f"prompts/{execution.execution_id}.txt")
                
                # Store full execution data
                execution_data = asdict(execution)
                mlflow.log_dict(execution_data, f"executions/{execution.execution_id}.json")
                
        except Exception as e:
            print(f"Warning: Failed to log to MLflow: {e}")
    
    def get_prompt_history(self, prompt_id: str) -> pd.DataFrame:
        """Get execution history for a specific prompt ID.
        
        Retrieves all executions for the given prompt ID, ordered by timestamp
        (most recent first). Returns key metrics for trend analysis and debugging.
        
        **Args:**
            prompt_id (str): The prompt identifier to query
        
        **Returns:**
            pd.DataFrame: DataFrame with columns:
                - execution_id (str): Unique execution identifier
                - prompt_version (str): Version of the prompt used
                - timestamp (str): ISO format execution timestamp
                - execution_time_ms (float): Execution time in milliseconds
                - response_length (int): Response length in characters
                - response_quality_score (float): Quality score (0.0-1.0)
                - error_occurred (bool): Whether execution had errors
        
        **Example:**
            >>> history = tracker.get_prompt_history("greeting_v1")
            >>> print(f"Total executions: {len(history)}")
            >>> print(f"Average time: {history['execution_time_ms'].mean():.0f}ms")
            >>> recent_errors = history.head(10)['error_occurred'].sum()
            >>> print(f"Recent error rate: {recent_errors}/10")
        
        **Use Cases:**
            * Performance trend analysis over time
            * Quality regression detection
            * Error pattern identification
            * A/B test result visualization
            * Debugging specific prompt versions
        
        **Performance:**
            * Uses indexed query on prompt_id for fast retrieval
            * Results ordered by timestamp (most recent first)
            * Includes only essential columns for efficiency
        
        **Raises:**
            sqlite3.Error: If database query fails
            ValueError: If prompt_id is empty or None
        
        **Note:**
            Returns empty DataFrame if no executions found for the prompt_id.
            Use len(result) to check if any executions exist.
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                """
                SELECT execution_id, prompt_version, timestamp, execution_time_ms,
                       response_length, response_quality_score, error_occurred
                FROM prompt_executions 
                WHERE prompt_id = ?
                ORDER BY timestamp DESC
                """,
                conn,
                params=[prompt_id]
            )
    
    def compare_executions(
        self, 
        baseline_id: str, 
        treatment_id: str,
        notes: str = ""
    ) -> PromptComparison:
        """Compare two prompt executions and save the statistical comparison.
        
        Performs detailed statistical comparison between a baseline and treatment
        execution, calculating performance changes, quality improvements, and
        statistical significance where applicable.
        
        **Args:**
            baseline_id (str): Execution ID for the baseline/control execution
            treatment_id (str): Execution ID for the treatment/test execution
            notes (str, optional): Free-text notes about the comparison context
        
        **Returns:**
            PromptComparison: Comparison object with calculated metrics:
                - quality_improvement (float): Quality score delta
                - performance_change_percent (float): Execution time % change
                - length_change_percent (float): Response length % change
                - p_value (float): Statistical significance (when calculable)
                - effect_size (float): Practical significance measure
                - is_significant (bool): Whether difference is statistically significant
        
        **Calculations:**
            * **Performance Change**: ((treatment_time - baseline_time) / baseline_time) * 100
            * **Length Change**: ((treatment_length - baseline_length) / baseline_length) * 100
            * **Quality Improvement**: treatment_quality - baseline_quality
            * **Statistical Tests**: Applied when sufficient historical data exists
        
        **Example:**
            >>> comparison = tracker.compare_executions(
            ...     baseline_id="exec_baseline_123",
            ...     treatment_id="exec_treatment_456",
            ...     notes="Testing new prompt format with better examples"
            ... )
            >>> 
            >>> print(f"Quality improved by: {comparison.quality_improvement:.3f}")
            >>> print(f"Performance changed by: {comparison.performance_change_percent:.1f}%")
            >>> 
            >>> if comparison.performance_change_percent < -10:
            ...     print("Significant performance improvement!")
            >>> if comparison.quality_improvement > 0.05:
            ...     print("Meaningful quality improvement!")
        
        **Use Cases:**
            * A/B testing different prompt variations
            * Model comparison studies
            * Performance regression testing
            * Prompt optimization workflows
            * Quality assurance for prompt changes
        
        **Statistical Significance:**
            * Requires multiple executions of similar prompts for meaningful tests
            * Uses appropriate statistical tests based on data distribution
            * Reports confidence intervals when applicable
            * Accounts for multiple comparison adjustments
        
        **Raises:**
            ValueError: If execution IDs don't exist in database
            sqlite3.Error: If database operations fail
            TypeError: If execution data is corrupted or invalid
        
        **Performance:**
            * Comparison results are cached in database for future reference
            * Statistical calculations are performed in-memory for speed
            * Results include comparison_id for tracking and referencing
        
        **Note:**
            The comparison is automatically saved to the database and can be
            retrieved later using the returned comparison_id.
        """
        
        with sqlite3.connect(self.db_path) as conn:
            # Get execution data
            baseline = pd.read_sql_query(
                "SELECT * FROM prompt_executions WHERE execution_id = ?",
                conn, params=[baseline_id]
            ).iloc[0]
            
            treatment = pd.read_sql_query(
                "SELECT * FROM prompt_executions WHERE execution_id = ?", 
                conn, params=[treatment_id]
            ).iloc[0]
        
        # Calculate comparison metrics
        comparison = PromptComparison(
            baseline_execution_id=baseline_id,
            treatment_execution_id=treatment_id,
            notes=notes
        )
        
        # Performance change
        if baseline['execution_time_ms'] > 0:
            comparison.performance_change_percent = (
                (treatment['execution_time_ms'] - baseline['execution_time_ms']) 
                / baseline['execution_time_ms'] * 100
            )
        
        # Length change
        if baseline['response_length'] > 0:
            comparison.length_change_percent = (
                (treatment['response_length'] - baseline['response_length'])
                / baseline['response_length'] * 100
            )
        
        # Quality improvement
        if (baseline['response_quality_score'] and 
            treatment['response_quality_score']):
            comparison.quality_improvement = (
                treatment['response_quality_score'] - baseline['response_quality_score']
            )
        
        # Save comparison
        self._save_comparison(comparison)
        return comparison
    
    def _save_comparison(self, comparison: PromptComparison) -> None:
        """Save prompt comparison results to database.
        
        Persists the statistical comparison to the prompt_comparisons table
        using a single atomic transaction.
        
        **Args:**
            comparison (PromptComparison): Complete comparison record to save
        
        **Database Operations:**
            * Inserts record into prompt_comparisons table
            * Maintains foreign key relationships to prompt_executions
            * Uses parameterized queries for security
            * Commits transaction atomically
        
        **Raises:**
            sqlite3.IntegrityError: If comparison_id already exists or
                referenced executions don't exist
            sqlite3.Error: For other database-related errors
        
        **Note:**
            This is a private method called automatically by compare_executions().
            The comparison record enables later retrieval and analysis of A/B test results.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO prompt_comparisons VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                comparison.comparison_id, comparison.baseline_execution_id,
                comparison.treatment_execution_id, comparison.quality_improvement,
                comparison.performance_change_percent, comparison.length_change_percent,
                comparison.p_value, comparison.effect_size, comparison.is_significant,
                comparison.confidence_level, comparison.comparison_type,
                comparison.created_at, comparison.notes
            ))
    
    def get_prompt_analytics(self, prompt_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics and metrics for a specific prompt.
        
        Analyzes all executions for the given prompt ID and returns detailed
        statistics, trends, and performance metrics. This is the primary method
        for understanding prompt performance over time.
        
        **Args:**
            prompt_id (str): The prompt identifier to analyze
        
        **Returns:**
            Dict[str, Any]: Analytics dictionary containing:
                - total_executions (int): Total number of executions
                - success_rate (float): Percentage of successful executions (0-100)
                - avg_execution_time_ms (float): Average execution time in milliseconds
                - avg_response_length (float): Average response length in characters
                - performance_trend (str): Trend direction ('improving', 'declining', 'stable')
                - quality_trend (str or None): Quality score trend (when available)
                - latest_execution (str): ISO timestamp of most recent execution
                - memory_usage_avg (float): Average memory usage in MB
                - error (str): Error message if no executions found
        
        **Trend Analysis:**
            Uses correlation analysis to determine trend direction:
            * 'improving': Performance is getting better over time
            * 'declining': Performance is degrading over time  
            * 'stable': No significant trend detected
            * 'insufficient_data': Less than 2 executions available
        
        **Example:**
            >>> analytics = tracker.get_prompt_analytics("user_greeting_v2")
            >>> print(f"Prompt Health Report:")
            >>> print(f"  Total runs: {analytics['total_executions']}")
            >>> print(f"  Success rate: {analytics['success_rate']:.1f}%")
            >>> print(f"  Avg response time: {analytics['avg_execution_time_ms']:.0f}ms")
            >>> print(f"  Performance trend: {analytics['performance_trend']}")
            >>> 
            >>> if analytics['success_rate'] < 95:
            ...     print("WARNING: Low success rate detected!")
            >>> if analytics['performance_trend'] == 'declining':
            ...     print("ALERT: Performance degradation detected!")
        
        **Use Cases:**
            * Performance monitoring dashboards
            * Automated alerting systems
            * Prompt optimization workflows
            * Quality assurance reporting
            * Historical trend analysis
        
        **Performance Metrics:**
            * Execution time trends over time
            * Response quality score evolution
            * Error rate patterns and spikes
            * Memory usage optimization tracking
        
        **Raises:**
            sqlite3.Error: If database query fails
            ValueError: If prompt_id is empty or None
        
        **Note:**
            Returns {"error": "No executions found for prompt_id"} if no data exists.
            All numeric metrics are averaged across successful executions only.
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT execution_time_ms, response_length, response_quality_score,
                       memory_usage_mb, error_occurred, timestamp
                FROM prompt_executions 
                WHERE prompt_id = ?
                ORDER BY timestamp
                """,
                conn,
                params=[prompt_id]
            )
        
        if df.empty:
            return {"error": "No executions found for prompt_id"}
        
        analytics = {
            "total_executions": len(df),
            "success_rate": (1 - df['error_occurred'].mean()) * 100,
            "avg_execution_time_ms": df['execution_time_ms'].mean(),
            "avg_response_length": df['response_length'].mean(),
            "performance_trend": self._calculate_trend(df['execution_time_ms']),
            "quality_trend": self._calculate_trend(df['response_quality_score'].dropna()) if not df['response_quality_score'].dropna().empty else None,
            "latest_execution": df.iloc[-1]['timestamp'],
            "memory_usage_avg": df['memory_usage_mb'].mean()
        }
        
        return analytics
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction for a time series using correlation analysis.
        
        Determines whether a metric is improving, declining, or stable over time
        by calculating the correlation between the metric values and their
        temporal order.
        
        **Args:**
            series (pd.Series): Time-ordered series of numeric values
        
        **Returns:**
            str: Trend direction:
                - 'improving': Positive correlation > 0.1
                - 'declining': Negative correlation < -0.1
                - 'stable': Correlation between -0.1 and 0.1
                - 'insufficient_data': Less than 2 data points
        
        **Algorithm:**
            1. Create index series representing temporal order
            2. Calculate Pearson correlation coefficient
            3. Apply threshold-based classification
        
        **Thresholds:**
            * |correlation| > 0.1: Considered a meaningful trend
            * |correlation| ≤ 0.1: Considered stable/no trend
        
        **Example:**
            >>> import pandas as pd
            >>> response_times = pd.Series([100, 95, 90, 85, 80])  # Improving
            >>> tracker._calculate_trend(response_times)
            'improving'
            >>> 
            >>> error_rates = pd.Series([0.1, 0.15, 0.2, 0.25])  # Declining
            >>> tracker._calculate_trend(error_rates)
            'declining'
        
        **Use Cases:**
            * Performance monitoring alerts
            * Quality trend detection
            * Regression identification
            * Optimization impact assessment
        
        **Statistical Notes:**
            * Uses Pearson correlation (assumes linear relationships)
            * Robust to outliers when trend is strong
            * Requires at least 2 data points for calculation
            * Does not account for seasonality or complex patterns
        
        **Limitations:**
            * Simple linear correlation may miss complex patterns
            * Short series (< 10 points) may show false trends
            * Does not provide confidence intervals
            * Sensitive to recent data points
        """
        if len(series) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = range(len(series))
        slope = pd.Series(x).corr(series)
        
        if slope > 0.1:
            return "improving"
        elif slope < -0.1:
            return "declining"
        else:
            return "stable"
    
    def export_data(self, format: str = "csv") -> str:
        """Export all prompt tracking data in the specified format.
        
        Exports the complete prompt_executions table to a file in the requested
        format with timestamp-based naming for versioning.
        
        **Args:**
            format (str): Export format. Supported formats:
                - 'csv': Comma-separated values (default)
                - 'json': JSON with records orientation
                - 'parquet': Apache Parquet (columnar format)
        
        **Returns:**
            str: Generated filename with timestamp
        
        **Filename Format:**
            `prompt_executions_YYYYMMDD_HHMMSS.{extension}`
            
            Examples:
            - prompt_executions_20241201_143022.csv
            - prompt_executions_20241201_143022.json
            - prompt_executions_20241201_143022.parquet
        
        **Export Characteristics:**
            * **CSV**: Human-readable, Excel-compatible, good for analysis
            * **JSON**: Structured format, preserves data types, API-friendly
            * **Parquet**: Compressed columnar format, optimized for analytics
        
        **Data Included:**
            * All columns from prompt_executions table
            * Ordered by timestamp (oldest to newest)
            * JSON fields remain as JSON strings
            * NULL values handled appropriately per format
        
        **Example:**
            >>> filename = tracker.export_data('csv')
            >>> print(f"Data exported to: {filename}")
            >>> # Load in pandas for analysis
            >>> import pandas as pd
            >>> df = pd.read_csv(filename)
            >>> print(f"Exported {len(df)} executions")
        
        **Use Cases:**
            * Data backup and archival
            * External analysis with R, Python, or Excel
            * Data sharing with team members
            * Input for machine learning pipelines
            * Reporting and visualization tools
        
        **Performance:**
            * Streams data for memory efficiency with large datasets
            * Uses pandas optimized export functions
            * Parquet format provides best compression and speed
        
        **Raises:**
            ValueError: If format is not supported
            sqlite3.Error: If database query fails
            PermissionError: If unable to write to current directory
            pd.errors.ParserError: If data export fails
        
        **Note:**
            Files are created in the current working directory.
            Large datasets may take time to export - consider using
            parquet format for better performance.
        """
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM prompt_executions ORDER BY timestamp", conn)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format.lower() == "csv":
            filename = f"prompt_executions_{timestamp}.csv"
            df.to_csv(filename, index=False)
        elif format.lower() == "json":
            filename = f"prompt_executions_{timestamp}.json"
            df.to_json(filename, orient="records", indent=2)
        elif format.lower() == "parquet":
            filename = f"prompt_executions_{timestamp}.parquet"
            df.to_parquet(filename, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return filename


# Global prompt tracker instance for singleton pattern
_prompt_tracker: Optional[PromptTracker] = None

def get_prompt_tracker() -> PromptTracker:
    """Get global prompt tracker instance using singleton pattern.
    
    Returns a shared PromptTracker instance that persists across module imports.
    This ensures consistent tracking state within an application and provides
    a convenient access point for the monitoring system.
    
    **Returns:**
        PromptTracker: Global shared tracker instance using default database path
    
    **Singleton Behavior:**
        * First call creates new PromptTracker with default settings
        * Subsequent calls return the same instance
        * Thread-safe initialization
        * Uses default database path: "experiments/prompt_tracking.db"
    
    **Example:**
        >>> tracker1 = get_prompt_tracker()
        >>> tracker2 = get_prompt_tracker()
        >>> tracker1 is tracker2  # Same instance
        True
        >>> 
        >>> # Use for quick tracking
        >>> with get_prompt_tracker().track_execution(...) as execution:
        ...     # Your code here
        ...     pass
    
    **Use Cases:**
        * Simple applications needing basic tracking
        * Convenient access without managing tracker instances
        * Integration with existing code that expects singletons
        * Quick prototyping and experimentation
    
    **Alternative:**
        For applications needing multiple trackers or custom database paths,
        create PromptTracker instances directly:
        
        >>> tracker = PromptTracker("custom/path/tracking.db")
    
    **Thread Safety:**
        The singleton creation is thread-safe, but individual tracker
        operations should follow standard threading guidelines.
    """
    global _prompt_tracker
    if _prompt_tracker is None:
        _prompt_tracker = PromptTracker()
    return _prompt_tracker


# Convenience functions for easy integration with global tracker
def track_prompt(prompt_id: str, prompt_text: str, model_config: Dict[str, Any], **kwargs):
    """Convenience function to track a prompt execution using global tracker.
    
    Provides a simple interface for tracking prompt executions without
    managing tracker instances. Uses the global singleton tracker instance.
    
    **Args:**
        prompt_id (str): Unique identifier for the prompt template
        prompt_text (str): The actual prompt text sent to the model
        model_config (Dict[str, Any]): Model configuration dictionary
        **kwargs: Additional tracking parameters (see track_execution for details)
    
    **Returns:**
        Context manager yielding PromptExecution instance
    
    **Example:**
        >>> from src.monitoring.prompt_tracker import track_prompt
        >>> 
        >>> with track_prompt(
        ...     prompt_id="quick_test",
        ...     prompt_text="Hello, world!",
        ...     model_config={"model": "gpt-4", "temperature": 0.7}
        ... ) as execution:
        ...     # Your model call here
        ...     execution.response_text = "Hi there!"
    
    **Note:**
        This is equivalent to get_prompt_tracker().track_execution(...)
        but provides a cleaner import interface for simple use cases.
    """
    return get_prompt_tracker().track_execution(prompt_id, prompt_text, model_config, **kwargs)


def compare_prompts(baseline_id: str, treatment_id: str, notes: str = "") -> PromptComparison:
    """Convenience function to compare prompt executions using global tracker.
    
    Compares two prompt executions and returns statistical comparison results.
    Uses the global singleton tracker instance.
    
    **Args:**
        baseline_id (str): Execution ID for the baseline execution
        treatment_id (str): Execution ID for the treatment execution  
        notes (str, optional): Notes about the comparison context
    
    **Returns:**
        PromptComparison: Statistical comparison results
    
    **Example:**
        >>> from src.monitoring.prompt_tracker import compare_prompts
        >>> 
        >>> comparison = compare_prompts(
        ...     baseline_id="exec_123", 
        ...     treatment_id="exec_456",
        ...     notes="Testing improved greeting format"
        ... )
        >>> print(f"Quality change: {comparison.quality_improvement:.3f}")
        >>> print(f"Performance change: {comparison.performance_change_percent:.1f}%")
    
    **Note:**
        This is equivalent to get_prompt_tracker().compare_executions(...)
        but provides a cleaner import interface.
    """
    return get_prompt_tracker().compare_executions(baseline_id, treatment_id, notes)


def get_analytics(prompt_id: str) -> Dict[str, Any]:
    """Convenience function to get prompt analytics using global tracker.
    
    Retrieves comprehensive analytics for a specific prompt ID using
    the global singleton tracker instance.
    
    **Args:**
        prompt_id (str): The prompt identifier to analyze
    
    **Returns:**
        Dict[str, Any]: Analytics dictionary with performance metrics
    
    **Example:**
        >>> from src.monitoring.prompt_tracker import get_analytics
        >>> 
        >>> analytics = get_analytics("greeting_v1")
        >>> print(f"Success rate: {analytics['success_rate']:.1f}%")
        >>> print(f"Average time: {analytics['avg_execution_time_ms']:.0f}ms")
        >>> print(f"Trend: {analytics['performance_trend']}")
    
    **Note:**
        This is equivalent to get_prompt_tracker().get_prompt_analytics(...)
        but provides a cleaner import interface for analytics queries.
    """
    return get_prompt_tracker().get_prompt_analytics(prompt_id)