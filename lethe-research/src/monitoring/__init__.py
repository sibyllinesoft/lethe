"""
Lethe Prompt Monitoring System

A comprehensive monitoring and analytics system for LLM prompt executions,
providing real-time tracking, performance analysis, A/B testing capabilities,
and interactive dashboards for research and production workflows.

**Key Components:**
    * PromptTracker: Core execution tracking and analytics engine
    * PromptDashboard: Interactive web-based visualization and reporting
    * LethePromptMonitor: Advanced integration and monitoring features
    * CLI Tools: Command-line interface for operations and analysis

**Features:**
    * Automatic execution tracking with context managers
    * Real-time performance monitoring and alerting
    * Statistical A/B testing and comparison tools
    * Interactive Streamlit dashboards with plotly visualizations
    * MLflow integration for experiment tracking
    * Multiple export formats (CSV, JSON, Parquet)
    * Comprehensive CLI for automation and scripting

**Quick Start:**
    Basic prompt tracking:
    
    >>> from src.monitoring import track_prompt
    >>> with track_prompt(
    ...     prompt_id="greeting_v1",
    ...     prompt_text="Hello! How can I help?",
    ...     model_config={"model": "gpt-4", "temperature": 0.7}
    ... ) as execution:
    ...     # Your LLM call here
    ...     execution.response_text = "Hi there!"
    
    Analytics and comparison:
    
    >>> from src.monitoring import get_analytics, compare_prompts
    >>> analytics = get_analytics("greeting_v1")
    >>> print(f"Success rate: {analytics['success_rate']:.1f}%")
    >>> 
    >>> comparison = compare_prompts("exec_123", "exec_456")
    >>> print(f"Performance change: {comparison.performance_change_percent:.1f}%")
    
    Dashboard launch:
    
    >>> from src.monitoring import create_streamlit_dashboard
    >>> create_streamlit_dashboard()  # Access at http://localhost:8501

**Architecture:**
    * **Data Layer**: SQLite database with optimized schema and indexes
    * **Analytics Layer**: Pandas-based aggregation and statistical analysis
    * **Visualization Layer**: Plotly charts and Streamlit dashboard
    * **Integration Layer**: MLflow, Lethe components, external APIs

**Use Cases:**
    * Research experiment tracking and comparison
    * Production prompt monitoring and alerting
    * A/B testing for prompt optimization
    * Performance regression detection
    * Quality assurance and debugging workflows
    * Data export for external analysis tools

**Performance:**
    * Thread-safe operations for concurrent usage
    * Optimized database queries with proper indexing
    * Memory-efficient data processing for large datasets
    * Sub-second response times for dashboard interactions

**Integration Examples:**
    With existing Lethe components:
    
    >>> from src.monitoring.integration_examples import LethePromptMonitor
    >>> monitor = LethePromptMonitor()
    >>> monitor.setup_automated_monitoring()
    
    Command-line operations:
    
    ```bash
    python -m scripts.prompt_monitor status
    python -m scripts.prompt_monitor analyze greeting_v1 --verbose
    python -m scripts.prompt_monitor export --format parquet
    ```
"""

# Core tracking components
from .prompt_tracker import (
    PromptTracker,
    PromptExecution,
    PromptComparison,
    track_prompt,
    compare_prompts,
    get_analytics,
    get_prompt_tracker
)

# Dashboard and visualization components  
try:
    from .dashboard import (
        PromptDashboard,
        create_streamlit_dashboard
    )
    _DASHBOARD_AVAILABLE = True
except ImportError:
    _DASHBOARD_AVAILABLE = False
    PromptDashboard = None
    create_streamlit_dashboard = None

# Advanced integration and monitoring
try:
    from .integration_examples import (
        LethePromptMonitor,
        PromptVersionManager,
        example_integration_workflow
    )
    _INTEGRATION_AVAILABLE = True
except ImportError:
    _INTEGRATION_AVAILABLE = False
    LethePromptMonitor = None
    PromptVersionManager = None
    example_integration_workflow = None

# Build __all__ list based on available components
__all__ = [
    # Core tracking classes and data structures (always available)
    "PromptTracker",
    "PromptExecution", 
    "PromptComparison",
    "get_prompt_tracker",
    
    # Convenience functions for easy integration (always available)
    "track_prompt",
    "compare_prompts",
    "get_analytics",
]

# Add optional components if available
if _DASHBOARD_AVAILABLE:
    __all__.extend([
        "PromptDashboard",
        "create_streamlit_dashboard",
    ])

if _INTEGRATION_AVAILABLE:
    __all__.extend([
        "LethePromptMonitor",
        "PromptVersionManager",
        "example_integration_workflow",
    ])

# Version information
__version__ = "1.0.0"
__author__ = "Lethe Research Team"
__email__ = "research@lethe.ai"
__description__ = "Comprehensive LLM prompt monitoring and analytics system"

# Module-level configuration
DEFAULT_DATABASE_PATH = "experiments/prompt_tracking.db"
DEFAULT_DASHBOARD_PORT = 8501
DEFAULT_TIMELINE_DAYS = 7

# Dependency availability flags (for conditional feature support)
MLFLOW_AVAILABLE = True
STREAMLIT_AVAILABLE = True
PLOTLY_AVAILABLE = True

try:
    import mlflow
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import streamlit
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import plotly
except ImportError:
    PLOTLY_AVAILABLE = False

# Feature availability summary
FEATURES = {
    "core_tracking": True,  # Always available
    "mlflow_integration": MLFLOW_AVAILABLE,
    "web_dashboard": STREAMLIT_AVAILABLE and _DASHBOARD_AVAILABLE,
    "interactive_charts": PLOTLY_AVAILABLE and _DASHBOARD_AVAILABLE,
    "advanced_integration": _INTEGRATION_AVAILABLE,
    "statistical_analysis": True,  # Pandas-based, always available
    "data_export": True,  # Always available
    "cli_tools": True  # Always available
}

def get_feature_status() -> dict:
    """Get current feature availability status.
    
    **Returns:**
        dict: Feature availability mapping with installation suggestions:
            - features (dict): Boolean availability for each feature
            - install_suggestions (dict): Installation commands for missing features
            - core_functional (bool): Whether core tracking works
    
    **Example:**
        >>> status = get_feature_status()
        >>> if status['features']['web_dashboard']:
        ...     print("Dashboard available!")
        >>> else:
        ...     print(f"Install dashboard: {status['install_suggestions']['web_dashboard']}")
    """
    status = FEATURES.copy()
    
    suggestions = {}
    if not MLFLOW_AVAILABLE:
        suggestions["mlflow_integration"] = "pip install mlflow"
    if not STREAMLIT_AVAILABLE:
        suggestions["web_dashboard"] = "pip install streamlit plotly"
    if not PLOTLY_AVAILABLE and STREAMLIT_AVAILABLE:
        suggestions["interactive_charts"] = "pip install plotly"
    
    return {
        "features": status,
        "install_suggestions": suggestions,
        "core_functional": True  # Core tracking always works
    }

def print_feature_status() -> None:
    """Print human-readable feature availability status.
    
    Displays a formatted table showing which features are available
    and provides installation suggestions for missing dependencies.
    
    **Example Output:**
        ```
        🔍 Lethe Prompt Monitoring - Feature Status
        ==================================================
        ✅ Core Tracking: Available
        ✅ Statistical Analysis: Available  
        ❌ Web Dashboard: Not Available
        ✅ Data Export: Available
        
        📦 Installation Suggestions:
          • Web Dashboard: pip install streamlit plotly
        
        🚀 Core functionality is fully functional
        ```
    """
    status = get_feature_status()
    
    print("🔍 Lethe Prompt Monitoring - Feature Status")
    print("=" * 50)
    
    for feature, available in status["features"].items():
        icon = "✅" if available else "❌"
        feature_name = feature.replace('_', ' ').title()
        availability = 'Available' if available else 'Not Available'
        print(f"{icon} {feature_name}: {availability}")
    
    if status["install_suggestions"]:
        print("\n📦 Installation Suggestions:")
        for feature, cmd in status["install_suggestions"].items():
            feature_name = feature.replace('_', ' ').title()
            print(f"  • {feature_name}: {cmd}")
    
    core_status = "fully functional" if status['core_functional'] else "limited"
    print(f"\n🚀 Core functionality is {core_status}")

# Add utility functions to __all__
__all__.extend([
    "get_feature_status",
    "print_feature_status",
])

# Module initialization message (optional, for debugging)
import os
if os.getenv('LETHE_DEBUG_IMPORTS', '').lower() in ('1', 'true', 'yes'):
    print(f"📁 Loaded Lethe Monitoring v{__version__}")
    missing_features = [name for name, available in FEATURES.items() if not available]
    if missing_features:
        print(f"⚠️  Missing features: {', '.join(missing_features)}")
        print("💡 Run print_feature_status() for installation help")