# Quick Start Tutorial: Lethe Prompt Monitoring

Welcome to the Lethe Prompt Monitoring System! This tutorial will get you up and running in 10 minutes.

## 📋 Prerequisites

- Python 3.9+
- Lethe research environment
- Basic familiarity with Python

## 🚀 Installation & Setup

### 1. Verify Installation

First, check that the monitoring components are available:

```bash
# Check if core modules are accessible
python -c "from src.monitoring import PromptTracker; print('✅ PromptTracker available')"
python -c "from src.monitoring import track_prompt; print('✅ Tracking functions available')"
```

**Expected Output:**
```
✅ PromptTracker available
✅ Tracking functions available
```

### 2. Initialize Database

The system automatically creates a SQLite database, but let's verify it works:

```python
# File: examples/verify_setup.py
from src.monitoring import get_prompt_tracker

# Initialize the tracker (creates database if needed)
tracker = get_prompt_tracker()
print(f"✅ Database initialized at: {tracker.db_path}")

# Test basic functionality
stats = tracker.get_basic_stats()
print(f"📊 Current executions in database: {stats.get('total_executions', 0)}")
```

**Run it:**
```bash
python examples/verify_setup.py
```

## 🎯 Your First Prompt Tracking

### Simple Tracking Example

Create your first tracked prompt execution:

```python
# File: examples/first_tracking.py
import time
from src.monitoring import track_prompt

# Basic tracking with context manager
with track_prompt(
    prompt_id="hello_world",
    prompt_text="Say hello to the world of prompt monitoring!",
    model_config={"model": "gpt-4", "temperature": 0.7}
) as execution:
    # Simulate processing time
    time.sleep(0.5)
    
    # Simulate a response
    response = "Hello, world! Welcome to systematic prompt monitoring with Lethe."
    
    # Update execution with results
    execution.response_text = response
    execution.response_quality_score = 0.95
    execution.tokens_used = len(response.split())
    execution.success = True

print("✅ First prompt execution tracked!")
print(f"📝 Execution ID: {execution.execution_id}")
print(f"⏱️ Duration: {execution.execution_time_ms}ms")
```

**Run it:**
```bash
python examples/first_tracking.py
```

**Expected Output:**
```
✅ First prompt execution tracked!
📝 Execution ID: hello_world_20241226_143052_001
⏱️ Duration: 523ms
```

### Tracking with Error Handling

Learn how to track failed executions:

```python
# File: examples/error_tracking.py
from src.monitoring import track_prompt

# Track a prompt that fails
try:
    with track_prompt(
        prompt_id="error_example",
        prompt_text="This will simulate an error",
        model_config={"model": "test-model"}
    ) as execution:
        # Simulate an error
        raise ValueError("Simulated API error")
        
except Exception as e:
    # The execution context automatically tracks the error
    print(f"❌ Error tracked: {str(e)}")
    print(f"📝 Execution ID: {execution.execution_id}")
    print(f"🚨 Success status: {execution.success}")
```

## 📊 Quick Analytics

Get immediate insights from your tracked executions:

```python
# File: examples/quick_analytics.py
from src.monitoring import get_analytics

# Get basic statistics
analytics = get_analytics()

print("📈 Quick Analytics Report")
print("=" * 30)
print(f"Total executions: {analytics['summary']['total_executions']}")
print(f"Success rate: {analytics['summary']['success_rate']:.2%}")
print(f"Average duration: {analytics['summary']['avg_execution_time']:.0f}ms")

# Recent executions
recent = analytics.get('recent_executions', [])
if recent:
    print(f"\n📝 Recent Executions ({len(recent)}):")
    for exec in recent[:3]:  # Show last 3
        status = "✅" if exec['success'] else "❌"
        print(f"  {status} {exec['prompt_id']} - {exec['execution_time_ms']}ms")
```

## 🌐 Launch the Dashboard

Start the web dashboard to visualize your data:

```bash
# Option 1: Using the CLI tool
python scripts/prompt_monitor.py dashboard

# Option 2: Direct launch
python -c "from src.monitoring import create_streamlit_dashboard; create_streamlit_dashboard()"
```

**Expected Output:**
```
🚀 Starting Lethe Prompt Monitoring Dashboard
📊 Loading data from database...
🌐 Dashboard running at: http://localhost:8501

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501
```

### Dashboard Quick Tour

Once the dashboard opens in your browser:

1. **📊 Overview Tab**: See summary statistics and recent activity
2. **📈 Analytics Tab**: Dive into performance trends and patterns  
3. **🔍 Executions Tab**: Browse detailed execution logs
4. **📋 Export Tab**: Download data for external analysis

## 🛠️ CLI Quick Commands

Essential command-line operations:

```bash
# Check system status
python scripts/prompt_monitor.py status

# View recent executions
python scripts/prompt_monitor.py list --limit 5

# Get analytics summary
python scripts/prompt_monitor.py analytics --summary

# Export data
python scripts/prompt_monitor.py export --format csv --output recent_data.csv
```

## 🧪 Run Test Examples

Generate sample data and test all features:

```bash
# Run comprehensive test suite
python test_prompt_monitoring.py

# This creates sample data and validates:
# - Basic tracking functionality
# - Error handling 
# - Analytics generation
# - Dashboard compatibility
# - Export capabilities
```

## ✅ Verification Checklist

Confirm everything is working:

- [ ] Database initializes without errors
- [ ] Basic prompt tracking works
- [ ] Error tracking captures failures
- [ ] Analytics return meaningful data
- [ ] Dashboard launches and displays data
- [ ] CLI commands execute successfully
- [ ] Test suite passes completely

## 🚀 Next Steps

Now that you have the basics working:

1. **Integration Tutorial** → Learn to integrate with existing Lethe workflows
2. **Advanced Analytics** → Explore statistical analysis and comparisons
3. **Custom Tracking** → Add domain-specific metrics and monitoring
4. **Dashboard Deep Dive** → Master all visualization and export features

## 🆘 Troubleshooting

### Common Issues

**Database Connection Error**
```bash
# Check permissions
ls -la experiments/
# Should show prompt_tracking.db with write permissions
```

**Import Errors**
```bash
# Verify Python path
python -c "import sys; print(sys.path)"
# Should include current directory
```

**Dashboard Won't Start**
```bash
# Check if Streamlit is installed
python -c "import streamlit; print('Streamlit available')"

# Try alternative port
streamlit run src/monitoring/dashboard.py --server.port 8502
```

### Getting Help

1. Check logs in `experiments/logs/`
2. Run diagnostic: `python scripts/prompt_monitor.py diagnose`  
3. Review test output: `python test_prompt_monitoring.py --verbose`

---

**🎉 Congratulations!** You've successfully set up and tested the Lethe Prompt Monitoring System. You're ready to start tracking and analyzing your prompt executions systematically.