# Troubleshooting Guide

Common issues and solutions for the Lethe Prompt Monitoring System.

## 🚨 Common Issues

### Installation Problems

#### Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'src.monitoring'
# Solution: Ensure you're in the project root directory
cd /path/to/lethe-research
python -c "from src.monitoring import PromptTracker"

# Alternative: Add to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/lethe-research"
```

#### Database Creation Issues
```bash
# Error: sqlite3.OperationalError: unable to open database file
# Solution: Check permissions and create directory
mkdir -p experiments
chmod 755 experiments
ls -la experiments/  # Should show write permissions
```

#### Missing Dependencies
```bash
# Error: ImportError: No module named 'streamlit'
# Solution: Install required dependencies
pip install streamlit plotly pandas numpy scipy

# For development features
pip install mlflow asyncpg nltk textstat
```

### Runtime Problems

#### Dashboard Won't Start
```python
# Error: "Address already in use"
# Solution: Check for running processes
lsof -i :8501  # Check port 8501
# Kill existing process or use different port
python scripts/prompt_monitor.py dashboard --port 8502

# Error: "Streamlit not found"
# Solution: Verify Streamlit installation
python -c "import streamlit; print('Streamlit available')"
```

#### Database Connection Errors
```python
# Error: "database is locked"
# Solution: Close other connections and check file locks
python -c "
import sqlite3
conn = sqlite3.connect('experiments/prompt_tracking.db')
conn.execute('PRAGMA journal_mode=WAL;')  # Enable WAL mode
conn.close()
"

# Error: "no such table: prompt_executions"
# Solution: Initialize database
python -c "
from src.monitoring import get_prompt_tracker
tracker = get_prompt_tracker()  # Creates tables if needed
"
```

#### Performance Issues
```bash
# Slow dashboard loading
# Solution: Limit data scope and optimize queries
python scripts/prompt_monitor.py cleanup --optimize
python scripts/prompt_monitor.py cleanup --archive --days 90

# High memory usage
# Solution: Check for memory leaks and optimize batch size
python scripts/prompt_monitor.py health --performance
```

### Data Issues

#### Missing or Incomplete Executions
```python
# Issue: Executions not saving properly
# Debug: Check for exceptions in tracking
with track_prompt("test_id", "test prompt", {"model": "test"}) as execution:
    try:
        # Your code here
        execution.response_text = "test response"
        execution.success = True
    except Exception as e:
        print(f"Tracking error: {e}")
        # Execution auto-saves on context exit
```

#### Inconsistent Quality Scores
```python
# Issue: Quality scores vary unexpectedly
# Debug: Validate quality calculation
from src.monitoring import get_prompt_tracker

tracker = get_prompt_tracker()
executions = tracker.get_recent_executions(limit=10)

for exec_data in executions:
    score = exec_data.get('response_quality_score')
    if score is None:
        print(f"Missing quality score: {exec_data['execution_id']}")
    elif not (0.0 <= score <= 1.0):
        print(f"Invalid quality score {score}: {exec_data['execution_id']}")
```

#### Analytics Calculation Errors
```python
# Issue: Analytics return unexpected values
# Debug: Verify data integrity
python scripts/prompt_monitor.py health --data-integrity

# Check for data anomalies
python -c "
from src.monitoring import get_analytics
analytics = get_analytics()
print('Analytics validation:')
for key, value in analytics.get('summary', {}).items():
    if value is None:
        print(f'  {key}: None (check data)')
    elif isinstance(value, (int, float)) and value < 0:
        print(f'  {key}: {value} (negative value - check logic)')
"
```

## 🔧 Diagnostic Commands

### System Health Checks
```bash
# Comprehensive system check
python scripts/prompt_monitor.py health

# Database integrity check
python scripts/prompt_monitor.py health --data-integrity

# Performance analysis
python scripts/prompt_monitor.py health --performance

# Configuration validation
python scripts/prompt_monitor.py config --validate
```

### Data Validation
```bash
# Check recent executions
python scripts/prompt_monitor.py list --limit 10

# Validate database schema
python -c "
import sqlite3
conn = sqlite3.connect('experiments/prompt_tracking.db')
cursor = conn.cursor()
cursor.execute('PRAGMA table_info(prompt_executions)')
columns = [row[1] for row in cursor.fetchall()]
expected = ['execution_id', 'timestamp', 'prompt_id', 'success']
missing = [col for col in expected if col not in columns]
if missing:
    print(f'Missing columns: {missing}')
else:
    print('Database schema valid')
conn.close()
"
```

### Performance Diagnostics
```bash
# Check database size and performance
python -c "
import os
db_path = 'experiments/prompt_tracking.db'
if os.path.exists(db_path):
    size_mb = os.path.getsize(db_path) / 1024 / 1024
    print(f'Database size: {size_mb:.1f} MB')
else:
    print('Database not found')
"

# Analyze slow queries
python scripts/prompt_monitor.py analytics --performance --days 1
```

## 🐛 Debug Mode

### Enable Verbose Logging
```python
# Set up debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export PROMPT_MONITOR_LOG_LEVEL=DEBUG
python scripts/prompt_monitor.py status
```

### Debug Specific Components
```python
# Debug tracking operations
import logging
from src.monitoring import track_prompt

logging.getLogger('src.monitoring').setLevel(logging.DEBUG)

with track_prompt("debug_test", "Debug prompt", {"model": "test"}) as execution:
    print(f"Execution ID: {execution.execution_id}")
    execution.response_text = "Debug response"
    execution.success = True
    print(f"Execution saved: {execution.execution_id}")
```

### Memory and Resource Monitoring
```python
# Monitor memory usage
import psutil
import os

def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

# Use before and after operations
print_memory_usage()
# Your monitoring operations here
print_memory_usage()
```

## 🔄 Recovery Procedures

### Database Recovery
```bash
# Backup current database
cp experiments/prompt_tracking.db experiments/prompt_tracking_backup.db

# Check database integrity
sqlite3 experiments/prompt_tracking.db "PRAGMA integrity_check;"

# Repair database if corrupted
sqlite3 experiments/prompt_tracking.db ".recover" | sqlite3 experiments/prompt_tracking_recovered.db

# If recovery fails, initialize fresh database
mv experiments/prompt_tracking.db experiments/prompt_tracking_corrupted.db
python -c "from src.monitoring import get_prompt_tracker; get_prompt_tracker()"
```

### Configuration Reset
```bash
# Reset to default configuration
rm -f ~/.prompt_monitor_config.yml
python scripts/prompt_monitor.py config --generate-default

# Verify configuration
python scripts/prompt_monitor.py config --validate --verbose
```

### Dashboard Reset
```bash
# Clear Streamlit cache
rm -rf ~/.streamlit/
python -c "import streamlit as st; st.cache_data.clear(); st.cache_resource.clear()"

# Restart dashboard with fresh state
python scripts/prompt_monitor.py dashboard --port 8502
```

## 🚨 Emergency Procedures

### Complete System Reset
```bash
#!/bin/bash
# emergency_reset.sh - Complete system reset (USE WITH CAUTION)

echo "⚠️  EMERGENCY SYSTEM RESET - This will delete all data!"
read -p "Are you sure? Type 'yes' to continue: " confirm

if [ "$confirm" = "yes" ]; then
    echo "🔄 Backing up current data..."
    mkdir -p backups/emergency_$(date +%Y%m%d_%H%M%S)
    cp -r experiments backups/emergency_$(date +%Y%m%d_%H%M%S)/
    
    echo "🗑️  Removing current data..."
    rm -rf experiments/prompt_tracking.db
    rm -rf experiments/logs/
    rm -rf ~/.streamlit/
    
    echo "🔧 Reinitializing system..."
    python -c "from src.monitoring import get_prompt_tracker; get_prompt_tracker()"
    
    echo "✅ System reset complete. Previous data backed up."
    echo "🧪 Run verification: python examples/verify_setup.py"
else
    echo "❌ Reset cancelled."
fi
```

### Data Migration
```bash
# Migrate data to new version
python scripts/data_migration.py --from-version 1.0 --to-version 1.1

# Export data before migration
python scripts/prompt_monitor.py export --format json --output pre_migration_backup.json

# Verify migration success
python scripts/prompt_monitor.py health --data-integrity
```

## 📞 Getting Help

### Self-Diagnosis Checklist
- [ ] Run `python examples/verify_setup.py`
- [ ] Check `python scripts/prompt_monitor.py health`
- [ ] Verify database file exists and has write permissions
- [ ] Confirm all dependencies are installed
- [ ] Check available disk space and memory
- [ ] Review recent error logs in `experiments/logs/`

### Information to Collect
When reporting issues, please include:

1. **System Information**
   ```bash
   python --version
   pip list | grep -E "(streamlit|plotly|pandas|numpy|scipy)"
   uname -a  # On Linux/Mac
   ```

2. **Error Details**
   ```bash
   python scripts/prompt_monitor.py health --verbose 2>&1
   ```

3. **Database Status**
   ```bash
   ls -la experiments/
   sqlite3 experiments/prompt_tracking.db "SELECT count(*) FROM prompt_executions;"
   ```

4. **Configuration**
   ```bash
   python scripts/prompt_monitor.py config --show
   ```

### Advanced Debugging

#### Enable SQL Query Logging
```python
import sqlite3
import logging

# Enable SQLite logging
logging.getLogger('sqlite3').setLevel(logging.DEBUG)

# Or use connection callback
def trace_queries(statement):
    print(f"SQL: {statement}")

conn = sqlite3.connect('experiments/prompt_tracking.db')
conn.set_trace_callback(trace_queries)
```

#### Profile Performance
```python
import cProfile
import pstats

# Profile specific operation
def profile_operation():
    from src.monitoring import get_analytics
    return get_analytics()

# Run profiler
cProfile.run('profile_operation()', 'profile_stats.prof')

# Analyze results
stats = pstats.Stats('profile_stats.prof')
stats.sort_stats('cumulative').print_stats(10)
```

---

**🆘 Still need help?** If these solutions don't resolve your issue:
1. Check the [GitHub Issues](https://github.com/your-repo/issues) for similar problems
2. Create a new issue with the diagnostic information above
3. Include specific error messages and steps to reproduce
4. Tag the issue with appropriate labels (bug, help wanted, etc.)