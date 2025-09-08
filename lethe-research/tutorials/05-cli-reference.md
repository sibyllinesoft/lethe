# CLI Reference Guide: Command-Line Operations

Complete reference for the `prompt_monitor.py` CLI tool with examples, automation scripts, and troubleshooting.

## 🎯 Overview

The Prompt Monitor CLI provides command-line access to all monitoring system features:

- **System Operations**: Status checks, health monitoring, diagnostics
- **Data Management**: List, filter, and export prompt execution data
- **Analytics**: Generate reports and statistical analysis  
- **Dashboard Control**: Launch and manage the web interface
- **Automation**: Scriptable operations for CI/CD and monitoring workflows

## 🚀 Quick Reference

### Basic Syntax

```bash
python scripts/prompt_monitor.py <command> [options]

# Or make it executable (Linux/Mac)
chmod +x scripts/prompt_monitor.py
./scripts/prompt_monitor.py <command> [options]
```

### Global Options

```bash
--help, -h          Show help message
--verbose, -v       Enable verbose output  
--quiet, -q         Suppress non-error output
--config FILE       Use custom configuration file
--database PATH     Use custom database path
```

## 📋 Command Reference

### 1. Status Commands

#### `status` - System Health Check

**Purpose**: Check overall system health and recent activity.

```bash
# Basic status check
python scripts/prompt_monitor.py status

# Detailed status with metrics
python scripts/prompt_monitor.py status --detailed

# Status for specific time period
python scripts/prompt_monitor.py status --hours 24

# Machine-readable output
python scripts/prompt_monitor.py status --format json
```

**Example Output**:
```
🟢 Lethe Prompt Monitoring System Status
========================================

📊 System Health:
  Database: ✅ Connected (prompt_tracking.db)
  Last Activity: 2 minutes ago
  Uptime: 2 days, 14 hours

📈 Recent Activity (Last 24 hours):
  Total Executions: 1,247
  Successful: 1,178 (94.5%)
  Failed: 69 (5.5%)
  Average Response Time: 1,234ms
  Average Quality Score: 0.847

🏷️ Top Prompt IDs:
  retrieval_query: 234 executions
  summarization: 156 executions
  analysis_prompt: 98 executions

⚠️ Alerts:
  None - System operating normally
```

#### `health` - Detailed Health Diagnostics

**Purpose**: Comprehensive system health check with detailed diagnostics.

```bash
# Full health check
python scripts/prompt_monitor.py health

# Health check with repair suggestions  
python scripts/prompt_monitor.py health --suggest-repairs

# Export health report
python scripts/prompt_monitor.py health --export health_report.json
```

**Example Output**:
```
🔍 System Health Diagnostics
============================

✅ Database Connectivity:
  - Connection: OK
  - Read Access: OK  
  - Write Access: OK
  - Index Performance: OK

✅ Data Integrity:
  - Schema Version: 1.2.0 (current)
  - Record Consistency: 100%
  - Orphaned Records: 0

⚠️ Performance Issues:
  - Slow Queries: 3 detected
  - Large Tables: executions (1.2M rows)
  - Missing Indexes: None

🔧 Recommendations:
  1. Archive data older than 90 days
  2. Optimize execution_time_ms queries
  3. Schedule regular maintenance
```

### 2. Data Management Commands

#### `list` - Browse Executions

**Purpose**: List and filter prompt executions with flexible criteria.

```bash
# List recent executions (default: 10)
python scripts/prompt_monitor.py list

# List specific number of executions
python scripts/prompt_monitor.py list --limit 50

# List with specific filters
python scripts/prompt_monitor.py list --prompt-id "retrieval_query" --limit 20

# List failed executions only
python scripts/prompt_monitor.py list --failed-only --limit 25

# List from date range
python scripts/prompt_monitor.py list --from "2024-12-01" --to "2024-12-07"

# List with specific tags
python scripts/prompt_monitor.py list --tags "experiment,v2" --limit 15

# List with quality threshold
python scripts/prompt_monitor.py list --min-quality 0.8 --limit 20

# List with output format options
python scripts/prompt_monitor.py list --format table --limit 10
python scripts/prompt_monitor.py list --format json --limit 5
python scripts/prompt_monitor.py list --format csv --limit 100
```

**Example Output (Table Format)**:
```
📋 Recent Prompt Executions
============================

┌────────────────────────────────┬─────────────────────┬─────────────────┬─────────┬──────────┬─────────┐
│ Execution ID                   │ Timestamp           │ Prompt ID       │ Success │ Time(ms) │ Quality │
├────────────────────────────────┼─────────────────────┼─────────────────┼─────────┼──────────┼─────────┤
│ retrieval_query_20241226_14... │ 2024-12-26 14:30:15│ retrieval_query │ ✅      │ 1,234    │ 0.847   │
│ summarization_20241226_14...   │ 2024-12-26 14:29:42│ summarization   │ ✅      │ 2,156    │ 0.923   │
│ analysis_prompt_20241226_14... │ 2024-12-26 14:28:33│ analysis_prompt │ ❌      │ 5,678    │ N/A     │
└────────────────────────────────┴─────────────────────┴─────────────────┴─────────┴──────────┴─────────┘

📊 Summary: 10 executions shown, 8 successful (80.0%), avg time: 1,678ms
```

#### `show` - Detailed Execution View

**Purpose**: Display complete details for specific execution(s).

```bash
# Show specific execution
python scripts/prompt_monitor.py show <execution_id>

# Show with full response text
python scripts/prompt_monitor.py show <execution_id> --full

# Show multiple executions
python scripts/prompt_monitor.py show <id1> <id2> <id3>

# Show and save to file
python scripts/prompt_monitor.py show <execution_id> --output execution_detail.json
```

**Example Output**:
```
🔍 Execution Details
====================

📝 Basic Information:
  ID: retrieval_query_20241226_143015_001
  Timestamp: 2024-12-26 14:30:15.123Z
  Prompt ID: retrieval_query
  Success: ✅ True
  Duration: 1,234ms

🤖 Model Configuration:
  Model: gpt-4
  Temperature: 0.7
  Max Tokens: 1000

📄 Prompt Text:
  "What are the latest developments in hybrid retrieval systems for academic research?"

📄 Response Text (truncated):
  "Hybrid retrieval systems have seen significant advancements in recent years..."

🎯 Quality Assessment:
  Score: 0.847
  Tokens Used: 245

🏷️ Tags: ['retrieval', 'academic', 'experiment']

📊 Metadata:
  user_id: researcher_123
  session_id: session_456789
  experiment_version: v2.1
```

#### `search` - Advanced Search

**Purpose**: Search executions using flexible text and metadata queries.

```bash
# Text search in prompts and responses
python scripts/prompt_monitor.py search "machine learning"

# Search with multiple terms
python scripts/prompt_monitor.py search "retrieval" "hybrid" --match-all

# Search in specific fields
python scripts/prompt_monitor.py search "gpt-4" --field model_config

# Search with date constraints
python scripts/prompt_monitor.py search "error" --failed-only --days 7

# Search with output formatting
python scripts/prompt_monitor.py search "experiment" --format json --limit 50
```

### 3. Analytics Commands

#### `analytics` - Generate Analytics Reports

**Purpose**: Generate comprehensive analytics and performance reports.

```bash
# Basic analytics summary
python scripts/prompt_monitor.py analytics

# Detailed analytics report
python scripts/prompt_monitor.py analytics --detailed

# Analytics for specific time period
python scripts/prompt_monitor.py analytics --days 30

# Analytics for specific prompts
python scripts/prompt_monitor.py analytics --prompt-id "retrieval_query"

# Analytics with specific metrics
python scripts/prompt_monitor.py analytics --metrics "time,quality,tokens"

# Export analytics to file
python scripts/prompt_monitor.py analytics --export analytics_report.json

# Generate visual reports (if supported)
python scripts/prompt_monitor.py analytics --charts --export visual_report.html
```

**Example Output**:
```
📊 Prompt Analytics Report
==========================
Report Period: Last 7 days (2024-12-19 to 2024-12-26)

📈 Overall Statistics:
  Total Executions: 3,456
  Success Rate: 94.2% (3,256 successful, 200 failed)
  Average Response Time: 1,378ms (σ=892ms)
  Average Quality Score: 0.831 (σ=0.156)
  Total Tokens Used: 1,234,567

⚡ Performance Metrics:
  Fastest Execution: 123ms
  Slowest Execution: 12,345ms
  P50 Response Time: 1,234ms
  P95 Response Time: 3,456ms
  P99 Response Time: 7,890ms

🎯 Quality Distribution:
  Excellent (>0.9): 1,156 executions (33.5%)
  Good (0.7-0.9): 1,789 executions (51.8%)
  Fair (0.5-0.7): 456 executions (13.2%)
  Poor (<0.5): 55 executions (1.6%)

🏷️ Top Prompt IDs:
  1. retrieval_query: 867 executions (avg: 1,234ms, quality: 0.845)
  2. summarization: 543 executions (avg: 2,156ms, quality: 0.912)
  3. analysis_prompt: 432 executions (avg: 987ms, quality: 0.756)

🚨 Error Analysis:
  API Errors: 89 (44.5% of failures)
  Timeout Errors: 67 (33.5% of failures)
  Validation Errors: 44 (22.0% of failures)
```

#### `compare` - Compare Prompt Performance

**Purpose**: Statistical comparison between different prompts or time periods.

```bash
# Compare two prompt IDs
python scripts/prompt_monitor.py compare "prompt_v1" "prompt_v2"

# Compare multiple prompts
python scripts/prompt_monitor.py compare "prompt_a" "prompt_b" "prompt_c"

# Compare with statistical tests
python scripts/prompt_monitor.py compare "old_prompt" "new_prompt" --statistical

# Compare over time periods
python scripts/prompt_monitor.py compare --time-periods "2024-12-01,2024-12-15" "2024-12-15,2024-12-30"

# Export comparison results
python scripts/prompt_monitor.py compare "v1" "v2" --export comparison_report.csv
```

**Example Output**:
```
🆚 Prompt Performance Comparison
=================================

Comparing: prompt_v1 vs prompt_v2
Time Period: Last 30 days
Sample Sizes: 234 vs 198 executions

📊 Performance Metrics:
                    prompt_v1    prompt_v2    Difference   P-value
Response Time (ms)      1,456        1,123         -333    0.023*
Quality Score           0.789        0.834       +0.045    0.012*
Success Rate           92.3%        96.0%        +3.7%    0.089
Tokens Used              234          198          -36    0.156

🧮 Statistical Analysis:
  * Statistically significant difference (p < 0.05)
  
📈 Key Findings:
  ✅ prompt_v2 is 23% faster than prompt_v1
  ✅ prompt_v2 has significantly higher quality scores
  ✅ prompt_v2 uses fewer tokens on average
  ⚠️ Success rate difference is not statistically significant

🏆 Recommendation: Migrate to prompt_v2 for better performance and quality
```

### 4. Export Commands

#### `export` - Data Export

**Purpose**: Export prompt execution data in various formats for external analysis.

```bash
# Export all data to CSV
python scripts/prompt_monitor.py export --format csv --output all_executions.csv

# Export filtered data
python scripts/prompt_monitor.py export --format json --prompt-id "retrieval_query" --output retrieval_data.json

# Export with date range
python scripts/prompt_monitor.py export --format excel --from "2024-12-01" --to "2024-12-31" --output december_data.xlsx

# Export summary statistics only
python scripts/prompt_monitor.py export --format json --summary-only --output summary_stats.json

# Export with custom fields
python scripts/prompt_monitor.py export --format csv --fields "id,timestamp,prompt_id,success,execution_time_ms,quality_score" --output custom_export.csv

# Export compressed data
python scripts/prompt_monitor.py export --format json --compress --output large_dataset.json.gz
```

**Format Options**:
```bash
--format csv        # Comma-separated values
--format json       # JSON format
--format excel      # Excel workbook (.xlsx)
--format parquet    # Parquet format (efficient for large datasets)
--format sqlite     # SQLite database file
```

### 5. Dashboard Commands

#### `dashboard` - Web Interface Control

**Purpose**: Launch and manage the Streamlit web dashboard.

```bash
# Launch dashboard (default port 8501)
python scripts/prompt_monitor.py dashboard

# Launch on custom port
python scripts/prompt_monitor.py dashboard --port 8502

# Launch with custom host
python scripts/prompt_monitor.py dashboard --host 0.0.0.0 --port 8501

# Launch in development mode (auto-reload)
python scripts/prompt_monitor.py dashboard --dev

# Launch with specific theme
python scripts/prompt_monitor.py dashboard --theme dark

# Check if dashboard is running
python scripts/prompt_monitor.py dashboard --status
```

### 6. Maintenance Commands

#### `cleanup` - Database Maintenance

**Purpose**: Perform database cleanup and optimization tasks.

```bash
# Archive old data (default: 90 days)
python scripts/prompt_monitor.py cleanup --archive --days 90

# Delete old data permanently
python scripts/prompt_monitor.py cleanup --delete --days 180 --confirm

# Optimize database (rebuild indexes, vacuum)
python scripts/prompt_monitor.py cleanup --optimize

# Full maintenance (archive + optimize)
python scripts/prompt_monitor.py cleanup --full-maintenance

# Dry run (show what would be cleaned)
python scripts/prompt_monitor.py cleanup --archive --days 60 --dry-run
```

#### `backup` - Backup Operations

**Purpose**: Create and restore database backups.

```bash
# Create backup
python scripts/prompt_monitor.py backup --create backup_20241226.sqlite

# Create compressed backup
python scripts/prompt_monitor.py backup --create backup_20241226.sqlite.gz --compress

# List available backups
python scripts/prompt_monitor.py backup --list

# Restore from backup
python scripts/prompt_monitor.py backup --restore backup_20241226.sqlite --confirm

# Validate backup integrity
python scripts/prompt_monitor.py backup --validate backup_20241226.sqlite
```

## 🤖 Automation & Scripting

### Bash Automation Scripts

#### Daily Health Check Script

```bash
#!/bin/bash
# daily_health_check.sh - Automated daily monitoring

DATE=$(date +%Y-%m-%d)
LOG_FILE="logs/health_check_$DATE.log"

echo "🔍 Daily Health Check - $DATE" | tee $LOG_FILE

# System status
python scripts/prompt_monitor.py status --detailed >> $LOG_FILE

# Health diagnostics  
python scripts/prompt_monitor.py health >> $LOG_FILE

# Analytics summary
python scripts/prompt_monitor.py analytics --days 1 >> $LOG_FILE

# Check for alerts
ERROR_RATE=$(python scripts/prompt_monitor.py analytics --days 1 --format json | jq -r '.summary.error_rate')

if (( $(echo "$ERROR_RATE > 0.1" | bc -l) )); then
    echo "🚨 ALERT: Error rate is $ERROR_RATE (>10%)" | tee -a $LOG_FILE
    # Send alert notification (email, Slack, etc.)
fi

echo "✅ Health check completed" | tee -a $LOG_FILE
```

#### Weekly Report Generation

```bash
#!/bin/bash
# weekly_report.sh - Generate comprehensive weekly reports

WEEK_START=$(date -d "7 days ago" +%Y-%m-%d)
WEEK_END=$(date +%Y-%m-%d)
REPORT_DIR="reports/week_$(date +%Y_W%V)"

mkdir -p $REPORT_DIR

echo "📊 Generating Weekly Report: $WEEK_START to $WEEK_END"

# Generate analytics report
python scripts/prompt_monitor.py analytics \
    --from "$WEEK_START" \
    --to "$WEEK_END" \
    --detailed \
    --export "$REPORT_DIR/weekly_analytics.json"

# Export raw data
python scripts/prompt_monitor.py export \
    --format csv \
    --from "$WEEK_START" \
    --to "$WEEK_END" \
    --output "$REPORT_DIR/weekly_executions.csv"

# Generate comparison with previous week
PREV_WEEK_START=$(date -d "14 days ago" +%Y-%m-%d)
PREV_WEEK_END=$(date -d "7 days ago" +%Y-%m-%d)

python scripts/prompt_monitor.py compare \
    --time-periods "$PREV_WEEK_START,$PREV_WEEK_END" "$WEEK_START,$WEEK_END" \
    --statistical \
    --export "$REPORT_DIR/week_comparison.csv"

echo "✅ Weekly report generated in $REPORT_DIR"
```

#### Automated Backup Script

```bash
#!/bin/bash
# automated_backup.sh - Regular database backups

BACKUP_DIR="backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="prompt_monitoring_backup_$DATE"

mkdir -p $BACKUP_DIR

echo "💾 Creating backup: $BACKUP_NAME"

# Create compressed backup
python scripts/prompt_monitor.py backup \
    --create "$BACKUP_DIR/$BACKUP_NAME.sqlite.gz" \
    --compress

# Validate backup
python scripts/prompt_monitor.py backup \
    --validate "$BACKUP_DIR/$BACKUP_NAME.sqlite.gz"

if [ $? -eq 0 ]; then
    echo "✅ Backup created and validated successfully"
    
    # Clean old backups (keep last 30 days)
    find $BACKUP_DIR -name "*.sqlite.gz" -mtime +30 -delete
    echo "🧹 Cleaned old backups"
else
    echo "❌ Backup validation failed"
    exit 1
fi
```

### Python Automation Scripts

#### Monitoring Automation

```python
#!/usr/bin/env python3
# monitoring_automation.py - Automated monitoring and alerting

import subprocess
import json
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText

class MonitoringAutomation:
    def __init__(self):
        self.thresholds = {
            'error_rate': 0.1,      # 10%
            'avg_response_time': 5000,  # 5 seconds
            'min_quality_score': 0.5
        }
    
    def get_system_status(self):
        """Get current system status using CLI."""
        result = subprocess.run([
            'python', 'scripts/prompt_monitor.py', 
            'status', '--format', 'json'
        ], capture_output=True, text=True)
        
        return json.loads(result.stdout)
    
    def check_thresholds(self, status):
        """Check if any thresholds are exceeded."""
        alerts = []
        
        error_rate = status.get('error_rate', 0)
        if error_rate > self.thresholds['error_rate']:
            alerts.append(f"High error rate: {error_rate:.1%}")
        
        avg_time = status.get('avg_response_time', 0)
        if avg_time > self.thresholds['avg_response_time']:
            alerts.append(f"Slow response time: {avg_time}ms")
        
        avg_quality = status.get('avg_quality_score', 1.0)
        if avg_quality < self.thresholds['min_quality_score']:
            alerts.append(f"Low quality score: {avg_quality:.3f}")
        
        return alerts
    
    def send_alert(self, alerts):
        """Send alert notifications."""
        subject = f"Prompt Monitoring Alert - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        body = "The following issues were detected:\n\n" + "\n".join(f"• {alert}" for alert in alerts)
        
        # Implement your notification method here
        # (email, Slack, Discord, etc.)
        print(f"ALERT: {subject}")
        print(body)
    
    def run_monitoring_cycle(self):
        """Run one monitoring cycle."""
        try:
            status = self.get_system_status()
            alerts = self.check_thresholds(status)
            
            if alerts:
                self.send_alert(alerts)
            else:
                print("✅ All systems normal")
                
        except Exception as e:
            print(f"❌ Monitoring error: {e}")

if __name__ == "__main__":
    monitor = MonitoringAutomation()
    monitor.run_monitoring_cycle()
```

## 🔧 Configuration & Customization

### CLI Configuration File

Create `~/.prompt_monitor_config.yml`:

```yaml
# CLI Configuration
database:
  path: "experiments/prompt_tracking.db"
  backup_path: "backups"

output:
  default_format: "table"
  max_results: 50
  date_format: "%Y-%m-%d %H:%M:%S"

dashboard:
  default_port: 8501
  default_host: "localhost"
  theme: "light"

alerts:
  error_rate_threshold: 0.1
  response_time_threshold: 5000
  quality_score_threshold: 0.5

export:
  default_directory: "exports"
  compress_large_files: true
  include_metadata: true
```

### Environment Variables

```bash
# Set environment variables for CLI behavior
export PROMPT_MONITOR_DB_PATH="custom_database.db"
export PROMPT_MONITOR_CONFIG="custom_config.yml"
export PROMPT_MONITOR_LOG_LEVEL="DEBUG"
export PROMPT_MONITOR_DEFAULT_FORMAT="json"
```

## 🐛 Troubleshooting

### Common Issues

#### Database Connection Errors

```bash
# Check database status
python scripts/prompt_monitor.py health

# Verify database file exists and is accessible
ls -la experiments/prompt_tracking.db

# Test database connectivity
python -c "import sqlite3; sqlite3.connect('experiments/prompt_tracking.db').execute('SELECT 1')"

# Repair database if corrupted
python scripts/prompt_monitor.py cleanup --repair
```

#### Permission Issues

```bash
# Check file permissions
ls -la scripts/prompt_monitor.py
ls -la experiments/

# Fix permissions
chmod +x scripts/prompt_monitor.py
chmod 755 experiments/
chmod 664 experiments/prompt_tracking.db
```

#### Performance Issues

```bash
# Check database size and performance
python scripts/prompt_monitor.py health --performance

# Optimize database
python scripts/prompt_monitor.py cleanup --optimize

# Archive old data
python scripts/prompt_monitor.py cleanup --archive --days 90
```

### Debug Mode

Enable verbose output for troubleshooting:

```bash
# Enable debug logging
python scripts/prompt_monitor.py --verbose status

# Check command parsing
python scripts/prompt_monitor.py --help

# Validate configuration
python scripts/prompt_monitor.py config --validate
```

## 🎯 Best Practices

### Regular Maintenance

```bash
# Daily: Health check
python scripts/prompt_monitor.py health

# Weekly: Analytics review
python scripts/prompt_monitor.py analytics --days 7 --detailed

# Monthly: Database maintenance
python scripts/prompt_monitor.py cleanup --full-maintenance

# Quarterly: Full backup
python scripts/prompt_monitor.py backup --create quarterly_backup.sqlite.gz --compress
```

### Automation Integration

```bash
# Cron job examples (add to crontab with `crontab -e`)

# Health check every hour
0 * * * * /path/to/python /path/to/scripts/prompt_monitor.py health >> /var/log/prompt_monitor.log 2>&1

# Daily analytics report
0 9 * * * /path/to/daily_health_check.sh

# Weekly backup
0 2 * * 1 /path/to/automated_backup.sh

# Monthly cleanup
0 3 1 * * /path/to/python /path/to/scripts/prompt_monitor.py cleanup --full-maintenance
```

---

**🎉 CLI mastery achieved!** You can now operate the entire monitoring system from the command line, create automation scripts, and integrate with existing workflows efficiently.