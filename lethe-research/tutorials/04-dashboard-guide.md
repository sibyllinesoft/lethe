# Dashboard Guide: Navigation & Features

Master the Streamlit-based web dashboard for comprehensive prompt monitoring, analytics, and reporting.

## 🎯 Overview

The Lethe Prompt Monitoring Dashboard provides:

- **Real-time Monitoring**: Live view of prompt executions and system health
- **Interactive Analytics**: Explore performance trends and patterns  
- **Comparative Analysis**: Side-by-side prompt variant comparisons
- **Export Capabilities**: Download data and reports in multiple formats
- **Visual Insights**: Charts, graphs, and statistical visualizations

## 🚀 Getting Started

### Launch the Dashboard

```bash
# Method 1: Using CLI tool (recommended)
python scripts/prompt_monitor.py dashboard

# Method 2: Direct launch
streamlit run src/monitoring/dashboard.py

# Method 3: Custom port
streamlit run src/monitoring/dashboard.py --server.port 8502
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

### First Look

When you open the dashboard, you'll see:

1. **Sidebar Navigation**: Switch between different views and filters
2. **Main Content Area**: Primary dashboard content
3. **Status Bar**: System health and data refresh indicators
4. **Filter Controls**: Date ranges, tags, and prompt ID filters

## 🗂️ Dashboard Sections

### 1. Overview Tab - System Health at a Glance

**Purpose**: High-level system monitoring and key performance indicators.

**Key Components**:

```python
# What you'll see in the Overview tab
📊 Key Metrics:
  • Total Executions: 1,247
  • Success Rate: 94.2%
  • Avg Response Time: 1,235ms
  • Avg Quality Score: 0.847

📈 Real-time Charts:
  • Executions per hour (last 24h)
  • Success rate trend
  • Response time distribution
  • Quality score histogram

🔍 Recent Activity:
  • Last 10 executions with status
  • Recent errors and warnings
  • System alerts and notifications
```

**Navigation Tips**:
- Use the **refresh interval** dropdown to set auto-refresh (15s, 30s, 1m, 5m)
- Click **"Refresh Data"** for immediate updates
- Hover over charts for detailed tooltips
- Use the **time range selector** to focus on specific periods

**Interpreting the Overview**:

```python
# Success Rate Analysis
✅ >95%: Excellent - System operating smoothly
⚠️ 90-95%: Good - Monitor for patterns
❌ <90%: Needs attention - Investigate errors

# Response Time Analysis  
🟢 <500ms: Fast - Optimal user experience
🟡 500-2000ms: Moderate - Acceptable performance
🔴 >2000ms: Slow - Optimization needed

# Quality Score Analysis
🏆 >0.8: High quality responses
📊 0.6-0.8: Good quality, room for improvement  
⚠️ <0.6: Low quality, review prompts
```

### 2. Analytics Tab - Deep Performance Insights

**Purpose**: Detailed analysis of prompt performance, trends, and patterns.

**Key Features**:

#### Time Series Analysis
```python
📈 Available Charts:
  • Execution time trends (line chart)
  • Quality score evolution (line chart)
  • Success rate over time (area chart)
  • Token usage patterns (bar chart)
  • Error frequency (line chart)

🎛️ Controls:
  • Date range picker
  • Granularity selector (hourly, daily, weekly)
  • Metric selector (multiple metrics)
  • Smoothing options (rolling average)
```

#### Performance Distribution
```python
📊 Visualizations:
  • Response time histogram
  • Quality score distribution
  • Token usage box plots
  • Success/failure pie charts

🔍 Insights Panel:
  • Statistical summaries
  • Percentile breakdowns (P50, P90, P95, P99)
  • Outlier identification
  • Trend detection
```

#### Comparative Analysis
```python
🆚 Compare By:
  • Prompt IDs
  • Model types
  • Time periods
  • Tags/categories

📋 Comparison Metrics:
  • Performance (speed, reliability)
  • Quality (scores, consistency)  
  • Efficiency (tokens, cost)
  • Usage patterns
```

**How to Use Analytics**:

1. **Select Time Range**: Choose your analysis period using the date picker
2. **Choose Metrics**: Select which performance metrics to analyze
3. **Apply Filters**: Filter by prompt ID, tags, or model type
4. **Explore Charts**: Interact with visualizations for detailed insights
5. **Export Data**: Download charts or raw data for external analysis

### 3. Executions Tab - Detailed Execution Browser

**Purpose**: Browse, search, and analyze individual prompt executions.

**Key Components**:

#### Execution List
```python
📋 Table Columns:
  • Execution ID (clickable for details)
  • Timestamp
  • Prompt ID  
  • Success Status (✅/❌)
  • Response Time (ms)
  • Quality Score
  • Tokens Used
  • Model Used
  • Tags

🔍 Search & Filter:
  • Text search across prompt IDs and responses
  • Date range filtering
  • Success/failure filtering
  • Model type filtering
  • Tag-based filtering
  • Quality score range filtering
```

#### Execution Details
```python
📄 Detailed View (click any execution):
  • Full prompt text
  • Complete response text  
  • All metadata fields
  • Error details (if failed)
  • Execution timeline
  • Related executions

🏷️ Metadata Display:
  • Model configuration
  • Custom metadata fields
  • Tags and categories
  • Performance metrics
  • Quality assessment details
```

#### Bulk Operations
```python
⚡ Batch Actions:
  • Select multiple executions
  • Bulk export to CSV/JSON
  • Batch reprocessing (if supported)
  • Tag management
  • Quality re-evaluation
```

**Navigation Tips**:
- Use **pagination controls** at bottom for large datasets
- **Sort by any column** by clicking column headers
- **Multi-select** executions using checkboxes
- Use **quick filters** in sidebar for common searches

### 4. Comparison Tab - A/B Testing Interface

**Purpose**: Side-by-side comparison of prompt variants and A/B test results.

**Key Features**:

#### Prompt Comparison Setup
```python
🔧 Configuration:
  • Select prompt IDs to compare (2-5 prompts)
  • Choose comparison metrics
  • Set time range for analysis
  • Apply common filters

📊 Comparison Views:
  • Side-by-side metrics table
  • Performance radar charts
  • Statistical significance tests
  • Trend comparisons
```

#### Statistical Analysis
```python
🧮 Available Tests:
  • T-tests for mean differences
  • Chi-square for success rates
  • Mann-Whitney U (non-parametric)
  • Effect size calculations (Cohen's d)

📈 Results Display:
  • P-values and significance indicators
  • Confidence intervals
  • Effect size interpretations
  • Practical significance assessment
```

#### Visual Comparisons
```python
📊 Chart Types:
  • Box plots (distribution comparison)
  • Violin plots (detailed distributions)
  • Bar charts (metric comparisons)
  • Scatter plots (correlation analysis)
  • Heatmaps (multi-dimensional comparison)
```

**Using the Comparison Tool**:

1. **Select Prompts**: Choose 2-5 prompt IDs from dropdown
2. **Configure Analysis**: Set time range and metrics to compare
3. **Review Statistics**: Examine statistical test results
4. **Explore Visuals**: Use interactive charts for deeper insights
5. **Export Results**: Save comparison reports for documentation

### 5. Quality Tab - Quality Assessment Dashboard

**Purpose**: Focus on response quality metrics and improvement opportunities.

**Key Sections**:

#### Quality Overview
```python
🎯 Quality Metrics:
  • Overall quality distribution
  • Quality trends over time
  • Model-specific quality comparison
  • Tag/category quality breakdown

📊 Quality Insights:
  • Top performing prompts
  • Quality improvement opportunities
  • Correlation with other metrics
  • Quality consistency analysis
```

#### Custom Quality Metrics
```python
🔧 Domain-Specific Assessment:
  • Research quality (citations, methodology)
  • Summarization quality (coverage, conciseness)
  • Code quality (correctness, efficiency)
  • Custom metric definitions

📈 Quality Analytics:
  • Metric correlation analysis
  • Quality prediction modeling
  • Improvement trend tracking
  • Comparative quality assessment
```

### 6. Export Tab - Data Export & Reporting

**Purpose**: Generate reports and export data in various formats.

**Export Options**:

#### Data Formats
```python
📁 Available Formats:
  • CSV: Tabular data for spreadsheet analysis
  • JSON: Structured data for programmatic use
  • Excel: Multiple sheets with charts
  • PDF: Formatted reports with visualizations

🎯 Export Scopes:
  • All data (complete dataset)
  • Filtered data (current view)
  • Selected executions
  • Date range specific
  • Prompt-specific datasets
```

#### Report Types
```python
📋 Pre-built Reports:
  • Executive Summary (high-level KPIs)
  • Performance Report (detailed analytics)
  • Quality Assessment (quality metrics)
  • Error Analysis (failure patterns)
  • A/B Test Results (comparison studies)

🎨 Custom Reports:
  • Choose specific metrics
  • Select visualization types
  • Configure time ranges
  • Add custom annotations
```

#### Scheduled Exports
```python
⏰ Automation Options:
  • Daily summary reports
  • Weekly performance reports
  • Monthly trend analysis
  • Alert-based exports (on thresholds)

📧 Delivery Methods:
  • Email attachments
  • Shared folder exports
  • API endpoint posting
  • Cloud storage integration
```

## 🎛️ Advanced Dashboard Features

### Custom Filters & Views

Create and save custom dashboard configurations:

```python
# Example: Creating a Custom View
🔍 Custom Filter Setup:
  1. Apply desired filters (date, tags, prompts)
  2. Configure chart preferences
  3. Set refresh intervals
  4. Click "Save View" 
  5. Name your custom view
  6. Access from sidebar "Saved Views"

💾 Saved Views Examples:
  • "Morning Health Check" - Last 8 hours, key metrics
  • "A/B Test Dashboard" - Comparison view, statistical focus
  • "Error Analysis" - Failed executions, diagnostic charts
  • "Quality Review" - Quality metrics, improvement trends
```

### Real-time Monitoring

Set up live monitoring capabilities:

```python
🔴 Live Monitoring Setup:
  • Enable auto-refresh (15-60 second intervals)
  • Configure alert thresholds
  • Set up notification preferences
  • Create monitoring dashboards

⚠️ Alert Conditions:
  • Error rate > 10%
  • Response time > 5 seconds
  • Quality score < 0.5
  • No executions in 1 hour
```

### Dashboard Customization

Personalize your dashboard experience:

```python
🎨 Customization Options:
  • Theme selection (light/dark mode)
  • Chart color schemes
  • Default time ranges
  • Column preferences
  • Metric display formats

⚙️ Configuration File:
  • Save preferences to config
  • Share configurations with team
  • Version control dashboard settings
  • Environment-specific configs
```

## 📊 Dashboard Performance Tips

### Optimizing Load Times

```python
⚡ Performance Best Practices:
  • Use appropriate date ranges (avoid loading all data)
  • Apply filters before loading charts
  • Limit concurrent chart rendering
  • Use caching for repeated queries
  • Configure reasonable refresh intervals

💾 Data Management:
  • Archive old data periodically
  • Use database indexes effectively
  • Implement data pagination
  • Cache expensive calculations
```

### Troubleshooting Common Issues

```python
🐛 Common Problems & Solutions:

1. Dashboard Won't Load:
   • Check database connectivity
   • Verify Streamlit installation
   • Check port availability
   • Review error logs

2. Charts Not Displaying:
   • Verify data availability
   • Check date range filters
   • Confirm metric selections
   • Clear browser cache

3. Performance Issues:
   • Reduce data scope
   • Increase database resources
   • Optimize query performance
   • Use data sampling

4. Export Failures:
   • Check file permissions
   • Verify export directory
   • Reduce data size
   • Check memory usage
```

## 🎯 Best Practices

### Dashboard Navigation

```python
📍 Efficient Navigation:
  • Start with Overview for system health
  • Use Analytics for performance deep-dives
  • Browse Executions for detailed investigation
  • Use Comparison for A/B testing
  • Export data for external analysis

🔄 Regular Monitoring Routine:
  1. Daily: Check Overview tab for health
  2. Weekly: Review Analytics trends
  3. Monthly: Conduct comparative analysis
  4. Quarterly: Generate comprehensive reports
```

### Data Interpretation

```python
📈 Reading the Charts:
  • Look for patterns, not just points
  • Consider seasonality and context
  • Compare relative vs. absolute changes
  • Validate outliers before acting
  • Use statistical significance for decisions

⚠️ Common Interpretation Mistakes:
  • Acting on single data points
  • Ignoring confidence intervals
  • Confusing correlation with causation
  • Over-reacting to normal variations
  • Missing long-term trends
```

### Team Collaboration

```python
👥 Sharing Insights:
  • Use saved views for team consistency
  • Export reports for stakeholder updates
  • Document findings and decisions
  • Share dashboard URLs for real-time viewing
  • Create standard operating procedures

📋 Regular Reviews:
  • Schedule team dashboard reviews
  • Assign monitoring responsibilities
  • Create escalation procedures
  • Document improvement actions
  • Track optimization outcomes
```

## 🚀 Next Steps

1. **CLI Reference Guide** → Master command-line operations and automation
2. **Development Guide** → Extend dashboard with custom features
3. **Advanced Analytics** → Implement machine learning insights
4. **Integration Examples** → Connect with other monitoring systems

---

**🎉 You're now ready to master the dashboard!** Use it to monitor, analyze, and optimize your prompt performance systematically. The visual insights and interactive features will help you make data-driven decisions about your prompt engineering efforts.