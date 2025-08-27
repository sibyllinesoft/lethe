# Lethe Prompt Monitoring Tutorials

Comprehensive tutorials and examples for mastering the Lethe Prompt Monitoring System.

## 🎯 Learning Path

### 🚀 Beginner (Start Here)
1. **[Quick Start Tutorial](01-quick-start.md)** - Get up and running in 10 minutes
   - Installation and setup verification
   - Your first prompt tracking
   - Basic dashboard usage
   - CLI basics and troubleshooting

### 🔗 Intermediate (Integration)
2. **[Integration Guide](02-integration-guide.md)** - Connect with existing workflows
   - Lethe workflow integration patterns
   - MLflow experiment tracking setup
   - A/B testing frameworks
   - Production monitoring setup

### 📊 Advanced (Analytics)
3. **[Advanced Usage Guide](03-advanced-usage.md)** - Master analytics and custom metrics
   - Statistical comparison methods
   - Domain-specific quality assessors
   - Performance deep-dive analysis
   - Automated insights and ML patterns

### 🌐 Interface Mastery
4. **[Dashboard Guide](04-dashboard-guide.md)** - Navigate and utilize the web interface
   - Complete feature tour
   - Interactive analytics exploration
   - Export and reporting workflows
   - Customization and optimization

5. **[CLI Reference Guide](05-cli-reference.md)** - Command-line operations and automation
   - Complete command reference
   - Automation scripts and cron jobs
   - Configuration management
   - Troubleshooting and debugging

### 🔧 Expert (Development)
6. **[Development Guide](06-development-guide.md)** - Extend and customize the system
   - Custom quality metrics development
   - Plugin architecture and creation
   - Storage backend integration
   - Production deployment patterns

## 📁 Tutorial Structure

```
tutorials/
├── README.md                 # This file - learning path overview
├── 01-quick-start.md         # Installation and basic usage
├── 02-integration-guide.md   # Workflow and MLflow integration
├── 03-advanced-usage.md      # Analytics and custom metrics
├── 04-dashboard-guide.md     # Web interface mastery
├── 05-cli-reference.md       # Command-line operations
└── 06-development-guide.md   # Extension and customization

examples/
├── verify_setup.py           # Installation verification script
├── first_tracking.py         # Basic tracking examples
├── error_tracking.py         # Error handling patterns
├── quick_analytics.py        # Analytics usage examples
├── lethe_basic_integration.py # Lethe workflow integration
├── statistical_analysis.py   # Advanced statistical methods
├── custom_metrics.py         # Domain-specific quality metrics
├── performance_deep_dive.py  # Performance optimization
├── custom_quality_assessor.py # Custom assessor development
├── plugin_system.py          # Plugin architecture
├── custom_storage.py         # Storage backend examples
├── testing_framework.py     # Testing and validation
└── automation_scripts/       # Ready-to-use automation
    ├── daily_health_check.sh
    ├── weekly_report.sh
    └── monitoring_automation.py

guides/
├── troubleshooting.md        # Common issues and solutions
├── best-practices.md         # Recommended usage patterns
└── migration-guide.md        # Version migration assistance
```

## 🎓 Recommended Learning Sequence

### For Researchers & Data Scientists
```
1. Quick Start (01) → Get familiar with tracking
2. Integration Guide (02) → Connect to workflows  
3. Dashboard Guide (04) → Visual analysis
4. Advanced Usage (03) → Statistical methods
5. CLI Reference (05) → Automation
```

### For Developers & Engineers
```
1. Quick Start (01) → Understand basics
2. CLI Reference (05) → Command-line mastery
3. Development Guide (06) → Extension patterns
4. Integration Guide (02) → Production setup
5. Advanced Usage (03) → Performance optimization
```

### For Team Leads & Managers
```
1. Quick Start (01) → System overview
2. Dashboard Guide (04) → Team visibility
3. Integration Guide (02) → Process integration
4. CLI Reference (05) → Automation setup
5. Best Practices → Team adoption
```

## 🛠️ Prerequisites

### System Requirements
- Python 3.9+ with pip
- SQLite 3.35+ (included with Python)
- 2GB+ available disk space
- Network access for optional integrations

### Python Dependencies
```bash
# Core dependencies (automatically installed)
pip install streamlit plotly pandas numpy scipy

# Optional enhancements
pip install mlflow asyncpg psycopg2-binary  # For MLflow and PostgreSQL
pip install nltk textstat  # For advanced quality metrics
pip install clickhouse-driver  # For ClickHouse analytics
```

### Knowledge Prerequisites
- **Basic**: Python familiarity, command-line basics
- **Intermediate**: Research workflows, statistical concepts
- **Advanced**: Database systems, web development, testing frameworks

## 🎯 Tutorial Objectives

Each tutorial is designed with specific learning outcomes:

### 01-Quick Start
- ✅ Install and verify the monitoring system
- ✅ Track your first prompt execution
- ✅ Navigate the basic dashboard
- ✅ Use essential CLI commands
- ✅ Troubleshoot common setup issues

### 02-Integration Guide  
- ✅ Integrate with existing Lethe workflows
- ✅ Set up MLflow experiment tracking
- ✅ Design and run A/B tests
- ✅ Monitor production environments
- ✅ Handle authentication and security

### 03-Advanced Usage
- ✅ Perform statistical prompt comparisons
- ✅ Create domain-specific quality metrics
- ✅ Analyze performance patterns and trends
- ✅ Implement automated insights
- ✅ Optimize system performance

### 04-Dashboard Guide
- ✅ Navigate all dashboard sections expertly
- ✅ Create custom views and filters
- ✅ Generate and export reports
- ✅ Set up real-time monitoring
- ✅ Customize dashboard preferences

### 05-CLI Reference
- ✅ Master all CLI commands and options
- ✅ Create automation scripts
- ✅ Set up scheduled monitoring
- ✅ Configure system preferences
- ✅ Debug and resolve issues

### 06-Development Guide
- ✅ Understand system architecture
- ✅ Develop custom extensions
- ✅ Create domain-specific plugins
- ✅ Integrate alternative storage backends
- ✅ Deploy in production environments

## 📚 Additional Resources

### Documentation
- **[Main Documentation](../docs/prompt-monitoring-guide.md)** - Comprehensive reference
- **[API Reference](../src/monitoring/)** - Code documentation
- **[Configuration Guide](../guides/configuration.md)** - System setup options

### Example Projects
- **[Basic Integration](../examples/lethe_basic_integration.py)** - Simple workflow integration
- **[Advanced Analytics](../examples/statistical_analysis.py)** - Statistical analysis examples
- **[Custom Extensions](../examples/custom_quality_assessor.py)** - Extension development

### Community & Support
- **Issues**: Report bugs and request features via project issues
- **Discussions**: Ask questions and share experiences
- **Contributing**: Guidelines for contributing improvements

## 🚀 Getting Started

Ready to begin? Choose your path:

**🆕 New to the system?** Start with [Quick Start Tutorial](01-quick-start.md)

**🔗 Need integration?** Jump to [Integration Guide](02-integration-guide.md)  

**📊 Want advanced analytics?** Explore [Advanced Usage Guide](03-advanced-usage.md)

**🎛️ Prefer visual interfaces?** Check out [Dashboard Guide](04-dashboard-guide.md)

**⌨️ Command-line user?** Begin with [CLI Reference Guide](05-cli-reference.md)

**👩‍💻 Developer looking to extend?** Dive into [Development Guide](06-development-guide.md)

---

**💡 Pro Tip**: Each tutorial includes working code examples you can copy and run immediately. All examples are tested and include expected outputs for verification.

**🎯 Success Metrics**: By completing these tutorials, you'll be able to monitor, analyze, and optimize prompt performance systematically, leading to measurably better AI system outcomes.