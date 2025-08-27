# Best Practices Guide

Recommended patterns and practices for effective prompt monitoring and optimization.

## 🎯 Core Principles

### 1. Systematic Monitoring
- **Track Everything**: Monitor all prompt interactions, not just successful ones
- **Consistent Metadata**: Use standardized tags and metadata fields across all executions
- **Quality First**: Always include quality assessment, even if simple
- **Context Preservation**: Capture sufficient context for later analysis

### 2. Data Hygiene
- **Sanitize Sensitive Data**: Remove PII and credentials before storage
- **Version Control**: Track prompt versions and model configurations
- **Archive Strategy**: Regularly archive old data to maintain performance
- **Backup Routine**: Implement regular, tested backup procedures

### 3. Performance Optimization
- **Batch Operations**: Use batch processing for high-volume scenarios
- **Efficient Queries**: Optimize database queries with proper indexing
- **Caching Strategy**: Cache expensive analytics calculations
- **Resource Monitoring**: Monitor system resources and scale appropriately

## 📊 Monitoring Strategy

### Prompt Lifecycle Management

```python
# Best Practice: Comprehensive prompt lifecycle tracking
from src.monitoring import track_prompt
from datetime import datetime

def track_prompt_lifecycle(prompt_id, prompt_text, model_config, context=None):
    """Track complete prompt lifecycle with consistent metadata."""
    
    standard_metadata = {
        "user_id": context.get("user_id", "system"),
        "session_id": context.get("session_id", generate_session_id()),
        "experiment_name": context.get("experiment", "default"),
        "version": context.get("version", "1.0"),
        "environment": context.get("env", "development"),
        "tracking_timestamp": datetime.now().isoformat()
    }
    
    tags = [
        context.get("domain", "general"),
        f"env-{standard_metadata['environment']}",
        f"version-{standard_metadata['version']}"
    ]
    
    with track_prompt(
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        model_config=model_config,
        tags=tags,
        metadata=standard_metadata
    ) as execution:
        try:
            # Your LLM processing here
            response = process_llm_request(prompt_text, model_config)
            
            # Quality assessment (always include)
            quality_score = assess_response_quality(prompt_text, response, context)
            
            # Update execution
            execution.response_text = response
            execution.response_quality_score = quality_score
            execution.tokens_used = count_tokens(prompt_text, response)
            execution.success = True
            
            return response
            
        except Exception as e:
            # Error is automatically tracked
            execution.metadata["error_type"] = type(e).__name__
            execution.metadata["error_context"] = str(e)[:200]
            raise
```

### Quality Assessment Standards

```python
# Best Practice: Multi-dimensional quality assessment
class StandardQualityAssessor:
    """Standard quality assessment with consistent metrics."""
    
    def assess_response_quality(self, prompt, response, context=None):
        """Comprehensive quality assessment."""
        
        assessments = {
            "relevance": self.assess_relevance(prompt, response),
            "completeness": self.assess_completeness(prompt, response),
            "accuracy": self.assess_accuracy(response, context),
            "clarity": self.assess_clarity(response),
            "safety": self.assess_safety(response)
        }
        
        # Weighted overall score
        weights = {
            "relevance": 0.3,
            "completeness": 0.25,
            "accuracy": 0.25,
            "clarity": 0.15,
            "safety": 0.05
        }
        
        overall_score = sum(
            score * weights[metric] 
            for metric, score in assessments.items()
        )
        
        return overall_score, assessments
    
    def assess_relevance(self, prompt, response):
        """Assess how well response addresses the prompt."""
        # Implementation based on semantic similarity
        pass
    
    def assess_completeness(self, prompt, response):
        """Assess response completeness."""
        # Implementation based on prompt requirements
        pass
    
    def assess_accuracy(self, response, context):
        """Assess factual accuracy where possible."""
        # Implementation using fact-checking tools
        pass
    
    def assess_clarity(self, response):
        """Assess response clarity and readability."""
        # Implementation using readability metrics
        pass
    
    def assess_safety(self, response):
        """Assess response safety and appropriateness."""
        # Implementation using safety classifiers
        pass
```

### Tagging Strategy

```python
# Best Practice: Hierarchical and consistent tagging
class TaggingStrategy:
    """Standardized tagging for consistent categorization."""
    
    @staticmethod
    def generate_tags(context):
        """Generate consistent tags based on context."""
        
        tags = []
        
        # Domain tags (required)
        domain = context.get("domain", "general")
        tags.append(f"domain-{domain}")
        
        # Task type tags
        task_type = context.get("task_type")
        if task_type:
            tags.append(f"task-{task_type}")
        
        # Model tags
        model = context.get("model_config", {}).get("model", "unknown")
        tags.append(f"model-{model}")
        
        # Environment tags
        env = context.get("environment", "dev")
        tags.append(f"env-{env}")
        
        # Experiment tags
        experiment = context.get("experiment")
        if experiment:
            tags.append(f"exp-{experiment}")
        
        # Version tags
        version = context.get("version")
        if version:
            tags.append(f"v{version}")
        
        # User segment tags
        user_segment = context.get("user_segment")
        if user_segment:
            tags.append(f"segment-{user_segment}")
        
        return tags

# Usage example
def track_with_standard_tags(prompt_id, prompt_text, model_config, context):
    tags = TaggingStrategy.generate_tags(context)
    
    with track_prompt(
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        model_config=model_config,
        tags=tags
    ) as execution:
        # Processing logic here
        pass
```

## 🔄 A/B Testing Best Practices

### Statistical Rigor

```python
# Best Practice: Statistically sound A/B testing
import random
from scipy import stats
from typing import List, Dict, Tuple

class StatisticalABTester:
    """Statistically rigorous A/B testing framework."""
    
    def __init__(self, significance_level=0.05, power=0.8, effect_size=0.1):
        self.significance_level = significance_level
        self.power = power
        self.effect_size = effect_size
    
    def calculate_sample_size(self, baseline_mean, baseline_std):
        """Calculate required sample size for statistical power."""
        
        # Cohen's d for effect size
        cohens_d = self.effect_size
        
        # Two-sample t-test sample size calculation
        from scipy.stats import norm
        z_alpha = norm.ppf(1 - self.significance_level / 2)
        z_beta = norm.ppf(self.power)
        
        n = 2 * ((z_alpha + z_beta) / cohens_d) ** 2
        
        return int(n) + 1
    
    def run_ab_test(self, control_prompt, variant_prompt, test_cases, model_config):
        """Run statistically valid A/B test."""
        
        # Calculate required sample size
        # (This would use historical data in practice)
        required_sample_size = max(30, len(test_cases) // 2)
        
        if len(test_cases) < required_sample_size * 2:
            print(f"⚠️ Warning: Sample size {len(test_cases)} may be too small")
            print(f"   Recommended: {required_sample_size * 2} for statistical power")
        
        # Randomize assignment
        random.shuffle(test_cases)
        control_cases = test_cases[:len(test_cases)//2]
        variant_cases = test_cases[len(test_cases)//2:]
        
        # Run experiments
        control_results = self._run_group(control_prompt, control_cases, model_config, "control")
        variant_results = self._run_group(variant_prompt, variant_cases, model_config, "variant")
        
        # Statistical analysis
        analysis = self._analyze_results(control_results, variant_results)
        
        return analysis
    
    def _analyze_results(self, control_results, variant_results):
        """Perform comprehensive statistical analysis."""
        
        control_scores = [r["quality_score"] for r in control_results]
        variant_scores = [r["quality_score"] for r in variant_results]
        
        # Descriptive statistics
        control_mean = np.mean(control_scores)
        variant_mean = np.mean(variant_scores)
        control_std = np.std(control_scores)
        variant_std = np.std(variant_scores)
        
        # Statistical tests
        t_stat, p_value = stats.ttest_ind(control_scores, variant_scores)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((control_std**2 + variant_std**2) / 2)
        cohens_d = (variant_mean - control_mean) / pooled_std
        
        # Confidence interval for difference
        diff = variant_mean - control_mean
        se_diff = np.sqrt(control_std**2/len(control_scores) + variant_std**2/len(variant_scores))
        ci_lower = diff - 1.96 * se_diff
        ci_upper = diff + 1.96 * se_diff
        
        return {
            "control": {
                "n": len(control_scores),
                "mean": control_mean,
                "std": control_std
            },
            "variant": {
                "n": len(variant_scores),
                "mean": variant_mean,
                "std": variant_std
            },
            "statistical_test": {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < self.significance_level
            },
            "effect_size": {
                "cohens_d": cohens_d,
                "interpretation": self._interpret_effect_size(cohens_d)
            },
            "confidence_interval": {
                "difference": diff,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "ci_excludes_zero": not (ci_lower <= 0 <= ci_upper)
            },
            "recommendation": self._generate_recommendation(p_value, cohens_d, ci_lower, ci_upper)
        }
```

## 🚀 Production Deployment

### Monitoring Architecture

```python
# Best Practice: Production monitoring with circuit breakers
import time
from enum import Enum
from typing import Optional

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Circuit breaker tripped
    HALF_OPEN = "half_open"  # Testing recovery

class ProductionMonitor:
    """Production-ready monitoring with circuit breaker pattern."""
    
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.circuit_state = CircuitState.CLOSED
    
    def track_with_circuit_breaker(self, prompt_id, prompt_text, model_config):
        """Track with circuit breaker for resilience."""
        
        # Check circuit state
        if self.circuit_state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.circuit_state = CircuitState.HALF_OPEN
                self.failure_count = 0
            else:
                # Circuit is open, fail fast
                raise Exception("Circuit breaker is OPEN - monitoring unavailable")
        
        try:
            with track_prompt(
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                model_config=model_config,
                metadata={"circuit_state": self.circuit_state.value}
            ) as execution:
                # Your processing logic
                response = self._process_with_timeout(prompt_text, model_config)
                
                execution.response_text = response
                execution.success = True
                
                # Reset circuit breaker on success
                if self.circuit_state == CircuitState.HALF_OPEN:
                    self.circuit_state = CircuitState.CLOSED
                    self.failure_count = 0
                
                return response
                
        except Exception as e:
            self._handle_failure(e)
            raise
    
    def _handle_failure(self, error):
        """Handle failures and update circuit breaker state."""
        
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.circuit_state = CircuitState.OPEN
            print(f"⚠️ Circuit breaker OPENED after {self.failure_count} failures")
```

### Health Monitoring

```python
# Best Practice: Comprehensive health monitoring
class HealthMonitor:
    """Production health monitoring with alerting."""
    
    def __init__(self):
        self.health_checks = []
        self.alert_thresholds = {
            "error_rate": 0.05,      # 5% error rate
            "response_time_p95": 5000,  # 5 seconds
            "quality_score_avg": 0.7     # 70% quality
        }
    
    def register_health_check(self, name, check_function):
        """Register a health check function."""
        self.health_checks.append((name, check_function))
    
    def run_health_checks(self):
        """Run all registered health checks."""
        
        results = {}
        overall_healthy = True
        
        for name, check_function in self.health_checks:
            try:
                result = check_function()
                results[name] = {
                    "status": "healthy" if result["healthy"] else "unhealthy",
                    "details": result.get("details", {}),
                    "metrics": result.get("metrics", {})
                }
                
                if not result["healthy"]:
                    overall_healthy = False
                    
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e)
                }
                overall_healthy = False
        
        # Check alert thresholds
        alerts = self._check_alert_thresholds(results)
        
        return {
            "overall_healthy": overall_healthy,
            "checks": results,
            "alerts": alerts,
            "timestamp": time.time()
        }
    
    def _check_alert_thresholds(self, results):
        """Check if any metrics exceed alert thresholds."""
        
        alerts = []
        
        # Get recent metrics
        from src.monitoring import get_analytics
        analytics = get_analytics()
        summary = analytics.get("summary", {})
        
        # Check error rate
        error_rate = 1 - summary.get("success_rate", 1.0)
        if error_rate > self.alert_thresholds["error_rate"]:
            alerts.append({
                "type": "error_rate",
                "value": error_rate,
                "threshold": self.alert_thresholds["error_rate"],
                "severity": "high"
            })
        
        # Check response time
        p95_time = summary.get("performance", {}).get("p95", 0)
        if p95_time > self.alert_thresholds["response_time_p95"]:
            alerts.append({
                "type": "response_time_p95",
                "value": p95_time,
                "threshold": self.alert_thresholds["response_time_p95"],
                "severity": "medium"
            })
        
        # Check quality score
        avg_quality = summary.get("avg_quality_score", 1.0)
        if avg_quality < self.alert_thresholds["quality_score_avg"]:
            alerts.append({
                "type": "quality_score_avg",
                "value": avg_quality,
                "threshold": self.alert_thresholds["quality_score_avg"],
                "severity": "medium"
            })
        
        return alerts
```

## 📈 Performance Optimization

### Database Optimization

```sql
-- Best Practice: Database optimization queries
-- Run these periodically for optimal performance

-- Create indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_prompt_executions_timestamp 
    ON prompt_executions(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_prompt_executions_prompt_id_timestamp 
    ON prompt_executions(prompt_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_prompt_executions_success_timestamp 
    ON prompt_executions(success, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_prompt_executions_quality_score 
    ON prompt_executions(response_quality_score) 
    WHERE response_quality_score IS NOT NULL;

-- Analyze table statistics
ANALYZE prompt_executions;

-- Vacuum database (SQLite)
VACUUM;

-- Update statistics
PRAGMA optimize;
```

### Caching Strategy

```python
# Best Practice: Multi-level caching
import functools
import hashlib
import json
import time
from typing import Dict, Any

class CacheManager:
    """Multi-level caching for analytics and queries."""
    
    def __init__(self):
        self.memory_cache = {}
        self.cache_ttl = {}
        self.default_ttl = 300  # 5 minutes
    
    def cache_key(self, *args, **kwargs):
        """Generate consistent cache key."""
        
        cache_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def cached_analytics(self, ttl=None):
        """Decorator for caching analytics functions."""
        
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                key = f"{func.__name__}:{self.cache_key(*args, **kwargs)}"
                now = time.time()
                
                # Check cache
                if key in self.memory_cache:
                    cached_time = self.cache_ttl.get(key, 0)
                    cache_age = now - cached_time
                    max_age = ttl or self.default_ttl
                    
                    if cache_age < max_age:
                        return self.memory_cache[key]
                
                # Compute and cache result
                result = func(*args, **kwargs)
                self.memory_cache[key] = result
                self.cache_ttl[key] = now
                
                return result
            
            return wrapper
        
        return decorator
    
    def clear_cache(self, pattern=None):
        """Clear cache entries matching pattern."""
        
        if pattern is None:
            self.memory_cache.clear()
            self.cache_ttl.clear()
        else:
            keys_to_remove = [
                key for key in self.memory_cache.keys()
                if pattern in key
            ]
            
            for key in keys_to_remove:
                self.memory_cache.pop(key, None)
                self.cache_ttl.pop(key, None)

# Usage
cache_manager = CacheManager()

@cache_manager.cached_analytics(ttl=600)  # 10 minute cache
def get_expensive_analytics(filters):
    """Expensive analytics computation with caching."""
    # Expensive computation here
    pass
```

## 🔒 Security Best Practices

### Data Sanitization

```python
# Best Practice: Comprehensive data sanitization
import re
from typing import Dict, Any

class DataSanitizer:
    """Comprehensive data sanitization for security and privacy."""
    
    def __init__(self):
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
        
        self.sensitive_config_keys = [
            'api_key', 'token', 'secret', 'password', 'credential',
            'auth_token', 'access_token', 'private_key'
        ]
    
    def sanitize_execution_data(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize execution data before storage."""
        
        sanitized = execution_data.copy()
        
        # Sanitize text fields
        if 'prompt_text' in sanitized:
            sanitized['prompt_text'] = self.sanitize_text(sanitized['prompt_text'])
        
        if 'response_text' in sanitized:
            sanitized['response_text'] = self.sanitize_text(sanitized['response_text'])
        
        # Sanitize model configuration
        if 'model_config' in sanitized:
            sanitized['model_config'] = self.sanitize_config(sanitized['model_config'])
        
        # Sanitize metadata
        if 'metadata' in sanitized:
            sanitized['metadata'] = self.sanitize_metadata(sanitized['metadata'])
        
        return sanitized
    
    def sanitize_text(self, text: str) -> str:
        """Remove PII patterns from text."""
        
        sanitized_text = text
        
        for pii_type, pattern in self.pii_patterns.items():
            replacement = f'[{pii_type.upper()}_REDACTED]'
            sanitized_text = re.sub(pattern, replacement, sanitized_text, flags=re.IGNORECASE)
        
        return sanitized_text
    
    def sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive configuration values."""
        
        sanitized_config = config.copy()
        
        for key in list(sanitized_config.keys()):
            if any(sensitive_key in key.lower() for sensitive_key in self.sensitive_config_keys):
                sanitized_config[key] = '[REDACTED]'
        
        return sanitized_config
    
    def sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metadata fields."""
        
        sanitized_metadata = {}
        
        for key, value in metadata.items():
            # Skip potentially sensitive keys
            if any(sensitive_key in key.lower() for sensitive_key in self.sensitive_config_keys):
                sanitized_metadata[key] = '[REDACTED]'
            elif isinstance(value, str):
                sanitized_metadata[key] = self.sanitize_text(value)
            else:
                sanitized_metadata[key] = value
        
        return sanitized_metadata
```

## 📋 Maintenance Practices

### Regular Maintenance Schedule

```bash
#!/bin/bash
# maintenance_schedule.sh - Automated maintenance tasks

# Daily maintenance (run at 2 AM)
if [ "$(date +%H)" = "02" ]; then
    echo "🔄 Running daily maintenance..."
    
    # Health check
    python scripts/prompt_monitor.py health --verbose
    
    # Performance optimization
    python scripts/prompt_monitor.py cleanup --optimize
    
    # Generate daily report
    python scripts/prompt_monitor.py analytics --days 1 --export daily_report.json
fi

# Weekly maintenance (run on Sunday at 3 AM) 
if [ "$(date +%u)" = "7" ] && [ "$(date +%H)" = "03" ]; then
    echo "🔄 Running weekly maintenance..."
    
    # Archive old data (keep 90 days)
    python scripts/prompt_monitor.py cleanup --archive --days 90
    
    # Full database optimization
    python scripts/prompt_monitor.py cleanup --full-maintenance
    
    # Create backup
    python scripts/prompt_monitor.py backup --create weekly_backup_$(date +%Y%m%d).sqlite.gz --compress
    
    # Generate weekly report
    python scripts/prompt_monitor.py analytics --days 7 --detailed --export weekly_report.json
fi

# Monthly maintenance (run on 1st at 4 AM)
if [ "$(date +%d)" = "01" ] && [ "$(date +%H)" = "04" ]; then
    echo "🔄 Running monthly maintenance..."
    
    # Deep archive (keep 1 year)
    python scripts/prompt_monitor.py cleanup --archive --days 365
    
    # Clean old backups (keep 6 months)
    find backups/ -name "*.sqlite.gz" -mtime +180 -delete
    
    # Generate monthly trend analysis
    python scripts/prompt_monitor.py compare --time-periods "$(date -d '2 months ago' +%Y-%m-%d),$(date -d '1 month ago' +%Y-%m-%d)" "$(date -d '1 month ago' +%Y-%m-%d),$(date +%Y-%m-%d)" --statistical --export monthly_comparison.csv
fi
```

### Monitoring Dashboards

```python
# Best Practice: Custom monitoring dashboard
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

def create_executive_dashboard():
    """Executive-level monitoring dashboard."""
    
    st.set_page_config(
        page_title="Prompt Monitoring Executive Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎯 Executive Prompt Monitoring Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    # Get current metrics
    from src.monitoring import get_analytics
    analytics = get_analytics()
    summary = analytics.get("summary", {})
    
    with col1:
        st.metric(
            "Success Rate",
            f"{summary.get('success_rate', 0):.1%}",
            delta=f"+{0.2:.1%}",  # This would be calculated from historical data
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "Avg Response Time",
            f"{summary.get('avg_execution_time', 0):.0f}ms",
            delta=f"-{50:.0f}ms",  # This would be calculated from historical data
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Quality Score",
            f"{summary.get('avg_quality_score', 0):.3f}",
            delta=f"+{0.015:.3f}",  # This would be calculated from historical data
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            "Total Executions",
            f"{summary.get('total_executions', 0):,}",
            delta=f"+{245:,}",  # This would be calculated from historical data
            delta_color="normal"
        )
    
    # Trend charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Success Rate Trend")
        # Implementation would include actual trend data
        
    with col2:
        st.subheader("⚡ Performance Trend")
        # Implementation would include actual performance data
    
    # Alerts section
    st.subheader("🚨 Current Alerts")
    
    # This would check actual alert conditions
    alerts = [
        {"type": "warning", "message": "Quality score dropped 5% in last hour"},
        {"type": "info", "message": "Database size approaching 1GB - consider archiving"}
    ]
    
    if alerts:
        for alert in alerts:
            if alert["type"] == "warning":
                st.warning(alert["message"])
            elif alert["type"] == "error":
                st.error(alert["message"])
            else:
                st.info(alert["message"])
    else:
        st.success("✅ No active alerts - all systems operating normally")
```

---

**🎯 Key Takeaways**
1. **Consistency is Key**: Use standardized patterns across all monitoring
2. **Quality First**: Always prioritize response quality assessment
3. **Statistical Rigor**: Apply proper statistical methods for comparisons
4. **Security Minded**: Sanitize sensitive data and implement proper access controls
5. **Maintenance Focused**: Regular maintenance prevents performance degradation
6. **Monitoring the Monitor**: Track the monitoring system's own health and performance