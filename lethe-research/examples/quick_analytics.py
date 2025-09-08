#!/usr/bin/env python3
"""
Quick Analytics Example - Quick Start Tutorial

This script demonstrates how to get immediate insights from tracked prompt
executions using the built-in analytics functions.
"""

import json
from datetime import datetime, timedelta
from src.monitoring import get_analytics, get_prompt_tracker

def format_duration(ms):
    """Format milliseconds in a human-readable way."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    else:
        return f"{ms/60000:.1f}min"

def format_percentage(value):
    """Format a decimal as a percentage."""
    if value is None:
        return "N/A"
    return f"{value:.1%}"

def print_section_header(title):
    """Print a formatted section header."""
    print(f"\n{title}")
    print("=" * len(title))

def main():
    """Demonstrate quick analytics capabilities."""
    print("📈 Quick Analytics Example")
    print("=" * 40)
    
    # Get comprehensive analytics
    try:
        analytics = get_analytics()
        print("✅ Analytics data loaded successfully")
    except Exception as e:
        print(f"❌ Error loading analytics: {e}")
        return 1
    
    # Summary Statistics
    print_section_header("📊 Summary Statistics")
    
    summary = analytics.get('summary', {})
    
    print(f"Total Executions: {summary.get('total_executions', 0):,}")
    print(f"Success Rate: {format_percentage(summary.get('success_rate'))}")
    print(f"Average Duration: {format_duration(summary.get('avg_execution_time', 0))}")
    print(f"Total Tokens Used: {summary.get('total_tokens', 0):,}")
    
    if summary.get('total_executions', 0) > 0:
        print(f"Average Quality Score: {summary.get('avg_quality_score', 0):.3f}")
        print(f"Date Range: {summary.get('date_range_start', 'N/A')} to {summary.get('date_range_end', 'N/A')}")
    
    # Recent Activity
    print_section_header("🕒 Recent Activity")
    
    recent_executions = analytics.get('recent_executions', [])
    
    if recent_executions:
        print(f"Showing last {min(5, len(recent_executions))} executions:")
        
        for i, execution in enumerate(recent_executions[:5], 1):
            status_icon = "✅" if execution.get('success', False) else "❌"
            duration = format_duration(execution.get('execution_time_ms', 0))
            quality = execution.get('response_quality_score')
            quality_str = f"{quality:.3f}" if quality is not None else "N/A"
            
            print(f"{i:2d}. {status_icon} {execution.get('prompt_id', 'Unknown')[:30]}")
            print(f"     ⏱️ {duration} | 🎯 Quality: {quality_str} | 🕐 {execution.get('timestamp', 'Unknown')}")
    else:
        print("No recent executions found")
    
    # Performance Metrics
    print_section_header("⚡ Performance Metrics")
    
    performance = analytics.get('performance', {})
    
    if performance:
        print(f"Fastest Execution: {format_duration(performance.get('min_execution_time', 0))}")
        print(f"Slowest Execution: {format_duration(performance.get('max_execution_time', 0))}")
        print(f"Median Duration: {format_duration(performance.get('median_execution_time', 0))}")
        
        percentiles = performance.get('percentiles', {})
        if percentiles:
            print(f"95th Percentile: {format_duration(percentiles.get('p95', 0))}")
            print(f"99th Percentile: {format_duration(percentiles.get('p99', 0))}")
    else:
        print("No performance data available")
    
    # Quality Analysis
    print_section_header("🎯 Quality Analysis")
    
    quality = analytics.get('quality', {})
    
    if quality:
        print(f"Average Quality Score: {quality.get('avg_quality_score', 0):.3f}")
        print(f"Best Quality Score: {quality.get('max_quality_score', 0):.3f}")
        print(f"Worst Quality Score: {quality.get('min_quality_score', 0):.3f}")
        
        quality_distribution = quality.get('quality_distribution', {})
        if quality_distribution:
            print("\nQuality Distribution:")
            for range_label, count in quality_distribution.items():
                percentage = (count / summary.get('total_executions', 1)) * 100
                print(f"  {range_label}: {count} executions ({percentage:.1f}%)")
    else:
        print("No quality data available")
    
    # Error Analysis
    print_section_header("🚨 Error Analysis")
    
    errors = analytics.get('errors', {})
    
    if errors and errors.get('total_errors', 0) > 0:
        print(f"Total Errors: {errors.get('total_errors', 0)}")
        print(f"Error Rate: {format_percentage(errors.get('error_rate'))}")
        
        # Recent errors
        recent_errors = errors.get('recent_errors', [])
        if recent_errors:
            print(f"\nRecent Errors ({len(recent_errors)}):")
            for i, error in enumerate(recent_errors[:3], 1):
                print(f"{i}. {error.get('prompt_id', 'Unknown')}")
                print(f"   💬 {error.get('error_message', 'No message')[:60]}...")
                print(f"   🕐 {error.get('timestamp', 'Unknown')}")
        
        # Error types
        error_types = errors.get('error_types', {})
        if error_types:
            print(f"\nError Types:")
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {error_type}: {count} occurrences")
    else:
        print("No errors found - great job! 🎉")
    
    # Tag Analysis
    print_section_header("🏷️ Tag Analysis")
    
    tags = analytics.get('tags', {})
    
    if tags:
        print("Most Common Tags:")
        sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
        for tag, count in sorted_tags[:5]:
            percentage = (count / summary.get('total_executions', 1)) * 100
            print(f"  {tag}: {count} executions ({percentage:.1f}%)")
    else:
        print("No tags found")
    
    # Model Usage
    print_section_header("🤖 Model Usage")
    
    models = analytics.get('models', {})
    
    if models:
        print("Model Distribution:")
        sorted_models = sorted(models.items(), key=lambda x: x[1], reverse=True)
        for model, count in sorted_models:
            percentage = (count / summary.get('total_executions', 1)) * 100
            print(f"  {model}: {count} executions ({percentage:.1f}%)")
    else:
        print("No model data available")
    
    # Advanced Analytics
    print_section_header("📊 Advanced Insights")
    
    # Get direct access to tracker for custom analytics
    tracker = get_prompt_tracker()
    
    try:
        # Custom query: executions by hour of day
        all_executions = tracker.get_recent_executions(limit=100)
        
        if all_executions:
            # Group by hour of day
            hour_distribution = {}
            for execution in all_executions:
                try:
                    # Parse timestamp and extract hour
                    timestamp = execution.get('timestamp', '')
                    if timestamp:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        hour = dt.hour
                        hour_distribution[hour] = hour_distribution.get(hour, 0) + 1
                except Exception:
                    continue
            
            if hour_distribution:
                print("Activity by Hour of Day:")
                for hour in sorted(hour_distribution.keys()):
                    count = hour_distribution[hour]
                    bar = "█" * min(20, count)
                    print(f"  {hour:02d}:00 |{bar:<20}| {count}")
        
        # Performance trends
        if len(all_executions) >= 10:
            recent_10 = all_executions[:10]
            older_10 = all_executions[-10:] if len(all_executions) >= 20 else []
            
            if older_10:
                recent_avg = sum(e.get('execution_time_ms', 0) for e in recent_10) / len(recent_10)
                older_avg = sum(e.get('execution_time_ms', 0) for e in older_10) / len(older_10)
                
                trend = "📈" if recent_avg > older_avg else "📉"
                change = abs(recent_avg - older_avg) / older_avg * 100 if older_avg > 0 else 0
                
                print(f"\nPerformance Trend {trend}:")
                print(f"  Recent 10 avg: {format_duration(recent_avg)}")
                print(f"  Older 10 avg: {format_duration(older_avg)}")
                print(f"  Change: {change:.1f}%")
        
    except Exception as e:
        print(f"⚠️ Could not generate advanced insights: {e}")
    
    # Export Options
    print_section_header("📤 Export Options")
    
    print("Available export formats:")
    print("  1. JSON: python scripts/prompt_monitor.py export --format json")
    print("  2. CSV: python scripts/prompt_monitor.py export --format csv")
    print("  3. Dashboard: python scripts/prompt_monitor.py dashboard")
    
    # Save analytics to file
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analytics_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(analytics, f, indent=2, default=str)
        
        print(f"\n💾 Analytics report saved to: {filename}")
    except Exception as e:
        print(f"⚠️ Could not save report: {e}")
    
    print(f"\n🎉 Analytics analysis complete!")
    print(f"💡 Pro Tips:")
    print(f"  - Use tags to organize and filter your prompt experiments")
    print(f"  - Monitor error rates to identify problematic patterns")
    print(f"  - Track quality scores to measure prompt improvements")
    print(f"  - Use the dashboard for interactive exploration")

if __name__ == "__main__":
    main()