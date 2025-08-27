#!/usr/bin/env python3
"""
Lethe Prompt Monitoring CLI

Command-line interface for managing and analyzing prompt executions.
Provides easy access to monitoring capabilities for research workflows.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.monitoring.prompt_tracker import PromptTracker, get_prompt_tracker
from src.monitoring.dashboard import PromptDashboard
from src.monitoring.integration_examples import LethePromptMonitor


class PromptMonitorCLI:
    """Command-line interface for prompt monitoring operations."""
    
    def __init__(self):
        """Initialize CLI with tracker instances."""
        self.tracker = get_prompt_tracker()
        self.dashboard = PromptDashboard()
        self.monitor = LethePromptMonitor()
    
    def status(self, args) -> None:
        """Display monitoring system status."""
        print("🔍 Lethe Prompt Monitoring Status")
        print("=" * 50)
        
        stats = self.dashboard.get_summary_stats()
        
        print(f"📊 Total Executions: {stats['total_executions']}")
        print(f"🎯 Unique Prompts: {stats['unique_prompts']}")
        print(f"✅ Success Rate: {stats['success_rate']:.1f}%")
        print(f"⚡ Avg Response Time: {stats['avg_execution_time_ms']:.0f}ms")
        print(f"🕐 Recent Activity (24h): {stats['recent_executions_24h']} executions")
        
        print(f"\n💾 Database: {self.tracker.db_path}")
        print(f"📁 Database Size: {self.tracker.db_path.stat().st_size / 1024:.1f} KB")
    
    def list_prompts(self, args) -> None:
        """List all tracked prompts with summary statistics."""
        print("📋 Tracked Prompts")
        print("=" * 80)
        
        perf_df = self.dashboard.get_prompt_performance(args.limit)
        
        if perf_df.empty:
            print("No prompts found in the database.")
            return
        
        print(f"{'Prompt ID':<30} {'Executions':<12} {'Avg Time':<12} {'Success Rate':<12} {'Last Used':<20}")
        print("-" * 80)
        
        for _, row in perf_df.iterrows():
            print(f"{row['prompt_id']:<30} {row['execution_count']:<12} "
                  f"{row['avg_execution_time']:.0f}ms{'':<7} {row['success_rate']:.1f}%{'':<7} "
                  f"{row['last_used'][:19]:<20}")
    
    def analyze_prompt(self, args) -> None:
        """Analyze a specific prompt in detail."""
        prompt_id = args.prompt_id
        print(f"🔍 Analyzing Prompt: {prompt_id}")
        print("=" * 60)
        
        analytics = self.tracker.get_prompt_analytics(prompt_id)
        
        if "error" in analytics:
            print(f"❌ Error: {analytics['error']}")
            return
        
        print(f"📊 Total Executions: {analytics['total_executions']}")
        print(f"✅ Success Rate: {analytics['success_rate']:.1f}%")
        print(f"⚡ Average Response Time: {analytics['avg_execution_time_ms']:.1f}ms")
        print(f"📝 Average Response Length: {analytics['avg_response_length']:.0f} chars")
        print(f"💾 Average Memory Usage: {analytics['memory_usage_avg']:.1f} MB")
        print(f"📈 Performance Trend: {analytics['performance_trend']}")
        
        if analytics['quality_trend']:
            print(f"🎯 Quality Trend: {analytics['quality_trend']}")
        
        print(f"🕐 Latest Execution: {analytics['latest_execution']}")
        
        # Show execution history
        if args.verbose:
            print(f"\n📜 Execution History:")
            history = self.tracker.get_prompt_history(prompt_id)
            print(history.to_string(index=False))
    
    def compare_executions(self, args) -> None:
        """Compare two prompt executions."""
        baseline_id = args.baseline_id
        treatment_id = args.treatment_id
        
        print(f"⚖️ Comparing Executions")
        print(f"Baseline: {baseline_id}")
        print(f"Treatment: {treatment_id}")
        print("=" * 60)
        
        try:
            comparison = self.tracker.compare_executions(
                baseline_id, treatment_id, 
                notes=args.notes or f"CLI comparison at {datetime.now()}"
            )
            
            print(f"🆔 Comparison ID: {comparison.comparison_id}")
            
            if comparison.quality_improvement is not None:
                print(f"🎯 Quality Change: {comparison.quality_improvement:+.3f}")
            
            if comparison.performance_change_percent is not None:
                print(f"⚡ Performance Change: {comparison.performance_change_percent:+.1f}%")
            
            if comparison.length_change_percent is not None:
                print(f"📝 Length Change: {comparison.length_change_percent:+.1f}%")
            
            print(f"📊 Statistical Significance: {'Yes' if comparison.is_significant else 'No'}")
            
            if comparison.p_value:
                print(f"🧮 P-value: {comparison.p_value:.4f}")
            
        except Exception as e:
            print(f"❌ Error comparing executions: {e}")
    
    def export_data(self, args) -> None:
        """Export prompt tracking data."""
        print(f"📤 Exporting data in {args.format.upper()} format...")
        
        try:
            filename = self.tracker.export_data(args.format)
            print(f"✅ Data exported to: {filename}")
            print(f"📁 File size: {Path(filename).stat().st_size / 1024:.1f} KB")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    def show_execution(self, args) -> None:
        """Show detailed information about a specific execution."""
        execution_id = args.execution_id
        
        print(f"🔍 Execution Details: {execution_id}")
        print("=" * 80)
        
        comparison = self.dashboard.get_before_after_comparison(execution_id)
        
        if "error" in comparison:
            print(f"❌ Error: {comparison['error']}")
            return
        
        exec_data = comparison["current_execution"]
        
        # Basic information
        print(f"📝 Prompt ID: {exec_data['prompt_id']}")
        print(f"🏷️ Version: {exec_data['prompt_version']}")
        print(f"🤖 Model: {exec_data['model_name']}")
        print(f"🕐 Timestamp: {exec_data['timestamp']}")
        print(f"⚡ Execution Time: {exec_data['execution_time_ms']:.1f}ms")
        print(f"📏 Response Length: {exec_data['response_length']} chars")
        
        if exec_data['response_quality_score']:
            print(f"🎯 Quality Score: {exec_data['response_quality_score']:.3f}")
        
        if exec_data['memory_usage_mb']:
            print(f"💾 Memory Usage: {exec_data['memory_usage_mb']:.1f} MB")
        
        # Error information
        if exec_data['error_occurred']:
            print(f"❌ Error: {exec_data['error_message']}")
            print(f"🔧 Error Type: {exec_data['error_type']}")
        else:
            print("✅ Execution Successful")
        
        # Changes detected
        if comparison["changes_detected"]:
            print(f"\n🔄 Changes Detected:")
            for change in comparison["changes_detected"]:
                print(f"  • {change}")
        else:
            print(f"\n📊 No significant changes detected")
        
        # Show prompt and response if verbose
        if args.verbose:
            print(f"\n📝 Prompt Text:")
            print("-" * 40)
            print(exec_data['prompt_text'])
            
            if exec_data['response_text']:
                print(f"\n💬 Response Text:")
                print("-" * 40)
                print(exec_data['response_text'][:500] + "..." if len(exec_data['response_text']) > 500 else exec_data['response_text'])
    
    def dashboard(self, args) -> None:
        """Launch the monitoring dashboard."""
        print("🚀 Starting Prompt Monitoring Dashboard...")
        print("Dashboard will be available at: http://localhost:8501")
        print("Press Ctrl+C to stop the dashboard")
        
        try:
            import streamlit as st
            import subprocess
            import sys
            
            # Run the dashboard
            dashboard_path = Path(__file__).parent.parent / "src" / "monitoring" / "dashboard.py"
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                str(dashboard_path), "--server.port", str(args.port)
            ])
            
        except ImportError:
            print("❌ Streamlit not installed. Install with: pip install streamlit plotly")
        except Exception as e:
            print(f"❌ Failed to start dashboard: {e}")
    
    def cleanup(self, args) -> None:
        """Clean up old tracking data."""
        if args.days:
            cutoff_date = (datetime.now() - timedelta(days=args.days)).isoformat()
            print(f"🧹 Cleaning up data older than {args.days} days ({cutoff_date[:10]})")
            
            import sqlite3
            with sqlite3.connect(self.tracker.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM prompt_executions WHERE timestamp < ?",
                    [cutoff_date]
                )
                deleted_count = cursor.rowcount
                
                # Also clean up orphaned comparisons
                conn.execute("""
                    DELETE FROM prompt_comparisons 
                    WHERE baseline_execution_id NOT IN (SELECT execution_id FROM prompt_executions)
                       OR treatment_execution_id NOT IN (SELECT execution_id FROM prompt_executions)
                """)
            
            print(f"✅ Cleaned up {deleted_count} old execution records")
        
        if args.vacuum:
            print("🗜️ Compacting database...")
            import sqlite3
            with sqlite3.connect(self.tracker.db_path) as conn:
                conn.execute("VACUUM")
            print("✅ Database compacted")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Lethe Prompt Monitoring CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status                           # Show monitoring status
  %(prog)s list --limit 10                 # List top 10 prompts
  %(prog)s analyze my_prompt_id --verbose  # Detailed prompt analysis
  %(prog)s compare baseline_id treatment_id # Compare two executions
  %(prog)s show execution_id               # Show execution details
  %(prog)s export --format csv             # Export data to CSV
  %(prog)s dashboard --port 8502           # Launch dashboard on port 8502
  %(prog)s cleanup --days 30 --vacuum      # Clean old data and compact DB
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    subparsers.add_parser('status', help='Show monitoring system status')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all tracked prompts')
    list_parser.add_argument('--limit', type=int, default=20, help='Number of prompts to show')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a specific prompt')
    analyze_parser.add_argument('prompt_id', help='Prompt ID to analyze')
    analyze_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed history')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two executions')
    compare_parser.add_argument('baseline_id', help='Baseline execution ID')
    compare_parser.add_argument('treatment_id', help='Treatment execution ID')
    compare_parser.add_argument('--notes', help='Comparison notes')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show execution details')
    show_parser.add_argument('execution_id', help='Execution ID to show')
    show_parser.add_argument('--verbose', '-v', action='store_true', help='Show prompt and response text')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export tracking data')
    export_parser.add_argument('--format', choices=['csv', 'json', 'parquet'], 
                              default='csv', help='Export format')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch monitoring dashboard')
    dashboard_parser.add_argument('--port', type=int, default=8501, help='Dashboard port')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old data')
    cleanup_parser.add_argument('--days', type=int, help='Delete data older than N days')
    cleanup_parser.add_argument('--vacuum', action='store_true', help='Compact database')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = PromptMonitorCLI()
    
    # Execute the requested command
    command_method = getattr(cli, args.command)
    command_method(args)


if __name__ == "__main__":
    main()