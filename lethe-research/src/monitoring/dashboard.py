#!/usr/bin/env python3
"""
Prompt Monitoring Dashboard

Provides web-based visualization and analytics for prompt execution tracking.
Integrates with existing Lethe monitoring infrastructure.
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

import pandas as pd

# Optional visualization dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from .prompt_tracker import PromptTracker


class PromptDashboard:
    """Interactive dashboard for prompt monitoring and analytics."""
    
    def __init__(self, db_path: str = "experiments/prompt_tracking.db"):
        """Initialize dashboard with database connection."""
        self.db_path = Path(db_path)
        self.tracker = PromptTracker(db_path)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get high-level summary statistics."""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # Total executions
            stats['total_executions'] = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM prompt_executions", conn
            ).iloc[0]['count']
            
            # Unique prompts
            stats['unique_prompts'] = pd.read_sql_query(
                "SELECT COUNT(DISTINCT prompt_id) as count FROM prompt_executions", conn
            ).iloc[0]['count']
            
            # Success rate
            success_rate = pd.read_sql_query(
                "SELECT AVG(CASE WHEN error_occurred THEN 0 ELSE 1 END) as rate FROM prompt_executions", 
                conn
            ).iloc[0]['rate']
            stats['success_rate'] = success_rate * 100 if success_rate else 0
            
            # Average response time
            avg_time = pd.read_sql_query(
                "SELECT AVG(execution_time_ms) as avg_time FROM prompt_executions WHERE error_occurred = 0", 
                conn
            ).iloc[0]['avg_time']
            stats['avg_execution_time_ms'] = avg_time or 0
            
            # Recent activity (last 24 hours)
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            recent_count = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM prompt_executions WHERE timestamp > ?",
                conn, params=[yesterday]
            ).iloc[0]['count']
            stats['recent_executions_24h'] = recent_count
            
        return stats
    
    def get_execution_timeline(self, days: int = 7) -> pd.DataFrame:
        """Get execution timeline for the last N days."""
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as total_executions,
                    AVG(execution_time_ms) as avg_execution_time,
                    AVG(response_length) as avg_response_length,
                    SUM(CASE WHEN error_occurred THEN 1 ELSE 0 END) as errors
                FROM prompt_executions 
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, conn, params=[start_date])
    
    def get_prompt_performance(self, limit: int = 20) -> pd.DataFrame:
        """Get top prompts by performance metrics."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("""
                SELECT 
                    prompt_id,
                    COUNT(*) as execution_count,
                    AVG(execution_time_ms) as avg_execution_time,
                    AVG(response_length) as avg_response_length,
                    AVG(response_quality_score) as avg_quality_score,
                    MAX(timestamp) as last_used,
                    SUM(CASE WHEN error_occurred THEN 1 ELSE 0 END) as error_count,
                    (1.0 - CAST(SUM(CASE WHEN error_occurred THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*)) * 100 as success_rate
                FROM prompt_executions
                GROUP BY prompt_id
                ORDER BY execution_count DESC
                LIMIT ?
            """, conn, params=[limit])
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Compare performance across different models."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("""
                SELECT 
                    model_name,
                    COUNT(*) as execution_count,
                    AVG(execution_time_ms) as avg_execution_time,
                    AVG(response_length) as avg_response_length,
                    AVG(response_quality_score) as avg_quality_score,
                    SUM(CASE WHEN error_occurred THEN 1 ELSE 0 END) as error_count
                FROM prompt_executions
                GROUP BY model_name
                ORDER BY execution_count DESC
            """, conn)
    
    def create_timeline_chart(self, df: pd.DataFrame):
        """Create execution timeline chart."""
        if not PLOTLY_AVAILABLE:
            return None
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Executions', 'Average Response Time', 
                          'Average Response Length', 'Error Count'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Daily executions
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['total_executions'],
                      mode='lines+markers', name='Executions'),
            row=1, col=1
        )
        
        # Response time
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['avg_execution_time'],
                      mode='lines+markers', name='Avg Time (ms)', line=dict(color='orange')),
            row=1, col=2
        )
        
        # Response length  
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['avg_response_length'],
                      mode='lines+markers', name='Avg Length', line=dict(color='green')),
            row=2, col=1
        )
        
        # Errors
        fig.add_trace(
            go.Bar(x=df['date'], y=df['errors'], name='Errors', marker_color='red'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="Prompt Execution Timeline",
                         showlegend=False)
        return fig
    
    def create_prompt_performance_chart(self, df: pd.DataFrame):
        """Create prompt performance comparison chart."""
        if not PLOTLY_AVAILABLE:
            return None
            
        fig = go.Figure()
        
        # Bubble chart: execution count vs avg time, bubble size = success rate
        fig.add_trace(go.Scatter(
            x=df['avg_execution_time'],
            y=df['execution_count'],
            mode='markers',
            marker=dict(
                size=df['success_rate'],
                color=df['avg_quality_score'],
                colorscale='Viridis',
                showscale=True,
                sizemode='diameter',
                sizeref=2.*max(df['success_rate'])/(40.**2),
                sizemin=4,
                colorbar=dict(title="Avg Quality Score")
            ),
            text=df['prompt_id'],
            hovertemplate='<b>%{text}</b><br>' +
                         'Avg Time: %{x:.1f}ms<br>' +
                         'Executions: %{y}<br>' +
                         'Success Rate: %{marker.size:.1f}%<br>' +
                         '<extra></extra>',
            showlegend=False
        ))
        
        fig.update_layout(
            title="Prompt Performance Overview",
            xaxis_title="Average Execution Time (ms)",
            yaxis_title="Total Executions", 
            height=500
        )
        
        return fig
    
    def create_model_comparison_chart(self, df: pd.DataFrame):
        """Create model performance comparison chart."""
        if not PLOTLY_AVAILABLE:
            return None
            
        fig = go.Figure()
        
        # Bar chart comparing models
        fig.add_trace(go.Bar(
            x=df['model_name'],
            y=df['avg_execution_time'],
            name='Avg Execution Time (ms)',
            yaxis='y',
            marker_color='skyblue'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['model_name'],
            y=df['avg_quality_score'],
            mode='markers+lines',
            name='Avg Quality Score',
            yaxis='y2',
            marker=dict(size=10, color='red')
        ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis=dict(title="Execution Time (ms)", side="left"),
            yaxis2=dict(title="Quality Score", side="right", overlaying="y"),
            height=400
        )
        
        return fig
    
    def get_detailed_execution(self, execution_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific execution."""
        with sqlite3.connect(self.db_path) as conn:
            result = pd.read_sql_query(
                "SELECT * FROM prompt_executions WHERE execution_id = ?",
                conn, params=[execution_id]
            )
            
            if result.empty:
                return {"error": "Execution not found"}
            
            execution = result.iloc[0].to_dict()
            
            # Parse JSON fields
            for field in ['prompt_variables_json', 'model_parameters_json', 'environment_json']:
                if execution[field]:
                    try:
                        execution[field] = json.loads(execution[field])
                    except:
                        pass
            
            return execution
    
    def get_before_after_comparison(self, execution_id: str) -> Dict[str, Any]:
        """Get before/after comparison for an execution."""
        execution = self.get_detailed_execution(execution_id)
        
        if "error" in execution:
            return execution
        
        # Find similar executions (same prompt_id, different versions or parameters)
        with sqlite3.connect(self.db_path) as conn:
            similar = pd.read_sql_query("""
                SELECT execution_id, prompt_version, timestamp, execution_time_ms,
                       response_length, response_quality_score, prompt_hash
                FROM prompt_executions 
                WHERE prompt_id = ? AND execution_id != ?
                ORDER BY timestamp DESC
                LIMIT 5
            """, conn, params=[execution['prompt_id'], execution_id])
        
        comparison = {
            "current_execution": execution,
            "similar_executions": similar.to_dict('records'),
            "changes_detected": []
        }
        
        # Analyze changes
        if not similar.empty:
            latest_similar = similar.iloc[0]
            
            if execution['prompt_hash'] != latest_similar['prompt_hash']:
                comparison["changes_detected"].append("Prompt content changed")
            
            time_change = ((execution['execution_time_ms'] - latest_similar['execution_time_ms']) 
                          / latest_similar['execution_time_ms'] * 100)
            if abs(time_change) > 10:  # 10% threshold
                comparison["changes_detected"].append(
                    f"Execution time changed by {time_change:+.1f}%"
                )
            
            if (execution['response_quality_score'] and 
                latest_similar['response_quality_score']):
                quality_change = (execution['response_quality_score'] - 
                                 latest_similar['response_quality_score'])
                if abs(quality_change) > 0.1:
                    comparison["changes_detected"].append(
                        f"Quality score changed by {quality_change:+.2f}"
                    )
        
        return comparison


def create_streamlit_dashboard():
    """Create Streamlit dashboard for prompt monitoring."""
    if not STREAMLIT_AVAILABLE:
        print("❌ Streamlit not available. Install with: pip install streamlit plotly")
        return
        
    if not PLOTLY_AVAILABLE:
        print("❌ Plotly not available. Install with: pip install plotly")
        return
        
    st.set_page_config(page_title="Lethe Prompt Monitoring", layout="wide")
    
    st.title("🔍 Lethe Prompt Monitoring Dashboard")
    st.markdown("Real-time monitoring and analytics for prompt executions")
    
    dashboard = PromptDashboard()
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    days = st.sidebar.slider("Timeline Days", 1, 30, 7)
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    
    if auto_refresh:
        st.sidebar.markdown("🔄 *Auto-refreshing every 30 seconds*")
        time.sleep(30)
        st.experimental_rerun()
    
    # Summary metrics
    st.header("📊 Summary Statistics")
    stats = dashboard.get_summary_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Executions", stats['total_executions'])
    with col2:
        st.metric("Unique Prompts", stats['unique_prompts'])
    with col3:
        st.metric("Success Rate", f"{stats['success_rate']:.1f}%")
    with col4:
        st.metric("Avg Response Time", f"{stats['avg_execution_time_ms']:.0f}ms")
    
    # Timeline charts
    st.header("📈 Execution Timeline")
    timeline_df = dashboard.get_execution_timeline(days)
    
    if not timeline_df.empty:
        timeline_chart = dashboard.create_timeline_chart(timeline_df)
        st.plotly_chart(timeline_chart, use_container_width=True)
    else:
        st.info("No execution data available for the selected time period.")
    
    # Performance analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("🎯 Prompt Performance")
        perf_df = dashboard.get_prompt_performance()
        if not perf_df.empty:
            perf_chart = dashboard.create_prompt_performance_chart(perf_df)
            st.plotly_chart(perf_chart, use_container_width=True)
            
            st.subheader("Top Performing Prompts")
            st.dataframe(perf_df.head(10), use_container_width=True)
    
    with col2:
        st.header("🤖 Model Comparison")
        model_df = dashboard.get_model_comparison()
        if not model_df.empty:
            model_chart = dashboard.create_model_comparison_chart(model_df)
            st.plotly_chart(model_chart, use_container_width=True)
            
            st.subheader("Model Performance")
            st.dataframe(model_df, use_container_width=True)
    
    # Detailed execution lookup
    st.header("🔍 Execution Details")
    execution_id = st.text_input("Enter Execution ID for detailed analysis:")
    
    if execution_id:
        comparison = dashboard.get_before_after_comparison(execution_id)
        
        if "error" not in comparison:
            st.subheader("Current Execution")
            exec_data = comparison["current_execution"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.json({
                    "execution_id": exec_data["execution_id"],
                    "prompt_id": exec_data["prompt_id"],
                    "model_name": exec_data["model_name"],
                    "execution_time_ms": exec_data["execution_time_ms"],
                    "response_length": exec_data["response_length"],
                    "quality_score": exec_data["response_quality_score"]
                })
            
            with col2:
                st.subheader("Changes Detected")
                for change in comparison["changes_detected"]:
                    st.success(f"✓ {change}")
                
                if not comparison["changes_detected"]:
                    st.info("No significant changes detected")
            
            st.subheader("Prompt Text")
            st.text_area("", exec_data["prompt_text"], height=100, disabled=True)
            
            st.subheader("Response Text")
            st.text_area("", exec_data["response_text"] or "No response recorded", height=100, disabled=True)
            
            if comparison["similar_executions"]:
                st.subheader("Similar Executions")
                similar_df = pd.DataFrame(comparison["similar_executions"])
                st.dataframe(similar_df, use_container_width=True)
        else:
            st.error(comparison["error"])


if __name__ == "__main__":
    create_streamlit_dashboard()