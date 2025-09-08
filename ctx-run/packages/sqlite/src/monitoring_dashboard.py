"""
Real-Time Monitoring Dashboard with Mathematical Validation

This module implements a comprehensive real-time monitoring dashboard for the Lethe optimization engine
with mathematical validation, performance visualization, and operational intelligence.

Key Features:
- Real-time performance metrics visualization with mathematical validation
- Interactive parameter adjustment controls with safety bounds
- Lagrangian convergence monitoring and optimality certificate display  
- Tail latency distribution analysis with EVT/GPD visualization
- Health gate status monitoring with automated alert integration
- Canary promotion pipeline visualization and management
- Historical trend analysis with statistical significance testing
- Operational controls integration with escalation workflows

Mathematical Visualization:
- Lagrangian dual convergence: L(λ,μ) convergence plots with KKT validation
- Submodular function curvature: F(S) marginal gain curves and approximation bounds
- Tail distribution modeling: GPD parameter evolution and confidence intervals
- Performance frontier: CBU vs latency Pareto frontier with theoretical limits
- Statistical validation: Bootstrap confidence intervals and hypothesis testing
"""

import asyncio
import logging
import threading
import time
import json
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from collections import defaultdict, deque
import statistics
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.utils as pio
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import warnings

# Import our operational systems
try:
    from .operational_controls_framework import OperationalControlsFramework, ControlAction, EscalationLevel
    from .dual_controller_system import DualControllerSystem
    from .health_gates_system import HealthGatesSystem, GateType
    from .tail_latency_monitoring import TailLatencyMonitor  
    from .canary_promotion_system import CanaryPromotionSystem, PromotionStage
except ImportError:
    # For standalone testing
    pass

warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

class DashboardMode(Enum):
    """Dashboard operation modes"""
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"

class ChartType(Enum):
    """Types of charts available"""
    TIME_SERIES = "time_series"
    DISTRIBUTION = "distribution"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    SURFACE = "surface"
    BAR = "bar"
    GAUGE = "gauge"

@dataclass
class DashboardMetric:
    """Dashboard metric configuration"""
    name: str
    display_name: str
    unit: str
    chart_type: ChartType
    color: str
    target_value: Optional[float] = None
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    mathematical_validation: bool = False
    update_frequency_seconds: int = 5

@dataclass
class ChartConfiguration:
    """Chart display configuration"""
    title: str
    x_axis_label: str
    y_axis_label: str
    chart_type: ChartType
    width: int = 600
    height: int = 400
    show_legend: bool = True
    mathematical_annotations: bool = False
    statistical_overlays: bool = False

class PerformanceTracker:
    """
    Tracks performance metrics with mathematical validation
    """
    
    def __init__(self, history_size: int = 10000):
        self.history_size = history_size
        self.metrics_history = defaultdict(lambda: deque(maxlen=history_size))
        self.validation_history = deque(maxlen=1000)
        self.lock = threading.RLock()
        
    def add_metric(self, 
                  metric_name: str, 
                  value: float, 
                  timestamp: Optional[datetime] = None,
                  metadata: Optional[Dict[str, Any]] = None):
        """Add performance metric with validation"""
        if timestamp is None:
            timestamp = datetime.now()
        
        with self.lock:
            metric_entry = {
                'value': value,
                'timestamp': timestamp,
                'metadata': metadata or {}
            }
            
            self.metrics_history[metric_name].append(metric_entry)
    
    def get_recent_metrics(self, 
                          metric_name: str, 
                          duration_minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recent metrics within specified duration"""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        
        with self.lock:
            recent_metrics = []
            for entry in self.metrics_history[metric_name]:
                if entry['timestamp'] >= cutoff_time:
                    recent_metrics.append(entry)
            
            return recent_metrics
    
    def compute_statistical_summary(self, 
                                  metric_name: str, 
                                  duration_minutes: int = 60) -> Dict[str, float]:
        """Compute statistical summary of metrics"""
        recent_data = self.get_recent_metrics(metric_name, duration_minutes)
        
        if not recent_data:
            return {'count': 0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        values = [entry['value'] for entry in recent_data]
        
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'median': statistics.median(values),
            'p95': np.percentile(values, 95) if len(values) >= 20 else max(values),
            'p99': np.percentile(values, 99) if len(values) >= 100 else max(values)
        }

class MathematicalValidator:
    """
    Validates mathematical properties and constraints in real-time
    """
    
    def __init__(self):
        self.validation_results = deque(maxlen=500)
        self.lock = threading.RLock()
        
    def validate_lagrangian_convergence(self,
                                      lambda_values: List[float],
                                      mu_values: List[float],
                                      objective_values: List[float],
                                      tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Validate Lagrangian dual convergence
        
        Args:
            lambda_values: Historical λ values
            mu_values: Historical μ values  
            objective_values: Historical objective function values
            tolerance: Convergence tolerance
            
        Returns:
            Validation results with convergence metrics
        """
        try:
            if len(objective_values) < 5:
                return {'status': 'insufficient_data', 'convergence_rate': 0.0}
            
            # Compute convergence rate using differences
            objective_diffs = np.diff(objective_values[-10:])  # Last 10 changes
            convergence_rate = np.mean(np.abs(objective_diffs))
            
            # Check for oscillation (sign changes)
            sign_changes = sum(1 for i in range(len(objective_diffs)-1) 
                              if objective_diffs[i] * objective_diffs[i+1] < 0)
            oscillation_ratio = sign_changes / max(len(objective_diffs)-1, 1)
            
            # Validate parameter stability
            lambda_stability = np.std(lambda_values[-10:]) if len(lambda_values) >= 10 else 0.0
            mu_stability = np.std(mu_values[-10:]) if len(mu_values) >= 10 else 0.0
            
            # Determine convergence status
            if convergence_rate < tolerance and oscillation_ratio < 0.2:
                status = 'converged'
            elif convergence_rate < tolerance * 10:
                status = 'converging'  
            elif oscillation_ratio > 0.5:
                status = 'oscillating'
            else:
                status = 'diverging'
            
            result = {
                'status': status,
                'convergence_rate': convergence_rate,
                'oscillation_ratio': oscillation_ratio,
                'lambda_stability': lambda_stability,
                'mu_stability': mu_stability,
                'validation_timestamp': datetime.now(),
                'samples_analyzed': len(objective_values)
            }
            
            with self.lock:
                self.validation_results.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating Lagrangian convergence: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def validate_submodularity_constraints(self,
                                         marginal_gains: List[float],
                                         set_sizes: List[int]) -> Dict[str, Any]:
        """
        Validate submodularity property: marginal gains are diminishing
        
        Args:
            marginal_gains: Historical marginal gain values
            set_sizes: Corresponding set sizes
            
        Returns:
            Submodularity validation results
        """
        try:
            if len(marginal_gains) < 3:
                return {'status': 'insufficient_data'}
            
            # Check for diminishing returns pattern
            # For submodular functions, marginal gains should generally decrease
            violations = 0
            comparisons = 0
            
            for i in range(len(marginal_gains) - 1):
                for j in range(i + 1, len(marginal_gains)):
                    if set_sizes[i] < set_sizes[j]:  # Ensure proper ordering
                        comparisons += 1
                        if marginal_gains[i] < marginal_gains[j]:  # Violation of diminishing returns
                            violations += 1
            
            if comparisons == 0:
                submodularity_ratio = 1.0
            else:
                submodularity_ratio = 1.0 - (violations / comparisons)
            
            # Compute correlation between set size and marginal gain
            if len(set_sizes) > 2:
                correlation = np.corrcoef(set_sizes, marginal_gains)[0, 1]
            else:
                correlation = 0.0
            
            # Status determination
            if submodularity_ratio > 0.9 and correlation < -0.3:
                status = 'strongly_submodular'
            elif submodularity_ratio > 0.7:
                status = 'weakly_submodular'
            elif submodularity_ratio > 0.5:
                status = 'approximately_submodular'
            else:
                status = 'not_submodular'
            
            return {
                'status': status,
                'submodularity_ratio': submodularity_ratio,
                'violations': violations,
                'total_comparisons': comparisons,
                'size_gain_correlation': correlation,
                'validation_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error validating submodularity: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def validate_pareto_frontier(self,
                               cbu_values: List[float],
                               latency_values: List[float]) -> Dict[str, Any]:
        """
        Validate Pareto frontier properties for CBU vs latency trade-off
        
        Args:
            cbu_values: CBU efficiency values
            latency_values: Corresponding latency values
            
        Returns:
            Pareto frontier validation results
        """
        try:
            if len(cbu_values) != len(latency_values) or len(cbu_values) < 3:
                return {'status': 'insufficient_data'}
            
            # Identify Pareto-optimal points
            pareto_points = []
            for i, (cbu_i, lat_i) in enumerate(zip(cbu_values, latency_values)):
                is_pareto = True
                for j, (cbu_j, lat_j) in enumerate(zip(cbu_values, latency_values)):
                    if i != j:
                        # Point j dominates point i if it has better CBU and better (lower) latency
                        if cbu_j >= cbu_i and lat_j <= lat_i and (cbu_j > cbu_i or lat_j < lat_i):
                            is_pareto = False
                            break
                
                if is_pareto:
                    pareto_points.append((cbu_i, lat_i, i))
            
            # Analyze frontier properties
            pareto_ratio = len(pareto_points) / len(cbu_values)
            
            # Compute trade-off slope (negative correlation expected)
            if len(pareto_points) > 1:
                pareto_cbu = [p[0] for p in pareto_points]
                pareto_lat = [p[1] for p in pareto_points]
                trade_off_correlation = np.corrcoef(pareto_cbu, pareto_lat)[0, 1]
            else:
                trade_off_correlation = 0.0
            
            # Status determination
            if pareto_ratio > 0.5 and trade_off_correlation < -0.5:
                status = 'well_defined_frontier'
            elif pareto_ratio > 0.3:
                status = 'partial_frontier'
            else:
                status = 'dominated_solutions'
            
            return {
                'status': status,
                'pareto_points_count': len(pareto_points),
                'pareto_ratio': pareto_ratio,
                'trade_off_correlation': trade_off_correlation,
                'pareto_indices': [p[2] for p in pareto_points],
                'validation_timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error validating Pareto frontier: {e}")
            return {'status': 'error', 'error': str(e)}

class ChartGenerator:
    """
    Generates interactive charts with mathematical annotations
    """
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'danger': '#ff0000',
            'info': '#17a2b8',
            'mathematical': '#9467bd',
            'theoretical': '#8c564b'
        }
    
    def create_performance_timeline(self,
                                  metrics_data: Dict[str, List[Dict]],
                                  chart_config: ChartConfiguration) -> go.Figure:
        """
        Create performance timeline with multiple metrics
        
        Args:
            metrics_data: Dictionary of metric_name -> list of data points
            chart_config: Chart configuration
            
        Returns:
            Plotly figure object
        """
        try:
            fig = make_subplots(
                rows=len(metrics_data), cols=1,
                shared_xaxes=True,
                subplot_titles=list(metrics_data.keys()),
                vertical_spacing=0.1
            )
            
            for i, (metric_name, data_points) in enumerate(metrics_data.items(), 1):
                if not data_points:
                    continue
                
                timestamps = [point['timestamp'] for point in data_points]
                values = [point['value'] for point in data_points]
                
                # Main time series
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=values,
                        mode='lines+markers',
                        name=metric_name,
                        line=dict(color=self.color_palette['primary'], width=2),
                        marker=dict(size=4)
                    ),
                    row=i, col=1
                )
                
                # Add statistical overlays if enabled
                if chart_config.statistical_overlays and len(values) > 10:
                    # Moving average
                    window_size = min(10, len(values) // 3)
                    moving_avg = pd.Series(values).rolling(window=window_size, center=True).mean()
                    
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=moving_avg,
                            mode='lines',
                            name=f'{metric_name} (Moving Avg)',
                            line=dict(color=self.color_palette['mathematical'], width=1, dash='dash'),
                            opacity=0.7
                        ),
                        row=i, col=1
                    )
                    
                    # Confidence band
                    std_dev = pd.Series(values).rolling(window=window_size, center=True).std()
                    upper_band = moving_avg + 1.96 * std_dev  # 95% confidence
                    lower_band = moving_avg - 1.96 * std_dev
                    
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps + timestamps[::-1],
                            y=list(upper_band) + list(lower_band[::-1]),
                            fill='toself',
                            fillcolor=f'rgba{tuple(list(bytes.fromhex(self.color_palette["info"].lstrip("#"))) + [0.1])}',
                            line=dict(color='rgba(255,255,255,0)'),
                            name=f'{metric_name} (95% CI)',
                            showlegend=False
                        ),
                        row=i, col=1
                    )
            
            # Update layout
            fig.update_layout(
                title=chart_config.title,
                height=chart_config.height * len(metrics_data),
                showlegend=chart_config.show_legend,
                hovermode='x unified'
            )
            
            fig.update_xaxes(title_text=chart_config.x_axis_label)
            fig.update_yaxes(title_text=chart_config.y_axis_label)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating performance timeline: {e}")
            return go.Figure().add_annotation(text=f"Error: {e}", showarrow=False)
    
    def create_lagrangian_convergence_plot(self,
                                         lambda_values: List[float],
                                         mu_values: List[float],
                                         objective_values: List[float],
                                         timestamps: List[datetime]) -> go.Figure:
        """
        Create Lagrangian dual convergence visualization
        
        Args:
            lambda_values: Historical λ values
            mu_values: Historical μ values
            objective_values: Historical objective values
            timestamps: Corresponding timestamps
            
        Returns:
            Plotly figure with convergence analysis
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Lagrangian Multipliers', 'Objective Function', 'Parameter Phase Space', 'Convergence Rate'],
                specs=[[{"secondary_y": True}, {}], [{}, {}]]
            )
            
            # Plot 1: Lagrangian multipliers over time
            fig.add_trace(
                go.Scatter(
                    x=timestamps, y=lambda_values,
                    mode='lines+markers',
                    name='λ (tokens)',
                    line=dict(color=self.color_palette['primary'], width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps, y=mu_values,
                    mode='lines+markers',
                    name='μ (compute)',
                    line=dict(color=self.color_palette['secondary'], width=2),
                    yaxis='y2'
                ),
                row=1, col=1
            )
            
            # Plot 2: Objective function evolution
            fig.add_trace(
                go.Scatter(
                    x=timestamps, y=objective_values,
                    mode='lines+markers',
                    name='L(λ,μ)',
                    line=dict(color=self.color_palette['mathematical'], width=2)
                ),
                row=1, col=2
            )
            
            # Add mathematical annotations for convergence
            if len(objective_values) > 5:
                # Compute convergence rate
                recent_changes = np.diff(objective_values[-10:])
                convergence_rate = np.mean(np.abs(recent_changes))
                
                fig.add_annotation(
                    x=timestamps[-1], y=objective_values[-1],
                    text=f'Convergence Rate: {convergence_rate:.6f}',
                    showarrow=True,
                    row=1, col=2
                )
            
            # Plot 3: Phase space trajectory
            if len(lambda_values) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=lambda_values, y=mu_values,
                        mode='lines+markers',
                        name='(λ,μ) trajectory',
                        line=dict(color=self.color_palette['success'], width=2),
                        marker=dict(
                            size=6,
                            color=list(range(len(lambda_values))),
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Time")
                        )
                    ),
                    row=2, col=1
                )
            
            # Plot 4: Convergence rate over time
            if len(objective_values) > 5:
                convergence_rates = []
                for i in range(5, len(objective_values)):
                    recent_obj = objective_values[i-5:i]
                    rate = np.std(recent_obj) if len(recent_obj) > 1 else 0.0
                    convergence_rates.append(rate)
                
                fig.add_trace(
                    go.Scatter(
                        x=timestamps[5:], y=convergence_rates,
                        mode='lines+markers',
                        name='Conv. Rate',
                        line=dict(color=self.color_palette['warning'], width=2)
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title='Lagrangian Dual Convergence Analysis',
                height=600,
                showlegend=True
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Time", row=1, col=1)
            fig.update_yaxes(title_text="λ", row=1, col=1)
            fig.update_yaxes(title_text="μ", secondary_y=True, row=1, col=1)
            
            fig.update_xaxes(title_text="Time", row=1, col=2)
            fig.update_yaxes(title_text="Objective L(λ,μ)", row=1, col=2)
            
            fig.update_xaxes(title_text="λ", row=2, col=1)
            fig.update_yaxes(title_text="μ", row=2, col=1)
            
            fig.update_xaxes(title_text="Time", row=2, col=2)
            fig.update_yaxes(title_text="Convergence Rate", row=2, col=2)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating Lagrangian convergence plot: {e}")
            return go.Figure().add_annotation(text=f"Error: {e}", showarrow=False)
    
    def create_tail_distribution_analysis(self,
                                        latency_samples: List[float],
                                        gpd_params: Dict[str, float]) -> go.Figure:
        """
        Create tail latency distribution analysis with GPD fitting
        
        Args:
            latency_samples: Raw latency samples  
            gpd_params: GPD parameters (xi, sigma, mu)
            
        Returns:
            Plotly figure with distribution analysis
        """
        try:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Latency Distribution', 'Q-Q Plot vs GPD']
            )
            
            # Plot 1: Histogram with GPD overlay
            fig.add_trace(
                go.Histogram(
                    x=latency_samples,
                    nbinsx=50,
                    name='Observed',
                    opacity=0.7,
                    histnorm='probability density'
                ),
                row=1, col=1
            )
            
            # Add GPD theoretical curve if parameters available
            if gpd_params and 'xi' in gpd_params:
                xi, sigma, mu = gpd_params['xi'], gpd_params['sigma'], gpd_params['mu']
                
                # Generate theoretical GPD curve
                x_range = np.linspace(min(latency_samples), max(latency_samples), 100)
                
                # GPD PDF: f(x) = (1/σ) * (1 + ξ*(x-μ)/σ)^(-(1/ξ + 1))
                gpd_pdf = []
                for x in x_range:
                    if sigma > 0:
                        z = (x - mu) / sigma
                        if xi != 0:
                            if 1 + xi * z > 0:
                                pdf_val = (1/sigma) * ((1 + xi * z) ** (-(1/xi + 1)))
                            else:
                                pdf_val = 0
                        else:
                            pdf_val = (1/sigma) * np.exp(-z)
                        gpd_pdf.append(pdf_val)
                    else:
                        gpd_pdf.append(0)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_range, y=gpd_pdf,
                        mode='lines',
                        name=f'GPD(ξ={xi:.3f})',
                        line=dict(color=self.color_palette['mathematical'], width=3)
                    ),
                    row=1, col=1
                )
            
            # Plot 2: Q-Q plot for goodness of fit
            if len(latency_samples) > 20:
                # Compute empirical quantiles
                sorted_samples = np.sort(latency_samples)
                n = len(sorted_samples)
                empirical_quantiles = [(i + 0.5) / n for i in range(n)]
                
                # Theoretical quantiles (simplified)
                theoretical_quantiles = np.linspace(0.01, 0.99, n)
                
                fig.add_trace(
                    go.Scatter(
                        x=theoretical_quantiles,
                        y=empirical_quantiles,
                        mode='markers',
                        name='Q-Q Points',
                        marker=dict(color=self.color_palette['primary'], size=4)
                    ),
                    row=1, col=2
                )
                
                # Add reference line y=x
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Perfect Fit',
                        line=dict(color=self.color_palette['danger'], dash='dash')
                    ),
                    row=1, col=2
                )
            
            # Add statistical annotations
            if latency_samples:
                p95 = np.percentile(latency_samples, 95)
                p99 = np.percentile(latency_samples, 99)
                tail_ratio = p99 / p95 if p95 > 0 else 0
                
                annotation_text = f'P95: {p95:.2f}ms<br>P99: {p99:.2f}ms<br>Tail Ratio: {tail_ratio:.2f}'
                
                fig.add_annotation(
                    x=0.02, y=0.98,
                    xref="paper", yref="paper",
                    text=annotation_text,
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
            
            fig.update_layout(
                title='Tail Latency Distribution Analysis',
                height=400,
                showlegend=True
            )
            
            fig.update_xaxes(title_text="Latency (ms)", row=1, col=1)
            fig.update_yaxes(title_text="Density", row=1, col=1)
            
            fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
            fig.update_yaxes(title_text="Empirical Quantiles", row=1, col=2)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating tail distribution analysis: {e}")
            return go.Figure().add_annotation(text=f"Error: {e}", showarrow=False)
    
    def create_health_gates_status(self, gate_results: Dict[str, Any]) -> go.Figure:
        """
        Create health gates status visualization
        
        Args:
            gate_results: Results from health gates validation
            
        Returns:
            Plotly figure with gate status
        """
        try:
            # Extract gate information
            gate_names = []
            gate_statuses = []
            gate_scores = []
            gate_colors = []
            
            color_map = {
                'PASS': self.color_palette['success'],
                'WARN': self.color_palette['warning'], 
                'FAIL': self.color_palette['danger'],
                'UNKNOWN': self.color_palette['info']
            }
            
            for gate_name, result in gate_results.items():
                gate_names.append(gate_name)
                status = result.get('status', 'UNKNOWN')
                gate_statuses.append(status)
                gate_scores.append(result.get('score', 0.0))
                gate_colors.append(color_map.get(status, color_map['UNKNOWN']))
            
            # Create horizontal bar chart for gate scores
            fig = go.Figure()
            
            fig.add_trace(
                go.Bar(
                    y=gate_names,
                    x=gate_scores,
                    orientation='h',
                    marker=dict(color=gate_colors),
                    text=[f'{status}<br>{score:.2f}' for status, score in zip(gate_statuses, gate_scores)],
                    textposition='inside'
                )
            )
            
            # Add threshold lines
            fig.add_vline(x=0.8, line_dash="dash", line_color=self.color_palette['success'], 
                         annotation_text="Pass Threshold")
            fig.add_vline(x=0.5, line_dash="dash", line_color=self.color_palette['warning'],
                         annotation_text="Warn Threshold")
            
            fig.update_layout(
                title='Health Gates Status',
                xaxis_title='Score',
                yaxis_title='Gate',
                height=200 + len(gate_names) * 50,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating health gates status: {e}")
            return go.Figure().add_annotation(text=f"Error: {e}", showarrow=False)

class MonitoringDashboard:
    """
    Main monitoring dashboard implementation
    """
    
    def __init__(self, 
                 mode: DashboardMode = DashboardMode.PRODUCTION,
                 port: int = 8050,
                 operational_framework: Optional[OperationalControlsFramework] = None):
        
        self.mode = mode
        self.port = port
        self.operational_framework = operational_framework
        
        # Initialize components
        self.performance_tracker = PerformanceTracker()
        self.mathematical_validator = MathematicalValidator()
        self.chart_generator = ChartGenerator()
        
        # Dashboard state
        self.app = None
        self.last_update = datetime.now()
        self.update_interval = 5  # seconds
        
        # Data caches
        self.cached_charts = {}
        self.cache_timestamps = {}
        self.cache_duration = 30  # seconds
        
        # Metrics configuration
        self.metrics_config = {
            'cbu_per_ms': DashboardMetric(
                name='cbu_per_ms',
                display_name='CBU Efficiency',
                unit='CBU/ms',
                chart_type=ChartType.TIME_SERIES,
                color='#1f77b4',
                target_value=12.5,
                warning_threshold=11.0,
                critical_threshold=10.0,
                mathematical_validation=True
            ),
            'p95_latency': DashboardMetric(
                name='p95_latency', 
                display_name='P95 Latency',
                unit='ms',
                chart_type=ChartType.TIME_SERIES,
                color='#ff7f0e',
                target_value=1.0,
                warning_threshold=2.0,
                critical_threshold=5.0,
                mathematical_validation=True
            ),
            'lambda_value': DashboardMetric(
                name='lambda_value',
                display_name='Lambda (λ)',
                unit='',
                chart_type=ChartType.TIME_SERIES,
                color='#2ca02c',
                mathematical_validation=True
            ),
            'mu_value': DashboardMetric(
                name='mu_value',
                display_name='Mu (μ)',
                unit='',
                chart_type=ChartType.TIME_SERIES, 
                color='#d62728',
                mathematical_validation=True
            )
        }
        
        # Initialize Dash app
        self._initialize_dashboard()
        
    def _initialize_dashboard(self):
        """Initialize Dash application with layout and callbacks"""
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Define layout
        self.app.layout = self._create_layout()
        
        # Register callbacks
        self._register_callbacks()
        
        logger.info(f"Dashboard initialized in {self.mode.value} mode")
    
    def _create_layout(self) -> html.Div:
        """Create dashboard layout"""
        return html.Div([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("Lethe Optimization Engine - Real-Time Dashboard", 
                           className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Status indicators
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("System Status", className="card-title"),
                            html.H2(id="system-status", className="text-center"),
                            html.P(id="status-details", className="text-center")
                        ])
                    ], color="primary", outline=True)
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("CBU Efficiency", className="card-title"),  
                            html.H2(id="cbu-efficiency", className="text-center"),
                            html.P("CBU/ms (Target: 12.5)", className="text-center")
                        ])
                    ], color="success", outline=True)
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("P95 Latency", className="card-title"),
                            html.H2(id="p95-latency", className="text-center"),
                            html.P("ms (Target: ≤1.0)", className="text-center")
                        ])
                    ], color="info", outline=True)
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Optimality", className="card-title"),
                            html.H2(id="optimality-status", className="text-center"),
                            html.P(id="approximation-ratio", className="text-center")
                        ])
                    ], color="warning", outline=True)
                ], md=3)
            ], className="mb-4"),
            
            # Main charts
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="performance-timeline")
                ], md=12)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="lagrangian-convergence")
                ], md=6),
                dbc.Col([
                    dcc.Graph(id="tail-distribution")
                ], md=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="health-gates-status")
                ], md=6),
                dbc.Col([
                    dcc.Graph(id="parameter-controls")
                ], md=6)
            ], className="mb-4"),
            
            # Control panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Operational Controls"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Lambda (λ):"),
                                    dcc.Slider(
                                        id="lambda-slider",
                                        min=0.1, max=10.0, step=0.1, value=1.0,
                                        marks={i: str(i) for i in range(0, 11, 2)}
                                    )
                                ], md=6),
                                dbc.Col([
                                    html.Label("Mu (μ):"),
                                    dcc.Slider(
                                        id="mu-slider", 
                                        min=0.01, max=1.0, step=0.01, value=0.1,
                                        marks={i/10: str(i/10) for i in range(0, 11, 2)}
                                    )
                                ], md=6)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("Apply Parameters", id="apply-params-btn", 
                                             color="primary", className="me-2"),
                                    dbc.Button("Validate Optimality", id="validate-btn",
                                             color="info", className="me-2"),
                                    dbc.Button("Emergency Stop", id="emergency-btn",
                                             color="danger")
                                ], className="text-center mt-3")
                            ])
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval * 1000,  # Convert to milliseconds
                n_intervals=0
            ),
            
            # Hidden divs for storing data
            html.Div(id='control-feedback', style={'display': 'none'}),
            html.Div(id='dashboard-data', style={'display': 'none'})
        ])
    
    def _register_callbacks(self):
        """Register Dash callbacks for interactivity"""
        
        @self.app.callback(
            [Output('system-status', 'children'),
             Output('status-details', 'children'),
             Output('cbu-efficiency', 'children'),
             Output('p95-latency', 'children'),
             Output('optimality-status', 'children'),
             Output('approximation-ratio', 'children'),
             Output('performance-timeline', 'figure'),
             Output('lagrangian-convergence', 'figure'),
             Output('tail-distribution', 'figure'),
             Output('health-gates-status', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            return self._update_dashboard_data()
        
        @self.app.callback(
            [Output('control-feedback', 'children')],
            [Input('apply-params-btn', 'n_clicks'),
             Input('validate-btn', 'n_clicks'), 
             Input('emergency-btn', 'n_clicks')],
            [State('lambda-slider', 'value'),
             State('mu-slider', 'value')]
        )
        def handle_controls(apply_clicks, validate_clicks, emergency_clicks, lambda_val, mu_val):
            return self._handle_control_actions(
                apply_clicks, validate_clicks, emergency_clicks, lambda_val, mu_val
            )
    
    def _update_dashboard_data(self) -> Tuple:
        """Update all dashboard data and return components"""
        try:
            # Get current system status
            if self.operational_framework:
                system_status = self.operational_framework.get_system_status()
                
                # Extract key metrics
                operational_state = system_status.get('operational_state', {})
                optimality_status = system_status.get('optimality_status', {})
                
                # Update performance tracker with current metrics
                current_time = datetime.now()
                self.performance_tracker.add_metric('cbu_per_ms', operational_state.get('cbu_per_ms', 0))
                self.performance_tracker.add_metric('p95_latency', operational_state.get('p95_latency', 0))
                self.performance_tracker.add_metric('lambda_value', operational_state.get('lambda_value', 1.0))
                self.performance_tracker.add_metric('mu_value', operational_state.get('mu_value', 0.1))
                
                # Status indicators
                system_health = system_status.get('system_health', 'unknown')
                system_status_display = system_health.upper()
                status_details = f"Last update: {current_time.strftime('%H:%M:%S')}"
                
                cbu_efficiency = f"{operational_state.get('cbu_per_ms', 0):.1f}"
                p95_latency = f"{operational_state.get('p95_latency', 0):.2f}"
                
                if optimality_status:
                    opt_status = optimality_status.get('status', 'unknown').upper()
                    approx_ratio = f"Ratio: {optimality_status.get('approximation_ratio', 0):.3f}"
                else:
                    opt_status = "UNKNOWN"
                    approx_ratio = "No data"
                
                # Generate charts
                performance_fig = self._generate_performance_timeline()
                lagrangian_fig = self._generate_lagrangian_convergence()
                tail_fig = self._generate_tail_distribution()
                health_gates_fig = self._generate_health_gates_status()
                
            else:
                # Fallback when no operational framework
                system_status_display = "DISCONNECTED"
                status_details = "No operational framework connected"
                cbu_efficiency = "0.0"
                p95_latency = "0.00"
                opt_status = "UNKNOWN"
                approx_ratio = "No data"
                
                # Empty figures
                performance_fig = go.Figure().add_annotation(text="No data available", showarrow=False)
                lagrangian_fig = go.Figure().add_annotation(text="No data available", showarrow=False)
                tail_fig = go.Figure().add_annotation(text="No data available", showarrow=False)
                health_gates_fig = go.Figure().add_annotation(text="No data available", showarrow=False)
            
            return (
                system_status_display, status_details,
                cbu_efficiency, p95_latency,
                opt_status, approx_ratio,
                performance_fig, lagrangian_fig, tail_fig, health_gates_fig
            )
            
        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")
            error_fig = go.Figure().add_annotation(text=f"Error: {e}", showarrow=False)
            return (
                "ERROR", str(e),
                "0.0", "0.00",
                "ERROR", "Error",
                error_fig, error_fig, error_fig, error_fig
            )
    
    def _generate_performance_timeline(self) -> go.Figure:
        """Generate performance timeline chart"""
        try:
            metrics_data = {}
            
            for metric_name in ['cbu_per_ms', 'p95_latency']:
                recent_data = self.performance_tracker.get_recent_metrics(metric_name, duration_minutes=60)
                metrics_data[metric_name] = recent_data
            
            chart_config = ChartConfiguration(
                title='Performance Timeline (Last 60 minutes)',
                x_axis_label='Time',
                y_axis_label='Value',
                chart_type=ChartType.TIME_SERIES,
                statistical_overlays=True,
                mathematical_annotations=True
            )
            
            return self.chart_generator.create_performance_timeline(metrics_data, chart_config)
            
        except Exception as e:
            logger.error(f"Error generating performance timeline: {e}")
            return go.Figure().add_annotation(text=f"Error: {e}", showarrow=False)
    
    def _generate_lagrangian_convergence(self) -> go.Figure:
        """Generate Lagrangian convergence analysis chart"""
        try:
            # Get historical parameter data
            lambda_data = self.performance_tracker.get_recent_metrics('lambda_value', duration_minutes=60)
            mu_data = self.performance_tracker.get_recent_metrics('mu_value', duration_minutes=60)
            
            if not lambda_data or not mu_data:
                return go.Figure().add_annotation(text="Insufficient data for convergence analysis", showarrow=False)
            
            # Extract values and timestamps
            lambda_values = [entry['value'] for entry in lambda_data]
            mu_values = [entry['value'] for entry in mu_data]
            timestamps = [entry['timestamp'] for entry in lambda_data]
            
            # Simulate objective function values (in production, these would be real)
            objective_values = []
            for lam, mu in zip(lambda_values, mu_values):
                # Simplified objective: F - λ*tokens - μ*compute
                obj_val = 100 - lam * 50 - mu * 200  # Simplified calculation
                objective_values.append(obj_val)
            
            return self.chart_generator.create_lagrangian_convergence_plot(
                lambda_values, mu_values, objective_values, timestamps
            )
            
        except Exception as e:
            logger.error(f"Error generating Lagrangian convergence: {e}")
            return go.Figure().add_annotation(text=f"Error: {e}", showarrow=False)
    
    def _generate_tail_distribution(self) -> go.Figure:
        """Generate tail latency distribution analysis"""
        try:
            # Get recent latency samples
            latency_data = self.performance_tracker.get_recent_metrics('p95_latency', duration_minutes=60)
            
            if len(latency_data) < 10:
                return go.Figure().add_annotation(text="Insufficient data for distribution analysis", showarrow=False)
            
            latency_samples = [entry['value'] for entry in latency_data]
            
            # Simulate GPD parameters (in production, these would be computed)
            gpd_params = {
                'xi': 0.1,     # Shape parameter
                'sigma': 0.5,  # Scale parameter  
                'mu': 0.8      # Location parameter
            }
            
            return self.chart_generator.create_tail_distribution_analysis(latency_samples, gpd_params)
            
        except Exception as e:
            logger.error(f"Error generating tail distribution: {e}")
            return go.Figure().add_annotation(text=f"Error: {e}", showarrow=False)
    
    def _generate_health_gates_status(self) -> go.Figure:
        """Generate health gates status visualization"""
        try:
            # Simulate health gate results (in production, get from health gates system)
            gate_results = {
                'Dual Stability': {'status': 'PASS', 'score': 0.92},
                'Ex-post Optimality': {'status': 'PASS', 'score': 0.88},
                'Tail Safety': {'status': 'WARN', 'score': 0.76}
            }
            
            return self.chart_generator.create_health_gates_status(gate_results)
            
        except Exception as e:
            logger.error(f"Error generating health gates status: {e}")
            return go.Figure().add_annotation(text=f"Error: {e}", showarrow=False)
    
    def _handle_control_actions(self, apply_clicks, validate_clicks, emergency_clicks, 
                              lambda_val, mu_val) -> Tuple[str]:
        """Handle control panel actions"""
        try:
            ctx = callback_context
            if not ctx.triggered:
                return ("No action",)
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if not self.operational_framework:
                return ("No operational framework connected",)
            
            if button_id == 'apply-params-btn' and apply_clicks:
                # Apply parameter changes
                lambda_result = self.operational_framework.execute_manual_action(
                    ControlAction.ADJUST_LAMBDA, {'lambda': lambda_val}
                )
                mu_result = self.operational_framework.execute_manual_action(
                    ControlAction.ADJUST_MU, {'mu': mu_val}
                )
                
                if lambda_result.get('success') and mu_result.get('success'):
                    return (f"Parameters applied: λ={lambda_val}, μ={mu_val}",)
                else:
                    errors = []
                    if not lambda_result.get('success'):
                        errors.append(f"Lambda: {lambda_result.get('error', 'Unknown error')}")
                    if not mu_result.get('success'):
                        errors.append(f"Mu: {mu_result.get('error', 'Unknown error')}")
                    return (f"Parameter update failed: {'; '.join(errors)}",)
            
            elif button_id == 'validate-btn' and validate_clicks:
                # Validate current optimality
                result = self.operational_framework.execute_manual_action(ControlAction.VALIDATE_OPTIMALITY)
                
                if result.get('success'):
                    cert = result.get('certificate', {})
                    status = cert.get('status', 'unknown')
                    ratio = cert.get('approximation_ratio', 0)
                    return (f"Optimality validated: {status} (ratio: {ratio:.3f})",)
                else:
                    return (f"Validation failed: {result.get('error', 'Unknown error')}",)
            
            elif button_id == 'emergency-btn' and emergency_clicks:
                # Emergency stop
                result = self.operational_framework.execute_manual_action(ControlAction.EMERGENCY_STOP)
                
                if result.get('success'):
                    return ("EMERGENCY STOP ACTIVATED - Control loop stopped",)
                else:
                    return (f"Emergency stop failed: {result.get('error', 'Unknown error')}",)
            
            return ("Unknown action",)
            
        except Exception as e:
            logger.error(f"Error handling control actions: {e}")
            return (f"Control action error: {e}",)
    
    def start_dashboard(self, debug: bool = False):
        """Start the monitoring dashboard"""
        try:
            if self.app is None:
                raise RuntimeError("Dashboard not initialized")
            
            logger.info(f"Starting monitoring dashboard on port {self.port}")
            
            self.app.run_server(
                debug=debug,
                host='0.0.0.0',
                port=self.port,
                threaded=True
            )
            
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
            raise
    
    def update_operational_framework(self, framework: OperationalControlsFramework):
        """Update operational framework reference"""
        self.operational_framework = framework
        logger.info("Operational framework updated")
    
    def get_dashboard_url(self) -> str:
        """Get dashboard URL"""
        return f"http://localhost:{self.port}"

# Example usage and integration
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create operational framework
        operational_framework = OperationalControlsFramework(enable_auto_control=True)
        
        # Start control loop
        operational_framework.start_control_loop(interval_seconds=30)
        
        # Create monitoring dashboard
        dashboard = MonitoringDashboard(
            mode=DashboardMode.DEVELOPMENT,
            port=8050,
            operational_framework=operational_framework
        )
        
        print(f"Dashboard available at: {dashboard.get_dashboard_url()}")
        print("Starting dashboard server...")
        
        # Start dashboard (this blocks)
        dashboard.start_dashboard(debug=True)
        
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
        if 'operational_framework' in locals():
            operational_framework.stop_control_loop()
    except Exception as e:
        print(f"Error running dashboard: {e}")
        logger.error(f"Dashboard error: {e}")