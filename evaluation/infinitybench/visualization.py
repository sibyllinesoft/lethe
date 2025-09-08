"""
Visualization System for InfinityBench Results
Publication-ready plots showing precision/recall curves and efficiency metrics.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

# Set publication-ready style
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Color scheme for methods
COLORS = {
    'lethe': '#2E86AB',      # Professional blue
    'bm25': '#F24236',       # Distinct red
    'chunking_uniform': '#F6AE2D',   # Warm orange
    'chunking_random': '#F26B5B',    # Coral
    'chunking_first': '#A23B72'      # Plum
}

MARKERS = {
    'lethe': 'o',
    'bm25': 's',
    'chunking_uniform': '^',
    'chunking_random': 'v', 
    'chunking_first': 'D'
}

def plot_precision_recall_curves(
    results_data: Dict[str, Dict[str, Any]], 
    output_path: str,
    title: str = "Precision-Recall Curves",
    show_efficiency: bool = True
) -> None:
    """
    Plot precision-recall curves with optional efficiency overlay.
    
    Args:
        results_data: Dictionary mapping method names to their PR curve data
        output_path: Path to save the plot
        title: Plot title
        show_efficiency: Whether to show efficiency metrics on secondary y-axis
    """
    if show_efficiency:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot 1: Precision-Recall Curves
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.05)
    
    # Plot each method's PR curve
    for method_name, data in results_data.items():
        if 'precision_recall_curves' in data:
            pr_data = data['precision_recall_curves']
            
            # Use interpolated PR curve if available
            if 'interpolated_precision_recall' in data:
                interp_data = data['interpolated_precision_recall']
                recall_points = interp_data['recall_points']
                precision_values = interp_data['interpolated_precision']
            else:
                # Fallback to standard PR curve
                recall_values = pr_data['recall']
                precision_values = pr_data['precision']
                recall_points = recall_values
            
            color = COLORS.get(method_name, '#666666')
            marker = MARKERS.get(method_name, 'o')
            
            ax1.plot(recall_points, precision_values, 
                    color=color, marker=marker, 
                    label=format_method_name(method_name),
                    linewidth=2, markersize=6)
    
    ax1.legend()
    
    # Plot 2: Efficiency Curves (if requested)
    if show_efficiency:
        ax2.set_xlabel('Top-k Results')
        ax2.set_ylabel('Efficiency (%)')
        ax2.set_title('Efficiency vs Results Retrieved')
        ax2.grid(True, alpha=0.3)
        
        for method_name, data in results_data.items():
            if 'precision_recall_curves' in data:
                pr_data = data['precision_recall_curves']
                k_values = pr_data['k_values']
                efficiency_values = [eff * 100 for eff in pr_data['efficiency']]
                
                color = COLORS.get(method_name, '#666666')
                marker = MARKERS.get(method_name, 'o')
                
                ax2.plot(k_values, efficiency_values,
                        color=color, marker=marker,
                        label=format_method_name(method_name),
                        linewidth=2, markersize=6)
        
        ax2.legend()
        ax2.set_xlim(0, max(pr_data['k_values']) + 5)
        ax2.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_dual_axis_efficiency_curves(
    results_data: Dict[str, Dict[str, Any]], 
    output_path: str,
    title: str = "Precision/Recall and Efficiency Analysis"
) -> None:
    """
    Plot precision/recall curves with efficiency on dual y-axis.
    
    This shows both accuracy (precision/recall) and efficiency (waste reduction)
    on the same plot to demonstrate Lethe's advantage.
    """
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Primary axis: Precision at different k values
    ax1.set_xlabel('Top-k Results Retrieved', fontsize=12)
    ax1.set_ylabel('Precision', color='black', fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Secondary axis: Efficiency percentage
    ax2 = ax1.twinx()
    ax2.set_ylabel('Efficiency (% Relevant Results)', color='darkred', fontsize=12)
    
    # Plot precision curves (solid lines)
    for method_name, data in results_data.items():
        if 'precision_recall_curves' in data:
            pr_data = data['precision_recall_curves']
            k_values = pr_data['k_values']
            precision_values = pr_data['precision']
            efficiency_values = [eff * 100 for eff in pr_data['efficiency']]
            
            color = COLORS.get(method_name, '#666666')
            marker = MARKERS.get(method_name, 'o')
            method_label = format_method_name(method_name)
            
            # Precision on primary axis (solid line)
            line1 = ax1.plot(k_values, precision_values,
                           color=color, marker=marker,
                           label=f'{method_label} (Precision)',
                           linewidth=2.5, markersize=7,
                           linestyle='-')
            
            # Efficiency on secondary axis (dashed line, same color)
            line2 = ax2.plot(k_values, efficiency_values,
                           color=color, marker=marker,
                           label=f'{method_label} (Efficiency)',
                           linewidth=2, markersize=5,
                           linestyle='--', alpha=0.8)
    
    # Configure axes
    ax1.set_xlim(0, max(k_values) + 5)
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(0, 105)
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Separate precision and efficiency legends
    precision_lines = [l for l, label in zip(lines1, labels1) if 'Precision' in label]
    efficiency_lines = [l for l, label in zip(lines2, labels2) if 'Efficiency' in label]
    
    # Create legend with custom formatting
    legend_elements = []
    
    # Add method names with line style indicators
    for method_name in results_data.keys():
        color = COLORS.get(method_name, '#666666')
        method_label = format_method_name(method_name)
        
        # Create a proxy artist for the legend
        solid_line = mpatches.Patch(color=color, label=f'{method_label}')
        legend_elements.append(solid_line)
    
    # Add line style legend
    solid_line = plt.Line2D([0], [0], color='black', linewidth=2, linestyle='-', label='Precision')
    dashed_line = plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Efficiency')
    legend_elements.extend([solid_line, dashed_line])
    
    ax1.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(1.15, 0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_efficiency_comparison_bar(
    results_data: Dict[str, Dict[str, Any]], 
    output_path: str,
    k_values: List[int] = [10, 20, 50],
    title: str = "Efficiency Comparison at Different K Values"
) -> None:
    """
    Create bar chart comparing efficiency across methods at key k values.
    """
    methods = list(results_data.keys())
    n_methods = len(methods)
    n_k_values = len(k_values)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up bar positions
    x_pos = np.arange(n_k_values)
    bar_width = 0.15
    
    # Plot bars for each method
    for i, method_name in enumerate(methods):
        method_data = results_data[method_name]
        
        if 'efficiency_metrics' in method_data:
            efficiency_data = method_data['efficiency_metrics']['efficiency_at_k']
            
            # Extract efficiency values for the specified k values
            efficiencies = []
            for k in k_values:
                eff_key = f'k_{k}'
                if eff_key in efficiency_data:
                    efficiencies.append(efficiency_data[eff_key] * 100)
                else:
                    efficiencies.append(0.0)
            
            color = COLORS.get(method_name, '#666666')
            method_label = format_method_name(method_name)
            
            # Plot bars with offset
            bar_pos = x_pos + (i - n_methods/2 + 0.5) * bar_width
            bars = ax.bar(bar_pos, efficiencies, bar_width, 
                         color=color, alpha=0.8, label=method_label)
            
            # Add value labels on bars
            for bar, eff in zip(bars, efficiencies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{eff:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Configure plot
    ax.set_xlabel('Top-k Results')
    ax.set_ylabel('Efficiency (% Relevant)')
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_waste_reduction_analysis(
    results_data: Dict[str, Dict[str, Any]], 
    output_path: str,
    title: str = "Waste Reduction Analysis"
) -> None:
    """
    Create visualization showing waste percentage reduction compared to baselines.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract waste percentage data
    k_values = None
    waste_data = {}
    
    for method_name, data in results_data.items():
        if 'efficiency_metrics' in data:
            efficiency_metrics = data['efficiency_metrics']
            waste_percentages = efficiency_metrics['waste_percentage_at_k']
            
            if k_values is None:
                k_values = [int(k.split('_')[1]) for k in waste_percentages.keys()]
                k_values.sort()
            
            waste_values = []
            for k in k_values:
                waste_key = f'k_{k}'
                if waste_key in waste_percentages:
                    waste_values.append(waste_percentages[waste_key] * 100)
                else:
                    waste_values.append(100.0)
            
            waste_data[method_name] = waste_values
    
    # Plot 1: Waste percentage trends
    ax1.set_xlabel('Top-k Results')
    ax1.set_ylabel('Waste Percentage (%)')
    ax1.set_title('Waste Percentage by Method')
    ax1.grid(True, alpha=0.3)
    
    for method_name, waste_values in waste_data.items():
        color = COLORS.get(method_name, '#666666')
        marker = MARKERS.get(method_name, 'o')
        method_label = format_method_name(method_name)
        
        ax1.plot(k_values, waste_values,
                color=color, marker=marker, 
                label=method_label,
                linewidth=2, markersize=6)
    
    ax1.legend()
    ax1.set_ylim(0, 105)
    
    # Plot 2: Waste reduction compared to BM25 baseline
    if 'bm25' in waste_data and 'lethe' in waste_data:
        ax2.set_xlabel('Top-k Results')
        ax2.set_ylabel('Waste Reduction vs BM25 (%)')
        ax2.set_title('Lethe Waste Reduction vs BM25')
        ax2.grid(True, alpha=0.3)
        
        bm25_waste = waste_data['bm25']
        lethe_waste = waste_data['lethe']
        
        waste_reduction = []
        for bm25_w, lethe_w in zip(bm25_waste, lethe_waste):
            if bm25_w > 0:
                reduction = ((bm25_w - lethe_w) / bm25_w) * 100
                waste_reduction.append(max(0, reduction))  # Ensure non-negative
            else:
                waste_reduction.append(0)
        
        ax2.bar(range(len(k_values)), waste_reduction, 
               color=COLORS['lethe'], alpha=0.7)
        ax2.set_xticks(range(len(k_values)))
        ax2.set_xticklabels([f'k={k}' for k in k_values])
        
        # Add value labels
        for i, reduction in enumerate(waste_reduction):
            ax2.text(i, reduction + 1, f'{reduction:.1f}%', 
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_evaluation_report(
    results_data: Dict[str, Dict[str, Any]], 
    output_dir: str,
    task_name: str = "InfinityBench Evaluation"
) -> Dict[str, str]:
    """
    Create comprehensive visualization report with multiple plots.
    
    Returns:
        Dictionary mapping plot types to their file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plot_files = {}
    
    # 1. Precision-Recall Curves
    pr_path = output_path / f"{task_name.lower().replace(' ', '_')}_precision_recall.png"
    plot_precision_recall_curves(results_data, str(pr_path), 
                                title=f"{task_name}: Precision-Recall Analysis")
    plot_files['precision_recall'] = str(pr_path)
    
    # 2. Dual-axis efficiency plot
    dual_path = output_path / f"{task_name.lower().replace(' ', '_')}_dual_axis.png"
    plot_dual_axis_efficiency_curves(results_data, str(dual_path),
                                   title=f"{task_name}: Precision & Efficiency Analysis")
    plot_files['dual_axis'] = str(dual_path)
    
    # 3. Efficiency comparison bars
    bar_path = output_path / f"{task_name.lower().replace(' ', '_')}_efficiency_bars.png"
    plot_efficiency_comparison_bar(results_data, str(bar_path),
                                  title=f"{task_name}: Efficiency Comparison")
    plot_files['efficiency_bars'] = str(bar_path)
    
    # 4. Waste reduction analysis
    waste_path = output_path / f"{task_name.lower().replace(' ', '_')}_waste_analysis.png"
    plot_waste_reduction_analysis(results_data, str(waste_path),
                                 title=f"{task_name}: Waste Reduction Analysis")
    plot_files['waste_analysis'] = str(waste_path)
    
    # 5. Summary statistics table (save as JSON for external processing)
    summary_path = output_path / f"{task_name.lower().replace(' ', '_')}_summary.json"
    summary_stats = generate_summary_statistics(results_data)
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    plot_files['summary_json'] = str(summary_path)
    
    return plot_files

def generate_summary_statistics(results_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Generate summary statistics for all methods."""
    summary = {
        'methods_compared': list(results_data.keys()),
        'method_summaries': {},
        'comparative_analysis': {}
    }
    
    # Per-method summaries
    for method_name, data in results_data.items():
        method_summary = {
            'method_name': format_method_name(method_name)
        }
        
        if 'summary' in data:
            method_summary.update(data['summary'])
        
        if 'efficiency_metrics' in data:
            eff_metrics = data['efficiency_metrics']
            method_summary.update({
                'overall_efficiency': eff_metrics.get('overall_efficiency', 0.0),
                'overall_waste': eff_metrics.get('overall_waste', 1.0),
                'total_relevant': eff_metrics.get('total_relevant', 0),
                'total_results': eff_metrics.get('total_results', 0)
            })
        
        summary['method_summaries'][method_name] = method_summary
    
    # Comparative analysis
    if 'bm25' in results_data and 'lethe' in results_data:
        bm25_eff = results_data['bm25'].get('efficiency_metrics', {}).get('overall_efficiency', 0.0)
        lethe_eff = results_data['lethe'].get('efficiency_metrics', {}).get('overall_efficiency', 0.0)
        
        if bm25_eff > 0:
            efficiency_improvement = ((lethe_eff - bm25_eff) / bm25_eff) * 100
        else:
            efficiency_improvement = 0.0
        
        summary['comparative_analysis']['lethe_vs_bm25'] = {
            'efficiency_improvement_percent': efficiency_improvement,
            'lethe_efficiency': lethe_eff,
            'bm25_efficiency': bm25_eff
        }
    
    return summary

def format_method_name(method_name: str) -> str:
    """Format method names for display."""
    name_mapping = {
        'lethe': 'Lethe',
        'bm25': 'BM25',
        'chunking_uniform': 'Uniform Chunking',
        'chunking_random': 'Random Chunking', 
        'chunking_first': 'First-N Chunking'
    }
    
    return name_mapping.get(method_name, method_name.replace('_', ' ').title())