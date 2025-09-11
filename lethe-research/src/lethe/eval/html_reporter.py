#!/usr/bin/env python3
"""
HTML Reporter Integration for Lethe Evaluation Pipeline

This module integrates HTML report generation into the main evaluation pipeline,
automatically creating interactive performance reports alongside CSV and JSON outputs.

Integration points:
- Called from postprocess.py after metrics and advantage map generation
- Triggered by --emit-html-report flag
- Outputs performance_report.html in the same directory as other artifacts
"""

import json
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

def generate_performance_html(
    metrics_csv_path: str, 
    advantage_map_path: str, 
    output_dir: str,
    run_id: Optional[str] = None
) -> str:
    """
    Generate HTML performance report for Lethe evaluation results
    
    Args:
        metrics_csv_path: Path to metrics_summary.csv
        advantage_map_path: Path to advantage_map.json  
        output_dir: Output directory for HTML report
        run_id: Optional run identifier for report metadata
        
    Returns:
        str: Path to generated HTML report
        
    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If data is malformed
    """
    
    output_path = Path(output_dir) / "performance_report.html"
    
    try:
        # Validate input files exist
        metrics_path = Path(metrics_csv_path)
        advantage_path = Path(advantage_map_path)
        
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv_path}")
        if not advantage_path.exists():
            raise FileNotFoundError(f"Advantage map not found: {advantage_map_path}")
        
        logger.info(f"Generating HTML performance report from {metrics_path.name} and {advantage_path.name}")
        
        # Load and validate data
        df = pd.read_csv(metrics_path)
        with open(advantage_path, 'r') as f:
            adv_map = json.load(f)
        
        # Validate required structure
        _validate_data_structure(df, adv_map)
        
        # Generate report
        html_content = _generate_html_report_content(df, adv_map, run_id)
        
        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")
        raise

def _validate_data_structure(df: pd.DataFrame, adv_map: Dict[str, Any]) -> None:
    """Validate that input data has required structure"""
    required_columns = ['adapter', 'k_value', 'keep_percentage', 'metric', 'mean']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in metrics CSV: {missing_cols}")
        
    if 'advantage_matrix' not in adv_map:
        raise ValueError("advantage_matrix not found in advantage map JSON")
    
    # Check for k=5 data
    k5_data = df[df["k_value"] == 5.0]
    if k5_data.empty:
        raise ValueError("No k=5 data found for leaderboard generation")
    
    # Check for required budget levels
    required_budgets = {0.08, 0.15, 0.30}
    available_budgets = set(df["keep_percentage"].unique())
    missing_budgets = required_budgets - available_budgets
    if missing_budgets:
        logger.warning(f"Missing budget levels: {missing_budgets}")

def _generate_html_report_content(
    df: pd.DataFrame, 
    adv_map: Dict[str, Any], 
    run_id: Optional[str]
) -> str:
    """Generate complete HTML report content"""
    
    # Extract metadata
    run_id = run_id or adv_map.get("run_id", "unknown")
    timestamp = adv_map.get("timestamp", datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"))
    
    # Prepare leaderboard data
    leaderboards = _prepare_leaderboard_data(df, adv_map)
    
    # Generate HTML sections
    html_content = _get_base_html_template().format(
        run_id=run_id,
        timestamp=timestamp,
        total_adapters=len(adv_map["advantage_matrix"]),
        total_measurements=len(df),
        champions_section=_generate_champions_section(leaderboards),
        metrics_grid=_generate_metrics_grid(adv_map),
        performance_insights=_generate_performance_insights(),
        tab_headers=_generate_tab_headers(),
        tab_content=_generate_tab_content(leaderboards),
        methodology_sample_size=len(df)
    )
    
    return html_content

def _prepare_leaderboard_data(df: pd.DataFrame, adv_map: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Prepare leaderboard data for all budget levels"""
    
    budgets = [0.08, 0.15, 0.30]
    leaderboards = {}
    
    for budget in budgets:
        budget_str = f"{int(budget*100)}%"
        
        # Filter for k=5, score metric, and this budget
        df_budget = df[
            (df["k_value"] == 5.0) & 
            (df["keep_percentage"] == budget) & 
            (df["metric"] == "score")
        ]
        
        adapter_stats = []
        
        for adapter in df_budget["adapter"].unique():
            if str(adapter).startswith("# RUN_ID"):
                continue
                
            # Get score
            adapter_data = df_budget[df_budget["adapter"] == adapter]
            if adapter_data.empty:
                continue
                
            score = adapter_data["mean"].mean()
            
            # Get performance metrics for this adapter and budget
            df_perf = df[
                (df["adapter"] == adapter) & 
                (df["k_value"] == 5.0) & 
                (df["keep_percentage"] == budget)
            ]
            
            # Calculate latency (p95 approximation)
            latency = 0
            latency_data = df_perf[df_perf["metric"] == "response_time_ms"]
            if not latency_data.empty:
                rt_mean = latency_data["mean"].mean()
                rt_std = latency_data["std"].mean()
                latency = rt_mean + 1.65 * rt_std  # p95 approximation
            
            # Get memory usage
            memory = 0
            memory_data = df_perf[df_perf["metric"] == "memory_mb"]
            if not memory_data.empty:
                memory = memory_data["mean"].mean()
            
            # Get advantage from map
            advantage = 0
            if adapter in adv_map["advantage_matrix"]:
                if budget_str in adv_map["advantage_matrix"][adapter]:
                    advantage = adv_map["advantage_matrix"][adapter][budget_str]
            
            adapter_stats.append({
                "adapter": adapter,
                "family": adapter.split(":")[0],
                "score": score,
                "advantage": advantage,
                "latency": latency,
                "memory": memory
            })
        
        # Sort by score descending
        adapter_stats.sort(key=lambda x: x["score"], reverse=True)
        leaderboards[budget_str] = adapter_stats
    
    return leaderboards

def _get_base_html_template() -> str:
    """Return the base HTML template"""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Context Selection Performance Report - {run_id}</title>
    <style>
        :root {{
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --light-bg: #ecf0f1;
            --white: #ffffff;
            --text-dark: #2c3e50;
            --text-light: #7f8c8d;
            --border-color: #bdc3c7;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: var(--text-dark);
            background-color: var(--light-bg);
        }}

        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}

        .header {{
            background: var(--white);
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}

        .header h1 {{
            color: var(--primary-color);
            margin-bottom: 10px;
            font-size: 2.5em;
            font-weight: 300;
        }}

        .header .meta {{
            color: var(--text-light);
            font-size: 1.1em;
        }}

        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .summary-card {{
            background: var(--white);
            padding: 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .summary-card h3 {{
            color: var(--primary-color);
            margin-bottom: 15px;
            font-size: 1.3em;
        }}

        .tabs {{
            display: flex;
            background: var(--white);
            border-radius: 8px 8px 0 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 0;
        }}

        .tab {{
            flex: 1;
            padding: 15px 20px;
            background: var(--white);
            border: none;
            cursor: pointer;
            font-size: 16px;
            font-weight: 500;
            color: var(--text-light);
            transition: all 0.3s ease;
            border-bottom: 3px solid transparent;
        }}

        .tab.active {{
            color: var(--secondary-color);
            border-bottom-color: var(--secondary-color);
        }}

        .tab-content {{
            background: var(--white);
            padding: 30px;
            border-radius: 0 0 8px 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}

        .tab-pane {{
            display: none;
        }}

        .tab-pane.active {{
            display: block;
        }}

        .leaderboard-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}

        .leaderboard-table th,
        .leaderboard-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}

        .leaderboard-table th {{
            background: var(--light-bg);
            font-weight: 600;
            color: var(--primary-color);
        }}

        .rank-badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            font-weight: bold;
            color: white;
            font-size: 0.9em;
        }}

        .rank-1 {{ background: #FFD700; color: #333; }}
        .rank-2 {{ background: #C0C0C0; color: #333; }}
        .rank-3 {{ background: #CD7F32; color: white; }}
        .rank-default {{ background: var(--text-light); }}

        .family-badge {{
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 500;
            color: white;
        }}

        .family-long {{ background: var(--success-color); }}
        .family-rag {{ background: var(--secondary-color); }}
        .family-rerank {{ background: var(--warning-color); }}
        .family-selector {{ background: var(--text-light); }}

        .methodology {{
            background: var(--white);
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: 30px;
        }}

        @media (max-width: 768px) {{
            .container {{ padding: 10px; }}
            .tabs {{ flex-direction: column; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Context Selection Performance Report</h1>
            <div class="meta">
                <strong>Run ID:</strong> {run_id} &nbsp;|&nbsp;
                <strong>Generated:</strong> {timestamp} &nbsp;|&nbsp;
                <strong>Adapters:</strong> {total_adapters} &nbsp;|&nbsp;
                <strong>Measurements:</strong> {total_measurements:,}
            </div>
        </div>

        {champions_section}

        <div class="tabs">{tab_headers}</div>
        <div class="tab-content">{tab_content}</div>

        <div class="methodology">
            <h2>🔬 Methodology</h2>
            <p><strong>Sample Size:</strong> {methodology_sample_size:,} measurements across 3 datasets, 3 budgets, 3 seeds</p>
            <p><strong>Baseline:</strong> selector:random_within_type (explicit placebo)</p>
            <p><strong>Metric:</strong> Score advantage = mean_datasets[adapter_score - placebo_score]</p>
        </div>
    </div>

    <script>
        function showTab(event, tabName) {{
            document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.getElementById(tabName).classList.add('active');
            event.currentTarget.classList.add('active');
        }}
    </script>
</body>
</html>'''

def _generate_champions_section(leaderboards: Dict[str, List[Dict[str, Any]]]) -> str:
    """Generate champions summary section"""
    html = '<div class="summary-cards"><div class="summary-card"><h3>🏆 Champions by Budget</h3>'
    
    for budget_str, leaderboard in leaderboards.items():
        if leaderboard:
            champ = leaderboard[0]
            html += f'<p><strong>{budget_str}:</strong> {champ["adapter"]} ({champ["score"]:.3f}, +{champ["advantage"]:.3f})</p>'
    
    html += '</div></div>'
    return html

def _generate_metrics_grid(adv_map: Dict[str, Any]) -> str:
    """Generate metrics summary grid"""
    return ""  # Simplified for integration

def _generate_performance_insights() -> str:
    """Generate performance insights section"""
    return ""  # Simplified for integration

def _generate_tab_headers() -> str:
    """Generate tab headers"""
    headers = ""
    for i, budget_str in enumerate(["8%", "15%", "30%"]):
        active = "active" if i == 0 else ""
        headers += f'<button class="tab {active}" onclick="showTab(event, \'{budget_str}\')">{budget_str} Budget</button>'
    return headers

def _generate_tab_content(leaderboards: Dict[str, List[Dict[str, Any]]]) -> str:
    """Generate tab content for all budgets"""
    content = ""
    
    for i, (budget_str, leaderboard) in enumerate(leaderboards.items()):
        active = "active" if i == 0 else ""
        content += f'<div class="tab-pane {active}" id="{budget_str}">'
        content += f'<h2>{budget_str} Budget Leaderboard</h2>'
        content += f'<table class="leaderboard-table"><thead><tr><th>Rank</th><th>Adapter</th><th>Family</th><th>Score</th><th>Advantage</th></tr></thead><tbody>'
        
        for rank, item in enumerate(leaderboard[:10], 1):  # Top 10 only
            rank_class = f"rank-{rank}" if rank <= 3 else "rank-default"
            family_class = f"family-{item['family']}"
            
            content += f'''
                <tr>
                    <td><span class="rank-badge {rank_class}">{rank}</span></td>
                    <td>{item['adapter']}</td>
                    <td><span class="family-badge {family_class}">{item['family']}</span></td>
                    <td>{item['score']:.4f}</td>
                    <td>+{item['advantage']:.4f}</td>
                </tr>
            '''
        
        content += '</tbody></table></div>'
    
    return content

# Integration functions for the main pipeline

def add_html_reporting_args(parser):
    """Add HTML reporting arguments to argument parser"""
    parser.add_argument(
        "--emit-html-report",
        action="store_true",
        help="Generate interactive HTML performance report"
    )
    parser.add_argument(
        "--html-report-name",
        default="performance_report.html",
        help="Name for HTML report file (default: performance_report.html)"
    )

def generate_html_if_requested(args, metrics_csv_path: str, advantage_map_path: str, output_dir: str) -> Optional[str]:
    """Generate HTML report if requested via command line args"""
    if not getattr(args, 'emit_html_report', False):
        return None
    
    try:
        html_path = generate_performance_html(
            metrics_csv_path=metrics_csv_path,
            advantage_map_path=advantage_map_path, 
            output_dir=output_dir,
            run_id=getattr(args, 'run_id', None)
        )
        
        logger.info(f"HTML report generated: {html_path}")
        return html_path
        
    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")
        return None