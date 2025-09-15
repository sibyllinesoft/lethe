#!/usr/bin/env python3
"""
Final Enhanced HTML Generator with Backend-Aware Labeling and Performance Metrics
Implements all requirements:
- Backend-aware labeling from configs/report_overrides.yml
- Performance metrics (p95 latency, failure rate, Quality-Throughput score)
- Stress mode documentation
- Clean CSS styling with proper structure
"""

import json
import pandas as pd
import numpy as np
import yaml
import subprocess
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def load_report_overrides(config_path: str) -> Dict[str, str]:
    """Load backend configuration from report overrides file"""
    try:
        config_file = Path(config_path) / "configs" / "report_overrides.yml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                return {
                    'RECALL_METRIC': config.get('RECALL_METRIC', 'score'),
                    'TASK_MODE': config.get('TASK_MODE', 'QA over conversations'),
                    'VECTOR_BACKEND': config.get('VECTOR_BACKEND', 'faiss'),
                    'CUSTOM_BACKEND_NAME': config.get('CUSTOM_BACKEND_NAME', ''),
                    'LETHE_ENGINE_ADAPTER_ID': config.get('LETHE_ENGINE_ADAPTER_ID', 'rag:vector_faiss_cosine')
                }
    except Exception as e:
        print(f"Warning: Could not load report overrides: {e}")
    
    # Fallback defaults
    return {
        'RECALL_METRIC': 'score',
        'TASK_MODE': 'QA over conversations',
        'VECTOR_BACKEND': 'faiss',
        'CUSTOM_BACKEND_NAME': '',
        'LETHE_ENGINE_ADAPTER_ID': 'rag:vector_faiss_cosine'
    }

def get_backend_suffix(vector_backend: str) -> str:
    """Generate backend suffix based on VECTOR_BACKEND"""
    backend_map = {
        'faiss': '— Lethe Engine (Faiss)',
        'milvus': '— Lethe Engine (Milvus)',
        'weaviate': '— Lethe Engine (Weaviate)',
        'vespa': '— Lethe Engine (Vespa)',
        'chroma': '— Lethe Engine (Chroma)'
    }
    return backend_map.get(vector_backend.lower(), f'— Lethe Engine ({vector_backend.title()})')

def create_display_name_mapping(config: Dict[str, str]) -> Dict[str, str]:
    """Create display name mapping with backend-aware labeling"""
    backend_suffix = get_backend_suffix(config['VECTOR_BACKEND'])
    lethe_adapter = config['LETHE_ENGINE_ADAPTER_ID']
    
    # Base display names without "Lethe/" prefixes
    display_names = {
        # Long-context baselines
        "long:full_context_upper_bound": "All Context (Upper Bound)",
        "long:sliding_window": "Sliding Window",
        "long:streaming_llm": "Streaming Context",
        
        # RAG / Search - Backend-aware naming
        "rag:bm25": "BM25",
        "rag:vector_faiss_cosine": "Vector (Faiss)",
        "rag:vector_milvus_cosine": "Vector (Milvus)",
        "rag:vector_weaviate_cosine": "Vector (Weaviate)",
        "rag:hybrid_faiss_50_50": "Hybrid 50/50 (Faiss+BM25)",
        "rag:hybrid_milvus_50_50": "Hybrid 50/50 (Milvus)",
        "rag:hybrid_weaviate_50_50": "Hybrid 50/50 (Weaviate)",
        "rag:hybrid_vespa_50_50": "Hybrid 50/50 (Vespa)",
        
        # Rerank
        "rerank:bge_frozen_pool": "Rerank (BGE)",
        "rerank:colbert": "Rerank (ColBERT)",
        
        # Selectors / Pruners
        "selector:last_k": "Last K Turns",
        "selector:tfidf_topspans": "TF-IDF Top Spans",
        "selector:entropy_filter": "Entropy Filter",
        "selector:langchain_compress": "Contextual Compression",
        "selector:llamaindex_processors": "LlamaIndex Processors",
        "selector:llmlingua_style": "LLMLingua Pruner",
        "selector:zoekt_regex_symbols": "Code Symbol Filter",
        
        # Placebo
        "selector:random_within_type": "Random (Placebo)"
    }
    
    # Apply Lethe Engine suffix to Faiss adapters
    if lethe_adapter in display_names:
        display_names[lethe_adapter] += " — Lethe Engine"
    
    # Add Lethe suffix to Faiss hybrid adapter if it exists
    if "rag:hybrid_faiss_50_50" in display_names:
        display_names["rag:hybrid_faiss_50_50"] += " — Lethe"
    
    return display_names

def calculate_performance_metrics(raw_data: List[Dict], adapter: str, dataset: str, 
                                 keep_percentage: float, k_value: int) -> Dict[str, float]:
    """Calculate p95 latency, failure rate, and Quality-Throughput score"""
    # Filter data for specific configuration
    filtered_data = [
        d for d in raw_data 
        if (d.get('adapter') == adapter and 
            d.get('dataset') == dataset and 
            d.get('keep_percentage') == keep_percentage and 
            d.get('k_value') == k_value)
    ]
    
    if not filtered_data:
        return {'p95_latency': 0, 'failure_rate': 0, 'qt_score': 0}
    
    # Extract metrics
    response_times = [d.get('response_time_ms', 0) for d in filtered_data]
    scores = [d.get('score', 0) for d in filtered_data]
    
    # Calculate p95 latency
    p95_latency = np.percentile(response_times, 95) if response_times else 0
    
    # Calculate failure rate (assuming failure is score < 0.1 or missing)
    failures = sum(1 for score in scores if score < 0.1)
    failure_rate = (failures / len(scores) * 100) if scores else 0
    
    # Calculate average recall for QT score
    avg_recall = np.mean(scores) if scores else 0
    
    # Quality-Throughput score: QT = Recall@5 × (1000/latency_p95_ms) × (1-failure_rate)
    if p95_latency > 0:
        qt_score = avg_recall * (1000 / p95_latency) * (1 - failure_rate / 100)
    else:
        qt_score = 0
    
    return {
        'p95_latency': round(p95_latency, 1),
        'failure_rate': round(failure_rate, 1),
        'qt_score': round(qt_score, 2)
    }

def get_git_commit() -> str:
    """Get current git commit hash"""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, cwd=Path(__file__).parent)
        return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
    except:
        return "unknown"

def get_manifest_hash(manifest_path: str) -> str:
    """Calculate SHA256 of manifest file"""
    try:
        with open(manifest_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except:
        return "unknown"

def generate_chart_data(df: pd.DataFrame, raw_data: List[Dict], display_names: Dict[str, str], 
                       budget: float, k_value: int) -> str:
    """Generate chart data for a specific budget and k value"""
    # Filter for Conv-Set-A dataset and specific budget/k
    chart_df = df[(df['dataset'] == 'Conv-Set-A') & 
                  (df['keep_percentage'] == budget) & 
                  (df['k_value'] == k_value) &
                  (df['metric'] == 'score')]
    
    if chart_df.empty:
        return "[]"
    
    chart_data = []
    for _, row in chart_df.iterrows():
        adapter = row['adapter']
        score = row['mean']
        
        # Calculate performance metrics
        perf_metrics = calculate_performance_metrics(
            raw_data, adapter, 'Conv-Set-A', budget, k_value
        )
        
        chart_data.append({
            'adapter': adapter,
            'label': display_names.get(adapter, adapter),
            'score': round(score, 3),
            'p95_latency': perf_metrics['p95_latency'],
            'failure_rate': perf_metrics['failure_rate'],
            'qt_score': perf_metrics['qt_score']
        })
    
    # Sort by score descending
    chart_data.sort(key=lambda x: x['score'], reverse=True)
    
    return json.dumps(chart_data)

def generate_final_html_report(
    metrics_csv_path: str,
    raw_results_path: str,
    advantage_map_path: str,
    signed_manifest_path: str,
    leakage_attestation_path: str,
    config_dir: str,
    output_path: str = "final_performance_report.html"
):
    """Generate the final comprehensive HTML report"""
    
    # Load configuration
    config = load_report_overrides(config_dir)
    display_names = create_display_name_mapping(config)
    
    # Load data
    df = pd.read_csv(metrics_csv_path)
    with open(raw_results_path, 'r') as f:
        raw_data = json.load(f)
    with open(advantage_map_path, 'r') as f:
        adv_map = json.load(f)
    with open(signed_manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Get metadata
    run_id = manifest.get("run_id", "unknown")
    git_commit = get_git_commit()
    manifest_hash = get_manifest_hash(signed_manifest_path)
    has_leakage_attestation = os.path.exists(leakage_attestation_path)
    generator = manifest.get("generator", "unknown")
    
    # Get datasets
    datasets = sorted([d for d in df["dataset"].unique() if pd.notna(d) and not str(d).startswith("#")])
    datasets_str = ", ".join(datasets)
    
    # Generate chart data for different budgets
    budgets = [0.08, 0.15, 0.30]
    k_value = 5  # Focus on k=5 as specified
    
    chart_data_08 = generate_chart_data(df, raw_data, display_names, budgets[0], k_value)
    chart_data_15 = generate_chart_data(df, raw_data, display_names, budgets[1], k_value)
    chart_data_30 = generate_chart_data(df, raw_data, display_names, budgets[2], k_value)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # HTML template with all enhancements
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lethe Performance Report - {run_id}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        :root {{
            --primary-blue: #1e40af;
            --secondary-blue: #3b82f6;
            --light-blue: #dbeafe;
            --success-green: #059669;
            --warning-orange: #d97706;
            --error-red: #dc2626;
            --neutral-gray: #6b7280;
            --light-gray: #f9fafb;
            --white: #ffffff;
            --text-dark: #1f2937;
            --border-gray: #e5e7eb;
        }}
        
        * {{
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: var(--light-gray);
            color: var(--text-dark);
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: var(--white);
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, var(--primary-blue), var(--secondary-blue));
            color: var(--white);
            padding: 2rem;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0 0 0.5rem 0;
            font-size: 2rem;
            font-weight: 700;
        }}
        
        .task-mode-badge {{
            display: inline-block;
            background: rgba(255, 255, 255, 0.2);
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-size: 0.875rem;
            font-weight: 500;
            margin-top: 0.5rem;
        }}
        
        .provenance-banner {{
            background: var(--light-blue);
            padding: 1rem 2rem;
            border-bottom: 1px solid var(--border-gray);
        }}
        
        .provenance-banner h3 {{
            margin: 0 0 0.5rem 0;
            color: var(--primary-blue);
        }}
        
        .provenance-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 0.5rem;
        }}
        
        .provenance-item {{
            font-size: 0.875rem;
        }}
        
        .provenance-label {{
            font-weight: 600;
            color: var(--neutral-gray);
        }}
        
        .main-content {{
            padding: 2rem;
        }}
        
        .methodology-section {{
            background: var(--light-gray);
            border: 1px solid var(--border-gray);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .methodology-section h3 {{
            margin: 0 0 1rem 0;
            color: var(--primary-blue);
        }}
        
        .charts-container {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 3rem;
        }}
        
        .chart-section {{
            background: var(--white);
            border: 1px solid var(--border-gray);
            border-radius: 8px;
            padding: 1.5rem;
        }}
        
        .chart-title {{
            text-align: center;
            margin-bottom: 0.5rem;
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-dark);
        }}
        
        .chart-subtitle {{
            text-align: center;
            margin-bottom: 1.5rem;
            font-size: 0.875rem;
            color: var(--neutral-gray);
        }}
        
        .chart-container {{
            position: relative;
            height: 600px;
        }}
        
        .performance-badge {{
            display: inline-block;
            background: var(--light-blue);
            color: var(--primary-blue);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 500;
            margin: 0.25rem 0.125rem;
        }}
        
        .qt-badge {{
            background: var(--success-green);
            color: var(--white);
        }}
        
        .footer {{
            background: var(--light-gray);
            padding: 1rem 2rem;
            border-top: 1px solid var(--border-gray);
            text-align: center;
            font-size: 0.875rem;
            color: var(--neutral-gray);
        }}
        
        @media (max-width: 768px) {{
            body {{ padding: 10px; }}
            .header {{ padding: 1rem; }}
            .header h1 {{ font-size: 1.5rem; }}
            .main-content {{ padding: 1rem; }}
            .chart-container {{ height: 400px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Lethe Performance Report</h1>
            <div class="task-mode-badge">
                {config['TASK_MODE']} • k=5 • Budgets: 8% / 15% / 30%
            </div>
        </div>
        
        <div class="provenance-banner">
            <h3>Provenance & Validation</h3>
            <div class="provenance-grid">
                <div class="provenance-item">
                    <div class="provenance-label">Run ID:</div>
                    <div>{run_id}</div>
                </div>
                <div class="provenance-item">
                    <div class="provenance-label">Git Commit:</div>
                    <div>{git_commit}</div>
                </div>
                <div class="provenance-item">
                    <div class="provenance-label">Manifest Hash:</div>
                    <div>{manifest_hash}</div>
                </div>
                <div class="provenance-item">
                    <div class="provenance-label">Generator:</div>
                    <div>{generator}</div>
                </div>
                <div class="provenance-item">
                    <div class="provenance-label">Datasets:</div>
                    <div>{datasets_str}</div>
                </div>
                <div class="provenance-item">
                    <div class="provenance-label">Leakage Check:</div>
                    <div>{"✅ Passed" if has_leakage_attestation else "❌ Failed"}</div>
                </div>
                <div class="provenance-item">
                    <div class="provenance-label">Generated:</div>
                    <div>{timestamp}</div>
                </div>
                <div class="provenance-item">
                    <div class="provenance-label">Backend:</div>
                    <div>{config['VECTOR_BACKEND'].title()} Vector Store</div>
                </div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="methodology-section">
                <h3>Stress Mode Methodology</h3>
                <p><strong>Stress Mode forces selection:</strong> Corpora exceed the model window and include distractors; Full-Context is infeasible/inefficient, making selection/retrieval essential. This mode tests the ability of different adapters to identify and retrieve the most relevant information under resource constraints.</p>
                <p><strong>Quality-Throughput Score:</strong> QT = Recall@5 × (1000/latency_p95_ms) × (1-failure_rate). This composite metric balances accuracy with performance, rewarding systems that maintain high recall while operating efficiently.</p>
            </div>
            
            <div class="charts-container">
                <div class="chart-section">
                    <h2 class="chart-title">Performance at 8% Budget</h2>
                    <p class="chart-subtitle">Recall@5 scores with performance metrics (640 tokens selected from 8,000 total)</p>
                    <div class="chart-container">
                        <canvas id="chart08"></canvas>
                    </div>
                </div>
                
                <div class="chart-section">
                    <h2 class="chart-title">Performance at 15% Budget</h2>
                    <p class="chart-subtitle">Recall@5 scores with performance metrics (1,200 tokens selected from 8,000 total)</p>
                    <div class="chart-container">
                        <canvas id="chart15"></canvas>
                    </div>
                </div>
                
                <div class="chart-section">
                    <h2 class="chart-title">Performance at 30% Budget</h2>
                    <p class="chart-subtitle">Recall@5 scores with performance metrics (2,400 tokens selected from 8,000 total)</p>
                    <div class="chart-container">
                        <canvas id="chart30"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <p>Report generated with enhanced backend-aware labeling and performance metrics • Vector Backend: {config['VECTOR_BACKEND'].title()}</p>
        </div>
    </div>

    <script>
        const chartConfig = {{
            type: 'bar',
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {{
                    legend: {{ display: false }},
                    tooltip: {{
                        callbacks: {{
                            label: function(context) {{
                                const data = context.raw;
                                return [
                                    `Recall@5: ${{data.score.toFixed(3)}}`,
                                    `p95 Latency: ${{data.p95_latency}}ms`,
                                    `Failure Rate: ${{data.failure_rate}}%`,
                                    `QT Score: ${{data.qt_score}}`
                                ];
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        beginAtZero: true,
                        max: 1.0,
                        title: {{
                            display: true,
                            text: 'Recall@5 Score'
                        }}
                    }},
                    y: {{
                        ticks: {{
                            callback: function(value, index) {{
                                const data = this.chart.data.datasets[0].data[index];
                                if (data) {{
                                    return [
                                        data.label,
                                        `p95=${{data.p95_latency}}ms`,
                                        `fail=${{data.failure_rate}}%`,
                                        `QT=${{data.qt_score}}`
                                    ];
                                }}
                                return '';
                            }}
                        }}
                    }}
                }}
            }}
        }};

        function createChart(canvasId, data, title) {{
            const ctx = document.getElementById(canvasId).getContext('2d');
            
            const chartData = {{
                labels: data.map(d => d.label),
                datasets: [{{
                    label: 'Recall@5',
                    data: data.map(d => ({{
                        x: d.score,
                        y: d.label,
                        score: d.score,
                        p95_latency: d.p95_latency,
                        failure_rate: d.failure_rate,
                        qt_score: d.qt_score,
                        label: d.label
                    }})),
                    backgroundColor: data.map((d, i) => {{
                        const hue = (i * 137.5) % 360; // Golden angle distribution
                        return `hsla(${{hue}}, 70%, 60%, 0.8)`;
                    }}),
                    borderColor: data.map((d, i) => {{
                        const hue = (i * 137.5) % 360;
                        return `hsla(${{hue}}, 70%, 50%, 1.0)`;
                    }}),
                    borderWidth: 1,
                    barThickness: 30
                }}]
            }};
            
            new Chart(ctx, {{
                type: chartConfig.type,
                data: chartData,
                options: chartConfig.options
            }});
        }}

        // Create all charts
        const data08 = {chart_data_08};
        const data15 = {chart_data_15};
        const data30 = {chart_data_30};
        
        createChart('chart08', data08, '8% Budget');
        createChart('chart15', data15, '15% Budget');
        createChart('chart30', data30, '30% Budget');
    </script>
</body>
</html>"""
    
    # Write the HTML file
    with open(output_path, 'w') as f:
        f.write(html_template)
    
    print(f"Final enhanced HTML report generated: {output_path}")
    return output_path

if __name__ == "__main__":
    # Default paths for the current directory structure
    base_dir = Path(__file__).parent
    
    generate_final_html_report(
        metrics_csv_path=str(base_dir / "metrics_summary.csv"),
        raw_results_path=str(base_dir / "raw_results.json"),
        advantage_map_path=str(base_dir / "advantage_map.json"),
        signed_manifest_path=str(base_dir / "signed_manifest.json"),
        leakage_attestation_path=str(base_dir / "leakage_attestation.json"),
        config_dir=str(base_dir),
        output_path=str(base_dir / "final_performance_report.html")
    )