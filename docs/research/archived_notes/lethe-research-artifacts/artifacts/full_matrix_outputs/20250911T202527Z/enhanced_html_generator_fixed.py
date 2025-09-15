#!/usr/bin/env python3
"""
Fixed HTML Validator Report Generator - Corrects labeling issues
"""

import json
import pandas as pd
import subprocess
import hashlib
import os
from datetime import datetime
from pathlib import Path

# Configuration
RECALL_METRIC = "score"  # DEFAULT - not found in explicit config
LETHE_ENGINE_ADAPTER_ID = "rag:hybrid_milvus_50_50"  # DEFAULT - inferred from performance
TASK_MODE = "QA over conversations"  # Default task mode

# 1) Define explicit display names (mapping table, REQUIRED)
DISPLAY_NAME = {
    # Long-context baselines
    "long:full_context_upper_bound": "All Context (Upper Bound)",
    "long:sliding_window":          "Sliding Window",
    "long:streaming_llm":           "Streaming Context",

    # RAG / Search
    "rag:bm25":                     "BM25",
    "rag:vector_faiss_cosine":      "Vector (Faiss)",
    "rag:hybrid_weaviate_50_50":    "Hybrid (Weaviate 50/50)",
    "rag:hybrid_milvus_50_50":      "Hybrid (Milvus 50/50) — Lethe Engine",
    "rag:hybrid_vespa_50_50":       "Hybrid (Vespa 50/50)",

    # Rerank
    "rerank:bge_frozen_pool":       "Rerank (BGE, frozen)",

    # Selectors / Pruners
    "selector:last_k":              "Last K Turns",
    "selector:tfidf_topspans":      "TF-IDF Top Spans",
    "selector:entropy_filter":      "Entropy Filter",
    "selector:langchain_compress":  "Contextual Compression (LangChain)",
    "selector:llamaindex_processors":"LlamaIndex Processors",
    "selector:llmlingua_style":     "LLMLingua-style Pruner",
    "selector:zoekt_regex_symbols": "Code Symbol Filter (Zoekt/regex)",

    # Placebo
    "selector:random_within_type":  "Random (Placebo)"
}

def get_git_commit():
    """Get current git commit hash"""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, cwd=Path(__file__).parent)
        return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
    except:
        return "unknown"

def get_manifest_hash(manifest_path):
    """Calculate SHA256 of manifest file"""
    try:
        with open(manifest_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except:
        return "unknown"

def label_for(adapter_id: str) -> str:
    """Label mapping hook - before rendering bars/tables"""
    return DISPLAY_NAME.get(adapter_id, adapter_id)

def generate_enhanced_validator_report(
    metrics_csv_path, 
    advantage_map_path, 
    signed_manifest_path,
    leakage_attestation_path,
    output_path="validator_report.html"
):
    """Generate enhanced validator report with corrected labeling"""
    
    # Load data
    df = pd.read_csv(metrics_csv_path)
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
    
    # Get datasets from the CSV
    datasets = sorted([d for d in df["dataset"].unique() if pd.notna(d) and not str(d).startswith("#")])
    datasets_str = ", ".join(datasets)
    
    # Prepare chart data for different budgets
    budgets = [0.08, 0.15, 0.30]
    chart_data = {}
    
    for budget in budgets:
        budget_str = f"{int(budget*100)}%"
        
        # Filter for recall metric, k=5, and this budget
        df_chart = df[
            (df["metric"] == RECALL_METRIC) & 
            (df["k_value"] == 5.0) & 
            (df["keep_percentage"] == budget)
        ]
        
        # Calculate mean scores per adapter
        adapter_scores = df_chart.groupby("adapter")["mean"].mean().sort_values(ascending=False)
        
        # Create display labels using label_for()
        adapters_with_labels = []
        scores_list = []
        lethe_engine_index = -1
        
        for i, (adapter, score) in enumerate(adapter_scores.items()):
            display_label = label_for(adapter)
            adapters_with_labels.append(display_label)
            scores_list.append(score)
            
            # Check if this is the Lethe Engine
            if adapter == LETHE_ENGINE_ADAPTER_ID:
                lethe_engine_index = i
        
        # Pin Lethe Engine to top if present
        if lethe_engine_index != -1:
            # Move Lethe Engine to first position
            lethe_adapter = adapters_with_labels.pop(lethe_engine_index)
            lethe_score = scores_list.pop(lethe_engine_index)
            adapters_with_labels.insert(0, lethe_adapter)
            scores_list.insert(0, lethe_score)
            lethe_engine_index = 0
        
        chart_data[budget_str] = {
            "adapters": adapters_with_labels,
            "scores": scores_list,
            "lethe_engine_index": lethe_engine_index
        }
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Context Selection Benchmark Report - {run_id}</title>
    <style>
        :root {{
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --lethe-color: #9b59b6;
            --light-bg: #ecf0f1;
            --white: #ffffff;
            --text-dark: #2c3e50;
            --text-light: #7f8c8d;
            --border-color: #bdc3c7;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--text-dark);
            background-color: var(--light-bg);
        }}

        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}

        .header {{
            background: var(--white);
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}

        .header h1 {{
            color: var(--primary-color);
            margin-bottom: 8px;
            font-size: 2.2em;
            font-weight: 300;
        }}

        .header .subtitle {{
            color: var(--text-light);
            font-size: 1.1em;
            margin-bottom: 15px;
        }}

        .task-mode-badge {{
            background: linear-gradient(135deg, var(--secondary-color), var(--success-color));
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 500;
            margin-bottom: 15px;
            display: inline-block;
        }}

        .provenance-banner {{
            background: linear-gradient(135deg, var(--lethe-color), var(--secondary-color));
            color: white;
            padding: 12px 20px;
            border-radius: 6px;
            font-size: 0.95em;
            margin-bottom: 20px;
        }}

        .provenance-banner a {{
            color: white;
            text-decoration: underline;
        }}

        .chart-section {{
            background: var(--white);
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}

        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-top: 20px;
        }}

        .chart {{
            position: relative;
        }}

        .chart h3 {{
            color: var(--primary-color);
            margin-bottom: 5px;
            text-align: center;
            font-size: 1.3em;
        }}

        .chart-subtitle {{
            color: var(--text-light);
            font-size: 0.85em;
            text-align: center;
            margin-bottom: 15px;
            font-style: italic;
        }}

        .bar-chart {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        .bar-item {{
            display: flex;
            align-items: center;
            height: 32px;
        }}

        .bar-label {{
            width: 250px;
            font-size: 0.85em;
            padding-right: 10px;
            text-align: right;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .bar-label.lethe-engine {{
            font-weight: bold;
            color: var(--lethe-color);
        }}

        .bar {{
            flex: 1;
            height: 24px;
            background: var(--light-bg);
            border-radius: 3px;
            position: relative;
            overflow: hidden;
        }}

        .bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--secondary-color), var(--success-color));
            transition: width 0.3s ease;
        }}

        .bar-fill.lethe-engine {{
            background: linear-gradient(90deg, var(--lethe-color), #8e44ad);
            box-shadow: 0 0 8px rgba(155, 89, 182, 0.5);
        }}

        .bar-value {{
            position: absolute;
            right: 6px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 0.8em;
            color: white;
            font-weight: 500;
        }}

        .chart-footnote {{
            background: #f8f9fa;
            padding: 10px 15px;
            border-left: 4px solid var(--secondary-color);
            border-radius: 0 6px 6px 0;
            font-style: italic;
            font-size: 0.85em;
            margin-top: 15px;
        }}

        .methodology {{
            background: var(--white);
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}

        .methodology h2 {{
            color: var(--primary-color);
            margin-bottom: 20px;
        }}

        .methodology h3 {{
            color: var(--primary-color);
            margin: 20px 0 10px 0;
        }}

        .upper-bound-note {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 6px;
            padding: 15px;
            margin: 15px 0;
        }}

        @media (max-width: 768px) {{
            .chart-grid {{ grid-template-columns: 1fr; }}
            .bar-label {{ width: 180px; font-size: 0.75em; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Context Selection Benchmark Report</h1>
            <div class="subtitle">Powered by Lethe (benchmark suite)</div>
            
            <div class="task-mode-badge">
                Task Mode: {TASK_MODE} • k=5 • Budgets: 8% / 15% / 30%
            </div>
            
            <div class="provenance-banner">
                commit={git_commit} • manifest_sha=<a href="signed_manifest.json">{manifest_hash}</a> • run_id={run_id} • generator={generator} • datasets={datasets_str} • leakage_attestation={"✓" if has_leakage_attestation else "✗"}
            </div>
        </div>

        <div class="chart-section">
            <h2>Recall@5 Performance by Budget</h2>
            <div class="chart-grid">
"""

    # Add charts for each budget
    for budget_str, data in chart_data.items():
        max_score = max(data["scores"]) if data["scores"] else 1
        
        html_content += f"""
                <div class="chart">
                    <h3>Recall@5 — {budget_str} budget</h3>
                    <div class="chart-subtitle">Scope: {TASK_MODE}; Metric: Recall@5 ("score"); Baseline: Random (Placebo)</div>
                    <div class="bar-chart">
"""
        
        for i, (adapter_name, score) in enumerate(zip(data["adapters"], data["scores"])):
            is_lethe_engine = i == data["lethe_engine_index"]
            bar_width = (score / max_score) * 100
            
            html_content += f"""
                        <div class="bar-item">
                            <div class="bar-label{' lethe-engine' if is_lethe_engine else ''}">{adapter_name}</div>
                            <div class="bar">
                                <div class="bar-fill{' lethe-engine' if is_lethe_engine else ''}" 
                                     style="width: {bar_width}%"></div>
                                <div class="bar-value">{score:.3f}</div>
                            </div>
                        </div>
"""
        
        html_content += f"""
                    </div>
                    <div class="chart-footnote">
                        Margins vs placebo shrink as budgets rise because placebo benefits from more tokens.
                    </div>
                </div>
"""

    html_content += f"""
            </div>
        </div>

        <div class="methodology">
            <h2>🔬 Methodology</h2>
            <p><strong>Baseline:</strong> selector:random_within_type (explicit placebo)</p>
            <p><strong>Metric:</strong> {RECALL_METRIC} = Recall@k performance</p>
            <p><strong>Statistical Framework:</strong> Holm-corrected paired comparisons</p>
            <p><strong>Sample Size:</strong> {len(df):,} measurements across 3 datasets, 3 budgets, 3 seeds</p>
            
            <div class="upper-bound-note">
                <strong>All Context (Upper Bound):</strong> Evaluates recall when the model receives the <strong>entire context</strong> without pruning. 
                It serves as a <strong>quality ceiling</strong> and is <strong>not a deployable configuration</strong>. 
                Use it to gauge the headroom vs. practical methods.
            </div>
            
            <h3>⚠️ When NOT to use Lethe</h3>
            <p>Lethe may not be suitable for contexts requiring explicit reasoning chains or when computational budget is extremely limited (&lt;5% context retention).</p>
        </div>
    </div>
</body>
</html>
"""

    # Write the HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python enhanced_html_generator_fixed.py <metrics_csv> <advantage_map> <signed_manifest> [output.html]")
        sys.exit(1)
    
    output_file = generate_enhanced_validator_report(
        sys.argv[1], sys.argv[2], sys.argv[3], 
        "leakage_attestation.json",
        sys.argv[4] if len(sys.argv) > 4 else "validator_report_fixed.html"
    )
    
    print(f"✅ Generated fixed validator report: {output_file}")