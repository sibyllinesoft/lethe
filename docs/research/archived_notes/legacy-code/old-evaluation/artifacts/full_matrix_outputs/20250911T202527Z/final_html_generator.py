#!/usr/bin/env python3
"""
HTML Report Generator with Improved Label Clarity
Implements the regression debug protocol label requirements for enhanced readability.
"""

import json
import html
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

class HTMLReportGenerator:
    """Generates HTML reports with improved label clarity and tooltips."""
    
    # Primary labels for main display (clean, user-friendly)
    PRIMARY_LABELS = {
        "hybrid_50_50_milvus": "Hybrid 50/50",
        "hybrid_milvus": "Hybrid 50/50", 
        "vector_faiss_lethe": "Vector",
        "vector_faiss": "Vector",
        "bm25": "BM25",
        "all_context_upper_bound": "All Context (Upper Bound)",
        "streaming_context": "Streaming Context", 
        "sliding_window": "Sliding Window",
        "rerank_bge": "Rerank (BGE)",
        "contextual_compression": "Contextual Compression"
    }
    
    # Engine/vendor information for tooltips and badges
    ENGINE_INFO = {
        "hybrid_50_50_milvus": {"engine": "Milvus", "type": "Vector Database"},
        "hybrid_milvus": {"engine": "Milvus", "type": "Vector Database"},
        "vector_faiss_lethe": {"engine": "Faiss", "type": "Vector Database", "badge": "Lethe Engine"},
        "vector_faiss": {"engine": "Faiss", "type": "Vector Database", "badge": "Lethe Engine"},
        "bm25": {"engine": "Elasticsearch", "type": "Text Search"},
        "all_context_upper_bound": {"engine": "Direct", "type": "Baseline"},
        "streaming_context": {"engine": "Custom", "type": "Context Management"},
        "sliding_window": {"engine": "Custom", "type": "Context Management"},
        "rerank_bge": {"engine": "BGE", "type": "Reranking Model"},
        "contextual_compression": {"engine": "Custom", "type": "Compression"}
    }
    
    # Forbidden old label patterns (for validation)
    FORBIDDEN_PATTERNS = [">rag:", ">selector:", ">long:", ">rerank:"]
    
    def __init__(self):
        """Initialize the HTML report generator."""
        self.report_timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")
        
    def normalize_method_key(self, method_name: str) -> str:
        """Normalize method name to consistent key format."""
        # Convert various naming patterns to standardized keys
        normalized = method_name.lower().replace(" ", "_").replace("-", "_")
        
        # Handle specific mappings
        mappings = {
            "hybrid_50/50_(milvus)": "hybrid_50_50_milvus",
            "hybrid_50/50": "hybrid_50_50_milvus", 
            "vector_(faiss)_—_lethe_engine": "vector_faiss_lethe",
            "vector_faiss": "vector_faiss_lethe",
            "vector_(faiss)": "vector_faiss_lethe",
            "all_context_(upper_bound)": "all_context_upper_bound",
            "rerank_(bge)": "rerank_bge"
        }
        
        return mappings.get(normalized, normalized)
        
    def get_display_label(self, method_name: str) -> str:
        """Get clean display label for a method."""
        key = self.normalize_method_key(method_name)
        return self.PRIMARY_LABELS.get(key, method_name)
        
    def get_engine_info(self, method_name: str) -> Dict[str, str]:
        """Get engine information for tooltips and badges."""
        key = self.normalize_method_key(method_name) 
        return self.ENGINE_INFO.get(key, {"engine": "Unknown", "type": "Unknown"})
        
    def create_method_label_html(self, method_name: str, include_tooltip: bool = True) -> str:
        """Create HTML for method label with tooltip and badge."""
        display_label = self.get_display_label(method_name)
        engine_info = self.get_engine_info(method_name)
        
        # Create tooltip content
        tooltip_content = f"Engine: {engine_info['engine']} | Type: {engine_info['type']}"
        
        # Create badge if specified
        badge_html = ""
        if "badge" in engine_info:
            badge_html = f' <span class="engine-badge">— {engine_info["badge"]}</span>'
            
        # Build complete HTML
        if include_tooltip:
            return f'''<span class="method-label" title="{tooltip_content}">{html.escape(display_label)}{badge_html}</span>'''
        else:
            return f'''<span class="method-label">{html.escape(display_label)}{badge_html}</span>'''
            
    def generate_css_styles(self) -> str:
        """Generate CSS styles for the report."""
        return """
        <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background-color: #f8f9fa; 
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background-color: white; 
            padding: 30px; 
            border-radius: 8px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
        }
        h1, h2, h3 { 
            color: #2c3e50; 
            margin-top: 0; 
        }
        h1 { 
            border-bottom: 3px solid #3498db; 
            padding-bottom: 10px; 
        }
        .method-label {
            font-weight: 600;
            color: #2c3e50;
            cursor: help;
            position: relative;
        }
        .method-label:hover {
            color: #3498db;
        }
        .engine-badge {
            font-weight: 400;
            color: #7f8c8d;
            font-size: 0.9em;
        }
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 1px 5px rgba(0,0,0,0.1);
        }
        .metrics-table th, .metrics-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .metrics-table th {
            background-color: #34495e;
            color: white;
            font-weight: 600;
        }
        .metrics-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .metrics-table tr:hover {
            background-color: #e8f4fd;
        }
        .score-high { color: #27ae60; font-weight: 600; }
        .score-medium { color: #f39c12; font-weight: 600; }
        .score-low { color: #e74c3c; font-weight: 600; }
        .timestamp {
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 20px;
        }
        .summary-box {
            background-color: #ecf0f1;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
        }
        .warning-box {
            background-color: #fdf2e9;
            border-left: 4px solid #e67e22;
            padding: 15px;
            margin: 20px 0;
        }
        .validation-passed {
            background-color: #d5f4e6;
            border-left: 4px solid #27ae60;
            padding: 15px;
            margin: 20px 0;
        }
        .validation-failed {
            background-color: #fadbd8;
            border-left: 4px solid #e74c3c;
            padding: 15px;
            margin: 20px 0;
        }
        </style>
        """
        
    def generate_validation_report_html(self, results_data: Dict[str, Any], 
                                      output_path: str = None) -> str:
        """Generate the main validator report HTML."""
        
        if output_path is None:
            output_path = f"validator_report_{self.report_timestamp}.html"
            
        # Check for forbidden patterns in input data
        validation_status = self._validate_label_compliance(results_data)
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Lethe Evaluation - Validator Report</title>
            {self.generate_css_styles()}
        </head>
        <body>
            <div class="container">
                <h1>Lethe Evaluation - Validator Report</h1>
                
                {self._generate_validation_status_section(validation_status)}
                
                <h2>Method Performance Comparison</h2>
                {self._generate_methods_comparison_table(results_data)}
                
                <h2>Label Mapping Verification</h2>
                {self._generate_label_mapping_table()}
                
                <h2>Engine Information</h2>
                {self._generate_engine_info_table()}
                
                <div class="timestamp">
                    Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write to file
        Path(output_path).write_text(html_content, encoding='utf-8')
        
        return html_content
        
    def _validate_label_compliance(self, results_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that no forbidden label patterns exist."""
        validation_result = {
            "passed": True,
            "forbidden_found": [],
            "methods_processed": 0,
            "primary_labels_found": []
        }
        
        # Convert data to string to search for forbidden patterns
        data_str = json.dumps(results_data)
        
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern in data_str:
                validation_result["passed"] = False
                validation_result["forbidden_found"].append(pattern)
                
        # Check for expected primary labels
        for method_key in results_data.keys():
            if isinstance(method_key, str):
                display_label = self.get_display_label(method_key)
                validation_result["primary_labels_found"].append(display_label)
                validation_result["methods_processed"] += 1
                
        return validation_result
        
    def _generate_validation_status_section(self, validation_status: Dict[str, Any]) -> str:
        """Generate HTML section for validation status."""
        if validation_status["passed"]:
            return f"""
            <div class="validation-passed">
                <h3>✅ Label Validation: PASSED</h3>
                <p>All forbidden label patterns have been successfully removed.</p>
                <p>Methods processed: {validation_status["methods_processed"]}</p>
                <p>Primary labels found: {', '.join(validation_status["primary_labels_found"])}</p>
            </div>
            """
        else:
            return f"""
            <div class="validation-failed">
                <h3>❌ Label Validation: FAILED</h3>
                <p>Found forbidden label patterns: {', '.join(validation_status["forbidden_found"])}</p>
                <p>These patterns must be removed from the data source.</p>
            </div>
            """
            
    def _generate_methods_comparison_table(self, results_data: Dict[str, Any]) -> str:
        """Generate HTML table comparing method performance."""
        table_rows = []
        
        for method_key, method_data in results_data.items():
            if not isinstance(method_data, dict):
                continue
                
            display_label = self.create_method_label_html(method_key)
            engine_info = self.get_engine_info(method_key)
            
            # Extract performance metrics (placeholder - adapt to actual data structure)
            precision = method_data.get('precision', 'N/A')
            recall = method_data.get('recall', 'N/A') 
            f1_score = method_data.get('f1_score', 'N/A')
            efficiency = method_data.get('efficiency', 'N/A')
            
            # Format scores with color coding
            precision_html = self._format_score_html(precision)
            recall_html = self._format_score_html(recall)
            f1_html = self._format_score_html(f1_score)
            efficiency_html = self._format_score_html(efficiency)
            
            row = f"""
            <tr>
                <td>{display_label}</td>
                <td>{engine_info['engine']}</td>
                <td>{engine_info['type']}</td>
                <td>{precision_html}</td>
                <td>{recall_html}</td>
                <td>{f1_html}</td>
                <td>{efficiency_html}</td>
            </tr>
            """
            table_rows.append(row)
            
        return f"""
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Method</th>
                    <th>Engine</th>
                    <th>Type</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>Efficiency</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
        """
        
    def _generate_label_mapping_table(self) -> str:
        """Generate HTML table showing label mappings."""
        table_rows = []
        
        for key, label in self.PRIMARY_LABELS.items():
            engine_info = self.ENGINE_INFO.get(key, {})
            badge = engine_info.get('badge', '')
            badge_display = f" — {badge}" if badge else ""
            
            row = f"""
            <tr>
                <td><code>{key}</code></td>
                <td><strong>{label}</strong>{badge_display}</td>
                <td>{engine_info.get('engine', 'N/A')}</td>
                <td>{engine_info.get('type', 'N/A')}</td>
            </tr>
            """
            table_rows.append(row)
            
        return f"""
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Internal Key</th>
                    <th>Display Label</th>
                    <th>Engine</th>
                    <th>Type</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
        """
        
    def _generate_engine_info_table(self) -> str:
        """Generate HTML table showing engine information."""
        engine_summary = {}
        
        for key, info in self.ENGINE_INFO.items():
            engine = info['engine']
            if engine not in engine_summary:
                engine_summary[engine] = {'methods': [], 'type': info['type']}
            
            display_label = self.PRIMARY_LABELS.get(key, key)
            engine_summary[engine]['methods'].append(display_label)
            
        table_rows = []
        for engine, info in engine_summary.items():
            methods_list = ', '.join(info['methods'])
            row = f"""
            <tr>
                <td><strong>{engine}</strong></td>
                <td>{info['type']}</td>
                <td>{methods_list}</td>
                <td>{len(info['methods'])}</td>
            </tr>
            """
            table_rows.append(row)
            
        return f"""
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Engine</th>
                    <th>Type</th>
                    <th>Methods Using This Engine</th>
                    <th>Count</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
        """
        
    def _format_score_html(self, score) -> str:
        """Format numerical scores with color coding."""
        if score == 'N/A' or score is None:
            return '<span class="score-low">N/A</span>'
            
        try:
            score_val = float(score)
            if score_val >= 0.8:
                css_class = "score-high"
            elif score_val >= 0.6:
                css_class = "score-medium"
            else:
                css_class = "score-low"
                
            return f'<span class="{css_class}">{score_val:.3f}</span>'
        except (ValueError, TypeError):
            return f'<span class="score-low">{html.escape(str(score))}</span>'


def create_sample_report():
    """Create a sample validator report for testing."""
    generator = HTMLReportGenerator()
    
    # Sample data with methods to be relabeled
    sample_data = {
        "hybrid_50_50_milvus": {
            "precision": 0.85,
            "recall": 0.78, 
            "f1_score": 0.81,
            "efficiency": 0.72
        },
        "vector_faiss_lethe": {
            "precision": 0.82,
            "recall": 0.75,
            "f1_score": 0.78,
            "efficiency": 0.69
        },
        "bm25": {
            "precision": 0.71,
            "recall": 0.68,
            "f1_score": 0.69,
            "efficiency": 0.45
        },
        "all_context_upper_bound": {
            "precision": 0.95,
            "recall": 0.92,
            "f1_score": 0.93,
            "efficiency": 0.89
        },
        "streaming_context": {
            "precision": 0.73,
            "recall": 0.71,
            "f1_score": 0.72,
            "efficiency": 0.52
        },
        "sliding_window": {
            "precision": 0.69,
            "recall": 0.65,
            "f1_score": 0.67,
            "efficiency": 0.48
        },
        "rerank_bge": {
            "precision": 0.79,
            "recall": 0.74,
            "f1_score": 0.76,
            "efficiency": 0.61
        },
        "contextual_compression": {
            "precision": 0.77,
            "recall": 0.72,
            "f1_score": 0.74,
            "efficiency": 0.58
        }
    }
    
    # Generate the report
    html_content = generator.generate_validation_report_html(sample_data, "validator_report.html")
    
    print("Sample validator report generated: validator_report.html")
    print(f"Report contains {len(sample_data)} methods with improved labels")
    
    return html_content


if __name__ == "__main__":
    create_sample_report()