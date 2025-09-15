# HTML Performance Report Integration

This document explains how to integrate the HTML performance report generator into the Lethe evaluation pipeline.

## Overview

The HTML report generator creates interactive, professional dashboards from evaluation results, providing:

- **Interactive Leaderboards**: Sortable and filterable tables for each budget tier
- **Champion Summary**: Quick overview of top performers across all budgets  
- **Performance Visualization**: Score bars, family badges, and ranking indicators
- **Methodology Documentation**: Built-in explanation of statistical approach
- **Mobile Responsive**: Works on all device sizes

## Integration Points

### 1. Postprocess Integration

Add HTML generation to the postprocess pipeline:

```python
# In postprocess.py
from lethe.eval.html_reporter import generate_html_if_requested, add_html_reporting_args

def main():
    parser = argparse.ArgumentParser(...)
    # Add HTML reporting args
    add_html_reporting_args(parser)
    
    args = parser.parse_args()
    
    # After generating metrics_summary.csv and advantage_map.json
    html_path = generate_html_if_requested(
        args=args,
        metrics_csv_path=metrics_csv_path,
        advantage_map_path=advantage_map_path, 
        output_dir=output_dir
    )
    
    if html_path:
        print(f"📊 HTML report generated: {html_path}")
```

### 2. Command Line Usage

```bash
# Generate all outputs including HTML report
python -m lethe.eval.postprocess \
  --input raw_results.parquet \
  --out-dir results/ \
  --emit-metrics-csv \
  --emit-advantage-map \
  --emit-html-report

# Custom HTML report name
python -m lethe.eval.postprocess \
  --input raw_results.parquet \
  --out-dir results/ \
  --emit-html-report \
  --html-report-name evaluation_dashboard.html
```

### 3. Standalone Usage

The HTML generator can also be used independently:

```python
from lethe.eval.html_reporter import generate_performance_html

# Generate report from existing files
html_path = generate_performance_html(
    metrics_csv_path="metrics_summary.csv",
    advantage_map_path="advantage_map.json",
    output_dir="reports/",
    run_id="20250911T202527Z"
)

print(f"Report generated: {html_path}")
```

### 4. Command Line Tool

```bash
# Using the standalone script
python html_report_generator.py metrics_summary.csv advantage_map.json report.html

# Using as module
python -m lethe.eval.html_reporter metrics.csv advantage.json
```

## Features

### Interactive Leaderboards

- **Budget Tabs**: 8% (Conservative), 15% (Balanced), 30% (Aggressive)
- **Family Filtering**: Filter by adapter family (long, rag, rerank, selector)
- **Dynamic Sorting**: Sort by score, advantage, latency, or memory usage
- **Live Ranking**: Ranks update automatically based on filters and sorting

### Performance Visualizations

- **Score Bars**: Visual representation of relative performance
- **Family Badges**: Color-coded adapter family indicators
- **Rank Medals**: Gold/Silver/Bronze badges for top 3 positions
- **Champion Cards**: Summary of top performers per budget

### Technical Details

- **Statistical Framework**: Clear explanation of baseline and metrics
- **Performance Metrics**: p95 latency calculations, memory usage tracking
- **Quality Validation**: Placebo baseline validation and significance testing
- **Sample Size**: Comprehensive measurement statistics

## File Structure

```
results/
├── metrics_summary.csv          # Raw statistical data
├── advantage_map.json           # Advantage calculations
├── performance_report.html      # Interactive dashboard ✨
├── raw_results.parquet         # Raw evaluation results
└── validator_report.html       # Quality assurance report
```

## Browser Compatibility

The HTML reports work in all modern browsers:

- ✅ Chrome/Chromium (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Edge
- ✅ Mobile browsers (responsive design)

## Performance

- **File Size**: ~70KB for typical reports (self-contained, no external dependencies)
- **Load Time**: Instant (pure HTML/CSS/JS, no server required)
- **Interactivity**: Real-time filtering and sorting via JavaScript
- **Offline**: Works completely offline once downloaded

## Customization

### Styling

Modify CSS variables in the template to match your brand:

```css
:root {
  --primary-color: #2c3e50;    /* Headers and titles */
  --secondary-color: #3498db;  /* Links and highlights */
  --success-color: #27ae60;    /* Long-context family */
  --warning-color: #f39c12;    /* Rerank family */
}
```

### Content

The template supports customization of:

- Report title and metadata
- Champion summary cards
- Methodology sections
- Performance insights
- Family color coding

## Error Handling

The generator includes comprehensive error handling:

```python
try:
    html_path = generate_performance_html(...)
except FileNotFoundError as e:
    print(f"Input file missing: {e}")
except ValueError as e:
    print(f"Data validation failed: {e}")
except Exception as e:
    print(f"Generation failed: {e}")
```

Common issues and solutions:

- **Missing k=5 data**: Ensure evaluation includes k_value=5.0 measurements
- **Invalid advantage map**: Check that advantage_matrix exists in JSON
- **Budget mismatch**: Verify keep_percentage includes 0.08, 0.15, 0.30
- **Adapter count**: Confirm metrics CSV and advantage map have same adapters

## Examples

### Minimal Integration

```python
# Add to existing postprocess workflow
if args.emit_html_report:
    from lethe.eval.html_reporter import generate_performance_html
    
    html_path = generate_performance_html(
        metrics_csv_path=f"{output_dir}/metrics_summary.csv",
        advantage_map_path=f"{output_dir}/advantage_map.json", 
        output_dir=output_dir
    )
    
    artifacts["performance_report.html"] = html_path
```

### Full Pipeline Integration

```python
def run_evaluation_with_html(input_path, output_dir, emit_html=True):
    """Run complete evaluation with HTML report generation"""
    
    # 1. Run evaluation
    results = evaluate_adapters(input_path)
    
    # 2. Generate statistical outputs
    metrics_path = generate_metrics_csv(results, output_dir)
    advantage_path = generate_advantage_map(results, output_dir)
    
    # 3. Generate HTML report
    if emit_html:
        html_path = generate_performance_html(
            metrics_csv_path=metrics_path,
            advantage_map_path=advantage_path,
            output_dir=output_dir
        )
        
        print(f"📊 Interactive report: {html_path}")
        return html_path
    
    return None
```

## Testing

Test the HTML generation with sample data:

```bash
# Generate test report
cd artifacts/full_matrix_outputs/20250911T202527Z/
python html_report_generator.py metrics_summary.csv advantage_map.json test.html

# Open in browser to verify
open test.html  # macOS
xdg-open test.html  # Linux
start test.html  # Windows
```

## Production Deployment

For production deployments:

1. **Automated Generation**: Add `--emit-html-report` to all evaluation runs
2. **Web Hosting**: HTML reports are self-contained and can be hosted anywhere
3. **Link Sharing**: Share direct URLs to reports for stakeholder review
4. **Archival**: Include HTML reports in evaluation artifacts for historical reference

The HTML reports provide a professional, interactive way to share evaluation results with both technical and non-technical stakeholders.