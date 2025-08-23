#!/bin/bash

# Paper Generation Script
# =======================
# Generates LaTeX paper from experimental results

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_DIR="$(dirname "$SCRIPT_DIR")"

# Default parameters
ANALYSIS_DIR="${1:-$RESEARCH_DIR/artifacts/analysis}"
OUTPUT_DIR="${2:-$RESEARCH_DIR/paper/generated_$(date +%Y%m%d_%H%M%S)}"
TEMPLATE_PATH="${3:-$RESEARCH_DIR/paper/template.tex}"

echo "📝 Generating LaTeX Paper"
echo "========================="
echo "Analysis: $ANALYSIS_DIR"
echo "Output: $OUTPUT_DIR"
echo "Template: $TEMPLATE_PATH"
echo

if [ ! -d "$ANALYSIS_DIR" ]; then
    echo "❌ Analysis directory not found: $ANALYSIS_DIR"
    echo "Run statistical analysis first"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate paper
python3 "$SCRIPT_DIR/generate_paper.py" \
    --analysis-dir "$ANALYSIS_DIR" \
    --output "$OUTPUT_DIR" \
    --template "$TEMPLATE_PATH" \
    --verbose

# Compile LaTeX if available
if command -v pdflatex >/dev/null 2>&1; then
    echo "🔧 Compiling LaTeX..."
    cd "$OUTPUT_DIR"
    pdflatex lethe_paper.tex >/dev/null 2>&1 || echo "LaTeX compilation failed"
    pdflatex lethe_paper.tex >/dev/null 2>&1  # Second pass for references
    cd - >/dev/null
    
    if [ -f "$OUTPUT_DIR/lethe_paper.pdf" ]; then
        echo "✅ PDF generated: $OUTPUT_DIR/lethe_paper.pdf"
    fi
else
    echo "⚠️  pdflatex not available - LaTeX source only"
fi

echo
echo "✅ Paper generation complete: $OUTPUT_DIR"