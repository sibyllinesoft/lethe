#!/bin/bash

# LetheBench Dataset Creation Script
# ==================================
# Creates the evaluation dataset from existing ctx-run examples
# and generates synthetic queries for comprehensive evaluation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$RESEARCH_DIR")"

# Default parameters
OUTPUT_PATH="${1:-$RESEARCH_DIR/datasets/lethebench.json}"
EXAMPLES_DIR="${2:-$PROJECT_DIR/ctx-run/examples}"
CONFIG_PATH="${3:-$RESEARCH_DIR/experiments/grid_config.yaml}"

echo "🔧 Creating LetheBench Dataset"
echo "=============================="
echo "Examples: $EXAMPLES_DIR"
echo "Output: $OUTPUT_PATH"
echo "Config: $CONFIG_PATH"
echo

# Create output directory
mkdir -p "$(dirname "$OUTPUT_PATH")"

# Run Python dataset creation script
python3 "$SCRIPT_DIR/create_lethebench.py" \
    --examples-dir "$EXAMPLES_DIR" \
    --output "$OUTPUT_PATH" \
    --config "$CONFIG_PATH" \
    --verbose

echo
echo "✅ LetheBench dataset created: $OUTPUT_PATH"