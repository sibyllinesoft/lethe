#!/bin/bash

# Lethe Grid Search Script
# ========================
# Executes parameter grid search for Lethe system optimization

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$RESEARCH_DIR")"

# Default parameters
DATASET_PATH="${1:-$RESEARCH_DIR/datasets/lethebench.json}"
OUTPUT_DIR="${2:-$RESEARCH_DIR/artifacts/grid_search_$(date +%Y%m%d_%H%M%S)}"
CONFIG_PATH="${3:-$RESEARCH_DIR/experiments/grid_config.yaml}"
MAX_PARALLEL="${4:-4}"

echo "🔬 Lethe Parameter Grid Search"
echo "=============================="
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Config: $CONFIG_PATH"
echo "Parallel: $MAX_PARALLEL"
echo

if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Dataset not found: $DATASET_PATH"
    echo "Run create_dataset.sh first"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run Python grid search
python3 "$SCRIPT_DIR/run_grid_search.py" \
    --dataset "$DATASET_PATH" \
    --output "$OUTPUT_DIR" \
    --config "$CONFIG_PATH" \
    --ctx-run-path "$PROJECT_DIR/ctx-run/packages/cli/dist/index.js" \
    --parallel "$MAX_PARALLEL" \
    --verbose

echo
echo "✅ Grid search complete: $OUTPUT_DIR"