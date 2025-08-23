#!/bin/bash

# Baseline Evaluation Script
# ==========================
# Evaluates all 7 baseline implementations for comparative analysis

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_DIR="$(dirname "$SCRIPT_DIR")"

# Default parameters
DATASET_PATH="${1:-$RESEARCH_DIR/datasets/lethebench.json}"
OUTPUT_DIR="${2:-$RESEARCH_DIR/artifacts/baselines_$(date +%Y%m%d_%H%M%S)}"
CONFIG_PATH="${3:-$RESEARCH_DIR/experiments/grid_config.yaml}"
MAX_PARALLEL="${4:-4}"

echo "📊 Baseline Evaluation"
echo "======================"
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

# Run baseline evaluation
python3 "$SCRIPT_DIR/baseline_implementations.py" \
    --dataset "$DATASET_PATH" \
    --output "$OUTPUT_DIR" \
    --config "$CONFIG_PATH" \
    --parallel "$MAX_PARALLEL" \
    --verbose

echo
echo "✅ Baseline evaluation complete: $OUTPUT_DIR"