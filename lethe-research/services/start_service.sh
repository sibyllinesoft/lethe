#!/bin/bash
# Start the Lethe ML Prediction Service
# Workstream A, Phase 2.1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Starting Lethe ML Prediction Service..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Set Python path to include experiments
export PYTHONPATH="$SCRIPT_DIR/../experiments:$PYTHONPATH"

# Default configuration
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8080}
LOG_LEVEL=${LOG_LEVEL:-info}
WORKERS=${WORKERS:-1}

echo "🌐 Service will run on http://$HOST:$PORT"
echo "📊 Log level: $LOG_LEVEL"
echo "⚙️  Workers: $WORKERS"
echo ""

# Start the service
python prediction_service.py \
    --host "$HOST" \
    --port "$PORT" \
    --log-level "$LOG_LEVEL" \
    --workers "$WORKERS"