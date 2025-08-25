#!/bin/bash

# MLflow Server Setup Script for Lethe Research Framework
# ========================================================
# 
# This script starts the MLflow tracking server for experiment tracking
# as required by Phase 2.4: MLflow Integration
#
# Usage: ./start_mlflow_server.sh [--port PORT] [--host HOST] [--backend-store-uri URI]
# 
# Default configuration matches Phase 2.4 requirements:
# - Backend store: ./mlruns (local filesystem)
# - Host: 127.0.0.1 (localhost)  
# - Port: 5000

set -euo pipefail

# Default configuration
DEFAULT_HOST="127.0.0.1"
DEFAULT_PORT="5000"
DEFAULT_BACKEND_STORE="./mlruns"
DEFAULT_ARTIFACT_ROOT="./mlruns"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
HOST="$DEFAULT_HOST"
PORT="$DEFAULT_PORT"
BACKEND_STORE_URI="$DEFAULT_BACKEND_STORE"
ARTIFACT_ROOT="$DEFAULT_ARTIFACT_ROOT"
WORKERS="1"
LOG_LEVEL="INFO"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date +'%Y-%m-%d %H:%M:%S') $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date +'%Y-%m-%d %H:%M:%S') $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date +'%Y-%m-%d %H:%M:%S') $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date +'%Y-%m-%d %H:%M:%S') $1"
}

# Help function
show_help() {
    cat << EOF
MLflow Server Setup Script for Lethe Research Framework

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -h, --help                    Show this help message
    --host HOST                   Host to bind to (default: $DEFAULT_HOST)
    --port PORT                   Port to bind to (default: $DEFAULT_PORT)
    --backend-store-uri URI       Backend store URI (default: $DEFAULT_BACKEND_STORE)
    --artifact-root ROOT          Artifact root directory (default: $DEFAULT_ARTIFACT_ROOT)
    --workers N                   Number of worker processes (default: 1)
    --log-level LEVEL             Log level (default: INFO)
    --check-only                  Only check if MLflow is available
    --daemon                      Run as daemon process
    --stop                        Stop any running MLflow server

EXAMPLES:
    # Start with default settings (Phase 2.4 configuration)
    $0
    
    # Start on different port
    $0 --port 5001
    
    # Use remote database backend
    $0 --backend-store-uri postgresql://user:pass@localhost/mlflow
    
    # Check if MLflow is available
    $0 --check-only

NOTES:
    - Default configuration matches Phase 2.4 requirements
    - Backend store './mlruns' will be created if it doesn't exist
    - Server will be accessible at http://$DEFAULT_HOST:$DEFAULT_PORT
    - Use Ctrl+C to stop the server (or --stop option)
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --backend-store-uri)
            BACKEND_STORE_URI="$2"
            shift 2
            ;;
        --artifact-root)
            ARTIFACT_ROOT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --check-only)
            CHECK_ONLY="true"
            shift
            ;;
        --daemon)
            DAEMON="true"
            shift
            ;;
        --stop)
            STOP_SERVER="true"
            shift
            ;;
        *)
            log_error "Unknown argument: $1"
            show_help
            exit 1
            ;;
    esac
done

# Function to check if MLflow is available
check_mlflow() {
    log_info "Checking MLflow availability..."
    
    if ! command -v mlflow >/dev/null 2>&1; then
        log_error "MLflow not found. Please install it:"
        echo "  pip install mlflow>=2.10.0"
        echo "  # Or install from requirements:"
        echo "  pip install -r $RESEARCH_DIR/experiments/requirements.txt"
        return 1
    fi
    
    mlflow_version=$(mlflow --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
    log_success "MLflow version $mlflow_version is available"
    return 0
}

# Function to check if server is already running
check_server_running() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to stop existing server
stop_server() {
    log_info "Stopping MLflow server on port $PORT..."
    
    if check_server_running $PORT; then
        local pids=$(lsof -ti :$PORT)
        for pid in $pids; do
            log_info "Stopping process $pid"
            kill -TERM $pid 2>/dev/null || true
            sleep 2
            if kill -0 $pid 2>/dev/null; then
                log_warning "Process $pid didn't stop gracefully, forcing..."
                kill -KILL $pid 2>/dev/null || true
            fi
        done
        log_success "Stopped MLflow server"
    else
        log_warning "No MLflow server running on port $PORT"
    fi
}

# Function to create backend store directory
setup_backend_store() {
    if [[ "$BACKEND_STORE_URI" == ./* ]] || [[ "$BACKEND_STORE_URI" != *://* ]]; then
        # Local filesystem backend
        local store_dir
        if [[ "$BACKEND_STORE_URI" == ./* ]]; then
            store_dir="$RESEARCH_DIR/$BACKEND_STORE_URI"
        else
            store_dir="$BACKEND_STORE_URI"
        fi
        
        if [[ ! -d "$store_dir" ]]; then
            log_info "Creating backend store directory: $store_dir"
            mkdir -p "$store_dir"
        fi
        
        # Update to absolute path
        BACKEND_STORE_URI="$(cd "$store_dir" && pwd)"
    fi
    
    # Setup artifact root similarly
    if [[ "$ARTIFACT_ROOT" == ./* ]] || [[ "$ARTIFACT_ROOT" != *://* ]]; then
        local artifact_dir
        if [[ "$ARTIFACT_ROOT" == ./* ]]; then
            artifact_dir="$RESEARCH_DIR/$ARTIFACT_ROOT"
        else
            artifact_dir="$ARTIFACT_ROOT"
        fi
        
        if [[ ! -d "$artifact_dir" ]]; then
            log_info "Creating artifact root directory: $artifact_dir"
            mkdir -p "$artifact_dir"
        fi
        
        # Update to absolute path
        ARTIFACT_ROOT="$(cd "$artifact_dir" && pwd)"
    fi
}

# Function to start MLflow server
start_server() {
    log_info "Starting MLflow server..."
    log_info "Configuration:"
    log_info "  Host: $HOST"
    log_info "  Port: $PORT"
    log_info "  Backend Store: $BACKEND_STORE_URI"
    log_info "  Artifact Root: $ARTIFACT_ROOT"
    log_info "  Workers: $WORKERS"
    log_info "  Log Level: $LOG_LEVEL"
    
    # Check if already running
    if check_server_running $PORT; then
        log_error "Server already running on port $PORT"
        log_info "Use --stop to stop the existing server"
        exit 1
    fi
    
    # Setup directories
    setup_backend_store
    
    # Build command
    local mlflow_cmd=(
        mlflow
        server
        --backend-store-uri "$BACKEND_STORE_URI"
        --default-artifact-root "$ARTIFACT_ROOT"
        --host "$HOST"
        --port "$PORT"
        --workers "$WORKERS"
    )
    
    # Add logging configuration
    export MLFLOW_TRACKING_USERNAME=""
    export MLFLOW_TRACKING_PASSWORD=""
    
    # Start server
    log_info "Command: ${mlflow_cmd[*]}"
    log_success "Starting MLflow server..."
    log_info "Server will be available at: http://$HOST:$PORT"
    log_info "Use Ctrl+C to stop the server"
    echo
    
    if [[ "${DAEMON:-}" == "true" ]]; then
        log_info "Starting in daemon mode..."
        nohup "${mlflow_cmd[@]}" > mlflow-server.log 2>&1 &
        local server_pid=$!
        echo $server_pid > mlflow-server.pid
        log_success "MLflow server started as daemon (PID: $server_pid)"
        log_info "Logs: mlflow-server.log"
        log_info "Stop with: kill $server_pid"
    else
        # Run in foreground
        exec "${mlflow_cmd[@]}"
    fi
}

# Function to show status
show_status() {
    log_info "MLflow Server Status"
    log_info "==================="
    
    if check_server_running $PORT; then
        log_success "Server is running on port $PORT"
        log_info "URL: http://$HOST:$PORT"
        
        # Show process info
        local pids=$(lsof -ti :$PORT)
        for pid in $pids; do
            local cmd=$(ps -p $pid -o comm= 2>/dev/null || echo "unknown")
            log_info "Process: $pid ($cmd)"
        done
    else
        log_warning "No server running on port $PORT"
    fi
}

# Main execution
main() {
    cd "$RESEARCH_DIR"
    
    echo "================================================================="
    echo "🚀 MLflow Server Setup - Lethe Research Framework"
    echo "================================================================="
    echo "Phase 2.4: MLflow Integration"
    echo "================================================================="
    echo
    
    # Check MLflow availability first
    if ! check_mlflow; then
        exit 1
    fi
    
    # Handle different operations
    if [[ "${CHECK_ONLY:-}" == "true" ]]; then
        log_success "MLflow is properly installed and available"
        show_status
        exit 0
    elif [[ "${STOP_SERVER:-}" == "true" ]]; then
        stop_server
        exit 0
    else
        start_server
    fi
}

# Handle interrupts gracefully
trap 'log_info "Shutting down MLflow server..."; exit 0' INT TERM

# Execute main function
main "$@"