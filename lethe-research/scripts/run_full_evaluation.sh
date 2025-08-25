#!/bin/bash

# Lethe Full Evaluation Pipeline
# ===============================
# Complete automated research pipeline from dataset creation to paper generation
#
# Usage: ./run_full_evaluation.sh [--config CONFIG_PATH] [--output OUTPUT_DIR]
#
# Dependencies: All components must pass health check
# Expected runtime: 4-8 hours depending on grid size
# Output: Complete research artifacts ready for NeurIPS submission

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$RESEARCH_DIR")"
CTX_RUN_DIR="$PROJECT_DIR/ctx-run"

# Default configuration
CONFIG_PATH="${1:-$RESEARCH_DIR/experiments/grid_config.yaml}"
OUTPUT_DIR="${2:-$RESEARCH_DIR/artifacts/$(date +%Y%m%d_%H%M%S)}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
SKIP_BASELINES="${SKIP_BASELINES:-false}"
SKIP_DATASET="${SKIP_DATASET:-false}"

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

# Function to check dependencies
check_dependencies() {
    log_info "Checking system dependencies..."
    
    # Check required commands
    local required_commands=("python3" "node" "npm" "git")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            log_error "Required command not found: $cmd"
            exit 1
        fi
    done
    
    # Check Python packages including MLflow
    python3 -c "import numpy, pandas, scipy, sklearn, mlflow" 2>/dev/null || {
        log_error "Required Python packages missing. Install requirements:"
        log_error "  pip install -r $RESEARCH_DIR/experiments/requirements.txt"
        exit 1
    }
    
    # Check ctx-run health
    if ! cd "$CTX_RUN_DIR" && npm run build >/dev/null 2>&1; then
        log_error "Failed to build ctx-run. Check dependencies."
        exit 1
    fi
    
    # Test ctx-run diagnose in a temp directory
    local temp_dir=$(mktemp -d)
    cd "$temp_dir"
    if ! "$CTX_RUN_DIR/packages/cli/dist/index.js" init . >/dev/null 2>&1; then
        log_error "Failed to initialize Lethe context"
        rm -rf "$temp_dir"
        exit 1
    fi
    
    if ! "$CTX_RUN_DIR/packages/cli/dist/index.js" diagnose >/dev/null 2>&1; then
        log_error "Lethe health check failed"
        rm -rf "$temp_dir"
        exit 1
    fi
    rm -rf "$temp_dir"
    
    log_success "All dependencies verified"
}

# Function to setup environment
setup_environment() {
    log_info "Setting up evaluation environment..."
    
    # Create output directory structure
    mkdir -p "$OUTPUT_DIR"/{datasets,baselines,lethe_runs,analysis,logs,figures}
    
    # Copy configuration
    cp "$CONFIG_PATH" "$OUTPUT_DIR/config.yaml"
    
    # Setup MLflow tracking directory
    local mlflow_dir="$RESEARCH_DIR/mlruns"
    mkdir -p "$mlflow_dir"
    
    # Create environment snapshot
    cat > "$OUTPUT_DIR/environment.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "hostname": "$(hostname)",
    "pwd": "$(pwd)",
    "git_sha": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')",
    "node_version": "$(node --version)",
    "python_version": "$(python3 --version)",
    "mlflow_version": "$(python3 -c 'import mlflow; print(mlflow.__version__)' 2>/dev/null || echo 'unknown')",
    "config_path": "$CONFIG_PATH",
    "output_dir": "$OUTPUT_DIR",
    "max_parallel": "$MAX_PARALLEL",
    "mlflow_tracking_uri": "$mlflow_dir"
}
EOF
    
    log_success "Environment setup complete: $OUTPUT_DIR"
}

# Function to start MLflow tracking server
start_mlflow_server() {
    log_info "Starting MLflow tracking server..."
    
    local mlflow_port=5000
    local mlflow_host="127.0.0.1"
    local mlflow_dir="$RESEARCH_DIR/mlruns"
    
    # Check if MLflow server is already running
    if lsof -Pi :$mlflow_port -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_info "MLflow server already running on port $mlflow_port"
        return 0
    fi
    
    # Start MLflow server in background
    log_info "Starting MLflow server on http://$mlflow_host:$mlflow_port"
    log_info "Backend store: $mlflow_dir"
    
    cd "$RESEARCH_DIR"
    nohup mlflow server \
        --backend-store-uri "$mlflow_dir" \
        --default-artifact-root "$mlflow_dir" \
        --host "$mlflow_host" \
        --port "$mlflow_port" \
        > "$OUTPUT_DIR/logs/mlflow_server.log" 2>&1 &
    
    local mlflow_pid=$!
    echo $mlflow_pid > "$OUTPUT_DIR/mlflow_server.pid"
    
    # Wait for server to be ready
    log_info "Waiting for MLflow server to start..."
    local retries=0
    local max_retries=30
    
    while [[ $retries -lt $max_retries ]]; do
        if curl -s "http://$mlflow_host:$mlflow_port" >/dev/null 2>&1; then
            log_success "MLflow server is ready at http://$mlflow_host:$mlflow_port"
            log_info "MLflow UI: http://$mlflow_host:$mlflow_port"
            return 0
        fi
        
        if ! kill -0 $mlflow_pid 2>/dev/null; then
            log_error "MLflow server failed to start. Check logs: $OUTPUT_DIR/logs/mlflow_server.log"
            return 1
        fi
        
        sleep 2
        retries=$((retries + 1))
    done
    
    log_error "MLflow server failed to start within timeout"
    return 1
}

# Function to stop MLflow server
stop_mlflow_server() {
    if [[ -f "$OUTPUT_DIR/mlflow_server.pid" ]]; then
        local mlflow_pid=$(cat "$OUTPUT_DIR/mlflow_server.pid")
        if kill -0 $mlflow_pid 2>/dev/null; then
            log_info "Stopping MLflow server (PID: $mlflow_pid)..."
            kill -TERM $mlflow_pid
            sleep 2
            if kill -0 $mlflow_pid 2>/dev/null; then
                kill -KILL $mlflow_pid
            fi
            rm -f "$OUTPUT_DIR/mlflow_server.pid"
            log_success "MLflow server stopped"
        fi
    fi
}

# Function to create or load dataset
prepare_dataset() {
    if [ "$SKIP_DATASET" == "true" ]; then
        log_info "Skipping dataset creation (SKIP_DATASET=true)"
        return
    fi
    
    log_info "Preparing LetheBench dataset..."
    
    local dataset_path="$OUTPUT_DIR/datasets/lethebench.json"
    
    if [ -f "$dataset_path" ] && [ "${FORCE_DATASET:-false}" != "true" ]; then
        log_info "Dataset already exists, skipping creation"
        return
    fi
    
    # Run dataset creation script
    python3 "$SCRIPT_DIR/create_lethebench.py" \
        --output "$dataset_path" \
        --examples-dir "$CTX_RUN_DIR/examples" \
        --config "$CONFIG_PATH" \
        --log-level "$LOG_LEVEL" 2>&1 | tee "$OUTPUT_DIR/logs/dataset_creation.log"
    
    if [ ! -f "$dataset_path" ]; then
        log_error "Dataset creation failed"
        exit 1
    fi
    
    log_success "LetheBench dataset ready: $dataset_path"
}

# Function to run baseline evaluations
run_baseline_evaluations() {
    if [ "$SKIP_BASELINES" == "true" ]; then
        log_info "Skipping baseline evaluations (SKIP_BASELINES=true)"
        return
    fi
    
    log_info "Running baseline evaluations..."
    
    local dataset_path="$OUTPUT_DIR/datasets/lethebench.json"
    local baseline_output="$OUTPUT_DIR/baselines"
    
    if [ ! -f "$dataset_path" ]; then
        log_error "Dataset not found: $dataset_path"
        exit 1
    fi
    
    # Run baseline implementations
    python3 "$SCRIPT_DIR/baseline_implementations.py" \
        --dataset "$dataset_path" \
        --output "$baseline_output" \
        --config "$CONFIG_PATH" \
        --parallel "$MAX_PARALLEL" \
        --log-level "$LOG_LEVEL" 2>&1 | tee "$OUTPUT_DIR/logs/baseline_evaluation.log"
    
    if [ ! -d "$baseline_output" ] || [ -z "$(ls -A "$baseline_output")" ]; then
        log_error "Baseline evaluation failed or produced no results"
        exit 1
    fi
    
    log_success "Baseline evaluations complete: $baseline_output"
}

# Function to run Lethe grid search with MLflow tracking
run_lethe_grid_search() {
    log_info "Running Lethe parameter grid search with MLflow tracking..."
    
    local dataset_path="$OUTPUT_DIR/datasets/lethebench.json"
    local lethe_output="$OUTPUT_DIR/lethe_runs"
    local mlflow_uri="$RESEARCH_DIR/mlruns"
    
    if [ ! -f "$dataset_path" ]; then
        log_error "Dataset not found: $dataset_path"
        exit 1
    fi
    
    # Run grid search with MLflow integration
    cd "$RESEARCH_DIR/experiments"
    python3 run.py \
        --config "$CONFIG_PATH" \
        --output "$lethe_output" \
        --workers "$MAX_PARALLEL" \
        --mlflow-tracking-uri "$mlflow_uri" \
        --mlflow-experiment-name "lethe_full_evaluation_$(date +%Y%m%d_%H%M%S)" \
        2>&1 | tee "$OUTPUT_DIR/logs/grid_search_mlflow.log"
    
    if [ $? -ne 0 ]; then
        log_error "Lethe grid search with MLflow tracking failed"
        exit 1
    fi
    
    if [ ! -d "$lethe_output" ] || [ -z "$(ls -A "$lethe_output")" ]; then
        log_error "Lethe grid search produced no results"
        exit 1
    fi
    
    log_success "Lethe grid search with MLflow tracking complete: $lethe_output"
    log_info "View experiment results at: http://127.0.0.1:5000"
}

# Function to run statistical analysis
run_statistical_analysis() {
    log_info "Running statistical analysis..."
    
    local baseline_results="$OUTPUT_DIR/baselines"
    local lethe_results="$OUTPUT_DIR/lethe_runs"
    local analysis_output="$OUTPUT_DIR/analysis"
    
    if [ ! -d "$baseline_results" ] || [ ! -d "$lethe_results" ]; then
        log_error "Missing evaluation results for analysis"
        exit 1
    fi
    
    # Run comprehensive analysis
    python3 "$SCRIPT_DIR/run_analysis.py" \
        --baseline-results "$baseline_results" \
        --lethe-results "$lethe_results" \
        --output "$analysis_output" \
        --config "$CONFIG_PATH" \
        --hypothesis-framework "$RESEARCH_DIR/experiments/hypothesis_framework.json" \
        --log-level "$LOG_LEVEL" 2>&1 | tee "$OUTPUT_DIR/logs/statistical_analysis.log"
    
    if [ ! -f "$analysis_output/summary_report.json" ]; then
        log_error "Statistical analysis failed"
        exit 1
    fi
    
    log_success "Statistical analysis complete: $analysis_output"
}

# Function to generate visualizations
generate_visualizations() {
    log_info "Generating visualizations..."
    
    local analysis_results="$OUTPUT_DIR/analysis"
    local figures_output="$OUTPUT_DIR/figures"
    
    if [ ! -f "$analysis_results/summary_report.json" ]; then
        log_error "Analysis results not found"
        exit 1
    fi
    
    # Generate plots and figures
    python3 "$SCRIPT_DIR/generate_figures.py" \
        --analysis-results "$analysis_results" \
        --output "$figures_output" \
        --config "$CONFIG_PATH" \
        --log-level "$LOG_LEVEL" 2>&1 | tee "$OUTPUT_DIR/logs/visualization.log"
    
    if [ ! -d "$figures_output" ] || [ -z "$(ls -A "$figures_output")" ]; then
        log_error "Visualization generation failed"
        exit 1
    fi
    
    log_success "Visualizations complete: $figures_output"
}

# Function to generate paper
generate_paper() {
    log_info "Generating LaTeX paper..."
    
    local analysis_results="$OUTPUT_DIR/analysis"
    local figures_dir="$OUTPUT_DIR/figures"
    local paper_output="$OUTPUT_DIR/paper"
    
    if [ ! -f "$analysis_results/summary_report.json" ] || [ ! -d "$figures_dir" ]; then
        log_error "Missing analysis results or figures for paper generation"
        exit 1
    fi
    
    # Generate LaTeX paper
    python3 "$SCRIPT_DIR/generate_paper.py" \
        --analysis-results "$analysis_results" \
        --figures-dir "$figures_dir" \
        --template "$RESEARCH_DIR/paper/template.tex" \
        --output "$paper_output" \
        --config "$CONFIG_PATH" \
        --log-level "$LOG_LEVEL" 2>&1 | tee "$OUTPUT_DIR/logs/paper_generation.log"
    
    if [ ! -f "$paper_output/lethe_paper.tex" ]; then
        log_error "Paper generation failed"
        exit 1
    fi
    
    # Compile LaTeX if pdflatex available
    if command -v pdflatex >/dev/null 2>&1; then
        log_info "Compiling LaTeX to PDF..."
        cd "$paper_output"
        pdflatex lethe_paper.tex >/dev/null 2>&1 && \
        pdflatex lethe_paper.tex >/dev/null 2>&1  # Run twice for references
        cd - >/dev/null
        
        if [ -f "$paper_output/lethe_paper.pdf" ]; then
            log_success "Paper compiled to PDF: $paper_output/lethe_paper.pdf"
        else
            log_warning "PDF compilation failed, but LaTeX source is available"
        fi
    else
        log_warning "pdflatex not available, only LaTeX source generated"
    fi
    
    log_success "Paper generation complete: $paper_output"
}

# Function to validate results
validate_results() {
    log_info "Validating research results..."
    
    local validation_errors=0
    
    # Check that all expected outputs exist
    local required_files=(
        "$OUTPUT_DIR/datasets/lethebench.json"
        "$OUTPUT_DIR/analysis/summary_report.json"
        "$OUTPUT_DIR/figures"
        "$OUTPUT_DIR/paper/lethe_paper.tex"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -e "$file" ]; then
            log_error "Missing required output: $file"
            ((validation_errors++))
        fi
    done
    
    # Run sanity checks on results
    python3 "$SCRIPT_DIR/validate_results.py" \
        --results-dir "$OUTPUT_DIR" \
        --log-level "$LOG_LEVEL" 2>&1 | tee "$OUTPUT_DIR/logs/validation.log"
    
    local validation_exit_code=$?
    if [ $validation_exit_code -ne 0 ]; then
        log_error "Results validation failed"
        ((validation_errors++))
    fi
    
    if [ $validation_errors -eq 0 ]; then
        log_success "All validation checks passed"
        return 0
    else
        log_error "Validation failed with $validation_errors errors"
        return 1
    fi
}

# Function to generate final report
generate_final_report() {
    log_info "Generating final research report..."
    
    local summary_path="$OUTPUT_DIR/RESEARCH_SUMMARY.md"
    
    cat > "$summary_path" << EOF
# Lethe Research Results

**Generated**: $(date -Iseconds)  
**Configuration**: $(basename "$CONFIG_PATH")  
**Total Runtime**: ${SECONDS}s

## 🎯 Hypothesis Results

EOF
    
    # Extract key findings from analysis
    if [ -f "$OUTPUT_DIR/analysis/summary_report.json" ]; then
        python3 -c "
import json
with open('$OUTPUT_DIR/analysis/summary_report.json') as f:
    data = json.load(f)
    
print('### H1 (Quality)')
for metric in ['ndcg_at_10', 'recall_at_10', 'mrr_at_10']:
    if metric in data.get('hypothesis_results', {}):
        result = data['hypothesis_results'][metric]
        status = '✅ SUPPORTED' if result.get('significant', False) else '❌ NOT SUPPORTED'
        print(f'- **{metric.upper()}**: {status} (p={result.get(\"p_value\", \"N/A\"):.4f})')

print('\n### H2 (Efficiency)')
efficiency = data.get('efficiency_metrics', {})
latency_p95 = efficiency.get('latency_p95', 'N/A')
memory_peak = efficiency.get('memory_peak', 'N/A')
print(f'- **Latency P95**: {latency_p95}ms (target: <3000ms)')
print(f'- **Memory Peak**: {memory_peak}MB (target: <1500MB)')

print('\n### H3 (Coverage)')
coverage = data.get('coverage_metrics', {})
for n in [10, 20, 50]:
    cov = coverage.get(f'coverage_at_{n}', 'N/A')
    print(f'- **Coverage@{n}**: {cov}')

print('\n### H4 (Consistency)')  
consistency = data.get('consistency_metrics', {})
contradiction_rate = consistency.get('contradiction_rate', 'N/A')
print(f'- **Contradiction Rate**: {contradiction_rate}%')
" >> "$summary_path"
    fi
    
    cat >> "$summary_path" << EOF

## 📁 Generated Artifacts

- **Dataset**: \`datasets/lethebench.json\`
- **Baseline Results**: \`baselines/\`
- **Lethe Results**: \`lethe_runs/\`  
- **Analysis**: \`analysis/summary_report.json\`
- **Figures**: \`figures/\`
- **Paper**: \`paper/lethe_paper.tex\`
- **MLflow Tracking**: \`mlruns/\` (Phase 2.4 Integration)

## 🔬 MLflow Experiment Tracking

**MLflow UI**: [http://127.0.0.1:5000](http://127.0.0.1:5000)

The MLflow tracking system provides comprehensive experiment tracking including:
- All grid search parameters logged automatically
- Key metrics: \`ndcg_at_10\`, \`recall_at_50\`, \`latency_p95\`, \`memory_peak\`
- Model artifacts (*.joblib files) for reproducibility
- Git commit SHA for full reproducibility
- Experiment comparison and visualization tools

## 🚀 Next Steps

1. **Review MLflow Experiments**: Open http://127.0.0.1:5000 to explore all runs
2. **Analyze Results**: Review \`analysis/summary_report.json\` and MLflow metrics
3. **Compare Configurations**: Use MLflow UI to compare parameter combinations
4. **Export Best Models**: Download model artifacts from top-performing runs
5. **Generate Figures**: Examine visualizations in \`figures/\`
6. **Review Paper**: Read draft in \`paper/lethe_paper.tex\`
7. **Validate Results**: Address any warnings in logs
8. **Reproduce Results**: Use Git SHA and MLflow artifacts for exact reproduction
9. **Submit to NeurIPS 2025**: All evidence and artifacts ready!

---
*Generated by Lethe Research Framework*
EOF
    
    log_success "Final report generated: $summary_path"
}

# Main execution flow
main() {
    local start_time=$(date +%s)
    
    echo "================================================================="
    echo "🚀 Lethe Full Evaluation Pipeline"
    echo "================================================================="
    echo "Config: $CONFIG_PATH"
    echo "Output: $OUTPUT_DIR"
    echo "Parallel: $MAX_PARALLEL"
    echo "================================================================="
    echo
    
    # Setup
    check_dependencies
    setup_environment
    
    # Start MLflow tracking server
    if ! start_mlflow_server; then
        log_error "Failed to start MLflow server"
        exit 1
    fi
    
    # Data preparation
    prepare_dataset
    
    # Evaluations
    run_baseline_evaluations
    run_lethe_grid_search
    
    # Analysis and reporting
    run_statistical_analysis
    generate_visualizations
    generate_paper
    
    # Validation
    if ! validate_results; then
        log_warning "Some validation checks failed - review logs before proceeding"
    fi
    
    # Final report
    generate_final_report
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo
    echo "================================================================="
    log_success "🎉 Full evaluation pipeline complete!"
    echo "⏱️  Total runtime: ${duration}s"
    echo "📁 Results: $OUTPUT_DIR"
    echo "📄 Summary: $OUTPUT_DIR/RESEARCH_SUMMARY.md"
    echo "🔬 MLflow UI: http://127.0.0.1:5000"
    echo "================================================================="
    echo
    log_info "MLflow server is still running for experiment analysis"
    log_info "To stop the MLflow server: kill \$(cat $OUTPUT_DIR/mlflow_server.pid)"
}

# Handle interrupts gracefully
cleanup_on_exit() {
    log_info "Cleaning up..."
    stop_mlflow_server
    log_error "Pipeline interrupted by user"
    exit 130
}

trap 'cleanup_on_exit' INT TERM

# Execute main function
main "$@"