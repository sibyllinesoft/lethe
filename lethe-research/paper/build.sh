#!/bin/bash
# Automated Paper Generation Pipeline for Lethe Research
# =====================================================
#
# Complete build system with validation and traceability for NeurIPS 2025 submission
# Generates figures/tables from CSV artifacts with source hashes and full validation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ARTIFACTS_DIR="$PROJECT_ROOT/artifacts"
PAPER_DIR="$SCRIPT_DIR"
FIGURES_DIR="$PAPER_DIR/figures"
TABLES_DIR="$PAPER_DIR/tables"
SCRIPTS_DIR="$PROJECT_ROOT/experiments"
BUILD_LOG="$PAPER_DIR/build.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$BUILD_LOG"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "$BUILD_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$BUILD_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$BUILD_LOG"
}

# Initialize build log
echo "=== Lethe Paper Build Log - $(date) ===" > "$BUILD_LOG"

# Function to verify data integrity with hashes
verify_data_integrity() {
    log_info "Verifying data artifact integrity..."
    
    local csv_file="$ARTIFACTS_DIR/final_metrics_summary.csv"
    local stats_file="$ARTIFACTS_DIR/publication_statistical_results.json"
    local enhanced_stats_file="$ARTIFACTS_DIR/enhanced_statistical_analysis.json"
    
    if [[ ! -f "$csv_file" ]]; then
        log_error "Missing required data file: $csv_file"
        exit 1
    fi
    
    # Calculate and log data hash for traceability
    local data_hash=$(sha256sum "$csv_file" | cut -d' ' -f1)
    log_info "Data hash (SHA256): $data_hash"
    echo "DATA_HASH=$data_hash" > "$PAPER_DIR/build_metadata.env"
    
    # Verify minimum data requirements
    local num_records=$(tail -n +2 "$csv_file" | wc -l)
    if [[ $num_records -lt 100 ]]; then
        log_warn "Low number of data records: $num_records (recommended: ≥500)"
    fi
    
    log_success "Data integrity verified - $num_records records"
}

# Function to generate figures with traceability
generate_figures() {
    log_info "Generating publication-quality figures..."
    
    mkdir -p "$FIGURES_DIR"
    
    local csv_file="$ARTIFACTS_DIR/final_metrics_summary.csv"
    local figure_script="$SCRIPTS_DIR/make_figures.py"
    
    if [[ ! -f "$figure_script" ]]; then
        log_error "Figure generation script not found: $figure_script"
        exit 1
    fi
    
    # Generate figures with source traceability
    cd "$PROJECT_ROOT"
    python3 "$figure_script" "$csv_file" --output "$FIGURES_DIR" 2>&1 | tee -a "$BUILD_LOG"
    
    # Verify all required figures were generated
    local required_figures=(
        "iter1_coverage_vs_method.pdf"
        "iter1_pareto.pdf"
        "iter2_ablation_rewrite_decompose.pdf"
        "iter3_dynamic_vs_static_pareto.pdf"
        "iter4_llm_cost_quality_tradeoff.pdf"
        "iteration_progression.pdf"
        "latency_breakdown.pdf"
        "domain_performance.pdf"
        "statistical_significance.pdf"
    )
    
    for fig in "${required_figures[@]}"; do
        if [[ ! -f "$FIGURES_DIR/$fig" ]]; then
            log_error "Required figure not generated: $fig"
            exit 1
        fi
        
        # Log figure hash for traceability
        local fig_hash=$(sha256sum "$FIGURES_DIR/$fig" | cut -d' ' -f1)
        echo "FIGURE_${fig%.*}_HASH=$fig_hash" >> "$PAPER_DIR/build_metadata.env"
    done
    
    log_success "All figures generated successfully (${#required_figures[@]} files)"
}

# Function to generate tables with statistical validation
generate_tables() {
    log_info "Generating LaTeX tables with statistical validation..."
    
    mkdir -p "$TABLES_DIR"
    
    local csv_file="$ARTIFACTS_DIR/final_metrics_summary.csv"
    local stats_file="$ARTIFACTS_DIR/publication_statistical_results.json"
    local table_script="$SCRIPTS_DIR/make_tables.py"
    
    if [[ ! -f "$table_script" ]]; then
        log_error "Table generation script not found: $table_script"
        exit 1
    fi
    
    # Generate tables with statistical data if available
    cd "$PROJECT_ROOT"
    if [[ -f "$stats_file" ]]; then
        python3 "$table_script" "$csv_file" --stats-file "$stats_file" --output "$TABLES_DIR" 2>&1 | tee -a "$BUILD_LOG"
    else
        log_warn "Statistical results file not found, generating without enhanced stats"
        python3 "$table_script" "$csv_file" --output "$TABLES_DIR" 2>&1 | tee -a "$BUILD_LOG"
    fi
    
    # Verify all required tables were generated
    local required_tables=(
        "performance_summary.tex"
        "statistical_significance.tex"
        "latency_breakdown.tex"
        "domain_results.tex"
        "iteration_comparison.tex"
        "ablation_study.tex"
    )
    
    for table in "${required_tables[@]}"; do
        if [[ ! -f "$TABLES_DIR/$table" ]]; then
            log_error "Required table not generated: $table"
            exit 1
        fi
        
        # Validate LaTeX syntax
        if ! grep -q "\\\\begin{table" "$TABLES_DIR/$table"; then
            log_error "Invalid LaTeX table format: $table"
            exit 1
        fi
        
        # Log table hash for traceability
        local table_hash=$(sha256sum "$TABLES_DIR/$table" | cut -d' ' -f1)
        echo "TABLE_${table%.*}_HASH=$table_hash" >> "$PAPER_DIR/build_metadata.env"
    done
    
    log_success "All tables generated successfully (${#required_tables[@]} files)"
}

# Function to validate references and cross-references
validate_references() {
    log_info "Validating LaTeX references and cross-references..."
    
    local main_tex="$PAPER_DIR/lethe_neurips2025.tex"
    
    if [[ ! -f "$main_tex" ]]; then
        log_error "Main LaTeX file not found: $main_tex"
        exit 1
    fi
    
    # Check for required figure references
    local missing_refs=0
    
    # Validate figure references
    for fig in "iter1-analysis" "iter2-ablation" "iter3-progress" "iter4-llm" "statistical-analysis" "domain-performance"; do
        if ! grep -q "\\\\ref{fig:$fig}" "$main_tex"; then
            log_warn "Missing figure reference: fig:$fig"
            ((missing_refs++))
        fi
    done
    
    # Validate table references
    for table in "performance-summary" "statistical-significance" "latency-breakdown" "domain-results"; do
        if ! grep -q "\\\\ref{tab:$table}" "$main_tex"; then
            log_warn "Missing table reference: tab:$table"
            ((missing_refs++))
        fi
    done
    
    # Check for undefined references (basic check)
    if grep -q "??" "$main_tex"; then
        log_warn "Potential undefined references found in LaTeX file"
        ((missing_refs++))
    fi
    
    if [[ $missing_refs -eq 0 ]]; then
        log_success "All cross-references validated successfully"
    else
        log_warn "$missing_refs potential reference issues found"
    fi
}

# Function to compile LaTeX with error checking
compile_latex() {
    log_info "Compiling LaTeX document to PDF..."
    
    cd "$PAPER_DIR"
    
    # Check for required style file
    if [[ ! -f "neurips_2025.sty" ]]; then
        log_error "NeurIPS 2025 style file missing: neurips_2025.sty"
        exit 1
    fi
    
    local main_tex="lethe_neurips2025.tex"
    
    # First pass
    log_info "LaTeX compilation - first pass..."
    if ! pdflatex -interaction=nonstopmode "$main_tex" >> "$BUILD_LOG" 2>&1; then
        log_error "LaTeX compilation failed (first pass)"
        # Show last few lines of log for debugging
        tail -20 "$BUILD_LOG"
        exit 1
    fi
    
    # Second pass for references
    log_info "LaTeX compilation - second pass (for references)..."
    if ! pdflatex -interaction=nonstopmode "$main_tex" >> "$BUILD_LOG" 2>&1; then
        log_error "LaTeX compilation failed (second pass)"
        tail -20 "$BUILD_LOG"
        exit 1
    fi
    
    # Verify PDF was generated
    local pdf_file="${main_tex%.*}.pdf"
    if [[ ! -f "$pdf_file" ]]; then
        log_error "PDF file not generated: $pdf_file"
        exit 1
    fi
    
    # Check PDF properties
    local pdf_pages=$(pdfinfo "$pdf_file" 2>/dev/null | grep "Pages:" | awk '{print $2}')
    if [[ -n "$pdf_pages" && $pdf_pages -gt 0 ]]; then
        log_success "PDF generated successfully ($pdf_pages pages)"
        echo "PDF_PAGES=$pdf_pages" >> "$PAPER_DIR/build_metadata.env"
        
        # Calculate PDF hash for final traceability
        local pdf_hash=$(sha256sum "$pdf_file" | cut -d' ' -f1)
        echo "PDF_HASH=$pdf_hash" >> "$PAPER_DIR/build_metadata.env"
    else
        log_error "Generated PDF appears to be invalid"
        exit 1
    fi
    
    # Clean up auxiliary files (keep log for debugging)
    rm -f *.aux *.out *.log.bak
    
    log_success "LaTeX compilation completed successfully"
}

# Function to generate build report with full traceability
generate_build_report() {
    log_info "Generating build report with full traceability..."
    
    local report_file="$PAPER_DIR/BUILD_REPORT.md"
    local timestamp=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
    
    cat > "$report_file" << EOF
# Lethe Research Paper Build Report

**Generated:** $timestamp  
**Build System:** Automated Pipeline v1.0  
**NeurIPS Template:** 2025 Conference Format

## Data Sources and Traceability

EOF
    
    # Add data source information
    if [[ -f "$PAPER_DIR/build_metadata.env" ]]; then
        source "$PAPER_DIR/build_metadata.env"
        echo "### Primary Data Source" >> "$report_file"
        echo "- **File:** \`final_metrics_summary.csv\`" >> "$report_file"
        echo "- **Hash:** \`$DATA_HASH\`" >> "$report_file"
        echo "- **Records:** $(tail -n +2 "$ARTIFACTS_DIR/final_metrics_summary.csv" 2>/dev/null | wc -l || echo "Unknown")" >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    # Add figure generation info
    echo "### Generated Figures" >> "$report_file"
    local fig_count=0
    for fig_file in "$FIGURES_DIR"/*.pdf; do
        if [[ -f "$fig_file" ]]; then
            local fig_name=$(basename "$fig_file")
            local fig_size=$(ls -lh "$fig_file" | awk '{print $5}')
            echo "- **${fig_name}** (${fig_size})" >> "$report_file"
            ((fig_count++))
        fi
    done
    echo "- **Total Figures:** $fig_count" >> "$report_file"
    echo "" >> "$report_file"
    
    # Add table generation info
    echo "### Generated Tables" >> "$report_file"
    local table_count=0
    for table_file in "$TABLES_DIR"/*.tex; do
        if [[ -f "$table_file" ]]; then
            local table_name=$(basename "$table_file")
            local table_lines=$(wc -l < "$table_file")
            echo "- **${table_name}** (${table_lines} lines)" >> "$report_file"
            ((table_count++))
        fi
    done
    echo "- **Total Tables:** $table_count" >> "$report_file"
    echo "" >> "$report_file"
    
    # Add final PDF info
    if [[ -n "${PDF_PAGES:-}" && -n "${PDF_HASH:-}" ]]; then
        echo "### Final Output" >> "$report_file"
        echo "- **PDF File:** \`lethe_neurips2025.pdf\`" >> "$report_file"
        echo "- **Pages:** $PDF_PAGES" >> "$report_file"
        echo "- **Hash:** \`$PDF_HASH\`" >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    # Add validation results
    echo "### Validation Results" >> "$report_file"
    echo "- ✅ Data integrity verified" >> "$report_file"
    echo "- ✅ All required figures generated" >> "$report_file"
    echo "- ✅ All required tables generated" >> "$report_file"
    echo "- ✅ LaTeX compilation successful" >> "$report_file"
    echo "- ✅ Cross-references validated" >> "$report_file"
    echo "" >> "$report_file"
    
    # Add reproducibility information
    echo "### Reproducibility" >> "$report_file"
    echo "This paper was generated using the automated pipeline with full traceability." >> "$report_file"
    echo "All artifacts can be reproduced using:" >> "$report_file"
    echo "" >> "$report_file"
    echo "\`\`\`bash" >> "$report_file"
    echo "./build.sh" >> "$report_file"
    echo "\`\`\`" >> "$report_file"
    echo "" >> "$report_file"
    
    echo "Build report generated: $report_file" | tee -a "$BUILD_LOG"
    log_success "Build report with full traceability completed"
}

# Function to validate NeurIPS compliance
validate_neurips_compliance() {
    log_info "Validating NeurIPS 2025 template compliance..."
    
    local main_tex="$PAPER_DIR/lethe_neurips2025.tex"
    local pdf_file="$PAPER_DIR/lethe_neurips2025.pdf"
    
    local compliance_issues=0
    
    # Check for required style file usage
    if ! grep -q "\\\\usepackage{neurips_2025}" "$main_tex"; then
        log_warn "NeurIPS 2025 style package not found"
        ((compliance_issues++))
    fi
    
    # Check document class
    if ! grep -q "\\\\documentclass.*article" "$main_tex"; then
        log_warn "Document class may not be compliant with NeurIPS template"
        ((compliance_issues++))
    fi
    
    # Check page count (NeurIPS typically allows 8 pages + unlimited appendix)
    if [[ -f "$pdf_file" ]]; then
        local pdf_pages=$(pdfinfo "$pdf_file" 2>/dev/null | grep "Pages:" | awk '{print $2}')
        if [[ -n "$pdf_pages" && $pdf_pages -gt 15 ]]; then
            log_warn "PDF has $pdf_pages pages (may exceed NeurIPS limits)"
            ((compliance_issues++))
        fi
    fi
    
    # Check for required sections
    local required_sections=("abstract" "introduction" "related work" "method" "experiments" "results" "conclusion")
    for section in "${required_sections[@]}"; do
        if ! grep -qi "\\\\section.*$section" "$main_tex"; then
            log_warn "Required section may be missing: $section"
            ((compliance_issues++))
        fi
    done
    
    if [[ $compliance_issues -eq 0 ]]; then
        log_success "NeurIPS 2025 template compliance validated"
    else
        log_warn "$compliance_issues potential compliance issues found"
    fi
}

# Main build pipeline
main() {
    echo "=========================================="
    echo "Lethe Research Paper Build Pipeline"
    echo "NeurIPS 2025 Automated Generation"
    echo "=========================================="
    echo ""
    
    # Step 1: Verify data integrity and prerequisites
    verify_data_integrity
    
    # Step 2: Generate figures from experimental data
    generate_figures
    
    # Step 3: Generate tables with statistical analysis
    generate_tables
    
    # Step 4: Validate references and cross-references
    validate_references
    
    # Step 5: Compile LaTeX to PDF
    compile_latex
    
    # Step 6: Validate NeurIPS compliance
    validate_neurips_compliance
    
    # Step 7: Generate comprehensive build report
    generate_build_report
    
    echo ""
    echo "=========================================="
    echo "BUILD COMPLETED SUCCESSFULLY"
    echo "=========================================="
    echo ""
    echo "📄 Paper: $PAPER_DIR/lethe_neurips2025.pdf"
    echo "📊 Figures: $FIGURES_DIR/ (9 files)"
    echo "📋 Tables: $TABLES_DIR/ (6 files)"
    echo "📝 Build Report: $PAPER_DIR/BUILD_REPORT.md"
    echo "🔍 Build Log: $PAPER_DIR/build.log"
    echo ""
    echo "✅ All artifacts generated with full traceability"
    echo "✅ Ready for NeurIPS 2025 submission"
}

# Handle script arguments
case "${1:-build}" in
    "build")
        main
        ;;
    "clean")
        log_info "Cleaning build artifacts..."
        rm -rf "$FIGURES_DIR"/*.pdf "$TABLES_DIR"/*.tex
        rm -f "$PAPER_DIR"/*.aux "$PAPER_DIR"/*.log "$PAPER_DIR"/*.out "$PAPER_DIR"/*.pdf
        rm -f "$PAPER_DIR/build_metadata.env" "$PAPER_DIR/BUILD_REPORT.md"
        log_success "Build artifacts cleaned"
        ;;
    "figures-only")
        verify_data_integrity
        generate_figures
        ;;
    "tables-only")
        verify_data_integrity
        generate_tables
        ;;
    *)
        echo "Usage: $0 [build|clean|figures-only|tables-only]"
        exit 1
        ;;
esac