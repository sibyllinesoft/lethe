# Lethe: AI-Powered Context Manager - Professional Release Makefile
# ================================================================
# Professional one-liner commands for all major operations
# Fresh clone + `make eval_all` reproduces headline numbers on target hardware

.PHONY: help build_indices run_service eval_all figures smoke_tests fresh_install \
        setup validate quick_test clean release_check security_scan

# Default target: show help
help:
	@echo "🔥 Lethe: AI-Powered Context Manager"
	@echo "======================================"
	@echo ""
	@echo "🚀 One-liner Commands:"
	@echo "  make build_indices   Build all search indices (BM25, vector, entities)"
	@echo "  make run_service     Launch the agent-context service"
	@echo "  make eval_all       Run complete evaluation pipeline"
	@echo "  make figures        Generate publication plots and tables"
	@echo ""
	@echo "🔧 Development Commands:"
	@echo "  make setup          Initial setup for fresh clone"
	@echo "  make validate       Validate installation and dependencies"
	@echo "  make smoke_tests    Run comprehensive smoke test suite"
	@echo "  make quick_test     Fast validation test (development)"
	@echo ""
	@echo "🛡️ Quality Assurance:"
	@echo "  make security_scan  Run security and privacy validation"
	@echo "  make release_check  Complete release readiness validation"
	@echo "  make clean         Clean all generated artifacts"
	@echo ""
	@echo "📋 Prerequisites:"
	@echo "  - Node.js 18+"
	@echo "  - Python 3.8+"
	@echo "  - 16GB RAM (recommended)"
	@echo "  - 10GB disk space"

# =============================================================================
# PRIMARY ONE-LINER COMMANDS (Milestone 9 Requirements)
# =============================================================================

build_indices: validate
	@echo "🔨 Building all search indices..."
	@echo "This builds BM25 (FTS5), vector (HNSW), and entity indices"
	@cd lethe-research && python scripts/build_indices.py --all --verbose
	@cd ctx-run && npm run build:indices
	@echo "✅ All indices built successfully!"
	@echo "📊 Index statistics:"
	@find . -name "*.db" -o -name "*.idx" -o -name "*.bin" | xargs ls -lh | head -10

run_service: build_indices
	@echo "🚀 Launching Lethe agent-context service..."
	@echo "Service will be available at: http://localhost:8080"
	@echo "Press Ctrl+C to stop the service"
	@cd ctx-run && npm start &
	@cd lethe-research/services && python prediction_service.py &
	@echo "✅ Services launched! Check http://localhost:8080/health"

eval_all: validate build_indices
	@echo "📊 Running complete evaluation pipeline..."
	@echo "This reproduces all headline numbers from the paper"
	@mkdir -p results/$(shell date +%Y%m%d_%H%M%S)
	@cd lethe-research && make reproduce-all
	@cd lethe-research && python run_milestone6_evaluation.py
	@cd lethe-research && python run_milestone7_analysis.py
	@echo "✅ Complete evaluation finished!"
	@echo "📈 Results saved in: results/$(shell date +%Y%m%d_%H%M%S)/"
	@echo "🎯 Headline numbers:"
	@tail -10 lethe-research/analysis/publication_report.json

figures: eval_all
	@echo "📈 Generating publication plots and tables..."
	@cd lethe-research && make figures
	@cd lethe-research/paper && python scripts/generate_figures.py
	@cd lethe-research/paper && python scripts/generate_tables.py
	@echo "✅ All figures and tables generated!"
	@echo "📁 Outputs:"
	@find . -name "*.png" -o -name "*.pdf" -o -name "*.tex" | grep -E "(figure|table)" | sort

# =============================================================================
# QUALITY ASSURANCE COMMANDS
# =============================================================================

smoke_tests: build_indices
	@echo "🔥 Running comprehensive smoke test suite..."
	@python scripts/smoke_tests.py
	@echo "✅ Smoke tests complete!"

security_scan:
	@echo "🔒 Running security and privacy validation..."
	@echo "Checking for secrets, tokens, and privacy compliance..."
	@cd lethe-research && python datasets/privacy_scrubber.py --validate
	@npm audit --audit-level=moderate
	@find . -name "*.py" -o -name "*.ts" -o -name "*.js" | xargs grep -l "password\|secret\|token" | head -5
	@echo "✅ Security scan complete!"

release_check: validate smoke_tests security_scan
	@echo "🚀 Running complete release readiness check..."
	@echo "This validates ALL quality gates before release"
	@python scripts/release_readiness_check.py
	@echo "✅ Release check complete - ready for production!"

# =============================================================================
# SETUP AND VALIDATION COMMANDS  
# =============================================================================

setup: fresh_install validate
	@echo "✅ Lethe setup complete!"

fresh_install:
	@echo "🔧 Setting up Lethe from fresh clone..."
	@echo "Installing Node.js dependencies..."
	@npm install
	@cd ctx-run && npm install
	@echo "Installing Python dependencies..."
	@cd lethe-research && pip install -r requirements.txt
	@cd lethe-research && pip install -r requirements_ir.txt
	@echo "Setting up environment..."
	@cp .env.example .env 2>/dev/null || echo "# Lethe Configuration" > .env
	@echo "✅ Fresh installation complete!"

validate:
	@echo "🔍 Validating Lethe installation..."
	@echo "Checking Node.js version..."
	@node --version | grep -E "v1[89]|v[2-9][0-9]" || (echo "❌ Node.js 18+ required" && exit 1)
	@echo "Checking Python version..."
	@python3 --version | grep -E "Python 3\.[8-9]|Python 3\.[1-9][0-9]" || (echo "❌ Python 3.8+ required" && exit 1)
	@echo "Checking dependencies..."
	@cd ctx-run && npm list --depth=0 >/dev/null || (echo "❌ Node.js deps missing" && exit 1)
	@cd lethe-research && python -c "import sqlite3, numpy, pandas" || (echo "❌ Python deps missing" && exit 1)
	@echo "Checking hardware requirements..."
	@python scripts/benchmark_env_check.sh 2>/dev/null || echo "⚠️  Hardware check script not found"
	@echo "✅ Validation complete!"

quick_test:
	@echo "🧪 Running quick validation test..."
	@cd lethe-research && make baseline-quick-test
	@cd ctx-run && npm test -- --maxWorkers=1
	@echo "✅ Quick test complete!"

# =============================================================================
# MAINTENANCE COMMANDS
# =============================================================================

clean:
	@echo "🧹 Cleaning all generated artifacts..."
	@rm -rf results/
	@rm -rf lethe-research/results/
	@rm -rf lethe-research/artifacts/
	@rm -rf lethe-research/mlruns/
	@rm -rf lethe-research/indices/
	@rm -rf node_modules/.cache/
	@rm -rf ctx-run/node_modules/.cache/
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# =============================================================================
# ADVANCED COMMANDS
# =============================================================================

# Research pipeline with full configuration
research_pipeline:
	@echo "🔬 Running full research pipeline..."
	@cd lethe-research && python scripts/create_lethebench.py
	@cd lethe-research && make baselines
	@cd lethe-research && make figures
	@echo "✅ Research pipeline complete!"

# Benchmark against specific hardware profile
benchmark_hardware:
	@echo "🖥️  Running hardware-specific benchmarks..."
	@cd lethe-research && scripts/benchmark_env_check.sh
	@cd lethe-research && python scripts/run_grid_search.py --hardware-profile auto
	@echo "✅ Hardware benchmarking complete!"

# Development server with hot reload
dev_server:
	@echo "🔧 Starting development server with hot reload..."
	@cd ctx-run && npm run dev &
	@cd packages/devserver && npm start &
	@echo "✅ Development servers running!"
	@echo "🌐 Web interface: http://localhost:3000"
	@echo "📡 API server: http://localhost:8080"

# Performance profiling
profile_performance:
	@echo "⚡ Running performance profiling..."
	@cd lethe-research && python scripts/enhanced_statistical_analysis.py --profile
	@cd ctx-run && npm run perf:profile
	@echo "✅ Performance profiling complete!"

# Update all dependencies
update_deps:
	@echo "⬆️  Updating all dependencies..."
	@npm update
	@cd ctx-run && npm update
	@cd lethe-research && pip install --upgrade -r requirements.txt
	@echo "✅ Dependencies updated!"

# =============================================================================
# HELP AND INFORMATION
# =============================================================================

info:
	@echo "ℹ️  Lethe System Information:"
	@echo "=============================="
	@echo "Version: $(shell cat lethe_version.json | grep version | cut -d'"' -f4)"
	@echo "Node.js: $(shell node --version)"
	@echo "Python: $(shell python3 --version)"
	@echo "Platform: $(shell uname -s) $(shell uname -m)"
	@echo "Memory: $(shell free -h 2>/dev/null | grep Mem | awk '{print $$2}' || echo 'N/A')"
	@echo "Disk: $(shell df -h . | tail -1 | awk '{print $$4 " available"}')"
	@echo ""
	@echo "📁 Directory structure:"
	@tree -L 2 -I 'node_modules|venv|__pycache__|.git' || ls -la

# Show current status
status:
	@echo "📊 Lethe Status:"
	@echo "================"
	@echo "🔧 Installation: $(shell test -f ctx-run/package.json && echo '✅ Complete' || echo '❌ Missing')"
	@echo "📦 Dependencies: $(shell cd ctx-run && npm list >/dev/null 2>&1 && echo '✅ OK' || echo '❌ Issues')"
	@echo "🗃️  Indices: $(shell test -d indices && echo '✅ Built' || echo '❌ Missing')"
	@echo "🚀 Service: $(shell curl -s http://localhost:8080/health >/dev/null 2>&1 && echo '✅ Running' || echo '❌ Stopped')"
	@echo ""
	@echo "Recent activity:"
	@ls -lt results/ 2>/dev/null | head -3 || echo "No results yet"