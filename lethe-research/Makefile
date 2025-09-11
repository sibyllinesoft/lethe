# Lethe Research Makefile - Complete Pipeline with Analysis
.PHONY: reproduce-all validate-setup baselines milestone4-baselines baseline-quick-test test-baselines figures milestone7-analysis milestone7-quick ladder ladder-quick

# Quick reproduction check
reproduce-all:
	@echo "🔍 Running reproduction check..."
	python scripts/validate_setup.py
	@echo "✅ Reproduction check complete"

# Setup validation
validate-setup:
	python scripts/validate_setup.py

# Milestone 4: Run all baseline evaluations (main target)
baselines: milestone4-baselines

milestone4-baselines:
	@echo "🚀 Running Milestone 4 baseline evaluation..."
	@echo "This will evaluate 6 baselines on the full LetheBench dataset"
	@mkdir -p results
	python scripts/run_milestone4_baselines.py \
		--dataset datasets/lethebench \
		--output results/milestone4_baselines_$(shell date +%Y%m%d_%H%M%S).json \
		--k 100
	@echo "✅ Baseline evaluation complete! Check results/ directory."

# Quick test with limited queries for development
baseline-quick-test:
	@echo "🧪 Running quick baseline test..."
	@mkdir -p results
	python scripts/run_milestone4_baselines.py \
		--dataset datasets/lethebench \
		--output results/baseline_quick_test.json \
		--k 20 \
		--max-queries 5
	@echo "✅ Quick test complete!"

# Individual baselines for debugging
baseline-bm25-only:
	@mkdir -p results
	python scripts/run_milestone4_baselines.py \
		--dataset datasets/lethebench \
		--output results/bm25_only_test.json \
		--skip-baselines vector_only hybrid_static mmr_diversity doc2query_expansion crossencoder_rerank

baseline-vector-only:
	@mkdir -p results
	python scripts/run_milestone4_baselines.py \
		--dataset datasets/lethebench \
		--output results/vector_only_test.json \
		--skip-baselines bm25_only hybrid_static mmr_diversity doc2query_expansion crossencoder_rerank

# Custom configuration example
baseline-custom:
	@mkdir -p results
	python scripts/run_milestone4_baselines.py \
		--dataset datasets/lethebench \
		--output results/custom_baseline_test.json \
		--k 50 \
		--alpha 0.6 \
		--mmr-lambda 0.8 \
		--doc2query-expansions 5

# Test baseline implementations before running full evaluation
test-baselines:
	@echo "🧪 Testing Milestone 4 baseline implementations..."
	python3 scripts/validate_milestone4_standalone.py
	@echo "✅ Implementation validation complete!"

# ============================================================================
# MILESTONE 7: PUBLICATION-READY ANALYSIS PIPELINE
# ============================================================================

# Generate all figures and tables (main target)
figures: milestone7-analysis

# Complete publication analysis with existing data
milestone7-analysis:
	@echo "📊 Running Milestone 7: Publication-Ready Analysis..."
	@mkdir -p analysis
	python3 run_milestone7_analysis.py \
		--metrics-file analysis/final_statistical_gatekeeper_results.json \
		--train-data datasets/lethebench \
		--test-data datasets/lethebench \
		--output-dir analysis
	@echo "✅ Publication analysis complete! Check analysis/ directory."

# Quick test with synthetic data
milestone7-quick:
	@echo "🧪 Running quick Milestone 7 test with synthetic data..."
	@mkdir -p analysis
	python3 run_milestone7_analysis.py \
		--quick-test \
		--output-dir analysis
	@echo "✅ Quick analysis test complete!"

# Generate only tables (CSV + LaTeX)
tables:
	@echo "📋 Generating publication tables only..."
	python3 run_milestone7_analysis.py \
		--metrics-file analysis/final_statistical_gatekeeper_results.json \
		--train-data datasets/lethebench \
		--test-data datasets/lethebench \
		--output-dir analysis
	@echo "✅ Tables generated in analysis/hardware_profiles/*/tables/"

# Generate only plots (PNG)
plots:
	@echo "📈 Generating publication plots only..."  
	python3 run_milestone7_analysis.py \
		--metrics-file analysis/final_statistical_gatekeeper_results.json \
		--train-data datasets/lethebench \
		--test-data datasets/lethebench \
		--output-dir analysis
	@echo "✅ Plots generated in analysis/hardware_profiles/*/figures/"

# Run sanity checks only
sanity-checks:
	@echo "🔍 Running experimental sanity checks..."
	python3 run_milestone7_analysis.py \
		--metrics-file analysis/final_statistical_gatekeeper_results.json \
		--train-data datasets/lethebench \
		--test-data datasets/lethebench \
		--output-dir analysis
	@echo "✅ Sanity checks complete in analysis/hardware_profiles/*/sanity_checks/"

# Custom hardware profile analysis
milestone7-custom:
	@echo "🖥️  Running analysis with custom hardware profile..."
	@read -p "Enter hardware profile name: " profile_name; \
	python3 run_milestone7_analysis.py \
		--metrics-file analysis/final_statistical_gatekeeper_results.json \
		--train-data datasets/lethebench \
		--test-data datasets/lethebench \
		--output-dir analysis \
		--hardware-profile "$$profile_name"
	@echo "✅ Custom profile analysis complete!"

# Clean analysis outputs
clean-analysis:
	@echo "🧹 Cleaning analysis outputs..."
	rm -rf analysis/hardware_profiles
	rm -f /tmp/synthetic_*.json
	@echo "✅ Analysis outputs cleaned!"

# Show analysis results summary
analysis-summary:
	@echo "📄 Analysis Results Summary:"
	@find analysis/hardware_profiles -name "*.json" -o -name "*.csv" -o -name "*.png" -o -name "*.tex" | sort
	@echo ""
	@echo "📁 Available hardware profiles:"
	@ls -1 analysis/hardware_profiles/ 2>/dev/null || echo "   No profiles generated yet"

# ============================================================================
# DIAGNOSTIC LADDER OF PROOFS - PIPELINE VALIDATION
# ============================================================================

# Run complete diagnostic ladder (main target - 200 samples across tasks)
ladder:
	@echo "🔍 Running complete diagnostic ladder (Rungs 0-3) on 200 mixed samples..."
	@mkdir -p diagnostic_results
	python3 scripts/run_diagnostic_ladder.py \
		--samples_file test_infinitebench_results.json \
		--output_dir diagnostic_results \
		--max_samples 200 \
		--verbose
	@echo "✅ Diagnostic ladder complete! Check diagnostic_results/ for detailed analysis."

# Quick diagnostic ladder (50 samples, essential rungs only)
ladder-quick:
	@echo "🧪 Running quick diagnostic ladder (50 samples, essential rungs)..."
	@mkdir -p diagnostic_results
	python3 scripts/run_diagnostic_ladder.py \
		--samples_file test_infinitebench_results.json \
		--output_dir diagnostic_results \
		--quick \
		--max_samples 50
	@echo "✅ Quick diagnostic complete!"

# Diagnostic ladder with custom evaluation data
ladder-custom:
	@echo "🔧 Running diagnostic ladder with custom data..."
	@read -p "Enter samples file path: " samples_file; \
	@read -p "Enter selected atoms file (optional): " atoms_file; \
	python3 scripts/run_diagnostic_ladder.py \
		--samples_file "$$samples_file" \
		$${atoms_file:+--atoms_file "$$atoms_file"} \
		--output_dir diagnostic_results \
		--max_samples 200 \
		--verbose

# Run only specific rungs for targeted diagnosis
ladder-rung0:
	@echo "🎯 Running Rung 0 only: Scoring sanity checks..."
	python3 scripts/run_diagnostic_ladder.py \
		--samples_file test_infinitebench_results.json \
		--output_dir diagnostic_results \
		--max_samples 100 \
		--skip_curriculum \
		--no_fail_fast

ladder-coverage:
	@echo "🎯 Running Rung 1 only: Coverage analysis..."
	python3 scripts/run_diagnostic_ladder.py \
		--samples_file test_infinitebench_results.json \
		--output_dir diagnostic_results \
		--max_samples 100 \
		--skip_rung0 \
		--skip_curriculum

# Clean diagnostic outputs
clean-diagnostics:
	@echo "🧹 Cleaning diagnostic outputs..."
	rm -rf diagnostic_results
	rm -f diagnostic_ladder.log
	@echo "✅ Diagnostic outputs cleaned!"

# Show diagnostic results summary
diagnostics-summary:
	@echo "📄 Diagnostic Results Summary:"
	@find diagnostic_results -name "*.json" -o -name "*.log" -o -name "*.db" | sort
	@echo ""
	@if [ -f diagnostic_results/diagnostic_ladder_result.json ]; then \
		echo "📊 Latest Diagnostic:"; \
		python3 -c "import json; r=json.load(open('diagnostic_results/diagnostic_ladder_result.json')); print(f'  Diagnosis: {r[\"primary_diagnosis\"]}'); print('  Actions:'); [print(f'    - {a}') for a in r['recommended_actions']]"; \
	else \
		echo "   No diagnostic results found"; \
	fi