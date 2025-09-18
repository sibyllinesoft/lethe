# Suggested Commands

## Core Development Commands
```bash
# Build the project
npm run build

# Run development mode
npm run dev

# Run linting
npm run lint

# Run tests
npm run test

# Type checking
npm run type-check

# Clean everything
npm run clean
```

## Research Pipeline Commands
```bash
# Complete research pipeline (4-8 hours)
./lethe-research/scripts/run_full_evaluation.sh

# Individual components
./lethe-research/scripts/create_dataset.sh      # LetheBench construction
./lethe-research/scripts/run_grid_search.sh     # Parameter optimization  
./lethe-research/scripts/evaluate_baselines.sh  # Comparative evaluation
./lethe-research/scripts/generate_paper.sh      # LaTeX compilation

# Validation and setup
python3 lethe-research/scripts/validate_setup.py
```

## Paper Generation Commands
```bash
# Compile LaTeX paper
cd lethe-research/paper
pdflatex lethe_neurips2025.tex
bibtex lethe_neurips2025
pdflatex lethe_neurips2025.tex
pdflatex lethe_neurips2025.tex
```

## Research Analysis Commands
```bash
# Run statistical analysis
python3 lethe-research/experiments/score.py

# Generate plots and figures
python3 lethe-research/experiments/plots.py

# Fraud-proof validation
python3 lethe-research/experiments/fraud_proof.py
```

## System Commands
```bash
# Check system health
npx ctx-run diagnose

# List available models
npx ctx-run models

# Version information
cat lethe_version.json
```