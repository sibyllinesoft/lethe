# Task Completion Protocol

## When a Research Task is Complete

### 1. Code Quality Checks
```bash
npm run lint        # Code style validation
npm run type-check  # TypeScript compliance
npm run test       # Unit test validation
```

### 2. Research Validation
```bash
python3 lethe-research/scripts/validate_setup.py  # 52 checks must pass
python3 lethe-research/experiments/fraud_proof.py # Data integrity validation
```

### 3. Statistical Validation
- Bootstrap confidence intervals computed
- Effect sizes calculated and reported
- Multiple comparison correction applied
- Significance tests completed with proper α levels

### 4. Paper Requirements
- All results tables auto-generated from data
- Figures have proper mathematical notation
- Statistical claims backed by experimental evidence
- Reproducibility checklist completed
- LaTeX compiles without errors

### 5. Reproducibility Standards
- Fixed random seeds documented
- Environment snapshot captured in artifacts
- All configurations version-controlled
- Complete experimental pipeline executable

### 6. Publication Readiness
- NeurIPS 2025 format compliance
- 9-page limit respected (excluding references/appendix)
- Mathematical notation properly formatted
- Citations complete and properly formatted
- Appendix contains full technical details

### 7. Final Validation
```bash
# Complete pipeline execution test
./lethe-research/scripts/run_full_evaluation.sh

# Paper compilation test
cd lethe-research/paper && pdflatex lethe_neurips2025.tex
```

## Success Criteria
- ✅ All validation checks pass
- ✅ Paper compiles to PDF
- ✅ Results are statistically significant
- ✅ Complete reproducibility demonstrated
- ✅ Open-source implementation available