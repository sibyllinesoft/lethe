# HTML Validator Report Testing Suite

This directory contains a comprehensive test suite for the HTML validator report generator that enforces all critical requirements and prevents regressions.

## Files

### Core Testing Files
- **`test_validator_report.py`** - Comprehensive pytest test suite (~50 lines) with 9 test functions
- **`postprocess_validation.py`** - Standalone validation script for CI/CD integration
- **`enhanced_html_generator.py`** - Enhanced with integrated validation

## Hard Tests (Critical Requirements)

These tests MUST pass for any validator report:

1. **`RECALL_METRIC == "score"`** - Ensures the correct recall metric is used
2. **`LETHE_ENGINE_ADAPTER_ID` in CSV** - Verifies the Lethe engine adapter exists in data
3. **All 3 budget charts present (8%, 15%, 30%)** - Confirms complete budget coverage
4. **"Lethe Engine" label in HTML** - Validates proper branding
5. **Manifest SHA matches** - Ensures provenance integrity

## Usage

### Run Full Test Suite
```bash
python3 -m pytest test_validator_report.py -v
```

### Run Postprocess Validation (for CI/CD)
```bash
python3 postprocess_validation.py metrics.csv report.html manifest.json
```

### Generate Report with Validation
```bash
python3 enhanced_html_generator.py metrics.csv advantage_map.json manifest.json output.html
```

## Integration Points

### In enhanced_html_generator.py
The `generate_enhanced_validator_report()` function now returns:
```python
output_path, validation_results = generate_enhanced_validator_report(...)
```

Where `validation_results` contains:
- `validation_passed`: Overall pass/fail
- Individual check results for each requirement
- Detailed error information if any check fails

### CI/CD Integration
Use `postprocess_validation.py` as a build step:
```bash
# Generate report
python3 enhanced_html_generator.py data/metrics.csv data/advantage_map.json data/manifest.json report.html

# Validate (exits with code 1 on failure)
python3 postprocess_validation.py data/metrics.csv report.html data/manifest.json
```

## Test Coverage

- **Critical Requirements**: All 5 hard requirements enforced
- **Integration**: End-to-end report generation
- **Quality Gates**: Methodology, provenance, and section completeness
- **Error Handling**: File validation and HTML structure checks
- **Regression Prevention**: Fail-fast on any critical feature regression

## Benefits

1. **Drop-in Ready**: Tests work with existing files and structure
2. **Fast Feedback**: Immediate failure on requirement violations
3. **CI/CD Safe**: Exit codes and structured output for automation
4. **Comprehensive**: Covers all critical features we implemented
5. **Regression Prevention**: Prevents accidental removal of key features