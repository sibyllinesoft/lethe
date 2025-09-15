#!/bin/bash
# CI/CD Integration Example for HTML Validator Report Testing
# This script shows how to integrate the test suite into automated pipelines

set -euo pipefail

echo "🏗️  HTML Validator Report CI/CD Pipeline"
echo "========================================"

# Configuration
METRICS_CSV="metrics_summary.csv"
ADVANTAGE_MAP="advantage_map.json" 
MANIFEST="signed_manifest.json"
OUTPUT_HTML="validator_report.html"

# Step 1: Verify all required input files exist
echo "📋 Step 1: Checking input files..."
for file in "$METRICS_CSV" "$ADVANTAGE_MAP" "$MANIFEST"; do
    if [[ ! -f "$file" ]]; then
        echo "❌ FATAL: Required file missing: $file"
        exit 1
    fi
    echo "✅ Found: $file"
done

# Step 2: Run unit tests to ensure code quality
echo ""
echo "🧪 Step 2: Running unit tests..."
python3 -m pytest test_validator_report.py -v --tb=short
if [[ $? -eq 0 ]]; then
    echo "✅ All unit tests passed!"
else
    echo "❌ Unit tests failed!"
    exit 1
fi

# Step 3: Generate the HTML report
echo ""
echo "🎨 Step 3: Generating HTML validator report..."
python3 enhanced_html_generator.py "$METRICS_CSV" "$ADVANTAGE_MAP" "$MANIFEST" "$OUTPUT_HTML"
if [[ $? -eq 0 ]]; then
    echo "✅ HTML report generated successfully!"
else
    echo "❌ HTML report generation failed!"
    exit 1
fi

# Step 4: Run postprocess validation (hard tests)
echo ""
echo "🔍 Step 4: Running postprocess validation..."
python3 postprocess_validation.py "$METRICS_CSV" "$OUTPUT_HTML" "$MANIFEST"
if [[ $? -eq 0 ]]; then
    echo "✅ All hard tests passed!"
else
    echo "❌ Critical validation failures detected!"
    echo "🚨 PIPELINE FAILED - Report does not meet requirements"
    exit 1
fi

# Step 5: Additional quality checks (optional)
echo ""
echo "🔬 Step 5: Additional quality checks..."

# Check HTML file size (should be substantial)
HTML_SIZE=$(stat -f%z "$OUTPUT_HTML" 2>/dev/null || stat -c%s "$OUTPUT_HTML" 2>/dev/null || echo "0")
if [[ "$HTML_SIZE" -lt 10000 ]]; then
    echo "⚠️  WARNING: HTML file seems small ($HTML_SIZE bytes)"
fi

# Check for common HTML issues
if ! grep -q "<!DOCTYPE html>" "$OUTPUT_HTML"; then
    echo "❌ Missing DOCTYPE declaration"
    exit 1
fi

if ! grep -q "<title>" "$OUTPUT_HTML"; then
    echo "❌ Missing title tag"
    exit 1
fi

echo "✅ Additional quality checks passed!"

# Step 6: Success summary
echo ""
echo "🎯 PIPELINE SUCCESS!"
echo "==================="
echo "✅ All unit tests passed"
echo "✅ HTML report generated: $OUTPUT_HTML"  
echo "✅ All critical requirements validated"
echo "✅ Report ready for deployment/distribution"

# Optional: Print report statistics
echo ""
echo "📊 Report Statistics:"
echo "- HTML file size: $HTML_SIZE bytes"
echo "- Generated at: $(date)"
echo "- Report file: $(realpath "$OUTPUT_HTML")"

echo ""
echo "🚀 Ready for deployment!"