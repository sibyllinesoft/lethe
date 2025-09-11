#!/bin/bash
# Lethe Determinism Service V2.1.0 - Deployment Validation
# Generated on: 2025-09-10
# Review this script before execution

set -euo pipefail  # Exit on any error

echo "🔍 Validating Lethe Determinism Service V2.1.0 deployment..."

# Configuration
SERVICE_URL="${SERVICE_URL:-http://localhost:8080}"
TIMEOUT_SECONDS=30
MAX_RETRIES=5
VALIDATION_RESULTS=()

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Wait for service to be ready
wait_for_service() {
    local url="$1"
    local max_attempts="$2"
    local attempt=1
    
    log_info "Waiting for service at $url (max $max_attempts attempts)..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s --connect-timeout 5 "$url/health" > /dev/null 2>&1; then
            log_info "Service is responding (attempt $attempt)"
            return 0
        else
            log_warn "Service not ready, attempt $attempt/$max_attempts"
            sleep 5
            ((attempt++))
        fi
    done
    
    log_error "Service failed to respond after $max_attempts attempts"
    return 1
}

# Test basic health endpoint
test_health_endpoint() {
    log_info "Testing health endpoint..."
    
    local response=$(curl -s -w "HTTPSTATUS:%{http_code}" "$SERVICE_URL/health" 2>/dev/null || echo "HTTPSTATUS:000")
    local http_code=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    local body=$(echo "$response" | sed -E 's/HTTPSTATUS:[0-9]*$//')
    
    if [ "$http_code" == "200" ]; then
        log_info "✅ Health endpoint: PASS"
        VALIDATION_RESULTS+=("Health endpoint: PASS")
        echo "Response: $body"
        return 0
    else
        log_error "❌ Health endpoint: FAIL (HTTP $http_code)"
        VALIDATION_RESULTS+=("Health endpoint: FAIL (HTTP $http_code)")
        return 1
    fi
}

# Test detailed health endpoint
test_detailed_health() {
    log_info "Testing detailed health endpoint..."
    
    local response=$(curl -s -w "HTTPSTATUS:%{http_code}" "$SERVICE_URL/health/detailed" 2>/dev/null || echo "HTTPSTATUS:000")
    local http_code=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    local body=$(echo "$response" | sed -E 's/HTTPSTATUS:[0-9]*$//')
    
    if [ "$http_code" == "200" ]; then
        log_info "✅ Detailed health endpoint: PASS"
        VALIDATION_RESULTS+=("Detailed health: PASS")
        
        # Parse and validate component statuses
        if echo "$body" | grep -q "database.*healthy"; then
            log_info "  ✅ Database: Healthy"
        else
            log_warn "  ⚠️  Database status unclear"
        fi
        
        if echo "$body" | grep -q "circuit_breaker.*closed"; then
            log_info "  ✅ Circuit Breaker: Closed (Normal)"
        else
            log_warn "  ⚠️  Circuit Breaker status unclear"
        fi
        
        return 0
    else
        log_error "❌ Detailed health endpoint: FAIL (HTTP $http_code)"
        VALIDATION_RESULTS+=("Detailed health: FAIL (HTTP $http_code)")
        return 1
    fi
}

# Test metrics endpoint
test_metrics_endpoint() {
    log_info "Testing Prometheus metrics endpoint..."
    
    local response=$(curl -s -w "HTTPSTATUS:%{http_code}" "$SERVICE_URL/metrics" 2>/dev/null || echo "HTTPSTATUS:000")
    local http_code=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    local body=$(echo "$response" | sed -E 's/HTTPSTATUS:[0-9]*$//')
    
    if [ "$http_code" == "200" ]; then
        log_info "✅ Metrics endpoint: PASS"
        VALIDATION_RESULTS+=("Metrics endpoint: PASS")
        
        # Validate key metrics are present
        local key_metrics=("determinism_rate" "ece_score" "p95_latency_ms" "gate_status")
        for metric in "${key_metrics[@]}"; do
            if echo "$body" | grep -q "$metric"; then
                log_info "  ✅ Metric '$metric' found"
            else
                log_warn "  ⚠️  Metric '$metric' missing"
            fi
        done
        
        return 0
    else
        log_error "❌ Metrics endpoint: FAIL (HTTP $http_code)"
        VALIDATION_RESULTS+=("Metrics endpoint: FAIL (HTTP $http_code)")
        return 1
    fi
}

# Test determinism replay endpoint
test_determinism_endpoint() {
    log_info "Testing determinism replay endpoint..."
    
    local test_payload='{"slice_id": "test_validation", "config": {"iterations": 1}}'
    local response=$(curl -s -w "HTTPSTATUS:%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$test_payload" \
        "$SERVICE_URL/determinism/replay/test_validation" 2>/dev/null || echo "HTTPSTATUS:000")
    
    local http_code=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    local body=$(echo "$response" | sed -E 's/HTTPSTATUS:[0-9]*$//')
    
    if [ "$http_code" == "200" ] || [ "$http_code" == "202" ]; then
        log_info "✅ Determinism replay endpoint: PASS"
        VALIDATION_RESULTS+=("Determinism replay: PASS")
        echo "Response preview: $(echo "$body" | head -c 200)..."
        return 0
    else
        log_error "❌ Determinism replay endpoint: FAIL (HTTP $http_code)"
        VALIDATION_RESULTS+=("Determinism replay: FAIL (HTTP $http_code)")
        return 1
    fi
}

# Test version endpoint
test_version_endpoint() {
    log_info "Testing version endpoint..."
    
    local response=$(curl -s -w "HTTPSTATUS:%{http_code}" "$SERVICE_URL/version" 2>/dev/null || echo "HTTPSTATUS:000")
    local http_code=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    local body=$(echo "$response" | sed -E 's/HTTPSTATUS:[0-9]*$//')
    
    if [ "$http_code" == "200" ]; then
        log_info "✅ Version endpoint: PASS"
        VALIDATION_RESULTS+=("Version endpoint: PASS")
        
        # Validate version contains V2.1.0
        if echo "$body" | grep -q "2\.1\.0"; then
            log_info "  ✅ Version 2.1.0 confirmed"
        else
            log_warn "  ⚠️  Version 2.1.0 not found in response"
        fi
        
        echo "Version info: $body"
        return 0
    else
        log_error "❌ Version endpoint: FAIL (HTTP $http_code)"
        VALIDATION_RESULTS+=("Version endpoint: FAIL (HTTP $http_code)")
        return 1
    fi
}

# Test WebSocket endpoint
test_websocket_endpoint() {
    log_info "Testing WebSocket endpoint availability..."
    
    # Check if websocat is available for WebSocket testing
    if command -v websocat &> /dev/null; then
        log_info "Testing WebSocket connection with websocat..."
        
        # Test WebSocket connection (timeout after 5 seconds)
        if timeout 5 websocat "ws://localhost:8080/ws" <<< '{"type":"ping"}' > /dev/null 2>&1; then
            log_info "✅ WebSocket endpoint: PASS"
            VALIDATION_RESULTS+=("WebSocket: PASS")
        else
            log_warn "⚠️  WebSocket endpoint: Could not establish connection"
            VALIDATION_RESULTS+=("WebSocket: WARN (connection failed)")
        fi
    else
        log_warn "⚠️  WebSocket test skipped (websocat not installed)"
        VALIDATION_RESULTS+=("WebSocket: SKIPPED (no test tool)")
    fi
}

# Test production gates status
test_production_gates() {
    log_info "Validating production gates status..."
    
    local response=$(curl -s "$SERVICE_URL/gates/status" 2>/dev/null || echo "{}")
    
    if echo "$response" | grep -q "gates"; then
        log_info "✅ Production gates endpoint accessible"
        
        # Count GREEN gates (this would need to parse actual JSON in real implementation)
        local green_gates=$(echo "$response" | grep -o '"status":"GREEN"' | wc -l || echo "0")
        
        if [ "$green_gates" -ge 10 ]; then
            log_info "  ✅ $green_gates gates are GREEN (target: ≥10)"
            VALIDATION_RESULTS+=("Production gates: PASS ($green_gates GREEN)")
        else
            log_warn "  ⚠️  Only $green_gates gates are GREEN (target: ≥10)"
            VALIDATION_RESULTS+=("Production gates: WARN ($green_gates GREEN)")
        fi
    else
        log_warn "⚠️  Production gates endpoint not responding correctly"
        VALIDATION_RESULTS+=("Production gates: WARN (endpoint issue)")
    fi
}

# Check monitoring stack connectivity
test_monitoring_connectivity() {
    log_info "Testing monitoring stack connectivity..."
    
    # Test Prometheus
    if curl -s "http://localhost:9090/api/v1/query?query=up" > /dev/null 2>&1; then
        log_info "✅ Prometheus: Connected"
        VALIDATION_RESULTS+=("Prometheus: PASS")
    else
        log_warn "⚠️  Prometheus: Not accessible at localhost:9090"
        VALIDATION_RESULTS+=("Prometheus: WARN")
    fi
    
    # Test Grafana
    if curl -s "http://localhost:3000/api/health" > /dev/null 2>&1; then
        log_info "✅ Grafana: Connected"
        VALIDATION_RESULTS+=("Grafana: PASS")
    else
        log_warn "⚠️  Grafana: Not accessible at localhost:3000"
        VALIDATION_RESULTS+=("Grafana: WARN")
    fi
}

# Generate validation report
generate_report() {
    local timestamp=$(date -u '+%Y-%m-%d %H:%M:%S UTC')
    local report_file="validation_report_$(date +%Y%m%d_%H%M%S).txt"
    
    echo "📋 Generating validation report..."
    
    cat > "$report_file" << EOF
LETHE DETERMINISM SERVICE V2.1.0 - DEPLOYMENT VALIDATION REPORT
Generated: $timestamp
Service URL: $SERVICE_URL

VALIDATION RESULTS:
==================
EOF
    
    local pass_count=0
    local total_count=0
    
    for result in "${VALIDATION_RESULTS[@]}"; do
        echo "$result" >> "$report_file"
        ((total_count++))
        
        if echo "$result" | grep -q "PASS"; then
            ((pass_count++))
        fi
    done
    
    local pass_rate=$((pass_count * 100 / total_count))
    
    cat >> "$report_file" << EOF

SUMMARY:
========
Total Tests: $total_count
Passed: $pass_count
Pass Rate: ${pass_rate}%

EOF
    
    if [ $pass_rate -ge 80 ]; then
        echo "OVERALL STATUS: ✅ DEPLOYMENT VALIDATED (${pass_rate}% pass rate)" >> "$report_file"
        log_info "✅ Overall validation: PASSED (${pass_rate}% pass rate)"
    else
        echo "OVERALL STATUS: ❌ DEPLOYMENT FAILED VALIDATION (${pass_rate}% pass rate)" >> "$report_file"
        log_error "❌ Overall validation: FAILED (${pass_rate}% pass rate)"
    fi
    
    echo "" >> "$report_file"
    echo "Report saved to: $report_file"
    log_info "📄 Detailed report saved to: $report_file"
    
    return $((pass_rate >= 80 ? 0 : 1))
}

# Main validation function
main() {
    log_info "🚀 Starting deployment validation for Lethe Determinism Service V2.1.0"
    log_info "Service URL: $SERVICE_URL"
    echo ""
    
    # Wait for service to be ready
    if ! wait_for_service "$SERVICE_URL" "$MAX_RETRIES"; then
        log_error "❌ Service is not responding. Validation aborted."
        exit 1
    fi
    
    echo ""
    log_info "🔍 Running validation tests..."
    echo ""
    
    # Run all validation tests
    test_health_endpoint
    sleep 1
    test_detailed_health
    sleep 1
    test_metrics_endpoint
    sleep 1
    test_version_endpoint
    sleep 1
    test_determinism_endpoint
    sleep 2
    test_websocket_endpoint
    sleep 1
    test_production_gates
    sleep 1
    test_monitoring_connectivity
    
    echo ""
    log_info "📊 Validation complete. Generating report..."
    
    # Generate and display report
    if generate_report; then
        echo ""
        log_info "🎉 Deployment validation PASSED! Service is ready for production."
        echo ""
        log_info "🎯 Next steps:"
        echo "1. Import Grafana dashboards from monitoring/grafana/"
        echo "2. Configure alerting rules in Prometheus"
        echo "3. Set up log aggregation"
        echo "4. Schedule regular health checks"
        echo ""
        exit 0
    else
        echo ""
        log_error "🚨 Deployment validation FAILED! Review issues before production deployment."
        echo ""
        log_error "🔧 Troubleshooting:"
        echo "1. Check service logs: docker logs <container-name>"
        echo "2. Verify configuration files"
        echo "3. Ensure all dependencies are running"
        echo "4. Review network connectivity"
        echo ""
        exit 1
    fi
}

# Execute main function
main "$@"