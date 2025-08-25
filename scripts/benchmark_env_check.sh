#!/bin/bash

# benchmark_env_check.sh - Environment validation for Lethe reproducible benchmarks
# This script verifies that the hardware, software, and configuration meet requirements

set -euo pipefail

# ANSI color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters for validation results
PASSED=0
FAILED=0
WARNINGS=0

# Print header
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}                    Lethe Benchmark Environment Validation${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Helper functions
pass_check() {
    echo -e "  ✅ ${GREEN}PASS${NC}: $1"
    ((PASSED++))
}

fail_check() {
    echo -e "  ❌ ${RED}FAIL${NC}: $1"
    ((FAILED++))
}

warn_check() {
    echo -e "  ⚠️  ${YELLOW}WARN${NC}: $1"
    ((WARNINGS++))
}

info_check() {
    echo -e "  ℹ️  ${BLUE}INFO${NC}: $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get version numbers
get_node_version() {
    if command_exists node; then
        node --version | sed 's/v//'
    else
        echo "NOT_FOUND"
    fi
}

get_sqlite_version() {
    if command_exists sqlite3; then
        sqlite3 --version | awk '{print $1}'
    else
        echo "NOT_FOUND"
    fi
}

get_python_version() {
    if command_exists python3; then
        python3 --version | awk '{print $2}'
    else
        echo "NOT_FOUND"
    fi
}

# Version comparison helper
version_gte() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
}

# Hardware detection
detect_hardware() {
    local os_name=$(uname -s)
    local arch=$(uname -m)
    
    case "$os_name" in
        "Darwin")
            if [[ "$arch" == "arm64" ]]; then
                local model=$(sysctl -n hw.model 2>/dev/null || echo "Unknown")
                local cpu_brand=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
                local mem_gb=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1024 / 1024 / 1024 ))
                
                echo "macOS_ARM64"
                info_check "Detected: macOS on $arch"
                info_check "Model: $model"
                info_check "CPU: $cpu_brand"
                info_check "Memory: ${mem_gb} GB"
                
                # Check if it's an M2 MacBook Air
                if [[ "$cpu_brand" == *"Apple M2"* ]]; then
                    if [[ "$mem_gb" -ge 16 ]]; then
                        pass_check "Hardware matches target: MacBook Air M2 with ${mem_gb}GB RAM"
                        return 0
                    else
                        fail_check "Memory insufficient: ${mem_gb}GB < 16GB required"
                        return 1
                    fi
                else
                    warn_check "CPU not M2: $cpu_brand (results will be non-standard)"
                    return 2
                fi
            else
                warn_check "macOS on Intel detected (non-standard configuration)"
                return 2
            fi
            ;;
        "Linux")
            info_check "Detected: Linux on $arch"
            warn_check "Linux not primary target (results will be non-standard)"
            return 2
            ;;
        "MINGW"*|"CYGWIN"*|"MSYS"*)
            info_check "Detected: Windows on $arch"
            warn_check "Windows not primary target (results will be non-standard)"
            return 2
            ;;
        *)
            fail_check "Unsupported OS: $os_name"
            return 1
            ;;
    esac
}

# Main validation checks
echo "🔍 Hardware Profile Validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
detect_hardware
echo ""

echo "🔧 Software Version Validation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Node.js version check
NODE_VERSION=$(get_node_version)
if [[ "$NODE_VERSION" != "NOT_FOUND" ]]; then
    if version_gte "$NODE_VERSION" "20.11.0"; then
        if [[ "$NODE_VERSION" == 20.11.* ]]; then
            pass_check "Node.js version: v$NODE_VERSION (exact target match)"
        else
            warn_check "Node.js version: v$NODE_VERSION (compatible but not exact target v20.11.0)"
        fi
    else
        fail_check "Node.js version: v$NODE_VERSION (< v20.11.0 required)"
    fi
else
    fail_check "Node.js not found"
fi

# SQLite version check
SQLITE_VERSION=$(get_sqlite_version)
if [[ "$SQLITE_VERSION" != "NOT_FOUND" ]]; then
    if version_gte "$SQLITE_VERSION" "3.42.0"; then
        pass_check "SQLite version: $SQLITE_VERSION"
    else
        fail_check "SQLite version: $SQLITE_VERSION (< 3.42.0 required)"
    fi
else
    fail_check "SQLite not found"
fi

# Python version check
PYTHON_VERSION=$(get_python_version)
if [[ "$PYTHON_VERSION" != "NOT_FOUND" ]]; then
    if version_gte "$PYTHON_VERSION" "3.11.0"; then
        pass_check "Python version: $PYTHON_VERSION"
    else
        warn_check "Python version: $PYTHON_VERSION (recommend >= 3.11.0)"
    fi
else
    warn_check "Python3 not found (required for data processing)"
fi

# Git version check
if command_exists git; then
    GIT_VERSION=$(git --version | awk '{print $3}')
    if version_gte "$GIT_VERSION" "2.40.0"; then
        pass_check "Git version: $GIT_VERSION"
    else
        warn_check "Git version: $GIT_VERSION (recommend >= 2.40.0)"
    fi
else
    fail_check "Git not found"
fi

echo ""
echo "🌱 Environment Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check environment variables
check_env_var() {
    local var_name="$1"
    local expected_value="$2"
    local description="$3"
    
    if [[ -n "${!var_name:-}" ]]; then
        if [[ "${!var_name}" == "$expected_value" ]]; then
            pass_check "$description: ${!var_name}"
        else
            warn_check "$description: ${!var_name} (expected: $expected_value)"
        fi
    else
        warn_check "$description: not set (expected: $expected_value)"
    fi
}

check_env_var "LETHE_RANDOM_SEED" "42" "Global random seed"
check_env_var "LETHE_HNSW_SEED" "123" "HNSW index seed"
check_env_var "LETHE_DATASET_SEED" "456" "Dataset generation seed"
check_env_var "LETHE_EVAL_SEED" "789" "Evaluation seed"
check_env_var "NODE_ENV" "benchmark" "Node environment"

# Check UV_THREADPOOL_SIZE
if [[ -n "${UV_THREADPOOL_SIZE:-}" ]]; then
    if [[ "${UV_THREADPOOL_SIZE}" == "8" ]]; then
        pass_check "UV threadpool size: ${UV_THREADPOOL_SIZE}"
    else
        info_check "UV threadpool size: ${UV_THREADPOOL_SIZE} (recommend: 8)"
    fi
else
    warn_check "UV_THREADPOOL_SIZE not set (recommend: 8)"
fi

echo ""
echo "📦 Dependencies & Lock Files"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check for package-lock.json
if [[ -f "package-lock.json" ]]; then
    pass_check "package-lock.json found"
else
    fail_check "package-lock.json missing (run 'npm install' first)"
fi

# Check for scripts directory
if [[ -d "scripts" ]]; then
    pass_check "scripts directory found"
else
    fail_check "scripts directory missing"
fi

# Check available disk space
if command_exists df; then
    AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [[ "$AVAILABLE_SPACE" -gt 10 ]]; then
        pass_check "Available disk space: ${AVAILABLE_SPACE}GB"
    else
        warn_check "Available disk space: ${AVAILABLE_SPACE}GB (recommend >10GB)"
    fi
fi

echo ""
echo "📊 Results Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -e "  ✅ Passed: ${GREEN}$PASSED${NC}"
echo -e "  ⚠️  Warnings: ${YELLOW}$WARNINGS${NC}"
echo -e "  ❌ Failed: ${RED}$FAILED${NC}"
echo ""

# Final verdict
if [[ $FAILED -eq 0 ]]; then
    if [[ $WARNINGS -eq 0 ]]; then
        echo -e "🎉 ${GREEN}ENVIRONMENT VALIDATION PASSED${NC}"
        echo -e "   All requirements met. Ready for reproducible benchmarking."
        exit 0
    else
        echo -e "⚠️  ${YELLOW}ENVIRONMENT VALIDATION PASSED WITH WARNINGS${NC}"
        echo -e "   Core requirements met, but some recommendations not followed."
        echo -e "   Results may not exactly match reference benchmarks."
        exit 0
    fi
else
    echo -e "❌ ${RED}ENVIRONMENT VALIDATION FAILED${NC}"
    echo -e "   $FAILED critical requirements not met."
    echo -e "   Please address failures before running benchmarks."
    exit 1
fi