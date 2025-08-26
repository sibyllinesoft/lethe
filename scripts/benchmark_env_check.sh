#!/bin/bash
# Lethe Benchmark Environment Check
# =================================
# Validates hardware and software environment for reproducible benchmarks
# All reported numbers must be from the verified target environment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 Lethe Benchmark Environment Check${NC}"
echo "===================================="

# Exit codes
EXIT_SUCCESS=0
EXIT_WARNING=1
EXIT_ERROR=2

WARNINGS=0
ERRORS=0

check_passed() {
    echo -e "  ${GREEN}✅ $1${NC}"
}

check_warning() {
    echo -e "  ${YELLOW}⚠️  $1${NC}"
    ((WARNINGS++))
}

check_failed() {
    echo -e "  ${RED}❌ $1${NC}"
    ((ERRORS++))
}

# Hardware Profile Detection
echo ""
echo -e "${BLUE}🖥️  Hardware Profile${NC}"
echo "===================="

# CPU Information
if command -v lscpu &> /dev/null; then
    CPU_MODEL=$(lscpu | grep "Model name" | sed 's/Model name:[[:space:]]*//')
    CPU_CORES=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
    echo "CPU: $CPU_MODEL"
    echo "Cores: $CPU_CORES"
    
    if [ "$CPU_CORES" -ge 4 ]; then
        check_passed "CPU cores ($CPU_CORES) meet minimum requirement"
    else
        check_warning "CPU cores ($CPU_CORES) below recommended minimum (4)"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    CPU_MODEL=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
    CPU_CORES=$(sysctl -n hw.physicalcpu 2>/dev/null || echo "Unknown")
    echo "CPU: $CPU_MODEL"
    echo "Cores: $CPU_CORES"
    
    if [[ "$CPU_CORES" =~ ^[0-9]+$ ]] && [ "$CPU_CORES" -ge 4 ]; then
        check_passed "CPU cores ($CPU_CORES) meet minimum requirement"
    else
        check_warning "CPU cores ($CPU_CORES) below recommended minimum (4)"
    fi
else
    check_warning "Could not detect CPU information"
fi

# Memory Information
echo ""
if command -v free &> /dev/null; then
    MEMORY_GB=$(free -g | grep "Mem:" | awk '{print $2}')
    echo "Memory: ${MEMORY_GB}GB total"
    
    if [ "$MEMORY_GB" -ge 16 ]; then
        check_passed "Memory (${MEMORY_GB}GB) meets recommended requirement"
    elif [ "$MEMORY_GB" -ge 8 ]; then
        check_warning "Memory (${MEMORY_GB}GB) meets minimum but below recommended (16GB)"
    else
        check_failed "Memory (${MEMORY_GB}GB) below minimum requirement (8GB)"
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    MEMORY_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
    MEMORY_GB=$((MEMORY_BYTES / 1024 / 1024 / 1024))
    echo "Memory: ${MEMORY_GB}GB total"
    
    if [ "$MEMORY_GB" -ge 16 ]; then
        check_passed "Memory (${MEMORY_GB}GB) meets recommended requirement"
    elif [ "$MEMORY_GB" -ge 8 ]; then
        check_warning "Memory (${MEMORY_GB}GB) meets minimum but below recommended (16GB)"
    else
        check_failed "Memory (${MEMORY_GB}GB) below minimum requirement (8GB)"
    fi
else
    check_warning "Could not detect memory information"
fi

# Operating System
echo ""
echo -e "${BLUE}🐧 Operating System${NC}"
echo "==================="

OS_NAME=$(uname -s)
OS_VERSION=$(uname -r)
echo "OS: $OS_NAME $OS_VERSION"

if [[ "$OS_NAME" == "Linux" ]]; then
    check_passed "Linux environment detected"
elif [[ "$OS_NAME" == "Darwin" ]]; then
    check_passed "macOS environment detected"
else
    check_warning "Untested operating system: $OS_NAME"
fi

# Software Dependencies
echo ""
echo -e "${BLUE}📦 Software Dependencies${NC}"
echo "========================"

# Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo "Node.js: $NODE_VERSION"
    
    NODE_MAJOR=$(echo $NODE_VERSION | sed 's/v\([0-9]*\).*/\1/')
    if [ "$NODE_MAJOR" -ge 18 ]; then
        check_passed "Node.js version meets requirement (18+)"
    else
        check_failed "Node.js version ($NODE_VERSION) below requirement (18+)"
    fi
else
    check_failed "Node.js not found"
fi

# Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "Python: $PYTHON_VERSION"
    
    if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
        check_passed "Python version meets requirement (3.8+)"
    else
        check_failed "Python version ($PYTHON_VERSION) below requirement (3.8+)"
    fi
else
    check_failed "Python3 not found"
fi

echo ""
echo -e "${BLUE}📊 Environment Check Summary${NC}"
echo "============================"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✅ Perfect! Environment meets all requirements${NC}"
    echo "🚀 Ready for production benchmarks"
    exit $EXIT_SUCCESS
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Environment acceptable with $WARNINGS warning(s)${NC}"
    echo "🧪 Suitable for development and testing"
    exit $EXIT_WARNING  
else
    echo -e "${RED}❌ Environment has $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    echo "🔧 Please address errors before running benchmarks"
    exit $EXIT_ERROR
fi