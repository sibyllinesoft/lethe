#!/bin/bash
#
# Cross-Encoder Diagnosis Runner
# =============================
#
# Convenience script to run cross-encoder diagnosis with the most common settings.
# Automatically handles different scenarios and provides clear output.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODEL="cross-encoder/ms-marco-MiniLM-L-6-v2"
DEVICE="cpu"
OUTPUT_FILE=""
LOG_LEVEL="INFO"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Cross-Encoder Diagnosis Runner"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -m, --model MODEL      HuggingFace model to diagnose (default: $MODEL)"
            echo "  -d, --device DEVICE    Device to use: cpu or cuda (default: $DEVICE)"
            echo "  -o, --output FILE      Save results to file"
            echo "  -l, --log-level LEVEL  Logging level: DEBUG, INFO, WARNING, ERROR (default: $LOG_LEVEL)"
            echo "  -h, --help             Show this help"
            echo ""
            echo "Examples:"
            echo "  $0  # Diagnose default model"
            echo "  $0 -m cross-encoder/ms-marco-TinyBERT-L-2-v2"
            echo "  $0 -m MODEL_NAME -d cuda -o results.json"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if we're in the right directory
if [ ! -f "scripts/debug_cross_encoder_standalone.py" ]; then
    echo -e "${RED}Error: Must run from lethe-research directory${NC}"
    echo "cd /path/to/lethe-research && ./scripts/run_ce_diagnosis.sh"
    exit 1
fi

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

echo -e "${BLUE}🔍 Cross-Encoder Diagnosis Starting${NC}"
echo -e "${BLUE}Model: ${MODEL}${NC}"
echo -e "${BLUE}Device: ${DEVICE}${NC}"
echo -e "${BLUE}Log Level: ${LOG_LEVEL}${NC}"
echo ""

# Build command
CMD="python3 scripts/debug_cross_encoder_standalone.py --model \"$MODEL\" --device $DEVICE --log-level $LOG_LEVEL"

if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD --output \"$OUTPUT_FILE\""
    echo -e "${BLUE}Output: ${OUTPUT_FILE}${NC}"
fi

echo ""

# Run diagnosis
echo -e "${GREEN}Running diagnosis...${NC}"
echo ""

# Execute and capture exit code
if eval $CMD; then
    EXIT_CODE=$?
    case $EXIT_CODE in
        0)
            echo ""
            echo -e "${GREEN}✅ DIAGNOSIS COMPLETE: Cross-encoder appears healthy${NC}"
            ;;
        2)
            echo ""
            echo -e "${YELLOW}⚠️  DIAGNOSIS COMPLETE: Cross-encoder has issues but may be functional${NC}"
            echo -e "${YELLOW}Check the output above for recommended fixes${NC}"
            ;;
        *)
            echo ""
            echo -e "${RED}❌ DIAGNOSIS COMPLETE: Unknown status (exit code: $EXIT_CODE)${NC}"
            ;;
    esac
else
    EXIT_CODE=$?
    case $EXIT_CODE in
        1)
            echo ""
            echo -e "${RED}❌ DIAGNOSIS COMPLETE: Cross-encoder has critical issues${NC}"
            echo -e "${RED}IMMEDIATE ACTION REQUIRED:${NC}"
            echo -e "${RED}1. Activate safe mode parameters (see output above)${NC}"
            echo -e "${RED}2. Use fallback scoring: 60% bi-encoder + 40% BM25F${NC}"
            echo -e "${RED}3. Increase K1=5000, K2=1200, disable DPP (δ=0)${NC}"
            ;;
        130)
            echo ""
            echo -e "${YELLOW}⚠️  Diagnosis interrupted by user${NC}"
            ;;
        *)
            echo ""
            echo -e "${RED}❌ Diagnosis failed with exit code: $EXIT_CODE${NC}"
            ;;
    esac
fi

if [ -n "$OUTPUT_FILE" ]; then
    echo ""
    echo -e "${BLUE}📄 Results saved to: ${OUTPUT_FILE}${NC}"
fi

echo ""
echo -e "${BLUE}🎯 Quick Fix Summary:${NC}"
echo -e "${BLUE}If flat scoring detected, apply immediately:${NC}"
echo -e "${BLUE}  K1 = 5000  # Larger candidate pool${NC}"
echo -e "${BLUE}  K2 = 1200  # More reranking budget${NC}"
echo -e "${BLUE}  diversity_delta = 0.0  # Disable DPP${NC}"
echo -e "${BLUE}  facility_gamma = 0.8   # Emphasize coverage${NC}"
echo ""

exit $EXIT_CODE