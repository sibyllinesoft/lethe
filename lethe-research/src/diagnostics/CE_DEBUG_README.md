# Cross-Encoder Debugging & Attestation System

## Overview

Comprehensive diagnostic system for identifying and resolving cross-encoder issues that cause flat scoring (std=0.019, only 2 unique values). The system provides systematic debugging, automatic fallback, and actionable fix recommendations.

## Problem Diagnosis

The S2 cross-encoder layer is producing flat, undifferentiated scores:
- **Standard deviation**: 0.019 (should be >0.2)
- **Unique values**: Only 2 (should have diverse scores)
- **Result**: Destroys selection quality, coverage drops to 0%

## Components

### 1. CE Attestation System (`ce_attestation.py`)

Runtime validation that logs all critical cross-encoder configuration parameters:

**Validates:**
- Model & tokenizer IDs with checksums
- Tokenization parameters (max_seq_len, truncation, special tokens) 
- Model precision (fp16/bf16/fp32)
- Head configuration (binary_cls, regression, pairwise)
- Evaluation mode settings

**Aborts if any critical parameters are `None`.**

### 2. Synthetic Test Suite (`ce_synthetic_tests.py`)

Mandatory tests that must pass before any full evaluation:

**Test Pairs:**
```python
pairs = [
  ("the quick brown fox", "the quick brown fox"),   # identical
  ("abc def", "xyz uvw"),                           # disjoint  
  ("sum of squares", "sum of squares formula a^2"), # partial overlap
]
```

**Requirements:**
- `rank(logits)[0] == 0` (identical > partial > disjoint)
- `np.std(logits) > 0.2` (fail if flat)

### 3. Input Format Debugging (`ce_input_debugging.py`)

Validates tokenization and input formatting:

**Checks:**
- **BERT/DeBERTa-style**: `[CLS] query [SEP] passage [SEP]` with `token_type_ids`
- **RoBERTa/DeBERTa-v3**: `<s> query </s></s> passage </s>` without `token_type_ids`
- Truncation behavior (longest_first vs only_second)
- Attention mask validation

**Logs actual inputs for first 5 pairs for manual inspection.**

### 4. Head Architecture Validation (`ce_head_validation.py`)

Validates classification head wiring and precision:

**Validates:**
- Evaluation mode (`model.eval()` with dropout frozen)
- Correct head type (binary logits `[not_rel, rel]` vs regression)
- Checkpoint alignment (head layer keys and shapes)
- Precision testing (fp16 vs fp32 underflow/saturation)
- Output processing (softmax vs direct regression)

### 5. Safe Mode Implementation (`ce_safe_mode.py`)

Fallback scoring while fixing cross-encoder issues:

**Configuration:**
```python
score = 0.6 * bi_encoder_dot + 0.4 * BM25F
# with γ=0.8, δ=0 (no DPP)
```

**Parameters:**
- K1 = 4000-6000 (increased candidate pool)
- K2 = 1000-1500 (more reranking budget)
- dims = 768 (use full dimensionality)

## Usage

### Command Line Interface

```bash
# Basic diagnosis
python scripts/debug_cross_encoder.py --model cross-encoder/ms-marco-MiniLM-L-6-v2

# With custom configuration
python scripts/debug_cross_encoder.py --model MODEL_NAME --config config/ce_debug_config.json

# GPU with detailed logging
python scripts/debug_cross_encoder.py --model MODEL_NAME --device cuda --log-level DEBUG

# Save results to file
python scripts/debug_cross_encoder.py --model MODEL_NAME --output results.json

# Activate safe mode immediately
python scripts/debug_cross_encoder.py --model MODEL_NAME --safe-mode-only
```

### Programmatic Usage

```python
from diagnostics import (
    CrossEncoderAttestationSystem,
    CrossEncoderSyntheticTester,
    CrossEncoderInputDebugger,
    CrossEncoderHeadValidator,
    CrossEncoderSafeMode
)

# Initialize components
attestation = CrossEncoderAttestationSystem()
tester = CrossEncoderSyntheticTester()
debugger = CrossEncoderInputDebugger()
validator = CrossEncoderHeadValidator()
safe_mode = CrossEncoderSafeMode()

# Run attestation
result = attestation.attest_cross_encoder(model, tokenizer, model_name, device)
if result.abort_required:
    safe_mode.activate_safe_mode("Attestation failed")

# Run synthetic tests
test_result = tester.run_synthetic_tests(cross_encoder, tokenizer)
if test_result.flat_scoring_detected:
    safe_mode.activate_safe_mode("Flat scoring detected")

# Continue with other diagnostics...
```

## Pass/Fail Gates

### Before Full Run
- Synthetic pairs pass with `std(logits) > 0.2`
- Attestation validates all critical parameters
- No `None` values in configuration

### After Fixes
- 50 real pairs: `std > 0.1`, `range > 0.3`
- Coverage canary: SpanCoverage ≥ 10%, SymbolCoverage ≥ 10% at 30% keep

## Integration with Existing Pipeline

The system integrates with the existing diagnostic infrastructure:
- Uses existing `ProbeResult` and diagnostic patterns
- Works with current cross-encoder models and tokenizers  
- Leverages InfiniteBench evaluation data
- Provides detailed logging and actionable recommendations

## Immediate Parameter Adjustments

When cross-encoder issues are detected, apply these parameter changes immediately:

```python
# Compensate for CE issues with larger pools
K1 = 5000  # Increase candidate pool
K2 = 1200  # More reranking budget  
dims = 768 # Use full dimensionality for code
diversity_delta = 0.0      # Disable DPP temporarily  
facility_gamma = 0.8       # Emphasize facility-location
```

## Expected Outcomes

After running the diagnostic system:

1. **Root Cause Identified**: Specific issue causing flat scores
2. **Safe Mode Active**: Fallback scoring maintains basic functionality
3. **Fix Recommendations**: Actionable steps to resolve issues
4. **Parameter Adjustments**: Immediate changes to improve coverage
5. **Coverage Recovery**: Target >10% SpanCoverage and SymbolCoverage

## Files Created

```
src/diagnostics/
├── ce_attestation.py           # Configuration validation
├── ce_synthetic_tests.py       # Synthetic test suite
├── ce_input_debugging.py       # Input format debugging
├── ce_head_validation.py       # Head architecture validation
├── ce_safe_mode.py             # Fallback scoring system
└── CE_DEBUG_README.md          # This documentation

scripts/
└── debug_cross_encoder.py      # CLI interface

config/
└── ce_debug_config.json        # Configuration template
```

The goal is to systematically debug why the cross-encoder produces flat scores and restore meaningful ranking that brings coverage above 0%.