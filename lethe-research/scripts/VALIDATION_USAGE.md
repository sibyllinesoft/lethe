# Comprehensive Validation Sentinels Usage Guide

## Overview

The validation sentinels system provides **fail-closed** validation for three critical measurement pipes in your research evaluation system:

1. **ΔCBU Computation Pipeline** - Validates cost-benefit computation
2. **Token Accounting Pipeline** - Validates token counting and compression 
3. **KV-Reuse Measurement Pipeline** - Validates key-value reuse metrics

## Files Created

```
scripts/
├── validation_sentinels.py          # Core validation system
├── test_validation_sentinels.py     # Test with synthetic data
├── validate_actual_results.py       # Test with actual evaluation results
└── VALIDATION_USAGE.md             # This documentation
```

## Integration Points

### 1. Automatic Integration (Already Done)

Your main evaluation pipeline (`run_hybrid_infinitebench.py`) now automatically uses the comprehensive validation:

```python
# Before saving any results, validation runs
validate_measurement_pipeline(flat_results)
```

If any sentinel fails, the process **immediately exits** with detailed error information.

### 2. Manual Validation

You can validate any results manually:

```python
from validation_sentinels import validate_measurement_pipeline_v2, ValidationThresholds

# Your evaluation results
results = [
    {
        'method_name': 'streaming',
        'dataset': 'code_debug',
        'keep_ratio': 0.08,
        'p_at_k': {5: 0.65, 10: 0.72},
        'delta_cbu_per_1k': 0.012,
        'kv_reuse': 0.75,
        'tokens_kept': 800,
        # ... other fields
    },
    # ... more results
]

# Run validation
report = validate_measurement_pipeline_v2(
    results,
    fail_fast=True  # Exit immediately on failure
)

if report.success:
    print("✅ All measurement pipes validated!")
else:
    print("❌ Critical failures detected!")
```

### 3. Custom Thresholds

You can customize validation thresholds:

```python
from validation_sentinels import ValidationThresholds

custom_thresholds = ValidationThresholds(
    delta_cbu_variance_epsilon=1e-4,  # Stricter variance requirement
    zh_qa_min_tokens_at_8pct=600,     # Higher minimum for zh_qa
    prefix_jaccard_nonzero_min=0.9    # Require 90% non-zero KV reuse
)

report = validate_measurement_pipeline_v2(
    results,
    thresholds=custom_thresholds
)
```

## Validation Sentinels Explained

### 1. ΔCBU Validation Sentinel

**Purpose**: Ensures cost-benefit computation is working correctly

**Checks**:
- ✅ Variance across scenarios > ε (default: 1e-3)
- ✅ Correlation with P@5 performance > 0.3  
- ✅ No constant values across systems
- ✅ V2 payload present (not zero-filled)

**Failure Examples**:
```
❌ ΔCBU variance too low (0.000001) - likely constant values
❌ ΔCBU shows no correlation with P@5 - pipe disconnected  
❌ System 'hybrid' has constant ΔCBU values
```

### 2. Token Accounting Validation Sentinel

**Purpose**: Ensures token counting and compression ratios are correct

**Checks**:
- ✅ Monotonicity: tokens@30% > tokens@15% > tokens@8%
- ✅ zh_qa sanity: median(tokens@8%) > 500
- ✅ Compression ratios: 0.07 < ratio < 0.09  
- ✅ No clustering at tiny values {4,5,6,...}

**Failure Examples**:
```
❌ Token accounting not monotonic - keep_ratio not working
❌ zh_qa median tokens@8% too low (50) - window/sink confusion?
❌ Tokens clustered at tiny values - accounting error
```

### 3. KV-Reuse Validation Sentinel

**Purpose**: Ensures key-value reuse measurement is functioning

**Checks**:
- ✅ Non-zero mass: 80%+ samples have jaccard > 0.1
- ✅ Dataset medians: Code.Debug ≥0.7, Code.QA ~0.6, Zh.QA ~0.5  
- ✅ No universal zeros (arranger wired)

**Failure Examples**:
```
❌ All KV reuse values are zero - arranger not wired
❌ Too few samples with meaningful KV reuse (30%)  
❌ code_debug KV reuse median too low - expected high reuse
```

## Command Line Usage

### Test the System

```bash
# Test with synthetic data (good and bad)
python3 scripts/test_validation_sentinels.py

# Test with your actual evaluation results  
python3 scripts/validate_actual_results.py
```

### Run Evaluation with Validation

```bash
# Your normal evaluation command
python3 scripts/run_hybrid_infinitebench.py

# Now automatically includes fail-closed validation
# Will exit immediately if any pipe is broken
```

## Output Examples

### ✅ Success Output

```
🔒 Starting fail-closed validation sentinels
🔍 ΔCBU Validation Sentinel
✅ ΔCBU Sentinel PASSED (std=0.004427, correlation validated)
🔍 Token Accounting Validation Sentinel  
✅ Token Accounting Sentinel PASSED (monotonicity: 750 < 1200 < 2400)
🔍 KV-Reuse Validation Sentinel
✅ KV-Reuse Sentinel PASSED (100.0% non-zero, medians validated)
✅ ALL VALIDATION SENTINELS PASSED - Pipeline verified
```

### ❌ Failure Output

```
🚨 VALIDATION FAILED - STOPPING EXECUTION
Pipeline has critical measurement failures that would invalidate results

================================================================================
CRITICAL VALIDATION FAILURES DETECTED
================================================================================
❌ ΔCBU_Sentinel.InsufficientVariance: ΔCBU variance too low (0.000000) - likely constant values across systems
❌ TokenAccounting_Sentinel.MonotonicityViolation: Token accounting not monotonic - keep_ratio not controlling token retention  
❌ KVReuse_Sentinel.UniversalZeros: All KV reuse values are zero - arranger not wired or completely broken

Failed Sentinels: ΔCBU_Sentinel, TokenAccounting_Sentinel, KVReuse_Sentinel
Fix these issues before generating any metrics or claims.
================================================================================
```

## Real-World Issues Detected

Your current evaluation (as shown in the logs) would trigger several failures:

1. **Universal P@5 = 0.0** → Indicates label join failure or evaluation logic broken
2. **All accuracy = 0.0** → Suggests retrieval/evaluation pipeline completely broken  
3. **Likely constant ΔCBU** → Cost computation not varying with conditions
4. **Possible KV reuse issues** → Arranger may not be properly wired

These are exactly the types of critical pipe failures the sentinels are designed to catch before you generate any public metrics or research claims.

## Benefits

### 🔒 Fail-Closed Security
- **No false positives**: If validation passes, your pipes are working
- **No silent failures**: Broken pipes immediately stop execution
- **Clear diagnostics**: Exact failure points and suggested fixes

### 🎯 Research Integrity  
- **Prevents invalid results**: Catches measurement issues before publication
- **Systematic validation**: Comprehensive check of all critical pipes
- **Reproducible standards**: Same validation across all evaluations

### 🔧 Development Efficiency
- **Fast debugging**: Immediate feedback on pipe failures  
- **Regression detection**: Catches breaks during interface changes
- **Quality assurance**: Ensures measurement improvements actually work

## Next Steps

1. **Fix Current Issues**: Address the universal zero accuracy/P@5 problems in your evaluation
2. **Verify Interface Fixes**: Run evaluation again - should now fail validation and show exactly what's broken
3. **Iterative Debugging**: Use validation failures to systematically fix measurement pipes  
4. **Production Integration**: Once pipes work, validation ensures they stay working

The validation system is now integrated and ready to catch measurement failures before they invalidate your research results.