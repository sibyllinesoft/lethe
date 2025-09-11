#!/usr/bin/env python3
"""
Hard validation sentinels for measurement pipeline
Fail-closed validation for ΔCBU, token counting, KV-reuse
"""

import numpy as np
import statistics
from typing import List, Dict, Any
from scipy.stats import pearsonr, spearmanr

def validate_delta_cbu_computation(results: List[Dict[str, Any]], epsilon: float = 1e-3) -> Dict[str, Any]:
    """
    Validate ΔCBU computation with fail-closed sentinels
    
    Requirements:
    1. Non-trivial variance across methods/budgets  
    2. Correlation with accuracy (Pearson/Spearman > 0.3)
    3. No constant defaults across scenarios
    """
    
    validation = {
        'status': 'UNKNOWN',
        'failures': [],
        'metrics': {}
    }
    
    # Group by (dataset, keep_ratio) to check variance across methods
    scenarios = {}
    for r in results:
        key = (r.get('dataset'), r.get('keep_ratio'))
        if key not in scenarios:
            scenarios[key] = []
        scenarios[key].append(r)
    
    variance_failures = []
    for scenario_key, scenario_results in scenarios.items():
        delta_values = [r.get('delta_cbu_per_1k', 0) for r in scenario_results]
        if len(delta_values) > 1:
            variance = np.var(delta_values)
            if variance < epsilon:
                variance_failures.append(f"Scenario {scenario_key}: variance={variance:.6f} < {epsilon}")
    
    if variance_failures:
        validation['failures'].extend(variance_failures)
        validation['status'] = 'FAILED'
    
    # Check correlation with accuracy
    delta_values = [r.get('delta_cbu_per_1k', 0) for r in results]
    accuracy_values = [r.get('p_at_k', {}).get('5', 0) for r in results]
    
    # Only compute correlation if we have non-zero variance
    if len(set(delta_values)) > 1 and len(set(accuracy_values)) > 1:
        try:
            pearson_r, _ = pearsonr(delta_values, accuracy_values)
            spearman_r, _ = spearmanr(delta_values, accuracy_values)
            
            validation['metrics']['pearson_correlation'] = pearson_r
            validation['metrics']['spearman_correlation'] = spearman_r
            
            if pearson_r < 0.3 and spearman_r < 0.3:
                validation['failures'].append(f"Low correlation with accuracy: Pearson={pearson_r:.3f}, Spearman={spearman_r:.3f} < 0.3")
        except Exception as e:
            validation['failures'].append(f"Correlation computation failed: {e}")
    else:
        validation['failures'].append("Insufficient variance in ΔCBU or accuracy for correlation analysis")
    
    # Check for eval_ok requirements (simulated - would need actual field)
    constant_scenarios = sum(1 for key, scenario_results in scenarios.items() 
                           if np.var([r.get('delta_cbu_per_1k', 0) for r in scenario_results]) < epsilon)
    
    validation['metrics']['constant_scenarios'] = constant_scenarios
    validation['metrics']['total_scenarios'] = len(scenarios)
    
    if constant_scenarios > 0:
        validation['failures'].append(f"{constant_scenarios}/{len(scenarios)} scenarios have constant ΔCBU")
    
    if validation['status'] == 'UNKNOWN' and not validation['failures']:
        validation['status'] = 'PASSED'
    elif validation['status'] == 'UNKNOWN':
        validation['status'] = 'FAILED'
    
    return validation

def validate_token_accounting(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate token counting with fail-closed sentinels
    
    Requirements:
    1. Monotonicity: 30% > 15% > 8% keep ratios
    2. Sanity: zh_qa median(tokens@8%) > 500  
    3. Compression ratios in expected ranges
    """
    
    validation = {
        'status': 'UNKNOWN',
        'failures': [],
        'metrics': {}
    }
    
    # Group by dataset and keep_ratio
    by_dataset_ratio = {}
    for r in results:
        dataset = r.get('dataset')
        keep_ratio = r.get('keep_ratio')
        tokens_kept = r.get('tokens_kept', 0)
        
        key = dataset
        if key not in by_dataset_ratio:
            by_dataset_ratio[key] = {}
        if keep_ratio not in by_dataset_ratio[key]:
            by_dataset_ratio[key][keep_ratio] = []
        by_dataset_ratio[key][keep_ratio].append(tokens_kept)
    
    # Check monotonicity for each dataset
    for dataset, ratios in by_dataset_ratio.items():
        if 0.08 in ratios and 0.15 in ratios and 0.30 in ratios:
            median_08 = statistics.median(ratios[0.08])
            median_15 = statistics.median(ratios[0.15])
            median_30 = statistics.median(ratios[0.30])
            
            validation['metrics'][f'{dataset}_monotonicity'] = {
                '8%': median_08,
                '15%': median_15, 
                '30%': median_30
            }
            
            if not (median_08 < median_15 < median_30):
                validation['failures'].append(
                    f"{dataset}: Token monotonicity violated: {median_08:.1f} < {median_15:.1f} < {median_30:.1f}"
                )
    
    # Check zh_qa sanity threshold
    if 'zh_qa' in by_dataset_ratio and 0.08 in by_dataset_ratio['zh_qa']:
        zh_qa_median_08 = statistics.median(by_dataset_ratio['zh_qa'][0.08])
        validation['metrics']['zh_qa_tokens_at_8pct'] = zh_qa_median_08
        
        if zh_qa_median_08 < 500:
            validation['failures'].append(
                f"zh_qa tokens@8% = {zh_qa_median_08:.1f} < 500 (likely window/sink confusion)"
            )
    
    # Check for tiny integer clusters (sentinel for window/sink confusion)
    all_tokens = [r.get('tokens_kept', 0) for r in results]
    tiny_values = [t for t in all_tokens if 0 < t < 50]
    if len(tiny_values) > len(all_tokens) * 0.3:  # More than 30% are suspiciously small
        validation['failures'].append(
            f"Token clustering at tiny values: {len(tiny_values)}/{len(all_tokens)} results < 50 tokens"
        )
    
    # Check compression ratios (if available)
    compression_ratios = [r.get('compression_ratio', 0) for r in results if r.get('compression_ratio', 0) > 0]
    if compression_ratios:
        by_keep_ratio = {}
        for r in results:
            if r.get('compression_ratio', 0) > 0:
                kr = r.get('keep_ratio')
                if kr not in by_keep_ratio:
                    by_keep_ratio[kr] = []
                by_keep_ratio[kr].append(r.get('compression_ratio'))
        
        for kr, ratios in by_keep_ratio.items():
            median_ratio = statistics.median(ratios)
            expected_min = kr * 0.7  # Allow 30% tolerance
            expected_max = kr * 1.3
            
            validation['metrics'][f'compression_ratio_{kr:.0%}'] = median_ratio
            
            if not (expected_min <= median_ratio <= expected_max):
                validation['failures'].append(
                    f"Compression ratio@{kr:.0%} = {median_ratio:.3f} outside expected range [{expected_min:.3f}, {expected_max:.3f}]"
                )
    
    if validation['status'] == 'UNKNOWN' and not validation['failures']:
        validation['status'] = 'PASSED'
    elif validation['status'] == 'UNKNOWN':
        validation['status'] = 'FAILED'
    
    return validation

def validate_kv_reuse_computation(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate KV-reuse/prefix stability with fail-closed sentinels
    
    Requirements:
    1. Non-zero mass: >80% of results have prefix_jaccard > 0.1
    2. Expected medians by dataset (Code.Debug ≥0.7, Code.QA ~0.6, Zh.QA ~0.5)
    3. No universal zeros (arranger not wired)
    """
    
    validation = {
        'status': 'UNKNOWN', 
        'failures': [],
        'metrics': {}
    }
    
    kv_values = [r.get('kv_reuse', 0) for r in results]
    
    # Check for universal zeros
    non_zero_count = sum(1 for kv in kv_values if kv > 0.1)
    total_count = len(kv_values)
    non_zero_rate = non_zero_count / total_count if total_count > 0 else 0
    
    validation['metrics']['non_zero_rate'] = non_zero_rate
    validation['metrics']['total_samples'] = total_count
    
    if non_zero_rate < 0.8:
        validation['failures'].append(
            f"Low non-zero KV reuse rate: {non_zero_rate:.1%} < 80% (arranger not wired?)"
        )
    
    # Check by dataset
    by_dataset = {}
    for r in results:
        dataset = r.get('dataset')
        kv_reuse = r.get('kv_reuse', 0)
        if dataset not in by_dataset:
            by_dataset[dataset] = []
        by_dataset[dataset].append(kv_reuse)
    
    expected_medians = {
        'code_debug': 0.7,
        'code_qa': 0.6, 
        'zh_qa': 0.5
    }
    
    for dataset, kv_values_dataset in by_dataset.items():
        if len(kv_values_dataset) > 0:
            median_kv = statistics.median(kv_values_dataset)
            validation['metrics'][f'{dataset}_median_kv'] = median_kv
            
            if dataset in expected_medians:
                expected = expected_medians[dataset]
                if median_kv < expected * 0.7:  # Allow 30% tolerance below expected
                    validation['failures'].append(
                        f"{dataset}: median KV reuse = {median_kv:.3f} < {expected * 0.7:.3f} (70% of expected {expected:.1f})"
                    )
    
    # Check for scenario-level collapses (all zeros in a scenario)
    scenarios = {}
    for r in results:
        key = (r.get('dataset'), r.get('keep_ratio'))
        if key not in scenarios:
            scenarios[key] = []
        scenarios[key].append(r.get('kv_reuse', 0))
    
    zero_scenarios = []
    for scenario_key, kv_values_scenario in scenarios.items():
        if all(kv == 0.0 for kv in kv_values_scenario):
            zero_scenarios.append(scenario_key)
    
    if zero_scenarios:
        validation['failures'].append(
            f"Scenarios with universal KV=0: {zero_scenarios}"
        )
    
    if validation['status'] == 'UNKNOWN' and not validation['failures']:
        validation['status'] = 'PASSED'
    elif validation['status'] == 'UNKNOWN':
        validation['status'] = 'FAILED'
    
    return validation

def run_all_validation_sentinels(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run all validation sentinels with fail-closed logic"""
    
    print("🔍 RUNNING MEASUREMENT PIPE VALIDATION SENTINELS")
    print("=" * 60)
    
    # Run all validations
    delta_cbu_validation = validate_delta_cbu_computation(results)
    token_validation = validate_token_accounting(results)
    kv_validation = validate_kv_reuse_computation(results)
    
    # Report results
    validations = {
        'delta_cbu': delta_cbu_validation,
        'token_accounting': token_validation,
        'kv_reuse': kv_validation
    }
    
    print(f"\n📊 ΔCBU COMPUTATION: {delta_cbu_validation['status']}")
    if delta_cbu_validation['failures']:
        for failure in delta_cbu_validation['failures']:
            print(f"  ❌ {failure}")
    else:
        print("  ✅ All ΔCBU checks passed")
        if 'pearson_correlation' in delta_cbu_validation['metrics']:
            print(f"  📈 Correlation with accuracy: r={delta_cbu_validation['metrics']['pearson_correlation']:.3f}")
    
    print(f"\n📊 TOKEN ACCOUNTING: {token_validation['status']}")
    if token_validation['failures']:
        for failure in token_validation['failures']:
            print(f"  ❌ {failure}")
    else:
        print("  ✅ All token accounting checks passed")
        if 'zh_qa_tokens_at_8pct' in token_validation['metrics']:
            print(f"  📏 zh_qa tokens@8%: {token_validation['metrics']['zh_qa_tokens_at_8pct']:.1f}")
    
    print(f"\n📊 KV REUSE COMPUTATION: {kv_validation['status']}")
    if kv_validation['failures']:
        for failure in kv_validation['failures']:
            print(f"  ❌ {failure}")
    else:
        print("  ✅ All KV reuse checks passed")
        if 'non_zero_rate' in kv_validation['metrics']:
            print(f"  🔄 Non-zero rate: {kv_validation['metrics']['non_zero_rate']:.1%}")
    
    # Overall status
    all_passed = all(v['status'] == 'PASSED' for v in validations.values())
    
    print(f"\n🎯 OVERALL VALIDATION: {'PASSED' if all_passed else 'FAILED'}")
    
    if all_passed:
        print("✅ All measurement pipes validated - ready for production data")
    else:
        print("❌ Pipeline validation failed - DO NOT PUBLISH these results")
        failed_pipes = [name for name, val in validations.items() if val['status'] == 'FAILED']
        print(f"   Failed pipes: {', '.join(failed_pipes)}")
    
    return {
        'overall_status': 'PASSED' if all_passed else 'FAILED',
        'validations': validations,
        'summary': {
            'total_samples': len(results),
            'failed_pipes': [name for name, val in validations.items() if val['status'] == 'FAILED'],
            'ready_for_production': all_passed
        }
    }

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) != 2:
        print("Usage: python validate_measurement_pipes.py <results.json>")
        sys.exit(1)
    
    with open(sys.argv[1]) as f:
        data = json.load(f)
    
    # Flatten results
    flat_results = []
    for method in data['results']:
        for result in data['results'][method]:
            flat_results.append(result)
    
    validation_results = run_all_validation_sentinels(flat_results)
    
    # Exit with error code if validation failed
    sys.exit(0 if validation_results['overall_status'] == 'PASSED' else 1)