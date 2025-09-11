#!/usr/bin/env python3
"""
Forensic audit for measurement pipeline breaks
Identifies exact failure modes: ID drift, label joins, tokenizer changes, metric defaults
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import statistics

def audit_measurement_pipeline(results_file):
    """Run all forensic probes to identify pipeline breaks"""
    
    with open(results_file) as f:
        data = json.load(f)
    
    print("🔍 FORENSIC AUDIT: Measurement Pipeline Break Analysis")
    print("=" * 60)
    
    # Probe 1: Key-space audit (dataset collapse detection)
    print("\n📊 PROBE 1: Key-space & Dataset Audit")
    
    expected_datasets = set(data['config']['datasets'])  # [code_debug, code_qa, zh_qa]
    actual_datasets = set()
    
    all_results = []
    for method in data['results']:
        all_results.extend(data['results'][method])
    
    dataset_counts = Counter()
    for result in all_results:
        dataset = result['dataset']
        actual_datasets.add(dataset)
        dataset_counts[dataset] += 1
    
    print(f"  Expected datasets: {expected_datasets}")
    print(f"  Actual datasets: {actual_datasets}")
    print(f"  Dataset counts: {dict(dataset_counts)}")
    
    # Critical failure: dataset collapse
    if 'code' in actual_datasets and ('code_debug' not in actual_datasets or 'code_qa' not in actual_datasets):
        print("  ❌ CRITICAL: Dataset collapse detected! 'code_debug'/'code_qa' → 'code'")
        print("     This breaks label joins and explains P@5=0")
    else:
        print("  ✅ Dataset names preserved")
    
    # Probe 2: Metric validation (impossible values detection)
    print("\n📊 PROBE 2: Metric Validation & Defaults Detection")
    
    p_at_5_values = []
    kv_reuse_values = []
    delta_cbu_values = []
    tokens_kept_zh = []
    
    for result in all_results:
        p_at_5_values.append(result['p_at_k']['5'])
        kv_reuse_values.append(result['kv_reuse'])
        delta_cbu_values.append(result['delta_cbu_per_1k'])
        
        if result['dataset'] == 'zh_qa':
            tokens_kept_zh.append((result['keep_ratio'], result['tokens_kept']))
    
    # Check for universal zeros (impossible)
    p_at_5_zero_rate = sum(1 for x in p_at_5_values if x == 0.0) / len(p_at_5_values)
    kv_reuse_zero_rate = sum(1 for x in kv_reuse_values if x == 0.0) / len(kv_reuse_values)
    
    print(f"  P@5 = 0.0 rate: {p_at_5_zero_rate:.1%} ({sum(1 for x in p_at_5_values if x == 0.0)}/{len(p_at_5_values)})")
    print(f"  KV reuse = 0.0 rate: {kv_reuse_zero_rate:.1%}")
    
    if p_at_5_zero_rate > 0.9:
        print("  ❌ CRITICAL: Universal P@5=0 indicates label join failure")
    
    if kv_reuse_zero_rate > 0.9:
        print("  ❌ CRITICAL: Universal KV reuse=0 indicates metric defaulting")
    
    # Check delta_cbu variance (should vary by method/scenario)
    delta_variance = statistics.variance(delta_cbu_values) if len(delta_cbu_values) > 1 else 0
    print(f"  ΔCBU variance: {delta_variance:.6f}")
    
    if delta_variance < 1e-6:
        print("  ❌ CRITICAL: ΔCBU values are constants, not computed")
    
    # Probe 3: Token count sanity (zh_qa specific)
    print("\n📊 PROBE 3: Token Count Sanity (zh_qa)")
    
    zh_tokens_by_ratio = defaultdict(list)
    for keep_ratio, tokens in tokens_kept_zh:
        zh_tokens_by_ratio[keep_ratio].append(tokens)
    
    for ratio in sorted(zh_tokens_by_ratio.keys()):
        tokens_list = zh_tokens_by_ratio[ratio]
        avg_tokens = statistics.mean(tokens_list)
        print(f"  Keep ratio {ratio:.1%}: avg {avg_tokens:.1f} tokens")
        
        if avg_tokens < 100:
            print(f"    ❌ CRITICAL: {avg_tokens:.1f} tokens is implausibly low")
            print("       Likely confusion with window/sink counts")
    
    # Probe 4: Method-specific failure patterns
    print("\n📊 PROBE 4: Method-Specific Patterns")
    
    method_results = defaultdict(list)
    for method in data['results']:
        for result in data['results'][method]:
            method_results[method].append(result)
    
    for method in method_results:
        results = method_results[method]
        p95_times = [r['middleware_p95_ms'] for r in results]
        p95_variance = statistics.variance(p95_times) if len(p95_times) > 1 else 0
        
        print(f"  {method}: {len(results)} results, P95 variance: {p95_variance:.1f}")
        
        # Check if all times are suspiciously similar (indicates mock/default data)
        if p95_variance < 1.0:
            print(f"    ⚠️  Low P95 variance suggests mock/default timing data")
    
    # Summary assessment
    print("\n🎯 FORENSIC SUMMARY")
    print("=" * 60)
    
    critical_issues = []
    
    if 'code' in actual_datasets and 'code_debug' not in actual_datasets:
        critical_issues.append("Dataset ID collapse (code_debug/code_qa → code)")
    
    if p_at_5_zero_rate > 0.9:
        critical_issues.append("Universal P@5=0 (label join failure)")
    
    if kv_reuse_zero_rate > 0.9:
        critical_issues.append("Universal KV reuse=0 (metric defaulting)")
    
    if delta_variance < 1e-6:
        critical_issues.append("Constant ΔCBU values (not computed)")
    
    # Check zh_qa token sanity
    zh_low_tokens = any(statistics.mean(zh_tokens_by_ratio[r]) < 100 
                       for r in zh_tokens_by_ratio)
    if zh_low_tokens:
        critical_issues.append("zh_qa token counts impossibly low")
    
    if critical_issues:
        print("❌ MEASUREMENT PIPELINE IS BROKEN")
        print("Critical issues found:")
        for issue in critical_issues:
            print(f"  • {issue}")
        print("\n🚨 DO NOT PUBLISH these results - they are measurement artifacts")
        return False
    else:
        print("✅ Measurement pipeline appears healthy")
        return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python forensic_audit.py <results.json>")
        sys.exit(1)
    
    results_file = Path(sys.argv[1])
    if not results_file.exists():
        print(f"Error: {results_file} not found")
        sys.exit(1)
    
    is_healthy = audit_measurement_pipeline(results_file)
    sys.exit(0 if is_healthy else 1)

if __name__ == "__main__":
    main()