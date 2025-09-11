#!/usr/bin/env python3
"""
Emergency fix script for measurement pipeline breaks
Patches the 5 critical issues identified by forensic audit
"""

import sys
from pathlib import Path

def find_and_patch_dataset_naming(script_path):
    """Fix dataset collapse: code_debug/code_qa -> code"""
    
    print("🔧 FIXING: Dataset naming collapse (code_debug/code_qa → code)")
    
    script_file = Path(script_path)
    if not script_file.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    # Read the current script
    with open(script_file) as f:
        content = f.read()
    
    # Look for the problematic dataset assignment
    if 'dataset = "code"' in content:
        print("  Found hardcoded dataset='code' assignment")
        
        # Replace with logic to preserve original dataset names
        fixed_content = content.replace(
            'dataset = "code"',
            '''# Preserve original dataset names for proper label joins
            if 'code_debug' in str(samples_file) or sample_data.get('task') == 'code_debug':
                dataset = "code_debug"
            elif 'code_run' in str(samples_file) or sample_data.get('task') == 'code_run':
                dataset = "code_qa"
            else:
                dataset = "code"  # fallback'''
        )
        
        with open(script_file, 'w') as f:
            f.write(fixed_content)
        
        print("  ✅ Fixed dataset naming logic")
        return True
    
    # Alternative pattern - look for dataset assignment logic
    if 'dataset' in content and 'code_debug' in content:
        print("  ⚠️  Dataset logic found but pattern unclear")
        print("     Manual inspection needed for dataset naming")
        return False
    
    print("  ❌ Dataset assignment pattern not found")
    return False

def find_and_patch_token_counting(script_path):
    """Fix zh_qa token counts (4/5/6 tokens -> should be hundreds)"""
    
    print("\n🔧 FIXING: Token counting confusion (window/sink counts vs tokens)")
    
    script_file = Path(script_path)
    if not script_file.exists():
        return False
    
    with open(script_file) as f:
        content = f.read()
    
    # Look for token counting logic
    patterns_to_fix = [
        ('tokens_kept = len(', 'tokens_kept = len(tokenizer.encode('),
        ('tokens_kept = num_windows', 'tokens_kept = len(context.split())  # Fixed: was num_windows'),
        ('tokens_kept = sink_count', 'tokens_kept = len(context.split())  # Fixed: was sink_count'),
    ]
    
    fixed = False
    for old_pattern, new_pattern in patterns_to_fix:
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)
            print(f"  ✅ Fixed: {old_pattern} -> {new_pattern}")
            fixed = True
    
    if fixed:
        with open(script_file, 'w') as f:
            f.write(content)
        print("  ✅ Token counting logic patched")
        return True
    
    print("  ⚠️  Token counting patterns not found - may need manual fix")
    return False

def patch_metric_defaults(script_path):
    """Fix metric defaulting (KV reuse=0, constant ΔCBU)"""
    
    print("\n🔧 FIXING: Metric defaulting (KV reuse=0, constant ΔCBU)")
    
    script_file = Path(script_path)
    if not script_file.exists():
        return False
    
    with open(script_file) as f:
        content = f.read()
    
    # Look for hardcoded metric defaults
    defaults_to_fix = [
        ('kv_reuse = 0.0', 'kv_reuse = compute_kv_reuse(result)  # Fixed: was hardcoded 0.0'),
        ('kv_reuse = 0', 'kv_reuse = compute_kv_reuse(result)  # Fixed: was hardcoded 0'),
        ('delta_cbu_per_1k = 0.0102', 'delta_cbu_per_1k = compute_delta_cbu(result)  # Fixed: was constant'),
    ]
    
    fixed = False
    for old_pattern, new_pattern in defaults_to_fix:
        if old_pattern in content:
            content = content.replace(old_pattern, new_pattern)
            print(f"  ✅ Fixed hardcoded: {old_pattern}")
            fixed = True
    
    if fixed:
        with open(script_file, 'w') as f:
            f.write(content)
        print("  ✅ Metric defaults removed")
        return True
    
    print("  ⚠️  Metric default patterns not found")
    return False

def add_pipeline_validation(script_path):
    """Add validation to catch pipeline breaks early"""
    
    print("\n🔧 ADDING: Pipeline validation guards")
    
    script_file = Path(script_path)
    if not script_file.exists():
        return False
    
    with open(script_file) as f:
        content = f.read()
    
    # Add validation function at the top
    validation_code = '''
def validate_measurement_pipeline(results):
    """Validate measurement pipeline - fail fast on breaks"""
    
    # Check for dataset collapse
    datasets = set(r.get('dataset', '') for r in results)
    if 'code' in datasets and ('code_debug' not in datasets or 'code_qa' not in datasets):
        raise ValueError("Dataset collapse detected: code_debug/code_qa -> code")
    
    # Check for universal zeros
    p_at_5_values = [r.get('p_at_k', {}).get('5', 0) for r in results]
    if all(p == 0.0 for p in p_at_5_values):
        raise ValueError("Universal P@5=0 indicates label join failure")
    
    # Check for metric defaults
    kv_values = [r.get('kv_reuse', 0) for r in results]
    if all(kv == 0.0 for kv in kv_values):
        raise ValueError("Universal KV reuse=0 indicates metric defaulting")
    
    # Check zh_qa token sanity
    zh_results = [r for r in results if r.get('dataset') == 'zh_qa']
    for r in zh_results:
        tokens = r.get('tokens_kept', 0)
        if tokens < 100:
            raise ValueError(f"zh_qa tokens_kept={tokens} impossibly low (window/sink confusion?)")
    
    print("✅ Pipeline validation passed")
    return True

'''
    
    # Insert validation function near the top
    if 'def validate_measurement_pipeline' not in content:
        # Find a good insertion point (after imports)
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_idx = i + 1
            elif line.strip() and not line.startswith('#'):
                break
        
        lines.insert(insert_idx, validation_code)
        content = '\n'.join(lines)
        
        # Add validation call before saving results
        if 'save_results' in content or 'json.dump' in content:
            content = content.replace(
                'json.dump(',
                'validate_measurement_pipeline(all_results)\n    json.dump('
            )
        
        with open(script_file, 'w') as f:
            f.write(content)
        
        print("  ✅ Added pipeline validation guards")
        return True
    
    print("  ✅ Validation already present")
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python fix_pipeline_breaks.py <benchmark_script.py>")
        sys.exit(1)
    
    script_path = Path(sys.argv[1])
    
    print("🚨 EMERGENCY PIPELINE FIXES")
    print("=" * 50)
    print(f"Target script: {script_path}")
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        sys.exit(1)
    
    # Create backup
    backup_path = script_path.with_suffix('.py.backup')
    import shutil
    shutil.copy2(script_path, backup_path)
    print(f"📋 Backup created: {backup_path}")
    
    # Apply fixes
    fixes_applied = []
    
    if find_and_patch_dataset_naming(script_path):
        fixes_applied.append("Dataset naming")
    
    if find_and_patch_token_counting(script_path):
        fixes_applied.append("Token counting")
    
    if patch_metric_defaults(script_path):
        fixes_applied.append("Metric defaults")
    
    if add_pipeline_validation(script_path):
        fixes_applied.append("Pipeline validation")
    
    print("\n🎯 SUMMARY")
    print("=" * 50)
    
    if fixes_applied:
        print("✅ Fixes applied:")
        for fix in fixes_applied:
            print(f"  • {fix}")
        print(f"\n📋 Backup available: {backup_path}")
        print("🚀 Ready for clean rerun")
    else:
        print("⚠️  No automatic fixes applied")
        print("   Manual inspection required")
    
    print("\nNext steps:")
    print("1. Review the fixes above")
    print("2. Run the benchmark script")
    print("3. Verify results with: python forensic_audit.py <results.json>")

if __name__ == "__main__":
    main()