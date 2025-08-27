#!/usr/bin/env python3
"""
Repository Cleanup Script

This script safely removes redundant, temporary, and outdated files from the
Lethe research repository to maintain a clean and organized structure.
"""

import os
import shutil
from pathlib import Path
from typing import List, Set
import json

# Files and directories to remove
CLEANUP_TARGETS = {
    # Redundant milestone/task files (consolidated into unified framework)
    'redundant_reports': [
        'MILESTONE4_IMPLEMENTATION_SUMMARY.md',
        'MILESTONE6_COMPLETION_REPORT.md', 
        'MILESTONE7_COMPLETION_REPORT.md',
        'MILESTONE7_IMPLEMENTATION_SUMMARY.md',
        'MILESTONE8_COMPLETION_REPORT.md',
        'TASK4_IMPLEMENTATION_SUMMARY.md',
        'TASK5_COMPLETION_REPORT.md',
        'milestone7_validation_report.txt'
    ],
    
    # Temporary debug files
    'debug_files': [
        'debug_ml_plan.js',
        'debug_ts_ml.js',
        'enhanced_analysis.log'
    ],
    
    # Legacy analysis files (replaced by unified framework)
    'legacy_analysis': [
        'ADVANCED_REFACTORING_EXAMPLES.md',
        'REFACTORING_EXAMPLES.md', 
        'REFACTORING_SUMMARY.md'
    ],
    
    # Redundant documentation (consolidated)
    'redundant_docs': [
        'README_IR_SYSTEM.md',
        'RESEARCH_BUNDLE_INDEX.md',
        'RESEARCH_CONTRIBUTION_SUMMARY.md',
        'RESEARCH_STATUS.md',
        'SUBMISSION_READY_BUNDLE.md',
        'SUPPLEMENTARY_MATERIALS.md',
        'REPRODUCIBILITY_PACKAGE.md'
    ],
    
    # Test artifacts (temporary)
    'test_artifacts': [
        'test_outputs/',
        'test_artifacts/',
        'smoke-test-results/'
    ],
    
    # Temporary virtual environments
    'temp_environments': [
        'venv_stats/',
        'services/venv/'
    ]
}

# Important files to preserve (even if they match patterns)
PRESERVE_FILES = {
    'README.md',
    'src/UNIFIED_ANALYSIS_README.md',
    'paper/README.md',
    'datasets/README.md',
    'scripts/README.md',
    'docs/README.md'
}

def should_preserve_file(filepath: Path) -> bool:
    """Check if a file should be preserved."""
    return (
        filepath.name in PRESERVE_FILES or
        filepath.suffix in ['.py', '.tex', '.bib'] or
        'requirements' in filepath.name or
        filepath.name.startswith('.')
    )

def cleanup_category(category: str, targets: List[str], dry_run: bool = True) -> List[str]:
    """Clean up files in a specific category."""
    removed_files = []
    
    print(f"\n🧹 Cleaning up: {category}")
    
    for target in targets:
        target_path = Path(target)
        
        if target_path.exists():
            if not should_preserve_file(target_path):
                if dry_run:
                    print(f"  [DRY RUN] Would remove: {target}")
                else:
                    try:
                        if target_path.is_dir():
                            shutil.rmtree(target_path)
                            print(f"  ✅ Removed directory: {target}")
                        else:
                            target_path.unlink()
                            print(f"  ✅ Removed file: {target}")
                        removed_files.append(target)
                    except Exception as e:
                        print(f"  ❌ Failed to remove {target}: {e}")
            else:
                print(f"  🔒 Preserved (important): {target}")
        else:
            print(f"  ⚠️  Not found: {target}")
    
    return removed_files

def find_empty_directories(root: Path = Path(".")) -> List[Path]:
    """Find empty directories that can be removed."""
    empty_dirs = []
    
    for dirpath, dirnames, filenames in os.walk(root):
        if not dirnames and not filenames:
            empty_dirs.append(Path(dirpath))
    
    return empty_dirs

def cleanup_large_artifacts(dry_run: bool = True) -> List[str]:
    """Clean up large artifact directories but preserve structure."""
    cleaned_paths = []
    large_dirs = [
        Path("mlartifacts"),
        Path("services/mlruns") 
    ]
    
    print(f"\n🗂️  Cleaning up large artifacts (preserving structure)")
    
    for large_dir in large_dirs:
        if large_dir.exists():
            if dry_run:
                print(f"  [DRY RUN] Would clean: {large_dir}")
            else:
                # Keep directory structure but remove contents
                for item in large_dir.rglob("*"):
                    if item.is_file():
                        item.unlink()
                        cleaned_paths.append(str(item))
                print(f"  ✅ Cleaned contents of: {large_dir}")
    
    return cleaned_paths

def generate_cleanup_report(removed_files: List[str], dry_run: bool) -> str:
    """Generate a cleanup report."""
    report = {
        "cleanup_date": "2025-08-26",
        "dry_run": dry_run,
        "files_removed": len(removed_files),
        "removed_files": removed_files,
        "status": "dry_run" if dry_run else "completed"
    }
    
    return json.dumps(report, indent=2)

def main():
    """Main cleanup function."""
    print("🧹 Lethe Repository Cleanup")
    print("=" * 50)
    
    # Ask for confirmation
    dry_run = input("Run in dry-run mode? (y/n): ").lower().startswith('y')
    
    if not dry_run:
        confirm = input("⚠️  This will permanently delete files. Continue? (y/n): ")
        if not confirm.lower().startswith('y'):
            print("Cleanup cancelled.")
            return
    
    all_removed_files = []
    
    # Clean up each category
    for category, targets in CLEANUP_TARGETS.items():
        removed = cleanup_category(category, targets, dry_run)
        all_removed_files.extend(removed)
    
    # Clean up large artifacts
    artifact_files = cleanup_large_artifacts(dry_run)
    all_removed_files.extend(artifact_files)
    
    # Remove empty directories
    empty_dirs = find_empty_directories()
    if empty_dirs:
        print(f"\n📁 Found {len(empty_dirs)} empty directories")
        for empty_dir in empty_dirs:
            if not dry_run and empty_dir.exists():
                try:
                    empty_dir.rmdir()
                    print(f"  ✅ Removed empty directory: {empty_dir}")
                    all_removed_files.append(str(empty_dir))
                except OSError:
                    pass  # Directory not actually empty
    
    # Generate report
    report = generate_cleanup_report(all_removed_files, dry_run)
    
    print("\n" + "=" * 50)
    print(f"🎯 Cleanup {'simulation' if dry_run else 'completed'}")
    print(f"📊 Total items processed: {len(all_removed_files)}")
    
    if dry_run:
        print("\n💡 Run with dry_run=False to actually remove files")
    else:
        print("\n✅ Repository cleaned up successfully")
        
        # Save report
        with open("cleanup_report.json", "w") as f:
            f.write(report)
        print("📄 Cleanup report saved to cleanup_report.json")

if __name__ == "__main__":
    main()