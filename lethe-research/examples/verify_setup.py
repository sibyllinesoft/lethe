#!/usr/bin/env python3
"""
Verify Setup Example - Quick Start Tutorial

This script verifies that the Lethe Prompt Monitoring System is properly installed
and configured. Run this first to ensure everything is working.
"""

import sys
import os
from pathlib import Path

def verify_imports():
    """Verify all required modules can be imported."""
    print("🔍 Verifying module imports...")
    
    try:
        from src.monitoring import get_prompt_tracker
        print("  ✅ get_prompt_tracker imported successfully")
        
        from src.monitoring import track_prompt, get_analytics
        print("  ✅ tracking functions imported successfully")
        
        from src.monitoring import PromptTracker, PromptExecution
        print("  ✅ core classes imported successfully")
        
        return True
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False

def verify_database():
    """Initialize tracker and verify database functionality."""
    print("\n💾 Verifying database functionality...")
    
    try:
        from src.monitoring import get_prompt_tracker
        
        # Initialize the tracker (creates database if needed)
        tracker = get_prompt_tracker()
        print(f"  ✅ Database initialized at: {tracker.db_path}")
        
        # Test basic functionality
        stats = tracker.get_basic_stats()
        total_executions = stats.get('total_executions', 0)
        print(f"  📊 Current executions in database: {total_executions}")
        
        # Test a simple query
        recent = tracker.get_recent_executions(limit=1)
        print(f"  📝 Recent executions query successful: {len(recent)} results")
        
        return True
    except Exception as e:
        print(f"  ❌ Database error: {e}")
        return False

def verify_optional_dependencies():
    """Check optional dependencies for enhanced features."""
    print("\n📦 Checking optional dependencies...")
    
    # Check MLflow
    try:
        import mlflow
        print("  ✅ MLflow available - experiment tracking enabled")
    except ImportError:
        print("  ⚠️ MLflow not available - experiment tracking disabled")
    
    # Check Streamlit
    try:
        import streamlit
        print("  ✅ Streamlit available - dashboard enabled")
    except ImportError:
        print("  ⚠️ Streamlit not available - dashboard disabled")
    
    # Check Plotly
    try:
        import plotly
        print("  ✅ Plotly available - enhanced visualizations enabled")
    except ImportError:
        print("  ⚠️ Plotly not available - basic visualizations only")

def verify_file_permissions():
    """Check file system permissions for database and logs."""
    print("\n🔐 Verifying file permissions...")
    
    # Check experiments directory
    experiments_dir = Path("experiments")
    if experiments_dir.exists():
        print(f"  ✅ Experiments directory exists: {experiments_dir.absolute()}")
        if os.access(experiments_dir, os.W_OK):
            print("  ✅ Write permissions confirmed")
        else:
            print("  ❌ No write permissions - database creation may fail")
    else:
        print("  ⚠️ Experiments directory doesn't exist - will be created automatically")
    
    # Check logs directory
    logs_dir = experiments_dir / "logs"
    if not logs_dir.exists():
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
            print("  ✅ Logs directory created")
        except Exception as e:
            print(f"  ❌ Cannot create logs directory: {e}")

def main():
    """Run all verification checks."""
    print("🚀 Lethe Prompt Monitoring System - Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Module Imports", verify_imports),
        ("Database Functionality", verify_database), 
        ("Optional Dependencies", verify_optional_dependencies),
        ("File Permissions", verify_file_permissions)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ❌ Unexpected error in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Verification Summary:")
    
    passed = 0
    for name, result in results:
        if result:
            print(f"  ✅ {name}: PASSED")
            passed += 1
        else:
            print(f"  ❌ {name}: FAILED")
    
    print(f"\n🎯 Overall Result: {passed}/{len(results)} checks passed")
    
    if passed == len(results):
        print("🎉 Setup verification successful! You're ready to use the monitoring system.")
        return 0
    else:
        print("⚠️ Some checks failed. Please review the errors above and fix any issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())