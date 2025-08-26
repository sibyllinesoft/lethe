#!/usr/bin/env python3
"""
Minimal test to verify core components work without dependencies.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_individual_modules():
    """Test each module individually to isolate import issues."""
    
    modules_to_test = [
        ("retriever.timing", ["TimingHarness"]),
        ("retriever.config", ["RetrieverConfig"]),
        ("retriever.metadata", ["IndexMetadata"])
    ]
    
    for module_name, classes in modules_to_test:
        try:
            print(f"Testing {module_name}...")
            module = __import__(module_name, fromlist=classes)
            
            for class_name in classes:
                cls = getattr(module, class_name)
                print(f"  ✓ {class_name} imported")
                
            print(f"✓ {module_name} successful")
            
        except Exception as e:
            print(f"❌ {module_name} failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    
def test_timing_basic():
    """Basic timing test."""
    try:
        from retriever.timing import TimingHarness
        
        harness = TimingHarness(cold_cycles=2, warm_cycles=3)
        
        def dummy_op():
            return sum(range(100))
            
        with harness.measure("test"):
            dummy_op()
            
        print("✓ Basic timing test passed")
        return True
    except Exception as e:
        print(f"❌ Timing test failed: {e}")
        return False

def test_config_basic():
    """Basic config test.""" 
    try:
        from retriever.config import RetrieverConfig
        
        config = RetrieverConfig()
        errors = config.validate()
        
        if errors:
            print(f"❌ Config validation failed: {errors}")
            return False
            
        print("✓ Basic config test passed")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Minimal IR System Test")
    print("=" * 30)
    
    test_individual_modules()
    print()
    
    success = True
    success &= test_timing_basic()
    success &= test_config_basic()
    
    print("=" * 30)
    if success:
        print("✅ Minimal tests passed!")
    else:
        print("❌ Some tests failed")
        
    sys.exit(0 if success else 1)