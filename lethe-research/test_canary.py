#!/usr/bin/env python3
"""
Canary test to verify RetrievalResult fix
"""

import sys
sys.path.append('scripts')

def test_retrieval_result_fix():
    """Test that RetrievalResult has response attribute"""
    print("🔍 Testing RetrievalResult fix...")
    
    try:
        # Import from the actual location
        sys.path.append('src/infinitebench')
        from baselines import RetrievalResult
        
        # Test creating a RetrievalResult with response
        result = RetrievalResult(
            query_id=1,
            retrieved_chunks=[],
            context_used="test",
            processing_time_ms=100.0,
            metadata={},
            response="test_response"
        )
        
        print(f"✅ RetrievalResult created with response: '{result.response}'")
        
        # Test that response attribute exists and can be accessed
        if hasattr(result, 'response'):
            print("✅ Response attribute exists")
            return True
        else:
            print("❌ Response attribute missing")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_retrieval_result_fix()
    if success:
        print("🚀 RetrievalResult fix verified - ready for evaluation!")
    else:
        print("❌ Fix failed - need to debug further")
    sys.exit(0 if success else 1)