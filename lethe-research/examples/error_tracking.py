#!/usr/bin/env python3
"""
Error Tracking Example - Quick Start Tutorial

This script demonstrates how the monitoring system handles and tracks errors
that occur during prompt execution. It shows various error scenarios and
how they are captured and analyzed.
"""

import time
import random
from src.monitoring import track_prompt

def simulate_api_error():
    """Simulate different types of API errors."""
    error_types = [
        ("rate_limit", "Rate limit exceeded: Too many requests"),
        ("timeout", "Request timeout: Server did not respond within 30 seconds"),
        ("invalid_key", "Authentication failed: Invalid API key"),
        ("model_error", "Model error: Requested model is temporarily unavailable"),
        ("quota_exceeded", "Quota exceeded: Monthly usage limit reached"),
    ]
    
    error_type, error_message = random.choice(error_types)
    
    if error_type == "timeout":
        time.sleep(0.1)  # Simulate timeout delay
        raise TimeoutError(error_message)
    elif error_type == "rate_limit":
        raise Exception(f"RateLimitError: {error_message}")
    elif error_type == "invalid_key":
        raise PermissionError(error_message)
    else:
        raise ValueError(error_message)

def simulate_processing_error():
    """Simulate errors that occur during response processing."""
    error_scenarios = [
        ("json_parse", "Failed to parse JSON response"),
        ("validation", "Response validation failed: Missing required fields"),
        ("encoding", "Unicode decode error in response text"),
        ("memory", "Out of memory while processing large response"),
    ]
    
    error_type, error_message = random.choice(error_scenarios)
    
    if error_type == "json_parse":
        raise ValueError(f"JSONDecodeError: {error_message}")
    elif error_type == "validation":
        raise AssertionError(error_message)
    elif error_type == "encoding":
        raise UnicodeDecodeError("utf-8", b"", 0, 1, error_message)
    else:
        raise MemoryError(error_message)

def main():
    """Demonstrate error tracking capabilities."""
    print("🚨 Error Tracking Examples")
    print("=" * 40)
    
    error_examples = []
    
    # Example 1: API Error
    print("\n📝 Example 1: API Error Tracking")
    
    try:
        with track_prompt(
            prompt_id="api_error_example",
            prompt_text="This request will simulate an API error",
            model_config={"model": "test-model", "temperature": 0.7},
            tags=["tutorial", "error-example", "api-error"]
        ) as execution:
            print("  🔄 Simulating API call...")
            
            # Simulate API error
            simulate_api_error()
            
            # This won't be reached
            execution.response_text = "This should not appear"
            execution.success = True
            
    except Exception as e:
        # The execution context automatically tracks the error
        print(f"  ❌ Error caught and tracked: {str(e)}")
        print(f"  📝 Execution ID: {execution.execution_id}")
        print(f"  🚨 Success status: {execution.success}")
        print(f"  ⏱️ Duration before error: {execution.execution_time_ms}ms")
        
        error_examples.append({
            'id': execution.execution_id,
            'type': 'API Error',
            'message': str(e),
            'duration': execution.execution_time_ms
        })
    
    # Example 2: Processing Error
    print("\n📝 Example 2: Processing Error Tracking")
    
    try:
        with track_prompt(
            prompt_id="processing_error_example",
            prompt_text="This request will fail during response processing",
            model_config={"model": "gpt-4", "temperature": 0.5},
            tags=["tutorial", "error-example", "processing-error"]
        ) as execution:
            print("  🔄 Simulating response processing...")
            
            # Simulate successful API call
            time.sleep(0.2)
            
            # Then simulate processing error
            simulate_processing_error()
            
            # This won't be reached
            execution.response_text = "This should not appear"
            execution.success = True
            
    except Exception as e:
        print(f"  ❌ Processing error tracked: {str(e)}")
        print(f"  📝 Execution ID: {execution.execution_id}")
        print(f"  🚨 Success status: {execution.success}")
        
        error_examples.append({
            'id': execution.execution_id,
            'type': 'Processing Error',
            'message': str(e),
            'duration': execution.execution_time_ms
        })
    
    # Example 3: Partial Success (with warnings)
    print("\n📝 Example 3: Partial Success with Warnings")
    
    try:
        with track_prompt(
            prompt_id="partial_success_example",
            prompt_text="This request will succeed but with warnings",
            model_config={"model": "gpt-4", "temperature": 0.7},
            tags=["tutorial", "warning-example", "partial-success"]
        ) as execution:
            print("  🔄 Processing with potential warnings...")
            
            # Simulate processing
            time.sleep(0.3)
            
            # Simulate a successful response with some issues
            response = "This is a response, but it may have quality issues."
            
            # Update execution
            execution.response_text = response
            execution.tokens_used = len(response.split())
            
            # Simulate quality check that reveals issues
            quality_score = 0.3  # Low quality score
            
            if quality_score < 0.5:
                # Log warning but don't fail
                warning_msg = f"Low quality response detected: score={quality_score:.2f}"
                execution.metadata["warning"] = warning_msg
                execution.metadata["quality_threshold_met"] = False
                print(f"  ⚠️ Warning: {warning_msg}")
            
            execution.response_quality_score = quality_score
            execution.success = True  # Still successful despite warning
            
        print(f"  ✅ Completed with warnings")
        print(f"  📝 Execution ID: {execution.execution_id}")
        print(f"  🎯 Quality Score: {execution.response_quality_score}")
        print(f"  ⚠️ Warnings logged in metadata")
        
        error_examples.append({
            'id': execution.execution_id,
            'type': 'Warning',
            'message': execution.metadata.get("warning", "Quality warning"),
            'duration': execution.execution_time_ms
        })
        
    except Exception as e:
        print(f"  ❌ Unexpected error: {str(e)}")
    
    # Example 4: Graceful Recovery
    print("\n📝 Example 4: Error Recovery Pattern")
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            with track_prompt(
                prompt_id=f"retry_example_attempt_{attempt + 1}",
                prompt_text="This demonstrates retry logic with error recovery",
                model_config={"model": "gpt-4", "temperature": 0.7},
                tags=["tutorial", "retry-example", f"attempt-{attempt + 1}"],
                metadata={"retry_attempt": attempt + 1, "max_retries": max_retries}
            ) as execution:
                print(f"  🔄 Attempt {attempt + 1}/{max_retries}")
                
                # Simulate failure for first attempts, success on last
                if attempt < max_retries - 1 and random.random() < 0.8:
                    raise Exception(f"Simulated failure on attempt {attempt + 1}")
                
                # Success!
                time.sleep(0.2)
                response = f"Success on attempt {attempt + 1}! This demonstrates error recovery."
                
                execution.response_text = response
                execution.response_quality_score = 0.9
                execution.tokens_used = len(response.split())
                execution.success = True
                
                print(f"  ✅ Success on attempt {attempt + 1}")
                break
                
        except Exception as e:
            print(f"  ❌ Attempt {attempt + 1} failed: {str(e)}")
            
            error_examples.append({
                'id': execution.execution_id,
                'type': f'Retry Attempt {attempt + 1}',
                'message': str(e),
                'duration': execution.execution_time_ms
            })
            
            if attempt == max_retries - 1:
                print(f"  🚨 All retry attempts exhausted")
    
    # Summary
    print(f"\n📊 Error Tracking Summary")
    print(f"=" * 30)
    print(f"Total error examples tracked: {len(error_examples)}")
    
    for i, error in enumerate(error_examples, 1):
        print(f"{i}. [{error['type']}] {error['id']}")
        print(f"   💬 {error['message'][:60]}...")
        print(f"   ⏱️ Duration: {error['duration']:.0f}ms")
    
    # Analytics on errors
    print(f"\n📈 Error Analytics:")
    
    from src.monitoring import get_prompt_tracker
    tracker = get_prompt_tracker()
    
    # Get recent failed executions
    all_executions = tracker.get_recent_executions(limit=20)
    failed_executions = [e for e in all_executions if not e['success']]
    
    print(f"  📊 Recent failed executions: {len(failed_executions)}")
    
    if failed_executions:
        avg_fail_time = sum(e['execution_time_ms'] for e in failed_executions) / len(failed_executions)
        print(f"  ⏱️ Average failure time: {avg_fail_time:.0f}ms")
        
        # Common error patterns
        error_messages = [e.get('error_message', 'Unknown') for e in failed_executions]
        print(f"  🔍 Recent error types:")
        for msg in error_messages[-3:]:  # Show last 3 errors
            print(f"    - {msg[:50]}...")
    
    print(f"\n💡 Next Steps:")
    print(f"  1. Check error analytics in dashboard")
    print(f"  2. Implement retry patterns for robust workflows")
    print(f"  3. Set up error alerting and monitoring")
    print(f"  4. Use error data to improve prompt reliability")

if __name__ == "__main__":
    main()