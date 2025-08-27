#!/usr/bin/env python3
"""
First Tracking Example - Quick Start Tutorial

This script demonstrates the basic prompt tracking functionality.
It shows how to use the context manager to track a simple prompt execution.
"""

import time
import random
from datetime import datetime
from src.monitoring import track_prompt

def simulate_llm_processing(prompt_text):
    """Simulate LLM processing with realistic timing and response."""
    # Simulate variable processing time based on prompt length
    base_time = 0.3
    char_time = len(prompt_text) * 0.001
    processing_time = base_time + char_time + random.uniform(0.1, 0.3)
    
    time.sleep(processing_time)
    
    # Generate a realistic response
    responses = [
        f"Hello! I understand your request about '{prompt_text[:30]}...' and I'm happy to help with systematic prompt monitoring using Lethe.",
        f"Great question! The prompt '{prompt_text[:30]}...' demonstrates the importance of tracking and analyzing prompt performance over time.",
        f"Thanks for the prompt: '{prompt_text[:30]}...' - This is exactly the kind of interaction that benefits from comprehensive monitoring and analysis.",
    ]
    
    return random.choice(responses)

def calculate_quality_score(prompt_text, response_text):
    """Simple quality scoring based on response characteristics."""
    # Basic heuristics for response quality
    score = 0.7  # Base score
    
    # Bonus for longer, more detailed responses
    if len(response_text) > 100:
        score += 0.1
    
    # Bonus for mentioning key terms from prompt
    key_terms = ["monitoring", "prompt", "lethe", "tracking"]
    for term in key_terms:
        if term.lower() in response_text.lower():
            score += 0.05
    
    # Small random variation to simulate real scoring
    score += random.uniform(-0.05, 0.05)
    
    return min(1.0, max(0.0, score))

def main():
    """Demonstrate basic prompt tracking."""
    print("🎯 First Prompt Tracking Example")
    print("=" * 40)
    
    # Example 1: Basic tracking
    print("\n📝 Example 1: Basic Prompt Tracking")
    
    prompt_text = "Say hello to the world of prompt monitoring!"
    model_config = {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    with track_prompt(
        prompt_id="hello_world",
        prompt_text=prompt_text,
        model_config=model_config,
        tags=["tutorial", "greeting", "first-example"]
    ) as execution:
        print(f"  🔄 Processing prompt: '{prompt_text[:50]}...'")
        
        # Simulate processing
        response = simulate_llm_processing(prompt_text)
        quality_score = calculate_quality_score(prompt_text, response)
        
        # Update execution with results
        execution.response_text = response
        execution.response_quality_score = quality_score
        execution.tokens_used = len(response.split())
        execution.success = True
        
        print(f"  ✅ Processing completed successfully!")
    
    print(f"\n📊 Execution Results:")
    print(f"  📝 Execution ID: {execution.execution_id}")
    print(f"  ⏱️ Duration: {execution.execution_time_ms}ms")
    print(f"  🎯 Quality Score: {execution.response_quality_score:.3f}")
    print(f"  🔢 Tokens Used: {execution.tokens_used}")
    print(f"  📄 Response: {execution.response_text[:100]}...")
    
    # Example 2: Batch tracking
    print(f"\n📝 Example 2: Multiple Prompt Tracking")
    
    prompts = [
        ("weather_query", "What's the weather like for outdoor activities?"),
        ("recipe_request", "Can you suggest a healthy breakfast recipe?"), 
        ("coding_help", "How do I implement error handling in Python?"),
    ]
    
    results = []
    
    for prompt_id, prompt_text in prompts:
        with track_prompt(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            model_config=model_config,
            tags=["tutorial", "batch-example"]
        ) as execution:
            # Simulate processing
            response = simulate_llm_processing(prompt_text)
            quality_score = calculate_quality_score(prompt_text, response)
            
            # Update execution
            execution.response_text = response
            execution.response_quality_score = quality_score
            execution.tokens_used = len(response.split())
            execution.success = True
            
            results.append({
                'id': execution.execution_id,
                'duration': execution.execution_time_ms,
                'quality': execution.response_quality_score,
                'tokens': execution.tokens_used
            })
    
    # Summary of batch results
    print(f"\n📈 Batch Results Summary:")
    total_duration = sum(r['duration'] for r in results)
    avg_quality = sum(r['quality'] for r in results) / len(results)
    total_tokens = sum(r['tokens'] for r in results)
    
    print(f"  📊 Processed {len(results)} prompts")
    print(f"  ⏱️ Total time: {total_duration:.0f}ms")
    print(f"  🎯 Average quality: {avg_quality:.3f}")
    print(f"  🔢 Total tokens: {total_tokens}")
    
    for i, result in enumerate(results, 1):
        print(f"    {i}. {result['id']} - {result['duration']:.0f}ms - Quality: {result['quality']:.3f}")
    
    # Example 3: Custom metadata
    print(f"\n📝 Example 3: Custom Metadata Tracking")
    
    with track_prompt(
        prompt_id="custom_metadata",
        prompt_text="Analyze the sentiment of this tutorial example.",
        model_config=model_config,
        tags=["tutorial", "sentiment", "metadata"],
        metadata={
            "user_id": "tutorial_user",
            "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "experiment_version": "v1.0",
            "custom_field": "This is custom metadata"
        }
    ) as execution:
        # Simulate processing
        response = simulate_llm_processing(execution.prompt_text)
        quality_score = calculate_quality_score(execution.prompt_text, response)
        
        # Update with results and additional metadata
        execution.response_text = response
        execution.response_quality_score = quality_score
        execution.tokens_used = len(response.split())
        execution.success = True
        
        # Add custom metrics
        execution.metadata["sentiment_score"] = random.uniform(0.6, 0.9)
        execution.metadata["confidence_level"] = random.uniform(0.8, 0.95)
    
    print(f"  📊 Custom Metadata Example:")
    print(f"  📝 Execution ID: {execution.execution_id}")
    print(f"  🏷️ Tags: {execution.tags}")
    print(f"  📋 Custom Metadata Keys: {list(execution.metadata.keys())}")
    
    print(f"\n🎉 All examples completed successfully!")
    print(f"💡 Next steps:")
    print(f"  1. Check the dashboard: python scripts/prompt_monitor.py dashboard")
    print(f"  2. View analytics: python examples/quick_analytics.py")
    print(f"  3. Explore error handling: python examples/error_tracking.py")

if __name__ == "__main__":
    main()