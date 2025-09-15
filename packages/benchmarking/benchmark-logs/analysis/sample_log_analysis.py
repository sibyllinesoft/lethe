#!/usr/bin/env python3
"""
Sample Log Analysis Demonstration
=================================

This script demonstrates the kind of insights available from benchmark proxy logs
by creating sample data and showing analysis patterns.

This is for demonstration purposes - run actual benchmarks to get real data.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

def generate_sample_logs(num_requests: int = 50) -> List[Dict[str, Any]]:
    """Generate sample proxy logs for demonstration."""
    
    providers = ["openai", "anthropic", "groq"]
    models = {
        "openai": ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        "anthropic": ["claude-3-sonnet", "claude-3-haiku", "claude-3-opus"],
        "groq": ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]
    }
    
    benchmark_types = ["competitive", "semantic", "longcontext"]
    datasets = ["infinitebench_kv_retrieval", "longbench_qa", "ruler_retrieval"]
    
    logs = []
    run_id = f"sample-demo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    start_time = datetime.now() - timedelta(hours=2)
    
    for i in range(num_requests):
        provider = random.choice(providers)
        model = random.choice(models[provider])
        benchmark_type = random.choice(benchmark_types)
        dataset = random.choice(datasets)
        
        # Request timestamp with some spread
        timestamp = start_time + timedelta(seconds=i * 30 + random.randint(0, 60))
        request_id = f"benchmark-{run_id}-{provider}-{i:03d}"
        
        # Request transform log
        request_size = random.randint(2000, 50000)  # 2KB to 50KB
        response_size = random.randint(100, 2000)   # 100B to 2KB
        transform_duration = random.randint(1, 15)
        total_duration = random.randint(500, 8000)  # 0.5s to 8s
        
        # Simulate transformation changes
        changes = ["system_prelude_added", "benchmark_metadata_added"]
        if random.random() > 0.7:  # 30% chance of content rewriting
            changes.append("user_content_rewritten")
        if random.random() > 0.9:  # 10% chance of format conversion
            changes.append("format_converted")
        
        size_change_percent = (response_size - request_size) / request_size * 100
        
        # Request transform event
        request_log = {
            "timestamp": timestamp.isoformat() + "Z",
            "level": "INFO",
            "event": "proxy_request_transform",
            "request_id": request_id,
            "benchmark_metadata": {
                "run_id": run_id,
                "query_id": f"{dataset}-{i:03d}",
                "provider": provider,
                "benchmark_type": benchmark_type,
                "dataset": dataset
            },
            "provider": provider,
            "path": "/v1/chat/completions" if provider != "anthropic" else "/v1/messages",
            "method": "POST",
            "transform": {
                "enabled": True,
                "duration_ms": transform_duration,
                "changes": changes,
                "size_change_percent": size_change_percent
            },
            "pre_transform": {
                "size_bytes": request_size,
                "token_estimate": request_size // 4,  # Rough estimate
                "payload": {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": f"Sample benchmark query {i} for {dataset}..."}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            },
            "post_transform": {
                "size_bytes": request_size + int(request_size * 0.1),  # 10% larger after transform
                "token_estimate": (request_size + int(request_size * 0.1)) // 4,
                "payload": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are an AI assistant participating in a Lethe benchmark evaluation..."},
                        {"role": "user", "content": f"Sample benchmark query {i} for {dataset}..."}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1000
                }
            },
            "performance": {
                "transform_duration_ms": transform_duration,
                "total_request_duration_ms": None,  # Will be filled in response
                "pre_transform_size_bytes": request_size,
                "post_transform_size_bytes": request_size + int(request_size * 0.1),
                "size_change_percent": 10.0
            }
        }
        logs.append(request_log)
        
        # Response event (simulate 95% success rate)
        if random.random() < 0.95:  # 95% success rate
            response_time = timestamp + timedelta(milliseconds=total_duration)
            
            response_log = {
                "timestamp": response_time.isoformat() + "Z",
                "level": "INFO",
                "event": "proxy_response",
                "request_id": request_id,
                "provider": provider,
                "status_code": 200,
                "response_size_bytes": response_size,
                "performance": {
                    "transform_duration_ms": transform_duration,
                    "total_request_duration_ms": total_duration,
                    "response_tokens": response_size // 4,  # Rough estimate
                    "response_time_ms": total_duration - transform_duration
                }
            }
            logs.append(response_log)
        else:
            # Error response
            error_time = timestamp + timedelta(milliseconds=random.randint(1000, 3000))
            error_log = {
                "timestamp": error_time.isoformat() + "Z", 
                "level": "ERROR",
                "event": "proxy_error",
                "request_id": request_id,
                "provider": provider,
                "error": {
                    "type": "api_error",
                    "message": "Rate limit exceeded" if random.random() < 0.6 else "Authentication failed",
                    "status_code": 429 if "Rate limit" in ("Rate limit exceeded") else 401
                }
            }
            logs.append(error_log)
    
    return sorted(logs, key=lambda x: x["timestamp"])

def analyze_sample_logs(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze sample logs to demonstrate insights."""
    
    requests = [l for l in logs if l.get("event") == "proxy_request_transform"]
    responses = [l for l in logs if l.get("event") == "proxy_response"]
    errors = [l for l in logs if l.get("event") == "proxy_error"]
    
    # Basic metrics
    total_requests = len(requests)
    successful_responses = len(responses)
    error_count = len(errors)
    success_rate = successful_responses / total_requests if total_requests > 0 else 0
    
    # Provider analysis
    provider_stats = {}
    for req in requests:
        provider = req.get("provider")
        if provider not in provider_stats:
            provider_stats[provider] = {
                "requests": 0,
                "responses": 0,
                "errors": 0,
                "response_times": [],
                "request_sizes": []
            }
        
        provider_stats[provider]["requests"] += 1
        
        # Add request size
        size = req.get("pre_transform", {}).get("size_bytes", 0)
        if size > 0:
            provider_stats[provider]["request_sizes"].append(size)
        
        # Find matching response or error
        request_id = req.get("request_id")
        response = next((r for r in responses if r.get("request_id") == request_id), None)
        error = next((e for e in errors if e.get("request_id") == request_id), None)
        
        if response:
            provider_stats[provider]["responses"] += 1
            duration = response.get("performance", {}).get("total_request_duration_ms", 0)
            if duration > 0:
                provider_stats[provider]["response_times"].append(duration)
        elif error:
            provider_stats[provider]["errors"] += 1
    
    # Calculate provider metrics
    for provider in provider_stats:
        stats = provider_stats[provider]
        stats["success_rate"] = stats["responses"] / stats["requests"] if stats["requests"] > 0 else 0
        stats["avg_response_time"] = sum(stats["response_times"]) / len(stats["response_times"]) if stats["response_times"] else 0
        stats["avg_request_size"] = sum(stats["request_sizes"]) / len(stats["request_sizes"]) if stats["request_sizes"] else 0
    
    # Transformation analysis
    all_changes = []
    for req in requests:
        changes = req.get("transform", {}).get("changes", [])
        all_changes.extend(changes)
    
    from collections import Counter
    transformation_counts = Counter(all_changes)
    
    # Time-based analysis
    timestamps = []
    for log in logs:
        timestamp_str = log.get("timestamp", "")
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                timestamps.append(timestamp)
            except ValueError:
                continue
    
    time_range = {
        "start": min(timestamps).isoformat() if timestamps else None,
        "end": max(timestamps).isoformat() if timestamps else None,
        "duration_minutes": (max(timestamps) - min(timestamps)).total_seconds() / 60 if len(timestamps) > 1 else 0
    }
    
    return {
        "summary": {
            "total_requests": total_requests,
            "successful_responses": successful_responses,
            "error_count": error_count,
            "success_rate_percent": success_rate * 100,
            "time_range": time_range
        },
        "provider_performance": provider_stats,
        "transformations": {
            "total_changes": len(all_changes),
            "change_types": dict(transformation_counts),
            "most_common": transformation_counts.most_common(1)[0][0] if transformation_counts else None
        }
    }

def generate_sample_visualizations(analysis: Dict[str, Any], output_dir: Path):
    """Generate sample visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    
    provider_stats = analysis["provider_performance"]
    providers = list(provider_stats.keys())
    
    # 1. Provider Performance Comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Success rates
    success_rates = [provider_stats[p]["success_rate"] * 100 for p in providers]
    bars1 = ax1.bar(providers, success_rates, alpha=0.7, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Success Rate by Provider")
    ax1.set_ylim(0, 100)
    
    for bar, rate in zip(bars1, success_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f"{rate:.1f}%", ha="center", va="bottom")
    
    # Average response times
    avg_times = [provider_stats[p]["avg_response_time"] for p in providers]
    bars2 = ax2.bar(providers, avg_times, alpha=0.7, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax2.set_ylabel("Average Response Time (ms)")
    ax2.set_title("Response Time by Provider")
    
    for bar, time_val in zip(bars2, avg_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f"{time_val:.0f}ms", ha="center", va="bottom")
    
    # Request sizes
    avg_sizes = [provider_stats[p]["avg_request_size"] / 1024 for p in providers]  # Convert to KB
    bars3 = ax3.bar(providers, avg_sizes, alpha=0.7, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax3.set_ylabel("Average Request Size (KB)")
    ax3.set_title("Request Size by Provider")
    
    for bar, size in zip(bars3, avg_sizes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f"{size:.1f}KB", ha="center", va="bottom")
    
    # Transformation frequency
    transform_data = analysis["transformations"]
    change_types = list(transform_data["change_types"].keys())
    change_counts = list(transform_data["change_types"].values())
    
    bars4 = ax4.bar(range(len(change_types)), change_counts, alpha=0.7)
    ax4.set_xlabel("Transformation Type")
    ax4.set_ylabel("Count")
    ax4.set_title("Request Transformations Applied")
    ax4.set_xticks(range(len(change_types)))
    ax4.set_xticklabels(change_types, rotation=45, ha="right")
    
    for bar, count in zip(bars4, change_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                str(count), ha="center", va="bottom")
    
    plt.suptitle("Benchmark Proxy Log Analysis - Sample Data", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(output_dir / "sample_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Response Time Distribution
    all_response_times = []
    for provider in providers:
        all_response_times.extend(provider_stats[provider]["response_times"])
    
    if all_response_times:
        plt.figure(figsize=(10, 6))
        plt.hist(all_response_times, bins=20, alpha=0.7, edgecolor="black")
        plt.xlabel("Response Time (ms)")
        plt.ylabel("Frequency")
        plt.title("Response Time Distribution - All Providers")
        plt.axvline(sum(all_response_times) / len(all_response_times), 
                   color="red", linestyle="--", label=f"Mean: {sum(all_response_times) / len(all_response_times):.0f}ms")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "response_time_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

def create_sample_analysis():
    """Create a complete sample analysis demonstration."""
    
    print("🔍 Generating Sample Benchmark Logs Analysis")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("benchmark-logs/analysis/sample-output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate sample logs
    print("📊 Generating sample proxy logs...")
    logs = generate_sample_logs(num_requests=75)  # 75 requests across providers
    
    # Save sample logs
    logs_file = output_dir / "sample-proxy-logs.jsonl"
    with open(logs_file, "w") as f:
        for log in logs:
            f.write(json.dumps(log) + "\\n")
    
    print(f"   Generated {len(logs)} log entries")
    print(f"   Saved to: {logs_file}")
    
    # Analyze logs
    print("🔬 Analyzing logs...")
    analysis = analyze_sample_logs(logs)
    
    # Save analysis
    analysis_file = output_dir / "sample-analysis.json" 
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"   Analysis saved to: {analysis_file}")
    
    # Generate visualizations
    print("📈 Generating visualizations...")
    generate_sample_visualizations(analysis, output_dir)
    print(f"   Plots saved to: {output_dir}")
    
    # Print summary
    summary = analysis["summary"]
    print("\\n📋 Sample Analysis Results:")
    print("=" * 40)
    print(f"Total Requests: {summary['total_requests']}")
    print(f"Successful Responses: {summary['successful_responses']}")
    print(f"Success Rate: {summary['success_rate_percent']:.1f}%")
    print(f"Error Count: {summary['error_count']}")
    print(f"Duration: {summary['time_range']['duration_minutes']:.1f} minutes")
    
    print("\\n🏆 Provider Performance:")
    for provider, stats in analysis["provider_performance"].items():
        print(f"  {provider.upper()}:")
        print(f"    Success Rate: {stats['success_rate']*100:.1f}%")
        print(f"    Avg Response Time: {stats['avg_response_time']:.0f}ms")
        print(f"    Avg Request Size: {stats['avg_request_size']/1024:.1f}KB")
    
    transform_data = analysis["transformations"]
    print("\\n🔄 Transformations Applied:")
    print(f"  Total Changes: {transform_data['total_changes']}")
    print(f"  Most Common: {transform_data['most_common']}")
    for change_type, count in transform_data["change_types"].items():
        print(f"    {change_type}: {count}")
    
    print("\\n✅ Sample analysis complete!")
    print(f"📁 All files saved to: {output_dir}")
    print("\\nℹ️  This demonstrates the analysis capabilities available")
    print("   when running real benchmarks with proxy logging enabled.")
    
    return output_dir

if __name__ == "__main__":
    create_sample_analysis()