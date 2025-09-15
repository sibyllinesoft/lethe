#!/usr/bin/env python3
"""
Benchmark Log Analysis Utilities
===============================

Tools for analyzing proxy logs generated during benchmark runs to extract
insights about LLM request patterns, performance, and transformations.

Features:
- Parse structured proxy logs (JSON Lines format)
- Correlate requests with benchmark queries
- Generate performance metrics and visualizations
- Analyze request/response transformations
- Export analysis results in multiple formats

Usage:
    python log_analyzer.py --run-id competitive-20240909-141530
    python log_analyzer.py --log-file path/to/proxy.jsonl --output-format html
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BenchmarkLogAnalyzer:
    """Analyzer for benchmark proxy logs."""
    
    def __init__(self, log_file_path: str):
        self.log_file = Path(log_file_path)
        self.logs = []
        self.parsed_logs = {}
        
        if not self.log_file.exists():
            raise FileNotFoundError(f"Log file not found: {self.log_file}")
        
        self._load_logs()
        self._parse_logs()
    
    def _load_logs(self):
        """Load logs from JSON Lines file."""
        logger.info(f"Loading logs from {self.log_file}")
        
        with open(self.log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        log_entry = json.loads(line)
                        log_entry['_line_number'] = line_num
                        self.logs.append(log_entry)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num}: {e}")
        
        logger.info(f"Loaded {len(self.logs)} log entries")
    
    def _parse_logs(self):
        """Parse logs into structured format."""
        self.parsed_logs = {
            'requests': [l for l in self.logs if l.get('event') == 'proxy_request_transform'],
            'responses': [l for l in self.logs if l.get('event') == 'proxy_response'],
            'errors': [l for l in self.logs if l.get('event') == 'proxy_error'],
            'debug': [l for l in self.logs if l.get('event') == 'proxy_debug']
        }
        
        logger.info(f"Parsed logs: {len(self.parsed_logs['requests'])} requests, "
                   f"{len(self.parsed_logs['responses'])} responses, "
                   f"{len(self.parsed_logs['errors'])} errors")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary of benchmark run."""
        requests = self.parsed_logs['requests']
        responses = self.parsed_logs['responses']
        errors = self.parsed_logs['errors']
        
        if not requests:
            return {"error": "No request logs found"}
        
        # Basic metrics
        providers = list(set(r.get('provider') for r in requests if r.get('provider')))
        total_requests = len(requests)
        total_responses = len(responses)
        error_rate = len(errors) / total_requests if total_requests > 0 else 0
        
        # Timing analysis
        response_times = [
            r.get('performance', {}).get('total_request_duration_ms', 0)
            for r in responses if r.get('performance')
        ]
        
        # Request size analysis
        request_sizes = [
            r.get('pre_transform', {}).get('size_bytes', 0)
            for r in requests if r.get('pre_transform')
        ]
        
        response_sizes = [
            r.get('post_transform', {}).get('size_bytes', 0) 
            for r in responses if r.get('post_transform')
        ]
        
        # Transformation analysis
        transformations = []
        for req in requests:
            transform = req.get('transform', {})
            if transform.get('changes'):
                transformations.extend(transform['changes'])
        
        from collections import Counter
        transformation_counts = Counter(transformations)
        
        # Time range
        timestamps = [
            datetime.fromisoformat(log.get('timestamp', '').replace('Z', '+00:00'))
            for log in self.logs if log.get('timestamp')
        ]
        
        time_range = {
            "start": min(timestamps).isoformat() if timestamps else None,
            "end": max(timestamps).isoformat() if timestamps else None,
            "duration_seconds": (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) > 1 else 0
        }
        
        return {
            "summary": {
                "total_requests": total_requests,
                "total_responses": total_responses,
                "error_count": len(errors),
                "error_rate_percent": error_rate * 100,
                "providers_used": providers,
                "time_range": time_range
            },
            "performance": {
                "avg_response_time_ms": sum(response_times) / len(response_times) if response_times else 0,
                "min_response_time_ms": min(response_times) if response_times else 0,
                "max_response_time_ms": max(response_times) if response_times else 0,
                "p95_response_time_ms": self._percentile(response_times, 95) if response_times else 0
            },
            "data_transfer": {
                "avg_request_size_bytes": sum(request_sizes) / len(request_sizes) if request_sizes else 0,
                "avg_response_size_bytes": sum(response_sizes) / len(response_sizes) if response_sizes else 0,
                "total_data_transferred_mb": (sum(request_sizes) + sum(response_sizes)) / 1024 / 1024,
                "compression_ratio": sum(response_sizes) / sum(request_sizes) if sum(request_sizes) > 0 else 0
            },
            "transformations": {
                "total_transformations": len(transformations),
                "transformation_types": dict(transformation_counts),
                "most_common_transformation": transformation_counts.most_common(1)[0][0] if transformation_counts else None
            }
        }
    
    def analyze_provider_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics by provider."""
        requests = self.parsed_logs['requests']
        responses = self.parsed_logs['responses']
        
        # Group by provider
        provider_data = {}
        
        for request in requests:
            provider = request.get('provider')
            if not provider:
                continue
                
            if provider not in provider_data:
                provider_data[provider] = {
                    'requests': [],
                    'responses': [],
                    'request_sizes': [],
                    'response_times': []
                }
            
            provider_data[provider]['requests'].append(request)
            
            # Find corresponding response
            request_id = request.get('request_id')
            matching_response = next(
                (r for r in responses if r.get('request_id') == request_id), None
            )
            
            if matching_response:
                provider_data[provider]['responses'].append(matching_response)
                
                # Extract metrics
                if request.get('pre_transform', {}).get('size_bytes'):
                    provider_data[provider]['request_sizes'].append(
                        request['pre_transform']['size_bytes']
                    )
                
                if matching_response.get('performance', {}).get('total_request_duration_ms'):
                    provider_data[provider]['response_times'].append(
                        matching_response['performance']['total_request_duration_ms']
                    )
        
        # Calculate provider-specific metrics
        provider_analysis = {}
        for provider, data in provider_data.items():
            response_times = data['response_times']
            request_sizes = data['request_sizes']
            
            provider_analysis[provider] = {
                "request_count": len(data['requests']),
                "response_count": len(data['responses']),
                "success_rate_percent": (len(data['responses']) / len(data['requests']) * 100) 
                    if data['requests'] else 0,
                "avg_response_time_ms": sum(response_times) / len(response_times) if response_times else 0,
                "avg_request_size_bytes": sum(request_sizes) / len(request_sizes) if request_sizes else 0,
                "p95_response_time_ms": self._percentile(response_times, 95) if response_times else 0
            }
        
        return provider_analysis
    
    def analyze_request_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in request content and transformations."""
        requests = self.parsed_logs['requests']
        
        # Model analysis
        models_used = []
        for req in requests:
            payload = req.get('pre_transform', {}).get('payload', {})
            if 'model' in payload:
                models_used.append(payload['model'])
        
        # Message structure analysis  
        message_patterns = {
            "system_messages": 0,
            "user_messages": 0,
            "assistant_messages": 0,
            "avg_messages_per_request": 0
        }
        
        total_message_counts = []
        for req in requests:
            payload = req.get('pre_transform', {}).get('payload', {})
            messages = payload.get('messages', [])
            total_message_counts.append(len(messages))
            
            for msg in messages:
                role = msg.get('role', '')
                if role == 'system':
                    message_patterns["system_messages"] += 1
                elif role == 'user':
                    message_patterns["user_messages"] += 1
                elif role == 'assistant':
                    message_patterns["assistant_messages"] += 1
        
        message_patterns["avg_messages_per_request"] = (
            sum(total_message_counts) / len(total_message_counts) if total_message_counts else 0
        )
        
        from collections import Counter
        
        return {
            "models_used": dict(Counter(models_used)),
            "message_patterns": message_patterns,
            "avg_messages_per_request": message_patterns["avg_messages_per_request"],
            "transformation_frequency": self._analyze_transformation_frequency()
        }
    
    def _analyze_transformation_frequency(self) -> Dict[str, Any]:
        """Analyze how often different transformations are applied."""
        requests = self.parsed_logs['requests']
        
        transformation_stats = {
            "total_requests": len(requests),
            "requests_with_transforms": 0,
            "transformation_types": {}
        }
        
        for req in requests:
            transform = req.get('transform', {})
            if transform.get('enabled') and transform.get('changes'):
                transformation_stats["requests_with_transforms"] += 1
                
                for change in transform['changes']:
                    if change not in transformation_stats["transformation_types"]:
                        transformation_stats["transformation_types"][change] = 0
                    transformation_stats["transformation_types"][change] += 1
        
        transformation_stats["transform_rate_percent"] = (
            transformation_stats["requests_with_transforms"] / len(requests) * 100 
            if requests else 0
        )
        
        return transformation_stats
    
    def generate_visualizations(self, output_dir: Path):
        """Generate visualization plots."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Response time distribution
        self._plot_response_times(output_dir)
        
        # 2. Provider performance comparison
        self._plot_provider_performance(output_dir)
        
        # 3. Request size over time
        self._plot_request_sizes(output_dir)
        
        # 4. Transformation analysis
        self._plot_transformations(output_dir)
        
        logger.info(f"Generated visualizations in {output_dir}")
    
    def _plot_response_times(self, output_dir: Path):
        """Plot response time distribution."""
        responses = self.parsed_logs['responses']
        response_times = [
            r.get('performance', {}).get('total_request_duration_ms', 0)
            for r in responses if r.get('performance')
        ]
        
        if not response_times:
            logger.warning("No response time data found")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(response_times, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Response Time (ms)')
        plt.ylabel('Frequency')
        plt.title('Response Time Distribution')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(1, 2, 2)
        plt.boxplot(response_times)
        plt.ylabel('Response Time (ms)')
        plt.title('Response Time Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'response_times.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_provider_performance(self, output_dir: Path):
        """Plot provider performance comparison."""
        provider_analysis = self.analyze_provider_performance()
        
        if not provider_analysis:
            logger.warning("No provider performance data found")
            return
        
        providers = list(provider_analysis.keys())
        avg_times = [provider_analysis[p]['avg_response_time_ms'] for p in providers]
        success_rates = [provider_analysis[p]['success_rate_percent'] for p in providers]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Average response time
        bars1 = ax1.bar(providers, avg_times, alpha=0.7)
        ax1.set_xlabel('Provider')
        ax1.set_ylabel('Average Response Time (ms)')
        ax1.set_title('Average Response Time by Provider')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, avg_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.0f}', ha='center', va='bottom')
        
        # Success rate
        bars2 = ax2.bar(providers, success_rates, alpha=0.7, color='green')
        ax2.set_xlabel('Provider')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rate by Provider')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars2, success_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'provider_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_request_sizes(self, output_dir: Path):
        """Plot request sizes over time."""
        requests = self.parsed_logs['requests']
        
        # Extract timestamps and sizes
        times_and_sizes = []
        for req in requests:
            timestamp_str = req.get('timestamp')
            size = req.get('pre_transform', {}).get('size_bytes')
            
            if timestamp_str and size:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    times_and_sizes.append((timestamp, size))
                except ValueError:
                    continue
        
        if not times_and_sizes:
            logger.warning("No request size data found")
            return
        
        times_and_sizes.sort(key=lambda x: x[0])
        timestamps, sizes = zip(*times_and_sizes)
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, sizes, marker='o', markersize=4, alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Request Size (bytes)')
        plt.title('Request Size Over Time')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'request_sizes.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_transformations(self, output_dir: Path):
        """Plot transformation analysis."""
        transform_stats = self._analyze_transformation_frequency()
        
        if not transform_stats["transformation_types"]:
            logger.warning("No transformation data found")
            return
        
        transform_types = list(transform_stats["transformation_types"].keys())
        transform_counts = list(transform_stats["transformation_types"].values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(transform_types, transform_counts, alpha=0.7)
        plt.xlabel('Transformation Type')
        plt.ylabel('Count')
        plt.title('Request Transformations Applied')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, transform_counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'transformations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_analysis(self, output_path: Path, format: str = 'json'):
        """Export analysis results in specified format."""
        analysis = {
            "summary": self.generate_summary_report(),
            "provider_performance": self.analyze_provider_performance(),
            "request_patterns": self.analyze_request_patterns(),
            "generated_at": datetime.now().isoformat()
        }
        
        if format == 'json':
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(analysis, f, indent=2)
        elif format == 'html':
            self._export_html_report(analysis, output_path.with_suffix('.html'))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Analysis exported to {output_path}")
    
    def _export_html_report(self, analysis: Dict, output_path: Path):
        """Export analysis as HTML report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Benchmark Log Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { border-bottom: 2px solid #ccc; padding-bottom: 20px; }
                .section { margin: 30px 0; }
                .metric { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .success { color: green; }
                .warning { color: orange; }
                .error { color: red; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Benchmark Log Analysis Report</h1>
                <p>Generated: {generated_at}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                {summary_html}
            </div>
            
            <div class="section">
                <h2>Provider Performance</h2>
                {provider_html}
            </div>
            
            <div class="section">
                <h2>Request Patterns</h2>
                {patterns_html}
            </div>
        </body>
        </html>
        """
        
        # Generate HTML sections
        summary = analysis['summary']
        summary_html = f"""
        <div class="metric">Total Requests: {summary['summary']['total_requests']}</div>
        <div class="metric">Total Responses: {summary['summary']['total_responses']}</div>
        <div class="metric">Error Rate: {summary['summary']['error_rate_percent']:.1f}%</div>
        <div class="metric">Average Response Time: {summary['performance']['avg_response_time_ms']:.0f} ms</div>
        <div class="metric">Data Transferred: {summary['data_transfer']['total_data_transferred_mb']:.2f} MB</div>
        """
        
        # Provider performance table
        provider_rows = ""
        for provider, metrics in analysis['provider_performance'].items():
            provider_rows += f"""
            <tr>
                <td>{provider}</td>
                <td>{metrics['request_count']}</td>
                <td>{metrics['success_rate_percent']:.1f}%</td>
                <td>{metrics['avg_response_time_ms']:.0f} ms</td>
            </tr>
            """
        
        provider_html = f"""
        <table>
            <tr><th>Provider</th><th>Requests</th><th>Success Rate</th><th>Avg Response Time</th></tr>
            {provider_rows}
        </table>
        """
        
        patterns_html = f"""
        <div class="metric">Models Used: {list(analysis['request_patterns']['models_used'].keys())}</div>
        <div class="metric">Avg Messages per Request: {analysis['request_patterns']['avg_messages_per_request']:.1f}</div>
        <div class="metric">Transform Rate: {analysis['request_patterns']['transformation_frequency']['transform_rate_percent']:.1f}%</div>
        """
        
        # Write HTML file
        html_content = html_template.format(
            generated_at=analysis['generated_at'],
            summary_html=summary_html,
            provider_html=provider_html,
            patterns_html=patterns_html
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(percentile / 100.0 * (len(sorted_data) - 1))
        return sorted_data[index]

def find_run_logs(run_id: str) -> Optional[Path]:
    """Find log files for a specific run ID."""
    benchmark_logs_dir = Path("benchmark-logs")
    run_dir = benchmark_logs_dir / "runs" / run_id
    
    if not run_dir.exists():
        return None
    
    proxy_log_file = run_dir / "proxy-logs" / f"{run_id}-proxy.jsonl"
    if proxy_log_file.exists():
        return proxy_log_file
    
    return None

def main():
    """Main analysis execution."""
    parser = argparse.ArgumentParser(description="Benchmark Log Analyzer")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", help="Benchmark run ID to analyze")
    group.add_argument("--log-file", help="Path to proxy log file")
    
    parser.add_argument("--output-dir", default="./analysis-output",
                       help="Output directory for analysis results")
    parser.add_argument("--output-format", choices=["json", "html"], default="json",
                       help="Output format for analysis results")
    parser.add_argument("--generate-plots", action="store_true",
                       help="Generate visualization plots")
    
    args = parser.parse_args()
    
    # Find log file
    if args.run_id:
        log_file = find_run_logs(args.run_id)
        if not log_file:
            logger.error(f"No logs found for run ID: {args.run_id}")
            return 1
    else:
        log_file = Path(args.log_file)
    
    try:
        # Initialize analyzer
        analyzer = BenchmarkLogAnalyzer(str(log_file))
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate analysis
        logger.info("Generating analysis...")
        
        # Export analysis results
        output_file = output_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        analyzer.export_analysis(output_file, args.output_format)
        
        # Generate plots if requested
        if args.generate_plots:
            plot_dir = output_dir / "plots"
            analyzer.generate_visualizations(plot_dir)
        
        # Print summary
        summary = analyzer.generate_summary_report()
        print("\n" + "="*60)
        print("BENCHMARK LOG ANALYSIS SUMMARY")
        print("="*60)
        print(f"Log file: {log_file}")
        print(f"Total requests: {summary['summary']['total_requests']}")
        print(f"Error rate: {summary['summary']['error_rate_percent']:.1f}%")
        print(f"Avg response time: {summary['performance']['avg_response_time_ms']:.0f} ms")
        print(f"Data transferred: {summary['data_transfer']['total_data_transferred_mb']:.2f} MB")
        print(f"Providers: {', '.join(summary['summary']['providers_used'])}")
        print(f"\nResults saved to: {output_dir}")
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())