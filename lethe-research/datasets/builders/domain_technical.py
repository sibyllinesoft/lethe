#!/usr/bin/env python3
"""
Technical Domain Builder for LetheBench
=======================================

Specialized builder for generating high-quality technical system queries that represent
realistic scenarios involving tool outputs, system messages, error logs, configuration
files, and other technical artifacts that require interpretation and analysis.

Features:
- Realistic technical query patterns involving system outputs
- Multi-tool coverage (CLI tools, logs, configs, metrics, traces)
- Diverse technical scenarios (debugging, monitoring, troubleshooting)
- Ground truth technical documentation and explanations
- Cross-reference capabilities between technical concepts
"""

import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from schema import (
    QueryRecord, GroundTruthDocument, QueryMetadata, 
    ComplexityLevel, DomainType, ValidationResult
)


@dataclass
class TechnicalPattern:
    """Technical pattern for generating realistic queries with system outputs"""
    name: str
    template: str
    complexity: ComplexityLevel
    tool_types: List[str]
    output_types: List[str]
    example_output: str
    analysis_type: str


class TechnicalDomainBuilder:
    """
    Builder for technical domain queries with realistic system output analysis scenarios
    """
    
    def __init__(self, seed: int = 42, logger: Optional[logging.Logger] = None):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize tool types and their outputs
        self.tool_categories = [
            "cli_tools", "monitoring_tools", "logs", "configuration", "metrics",
            "network_tools", "database_tools", "container_tools", "security_tools"
        ]
        
        # Initialize technical patterns by complexity
        self.technical_patterns = self._initialize_technical_patterns()
        
        # Technical concepts and their relationships
        self.technical_concepts = self._initialize_technical_concepts()
        
        # Tool-specific output formats
        self.tool_outputs = self._initialize_tool_outputs()
        
    def _initialize_technical_patterns(self) -> Dict[ComplexityLevel, List[TechnicalPattern]]:
        """Initialize comprehensive technical patterns for query generation"""
        
        patterns = {
            ComplexityLevel.SIMPLE: [
                TechnicalPattern(
                    name="error_interpretation",
                    template="What does this error mean: {error_output}",
                    complexity=ComplexityLevel.SIMPLE,
                    tool_types=["cli_tools", "logs", "container_tools"],
                    output_types=["error_message", "stack_trace", "warning"],
                    example_output="Error: connection refused on port 8080",
                    analysis_type="error_diagnosis"
                ),
                TechnicalPattern(
                    name="status_check",
                    template="What does this status output indicate: {status_output}",
                    complexity=ComplexityLevel.SIMPLE,
                    tool_types=["monitoring_tools", "cli_tools", "container_tools"],
                    output_types=["status_report", "health_check", "service_state"],
                    example_output="Service: nginx [running] PID: 1234",
                    analysis_type="status_interpretation"
                ),
                TechnicalPattern(
                    name="command_output",
                    template="Can you explain this command output: {command_output}",
                    complexity=ComplexityLevel.SIMPLE,
                    tool_types=["cli_tools", "network_tools", "database_tools"],
                    output_types=["command_result", "listing", "summary"],
                    example_output="total 64K\ndrwxr-xr-x 2 user group 4.0K",
                    analysis_type="output_explanation"
                ),
                TechnicalPattern(
                    name="configuration_syntax",
                    template="Is this configuration correct: {config_snippet}",
                    complexity=ComplexityLevel.SIMPLE,
                    tool_types=["configuration", "container_tools", "security_tools"],
                    output_types=["config_file", "yaml", "json"],
                    example_output="server {\n  listen 80;\n  server_name example.com;\n}",
                    analysis_type="syntax_validation"
                ),
                TechnicalPattern(
                    name="log_interpretation",
                    template="What happened based on this log entry: {log_entry}",
                    complexity=ComplexityLevel.SIMPLE,
                    tool_types=["logs", "monitoring_tools", "container_tools"],
                    output_types=["log_line", "access_log", "error_log"],
                    example_output="2024-01-15 14:30:22 INFO [main] Application started on port 8080",
                    analysis_type="log_analysis"
                )
            ],
            
            ComplexityLevel.MEDIUM: [
                TechnicalPattern(
                    name="performance_analysis",
                    template="Analyze this performance data and tell me what might be causing issues: {perf_data}",
                    complexity=ComplexityLevel.MEDIUM,
                    tool_types=["monitoring_tools", "metrics", "database_tools"],
                    output_types=["metrics_report", "performance_stats", "profiling_data"],
                    example_output="CPU: 85% Memory: 2.1GB/4GB Response Time: P95: 450ms Throughput: 150 req/sec",
                    analysis_type="performance_diagnosis"
                ),
                TechnicalPattern(
                    name="network_troubleshooting",
                    template="Help me troubleshoot this network issue based on the following output: {network_output}",
                    complexity=ComplexityLevel.MEDIUM,
                    tool_types=["network_tools", "monitoring_tools", "logs"],
                    output_types=["network_trace", "connectivity_test", "routing_table"],
                    example_output="traceroute: 3 hops to destination\n1. 192.168.1.1 (2ms)\n2. * * * Request timeout\n3. 10.0.0.1 (timeout)",
                    analysis_type="network_diagnosis"
                ),
                TechnicalPattern(
                    name="security_incident",
                    template="I found this in my security logs, what should I be concerned about: {security_log}",
                    complexity=ComplexityLevel.MEDIUM,
                    tool_types=["security_tools", "logs", "monitoring_tools"],
                    output_types=["security_alert", "audit_log", "intrusion_detection"],
                    example_output="[ALERT] Multiple failed login attempts from IP 192.168.1.100 (15 attempts in 60 seconds)",
                    analysis_type="security_analysis"
                ),
                TechnicalPattern(
                    name="database_performance",
                    template="This database query is running slowly, can you analyze the execution plan: {db_output}",
                    complexity=ComplexityLevel.MEDIUM,
                    tool_types=["database_tools", "monitoring_tools", "metrics"],
                    output_types=["query_plan", "database_stats", "slow_query_log"],
                    example_output="Seq Scan on users (cost=0.00..18584.82 rows=1000000 width=45) (actual time=0.123..892.456 rows=999999)",
                    analysis_type="database_optimization"
                ),
                TechnicalPattern(
                    name="container_debugging",
                    template="My container isn't working properly, here's the debug output: {container_output}",
                    complexity=ComplexityLevel.MEDIUM,
                    tool_types=["container_tools", "logs", "monitoring_tools"],
                    output_types=["container_logs", "resource_usage", "health_check"],
                    example_output="Container app-1 Status: Exited (1) 30 seconds ago\nError: OOMKilled\nMemory Usage: 512MB/512MB",
                    analysis_type="container_troubleshooting"
                )
            ],
            
            ComplexityLevel.COMPLEX: [
                TechnicalPattern(
                    name="system_failure_analysis",
                    template="We had a system outage, can you help analyze what went wrong based on these multiple data sources: {multi_source_data}",
                    complexity=ComplexityLevel.COMPLEX,
                    tool_types=["monitoring_tools", "logs", "metrics", "network_tools"],
                    output_types=["incident_report", "system_metrics", "error_correlation"],
                    example_output="Timeline:\n14:32 - Load balancer errors spike\n14:33 - Database connection pool exhausted\n14:34 - Memory usage hits 95%\n14:35 - Service becomes unresponsive",
                    analysis_type="incident_investigation"
                ),
                TechnicalPattern(
                    name="capacity_planning",
                    template="Based on this growth data and current performance metrics, help me plan for scaling: {capacity_data}",
                    complexity=ComplexityLevel.COMPLEX,
                    tool_types=["monitoring_tools", "metrics", "database_tools"],
                    output_types=["growth_metrics", "resource_trends", "forecasting_data"],
                    example_output="6-month trends:\nTraffic: +45% (1000 → 1450 req/min)\nDB queries: +62% (500 → 810 queries/sec)\nMemory: 65% → 78% utilization\nStorage: +2GB/month",
                    analysis_type="capacity_forecasting"
                ),
                TechnicalPattern(
                    name="distributed_tracing",
                    template="This distributed trace shows a performance problem, help me identify the bottleneck: {trace_data}",
                    complexity=ComplexityLevel.COMPLEX,
                    tool_types=["monitoring_tools", "metrics", "logs"],
                    output_types=["distributed_trace", "span_analysis", "service_map"],
                    example_output="Request ID: abc123\nservice-a (15ms) → service-b (450ms) → database (380ms) → cache (5ms)\nTotal: 850ms (SLA: 200ms)",
                    analysis_type="distributed_debugging"
                ),
                TechnicalPattern(
                    name="security_forensics",
                    template="We detected a potential security breach, help me analyze these forensic artifacts: {forensic_data}",
                    complexity=ComplexityLevel.COMPLEX,
                    tool_types=["security_tools", "logs", "monitoring_tools", "network_tools"],
                    output_types=["forensic_evidence", "attack_timeline", "compromise_indicators"],
                    example_output="Incident Timeline:\n09:15 - Suspicious login from foreign IP\n09:18 - Privilege escalation detected\n09:23 - Unusual file access patterns\n09:30 - Data exfiltration attempt blocked",
                    analysis_type="security_forensics"
                ),
                TechnicalPattern(
                    name="architecture_review",
                    template="Review this system architecture and identify potential issues based on these operational metrics: {arch_metrics}",
                    complexity=ComplexityLevel.COMPLEX,
                    tool_types=["monitoring_tools", "metrics", "network_tools", "database_tools"],
                    output_types=["architecture_analysis", "bottleneck_report", "scalability_assessment"],
                    example_output="Architecture Components:\nLoad Balancer → 3x App Servers → Database Cluster\nMetrics: 99th percentile latency increasing, DB CPU at 90%, connection pool saturation",
                    analysis_type="architecture_optimization"
                )
            ]
        }
        
        return patterns
    
    def _initialize_technical_concepts(self) -> Dict[str, Dict[str, Any]]:
        """Initialize technical concepts and their relationships"""
        return {
            "system_administration": {
                "keywords": ["process", "service", "daemon", "systemctl", "ps", "top", "memory", "cpu"],
                "related": ["performance", "monitoring", "troubleshooting"],
                "tools": ["htop", "systemctl", "ps", "netstat", "df", "du"]
            },
            "networking": {
                "keywords": ["tcp", "udp", "dns", "firewall", "port", "protocol", "routing", "latency"],
                "related": ["connectivity", "security", "performance"],
                "tools": ["ping", "traceroute", "netstat", "tcpdump", "iptables", "nslookup"]
            },
            "containers": {
                "keywords": ["docker", "kubernetes", "container", "image", "pod", "deployment", "orchestration"],
                "related": ["microservices", "scaling", "resource_management"],
                "tools": ["docker", "kubectl", "helm", "docker-compose"]
            },
            "databases": {
                "keywords": ["sql", "query", "index", "transaction", "lock", "replication", "backup"],
                "related": ["performance", "data_integrity", "scaling"],
                "tools": ["psql", "mysql", "mongosh", "redis-cli", "pg_dump", "explain"]
            },
            "monitoring": {
                "keywords": ["metrics", "alerts", "dashboard", "logs", "traces", "sla", "uptime"],
                "related": ["observability", "incident_response", "performance"],
                "tools": ["prometheus", "grafana", "elk", "datadog", "newrelic", "jaeger"]
            },
            "security": {
                "keywords": ["authentication", "authorization", "encryption", "vulnerability", "breach", "audit"],
                "related": ["compliance", "incident_response", "forensics"],
                "tools": ["nmap", "wireshark", "openssl", "fail2ban", "ossec", "nessus"]
            },
            "performance": {
                "keywords": ["latency", "throughput", "bottleneck", "optimization", "profiling", "benchmark"],
                "related": ["monitoring", "scaling", "tuning"],
                "tools": ["perf", "strace", "tcpdump", "iotop", "vmstat", "iostat"]
            },
            "automation": {
                "keywords": ["ci/cd", "pipeline", "deployment", "infrastructure", "configuration", "provisioning"],
                "related": ["devops", "reliability", "scaling"],
                "tools": ["jenkins", "github-actions", "ansible", "terraform", "chef", "puppet"]
            }
        }
    
    def _initialize_tool_outputs(self) -> Dict[str, List[str]]:
        """Initialize realistic tool output templates"""
        return {
            "cli_tools": [
                "Command '{command}' completed successfully",
                "Error: {command} failed with exit code {code}",
                "Usage: {command} [options] <args>",
                "No such file or directory: {path}",
                "Permission denied: {resource}"
            ],
            "monitoring_tools": [
                "CPU: {cpu_percent}% Memory: {memory_gb}GB Disk: {disk_percent}%",
                "Service {service_name} is {status}",
                "Alert: {metric} exceeded threshold ({value} > {threshold})",
                "Uptime: {days}d {hours}h {minutes}m",
                "Load average: {load1} {load5} {load15}"
            ],
            "logs": [
                "[{timestamp}] {level}: {message}",
                "{ip} - - [{timestamp}] \"{request}\" {status} {size}",
                "ERROR {timestamp} [{thread}] {logger}: {error_message}",
                "INFO Starting application on port {port}",
                "WARN Connection timeout after {timeout}ms"
            ],
            "network_tools": [
                "PING {host} ({ip}): 56 data bytes\n64 bytes from {ip}: icmp_seq=0 time={time}ms",
                "traceroute to {host} ({ip}), {max_hops} hops max",
                "Active Internet connections:\nProto Local Address Foreign Address State",
                "iptables: Chain INPUT (policy ACCEPT)\ntarget prot opt source destination",
                "DNS lookup for {domain}: {ip}"
            ],
            "database_tools": [
                "Query executed successfully. {rows} rows affected.",
                "ERROR: relation \"{table}\" does not exist",
                "Index Scan using {index} on {table} (cost={cost})",
                "Connection to database established",
                "Slow query detected: {duration}ms"
            ],
            "container_tools": [
                "Container {container_id} is running",
                "Error: Container exited with code {exit_code}",
                "Image {image}:{tag} pulled successfully",
                "Mounting volume {volume} to {mount_point}",
                "Port {host_port} mapped to container port {container_port}"
            ],
            "security_tools": [
                "Vulnerability detected: {cve} (severity: {severity})",
                "Authentication failed for user {user} from {ip}",
                "SSL certificate expires in {days} days",
                "Firewall rule added: allow {port}/{protocol}",
                "Intrusion detected from {ip} at {timestamp}"
            ]
        }
    
    def generate_queries(self, count: int, complexity_distribution: Optional[Dict[str, float]] = None) -> List[QueryRecord]:
        """
        Generate realistic technical domain queries
        
        Args:
            count: Number of queries to generate
            complexity_distribution: Distribution of complexity levels (default: balanced)
        
        Returns:
            List of QueryRecord objects with realistic technical queries
        """
        if complexity_distribution is None:
            complexity_distribution = {
                ComplexityLevel.SIMPLE: 0.5,
                ComplexityLevel.MEDIUM: 0.35, 
                ComplexityLevel.COMPLEX: 0.15
            }
        
        queries = []
        
        for i in range(count):
            # Deterministic complexity assignment
            complexity_rng = np.random.RandomState(self.seed + i)
            complexity = complexity_rng.choice(
                list(complexity_distribution.keys()),
                p=list(complexity_distribution.values())
            )
            
            # Generate query based on complexity
            query = self._generate_single_query(i, complexity)
            queries.append(query)
            
        self.logger.info(f"Generated {len(queries)} technical domain queries")
        return queries
    
    def _generate_single_query(self, query_index: int, complexity: ComplexityLevel) -> QueryRecord:
        """Generate a single realistic technical query"""
        
        # Create deterministic RNG for this query
        query_rng = np.random.RandomState(self.seed + query_index + 3000)
        
        # Select pattern for this complexity level
        patterns = self.technical_patterns[complexity]
        pattern = query_rng.choice(patterns)
        
        # Generate query text from pattern
        query_text = self._populate_query_template(pattern, query_rng)
        
        # Generate ground truth documents
        ground_truth_docs = self._generate_ground_truth_docs(
            pattern, query_text, query_rng, query_index
        )
        
        # Calculate quality score based on pattern and content
        quality_score = self._calculate_technical_quality(query_text, pattern, ground_truth_docs)
        
        # Create metadata
        metadata = QueryMetadata(
            creation_seed=self.seed + query_index + 3000,
            query_index=query_index,
            template_used=pattern.name,
            length_chars=len(query_text),
            complexity_score={"simple": 1, "medium": 2, "complex": 3}[str(complexity)],
            quality_score=quality_score,
            n_ground_truth_docs=len(ground_truth_docs),
            token_count=len(query_text.split()),
            has_code_blocks=bool(query_rng.random() > 0.4),  # 60% chance of technical output blocks
        )
        
        # Generate query record
        query_id = f"tool_results_query_{query_index:06d}"
        
        # Generate content hash to match schema validation
        hash_content = {
            "query_id": query_id,
            "query_text": query_text,
            "domain": DomainType.TOOL_RESULTS,
            "complexity": complexity,
            "ground_truth_docs": sorted([doc.doc_id for doc in ground_truth_docs])
        }
        content_json = json.dumps(hash_content, sort_keys=True, separators=(',', ':'), default=str)
        content_hash = hashlib.sha256(content_json.encode()).hexdigest()
        
        return QueryRecord(
            query_id=query_id,
            domain=DomainType.TOOL_RESULTS,
            complexity=complexity,
            session_id=f"tool_results_session_{query_index // 10:04d}",
            turn_index=query_rng.randint(1, 8),
            query_text=query_text,
            ground_truth_docs=ground_truth_docs,
            metadata=metadata,
            content_hash=content_hash,
            creation_timestamp=datetime.now(timezone.utc)
        )
    
    def _populate_query_template(self, pattern: TechnicalPattern, rng: np.random.RandomState) -> str:
        """Populate query template with realistic technical outputs"""
        
        template = pattern.template
        
        # Generate realistic technical output based on pattern
        output_placeholder = None
        for key in ["error_output", "status_output", "command_output", "config_snippet", 
                   "log_entry", "perf_data", "network_output", "security_log", "db_output",
                   "container_output", "multi_source_data", "capacity_data", "trace_data",
                   "forensic_data", "arch_metrics"]:
            if f"{{{key}}}" in template:
                output_placeholder = key
                break
        
        if output_placeholder:
            technical_output = self._generate_technical_output(pattern, output_placeholder, rng)
            template = template.replace(f"{{{output_placeholder}}}", technical_output)
        
        # Add complexity-based context
        if pattern.complexity == ComplexityLevel.COMPLEX:
            context_additions = [
                " This is affecting our production system and needs urgent investigation.",
                " We need a comprehensive analysis including root cause and remediation steps.",
                " Please provide detailed troubleshooting steps and prevention strategies.",
                " This appears to be part of a larger systemic issue we need to understand."
            ]
            template += rng.choice(context_additions)
        elif pattern.complexity == ComplexityLevel.MEDIUM:
            context_additions = [
                " This is impacting our service performance.",
                " We need to understand what's happening and how to fix it.",
                " Please provide actionable troubleshooting steps.",
                " What should we check to resolve this issue?"
            ]
            template += rng.choice(context_additions)
        
        return template
    
    def _generate_technical_output(
        self, 
        pattern: TechnicalPattern, 
        output_type: str,
        rng: np.random.RandomState
    ) -> str:
        """Generate realistic technical output for the query"""
        
        if output_type == "error_output":
            error_templates = [
                "ERROR: Connection refused: {host}:{port}",
                "FATAL: database connection failed: {error_detail}",
                "Exception in thread \"{thread}\": {exception_type}: {error_message}",
                "docker: Error response from daemon: {docker_error}",
                "nginx: [error] {pid}#{tid}: *{connection_id} {error_detail}",
                "SSH: Permission denied (publickey,password)",
                "OOM killer: Killed process {pid} ({process}) total-vm:{memory}kB",
                "SSL: certificate verify failed: self signed certificate",
            ]
            template = rng.choice(error_templates)
            
            # Fill template variables
            replacements = {
                "host": rng.choice(["localhost", "db.example.com", "api.service", "192.168.1.100"]),
                "port": rng.choice(["8080", "3306", "5432", "443", "22", "80"]),
                "error_detail": rng.choice(["timeout", "access denied", "invalid credentials", "resource not found"]),
                "thread": rng.choice(["main", "worker-1", "http-pool-2", "scheduler"]),
                "exception_type": rng.choice(["NullPointerException", "ConnectionException", "TimeoutException"]),
                "error_message": rng.choice(["null value encountered", "connection timeout", "invalid parameter"]),
                "docker_error": rng.choice(["port already allocated", "image not found", "insufficient memory"]),
                "pid": str(rng.randint(1000, 9999)),
                "tid": str(rng.randint(1, 99)),
                "connection_id": str(rng.randint(100000, 999999)),
                "process": rng.choice(["java", "python", "node", "postgres", "nginx"]),
                "memory": str(rng.randint(100000, 2000000))
            }
            
            for key, value in replacements.items():
                template = template.replace(f"{{{key}}}", value)
            
            return template
            
        elif output_type == "status_output":
            status_templates = [
                "● {service}.service - {description}\n   Loaded: loaded (/etc/systemd/system/{service}.service; enabled)\n   Active: {status} since {timestamp}",
                "NAME       READY   STATUS    RESTARTS   AGE\n{pod_name}   {ready}/{total}     {status}        {restarts}          {age}",
                "CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS          PORTS     NAMES\n{container_id}   {image}   {command}   {created}   {status}   {ports}   {name}",
                "Server Status: {status}\nUptime: {uptime}\nConnections: {connections}\nQueries per second: {qps}",
                "Health Check: {health_status}\nLast Check: {timestamp}\nResponse Time: {response_time}ms\nStatus Code: {status_code}"
            ]
            template = rng.choice(status_templates)
            
            replacements = {
                "service": rng.choice(["nginx", "postgres", "redis", "docker", "ssh"]),
                "description": rng.choice(["HTTP Server", "Database Server", "Cache Server", "Container Runtime"]),
                "status": rng.choice(["active (running)", "failed", "inactive (dead)", "activating"]),
                "timestamp": "Mon 2024-01-15 14:30:22 UTC",
                "pod_name": f"app-{rng.randint(1000, 9999)}-{rng.choice(['abc', 'def', 'xyz'])}",
                "ready": "1", "total": "1", "restarts": str(rng.randint(0, 5)), "age": "2d",
                "container_id": f"{rng.randint(100000000000, 999999999999):x}"[:12],
                "image": rng.choice(["nginx:latest", "postgres:13", "redis:6", "app:v1.2.0"]),
                "command": rng.choice(["\"/docker-entrypoint.sh\"", "\"redis-server\"", "\"python app.py\""]),
                "created": "2 hours ago", "ports": "0.0.0.0:80->80/tcp", "name": "app_container",
                "health_status": rng.choice(["Healthy", "Unhealthy", "Starting"]),
                "uptime": f"{rng.randint(1, 100)} days",
                "connections": str(rng.randint(10, 1000)),
                "qps": str(rng.randint(100, 5000)),
                "response_time": str(rng.randint(50, 500)),
                "status_code": rng.choice(["200", "503", "404"])
            }
            
            for key, value in replacements.items():
                template = template.replace(f"{{{key}}}", value)
            
            return template
            
        elif output_type == "perf_data":
            perf_templates = [
                "CPU Usage: {cpu}%\nMemory: {memory_used}GB / {memory_total}GB ({memory_percent}%)\nDisk I/O: Read {read_iops} IOPS, Write {write_iops} IOPS\nNetwork: In {net_in}Mbps, Out {net_out}Mbps",
                "Response Times (ms):\n  Average: {avg_response}\n  P95: {p95_response}\n  P99: {p99_response}\nThroughput: {throughput} requests/second\nError Rate: {error_rate}%",
                "Database Performance:\n  Active Connections: {active_conn} / {max_conn}\n  Slow Queries: {slow_queries}\n  Lock Waits: {lock_waits}ms\n  Cache Hit Ratio: {cache_ratio}%",
                "Load Average: {load1} {load5} {load15}\nProcesses: {processes} total, {running} running\nMemory Usage: {mem_usage}%\nSwap Usage: {swap_usage}%"
            ]
            template = rng.choice(perf_templates)
            
            replacements = {
                "cpu": str(rng.randint(20, 95)),
                "memory_used": f"{rng.randint(1, 8)}.{rng.randint(0, 9)}",
                "memory_total": "8.0",
                "memory_percent": str(rng.randint(40, 90)),
                "read_iops": str(rng.randint(100, 1000)),
                "write_iops": str(rng.randint(50, 500)),
                "net_in": str(rng.randint(10, 100)),
                "net_out": str(rng.randint(5, 50)),
                "avg_response": str(rng.randint(100, 500)),
                "p95_response": str(rng.randint(200, 800)),
                "p99_response": str(rng.randint(500, 1500)),
                "throughput": str(rng.randint(50, 1000)),
                "error_rate": f"{rng.randint(0, 10)}.{rng.randint(0, 9)}",
                "active_conn": str(rng.randint(10, 80)),
                "max_conn": "100",
                "slow_queries": str(rng.randint(5, 50)),
                "lock_waits": str(rng.randint(10, 200)),
                "cache_ratio": str(rng.randint(70, 95)),
                "load1": f"{rng.randint(1, 5)}.{rng.randint(0, 99):02d}",
                "load5": f"{rng.randint(1, 4)}.{rng.randint(0, 99):02d}",
                "load15": f"{rng.randint(1, 3)}.{rng.randint(0, 99):02d}",
                "processes": str(rng.randint(150, 300)),
                "running": str(rng.randint(1, 10)),
                "mem_usage": str(rng.randint(60, 85)),
                "swap_usage": str(rng.randint(0, 20))
            }
            
            for key, value in replacements.items():
                template = template.replace(f"{{{key}}}", value)
            
            return template
            
        else:
            # Generate simple output for other types
            return pattern.example_output
    
    def _generate_ground_truth_docs(
        self, 
        pattern: TechnicalPattern, 
        query_text: str, 
        rng: np.random.RandomState,
        query_index: int
    ) -> List[GroundTruthDocument]:
        """Generate realistic ground truth technical documentation"""
        
        # Determine number of documents based on complexity
        n_docs = {
            ComplexityLevel.SIMPLE: rng.randint(2, 4),
            ComplexityLevel.MEDIUM: rng.randint(3, 6),
            ComplexityLevel.COMPLEX: rng.randint(5, 8)
        }[pattern.complexity]
        
        docs = []
        
        for doc_idx in range(n_docs):
            doc_id = f"tool_results_doc_{query_index:06d}_{doc_idx:02d}"
            
            # Generate document content based on analysis type
            doc_type = rng.choice(["explanation", "troubleshooting", "reference", "analysis"])
            content = self._generate_technical_doc_content(pattern, doc_type, rng)
            
            # Calculate relevance score
            relevance_score = 0.6 + rng.random() * 0.4  # Between 0.6 and 1.0
            
            # Create content hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            doc = GroundTruthDocument(
                doc_id=doc_id,
                content=content,
                relevance_score=relevance_score,
                doc_type=doc_type,
                content_hash=content_hash,
                metadata={
                    "pattern_name": pattern.name,
                    "tool_types": pattern.tool_types,
                    "analysis_type": pattern.analysis_type
                }
            )
            docs.append(doc)
        
        return docs
    
    def _generate_technical_doc_content(
        self, 
        pattern: TechnicalPattern, 
        doc_type: str, 
        rng: np.random.RandomState
    ) -> str:
        """Generate realistic technical documentation content"""
        
        if doc_type == "explanation":
            return f"""# Technical Analysis: {pattern.name.replace('_', ' ').title()}

## What This Output Means
This {pattern.analysis_type.replace('_', ' ')} indicates a specific condition in your system.

## Key Indicators
- **Status**: The system is reporting {rng.choice(['normal operation', 'warning condition', 'error state', 'performance issue'])}
- **Impact**: {rng.choice(['Low', 'Medium', 'High'])} impact on system functionality
- **Urgency**: {rng.choice(['Monitor', 'Investigate', 'Immediate action required'])}

## Technical Details
The output shows {rng.choice(['resource utilization', 'connection status', 'error conditions', 'performance metrics'])} 
that {rng.choice(['meets expectations', 'exceeds normal thresholds', 'indicates potential issues'])}.

## Next Steps
1. {rng.choice(['Monitor for trends', 'Check related systems', 'Review configuration', 'Investigate root cause'])}
2. {rng.choice(['Verify system health', 'Analyze logs', 'Test connectivity', 'Review metrics'])}
3. {rng.choice(['Document findings', 'Implement fix', 'Schedule maintenance', 'Alert stakeholders'])}

## Related Commands
```bash
# Check system status
{rng.choice(['systemctl status service', 'docker ps', 'kubectl get pods', 'ps aux | grep process'])}

# View detailed information  
{rng.choice(['journalctl -u service', 'docker logs container', 'kubectl describe pod', 'top -p PID'])}
```
"""
        
        elif doc_type == "troubleshooting":
            return f"""# Troubleshooting Guide: {pattern.analysis_type.replace('_', ' ').title()}

## Problem Description
This issue typically occurs when {rng.choice(['system resources are constrained', 'network connectivity problems exist', 'configuration is incorrect', 'dependencies are missing'])}.

## Diagnostic Steps

### Step 1: Initial Assessment
Check the current system state:
```bash
# System health check
{rng.choice(['systemctl --failed', 'docker system df', 'free -h', 'df -h'])}

# Process information
{rng.choice(['ps aux --sort=-cpu', 'top -bn1', 'htop', 'iostat'])}
```

### Step 2: Detailed Investigation
Gather more information:
```bash
# Log analysis
{rng.choice(['journalctl -xe', 'tail -f /var/log/syslog', 'dmesg | tail', 'docker logs --tail 100 container'])}

# Network connectivity
{rng.choice(['ping google.com', 'netstat -tulpn', 'ss -tulpn', 'tcpdump -i eth0'])}
```

### Step 3: Common Solutions
Try these remediation steps:

**Option 1: Restart Services**
```bash
{rng.choice(['systemctl restart service', 'docker restart container', 'kubectl rollout restart deployment', 'service nginx reload'])}
```

**Option 2: Resource Management**
```bash
{rng.choice(['kill -9 PID', 'docker system prune', 'apt autoremove', 'yum clean all'])}
```

**Option 3: Configuration Fix**
- Review configuration files
- Validate syntax and parameters
- Compare with working configurations
- Apply security best practices

## Prevention Strategies
- Implement monitoring and alerting
- Set up automated health checks
- Establish backup and recovery procedures
- Regular system maintenance schedules

## When to Escalate
Contact system administrators if:
- Issue persists after following these steps
- System performance continues to degrade
- Security implications are suspected
- Business-critical services are affected
"""
        
        elif doc_type == "reference":
            return f"""# Technical Reference: {pattern.analysis_type.replace('_', ' ').title()}

## Command Reference
Common commands for this analysis type:

### Monitoring Commands
```bash
# System monitoring
{rng.choice(['htop', 'iotop', 'nethogs', 'vmstat 1'])}

# Service status
{rng.choice(['systemctl status', 'docker stats', 'kubectl top', 'supervisorctl status'])}

# Log monitoring
{rng.choice(['tail -f /var/log/messages', 'journalctl -f', 'docker logs -f', 'kubectl logs -f'])}
```

### Configuration Files
Key configuration locations:
- System: `/etc/systemd/system/`, `/etc/init.d/`
- Docker: `/etc/docker/daemon.json`, `docker-compose.yml`
- Network: `/etc/network/interfaces`, `/etc/hosts`
- Logs: `/var/log/`, `/etc/rsyslog.conf`

### Environment Variables
Important environment settings:
- `PATH`: Executable search paths
- `LD_LIBRARY_PATH`: Library search paths  
- `HOME`: User home directory
- `TMPDIR`: Temporary file location

### Exit Codes
Common exit codes and meanings:
- 0: Success
- 1: General error
- 2: Misuse of shell command
- 126: Command not executable
- 127: Command not found
- 128+n: Fatal error signal "n"

### Performance Metrics
Key metrics to monitor:
- CPU utilization (%)
- Memory usage (MB/GB)
- Disk I/O (IOPS, MB/s)
- Network throughput (Mbps)
- Load average (1m, 5m, 15m)
- Response time (ms)

### Best Practices
1. Always backup before making changes
2. Test in non-production environments first
3. Monitor system metrics continuously
4. Document all configuration changes
5. Use version control for configurations
6. Implement proper logging and monitoring
7. Follow security hardening guidelines
"""
        
        else:  # analysis
            return f"""# System Analysis Report: {pattern.analysis_type.replace('_', ' ').title()}

## Executive Summary
Analysis of system output reveals {rng.choice(['normal operation', 'performance concerns', 'error conditions', 'optimization opportunities'])}.

## Findings

### Primary Observations
1. **System State**: {rng.choice(['Stable', 'Degraded', 'Critical', 'Recovering'])}
2. **Performance**: {rng.choice(['Within normal parameters', 'Below expectations', 'Exceeding capacity', 'Requires optimization'])}
3. **Resource Utilization**: {rng.choice(['Low', 'Moderate', 'High', 'Critical'])} usage detected

### Detailed Analysis
The technical output indicates:

**Positive Indicators:**
- {rng.choice(['Service is responding normally', 'Resource usage is stable', 'No critical errors detected', 'Performance metrics within SLA'])}
- {rng.choice(['Log entries show normal operation', 'Network connectivity is healthy', 'Database performance is adequate', 'Security status is good'])}

**Areas of Concern:**
- {rng.choice(['Memory usage trending upward', 'Response times increasing', 'Error rate above baseline', 'Resource exhaustion possible'])}
- {rng.choice(['Configuration drift detected', 'Dependencies may be outdated', 'Capacity limits approaching', 'Security vulnerabilities present'])}

## Risk Assessment
**Risk Level**: {rng.choice(['Low', 'Medium', 'High', 'Critical'])}

**Impact Analysis:**
- Business continuity: {rng.choice(['No impact', 'Minor disruption', 'Service degradation', 'System unavailable'])}
- Data integrity: {rng.choice(['Secure', 'At risk', 'Compromised', 'Unknown'])}
- Performance: {rng.choice(['Optimal', 'Acceptable', 'Degraded', 'Unacceptable'])}

## Recommendations

### Immediate Actions
1. {rng.choice(['Monitor system closely', 'Investigate error conditions', 'Scale resources', 'Implement fixes'])}
2. {rng.choice(['Review configuration', 'Update dependencies', 'Patch vulnerabilities', 'Optimize performance'])}
3. {rng.choice(['Alert stakeholders', 'Document findings', 'Schedule maintenance', 'Plan capacity upgrades'])}

### Long-term Strategy
- Implement proactive monitoring
- Establish automated remediation
- Regular system audits and updates
- Capacity planning and scaling strategy
- Disaster recovery planning

## Metrics and KPIs
Track these key performance indicators:
- System availability: Target 99.9%
- Response time: Target <200ms
- Error rate: Target <1%
- Resource utilization: Target <80%

## Follow-up Actions
Schedule review in {rng.choice(['24 hours', '1 week', '1 month'])} to assess:
- Implementation of recommendations
- System performance improvements  
- Recurring issue patterns
- Additional optimization opportunities
"""
    
    def _calculate_technical_quality(
        self, 
        query_text: str, 
        pattern: TechnicalPattern, 
        ground_truth_docs: List[GroundTruthDocument]
    ) -> float:
        """Calculate quality score for technical domain query"""
        
        score = 0.0
        
        # Length appropriateness (varies by complexity)
        length = len(query_text)
        if pattern.complexity == ComplexityLevel.SIMPLE and 30 <= length <= 200:
            score += 0.25
        elif pattern.complexity == ComplexityLevel.MEDIUM and 50 <= length <= 400:
            score += 0.25  
        elif pattern.complexity == ComplexityLevel.COMPLEX and 100 <= length <= 600:
            score += 0.25
        else:
            score += 0.1
        
        # Complexity appropriateness
        complexity_scores = {
            ComplexityLevel.SIMPLE: 0.25,
            ComplexityLevel.MEDIUM: 0.25,
            ComplexityLevel.COMPLEX: 0.2
        }
        score += complexity_scores[pattern.complexity]
        
        # Technical specificity
        technical_keywords = [
            "error", "output", "log", "status", "analyze", "troubleshoot", "debug",
            "performance", "metrics", "trace", "monitor", "system", "service"
        ]
        if any(keyword in query_text.lower() for keyword in technical_keywords):
            score += 0.2
        
        # Technical output presence (looking for technical artifacts)
        technical_artifacts = [
            ":", "=", "[", "]", "{", "}", "%", "#", "$", ">", "<", 
            "error", "failed", "success", "status", "pid", "port"
        ]
        artifact_count = sum(1 for artifact in technical_artifacts if artifact in query_text)
        if artifact_count >= 3:
            score += 0.15
        elif artifact_count >= 1:
            score += 0.1
        
        # Ground truth appropriateness 
        n_docs = len(ground_truth_docs)
        if pattern.complexity == ComplexityLevel.SIMPLE and 2 <= n_docs <= 4:
            score += 0.1
        elif pattern.complexity == ComplexityLevel.MEDIUM and 3 <= n_docs <= 6:
            score += 0.1
        elif pattern.complexity == ComplexityLevel.COMPLEX and 5 <= n_docs <= 8:
            score += 0.1
        else:
            score += 0.05
        
        # Analysis type appropriateness
        analysis_bonus = {
            "error_diagnosis": 0.05,
            "performance_diagnosis": 0.05,
            "incident_investigation": 0.1,
            "security_analysis": 0.05
        }
        score += analysis_bonus.get(pattern.analysis_type, 0.0)
        
        return min(1.0, max(0.0, score))
    
    def validate_queries(self, queries: List[QueryRecord]) -> ValidationResult:
        """Validate generated technical domain queries"""
        
        errors = []
        warnings = []
        query_errors = {}
        
        # Check domain consistency
        non_technical_queries = [q for q in queries if q.domain != DomainType.TOOL_RESULTS]
        if non_technical_queries:
            errors.append(f"Found {len(non_technical_queries)} non-technical queries")
        
        # Check for technical content patterns
        technical_patterns = [
            r'\b(error|output|log|status|analyze|debug|troubleshoot)\b',
            r'[\[\]{}():%$#><=]',  # Technical characters
            r'\b(cpu|memory|disk|network|database|container|service)\b'
        ]
        
        for query in queries:
            query_technical_score = sum(
                1 for pattern in technical_patterns
                if __import__('re').search(pattern, query.query_text.lower())
            )
            
            if query_technical_score == 0:
                query_errors[query.query_id] = ["Query lacks technical output content"]
                
        # Check analysis type distribution
        analysis_types = {}
        for query in queries:
            analysis_type = query.metadata.template_used
            analysis_types[analysis_type] = analysis_types.get(analysis_type, 0) + 1
        
        # Check quality scores
        if queries:
            avg_quality = sum(q.metadata.quality_score for q in queries) / len(queries)
            if avg_quality < 0.7:
                warnings.append(f"Average quality score below threshold: {avg_quality:.3f}")
        
        # Check for realistic technical artifacts
        artifact_queries = [
            q for q in queries 
            if any(char in q.query_text for char in [":", "[", "{", "%", "#", "$"])
        ]
        
        if len(artifact_queries) / len(queries) < 0.5 if queries else 0:
            warnings.append("Low ratio of queries with technical artifacts")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            query_errors=query_errors,
            statistics={
                "total_queries": len(queries),
                "avg_quality_score": avg_quality if queries else 0,
                "technical_content_ratio": (len(queries) - len(query_errors)) / len(queries) if queries else 0,
                "analysis_type_diversity": len(analysis_types),
                "technical_artifact_ratio": len(artifact_queries) / len(queries) if queries else 0
            },
            recommendations=[
                "Ensure all queries contain technical output or system artifacts", 
                "Balance analysis types across error diagnosis, performance, and troubleshooting",
                "Include diverse technical domains (networking, databases, containers, etc.)",
                "Add realistic technical output formats and error messages"
            ]
        )