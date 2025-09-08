#!/usr/bin/env python3
"""
Documentation Domain Builder for LetheBench
==========================================

Specialized builder for generating high-quality documentation-focused queries that
represent realistic scenarios where users seek information from technical documentation,
API references, user guides, and explanatory content.

Features:
- Realistic documentation query patterns
- Multi-format coverage (API docs, tutorials, references, guides)
- Diverse information-seeking behaviors
- Ground truth documentation excerpts and explanations
- Cross-reference capabilities between topics
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
class DocumentationPattern:
    """Documentation pattern for generating realistic queries"""
    name: str
    template: str
    complexity: ComplexityLevel
    doc_types: List[str]
    topics: List[str]
    example_content: str
    information_need: str


class DocumentationDomainBuilder:
    """
    Builder for documentation-focused domain queries with realistic information-seeking scenarios
    """
    
    def __init__(self, seed: int = 42, logger: Optional[logging.Logger] = None):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize documentation types
        self.doc_types = [
            "API_reference", "user_guide", "tutorial", "FAQ", "changelog",
            "installation_guide", "troubleshooting", "best_practices", "examples"
        ]
        
        # Initialize documentation patterns by complexity
        self.doc_patterns = self._initialize_doc_patterns()
        
        # Documentation topics and their relationships
        self.doc_topics = self._initialize_doc_topics()
        
        # Question types and information needs
        self.question_types = self._initialize_question_types()
        
    def _initialize_doc_patterns(self) -> Dict[ComplexityLevel, List[DocumentationPattern]]:
        """Initialize comprehensive documentation patterns for query generation"""
        
        patterns = {
            ComplexityLevel.SIMPLE: [
                DocumentationPattern(
                    name="basic_how_to",
                    template="How do I {action} using {tool_or_service}?",
                    complexity=ComplexityLevel.SIMPLE,
                    doc_types=["user_guide", "tutorial", "FAQ"],
                    topics=["getting_started", "basic_usage", "configuration"],
                    example_content="## Getting Started\n\nTo get started with X, follow these simple steps:\n1. Install the package\n2. Configure your settings\n3. Run your first command",
                    information_need="procedural_knowledge"
                ),
                DocumentationPattern(
                    name="definition_lookup",
                    template="What is {concept} in {context}?",
                    complexity=ComplexityLevel.SIMPLE,
                    doc_types=["API_reference", "user_guide", "FAQ"],
                    topics=["definitions", "concepts", "terminology"],
                    example_content="**{concept}**: A fundamental component that enables...",
                    information_need="factual_knowledge"
                ),
                DocumentationPattern(
                    name="feature_availability",
                    template="Does {tool_or_service} support {feature}?",
                    complexity=ComplexityLevel.SIMPLE,
                    doc_types=["user_guide", "FAQ", "changelog"],
                    topics=["features", "capabilities", "limitations"],
                    example_content="## Supported Features\n- Feature A: ✅ Available\n- Feature B: ❌ Not supported\n- Feature C: ⚠️ Experimental",
                    information_need="factual_knowledge"
                ),
                DocumentationPattern(
                    name="quick_reference",
                    template="What are the {item_type} for {tool_or_service}?",
                    complexity=ComplexityLevel.SIMPLE,
                    doc_types=["API_reference", "user_guide", "examples"],
                    topics=["commands", "parameters", "options"],
                    example_content="## Quick Reference\n\n### Basic Commands\n- `command1`: Brief description\n- `command2`: Brief description",
                    information_need="reference_lookup"
                ),
                DocumentationPattern(
                    name="installation_help",
                    template="How do I install {tool_or_service} on {platform}?",
                    complexity=ComplexityLevel.SIMPLE,
                    doc_types=["installation_guide", "user_guide", "FAQ"],
                    topics=["installation", "setup", "requirements"],
                    example_content="## Installation on {platform}\n\n1. Download the installer\n2. Run the installation command\n3. Verify the installation",
                    information_need="procedural_knowledge"
                )
            ],
            
            ComplexityLevel.MEDIUM: [
                DocumentationPattern(
                    name="configuration_guide",
                    template="How do I configure {tool_or_service} for {use_case} with {requirements}?",
                    complexity=ComplexityLevel.MEDIUM,
                    doc_types=["user_guide", "best_practices", "examples"],
                    topics=["configuration", "customization", "optimization"],
                    example_content="## Configuration for {use_case}\n\n### Requirements\n- Requirement A\n- Requirement B\n\n### Configuration Steps\n1. Modify config file\n2. Set environment variables\n3. Restart service",
                    information_need="procedural_knowledge"
                ),
                DocumentationPattern(
                    name="troubleshooting_guide",
                    template="I'm getting {error_symptom} when {action}. How do I fix this?",
                    complexity=ComplexityLevel.MEDIUM,
                    doc_types=["troubleshooting", "FAQ", "user_guide"],
                    topics=["error_resolution", "debugging", "common_issues"],
                    example_content="## Troubleshooting: {error_symptom}\n\n### Symptoms\n- Error message appears\n- System behavior changes\n\n### Solutions\n1. Check configuration\n2. Verify dependencies\n3. Restart service",
                    information_need="problem_solving"
                ),
                DocumentationPattern(
                    name="feature_comparison",
                    template="What's the difference between {option_a} and {option_b} in {tool_or_service}?",
                    complexity=ComplexityLevel.MEDIUM,
                    doc_types=["user_guide", "best_practices", "API_reference"],
                    topics=["comparisons", "alternatives", "decision_making"],
                    example_content="## {option_a} vs {option_b}\n\n| Aspect | {option_a} | {option_b} |\n|--------|------------|------------|\n| Performance | High | Medium |\n| Complexity | Low | High |",
                    information_need="comparative_analysis"
                ),
                DocumentationPattern(
                    name="integration_guide",
                    template="How do I integrate {tool_or_service} with {external_system} for {purpose}?",
                    complexity=ComplexityLevel.MEDIUM,
                    doc_types=["user_guide", "examples", "API_reference"],
                    topics=["integration", "connectivity", "interoperability"],
                    example_content="## Integration with {external_system}\n\n### Prerequisites\n- Access credentials\n- Network connectivity\n\n### Integration Steps\n1. Configure authentication\n2. Establish connection\n3. Test integration",
                    information_need="procedural_knowledge"
                ),
                DocumentationPattern(
                    name="best_practices",
                    template="What are the best practices for {activity} with {tool_or_service}?",
                    complexity=ComplexityLevel.MEDIUM,
                    doc_types=["best_practices", "user_guide", "examples"],
                    topics=["optimization", "performance", "security", "maintainability"],
                    example_content="## Best Practices for {activity}\n\n### Performance\n- Optimize configurations\n- Use caching appropriately\n\n### Security\n- Enable authentication\n- Use secure connections\n\n### Maintenance\n- Regular backups\n- Monitor logs",
                    information_need="strategic_guidance"
                )
            ],
            
            ComplexityLevel.COMPLEX: [
                DocumentationPattern(
                    name="architecture_design",
                    template="How should I architect a {system_type} solution using {tool_or_service} that handles {requirements}?",
                    complexity=ComplexityLevel.COMPLEX,
                    doc_types=["best_practices", "user_guide", "examples"],
                    topics=["system_design", "architecture", "scalability", "reliability"],
                    example_content="## Architecture Design for {system_type}\n\n### Design Principles\n- Scalability considerations\n- Fault tolerance\n- Performance optimization\n\n### Component Architecture\n```\n[Component A] --> [Component B]\n       ↓              ↓\n[Component C] <-- [Component D]\n```\n\n### Implementation Guidelines\n1. Define service boundaries\n2. Implement monitoring\n3. Plan for disaster recovery",
                    information_need="strategic_guidance"
                ),
                DocumentationPattern(
                    name="advanced_customization",
                    template="How can I customize {tool_or_service} to {complex_requirement} while maintaining {constraint}?",
                    complexity=ComplexityLevel.COMPLEX,
                    doc_types=["user_guide", "best_practices", "API_reference"],
                    topics=["advanced_configuration", "extensibility", "customization"],
                    example_content="## Advanced Customization\n\n### Custom Implementation\n```yaml\nconfiguration:\n  custom_module:\n    enabled: true\n    parameters:\n      setting_a: value_a\n      setting_b: value_b\n```\n\n### Considerations\n- Performance impact\n- Maintenance overhead\n- Compatibility with updates\n\n### Testing Strategy\n1. Unit testing of custom components\n2. Integration testing\n3. Performance benchmarking",
                    information_need="expert_implementation"
                ),
                DocumentationPattern(
                    name="migration_strategy",
                    template="How do I migrate from {old_system} to {tool_or_service} with minimal {impact_area}?",
                    complexity=ComplexityLevel.COMPLEX,
                    doc_types=["user_guide", "best_practices", "examples"],
                    topics=["migration", "data_transfer", "compatibility", "rollback"],
                    example_content="## Migration Strategy: {old_system} → {tool_or_service}\n\n### Pre-Migration\n1. Data audit and cleanup\n2. Compatibility assessment\n3. Rollback plan development\n\n### Migration Phases\n**Phase 1: Setup**\n- Install new system\n- Configure basic settings\n\n**Phase 2: Data Migration**\n- Export data from old system\n- Transform data format\n- Import to new system\n\n**Phase 3: Validation**\n- Data integrity checks\n- Functionality testing\n- Performance validation\n\n### Post-Migration\n- Monitor system performance\n- Train users on new system\n- Decommission old system",
                    information_need="strategic_guidance"
                ),
                DocumentationPattern(
                    name="performance_optimization",
                    template="How do I optimize {tool_or_service} performance for {workload_type} while handling {scale_requirement}?",
                    complexity=ComplexityLevel.COMPLEX,
                    doc_types=["best_practices", "troubleshooting", "user_guide"],
                    topics=["performance_tuning", "scaling", "monitoring", "optimization"],
                    example_content="## Performance Optimization for {workload_type}\n\n### Performance Analysis\n1. Baseline measurement\n2. Bottleneck identification\n3. Resource utilization analysis\n\n### Optimization Strategies\n**Configuration Tuning**\n- Memory allocation: 8GB minimum\n- Connection pooling: 100 connections\n- Cache size: 2GB\n\n**Resource Scaling**\n- Horizontal: Add more instances\n- Vertical: Increase instance size\n\n**Monitoring Setup**\n```yaml\nmetrics:\n  - cpu_utilization\n  - memory_usage\n  - request_latency\n  - error_rate\n```\n\n### Performance Targets\n- Response time: <100ms p95\n- Throughput: >1000 req/sec\n- Availability: 99.9%",
                    information_need="expert_implementation"
                ),
                DocumentationPattern(
                    name="security_implementation",
                    template="How do I implement {security_requirement} in {tool_or_service} that complies with {standards}?",
                    complexity=ComplexityLevel.COMPLEX,
                    doc_types=["best_practices", "user_guide", "API_reference"],
                    topics=["security", "compliance", "authentication", "encryption"],
                    example_content="## Security Implementation: {security_requirement}\n\n### Compliance Requirements\n- {standards} compliance\n- Audit trail maintenance\n- Regular security reviews\n\n### Implementation Components\n**Authentication Layer**\n- Multi-factor authentication\n- Token-based access\n- Session management\n\n**Authorization Framework**\n- Role-based access control\n- Permission matrices\n- Resource-level security\n\n**Data Protection**\n- Encryption at rest (AES-256)\n- Encryption in transit (TLS 1.3)\n- Key management policies\n\n### Security Monitoring\n```yaml\nsecurity_events:\n  - failed_login_attempts\n  - privilege_escalation\n  - data_access_violations\n  - configuration_changes\n```\n\n### Incident Response\n1. Automated threat detection\n2. Alert notification system\n3. Response procedures\n4. Recovery protocols",
                    information_need="expert_implementation"
                )
            ]
        }
        
        return patterns
    
    def _initialize_doc_topics(self) -> Dict[str, Dict[str, Any]]:
        """Initialize documentation topics and their relationships"""
        return {
            "getting_started": {
                "keywords": ["install", "setup", "configure", "initialize", "quickstart"],
                "related": ["installation", "configuration", "basic_usage"],
                "doc_types": ["installation_guide", "user_guide", "tutorial"]
            },
            "configuration": {
                "keywords": ["config", "settings", "parameters", "options", "customize"],
                "related": ["installation", "optimization", "troubleshooting"],
                "doc_types": ["user_guide", "best_practices", "API_reference"]
            },
            "troubleshooting": {
                "keywords": ["error", "issue", "problem", "bug", "fix", "debug"],
                "related": ["error_resolution", "common_issues", "debugging"],
                "doc_types": ["troubleshooting", "FAQ", "user_guide"]
            },
            "integration": {
                "keywords": ["connect", "integrate", "API", "webhook", "plugin", "extension"],
                "related": ["connectivity", "interoperability", "API_usage"],
                "doc_types": ["API_reference", "user_guide", "examples"]
            },
            "performance": {
                "keywords": ["optimize", "performance", "speed", "scale", "tuning"],
                "related": ["optimization", "scaling", "monitoring", "benchmarking"],
                "doc_types": ["best_practices", "troubleshooting", "user_guide"]
            },
            "security": {
                "keywords": ["secure", "authentication", "authorization", "permission", "encrypt"],
                "related": ["compliance", "privacy", "access_control"],
                "doc_types": ["best_practices", "user_guide", "API_reference"]
            },
            "migration": {
                "keywords": ["migrate", "upgrade", "transfer", "import", "export"],
                "related": ["data_transfer", "compatibility", "version_upgrade"],
                "doc_types": ["user_guide", "best_practices", "examples"]
            },
            "monitoring": {
                "keywords": ["monitor", "logs", "metrics", "alerts", "dashboard"],
                "related": ["observability", "debugging", "performance"],
                "doc_types": ["user_guide", "best_practices", "troubleshooting"]
            }
        }
    
    def _initialize_question_types(self) -> Dict[str, List[str]]:
        """Initialize question types by information need"""
        return {
            "factual_knowledge": [
                "What is", "What does", "Does it support", "Is it possible to",
                "Can I", "Which", "Where"
            ],
            "procedural_knowledge": [
                "How do I", "How can I", "What steps", "Walk me through",
                "Show me how", "Guide me", "Help me"
            ],
            "problem_solving": [
                "I'm getting an error", "This isn't working", "How do I fix",
                "Troubleshoot", "Debug", "Resolve", "Why is"
            ],
            "comparative_analysis": [
                "What's the difference", "Compare", "Which is better",
                "Pros and cons", "Should I use", "Alternative"
            ],
            "strategic_guidance": [
                "Best practices for", "Recommended approach", "How should I",
                "What's the strategy", "Design patterns", "Architecture"
            ],
            "reference_lookup": [
                "List of", "Available options", "Parameters for", "Commands",
                "Reference", "Documentation", "Syntax"
            ]
        }
    
    def generate_queries(self, count: int, complexity_distribution: Optional[Dict[str, float]] = None) -> List[QueryRecord]:
        """
        Generate realistic documentation domain queries
        
        Args:
            count: Number of queries to generate
            complexity_distribution: Distribution of complexity levels (default: balanced)
        
        Returns:
            List of QueryRecord objects with realistic documentation queries
        """
        if complexity_distribution is None:
            complexity_distribution = {
                ComplexityLevel.SIMPLE: 0.4,
                ComplexityLevel.MEDIUM: 0.45, 
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
            
        self.logger.info(f"Generated {len(queries)} documentation domain queries")
        return queries
    
    def _generate_single_query(self, query_index: int, complexity: ComplexityLevel) -> QueryRecord:
        """Generate a single realistic documentation query"""
        
        # Create deterministic RNG for this query
        query_rng = np.random.RandomState(self.seed + query_index + 2000)
        
        # Select pattern for this complexity level
        patterns = self.doc_patterns[complexity]
        pattern = query_rng.choice(patterns)
        
        # Generate query text from pattern
        query_text = self._populate_query_template(pattern, query_rng)
        
        # Generate ground truth documents
        ground_truth_docs = self._generate_ground_truth_docs(
            pattern, query_text, query_rng, query_index
        )
        
        # Calculate quality score based on pattern and content
        quality_score = self._calculate_docs_quality(query_text, pattern, ground_truth_docs)
        
        # Create metadata
        metadata = QueryMetadata(
            creation_seed=self.seed + query_index + 2000,
            query_index=query_index,
            template_used=pattern.name,
            length_chars=len(query_text),
            complexity_score={"simple": 1, "medium": 2, "complex": 3}[str(complexity)],
            quality_score=quality_score,
            n_ground_truth_docs=len(ground_truth_docs),
            token_count=len(query_text.split()),
            language_detected="english",
            has_code_blocks=False
        )
        
        # Generate query record
        query_id = f"chatty_prose_query_{query_index:06d}"
        
        # Generate content hash to match schema validation
        hash_content = {
            "query_id": query_id,
            "query_text": query_text,
            "domain": DomainType.CHATTY_PROSE,
            "complexity": complexity,
            "ground_truth_docs": sorted([doc.doc_id for doc in ground_truth_docs])
        }
        content_json = json.dumps(hash_content, sort_keys=True, separators=(',', ':'), default=str)
        content_hash = hashlib.sha256(content_json.encode()).hexdigest()
        
        return QueryRecord(
            query_id=query_id,
            domain=DomainType.CHATTY_PROSE,
            complexity=complexity,
            session_id=f"chatty_prose_session_{query_index // 10:04d}",
            turn_index=query_rng.randint(1, 8),
            query_text=query_text,
            ground_truth_docs=ground_truth_docs,
            metadata=metadata,
            content_hash=content_hash,
            creation_timestamp=datetime.now(timezone.utc)
        )
    
    def _populate_query_template(self, pattern: DocumentationPattern, rng: np.random.RandomState) -> str:
        """Populate query template with realistic values"""
        
        template = pattern.template
        
        # Pattern-specific replacements
        replacements = {
            "action": ["set up", "configure", "install", "use", "enable", "troubleshoot", "optimize"],
            "tool_or_service": ["Docker", "Kubernetes", "Jenkins", "Git", "AWS S3", "PostgreSQL", 
                               "Redis", "Nginx", "React", "Django", "Elasticsearch", "Terraform"],
            "concept": ["container orchestration", "CI/CD pipeline", "microservices", "load balancing",
                       "database indexing", "caching strategy", "authentication", "monitoring"],
            "context": ["cloud computing", "web development", "data engineering", "DevOps",
                       "machine learning", "system administration", "database management"],
            "feature": ["auto-scaling", "backup and restore", "real-time monitoring", "SSO integration",
                       "multi-region deployment", "data encryption", "API versioning", "caching"],
            "item_type": ["commands", "configuration options", "API endpoints", "environment variables",
                         "parameters", "supported formats", "error codes", "status codes"],
            "platform": ["Ubuntu", "CentOS", "macOS", "Windows", "Docker", "Kubernetes", "AWS", "Azure"],
            "use_case": ["production deployment", "development environment", "testing setup", "staging",
                        "high availability", "disaster recovery", "data migration", "integration testing"],
            "requirements": ["SSL certificates", "load balancing", "auto-scaling", "monitoring",
                            "backup strategy", "security compliance", "performance optimization"],
            "error_symptom": ["connection timeout", "authentication failed", "service unavailable",
                             "permission denied", "configuration error", "memory leak", "high CPU usage"],
            "option_a": ["REST API", "GraphQL", "JWT tokens", "Docker Swarm", "MySQL", "Redis"],
            "option_b": ["WebSocket", "RPC", "Session cookies", "Kubernetes", "PostgreSQL", "Memcached"],
            "external_system": ["Slack", "JIRA", "GitHub", "AWS Lambda", "Stripe", "SendGrid", "Datadog"],
            "purpose": ["notifications", "issue tracking", "code deployment", "payment processing",
                       "email delivery", "monitoring alerts", "data synchronization"],
            "activity": ["deployment", "testing", "monitoring", "backup", "scaling", "security",
                        "configuration management", "data migration", "performance tuning"],
            "system_type": ["microservices", "event-driven", "data processing", "real-time",
                           "distributed", "serverless", "batch processing", "streaming"],
            "complex_requirement": ["support multiple tenants", "handle millions of requests",
                                  "integrate with legacy systems", "provide real-time analytics",
                                  "maintain 99.99% uptime", "comply with GDPR"],
            "constraint": ["backward compatibility", "minimal downtime", "limited resources",
                          "existing infrastructure", "security policies", "budget constraints"],
            "old_system": ["Jenkins", "SVN", "MySQL 5.7", "PHP 5.6", "Ubuntu 16.04", "Redis 3.2"],
            "impact_area": ["downtime", "data loss", "user disruption", "performance degradation",
                           "cost increase", "security risks", "maintenance overhead"],
            "workload_type": ["read-heavy", "write-intensive", "batch processing", "real-time streaming",
                             "mixed workloads", "analytical queries", "transactional processing"],
            "scale_requirement": ["100K users", "1M requests per hour", "10TB of data", "global distribution",
                                 "24/7 availability", "sub-second response times", "auto-scaling"],
            "security_requirement": ["end-to-end encryption", "multi-factor authentication", "role-based access control",
                                   "audit logging", "data masking", "secure communication", "vulnerability scanning"],
            "standards": ["SOC 2", "GDPR", "HIPAA", "ISO 27001", "PCI DSS", "NIST", "OWASP Top 10"]
        }
        
        # Apply replacements
        for placeholder, options in replacements.items():
            if f"{{{placeholder}}}" in template:
                value = rng.choice(options)
                template = template.replace(f"{{{placeholder}}}", value)
        
        # Add complexity-based elaboration
        if pattern.complexity == ComplexityLevel.COMPLEX:
            elaborations = [
                " I need a comprehensive guide with examples and best practices.",
                " Please include architectural considerations and potential trade-offs.",
                " Can you provide detailed steps with monitoring and troubleshooting guidance?",
                " I need to understand both the technical implementation and operational aspects."
            ]
            template += rng.choice(elaborations)
        elif pattern.complexity == ComplexityLevel.MEDIUM:
            elaborations = [
                " Please include relevant examples.",
                " I'd like to understand the key considerations.",
                " Can you provide step-by-step guidance?",
                " What are the important things to watch out for?"
            ]
            template += rng.choice(elaborations)
        
        return template
    
    def _generate_ground_truth_docs(
        self, 
        pattern: DocumentationPattern, 
        query_text: str, 
        rng: np.random.RandomState,
        query_index: int
    ) -> List[GroundTruthDocument]:
        """Generate realistic ground truth documentation for queries"""
        
        # Determine number of documents based on complexity
        n_docs = {
            ComplexityLevel.SIMPLE: rng.randint(2, 4),
            ComplexityLevel.MEDIUM: rng.randint(3, 6),
            ComplexityLevel.COMPLEX: rng.randint(4, 8)
        }[pattern.complexity]
        
        docs = []
        
        for doc_idx in range(n_docs):
            doc_id = f"chatty_prose_doc_{query_index:06d}_{doc_idx:02d}"
            
            # Generate document content based on type
            doc_type = rng.choice(pattern.doc_types)
            content = self._generate_doc_content(pattern, doc_type, rng)
            
            # Calculate relevance score
            relevance_score = 0.7 + rng.random() * 0.3  # Between 0.7 and 1.0
            
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
                    "doc_types": pattern.doc_types,
                    "topics": pattern.topics,
                    "information_need": pattern.information_need
                }
            )
            docs.append(doc)
        
        return docs
    
    def _generate_doc_content(
        self, 
        pattern: DocumentationPattern, 
        doc_type: str, 
        rng: np.random.RandomState
    ) -> str:
        """Generate realistic documentation content"""
        
        base_content = pattern.example_content
        
        if doc_type == "API_reference":
            return f"""# API Reference
            
{base_content}

## Endpoints

### GET /api/resource
Returns a list of resources.

**Parameters:**
- `limit` (integer): Maximum number of results (default: 10)
- `offset` (integer): Number of results to skip (default: 0)
- `filter` (string): Filter criteria

**Response:**
```json
{{
  "data": [...],
  "meta": {{
    "total": 100,
    "limit": 10,
    "offset": 0
  }}
}}
```

**Status Codes:**
- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 500: Internal Server Error
"""
        
        elif doc_type == "user_guide":
            return f"""# User Guide

{base_content}

## Overview
This guide walks you through the essential features and workflows.

## Prerequisites
- System requirements
- Required dependencies
- Account setup

## Step-by-Step Instructions

### Basic Setup
1. Download and install the software
2. Create your configuration file
3. Run the initial setup command
4. Verify the installation

### Advanced Configuration
- Custom settings
- Performance tuning
- Security considerations

## Common Workflows
- Daily operations
- Maintenance tasks
- Troubleshooting steps

## Tips and Tricks
- Keyboard shortcuts
- Hidden features
- Performance optimizations
"""
        
        elif doc_type == "tutorial":
            return f"""# Tutorial: Getting Started

{base_content}

## Learning Objectives
By the end of this tutorial, you will:
- Understand core concepts
- Be able to perform basic operations
- Know how to troubleshoot common issues

## Prerequisites
Before starting this tutorial:
- Basic familiarity with the domain
- Required software installed
- Access to example data

## Tutorial Steps

### Step 1: Environment Setup
Set up your development environment with the necessary tools and configurations.

### Step 2: First Example
Walk through a simple example to understand the basics.

### Step 3: Advanced Features  
Explore more sophisticated functionality.

### Step 4: Best Practices
Learn recommended approaches and common pitfalls to avoid.

## Next Steps
- Explore additional features
- Join the community
- Contribute to the project

## Additional Resources
- API documentation
- Example projects
- Community forums
"""
        
        elif doc_type == "FAQ":
            return f"""# Frequently Asked Questions

{base_content}

## General Questions

**Q: What is this tool used for?**
A: This tool is designed to help you accomplish specific tasks efficiently and reliably.

**Q: What are the system requirements?**
A: Minimum requirements include...

**Q: Is this tool free to use?**
A: Yes, the basic version is free. Enterprise features require a license.

## Installation & Setup

**Q: How do I install on Windows/macOS/Linux?**
A: Follow the platform-specific installation guides in our documentation.

**Q: I'm getting a permission error during installation. How do I fix this?**
A: This usually means you need administrator privileges. Try running with sudo or as administrator.

## Common Issues

**Q: The application won't start. What should I check?**
A: First, verify that all dependencies are installed. Check the logs for specific error messages.

**Q: Performance is slow. How can I optimize it?**
A: Several factors can affect performance. Try adjusting configuration settings, increasing memory allocation, or optimizing your data.

## Advanced Usage

**Q: Can I integrate this with my existing workflow?**
A: Yes, we provide APIs and plugins for popular tools.

**Q: How do I backup my configuration?**
A: Configuration files are stored in the user directory and can be backed up manually or through our backup utility.
"""
        
        elif doc_type == "troubleshooting":
            return f"""# Troubleshooting Guide

{base_content}

## Common Problems and Solutions

### Problem: Service Won't Start
**Symptoms:**
- Error messages in logs
- Service status shows as failed
- Application becomes unresponsive

**Possible Causes:**
- Configuration errors
- Port conflicts
- Missing dependencies
- Insufficient permissions

**Solutions:**
1. Check configuration file syntax
2. Verify port availability
3. Install missing dependencies
4. Run with appropriate permissions

### Problem: Performance Issues
**Symptoms:**
- Slow response times
- High CPU or memory usage
- Timeouts and errors

**Diagnostic Steps:**
1. Monitor system resources
2. Check application logs
3. Profile database queries
4. Analyze network connectivity

**Solutions:**
- Optimize configuration settings
- Increase resource allocation
- Implement caching strategies
- Scale infrastructure

### Problem: Authentication Failures
**Symptoms:**
- Login attempts fail
- API calls return 401/403 errors
- Users cannot access resources

**Solutions:**
1. Verify credentials
2. Check token expiration
3. Review permission settings
4. Validate certificate chains

## Getting Help

If these solutions don't resolve your issue:
1. Check the community forums
2. Search existing issues
3. Create a support ticket with:
   - Detailed error messages
   - System information
   - Steps to reproduce
   - Expected vs actual behavior
"""
        
        else:  # best_practices, examples, etc.
            return f"""# Best Practices Guide

{base_content}

## Overview
This guide outlines recommended approaches and patterns for optimal results.

## Design Principles
- Keep it simple
- Plan for scale
- Security by design
- Monitor everything
- Automate when possible

## Configuration Best Practices

### Development Environment
- Use version control for configs
- Separate environment-specific settings
- Document configuration options
- Test configuration changes

### Production Environment
- Use secure defaults
- Enable monitoring and logging
- Implement backup strategies
- Plan for disaster recovery

## Performance Guidelines
- Profile before optimizing
- Cache frequently accessed data
- Use appropriate data structures
- Monitor resource usage
- Scale horizontally when possible

## Security Recommendations
- Enable authentication and authorization
- Use encrypted connections
- Regularly update dependencies
- Implement audit logging
- Follow principle of least privilege

## Maintenance Practices
- Regular health checks
- Automated testing
- Gradual rollouts
- Rollback procedures
- Documentation updates

## Common Pitfalls to Avoid
- Hardcoding configuration values
- Ignoring error handling
- Skipping input validation
- Over-engineering solutions
- Neglecting documentation
"""
    
    def _calculate_docs_quality(
        self, 
        query_text: str, 
        pattern: DocumentationPattern, 
        ground_truth_docs: List[GroundTruthDocument]
    ) -> float:
        """Calculate quality score for documentation domain query"""
        
        score = 0.0
        
        # Length appropriateness (30-400 chars for doc queries)
        length = len(query_text)
        if 40 <= length <= 250:
            score += 0.25
        elif 30 <= length <= 400:
            score += 0.15
        
        # Complexity appropriateness
        complexity_scores = {
            ComplexityLevel.SIMPLE: 0.25,
            ComplexityLevel.MEDIUM: 0.25,
            ComplexityLevel.COMPLEX: 0.2  # Complex queries are harder to get right
        }
        score += complexity_scores[pattern.complexity]
        
        # Documentation specificity
        doc_keywords = [
            "how", "what", "configure", "setup", "install", "guide", "documentation",
            "help", "explain", "tutorial", "example", "reference", "troubleshoot"
        ]
        if any(keyword in query_text.lower() for keyword in doc_keywords):
            score += 0.2
        
        # Information need appropriateness
        information_need_bonus = {
            "factual_knowledge": 0.1,
            "procedural_knowledge": 0.15,
            "problem_solving": 0.1,
            "strategic_guidance": 0.05
        }
        score += information_need_bonus.get(pattern.information_need, 0.05)
        
        # Ground truth appropriateness 
        n_docs = len(ground_truth_docs)
        if pattern.complexity == ComplexityLevel.SIMPLE and 2 <= n_docs <= 4:
            score += 0.15
        elif pattern.complexity == ComplexityLevel.MEDIUM and 3 <= n_docs <= 6:
            score += 0.15
        elif pattern.complexity == ComplexityLevel.COMPLEX and 4 <= n_docs <= 8:
            score += 0.15
        else:
            score += 0.05
        
        # Question clarity bonus
        question_starters = ["how", "what", "where", "when", "why", "which", "can", "does", "is"]
        if any(query_text.lower().startswith(starter) for starter in question_starters):
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def validate_queries(self, queries: List[QueryRecord]) -> ValidationResult:
        """Validate generated documentation domain queries"""
        
        errors = []
        warnings = []
        query_errors = {}
        
        # Check domain consistency
        non_docs_queries = [q for q in queries if q.domain != DomainType.CHATTY_PROSE]
        if non_docs_queries:
            errors.append(f"Found {len(non_docs_queries)} non-documentation queries")
        
        # Check for documentation-specific content
        doc_patterns = [
            r'\b(how|what|where|configure|setup|install|guide|help|explain|tutorial)\b',
            r'\b(documentation|reference|example|troubleshoot|best.practice)\b',
            r'\b(api|user.guide|manual|faq|installation)\b'
        ]
        
        for query in queries:
            query_doc_score = sum(
                1 for pattern in doc_patterns
                if __import__('re').search(pattern, query.query_text.lower())
            )
            
            if query_doc_score == 0:
                query_errors[query.query_id] = ["Query lacks documentation-specific content"]
                
        # Check information need distribution
        information_needs = {}
        for query in queries:
            need = query.metadata.template_used
            information_needs[need] = information_needs.get(need, 0) + 1
        
        # Check quality scores
        if queries:
            avg_quality = sum(q.metadata.quality_score for q in queries) / len(queries)
            if avg_quality < 0.75:
                warnings.append(f"Average quality score below threshold: {avg_quality:.3f}")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            query_errors=query_errors,
            statistics={
                "total_queries": len(queries),
                "avg_quality_score": avg_quality if queries else 0,
                "documentation_content_ratio": (len(queries) - len(query_errors)) / len(queries) if queries else 0,
                "information_need_diversity": len(information_needs)
            },
            recommendations=[
                "Ensure all queries contain documentation-seeking language patterns",
                "Balance information needs across factual, procedural, and strategic types",
                "Include diverse documentation types (guides, references, tutorials, FAQs)",
                "Add realistic technical scenarios and use cases"
            ]
        )