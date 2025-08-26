#!/usr/bin/env python3
"""
Agent Conversation Harness for LetheBench-Agents Dataset Generation
==================================================================

Deterministic agent simulation harness that generates realistic multi-turn
conversations with proper tool usage patterns. Supports four key scenarios:
- Coding: Software development workflows with file operations
- Data Wrangling: Data processing and analysis tasks
- Web-augmented QA: Local-only research and documentation lookup
- CLI Automation: System administration and shell command sequences

Key Features:
- Deterministic execution with consistent seeding
- Realistic tool usage patterns with proper error handling
- Comprehensive atom logging (action, args, observation, plan, tool outputs)
- Configurable scenario complexity and session length
- Privacy-safe content generation with built-in scrubbing
"""

import json
import random
import time
import hashlib
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import logging
import os
import subprocess
import tempfile
import shutil

# Scenario definitions for different agent workflows
@dataclass
class Scenario:
    """Agent scenario configuration"""
    name: str
    description: str
    max_turns: int
    complexity_factors: Dict[str, float]
    required_tools: Set[str]
    common_patterns: List[str]

@dataclass
class AgentAtom:
    """Individual atom in agent conversation trace"""
    atom_id: str
    session_id: str
    turn_index: int
    atom_type: str  # 'user_request', 'agent_plan', 'tool_action', 'tool_observation', 'agent_response'
    content: str
    metadata: Dict[str, Any]
    timestamp: float
    entities: List[Dict[str, Any]]  # Extracted entities with weak labels
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ToolCall:
    """Tool call with arguments and expected outcome"""
    tool_name: str
    arguments: Dict[str, Any]
    expected_success: bool
    expected_output_pattern: Optional[str]
    failure_modes: List[str]

class AgentScenarios:
    """Predefined agent scenarios with realistic workflows"""
    
    @staticmethod
    def get_scenarios() -> Dict[str, Scenario]:
        return {
            'coding': Scenario(
                name='coding',
                description='Software development and debugging workflows',
                max_turns=50,
                complexity_factors={'file_operations': 0.8, 'debugging': 0.9, 'testing': 0.7},
                required_tools={'file_read', 'file_write', 'code_execute', 'git_operations'},
                common_patterns=[
                    'debug_error_sequence', 'implement_feature', 'refactor_code',
                    'setup_project', 'write_tests', 'fix_failing_tests'
                ]
            ),
            'data_wrangling': Scenario(
                name='data_wrangling',
                description='Data processing, analysis, and visualization tasks',
                max_turns=40,
                complexity_factors={'data_loading': 0.6, 'transformation': 0.8, 'analysis': 0.9},
                required_tools={'file_read', 'python_execute', 'data_query', 'visualization'},
                common_patterns=[
                    'load_analyze_data', 'clean_dataset', 'generate_report',
                    'create_dashboard', 'statistical_analysis', 'data_validation'
                ]
            ),
            'web_qa': Scenario(
                name='web_qa',
                description='Web-augmented question answering and research (local only)',
                max_turns=30,
                complexity_factors={'search': 0.7, 'synthesis': 0.8, 'verification': 0.6},
                required_tools={'web_search', 'document_read', 'knowledge_base'},
                common_patterns=[
                    'research_topic', 'fact_checking', 'comparative_analysis',
                    'technical_documentation', 'troubleshoot_issue', 'find_resources'
                ]
            ),
            'cli_automation': Scenario(
                name='cli_automation',
                description='System administration and command-line automation',
                max_turns=35,
                complexity_factors={'system_ops': 0.8, 'automation': 0.9, 'monitoring': 0.7},
                required_tools={'shell_execute', 'file_operations', 'system_monitor', 'process_control'},
                common_patterns=[
                    'system_maintenance', 'log_analysis', 'service_deployment',
                    'backup_restore', 'performance_monitoring', 'security_audit'
                ]
            )
        }

class MockToolExecutor:
    """Mock tool executor that simulates realistic tool behavior"""
    
    def __init__(self, scenario: str, seed: int = 42):
        self.scenario = scenario
        self.seed = seed
        self.rng = random.Random(seed)
        self.temp_dir = tempfile.mkdtemp(prefix=f"agent_sim_{scenario}_")
        self.file_system = {}  # Mock file system state
        self.process_state = {}  # Mock process state
        self.network_state = {}  # Mock network state
        
        # Initialize scenario-specific state
        self._init_scenario_state()
    
    def _init_scenario_state(self):
        """Initialize scenario-specific mock state"""
        if self.scenario == 'coding':
            self.file_system = {
                'main.py': 'print("Hello, World!")',
                'requirements.txt': 'requests==2.28.1\npandas==1.5.0',
                'README.md': '# Sample Project\n\nA simple Python project.',
                'tests/test_main.py': 'import unittest\n\nclass TestMain(unittest.TestCase):\n    pass'
            }
        elif self.scenario == 'data_wrangling':
            self.file_system = {
                'data/sample.csv': 'id,name,value\n1,Alice,100\n2,Bob,200\n3,Charlie,150',
                'analysis.py': 'import pandas as pd\n\ndf = pd.read_csv("data/sample.csv")\nprint(df.describe())',
                'config.yaml': 'data_source: "data/sample.csv"\noutput_format: "json"'
            }
        elif self.scenario == 'web_qa':
            self.network_state = {
                'search_results': ['doc1.md', 'doc2.md', 'doc3.md'],
                'cached_pages': {'doc1.md': 'Comprehensive guide to topic X...'}
            }
        elif self.scenario == 'cli_automation':
            self.process_state = {
                'running_services': ['nginx', 'postgres', 'redis'],
                'system_load': {'cpu': 45.2, 'memory': 62.1, 'disk': 78.9},
                'log_files': ['/var/log/app.log', '/var/log/error.log']
            }
    
    def execute_tool(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Execute a tool call and return realistic results"""
        tool_name = tool_call.tool_name
        args = tool_call.arguments
        
        # Simulate execution time
        execution_time = self.rng.uniform(0.1, 2.0)
        time.sleep(min(execution_time, 0.1))  # Cap actual sleep for performance
        
        try:
            if tool_name == 'file_read':
                return self._execute_file_read(args)
            elif tool_name == 'file_write':
                return self._execute_file_write(args)
            elif tool_name == 'code_execute':
                return self._execute_code(args)
            elif tool_name == 'shell_execute':
                return self._execute_shell(args)
            elif tool_name == 'web_search':
                return self._execute_web_search(args)
            elif tool_name == 'data_query':
                return self._execute_data_query(args)
            else:
                return self._execute_generic_tool(tool_name, args)
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'execution_time': execution_time,
                'timestamp': time.time()
            }
    
    def _execute_file_read(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock file read operation"""
        file_path = args.get('path', '')
        
        if file_path in self.file_system:
            content = self.file_system[file_path]
            return {
                'success': True,
                'content': content,
                'file_size': len(content),
                'file_path': file_path,
                'execution_time': self.rng.uniform(0.05, 0.3)
            }
        else:
            return {
                'success': False,
                'error': f'FileNotFoundError: No such file or directory: {file_path}',
                'error_type': 'FileNotFoundError',
                'execution_time': self.rng.uniform(0.01, 0.1)
            }
    
    def _execute_file_write(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock file write operation"""
        file_path = args.get('path', '')
        content = args.get('content', '')
        
        self.file_system[file_path] = content
        
        return {
            'success': True,
            'bytes_written': len(content),
            'file_path': file_path,
            'execution_time': self.rng.uniform(0.1, 0.5)
        }
    
    def _execute_code(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock code execution"""
        code = args.get('code', '')
        language = args.get('language', 'python')
        
        # Simulate different execution outcomes
        if self.rng.random() < 0.8:  # 80% success rate
            if 'error' in code.lower() or 'exception' in code.lower():
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': 'TypeError: unsupported operand type(s) for +: \'str\' and \'int\'',
                    'exit_code': 1,
                    'execution_time': self.rng.uniform(0.2, 1.0)
                }
            else:
                output = self._generate_code_output(code, language)
                return {
                    'success': True,
                    'stdout': output,
                    'stderr': '',
                    'exit_code': 0,
                    'execution_time': self.rng.uniform(0.5, 3.0)
                }
        else:  # 20% failure rate
            error_types = ['SyntaxError', 'NameError', 'ImportError', 'TypeError']
            error_type = self.rng.choice(error_types)
            return {
                'success': False,
                'stdout': '',
                'stderr': f'{error_type}: {self._generate_error_message(error_type)}',
                'exit_code': 1,
                'execution_time': self.rng.uniform(0.1, 0.5)
            }
    
    def _execute_shell(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock shell command execution"""
        command = args.get('command', '')
        
        # Parse common commands and generate realistic outputs
        if command.startswith('ls'):
            files = list(self.file_system.keys())[:5]  # Show up to 5 files
            output = '\n'.join(files)
            return {
                'success': True,
                'stdout': output,
                'stderr': '',
                'exit_code': 0,
                'execution_time': self.rng.uniform(0.05, 0.2)
            }
        elif command.startswith('ps'):
            processes = ['python main.py', 'nginx: master process', 'postgres: writer process']
            output = 'PID TTY TIME CMD\n' + '\n'.join(f'{1000+i} pts/0 00:00:0{i} {proc}' 
                                                      for i, proc in enumerate(processes))
            return {
                'success': True,
                'stdout': output,
                'stderr': '',
                'exit_code': 0,
                'execution_time': self.rng.uniform(0.1, 0.4)
            }
        elif command.startswith('grep'):
            # Simulate grep results
            if self.rng.random() < 0.7:
                matches = ['line 23: matching content here', 'line 45: another match']
                return {
                    'success': True,
                    'stdout': '\n'.join(matches),
                    'stderr': '',
                    'exit_code': 0,
                    'execution_time': self.rng.uniform(0.2, 0.8)
                }
            else:
                return {
                    'success': True,
                    'stdout': '',
                    'stderr': '',
                    'exit_code': 1,  # No matches found
                    'execution_time': self.rng.uniform(0.1, 0.3)
                }
        else:
            # Generic command simulation
            if self.rng.random() < 0.85:
                return {
                    'success': True,
                    'stdout': f'Output from: {command}',
                    'stderr': '',
                    'exit_code': 0,
                    'execution_time': self.rng.uniform(0.1, 1.0)
                }
            else:
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': f'bash: {command.split()[0]}: command not found',
                    'exit_code': 127,
                    'execution_time': self.rng.uniform(0.05, 0.1)
                }
    
    def _execute_web_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock web search operation (local only)"""
        query = args.get('query', '')
        max_results = args.get('max_results', 5)
        
        # Generate mock search results
        results = []
        for i in range(min(max_results, self.rng.randint(2, 8))):
            results.append({
                'title': f'Result {i+1} for "{query}"',
                'url': f'https://example.com/doc{i+1}',
                'snippet': f'This document discusses {query} in detail...',
                'relevance_score': self.rng.uniform(0.6, 0.95)
            })
        
        return {
            'success': True,
            'results': results,
            'total_results': len(results),
            'query': query,
            'execution_time': self.rng.uniform(0.3, 1.2)
        }
    
    def _execute_data_query(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Mock data query operation"""
        query = args.get('query', '')
        data_source = args.get('data_source', 'default')
        
        # Generate mock data results
        mock_data = [
            {'id': i, 'value': self.rng.randint(10, 100), 'category': self.rng.choice(['A', 'B', 'C'])}
            for i in range(self.rng.randint(5, 20))
        ]
        
        return {
            'success': True,
            'data': mock_data,
            'row_count': len(mock_data),
            'columns': ['id', 'value', 'category'],
            'query': query,
            'data_source': data_source,
            'execution_time': self.rng.uniform(0.2, 0.8)
        }
    
    def _execute_generic_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generic tool execution fallback"""
        success = self.rng.random() < 0.85  # 85% success rate
        
        if success:
            return {
                'success': True,
                'result': f'Successfully executed {tool_name}',
                'execution_time': self.rng.uniform(0.1, 1.0),
                'args_processed': args
            }
        else:
            return {
                'success': False,
                'error': f'Tool {tool_name} execution failed',
                'error_type': 'ToolExecutionError',
                'execution_time': self.rng.uniform(0.05, 0.2)
            }
    
    def _generate_code_output(self, code: str, language: str) -> str:
        """Generate realistic code output"""
        if 'print' in code:
            if 'hello' in code.lower():
                return 'Hello, World!'
            elif 'data' in code.lower():
                return 'Processing data...\nData loaded: 1000 rows\nAnalysis complete.'
            else:
                return 'Code executed successfully.'
        elif 'df.describe()' in code:
            return '''       id      value
count  3.000000   3.000000
mean   2.000000  150.000000
std    1.000000   50.000000
min    1.000000  100.000000
25%    1.500000  125.000000
50%    2.000000  150.000000
75%    2.500000  175.000000
max    3.000000  200.000000'''
        else:
            return f'{language} code executed successfully'
    
    def _generate_error_message(self, error_type: str) -> str:
        """Generate realistic error messages"""
        error_messages = {
            'SyntaxError': 'invalid syntax at line 42',
            'NameError': "name 'undefined_variable' is not defined",
            'ImportError': 'No module named \'nonexistent_module\'',
            'TypeError': "unsupported operand type(s) for +: 'str' and 'int'"
        }
        return error_messages.get(error_type, 'An unexpected error occurred')
    
    def cleanup(self):
        """Clean up temporary resources"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

class AgentConversationSimulator:
    """Main agent conversation simulator"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        self.scenarios = AgentScenarios.get_scenarios()
        self.logger = logging.getLogger(__name__)
        
        # Pattern templates for generating realistic conversations
        self.conversation_patterns = self._load_conversation_patterns()
        
        # Entity extraction patterns
        self.entity_patterns = {
            'file_path': r'([a-zA-Z_][a-zA-Z0-9_./\\-]*\.(?:py|js|java|cpp|h|ts|jsx|tsx|go|rs|rb|php|txt|md|csv|json|yaml|yml|xml))',
            'function_name': r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
            'variable_name': r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=',
            'error_type': r'([A-Za-z_][A-Za-z0-9_]*Error|[A-Za-z_][A-Za-z0-9_]*Exception)',
            'command': r'`([^`]+)`',
            'package_name': r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*)',
            'url': r'https?://[^\s]+',
            'process_id': r'PID:\s*(\d+)',
        }
    
    def _load_conversation_patterns(self) -> Dict[str, List[str]]:
        """Load conversation patterns for each scenario"""
        return {
            'coding': [
                "I need to implement a {function_type} that {task_description}",
                "I'm getting a {error_type} when I run {file_name}",
                "Can you help me debug this {language} code?",
                "How do I optimize this {algorithm_type} algorithm?",
                "I want to add {feature_type} to my {project_type} project",
                "The tests are failing with {error_description}"
            ],
            'data_wrangling': [
                "I need to analyze this {data_type} dataset",
                "Can you help me clean and process {file_name}?",
                "I want to create a {visualization_type} from this data",
                "How do I merge these {number} datasets?",
                "The data loading is failing with {error_message}",
                "I need to generate a report showing {metric_type}"
            ],
            'web_qa': [
                "What is the current best practice for {topic}?",
                "Can you find information about {research_topic}?",
                "I need to compare {option_a} vs {option_b}",
                "How does {technology} work exactly?",
                "What are the pros and cons of {approach}?",
                "Can you help me understand {concept}?"
            ],
            'cli_automation': [
                "I need to monitor {service_name} performance",
                "Can you help me automate {task_type}?",
                "The {service_name} service is not responding",
                "I want to analyze these log files for {issue_type}",
                "How do I set up automated {process_type}?",
                "The system is showing high {resource_type} usage"
            ]
        }
    
    def generate_session(self, scenario_name: str, target_turns: int = None, 
                        complexity_level: str = 'medium') -> List[AgentAtom]:
        """Generate a complete agent conversation session"""
        
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = self.scenarios[scenario_name]
        session_id = f"{scenario_name}_session_{uuid.uuid4().hex[:8]}"
        
        # Determine session length
        if target_turns is None:
            complexity_multiplier = {'simple': 0.5, 'medium': 1.0, 'complex': 1.5}[complexity_level]
            target_turns = int(scenario.max_turns * complexity_multiplier * self.rng.uniform(0.6, 1.0))
        
        # Initialize tool executor for this scenario
        tool_executor = MockToolExecutor(scenario_name, self.seed)
        
        atoms = []
        current_turn = 0
        
        try:
            # Generate initial user request
            initial_request = self._generate_user_request(scenario_name, complexity_level)
            atoms.append(self._create_atom(
                session_id, current_turn, 'user_request', initial_request,
                {'complexity_level': complexity_level, 'scenario': scenario_name}
            ))
            
            current_turn += 1
            
            # Generate conversation turns
            while current_turn < target_turns:
                # Agent planning phase
                plan = self._generate_agent_plan(scenario_name, atoms, current_turn)
                atoms.append(self._create_atom(
                    session_id, current_turn, 'agent_plan', plan,
                    {'planning_context': self._get_recent_context(atoms, 3)}
                ))
                
                # Tool execution phase (if plan requires tools)
                tool_calls = self._extract_tool_calls_from_plan(plan, scenario_name)
                for tool_call in tool_calls:
                    # Tool action
                    action_content = f"Using {tool_call.tool_name} with args: {tool_call.arguments}"
                    atoms.append(self._create_atom(
                        session_id, current_turn, 'tool_action', action_content,
                        {'tool_call': asdict(tool_call)}
                    ))
                    
                    # Tool execution and observation
                    result = tool_executor.execute_tool(tool_call)
                    observation_content = json.dumps(result, indent=2)
                    atoms.append(self._create_atom(
                        session_id, current_turn, 'tool_observation', observation_content,
                        {'tool_result': result, 'tool_name': tool_call.tool_name}
                    ))
                
                # Agent response phase
                response = self._generate_agent_response(scenario_name, atoms, current_turn)
                atoms.append(self._create_atom(
                    session_id, current_turn, 'agent_response', response,
                    {'response_context': self._get_recent_context(atoms, 5)}
                ))
                
                current_turn += 1
                
                # Occasionally generate follow-up user requests
                if current_turn < target_turns and self.rng.random() < 0.3:
                    followup = self._generate_followup_request(scenario_name, atoms)
                    atoms.append(self._create_atom(
                        session_id, current_turn, 'user_request', followup,
                        {'followup': True, 'context_atoms': len(atoms)}
                    ))
                    current_turn += 1
                
        finally:
            tool_executor.cleanup()
        
        return atoms
    
    def _create_atom(self, session_id: str, turn_index: int, atom_type: str, 
                    content: str, metadata: Dict[str, Any]) -> AgentAtom:
        """Create an agent atom with proper formatting and entity extraction"""
        atom_id = f"{session_id}_turn{turn_index}_{atom_type}_{uuid.uuid4().hex[:8]}"
        
        # Extract entities from content
        entities = self._extract_entities(content, atom_type)
        
        # Add standard metadata
        metadata.update({
            'content_length': len(content),
            'word_count': len(content.split()),
            'has_code': '```' in content or '`' in content,
            'has_error': 'error' in content.lower() or 'exception' in content.lower()
        })
        
        return AgentAtom(
            atom_id=atom_id,
            session_id=session_id,
            turn_index=turn_index,
            atom_type=atom_type,
            content=content,
            metadata=metadata,
            timestamp=time.time(),
            entities=entities
        )
    
    def _extract_entities(self, content: str, atom_type: str) -> List[Dict[str, Any]]:
        """Extract entities from content using pattern matching"""
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            import re
            matches = re.findall(pattern, content)
            for match in matches:
                entity_value = match if isinstance(match, str) else match[0]
                entities.append({
                    'type': entity_type,
                    'value': entity_value,
                    'confidence': 0.8,  # Base confidence for pattern matching
                    'source_atom_type': atom_type
                })
        
        return entities
    
    def _generate_user_request(self, scenario_name: str, complexity_level: str) -> str:
        """Generate initial user request for scenario"""
        patterns = self.conversation_patterns[scenario_name]
        pattern = self.rng.choice(patterns)
        
        # Fill in pattern variables based on scenario
        variables = self._get_pattern_variables(scenario_name, complexity_level)
        
        try:
            request = pattern.format(**variables)
        except KeyError:
            # Fallback if pattern variables don't match
            request = pattern.replace('{', '').replace('}', '')
        
        return request
    
    def _get_pattern_variables(self, scenario_name: str, complexity_level: str) -> Dict[str, str]:
        """Get variables for pattern formatting"""
        base_vars = {
            'complexity_level': complexity_level,
            'number': str(self.rng.randint(2, 10))
        }
        
        if scenario_name == 'coding':
            base_vars.update({
                'function_type': self.rng.choice(['function', 'class', 'module', 'API endpoint']),
                'task_description': self.rng.choice(['processes user data', 'handles authentication', 'manages file uploads']),
                'error_type': self.rng.choice(['TypeError', 'ValueError', 'ImportError', 'SyntaxError']),
                'file_name': self.rng.choice(['main.py', 'utils.js', 'config.yaml', 'test.py']),
                'language': self.rng.choice(['Python', 'JavaScript', 'Java', 'C++']),
                'algorithm_type': self.rng.choice(['sorting', 'searching', 'graph traversal', 'dynamic programming']),
                'feature_type': self.rng.choice(['user authentication', 'data validation', 'API integration', 'caching']),
                'project_type': self.rng.choice(['web application', 'CLI tool', 'data pipeline', 'microservice']),
                'error_description': self.rng.choice(['assertion errors', 'timeout issues', 'import failures', 'connection errors'])
            })
        elif scenario_name == 'data_wrangling':
            base_vars.update({
                'data_type': self.rng.choice(['CSV', 'JSON', 'database', 'API', 'Excel']),
                'file_name': self.rng.choice(['data.csv', 'results.json', 'measurements.xlsx', 'logs.txt']),
                'visualization_type': self.rng.choice(['bar chart', 'line graph', 'scatter plot', 'heatmap', 'dashboard']),
                'error_message': self.rng.choice(['encoding errors', 'missing columns', 'data type mismatch', 'memory issues']),
                'metric_type': self.rng.choice(['sales trends', 'user engagement', 'performance metrics', 'quality scores'])
            })
        elif scenario_name == 'web_qa':
            base_vars.update({
                'topic': self.rng.choice(['machine learning', 'web development', 'data science', 'cloud computing']),
                'research_topic': self.rng.choice(['latest frameworks', 'best practices', 'performance optimization', 'security measures']),
                'option_a': self.rng.choice(['React', 'PostgreSQL', 'Docker', 'AWS']),
                'option_b': self.rng.choice(['Vue', 'MySQL', 'Kubernetes', 'Azure']),
                'technology': self.rng.choice(['GraphQL', 'blockchain', 'microservices', 'serverless']),
                'approach': self.rng.choice(['agile methodology', 'TDD', 'microservices architecture', 'event sourcing']),
                'concept': self.rng.choice(['dependency injection', 'eventual consistency', 'CAP theorem', 'SOLID principles'])
            })
        elif scenario_name == 'cli_automation':
            base_vars.update({
                'service_name': self.rng.choice(['nginx', 'postgresql', 'redis', 'elasticsearch', 'docker']),
                'task_type': self.rng.choice(['backup process', 'log rotation', 'deployment pipeline', 'monitoring alerts']),
                'issue_type': self.rng.choice(['errors', 'performance issues', 'security incidents', 'unusual patterns']),
                'process_type': self.rng.choice(['testing', 'deployment', 'monitoring', 'backup']),
                'resource_type': self.rng.choice(['CPU', 'memory', 'disk', 'network'])
            })
        
        return base_vars
    
    def _generate_agent_plan(self, scenario_name: str, atoms: List[AgentAtom], turn_index: int) -> str:
        """Generate agent planning response"""
        recent_context = self._get_recent_context(atoms, 3)
        
        plan_templates = {
            'coding': [
                "I'll help you implement this functionality. Let me start by examining the current code structure.",
                "I need to debug this error. Let me first check the file and then trace the execution path.",
                "To add this feature, I'll need to modify the existing code and write some tests.",
                "Let me analyze the code structure and identify the root cause of this issue."
            ],
            'data_wrangling': [
                "I'll help you process this data. Let me first examine the data structure and quality.",
                "To create this analysis, I need to load the data and explore its characteristics.",
                "I'll start by cleaning the dataset and then proceed with the transformation.",
                "Let me investigate this data issue and propose a solution."
            ],
            'web_qa': [
                "I'll research this topic for you. Let me search for the most current information.",
                "To provide a comprehensive answer, I need to gather information from multiple sources.",
                "Let me look up the latest documentation and best practices for this.",
                "I'll find relevant resources and synthesize the key information."
            ],
            'cli_automation': [
                "I'll help you set up this automation. Let me first check the current system state.",
                "To troubleshoot this issue, I need to examine the logs and system status.",
                "I'll create a monitoring solution and set up the necessary alerts.",
                "Let me analyze the system performance and identify optimization opportunities."
            ]
        }
        
        template = self.rng.choice(plan_templates[scenario_name])
        return template
    
    def _extract_tool_calls_from_plan(self, plan: str, scenario_name: str) -> List[ToolCall]:
        """Extract tool calls implied by agent plan"""
        tool_calls = []
        
        # Pattern-based tool call extraction
        if 'examine' in plan.lower() or 'check' in plan.lower():
            if scenario_name == 'coding':
                tool_calls.append(ToolCall('file_read', {'path': 'main.py'}, True, None, []))
            elif scenario_name == 'data_wrangling':
                tool_calls.append(ToolCall('file_read', {'path': 'data/sample.csv'}, True, None, []))
            elif scenario_name == 'cli_automation':
                tool_calls.append(ToolCall('shell_execute', {'command': 'ps aux'}, True, None, []))
        
        if 'search' in plan.lower() and scenario_name == 'web_qa':
            tool_calls.append(ToolCall('web_search', {'query': 'current best practices', 'max_results': 5}, True, None, []))
        
        if 'run' in plan.lower() or 'execute' in plan.lower():
            if scenario_name == 'coding':
                tool_calls.append(ToolCall('code_execute', {'code': 'print("test")', 'language': 'python'}, True, None, []))
            elif scenario_name == 'data_wrangling':
                tool_calls.append(ToolCall('data_query', {'query': 'SELECT * FROM data LIMIT 10'}, True, None, []))
        
        # Ensure at least one tool call per turn for realism
        if not tool_calls:
            default_tools = {
                'coding': ToolCall('file_read', {'path': 'README.md'}, True, None, []),
                'data_wrangling': ToolCall('data_query', {'query': 'SHOW TABLES'}, True, None, []),
                'web_qa': ToolCall('web_search', {'query': 'documentation'}, True, None, []),
                'cli_automation': ToolCall('shell_execute', {'command': 'ls -la'}, True, None, [])
            }
            tool_calls.append(default_tools[scenario_name])
        
        return tool_calls
    
    def _generate_agent_response(self, scenario_name: str, atoms: List[AgentAtom], turn_index: int) -> str:
        """Generate agent response based on recent tool results"""
        recent_observations = [atom for atom in atoms[-10:] if atom.atom_type == 'tool_observation']
        
        if not recent_observations:
            return "I'm working on this task. Let me gather the necessary information."
        
        latest_observation = recent_observations[-1]
        tool_result = latest_observation.metadata.get('tool_result', {})
        
        if tool_result.get('success', False):
            return self._generate_success_response(scenario_name, tool_result, latest_observation)
        else:
            return self._generate_error_response(scenario_name, tool_result, latest_observation)
    
    def _generate_success_response(self, scenario_name: str, tool_result: Dict[str, Any], 
                                 observation_atom: AgentAtom) -> str:
        """Generate response for successful tool execution"""
        tool_name = observation_atom.metadata.get('tool_name', 'unknown')
        
        response_templates = {
            'file_read': "I've successfully read the file. Here's what I found:\n\n```\n{content}\n```\n\nNow I'll analyze this content and proceed with the next steps.",
            'code_execute': "The code executed successfully with the following output:\n\n```\n{stdout}\n```\n\nThis confirms that the implementation is working as expected.",
            'shell_execute': "Command executed successfully. Here's the output:\n\n```\n{stdout}\n```\n\nBased on this information, I can see the current system state.",
            'web_search': "I found {total_results} relevant results. Here are the key findings:\n\n{summary}\n\nThis gives us good insights into the current best practices.",
            'data_query': "The query returned {row_count} records. Here's a sample of the data:\n\n{data_preview}\n\nThe data structure looks consistent and ready for analysis."
        }
        
        template = response_templates.get(tool_name, "Operation completed successfully. {result}")
        
        try:
            # Format template with tool result data
            formatted_response = template.format(
                content=tool_result.get('content', '')[:200] + '...' if tool_result.get('content') else '',
                stdout=tool_result.get('stdout', '')[:200] + '...' if tool_result.get('stdout') else '',
                total_results=tool_result.get('total_results', 0),
                summary=self._summarize_search_results(tool_result.get('results', [])),
                row_count=tool_result.get('row_count', 0),
                data_preview=json.dumps(tool_result.get('data', [])[:3], indent=2) if tool_result.get('data') else '',
                result=str(tool_result.get('result', ''))
            )
            return formatted_response
        except (KeyError, ValueError):
            return f"Successfully executed {tool_name}. The operation completed as expected."
    
    def _generate_error_response(self, scenario_name: str, tool_result: Dict[str, Any], 
                               observation_atom: AgentAtom) -> str:
        """Generate response for failed tool execution"""
        tool_name = observation_atom.metadata.get('tool_name', 'unknown')
        error = tool_result.get('error', 'Unknown error occurred')
        error_type = tool_result.get('error_type', 'Error')
        
        return f"I encountered a {error_type} while executing {tool_name}:\n\n```\n{error}\n```\n\nLet me try a different approach to resolve this issue."
    
    def _summarize_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Summarize web search results"""
        if not results:
            return "No results found."
        
        summaries = []
        for i, result in enumerate(results[:3]):  # Show top 3 results
            title = result.get('title', f'Result {i+1}')
            snippet = result.get('snippet', 'No description available')
            summaries.append(f"• **{title}**: {snippet[:100]}...")
        
        return "\n".join(summaries)
    
    def _generate_followup_request(self, scenario_name: str, atoms: List[AgentAtom]) -> str:
        """Generate follow-up user request based on conversation context"""
        followup_templates = {
            'coding': [
                "Can you also add error handling to this code?",
                "How do I test this functionality?",
                "Can you optimize this for better performance?",
                "What about edge cases we should consider?",
                "Can you add logging to this implementation?"
            ],
            'data_wrangling': [
                "Can you create a visualization of this data?",
                "How do I export this analysis to Excel?",
                "Can you add statistical significance testing?",
                "What about handling missing values?",
                "Can you automate this analysis pipeline?"
            ],
            'web_qa': [
                "What are the alternatives to this approach?",
                "Can you find more recent information?",
                "How does this compare to industry standards?",
                "Are there any risks or limitations?",
                "Can you provide implementation examples?"
            ],
            'cli_automation': [
                "Can you set up monitoring for this?",
                "How do I automate this task with cron?",
                "What about error handling and notifications?",
                "Can you create a backup strategy?",
                "How do I scale this solution?"
            ]
        }
        
        return self.rng.choice(followup_templates[scenario_name])
    
    def _get_recent_context(self, atoms: List[AgentAtom], count: int) -> List[str]:
        """Get recent conversation context"""
        recent = atoms[-count:] if len(atoms) >= count else atoms
        return [atom.content[:100] + "..." if len(atom.content) > 100 else atom.content for atom in recent]
    
    def cleanup(self):
        """Clean up simulator resources"""
        pass

# Example usage and testing
if __name__ == "__main__":
    simulator = AgentConversationSimulator(seed=42)
    
    # Generate a sample conversation
    atoms = simulator.generate_session('coding', target_turns=10, complexity_level='medium')
    
    print(f"Generated {len(atoms)} atoms for coding session")
    
    for atom in atoms[:5]:  # Show first 5 atoms
        print(f"\nAtom: {atom.atom_type}")
        print(f"Content: {atom.content[:100]}...")
        print(f"Entities: {len(atom.entities)} found")
    
    simulator.cleanup()