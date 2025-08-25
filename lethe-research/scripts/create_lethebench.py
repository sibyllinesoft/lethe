#!/usr/bin/env python3
"""
LetheBench Dataset Creation Script
=================================

Creates comprehensive evaluation dataset by:
1. Processing existing ctx-run examples
2. Generating synthetic queries with known relevance
3. Creating adversarial test cases for robustness evaluation
4. Ensuring balanced coverage across domains and difficulty levels

Usage:
    python create_lethebench.py --examples-dir /path/to/examples --output dataset.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import hashlib
import random
from datetime import datetime, timezone
import yaml

# Add research directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def setup_logging(level: str) -> logging.Logger:
    """Configure logging with specified level."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

class LetheBenchCreator:
    """Creates the LetheBench evaluation dataset."""
    
    def __init__(self, examples_dir: Path, config_path: Path, logger: logging.Logger):
        self.examples_dir = examples_dir
        self.config_path = config_path
        self.logger = logger
        self.config = self._load_config()
        
        # Dataset parameters
        self.queries_per_domain = self.config.get('dataset', {}).get('queries_per_domain', 100)
        self.difficulty_levels = ['easy', 'medium', 'hard']
        self.domains = ['web', 'api', 'cli', 'config', 'docs']
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            self.logger.warning(f"Config file not found: {self.config_path}")
            return {}
            
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def discover_examples(self) -> List[Dict[str, Any]]:
        """Discover and process ctx-run example projects and conversation JSON files."""
        examples = []
        
        if not self.examples_dir.exists():
            self.logger.error(f"Examples directory not found: {self.examples_dir}")
            return examples
            
        self.logger.info(f"Scanning examples directory: {self.examples_dir}")
        
        for example_path in self.examples_dir.iterdir():
            if example_path.is_dir() and not example_path.name.startswith('.'):
                # Process directory-based examples (original functionality)
                example_data = self._process_example(example_path)
                if example_data:
                    examples.append(example_data)
            elif example_path.is_file() and example_path.suffix == '.json':
                # Process conversation JSON files (new functionality)
                conversation_data = self._process_conversation_file(example_path)
                if conversation_data:
                    examples.append(conversation_data)
                
        self.logger.info(f"Discovered {len(examples)} examples")
        return examples
    
    def _process_example(self, example_path: Path) -> Dict[str, Any]:
        """Process individual example directory."""
        try:
            # Read package.json or equivalent metadata
            metadata_file = example_path / "package.json"
            ctx_file = example_path / ".ctx"
            readme_file = example_path / "README.md"
            
            example_data = {
                'id': self._generate_id(example_path.name),
                'name': example_path.name,
                'path': str(example_path),
                'domain': self._classify_domain(example_path),
                'files': self._scan_files(example_path),
                'metadata': {}
            }
            
            # Extract metadata
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        example_data['metadata'] = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to read {metadata_file}: {e}")
            
            # Check for ctx configuration
            if ctx_file.exists():
                try:
                    with open(ctx_file) as f:
                        example_data['ctx_config'] = json.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to read {ctx_file}: {e}")
            
            # Extract description from README
            if readme_file.exists():
                try:
                    with open(readme_file) as f:
                        content = f.read()
                        # Extract first paragraph as description
                        lines = content.split('\n')
                        description_lines = []
                        for line in lines:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                description_lines.append(line)
                                if len(description_lines) >= 3:  # Get first few sentences
                                    break
                        example_data['description'] = ' '.join(description_lines)
                except Exception as e:
                    self.logger.warning(f"Failed to read {readme_file}: {e}")
            
            return example_data
            
        except Exception as e:
            self.logger.error(f"Failed to process example {example_path}: {e}")
            return None
    
    def _process_conversation_file(self, file_path: Path) -> Dict[str, Any]:
        """Process conversation JSON file."""
        try:
            with open(file_path) as f:
                conversation = json.load(f)
            
            # Validate conversation structure
            if not isinstance(conversation, dict):
                self.logger.warning(f"Invalid conversation format in {file_path}")
                return None
                
            messages = conversation.get('messages', [])
            if not messages:
                self.logger.warning(f"No messages found in conversation {file_path}")
                return None
            
            # Extract conversation metadata
            session_id = conversation.get('session_id', file_path.stem)
            
            # Classify domain based on filename and content
            domain = self._classify_conversation_domain(file_path, messages)
            
            # Generate content chunks from conversation
            chunks = self._extract_conversation_chunks(messages)
            
            # Create example data structure
            example_data = {
                'id': self._generate_id(session_id),
                'name': file_path.stem,
                'path': str(file_path),
                'domain': domain,
                'type': 'conversation',
                'session_id': session_id,
                'message_count': len(messages),
                'chunks': chunks,
                'metadata': {
                    'source_file': str(file_path),
                    'conversation_length': len(messages),
                    'timestamp_range': self._get_timestamp_range(messages)
                }
            }
            
            return example_data
            
        except Exception as e:
            self.logger.error(f"Failed to process conversation file {file_path}: {e}")
            return None
    
    def _scan_files(self, example_path: Path) -> List[Dict[str, Any]]:
        """Scan files in example directory."""
        files = []
        
        for file_path in example_path.rglob('*'):
            if file_path.is_file() and not self._should_ignore_file(file_path):
                try:
                    stat = file_path.stat()
                    file_info = {
                        'path': str(file_path.relative_to(example_path)),
                        'size': stat.st_size,
                        'extension': file_path.suffix,
                        'type': self._classify_file_type(file_path)
                    }
                    files.append(file_info)
                except Exception as e:
                    self.logger.warning(f"Failed to stat file {file_path}: {e}")
        
        return files
    
    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored."""
        ignore_patterns = [
            'node_modules', '.git', '.DS_Store', '__pycache__',
            '.pytest_cache', 'dist', 'build', '.next'
        ]
        
        path_str = str(file_path)
        return any(pattern in path_str for pattern in ignore_patterns)
    
    def _classify_file_type(self, file_path: Path) -> str:
        """Classify file type based on extension."""
        ext = file_path.suffix.lower()
        
        if ext in ['.js', '.ts', '.jsx', '.tsx']:
            return 'javascript'
        elif ext in ['.py']:
            return 'python'
        elif ext in ['.json', '.yaml', '.yml']:
            return 'config'
        elif ext in ['.md', '.txt', '.rst']:
            return 'documentation'
        elif ext in ['.html', '.css', '.scss']:
            return 'web'
        elif ext in ['.sh', '.bash']:
            return 'script'
        else:
            return 'other'
    
    def _classify_domain(self, example_path: Path) -> str:
        """Classify example domain based on path and contents."""
        name = example_path.name.lower()
        
        if any(keyword in name for keyword in ['web', 'react', 'next', 'frontend']):
            return 'web'
        elif any(keyword in name for keyword in ['api', 'server', 'backend']):
            return 'api'
        elif any(keyword in name for keyword in ['cli', 'command', 'tool']):
            return 'cli'
        elif any(keyword in name for keyword in ['config', 'setup', 'template']):
            return 'config'
        elif any(keyword in name for keyword in ['docs', 'documentation', 'guide']):
            return 'docs'
        else:
            return 'general'
    
    def _classify_conversation_domain(self, file_path: Path, messages: List[Dict[str, Any]]) -> str:
        """Classify conversation domain based on filename and content."""
        filename = file_path.name.lower()
        
        # Check filename for domain indicators
        if 'code' in filename or 'component' in filename:
            return 'web'
        elif 'api' in filename or 'backend' in filename:
            return 'api'
        elif 'tool' in filename or 'deploy' in filename:
            return 'cli'
        elif 'config' in filename:
            return 'config'
        elif 'doc' in filename:
            return 'docs'
        
        # Analyze message content for domain clues
        all_text = ' '.join([msg.get('text', '').lower() for msg in messages])
        
        if any(keyword in all_text for keyword in ['react', 'component', 'css', 'html', 'frontend', 'ui']):
            return 'web'
        elif any(keyword in all_text for keyword in ['api', 'server', 'endpoint', 'backend', 'database']):
            return 'api'
        elif any(keyword in all_text for keyword in ['deployment', 'github actions', 'ci/cd', 'docker']):
            return 'cli'
        elif any(keyword in all_text for keyword in ['config', 'setup', 'environment']):
            return 'config'
        elif any(keyword in all_text for keyword in ['documentation', 'guide', 'tutorial']):
            return 'docs'
        
        return 'general'
    
    def _extract_conversation_chunks(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract meaningful chunks from conversation messages."""
        chunks = []
        
        for i, message in enumerate(messages):
            text = message.get('text', '').strip()
            if not text:
                continue
                
            # Extract code blocks if present
            code_blocks = self._extract_code_blocks(text)
            
            chunk = {
                'id': f"chunk_{i}",
                'turn': message.get('turn', i + 1),
                'role': message.get('role', 'unknown'),
                'text': text,
                'timestamp': message.get('timestamp'),
                'code_blocks': code_blocks,
                'has_code': len(code_blocks) > 0,
                'length': len(text)
            }
            chunks.append(chunk)
        
        return chunks
    
    def _extract_code_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Extract code blocks from message text."""
        import re
        code_blocks = []
        
        # Match triple backtick code blocks
        pattern = r'```(\w+)?\n?(.*?)\n?```'
        matches = re.finditer(pattern, text, re.DOTALL)
        
        for match in matches:
            language = match.group(1) or 'text'
            code = match.group(2).strip()
            
            if code:
                code_blocks.append({
                    'language': language,
                    'code': code,
                    'length': len(code)
                })
        
        return code_blocks
    
    def _get_timestamp_range(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get timestamp range from conversation messages."""
        timestamps = [msg.get('timestamp') for msg in messages if msg.get('timestamp')]
        
        if timestamps:
            return {
                'start': min(timestamps),
                'end': max(timestamps),
                'duration': max(timestamps) - min(timestamps)
            }
        
        return {'start': None, 'end': None, 'duration': None}
    
    def generate_queries(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate synthetic queries for evaluation."""
        queries = []
        
        self.logger.info("Generating synthetic queries...")
        
        # Generate queries from conversation examples
        conversation_queries = self._generate_conversation_queries(examples)
        queries.extend(conversation_queries)
        
        # Generate template-based queries for domain coverage
        for domain in self.domains:
            domain_examples = [ex for ex in examples if ex.get('domain') == domain]
            if not domain_examples:
                self.logger.warning(f"No examples found for domain: {domain}")
                continue
                
            for difficulty in self.difficulty_levels:
                queries_count = max(1, self.queries_per_domain // len(self.difficulty_levels) // 2)  # Reduced count since we have conversation queries
                domain_queries = self._generate_domain_queries(
                    domain, domain_examples, difficulty, queries_count
                )
                queries.extend(domain_queries)
        
        # Add adversarial queries for robustness testing
        adversarial_queries = self._generate_adversarial_queries(examples)
        queries.extend(adversarial_queries)
        
        self.logger.info(f"Generated {len(queries)} total queries")
        return queries
    
    def _generate_conversation_queries(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate queries based on conversation content."""
        queries = []
        
        conversation_examples = [ex for ex in examples if ex.get('type') == 'conversation']
        
        for example in conversation_examples:
            chunks = example.get('chunks', [])
            
            # Generate queries from user messages (natural questions)
            user_queries = self._extract_user_queries(example, chunks)
            queries.extend(user_queries)
            
            # Generate synthetic queries based on assistant responses
            synthetic_queries = self._generate_synthetic_conversation_queries(example, chunks)
            queries.extend(synthetic_queries)
        
        return queries
    
    def _extract_user_queries(self, example: Dict[str, Any], chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract actual user questions from conversation as evaluation queries."""
        queries = []
        
        for chunk in chunks:
            if chunk['role'] == 'user':
                text = chunk['text']
                
                # Filter out very short queries or non-questions
                if len(text.strip()) < 10:
                    continue
                
                query = {
                    'id': self._generate_id(f"user_query_{example['id']}_{chunk['id']}"),
                    'text': text,
                    'domain': example['domain'],
                    'difficulty': self._classify_query_difficulty(text),
                    'expected_results': self._generate_conversation_expected_results(example, chunk),
                    'metadata': {
                        'source_type': 'user_query',
                        'source_example': example['id'],
                        'source_chunk': chunk['id'],
                        'has_followup': self._has_followup_response(chunks, chunk),
                        'turn': chunk['turn']
                    }
                }
                queries.append(query)
        
        return queries
    
    def _generate_synthetic_conversation_queries(self, example: Dict[str, Any], chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate synthetic queries based on assistant responses."""
        queries = []
        
        assistant_chunks = [chunk for chunk in chunks if chunk['role'] == 'assistant' and chunk.get('has_code', False)]
        
        for chunk in assistant_chunks[:3]:  # Limit to first 3 code-containing responses
            # Generate "how to" questions based on code content
            code_blocks = chunk.get('code_blocks', [])
            
            for code_block in code_blocks:
                language = code_block['language']
                
                # Generate synthetic query based on code type
                synthetic_text = self._generate_code_based_query(language, code_block['code'], example['domain'])
                
                if synthetic_text:
                    query = {
                        'id': self._generate_id(f"synthetic_{example['id']}_{chunk['id']}_{language}"),
                        'text': synthetic_text,
                        'domain': example['domain'],
                        'difficulty': 'medium',
                        'expected_results': self._generate_conversation_expected_results(example, chunk),
                        'metadata': {
                            'source_type': 'synthetic_code',
                            'source_example': example['id'],
                            'source_chunk': chunk['id'],
                            'code_language': language,
                            'turn': chunk['turn']
                        }
                    }
                    queries.append(query)
        
        return queries
    
    def _classify_query_difficulty(self, text: str) -> str:
        """Classify query difficulty based on complexity indicators."""
        text_lower = text.lower()
        
        # Hard indicators
        hard_indicators = ['architecture', 'optimization', 'performance', 'security', 'scalability', 'advanced', 'complex']
        if any(indicator in text_lower for indicator in hard_indicators):
            return 'hard'
        
        # Medium indicators  
        medium_indicators = ['implement', 'integrate', 'configure', 'deploy', 'middleware', 'authentication']
        if any(indicator in text_lower for indicator in medium_indicators):
            return 'medium'
        
        # Easy by default for basic questions
        return 'easy'
    
    def _generate_code_based_query(self, language: str, code: str, domain: str) -> str:
        """Generate a synthetic query based on code content."""
        code_lower = code.lower()
        
        if language == 'jsx' or 'react' in code_lower:
            if 'component' in code_lower:
                return "How do I create a reusable React component?"
            elif 'usestate' in code_lower or 'state' in code_lower:
                return "How do I manage state in React components?"
            else:
                return "How do I build React applications?"
                
        elif language == 'javascript' or language == 'js':
            if 'async' in code_lower or 'await' in code_lower:
                return "How do I handle asynchronous operations in JavaScript?"
            elif 'express' in code_lower or 'router' in code_lower:
                return "How do I create REST API endpoints?"
            else:
                return "How do I write modern JavaScript code?"
                
        elif language == 'css':
            if '@media' in code_lower:
                return "How do I make my website responsive?"
            elif 'flexbox' in code_lower or 'flex' in code_lower:
                return "How do I create flexible layouts with CSS?"
            else:
                return "How do I style web components?"
                
        elif language == 'yaml' or language == 'yml':
            if 'github' in code_lower or 'actions' in code_lower:
                return "How do I set up CI/CD with GitHub Actions?"
            else:
                return "How do I write configuration files?"
        
        # Generic fallback
        return f"How do I work with {language} in {domain} development?"
    
    def _has_followup_response(self, chunks: List[Dict[str, Any]], user_chunk: Dict[str, Any]) -> bool:
        """Check if user query has a followup assistant response."""
        user_turn = user_chunk['turn']
        return any(chunk['role'] == 'assistant' and chunk['turn'] > user_turn for chunk in chunks)
    
    def _generate_conversation_expected_results(self, example: Dict[str, Any], relevant_chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate expected results for conversation-based queries."""
        results = []
        
        # Primary result from the conversation itself
        primary_result = {
            'example_id': example['id'],
            'chunk_id': relevant_chunk['id'],
            'relevance_score': 1.0,
            'explanation': f"Direct match from {example['name']} conversation",
            'rank': 1,
            'content_type': 'conversation'
        }
        results.append(primary_result)
        
        # If there's a followup response, include it as highly relevant
        chunks = example.get('chunks', [])
        current_turn = relevant_chunk['turn']
        
        for chunk in chunks:
            if chunk['turn'] == current_turn + 1 and chunk['role'] == 'assistant':
                followup_result = {
                    'example_id': example['id'],
                    'chunk_id': chunk['id'],
                    'relevance_score': 0.9,
                    'explanation': f"Direct response to query in {example['name']}",
                    'rank': 2,
                    'content_type': 'conversation'
                }
                results.append(followup_result)
                break
        
        return results
    
    def _generate_domain_queries(self, domain: str, examples: List[Dict[str, Any]], 
                                difficulty: str, count: int) -> List[Dict[str, Any]]:
        """Generate queries for specific domain and difficulty."""
        queries = []
        
        query_templates = self._get_query_templates(domain, difficulty)
        
        for i in range(count):
            template = random.choice(query_templates)
            example = random.choice(examples)
            
            query = {
                'id': self._generate_id(f"{domain}_{difficulty}_{i}"),
                'text': self._fill_template(template, example),
                'domain': domain,
                'difficulty': difficulty,
                'expected_results': self._generate_expected_results(example, difficulty),
                'metadata': {
                    'template_used': template,
                    'source_example': example['id']
                }
            }
            queries.append(query)
        
        return queries
    
    def _get_query_templates(self, domain: str, difficulty: str) -> List[str]:
        """Get query templates for domain and difficulty."""
        templates = {
            'web': {
                'easy': [
                    "How do I create a React component?",
                    "What is the basic HTML structure?",
                    "How do I style elements with CSS?"
                ],
                'medium': [
                    "How do I implement routing in a React application?",
                    "What's the best way to handle form validation?",
                    "How do I optimize web performance?"
                ],
                'hard': [
                    "How do I implement server-side rendering with hydration?",
                    "What's the best architecture for a micro-frontend system?",
                    "How do I debug complex state management issues?"
                ]
            },
            'api': {
                'easy': [
                    "How do I create a REST endpoint?",
                    "What is API authentication?",
                    "How do I handle HTTP requests?"
                ],
                'medium': [
                    "How do I implement rate limiting?",
                    "What's the best way to version APIs?",
                    "How do I handle async operations?"
                ],
                'hard': [
                    "How do I design a GraphQL schema for complex relationships?",
                    "What's the best approach for API gateway architecture?",
                    "How do I implement distributed tracing?"
                ]
            },
            'cli': {
                'easy': [
                    "How do I parse command line arguments?",
                    "What is the basic CLI structure?",
                    "How do I handle user input?"
                ],
                'medium': [
                    "How do I implement command autocomplete?",
                    "What's the best way to handle configuration files?",
                    "How do I create interactive prompts?"
                ],
                'hard': [
                    "How do I implement a plugin system for CLI tools?",
                    "What's the best approach for cross-platform compatibility?",
                    "How do I optimize CLI startup performance?"
                ]
            }
        }
        
        return templates.get(domain, {}).get(difficulty, [
            f"How do I work with {domain}?",
            f"What are best practices for {domain}?",
            f"How do I debug {domain} issues?"
        ])
    
    def _fill_template(self, template: str, example: Dict[str, Any]) -> str:
        """Fill template with example-specific information."""
        # Simple template filling - in practice, this would be more sophisticated
        return template.replace("{example}", example.get('name', 'example'))
    
    def _generate_expected_results(self, example: Dict[str, Any], difficulty: str) -> List[Dict[str, Any]]:
        """Generate expected results for query based on example."""
        results = []
        
        # Primary result from the source example
        primary_result = {
            'example_id': example['id'],
            'relevance_score': 1.0,
            'explanation': f"Direct match from {example['name']} example",
            'rank': 1
        }
        results.append(primary_result)
        
        # Add secondary results based on difficulty
        secondary_count = {'easy': 2, 'medium': 3, 'hard': 5}.get(difficulty, 2)
        
        for i in range(secondary_count):
            secondary_result = {
                'example_id': f"related_{i}",
                'relevance_score': max(0.3, 1.0 - (i * 0.2)),
                'explanation': f"Related concept or pattern",
                'rank': i + 2
            }
            results.append(secondary_result)
        
        return results
    
    def _generate_adversarial_queries(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate adversarial queries for robustness testing."""
        adversarial_queries = []
        
        # Edge cases
        edge_cases = [
            {
                'text': "",  # Empty query
                'domain': 'adversarial',
                'difficulty': 'edge_case',
                'expected_results': []
            },
            {
                'text': "a" * 1000,  # Very long query
                'domain': 'adversarial',
                'difficulty': 'edge_case',
                'expected_results': []
            },
            {
                'text': "🚀 How do I implement 🎯 performance optimization with 💻?",  # Emoji query
                'domain': 'adversarial',
                'difficulty': 'edge_case',
                'expected_results': []
            }
        ]
        
        for i, case in enumerate(edge_cases):
            case['id'] = self._generate_id(f"adversarial_edge_{i}")
            adversarial_queries.append(case)
        
        return adversarial_queries
    
    def _generate_id(self, text: str) -> str:
        """Generate deterministic ID from text."""
        return hashlib.md5(text.encode()).hexdigest()[:12]
    
    def create_dataset(self, output_path: Path) -> Dict[str, Any]:
        """Create complete LetheBench dataset."""
        self.logger.info("Creating LetheBench dataset...")
        
        # Discover examples
        examples = self.discover_examples()
        if not examples:
            raise ValueError("No examples found to create dataset")
        
        # Generate queries
        queries = self.generate_queries(examples)
        if not queries:
            raise ValueError("No queries generated for dataset")
        
        # Create dataset structure
        dataset = {
            'metadata': {
                'name': 'LetheBench',
                'version': '1.0.0',
                'created': datetime.now(timezone.utc).isoformat(),
                'examples_source': str(self.examples_dir),
                'total_examples': len(examples),
                'total_queries': len(queries),
                'domains': list(set(ex.get('domain') for ex in examples)),
                'difficulty_levels': self.difficulty_levels
            },
            'examples': examples,
            'queries': queries,
            'evaluation_config': {
                'metrics': ['ndcg_at_10', 'recall_at_10', 'mrr_at_10'],
                'k_values': [5, 10, 20],
                'bootstrap_samples': 10000,
                'confidence_level': 0.95
            }
        }
        
        # Save dataset
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        self.logger.info(f"Dataset saved to: {output_path}")
        self.logger.info(f"Examples: {len(examples)}, Queries: {len(queries)}")
        
        return dataset

def main():
    parser = argparse.ArgumentParser(description='Create LetheBench evaluation dataset')
    parser.add_argument('--examples-dir', type=Path, required=True,
                       help='Directory containing ctx-run examples')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output path for dataset JSON file')
    parser.add_argument('--config', type=Path,
                       help='Configuration file path')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else args.log_level
    logger = setup_logging(log_level)
    
    try:
        # Create dataset creator
        creator = LetheBenchCreator(
            examples_dir=args.examples_dir,
            config_path=args.config or Path('experiments/grid_config.yaml'),
            logger=logger
        )
        
        # Create dataset
        dataset = creator.create_dataset(args.output)
        
        logger.info("✅ LetheBench dataset creation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Dataset creation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()