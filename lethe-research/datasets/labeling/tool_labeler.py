#!/usr/bin/env python3
"""
Tool Gold Labeler for LetheBench

Implements weak supervision for identifying tool outputs and their dependencies
in CLI tutorials, notebook transcripts, and technical documentation.

For NeurIPS LetheBench Paper - Dataset Construction Pipeline
"""

import re
import json
from typing import List, Dict, Set, Tuple, Optional
import logging
from dataclasses import dataclass

@dataclass
class ToolChunk:
    """Represents a gold tool output chunk annotation."""
    chunk_id: str
    session_id: str
    turn_id: int
    content: str
    chunk_type: str  # 'command_output', 'table', 'json_response', 'log_entry', 'error_output'
    tool_name: str
    dependencies: List[int]  # Turn IDs this chunk depends on
    context_start: int
    context_end: int
    confidence: float
    metadata: Dict

class ToolLabeler:
    """
    Generates gold annotations for tool-result dialog sessions.
    
    Uses weak supervision to identify:
    - Command line tool outputs
    - API responses and structured data
    - Log entries and error messages  
    - Tables and formatted results
    - Dependencies between tool invocations and outputs
    """
    
    def __init__(self):
        """Initialize tool labeler with pattern recognition."""
        self.logger = logging.getLogger(__name__)
        
        # Tool output patterns
        self.tool_patterns = {
            'command_prompt': {
                'bash': r'^\$\s+(.+)$',
                'powershell': r'^PS\s+[^>]+>\s+(.+)$',
                'cmd': r'^[A-Z]:\\[^>]*>\s*(.+)$',
                'generic': r'^[#$%>]\s+(.+)$'
            },
            'command_output': {
                'ls_output': r'^(?:[-drwx]{10}|\d+)\s+.*$',
                'ps_output': r'^\s*PID\s+.*|^\s*\d+\s+.*$',
                'df_output': r'^Filesystem\s+.*|^[/\w]+\s+\d+.*$',
                'git_log': r'^commit\s+[a-f0-9]{40}$',
                'npm_list': r'^[├└│─]+\s*[\w@-]+@[\d.]+$',
                'pip_list': r'^[\w-]+\s+[\d.]+.*$'
            },
            'structured_output': {
                'json': r'^\s*[\{\[].*[\}\]]\s*$',
                'yaml': r'^\s*[\w-]+:\s*.*$',
                'csv': r'^[\w\s,().-]+,.*$',
                'table_header': r'^[\w\s|+-]+$',
                'table_row': r'^[\w\s|+-]*\|.*$'
            },
            'log_entry': {
                'timestamp': r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}',
                'log_level': r'\b(?:DEBUG|INFO|WARN|ERROR|FATAL|TRACE)\b',
                'http_status': r'\b(?:[1-5]\d{2})\b',
                'error_pattern': r'\b(?:Error|Exception|Failed|Timeout)\b'
            },
            'file_paths': {
                'unix_path': r'/[\w/.-]+',
                'windows_path': r'[A-Z]:\\[\w\\.-]+',
                'url': r'https?://[\w.-]+(?:/[\w.-]*)*',
                'relative_path': r'\.{1,2}/[\w/.-]+'
            }
        }
        
        # Tool names and their common output patterns
        self.tool_signatures = {
            'git': ['commit', 'branch', 'status', 'diff', 'log', 'push', 'pull'],
            'docker': ['CONTAINER ID', 'IMAGE', 'STATUS', 'PORTS'],
            'kubectl': ['NAME', 'READY', 'STATUS', 'RESTARTS', 'AGE'],
            'npm': ['npm', 'node_modules', 'package.json', 'dependencies'],
            'pip': ['Successfully installed', 'Requirement already satisfied'],
            'curl': ['HTTP/1.1', 'Content-Type', 'Content-Length'],
            'ls': ['total', 'drwx', '-rw-', 'rwx'],
            'ps': ['PID', 'TTY', 'TIME', 'CMD'],
            'df': ['Filesystem', 'Size', 'Used', 'Avail', 'Mounted'],
            'top': ['PID', 'USER', '%CPU', '%MEM'],
            'netstat': ['Proto', 'Local Address', 'State'],
            'systemctl': ['Active:', 'Loaded:', 'Main PID:'],
            'journalctl': ['systemd', 'kernel', 'dmesg']
        }
        
        # API response patterns
        self.api_patterns = {
            'rest_response': r'\{[^{}]*"(?:status|code|message|data|result)"[^{}]*\}',
            'graphql_response': r'\{[^{}]*"(?:data|errors)"[^{}]*\}',
            'http_headers': r'^[\w-]+:\s+.*$',
            'status_line': r'^HTTP/[\d.]+\s+\d{3}\s+.*$'
        }
    
    def identify_tool_outputs(self, text: str) -> List[Dict]:
        """
        Identify tool outputs and structured data in text.
        
        Args:
            text: Text content to analyze
            
        Returns:
            List of tool output annotations with metadata
        """
        outputs = []
        lines = text.split('\n')
        
        # Analyze line by line for patterns
        current_block = []
        block_type = None
        block_start = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                if current_block:
                    # End of block
                    output = self._process_output_block(
                        current_block, block_type, block_start, i
                    )
                    if output:
                        outputs.append(output)
                    current_block = []
                    block_type = None
                continue
            
            # Detect block type
            detected_type = self._detect_line_type(line)
            
            if detected_type:
                if block_type != detected_type:
                    # Start new block
                    if current_block:
                        output = self._process_output_block(
                            current_block, block_type, block_start, i
                        )
                        if output:
                            outputs.append(output)
                    
                    current_block = [line]
                    block_type = detected_type
                    block_start = i
                else:
                    # Continue current block
                    current_block.append(line)
            else:
                # Unstructured line
                if current_block and block_type:
                    current_block.append(line)
                else:
                    # Single-line analysis for inline outputs
                    inline_output = self._analyze_inline_output(line, i)
                    if inline_output:
                        outputs.append(inline_output)
        
        # Process final block
        if current_block:
            output = self._process_output_block(
                current_block, block_type, block_start, len(lines)
            )
            if output:
                outputs.append(output)
        
        return outputs
    
    def _detect_line_type(self, line: str) -> Optional[str]:
        """Detect the type of a line based on patterns."""
        line_stripped = line.strip()
        
        # Command prompts
        for prompt_type, pattern in self.tool_patterns['command_prompt'].items():
            if re.match(pattern, line_stripped):
                return 'command'
        
        # Structured data
        for struct_type, pattern in self.tool_patterns['structured_output'].items():
            if re.match(pattern, line_stripped, re.MULTILINE):
                return 'structured_data'
        
        # Log entries
        if any(re.search(pattern, line_stripped) for pattern in self.tool_patterns['log_entry'].values()):
            return 'log_entry'
        
        # Tool-specific outputs
        for tool_name, signatures in self.tool_signatures.items():
            if any(sig.lower() in line_stripped.lower() for sig in signatures):
                return f'tool_output_{tool_name}'
        
        return None
    
    def _process_output_block(self, lines: List[str], block_type: str, start_line: int, end_line: int) -> Optional[Dict]:
        """Process a block of related output lines."""
        if not lines or not block_type:
            return None
        
        content = '\n'.join(lines)
        
        # Determine tool name
        tool_name = self._identify_tool(content, block_type)
        
        # Calculate confidence
        confidence = self._calculate_output_confidence(content, block_type, tool_name)
        
        if confidence < 0.3:  # Minimum threshold
            return None
        
        # Detect structure type
        structure_type = self._classify_output_structure(content)
        
        return {
            'type': structure_type,
            'tool_name': tool_name,
            'content': content,
            'start_line': start_line,
            'end_line': end_line,
            'confidence': confidence,
            'metadata': {
                'block_type': block_type,
                'line_count': len(lines),
                'has_structured_data': self._has_structured_data(content),
                'has_errors': self._has_error_indicators(content)
            }
        }
    
    def _analyze_inline_output(self, line: str, line_number: int) -> Optional[Dict]:
        """Analyze a single line for inline tool outputs."""
        # JSON responses
        if re.search(self.api_patterns['rest_response'], line):
            return {
                'type': 'json_response',
                'tool_name': 'api',
                'content': line.strip(),
                'start_line': line_number,
                'end_line': line_number,
                'confidence': 0.8,
                'metadata': {'inline': True, 'response_type': 'json'}
            }
        
        # HTTP status lines
        if re.match(self.api_patterns['status_line'], line.strip()):
            return {
                'type': 'http_response',
                'tool_name': 'curl',
                'content': line.strip(),
                'start_line': line_number,
                'end_line': line_number,
                'confidence': 0.9,
                'metadata': {'inline': True, 'response_type': 'http'}
            }
        
        return None
    
    def _identify_tool(self, content: str, block_type: str) -> str:
        """Identify the tool that generated the output."""
        content_lower = content.lower()
        
        # Check tool signatures
        for tool_name, signatures in self.tool_signatures.items():
            if any(sig.lower() in content_lower for sig in signatures):
                return tool_name
        
        # Extract from block type if tool-specific
        if block_type.startswith('tool_output_'):
            return block_type.replace('tool_output_', '')
        
        # Default classification
        if 'http' in content_lower or 'api' in content_lower:
            return 'api'
        elif any(char in content for char in '{}[]'):
            return 'json_tool'
        else:
            return 'unknown'
    
    def _classify_output_structure(self, content: str) -> str:
        """Classify the structure type of tool output."""
        content_stripped = content.strip()
        
        # JSON structure
        if (content_stripped.startswith('{') and content_stripped.endswith('}')) or \
           (content_stripped.startswith('[') and content_stripped.endswith(']')):
            try:
                json.loads(content_stripped)
                return 'json_response'
            except json.JSONDecodeError:
                pass
        
        # Table structure
        if '|' in content and content.count('|') > 2:
            return 'table'
        
        # Log entry
        if any(re.search(pattern, content) for pattern in self.tool_patterns['log_entry'].values()):
            return 'log_entry'
        
        # Command output
        if content.count('\n') > 1 and any(keyword in content.lower() for keyword in ['pid', 'size', 'status', 'name']):
            return 'command_output'
        
        # Error output
        if any(error_word in content.lower() for error_word in ['error', 'exception', 'failed', 'timeout']):
            return 'error_output'
        
        return 'command_output'  # Default
    
    def _has_structured_data(self, content: str) -> bool:
        """Check if content contains structured data."""
        structured_indicators = ['{', '[', '|', ':', '=', '<', '>']
        return any(indicator in content for indicator in structured_indicators)
    
    def _has_error_indicators(self, content: str) -> bool:
        """Check if content contains error indicators."""
        error_indicators = ['error', 'exception', 'failed', 'timeout', 'denied', 'not found']
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in error_indicators)
    
    def _calculate_output_confidence(self, content: str, block_type: str, tool_name: str) -> float:
        """Calculate confidence score for tool output identification."""
        base_confidence = {
            'command': 0.8,
            'structured_data': 0.7,
            'log_entry': 0.6,
            'json_response': 0.9,
            'table': 0.8
        }.get(block_type, 0.5)
        
        # Boost for recognized tools
        if tool_name in self.tool_signatures:
            base_confidence += 0.2
        
        # Boost for structured content
        if self._has_structured_data(content):
            base_confidence += 0.1
        
        # Boost for typical output length
        if 50 < len(content) < 2000:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def trace_dependencies(self, session_turns: List[Dict]) -> Dict[int, List[int]]:
        """
        Trace dependencies between tool invocations and their outputs.
        
        Args:
            session_turns: List of turns in LetheBench format
            
        Returns:
            Dictionary mapping turn IDs to their dependency turn IDs
        """
        dependencies = {}
        
        # Find tool command and output pairs
        for i, turn in enumerate(session_turns):
            turn_id = turn['turn']
            text = turn['text']
            
            # Identify tool outputs
            outputs = self.identify_tool_outputs(text)
            
            if outputs:
                # Look backwards for related commands/setup
                deps = self._find_backward_dependencies(session_turns, i, outputs)
                if deps:
                    dependencies[turn_id] = deps
        
        return dependencies
    
    def _find_backward_dependencies(self, turns: List[Dict], current_idx: int, outputs: List[Dict]) -> List[int]:
        """Find turns that the current outputs depend on."""
        dependencies = []
        
        # Look at previous turns for commands that might have generated these outputs
        for i in range(max(0, current_idx - 5), current_idx):
            prev_turn = turns[i]
            prev_text = prev_turn['text']
            
            # Check for command patterns that might generate current outputs
            for output in outputs:
                tool_name = output['tool_name']
                
                # Look for command invocations
                if tool_name in prev_text.lower():
                    dependencies.append(prev_turn['turn'])
                    break
                
                # Look for setup/configuration commands
                setup_patterns = [
                    r'\$\s+' + re.escape(tool_name),
                    r'run\s+' + re.escape(tool_name),
                    r'execute\s+' + re.escape(tool_name)
                ]
                
                if any(re.search(pattern, prev_text, re.IGNORECASE) for pattern in setup_patterns):
                    dependencies.append(prev_turn['turn'])
                    break
        
        return dependencies
    
    def label_session_turns(self, session_turns: List[Dict]) -> List[ToolChunk]:
        """
        Generate gold tool output chunks for a complete dialog session.
        
        Args:
            session_turns: List of turns in LetheBench format
            
        Returns:
            List of ToolChunk annotations
        """
        chunks = []
        session_id = session_turns[0]['session_id'] if session_turns else "unknown"
        
        # Trace dependencies first
        dependencies = self.trace_dependencies(session_turns)
        
        # Process each turn for tool outputs
        for turn in session_turns:
            outputs = self.identify_tool_outputs(turn['text'])
            
            # Get dependencies for this turn
            turn_deps = dependencies.get(turn['turn'], [])
            
            # Create chunks from outputs
            for i, output in enumerate(outputs):
                if output['confidence'] > 0.5:  # Quality threshold
                    chunk_id = f"{session_id}_turn{turn['turn']}_chunk{i}"
                    
                    chunks.append(ToolChunk(
                        chunk_id=chunk_id,
                        session_id=session_id,
                        turn_id=turn['turn'],
                        content=output['content'],
                        chunk_type=output['type'],
                        tool_name=output['tool_name'],
                        dependencies=turn_deps,
                        context_start=0,  # Would need character-level tracking
                        context_end=len(output['content']),
                        confidence=output['confidence'],
                        metadata={
                            'turn_role': turn['role'],
                            'turn_meta': turn.get('meta', {}),
                            'output_metadata': output['metadata'],
                            'line_range': (output.get('start_line', 0), output.get('end_line', 0))
                        }
                    ))
        
        return chunks

def test_tool_labeler():
    """Test the tool labeler with sample data."""
    labeler = ToolLabeler()
    
    # Test tool output identification
    sample_tool_session = '''
$ docker ps
CONTAINER ID   IMAGE          COMMAND                  CREATED       STATUS       PORTS     NAMES
abc123def456   nginx:latest   "/docker-entrypoint.…"   2 hours ago   Up 2 hours   80/tcp    web-server

$ curl -X GET https://api.example.com/users
HTTP/1.1 200 OK
Content-Type: application/json

{
  "users": [
    {"id": 1, "name": "John Doe", "email": "john@example.com"},
    {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
  ],
  "total": 2
}

$ git log --oneline
abc1234 Add user authentication
def5678 Fix database connection
ghi9012 Initial commit
'''
    
    print("Testing tool output identification...")
    outputs = labeler.identify_tool_outputs(sample_tool_session)
    
    for output in outputs:
        print(f"  {output['type']}: {output['tool_name']} (conf: {output['confidence']:.2f})")
        print(f"    Content preview: {output['content'][:100]}...")
        print()
    
    # Test session labeling
    sample_session = [
        {
            'session_id': 'tool_test_session',
            'turn': 0,
            'role': 'user',
            'text': 'How do I check running Docker containers?',
            'meta': {}
        },
        {
            'session_id': 'tool_test_session',
            'turn': 1,
            'role': 'assistant', 
            'text': 'You can use the docker ps command:\n\n' + sample_tool_session,
            'meta': {}
        }
    ]
    
    print("Testing session labeling...")
    chunks = labeler.label_session_turns(sample_session)
    
    print(f"Generated {len(chunks)} tool chunks:")
    for chunk in chunks:
        print(f"  {chunk.chunk_id}: {chunk.tool_name} ({chunk.chunk_type})")
        print(f"    Dependencies: {chunk.dependencies}")
        print(f"    Content: {chunk.content[:100]}...")
        print()

if __name__ == "__main__":
    test_tool_labeler()