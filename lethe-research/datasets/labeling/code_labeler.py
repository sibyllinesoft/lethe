#!/usr/bin/env python3
"""
Code Gold Labeler for LetheBench

Implements weak supervision for extracting gold annotations from code discussions.
Identifies code symbols, function names, file paths, and technical references
from accepted answers and merged pull requests.

For NeurIPS LetheBench Paper - Dataset Construction Pipeline
"""

import ast
import re
from typing import List, Dict, Set, Tuple, Optional
import logging
from dataclasses import dataclass

@dataclass
class CodeChunk:
    """Represents a gold code chunk annotation."""
    chunk_id: str
    session_id: str
    turn_id: int
    content: str
    chunk_type: str  # 'function', 'class', 'variable', 'import', 'file_path', 'error_message'
    context_start: int
    context_end: int
    confidence: float
    metadata: Dict

class CodeLabeler:
    """
    Generates gold annotations for code-centric dialog sessions.
    
    Uses weak supervision to identify:
    - Function and class definitions/references
    - Variable names and API calls
    - File paths and module imports
    - Error messages and stack traces
    - Code symbols mentioned in accepted answers
    """
    
    def __init__(self):
        """Initialize code labeler with pattern recognition."""
        self.logger = logging.getLogger(__name__)
        
        # Code patterns for different languages
        self.code_patterns = {
            'function_def': {
                'python': r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
                'javascript': r'function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
                'java': r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
                'cpp': r'(?:\w+\s+)*([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{'
            },
            'class_def': {
                'python': r'class\s+([A-Za-z_][A-Za-z0-9_]*)\s*[\(:]',
                'javascript': r'class\s+([A-Za-z_][A-Za-z0-9_]*)',
                'java': r'(?:public|private)?\s*class\s+([A-Za-z_][A-Za-z0-9_]*)',
                'cpp': r'class\s+([A-Za-z_][A-Za-z0-9_]*)'
            },
            'import_statement': {
                'python': r'(?:from\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s+)?import\s+([a-zA-Z_][a-zA-Z0-9_., ]*)',
                'javascript': r'import\s+.*?\s+from\s+[\'"]([^\'\"]+)[\'"]',
                'java': r'import\s+([a-zA-Z_][a-zA-Z0-9_.]*);'
            },
            'file_path': r'([a-zA-Z_][a-zA-Z0-9_./\\-]*\.(?:py|js|java|cpp|h|ts|jsx|tsx|go|rs|rb|php))',
            'api_call': r'([a-zA-Z_][a-zA-Z0-9_.]*)\s*\(',
            'variable_assignment': r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=',
            'error_class': r'([A-Za-z_][A-Za-z0-9_]*Error|[A-Za-z_][A-Za-z0-9_]*Exception)',
            'stack_trace': r'File "([^"]+)", line (\d+)',
            'git_reference': r'(?:commit|PR|pull request)\s+([a-f0-9]{7,40})',
            'issue_reference': r'#(\d+)',
            'code_block': r'```(?:(\w+)\n)?(.*?)```'
        }
        
        # Language keywords to filter out common false positives
        self.language_keywords = {
            'python': {'def', 'class', 'import', 'from', 'if', 'else', 'elif', 'for', 'while', 
                      'try', 'except', 'finally', 'with', 'as', 'pass', 'break', 'continue',
                      'return', 'yield', 'lambda', 'and', 'or', 'not', 'in', 'is', 'True', 'False', 'None'},
            'javascript': {'function', 'var', 'let', 'const', 'if', 'else', 'for', 'while', 
                          'do', 'try', 'catch', 'finally', 'return', 'break', 'continue',
                          'true', 'false', 'null', 'undefined', 'this', 'new', 'typeof'},
            'java': {'public', 'private', 'protected', 'static', 'final', 'class', 'interface',
                    'extends', 'implements', 'if', 'else', 'for', 'while', 'do', 'try', 'catch',
                    'finally', 'return', 'break', 'continue', 'true', 'false', 'null'}
        }
    
    def extract_code_symbols(self, text: str, language_hint: Optional[str] = None) -> List[Dict]:
        """
        Extract code symbols from text using pattern matching and AST analysis.
        
        Args:
            text: Text content to analyze
            language_hint: Programming language hint for better parsing
            
        Returns:
            List of code symbol dictionaries with metadata
        """
        symbols = []
        
        # Auto-detect language if not provided
        if not language_hint:
            language_hint = self._detect_language(text)
        
        # Extract code blocks first
        code_blocks = []
        for match in re.finditer(self.code_patterns['code_block'], text, re.DOTALL):
            block_language = match.group(1) or language_hint
            block_code = match.group(2)
            code_blocks.append({
                'language': block_language,
                'code': block_code,
                'start': match.start(),
                'end': match.end()
            })
        
        # Analyze each code block
        for block in code_blocks:
            block_symbols = self._extract_from_code_block(
                block['code'], 
                block['language'],
                block['start']
            )
            symbols.extend(block_symbols)
        
        # Extract inline code references
        inline_symbols = self._extract_inline_references(text, language_hint)
        symbols.extend(inline_symbols)
        
        return symbols
    
    def _detect_language(self, text: str) -> str:
        """Detect programming language from text content."""
        language_indicators = {
            'python': ['def ', 'import ', 'from ', '.py', 'python', 'pip install'],
            'javascript': ['function ', 'var ', 'let ', 'const ', '.js', 'npm install', 'node'],
            'java': ['public class', 'private ', 'public static void main', '.java'],
            'cpp': ['#include', 'using namespace', '.cpp', '.h', 'std::'],
            'go': ['package ', 'func ', 'import ', '.go', 'go mod'],
            'rust': ['fn ', 'let mut', 'use ', '.rs', 'cargo']
        }
        
        scores = {}
        text_lower = text.lower()
        
        for language, indicators in language_indicators.items():
            score = sum(text_lower.count(indicator) for indicator in indicators)
            scores[language] = score
        
        return max(scores, key=scores.get) if scores else 'python'
    
    def _extract_from_code_block(self, code: str, language: str, offset: int) -> List[Dict]:
        """Extract symbols from a specific code block."""
        symbols = []
        
        if not code.strip():
            return symbols
        
        # Try AST parsing for Python
        if language == 'python':
            try:
                tree = ast.parse(code)
                ast_symbols = self._extract_from_ast(tree, code, offset)
                symbols.extend(ast_symbols)
            except SyntaxError:
                # Fall back to regex if AST parsing fails
                pass
        
        # Use regex patterns for all languages
        regex_symbols = self._extract_with_regex(code, language, offset)
        symbols.extend(regex_symbols)
        
        return symbols
    
    def _extract_from_ast(self, tree: ast.AST, code: str, offset: int) -> List[Dict]:
        """Extract symbols using Python AST analysis."""
        symbols = []
        lines = code.split('\n')
        
        class SymbolVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                symbols.append({
                    'type': 'function',
                    'name': node.name,
                    'line': node.lineno,
                    'col_offset': node.col_offset,
                    'confidence': 0.95,
                    'context': self._get_line_context(lines, node.lineno - 1)
                })
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                symbols.append({
                    'type': 'class',
                    'name': node.name,
                    'line': node.lineno,
                    'col_offset': node.col_offset,
                    'confidence': 0.95,
                    'context': self._get_line_context(lines, node.lineno - 1)
                })
                self.generic_visit(node)
            
            def visit_Import(self, node):
                for alias in node.names:
                    symbols.append({
                        'type': 'import',
                        'name': alias.name,
                        'line': node.lineno,
                        'col_offset': node.col_offset,
                        'confidence': 0.90,
                        'context': self._get_line_context(lines, node.lineno - 1)
                    })
                self.generic_visit(node)
            
            def visit_ImportFrom(self, node):
                module = node.module or ''
                for alias in node.names:
                    symbols.append({
                        'type': 'import',
                        'name': f"{module}.{alias.name}" if module else alias.name,
                        'line': node.lineno,
                        'col_offset': node.col_offset,
                        'confidence': 0.90,
                        'context': self._get_line_context(lines, node.lineno - 1)
                    })
                self.generic_visit(node)
            
            def _get_line_context(self, lines, line_idx):
                start = max(0, line_idx - 2)
                end = min(len(lines), line_idx + 3)
                return '\n'.join(lines[start:end])
        
        visitor = SymbolVisitor()
        visitor.visit(tree)
        
        return symbols
    
    def _extract_with_regex(self, code: str, language: str, offset: int) -> List[Dict]:
        """Extract symbols using regex patterns."""
        symbols = []
        
        # Get language-specific patterns
        patterns = {}
        for pattern_type, lang_patterns in self.code_patterns.items():
            if isinstance(lang_patterns, dict) and language in lang_patterns:
                patterns[pattern_type] = lang_patterns[language]
            elif isinstance(lang_patterns, str):
                patterns[pattern_type] = lang_patterns
        
        # Apply patterns
        for pattern_type, pattern in patterns.items():
            for match in re.finditer(pattern, code, re.MULTILINE):
                symbol_name = match.group(1) if match.groups() else match.group(0)
                
                # Filter out language keywords
                keywords = self.language_keywords.get(language, set())
                if symbol_name.lower() in keywords:
                    continue
                
                # Calculate confidence based on context
                confidence = self._calculate_confidence(match, code, pattern_type)
                
                if confidence > 0.3:  # Minimum confidence threshold
                    symbols.append({
                        'type': pattern_type,
                        'name': symbol_name,
                        'start': match.start() + offset,
                        'end': match.end() + offset,
                        'confidence': confidence,
                        'context': self._get_match_context(code, match)
                    })
        
        return symbols
    
    def _extract_inline_references(self, text: str, language: str) -> List[Dict]:
        """Extract inline code references from natural language text."""
        symbols = []
        
        # Look for code references in backticks
        inline_code_pattern = r'`([^`]+)`'
        for match in re.finditer(inline_code_pattern, text):
            code_snippet = match.group(1)
            
            # Check if it looks like a code symbol
            if self._is_code_symbol(code_snippet, language):
                symbols.append({
                    'type': 'inline_reference',
                    'name': code_snippet,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.7,
                    'context': self._get_surrounding_text(text, match.start(), match.end(), 100)
                })
        
        return symbols
    
    def _is_code_symbol(self, text: str, language: str) -> bool:
        """Heuristic to determine if text looks like a code symbol."""
        # Basic checks for code-like patterns
        if len(text) > 50:  # Too long to be a simple symbol
            return False
        
        # Contains programming constructs
        code_indicators = ['()', '.', '_', '[]', '{}', '=', '->', '=>', '::']
        has_indicator = any(indicator in text for indicator in code_indicators)
        
        # Matches identifier pattern
        identifier_pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*(?:\(\))?$'
        is_identifier = re.match(identifier_pattern, text)
        
        return has_indicator or is_identifier
    
    def _calculate_confidence(self, match: re.Match, text: str, pattern_type: str) -> float:
        """Calculate confidence score for a symbol match."""
        base_confidence = {
            'function_def': 0.9,
            'class_def': 0.9,
            'import_statement': 0.8,
            'api_call': 0.6,
            'variable_assignment': 0.5,
            'file_path': 0.8,
            'error_class': 0.8
        }.get(pattern_type, 0.5)
        
        # Adjust based on context
        context = self._get_match_context(text, match, window=50)
        
        # Boost confidence if in code block
        if '```' in context:
            base_confidence += 0.2
        
        # Boost if surrounded by technical discussion
        technical_terms = ['function', 'method', 'class', 'variable', 'error', 'exception', 
                          'import', 'module', 'library', 'API', 'call', 'return']
        technical_count = sum(term.lower() in context.lower() for term in technical_terms)
        base_confidence += min(0.3, technical_count * 0.05)
        
        return min(1.0, base_confidence)
    
    def _get_match_context(self, text: str, match: re.Match, window: int = 100) -> str:
        """Get surrounding context for a regex match."""
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        return text[start:end]
    
    def _get_surrounding_text(self, text: str, start: int, end: int, window: int) -> str:
        """Get surrounding text for a position range."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def label_session_turns(self, session_turns: List[Dict]) -> List[CodeChunk]:
        """
        Generate gold code chunks for a complete dialog session.
        
        Args:
            session_turns: List of turns in LetheBench format
            
        Returns:
            List of CodeChunk annotations
        """
        chunks = []
        session_id = session_turns[0]['session_id'] if session_turns else "unknown"
        
        # Identify accepted answers and high-value turns
        high_value_turns = self._identify_high_value_turns(session_turns)
        
        for turn in high_value_turns:
            turn_symbols = self.extract_code_symbols(
                turn['text'], 
                self._get_language_hint(turn)
            )
            
            # Convert symbols to chunks
            for symbol in turn_symbols:
                if symbol['confidence'] > 0.5:  # Quality threshold
                    chunk_id = f"{session_id}_turn{turn['turn']}_chunk{len(chunks)}"
                    
                    chunks.append(CodeChunk(
                        chunk_id=chunk_id,
                        session_id=session_id,
                        turn_id=turn['turn'],
                        content=symbol['name'],
                        chunk_type=symbol['type'],
                        context_start=symbol.get('start', 0),
                        context_end=symbol.get('end', len(symbol['name'])),
                        confidence=symbol['confidence'],
                        metadata={
                            'language': self._get_language_hint(turn),
                            'symbol_context': symbol.get('context', ''),
                            'turn_role': turn['role'],
                            'turn_meta': turn.get('meta', {})
                        }
                    ))
        
        return chunks
    
    def _identify_high_value_turns(self, turns: List[Dict]) -> List[Dict]:
        """Identify turns most likely to contain valuable code information."""
        high_value = []
        
        for turn in turns:
            # Check for indicators of high-value content
            meta = turn.get('meta', {})
            text = turn['text']
            
            # Accepted answers have highest value
            if meta.get('is_accepted', False):
                high_value.append(turn)
                continue
            
            # High-scored answers
            if meta.get('score', 0) > 5:
                high_value.append(turn)
                continue
            
            # Contains substantial code
            code_block_count = text.count('```')
            if code_block_count >= 2:  # At least one complete code block
                high_value.append(turn)
                continue
            
            # Contains multiple inline code references
            inline_code_count = text.count('`')
            if inline_code_count >= 4:  # At least 2 inline references
                high_value.append(turn)
                continue
            
            # Long technical responses
            if len(text) > 500 and turn['role'] == 'assistant':
                high_value.append(turn)
                continue
        
        return high_value
    
    def _get_language_hint(self, turn: Dict) -> str:
        """Get programming language hint from turn metadata."""
        meta = turn.get('meta', {})
        
        # From tags (Stack Overflow)
        tags = meta.get('tags', [])
        language_tags = {'python', 'javascript', 'java', 'c++', 'cpp', 'go', 'rust', 'php', 'ruby'}
        for tag in tags:
            if tag.lower() in language_tags:
                return tag.lower()
        
        # From repository language (GitHub)
        if 'repository' in meta:
            # Could derive from repository metadata
            pass
        
        # Default detection from content
        return self._detect_language(turn['text'])

def test_code_labeler():
    """Test the code labeler with sample data."""
    labeler = CodeLabeler()
    
    # Test code symbol extraction
    sample_code = '''
```python
def calculate_metrics(data, threshold=0.5):
    """Calculate accuracy and F1 score."""
    from sklearn.metrics import accuracy_score, f1_score
    
    predictions = model.predict(data)
    accuracy = accuracy_score(y_true, predictions)
    
    if accuracy > threshold:
        return {"accuracy": accuracy, "f1": f1_score(y_true, predictions)}
    else:
        raise ValueError("Accuracy below threshold")
```

The `calculate_metrics` function uses `sklearn.metrics` to compute scores.
You can call it like `calculate_metrics(test_data, 0.8)` for validation.
'''
    
    print("Testing code symbol extraction...")
    symbols = labeler.extract_code_symbols(sample_code, 'python')
    
    for symbol in symbols:
        print(f"  {symbol['type']}: {symbol['name']} (conf: {symbol['confidence']:.2f})")
    
    # Test session labeling
    sample_session = [
        {
            'session_id': 'test_session',
            'turn': 0,
            'role': 'user',
            'text': 'How do I calculate accuracy in scikit-learn?',
            'meta': {}
        },
        {
            'session_id': 'test_session', 
            'turn': 1,
            'role': 'assistant',
            'text': sample_code,
            'meta': {'is_accepted': True, 'score': 15, 'tags': ['python', 'scikit-learn']}
        }
    ]
    
    print(f"\nTesting session labeling...")
    chunks = labeler.label_session_turns(sample_session)
    
    print(f"Generated {len(chunks)} code chunks:")
    for chunk in chunks:
        print(f"  {chunk.chunk_id}: {chunk.content} ({chunk.chunk_type})")

if __name__ == "__main__":
    test_code_labeler()