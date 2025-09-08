#!/usr/bin/env python3
"""
Format Validator for LetheBench

Validates JSONL format compliance and data structure consistency
across all LetheBench dataset files.

For NeurIPS LetheBench Paper - Dataset Construction Pipeline
"""

import json
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ValidationResult:
    """Results from format validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]

class FormatValidator:
    """
    Validates LetheBench JSONL format compliance.
    
    Ensures:
    - Valid JSONL structure (one JSON object per line)
    - Required fields present: session_id, turn, role, text, ts, meta
    - Field type consistency
    - Session integrity (sequential turns, consistent session_ids)
    - Genre-specific requirements met
    """
    
    def __init__(self):
        """Initialize format validator."""
        self.logger = logging.getLogger(__name__)
        
        # Required fields for all turns
        self.required_fields = {
            'session_id': str,
            'turn': int, 
            'role': str,
            'text': str,
            'ts': str,
            'meta': dict
        }
        
        # Valid role values
        self.valid_roles = {'user', 'assistant', 'system'}
        
        # Genre-specific meta field requirements
        self.genre_meta_requirements = {
            'code': {
                'required': ['license', 'source'],
                'optional': ['tags', 'repository', 'language', 'is_accepted', 'score']
            },
            'tool': {
                'required': ['source'],
                'optional': ['tool_name', 'dependencies', 'command_type', 'output_type']
            },
            'prose': {
                'required': ['license', 'source'],
                'optional': ['topics', 'speakers', 'date', 'event_type']
            }
        }
    
    def validate_file(self, file_path: Path, genre: str) -> ValidationResult:
        """
        Validate a single JSONL file.
        
        Args:
            file_path: Path to JSONL file
            genre: Expected genre ('code', 'tool', 'prose')
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        stats = {
            'total_lines': 0,
            'valid_json_lines': 0,
            'unique_sessions': set(),
            'role_counts': {'user': 0, 'assistant': 0, 'system': 0},
            'turn_counts': {},
            'avg_text_length': 0,
            'total_text_length': 0
        }
        
        if not file_path.exists():
            errors.append(f"File does not exist: {file_path}")
            return ValidationResult(False, errors, warnings, stats)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    stats['total_lines'] += 1
                    
                    # Parse JSON
                    try:
                        turn_data = json.loads(line.strip())
                        stats['valid_json_lines'] += 1
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {line_num}: Invalid JSON - {e}")
                        continue
                    
                    # Validate turn structure
                    turn_errors, turn_warnings = self._validate_turn(
                        turn_data, line_num, genre
                    )
                    errors.extend(turn_errors)
                    warnings.extend(turn_warnings)
                    
                    # Update statistics
                    if not turn_errors:  # Only count valid turns in stats
                        self._update_stats(turn_data, stats)
        
        except Exception as e:
            errors.append(f"Failed to read file: {e}")
            return ValidationResult(False, errors, warnings, stats)
        
        # Finalize statistics
        if stats['valid_json_lines'] > 0:
            stats['avg_text_length'] = stats['total_text_length'] / stats['valid_json_lines']
        
        stats['unique_sessions'] = len(stats['unique_sessions'])
        
        # Validate session integrity
        session_errors = self._validate_session_integrity(file_path, genre)
        errors.extend(session_errors)
        
        is_valid = len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, stats)
    
    def _validate_turn(self, turn_data: Dict, line_num: int, genre: str) -> tuple[List[str], List[str]]:
        """Validate individual turn structure."""
        errors = []
        warnings = []
        
        # Check required fields
        for field, expected_type in self.required_fields.items():
            if field not in turn_data:
                errors.append(f"Line {line_num}: Missing required field '{field}'")
                continue
            
            if not isinstance(turn_data[field], expected_type):
                errors.append(
                    f"Line {line_num}: Field '{field}' should be {expected_type.__name__}, "
                    f"got {type(turn_data[field]).__name__}"
                )
        
        # Validate specific field values
        if 'role' in turn_data:
            if turn_data['role'] not in self.valid_roles:
                errors.append(
                    f"Line {line_num}: Invalid role '{turn_data['role']}'. "
                    f"Must be one of: {', '.join(self.valid_roles)}"
                )
        
        if 'turn' in turn_data:
            if turn_data['turn'] < 0:
                errors.append(f"Line {line_num}: Turn number must be non-negative")
        
        if 'text' in turn_data:
            if not turn_data['text'].strip():
                warnings.append(f"Line {line_num}: Empty text field")
            elif len(turn_data['text']) > 50000:
                warnings.append(f"Line {line_num}: Very long text ({len(turn_data['text'])} chars)")
        
        # Validate timestamp format
        if 'ts' in turn_data:
            if not self._is_valid_timestamp(turn_data['ts']):
                errors.append(f"Line {line_num}: Invalid timestamp format '{turn_data['ts']}'")
        
        # Validate meta field
        if 'meta' in turn_data:
            meta_errors, meta_warnings = self._validate_meta_field(
                turn_data['meta'], line_num, genre
            )
            errors.extend(meta_errors)
            warnings.extend(meta_warnings)
        
        return errors, warnings
    
    def _validate_meta_field(self, meta: Dict, line_num: int, genre: str) -> tuple[List[str], List[str]]:
        """Validate meta field structure."""
        errors = []
        warnings = []
        
        if not isinstance(meta, dict):
            errors.append(f"Line {line_num}: Meta field must be a dictionary")
            return errors, warnings
        
        # Check genre-specific requirements
        if genre in self.genre_meta_requirements:
            requirements = self.genre_meta_requirements[genre]
            
            # Check required fields
            for required_field in requirements['required']:
                if required_field not in meta:
                    errors.append(
                        f"Line {line_num}: Meta missing required field '{required_field}' for genre '{genre}'"
                    )
            
            # Check for unexpected fields
            all_expected = set(requirements['required'] + requirements['optional'])
            unexpected = set(meta.keys()) - all_expected
            
            if unexpected:
                warnings.append(
                    f"Line {line_num}: Meta contains unexpected fields for genre '{genre}': "
                    f"{', '.join(unexpected)}"
                )
        
        return errors, warnings
    
    def _is_valid_timestamp(self, ts: str) -> bool:
        """Check if timestamp is in valid ISO format."""
        import datetime
        
        # Common valid formats
        formats = [
            '%Y-%m-%dT%H:%M:%S.%fZ',  # ISO with microseconds and Z
            '%Y-%m-%dT%H:%M:%SZ',     # ISO with Z
            '%Y-%m-%dT%H:%M:%S',      # ISO without Z
            '%Y-%m-%d %H:%M:%S',      # Space separated
            '%Y-%m-%d',               # Date only
        ]
        
        for fmt in formats:
            try:
                datetime.datetime.strptime(ts, fmt)
                return True
            except ValueError:
                continue
        
        return False
    
    def _update_stats(self, turn_data: Dict, stats: Dict):
        """Update statistics with turn data."""
        if 'session_id' in turn_data:
            stats['unique_sessions'].add(turn_data['session_id'])
        
        if 'role' in turn_data and turn_data['role'] in stats['role_counts']:
            stats['role_counts'][turn_data['role']] += 1
        
        if 'turn' in turn_data:
            turn_num = turn_data['turn']
            stats['turn_counts'][turn_num] = stats['turn_counts'].get(turn_num, 0) + 1
        
        if 'text' in turn_data:
            text_len = len(turn_data['text'])
            stats['total_text_length'] += text_len
    
    def _validate_session_integrity(self, file_path: Path, genre: str) -> List[str]:
        """Validate that sessions have proper turn sequences."""
        errors = []
        sessions = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        turn_data = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON (already reported)
                    
                    if 'session_id' not in turn_data or 'turn' not in turn_data:
                        continue  # Skip incomplete turns (already reported)
                    
                    session_id = turn_data['session_id']
                    turn_num = turn_data['turn']
                    
                    if session_id not in sessions:
                        sessions[session_id] = {'turns': {}, 'first_line': line_num}
                    
                    if turn_num in sessions[session_id]['turns']:
                        errors.append(
                            f"Duplicate turn {turn_num} in session '{session_id}' "
                            f"(line {line_num}, previously at line {sessions[session_id]['turns'][turn_num]})"
                        )
                    else:
                        sessions[session_id]['turns'][turn_num] = line_num
        
        except Exception:
            # File reading errors already handled in main validation
            pass
        
        # Check turn sequence completeness
        for session_id, session_data in sessions.items():
            turns = list(session_data['turns'].keys())
            if not turns:
                continue
            
            turns.sort()
            
            # Check if turns start at 0
            if turns[0] != 0:
                errors.append(
                    f"Session '{session_id}' turns should start at 0, starts at {turns[0]}"
                )
            
            # Check for gaps in turn sequence  
            for i in range(1, len(turns)):
                if turns[i] != turns[i-1] + 1:
                    errors.append(
                        f"Session '{session_id}' has gap in turn sequence: "
                        f"{turns[i-1]} -> {turns[i]}"
                    )
        
        return errors
    
    def validate_dataset_splits(self, 
                               dataset_dir: Path, 
                               genre: str) -> Dict[str, ValidationResult]:
        """
        Validate all splits for a genre (train, dev, test).
        
        Args:
            dataset_dir: Directory containing the dataset files
            genre: Genre to validate ('code', 'tool', 'prose')
            
        Returns:
            Dictionary mapping split names to ValidationResults
        """
        results = {}
        splits = ['train', 'dev', 'test']
        
        for split in splits:
            split_file = dataset_dir / genre / f"{split}.jsonl"
            results[split] = self.validate_file(split_file, genre)
        
        return results
    
    def generate_validation_report(self, 
                                 validation_results: Dict[str, ValidationResult],
                                 output_file: Optional[Path] = None) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            validation_results: Results from validate_dataset_splits
            output_file: Optional file to write report to
            
        Returns:
            Report text
        """
        report_lines = [
            "# LetheBench Format Validation Report",
            f"Generated: {self._get_timestamp()}\n"
        ]
        
        # Summary
        total_errors = sum(len(result.errors) for result in validation_results.values())
        total_warnings = sum(len(result.warnings) for result in validation_results.values())
        
        report_lines.extend([
            "## Summary",
            f"- Total Errors: {total_errors}",
            f"- Total Warnings: {total_warnings}",
            f"- Overall Status: {'✓ PASS' if total_errors == 0 else '✗ FAIL'}\n"
        ])
        
        # Detailed results by split
        for split_name, result in validation_results.items():
            report_lines.extend([
                f"## Split: {split_name}",
                f"- Status: {'✓ PASS' if result.is_valid else '✗ FAIL'}",
                f"- Errors: {len(result.errors)}",
                f"- Warnings: {len(result.warnings)}",
                f"- Total Lines: {result.stats.get('total_lines', 0)}",
                f"- Valid JSON Lines: {result.stats.get('valid_json_lines', 0)}",
                f"- Unique Sessions: {result.stats.get('unique_sessions', 0)}",
                f"- Average Text Length: {result.stats.get('avg_text_length', 0):.1f} chars",
            ])
            
            # Role distribution
            role_counts = result.stats.get('role_counts', {})
            if role_counts:
                report_lines.append("- Role Distribution:")
                for role, count in role_counts.items():
                    report_lines.append(f"  - {role}: {count}")
            
            # Errors
            if result.errors:
                report_lines.append("\n### Errors:")
                for error in result.errors[:20]:  # Limit to first 20 errors
                    report_lines.append(f"- {error}")
                
                if len(result.errors) > 20:
                    report_lines.append(f"... and {len(result.errors) - 20} more errors")
            
            # Warnings
            if result.warnings:
                report_lines.append("\n### Warnings:")
                for warning in result.warnings[:20]:  # Limit to first 20 warnings
                    report_lines.append(f"- {warning}")
                
                if len(result.warnings) > 20:
                    report_lines.append(f"... and {len(result.warnings) - 20} more warnings")
            
            report_lines.append("")  # Blank line between splits
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for reports."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def test_format_validator():
    """Test the format validator with sample data."""
    validator = FormatValidator()
    
    # Create sample JSONL data
    sample_turns = [
        {
            'session_id': 'test_session_1',
            'turn': 0,
            'role': 'user',
            'text': 'How do I implement a binary search?',
            'ts': '2024-01-15T10:30:00Z',
            'meta': {
                'license': 'CC BY-SA 4.0',
                'source': 'stackoverflow',
                'tags': ['python', 'algorithm']
            }
        },
        {
            'session_id': 'test_session_1',
            'turn': 1,
            'role': 'assistant',
            'text': 'Here is a Python implementation of binary search:\n\n```python\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```',
            'ts': '2024-01-15T10:31:22Z',
            'meta': {
                'license': 'CC BY-SA 4.0',
                'source': 'stackoverflow',
                'is_accepted': True,
                'score': 15
            }
        }
    ]
    
    # Test individual turn validation
    print("Testing turn validation...")
    for i, turn in enumerate(sample_turns):
        errors, warnings = validator._validate_turn(turn, i+1, 'code')
        print(f"Turn {i}: {len(errors)} errors, {len(warnings)} warnings")
        for error in errors:
            print(f"  Error: {error}")
        for warning in warnings:
            print(f"  Warning: {warning}")
    
    # Test with invalid data
    invalid_turn = {
        'session_id': 'test_session_2',
        'turn': -1,  # Invalid negative turn
        'role': 'invalid_role',  # Invalid role
        'text': '',  # Empty text (warning)
        'ts': 'invalid_timestamp',  # Invalid timestamp
        'meta': 'not_a_dict'  # Should be dict
    }
    
    print(f"\nTesting invalid turn...")
    errors, warnings = validator._validate_turn(invalid_turn, 3, 'code')
    print(f"Invalid turn: {len(errors)} errors, {len(warnings)} warnings")
    for error in errors:
        print(f"  Error: {error}")

if __name__ == "__main__":
    test_format_validator()