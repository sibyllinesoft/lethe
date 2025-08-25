#!/usr/bin/env python3
"""
LetheBench Privacy Redaction System

Implements deterministic, comprehensive privacy redaction for dataset construction.
Ensures compliance with privacy requirements while maintaining data utility.

For NeurIPS LetheBench Paper - Dataset Construction Pipeline
"""

import re
import hashlib
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import logging

@dataclass 
class RedactionResult:
    """Results from privacy redaction process."""
    original_text: str
    redacted_text: str
    redactions_made: List[Tuple[str, str]]  # (pattern_type, redacted_content)
    redaction_count: int

class PrivacyRedactor:
    """
    Deterministic privacy redaction system for LetheBench dataset construction.
    
    Features:
    - Comprehensive pattern matching for PII and sensitive data
    - Deterministic replacement (same input = same output)
    - Maintains format integrity for code/structured data
    - Audit logging for compliance verification
    """
    
    def __init__(self, salt: str = "lethebench_neurips_2024"):
        """Initialize redactor with deterministic salt for consistent hashing."""
        self.salt = salt
        self.redaction_log: List[Dict] = []
        
        # Comprehensive redaction patterns
        self.patterns = {
            'email': {
                'regex': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'replacement': lambda m: f"email_{self._deterministic_hash(m.group())}@redacted.com"
            },
            'github_token': {
                'regex': r'gh[pousr]_[A-Za-z0-9_]{36}',
                'replacement': lambda m: f"ghp_{'x' * 36}"
            },
            'api_key_generic': {
                'regex': r'\b[Aa][Pp][Ii]_?[Kk][Ee][Yy]\s*[=:]\s*[\'"]?([A-Za-z0-9_\-]{20,})[\'"]?',
                'replacement': lambda m: m.group().replace(m.group(1), 'x' * len(m.group(1)))
            },
            'aws_access_key': {
                'regex': r'AKIA[0-9A-Z]{16}',
                'replacement': lambda m: f"AKIA{'X' * 16}"
            },
            'aws_secret_key': {
                'regex': r'[0-9a-zA-Z/+]{40}',
                'replacement': lambda m: 'x' * 40 if self._looks_like_aws_secret(m.group()) else m.group()
            },
            'ip_address': {
                'regex': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                'replacement': lambda m: self._redact_ip(m.group())
            },
            'phone_us': {
                'regex': r'\b(?:\+1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                'replacement': lambda m: f"+1-{self._deterministic_hash(m.group(), 3)[:3]}-{self._deterministic_hash(m.group(), 3)[3:6]}-{self._deterministic_hash(m.group(), 4)[:4]}"
            },
            'ssn': {
                'regex': r'\b[0-9]{3}-?[0-9]{2}-?[0-9]{4}\b',
                'replacement': lambda m: f"{self._deterministic_hash(m.group(), 3)[:3]}-{self._deterministic_hash(m.group(), 2)[:2]}-{self._deterministic_hash(m.group(), 4)[:4]}"
            },
            'credit_card': {
                'regex': r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
                'replacement': lambda m: 'x' * len(m.group())
            },
            'github_url_with_token': {
                'regex': r'https://([a-zA-Z0-9_\-]+):([a-zA-Z0-9_\-]+)@github\.com',
                'replacement': lambda m: f"https://user:token@github.com"
            },
            'jwt_token': {
                'regex': r'eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*',
                'replacement': lambda m: f"eyJ{'x' * (len(m.group()) - 3)}"
            },
            'bearer_token': {
                'regex': r'Bearer\s+([A-Za-z0-9_\-\.]{20,})',
                'replacement': lambda m: f"Bearer {'x' * len(m.group(1))}"
            },
            'slack_token': {
                'regex': r'xox[baprs]-([A-Za-z0-9_\-]{10,})',
                'replacement': lambda m: f"xox{m.group().split('-')[0][-1]}-{'x' * len(m.group(1))}"
            }
        }
    
    def _deterministic_hash(self, text: str, length: int = 8) -> str:
        """Generate deterministic hash of specified length."""
        hash_input = f"{self.salt}:{text}".encode()
        hash_obj = hashlib.sha256(hash_input)
        hex_hash = hash_obj.hexdigest()
        
        # Convert to alphanumeric for better compatibility
        result = ""
        for i in range(0, min(len(hex_hash), length * 2), 2):
            byte_val = int(hex_hash[i:i+2], 16)
            if byte_val < 26:
                result += chr(ord('a') + byte_val)
            elif byte_val < 52:
                result += chr(ord('A') + byte_val - 26)
            else:
                result += chr(ord('0') + (byte_val % 10))
        
        return result[:length]
    
    def _looks_like_aws_secret(self, text: str) -> bool:
        """Heuristic to identify potential AWS secret keys."""
        if len(text) != 40:
            return False
        # Check for mix of upper/lower case and some special chars
        has_upper = any(c.isupper() for c in text)
        has_lower = any(c.islower() for c in text)
        has_special = any(c in '/+' for c in text)
        return has_upper and has_lower and has_special
    
    def _redact_ip(self, ip: str) -> str:
        """Redact IP address while preserving private vs public classification."""
        parts = ip.split('.')
        if parts[0] in ['10', '172', '192']:  # Private IP ranges
            return f"{parts[0]}.xxx.xxx.xxx"
        else:  # Public IP
            return "xxx.xxx.xxx.xxx"
    
    def redact_text(self, text: str, preserve_structure: bool = True) -> RedactionResult:
        """
        Apply comprehensive privacy redaction to text.
        
        Args:
            text: Input text to redact
            preserve_structure: Whether to preserve code/JSON structure
            
        Returns:
            RedactionResult with original, redacted text and metadata
        """
        redacted = text
        redactions_made = []
        
        for pattern_name, pattern_config in self.patterns.items():
            matches = list(re.finditer(pattern_config['regex'], redacted, re.IGNORECASE))
            
            for match in matches:
                original_content = match.group()
                replacement = pattern_config['replacement'](match)
                
                # Log redaction for audit
                self.redaction_log.append({
                    'pattern_type': pattern_name,
                    'original_length': len(original_content),
                    'redacted_length': len(replacement),
                    'context_start': max(0, match.start() - 20),
                    'context_end': min(len(redacted), match.end() + 20)
                })
                
                redacted = redacted[:match.start()] + replacement + redacted[match.end():]
                redactions_made.append((pattern_name, original_content))
                
                # Update subsequent match positions
                length_diff = len(replacement) - len(original_content)
                for remaining_match in matches[matches.index(match) + 1:]:
                    if hasattr(remaining_match, '_start'):
                        remaining_match._start += length_diff
                        remaining_match._end += length_diff
        
        return RedactionResult(
            original_text=text,
            redacted_text=redacted,
            redactions_made=redactions_made,
            redaction_count=len(redactions_made)
        )
    
    def validate_redaction(self, result: RedactionResult) -> Dict[str, bool]:
        """
        Validate that redaction was successful and complete.
        
        Returns:
            Dictionary of validation checks and their pass/fail status
        """
        checks = {}
        
        # Check for remaining PII patterns
        checks['no_emails_remaining'] = not re.search(self.patterns['email']['regex'], 
                                                     result.redacted_text, re.IGNORECASE)
        
        checks['no_api_keys_remaining'] = not re.search(self.patterns['api_key_generic']['regex'], 
                                                       result.redacted_text, re.IGNORECASE)
        
        checks['no_phone_numbers_remaining'] = not re.search(self.patterns['phone_us']['regex'], 
                                                           result.redacted_text)
        
        # Check format preservation for structured data
        if '{' in result.original_text and '}' in result.original_text:
            checks['json_structure_preserved'] = (result.original_text.count('{') == 
                                                result.redacted_text.count('{'))
        
        if 'def ' in result.original_text or 'function ' in result.original_text:
            checks['code_structure_preserved'] = ('def ' in result.redacted_text or 
                                                'function ' in result.redacted_text)
        
        return checks
    
    def get_redaction_stats(self) -> Dict:
        """Get statistics about redactions performed."""
        if not self.redaction_log:
            return {}
        
        pattern_counts = {}
        total_redactions = len(self.redaction_log)
        
        for entry in self.redaction_log:
            pattern_type = entry['pattern_type']
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        return {
            'total_redactions': total_redactions,
            'patterns_found': pattern_counts,
            'most_common_pattern': max(pattern_counts.items(), key=lambda x: x[1]) if pattern_counts else None
        }
    
    def clear_log(self):
        """Clear the redaction log for new processing batch."""
        self.redaction_log.clear()

def test_redaction_system():
    """Test the privacy redaction system with sample data."""
    redactor = PrivacyRedactor()
    
    test_cases = [
        "Contact john.doe@example.com for API access with key api_key=abc123xyz789",
        "GitHub token: ghp_1234567890123456789012345678901234567890",
        "Server IP: 192.168.1.100 or public 203.0.113.1",
        "Call (555) 123-4567 or SSN 123-45-6789 for verification",
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}:")
        print(f"Original: {test_case}")
        
        result = redactor.redact_text(test_case)
        print(f"Redacted: {result.redacted_text}")
        print(f"Redactions: {result.redaction_count}")
        
        validation = redactor.validate_redaction(result)
        print(f"Validation: {validation}")
    
    print(f"\nOverall Stats: {redactor.get_redaction_stats()}")

if __name__ == "__main__":
    test_redaction_system()