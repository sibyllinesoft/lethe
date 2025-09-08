#!/usr/bin/env python3
"""
Privacy Validator for LetheBench

Validates that privacy redaction has been properly applied and no PII
or sensitive information remains in the dataset.

For NeurIPS LetheBench Paper - Dataset Construction Pipeline
"""

import re
import json
import logging
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import hashlib

@dataclass
class PrivacyViolation:
    """Represents a privacy violation found in the data."""
    file_path: str
    line_number: int
    session_id: str
    turn_id: int
    violation_type: str
    violation_text: str
    context: str
    confidence: float

@dataclass
class PrivacyValidationResult:
    """Results from privacy validation."""
    is_compliant: bool
    violations: List[PrivacyViolation]
    total_files_scanned: int
    total_lines_scanned: int
    redaction_coverage: float
    summary_stats: Dict[str, int]

class PrivacyValidator:
    """
    Validates privacy compliance in LetheBench dataset.
    
    Features:
    - Detects remaining PII patterns after redaction
    - Validates redaction consistency and completeness
    - Generates compliance reports for audit
    - Supports custom privacy pattern definitions
    - Provides confidence scoring for violations
    """
    
    def __init__(self):
        """Initialize privacy validator with comprehensive PII patterns."""
        self.logger = logging.getLogger(__name__)
        
        # Enhanced PII detection patterns
        self.pii_patterns = {
            'email': {
                'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'confidence': 0.95,
                'description': 'Email addresses'
            },
            'phone_us': {
                'pattern': r'\b(?:\+1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                'confidence': 0.85,
                'description': 'US phone numbers'
            },
            'phone_international': {
                'pattern': r'\+[1-9]\d{1,14}',
                'confidence': 0.80,
                'description': 'International phone numbers'
            },
            'ssn': {
                'pattern': r'\b[0-9]{3}-?[0-9]{2}-?[0-9]{4}\b',
                'confidence': 0.90,
                'description': 'Social Security Numbers'
            },
            'credit_card': {
                'pattern': r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
                'confidence': 0.95,
                'description': 'Credit card numbers'
            },
            'ip_address': {
                'pattern': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                'confidence': 0.70,  # Lower confidence due to false positives
                'description': 'IP addresses'
            },
            'github_token': {
                'pattern': r'gh[pousr]_[A-Za-z0-9_]{36}',
                'confidence': 0.98,
                'description': 'GitHub personal access tokens'
            },
            'aws_access_key': {
                'pattern': r'AKIA[0-9A-Z]{16}',
                'confidence': 0.95,
                'description': 'AWS access keys'
            },
            'aws_secret_key': {
                'pattern': r'[0-9a-zA-Z/+]{40}',
                'confidence': 0.50,  # Very low confidence due to false positives
                'description': 'AWS secret keys (potential)'
            },
            'jwt_token': {
                'pattern': r'eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*',
                'confidence': 0.85,
                'description': 'JWT tokens'
            },
            'bearer_token': {
                'pattern': r'Bearer\s+([A-Za-z0-9_\-\.]{20,})',
                'confidence': 0.90,
                'description': 'Bearer tokens'
            },
            'api_key_generic': {
                'pattern': r'\b[Aa][Pp][Ii]_?[Kk][Ee][Yy]\s*[=:]\s*[\'"]?([A-Za-z0-9_\-]{20,})[\'"]?',
                'confidence': 0.85,
                'description': 'Generic API keys'
            },
            'slack_token': {
                'pattern': r'xox[baprs]-([A-Za-z0-9_\-]{10,})',
                'confidence': 0.90,
                'description': 'Slack tokens'
            },
            'private_key_header': {
                'pattern': r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----',
                'confidence': 0.98,
                'description': 'Private key headers'
            },
            'database_url': {
                'pattern': r'(?:postgresql|mysql|mongodb)://[^:\s]+:[^@\s]+@[^/\s]+',
                'confidence': 0.90,
                'description': 'Database connection URLs with credentials'
            },
            'personal_name': {
                'pattern': r'\b(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',
                'confidence': 0.60,  # Lower confidence due to false positives
                'description': 'Personal names with titles'
            },
            'address_partial': {
                'pattern': r'\b\d{1,5}\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b',
                'confidence': 0.70,
                'description': 'Partial street addresses'
            },
            'zip_code': {
                'pattern': r'\b\d{5}(?:-\d{4})?\b',
                'confidence': 0.40,  # Very low confidence due to false positives
                'description': 'ZIP codes (potential)'
            }
        }
        
        # Whitelist patterns for common false positives
        self.whitelist_patterns = {
            'example_domains': r'@(?:example\.com|example\.org|test\.com|localhost)',
            'documentation_ips': r'\b(?:127\.0\.0\.1|192\.168\.1\.1|10\.0\.0\.1|172\.16\.0\.1)\b',
            'version_numbers': r'\b\d{1,2}\.\d{1,2}\.\d{1,2}\b',
            'dummy_data': r'\b(?:john\.doe|jane\.smith|foo\.bar|test\.user)@',
            'placeholder_tokens': r'\b(?:your_token_here|placeholder|example_key|dummy_key)\b',
            'redacted_markers': r'\b(?:xxx|redacted|hidden|masked)[\w_]*\b'
        }
        
        # Context window for violation reporting
        self.context_window = 50
    
    def validate_file(self, file_path: Path) -> PrivacyValidationResult:
        """
        Validate privacy compliance for a single JSONL file.
        
        Args:
            file_path: Path to JSONL file to validate
            
        Returns:
            PrivacyValidationResult with detailed findings
        """
        violations = []
        total_lines = 0
        violation_counts = {}
        
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return PrivacyValidationResult(
                is_compliant=False,
                violations=[],
                total_files_scanned=0,
                total_lines_scanned=0,
                redaction_coverage=0.0,
                summary_stats={}
            )
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    total_lines += 1
                    
                    try:
                        turn_data = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue  # Skip invalid JSON
                    
                    # Extract text content to scan
                    text_to_scan = self._extract_scannable_text(turn_data)
                    
                    # Scan for privacy violations
                    line_violations = self._scan_text_for_violations(
                        text_to_scan, 
                        file_path.name, 
                        line_num, 
                        turn_data.get('session_id', 'unknown'),
                        turn_data.get('turn', -1)
                    )
                    
                    violations.extend(line_violations)
                    
                    # Count violation types
                    for violation in line_violations:
                        violation_counts[violation.violation_type] = \
                            violation_counts.get(violation.violation_type, 0) + 1
        
        except Exception as e:
            self.logger.error(f"Error scanning file {file_path}: {e}")
            return PrivacyValidationResult(
                is_compliant=False,
                violations=[],
                total_files_scanned=0,
                total_lines_scanned=0,
                redaction_coverage=0.0,
                summary_stats={}
            )
        
        # Calculate redaction coverage
        total_potential_violations = len(violations)
        redaction_coverage = max(0.0, (total_lines - total_potential_violations) / total_lines) if total_lines > 0 else 1.0
        
        is_compliant = len(violations) == 0
        
        return PrivacyValidationResult(
            is_compliant=is_compliant,
            violations=violations,
            total_files_scanned=1,
            total_lines_scanned=total_lines,
            redaction_coverage=redaction_coverage * 100,  # Convert to percentage
            summary_stats=violation_counts
        )
    
    def _extract_scannable_text(self, turn_data: Dict) -> str:
        """Extract all text content that should be scanned for PII."""
        scannable_parts = []
        
        # Main text field
        if 'text' in turn_data and isinstance(turn_data['text'], str):
            scannable_parts.append(turn_data['text'])
        
        # Meta field content (may contain URLs, etc.)
        if 'meta' in turn_data and isinstance(turn_data['meta'], dict):
            for key, value in turn_data['meta'].items():
                if isinstance(value, str):
                    scannable_parts.append(value)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, str):
                            scannable_parts.append(item)
        
        return ' '.join(scannable_parts)
    
    def _scan_text_for_violations(self, 
                                text: str, 
                                filename: str, 
                                line_num: int, 
                                session_id: str, 
                                turn_id: int) -> List[PrivacyViolation]:
        """Scan text for privacy violations."""
        violations = []
        
        for pattern_name, pattern_info in self.pii_patterns.items():
            pattern = pattern_info['pattern']
            base_confidence = pattern_info['confidence']
            
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matched_text = match.group(0)
                
                # Check against whitelist patterns
                if self._is_whitelisted(matched_text, text, match):
                    continue
                
                # Calculate adjusted confidence
                adjusted_confidence = self._calculate_violation_confidence(
                    matched_text, pattern_name, text, match
                )
                
                if adjusted_confidence > 0.3:  # Minimum confidence threshold
                    # Get surrounding context
                    context = self._get_context(text, match.start(), match.end())
                    
                    violations.append(PrivacyViolation(
                        file_path=filename,
                        line_number=line_num,
                        session_id=session_id,
                        turn_id=turn_id,
                        violation_type=pattern_name,
                        violation_text=matched_text,
                        context=context,
                        confidence=adjusted_confidence
                    ))
        
        return violations
    
    def _is_whitelisted(self, matched_text: str, full_text: str, match: re.Match) -> bool:
        """Check if a match should be whitelisted (not a real violation)."""
        # Check direct whitelist patterns
        for pattern in self.whitelist_patterns.values():
            if re.search(pattern, matched_text, re.IGNORECASE):
                return True
        
        # Check context for whitelist indicators
        context = self._get_context(full_text, match.start(), match.end(), window=20)
        context_lower = context.lower()
        
        whitelist_indicators = [
            'example', 'placeholder', 'dummy', 'test', 'sample',
            'redacted', 'masked', 'hidden', 'xxx', 'anonymized'
        ]
        
        if any(indicator in context_lower for indicator in whitelist_indicators):
            return True
        
        return False
    
    def _calculate_violation_confidence(self, 
                                      matched_text: str, 
                                      pattern_name: str, 
                                      full_text: str, 
                                      match: re.Match) -> float:
        """Calculate adjusted confidence for a potential violation."""
        base_confidence = self.pii_patterns[pattern_name]['confidence']
        
        # Adjust based on context
        context = self._get_context(full_text, match.start(), match.end())
        context_lower = context.lower()
        
        # Reduce confidence for code/technical contexts
        if any(indicator in context_lower for indicator in ['function', 'class', 'variable', 'method', '```']):
            base_confidence *= 0.7
        
        # Reduce confidence for obvious examples
        if any(indicator in matched_text.lower() for indicator in ['example', 'test', 'dummy', 'placeholder']):
            base_confidence *= 0.3
        
        # Increase confidence for realistic-looking data
        if pattern_name == 'email' and '@gmail.com' in matched_text.lower():
            base_confidence *= 1.2
        
        # Special handling for AWS secret keys (high false positive rate)
        if pattern_name == 'aws_secret_key':
            # Only flag if it looks like a real AWS secret (mix of cases and special chars)
            if not (any(c.isupper() for c in matched_text) and 
                   any(c.islower() for c in matched_text) and 
                   any(c in '/+' for c in matched_text)):
                base_confidence *= 0.1
        
        # Special handling for IP addresses
        if pattern_name == 'ip_address':
            # Check if it's a common private/test IP
            if matched_text in ['127.0.0.1', '192.168.1.1', '10.0.0.1', '172.16.0.1']:
                base_confidence *= 0.1
        
        return min(1.0, base_confidence)
    
    def _get_context(self, text: str, start: int, end: int, window: Optional[int] = None) -> str:
        """Get surrounding context for a match."""
        if window is None:
            window = self.context_window
        
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        
        return text[context_start:context_end]
    
    def validate_dataset(self, dataset_dir: Path) -> Dict[str, PrivacyValidationResult]:
        """
        Validate privacy compliance for entire dataset.
        
        Args:
            dataset_dir: Directory containing dataset files
            
        Returns:
            Dictionary mapping file paths to validation results
        """
        results = {}
        
        # Find all JSONL files
        jsonl_files = list(dataset_dir.rglob('*.jsonl'))
        
        for file_path in jsonl_files:
            self.logger.info(f"Validating privacy compliance: {file_path}")
            results[str(file_path.relative_to(dataset_dir))] = self.validate_file(file_path)
        
        return results
    
    def generate_privacy_report(self, 
                              validation_results: Dict[str, PrivacyValidationResult],
                              output_file: Optional[Path] = None) -> str:
        """
        Generate comprehensive privacy compliance report.
        
        Args:
            validation_results: Results from validate_dataset
            output_file: Optional file to write report to
            
        Returns:
            Report text
        """
        report_lines = [
            "# LetheBench Privacy Compliance Report",
            f"Generated: {self._get_timestamp()}\n"
        ]
        
        # Overall summary
        total_violations = sum(len(result.violations) for result in validation_results.values())
        total_files = len(validation_results)
        compliant_files = sum(1 for result in validation_results.values() if result.is_compliant)
        total_lines = sum(result.total_lines_scanned for result in validation_results.values())
        
        avg_coverage = sum(result.redaction_coverage for result in validation_results.values()) / max(1, total_files)
        
        report_lines.extend([
            "## Executive Summary",
            f"- Total Files Scanned: {total_files}",
            f"- Compliant Files: {compliant_files} ({compliant_files/max(1,total_files)*100:.1f}%)",
            f"- Total Privacy Violations: {total_violations}",
            f"- Total Lines Scanned: {total_lines:,}",
            f"- Average Redaction Coverage: {avg_coverage:.1f}%",
            f"- Overall Compliance: {'✓ PASS' if total_violations == 0 else '✗ FAIL'}\n"
        ])
        
        # Violation type breakdown
        all_violation_types = {}
        for result in validation_results.values():
            for violation_type, count in result.summary_stats.items():
                all_violation_types[violation_type] = \
                    all_violation_types.get(violation_type, 0) + count
        
        if all_violation_types:
            report_lines.extend([
                "## Violation Types",
                "| Type | Count | Description |",
                "|------|-------|-------------|"
            ])
            
            for violation_type in sorted(all_violation_types.keys()):
                count = all_violation_types[violation_type]
                description = self.pii_patterns.get(violation_type, {}).get('description', 'Unknown')
                report_lines.append(f"| {violation_type} | {count} | {description} |")
            
            report_lines.append("")
        
        # Detailed results by file
        report_lines.append("## Detailed Results by File")
        
        for file_path, result in validation_results.items():
            report_lines.extend([
                f"\n### {file_path}",
                f"- Compliance Status: {'✓ PASS' if result.is_compliant else '✗ FAIL'}",
                f"- Violations Found: {len(result.violations)}",
                f"- Lines Scanned: {result.total_lines_scanned:,}",
                f"- Redaction Coverage: {result.redaction_coverage:.1f}%"
            ])
            
            if result.violations:
                # Group violations by type
                violations_by_type = {}
                for violation in result.violations:
                    if violation.violation_type not in violations_by_type:
                        violations_by_type[violation.violation_type] = []
                    violations_by_type[violation.violation_type].append(violation)
                
                report_lines.append("\n#### Violations:")
                
                for violation_type, violations in violations_by_type.items():
                    report_lines.append(f"\n**{violation_type}** ({len(violations)} instances):")
                    
                    # Show top 3 violations for this type
                    for violation in sorted(violations, key=lambda x: x.confidence, reverse=True)[:3]:
                        report_lines.extend([
                            f"- Line {violation.line_number}, Session: {violation.session_id}, Turn: {violation.turn_id}",
                            f"  - Confidence: {violation.confidence:.2f}",
                            f"  - Text: `{violation.violation_text}`",
                            f"  - Context: `{violation.context[:100]}...`"
                        ])
                    
                    if len(violations) > 3:
                        report_lines.append(f"  - ... and {len(violations) - 3} more instances")
        
        # Recommendations
        if total_violations > 0:
            report_lines.extend([
                "\n## Recommendations",
                "1. Review and enhance privacy redaction patterns",
                "2. Implement additional validation for high-risk PII types",
                "3. Consider manual review of high-confidence violations",
                "4. Update redaction algorithms based on violation patterns",
                "5. Implement automated redaction quality monitoring"
            ])
        
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for reports."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def test_privacy_validator():
    """Test the privacy validator with sample data."""
    validator = PrivacyValidator()
    
    # Test data with various PII types
    test_cases = [
        "Contact me at john.doe@gmail.com or call (555) 123-4567",
        "My GitHub token is ghp_1234567890123456789012345678901234567890",
        "Database URL: postgresql://user:pass123@db.example.com:5432/mydb",
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        "Server IP is 192.168.1.100 and SSH key is ssh-rsa AAAAB3NzaC1yc...",
        "Example email: user@example.com (this should be whitelisted)",
        "My SSN is 123-45-6789 and credit card is 4532-1234-5678-9012",
        "API key: sk-1234567890abcdef1234567890abcdef",
        "Test data with localhost 127.0.0.1 (should be whitelisted)"
    ]
    
    print("Testing privacy violation detection...")
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {test_case[:50]}...")
        
        violations = validator._scan_text_for_violations(
            test_case, 'test.jsonl', i+1, f'session_{i}', 0
        )
        
        if violations:
            for violation in violations:
                print(f"  {violation.violation_type}: '{violation.violation_text}' (conf: {violation.confidence:.2f})")
        else:
            print("  No violations detected")
    
    print(f"\nTesting whitelist functionality...")
    
    whitelisted_cases = [
        "Email me at test@example.com",
        "Use placeholder credentials: dummy_key_12345",
        "Sample IP: 127.0.0.1 for testing"
    ]
    
    for case in whitelisted_cases:
        violations = validator._scan_text_for_violations(
            case, 'test.jsonl', 1, 'session_test', 0
        )
        print(f"'{case}' -> {len(violations)} violations (should be 0 if whitelisted correctly)")

if __name__ == "__main__":
    test_privacy_validator()