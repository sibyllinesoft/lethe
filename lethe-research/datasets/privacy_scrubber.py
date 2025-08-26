#!/usr/bin/env python3
"""
Privacy Scrubbing System for LetheBench-Agents
==============================================

Configurable privacy scrubbing system that removes sensitive information
from agent conversation traces while preserving their utility for evaluation.
Implements multiple scrubbing strategies with configurable rules and fallbacks.

Privacy Protection Features:
- Email address redaction with domain preservation options
- Token/secret detection and removal with entropy analysis
- File path hashing with extension preservation
- URL redaction with optional domain/protocol preservation
- Personal information detection and anonymization
- Content sanitization while preserving conversation structure

Key Design Principles:
- Configurable scrubbing intensity (minimal, standard, aggressive)
- Deterministic hashing for consistent redaction
- Structure preservation for evaluation validity
- Audit trail for scrubbing decisions
- Reversible anonymization for debugging (optional)
"""

import re
import hashlib
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import urlparse
import base64

@dataclass
class ScrubberConfig:
    """Configuration for privacy scrubbing behavior"""
    # Email scrubbing
    scrub_emails: bool = True
    preserve_email_domains: bool = False
    email_replacement_pattern: str = "user{hash}@{domain}"
    
    # Token/secret scrubbing  
    scrub_tokens: bool = True
    min_token_entropy: float = 4.0
    token_replacement_pattern: str = "[TOKEN_{hash}]"
    
    # File path scrubbing
    scrub_file_paths: bool = True
    preserve_extensions: bool = True
    preserve_directory_structure: bool = False
    path_replacement_pattern: str = "{hash}{ext}"
    
    # URL scrubbing
    scrub_urls: bool = True
    preserve_domains: bool = False
    preserve_protocols: bool = False
    url_replacement_pattern: str = "{protocol}://{domain}/[REDACTED_{hash}]"
    
    # Personal information
    scrub_names: bool = True
    scrub_phone_numbers: bool = True
    scrub_addresses: bool = True
    
    # Content preservation
    preserve_code_structure: bool = True
    preserve_error_types: bool = True
    preserve_command_structure: bool = True
    
    # Hashing configuration
    hash_salt: str = "lethe_privacy_salt_2024"
    hash_length: int = 8
    deterministic_hashing: bool = True
    
    # Scrubbing intensity
    scrubbing_level: str = "standard"  # minimal, standard, aggressive

@dataclass
class ScrubberAuditEntry:
    """Audit entry for tracking scrubbing decisions"""
    original_value: str
    scrubbed_value: str
    scrubber_type: str
    confidence: float
    context: str
    metadata: Dict[str, Any]

class PatternDetector:
    """Detects sensitive patterns in text content"""
    
    def __init__(self):
        # Pattern definitions for different types of sensitive data
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'https?://(?:[-\w.])+(?::\d+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?)?',
            'ipv4': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'phone': r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'api_key': r'(?:api[_-]?key|token|secret)[\'"\s]*[:=][\'"\s]*([A-Za-z0-9+/]{20,})',
            'jwt_token': r'eyJ[A-Za-z0-9+/=]+\.eyJ[A-Za-z0-9+/=]+\.[A-Za-z0-9+/=]*',
            'aws_key': r'AKIA[0-9A-Z]{16}',
            'github_token': r'ghp_[a-zA-Z0-9]{36}',
            'slack_token': r'xox[baprs]-[0-9]{12}-[0-9]{12}-[0-9a-zA-Z]{24}',
            'file_path': r'(?:[A-Za-z]:\\|/)[^\s<>"\'|?*\n]+',
            'mac_address': r'([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})',
            'guid': r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'
        }
        
        # High entropy patterns that might be secrets
        self.entropy_patterns = [
            r'\b[A-Za-z0-9+/]{32,}={0,2}\b',  # Base64-like strings
            r'\b[0-9a-fA-F]{32,}\b',          # Hex strings
            r'\b[A-Za-z0-9]{24,}\b'           # Random-looking alphanumeric
        ]
        
        # Name patterns (common first/last names)
        self.name_indicators = [
            r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            r'\b[A-Z][a-z]+\s+(?:Smith|Johnson|Williams|Brown|Jones|Garcia|Miller|Davis|Rodriguez|Martinez)\b'
        ]
    
    def detect_patterns(self, text: str) -> Dict[str, List[Tuple[str, int, int]]]:
        """Detect all sensitive patterns in text"""
        detections = {}
        
        for pattern_type, pattern in self.patterns.items():
            matches = []
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append((match.group(), match.start(), match.end()))
            if matches:
                detections[pattern_type] = matches
        
        # Check entropy patterns
        entropy_matches = []
        for pattern in self.entropy_patterns:
            for match in re.finditer(pattern, text):
                value = match.group()
                if self._calculate_entropy(value) > 4.0:  # High entropy threshold
                    entropy_matches.append((value, match.start(), match.end()))
        
        if entropy_matches:
            detections['high_entropy'] = entropy_matches
        
        # Check name patterns
        name_matches = []
        for pattern in self.name_indicators:
            for match in re.finditer(pattern, text):
                name_matches.append((match.group(), match.start(), match.end()))
        
        if name_matches:
            detections['names'] = name_matches
        
        return detections
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        text_length = len(text)
        
        for count in char_counts.values():
            probability = count / text_length
            if probability > 0:
                entropy -= probability * (probability ** 0.5).bit_length()
        
        return entropy

class HashGenerator:
    """Generates consistent hashes for anonymization"""
    
    def __init__(self, salt: str = "default_salt", deterministic: bool = True):
        self.salt = salt
        self.deterministic = deterministic
        self._hash_cache = {}
    
    def generate_hash(self, value: str, prefix: str = "", length: int = 8) -> str:
        """Generate a consistent hash for a value"""
        if self.deterministic and value in self._hash_cache:
            return self._hash_cache[value]
        
        # Create hash input
        hash_input = f"{self.salt}:{value}".encode('utf-8')
        hash_obj = hashlib.sha256(hash_input)
        hash_digest = hash_obj.hexdigest()[:length]
        
        # Add prefix if provided
        result = f"{prefix}{hash_digest}" if prefix else hash_digest
        
        if self.deterministic:
            self._hash_cache[value] = result
        
        return result
    
    def clear_cache(self):
        """Clear hash cache (use with caution)"""
        self._hash_cache.clear()

class PrivacyScrubber:
    """Main privacy scrubbing system"""
    
    def __init__(self, config: Optional[ScrubberConfig] = None):
        self.config = config or ScrubberConfig()
        self.pattern_detector = PatternDetector()
        self.hash_generator = HashGenerator(
            salt=self.config.hash_salt,
            deterministic=self.config.deterministic_hashing
        )
        self.logger = logging.getLogger(__name__)
        self.audit_trail = []
        
        # Context-aware scrubbing patterns
        self.context_preservers = {
            'code_block': r'```[\s\S]*?```',
            'inline_code': r'`[^`]+`',
            'error_message': r'(?:Error|Exception):\s*[^\n]+',
            'command': r'(?:^|\$|\>)\s*[^\n]+',
            'log_entry': r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[^\n]*'
        }
    
    def scrub_content(self, content: str, context: str = "general") -> Tuple[str, List[ScrubberAuditEntry]]:
        """Scrub sensitive information from content"""
        scrubbed_content = content
        audit_entries = []
        
        # Detect sensitive patterns
        detections = self.pattern_detector.detect_patterns(content)
        
        # Process detections in order of priority (most specific first)
        priority_order = [
            'jwt_token', 'api_key', 'github_token', 'aws_key', 'slack_token',
            'credit_card', 'ssn', 'email', 'phone', 'high_entropy',
            'url', 'file_path', 'ipv4', 'mac_address', 'guid', 'names'
        ]
        
        # Track replacements to avoid double-scrubbing
        replacements = []
        
        for pattern_type in priority_order:
            if pattern_type in detections:
                for value, start, end in detections[pattern_type]:
                    # Skip if this range was already replaced
                    if any(self._ranges_overlap((start, end), (r_start, r_end)) 
                          for _, r_start, r_end in replacements):
                        continue
                    
                    scrubbed_value, audit_entry = self._scrub_by_type(
                        value, pattern_type, context
                    )
                    
                    if scrubbed_value != value:
                        # Replace in content (need to adjust for previous replacements)
                        adjusted_start, adjusted_end = self._adjust_positions(
                            start, end, replacements
                        )
                        
                        scrubbed_content = (
                            scrubbed_content[:adjusted_start] + 
                            scrubbed_value + 
                            scrubbed_content[adjusted_end:]
                        )
                        
                        # Track replacement
                        length_diff = len(scrubbed_value) - len(value)
                        replacements.append((scrubbed_value, start, end, length_diff))
                        
                        # Add to audit trail
                        audit_entries.append(audit_entry)
        
        # Context-aware post-processing
        scrubbed_content = self._apply_context_preservation(scrubbed_content, context)
        
        return scrubbed_content, audit_entries
    
    def _scrub_by_type(self, value: str, pattern_type: str, context: str) -> Tuple[str, ScrubberAuditEntry]:
        """Scrub a specific value based on its type"""
        confidence = 0.9  # Default high confidence for pattern matches
        
        if pattern_type == 'email' and self.config.scrub_emails:
            return self._scrub_email(value, context)
        elif pattern_type in ['api_key', 'jwt_token', 'github_token', 'aws_key', 'slack_token'] and self.config.scrub_tokens:
            return self._scrub_token(value, pattern_type, context)
        elif pattern_type == 'high_entropy' and self.config.scrub_tokens:
            return self._scrub_high_entropy(value, context)
        elif pattern_type == 'url' and self.config.scrub_urls:
            return self._scrub_url(value, context)
        elif pattern_type == 'file_path' and self.config.scrub_file_paths:
            return self._scrub_file_path(value, context)
        elif pattern_type == 'phone' and self.config.scrub_phone_numbers:
            return self._scrub_phone(value, context)
        elif pattern_type == 'names' and self.config.scrub_names:
            return self._scrub_name(value, context)
        elif pattern_type in ['credit_card', 'ssn']:
            return self._scrub_financial(value, pattern_type, context)
        elif pattern_type in ['ipv4', 'mac_address', 'guid']:
            return self._scrub_identifier(value, pattern_type, context)
        
        # Return unchanged if no scrubbing rule applies
        return value, ScrubberAuditEntry(
            original_value=value,
            scrubbed_value=value,
            scrubber_type=pattern_type,
            confidence=0.0,
            context=context,
            metadata={'action': 'no_scrubbing_needed'}
        )
    
    def _scrub_email(self, email: str, context: str) -> Tuple[str, ScrubberAuditEntry]:
        """Scrub email address"""
        if '@' not in email:
            return email, self._create_audit_entry(email, email, 'email', 0.0, context, {'error': 'invalid_email'})
        
        local, domain = email.rsplit('@', 1)
        hash_val = self.hash_generator.generate_hash(local, length=self.config.hash_length)
        
        if self.config.preserve_email_domains:
            scrubbed = self.config.email_replacement_pattern.format(hash=hash_val, domain=domain)
        else:
            domain_hash = self.hash_generator.generate_hash(domain, length=4)
            scrubbed = f"user{hash_val}@example{domain_hash}.com"
        
        return scrubbed, self._create_audit_entry(email, scrubbed, 'email', 0.95, context, {
            'preserved_domain': self.config.preserve_email_domains
        })
    
    def _scrub_token(self, token: str, token_type: str, context: str) -> Tuple[str, ScrubberAuditEntry]:
        """Scrub API tokens and secrets"""
        hash_val = self.hash_generator.generate_hash(token, length=self.config.hash_length)
        scrubbed = self.config.token_replacement_pattern.format(hash=hash_val)
        
        return scrubbed, self._create_audit_entry(token, scrubbed, token_type, 0.98, context, {
            'token_length': len(token),
            'token_type': token_type
        })
    
    def _scrub_high_entropy(self, value: str, context: str) -> Tuple[str, ScrubberAuditEntry]:
        """Scrub high-entropy strings that might be secrets"""
        entropy = self.pattern_detector._calculate_entropy(value)
        
        if entropy < self.config.min_token_entropy:
            return value, self._create_audit_entry(value, value, 'high_entropy', 0.0, context, {
                'entropy': entropy, 'threshold': self.config.min_token_entropy
            })
        
        hash_val = self.hash_generator.generate_hash(value, length=self.config.hash_length)
        scrubbed = f"[HIGH_ENTROPY_{hash_val}]"
        
        confidence = min(0.9, entropy / 6.0)  # Scale confidence with entropy
        
        return scrubbed, self._create_audit_entry(value, scrubbed, 'high_entropy', confidence, context, {
            'entropy': entropy,
            'original_length': len(value)
        })
    
    def _scrub_url(self, url: str, context: str) -> Tuple[str, ScrubberAuditEntry]:
        """Scrub URL while optionally preserving structure"""
        try:
            parsed = urlparse(url)
            hash_val = self.hash_generator.generate_hash(url, length=self.config.hash_length)
            
            protocol = parsed.scheme if self.config.preserve_protocols else "https"
            domain = parsed.netloc if self.config.preserve_domains else f"example{hash_val[:4]}.com"
            
            scrubbed = self.config.url_replacement_pattern.format(
                protocol=protocol,
                domain=domain,
                hash=hash_val
            )
            
            return scrubbed, self._create_audit_entry(url, scrubbed, 'url', 0.9, context, {
                'preserved_protocol': self.config.preserve_protocols,
                'preserved_domain': self.config.preserve_domains
            })
            
        except Exception as e:
            # Fallback for malformed URLs
            hash_val = self.hash_generator.generate_hash(url, length=self.config.hash_length)
            scrubbed = f"[URL_{hash_val}]"
            
            return scrubbed, self._create_audit_entry(url, scrubbed, 'url', 0.7, context, {
                'error': str(e), 'fallback_used': True
            })
    
    def _scrub_file_path(self, path: str, context: str) -> Tuple[str, ScrubberAuditEntry]:
        """Scrub file paths while preserving structure if configured"""
        path_obj = Path(path)
        hash_val = self.hash_generator.generate_hash(path, length=self.config.hash_length)
        
        if self.config.preserve_extensions and path_obj.suffix:
            extension = path_obj.suffix
        else:
            extension = ""
        
        if self.config.preserve_directory_structure:
            # Preserve directory depth
            parts = path_obj.parts
            depth = len(parts) - 1  # Exclude filename
            scrubbed = "/".join([f"dir{i}" for i in range(depth)]) + f"/file{hash_val}{extension}"
            if scrubbed.startswith("/"):
                scrubbed = scrubbed[1:]  # Remove leading slash
        else:
            scrubbed = self.config.path_replacement_pattern.format(hash=hash_val, ext=extension)
        
        return scrubbed, self._create_audit_entry(path, scrubbed, 'file_path', 0.85, context, {
            'preserved_extension': bool(extension),
            'preserved_structure': self.config.preserve_directory_structure
        })
    
    def _scrub_phone(self, phone: str, context: str) -> Tuple[str, ScrubberAuditEntry]:
        """Scrub phone numbers"""
        # Extract just digits
        digits = re.sub(r'[^\d]', '', phone)
        hash_val = self.hash_generator.generate_hash(digits, length=6)
        
        # Preserve format structure
        if len(digits) == 10:
            scrubbed = f"({hash_val[:3]}) {hash_val[3:6]}-{hash_val[6:10]}"
        else:
            scrubbed = f"[PHONE_{hash_val}]"
        
        return scrubbed, self._create_audit_entry(phone, scrubbed, 'phone', 0.9, context, {
            'digit_count': len(digits)
        })
    
    def _scrub_name(self, name: str, context: str) -> Tuple[str, ScrubberAuditEntry]:
        """Scrub personal names"""
        hash_val = self.hash_generator.generate_hash(name, length=4)
        
        # Try to preserve name structure (First Last -> FirstA LastB)
        words = name.split()
        if len(words) == 2:
            scrubbed = f"Person{hash_val[:2]} {hash_val[2:]}"
        else:
            scrubbed = f"Person{hash_val}"
        
        confidence = 0.7  # Names are harder to detect reliably
        
        return scrubbed, self._create_audit_entry(name, scrubbed, 'name', confidence, context, {
            'word_count': len(words)
        })
    
    def _scrub_financial(self, value: str, pattern_type: str, context: str) -> Tuple[str, ScrubberAuditEntry]:
        """Scrub financial information (credit cards, SSNs)"""
        hash_val = self.hash_generator.generate_hash(value, length=6)
        
        if pattern_type == 'credit_card':
            scrubbed = f"****-****-****-{hash_val[:4]}"
        elif pattern_type == 'ssn':
            scrubbed = f"***-**-{hash_val[:4]}"
        else:
            scrubbed = f"[{pattern_type.upper()}_{hash_val}]"
        
        return scrubbed, self._create_audit_entry(value, scrubbed, pattern_type, 0.95, context, {
            'high_sensitivity': True
        })
    
    def _scrub_identifier(self, value: str, pattern_type: str, context: str) -> Tuple[str, ScrubberAuditEntry]:
        """Scrub system identifiers (IPs, MACs, GUIDs)"""
        hash_val = self.hash_generator.generate_hash(value, length=self.config.hash_length)
        
        if pattern_type == 'ipv4':
            # Preserve IP format
            scrubbed = f"192.168.{hash_val[:3]}.{hash_val[3:6]}"
        elif pattern_type == 'mac_address':
            # Preserve MAC format
            scrubbed = f"XX:XX:XX:{hash_val[:2]}:{hash_val[2:4]}:{hash_val[4:6]}"
        else:
            scrubbed = f"[{pattern_type.upper()}_{hash_val}]"
        
        return scrubbed, self._create_audit_entry(value, scrubbed, pattern_type, 0.85, context, {
            'identifier_type': pattern_type
        })
    
    def _apply_context_preservation(self, content: str, context: str) -> str:
        """Apply context-aware preservation rules"""
        if not self.config.preserve_code_structure:
            return content
        
        # Preserve code block structure
        # This is a placeholder for more sophisticated context preservation
        return content
    
    def _create_audit_entry(self, original: str, scrubbed: str, scrubber_type: str,
                          confidence: float, context: str, metadata: Dict[str, Any]) -> ScrubberAuditEntry:
        """Create audit entry for scrubbing action"""
        return ScrubberAuditEntry(
            original_value=original,
            scrubbed_value=scrubbed,
            scrubber_type=scrubber_type,
            confidence=confidence,
            context=context,
            metadata=metadata
        )
    
    def _ranges_overlap(self, range1: Tuple[int, int], range2: Tuple[int, int]) -> bool:
        """Check if two ranges overlap"""
        return max(range1[0], range2[0]) < min(range1[1], range2[1])
    
    def _adjust_positions(self, start: int, end: int, 
                        replacements: List[Tuple[str, int, int, int]]) -> Tuple[int, int]:
        """Adjust positions based on previous replacements"""
        offset = 0
        for _, r_start, r_end, length_diff in replacements:
            if r_end <= start:  # Replacement was before this position
                offset += length_diff
        
        return start + offset, end + offset
    
    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get audit trail of all scrubbing actions"""
        return [asdict(entry) for entry in self.audit_trail]
    
    def export_audit_trail(self, output_path: str):
        """Export audit trail to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.get_audit_trail(), f, indent=2, ensure_ascii=False)
    
    def clear_audit_trail(self):
        """Clear the audit trail"""
        self.audit_trail.clear()
    
    def get_scrubbing_stats(self) -> Dict[str, Any]:
        """Get statistics about scrubbing operations"""
        if not self.audit_trail:
            return {'total_operations': 0}
        
        scrubber_types = [entry.scrubber_type for entry in self.audit_trail]
        confidences = [entry.confidence for entry in self.audit_trail]
        
        from collections import Counter
        
        return {
            'total_operations': len(self.audit_trail),
            'scrubber_type_distribution': dict(Counter(scrubber_types)),
            'avg_confidence': sum(confidences) / len(confidences),
            'high_confidence_operations': sum(1 for c in confidences if c > 0.8),
            'low_confidence_operations': sum(1 for c in confidences if c < 0.5)
        }

# Predefined configurations for different use cases
def create_minimal_scrubber() -> PrivacyScrubber:
    """Create scrubber with minimal privacy protection (research environments)"""
    config = ScrubberConfig(
        scrubbing_level="minimal",
        scrub_emails=False,
        scrub_tokens=True,
        scrub_file_paths=False,
        scrub_urls=False,
        scrub_names=False,
        scrub_phone_numbers=True,
        preserve_email_domains=True,
        preserve_extensions=True,
        preserve_directory_structure=True
    )
    return PrivacyScrubber(config)

def create_standard_scrubber() -> PrivacyScrubber:
    """Create scrubber with standard privacy protection"""
    config = ScrubberConfig(
        scrubbing_level="standard"
    )
    return PrivacyScrubber(config)

def create_aggressive_scrubber() -> PrivacyScrubber:
    """Create scrubber with aggressive privacy protection (public datasets)"""
    config = ScrubberConfig(
        scrubbing_level="aggressive",
        preserve_email_domains=False,
        preserve_domains=False,
        preserve_protocols=False,
        preserve_extensions=False,
        preserve_directory_structure=False
    )
    return PrivacyScrubber(config)

# Example usage and testing
if __name__ == "__main__":
    # Test content with various sensitive information
    test_content = """
    Hi john.doe@example.com, here's your API key: sk-abc123def456ghi789jkl
    
    The error occurred in /home/user/secret_project/config.py at line 42.
    Stack trace shows: https://github.com/mycompany/private-repo/issues/123
    
    Contact me at (555) 123-4567 or check the logs at 192.168.1.100
    
    JWT token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc
    
    SSN: 123-45-6789
    Credit Card: 4532-1234-5678-9012
    """
    
    # Test with different scrubbing levels
    for scrubber_name, scrubber in [
        ("minimal", create_minimal_scrubber()),
        ("standard", create_standard_scrubber()),
        ("aggressive", create_aggressive_scrubber())
    ]:
        print(f"\n=== {scrubber_name.upper()} SCRUBBING ===")
        
        scrubbed_content, audit_entries = scrubber.scrub_content(test_content, "test")
        
        print("Scrubbed content:")
        print(scrubbed_content)
        
        print(f"\nScrubbing operations: {len(audit_entries)}")
        for entry in audit_entries:
            print(f"  {entry.scrubber_type}: {entry.original_value[:20]}... -> {entry.scrubbed_value} (conf: {entry.confidence:.2f})")
        
        print(f"\nStats: {scrubber.get_scrubbing_stats()}")