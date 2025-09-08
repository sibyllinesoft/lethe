# Security Policy

## 🔒 Security Overview

Lethe is designed with privacy and security as core principles. This document outlines our security practices, data handling policies, and procedures for reporting security vulnerabilities.

## 🛡️ Data Protection & Privacy

### Agent Conversation Data

Lethe processes agent conversation traces containing potentially sensitive information. Our privacy protection includes:

#### Automatic Data Scrubbing
- **Email Protection**: Email addresses are automatically redacted while optionally preserving domains for functionality
- **Token/Secret Detection**: High-entropy strings (API keys, tokens, secrets) are detected and removed using entropy analysis (min threshold: 4.0 bits)
- **File Path Anonymization**: File paths are hashed while preserving extensions for context utility
- **URL Sanitization**: URLs are redacted with configurable domain/protocol preservation
- **Personal Information**: PII detection and anonymization with reversible hashing for debugging

#### Configurable Privacy Levels
```python
# Available scrubbing intensities
PRIVACY_LEVELS = {
    "minimal": "Scrub only high-risk secrets and tokens",
    "standard": "Balanced privacy with utility preservation", 
    "aggressive": "Maximum privacy protection"
}
```

### Local-First Architecture
- **No Cloud Dependencies**: Default configuration requires no external services
- **SQLite Storage**: All data stored locally in encrypted SQLite databases
- **Optional Cloud Features**: Cloud integrations are explicitly opt-in with clear warnings

### Data Retention & Cleanup
- **Session Isolation**: Conversation data is isolated by session with automatic cleanup
- **Configurable Retention**: Data retention periods are user-configurable
- **Secure Deletion**: Cryptographic erasure of sensitive data on cleanup

## 🔐 Security Practices

### Code Security
- **Static Analysis**: Automated SAST scanning with Semgrep rules
- **Dependency Scanning**: Regular vulnerability scans of all dependencies
- **Input Validation**: All user inputs validated and sanitized
- **SQL Injection Protection**: Parameterized queries and prepared statements

### Cryptographic Security
- **Deterministic Hashing**: SHA-256 with salts for consistent anonymization
- **Secure Random Generation**: Cryptographically secure random number generation
- **Key Derivation**: PBKDF2 for key derivation with configurable iterations

### Network Security  
- **TLS Enforcement**: All network communication over TLS 1.2+
- **Certificate Validation**: Strict certificate validation for external connections
- **Rate Limiting**: Built-in rate limiting to prevent abuse
- **CORS Protection**: Configurable CORS policies for web interfaces

## 🐛 Vulnerability Reporting

### Reporting Security Issues

**Please DO NOT report security vulnerabilities through public GitHub issues.**

For security vulnerabilities, please email: **security@sibyllinesoft.com**

Include the following information:
- **Description**: Detailed description of the vulnerability
- **Reproduction Steps**: Step-by-step instructions to reproduce
- **Impact Assessment**: Your assessment of the potential impact
- **Proof of Concept**: If applicable, include PoC code or screenshots
- **Suggested Fix**: Any suggestions for fixing the issue

### Security Response Process

1. **Acknowledgment**: We will acknowledge receipt within 48 hours
2. **Assessment**: Security team will assess severity and impact within 5 business days
3. **Communication**: We will provide a response timeline and regular updates
4. **Disclosure**: Coordinated disclosure timeline will be agreed upon
5. **Resolution**: Fixes will be developed, tested, and released
6. **Recognition**: Contributors will be acknowledged (with permission)

### Security Advisory Process
- **Critical Issues**: Emergency patches within 48-72 hours
- **High Severity**: Patches within 1-2 weeks  
- **Medium/Low**: Addressed in next regular release cycle
- **Public Disclosure**: After fix is available and deployed

## 🏆 Security Best Practices for Users

### Installation Security
```bash
# Verify package integrity
npm audit before installing
npx ctx-run --verify-install

# Use specific versions to avoid supply chain attacks
npm install ctx-run@^2.1.4
```

### Configuration Security
- **Review Configurations**: Always review generated configurations
- **Minimal Permissions**: Use least-privilege principle for file access
- **Secure Storage**: Store sensitive configurations in secure locations
- **Regular Updates**: Keep Lethe and dependencies updated

### Data Handling
- **Review Scrubbing**: Verify privacy scrubbing meets your requirements
- **Backup Security**: Secure any exported data or backups
- **Access Control**: Limit access to Lethe data files
- **Audit Logs**: Monitor access and usage patterns

## 🔍 Security Compliance

### Standards Compliance
- **OWASP Guidelines**: Following OWASP secure coding practices
- **NIST Framework**: Aligned with NIST cybersecurity guidelines  
- **Privacy Regulations**: Designed to support GDPR/CCPA compliance
- **SOC 2 Alignment**: Infrastructure practices align with SOC 2 Type II

### Audit Trail
- **Access Logging**: All data access is logged with timestamps
- **Change Tracking**: Configuration and data changes are tracked
- **Privacy Actions**: All scrubbing actions are auditable
- **Export Records**: Data exports and sharing are logged

## ⚠️ Known Security Considerations

### Current Limitations
- **Client-Side Storage**: Data is stored client-side - secure your environment
- **Process Memory**: Sensitive data may exist in process memory during execution
- **Log Files**: Debug logs may contain sensitive information - review before sharing
- **Embedding Models**: Local models may cache input data

### Risk Mitigation
- **Process Isolation**: Run in isolated environments for sensitive data
- **Memory Protection**: Clear sensitive variables after use
- **Log Sanitization**: Automatic log scrubbing in production mode
- **Model Security**: Use vetted embedding models from trusted sources

## 🚀 Security Roadmap

### Planned Enhancements
- **End-to-End Encryption**: Full E2E encryption for stored conversations
- **Hardware Security**: TPM/secure enclave support for key storage
- **Zero-Knowledge Proofs**: Privacy-preserving evaluation capabilities
- **Formal Verification**: Mathematical verification of privacy guarantees
- **Differential Privacy**: Statistical privacy for aggregate queries

### Research Initiatives
- **Privacy-Preserving Search**: Homomorphic encryption for vector operations
- **Federated Learning**: Distributed model training without data sharing
- **Secure Multiparty Computation**: Collaborative evaluation without data exposure

## 📞 Security Contact Information

- **Security Email**: security@sibyllinesoft.com
- **PGP Key**: [Available on request]
- **Security Team**: Nathan Rice (Lead Security Engineer)
- **Response SLA**: 48 hours for acknowledgment, 5 business days for assessment

## 📋 Security Checklist for Contributors

### Before Submitting Code
- [ ] Run security linters (`npm run security:check`)
- [ ] Scan dependencies (`npm audit`)  
- [ ] Review code for hardcoded secrets
- [ ] Validate input handling and sanitization
- [ ] Test with malicious inputs
- [ ] Document security implications

### For Security-Related PRs
- [ ] Include security impact assessment
- [ ] Provide test cases for security scenarios
- [ ] Update relevant security documentation
- [ ] Request security team review
- [ ] Consider coordinated disclosure if needed

## 📊 Security Metrics & Monitoring

### Security KPIs
- **Vulnerability Response Time**: Target < 48 hours for critical issues
- **Patch Deployment Time**: Target < 72 hours for critical patches  
- **Security Test Coverage**: Target > 90% for security-critical code
- **Dependency Vulnerabilities**: Target 0 high/critical unpatched vulnerabilities

### Monitoring & Alerting
- **Automated Scanning**: Daily dependency and container scans
- **Anomaly Detection**: Unusual access pattern monitoring
- **Performance Monitoring**: Security overhead measurement
- **Compliance Tracking**: Continuous compliance validation

---

**Last Updated**: December 2024  
**Version**: 1.0  
**Next Review**: March 2025

For questions about this security policy, contact: security@sibyllinesoft.com