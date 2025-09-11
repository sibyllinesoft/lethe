use crate::{
    json_canon::CanonicalJson,
    security_testing::{SecurityError, SelectionCertificate, RedactionEntry, RedactionType},
    types::*,
};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use chrono::{DateTime, Utc};
use regex::Regex;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Privacy and Secrets Protection Framework
/// Validates that sensitive information is properly protected in certificates
pub struct PrivacyProtectionValidator {
    canonical_json: Arc<CanonicalJson>,
    pii_patterns: Vec<PIIPattern>,
    secret_patterns: Vec<SecretPattern>,
    token_patterns: Vec<TokenPattern>,
    redaction_policies: RedactionPolicies,
}

/// Patterns for detecting Personally Identifiable Information
#[derive(Debug, Clone)]
pub struct PIIPattern {
    pub pattern_type: PIIType,
    pub regex: Regex,
    pub severity: PrivacySeverity,
    pub context_required: bool,
}

/// Patterns for detecting secrets and credentials
#[derive(Debug, Clone)]
pub struct SecretPattern {
    pub secret_type: SecretType,
    pub regex: Regex,
    pub entropy_threshold: f64,
    pub severity: PrivacySeverity,
}

/// Patterns for detecting tokens and API keys
#[derive(Debug, Clone)]
pub struct TokenPattern {
    pub token_type: TokenType,
    pub regex: Regex,
    pub length_range: (usize, usize),
    pub severity: PrivacySeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PIIType {
    Email,
    Phone,
    SocialSecurity,
    CreditCard,
    IPAddress,
    MacAddress,
    PersonalName,
    Address,
    DateOfBirth,
    DriverLicense,
    Passport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecretType {
    Password,
    APIKey,
    PrivateKey,
    Certificate,
    DatabaseConnection,
    OAuth,
    JWT,
    AwsCredentials,
    ProviderSecret,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TokenType {
    BearerToken,
    SessionToken,
    RefreshToken,
    AccessToken,
    CSRFToken,
    APIToken,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrivacySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Redaction policies for different types of sensitive data
#[derive(Debug, Clone)]
pub struct RedactionPolicies {
    pub pii_policy: RedactionPolicy,
    pub secrets_policy: RedactionPolicy,
    pub tokens_policy: RedactionPolicy,
    pub provider_secrets_policy: RedactionPolicy,
}

#[derive(Debug, Clone)]
pub struct RedactionPolicy {
    pub action: RedactionAction,
    pub preserve_format: bool,
    pub hash_replacement: bool,
    pub audit_logging: bool,
}

#[derive(Debug, Clone)]
pub enum RedactionAction {
    Remove,
    Mask,
    Hash,
    Encrypt,
    Tokenize,
}

/// Result of privacy validation
#[derive(Debug, Serialize, Deserialize)]
pub struct PrivacyValidationResult {
    pub validation_passed: bool,
    pub violations: Vec<PrivacyViolation>,
    pub redaction_effectiveness: f64,
    pub leakage_risk_score: f64,
    pub compliance_status: ComplianceStatus,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PrivacyViolation {
    pub violation_id: Uuid,
    pub violation_type: PrivacyViolationType,
    pub field_path: String,
    pub detected_pattern: String,
    pub severity: PrivacySeverity,
    pub exposure_risk: ExposureRisk,
    pub remediation_required: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum PrivacyViolationType {
    PIIExposure,
    SecretLeakage,
    TokenExposure,
    ProviderSecretLeak,
    InsufficientRedaction,
    RedactionBypass,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExposureRisk {
    pub likelihood: f64,
    pub impact: f64,
    pub overall_risk: f64,
    pub affected_parties: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ComplianceStatus {
    Compliant,
    NonCompliant { violations: u32 },
    PartialCompliance { score: f64 },
    UnderReview,
}

impl PrivacyProtectionValidator {
    pub fn new() -> Result<Self, SecurityError> {
        let pii_patterns = Self::initialize_pii_patterns()?;
        let secret_patterns = Self::initialize_secret_patterns()?;
        let token_patterns = Self::initialize_token_patterns()?;
        
        Ok(Self {
            canonical_json: Arc::new(CanonicalJson::new()),
            pii_patterns,
            secret_patterns,
            token_patterns,
            redaction_policies: RedactionPolicies::default(),
        })
    }

    /// Validate that provider secrets are not leaked in V2 certificate fields
    pub fn validate_provider_secret_protection(&self, cert: &SelectionCertificate) -> Result<PrivacyValidationResult, SecurityError> {
        let mut violations = Vec::new();
        
        // Check V2 extended fields for provider secrets
        if let crate::security_testing::CertificateVersion::V2 { extended_fields } = &cert.version {
            for (field_name, field_value) in extended_fields {
                if let Some(violation) = self.detect_provider_secret_in_field(field_name, field_value)? {
                    violations.push(violation);
                }
            }
        }
        
        // Check transform metadata for provider secrets
        for (i, transform) in cert.transforms.iter().enumerate() {
            for (key, value) in &transform.metadata {
                let field_path = format!("transforms[{}].metadata.{}", i, key);
                if let Some(violation) = self.detect_provider_secret_in_value(&field_path, value)? {
                    violations.push(violation);
                }
            }
        }
        
        // Validate redaction log completeness
        let redaction_violations = self.validate_redaction_completeness(cert, &violations)?;
        violations.extend(redaction_violations);
        
        let leakage_risk_score = self.calculate_leakage_risk(&violations);
        let compliance_status = self.assess_compliance_status(&violations);
        
        Ok(PrivacyValidationResult {
            validation_passed: violations.is_empty(),
            redaction_effectiveness: self.calculate_redaction_effectiveness(&violations),
            leakage_risk_score,
            compliance_status,
            violations,
            recommendations: self.generate_remediation_recommendations(&violations),
        })
    }

    /// Validate redaction before certificate hashing
    pub fn validate_redaction_before_hashing(&self, cert: &SelectionCertificate) -> Result<RedactionValidationResult, SecurityError> {
        let mut redaction_issues = Vec::new();
        
        // Create a version with all sensitive data redacted
        let redacted_cert = self.apply_complete_redaction(cert)?;
        
        // Validate that redaction was effective
        let post_redaction_violations = self.scan_for_sensitive_data(&redacted_cert)?;
        
        if !post_redaction_violations.is_empty() {
            redaction_issues.push(RedactionIssue {
                issue_type: RedactionIssueType::IncompleteRedaction,
                field_path: "multiple".to_string(),
                description: format!("{} violations remain after redaction", post_redaction_violations.len()),
                severity: PrivacySeverity::Critical,
            });
        }
        
        // Validate hash consistency after redaction
        let original_sensitive_hash = self.canonical_json.hash_value(cert)?;
        let redacted_hash = self.canonical_json.hash_value(&redacted_cert)?;
        
        if original_sensitive_hash == redacted_hash && !cert.redaction_log.is_empty() {
            redaction_issues.push(RedactionIssue {
                issue_type: RedactionIssueType::HashInconsistency,
                field_path: "digest".to_string(),
                description: "Hash unchanged despite redaction operations".to_string(),
                severity: PrivacySeverity::High,
            });
        }
        
        // Check for redaction bypass attempts
        let bypass_attempts = self.detect_redaction_bypasses(cert, &redacted_cert)?;
        redaction_issues.extend(bypass_attempts);
        
        Ok(RedactionValidationResult {
            redaction_successful: redaction_issues.is_empty(),
            issues: redaction_issues,
            redacted_certificate: redacted_cert,
            hash_before_redaction: original_sensitive_hash,
            hash_after_redaction: redacted_hash,
        })
    }

    /// Detect PII in transform metadata
    pub fn detect_pii_in_metadata(&self, cert: &SelectionCertificate) -> Result<Vec<PrivacyViolation>, SecurityError> {
        let mut violations = Vec::new();
        
        for (transform_idx, transform) in cert.transforms.iter().enumerate() {
            for (key, value) in &transform.metadata {
                let field_path = format!("transforms[{}].metadata.{}", transform_idx, key);
                
                // Scan for PII patterns
                for pii_pattern in &self.pii_patterns {
                    if let Some(violation) = self.check_pii_pattern(&field_path, value, pii_pattern)? {
                        violations.push(violation);
                    }
                }
            }
        }
        
        Ok(violations)
    }

    /// Validate token content sanitization
    pub fn validate_token_sanitization(&self, cert: &SelectionCertificate) -> Result<TokenSanitizationResult, SecurityError> {
        let mut token_violations = Vec::new();
        let mut sanitization_failures = Vec::new();
        
        // Scan entire certificate for tokens
        let certificate_json = serde_json::to_string(cert)
            .map_err(|e| SecurityError::ValidationFailed { 
                reason: format!("Failed to serialize certificate for token scan: {}", e) 
            })?;
        
        for token_pattern in &self.token_patterns {
            let matches = token_pattern.regex.find_iter(&certificate_json);
            
            for match_result in matches {
                let matched_text = match_result.as_str();
                
                // Check if token should have been sanitized
                if self.should_sanitize_token(&token_pattern.token_type, matched_text) {
                    let field_path = self.locate_token_field_path(cert, matched_text)?;
                    
                    token_violations.push(TokenViolation {
                        token_type: token_pattern.token_type.clone(),
                        field_path,
                        token_preview: self.create_safe_preview(matched_text),
                        severity: token_pattern.severity.clone(),
                        should_be_redacted: true,
                    });
                }
            }
        }
        
        // Check sanitization effectiveness
        for violation in &token_violations {
            if let Some(redaction_entry) = cert.redaction_log.iter().find(|r| r.field_path == violation.field_path) {
                if !self.verify_token_redaction_effectiveness(redaction_entry, &violation.token_preview)? {
                    sanitization_failures.push(SanitizationFailure {
                        field_path: violation.field_path.clone(),
                        failure_reason: "Token still detectable after claimed redaction".to_string(),
                        redaction_method: redaction_entry.redaction_type.clone(),
                    });
                }
            } else {
                sanitization_failures.push(SanitizationFailure {
                    field_path: violation.field_path.clone(),
                    failure_reason: "No redaction log entry for sensitive token".to_string(),
                    redaction_method: RedactionType::SystemInternal,
                });
            }
        }
        
        Ok(TokenSanitizationResult {
            sanitization_effective: sanitization_failures.is_empty(),
            token_violations,
            sanitization_failures,
            compliance_score: self.calculate_token_compliance_score(&token_violations, &sanitization_failures),
        })
    }

    // Helper methods for privacy validation

    fn initialize_pii_patterns() -> Result<Vec<PIIPattern>, SecurityError> {
        let mut patterns = Vec::new();
        
        // Email pattern
        patterns.push(PIIPattern {
            pattern_type: PIIType::Email,
            regex: Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
                .map_err(|e| SecurityError::ValidationFailed { 
                    reason: format!("Invalid email regex: {}", e) 
                })?,
            severity: PrivacySeverity::High,
            context_required: false,
        });
        
        // Phone number pattern
        patterns.push(PIIPattern {
            pattern_type: PIIType::Phone,
            regex: Regex::new(r"(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}")
                .map_err(|e| SecurityError::ValidationFailed { 
                    reason: format!("Invalid phone regex: {}", e) 
                })?,
            severity: PrivacySeverity::Medium,
            context_required: true,
        });
        
        // Social Security Number
        patterns.push(PIIPattern {
            pattern_type: PIIType::SocialSecurity,
            regex: Regex::new(r"\b\d{3}-?\d{2}-?\d{4}\b")
                .map_err(|e| SecurityError::ValidationFailed { 
                    reason: format!("Invalid SSN regex: {}", e) 
                })?,
            severity: PrivacySeverity::Critical,
            context_required: true,
        });
        
        // Credit Card Number
        patterns.push(PIIPattern {
            pattern_type: PIIType::CreditCard,
            regex: Regex::new(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
                .map_err(|e| SecurityError::ValidationFailed { 
                    reason: format!("Invalid credit card regex: {}", e) 
                })?,
            severity: PrivacySeverity::Critical,
            context_required: false,
        });
        
        // IP Address
        patterns.push(PIIPattern {
            pattern_type: PIIType::IPAddress,
            regex: Regex::new(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
                .map_err(|e| SecurityError::ValidationFailed { 
                    reason: format!("Invalid IP regex: {}", e) 
                })?,
            severity: PrivacySeverity::Medium,
            context_required: true,
        });
        
        Ok(patterns)
    }

    fn initialize_secret_patterns() -> Result<Vec<SecretPattern>, SecurityError> {
        let mut patterns = Vec::new();
        
        // API Key pattern
        patterns.push(SecretPattern {
            secret_type: SecretType::APIKey,
            regex: Regex::new(r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?([a-z0-9]{20,})['\"]?")
                .map_err(|e| SecurityError::ValidationFailed { 
                    reason: format!("Invalid API key regex: {}", e) 
                })?,
            entropy_threshold: 3.5,
            severity: PrivacySeverity::Critical,
        });
        
        // AWS Credentials
        patterns.push(SecretPattern {
            secret_type: SecretType::AwsCredentials,
            regex: Regex::new(r"(?i)(AKIA[0-9A-Z]{16}|aws_access_key_id|aws_secret_access_key)")
                .map_err(|e| SecurityError::ValidationFailed { 
                    reason: format!("Invalid AWS regex: {}", e) 
                })?,
            entropy_threshold: 4.0,
            severity: PrivacySeverity::Critical,
        });
        
        // Private Key
        patterns.push(SecretPattern {
            secret_type: SecretType::PrivateKey,
            regex: Regex::new(r"-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----")
                .map_err(|e| SecurityError::ValidationFailed { 
                    reason: format!("Invalid private key regex: {}", e) 
                })?,
            entropy_threshold: 5.0,
            severity: PrivacySeverity::Critical,
        });
        
        // Password patterns
        patterns.push(SecretPattern {
            secret_type: SecretType::Password,
            regex: Regex::new(r"(?i)(password|passwd|pwd)\s*[:=]\s*['\"]?([^\s'\"]{8,})['\"]?")
                .map_err(|e| SecurityError::ValidationFailed { 
                    reason: format!("Invalid password regex: {}", e) 
                })?,
            entropy_threshold: 2.5,
            severity: PrivacySeverity::High,
        });
        
        // Provider secrets (specific to the system)
        patterns.push(SecretPattern {
            secret_type: SecretType::ProviderSecret,
            regex: Regex::new(r"(?i)(provider[_-]?secret|provider[_-]?key|system[_-]?secret)")
                .map_err(|e| SecurityError::ValidationFailed { 
                    reason: format!("Invalid provider secret regex: {}", e) 
                })?,
            entropy_threshold: 3.0,
            severity: PrivacySeverity::Critical,
        });
        
        Ok(patterns)
    }

    fn initialize_token_patterns() -> Result<Vec<TokenPattern>, SecurityError> {
        let mut patterns = Vec::new();
        
        // Bearer Token
        patterns.push(TokenPattern {
            token_type: TokenType::BearerToken,
            regex: Regex::new(r"(?i)bearer\s+([a-z0-9\-._~+/]+=*)")
                .map_err(|e| SecurityError::ValidationFailed { 
                    reason: format!("Invalid bearer token regex: {}", e) 
                })?,
            length_range: (20, 2048),
            severity: PrivacySeverity::High,
        });
        
        // JWT Token
        patterns.push(TokenPattern {
            token_type: TokenType::AccessToken,
            regex: Regex::new(r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*")
                .map_err(|e| SecurityError::ValidationFailed { 
                    reason: format!("Invalid JWT regex: {}", e) 
                })?,
            length_range: (50, 4096),
            severity: PrivacySeverity::High,
        });
        
        // Session Token
        patterns.push(TokenPattern {
            token_type: TokenType::SessionToken,
            regex: Regex::new(r"(?i)(session[_-]?token|sess[_-]?id)\s*[:=]\s*['\"]?([a-z0-9]{16,})['\"]?")
                .map_err(|e| SecurityError::ValidationFailed { 
                    reason: format!("Invalid session token regex: {}", e) 
                })?,
            length_range: (16, 256),
            severity: PrivacySeverity::Medium,
        });
        
        Ok(patterns)
    }

    fn detect_provider_secret_in_field(&self, field_name: &str, field_value: &serde_json::Value) -> Result<Option<PrivacyViolation>, SecurityError> {
        // Check field name for provider secret indicators
        for secret_pattern in &self.secret_patterns {
            if matches!(secret_pattern.secret_type, SecretType::ProviderSecret) {
                if secret_pattern.regex.is_match(field_name) {
                    return Ok(Some(PrivacyViolation {
                        violation_id: Uuid::new_v4(),
                        violation_type: PrivacyViolationType::ProviderSecretLeak,
                        field_path: field_name.to_string(),
                        detected_pattern: field_name.to_string(),
                        severity: secret_pattern.severity.clone(),
                        exposure_risk: ExposureRisk {
                            likelihood: 0.9,
                            impact: 0.95,
                            overall_risk: 0.9 * 0.95,
                            affected_parties: vec!["system".to_string(), "users".to_string()],
                        },
                        remediation_required: true,
                    }));
                }
            }
        }
        
        // Check field value content
        if let serde_json::Value::String(value_str) = field_value {
            self.detect_provider_secret_in_value(field_name, field_value)
        } else {
            Ok(None)
        }
    }

    fn detect_provider_secret_in_value(&self, field_path: &str, value: &serde_json::Value) -> Result<Option<PrivacyViolation>, SecurityError> {
        let value_str = match value {
            serde_json::Value::String(s) => s,
            _ => return Ok(None),
        };
        
        for secret_pattern in &self.secret_patterns {
            if secret_pattern.regex.is_match(value_str) {
                let entropy = self.calculate_shannon_entropy(value_str);
                
                if entropy >= secret_pattern.entropy_threshold {
                    return Ok(Some(PrivacyViolation {
                        violation_id: Uuid::new_v4(),
                        violation_type: match secret_pattern.secret_type {
                            SecretType::ProviderSecret => PrivacyViolationType::ProviderSecretLeak,
                            _ => PrivacyViolationType::SecretLeakage,
                        },
                        field_path: field_path.to_string(),
                        detected_pattern: self.create_safe_preview(value_str),
                        severity: secret_pattern.severity.clone(),
                        exposure_risk: ExposureRisk {
                            likelihood: 0.8,
                            impact: match secret_pattern.severity {
                                PrivacySeverity::Critical => 0.95,
                                PrivacySeverity::High => 0.8,
                                PrivacySeverity::Medium => 0.6,
                                PrivacySeverity::Low => 0.3,
                            },
                            overall_risk: 0.8 * 0.8, // Default calculation
                            affected_parties: vec!["system".to_string()],
                        },
                        remediation_required: true,
                    }));
                }
            }
        }
        
        Ok(None)
    }

    fn validate_redaction_completeness(&self, cert: &SelectionCertificate, detected_violations: &[PrivacyViolation]) -> Result<Vec<PrivacyViolation>, SecurityError> {
        let mut redaction_violations = Vec::new();
        
        // Create a set of redacted field paths
        let redacted_paths: HashSet<String> = cert.redaction_log.iter()
            .map(|r| r.field_path.clone())
            .collect();
        
        // Check if all detected violations have corresponding redaction entries
        for violation in detected_violations {
            if !redacted_paths.contains(&violation.field_path) {
                redaction_violations.push(PrivacyViolation {
                    violation_id: Uuid::new_v4(),
                    violation_type: PrivacyViolationType::InsufficientRedaction,
                    field_path: violation.field_path.clone(),
                    detected_pattern: "Missing redaction entry".to_string(),
                    severity: PrivacySeverity::High,
                    exposure_risk: ExposureRisk {
                        likelihood: 0.9,
                        impact: 0.7,
                        overall_risk: 0.63,
                        affected_parties: vec!["audit".to_string()],
                    },
                    remediation_required: true,
                });
            }
        }
        
        // Check for redaction log entries without corresponding current violations
        // (These might be legitimate past redactions)
        for redaction_entry in &cert.redaction_log {
            let has_current_violation = detected_violations.iter()
                .any(|v| v.field_path == redaction_entry.field_path);
            
            if !has_current_violation {
                // This could indicate successful redaction or orphaned log entry
                // For now, we'll validate that the field is actually clean
                if let Some(residual_violation) = self.check_field_for_residual_data(cert, &redaction_entry.field_path)? {
                    redaction_violations.push(residual_violation);
                }
            }
        }
        
        Ok(redaction_violations)
    }

    fn calculate_leakage_risk(&self, violations: &[PrivacyViolation]) -> f64 {
        if violations.is_empty() {
            return 0.0;
        }
        
        let total_risk: f64 = violations.iter()
            .map(|v| v.exposure_risk.overall_risk)
            .sum();
        
        let avg_risk = total_risk / violations.len() as f64;
        
        // Adjust for number of violations (more violations = higher overall risk)
        let violation_factor = 1.0 + (violations.len() as f64 * 0.1).min(0.5);
        
        (avg_risk * violation_factor).min(1.0)
    }

    fn assess_compliance_status(&self, violations: &[PrivacyViolation]) -> ComplianceStatus {
        let critical_violations = violations.iter().filter(|v| matches!(v.severity, PrivacySeverity::Critical)).count();
        let high_violations = violations.iter().filter(|v| matches!(v.severity, PrivacySeverity::High)).count();
        
        if critical_violations > 0 {
            ComplianceStatus::NonCompliant { 
                violations: violations.len() as u32 
            }
        } else if high_violations > 0 {
            let compliance_score = 1.0 - (violations.len() as f64 * 0.1).min(0.8);
            ComplianceStatus::PartialCompliance { score: compliance_score }
        } else if violations.is_empty() {
            ComplianceStatus::Compliant
        } else {
            let compliance_score = 1.0 - (violations.len() as f64 * 0.05).min(0.5);
            ComplianceStatus::PartialCompliance { score: compliance_score }
        }
    }

    fn calculate_redaction_effectiveness(&self, violations: &[PrivacyViolation]) -> f64 {
        let redaction_failures = violations.iter()
            .filter(|v| matches!(v.violation_type, 
                PrivacyViolationType::InsufficientRedaction | 
                PrivacyViolationType::RedactionBypass))
            .count();
        
        if violations.is_empty() {
            1.0
        } else {
            1.0 - (redaction_failures as f64 / violations.len() as f64)
        }
    }

    fn generate_remediation_recommendations(&self, violations: &[PrivacyViolation]) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if violations.iter().any(|v| matches!(v.violation_type, PrivacyViolationType::ProviderSecretLeak)) {
            recommendations.push("Immediately rotate all provider secrets found in certificate data".to_string());
            recommendations.push("Implement automated secret scanning in certificate generation pipeline".to_string());
        }
        
        if violations.iter().any(|v| matches!(v.violation_type, PrivacyViolationType::PIIExposure)) {
            recommendations.push("Review data collection practices to minimize PII in certificates".to_string());
            recommendations.push("Implement field-level encryption for necessary PII data".to_string());
        }
        
        if violations.iter().any(|v| matches!(v.violation_type, PrivacyViolationType::TokenExposure)) {
            recommendations.push("Implement token redaction before certificate serialization".to_string());
            recommendations.push("Use short-lived tokens and refresh mechanisms".to_string());
        }
        
        if violations.iter().any(|v| matches!(v.violation_type, PrivacyViolationType::InsufficientRedaction)) {
            recommendations.push("Audit and update redaction policies and procedures".to_string());
            recommendations.push("Implement mandatory redaction validation before certificate finalization".to_string());
        }
        
        recommendations.push("Establish regular privacy compliance audits".to_string());
        recommendations.push("Implement data retention and deletion policies".to_string());
        
        recommendations
    }

    fn apply_complete_redaction(&self, cert: &SelectionCertificate) -> Result<SelectionCertificate, SecurityError> {
        let mut redacted_cert = cert.clone();
        
        // Apply all configured redaction policies
        redacted_cert = self.redact_pii_data(redacted_cert)?;
        redacted_cert = self.redact_secrets(redacted_cert)?;
        redacted_cert = self.redact_tokens(redacted_cert)?;
        
        Ok(redacted_cert)
    }

    fn redact_pii_data(&self, mut cert: SelectionCertificate) -> Result<SelectionCertificate, SecurityError> {
        // Apply PII redaction to all text fields
        for transform in &mut cert.transforms {
            for (key, value) in &mut transform.metadata {
                if let serde_json::Value::String(ref mut s) = value {
                    for pii_pattern in &self.pii_patterns {
                        *s = pii_pattern.regex.replace_all(s, "[REDACTED-PII]").to_string();
                    }
                }
            }
        }
        
        Ok(cert)
    }

    fn redact_secrets(&self, mut cert: SelectionCertificate) -> Result<SelectionCertificate, SecurityError> {
        // Apply secret redaction
        for transform in &mut cert.transforms {
            for (key, value) in &mut transform.metadata {
                if let serde_json::Value::String(ref mut s) = value {
                    for secret_pattern in &self.secret_patterns {
                        *s = secret_pattern.regex.replace_all(s, "[REDACTED-SECRET]").to_string();
                    }
                }
            }
        }
        
        Ok(cert)
    }

    fn redact_tokens(&self, mut cert: SelectionCertificate) -> Result<SelectionCertificate, SecurityError> {
        // Apply token redaction
        for transform in &mut cert.transforms {
            for (key, value) in &mut transform.metadata {
                if let serde_json::Value::String(ref mut s) = value {
                    for token_pattern in &self.token_patterns {
                        *s = token_pattern.regex.replace_all(s, "[REDACTED-TOKEN]").to_string();
                    }
                }
            }
        }
        
        Ok(cert)
    }

    fn scan_for_sensitive_data(&self, cert: &SelectionCertificate) -> Result<Vec<PrivacyViolation>, SecurityError> {
        let mut violations = Vec::new();
        
        // Scan for any remaining sensitive patterns
        violations.extend(self.detect_pii_in_metadata(cert)?);
        
        // Additional scans would go here
        
        Ok(violations)
    }

    fn check_pii_pattern(&self, field_path: &str, value: &serde_json::Value, pattern: &PIIPattern) -> Result<Option<PrivacyViolation>, SecurityError> {
        if let serde_json::Value::String(value_str) = value {
            if pattern.regex.is_match(value_str) {
                return Ok(Some(PrivacyViolation {
                    violation_id: Uuid::new_v4(),
                    violation_type: PrivacyViolationType::PIIExposure,
                    field_path: field_path.to_string(),
                    detected_pattern: self.create_safe_preview(value_str),
                    severity: pattern.severity.clone(),
                    exposure_risk: ExposureRisk {
                        likelihood: 0.7,
                        impact: match pattern.severity {
                            PrivacySeverity::Critical => 0.9,
                            PrivacySeverity::High => 0.7,
                            PrivacySeverity::Medium => 0.5,
                            PrivacySeverity::Low => 0.3,
                        },
                        overall_risk: 0.7 * 0.7,
                        affected_parties: vec!["individuals".to_string()],
                    },
                    remediation_required: matches!(pattern.severity, PrivacySeverity::Critical | PrivacySeverity::High),
                }));
            }
        }
        
        Ok(None)
    }

    fn detect_redaction_bypasses(&self, original: &SelectionCertificate, redacted: &SelectionCertificate) -> Result<Vec<RedactionIssue>, SecurityError> {
        let mut bypass_issues = Vec::new();
        
        // Check if sensitive data appears to have moved rather than being redacted
        let original_json = serde_json::to_string(original)?;
        let redacted_json = serde_json::to_string(redacted)?;
        
        // Look for suspicious patterns that might indicate bypass attempts
        for secret_pattern in &self.secret_patterns {
            let original_matches: Vec<_> = secret_pattern.regex.find_iter(&original_json).collect();
            let redacted_matches: Vec<_> = secret_pattern.regex.find_iter(&redacted_json).collect();
            
            if !original_matches.is_empty() && !redacted_matches.is_empty() {
                // Secrets found in both - potential bypass
                bypass_issues.push(RedactionIssue {
                    issue_type: RedactionIssueType::RedactionBypass,
                    field_path: "certificate_wide".to_string(),
                    description: format!("Secret type {:?} still present after redaction", secret_pattern.secret_type),
                    severity: PrivacySeverity::Critical,
                });
            }
        }
        
        Ok(bypass_issues)
    }

    fn should_sanitize_token(&self, token_type: &TokenType, token_text: &str) -> bool {
        // All tokens should be sanitized by default
        match token_type {
            TokenType::BearerToken | TokenType::AccessToken => true,
            TokenType::SessionToken | TokenType::RefreshToken => true,
            TokenType::CSRFToken => false, // Might be OK in some contexts
            TokenType::APIToken => true,
        }
    }

    fn locate_token_field_path(&self, cert: &SelectionCertificate, token_text: &str) -> Result<String, SecurityError> {
        // Simple implementation - would need more sophisticated field path detection
        for (i, transform) in cert.transforms.iter().enumerate() {
            for (key, value) in &transform.metadata {
                if let serde_json::Value::String(s) = value {
                    if s.contains(token_text) {
                        return Ok(format!("transforms[{}].metadata.{}", i, key));
                    }
                }
            }
        }
        
        Ok("unknown_field".to_string())
    }

    fn verify_token_redaction_effectiveness(&self, redaction_entry: &RedactionEntry, token_preview: &str) -> Result<bool, SecurityError> {
        // Verify that the token was actually redacted effectively
        // This is a simplified check - real implementation would be more sophisticated
        Ok(matches!(redaction_entry.redaction_type, 
            RedactionType::TokenContent | 
            RedactionType::SystemInternal))
    }

    fn calculate_token_compliance_score(&self, violations: &[TokenViolation], failures: &[SanitizationFailure]) -> f64 {
        let total_issues = violations.len() + failures.len();
        
        if total_issues == 0 {
            1.0
        } else {
            let critical_issues = violations.iter()
                .filter(|v| matches!(v.severity, PrivacySeverity::Critical))
                .count();
            
            let base_score = 1.0 - (total_issues as f64 * 0.1);
            let critical_penalty = critical_issues as f64 * 0.3;
            
            (base_score - critical_penalty).max(0.0)
        }
    }

    fn check_field_for_residual_data(&self, cert: &SelectionCertificate, field_path: &str) -> Result<Option<PrivacyViolation>, SecurityError> {
        // Check if the specified field still contains sensitive data
        // This would need actual field traversal logic
        Ok(None) // Simplified implementation
    }

    fn calculate_shannon_entropy(&self, s: &str) -> f64 {
        if s.is_empty() {
            return 0.0;
        }
        
        let mut char_counts = HashMap::new();
        for c in s.chars() {
            *char_counts.entry(c).or_insert(0) += 1;
        }
        
        let len = s.len() as f64;
        char_counts.values()
            .map(|&count| {
                let p = count as f64 / len;
                -p * p.log2()
            })
            .sum()
    }

    fn create_safe_preview(&self, text: &str) -> String {
        if text.len() <= 8 {
            "[REDACTED]".to_string()
        } else {
            format!("{}***{}", &text[..2], &text[text.len()-2..])
        }
    }
}

impl RedactionPolicies {
    fn default() -> Self {
        Self {
            pii_policy: RedactionPolicy {
                action: RedactionAction::Hash,
                preserve_format: true,
                hash_replacement: true,
                audit_logging: true,
            },
            secrets_policy: RedactionPolicy {
                action: RedactionAction::Remove,
                preserve_format: false,
                hash_replacement: false,
                audit_logging: true,
            },
            tokens_policy: RedactionPolicy {
                action: RedactionAction::Mask,
                preserve_format: true,
                hash_replacement: true,
                audit_logging: true,
            },
            provider_secrets_policy: RedactionPolicy {
                action: RedactionAction::Remove,
                preserve_format: false,
                hash_replacement: false,
                audit_logging: true,
            },
        }
    }
}

// Supporting types for privacy validation

#[derive(Debug, Serialize, Deserialize)]
pub struct RedactionValidationResult {
    pub redaction_successful: bool,
    pub issues: Vec<RedactionIssue>,
    pub redacted_certificate: SelectionCertificate,
    pub hash_before_redaction: String,
    pub hash_after_redaction: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RedactionIssue {
    pub issue_type: RedactionIssueType,
    pub field_path: String,
    pub description: String,
    pub severity: PrivacySeverity,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum RedactionIssueType {
    IncompleteRedaction,
    RedactionBypass,
    HashInconsistency,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TokenSanitizationResult {
    pub sanitization_effective: bool,
    pub token_violations: Vec<TokenViolation>,
    pub sanitization_failures: Vec<SanitizationFailure>,
    pub compliance_score: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TokenViolation {
    pub token_type: TokenType,
    pub field_path: String,
    pub token_preview: String,
    pub severity: PrivacySeverity,
    pub should_be_redacted: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SanitizationFailure {
    pub field_path: String,
    pub failure_reason: String,
    pub redaction_method: RedactionType,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security_testing::{CertificateMetadata, SecurityLevel, ValidationStatus, SecurityAttestation, SecurityProperties};

    fn create_test_certificate_with_secrets() -> SelectionCertificate {
        use crate::security_testing::*;
        
        let mut cert = SelectionCertificate {
            certificate_id: Uuid::new_v4(),
            version: CertificateVersion::V1,
            timestamp: Utc::now(),
            digest: "test_digest".to_string(),
            transforms: vec![],
            metadata: CertificateMetadata {
                created_by: "test".to_string(),
                environment: "test".to_string(),
                system_version: "1.0.0".to_string(),
                security_level: SecurityLevel::Testing,
                validation_status: ValidationStatus::Valid,
            },
            security_attestation: SecurityAttestation {
                signature: "test_sig".to_string(),
                attestation_timestamp: Utc::now(),
                security_properties: SecurityProperties {
                    deterministic: true,
                    tamper_resistant: true,
                    privacy_preserving: true,
                    non_repudiation: true,
                    byzantine_fault_tolerant: true,
                },
                threat_model_version: "1.0".to_string(),
            },
            redaction_log: vec![],
        };
        
        // Add transform with sensitive data
        let mut metadata = HashMap::new();
        metadata.insert("api_key".to_string(), serde_json::Value::String("sk_test_abcdef123456789".to_string()));
        metadata.insert("user_email".to_string(), serde_json::Value::String("user@example.com".to_string()));
        metadata.insert("provider_secret".to_string(), serde_json::Value::String("secret_key_xyz789".to_string()));
        
        cert.transforms.push(crate::security_testing::TransformEntry {
            transform_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            transform_type: "test_transform".to_string(),
            input_hash: "input_hash".to_string(),
            output_hash: "output_hash".to_string(),
            metadata,
            causality_chain: vec![Uuid::new_v4()],
        });
        
        cert
    }

    #[test]
    fn test_provider_secret_detection() {
        let validator = PrivacyProtectionValidator::new().unwrap();
        let cert = create_test_certificate_with_secrets();
        
        let result = validator.validate_provider_secret_protection(&cert).unwrap();
        
        assert!(!result.validation_passed, "Should detect provider secrets");
        assert!(result.leakage_risk_score > 0.5, "Risk score should be significant");
        
        let provider_violations = result.violations.iter()
            .filter(|v| matches!(v.violation_type, PrivacyViolationType::ProviderSecretLeak))
            .count();
        
        assert!(provider_violations > 0, "Should detect provider secret violations");
    }

    #[test]
    fn test_pii_detection_in_metadata() {
        let validator = PrivacyProtectionValidator::new().unwrap();
        let cert = create_test_certificate_with_secrets();
        
        let violations = validator.detect_pii_in_metadata(&cert).unwrap();
        
        let email_violations = violations.iter()
            .filter(|v| v.detected_pattern.contains("example.com") || v.detected_pattern.contains("REDACTED"))
            .count();
        
        assert!(email_violations > 0, "Should detect email PII");
    }

    #[test]
    fn test_token_sanitization_validation() {
        let validator = PrivacyProtectionValidator::new().unwrap();
        
        // Create certificate with JWT token
        let mut cert = create_test_certificate_with_secrets();
        cert.transforms[0].metadata.insert(
            "jwt_token".to_string(), 
            serde_json::Value::String("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c".to_string())
        );
        
        let result = validator.validate_token_sanitization(&cert).unwrap();
        
        assert!(!result.sanitization_effective, "Should detect unsanitized tokens");
        assert!(!result.token_violations.is_empty(), "Should find token violations");
        assert!(result.compliance_score < 0.8, "Compliance score should be low");
    }

    #[test]
    fn test_redaction_validation() {
        let validator = PrivacyProtectionValidator::new().unwrap();
        let cert = create_test_certificate_with_secrets();
        
        let result = validator.validate_redaction_before_hashing(&cert).unwrap();
        
        // Should successfully redact sensitive data
        assert_ne!(result.hash_before_redaction, result.hash_after_redaction, 
                  "Hash should change after redaction");
        
        // Check that redacted certificate is cleaner
        let original_violations = validator.detect_pii_in_metadata(&cert).unwrap();
        let redacted_violations = validator.detect_pii_in_metadata(&result.redacted_certificate).unwrap();
        
        assert!(redacted_violations.len() < original_violations.len(), 
               "Redacted certificate should have fewer violations");
    }

    #[test]
    fn test_shannon_entropy_calculation() {
        let validator = PrivacyProtectionValidator::new().unwrap();
        
        // High entropy string (likely secret)
        let high_entropy = validator.calculate_shannon_entropy("aB3$kL9#mN8&pQ2!");
        
        // Low entropy string (unlikely secret)
        let low_entropy = validator.calculate_shannon_entropy("aaaaaaaaaa");
        
        assert!(high_entropy > low_entropy, "High entropy string should have higher entropy score");
        assert!(high_entropy > 3.0, "Random-looking string should have high entropy");
        assert!(low_entropy < 1.0, "Repetitive string should have low entropy");
    }

    #[test]
    fn test_compliance_status_assessment() {
        let validator = PrivacyProtectionValidator::new().unwrap();
        
        // Test with critical violations
        let critical_violations = vec![
            PrivacyViolation {
                violation_id: Uuid::new_v4(),
                violation_type: PrivacyViolationType::ProviderSecretLeak,
                field_path: "test".to_string(),
                detected_pattern: "test".to_string(),
                severity: PrivacySeverity::Critical,
                exposure_risk: ExposureRisk {
                    likelihood: 0.9,
                    impact: 0.9,
                    overall_risk: 0.81,
                    affected_parties: vec!["system".to_string()],
                },
                remediation_required: true,
            }
        ];
        
        let status = validator.assess_compliance_status(&critical_violations);
        assert!(matches!(status, ComplianceStatus::NonCompliant { .. }));
        
        // Test with no violations
        let clean_status = validator.assess_compliance_status(&[]);
        assert!(matches!(clean_status, ComplianceStatus::Compliant));
    }
}