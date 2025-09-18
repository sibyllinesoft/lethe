use crate::{
    json_canon::CanonicalJson,
    types::*,
    v2_features::V2FeatureExtractor,
};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::{Duration, Instant},
};
use thiserror::Error;
use uuid::Uuid;

/// Security-specific error types for certificate system
#[derive(Debug, Error)]
pub enum SecurityError {
    #[error("Certificate tampering detected: {0}")]
    TamperingDetected(String),
    
    #[error("Privacy violation: {details}")]
    PrivacyViolation { details: String },
    
    #[error("Secrets leakage detected: {field_path}")]
    SecretsLeakage { field_path: String },
    
    #[error("Certificate validation failed: {reason}")]
    ValidationFailed { reason: String },
    
    #[error("Adversarial attack detected: {attack_type}")]
    AdversarialAttack { attack_type: String },
    
    #[error("Formal verification failed: {property}")]
    FormalVerificationFailed { property: String },
    
    #[error("Memory exhaustion attack: {size_mb}MB exceeds limit")]
    MemoryExhaustion { size_mb: u64 },
    
    #[error("Rate limit exceeded: {current}/{limit} requests")]
    RateLimitExceeded { current: u64, limit: u64 },
    
    #[error("Input validation failed: {field} - {reason}")]
    InputValidation { field: String, reason: String },
}

/// Selection Certificate structure for the determinism system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SelectionCertificate {
    pub certificate_id: Uuid,
    pub version: CertificateVersion,
    pub timestamp: DateTime<Utc>,
    pub digest: String,
    pub transforms: Vec<TransformEntry>,
    pub metadata: CertificateMetadata,
    pub security_attestation: SecurityAttestation,
    pub redaction_log: Vec<RedactionEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CertificateVersion {
    V1,
    V2 { extended_fields: HashMap<String, serde_json::Value> },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TransformEntry {
    pub transform_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub transform_type: String,
    pub input_hash: String,
    pub output_hash: String,
    pub metadata: HashMap<String, serde_json::Value>,
    pub causality_chain: Vec<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CertificateMetadata {
    pub created_by: String,
    pub environment: String,
    pub system_version: String,
    pub security_level: SecurityLevel,
    pub validation_status: ValidationStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SecurityLevel {
    Development,
    Testing,
    Staging,
    Production,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationStatus {
    Pending,
    Valid,
    Invalid { reason: String },
    Revoked { timestamp: DateTime<Utc> },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SecurityAttestation {
    pub signature: String,
    pub attestation_timestamp: DateTime<Utc>,
    pub security_properties: SecurityProperties,
    pub threat_model_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SecurityProperties {
    pub deterministic: bool,
    pub tamper_resistant: bool,
    pub privacy_preserving: bool,
    pub non_repudiation: bool,
    pub byzantine_fault_tolerant: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RedactionEntry {
    pub field_path: String,
    pub redaction_type: RedactionType,
    pub timestamp: DateTime<Utc>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RedactionType {
    ProviderSecret,
    PersonalIdentifiableInfo,
    TokenContent,
    SystemInternal,
}

/// Tamper detection and security diff system
pub struct TamperDetector {
    baseline_digest: String,
    modification_threshold: f64,
    human_readable_diff: bool,
    canonical_json: Arc<CanonicalJson>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TamperReport {
    pub tampering_detected: bool,
    pub confidence_score: f64,
    pub modifications: Vec<SecurityModification>,
    pub risk_assessment: RiskLevel,
    pub recommended_action: String,
    pub forensic_evidence: ForensicEvidence,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SecurityModification {
    pub field_path: String,
    pub modification_type: ModificationType,
    pub severity: RiskLevel,
    pub before_value: Option<String>,
    pub after_value: Option<String>,
    pub integrity_check: IntegrityCheck,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum ModificationType {
    FieldAdded,
    FieldRemoved,
    ValueChanged,
    StructuralModification,
    HashMismatch,
    TimestampManipulation,
    SignatureViolation,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IntegrityCheck {
    pub hash_valid: bool,
    pub signature_valid: bool,
    pub timestamp_consistent: bool,
    pub causality_preserved: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ForensicEvidence {
    pub modification_timestamp: DateTime<Utc>,
    pub source_indicators: Vec<String>,
    pub attack_vector_analysis: String,
    pub recovery_procedure: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SecurityDiff {
    pub summary: String,
    pub detailed_changes: Vec<DetailedChange>,
    pub security_implications: Vec<SecurityImplication>,
    pub remediation_steps: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DetailedChange {
    pub path: String,
    pub change_type: String,
    pub old_value: String,
    pub new_value: String,
    pub security_risk: RiskLevel,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SecurityImplication {
    pub category: String,
    pub description: String,
    pub impact: RiskLevel,
    pub mitigation: String,
}

impl TamperDetector {
    pub fn new(baseline_digest: String, modification_threshold: f64) -> Self {
        Self {
            baseline_digest,
            modification_threshold,
            human_readable_diff: true,
            canonical_json: Arc::new(CanonicalJson::new()),
        }
    }

    pub fn detect_tampering(&self, cert: &SelectionCertificate) -> Result<TamperReport, SecurityError> {
        let current_digest = self.calculate_certificate_digest(cert)?;
        let tampering_detected = current_digest != self.baseline_digest;
        
        if !tampering_detected {
            return Ok(TamperReport {
                tampering_detected: false,
                confidence_score: 1.0,
                modifications: vec![],
                risk_assessment: RiskLevel::Low,
                recommended_action: "No action required - certificate is valid".to_string(),
                forensic_evidence: ForensicEvidence {
                    modification_timestamp: Utc::now(),
                    source_indicators: vec![],
                    attack_vector_analysis: "No tampering detected".to_string(),
                    recovery_procedure: "No recovery needed".to_string(),
                },
            });
        }

        // Detailed analysis of modifications
        let modifications = self.analyze_modifications(cert)?;
        let risk_assessment = self.assess_risk(&modifications);
        let confidence_score = self.calculate_confidence(&modifications);
        
        let forensic_evidence = ForensicEvidence {
            modification_timestamp: Utc::now(),
            source_indicators: self.extract_source_indicators(cert),
            attack_vector_analysis: self.analyze_attack_vector(&modifications),
            recovery_procedure: self.generate_recovery_procedure(&risk_assessment),
        };

        Ok(TamperReport {
            tampering_detected: true,
            confidence_score,
            modifications,
            risk_assessment,
            recommended_action: self.recommend_action(&risk_assessment),
            forensic_evidence,
        })
    }

    pub fn generate_diff(&self, original: &str, modified: &str) -> Result<SecurityDiff, SecurityError> {
        // Parse both certificates
        let original_cert: SelectionCertificate = serde_json::from_str(original)
            .map_err(|e| SecurityError::ValidationFailed { 
                reason: format!("Failed to parse original certificate: {}", e) 
            })?;
            
        let modified_cert: SelectionCertificate = serde_json::from_str(modified)
            .map_err(|e| SecurityError::ValidationFailed { 
                reason: format!("Failed to parse modified certificate: {}", e) 
            })?;

        let detailed_changes = self.compute_detailed_changes(&original_cert, &modified_cert)?;
        let security_implications = self.analyze_security_implications(&detailed_changes);
        let remediation_steps = self.generate_remediation_steps(&security_implications);

        Ok(SecurityDiff {
            summary: format!(
                "Detected {} changes with {} high-risk modifications", 
                detailed_changes.len(),
                detailed_changes.iter().filter(|c| matches!(c.security_risk, RiskLevel::High | RiskLevel::Critical)).count()
            ),
            detailed_changes,
            security_implications,
            remediation_steps,
        })
    }

    pub fn fail_closed_with_diff(&self, error: &SecurityError) -> ! {
        eprintln!("SECURITY FAILURE - SYSTEM SHUTTING DOWN");
        eprintln!("Error: {}", error);
        eprintln!("Timestamp: {}", Utc::now());
        eprintln!("Baseline Digest: {}", self.baseline_digest);
        
        // In production, this would trigger alerts, logging, and graceful shutdown
        std::process::exit(1);
    }

    fn calculate_certificate_digest(&self, cert: &SelectionCertificate) -> Result<String, SecurityError> {
        // Create a sanitized version for hashing (remove dynamic fields)
        let mut sanitized = cert.clone();
        sanitized.timestamp = DateTime::from_timestamp(0, 0).unwrap(); // Normalize timestamp
        
        self.canonical_json.hash_value(&sanitized)
            .map_err(|e| SecurityError::ValidationFailed { 
                reason: format!("Failed to calculate digest: {}", e) 
            })
    }

    fn analyze_modifications(&self, cert: &SelectionCertificate) -> Result<Vec<SecurityModification>, SecurityError> {
        let mut modifications = Vec::new();
        
        // Check each field for modifications
        // This is a simplified implementation - in practice, would compare against baseline
        
        if cert.metadata.validation_status != ValidationStatus::Valid {
            modifications.push(SecurityModification {
                field_path: "metadata.validation_status".to_string(),
                modification_type: ModificationType::ValueChanged,
                severity: RiskLevel::High,
                before_value: Some("Valid".to_string()),
                after_value: Some(format!("{:?}", cert.metadata.validation_status)),
                integrity_check: IntegrityCheck {
                    hash_valid: false,
                    signature_valid: false,
                    timestamp_consistent: true,
                    causality_preserved: true,
                },
            });
        }

        // Check transforms for tampering
        for (i, transform) in cert.transforms.iter().enumerate() {
            if transform.causality_chain.is_empty() {
                modifications.push(SecurityModification {
                    field_path: format!("transforms[{}].causality_chain", i),
                    modification_type: ModificationType::FieldRemoved,
                    severity: RiskLevel::Critical,
                    before_value: Some("non-empty causality chain".to_string()),
                    after_value: Some("empty".to_string()),
                    integrity_check: IntegrityCheck {
                        hash_valid: false,
                        signature_valid: false,
                        timestamp_consistent: false,
                        causality_preserved: false,
                    },
                });
            }
        }

        Ok(modifications)
    }

    fn assess_risk(&self, modifications: &[SecurityModification]) -> RiskLevel {
        let critical_count = modifications.iter().filter(|m| matches!(m.severity, RiskLevel::Critical)).count();
        let high_count = modifications.iter().filter(|m| matches!(m.severity, RiskLevel::High)).count();
        
        if critical_count > 0 {
            RiskLevel::Critical
        } else if high_count > 2 {
            RiskLevel::High
        } else if high_count > 0 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        }
    }

    fn calculate_confidence(&self, modifications: &[SecurityModification]) -> f64 {
        if modifications.is_empty() {
            return 0.0;
        }
        
        let total_integrity_failures = modifications.iter()
            .map(|m| {
                let checks = &m.integrity_check;
                (!checks.hash_valid as u32) + 
                (!checks.signature_valid as u32) + 
                (!checks.timestamp_consistent as u32) + 
                (!checks.causality_preserved as u32)
            })
            .sum::<u32>() as f64;
            
        let max_possible_failures = modifications.len() as f64 * 4.0;
        (total_integrity_failures / max_possible_failures).min(1.0)
    }

    fn extract_source_indicators(&self, cert: &SelectionCertificate) -> Vec<String> {
        vec![
            format!("Certificate ID: {}", cert.certificate_id),
            format!("Created by: {}", cert.metadata.created_by),
            format!("Environment: {}", cert.metadata.environment),
            format!("Version: {:?}", cert.version),
            format!("Transform count: {}", cert.transforms.len()),
        ]
    }

    fn analyze_attack_vector(&self, modifications: &[SecurityModification]) -> String {
        let mut attack_indicators = Vec::new();
        
        for modification in modifications {
            match modification.modification_type {
                ModificationType::HashMismatch => attack_indicators.push("Hash tampering"),
                ModificationType::SignatureViolation => attack_indicators.push("Digital signature forgery"),
                ModificationType::TimestampManipulation => attack_indicators.push("Temporal attack"),
                ModificationType::StructuralModification => attack_indicators.push("Structural integrity violation"),
                _ => {}
            }
        }
        
        if attack_indicators.is_empty() {
            "Unknown attack vector".to_string()
        } else {
            format!("Potential attack vectors: {}", attack_indicators.join(", "))
        }
    }

    fn generate_recovery_procedure(&self, risk: &RiskLevel) -> String {
        match risk {
            RiskLevel::Critical => "IMMEDIATE ACTION REQUIRED: Isolate system, preserve evidence, contact security team, initiate incident response".to_string(),
            RiskLevel::High => "Urgent action required: Validate certificate chain, check system integrity, review recent changes".to_string(),
            RiskLevel::Medium => "Monitor closely: Increase logging, validate related certificates, schedule security review".to_string(),
            RiskLevel::Low => "Standard monitoring: Document incident, update security metrics, continue normal operations".to_string(),
        }
    }

    fn recommend_action(&self, risk: &RiskLevel) -> String {
        match risk {
            RiskLevel::Critical => "FAIL CLOSED - Reject certificate and halt processing".to_string(),
            RiskLevel::High => "Quarantine certificate for manual review".to_string(),
            RiskLevel::Medium => "Flag for security review but allow processing".to_string(),
            RiskLevel::Low => "Log for audit purposes and continue processing".to_string(),
        }
    }

    fn compute_detailed_changes(&self, original: &SelectionCertificate, modified: &SelectionCertificate) -> Result<Vec<DetailedChange>, SecurityError> {
        let mut changes = Vec::new();
        
        // Compare digests
        if original.digest != modified.digest {
            changes.push(DetailedChange {
                path: "digest".to_string(),
                change_type: "Value modified".to_string(),
                old_value: original.digest.clone(),
                new_value: modified.digest.clone(),
                security_risk: RiskLevel::Critical,
            });
        }
        
        // Compare metadata
        if original.metadata.validation_status != modified.metadata.validation_status {
            changes.push(DetailedChange {
                path: "metadata.validation_status".to_string(),
                change_type: "Status change".to_string(),
                old_value: format!("{:?}", original.metadata.validation_status),
                new_value: format!("{:?}", modified.metadata.validation_status),
                security_risk: RiskLevel::High,
            });
        }
        
        // Compare transform counts
        if original.transforms.len() != modified.transforms.len() {
            changes.push(DetailedChange {
                path: "transforms".to_string(),
                change_type: "Array length changed".to_string(),
                old_value: original.transforms.len().to_string(),
                new_value: modified.transforms.len().to_string(),
                security_risk: RiskLevel::High,
            });
        }
        
        Ok(changes)
    }

    fn analyze_security_implications(&self, changes: &[DetailedChange]) -> Vec<SecurityImplication> {
        let mut implications = Vec::new();
        
        for change in changes {
            match change.path.as_str() {
                "digest" => {
                    implications.push(SecurityImplication {
                        category: "Integrity".to_string(),
                        description: "Certificate digest modification indicates potential tampering".to_string(),
                        impact: RiskLevel::Critical,
                        mitigation: "Validate entire certificate chain and revoke if necessary".to_string(),
                    });
                }
                path if path.contains("validation_status") => {
                    implications.push(SecurityImplication {
                        category: "Authorization".to_string(),
                        description: "Validation status change may indicate privilege escalation attempt".to_string(),
                        impact: RiskLevel::High,
                        mitigation: "Re-validate certificate through authorized channels".to_string(),
                    });
                }
                path if path.contains("transforms") => {
                    implications.push(SecurityImplication {
                        category: "Audit Trail".to_string(),
                        description: "Transform log modification compromises audit trail integrity".to_string(),
                        impact: RiskLevel::High,
                        mitigation: "Cross-reference with external audit logs and immutable storage".to_string(),
                    });
                }
                _ => {}
            }
        }
        
        implications
    }

    fn generate_remediation_steps(&self, implications: &[SecurityImplication]) -> Vec<String> {
        let mut steps = Vec::new();
        
        steps.push("1. Immediately isolate affected certificate from processing pipeline".to_string());
        steps.push("2. Preserve forensic evidence and create incident report".to_string());
        steps.push("3. Validate certificate chain integrity using backup systems".to_string());
        steps.push("4. Review system access logs for unauthorized modifications".to_string());
        steps.push("5. Implement additional monitoring for similar attack patterns".to_string());
        
        for implication in implications {
            if matches!(implication.impact, RiskLevel::Critical | RiskLevel::High) {
                steps.push(format!("PRIORITY: {}", implication.mitigation));
            }
        }
        
        steps.push("6. Update threat model and security controls based on findings".to_string());
        steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_certificate() -> SelectionCertificate {
        SelectionCertificate {
            certificate_id: Uuid::new_v4(),
            version: CertificateVersion::V1,
            timestamp: Utc::now(),
            digest: "test_digest_12345".to_string(),
            transforms: vec![
                TransformEntry {
                    transform_id: Uuid::new_v4(),
                    timestamp: Utc::now(),
                    transform_type: "test_transform".to_string(),
                    input_hash: "input_hash".to_string(),
                    output_hash: "output_hash".to_string(),
                    metadata: HashMap::new(),
                    causality_chain: vec![Uuid::new_v4()],
                }
            ],
            metadata: CertificateMetadata {
                created_by: "test_system".to_string(),
                environment: "test".to_string(),
                system_version: "1.0.0".to_string(),
                security_level: SecurityLevel::Testing,
                validation_status: ValidationStatus::Valid,
            },
            security_attestation: SecurityAttestation {
                signature: "test_signature".to_string(),
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
        }
    }

    #[test]
    fn test_valid_certificate_no_tampering() {
        let cert = create_test_certificate();
        let canonical_json = CanonicalJson::new();
        let digest = canonical_json.hash_value(&cert).unwrap();
        
        let detector = TamperDetector::new(digest.clone(), 0.1);
        let report = detector.detect_tampering(&cert).unwrap();
        
        assert!(!report.tampering_detected);
        assert_eq!(report.confidence_score, 1.0);
        assert!(report.modifications.is_empty());
    }

    #[test]
    fn test_tampered_certificate_detection() {
        let mut cert = create_test_certificate();
        let canonical_json = CanonicalJson::new();
        let original_digest = canonical_json.hash_value(&cert).unwrap();
        
        // Tamper with the certificate
        cert.metadata.validation_status = ValidationStatus::Invalid { 
            reason: "Manually modified".to_string() 
        };
        
        let detector = TamperDetector::new(original_digest, 0.1);
        let report = detector.detect_tampering(&cert).unwrap();
        
        assert!(report.tampering_detected);
        assert!(report.confidence_score > 0.0);
        assert!(!report.modifications.is_empty());
    }

    #[test]
    fn test_missing_causality_chain_detection() {
        let mut cert = create_test_certificate();
        let canonical_json = CanonicalJson::new();
        let original_digest = canonical_json.hash_value(&cert).unwrap();
        
        // Remove causality chain (critical security violation)
        cert.transforms[0].causality_chain.clear();
        
        let detector = TamperDetector::new(original_digest, 0.1);
        let report = detector.detect_tampering(&cert).unwrap();
        
        assert!(report.tampering_detected);
        assert!(matches!(report.risk_assessment, RiskLevel::Critical));
        
        let critical_mods = report.modifications.iter()
            .filter(|m| matches!(m.severity, RiskLevel::Critical))
            .count();
        assert!(critical_mods > 0);
    }

    #[test]
    fn test_security_diff_generation() {
        let original_cert = create_test_certificate();
        let mut modified_cert = original_cert.clone();
        modified_cert.digest = "tampered_digest".to_string();
        
        let original_json = serde_json::to_string(&original_cert).unwrap();
        let modified_json = serde_json::to_string(&modified_cert).unwrap();
        
        let detector = TamperDetector::new("test_baseline".to_string(), 0.1);
        let diff = detector.generate_diff(&original_json, &modified_json).unwrap();
        
        assert!(!diff.detailed_changes.is_empty());
        assert!(!diff.security_implications.is_empty());
        assert!(!diff.remediation_steps.is_empty());
        
        // Should detect digest change as critical
        let digest_changes = diff.detailed_changes.iter()
            .filter(|c| c.path == "digest")
            .count();
        assert_eq!(digest_changes, 1);
    }
}