use crate::{
    json_canon::CanonicalJson,
    security_testing::{SecurityError, SelectionCertificate, TransformEntry},
    types::*,
};
use std::collections::{HashMap, HashSet};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use serde_json::Value;
use rand::{thread_rng, Rng, seq::SliceRandom};
use std::sync::Arc;

/// Metamorphic security testing framework
/// Tests that fundamental security properties hold under various transformations
pub struct MetamorphicSecurityTester {
    canonical_json: Arc<CanonicalJson>,
    max_test_iterations: usize,
    float_precision_threshold: f64,
}

/// Property-based test results for security invariants
#[derive(Debug, Clone)]
pub struct MetamorphicTestResult {
    pub property_name: String,
    pub test_passed: bool,
    pub iterations_completed: usize,
    pub failure_details: Option<String>,
    pub confidence_level: f64,
    pub attack_vectors_tested: Vec<String>,
}

/// Specific property violations found during testing
#[derive(Debug, Clone)]
pub struct PropertyViolation {
    pub property: String,
    pub input_data: String,
    pub expected_behavior: String,
    pub actual_behavior: String,
    pub security_impact: SecurityImpact,
}

#[derive(Debug, Clone)]
pub enum SecurityImpact {
    None,
    Low,
    Medium,
    High,
    Critical,
}

impl MetamorphicSecurityTester {
    pub fn new() -> Self {
        Self {
            canonical_json: Arc::new(CanonicalJson::new()),
            max_test_iterations: 1000,
            float_precision_threshold: 1e-10,
        }
    }

    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.max_test_iterations = iterations;
        self
    }

    /// Test that atom order in certificate structures doesn't affect security properties
    /// Property: sort(atoms) ≡ atoms (canonical ordering invariant)
    pub fn test_atom_order_invariant(&self) -> Result<MetamorphicTestResult, SecurityError> {
        let mut violations = Vec::new();
        let mut iterations = 0;
        let mut attack_vectors = Vec::new();
        
        for _ in 0..self.max_test_iterations {
            iterations += 1;
            
            // Create test certificate with randomized atom order
            let original_cert = self.create_test_certificate_with_atoms();
            let reordered_cert = self.randomize_atom_order(&original_cert)?;
            
            // Test invariant: canonical hash should be identical
            let original_hash = self.canonical_json.hash_value(&original_cert)
                .map_err(|e| SecurityError::ValidationFailed { 
                    reason: format!("Failed to hash original: {}", e) 
                })?;
                
            let reordered_hash = self.canonical_json.hash_value(&reordered_cert)
                .map_err(|e| SecurityError::ValidationFailed { 
                    reason: format!("Failed to hash reordered: {}", e) 
                })?;
            
            if original_hash != reordered_hash {
                violations.push(PropertyViolation {
                    property: "atom_order_invariant".to_string(),
                    input_data: format!("Iteration {}: atoms reordered", iterations),
                    expected_behavior: "Identical canonical hash".to_string(),
                    actual_behavior: format!("Hash changed: {} != {}", original_hash, reordered_hash),
                    security_impact: SecurityImpact::Critical,
                });
                break;
            }
            
            // Test various atom ordering attack vectors
            if iterations % 100 == 0 {
                attack_vectors.push(format!("Random reordering (iteration {})", iterations));
            }
        }
        
        // Additional targeted attacks on atom ordering
        self.test_targeted_atom_attacks(&mut violations, &mut attack_vectors)?;
        
        Ok(MetamorphicTestResult {
            property_name: "atom_order_invariant".to_string(),
            test_passed: violations.is_empty(),
            iterations_completed: iterations,
            failure_details: if violations.is_empty() { 
                None 
            } else { 
                Some(format!("Found {} violations", violations.len())) 
            },
            confidence_level: if violations.is_empty() { 0.999 } else { 0.0 },
            attack_vectors_tested: attack_vectors,
        })
    }

    /// Test that duplicate elimination is secure and doesn't introduce vulnerabilities
    /// Property: dedupe(S ∪ {x, x}) ≡ dedupe(S ∪ {x}) (idempotent deduplication)
    pub fn test_duplicate_neutralization(&self) -> Result<MetamorphicTestResult, SecurityError> {
        let mut violations = Vec::new();
        let mut iterations = 0;
        let mut attack_vectors = Vec::new();
        
        for _ in 0..self.max_test_iterations {
            iterations += 1;
            
            // Create certificate with potential duplicate fields
            let base_cert = self.create_test_certificate_with_duplicates();
            let deduplicated = self.apply_deduplication(&base_cert)?;
            let double_deduplicated = self.apply_deduplication(&deduplicated)?;
            
            // Test idempotency: deduplication should be stable
            let hash1 = self.canonical_json.hash_value(&deduplicated)
                .map_err(|e| SecurityError::ValidationFailed { 
                    reason: format!("Failed to hash first dedup: {}", e) 
                })?;
                
            let hash2 = self.canonical_json.hash_value(&double_deduplicated)
                .map_err(|e| SecurityError::ValidationFailed { 
                    reason: format!("Failed to hash second dedup: {}", e) 
                })?;
            
            if hash1 != hash2 {
                violations.push(PropertyViolation {
                    property: "duplicate_neutralization".to_string(),
                    input_data: format!("Certificate with {} transforms", base_cert.transforms.len()),
                    expected_behavior: "Idempotent deduplication".to_string(),
                    actual_behavior: format!("Hash changed after second dedup: {} != {}", hash1, hash2),
                    security_impact: SecurityImpact::High,
                });
            }
            
            // Test for duplicate-based injection attacks
            if iterations % 50 == 0 {
                let injection_result = self.test_duplicate_injection_attack(&base_cert)?;
                if !injection_result.secure {
                    violations.push(PropertyViolation {
                        property: "duplicate_injection_resistance".to_string(),
                        input_data: injection_result.attack_payload,
                        expected_behavior: "Reject or neutralize malicious duplicates".to_string(),
                        actual_behavior: injection_result.vulnerability_description,
                        security_impact: SecurityImpact::Critical,
                    });
                }
                attack_vectors.push("Duplicate injection attack".to_string());
            }
        }
        
        Ok(MetamorphicTestResult {
            property_name: "duplicate_neutralization".to_string(),
            test_passed: violations.is_empty(),
            iterations_completed: iterations,
            failure_details: if violations.is_empty() { 
                None 
            } else { 
                Some(format!("Found {} violations", violations.len())) 
            },
            confidence_level: if violations.is_empty() { 0.995 } else { 0.0 },
            attack_vectors_tested: attack_vectors,
        })
    }

    /// Test resistance to benign field injection attacks
    /// Property: inject(cert, benign_field) should not alter security properties
    pub fn test_benign_field_injection(&self) -> Result<MetamorphicTestResult, SecurityError> {
        let mut violations = Vec::new();
        let mut iterations = 0;
        let mut attack_vectors = Vec::new();
        
        for _ in 0..self.max_test_iterations {
            iterations += 1;
            
            let original_cert = self.create_test_certificate_minimal();
            
            // Inject various benign-looking fields
            let benign_fields = self.generate_benign_fields();
            let injected_cert = self.inject_fields(&original_cert, &benign_fields)?;
            
            // Verify security properties are preserved
            let security_check = self.verify_security_properties(&original_cert, &injected_cert)?;
            
            if !security_check.properties_preserved {
                violations.push(PropertyViolation {
                    property: "benign_field_injection_resistance".to_string(),
                    input_data: format!("Injected fields: {:?}", benign_fields.keys().collect::<Vec<_>>()),
                    expected_behavior: "Security properties preserved".to_string(),
                    actual_behavior: security_check.violation_description,
                    security_impact: SecurityImpact::Medium,
                });
            }
            
            // Test specific injection patterns
            if iterations % 25 == 0 {
                let attack_pattern = format!("Pattern {}: {}", iterations / 25, self.get_injection_pattern_name(iterations));
                attack_vectors.push(attack_pattern);
                
                // Test comment injection
                let comment_attack = self.test_comment_injection_attack(&original_cert)?;
                if !comment_attack.secure {
                    violations.push(PropertyViolation {
                        property: "comment_injection_resistance".to_string(),
                        input_data: comment_attack.attack_payload,
                        expected_behavior: "Comments should not affect security".to_string(),
                        actual_behavior: comment_attack.vulnerability_description,
                        security_impact: SecurityImpact::Medium,
                    });
                }
            }
        }
        
        Ok(MetamorphicTestResult {
            property_name: "benign_field_injection_resistance".to_string(),
            test_passed: violations.is_empty(),
            iterations_completed: iterations,
            failure_details: if violations.is_empty() { 
                None 
            } else { 
                Some(format!("Found {} violations", violations.len())) 
            },
            confidence_level: if violations.is_empty() { 0.99 } else { 0.0 },
            attack_vectors_tested: attack_vectors,
        })
    }

    /// Test float precision attack resistance
    /// Property: normalize(float ± ε) should produce consistent security hashes when ε < threshold
    pub fn test_float_precision_attacks(&self) -> Result<MetamorphicTestResult, SecurityError> {
        let mut violations = Vec::new();
        let mut iterations = 0;
        let mut attack_vectors = Vec::new();
        
        let test_floats = vec![
            3.14159265359, // π
            2.71828182846, // e
            1.41421356237, // √2
            1.61803398875, // φ (golden ratio)
            0.57721566490, // γ (Euler-Mascheroni constant)
        ];
        
        for base_float in test_floats {
            for precision_attack in 0..200 {
                iterations += 1;
                
                // Create subtle precision variations
                let epsilon = self.float_precision_threshold * (precision_attack as f64 + 1.0) / 100.0;
                let modified_float = base_float + epsilon;
                
                let original_cert = self.create_certificate_with_float(base_float);
                let modified_cert = self.create_certificate_with_float(modified_float);
                
                let original_hash = self.canonical_json.hash_value(&original_cert)
                    .map_err(|e| SecurityError::ValidationFailed { 
                        reason: format!("Failed to hash original float cert: {}", e) 
                    })?;
                    
                let modified_hash = self.canonical_json.hash_value(&modified_cert)
                    .map_err(|e| SecurityError::ValidationFailed { 
                        reason: format!("Failed to hash modified float cert: {}", e) 
                    })?;
                
                // For small epsilon, hashes should be identical (canonical rounding)
                // For large epsilon, hashes should be different (significant change)
                let should_be_same = epsilon < self.float_precision_threshold;
                let hashes_same = original_hash == modified_hash;
                
                if should_be_same && !hashes_same {
                    violations.push(PropertyViolation {
                        property: "float_precision_stability".to_string(),
                        input_data: format!("Base: {}, Modified: {}, Epsilon: {}", base_float, modified_float, epsilon),
                        expected_behavior: "Small precision changes should not affect hash".to_string(),
                        actual_behavior: format!("Hash changed for small epsilon: {} != {}", original_hash, modified_hash),
                        security_impact: SecurityImpact::High,
                    });
                } else if !should_be_same && hashes_same {
                    violations.push(PropertyViolation {
                        property: "float_precision_sensitivity".to_string(),
                        input_data: format!("Base: {}, Modified: {}, Epsilon: {}", base_float, modified_float, epsilon),
                        expected_behavior: "Significant precision changes should affect hash".to_string(),
                        actual_behavior: format!("Hash unchanged for large epsilon: {} == {}", original_hash, modified_hash),
                        security_impact: SecurityImpact::Medium,
                    });
                }
                
                if precision_attack % 50 == 0 {
                    attack_vectors.push(format!("Precision attack on {:.5} with ε={:.2e}", base_float, epsilon));
                }
            }
        }
        
        // Test special float values (NaN, Infinity, -0.0)
        self.test_special_float_attacks(&mut violations, &mut attack_vectors)?;
        
        Ok(MetamorphicTestResult {
            property_name: "float_precision_attack_resistance".to_string(),
            test_passed: violations.is_empty(),
            iterations_completed: iterations,
            failure_details: if violations.is_empty() { 
                None 
            } else { 
                Some(format!("Found {} violations", violations.len())) 
            },
            confidence_level: if violations.is_empty() { 0.999 } else { 0.0 },
            attack_vectors_tested: attack_vectors,
        })
    }

    // Helper methods for test implementation

    fn create_test_certificate_with_atoms(&self) -> SelectionCertificate {
        let mut cert = self.create_base_certificate();
        
        // Add multiple transforms that can be reordered
        for i in 0..5 {
            cert.transforms.push(TransformEntry {
                transform_id: Uuid::new_v4(),
                timestamp: Utc::now(),
                transform_type: format!("test_transform_{}", i),
                input_hash: format!("input_hash_{}", i),
                output_hash: format!("output_hash_{}", i),
                metadata: {
                    let mut map = HashMap::new();
                    map.insert(format!("key_{}", i), serde_json::Value::String(format!("value_{}", i)));
                    map.insert("order_id".to_string(), serde_json::Value::Number(serde_json::Number::from(i)));
                    map
                },
                causality_chain: vec![Uuid::new_v4()],
            });
        }
        
        cert
    }

    fn randomize_atom_order(&self, cert: &SelectionCertificate) -> Result<SelectionCertificate, SecurityError> {
        let mut randomized = cert.clone();
        let mut rng = thread_rng();
        
        // Shuffle transforms (main "atoms" in this context)
        randomized.transforms.shuffle(&mut rng);
        
        // Shuffle metadata within each transform
        for transform in &mut randomized.transforms {
            // Create a new HashMap with randomized insertion order
            let mut keys: Vec<String> = transform.metadata.keys().cloned().collect();
            keys.shuffle(&mut rng);
            
            let mut new_metadata = HashMap::new();
            for key in keys {
                if let Some(value) = transform.metadata.get(&key) {
                    new_metadata.insert(key, value.clone());
                }
            }
            transform.metadata = new_metadata;
        }
        
        Ok(randomized)
    }

    fn test_targeted_atom_attacks(&self, violations: &mut Vec<PropertyViolation>, attack_vectors: &mut Vec<String>) -> Result<(), SecurityError> {
        // Test reverse ordering attack
        let cert = self.create_test_certificate_with_atoms();
        let mut reversed = cert.clone();
        reversed.transforms.reverse();
        
        let original_hash = self.canonical_json.hash_value(&cert)?;
        let reversed_hash = self.canonical_json.hash_value(&reversed)?;
        
        if original_hash != reversed_hash {
            violations.push(PropertyViolation {
                property: "atom_reverse_order_attack".to_string(),
                input_data: "Reversed transform order".to_string(),
                expected_behavior: "Identical canonical hash".to_string(),
                actual_behavior: format!("Hash changed: {} != {}", original_hash, reversed_hash),
                security_impact: SecurityImpact::Critical,
            });
        }
        attack_vectors.push("Reverse ordering attack".to_string());
        
        // Test partial ordering attack (adversary controls subset)
        let mut partial_reorder = cert.clone();
        if partial_reorder.transforms.len() >= 3 {
            partial_reorder.transforms.swap(0, 2);
            
            let partial_hash = self.canonical_json.hash_value(&partial_reorder)?;
            if original_hash != partial_hash {
                violations.push(PropertyViolation {
                    property: "atom_partial_reorder_attack".to_string(),
                    input_data: "Swapped first and third transforms".to_string(),
                    expected_behavior: "Identical canonical hash".to_string(),
                    actual_behavior: format!("Hash changed: {} != {}", original_hash, partial_hash),
                    security_impact: SecurityImpact::Critical,
                });
            }
        }
        attack_vectors.push("Partial reordering attack".to_string());
        
        Ok(())
    }

    fn create_test_certificate_with_duplicates(&self) -> SelectionCertificate {
        let mut cert = self.create_base_certificate();
        
        // Create transforms with potential duplicates
        let base_transform = TransformEntry {
            transform_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            transform_type: "duplicate_test".to_string(),
            input_hash: "duplicate_hash".to_string(),
            output_hash: "duplicate_output".to_string(),
            metadata: {
                let mut map = HashMap::new();
                map.insert("duplicate_field".to_string(), serde_json::Value::String("duplicate_value".to_string()));
                map
            },
            causality_chain: vec![Uuid::new_v4()],
        };
        
        // Add original and duplicate
        cert.transforms.push(base_transform.clone());
        cert.transforms.push(base_transform.clone()); // Exact duplicate
        
        // Add near-duplicate with different timestamp
        let mut near_duplicate = base_transform.clone();
        near_duplicate.timestamp = Utc::now();
        near_duplicate.transform_id = Uuid::new_v4();
        cert.transforms.push(near_duplicate);
        
        cert
    }

    fn apply_deduplication(&self, cert: &SelectionCertificate) -> Result<SelectionCertificate, SecurityError> {
        let mut deduplicated = cert.clone();
        let mut seen_hashes = HashSet::new();
        
        // Deduplication strategy: keep first occurrence based on canonical hash
        let mut unique_transforms = Vec::new();
        
        for transform in &cert.transforms {
            let transform_hash = self.canonical_json.hash_value(transform)
                .map_err(|e| SecurityError::ValidationFailed { 
                    reason: format!("Failed to hash transform for deduplication: {}", e) 
                })?;
                
            if seen_hashes.insert(transform_hash) {
                unique_transforms.push(transform.clone());
            }
        }
        
        deduplicated.transforms = unique_transforms;
        Ok(deduplicated)
    }

    fn test_duplicate_injection_attack(&self, cert: &SelectionCertificate) -> Result<InjectionTestResult, SecurityError> {
        // Attempt to inject malicious duplicate with subtle differences
        let mut attack_cert = cert.clone();
        
        if let Some(first_transform) = cert.transforms.first() {
            let mut malicious_duplicate = first_transform.clone();
            
            // Inject hidden payload in metadata
            malicious_duplicate.metadata.insert(
                "hidden_payload".to_string(), 
                serde_json::Value::String("malicious_code_execution()".to_string())
            );
            
            // Try to hide it as a "duplicate"
            malicious_duplicate.transform_id = first_transform.transform_id; // Same ID (suspicious)
            
            attack_cert.transforms.push(malicious_duplicate);
        }
        
        // Check if deduplication properly handles this attack
        let deduplicated = self.apply_deduplication(&attack_cert)?;
        
        // Search for malicious payload in deduplicated result
        let contains_payload = deduplicated.transforms.iter().any(|t| {
            t.metadata.values().any(|v| {
                if let Value::String(s) = v {
                    s.contains("malicious_code_execution")
                } else {
                    false
                }
            })
        });
        
        Ok(InjectionTestResult {
            secure: !contains_payload,
            attack_payload: "Duplicate with hidden malicious metadata".to_string(),
            vulnerability_description: if contains_payload {
                "Malicious payload survived deduplication".to_string()
            } else {
                "Attack successfully neutralized".to_string()
            },
        })
    }

    fn create_test_certificate_minimal(&self) -> SelectionCertificate {
        self.create_base_certificate()
    }

    fn generate_benign_fields(&self) -> HashMap<String, serde_json::Value> {
        let mut fields = HashMap::new();
        
        fields.insert("comment".to_string(), serde_json::Value::String("This is a comment".to_string()));
        fields.insert("debug_info".to_string(), serde_json::Value::String("debug_value".to_string()));
        fields.insert("version_info".to_string(), serde_json::Value::String("1.0.0".to_string()));
        fields.insert("metadata_extra".to_string(), serde_json::Value::Object({
            let mut obj = serde_json::Map::new();
            obj.insert("nested_field".to_string(), serde_json::Value::String("nested_value".to_string()));
            obj
        }));
        
        fields
    }

    fn inject_fields(&self, cert: &SelectionCertificate, fields: &HashMap<String, serde_json::Value>) -> Result<SelectionCertificate, SecurityError> {
        let mut injected = cert.clone();
        
        // Inject fields into first transform's metadata
        if let Some(transform) = injected.transforms.first_mut() {
            for (key, value) in fields {
                transform.metadata.insert(key.clone(), value.clone());
            }
        }
        
        Ok(injected)
    }

    fn verify_security_properties(&self, original: &SelectionCertificate, modified: &SelectionCertificate) -> Result<SecurityVerificationResult, SecurityError> {
        // Check that core security properties are preserved
        let original_core = self.extract_security_core(original)?;
        let modified_core = self.extract_security_core(modified)?;
        
        let properties_preserved = original_core == modified_core;
        
        Ok(SecurityVerificationResult {
            properties_preserved,
            violation_description: if properties_preserved {
                "All security properties preserved".to_string()
            } else {
                "Security properties were altered by field injection".to_string()
            },
        })
    }

    fn extract_security_core(&self, cert: &SelectionCertificate) -> Result<String, SecurityError> {
        // Create a version of the certificate with only security-relevant fields
        let core = SecurityCore {
            certificate_id: cert.certificate_id,
            digest: cert.digest.clone(),
            transform_hashes: cert.transforms.iter().map(|t| {
                format!("{}:{}", t.input_hash, t.output_hash)
            }).collect(),
            security_attestation_signature: cert.security_attestation.signature.clone(),
        };
        
        self.canonical_json.hash_value(&core)
            .map_err(|e| SecurityError::ValidationFailed { 
                reason: format!("Failed to extract security core: {}", e) 
            })
    }

    fn test_comment_injection_attack(&self, cert: &SelectionCertificate) -> Result<InjectionTestResult, SecurityError> {
        let mut attack_cert = cert.clone();
        
        // Inject comment that looks benign but could affect parsing
        if let Some(transform) = attack_cert.transforms.first_mut() {
            transform.metadata.insert(
                "comment".to_string(), 
                serde_json::Value::String("/* */ SELECT * FROM secrets; --".to_string())
            );
        }
        
        // Verify that canonical serialization neutralizes any potential issues
        let original_hash = self.canonical_json.hash_value(cert)?;
        let comment_core = self.extract_security_core(&attack_cert)?;
        
        let secure = original_hash == comment_core;
        
        Ok(InjectionTestResult {
            secure,
            attack_payload: "SQL injection in comment field".to_string(),
            vulnerability_description: if secure {
                "Comment injection had no impact on security core".to_string()
            } else {
                "Comment injection affected security properties".to_string()
            },
        })
    }

    fn get_injection_pattern_name(&self, iteration: usize) -> String {
        let patterns = [
            "comment_injection",
            "metadata_pollution",
            "nested_object_injection",
            "array_injection",
            "type_confusion_injection"
        ];
        patterns[iteration % patterns.len()].to_string()
    }

    fn create_certificate_with_float(&self, float_value: f64) -> SelectionCertificate {
        let mut cert = self.create_base_certificate();
        
        // Inject float into transform metadata
        if let Some(transform) = cert.transforms.first_mut() {
            transform.metadata.insert(
                "precision_test_value".to_string(),
                serde_json::Value::Number(serde_json::Number::from_f64(float_value).unwrap_or(serde_json::Number::from(0)))
            );
        }
        
        cert
    }

    fn test_special_float_attacks(&self, violations: &mut Vec<PropertyViolation>, attack_vectors: &mut Vec<String>) -> Result<(), SecurityError> {
        let special_values = vec![
            (f64::NAN, "NaN"),
            (f64::INFINITY, "Infinity"),
            (f64::NEG_INFINITY, "-Infinity"),
            (-0.0_f64, "-0.0"),
            (0.0_f64, "0.0"),
        ];
        
        for (value, name) in special_values {
            let cert = self.create_certificate_with_float(value);
            
            // Verify that special values are handled consistently
            match self.canonical_json.hash_value(&cert) {
                Ok(hash) => {
                    // Hash should be deterministic and not crash
                    if hash.is_empty() {
                        violations.push(PropertyViolation {
                            property: "special_float_handling".to_string(),
                            input_data: format!("Special float: {}", name),
                            expected_behavior: "Deterministic hash generation".to_string(),
                            actual_behavior: "Empty hash produced".to_string(),
                            security_impact: SecurityImpact::High,
                        });
                    }
                }
                Err(_) => {
                    violations.push(PropertyViolation {
                        property: "special_float_crash_resistance".to_string(),
                        input_data: format!("Special float: {}", name),
                        expected_behavior: "Graceful handling of special values".to_string(),
                        actual_behavior: "Hash generation failed".to_string(),
                        security_impact: SecurityImpact::Medium,
                    });
                }
            }
            
            attack_vectors.push(format!("Special float attack: {}", name));
        }
        
        Ok(())
    }

    fn create_base_certificate(&self) -> SelectionCertificate {
        use crate::security_testing::*;
        
        SelectionCertificate {
            certificate_id: Uuid::new_v4(),
            version: CertificateVersion::V1,
            timestamp: Utc::now(),
            digest: "base_digest".to_string(),
            transforms: vec![],
            metadata: CertificateMetadata {
                created_by: "test_system".to_string(),
                environment: "test".to_string(),
                system_version: "1.0.0".to_string(),
                security_level: SecurityLevel::Testing,
                validation_status: ValidationStatus::Valid,
            },
            security_attestation: SecurityAttestation {
                signature: "base_signature".to_string(),
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
}

// Supporting types for test results

#[derive(Debug)]
struct InjectionTestResult {
    secure: bool,
    attack_payload: String,
    vulnerability_description: String,
}

#[derive(Debug)]
struct SecurityVerificationResult {
    properties_preserved: bool,
    violation_description: String,
}

#[derive(Debug, serde::Serialize)]
struct SecurityCore {
    certificate_id: Uuid,
    digest: String,
    transform_hashes: Vec<String>,
    security_attestation_signature: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atom_order_invariant_basic() {
        let tester = MetamorphicSecurityTester::new().with_iterations(10);
        let result = tester.test_atom_order_invariant().unwrap();
        
        assert!(result.test_passed, "Atom order invariant should hold: {:?}", result.failure_details);
        assert!(result.confidence_level > 0.9);
        assert!(result.iterations_completed > 0);
    }

    #[test]
    fn test_duplicate_neutralization_basic() {
        let tester = MetamorphicSecurityTester::new().with_iterations(10);
        let result = tester.test_duplicate_neutralization().unwrap();
        
        assert!(result.test_passed, "Duplicate neutralization should work: {:?}", result.failure_details);
        assert!(result.confidence_level > 0.9);
    }

    #[test]
    fn test_benign_field_injection_basic() {
        let tester = MetamorphicSecurityTester::new().with_iterations(10);
        let result = tester.test_benign_field_injection().unwrap();
        
        assert!(result.test_passed, "Benign field injection should be safe: {:?}", result.failure_details);
        assert!(result.confidence_level > 0.9);
    }

    #[test]
    fn test_float_precision_attacks_basic() {
        let tester = MetamorphicSecurityTester::new();
        let result = tester.test_float_precision_attacks().unwrap();
        
        assert!(result.test_passed, "Float precision attacks should be resisted: {:?}", result.failure_details);
        assert!(result.confidence_level > 0.9);
    }

    #[test]
    fn test_comprehensive_metamorphic_suite() {
        let tester = MetamorphicSecurityTester::new().with_iterations(5);
        
        let tests = vec![
            tester.test_atom_order_invariant(),
            tester.test_duplicate_neutralization(),
            tester.test_benign_field_injection(),
            tester.test_float_precision_attacks(),
        ];
        
        let mut all_passed = true;
        let mut total_attack_vectors = 0;
        
        for test_result in tests {
            match test_result {
                Ok(result) => {
                    if !result.test_passed {
                        all_passed = false;
                        eprintln!("Test failed: {} - {:?}", result.property_name, result.failure_details);
                    }
                    total_attack_vectors += result.attack_vectors_tested.len();
                }
                Err(e) => {
                    all_passed = false;
                    eprintln!("Test error: {}", e);
                }
            }
        }
        
        assert!(all_passed, "All metamorphic security tests should pass");
        assert!(total_attack_vectors > 0, "Should have tested multiple attack vectors");
    }
}