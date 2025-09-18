#![allow(dead_code, unused_imports, unused_variables)]

use crate::types::ValidationError;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use unicode_normalization::UnicodeNormalization;

/// Canonical JSON serialization with deterministic output
/// Ensures identical JSON for identical data regardless of input order or formatting
pub struct CanonicalJson {
    /// Significant figures for float rounding (default: 6)
    float_precision: u32,
}

impl CanonicalJson {
    pub fn new() -> Self {
        Self { float_precision: 6 }
    }

    /// Create a stable, canonical JSON representation
    /// This is the single-valued `stable_json(x)` function referenced in requirements
    pub fn serialize<T>(&self, value: &T) -> Result<String, ValidationError>
    where
        T: serde::Serialize,
    {
        let json_value = serde_json::to_value(value)?;
        let canonical_value = self.canonicalize_value(json_value);
        let canonical_string = serde_json::to_string(&canonical_value)?;
        Ok(canonical_string)
    }

    /// Generate a deterministic hash of the canonical representation
    pub fn hash(&self, canonical_json: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(canonical_json.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Hash any serializable value directly
    pub fn hash_value<T>(&self, value: &T) -> Result<String, ValidationError>
    where
        T: serde::Serialize,
    {
        let canonical = self.serialize(value)?;
        Ok(self.hash(&canonical))
    }

    /// Canonicalize a JSON value according to our deterministic rules
    fn canonicalize_value(&self, value: Value) -> Value {
        match value {
            Value::Object(obj) => self.canonicalize_object(obj),
            Value::Array(arr) => Value::Array(
                arr.into_iter()
                    .map(|v| self.canonicalize_value(v))
                    .collect(),
            ),
            Value::String(s) => Value::String(self.normalize_string(s)),
            Value::Number(n) => self.canonicalize_number(n),
            Value::Bool(_) | Value::Null => value,
        }
    }

    /// Canonicalize JSON objects with sorted keys
    fn canonicalize_object(&self, obj: Map<String, Value>) -> Value {
        // Convert to BTreeMap for deterministic key ordering
        let sorted_map: BTreeMap<String, Value> = obj
            .into_iter()
            .map(|(k, v)| {
                let normalized_key = self.normalize_string(k);
                let canonical_value = self.canonicalize_value(v);
                (normalized_key, canonical_value)
            })
            .collect();

        // Convert back to JSON object
        let mut canonical_obj = Map::new();
        for (key, value) in sorted_map {
            // Handle explicit null vs omitted field distinction
            match &value {
                Value::Null => {
                    // Include explicit nulls
                    canonical_obj.insert(key, value);
                }
                _ => {
                    canonical_obj.insert(key, value);
                }
            }
        }

        Value::Object(canonical_obj)
    }

    /// Normalize strings using UTF-8 NFC normalization
    fn normalize_string(&self, s: String) -> String {
        s.nfc().collect()
    }

    /// Canonicalize numbers with fixed precision
    fn canonicalize_number(&self, n: serde_json::Number) -> Value {
        if let Some(_i) = n.as_i64() {
            // Keep integers as-is
            Value::Number(n)
        } else if let Some(_u) = n.as_u64() {
            // Keep unsigned integers as-is
            Value::Number(n)
        } else if let Some(f) = n.as_f64() {
            // Round floats to specified precision
            let rounded = self.round_to_significant_figures(f, self.float_precision);

            // Handle special float values
            if rounded.is_nan() {
                Value::Null // Convert NaN to null for deterministic comparison
            } else if rounded.is_infinite() {
                if rounded.is_sign_positive() {
                    Value::String("Infinity".to_string())
                } else {
                    Value::String("-Infinity".to_string())
                }
            } else {
                // Create number from rounded value
                match serde_json::Number::from_f64(rounded) {
                    Some(num) => Value::Number(num),
                    None => Value::Null,
                }
            }
        } else {
            Value::Null
        }
    }

    /// Round a float to specified significant figures
    fn round_to_significant_figures(&self, value: f64, sig_figs: u32) -> f64 {
        if value == 0.0 || !value.is_finite() {
            return value;
        }

        let magnitude = value.abs().log10().floor();
        let scale = 10_f64.powi(sig_figs as i32 - 1 - magnitude as i32);

        (value * scale).round() / scale
    }
}

impl Default for CanonicalJson {
    fn default() -> Self {
        Self::new()
    }
}

/// Utilities for working with canonical JSON in different contexts
pub mod utils {
    use super::*;

    /// Compare two values for canonical equality
    pub fn canonical_eq<T, U>(a: &T, b: &U) -> Result<bool, ValidationError>
    where
        T: serde::Serialize,
        U: serde::Serialize,
    {
        let canonicalizer = CanonicalJson::new();
        let hash_a = canonicalizer.hash_value(a)?;
        let hash_b = canonicalizer.hash_value(b)?;
        Ok(hash_a == hash_b)
    }

    /// Create a canonical diff between two JSON values
    pub fn canonical_diff<T, U>(a: &T, b: &U) -> Result<Vec<String>, ValidationError>
    where
        T: serde::Serialize,
        U: serde::Serialize,
    {
        let canonicalizer = CanonicalJson::new();
        let json_a = canonicalizer.serialize(a)?;
        let json_b = canonicalizer.serialize(b)?;

        let mut differences = Vec::new();

        if json_a != json_b {
            // Parse both for detailed comparison
            let value_a: Value = serde_json::from_str(&json_a)?;
            let value_b: Value = serde_json::from_str(&json_b)?;

            find_differences("", &value_a, &value_b, &mut differences);
        }

        Ok(differences)
    }

    fn find_differences(path: &str, a: &Value, b: &Value, differences: &mut Vec<String>) {
        match (a, b) {
            (Value::Object(obj_a), Value::Object(obj_b)) => {
                // Check for keys in a but not in b
                for key in obj_a.keys() {
                    let new_path = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{}.{}", path, key)
                    };

                    match obj_b.get(key) {
                        Some(val_b) => find_differences(&new_path, &obj_a[key], val_b, differences),
                        None => {
                            differences.push(format!("Key '{}' missing in second object", new_path))
                        }
                    }
                }

                // Check for keys in b but not in a
                for key in obj_b.keys() {
                    if !obj_a.contains_key(key) {
                        let new_path = if path.is_empty() {
                            key.clone()
                        } else {
                            format!("{}.{}", path, key)
                        };
                        differences.push(format!("Key '{}' missing in first object", new_path));
                    }
                }
            }
            (Value::Array(arr_a), Value::Array(arr_b)) => {
                if arr_a.len() != arr_b.len() {
                    differences.push(format!(
                        "Array length differs at '{}': {} vs {}",
                        path,
                        arr_a.len(),
                        arr_b.len()
                    ));
                } else {
                    for (i, (val_a, val_b)) in arr_a.iter().zip(arr_b.iter()).enumerate() {
                        let new_path = format!("{}[{}]", path, i);
                        find_differences(&new_path, val_a, val_b, differences);
                    }
                }
            }
            _ => {
                if a != b {
                    differences.push(format!("Value differs at '{}': {:?} vs {:?}", path, a, b));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn test_canonical_json_deterministic() {
        let canonicalizer = CanonicalJson::new();

        // Same data in different orders should produce identical results
        let obj1 = json!({
            "b": 2,
            "a": 1,
            "c": "test"
        });

        let obj2 = json!({
            "a": 1,
            "c": "test",
            "b": 2
        });

        let canonical1 = canonicalizer.serialize(&obj1).unwrap();
        let canonical2 = canonicalizer.serialize(&obj2).unwrap();

        assert_eq!(canonical1, canonical2);
        assert_eq!(
            canonicalizer.hash(&canonical1),
            canonicalizer.hash(&canonical2)
        );
    }

    #[test]
    fn test_string_normalization() {
        let canonicalizer = CanonicalJson::new();

        // Different Unicode representations of the same string
        let str1 = "café"; // composed
        let str2 = "cafe\u{0301}"; // decomposed (e + combining acute accent)

        let obj1 = json!({ "text": str1 });
        let obj2 = json!({ "text": str2 });

        let canonical1 = canonicalizer.serialize(&obj1).unwrap();
        let canonical2 = canonicalizer.serialize(&obj2).unwrap();

        assert_eq!(canonical1, canonical2);
    }

    #[test]
    fn test_float_precision() {
        let canonicalizer = CanonicalJson::new();

        let obj1 = json!({ "value": 3.141592653589793 });
        let obj2 = json!({ "value": 3.141593 }); // Rounded to 6 significant figures

        let canonical1 = canonicalizer.serialize(&obj1).unwrap();
        let canonical2 = canonicalizer.serialize(&obj2).unwrap();

        // Should be equal after rounding
        assert_eq!(
            canonicalizer.hash(&canonical1),
            canonicalizer.hash(&canonical2)
        );
    }

    #[test]
    fn test_null_vs_omitted() {
        let canonicalizer = CanonicalJson::new();

        let obj_with_null = json!({ "a": 1, "b": null });
        let mut obj_without_b = HashMap::new();
        obj_without_b.insert("a", 1);

        let canonical1 = canonicalizer.serialize(&obj_with_null).unwrap();
        let canonical2 = canonicalizer.serialize(&obj_without_b).unwrap();

        // These should be different (explicit null vs omitted)
        assert_ne!(canonical1, canonical2);
    }

    #[test]
    fn test_canonical_diff() {
        use utils::canonical_diff;

        let obj1 = json!({ "a": 1, "b": 2 });
        let obj2 = json!({ "a": 1, "b": 3 });

        let differences = canonical_diff(&obj1, &obj2).unwrap();
        assert!(!differences.is_empty());
        assert!(differences[0].contains("Value differs at 'b'"));
    }
}
