#![allow(dead_code, unused_imports, unused_variables)]

use crate::types::{ChangeType, TransformChangeV2, V2Features};
use std::collections::HashMap;

/// V2 Feature extraction for learning loop closure
pub struct V2FeatureExtractor {
    /// Pattern sequence cache for efficiency
    pattern_cache: HashMap<String, u32>,
    /// Entropy calculation cache
    entropy_cache: HashMap<Vec<String>, f64>,
}

impl V2FeatureExtractor {
    pub fn new() -> Self {
        Self {
            pattern_cache: HashMap::new(),
            entropy_cache: HashMap::new(),
        }
    }

    /// Extract V2 features from a sequence of transform changes
    pub fn extract_features(&mut self, changes: &[TransformChangeV2]) -> V2Features {
        if changes.is_empty() {
            return V2Features::default();
        }

        let error_fix_chains = self.count_pattern(changes, &[ChangeType::Error, ChangeType::Fix]);
        let tool_normalize_chains =
            self.count_pattern(changes, &[ChangeType::Tool, ChangeType::Normalize]);
        let rollback_occurred = changes
            .iter()
            .any(|c| matches!(c.change_type, ChangeType::Rollback));
        let edit_depth = changes.iter().map(|c| c.metadata.depth).max().unwrap_or(0);
        let change_entropy = self.calculate_change_entropy(changes);
        let code_error_ratio = self.calculate_code_error_ratio(changes);
        let late_head_edits = self.count_late_head_edits(changes);
        let kv_prefix_impact = self.calculate_kv_prefix_impact(changes);

        V2Features {
            error_fix_chains,
            tool_normalize_chains,
            rollback_occurred,
            edit_depth,
            change_entropy,
            code_error_ratio,
            late_head_edits,
            kv_prefix_impact,
        }
    }

    /// Count occurrences of specific change pattern sequences (ERROR→FIX, TOOL→NORMALIZE)
    fn count_pattern(&mut self, changes: &[TransformChangeV2], pattern: &[ChangeType]) -> u32 {
        if pattern.is_empty() || changes.len() < pattern.len() {
            return 0;
        }

        let pattern_key = format!("{:?}", pattern);
        if let Some(&cached) = self.pattern_cache.get(&pattern_key) {
            return cached;
        }

        let mut count = 0;
        for window in changes.windows(pattern.len()) {
            if window
                .iter()
                .zip(pattern.iter())
                .all(|(change, expected_type)| {
                    std::mem::discriminant(&change.change_type)
                        == std::mem::discriminant(expected_type)
                })
            {
                count += 1;
            }
        }

        self.pattern_cache.insert(pattern_key, count);
        count
    }

    /// Calculate Shannon entropy of change types to measure sequence complexity
    fn calculate_change_entropy(&mut self, changes: &[TransformChangeV2]) -> f64 {
        let change_types: Vec<String> = changes
            .iter()
            .map(|c| format!("{:?}", c.change_type))
            .collect();

        if let Some(&cached) = self.entropy_cache.get(&change_types) {
            return cached;
        }

        let mut frequency_map: HashMap<String, u32> = HashMap::new();
        for change_type in &change_types {
            *frequency_map.entry(change_type.clone()).or_insert(0) += 1;
        }

        let total = change_types.len() as f64;
        let entropy = frequency_map
            .values()
            .map(|&freq| {
                let p = freq as f64 / total;
                if p > 0.0 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum();

        self.entropy_cache.insert(change_types, entropy);
        entropy
    }

    /// Calculate ratio of CODE and ERROR changes to total changes
    fn calculate_code_error_ratio(&self, changes: &[TransformChangeV2]) -> f64 {
        let code_error_count = changes
            .iter()
            .filter(|c| matches!(c.change_type, ChangeType::Code | ChangeType::Error))
            .count();

        if changes.is_empty() {
            0.0
        } else {
            code_error_count as f64 / changes.len() as f64
        }
    }

    /// Count late head summary edits (potentially problematic for KV cache impact)
    fn count_late_head_edits(&self, changes: &[TransformChangeV2]) -> u32 {
        if changes.len() < 2 {
            return 0;
        }

        // Consider edits in the last 20% of the sequence as "late"
        let late_threshold = (changes.len() as f64 * 0.8) as usize;

        changes
            .iter()
            .skip(late_threshold)
            .filter(|c| matches!(c.change_type, ChangeType::HeadSummary))
            .count() as u32
    }

    /// Calculate KV prefix impact using Jaccard similarity drop
    fn calculate_kv_prefix_impact(&self, changes: &[TransformChangeV2]) -> f64 {
        if changes.len() < 2 {
            return 0.0;
        }

        // Simulate prefix stability analysis
        // In a real implementation, this would analyze actual KV cache prefix stability
        let kv_changes = changes
            .iter()
            .filter(|c| {
                matches!(
                    c.change_type,
                    ChangeType::KvUpdate | ChangeType::HeadSummary
                )
            })
            .count();

        let total_changes = changes.len();
        let instability_ratio = kv_changes as f64 / total_changes as f64;

        // Return Jaccard drop approximation - higher values indicate more prefix instability
        instability_ratio.min(1.0)
    }

    /// Reset caches to prevent unbounded memory growth
    pub fn clear_caches(&mut self) {
        self.pattern_cache.clear();
        self.entropy_cache.clear();
    }
}

impl Default for V2Features {
    fn default() -> Self {
        Self {
            error_fix_chains: 0,
            tool_normalize_chains: 0,
            rollback_occurred: false,
            edit_depth: 0,
            change_entropy: 0.0,
            code_error_ratio: 0.0,
            late_head_edits: 0,
            kv_prefix_impact: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChangeMetadata, ChangeType, TransformChangeV2};
    use chrono::Utc;
    use uuid::Uuid;

    fn create_test_change(change_type: ChangeType, depth: u32) -> TransformChangeV2 {
        TransformChangeV2 {
            change_id: Uuid::new_v4(),
            timestamp: Utc::now(),
            change_type,
            metadata: ChangeMetadata {
                depth,
                complexity_score: 1.0,
                edit_distance: Some(10),
                context_size: 100,
                causality_chain: vec![],
            },
            before_state: None,
            after_state: None,
            performance_impact: None,
        }
    }

    #[test]
    fn test_error_fix_chain_detection() {
        let mut extractor = V2FeatureExtractor::new();
        let changes = vec![
            create_test_change(ChangeType::Error, 1),
            create_test_change(ChangeType::Fix, 1),
            create_test_change(ChangeType::Code, 1),
            create_test_change(ChangeType::Error, 1),
            create_test_change(ChangeType::Fix, 1),
        ];

        let features = extractor.extract_features(&changes);
        assert_eq!(features.error_fix_chains, 2);
    }

    #[test]
    fn test_rollback_detection() {
        let mut extractor = V2FeatureExtractor::new();
        let changes = vec![
            create_test_change(ChangeType::Code, 1),
            create_test_change(ChangeType::Rollback, 1),
            create_test_change(ChangeType::Fix, 1),
        ];

        let features = extractor.extract_features(&changes);
        assert!(features.rollback_occurred);
    }

    #[test]
    fn test_entropy_calculation() {
        let mut extractor = V2FeatureExtractor::new();

        // Uniform distribution should have higher entropy
        let uniform_changes = vec![
            create_test_change(ChangeType::Code, 1),
            create_test_change(ChangeType::Error, 1),
            create_test_change(ChangeType::Fix, 1),
            create_test_change(ChangeType::Tool, 1),
        ];

        // Skewed distribution should have lower entropy
        let skewed_changes = vec![
            create_test_change(ChangeType::Code, 1),
            create_test_change(ChangeType::Code, 1),
            create_test_change(ChangeType::Code, 1),
            create_test_change(ChangeType::Error, 1),
        ];

        let uniform_entropy = extractor.extract_features(&uniform_changes).change_entropy;
        let skewed_entropy = extractor.extract_features(&skewed_changes).change_entropy;

        assert!(uniform_entropy > skewed_entropy);
    }

    #[test]
    fn test_code_error_ratio() {
        let mut extractor = V2FeatureExtractor::new();
        let changes = vec![
            create_test_change(ChangeType::Code, 1),
            create_test_change(ChangeType::Error, 1),
            create_test_change(ChangeType::Tool, 1),
            create_test_change(ChangeType::Fix, 1),
        ];

        let features = extractor.extract_features(&changes);
        assert_eq!(features.code_error_ratio, 0.5); // 2 out of 4 changes
    }
}
