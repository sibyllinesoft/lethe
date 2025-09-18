use napi::bindgen_prelude::*;
use rayon::prelude::*;
use ahash::AHashSet;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
pub struct InternalAtom {
  pub id: String,
  pub tokens: u32,
  pub chunk_type: String,
  pub importance: f64,
  pub dependencies: Vec<String>,
  pub text_start: usize,
  pub text_len: usize,
}

#[derive(Debug, Clone)]
pub struct InternalQuota {
  pub chunk_type: String,
  pub min_tokens: u32,
  pub target_ratio: f64,
}

pub struct OptimizationEngine {
  simd_hasher: SimdHasher,
}

struct SimdHasher {
  // Simplified SIMD implementation without wide crate complexities
}

impl SimdHasher {
  fn new() -> Self {
    Self {}
  }

  fn compute_hash(&self, text: &[u8]) -> u64 {
    // Simplified hash implementation using standard library
    use std::collections::hash_map::DefaultHasher;
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
  }

  fn parallel_minhash(&self, texts: &[&[u8]], num_hashes: usize) -> Vec<Vec<u64>> {
    texts
      .par_iter()
      .map(|text| {
        (0..num_hashes)
          .map(|seed| {
            let mut combined = Vec::with_capacity(text.len() + 8);
            combined.extend_from_slice(text);
            combined.extend_from_slice(&seed.to_le_bytes());
            self.compute_hash(&combined)
          })
          .collect()
      })
      .collect()
  }
}

impl OptimizationEngine {
  pub fn new() -> Self {
    Self {
      simd_hasher: SimdHasher::new(),
    }
  }

  pub fn optimize_selection(
    &self,
    atoms: &[InternalAtom],
    quotas: &[InternalQuota],
    token_budget: u32,
    lambda_threshold: f64,
    text_buffer: &[u8],
  ) -> napi::Result<Vec<InternalAtom>> {
    // Phase 1: S0 Streaming with deduplication
    let filtered_atoms = self.s0_streaming_with_dedup(atoms, text_buffer)?;
    
    // Phase 2: Lazy-greedy with marginal gains  
    let selected = self.lazy_greedy_with_marginal_gains(&filtered_atoms, token_budget, lambda_threshold)?;
    
    // Phase 3: Feasibility enforcement
    let feasible = self.enforce_feasibility(&selected, quotas, token_budget)?;

    Ok(feasible)
  }

  fn s0_streaming_with_dedup(
    &self,
    atoms: &[InternalAtom],
    text_buffer: &[u8],
  ) -> napi::Result<Vec<InternalAtom>> {
    const NUM_HASHES: usize = 64;
    const SIMILARITY_THRESHOLD: f64 = 0.8;

    // Extract text segments for each atom
    let text_segments: Vec<&[u8]> = atoms
      .iter()
      .filter_map(|atom| {
        if atom.text_start + atom.text_len <= text_buffer.len() {
          Some(&text_buffer[atom.text_start..atom.text_start + atom.text_len])
        } else {
          None
        }
      })
      .collect();

    // Compute MinHash signatures in parallel
    let signatures = self.simd_hasher.parallel_minhash(&text_segments, NUM_HASHES);

    // Streaming deduplication
    let mut unique_atoms = Vec::new();
    let mut seen_signatures: Vec<Vec<u64>> = Vec::new();

    for (i, atom) in atoms.iter().enumerate() {
      if i >= signatures.len() {
        continue;
      }

      let current_sig = &signatures[i];
      let mut is_unique = true;

      // Check similarity with all previously seen atoms
      for prev_sig in &seen_signatures {
        let similarity = jaccard_similarity(current_sig, prev_sig);
        if similarity > SIMILARITY_THRESHOLD {
          is_unique = false;
          break;
        }
      }

      if is_unique {
        unique_atoms.push(atom.clone());
        seen_signatures.push(current_sig.clone());
      }
    }

    Ok(unique_atoms)
  }

  fn lazy_greedy_with_marginal_gains(
    &self,
    atoms: &[InternalAtom],
    token_budget: u32,
    lambda_threshold: f64,
  ) -> napi::Result<Vec<InternalAtom>> {
    let mut selected = Vec::new();
    let mut remaining: AHashSet<_> = (0..atoms.len()).collect();
    let mut current_tokens = 0u32;

    // Initialize marginal gains
    let mut marginal_gains: Vec<f64> = atoms
      .par_iter()
      .map(|atom| self.compute_initial_gain(atom))
      .collect();

    while current_tokens < token_budget && !remaining.is_empty() {
      // Find atom with highest marginal gain
      let mut best_idx = None;
      let mut best_gain = lambda_threshold;

      for &idx in &remaining {
        if current_tokens + atoms[idx].tokens <= token_budget && marginal_gains[idx] > best_gain {
          best_gain = marginal_gains[idx];
          best_idx = Some(idx);
        }
      }

      if let Some(idx) = best_idx {
        selected.push(atoms[idx].clone());
        current_tokens += atoms[idx].tokens;
        remaining.remove(&idx);

        // Update marginal gains for remaining atoms
        self.update_marginal_gains(&mut marginal_gains, &atoms, &selected, &remaining);
      } else {
        break;
      }
    }

    Ok(selected)
  }

  fn compute_initial_gain(&self, atom: &InternalAtom) -> f64 {
    // Simplified gain computation based on importance and token efficiency
    atom.importance * (1.0 / (atom.tokens as f64 + 1.0))
  }

  fn update_marginal_gains(
    &self,
    gains: &mut [f64],
    _atoms: &[InternalAtom],
    _selected: &[InternalAtom],
    remaining: &AHashSet<usize>,
  ) {
    // Update gains based on coverage and diversity impact
    for &idx in remaining {
      gains[idx] *= 0.95; // Simple diminishing returns model
    }
  }

  fn enforce_feasibility(
    &self,
    selected: &[InternalAtom],
    quotas: &[InternalQuota],
    token_budget: u32,
  ) -> napi::Result<Vec<InternalAtom>> {
    use crate::feasibility::FeasibilityEnforcer;
    
    let enforcer = FeasibilityEnforcer::new();
    enforcer.enforce_constraints(selected, quotas, token_budget)
  }
}

fn jaccard_similarity(sig1: &[u64], sig2: &[u64]) -> f64 {
  let matches = sig1.iter().zip(sig2.iter()).filter(|(a, b)| a == b).count();
  matches as f64 / sig1.len() as f64
}