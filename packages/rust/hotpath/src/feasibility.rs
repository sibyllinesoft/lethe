use napi::bindgen_prelude::*;
use ahash::{AHashMap, AHashSet};
use rand::{SeedableRng, rngs::StdRng};
use crate::optimization::{InternalAtom, InternalQuota};

pub struct FeasibilityEnforcer {
  rng: StdRng,
}

impl FeasibilityEnforcer {
  pub fn new() -> Self {
    Self {
      rng: StdRng::seed_from_u64(42), // Deterministic seed
    }
  }

  pub fn enforce_constraints(
    &self,
    selected: &[InternalAtom],
    quotas: &[InternalQuota],
    token_budget: u32,
  ) -> napi::Result<Vec<InternalAtom>> {
    // Phase 1: Ancestor closure enforcement
    let closure_satisfied = self.enforce_ancestor_closure(selected)?;
    
    // Phase 2: Type quota satisfaction
    let quota_satisfied = self.satisfy_type_quotas(&closure_satisfied, quotas, token_budget)?;
    
    // Phase 3: 2-swap local search optimization
    let optimized = self.two_swap_optimization(&quota_satisfied, token_budget)?;

    Ok(optimized)
  }

  fn enforce_ancestor_closure(&self, selected: &[InternalAtom]) -> napi::Result<Vec<InternalAtom>> {
    let mut result = selected.to_vec();
    let mut atom_map: AHashMap<String, InternalAtom> = result
      .iter()
      .map(|atom| (atom.id.clone(), atom.clone()))
      .collect();

    let mut changed = true;
    while changed {
      changed = false;
      let current_ids: AHashSet<String> = atom_map.keys().cloned().collect();
      
      for atom in result.iter() {
        for dep_id in &atom.dependencies {
          if !current_ids.contains(dep_id) {
            // In a real implementation, we'd look up the missing dependency
            // For now, we'll skip missing dependencies
            log::warn!("Missing dependency: {}", dep_id);
          }
        }
      }
    }

    Ok(result)
  }

  fn satisfy_type_quotas(
    &self,
    atoms: &[InternalAtom],
    quotas: &[InternalQuota],
    token_budget: u32,
  ) -> napi::Result<Vec<InternalAtom>> {
    let mut result = Vec::new();
    let mut type_tokens: AHashMap<String, u32> = AHashMap::new();
    let mut total_tokens = 0u32;

    // Group atoms by type
    let mut atoms_by_type: AHashMap<String, Vec<&InternalAtom>> = AHashMap::new();
    for atom in atoms {
      atoms_by_type
        .entry(atom.chunk_type.clone())
        .or_insert_with(Vec::new)
        .push(atom);
    }

    // Sort atoms within each type by importance (descending)
    for type_atoms in atoms_by_type.values_mut() {
      type_atoms.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal));
    }

    // Satisfy minimum quotas first
    for quota in quotas {
      if let Some(candidates) = atoms_by_type.get(&quota.chunk_type) {
        let selected_for_type = self.knapsack_selection(
          candidates,
          quota.min_tokens,
          token_budget.saturating_sub(total_tokens),
        );
        
        for atom in selected_for_type {
          if total_tokens + atom.tokens <= token_budget {
            result.push(atom.clone());
            total_tokens += atom.tokens;
            *type_tokens.entry(quota.chunk_type.clone()).or_insert(0) += atom.tokens;
          }
        }
      }
    }

    Ok(result)
  }

  fn knapsack_selection(
    &self,
    candidates: &[&InternalAtom],
    min_tokens: u32,
    max_budget: u32,
  ) -> Vec<InternalAtom> {
    if candidates.is_empty() || max_budget == 0 {
      return Vec::new();
    }

    // Simple greedy knapsack approximation
    let mut selected = Vec::new();
    let mut current_tokens = 0u32;
    let mut remaining_candidates = candidates.to_vec();
    
    // Sort by value-to-weight ratio (importance per token)
    remaining_candidates.sort_by(|a, b| {
      let ratio_a = a.importance / (a.tokens as f64 + 1.0);
      let ratio_b = b.importance / (b.tokens as f64 + 1.0);
      ratio_b.partial_cmp(&ratio_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    for candidate in remaining_candidates {
      if current_tokens + candidate.tokens <= max_budget {
        selected.push(candidate.clone());
        current_tokens += candidate.tokens;
        
        if current_tokens >= min_tokens {
          break;
        }
      }
    }

    selected
  }

  fn two_swap_optimization(
    &self,
    atoms: &[InternalAtom],
    token_budget: u32,
  ) -> napi::Result<Vec<InternalAtom>> {
    let mut result = atoms.to_vec();
    let mut improved = true;

    while improved {
      improved = false;
      
      for i in 0..result.len() {
        for j in i + 1..result.len() {
          // Try swapping atoms[i] and atoms[j]
          let current_score = self.evaluate_selection(&result);
          
          // Perform swap
          result.swap(i, j);
          let new_score = self.evaluate_selection(&result);
          
          if new_score > current_score && self.check_token_budget(&result, token_budget) {
            improved = true;
          } else {
            // Revert swap
            result.swap(i, j);
          }
        }
      }
    }

    Ok(result)
  }

  fn evaluate_selection(&self, atoms: &[InternalAtom]) -> f64 {
    // Simple evaluation based on total importance
    atoms.iter().map(|atom| atom.importance).sum()
  }

  fn check_token_budget(&self, atoms: &[InternalAtom], budget: u32) -> bool {
    let total_tokens: u32 = atoms.iter().map(|atom| atom.tokens).sum();
    total_tokens <= budget
  }
}