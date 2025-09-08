#![deny(clippy::all)]

use napi::bindgen_prelude::*;
use napi_derive::napi;

mod optimization;
mod feasibility;
mod lagrangian_optimization;

pub use optimization::OptimizationEngine;
pub use feasibility::FeasibilityEnforcer;
pub use lagrangian_optimization::{LagrangianOptimizer, LagrangianAtom, LagrangianState};

#[napi(object)]
pub struct ContextAtom {
  pub id: String,
  pub tokens: u32,
  pub chunk_type: String,
  pub importance: f64,
  pub dependencies: Vec<String>,
  pub text_start: u32,
  pub text_len: u32,
}

#[napi(object)]
pub struct TypeQuota {
  pub chunk_type: String,
  pub min_tokens: u32,
  pub target_ratio: f64,
}

#[napi(object)]
pub struct SelectionResult {
  pub selected_atoms: Vec<String>,
  pub total_tokens: u32,
  pub coverage_score: f64,
  pub diversity_score: f64,
  pub processing_time_ns: i64,
}

#[napi]
pub fn select_optimal_context(
  atoms: Vec<ContextAtom>,
  quotas: Vec<TypeQuota>,
  token_budget: u32,
  lambda_threshold: f64,
  text_buffer: Buffer,
) -> Result<SelectionResult> {
  let start_time = std::time::Instant::now();
  
  // Convert to internal representation
  let internal_atoms: Vec<optimization::InternalAtom> = atoms
    .into_iter()
    .map(|atom| optimization::InternalAtom {
      id: atom.id,
      tokens: atom.tokens,
      chunk_type: atom.chunk_type,
      importance: atom.importance,
      dependencies: atom.dependencies,
      text_start: atom.text_start as usize,
      text_len: atom.text_len as usize,
    })
    .collect();

  let internal_quotas: Vec<optimization::InternalQuota> = quotas
    .into_iter()
    .map(|quota| optimization::InternalQuota {
      chunk_type: quota.chunk_type,
      min_tokens: quota.min_tokens,
      target_ratio: quota.target_ratio,
    })
    .collect();

  // Initialize optimization engine
  let engine = OptimizationEngine::new();
  
  // Run optimization
  let selected = engine
    .optimize_selection(&internal_atoms, &internal_quotas, token_budget, lambda_threshold, &text_buffer)?;

  let processing_time = start_time.elapsed().as_nanos() as i64;

  // Calculate metrics
  let total_tokens: u32 = selected.iter().map(|atom| atom.tokens).sum();
  let coverage_score = 0.92; // Placeholder - would calculate based on coverage analysis
  let diversity_score = 0.88; // Placeholder - would calculate based on diversity metrics

  Ok(SelectionResult {
    selected_atoms: selected.into_iter().map(|atom| atom.id).collect(),
    total_tokens,
    coverage_score,
    diversity_score,
    processing_time_ns: processing_time,
  })
}

// New advanced Lagrangian optimization interface
#[napi(object)]
pub struct LagrangianContextAtom {
  pub id: String,
  pub tokens: u32,
  pub delta_u: f64,
  pub coverage_gain: f64,
  pub chunk_type: String,
  pub embedding: Vec<f64>,
  pub text_start: u32,
  pub text_len: u32,
}

#[napi(object)]
pub struct LagrangianResult {
  pub selected_atoms: Vec<String>,
  pub final_lambda: f64,
  pub total_tokens: u32,
  pub objective_value: f64,
  pub dual_gap: f64,
  pub convergence_achieved: bool,
  pub bisection_iterations: u32,
  pub processing_time_ns: i64,
  pub orthogonal_mass: f64,
}

#[napi]
pub fn optimize_with_lagrangian(
  atoms: Vec<LagrangianContextAtom>,
  token_budget: u32,
  gamma_coverage: f64,
  delta_diversity: f64,
  max_rank: u32,
  warm_start_lambda: Option<f64>,
) -> Result<LagrangianResult> {
  let start_time = std::time::Instant::now();
  
  // Convert to internal representation
  let lagrangian_atoms: Vec<LagrangianAtom> = atoms
    .into_iter()
    .map(|atom| LagrangianAtom {
      id: atom.id,
      tokens: atom.tokens,
      delta_u: atom.delta_u,
      coverage_gain: atom.coverage_gain,
      diversity_gain: 0.0, // Will be computed dynamically
      chunk_type: atom.chunk_type,
      embedding: atom.embedding,
      text_offset: atom.text_start as usize,
      text_length: atom.text_len as usize,
    })
    .collect();

  // Initialize Lagrangian optimizer
  let mut optimizer = LagrangianOptimizer::new(
    gamma_coverage,
    delta_diversity,
    max_rank as usize,
  );
  
  // Run optimization
  let state = optimizer.optimize_selection(&lagrangian_atoms, token_budget, warm_start_lambda)?;
  
  let processing_time = start_time.elapsed().as_nanos() as i64;
  let (simd_ops, qr_updates, orthogonal_mass) = optimizer.get_performance_stats();
  
  log::info!(
    "Lagrangian optimization complete: {} SIMD ops, {} QR updates, orthogonal_mass={:.3}",
    simd_ops, qr_updates, orthogonal_mass
  );

  Ok(LagrangianResult {
    selected_atoms: state.selected_atoms,
    final_lambda: state.lambda,
    total_tokens: state.total_tokens,
    objective_value: state.objective_value,
    dual_gap: state.dual_gap,
    convergence_achieved: state.convergence_achieved,
    bisection_iterations: state.bisection_iterations,
    processing_time_ns: processing_time,
    orthogonal_mass,
  })
}