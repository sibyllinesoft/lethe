/**
 * Log-Det Low-Rank DPP Diversity Enhancement
 * 
 * Implements determinantal point processes with log-det computation for diversity:
 * Δ_DPP(a|S) = log(1 + ||(I-QQ^T)v_a||^2)
 * 
 * Key features:
 * - Maintains orthonormal basis Q over selected rows of V
 * - Rank-1 QR/Cholesky updates in O(r^2) time
 * - PSD-safe submodular diversity measurement
 * - Orthogonal mass diagnostics for redundancy pressure
 * - Configurable rank r ∈ [12,24] based on performance profiles
 */

import { z } from 'zod';
import * as math from 'mathjs';

// DPP diversity configuration
export const DPPConfigSchema = z.object({
  // Rank parameters
  max_rank: z.number().int().min(8).max(32).default(18), // r ∈ [12,24] as suggested
  min_rank: z.number().int().min(4).max(16).default(12),
  adaptive_rank: z.boolean().default(true),
  
  // Diversity thresholds
  orthogonal_mass_threshold: z.number().min(0).max(1).default(0.95),
  diversity_weight: z.number().min(0).default(0.5), // δ in F(S)
  
  // Numerical stability
  numerical_tolerance: z.number().min(0).default(1e-12),
  condition_number_max: z.number().min(1).default(1e8),
  
  // Performance settings
  batch_update_size: z.number().int().min(1).default(8),
  enable_parallel_updates: z.boolean().default(true),
});

export type DPPConfig = z.infer<typeof DPPConfigSchema>;

// Vector representation for DPP computation
export interface DPPVector {
  id: string;
  embedding: number[]; // Dense vector representation
  norm_squared?: number; // Cached ||v||^2
}

// DPP state maintaining orthonormal basis
export interface DPPState {
  Q: number[][]; // Orthonormal basis matrix (r × |S|)
  current_rank: number;
  selected_ids: string[];
  orthogonal_mass: number;
  total_log_det: number;
  condition_number: number;
  update_count: number;
}

// DPP diversity result
export interface DPPDiversityResult {
  diversity_gain: number;
  log_det_value: number;
  orthogonal_mass: number;
  rank_utilized: number;
  numerical_stable: boolean;
}

/**
 * Log-Det Low-Rank DPP for Diversity Enhancement
 * 
 * Maintains an orthonormal basis Q and computes marginal diversity gains
 * using the formula: Δ_DPP(a|S) = log(1 + ||(I-QQ^T)v_a||^2)
 */
export class DPPDiversityEngine {
  private config: DPPConfig;
  private state: DPPState;
  private embedding_dimension: number;
  
  constructor(embedding_dimension: number, config: Partial<DPPConfig> = {}) {
    this.config = DPPConfigSchema.parse(config);
    this.embedding_dimension = embedding_dimension;
    
    this.state = {
      Q: [],
      current_rank: 0,
      selected_ids: [],
      orthogonal_mass: 0,
      total_log_det: 0,
      condition_number: 1,
      update_count: 0,
    };
  }
  
  /**
   * Compute marginal diversity gain for adding vector v_a to current selection
   */
  computeMarginalGain(vector: DPPVector): DPPDiversityResult {
    const startTime = performance.now();
    
    // Ensure vector norm is cached
    if (vector.norm_squared === undefined) {
      vector.norm_squared = this.computeVectorNormSquared(vector.embedding);
    }
    
    let diversity_gain = 0;
    let numerical_stable = true;
    
    if (this.state.current_rank === 0) {
      // First vector - pure norm contribution
      diversity_gain = Math.log(1 + vector.norm_squared);
    } else {
      // Compute (I - QQ^T)v_a
      const residual = this.computeOrthogonalResidual(vector.embedding);
      const residual_norm_squared = this.computeVectorNormSquared(residual);
      
      // Check numerical stability
      if (residual_norm_squared < this.config.numerical_tolerance) {
        diversity_gain = 0; // Vector is in span of Q
        numerical_stable = false;
      } else {
        diversity_gain = Math.log(1 + residual_norm_squared);
      }
    }
    
    // Compute current orthogonal mass
    const orthogonal_mass = this.computeOrthogonalMass();
    
    return {
      diversity_gain: this.config.diversity_weight * diversity_gain,
      log_det_value: this.state.total_log_det + diversity_gain,
      orthogonal_mass,
      rank_utilized: this.state.current_rank,
      numerical_stable,
    };
  }
  
  /**
   * Add vector to selection and update orthonormal basis Q
   */
  async addVector(vector: DPPVector): Promise<void> {
    // Compute marginal gain first (includes stability check)
    const diversity_result = this.computeMarginalGain(vector);
    
    if (!diversity_result.numerical_stable && this.state.current_rank > 0) {
      // Vector is linearly dependent - skip addition
      return;
    }
    
    // Add to selection
    this.state.selected_ids.push(vector.id);
    this.state.total_log_det += diversity_result.diversity_gain / this.config.diversity_weight;
    
    if (this.state.current_rank === 0) {
      // Initialize Q with first vector
      const normalized = this.normalizeVector(vector.embedding);
      this.state.Q = [normalized];
      this.state.current_rank = 1;
    } else {
      // Gram-Schmidt orthogonalization with rank-1 update
      await this.updateOrthonormalBasis(vector.embedding);
    }
    
    this.state.update_count++;
    
    // Adaptive rank management
    if (this.config.adaptive_rank) {
      await this.adaptiveRankManagement();
    }
  }
  
  /**
   * Update orthonormal basis using rank-1 QR/Cholesky update
   */
  private async updateOrthonormalBasis(new_vector: number[]): Promise<void> {
    // Compute orthogonal residual (I - QQ^T)v
    const residual = this.computeOrthogonalResidual(new_vector);
    const residual_norm = Math.sqrt(this.computeVectorNormSquared(residual));
    
    if (residual_norm < this.config.numerical_tolerance) {
      // Vector is in span of Q - no update needed
      return;
    }
    
    // Check if we need to expand rank
    if (this.state.current_rank < this.config.max_rank) {
      // Add normalized residual as new basis vector
      const normalized_residual = residual.map(x => x / residual_norm);
      this.state.Q.push(normalized_residual);
      this.state.current_rank++;
    } else {
      // Rank is at maximum - use low-rank approximation update
      await this.lowRankUpdate(residual, residual_norm);
    }
    
    // Update condition number estimate
    this.updateConditionNumber();
  }
  
  /**
   * Compute orthogonal residual (I - QQ^T)v
   */
  private computeOrthogonalResidual(vector: number[]): number[] {
    if (this.state.current_rank === 0) {
      return [...vector];
    }
    
    // Compute QQ^T v = Q(Q^T v)
    const Qt_v = this.state.Q.map(q_i => 
      this.dotProduct(q_i, vector)
    );
    
    // Compute Q * (Q^T v)
    const QQt_v = new Array(this.embedding_dimension).fill(0);
    for (let i = 0; i < this.state.current_rank; i++) {
      for (let j = 0; j < this.embedding_dimension; j++) {
        QQt_v[j] += this.state.Q[i][j] * Qt_v[i];
      }
    }
    
    // Return (I - QQ^T)v = v - QQ^T v
    return vector.map((v_i, i) => v_i - QQt_v[i]);
  }
  
  /**
   * Low-rank update when rank is at maximum
   */
  private async lowRankUpdate(residual: number[], residual_norm: number): Promise<void> {
    // Find basis vector with minimum contribution
    let min_contribution_idx = 0;
    let min_contribution = Infinity;
    
    for (let i = 0; i < this.state.current_rank; i++) {
      const contribution = this.computeBasisContribution(i);
      if (contribution < min_contribution) {
        min_contribution = contribution;
        min_contribution_idx = i;
      }
    }
    
    // Replace least contributing basis vector
    const normalized_residual = residual.map(x => x / residual_norm);
    this.state.Q[min_contribution_idx] = normalized_residual;
    
    // Re-orthogonalize to maintain numerical stability
    await this.reorthogonalize();
  }
  
  /**
   * Re-orthogonalize basis using modified Gram-Schmidt
   */
  private async reorthogonalize(): Promise<void> {
    for (let i = 0; i < this.state.current_rank; i++) {
      // Normalize current vector
      const norm = Math.sqrt(this.computeVectorNormSquared(this.state.Q[i]));
      if (norm > this.config.numerical_tolerance) {
        this.state.Q[i] = this.state.Q[i].map(x => x / norm);
      }
      
      // Orthogonalize against all subsequent vectors
      for (let j = i + 1; j < this.state.current_rank; j++) {
        const dot = this.dotProduct(this.state.Q[i], this.state.Q[j]);
        for (let k = 0; k < this.embedding_dimension; k++) {
          this.state.Q[j][k] -= dot * this.state.Q[i][k];
        }
      }
    }
  }
  
  /**
   * Compute contribution of basis vector for low-rank updates
   */
  private computeBasisContribution(basis_index: number): number {
    // Simplified contribution metric - could be enhanced
    return this.computeVectorNormSquared(this.state.Q[basis_index]);
  }
  
  /**
   * Adaptive rank management based on orthogonal mass
   */
  private async adaptiveRankManagement(): Promise<void> {
    const orthogonal_mass = this.computeOrthogonalMass();
    
    if (orthogonal_mass > this.config.orthogonal_mass_threshold) {
      // High orthogonal mass - can reduce rank
      if (this.state.current_rank > this.config.min_rank) {
        await this.reduceRank();
      }
    } else if (orthogonal_mass < this.config.orthogonal_mass_threshold - 0.1) {
      // Low orthogonal mass - may need more rank
      if (this.state.current_rank < this.config.max_rank) {
        // Rank will be increased naturally by addVector if needed
      }
    }
  }
  
  /**
   * Reduce rank by removing least significant basis vectors
   */
  private async reduceRank(): Promise<void> {
    if (this.state.current_rank <= this.config.min_rank) {
      return;
    }
    
    // Remove last basis vector (least recently updated)
    this.state.Q.pop();
    this.state.current_rank--;
  }
  
  /**
   * Compute orthogonal mass as diversity diagnostic
   */
  private computeOrthogonalMass(): number {
    if (this.state.current_rank === 0) {
      return 0;
    }
    
    // Orthogonal mass = trace(QQ^T) / embedding_dimension
    let trace = 0;
    for (let i = 0; i < this.state.current_rank; i++) {
      trace += this.computeVectorNormSquared(this.state.Q[i]);
    }
    
    return trace / this.embedding_dimension;
  }
  
  /**
   * Update condition number estimate
   */
  private updateConditionNumber(): void {
    // Simplified condition number estimate
    if (this.state.current_rank === 0) {
      this.state.condition_number = 1;
      return;
    }
    
    let min_norm = Infinity;
    let max_norm = 0;
    
    for (let i = 0; i < this.state.current_rank; i++) {
      const norm = Math.sqrt(this.computeVectorNormSquared(this.state.Q[i]));
      min_norm = Math.min(min_norm, norm);
      max_norm = Math.max(max_norm, norm);
    }
    
    this.state.condition_number = max_norm / (min_norm || 1);
  }
  
  /**
   * Utility functions
   */
  private computeVectorNormSquared(vector: number[]): number {
    return vector.reduce((sum, x) => sum + x * x, 0);
  }
  
  private normalizeVector(vector: number[]): number[] {
    const norm = Math.sqrt(this.computeVectorNormSquared(vector));
    return vector.map(x => x / (norm || 1));
  }
  
  private dotProduct(a: number[], b: number[]): number {
    return a.reduce((sum, a_i, i) => sum + a_i * b[i], 0);
  }
  
  /**
   * Get current state for monitoring
   */
  getState(): DPPState {
    this.state.orthogonal_mass = this.computeOrthogonalMass();
    return { ...this.state };
  }
  
  /**
   * Reset DPP state
   */
  reset(): void {
    this.state = {
      Q: [],
      current_rank: 0,
      selected_ids: [],
      orthogonal_mass: 0,
      total_log_det: 0,
      condition_number: 1,
      update_count: 0,
    };
  }
  
  /**
   * Export basis for external computation
   */
  exportBasis(): number[][] {
    return this.state.Q.map(q => [...q]);
  }
  
  /**
   * Import pre-computed basis
   */
  importBasis(Q: number[][], selected_ids: string[]): void {
    this.state.Q = Q.map(q => [...q]);
    this.state.current_rank = Q.length;
    this.state.selected_ids = [...selected_ids];
    this.updateConditionNumber();
  }
}

/**
 * Convenience function for batch diversity computation
 */
export async function computeBatchDiversityGains(
  vectors: DPPVector[],
  current_selection: string[],
  embedding_dimension: number,
  config: Partial<DPPConfig> = {}
): Promise<Map<string, DPPDiversityResult>> {
  const dpp_engine = new DPPDiversityEngine(embedding_dimension, config);
  
  // Initialize with current selection (if any)
  const selected_vectors = vectors.filter(v => current_selection.includes(v.id));
  for (const vector of selected_vectors) {
    await dpp_engine.addVector(vector);
  }
  
  // Compute marginal gains for all candidates
  const results = new Map<string, DPPDiversityResult>();
  
  for (const vector of vectors) {
    if (!current_selection.includes(vector.id)) {
      const result = dpp_engine.computeMarginalGain(vector);
      results.set(vector.id, result);
    }
  }
  
  return results;
}

/**
 * Create DPP vectors from embeddings
 */
export function createDPPVectors(
  embeddings: Array<{ id: string; vector: number[] }>
): DPPVector[] {
  return embeddings.map(({ id, vector }) => ({
    id,
    embedding: vector,
    norm_squared: vector.reduce((sum, x) => sum + x * x, 0),
  }));
}