/**
 * Hierarchical Interleaving System for Multi-turn Task Attribution
 * 
 * Implements the sophisticated interleaving strategy described in TODO.md:
 * - Atom-level interleaving within turns
 * - Cluster-pair interleaving at session level (turns t...t+k) 
 * - Session-level attribution for multi-turn plan/tool outcomes
 * - Statistical power calculations for ΔnDCG@10 improvements
 * 
 * This addresses the credit assignment problem in multi-turn utility where
 * single-turn ranking can invert due to plan/tool outcome attribution issues.
 */

import type { DB } from '@lethe/sqlite';
import type { Candidate } from './index.js';

// Core interfaces for hierarchical interleaving
export interface Turn {
  turn_id: string;
  session_id: string;
  turn_index: number;
  timestamp: string;
  query: string;
  candidates: InterleavingCandidate[];
  selected_atoms: string[];
  plan_outcome?: PlanOutcome;
  tool_results?: ToolResult[];
}

export interface InterleavingCandidate extends Candidate {
  atom_id: string;
  cluster_id: string;
  interleaving_position: number;
  attribution_weight: number;
  within_turn_rank: number;
  cross_turn_influence: number;
}

export interface PlanOutcome {
  plan_type: 'explore' | 'exploit' | 'verify';
  success_metric: number;
  attribution_scope: 'single_turn' | 'multi_turn';
  dependent_turns: string[]; // Turn IDs this outcome depends on
}

export interface ToolResult {
  tool_id: string;
  execution_success: boolean;
  contribution_to_session: number;
  causal_dependencies: string[]; // Atom IDs that led to this tool use
}

export interface SessionCluster {
  cluster_id: string;
  turn_range: [number, number]; // [start_turn, end_turn]
  atoms: InterleavingCandidate[];
  cluster_coherence: number;
  cross_cluster_influence: Map<string, number>; // Influence on other clusters
}

export interface HierarchicalInterleavingConfig {
  // Atom-level interleaving within turns
  atom_interleaving: {
    max_atoms_per_turn: number;
    position_weighting: 'linear' | 'exponential' | 'balanced';
    diversity_enforcement: boolean;
  };
  
  // Cluster-pair interleaving at session level  
  cluster_interleaving: {
    max_clusters_per_session: number;
    temporal_decay_factor: number; // Influence decay over turns
    cluster_coherence_threshold: number;
    cross_cluster_boost: number;
  };
  
  // Attribution and credit assignment
  attribution: {
    attribution_window: number; // Number of turns to consider for attribution
    plan_outcome_weight: number; // Weight for plan success attribution
    tool_outcome_weight: number; // Weight for tool success attribution
    temporal_discount_factor: number; // Discount factor for distant contributions
  };
  
  // Statistical validation
  statistical: {
    target_delta_ndcg: number; // +2pp ΔnDCG@10 as per TODO.md
    session_variance_assumption: number; // σ²≈0.08
    required_statistical_power: number; // 80%
    early_stopping_significance: number; // Holm-corrected significance
  };
}

export const DEFAULT_HIERARCHICAL_CONFIG: HierarchicalInterleavingConfig = {
  atom_interleaving: {
    max_atoms_per_turn: 10,
    position_weighting: 'balanced',
    diversity_enforcement: true,
  },
  cluster_interleaving: {
    max_clusters_per_session: 5,
    temporal_decay_factor: 0.9,
    cluster_coherence_threshold: 0.7,
    cross_cluster_boost: 0.15,
  },
  attribution: {
    attribution_window: 5,
    plan_outcome_weight: 0.6,
    tool_outcome_weight: 0.4,
    temporal_discount_factor: 0.85,
  },
  statistical: {
    target_delta_ndcg: 0.02, // +2pp improvement target
    session_variance_assumption: 0.08, // σ²≈0.08 from TODO.md
    required_statistical_power: 0.8, // 80% power
    early_stopping_significance: 0.05, // 5% significance with Holm correction
  },
};

/**
 * Hierarchical Interleaving Engine
 * 
 * Manages multi-level interleaving and attribution across session turns
 */
export class HierarchicalInterleavingEngine {
  private db: DB;
  private config: HierarchicalInterleavingConfig;
  private sessionState: Map<string, SessionState> = new Map();

  constructor(db: DB, config: Partial<HierarchicalInterleavingConfig> = {}) {
    this.db = db;
    this.config = { ...DEFAULT_HIERARCHICAL_CONFIG, ...config };
  }

  /**
   * Execute hierarchical interleaving for a multi-turn session
   * Returns interleaved candidates with session-level attribution
   */
  async executeHierarchicalInterleaving(
    sessionId: string,
    currentTurn: Turn,
    previousTurns: Turn[]
  ): Promise<HierarchicalInterleavingResult> {
    const startTime = performance.now();

    console.log(`🔀 Executing hierarchical interleaving for session ${sessionId}, turn ${currentTurn.turn_index}`);

    try {
      // Step 1: Build or update session clusters
      const sessionClusters = await this.buildSessionClusters(sessionId, currentTurn, previousTurns);

      // Step 2: Execute atom-level interleaving within current turn
      const atomInterleavedCandidates = await this.executeAtomLevelInterleaving(
        currentTurn,
        sessionClusters
      );

      // Step 3: Execute cluster-pair interleaving at session level
      const clusterInterleavedResult = await this.executeClusterPairInterleaving(
        sessionClusters,
        currentTurn.turn_index
      );

      // Step 4: Attribution calculation for plan/tool outcomes
      const attributionResult = await this.calculateSessionLevelAttribution(
        sessionId,
        currentTurn,
        previousTurns,
        sessionClusters
      );

      // Step 5: Statistical power validation
      const statisticalValidation = await this.validateStatisticalPower(
        sessionId,
        attributionResult
      );

      const processingTime = performance.now() - startTime;

      const result: HierarchicalInterleavingResult = {
        session_id: sessionId,
        turn_id: currentTurn.turn_id,
        interleaved_candidates: atomInterleavedCandidates,
        session_clusters: sessionClusters,
        cluster_interleaving: clusterInterleavedResult,
        attribution: attributionResult,
        statistical_validation: statisticalValidation,
        processing_metrics: {
          total_processing_time_ms: processingTime,
          atoms_processed: atomInterleavedCandidates.length,
          clusters_formed: sessionClusters.length,
          attribution_calculations: attributionResult.attribution_scores.size,
        },
      };

      // Update session state
      this.updateSessionState(sessionId, result);

      console.log(`✅ Hierarchical interleaving complete: ${atomInterleavedCandidates.length} atoms, ${sessionClusters.length} clusters, ${processingTime.toFixed(1)}ms`);

      return result;

    } catch (error) {
      console.error(`❌ Hierarchical interleaving failed: ${error}`);
      throw new Error(`Hierarchical interleaving failed: ${error}`);
    }
  }

  /**
   * Step 1: Build session clusters for cross-turn coherence
   */
  private async buildSessionClusters(
    sessionId: string,
    currentTurn: Turn,
    previousTurns: Turn[]
  ): Promise<SessionCluster[]> {
    console.log('🧩 Building session clusters...');

    const allTurns = [...previousTurns, currentTurn];
    const clusters: SessionCluster[] = [];

    // Extract all atoms from all turns
    const allAtoms: Array<{ atom: InterleavingCandidate; turn_index: number }> = [];
    for (const turn of allTurns) {
      for (const candidate of turn.candidates) {
        allAtoms.push({
          atom: candidate,
          turn_index: turn.turn_index,
        });
      }
    }

    // Cluster atoms based on semantic and temporal similarity
    const clusterAssignments = await this.clusterAtomsBySimilarity(allAtoms);

    // Build cluster objects with metadata
    for (const [clusterId, atomIndices] of clusterAssignments.entries()) {
      const clusterAtoms = atomIndices.map(idx => allAtoms[idx].atom);
      const turnIndices = atomIndices.map(idx => allAtoms[idx].turn_index);
      
      const cluster: SessionCluster = {
        cluster_id: clusterId,
        turn_range: [Math.min(...turnIndices), Math.max(...turnIndices)],
        atoms: clusterAtoms,
        cluster_coherence: this.calculateClusterCoherence(clusterAtoms),
        cross_cluster_influence: new Map(),
      };

      clusters.push(cluster);
    }

    // Calculate cross-cluster influences
    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        const influence = this.calculateCrossClusterInfluence(clusters[i], clusters[j]);
        clusters[i].cross_cluster_influence.set(clusters[j].cluster_id, influence);
        clusters[j].cross_cluster_influence.set(clusters[i].cluster_id, influence);
      }
    }

    console.log(`🧩 Built ${clusters.length} session clusters`);
    return clusters;
  }

  /**
   * Step 2: Atom-level interleaving within current turn
   */
  private async executeAtomLevelInterleaving(
    currentTurn: Turn,
    sessionClusters: SessionCluster[]
  ): Promise<InterleavingCandidate[]> {
    console.log('⚛️ Executing atom-level interleaving...');

    const candidates = [...currentTurn.candidates];
    
    // Assign interleaving positions based on configured strategy
    switch (this.config.atom_interleaving.position_weighting) {
      case 'linear':
        this.applyLinearPositionWeighting(candidates);
        break;
      case 'exponential':
        this.applyExponentialPositionWeighting(candidates);
        break;
      case 'balanced':
        this.applyBalancedPositionWeighting(candidates);
        break;
    }

    // Apply diversity enforcement if enabled
    if (this.config.atom_interleaving.diversity_enforcement) {
      this.enforceAtomDiversity(candidates);
    }

    // Calculate attribution weights based on session context
    this.calculateAtomAttributionWeights(candidates, sessionClusters);

    // Sort by interleaving position
    candidates.sort((a, b) => a.interleaving_position - b.interleaving_position);

    // Limit to max atoms per turn
    const interleavedCandidates = candidates.slice(0, this.config.atom_interleaving.max_atoms_per_turn);

    console.log(`⚛️ Atom-level interleaving: ${interleavedCandidates.length} atoms positioned`);
    return interleavedCandidates;
  }

  /**
   * Step 3: Cluster-pair interleaving at session level
   */
  private async executeClusterPairInterleaving(
    sessionClusters: SessionCluster[],
    currentTurnIndex: number
  ): Promise<ClusterInterleavingResult> {
    console.log('🔗 Executing cluster-pair interleaving...');

    // Select top clusters based on coherence and influence
    const rankedClusters = sessionClusters
      .filter(cluster => cluster.cluster_coherence >= this.config.cluster_interleaving.cluster_coherence_threshold)
      .sort((a, b) => b.cluster_coherence - a.cluster_coherence)
      .slice(0, this.config.cluster_interleaving.max_clusters_per_session);

    // Calculate temporal decay factors
    const temporalFactors = new Map<string, number>();
    for (const cluster of rankedClusters) {
      const distanceFromCurrent = Math.abs(currentTurnIndex - cluster.turn_range[1]);
      const decay = Math.pow(this.config.cluster_interleaving.temporal_decay_factor, distanceFromCurrent);
      temporalFactors.set(cluster.cluster_id, decay);
    }

    // Generate cluster pairs for interleaving
    const clusterPairs: ClusterPair[] = [];
    for (let i = 0; i < rankedClusters.length; i++) {
      for (let j = i + 1; j < rankedClusters.length; j++) {
        const clusterA = rankedClusters[i];
        const clusterB = rankedClusters[j];
        
        const crossInfluence = clusterA.cross_cluster_influence.get(clusterB.cluster_id) || 0;
        const temporalA = temporalFactors.get(clusterA.cluster_id) || 0;
        const temporalB = temporalFactors.get(clusterB.cluster_id) || 0;
        
        clusterPairs.push({
          cluster_a_id: clusterA.cluster_id,
          cluster_b_id: clusterB.cluster_id,
          interleaving_strength: crossInfluence * (temporalA + temporalB) / 2,
          temporal_distance: Math.abs(clusterA.turn_range[1] - clusterB.turn_range[1]),
        });
      }
    }

    // Sort pairs by interleaving strength
    clusterPairs.sort((a, b) => b.interleaving_strength - a.interleaving_strength);

    console.log(`🔗 Cluster-pair interleaving: ${clusterPairs.length} pairs evaluated`);

    return {
      selected_clusters: rankedClusters.map(c => c.cluster_id),
      cluster_pairs: clusterPairs,
      temporal_decay_factors: temporalFactors,
      interleaving_quality: this.assessInterleavingQuality(clusterPairs),
    };
  }

  /**
   * Step 4: Session-level attribution for plan/tool outcomes
   */
  private async calculateSessionLevelAttribution(
    sessionId: string,
    currentTurn: Turn,
    previousTurns: Turn[],
    sessionClusters: SessionCluster[]
  ): Promise<AttributionResult> {
    console.log('🎯 Calculating session-level attribution...');

    const attributionWindow = Math.min(this.config.attribution.attribution_window, previousTurns.length + 1);
    const relevantTurns = [...previousTurns.slice(-attributionWindow), currentTurn];

    const attributionScores = new Map<string, number>(); // atom_id -> attribution score

    // Attribute plan outcomes
    for (const turn of relevantTurns) {
      if (turn.plan_outcome) {
        await this.attributePlanOutcome(turn, attributionScores, relevantTurns);
      }
    }

    // Attribute tool results
    for (const turn of relevantTurns) {
      if (turn.tool_results && turn.tool_results.length > 0) {
        await this.attributeToolResults(turn, attributionScores, relevantTurns);
      }
    }

    // Apply temporal discounting
    this.applyTemporalDiscounting(attributionScores, relevantTurns, currentTurn.turn_index);

    // Calculate session-level metrics
    const sessionMetrics = this.calculateSessionMetrics(relevantTurns, attributionScores);

    console.log(`🎯 Attribution calculated for ${attributionScores.size} atoms`);

    return {
      session_id: sessionId,
      attribution_window_turns: attributionWindow,
      attribution_scores: attributionScores,
      session_metrics: sessionMetrics,
      plan_attribution_quality: this.assessPlanAttributionQuality(relevantTurns, attributionScores),
      tool_attribution_quality: this.assessToolAttributionQuality(relevantTurns, attributionScores),
    };
  }

  /**
   * Step 5: Statistical power validation for ΔnDCG@10 improvements
   */
  private async validateStatisticalPower(
    sessionId: string,
    attributionResult: AttributionResult
  ): Promise<StatisticalValidationResult> {
    console.log('📊 Validating statistical power...');

    const { target_delta_ndcg, session_variance_assumption, required_statistical_power } = this.config.statistical;

    // Power calculation: n = (z_α + z_β)² * σ² / Δ²
    const z_alpha = 1.96; // 95% confidence (α = 0.05)
    const z_beta = 0.84;  // 80% power (β = 0.20)
    const delta = target_delta_ndcg;
    const sigma_squared = session_variance_assumption;

    const requiredTurns = Math.ceil(
      Math.pow(z_alpha + z_beta, 2) * sigma_squared / Math.pow(delta, 2)
    );

    // Calculate current statistical power based on available data
    const availableTurns = attributionResult.attribution_window_turns;
    const currentPower = this.calculateCurrentStatisticalPower(
      availableTurns,
      delta,
      sigma_squared
    );

    // Assess significance with Holm correction
    const holmCorrectedSignificance = this.calculateHolmCorrectedSignificance(
      attributionResult.session_metrics
    );

    console.log(`📊 Statistical validation: ${availableTurns}/${requiredTurns} turns, power=${(currentPower * 100).toFixed(1)}%`);

    return {
      target_delta_ndcg: target_delta_ndcg,
      required_turns_for_power: requiredTurns,
      available_turns: availableTurns,
      current_statistical_power: currentPower,
      power_adequate: currentPower >= required_statistical_power,
      holm_corrected_significance: holmCorrectedSignificance,
      early_stopping_recommended: holmCorrectedSignificance < this.config.statistical.early_stopping_significance && currentPower >= required_statistical_power,
      session_variance_observed: this.calculateObservedSessionVariance(attributionResult),
    };
  }

  // Helper methods for clustering and similarity calculations

  private async clusterAtomsBySimilarity(
    atoms: Array<{ atom: InterleavingCandidate; turn_index: number }>
  ): Promise<Map<string, number[]>> {
    // Simplified clustering - in production would use sophisticated similarity measures
    const clusters = new Map<string, number[]>();
    
    // Group by content similarity (simplified)
    for (let i = 0; i < atoms.length; i++) {
      const atom = atoms[i];
      let assignedCluster = false;

      for (const [clusterId, atomIndices] of clusters.entries()) {
        if (atomIndices.length > 0) {
          const representativeAtom = atoms[atomIndices[0]].atom;
          if (this.calculateAtomSimilarity(atom.atom, representativeAtom) > 0.7) {
            atomIndices.push(i);
            assignedCluster = true;
            break;
          }
        }
      }

      if (!assignedCluster) {
        const newClusterId = `cluster_${clusters.size}`;
        clusters.set(newClusterId, [i]);
      }
    }

    return clusters;
  }

  private calculateAtomSimilarity(atomA: InterleavingCandidate, atomB: InterleavingCandidate): number {
    // Simplified similarity calculation
    if (!atomA.text || !atomB.text) return 0;
    
    const wordsA = new Set(atomA.text.toLowerCase().split(/\s+/));
    const wordsB = new Set(atomB.text.toLowerCase().split(/\s+/));
    
    const intersection = new Set([...wordsA].filter(word => wordsB.has(word)));
    const union = new Set([...wordsA, ...wordsB]);
    
    return intersection.size / union.size; // Jaccard similarity
  }

  private calculateClusterCoherence(atoms: InterleavingCandidate[]): number {
    if (atoms.length <= 1) return 1.0;

    let totalSimilarity = 0;
    let comparisons = 0;

    for (let i = 0; i < atoms.length; i++) {
      for (let j = i + 1; j < atoms.length; j++) {
        totalSimilarity += this.calculateAtomSimilarity(atoms[i], atoms[j]);
        comparisons++;
      }
    }

    return comparisons > 0 ? totalSimilarity / comparisons : 0;
  }

  private calculateCrossClusterInfluence(clusterA: SessionCluster, clusterB: SessionCluster): number {
    // Calculate influence based on temporal proximity and content similarity
    const temporalDistance = Math.abs(clusterA.turn_range[1] - clusterB.turn_range[0]);
    const temporalInfluence = Math.exp(-temporalDistance * 0.1);

    let maxSimilarity = 0;
    for (const atomA of clusterA.atoms) {
      for (const atomB of clusterB.atoms) {
        const similarity = this.calculateAtomSimilarity(atomA, atomB);
        maxSimilarity = Math.max(maxSimilarity, similarity);
      }
    }

    return temporalInfluence * maxSimilarity;
  }

  // Position weighting strategies
  private applyLinearPositionWeighting(candidates: InterleavingCandidate[]): void {
    candidates.forEach((candidate, index) => {
      candidate.interleaving_position = index;
      candidate.attribution_weight = 1.0 - (index / candidates.length) * 0.5;
    });
  }

  private applyExponentialPositionWeighting(candidates: InterleavingCandidate[]): void {
    candidates.forEach((candidate, index) => {
      candidate.interleaving_position = index;
      candidate.attribution_weight = Math.exp(-index * 0.2);
    });
  }

  private applyBalancedPositionWeighting(candidates: InterleavingCandidate[]): void {
    candidates.forEach((candidate, index) => {
      candidate.interleaving_position = index;
      // Balanced: higher weight for both top and diverse positions
      const topWeight = Math.exp(-index * 0.1);
      const diversityWeight = index % 3 === 0 ? 1.2 : 1.0; // Boost every 3rd position
      candidate.attribution_weight = topWeight * diversityWeight;
    });
  }

  private enforceAtomDiversity(candidates: InterleavingCandidate[]): void {
    // Reorder candidates to enforce diversity while maintaining quality
    const diversifiedOrder: InterleavingCandidate[] = [];
    const remaining = [...candidates];

    while (remaining.length > 0 && diversifiedOrder.length < this.config.atom_interleaving.max_atoms_per_turn) {
      // Pick the best remaining candidate
      const best = remaining.shift()!;
      diversifiedOrder.push(best);

      // Remove similar candidates from consideration for next few positions
      const similarityThreshold = 0.8;
      for (let i = remaining.length - 1; i >= 0; i--) {
        if (this.calculateAtomSimilarity(best, remaining[i]) > similarityThreshold) {
          // Move similar candidates to later positions instead of removing
          const similar = remaining.splice(i, 1)[0];
          remaining.push(similar);
        }
      }
    }

    // Update positions
    diversifiedOrder.forEach((candidate, index) => {
      candidate.interleaving_position = index;
    });
  }

  private calculateAtomAttributionWeights(
    candidates: InterleavingCandidate[],
    sessionClusters: SessionCluster[]
  ): void {
    for (const candidate of candidates) {
      // Find cluster membership
      const memberCluster = sessionClusters.find(cluster =>
        cluster.atoms.some(atom => atom.atom_id === candidate.atom_id)
      );

      if (memberCluster) {
        // Boost based on cluster coherence and cross-cluster influence
        const clusterBoost = memberCluster.cluster_coherence;
        const influenceBoost = Array.from(memberCluster.cross_cluster_influence.values())
          .reduce((sum, influence) => sum + influence, 0) / memberCluster.cross_cluster_influence.size || 0;

        candidate.attribution_weight *= (1 + clusterBoost * 0.2 + influenceBoost * 0.1);
        candidate.cluster_id = memberCluster.cluster_id;
      }
    }
  }

  // Attribution calculation methods
  private async attributePlanOutcome(
    turn: Turn,
    attributionScores: Map<string, number>,
    relevantTurns: Turn[]
  ): Promise<void> {
    if (!turn.plan_outcome) return;

    const planWeight = this.config.attribution.plan_outcome_weight;
    const planSuccess = turn.plan_outcome.success_metric;

    for (const atomId of turn.selected_atoms) {
      const currentScore = attributionScores.get(atomId) || 0;
      attributionScores.set(atomId, currentScore + planSuccess * planWeight);
    }

    // Handle multi-turn plan dependencies
    if (turn.plan_outcome.attribution_scope === 'multi_turn') {
      for (const dependentTurnId of turn.plan_outcome.dependent_turns) {
        const dependentTurn = relevantTurns.find(t => t.turn_id === dependentTurnId);
        if (dependentTurn) {
          for (const atomId of dependentTurn.selected_atoms) {
            const currentScore = attributionScores.get(atomId) || 0;
            attributionScores.set(atomId, currentScore + planSuccess * planWeight * 0.5); // Reduced weight for dependencies
          }
        }
      }
    }
  }

  private async attributeToolResults(
    turn: Turn,
    attributionScores: Map<string, number>,
    relevantTurns: Turn[]
  ): Promise<void> {
    if (!turn.tool_results) return;

    const toolWeight = this.config.attribution.tool_outcome_weight;

    for (const toolResult of turn.tool_results) {
      const toolContribution = toolResult.execution_success ? toolResult.contribution_to_session : -0.1;

      // Attribute to causal dependencies
      for (const atomId of toolResult.causal_dependencies) {
        const currentScore = attributionScores.get(atomId) || 0;
        attributionScores.set(atomId, currentScore + toolContribution * toolWeight);
      }
    }
  }

  private applyTemporalDiscounting(
    attributionScores: Map<string, number>,
    relevantTurns: Turn[],
    currentTurnIndex: number
  ): void {
    const discountFactor = this.config.attribution.temporal_discount_factor;

    for (const turn of relevantTurns) {
      const turnDistance = Math.abs(currentTurnIndex - turn.turn_index);
      const discount = Math.pow(discountFactor, turnDistance);

      for (const atomId of turn.selected_atoms) {
        const currentScore = attributionScores.get(atomId) || 0;
        attributionScores.set(atomId, currentScore * discount);
      }
    }
  }

  // Assessment and metrics calculation methods
  private calculateSessionMetrics(
    relevantTurns: Turn[],
    attributionScores: Map<string, number>
  ): SessionMetrics {
    const totalTurns = relevantTurns.length;
    const totalAtoms = Array.from(attributionScores.keys()).length;
    
    const attributionValues = Array.from(attributionScores.values());
    const meanAttribution = attributionValues.reduce((sum, val) => sum + val, 0) / attributionValues.length || 0;
    
    const attributionVariance = attributionValues.reduce((sum, val) => 
      sum + Math.pow(val - meanAttribution, 2), 0) / attributionValues.length || 0;

    return {
      total_turns: totalTurns,
      total_atoms: totalAtoms,
      mean_attribution_score: meanAttribution,
      attribution_variance: attributionVariance,
      attribution_distribution: this.calculateAttributionDistribution(attributionValues),
    };
  }

  private calculateAttributionDistribution(values: number[]): number[] {
    // Calculate percentiles: [p10, p25, p50, p75, p90]
    if (values.length === 0) return [0, 0, 0, 0, 0];

    const sorted = [...values].sort((a, b) => a - b);
    const percentiles = [0.1, 0.25, 0.5, 0.75, 0.9];
    
    return percentiles.map(p => {
      const index = Math.floor(p * (sorted.length - 1));
      return sorted[index];
    });
  }

  private assessInterleavingQuality(clusterPairs: ClusterPair[]): number {
    if (clusterPairs.length === 0) return 0;

    const avgInterleavingStrength = clusterPairs.reduce((sum, pair) => 
      sum + pair.interleaving_strength, 0) / clusterPairs.length;

    const temporalDiversityScore = this.calculateTemporalDiversity(clusterPairs);

    return (avgInterleavingStrength + temporalDiversityScore) / 2;
  }

  private calculateTemporalDiversity(clusterPairs: ClusterPair[]): number {
    if (clusterPairs.length <= 1) return 1;

    const distances = clusterPairs.map(pair => pair.temporal_distance);
    const avgDistance = distances.reduce((sum, d) => sum + d, 0) / distances.length;
    const maxDistance = Math.max(...distances);

    return maxDistance > 0 ? avgDistance / maxDistance : 0;
  }

  private assessPlanAttributionQuality(
    turns: Turn[],
    attributionScores: Map<string, number>
  ): number {
    const turnsWithPlans = turns.filter(t => t.plan_outcome);
    if (turnsWithPlans.length === 0) return 1;

    let totalQuality = 0;
    for (const turn of turnsWithPlans) {
      const planSuccess = turn.plan_outcome!.success_metric;
      const avgAtomAttribution = turn.selected_atoms.reduce((sum, atomId) => 
        sum + (attributionScores.get(atomId) || 0), 0) / turn.selected_atoms.length || 0;
      
      // Quality is correlation between plan success and atom attribution
      totalQuality += Math.abs(planSuccess - avgAtomAttribution) < 0.1 ? 1 : 0.5;
    }

    return totalQuality / turnsWithPlans.length;
  }

  private assessToolAttributionQuality(
    turns: Turn[],
    attributionScores: Map<string, number>
  ): number {
    const turnsWithTools = turns.filter(t => t.tool_results && t.tool_results.length > 0);
    if (turnsWithTools.length === 0) return 1;

    let totalQuality = 0;
    let totalTools = 0;

    for (const turn of turnsWithTools) {
      for (const toolResult of turn.tool_results!) {
        const avgCausalAttribution = toolResult.causal_dependencies.reduce((sum, atomId) => 
          sum + (attributionScores.get(atomId) || 0), 0) / toolResult.causal_dependencies.length || 0;
        
        const expectedAttribution = toolResult.execution_success ? toolResult.contribution_to_session : 0;
        totalQuality += Math.abs(expectedAttribution - avgCausalAttribution) < 0.2 ? 1 : 0.5;
        totalTools++;
      }
    }

    return totalTools > 0 ? totalQuality / totalTools : 1;
  }

  private calculateCurrentStatisticalPower(
    availableTurns: number,
    delta: number,
    sigmaSquared: number
  ): number {
    // Power = Φ(√(n * Δ² / σ²) - z_α)
    const z_alpha = 1.96; // 95% confidence
    const effectSize = Math.sqrt(availableTurns * Math.pow(delta, 2) / sigmaSquared);
    
    // Approximation of standard normal CDF
    return this.standardNormalCDF(effectSize - z_alpha);
  }

  private calculateHolmCorrectedSignificance(metrics: SessionMetrics): number {
    // Simplified Holm correction - would be more sophisticated in production
    const baseSignificance = 0.05;
    const numberOfComparisons = 3; // Approximate number of statistical tests
    
    return baseSignificance / numberOfComparisons;
  }

  private calculateObservedSessionVariance(attributionResult: AttributionResult): number {
    return attributionResult.session_metrics.attribution_variance;
  }

  private standardNormalCDF(x: number): number {
    // Approximation of standard normal CDF using error function
    return 0.5 * (1 + Math.sign(x) * Math.sqrt(1 - Math.exp(-2 * x * x / Math.PI)));
  }

  private updateSessionState(sessionId: string, result: HierarchicalInterleavingResult): void {
    const state: SessionState = {
      session_id: sessionId,
      last_turn_id: result.turn_id,
      active_clusters: result.session_clusters.map(c => c.cluster_id),
      attribution_history: result.attribution.attribution_scores,
      statistical_power: result.statistical_validation.current_statistical_power,
    };

    this.sessionState.set(sessionId, state);
  }

  /**
   * Get current session state for monitoring
   */
  getSessionState(sessionId: string): SessionState | undefined {
    return this.sessionState.get(sessionId);
  }

  /**
   * Validate interleaving performance against targets
   */
  async validateInterleavingPerformance(sessionId: string): Promise<InterleavingPerformanceResult> {
    const state = this.sessionState.get(sessionId);
    if (!state) {
      throw new Error(`No session state found for ${sessionId}`);
    }

    const statisticalPowerAdequate = state.statistical_power >= this.config.statistical.required_statistical_power;
    const clusterDiversityScore = state.active_clusters.length / this.config.cluster_interleaving.max_clusters_per_session;

    return {
      session_id: sessionId,
      statistical_power_adequate: statisticalPowerAdequate,
      cluster_diversity_score: clusterDiversityScore,
      attribution_quality: this.assessAttributionQuality(state.attribution_history),
      performance_meets_targets: statisticalPowerAdequate && clusterDiversityScore >= 0.6,
      recommendations: this.generatePerformanceRecommendations(statisticalPowerAdequate, clusterDiversityScore),
    };
  }

  private assessAttributionQuality(attributionHistory: Map<string, number>): number {
    const scores = Array.from(attributionHistory.values());
    if (scores.length === 0) return 0;

    const mean = scores.reduce((sum, score) => sum + score, 0) / scores.length;
    const variance = scores.reduce((sum, score) => sum + Math.pow(score - mean, 2), 0) / scores.length;
    
    // Quality is inversely related to variance (more consistent attribution is better)
    return variance > 0 ? Math.exp(-variance) : 1;
  }

  private generatePerformanceRecommendations(
    powerAdequate: boolean,
    diversityScore: number
  ): string[] {
    const recommendations: string[] = [];

    if (!powerAdequate) {
      recommendations.push('Increase session length for statistical significance');
      recommendations.push('Consider early stopping with Holm correction if trends are clear');
    }

    if (diversityScore < 0.6) {
      recommendations.push('Increase cluster diversity in interleaving');
      recommendations.push('Tune cluster coherence threshold');
    }

    if (powerAdequate && diversityScore >= 0.6) {
      recommendations.push('Interleaving performance meets targets');
      recommendations.push('Continue current configuration');
    }

    return recommendations;
  }
}

// Supporting interfaces and types
interface SessionState {
  session_id: string;
  last_turn_id: string;
  active_clusters: string[];
  attribution_history: Map<string, number>;
  statistical_power: number;
}

interface ClusterPair {
  cluster_a_id: string;
  cluster_b_id: string;
  interleaving_strength: number;
  temporal_distance: number;
}

interface ClusterInterleavingResult {
  selected_clusters: string[];
  cluster_pairs: ClusterPair[];
  temporal_decay_factors: Map<string, number>;
  interleaving_quality: number;
}

interface AttributionResult {
  session_id: string;
  attribution_window_turns: number;
  attribution_scores: Map<string, number>;
  session_metrics: SessionMetrics;
  plan_attribution_quality: number;
  tool_attribution_quality: number;
}

interface SessionMetrics {
  total_turns: number;
  total_atoms: number;
  mean_attribution_score: number;
  attribution_variance: number;
  attribution_distribution: number[]; // [p10, p25, p50, p75, p90]
}

interface StatisticalValidationResult {
  target_delta_ndcg: number;
  required_turns_for_power: number;
  available_turns: number;
  current_statistical_power: number;
  power_adequate: boolean;
  holm_corrected_significance: number;
  early_stopping_recommended: boolean;
  session_variance_observed: number;
}

export interface HierarchicalInterleavingResult {
  session_id: string;
  turn_id: string;
  interleaved_candidates: InterleavingCandidate[];
  session_clusters: SessionCluster[];
  cluster_interleaving: ClusterInterleavingResult;
  attribution: AttributionResult;
  statistical_validation: StatisticalValidationResult;
  processing_metrics: {
    total_processing_time_ms: number;
    atoms_processed: number;
    clusters_formed: number;
    attribution_calculations: number;
  };
}

export interface InterleavingPerformanceResult {
  session_id: string;
  statistical_power_adequate: boolean;
  cluster_diversity_score: number;
  attribution_quality: number;
  performance_meets_targets: boolean;
  recommendations: string[];
}

/**
 * Utility function to create and execute hierarchical interleaving
 */
export async function executeHierarchicalInterleaving(
  db: DB,
  sessionId: string,
  currentTurn: Turn,
  previousTurns: Turn[],
  config?: Partial<HierarchicalInterleavingConfig>
): Promise<HierarchicalInterleavingResult> {
  const engine = new HierarchicalInterleavingEngine(db, config);
  return await engine.executeHierarchicalInterleaving(sessionId, currentTurn, previousTurns);
}