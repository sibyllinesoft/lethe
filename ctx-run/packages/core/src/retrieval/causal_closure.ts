/**
 * Causal-Closure Groups for Atom Bundling
 * 
 * Treats atoms as groups with precomputed transitive closures:
 * - For any candidate a, precompute transitive closure C(a) (ancestors + required siblings)
 * - Convert laminar constraints to single knapsack over groups
 * - Group-aware greedy selection with joint tokens c(C) and joint gain g(C)
 * 
 * This collapses many infeasibilities and keeps ILP usage in single digits.
 */

import { z } from 'zod';

// Causal dependency types
export enum DependencyType {
  ANCESTOR = 'ancestor',          // Must include parent before child
  SIBLING = 'sibling',           // Must include all siblings together
  IMPLICATION = 'implication',   // If A then B must be included
  EXCLUSION = 'exclusion',       // A and B cannot both be included
  CONDITIONAL = 'conditional',   // A requires B only if C is included
}

// Causal dependency relationship
export interface CausalDependency {
  type: DependencyType;
  source_id: string;
  target_id: string;
  condition_id?: string; // For conditional dependencies
  weight: number;        // Strength of dependency [0,1]
}

// Atom with causal relationships
export interface CausalAtom {
  id: string;
  tokens: number;
  importance: number;
  chunk_type: string;
  text?: string;
  
  // Causal relationships
  dependencies: CausalDependency[];
  parent_id?: string;
  children_ids: string[];
  sibling_group?: string;
  
  // Precomputed closure
  transitive_closure?: string[];
  closure_computed: boolean;
}

// Causal group (bundled atoms)
export interface CausalGroup {
  id: string;
  atom_ids: string[];
  total_tokens: number;
  avg_importance: number;
  group_gain: number;
  
  // Group properties
  is_closure_complete: boolean;
  has_required_ancestors: boolean;
  sibling_groups: string[];
  
  // Constraint satisfaction
  violates_exclusions: boolean;
  satisfies_conditions: boolean;
  
  // Performance metrics
  tokens_per_importance: number;
  group_cohesion: number;
}

// Configuration for causal closure
export const CausalClosureConfigSchema = z.object({
  // Closure computation
  max_closure_size: z.number().int().min(1).default(50),
  max_transitive_depth: z.number().int().min(1).default(10),
  enable_sibling_bundling: z.boolean().default(true),
  
  // Group optimization  
  min_group_size: z.number().int().min(1).default(1),
  max_group_size: z.number().int().min(1).default(20),
  group_cohesion_weight: z.number().min(0).default(0.1),
  
  // Constraint handling
  soft_constraint_penalty: z.number().min(0).default(0.5),
  hard_constraint_enforcement: z.boolean().default(true),
  exclusion_tolerance: z.number().min(0).max(1).default(0.0),
  
  // Performance settings
  enable_parallel_closure: z.boolean().default(true),
  cache_closure_results: z.boolean().default(true),
  closure_cache_size: z.number().int().min(10).default(1000),
});

export type CausalClosureConfig = z.infer<typeof CausalClosureConfigSchema>;

// Result of causal closure computation
export interface CausalClosureResult {
  groups: CausalGroup[];
  total_atoms: number;
  total_groups: number;
  average_group_size: number;
  constraint_violations: number;
  ilp_candidates: CausalGroup[]; // Groups requiring ILP resolution
  processing_time_ms: number;
}

/**
 * Causal Closure Engine for Atom Grouping
 * 
 * Precomputes transitive closures and creates bundled groups to convert
 * laminar constraints into a single knapsack optimization problem.
 */
export class CausalClosureEngine {
  private config: CausalClosureConfig;
  private closure_cache: Map<string, string[]>;
  private dependency_graph: Map<string, CausalAtom>;
  
  constructor(config: Partial<CausalClosureConfig> = {}) {
    this.config = CausalClosureConfigSchema.parse(config);
    this.closure_cache = new Map();
    this.dependency_graph = new Map();
  }
  
  /**
   * Compute causal closure groups for atoms
   */
  async computeCausalGroups(atoms: CausalAtom[]): Promise<CausalClosureResult> {
    const startTime = performance.now();
    
    // Build dependency graph
    this.buildDependencyGraph(atoms);
    
    // Compute transitive closures for all atoms
    if (this.config.enable_parallel_closure) {
      await this.computeClosuresParallel(atoms);
    } else {
      await this.computeClosuresSequential(atoms);
    }
    
    // Create causal groups from closures
    const groups = this.createCausalGroups(atoms);
    
    // Identify ILP candidates (complex constraint groups)
    const ilp_candidates = this.identifyILPCandidates(groups);
    
    const processingTime = performance.now() - startTime;
    
    return {
      groups,
      total_atoms: atoms.length,
      total_groups: groups.length,
      average_group_size: groups.reduce((sum, g) => sum + g.atom_ids.length, 0) / groups.length,
      constraint_violations: this.countConstraintViolations(groups),
      ilp_candidates,
      processing_time_ms: processingTime,
    };
  }
  
  /**
   * Build dependency graph from atoms
   */
  private buildDependencyGraph(atoms: CausalAtom[]): void {
    this.dependency_graph.clear();
    
    for (const atom of atoms) {
      this.dependency_graph.set(atom.id, atom);
    }
    
    // Build reverse dependencies for efficient traversal
    for (const atom of atoms) {
      for (const dep of atom.dependencies) {
        const target = this.dependency_graph.get(dep.target_id);
        if (target && !target.dependencies.some(d => d.source_id === atom.id)) {
          // Add reverse dependency
          const reverse_dep: CausalDependency = {
            type: this.reverseDepType(dep.type),
            source_id: dep.target_id,
            target_id: dep.source_id,
            weight: dep.weight,
          };
          target.dependencies.push(reverse_dep);
        }
      }
    }
  }
  
  /**
   * Get reverse dependency type
   */
  private reverseDepType(type: DependencyType): DependencyType {
    switch (type) {
      case DependencyType.ANCESTOR:
        return DependencyType.ANCESTOR; // Bidirectional
      case DependencyType.SIBLING:
        return DependencyType.SIBLING; // Symmetric
      case DependencyType.IMPLICATION:
        return DependencyType.ANCESTOR; // Reverse is ancestor
      default:
        return type;
    }
  }
  
  /**
   * Compute transitive closures in parallel
   */
  private async computeClosuresParallel(atoms: CausalAtom[]): Promise<void> {
    const batch_size = Math.min(10, atoms.length);
    const batches: CausalAtom[][] = [];
    
    for (let i = 0; i < atoms.length; i += batch_size) {
      batches.push(atoms.slice(i, i + batch_size));
    }
    
    await Promise.all(
      batches.map(async batch => {
        for (const atom of batch) {
          await this.computeTransitiveClosure(atom);
        }
      })
    );
  }
  
  /**
   * Compute transitive closures sequentially
   */
  private async computeClosuresSequential(atoms: CausalAtom[]): Promise<void> {
    for (const atom of atoms) {
      await this.computeTransitiveClosure(atom);
    }
  }
  
  /**
   * Compute transitive closure C(a) for atom a
   */
  private async computeTransitiveClosure(atom: CausalAtom): Promise<string[]> {
    // Check cache first
    if (this.config.cache_closure_results && this.closure_cache.has(atom.id)) {
      atom.transitive_closure = this.closure_cache.get(atom.id);
      atom.closure_computed = true;
      return atom.transitive_closure!;
    }
    
    const closure = new Set<string>();
    const visited = new Set<string>();
    const stack = [atom.id];
    let depth = 0;
    
    while (stack.length > 0 && depth < this.config.max_transitive_depth) {
      const current_id = stack.pop()!;
      
      if (visited.has(current_id) || closure.size >= this.config.max_closure_size) {
        continue;
      }
      
      visited.add(current_id);
      closure.add(current_id);
      
      const current_atom = this.dependency_graph.get(current_id);
      if (!current_atom) continue;
      
      // Add dependencies based on type
      for (const dep of current_atom.dependencies) {
        if (this.shouldIncludeInClosure(dep)) {
          stack.push(dep.target_id);
        }
      }
      
      // Add parent (ancestor relationship)
      if (current_atom.parent_id) {
        stack.push(current_atom.parent_id);
      }
      
      // Add siblings if enabled
      if (this.config.enable_sibling_bundling && current_atom.sibling_group) {
        const siblings = this.getSiblingsInGroup(current_atom.sibling_group);
        for (const sibling_id of siblings) {
          if (sibling_id !== current_id) {
            stack.push(sibling_id);
          }
        }
      }
      
      depth++;
    }
    
    const closure_array = Array.from(closure);
    
    // Cache result
    if (this.config.cache_closure_results) {
      if (this.closure_cache.size >= this.config.closure_cache_size) {
        // Simple LRU: remove first entry
        const first_key = this.closure_cache.keys().next().value;
        this.closure_cache.delete(first_key);
      }
      this.closure_cache.set(atom.id, closure_array);
    }
    
    atom.transitive_closure = closure_array;
    atom.closure_computed = true;
    
    return closure_array;
  }
  
  /**
   * Check if dependency should be included in closure
   */
  private shouldIncludeInClosure(dep: CausalDependency): boolean {
    switch (dep.type) {
      case DependencyType.ANCESTOR:
      case DependencyType.SIBLING:
      case DependencyType.IMPLICATION:
        return dep.weight >= 0.5; // Only strong dependencies
      case DependencyType.EXCLUSION:
        return false; // Exclusions handled separately
      case DependencyType.CONDITIONAL:
        return dep.weight >= 0.7; // Higher threshold for conditionals
      default:
        return false;
    }
  }
  
  /**
   * Get siblings in the same group
   */
  private getSiblingsInGroup(group_id: string): string[] {
    const siblings: string[] = [];
    
    for (const [atom_id, atom] of this.dependency_graph) {
      if (atom.sibling_group === group_id) {
        siblings.push(atom_id);
      }
    }
    
    return siblings;
  }
  
  /**
   * Create causal groups from computed closures
   */
  private createCausalGroups(atoms: CausalAtom[]): CausalGroup[] {
    const groups: CausalGroup[] = [];
    const atom_to_group = new Map<string, string>();
    const processed_closures = new Set<string>();
    
    for (const atom of atoms) {
      if (!atom.transitive_closure || atom_to_group.has(atom.id)) {
        continue;
      }
      
      // Create closure signature for deduplication
      const closure_signature = atom.transitive_closure
        .slice()
        .sort()
        .join('|');
      
      if (processed_closures.has(closure_signature)) {
        continue;
      }
      
      processed_closures.add(closure_signature);
      
      // Create group from closure
      const group = this.createGroupFromClosure(atom.transitive_closure, atoms);
      
      if (group.atom_ids.length >= this.config.min_group_size &&
          group.atom_ids.length <= this.config.max_group_size) {
        groups.push(group);
        
        // Mark atoms as assigned to group
        for (const atom_id of group.atom_ids) {
          atom_to_group.set(atom_id, group.id);
        }
      }
    }
    
    // Handle remaining atoms as singleton groups
    for (const atom of atoms) {
      if (!atom_to_group.has(atom.id)) {
        const singleton_group = this.createSingletonGroup(atom);
        groups.push(singleton_group);
        atom_to_group.set(atom.id, singleton_group.id);
      }
    }
    
    return groups;
  }
  
  /**
   * Create causal group from transitive closure
   */
  private createGroupFromClosure(closure: string[], atoms: CausalAtom[]): CausalGroup {
    const group_atoms = atoms.filter(a => closure.includes(a.id));
    const atom_ids = group_atoms.map(a => a.id);
    
    const total_tokens = group_atoms.reduce((sum, a) => sum + a.tokens, 0);
    const avg_importance = group_atoms.reduce((sum, a) => sum + a.importance, 0) / group_atoms.length;
    
    // Compute group gain with cohesion bonus
    let group_gain = group_atoms.reduce((sum, a) => sum + a.importance, 0);
    const group_cohesion = this.computeGroupCohesion(group_atoms);
    group_gain += this.config.group_cohesion_weight * group_cohesion;
    
    // Check constraints
    const violates_exclusions = this.checkExclusionViolations(group_atoms);
    const satisfies_conditions = this.checkConditionalConstraints(group_atoms);
    
    return {
      id: `group_${closure[0]}_${closure.length}`,
      atom_ids,
      total_tokens,
      avg_importance,
      group_gain,
      is_closure_complete: true,
      has_required_ancestors: this.checkAncestorRequirements(group_atoms),
      sibling_groups: this.identifySiblingGroups(group_atoms),
      violates_exclusions,
      satisfies_conditions,
      tokens_per_importance: total_tokens / (avg_importance || 1),
      group_cohesion,
    };
  }
  
  /**
   * Create singleton group for isolated atom
   */
  private createSingletonGroup(atom: CausalAtom): CausalGroup {
    return {
      id: `singleton_${atom.id}`,
      atom_ids: [atom.id],
      total_tokens: atom.tokens,
      avg_importance: atom.importance,
      group_gain: atom.importance,
      is_closure_complete: true,
      has_required_ancestors: true, // Singleton trivially satisfies
      sibling_groups: [],
      violates_exclusions: false,
      satisfies_conditions: true,
      tokens_per_importance: atom.tokens / (atom.importance || 1),
      group_cohesion: 1.0,
    };
  }
  
  /**
   * Compute group cohesion metric
   */
  private computeGroupCohesion(atoms: CausalAtom[]): number {
    if (atoms.length <= 1) return 1.0;
    
    let total_connections = 0;
    let actual_connections = 0;
    
    for (let i = 0; i < atoms.length; i++) {
      for (let j = i + 1; j < atoms.length; j++) {
        total_connections++;
        
        // Check if atoms are connected
        if (this.areAtomsConnected(atoms[i], atoms[j])) {
          actual_connections++;
        }
      }
    }
    
    return total_connections > 0 ? actual_connections / total_connections : 1.0;
  }
  
  /**
   * Check if two atoms are directly connected
   */
  private areAtomsConnected(atom1: CausalAtom, atom2: CausalAtom): boolean {
    return atom1.dependencies.some(d => d.target_id === atom2.id) ||
           atom2.dependencies.some(d => d.target_id === atom1.id) ||
           atom1.sibling_group === atom2.sibling_group;
  }
  
  /**
   * Check exclusion constraint violations
   */
  private checkExclusionViolations(atoms: CausalAtom[]): boolean {
    for (const atom of atoms) {
      for (const dep of atom.dependencies) {
        if (dep.type === DependencyType.EXCLUSION) {
          const excluded_atom = atoms.find(a => a.id === dep.target_id);
          if (excluded_atom) {
            return true; // Exclusion violation
          }
        }
      }
    }
    return false;
  }
  
  /**
   * Check conditional constraint satisfaction
   */
  private checkConditionalConstraints(atoms: CausalAtom[]): boolean {
    for (const atom of atoms) {
      for (const dep of atom.dependencies) {
        if (dep.type === DependencyType.CONDITIONAL && dep.condition_id) {
          const condition_atom = atoms.find(a => a.id === dep.condition_id);
          const target_atom = atoms.find(a => a.id === dep.target_id);
          
          if (condition_atom && !target_atom) {
            return false; // Conditional not satisfied
          }
        }
      }
    }
    return true;
  }
  
  /**
   * Check ancestor requirements
   */
  private checkAncestorRequirements(atoms: CausalAtom[]): boolean {
    const atom_ids = new Set(atoms.map(a => a.id));
    
    for (const atom of atoms) {
      if (atom.parent_id && !atom_ids.has(atom.parent_id)) {
        return false; // Missing ancestor
      }
    }
    return true;
  }
  
  /**
   * Identify sibling groups
   */
  private identifySiblingGroups(atoms: CausalAtom[]): string[] {
    const sibling_groups = new Set<string>();
    
    for (const atom of atoms) {
      if (atom.sibling_group) {
        sibling_groups.add(atom.sibling_group);
      }
    }
    
    return Array.from(sibling_groups);
  }
  
  /**
   * Identify groups requiring ILP resolution
   */
  private identifyILPCandidates(groups: CausalGroup[]): CausalGroup[] {
    return groups.filter(group => 
      group.violates_exclusions ||
      !group.satisfies_conditions ||
      group.sibling_groups.length > 3 ||
      group.atom_ids.length > this.config.max_group_size * 0.8
    );
  }
  
  /**
   * Count constraint violations across all groups
   */
  private countConstraintViolations(groups: CausalGroup[]): number {
    return groups.reduce((count, group) => {
      let violations = 0;
      if (group.violates_exclusions) violations++;
      if (!group.satisfies_conditions) violations++;
      if (!group.has_required_ancestors) violations++;
      return count + violations;
    }, 0);
  }
  
  /**
   * Get closure for specific atom (with caching)
   */
  async getClosure(atom_id: string): Promise<string[]> {
    const atom = this.dependency_graph.get(atom_id);
    if (!atom) return [];
    
    if (!atom.closure_computed) {
      await this.computeTransitiveClosure(atom);
    }
    
    return atom.transitive_closure || [];
  }
  
  /**
   * Clear caches and reset state
   */
  reset(): void {
    this.closure_cache.clear();
    this.dependency_graph.clear();
  }
  
  /**
   * Get cache statistics
   */
  getCacheStats(): { size: number; hit_rate: number; max_size: number } {
    return {
      size: this.closure_cache.size,
      hit_rate: 0, // Would need to track hits/misses
      max_size: this.config.closure_cache_size,
    };
  }
}

/**
 * Convenience function for causal group computation
 */
export async function computeCausalGroups(
  atoms: CausalAtom[],
  config: Partial<CausalClosureConfig> = {}
): Promise<CausalClosureResult> {
  const engine = new CausalClosureEngine(config);
  return engine.computeCausalGroups(atoms);
}

/**
 * Create causal dependencies from simple relationships
 */
export function createCausalDependencies(
  parent_child_pairs: Array<[string, string]>,
  sibling_groups: Array<string[]>,
  implications: Array<[string, string]>,
  exclusions: Array<[string, string]>
): CausalDependency[] {
  const dependencies: CausalDependency[] = [];
  
  // Parent-child relationships
  for (const [parent, child] of parent_child_pairs) {
    dependencies.push({
      type: DependencyType.ANCESTOR,
      source_id: child,
      target_id: parent,
      weight: 1.0,
    });
  }
  
  // Sibling groups
  for (const siblings of sibling_groups) {
    for (let i = 0; i < siblings.length; i++) {
      for (let j = i + 1; j < siblings.length; j++) {
        dependencies.push({
          type: DependencyType.SIBLING,
          source_id: siblings[i],
          target_id: siblings[j],
          weight: 0.9,
        });
      }
    }
  }
  
  // Implications
  for (const [source, target] of implications) {
    dependencies.push({
      type: DependencyType.IMPLICATION,
      source_id: source,
      target_id: target,
      weight: 0.8,
    });
  }
  
  // Exclusions
  for (const [source, target] of exclusions) {
    dependencies.push({
      type: DependencyType.EXCLUSION,
      source_id: source,
      target_id: target,
      weight: 1.0,
    });
  }
  
  return dependencies;
}