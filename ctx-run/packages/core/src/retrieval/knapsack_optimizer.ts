/**
 * Global Token-Budget Knapsack Optimizer with Bookend Packing
 * 
 * Implements 0/1 knapsack optimization with group constraints for token budget allocation.
 * Supports bookend packing (head/tail anchors with zig-zag placement) for context preservation.
 * 
 * Key features:
 * - Group constraints (keep all sentences from a group or none)
 * - Head/tail anchor preservation with priority scoring
 * - Zig-zag placement between anchors for optimal context flow
 * - Token budget enforcement with safety margins
 * - Importance-aware selection with diminishing returns modeling
 */

import { z } from 'zod';

// Input validation schemas
export const KnapsackItemSchema = z.object({
  id: z.string(),
  tokens: z.number().int().positive(),
  importance: z.number().min(0).max(1),
  group_id: z.string().optional(),
  chunk_id: z.string(),
  sentence_index: z.number().int().nonnegative(),
  is_head_anchor: z.boolean().default(false),
  is_tail_anchor: z.boolean().default(false),
  co_entailing_group: z.array(z.string()).optional(),
  text: z.string()
});

export const KnapsackConfigSchema = z.object({
  max_tokens: z.number().int().positive(),
  safety_margin: z.number().min(0).max(0.2).default(0.05), // 5% safety margin
  head_anchor_weight: z.number().min(1).default(2.0),
  tail_anchor_weight: z.number().min(1).default(1.5),
  group_bonus: z.number().min(0).default(0.1),
  diminishing_returns_factor: z.number().min(0).max(1).default(0.8),
  zigzag_placement: z.boolean().default(true),
  preserve_chunk_order: z.boolean().default(true)
});

export const PackedResultSchema = z.object({
  selected_items: z.array(KnapsackItemSchema),
  total_tokens: z.number().int().nonnegative(),
  total_importance: z.number().min(0),
  utilization_ratio: z.number().min(0).max(1),
  head_anchors: z.array(z.string()),
  tail_anchors: z.array(z.string()),
  placement_order: z.array(z.string()),
  groups_selected: z.array(z.string()),
  algorithm_used: z.enum(['exact_dp', 'greedy_approx', 'bookend_priority']),
  computation_time_ms: z.number().nonnegative()
});

export type KnapsackItem = z.infer<typeof KnapsackItemSchema>;
export type KnapsackConfig = z.infer<typeof KnapsackConfigSchema>;
export type PackedResult = z.infer<typeof PackedResultSchema>;

interface GroupInfo {
  id: string;
  items: KnapsackItem[];
  total_tokens: number;
  avg_importance: number;
  has_anchors: boolean;
}

/**
 * Global knapsack optimizer with group constraints and bookend packing
 */
export class KnapsackOptimizer {
  private config: KnapsackConfig;

  constructor(config: Partial<KnapsackConfig> = {}) {
    this.config = KnapsackConfigSchema.parse(config);
  }

  /**
   * Main optimization entry point
   */
  async knapsackPack(items: KnapsackItem[]): Promise<PackedResult> {
    const startTime = performance.now();

    // Validate inputs
    const validatedItems = items.map(item => KnapsackItemSchema.parse(item));
    
    // Calculate effective budget with safety margin
    const effectiveTokens = Math.floor(this.config.max_tokens * (1 - this.config.safety_margin));
    
    // Group items and analyze structure
    const groups = this.analyzeGroups(validatedItems);
    const anchors = this.identifyAnchors(validatedItems);
    
    // Choose optimization strategy based on problem size and constraints
    let result: PackedResult;
    
    if (groups.size <= 20 && validatedItems.length <= 100) {
      // Use exact dynamic programming for smaller problems
      result = this.exactKnapsackDP(validatedItems, groups, effectiveTokens);
    } else if (anchors.head.length > 0 || anchors.tail.length > 0) {
      // Use bookend-priority strategy for large problems with anchors
      result = this.bookendPriorityPacking(validatedItems, groups, anchors, effectiveTokens);
    } else {
      // Use greedy approximation for large problems without anchors
      result = this.greedyApproximation(validatedItems, groups, effectiveTokens);
    }

    // Apply bookend linearization if enabled
    if (this.config.zigzag_placement && (anchors.head.length > 0 || anchors.tail.length > 0)) {
      result.placement_order = this.bookendLinearize(result.selected_items, anchors);
    } else if (this.config.preserve_chunk_order) {
      result.placement_order = this.preserveOriginalOrder(result.selected_items);
    }

    result.computation_time_ms = performance.now() - startTime;
    
    return PackedResultSchema.parse(result);
  }

  /**
   * Analyze items into groups with constraints
   */
  private analyzeGroups(items: KnapsackItem[]): Map<string, GroupInfo> {
    const groups = new Map<string, GroupInfo>();
    
    for (const item of items) {
      const groupId = item.group_id || `singleton_${item.id}`;
      
      if (!groups.has(groupId)) {
        groups.set(groupId, {
          id: groupId,
          items: [],
          total_tokens: 0,
          avg_importance: 0,
          has_anchors: false
        });
      }
      
      const group = groups.get(groupId)!;
      group.items.push(item);
      group.total_tokens += item.tokens;
      group.has_anchors = group.has_anchors || item.is_head_anchor || item.is_tail_anchor;
    }
    
    // Calculate average importance for each group
    for (const group of groups.values()) {
      group.avg_importance = group.items.reduce((sum, item) => sum + item.importance, 0) / group.items.length;
      
      // Apply group bonus for cohesive content
      if (group.items.length > 1) {
        group.avg_importance = Math.min(1.0, group.avg_importance * (1 + this.config.group_bonus));
      }
    }
    
    return groups;
  }

  /**
   * Identify head and tail anchors
   */
  private identifyAnchors(items: KnapsackItem[]): { head: KnapsackItem[], tail: KnapsackItem[] } {
    const head = items.filter(item => item.is_head_anchor);
    const tail = items.filter(item => item.is_tail_anchor);
    
    return { head, tail };
  }

  /**
   * Exact dynamic programming solution for smaller problems
   */
  private exactKnapsackDP(items: KnapsackItem[], groups: Map<string, GroupInfo>, maxTokens: number): PackedResult {
    const groupList = Array.from(groups.values());
    const n = groupList.length;
    
    // DP table: dp[i][w] = maximum importance using first i groups with weight <= w
    const dp: number[][] = Array(n + 1).fill(null).map(() => Array(maxTokens + 1).fill(0));
    const keep: boolean[][] = Array(n + 1).fill(null).map(() => Array(maxTokens + 1).fill(false));
    
    // Fill DP table
    for (let i = 1; i <= n; i++) {
      const group = groupList[i - 1];
      const weight = group.total_tokens;
      const value = this.calculateGroupValue(group);
      
      for (let w = 0; w <= maxTokens; w++) {
        // Don't take this group
        dp[i][w] = dp[i - 1][w];
        
        // Try taking this group if it fits
        if (weight <= w) {
          const takeValue = dp[i - 1][w - weight] + value;
          if (takeValue > dp[i][w]) {
            dp[i][w] = takeValue;
            keep[i][w] = true;
          }
        }
      }
    }
    
    // Reconstruct solution
    const selectedGroups = new Set<string>();
    let w = maxTokens;
    
    for (let i = n; i >= 1; i--) {
      if (keep[i][w]) {
        const group = groupList[i - 1];
        selectedGroups.add(group.id);
        w -= group.total_tokens;
      }
    }
    
    return this.buildResult(items, selectedGroups, 'exact_dp');
  }

  /**
   * Greedy approximation for larger problems
   */
  private greedyApproximation(items: KnapsackItem[], groups: Map<string, GroupInfo>, maxTokens: number): PackedResult {
    const groupList = Array.from(groups.values());
    
    // Sort groups by value-to-weight ratio with anchor priority
    groupList.sort((a, b) => {
      const ratioA = this.calculateGroupValue(a) / a.total_tokens;
      const ratioB = this.calculateGroupValue(b) / b.total_tokens;
      
      // Prioritize anchor groups
      if (a.has_anchors && !b.has_anchors) return -1;
      if (!a.has_anchors && b.has_anchors) return 1;
      
      return ratioB - ratioA;
    });
    
    const selectedGroups = new Set<string>();
    let usedTokens = 0;
    
    for (const group of groupList) {
      if (usedTokens + group.total_tokens <= maxTokens) {
        selectedGroups.add(group.id);
        usedTokens += group.total_tokens;
      }
    }
    
    return this.buildResult(items, selectedGroups, 'greedy_approx');
  }

  /**
   * Bookend-priority packing for anchor preservation
   */
  private bookendPriorityPacking(
    items: KnapsackItem[], 
    groups: Map<string, GroupInfo>, 
    anchors: { head: KnapsackItem[], tail: KnapsackItem[] },
    maxTokens: number
  ): PackedResult {
    const selectedGroups = new Set<string>();
    let usedTokens = 0;
    
    // Force include anchor groups first
    const anchorGroups = new Set<string>();
    
    for (const anchor of [...anchors.head, ...anchors.tail]) {
      const groupId = anchor.group_id || `singleton_${anchor.id}`;
      if (!anchorGroups.has(groupId)) {
        const group = groups.get(groupId);
        if (group && usedTokens + group.total_tokens <= maxTokens) {
          selectedGroups.add(groupId);
          anchorGroups.add(groupId);
          usedTokens += group.total_tokens;
        }
      }
    }
    
    // Fill remaining capacity with highest value groups
    const remainingGroups = Array.from(groups.values())
      .filter(group => !selectedGroups.has(group.id))
      .sort((a, b) => this.calculateGroupValue(b) / b.total_tokens - this.calculateGroupValue(a) / a.total_tokens);
    
    for (const group of remainingGroups) {
      if (usedTokens + group.total_tokens <= maxTokens) {
        selectedGroups.add(group.id);
        usedTokens += group.total_tokens;
      }
    }
    
    return this.buildResult(items, selectedGroups, 'bookend_priority');
  }

  /**
   * Calculate value of a group considering anchors and diminishing returns
   */
  private calculateGroupValue(group: GroupInfo): number {
    let baseValue = group.avg_importance * group.items.length;
    
    // Apply anchor weights
    for (const item of group.items) {
      if (item.is_head_anchor) {
        baseValue *= this.config.head_anchor_weight;
      }
      if (item.is_tail_anchor) {
        baseValue *= this.config.tail_anchor_weight;
      }
    }
    
    // Apply diminishing returns for large groups
    if (group.items.length > 3) {
      const diminishingFactor = Math.pow(this.config.diminishing_returns_factor, group.items.length - 3);
      baseValue *= diminishingFactor;
    }
    
    return baseValue;
  }

  /**
   * Build final result from selected groups
   */
  private buildResult(items: KnapsackItem[], selectedGroups: Set<string>, algorithm: string): PackedResult {
    const selectedItems = items.filter(item => {
      const groupId = item.group_id || `singleton_${item.id}`;
      return selectedGroups.has(groupId);
    });
    
    const totalTokens = selectedItems.reduce((sum, item) => sum + item.tokens, 0);
    const totalImportance = selectedItems.reduce((sum, item) => sum + item.importance, 0);
    
    const headAnchors = selectedItems.filter(item => item.is_head_anchor).map(item => item.id);
    const tailAnchors = selectedItems.filter(item => item.is_tail_anchor).map(item => item.id);
    
    return {
      selected_items: selectedItems,
      total_tokens: totalTokens,
      total_importance: totalImportance,
      utilization_ratio: totalTokens / this.config.max_tokens,
      head_anchors: headAnchors,
      tail_anchors: tailAnchors,
      placement_order: selectedItems.map(item => item.id), // Will be updated by linearization
      groups_selected: Array.from(selectedGroups),
      algorithm_used: algorithm as any,
      computation_time_ms: 0 // Will be set by caller
    };
  }

  /**
   * Bookend linearization with zig-zag placement
   */
  private bookendLinearize(items: KnapsackItem[], anchors: { head: KnapsackItem[], tail: KnapsackItem[] }): string[] {
    const placement: string[] = [];
    const used = new Set<string>();
    
    // Place head anchors first
    const sortedHeadAnchors = anchors.head
      .filter(anchor => items.find(item => item.id === anchor.id))
      .sort((a, b) => a.sentence_index - b.sentence_index);
    
    for (const anchor of sortedHeadAnchors) {
      placement.push(anchor.id);
      used.add(anchor.id);
    }
    
    // Place tail anchors last (but collect them now)
    const sortedTailAnchors = anchors.tail
      .filter(anchor => items.find(item => item.id === anchor.id))
      .sort((a, b) => b.sentence_index - a.sentence_index);
    
    for (const anchor of sortedTailAnchors) {
      used.add(anchor.id);
    }
    
    // Zig-zag placement of remaining items between anchors
    const remainingItems = items
      .filter(item => !used.has(item.id))
      .sort((a, b) => {
        // Sort by chunk order, then sentence index
        if (a.chunk_id !== b.chunk_id) {
          return a.chunk_id.localeCompare(b.chunk_id);
        }
        return a.sentence_index - b.sentence_index;
      });
    
    // Alternate between front and back placement for context flow
    let placeFront = true;
    const zigzagBuffer: string[] = [];
    
    for (const item of remainingItems) {
      if (placeFront) {
        zigzagBuffer.unshift(item.id);
      } else {
        zigzagBuffer.push(item.id);
      }
      placeFront = !placeFront;
    }
    
    // Add zig-zag items to placement
    placement.push(...zigzagBuffer);
    
    // Add tail anchors at the end
    for (const anchor of sortedTailAnchors) {
      placement.push(anchor.id);
    }
    
    return placement;
  }

  /**
   * Preserve original chunk and sentence order
   */
  private preserveOriginalOrder(items: KnapsackItem[]): string[] {
    return items
      .sort((a, b) => {
        if (a.chunk_id !== b.chunk_id) {
          return a.chunk_id.localeCompare(b.chunk_id);
        }
        return a.sentence_index - b.sentence_index;
      })
      .map(item => item.id);
  }
}

/**
 * Convenience function for knapsack optimization
 */
export async function knapsackPack(
  items: KnapsackItem[],
  config: Partial<KnapsackConfig> = {}
): Promise<PackedResult> {
  const optimizer = new KnapsackOptimizer(config);
  return optimizer.knapsackPack(items);
}

/**
 * Convenience function for bookend linearization
 */
export function bookendLinearize(
  items: KnapsackItem[],
  anchors: { head: KnapsackItem[], tail: KnapsackItem[] },
  zigzagPlacement: boolean = true
): string[] {
  const optimizer = new KnapsackOptimizer({ zigzag_placement: zigzagPlacement });
  return optimizer['bookendLinearize'](items, anchors);
}