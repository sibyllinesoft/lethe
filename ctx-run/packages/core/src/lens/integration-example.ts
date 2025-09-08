/**
 * Integration example showing how to use Lens search server within the existing retrieval pipeline
 * This demonstrates the "S± stage" concept from TODO.md
 */

import type { DB } from '@lethe/sqlite';
import type { Embeddings } from '@lethe/embeddings';
import {
  getLensService,
  detectCodeIntent,
  calculateLagrangianCost,
  symbolGroupsToRetrievalCandidates,
  type LensService,
  type SymbolGroup,
  type CodeIntentResult,
  type LagrangianCostResult
} from './index.js';
import type { Candidate, HybridRetrievalOptions } from '../retrieval/index.js';

/**
 * Enhanced retrieval options that include Lens integration
 */
export interface LensEnhancedRetrievalOptions extends HybridRetrievalOptions {
  /** Context about recent files for code intent detection */
  recent_files?: string[];
  /** Recent activity type for boosting code intent confidence */
  recent_activity?: 'code' | 'docs' | 'mixed';
  /** Current token count for cost analysis */
  current_token_count?: number;
  /** Total token budget for cost analysis */
  total_token_budget?: number;
  /** Enable Lens integration (can be overridden at runtime) */
  enable_lens?: boolean;
}

/**
 * Main integration function implementing the "MAYBE_LENS" concept from TODO.md
 * This acts as the S± stage between S0 and S1 in the retrieval pipeline
 */
export async function maybeLens(
  query: string,
  options: LensEnhancedRetrievalOptions
): Promise<{
  lens_candidates: Candidate[];
  cost_analysis: LagrangianCostResult | null;
  code_intent: CodeIntentResult;
  processing_time_ms: number;
  used_lens: boolean;
  fallback_reason?: string;
}> {
  const startTime = performance.now();
  
  console.log('🔍 MAYBE_LENS: Evaluating query for Lens integration...');
  
  // Step 1: Code intent detection
  const codeIntent = detectCodeIntent(
    query,
    options.recent_files,
    options.recent_activity
  );
  
  console.log(`   Code intent: ${codeIntent.is_code_intent} (confidence: ${(codeIntent.confidence * 100).toFixed(1)}%)`);
  
  // Early exit conditions from TODO.md pseudo-code
  if (!codeIntent.is_code_intent) {
    return {
      lens_candidates: [],
      cost_analysis: null,
      code_intent: codeIntent,
      processing_time_ms: performance.now() - startTime,
      used_lens: false,
      fallback_reason: 'Not code-intent query'
    };
  }
  
  // Check SLA budget constraint
  const remainingBudget = 150; // This would come from actual SLA tracking
  if (remainingBudget < 120) {
    return {
      lens_candidates: [],
      cost_analysis: null,
      code_intent: codeIntent,
      processing_time_ms: performance.now() - startTime,
      used_lens: false,
      fallback_reason: 'Insufficient SLA budget remaining'
    };
  }
  
  // Check if Lens is enabled in options
  if (options.enable_lens === false) {
    return {
      lens_candidates: [],
      cost_analysis: null,
      code_intent: codeIntent,
      processing_time_ms: performance.now() - startTime,
      used_lens: false,
      fallback_reason: 'Lens disabled in options'
    };
  }
  
  try {
    // Get Lens service instance
    const lensService = await getLensService(options.db);
    
    // Check if Lens server is available
    const isAvailable = await lensService.isAvailable();
    if (!isAvailable) {
      return {
        lens_candidates: [],
        cost_analysis: null,
        code_intent: codeIntent,
        processing_time_ms: performance.now() - startTime,
        used_lens: false,
        fallback_reason: 'Lens server not available'
      };
    }
    
    // Step 2: Perform Lens search with bounded configuration
    console.log('   🚀 Performing Lens search...');
    const searchResult = await lensService.search({
      query,
      max_groups: 10,
      topic_fanout_k: 240, // Bounded fanout from TODO.md
      weight_cap: 0.4,     // RAPTOR weight cap
      language_hints: codeIntent.detected_language ? [codeIntent.detected_language] : undefined,
      timeout_ms: Math.min(remainingBudget, 150) // Respect SLA constraint
    });
    
    console.log(`   Found ${searchResult.symbol_groups.length} symbol groups in ${searchResult.processing_time_ms}ms`);
    
    if (searchResult.timeout_hit) {
      return {
        lens_candidates: [],
        cost_analysis: null,
        code_intent: codeIntent,
        processing_time_ms: performance.now() - startTime,
        used_lens: false,
        fallback_reason: 'Lens search timeout'
      };
    }
    
    // Step 3: Lagrangian cost analysis
    console.log('   💰 Performing cost-benefit analysis...');
    const costAnalysis = calculateLagrangianCost(
      searchResult.symbol_groups,
      { 
        lambda_multiplier: 1.2, 
        mu_multiplier: 1.0,
        sla_recall_ms: 150,
        topic_fanout_k: 240,
        weight_cap: 0.4
      } as any, // Simplified for example
      options.current_token_count || 1000,
      options.total_token_budget || 4000,
      searchResult.processing_time_ms
    );
    
    console.log(`   Cost analysis: total_cost=${costAnalysis.total_cost.toFixed(2)}, benefit=${costAnalysis.expected_benefit.toFixed(2)}, acceptable=${costAnalysis.cost_acceptable}`);
    
    // Step 4: Apply cost constraints
    if (!costAnalysis.cost_acceptable || !costAnalysis.sla_constraint_met) {
      return {
        lens_candidates: [],
        cost_analysis: costAnalysis,
        code_intent: codeIntent,
        processing_time_ms: performance.now() - startTime,
        used_lens: false,
        fallback_reason: `Cost constraints not met: acceptable=${costAnalysis.cost_acceptable}, sla=${costAnalysis.sla_constraint_met}`
      };
    }
    
    // Step 5: Convert symbol groups to retrieval candidates
    console.log('   📦 Converting symbol groups to candidates...');
    const lensCandidates = symbolGroupsToRetrievalCandidates(searchResult.symbol_groups);
    
    // Apply "earn its place" filtering if in that mode
    const filteredCandidates = lensCandidates.filter(candidate => {
      // In "earn-its-place" mode, raise the bar for Lens candidates
      const minScore = 0.4; // Higher threshold
      return candidate.score >= minScore;
    });
    
    console.log(`   ✅ LENS integration successful: ${filteredCandidates.length} candidates added`);
    
    return {
      lens_candidates: filteredCandidates,
      cost_analysis: costAnalysis,
      code_intent: codeIntent,
      processing_time_ms: performance.now() - startTime,
      used_lens: true
    };
    
  } catch (error) {
    console.warn(`⚠️ LENS integration failed: ${error}`);
    
    return {
      lens_candidates: [],
      cost_analysis: null,
      code_intent: codeIntent,
      processing_time_ms: performance.now() - startTime,
      used_lens: false,
      fallback_reason: `Lens error: ${error}`
    };
  }
}

/**
 * Enhanced hybrid retrieval that integrates Lens as S± stage
 * This demonstrates how to integrate Lens into the existing retrieval pipeline
 */
export async function lensEnhancedHybridRetrieval(
  queries: string[],
  options: LensEnhancedRetrievalOptions
): Promise<{
  candidates: Candidate[];
  lens_contribution: {
    candidates_count: number;
    processing_time_ms: number;
    used_lens: boolean;
    fallback_reason?: string;
  };
  processing_stats: {
    total_time_ms: number;
    lens_time_ms: number;
    traditional_retrieval_time_ms: number;
  };
}> {
  const totalStartTime = performance.now();
  
  console.log(`🔄 Enhanced hybrid retrieval with Lens integration for ${queries.length} queries`);
  
  // Combine queries for Lens analysis
  const combinedQuery = queries.join(' ');
  
  // Step 1: MAYBE_LENS - Try Lens integration
  const lensResult = await maybeLens(combinedQuery, options);
  
  // Step 2: Traditional hybrid retrieval
  console.log('📊 Running traditional hybrid retrieval...');
  const traditionalStart = performance.now();
  
  // Import the original hybrid retrieval function
  // Note: In practice, this would be imported from '../retrieval/index.js'
  // For this example, we'll simulate it
  const traditionalCandidates: Candidate[] = await simulateTraditionalRetrieval(queries, options);
  
  const traditionalTime = performance.now() - traditionalStart;
  
  // Step 3: Combine and deduplicate candidates
  const allCandidates = [
    ...lensResult.lens_candidates,
    ...traditionalCandidates
  ];
  
  // Simple deduplication by docId (in practice, more sophisticated merging would occur)
  const candidateMap = new Map<string, Candidate>();
  allCandidates.forEach(candidate => {
    const existing = candidateMap.get(candidate.docId);
    if (!existing || candidate.score > existing.score) {
      candidateMap.set(candidate.docId, candidate);
    }
  });
  
  const finalCandidates = Array.from(candidateMap.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, 20); // Take top 20
  
  const totalTime = performance.now() - totalStartTime;
  
  console.log(`🎯 Enhanced retrieval complete: ${finalCandidates.length} final candidates (${totalTime.toFixed(1)}ms total)`);
  console.log(`   Lens contribution: ${lensResult.lens_candidates.length} candidates (used: ${lensResult.used_lens})`);
  
  return {
    candidates: finalCandidates,
    lens_contribution: {
      candidates_count: lensResult.lens_candidates.length,
      processing_time_ms: lensResult.processing_time_ms,
      used_lens: lensResult.used_lens,
      fallback_reason: lensResult.fallback_reason
    },
    processing_stats: {
      total_time_ms: totalTime,
      lens_time_ms: lensResult.processing_time_ms,
      traditional_retrieval_time_ms: traditionalTime
    }
  };
}

/**
 * Simulate traditional retrieval for the example
 * In practice, this would call the real hybridRetrieval function
 */
async function simulateTraditionalRetrieval(
  queries: string[],
  options: HybridRetrievalOptions
): Promise<Candidate[]> {
  // Simulate some processing time
  await new Promise(resolve => setTimeout(resolve, 50));
  
  // Return mock candidates
  return [
    {
      docId: 'traditional_1',
      score: 0.8,
      text: 'Traditional retrieval result 1',
      kind: 'text'
    },
    {
      docId: 'traditional_2',
      score: 0.7,
      text: 'Traditional retrieval result 2',
      kind: 'code'
    }
  ];
}

/**
 * Configuration profiles as mentioned in TODO.md
 */
export const LENS_PROFILES = {
  // Small context (≤4k available): enable Lens auto mode
  small_context: {
    enabled: true,
    mode: 'auto' as const,
    lens_tokens_cap: 2000,
    topic_fanout_k: 120,
    weight_cap: 0.4,
    dpp_rank: 12,
    sla_recall_ms: 120
  },
  
  // Medium context: balanced approach
  medium_context: {
    enabled: true,
    mode: 'auto' as const,
    lens_tokens_cap: 4000,
    topic_fanout_k: 240,
    weight_cap: 0.4,
    dpp_rank: 14,
    sla_recall_ms: 150
  },
  
  // Large context (~100k): earn-its-place mode with higher bar
  large_context: {
    enabled: true,
    mode: 'earn-its-place' as const,
    lens_tokens_cap: 4000,
    topic_fanout_k: 320,
    weight_cap: 0.3, // Tighter weight cap
    dpp_rank: 16,
    sla_recall_ms: 150,
    lambda_multiplier: 1.4 // Higher cost for tokens
  }
};

/**
 * Example usage demonstrating the integration
 */
export async function exampleUsage() {
  // Example 1: Code-focused query with small context
  console.log('=== Example 1: Small Context Code Query ===');
  
  const codeQuery = ['fix error in calculateBM25 function', 'bm25 implementation bug'];
  const smallContextOptions: LensEnhancedRetrievalOptions = {
    db: {} as DB, // Mock DB
    embeddings: {} as Embeddings, // Mock embeddings
    sessionId: 'example_session',
    recent_files: ['src/retrieval/index.ts', 'src/retrieval/bm25.ts'],
    recent_activity: 'code',
    current_token_count: 1500,
    total_token_budget: 4000,
    enable_lens: true
  };
  
  try {
    const result = await lensEnhancedHybridRetrieval(codeQuery, smallContextOptions);
    console.log(`Results: ${result.candidates.length} candidates`);
    console.log(`Lens used: ${result.lens_contribution.used_lens}`);
    console.log(`Processing: ${result.processing_stats.total_time_ms.toFixed(1)}ms total`);
  } catch (error) {
    console.log(`Example failed (expected in test environment): ${error}`);
  }
  
  // Example 2: Non-code query (should skip Lens)
  console.log('\n=== Example 2: Non-Code Query ===');
  
  const generalQuery = ['what is the weather today', 'current temperature'];
  const generalOptions: LensEnhancedRetrievalOptions = {
    ...smallContextOptions,
    recent_activity: 'docs'
  };
  
  try {
    const result = await lensEnhancedHybridRetrieval(generalQuery, generalOptions);
    console.log(`Results: ${result.candidates.length} candidates`);
    console.log(`Lens used: ${result.lens_contribution.used_lens}`);
    console.log(`Fallback reason: ${result.lens_contribution.fallback_reason}`);
  } catch (error) {
    console.log(`Example failed (expected in test environment): ${error}`);
  }
}

// Export the integration components
export {
  maybeLens,
  lensEnhancedHybridRetrieval,
  LENS_PROFILES,
  exampleUsage
};