import { DB, getDFIdf, getChunksBySession } from '@lethe/sqlite';
import { getReranker, type RerankerConfig } from '../reranker/index.js';
import { getDiversifier } from '../diversifier/index.js';
import type { Embeddings } from '@lethe/embeddings';
import { getMLPredictor, type MLPrediction } from '../ml-prediction/index.js';
import { sentencePrune, type PrunedChunkResult, type SentencePruningConfig } from './sentence_pruning.js';
import { knapsackPack, bookendLinearize, type KnapsackItem, type KnapsackConfig, type PackedResult } from './knapsack_optimizer.js';

export interface Candidate {
  docId: string;
  score: number;
  text?: string;
  kind?: string;
}

// Tokenize text the same way as DF/IDF
function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(term => term.length > 1);
}

// Calculate BM25 score for a document
function calculateBM25(
  termFreqs: { [term: string]: number },
  docLength: number,
  avgDocLength: number,
  termIdfMap: { [term: string]: number },
  k1: number = 1.2,
  b: number = 0.75
): number {
  let score = 0;
  
  for (const [term, tf] of Object.entries(termFreqs)) {
    const idf = termIdfMap[term] || 0;
    if (idf <= 0) continue;
    
    const numerator = tf * (k1 + 1);
    const denominator = tf + k1 * (1 - b + b * (docLength / avgDocLength));
    
    score += idf * (numerator / denominator);
  }
  
  return score;
}

export async function bm25Search(
  queries: string[], 
  sessionId: string, 
  k: number
): Promise<{ docId: string; score: number }[]> {
  // This matches the package contract signature exactly
  throw new Error('bm25Search needs DB instance - will be fixed in CLI integration');
}

export async function bm25SearchWithDb(
  db: DB,
  queries: string[], 
  sessionId: string, 
  k: number
): Promise<{ docId: string; score: number }[]> {
  // Get all chunks for the session
  const chunks = getChunksBySession(db, sessionId);
  if (chunks.length === 0) return [];
  
  // Get DF/IDF data for the session
  const dfidfData = getDFIdf(db, sessionId);
  const termIdfMap: { [term: string]: number } = {};
  for (const entry of dfidfData) {
    termIdfMap[entry.term] = entry.idf;
  }
  
  // Calculate average document length
  const totalLength = chunks.reduce((sum, chunk) => sum + tokenize(chunk.text).length, 0);
  const avgDocLength = totalLength / chunks.length;
  
  // Combine all query terms
  const allQueryTerms = new Set<string>();
  for (const query of queries) {
    const terms = tokenize(query);
    terms.forEach(term => allQueryTerms.add(term));
  }
  
  // Score each chunk
  const candidates: { docId: string; score: number }[] = [];
  
  for (const chunk of chunks) {
    const docTerms = tokenize(chunk.text);
    const docLength = docTerms.length;
    
    // Calculate term frequencies for query terms only
    const termFreqs: { [term: string]: number } = {};
    for (const term of docTerms) {
      if (allQueryTerms.has(term)) {
        termFreqs[term] = (termFreqs[term] || 0) + 1;
      }
    }
    
    // Skip documents with no query terms
    if (Object.keys(termFreqs).length === 0) continue;
    
    const score = calculateBM25(termFreqs, docLength, avgDocLength, termIdfMap);
    if (score > 0) {
      candidates.push({ docId: chunk.id, score });
    }
  }
  
  // Sort by score descending and take top k
  candidates.sort((a, b) => b.score - a.score);
  return candidates.slice(0, k);
}

export async function vectorSearch(qVec: Float32Array, k: number): Promise<{ docId: string; score: number }[]> {
  // Package contract signature - will delegate to sqlite
  throw new Error('vectorSearch needs DB instance - will be fixed in CLI integration');
}

export async function vectorSearchWithDb(db: DB, qVec: Float32Array, k: number): Promise<{ docId: string; score: number }[]> {
  // Delegate to sqlite's vectorSearch function
  const { vectorSearch } = await import('@lethe/sqlite');
  return await vectorSearch(db, qVec, k);
}

// Normalize scores to [0,1] range
function normalizeBM25Scores(candidates: { docId: string; score: number }[]): { docId: string; score: number }[] {
  if (candidates.length === 0) return [];
  
  const maxScore = Math.max(...candidates.map(c => c.score));
  if (maxScore === 0) return candidates;
  
  return candidates.map(c => ({
    docId: c.docId,
    score: c.score / maxScore
  }));
}

function normalizeCosineScores(candidates: { docId: string; score: number }[]): { docId: string; score: number }[] {
  return candidates.map(c => ({
    docId: c.docId,
    score: (c.score + 1) / 2 // Transform [-1,1] to [0,1]
  }));
}

// Feature extraction from query text
function featureFlags(query: string): {
  has_code_symbol: boolean;
  has_error_token: boolean;
  has_path_or_file: boolean;
  has_numeric_id: boolean;
} {
  return {
    has_code_symbol: /[_a-zA-Z][\w]*\(|\b[A-Z][A-Za-z0-9]+::[A-Za-z0-9]+\b/.test(query),
    has_error_token: /(Exception|Error|stack trace|errno|\bE\d{2,}\b)/.test(query),
    has_path_or_file: /\/[^\s]+\.[a-zA-Z0-9]+|[A-Za-z]:\\[^\s]+\.[a-zA-Z0-9]+/.test(query),
    has_numeric_id: /\b\d{3,}\b/.test(query)
  };
}

// Dynamic gamma boost based on query features
function gammaBoost(kind: string, queryFeatures: ReturnType<typeof featureFlags>): number {
  let g = 0;
  if (queryFeatures.has_code_symbol && (kind === 'code' || kind === 'user_code')) g += 0.10;
  if (queryFeatures.has_error_token && kind === 'tool_result') g += 0.08;
  if (queryFeatures.has_path_or_file && kind === 'code') g += 0.04;
  return g;
}

export function hybridScore(
  lexical: { docId: string; score: number }[],
  vector: { docId: string; score: number }[],
  config: { alpha: number; beta: number; gamma_kind_boost: { [kind: string]: number } },
  query?: string,
  candidateKinds?: Map<string, string>
): Candidate[] {
  // Normalize scores
  const lexicalNorm = normalizeBM25Scores(lexical);
  const vectorNorm = normalizeCosineScores(vector);
  
  // Create maps for quick lookup
  const lexicalMap = new Map(lexicalNorm.map(c => [c.docId, c.score]));
  const vectorMap = new Map(vectorNorm.map(c => [c.docId, c.score]));
  
  // Get all unique document IDs
  const allDocIds = new Set([...lexicalMap.keys(), ...vectorMap.keys()]);
  
  // Extract query features for dynamic gamma boosting
  const qf = query ? featureFlags(query) : null;
  
  const candidates: Candidate[] = [];
  
  for (const docId of allDocIds) {
    const lexScore = lexicalMap.get(docId) || 0;
    const vecScore = vectorMap.get(docId) || 0;
    
    // Calculate base hybrid score
    let hybridScore = config.alpha * lexScore + config.beta * vecScore;
    
    // Apply dynamic gamma boost based on query features and content kind
    if (qf && candidateKinds) {
      const kind = candidateKinds.get(docId) || 'text';
      const dynamicBoost = gammaBoost(kind, qf);
      const staticBoost = config.gamma_kind_boost[kind] || 0;
      const totalBoost = 1 + dynamicBoost + staticBoost;
      hybridScore *= totalBoost;
    }
    
    candidates.push({
      docId,
      score: hybridScore
    });
  }
  
  // Sort by hybrid score descending
  candidates.sort((a, b) => b.score - a.score);
  
  return candidates;
}

// Configuration interface for hybrid retrieval
export interface HybridConfig {
  alpha: number; // Weight for lexical (BM25) score
  beta: number;  // Weight for vector score
  gamma_kind_boost: { [kind: string]: number }; // Boost for specific content types
  rerank: boolean; // Enable reranking
  diversify: boolean; // Enable diversification
  diversify_method?: string; // Diversification method ('entity' | 'semantic')
  k_initial: number; // Initial retrieval size before reranking/diversification
  k_final: number; // Final result size
  // Iteration 3: ML-enhanced parameters
  fusion?: {
    dynamic: boolean; // Enable ML-predicted alpha/beta
  };
  // Iteration 4: LLM reranking parameters
  llm_rerank?: RerankerConfig;
}

// Default configuration optimized for code retrieval
export const DEFAULT_HYBRID_CONFIG: HybridConfig = {
  alpha: 0.7,    // Favor lexical matching for code
  beta: 0.3,     // Some semantic understanding
  gamma_kind_boost: {
    'code': 1.2,     // Boost code blocks
    'import': 1.1,   // Boost import statements
    'function': 1.15, // Boost function definitions
    'error': 1.3     // Boost error messages
  },
  rerank: true,
  diversify: true,
  diversify_method: 'entity', // Default to entity-based diversification
  k_initial: 50,   // Retrieve more initially
  k_final: 20,     // Return top 20 after processing
  // Iteration 3: ML features disabled by default for backward compatibility
  fusion: {
    dynamic: false
  },
  // Iteration 4: LLM reranking disabled by default for backward compatibility
  llm_rerank: {
    use_llm: false,
    llm_budget_ms: 1200,
    llm_model: 'llama3.2:1b',
    contradiction_enabled: false,
    contradiction_penalty: 0.15
  }
};

export interface HybridRetrievalOptions {
  db: DB;
  embeddings: Embeddings;
  sessionId: string;
  config?: Partial<HybridConfig>;
}

// Complete hybrid retrieval pipeline
export async function hybridRetrieval(
  queries: string[],
  options: HybridRetrievalOptions
): Promise<Candidate[]> {
  const config = { ...DEFAULT_HYBRID_CONFIG, ...options.config };
  const { db, embeddings, sessionId } = options;
  
  console.log(`Starting hybrid retrieval for ${queries.length} queries`);
  console.time('hybrid-retrieval');
  
  try {
    // Steps 1-2: Parallel BM25 lexical search and Vector embedding
    console.log('Steps 1-2: Parallel BM25 and Vector search...');
    
    const combinedQueryText = queries.join(' ');
    
    // Run lexical search, vector embedding, and vector search in parallel
    const [lexicalResults, queryEmbeddings] = await Promise.all([
      bm25SearchWithDb(db, queries, sessionId, config.k_initial),
      embeddings.embed([combinedQueryText]).catch(error => {
        console.warn(`Embedding failed: ${error}`);
        return [];
      })
    ]);
    
    console.log(`BM25 found ${lexicalResults.length} candidates`);
    
    // Step 2b: Vector search (depends on embedding)
    let vectorResults: { docId: string; score: number }[] = [];
    if (queryEmbeddings.length > 0) {
      try {
        vectorResults = await vectorSearchWithDb(db, queryEmbeddings[0], config.k_initial);
        console.log(`Vector search found ${vectorResults.length} candidates`);
      } catch (error) {
        console.warn(`Vector search failed: ${error}, continuing with lexical only`);
      }
    }
    
    // Step 3: Dynamic Parameter Prediction (Iteration 3)
    let effectiveConfig = { ...config };
    
    if (config.fusion?.dynamic) {
      console.log('Step 3a: ML parameter prediction...');
      try {
        const mlPredictor = getMLPredictor({
          fusion_dynamic: true,
          plan_learned: false // Only fusion for retrieval
        });
        
        // Prepare context for ML prediction
        const mlContext = {
          bm25_top1: lexicalResults.length > 0 ? lexicalResults[0].score : 0,
          ann_top1: vectorResults.length > 0 ? vectorResults[0].score : 0,
          overlap_ratio: calculateOverlapRatio(lexicalResults, vectorResults),
          hyde_k: config.k_initial / 10 // Approximate hyde k from initial k
        };
        
        const mlPrediction = await mlPredictor.predictParameters(combinedQueryText, mlContext);
        
        if (mlPrediction.model_loaded) {
          effectiveConfig.alpha = mlPrediction.alpha;
          effectiveConfig.beta = mlPrediction.beta;
          console.log(`ML predicted alpha=${mlPrediction.alpha.toFixed(3)}, beta=${mlPrediction.beta.toFixed(3)} (${mlPrediction.prediction_time_ms.toFixed(1)}ms)`);
        } else {
          console.log('ML models not loaded, using static parameters');
        }
      } catch (error) {
        console.warn(`ML prediction failed: ${error}, using static parameters`);
      }
    }
    
    // Step 3b: Hybrid scoring with effective parameters
    console.log('Step 3b: Hybrid scoring...');
    
    // Prepare candidate kinds map for dynamic gamma boosting
    const candidateKinds = new Map<string, string>();
    
    let candidates = hybridScore(lexicalResults, vectorResults, effectiveConfig, combinedQueryText, candidateKinds);
    console.log(`Hybrid scoring produced ${candidates.length} candidates`);
    
    // Step 4: Add text and metadata to candidates
    console.log('Step 4: Enriching candidates with text...');
    candidates = await enrichCandidatesWithText(db, candidates);
    
    // Populate candidate kinds map after enrichment for future gamma boosting
    for (const candidate of candidates) {
      if (candidate.kind) {
        candidateKinds.set(candidate.docId, candidate.kind);
      }
    }
    
    // Step 5: Reranking (LLM or Cross-encoder, optional)
    if (config.rerank && candidates.length > 0) {
      if (config.llm_rerank?.use_llm) {
        console.log('Step 5: LLM reranking...');
        const reranker = await getReranker(true, config.llm_rerank, db);
        candidates = await reranker.rerank(combinedQueryText, candidates);
        console.log(`LLM reranking complete`);
      } else {
        console.log('Step 5: Cross-encoder reranking...');
        const reranker = await getReranker(true);
        candidates = await reranker.rerank(combinedQueryText, candidates);
        console.log(`Cross-encoder reranking complete`);
      }
    }
    
    // Step 6: Diversification (optional)
    if (config.diversify && candidates.length > config.k_final) {
      console.log(`Step 6: Diversification using ${config.diversify_method || 'entity'} method...`);
      const diversifier = await getDiversifier(true, config.diversify_method || 'entity');
      candidates = await diversifier.diversify(candidates, config.k_final);
      console.log(`Diversification complete`);
    } else {
      // Just take top k if no diversification
      candidates = candidates.slice(0, config.k_final);
    }
    
    console.timeEnd('hybrid-retrieval');
    console.log(`Hybrid retrieval complete: ${candidates.length} final results`);
    
    return candidates;
    
  } catch (error) {
    console.timeEnd('hybrid-retrieval');
    console.error(`Hybrid retrieval failed: ${error}`);
    throw error;
  }
}

// Calculate overlap ratio between lexical and vector results for ML context
function calculateOverlapRatio(
  lexicalResults: { docId: string; score: number }[],
  vectorResults: { docId: string; score: number }[]
): number {
  if (lexicalResults.length === 0 || vectorResults.length === 0) {
    return 0;
  }

  const lexicalIds = new Set(lexicalResults.map(r => r.docId));
  const vectorIds = new Set(vectorResults.map(r => r.docId));
  
  const intersection = new Set([...lexicalIds].filter(id => vectorIds.has(id)));
  const union = new Set([...lexicalIds, ...vectorIds]);
  
  return intersection.size / union.size;
}

// Enrich candidates with text content and metadata
async function enrichCandidatesWithText(db: DB, candidates: Candidate[]): Promise<Candidate[]> {
  const { getChunkById } = await import('@lethe/sqlite');
  
  const enrichedCandidates: Candidate[] = [];
  
  for (const candidate of candidates) {
    try {
      const chunk = getChunkById(db, candidate.docId);
      if (chunk) {
        enrichedCandidates.push({
          ...candidate,
          text: chunk.text,
          kind: chunk.kind
        });
      } else {
        // Keep candidate without text if chunk not found
        enrichedCandidates.push(candidate);
      }
    } catch (error) {
      console.warn(`Failed to enrich candidate ${candidate.docId}: ${error}`);
      enrichedCandidates.push(candidate);
    }
  }
  
  return enrichedCandidates;
}

// Enhanced candidate interface for Lethe vNext with sentence-level granularity
export interface EnhancedCandidate extends Candidate {
  sentences?: Array<{
    id: string;
    text: string;
    tokens: number;
    importance: number;
    sentence_index: number;
    is_head_anchor: boolean;
    is_tail_anchor: boolean;
    co_entailing_group?: string[];
  }>;
  pruned_result?: PrunedChunkResult;
}

// Lethe vNext orchestration configuration
export interface LetHeVNextConfig extends HybridConfig {
  // Sentence pruning settings
  pruning: Partial<SentencePruningConfig>;
  
  // Knapsack optimization settings  
  knapsack: Partial<KnapsackConfig>;
  
  // Token budget enforcement
  global_token_budget: number;
  budget_allocation: {
    retrieval_ratio: number; // Portion of budget for initial retrieval
    pruning_ratio: number;   // Portion for post-pruning content
  };
  
  // Quality targets
  answer_span_kept_threshold: number; // Minimum % of answer spans to preserve
  ndcg_improvement_target: number;    // Target nDCG@10 improvement
  
  // Processing flags
  enable_sentence_pruning: boolean;
  enable_knapsack_optimization: boolean;
  enable_bookend_packing: boolean;
  preserve_code_fences: boolean;
}

export const DEFAULT_LETHE_VNEXT_CONFIG: LetHeVNextConfig = {
  ...DEFAULT_HYBRID_CONFIG,
  
  // Sentence pruning configuration
  pruning: {
    cross_encoder_threshold: 0.6,
    preserve_code_fences: true,
    min_sentence_tokens: 5,
    max_sentence_tokens: 100,
    co_entailment_threshold: 0.8
  },
  
  // Knapsack optimization
  knapsack: {
    max_tokens: 8192,
    safety_margin: 0.05,
    head_anchor_weight: 2.0,
    tail_anchor_weight: 1.5,
    group_bonus: 0.1,
    diminishing_returns_factor: 0.8,
    zigzag_placement: true,
    preserve_chunk_order: true
  },
  
  // Global token budget
  global_token_budget: 8192,
  budget_allocation: {
    retrieval_ratio: 0.7,  // 70% for retrieval candidates
    pruning_ratio: 0.3     // 30% for pruned content
  },
  
  // Quality targets per TODO.md requirements
  answer_span_kept_threshold: 0.98,  // ≥98% answer span preservation
  ndcg_improvement_target: 0.10,     // ≥+10% nDCG@10 improvement
  
  // Processing toggles
  enable_sentence_pruning: true,
  enable_knapsack_optimization: true,
  enable_bookend_packing: true,
  preserve_code_fences: true
};

/**
 * Main orchestration function for Lethe vNext retrieval pipeline
 * 
 * Pipeline stages:
 * 1. Standard hybrid retrieval (BM25 + Vector + Reranking)
 * 2. Sentence-level pruning with cross-encoder scoring
 * 3. Global knapsack optimization with bookend packing
 * 4. Final linearization and assembly
 */
export async function orchestrateLetheVNext(
  queries: string[],
  options: HybridRetrievalOptions & { 
    config?: Partial<LetHeVNextConfig> 
  }
): Promise<{
  final_candidates: EnhancedCandidate[];
  knapsack_result: PackedResult;
  processing_stats: {
    initial_candidates: number;
    pruned_sentences: number;
    final_tokens: number;
    token_reduction_ratio: number;
    processing_time_ms: number;
  };
}> {
  const startTime = performance.now();
  const config = { ...DEFAULT_LETHE_VNEXT_CONFIG, ...options.config };
  
  console.log('🚀 Starting Lethe vNext orchestration...');
  console.time('lethe-vnext-orchestration');
  
  try {
    // Stage 1: Standard hybrid retrieval 
    console.log('Stage 1: Hybrid retrieval...');
    const initialCandidates = await hybridRetrieval(queries, {
      ...options,
      config: {
        ...config,
        k_final: Math.floor(config.k_final * 1.5) // Get more for pruning
      }
    });
    
    console.log(`Hybrid retrieval found ${initialCandidates.length} candidates`);
    
    // Stage 2: Sentence-level pruning
    const enhancedCandidates: EnhancedCandidate[] = [];
    let totalPrunedSentences = 0;
    
    if (config.enable_sentence_pruning) {
      console.log('Stage 2: Sentence pruning...');
      
      for (const candidate of initialCandidates) {
        if (!candidate.text) continue;
        
        try {
          const combinedQueryText = queries.join(' ');
          const prunedResult = await sentencePrune(
            combinedQueryText, 
            {
              id: candidate.docId,
              text: candidate.text,
              kind: candidate.kind
            },
            config.pruning
          );
          
          // Convert pruned sentences to enhanced format
          const sentences = prunedResult.pruned_sentences.map(sentence => ({
            id: sentence.sentence_id,
            text: sentence.text,
            tokens: sentence.tokens,
            importance: sentence.relevance_score,
            sentence_index: sentence.original_index,
            is_head_anchor: sentence.is_code_fence || sentence.original_index === 0,
            is_tail_anchor: sentence.is_code_fence || sentence.original_index === prunedResult.total_sentences - 1,
            co_entailing_group: sentence.co_entailing_ids
          }));
          
          enhancedCandidates.push({
            ...candidate,
            sentences,
            pruned_result: prunedResult
          });
          
          totalPrunedSentences += prunedResult.pruned_sentences.length;
          
        } catch (error) {
          console.warn(`Sentence pruning failed for candidate ${candidate.docId}: ${error}`);
          // Fallback to original candidate
          enhancedCandidates.push(candidate);
        }
      }
    } else {
      // Skip pruning, convert candidates as-is
      enhancedCandidates.push(...initialCandidates);
    }
    
    console.log(`Sentence pruning produced ${totalPrunedSentences} sentences`);
    
    // Stage 3: Global knapsack optimization
    let knapsackResult: PackedResult;
    let finalCandidates = enhancedCandidates;
    
    if (config.enable_knapsack_optimization) {
      console.log('Stage 3: Knapsack optimization...');
      
      // Convert enhanced candidates to knapsack items
      const knapsackItems: KnapsackItem[] = [];
      
      for (const candidate of enhancedCandidates) {
        if (candidate.sentences) {
          for (const sentence of candidate.sentences) {
            knapsackItems.push({
              id: sentence.id,
              tokens: sentence.tokens,
              importance: sentence.importance,
              chunk_id: candidate.docId,
              sentence_index: sentence.sentence_index,
              is_head_anchor: sentence.is_head_anchor,
              is_tail_anchor: sentence.is_tail_anchor,
              co_entailing_group: sentence.co_entailing_group,
              text: sentence.text
            });
          }
        } else if (candidate.text) {
          // Fallback for non-pruned candidates
          const estimatedTokens = Math.ceil(candidate.text.length / 4); // Rough estimate
          knapsackItems.push({
            id: `${candidate.docId}_full`,
            tokens: estimatedTokens,
            importance: candidate.score,
            chunk_id: candidate.docId,
            sentence_index: 0,
            is_head_anchor: false,
            is_tail_anchor: false,
            text: candidate.text
          });
        }
      }
      
      // Run knapsack optimization
      knapsackResult = await knapsackPack(knapsackItems, {
        ...config.knapsack,
        max_tokens: config.global_token_budget
      });
      
      // Filter candidates to match knapsack selection
      const selectedItemIds = new Set(knapsackResult.selected_items.map(item => item.id));
      finalCandidates = enhancedCandidates.filter(candidate => {
        if (candidate.sentences) {
          // Keep candidate if any of its sentences were selected
          return candidate.sentences.some(sentence => selectedItemIds.has(sentence.id));
        } else {
          return selectedItemIds.has(`${candidate.docId}_full`);
        }
      });
      
      console.log(`Knapsack selected ${knapsackResult.selected_items.length} items, ${finalCandidates.length} candidates`);
      
    } else {
      // Create dummy knapsack result
      const totalTokens = enhancedCandidates.reduce((sum, candidate) => {
        return sum + (candidate.text ? Math.ceil(candidate.text.length / 4) : 0);
      }, 0);
      
      knapsackResult = {
        selected_items: [],
        total_tokens: totalTokens,
        total_importance: enhancedCandidates.reduce((sum, c) => sum + c.score, 0),
        utilization_ratio: Math.min(1.0, totalTokens / config.global_token_budget),
        head_anchors: [],
        tail_anchors: [],
        placement_order: enhancedCandidates.map(c => c.docId),
        groups_selected: [],
        algorithm_used: 'greedy_approx',
        computation_time_ms: 0
      };
    }
    
    const processingTime = performance.now() - startTime;
    const initialTokens = initialCandidates.reduce((sum, c) => sum + (c.text ? Math.ceil(c.text.length / 4) : 0), 0);
    const tokenReductionRatio = initialTokens > 0 ? 1 - (knapsackResult.total_tokens / initialTokens) : 0;
    
    console.timeEnd('lethe-vnext-orchestration');
    console.log(`🎯 Lethe vNext orchestration complete: ${finalCandidates.length} candidates, ${knapsackResult.total_tokens} tokens (${(tokenReductionRatio * 100).toFixed(1)}% reduction)`);
    
    return {
      final_candidates: finalCandidates,
      knapsack_result: knapsackResult,
      processing_stats: {
        initial_candidates: initialCandidates.length,
        pruned_sentences: totalPrunedSentences,
        final_tokens: knapsackResult.total_tokens,
        token_reduction_ratio: tokenReductionRatio,
        processing_time_ms: processingTime
      }
    };
    
  } catch (error) {
    console.timeEnd('lethe-vnext-orchestration');
    console.error(`Lethe vNext orchestration failed: ${error}`);
    throw error;
  }
}