/**
 * Sentence Pruning Implementation for Lethe vNext
 * =================================================
 * 
 * Implements Provence-style query-conditioned sentence masking within selected chunks.
 * This is a critical component for achieving 30-50% token reduction while maintaining
 * Answer-Span-Kept ≥ 98%.
 * 
 * Core Features:
 * - Cross-encoder scoring for sentence-query relevance
 * - Binary masking with group-keep rules
 * - Code fence preservation
 * - Adjacent co-entailing sentence preservation
 * - Fallback to unpruned on failure
 */

import { DB } from '@lethe/sqlite';
import crypto from 'crypto';

// Schema interfaces matching verification/schemas/pruned.json
export interface SentenceData {
  sid: string;
  span: [number, number];
  tokens: number;
  score: number;
  content: string;
  group_kept?: boolean;
  co_entailing?: string[];
  code_fence?: boolean;
}

export interface PrunedChunkResult {
  chunk_id: string;
  kept: SentenceData[];
  chunk_score: number;
  metadata: {
    original_token_count: number;
    pruned_token_count: number;
    pruning_ratio: number;
    query_hash: string;
    threshold_used?: number;
    fallback_unpruned?: boolean;
    processing_time_ms: number;
    cross_encoder_model?: string;
  };
}

export interface SentencePruningConfig {
  threshold: number;
  preserve_code_fences: boolean;
  enable_co_entailing: boolean;
  enable_group_rules: boolean;
  fallback_on_empty: boolean;
  max_processing_time_ms: number;
  cross_encoder_model: string;
}

export const DEFAULT_SENTENCE_PRUNING_CONFIG: SentencePruningConfig = {
  threshold: 0.5,
  preserve_code_fences: true,
  enable_co_entailing: true,
  enable_group_rules: true,
  fallback_on_empty: true,
  max_processing_time_ms: 2000,
  cross_encoder_model: 'ms-marco-MiniLM-L-6-v2'
};

/**
 * Simple sentence splitter that preserves code blocks and structure
 */
export function splitIntoSentences(text: string): Array<{
  content: string;
  span: [number, number];
  is_code_fence: boolean;
}> {
  const sentences: Array<{
    content: string;
    span: [number, number];
    is_code_fence: boolean;
  }> = [];
  
  // Track code fence boundaries
  const codeFenceRegex = /```[\s\S]*?```|`[^`\n]+`/g;
  const codeFences: Array<[number, number]> = [];
  let match;
  
  while ((match = codeFenceRegex.exec(text)) !== null) {
    codeFences.push([match.index, match.index + match[0].length]);
  }
  
  // Function to check if position is within code fence
  const isInCodeFence = (pos: number): boolean => {
    return codeFences.some(([start, end]) => pos >= start && pos < end);
  };
  
  // Split by sentence boundaries while respecting code fences
  const sentenceBoundaries = /[.!?]+\s+/g;
  let lastEnd = 0;
  
  while ((match = sentenceBoundaries.exec(text)) !== null) {
    const endPos = match.index + match[0].length;
    
    // Don't split inside code fences
    if (!isInCodeFence(match.index)) {
      const sentenceText = text.slice(lastEnd, endPos).trim();
      if (sentenceText.length > 0) {
        const hasCodeFence = codeFences.some(([start, end]) => 
          start >= lastEnd && end <= endPos
        );
        
        sentences.push({
          content: sentenceText,
          span: [lastEnd, endPos],
          is_code_fence: hasCodeFence
        });
        lastEnd = endPos;
      }
    }
  }
  
  // Add remaining text as final sentence
  if (lastEnd < text.length) {
    const remainingText = text.slice(lastEnd).trim();
    if (remainingText.length > 0) {
      const hasCodeFence = codeFences.some(([start, end]) => 
        start >= lastEnd && end <= text.length
      );
      
      sentences.push({
        content: remainingText,
        span: [lastEnd, text.length],
        is_code_fence: hasCodeFence
      });
    }
  }
  
  return sentences;
}

/**
 * Simple tokenizer that matches the rest of the system
 */
export function tokenCount(text: string): number {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(term => term.length > 0).length;
}

/**
 * Mock cross-encoder scoring function
 * In production, this would use a real cross-encoder model
 */
export async function scoreSentenceRelevance(
  query: string,
  sentence: string,
  model: string = 'ms-marco-MiniLM-L-6-v2'
): Promise<number> {
  // Simple scoring based on term overlap as placeholder
  // In production, this would use actual cross-encoder model
  
  const queryTerms = new Set(
    query.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(term => term.length > 1)
  );
  
  const sentenceTerms = new Set(
    sentence.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(term => term.length > 1)
  );
  
  if (queryTerms.size === 0 || sentenceTerms.size === 0) {
    return 0.0;
  }
  
  // Calculate Jaccard similarity as proxy for relevance
  const intersection = new Set([...queryTerms].filter(x => sentenceTerms.has(x)));
  const union = new Set([...queryTerms, ...sentenceTerms]);
  
  const baseScore = intersection.size / union.size;
  
  // Add some randomness to simulate model variation
  const noise = (Math.random() - 0.5) * 0.1;
  const score = Math.max(0, Math.min(1, baseScore + noise));
  
  return score;
}

/**
 * Check if sentences are co-entailing (simple heuristic)
 */
export function areCoEntailing(sentence1: string, sentence2: string): boolean {
  // Simple heuristic: high term overlap indicates co-entailment
  const terms1 = new Set(
    sentence1.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(term => term.length > 2)
  );
  
  const terms2 = new Set(
    sentence2.toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(term => term.length > 2)
  );
  
  if (terms1.size === 0 || terms2.size === 0) return false;
  
  const intersection = new Set([...terms1].filter(x => terms2.has(x)));
  const minSize = Math.min(terms1.size, terms2.size);
  
  // Consider co-entailing if they share >60% of terms
  return intersection.size / minSize > 0.6;
}

/**
 * Apply group-keep rules for important sentence types
 */
export function shouldKeepByGroupRules(sentence: string, kind?: string): boolean {
  // Always keep sentences with code patterns
  if (/```|`[^`]+`|\b(function|class|import|export|def|if|for|while)\b/.test(sentence)) {
    return true;
  }
  
  // Always keep error messages and stack traces
  if (/(Error|Exception|at\s+.*:\d+|\bstdout\b|\bstderr\b)/.test(sentence)) {
    return true;
  }
  
  // Keep sentences with specific file paths or URLs
  if (/\/[^\s]+\.[a-zA-Z0-9]+|https?:\/\/[^\s]+/.test(sentence)) {
    return true;
  }
  
  return false;
}

/**
 * Main sentence pruning function implementing Provence-style masking
 */
export async function sentencePrune(
  query: string,
  chunk: { id: string; text: string; kind?: string },
  config: Partial<SentencePruningConfig> = {}
): Promise<PrunedChunkResult> {
  const startTime = Date.now();
  const effectiveConfig = { ...DEFAULT_SENTENCE_PRUNING_CONFIG, ...config };
  
  console.log(`Starting sentence pruning for chunk ${chunk.id}, query: "${query.slice(0, 50)}..."`);
  
  // Split chunk into sentences
  const rawSentences = splitIntoSentences(chunk.text);
  console.log(`Split into ${rawSentences.length} sentences`);
  
  if (rawSentences.length === 0) {
    // Empty chunk - return minimal result
    return {
      chunk_id: chunk.id,
      kept: [],
      chunk_score: 0,
      metadata: {
        original_token_count: 0,
        pruned_token_count: 0,
        pruning_ratio: 0,
        query_hash: crypto.createHash('sha256').update(query).digest('hex'),
        fallback_unpruned: true,
        processing_time_ms: Date.now() - startTime,
        cross_encoder_model: effectiveConfig.cross_encoder_model
      }
    };
  }
  
  // Score each sentence with cross-encoder
  const scoredSentences: SentenceData[] = [];
  
  for (let i = 0; i < rawSentences.length; i++) {
    const rawSentence = rawSentences[i];
    const sid = `${chunk.id}.${i}`;
    
    try {
      // Score sentence relevance
      const score = await scoreSentenceRelevance(
        query, 
        rawSentence.content, 
        effectiveConfig.cross_encoder_model
      );
      
      // Check group rules
      const groupKept = effectiveConfig.enable_group_rules && 
        shouldKeepByGroupRules(rawSentence.content, chunk.kind);
      
      // Find co-entailing sentences if enabled
      let coEntailing: string[] = [];
      if (effectiveConfig.enable_co_entailing) {
        for (let j = 0; j < rawSentences.length; j++) {
          if (i !== j && areCoEntailing(rawSentence.content, rawSentences[j].content)) {
            coEntailing.push(`${chunk.id}.${j}`);
          }
        }
      }
      
      scoredSentences.push({
        sid,
        span: rawSentence.span,
        tokens: tokenCount(rawSentence.content),
        score,
        content: rawSentence.content,
        group_kept: groupKept,
        co_entailing: coEntailing,
        code_fence: rawSentence.is_code_fence
      });
      
    } catch (error) {
      console.warn(`Failed to score sentence ${sid}: ${error}`);
      // Add with zero score to avoid dropping completely
      scoredSentences.push({
        sid,
        span: rawSentence.span,
        tokens: tokenCount(rawSentence.content),
        score: 0,
        content: rawSentence.content,
        group_kept: false,
        code_fence: rawSentence.is_code_fence
      });
    }
  }
  
  // Apply pruning logic
  const keptSentences: SentenceData[] = [];
  const processedIds = new Set<string>();
  
  for (const sentence of scoredSentences) {
    if (processedIds.has(sentence.sid)) continue;
    
    // Keep if above threshold OR group rules OR code fence preservation
    const shouldKeep = 
      sentence.score >= effectiveConfig.threshold ||
      sentence.group_kept ||
      (effectiveConfig.preserve_code_fences && sentence.code_fence);
    
    if (shouldKeep) {
      keptSentences.push(sentence);
      processedIds.add(sentence.sid);
      
      // Also keep co-entailing sentences if enabled
      if (effectiveConfig.enable_co_entailing && sentence.co_entailing) {
        for (const coEntailingId of sentence.co_entailing) {
          if (!processedIds.has(coEntailingId)) {
            const coEntailingSentence = scoredSentences.find(s => s.sid === coEntailingId);
            if (coEntailingSentence) {
              keptSentences.push(coEntailingSentence);
              processedIds.add(coEntailingId);
            }
          }
        }
      }
    }
  }
  
  // Calculate metrics
  const originalTokens = scoredSentences.reduce((sum, s) => sum + s.tokens, 0);
  const prunedTokens = keptSentences.reduce((sum, s) => sum + s.tokens, 0);
  const pruningRatio = originalTokens > 0 ? 1 - (prunedTokens / originalTokens) : 0;
  
  // Fallback to unpruned if no sentences kept and fallback enabled
  let finalKeptSentences = keptSentences;
  let fallbackUsed = false;
  
  if (keptSentences.length === 0 && effectiveConfig.fallback_on_empty) {
    console.warn(`No sentences kept for chunk ${chunk.id}, using fallback`);
    finalKeptSentences = scoredSentences;
    fallbackUsed = true;
  }
  
  // Calculate chunk score (max of kept sentences)
  const chunkScore = finalKeptSentences.length > 0 
    ? Math.max(...finalKeptSentences.map(s => s.score))
    : 0;
  
  const processingTime = Date.now() - startTime;
  
  console.log(
    `Sentence pruning complete: ${finalKeptSentences.length}/${scoredSentences.length} kept, ` +
    `${pruningRatio.toFixed(2)} pruned, ${processingTime}ms`
  );
  
  return {
    chunk_id: chunk.id,
    kept: finalKeptSentences,
    chunk_score: chunkScore,
    metadata: {
      original_token_count: originalTokens,
      pruned_token_count: prunedTokens,
      pruning_ratio,
      query_hash: crypto.createHash('sha256').update(query).digest('hex'),
      threshold_used: effectiveConfig.threshold,
      fallback_unpruned: fallbackUsed,
      processing_time_ms: processingTime,
      cross_encoder_model: effectiveConfig.cross_encoder_model
    }
  };
}

/**
 * Batch sentence pruning for multiple chunks
 */
export async function batchSentencePrune(
  query: string,
  chunks: Array<{ id: string; text: string; kind?: string }>,
  config: Partial<SentencePruningConfig> = {}
): Promise<PrunedChunkResult[]> {
  console.log(`Starting batch sentence pruning for ${chunks.length} chunks`);
  const startTime = Date.now();
  
  // Process chunks in parallel (with reasonable concurrency limit)
  const BATCH_SIZE = 5;
  const results: PrunedChunkResult[] = [];
  
  for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
    const batch = chunks.slice(i, i + BATCH_SIZE);
    const batchPromises = batch.map(chunk => 
      sentencePrune(query, chunk, config)
    );
    
    try {
      const batchResults = await Promise.all(batchPromises);
      results.push(...batchResults);
    } catch (error) {
      console.error(`Batch sentence pruning failed for batch starting at ${i}: ${error}`);
      // Add empty results to maintain order
      for (const chunk of batch) {
        results.push({
          chunk_id: chunk.id,
          kept: [],
          chunk_score: 0,
          metadata: {
            original_token_count: tokenCount(chunk.text),
            pruned_token_count: 0,
            pruning_ratio: 1,
            query_hash: crypto.createHash('sha256').update(query).digest('hex'),
            fallback_unpruned: true,
            processing_time_ms: 0,
            cross_encoder_model: config.cross_encoder_model || DEFAULT_SENTENCE_PRUNING_CONFIG.cross_encoder_model
          }
        });
      }
    }
  }
  
  const totalTime = Date.now() - startTime;
  console.log(`Batch sentence pruning complete: ${chunks.length} chunks in ${totalTime}ms`);
  
  return results;
}

/**
 * Validate pruned result against schema constraints
 */
export function validatePrunedResult(result: PrunedChunkResult): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];
  
  // Check basic structure
  if (!result.chunk_id || typeof result.chunk_id !== 'string') {
    errors.push('Missing or invalid chunk_id');
  }
  
  if (!Array.isArray(result.kept)) {
    errors.push('kept must be an array');
  }
  
  if (typeof result.chunk_score !== 'number' || result.chunk_score < 0 || result.chunk_score > 1) {
    errors.push('chunk_score must be a number between 0 and 1');
  }
  
  // Validate each kept sentence
  for (let i = 0; i < result.kept.length; i++) {
    const sentence = result.kept[i];
    const prefix = `kept[${i}]`;
    
    if (!sentence.sid || typeof sentence.sid !== 'string') {
      errors.push(`${prefix}.sid is required and must be string`);
    }
    
    if (!Array.isArray(sentence.span) || sentence.span.length !== 2) {
      errors.push(`${prefix}.span must be array of length 2`);
    }
    
    if (typeof sentence.tokens !== 'number' || sentence.tokens < 0) {
      errors.push(`${prefix}.tokens must be non-negative number`);
    }
    
    if (typeof sentence.score !== 'number' || sentence.score < 0 || sentence.score > 1) {
      errors.push(`${prefix}.score must be number between 0 and 1`);
    }
    
    if (!sentence.content || typeof sentence.content !== 'string') {
      errors.push(`${prefix}.content is required and must be string`);
    }
  }
  
  // Validate metadata
  if (!result.metadata) {
    errors.push('metadata is required');
  } else {
    const meta = result.metadata;
    
    if (typeof meta.original_token_count !== 'number' || meta.original_token_count < 0) {
      errors.push('metadata.original_token_count must be non-negative number');
    }
    
    if (typeof meta.pruned_token_count !== 'number' || meta.pruned_token_count < 0) {
      errors.push('metadata.pruned_token_count must be non-negative number');
    }
    
    if (typeof meta.pruning_ratio !== 'number' || meta.pruning_ratio < 0 || meta.pruning_ratio > 1) {
      errors.push('metadata.pruning_ratio must be number between 0 and 1');
    }
    
    if (!meta.query_hash || !/^[a-f0-9]{64}$/.test(meta.query_hash)) {
      errors.push('metadata.query_hash must be valid SHA-256 hex string');
    }
  }
  
  return {
    valid: errors.length === 0,
    errors
  };
}