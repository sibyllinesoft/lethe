import type { DB } from '@lethe/sqlite';
import { getConfig } from '@lethe/sqlite';
import { getOllamaBridge, safeParseJSON } from '../ollama/index.js';

export interface QueryUnderstandingConfig {
  // Feature toggles
  query_rewrite: boolean;
  query_decompose: boolean;
  
  // LLM settings
  llm_model: string;
  max_tokens: number;
  temperature: number;
  
  // Timeouts (required from specification)
  rewrite_timeout_ms: number;
  decompose_timeout_ms: number;
  
  // Quality controls
  max_subqueries: number;
  similarity_threshold: number;
  enabled: boolean;
}

export const DEFAULT_QUERY_UNDERSTANDING_CONFIG: QueryUnderstandingConfig = {
  query_rewrite: true,
  query_decompose: true,
  llm_model: 'xgen-small:4b',
  max_tokens: 512,
  temperature: 0.1,
  rewrite_timeout_ms: 1500,  // As per specification
  decompose_timeout_ms: 2000, // As per specification
  max_subqueries: 5,
  similarity_threshold: 0.8,
  enabled: true
};

// Load configuration from database with fallback to defaults
export function getQueryUnderstandingConfig(db: DB, overrides?: Partial<QueryUnderstandingConfig>): QueryUnderstandingConfig {
  let config = { ...DEFAULT_QUERY_UNDERSTANDING_CONFIG };
  
  try {
    // Load plan settings
    const planConfig = getConfig(db, 'plan');
    if (planConfig) {
      if (planConfig.query_rewrite !== undefined) config.query_rewrite = planConfig.query_rewrite;
      if (planConfig.query_decompose !== undefined) config.query_decompose = planConfig.query_decompose;
    }
    
    // Load timeouts
    const timeoutConfig = getConfig(db, 'timeouts');
    if (timeoutConfig) {
      if (timeoutConfig.rewrite_ms !== undefined) config.rewrite_timeout_ms = timeoutConfig.rewrite_ms;
      if (timeoutConfig.decompose_ms !== undefined) config.decompose_timeout_ms = timeoutConfig.decompose_ms;
    }
    
    // Load query understanding specific settings
    const quConfig = getConfig(db, 'query_understanding');
    if (quConfig) {
      if (quConfig.enabled !== undefined) config.enabled = quConfig.enabled;
      if (quConfig.llm_model !== undefined) config.llm_model = quConfig.llm_model;
      if (quConfig.max_tokens !== undefined) config.max_tokens = quConfig.max_tokens;
      if (quConfig.temperature !== undefined) config.temperature = quConfig.temperature;
      if (quConfig.max_subqueries !== undefined) config.max_subqueries = quConfig.max_subqueries;
      if (quConfig.similarity_threshold !== undefined) config.similarity_threshold = quConfig.similarity_threshold;
    }
    
    // Load iteration2 settings
    const iter2Config = getConfig(db, 'iteration2');
    if (iter2Config) {
      if (iter2Config.enable_query_preprocessing !== undefined) {
        config.enabled = iter2Config.enable_query_preprocessing;
      }
    }
    
  } catch (error) {
    console.debug(`Could not load query understanding config: ${error}`);
  }
  
  // Apply overrides
  if (overrides) {
    config = { ...config, ...overrides };
  }
  
  return config;
}

export interface ProcessedQuery {
  original_query: string;
  canonical_query?: string;
  subqueries?: string[];
  processing_time_ms: number;
  llm_calls_made: number;
  rewrite_success: boolean;
  decompose_success: boolean;
  errors: string[];
}

export interface ConversationTurn {
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

// Query rewriting component
export async function rewriteQuery(
  db: DB,
  query: string,
  recentTurns: ConversationTurn[] = [],
  config?: Partial<QueryUnderstandingConfig>
): Promise<{ canonical: string; success: boolean; error?: string }> {
  const finalConfig = getQueryUnderstandingConfig(db, config);
  
  if (!finalConfig.query_rewrite || !finalConfig.enabled) {
    return { canonical: query, success: false, error: 'Query rewrite disabled' };
  }

  try {
    const bridge = await getOllamaBridge(db);
    
    if (!(await bridge.isAvailable())) {
      return { canonical: query, success: false, error: 'Ollama not available' };
    }

    // Build context from recent turns (up to 6 as per specification)
    const contextTurns = recentTurns.slice(-6);
    const contextText = contextTurns.length > 0 
      ? contextTurns.map(turn => `${turn.role}: ${turn.content}`).join('\n')
      : '';

    const prompt = buildRewritePrompt(query, contextText);
    
    const startTime = Date.now();
    
    // Use the specified timeout of 1500ms
    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error('Rewrite timeout')), finalConfig.rewrite_timeout_ms);
    });

    const responsePromise = bridge.generate({
      model: finalConfig.llm_model,
      prompt,
      temperature: finalConfig.temperature,
      max_tokens: finalConfig.max_tokens
    });

    const response = await Promise.race([responsePromise, timeoutPromise]);
    
    const duration = Date.now() - startTime;
    console.debug(`Query rewrite took ${duration}ms`);

    // Parse JSON response as per specification
    const result = safeParseJSON(response.response, { canonical: query });
    
    if (!result.canonical || typeof result.canonical !== 'string') {
      return { canonical: query, success: false, error: 'Invalid JSON response structure' };
    }

    // Validate the rewritten query
    const validatedQuery = validateRewrite(query, result.canonical);
    
    return { 
      canonical: validatedQuery, 
      success: validatedQuery !== query,
      error: validatedQuery === query ? 'Rewrite failed validation' : undefined
    };

  } catch (error: any) {
    const errorMsg = error?.message || String(error);
    console.warn(`Query rewrite failed: ${errorMsg}`);
    return { canonical: query, success: false, error: errorMsg };
  }
}

// Query decomposition component  
export async function decomposeQuery(
  db: DB,
  query: string,
  config?: Partial<QueryUnderstandingConfig>
): Promise<{ subs: string[]; success: boolean; error?: string }> {
  const finalConfig = getQueryUnderstandingConfig(db, config);
  
  if (!finalConfig.query_decompose || !finalConfig.enabled) {
    return { subs: [], success: false, error: 'Query decomposition disabled' };
  }

  // Skip decomposition for simple queries
  if (query.split(' ').length < 5) {
    return { subs: [], success: false, error: 'Query too simple for decomposition' };
  }

  try {
    const bridge = await getOllamaBridge(db);
    
    if (!(await bridge.isAvailable())) {
      return { subs: [], success: false, error: 'Ollama not available' };
    }

    const prompt = buildDecomposePrompt(query, finalConfig.max_subqueries);
    
    const startTime = Date.now();
    
    // Use the specified timeout of 2000ms
    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error('Decompose timeout')), finalConfig.decompose_timeout_ms);
    });

    const responsePromise = bridge.generate({
      model: finalConfig.llm_model,
      prompt,
      temperature: finalConfig.temperature,
      max_tokens: finalConfig.max_tokens
    });

    const response = await Promise.race([responsePromise, timeoutPromise]);
    
    const duration = Date.now() - startTime;
    console.debug(`Query decomposition took ${duration}ms`);

    // Parse JSON response as per specification
    const result = safeParseJSON(response.response, { subs: [] });
    
    if (!Array.isArray(result.subs)) {
      return { subs: [], success: false, error: 'Invalid JSON response structure' };
    }

    // Filter and validate subqueries
    const validSubs = result.subs
      .filter((sub: any) => typeof sub === 'string' && sub.trim().length > 0)
      .slice(0, finalConfig.max_subqueries);

    return { 
      subs: validSubs, 
      success: validSubs.length > 0,
      error: validSubs.length === 0 ? 'No valid subqueries generated' : undefined
    };

  } catch (error: any) {
    const errorMsg = error?.message || String(error);
    console.warn(`Query decomposition failed: ${errorMsg}`);
    return { subs: [], success: false, error: errorMsg };
  }
}

// Main query understanding pipeline
export async function processQuery(
  db: DB,
  originalQuery: string,
  recentTurns: ConversationTurn[] = [],
  config?: Partial<QueryUnderstandingConfig>
): Promise<ProcessedQuery> {
  const finalConfig = getQueryUnderstandingConfig(db, config);
  const startTime = Date.now();
  
  const result: ProcessedQuery = {
    original_query: originalQuery,
    processing_time_ms: 0,
    llm_calls_made: 0,
    rewrite_success: false,
    decompose_success: false,
    errors: []
  };

  // Step 1: Query rewriting (runs before decomposition as per specification)
  if (finalConfig.query_rewrite) {
    try {
      const rewriteResult = await rewriteQuery(db, originalQuery, recentTurns, config);
      result.canonical_query = rewriteResult.canonical;
      result.rewrite_success = rewriteResult.success;
      result.llm_calls_made++;
      
      if (rewriteResult.error) {
        result.errors.push(`Rewrite: ${rewriteResult.error}`);
      }
    } catch (error: any) {
      result.errors.push(`Rewrite exception: ${error?.message || String(error)}`);
    }
  }

  // Step 2: Query decomposition (only if rewrite was successful, as per specification)
  if (finalConfig.query_decompose && result.rewrite_success && result.canonical_query) {
    try {
      const decomposeResult = await decomposeQuery(db, result.canonical_query, config);
      result.subqueries = decomposeResult.subs;
      result.decompose_success = decomposeResult.success;
      result.llm_calls_made++;
      
      if (decomposeResult.error) {
        result.errors.push(`Decompose: ${decomposeResult.error}`);
      }
    } catch (error: any) {
      result.errors.push(`Decompose exception: ${error?.message || String(error)}`);
    }
  }

  result.processing_time_ms = Date.now() - startTime;
  
  console.debug(`Query processing complete: ${result.llm_calls_made} LLM calls, ${result.processing_time_ms}ms`);
  
  return result;
}

// Helper functions

function buildRewritePrompt(query: string, context: string): string {
  const contextSection = context 
    ? `\nRecent conversation:\n${context}\n` 
    : '';

  return `Rewrite the user's last message into a standalone, specific query using up to 20 tokens from the provided recent turns. Return JSON: {"canonical": "..."}. No explanations.${contextSection}

User's last message: ${query}

Rewritten query:`;
}

function buildDecomposePrompt(query: string, maxSubqueries: number): string {
  return `If the query is multifaceted, output 2-${maxSubqueries} focused sub-queries covering distinct aspects. JSON: {"subs":["...","..."]}. Else, {"subs":[]}.

Query: ${query}

Response:`;
}

function validateRewrite(original: string, rewritten: string): string {
  // Basic validation
  if (!rewritten || rewritten.length < 5) {
    return original;
  }
  
  // Length check
  if (rewritten.length > 500) {
    return original;
  }
  
  // Simple similarity check (word overlap)
  const origWords = new Set(original.toLowerCase().split(/\s+/));
  const rewriteWords = new Set(rewritten.toLowerCase().split(/\s+/));
  const intersection = new Set([...origWords].filter(x => rewriteWords.has(x)));
  const overlap = intersection.size / origWords.size;
  
  // Reject if too dissimilar (less than 30% word overlap)
  if (overlap < 0.3) {
    return original;
  }
  
  return rewritten;
}

// Diagnostic function for CLI
export async function testQueryUnderstanding(
  db: DB, 
  query: string = 'async error handling patterns',
  recentTurns: ConversationTurn[] = []
): Promise<{
  success: boolean;
  result?: ProcessedQuery;
  error?: string;
}> {
  try {
    const result = await processQuery(db, query, recentTurns);
    return { success: true, result };
  } catch (error: any) {
    return { success: false, error: error?.message || String(error) };
  }
}