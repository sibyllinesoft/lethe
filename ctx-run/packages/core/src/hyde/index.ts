import type { DB } from '@lethe/sqlite';
import { getConfig } from '@lethe/sqlite';
import { getOllamaBridge, safeParseJSON } from '../ollama/index.js';

export interface HydeResult {
  queries: string[];
  pseudo: string;
}

export interface HydeConfig {
  model: string;
  temperature: number;
  numQueries: number;
  maxTokens: number;
  timeoutMs: number;
  enabled: boolean;
}

export const DEFAULT_HYDE_CONFIG: HydeConfig = {
  model: 'xgen-small:4b',
  temperature: 0.3,
  numQueries: 3,
  maxTokens: 512,
  timeoutMs: 10000,
  enabled: true
};

// Extract anchor terms (rare, specific terms that are good for retrieval)
function extractAnchorTerms(text: string): string[] {
  // Tokenize and filter
  const tokens = text.toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(term => term.length >= 3); // Minimum length

  // Count frequency
  const freq: { [term: string]: number } = {};
  tokens.forEach(term => {
    freq[term] = (freq[term] || 0) + 1;
  });

  // Get rare terms (appear 1-2 times and are longer)
  const rareTerms = Object.entries(freq)
    .filter(([term, count]) => 
      count <= 2 && 
      term.length >= 4 && 
      !isCommonWord(term)
    )
    .map(([term]) => term)
    .slice(0, 5); // Top 5 anchor terms

  return rareTerms;
}

// Check if a term is too common/generic to be useful as an anchor
function isCommonWord(term: string): boolean {
  const commonWords = new Set([
    'this', 'that', 'with', 'have', 'will', 'from', 'they', 'know',
    'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when',
    'come', 'may', 'say', 'each', 'which', 'she', 'do', 'how', 'their',
    'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
    'her', 'would', 'make', 'like', 'into', 'him', 'has', 'two',
    'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first',
    'water', 'been', 'call', 'who', 'oil', 'its', 'now', 'find',
    'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part',
    // Technical common words
    'function', 'method', 'class', 'object', 'string', 'number',
    'array', 'list', 'data', 'value', 'result', 'type', 'parameter',
    'variable', 'code', 'file', 'line', 'text', 'content'
  ]);
  
  return commonWords.has(term);
}

function buildHydePrompt(originalQuery: string, anchorTerms: string[]): string {
  const anchorTermsText = anchorTerms.length > 0 
    ? `\nImportant terms to incorporate: ${anchorTerms.join(', ')}`
    : '';

  return `You are an expert at generating discriminative search queries for code and technical documentation retrieval.

Original query: "${originalQuery}"${anchorTermsText}

Generate ${DEFAULT_HYDE_CONFIG.numQueries} alternative search queries that would find relevant technical content. Each query should:
1. Be specific and technical (avoid generic terms like "how to", "what is")
2. Include concrete technical terms, APIs, patterns, or error conditions
3. Focus on actionable, searchable concepts
4. Incorporate the important terms mentioned above when relevant
5. Be different from each other to capture various aspects

Also write a hypothetical document excerpt (2-3 sentences) that would contain the answer to the original query.

Format your response as JSON:
{
  "queries": [
    "specific technical query 1",
    "specific technical query 2", 
    "specific technical query 3"
  ],
  "pseudo": "A hypothetical document excerpt that would answer the original query. Should be technical and specific, mentioning relevant APIs, patterns, or solutions."
}`;
}

export async function generateHyde(
  db: DB, 
  originalQuery: string, 
  config?: Partial<HydeConfig>
): Promise<HydeResult> {
  const finalConfig = { ...DEFAULT_HYDE_CONFIG, ...config };
  
  // Check if HyDE is enabled in config
  if (!finalConfig.enabled) {
    console.debug('HyDE disabled in config');
    return {
      queries: [originalQuery],
      pseudo: `Relevant technical content for: ${originalQuery}`
    };
  }

  // Override with database config if available
  try {
    const timeoutConfig = getConfig(db, 'timeouts');
    if (timeoutConfig?.hyde_ms) {
      finalConfig.timeoutMs = timeoutConfig.hyde_ms;
    }
  } catch (error) {
    console.debug(`Could not load HyDE timeout config: ${error}`);
  }

  try {
    const bridge = await getOllamaBridge(db);
    
    // Check if Ollama is available
    const isAvailable = await bridge.isAvailable();
    if (!isAvailable) {
      console.warn('Ollama not available, falling back to original query');
      return {
        queries: [originalQuery],
        pseudo: `Technical documentation or code related to: ${originalQuery}`
      };
    }

    // Extract anchor terms from the original query
    const anchorTerms = extractAnchorTerms(originalQuery);
    console.debug(`HyDE anchor terms: ${anchorTerms.join(', ')}`);
    
    const prompt = buildHydePrompt(originalQuery, anchorTerms);
    
    console.debug('Generating HyDE queries...');
    const startTime = Date.now();
    
    const response = await bridge.generate({
      model: finalConfig.model,
      prompt,
      temperature: finalConfig.temperature,
      max_tokens: finalConfig.maxTokens
    });

    const duration = Date.now() - startTime;
    console.debug(`HyDE generation took ${duration}ms`);

    // Parse the JSON response with fallback
    const fallback: HydeResult = {
      queries: [originalQuery, `${originalQuery} implementation`, `${originalQuery} example`],
      pseudo: `Technical content addressing ${originalQuery} with implementation details and examples.`
    };

    const result = safeParseJSON(response.response, fallback);
    
    // Validate result structure
    if (!result.queries || !Array.isArray(result.queries) || result.queries.length === 0) {
      console.warn('Invalid HyDE response structure, using fallback');
      return fallback;
    }

    if (!result.pseudo || typeof result.pseudo !== 'string') {
      result.pseudo = fallback.pseudo;
    }

    // Ensure queries are strings and filter empty ones
    result.queries = result.queries
      .filter(q => typeof q === 'string' && q.trim().length > 0)
      .slice(0, finalConfig.numQueries);

    if (result.queries.length === 0) {
      console.warn('No valid queries generated, using fallback');
      return fallback;
    }

    console.debug(`HyDE generated ${result.queries.length} queries`);
    return result;

  } catch (error) {
    console.warn(`HyDE generation failed: ${error}, using original query`);
    return {
      queries: [originalQuery],
      pseudo: `Technical documentation for: ${originalQuery}`
    };
  }
}

// Test function for CLI diagnostics
export async function testHyde(db: DB, query: string = 'async error handling'): Promise<{
  success: boolean;
  result?: HydeResult;
  duration?: number;
  error?: string;
}> {
  try {
    const startTime = Date.now();
    const result = await generateHyde(db, query);
    const duration = Date.now() - startTime;
    
    return { success: true, result, duration };
  } catch (error: any) {
    return { success: false, error: error?.message || String(error) };
  }
}