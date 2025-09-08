// Lens integration for symbol-aware search server

/**
 * Symbol group representing a cohesive set of code symbols and their context
 * Each group contains definitions, key references, and implementations 
 * as returned by the Lens search server
 */
export interface SymbolGroup {
  /** Unique identifier for this symbol group */
  id: string;
  /** Primary symbol being represented (function, class, variable, etc.) */
  primary_symbol: string;
  /** Programming language/file type */
  language: string;
  /** File path where the primary symbol is defined */
  file_path: string;
  /** Symbol definition with precise location */
  definition: CodeAtom;
  /** Key references and call sites */
  references: CodeAtom[];
  /** Related implementations or overrides */
  implementations: CodeAtom[];
  /** Test hints if available */
  test_hints?: CodeAtom[];
  /** Estimated token count for this group */
  estimated_tokens: number;
  /** Relevance score from Lens search */
  relevance_score: number;
  /** RAPTOR topic weight (bounded by weight_cap) */
  topic_weight: number;
  /** Whether this is from LSP precision spine */
  is_precise_match: boolean;
}

/**
 * Individual code atom representing a specific piece of code with precise location
 */
export interface CodeAtom {
  /** Unique identifier */
  id: string;
  /** Code content */
  content: string;
  /** File path */
  file_path: string;
  /** Start line (1-indexed) */
  start_line: number;
  /** End line (1-indexed) */
  end_line: number;
  /** Start character offset within start line */
  start_char: number;
  /** End character offset within end line */
  end_char: number;
  /** Type of code atom (definition, reference, implementation, test) */
  atom_type: 'definition' | 'reference' | 'implementation' | 'test';
  /** Symbol name this atom relates to */
  symbol_name: string;
  /** Estimated token count */
  tokens: number;
  /** Context importance score */
  importance: number;
}

/**
 * Configuration for Lens search server integration
 */
export interface LensConfig {
  /** Lens server base URL */
  base_url: string;
  /** Connection timeout in milliseconds */
  connect_timeout_ms: number;
  /** Request timeout in milliseconds */
  request_timeout_ms: number;
  /** SLA budget for recall constraint (default: 150ms) */
  sla_recall_ms: number;
  /** Topic fanout bound (k parameter) */
  topic_fanout_k: number;
  /** RAPTOR weight cap to prevent semantic drowning */
  weight_cap: number;
  /** Maximum tokens per Lens response */
  max_tokens_per_response: number;
  /** Enable/disable Lens integration */
  enabled: boolean;
  /** Lens operation mode */
  mode: 'auto' | 'earn-its-place' | 'disabled';
  /** DPP rank for diversity (r parameter) */
  dpp_rank: number;
  /** Enable facility location over entities */
  enable_facility_location: boolean;
  /** Enable bounded log-det DPP for avoiding near-duplicates */
  enable_log_det_dpp: boolean;
  /** Lambda multiplier for Lagrangian cost control */
  lambda_multiplier: number;
  /** Mu multiplier for compute cost control */
  mu_multiplier: number;
  /** Token budget cap for Lens results */
  lens_tokens_cap: number;
}

/**
 * Lens search request payload
 */
export interface LensSearchRequest {
  /** Search query text */
  query: string;
  /** Maximum number of symbol groups to return */
  max_groups: number;
  /** Topic fanout parameter */
  topic_fanout_k?: number;
  /** Weight cap for RAPTOR weights */
  weight_cap?: number;
  /** File path context if available */
  file_context?: string[];
  /** Repository context if available */
  repo_context?: string;
  /** Language hints */
  language_hints?: string[];
  /** Request timeout for this specific search */
  timeout_ms?: number;
}

/**
 * Lens search response
 */
export interface LensSearchResponse {
  /** Symbol groups found */
  symbol_groups: SymbolGroup[];
  /** Total processing time on server */
  processing_time_ms: number;
  /** Whether LSP was available for precision spine */
  lsp_available: boolean;
  /** Number of topics expanded in RAPTOR */
  topics_expanded: number;
  /** Whether the request hit any timeout constraints */
  timeout_hit: boolean;
  /** Additional metadata */
  metadata: {
    /** Server version */
    version: string;
    /** Query analysis results */
    query_analysis: {
      /** Detected programming language */
      detected_language?: string;
      /** Extracted symbols from query */
      extracted_symbols: string[];
      /** Confidence in code intent */
      code_intent_confidence: number;
    };
  };
}

/**
 * Code intent detection result
 */
export interface CodeIntentResult {
  /** Whether the query appears to be code-related */
  is_code_intent: boolean;
  /** Confidence score 0-1 */
  confidence: number;
  /** Detected patterns that indicate code intent */
  patterns: {
    /** Has programming symbols (functions, methods, etc.) */
    has_code_symbols: boolean;
    /** Has error tokens */
    has_error_tokens: boolean;
    /** Has file/path references */
    has_file_paths: boolean;
    /** Has language keywords */
    has_language_keywords: boolean;
    /** Has numeric IDs (issue numbers, etc.) */
    has_numeric_ids: boolean;
  };
  /** Detected programming language if any */
  detected_language?: string;
  /** Extracted symbol names */
  extracted_symbols: string[];
}

/**
 * Lagrangian cost analysis result
 */
export interface LagrangianCostResult {
  /** Token cost component */
  token_cost: number;
  /** Compute cost component */
  compute_cost: number;
  /** Total Lagrangian cost */
  total_cost: number;
  /** Expected benefit/utility */
  expected_benefit: number;
  /** Cost-benefit ratio */
  cost_benefit_ratio: number;
  /** Whether cost is acceptable given budget */
  cost_acceptable: boolean;
  /** SLA constraint status */
  sla_constraint_met: boolean;
  /** Detailed cost breakdown */
  cost_breakdown: {
    /** Base retrieval token cost */
    base_tokens: number;
    /** Lens-specific token cost */
    lens_tokens: number;
    /** Cross-encoder cost */
    ce_compute: number;
    /** DPP computation cost */
    dpp_compute: number;
    /** Total estimated latency */
    estimated_latency_ms: number;
  };
}

/**
 * Main Lens service interface
 */
export interface LensService {
  /** Search for symbol groups based on query */
  search(request: LensSearchRequest): Promise<LensSearchResponse>;
  
  /** Check if Lens server is available */
  isAvailable(): Promise<boolean>;
  
  /** Test connection to Lens server */
  testConnection(): Promise<{
    available: boolean;
    latency_ms?: number;
    error?: string;
  }>;
  
  /** Get server health and status */
  getStatus(): Promise<{
    healthy: boolean;
    version: string;
    lsp_available: boolean;
    raptor_cache_status: 'warm' | 'cold' | 'unavailable';
  }>;
}

/**
 * Implementation of Lens service client
 */
class LensServiceImpl implements LensService {
  private config: LensConfig;

  constructor(config: LensConfig) {
    this.config = config;
  }

  async search(request: LensSearchRequest): Promise<LensSearchResponse> {
    if (!this.config.enabled) {
      return this.getEmptyResponse();
    }

    const controller = new AbortController();
    const timeout_ms = request.timeout_ms || this.config.request_timeout_ms;
    const timeoutId = setTimeout(() => controller.abort(), timeout_ms);

    try {
      const searchPayload = {
        ...request,
        topic_fanout_k: request.topic_fanout_k || this.config.topic_fanout_k,
        weight_cap: request.weight_cap || this.config.weight_cap
      };

      const response = await fetch(`${this.config.base_url}/api/search`, {
        method: 'POST',
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(searchPayload)
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Lens HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Validate response structure
      if (!data.symbol_groups || !Array.isArray(data.symbol_groups)) {
        throw new Error('Invalid Lens response: missing symbol_groups array');
      }

      return data as LensSearchResponse;
    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error instanceof Error && error.name === 'AbortError') {
        console.warn(`Lens search timeout after ${timeout_ms}ms`);
        return this.getTimeoutResponse();
      }
      
      console.warn(`Lens search failed: ${error}`);
      throw error;
    }
  }

  async isAvailable(): Promise<boolean> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.config.connect_timeout_ms);

      const response = await fetch(`${this.config.base_url}/api/health`, {
        signal: controller.signal,
        headers: {
          'Accept': 'application/json'
        }
      });

      clearTimeout(timeoutId);
      return response.ok;
    } catch (error) {
      console.debug(`Lens not available: ${error}`);
      return false;
    }
  }

  async testConnection(): Promise<{
    available: boolean;
    latency_ms?: number;
    error?: string;
  }> {
    const startTime = Date.now();
    
    try {
      const available = await this.isAvailable();
      const latency_ms = Date.now() - startTime;
      
      return { available, latency_ms };
    } catch (error: any) {
      const latency_ms = Date.now() - startTime;
      return {
        available: false,
        latency_ms,
        error: error?.message || String(error)
      };
    }
  }

  async getStatus(): Promise<{
    healthy: boolean;
    version: string;
    lsp_available: boolean;
    raptor_cache_status: 'warm' | 'cold' | 'unavailable';
  }> {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.config.connect_timeout_ms);

      const response = await fetch(`${this.config.base_url}/api/status`, {
        signal: controller.signal,
        headers: {
          'Accept': 'application/json'
        }
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      return {
        healthy: data.healthy || false,
        version: data.version || 'unknown',
        lsp_available: data.lsp_available || false,
        raptor_cache_status: data.raptor_cache_status || 'unavailable'
      };
    } catch (error) {
      return {
        healthy: false,
        version: 'unknown',
        lsp_available: false,
        raptor_cache_status: 'unavailable'
      };
    }
  }

  private getEmptyResponse(): LensSearchResponse {
    return {
      symbol_groups: [],
      processing_time_ms: 0,
      lsp_available: false,
      topics_expanded: 0,
      timeout_hit: false,
      metadata: {
        version: 'disabled',
        query_analysis: {
          extracted_symbols: [],
          code_intent_confidence: 0
        }
      }
    };
  }

  private getTimeoutResponse(): LensSearchResponse {
    return {
      symbol_groups: [],
      processing_time_ms: this.config.request_timeout_ms,
      lsp_available: false,
      topics_expanded: 0,
      timeout_hit: true,
      metadata: {
        version: 'timeout',
        query_analysis: {
          extracted_symbols: [],
          code_intent_confidence: 0
        }
      }
    };
  }
}

/**
 * Default Lens configuration
 */
export const DEFAULT_LENS_CONFIG: LensConfig = {
  base_url: 'http://localhost:5678',
  connect_timeout_ms: 500,
  request_timeout_ms: 150, // SLA-Recall@150ms constraint
  sla_recall_ms: 150,
  topic_fanout_k: 240, // Balanced between 120-320 range
  weight_cap: 0.4, // Prevent semantic drowning
  max_tokens_per_response: 4000,
  enabled: false, // Disabled by default
  mode: 'auto',
  dpp_rank: 14, // Balanced between 12-16 range
  enable_facility_location: true,
  enable_log_det_dpp: true,
  lambda_multiplier: 1.2, // Slightly higher cost for Lens tokens
  mu_multiplier: 1.0, // Standard compute cost
  lens_tokens_cap: 4000 // 2-4k range for small context
};

// Cache for service instances
const lensServiceCache = new Map<string, LensService>();

/**
 * Get or create a Lens service instance
 */
export async function getLensService(db?: any): Promise<LensService> {
  let config = DEFAULT_LENS_CONFIG;

  // Override with database config if available
  if (db) {
    try {
      // Configuration will be loaded from the enhanced config system
      const { getConfig } = await import('../config/index.js');
      const fullConfig = getConfig();
      
      if (fullConfig.lens) {
        config = {
          ...config,
          ...fullConfig.lens
        };
      }
    } catch (error: any) {
      console.debug(`Could not load Lens config: ${error?.message || error}`);
    }
  }

  const cacheKey = JSON.stringify(config);
  
  if (!lensServiceCache.has(cacheKey)) {
    lensServiceCache.set(cacheKey, new LensServiceImpl(config));
  }

  return lensServiceCache.get(cacheKey)!;
}

/**
 * Code intent detection utility
 * Determines if a query is code-focused based on patterns and context
 */
export function detectCodeIntent(
  query: string,
  recentFiles?: string[],
  recentActivity?: 'code' | 'docs' | 'mixed'
): CodeIntentResult {
  const lowerQuery = query.toLowerCase();
  
  // Programming symbols pattern
  const codeSymbolPatterns = [
    /\b[a-zA-Z_]\w*\s*\(/, // function calls
    /\b[A-Z][A-Za-z0-9]*::[A-Za-z0-9]+\b/, // namespace/class methods
    /\b(class|interface|function|def|func|method|async|await)\s+\w+/,
    /\b(import|require|from|export|module)\b/,
    /\b(const|let|var|final|static)\s+\w+/,
    /\b(public|private|protected|internal)\s+(class|fun|def|function)/,
    /\b(std::|np\.|pd\.|torch\.)/  // common library prefixes
  ];

  // Error and debugging patterns
  const errorPatterns = [
    /\b(exception|error|traceback|stacktrace|errno)\b/i,
    /\b(E\d{2,})\b/, // error codes like E404
    /\b(failed|failing|broken|bug|issue)\b/i,
    /\b(debug|debugging|trace|log)\b/i
  ];

  // File and path patterns
  const pathPatterns = [
    /\/[^\s]+\.[a-zA-Z0-9]{1,4}\b/, // Unix paths with extensions
    /[A-Za-z]:[\\\/][^\s]+\.[a-zA-Z0-9]{1,4}\b/, // Windows paths
    /\b\w+\.(js|ts|py|rs|go|java|cpp|hpp|h|c|rb|php|swift|kt|scala)\b/,
    /\bsrc\/|tests?\/|lib\/|bin\/|build\/|dist\//
  ];

  // Language keywords
  const languageKeywords = [
    // JavaScript/TypeScript
    /\b(javascript|typescript|node|npm|yarn|webpack|babel|react|vue|angular)\b/i,
    // Python
    /\b(python|django|flask|pandas|numpy|pytorch|tensorflow)\b/i,
    // Java
    /\b(java|spring|maven|gradle|junit)\b/i,
    // C/C++
    /\b(cpp|c\+\+|gcc|clang|cmake|makefile)\b/i,
    // Rust
    /\b(rust|cargo|rustc)\b/i,
    // Go
    /\b(golang|go\s+build|go\s+run)\b/i,
    // Database
    /\b(sql|mysql|postgresql|mongodb|redis)\b/i,
    // General
    /\b(api|rest|graphql|json|xml|yaml)\b/i
  ];

  // Numeric ID patterns (issue numbers, etc.)
  const numericIdPatterns = [
    /\b(issue|bug|pr|pull\s*request|ticket)\s*#?\d{2,}/i,
    /\b\d{3,}\b/ // Generic numeric IDs
  ];

  // Check patterns
  const hasCodeSymbols = codeSymbolPatterns.some(pattern => pattern.test(query));
  const hasErrorTokens = errorPatterns.some(pattern => pattern.test(query));
  const hasFilePaths = pathPatterns.some(pattern => pattern.test(query));
  const hasLanguageKeywords = languageKeywords.some(pattern => pattern.test(query));
  const hasNumericIds = numericIdPatterns.some(pattern => pattern.test(query));

  // Extract symbol names
  const symbolMatches = query.match(/\b[a-zA-Z_]\w*(?=\s*\()|[A-Z][A-Za-z0-9]*::[A-Za-z0-9]+/g) || [];
  const extractedSymbols = Array.from(new Set(symbolMatches));

  // Detect programming language
  let detectedLanguage: string | undefined;
  if (query.match(/\.(js|jsx|ts|tsx)\b|javascript|typescript|node/i)) {
    detectedLanguage = 'javascript';
  } else if (query.match(/\.py\b|python|django|flask/i)) {
    detectedLanguage = 'python';
  } else if (query.match(/\.rs\b|rust|cargo/i)) {
    detectedLanguage = 'rust';
  } else if (query.match(/\.go\b|golang/i)) {
    detectedLanguage = 'go';
  } else if (query.match(/\.(java|kt)\b|kotlin|spring/i)) {
    detectedLanguage = 'java';
  } else if (query.match(/\.(cpp|hpp|h|c)\b|c\+\+|gcc/i)) {
    detectedLanguage = 'cpp';
  }

  // Calculate confidence
  let confidence = 0;
  if (hasCodeSymbols) confidence += 0.4;
  if (hasErrorTokens) confidence += 0.2;
  if (hasFilePaths) confidence += 0.3;
  if (hasLanguageKeywords) confidence += 0.2;
  if (hasNumericIds) confidence += 0.1;
  if (extractedSymbols.length > 0) confidence += 0.2;
  if (detectedLanguage) confidence += 0.1;

  // Boost confidence based on recent activity
  if (recentActivity === 'code') confidence += 0.1;
  if (recentFiles && recentFiles.some(f => /\.(js|ts|py|rs|go|java|cpp|h)$/.test(f))) {
    confidence += 0.15;
  }

  // Cap confidence at 1.0
  confidence = Math.min(1.0, confidence);

  // Determine if it's code intent (threshold: 0.3)
  const isCodeIntent = confidence >= 0.3;

  return {
    is_code_intent: isCodeIntent,
    confidence,
    patterns: {
      has_code_symbols: hasCodeSymbols,
      has_error_tokens: hasErrorTokens,
      has_file_paths: hasFilePaths,
      has_language_keywords: hasLanguageKeywords,
      has_numeric_ids: hasNumericIds
    },
    detected_language: detectedLanguage,
    extracted_symbols: extractedSymbols
  };
}

/**
 * Lagrangian cost controller
 * Calculates token and compute costs with lambda/mu multipliers
 */
export function calculateLagrangianCost(
  symbolGroups: SymbolGroup[],
  config: LensConfig,
  currentTokens: number,
  totalBudget: number,
  estimatedLatencyMs: number
): LagrangianCostResult {
  // Calculate token costs
  const lensTokens = symbolGroups.reduce((sum, group) => sum + group.estimated_tokens, 0);
  const baseTokens = currentTokens;
  const totalTokens = baseTokens + lensTokens;
  
  // Token cost with lambda multiplier
  const tokenCost = config.lambda_multiplier * lensTokens;
  
  // Compute cost estimation
  const ceComputeCost = symbolGroups.length * 0.5; // Rough CE cost per group
  const dppComputeCost = symbolGroups.length > 1 ? Math.log(symbolGroups.length) * 2 : 0;
  const baseComputeCost = ceComputeCost + dppComputeCost;
  
  // Compute cost with mu multiplier
  const computeCost = config.mu_multiplier * baseComputeCost;
  
  // Total Lagrangian cost
  const totalCost = tokenCost + computeCost;
  
  // Expected benefit estimation (simplified)
  const precisionSpineBonus = symbolGroups.filter(g => g.is_precise_match).length * 0.3;
  const diversityBonus = Math.min(0.2, symbolGroups.length * 0.05);
  const topicWeightSum = symbolGroups.reduce((sum, g) => sum + g.topic_weight, 0);
  const expectedBenefit = precisionSpineBonus + diversityBonus + topicWeightSum * 0.1;
  
  // Cost-benefit analysis
  const costBenefitRatio = expectedBenefit > 0 ? totalCost / expectedBenefit : Infinity;
  const costAcceptable = totalTokens <= totalBudget && costBenefitRatio <= 3.0;
  
  // SLA constraint check
  const slaConstraintMet = estimatedLatencyMs <= config.sla_recall_ms;
  
  return {
    token_cost: tokenCost,
    compute_cost: computeCost,
    total_cost: totalCost,
    expected_benefit: expectedBenefit,
    cost_benefit_ratio: costBenefitRatio,
    cost_acceptable: costAcceptable,
    sla_constraint_met: slaConstraintMet,
    cost_breakdown: {
      base_tokens: baseTokens,
      lens_tokens: lensTokens,
      ce_compute: ceComputeCost,
      dpp_compute: dppComputeCost,
      estimated_latency_ms: estimatedLatencyMs
    }
  };
}

/**
 * Convert SymbolGroup to candidate format for integration with retrieval pipeline
 */
export function symbolGroupsToRetrievalCandidates(
  symbolGroups: SymbolGroup[]
): Array<{
  docId: string;
  score: number;
  text: string;
  kind: string;
  metadata: {
    is_lens_result: boolean;
    symbol_group: SymbolGroup;
  };
}> {
  return symbolGroups.map(group => ({
    docId: `lens_${group.id}`,
    score: group.relevance_score * group.topic_weight, // Combined scoring
    text: formatSymbolGroupAsText(group),
    kind: 'code_symbol',
    metadata: {
      is_lens_result: true,
      symbol_group: group
    }
  }));
}

/**
 * Format a SymbolGroup as readable text for context
 */
function formatSymbolGroupAsText(group: SymbolGroup): string {
  const parts: string[] = [];
  
  // Header with symbol information
  parts.push(`// Symbol: ${group.primary_symbol} (${group.language})`);
  parts.push(`// File: ${group.file_path}`);
  parts.push('');
  
  // Definition
  if (group.definition) {
    parts.push('// Definition:');
    parts.push(group.definition.content);
    parts.push('');
  }
  
  // Key references (limit to 2-3 most important)
  if (group.references.length > 0) {
    parts.push('// Key References:');
    const topReferences = group.references
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 3);
    
    for (const ref of topReferences) {
      parts.push(`// ${ref.file_path}:${ref.start_line}`);
      parts.push(ref.content);
      parts.push('');
    }
  }
  
  // Implementations (if different from definition)
  if (group.implementations.length > 0) {
    parts.push('// Implementations:');
    const topImpls = group.implementations
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 2);
    
    for (const impl of topImpls) {
      parts.push(`// ${impl.file_path}:${impl.start_line}`);
      parts.push(impl.content);
      parts.push('');
    }
  }
  
  // Test hints if available
  if (group.test_hints && group.test_hints.length > 0) {
    parts.push('// Test Examples:');
    const topTests = group.test_hints
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 2);
    
    for (const test of topTests) {
      parts.push(`// ${test.file_path}:${test.start_line}`);
      parts.push(test.content);
      parts.push('');
    }
  }
  
  return parts.join('\n');
}

/**
 * Test function for CLI diagnostics
 */
export async function testLensIntegration(db?: any): Promise<{
  available: boolean;
  latency_ms?: number;
  search_test?: boolean;
  code_intent_test?: boolean;
  cost_analysis_test?: boolean;
  error?: string;
}> {
  try {
    const lensService = await getLensService(db);
    
    // Test basic availability
    const connectionTest = await lensService.testConnection();
    if (!connectionTest.available) {
      return { 
        available: false, 
        latency_ms: connectionTest.latency_ms,
        error: connectionTest.error || 'Lens service not available'
      };
    }

    // Test code intent detection
    const codeIntentTest = detectCodeIntent('fix error in calculateBM25 function', ['index.ts'], 'code');
    const codeIntentPassed = codeIntentTest.is_code_intent && codeIntentTest.confidence > 0.5;

    // Test search (with timeout)
    let searchTest = false;
    try {
      const searchResult = await lensService.search({
        query: 'test search',
        max_groups: 5,
        timeout_ms: 1000
      });
      searchTest = searchResult.symbol_groups !== undefined;
    } catch (error) {
      console.debug(`Lens search test failed: ${error}`);
    }

    // Test cost analysis
    const mockSymbolGroups: SymbolGroup[] = [{
      id: 'test_1',
      primary_symbol: 'testFunction',
      language: 'typescript',
      file_path: 'test.ts',
      definition: {
        id: 'def_1',
        content: 'function testFunction() {}',
        file_path: 'test.ts',
        start_line: 1,
        end_line: 1,
        start_char: 0,
        end_char: 25,
        atom_type: 'definition',
        symbol_name: 'testFunction',
        tokens: 10,
        importance: 1.0
      },
      references: [],
      implementations: [],
      estimated_tokens: 50,
      relevance_score: 0.8,
      topic_weight: 0.3,
      is_precise_match: true
    }];

    const costAnalysis = calculateLagrangianCost(
      mockSymbolGroups,
      DEFAULT_LENS_CONFIG,
      1000, // current tokens
      4000, // total budget
      100   // estimated latency
    );

    const costAnalysisTest = costAnalysis.total_cost > 0 && costAnalysis.sla_constraint_met;

    return {
      available: true,
      latency_ms: connectionTest.latency_ms,
      search_test: searchTest,
      code_intent_test: codeIntentPassed,
      cost_analysis_test: costAnalysisTest
    };

  } catch (error: any) {
    return {
      available: false,
      error: error?.message || String(error)
    };
  }
}