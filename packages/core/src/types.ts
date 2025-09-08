export interface Config {
  models: {
    embed: string;
    rerank: string;
    hyde?: string;
    summarize?: string;
    llm_rerank?: string; // For LLM-based reranking
  };
  retrieval: {
    alpha: number;          // BM25 weight (0.0-2.0)
    beta: number;           // Vector weight (0.0-2.0) 
    gamma_kind_boost: {     // Kind-specific score boosts
      tool_result: number;
      user_code: number;
      prose: number;
      code: number;
    };
    variant: 'window' | 'bm25' | 'vector' | 'hybrid'; // Retrieval algorithm variant
    window_size?: number;   // For window-based retrieval
  };
  chunking: {
    target_tokens: number;
    overlap: number;
    split_code_blocks: boolean;
    split_sentences: boolean;
    strategy: 'basic' | 'ast' | 'hierarchical' | 'propositional'; // Chunking strategy
    ast_max_depth?: number; // For AST chunking
    hierarchical_levels?: number; // For hierarchical chunking
  };
  rerank: {
    topk_in: number;
    topk_out: number;
    batch_size: number;
    use_llm: boolean;       // Enable LLM reranking
    llm_batch_size?: number; // Batch size for LLM reranking
  };
  diversify: {
    pack_chunks: number;
    method: 'entity' | 'semantic'; // Diversification method
    semantic_threshold?: number;    // Semantic similarity threshold
    entity_boost?: number;          // Entity coverage boost factor
  };
  plan: {
    explore: { hyde_k: number; granularity: string; beta: number };
    verify: { hyde_k: number; granularity: string; beta: number };
    exploit: { hyde_k: number; granularity: string; beta: number };
    query_rewrite: boolean;   // Enable query rewrite/decompose
    decompose: boolean;       // Enable query decomposition
  };
  fusion: {
    dynamic: boolean;         // Enable dynamic fusion
    weights: number[];        // Fusion weights for multiple queries
  };
  contradiction: {
    enabled: boolean;         // Enable contradiction detection
    threshold: number;        // Contradiction confidence threshold
  };
  performance: {
    budget_parity: boolean;   // Maintain budget parity (±5% params/FLOPs)
    max_latency_p50: number;  // Maximum p50 latency in seconds
    max_latency_p95: number;  // Maximum p95 latency in seconds
    max_memory_rss: number;   // Maximum RSS memory in GB
  };
  telemetry: {
    enabled: boolean;         // Enable telemetry logging
    log_format: 'json' | 'jsonl'; // Log format
    include_config_hash: boolean;  // Include config hash in logs
    include_timings: boolean;      // Include timing metrics
    include_memory: boolean;       // Include memory metrics
  };
}

export interface Candidate {
  id: string;
  text: string;
  messageId: string;
  kind: 'prose' | 'code' | 'tool_result' | 'user_code';
  bm25Score: number;
  vectorScore: number;
  hybridScore: number;
  rerankScore?: number;
}

export interface HydeResult {
  queries: string[];
  pseudo: string;
}

export interface ContextPack {
  id: string;
  sessionId: string;
  query: string;
  summary: string;
  keyEntities: string[];
  claims: Array<{ text: string; chunks: string[] }>;
  contradictions: Array<{ issue: string; chunks: string[] }>;
  citations: Record<string, { messageId: string; span: [number, number] }>;
  debug?: {
    hydeQueries: string[];
    candidateCount: number;
    rerankTime: number;
    totalTime: number;
  };
}

export type PlanType = 'explore' | 'verify' | 'exploit';
// Result types for comprehensive error handling
export type Result<T, E = Error> = 
  | { success: true; data: T }
  | { success: false; error: E };

export interface LetheError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
  timestamp: number;
  stack?: string;
}

// Enhanced candidate type with additional metadata
export interface EnhancedCandidate extends Candidate {
  metadata?: {
    entities: string[];
    semantic_cluster?: number;
    contradiction_score?: number;
    citation_spans: Array<{ start: number; end: number }>;
  };
  diversityScore?: number;
  rerankScore?: number;
}

// Performance metrics
export interface PerformanceMetrics {
  latency_p50: number;
  latency_p95: number;
  memory_rss_mb: number;
  cpu_usage_percent: number;
  params_count?: number;
  flops_count?: number;
  timestamp: number;
}

// Telemetry data
export interface TelemetryEvent {
  event_type: 'retrieval' | 'rerank' | 'diversify' | 'chunk' | 'orchestrate';
  session_id: string;
  config_hash: string;
  performance: PerformanceMetrics;
  metadata?: Record<string, unknown>;
  timestamp: number;
}

// Configuration validation schema types
export interface ConfigValidationError {
  field: string;
  message: string;
  expected?: string;
  actual?: unknown;
}

// Query processing types
export interface QueryDecomposition {
  original: string;
  subqueries: string[];
  rewritten?: string;
  strategy: 'none' | 'rewrite' | 'decompose' | 'both';
}

// Ranking and diversification types
export interface RankingContext {
  query: string;
  candidates: EnhancedCandidate[];
  user_context?: Record<string, unknown>;
  semantic_clusters?: number[][];
}

export interface DiversificationResult {
  selected: EnhancedCandidate[];
  coverage_score: number;
  diversity_score: number;
  total_entities: number;
  unique_entities: number;
}

// Chunking strategy types
export interface ChunkingMetadata {
  ast_node_type?: string;
  hierarchical_level?: number;
  propositional_relations?: string[];
  semantic_boundary_score?: number;
}

export interface EnhancedChunk extends Chunk {
  metadata?: ChunkingMetadata;
  quality_score?: number;
}

// LLM integration types
export interface LLMRerankingRequest {
  query: string;
  candidates: EnhancedCandidate[];
  context?: string;
  max_tokens?: number;
}

export interface LLMRerankingResponse {
  rankings: Array<{ id: string; score: number; explanation?: string }>;
  contradictions?: Array<{ id: string; contradiction: string; confidence: number }>;
  processing_time_ms: number;
}

// Contradiction detection types
export interface ContradictionPair {
  chunk_id_1: string;
  chunk_id_2: string;
  contradiction_text: string;
  confidence: number;
  semantic_similarity: number;
}

// JSON Schema validation types
export interface ValidationResult<T> {
  valid: boolean;
  data?: T;
  errors: ConfigValidationError[];
}

// Performance budget tracking
export interface BudgetTracker {
  params_baseline: number;
  flops_baseline: number;
  current_params: number;
  current_flops: number;
  within_budget: boolean;
  variance_percent: number;
}

// Runtime configuration flags
export interface RuntimeFlags {
  enable_debug_logging: boolean;
  force_recompute_embeddings: boolean;
  skip_contradiction_detection: boolean;
  use_experimental_features: boolean;
  profiling_enabled: boolean;
}

// System health monitoring
export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'critical';
  last_check: number;
  services: {
    retrieval: 'up' | 'down' | 'degraded';
    reranking: 'up' | 'down' | 'degraded';  
    llm_integration: 'up' | 'down' | 'degraded';
    database: 'up' | 'down' | 'degraded';
  };
  performance_alerts: string[];
}
