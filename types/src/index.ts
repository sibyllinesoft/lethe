// Core Lethe types exported from the core package
export interface Config {
  models: {
    embed: string;
    rerank: string;
    hyde?: string;
    summarize?: string;
    llm_rerank?: string;
  };
  retrieval: {
    alpha: number;
    beta: number;
    gamma_kind_boost: {
      tool_result: number;
      user_code: number;
      prose: number;
      code: number;
    };
    variant: 'window' | 'bm25' | 'vector' | 'hybrid';
    window_size?: number;
  };
  chunking: {
    target_tokens: number;
    overlap: number;
    split_code_blocks: boolean;
    split_sentences: boolean;
    strategy: 'basic' | 'ast' | 'hierarchical' | 'propositional';
    ast_max_depth?: number;
    hierarchical_levels?: number;
  };
  rerank: {
    topk_in: number;
    topk_out: number;
    batch_size: number;
    use_llm: boolean;
    llm_batch_size?: number;
  };
  diversify: {
    pack_chunks: number;
    method: 'entity' | 'semantic';
    semantic_threshold?: number;
    entity_boost?: number;
  };
  plan: {
    explore: { hyde_k: number; granularity: string; beta: number };
    verify: { hyde_k: number; granularity: string; beta: number };
    exploit: { hyde_k: number; granularity: string; beta: number };
    query_rewrite: boolean;
    decompose: boolean;
  };
  fusion: {
    dynamic: boolean;
    weights: number[];
  };
  contradiction: {
    enabled: boolean;
    threshold: number;
  };
  performance: {
    budget_parity: boolean;
    max_latency_p50: number;
    max_latency_p95: number;
    max_memory_rss: number;
  };
  telemetry: {
    enabled: boolean;
    log_format: 'json' | 'jsonl';
    include_config_hash: boolean;
    include_timings: boolean;
    include_memory: boolean;
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

// LLM Analyzer specific types
export interface LLMCall {
  id: string;
  timestamp: number;
  model: string;
  prompt: string;
  completion: string;
  metadata?: Record<string, unknown>;
}

export interface CallPair {
  before: LLMCall;
  after: LLMCall;
  diff?: string;
}

export interface ProxyLogEntry {
  timestamp: number;
  method: string;
  url: string;
  headers: Record<string, string>;
  body?: string;
  response?: {
    status: number;
    headers: Record<string, string>;
    body?: string;
  };
}

// Performance and telemetry types
export interface PerformanceMetrics {
  latency_p50: number;
  latency_p95: number;
  memory_rss_mb: number;
  cpu_usage_percent: number;
  params_count?: number;
  flops_count?: number;
  timestamp: number;
}

export interface TelemetryEvent {
  event_type: 'retrieval' | 'rerank' | 'diversify' | 'chunk' | 'orchestrate';
  session_id: string;
  config_hash: string;
  performance: PerformanceMetrics;
  metadata?: Record<string, unknown>;
  timestamp: number;
}
