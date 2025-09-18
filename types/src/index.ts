/**
 * Shared TypeScript types for the Lethe monorepo.
 * These definitions are intentionally lightweight and focused on
 * the feature-oriented architecture (core logic, CLI, API server, analyzer).
 */

// ---------------------------------------------------------------------------
// Utility types
// ---------------------------------------------------------------------------

export type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

export interface LetheError {
  code: string;
  message: string;
  timestamp: number;
  details?: Record<string, unknown>;
  stack?: string;
}

// ---------------------------------------------------------------------------
// Core domain types
// ---------------------------------------------------------------------------

export type MessageRole = 'user' | 'assistant' | 'system' | 'tool';

export interface SessionMessage {
  id: string;
  sessionId: string;
  role: MessageRole;
  text: string;
  timestamp: number;
  metadata?: Record<string, unknown>;
}

export interface SessionSummary {
  sessionId: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messageCount: number;
}

export interface RetrievalWeights {
  lexical: number;
  semantic: number;
  diversity: number;
}

export interface RetrievalConfig {
  topK: number;
  weights: RetrievalWeights;
  minRelevance: number;
}

export interface ChunkingConfig {
  maxTokens: number;
  overlap: number;
  splitCodeBlocks: boolean;
  splitSentences: boolean;
}

export interface SummarizationConfig {
  enabled: boolean;
  maxSummaryTokens: number;
}

export interface LetheConfig {
  retrieval: RetrievalConfig;
  chunking: ChunkingConfig;
  summarization: SummarizationConfig;
}

export interface RetrievalHighlight {
  start: number;
  end: number;
}

export interface RetrievalCandidate {
  message: SessionMessage;
  lexicalScore: number;
  semanticScore: number;
  diversityScore: number;
  hybridScore: number;
  highlights: RetrievalHighlight[];
}

export interface ContextPack {
  id: string;
  sessionId: string;
  query: string;
  summary: string;
  createdAt: number;
  messages: RetrievalCandidate[];
  metadata?: {
    generator: string;
    configHash?: string;
  };
}

// ---------------------------------------------------------------------------
// Analyzer domain types
// ---------------------------------------------------------------------------

export interface ProxyLogEntry {
  timestamp: string;
  level: string;
  event: string;
  request_id: string;
  metadata?: Record<string, unknown>;
}

export interface ProxyRequestTransform extends ProxyLogEntry {
  event: 'proxy_request_transform';
  provider: string;
  path: string;
  method: string;
  transform: {
    enabled: boolean;
    duration_ms: number;
    changes: string[];
    size_change_percent: number;
  };
  pre_transform: {
    size_bytes: number;
    token_estimate: number;
    payload: {
      model: string;
      messages: Array<{ role: string; content: string }>;
      temperature: number;
      max_tokens: number;
      [key: string]: unknown;
    };
  };
  post_transform: {
    size_bytes: number;
    token_estimate: number;
    payload: {
      model: string;
      messages: Array<{ role: string; content: string }>;
      temperature: number;
      max_tokens: number;
      [key: string]: unknown;
    };
  };
  benchmark_metadata?: {
    run_id: string;
    query_id: string;
    provider: string;
    benchmark_type: string;
    dataset: string;
    golden_answer?: string | string[];
  };
}

export interface ProxyResponse extends ProxyLogEntry {
  event: 'proxy_response';
  provider: string;
  status_code: number;
  response_size_bytes: number;
  performance: {
    transform_duration_ms: number | null;
    total_request_duration_ms: number | null;
    response_tokens: number | null;
    response_time_ms: number | null;
  };
}

export interface CallPair {
  id: string;
  timestamp: string;
  run_id: string;
  query_id: string;
  provider: string;
  model: string;
  status: 'success' | 'error' | 'pending';
  benchmark_type?: string;
  dataset?: string;
  input_tokens: number;
  output_tokens: number;
  latency_ms: number;
  temperature: number;
  max_tokens: number;
  transform_changes: string[];
  prompt: string;
  completion?: string;
  pre_context: string[];
  post_context: string[];
  request: ProxyRequestTransform;
  response?: ProxyResponse;
  metadata?: Record<string, unknown>;
}

export interface CallsListResponse {
  calls: CallPair[];
  total: number;
  page: number;
  limit: number;
}

export interface CallsFilters {
  since?: string;
  run_id?: string;
  provider?: string;
  model?: string;
  status?: string;
  benchmark_type?: string;
  dataset?: string;
  page?: number;
  limit?: number;
}

export interface CompareRequest {
  call_id_a: string;
  call_id_b: string;
}

export interface PrePostDiff {
  prompt: string;
  pre_context: string[];
  post_context: string[];
  size_diff?: {
    pre_bytes: number;
    post_bytes: number;
    change_bytes: number;
    change_percent: number;
  };
  transformations?: string[];
  payload_diff?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface DiffSegment {
  value: string;
  added?: boolean;
  removed?: boolean;
  similarity?: number;
  text?: string;
  pre?: string;
  post?: string;
  [key: string]: unknown;
}

export interface DiffResult {
  prompt_diff?: DiffSegment[];
  context_diff?: DiffSegment[];
  output_diff?: DiffSegment[];
  params_diff?: Record<string, { before: unknown; after: unknown }>;
  metadata?: Record<string, unknown>;
  performance_diff?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface AnalyzerStats {
  total_calls: number;
  providers: string[];
  models: string[];
  average_latency_ms: number;
}

export interface RunComparison {
  run_id: string;
  call_ids: string[];
}

// ---------------------------------------------------------------------------
// Tokenizer helper types
// ---------------------------------------------------------------------------

export interface TokenBreakdown {
  totalTokens: number;
  uniqueTokens: number;
  topTokens: Array<{ token: string; count: number }>;
}
