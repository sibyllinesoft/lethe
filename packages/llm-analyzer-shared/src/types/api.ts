import { LLMCall, LLMCallSummary, CallFilters, CallMetrics } from './call';

// API Response wrapper
export interface ApiResponse<T = unknown> {
  success: boolean;
  data: T;
  error?: string;
  pagination?: PaginationInfo;
}

export interface PaginationInfo {
  page: number;
  limit: number;
  total: number;
  totalPages: number;
}

// API Request/Response types
export interface GetCallsRequest extends CallFilters {
  page?: number;
  limit?: number;
  sortBy?: 'timestamp' | 'duration' | 'tokens' | 'cost';
  sortOrder?: 'asc' | 'desc';
}

export interface GetCallsResponse {
  calls: LLMCallSummary[];
  pagination: PaginationInfo;
  metrics: CallMetrics;
}

export interface GetCallResponse {
  call: LLMCall;
}

export interface CompareCallsRequest {
  callIds: string[];
}

export interface CompareCallsResponse {
  calls: LLMCall[];
  comparison: CallComparison;
}

export interface CallComparison {
  requestDiff?: DiffResult;
  responseDiff?: DiffResult;
  metricsDiff: MetricComparison;
}

export interface DiffResult {
  added: string[];
  removed: string[];
  modified: Array<{
    path: string;
    oldValue: unknown;
    newValue: unknown;
  }>;
}

export interface MetricComparison {
  duration: MetricDiff;
  inputTokens?: MetricDiff;
  outputTokens?: MetricDiff;
  totalTokens?: MetricDiff;
  cost?: MetricDiff;
}

export interface MetricDiff {
  values: number[];
  difference: number;
  percentageChange: number;
}

// Ingestion types
export interface IngestNDJSONRequest {
  data: string; // NDJSON string
}

export interface IngestNDJSONResponse {
  processed: number;
  errors: Array<{
    line: number;
    error: string;
  }>;
}