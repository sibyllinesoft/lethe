// Shared types matching the React app
export interface ProxyLogEntry {
  timestamp: string;
  level: string;
  event: string;
  request_id: string;
  _line_number?: number;
}

export interface ProxyRequestTransform extends ProxyLogEntry {
  event: 'proxy_request_transform';
  benchmark_metadata: {
    run_id: string;
    query_id: string;
    provider: string;
    benchmark_type: string;
    dataset: string;
  };
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
      messages: Array<{
        role: string;
        content: string;
      }>;
      temperature: number;
      max_tokens: number;
      [key: string]: any;
    };
  };
  post_transform: {
    size_bytes: number;
    token_estimate: number;
    payload: {
      model: string;
      messages: Array<{
        role: string;
        content: string;
      }>;
      temperature: number;
      max_tokens: number;
      [key: string]: any;
    };
  };
  performance: {
    transform_duration_ms: number;
    total_request_duration_ms: number | null;
    pre_transform_size_bytes: number;
    post_transform_size_bytes: number;
    size_change_percent: number;
  };
}

export interface ProxyResponse extends ProxyLogEntry {
  event: 'proxy_response';
  provider: string;
  status_code: number;
  response_size_bytes: number;
  performance: {
    transform_duration_ms: number;
    total_request_duration_ms: number;
    response_tokens: number;
    response_time_ms: number;
  };
}

export interface CallPair {
  id: string;
  timestamp: string;
  run_id: string;
  query_id: string;
  provider: string;
  model: string;
  benchmark_type: string;
  dataset: string;
  request: ProxyRequestTransform;
  response?: ProxyResponse;
  pre_context: any[];
  post_context: any[];
  prompt: string;
  completion?: string;
  input_tokens: number;
  output_tokens: number;
  latency_ms: number;
  status: 'success' | 'error' | 'pending';
  temperature: number;
  max_tokens: number;
  transform_changes: string[];
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