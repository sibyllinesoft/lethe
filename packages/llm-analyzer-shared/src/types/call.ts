export interface LLMCall {
  id: string;
  timestamp: string;
  provider: string;
  model: string;
  endpoint: string;
  method: string;
  status: number;
  
  // Request data
  requestHeaders: Record<string, string>;
  requestBody: unknown;
  
  // Response data  
  responseHeaders: Record<string, string>;
  responseBody: unknown;
  
  // Metrics
  duration: number;
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  cost?: number;
  
  // Metadata
  userId?: string;
  sessionId?: string;
  tags: string[];
  
  // Error information
  error?: {
    type: string;
    message: string;
    stack?: string;
  };
}

export interface LLMCallSummary {
  id: string;
  timestamp: string;
  provider: string;
  model: string;
  status: number;
  duration: number;
  inputTokens?: number;
  outputTokens?: number;
  totalTokens?: number;
  cost?: number;
  error?: string;
  tags: string[];
}

export interface CallFilters {
  provider?: string[];
  model?: string[];
  status?: number[];
  dateFrom?: string;
  dateTo?: string;
  tags?: string[];
  hasError?: boolean;
  minDuration?: number;
  maxDuration?: number;
}

export interface CallMetrics {
  totalCalls: number;
  successfulCalls: number;
  errorCalls: number;
  averageDuration: number;
  totalTokens: number;
  totalCost: number;
  providersUsed: string[];
  modelsUsed: string[];
}