export interface BenchmarkRun {
  id: string;
  name: string;
  description?: string;
  createdAt: string;
  status: 'running' | 'completed' | 'failed';
  
  // Configuration
  config: BenchmarkConfig;
  
  // Results
  totalCalls: number;
  completedCalls: number;
  failedCalls: number;
  startTime?: string;
  endTime?: string;
  duration?: number;
  
  // Aggregated metrics
  metrics?: BenchmarkMetrics;
}

export interface BenchmarkConfig {
  providers: string[];
  models: string[];
  prompts: BenchmarkPrompt[];
  concurrency: number;
  iterations: number;
  timeout: number;
}

export interface BenchmarkPrompt {
  id: string;
  name: string;
  systemPrompt?: string;
  userPrompt: string;
  expectedOutput?: string;
  tags: string[];
}

export interface BenchmarkMetrics {
  averageDuration: number;
  medianDuration: number;
  p95Duration: number;
  p99Duration: number;
  
  averageTokens: number;
  totalTokens: number;
  
  averageCost: number;
  totalCost: number;
  
  successRate: number;
  errorRate: number;
  
  // Per-provider/model breakdowns
  providerMetrics: Record<string, ProviderMetrics>;
  modelMetrics: Record<string, ModelMetrics>;
}

export interface ProviderMetrics {
  calls: number;
  successRate: number;
  averageDuration: number;
  averageTokens: number;
  averageCost: number;
}

export interface ModelMetrics {
  calls: number;
  successRate: number;
  averageDuration: number;
  averageTokens: number;
  averageCost: number;
}

export interface CreateBenchmarkRequest {
  name: string;
  description?: string;
  config: BenchmarkConfig;
}

export interface CreateBenchmarkResponse {
  benchmarkId: string;
}

export interface GetBenchmarksResponse {
  benchmarks: BenchmarkRun[];
  pagination: import('./api').PaginationInfo;
}