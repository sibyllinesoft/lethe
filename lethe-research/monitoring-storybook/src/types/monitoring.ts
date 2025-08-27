/**
 * TypeScript interfaces matching the Python monitoring system data structures
 */

export interface PromptExecution {
  // Unique identifiers
  execution_id: string;
  prompt_id: string;
  prompt_version: string;
  conversation_id?: string;

  // Prompt content and metadata
  prompt_text: string;
  prompt_hash: string;
  prompt_template?: string;
  prompt_variables: Record<string, any>;

  // Model configuration
  model_name: string;
  model_version: string;
  model_parameters: Record<string, any>;
  temperature: number;
  max_tokens?: number;

  // Execution context
  timestamp: string;
  environment: Record<string, string>;
  git_commit?: string;

  // Input metrics
  context_length: number;
  conversation_turn: number;

  // Response metrics
  response_text: string;
  response_length: number;
  response_tokens?: number;

  // Performance metrics
  execution_time_ms: number;
  tokens_per_second?: number;
  memory_usage_mb: number;

  // Quality metrics
  response_quality_score?: number;
  coherence_score?: number;
  relevance_score?: number;

  // Error tracking
  error_occurred: boolean;
  error_message?: string;
  error_type?: string;

  // Comparison and A/B testing
  baseline_execution_id?: string;
  ab_test_group?: string;
  experiment_tag?: string;
}

export interface PromptComparison {
  comparison_id: string;
  baseline_execution_id: string;
  treatment_execution_id: string;
  comparison_timestamp: string;
  notes?: string;
  
  // Computed metrics
  quality_improvement?: number;
  performance_change_percent?: number;
  length_change_percent?: number;
  is_significant: boolean;
  p_value?: number;
  
  // Metadata
  comparison_type: 'manual' | 'automated' | 'ab_test';
  tags: string[];
}

export interface SummaryStats {
  total_executions: number;
  unique_prompts: number;
  success_rate: number;
  avg_execution_time_ms: number;
  recent_executions_24h: number;
}

export interface TimelineDataPoint {
  date: string;
  total_executions: number;
  avg_execution_time: number;
  avg_response_length: number;
  errors: number;
}

export interface PromptPerformance {
  prompt_id: string;
  execution_count: number;
  avg_execution_time: number;
  avg_response_length: number;
  avg_quality_score?: number;
  last_used: string;
  error_count: number;
  success_rate: number;
}

export interface ModelComparison {
  model_name: string;
  execution_count: number;
  avg_execution_time: number;
  avg_response_length: number;
  avg_quality_score?: number;
  error_count: number;
}

export interface ExecutionComparison {
  current_execution: PromptExecution;
  similar_executions: Partial<PromptExecution>[];
  changes_detected: string[];
}

export interface PromptAnalytics {
  total_executions: number;
  success_rate: number;
  avg_execution_time_ms: number;
  avg_response_length: number;
  memory_usage_avg: number;
  performance_trend: 'improving' | 'stable' | 'degrading';
  quality_trend?: 'improving' | 'stable' | 'degrading';
  latest_execution: string;
}

// UI Component Props
export interface DashboardCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  trend?: 'up' | 'down' | 'stable';
  trendValue?: string;
  icon?: React.ReactNode;
  className?: string;
}

export interface ChartDataPoint {
  [key: string]: string | number;
}

export interface ChartProps {
  data: ChartDataPoint[];
  width?: number;
  height?: number;
  className?: string;
}

// CLI Output representation
export interface CLIOutput {
  command: string;
  timestamp: string;
  output: string[];
  status: 'success' | 'error' | 'warning';
  duration_ms?: number;
}

// Filter and sorting options
export interface TableFilters {
  search?: string;
  dateRange?: {
    start: string;
    end: string;
  };
  models?: string[];
  status?: ('success' | 'error')[];
  qualityRange?: {
    min: number;
    max: number;
  };
}

export interface SortConfig {
  key: string;
  direction: 'asc' | 'desc';
}