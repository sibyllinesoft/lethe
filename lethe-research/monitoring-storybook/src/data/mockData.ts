import { 
  PromptExecution, 
  PromptComparison, 
  SummaryStats,
  TimelineDataPoint,
  PromptPerformance,
  ModelComparison,
  ExecutionComparison,
  PromptAnalytics,
  CLIOutput
} from '../types/monitoring';
import { subDays, format, addHours } from 'date-fns';

/**
 * Mock data generator for Lethe Prompt Monitoring System
 * Based on actual data structures from the Python implementation
 */

const MODELS = [
  'gpt-4-turbo-preview',
  'gpt-4',
  'gpt-3.5-turbo',
  'claude-3-opus-20240229',
  'claude-3-sonnet-20240229',
  'claude-3-haiku-20240307',
  'gemini-pro'
];

const PROMPT_TEMPLATES = [
  'code_generation',
  'text_analysis',
  'creative_writing',
  'question_answering',
  'summarization',
  'translation',
  'code_review'
];

const ERROR_TYPES = [
  'timeout',
  'rate_limit',
  'token_limit_exceeded',
  'model_unavailable',
  'invalid_input',
  'network_error'
];

function randomChoice<T>(array: T[]): T {
  return array[Math.floor(Math.random() * array.length)];
}

function randomFloat(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function generateExecutionId(): string {
  return `exec_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

function generatePromptId(): string {
  const template = randomChoice(PROMPT_TEMPLATES);
  return `${template}_${Math.random().toString(36).substr(2, 6)}`;
}

export function generateMockPromptExecution(overrides?: Partial<PromptExecution>): PromptExecution {
  const hasError = Math.random() < 0.05; // 5% error rate
  const model = randomChoice(MODELS);
  const promptId = generatePromptId();
  const executionTime = randomFloat(50, 5000);
  const responseLength = randomInt(100, 2000);
  
  const baseExecution: PromptExecution = {
    execution_id: generateExecutionId(),
    prompt_id: promptId,
    prompt_version: `${randomInt(1, 3)}.${randomInt(0, 9)}.${randomInt(0, 9)}`,
    conversation_id: Math.random() > 0.3 ? `conv_${Math.random().toString(36).substr(2, 8)}` : undefined,
    
    prompt_text: `Generate a ${randomChoice(['function', 'class', 'component', 'algorithm'])} that ${randomChoice(['processes data', 'handles user input', 'manages state', 'optimizes performance'])} for ${randomChoice(['web applications', 'mobile apps', 'data analysis', 'machine learning'])}.`,
    prompt_hash: Math.random().toString(36).substr(2, 16),
    prompt_template: randomChoice(PROMPT_TEMPLATES),
    prompt_variables: {
      context: randomChoice(['production', 'development', 'testing']),
      user_type: randomChoice(['admin', 'user', 'guest']),
      complexity: randomChoice(['simple', 'medium', 'complex'])
    },
    
    model_name: model,
    model_version: model.includes('gpt') ? '2024-03-01' : '20240229',
    model_parameters: {
      temperature: randomFloat(0, 1.2),
      max_tokens: randomChoice([1000, 2000, 4000, 8000]),
      top_p: randomFloat(0.8, 1.0)
    },
    temperature: randomFloat(0, 1.2),
    max_tokens: randomChoice([1000, 2000, 4000, 8000]),
    
    timestamp: subDays(new Date(), randomInt(0, 30)).toISOString(),
    environment: {
      platform: randomChoice(['linux', 'darwin', 'win32']),
      python_version: randomChoice(['3.9.0', '3.10.0', '3.11.0']),
      node_version: randomChoice(['18.17.0', '20.5.0']),
      environment_type: randomChoice(['development', 'staging', 'production'])
    },
    git_commit: `${Math.random().toString(36).substr(2, 7)}`,
    
    context_length: randomInt(100, 4000),
    conversation_turn: randomInt(1, 10),
    
    response_text: hasError ? '' : `Generated response for ${promptId}. This is a comprehensive solution that addresses the requirements with proper error handling, optimization, and best practices. The implementation follows modern standards and includes comprehensive testing.`,
    response_length: hasError ? 0 : responseLength,
    response_tokens: hasError ? 0 : Math.ceil(responseLength / 4),
    
    execution_time_ms: executionTime,
    tokens_per_second: hasError ? 0 : responseLength / 4 / (executionTime / 1000),
    memory_usage_mb: randomFloat(50, 500),
    
    response_quality_score: hasError ? 0 : randomFloat(0.6, 1.0),
    coherence_score: hasError ? 0 : randomFloat(0.7, 1.0),
    relevance_score: hasError ? 0 : randomFloat(0.6, 0.95),
    
    error_occurred: hasError,
    error_message: hasError ? `${randomChoice(ERROR_TYPES)}: Sample error message` : undefined,
    error_type: hasError ? randomChoice(ERROR_TYPES) : undefined,
    
    baseline_execution_id: Math.random() > 0.7 ? generateExecutionId() : undefined,
    ab_test_group: Math.random() > 0.8 ? randomChoice(['A', 'B', 'control']) : undefined,
    experiment_tag: Math.random() > 0.9 ? `experiment_${randomInt(1, 5)}` : undefined,
  };

  return { ...baseExecution, ...overrides };
}

export function generateMockExecutions(count: number): PromptExecution[] {
  return Array.from({ length: count }, () => generateMockPromptExecution());
}

export function generateMockSummaryStats(): SummaryStats {
  return {
    total_executions: randomInt(1000, 50000),
    unique_prompts: randomInt(50, 500),
    success_rate: randomFloat(92, 99),
    avg_execution_time_ms: randomFloat(200, 1500),
    recent_executions_24h: randomInt(50, 500)
  };
}

export function generateMockTimelineData(days: number = 7): TimelineDataPoint[] {
  return Array.from({ length: days }, (_, i) => {
    const date = subDays(new Date(), days - i - 1);
    return {
      date: format(date, 'yyyy-MM-dd'),
      total_executions: randomInt(20, 200),
      avg_execution_time: randomFloat(150, 800),
      avg_response_length: randomFloat(300, 1200),
      errors: randomInt(0, 10)
    };
  });
}

export function generateMockPromptPerformance(count: number = 10): PromptPerformance[] {
  return Array.from({ length: count }, () => ({
    prompt_id: generatePromptId(),
    execution_count: randomInt(5, 100),
    avg_execution_time: randomFloat(100, 2000),
    avg_response_length: randomFloat(200, 1500),
    avg_quality_score: randomFloat(0.7, 0.95),
    last_used: subDays(new Date(), randomInt(0, 30)).toISOString(),
    error_count: randomInt(0, 5),
    success_rate: randomFloat(90, 100)
  }));
}

export function generateMockModelComparison(): ModelComparison[] {
  return MODELS.map(model => ({
    model_name: model,
    execution_count: randomInt(10, 1000),
    avg_execution_time: randomFloat(100, 3000),
    avg_response_length: randomFloat(300, 1800),
    avg_quality_score: randomFloat(0.6, 0.95),
    error_count: randomInt(0, 20)
  }));
}

export function generateMockExecutionComparison(): ExecutionComparison {
  const currentExecution = generateMockPromptExecution();
  const similarExecutions = Array.from({ length: 3 }, () => 
    generateMockPromptExecution({ 
      prompt_id: currentExecution.prompt_id,
      execution_id: generateExecutionId()
    })
  );

  return {
    current_execution: currentExecution,
    similar_executions: similarExecutions.map(exec => ({
      execution_id: exec.execution_id,
      prompt_version: exec.prompt_version,
      timestamp: exec.timestamp,
      execution_time_ms: exec.execution_time_ms,
      response_length: exec.response_length,
      response_quality_score: exec.response_quality_score,
      prompt_hash: exec.prompt_hash
    })),
    changes_detected: [
      'Prompt content changed',
      'Execution time improved by +15.2%',
      'Quality score increased by +0.12'
    ]
  };
}

export function generateMockPromptAnalytics(): PromptAnalytics {
  return {
    total_executions: randomInt(50, 500),
    success_rate: randomFloat(88, 99),
    avg_execution_time_ms: randomFloat(200, 1200),
    avg_response_length: randomFloat(400, 1600),
    memory_usage_avg: randomFloat(100, 400),
    performance_trend: randomChoice(['improving', 'stable', 'degrading']),
    quality_trend: randomChoice(['improving', 'stable', 'degrading']),
    latest_execution: subDays(new Date(), randomInt(0, 7)).toISOString()
  };
}

export function generateMockCLIOutput(command: string): CLIOutput {
  const outputs: Record<string, string[]> = {
    'prompt-monitor status': [
      '🔍 Lethe Prompt Monitoring Status',
      '==================================================',
      '📊 Total Executions: 15,432',
      '🎯 Unique Prompts: 127',
      '✅ Success Rate: 96.8%',
      '⚡ Avg Response Time: 485ms',
      '🕐 Recent Activity (24h): 89 executions',
      '',
      '💾 Database: experiments/prompt_tracking.db',
      '📁 Database Size: 2.3 MB'
    ],
    'prompt-monitor list-prompts': [
      '📋 Tracked Prompts',
      '================================================================================',
      'Prompt ID                      Executions   Avg Time     Success Rate Last Used          ',
      '--------------------------------------------------------------------------------',
      'code_generation_abc123         45           324ms        98.5%        2024-01-15T14:30:00',
      'text_analysis_def456           32           567ms        95.2%        2024-01-15T12:15:00',
      'creative_writing_ghi789        28           1,245ms      97.1%        2024-01-14T16:45:00',
      'question_answering_jkl012      67           234ms        99.1%        2024-01-15T13:20:00'
    ],
    'prompt-monitor analyze code_generation_abc123': [
      '🔍 Analyzing Prompt: code_generation_abc123',
      '============================================================',
      '📊 Total Executions: 45',
      '✅ Success Rate: 98.5%',
      '⚡ Average Response Time: 324.2ms',
      '📝 Average Response Length: 856 chars',
      '💾 Average Memory Usage: 125.3 MB',
      '📈 Performance Trend: improving',
      '🎯 Quality Trend: stable',
      '🕐 Latest Execution: 2024-01-15T14:30:15'
    ]
  };

  return {
    command,
    timestamp: new Date().toISOString(),
    output: outputs[command] || ['Command output not available'],
    status: Math.random() > 0.1 ? 'success' : 'error',
    duration_ms: randomInt(50, 500)
  };
}

// Comprehensive mock dataset for stories
export const mockDataset = {
  summaryStats: generateMockSummaryStats(),
  timelineData: generateMockTimelineData(14), // 2 weeks of data
  promptPerformance: generateMockPromptPerformance(20),
  modelComparison: generateMockModelComparison(),
  recentExecutions: generateMockExecutions(10),
  executionComparison: generateMockExecutionComparison(),
  promptAnalytics: generateMockPromptAnalytics(),
  cliOutputs: {
    status: generateMockCLIOutput('prompt-monitor status'),
    list: generateMockCLIOutput('prompt-monitor list-prompts'),
    analyze: generateMockCLIOutput('prompt-monitor analyze code_generation_abc123')
  }
};