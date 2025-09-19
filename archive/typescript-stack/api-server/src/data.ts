import { randomUUID } from 'crypto';
import type {
  AnalyzerStats,
  CallPair,
  CallsFilters,
  CallsListResponse,
  DiffResult,
  RunComparison,
} from '@lethe/types';

const NOW = Date.now();

function buildCall(overrides: Partial<CallPair> = {}): CallPair {
  const id = overrides.id ?? randomUUID();
  const runId = overrides.run_id ?? 'run-2024-09-01';
  const queryId = overrides.query_id ?? `query-${id.slice(0, 6)}`;
  const timestamp = overrides.timestamp ?? new Date(NOW - Math.random() * 1000 * 60 * 60).toISOString();
  const provider = overrides.provider ?? 'openai';
  const model = overrides.model ?? 'gpt-4o';
  const status = overrides.status ?? 'success';
  const prompt = overrides.prompt ?? 'Summarize the latest deployment results.';
  const completion = overrides.completion ?? 'Deployment succeeded with minor delays.';

  return {
    id,
    timestamp,
    run_id: runId,
    query_id: queryId,
    provider,
    model,
    status,
    benchmark_type: overrides.benchmark_type ?? 'regression',
    dataset: overrides.dataset ?? 'internal-suite',
    input_tokens: overrides.input_tokens ?? 320,
    output_tokens: overrides.output_tokens ?? 95,
    latency_ms: overrides.latency_ms ?? Math.round(250 + Math.random() * 120),
    temperature: overrides.temperature ?? 0,
    max_tokens: overrides.max_tokens ?? 512,
    transform_changes: overrides.transform_changes ?? ['hydrate_context', 'limit_tokens'],
    prompt,
    completion,
    pre_context: overrides.pre_context ?? ['Previous answer about monitoring', 'Alert escalation steps'],
    post_context: overrides.post_context ?? ['Updated summary with metrics', 'Next steps for canary'],
    request: overrides.request ?? {
      event: 'proxy_request_transform',
      timestamp,
      level: 'info',
      provider,
      path: '/v1/chat/completions',
      method: 'POST',
      request_id: `req-${id}`,
      transform: {
        enabled: true,
        duration_ms: 12,
        changes: ['rewrite_prompt', 'attach_context'],
        size_change_percent: -12,
      },
      pre_transform: {
        size_bytes: 1024,
        token_estimate: 350,
        payload: {
          model,
          messages: [
            { role: 'system', content: 'You are an assistant that summarizes incidents.' },
            { role: 'user', content: prompt },
          ],
          temperature: 0,
          max_tokens: 512,
        },
      },
      post_transform: {
        size_bytes: 860,
        token_estimate: 280,
        payload: {
          model,
          messages: [
            { role: 'system', content: 'You are an assistant that summarizes incidents.' },
            { role: 'user', content: `${prompt} Include deployment metrics.` },
          ],
          temperature: 0,
          max_tokens: 512,
        },
      },
      benchmark_metadata: {
        run_id: runId,
        query_id: queryId,
        provider,
        benchmark_type: 'regression',
        dataset: 'internal-suite',
      },
    },
    response: overrides.response ?? {
      event: 'proxy_response',
      timestamp,
      level: 'info',
      provider,
      request_id: `req-${id}`,
      status_code: 200,
      response_size_bytes: 2048,
      performance: {
        transform_duration_ms: 5,
        total_request_duration_ms: 280,
        response_tokens: 95,
        response_time_ms: 280,
      },
    },
    metadata: overrides.metadata ?? {},
  };
}

const DATASET: CallPair[] = [
  buildCall(),
  buildCall({
    provider: 'anthropic',
    model: 'claude-3.5-sonnet',
    run_id: 'run-2024-09-02',
    status: 'success',
    latency_ms: 320,
    transform_changes: ['rewrite_prompt'],
  }),
  buildCall({
    provider: 'openai',
    model: 'gpt-4o',
    run_id: 'run-2024-09-02',
    status: 'error',
    latency_ms: 480,
    transform_changes: ['hydrate_context', 'truncate_context'],
  }),
  buildCall({
    provider: 'openrouter',
    model: 'mistral-large',
    run_id: 'run-2024-09-03',
    status: 'pending',
    completion: undefined,
  }),
];

export function listCalls(filters: CallsFilters = {}): CallsListResponse {
  let calls = [...DATASET];

  if (filters.run_id) {
    calls = calls.filter((call) => call.run_id === filters.run_id);
  }
  if (filters.provider) {
    calls = calls.filter((call) => call.provider === filters.provider);
  }
  if (filters.model) {
    calls = calls.filter((call) => call.model === filters.model);
  }
  if (filters.status) {
    calls = calls.filter((call) => call.status === filters.status);
  }
  if (filters.benchmark_type) {
    calls = calls.filter((call) => call.benchmark_type === filters.benchmark_type);
  }
  if (filters.dataset) {
    calls = calls.filter((call) => call.dataset === filters.dataset);
  }

  const page = filters.page ?? 1;
  const limit = filters.limit ?? 100;
  const offset = (page - 1) * limit;

  return {
    calls: calls.slice(offset, offset + limit),
    total: calls.length,
    page,
    limit,
  };
}

export function getCall(id: string): CallPair | undefined {
  return DATASET.find((call) => call.id === id);
}

export function getStats(): AnalyzerStats {
  const providers = new Set(DATASET.map((call) => call.provider));
  const models = new Set(DATASET.map((call) => call.model));
  const latencies = DATASET.map((call) => call.latency_ms);

  return {
    total_calls: DATASET.length,
    providers: Array.from(providers),
    models: Array.from(models),
    average_latency_ms: Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length),
  };
}

export function getRuns(): string[] {
  return Array.from(new Set(DATASET.map((call) => call.run_id)));
}

export function getRunComparison(runId: string): RunComparison | undefined {
  const callIds = DATASET.filter((call) => call.run_id === runId).map((call) => call.id);
  return callIds.length ? { run_id: runId, call_ids: callIds } : undefined;
}

export function diffCalls(callA: CallPair, callB: CallPair): DiffResult {
  const toSegments = (before: string, after: string) => {
    if (before === after) {
      return [{ value: before }];
    }
    return [
      { value: before, removed: true },
      { value: after, added: true },
    ];
  };

  return {
    prompt_diff: toSegments(callA.prompt, callB.prompt),
    output_diff: toSegments(callA.completion ?? '', callB.completion ?? ''),
    params_diff: {
      temperature: { before: callA.temperature, after: callB.temperature },
      max_tokens: { before: callA.max_tokens, after: callB.max_tokens },
    },
  };
}
