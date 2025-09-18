import { describe, expect, test } from 'bun:test';
import type {
  CallPair,
  ContextPack,
  RetrievalCandidate,
  SessionMessage,
} from '@lethe/types';

const message: SessionMessage = {
  id: 'm-1',
  sessionId: 's-1',
  role: 'assistant',
  text: 'Rollback completed successfully.',
  timestamp: Date.now(),
};

const candidate: RetrievalCandidate = {
  message,
  lexicalScore: 0.6,
  semanticScore: 0.7,
  diversityScore: 0.2,
  hybridScore: 0.58,
  highlights: [{ start: 0, end: 8 }],
};

describe('Shared types', () => {
  test('ContextPack shape matches expectations', () => {
    const pack: ContextPack = {
      id: 'pack-1',
      sessionId: 's-1',
      query: 'deployment status',
      summary: 'Deployment healthy',
      createdAt: Date.now(),
      messages: [candidate],
    };

    expect(pack.messages[0].message).toBe(message);
    expect(pack.messages[0].highlights[0].start).toBe(0);
  });

  test('CallPair supports analyzer fields', () => {
    const call: CallPair = {
      id: 'call-1',
      timestamp: new Date().toISOString(),
      run_id: 'run-1',
      query_id: 'query-1',
      provider: 'openai',
      model: 'gpt-4o',
      status: 'success',
      input_tokens: 120,
      output_tokens: 25,
      latency_ms: 240,
      temperature: 0,
      max_tokens: 256,
      transform_changes: ['hydrate_context'],
      prompt: 'Summarize deployment health.',
      completion: 'Deployment is healthy with low latency.',
      pre_context: ['last deploy status'],
      post_context: ['summary delivered'],
      request: {
        event: 'proxy_request_transform',
        timestamp: new Date().toISOString(),
        level: 'info',
        request_id: 'req-1',
        provider: 'openai',
        path: '/v1/chat/completions',
        method: 'POST',
        transform: {
          enabled: true,
          duration_ms: 10,
          changes: ['attach_context'],
          size_change_percent: -10,
        },
        pre_transform: {
          size_bytes: 1024,
          token_estimate: 200,
          payload: {
            model: 'gpt-4o',
            messages: [],
            temperature: 0,
            max_tokens: 256,
          },
        },
        post_transform: {
          size_bytes: 900,
          token_estimate: 180,
          payload: {
            model: 'gpt-4o',
            messages: [],
            temperature: 0,
            max_tokens: 256,
          },
        },
      },
    };

    expect(call.transform_changes.length).toBe(1);
    expect(call.status).toBe('success');
  });
});
