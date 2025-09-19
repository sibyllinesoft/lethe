import { describe, expect, test } from 'bun:test';
import { ContextOrchestrator, mergeConfig, defaultConfig } from '@lethe/core';
import type { SessionMessage } from '@lethe/types';

const demoSession = 'session-alpha';
const messages: SessionMessage[] = [
  {
    id: 'm-1',
    sessionId: demoSession,
    role: 'user',
    text: 'How did the canary deploy go yesterday? I need latency numbers.',
    timestamp: Date.now() - 1000,
  },
  {
    id: 'm-2',
    sessionId: demoSession,
    role: 'assistant',
    text: 'Latency p95 was 212ms with error rate under 0.2%.',
    timestamp: Date.now() - 900,
  },
  {
    id: 'm-3',
    sessionId: demoSession,
    role: 'assistant',
    text: 'We rolled back the risky feature flag because of missing metrics.',
    timestamp: Date.now() - 800,
  },
];

describe('ContextOrchestrator', () => {
  test('builds a context pack with relevant messages', () => {
    const orchestrator = new ContextOrchestrator();
    orchestrator.getStore().upsertMessages(messages);

    const result = orchestrator.buildContext(demoSession, 'latency metrics');
    expect(result.success).toBe(true);
    if (!result.success) {
      throw new Error('Expected successful context build');
    }

    const pack = result.data;
    expect(pack.messages.length).toBeGreaterThan(0);
    expect(pack.summary).toContain('latency');
    expect(pack.messages[0].hybridScore).toBeGreaterThan(0);
  });

  test('accepts partial configuration overrides', () => {
    const orchestrator = new ContextOrchestrator({
      config: mergeConfig({
        retrieval: {
          ...defaultConfig.retrieval,
          topK: 2,
        },
      }),
    });
    orchestrator.getStore().upsertMessages(messages);

    const result = orchestrator.buildContext(demoSession, 'feature flag');
    expect(result.success).toBe(true);
    if (!result.success) {
      throw new Error('Expected successful context build');
    }
    expect(result.data.messages.length).toBeLessThanOrEqual(2);
  });
});
