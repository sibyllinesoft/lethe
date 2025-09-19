import { describe, expect, test } from 'bun:test';
import { defaultConfig, mergeConfig, summarize } from '@lethe/core';

const PASSAGES = [
  'Deployment succeeded and metrics look healthy.',
  'Error budget remaining is 98% with no regression detected.',
];

describe('Core utilities', () => {
  test('mergeConfig overrides nested values without mutation', () => {
    const merged = mergeConfig({
      retrieval: {
        ...defaultConfig.retrieval,
        topK: 3,
      },
    });

    expect(merged.retrieval.topK).toBe(3);
    expect(defaultConfig.retrieval.topK).not.toBe(3);
  });

  test('summarize produces deterministic preview', () => {
    const summary = summarize('deployment status', PASSAGES, defaultConfig.summarization);
    expect(summary).toContain('deployment status');
    expect(summary).toContain('Key terms');
    expect(summary.length).toBeGreaterThan(20);
  });
});
