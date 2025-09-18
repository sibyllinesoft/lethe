import { describe, expect, test } from 'bun:test';
import { analyzeTokens, countTokens, tokenRatio, tokenize } from '@lethe/tokenizer';

describe('Tokenizer', () => {
  test('splits tokens case-insensitively', () => {
    const tokens = tokenize('Latency P95 is 210ms.');
    expect(tokens).toEqual(['latency', 'p95', 'is', '210ms']);
  });

  test('tokenRatio highlights overlap', () => {
    const ratio = tokenRatio('Latency p95 improved to 210ms', 'latency budget metrics');
    expect(ratio).toBeGreaterThan(0);
    expect(ratio).toBeLessThan(1);
  });

  test('analyzeTokens returns counts and ranking', () => {
    const breakdown = analyzeTokens('a a b c c c');
    expect(breakdown.totalTokens).toBe(6);
    expect(breakdown.uniqueTokens).toBe(3);
    expect(breakdown.topTokens[0].token).toBe('c');
  });

  test('countTokens matches tokenize length', () => {
    const text = 'Incident response needed immediately';
    expect(countTokens(text)).toBe(tokenize(text).length);
  });
});
