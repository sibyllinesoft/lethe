/**
 * Lightweight tokenizer utilities used across the Lethe monorepo.
 * The implementation intentionally avoids heavyweight dependencies – we only
 * rely on a few helpers from Bun's standard library and basic string logic.
 */

import { TokenBreakdown } from '@lethe/types';

const TOKEN_REGEX = /[\p{L}\p{N}_-]+/gu;

export function tokenize(text: string): string[] {
  if (!text) {
    return [];
  }

  return Array.from(text.matchAll(TOKEN_REGEX), (match) => match[0].toLowerCase());
}

export function countTokens(text: string): number {
  return tokenize(text).length;
}

export function tokenRatio(text: string, query: string): number {
  const textTokens = tokenize(text);
  const queryTokens = new Set(tokenize(query));

  if (queryTokens.size === 0 || textTokens.length === 0) {
    return 0;
  }

  let hits = 0;
  for (const token of textTokens) {
    if (queryTokens.has(token)) {
      hits += 1;
    }
  }

  return hits / textTokens.length;
}

export function analyzeTokens(text: string, topN = 5): TokenBreakdown {
  const tokens = tokenize(text);
  const counts = new Map<string, number>();

  for (const token of tokens) {
    counts.set(token, (counts.get(token) ?? 0) + 1);
  }

  const sorted = Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN)
    .map(([token, count]) => ({ token, count }));

  return {
    totalTokens: tokens.length,
    uniqueTokens: counts.size,
    topTokens: sorted,
  };
}

export const tokenizer = {
  tokenize,
  countTokens,
  tokenRatio,
  analyzeTokens,
};

export default tokenizer;
