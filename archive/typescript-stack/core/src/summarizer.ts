import type { SummarizationConfig } from '@lethe/types';
import { analyzeTokens } from '@lethe/tokenizer';

export function summarize(query: string, passages: string[], config: SummarizationConfig): string {
  if (!config.enabled || passages.length === 0) {
    return `Context for “${query}” with ${passages.length} supporting passages.`;
  }

  const combined = passages.join(' ');
  const breakdown = analyzeTokens(combined, 8);

  const topTokens = breakdown.topTokens
    .map(({ token }) => token)
    .filter((token) => token.length > 2)
    .slice(0, 5);

  const trimmed = combined.length > config.maxSummaryTokens * 4
    ? `${combined.slice(0, config.maxSummaryTokens * 4)}...`
    : combined;

  return [
    `Query: ${query}`,
    topTokens.length ? `Key terms: ${topTokens.join(', ')}` : undefined,
    `Preview: ${trimmed}`,
  ]
    .filter(Boolean)
    .join('\n');
}
