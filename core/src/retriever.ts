import { randomUUID } from 'crypto';
import type {
  ContextPack,
  LetheConfig,
  RetrievalCandidate,
  RetrievalHighlight,
  SessionMessage,
} from '@lethe/types';
import { analyzeTokens, tokenRatio } from '@lethe/tokenizer';
import { summarize } from './summarizer';

interface RetrievalInputs {
  messages: SessionMessage[];
  query: string;
  config: LetheConfig;
}

function lexicalScore(message: SessionMessage, query: string): number {
  return tokenRatio(message.text, query);
}

function semanticScore(message: SessionMessage, query: string): number {
  const { totalTokens, uniqueTokens } = analyzeTokens(message.text);
  if (totalTokens === 0) {
    return 0;
  }
  const queryTokens = analyzeTokens(query);
  const overlap = queryTokens.topTokens
    .filter(({ token }) => message.text.toLowerCase().includes(token))
    .reduce((acc, item) => acc + item.count, 0);

  return Math.min(1, overlap / (uniqueTokens || 1));
}

function diversityScore(candidate: SessionMessage, selected: RetrievalCandidate[]): number {
  if (selected.length === 0) {
    return 1;
  }

  const { topTokens: candidateTokens } = analyzeTokens(candidate.text);
  const existingTokens = new Set<string>();
  for (const item of selected) {
    analyzeTokens(item.message.text).topTokens.forEach(({ token }) => existingTokens.add(token));
  }

  const novel = candidateTokens.filter(({ token }) => !existingTokens.has(token));
  return candidateTokens.length === 0 ? 0 : novel.length / candidateTokens.length;
}

function buildHighlights(message: SessionMessage, query: string): RetrievalHighlight[] {
  const segments: RetrievalHighlight[] = [];
  const lowered = message.text.toLowerCase();
  for (const token of new Set(query.toLowerCase().split(/\s+/g))) {
    if (!token || token.length < 2) continue;
    let index = lowered.indexOf(token);
    while (index !== -1) {
      segments.push({ start: index, end: index + token.length });
      index = lowered.indexOf(token, index + token.length);
    }
  }
  return segments;
}

function buildCandidates({ messages, query, config }: RetrievalInputs): RetrievalCandidate[] {
  const candidates: RetrievalCandidate[] = [];
  for (const message of messages) {
    const lexical = lexicalScore(message, query);
    const semantic = semanticScore(message, query);

    if (lexical < config.retrieval.minRelevance && semantic < config.retrieval.minRelevance) {
      continue;
    }

    candidates.push({
      message,
      lexicalScore: lexical,
      semanticScore: semantic,
      diversityScore: 0,
      hybridScore: 0,
      highlights: buildHighlights(message, query),
    });
  }

  candidates.sort((a, b) => b.lexicalScore + b.semanticScore - (a.lexicalScore + a.semanticScore));

  const selected: RetrievalCandidate[] = [];
  for (const candidate of candidates) {
    const diversity = diversityScore(candidate.message, selected);
    const hybrid =
      candidate.lexicalScore * config.retrieval.weights.lexical +
      candidate.semanticScore * config.retrieval.weights.semantic +
      diversity * config.retrieval.weights.diversity;

    const enriched: RetrievalCandidate = {
      ...candidate,
      diversityScore: diversity,
      hybridScore: hybrid,
    };

    selected.push(enriched);
    if (selected.length >= config.retrieval.topK) {
      break;
    }
  }

  return selected.sort((a, b) => b.hybridScore - a.hybridScore);
}

export function generateContextPack(inputs: RetrievalInputs): ContextPack {
  const { messages, query, config } = inputs;
  const candidates = buildCandidates({ messages, query, config });

  const summary = summarize(
    query,
    candidates.map((candidate) => candidate.message.text),
    config.summarization
  );

  return {
    id: randomUUID(),
    sessionId: messages[0]?.sessionId ?? 'unknown',
    query,
    summary,
    createdAt: Date.now(),
    messages: candidates,
    metadata: {
      generator: 'ContextOrchestrator',
    },
  };
}
