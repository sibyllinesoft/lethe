import type { LetheConfig } from '@lethe/types';

export const defaultConfig: LetheConfig = {
  retrieval: {
    topK: 8,
    weights: {
      lexical: 0.5,
      semantic: 0.4,
      diversity: 0.1,
    },
    minRelevance: 0.05,
  },
  chunking: {
    maxTokens: 120,
    overlap: 15,
    splitCodeBlocks: true,
    splitSentences: true,
  },
  summarization: {
    enabled: true,
    maxSummaryTokens: 80,
  },
};

export function mergeConfig(partial?: Partial<LetheConfig>): LetheConfig {
  if (!partial) {
    return structuredClone(defaultConfig);
  }

  return {
    retrieval: {
      ...defaultConfig.retrieval,
      ...partial.retrieval,
      weights: {
        ...defaultConfig.retrieval.weights,
        ...partial.retrieval?.weights,
      },
    },
    chunking: {
      ...defaultConfig.chunking,
      ...partial.chunking,
    },
    summarization: {
      ...defaultConfig.summarization,
      ...partial.summarization,
    },
  };
}
