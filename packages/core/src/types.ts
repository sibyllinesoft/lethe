export interface Config {
  models: {
    embed: string;
    rerank: string;
    hyde?: string;
    summarize?: string;
  };
  retrieval: {
    alpha: number;          // BM25 weight
    beta: number;           // Vector weight
    gamma_kind_boost: {     // Kind-specific score boosts
      tool_result: number;
      user_code: number;
    };
  };
  chunking: {
    target_tokens: number;
    overlap: number;
    split_code_blocks: boolean;
    split_sentences: boolean;
  };
  rerank: {
    topk_in: number;
    topk_out: number;
    batch_size: number;
  };
  diversify: {
    pack_chunks: number;
  };
  plan: {
    explore: { hyde_k: number; granularity: string; beta: number };
    verify: { hyde_k: number; granularity: string; beta: number };
    exploit: { hyde_k: number; granularity: string; beta: number };
  };
}

export interface Candidate {
  id: string;
  text: string;
  messageId: string;
  kind: 'prose' | 'code' | 'tool_result' | 'user_code';
  bm25Score: number;
  vectorScore: number;
  hybridScore: number;
  rerankScore?: number;
}

export interface HydeResult {
  queries: string[];
  pseudo: string;
}

export interface ContextPack {
  id: string;
  sessionId: string;
  query: string;
  summary: string;
  keyEntities: string[];
  claims: Array<{ text: string; chunks: string[] }>;
  contradictions: Array<{ issue: string; chunks: string[] }>;
  citations: Record<string, { messageId: string; span: [number, number] }>;
  debug?: {
    hydeQueries: string[];
    candidateCount: number;
    rerankTime: number;
    totalTime: number;
  };
}

export type PlanType = 'explore' | 'verify' | 'exploit';