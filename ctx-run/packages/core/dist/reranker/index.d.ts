import type { Candidate } from '../retrieval/index.js';
import type { DB } from '@lethe/sqlite';
export interface Reranker {
    name: string;
    rerank(query: string, candidates: Candidate[]): Promise<Candidate[]>;
}
export interface RerankerConfig {
    use_llm: boolean;
    llm_budget_ms: number;
    llm_model: string;
    contradiction_enabled: boolean;
    contradiction_penalty: number;
}
export declare class CrossEncoderReranker implements Reranker {
    name: string;
    private model;
    constructor(modelId?: string);
    init(): Promise<void>;
    rerank(query: string, candidates: Candidate[]): Promise<Candidate[]>;
    private fallbackRerank;
    private tokenize;
}
export declare class LLMReranker implements Reranker {
    name: string;
    private db;
    private config;
    constructor(db: DB, config: RerankerConfig);
    rerank(query: string, candidates: Candidate[]): Promise<Candidate[]>;
    private llmRerankWithTimeout;
    private applyLLMScores;
    private applyContradictionPenalties;
    private checkContradiction;
}
export declare class NoOpReranker implements Reranker {
    name: string;
    rerank(query: string, candidates: Candidate[]): Promise<Candidate[]>;
}
export declare function getReranker(enabled?: boolean, config?: RerankerConfig, db?: DB): Promise<Reranker>;
export declare const DEFAULT_LLM_RERANKER_CONFIG: RerankerConfig;
//# sourceMappingURL=index.d.ts.map