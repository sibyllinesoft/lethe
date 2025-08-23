import type { Candidate } from '../retrieval/index.js';
export interface Reranker {
    name: string;
    rerank(query: string, candidates: Candidate[]): Promise<Candidate[]>;
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
export declare class NoOpReranker implements Reranker {
    name: string;
    rerank(query: string, candidates: Candidate[]): Promise<Candidate[]>;
}
export declare function getReranker(enabled?: boolean): Promise<Reranker>;
//# sourceMappingURL=index.d.ts.map