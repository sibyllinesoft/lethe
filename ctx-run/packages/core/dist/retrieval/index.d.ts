import { DB } from '@lethe/sqlite';
import type { Embeddings } from '@lethe/embeddings';
export interface Candidate {
    docId: string;
    score: number;
    text?: string;
    kind?: string;
}
export declare function bm25Search(queries: string[], sessionId: string, k: number): Promise<{
    docId: string;
    score: number;
}[]>;
export declare function bm25SearchWithDb(db: DB, queries: string[], sessionId: string, k: number): Promise<{
    docId: string;
    score: number;
}[]>;
export declare function vectorSearch(qVec: Float32Array, k: number): Promise<{
    docId: string;
    score: number;
}[]>;
export declare function vectorSearchWithDb(db: DB, qVec: Float32Array, k: number): Promise<{
    docId: string;
    score: number;
}[]>;
export declare function hybridScore(lexical: {
    docId: string;
    score: number;
}[], vector: {
    docId: string;
    score: number;
}[], config: {
    alpha: number;
    beta: number;
    gamma_kind_boost: {
        [kind: string]: number;
    };
}): Candidate[];
export interface HybridConfig {
    alpha: number;
    beta: number;
    gamma_kind_boost: {
        [kind: string]: number;
    };
    rerank: boolean;
    diversify: boolean;
    k_initial: number;
    k_final: number;
}
export declare const DEFAULT_HYBRID_CONFIG: HybridConfig;
export interface HybridRetrievalOptions {
    db: DB;
    embeddings: Embeddings;
    sessionId: string;
    config?: Partial<HybridConfig>;
}
export declare function hybridRetrieval(queries: string[], options: HybridRetrievalOptions): Promise<Candidate[]>;
//# sourceMappingURL=index.d.ts.map