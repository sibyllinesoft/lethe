import type { DB } from '@lethe/sqlite';
import type { Candidate } from '../retrieval/index.js';
export interface SummarizeResult {
    summary: string;
    key_entities: string[];
    claims: string[];
    contradictions: string[];
    citations: {
        id: number;
        chunk_id: string;
        relevance: number;
    }[];
}
export interface ContextPack {
    id: string;
    session_id: string;
    query: string;
    created_at: string;
    summary: string;
    key_entities: string[];
    claims: string[];
    contradictions: string[];
    chunks: Array<{
        id: string;
        score: number;
        kind: string;
        text: string;
    }>;
    citations: Array<{
        id: number;
        chunk_id: string;
        relevance: number;
    }>;
}
export type Granularity = 'loose' | 'medium' | 'tight';
export interface SummarizeConfig {
    model: string;
    temperature: number;
    maxTokens: number;
    timeoutMs: number;
    enabled: boolean;
    granularity: Granularity;
}
export declare const DEFAULT_SUMMARIZE_CONFIG: SummarizeConfig;
export declare function summarizeChunks(db: DB, query: string, chunks: Candidate[], config?: Partial<SummarizeConfig>): Promise<SummarizeResult>;
export declare function buildContextPack(db: DB, sessionId: string, query: string, chunks: Candidate[], config?: Partial<SummarizeConfig>): Promise<ContextPack>;
export declare function testSummarization(db: DB, query?: string): Promise<{
    success: boolean;
    result?: SummarizeResult;
    duration?: number;
    error?: string;
}>;
//# sourceMappingURL=index.d.ts.map