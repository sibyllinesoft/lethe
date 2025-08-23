import type { DB } from '@lethe/sqlite';
import type { Embeddings } from '@lethe/embeddings';
import type { PlanSelection } from '../state/index.js';
export interface EnhancedQueryOptions {
    db: DB;
    embeddings: Embeddings;
    sessionId: string;
    enableHyde?: boolean;
    enableSummarization?: boolean;
    enablePlanSelection?: boolean;
}
export interface EnhancedQueryResult {
    pack: any;
    plan: PlanSelection;
    hydeQueries?: string[];
    duration: {
        total: number;
        hyde?: number;
        retrieval: number;
        summarization?: number;
    };
    debug: {
        originalQuery: string;
        finalQueries: string[];
        retrievalCandidates: number;
        plan: PlanSelection;
    };
}
export declare function enhancedQuery(originalQuery: string, options: EnhancedQueryOptions): Promise<EnhancedQueryResult>;
export declare function diagnoseEnhancedPipeline(db: DB): Promise<{
    ollama: {
        available: boolean;
        models: string[];
    };
    state: {
        sessions: string[];
    };
    config: any;
}>;
//# sourceMappingURL=index.d.ts.map