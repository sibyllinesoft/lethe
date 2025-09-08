import type { DB } from '@lethe/sqlite';
import type { Embeddings } from '@lethe/embeddings';
import type { PlanSelection } from '../state/index.js';
import { type ConversationTurn, type QueryUnderstandingConfig } from '../query-understanding/index.js';
export interface EnhancedQueryOptions {
    db: DB;
    embeddings: Embeddings;
    sessionId: string;
    enableHyde?: boolean;
    enableSummarization?: boolean;
    enablePlanSelection?: boolean;
    enableQueryUnderstanding?: boolean;
    queryUnderstandingConfig?: Partial<QueryUnderstandingConfig>;
    recentTurns?: ConversationTurn[];
    enableMLPrediction?: boolean;
    mlConfig?: {
        fusion_dynamic?: boolean;
        plan_learned?: boolean;
    };
    llmRerankConfig?: {
        use_llm?: boolean;
        llm_budget_ms?: number;
        llm_model?: string;
        contradiction_enabled?: boolean;
        contradiction_penalty?: number;
    };
}
export interface EnhancedQueryResult {
    pack: any;
    plan: PlanSelection;
    hydeQueries?: string[];
    queryUnderstanding?: {
        canonical_query?: string;
        subqueries?: string[];
        rewrite_success: boolean;
        decompose_success: boolean;
        llm_calls_made: number;
        errors: string[];
    };
    mlPrediction?: {
        alpha?: number;
        beta?: number;
        predicted_plan?: string;
        prediction_time_ms?: number;
        model_loaded?: boolean;
    };
    duration: {
        total: number;
        queryUnderstanding?: number;
        hyde?: number;
        retrieval: number;
        summarization?: number;
        mlPrediction?: number;
    };
    debug: {
        originalQuery: string;
        finalQueries: string[];
        retrievalCandidates: number;
        plan: PlanSelection;
        queryProcessingEnabled?: boolean;
        rewriteFailureRate?: number;
        decomposeFailureRate?: number;
        mlPredictionEnabled?: boolean;
        staticAlpha?: number;
        staticBeta?: number;
        predictedAlpha?: number;
        predictedBeta?: number;
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