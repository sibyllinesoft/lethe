import type { DB } from '@lethe/sqlite';
import type { ContextPack } from '../summarize/index.js';
export type PlanType = 'explore' | 'verify' | 'exploit';
export interface SessionState {
    sessionId: string;
    recentEntities: string[];
    lastPackClaims: string[];
    lastPackContradictions: string[];
    lastPackId?: string;
    updatedAt: string;
}
export interface PlanSelection {
    plan: PlanType;
    reasoning: string;
    parameters: {
        hyde_k?: number;
        beta?: number;
        granularity?: 'loose' | 'medium' | 'tight';
        k_final?: number;
    };
}
export declare class StateManager {
    private db;
    constructor(db: DB);
    private ensureStateTable;
    getSessionState(sessionId: string): SessionState;
    updateSessionState(sessionId: string, pack: ContextPack): void;
    selectPlan(sessionId: string, currentQuery: string): PlanSelection;
    getRecentContext(sessionId: string): {
        entityCount: number;
        recentEntities: string[];
        lastPackSummary?: string;
        lastPackContradictions: string[];
    };
    clearSessionState(sessionId: string): void;
    getAllSessions(): string[];
}
export declare function getStateManager(db: DB): StateManager;
//# sourceMappingURL=index.d.ts.map