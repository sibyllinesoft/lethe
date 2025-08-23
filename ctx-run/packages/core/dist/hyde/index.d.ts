import type { DB } from '@lethe/sqlite';
export interface HydeResult {
    queries: string[];
    pseudo: string;
}
export interface HydeConfig {
    model: string;
    temperature: number;
    numQueries: number;
    maxTokens: number;
    timeoutMs: number;
    enabled: boolean;
}
export declare const DEFAULT_HYDE_CONFIG: HydeConfig;
export declare function generateHyde(db: DB, originalQuery: string, config?: Partial<HydeConfig>): Promise<HydeResult>;
export declare function testHyde(db: DB, query?: string): Promise<{
    success: boolean;
    result?: HydeResult;
    duration?: number;
    error?: string;
}>;
//# sourceMappingURL=index.d.ts.map