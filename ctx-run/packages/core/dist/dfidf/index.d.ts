import { DB } from '@lethe/sqlite';
export declare function rebuild(sessionId: string): Promise<void>;
export declare function rebuildForDb(db: DB, sessionId: string): Promise<void>;
export declare function topRare(sessionId: string, n: number): Promise<string[]>;
export declare function topRareForDb(db: DB, sessionId: string, n: number): Promise<string[]>;
export declare function topHead(sessionId: string, n: number): Promise<string[]>;
export declare function topHeadForDb(db: DB, sessionId: string, n: number): Promise<string[]>;
//# sourceMappingURL=index.d.ts.map