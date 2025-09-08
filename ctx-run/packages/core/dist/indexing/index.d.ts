import { DB, Message } from '@lethe/sqlite';
export declare function upsertMessages(sessionId: string, messages: Message[]): Promise<void>;
export declare function upsertMessagesWithDb(db: DB, sessionId: string, messages: Message[]): Promise<void>;
export declare function ensureEmbeddings(sessionId: string): Promise<void>;
export declare function ensureEmbeddingsWithDb(db: DB, sessionId: string): Promise<void>;
export declare function ensureVectorIndex(): Promise<void>;
//# sourceMappingURL=index.d.ts.map