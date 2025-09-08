import Database from 'better-sqlite3';
export type DB = Database.Database;
export * from './atoms-types.js';
export * from './adaptive-planning.js';
export * from './adaptive-atoms-db.js';
export * from './grid-search.js';
export * from './atoms-db.js';
export * from './entity-extraction.js';
export * from './session-idf.js';
export interface Message {
    id: string;
    session_id: string;
    turn: number;
    role: string;
    text: string;
    ts: number;
    meta?: any;
}
export interface Chunk {
    id: string;
    message_id: string;
    session_id: string;
    offset_start: number;
    offset_end: number;
    kind: string;
    text: string;
    tokens: number;
}
export interface DFIdf {
    term: string;
    session_id: string;
    df: number;
    idf: number;
}
export interface Pack {
    id: string;
    session_id: string;
    created_at: number;
    json: any;
}
export interface State {
    session_id: string;
    json: any;
}
export interface Config {
    key: string;
    value: any;
}
export declare function openDb(path: string): DB;
export declare function migrate(db: DB): Promise<void>;
export declare function loadVectorExtension(db: DB): Promise<boolean>;
export declare function upsertConfig(db: DB, key: string, value: any): void;
export declare function getConfig(db: DB, key: string): any;
export declare function getState(db: DB, sessionId: string): any;
export declare function setState(db: DB, sessionId: string, json: any): void;
export declare function insertMessages(db: DB, messages: Message[]): void;
export declare function insertChunks(db: DB, chunks: Chunk[]): void;
export declare function insertEmbeddings(db: DB, rows: {
    chunk_id: string;
    dim: number;
    vec: Buffer;
}[]): void;
export declare function vectorSearch(db: DB, qVec: Float32Array, k: number): Promise<{
    docId: string;
    score: number;
}[]>;
export declare function refreshVectorIndex(): void;
export declare function getChunksBySession(db: DB, sessionId: string): Chunk[];
export declare function getDFIdf(db: DB, sessionId: string): DFIdf[];
export declare function getChunkById(db: DB, chunkId: string): Chunk | null;
//# sourceMappingURL=index.d.ts.map