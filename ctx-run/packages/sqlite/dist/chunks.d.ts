import { DB } from './types';
import { Chunk } from './chunk-types';
export declare function getChunksForSession(db: DB, sessionId: string): {
    text: string;
}[];
export declare function getChunksWithoutEmbeddings(db: DB, sessionId: string): Chunk[];
//# sourceMappingURL=chunks.d.ts.map