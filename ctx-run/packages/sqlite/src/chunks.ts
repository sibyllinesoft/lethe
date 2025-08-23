import { DB } from './types';
import { Chunk } from './chunk-types';

export function getChunksForSession(db: DB, sessionId: string): {text: string}[] {
    const stmt = db.prepare('SELECT c.text FROM chunks c JOIN messages m ON c.message_id = m.id WHERE m.session_id = ?');
    return stmt.all(sessionId) as {text: string}[];
}

export function getChunksWithoutEmbeddings(db: DB, sessionId: string): Chunk[] {
    const stmt = db.prepare(
        'SELECT c.* FROM chunks c JOIN messages m ON c.message_id = m.id LEFT JOIN embeddings e ON c.id = e.chunk_id WHERE m.session_id = ? AND e.chunk_id IS NULL'
    );
    return stmt.all(sessionId) as Chunk[];
}
