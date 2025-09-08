import { DB, Message } from './types';
import { Chunk } from './chunk-types';

export function upsertMessages(db: DB, messages: Message[]): void {
    const stmt = db.prepare(
        'INSERT OR REPLACE INTO messages (id, session_id, turn, role, text, ts, meta) VALUES (?, ?, ?, ?, ?, ?, ?)'
    );
    db.transaction((messages: Message[]) => {
        for (const msg of messages) {
            stmt.run(msg.id, msg.session_id, msg.turn, msg.role, msg.text, msg.ts, msg.meta ? JSON.stringify(msg.meta) : null);
        }
    })(messages);
}

export function upsertChunks(db: DB, chunks: Chunk[]): void {
    const stmt = db.prepare(
        'INSERT OR REPLACE INTO chunks (id, message_id, offset_start, offset_end, kind, text, tokens) VALUES (?, ?, ?, ?, ?, ?, ?)'
    );
    db.transaction((chunks: Chunk[]) => {
        for (const chunk of chunks) {
            stmt.run(chunk.id, chunk.message_id, chunk.offset_start, chunk.offset_end, chunk.kind, chunk.text, chunk.tokens);
        }
    })(chunks);
}