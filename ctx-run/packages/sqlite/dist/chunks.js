"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getChunksForSession = getChunksForSession;
exports.getChunksWithoutEmbeddings = getChunksWithoutEmbeddings;
function getChunksForSession(db, sessionId) {
    const stmt = db.prepare('SELECT c.text FROM chunks c JOIN messages m ON c.message_id = m.id WHERE m.session_id = ?');
    return stmt.all(sessionId);
}
function getChunksWithoutEmbeddings(db, sessionId) {
    const stmt = db.prepare('SELECT c.* FROM chunks c JOIN messages m ON c.message_id = m.id LEFT JOIN embeddings e ON c.id = e.chunk_id WHERE m.session_id = ? AND e.chunk_id IS NULL');
    return stmt.all(sessionId);
}
//# sourceMappingURL=chunks.js.map