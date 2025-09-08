"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.rebuild = rebuild;
exports.rebuildForDb = rebuildForDb;
exports.topRare = topRare;
exports.topRareForDb = topRareForDb;
exports.topHead = topHead;
exports.topHeadForDb = topHeadForDb;
const sqlite_1 = require("@lethe/sqlite");
function calculateIdf(N, df) {
    // IDF as specified: log((N - df + 0.5) / (df + 0.5)) bounded below at 0
    const idf = Math.log((N - df + 0.5) / (df + 0.5));
    return Math.max(0, idf);
}
// Tokenize text into terms (simple whitespace + punctuation split)
function tokenize(text) {
    return text
        .toLowerCase()
        .replace(/[^\w\s]/g, ' ')
        .split(/\s+/)
        .filter(term => term.length > 1); // Filter out single characters
}
async function rebuild(sessionId) {
    // This function signature matches the spec exactly
    // Implementation will be injected with DB via closure or dependency injection
    throw new Error('rebuild needs DB instance - will be fixed in CLI integration');
}
async function rebuildForDb(db, sessionId) {
    const chunks = (0, sqlite_1.getChunksBySession)(db, sessionId);
    const N = chunks.length;
    if (N === 0)
        return; // No chunks to process
    const termDocFreq = {};
    // Count document frequency for each term
    for (const chunk of chunks) {
        const terms = new Set(tokenize(chunk.text));
        for (const term of terms) {
            termDocFreq[term] = (termDocFreq[term] || 0) + 1;
        }
    }
    // Clear existing DF/IDF for this session
    db.prepare('DELETE FROM dfidf WHERE session_id = ?').run(sessionId);
    // Insert new DF/IDF values
    const stmt = db.prepare('INSERT INTO dfidf (term, session_id, df, idf) VALUES (?, ?, ?, ?)');
    const transaction = db.transaction(() => {
        for (const [term, df] of Object.entries(termDocFreq)) {
            const idf = calculateIdf(N, df);
            stmt.run(term, sessionId, df, idf);
        }
    });
    transaction();
}
async function topRare(sessionId, n) {
    // This function signature matches the spec exactly
    throw new Error('topRare needs DB instance - will be fixed in CLI integration');
}
async function topRareForDb(db, sessionId, n) {
    const stmt = db.prepare('SELECT term FROM dfidf WHERE session_id = ? ORDER BY idf DESC LIMIT ?');
    const results = stmt.all(sessionId, n);
    return results.map(r => r.term);
}
async function topHead(sessionId, n) {
    // This function signature matches the spec exactly
    throw new Error('topHead needs DB instance - will be fixed in CLI integration');
}
async function topHeadForDb(db, sessionId, n) {
    const stmt = db.prepare('SELECT term FROM dfidf WHERE session_id = ? ORDER BY df DESC LIMIT ?');
    const results = stmt.all(sessionId, n);
    return results.map(r => r.term);
}
//# sourceMappingURL=index.js.map