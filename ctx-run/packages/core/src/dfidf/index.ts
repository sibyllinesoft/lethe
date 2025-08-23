import { DB, getChunksBySession, getDFIdf } from '@lethe/sqlite';

function calculateIdf(N: number, df: number): number {
    // IDF as specified: log((N - df + 0.5) / (df + 0.5)) bounded below at 0
    const idf = Math.log((N - df + 0.5) / (df + 0.5));
    return Math.max(0, idf);
}

// Tokenize text into terms (simple whitespace + punctuation split)
function tokenize(text: string): string[] {
    return text
        .toLowerCase()
        .replace(/[^\w\s]/g, ' ')
        .split(/\s+/)
        .filter(term => term.length > 1); // Filter out single characters
}

export async function rebuild(sessionId: string): Promise<void> {
    // This function signature matches the spec exactly
    // Implementation will be injected with DB via closure or dependency injection
    throw new Error('rebuild needs DB instance - will be fixed in CLI integration');
}

export async function rebuildForDb(db: DB, sessionId: string): Promise<void> {
    const chunks = getChunksBySession(db, sessionId);
    const N = chunks.length;
    
    if (N === 0) return; // No chunks to process
    
    const termDocFreq: { [term: string]: number } = {};

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

export async function topRare(sessionId: string, n: number): Promise<string[]> {
    // This function signature matches the spec exactly
    throw new Error('topRare needs DB instance - will be fixed in CLI integration');
}

export async function topRareForDb(db: DB, sessionId: string, n: number): Promise<string[]> {
    const stmt = db.prepare('SELECT term FROM dfidf WHERE session_id = ? ORDER BY idf DESC LIMIT ?');
    const results = stmt.all(sessionId, n) as { term: string }[];
    return results.map(r => r.term);
}

export async function topHead(sessionId: string, n: number): Promise<string[]> {
    // This function signature matches the spec exactly
    throw new Error('topHead needs DB instance - will be fixed in CLI integration');
}

export async function topHeadForDb(db: DB, sessionId: string, n: number): Promise<string[]> {
    const stmt = db.prepare('SELECT term FROM dfidf WHERE session_id = ? ORDER BY df DESC LIMIT ?');
    const results = stmt.all(sessionId, n) as { term: string }[];
    return results.map(r => r.term);
}