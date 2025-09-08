"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.StateManager = void 0;
exports.getStateManager = getStateManager;
// State management using database storage
const STATE_TABLE = 'session_states';
class StateManager {
    db;
    constructor(db) {
        this.db = db;
        this.ensureStateTable();
    }
    ensureStateTable() {
        try {
            this.db.exec(`
        CREATE TABLE IF NOT EXISTS ${STATE_TABLE} (
          session_id TEXT PRIMARY KEY,
          recent_entities TEXT NOT NULL DEFAULT '[]',
          last_pack_claims TEXT NOT NULL DEFAULT '[]',
          last_pack_contradictions TEXT NOT NULL DEFAULT '[]',
          last_pack_id TEXT,
          updated_at TEXT NOT NULL
        )
      `);
        }
        catch (error) {
            console.warn(`Could not create state table: ${error}`);
        }
    }
    getSessionState(sessionId) {
        try {
            const stmt = this.db.prepare(`
        SELECT * FROM ${STATE_TABLE} WHERE session_id = ?
      `);
            const row = stmt.get(sessionId);
            if (row) {
                return {
                    sessionId,
                    recentEntities: JSON.parse(row.recent_entities || '[]'),
                    lastPackClaims: JSON.parse(row.last_pack_claims || '[]'),
                    lastPackContradictions: JSON.parse(row.last_pack_contradictions || '[]'),
                    lastPackId: row.last_pack_id,
                    updatedAt: row.updated_at
                };
            }
        }
        catch (error) {
            console.warn(`Could not retrieve session state: ${error}`);
        }
        // Return default state
        return {
            sessionId,
            recentEntities: [],
            lastPackClaims: [],
            lastPackContradictions: [],
            updatedAt: new Date().toISOString()
        };
    }
    updateSessionState(sessionId, pack) {
        try {
            const currentState = this.getSessionState(sessionId);
            // Update recent entities (keep last 200)
            const newEntities = [...currentState.recentEntities, ...pack.key_entities];
            const uniqueEntities = Array.from(new Set(newEntities));
            const recentEntities = uniqueEntities.slice(-200);
            const stmt = this.db.prepare(`
        INSERT OR REPLACE INTO ${STATE_TABLE} 
        (session_id, recent_entities, last_pack_claims, last_pack_contradictions, last_pack_id, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
      `);
            stmt.run(sessionId, JSON.stringify(recentEntities), JSON.stringify(pack.claims), JSON.stringify(pack.contradictions), pack.id, new Date().toISOString());
            console.debug(`Updated state for session ${sessionId}: ${recentEntities.length} entities, ${pack.claims.length} claims, ${pack.contradictions.length} contradictions`);
        }
        catch (error) {
            console.warn(`Could not update session state: ${error}`);
        }
    }
    selectPlan(sessionId, currentQuery) {
        const state = this.getSessionState(sessionId);
        // Extract entities from current query for overlap analysis
        const queryEntities = extractEntitiesFromText(currentQuery);
        // Plan selection logic
        // 1. VERIFY: If last pack had contradictions
        if (state.lastPackContradictions.length > 0) {
            return {
                plan: 'verify',
                reasoning: `Last pack had ${state.lastPackContradictions.length} contradictions, need verification`,
                parameters: {
                    hyde_k: 5,
                    beta: 0.4, // Slightly favor lexical for verification
                    granularity: 'tight',
                    k_final: 8
                }
            };
        }
        // 2. EXPLORE: If low entity overlap with recent entities
        const entityOverlap = calculateEntityOverlap(queryEntities, state.recentEntities);
        if (entityOverlap < 0.3) {
            return {
                plan: 'explore',
                reasoning: `Low entity overlap (${(entityOverlap * 100).toFixed(1)}%) suggests new topic exploration`,
                parameters: {
                    hyde_k: 4,
                    beta: 0.6, // Favor semantic search for exploration
                    granularity: 'loose',
                    k_final: 12
                }
            };
        }
        // 3. EXPLOIT: Default case - refining known topics
        return {
            plan: 'exploit',
            reasoning: `High entity overlap (${(entityOverlap * 100).toFixed(1)}%) suggests refining known topics`,
            parameters: {
                hyde_k: 3,
                beta: 0.5, // Balanced approach
                granularity: 'medium',
                k_final: 10
            }
        };
    }
    // Get recent context for debugging
    getRecentContext(sessionId) {
        const state = this.getSessionState(sessionId);
        return {
            entityCount: state.recentEntities.length,
            recentEntities: state.recentEntities.slice(-20), // Last 20 for display
            lastPackContradictions: state.lastPackContradictions
        };
    }
    // Clear state for session (useful for testing)
    clearSessionState(sessionId) {
        try {
            const stmt = this.db.prepare(`DELETE FROM ${STATE_TABLE} WHERE session_id = ?`);
            stmt.run(sessionId);
            console.debug(`Cleared state for session ${sessionId}`);
        }
        catch (error) {
            console.warn(`Could not clear session state: ${error}`);
        }
    }
    // Get all sessions with state
    getAllSessions() {
        try {
            const stmt = this.db.prepare(`SELECT session_id FROM ${STATE_TABLE} ORDER BY updated_at DESC`);
            const rows = stmt.all();
            return rows.map(row => row.session_id);
        }
        catch (error) {
            console.warn(`Could not get all sessions: ${error}`);
            return [];
        }
    }
}
exports.StateManager = StateManager;
// Extract entities from text (simple heuristic)
function extractEntitiesFromText(text) {
    const words = text.toLowerCase()
        .replace(/[^\w\s]/g, ' ')
        .split(/\s+/)
        .filter(word => word.length >= 3 &&
        !isStopWord(word) &&
        !isCommonProgrammingWord(word));
    // Count frequency and take the more frequent terms
    const freq = {};
    words.forEach(word => {
        freq[word] = (freq[word] || 0) + 1;
    });
    return Object.keys(freq)
        .sort((a, b) => freq[b] - freq[a])
        .slice(0, 10);
}
// Calculate overlap between two entity sets
function calculateEntityOverlap(entities1, entities2) {
    if (entities1.length === 0 || entities2.length === 0) {
        return 0;
    }
    const set1 = new Set(entities1.map(e => e.toLowerCase()));
    const set2 = new Set(entities2.map(e => e.toLowerCase()));
    const intersection = new Set([...set1].filter(x => set2.has(x)));
    const union = new Set([...set1, ...set2]);
    return intersection.size / union.size; // Jaccard similarity
}
function isStopWord(word) {
    const stopWords = new Set([
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
        'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'among', 'under', 'over', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    ]);
    return stopWords.has(word);
}
function isCommonProgrammingWord(word) {
    const commonWords = new Set([
        'function', 'var', 'let', 'const', 'if', 'else', 'for', 'while', 'return',
        'true', 'false', 'null', 'undefined', 'new', 'this', 'class', 'extends',
        'import', 'export', 'from', 'as', 'default', 'try', 'catch', 'throw',
        'async', 'await', 'then', 'error', 'data', 'result', 'value'
    ]);
    return commonWords.has(word);
}
// Factory function
function getStateManager(db) {
    return new StateManager(db);
}
//# sourceMappingURL=index.js.map